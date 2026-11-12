"""Multi-token (speculative verify) Qwen3.5 Gated DeltaNet step for one TP rank (NKI, LNC=1).

Same fused block as ``gdn_decode.gdn_decode_tkg`` (in_proj -> conv + SiLU -> l2norm / gates ->
gated delta rule -> gated RMSNorm -> out_proj slice), but every sequence brings ``S = k + 1``
new tokens (the last accepted token + k draft tokens) that run through the conv and the
recurrence sequentially, in one kernel call. Reference: ``gdn_decode_torch.gdn_verify_torch``.

State handling for speculative decoding:
  * the kernel writes a SNAPSHOT of the conv history and of the recurrent state after every
    one of the S steps: ``conv_out[b, s]`` / ``rec_out[b, s]`` = state after token s;
  * it reads its starting state from ``conv_in[b, src_slot[b]]`` / ``rec_in[b, src_slot[b]]``
    (indirect DMA), so the NEXT round passes this round's snapshots plus ``src_slot = n_b - 1``
    (n_b = tokens of sequence b accepted this round) and rollback costs nothing extra: the
    rejected snapshots are simply never read. ``conv_in`` / ``rec_in`` may have any slot count
    (1 for the first round after prefill). In and out buffers ping-pong between rounds.

Token order: ``x`` / ``out`` rows are SEQUENCE-major (row ``b*S + s``: what the TP AllGather of
per-rank ``[S, H]`` shards yields). Internally tokens are STEP-major (column ``s*B + b``) so that
step s of every sequence is one contiguous ``[*, B]`` slice for the conv and the recurrence; the
reorder happens in the x load / out store DMAs (strided HBM rows).

Layouts (as gdn_decode, plus the slot axis):
  w_proj_p  [128, H/128, 2Kd+2Vd+2vh]   rows pre-permuted (w_p[p, h1, c] = W[h1*128+p, c])
  conv_in   [B, NS, Cc, K-1]            raw history per slot, oldest first
  rec_in    [B, NS, 128k, vh, 128v]     fp32
  src_slot  [B, 1] int32                slot of conv_in / rec_in to start from
  conv_out  [B, S, Cc, K-1]             snapshot after step s
  rec_out   [B, S, 128k, vh, 128v]
"""

import nki
import nki.isa as nisa
import nki.language as nl

from nkilib.core.utils.kernel_assert import kernel_assert

P = 128
HEAD = 128
MCH = 512  # fp32 PSUM bank width
SLAB = 8


def _col_sum_bcast(dst, src, n, ones_col, ones_row):
    """dst[P, n] (SBUF fp32) = column sums of src[P, n], broadcast to every partition."""
    for c0 in range(0, n, MCH):
        cn = min(MCH, n - c0)
        ps_sum = nl.ndarray((1, MCH), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(ps_sum[:, 0:cn], ones_col, src[:, c0 : c0 + cn])
        row = nl.ndarray((1, MCH), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=row[:, 0:cn], src=ps_sum[:, 0:cn])
        ps_bc = nl.ndarray((P, MCH), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(ps_bc[:, 0:cn], ones_row, row[:, 0:cn])
        nisa.tensor_copy(dst=dst[:, c0 : c0 + cn], src=ps_bc[:, 0:cn])


@nki.jit
def gdn_verify_tkg(
    x,  # [B*S, H] bf16, sequence-major rows
    w_proj_p,  # [128, H/128, 2*Kd + 2*Vd + 2*vh] bf16
    conv_w,  # [Cc, K] bf16
    conv_in,  # [B, NS, Cc, K-1] bf16
    a_log,  # [vh] fp32
    dt_bias,  # [vh] fp32
    norm_w,  # [128] fp32
    rec_in,  # [B, NS, 128, vh, 128] fp32
    src_slot,  # [B, 1] int32
    w_out,  # [Vd, H] bf16
):
    B = src_slot.shape[0]
    T, H = x.shape
    S = T // B
    NS = rec_in.shape[1]
    vh = a_log.shape[0]
    Vd = vh * HEAD
    NT = w_proj_p.shape[2]
    NW = NT - 2 * vh
    Kd = (NW - 2 * Vd) // 2
    kh = Kd // HEAD
    gqa = vh // kh
    Cc = 2 * Kd + Vd
    K = conv_w.shape[1]
    Kh = K - 1
    H1 = H // P
    n_ct = NW // P
    n_conv_t = Cc // P
    q_t0, k_t0, v_t0, z_t0 = 0, kh, 2 * kh, 2 * kh + vh
    G = min(8, B)
    eps = 1e-6
    kernel_assert(T == B * S and S >= 1, "x rows must be B * S")
    kernel_assert(B <= P and B % G == 0, f"B must be <= 128 and a multiple of {G}, got {B}")
    kernel_assert(T <= MCH, f"B * S must be <= {MCH}, got {T}")
    kernel_assert(H % (P * SLAB) == 0 and Kd % P == 0, "H must be a multiple of 1024, key dim of 128")
    kernel_assert(rec_in.shape == (B, NS, HEAD, vh, HEAD), "rec_in must be [B, NS, 128, vh, 128]")
    kernel_assert(conv_in.shape == (B, NS, Cc, Kh), "conv_in must be [B, NS, Cc, K-1]")
    kernel_assert(w_proj_p.shape[0] == P and w_proj_p.shape[1] == H1, "w_proj_p must be [128, H/128, cols]")
    kernel_assert(vh * G * 2 <= MCH, "vh * G too large for one PSUM bank")

    N = w_out.shape[1]  # out-projection width: H, or H + folded router columns (low-C verify)
    out = nl.ndarray((T, N), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    conv_out = nl.ndarray((B, S, Cc, Kh), dtype=conv_in.dtype, buffer=nl.shared_hbm)
    rec_out = nl.ndarray((B, S, HEAD, vh, HEAD), dtype=nl.float32, buffer=nl.shared_hbm)

    # Row tiles of whole steps: tile = steps [s_a, s_b), internal rows [s_a*B, s_b*B) (<= 128 rows).
    spt = max(1, P // B)
    tiles = [(s_a, min(S, s_a + spt)) for s_a in range(0, S, spt)]
    NR = len(tiles)

    # ---------------- constants ----------------
    ones_col = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(ones_col, 1.0)
    ones_row = nl.ndarray((1, P), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(ones_row, 1.0)
    iota_g = nl.ndarray((G, G), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(iota_g, pattern=[[1, G]], offset=0, channel_multiplier=-1)
    ident = nl.ndarray((G, G), dtype=nl.float32, buffer=nl.sbuf)  # ident[p, b] = (p == b)
    nisa.tensor_scalar(dst=ident, data=iota_g, op0=nl.equal, operand0=0)
    iota_h = nl.ndarray((vh, vh, P), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(iota_h, pattern=[[1, vh], [0, P]], offset=0, channel_multiplier=-1)
    sel_h = nl.ndarray((vh, vh, P), dtype=nl.float32, buffer=nl.sbuf)  # sel_h[p, h, m] = (p == h)
    nisa.tensor_scalar(dst=sel_h, data=iota_h, op0=nl.equal, operand0=0)

    # ---------------- start-slot offsets (indirect state reads) ----------------
    # conv: one row per sequence of conv_in viewed as [B*NS, Cc*Kh]: row b*NS + slot[b]
    slot_t = nl.ndarray((P, 1), dtype=nl.int32, buffer=nl.sbuf)[:B]
    nisa.dma_copy(dst=slot_t, src=src_slot)
    conv_row = nl.ndarray((P, 1), dtype=nl.int32, buffer=nl.sbuf)[:B]
    nisa.iota(conv_row, pattern=[[0, 1]], offset=0, channel_multiplier=NS)
    nisa.tensor_tensor(dst=conv_row, data1=conv_row, data2=slot_t, op=nl.add)
    # rec: rec_in viewed as [B*NS*128, vh*128]; partition p (= k index) of sequence b reads row
    # (b*NS + slot[b])*128 + p
    slot_bc = nl.ndarray((P, B), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=slot_bc, src=src_slot.ap(pattern=[[0, P], [1, B]], offset=0))
    kernel_assert(B * NS * P < (1 << 24), "state row offsets must stay below 2^24")
    rec_row = nl.ndarray((P, B), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(rec_row, pattern=[[NS * P, B]], offset=0, channel_multiplier=1)
    nisa.scalar_tensor_tensor(
        dst=rec_row, data=slot_bc, op0=nl.multiply, operand0=float(P), op1=nl.add, operand1=rec_row
    )

    # ---------------- x^T [128, H1, T] in step-major token order ----------------
    xT = nl.ndarray((P, H1, T), dtype=nl.bfloat16, buffer=nl.sbuf)
    for s_a, s_b in tiles:
        rows = (s_b - s_a) * B
        x_sb = nl.ndarray((P, H), dtype=nl.bfloat16, buffer=nl.sbuf)[:rows]
        for s in range(s_a, s_b):
            nisa.dma_copy(
                dst=x_sb[(s - s_a) * B : (s - s_a + 1) * B], src=x.ap(pattern=[[S * H, B], [1, H]], offset=s * H)
            )
        for h1 in range(H1):
            tp = nl.ndarray((P, rows), dtype=nl.bfloat16, buffer=nl.psum)
            nisa.nc_transpose(dst=tp, data=x_sb[:, h1 * P : (h1 + 1) * P])
            nisa.tensor_copy(dst=xT[:, h1, s_a * B : s_b * B], src=tp)

    # ---------------- projections (tokens stationary, weights streamed once) ----------------
    # Column groups keep (row tiles x 512-column chunks) within 4 PSUM banks; every weight byte is
    # loaded exactly once whatever the number of row tiles.
    n_mch = (NT + MCH - 1) // MCH
    CG = max(1, 4 // NR)
    tok = [nl.ndarray((P, NT), dtype=nl.bfloat16, buffer=nl.sbuf)[: (s_b - s_a) * B] for s_a, s_b in tiles]
    for g0 in range(0, n_mch, CG):
        g1 = min(n_mch, g0 + CG)
        lo, hi = g0 * MCH, min(NT, g1 * MCH)
        ps = {}
        for rt, (s_a, s_b) in enumerate(tiles):
            for mc in range(g0, g1):
                ps[rt, mc] = nl.ndarray((P, MCH), dtype=nl.float32, buffer=nl.psum)[: (s_b - s_a) * B]
        for s0 in range(0, H1, SLAB):
            w_sb = nl.ndarray((P, SLAB, hi - lo), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=w_sb, src=w_proj_p[:, s0 : s0 + SLAB, lo:hi])
            for hs in range(SLAB):
                for rt, (s_a, s_b) in enumerate(tiles):
                    for mc in range(g0, g1):
                        c0, c1 = mc * MCH, min(NT, (mc + 1) * MCH)
                        nisa.nc_matmul(
                            ps[rt, mc][:, 0 : c1 - c0], xT[:, s0 + hs, s_a * B : s_b * B], w_sb[:, hs, c0 - lo : c1 - lo]
                        )
        for rt in range(NR):
            for mc in range(g0, g1):
                c0, c1 = mc * MCH, min(NT, (mc + 1) * MCH)
                nisa.tensor_copy(dst=tok[rt][:, c0:c1], src=ps[rt, mc][:, 0 : c1 - c0])

    proj = nl.ndarray((P, n_ct, T), dtype=nl.bfloat16, buffer=nl.sbuf)  # feature-major q|k|v|z
    a_rows = nl.ndarray((vh, T), dtype=nl.float32, buffer=nl.sbuf)
    b_rows = nl.ndarray((vh, T), dtype=nl.float32, buffer=nl.sbuf)
    for rt, (s_a, s_b) in enumerate(tiles):
        rows = (s_b - s_a) * B
        r0, r1 = s_a * B, s_b * B
        for ct in range(n_ct):
            tpp = nl.ndarray((P, rows), dtype=nl.bfloat16, buffer=nl.psum)
            nisa.nc_transpose(dst=tpp, data=tok[rt][:, ct * P : (ct + 1) * P])
            nisa.tensor_copy(dst=proj[:, ct, r0:r1], src=tpp)
        for gi, dst in enumerate((a_rows, b_rows)):
            tpg = nl.ndarray((vh, rows), dtype=nl.bfloat16, buffer=nl.psum)
            nisa.nc_transpose(dst=tpg, data=tok[rt][:, NW + gi * vh : NW + (gi + 1) * vh])
            nisa.tensor_copy(dst=dst[:, r0:r1], src=tpg)

    # g = -exp(A_log) * softplus(a + dt_bias); eg = exp(g); beta = sigmoid(b)
    alog_sb = nl.ndarray((vh, 1), dtype=nl.float32, buffer=nl.sbuf)
    dt_sb = nl.ndarray((vh, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=alog_sb, src=a_log.reshape((vh, 1)))
    nisa.dma_copy(dst=dt_sb, src=dt_bias.reshape((vh, 1)))
    neg_exp_a = nl.ndarray((vh, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=neg_exp_a, op=nl.exp, data=alog_sb)
    nisa.tensor_scalar(dst=neg_exp_a, data=neg_exp_a, op0=nl.multiply, operand0=-1.0)
    sp = nl.ndarray((vh, T), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=sp, op=nl.softplus, data=a_rows, bias=dt_sb)
    g_rows = nl.ndarray((vh, T), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=g_rows, data=sp, op0=nl.multiply, operand0=neg_exp_a)
    eg_rows = nl.ndarray((vh, T), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=eg_rows, op=nl.exp, data=g_rows)
    beta_rows = nl.ndarray((vh, T), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=beta_rows, op=nl.sigmoid, data=b_rows)
    eg_bc = nl.ndarray((P, vh, T), dtype=nl.float32, buffer=nl.sbuf)
    beta_bc = nl.ndarray((P, vh, T), dtype=nl.float32, buffer=nl.sbuf)
    for h in range(vh):
        ps_e = nl.ndarray((P, T), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(ps_e, sel_h[:, h, :], eg_rows)
        nisa.tensor_copy(dst=eg_bc[:, h, :], src=ps_e)
        ps_b = nl.ndarray((P, T), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(ps_b, sel_h[:, h, :], beta_rows)
        nisa.tensor_copy(dst=beta_bc[:, h, :], src=ps_b)

    # ---------------- causal depthwise conv over [h0 h1 h2 x_0 .. x_{S-1}] + SiLU ----------------
    hist_tok = nl.ndarray((P, Cc, Kh), dtype=conv_in.dtype, buffer=nl.sbuf)[:B]
    conv_flat = conv_in.reshape((B * NS, Cc * Kh))
    nisa.dma_copy(
        dst=hist_tok.reshape((B, Cc * Kh)),
        src=conv_flat.ap(pattern=[[Cc * Kh, B], [1, Cc * Kh]], offset=0, vector_offset=conv_row, indirect_dim=0),
        dge_mode=nisa.dge_mode.swdge,
    )
    cw = nl.ndarray((P, n_conv_t, K), dtype=conv_w.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=cw, src=conv_w.ap(pattern=[[K, P], [P * K, n_conv_t], [1, K]], offset=0))
    cwf = nl.ndarray((P, n_conv_t, K), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=cwf, src=cw)
    convo = nl.ndarray((P, n_conv_t, T), dtype=nl.float32, buffer=nl.sbuf)
    snap = nl.ndarray((P, S, Cc, Kh), dtype=conv_in.dtype, buffer=nl.sbuf)[:B]  # token-major snapshots
    for ct in range(n_conv_t):
        cs = slice(ct * P, (ct + 1) * P)
        hist = nl.ndarray((P, Kh, B), dtype=conv_in.dtype, buffer=nl.sbuf)
        for j in range(Kh):
            tph = nl.ndarray((P, B), dtype=conv_in.dtype, buffer=nl.psum)
            nisa.nc_transpose(dst=tph, data=hist_tok[:, cs, j])
            nisa.tensor_copy(dst=hist[:, j, :], src=tph)

        def tap(i):  # extended sequence E[i], feature-major [P, B]
            return hist[:, i, :] if i < Kh else proj[:, ct, (i - Kh) * B : (i - Kh + 1) * B]

        for s in range(S):
            acc = nl.ndarray((P, B), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=acc, data=tap(s + Kh), op0=nl.multiply, operand0=cwf[:, ct, Kh : Kh + 1])
            for j in range(Kh):
                nisa.scalar_tensor_tensor(
                    dst=acc, data=tap(s + j), op0=nl.multiply, operand0=cwf[:, ct, j : j + 1], op1=nl.add,
                    operand1=acc,
                )
            conv_bf = nl.ndarray((P, B), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.activation(dst=conv_bf, op=nl.silu, data=acc)
            nisa.tensor_copy(dst=convo[:, ct, s * B : (s + 1) * B], src=conv_bf)

        # token-major new inputs, then snapshot s = [E[s+1] .. E[s+Kh]]
        x_tok = nl.ndarray((P, S, P), dtype=nl.bfloat16, buffer=nl.sbuf)[:B]
        for s in range(S):
            tpx = nl.ndarray((B, P), dtype=nl.bfloat16, buffer=nl.psum)
            nisa.nc_transpose(dst=tpx, data=proj[:, ct, s * B : (s + 1) * B])
            nisa.tensor_copy(dst=x_tok[:, s, :], src=tpx)
        for s in range(S):
            for j in range(Kh):
                i = s + 1 + j
                src = hist_tok[:, cs, i] if i < Kh else x_tok[:, i - Kh, :]
                nisa.tensor_copy(dst=snap[:, s, cs, j], src=src)
    nisa.dma_copy(dst=conv_out, src=snap)

    silu_z_all = nl.ndarray((P, vh, T), dtype=nl.float32, buffer=nl.sbuf)
    for h in range(vh):
        nisa.activation(dst=silu_z_all[:, h, :], op=nl.silu, data=proj[:, z_t0 + h, :])

    # ---------------- l2norm(q), l2norm(k), q.k (all T tokens) ----------------
    qn_f = nl.ndarray((P, kh, T), dtype=nl.float32, buffer=nl.sbuf)
    kn_f = nl.ndarray((P, kh, T), dtype=nl.float32, buffer=nl.sbuf)
    qk_bc = nl.ndarray((P, kh, T), dtype=nl.float32, buffer=nl.sbuf)
    kq_bf = nl.ndarray((P, kh, T, 2), dtype=nl.bfloat16, buffer=nl.sbuf)
    kn_bf = nl.ndarray((P, kh, T), dtype=nl.bfloat16, buffer=nl.sbuf)
    for j in range(kh):
        for src_t, dst, scale in ((q_t0 + j, qn_f, HEAD**-0.5), (k_t0 + j, kn_f, 1.0)):
            sq = nl.ndarray((P, T), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=sq, data1=convo[:, src_t, :], data2=convo[:, src_t, :], op=nl.multiply)
            ssq = nl.ndarray((P, T), dtype=nl.float32, buffer=nl.sbuf)
            _col_sum_bcast(ssq, sq, T, ones_col, ones_row)
            rs = nl.ndarray((P, T), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=rs, op=nl.rsqrt, data=ssq, bias=eps)
            nisa.scalar_tensor_tensor(
                dst=dst[:, j, :], data=convo[:, src_t, :], op0=nl.multiply, operand0=scale, op1=nl.multiply,
                operand1=rs,
            )
        prod = nl.ndarray((P, T), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=prod, data1=qn_f[:, j, :], data2=kn_f[:, j, :], op=nl.multiply)
        _col_sum_bcast(qk_bc[:, j, :], prod, T, ones_col, ones_row)
        nisa.tensor_copy(dst=kq_bf[:, j, :, 0], src=kn_f[:, j, :])
        nisa.tensor_copy(dst=kq_bf[:, j, :, 1], src=qn_f[:, j, :])
        nisa.tensor_copy(dst=kn_bf[:, j, :], src=kn_f[:, j, :])

    norm_sb = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=norm_sb, src=norm_w.reshape((P, 1)))
    o_all = nl.ndarray((P, vh, T), dtype=nl.bfloat16, buffer=nl.sbuf)

    # ---------------- recurrence: sequence groups x S sequential steps x all heads ----------------
    rec_flat = rec_in.reshape((B * NS * HEAD, vh * HEAD))
    out_pat = [[vh * P, P], [S * P * vh * P, G], [1, vh * P]]
    for b0 in range(0, B, G):
        # start state of G sequences from their src slots: [128k, G, vh*128v]
        S_cur = nl.ndarray((P, G, vh * P), dtype=nl.float32, buffer=nl.sbuf)
        if NS == 1:  # single slot: plain strided read (src_slot must be 0)
            nisa.dma_copy(
                dst=S_cur, src=rec_in.ap(pattern=[[vh * P, P], [P * vh * P, G], [1, vh * P]], offset=b0 * P * vh * P)
            )
        else:  # one indirect gather per sequence: partition p reads row (b*NS + slot[b])*128 + p
            for g in range(G):
                nisa.dma_copy(
                    dst=S_cur[:, g, :],
                    src=rec_flat.ap(
                        pattern=[[vh * P, P], [1, vh * P]], offset=0, vector_offset=rec_row[:, b0 + g : b0 + g + 1],
                        indirect_dim=0,
                    ),
                    dge_mode=nisa.dge_mode.swdge,
                )
        for s in range(S):
            c0 = s * B + b0  # token columns of (step s, sequences b0..b0+G)
            ksel = nl.ndarray((G, kh, G, P), dtype=nl.bfloat16, buffer=nl.sbuf)
            for j in range(kh):
                tpk = nl.ndarray((G, P), dtype=nl.bfloat16, buffer=nl.psum)
                nisa.nc_transpose(dst=tpk, data=kn_bf[:, j, c0 : c0 + G])
                k_tok = nl.ndarray((G, P), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.tensor_copy(dst=k_tok, src=tpk)
                for b in range(G):
                    nisa.tensor_scalar(
                        dst=ksel[:, j, b, :], data=k_tok, op0=nl.multiply, operand0=ident[:, b : b + 1],
                        engine=nisa.vector_engine if b % 2 == 0 else nisa.scalar_engine,
                    )

            ps_kq = nl.ndarray((P, vh, G, 2), dtype=nl.float32, buffer=nl.psum)
            for b in range(G):
                for h in range(vh):
                    j = h // gqa
                    S_bf = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.sbuf)
                    nisa.activation(dst=S_bf, op=nl.copy, data=S_cur[:, b, h * P : (h + 1) * P])
                    nisa.nc_matmul(ps_kq[:, h, b, :], S_bf, kq_bf[:, j, c0 + b, :])

            delta_toks = nl.ndarray((G, vh, P), dtype=nl.bfloat16, buffer=nl.sbuf)
            for h in range(vh):
                j = h // gqa
                egh = eg_bc[:, h, c0 : c0 + G]
                tmp = nl.ndarray((P, G), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=tmp, data1=egh, data2=ps_kq[:, h, :, 0], op=nl.multiply)
                diff = nl.ndarray((P, G), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=diff, data1=convo[:, v_t0 + h, c0 : c0 + G], data2=tmp, op=nl.subtract)
                delta = nl.ndarray((P, G), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=delta, data1=diff, data2=beta_bc[:, h, c0 : c0 + G], op=nl.multiply)
                o1 = nl.ndarray((P, G), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=o1, data1=egh, data2=ps_kq[:, h, :, 1], op=nl.multiply)
                o2 = nl.ndarray((P, G), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=o2, data1=qk_bc[:, j, c0 : c0 + G], data2=delta, op=nl.multiply)
                nisa.tensor_tensor(dst=o_all[:, h, c0 : c0 + G], data1=o1, data2=o2, op=nl.add)
                delta_bf = nl.ndarray((P, G), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.tensor_copy(dst=delta_bf, src=delta)
                tpd = nl.ndarray((G, P), dtype=nl.bfloat16, buffer=nl.psum)
                nisa.nc_transpose(dst=tpd, data=delta_bf)
                nisa.tensor_copy(dst=delta_toks[:, h, :], src=tpd)

            S_new = nl.ndarray((P, G, vh * P), dtype=nl.float32, buffer=nl.sbuf)
            for b in range(G):
                for h in range(vh):
                    j = h // gqa
                    kd = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.psum)
                    nisa.nc_matmul(kd, ksel[:, j, b, :], delta_toks[:, h, :])
                    nisa.scalar_tensor_tensor(
                        dst=S_new[:, b, h * P : (h + 1) * P], data=S_cur[:, b, h * P : (h + 1) * P], op0=nl.multiply,
                        operand0=eg_bc[:, h, c0 + b : c0 + b + 1], op1=nl.add, operand1=kd,
                    )
            # snapshot after step s
            nisa.dma_copy(dst=rec_out.ap(pattern=out_pat, offset=(b0 * S + s) * P * vh * P), src=S_new)
            S_cur = S_new

    # ---------------- gated RMSNorm over the head dim ----------------
    n_o = vh * T
    core = nl.ndarray((P, vh, T), dtype=nl.bfloat16, buffer=nl.sbuf)
    o_flat = o_all.reshape((P, n_o))
    osq = nl.ndarray((P, n_o), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=osq, data1=o_flat, data2=o_flat, op=nl.multiply)
    ms = nl.ndarray((P, n_o), dtype=nl.float32, buffer=nl.sbuf)
    _col_sum_bcast(ms, osq, n_o, ones_col, ones_row)
    rms = nl.ndarray((P, n_o), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=rms, op=nl.rsqrt, data=ms, scale=1.0 / HEAD, bias=eps)
    on_bf = nl.ndarray((P, n_o), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=on_bf, data1=o_flat, data2=rms, op=nl.multiply)
    nisa.scalar_tensor_tensor(
        dst=core.reshape((P, n_o)), data=on_bf, op0=nl.multiply, operand0=norm_sb, op1=nl.multiply,
        operand1=silu_z_all.reshape((P, n_o)),
    )

    # ---------------- out_proj per row tile, stored back in sequence-major order ----------------
    wo = nl.ndarray((P, vh, N), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=wo, src=w_out.ap(pattern=[[N, P], [P * N, vh], [1, N]], offset=0))
    for s_a, s_b in tiles:
        rows = (s_b - s_a) * B
        o_sb = nl.ndarray((P, N), dtype=nl.bfloat16, buffer=nl.sbuf)[:rows]
        for c0 in range(0, N, MCH):
            cn = min(MCH, N - c0)
            ps_o = nl.ndarray((P, MCH), dtype=nl.float32, buffer=nl.psum)[:rows]
            for h in range(vh):
                nisa.nc_matmul(ps_o[:, 0:cn], core[:, h, s_a * B : s_b * B], wo[:, h, c0 : c0 + cn])
            nisa.tensor_copy(dst=o_sb[:, c0 : c0 + cn], src=ps_o[:, 0:cn])
        for s in range(s_a, s_b):
            nisa.dma_copy(
                dst=out.ap(pattern=[[S * N, B], [1, N]], offset=s * N), src=o_sb[(s - s_a) * B : (s - s_a + 1) * B]
            )

    return out, conv_out, rec_out
