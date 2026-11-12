"""Multi-query (speculative verify) Qwen3.5 gated full-attention step for one TP rank (NKI, LNC=1).

Same block as ``attn_decode.attn_decode_tkg`` but each of the B sequences brings S = k + 1 new
tokens at positions ``pos[b] .. pos[b] + S - 1`` (last accepted token + k drafts). Token s of
sequence b attends to the cached history ``0 .. pos[b]-1`` plus new tokens ``0 .. s`` (causal
among the new tokens), and all S new K/V rows are written at ``pos[b] + s``. Writes of rejected
draft tokens need no rollback: the next round's tokens start at the first rejected position and
overwrite them, and the history mask (``column < pos``) never exposes stale slots.

Rows: ``x`` / ``out`` / ``cos`` / ``sin`` are sequence-major token rows ``t = b*S + s`` (what the
attention-TP AllGather of per-rank [S, H] shards yields). Score rows are ``r = t*nq + h``; all
``R = B*S*nq <= 128`` rows share one PSUM score tile, exactly as the 1-token kernel's B*nq rows.
The S new keys enter as S extra score columns with a causal bias, the S new values as an S-row
P.V matmul, so the history read never depends on this step's KV writes.
"""

import nki
import nki.isa as nisa
import nki.language as nl

from nkilib.core.utils.kernel_assert import kernel_assert

P = 128
ROT = 64
SCH = 2048
NEG = 30000.0
MCH = 512


def _row_to_col(row, n):
    """[1, n] fp32 SBUF row -> [n, 1] fp32 SBUF column (outer product with a 1x1 one on the PE)."""
    one = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(one, 1.0)
    ps = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.psum)[:n]
    nisa.nc_matmul(ps, row, one)
    col = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)[:n]
    nisa.tensor_copy(dst=col, src=ps)
    return col


@nki.jit
def attn_verify_tkg(
    x,  # [B*S, H] bf16 (post input-RMSNorm), sequence-major rows
    w_qkv,  # [H, (nq + 2) * d] bf16
    q_norm_w,  # [1, d] fp32 raw zero-centred
    k_norm_w,  # [1, d] fp32
    w_gate_p,  # [128, H/128, nq * d] bf16
    cos,  # [B*S, ROT/2] fp32 cos((pos[b] + s) * inv_freq)
    sin,  # [B*S, ROT/2] fp32
    k_cache_t,  # [B, d, L] bf16
    v_cache,  # [B, L, d] bf16
    pos,  # [B, 1] int32 position of each sequence's first new token (= cached history length)
    w_o,  # [nq * d, H] bf16
):
    B = pos.shape[0]
    T, H = x.shape
    S = T // B
    d = v_cache.shape[2]
    L = v_cache.shape[1]
    nq = w_gate_p.shape[2] // d
    nd = nq * d
    H1 = H // P
    DH = d // P
    SQ = S * nq  # score rows per sequence
    R = T * nq
    NQKV = (nq + 2) * d
    n_ot = nd // P
    half = ROT // 2
    kernel_assert(T == B * S, "x rows must be B * S")
    kernel_assert(d % P == 0 and R <= P and L % P == 0, f"need d%128==0, B*S*nq<=128, L%128==0 (R={R}, L={L})")
    kernel_assert(d * L < (1 << 24) and B * L < (1 << 24), "cache element offsets must stay below 2^24")

    # ---------------- x^T [128, H1, T] ----------------
    x_sb = nl.ndarray((P, H), dtype=nl.bfloat16, buffer=nl.sbuf)[:T]
    nisa.dma_copy(dst=x_sb, src=x)
    xT = nl.ndarray((P, H1, T), dtype=nl.bfloat16, buffer=nl.sbuf)
    for h1 in range(H1):
        tp = nl.ndarray((P, T), dtype=nl.bfloat16, buffer=nl.psum)
        nisa.nc_transpose(dst=tp, data=x_sb[:, h1 * P : (h1 + 1) * P])
        nisa.tensor_copy(dst=xT[:, h1, :], src=tp)

    # ---------------- QKV projection -> token-major [T, NQKV] ----------------
    qkv = nl.ndarray((P, NQKV), dtype=nl.bfloat16, buffer=nl.sbuf)[:T]
    for c0 in range(0, NQKV, MCH):
        cn = min(MCH, NQKV - c0)
        wq = nl.ndarray((P, H1, MCH), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=wq[:, :, 0:cn], src=w_qkv.ap(pattern=[[NQKV, P], [P * NQKV, H1], [1, cn]], offset=c0))
        ps = nl.ndarray((P, MCH), dtype=nl.float32, buffer=nl.psum)[:T]
        for h1 in range(H1):
            nisa.nc_matmul(ps[:, 0:cn], xT[:, h1, :], wq[:, h1, 0:cn])
        nisa.tensor_copy(dst=qkv[:, c0 : c0 + cn], src=ps[:, 0:cn])

    # ---------------- Q/K zero-centred RMSNorm per head ----------------
    NH = nq + 1
    wn = nl.ndarray((P, NH, d), dtype=nl.float32, buffer=nl.sbuf)[:T]
    for hh in range(NH):
        src = q_norm_w if hh < nq else k_norm_w
        nisa.dma_copy(dst=wn[:, hh, :], src=src.ap(pattern=[[0, T], [1, d]], offset=0))
    nisa.tensor_scalar(dst=wn, data=wn, op0=nl.add, operand0=1.0)
    ss = nl.ndarray((P, NH), dtype=nl.float32, buffer=nl.sbuf)[:T]
    for hh in range(NH):
        sq = nl.ndarray((P, d), dtype=nl.float32, buffer=nl.sbuf)[:T]
        nisa.activation_reduce(
            dst=sq, op=nl.square, data=qkv[:, hh * d : (hh + 1) * d], reduce_op=nl.add, reduce_res=ss[:, hh : hh + 1]
        )
    rs = nl.ndarray((P, NH), dtype=nl.float32, buffer=nl.sbuf)[:T]
    nisa.activation(dst=rs, op=nl.rsqrt, data=ss, scale=1.0 / d, bias=1e-6)
    qk = nl.ndarray((P, NH, d), dtype=nl.bfloat16, buffer=nl.sbuf)[:T]
    for hh in range(NH):
        nisa.scalar_tensor_tensor(
            dst=qk[:, hh, :], data=qkv[:, hh * d : (hh + 1) * d], op0=nl.multiply, operand0=rs[:, hh : hh + 1],
            op1=nl.multiply, operand1=wn[:, hh, :],
        )

    # ---------------- partial RoPE (per-token positions pos[b] + s) ----------------
    cs = nl.ndarray((P, 2, NH, half), dtype=nl.float32, buffer=nl.sbuf)[:T]
    nisa.dma_copy(dst=cs[:, 0, :, :], src=cos.ap(pattern=[[half, T], [0, NH], [1, half]], offset=0))
    nisa.dma_copy(dst=cs[:, 1, :, :], src=sin.ap(pattern=[[half, T], [0, NH], [1, half]], offset=0))
    x1 = qk[:, :, 0:half]
    x2 = qk[:, :, half:ROT]
    t1 = nl.ndarray((P, NH, half), dtype=nl.float32, buffer=nl.sbuf)[:T]
    t2 = nl.ndarray((P, NH, half), dtype=nl.float32, buffer=nl.sbuf)[:T]
    r1 = nl.ndarray((P, NH, half), dtype=nl.float32, buffer=nl.sbuf)[:T]
    r2 = nl.ndarray((P, NH, half), dtype=nl.float32, buffer=nl.sbuf)[:T]
    nisa.tensor_tensor(dst=t1, data1=x1, data2=cs[:, 0, :, :], op=nl.multiply)
    nisa.tensor_tensor(dst=t2, data1=x2, data2=cs[:, 1, :, :], op=nl.multiply)
    nisa.tensor_tensor(dst=r1, data1=t1, data2=t2, op=nl.subtract)
    nisa.tensor_tensor(dst=t1, data1=x2, data2=cs[:, 0, :, :], op=nl.multiply)
    nisa.tensor_tensor(dst=t2, data1=x1, data2=cs[:, 1, :, :], op=nl.multiply)
    nisa.tensor_tensor(dst=r2, data1=t1, data2=t2, op=nl.add)
    nisa.tensor_copy(dst=qk[:, :, 0:half], src=r1)
    nisa.tensor_copy(dst=qk[:, :, half:ROT], src=r2)

    # ---------------- query / new-key transposes ----------------
    # Qz[:, b, j] = scaled q^T of sequence b's S tokens (all heads) in score columns b*SQ .. b*SQ+SQ-1,
    # zero elsewhere: accumulating Qz_b . K_b^T over b gives every row its own sequence's scores.
    scale = float(d) ** -0.5
    qT = nl.ndarray((P, DH, T, nq), dtype=nl.bfloat16, buffer=nl.sbuf)
    for h in range(nq):
        for j in range(DH):
            tp = nl.ndarray((P, T), dtype=nl.bfloat16, buffer=nl.psum)
            nisa.nc_transpose(dst=tp, data=qk[:, h, j * P : (j + 1) * P])
            nisa.tensor_scalar(dst=qT[:, j, :, h], data=tp, op0=nl.multiply, operand0=scale)
    Qz5 = nl.ndarray((P, B, DH, T, nq), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.memset(Qz5, 0.0)
    for b in range(B):
        nisa.tensor_copy(dst=Qz5[:, b, :, b * S : (b + 1) * S, :], src=qT[:, :, b * S : (b + 1) * S, :])
    Qz = Qz5.reshape((P, B, DH, R))
    kT_new = nl.ndarray((P, DH, T), dtype=nl.bfloat16, buffer=nl.sbuf)
    for j in range(DH):
        tp = nl.ndarray((P, T), dtype=nl.bfloat16, buffer=nl.psum)
        nisa.nc_transpose(dst=tp, data=qk[:, nq, j * P : (j + 1) * P])
        nisa.tensor_copy(dst=kT_new[:, j, :], src=tp)
    v_new = qkv[:, nd + d : nd + 2 * d]  # [T, d]

    # ---------------- masks ----------------
    # history: column c valid iff c < pos[b]
    pos_r = nl.ndarray((P, 1), dtype=nl.int32, buffer=nl.sbuf)[:R]
    for b in range(B):
        nisa.dma_copy(dst=pos_r[b * SQ : (b + 1) * SQ, :], src=pos.ap(pattern=[[0, SQ], [1, 1]], offset=b))
    pos_rf = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)[:R]
    nisa.tensor_copy(dst=pos_rf, src=pos_r)
    iota_s = nl.ndarray((P, L), dtype=nl.int32, buffer=nl.sbuf)[:R]
    nisa.iota(iota_s, pattern=[[1, L]], offset=0, channel_multiplier=0)
    bias = nl.ndarray((P, L), dtype=nl.float32, buffer=nl.sbuf)[:R]
    nisa.tensor_scalar(dst=bias, data=iota_s, op0=nl.less, operand0=pos_rf)
    nisa.tensor_scalar(dst=bias, data=bias, op0=nl.subtract, operand0=1.0, op1=nl.multiply, operand1=NEG)
    # new tokens: row r = (b*S + s)*nq + h may see new column s' iff s' <= s
    srow_i = nl.ndarray((1, B, S, nq), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(srow_i, pattern=[[0, B], [1, S], [0, nq]], offset=0, channel_multiplier=0)
    srow = nl.ndarray((1, R), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=srow, src=srow_i.reshape((1, R)))
    s_col = _row_to_col(srow, R)
    iota_n = nl.ndarray((P, S), dtype=nl.int32, buffer=nl.sbuf)[:R]
    nisa.iota(iota_n, pattern=[[1, S]], offset=0, channel_multiplier=0)
    bias_n = nl.ndarray((P, S), dtype=nl.float32, buffer=nl.sbuf)[:R]
    nisa.tensor_scalar(dst=bias_n, data=iota_n, op0=nl.greater, operand0=s_col)
    nisa.tensor_scalar(dst=bias_n, data=bias_n, op0=nl.multiply, operand0=-NEG)

    # ---------------- scores [R, L + S] ----------------
    sc = nl.ndarray((P, L + S), dtype=nl.float32, buffer=nl.sbuf)[:R]
    for s0 in range(0, L, SCH):
        sw = min(SCH, L - s0)
        ps_s = nl.ndarray((P, SCH), dtype=nl.float32, buffer=nl.psum)[:R]
        for b in range(B):
            kT = nl.ndarray((P, DH, SCH), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=kT[:, :, 0:sw], src=k_cache_t.ap(pattern=[[L, P], [P * L, DH], [1, sw]], offset=b * d * L + s0))
            for j in range(DH):
                for c0 in range(0, sw, MCH):
                    cn = min(MCH, sw - c0)
                    nisa.nc_matmul(ps_s[:, c0 : c0 + cn], Qz[:, b, j, :], kT[:, j, c0 : c0 + cn])
        nisa.tensor_tensor(dst=sc[:, s0 : s0 + sw], data1=ps_s[:, 0:sw], data2=bias[:, s0 : s0 + sw], op=nl.add)
    ps_n = nl.ndarray((P, S), dtype=nl.float32, buffer=nl.psum)[:R]
    for b in range(B):
        for j in range(DH):
            nisa.nc_matmul(ps_n, Qz[:, b, j, :], kT_new[:, j, b * S : (b + 1) * S])
    nisa.tensor_tensor(dst=sc[:, L : L + S], data1=ps_n, data2=bias_n, op=nl.add)

    # ---------------- softmax ----------------
    mx = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)[:R]
    nisa.tensor_reduce(dst=mx, op=nl.max, data=sc, axis=1)
    nisa.tensor_scalar(dst=mx, data=mx, op0=nl.multiply, operand0=-1.0)
    ex = nl.ndarray((P, L + S), dtype=nl.float32, buffer=nl.sbuf)[:R]
    ssum = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)[:R]
    nisa.activation_reduce(dst=ex, op=nl.exp, data=sc, reduce_op=nl.add, reduce_res=ssum, bias=mx)
    rsum = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)[:R]
    nisa.reciprocal(dst=rsum, data=ssum)
    pb = nl.ndarray((P, L + S), dtype=nl.bfloat16, buffer=nl.sbuf)[:R]
    nisa.tensor_scalar(dst=pb, data=ex, op0=nl.multiply, operand0=rsum)

    NC = L // P
    PT = nl.ndarray((P, NC, R), dtype=nl.bfloat16, buffer=nl.sbuf)
    for c in range(NC):
        tp = nl.ndarray((P, R), dtype=nl.bfloat16, buffer=nl.psum)
        nisa.nc_transpose(dst=tp, data=pb[:, c * P : (c + 1) * P])
        nisa.tensor_copy(dst=PT[:, c, :], src=tp)
    PTn = nl.ndarray((S, R), dtype=nl.bfloat16, buffer=nl.sbuf)  # new-token probabilities, transposed
    tpn = nl.ndarray((S, R), dtype=nl.bfloat16, buffer=nl.psum)
    nisa.nc_transpose(dst=tpn, data=pb[:, L : L + S])
    nisa.tensor_copy(dst=PTn, src=tpn)
    # new V rows of every sequence on partitions 0..S-1: v_rows[s, b, :] = v_new[b*S + s]
    v_hbm = nl.ndarray((T, d), dtype=nl.bfloat16, buffer=nl.private_hbm)
    nisa.dma_copy(dst=v_hbm, src=v_new)
    v_rows = nl.ndarray((S, B, d), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=v_rows, src=v_hbm.ap(pattern=[[d, S], [S * d, B], [1, d]], offset=0))

    # ---------------- P.V per sequence -> attn [T, nq*d] (HBM hop to token-major) ----------------
    attn_hbm = nl.ndarray((B, SQ, d), dtype=nl.bfloat16, buffer=nl.private_hbm)
    VCH = SCH // P
    for b in range(B):
        ps_o = nl.ndarray((SQ, d), dtype=nl.float32, buffer=nl.psum)
        for c0 in range(0, NC, VCH):
            cn = min(VCH, NC - c0)
            vb = nl.ndarray((P, VCH, d), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=vb[:, 0:cn, :], src=v_cache.ap(pattern=[[d, P], [P * d, cn], [1, d]], offset=(b * L + c0 * P) * d))
            for c in range(cn):
                nisa.nc_matmul(ps_o, PT[:, c0 + c, b * SQ : (b + 1) * SQ], vb[:, c, :])
        nisa.nc_matmul(ps_o, PTn[:, b * SQ : (b + 1) * SQ], v_rows[:, b, :])
        o_b = nl.ndarray((SQ, d), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=o_b, src=ps_o)
        nisa.dma_copy(dst=attn_hbm[b], src=o_b)

    # ---------------- KV writes for all S new tokens ----------------
    # K (transposed cache): the S new positions of (sequence b, dim row) are contiguous, so one
    # indirect DMA per (head-dim tile j, sequence b) writes [128 dims x S positions].
    pos_bc = nl.ndarray((P, DH * B), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=pos_bc.reshape((P, DH, B)), src=pos.ap(pattern=[[0, P], [0, DH], [1, B]], offset=0))
    k_off = nl.ndarray((P, DH * B), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(k_off.reshape((P, DH, B)), pattern=[[P, DH], [0, B]], offset=0, channel_multiplier=1)
    nisa.tensor_scalar(dst=k_off, data=k_off, op0=nl.multiply, operand0=L)
    nisa.tensor_tensor(dst=k_off, data1=k_off, data2=pos_bc, op=nl.add)
    k_flat = k_cache_t.reshape((B * d * L, 1))
    for j in range(DH):
        for b in range(B):
            nisa.dma_copy(
                dst=k_flat.ap(
                    pattern=[[1, P], [1, S]], offset=b * d * L, vector_offset=k_off[:, j * B + b : j * B + b + 1],
                    indirect_dim=0,
                ),
                src=kT_new[:, j, b * S : (b + 1) * S],
                dge_mode=nisa.dge_mode.swdge,
            )
    # V: token t = b*S + s goes to row b*L + pos[b] + s of v_cache viewed as [B*L, d]
    vrow_i = nl.ndarray((1, B, S), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(vrow_i, pattern=[[L, B], [1, S]], offset=0, channel_multiplier=0)
    vrow = nl.ndarray((1, T), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=vrow, src=vrow_i.reshape((1, T)))
    v_base = _row_to_col(vrow, T)
    pos_t = nl.ndarray((P, 1), dtype=nl.int32, buffer=nl.sbuf)[:T]
    for b in range(B):
        nisa.dma_copy(dst=pos_t[b * S : (b + 1) * S, :], src=pos.ap(pattern=[[0, S], [1, 1]], offset=b))
    pos_tf = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)[:T]
    nisa.tensor_copy(dst=pos_tf, src=pos_t)
    v_off_f = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)[:T]
    nisa.tensor_tensor(dst=v_off_f, data1=v_base, data2=pos_tf, op=nl.add)
    v_off = nl.ndarray((P, 1), dtype=nl.int32, buffer=nl.sbuf)[:T]
    nisa.tensor_copy(dst=v_off, src=v_off_f)
    v_flat = v_cache.reshape((B * L, d))
    nisa.dma_copy(
        dst=v_flat.ap(pattern=[[d, T], [1, d]], offset=0, vector_offset=v_off, indirect_dim=0),
        src=v_new,
        dge_mode=nisa.dge_mode.swdge,
    )

    # ---------------- gate = sigmoid(x W_gate) ----------------
    a_sb = nl.ndarray((P, nd), dtype=nl.bfloat16, buffer=nl.sbuf)[:T]
    nisa.dma_copy(dst=a_sb, src=attn_hbm.reshape((T, nd)))
    og = nl.ndarray((P, nd), dtype=nl.bfloat16, buffer=nl.sbuf)[:T]
    for c0 in range(0, nd, MCH):
        cn = min(MCH, nd - c0)
        wg = nl.ndarray((P, H1, MCH), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=wg[:, :, 0:cn], src=w_gate_p[:, :, c0 : c0 + cn])
        ps_g = nl.ndarray((P, MCH), dtype=nl.float32, buffer=nl.psum)[:T]
        for h1 in range(H1):
            nisa.nc_matmul(ps_g[:, 0:cn], xT[:, h1, :], wg[:, h1, 0:cn])
        g_bf = nl.ndarray((P, MCH), dtype=nl.bfloat16, buffer=nl.sbuf)[:T]
        nisa.tensor_copy(dst=g_bf[:, 0:cn], src=ps_g[:, 0:cn])
        sg = nl.ndarray((P, MCH), dtype=nl.float32, buffer=nl.sbuf)[:T]
        nisa.activation(dst=sg[:, 0:cn], op=nl.sigmoid, data=g_bf[:, 0:cn])
        nisa.tensor_tensor(dst=og[:, c0 : c0 + cn], data1=a_sb[:, c0 : c0 + cn], data2=sg[:, 0:cn], op=nl.multiply)

    # ---------------- o_proj ----------------
    oT = nl.ndarray((P, n_ot, T), dtype=nl.bfloat16, buffer=nl.sbuf)
    for t in range(n_ot):
        tpo = nl.ndarray((P, T), dtype=nl.bfloat16, buffer=nl.psum)
        nisa.nc_transpose(dst=tpo, data=og[:, t * P : (t + 1) * P])
        nisa.tensor_copy(dst=oT[:, t, :], src=tpo)
    N = w_o.shape[1]  # out-projection width: H, or H + folded router columns (low-C verify)
    out = nl.ndarray((T, N), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    o_sb = nl.ndarray((P, N), dtype=nl.bfloat16, buffer=nl.sbuf)[:T]
    for c0 in range(0, N, MCH):
        cn = min(MCH, N - c0)
        wo = nl.ndarray((P, n_ot, MCH), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=wo[:, :, 0:cn], src=w_o.ap(pattern=[[N, P], [P * N, n_ot], [1, cn]], offset=c0))
        ps_o = nl.ndarray((P, MCH), dtype=nl.float32, buffer=nl.psum)[:T]
        for t in range(n_ot):
            nisa.nc_matmul(ps_o[:, 0:cn], oT[:, t, :], wo[:, t, 0:cn])
        nisa.tensor_copy(dst=o_sb[:, c0 : c0 + cn], src=ps_o[:, 0:cn])
    nisa.dma_copy(dst=out, src=o_sb)
    return out, k_cache_t, v_cache
