# SPDX-License-Identifier: Apache-2.0
"""MiMo-V2's prefill attention front in one NKI kernel (attention.MiMoAttention.prefill): what the chunk's
input RMSNorm, QKV projection, RoPE, FP8 KV scaling and the paged K/V write did in XLA (and kv_write.py's scatter).

Per 128-token tile (the tiles alternate between the two programs of LNC2), with this rank's weight resident in SBUF:

    xn       = RMSNorm(x)                     fp32 math, bf16 result, times the weight (transformer/model.RMSNorm)
    qkv      = xn @ W_qkv                     [128, (heads + 2) x d] rounded to bf16, as the bf16 matmul's output
    q, k     = RoPE(q), RoPE(k)               contiguous halves (rope.rotate): t x [c, c] + [-t2, t1] x [s, s], in fp32
    k, v     = clamp(t x kv_scale, +-240)     over an FP8 cache (attention.FP8_KV_MAX), then the cache's dtype
    cache    = k, v at slot                   one KV head a rank: the flat row is the slot; a slot outside the pool -> 0

Returns q [heads, T, d] bf16 (the segmented attention's layout), the chunk's K / V rows [T, d] as the cache holds them
(a DCP prefill attends them directly) and the two caches, so the compiler reports the write as an output alias and the
attention that reads them is ordered after it (kv_write.py)."""

import nki
import nki.isa as nisa
import nki.language as nl
from nki.isa import oob_mode

P = 128
COLS = 512  # fp32 columns of one PSUM bank
FP8_KV_MAX = 240.0


def _rope_half(t32, c, s, h0: int, half: int, out_bf, name_tmp):
    """RoPE on the head at columns h0 .. h0 + 2 x half of ``t32`` [128, *] fp32, into ``out_bf`` [128, 2 x half] bf16
    (rope.rotate's fp32 math): o1 = t1 c - t2 s, o2 = t2 c + t1 s."""
    a = nl.ndarray((P, half), dtype=nl.float32, buffer=nl.sbuf)
    b = nl.ndarray((P, half), dtype=nl.float32, buffer=nl.sbuf)
    o = nl.ndarray((P, 2 * half), dtype=nl.float32, buffer=nl.sbuf)
    t1 = t32[0:P, h0:h0 + half]
    t2 = t32[0:P, h0 + half:h0 + 2 * half]
    nisa.tensor_tensor(dst=a[0:P, 0:half], data1=t1, data2=c[0:P, 0:half], op=nl.multiply)
    nisa.tensor_tensor(dst=b[0:P, 0:half], data1=t2, data2=s[0:P, 0:half], op=nl.multiply)
    nisa.tensor_tensor(dst=o[0:P, 0:half], data1=a[0:P, 0:half], data2=b[0:P, 0:half], op=nl.subtract)
    nisa.tensor_tensor(dst=a[0:P, 0:half], data1=t2, data2=c[0:P, 0:half], op=nl.multiply)
    nisa.tensor_tensor(dst=b[0:P, 0:half], data1=t1, data2=s[0:P, 0:half], op=nl.multiply)
    nisa.tensor_tensor(dst=o[0:P, half:2 * half], data1=a[0:P, 0:half], data2=b[0:P, 0:half], op=nl.add)
    nisa.tensor_copy(dst=out_bf[0:P, 0:2 * half], src=o[0:P, 0:2 * half])


def _to_cache(src_bf, d: int, kv_scale: float, dtype):
    """[128, d] bf16 -> the cache's dtype, times the FP8 scale and clamped where the cache is FP8."""
    if dtype == nl.bfloat16:
        return src_bf
    f = nl.ndarray((P, d), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=f[0:P, 0:d], data=src_bf[0:P, 0:d], op0=nl.multiply, operand0=kv_scale)
    # the torch path scales the bf16 K / V in bf16: round the product back through bf16 first
    r = nl.ndarray((P, d), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=r[0:P, 0:d], src=f[0:P, 0:d])
    nisa.tensor_scalar(dst=r[0:P, 0:d], data=r[0:P, 0:d], op0=nl.minimum, operand0=FP8_KV_MAX, op1=nl.maximum,
                       operand1=-FP8_KV_MAX)
    out = nl.ndarray((P, d), dtype=dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=out[0:P, 0:d], src=r[0:P, 0:d])
    return out


@nki.jit
def prefill_front(x, norm_w, w_qkv, cos, sin, slots, k_cache, v_cache, eps=1e-6, num_heads=1, kv_scale=1.0,
                  q_scale=1.0):
    """x [T, H] bf16 (T a multiple of 128) the chunk's residual stream; norm_w [1, H] bf16; w_qkv [H, (heads + 2) x d]
    bf16 (one KV head); cos / sin [T, d / 2] bf16 (rope.RotaryTables); slots [T, 1] int32 the pool slots; k_cache /
    v_cache [blocks, 1, page, d] (bf16, or FP8 at ``kv_scale``). Returns (q [heads, T, d] bf16 times ``q_scale`` (the
    softmax scale the segmented attention applies to q, rounded as its bf16 product), k [T, d], v [T, d] in the cache's
    dtype, k_cache, v_cache)."""
    T, H = x.shape
    I = w_qkv.shape[1]
    blocks, kv_heads, page, d = k_cache.shape
    half = d // 2
    nq = num_heads
    H1 = H // P
    n_rows = blocks * kv_heads * page
    n_prg = nl.num_programs(axes=0)
    pid = nl.program_id(axis=0)
    k_flat = k_cache.reshape((n_rows, d))
    v_flat = v_cache.reshape((n_rows, d))

    # this rank's weight, resident: [128, H / 128, I], row h1 x 128 + p at [p, h1, :]
    w = nl.ndarray((P, H1, I), dtype=w_qkv.dtype, buffer=nl.sbuf)
    for h1 in range(H1):
        nisa.dma_copy(dst=w[0:P, h1, 0:I], src=w_qkv.ap(pattern=[[I, P], [1, I]], offset=h1 * P * I),
                      dge_mode=nisa.dge_mode.hwdge)
    gam = nl.ndarray((P, H), dtype=norm_w.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=gam[0:P, 0:H], src=norm_w.ap(pattern=[[0, P], [1, H]], offset=0))

    q_out = nl.ndarray((nq, T, d), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="prefill_front_q")
    k_out = nl.ndarray((T, d), dtype=k_cache.dtype, buffer=nl.shared_hbm, name="prefill_front_k")
    v_out = nl.ndarray((T, d), dtype=v_cache.dtype, buffer=nl.shared_hbm, name="prefill_front_v")
    # the tiles alternate between the programs (a runtime loop: one copy of the body, not one a tile); with fewer
    # tiles than programs each runs them all (the same values to the same places), since an idle program fails the
    # compiler's allocation
    NT = T // P
    split = n_prg > 1 and NT >= n_prg and NT % n_prg == 0
    for i in nl.sequential_range(NT // n_prg if split else NT):
        t0 = (i * n_prg + pid) * P if split else i * P
        # ---- RMSNorm ----
        xt = nl.ndarray((P, H), dtype=x.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=xt[0:P, 0:H], src=x[nl.ds(t0, P), 0:H])
        sq = nl.ndarray((P, H), dtype=nl.float32, buffer=nl.sbuf)
        ss = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=sq[0:P, 0:H], op=nl.square, data=xt[0:P, 0:H], reduce_op=nl.add, reduce_res=ss[0:P, 0:1],
                        reduce_cmd=nisa.reduce_cmd.reset_reduce)
        rr = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=rr[0:P, 0:1], op=nl.rsqrt, data=ss[0:P, 0:1], scale=1.0 / H, bias=eps)
        nb = nl.ndarray((P, H), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=nb[0:P, 0:H], data=xt[0:P, 0:H], op0=nl.multiply, operand0=rr[0:P, 0:1])
        xn = nl.ndarray((P, H), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=xn[0:P, 0:H], data1=nb[0:P, 0:H], data2=gam[0:P, 0:H], op=nl.multiply)
        xT = nl.ndarray((P, H1, P), dtype=nl.bfloat16, buffer=nl.sbuf)
        for h1 in range(H1):
            tp = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.psum)
            nisa.nc_transpose(dst=tp[0:P, 0:P], data=xn[0:P, h1 * P:(h1 + 1) * P])
            nisa.tensor_copy(dst=xT[0:P, h1, 0:P], src=tp[0:P, 0:P])
        # ---- the projection, rounded to bf16 then held in fp32 ----
        qkv = nl.ndarray((P, I), dtype=nl.float32, buffer=nl.sbuf)
        for c0 in range(0, I, COLS):
            cs = min(COLS, I - c0)
            ps = nl.ndarray((P, COLS), dtype=nl.float32, buffer=nl.psum)
            for h1 in range(H1):
                nisa.nc_matmul(dst=ps[0:P, 0:cs], stationary=xT[0:P, h1, 0:P], moving=w[0:P, h1, c0:c0 + cs])
            rb = nl.ndarray((P, COLS), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=rb[0:P, 0:cs], src=ps[0:P, 0:cs])
            nisa.tensor_copy(dst=qkv[0:P, c0:c0 + cs], src=rb[0:P, 0:cs])
        # ---- RoPE on the query heads and K ----
        c = nl.ndarray((P, half), dtype=nl.float32, buffer=nl.sbuf)
        s = nl.ndarray((P, half), dtype=nl.float32, buffer=nl.sbuf)
        cb = nl.ndarray((P, half), dtype=cos.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=cb[0:P, 0:half], src=cos[nl.ds(t0, P), 0:half])
        nisa.tensor_copy(dst=c[0:P, 0:half], src=cb[0:P, 0:half])
        sb = nl.ndarray((P, half), dtype=sin.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=sb[0:P, 0:half], src=sin[nl.ds(t0, P), 0:half])
        nisa.tensor_copy(dst=s[0:P, 0:half], src=sb[0:P, 0:half])
        for h in range(nq):
            qh = nl.ndarray((P, d), dtype=nl.bfloat16, buffer=nl.sbuf)
            _rope_half(qkv, c, s, h * d, half, qh, f"q{h}")
            if q_scale != 1.0:
                nisa.tensor_scalar(dst=qh[0:P, 0:d], data=qh[0:P, 0:d], op0=nl.multiply, operand0=q_scale)
            nisa.dma_copy(dst=q_out[h, nl.ds(t0, P), 0:d], src=qh[0:P, 0:d])
        kh = nl.ndarray((P, d), dtype=nl.bfloat16, buffer=nl.sbuf)
        _rope_half(qkv, c, s, nq * d, half, kh, "k")
        vh = nl.ndarray((P, d), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=vh[0:P, 0:d], src=qkv[0:P, (nq + 1) * d:(nq + 2) * d])
        kc = _to_cache(kh, d, kv_scale, k_cache.dtype)
        vc = _to_cache(vh, d, kv_scale, v_cache.dtype)
        nisa.dma_copy(dst=k_out[nl.ds(t0, P), 0:d], src=kc[0:P, 0:d])
        nisa.dma_copy(dst=v_out[nl.ds(t0, P), 0:d], src=vc[0:P, 0:d])
        # ---- the slots: outside the pool -> 0 (the reserved padding page) ----
        sl = nl.ndarray((P, 1), dtype=slots.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=sl[0:P, 0:1], src=slots[nl.ds(t0, P), 0:1])
        sf = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=sf[0:P, 0:1], src=sl[0:P, 0:1])
        ok = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=ok[0:P, 0:1], data=sf[0:P, 0:1], op0=nl.greater_equal, operand0=0.0)
        lt = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=lt[0:P, 0:1], data=sf[0:P, 0:1], op0=nl.less, operand0=float(n_rows))
        nisa.tensor_tensor(dst=ok[0:P, 0:1], data1=ok[0:P, 0:1], data2=lt[0:P, 0:1], op=nl.multiply)
        nisa.tensor_tensor(dst=sf[0:P, 0:1], data1=sf[0:P, 0:1], data2=ok[0:P, 0:1], op=nl.multiply)
        idx = nl.ndarray((P, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=idx[0:P, 0:1], src=sf[0:P, 0:1])
        nisa.dma_copy(dst=k_flat.ap(pattern=[[d, P], [1, d]], offset=0, vector_offset=idx[0:P, 0:1], indirect_dim=0),
                      src=kc[0:P, 0:d], oob_mode=oob_mode.skip)
        nisa.dma_copy(dst=v_flat.ap(pattern=[[d, P], [1, d]], offset=0, vector_offset=idx[0:P, 0:1], indirect_dim=0),
                      src=vc[0:P, 0:d], oob_mode=oob_mode.skip)
    return q_out, k_out, v_out, k_cache, v_cache


@nki.jit
def prefill_back(attn, o_w, x, norm_w, mesh, eps=1e-6, v_head_dim=128, kv_scale=1.0):
    """The chunk's attention output through this rank's output projection, all-reduced over ``mesh``, added to the
    residual stream and normed for the MLP, in one kernel (attention.MiMoAttention._prefill_attend's tail, the decoder
    layer's residual add and post-attention RMSNorm):

        h  = x + all_reduce((attn / kv_scale)[:, :, :v_head_dim] @ o_w)
        hn = RMSNorm(h)                       fp32 math, bf16 result, times the weight

    attn [heads, T, d] bf16 (the segmented attention's output, T a multiple of 128); o_w [heads x v_head_dim, H] bf16;
    x [T, H] bf16; norm_w [1, H] bf16. Returns (h, hn) [T, H] bf16. The two programs of LNC2 take one half of H each
    (the projection's columns and the residual); the norm's sum of squares crosses them by sendrecv. The all-reduce is
    one, of the chunk's partial in shared HBM between core barriers (_hbm_all_reduce)."""
    from sglang_neuron.models.mimo_v2.kernels.head import _replica_group

    nq, T, d = attn.shape
    HV, H = o_w.shape
    vd = v_head_dim
    K1 = HV // P
    n_prg = nl.num_programs(axes=0)
    pid = nl.program_id(axis=0)
    other = 1 - pid
    HH = H // n_prg
    h0 = pid * HH
    rg = _replica_group(mesh)
    n_tiles = T // P

    w = nl.ndarray((P, K1, HH), dtype=o_w.dtype, buffer=nl.sbuf)
    for k in range(K1):
        nisa.dma_copy(dst=w[0:P, k, 0:HH], src=o_w.ap(pattern=[[H, P], [1, HH]], offset=k * P * H + h0),
                      dge_mode=nisa.dge_mode.hwdge)
    gam = nl.ndarray((P, HH), dtype=norm_w.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=gam[0:P, 0:HH], src=norm_w.ap(pattern=[[0, P], [1, HH]], offset=h0))

    h_out = nl.ndarray((T, H), dtype=x.dtype, buffer=nl.shared_hbm, name="prefill_back_h")
    hn_out = nl.ndarray((T, H), dtype=x.dtype, buffer=nl.shared_hbm, name="prefill_back_hn")
    part = nl.ndarray((T, H), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="prefill_back_part")
    for t in nl.sequential_range(n_tiles):
        t0 = t * P
        # the heads' first vd channels times 1 / the FP8 scale, transposed: [vd, tokens]
        aT = nl.ndarray((P, K1, P), dtype=nl.bfloat16, buffer=nl.sbuf)
        for h in range(nq):
            a = nl.ndarray((P, vd), dtype=attn.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=a[0:P, 0:vd], src=attn[h, nl.ds(t0, P), 0:vd])
            ab = nl.ndarray((P, vd), dtype=nl.bfloat16, buffer=nl.sbuf)
            if kv_scale != 1.0:
                nisa.tensor_scalar(dst=ab[0:P, 0:vd], data=a[0:P, 0:vd], op0=nl.multiply, operand0=1.0 / kv_scale)
            else:
                nisa.tensor_copy(dst=ab[0:P, 0:vd], src=a[0:P, 0:vd])
            for c in range(vd // P):
                tp = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.psum)
                nisa.nc_transpose(dst=tp[0:P, 0:P], data=ab[0:P, c * P:(c + 1) * P])
                nisa.tensor_copy(dst=aT[0:P, h * (vd // P) + c, 0:P], src=tp[0:P, 0:P])
        pt = nl.ndarray((P, HH), dtype=nl.bfloat16, buffer=nl.sbuf)
        for c0 in range(0, HH, COLS):
            cs = min(COLS, HH - c0)
            ps = nl.ndarray((P, COLS), dtype=nl.float32, buffer=nl.psum)
            for k in range(K1):
                nisa.nc_matmul(dst=ps[0:P, 0:cs], stationary=aT[0:P, k, 0:P], moving=w[0:P, k, c0:c0 + cs])
            nisa.tensor_copy(dst=pt[0:P, c0:c0 + cs], src=ps[0:P, 0:cs])
        nisa.dma_copy(dst=part[nl.ds(t0, P), h0:h0 + HH], src=pt[0:P, 0:HH])
    red = _hbm_all_reduce(part, rg, n_prg, "prefill_back_reduced")
    for t in nl.sequential_range(n_tiles):
        t0 = t * P
        ht = _add_rows(x, red, h_out, t0, HH, H, h0)
        sq = nl.ndarray((P, HH), dtype=nl.float32, buffer=nl.sbuf)
        ss = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=sq[0:P, 0:HH], op=nl.square, data=ht[0:P, 0:HH], reduce_op=nl.add,
                        reduce_res=ss[0:P, 0:1], reduce_cmd=nisa.reduce_cmd.reset_reduce)
        if n_prg > 1:
            theirs = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.sendrecv(src=ss[0:P, 0:1], dst=theirs[0:P, 0:1], send_to_rank=other, recv_from_rank=other,
                          pipe_id=0)
            nisa.tensor_tensor(dst=ss[0:P, 0:1], data1=ss[0:P, 0:1], data2=theirs[0:P, 0:1], op=nl.add)
        rr = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=rr[0:P, 0:1], op=nl.rsqrt, data=ss[0:P, 0:1], scale=1.0 / H, bias=eps)
        nb = nl.ndarray((P, HH), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=nb[0:P, 0:HH], data=ht[0:P, 0:HH], op0=nl.multiply, operand0=rr[0:P, 0:1])
        hn = nl.ndarray((P, HH), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=hn[0:P, 0:HH], data1=nb[0:P, 0:HH], data2=gam[0:P, 0:HH], op=nl.multiply)
        nisa.dma_copy(dst=hn_out[nl.ds(t0, P), h0:h0 + HH], src=hn[0:P, 0:HH])
    return h_out, hn_out


@nki.jit
def prefill_route(hn, router_wT, expert_bias, ep_rank, positions, top_k=8, route_scale=1.0, n_local=4,
                  block_size=256, num_blocks=1):
    """A chunk's routing and its blockwise expert mapping in one kernel (moe.route, the local experts' columns and
    ops/moe/moe_blockwise.build_blockwise_mapping, which moe_cte reads):

        scores    = sigmoid(hn @ W_r^T)                fp32 (bf16 operands, fp32 accumulate)
        chosen    = the top_k of scores + bias         the bias selects only
        aff       = chosen / (sum chosen + 1e-20) x route_scale, this rank's n_local columns, rounded through bf16
        selected  = aff != 0 on a real token           a padding token (the runtime repeats the last real position)
                                                       routes nowhere
        slot      = first_block[e] x block_size + the token's rank among expert e's tokens
        tokens    = token id at its slots, -1 elsewhere; block_to_expert = each block's expert (the last past the
                    used blocks); conditions = 1 for a used block, then two 0 (the hybrid kernel's trailing reads)

    hn [T, H] bf16 (T a multiple of 128) the normed chunk; router_wT [H, EA] bf16; expert_bias [1, EA] fp32; ep_rank
    [1, 1] int32 (its first local expert is ep_rank x n_local); positions [T, 1] int32. Returns (aff [T x n_local, 1]
    bf16, tokens [num_blocks x block_size] int32, block_to_expert [num_blocks, 1] int32, conditions [num_blocks + 2]
    int32). Both programs of LNC2 compute it; the first writes it."""
    T, H = hn.shape
    EA = router_wT.shape[1]
    E = n_local
    H1 = H // P
    NT = T // P
    NB = num_blocks * block_size
    n_prg = nl.num_programs(axes=0)
    pid = nl.program_id(axis=0)
    shift = block_size.bit_length() - 1  # block_size a power of two

    aff_out = nl.ndarray((T * E, 1), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="prefill_route_aff")
    tok_out = nl.ndarray((NB,), dtype=nl.int32, buffer=nl.shared_hbm, name="prefill_route_tokens")
    b2e_out = nl.ndarray((num_blocks, 1), dtype=nl.int32, buffer=nl.shared_hbm, name="prefill_route_b2e")
    cond_out = nl.ndarray((num_blocks + 2,), dtype=nl.int32, buffer=nl.shared_hbm, name="prefill_route_cond")
    # the slots plus a run of room past them, which the last expert's run's -1 tail writes
    scratch = nl.ndarray((NB + T + P, 1), dtype=nl.int32, buffer=nl.shared_hbm, name="prefill_route_scratch")
    # both programs compute (an idle program fails the compiler's allocation); the first writes
    w0 = n_prg == 1 or pid == 0

    rw = nl.ndarray((P, H1, EA), dtype=router_wT.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=rw[0:P, 0:H1, 0:EA], src=router_wT.ap(pattern=[[EA, P], [P * EA, H1], [1, EA]], offset=0),
                  dge_mode=nisa.dge_mode.hwdge)
    bia = nl.ndarray((P, EA), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=bia[0:P, 0:EA], src=expert_bias.ap(pattern=[[0, P], [1, EA]], offset=0))
    eid = nl.ndarray((P, EA), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(eid[0:P, 0:EA], pattern=[[1, EA]], offset=0, channel_multiplier=0)
    eb_i = nl.ndarray((P, 1), dtype=ep_rank.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=eb_i[0:P, 0:1], src=ep_rank.reshape((1, 1)).ap(pattern=[[0, P], [1, 1]], offset=0))
    eb = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=eb[0:P, 0:1], src=eb_i[0:P, 0:1])
    nisa.tensor_scalar(dst=eb[0:P, 0:1], data=eb[0:P, 0:1], op0=nl.multiply, operand0=float(E))
    rel = nl.ndarray((P, EA), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=rel[0:P, 0:EA], data=eid[0:P, 0:EA], op0=nl.subtract, operand0=eb[0:P, 0:1])
    # tri[k, m] = 1 where k <= m (the inclusive prefix over a tile's tokens, as a matmul); ones for the tile's totals
    kk = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(kk[0:P, 0:P], pattern=[[1, P]], offset=0, channel_multiplier=-1)  # m - k
    tri = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=tri[0:P, 0:P], data=kk[0:P, 0:P], op0=nl.greater_equal, operand0=0.0)
    ones = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.memset(dst=ones[0:P, 0:P], value=1.0)
    part_f = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(part_f[0:P, 0:1], pattern=[[0, 1]], offset=0, channel_multiplier=1)

    # each tile's selected pairs, for the second pass; the tiles run in runtime loops (one copy of the
    # body), the first tile's own copy reading no position before it
    sel_h = nl.ndarray((T, E), dtype=nl.float32, buffer=nl.shared_hbm, name="prefill_route_sel")
    aff2d = aff_out.reshape((T, E))
    carry = nl.ndarray((P, E), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=carry[0:P, 0:E], value=0.0)

    def tile(t0, first_tile: bool):
        # ---- the router over the tile (moe_half's) ----
        xt = nl.ndarray((P, H), dtype=hn.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=xt[0:P, 0:H], src=hn[nl.ds(t0, P), 0:H])
        xT = nl.ndarray((P, H1, P), dtype=nl.bfloat16, buffer=nl.sbuf)
        for h1 in range(H1):
            tp = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.psum)
            nisa.nc_transpose(dst=tp[0:P, 0:P], data=xt[0:P, h1 * P:(h1 + 1) * P])
            nisa.tensor_copy(dst=xT[0:P, h1, 0:P], src=tp[0:P, 0:P])
        lg = nl.ndarray((P, EA), dtype=nl.float32, buffer=nl.psum)
        for h1 in range(H1):
            nisa.nc_matmul(dst=lg[0:P, 0:EA], stationary=xT[0:P, h1, 0:P], moving=rw[0:P, h1, 0:EA])
        scr = nl.ndarray((P, EA), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=scr[0:P, 0:EA], op=nl.sigmoid, data=lg[0:P, 0:EA])
        sb = nl.ndarray((P, EA), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=sb[0:P, 0:EA], data1=scr[0:P, 0:EA], data2=bia[0:P, 0:EA], op=nl.add)
        top8 = nl.ndarray((P, 8), dtype=nl.float32, buffer=nl.sbuf)
        nisa.max8(dst=top8[0:P, 0:8], src=sb[0:P, 0:EA])
        sel = nl.ndarray((P, EA), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=sel[0:P, 0:EA], data=sb[0:P, 0:EA], op0=nl.greater_equal,
                           operand0=top8[0:P, top_k - 1:top_k])
        wsel = nl.ndarray((P, EA), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=wsel[0:P, 0:EA], data1=scr[0:P, 0:EA], data2=sel[0:P, 0:EA], op=nl.multiply)
        wsum = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(wsum[0:P, 0:1], op=nl.add, data=wsel[0:P, 0:EA], axis=1, keepdims=True)
        nisa.tensor_scalar(dst=wsum[0:P, 0:1], data=wsum[0:P, 0:1], op0=nl.add, operand0=1e-20)
        winv = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.reciprocal(dst=winv[0:P, 0:1], data=wsum[0:P, 0:1])
        nisa.tensor_scalar(dst=winv[0:P, 0:1], data=winv[0:P, 0:1], op0=nl.multiply, operand0=route_scale)
        aff = nl.ndarray((P, E), dtype=nl.float32, buffer=nl.sbuf)
        for j in range(E):
            oh = nl.ndarray((P, EA), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=oh[0:P, 0:EA], data=rel[0:P, 0:EA], op0=nl.equal, operand0=float(j))
            pk = nl.ndarray((P, EA), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=pk[0:P, 0:EA], data1=wsel[0:P, 0:EA], data2=oh[0:P, 0:EA], op=nl.multiply)
            nisa.tensor_reduce(aff[0:P, j:j + 1], op=nl.add, data=pk[0:P, 0:EA], axis=1, keepdims=True)
        nisa.tensor_scalar(dst=aff[0:P, 0:E], data=aff[0:P, 0:E], op0=nl.multiply, operand0=winv[0:P, 0:1])
        aff_b = nl.ndarray((P, E), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=aff_b[0:P, 0:E], src=aff[0:P, 0:E])
        if w0:
            nisa.dma_copy(dst=aff2d[nl.ds(t0, P), 0:E], src=aff_b[0:P, 0:E])
        # ---- a real token: the first, or one whose position differs from the token before it ----
        pc = nl.ndarray((P, 1), dtype=positions.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=pc[0:P, 0:1], src=positions[nl.ds(t0, P), 0:1])
        pv = nl.ndarray((P, 1), dtype=positions.dtype, buffer=nl.sbuf)
        if first_tile:
            nisa.memset(dst=pv[0:1, 0:1], value=-1)
            nisa.dma_copy(dst=pv[1:P, 0:1], src=positions[0:P - 1, 0:1])
        else:
            nisa.dma_copy(dst=pv[0:P, 0:1], src=positions[nl.ds(t0 - 1, P), 0:1])
        pcf = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=pcf[0:P, 0:1], src=pc[0:P, 0:1])
        pvf = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=pvf[0:P, 0:1], src=pv[0:P, 0:1])
        real = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=real[0:P, 0:1], data1=pcf[0:P, 0:1], data2=pvf[0:P, 0:1], op=nl.not_equal)
        # ---- selected pairs, and the expert's count so far ----
        ab = nl.ndarray((P, E), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=ab[0:P, 0:E], src=aff_b[0:P, 0:E])
        s_f = nl.ndarray((P, E), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=s_f[0:P, 0:E], data=ab[0:P, 0:E], op0=nl.not_equal, operand0=0.0,
                           op1=nl.multiply, operand1=real[0:P, 0:1])
        if w0:
            nisa.dma_copy(dst=sel_h[nl.ds(t0, P), 0:E], src=s_f[0:P, 0:E])
        s_b = nl.ndarray((P, E), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=s_b[0:P, 0:E], src=s_f[0:P, 0:E])
        tot = nl.ndarray((P, E), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=tot[0:P, 0:E], stationary=ones[0:P, 0:P], moving=s_b[0:P, 0:E])
        nisa.tensor_tensor(dst=carry[0:P, 0:E], data1=carry[0:P, 0:E], data2=tot[0:P, 0:E], op=nl.add)


    tile(0, True)
    for t in nl.sequential_range(1, NT):
        tile(t * P, False)

    # ---- blocks per expert, their first block (every partition holds the same values) ----
    cnt_i = nl.ndarray((P, E), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=cnt_i[0:P, 0:E], src=carry[0:P, 0:E])
    nisa.tensor_scalar(dst=cnt_i[0:P, 0:E], data=cnt_i[0:P, 0:E], op0=nl.add, operand0=block_size - 1)
    nisa.tensor_scalar(dst=cnt_i[0:P, 0:E], data=cnt_i[0:P, 0:E], op0=nl.right_shift, operand0=shift)
    blk = nl.ndarray((P, E), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=blk[0:P, 0:E], src=cnt_i[0:P, 0:E])
    first = nl.ndarray((P, E + 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=first[0:P, 0:1], value=0.0)
    for e in range(E):
        nisa.tensor_tensor(dst=first[0:P, e + 1:e + 2], data1=first[0:P, e:e + 1], data2=blk[0:P, e:e + 1],
                           op=nl.add)
    base = nl.ndarray((P, E), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=base[0:P, 0:E], data=first[0:P, 0:E], op0=nl.multiply, operand0=float(block_size))

    # ---- the slots: each expert's selected tokens packed in order, then laid at its first slot ----
    # Within a tile, an expert's selected tokens take consecutive places after the expert's count before the tile, so a
    # one-hot matmul packs every expert's in one pass (place j = the token's inclusive prefix in the tile - 1) and one
    # contiguous 128-place write an expert lands them, -1 past them, in the expert's own staging run (-1 to start):
    # the next tile's window overwrites the -1 tail, and the experts' writes, to separate tensors, need no order. The
    # runs then land at their experts' first slots in expert order, each run's -1 tail under the next expert's run, the
    # scratch a run longer than the slots. A scattered write a pair took 5.5 ms of a 14 ms layer at an 8K chunk, an
    # (expert, tile) window loop 1.9 ms.
    R = T // P + 1
    neg = nl.ndarray((P, R), dtype=nl.int32, buffer=nl.sbuf)
    nisa.memset(dst=neg[0:P, 0:R], value=-1)
    runs = [nl.ndarray((T + P, 1), dtype=nl.int32, buffer=nl.shared_hbm, name=f"prefill_route_run{e}")
            for e in range(E)]
    if w0:
        for e in range(E):
            nisa.dma_copy(dst=runs[e].ap(pattern=[[R, P], [1, R]], offset=0), src=neg[0:P, 0:R])
        io = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.sbuf)
        nisa.iota(io[0:P, 0:P], pattern=[[1, P]], offset=0, channel_multiplier=0)  # the window place, every row
        pp_f = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.iota(pp_f[0:P, 0:1], pattern=[[0, 1]], offset=1, channel_multiplier=1)  # 1 + the row
        pp1 = nl.ndarray((P, 1), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=pp1[0:P, 0:1], src=pp_f[0:P, 0:1])
        before = nl.ndarray((1, E), dtype=nl.float32, buffer=nl.sbuf)  # each expert's selected pairs so far
        nisa.memset(dst=before[0:1, 0:E], value=0.0)
        t0f = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)  # the tile's first token id
        nisa.memset(dst=t0f[0:1, 0:1], value=0.0)
        for t in nl.sequential_range(NT):
            t0 = t * P
            st_t = nl.ndarray((P, E), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=st_t[0:P, 0:E], src=sel_h[nl.ds(t0, P), 0:E])
            st_b = nl.ndarray((P, E), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=st_b[0:P, 0:E], src=st_t[0:P, 0:E])
            pre = nl.ndarray((P, E), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=pre[0:P, 0:E], stationary=tri[0:P, 0:P], moving=st_b[0:P, 0:E])
            # j = selected x inclusive prefix - 1: the window place, -1 for an unselected pair
            jj = nl.ndarray((P, E), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=jj[0:P, 0:E], data1=pre[0:P, 0:E], data2=st_t[0:P, 0:E], op=nl.multiply)
            nisa.tensor_scalar(dst=jj[0:P, 0:E], data=jj[0:P, 0:E], op0=nl.add, operand0=-1.0)
            oh = nl.ndarray((P, E * P), dtype=nl.bfloat16, buffer=nl.sbuf)
            for e in range(E):
                nisa.tensor_scalar(dst=oh[0:P, e * P:(e + 1) * P], data=io[0:P, 0:P], op0=nl.equal,
                                   operand0=jj[0:P, e:e + 1])
            # v[e, j] = 1 + the row whose place is j, 0 for an empty place, four experts a matmul (a moving tile is at
            # most 512 wide); then the token id (t0 + row) at a filled place, -1 at an empty one
            val_i = nl.ndarray((1, E * P), dtype=nl.int32, buffer=nl.sbuf)
            for g0 in range(0, E, 4):
                w = min(4, E - g0) * P
                c0 = g0 * P
                v = nl.ndarray((1, 4 * P), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=v[0:1, 0:w], stationary=pp1[0:P, 0:1], moving=oh[0:P, c0:c0 + w])
                mk = nl.ndarray((1, 4 * P), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=mk[0:1, 0:w], data=v[0:1, 0:w], op0=nl.greater, operand0=0.5,
                                   op1=nl.multiply, operand1=t0f[0:1, 0:1])
                val = nl.ndarray((1, 4 * P), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=val[0:1, 0:w], data1=mk[0:1, 0:w], data2=v[0:1, 0:w], op=nl.add)
                nisa.tensor_scalar(dst=val[0:1, 0:w], data=val[0:1, 0:w], op0=nl.add, operand0=-1.0)
                nisa.tensor_copy(dst=val_i[0:1, c0:c0 + w], src=val[0:1, 0:w])
            cnt = nl.ndarray((1, E), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=cnt[0:1, 0:E], stationary=ones[0:P, 0:1], moving=st_b[0:P, 0:E])
            w_i = nl.ndarray((1, E), dtype=nl.int32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=w_i[0:1, 0:E], src=before[0:1, 0:E])
            for e in range(E):
                nisa.dma_copy(dst=runs[e].ap(pattern=[[1, 1], [1, P]], offset=0, vector_offset=w_i[0:1, e:e + 1],
                                             indirect_dim=0), src=val_i[0:1, e * P:(e + 1) * P])
            nisa.tensor_tensor(dst=before[0:1, 0:E], data1=before[0:1, 0:E], data2=cnt[0:1, 0:E], op=nl.add)
            nisa.tensor_scalar(dst=t0f[0:1, 0:1], data=t0f[0:1, 0:1], op0=nl.add, operand0=float(P))
        # -1 at every slot (the blocks past the used ones take no run), then each run at its expert's first slot, in
        # expert order
        RS = NB // P
        negs = nl.ndarray((P, RS), dtype=nl.int32, buffer=nl.sbuf)
        nisa.memset(dst=negs[0:P, 0:RS], value=-1)
        nisa.dma_copy(dst=scratch.ap(pattern=[[RS, P], [1, RS]], offset=0), src=negs[0:P, 0:RS])
        base_i = nl.ndarray((1, E), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=base_i[0:1, 0:E], src=base[0:1, 0:E])
        for e in range(E):
            run_sb = nl.ndarray((1, T + P), dtype=nl.int32, buffer=nl.sbuf)
            nisa.dma_copy(dst=run_sb[0:1, 0:T + P], src=runs[e].ap(pattern=[[T + P, 1], [1, T + P]], offset=0))
            nisa.dma_copy(dst=scratch.ap(pattern=[[1, 1], [1, T + P]], offset=0, vector_offset=base_i[0:1, e:e + 1],
                                         indirect_dim=0), src=run_sb[0:1, 0:T + P])
    if w0:
        nisa.dma_copy(dst=tok_out.reshape((NB, 1)), src=scratch.ap(pattern=[[1, NB], [1, 1]], offset=0))

    # ---- each block's expert (the last expert past the used blocks) and whether it is used ----
    nf = nl.ndarray((1, num_blocks + 2), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(nf[0:1, 0:num_blocks + 2], pattern=[[1, num_blocks + 2]], offset=0, channel_multiplier=0)
    be = nl.ndarray((1, num_blocks + 2), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=be[0:1, 0:num_blocks + 2], value=0.0)
    for e in range(1, E):
        ge = nl.ndarray((1, num_blocks + 2), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=ge[0:1, 0:num_blocks + 2], data=nf[0:1, 0:num_blocks + 2], op0=nl.greater_equal,
                           operand0=first[0:1, e:e + 1])
        nisa.tensor_tensor(dst=be[0:1, 0:num_blocks + 2], data1=be[0:1, 0:num_blocks + 2],
                           data2=ge[0:1, 0:num_blocks + 2], op=nl.add)
    be_i = nl.ndarray((1, num_blocks + 2), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=be_i[0:1, 0:num_blocks + 2], src=be[0:1, 0:num_blocks + 2])
    if w0:
        nisa.dma_copy(dst=b2e_out.reshape((1, num_blocks)), src=be_i[0:1, 0:num_blocks])
    used = nl.ndarray((1, num_blocks + 2), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=used[0:1, 0:num_blocks + 2], data=nf[0:1, 0:num_blocks + 2], op0=nl.less,
                       operand0=first[0:1, E:E + 1])
    used_i = nl.ndarray((1, num_blocks + 2), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=used_i[0:1, 0:num_blocks + 2], src=used[0:1, 0:num_blocks + 2])
    if w0:
        nisa.dma_copy(dst=cond_out.reshape((1, num_blocks + 2)), src=used_i[0:1, 0:num_blocks + 2])
    return aff_out, tok_out, b2e_out, cond_out


def _hbm_all_reduce(part, rg, n_prg: int, name: str):
    """The sum of ``part`` (shared HBM, both programs' halves written) over the group, into shared HBM: one collective,
    fenced by a core barrier before (both halves landed) and after (the sum landed before either program reads it), as
    kernels/layer._all_reduce (unfenced in-kernel sums came back wrong on trn2). The sum goes to a buffer of its own:
    an all-reduce into the kernel's outputs hung the 4096- and 8192-token prefill graphs."""
    from nki import collectives as nccl

    if n_prg > 1:
        nisa.core_barrier(data=part, cores=(0, 1))
    reduced = nl.ndarray(part.shape, dtype=part.dtype, buffer=nl.shared_hbm, name=name)
    nccl.all_reduce(dsts=[reduced], srcs=[part], op=nl.add, replica_group=rg)
    if n_prg > 1:
        nisa.core_barrier(data=reduced, cores=(0, 1))
    return reduced


def _add_rows(x, red, h_out, t0: int, HH: int, H: int, h0: int):
    """h_out[t0 .. t0 + 128, h0 .. h0 + HH] = x + red there; returns the tile [128, HH] bf16 in SBUF."""
    xt = nl.ndarray((P, HH), dtype=x.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=xt[0:P, 0:HH], src=x[nl.ds(t0, P), h0:h0 + HH])
    rt = nl.ndarray((P, HH), dtype=red.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=rt[0:P, 0:HH], src=red[nl.ds(t0, P), h0:h0 + HH])
    ht = nl.ndarray((P, HH), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=ht[0:P, 0:HH], data1=xt[0:P, 0:HH], data2=rt[0:P, 0:HH], op=nl.add)
    nisa.dma_copy(dst=h_out[nl.ds(t0, P), h0:h0 + HH], src=ht[0:P, 0:HH])
    return ht


@nki.jit
def prefill_residual(x, partial, mesh):
    """h = x + all_reduce(partial) over ``mesh``: a chunk's residual add after this rank's partial MLP output (the MoE
    experts'). x / partial [T, H] bf16, T a multiple of 128. One all-reduce of the partial (_hbm_all_reduce), then the
    two programs of LNC2 add one half of H each."""
    from sglang_neuron.models.mimo_v2.kernels.head import _replica_group

    T, H = x.shape
    n_prg = nl.num_programs(axes=0)
    pid = nl.program_id(axis=0)
    HH = H // n_prg
    h0 = pid * HH
    rg = _replica_group(mesh)
    h_out = nl.ndarray((T, H), dtype=x.dtype, buffer=nl.shared_hbm, name="prefill_residual_h")
    red = _hbm_all_reduce(partial, rg, n_prg, "prefill_residual_reduced")
    for t in nl.sequential_range(T // P):
        _add_rows(x, red, h_out, t * P, HH, H, h0)
    return h_out


@nki.jit
def prefill_mlp(hn, x, gate_w, up_w, down_w, mesh):
    """h = x + all_reduce((silu(hn @ gate_w) x (hn @ up_w)) @ down_w): a dense layer's MLP over a chunk (model.DenseMLP,
    each product rounded to bf16 as the torch path's), its all-reduce and the residual add, in one kernel. hn / x [T, H]
    bf16 (the normed and the residual stream, T a multiple of 128); gate_w / up_w [H, I] and down_w [I, H] bf16, this
    rank's intermediate shard resident in SBUF. Both programs of LNC2 compute the gated activation; each projects it
    onto its half of H into the chunk's partial in shared HBM, all-reduced once (_hbm_all_reduce), and adds that
    half."""
    from sglang_neuron.models.mimo_v2.kernels.head import _replica_group

    T, H = x.shape
    I = gate_w.shape[1]  # noqa: E741
    H1, I1 = H // P, I // P
    n_prg = nl.num_programs(axes=0)
    pid = nl.program_id(axis=0)
    HH = H // n_prg
    h0 = pid * HH
    rg = _replica_group(mesh)
    gw = nl.ndarray((P, H1, I), dtype=gate_w.dtype, buffer=nl.sbuf)
    uw = nl.ndarray((P, H1, I), dtype=up_w.dtype, buffer=nl.sbuf)
    for h1 in range(H1):
        nisa.dma_copy(dst=gw[0:P, h1, 0:I], src=gate_w.ap(pattern=[[I, P], [1, I]], offset=h1 * P * I),
                      dge_mode=nisa.dge_mode.hwdge)
        nisa.dma_copy(dst=uw[0:P, h1, 0:I], src=up_w.ap(pattern=[[I, P], [1, I]], offset=h1 * P * I),
                      dge_mode=nisa.dge_mode.hwdge)
    dw = nl.ndarray((P, I1, HH), dtype=down_w.dtype, buffer=nl.sbuf)
    for i1 in range(I1):
        nisa.dma_copy(dst=dw[0:P, i1, 0:HH], src=down_w.ap(pattern=[[H, P], [1, HH]], offset=i1 * P * H + h0),
                      dge_mode=nisa.dge_mode.hwdge)
    h_out = nl.ndarray((T, H), dtype=x.dtype, buffer=nl.shared_hbm, name="prefill_mlp_h")
    part = nl.ndarray((T, H), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="prefill_mlp_part")
    for t in nl.sequential_range(T // P):
        t0 = t * P
        xt = nl.ndarray((P, H), dtype=hn.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=xt[0:P, 0:H], src=hn[nl.ds(t0, P), 0:H])
        xT = nl.ndarray((P, H1, P), dtype=nl.bfloat16, buffer=nl.sbuf)
        for h1 in range(H1):
            tp = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.psum)
            nisa.nc_transpose(dst=tp[0:P, 0:P], data=xt[0:P, h1 * P:(h1 + 1) * P])
            nisa.tensor_copy(dst=xT[0:P, h1, 0:P], src=tp[0:P, 0:P])
        act = nl.ndarray((P, I), dtype=nl.bfloat16, buffer=nl.sbuf)
        for c0 in range(0, I, COLS):
            cs = min(COLS, I - c0)
            pg = nl.ndarray((P, COLS), dtype=nl.float32, buffer=nl.psum)
            pu = nl.ndarray((P, COLS), dtype=nl.float32, buffer=nl.psum)
            for h1 in range(H1):
                nisa.nc_matmul(dst=pg[0:P, 0:cs], stationary=xT[0:P, h1, 0:P], moving=gw[0:P, h1, c0:c0 + cs])
                nisa.nc_matmul(dst=pu[0:P, 0:cs], stationary=xT[0:P, h1, 0:P], moving=uw[0:P, h1, c0:c0 + cs])
            gb = nl.ndarray((P, COLS), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=gb[0:P, 0:cs], src=pg[0:P, 0:cs])
            ub = nl.ndarray((P, COLS), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=ub[0:P, 0:cs], src=pu[0:P, 0:cs])
            sg = nl.ndarray((P, COLS), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.activation(dst=sg[0:P, 0:cs], op=nl.silu, data=gb[0:P, 0:cs])
            nisa.tensor_tensor(dst=act[0:P, c0:c0 + cs], data1=sg[0:P, 0:cs], data2=ub[0:P, 0:cs], op=nl.multiply)
        aT = nl.ndarray((P, I1, P), dtype=nl.bfloat16, buffer=nl.sbuf)
        for i1 in range(I1):
            tp = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.psum)
            nisa.nc_transpose(dst=tp[0:P, 0:P], data=act[0:P, i1 * P:(i1 + 1) * P])
            nisa.tensor_copy(dst=aT[0:P, i1, 0:P], src=tp[0:P, 0:P])
        pt = nl.ndarray((P, HH), dtype=nl.bfloat16, buffer=nl.sbuf)
        for c0 in range(0, HH, COLS):
            cs = min(COLS, HH - c0)
            ps = nl.ndarray((P, COLS), dtype=nl.float32, buffer=nl.psum)
            for i1 in range(I1):
                nisa.nc_matmul(dst=ps[0:P, 0:cs], stationary=aT[0:P, i1, 0:P], moving=dw[0:P, i1, c0:c0 + cs])
            nisa.tensor_copy(dst=pt[0:P, c0:c0 + cs], src=ps[0:P, 0:cs])
        nisa.dma_copy(dst=part[nl.ds(t0, P), h0:h0 + HH], src=pt[0:P, 0:HH])
    red = _hbm_all_reduce(part, rg, n_prg, "prefill_mlp_reduced")
    for t in nl.sequential_range(T // P):
        _add_rows(x, red, h_out, t * P, HH, H, h0)
    return h_out


TWO_PI = 6.283185307179586
TWO_PI_HI = 6.25
TWO_PI_LO = TWO_PI - TWO_PI_HI
HALF_PI = 1.5707963267948966
PI = 3.141592653589793


def _rope_rows(pos_f, inv, F: int, half: int, cos_out, sin_out, t0: int):
    """cos / sin [128, half] bf16 of the tile's positions ``pos_f`` [128, 1] fp32 into rows t0 .. t0 + 128 of the
    [T, half] tables: rope.RotaryTables' channels (its F frequencies ``inv`` [128, F] fp32, then cos 1 and sin 0), the
    angle reduced Cody-Waite style and cos = sin(x + pi / 2) folded back into [-pi, pi] (kernels/step._rope)."""
    ang = nl.ndarray((P, F), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=ang[0:P, 0:F], data=inv[0:P, 0:F], op0=nl.multiply, operand0=pos_f[0:P, 0:1])
    k_f = nl.ndarray((P, F), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=k_f[0:P, 0:F], data=ang[0:P, 0:F], op0=nl.multiply, operand0=1.0 / TWO_PI)
    k_i = nl.ndarray((P, F), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=k_i[0:P, 0:F], src=k_f[0:P, 0:F])
    nisa.tensor_copy(dst=k_f[0:P, 0:F], src=k_i[0:P, 0:F])
    red = nl.ndarray((P, F), dtype=nl.float32, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(dst=red[0:P, 0:F], data=k_f[0:P, 0:F], op0=nl.multiply, operand0=-TWO_PI_HI, op1=nl.add,
                              operand1=ang[0:P, 0:F])
    nisa.scalar_tensor_tensor(dst=red[0:P, 0:F], data=k_f[0:P, 0:F], op0=nl.multiply, operand0=-TWO_PI_LO, op1=nl.add,
                              operand1=red[0:P, 0:F])
    for out, shift, rest in ((cos_out, HALF_PI, 1.0), (sin_out, 0.0, 0.0)):
        full = nl.ndarray((P, half), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.memset(dst=full[0:P, 0:half], value=rest)
        arg = nl.ndarray((P, F), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=arg[0:P, 0:F], data=red[0:P, 0:F], op0=nl.add, operand0=shift)
        if shift:
            over = nl.ndarray((P, F), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=over[0:P, 0:F], data=arg[0:P, 0:F], op0=nl.greater, operand0=PI)
            nisa.scalar_tensor_tensor(dst=arg[0:P, 0:F], data=over[0:P, 0:F], op0=nl.multiply, operand0=-TWO_PI,
                                      op1=nl.add, operand1=arg[0:P, 0:F])
        val = nl.ndarray((P, F), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=val[0:P, 0:F], op=nl.sin, data=arg[0:P, 0:F])
        nisa.tensor_copy(dst=full[0:P, 0:F], src=val[0:P, 0:F])
        nisa.dma_copy(dst=out.ap(pattern=[[half, P], [1, half]], offset=t0 * half), src=full[0:P, 0:half])


@nki.jit
def prefill_inputs(ids, positions, weight, rank, mesh, full_inv_freq, swa_inv_freq, half=128):
    """A prefill chunk's inputs in one kernel: what embed_tokens (the vocabulary-sharded lookup summed over the ranks)
    and rope.RotaryTables (both kinds' cos / sin) did in XLA.

    ids / positions [T, 1] int32 (T a multiple of 128); weight [Vr, H] bf16 this rank's vocabulary rows; rank [1] int32
    its place in ``mesh``; the inverse frequencies [1, F] fp32. Returns (the embeddings [T, H] bf16, full cos, full sin,
    sliding cos, sliding sin [T, half] bf16). The embedding rows are gathered into the chunk's partial in shared HBM
    (zero where another rank holds the id) and all-reduced once (_hbm_all_reduce); the two programs of LNC2 take one
    half of H each, and the first writes the tables."""
    from sglang_neuron.models.mimo_v2.kernels.head import _replica_group, _start

    T = ids.shape[0]
    Vr, H = weight.shape
    n_prg = nl.num_programs(axes=0)
    pid = nl.program_id(axis=0)
    HH = H // n_prg
    h0 = pid * HH
    rg = _replica_group(mesh)
    st = _start(rank, P, Vr)
    part = nl.ndarray((T, H), dtype=weight.dtype, buffer=nl.shared_hbm, name="prefill_inputs_part")
    tables = []
    for n in ("full_cos", "full_sin", "swa_cos", "swa_sin"):  # NKI's tracer takes no list comprehension
        tables.append(nl.ndarray((T, half), dtype=nl.bfloat16, buffer=nl.shared_hbm, name=f"prefill_inputs_{n}"))
    invs = []
    for src in (full_inv_freq, swa_inv_freq):
        F = src.shape[1]
        inv = nl.ndarray((P, F), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=inv[0:P, 0:F], src=src.ap(pattern=[[0, P], [1, F]], offset=0))
        invs.append((inv, F))
    for t in range(T // P):
        t0 = t * P
        ids_i = nl.ndarray((P, 1), dtype=ids.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=ids_i[0:P, 0:1], src=ids.ap(pattern=[[1, P], [1, 1]], offset=t0))
        ids_f = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=ids_f[0:P, 0:1], src=ids_i[0:P, 0:1])
        local = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=local[0:P, 0:1], data1=ids_f[0:P, 0:1], data2=st[0:P, 0:1], op=nl.subtract)
        lo = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=lo[0:P, 0:1], data=local[0:P, 0:1], op0=nl.greater_equal, operand0=0.0)
        hi = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=hi[0:P, 0:1], data=local[0:P, 0:1], op0=nl.less, operand0=float(Vr))
        ok = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=ok[0:P, 0:1], data1=lo[0:P, 0:1], data2=hi[0:P, 0:1], op=nl.multiply)
        # (local - Vr) x ok + Vr: the local row where the shard holds the id, Vr (skipped) elsewhere
        row_f = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=row_f[0:P, 0:1], data=local[0:P, 0:1], op0=nl.subtract, operand0=float(Vr))
        nisa.tensor_tensor(dst=row_f[0:P, 0:1], data1=row_f[0:P, 0:1], data2=ok[0:P, 0:1], op=nl.multiply)
        nisa.tensor_scalar(dst=row_f[0:P, 0:1], data=row_f[0:P, 0:1], op0=nl.add, operand0=float(Vr))
        row = nl.ndarray((P, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=row[0:P, 0:1], src=row_f[0:P, 0:1])
        rows = nl.ndarray((P, HH), dtype=weight.dtype, buffer=nl.sbuf)
        nisa.memset(dst=rows[0:P, 0:HH], value=0.0)
        nisa.dma_copy(dst=rows[0:P, 0:HH], src=weight.ap(pattern=[[H, P], [1, HH]], offset=h0,
                                                          vector_offset=row[0:P, 0:1], indirect_dim=0),
                      oob_mode=oob_mode.skip)
        nisa.dma_copy(dst=part.ap(pattern=[[H, P], [1, HH]], offset=t0 * H + h0), src=rows[0:P, 0:HH])
        if n_prg == 1 or pid == 0:
            pc = nl.ndarray((P, 1), dtype=positions.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=pc[0:P, 0:1], src=positions.ap(pattern=[[1, P], [1, 1]], offset=t0))
            pos_f = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=pos_f[0:P, 0:1], src=pc[0:P, 0:1])
            for k, (inv, F) in enumerate(invs):
                _rope_rows(pos_f, inv, F, half, tables[2 * k], tables[2 * k + 1], t0)
    emb = _hbm_all_reduce(part, rg, n_prg, "prefill_inputs_embed")
    return emb, tables[0], tables[1], tables[2], tables[3]


def _norm_rows(src, w, H: int, eps):
    """RMSNorm of a [128, H] bf16 tile with the weight tile ``w`` [128, H] (fp32 math, bf16 result, times the
    weight)."""
    sq = nl.ndarray((P, H), dtype=nl.float32, buffer=nl.sbuf)
    ss = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=sq[0:P, 0:H], op=nl.square, data=src[0:P, 0:H], reduce_op=nl.add, reduce_res=ss[0:P, 0:1],
                    reduce_cmd=nisa.reduce_cmd.reset_reduce)
    rr = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=rr[0:P, 0:1], op=nl.rsqrt, data=ss[0:P, 0:1], scale=1.0 / H, bias=eps)
    nb = nl.ndarray((P, H), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=nb[0:P, 0:H], data=src[0:P, 0:H], op0=nl.multiply, operand0=rr[0:P, 0:1])
    out = nl.ndarray((P, H), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=out[0:P, 0:H], data1=nb[0:P, 0:H], data2=w[0:P, 0:H], op=nl.multiply)
    return out


@nki.jit
def prefill_mtp_project(ids, at, token, prev, emb_w, enorm_w, hnorm_w, eh_wT, rank, mesh, eps=1e-6):
    """An MTP layer's prefill input in one kernel (model.MiMoV2Graph._mtp_prefill's mtp.rotate and mtp.MiMoMtp.project
    over a chunk):

        next  = rotate(ids, at, token)          ids one position ahead before ``at``, ``token`` at it, kept after
        x     = eh_proj([enorm(embed(next)) | hnorm(prev)])

    ids [T, 1] int32 (T a multiple of 128) the tokens the layer before read; at / token [1, 1] int32 the chunk's last
    real row and the token drafted there; prev [T, H] bf16 the target's states before the final norm; emb_w [Vr, H]
    bf16 this rank's vocabulary rows; enorm_w / hnorm_w [1, H] bf16; eh_wT [2H, H / ranks] bf16 this rank's eh_proj
    columns; rank [1] int32. Returns (x [T, H] bf16, next [T, 1] int32). The embedding rows are all-reduced and the
    output columns all-gathered once each, through shared HBM between core barriers."""
    from nki import collectives as nccl

    from sglang_neuron.models.mimo_v2.kernels.head import _replica_group, _start

    T = ids.shape[0]
    Vr, H = emb_w.shape
    OR = eh_wT.shape[1]
    H1 = H // P
    NT = T // P
    n_prg = nl.num_programs(axes=0)
    pid = nl.program_id(axis=0)
    HH = H // n_prg
    h0 = pid * HH
    rg = _replica_group(mesh)
    st = _start(rank, P, Vr)
    next_out = nl.ndarray((T, 1), dtype=nl.int32, buffer=nl.shared_hbm, name="prefill_mtp_next")
    part = nl.ndarray((T, H), dtype=emb_w.dtype, buffer=nl.shared_hbm, name="prefill_mtp_part")
    proj = nl.ndarray((T, OR), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="prefill_mtp_proj")
    at_i = nl.ndarray((P, 1), dtype=at.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=at_i[0:P, 0:1], src=at.reshape((1, 1)).ap(pattern=[[0, P], [1, 1]], offset=0))
    at_f = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=at_f[0:P, 0:1], src=at_i[0:P, 0:1])
    tk_i = nl.ndarray((P, 1), dtype=token.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=tk_i[0:P, 0:1], src=token.reshape((1, 1)).ap(pattern=[[0, P], [1, 1]], offset=0))
    tk_f = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=tk_f[0:P, 0:1], src=tk_i[0:P, 0:1])
    part_f = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(part_f[0:P, 0:1], pattern=[[0, 1]], offset=0, channel_multiplier=1)

    # ---- the rotated ids and this program's half of their embedding rows (a runtime loop; the last tile, whose
    # ahead token is its own last, its own copy) ----
    jrun = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)  # each row's place in the chunk
    nisa.tensor_copy(dst=jrun[0:P, 0:1], src=part_f[0:P, 0:1])

    def ids_tile(t0, last: bool):
        cur = nl.ndarray((P, 1), dtype=ids.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=cur[0:P, 0:1], src=ids[nl.ds(t0, P), 0:1])
        ahead = nl.ndarray((P, 1), dtype=ids.dtype, buffer=nl.sbuf)
        if not last:
            nisa.dma_copy(dst=ahead[0:P, 0:1], src=ids[nl.ds(t0 + 1, P), 0:1])
        else:
            nisa.dma_copy(dst=ahead[0:P - 1, 0:1], src=ids[T - P + 1:T, 0:1])
            nisa.dma_copy(dst=ahead[P - 1:P, 0:1], src=ids[T - 1:T, 0:1])
        cf = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=cf[0:P, 0:1], src=cur[0:P, 0:1])
        af = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=af[0:P, 0:1], src=ahead[0:P, 0:1])
        j = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=j[0:P, 0:1], src=jrun[0:P, 0:1])
        before = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=before[0:P, 0:1], data1=j[0:P, 0:1], data2=at_f[0:P, 0:1], op=nl.less)
        on = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=on[0:P, 0:1], data1=j[0:P, 0:1], data2=at_f[0:P, 0:1], op=nl.equal)
        # next = cur + before x (ahead - cur) + on x (token - cur)
        d = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=d[0:P, 0:1], data1=af[0:P, 0:1], data2=cf[0:P, 0:1], op=nl.subtract)
        nisa.tensor_tensor(dst=d[0:P, 0:1], data1=d[0:P, 0:1], data2=before[0:P, 0:1], op=nl.multiply)
        e = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=e[0:P, 0:1], data1=tk_f[0:P, 0:1], data2=cf[0:P, 0:1], op=nl.subtract)
        nisa.tensor_tensor(dst=e[0:P, 0:1], data1=e[0:P, 0:1], data2=on[0:P, 0:1], op=nl.multiply)
        nf = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=nf[0:P, 0:1], data1=cf[0:P, 0:1], data2=d[0:P, 0:1], op=nl.add)
        nisa.tensor_tensor(dst=nf[0:P, 0:1], data1=nf[0:P, 0:1], data2=e[0:P, 0:1], op=nl.add)
        ni = nl.ndarray((P, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=ni[0:P, 0:1], src=nf[0:P, 0:1])
        if n_prg == 1 or pid == 0:
            nisa.dma_copy(dst=next_out[nl.ds(t0, P), 0:1], src=ni[0:P, 0:1])
        local = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=local[0:P, 0:1], data1=nf[0:P, 0:1], data2=st[0:P, 0:1], op=nl.subtract)
        lo = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=lo[0:P, 0:1], data=local[0:P, 0:1], op0=nl.greater_equal, operand0=0.0)
        hi = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=hi[0:P, 0:1], data=local[0:P, 0:1], op0=nl.less, operand0=float(Vr))
        nisa.tensor_tensor(dst=lo[0:P, 0:1], data1=lo[0:P, 0:1], data2=hi[0:P, 0:1], op=nl.multiply)
        row_f = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=row_f[0:P, 0:1], data=local[0:P, 0:1], op0=nl.subtract, operand0=float(Vr))
        nisa.tensor_tensor(dst=row_f[0:P, 0:1], data1=row_f[0:P, 0:1], data2=lo[0:P, 0:1], op=nl.multiply)
        nisa.tensor_scalar(dst=row_f[0:P, 0:1], data=row_f[0:P, 0:1], op0=nl.add, operand0=float(Vr))
        row = nl.ndarray((P, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=row[0:P, 0:1], src=row_f[0:P, 0:1])
        rows = nl.ndarray((P, HH), dtype=emb_w.dtype, buffer=nl.sbuf)
        nisa.memset(dst=rows[0:P, 0:HH], value=0.0)
        nisa.dma_copy(dst=rows[0:P, 0:HH], src=emb_w.ap(pattern=[[H, P], [1, HH]], offset=h0,
                                                         vector_offset=row[0:P, 0:1], indirect_dim=0),
                      oob_mode=oob_mode.skip)
        nisa.dma_copy(dst=part[nl.ds(t0, P), h0:h0 + HH], src=rows[0:P, 0:HH])
        nisa.tensor_scalar(dst=jrun[0:P, 0:1], data=jrun[0:P, 0:1], op0=nl.add, operand0=float(P))

    for t in nl.sequential_range(NT - 1):
        ids_tile(t * P, False)
    ids_tile((NT - 1) * P, True)
    emb = _hbm_all_reduce(part, rg, n_prg, "prefill_mtp_embed")

    # ---- both norms and this rank's eh_proj columns, the tiles alternating between the programs ----
    ew = nl.ndarray((P, H), dtype=enorm_w.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=ew[0:P, 0:H], src=enorm_w.ap(pattern=[[0, P], [1, H]], offset=0))
    hw = nl.ndarray((P, H), dtype=hnorm_w.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=hw[0:P, 0:H], src=hnorm_w.ap(pattern=[[0, P], [1, H]], offset=0))
    wt = nl.ndarray((P, 2 * H1, OR), dtype=eh_wT.dtype, buffer=nl.sbuf)
    for k in range(2 * H1):
        nisa.dma_copy(dst=wt[0:P, k, 0:OR], src=eh_wT.ap(pattern=[[OR, P], [1, OR]], offset=k * P * OR),
                      dge_mode=nisa.dge_mode.hwdge)
    split = n_prg > 1 and NT >= n_prg and NT % n_prg == 0
    for i in nl.sequential_range(NT // n_prg if split else NT):
        t0 = (i * n_prg + pid) * P if split else i * P
        et = nl.ndarray((P, H), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=et[0:P, 0:H], src=emb[nl.ds(t0, P), 0:H])
        pv = nl.ndarray((P, H), dtype=prev.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=pv[0:P, 0:H], src=prev[nl.ds(t0, P), 0:H])
        xT = nl.ndarray((P, 2 * H1, P), dtype=nl.bfloat16, buffer=nl.sbuf)
        for part_i, (src, w) in enumerate(((et, ew), (pv, hw))):
            nrm = _norm_rows(src, w, H, eps)
            for h1 in range(H1):
                tp = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.psum)
                nisa.nc_transpose(dst=tp[0:P, 0:P], data=nrm[0:P, h1 * P:(h1 + 1) * P])
                nisa.tensor_copy(dst=xT[0:P, part_i * H1 + h1, 0:P], src=tp[0:P, 0:P])
        pb = nl.ndarray((P, OR), dtype=nl.bfloat16, buffer=nl.sbuf)
        for c0 in range(0, OR, COLS):
            cs = min(COLS, OR - c0)
            ps = nl.ndarray((P, COLS), dtype=nl.float32, buffer=nl.psum)
            for k in range(2 * H1):
                nisa.nc_matmul(dst=ps[0:P, 0:cs], stationary=xT[0:P, k, 0:P], moving=wt[0:P, k, c0:c0 + cs])
            nisa.tensor_copy(dst=pb[0:P, c0:c0 + cs], src=ps[0:P, 0:cs])
        nisa.dma_copy(dst=proj[nl.ds(t0, P), 0:OR], src=pb[0:P, 0:OR])
    if n_prg > 1:
        nisa.core_barrier(data=proj, cores=(0, 1))
    # an HBM all-gather stacks along dim 0 only: [ranks x T, OR], then rank r's rows to columns r x OR of [T, H]
    ranks = H // OR
    stacked = nl.ndarray((ranks * T, OR), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="prefill_mtp_stacked")
    nccl.all_gather(srcs=[proj], dsts=[stacked], replica_group=rg, collective_dim=0)
    if n_prg > 1:
        nisa.core_barrier(data=stacked, cores=(0, 1))
    out = nl.ndarray((T, H), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="prefill_mtp_x")
    for r in range(ranks):
        if n_prg > 1 and r % n_prg != pid:
            continue
        nisa.dma_copy(dst=out[0:T, r * OR:(r + 1) * OR], src=stacked[r * T:(r + 1) * T, 0:OR])
    if n_prg > 1:
        nisa.core_barrier(data=out, cores=(0, 1))
    return out, next_out


@nki.jit
def prefill_pack(k_cache, v_cache, table, cached, n_cols=1, window=0):
    """The pages a prefill chunk's segmented attention reads from token-major FP8 caches, in the layout it takes FP8 K
    in, in one kernel (attention.pack_prior for one row):

        first  = max(cached - window + 1, 0) / page where ``window`` (a sliding layer's), else 0
        cols   = min(first + i, width - 1), i < n_cols            each the row's table column
        page   = max(table[0, col], 0)
        K      = k_cache[page] packed [page / 2, d, 2]             token 2j + e at [j, :, e]
        V      = v_cache[page]
        table' = column j's place among the gathered pages where j - first is in [0, n_cols), 0 elsewhere; -1 where the
                 row's table had none

    k_cache / v_cache [blocks, 1, page, d] FP8; table [1, width] int32; cached [1, 1] int32 the chunk's prior length
    (page a power of two). Returns (K [n_cols, 1, page / 2, d, 2], V [n_cols, 1, page, d], table' [1, width] int32). The
    pages sit on the partitions, 128 a group, each program of LNC2 gathering one half of each group."""
    blocks, kv_heads, page, d = k_cache.shape
    width = table.shape[1]
    half = page // 2
    n_prg = nl.num_programs(axes=0)
    pid = nl.program_id(axis=0)
    k_out = nl.ndarray((n_cols, kv_heads, half, d, 2), dtype=k_cache.dtype, buffer=nl.shared_hbm,
                       name="prefill_pack_k")
    v_out = nl.ndarray((n_cols, kv_heads, page, d), dtype=v_cache.dtype, buffer=nl.shared_hbm, name="prefill_pack_v")
    t_out = nl.ndarray((1, width), dtype=nl.int32, buffer=nl.shared_hbm, name="prefill_pack_table")
    k_rows = k_cache.reshape((blocks, page * d))
    v_rows = v_cache.reshape((blocks, page * d))
    fc = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    if window:
        fc_i = nl.ndarray((P, 1), dtype=cached.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=fc_i[0:P, 0:1], src=cached.reshape((1, 1)).ap(pattern=[[0, P], [1, 1]], offset=0))
        nisa.tensor_scalar(dst=fc_i[0:P, 0:1], data=fc_i[0:P, 0:1], op0=nl.add, operand0=1 - window)
        nisa.tensor_scalar(dst=fc_i[0:P, 0:1], data=fc_i[0:P, 0:1], op0=nl.maximum, operand0=0)
        nisa.tensor_scalar(dst=fc_i[0:P, 0:1], data=fc_i[0:P, 0:1], op0=nl.right_shift,
                           operand0=page.bit_length() - 1)
        nisa.tensor_copy(dst=fc[0:P, 0:1], src=fc_i[0:P, 0:1])
    else:
        nisa.memset(dst=fc[0:P, 0:1], value=0.0)
    for g0 in range(0, n_cols, P):
        gn = min(P, n_cols - g0)
        # this program's pages of the group: the first or second half (both the whole of a group of one)
        lo = 0 if (n_prg == 1 or gn < 2 or pid == 0) else gn // 2
        hi = gn if (n_prg == 1 or gn < 2 or pid == 1) else gn // 2
        m = hi - lo
        i_f = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.iota(i_f[0:m, 0:1], pattern=[[0, 1]], offset=g0 + lo, channel_multiplier=1)
        col = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=col[0:m, 0:1], data1=i_f[0:m, 0:1], data2=fc[0:m, 0:1], op=nl.add)
        nisa.tensor_scalar(dst=col[0:m, 0:1], data=col[0:m, 0:1], op0=nl.minimum, operand0=float(width - 1))
        col_i = nl.ndarray((P, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=col_i[0:m, 0:1], src=col[0:m, 0:1])
        pg_i = nl.ndarray((P, 1), dtype=table.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=pg_i[0:m, 0:1], src=table.reshape((width, 1)).ap(
            pattern=[[1, m], [1, 1]], offset=0, vector_offset=col_i[0:m, 0:1], indirect_dim=0))
        nisa.tensor_scalar(dst=pg_i[0:m, 0:1], data=pg_i[0:m, 0:1], op0=nl.maximum, operand0=0)
        # V whole, K's even and odd tokens, each page a partition
        vt = nl.ndarray((P, page, d), dtype=v_cache.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=vt[0:m, 0:page, 0:d], src=v_rows.ap(pattern=[[page * d, m], [d, page], [1, d]], offset=0,
                                                              vector_offset=pg_i[0:m, 0:1], indirect_dim=0))
        kp = nl.ndarray((P, half, d, 2), dtype=k_cache.dtype, buffer=nl.sbuf)
        for e in range(2):
            ke = nl.ndarray((P, half, d), dtype=k_cache.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=ke[0:m, 0:half, 0:d], src=k_rows.ap(pattern=[[page * d, m], [2 * d, half], [1, d]],
                                                                  offset=e * d, vector_offset=pg_i[0:m, 0:1],
                                                                  indirect_dim=0))
            nisa.tensor_copy(dst=kp[0:m, 0:half, 0:d, e], src=ke[0:m, 0:half, 0:d])
        nisa.dma_copy(dst=v_out[g0 + lo:g0 + hi, 0, 0:page, 0:d], src=vt[0:m, 0:page, 0:d])
        nisa.dma_copy(dst=k_out[g0 + lo:g0 + hi, 0, 0:half, 0:d, 0:2], src=kp[0:m, 0:half, 0:d, 0:2])
    if n_prg == 1 or pid == 0:
        # column j: j - first where that is in [0, n_cols), else 0; -1 where the table had no page
        j = nl.ndarray((1, width), dtype=nl.float32, buffer=nl.sbuf)
        nisa.iota(j[0:1, 0:width], pattern=[[1, width]], offset=0, channel_multiplier=0)
        rel = nl.ndarray((1, width), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=rel[0:1, 0:width], data=j[0:1, 0:width], op0=nl.subtract, operand0=fc[0:1, 0:1])
        ins = nl.ndarray((1, width), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=ins[0:1, 0:width], data=rel[0:1, 0:width], op0=nl.greater_equal, operand0=0.0)
        lt = nl.ndarray((1, width), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=lt[0:1, 0:width], data=rel[0:1, 0:width], op0=nl.less, operand0=float(n_cols))
        nisa.tensor_tensor(dst=ins[0:1, 0:width], data1=ins[0:1, 0:width], data2=lt[0:1, 0:width], op=nl.multiply)
        nisa.tensor_tensor(dst=rel[0:1, 0:width], data1=rel[0:1, 0:width], data2=ins[0:1, 0:width], op=nl.multiply)
        tb_i = nl.ndarray((1, width), dtype=table.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=tb_i[0:1, 0:width], src=table[0:1, 0:width])
        tb = nl.ndarray((1, width), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=tb[0:1, 0:width], src=tb_i[0:1, 0:width])
        has = nl.ndarray((1, width), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=has[0:1, 0:width], data=tb[0:1, 0:width], op0=nl.greater_equal, operand0=0.0)
        # (rel + 1) x has - 1
        nisa.tensor_scalar(dst=rel[0:1, 0:width], data=rel[0:1, 0:width], op0=nl.add, operand0=1.0)
        nisa.tensor_tensor(dst=rel[0:1, 0:width], data1=rel[0:1, 0:width], data2=has[0:1, 0:width], op=nl.multiply)
        nisa.tensor_scalar(dst=rel[0:1, 0:width], data=rel[0:1, 0:width], op0=nl.add, operand0=-1.0)
        out_i = nl.ndarray((1, width), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=out_i[0:1, 0:width], src=rel[0:1, 0:width])
        nisa.dma_copy(dst=t_out[0:1, 0:width], src=out_i[0:1, 0:width])
    return k_out, v_out, t_out


@nki.jit
def prefill_dcp_window(k_cache, v_cache, table, cached, k_chunk, v_chunk, mesh, dcp=8, kv_scale=1.0):
    """A sliding layer's local KV for its chunk's prefill under DCP, in one kernel (attention._prefill_dcp's sliding
    branch): the window's prior is the last page before the chunk, which the group's last rank holds; with the chunk's
    own pages after it the layer attends locally (the segmented attention over this table).

        col     = max(cached / (dcp x page) - 1, 0);  page_id = max(table[0, col], 0)     this rank's last prior page
        prior   = all_gather(k_cache[page_id])[dcp - 1]                               the group's last rank's
        K, V    = [prior | the chunk's pages] in bf16 (an FP8 cache's divided by its scale)
        table'  = [min(j + (cached == 0), n - 1) for j < n];  prior_tokens = page where cached > 0, else 0

    k_cache / v_cache [blocks, 1, page, d] (bf16 or FP8 at ``kv_scale``); table [1, width] int32; cached [1, 1] int32;
    k_chunk / v_chunk [T, d] the chunk's K / V as the cache holds them (prefill_front's); ``mesh`` the DCP group.
    Returns (K, V [1 + T / page, 1, page, d] bf16, table' [1, 1 + T / page] int32, prior_tokens [1, 1] int32). The
    prior page's gather runs through shared HBM between core barriers."""
    from nki import collectives as nccl

    from sglang_neuron.models.mimo_v2.kernels.head import _replica_group

    blocks, kvh, page, d = k_cache.shape
    width = table.shape[1]
    T = k_chunk.shape[0]
    n = 1 + T // page
    n_prg = nl.num_programs(axes=0)
    pid = nl.program_id(axis=0)
    rg = _replica_group(mesh)
    shift = (dcp * page).bit_length() - 1  # dcp x page a power of two
    fp8 = k_cache.dtype != nl.bfloat16
    k_out = nl.ndarray((n, kvh, page, d), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="prefill_dcpw_k")
    v_out = nl.ndarray((n, kvh, page, d), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="prefill_dcpw_v")
    t_out = nl.ndarray((1, n), dtype=nl.int32, buffer=nl.shared_hbm, name="prefill_dcpw_table")
    pt_out = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.shared_hbm, name="prefill_dcpw_prior")
    mine = nl.ndarray((2, page, d), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="prefill_dcpw_mine")
    every = nl.ndarray((2 * dcp, page, d), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="prefill_dcpw_every")

    c_i = nl.ndarray((P, 1), dtype=cached.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=c_i[0:P, 0:1], src=cached.reshape((1, 1)).ap(pattern=[[0, P], [1, 1]], offset=0))
    col = nl.ndarray((P, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=col[0:P, 0:1], data=c_i[0:P, 0:1], op0=nl.right_shift, operand0=shift)
    nisa.tensor_scalar(dst=col[0:P, 0:1], data=col[0:P, 0:1], op0=nl.add, operand0=-1)
    nisa.tensor_scalar(dst=col[0:P, 0:1], data=col[0:P, 0:1], op0=nl.maximum, operand0=0, op1=nl.minimum,
                       operand1=width - 1)
    # the page id on every partition (each gathers the same column), then each token row of it: page_id x page + p
    pg = nl.ndarray((P, 1), dtype=table.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=pg[0:P, 0:1], src=table.reshape((width, 1)).ap(pattern=[[1, P], [1, 1]], offset=0,
                                                                      vector_offset=col[0:P, 0:1], indirect_dim=0))
    nisa.tensor_scalar(dst=pg[0:P, 0:1], data=pg[0:P, 0:1], op0=nl.maximum, operand0=0)
    pf = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=pf[0:P, 0:1], src=pg[0:P, 0:1])
    nisa.tensor_scalar(dst=pf[0:P, 0:1], data=pf[0:P, 0:1], op0=nl.multiply, operand0=float(page))
    pp = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(pp[0:P, 0:1], pattern=[[0, 1]], offset=0, channel_multiplier=1)
    nisa.tensor_tensor(dst=pf[0:P, 0:1], data1=pf[0:P, 0:1], data2=pp[0:P, 0:1], op=nl.add)
    ridx = nl.ndarray((P, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=ridx[0:P, 0:1], src=pf[0:P, 0:1])
    # this rank's prior page (K then V; page = 128 rows on the partitions), each program one of the two
    for which, cache in ((0, k_cache), (1, v_cache)):
        if n_prg > 1 and which != pid:
            continue
        rows = cache.reshape((blocks * kvh * page, d))
        raw = nl.ndarray((P, d), dtype=cache.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=raw[0:page, 0:d], src=rows.ap(pattern=[[d, page], [1, d]], offset=0,
                                                         vector_offset=ridx[0:page, 0:1], indirect_dim=0))
        f = nl.ndarray((P, d), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=f[0:page, 0:d], src=raw[0:page, 0:d])
        if fp8:
            nisa.tensor_scalar(dst=f[0:page, 0:d], data=f[0:page, 0:d], op0=nl.multiply, operand0=1.0 / kv_scale)
        b = nl.ndarray((P, d), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=b[0:page, 0:d], src=f[0:page, 0:d])
        nisa.dma_copy(dst=mine[which, 0:page, 0:d], src=b[0:page, 0:d])
    if n_prg > 1:
        nisa.core_barrier(data=mine, cores=(0, 1))
    nccl.all_gather(srcs=[mine], dsts=[every], replica_group=rg, collective_dim=0)
    if n_prg > 1:
        nisa.core_barrier(data=every, cores=(0, 1))
    # the last rank's K (rows 2 (dcp - 1)) and V (2 (dcp - 1) + 1), then the chunk's pages
    for which, chunk, out in ((0, k_chunk, k_out), (1, v_chunk, v_out)):
        if n_prg > 1 and which != pid:
            continue
        nisa.dma_copy(dst=out[0:1, 0, 0:page, 0:d], src=every[2 * (dcp - 1) + which:2 * (dcp - 1) + which + 1,
                                                                0:page, 0:d])
        for t0 in range(0, T, P):
            ct = nl.ndarray((P, d), dtype=chunk.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=ct[0:P, 0:d], src=chunk[t0:t0 + P, 0:d])
            cf = nl.ndarray((P, d), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=cf[0:P, 0:d], src=ct[0:P, 0:d])
            if fp8:
                nisa.tensor_scalar(dst=cf[0:P, 0:d], data=cf[0:P, 0:d], op0=nl.multiply, operand0=1.0 / kv_scale)
            cb = nl.ndarray((P, d), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=cb[0:P, 0:d], src=cf[0:P, 0:d])
            p_idx = 1 + t0 // page
            o0 = t0 % page
            nisa.dma_copy(dst=out[p_idx:p_idx + 1, 0, o0:o0 + P, 0:d], src=cb[0:P, 0:d])
    if n_prg == 1 or pid == 0:
        # table' = min(j + first, n - 1), first = (cached == 0); prior_tokens = page x (cached > 0)
        cf1 = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=cf1[0:1, 0:1], src=c_i[0:1, 0:1])
        has = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=has[0:1, 0:1], data=cf1[0:1, 0:1], op0=nl.greater, operand0=0.0)
        first = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=first[0:1, 0:1], data=has[0:1, 0:1], op0=nl.multiply, operand0=-1.0, op1=nl.add,
                           operand1=1.0)
        j = nl.ndarray((1, n), dtype=nl.float32, buffer=nl.sbuf)
        nisa.iota(j[0:1, 0:n], pattern=[[1, n]], offset=0, channel_multiplier=0)
        nisa.tensor_scalar(dst=j[0:1, 0:n], data=j[0:1, 0:n], op0=nl.add, operand0=first[0:1, 0:1], op1=nl.minimum,
                           operand1=float(n - 1))
        j_i = nl.ndarray((1, n), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=j_i[0:1, 0:n], src=j[0:1, 0:n])
        nisa.dma_copy(dst=t_out, src=j_i[0:1, 0:n])
        nisa.tensor_scalar(dst=has[0:1, 0:1], data=has[0:1, 0:1], op0=nl.multiply, operand0=float(page))
        h_i = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=h_i[0:1, 0:1], src=has[0:1, 0:1])
        nisa.dma_copy(dst=pt_out, src=h_i[0:1, 0:1])
    return k_out, v_out, t_out, pt_out


@nki.jit
def dcp_full_inputs(table, cached, dcp=8, kv_scale=1.0):
    """A full layer's DCP prefill inputs (attention._prefill_dcp's full branch, kernels/attention_block.
    prefill_kv_parallel's): the table with its missing columns (-1) at page 0, each rank's share of the prior
    (cached / dcp) and the FP8 scale tile 1 / kv_scale. table [1, width] int32; cached [1, 1] int32 (dcp a power of
    two). Returns (table' [1, width] int32, prior [1, 1] int32, scale [128, 1] fp32)."""
    width = table.shape[1]
    n_prg = nl.num_programs(axes=0)
    pid = nl.program_id(axis=0)
    t_out = nl.ndarray((1, width), dtype=nl.int32, buffer=nl.shared_hbm, name="dcp_full_table")
    p_out = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.shared_hbm, name="dcp_full_prior")
    s_out = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.shared_hbm, name="dcp_full_scale")
    tb = nl.ndarray((1, width), dtype=table.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=tb[0:1, 0:width], src=table[0:1, 0:width])
    nisa.tensor_scalar(dst=tb[0:1, 0:width], data=tb[0:1, 0:width], op0=nl.maximum, operand0=0)
    c = nl.ndarray((1, 1), dtype=cached.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=c[0:1, 0:1], src=cached.reshape((1, 1)))
    nisa.tensor_scalar(dst=c[0:1, 0:1], data=c[0:1, 0:1], op0=nl.right_shift, operand0=dcp.bit_length() - 1)
    sc = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=sc[0:P, 0:1], value=1.0 / kv_scale)
    if n_prg == 1 or pid == 0:
        nisa.dma_copy(dst=t_out, src=tb[0:1, 0:width])
        nisa.dma_copy(dst=p_out, src=c[0:1, 0:1])
        nisa.dma_copy(dst=s_out, src=sc[0:P, 0:1])
    return t_out, p_out, s_out
