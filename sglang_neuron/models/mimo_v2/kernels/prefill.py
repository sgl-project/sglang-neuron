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
    for t in range(T // P):
        # the tiles alternate between the programs; with fewer tiles than programs each runs them all (the same
        # values to the same places), since an idle program fails the compiler's allocation
        if n_prg > 1 and T // P >= n_prg and t % n_prg != pid:
            continue
        t0 = t * P
        # ---- RMSNorm ----
        xt = nl.ndarray((P, H), dtype=x.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=xt[0:P, 0:H], src=x.ap(pattern=[[H, P], [1, H]], offset=t0 * H))
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
        nisa.dma_copy(dst=cb[0:P, 0:half], src=cos.ap(pattern=[[half, P], [1, half]], offset=t0 * half))
        nisa.tensor_copy(dst=c[0:P, 0:half], src=cb[0:P, 0:half])
        sb = nl.ndarray((P, half), dtype=sin.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=sb[0:P, 0:half], src=sin.ap(pattern=[[half, P], [1, half]], offset=t0 * half))
        nisa.tensor_copy(dst=s[0:P, 0:half], src=sb[0:P, 0:half])
        for h in range(nq):
            qh = nl.ndarray((P, d), dtype=nl.bfloat16, buffer=nl.sbuf)
            _rope_half(qkv, c, s, h * d, half, qh, f"q{h}")
            if q_scale != 1.0:
                nisa.tensor_scalar(dst=qh[0:P, 0:d], data=qh[0:P, 0:d], op0=nl.multiply, operand0=q_scale)
            nisa.dma_copy(dst=q_out.ap(pattern=[[d, P], [1, d]], offset=h * T * d + t0 * d), src=qh[0:P, 0:d])
        kh = nl.ndarray((P, d), dtype=nl.bfloat16, buffer=nl.sbuf)
        _rope_half(qkv, c, s, nq * d, half, kh, "k")
        vh = nl.ndarray((P, d), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=vh[0:P, 0:d], src=qkv[0:P, (nq + 1) * d:(nq + 2) * d])
        kc = _to_cache(kh, d, kv_scale, k_cache.dtype)
        vc = _to_cache(vh, d, kv_scale, v_cache.dtype)
        nisa.dma_copy(dst=k_out.ap(pattern=[[d, P], [1, d]], offset=t0 * d), src=kc[0:P, 0:d])
        nisa.dma_copy(dst=v_out.ap(pattern=[[d, P], [1, d]], offset=t0 * d), src=vc[0:P, 0:d])
        # ---- the slots: outside the pool -> 0 (the reserved padding page) ----
        sl = nl.ndarray((P, 1), dtype=slots.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=sl[0:P, 0:1], src=slots.ap(pattern=[[1, P], [1, 1]], offset=t0))
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
    for t in range(n_tiles):
        t0 = t * P
        # the heads' first vd channels times 1 / the FP8 scale, transposed: [vd, tokens]
        aT = nl.ndarray((P, K1, P), dtype=nl.bfloat16, buffer=nl.sbuf)
        for h in range(nq):
            a = nl.ndarray((P, vd), dtype=attn.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=a[0:P, 0:vd], src=attn.ap(pattern=[[d, P], [1, vd]], offset=h * T * d + t0 * d))
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
        nisa.dma_copy(dst=part.ap(pattern=[[H, P], [1, HH]], offset=t0 * H + h0), src=pt[0:P, 0:HH])
    red = _hbm_all_reduce(part, rg, n_prg, "prefill_back_reduced")
    for t in range(n_tiles):
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
        nisa.dma_copy(dst=hn_out.ap(pattern=[[H, P], [1, HH]], offset=t0 * H + h0), src=hn[0:P, 0:HH])
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
    # the slots plus a sink past them, which the unselected (token, expert) pairs write
    scratch = nl.ndarray((NB + 1, 1), dtype=nl.int32, buffer=nl.shared_hbm, name="prefill_route_scratch")
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

    sel_all = nl.ndarray((P, NT, E), dtype=nl.float32, buffer=nl.sbuf)
    rank_all = nl.ndarray((P, NT, E), dtype=nl.float32, buffer=nl.sbuf)
    carry = nl.ndarray((P, E), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=carry[0:P, 0:E], value=0.0)
    for t in range(NT):
        t0 = t * P
        # ---- the router over the tile (moe_half's) ----
        xt = nl.ndarray((P, H), dtype=hn.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=xt[0:P, 0:H], src=hn.ap(pattern=[[H, P], [1, H]], offset=t0 * H))
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
            nisa.dma_copy(dst=aff_out.ap(pattern=[[E, P], [1, E]], offset=t0 * E), src=aff_b[0:P, 0:E])
        # ---- a real token: the first, or one whose position differs from the token before it ----
        pc = nl.ndarray((P, 1), dtype=positions.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=pc[0:P, 0:1], src=positions.ap(pattern=[[1, P], [1, 1]], offset=t0))
        pv = nl.ndarray((P, 1), dtype=positions.dtype, buffer=nl.sbuf)
        if t0 == 0:
            nisa.memset(dst=pv[0:1, 0:1], value=-1)
            if P > 1:
                nisa.dma_copy(dst=pv[1:P, 0:1], src=positions.ap(pattern=[[1, P - 1], [1, 1]], offset=0))
        else:
            nisa.dma_copy(dst=pv[0:P, 0:1], src=positions.ap(pattern=[[1, P], [1, 1]], offset=t0 - 1))
        pcf = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=pcf[0:P, 0:1], src=pc[0:P, 0:1])
        pvf = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=pvf[0:P, 0:1], src=pv[0:P, 0:1])
        real = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=real[0:P, 0:1], data1=pcf[0:P, 0:1], data2=pvf[0:P, 0:1], op=nl.not_equal)
        # ---- selected pairs and their inclusive rank among the expert's tokens so far ----
        ab = nl.ndarray((P, E), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=ab[0:P, 0:E], src=aff_b[0:P, 0:E])
        s_f = nl.ndarray((P, E), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=s_f[0:P, 0:E], data=ab[0:P, 0:E], op0=nl.not_equal, operand0=0.0,
                           op1=nl.multiply, operand1=real[0:P, 0:1])
        nisa.tensor_copy(dst=sel_all[0:P, t, 0:E], src=s_f[0:P, 0:E])
        s_b = nl.ndarray((P, E), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=s_b[0:P, 0:E], src=s_f[0:P, 0:E])
        pre = nl.ndarray((P, E), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=pre[0:P, 0:E], stationary=tri[0:P, 0:P], moving=s_b[0:P, 0:E])
        nisa.tensor_tensor(dst=rank_all[0:P, t, 0:E], data1=pre[0:P, 0:E], data2=carry[0:P, 0:E], op=nl.add)
        tot = nl.ndarray((P, E), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=tot[0:P, 0:E], stationary=ones[0:P, 0:P], moving=s_b[0:P, 0:E])
        nisa.tensor_tensor(dst=carry[0:P, 0:E], data1=carry[0:P, 0:E], data2=tot[0:P, 0:E], op=nl.add)

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

    # ---- the slots: -1 everywhere, then each selected pair's token id at its slot (the rest at the sink) ----
    neg = nl.ndarray((P, (NB + P) // P), dtype=nl.int32, buffer=nl.sbuf)
    nisa.memset(dst=neg[0:P, 0:(NB + P) // P], value=-1)
    full_rows = NB // P
    rest = NB + 1 - full_rows * P
    if w0:
        if full_rows:
            nisa.dma_copy(dst=scratch.ap(pattern=[[full_rows, P], [1, full_rows]], offset=0),
                          src=neg[0:P, 0:full_rows])
        nisa.dma_copy(dst=scratch.ap(pattern=[[1, rest], [1, 1]], offset=full_rows * P), src=neg[0:rest, 0:1])
    sink = float(NB)
    for t in range(NT):
        tid_f = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=tid_f[0:P, 0:1], data=part_f[0:P, 0:1], op0=nl.add, operand0=float(t * P))
        tid = nl.ndarray((P, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=tid[0:P, 0:1], src=tid_f[0:P, 0:1])
        # slot = sel x (base + rank - 1 - sink) + sink
        sl = nl.ndarray((P, E), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=sl[0:P, 0:E], data1=rank_all[0:P, t, 0:E], data2=base[0:P, 0:E], op=nl.add)
        nisa.tensor_scalar(dst=sl[0:P, 0:E], data=sl[0:P, 0:E], op0=nl.add, operand0=-1.0 - sink)
        nisa.tensor_tensor(dst=sl[0:P, 0:E], data1=sl[0:P, 0:E], data2=sel_all[0:P, t, 0:E], op=nl.multiply)
        nisa.tensor_scalar(dst=sl[0:P, 0:E], data=sl[0:P, 0:E], op0=nl.add, operand0=sink)
        sl_i = nl.ndarray((P, E), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=sl_i[0:P, 0:E], src=sl[0:P, 0:E])
        if w0:
            for e in range(E):
                nisa.dma_copy(dst=scratch.ap(pattern=[[1, P], [1, 1]], offset=0, vector_offset=sl_i[0:P, e:e + 1],
                                             indirect_dim=0), src=tid[0:P, 0:1])
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
    kernels/layer._all_reduce (unfenced in-kernel sums came back wrong on trn2)."""
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
    nisa.dma_copy(dst=xt[0:P, 0:HH], src=x.ap(pattern=[[H, P], [1, HH]], offset=t0 * H + h0))
    rt = nl.ndarray((P, HH), dtype=red.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=rt[0:P, 0:HH], src=red.ap(pattern=[[H, P], [1, HH]], offset=t0 * H + h0))
    ht = nl.ndarray((P, HH), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=ht[0:P, 0:HH], data1=xt[0:P, 0:HH], data2=rt[0:P, 0:HH], op=nl.add)
    nisa.dma_copy(dst=h_out.ap(pattern=[[H, P], [1, HH]], offset=t0 * H + h0), src=ht[0:P, 0:HH])
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
    for t in range(T // P):
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
    for t in range(T // P):
        t0 = t * P
        xt = nl.ndarray((P, H), dtype=hn.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=xt[0:P, 0:H], src=hn.ap(pattern=[[H, P], [1, H]], offset=t0 * H))
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
        nisa.dma_copy(dst=part.ap(pattern=[[H, P], [1, HH]], offset=t0 * H + h0), src=pt[0:P, 0:HH])
    red = _hbm_all_reduce(part, rg, n_prg, "prefill_mlp_reduced")
    for t in range(T // P):
        _add_rows(x, red, h_out, t * P, HH, H, h0)
    return h_out
