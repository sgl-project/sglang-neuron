# SPDX-License-Identifier: Apache-2.0
"""MiMo-V2's prefill attention front in one NKI kernel (attention.MiMoAttention.prefill): what the chunk's
input RMSNorm, QKV projection, RoPE, FP8 KV scaling and the paged K/V write did in XLA (and kv_write.py's scatter).

Per 128-token tile (the tiles alternate between the two programs of LNC2), with this rank's weight resident in SBUF:

    xn       = RMSNorm(x)                     fp32 math, bf16 result, times the weight (models/transformer/model.RMSNorm)
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
def prefill_front(x, norm_w, w_qkv, cos, sin, slots, k_cache, v_cache, eps=1e-6, num_heads=1, kv_scale=1.0):
    """x [T, H] bf16 (T a multiple of 128) the chunk's residual stream; norm_w [1, H] bf16; w_qkv [H, (heads + 2) x d]
    bf16 (one KV head); cos / sin [T, d / 2] bf16 (rope.RotaryTables); slots [T, 1] int32 the pool slots; k_cache /
    v_cache [blocks, 1, page, d] (bf16, or FP8 at ``kv_scale``). Returns (q [heads, T, d] bf16, k [T, d], v [T, d] in the
    cache's dtype, k_cache, v_cache)."""
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
        if n_prg > 1 and t % n_prg != pid:
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
