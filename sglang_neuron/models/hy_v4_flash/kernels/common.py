# SPDX-License-Identifier: Apache-2.0
"""Pieces the Hy4-Flash decode kernels share: SBUF transposes, row-scaled FP8 quantization, projections against an
HBM weight streamed a 128-row tile at a time, RMSNorm and the iHC mix, each as the graph computes it (ops.py)."""

import nki.isa as nisa
import nki.language as nl

P = 128
FP8_MAX = 240.0


def transpose(src, T: int, cols: int, dtype):
    """[128, ceil(cols / 128), T] of ``src`` [128, cols] rows 0 .. T - 1 (the contraction dim on the partitions; a last
    partial tile fills its first ``cols % 128`` partitions)."""
    n = (cols + P - 1) // P
    out = nl.ndarray((P, n, P), dtype=dtype, buffer=nl.sbuf)
    for t in range(n):
        w = min(P, cols - t * P)
        tp = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.psum)
        nisa.nc_transpose(dst=tp[0:w, 0:T], data=src[0:T, t * P:t * P + w])
        nisa.tensor_copy(dst=out[0:w, t, 0:T], src=tp[0:w, 0:T])
    return out


def token_scales(x, T: int, n: int):
    """(scale, 1 / scale) [128, 1] fp32 of each row of ``x`` [128, n]: 240 / its absolute max (a zero row: 1)."""
    hi = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    lo = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(hi[0:T, 0:1], op=nl.maximum, data=x[0:T, 0:n], axis=1, keepdims=True)
    nisa.tensor_reduce(lo[0:T, 0:1], op=nl.minimum, data=x[0:T, 0:n], axis=1, keepdims=True)
    amax = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=lo[0:T, 0:1], data=lo[0:T, 0:1], op0=nl.multiply, operand0=-1.0)
    nisa.tensor_tensor(amax[0:T, 0:1], hi[0:T, 0:1], lo[0:T, 0:1], nl.maximum)
    nisa.tensor_scalar(dst=amax[0:T, 0:1], data=amax[0:T, 0:1], op0=nl.maximum, operand0=1e-30)
    inv = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=inv[0:T, 0:1], data=amax[0:T, 0:1], op0=nl.multiply, operand0=1.0 / FP8_MAX)
    sc = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.reciprocal(dst=sc[0:T, 0:1], data=inv[0:T, 0:1])
    return sc, inv


def quantize_rows(x, T: int, n: int):
    """(x x 240 / its row's absolute max, FP8-range bf16 [128, n], 1 / scale [128, 1])."""
    sc, inv = token_scales(x, T, n)
    q = nl.ndarray((P, n), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=q[0:T, 0:n], data=x[0:T, 0:n], op0=nl.multiply, operand0=sc[0:T, 0:1],
                       op1=nl.minimum, operand1=FP8_MAX)
    nisa.tensor_scalar(dst=q[0:T, 0:n], data=q[0:T, 0:n], op0=nl.maximum, operand0=-FP8_MAX)
    return q, inv


def project(xT, T: int, w, out_dtype, col0: int = 0, cols=None, row0: int = 0, rows=None, ring: int = 6):
    """[128, cols] (rows 0 .. T - 1) of x @ w[row0:row0 + rows, col0:col0 + cols]: ``xT`` [128, k tiles, T] the input
    transposed (tile t holds input columns t x 128 ..), ``w`` [K, N] in HBM streamed one 128-row tile at a time
    through ``ring`` buffers, fp32 accumulation, ``out_dtype`` result. A 2-deep ring left the matmuls waiting on the
    tiles' DMAs: the sliding attention kernel ran 0.285 / 0.443 ms at 1 / 16 rows against 0.202 / 0.389 with 6."""
    K, N = w.shape
    rows = K - row0 if rows is None else rows
    cols = N - col0 if cols is None else cols
    n_k = (rows + P - 1) // P
    bufs = [nl.ndarray((P, cols), dtype=w.dtype, buffer=nl.sbuf) for _ in range(ring)]
    chunks = [(c, min(512, cols - c)) for c in range(0, cols, 512)]
    ps = [nl.ndarray((P, 512), dtype=nl.float32, buffer=nl.psum) for _ in chunks]
    for k in range(n_k):
        r = min(P, rows - k * P)
        buf = bufs[k % ring]
        nisa.dma_copy(dst=buf[0:r, 0:cols], src=w.ap(pattern=[[N, r], [1, cols]], offset=(row0 + k * P) * N + col0),
                      dge_mode=nisa.dge_mode.hwdge)
        for ci, (c, cs) in enumerate(chunks):
            nisa.nc_matmul(dst=ps[ci][0:T, 0:cs], stationary=xT[0:r, k, 0:T], moving=buf[0:r, c:c + cs])
    out = nl.ndarray((P, cols), dtype=out_dtype, buffer=nl.sbuf)
    for ci, (c, cs) in enumerate(chunks):
        nisa.tensor_copy(dst=out[0:T, c:c + cs], src=ps[ci][0:T, 0:cs])
    return out


def rms_norm(x, T: int, n: int, w, eps: float):
    """ops.rms_norm of ``x`` [128, n] bf16 by ``w`` [1, n] bf16 in HBM: normalized in fp32, rounded to bf16, times the
    weight in bf16."""
    gam = nl.ndarray((P, n), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=gam[0:T, 0:n], src=w.ap(pattern=[[0, T], [1, n]], offset=0))
    sq = nl.ndarray((P, n), dtype=nl.float32, buffer=nl.sbuf)
    ss = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=sq[0:T, 0:n], op=nl.square, data=x[0:T, 0:n], reduce_op=nl.add, reduce_res=ss[0:T, 0:1],
                    reduce_cmd=nisa.reduce_cmd.reset_reduce)
    rr = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=rr[0:T, 0:1], op=nl.rsqrt, data=ss[0:T, 0:1], scale=1.0 / n, bias=eps)
    nb = nl.ndarray((P, n), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=nb[0:T, 0:n], data=x[0:T, 0:n], op0=nl.multiply, operand0=rr[0:T, 0:1])
    out = nl.ndarray((P, n), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_tensor(out[0:T, 0:n], nb[0:T, 0:n], gam[0:T, 0:n], nl.multiply)
    return out


def hc_mix(streams, fn_hi, fn_lo, hc_base, hc_scale, T: int, eps: float, hc_eps: float, hc_mag: float,
           row0: int = 0):
    """ops.hc_mix of rows row0 .. row0 + T - 1 of ``streams`` [rows, hc, H] (HBM, T <= 128): (the streams in SBUF
    [128, hc H] bf16, post [128, hc] fp32, the block input [128, H] bf16). fn's mixes contract its bf16 part and its
    bf16 rest (fn_hi, fn_lo [hc H, 2 hc]) with fp32 accumulation."""
    rows, HC, H = streams.shape
    D = HC * H
    D1 = D // P
    st = nl.ndarray((P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=st[0:T, 0:D], src=streams.reshape((rows, D)).ap(pattern=[[D, T], [1, D]], offset=row0 * D))
    sq = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.sbuf)
    ss = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=sq[0:T, 0:D], op=nl.square, data=st[0:T, 0:D], reduce_op=nl.add, reduce_res=ss[0:T, 0:1],
                    reduce_cmd=nisa.reduce_cmd.reset_reduce)
    rr = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=rr[0:T, 0:1], op=nl.rsqrt, data=ss[0:T, 0:1], scale=1.0 / D, bias=eps)
    stT = transpose(st, T, D, nl.bfloat16)
    fh = nl.ndarray((P, D1, 2 * HC), dtype=nl.bfloat16, buffer=nl.sbuf)
    fl = nl.ndarray((P, D1, 2 * HC), dtype=nl.bfloat16, buffer=nl.sbuf)
    pat = [[2 * HC, P], [P * 2 * HC, D1], [1, 2 * HC]]
    nisa.dma_copy(dst=fh[0:P, 0:D1, 0:2 * HC], src=fn_hi.ap(pattern=pat, offset=0))
    nisa.dma_copy(dst=fl[0:P, 0:D1, 0:2 * HC], src=fn_lo.ap(pattern=pat, offset=0))
    mp = nl.ndarray((P, 2 * HC), dtype=nl.float32, buffer=nl.psum)
    for k in range(D1):
        nisa.nc_matmul(dst=mp[0:T, 0:2 * HC], stationary=stT[0:P, k, 0:T], moving=fh[0:P, k, 0:2 * HC])
        nisa.nc_matmul(dst=mp[0:T, 0:2 * HC], stationary=stT[0:P, k, 0:T], moving=fl[0:P, k, 0:2 * HC])
    mix = nl.ndarray((P, 2 * HC), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=mix[0:T, 0:2 * HC], data=mp[0:T, 0:2 * HC], op0=nl.multiply, operand0=rr[0:T, 0:1])
    hsc = nl.ndarray((P, 2), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=hsc[0:T, 0:2], src=hc_scale.ap(pattern=[[0, T], [1, 2]], offset=0))
    hb = nl.ndarray((P, 2 * HC), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=hb[0:T, 0:2 * HC], src=hc_base.ap(pattern=[[0, T], [1, 2 * HC]], offset=0))
    gate = nl.ndarray((P, 2 * HC), dtype=nl.float32, buffer=nl.sbuf)
    for half in range(2):
        c0 = half * HC
        nisa.scalar_tensor_tensor(dst=gate[0:T, c0:c0 + HC], data=mix[0:T, c0:c0 + HC], op0=nl.multiply,
                                  operand0=hsc[0:T, half:half + 1], op1=nl.add, operand1=hb[0:T, c0:c0 + HC])
    sig = nl.ndarray((P, 2 * HC), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=sig[0:T, 0:2 * HC], op=nl.sigmoid, data=gate[0:T, 0:2 * HC])
    pre = nl.ndarray((P, HC), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=pre[0:T, 0:HC], data=sig[0:T, 0:HC], op0=nl.add, operand0=hc_eps)
    post = nl.ndarray((P, HC), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=post[0:T, 0:HC], data=sig[0:T, HC:2 * HC], op0=nl.multiply, operand0=hc_mag,
                       op1=nl.add, operand1=hc_eps)
    xin = nl.ndarray((P, H), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=xin[0:T, 0:H], data=st[0:T, 0:H], op0=nl.multiply, operand0=pre[0:T, 0:1])
    for s in range(1, HC):
        nisa.scalar_tensor_tensor(dst=xin[0:T, 0:H], data=st[0:T, s * H:(s + 1) * H], op0=nl.multiply,
                                  operand0=pre[0:T, s:s + 1], op1=nl.add, operand1=xin[0:T, 0:H])
    xb = nl.ndarray((P, H), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=xb[0:T, 0:H], src=xin[0:T, 0:H])
    return st, post, xb


def store_post(post, T: int, HC: int, name: str):
    """post [T, hc] fp32 in shared HBM, each program storing its share of the columns (both hold the same values)."""
    n_prg = nl.num_programs(axes=0)
    pid = nl.program_id(axis=0)
    out = nl.ndarray((T, HC), dtype=nl.float32, buffer=nl.shared_hbm, name=name)
    share = HC // n_prg
    nisa.dma_copy(dst=out.ap(pattern=[[HC, T], [1, share]], offset=pid * share),
                  src=post[0:T, pid * share:(pid + 1) * share])
    return out
