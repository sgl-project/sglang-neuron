# SPDX-License-Identifier: Apache-2.0
# The expert loop follows MiMo-V2's decode MoE half (sgl-project/sglang-neuron#8, models/mimo_v2/kernels/moe_half.py,
# itself derived from Hy3's decode MoE block kernel); kernels/PROVENANCE.md lists what is copied and what changed.
"""Hy4-Flash's decode MoE half in one NKI kernel: from the residual streams after the attention to this rank's partial
of the LatentMoE output and the iHC post weights, with no XLA op in between.

Per layer and rank, for T <= 128 decode rows:
  iHC mix (ops.hc_mix): the streams' rms in fp32, their mixes through ``fn`` (bf16 hi + lo halves of the fp32 weight,
     fp32 accumulation), the sigmoid pre / post gates, the block input sum(pre_s x stream_s) in fp32, cast to bf16
  -> post_attention_layernorm (fp32 math, bf16 result, the weight in bf16)
  -> router (ops.route): fp32 sigmoid scores of bf16 x bf16 logits (fp32 accumulation), top 16 on score + bias by the
     16th largest biased score (two ``nisa.max8``), the chosen unbiased scores renormalized and scaled
  -> pre_expert_proj (bf16, fp32 accumulation, bf16 result) into the 1536-wide latent
  -> this rank's 16 local experts, all of them, each weighted by its affinity: bf16 activations x FP8 weights (or,
     ``fp8_act``, W8A8: each row quantized to FP8 at 240 / its absolute max), the per-output-row FP8 weight scales
     after each matmul, SwiGLU clamped at ``limit`` (gate above, up both ways)
  -> post_expert_proj of the routed latent (bf16) plus the rank's shard of the shared expert (SiLU, no clamp)
The outputs are the rank's partial [T, H] bf16, which the caller all-reduces, then combine_output_norm and the iHC add
(ops.hc_add) with ``post`` [T, hc] fp32.

The two programs of a logical NeuronCore each contract one half of the latent for the experts' gate / up and exchange
the halves of their partial sums (``sendrecv``); each keeps one half of I for the activation and the down projection,
and of the latent for post_expert_proj; a last exchange sums the two programs' output partials, each program storing
its half of H. The iHC mix, the norm, the router and pre_expert_proj run in both.

Inputs (this rank's shapes at TP32 / EP32: H 3072, hc 4, latent 1536, E 16 local of 512, I 1536, shared 48):
  streams        [T, hc, H] bf16
  fn_hi, fn_lo   [hc H, 2 hc] bf16, ffn_hc's fn transposed, its bf16 part and the bf16 rest
  hc_base        [1, 2 hc] fp32, hc_scale [1, 2] fp32
  ln_w           [1, H] bf16, post_attention_layernorm's weight
  router_wT      [H, EA] bf16; expert_bias [1, EA] fp32; rank [1, 1] fp32, this rank's EP rank
  pre_w          [H, L] bf16; post_w [L, H] bf16
  gate_up_w      [E, L, 2, I] fp8, down_w [E, I, L] fp8, gate_up_scale [E, 2, I] fp32, down_scale [E, L] fp32
  sh_gate_up     [H, 2 S] bf16 (gate columns then up), sh_down [S, H] bf16, the shared expert's shard
"""

import nki
import nki.isa as nisa
import nki.language as nl

P = 128
RING = 6
DN_COLS = 512
FP8_MAX = 240.0
BIG = 1.0e30


def _token_scales(x, T: int, n: int):
    """(scale, 1 / scale) [128, 1] fp32 of each row 0 .. T - 1 of ``x`` [128, n]: 240 / its absolute max (a zero row
    gets a scale of 1)."""
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


def _transpose(src, T: int, n_tiles: int, dtype):
    """[128, n_tiles, T] of ``src`` [128, n_tiles x 128] rows 0 .. T - 1: the contraction dim on the partitions."""
    out = nl.ndarray((P, n_tiles, P), dtype=dtype, buffer=nl.sbuf)
    for t in range(n_tiles):
        tp = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.psum)
        nisa.nc_transpose(dst=tp[0:P, 0:T], data=src[0:T, t * P:(t + 1) * P])
        nisa.tensor_copy(dst=out[0:P, t, 0:T], src=tp[0:P, 0:T])
    return out


def _quantize_rows(x, T: int, n: int):
    """(x x 240 / its row's absolute max in FP8-range bf16 [128, n], 1 / scale [128, 1])."""
    sc, inv = _token_scales(x, T, n)
    q = nl.ndarray((P, n), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=q[0:T, 0:n], data=x[0:T, 0:n], op0=nl.multiply, operand0=sc[0:T, 0:1],
                       op1=nl.minimum, operand1=FP8_MAX)
    nisa.tensor_scalar(dst=q[0:T, 0:n], data=q[0:T, 0:n], op0=nl.maximum, operand0=-FP8_MAX)
    return q, inv


@nki.jit
def moe_half(streams, fn_hi, fn_lo, hc_base, hc_scale, ln_w, router_wT, expert_bias, rank, pre_w, post_w,
             gate_up_w, down_w, gate_up_scale, down_scale, sh_gate_up, sh_down, eps=1e-5, hc_eps=1e-6,
             hc_mag=2.0, route_scale=1.0, top_k=16, limit=10.0, fp8_act=False, debug=False):
    T, HC, H = streams.shape
    D = HC * H
    E, L, _, I = gate_up_w.shape  # noqa: E741
    EA = router_wT.shape[1]
    S2 = sh_gate_up.shape[1]
    S = S2 // 2
    H1, D1, L1 = H // P, D // P, L // P
    n_prg = nl.num_programs(axes=0)
    pid = nl.program_id(axis=0)
    other = 1 - pid
    HH, LH, ILe = H // n_prg, L // n_prg, I // n_prg
    n_kl, n_ic = L1 // n_prg, ILe // P
    h0, l0 = pid * HH, pid * LH

    # ---- iHC mix (ops.hc_mix) ----
    st = nl.ndarray((P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=st[0:T, 0:D], src=streams.reshape((T, D)).ap(pattern=[[D, T], [1, D]], offset=0))
    sq = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.sbuf)
    ss = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=sq[0:T, 0:D], op=nl.square, data=st[0:T, 0:D], reduce_op=nl.add, reduce_res=ss[0:T, 0:1],
                    reduce_cmd=nisa.reduce_cmd.reset_reduce)
    rr = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=rr[0:T, 0:1], op=nl.rsqrt, data=ss[0:T, 0:1], scale=1.0 / D, bias=eps)
    stT = _transpose(st, T, D1, nl.bfloat16)
    fh = nl.ndarray((P, D1, 2 * HC), dtype=nl.bfloat16, buffer=nl.sbuf)
    fl = nl.ndarray((P, D1, 2 * HC), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=fh[0:P, 0:D1, 0:2 * HC], src=fn_hi.ap(pattern=[[2 * HC, P], [P * 2 * HC, D1], [1, 2 * HC]],
                                                           offset=0))
    nisa.dma_copy(dst=fl[0:P, 0:D1, 0:2 * HC], src=fn_lo.ap(pattern=[[2 * HC, P], [P * 2 * HC, D1], [1, 2 * HC]],
                                                           offset=0))
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

    # ---- post_attention_layernorm (models/transformer/model.RMSNorm) ----
    gam = nl.ndarray((P, H), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=gam[0:T, 0:H], src=ln_w.ap(pattern=[[0, T], [1, H]], offset=0))
    sq2 = nl.ndarray((P, H), dtype=nl.float32, buffer=nl.sbuf)
    ss2 = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=sq2[0:T, 0:H], op=nl.square, data=xb[0:T, 0:H], reduce_op=nl.add, reduce_res=ss2[0:T, 0:1],
                    reduce_cmd=nisa.reduce_cmd.reset_reduce)
    r2 = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=r2[0:T, 0:1], op=nl.rsqrt, data=ss2[0:T, 0:1], scale=1.0 / H, bias=eps)
    nb = nl.ndarray((P, H), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=nb[0:T, 0:H], data=xb[0:T, 0:H], op0=nl.multiply, operand0=r2[0:T, 0:1])
    xn = nl.ndarray((P, H), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_tensor(xn[0:T, 0:H], nb[0:T, 0:H], gam[0:T, 0:H], nl.multiply)
    nT = _transpose(xn, T, H1, nl.bfloat16)

    # ---- router (ops.route): top 16 by the 16th largest biased score ----
    rw = nl.ndarray((P, H1, EA), dtype=router_wT.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=rw[0:P, 0:H1, 0:EA], src=router_wT.ap(pattern=[[EA, P], [P * EA, H1], [1, EA]], offset=0),
                  dge_mode=nisa.dge_mode.hwdge)
    lg = nl.ndarray((P, EA), dtype=nl.float32, buffer=nl.psum)
    for h1 in range(H1):
        nisa.nc_matmul(dst=lg[0:T, 0:EA], stationary=nT[0:P, h1, 0:T], moving=rw[0:P, h1, 0:EA])
    scr = nl.ndarray((P, EA), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=scr[0:T, 0:EA], op=nl.sigmoid, data=lg[0:T, 0:EA])
    bia = nl.ndarray((P, EA), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=bia[0:T, 0:EA], src=expert_bias.ap(pattern=[[0, T], [1, EA]], offset=0))
    sb = nl.ndarray((P, EA), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(sb[0:T, 0:EA], scr[0:T, 0:EA], bia[0:T, 0:EA], nl.add)
    top_a = nl.ndarray((P, 8), dtype=nl.float32, buffer=nl.sbuf)
    nisa.max8(dst=top_a[0:T, 0:8], src=sb[0:T, 0:EA])
    thr = top_a[0:T, 7:8]
    if top_k > 8:
        # the 9th..16th: the scores at least the 8th largest pushed down out of the way, then the next 8
        above = nl.ndarray((P, EA), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=above[0:T, 0:EA], data=sb[0:T, 0:EA], op0=nl.greater_equal, operand0=top_a[0:T, 7:8])
        rest = nl.ndarray((P, EA), dtype=nl.float32, buffer=nl.sbuf)
        nisa.scalar_tensor_tensor(dst=rest[0:T, 0:EA], data=above[0:T, 0:EA], op0=nl.multiply, operand0=-BIG,
                                  op1=nl.add, operand1=sb[0:T, 0:EA])
        top_b = nl.ndarray((P, 8), dtype=nl.float32, buffer=nl.sbuf)
        nisa.max8(dst=top_b[0:T, 0:8], src=rest[0:T, 0:EA])
        thr = top_b[0:T, top_k - 9:top_k - 8]
    sel = nl.ndarray((P, EA), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=sel[0:T, 0:EA], data=sb[0:T, 0:EA], op0=nl.greater_equal, operand0=thr)
    wsel = nl.ndarray((P, EA), dtype=nl.float32, buffer=nl.sbuf)
    wsum = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(wsel[0:T, 0:EA], scr[0:T, 0:EA], sel[0:T, 0:EA], nl.multiply)
    nisa.tensor_reduce(wsum[0:T, 0:1], op=nl.add, data=wsel[0:T, 0:EA], axis=1, keepdims=True)
    nisa.tensor_scalar(dst=wsum[0:T, 0:1], data=wsum[0:T, 0:1], op0=nl.add, operand0=1e-20)
    winv = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.reciprocal(dst=winv[0:T, 0:1], data=wsum[0:T, 0:1])
    nisa.tensor_scalar(dst=winv[0:T, 0:1], data=winv[0:T, 0:1], op0=nl.multiply, operand0=route_scale)
    # the local experts' affinities: columns rank x E + j, picked by an iota compare
    eid = nl.ndarray((P, EA), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(eid[0:T, 0:EA], pattern=[[1, EA]], offset=0, channel_multiplier=0)
    eb = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=eb[0:T, 0:1], src=rank.ap(pattern=[[0, T], [1, 1]], offset=0))
    nisa.tensor_scalar(dst=eb[0:T, 0:1], data=eb[0:T, 0:1], op0=nl.multiply, operand0=float(E))
    rel = nl.ndarray((P, EA), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=rel[0:T, 0:EA], data=eid[0:T, 0:EA], op0=nl.subtract, operand0=eb[0:T, 0:1])
    aff = nl.ndarray((P, E), dtype=nl.float32, buffer=nl.sbuf)
    for j in range(E):
        oh = nl.ndarray((P, EA), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=oh[0:T, 0:EA], data=rel[0:T, 0:EA], op0=nl.equal, operand0=float(j))
        pk = nl.ndarray((P, EA), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(pk[0:T, 0:EA], wsel[0:T, 0:EA], oh[0:T, 0:EA], nl.multiply)
        nisa.tensor_reduce(aff[0:T, j:j + 1], op=nl.add, data=pk[0:T, 0:EA], axis=1, keepdims=True)
    nisa.tensor_scalar(dst=aff[0:T, 0:E], data=aff[0:T, 0:E], op0=nl.multiply, operand0=winv[0:T, 0:1])

    # ---- pre_expert_proj: latent [T, L] bf16 ----
    lat = nl.ndarray((P, L), dtype=nl.bfloat16, buffer=nl.sbuf)
    pw_ring = [nl.ndarray((P, L), dtype=pre_w.dtype, buffer=nl.sbuf) for _ in range(2)]
    lps = [nl.ndarray((P, 512), dtype=nl.float32, buffer=nl.psum) for _ in range(L // 512)]
    for h1 in range(H1):
        buf = pw_ring[h1 % 2]
        nisa.dma_copy(dst=buf[0:P, 0:L], src=pre_w.ap(pattern=[[L, P], [1, L]], offset=h1 * P * L),
                      dge_mode=nisa.dge_mode.hwdge)
        for c in range(L // 512):
            nisa.nc_matmul(dst=lps[c][0:T, 0:512], stationary=nT[0:P, h1, 0:T], moving=buf[0:P, c * 512:(c + 1) * 512])
    for c in range(L // 512):
        nisa.tensor_copy(dst=lat[0:T, c * 512:(c + 1) * 512], src=lps[c][0:T, 0:512])
    if fp8_act:
        # W8A8: the experts' input, each token's latent row in FP8 at a scale of its own, transposed; 256 rows a
        # matmul pass (double row) against 128 for bf16 activations, at about 3% error a matmul (3 mantissa bits)
        lq, inv_ls = _quantize_rows(lat, T, L)
        lqT = _transpose(lq, T, L1, nl.float8_e4m3)
    else:
        latT = _transpose(lat, T, L1, nl.bfloat16)

    # ---- the local experts (MiMo-V2's moe_half loop at the latent's width) ----
    rows = 2 if fp8_act else 1  # 128-row tiles a matmul pass
    gu_ring = [nl.ndarray((P, rows, 2 * I), dtype=gate_up_w.dtype, buffer=nl.sbuf) for _ in range(RING)]
    dn_ring = [nl.ndarray((P, rows, DN_COLS), dtype=down_w.dtype, buffer=nl.sbuf) for _ in range(RING)]
    acc = nl.ndarray((P, L), dtype=nl.float32, buffer=nl.sbuf)
    slots = [0, 0]
    chunks = [(g, c0, min(512, I - c0)) for g in range(2) for c0 in range(0, I, 512)]
    for e in range(E):
        ps = [nl.ndarray((P, 512), dtype=nl.float32, buffer=nl.psum) for _ in chunks]
        for k in range(n_kl // rows):
            buf = gu_ring[slots[0] % RING]
            slots[0] += 1
            row = l0 + rows * k * P
            t0 = pid * n_kl + rows * k
            nisa.dma_copy(dst=buf[0:P, 0:rows, 0:2 * I],
                          src=gate_up_w.ap(pattern=[[2 * I, P], [P * 2 * I, rows], [1, 2 * I]],
                                           offset=e * (L * 2 * I) + row * 2 * I), dge_mode=nisa.dge_mode.hwdge)
            for ci, (g, c0, cs) in enumerate(chunks):
                if fp8_act:
                    nisa.nc_matmul(dst=ps[ci][0:T, 0:cs], stationary=lqT[0:P, t0:t0 + 2, 0:T],
                                   moving=buf[0:P, 0:2, g * I + c0:g * I + c0 + cs],
                                   perf_mode=nisa.matmul_perf_mode.double_row)
                else:
                    nisa.nc_matmul(dst=ps[ci][0:T, 0:cs], stationary=latT[0:P, t0, 0:T],
                                   moving=buf[0:P, 0, g * I + c0:g * I + c0 + cs])
        part = nl.ndarray((P, 2, I), dtype=nl.float32, buffer=nl.sbuf)
        for ci, (g, c0, cs) in enumerate(chunks):
            if fp8_act:
                nisa.tensor_scalar(dst=part[0:T, g, c0:c0 + cs], data=ps[ci][0:T, 0:cs], op0=nl.multiply,
                                   operand0=inv_ls[0:T, 0:1])
            else:
                nisa.tensor_copy(dst=part[0:T, g, c0:c0 + cs], src=ps[ci][0:T, 0:cs])
        gu = nl.ndarray((P, 2, ILe), dtype=nl.float32, buffer=nl.sbuf)
        if n_prg > 1:
            send = nl.ndarray((P, 2, ILe), dtype=nl.float32, buffer=nl.sbuf)
            recv = nl.ndarray((P, 2, ILe), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=send[0:T, 0:2, 0:ILe], src=part[0:T, 0:2, other * ILe:(other + 1) * ILe])
            nisa.sendrecv(src=send[0:T, 0:2, 0:ILe], dst=recv[0:T, 0:2, 0:ILe], send_to_rank=other,
                          recv_from_rank=other, pipe_id=e % 2)
            nisa.tensor_tensor(gu[0:T, 0:2, 0:ILe], recv[0:T, 0:2, 0:ILe], part[0:T, 0:2, pid * ILe:(pid + 1) * ILe],
                               nl.add)
        else:
            nisa.tensor_copy(dst=gu[0:T, 0:2, 0:ILe], src=part[0:T, 0:2, 0:ILe])
        gus = nl.ndarray((P, 2, ILe), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=gus[0:T, 0:2, 0:ILe],
                      src=gate_up_scale.ap(pattern=[[0, T], [I, 2], [1, ILe]], offset=e * 2 * I + pid * ILe),
                      dge_mode=nisa.dge_mode.hwdge)
        nisa.tensor_tensor(gu[0:T, 0:2, 0:ILe], gu[0:T, 0:2, 0:ILe], gus[0:T, 0:2, 0:ILe], nl.multiply)
        if debug and e == 0:
            gu0 = nl.ndarray((P, 2, ILe), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=gu0[0:T, 0:2, 0:ILe], src=gu[0:T, 0:2, 0:ILe])
        # ops.swiglu: the gate clamped above, the up both ways
        nisa.tensor_scalar(dst=gu[0:T, 0, 0:ILe], data=gu[0:T, 0, 0:ILe], op0=nl.minimum, operand0=limit)
        nisa.tensor_scalar(dst=gu[0:T, 1, 0:ILe], data=gu[0:T, 1, 0:ILe], op0=nl.minimum, operand0=limit,
                           op1=nl.maximum, operand1=-limit)
        g_sb = nl.ndarray((P, ILe), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=g_sb[0:T, 0:ILe], op=nl.silu, data=gu[0:T, 0, 0:ILe])
        it = nl.ndarray((P, ILe), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_tensor(it[0:T, 0:ILe], gu[0:T, 1, 0:ILe], g_sb[0:T, 0:ILe], nl.multiply)
        col = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        if fp8_act:
            iq, inv_ds = _quantize_rows(it, T, ILe)
            nisa.tensor_tensor(col[0:T, 0:1], aff[0:T, e:e + 1], inv_ds[0:T, 0:1], nl.multiply)
            iqT = _transpose(iq, T, n_ic, nl.float8_e4m3)
        else:
            nisa.tensor_copy(dst=col[0:T, 0:1], src=aff[0:T, e:e + 1])
            iqT = _transpose(it, T, n_ic, nl.bfloat16)
        base = e * (I * L)
        for hq in range(L // DN_COLS):
            pd = nl.ndarray((P, DN_COLS), dtype=nl.float32, buffer=nl.psum)
            for j in range(n_ic // rows):
                db = dn_ring[slots[1] % RING]
                slots[1] += 1
                r0 = pid * ILe + rows * j * P
                nisa.dma_copy(dst=db[0:P, 0:rows, 0:DN_COLS],
                              src=down_w.ap(pattern=[[L, P], [P * L, rows], [1, DN_COLS]],
                                            offset=base + r0 * L + hq * DN_COLS), dge_mode=nisa.dge_mode.hwdge)
                if fp8_act:
                    nisa.nc_matmul(dst=pd[0:T, 0:DN_COLS], stationary=iqT[0:P, 2 * j:2 * j + 2, 0:T],
                                   moving=db[0:P, 0:2, 0:DN_COLS], perf_mode=nisa.matmul_perf_mode.double_row)
                else:
                    nisa.nc_matmul(dst=pd[0:T, 0:DN_COLS], stationary=iqT[0:P, j, 0:T], moving=db[0:P, 0, 0:DN_COLS])
            ds = nl.ndarray((P, DN_COLS), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=ds[0:T, 0:DN_COLS], src=down_scale.ap(pattern=[[0, T], [1, DN_COLS]],
                                                                    offset=e * L + hq * DN_COLS))
            c0 = hq * DN_COLS
            scaled = nl.ndarray((P, DN_COLS), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(scaled[0:T, 0:DN_COLS], pd[0:T, 0:DN_COLS], ds[0:T, 0:DN_COLS], nl.multiply)
            if e == 0:
                nisa.tensor_scalar(dst=acc[0:T, c0:c0 + DN_COLS], data=scaled[0:T, 0:DN_COLS], op0=nl.multiply,
                                   operand0=col[0:T, 0:1])
            else:
                nisa.scalar_tensor_tensor(dst=acc[0:T, c0:c0 + DN_COLS], data=scaled[0:T, 0:DN_COLS],
                                          op0=nl.multiply, operand0=col[0:T, 0:1], op1=nl.add,
                                          operand1=acc[0:T, c0:c0 + DN_COLS])

    # the routed latent: the two programs' partials summed, each program keeping its half of the latent
    rt = nl.ndarray((P, LH), dtype=nl.bfloat16, buffer=nl.sbuf)
    if n_prg > 1:
        send_l = nl.ndarray((P, LH), dtype=nl.float32, buffer=nl.sbuf)
        recv_l = nl.ndarray((P, LH), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=send_l[0:T, 0:LH], src=acc[0:T, other * LH:(other + 1) * LH])
        nisa.sendrecv(src=send_l[0:T, 0:LH], dst=recv_l[0:T, 0:LH], send_to_rank=other, recv_from_rank=other,
                      pipe_id=2)
        nisa.tensor_tensor(rt[0:T, 0:LH], recv_l[0:T, 0:LH], acc[0:T, l0:l0 + LH], nl.add)
    else:
        nisa.tensor_copy(dst=rt[0:T, 0:LH], src=acc[0:T, 0:LH])
    rtT = _transpose(rt, T, LH // P, nl.bfloat16)

    # ---- post_expert_proj of this program's latent half + the shared expert, [T, H] ----
    ops_ = [nl.ndarray((P, 512), dtype=nl.float32, buffer=nl.psum) for _ in range(H // 512)]
    po_ring = [nl.ndarray((P, H), dtype=post_w.dtype, buffer=nl.sbuf) for _ in range(2)]
    for k in range(LH // P):
        buf = po_ring[k % 2]
        nisa.dma_copy(dst=buf[0:P, 0:H], src=post_w.ap(pattern=[[H, P], [1, H]], offset=(l0 + k * P) * H),
                      dge_mode=nisa.dge_mode.hwdge)
        for c in range(H // 512):
            nisa.nc_matmul(dst=ops_[c][0:T, 0:512], stationary=rtT[0:P, k, 0:T], moving=buf[0:P, c * 512:(c + 1) * 512])
    out_f = nl.ndarray((P, H), dtype=nl.float32, buffer=nl.sbuf)
    for c in range(H // 512):
        nisa.tensor_copy(dst=out_f[0:T, c * 512:(c + 1) * 512], src=ops_[c][0:T, 0:512])
    if n_prg > 1:
        send_o = nl.ndarray((P, HH), dtype=nl.float32, buffer=nl.sbuf)
        recv_o = nl.ndarray((P, HH), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=send_o[0:T, 0:HH], src=out_f[0:T, other * HH:(other + 1) * HH])
        nisa.sendrecv(src=send_o[0:T, 0:HH], dst=recv_o[0:T, 0:HH], send_to_rank=other, recv_from_rank=other,
                      pipe_id=3)
        nisa.tensor_tensor(out_f[0:T, h0:h0 + HH], out_f[0:T, h0:h0 + HH], recv_o[0:T, 0:HH], nl.add)
    # the routed output rounded to bf16 as the torch path's (routed.to(bf16) @ post, a bf16 result)
    ob = nl.ndarray((P, HH), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=ob[0:T, 0:HH], src=out_f[0:T, h0:h0 + HH])
    # shared expert: gate / up [T, S] over all of H, then this program's half of the output columns
    sw = nl.ndarray((P, H1, S2), dtype=sh_gate_up.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=sw[0:P, 0:H1, 0:S2], src=sh_gate_up.ap(pattern=[[S2, P], [P * S2, H1], [1, S2]], offset=0),
                  dge_mode=nisa.dge_mode.hwdge)
    sp = nl.ndarray((P, S2), dtype=nl.float32, buffer=nl.psum)
    for h1 in range(H1):
        nisa.nc_matmul(dst=sp[0:T, 0:S2], stationary=nT[0:P, h1, 0:T], moving=sw[0:P, h1, 0:S2])
    sgu = nl.ndarray((P, S2), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=sgu[0:T, 0:S2], src=sp[0:T, 0:S2])
    sg = nl.ndarray((P, S), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=sg[0:T, 0:S], op=nl.silu, data=sgu[0:T, 0:S])
    sh = nl.ndarray((P, S), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_tensor(sh[0:T, 0:S], sg[0:T, 0:S], sgu[0:T, S:S2], nl.multiply)
    shT = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.sbuf)
    tps = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.psum)
    nisa.nc_transpose(dst=tps[0:S, 0:T], data=sh[0:T, 0:S])
    nisa.tensor_copy(dst=shT[0:S, 0:T], src=tps[0:S, 0:T])
    sd = nl.ndarray((P, HH), dtype=sh_down.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=sd[0:S, 0:HH], src=sh_down.ap(pattern=[[H, S], [1, HH]], offset=h0))
    o_sb = nl.ndarray((P, HH), dtype=nl.bfloat16, buffer=nl.sbuf)
    for c in range(HH // 512):
        so = nl.ndarray((P, 512), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=so[0:T, 0:512], stationary=shT[0:S, 0:T], moving=sd[0:S, c * 512:(c + 1) * 512])
        sob = nl.ndarray((P, 512), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=sob[0:T, 0:512], src=so[0:T, 0:512])
        nisa.tensor_tensor(o_sb[0:T, c * 512:(c + 1) * 512], ob[0:T, c * 512:(c + 1) * 512], sob[0:T, 0:512], nl.add)

    out = nl.ndarray((T, H), dtype=streams.dtype, buffer=nl.shared_hbm, name="moe_half_out")
    nisa.dma_copy(dst=out.ap(pattern=[[H, T], [1, HH]], offset=h0), src=o_sb[0:T, 0:HH])
    # both programs hold the same post weights: each stores one half of the columns
    post_out = nl.ndarray((T, HC), dtype=nl.float32, buffer=nl.shared_hbm, name="moe_half_post")
    HCH = HC // n_prg
    nisa.dma_copy(dst=post_out.ap(pattern=[[HC, T], [1, HCH]], offset=pid * HCH), src=post[0:T, pid * HCH:(pid + 1) * HCH])
    if debug:
        # the routed latent and the experts' input latent, for tools/hy4_flash/check_moe_half.py
        rt_out = nl.ndarray((T, L), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="moe_half_routed")
        nisa.dma_copy(dst=rt_out.ap(pattern=[[L, T], [1, LH]], offset=l0), src=rt[0:T, 0:LH])
        lat_out = nl.ndarray((T, L), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="moe_half_latent")
        nisa.dma_copy(dst=lat_out, src=lat[0:T, 0:L])
        aff_out = nl.ndarray((T, E), dtype=nl.float32, buffer=nl.shared_hbm, name="moe_half_aff")
        nisa.dma_copy(dst=aff_out, src=aff[0:T, 0:E])
        gu_out = nl.ndarray((T, 2, I), dtype=nl.float32, buffer=nl.shared_hbm, name="moe_half_gu0")
        nisa.dma_copy(dst=gu_out.ap(pattern=[[2 * I, T], [I, 2], [1, ILe]], offset=pid * ILe), src=gu0[0:T, 0:2, 0:ILe])
        return out, post_out, rt_out, lat_out, aff_out, gu_out
    return out, post_out
