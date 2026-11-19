# SPDX-License-Identifier: Apache-2.0
# Derived from Hy3's decode MoE block kernel (models/hy_v3/kernels/moe_tkg_fused.py, from the vllm-neuron Hy3 contrib,
# qingzwang/vllm-neuron#8); kernels/PROVENANCE.md lists the changes.
"""MiMo-V2's decode MoE half in one NKI kernel: from the residual stream after attention to this rank's partial of
the MoE output, with no XLA op in between.

Per layer and rank, for T <= 128 decode tokens:
  post-attention RMSNorm (fp32 math, bf16 result, as the torch path)
  -> router: bf16 logits with fp32 accumulation, sigmoid, + the selection-only bias, top 8 by the 8th largest biased
     score (``nisa.max8``), the chosen unbiased scores renormalized, x routed_scaling_factor; each affinity rounded
     through bf16 as the torch path's (moe.MiMoV2MoE.forward)
  -> this rank's local experts, all of them, each weighted by its affinity: weight-only FP8 (bf16 activations x FP8
     weights, fp32 accumulation, one matmul pass per 128 rows), the power-of-two per-output-row scales of
     experts_fp8.py applied after each matmul
The output is the rank's partial [T, H]; the caller all-reduces it.

The two programs of the logical NeuronCore each contract one half of H for gate / up and exchange the halves of their
partial sums (``sendrecv``); each keeps one half of I for the activation and the down projection, and a last exchange
sums the two halves of the output, as the Hy3 kernel does.

Inputs:
  hidden         [T, H] bf16, the residual stream after attention, before the norm
  ln_w           [1, H] bf16, post_attention_layernorm's weight
  router_wT      [H, EA] bf16, the router weight transposed
  expert_bias    [1, EA] fp32, the selection-only bias (e_score_correction_bias)
  ebase          [1, 1] fp32, this rank's first local expert id, or its EP rank [1] int32 (times E here)
  gate_up_w      [E, H, 2, I] fp8, down_w [E, I, H] fp8, the local experts
  gate_up_scale  [E, 2, I] fp32, down_scale [E, H] fp32, their per-output-row scales
  eps, route_scale, top_k: compile-time constants

Top-k by threshold: the experts whose biased score is at least the k-th largest are selected; an exact tie at the
threshold would select more than k (the Hy3 kernel's note: not seen with fp32 scores)."""

import nki
import nki.isa as nisa
import nki.language as nl

P = 128
RING = 6
DN_COLS = 1024
FP8_MAX = 240.0


def moe_prefetch(gate_up_w, n: int):
    """The first ``n`` gate / up weight tiles moe_half's W8A8 loop reads (expert 0 first, two 128-row tiles of this
    program's half of H a tile), DMA'd now into buffers of their own: a caller issues this before work the experts do
    not depend on (layer.py: before the attention and its all-reduce, while the DMA queues are idle) and hands the
    tiles to moe_half (``pre``). Splitting gate / up over I instead (no exchange between the programs) measured slower:
    its weight rows load in 1 KB runs."""
    E, H, _, I = gate_up_w.shape  # noqa: E741
    n_prg = nl.num_programs(axes=0)
    pid = nl.program_id(axis=0)
    n_k = H // P // n_prg
    h0 = pid * (H // n_prg)
    tiles = []
    for t in range(n):
        e, k = t // (n_k // 2), t % (n_k // 2)
        if e >= E:
            break
        buf = nl.ndarray((P, 2, 2 * I), dtype=gate_up_w.dtype, buffer=nl.sbuf)
        row = h0 + 2 * k * P
        nisa.dma_copy(dst=buf[0:P, 0:2, 0 : 2 * I],
                      src=gate_up_w.ap(pattern=[[2 * I, P], [P * 2 * I, 2], [1, 2 * I]],
                                       offset=e * (H * 2 * I) + row * 2 * I), dge_mode=nisa.dge_mode.hwdge)
        tiles.append(buf)
    return tiles


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
    # a zero row: scale 1 (its quantized row is zero either way)
    nisa.tensor_scalar(dst=amax[0:T, 0:1], data=amax[0:T, 0:1], op0=nl.maximum, operand0=1e-30)
    inv = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=inv[0:T, 0:1], data=amax[0:T, 0:1], op0=nl.multiply, operand0=1.0 / FP8_MAX)
    sc = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.reciprocal(dst=sc[0:T, 0:1], data=inv[0:T, 0:1])
    return sc, inv


@nki.jit
def moe_half(hidden, ln_w, router_wT, expert_bias, ebase, gate_up_w, down_w, gate_up_scale, down_scale, eps=1e-6,
             route_scale=1.0, top_k=8, hidden_sb=None, out_in_sbuf=False, fp8_act=True, pre=None):
    """Inside another kernel (layer.py) ``hidden_sb`` holds the rows in SBUF (``hidden`` then gives the shape only),
    and ``out_in_sbuf`` returns this program's half of H of the partial in SBUF (rows 0 .. T - 1 of a [128, H /
    programs] tile) in place of the HBM output. ``fp8_act`` (the default) quantizes each token's activations to FP8
    with a scale of its own (W8A8, dynamic per token: 240 / the row's absolute max, the expert input over the whole row,
    the down input over this program's half of I) so the expert matmuls contract 256 rows a pass
    (``matmul_perf_mode.double_row``); bf16 activations allow 128, which leaves the TensorE streaming each expert weight
    column twice as long (a decode step's MoE layers were TensorE-bound: tools/mimo_v2 profile, 2026-11-19)."""
    T, H = hidden.shape
    E, _, _, I = gate_up_w.shape  # noqa: E741 - the expert intermediate size, as the kernels name it
    EA = router_wT.shape[1]
    H1 = H // P
    n_prg = nl.num_programs(axes=0)
    pid = nl.program_id(axis=0)
    other = 1 - pid
    n_k = H1 // n_prg
    HH = H // n_prg
    h0 = pid * HH
    ILe = I // n_prg
    n_ic = ILe // P

    # ---- RMSNorm (fp32 math, bf16 result like the torch path) ----
    hf = nl.ndarray((P, H), dtype=nl.bfloat16, buffer=nl.sbuf)
    if hidden_sb is None:
        nisa.dma_copy(dst=hf[0:T, 0:H], src=hidden.ap(pattern=[[H, T], [1, H]], offset=0))
    else:
        nisa.tensor_copy(dst=hf[0:T, 0:H], src=hidden_sb[0:T, 0:H])
    gam = nl.ndarray((P, H), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=gam[0:T, 0:H], src=ln_w.ap(pattern=[[0, T], [1, H]], offset=0))
    sq = nl.ndarray((P, H), dtype=nl.float32, buffer=nl.sbuf)
    ss = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=sq[0:T, 0:H], op=nl.square, data=hf[0:T, 0:H], reduce_op=nl.add,
                    reduce_res=ss[0:T, 0:1], reduce_cmd=nisa.reduce_cmd.reset_reduce)
    rr = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=rr[0:T, 0:1], op=nl.rsqrt, data=ss[0:T, 0:1], scale=1.0 / H, bias=eps)
    nb = nl.ndarray((P, H), dtype=nl.bfloat16, buffer=nl.sbuf)
    # x * rsqrt(mean(x^2) + eps) rounded to bf16, then times the weight (models/transformer/model.RMSNorm)
    nisa.tensor_scalar(dst=nb[0:T, 0:H], data=hf[0:T, 0:H], op0=nl.multiply, operand0=rr[0:T, 0:1])
    nrm = nl.ndarray((P, H), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_tensor(nrm[0:T, 0:H], nb[0:T, 0:H], gam[0:T, 0:H], nl.multiply)
    nT = nl.ndarray((P, H1, P), dtype=nl.bfloat16, buffer=nl.sbuf)
    for h1 in range(H1):
        tp = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.psum)
        nisa.nc_transpose(dst=tp[0:P, 0:T], data=nrm[0:T, h1 * P : (h1 + 1) * P])
        nisa.tensor_copy(dst=nT[0:P, h1, 0:T], src=tp[0:P, 0:T])

    if fp8_act:
        # the experts' input: each token's row times 240 / its absolute max, in FP8, transposed like nT
        xs, inv_xs = _token_scales(nrm, T, H)
        nq = nl.ndarray((P, H), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=nq[0:T, 0:H], data=nrm[0:T, 0:H], op0=nl.multiply, operand0=xs[0:T, 0:1],
                           op1=nl.minimum, operand1=FP8_MAX)
        nisa.tensor_scalar(dst=nq[0:T, 0:H], data=nq[0:T, 0:H], op0=nl.maximum, operand0=-FP8_MAX)
        xq = nl.ndarray((P, H1, P), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        for h1 in range(H1):
            tq = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.psum)
            nisa.nc_transpose(dst=tq[0:P, 0:T], data=nq[0:T, h1 * P : (h1 + 1) * P])
            nisa.tensor_copy(dst=xq[0:P, h1, 0:T], src=tq[0:P, 0:T])

    # ---- router: logits [T, EA] = normed @ W^T (bf16 x bf16, fp32 accumulate) ----
    rw = nl.ndarray((P, H1, EA), dtype=router_wT.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=rw[0:P, 0:H1, 0:EA], src=router_wT.ap(pattern=[[EA, P], [P * EA, H1], [1, EA]], offset=0), dge_mode=nisa.dge_mode.hwdge)
    lg = nl.ndarray((P, EA), dtype=nl.float32, buffer=nl.psum)
    for h1 in range(H1):
        nisa.nc_matmul(dst=lg[0:T, 0:EA], stationary=nT[0:P, h1, 0:T], moving=rw[0:P, h1, 0:EA])
    scr = nl.ndarray((P, EA), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=scr[0:T, 0:EA], op=nl.sigmoid, data=lg[0:T, 0:EA])
    bia = nl.ndarray((P, EA), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=bia[0:T, 0:EA], src=expert_bias.ap(pattern=[[0, T], [1, EA]], offset=0), dge_mode=nisa.dge_mode.hwdge)
    sb = nl.ndarray((P, EA), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(sb[0:T, 0:EA], scr[0:T, 0:EA], bia[0:T, 0:EA], nl.add)
    top8 = nl.ndarray((P, 8), dtype=nl.float32, buffer=nl.sbuf)
    nisa.max8(dst=top8[0:T, 0:8], src=sb[0:T, 0:EA])
    sel = nl.ndarray((P, EA), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=sel[0:T, 0:EA], data=sb[0:T, 0:EA], op0=nl.greater_equal,
                       operand0=top8[0:T, top_k - 1 : top_k])
    wsel = nl.ndarray((P, EA), dtype=nl.float32, buffer=nl.sbuf)
    wsum = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(wsel[0:T, 0:EA], scr[0:T, 0:EA], sel[0:T, 0:EA], nl.multiply)
    nisa.tensor_reduce(wsum[0:T, 0:1], op=nl.add, data=wsel[0:T, 0:EA], axis=1, keepdims=True)
    nisa.tensor_scalar(dst=wsum[0:T, 0:1], data=wsum[0:T, 0:1], op0=nl.add, operand0=1e-20)
    winv = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.reciprocal(dst=winv[0:T, 0:1], data=wsum[0:T, 0:1])
    nisa.tensor_scalar(dst=winv[0:T, 0:1], data=winv[0:T, 0:1], op0=nl.multiply, operand0=route_scale)
    # the local experts' affinities: one-hot columns ebase + j, picked by an iota compare
    eid = nl.ndarray((P, EA), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(eid[0:T, 0:EA], pattern=[[1, EA]], offset=0, channel_multiplier=0)
    eb = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    if ebase.dtype == nl.float32:
        nisa.dma_copy(dst=eb[0:T, 0:1], src=ebase.ap(pattern=[[0, T], [1, 1]], offset=0))
    else:
        # the EP rank [1] int32: its first expert is rank x E
        eb_i = nl.ndarray((P, 1), dtype=ebase.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=eb_i[0:T, 0:1], src=ebase.reshape((1, 1)).ap(pattern=[[0, T], [1, 1]], offset=0))
        nisa.tensor_copy(dst=eb[0:T, 0:1], src=eb_i[0:T, 0:1])
        nisa.tensor_scalar(dst=eb[0:T, 0:1], data=eb[0:T, 0:1], op0=nl.multiply, operand0=float(E))
    rel = nl.ndarray((P, EA), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=rel[0:T, 0:EA], data=eid[0:T, 0:EA], op0=nl.subtract, operand0=eb[0:T, 0:1])
    aff = nl.ndarray((P, E), dtype=nl.float32, buffer=nl.sbuf)
    for j in range(E):
        oh = nl.ndarray((P, EA), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=oh[0:T, 0:EA], data=rel[0:T, 0:EA], op0=nl.equal, operand0=float(j))
        pk = nl.ndarray((P, EA), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(pk[0:T, 0:EA], wsel[0:T, 0:EA], oh[0:T, 0:EA], nl.multiply)
        nisa.tensor_reduce(aff[0:T, j : j + 1], op=nl.add, data=pk[0:T, 0:EA], axis=1, keepdims=True)
    nisa.tensor_scalar(dst=aff[0:T, 0:E], data=aff[0:T, 0:E], op0=nl.multiply, operand0=winv[0:T, 0:1])
    # rounded through bf16, as the torch path's affinities.to(x.dtype).float()
    aff_b = nl.ndarray((P, E), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=aff_b[0:T, 0:E], src=aff[0:T, 0:E])
    nisa.tensor_copy(dst=aff[0:T, 0:E], src=aff_b[0:T, 0:E])

    if fp8_act:
        gu2_ring = [nl.ndarray((P, 2, 2 * I), dtype=gate_up_w.dtype, buffer=nl.sbuf) for _ in range(RING)]
        dn2_ring = [nl.ndarray((P, 2, DN_COLS), dtype=down_w.dtype, buffer=nl.sbuf) for _ in range(RING)]
    else:
        gu_ring = [nl.ndarray((P, 2 * I), dtype=gate_up_w.dtype, buffer=nl.sbuf) for _ in range(RING)]
        dn_ring = [nl.ndarray((P, DN_COLS), dtype=down_w.dtype, buffer=nl.sbuf) for _ in range(RING)]
    acc = nl.ndarray((P, H), dtype=nl.float32, buffer=nl.sbuf)
    slots = [0, 0]
    chunks = [(g, c0, min(512, I - c0)) for g in range(2) for c0 in range(0, I, 512)]

    for e in range(E):
        # gate / up over this program's half of H: one 128-row pass per row tile
        ps = [nl.ndarray((P, 512), dtype=nl.float32, buffer=nl.psum) for _ in chunks]
        if fp8_act:
            # two 128-row tiles of H a pass (double row)
            for k in range(n_k // 2):
                tile = e * (n_k // 2) + k
                row = h0 + 2 * k * P
                if pre is not None and tile < len(pre):
                    buf = pre[tile]  # loaded ahead (moe_prefetch)
                else:
                    buf = gu2_ring[slots[0] % RING]
                    slots[0] += 1
                    nisa.dma_copy(dst=buf[0:P, 0:2, 0 : 2 * I],
                                  src=gate_up_w.ap(pattern=[[2 * I, P], [P * 2 * I, 2], [1, 2 * I]],
                                                   offset=e * (H * 2 * I) + row * 2 * I), dge_mode=nisa.dge_mode.hwdge)
                for ci, (g, c0, cs) in enumerate(chunks):
                    nisa.nc_matmul(dst=ps[ci][0:T, 0:cs], stationary=xq[0:P, pid * n_k + 2 * k:pid * n_k + 2 * k + 2, 0:T],
                                   moving=buf[0:P, 0:2, g * I + c0 : g * I + c0 + cs],
                                   perf_mode=nisa.matmul_perf_mode.double_row)
        else:
            for k in range(n_k):
                buf = gu_ring[slots[0] % RING]
                slots[0] += 1
                row = h0 + k * P
                nisa.dma_copy(dst=buf[0:P, 0 : 2 * I],
                              src=gate_up_w.ap(pattern=[[2 * I, P], [1, 2 * I]], offset=e * (H * 2 * I) + row * 2 * I), dge_mode=nisa.dge_mode.hwdge)
                for ci, (g, c0, cs) in enumerate(chunks):
                    nisa.nc_matmul(dst=ps[ci][0:T, 0:cs], stationary=nT[0:P, pid * n_k + k, 0:T],
                                   moving=buf[0:P, g * I + c0 : g * I + c0 + cs])
        part = nl.ndarray((P, 2, I), dtype=nl.float32, buffer=nl.sbuf)
        for ci, (g, c0, cs) in enumerate(chunks):
            if fp8_act:
                # back to the activations' scale: each token's row / its scale
                nisa.tensor_scalar(dst=part[0:T, g, c0 : c0 + cs], data=ps[ci][0:T, 0:cs], op0=nl.multiply,
                                   operand0=inv_xs[0:T, 0:1])
            else:
                nisa.tensor_copy(dst=part[0:T, g, c0 : c0 + cs], src=ps[ci][0:T, 0:cs])
        send = nl.ndarray((P, 2, ILe), dtype=nl.float32, buffer=nl.sbuf)
        recv = nl.ndarray((P, 2, ILe), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=send[0:T, 0:2, 0:ILe], src=part[0:T, 0:2, other * ILe : (other + 1) * ILe])
        nisa.sendrecv(src=send[0:T, 0:2, 0:ILe], dst=recv[0:T, 0:2, 0:ILe],
                      send_to_rank=other, recv_from_rank=other, pipe_id=e % 2)
        gu = nl.ndarray((P, 2, ILe), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(gu[0:T, 0:2, 0:ILe], recv[0:T, 0:2, 0:ILe],
                           part[0:T, 0:2, pid * ILe : (pid + 1) * ILe], nl.add)
        # the per-row (output column) scales of this program's half of I
        gus = nl.ndarray((P, 2, ILe), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=gus[0:T, 0:2, 0:ILe],
                      src=gate_up_scale.ap(pattern=[[0, T], [I, 2], [1, ILe]], offset=e * 2 * I + pid * ILe), dge_mode=nisa.dge_mode.hwdge)
        nisa.tensor_tensor(gu[0:T, 0:2, 0:ILe], gu[0:T, 0:2, 0:ILe], gus[0:T, 0:2, 0:ILe], nl.multiply)

        g_sb = nl.ndarray((P, ILe), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=g_sb[0:T, 0:ILe], op=nl.silu, data=gu[0:T, 0, 0:ILe])
        it = nl.ndarray((P, ILe), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_tensor(it[0:T, 0:ILe], gu[0:T, 1, 0:ILe], g_sb[0:T, 0:ILe], nl.multiply)
        if fp8_act:
            # the down input in FP8, each token's half row times 240 / its absolute max
            ds_, inv_ds = _token_scales(it, T, ILe)
            nisa.tensor_scalar(dst=it[0:T, 0:ILe], data=it[0:T, 0:ILe], op0=nl.multiply, operand0=ds_[0:T, 0:1],
                               op1=nl.minimum, operand1=FP8_MAX)
            nisa.tensor_scalar(dst=it[0:T, 0:ILe], data=it[0:T, 0:ILe], op0=nl.maximum, operand0=-FP8_MAX)
            # the expert's weight on each row, its activation scale folded in
            col = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(col[0:T, 0:1], aff[0:T, e : e + 1], inv_ds[0:T, 0:1], nl.multiply)
            itT = nl.ndarray((P, n_ic, P), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        else:
            col = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=col[0:T, 0:1], src=aff[0:T, e : e + 1])
            itT = nl.ndarray((P, n_ic, P), dtype=nl.bfloat16, buffer=nl.sbuf)
        for c in range(n_ic):
            tp2 = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.psum)
            nisa.nc_transpose(dst=tp2[0:P, 0:T], data=it[0:T, c * P : (c + 1) * P])
            nisa.tensor_copy(dst=itT[0:P, c, 0:T], src=tp2[0:P, 0:T])

        base = e * (I * H)
        for hq in range(H // DN_COLS):
            pd = [nl.ndarray((P, 512), dtype=nl.float32, buffer=nl.psum) for _ in range(DN_COLS // 512)]
            if fp8_act:
                for j in range(n_ic // 2):
                    db = dn2_ring[slots[1] % RING]
                    slots[1] += 1
                    r0 = pid * ILe + 2 * j * P
                    nisa.dma_copy(dst=db[0:P, 0:2, 0:DN_COLS],
                                  src=down_w.ap(pattern=[[H, P], [P * H, 2], [1, DN_COLS]],
                                                offset=base + r0 * H + hq * DN_COLS), dge_mode=nisa.dge_mode.hwdge)
                    for b in range(DN_COLS // 512):
                        nisa.nc_matmul(dst=pd[b][0:T, 0:512], stationary=itT[0:P, 2 * j:2 * j + 2, 0:T],
                                       moving=db[0:P, 0:2, b * 512 : (b + 1) * 512],
                                       perf_mode=nisa.matmul_perf_mode.double_row)
            else:
                for j in range(n_ic):
                    db = dn_ring[slots[1] % RING]
                    slots[1] += 1
                    r0 = pid * ILe + j * P
                    nisa.dma_copy(dst=db[0:P, 0:DN_COLS],
                                  src=down_w.ap(pattern=[[H, P], [1, DN_COLS]], offset=base + r0 * H + hq * DN_COLS), dge_mode=nisa.dge_mode.hwdge)
                    for b in range(DN_COLS // 512):
                        nisa.nc_matmul(dst=pd[b][0:T, 0:512], stationary=itT[0:P, j, 0:T],
                                       moving=db[0:P, b * 512 : (b + 1) * 512])
            ds = nl.ndarray((P, DN_COLS), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=ds[0:T, 0:DN_COLS],
                          src=down_scale.ap(pattern=[[0, T], [1, DN_COLS]], offset=e * H + hq * DN_COLS), dge_mode=nisa.dge_mode.hwdge)
            for b in range(DN_COLS // 512):
                c0 = hq * DN_COLS + b * 512
                scaled = nl.ndarray((P, 512), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(scaled[0:T, 0:512], pd[b][0:T, 0:512], ds[0:T, b * 512 : (b + 1) * 512],
                                   nl.multiply)
                if e == 0:
                    nisa.tensor_scalar(dst=acc[0:T, c0 : c0 + 512], data=scaled[0:T, 0:512],
                                       op0=nl.multiply, operand0=col[0:T, 0:1])
                else:
                    nisa.scalar_tensor_tensor(
                        dst=acc[0:T, c0 : c0 + 512], data=scaled[0:T, 0:512],
                        op0=nl.multiply, operand0=col[0:T, 0:1],
                        op1=nl.add, operand1=acc[0:T, c0 : c0 + 512],
                    )

    # sum the two programs' partials: each owns one half of H
    send_o = nl.ndarray((P, HH), dtype=nl.float32, buffer=nl.sbuf)
    recv_o = nl.ndarray((P, HH), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=send_o[0:T, 0:HH], src=acc[0:T, other * HH : (other + 1) * HH])
    nisa.sendrecv(src=send_o[0:T, 0:HH], dst=recv_o[0:T, 0:HH],
                  send_to_rank=other, recv_from_rank=other, pipe_id=2)
    o_sb = nl.ndarray((P, HH), dtype=hidden.dtype, buffer=nl.sbuf)
    nisa.tensor_tensor(o_sb[0:T, 0:HH], recv_o[0:T, 0:HH], acc[0:T, h0 : h0 + HH], nl.add)
    if out_in_sbuf:
        return o_sb
    out = nl.ndarray((T, H), dtype=hidden.dtype, buffer=nl.shared_hbm, name="moe_half_out")
    nisa.dma_copy(dst=out.ap(pattern=[[H, T], [1, HH]], offset=h0), src=o_sb[0:T, 0:HH])
    return out
