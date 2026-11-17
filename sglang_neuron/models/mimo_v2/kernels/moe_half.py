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
  ebase          [1, 1] fp32, this rank's first local expert id
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


@nki.jit
def moe_half(hidden, ln_w, router_wT, expert_bias, ebase, gate_up_w, down_w, gate_up_scale, down_scale, eps=1e-6,
             route_scale=1.0, top_k=8):
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

    out = nl.ndarray((T, H), dtype=hidden.dtype, buffer=nl.shared_hbm, name="moe_half_out")

    # ---- RMSNorm (fp32 math, bf16 result like the torch path) ----
    hf = nl.ndarray((P, H), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=hf[0:T, 0:H], src=hidden.ap(pattern=[[H, T], [1, H]], offset=0))
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

    # ---- router: logits [T, EA] = normed @ W^T (bf16 x bf16, fp32 accumulate) ----
    rw = nl.ndarray((P, H1, EA), dtype=router_wT.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=rw[0:P, 0:H1, 0:EA], src=router_wT.ap(pattern=[[EA, P], [P * EA, H1], [1, EA]], offset=0))
    lg = nl.ndarray((P, EA), dtype=nl.float32, buffer=nl.psum)
    for h1 in range(H1):
        nisa.nc_matmul(dst=lg[0:T, 0:EA], stationary=nT[0:P, h1, 0:T], moving=rw[0:P, h1, 0:EA])
    scr = nl.ndarray((P, EA), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=scr[0:T, 0:EA], op=nl.sigmoid, data=lg[0:T, 0:EA])
    bia = nl.ndarray((P, EA), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=bia[0:T, 0:EA], src=expert_bias.ap(pattern=[[0, T], [1, EA]], offset=0))
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
    nisa.dma_copy(dst=eb[0:T, 0:1], src=ebase.ap(pattern=[[0, T], [1, 1]], offset=0))
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

    gu_ring = [nl.ndarray((P, 2 * I), dtype=gate_up_w.dtype, buffer=nl.sbuf) for _ in range(RING)]
    dn_ring = [nl.ndarray((P, DN_COLS), dtype=down_w.dtype, buffer=nl.sbuf) for _ in range(RING)]
    acc = nl.ndarray((P, H), dtype=nl.float32, buffer=nl.sbuf)
    slots = [0, 0]
    chunks = [(g, c0, min(512, I - c0)) for g in range(2) for c0 in range(0, I, 512)]

    for e in range(E):
        # gate / up over this program's half of H: one 128-row pass per row tile
        ps = [nl.ndarray((P, 512), dtype=nl.float32, buffer=nl.psum) for _ in chunks]
        for k in range(n_k):
            buf = gu_ring[slots[0] % RING]
            slots[0] += 1
            row = h0 + k * P
            nisa.dma_copy(dst=buf[0:P, 0 : 2 * I],
                          src=gate_up_w.ap(pattern=[[2 * I, P], [1, 2 * I]], offset=e * (H * 2 * I) + row * 2 * I))
            for ci, (g, c0, cs) in enumerate(chunks):
                nisa.nc_matmul(dst=ps[ci][0:T, 0:cs], stationary=nT[0:P, pid * n_k + k, 0:T],
                               moving=buf[0:P, g * I + c0 : g * I + c0 + cs])
        part = nl.ndarray((P, 2, I), dtype=nl.float32, buffer=nl.sbuf)
        for ci, (g, c0, cs) in enumerate(chunks):
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
                      src=gate_up_scale.ap(pattern=[[0, T], [I, 2], [1, ILe]], offset=e * 2 * I + pid * ILe))
        nisa.tensor_tensor(gu[0:T, 0:2, 0:ILe], gu[0:T, 0:2, 0:ILe], gus[0:T, 0:2, 0:ILe], nl.multiply)

        g_sb = nl.ndarray((P, ILe), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=g_sb[0:T, 0:ILe], op=nl.silu, data=gu[0:T, 0, 0:ILe])
        it = nl.ndarray((P, ILe), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_tensor(it[0:T, 0:ILe], gu[0:T, 1, 0:ILe], g_sb[0:T, 0:ILe], nl.multiply)
        itT = nl.ndarray((P, n_ic, P), dtype=nl.bfloat16, buffer=nl.sbuf)
        for c in range(n_ic):
            tp2 = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.psum)
            nisa.nc_transpose(dst=tp2[0:P, 0:T], data=it[0:T, c * P : (c + 1) * P])
            nisa.tensor_copy(dst=itT[0:P, c, 0:T], src=tp2[0:P, 0:T])

        base = e * (I * H)
        for hq in range(H // DN_COLS):
            pd = [nl.ndarray((P, 512), dtype=nl.float32, buffer=nl.psum) for _ in range(DN_COLS // 512)]
            for j in range(n_ic):
                db = dn_ring[slots[1] % RING]
                slots[1] += 1
                r0 = pid * ILe + j * P
                nisa.dma_copy(dst=db[0:P, 0:DN_COLS],
                              src=down_w.ap(pattern=[[H, P], [1, DN_COLS]], offset=base + r0 * H + hq * DN_COLS))
                for b in range(DN_COLS // 512):
                    nisa.nc_matmul(dst=pd[b][0:T, 0:512], stationary=itT[0:P, j, 0:T],
                                   moving=db[0:P, b * 512 : (b + 1) * 512])
            ds = nl.ndarray((P, DN_COLS), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=ds[0:T, 0:DN_COLS],
                          src=down_scale.ap(pattern=[[0, T], [1, DN_COLS]], offset=e * H + hq * DN_COLS))
            for b in range(DN_COLS // 512):
                c0 = hq * DN_COLS + b * 512
                scaled = nl.ndarray((P, 512), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(scaled[0:T, 0:512], pd[b][0:T, 0:512], ds[0:T, b * 512 : (b + 1) * 512],
                                   nl.multiply)
                if e == 0:
                    nisa.tensor_scalar(dst=acc[0:T, c0 : c0 + 512], data=scaled[0:T, 0:512],
                                       op0=nl.multiply, operand0=aff[0:T, e : e + 1])
                else:
                    nisa.scalar_tensor_tensor(
                        dst=acc[0:T, c0 : c0 + 512], data=scaled[0:T, 0:512],
                        op0=nl.multiply, operand0=aff[0:T, e : e + 1],
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
    nisa.dma_copy(dst=out.ap(pattern=[[H, T], [1, HH]], offset=h0), src=o_sb[0:T, 0:HH])
    return out
