# SPDX-License-Identifier: Apache-2.0
"""Hy3's prefill router in one NKI kernel (moe.HyV3MoE._prefill_router_nki): the dense [T, E] affinities that
moe.hy3_route computes, for a prefill chunk of T rows.

moe.hy3_route runs the router matmul in fp32 (``x.float() @ W.float().t()``), which the Tensor engine runs at a quarter
of its bf16 rate: about 3.2 GFLOP a layer on a 2048-token chunk, the largest of the prefill's torch matmuls. Here the
operands are bf16 (the checkpoint's router is bf16; x is the bf16 normed hidden state) and the products accumulate in
fp32 PSUM, as the decode MoE kernel's router does (kernels/moe_tkg_fused.py, which matched the torch router 19/19 tokens
in serving). Then, on each row: sigmoid, + the selection-only expert bias, the top-k by threshold (the k-th largest of
the biased scores, ``nisa.max8``), the chosen scores, renormalized to one when ``route_norm``, times ``route_scale``.

Top-k by threshold selects every expert whose biased score is >= the k-th largest: an exact tie at the threshold would
select more than k (as in the decode kernel; not seen with fp32 scores).

  x            [T, H] bf16, T a multiple of 128 x the LNC programs
  router_wT    [H, E] bf16, the router weight transposed (moe_fused.make builds it)
  expert_bias  [1, E] fp32
  top_k (<= 8), route_norm, route_scale: compile-time constants
Returns [T, E] fp32 affinities, zero off each row's chosen experts. Each LNC program takes every other 128-row tile.
"""

import nki
import nki.isa as nisa
import nki.language as nl

P = 128


@nki.jit
def router_cte(x, router_wT, expert_bias, top_k=8, route_norm=True, route_scale=1.0):
    T, H = x.shape
    E = router_wT.shape[1]
    H1 = H // P
    n_prg = nl.num_programs(axes=0)
    pid = nl.program_id(axis=0)
    tiles = T // P // n_prg

    out = nl.ndarray((T, E), dtype=nl.float32, buffer=nl.shared_hbm)

    # the router weight and the bias, once for every tile
    rw = nl.ndarray((P, H1, E), dtype=router_wT.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=rw[0:P, 0:H1, 0:E], src=router_wT.ap(pattern=[[E, P], [P * E, H1], [1, E]], offset=0))
    bia = nl.ndarray((P, E), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=bia[0:P, 0:E], src=expert_bias.ap(pattern=[[0, P], [1, E]], offset=0))

    for i in range(tiles):
        r0 = (i * n_prg + pid) * P
        xs = nl.ndarray((P, H), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=xs[0:P, 0:H], src=x.ap(pattern=[[H, P], [1, H]], offset=r0 * H))
        xT = nl.ndarray((P, H1, P), dtype=nl.bfloat16, buffer=nl.sbuf)
        for h1 in range(H1):
            tp = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.psum)
            nisa.nc_transpose(dst=tp[0:P, 0:P], data=xs[0:P, h1 * P:(h1 + 1) * P])
            nisa.tensor_copy(dst=xT[0:P, h1, 0:P], src=tp[0:P, 0:P])

        # logits [P, E] = x @ W^T, bf16 operands, fp32 accumulation over H
        lg = nl.ndarray((P, E), dtype=nl.float32, buffer=nl.psum)
        for h1 in range(H1):
            nisa.nc_matmul(dst=lg[0:P, 0:E], stationary=xT[0:P, h1, 0:P], moving=rw[0:P, h1, 0:E])
        scr = nl.ndarray((P, E), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=scr[0:P, 0:E], op=nl.sigmoid, data=lg[0:P, 0:E])
        sb = nl.ndarray((P, E), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(sb[0:P, 0:E], scr[0:P, 0:E], bia[0:P, 0:E], nl.add)

        # the k-th largest biased score is the selection threshold
        top8 = nl.ndarray((P, 8), dtype=nl.float32, buffer=nl.sbuf)
        nisa.max8(dst=top8[0:P, 0:8], src=sb[0:P, 0:E])
        sel = nl.ndarray((P, E), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=sel[0:P, 0:E], data=sb[0:P, 0:E], op0=nl.greater_equal,
                           operand0=top8[0:P, top_k - 1:top_k])
        # the chosen (unbiased) scores
        wsel = nl.ndarray((P, E), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(wsel[0:P, 0:E], scr[0:P, 0:E], sel[0:P, 0:E], nl.multiply)
        aff = nl.ndarray((P, E), dtype=nl.float32, buffer=nl.sbuf)
        if route_norm:
            wsum = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(wsum[0:P, 0:1], op=nl.add, data=wsel[0:P, 0:E], axis=1, keepdims=True)
            nisa.tensor_scalar(dst=wsum[0:P, 0:1], data=wsum[0:P, 0:1], op0=nl.add, operand0=1e-20)
            winv = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.reciprocal(dst=winv[0:P, 0:1], data=wsum[0:P, 0:1])
            nisa.tensor_scalar(dst=winv[0:P, 0:1], data=winv[0:P, 0:1], op0=nl.multiply, operand0=route_scale)
            nisa.tensor_scalar(dst=aff[0:P, 0:E], data=wsel[0:P, 0:E], op0=nl.multiply, operand0=winv[0:P, 0:1])
        else:
            nisa.tensor_scalar(dst=aff[0:P, 0:E], data=wsel[0:P, 0:E], op0=nl.multiply, operand0=route_scale)
        nisa.dma_copy(dst=out.ap(pattern=[[E, P], [1, E]], offset=r0 * E), src=aff[0:P, 0:E])
    return out
