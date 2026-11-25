# SPDX-License-Identifier: Apache-2.0
"""Hy4-Flash's iHC mix (ops.hc_mix) of a prefill chunk in one NKI kernel, 128 rows at a time.

The graph's mix contracts the fp32 streams [T, hc H] with fn [2 hc, hc H] in fp32, a matmul 8 columns wide that the
compiler runs on the vector engine; with the streams' rms and the pre-gated sum it was about 200 ms of a 2048-token
chunk's 550 at TP32 (both mixes of all 42 layers). Here each 128-row tile runs kernels/common.hc_mix as the decode
kernels do: the contraction on the tensor engine against fn's bf16 part and its bf16 rest (fp32 accumulation), the
rms and the gates in fp32. The two programs of a logical NeuronCore take alternate tiles.

Folding the previous block's add (ops.hc_add) into the same pass lost: two decoder layers of a 2048-token chunk on
one core ran 164.8 ms with it against 148.0 ms with the graph's add, from 390.9 ms with the graph's mixes.

Inputs: streams [T, hc, H] bf16 (T a multiple of 128); fn_hi, fn_lo [hc H, 2 hc] bf16; hc_base [1, 2 hc] fp32;
hc_scale [1, 2] fp32. Outputs: post [T, hc] fp32 and the block input [T, H] bf16."""

import nki
import nki.isa as nisa
import nki.language as nl

from sglang_neuron.models.hy_v4_flash.kernels.common import P, hc_mix


@nki.jit
def hc_mix_rows(streams, fn_hi, fn_lo, hc_base, hc_scale, eps=1e-5, hc_eps=1e-6, hc_mag=2.0):
    T, HC, H = streams.shape
    assert T % P == 0, f"{T} rows: the kernel takes whole 128-row tiles"
    n_prg = nl.num_programs(axes=0)
    pid = nl.program_id(axis=0)
    post_out = nl.ndarray((T, HC), dtype=nl.float32, buffer=nl.shared_hbm, name="hc_mix_post")
    x_out = nl.ndarray((T, H), dtype=streams.dtype, buffer=nl.shared_hbm, name="hc_mix_x")
    for t in range(pid, T // P, n_prg):
        _, post, xb = hc_mix(streams, fn_hi, fn_lo, hc_base, hc_scale, P, eps, hc_eps, hc_mag, row0=t * P)
        nisa.dma_copy(dst=x_out.ap(pattern=[[H, P], [1, H]], offset=t * P * H), src=xb[0:P, 0:H])
        nisa.dma_copy(dst=post_out.ap(pattern=[[HC, P], [1, HC]], offset=t * P * HC), src=post[0:P, 0:HC])
    return post_out, x_out

