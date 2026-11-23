# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A one-token decode's routed experts on this rank: only the local experts the token chose, in a runtime loop.

Adapted from nkilib's selective-expert moe_tkg (``nkilib.core.moe.moe_tkg.selective_expert_impl``), whose loop runs
all top-k slots of a token at trace time. At EP32 a token's 16 experts are about half an expert a rank, but the
all-expert kernel reads all 16 local experts' weights (113 MB of FP8 a layer, about 0.6 ms), and the selective one
reads 16 experts too. Here the host side packs the token's local experts first with their count, and the kernel runs
``count`` iterations of ``nl.fori_loop``: each computes slot 0's expert, adds its affinity-weighted output, and shifts
the slots and affinities one to the left, so every access pattern is static."""

from __future__ import annotations

import nki
import nki.isa as nisa
import nki.language as nl
from nkilib.core.moe.moe_tkg.mlp_parameters import MLPBiasParameters, MLPExpertParameters, MLPParameters
from nkilib.core.moe.moe_tkg.mlp_tkg_constants import MLPTKGConstants
from nkilib.core.moe.moe_tkg.mlp_tkg_down_projection import process_down_projection
from nkilib.core.moe.moe_tkg.mlp_tkg_gate_up_projection import process_gate_up_projection
from nkilib.core.moe.moe_tkg.projection_utils import input_norm_load, transpose_store
from nkilib.core.moe.moe_tkg.selective_expert_impl import _select_quant_scales
from nkilib.core.utils.allocator import SbufManager
from nkilib.core.utils.common_types import ActFnType, ExpertAffinityScaleMode, GateUpDim, NormType, QuantizationType
from nkilib.core.utils.kernel_helpers import div_ceil
from nkilib.core.utils.logging import get_logger
from nkilib.core.utils.stream_shuffle_broadcast import stream_shuffle_broadcast


@nki.jit
def moe_tkg_hits(hidden, gate_up, down, slots, slot_aff, count, gate_up_scale, down_scale, limit):
    """[1, H] routed output of one token. hidden [1, H] bf16; gate_up [E, H, 2, I] / down [E, I, H] FP8 with row
    scales gate_up_scale [E, 2, I] / down_scale [E, H] (moe_tkg's ROW layout); slots [1, K] int32 the local expert
    of each of the token's local slots, those first; slot_aff [1, K] fp32 their affinities; count [1, 1] int32 how
    many slots are local. The SwiGLU clamps the gate above and the up both ways at ``limit``."""
    T, H = hidden.shape
    K = slots.shape[1]
    io_dtype = hidden.dtype
    expert_params = MLPExpertParameters(expert_affinities=slot_aff, expert_index=slots, expert_affinities_eager=None,
                                        expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE)
    params = MLPParameters(
        hidden_tensor=hidden, gate_proj_weights_tensor=gate_up, up_proj_weights_tensor=gate_up,
        down_proj_weights_tensor=down, activation_fn=ActFnType.SiLU, normalization_type=NormType.NO_NORM,
        quantization_type=QuantizationType.ROW, gate_w_scale=gate_up_scale, up_w_scale=gate_up_scale,
        down_w_scale=down_scale, output_dtype=io_dtype, gate_clamp_upper_limit=limit, up_clamp_upper_limit=limit,
        up_clamp_lower_limit=-limit, expert_params=expert_params)
    output = nl.ndarray((T, H), dtype=io_dtype, buffer=nl.shared_hbm)

    sbm = SbufManager(0, 200 * 1024, get_logger("moe_tkg_hits"), use_auto_alloc=True)
    sbm.open_scope()
    params.shard_on_h_disabled = False
    params.use_tkg_gate_up_proj_column_tiling = False
    params.use_tkg_down_proj_column_tiling = False
    dims = MLPTKGConstants.calculate_constants(params)
    pmax = dims._pmax

    input_sb = sbm.alloc_stack([dims.H0, T, dims.H1_shard], dtype=io_dtype, buffer=nl.sbuf, name="input_sb")
    input_norm_load(hidden, input_sb, params, dims, sbm=sbm)

    # everything the loop body carries across iterations lives outside it
    output_temp = nl.ndarray((dims.H0, dims.H1_shard, T), dtype=io_dtype, buffer=nl.sbuf)
    nisa.memset(output_temp, 0.0)
    idx_sb = nl.ndarray((1, K), dtype=slots.dtype, buffer=nl.sbuf)
    idx_tmp = nl.ndarray((1, K), dtype=slots.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=idx_sb, src=slots[0:1, 0:K])
    aff_sb = nl.ndarray((pmax, K), dtype=slot_aff.dtype, buffer=nl.sbuf)
    aff_tmp = nl.ndarray((pmax, K), dtype=slot_aff.dtype, buffer=nl.sbuf)
    nisa.memset(aff_sb, 0.0)
    nisa.dma_copy(dst=aff_sb[0:1, 0:K], src=slot_aff[0:1, 0:K])
    stream_shuffle_broadcast(aff_sb, aff_sb)
    count_sb = nl.ndarray((1, 1), dtype=count.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=count_sb, src=count[0:1, 0:1])
    n_reg = nisa.register_alloc()
    nisa.register_load(src=count_sb, dst=n_reg)

    num_i_tiles = div_ceil(dims.I, dims.I0)
    gate_up_out = nl.ndarray((dims.I0, num_i_tiles, 1), dtype=io_dtype, buffer=nl.sbuf)
    gate_fp32 = nl.ndarray((dims.I0, num_i_tiles, 1), dtype=nl.float32, buffer=nl.sbuf)
    up_fp32 = nl.ndarray((dims.I0, num_i_tiles, 1), dtype=nl.float32, buffer=nl.sbuf)
    down_sb = nl.ndarray((dims.H0, dims.H1_shard), dtype=io_dtype, buffer=nl.sbuf)

    gate_up_w = params.gate_proj_weights_tensor
    down_w = params.down_proj_weights_tensor
    quant = params.quant_params
    dims.T = 1

    def body(_):
        slot_sbm = SbufManager(0, 200 * 1024, get_logger("moe_tkg_hits_slot"), use_auto_alloc=True)
        slot_sbm.open_scope()
        e = idx_sb.ap(pattern=[[K, 1], [1, 1]], offset=0)
        params.gate_proj_weights_tensor = gate_up_w.select(dim=0, index=e).select(dim=1, index=GateUpDim.GATE.value)
        params.up_proj_weights_tensor = gate_up_w.select(dim=0, index=e).select(dim=1, index=GateUpDim.UP.value)
        params.down_proj_weights_tensor = down_w.select(dim=0, index=e)
        params.bias_params = MLPBiasParameters(gate_proj_bias_tensor=None, up_proj_bias_tensor=None,
                                               down_proj_bias_tensor=None)
        params.quant_params = _select_quant_scales(quant, e)
        tiles = process_gate_up_projection(hidden=input_sb.slice(dim=1, start=0, end=1), output=gate_up_out,
                                           params=params, dims=dims, sbm=slot_sbm, share_memory_scope=True,
                                           use_dge=True, prealloc_gate_up_buffers=(gate_fp32, up_fp32))
        process_down_projection(hidden=gate_up_out, output=down_sb, params=params, dims=dims, gate_tile_info=tiles,
                                sbm=slot_sbm, use_dge=True)
        nisa.tensor_scalar(dst=down_sb, data=down_sb, op0=nl.multiply, operand0=aff_sb[:, 0:1])
        acc = output_temp[0:dims.H0, 0:dims.H1_shard, 0]
        nisa.tensor_tensor(dst=acc, data1=acc, data2=down_sb, op=nl.add)
        # the next slot to the front
        nisa.tensor_copy(dst=idx_tmp[0:1, 0:K - 1], src=idx_sb[0:1, 1:K])
        nisa.tensor_copy(dst=idx_sb[0:1, 0:K - 1], src=idx_tmp[0:1, 0:K - 1])
        nisa.tensor_copy(dst=aff_tmp[:, 0:K - 1], src=aff_sb[:, 1:K])
        nisa.tensor_copy(dst=aff_sb[:, 0:K - 1], src=aff_tmp[:, 0:K - 1])
        slot_sbm.close_scope()

    nl.fori_loop(0, n_reg, body)

    transpose_store(output_temp, output, dims, params.output_dtype, sbm, 0)
    sbm.close_scope()
    return output
