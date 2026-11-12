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

"""This module implements the all-expert Mixture of Experts (MoE) token generation kernel with MX quantized weights."""

from typing import Optional

import nki
import nki.isa as nisa
import nki.language as nl

# common utils
from ..utils.common_types import ActFnType, ExpertAffinityScaleMode
from ..utils.kernel_assert import kernel_assert
from ..utils.kernel_helpers import sizeinbytes
from .all_expert_mx_impl import MOE_AFF_SLICE_LATE, _all_expert_moe_tkg_mx, set_affinity_slice

# MLP utils
from .mlp_parameters import MLPExpertParameters, MLPParameters, get_T_from_hidden_input
from .moe_tkg_affinity_masking import mask_expert_affinities
from .projection_mx_constants import MX_SCALE_DTYPE

# Constants
_SUPPORTED_MX_DTYPES = (nl.float4_e2m1fn_x4, nl.float8_e4m3fn_x4)
_MOE_TKG_ERROR_PREFIX = "[MoE TKG Kernel]"


@nki.jit
def moe_tkg(
    hidden_input: nl.NkiTensor,
    expert_gate_up_weights: nl.NkiTensor,
    expert_down_weights: nl.NkiTensor,
    expert_affinities: nl.NkiTensor,
    expert_index: nl.NkiTensor,
    rank_id: Optional[nl.NkiTensor] = None,
    expert_gate_up_bias: Optional[nl.NkiTensor] = None,
    expert_down_bias: Optional[nl.NkiTensor] = None,
    expert_gate_up_weights_scale: Optional[nl.NkiTensor] = None,
    expert_down_weights_scale: Optional[nl.NkiTensor] = None,
    hidden_input_scale: Optional[nl.NkiTensor] = None,
    mask_unselected_experts: bool = False,
    expert_affinities_scaling_mode: ExpertAffinityScaleMode = ExpertAffinityScaleMode.POST_SCALE,
    activation_fn: ActFnType = ActFnType.SiLU,
    output_dtype: nki.dtype = None,
    gate_clamp_upper_limit: Optional[float] = None,
    gate_clamp_lower_limit: Optional[float] = None,
    up_clamp_upper_limit: Optional[float] = None,
    up_clamp_lower_limit: Optional[float] = None,
    output: Optional[nl.NkiTensor] = None,
    token_count: Optional[nl.NkiTensor] = None,
    unit_affinities: bool = False,
    prefetched_gate_up=None,
) -> nl.NkiTensor:
    """
    Mixture of Experts (MoE) MLP token generation kernel.

    Every local expert processes every token (all-expert routing) using MX quantized weights.
    Both MX weight dtypes are supported: float4_e2m1fn_x4 (MXFP4) and float8_e4m3fn_x4 (MXFP8).
    The kernel runs on a single NeuronCore (LNC=1); launching it with more than one program is
    rejected because there is no work split and every program would write the same output.

    Supported hidden input data types: bfloat16, float16, or pre-quantized float8_e4m3fn_x4 when
    hidden_input_scale is supplied.

    Dimensions:
        T: Number of tokens (batch_size * seq_len)
        H: Hidden dimension
        I: Intermediate dimension
        E: Number of global experts
        E_L: Number of local experts processed by this kernel
        K: Top-k experts per token
        I_p: I//4 if I <= 512 else 128

    Args:
        hidden_input (nl.NkiTensor): [T, H] in HBM, or [H0, H/512, T] pre-quantized MX input in SBUF or HBM
            when hidden_input_scale is provided.
        expert_gate_up_weights (nl.NkiTensor): [E_L, 128, 2, ceil(H/512), I], Fused gate and up projection
            weights in MX packed layout.
        expert_down_weights (nl.NkiTensor): [E_L, I_p, ceil(I/512), H], Down projection weights in MX packed layout.
        expert_affinities (nl.NkiTensor): [T, E], Expert routing weights/affinities. With affinity scaling
            enabled, this is sliced to [T, E_L] internally.
        expert_index (nl.NkiTensor): [T, K], Top-K expert indices per token. Used to mask unselected experts
            when mask_unselected_experts=True.
        rank_id (nl.NkiTensor): [1, 1], Rank ID tensor specifying which worker processes experts
            [E_L * rank_id, E_L * (rank_id + 1)). Required: affinity scaling is always applied.
        expert_gate_up_bias (nl.NkiTensor, optional): [E_L, I_p, 2, ceil(I/512), 4], Bias for gate/up projections.
        expert_down_bias (nl.NkiTensor, optional): [E_L, H], Bias for down projection.
        expert_gate_up_weights_scale (nl.NkiTensor): [E_L, 16, 2, ceil(H/512), I] uint8, MX micro-scales for
            gate/up weights. Required.
        expert_down_weights_scale (nl.NkiTensor): [E_L, I_p/8, ceil(I/512), H] uint8, MX micro-scales for
            down weights. Required.
        hidden_input_scale (nl.NkiTensor, optional): [H0, H/512, T] uint8, MX micro-scales for a pre-quantized
            hidden_input. When provided, the kernel skips the internal swizzle + quantization step.
        mask_unselected_experts (bool): Whether to apply expert affinity masking based on expert_index. When
            True, affinities are masked to zero for experts not selected by each token. Only supported for
            T <= 128. (default: False)
        expert_affinities_scaling_mode (ExpertAffinityScaleMode): When to apply affinity scaling. POST_SCALE
            (scale each expert output by its affinity) is the only supported value. (default: POST_SCALE)
        activation_fn (ActFnType): Activation function type. (default: SiLU)
        output_dtype: Output tensor data type. Defaults to None.
            If None, uses hidden_input dtype when hidden_input is a >= 2 byte dtype, else bf16.
        gate_clamp_upper_limit (float, optional): Upper bound value to clamp gate projection results.
        gate_clamp_lower_limit (float, optional): Lower bound value to clamp gate projection results.
        up_clamp_upper_limit (float, optional): Upper bound value to clamp up projection results.
        up_clamp_lower_limit (float, optional): Lower bound value to clamp up projection results.
        output (nl.NkiTensor, optional): [T, H] caller-preallocated output tensor in shared HBM. Allocated by
            the kernel when None.
        token_count (nl.NkiTensor, optional): [1, 1] int32 in HBM, runtime number of valid tokens
            (1 <= token_count <= T, T <= 256). The gate/up projections then process only the first
            token_count tokens via hardware shape registers, and output rows past it are zero (no bias)
            and not meaningful. No served call passes it: it relies on hardware shape registers, which the
            released toolchain re-programs for every matmul, so the served build always processes all T
            tokens.
        unit_affinities (bool): Every affinity is 1 (an unweighted expert, e.g. GLM-5.2's shared expert):
            ``expert_affinities`` and ``rank_id`` are not read, and the post-scale multiplies by an immediate
            1.0 -- the same instruction and result as scaling by a ones table, without the per-call affinity
            slice. (default: False)
        prefetched_gate_up (PrefetchedGateUp, optional): One expert's gate/up loads the caller traced earlier
            with ``all_expert_mx_impl.prefetch_expert_gate_up`` (so they precede the caller's other SBUF work in
            program order); that expert's gate/up loads are then not traced again. (default: None)

    Returns:
        output (nl.NkiTensor): [T, H] in shared HBM. Output tensor with MoE computation results.

    Pseudocode:
        # Slice the affinities of this rank's experts, optionally masking unselected experts
        masked_expert_affinities = mask_expert_affinities(expert_affinities, expert_index, rank_id)

        # Process every local expert
        output = zeros([T, H])
        for each local expert:
            gate_proj_out = hidden_states @ gate_weights
            act_gate_proj = activation_fn(gate_proj_out)
            up_proj_out = hidden_states @ up_weights
            intermediate = act_gate_proj * up_proj_out
            expert_out = intermediate @ down_weights
            expert_out *= affinity
            output += expert_out
    """

    # Default output_dtype to hidden_input.dtype when hidden_input is not quantized,
    # and bf16 when hidden_input is quantized.
    if output_dtype is None:
        output_dtype = hidden_input.dtype if sizeinbytes(hidden_input.dtype) >= 2 else nl.bfloat16

    # Validate inputs before emitting any instruction
    _validate_moe_tkg_inputs(
        expert_weight_dtype=expert_gate_up_weights.dtype,
        expert_gate_up_weights_scale=expert_gate_up_weights_scale,
        expert_down_weights_scale=expert_down_weights_scale,
        expert_affinities_scaling_mode=expert_affinities_scaling_mode,
        rank_id=rank_id,
        output_dtype=output_dtype,
    )

    # Slice the global affinities down to this rank's local experts, masking unselected experts on request
    E_L = expert_gate_up_weights.shape[0]
    T = get_T_from_hidden_input(hidden_input, hidden_input_scale)
    K = expert_index.shape[-1]

    if unit_affinities:
        kernel_assert(
            not mask_unselected_experts,
            f"{_MOE_TKG_ERROR_PREFIX} unit_affinities excludes mask_unselected_experts",
        )
        # Never read: the post-scale uses an immediate 1.0
        masked_expert_affinities = expert_affinities
    elif MOE_AFF_SLICE_LATE and not mask_unselected_experts:
        # The expert kernel traces the slice before expert 0's down projection (GLM52K_MOE_AFF_SLICE_LATE); the
        # gathered table stands in for the shape checks until then
        masked_expert_affinities = expert_affinities
        set_affinity_slice(
            lambda: mask_expert_affinities(
                expert_affinities=expert_affinities,
                expert_index=expert_index,
                rank_id=rank_id,
                E_L=E_L,
                T=T,
                K=K,
                io_dtype=expert_affinities.dtype,
                mask_unselected_experts=False,
            )
        )
    else:
        masked_expert_affinities = mask_expert_affinities(
            expert_affinities=expert_affinities,
            expert_index=expert_index,
            rank_id=rank_id,
            E_L=E_L,
            T=T,
            K=K,
            io_dtype=expert_affinities.dtype,
            mask_unselected_experts=mask_unselected_experts,
        )

    # Initialize config objects
    expert_params = MLPExpertParameters(
        expert_affinities=masked_expert_affinities,
        expert_affinities_scaling_mode=expert_affinities_scaling_mode,
        unit_affinities=unit_affinities,
    )

    mlp_params = MLPParameters(
        hidden_tensor=hidden_input,
        gate_proj_weights_tensor=expert_gate_up_weights,
        down_proj_weights_tensor=expert_down_weights,
        activation_fn=activation_fn,
        gate_proj_bias_tensor=expert_gate_up_bias,
        down_proj_bias_tensor=expert_down_bias,
        gate_w_scale=expert_gate_up_weights_scale,
        down_w_scale=expert_down_weights_scale,
        hidden_input_scale=hidden_input_scale,
        expert_params=expert_params,
        gate_clamp_upper_limit=gate_clamp_upper_limit,
        gate_clamp_lower_limit=gate_clamp_lower_limit,
        up_clamp_upper_limit=up_clamp_upper_limit,
        up_clamp_lower_limit=up_clamp_lower_limit,
    )

    T = mlp_params.sequence_len
    H = mlp_params.hidden_size

    # Allocate output tensor if not provided by caller. private_hbm, not shared: the MoE layer
    # ReduceScatters this output directly, and at LNC=1 the runtime rejects a collective whose
    # sources are shared_hbm.
    if output is None:
        output = nl.ndarray((T, H), dtype=output_dtype, buffer=nl.private_hbm)

    _all_expert_moe_tkg_mx(mlp_params, output, token_count, prefetched_gate_up=prefetched_gate_up)

    return output


def _validate_moe_tkg_inputs(
    expert_weight_dtype: nki.dtype,
    expert_gate_up_weights_scale: Optional[nl.NkiTensor],
    expert_down_weights_scale: Optional[nl.NkiTensor],
    expert_affinities_scaling_mode: ExpertAffinityScaleMode,
    rank_id: Optional[nl.NkiTensor],
    output_dtype: nki.dtype,
) -> None:
    """
    Validate MoE TKG kernel input parameters.

    Args:
        expert_weight_dtype (nki.dtype): Dtype of the expert weights (must be an MX packed dtype).
        expert_gate_up_weights_scale (nl.NkiTensor, optional): MX micro-scales for gate/up weights.
        expert_down_weights_scale (nl.NkiTensor, optional): MX micro-scales for down weights.
        expert_affinities_scaling_mode (ExpertAffinityScaleMode): Affinity scaling mode (must be POST_SCALE).
        rank_id (nl.NkiTensor, optional): Rank ID tensor, required to slice this rank's local experts.
        output_dtype (nki.dtype): Resolved output dtype (must be at least 2 bytes).

    Returns:
        None
    """
    # Single NeuronCore only: there is no work split, so a second program would recompute the whole
    # result and race with the first one on the shared-HBM output.
    n_programs = nl.num_programs(axes=0) if nl.program_ndim() != 0 else 1
    kernel_assert(
        n_programs == 1,
        f"{_MOE_TKG_ERROR_PREFIX} MoE TKG runs on a single NeuronCore (LNC=1), got {n_programs} programs",
    )

    # Only MX weights are supported, and they require uint8 micro-scales
    kernel_assert(
        expert_weight_dtype in _SUPPORTED_MX_DTYPES,
        f"{_MOE_TKG_ERROR_PREFIX} MoE TKG only supports MX weights, got {expert_weight_dtype}",
    )
    kernel_assert(
        expert_gate_up_weights_scale != None and expert_down_weights_scale != None,
        f"{_MOE_TKG_ERROR_PREFIX} Scales must be set when using MX weights",
    )
    kernel_assert(
        expert_gate_up_weights_scale.dtype == MX_SCALE_DTYPE and expert_down_weights_scale.dtype == MX_SCALE_DTYPE,
        f"{_MOE_TKG_ERROR_PREFIX} Weight scales must be {MX_SCALE_DTYPE} hardware MX micro-scales, got "
        f"{expert_gate_up_weights_scale.dtype=}, {expert_down_weights_scale.dtype=}",
    )
    kernel_assert(
        nisa.get_nc_version() >= nisa.nc_version.gen4,
        f"{_MOE_TKG_ERROR_PREFIX} MX weights are only supported on gen4+ (Trn3+) but got {nisa.get_nc_version()=}",
    )

    # Expert affinities are always applied, which requires a rank_id to locate this rank's experts
    kernel_assert(
        expert_affinities_scaling_mode == ExpertAffinityScaleMode.POST_SCALE,
        f"{_MOE_TKG_ERROR_PREFIX} only {ExpertAffinityScaleMode.POST_SCALE} affinity scaling is supported, got "
        f"{expert_affinities_scaling_mode=}",
    )
    kernel_assert(rank_id != None, f"{_MOE_TKG_ERROR_PREFIX} rank_id is required to slice this rank's local experts")

    kernel_assert(
        sizeinbytes(output_dtype) >= 2,
        f"{_MOE_TKG_ERROR_PREFIX} output_dtype must be at least a 2-byte dtype, got {output_dtype}",
    )
