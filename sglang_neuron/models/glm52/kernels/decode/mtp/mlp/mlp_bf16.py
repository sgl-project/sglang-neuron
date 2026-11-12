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

"""Dense MLP token-generation kernel of the MTP draft's experts: bf16 tokens against FP8 weights with one factor each.

The MTP layer's routed and shared experts run here (``moe_layer_bf16.glm52_moe_rows_bf16``): plain ``nc_matmul`` with
the bf16 tokens stationary and the FP8 weights moving -- no swizzle, no input quantize -- rather than the MX path of
the target's ``decode/mlp`` package.
"""

from typing import Optional

import nki
import nki.isa as nisa
import nki.language as nl

# common utils
from ...utils.common_types import ActFnType
from ...utils.kernel_assert import kernel_assert

# MLP utils
from .mlp_parameters_bf16 import MAX_TOKENS, MLPParameters, validate_mlp_arguments
from .mlp_tkg_bf16_impl import mlp_tkg_bf16_impl
from .mlp_tkg_constants import MLPSharedWeightRing

_MLP_ERROR_PREFIX = "[MLP TKG bf16 Kernel]"

# The gate/up projection issues one PSUM bank per 512 columns of I and does not tile I further,
# so I is bounded by (psum free max) * (psum banks).
_MAX_I = 512 * 8

#
# **********************
# MLP Kernel ISA
# **********************
#


@nki.jit
def mlp_bf16(
    hidden_tensor: nl.NkiTensor,
    gate_weights: nl.NkiTensor,
    up_weights: nl.NkiTensor,
    down_weights: nl.NkiTensor,
    output: nl.NkiTensor,
    tensor_scales: nl.NkiTensor,
    weight_ring: MLPSharedWeightRing,
    activation_fn: ActFnType = ActFnType.SiLU,
    output_dtype: nki.dtype = None,
    gate_up_fused: Optional[nl.NkiTensor] = None,
) -> nl.NkiTensor:
    """
    Dense MLP token-generation (TKG) kernel: bf16 tokens, FP8 weights with one factor each.

    Computes ``down(activation_fn(gate(x)) * up(x))``. There is no normalization, no bias, no
    clamping and no fused residual add: with tensor parallelism the surrounding model applies
    the norm before this kernel and the residual add after the cross-rank reduction, so fusing
    either in here would break the math.

    The kernel runs on Trn3 (gen4) on a single NeuronCore (LNC=1): there is no work split, so
    launching more than one program is rejected because every program would compute the whole
    result and race on the same shared-HBM output rows. It emits a single token tile, so T is
    capped at 128; GLM-5.2's C32 x spec-7 decode step is 14 real tokens (16 padded slots) per rank,
    and the MoE layer's compacted routed path hands it <= 32 live tokens.

    Both projections keep the TOKENS stationary and stream 512 weight columns per ``nc_matmul``,
    so a decode step fills only T of the PE array's 128 columns. The kernel packs ``128 / w``
    independent matmuls side by side on PE column tiles of width ``w = 32 | 64`` (for T <= 32 | 64)
    so they run concurrently; above 64 tokens every matmul uses the full array.

    The kernel is weight-DMA bound, so the three projections stream their weights through the caller's
    SBUF ring (``weight_ring``) in loads as large as a slot (see ``mlp_tkg_constants``). The down weight is
    loaded as ``[128, i1, H]`` tiles, which requires I to be a multiple of 128.

    With colwise gate/up weights and a rowwise down weight, this rank produces a PARTIAL sum
    over its slice of I; the reduction across ranks happens outside the kernel.

    Dimensions:
        B: Batch size
        S: Sequence length
        T: Number of tokens (B * S)
        H: Hidden dimension
        I: Intermediate dimension (this rank's slice)

    Args:
        hidden_tensor (nl.NkiTensor): [B, S, H] hidden states in HBM. Already normed.
        gate_weights (nl.NkiTensor): [H, I] ``float8_e4m3fn`` gate projection weights.
        up_weights (nl.NkiTensor): [H, I] ``float8_e4m3fn`` up projection weights.
        down_weights (nl.NkiTensor): [I, H] ``float8_e4m3fn`` down projection weights.
        output (nl.NkiTensor): [T, H] SBUF tile of ``output_dtype``, which receives the down projection directly,
            with no HBM write: the caller reduces the partial on chip.
        tensor_scales (nl.NkiTensor): [1, 3] fp32, the weights' factors: gate's, up's, down's. See FP8 TENSOR
            SCALES below.
        weight_ring (MLPSharedWeightRing): The caller's weight ring, shared with its other calls; the loads continue
            its rotation at ``weight_ring.position`` and their DMA instructions are appended to ``weight_ring.dmas``
            (see ``MLPSharedWeightRing``).
        activation_fn (ActFnType): Activation function type. (default: SiLU)
        output_dtype: Output tensor data type. Defaults to None; if None, the hidden tensor's
            dtype is used.
        gate_up_fused (nl.NkiTensor, optional): [H, 2I], the tensor whose column halves ``[0, I)`` / ``[I, 2I)``
            ``gate_weights`` / ``up_weights`` are. When each projection is a single load, the gate and up weights
            then come in loads of whole ``2I`` rows: ONE ``[128, H / 128, 2I]`` load if it fits a ring slot, else
            the fewest ``[128, H / 128 / n, 2I]`` loads that fit and hold a multiple of the column-tiling factor of
            H tiles. Each partition reads one contiguous run per load instead of ``I``-element runs at a ``2I``
            stride; the matmuls read the same values, into the same PSUM windows, in the same order.

    FP8 TENSOR SCALES:
        The matmuls keep the bf16 tokens stationary and stream the FP8 weights as the moving operand (a mixed-dtype
        matmul accumulating in fp32), so the weights reach the PE array in FP8. One factor per weight costs nothing
        extra: the factors fold into instructions the MLP issues anyway. ``silu(s_g * gate)`` is the activation's
        own ``scale``, ``s_u`` joins the ``gate * up`` multiply (one ``scalar_tensor_tensor``), and ``s_d`` scales
        the down projection's PSUM eviction (the ``activation`` copy's ``scale``, or a ``tensor_scalar`` in place of
        the Vector copy). e4m3 is a float format, so its relative precision does not depend on the factor; for
        GLM-5.2's MTP experts one factor per weight is as accurate as 128 x 128 blocks.

    Returns:
        output (nl.NkiTensor): the SBUF ``output``, holding this rank's partial MLP result.

    Pseudocode:
        gate_proj_out = hidden_states @ gate_weights
        act_gate_proj = activation_fn(gate_proj_out)

        up_proj_out = hidden_states @ up_weights
        intermediate = act_gate_proj * up_proj_out

        output = intermediate @ down_weights
    """

    # If output_dtype is not provided, use the same dtype as the hidden tensor
    if output_dtype == None:
        output_dtype = hidden_tensor.dtype

    mlp_params = MLPParameters(
        hidden_tensor=hidden_tensor,
        gate_proj_weights_tensor=gate_weights,
        up_proj_weights_tensor=up_weights,
        down_proj_weights_tensor=down_weights,
        activation_fn=activation_fn,
        output_dtype=output_dtype,
        tensor_scales=tensor_scales,
        weight_ring=weight_ring,
        gate_up_fused=gate_up_fused,
    )

    # Validate MLP arguments before emitting any instruction
    validate_mlp_arguments(mlp_params)
    _validate_mlp_tkg_bf16_restrictions(mlp_params)

    T = mlp_params.batch_size * mlp_params.sequence_len
    kernel_assert(
        output.buffer == nl.sbuf and list(output.shape) == [T, mlp_params.hidden_size],
        f"{_MLP_ERROR_PREFIX} output must be an SBUF [T, H] = [{T}, {mlp_params.hidden_size}] tile, got "
        f"{output.buffer} {list(output.shape)}",
    )

    return mlp_tkg_bf16_impl(mlp_params, output)


def _validate_mlp_tkg_bf16_restrictions(params: MLPParameters) -> None:
    """
    Reject the launches and inputs this single-configuration kernel does not handle.

    Args:
        params (MLPParameters): Resolved MLP inputs and dimensions.

    Returns:
        None
    """
    # Single NeuronCore only: there is no work split, so a second program would recompute the
    # whole result and race with the first one on the shared-HBM output.
    kernel_assert(
        nl.program_ndim() == 0 or nl.program_ndim() == 1,
        f"{_MLP_ERROR_PREFIX} only no specialization or specialization along one axis is supported, got "
        f"nl.program_ndim()={nl.program_ndim()}",
    )
    n_programs = nl.num_programs(axes=0) if nl.program_ndim() != 0 else 1
    kernel_assert(
        n_programs == 1,
        f"{_MLP_ERROR_PREFIX} MLP TKG runs on a single NeuronCore (LNC=1), got {n_programs} programs",
    )

    # Trn3 only: the kernel is validated there, and the 32-wide PE column tile it uses for a decode
    # step cannot be expressed on gen2 (neither the row nor the column size of a PE tile may be 32).
    kernel_assert(
        nisa.get_nc_version() >= nisa.nc_version.gen4,
        f"{_MLP_ERROR_PREFIX} only supports gen4+ (Trn3+) but got {nisa.get_nc_version()=}",
    )

    # The kernel emits a single BxS tile: T tokens on T PSUM/SBUF partitions.
    T = params.batch_size * params.sequence_len
    kernel_assert(
        T <= MAX_TOKENS,
        f"{_MLP_ERROR_PREFIX} T = B * S must be at most {MAX_TOKENS} tokens, got T={T}",
    )

    # The gate/up projection allocates one PSUM bank per 512 columns of I and does not shard I.
    kernel_assert(
        params.intermediate_size <= _MAX_I,
        f"{_MLP_ERROR_PREFIX} intermediate size must be at most {_MAX_I}, got I={params.intermediate_size}",
    )
