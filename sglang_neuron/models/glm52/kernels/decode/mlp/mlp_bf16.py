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

"""Dense MLP token-generation kernel with bf16 weights.

This is GLM-5.2's ``GlmMoeDsaMLP`` for decoder layers 0..2. The checkpoint's
``quantization_config.exclude`` list keeps ``model.layers.{0,1,2}.mlp.{gate,up,down}_proj`` in
bf16, so those layers need plain ``nc_matmul`` -- no swizzle, no scale operand, no input
quantize -- rather than the MX path (``mlp_tkg_*`` in this package). Both paths exist
side by side; neither is a configuration of the other.
"""

from typing import Optional

import nki
import nki.isa as nisa
import nki.language as nl

# common utils
from ..utils.common_types import ActFnType
from ..utils.kernel_assert import kernel_assert

# MLP utils
from .mlp_parameters_bf16 import MAX_TOKENS, MLPParameters, validate_mlp_arguments
from .mlp_tkg_bf16_impl import mlp_tkg_bf16_impl

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
    activation_fn: ActFnType = ActFnType.SiLU,
    output_dtype: nki.dtype = None,
    column_tiling: bool = True,
    output: Optional[nl.NkiTensor] = None,
) -> nl.NkiTensor:
    """
    Dense MLP token-generation (TKG) kernel with bf16 weights.

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
    so a decode step fills only T of the PE array's 128 columns. With ``column_tiling`` the
    kernel packs ``128 / w`` independent matmuls side by side on PE column tiles of width
    ``w = 32 | 64`` (for T <= 32 | 64) so they run concurrently; off, every matmul uses the full
    array and the PE is ~T/128 utilized. The operands and the accumulation order are the same in
    both modes, so the result is bit-identical; the flag exists to measure the PE-side gain.

    The kernel is weight-DMA bound, so the three projections stream their weights through one
    shared SBUF ring of two 64 KB-per-partition slots in loads as large as a slot (8 MB per gate/up
    load, 6 MB per down load at H = 6144, I = 2048; see ``mlp_tkg_constants``). The down weight is
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
        gate_weights (nl.NkiTensor): [H, I] gate projection weights.
        up_weights (nl.NkiTensor): [H, I] up projection weights.
        down_weights (nl.NkiTensor): [I, H] down projection weights.
        activation_fn (ActFnType): Activation function type. (default: SiLU)
        output_dtype: Output tensor data type. Defaults to None; if None, the hidden tensor's
            dtype is used.
        column_tiling (bool): Pack independent matmuls on PE column tiles when T <= 64.
            (default: True)
        output (nl.NkiTensor, optional): [T, H] caller-preallocated output tensor in HBM. Allocated
            by the kernel in private HBM when None (see below); a standalone launch that must
            return a kernel output passes a shared-HBM tensor here instead.

    Returns:
        output (nl.NkiTensor): [T, H] in HBM. This rank's partial MLP result.

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
        column_tiling=column_tiling,
    )

    # Validate MLP arguments before emitting any instruction
    validate_mlp_arguments(mlp_params)
    _validate_mlp_tkg_bf16_restrictions(mlp_params)

    # private_hbm, not shared: the MoE layer ReduceScatters this output directly, and at LNC=1 the
    # runtime rejects a collective whose sources are shared_hbm. A top-level launch cannot RETURN a
    # private_hbm tensor, so a standalone caller preallocates the output in shared_hbm instead.
    T = mlp_params.batch_size * mlp_params.sequence_len
    if output is None:
        output = nl.ndarray((T, mlp_params.hidden_size), dtype=output_dtype, buffer=nl.private_hbm)
    kernel_assert(
        list(output.shape) == [T, mlp_params.hidden_size],
        f"{_MLP_ERROR_PREFIX} output must be [T, H] = [{T}, {mlp_params.hidden_size}], got {list(output.shape)}",
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
