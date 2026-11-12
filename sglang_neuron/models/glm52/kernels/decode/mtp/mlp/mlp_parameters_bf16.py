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

"""Parameter container and input validation for the bf16 dense MLP token-generation kernel.

There is no quantization concept here at all -- not a ``NONE`` member of an enum, but absent.
Weights are plain 2D bf16 matrices, so the dimensions are read straight off them; the sibling
MX package derives the same dimensions from x4-PACKED shapes, which is why the two cannot share
one parameter object.
"""

from dataclasses import dataclass
from typing import Optional

import nki.language as nl
from nki.language import NKIObject

# common utils
from ...utils.common_types import ActFnType
from ...utils.kernel_assert import kernel_assert

# Token ceiling: the kernel emits a single BxS tile, so T is bounded by the 128 PSUM/SBUF partitions.
# The PE column tiling adapts to T (32-wide tiles up to 32 tokens, 64-wide up to 64, untiled above),
# so it does not tighten this bound. See mlp_tkg_constants.column_tiling_dim.
MAX_TOKENS = 128


#
# ***********************
# MLP params and methods
# ***********************
#


@dataclass
class MLPParameters(NKIObject):
    """
    Resolved inputs and derived dimensions for the bf16 dense MLP token-generation kernel.

    The hidden input is ``[B, S, H]`` in HBM. All three projection weights are plain 2D:

        gate/up: [H, I]
        down:    [I, H]

    ``I`` is this tensor-parallel rank's slice of the intermediate dimension, so ``down`` is
    row-sharded and the kernel's output is a PARTIAL sum. The token count the sub-kernels work
    on is ``batch_size * sequence_len``.

    The weights are FP8 (``float8_e4m3fn``) with ONE fp32 factor per weight, ``tensor_scales`` ``[1, 3]``: the
    gate's, the up's and the down's (the dequantized weight is the FP8 value times its factor).

    ``weight_ring`` and ``gate_up_fused`` are caller-side schedule options (see ``mlp_bf16``); neither changes an
    operand or the accumulation order.
    """

    hidden_tensor: nl.NkiTensor
    gate_proj_weights_tensor: nl.NkiTensor
    up_proj_weights_tensor: nl.NkiTensor
    down_proj_weights_tensor: nl.NkiTensor
    activation_fn: ActFnType
    output_dtype: object
    batch_size: int
    sequence_len: int
    hidden_size: int
    intermediate_size: int
    tensor_scales: nl.NkiTensor
    weight_ring: object
    gate_up_fused: Optional[nl.NkiTensor]

    def __init__(
        self,
        hidden_tensor: nl.NkiTensor,
        gate_proj_weights_tensor: nl.NkiTensor,
        up_proj_weights_tensor: nl.NkiTensor,
        down_proj_weights_tensor: nl.NkiTensor,
        tensor_scales: nl.NkiTensor,
        weight_ring: object,
        activation_fn: ActFnType = ActFnType.SiLU,
        output_dtype: Optional[object] = None,
        gate_up_fused: Optional[nl.NkiTensor] = None,
    ):
        # The kernel DMAs the hidden tensor from HBM into its own [H0, T, H1] tile, so an SBUF
        # input is not accepted.
        kernel_assert(
            hidden_tensor.buffer != nl.sbuf,
            "Hidden tensor must be in HBM; this kernel loads and transposes it itself.",
        )
        kernel_assert(len(hidden_tensor.shape) == 3, "Hidden tensor must have 3D shape [B, S, H]")
        self.batch_size = hidden_tensor.shape[0]
        self.sequence_len = hidden_tensor.shape[1]

        # down is [I, H], gate/up are [H, I].
        self.hidden_size = down_proj_weights_tensor.shape[-1]
        self.intermediate_size = up_proj_weights_tensor.shape[1]

        self.hidden_tensor = hidden_tensor
        self.gate_proj_weights_tensor = gate_proj_weights_tensor
        self.up_proj_weights_tensor = up_proj_weights_tensor
        self.down_proj_weights_tensor = down_proj_weights_tensor
        self.activation_fn = activation_fn
        self.output_dtype = hidden_tensor.dtype if output_dtype is None else output_dtype
        self.tensor_scales = tensor_scales
        self.weight_ring = weight_ring
        self.gate_up_fused = gate_up_fused


def _validate_mlp_required_arguments(params: MLPParameters):
    kernel_assert(params.hidden_tensor != None, "Hidden tensor is a required argument")
    kernel_assert(
        params.gate_proj_weights_tensor != None,
        "Gate projection tensor is a required argument",
    )
    kernel_assert(
        params.up_proj_weights_tensor != None,
        "Up projection tensor is a required argument",
    )
    kernel_assert(
        params.down_proj_weights_tensor != None,
        "Down projection tensor is a required argument",
    )


def _validate_fp8_scales(params: MLPParameters):
    """One fp32 factor per FP8 weight: ``tensor_scales`` [1, 3] (gate, up, down) and ``float8_e4m3fn`` weights."""
    kernel_assert(
        list(params.tensor_scales.shape) == [1, 3] and params.tensor_scales.dtype == nl.float32,
        f"tensor_scales must be fp32 [1, 3] (gate, up, down), got {list(params.tensor_scales.shape)} "
        f"{params.tensor_scales.dtype}",
    )
    for w in (params.gate_proj_weights_tensor, params.up_proj_weights_tensor, params.down_proj_weights_tensor):
        kernel_assert(w.dtype == nl.float8_e4m3fn, f"FP8 MLP weights must be float8_e4m3fn, got {w.dtype}")


def _validate_mlp_arguments_shapes(params: MLPParameters):
    BxS = params.batch_size * params.sequence_len
    H = params.hidden_size
    I = params.intermediate_size

    kernel_assert(H % 128 == 0, f"Unsupported hidden dimension {H}; expected H % 128 == 0.")
    kernel_assert(BxS > 0, f'Unsupported batch by sequence dimension {BxS}; expected BxS to be positive.')
    kernel_assert(H > 0, f'Unsupported hidden dimension {H}; expected H to be positive.')
    kernel_assert(I > 0, f'Unsupported intermediate dimension {I}; expected I to be positive.')

    kernel_assert(
        list(params.up_proj_weights_tensor.shape) == [H, I],
        f"Up projection weight must have shape [H, I] = [{H}, {I}]. Got {list(params.up_proj_weights_tensor.shape)}",
    )
    kernel_assert(
        params.gate_proj_weights_tensor.shape == params.up_proj_weights_tensor.shape,
        f"Gate and up projection weight shapes do not match. "
        f"Got gate shape {params.gate_proj_weights_tensor.shape} "
        f"and up shape {params.up_proj_weights_tensor.shape}",
    )
    kernel_assert(
        list(params.down_proj_weights_tensor.shape) == [I, H],
        f"Down projection weight must have shape [I, H] = [{I}, {H}]. "
        f"Got {list(params.down_proj_weights_tensor.shape)}",
    )
    _validate_fp8_scales(params)
    _validate_gate_up_fused(params, H, I)


def _validate_gate_up_fused(params: MLPParameters, H: int, I: int):
    """``gate_up_fused`` is the [H, 2I] tensor gate and up are the column halves of."""
    if params.gate_up_fused is not None:
        kernel_assert(
            list(params.gate_up_fused.shape) == [H, 2 * I]
            and params.gate_up_fused.dtype == params.gate_proj_weights_tensor.dtype,
            f"gate_up_fused must be the [H, 2I] = [{H}, {2 * I}] weight whose column halves are gate and up, got "
            f"{list(params.gate_up_fused.shape)} {params.gate_up_fused.dtype}",
        )


def validate_mlp_arguments(params: MLPParameters):
    _validate_mlp_required_arguments(params)
    _validate_mlp_arguments_shapes(params)
