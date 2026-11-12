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

"""Parameter dataclasses, config initialization, and validation for all-expert MoE MX kernel."""

from dataclasses import dataclass

import nki.language as nl

from ..utils.common_types import ActFnType, ExpertAffinityScaleMode
from ..utils.kernel_assert import kernel_assert
from ..utils.kernel_helpers import div_ceil
from .mlp_parameters import MLPParameters
from .projection_mx_constants import (
    MAX_MATMULT_MX_UNPACKED_CONTRACT_DIM,
    MIN_MATMULT_MX_P_DIM,
    MX_PACKED_DTYPES,
    MX_SCALE_DTYPE,
    SUPPORTED_QMX_INPUT_DTYPES,
    _q_width,
)

# Constants
NUM_H4_FOLDS_PER_COLUMN = 32


@dataclass
class AllExpertMXInputTensors(nl.NKIObject):
    """Input tensors for all-expert MX kernel."""

    hidden_input: nl.NkiTensor
    gate_up_weights: nl.NkiTensor
    down_weights: nl.NkiTensor
    output: nl.NkiTensor
    expert_affinities_masked: nl.NkiTensor
    gate_up_weights_scale: nl.NkiTensor
    down_weights_scale: nl.NkiTensor
    hidden_input_scale: nl.NkiTensor
    gate_up_weights_bias: nl.NkiTensor
    down_weights_bias: nl.NkiTensor


@dataclass
class AllExpertMXKernelConfig(nl.NKIObject):
    """Kernel-level hyperparameters for all-expert MX kernel."""

    expert_affinities_scaling_mode: ExpertAffinityScaleMode
    hidden_act_fn: ActFnType
    gate_clamp_lower_limit: float
    gate_clamp_upper_limit: float
    up_clamp_lower_limit: float
    up_clamp_upper_limit: float
    output_in_sbuf: bool
    # Every affinity is 1: the post-scale multiplies by an immediate 1.0 (see moe_tkg's unit_affinities)
    unit_affinities: bool = False


@dataclass
class AllExpertMXDimensions(nl.NKIObject):
    """Tensor dimensions and derived tiling constants for all-expert MX kernel."""

    # Base dimensions
    T: int  # Total number of input tokens
    E_L: int  # Number of local experts
    I: int  # Intermediate dimension size
    H: int  # Hidden dimension size

    def __post_init__(self):
        """Derive tiling strategy from tensor dimensions."""
        self.pmax = nl.tile_size.pmax

        # Pad T to the next multiple of 4 to satisfy nc_matmul_mx even free-dim constraint.
        # T_physical retains the original (unpadded) token count for output writes.
        self.T_physical = self.T
        self.T = div_ceil(self.T, _q_width) * _q_width

        # Tiling strategy
        self.n_tiles_in_T = div_ceil(self.T, self.pmax)
        self.n_T32_tiles = div_ceil(self.T, NUM_H4_FOLDS_PER_COLUMN)
        self.n_H512_tiles = div_ceil(self.H, MAX_MATMULT_MX_UNPACKED_CONTRACT_DIM)
        self.tile_T = min(self.T, self.pmax)
        self.tile_H = self.H // self.n_H512_tiles // _q_width
        self.T32_H4 = self.pmax

        # Intermediate-dimension tiling. I_padded is the nc_matmul_mx free-dim (/8) padding.
        self.n_I512_tiles = div_ceil(self.I, MAX_MATMULT_MX_UNPACKED_CONTRACT_DIM)
        self.I_padded = div_ceil(self.I, MIN_MATMULT_MX_P_DIM) * MIN_MATMULT_MX_P_DIM


@dataclass
class ExpertWeightsSBUF(nl.NKIObject):
    """Expert weights, scales, and biases loaded in SBUF for one expert."""

    gate_weight_sb: nl.NkiTensor
    up_weight_sb: nl.NkiTensor
    down_weight_sb: nl.NkiTensor
    gate_weight_scale_sb: nl.NkiTensor
    up_weight_scale_sb: nl.NkiTensor
    down_weight_scale_sb: nl.NkiTensor
    gate_bias_sb: nl.NkiTensor
    up_bias_sb: nl.NkiTensor
    down_bias_sb: nl.NkiTensor
    # The down weight DMA instruction, which the next expert's gate weight DMA follows
    down_weight_dma: object = None


@dataclass
class PrefetchedGateUp(nl.NKIObject):
    """One expert's gate and up loads traced ahead of the expert kernel (``MOE_GU_PREFETCH``)."""

    expert_idx: int
    gate: tuple  # (weight, scale, bias) SBUF tiles
    up: tuple  # (weight, scale, bias) SBUF tiles
    dmas: list  # [gate weight DMA, up weight DMA]
    next_too: bool = False  # also prefetch expert e+1's gate/up right after expert e's gate/up projection


def init_all_expert_mx_configs(
    mlp_params: MLPParameters,
    output: nl.NkiTensor,
) -> tuple[AllExpertMXInputTensors, AllExpertMXKernelConfig, AllExpertMXDimensions]:
    """
    Initialize all sub-configs for the all-expert MX kernel from MLPParameters.

    Args:
        mlp_params (MLPParameters): Source parameters.
        output (nl.NkiTensor): Output tensor.

    Returns:
        tuple: (AllExpertMXInputTensors, AllExpertMXKernelConfig, AllExpertMXDimensions)
    """
    hidden_input = mlp_params.hidden_tensor
    hidden_input_scale = mlp_params.hidden_input_scale
    gate_up_weights = mlp_params.gate_proj_weights_tensor
    down_weights = mlp_params.down_proj_weights_tensor

    # Extract T based on the quantization state (MLPParameters already rejected every other layout)
    if hidden_input_scale != None:
        # Pre-quantized input (SBUF or HBM): [H0, H/512, T]
        T = hidden_input.shape[-1]
    else:
        # Unquantized input in HBM: [T, H]
        T, _ = hidden_input.shape

    # Extract dimensions from weight shapes
    E_L = gate_up_weights.shape[0]
    I = gate_up_weights.shape[-1]
    H = down_weights.shape[-1]

    # Derive output_in_sbuf from output buffer location
    output_in_sbuf = output.buffer == nl.sbuf

    input_tensors = AllExpertMXInputTensors(
        hidden_input=hidden_input,
        gate_up_weights=gate_up_weights,
        down_weights=down_weights,
        output=output,
        expert_affinities_masked=mlp_params.expert_params.expert_affinities,
        gate_up_weights_scale=mlp_params.quant_params.gate_w_scale,
        down_weights_scale=mlp_params.quant_params.down_w_scale,
        hidden_input_scale=hidden_input_scale,
        gate_up_weights_bias=mlp_params.bias_params.gate_proj_bias_tensor,
        down_weights_bias=mlp_params.bias_params.down_proj_bias_tensor,
    )
    kernel_cfg = AllExpertMXKernelConfig(
        expert_affinities_scaling_mode=mlp_params.expert_params.expert_affinities_scaling_mode,
        hidden_act_fn=mlp_params.activation_fn,
        gate_clamp_lower_limit=mlp_params.gate_clamp_lower_limit,
        gate_clamp_upper_limit=mlp_params.gate_clamp_upper_limit,
        up_clamp_lower_limit=mlp_params.up_clamp_lower_limit,
        up_clamp_upper_limit=mlp_params.up_clamp_upper_limit,
        output_in_sbuf=output_in_sbuf,
        unit_affinities=mlp_params.expert_params.unit_affinities,
    )
    dims = AllExpertMXDimensions(
        T=T,
        E_L=E_L,
        I=I,
        H=H,
    )

    return input_tensors, kernel_cfg, dims


# Validation helpers


def validate_all_expert_mx_inputs(
    input_tensors: AllExpertMXInputTensors,
    kernel_cfg: AllExpertMXKernelConfig,
    dims: AllExpertMXDimensions,
) -> None:
    """
    Validate input tensors and configuration for all-expert MX kernel.

    Args:
        input_tensors (AllExpertMXInputTensors): Tensor parameters.
        kernel_cfg (AllExpertMXKernelConfig): Scalar parameters.
        dims (AllExpertMXDimensions): Dimension parameters.

    Returns:
        None. Raises AssertionError via kernel_assert if validation fails.
    """

    # Validate input dtype based on quantization state
    if input_tensors.hidden_input_scale == None:
        kernel_assert(
            input_tensors.hidden_input.dtype in SUPPORTED_QMX_INPUT_DTYPES,
            f"Expected input dtype in {SUPPORTED_QMX_INPUT_DTYPES}, got {input_tensors.hidden_input.dtype=}.",
        )
    else:
        kernel_assert(
            input_tensors.hidden_input.dtype in MX_PACKED_DTYPES,
            f"Expected quantized input dtype in {MX_PACKED_DTYPES}, got {input_tensors.hidden_input.dtype=}",
        )
        kernel_assert(
            input_tensors.hidden_input_scale.dtype == MX_SCALE_DTYPE,
            f"Expected hidden_input_scale.dtype={MX_SCALE_DTYPE}, got {input_tensors.hidden_input_scale.dtype=}",
        )

    # Validate T size based on quantization.
    # When quantized, T can be divisible by 4; otherwise T must be divisible by 32.
    if input_tensors.hidden_input_scale is not None:
        kernel_assert(dims.T % 4 == 0, f"Expected T divisible by 4 with quantized input, got T={dims.T}")
    else:
        kernel_assert(
            dims.T_physical % 32 == 0,
            f"Expected T divisible by 32, got T={dims.T}. "
            "To use T divisible by 4, provide prequantized input and hidden_input_scale.",
        )

    # Validate expert affinities shape
    kernel_assert(
        len(input_tensors.expert_affinities_masked.shape) in (2, 3),
        f"Expected 2D or 3D expert_affinities_masked, got {input_tensors.expert_affinities_masked.shape=}",
    )

    # Validate output location
    kernel_assert(
        not kernel_cfg.output_in_sbuf,
        f"All-expert MX kernel does not yet support SBUF output, got {kernel_cfg.output_in_sbuf=}",
    )
