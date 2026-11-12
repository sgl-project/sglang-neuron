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

"""Parameter containers for the MX all-expert MoE token-generation MLP."""

from dataclasses import dataclass
from typing import Optional

import nki.language as nl
from nki.language import NKIObject

# common utils
from ..utils.common_types import ActFnType, ExpertAffinityScaleMode
from ..utils.kernel_assert import kernel_assert


def get_T_from_hidden_input(hidden_input: nl.NkiTensor, hidden_input_scale: Optional[nl.NkiTensor] = None) -> int:
    """
    Extract T (number of tokens) from hidden_input tensor based on its layout.

    Args:
        hidden_input: Input tensor with one of two layouts:
            - Pre-quantized MX input (SBUF or HBM): [H0, H/512, T]
            - Unquantized input in HBM: [T, H]
        hidden_input_scale: Micro-scale tensor, indicates pre-quantized MX input if present

    Returns:
        T: Number of tokens
    """
    if hidden_input_scale != None:
        # Pre-quantized input (SBUF or HBM): T is the last dimension
        return hidden_input.shape[-1]
    return hidden_input.shape[0]


#
#
# ****************************
# Quantization params
# ****************************
#
@dataclass
class MLPQuantizationParameters(NKIObject):
    """MX weight micro-scales, one uint8 scale tensor per projection."""

    gate_w_scale: Optional[nl.NkiTensor]
    down_w_scale: Optional[nl.NkiTensor]

    def __init__(
        self,
        gate_w_scale: Optional[nl.NkiTensor],
        down_w_scale: Optional[nl.NkiTensor],
    ):
        self.gate_w_scale = gate_w_scale
        self.down_w_scale = down_w_scale


@dataclass
class MLPExpertParameters(NKIObject):
    """Per-expert routing weights for the all-expert MoE path."""

    expert_affinities: nl.NkiTensor
    expert_affinities_scaling_mode: ExpertAffinityScaleMode = ExpertAffinityScaleMode.POST_SCALE
    # Every affinity is 1: expert_affinities is not read and the post-scale multiplies by 1.0
    unit_affinities: bool = False


#
# ***********************
# Bias params
# ***********************
#


@dataclass
class MLPBiasParameters(NKIObject):
    """Optional projection biases. gate_proj_bias_tensor holds the fused gate+up bias."""

    gate_proj_bias_tensor: Optional[nl.NkiTensor]
    down_proj_bias_tensor: Optional[nl.NkiTensor]

    def __init__(
        self,
        gate_proj_bias_tensor: Optional[nl.NkiTensor],
        down_proj_bias_tensor: Optional[nl.NkiTensor],
    ):
        self.gate_proj_bias_tensor = gate_proj_bias_tensor
        self.down_proj_bias_tensor = down_proj_bias_tensor


#
# ***********************
# MLP params
# ***********************
#


@dataclass
class MLPParameters(NKIObject):
    """
    Resolved inputs for the MX all-expert MoE MLP.

    Note that `gate_proj_weights_tensor` holds the FUSED gate+up weight tensor
    (its dim-2 selects gate vs up), and `gate_proj_bias_tensor` likewise holds the
    fused gate+up bias. The names are historical.

    The hidden input has one of two layouts:
      - Pre-quantized MX input (SBUF or HBM), with hidden_input_scale: [H0, H/512, T]
      - Unquantized input in HBM, without hidden_input_scale: [T, H]
    """

    hidden_tensor: nl.NkiTensor
    gate_proj_weights_tensor: nl.NkiTensor
    down_proj_weights_tensor: nl.NkiTensor
    activation_fn: ActFnType
    bias_params: Optional[MLPBiasParameters]
    quant_params: Optional[MLPQuantizationParameters]
    expert_params: Optional[MLPExpertParameters]
    sequence_len: int
    hidden_size: int
    hidden_input_scale: Optional[nl.NkiTensor]
    gate_clamp_lower_limit: Optional[float]
    gate_clamp_upper_limit: Optional[float]
    up_clamp_lower_limit: Optional[float]
    up_clamp_upper_limit: Optional[float]

    def __init__(
        self,
        hidden_tensor: nl.NkiTensor,
        gate_proj_weights_tensor: nl.NkiTensor,
        down_proj_weights_tensor: nl.NkiTensor,
        gate_proj_bias_tensor: Optional[nl.NkiTensor] = None,
        down_proj_bias_tensor: Optional[nl.NkiTensor] = None,
        activation_fn: ActFnType = ActFnType.SiLU,
        gate_w_scale: Optional[nl.NkiTensor] = None,
        down_w_scale: Optional[nl.NkiTensor] = None,
        gate_clamp_lower_limit: Optional[float] = None,
        gate_clamp_upper_limit: Optional[float] = None,
        up_clamp_lower_limit: Optional[float] = None,
        up_clamp_upper_limit: Optional[float] = None,
        expert_params: Optional[MLPExpertParameters] = None,
        hidden_input_scale: Optional[nl.NkiTensor] = None,
    ):
        self.hidden_input_scale = hidden_input_scale
        self.hidden_size = down_proj_weights_tensor.shape[-1]
        if hidden_input_scale != None:
            # Pre-quantized MX input, in SBUF or HBM: [H0, H/512, T]
            kernel_assert(len(hidden_tensor.shape) == 3, "Pre-quantized input must have 3D shape [H0, H/512, T]")
            self.sequence_len = hidden_tensor.shape[2]
        else:
            kernel_assert(hidden_tensor.buffer != nl.sbuf, "SBUF input requires a pre-quantized hidden_input_scale")
            kernel_assert(len(hidden_tensor.shape) == 2, "HBM input must have 2D shape [T, H]")
            self.sequence_len = hidden_tensor.shape[0]

        self.hidden_tensor = hidden_tensor
        self.gate_proj_weights_tensor = gate_proj_weights_tensor
        self.down_proj_weights_tensor = down_proj_weights_tensor
        self.activation_fn = activation_fn
        self.gate_clamp_lower_limit = gate_clamp_lower_limit
        self.gate_clamp_upper_limit = gate_clamp_upper_limit
        self.up_clamp_lower_limit = up_clamp_lower_limit
        self.up_clamp_upper_limit = up_clamp_upper_limit

        self.bias_params = MLPBiasParameters(gate_proj_bias_tensor, down_proj_bias_tensor)
        self.expert_params = expert_params
        self.quant_params = MLPQuantizationParameters(gate_w_scale, down_w_scale)
