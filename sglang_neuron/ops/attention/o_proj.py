# SPDX-License-Identifier: Apache-2.0
"""The prefill output projection: nkilib's output_projection_cte over the attention output's [B, heads, S, d]
layout."""

from nkilib.core.output_projection.output_projection_cte import output_projection_cte
from nkilib.core.utils.common_types import DtypeMode, OProjAttentionLayout, QuantizationType
from torch import Tensor

from sglang_neuron.backend.stack import wrap_nki
from sglang_neuron.utils.hardware_config import get_logical_nc_config
from sglang_neuron.utils.neuron_utils import can_run_kernel

# the most heads per rank the kernel is validated for
MAX_HEADS = 17


def o_proj(attention: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
    """attention [B, heads, S, d] @ weight [heads * d, H] + bias [1, H] -> [B, S, H]."""
    if not can_run_kernel(attention):
        B, heads, S, d = attention.shape
        return attention.permute(0, 2, 1, 3).reshape(B, S, heads * d) @ weight + bias
    return wrap_nki(output_projection_cte)[get_logical_nc_config()](
        attention=attention,
        weight=weight,
        bias=bias,
        quantization_type=QuantizationType.NONE,
        input_scales=None,
        weight_scales=None,
        output_dtype=None,
        dtype_mode=DtypeMode.NON_OCP,
        compact_weight_scales=False,
        attention_layout=OProjAttentionLayout.BNSd,
    )
