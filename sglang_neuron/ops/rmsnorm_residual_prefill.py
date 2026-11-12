# SPDX-License-Identifier: Apache-2.0
"""The prefill's input norms: nkilib's rmsnorm_residual_prefill adds the residual on the DMA engines as it loads and
norms the sum, returning both, so the sum becomes the next residual without another pass."""

from typing import Optional

import torch
from nkilib.experimental.subkernels.rmsnorm_residual_prefill import rmsnorm_residual_prefill as _nki_rmsnorm_residual
from torch import Tensor

from sglang_neuron.backend.stack import wrap_nki
from sglang_neuron.utils.hardware_config import get_logical_nc_config
from sglang_neuron.utils.neuron_utils import can_run_kernel


def rmsnorm_residual_prefill(hidden_states: Tensor, *, gamma: Tensor, residual: Optional[Tensor], eps: float,
                             unpadded_hidden_size: int):
    """``(norm, summed)`` of ``hidden_states + residual`` [T, H] (just hidden_states without a residual, and summed
    None). The mean of the squares is over ``unpadded_hidden_size``: the padding is zero, wherever the shuffled
    hidden order puts it."""
    gamma = gamma.unsqueeze(0)
    if not can_run_kernel(hidden_states):
        x = hidden_states if residual is None else hidden_states + residual
        xf = x.float()
        variance = xf.pow(2).sum(-1, keepdim=True) / unpadded_hidden_size
        return (xf * torch.rsqrt(variance + eps) * gamma.float()).to(hidden_states.dtype), \
            None if residual is None else x
    outs = wrap_nki(_nki_rmsnorm_residual)[get_logical_nc_config()](
        hidden=hidden_states, gamma=gamma, residual=residual, eps=eps, hidden_actual=unpadded_hidden_size,
    )
    if residual is None:
        out = outs[0] if isinstance(outs, (list, tuple)) else outs
        return out.view_as(hidden_states), None
    out, summed = outs
    return out.view_as(hidden_states), summed.view_as(hidden_states)
