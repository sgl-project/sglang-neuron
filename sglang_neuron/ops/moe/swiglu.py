# SPDX-License-Identifier: Apache-2.0
"""GPT-OSS's expert MLP in torch, the CPU path of the BF16 SwiGLU MoE kernels: gate clamped from above, up clamped on
both sides, and nkilib's Swish, x * sigmoid(1.702 x), on the gate."""

import torch
from torch import Tensor

# the slope of nkilib's Swish (ActFnType.Swish), a constant in the kernels
SWISH_ALPHA = 1.702


def swiglu_mlp(x: Tensor, *, gate_up: Tensor, gate_up_bias: Tensor, down: Tensor, down_bias: Tensor,
               gate_clamp_upper_limit: float, up_clamp_upper_limit: float, up_clamp_lower_limit: float) -> Tensor:
    """Each expert n's MLP over its rows x [N, B, H] in fp32: gate/up [N, H, 2, I] (gate at 0, up at 1) with biases
    [N, 2, I], down [N, I, H] with biases [N, H]."""
    x = x.float()
    gate = torch.einsum("nbh,nhi->nbi", x, gate_up[:, :, 0].float()) + gate_up_bias[:, None, 0].float()
    up = torch.einsum("nbh,nhi->nbi", x, gate_up[:, :, 1].float()) + gate_up_bias[:, None, 1].float()
    gate = gate.clamp(max=gate_clamp_upper_limit)
    up = up.clamp(min=up_clamp_lower_limit, max=up_clamp_upper_limit)
    glu = gate * torch.sigmoid(SWISH_ALPHA * gate) * up
    return torch.einsum("nbi,nih->nbh", glu, down.float()) + down_bias[:, None].float()
