# SPDX-License-Identifier: Apache-2.0
"""MiMo-V2's RoPE in the graphs' head layout (weights.qk_order): the kernels rotate channel pairs
(j, j + KV_HEAD_DIM / 2) as contiguous halves, so a table holds ``rotary_dim / 2`` frequencies and then cos 1 / sin 0
for the channels each half passes through. Torch only, so the CPU checks run it as the graphs do."""

import torch
from torch import nn

from sglang_neuron.models.mimo_v2.config import KV_HEAD_DIM
from sglang_neuron.models.transformer.config import DTYPE


class RotaryTables(nn.Module):
    """cos and sin [T, KV_HEAD_DIM / 2] of one RoPE base: ``rotary_dim / 2`` frequencies, then cos 1 and sin 0 for
    the channels each half passes through (the kernels rotate channel pairs (j, j + KV_HEAD_DIM / 2))."""

    def __init__(self, rope_theta: float, rotary_dim: int):
        super().__init__()
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.int64).float() / rotary_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.pass_through = KV_HEAD_DIM // 2 - rotary_dim // 2

    def forward(self, positions: torch.Tensor) -> tuple:
        angles = positions.float()[:, None] * self.inv_freq[None, :]
        cos, sin = angles.cos(), angles.sin()
        cos = torch.nn.functional.pad(cos, (0, self.pass_through), value=1.0)
        sin = torch.nn.functional.pad(sin, (0, self.pass_through), value=0.0)
        return cos.to(DTYPE), sin.to(DTYPE)


def rotate(t: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """RoPE over contiguous halves of t [T, heads, KV_HEAD_DIM]; cos / sin [T, KV_HEAD_DIM / 2]."""
    half = t.shape[-1] // 2
    c = torch.cat([cos, cos], dim=-1).unsqueeze(1).to(torch.float32)
    s = torch.cat([sin, sin], dim=-1).unsqueeze(1).to(torch.float32)
    t32 = t.float()
    rotated = torch.cat([-t32[..., half:], t32[..., :half]], dim=-1)
    return (t32 * c + rotated * s).to(t.dtype)
