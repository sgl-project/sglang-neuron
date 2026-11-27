# SPDX-License-Identifier: Apache-2.0
"""Hy3's SiLU-gated MLPs whose intermediate dimension shards over the attention TP: the dense layers' MLP and the MoE
block's shared expert. At TP64 a rank's shard is 208 rows (dense) or 24 (shared expert); each is zero-padded to a
multiple of 128 rows (``PAD``), the shapes the compiler handles. A zero row adds silu(0) * 0 = 0, so the padding is
exact."""

import torch
import torch.nn.functional as F
from torch import nn

from sglang_neuron.models.transformer.config import DTYPE
from sglang_neuron.parallel import collectives as neuron_collectives
from sglang_neuron.parallel import groups
from sglang_neuron.utils.weight_loader import SafetensorsWeightLoader, pad_to_shape, set_weight_loader

PAD = 128


def padded_rows(size: int, tp: int) -> tuple:
    """(this rank's rows, padded) of an intermediate dimension of ``size`` over ``tp`` ranks."""
    if size % tp:
        raise ValueError(f"an intermediate size of {size} does not shard over {tp} ranks")
    rows = size // tp
    return rows, -(-rows // PAD) * PAD


def gate_up_loader(rows: int, padded: int, hidden: int) -> SafetensorsWeightLoader:
    """gate_proj and up_proj ([I, H] each) as [H, 2 * padded]: this rank's gate rows, then its up rows, each padded."""

    def transform(slices: list, rank: int) -> torch.Tensor:
        gate, up = (pad_to_shape(s[rank * rows:(rank + 1) * rows], (padded, hidden)) for s in slices)
        return torch.cat([gate, up], dim=0).t()

    return SafetensorsWeightLoader(transform=transform)


def down_loader(rows: int, padded: int, hidden: int) -> SafetensorsWeightLoader:
    """down_proj ([H, I]) as [padded, H]: this rank's input columns, padded."""
    return SafetensorsWeightLoader(transform=lambda slices, rank: pad_to_shape(
        slices[0][:, rank * rows:(rank + 1) * rows].t(), (padded, hidden)))


class ShardedMLP(nn.Module):
    """down(silu(gate(x)) * up(x)) for this rank's padded shard of the intermediate dimension; ``partial`` returns
    this rank's share (summed over the attention TP by the caller), ``forward`` the all-reduced output. ``real`` (the
    prefill's real-token mask the MoE layers read) is ignored."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.attn_tp_group = groups.get_neuron_attn_tp_group()
        rows, padded = padded_rows(intermediate_size, self.attn_tp_group.world_size)
        self.gate_up_weight = nn.Parameter(torch.empty(hidden_size, 2 * padded, dtype=DTYPE))
        self.down_weight = nn.Parameter(torch.empty(padded, hidden_size, dtype=DTYPE))
        set_weight_loader(self.gate_up_weight, gate_up_loader(rows, padded, hidden_size))
        set_weight_loader(self.down_weight, down_loader(rows, padded, hidden_size))

    def partial(self, x):
        gate, up = (x @ self.gate_up_weight).chunk(2, dim=-1)
        return (F.silu(gate) * up) @ self.down_weight

    def forward(self, x, real=None):
        return neuron_collectives.all_reduce(self.attn_tp_group, self.partial(x))
