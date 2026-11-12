# SPDX-License-Identifier: Apache-2.0
from typing import Literal

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed._functional_collectives import all_gather_tensor

from sglang_neuron.parallel import collectives as neuron_collectives


class ColumnParallelLinear(nn.Module):
    """A linear layer whose output columns are sharded over ``tp_group``: each rank holds
    ``out_features / tp`` rows of the [out, in] weight (columns of the [in, out] weight with ``weight_layout="in_out"``,
    the layout the NKI kernels take) and, with ``bias``, of the bias. With ``gather_output`` the forward all-gathers
    the shards' outputs; without, each rank keeps its own. The owner sets the loaders."""

    def __init__(self, in_features: int, out_features: int, *, dtype, gather_output: bool, tp_group, bias: bool = False,
                 weight_layout: Literal["out_in", "in_out"] = "out_in"):
        super().__init__()
        self.tp_group = tp_group
        self.tp_size = dist.get_world_size(tp_group) if dist.is_initialized() else 1
        if out_features % self.tp_size:
            raise ValueError(f"{out_features} output columns do not shard evenly over {self.tp_size} ranks")
        self.out_features_per_rank = out_features // self.tp_size
        self.weight_layout = weight_layout
        shape = (self.out_features_per_rank, in_features)
        self.weight = nn.Parameter(torch.empty(shape if weight_layout == "out_in" else shape[::-1], dtype=dtype))
        self.bias = nn.Parameter(torch.empty(self.out_features_per_rank, dtype=dtype)) if bias else None
        self.gather_output = gather_output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = _linear(x, self.weight, self.bias, self.weight_layout)
        if self.tp_size == 1 or not self.gather_output:
            return out
        return all_gather_tensor(out, gather_dim=1, group=self.tp_group)


class RowParallelLinear(nn.Module):
    """A linear layer whose input features are sharded over ``tp_group``: each rank holds ``in_features / tp``
    columns of the [out, in] weight (rows of the [in, out] weight with ``weight_layout="in_out"``) and multiplies
    its shard of the input; the forward all-reduces the ranks' partial outputs, then adds the full ``bias``. The owner
    sets the loaders."""

    def __init__(self, in_features: int, out_features: int, *, dtype, tp_group, bias: bool = False,
                 weight_layout: Literal["out_in", "in_out"] = "out_in"):
        super().__init__()
        self.tp_group = tp_group
        self.tp_size = dist.get_world_size(tp_group) if dist.is_initialized() else 1
        if in_features % self.tp_size:
            raise ValueError(f"{in_features} input features do not shard evenly over {self.tp_size} ranks")
        self.in_features_per_rank = in_features // self.tp_size
        self.weight_layout = weight_layout
        shape = (out_features, self.in_features_per_rank)
        self.weight = nn.Parameter(torch.empty(shape if weight_layout == "out_in" else shape[::-1], dtype=dtype))
        self.bias = nn.Parameter(torch.empty(out_features, dtype=dtype)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = _linear(x, self.weight, None, self.weight_layout)
        if self.tp_size > 1:
            out = neuron_collectives.all_reduce_pg(self.tp_group, out, "sum", self.tp_size)
        return out if self.bias is None else out + self.bias


def _linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None, weight_layout: str) -> torch.Tensor:
    if weight_layout == "out_in":
        return F.linear(x, weight, bias)
    out = torch.matmul(x, weight)
    return out if bias is None else out + bias
