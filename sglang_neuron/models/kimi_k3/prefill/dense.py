# SPDX-License-Identifier: Apache-2.0
"""The dense MLP of the Kimi-K3 prefill (the first layer), intermediate-sharded, in the sequence-parallel form.

Rank ``r`` holds the rows ``r I / W .. (r + 1) I / W - 1`` of the gate and up projections and the same columns of the
down projection. One all-gather gives the input rows of all ranks, and a reduce-scatter by row sums the partial
outputs and gives each rank its row block.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def situ_and_mul(gate_up: Tensor, beta: float, linear_beta: float) -> Tensor:
    """Return ``SiTU(gate) * LinearSiTU(up)`` of ``gate_up = [gate | up]`` in fp32, in the input dtype.

    ``SiTU(g) = beta tanh(g / beta) sigmoid(g)`` (the gate saturates at ``+beta``) and
    ``LinearSiTU(u) = linear_beta tanh(u / linear_beta)``.
    """
    d = gate_up.shape[-1] // 2
    gate = gate_up[..., :d].to(torch.float32)
    up = gate_up[..., d:].to(torch.float32)
    situ = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    up = linear_beta * torch.tanh(up / linear_beta)
    return (situ * up).to(gate_up.dtype)


class Weight(nn.Module):
    """A module that holds one tensor as ``weight`` (the checkpoint name of a norm or a linear layer)."""

    def __init__(self, weight: Tensor):
        super().__init__()
        self.register_buffer("weight", weight)

    def extra_repr(self) -> str:
        return f"{tuple(self.weight.shape)}, {self.weight.dtype}"


class DenseMlp(nn.Module):
    """The intermediate-sharded dense MLP: ``gate_proj``, ``up_proj`` ``[I / W, H]`` and ``down_proj`` ``[H, I / W]``.

    Example:
        >>> mlp = DenseMlp(gate, up, down, situ_beta=4.0, situ_linear_beta=25.0)
        >>> y_rows = mlp(x_rows, comm)  # [T / W, H] -> [T / W, H]
    """

    def __init__(self, gate: Tensor, up: Tensor, down: Tensor, *, situ_beta: float, situ_linear_beta: float):
        super().__init__()
        self.gate_proj = Weight(gate)
        self.up_proj = Weight(up)
        self.down_proj = Weight(down)
        self.situ_beta = float(situ_beta)
        self.situ_linear_beta = float(situ_linear_beta)

    def forward(self, x_rows: Tensor, comm, row_sharded: bool = True) -> Tensor:
        """Run the MLP on the row block ``x_rows [T / W, H]``; give the row block of the output.

        ``row_sharded=False`` takes all ``T`` rows on every rank and gives all rows (one all-reduce).
        """
        x = comm.all_gather(x_rows, 0) if row_sharded else x_rows
        gate_up = torch.cat([F.linear(x, self.gate_proj.weight), F.linear(x, self.up_proj.weight)], dim=-1)
        y = F.linear(situ_and_mul(gate_up, self.situ_beta, self.situ_linear_beta), self.down_proj.weight)
        return comm.reduce_scatter(y, 0) if row_sharded else comm.all_reduce(y)
