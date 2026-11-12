# SPDX-License-Identifier: Apache-2.0
"""The collectives of one Kimi-K3 prefill stage: its TP group, which is also its DCP and EP group.

The stage is one SPMD graph for every rank of the group: a rank-dependent slice takes the rank from ``rank_tensor``
(graph data), never from the Python ``rank``. Row-sharded tensors use the row split of ``all_gather(dim=0)`` and
``reduce_scatter(dim=0)``: rank ``r`` holds rows ``r * n .. r * n + n - 1``.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol


@runtime_checkable
class StageComm(Protocol):
    rank: int
    world: int
    rank_tensor: torch.Tensor  # int64 [1] = rank

    def all_gather(self, x: torch.Tensor, dim: int = 0) -> torch.Tensor: ...

    def reduce_scatter(self, x: torch.Tensor, dim: int = 0) -> torch.Tensor: ...

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor: ...


class SingleRankComm:
    """The collectives of a one-rank stage: each is the identity."""

    rank = 0
    world = 1

    def __init__(self, device: torch.device | str | None = None):
        self.rank_tensor = torch.zeros(1, dtype=torch.int64, device=device)

    def all_gather(self, x: torch.Tensor, dim: int = 0) -> torch.Tensor:
        return x

    def reduce_scatter(self, x: torch.Tensor, dim: int = 0) -> torch.Tensor:
        return x

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        return x


class GroupComm:
    """The collectives of a ``torch.distributed`` process group: gloo on the host, the device TP group in a graph.

    Each collective is a functional collective (a new tensor), which torch-neuronx lowers into the stage graph with
    the group's replica groups.
    """

    def __init__(self, group: dist.ProcessGroup | None = None, device: torch.device | str | None = None):
        self.group = dist.group.WORLD if group is None else group
        self.rank = dist.get_rank(self.group)
        self.world = dist.get_world_size(self.group)
        self.rank_tensor = torch.tensor([self.rank], dtype=torch.int64, device=device)

    def all_gather(self, x: torch.Tensor, dim: int = 0) -> torch.Tensor:
        if self.world == 1:
            return x
        dim = dim % x.dim()
        source = x.movedim(dim, 0).contiguous()
        return funcol.all_gather_tensor(source, 0, self.group).movedim(0, dim)

    def reduce_scatter(self, x: torch.Tensor, dim: int = 0) -> torch.Tensor:
        if self.world == 1:
            return x
        dim = dim % x.dim()
        if x.shape[dim] % self.world:
            raise ValueError(f"reduce_scatter of {x.shape[dim]} rows over {self.world} ranks: the rows must divide")
        source = x.movedim(dim, 0).contiguous()
        return funcol.reduce_scatter_tensor(source, "sum", 0, self.group).movedim(0, dim)

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        if self.world == 1:
            return x
        return funcol.all_reduce(x, "sum", self.group)


def local_row_block(full: torch.Tensor, comm: StageComm) -> torch.Tensor:
    """This rank's row block of a tensor with all rows (``[T, ...]`` -> ``[T / world, ...]``), indexed by
    ``rank_tensor`` so every rank traces one graph."""
    rows = full.shape[0]
    if rows % comm.world:
        raise ValueError(f"{rows} rows do not divide into {comm.world} row blocks")
    blocks = full.reshape(comm.world, -1).index_select(0, comm.rank_tensor.reshape(-1)[:1].to(torch.int64))
    return blocks.view(rows // comm.world, *full.shape[1:])


def gather_owned_rows(local: torch.Tensor, rows: torch.Tensor, comm: StageComm) -> torch.Tensor:
    """The global rows ``rows`` of a row-sharded tensor, on every rank.

    Each rank takes the selected rows it holds and zeros elsewhere; one all-reduce adds the parts. Exactly one rank
    holds each row, so the result is exact. A row may repeat.

    Args:
        local: ``[n, ...]`` this rank's row block (global rows ``rank * n .. rank * n + n - 1``).
        rows: ``[k]`` int global row indices.
        comm: The stage collectives.

    Returns:
        ``[k, ...]``, equal on every rank.
    """
    n = local.shape[0]
    index = rows.reshape(-1).to(torch.int64) - comm.rank_tensor.reshape(-1)[:1].to(torch.int64) * n
    owned = (index >= 0) & (index < n)
    picked = local.index_select(0, index.clamp(0, n - 1))
    owned = owned.reshape(-1, *([1] * (local.dim() - 1)))
    picked = torch.where(owned, picked, torch.zeros_like(picked))
    return comm.all_reduce(picked)


def valid_row_mask(valid_len: torch.Tensor, num_rows: int, comm: StageComm) -> torch.Tensor:
    """``[T / world]`` bool: True for the rows of this rank's row block before ``valid_len`` (the non-pad rows)."""
    local = num_rows // comm.world
    first = comm.rank_tensor.reshape(-1)[:1].to(torch.int64) * local
    rows = first + torch.arange(local, dtype=torch.int64, device=valid_len.device)
    return rows < valid_len.reshape(-1)[:1].to(torch.int64)
