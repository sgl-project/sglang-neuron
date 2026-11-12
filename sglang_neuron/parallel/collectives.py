# SPDX-License-Identifier: Apache-2.0
"""Traceable collectives on SGLang's own GroupCoordinator objects.

Use c10d directly so Neuron's compiler lowers each operation into the model
graph. SGLang retains ownership of process groups and host communication.

The graphs trace torch.distributed's functional collectives, which torch-neuronx's
collective legalization lowers into the NEFF with the group's replica groups
(backend/stack.py). Each function returns a new tensor, whose result the caller
must use.
"""
from __future__ import annotations

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
from sglang.srt.distributed.parallel_state import GroupCoordinator

_OPS = {"sum": dist.ReduceOp.SUM, "max": dist.ReduceOp.MAX, "min": dist.ReduceOp.MIN}


def all_reduce(group: GroupCoordinator, tensor: torch.Tensor, op: str = "sum") -> torch.Tensor:
    """The reduction (``op``: sum, max or min) over the group, as a new tensor."""
    return all_reduce_pg(group.device_group, tensor, op, group.world_size)


def all_reduce_pg(process_group, tensor: torch.Tensor, op: str = "sum", world_size: int | None = None) -> torch.Tensor:
    """``all_reduce`` on a bare process group (a layer that holds the group itself)."""
    if world_size is None:
        world_size = dist.get_world_size(process_group)
    if world_size == 1:
        return tensor
    if op not in _OPS:
        raise ValueError(f"all_reduce op {op!r}: use one of {sorted(_OPS)}")
    return funcol.all_reduce(tensor, op, process_group)


def all_gather(group: GroupCoordinator, tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    if group.world_size == 1:
        return tensor
    if dim < 0:
        dim += tensor.ndim
    source = tensor.movedim(dim, 0).contiguous()
    return funcol.all_gather_tensor(source, 0, group.device_group).movedim(0, dim)


def reduce_scatter(group: GroupCoordinator, tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    if group.world_size == 1:
        return tensor
    if dim < 0:
        dim += tensor.ndim
    source = tensor.movedim(dim, 0).contiguous()
    if source.shape[0] % group.world_size:
        raise ValueError("reduce_scatter dimension must be divisible by the group size")
    return funcol.reduce_scatter_tensor(source, "sum", 0, group.device_group).movedim(0, dim)
