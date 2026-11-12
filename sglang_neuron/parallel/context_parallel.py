# SPDX-License-Identifier: Apache-2.0
"""Traced helpers of prefill and decode context parallelism over block-cyclic KV (mem_cache/cp_pages.py).

Every helper takes the CP rank as a tensor (``rank_t``, ``[1]`` int64) and selects with a one-hot mask, never a
rank-indexed slice: the graph is then the same on every rank, so all ranks of a group share one NEFF, and neuronx-cc
does not see a rank-dependent slice feeding a collective (which deadlocks its scheduler). The collective is passed in
(``all_reduce(x) -> x`` summed over the group), so the same code runs a device group or a CPU simulation of the ranks.

* Prefill CP (PCP) splits a replicated sub-block's rows (projections, the DSA indexer's scoring) contiguously over the
  group: ``pcp_rows`` keeps this rank's ``1 / size`` block, ``pcp_gather`` reassembles every rank's block with one sum
  all-reduce. Attention itself is not split this way (a contiguous causal split is not load-balanced).
* Decode / prefill over a CP-sharded latent: ``cp_window`` gathers a sequence's KV window, each rank contributing the
  blocks it owns (zeros elsewhere) to one sum all-reduce; ``merge_partials`` is the exact log-sum-exp merge of
  per-rank partial attention outputs, for a kernel that attends only its own shard.
"""

from __future__ import annotations

from typing import Callable, Sequence

import torch

AllReduce = Callable[[torch.Tensor], torch.Tensor]


def _onehot(rank_t: torch.Tensor, size: int, dtype, ndim: int, dim: int) -> torch.Tensor:
    hot = (torch.arange(size, device=rank_t.device) == rank_t.reshape(-1)[0]).to(dtype)
    return hot.view(*([1] * dim), size, *([1] * (ndim - dim)))


def pcp_rows(x: torch.Tensor, rank_t: torch.Tensor, size: int, dim: int = 1) -> torch.Tensor:
    """This rank's contiguous ``1 / size`` block of ``x`` along ``dim`` (which ``size`` must divide)."""
    if size == 1:
        return x
    if x.shape[dim] % size:
        raise ValueError(f"PCP {size} does not divide dimension {dim} of {tuple(x.shape)}")
    blocks = x.unflatten(dim, (size, -1))
    return (blocks * _onehot(rank_t, size, x.dtype, x.ndim, dim)).sum(dim).to(x.dtype)


def pcp_gather(x: torch.Tensor, rank_t: torch.Tensor, size: int, all_reduce: AllReduce, dim: int = 1) -> torch.Tensor:
    """The inverse of :func:`pcp_rows`: every rank's block along ``dim``, in rank order, on every rank."""
    if size == 1:
        return x
    blocks = x.unsqueeze(dim) * _onehot(rank_t, size, x.dtype, x.ndim, dim)
    return all_reduce(blocks).flatten(dim, dim + 1)


def cp_window(cache: torch.Tensor, native_slots: torch.Tensor, rank_t: torch.Tensor, *, cp_size: int, block: int,
              all_reduce: AllReduce) -> torch.Tensor:
    """The ``[W, ...]`` KV window at native slots ``native_slots`` ``[W]`` (int64) of a block-cyclic CP pool layer
    ``cache`` ``[blocks, block, ...]``, assembled on every rank of the group: each rank reads the slots it owns from its
    shard (block 0 elsewhere, masked to zero) and one sum all-reduce combines them."""
    page = cp_size * block
    slots = native_slots.to(torch.int64)
    off = slots % page
    owner = off // block
    mine = owner == rank_t.reshape(-1)[0]
    blk = torch.where(mine, slots // page, torch.zeros_like(slots))
    pos = torch.where(mine, off % block, torch.zeros_like(slots))
    rows = cache[blk, pos]
    mask = mine.view(-1, *([1] * (rows.ndim - 1))).to(rows.dtype)
    return all_reduce(rows * mask) if cp_size > 1 else rows


def merge_partials(outs: Sequence[torch.Tensor], lses: Sequence[torch.Tensor]) -> torch.Tensor:
    """The exact attention output from per-rank partials: ``outs[r]`` ``[..., D]`` normalized over rank ``r``'s keys
    and ``lses[r]`` ``[...]`` the log-sum-exp of its scores (``-inf`` for a rank with no key of the row). A row no rank
    has a key for is 0."""
    lse = torch.stack([l.float() for l in lses])
    top = lse.max(dim=0).values
    safe = torch.where(torch.isfinite(top), top, torch.zeros_like(top))
    weights = torch.exp(lse - safe)
    total = weights.sum(dim=0)
    weights = weights / torch.where(total > 0, total, torch.ones_like(total))
    out = sum(w.unsqueeze(-1) * o.float() for w, o in zip(weights, outs))
    return out.to(outs[0].dtype)


def lse_of(row_max: torch.Tensor, row_sum: torch.Tensor) -> torch.Tensor:
    """A partial's log-sum-exp from the (row max, row sum of ``exp(s - max)``) pair a flash kernel carries."""
    return row_max.float() + torch.log(row_sum.float())
