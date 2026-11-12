"""The rows of the Kimi-K3 MLA cache and their owner layout.

A cache row is ``[kv_a_layernorm(latent) | raw rope]`` (the rope part is stored as the projection gives it and is
never rotated), cast to ``float8_e4m3fn`` with scale 1.0 after a clamp to the finite range. A pool keeps the rows as
uint8 bytes, ``[num_blocks, STRIPE, width]``; block 0 is the reserved null block.

Decode context parallelism stripes the positions over the ranks: stripe ``s = pos // STRIPE`` belongs to rank
``s % world``. The positions of one rank fill consecutive rank-local slots in position order. A chunk of whole stripe
cycles (``STRIPE * world`` rows) holds the same number of rows of every rank, at any start: a prompt-cache hit can end
at any token, so the next chunk can start inside a stripe. Its rank-local prior (``local_prior``) is then not the same
on every rank.
"""

from __future__ import annotations

import torch

FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = 448.0
"""The largest finite ``float8_e4m3fn`` magnitude: a plain cast gives NaN from 464 up, so rows are clamped first."""

STRIPE = 128
"""Tokens of one DCP stripe, and rows of one pool block."""


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Normalize in fp32, round to the input dtype, then multiply by the weight (in the input dtype)."""
    dtype = x.dtype
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return weight * xf.to(dtype)


def cache_rows(latent: torch.Tensor, norm_weight: torch.Tensor, eps: float, kv_lora_rank: int) -> torch.Tensor:
    """Give the cache rows ``[..., R + DR]`` of the ``kv_a_proj_with_mqa`` output: normed latent, then the raw rope."""
    c = rms_norm(latent[..., :kv_lora_rank], norm_weight, eps)
    return torch.cat([c, latent[..., kv_lora_rank:]], dim=-1)


def quantize_rows(rows: torch.Tensor) -> torch.Tensor:
    """Give the ``float8_e4m3fn`` rows (scale 1.0): clamp to +-``FP8_MAX``, then round to nearest even."""
    return rows.float().clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)


def row_bytes(rows: torch.Tensor) -> torch.Tensor:
    """Give the uint8 bytes of FP8 rows (a view, no copy)."""
    return rows.contiguous().view(torch.uint8)


def byte_values(data: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Give the values of FP8 bytes in ``dtype``. A NaN code (bytes no write gave) reads as 0."""
    values = data.contiguous().view(FP8_DTYPE).to(dtype)
    return torch.nan_to_num(values, nan=0.0)


def owned_rows(rows: int, rank: torch.Tensor, world: int, stripe: int = STRIPE) -> torch.Tensor:
    """Give the chunk rows of ``rank``, ``[rows // world]`` int64 in stripe order.

    The chunk starts at a multiple of ``stripe * world``, so row ``i`` is in stripe ``i // stripe`` of the chunk and
    belongs to rank ``(i // stripe) % world``. The rank is tensor data: one graph serves every rank.

    Example:
        >>> owned_rows(512, torch.tensor([1]), 2)[:2], owned_rows(512, torch.tensor([1]), 2)[128]
        (tensor([128, 129]), tensor(384))
    """
    cycle = stripe * world
    if rows % cycle:
        raise ValueError(f"A chunk of {rows} rows does not tile by stripe x world = {cycle}. Use a multiple of it.")
    blocks = rank.reshape(-1)[:1].to(torch.int64) + world * torch.arange(rows // cycle, device=rank.device)
    return (blocks[:, None] * stripe + torch.arange(stripe, device=rank.device)[None, :]).reshape(-1)


def local_prior(start: torch.Tensor, rank: torch.Tensor, world: int, stripe: int = STRIPE) -> torch.Tensor:
    """Give the rank-local slots before position ``start``: ``[1]`` int64, the count of the positions before ``start``
    that the stripes of ``rank`` hold. The start and the rank are tensor data, so one graph serves every rank and
    every start. A start of whole stripe cycles gives ``start // world`` on every rank.

    Example:
        >>> local_prior(torch.tensor([300]), torch.tensor([0]), 2), local_prior(torch.tensor([300]), torch.tensor([1]), 2)
        (tensor([172]), tensor([128]))
    """
    cycle = stripe * world
    start = start.reshape(-1)[:1].to(torch.int64)
    rank = rank.reshape(-1)[:1].to(torch.int64).to(start.device)
    partial = (start % cycle - rank * stripe).clamp(0, stripe)
    return torch.div(start, cycle, rounding_mode="floor") * stripe + partial


def owned_rows_from(start: torch.Tensor, rows: int, rank: torch.Tensor, world: int,
                    stripe: int = STRIPE) -> torch.Tensor:
    """Give the rows of ``rank`` in a chunk of ``rows`` rows that starts at position ``start``: ``[rows // world]``
    int64 in position order. Row ``i`` holds position ``start + i``, and the rank owns it when that position is in one
    of its stripes. A chunk of whole stripe cycles holds ``rows // world`` positions of each rank at any start: they
    are the rank-local slots ``local_prior(start) + m``. A start of whole stripe cycles gives ``owned_rows``.

    Example:
        >>> owned_rows_from(torch.tensor([300]), 512, torch.tensor([1]), 2)[:2].tolist()
        [84, 85]
    """
    cycle = stripe * world
    if rows % cycle:
        raise ValueError(f"A chunk of {rows} rows does not tile by stripe x world = {cycle}. Use a multiple of it.")
    start = start.reshape(-1)[:1].to(torch.int64)
    rank = rank.reshape(-1)[:1].to(torch.int64).to(start.device)
    local = local_prior(start, rank, world, stripe) + torch.arange(rows // world, device=start.device)
    position = (torch.div(local, stripe, rounding_mode="floor") * world + rank) * stripe + local % stripe
    return position - start


def local_slot_rows(block_table: torch.Tensor, slots: torch.Tensor, stripe: int = STRIPE) -> torch.Tensor:
    """Give the flat pool row of each rank-local logical slot: ``block_table[j // stripe] * stripe + j % stripe``.

    A negative table entry reads the null block 0.
    """
    table = block_table.reshape(-1).to(torch.int64).clamp_min(0)
    slots = slots.to(torch.int64)
    return table.index_select(0, torch.div(slots, stripe, rounding_mode="floor")) * stripe + slots % stripe


def owner_write_slots(
    cached_len: int,
    rows: int,
    valid_len: int,
    block_table: torch.Tensor,
    rank: int,
    world: int,
    stripe: int = STRIPE,
) -> torch.Tensor:
    """Give the ``write_slots`` of one rank for a chunk: the pool row of each row the rank owns, -1 for the others.

    A pad row (at or after ``valid_len``) gets -1. ``cached_len`` can be any count of tokens (a chunk after a prefix
    hit inside a stripe). Host helper (plain ints): the runtime builds the slots this way from the rank's block table.

    Example:
        >>> owner_write_slots(0, 256, 200, torch.tensor([3, 5]), 1, 2)[127:130]
        tensor([ -1, 384, 385], dtype=torch.int32)
    """
    start, who = torch.tensor([int(cached_len)]), torch.tensor([int(rank)])
    own = owned_rows_from(start, rows, who, world, stripe)
    local = local_prior(start, who, world, stripe) + torch.arange(own.shape[0])
    slots = torch.full((rows,), -1, dtype=torch.int64)
    keep = own < valid_len
    slots[own[keep]] = local_slot_rows(block_table, local[keep], stripe)
    return slots.to(torch.int32)
