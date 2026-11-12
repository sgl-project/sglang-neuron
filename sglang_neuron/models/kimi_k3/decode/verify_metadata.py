# SPDX-License-Identifier: Apache-2.0
"""In-graph kernel metadata of the Kimi-K3 2-server verify: KDA state pages and slots, MLA kernel table and slots.

The verify program builds these from the host step tensors on the device, so each function keeps its torch ops,
dtypes, shapes and op order. Rank values come in as tensors, so all ranks share one program.

MLA pages: a host page holds ``block_size`` FP8 rows of 576 bytes on each rank (256 rows: two 128-row kernel pages,
one CP stripe each). The host table ``[B, W]`` names host pages; the kernel table ``[B, NB]`` names kernel pages
(``expand_block_table``: host page ``b`` is kernel pages ``2b, 2b + 1``). Kernel page 0 is the null page: an invalid
entry or slot maps to 0, where the kernel reads zeros and skips writes.

KDA pages: each request holds ``1 + Q`` state pages per KDA layer (committed state, then one slot per verify row);
see ``spec_math.verify_kda_slot_pages``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor

from sglang_neuron.models.kimi_k3.decode.spec_math import (
    MLA_BLOCK,
    mask_pad_rank_pages,
    to_token_major,
    valid_pages,
    verify_kda_slot_pages,
    verify_mla_slots,
    window_rows,
)

MLA_LOCAL_BLOCKS = 4
"""Local MLA blocks of the static MLA path."""
MLA_MIN_DYNAMIC_BLOCKS = 8
"""The smallest table of the dynamic MLA path; its tables are powers of two."""
MLA_MAX_LOOP_BLOCKS = 256
"""Local MLA blocks of the largest kernel table of the loop path (1M tokens at CP32)."""
MLA_CP32 = 32
"""The MLA CP of a TP32 replica."""


def mla_page_factor(page_rows: int, rows: int = MLA_BLOCK) -> int:
    """Return the kernel pages in one host page of ``page_rows`` rows."""
    if page_rows <= 0 or rows <= 0 or page_rows % rows:
        raise ValueError(f"The MLA page has {page_rows} rows. Use a multiple of {rows} rows.")
    return page_rows // rows


def expand_block_table(table: Tensor, factor: int) -> Tensor:
    """Expand a table of host pages to a table of kernel pages: entry ``j`` is ``factor * table[j // factor] + j %
    factor``; a negative entry stays the same negative value."""
    if factor < 1:
        raise ValueError(f"The page factor is {factor}. Use 1 or more.")
    if factor == 1:
        return table
    offsets = torch.arange(factor, dtype=table.dtype, device=table.device)
    expanded = table.unsqueeze(-1) * factor + offsets
    expanded = torch.where(table.unsqueeze(-1) < 0, table.unsqueeze(-1), expanded)
    return expanded.reshape(*table.shape[:-1], table.shape[-1] * factor)


def trim_kernel_table(table: Tensor, kernel_blocks: int | None) -> Tensor:
    """Drop the columns of an expanded table after ``kernel_blocks`` (None keeps all)."""
    if kernel_blocks is None or table.shape[-1] <= kernel_blocks:
        return table
    return table[..., :kernel_blocks]


def mla_kernel_block_count(width: int, max_blocks: int = MLA_MAX_LOOP_BLOCKS, cp_size: int = MLA_CP32) -> int:
    """Return the local block count of the kernel table for a table width: 4 on the static path, else the next power
    of two from 8 (at most ``max_blocks``)."""
    if width <= MLA_LOCAL_BLOCKS:
        return MLA_LOCAL_BLOCKS if cp_size <= MLA_CP32 else max(width, 1)
    blocks = MLA_MIN_DYNAMIC_BLOCKS
    while blocks < width:
        blocks *= 2
    if blocks > max_blocks:
        raise ValueError(
            f"The MLA table has {width} local blocks. The verify MLA body reads at most {max_blocks} local blocks "
            f"({max_blocks * MLA_BLOCK * cp_size} tokens)."
        )
    return blocks


def _row_pages(pages: Tensor, num_pages: int) -> Tensor:
    """Replace each invalid page of a table row by the first valid page of the row (page 0 for a row with none)."""
    pages = pages.to(torch.int32)
    valid = (pages > 0) & (pages < num_pages)
    index = torch.arange(pages.shape[1], device=pages.device).expand_as(pages)
    first = torch.where(valid, index, torch.full_like(index, pages.shape[1])).amin(dim=1, keepdim=True)
    # A masked max, not a gather: the first valid page is the only nonzero entry (0 for a row with no valid page).
    first_page = torch.where(index == first, pages, torch.zeros_like(pages)).amax(dim=1, keepdim=True)
    first_page = torch.where(first < pages.shape[1], first_page, torch.zeros_like(first_page))
    return torch.where(valid, pages, first_page.expand_as(pages)).to(torch.int32)


def _row_pages_gather(pages: Tensor, num_pages: int) -> Tensor:
    """Replace each invalid page of a table row by the first valid page of the row, found by a gather."""
    pages = pages.to(torch.int32)
    valid = (pages > 0) & (pages < num_pages)
    index = torch.arange(pages.shape[1], device=pages.device).expand_as(pages)
    first = torch.where(valid, index, torch.full_like(index, pages.shape[1])).amin(dim=1, keepdim=True)
    first_page = torch.gather(
        torch.cat((pages, torch.zeros_like(pages[:, :1])), dim=1), 1, first.clamp(max=pages.shape[1])
    )
    first_page = torch.where(first < pages.shape[1], first_page, torch.zeros_like(first_page))
    return torch.where(valid, pages, first_page.expand_as(pages))


def _row_runs(pages: Tensor, num_pages: int) -> Tensor:
    """Replace each invalid page of a table row so that each column pair stays a run of 2 pages.

    The packed MLA loop reads the pages of the columns (2c, 2c + 1) as one key group, and needs the run (p, p + 1)
    there for each group of a real sequence. A pair with one valid entry first gets the other page of its run (an even
    page v in an even column gives v + 1, an odd page v in an odd column gives v - 1). Then an invalid entry in column
    c takes p0 + c % 2, where (p0, p0 + 1) is the first valid run of the row. A row with no run takes ``_row_pages``.
    """
    pages = pages.to(torch.int32)
    pairs = pages.shape[1] // 2
    if pairs == 0:
        return _row_pages(pages, num_pages)
    valid = (pages > 0) & (pages < num_pages)
    even, odd = pages[:, 0 : 2 * pairs : 2], pages[:, 1 : 2 * pairs : 2]
    valid_even, valid_odd = valid[:, 0 : 2 * pairs : 2], valid[:, 1 : 2 * pairs : 2]
    # The other page of a half-valid pair, where it is in the pool.
    odd_partner = valid_even & ~valid_odd & (even % 2 == 0) & (even + 1 < num_pages)
    even_partner = valid_odd & ~valid_even & (odd % 2 == 1) & (odd - 1 > 0)
    pages = pages.clone()
    pages[:, 1 : 2 * pairs : 2] = torch.where(odd_partner, even + 1, odd)
    pages[:, 0 : 2 * pairs : 2] = torch.where(even_partner, odd - 1, even)
    fallback = _row_pages(pages, num_pages)
    valid = (pages > 0) & (pages < num_pages)
    even, odd = pages[:, 0 : 2 * pairs : 2], pages[:, 1 : 2 * pairs : 2]
    run = valid[:, 0 : 2 * pairs : 2] & valid[:, 1 : 2 * pairs : 2] & (odd == even + 1)
    index = torch.arange(pairs, device=pages.device).expand_as(run)
    first = torch.where(run, index, torch.full_like(index, pairs)).amin(dim=1, keepdim=True)
    # A masked max, not a gather: the first run start is the only nonzero entry (0 for a row with no run).
    base = torch.where(index == first, even, torch.zeros_like(even)).amax(dim=1, keepdim=True)
    # The column index takes the dtype of the pages, so the table stays int32.
    column = torch.arange(pages.shape[1], device=pages.device, dtype=pages.dtype).expand_as(pages)
    filled = torch.where(first < pairs, base + column % 2, fallback)
    return torch.where(valid, pages, filled).to(torch.int32)


def mla_kernel_block_table(
    block_table: Tensor,
    num_pages: int,
    local_blocks: int | None = None,
    max_blocks: int = MLA_MAX_LOOP_BLOCKS,
    row_pages: bool = False,
    cp_size: int = MLA_CP32,
    row_runs: bool = False,
) -> Tensor:
    """Return the local MLA kernel table of each sequence, ``[B, NB]`` int32.

    A table narrower than the kernel table gets null columns at the end. Without ``row_pages`` each invalid page is
    page 0; with it each invalid page (and pad column) takes a page of its own row (``_row_runs`` with ``row_runs``,
    else the first valid page, ``_row_pages_gather``), for zeroed MLA pages.
    """
    if block_table.ndim != 2:
        raise ValueError("The MLA block table shape is invalid. Use [sequences, blocks].")
    width = block_table.shape[1]
    if local_blocks is None:
        local_blocks = mla_kernel_block_count(width, max_blocks, cp_size)
    if width > local_blocks:
        raise ValueError(f"The MLA table has {width} local blocks. The kernel table has {local_blocks}.")
    if row_pages and row_runs:
        # The pad columns are invalid entries too, so the run fill covers them.
        if width < local_blocks:
            pad = torch.zeros(
                (block_table.shape[0], local_blocks - width), dtype=block_table.dtype, device=block_table.device
            )
            block_table = torch.cat((block_table, pad), dim=1)
        return _row_runs(block_table, num_pages)
    table = _row_pages_gather(block_table, num_pages) if row_pages else valid_pages(block_table, num_pages)
    if width < local_blocks:
        pad = torch.zeros((table.shape[0], local_blocks - width), dtype=table.dtype, device=table.device)
        if row_pages:
            pad = pad + table[:, :1]
        table = torch.cat((table, pad), dim=1)
    return table


def kernel_rank_info(global_rank: Tensor, dp_rank: Tensor, tp_rank: Tensor) -> Tensor:
    """Return ``rank_info`` ``[1, 4]`` int32: (global rank, DP rank, TP rank, 0)."""
    values = [
        global_rank.reshape(1).to(torch.int32),
        dp_rank.reshape(1).to(torch.int32),
        tp_rank.reshape(1).to(torch.int32),
        torch.zeros(1, dtype=torch.int32, device=global_rank.device),
    ]
    return torch.cat(values).reshape(1, 4)


@dataclass(frozen=True)
class VerifyMetadataConfig:
    """The static settings of the verify metadata (one per served form).

    Attributes:
        queries: Q, the verify rows of each request.
        cp_size: the MLA CP of the replica (the TP group, 32).
        kernel_blocks: the kernel MLA table width of a context bucket, or None to take it from the table width
            (``mla_kernel_block_count``; the 1M form has no context buckets).
        max_blocks: the largest kernel table (``MLA_MAX_LOOP_BLOCKS``).
        kda_null_page: the KDA page of an invalid state or slot write (0: the shared null page).
        kda_ranks: the ranks with a real KDA head, or None for all ranks.
        zeroed_mla_pages: the verify reads zeroed MLA pages, so invalid entries take a page of their row.
        row_runs: with ``zeroed_mla_pages``, the run fill of the invalid entries (``_row_runs``).
    """

    queries: int
    cp_size: int = MLA_CP32
    kernel_blocks: int | None = None
    max_blocks: int = MLA_MAX_LOOP_BLOCKS
    kda_null_page: int = 0
    kda_ranks: int | None = None
    zeroed_mla_pages: bool = False
    row_runs: bool = True


@dataclass(frozen=True)
class VerifyLayer:
    """The step inputs of one layer of the verify, in layer order.

    MLA layers share one host table ``[B, W]`` and one slot window ``[B, 2k + 1]`` (int64 host-page slots); each KDA
    layer reads the state table ``[B, 1 + Q]`` of its state group. ``pages`` is the host-page count of the layer's
    pool.
    """

    is_mla: bool
    block_table: Tensor
    pages: int
    block_size: int = 2 * MLA_BLOCK
    slot_window: Tensor | None = None


def spec_mla_kernel_table(block_table: Tensor, pages: int, factor: int, config: VerifyMetadataConfig) -> Tensor:
    """Return the kernel table ``[rows, NB]`` of the host table ``[rows, W]`` of an MLA layer of a verify step (the
    expand, the trim to the step's kernel blocks and the fill of the invalid entries)."""
    return mla_kernel_block_table(
        trim_kernel_table(expand_block_table(block_table, factor), config.kernel_blocks),
        pages * factor,
        local_blocks=config.kernel_blocks,
        max_blocks=config.max_blocks,
        cp_size=config.cp_size,
        row_pages=config.zeroed_mla_pages,
        row_runs=config.row_runs,
    )


def spec_mla_layer_inputs(
    layer: VerifyLayer,
    window_offset: Tensor,
    positions: Tensor,
    tp_rank: Tensor,
    config: VerifyMetadataConfig,
) -> tuple[Tensor, Tensor]:
    """Return the kernel table ``[B, NB]`` and the slots ``[Q, B, 1]`` of one MLA layer of a verify step."""
    queries = config.queries
    factor = mla_page_factor(layer.block_size)
    kernel_pages = layer.pages * factor
    table = spec_mla_kernel_table(layer.block_table, layer.pages, factor, config)
    if layer.slot_window is None:
        raise ValueError("An MLA layer of the verify takes the slot window of the step.")
    rows = window_rows(layer.slot_window, window_offset, queries)
    slots = verify_mla_slots(rows.reshape(-1), positions, tp_rank, kernel_pages, queries, cp_size=config.cp_size)
    return table, slots


def _check_shared_mla(first: VerifyLayer, layer: VerifyLayer) -> None:
    """The kernel reads every MLA layer with the table of the first one: refuse an MLA layer with other inputs."""
    if (
        layer.block_table is not first.block_table
        or layer.slot_window is not first.slot_window
        or layer.pages != first.pages
        or layer.block_size != first.block_size
    ):
        raise ValueError(
            "Every MLA layer of the verify takes the table, slot window, pool size and page rows of the first MLA "
            "layer: give them one MLA table and one slot window."
        )


def spec_verify_metadata(
    layers: Sequence[VerifyLayer],
    positions: Tensor,
    window_offset: Tensor,
    kda_replay_len: Tensor,
    kda_state_position: Tensor,
    tp_rank: Tensor,
    dp_rank: Tensor,
    world_rank: Tensor,
    config: VerifyMetadataConfig,
) -> dict[str, Tensor]:
    """Build the verify kernel inputs of one step from the host step tensors (request-major positions ``[B * Q]``).

    Args:
        layers: the inputs of each layer the verify call runs, in layer order.
        positions: ``[B * Q]`` request-major positions (``spec_math.step_positions``).
        window_offset: ``[B]`` the window offset of the true row 0.
        kda_replay_len, kda_state_position: ``[B]`` from ``spec_math.carry_spec_state``.
        tp_rank: the attention TP rank (= MLA CP rank), a one-element tensor.
        dp_rank: the DP replica, a one-element tensor.
        world_rank: the rank in the world, a one-element tensor.
        config: the static settings.

    Returns:
        ``kda_state_page [n_kda, B, 1]``, ``kda_record_page`` (zeros: the states form reads no records),
        ``kda_replay_len [B, 1]``, ``kda_state_position [B, 1]``, ``mla_block_table [n_mla, B, NB]`` and
        ``mla_slot [n_mla, Q, B, 1]`` (one table and one set of slots, built once, in the row of each MLA layer, which
        the kernel indexes by layer), ``position_rows [Q, B, 1]``, ``rank_info [1, 4]``,
        ``kda_state_slot_page [n_kda, B, 1]`` and ``kda_write_state_slots [n_kda, B, Q]``, int32.
    """
    queries = config.queries
    batch = positions.shape[0] // queries
    kda_ranks = config.kda_ranks
    null = config.kda_null_page
    states, records = [], []
    slot_reads, slot_writes = [], []
    first_mla, mla_inputs, mla_layers = None, None, 0
    for layer in layers:
        if layer.is_mla:
            if first_mla is None:
                first_mla = layer
                # every MLA layer reads the same table and slots: built once a step
                mla_inputs = spec_mla_layer_inputs(layer, window_offset, positions, tp_rank, config)
            else:
                _check_shared_mla(first_mla, layer)
            mla_layers += 1
        else:
            state, read, write = verify_kda_slot_pages(layer.block_table, layer.pages, kda_replay_len, queries, null)
            states.append(mask_pad_rank_pages(state, tp_rank, kda_ranks, null))
            # The kernel reads no records in the states form; the null page 0 fills the argument.
            records.append(torch.zeros_like(state))
            # A pad rank reads no slot (page 0 is off), and it writes its slots to the null page.
            slot_reads.append(mask_pad_rank_pages(read, tp_rank, kda_ranks))
            slot_writes.append(mask_pad_rank_pages(write, tp_rank, kda_ranks, null))
    position_rows = to_token_major(positions.to(torch.int32), queries).reshape(queries, batch, 1)
    metadata = {
        "kda_state_page": torch.stack(states),
        "kda_record_page": torch.stack(records),
        "kda_replay_len": kda_replay_len.to(torch.int32).reshape(batch, 1),
        "kda_state_position": kda_state_position.to(torch.int32).reshape(batch, 1),
        "mla_block_table": torch.stack([mla_inputs[0]] * mla_layers),
        "mla_slot": torch.stack([mla_inputs[1]] * mla_layers),
        "position_rows": position_rows.contiguous(),
        "rank_info": kernel_rank_info(world_rank, dp_rank, tp_rank),
    }
    metadata["kda_state_slot_page"] = torch.stack(slot_reads)
    metadata["kda_write_state_slots"] = torch.stack(slot_writes)
    return metadata
