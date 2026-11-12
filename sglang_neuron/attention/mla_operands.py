# SPDX-License-Identifier: Apache-2.0
"""The operand contract between the MLA / DSA pools and the kernels that read and write them (kernels/<model>/).

The pools (mem_cache/mla_pool.py) hold, per rank, latent ``[layers, blocks, block, kv_lora_rank + rope]`` and index keys
``[layers, blocks, index_block, 128]``; a native page id is the local block id on every CP rank (mem_cache/cp_pages.py).
A kernel takes, besides the pool tensors themselves (aliased in and out of the graph):

Decode, per CP rank, one row per request slot of the compiled batch (``rows``):

=================== ============== ===================================================================================
operand             shape, dtype   meaning
=================== ============== ===================================================================================
``mla_block_table`` [rows, W] i32  the rank's local blocks of the row's sequence (its first native pages); padding
                                   ``table_pad`` (0: reads the all-zero block 0; -1 for a kernel that skips it)
``mla_write_slot``  [rows] i32     the rank's slot of the token being decoded, ``DROP_SLOT`` (-1) where another rank
                                   owns it or the row is inactive; never a slot of block 0
``mla_positions``   [1, rows] i32  the rank's local token count before this token (the causal bound of its shard)
``dsa_positions``   [rows] i32     the global position of the token being decoded (index-K RoPE, selection bound)
``dsa_rank_id``     [1] i32        the CP rank, as a tensor so every rank runs one NEFF
``row_valid``       [rows] i32     1 for a live row
=================== ============== ===================================================================================

Prefill (one request chunk, ``chunk`` tokens compiled), per CP rank: ``cp_write_slots`` [chunk] i32 (the rank's slots
of the chunk's tokens it owns, ``DROP_SLOT`` elsewhere and past ``n_valid``), ``bt_loc`` [1, W] i32 (the rank's local
blocks of the whole sequence, prefix included), ``cp_q_start`` [1] i32 (the chunk's first position), ``n_loc`` [1] i32
(the rank's local tokens of the sequence after the chunk, rounded to whole blocks: the KV extent its attention reads),
``n_valid`` [1] i32.

The index keys share the block table (``index_sharding`` "owner") or are addressed by native page with the native
page's token offset (``index_sharding`` "replicated": every rank writes every token; the index write slot is the native
slot itself).

Sparse selection under CP (``index_sharding`` "owner"): each rank scores its own tokens and keeps candidates before
the group merges them. A rank can hold all ``index_topk`` global winners, so the per-rank candidate capacity must be
at least ``min(index_topk, local tokens)``, and ties at the threshold must all be counted, or the merge silently drops
winners (``check_cp_selection_capacity``).

Recurrent-state controls (KDA working rows, checkpoint slots) are the state framework's operands, not these.
"""

from __future__ import annotations

from dataclasses import dataclass
from operator import index
from typing import Sequence

import numpy as np

from sglang_neuron.mem_cache.cp_pages import DROP_SLOT, CpPages, _host_ints

_INT32_MAX = int(np.iinfo(np.int32).max)


def decode_operands(cp: CpPages, *, prior_lens: Sequence[int], native_pages: Sequence[Sequence[int]],
                    native_slots: Sequence[int], active: Sequence[int], rows: int, block_width: int,
                    table_pad: int = 0) -> tuple:
    """Each CP rank's decode operands (a tuple of dicts, rank order) for ``len(prior_lens)`` live requests in rows
    ``0..n-1`` of a ``rows``-row compiled batch. Request ``i`` has ``prior_lens[i]`` cached tokens, decodes one token at
    native slot ``native_slots[i]`` (allocated by SGLang; its page is the last of ``native_pages[i]``, which covers
    ``prior_lens[i] + 1`` tokens), and is live where ``active[i]``. Raises where a row's pages do not cover it, a table
    is too narrow, or two live rows write one slot."""
    if table_pad not in (0, -1):
        raise ValueError("block-table padding is block 0 (all zeros) or -1 (skipped)")
    prior_lens = _host_ints(prior_lens, "prior lengths")
    native_slots = _host_ints(native_slots, "native slots")
    active = _host_ints(active, "active flags")
    n = len(prior_lens)
    if not (n == len(native_pages) == len(native_slots) == len(active)) or n > rows:
        raise ValueError(f"{n} requests with mismatched controls for a {rows}-row batch")
    if any(a not in (0, 1) for a in active) or any(length < 0 for length in prior_lens):
        raise ValueError("decode lengths must be non-negative and active flags 0 or 1")
    if any(length + 1 > _INT32_MAX for length in prior_lens):
        raise ValueError("a position exceeds the kernels' int32 type")
    for i, (length, slot, live) in enumerate(zip(prior_lens, native_slots, active)):
        if live and slot % cp.page != length % cp.page:
            raise ValueError(f"row {i}: native slot {slot} disagrees with position {length}")
    out = []
    for rank in range(cp.cp_size):
        table = np.full((rows, block_width), table_pad, dtype=np.int32)
        local_pos = np.zeros((1, rows), dtype=np.int32)
        global_pos = np.zeros(rows, dtype=np.int32)
        valid = np.zeros(rows, dtype=np.int32)
        for i, (length, pages, live) in enumerate(zip(prior_lens, native_pages, active)):
            if not live:
                continue
            local = cp.local_pages(pages, length + 1, rank)
            if len(local) > block_width:
                raise ValueError(f"row {i}: {len(local)} local blocks exceed the compiled table width {block_width}")
            table[i, : len(local)] = local
            local_pos[0, i] = int(cp.local_count(length, rank))
            global_pos[i], valid[i] = length, 1
        write = np.full(rows, DROP_SLOT, dtype=np.int32)
        slots = np.where(np.asarray(active, dtype=bool), np.asarray(native_slots, dtype=np.int64), -1)
        write[:n] = cp.write_slots(slots, rank)
        mine = write[write >= 0]
        if len(np.unique(mine)) != len(mine):
            raise ValueError("live decode rows share a write slot")
        out.append(dict(mla_block_table=table, mla_write_slot=write, mla_positions=local_pos,
                        dsa_positions=global_pos, dsa_rank_id=np.array([rank], dtype=np.int32), row_valid=valid))
    return tuple(out)


def prefill_operands(cp: CpPages, *, rank: int, prefix_len: int, n_valid: int, native_slots: Sequence[int],
                     native_pages: Sequence[int], chunk: int, block_width: int, table_pad: int = 0) -> dict:
    """CP rank ``rank``'s operands for one request chunk of ``n_valid`` tokens at positions
    ``[prefix_len, prefix_len + n_valid)`` in a ``chunk``-token compiled prefill, the chunk's tokens at native slots
    ``native_slots`` and the sequence (prefix included) over ``native_pages``."""
    prefix_len, n_valid = index(prefix_len), index(n_valid)
    if not 0 <= n_valid <= chunk or prefix_len < 0 or prefix_len + n_valid > _INT32_MAX:
        raise ValueError(f"a chunk of {n_valid} tokens at {prefix_len} does not fit the compiled {chunk}-token chunk")
    slots = np.asarray(_host_ints(native_slots, "native slots"), dtype=np.int64)
    if slots.size != n_valid:
        raise ValueError(f"{slots.size} slots for {n_valid} tokens")
    positions = prefix_len + np.arange(n_valid, dtype=np.int64)
    if np.any(slots % cp.page != positions % cp.page):
        raise ValueError("native slot offsets disagree with the chunk's positions")
    local = cp.local_pages(native_pages, prefix_len + n_valid, rank)
    if len(local) > block_width:
        raise ValueError(f"{len(local)} local blocks exceed the compiled table width {block_width}")
    writes = np.full(chunk, DROP_SLOT, dtype=np.int32)
    writes[:n_valid] = cp.write_slots(slots, rank)
    table = np.full((1, block_width), table_pad, dtype=np.int32)
    table[0, : len(local)] = local
    return {"n_valid": np.array([n_valid], dtype=np.int32), "cp_write_slots": writes, "bt_loc": table,
            "cp_q_start": np.array([prefix_len], dtype=np.int32),
            "n_loc": np.array([len(local) * cp.block], dtype=np.int32)}


def replicated_index_slots(cp: CpPages, native_slots) -> np.ndarray:
    """Index-K write slots of a replicated index pool: the native slot on every rank (an index block is a whole
    native page), ``DROP_SLOT`` for padding; a slot of the reserved page 0 is refused."""
    slots = np.asarray(_host_ints(native_slots, "native slots"), dtype=np.int64)
    if np.any((slots >= 0) & (slots < cp.page)):
        raise ValueError("a native slot addresses the reserved page 0, which the zero-page contract keeps zero")
    return np.where(slots >= 0, slots, DROP_SLOT).astype(np.int32)


@dataclass(frozen=True)
class SelectionBudget:
    """The CP sparse selection a kernel compiles: ``index_topk`` global winners merged from per-rank candidate lists
    of ``local_capacity`` entries."""

    index_topk: int
    local_capacity: int


def check_cp_selection_capacity(budget: SelectionBudget, cp: CpPages, max_context: int) -> None:
    """Refuse a CP selection whose per-rank candidate list can drop global winners: one rank can hold every winner,
    so a rank must keep ``min(index_topk, its tokens at max_context)`` candidates (plus every tie at the threshold,
    which the kernel counts in full)."""
    if cp.cp_size == 1:
        return
    local_max = int(cp.local_count(max_context, 0))
    need = min(budget.index_topk, local_max)
    if budget.local_capacity < need:
        raise ValueError(f"CP{cp.cp_size} sparse selection keeps {budget.local_capacity} candidates per rank, but one "
                         f"rank can hold {need} of the {budget.index_topk} winners at a {max_context}-token context")
