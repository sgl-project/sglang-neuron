# SPDX-License-Identifier: Apache-2.0
"""Lower SGLang's decode batch to the host step tensors of the Kimi-K3 verify program (host only, numpy).

SGLang side (DCP of size D = 32 over a TP32 replica, ``DcpLayout(D, rank, 128)``): the allocator pages are virtual
pages of ``D x 128`` tokens; position ``p`` of a request has the virtual slot ``s = req_to_token[r, p]``, its owner is
DCP rank ``(s // 128) % D`` and its row on the owner is ``(s // (D x 128)) x 128 + s % 128``: block
``v = s // (D x 128)`` of the rank's ``[blocks, 128, 576]`` uint8 pool (block 0 reserved). Every rank keeps its
128-token stripe of virtual page ``v`` in its own block ``v``, so the block ids are the same on every rank.

Kernel side: the verify program takes the host-page table ``[B, W]`` of the MLA layers (host page ``b`` = kernel pages
``2b, 2b + 1``, 256 rows a rank), the host slots of the window positions ``[B, 2k + 1]`` (row ``256 b + o`` of host
page ``b``) and the KDA state tables ``[B, 1 + Q]`` of the state groups, and expands them in the graph
(``verify_metadata``). The kernel page of a pool view ``[2 P, 128, 576]`` is the SGLang block itself, so:

- host page ``b`` of request block ``i`` (tokens ``[2 i D 128, 2 (i + 1) D 128)``) is ``v_{2i} / 2``, which needs the
  pair rule: ``v_{2i}`` even and ``v_{2i + 1} = v_{2i} + 1`` (the packed MLA loop reads each column pair as one run of
  two kernel pages);
- the host slot of position ``p`` on its owner is ``256 b + 128 h + p % 128`` with ``h = (p // (D 128)) % 2``, which is
  the SGLang row ``128 v + p % 128``;
- a request's unallocated host-page entries take its first host page (the run fill), a pad row's entries are -1, a
  window slot of a position past the allocated pages, of another rank, or in host page 0 is the sink 0.

KDA: each request holds ``1 + Q`` state pages in each of the ``G`` state groups (column 0 the committed state, column
``1 + j`` the state after verify row ``j``); KDA layer ``j`` (in layer order among the KDA layers) reads group
``j % G``. ``kda_state_tables`` places them at ``base + g (1 + Q) + c``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional
from dataclasses import dataclass

import numpy as np
import torch

from sglang_neuron.models.kimi_k3.decode.verify_metadata import VerifyLayer

STRIPE = 128
"""Tokens of one DCP stripe: one kernel page (128 rows) on its owner."""
PAGE_STRIPES = 2
"""Kernel pages of one host page (256 FP8 rows a rank)."""
NULL_BLOCK = 0
"""The null host page (kernel pages 0 and 1): no request holds it, and a sink slot is in it."""
PAD_ENTRY = -1
"""A table entry with no page."""
KDA_STATE_GROUPS = 3
"""The state groups of the 69 KDA layers: KDA layer ``j`` is in group ``j % 3``."""


@dataclass(frozen=True)
class DecodeGeometry:
    """The static page geometry of one DCP rank of the decode.

    Attributes:
        cp_size: D, the DCP (= attention TP) size of the replica.
        cp_rank: this rank's DCP rank.
        queries: Q = k + 1.
        max_model_len: the context limit; the host table has ``ceil(max_model_len / host_page_tokens)`` columns.
        kda_groups: the KDA state groups.
    """

    cp_size: int = 32
    cp_rank: int = 0
    queries: int = 4
    max_model_len: int = 1 << 20
    kda_groups: int = KDA_STATE_GROUPS

    @property
    def virtual_page_tokens(self) -> int:
        """Tokens of one SGLang allocator page (one stripe on each rank)."""
        return self.cp_size * STRIPE

    @property
    def host_page_tokens(self) -> int:
        """Tokens of one host page over the replica (two stripes on each rank)."""
        return PAGE_STRIPES * self.virtual_page_tokens

    @property
    def host_page_rows(self) -> int:
        """Rows of one host page on one rank."""
        return PAGE_STRIPES * STRIPE

    @property
    def table_width(self) -> int:
        """Columns of the host-page table."""
        return -(-self.max_model_len // self.host_page_tokens)


def request_blocks(req_to_token_row: np.ndarray, kv_len: int, geometry: DecodeGeometry, checked: int = 0) -> np.ndarray:
    """Return the SGLang blocks ``[n]`` of a request's first ``kv_len`` positions: ``v_j`` of virtual page ``j``.

    Every page start is checked; the slots inside a page are checked from the page that holds position ``checked``
    on (the positions before it were checked at an earlier step and do not change while the request runs). A step of
    a 1M-token request reads its 256 page starts and the last pages, not its million slots.

    Raises:
        ValueError: a virtual page of the request is not one whole allocator page (its slots are not
            ``v * page + offset``), or it is the reserved page 0.
    """
    page = geometry.virtual_page_tokens
    kv_len = int(kv_len)
    pages = -(-kv_len // page)
    if pages == 0:
        return np.zeros(0, dtype=np.int64)
    starts = np.asarray(req_to_token_row[np.arange(pages) * page], dtype=np.int64)
    blocks = starts // page
    if (starts % page).any() or (blocks <= NULL_BLOCK).any():
        raise ValueError(f"The request's virtual pages start at slots {starts.tolist()[:8]}...: each must start a page "
                         f"of {page} tokens past the reserved page 0.")
    lo = min(max(int(checked), 0), kv_len) // page * page
    row = np.asarray(req_to_token_row[lo:kv_len], dtype=np.int64)
    positions = np.arange(lo, kv_len)
    expected = blocks[positions // page] * page + positions % page
    if not np.array_equal(row, expected):
        bad = int(np.flatnonzero(row != expected)[0])
        raise ValueError(f"Position {lo + bad} of the request has the slot {int(row[bad])}, not {int(expected[bad])}: "
                         "a virtual page holds one allocator page.")
    return blocks


def host_pages(blocks: np.ndarray) -> np.ndarray:
    """Return the host pages ``[ceil(n / 2)]`` of a request's SGLang blocks ``v_j``: ``v_{2i} / 2``.

    Raises:
        ValueError: a block pair is not one host page (``v_{2i}`` odd, or ``v_{2i + 1} != v_{2i} + 1``), or a block is
            in the null host page 0 (blocks 0 and 1).
    """
    blocks = np.asarray(blocks, dtype=np.int64)
    even, odd = blocks[0::2], blocks[1::2]
    bad = (even % 2 != 0) | (even < PAGE_STRIPES)
    bad[: odd.shape[0]] |= odd != even[: odd.shape[0]] + 1
    if bad.any():
        pair = int(np.flatnonzero(bad)[0])
        raise ValueError(f"The request's blocks {blocks[2 * pair : 2 * pair + 2].tolist()} of virtual pages "
                         f"{2 * pair}, {2 * pair + 1} are not one host page (2b, 2b + 1) with b >= 1: allocate the MLA "
                         "pages of a request in aligned pairs.")
    return even // PAGE_STRIPES


@dataclass(frozen=True)
class MlaTables:
    """The host-page table of the MLA layers of one step.

    Attributes:
        raw: ``[B, W]`` int32: host page of each request block, 0 where none (the slot windows read it).
        device: ``[B, W]`` int32 as the program takes it: 0 entries are -1, then each -1 entry of a row with a page
            takes the row's first page (the run fill); a row with no page stays -1.
    """

    raw: np.ndarray
    device: np.ndarray


def fill_rows_with_first_page(stage: np.ndarray) -> None:
    """Point each entry not above the null page at the first page of its row, in place; a row with none keeps its
    entries."""
    mask = stage > NULL_BLOCK
    first = stage[np.arange(stage.shape[0]), mask.argmax(axis=1)]
    has_page = mask.any(axis=1, keepdims=True)
    fill = ~mask & has_page
    np.copyto(stage, first[:, None], where=fill)


def mla_tables(pages_of_rows: Sequence[np.ndarray], batch: int, geometry: DecodeGeometry) -> MlaTables:
    """Return the MLA host-page tables of a step of ``len(pages_of_rows)`` requests padded to ``batch`` rows."""
    width = geometry.table_width
    raw = np.zeros((batch, width), dtype=np.int32)
    for row, pages in enumerate(pages_of_rows):
        if pages.shape[0] > width:
            raise ValueError(f"A request holds {pages.shape[0]} host pages; the table has {width} columns.")
        raw[row, : pages.shape[0]] = pages
    device = raw.copy()
    np.copyto(device, PAD_ENTRY, where=device == NULL_BLOCK)
    fill_rows_with_first_page(device)
    return MlaTables(raw, device)


def mla_slot_window(
    raw_table: np.ndarray, position_window: np.ndarray, num_reqs: int, geometry: DecodeGeometry
) -> np.ndarray:
    """Return the int64 host slots ``[B, 2k + 1]`` of the window positions on this rank.

    A position owned by another rank, past the table, past the request's host pages, or in the null host page gets
    the sink slot 0; a pad row (after ``num_reqs``) is all 0.
    """
    batch, width = position_window.shape
    page_tokens = geometry.host_page_tokens
    rows_a_page = geometry.host_page_rows
    slots = np.full(batch * width, NULL_BLOCK, dtype=np.int64)
    positions = position_window[:num_reqs].reshape(-1).astype(np.int64)
    req_rows = np.repeat(np.arange(num_reqs, dtype=np.int64), width)
    block_index = positions // page_tokens
    outside = block_index >= raw_table.shape[1]
    pages = raw_table[req_rows, np.where(outside, 0, block_index)].astype(np.int64)
    offset = positions % page_tokens
    local = offset // STRIPE % geometry.cp_size == geometry.cp_rank
    row = offset // (geometry.cp_size * STRIPE) * STRIPE + offset % STRIPE
    slot = np.where(outside | ~local, NULL_BLOCK, pages * rows_a_page + row)
    # A slot in the null host page (a position past the request's pages) writes the sink 0.
    slot = np.where((slot > 0) & (slot < rows_a_page), NULL_BLOCK, slot)
    slots[: positions.shape[0]] = slot
    return slots.reshape(batch, width)


def kda_state_tables(
    kda_base: Sequence[int], batch: int, geometry: DecodeGeometry
) -> list[np.ndarray]:
    """Return the KDA state tables ``[B, 1 + Q]`` int32 of each state group as the program takes them.

    Request ``r`` holds page ``kda_base[r] + g (1 + Q) + c`` for column ``c`` of group ``g``; a pad row is -1.
    """
    columns = 1 + geometry.queries
    tables = []
    for group in range(geometry.kda_groups):
        table = np.full((batch, columns), PAD_ENTRY, dtype=np.int32)
        for row, base in enumerate(kda_base):
            pages = int(base) + group * columns + np.arange(columns)
            if (pages <= NULL_BLOCK).any():
                raise ValueError(f"The KDA state pages of row {row} start at {int(base)}: page 0 is the null page.")
            table[row] = pages
        tables.append(table)
    return tables


@dataclass(frozen=True)
class LoweredBatch:
    """The host step inputs of SGLang's decode batch on one rank.

    Attributes:
        row0_positions: ``[B]`` int64: the committed length of each request (the bonus token's position), 0 for a
            pad row.
        mla: the MLA host-page tables.
        kda: the KDA state table of each state group.
        num_reqs: the request rows (the rows after them are pad rows).
    """

    row0_positions: np.ndarray
    mla: MlaTables
    kda: list[np.ndarray]
    num_reqs: int


def lower_decode_batch(
    req_to_token: np.ndarray,
    req_pool_indices: Sequence[int],
    seq_lens: Sequence[int],
    kv_lens: Sequence[int],
    kda_base: Sequence[int],
    batch: int,
    geometry: DecodeGeometry,
    checked: Optional[Sequence[int]] = None,
) -> LoweredBatch:
    """Lower one decode batch of SGLang's DP replica to the host step inputs of the verify program.

    Args:
        req_to_token: ``[rows, max_context]`` SGLang's virtual slot of each request position.
        req_pool_indices: ``[R]`` the req_to_token row of each request.
        seq_lens: ``[R]`` the committed length of each request (verify row 0 is the bonus token at this position).
        kv_lens: ``[R]`` the positions of each request with an allocated slot (from 0).
        kda_base: ``[R]`` the first KDA state page of each request (``kda_state_tables``).
        batch: B, the rows of the step (the requests, then pad rows).
        geometry: the page geometry of this rank.
        checked: ``[R]`` the positions of each request whose slots an earlier step checked (``request_blocks``);
            None checks every slot.
    """
    num_reqs = len(req_pool_indices)
    if not len(seq_lens) == len(kv_lens) == len(kda_base) == num_reqs or num_reqs > batch:
        raise ValueError(f"A step of {batch} rows takes one pool index, length, KV length and KDA base for each of at "
                         f"most {batch} requests.")
    pages = []
    checked = [0] * num_reqs if checked is None else list(checked)
    for index, seq_len, kv_len, done in zip(req_pool_indices, seq_lens, kv_lens, checked):
        if int(kv_len) < int(seq_len) + 1:
            raise ValueError(f"A request of {int(seq_len)} committed tokens has {int(kv_len)} allocated slots; its "
                             "verify rows need the slots from its committed length on.")
        pages.append(host_pages(request_blocks(req_to_token[int(index)], int(kv_len), geometry, int(done))))
    row0 = np.zeros(batch, dtype=np.int64)
    row0[:num_reqs] = np.asarray(seq_lens, dtype=np.int64)
    return LoweredBatch(row0, mla_tables(pages, batch, geometry), kda_state_tables(kda_base, batch, geometry), num_reqs)


def kda_page_groups(is_mla: Sequence[bool], groups: int = KDA_STATE_GROUPS) -> tuple[int, ...]:
    """Return the state group of each KDA layer of a verify call over all layers, in layer order: KDA layer ``j`` is
    in group ``j % G`` (numbered by first use, which is the same order)."""
    kda = sum(1 for mla in is_mla if not mla)
    return tuple(j % groups for j in range(kda))


def verify_layers(
    is_mla: Sequence[bool],
    mla_table: torch.Tensor,
    slot_window: torch.Tensor,
    kda_tables: Sequence[torch.Tensor],
    pool_pages: int,
    geometry: DecodeGeometry,
) -> list[VerifyLayer]:
    """Return the per-layer verify inputs in layer order: every MLA layer the MLA table and slot window, KDA layer
    ``j`` (among the KDA layers) the state table of group ``j % G``; ``pool_pages`` host pages in each layer's pool."""
    layers = []
    kda_index = 0
    for mla in is_mla:
        if mla:
            layers.append(VerifyLayer(True, mla_table, pool_pages, geometry.host_page_rows, slot_window))
        else:
            table = kda_tables[kda_index % geometry.kda_groups]
            layers.append(VerifyLayer(False, table, pool_pages))
            kda_index += 1
    return layers
