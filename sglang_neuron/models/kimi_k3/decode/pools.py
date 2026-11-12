# SPDX-License-Identifier: Apache-2.0
"""Device pools of one rank of the Kimi-K3 decode, and the host placement of a request's pages in them.

The decode programs read the pools in the layout of the decode kernels:

- ``kv``: 24 shared buffers ``[POOL_PAGES, PAGE_WIDTH]`` bf16. A page (147,456 B) holds 256 FP8 MLA rows of 576 B
  (two kernel pages of 128 rows: kernel page ``2b`` and ``2b + 1`` of page ``b``) or one KDA layer's state. Each of
  the 93 layers reads one buffer, ``pool_of_layer[l]`` (``hybrid_pool_of_layer``): buffer ``i`` holds MLA layer ``i``
  of the 24 and KDA layer ``i`` of each of the 3 KDA groups (KDA layer ``j`` in layer order is in group ``j % 3``).
  One page id space covers the rank: a page belongs to the MLA rows or to one KDA group, never both, and page 0 is
  the null page.
- ``rider``: the drafter rider pool ``[DRAFT_LAYERS * rider_pages * 2, 128 * RIDER_ROW_WIDTH]`` bf16, the 5 drafter
  layers stacked (layer ``d`` holds kernel rows ``[d * 2P, (d + 1) * 2P)``), FP8 rows as bf16 pairs. A request holds
  one ring of ``RIDER_RING_PAGES`` pages (page 0 is the null page).

A request of the step holds (``RequestPages``): its MLA pages (host pages ``b``, each the SGLang allocator page pair
``2b, 2b + 1``), ``1 + Q`` state pages in each KDA group (column 0 the committed state the prefill writes, column
``1 + j`` the state after verify row ``j``), and its rider ring. ``PageAllocator`` hands them out on the host; the
step tables (``mla_block_table``, ``kda_state_table_<g>``, ``spec_draft_rider_ring``) come from ``RequestPages``.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import numpy as np
import torch
from torch import Tensor

KV_POOLS = 24
"""The shared KV buffers of a rank (the verify entry takes at most 24 pools)."""
POOL_PAGES = 1700
"""Pages of each KV buffer; a traced shape of the verify program."""
PAGE_BYTES = 147_456
"""Bytes of one page: 256 FP8 rows of 576 B, or one KDA layer's state."""
PAGE_WIDTH = PAGE_BYTES // 2
"""bf16 elements of one page."""
ROW_BYTES = 576
"""Bytes of one FP8 MLA row: 512 normalized latent + 64 raw rope."""
KERNEL_PAGE_ROWS = 128
"""Rows of one kernel page (one CP stripe of 128 tokens on its owner rank)."""
PAGE_ROWS = 2 * KERNEL_PAGE_ROWS
"""Rows of one page on one rank."""
NULL_PAGE = 0
KDA_GROUPS = 3
"""The KDA state groups: KDA layer ``j`` (in layer order among the KDA layers) is in group ``j % 3``."""
KDA_CONV_BYTES = 6912
"""Bytes of the conv state of one KDA layer in its page (bf16 ``[1152, 3]``, DS layout); the recurrent state follows."""
KDA_STATE_BYTES = 105_216
"""Bytes of one KDA layer's state in its page (conv + recurrent bf16 ``[3, 128, 128]``)."""
DRAFT_LAYERS = 5
RIDER_ROW_WIDTH = ROW_BYTES // 2
"""bf16 elements of one FP8 rider row (576 FP8 bytes as bf16 pairs)."""
RIDER_RING_PAGES = 7
"""Pages of one rider ring: 14 kernel columns of 4096 positions >= the 49,152-token drafter window + 2 columns."""
DRAFT_WINDOW = 49_152
CP = 32
"""The MLA CP (= attention TP) of a replica."""
COLUMN_TOKENS = CP * KERNEL_PAGE_ROWS
"""Positions of one kernel column (one SGLang allocator page): 32 stripes of 128."""
PAGE_TOKENS = 2 * COLUMN_TOKENS
"""Positions of one page over the replica (8192)."""


def hybrid_pool_of_layer(is_mla: Sequence[bool], kda_groups: int = KDA_GROUPS) -> tuple[int, ...]:
    """Return the shared buffer of each layer, numbered by first use in layer order.

    The hybrid grouping of the served decode: the layers of each kind are split into groups of the size of the
    smaller kind (``layers[i::groups]``: 24 MLA layers in one group, 69 KDA layers in 3 groups of 23), and buffer ``i``
    holds layer ``i`` of each group. Numbered by first use, a block of 3 KDA layers and the MLA layer after them read
    one buffer.

    Example:
        >>> from sglang_neuron.models.kimi_k3.decode.pools import hybrid_pool_of_layer
        >>> hybrid_pool_of_layer([False, False, False, True] * 2)
        (0, 0, 0, 0, 1, 1, 1, 1)
    """
    mla = [i for i, m in enumerate(is_mla) if m]
    kda = [i for i, m in enumerate(is_mla) if not m]
    size = min(len(mla), len(kda)) if mla and kda else max(len(mla), len(kda))
    if max(len(mla), len(kda)) < size * 1.5:
        size = max(len(mla), len(kda))
    tensor_of: dict[int, int] = {}
    for layers in (kda, mla):
        groups = -(-len(layers) // size) if layers else 0
        for g in range(groups):
            for index, layer in enumerate(layers[g::groups]):
                tensor_of[layer] = index
    order: dict[int, int] = {}
    pools = []
    for layer in range(len(is_mla)):
        tensor = tensor_of[layer]
        if tensor not in order:
            order[tensor] = len(order)
        pools.append(order[tensor])
    if kda_groups and kda and -(-len(kda) // size) != kda_groups:
        raise ValueError(f"The KDA layers form {-(-len(kda) // size)} state groups, not {kda_groups}.")
    return tuple(pools)


def kda_group_of_layer(is_mla: Sequence[bool], kda_groups: int = KDA_GROUPS) -> dict[int, int]:
    """Return the state group of each KDA layer (by layer index): KDA layer ``j`` in layer order is in group ``j % G``."""
    kda = [i for i, m in enumerate(is_mla) if not m]
    return {layer: j % kda_groups for j, layer in enumerate(kda)}


def rider_slots(queries: int, pages: int = POOL_PAGES, mla_pages: int = 128, kda_groups: int = KDA_GROUPS) -> int:
    """Return the rider rings of a rank: the requests the pages hold at the 1M context.

    One request needs ``mla_pages`` pages (1,048,576 / 8192) and ``1 + Q`` pages in each KDA group, so the pool holds
    ``int(pages / (mla_pages + G (1 + Q)))`` of them (the decode's KV-holder cap; 11 at k = 3, 10 at k = 7).

    Example:
        >>> from sglang_neuron.models.kimi_k3.decode.pools import rider_slots
        >>> rider_slots(4), rider_slots(8)
        (11, 10)
    """
    return int(pages / (mla_pages + kda_groups * (1 + queries)))


def rider_pages(queries: int, pages: int = POOL_PAGES) -> int:
    """Return the pages of the rider pool: the null page and one ring for each ring slot (78 at k = 3, 71 at k = 7)."""
    return 1 + rider_slots(queries, pages) * RIDER_RING_PAGES


# --- the pools of one rank


@dataclasses.dataclass
class DecodePools:
    """One rank's device pools, bound in place by the programs (the verify writes the MLA rows and KDA states, the
    drafter its context rows).

    Attributes:
        kv: the ``KV_POOLS`` shared ``[pages, PAGE_WIDTH]`` bf16 buffers.
        pool_of_layer: the buffer of each of the 93 layers (static).
        rider: the drafter rider pool ``[DRAFT_LAYERS * 2 * rider_pages, 128 * RIDER_ROW_WIDTH]`` bf16.
    """

    kv: tuple[Tensor, ...]
    pool_of_layer: tuple[int, ...]
    rider: Tensor

    @classmethod
    def allocate(cls, is_mla: Sequence[bool], queries: int, *, pages: int = POOL_PAGES, device=None,
                 kv_pools: int = KV_POOLS, rider: int | None = None) -> DecodePools:
        """Allocate zeroed pools on ``device`` for a decode of ``queries`` rows a request (``rider`` rider pages;
        default the rings the ``pages`` hold, ``rider_pages``)."""
        pool_of_layer = hybrid_pool_of_layer(is_mla)
        count = max(pool_of_layer) + 1
        if count != kv_pools:
            raise ValueError(f"The layers read {count} shared buffers; the decode binds {kv_pools}.")
        kv = tuple(torch.zeros((pages, PAGE_WIDTH), dtype=torch.bfloat16, device=device) for _ in range(count))
        rider_count = rider_pages(queries, pages) if rider is None else int(rider)
        rider = torch.zeros((DRAFT_LAYERS * 2 * rider_count, KERNEL_PAGE_ROWS * RIDER_ROW_WIDTH),
                            dtype=torch.bfloat16, device=device)
        return cls(kv, pool_of_layer, rider)

    @property
    def pages(self) -> int:
        return int(self.kv[0].shape[0])

    @property
    def rider_pages(self) -> int:
        return int(self.rider.shape[0]) // (2 * DRAFT_LAYERS)

    def nbytes(self) -> dict[str, int]:
        return {"kv": sum(t.numel() * t.element_size() for t in self.kv),
                "rider": self.rider.numel() * self.rider.element_size()}

    # host views of one layer's rows (the golden load and tests; never inside a program)

    def mla_rows(self, layer: int) -> Tensor:
        """The FP8 rows ``[pages * 256, 576]`` uint8 of an MLA layer's buffer: row ``256 b + 128 h + r`` is row ``r`` of
        kernel page ``2b + h``."""
        return self.kv[self.pool_of_layer[layer]].view(torch.uint8).view(-1, ROW_BYTES)

    def kda_page(self, layer: int, page: int) -> Tensor:
        """The ``PAGE_BYTES`` uint8 bytes of page ``page`` of a KDA layer's buffer."""
        return self.kv[self.pool_of_layer[layer]][page].view(torch.uint8)

    def rider_rows(self, draft_layer: int) -> Tensor:
        """The rider rows ``[rider_pages * 256, 576]`` uint8 of drafter layer ``draft_layer``."""
        rows = self.rider_pages * 2
        part = self.rider[draft_layer * rows:(draft_layer + 1) * rows]
        return part.view(torch.uint8).view(-1, ROW_BYTES)


# --- host placement of a request's pages


@dataclasses.dataclass(frozen=True)
class RequestPages:
    """The pages of one request of a step.

    Attributes:
        mla: the host pages ``b`` of the request's MLA rows in order (page ``i`` holds positions
            ``[8192 i, 8192 (i + 1))``; SGLang allocator pages ``2b, 2b + 1``).
        kda: ``[G, 1 + Q]`` the state pages of each KDA group (column 0 the committed state).
        ring: the request's ``RIDER_RING_PAGES`` rider pages, or () for none.
    """

    mla: tuple[int, ...]
    kda: tuple[tuple[int, ...], ...]
    ring: tuple[int, ...] = ()

    def sglang_blocks(self) -> list[int]:
        """The SGLang allocator pages (4096-token virtual pages) of the MLA pages: ``2b, 2b + 1`` each."""
        return [v for b in self.mla for v in (2 * b, 2 * b + 1)]

    def req_to_token(self, length: int) -> np.ndarray:
        """SGLang's virtual slot of positions ``0 .. length - 1``: ``v * 4096 + p % 4096`` with ``v`` the allocator
        page of position ``p``."""
        blocks = np.asarray(self.sglang_blocks(), dtype=np.int64)
        positions = np.arange(int(length), dtype=np.int64)
        if positions.shape[0] > blocks.shape[0] * COLUMN_TOKENS:
            raise ValueError(f"{length} positions need {-(-int(length) // PAGE_TOKENS)} MLA pages; the request holds "
                             f"{len(self.mla)}.")
        return blocks[positions // COLUMN_TOKENS] * COLUMN_TOKENS + positions % COLUMN_TOKENS


class PageAllocator:
    """Hand out the pages of one rank's pools on the host: MLA and KDA pages from one id space (1 .. pages - 1, page
    0 the null page), rider rings from the rider pool's ring slots. Every rank of a replica allocates the same ids.

    Example:
        >>> from sglang_neuron.models.kimi_k3.decode.pools import PageAllocator
        >>> alloc = PageAllocator(pages=40, queries=4, rider_pages=15)
        >>> req = alloc.allocate("a", tokens=20000)
        >>> req.mla, req.kda[0], req.ring
        ((1, 2, 3), (4, 5, 6, 7, 8), (1, 2, 3, 4, 5, 6, 7))
        >>> alloc.free("a"); alloc.allocate("b", tokens=8192).mla
        (1,)
    """

    def __init__(self, pages: int = POOL_PAGES, queries: int = 4, rider_pages: int | None = None,
                 kda_groups: int = KDA_GROUPS) -> None:
        self.pages = int(pages)
        self.queries = int(queries)
        self.kda_groups = int(kda_groups)
        self.rider_pages = rider_pages if rider_pages is not None else 1 + rider_slots(queries, pages) * \
            RIDER_RING_PAGES
        self._free = list(range(1, self.pages))
        self._rings = list(range((self.rider_pages - 1) // RIDER_RING_PAGES))
        self._held: dict[str, RequestPages] = {}

    @staticmethod
    def mla_pages_for(tokens: int) -> int:
        """The MLA pages of a request of ``tokens`` positions (rows of positions ``0 .. tokens - 1``)."""
        return -(-int(tokens) // PAGE_TOKENS)

    def allocate(self, req_id: str, tokens: int, ring: bool = True) -> RequestPages:
        """Place a request of ``tokens`` positions (its prompt, its output and the verify rows of its last step).

        Raises:
            ValueError: the request already holds pages, or the pool or the rings are full.
        """
        if req_id in self._held:
            raise ValueError(f"Request {req_id} already holds pages.")
        mla = self.mla_pages_for(tokens)
        kda = self.kda_groups * (1 + self.queries)
        if mla + kda > len(self._free):
            raise ValueError(f"Request {req_id} needs {mla} MLA pages and {kda} KDA pages; {len(self._free)} are free.")
        if ring and not self._rings:
            raise ValueError(f"No rider ring is free for request {req_id}.")
        taken = [self._free.pop(0) for _ in range(mla + kda)]
        columns = 1 + self.queries
        groups = tuple(tuple(taken[mla + g * columns: mla + (g + 1) * columns]) for g in range(self.kda_groups))
        rings = ()
        if ring:
            slot = self._rings.pop(0)
            rings = tuple(range(1 + slot * RIDER_RING_PAGES, 1 + (slot + 1) * RIDER_RING_PAGES))
        pages = RequestPages(tuple(taken[:mla]), groups, rings)
        self._held[req_id] = pages
        return pages

    def grow(self, req_id: str, tokens: int) -> RequestPages:
        """Give a request the MLA pages of ``tokens`` positions (one more page as it crosses a page edge)."""
        pages = self._held[req_id]
        need = self.mla_pages_for(tokens) - len(pages.mla)
        if need <= 0:
            return pages
        if need > len(self._free):
            raise ValueError(f"Request {req_id} needs {need} more MLA pages; {len(self._free)} are free.")
        grown = dataclasses.replace(pages, mla=pages.mla + tuple(self._free.pop(0) for _ in range(need)))
        self._held[req_id] = grown
        return grown

    def free(self, req_id: str) -> None:
        pages = self._held.pop(req_id)
        self._free = sorted(self._free + list(pages.mla) + [p for group in pages.kda for p in group])
        if pages.ring:
            self._rings = sorted(self._rings + [(pages.ring[0] - 1) // RIDER_RING_PAGES])

    def held(self, req_id: str) -> RequestPages:
        return self._held[req_id]


def ring_kernel_pages(ring: Sequence[int]) -> np.ndarray:
    """The kernel pages ``[2 R]`` of a rider ring of ``R`` pages (``2b, 2b + 1`` each): ring entry ``c`` holds kernel
    column ``c``."""
    return np.asarray([k for b in ring for k in (2 * int(b), 2 * int(b) + 1)], dtype=np.int32)


def rider_ring_input(rings: Sequence[Sequence[int]], batch: int) -> Tensor:
    """``spec_draft_rider_ring`` ``[B, 2 RIDER_RING_PAGES]`` int32: the kernel pages of each request's ring, 0 for a
    pad row or a request without a ring."""
    out = np.zeros((batch, 2 * RIDER_RING_PAGES), dtype=np.int32)
    for row, ring in enumerate(rings):
        if ring:
            out[row] = ring_kernel_pages(ring)
    return torch.from_numpy(out)


def mla_host_table(requests: Sequence[RequestPages], batch: int, width: int = 128) -> np.ndarray:
    """The raw MLA host-page table ``[B, width]`` int32 (0 where none); ``lowering.mla_tables`` gives the device form
    from SGLang's ``req_to_token``, and this is the same table built from the pages directly."""
    table = np.zeros((batch, width), dtype=np.int32)
    for row, pages in enumerate(requests):
        if len(pages.mla) > width:
            raise ValueError(f"A request holds {len(pages.mla)} MLA pages; the table has {width} columns.")
        table[row, : len(pages.mla)] = pages.mla
    return table


def kda_state_tables(requests: Sequence[RequestPages], batch: int, queries: int,
                     kda_groups: int = KDA_GROUPS) -> list[np.ndarray]:
    """The KDA state table ``[B, 1 + Q]`` int32 of each group; a pad row is -1 (``lowering.kda_state_tables`` form)."""
    tables = []
    for group in range(kda_groups):
        table = np.full((batch, 1 + queries), -1, dtype=np.int32)
        for row, pages in enumerate(requests):
            table[row] = pages.kda[group]
        tables.append(table)
    return tables
