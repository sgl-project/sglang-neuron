# SPDX-License-Identifier: Apache-2.0
"""The Kimi-K3 decode role's pools on SGLang: one page space of host pages shared by the MLA rows and the KDA state,
the drafter rider rings, and what a PD prefill writes into them.

One decode rank (TP rank ``t`` of a TP32 = DCP32 replica) holds 24 shared buffers of ``POOL_HOST_PAGES`` host pages
each (``[1700, 73728]`` bf16 = ``[3400, 128, 576]`` uint8; the page count is a traced shape of the verify program).
Buffer ``i`` holds MLA layer ``i`` (in MLA layer order) and the ``i``-th KDA layer of each of the 3 KDA state groups
(KDA layer ``j`` in KDA layer order is in group ``j % 3``, at index ``j // 3``; ``pools.hybrid_pool_of_layer``, the
map the programs bind, ``pools.DecodePools``). These buffers are
SGLang's own MLA pool buffers: ``NeuronMLATokenToKVPool`` keeps one ``[3400 x 128, 1, 576]`` uint8 buffer per MLA
layer, so with ``--max-total-tokens = DecodePageLayout.max_total_tokens`` SGLang builds exactly the 24 buffers, and
its PD registration of the MLA entries is unchanged.

Host page ``b`` is SGLang allocator pages ``2b`` and ``2b + 1`` (4096 tokens each under DCP32: one 128-token stripe on
every rank, kernel page ``v`` of a rank at byte ``v * 73,728``). The page space is split statically:

- host page 0 (allocator pages 0, 1): the null page; no request holds it;
- host pages ``[1, kda_start)``: MLA pages. The packed verify reads each column pair of its table as one run of two
  kernel pages, so the allocator hands a request its pages in aligned pairs ``(2b, 2b + 1)``, ``b >= 1``
  (``PagePairBook``); SGLang's ``req_to_token`` keeps its meaning;
- host pages ``[kda_start, POOL_HOST_PAGES)``: KDA state. State slot ``s`` (SGLang's mamba slot, 1-based) holds
  ``3 (1 + Q)`` consecutive host pages from ``kda_base(s)``: group ``g`` column ``c`` at ``kda_base(s) + g (1 + Q) +
  c``; column 0 the committed state (the one a PD prefill writes), column ``1 + j`` the state after verify row ``j``.
  In buffer ``i`` the page holds the state of the KDA layer of group ``g`` in that buffer: bytes ``[0, 6912)`` the
  conv state bf16 channel-major ``[1152, 3]`` as three parts q, k, v of ``[384, 3]`` (2,304 B each), then
  ``[6912, 105216)`` the recurrent state bf16 ``[3, 128, 128]``.

The KDA state sits at the top of the page space because a PD prefill writes a request's committed state at a page it
computes from the state slot alone (``registration_frame``), and the lowering places a request's ``3 (1 + Q)`` state
pages at one base (``lowering.kda_state_tables``); the programs read any page id of the table, so the split changes
no traced shape.

The rider pool (``RiderPool``) holds the 5 drafter layers' context rows in the MLA page format, the null page and one
ring of ``RING_PAGES`` host pages per ring slot (``pools.rider_pages``: the most requests the pools hold at the full
context), and state slot ``s`` uses ring ``s``: ring page ``j`` of slot ``s`` is rider host page ``1 + (s - 1) * 7 + j``. A
drafter row of position ``p`` lives on rank ``(p // 128) % 32`` in ring page ``(p // 8192) % 7``, kernel half
``(p // 4096) % 2``. Every per-request page of the decode is a function of the request's state slot, which a PD
prefill receives as the state index: the decode announces the rule in its registration (``registration_frame``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from sglang_neuron.models.kimi_k3.decode import pools as pl

# The geometry of the decode programs' pools (pools.py), in the names of SGLang's side.
STRIPE = pl.KERNEL_PAGE_ROWS
"""Tokens of one DCP stripe: one kernel page (128 rows) on its owner rank."""
MLA_ROW_BYTES = pl.ROW_BYTES
"""One FP8 MLA row: 512 latent + 64 raw rope."""
KERNEL_PAGE_BYTES = STRIPE * MLA_ROW_BYTES
"""One kernel page on one rank: 73,728 B."""
HOST_PAGE_BYTES = pl.PAGE_BYTES
"""One host page (two kernel pages, one bf16 row of a ``[P, 73728]`` buffer): 147,456 B."""
POOL_HOST_PAGES = pl.POOL_PAGES
"""Host pages of each shared buffer: a traced shape of the verify program."""
DCP_SIZE = pl.CP
"""DCP (= attention TP) size of a decode replica."""
KDA_GROUPS = pl.KDA_GROUPS
"""KDA state groups: KDA layer ``j`` (in KDA layer order) is in group ``j % 3``."""
DRAFT_LAYERS = pl.DRAFT_LAYERS
"""Drafter layers whose context rows a PD prefill sends after the target entries."""
RING_PAGES = pl.RIDER_RING_PAGES
"""Rider ring host pages of a request: 57,344 tokens, at least the drafter window."""
DRAFT_WINDOW = pl.DRAFT_WINDOW
"""Drafter context window in tokens."""
CONV_PART_BYTES = 384 * 3 * 2
"""One conv part (q, k or v) of a KDA layer on one decode rank: 3 heads x 128 channels x 3 taps, bf16."""
CONV_PARTS = 3
RECURRENT_OFFSET = pl.KDA_CONV_BYTES
"""Byte offset of the recurrent state in a KDA state page: the three conv parts, 6,912."""
RECURRENT_BYTES = 3 * 128 * 128 * 2
"""The recurrent state of a KDA layer on one decode rank: 3 heads x [128, 128] bf16."""
KDA_STATE_BYTES = pl.KDA_STATE_BYTES
"""Bytes of one KDA layer's state in its page: 105,216."""
STATE_PARTS = (("conv_q", 0, CONV_PART_BYTES), ("conv_k", CONV_PART_BYTES, CONV_PART_BYTES),
               ("conv_v", 2 * CONV_PART_BYTES, CONV_PART_BYTES), ("recurrent", RECURRENT_OFFSET, RECURRENT_BYTES))
"""The parts of a KDA state page a prefill writes: (name, byte offset, bytes)."""


@dataclass(frozen=True)
class DecodePageLayout:
    """The page space of one decode rank.

    Attributes:
        queries: Q = k + 1, the verify rows of a request.
        state_slots: S, SGLang's mamba slots (1 .. S); each holds its KDA pages and its rider ring.
        host_pages: host pages of each shared buffer.
        kda_groups: KDA state groups.
        ring_pages: rider ring host pages of a slot.
        cp_size: DCP size of the replica.
    """

    queries: int
    state_slots: int
    host_pages: int = POOL_HOST_PAGES
    kda_groups: int = KDA_GROUPS
    ring_pages: int = RING_PAGES
    cp_size: int = DCP_SIZE

    def __post_init__(self):
        if self.queries < 2 or self.state_slots < 1:
            raise ValueError(f"a decode page layout takes Q >= 2 verify rows and one state slot or more, not Q = "
                             f"{self.queries} and {self.state_slots} slots")
        if self.state_slots > self.ring_slots:
            raise ValueError(f"{self.state_slots} state slots: the decode programs' rider pool holds {self.ring_slots} "
                             f"rings at Q = {self.queries}, one per request a rank's pools hold at the full context "
                             f"(pools.rider_slots); launch with --max-running-requests and "
                             f"--disaggregation-decode-extra-slots summing to {self.ring_slots} requests a DP replica")

    @property
    def ring_slots(self) -> int:
        """The rider rings of the decode programs' rider pool (pools.rider_slots): the most state slots a rank has."""
        return pl.rider_slots(self.queries, self.host_pages, kda_groups=self.kda_groups)

    @property
    def kda_pages_per_slot(self) -> int:
        """Host pages of one slot's KDA state: ``groups x (1 + Q)``."""
        return self.kda_groups * (1 + self.queries)

    @property
    def kda_start(self) -> int:
        """The first KDA host page; the MLA host pages are ``[1, kda_start)``."""
        return self.host_pages - self.state_slots * self.kda_pages_per_slot

    @property
    def mla_allocator_pages(self) -> range:
        """SGLang allocator pages of the MLA rows: ``[2, 2 kda_start)``."""
        return range(2, 2 * self.kda_start)

    @property
    def allocator_pages(self) -> int:
        """SGLang allocator pages of the whole space past page 0 (pages ``1 .. 2 P - 1``)."""
        return 2 * self.host_pages - 1

    @property
    def max_total_tokens(self) -> int:
        """``--max-total-tokens`` that makes SGLang's MLA pool exactly the shared buffers: a rank's pool tokens (one
        128-row block per allocator page past page 0; SGLang widens the allocator over the DCP group)."""
        return self.allocator_pages * STRIPE

    def _slot(self, slot: int) -> int:
        slot = int(slot)
        if not 1 <= slot <= self.state_slots:
            raise ValueError(f"state slot {slot} is outside [1, {self.state_slots}]")
        return slot

    def kda_base(self, slot: int) -> int:
        """The first KDA host page of state slot ``slot``."""
        return self.kda_start + (self._slot(slot) - 1) * self.kda_pages_per_slot

    def committed_page(self, slot: int, group: int) -> int:
        """The host page of slot ``slot``'s committed state in KDA group ``group`` (column 0)."""
        if not 0 <= group < self.kda_groups:
            raise ValueError(f"KDA group {group} is outside [0, {self.kda_groups})")
        return self.kda_base(slot) + group * (1 + self.queries)

    @property
    def rider_host_pages(self) -> int:
        """Host pages of the rider pool: the null page and ``ring_slots`` rings (pools.rider_pages, a traced shape of
        the draft program), of which slots ``1 .. S`` use the first ``S``."""
        return 1 + self.ring_slots * self.ring_pages

    def ring(self, slot: int) -> list[int]:
        """The rider host pages of slot ``slot``'s ring, in ring order."""
        first = 1 + (self._slot(slot) - 1) * self.ring_pages
        return list(range(first, first + self.ring_pages))

    def ring_kernel_pages(self, slot: int) -> list[int]:
        """The kernel pages of slot ``slot``'s ring: entry ``i`` is kernel page ``i % 2`` of ring page ``i // 2``."""
        return pl.ring_kernel_pages(self.ring(slot)).tolist()

    def ring_destination(self, slot: int, position: int) -> tuple[int, int, int]:
        """(rank, rider kernel page, row) of the drafter row of ``position`` of slot ``slot``'s request."""
        position = int(position)
        rank = position // STRIPE % self.cp_size
        page = self.ring(slot)[position // (2 * self.cp_size * STRIPE) % self.ring_pages]
        half = position // (self.cp_size * STRIPE) % 2
        return rank, 2 * page + half, position % STRIPE


def ring_table(layout: DecodePageLayout, slots: Sequence[Optional[int]], batch: int):
    """``spec_draft_rider_ring`` ``[B, 2 x ring]`` int32 (pools.rider_ring_input): the rider kernel pages of each
    row's ring, 0 for a pad row."""
    return pl.rider_ring_input([() if slot is None else layout.ring(slot) for slot in slots], batch)


def draft_window_start(prompt_len: int) -> int:
    """The first drafter context position a PD prefill sends (and ``draft_lo`` of the request's first step): the
    window's start, rounded down to a whole stripe."""
    return max(0, int(prompt_len) - DRAFT_WINDOW) // STRIPE * STRIPE


# -- MLA page pairs --------------------------------------------------------------------------------------------------


@dataclass
class PairPlan:
    """The pages one allocation hands out: ``pages[i]`` the new allocator pages of request ``i`` in order, ``taken``
    the free pages it removes (pair heads and their buddies), ``reserve`` the buddies it holds for a request that ends
    on a head, ``unreserve`` the held buddies it hands out."""

    pages: list[list[int]]
    taken: list[int] = field(default_factory=list)
    reserve: set[int] = field(default_factory=set)
    unreserve: set[int] = field(default_factory=set)


class PagePairBook:
    """Hand out allocator pages in aligned pairs ``(2b, 2b + 1)``, ``b >= 1``.

    A request that needs a new page and whose last page is a head ``2b`` takes its buddy ``2b + 1``, which the book
    held for it (``reserved``); any other new page starts a pair: the lowest free head whose buddy is free too, the
    buddy then held for the request. A free page whose partner is still in use waits in the free list until its
    partner comes back. A freed head releases its held buddy with it (``release``)."""

    def __init__(self) -> None:
        self.reserved: set[int] = set()

    @staticmethod
    def heads(free_pages: np.ndarray) -> np.ndarray:
        """The free pair heads (even pages from 2 whose buddy is free), ascending."""
        free = np.asarray(free_pages, dtype=np.int64)
        if free.size == 0:
            return free
        present = np.zeros(int(free.max()) + 2, dtype=bool)
        present[free] = True
        even = np.arange(2, present.size - 1, 2)
        return even[present[even] & present[even + 1]]

    def plan(self, last_pages: Sequence[Optional[int]], counts: Sequence[int], free_pages: np.ndarray) -> Optional[PairPlan]:
        """The pages of an allocation of ``counts[i]`` new pages after page ``last_pages[i]`` (None: the request has
        none) for each request, or None when the free pairs fall short. Changes nothing.

        Raises:
            RuntimeError: a request ends on a head whose buddy the book does not hold (the pair rule was broken).
        """
        heads = iter(self.heads(free_pages).tolist())
        plan = PairPlan([])
        held = set(self.reserved)
        for last, count in zip(last_pages, counts):
            current = None if last is None or int(last) < 0 else int(last)
            pages = []
            for _ in range(int(count)):
                if current is not None and current % 2 == 0:
                    buddy = current + 1
                    if buddy not in held:
                        raise RuntimeError(f"a request ends on page {current} and its buddy {buddy} is not held for it: "
                                           "the MLA pages were not handed out in aligned pairs")
                    held.discard(buddy)
                    if buddy in plan.reserve:
                        plan.reserve.discard(buddy)
                    else:
                        plan.unreserve.add(buddy)
                    current = buddy
                else:
                    head = next(heads, None)
                    if head is None:
                        return None
                    plan.taken += [head, head + 1]
                    plan.reserve.add(head + 1)
                    held.add(head + 1)
                    current = head
                pages.append(current)
            plan.pages.append(pages)
        return plan

    def apply(self, plan: PairPlan) -> None:
        self.reserved -= plan.unreserve
        self.reserved |= plan.reserve

    def release(self, page_ids: np.ndarray) -> np.ndarray:
        """``page_ids`` with the held buddy of each freed head, which goes back with it."""
        pages = np.asarray(page_ids, dtype=np.int64).reshape(-1)
        buddies = [int(p) + 1 for p in pages.tolist() if p % 2 == 0 and int(p) + 1 in self.reserved]
        if not buddies:
            return pages
        self.reserved -= set(buddies)
        return np.concatenate([pages, np.asarray(buddies, dtype=np.int64)])

    def clear(self) -> None:
        self.reserved = set()


# -- the zero rule after a transfer ----------------------------------------------------------------------------------


def tail_zero_blocks(req_to_token_row: np.ndarray, prompt_len: int, kv_len: int, cp_rank: int,
                     cp_size: int = DCP_SIZE) -> np.ndarray:
    """The allocator pages (this rank's blocks) of a request that the PD transfer of its first ``prompt_len`` tokens
    does not write on rank ``cp_rank``, which the decode zeroes once the transfer has landed: every block of a
    virtual page whose stripe on this rank starts at or past the prompt end, within the request's first ``kv_len``
    positions, and the held buddy of the request's last page when that page is a pair head (the verify reads it as the
    second kernel page of the pair). A prefill zeroes the rows past the prompt end inside the stripes it sends."""
    page = cp_size * STRIPE
    pages = -(-int(kv_len) // page)
    row = np.asarray(req_to_token_row[: pages * page : page], dtype=np.int64)
    blocks = row // page
    starts = np.arange(pages, dtype=np.int64) * page + int(cp_rank) * STRIPE
    out = blocks[starts >= int(prompt_len)]
    if pages and blocks[-1] % 2 == 0:
        out = np.concatenate([out, blocks[-1:] + 1])
    return out


# -- PD registration -------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class KvEntry:
    """A KV entry a decode registers: the global layer id it pairs by, its device address and bytes, and the bytes of
    one page (the stride of the page index the prefill writes)."""

    layer_id: int
    ptr: int
    nbytes: int
    item_bytes: int


@dataclass(frozen=True)
class StateEntry:
    """A part of one KDA layer's committed state: the prefill writes ``nbytes`` at ``ptr + slot * slot_stride`` for
    the request's state slot ``slot``."""

    layer_id: int
    group: int
    part: str
    ptr: int
    slot_stride: int
    nbytes: int


def mla_entries(mla_layer_ids: Sequence[int], buffer_ptrs: Sequence[int],
                buffer_bytes: int = POOL_HOST_PAGES * HOST_PAGE_BYTES) -> list[KvEntry]:
    """The 24 MLA entries: buffer ``i`` for MLA layer ``i``, one kernel page (one stripe) per allocator page."""
    if len(mla_layer_ids) != len(buffer_ptrs):
        raise ValueError(f"{len(mla_layer_ids)} MLA layers for {len(buffer_ptrs)} shared buffers")
    return [KvEntry(int(layer), int(p), int(buffer_bytes), KERNEL_PAGE_BYTES)
            for layer, p in zip(mla_layer_ids, buffer_ptrs)]


def draft_entries(rider_ptrs: Sequence[int], rider_bytes: int, num_hidden_layers: int) -> list[KvEntry]:
    """The 5 drafter entries after the target's, in the draft band above the target's layers (ids ``L .. L + 4``),
    one kernel page per stripe."""
    if len(rider_ptrs) != DRAFT_LAYERS:
        raise ValueError(f"{len(rider_ptrs)} rider layers; the drafter has {DRAFT_LAYERS}")
    return [KvEntry(num_hidden_layers + d, int(p), int(rider_bytes), KERNEL_PAGE_BYTES)
            for d, p in enumerate(rider_ptrs)]


def kda_state_entries(layout: DecodePageLayout, kda_layer_ids: Sequence[int], buffer_ptrs: Sequence[int],
                      pool_of_layer: Sequence[int]) -> list[StateEntry]:
    """The 4 parts of each KDA layer's committed state page, KDA layer ``j`` (global layer ``l``) in buffer
    ``pool_of_layer[l]`` and group ``j % 3``: ``ptr + slot * slot_stride`` is the part in slot ``slot``'s committed
    page of the layer's group."""
    stride = layout.kda_pages_per_slot * HOST_PAGE_BYTES
    entries = []
    for j, layer_id in enumerate(kda_layer_ids):
        group, buffer = j % layout.kda_groups, int(pool_of_layer[int(layer_id)])
        # slot 0 would sit one slot stride below slot 1's pages: SGLang's mamba slots start at 1
        page0 = layout.kda_start - layout.kda_pages_per_slot + group * (1 + layout.queries)
        base = int(buffer_ptrs[buffer]) + page0 * HOST_PAGE_BYTES
        for name, offset, nbytes in STATE_PARTS:
            entries.append(StateEntry(int(layer_id), group, name, base + offset, stride, nbytes))
    return entries


FRAME_PREFIX = "sglang-neuron:k3-decode-pages:"
DRAFT_ROWS = "ring"
"""The drafter-row form of the rider pool: the drafter window of each request in its state slot's ring pages
(``DecodePageLayout.ring_destination``), DCP-sharded like the target rows."""


def registration_frame(layout: DecodePageLayout) -> bytes:
    """What the decode announces of its per-request pages: every page is a function of the request's state slot ``s``
    (the state index the prefill receives). A prefill places, for state slot ``s``:

    - KDA layer ``j`` (KDA layer order; group ``j % 3``, buffer of its global layer's MLA entry ``j // 3``): the parts
      of ``STATE_PARTS`` at host page ``kda_start + (s - 1) slot_pages + (j % 3) group_pages`` of the buffer;
    - drafter rows: position ``p >= draft_lo`` on rank ``(p // 128) % 32``, rider host page ``1 + (s - 1) ring + (p //
      8192) % ring``, kernel half ``(p // 4096) % 2``, of each drafter entry; ``draft_lo`` is the window start
      (``draft_window_start``)."""
    return page_rule(layout).frame()


def page_rule(layout: DecodePageLayout):
    """The rule ``registration_frame`` announces, as the prefill's stripe sender reads it (``dcp_stripes.PageRule``)."""
    from sglang_neuron.disaggregation import dcp_stripes

    return dcp_stripes.PageRule(kda_start=layout.kda_start, slot_pages=layout.kda_pages_per_slot,
                                group_pages=1 + layout.queries, page=HOST_PAGE_BYTES, ring=layout.ring_pages,
                                window=DRAFT_WINDOW, state=tuple(STATE_PARTS))


def parse_registration_frame(payload: bytes) -> dict:
    """The fields of a page frame: the codec the prefill's stripe sender reads it with (``dcp_stripes``)."""
    from sglang_neuron.disaggregation import dcp_stripes

    return dcp_stripes.parse_page_frame(payload)


# -- SGLang adapters (scheduler processes) ---------------------------------------------------------------------------


def _layout_of(kvcache) -> DecodePageLayout:
    """The page layout the request table's K3 state pool built (``K3DecodeStatePool.layout``)."""
    state = getattr(kvcache, "mamba_pool", None)
    layout = getattr(state, "layout", None)
    if not isinstance(layout, DecodePageLayout):
        raise RuntimeError(f"the Kimi-K3 decode allocator runs over a hybrid pool whose state pool is the Kimi-K3 "
                           f"decode's (K3DecodeStatePool), not {type(state).__name__}")
    return layout


def _allocator_class():
    import torch

    from sglang_neuron.mem_cache.allocator import NeuronPagedTokenToKVPoolAllocator

    class K3PairPageAllocator(NeuronPagedTokenToKVPoolAllocator):
        """The platform's paged allocator over the Kimi-K3 decode's page space: it hands out the MLA allocator pages
        ``[2, 2 kda_start)`` only, in aligned pairs (PagePairBook); the KDA host pages above them belong to the state
        slots."""

        def clear(self):
            super().clear()
            self.pair_book = PagePairBook()
            layout = _layout_of(self.get_kvcache())
            if self.num_pages != layout.allocator_pages:
                raise RuntimeError(f"SGLang built the Kimi-K3 decode allocator over {self.num_pages} pages; its page "
                                   f"space has {layout.allocator_pages} (launch with --max-total-tokens "
                                   f"{layout.max_total_tokens})")
            pages = layout.mla_allocator_pages
            self.free_pages = torch.arange(pages.start, pages.stop, dtype=torch.int64, device=self.device)

        def _take(self, last_pages, counts):
            need = sum(int(c) for c in counts)
            if need == 0:
                return [[] for _ in counts]
            plan = self.pair_book.plan(last_pages, counts, self.free_pages.numpy())
            # the staged pages that are safe first (a pair takes two pages), then every staged page
            for want in (2 * need, None):
                if plan is not None or not (self.need_sort and self.staged_pages):
                    break
                self.merge_and_sort_free(want)
                plan = self.pair_book.plan(last_pages, counts, self.free_pages.numpy())
            if plan is None:
                return None
            if plan.taken:
                taken = torch.tensor(plan.taken, dtype=torch.int64)
                self._zero_handed_out(taken)
                self.free_pages = self.free_pages[~torch.isin(self.free_pages, taken)]
            self.pair_book.apply(plan)
            return plan.pages

        def alloc(self, need_size: int):
            if need_size % self.page_size:
                raise ValueError(f"an allocation of {need_size} tokens is not whole {self.page_size}-token pages")
            pages = self._take([None], [need_size // self.page_size])
            if pages is None:
                return None
            ids = torch.tensor(pages[0], dtype=torch.int64)
            return (ids[:, None] * self.page_size + torch.arange(self.page_size)).reshape(-1)

        def alloc_extend(self, prefix_lens, prefix_lens_cpu, seq_lens, seq_lens_cpu, last_loc, extend_num_tokens,
                         num_new_pages=None):
            page = self.page_size
            prefixes, seqs = prefix_lens_cpu.tolist(), seq_lens_cpu.tolist()
            lasts = last_loc.to("cpu").tolist()
            counts = [max(0, -(-s // page) - -(-p // page)) for p, s in zip(prefixes, seqs)]
            last_pages = [int(loc) // page if p > 0 else None for p, loc in zip(prefixes, lasts)]
            pages = self._take(last_pages, counts)
            if pages is None:
                return None
            out = torch.empty((extend_num_tokens,), dtype=torch.int64, device=self.device)
            pos = 0
            slot = torch.arange(page, dtype=torch.int64)
            for prefix, seq, last, new in zip(prefixes, seqs, lasts, pages):
                n = seq - prefix
                if n <= 0:
                    continue
                first = min(seq, -(-prefix // page) * page) - prefix
                if first > 0:
                    out[pos:pos + first] = int(last) + 1 + slot[:first]
                if n - first > 0:
                    block = (torch.tensor(new, dtype=torch.int64)[:, None] * page + slot).reshape(-1)
                    out[pos + first:pos + n] = block[:n - first]
                pos += n
            return out

        def alloc_decode(self, seq_lens, seq_lens_cpu, last_loc):
            page = self.page_size
            seqs, lasts = seq_lens_cpu.tolist(), last_loc.to("cpu").tolist()
            counts = [int((s - 1) % page == 0) for s in seqs]
            last_pages = [int(loc) // page if s > 1 else None for s, loc in zip(seqs, lasts)]
            pages = self._take(last_pages, counts)
            if pages is None:
                return None
            out = [new[0] * page if c else int(loc) + 1 for c, loc, new in zip(counts, lasts, pages)]
            return torch.tensor(out, dtype=torch.int64, device=self.device)

        def _release_page_ids(self, *page_ids):
            book = getattr(self, "pair_book", None)
            if book is not None:
                page_ids = tuple(torch.from_numpy(book.release(ids.to("cpu").numpy())) for ids in page_ids)
            super()._release_page_ids(*page_ids)

    return K3PairPageAllocator


_ALLOCATOR = None


def pair_page_allocator_class():
    """The Kimi-K3 decode's allocator class (a NeuronPagedTokenToKVPoolAllocator), built once SGLang is importable."""
    global _ALLOCATOR
    if _ALLOCATOR is None:
        _ALLOCATOR = _allocator_class()
    return _ALLOCATOR


# MambaPool constructor options and the value that means "off"
_STATE_OFF_OPTIONS = ("enable_memory_saver", "enable_linear_replayssm", "envelope_layout", "enable_linear_replayssm_spec",
                      "replayssm_request_ring")


def state_pool_class(queries: int, kda_layer_ids: Sequence[int]):
    """The request tables' state pool class of the Kimi-K3 decode (``NeuronModel.state_pool_class``): SGLang's mamba
    slots, with no state tensor of their own. A slot's KDA state lives in its host pages of the shared MLA buffers
    (``DecodePageLayout.kda_base``), which the verify program reads and writes; SGLang's per-draft-token speculative
    state is not built, since the verify keeps one state page per verify row."""
    from sglang.srt.mem_cache.memory_pool import MambaPool

    ids = tuple(int(i) for i in kda_layer_ids)

    class K3DecodeStatePool(MambaPool):
        kda_layers = ids

        def __init__(self, *, size: int, spec_state_size: int, cache_params, mamba_layer_ids, device,
                     speculative_num_draft_tokens: Optional[int] = None, speculative_eagle_topk: Optional[int] = None,
                     linear_replayssm_cache_len: int = 16, **options):
            unknown = sorted(set(options) - set(_STATE_OFF_OPTIONS))
            on = sorted(k for k in _STATE_OFF_OPTIONS if options.get(k))
            if speculative_eagle_topk not in (None, 0, 1):
                on.append("speculative_eagle_topk")
            if speculative_num_draft_tokens not in (None, 0, queries):
                on.append(f"speculative_num_draft_tokens={speculative_num_draft_tokens} (the verify runs {queries})")
            if unknown or on:
                raise ValueError(f"Kimi-K3 decode state pool: upstream options {on + unknown} are not served; the "
                                 "verify keeps one state page per verify row")
            if tuple(int(i) for i in mamba_layer_ids) != ids:
                raise ValueError(f"Kimi-K3 decode state pool: SGLang asks for state layers {list(mamba_layer_ids)}; the "
                                 f"decode keeps all {len(ids)} KDA layers {list(ids)}")
            self.size = int(size)
            self.layout = DecodePageLayout(queries, self.size)
            self.device = device
            self.mamba_layer_ids = list(ids)
            self.num_mamba_layers = len(ids)
            self.debug_memory_pool = False
            self.enable_custom_mem_pool, self.custom_mem_pool = False, None
            self.enable_linear_replayssm = self.enable_linear_replayssm_spec = False
            self.replayssm_write_pos = self.replayssm_spec_write_pos = None
            self.replayssm_cache_base = self.replayssm_is_flush = None
            self.replayssm_spec_fold = False
            self.mamba_cache = None
            self.mem_usage = 0.0
            self._buffer_ptrs: Optional[tuple[int, ...]] = None

        def bind_buffers(self, buffer_ptrs: Sequence[int], pool_of_layer: Sequence[int]) -> None:
            """The device addresses of the shared buffers (the MLA pool's) and the buffer of each layer (the map the
            programs bind), for the PD state entries."""
            self._buffer_ptrs = tuple(int(p) for p in buffer_ptrs)
            self._pool_of_layer = tuple(int(i) for i in pool_of_layer)

        def state_entries(self) -> list[StateEntry]:
            if self._buffer_ptrs is None:
                raise RuntimeError("the Kimi-K3 decode state pool has no shared buffers bound (bind_buffers)")
            return kda_state_entries(self.layout, self.mamba_layer_ids, self._buffer_ptrs, self._pool_of_layer)

        def get_contiguous_buf_infos(self):
            """The PD state entries: per KDA layer its 4 parts, each at the address of slot 0 of its group's committed
            pages with the slot stride as its item bytes (so ``ptr + slot * item`` is the request's part); a prefill
            writes the part's own bytes (registration_frame)."""
            entries = self.state_entries()
            return ([e.ptr for e in entries], [e.slot_stride * (self.size + 1) for e in entries],
                    [e.slot_stride for e in entries])

        def registration_frame(self) -> bytes:
            """The page rule the decode's PD registration carries (sglang_patches.decode_pool_frames)."""
            return registration_frame(self.layout)

        def get_state_layer_ids(self):
            return [e.layer_id for e in self.state_entries()]

        def _iter_transfer_state_entries(self):
            # upstream walks the state tensors of mamba_cache for the per-tensor slicing metadata (dim per tensor,
            # conv shard groups, slice outer counts); the decode's state lives in the shared buffers and is sent
            # whole per part (state_entries), so there is no such metadata
            return iter(())

        def get_state_dim_per_tensor(self):
            return []

        def get_kv_size_bytes(self):
            return 0

        def clear_slots(self, indices) -> None:
            # a slot's committed state arrives whole with its PD transfer, and each verify row's page is written
            # before it is read; an eager fill here would not be ordered before the transfer's write
            return None

        def copy_from(self, src_indices, dst_indices) -> None:
            raise NotImplementedError("the Kimi-K3 decode keeps no radix checkpoint of its KDA state")

        def register_slot_state(self, state) -> None:
            raise ValueError("the Kimi-K3 decode state pool holds no sibling slot state")

        def get_cpu_copy(self, *args, **kwargs):
            raise NotImplementedError("host offload of the Kimi-K3 decode state is not implemented")

        load_cpu_copy = get_cpu_copy

        def drain(self) -> None:
            return None

    return K3DecodeStatePool


class _NoWrappedView:
    """The layer router SGLang's PD registration reads of a draft pool: the rider pool wraps no token KV view, so
    its entries take the draft band ids by layer count."""

    @staticmethod
    def wrapped_view():
        return None

    @staticmethod
    def part_view(part):
        return None


class RiderPool:
    """The drafter context rows of the 5 drafter layers, stacked ``[5, rider host pages, 73728]`` bf16 (the MLA page
    format of the shared buffers), and the draft pool SGLang's PD decode registers after the target's entries
    (``BaseSpecWorker.primary_draft_kv_pool``): one entry per drafter layer, one kernel page per stripe. The draft
    program binds the same rows as ``kernel_rows`` (``pools.DecodePools.rider``)."""

    layer_router = _NoWrappedView()
    # the drafter-row form the decode announces in its PD registration (sglang_patches.decode_pool_frames)
    neuron_draft_rows = DRAFT_ROWS

    def __init__(self, layout: DecodePageLayout, device=None):
        import torch

        self.layout = layout
        self.layer_num = DRAFT_LAYERS
        self.page_size = STRIPE
        self.start_layer = 0
        self.rows = torch.zeros((DRAFT_LAYERS, layout.rider_host_pages, pl.PAGE_WIDTH), dtype=torch.bfloat16,
                                device=device)

    def kernel_rows(self):
        """The rows as the draft program binds them: ``[5 x 2 P, 128 x 288]`` bf16, one kernel page a row (drafter
        layer ``d`` holds rows ``[2 P d, 2 P (d + 1)]``, host page ``b`` rows ``2b, 2b + 1`` of it)."""
        return self.rows.view(DRAFT_LAYERS * 2 * self.layout.rider_host_pages,
                              pl.KERNEL_PAGE_ROWS * pl.RIDER_ROW_WIDTH)

    def get_contiguous_buf_infos(self):
        from sglang_neuron.backend.device_io import device_address

        layers = [self.rows[d] for d in range(DRAFT_LAYERS)]
        return ([device_address(t) for t in layers], [t.numel() * t.element_size() for t in layers],
                [KERNEL_PAGE_BYTES] * DRAFT_LAYERS)

    def get_kv_size_bytes(self):
        return self.rows.numel() * self.rows.element_size()
