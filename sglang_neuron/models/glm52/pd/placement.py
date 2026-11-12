# SPDX-License-Identifier: Apache-2.0
"""Where a GLM-5.2 prompt's KV lands when a PP prefill node hands it to the GK decode node.

The prefill (the prefill stages, ``stage_layout``) shows SGLang a page of ``prefill_ranks x 32`` tokens (1024 at PP4's
32 ranks a stage, 256 at PP16's 8), and its rank ``r`` keeps tokens ``[32 r, 32 r + 32)`` of every page: the 32-row
sub-block ``b = page x prefill_ranks + r`` of the prompt, the kernel's ``cp_interleave`` (``kv_pool``). The GK decode
deals each slot's 32-row blocks over its CP ranks: global block ``b`` sits on the slot's rank at CP position
``b % cp_degree``, at local block ``b // cp_degree`` (``layout.gk_kv_cache.deal_latent_blocks``), and the indexer keys
take the same deal (``indexer_shard_keys``; the key rank is the CP position). Since ``b = position // 32``
whatever the page, a position lands on the same decode rank, block and row behind either prefill point; only the
prefill rank that holds it and the page numbers differ.

So one prefill rank's sub-block of one page is exactly one decode block on one rank, written whole:

* the decode rank is the slot's rank at CP position ``r % cp_degree`` (``prefill_ranks`` is a multiple of
  ``cp_degree``, so the page's offset ``page x prefill_ranks`` never moves it);
* the decode block is ``(page x prefill_ranks + r) // cp_degree``, and the row is the token's row in the sub-block.

A decode point whose indexer pools its keys in whole top-k chunks (the chunk-dealt pool the 1M point serves,
``layout.idx_pool.IdxChunkPool``) keeps every key at the row the unpooled indexer uses, on the rank at the same key
position of the pool member that holds the key's chunk: chunk ``q = block // region_blocks`` of the slot at pool index
``s`` sits on member ``(s + q) % E``. So the latent sub-block and the key sub-block of one page go to the same local
block, the latent to the slot's own rank and the keys to that chunk's holder (`key_holder`): the slot's own rank for
chunk 0, another slot's rank of the same pool and CP position past it.

SGLang's PD names a decode's destination by page indices it publishes for each request. A GK decode's caches are slot
private with an identity block table, so its destination for page ``k`` of the request in slot ``S`` is
``S x pages_per_slot + k`` (`destination`): the slot and the page are read back from it on the prefill
(`split_destination`), and no allocator id is ever an address.

The layers pair by global id: the decode registers its 78 latent caches, then its 21 full layers' key caches, and
each prefill stage its own layers' latent caches, then its full layers' keys (`decode_entry_ids`, `stage_entry_ids`).
The MTP layer's caches are not registered while the decode serves with MTP off.

Pure Python over the kernel's geometry: no torch, NIXL or SGLang, so the transfer hooks, the launch checks and the CPU
tests share it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, NamedTuple, Sequence

#: The KV layout tag both GLM-5.2 PD pools publish in place of the plugin's shared head-major tag
#: (sglang_patches/nixl_head_major): a peer of any other layout is refused at its first contact.
GLM52_KV_LAYOUT = "sglang-neuron:glm52-gk"
#: Where a GLM-5.2 PD pool publishes its ``Glm52PdLayout`` for the handoff (models/glm52/pd/handoff reads it
#: when SGLang asks the pool for its layout).
LAYOUT_ATTR = "glm52_pd_layout"


#: The prefill's setting for a decode that drafts with the MTP layer: its last stage
#: registers the MTP layer's (checkpoint layer 78) prompt rows, the latent rows and bf16 indexer keys the stage graph
#: writes, as two more PD regions. A decode that drafts registers them and says so in its frame (mtp=1); one that does
#: not registers neither, and a prefill of the other kind is refused at first contact (models/glm52/pd/handoff
#: check_peer): upstream's pairing takes every prefill entry, so the choice is the prefill's launch, not each peer's.
PD_MTP_ENV = "SGLANG_NEURON_GLM52_PD_MTP"


def pd_mtp(environ=None) -> bool:
    """Whether this prefill sends the MTP rows (PD_MTP_ENV: 1, or unset / 0)."""
    import os

    raw = (os.environ if environ is None else environ).get(PD_MTP_ENV, "").strip()
    if raw not in ("", "0", "1"):
        raise ValueError(f"{PD_MTP_ENV}={raw!r}: 1 for a decode that drafts with the MTP layer, else unset or 0")
    return raw == "1"


@dataclass(frozen=True)
class Glm52PdLayout:
    """What a GLM-5.2 PD pool publishes for the handoff: the GK decode point both nodes pair on (a
    ``Glm52GkGeometry``) and this rank's kernel rank (the GK's rank on the decode, the stage's CP rank on the prefill,
    the rank the kernel's own deal and interleave place rows by)."""

    geo: object
    kernel_rank: int


class Glm52KvLayout(str):
    """SGLang's ``kv_layout`` of a GLM-5.2 PD pool: the string ``GLM52_KV_LAYOUT``, as both roles' layout check and the
    bootstrap server read it, carrying the pool's ``Glm52PdLayout`` (``pd_layout``) on to the KV manager."""

    def __new__(cls, pd_layout: Glm52PdLayout):
        value = super().__new__(cls, GLM52_KV_LAYOUT)
        value.pd_layout = pd_layout
        return value

    def __reduce__(self):
        # str's own reduction would rebuild it from the tag alone
        return type(self), (self.pd_layout,)


def publish_decode_layout(pool, geo, gk_rank: int) -> None:
    """For the GK decode's accounting pool: this rank's KV registration is the GK's (``geo``) at GK rank ``gk_rank``."""
    setattr(pool, LAYOUT_ATTR, Glm52PdLayout(geo, int(gk_rank)))


def publish_prefill_layout(pool, geo, cp_rank: int) -> None:
    """For a prefill stage's pool: its pages are placed for GK decode point ``geo``, and this rank is the stage's kernel
    CP rank ``cp_rank``."""
    setattr(pool, LAYOUT_ATTR, Glm52PdLayout(geo, int(cp_rank)))


class DecodeLocation(NamedTuple):
    """One prompt row on the decode: the global rank, its local 32-row block, and the row in the block."""

    rank: int
    block: int
    row: int


class SendPlan(NamedTuple):
    """One prefill rank's send of a request's pages (`PdPlacement.send_plan`): the decode rank that takes every
    page's latent sub-block, the local block of each page on its holders (the latent's and the key's alike), and the
    decode rank that takes each page's indexer-key sub-block (the latent's holder unless the decode pools its keys)."""

    holder: int
    blocks: list
    key_holders: list

    def key_pages(self, rank: int) -> list[int]:
        """The pages (by index in the send) whose key sub-blocks decode rank ``rank`` takes."""
        return [i for i, r in enumerate(self.key_holders) if r == rank]


@dataclass(frozen=True)
class PdPlacement:
    """The prefill-to-decode placement of one GK decode point (``geo``, a ``Glm52GkGeometry``) behind a prefill whose
    stages each run ``prefill_ranks`` ranks of 32-row sub-blocks.

    Raises:
        ValueError: At construction, for a pairing this placement cannot serve (see `problems`).
    """

    geo: object
    prefill_ranks: int
    _slot_ranks: tuple = field(init=False, repr=False, compare=False)
    #: [slot][cp][chunk] the rank that holds the slot's keys of one top-k chunk at one key position, and the local
    #: 32-key blocks of a chunk (one chunk of the whole slot without the pool)
    _key_ranks: tuple = field(init=False, repr=False, compare=False)
    _chunk_blocks: int = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        problems = placement_problems(self.geo, self.prefill_ranks)
        if problems:
            raise ValueError("the GLM PD placement cannot pair this prefill with the GK decode: " + "; ".join(problems))
        ranks = [[None] * self.geo.cp_degree for _ in range(self.slots)]
        for rank in range(self.geo.ranks):
            slot, cp = self.rank_slot(rank), self.rank_cp(rank)
            if ranks[slot][cp] is not None:
                raise ValueError(f"GK ranks {ranks[slot][cp]} and {rank} both hold CP position {cp} of slot {slot}")
            ranks[slot][cp] = rank
        object.__setattr__(self, "_slot_ranks", tuple(tuple(r) for r in ranks))
        self._place_keys()

    def _place_keys(self) -> None:
        """The key holders: the slot's own ranks, or the chunk-dealt pool's members (IdxChunkPool.holder), whose CP
        groups must be this placement's slots, a group's key shards its slot's CP positions."""
        geo = self.geo
        if not geo.idx_pool_degree:
            table = tuple(tuple((rank,) for rank in ranks) for ranks in self._slot_ranks)
            object.__setattr__(self, "_key_ranks", table)
            object.__setattr__(self, "_chunk_blocks", geo.local_blocks_per_seq)
            return
        from sglang_neuron.models.glm52.layout.decode.idx_pool import IdxChunkPool

        table = [None] * self.slots
        for pool_id in range(geo.idx_pool_split):
            pool = IdxChunkPool(geo, pool_id)
            slots, groups = pool.pool_slots, pool.groups
            for slot, group in zip(slots, groups):
                if tuple(int(r) for r in group) != self._slot_ranks[slot]:
                    raise ValueError(f"the indexer pool's CP group {tuple(group)} is not slot {slot}'s ranks "
                                     f"{self._slot_ranks[slot]} by CP position")
            for index, slot in enumerate(slots):
                table[slot] = tuple(
                    tuple(self._slot_ranks[slots[pool.holder(index, q)]][cp] for q in range(pool.regions))
                    for cp in range(geo.cp_degree))
        object.__setattr__(self, "_key_ranks", tuple(table))
        object.__setattr__(self, "_chunk_blocks", pool.region_blocks)

    # -- the decode's slots ------------------------------------------------------------------------------------------
    @property
    def slots(self) -> int:
        """The decode's sequence slots: SGLang's request rows 1..slots (row 0 is its padding row)."""
        return self.geo.dp_size * self.geo.num_seqs

    @property
    def block_size(self) -> int:
        return self.geo.block_size

    @property
    def page_tokens(self) -> int:
        """Tokens of one SGLang page on both nodes: one 32-row sub-block on each prefill rank."""
        return self.prefill_ranks * self.geo.block_size

    @property
    def pages_per_slot(self) -> int:
        """SGLang pages of one decode slot: the kernel's context over the page."""
        return self.geo.context // self.page_tokens

    def rank_slot(self, rank: int) -> int:
        """The slot a GK rank decodes (`Glm52GkGeometry.slot_of`)."""
        return self.geo.slot_of(rank)

    def rank_cp(self, rank: int) -> int:
        """A GK rank's CP position, which is also its indexer key rank."""
        return self.geo.coords(rank)["key_rank"]

    def slot_ranks(self, slot: int) -> tuple[int, ...]:
        """The GK ranks of one slot, by CP position."""
        return self._slot_ranks[self._slot(slot)]

    # -- one prefill sub-block -> one decode block -------------------------------------------------------------------
    def holder(self, slot: int, prefill_rank: int) -> int:
        """The decode rank that receives every sub-block prefill rank ``prefill_rank`` sends for ``slot``."""
        return self.slot_ranks(slot)[self._prefill_rank(prefill_rank) % self.geo.cp_degree]

    def decode_block(self, page: int, prefill_rank: int) -> int:
        """The holder's local block for prefill rank ``prefill_rank``'s sub-block of the request's page ``page``."""
        if not 0 <= page < self.pages_per_slot:
            raise ValueError(f"page {page} is past the {self.pages_per_slot} pages of a GK slot")
        return (page * self.prefill_ranks + self._prefill_rank(prefill_rank)) // self.geo.cp_degree

    def decode_location(self, position: int, slot: int) -> DecodeLocation:
        """Where prompt position ``position`` of the request in ``slot`` lands."""
        page, prefill_rank, row = self.prefill_location(position)
        return DecodeLocation(self.holder(slot, prefill_rank), self.decode_block(page, prefill_rank), row)

    def key_holder(self, slot: int, page: int, prefill_rank: int) -> int:
        """The decode rank that receives prefill rank ``prefill_rank``'s indexer-key sub-block of the request's page
        ``page`` in ``slot``, at the latent's local block (`decode_block`): the latent's holder, or under the
        chunk-dealt pool the holder of the sub-block's top-k chunk at the same key position."""
        cp = self._prefill_rank(prefill_rank) % self.geo.cp_degree
        return self._key_ranks[self._slot(slot)][cp][self.decode_block(page, prefill_rank) // self._chunk_blocks]

    def key_location(self, position: int, slot: int) -> DecodeLocation:
        """Where the indexer key of prompt position ``position`` of the request in ``slot`` lands."""
        page, prefill_rank, row = self.prefill_location(position)
        return DecodeLocation(self.key_holder(slot, page, prefill_rank), self.decode_block(page, prefill_rank), row)

    def prefill_location(self, position: int) -> tuple[int, int, int]:
        """``(page, prefill rank, row)`` of prompt position ``position`` on the prefill."""
        if not 0 <= position < self.geo.context:
            raise ValueError(f"position {position} is outside the GK's {self.geo.context}-position slot")
        page, rest = divmod(position, self.page_tokens)
        prefill_rank, row = divmod(rest, self.geo.block_size)
        return page, prefill_rank, row

    # -- the destination SGLang publishes for each request ------------------------------------------------------------
    def destination(self, slot: int, page: int) -> int:
        """The page index a GK decode publishes for page ``page`` of the request in ``slot``."""
        if not 0 <= page < self.pages_per_slot:
            raise ValueError(f"page {page} is past the {self.pages_per_slot} pages of a GK slot")
        return self._slot(slot) * self.pages_per_slot + page

    def destination_pages(self, slot: int, tokens: int) -> list[int]:
        """The destination of a request of ``tokens`` prompt tokens in ``slot``: one index for each of its pages."""
        pages = -(-int(tokens) // self.page_tokens)
        if not 0 < pages <= self.pages_per_slot:
            raise ValueError(f"a prompt of {tokens} tokens needs {pages} pages; a GK slot holds {self.pages_per_slot}")
        return [self.destination(slot, page) for page in range(pages)]

    def split_destination(self, index: int) -> tuple[int, int]:
        """``(slot, page)`` of one published destination index."""
        slot, page = divmod(int(index), self.pages_per_slot)
        self._slot(slot)
        return slot, page

    def holder_blocks(self, destination: Sequence[int], prefill_rank: int) -> tuple[int, list[int]]:
        """For one prefill rank's send of a request: the latent's holder rank and its local blocks, page by page.

        Raises:
            ValueError: If the destination names more than one slot.
        """
        plan = self.send_plan(destination, prefill_rank)
        return plan.holder, plan.blocks

    def send_plan(self, destination: Sequence[int], prefill_rank: int) -> "SendPlan":
        """For one prefill rank's send of a request's pages (``destination``, one published index per page, the
        prefill's pages in the same order): where each page's latent and key sub-blocks go.

        Raises:
            ValueError: If the destination names more than one slot, or its pages are not consecutive in ascending
                order (the prefill sends its pages in order, one destination index each).
        """
        split = [self.split_destination(i) for i in destination]
        slots = {s for s, _ in split}
        if len(slots) != 1:
            raise ValueError(f"a GK destination names one slot; this one names {sorted(slots)}")
        slot = slots.pop()
        pages = [page for _, page in split]
        if pages != list(range(pages[0], pages[0] + len(pages))):
            raise ValueError(f"a GK destination's pages run consecutively upward, one per page sent; slot {slot}'s "
                             f"destination names pages {pages}")
        blocks = [self.decode_block(page, prefill_rank) for _, page in split]
        chunk_ranks = self._key_ranks[slot][self._prefill_rank(prefill_rank) % self.geo.cp_degree]
        return SendPlan(self.holder(slot, prefill_rank), blocks, [chunk_ranks[b // self._chunk_blocks] for b in blocks])

    def sub_blocks(self, slot: int, tokens: int) -> Iterator[tuple[int, int, DecodeLocation]]:
        """Every 32-row sub-block a whole-page send of a ``tokens``-token prompt writes: ``(page, prefill rank,
        DecodeLocation of its row 0)``, including the sub-blocks of the last page wholly past the prompt."""
        for page in range(-(-int(tokens) // self.page_tokens)):
            for prefill_rank in range(self.prefill_ranks):
                yield page, prefill_rank, DecodeLocation(self.holder(slot, prefill_rank),
                                                         self.decode_block(page, prefill_rank), 0)

    # -- helpers -------------------------------------------------------------------------------------------------------
    def _slot(self, slot: int) -> int:
        slot = int(slot)
        if not 0 <= slot < self.slots:
            raise ValueError(f"slot {slot} is not one of the GK decode's {self.slots} slots")
        return slot

    def _prefill_rank(self, prefill_rank: int) -> int:
        prefill_rank = int(prefill_rank)
        if not 0 <= prefill_rank < self.prefill_ranks:
            raise ValueError(f"prefill rank {prefill_rank} is not one of the stage's {self.prefill_ranks} ranks")
        return prefill_rank


def placement_problems(geo, prefill_ranks: int) -> list[str]:
    """Why the SHARDED placement cannot pair a prefill of ``prefill_ranks`` ranks with GK decode point ``geo``."""
    problems = []
    cp = geo.cp_degree
    if prefill_ranks <= 0 or prefill_ranks % cp:
        problems.append(f"prefill stage ranks {prefill_ranks} are not a multiple of the decode's CP degree {cp}, so a "
                        "prefill rank's sub-blocks would move between the slot's ranks from page to page")
    if geo.idx_block_size != geo.block_size or geo.idx_shard_degree != cp:
        problems.append(f"the indexer keys are dealt in {geo.idx_block_size}-key blocks over {geo.idx_shard_degree} "
                        f"key ranks, not the latent's {geo.block_size}-row blocks over {cp} CP ranks")
    if prefill_ranks > 0 and geo.context % (prefill_ranks * geo.block_size):
        problems.append(f"the GK context {geo.context} is not a whole number of {prefill_ranks * geo.block_size}-token "
                        "pages")
    if geo.num_blocks != geo.local_blocks_per_seq:
        problems.append(f"a GK rank holds {geo.num_blocks} blocks, not one slot's {geo.local_blocks_per_seq} (a rank "
                        "of several slots or a KV depth override)")
    if getattr(geo, "head_shard", 1) != 1:
        problems.append(f"head shard {geo.head_shard}: two GK ranks would hold one CP position of a slot")
    return problems


def decode_entry_ids(geo, mtp: bool = False) -> list[int]:
    """The global layer ids of the GK decode's registered caches, in order: every layer's latent cache, then every
    full layer's indexer key cache; with ``mtp`` (an async MTP decode, models/glm52/mtp/graph.py) then the MTP
    layer's (``geo.num_layers``) latent and key caches, which a prefill launched with ``SGLANG_NEURON_GLM52_PD_MTP=1``
    sends (its last stage's entries pair by layer-id occurrence: the MTP latent before the MTP keys)."""
    ids = list(range(geo.num_layers)) + list(geo.full_layers)
    return ids + [geo.num_layers, geo.num_layers] if mtp else ids


def stage_entry_ids(geo, start_layer: int, end_layer: int) -> list[int]:
    """The global layer ids of one prefill stage's registered caches for layers ``[start_layer, end_layer)``, in
    order: its latent caches, then its full layers' key caches (no MTP layer)."""
    return list(range(start_layer, end_layer)) + [f for f in geo.full_layers if start_layer <= f < end_layer]


def entry_bytes(geo) -> tuple[int, int]:
    """Bytes of one registered item (one 32-row block) of a latent cache and of an indexer key cache."""
    return geo.block_size * geo.row_bytes, geo.idx_block_size * geo.idx_key_bytes
