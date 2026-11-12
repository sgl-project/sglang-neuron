# SPDX-License-Identifier: Apache-2.0
"""Host side of the pooled indexer (`Glm52GkGeometry.idx_pool`): the pool's membership and its per-step tables.

The pool deals whole top-k chunks (the chunk-dealt pool, CDP, `IdxChunkPool`), each at the rows the unpooled indexer
uses, so each holder runs the unpooled chunk loop on unpooled bytes and only the chunk's phase-1 candidates cross the
pool.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np

from sglang_neuron.models.glm52.layout.decode.gk_geometry import Glm52GkGeometry


CDP_TABLES = (
    "idx_pool_items",
    "idx_pool_n_items",
    "idx_pool_region_base",
    "idx_pool_new_key_rows",
    "idx_pool_unrot",
)

#: The columns of one ``idx_pool_items`` row (`IdxChunkPool.item_row`); columns 6 and 7 are zero.
ITEM_COLS = 8
HEADSUM_COL_TILE = 32  # indexer_constants.HEADSUM_COL_TILE
HEADSUM_SLICE = 512  # indexer_constants.HEADSUM_SLICE


def headsum_block_keys(num_tokens: int, num_keys: int) -> int:
    """Keys of one head-sum block of the chunk loop (the kernel's ``headsum_packed_geometry``): 8192 at 8 tokens.

    The block is the unit of the loop's block cursor (``_DynamicLoop.cur``), so region ``r`` starts at cursor
    ``r * region_keys / block_keys``.

    Example:
        >>> headsum_block_keys(8, 262144), headsum_block_keys(1, 262144), headsum_block_keys(8, 4096)
        (8192, 65536, 4096)
    """
    slot = 1
    while slot < num_tokens:
        slot *= 2
    return min(4 * (HEADSUM_COL_TILE // slot) * HEADSUM_SLICE, num_keys)


def candidate_words(index_topk: int) -> int:
    """uint32 words of one token's phase-1 candidate slot in the CDP send / recv buffers: ``index_topk`` bf16
    values (``index_topk / 2`` words), then ``index_topk`` int32 positions. 3072 at 2048."""
    return 3 * index_topk // 2


@dataclass
class IdxChunkPool:
    """The chunk-dealt pool (CDP): its membership, its deal and its per-step tables.

    CP group ``g`` is in pool ``g % idx_pool_split`` at pool index ``e = g // idx_pool_split``; ``pool_id`` selects
    one of the ``idx_pool_split`` pools, and `for_rank` picks the pool of a global rank. A pool's group at key
    position ``c`` is its ``E`` ranks at ``c``. At 1m (split 4) that is ``E = 8`` ranks ``16 e + 4 p + c``,
    one per chip.

    THE DEAL. A rank's indexer cache keeps the unpooled ``idx_keys`` rows, cut into ``R`` REGIONS of one top-k chunk
    each (`regions`, `region_keys`: 8 x 32,768 at 1M). Key position ``x`` of the sequence in pool slot ``s``::

        G = x // bs;  c = G % cp;  m = G // cp;  q = m // region_blocks     # bs 32, cp 4, region_blocks 1024
        holder = (s + q) % E;  row = bs * m + x % bs                           # the row the unpooled indexer uses

    So region ``r`` of holder ``e`` is chunk ``r`` of owner ``(e - r) % E`` (`region_owner`), and region 0 is the
    holder's own chunk 0. A chunk is live like the unpooled chunk loop's trip count at the chunk width: chunk 0
    always, chunk ``q`` iff ``live_keys > q * region_keys``.

    Example:
        >>> from sglang_neuron.models.glm52 import points
        >>> pool = IdxChunkPool(points.POINTS["1m"].decode_geometry(), 1)
        >>> pool.degree, pool.regions, pool.region_keys, [group[2] for group in pool.groups][:3]
        (8, 8, 32768, [6, 22, 38])
        >>> t = pool.tables(key_rank=0, pool_index=3, bases=[1040384] * 3 + [131072] * 5, counts=[8] * 8, spec=8)
        >>> int(t["idx_pool_n_items"][0, 0]), t["idx_pool_items"][0].tolist()
        (3, [1, 2, 4, 98304, 50176, 131072, 0, 0])
    """

    geo: Glm52GkGeometry
    pool_id: int = 0

    def __post_init__(self):
        geo = self.geo
        if not geo.idx_pool_degree:  # the geometry refuses a pool it cannot serve here
            raise ValueError("IdxChunkPool needs a geometry that pools its indexer (idx_pool)")
        if not 0 <= self.pool_id < geo.idx_pool_split:
            raise ValueError(f"pool_id {self.pool_id} is not below idx_pool_split {geo.idx_pool_split}")
        if self.region_keys % geo.idx_block_size:
            raise ValueError(
                f"a region of {self.region_keys} keys is not whole {geo.idx_block_size}-key blocks"
            )

    @classmethod
    def for_rank(cls, geo: Glm52GkGeometry, rank: int) -> "IdxChunkPool":
        """The pool that holds global ``rank``."""
        for g, group in enumerate(geo.replica_meshes()["cp"]):
            if rank in group:
                return cls(geo, g % geo.idx_pool_split)
        raise ValueError(f"rank {rank} is in no CP group")

    @property
    def split(self) -> int:
        return self.geo.idx_pool_split

    @cached_property
    def pool_slots(self) -> list[int]:
        """Global sequence slots (= CP groups) of this pool, in pool order."""
        return [e * self.split + self.pool_id for e in range(self.degree)]

    # The membership is a function of the frozen geometry and the pool id alone, and the decode reads it dozens of times
    # a step (`tables` through `degree`), so each pool computes it once.
    @cached_property
    def groups(self) -> list[tuple[int, ...]]:
        """This pool's CP groups in pool order (every ``idx_pool_split``-th group of the ``cp`` mesh), each a tuple
        of global ranks in key-shard order."""
        return [group for g, group in enumerate(self.geo.replica_meshes()["cp"]) if g % self.split == self.pool_id]

    @cached_property
    def degree(self) -> int:
        """E, the pooled groups: the geometry's CP groups over ``idx_pool_split``."""
        return len(self.groups)

    def pool_of_rank(self, rank: int) -> tuple[int, int]:
        """``(key_rank c, pool_index e)`` of a global rank."""
        for e, group in enumerate(self.groups):
            if rank in group:
                return group.index(rank), e
        raise ValueError(f"rank {rank} is in no CP group of pool {self.pool_id}")

    # the deal
    @cached_property
    def regions(self) -> int:
        """R, the top-k chunks (regions) of one rank's indexer cache (`Glm52GkGeometry.idx_topk_chunks`)."""
        return self.geo.idx_topk_chunks

    @cached_property
    def region_keys(self) -> int:
        """W, keys of one region = one top-k chunk (`Glm52GkGeometry.idx_pool_chunk_keys`)."""
        return self.geo.idx_pool_chunk_keys

    @cached_property
    def region_blocks(self) -> int:
        """32-key blocks of one region."""
        return self.region_keys // self.geo.idx_block_size

    def holder(self, slot_index: int, chunk: int) -> int:
        """Pool index of the member that holds chunk ``chunk`` of the sequence in pool slot ``slot_index``."""
        return (slot_index + chunk) % self.degree

    def region_owner(self, pool_index: int, region: int) -> int:
        """Pool slot of the sequence whose chunk ``region`` sits in region ``region`` of member ``pool_index``."""
        return (pool_index - region) % self.degree

    def live_keys(self, base: int | None, key_rank: int) -> int:
        """Keys of key shard ``key_rank`` below the cut ``base`` (None or negative = nothing resident).

        The unpooled kernel's ``live_blocks`` before its rounding: with ``rnd = cp * bs``,
        ``live = (b // rnd) * bs + clamp(b % rnd - bs * key_rank, 0, bs)``, ``b = max(base, 0)``, at most ``idx_keys``.
        """
        geo = self.geo
        bs, cp = geo.idx_block_size, geo.idx_shard_degree
        b = max(int(base), 0) if base is not None else 0
        rnd = cp * bs
        live = (b // rnd) * bs + min(max(b % rnd - bs * key_rank, 0), bs)
        return min(live, geo.idx_keys)

    def chunk_live(self, base: int | None, key_rank: int, chunk: int) -> bool:
        """Whether the unpooled chunk loop scores chunk ``chunk`` of a sequence: ``chunk == 0`` or
        ``live_keys > chunk * region_keys``."""
        return chunk == 0 or self.live_keys(base, key_rank) > chunk * self.region_keys

    def new_key_row(
        self, position: int, slot: int, key_rank: int, pool_index: int
    ) -> int:
        """Row on member ``(key_rank, pool_index)`` of a key at ``position`` of pool slot ``slot``'s sequence, or -1."""
        geo = self.geo
        bs, cp = geo.idx_block_size, geo.idx_shard_degree
        if position < 0:
            return -1
        g = position // bs
        m = g // cp
        if g % cp != key_rank or m >= geo.idx_blocks:
            return -1
        if self.holder(slot, m // self.region_blocks) != pool_index:
            return -1
        return m * bs + position % bs

    def new_key_rows(
        self,
        key_rank: int,
        pool_index: int,
        bases: list[int | None],
        counts: list[int],
        spec: int,
    ) -> np.ndarray:
        """``idx_pool_new_key_rows`` [E * spec, 1] int32: member ``(key_rank, pool_index)``'s row of pool slot
        ``s``'s new key ``t`` (at ``bases[s] + t``, real when ``t < counts[s]``), else -1."""
        out = np.full((self.degree * spec, 1), -1, np.int32)
        for s, base in enumerate(bases):
            if base is None:
                continue
            for t in range(min(counts[s], spec)):
                out[s * spec + t, 0] = self.new_key_row(
                    base + t, s, key_rank, pool_index
                )
        return out

    # the per-step tables
    def send_offsets(self, part: int, spec: int) -> tuple[int, int]:
        """``(values, positions)`` offsets of part ``part``'s token-0 candidate slot in the send (or recv) buffer
        ``[E parts x spec x candidate_words]`` uint32: in bf16 elements ``part * spec * 3 * topk`` and in int32
        elements ``part * spec * candidate_words + topk / 2``; token ``t`` is one ``candidate_words`` row further."""
        k = self.geo.index_topk
        words = candidate_words(k)
        return part * spec * 2 * words, part * spec * words + k // 2

    def item_row(self, pool_index: int, region: int, spec: int) -> list[int]:
        """One ``idx_pool_items`` row: region, owner, block cursor, send offsets (values, positions), label, 0, 0."""
        geo = self.geo
        owner = self.region_owner(pool_index, region)
        send_v, send_p = self.send_offsets(owner, spec)
        cursor = region * (self.region_keys // headsum_block_keys(spec, geo.idx_keys))
        label = region * geo.idx_shard_degree * self.region_keys
        return [region, owner, cursor, send_v, send_p, label, 0, 0]

    def foreign_items(
        self, key_rank: int, pool_index: int, bases: list[int | None]
    ) -> list[int]:
        """The live FOREIGN regions of member ``(key_rank, pool_index)``, ascending: region ``r >= 1`` whose owner's
        chunk ``r`` is live."""
        return [
            r
            for r in range(1, self.regions)
            if self.chunk_live(bases[self.region_owner(pool_index, r)], key_rank, r)
        ]

    def unrot(self, pool_index: int, spec: int) -> np.ndarray:
        """``idx_pool_unrot`` [R, 2] int32: for phase-1 slot ``q``, the recv offsets (values, positions) of part
        ``(e + q) % E``, the holder of the owner's chunk ``q`` (row 0, the owner's own part, is unused)."""
        return np.array(
            [
                self.send_offsets(self.holder(pool_index, q), spec)
                for q in range(self.regions)
            ],
            np.int32,
        ).reshape(self.regions, 2)

    def tables(
        self,
        key_rank: int,
        pool_index: int,
        bases: list[int | None],
        counts: list[int] | None = None,
        *,
        spec: int,
    ) -> dict[str, np.ndarray]:
        """The GK's per-step CDP inputs of member ``(key_rank, pool_index)`` (the spec of `IdxChunkPoolTables`).

        Args:
            key_rank, pool_index: This rank's ``(c, e)``.
            bases: Per pool slot (``E`` entries, pool order), the sequence's cut: the step's first position, or None
                for a slot with no request.
            counts: Per pool slot, the step's real tokens (new keys at ``base + t`` for ``t < count``); None = 0.
            spec: Query tokens per sequence (``idx_pool_tokens``), T.

        Returns:
            ``idx_pool_items`` [R, 8] int32: row ``i < n`` is the ``i``-th live foreign region, ascending: region
            ``r``, owner ``s = (e - r) % E``, block cursor ``r * W / block_keys`` (4r), send offset of the values in
            bf16 elements ``s * T * 3 * topk`` (``s * T * 6144``), of the positions in int32 elements
            ``s * T * 3 * topk / 2 + topk / 2`` (``s * T * 3072 + 1024``), label ``r * cp * W`` (131072 r), 0, 0. Rows
            ``>= n`` repeat row ``n - 1``, or are region 1's row when ``n`` is 0 (region 0's when R is 1): the key
            prefetch after the last item reads real rows.
            ``idx_pool_n_items`` [1, 1] int32: ``n``, the foreign loop's trip count (0 .. R - 1).
            ``idx_pool_region_base`` [R] int32: the cut of region ``r``'s owner (-1 = no request).
            ``idx_pool_new_key_rows`` [E * T, 1] int32 (`new_key_rows`).
            ``idx_pool_unrot`` [R, 2] int32 (`unrot`).
        """
        E, R = self.degree, self.regions
        if len(bases) != E:
            raise ValueError(f"{len(bases)} bases for a pool of {E}")
        counts = list(counts) if counts is not None else [0] * E
        live = self.foreign_items(key_rank, pool_index, bases)
        n = len(live)
        fill = live[-1] if n else min(1, R - 1)
        regions = live + [fill] * (R - n)
        items = np.array(
            [self.item_row(pool_index, r, spec) for r in regions], np.int32
        ).reshape(R, ITEM_COLS)
        region_base = np.array(
            [
                -1
                if bases[self.region_owner(pool_index, r)] is None
                else int(bases[self.region_owner(pool_index, r)])
                for r in range(R)
            ],
            np.int32,
        )
        return {
            "idx_pool_items": items,
            "idx_pool_n_items": np.array([[n]], np.int32),
            "idx_pool_region_base": region_base,
            "idx_pool_new_key_rows": self.new_key_rows(
                key_rank, pool_index, bases, counts, spec
            ),
            "idx_pool_unrot": self.unrot(pool_index, spec),
        }

