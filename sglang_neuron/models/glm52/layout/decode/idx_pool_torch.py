# SPDX-License-Identifier: Apache-2.0
"""The chunk-dealt pool's per-step tables as torch ops of fixed shape, for the traced decode forward.

`IdxChunkPoolTables` computes `IdxChunkPool.tables` (numpy, the specification) from the step's slot positions with
tensor ops only, so the runner's compiled forward builds them on device every step instead of moving the host tables.
The two are equal on every rank.
"""

from __future__ import annotations

import torch

from sglang_neuron.models.glm52.layout.decode.idx_pool import ITEM_COLS, IdxChunkPool, candidate_words, headsum_block_keys


class IdxChunkPoolTables(torch.nn.Module):
    """The chunk-dealt pool's (CDP) per-step GK inputs as fixed-shape torch ops, for the traced decode forward.

    `sglang_neuron.models.glm52.layout.decode.idx_pool.IdxChunkPool.tables` (numpy) is the specification, and the
    two agree rank by rank. Every rank-dependent value -- the key rank ``c``, the pool index ``e`` and the pool's global slots -- is a
    BUFFER: the decode NEFF is one graph for all ranks, so a Python int of the rank in `forward` would be folded from
    the tracing rank into every rank's tables. The rest are rank-invariant constants. O(R) ops per step.

    Args:
        pool: The pool (`IdxChunkPool`).
        key_rank, pool_index: This rank's ``(c, e)``.
        spec: Query tokens per sequence (``idx_pool_tokens``), T.
        device: Where the constant tensors live.

    Example:
        >>> import torch
        >>> from sglang_neuron.models.glm52 import points
        >>> from sglang_neuron.models.glm52.layout.decode.idx_pool import IdxChunkPool
        >>> geo = points.POINTS["1m"].decode_geometry()
        >>> tables = IdxChunkPoolTables(IdxChunkPool(geo, 0), key_rank=0, pool_index=0, spec=8)
        >>> out = tables(torch.full((32,), 1040384, dtype=torch.int32), torch.ones(32, dtype=torch.bool),
        ...              torch.full((32,), 8, dtype=torch.int32))
        >>> int(out["idx_pool_n_items"][0, 0]), tuple(out["idx_pool_items"].shape)
        (7, (8, 8))
    """

    def __init__(
        self,
        pool: IdxChunkPool,
        key_rank: int,
        pool_index: int,
        spec: int,
        device: str = "cpu",
    ):
        super().__init__()
        geo = pool.geo
        self.E, self.R, self.T = pool.degree, pool.regions, spec
        self.W = pool.region_keys
        self.bs, self.cp = geo.idx_block_size, geo.idx_shard_degree
        self.idx_keys, self.idx_blocks = geo.idx_keys, geo.idx_blocks
        self.region_blocks = pool.region_blocks
        self.topk = geo.index_topk
        self.words = candidate_words(self.topk)
        self.cursor_unit = self.W // headsum_block_keys(spec, geo.idx_keys)
        self.label_unit = self.cp * self.W
        self.fill_region = min(1, self.R - 1)
        self.item_cols = ITEM_COLS
        dev = torch.device(device)
        i64 = torch.int64
        buffers = {
            "region_ids": torch.arange(self.R, dtype=i64, device=dev),
            "slot_ids": torch.arange(self.E, dtype=i64, device=dev),
            "t": torch.arange(spec, dtype=i64, device=dev),
            # The rank's (key rank, pool index) and the global slots of its pool (pool order), as TENSORS.
            "key_rank_t": torch.tensor([key_rank], dtype=i64, device=dev),
            "pool_index_t": torch.tensor([pool_index], dtype=i64, device=dev),
            "pool_slots": torch.as_tensor(pool.pool_slots, dtype=i64, device=dev),
        }
        for name, tensor in buffers.items():
            self.register_buffer(name, tensor, persistent=False)

    def _offsets(self, part: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """`IdxChunkPool.send_offsets` of ``part`` (any shape): (bf16 elements, int32 elements)."""
        return (
            part * (self.T * 2 * self.words),
            part * (self.T * self.words) + self.topk // 2,
        )

    def forward(
        self, bases: torch.Tensor, live: torch.Tensor, counts: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            bases: ``[slots]`` int32, each slot's first position this step (the cut).
            live: ``[slots]`` bool, whether the slot has a request.
            counts: ``[slots]`` int32, each slot's real tokens this step (new keys at ``base + t``, ``t < count``).

        ``slots`` is every slot of the decode node; the tables keep this pool's ``E`` slots (``pool_slots``).

        Returns:
            ``idx_pool_items`` [R, 8], ``idx_pool_n_items`` [1, 1], ``idx_pool_region_base`` [R],
            ``idx_pool_new_key_rows`` [E * T, 1], ``idx_pool_unrot`` [R, 2], all int32 (`IdxChunkPool.tables`).
        """
        E, T, bs, cp = self.E, self.T, self.bs, self.cp
        b = bases.index_select(0, self.pool_slots).to(torch.int64)  # [E], pool order
        lv = live.index_select(0, self.pool_slots)
        cnt = counts.index_select(0, self.pool_slots).to(torch.int64)
        r = self.region_ids
        # region r holds chunk r of owner (e - r) % E: its cut, -1 for no request (the unpooled threshold)
        owner = torch.remainder(self.pool_index_t - r, E)  # [R]
        cut = torch.where(
            lv.index_select(0, owner), b.index_select(0, owner), torch.full_like(r, -1)
        )
        # the unpooled live_blocks: keys of key rank c below the cut
        cb = torch.clamp(cut, min=0)
        rnd = cp * bs
        part = torch.clamp(
            torch.remainder(cb, rnd) - bs * self.key_rank_t, min=0, max=bs
        )
        live_keys = torch.clamp(
            torch.div(cb, rnd, rounding_mode="floor") * bs + part, max=self.idx_keys
        )
        is_item = (r >= 1) & (live_keys > self.W * r)  # [R]: the live foreign regions
        cum = torch.cumsum(is_item.to(torch.int64), 0)  # inclusive
        n = is_item.to(torch.int64).sum(0, keepdim=True)  # [1]
        # row i: the region of the min(i + 1, n)-th item (rows >= n repeat the last), region 1 (0 at R = 1) when n is 0
        k = torch.minimum(r + 1, n)
        reg = (cum[None, :] < k[:, None]).to(torch.int64).sum(1)
        reg = torch.where(n > 0, reg, torch.full_like(reg, self.fill_region))
        s_of = owner.index_select(0, reg)
        send_v, send_p = self._offsets(s_of)
        zero = torch.zeros_like(reg)
        cols = [
            reg,
            s_of,
            reg * self.cursor_unit,
            send_v,
            send_p,
            reg * self.label_unit,
        ]
        cols += [zero] * (self.item_cols - len(cols))
        items = torch.stack(cols, dim=1)
        # the new keys: pool slot s token t at base + t, real when t < count; its row here or -1
        pos = b[:, None] + self.t[None, :]  # [E, T]
        real = lv[:, None] & (self.t[None, :] < cnt[:, None]) & (pos >= 0)
        g = torch.div(pos, bs, rounding_mode="floor")
        m = torch.div(g, cp, rounding_mode="floor")
        q = torch.div(m, self.region_blocks, rounding_mode="floor")
        holder = torch.remainder(self.slot_ids[:, None] + q, E)
        here = (
            (torch.remainder(g, cp) == self.key_rank_t)
            & (holder == self.pool_index_t)
            & (m < self.idx_blocks)
        )
        row = m * bs + torch.remainder(pos, bs)
        new_key_rows = torch.where(real & here, row, torch.full_like(row, -1))
        # phase-1 slot q <- recv part (e + q) % E
        unrot_v, unrot_p = self._offsets(torch.remainder(self.pool_index_t + r, E))
        return {
            "idx_pool_items": items.to(torch.int32),
            "idx_pool_n_items": n.reshape(1, 1).to(torch.int32),
            "idx_pool_region_base": cut.to(torch.int32),
            "idx_pool_new_key_rows": new_key_rows.reshape(E * T, 1).to(torch.int32),
            "idx_pool_unrot": torch.stack([unrot_v, unrot_p], dim=1).to(torch.int32),
        }


