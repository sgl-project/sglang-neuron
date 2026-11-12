# SPDX-License-Identifier: Apache-2.0
"""LingBot-World's paged K/V history: one pool per DiT rank and one session's read plans.

The DiT reads its attention history from per-layer pool views through a per-head block table
and mask bounds, and its commit pass writes the chunk's K/V through ``kv_cache_update_idx``
(``attention/paged_attention.py``). This module owns the pool and the host-side plan of one
session: one latent frame per block, the 9-frame sink, the 9-frame recent window (the current
chunk included), freed blocks reused first, and the fixed-width table of the read plan
(``num_sink_slots`` sink + ``num_recent_slots`` recent history entries, then the current
chunk's blocks). The paging is vLLM-Omni's AR-Diffusion paging for LingBot-World
(``chunk_window_skipped_tokens`` and ``chunk_slot_mapping`` below, copied from
``vllm_omni.experimental.ar_diffusion.kv_cache.paged``), so the DiT reads the same history.
"""

from __future__ import annotations

import collections

import msgspec
import torch

from sglang_neuron.diffusion.attention.paged_attention import (
    commit_block_aligned,
    key_pool_shape,
)

NULL_BLOCK = 0
#: LingBot-World-V2's paging, in latent frames: the attention sink and the whole attention window
#: (the sink, then the recent window with the current chunk included).
SINK_FRAMES = 9
WINDOW_FRAMES = 18


def chunk_window_skipped_tokens(
    num_computed_tokens: int, *, chunk_size: int, sliding_window: int, sink_chunks: int
) -> int:
    """Tokens evicted after the sink, snapped down to a window unit (``chunk_size``)."""
    skipped = max(0, num_computed_tokens - sliding_window - sink_chunks * chunk_size)
    return (skipped // chunk_size) * chunk_size


def chunk_slot_mapping(
    block_ids: list[int], num_computed_tokens: int, chunk_size: int, block_size: int
) -> torch.Tensor:
    """Slots ``block * block_size + offset`` of positions ``[start, start + chunk_size)``."""
    positions = torch.arange(num_computed_tokens, num_computed_tokens + chunk_size)
    blocks = torch.tensor(block_ids, dtype=torch.long)[positions // block_size]
    return blocks * block_size + positions % block_size


def per_head_block_table(table: list[int], heads: int) -> torch.Tensor:
    """``[N, W]`` int32: entry ``b`` of head ``h`` is ``b * N + h`` (the folded pool rows)."""
    base = torch.tensor(table, dtype=torch.int64).reshape(1, -1)
    return (base * heads + torch.arange(heads).reshape(-1, 1)).to(torch.int32).contiguous()


def fold_slots_per_head(slots: torch.Tensor, heads: int, block: int) -> torch.Tensor:
    """Slots ``block_id * block + offset`` -> ``[N, S]`` int32 rows of the per-head pool
    (``(block_id * N + h) * block + offset``: the pool viewed as ``[num_blocks * N * block, D]``)."""
    slots = slots.to(torch.int64).reshape(1, -1)
    head = torch.arange(heads).reshape(-1, 1)
    return (((slots // block) * heads + head) * block + slots % block).to(torch.int32).contiguous()


class PagingGeometry(msgspec.Struct, frozen=True):
    """The paging of one resolution, in tokens and blocks (a block is one latent frame)."""

    tokens_per_frame: int
    frames_per_chunk: int

    @property
    def block(self) -> int:
        return self.tokens_per_frame

    @property
    def chunk_tokens(self) -> int:
        return self.frames_per_chunk * self.tokens_per_frame

    @property
    def sink_tokens(self) -> int:
        return SINK_FRAMES * self.tokens_per_frame

    @property
    def window_tokens(self) -> int:
        """The recent window after the sink, the current chunk included."""
        return (WINDOW_FRAMES - SINK_FRAMES) * self.tokens_per_frame

    @property
    def num_sink_slots(self) -> int:
        return -(-self.sink_tokens // self.block)

    @property
    def num_recent_slots(self) -> int:
        return -(-(self.window_tokens - self.chunk_tokens) // self.block)

    @property
    def num_current_slots(self) -> int:
        return -(-self.chunk_tokens // self.block)

    @property
    def num_history_blocks(self) -> int:
        return self.num_sink_slots + self.num_recent_slots

    @property
    def blocks_per_session(self) -> int:
        """Blocks a session holds at most (sink + recent + current), plus the null block."""
        return 1 + self.num_history_blocks + self.num_current_slots

    def skipped_tokens(self, history: int) -> int:
        """The evicted gap after the sink once a chunk over ``history`` tokens is computed."""
        return chunk_window_skipped_tokens(
            history + self.chunk_tokens,
            chunk_size=self.tokens_per_frame,
            sliding_window=self.window_tokens,
            sink_chunks=SINK_FRAMES,
        )


class PagedPass(msgspec.Struct, frozen=True):
    """One DiT pass's paged inputs (the fields the DiT runtime and model read)."""

    block_table: torch.Tensor  # [N, W] int32 per-head ids (block * N + head)
    bounds: torch.Tensor  # [1, 3] int32 (sink_end, window_start, kv_len)
    kv_cache_update_idx: torch.Tensor | None  # [N, S] int32, commit pass only
    num_history_blocks: int
    history_full: bool
    host_bounds: tuple[int, int, int]  # bounds as host ints (the runtime's plan check)
    # The commit covers whole blocks inside the pool (the runtime's check; True off a commit).
    commit_block_aligned: bool

    @property
    def commit(self) -> bool:
        return self.kv_cache_update_idx is not None


class PagedPool:
    """Every layer's K/V pool on one rank: one zeroed backing buffer sliced into per-layer K
    (``key_pool_shape``) and V ``[num_blocks, N, block, D]`` views; block 0 is the null block."""

    def __init__(self, *, num_blocks, num_layers, heads, head_dim, block, device, dtype):
        self.num_blocks = num_blocks
        value_shape = (num_blocks, heads, block, head_dim)
        key_shape = key_pool_shape(num_blocks, heads, block, head_dim)
        per_tensor = num_blocks * heads * block * head_dim
        self.backing = torch.zeros((num_layers, 2, per_tensor), dtype=dtype, device=device)
        self.layers = [
            (self.backing[i, 0].view(key_shape), self.backing[i, 1].view(value_shape))
            for i in range(num_layers)
        ]

    @property
    def nbytes(self) -> int:
        return self.backing.numel() * self.backing.element_size()

    def flat(self) -> tuple[torch.Tensor, ...]:
        """``(k_0, v_0, k_1, v_1, ...)``: the views the DiT runtime binds."""
        return tuple(t for kv in self.layers for t in kv)


class PagedSession:
    """One session's frame blocks and per-pass read plans (host side; one session per pool)."""

    def __init__(self, geometry: PagingGeometry, *, num_blocks: int, heads: int, device):
        if num_blocks < geometry.blocks_per_session:
            raise ValueError(
                f"a session needs {geometry.blocks_per_session} pool blocks, the pool has "
                f"{num_blocks}"
            )
        self.geometry = geometry
        self.heads = heads
        self.device = device
        self.num_slots = num_blocks * heads * geometry.block  # the pool's per-layer token slots
        self.free = collections.deque(range(1, num_blocks))
        self.blocks: dict[int, int] = {}  # logical frame block -> physical block
        self.history = 0  # committed tokens
        self.current: list[int] | None = None

    def _plan(self) -> tuple[list[int], tuple[int, int, int], bool]:
        g, history = self.geometry, self.history
        block = g.block
        skipped = g.skipped_tokens(history)
        history_blocks = -(-history // block)
        if skipped == 0:  # filling: the whole history is visible
            listed = [self.blocks[j] for j in range(history_blocks)]
            bounds = (history, history, history)
        else:
            window_start = g.sink_tokens + skipped
            first_recent = window_start // block
            sink = [self.blocks[j] for j in range(min(g.num_sink_slots, history_blocks))]
            recent = [self.blocks[j] for j in range(first_recent, history_blocks)]
            if len(recent) > g.num_recent_slots:
                raise RuntimeError(f"{len(recent)} recent blocks > {g.num_recent_slots} slots")
            listed = sink + [NULL_BLOCK] * (g.num_sink_slots - len(sink)) + recent
            base = g.num_sink_slots * block - first_recent * block
            bounds = (min(g.sink_tokens, history), base + window_start, base + history)
        listed = listed + [NULL_BLOCK] * (g.num_history_blocks - len(listed))
        total = g.num_history_blocks * block
        full = bounds[0] >= total or (bounds[1] <= bounds[0] and bounds[2] >= total)
        return listed, bounds, full

    def paged_for(self, commit: bool) -> PagedPass:
        """The next pass's inputs (the chunk's current blocks are allocated on its first pass)."""
        g = self.geometry
        if self.current is None:
            self.current = [self.free.popleft() for _ in range(g.num_current_slots)]
        listed, bounds, full = self._plan()
        idx = None
        aligned = True
        if commit:
            first = self.history // g.block
            table = [self.blocks.get(j, NULL_BLOCK) for j in range(first)] + self.current
            slots = chunk_slot_mapping(table, self.history, g.chunk_tokens, g.block)
            idx = fold_slots_per_head(slots, self.heads, g.block)
            # On the host, before the upload; the kernel's block writes have no range guard.
            aligned = commit_block_aligned(idx, g.block, self.num_slots)
            idx = idx.to(self.device)
        return PagedPass(
            block_table=per_head_block_table(listed + self.current, self.heads).to(self.device),
            bounds=torch.tensor([bounds], dtype=torch.int32).to(self.device),
            kv_cache_update_idx=idx,
            num_history_blocks=g.num_history_blocks,
            history_full=full,
            host_bounds=tuple(int(b) for b in bounds),
            commit_block_aligned=aligned,
        )

    def plan_host(self, commit: bool) -> tuple[int, int, int]:
        """The next pass's bounds, host only (no upload): the session's bookkeeping when its
        plans were prefetched (the chunk's current blocks are allocated on its first pass)."""
        del commit  # a chunk's denoise and commit passes read the same history
        if self.current is None:
            self.current = [self.free.popleft() for _ in range(self.geometry.num_current_slots)]
        _, bounds, _ = self._plan()
        return tuple(int(b) for b in bounds)

    def commit(self) -> None:
        """The chunk's blocks join the history; blocks no later chunk can see are freed."""
        g = self.geometry
        first = self.history // g.block
        for j, b in enumerate(self.current):
            self.blocks[first + j] = b
        self.current = None
        self.history += g.chunk_tokens
        next_window_start = g.sink_tokens + g.skipped_tokens(self.history)
        for j in sorted(self.blocks):
            if j >= g.num_sink_slots and (j + 1) * g.block <= next_window_start:
                self.free.appendleft(self.blocks.pop(j))  # reused first
