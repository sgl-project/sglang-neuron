# SPDX-License-Identifier: Apache-2.0
"""Recurrent-state checkpoints on radix nodes: which token boundaries hold exact state, and whether SGLang's radix cache
can keep this model's checkpoints exact.

A request's live state advances with every token, so it can serve no earlier prefix. Only a copy taken at a boundary,
held in a slot the tree owns, is a reusable checkpoint.

Exact boundaries. A forward over tokens [prefix, end) leaves the state at ``end`` in its slot: exact. Inside the
forward the kernel can capture state only on its own tile grid relative to the forward's start (``StateSpec.
track_chunk``), and only when the model's kernel takes the capture control; a boundary off that grid needs another
forward that ends there. Upstream's scheduler encodes an intermediate capture request as ``aligned + 1``
(``mamba_track_seqlens``, its retrieval hint); the checkpoint depth is the aligned length (``aligned_track_length``, the
arithmetic of SGLang's ``ForwardBatch.mamba_track_aligned_lens``, kept here as host integers).

With the chunk-keyed radix tree (sglang_patches/chunk_keyed_tree) and one request per prefill forward on the chunk
grid (sglang_patches/prefill_chunk_grid) every tree page boundary is a forward end, so a final-state copy after each
on-grid forward is an exact checkpoint at every boundary a match can land on, with no in-kernel capture.
"""

from __future__ import annotations

from typing import List, Optional, Sequence


def aligned_track_length(track_seqlen: int, prefix_len: int, chunk: int) -> int:
    """The tracked suffix depth, relative to the forward's prefix, of upstream's tracking hint (``aligned + 1`` for an
    intermediate capture): floor((track - prefix) / chunk) * chunk."""
    if chunk <= 0:
        raise ValueError("tracking needs a positive chunk")
    return ((int(track_seqlen) - int(prefix_len)) // chunk) * chunk


def capture_boundary(prefix_len: int, extend_len: int, track_seqlen: int, track_chunk: int) -> int:
    """The absolute boundary a tracked capture lands on, refusing one the kernel cannot produce exactly."""
    end = prefix_len + extend_len
    if track_seqlen >= end:
        return end  # the final state
    if track_chunk <= 0:
        raise ValueError(f"a capture at {track_seqlen} inside the forward [{prefix_len}, {end}) needs an in-forward "
                         "capture this model's kernel does not provide (track_chunk 0)")
    depth = aligned_track_length(track_seqlen, prefix_len, track_chunk)
    if depth <= 0:
        raise ValueError(f"tracking hint {track_seqlen} aligns to no boundary inside [{prefix_len}, {end})")
    return prefix_len + depth


def radix_problem(spec, *, checkpoint_interval: int, page_size: int, extra_buffer: bool) -> Optional[str]:
    """Why upstream's radix cache cannot keep exact checkpoints of this model's state, or None.

    With the extra (tracking) buffer upstream asks the forward to capture state at every ``checkpoint_interval``
    boundary inside it; this kernel captures only on its ``track_chunk`` grid, so the interval must be a whole number
    of tiles. Without it upstream copies the state a forward leaves (exact at the forward's end) and keys the tree
    per token (page size 1, upstream's own requirement)."""
    if extra_buffer:
        if spec.track_chunk <= 0:
            return (f"{spec.family} state: the radix cache's tracking buffer captures state inside a forward, and "
                    "this model's kernel captures only the state a forward ends with (track_chunk 0); serve without "
                    "the mamba extra buffer")
        if checkpoint_interval <= 0 or checkpoint_interval % spec.track_chunk:
            return (f"{spec.family} state: checkpoint interval {checkpoint_interval} is not a whole number of the "
                    f"kernel's {spec.track_chunk}-token capture tiles; a checkpoint off that grid would not be exact")
        return None
    if page_size != 1:
        return (f"{spec.family} state: without the tracking buffer upstream's radix cache keeps forward-end state "
                f"and needs page size 1 (got {page_size})")
    return None


def chunk_grid_problem(spec, *, chunk: int, cache_chunk: int, extra_buffer: bool) -> Optional[str]:
    """Why the chunk-keyed radix tree (sglang_patches/chunk_keyed_tree: tree page = prefill chunk, one request per
    on-grid prefill forward) cannot keep this model's state, or None.

    On that tree upstream's checkpoint grid is lcm(cache chunk, tree page) = the chunk, and a forward covers exactly one
    chunk from an on-grid prefix, so every boundary upstream's tracking buffer asks for is the end of a forward: the
    state the forward leaves, copied into the tracking slot (``forward_end_tracks``). No in-forward capture is needed,
    whatever the kernel's ``track_chunk``. Upstream's other strategy (no buffer) keeps per-token pages, which a
    chunk-keyed tree does not have."""
    if not extra_buffer:
        return (f"{spec.family} state on the chunk-keyed radix tree: upstream's no-buffer state cache needs a page-1 "
                "tree; launch with the mamba tracking buffer (--mamba-radix-cache-strategy extra_buffer)")
    if chunk <= 0 or cache_chunk <= 0 or chunk % cache_chunk:
        return (f"{spec.family} state on the chunk-keyed radix tree: the prefill chunk {chunk} is not a whole number "
                f"of mamba cache chunks ({cache_chunk}), so upstream's checkpoint grid would not be the chunk and no "
                "one-chunk forward would end on it")
    return None


def forward_end_tracks(prefix_lens: Sequence[int], extend_lens: Sequence[int], track_seqlens: Sequence[int],
                       track_mask: Sequence[bool], track_chunk: int) -> List[tuple]:
    """The tracked captures of one extend batch, as (request index, absolute boundary, at forward end), refusing one
    the kernel cannot produce exactly (capture_boundary). A forward-end capture is a copy of the request's slot into
    its tracking slot after the forward (``NeuronStatePool.capture_tracks``); an inner one is the kernel's own."""
    out = []
    for i, (prefix, extend, track, on) in enumerate(zip(prefix_lens, extend_lens, track_seqlens, track_mask,
                                                        strict=True)):
        if not on:
            continue
        boundary = capture_boundary(int(prefix), int(extend), int(track), track_chunk)
        out.append((i, boundary, boundary == int(prefix) + int(extend)))
    return out
