# SPDX-License-Identifier: Apache-2.0
"""Chunked prefill from a prefix on the paged KV pool (docs/design/recurrent-state.md): the prefill programs
(bucket, prior tier), the cohort's choice of one, and the host tables of a chunk and of a decode step.

A prefilling replica runs one request's chunk per forward: ``n`` tokens at positions ``start .. start + n - 1`` after
``start`` cached tokens. SGLang's chunk grid (``--chunked-prefill-size`` C, sglang_patches.prefill_chunk_grid) and the
chunk-keyed radix tree (``--radix-cache-backend neuron-chunk``) put ``start`` on whole chunks and ``n <= C``; the KV
page PG divides C. The prefill program of a step is keyed by:

- its bucket ``Lb`` (``prefill_buckets``): every chunk but a prompt's last runs C; the last, the smallest bucket that
  holds it, its rows past ``n`` padding (``valid_len``);
- its prior tier (``prior_tiers``): the static prior capacity of the attention (tokens before the chunk it can read),
  0 for a fresh prompt.

Every rank of the world runs the same program (the MoE and the LM head span it), so the bucket and the tier are the
cohort's: the largest token count and the longest cached prefix of the prefilling replicas (``cohort_program``, from
the DP-attention snapshot's ``num_tokens`` and ``max_prefix_tokens``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

# the GDN prefill runs whole 512-token blocks (or one shorter block of 128s); the attention likewise
MIN_BUCKET = 512
# prior tiers (tokens) below the context: a fresh prompt, 16K and 64K; the context itself is the last
DEFAULT_TIERS = (0, 16384, 65536)


def prefill_buckets(chunk: int, page: int = 0) -> tuple:
    """The prefill token buckets of a chunk size: the chunk, and below it a short-last-chunk bucket of ``MIN_BUCKET``
    tokens, or of one KV page when pages are larger (a bucket writes whole pages)."""
    if chunk < 128 or chunk % 128 or (chunk > MIN_BUCKET and chunk % MIN_BUCKET):
        raise ValueError(f"--chunked-prefill-size {chunk}: the giga prefill runs chunks of whole {MIN_BUCKET}-token "
                         "blocks (or one shorter block of whole 128s)")
    low = max(MIN_BUCKET, int(page))
    return (low, chunk) if chunk > low else (chunk,)


def prior_tiers(context: int, page: int, tiers: Sequence[int] = DEFAULT_TIERS) -> tuple:
    """The prior capacities the prefill programs compile: each tier below the context (whole pages), then the context
    rounded up to whole pages (a chunk's prior is shorter than the context)."""
    cap = -(-int(context) // page) * page
    out = sorted({int(t) for t in tiers if 0 <= int(t) < cap})
    if any(t % page for t in out):
        raise ValueError(f"prior tiers {out} are not whole {page}-token pages")
    return tuple(out) + (cap,)


def pick(values: Sequence[int], need: int, what: str) -> int:
    """The smallest of ``values`` that is at least ``need``."""
    for v in sorted(values):
        if v >= need:
            return int(v)
    raise ValueError(f"{what} {need} exceeds the largest compiled one ({max(values)})")


@dataclass(frozen=True)
class PrefillProgram:
    bucket: int  # Lb tokens
    tier: int  # prior capacity, tokens (0: no prior)


def cohort_program(tokens: Sequence[int], prefixes: Sequence[int], buckets: Sequence[int],
                   tiers: Sequence[int]) -> PrefillProgram:
    """The program every rank runs when these replicas prefill (their token counts and cached prefixes)."""
    need_tokens = max([int(t) for t in tokens] + [1])
    need_prior = max([int(p) for p in prefixes] + [0])
    return PrefillProgram(pick(buckets, need_tokens, "a prefill chunk of"),
                          pick(tiers, need_prior, "a cached prefix of"))


@dataclass
class ChunkTables:
    """The host inputs of one chunk (``chunk_tables``)."""

    positions: np.ndarray  # [Lb] int64: start + t (pad rows clamped to the RoPE table)
    write_pages: np.ndarray  # [n_attn, Lb / PG] int32: each attention layer's pool row of each page of the chunk
    prior_pages: Optional[np.ndarray]  # [n_attn, tier / PG] int32 (None: tier 0)
    prior_len: int  # start
    valid_len: int  # n


def chunk_tables(*, pages: Sequence[int], start: int, n: int, program: PrefillProgram, page: int,
                 pages_per_layer: int, attention_layers: int, rope_len: int) -> ChunkTables:
    """The chunk ``[start, start + n)`` of a request whose KV pages (positions ``j * page ..``) are ``pages``, under
    ``program``. Pages of the bucket past the request's allocation, and the prior's padding, are SGLang's reserved page
    0 (finite, never attended: the chunk's pad rows land there or in the request's own last page)."""
    Lb, tier = program.bucket, program.tier
    if start < 0 or start % page:
        raise ValueError(f"a prefill chunk at position {start}: chunks start on whole {page}-token pages (the radix "
                         "tree's and the chunk grid's boundaries)")
    if not 1 <= n <= Lb or Lb % page:
        raise ValueError(f"a chunk of {n} tokens in a {Lb}-token bucket of {page}-token pages")
    if start > tier:
        raise ValueError(f"a chunk after {start} cached tokens in a program whose prior holds {tier}")
    pages = [int(p) for p in pages]
    need = -(-(start + n) // page)
    if len(pages) < need:
        raise ValueError(f"the request holds {len(pages)} KV pages; positions 0 .. {start + n - 1} need {need}")
    if any(p <= 0 or p >= pages_per_layer for p in pages[:need]):
        raise ValueError(f"the request's KV pages {pages[:need]} are outside the pool's pages 1 .. "
                         f"{pages_per_layer - 1}")
    first = start // page
    write = [pages[first + j] if first + j < need else 0 for j in range(Lb // page)]
    base = (np.arange(attention_layers, dtype=np.int64) * pages_per_layer)[:, None]
    write_pages = (base + np.asarray(write, np.int64)[None, :]).astype(np.int32)
    prior_pages = None
    if tier:
        prior = [pages[j] if j < first else 0 for j in range(tier // page)]
        prior_pages = (base + np.asarray(prior, np.int64)[None, :]).astype(np.int32)
    positions = np.minimum(start + np.arange(Lb, dtype=np.int64), rope_len - 1)
    return ChunkTables(positions=positions, write_pages=write_pages, prior_pages=prior_pages, prior_len=int(start),
                       valid_len=int(n))


def request_pages(req_to_token_row, length: int, page: int) -> list:
    """The KV pages of a request's positions ``0 .. length - 1`` from its row of SGLang's req_to_token table (page
    ``j`` = the slot of position ``j * page``, divided by the page), checking each page is whole and in order."""
    row = np.asarray(req_to_token_row[:length], dtype=np.int64)
    pages = row[::page] // page
    want = (pages[np.arange(length) // page] * page + np.arange(length) % page)
    if not np.array_equal(row, want):
        bad = int(np.flatnonzero(row != want)[0])
        raise RuntimeError(f"position {bad} of the request is at cache slot {int(row[bad])}, not in its page's order "
                           f"(the paged attention needs SGLang's paged allocator with --page-size {page})")
    return pages.tolist()


def decode_block_table(rows_pages: Sequence[Sequence[int]], rows: int, pages_per_row: int) -> np.ndarray:
    """``[rows, MP]`` int32: each decode row's pages, padded with page 0; padding rows read page 0 only."""
    bt = np.zeros((rows, pages_per_row), np.int32)
    for r, pages in enumerate(rows_pages):
        if len(pages) > pages_per_row:
            raise ValueError(f"decode row {r} holds {len(pages)} KV pages; the block table has {pages_per_row}")
        bt[r, :len(pages)] = pages
    return bt


def checkpoint_slot_bytes(dims, layout) -> int:
    """Bytes per rank of one GDN state slot (one radix checkpoint): every GDN layer's recurrent state (fp32) and conv
    window (bf16)."""
    r = layout.replica
    vh = dims.num_v_heads // r
    kh = max(1, (dims.num_k_heads * vh) // dims.num_v_heads)
    conv = 2 * kh * dims.head_k_dim + vh * dims.head_v_dim
    L_g = len(dims.gdn_layers)
    return L_g * (dims.head_k_dim * vh * dims.head_v_dim * 4 + conv * (dims.conv_kernel - 1) * 2)


def checkpoint_budget(dims, layout, prompt_tokens: int, chunk: int) -> dict:
    """Per rank: the radix checkpoints a cached prompt of ``prompt_tokens`` holds on the chunk-keyed tree (one per whole
    chunk), their bytes, and its KV bytes (sizing --max-mamba-cache-size and the KV pool)."""
    slots = int(prompt_tokens) // int(chunk)
    kv = int(prompt_tokens) * len(dims.attention_layers) * 2 * dims.head_dim * 2
    return {"checkpoints": slots, "checkpoint_bytes": slots * checkpoint_slot_bytes(dims, layout), "kv_bytes": kv}
