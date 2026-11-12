# SPDX-License-Identifier: Apache-2.0
"""Radix-cache backend ``neuron-chunk``: SGLang's unified radix tree with the prefill chunk as its page
(sglang_patches.chunk_keyed_tree has the launch side and the trade against the default tree).

On the prefill chunk grid (sglang_patches.prefill_chunk_grid) a cold prompt runs the forwards [0, C), [C, 2C), ...
With C as the tree page, ``RadixKey.page_aligned`` floors every match, insert and dedup to whole chunks, so the tree
holds only KV written by whole on-grid forwards and a hit at kC replays the cold prompt's own forwards from kC, over the
same KV bits. The allocator keeps its own KV page. A PD decode keys its own per-rank ``--chunked-prefill-size`` and
pairs only with prefills whose chunk divides it (sglang_patches.decode_radix_prefill_chunks).

A hybrid SSM model (a recurrent state per request beside its KV) is keyed too when its prefill graph writes the state
each forward ends with into the tracking slot SGLang's batch names (``ModelSpec.chunk_state_checkpoints``, upstream's
mamba extra buffer). Every forward then ends on a chunk boundary, so upstream's mamba component donates an exact
checkpoint for every chunk the tree keys, frees it with its node, and copies it into a hitting request's slot; the hit
restarts from it over the shared KV pages as the cold prompt's own next forward did.

Inserts stop at the prompt's last whole chunk (``insert_cap``): upstream's finish insert also takes the outputs, and a
chunk they complete would hold KV from a partial forward and decode steps, not from one cold forward. Every insert
reaches ``checkpoint_into_tree`` (through ``UnifiedRadixCache.checkpoint`` or the streaming session). Upstream sources
are md5-pinned against SGLang 43881a7 (``_SOURCES_MD5``) and checked before the backend registers."""

from __future__ import annotations

import dataclasses
import logging

from sglang.srt.mem_cache.registry import create_unified_radix_cache, register_radix_cache_backend
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

from sglang_neuron import serving_policy
from sglang_neuron.sglang_patches import decode_radix_prefill_chunks, prefill_chunk_grid, source_guard
from sglang_neuron.sglang_patches.chunk_keyed_tree import BACKEND

logger = logging.getLogger(__name__)

# The factory relies on create_unified_radix_cache using params.page_size as the tree page and, off HiCache, host-pool
# retraction and the external linker, attaching nothing else. The subclass relies on checkpoint_into_tree inserting
# [cache_protected_len, up_to), floored to the tree page, and leaving the rest to the release that follows.
_SOURCES_MD5 = {
    ("sglang.srt.mem_cache.registry", "create_unified_radix_cache"): "552f1ddd17921e92404871131f77c5b7",
    ("sglang.srt.mem_cache.unified_radix_cache", "UnifiedRadixCache.checkpoint_into_tree"):
        "4082530bb41b73ce76ed16c2e1531de1",
    # every insert, on release and at chunked checkpoints, reaches checkpoint_into_tree through it
    ("sglang.srt.mem_cache.unified_radix_cache", "UnifiedRadixCache.checkpoint"): "53beecb5431a8eff61c3b55894579d80",
    ("sglang.srt.mem_cache.common", "release_kv_cache"): "288de87ca4a5923c41b5dd0557935694",
    ("sglang.srt.mem_cache.common", "checkpoint_kv_cache"): "5449f01fe3e2e37a7d63c29de1b48623",
}


def insert_cap(tree, req, up_to: int) -> int:
    """Where ``req``'s insert into ``tree`` ends: its prompt's last whole chunk, so no partial chunk or decode KV enters
    the tree. Never below ``cache_protected_len``, which can pass that chunk (a retracted request matching its outputs,
    or a match on another request's longer prompt): upstream asserts an insert covers it and frees only past it."""
    page = int(tree.page_size)
    return min(int(up_to), max(len(req.origin_input_ids) // page * page, int(req.kv.cache_protected_len)))


class ChunkKeyedRadixCache(UnifiedRadixCache):
    """SGLang's unified radix tree with the prefill chunk as its page. Inserts stop at ``insert_cap``; the KV past it
    is released like any KV not inserted."""

    def checkpoint_into_tree(self, req, *, up_to: int, **kwargs) -> None:
        super().checkpoint_into_tree(req, up_to=insert_cap(self, req, up_to), **kwargs)


def build(ctx):
    """The backend factory: SGLang's unified radix tree for ``ctx`` (a TreeCacheBuildContext) keyed on this
    scheduler's prefill chunk, or on a PD decode its own per-rank chunk. Refuses a disabled radix cache, a pool other
    than plain KV (or an SSM pool of a model that keeps chunk state checkpoints), a scheduler off the prefill chunk
    grid, and a chunk that is not whole KV pages."""
    # fail closed: sglang_patches.chunk_keyed_tree.validate_launch and the platform already refuse these launches
    if ctx.disable_radix_cache:
        raise ValueError(f"--radix-cache-backend {BACKEND} with the radix cache disabled: the backend builds a tree, "
                         "and the launch asks for none")
    # a model whose prefill graph writes its recurrent state into SGLang's tracking slot at every forward's end
    # (ModelSpec.chunk_state_checkpoints) has an exact checkpoint at every chunk boundary, so its SSM pool is keyed too
    ssm_unkept = ctx.is_hybrid_ssm and not serving_policy.current().chunk_state_checkpoints
    pools = [name for name, on in (("a hybrid SWA pool", ctx.is_hybrid_swa), ("an SSM pool", ssm_unkept),
                                   ("a DSA pool", ctx.is_dsa)) if on]
    if pools:
        raise ValueError(f"--radix-cache-backend {BACKEND} keys a plain KV pool only, not {' or '.join(pools)}"
                         + (" of a model whose prefill keeps no state checkpoints at chunk boundaries" if ssm_unkept
                            else ""))
    chunk = (decode_radix_prefill_chunks.tree_chunk() if serving_policy.current().decode_radix_prefill_chunks
             else prefill_chunk_grid.grid_unit())
    if chunk is None:
        raise ValueError(f"--radix-cache-backend {BACKEND} on a scheduler that does not keep the prefill chunk grid: "
                         "the tree keys the chunks every prefill forward is cut to, and this scheduler cuts none")
    kv_page = int(ctx.params.token_to_kv_pool_allocator.page_size)
    if chunk % kv_page:
        raise ValueError(f"--radix-cache-backend {BACKEND}: the chunk ({chunk} tokens) is not a whole number of KV "
                         f"pages of {kv_page}; launch with a --chunked-prefill-size that is a multiple of {kv_page}")
    logger.info("[neuron] the radix tree keys whole prefill chunks of %d tokens (KV page %d)", chunk, kv_page)
    # copies, so SGLang's own context and params stay untouched
    params = dataclasses.replace(ctx.params, page_size=chunk)
    return create_unified_radix_cache(dataclasses.replace(ctx, params=params), cache_class=ChunkKeyedRadixCache)


def register_backend() -> None:
    """Check the pins and register ``build`` as backend ``neuron-chunk`` (once per process: SGLang refuses a second
    registration)."""
    source_guard.check_targets(_SOURCES_MD5, __name__)
    register_radix_cache_backend(BACKEND, build)
