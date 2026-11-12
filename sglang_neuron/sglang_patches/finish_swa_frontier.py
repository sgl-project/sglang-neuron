# SPDX-License-Identifier: Apache-2.0
"""Move a finished request's sliding-window frontier before its tree insert, as upstream does for an unfinished one.

On a hybrid SWA pool under the radix cache (mem_cache.hybrid_swa's radix mode, SGLANG_NEURON_HYBRID_SWA_RADIX=1),
``UnifiedRadixCache.checkpoint_into_tree`` frees the window-pool pages behind the request's window before inserting it,
so a node keeps about one window of live window KV. Upstream skips that for a finished request, so one that finishes in
its prefill (one output token, a stop on the first token, an abort mid-prefill) inserts its whole prompt as live window
KV, and a later request matching inside it locks all of that while it runs, well past the window pool's per-request
sizing (hybrid_swa.window_demand).

Here ``checkpoint_into_tree`` takes that branch for finished requests too (``UPSTREAM_GATE`` -> ``NEURON_GATE``). That
is safe: the allocator frees a window slot through the full-to-window mapping, which a freed slot no longer holds, so
the release after the insert frees each slot once.

Target md5-pinned against upstream SGLang 43881a7: ``UnifiedRadixCache.checkpoint_into_tree``; relied on:
``SWAComponent.free_out_of_window_slots`` (the frees, and the insert's tombstone bound) and
``SWATokenToKVPoolAllocator.free_swa`` (frees through the mapping), pinned in mem_cache.hybrid_swa.

Upstream: a one-line fix SGLang can take as is."""

from __future__ import annotations

TARGET = ("sglang.srt.mem_cache.unified_radix_cache", "UnifiedRadixCache.checkpoint_into_tree")
_TARGETS_MD5 = {
    TARGET: "4082530bb41b73ce76ed16c2e1531de1",
    ("sglang.srt.mem_cache.unified_cache.components.swa", "SWAComponent.free_out_of_window_slots"):
        "27e295e74ea1e92f7b11fa564c679027",
}
# (the method's source dedented, as the rebuild reads it)
UPSTREAM_GATE = """    if (
        not is_finished
        and envs.SGLANG_OPT_UNIFIED_CACHE_FREE_OUT_OF_WINDOW_SLOTS.get()
    ):
"""
NEURON_GATE = """    if envs.SGLANG_OPT_UNIFIED_CACHE_FREE_OUT_OF_WINDOW_SLOTS.get():
"""


def checkpoint_into_tree():
    """Upstream's UnifiedRadixCache.checkpoint_into_tree, with the out-of-window frees for finished requests too."""
    from sglang_neuron.sglang_patches import source_guard

    return source_guard.with_substitutions(source_guard.resolve(*TARGET), [(UPSTREAM_GATE, NEURON_GATE)])


def register(registry, replace) -> None:
    from sglang_neuron.sglang_patches import source_guard

    source_guard.check_targets(_TARGETS_MD5, __name__)
    registry.register(".".join(TARGET), checkpoint_into_tree(), replace)
