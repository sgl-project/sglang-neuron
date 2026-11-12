# SPDX-License-Identifier: Apache-2.0
"""Hybrid SWA KV memory: SGLang's two-pool layout for a model's sliding-window layers, served unless the launch passes
``--disable-hybrid-swa-memory``.

SGLang keeps the sliding layers' KV in a window pool (``SWAKVPool.swa_kv_pool``) that holds each running request's
window and frees the pages behind it (``ScheduleBatch.maybe_evict_swa``), and the full-attention layers' KV in
``full_kv_pool``. For GPT-OSS at 8K tokens a request needs about half the KV. Both pools and both allocators are the
platform's classes, driven in lock step by SGLang's ``SWATokenToKVPoolAllocator``; ``SWAKVPool.get_kv_buffer`` hands a
sliding layer the window pool's buffer. The plugin adds:

* :func:`configure`: SGLang's default where it is served, one pool everywhere else;
* :func:`check_window_pool`: a step-0 proof that the window pool holds the most the running requests can keep in it,
  so none is retracted for want of window pages;
* :func:`window_blocks`, :func:`window_slots`: the scheduler addresses KV by full-pool slot, so the graph runner
  translates a sliding layer's block table and slots into window-pool pages (``full_to_swa_index_mapping``), and
  :func:`check_window_resident` checks a cached prefix's window prior is there;
* on the radix cache, a finished request's window frees before its insert (sglang_patches.finish_swa_frontier).

Under a radix tree it is opt-in (``SGLANG_NEURON_HYBRID_SWA_RADIX=1``) because it changes what hits: window KV
survives only at the tree's checkpoints (chunk-boundary inserts, a finished request's last window), so a shared prefix
that one pool hits at its page floor hits at the last checkpoint before it, often none.
"""

from __future__ import annotations

import logging
import math
import os

import torch

logger = logging.getLogger(__name__)

# opt-in hybrid SWA memory under a radix tree, whose prefixes hit only at the tree's window checkpoints
RADIX_MODE = "SGLANG_NEURON_HYBRID_SWA_RADIX"

# SGLang sources this layout relies on, md5-pinned against SGLang 43881a7 (source_guard):
# * the pool: built over the platform's MHA pool, each layer routed to its sub-pool and translated through the
#   allocator's mapping (SWAKVPool.__init__, get_kv_buffer, translate_loc_from_full_to_swa);
# * the allocator: both sub-allocators give the same pages for the same lengths, so a full page maps to one window page
#   at the same in-page offset (alloc_extend, alloc_decode, set_full_to_swa_mapping), and frees reset the mapping to 0
#   (free_swa, free_swa_segment, _free_swa_pages and its host path, clear_full_to_swa_mapping): window_blocks
#   translates a page by its first slot, and an evicted page to 0;
# * the window frees: a window behind the position on the chunk cache, whole pages of max(window, page) under a tree
#   (retire_window, free_swa_out_of_window_slots), which window_demand counts;
# * the sizing: on the chunk cache with --max-running-requests the window pool is SGLang's request cap
#   (create_memory_pool_configurator, SWARequestCapPoolConfigurator, compute_swa_request_cap), which window_demand
#   restates.
_CONFIGURATOR = "sglang.srt.mem_cache.kv_cache_configurator"
_SIZING = "sglang.srt.model_executor.pool_configurator"
_ALLOCATOR = "sglang.srt.mem_cache.allocator.swa"
_POOL = "sglang.srt.mem_cache.swa_memory_pool"
_SOURCES_MD5 = {
    (_CONFIGURATOR, "KVCacheConfigurator._build_hybrid_swa_kv_pool"): "a1130a72185ec8f9927c501c621542c2",
    (_POOL, "SWAKVPool.__init__"): "2e4441625c80725a44efc89d7bbcd8fa",
    (_POOL, "SWAKVPool.get_kv_buffer"): "4449de8ece40ba07f3e5ad6cacb20639",
    (_POOL, "SWAKVPool.translate_loc_from_full_to_swa"): "7bfb0137cd0219c5ce9175e31d33f8d9",
    (_ALLOCATOR, "SWATokenToKVPoolAllocator.__init__"): "bbb30254d6876a76a26736bd16ca1117",
    (_ALLOCATOR, "SWATokenToKVPoolAllocator.alloc_extend"): "de7d7c8366217d157ce14317aa9d1ea2",
    (_ALLOCATOR, "SWATokenToKVPoolAllocator.alloc_decode"): "d0fab9e769202ab4aeb002f662b7c377",
    (_ALLOCATOR, "SWATokenToKVPoolAllocator.set_full_to_swa_mapping"): "645dd1532ace9efdefc0526eb91e8958",
    (_ALLOCATOR, "SWATokenToKVPoolAllocator.free_swa"): "080755a1f37a8077d3be55c4ff74bad5",
    (_ALLOCATOR, "SWATokenToKVPoolAllocator.free_swa_segment"): "e3aa7fdecdad4a2c2aaa3410a1d66134",
    (_ALLOCATOR, "SWATokenToKVPoolAllocator._free_swa_pages"): "e7837c9d40e1685e787761129dc138cf",
    (_ALLOCATOR, "SWATokenToKVPoolAllocator._free_swa_pages_none_cuda"): "34c28199df328a3e8ece2c37765fb1d5",
    (_ALLOCATOR, "SWATokenToKVPoolAllocator.clear_full_to_swa_mapping"): "0409e2abf738ca4f85745ca96e3e39c0",
    ("sglang.srt.mem_cache.allocation", "retire_window"): "ab4956a4e8601a68694c9709553c580a",
    ("sglang.srt.mem_cache.common", "free_swa_out_of_window_slots"): "d6bf67e07395bbe37a556edadae83c9b",
    (_SIZING, "create_memory_pool_configurator"): "2b934a749398e1fe464c95ee4099df08",
    (_SIZING, "SWARequestCapPoolConfigurator.__init__"): "49ad9735b956f4b6e0a1ac3a4c2e99cb",
    (_SIZING, "SWARequestCapPoolConfigurator.is_applicable"): "b13b8006be74527826a284f043ce18f9",
    (_SIZING, "compute_swa_request_cap"): "608f658e2ce49541b33ce2ad66f7bb60",
}


def radix_mode(environ=os.environ) -> bool:
    """Whether hybrid SWA memory runs under the radix tree (RADIX_MODE). Only ``0`` and ``1`` are accepted, so ``true``
    or ``yes`` cannot silently leave it off."""
    value = environ.get(RADIX_MODE, "0").strip()
    if value not in ("0", "1"):
        raise ValueError(f"{RADIX_MODE}={environ[RADIX_MODE]!r}: set 1 for hybrid SWA memory under the radix cache, "
                         "whose prefixes then hit only at the tree's window checkpoints, or 0 (the default)")
    return value == "1"


def unserved(server_args, architecture: str, spec, policy) -> list:
    """Why the launch is not served hybrid SWA memory; empty when it is.

    Served on an aggregated server whose model reads sliding layers through the translated tables
    (``catalog.ModelSpec.hybrid_swa_memory``), with chunked prefill: on the chunk cache, pipelined or not, and on
    SGLang's default radix tree in RADIX_MODE without PP. Pipelined stages size their pools alike
    (sglang_patches.pp_pool_resolve) and bind layers by global id (``NeuronModel.bind_kv_cache``), which needs every
    stage to hold both kinds of layer (``stage_layout_reasons``)."""
    from sglang.srt.environ import envs

    reasons = []
    if not spec.hybrid_swa_memory:
        reasons.append(f"{architecture}'s graphs, which read no window pool")
    radix = not getattr(server_args, "disable_radix_cache", False)
    if spec.hybrid_swa_memory and policy.pp_size > 1 and policy.role == "aggregated" and not radix:
        # only where PP is served (chunk cache); a model without hybrid SWA memory need not declare layer kinds
        reasons += stage_layout_reasons(server_args, architecture, policy.pp_size)
    if policy.role != "aggregated":
        # (DCP, served on a PD decode only, is refused with the PD roles)
        reasons.append(f"a {policy.role} server: a PD pair would send each request's window pages through SGLang's "
                       "window-state transfer, which slices a page between attention TPs per token, as if it were "
                       "token-major, while the Neuron pool's pages are head-major (sglang_patches.nixl_head_major "
                       "adapts the KV pages only)")
    # None: SGLang sets a positive chunk after the platform's defaults (handle_gpu_memory_settings)
    chunk = getattr(server_args, "chunked_prefill_size", None)
    if chunk is not None and chunk <= 0:
        reasons.append("no chunked prefill: an unchunked prompt keeps all of its window KV until its prefill ends")
    from sglang_neuron.sglang_patches import chunk_keyed_tree

    if radix and not radix_mode():
        reasons.append(f"the radix cache: SGLang's window tree hits a shared prefix only at the checkpoints whose window "
                       f"KV it keeps, where one pool hits at its page floor ({RADIX_MODE}=1 serves it so)")
    if radix and policy.pp_size > 1:
        reasons.append(f"--pp-size {policy.pp_size} with the radix cache: no run covers hybrid SWA memory under a "
                       "pipelined server's tree")
    if getattr(server_args, "enable_hierarchical_cache", False):
        reasons.append("--enable-hierarchical-cache: the Neuron host pool (mem_cache.host_pool) mirrors one KV pool")
    if chunk_keyed_tree.selected(server_args):
        reasons.append(f"--radix-cache-backend {chunk_keyed_tree.BACKEND}, whose tree keys a plain KV pool only "
                       "(mem_cache.chunk_keyed_tree)")
    if radix and not envs.SGLANG_OPT_UNIFIED_CACHE_FREE_OUT_OF_WINDOW_SLOTS.get():
        reasons.append("the radix cache with SGLANG_OPT_UNIFIED_CACHE_FREE_OUT_OF_WINDOW_SLOTS off: the tree would "
                       "keep a chunked prefill's whole window KV, which the window pool's sizing does not count")
    return reasons


def stage_layout_reasons(server_args, architecture: str, pp_size: int) -> list:
    """Why a pipelined launch's stages cannot hold hybrid SWA pools alike; empty when every stage holds a sliding and a
    full-attention layer.

    SGLang sizes a stage's pools by its layer kinds: a sliding-only stage takes the all-window branch, a stage with no
    sliding layer fails ``HybridSWAPoolConfigurator``'s assert, and only a stage with a full layer takes the request cap
    (``SWARequestCapPoolConfigurator.is_applicable``). Mismatched pool sizes would keep the stages from scheduling the
    same batches. The split is SGLang's ``get_pp_indices`` (honours SGLANG_PP_LAYER_PARTITION); the kinds come from the
    checkpoint's ``layer_types`` (models.contract)."""
    from types import SimpleNamespace

    from sglang.srt.distributed.utils import get_pp_indices

    from sglang_neuron.models import catalog, contract

    kinds = contract.derive_layer_kinds(SimpleNamespace(**catalog.checkpoint_config(server_args)), architecture)
    reasons = []
    for rank in range(pp_size):
        start, end = get_pp_indices(len(kinds), rank, pp_size)
        sliding = sum(kind.sliding_window is not None for kind in kinds[start:end])
        missing = [name for name, count in (("sliding-window", sliding), ("full-attention", end - start - sliding))
                   if count == 0]
        if missing:
            kinds_missing = " and no ".join(missing)
            reasons.append(f"pipeline stage {rank} of layers [{start}, {end}), which holds no {kinds_missing} layer: "
                           "SGLang sizes a stage's pools by the kinds of layer it holds, so the stages would not hold "
                           "the same full and window counts (set SGLANG_PP_LAYER_PARTITION so that every stage holds "
                           "both)")
    return reasons


def configure(server_args, architecture: str, spec, policy) -> None:
    """Resolve SGLang's ``disable_hybrid_swa_memory``: one pool if the launch passes the flag or is not served
    (``unserved``), else the hybrid pool with the pins checked. RADIX_MODE is an explicit request, so it is refused
    where not served, naming every reason."""
    if radix_mode():
        asked = ["--disable-hybrid-swa-memory, which asks for one pool"] \
            if getattr(server_args, "disable_hybrid_swa_memory", False) else []
        if getattr(server_args, "disable_radix_cache", False):
            asked.append("--disable-radix-cache: the mode is hybrid SWA memory under a radix tree")
        asked += unserved(server_args, architecture, spec, policy)
        if asked:
            raise ValueError(f"{RADIX_MODE}=1 is not served with {'; '.join(asked)}")
    if getattr(server_args, "disable_hybrid_swa_memory", False):
        return
    reasons = unserved(server_args, architecture, spec, policy)
    if reasons:
        if spec.hybrid_swa_memory:
            logger.info("[neuron] one KV pool for every layer: hybrid SWA memory is not served with %s",
                        "; ".join(reasons))
        server_args.disable_hybrid_swa_memory = True
        return
    from sglang_neuron.sglang_patches import source_guard

    source_guard.check_targets(_SOURCES_MD5, __name__)
    server_args.disable_hybrid_swa_memory = False


def window_demand(*, window: int, page: int, requests: int, interval: int, chunk: int, chunks_in_flight: int,
                  tree: bool) -> int:
    """The most window-pool tokens ``requests`` running requests of one attention-DP worker hold at once.

    Per request: what its frees keep behind its position (``kept``: the window on the chunk cache, whole pages of
    ``max(window, page)`` under a tree), the eviction interval, a page the free rounds down and the page the next decode
    step takes. Under a tree it also locks the matched checkpoint's window: up to ``kept`` plus the page the insert
    rounds to. Plus a chunk per prefill chunk in flight and the reserved page 0. On the chunk cache this is SGLang's
    request cap (compute_swa_request_cap; under PP with sglang_patches.pp_pool_resolve's chunk count)."""
    kept = -(-max(window, page) // page) * page if tree else window
    per_request = kept + interval + 2 * page
    if tree:
        per_request += kept + page
    return per_request * requests + chunks_in_flight * chunk + page


def prefill_chunks_in_flight(pp_size: int, overlap_off: bool) -> int:
    """Prefill chunks a stage holds window KV for at once: one, two under overlap scheduling, and under PP at least one
    per micro-batch, since SGLang's PP loop schedules a request's next chunk before the stage has run the previous one.
    sglang_patches.pp_pool_resolve counts the same."""
    return max(pp_size, 1 if overlap_off else 2)


def window_cell_share(pool) -> float:
    """A window-pool token's KV bytes over a full-pool token's in ``pool`` (SWAKVPool): what a window token costs the
    full pool when SGLang splits the KV budget by --swa-full-tokens-ratio."""
    def cell(kv):
        return int(kv.layer_num) * int(kv.head_num) * int(kv.head_dim) * kv.store_dtype.itemsize

    return cell(pool.swa_kv_pool) / cell(pool.full_kv_pool)


def fitting_ratio(full: int, pool: int, demand: int, share: float) -> float:
    """The --swa-full-tokens-ratio whose window pool holds ``demand`` tokens, given SGLang split the KV budget into
    ``full`` and ``pool`` tokens. SGLang sizes the full pool as budget / (1 + ratio * share) and the window pool as
    ratio times that, so the fitting ratio is ``demand / (budget - demand * share)`` with budget ``full + pool *
    share``, capped at 1 (a window pool as large as the full one never runs short first). Callers ask for a page more,
    since SGLang floors both pools to whole pages."""
    budget = full + pool * share
    if budget <= demand * share:
        return 1.0
    return min(1.0, math.ceil(demand / (budget - demand * share) * 1e4) / 1e4)


def check_window_pool(model_runner, window: int) -> None:
    """Step 0: refuse a window pool the running requests could run short of (bound: window_demand), since SGLang would
    retract a running request to free window pages. A window pool at least as large as the full pool never runs short
    first: every window slot in use pairs with a full one."""
    from sglang.srt.arg_groups.overrides import resolution_result
    from sglang.srt.environ import envs

    sa = model_runner.server_args
    full = int(model_runner.full_max_total_num_tokens)
    pool = int(model_runner.swa_max_total_num_tokens)
    if pool >= full:
        logger.info("[neuron] hybrid SWA memory: window pool %d tokens, the full pool's %d", pool, full)
        return
    tree = not bool(resolution_result(sa, "disable_radix_cache", False))
    chunk = int(resolution_result(sa, "chunked_prefill_size", 0) or 0)
    if chunk <= 0:
        raise ValueError("hybrid SWA memory needs chunked prefill, which SGLang resolved off: an unchunked prompt keeps "
                         "all of its window KV until its prefill ends; launch with --chunked-prefill-size, or with "
                         "--disable-hybrid-swa-memory")
    demand = window_demand(
        window=window, page=int(model_runner.page_size), requests=int(model_runner.max_running_requests),
        interval=max(1, int(envs.SGLANG_SWA_EVICTION_INTERVAL.get())), chunk=chunk,
        chunks_in_flight=prefill_chunks_in_flight(int(getattr(sa, "pp_size", 1) or 1),
                                                  bool(resolution_result(sa, "disable_overlap_schedule", False))),
        tree=tree)
    cache = "radix cache" if tree else "chunk cache"
    if pool < demand:
        share = window_cell_share(model_runner.token_to_kv_pool)
        raise ValueError(
            f"hybrid SWA memory: the window pool holds {pool} tokens and {model_runner.max_running_requests} running "
            f"requests on the {cache} can hold {demand} of it (window {window}, page {model_runner.page_size}), so "
            f"SGLang would retract running requests; launch with --swa-full-tokens-ratio "
            f"{fitting_ratio(full, pool, demand + int(model_runner.page_size), share):.4f}, lower "
            "--max-running-requests, or launch with --disable-hybrid-swa-memory")
    logger.info("[neuron] hybrid SWA memory: window pool %d tokens, at least the %d that %d running requests can hold "
                "on the %s", pool, demand, model_runner.max_running_requests, cache)


def window_pool(token_to_kv_pool):
    """The launch's hybrid pool, whose sliding layers' tables and slots the runtime translates; None on one pool."""
    from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool

    return token_to_kv_pool if isinstance(token_to_kv_pool, SWAKVPool) else None


def window_blocks(pool, table: torch.Tensor, page: int) -> torch.Tensor:
    """A full-pool block table ``[rows, columns]`` (int32, -1 past a row's pages) as window-pool pages. Each page
    translates by its first slot, since the allocator pairs a full page with one window page. An evicted page maps to
    the reserved page 0, which the window mask never reads."""
    slots = table.clamp(min=0).to(torch.long) * page
    pages = (pool.translate_loc_from_full_to_swa(slots.reshape(-1)) // page).reshape(table.shape).to(torch.int32)
    return torch.where(table >= 0, pages, table)


def window_slots(pool, slots: torch.Tensor) -> torch.Tensor:
    """Full-pool token slots (int64; padding writes slot 0) as window-pool slots; slot 0 stays 0, the reserved page."""
    return pool.translate_loc_from_full_to_swa(slots.to(torch.long))


def check_window_resident(pool, table: torch.Tensor, cached: int, window: int, page: int) -> None:
    """Raise unless the window pool holds the first query's window prior, keys ``[cached - (window - 1), cached)``, for
    a prefill over ``cached`` cached tokens. SGLang's tree ends a match only over live window KV, which the request
    locks; a key translated to page 0 would silently read that page's KV."""
    keys = torch.arange(max(cached - (window - 1), 0), cached, dtype=torch.long)
    if not keys.numel():
        return
    pages = table[0, keys // page].to(torch.long)
    slots = window_slots(pool, pages.clamp(min=0) * page + keys % page)
    missing = (pages <= 0) | (slots < page)
    if bool(missing.any()):
        key = int(keys[missing.nonzero()[0, 0]])
        raise RuntimeError(f"a prefill over a cached prefix of {cached} tokens: the window KV of position {key}, "
                           f"inside the first query's {window}-token window, is not in the window pool")
