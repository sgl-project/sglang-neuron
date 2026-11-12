# SPDX-License-Identifier: Apache-2.0
"""Re-solve every PP stage's KV pools from the synced token count, and count a prefill chunk per micro-batch in the
window pool.

SGLang syncs ``max_total_num_tokens`` to the stages' minimum (``KVCacheConfigurator._apply_token_constraints``) but
re-solves the pools from it only where the count changed (``config_from_budget``), so the stage holding the minimum
keeps its budget-path sizes. The two paths can split one count differently: ``SWARequestCapPoolConfigurator`` sizes
the window pool to the request cap on the budget path and to ``min(cap, count)`` on the count path. That stage's window
pool can then hold more than the others', and the stages, scheduling the same batches from their own allocators, run
out of window pages at different requests.

Here every stage of a pipelined launch re-solves from the synced count. The count path depends only on the count, the
page, the request cap or ratio, and the kinds of layer a stage holds, and every stage of a served hybrid launch holds
both kinds (mem_cache.hybrid_swa refuses a partition that does not), so all stages get the same full and window
counts. SGLang's unified memory pool under PP is refused: the count path does not carry its byte budget.

The window pool's request cap (``compute_swa_request_cap``, on the chunk cache with ``--max-running-requests``) counts
the prefill chunks a stage holds window KV for at once: one, or two under the overlap schedule. SGLang's PP loop
schedules a request's next chunk into the next micro-batch before the stage has run the previous one, so a stage can
hold one chunk per micro-batch and the cap counts ``pp_size`` of them. A PD decode's cap counts no chunks and stays
SGLang's.

Targets md5-pinned against upstream SGLang 43881a7: ``KVCacheConfigurator.config_from_budget`` and
``compute_swa_request_cap`` (wrapped). Relied on: ``KVCacheConfigurator._apply_token_constraints`` (the stages' MIN
sync of the count), ``create_memory_pool_configurator`` and ``SWARequestCapPoolConfigurator.is_applicable`` (the
configurator a stage takes), ``HybridSWAPoolConfigurator.__init__`` and ``_solve_pool_sizes`` (the split by the kinds
a stage holds), and the count paths (``calculate_pool_sizes_from_max_tokens``) of the default and the hybrid
configurators.

Upstream: generic PP fixes: re-solve from the synced count on every stage, and count a chunk per micro-batch in the
window cap."""

from __future__ import annotations

_CONFIGURATOR = "sglang.srt.mem_cache.kv_cache_configurator"
_SIZING = "sglang.srt.model_executor.pool_configurator"
TARGET = (_CONFIGURATOR, "KVCacheConfigurator.config_from_budget")
CAP = (_SIZING, "compute_swa_request_cap")
_TARGETS_MD5 = {
    TARGET: "a4d6c76f663f858a5079f680430b1484",
    CAP: "608f658e2ce49541b33ce2ad66f7bb60",
    (_CONFIGURATOR, "KVCacheConfigurator._apply_token_constraints"): "355f93e37c79615157cdd25b2034ffe5",
    (_SIZING, "create_memory_pool_configurator"): "2b934a749398e1fe464c95ee4099df08",
    (_SIZING, "SWARequestCapPoolConfigurator.is_applicable"): "b13b8006be74527826a284f043ce18f9",
    (_SIZING, "HybridSWAPoolConfigurator.__init__"): "b5cadec68514cf41347f63fd3b3f0d11",
    (_SIZING, "HybridSWAPoolConfigurator._solve_pool_sizes"): "28d72724b916cc473edebe581f38e1bc",
    (_SIZING, "DefaultPoolConfigurator.calculate_pool_sizes_from_max_tokens"): "73567ca549675d19fda56e692eab3408",
    (_SIZING, "SWARequestCapPoolConfigurator.calculate_pool_sizes_from_max_tokens"): "48fb88c72ee95db2f15b980aa8a4a862",
    (_SIZING, "HybridSWAPoolConfigurator.calculate_pool_sizes_from_max_tokens"): "2cdd50bf1d138223ca7ef88e4dd761a9",
}


def config_from_budget(original_fn, self, budget_bytes, *args, **kwargs):
    """AROUND KVCacheConfigurator.config_from_budget: under PP, re-solve the pools from the synced token count."""
    config = original_fn(self, budget_bytes, *args, **kwargs)
    from sglang.srt.runtime_context import get_memory, get_parallel, get_schedule

    if int(get_parallel().pp_size) <= 1:
        return config
    if get_memory().enable_unified_memory:
        raise ValueError("SGLang's unified memory pool (--enable-unified-memory) under pipeline parallelism is not "
                         "served: the stages re-solve their pools from the synced token count, which does not carry "
                         "the pool's byte budget; launch without --enable-unified-memory")
    from sglang.srt.model_executor.pool_configurator import create_memory_pool_configurator

    return create_memory_pool_configurator(self).calculate_pool_sizes_from_max_tokens(
        config.max_total_num_tokens, get_schedule().page_size)


def compute_swa_request_cap(original_fn, *args, **kwargs):
    """AROUND compute_swa_request_cap: under PP, count a prefill chunk per micro-batch in the window cap."""
    cap = original_fn(*args, **kwargs)
    from sglang.srt.runtime_context import get_disagg, get_parallel, get_schedule

    pp_size = int(get_parallel().pp_size)
    schedule = get_schedule()
    counted = 1 if schedule.disable_overlap_schedule else 2
    if pp_size <= counted or get_disagg().disaggregation_mode == "decode":
        return cap
    return cap + (pp_size - counted) * schedule.chunked_prefill_size


def register(registry, around) -> None:
    from sglang_neuron.sglang_patches import source_guard

    source_guard.check_targets(_TARGETS_MD5, __name__)
    registry.register(".".join(TARGET), config_from_budget, around)
    registry.register(".".join(CAP), compute_swa_request_cap, around)
