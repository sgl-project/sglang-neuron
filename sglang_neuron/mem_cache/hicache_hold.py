# SPDX-License-Identifier: Apache-2.0
"""Asynchronous HiCache loads in the scheduler: a prefill whose host pages are still loading is held back while the
running batch decodes, and runs once the load has landed.

SGLang starts a load-back when it builds the prefill batch that reads the pages, and overlaps the copy with that
batch's forward layer by layer. This runtime's forward is one compiled graph that reads every layer at once, so it
waits for the whole load first (host_pool.set_consumer_waits). That wait is on the scheduler thread: for the load's
duration no step is launched, the device idles, and every running request's next token is late. Holding the batch
instead keeps the decode steps going while the load runs; the prefill launches once the load is done and no longer
waits.

Every TP rank's scheduler must pick the same batch, so the decisions read only state that is the same on every rank:
the batch, the running batch, and whether the load's ack is still in the cache controller's ack_load_queue. SGLang's
loading_check pops an ack once the load has finished on every rank (a MIN all-reduce, run at the top of every
get_next_batch_to_run, before the prefill is chosen).

Only an aggregated server holds: a disaggregated prefill never decodes, and a disaggregated decode runs no HiCache.
Not held either: a batch with no load, a chunked request (get_next_batch_to_run would stash a chunk that has not run), a
running batch with nothing to decode (holding would only idle the scheduler), and a DP-attention server (whose ranks
sync the batch kind separately). The platform refuses HiCache with pipeline parallelism, but on the PD prefill of a
model that keeps chunk state checkpoints, which holds nothing.
``SGLANG_NEURON_HICACHE_ASYNC_LOAD=0`` turns the hold off."""

from __future__ import annotations

import logging
import os
import time

logger = logging.getLogger(__name__)

ENABLED = os.environ.get("SGLANG_NEURON_HICACHE_ASYNC_LOAD", "1") == "1"
_STATS = os.environ.get("SGLANG_NEURON_HICACHE_STATS", "0") == "1"
_ATTR = "_neuron_held_prefill"

# The upstream functions the hold wraps or relies on, pinned by the md5 of their source against SGLang 43881a7
# (sglang_patches.source_guard); a changed source stops the launch.
_SOURCES_MD5 = {
    # wrapped: the hold decides here which prefill batch runs
    ("sglang.srt.managers.scheduler", "Scheduler.get_new_batch_prefill"): "f26088aa5d6704902fea48c9b447a5b2",
    # wrapped: an abort finds the held requests through it
    ("sglang.srt.managers.scheduler", "Scheduler._collect_inflight_batches"): "15ee339b1f105825c68ca8a069b7ea2b",
    # with no prefill batch it decodes the running batch, and it stashes a chunked request before choosing
    ("sglang.srt.managers.scheduler", "Scheduler.get_next_batch_to_run"): "afb49b7b3041a07f40543c528a1f630f",
    # pops a load's ack only once the load is done on every TP rank (the hold's release condition)
    ("sglang.srt.mem_cache.unified_radix_cache", "UnifiedRadixCache.loading_check"): "0a24fdad11d6f7153b3c4b336be4bb53",
}


class _Held:
    __slots__ = ("batch", "ack", "t0", "steps")

    def __init__(self, batch, ack):
        self.batch = batch
        self.ack = ack  # the HiCacheAck start_loading queued for this batch's load
        self.t0 = time.perf_counter()
        self.steps = 0  # decode steps chosen while it was held


def _plan(batch, running_batch):
    from sglang.srt.managers.schedule_batch import NextBatchPlan

    return NextBatchPlan(batch_to_run=batch, running_batch=running_batch)


def _decodes(running_batch) -> bool:
    return running_batch is not None and not running_batch.is_empty() and not running_batch.is_prefill_only


def _ack_queue(scheduler):
    controller = getattr(scheduler.tree_cache, "cache_controller", None)
    return None if controller is None else controller.ack_load_queue


def _aggregated() -> bool:
    from sglang_neuron import serving_policy

    return serving_policy.current().role == "aggregated"


def held_batch(scheduler):
    held = getattr(scheduler, _ATTR, None)
    return None if held is None else held.batch


def get_new_batch_prefill(original, scheduler, running_batch, *args, **kwargs):
    """Scheduler.get_new_batch_prefill: release the held batch once its load's ack is gone (or nothing is left to
    decode), else keep it and let get_next_batch_to_run choose the decode batch; hold a new batch whose load is still
    queued while the running batch decodes."""
    held = getattr(scheduler, _ATTR, None)
    if held is not None:
        queue = _ack_queue(scheduler) or ()
        if _decodes(running_batch) and any(ack is held.ack for ack in queue):
            held.steps += 1
            return _plan(None, running_batch)
        setattr(scheduler, _ATTR, None)
        if _STATS:
            logger.info("[neuron hicache] prefill held %.2f ms over %d decode steps for its load",
                        1e3 * (time.perf_counter() - held.t0), held.steps)
        return _plan(held.batch, running_batch)
    plan = original(scheduler, running_batch, *args, **kwargs)
    batch = plan.batch_to_run
    if (not ENABLED or batch is None or getattr(batch, "hicache_consumer_index", -1) < 0
            or not getattr(scheduler, "enable_hierarchical_cache", False) or not _aggregated()
            or getattr(scheduler, "require_mlp_sync", False)
            or batch.chunked_req is not None or scheduler.chunked_req is not None
            or not _decodes(plan.running_batch)):
        return plan
    queue = _ack_queue(scheduler)
    if not queue:
        return plan
    setattr(scheduler, _ATTR, _Held(batch, queue[-1]))
    return _plan(None, plan.running_batch)


def collect_inflight_batches(original, scheduler, *args, **kwargs):
    """Scheduler._collect_inflight_batches with the held batch, so an abort finds its requests (they are in neither
    the waiting queue nor the running batch) and finishes them after their prefill."""
    batches = original(scheduler, *args, **kwargs)
    held = held_batch(scheduler)
    return batches if held is None else [*batches, held]


def register_hooks(registry, hook_type) -> None:
    from sglang_neuron.sglang_patches import source_guard

    source_guard.check_targets(_SOURCES_MD5, __name__)
    scheduler = "sglang.srt.managers.scheduler.Scheduler"
    registry.register(f"{scheduler}.get_new_batch_prefill", get_new_batch_prefill, hook_type.AROUND)
    registry.register(f"{scheduler}._collect_inflight_batches", collect_inflight_batches, hook_type.AROUND)
