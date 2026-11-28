# SPDX-License-Identifier: Apache-2.0
"""Only the stats-logging scheduler (attention TP rank 0, attention CP rank 0) writes Prometheus samples to the shared
multiprocess directory.

SchedulerMetricsCollector.init_new builds a collector on every TP rank, and prometheus_client's multiprocess values
write each rank's series to its own mmap file. At TP64 that is 64 copies of every scheduler histogram: /metrics merges
~200 files into 3.3 MB on every scrape, ~0.9 s of the HTTP server's event loop. Under a 1 Hz scraper (AgentX's server
metrics) a slow scrape makes the next ones queue behind it, the event loop never catches up, and requests through that
server stall. Upstream reports only the stats-logging rank (``current_scheduler_metrics_enabled``); the other ranks'
series are never read.

On every other rank (unless --enable-metrics-for-all-schedulers), the scheduler process switches prometheus_client to
its in-process value class before the collector is built, so those ranks keep their collectors (every call on them
still works) but write nothing to the shared directory.

Upstream: build the collector, or its multiprocess values, only where current_scheduler_metrics_enabled."""

import logging

logger = logging.getLogger(__name__)

_MODULE = "sglang.srt.observability.metrics_collector"
_TARGETS_MD5 = {(_MODULE, "SchedulerMetricsCollector.init_new"): "c9db15ff895580d790a2623d85744844"}


def _init_new(original_fn, cls, *args, **kwargs):
    import sys

    from prometheus_client import values

    mc = sys.modules[_MODULE]
    obs = mc.get_observability()
    parallel = mc.get_parallel()
    stats_rank = parallel.attn_tp_rank == 0 and parallel.attn_cp_rank == 0
    if obs.enable_metrics and not stats_rank and not obs.enable_metrics_for_all_schedulers:
        if values.ValueClass is not values.MutexValue:
            values.ValueClass = values.MutexValue
            logger.debug("[neuron] scheduler metrics of TP rank %s stay in-process", parallel.tp_rank)
    return original_fn(cls, *args, **kwargs)


def register(registry, around) -> None:
    from sglang_neuron.sglang_patches import source_guard

    source_guard.check_targets(_TARGETS_MD5, __name__)
    registry.register(f"{_MODULE}.SchedulerMetricsCollector.init_new", _init_new, around)
