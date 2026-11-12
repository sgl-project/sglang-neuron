# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""Bound a pipelined PD decode's admission at ``pp_max_micro_batch_size`` requests per micro-batch.

Upstream's PD decode admission (``SchedulerDisaggregationDecodeMixin._get_new_prebuilt_batch``) fills the micro-batch
it schedules up to ``max_running_requests``, whatever the pipeline. SGLang's PP prefill admission bounds each
micro-batch by ``pp_max_micro_batch_size`` but the decode's does not, so requests arriving together all join one
micro-batch and each stage idles while that micro-batch is at another. The admission is rebuilt with the bound right
after its batch size (``SUBSTITUTIONS``). SGLang's default bound, ``max(max_running_requests // pp_size, 1)``, is the
row count of the pipelined decode's graph (``distributed.pipeline.pp_decode_problems``). Only a pipelined decode
registers this module (``general_plugin._register_pipeline_parallel``).

Upstream: a generic PP decode fix, the same bound upstream's PP prefill admission applies.

Targets, md5-pinned against upstream SGLang 43881a7: ``SchedulerDisaggregationDecodeMixin._get_new_prebuilt_batch``
(rebuilt and replaced); relied on: ``SchedulerDisaggregationDecodeMixin.get_next_disagg_decode_batch_to_run`` (the
decode's batch step) and ``SchedulerDisaggregationDecodeMixin.get_new_prebuilt_batch``, which hand the micro-batch's
running batch to the admission."""

from __future__ import annotations

_DECODE = "sglang.srt.disaggregation.decode"
TARGET = f"{_DECODE}.SchedulerDisaggregationDecodeMixin._get_new_prebuilt_batch"
# (module, qualified name): md5 of the source; the rebuilt one, then the chain that hands it its running batch
_TARGETS_MD5 = {
    (_DECODE, "SchedulerDisaggregationDecodeMixin._get_new_prebuilt_batch"): "45538280933e4ba68439c2ce3b2f0c76",
    # the decode's batch step: hands the micro-batch's running batch to get_new_prebuilt_batch
    (_DECODE, "SchedulerDisaggregationDecodeMixin.get_next_disagg_decode_batch_to_run"):
        "3b90210c4c28d3c4cdb89f8f5e5a815e",
    # hands it to the admission
    (_DECODE, "SchedulerDisaggregationDecodeMixin.get_new_prebuilt_batch"): "7a068620d95e541fc993d0d5acd7e7ef",
}
_BATCH_SIZE = "    batch_size = min(self.req_to_token_pool.size, self.max_running_requests)\n"
SUBSTITUTIONS = [(_BATCH_SIZE, _BATCH_SIZE + (
    "    # a micro-batch holds at most pp_max_micro_batch_size requests, as the PP prefill admission bounds it, so\n"
    "    # requests that arrive together fill every micro-batch\n"
    "    batch_size = min(batch_size, get_parallel().pp_max_micro_batch_size)\n"))]


def prebuilt_batch_with_micro_batch_bound():
    """SchedulerDisaggregationDecodeMixin._get_new_prebuilt_batch rebuilt from source with the micro-batch bound
    (SUBSTITUTIONS)."""
    from sglang_neuron.sglang_patches import source_guard

    return source_guard.with_substitutions(
        source_guard.resolve(_DECODE, "SchedulerDisaggregationDecodeMixin._get_new_prebuilt_batch"), SUBSTITUTIONS)


def register(registry, replace) -> None:
    from sglang_neuron.sglang_patches import source_guard

    source_guard.check_targets(_TARGETS_MD5, __name__)
    registry.register(TARGET, prebuilt_batch_with_micro_batch_bound(), replace)
