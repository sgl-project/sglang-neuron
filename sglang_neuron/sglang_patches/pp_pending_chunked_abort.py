# SPDX-License-Identifier: Apache-2.0
"""Apply a client abort of the request a pipelined PD prefill is chunking before the stage schedules.

SGLang's ``abort_request`` only marks a request mid-chunking (``_pending_chunked_abort_req``), since tearing it down
mid-iteration is unsafe, and ``process_pending_chunked_abort`` drops it at the top of the next scheduling step. The PD
prefill PP loop (``SchedulerPPMixin.event_loop_pp_disagg_prefill``) never calls it, so an aborted request keeps
chunking to the end of its prompt. This runs it at the top of ``process_prefill_chunk``, the first scheduling step of
the PP loop and of its overlap-order rebuild (sglang_patches/pp_prefill_overlap_loop).

The AbortReq reaches every stage with the same iteration's requests, so every stage drops the request in that
iteration and they keep scheduling the same batches. Chunks already launched drain as their results are processed
(``process_batch_result_disagg_prefill`` skips an aborted middle chunk and releases nothing twice).

Upstream: generic: the PD prefill PP loop would call ``process_pending_chunked_abort`` like SGLang's other loops.

Targets, md5-pinned against upstream SGLang 43881a7: ``SchedulerDisaggregationPrefillMixin.process_prefill_chunk``
(wrapped), ``SchedulerPPMixin.event_loop_pp_disagg_prefill`` (the loop that calls it without the pending abort) and
``Scheduler.process_pending_chunked_abort`` (what the wrapper runs)."""

from __future__ import annotations

_PREFILL = "sglang.srt.disaggregation.prefill"
TARGET = f"{_PREFILL}.SchedulerDisaggregationPrefillMixin.process_prefill_chunk"
# (module, qualified name): md5 of the source; the wrapped one, then the loop and the abort it relies on
_TARGETS_MD5 = {
    (_PREFILL, "SchedulerDisaggregationPrefillMixin.process_prefill_chunk"): "dca76f30fd359b20c4c3b2e861490e34",
    ("sglang.srt.managers.scheduler_pp_mixin", "SchedulerPPMixin.event_loop_pp_disagg_prefill"):
        "8bbb0377fb4214f8ff289436470f2e7e",
    ("sglang.srt.managers.scheduler", "Scheduler.process_pending_chunked_abort"): "d2bd1a97dc566734536f2e687a65e37d",
}


def process_prefill_chunk(original_fn, self, *args, **kwargs):
    """AROUND SchedulerDisaggregationPrefillMixin.process_prefill_chunk: apply the pending chunked abort first."""
    self.process_pending_chunked_abort()
    return original_fn(self, *args, **kwargs)


def register(registry, around) -> None:
    from sglang_neuron.sglang_patches import source_guard

    source_guard.check_targets(_TARGETS_MD5, __name__)
    registry.register(TARGET, process_prefill_chunk, around)
