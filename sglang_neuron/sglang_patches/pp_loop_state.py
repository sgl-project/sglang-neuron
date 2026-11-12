# SPDX-License-Identifier: Apache-2.0
"""The plugin's PP loop state, set where every SGLang PP loop starts (``init_pp_loop_state``), and how the overlap-order
loops process a launched micro-batch.

- **Ring first.** The loop's gloo pairs connect before its first send (``distributed.pipeline.connect_pp_ring``), on
  every rank, at the same loop point.
- **Launched, unprocessed batches.** An overlap-order loop (``pp_prefill_overlap_loop``) processes a micro-batch a few
  iterations after launching it, so the batch leaves its ``mbs`` slot early. It is kept in ``pp_overlap_records``,
  which the drain check counts, so a flush, a weight-update check or the idle check never sees the micro-batches
  drained while a launched batch is unprocessed.

``process_step`` processes one launched batch: a non-last stage from the outputs its previous stage sends, handed on
only if the next stage is not the last; the last stage from its own outputs instead of getting them back round the
ring, since they are host tensors, the same bytes the ring would return.

A pipelined PD prefill's stages agree on the bootstrap queue through upstream's own consensus, pinned in
``_AGREEMENT_MD5`` so a change to it stops the launch for review: the first stage polls each room's
``PPConsensusStore`` minimum over every stage's status (``poll_and_all_reduce_prefill_pp``), and ``pp_sync_polls``
hands its polls to every later stage at the same loop point. A room that fails on one stage is dropped on every stage
in the same iteration, and a client abort reaches every stage before the first stage polls (``Scheduler.abort_request``
concludes the sender Failed, which ``_poll_with_status`` returns first). This needs every stage to take every
iteration's requests (``distributed.pipeline`` refuses a receive interval above 1).

A pipelined aggregated server registers only the loop state (``register_aggregated``) and runs SGLang's own
``SchedulerPPMixin.event_loop_pp``, whose order it relies on (``_AGGREGATED_LOOP_MD5``): a stage receives the previous
stage's proxy before launching a micro-batch and sends its own right after, and the last stage's host outputs go round
the ring and are processed on every stage's host before that micro-batch is scheduled again. So every stage has each
request's sampled tokens on the host, a decode's first stage takes its input ids from there, and no tokens are relayed
over the seam. That loop keeps a launched batch in ``mbs`` until it is processed, so upstream's drain check holds and
the records stay empty.

Targets, md5-pinned against upstream SGLang 43881a7: ``SchedulerPPMixin.init_pp_loop_state`` and
``Scheduler._pp_microbatches_drained`` (wrapped), the PP mixin methods ``process_step`` calls, the agreement's
functions (``_AGREEMENT_MD5``) and the aggregated PP loop (``_AGGREGATED_LOOP_MD5``)."""

from __future__ import annotations

from collections import deque

_MIXIN = "sglang.srt.managers.scheduler_pp_mixin"
_SCHEDULER = "sglang.srt.managers.scheduler"
INIT = f"{_MIXIN}.SchedulerPPMixin.init_pp_loop_state"
DRAINED = f"{_SCHEDULER}.Scheduler._pp_microbatches_drained"
# (module, qualified name): md5 of the source; the wrapped ones, then those process_step calls
_TARGETS_MD5 = {
    (_MIXIN, "SchedulerPPMixin.init_pp_loop_state"): "2deba5f7b367977316763050845e0288",
    (_SCHEDULER, "Scheduler._pp_microbatches_drained"): "57001adf8b45f9b89edbc8cd6571539f",
    (_MIXIN, "SchedulerPPMixin._pp_recv_dict_from_prev_stage"): "f0958889a27f62eff9f862738f7eb961",
    (_MIXIN, "SchedulerPPMixin._pp_send_dict_to_next_stage"): "a68ce1f2295efa4afa71e863298d8f9a",
    (_MIXIN, "SchedulerPPMixin._pp_commit_comm_work"): "935f872964c601a49544236d3ae8604f",
    (_MIXIN, "SchedulerPPMixin._pp_prep_batch_result"): "6b9e7a29b510f1b3d39861653abaeb7d",
    (_MIXIN, "SchedulerPPMixin._pp_process_batch_result"): "f6dfcb7c1de1de98542455b6ffae581c",
}
_PREFILL = "sglang.srt.disaggregation.prefill"
_UTILS = "sglang.srt.disaggregation.utils"
_COMMON = "sglang.srt.disaggregation.common.conn"
# the stages' bootstrap agreement (the module docstring): relied on, nothing wrapped
_AGREEMENT_MD5 = {
    # ingests the iteration's requests before it pops the bootstrap queue
    (_MIXIN, "SchedulerPPMixin.event_loop_pp_disagg_prefill"): "8bbb0377fb4214f8ff289436470f2e7e",
    (_MIXIN, "SchedulerPPMixin.process_bootstrapped_queue"): "fe06e4c8ce95584afa4e05d0554236e3",
    # takes its polls through the PP consensus and applies them to the stage's own queue
    (_PREFILL, "PrefillBootstrapQueue.pop_bootstrapped"): "c4093598e177c3e19dfa9942de8380e5",
    # returns at once for a request whose bootstrap finished
    (_PREFILL, "SchedulerDisaggregationPrefillMixin.check_bootstrap"): "cf5568253e09c58f1edc2068872961b0",
    # the first stage polls, every later stage takes its polls
    (_UTILS, "poll_and_all_reduce_prefill_pp"): "3d9f5236efb532e31f22cf3a65b07716",
    (_UTILS, "pp_sync_polls"): "7225f56b8844bccebd33ff25e6818191",
    (_UTILS, "_poll_with_failure_injection_pp"): "6d7f48cc253ad8b0664c536ef3302d53",
    # a room's poll is the minimum over every stage's status; an aborted sender polls Failed
    (_COMMON, "CommonKVSender.poll_pp_consensus"): "510624d0276d9269bf54c283b6fe8f38",
    (_COMMON, "CommonKVSender._poll_with_status"): "a0146ee1fd19a666fb61f4333523f996",
    (_COMMON, "CommonKVSender.abort"): "91c63738a82dd86a26a9b40102f3a918",
    (_COMMON, "CommonKVManager.check_status_pp_consensus"): "0a181bdb08e7d0ab657e17bd2d2b42d7",
    ("sglang.srt.disaggregation.pp_consensus_store", "PPConsensusStore.collect"): "aa6f50dc84040afdf7a9ede5843af864",
    # an abort concludes a bootstrapping sender and finds every in-flight request
    (_SCHEDULER, "Scheduler.abort_request"): "0fed02d75e27c867928d017e68f7b601",
    (_SCHEDULER, "Scheduler.collect_inflight_reqs"): "9efe4e457552a783858421f1ae9ffeec",
    # optimistic prefill off at pp_size > 1 on a PD prefill, so a bootstrapped request is never polled again
    ("sglang.srt.arg_groups.serving_hook", "handle_other_validations"): "677b5a70f2b74f40ca6fc60fb2fa5bf8",
}

# the aggregated server's PP loop (the module docstring): relied on, nothing wrapped
_AGGREGATED_LOOP_MD5 = {
    (_MIXIN, "SchedulerPPMixin.event_loop_pp"): "118c4d14113c2f066fbc47bc16ceea32",
    # the proxy received before the launch and sent after it
    (_MIXIN, "SchedulerPPMixin._pp_recv_proxy_tensors"): "6a8973312df0da42999f2585147e86c4",
    (_MIXIN, "SchedulerPPMixin._pp_launch_batch"): "59cebf2d14f04cc82dfbced48d974eff",
    (_MIXIN, "SchedulerPPMixin._pp_send_proxy_to_next_stage"): "1c70fc00fe2afb40c94f37811433253a",
    # the last stage's outputs round the ring, waited for and processed on every stage's host
    (_MIXIN, "SchedulerPPMixin._pp_commit_send_output_work_and_preprocess_output_tensors"):
        "e6726193bf2272339d964ff2cb83f0dc",
    (_MIXIN, "SchedulerPPMixin._pp_send_recv_and_preprocess_output_tensors"): "81bed90bc989a8a88a4ee041dfe356f1",
    (_MIXIN, "SchedulerPPMixin._pp_send_output_to_next_stage"): "e2025eddedf940579e994635b5553252",
}


def init_pp_loop_state(original_fn, self, *args, **kwargs):
    """AROUND SchedulerPPMixin.init_pp_loop_state: connect the ring and start with no launched batches."""
    from sglang_neuron.distributed.pipeline import connect_pp_ring

    result = original_fn(self, *args, **kwargs)
    connect_pp_ring(self)
    self.pp_overlap_records = deque()
    return result


def pp_microbatches_drained(original_fn, self):
    """AROUND Scheduler._pp_microbatches_drained: a launched batch with an unprocessed result keeps the micro-batches
    undrained."""
    return original_fn(self) and not any(r is not None for r in getattr(self, "pp_overlap_records", ()))


def process_step(self, batch, result, metadata, is_last: bool, forwards: bool) -> None:
    """Process one launched batch: the last rank from its own result (already sent on), every other rank from the
    outputs its previous stage sends, handed on unless the next stage is the last (``forwards``). Every batch's outputs
    travel the ring: distributed.pipeline refuses SGLANG_PP_SKIP_PURE_CHUNKED_OUTPUT_COMM, the one skip."""
    import torch

    from sglang.srt.model_executor.forward_batch_info import PPProxyTensors

    if is_last:
        self._pp_process_batch_result(batch, result)
        return
    with torch.profiler.record_function("recv_res_dict_from_prev_stage"):
        tensor_dict, _event = self._pp_recv_dict_from_prev_stage()
    outputs = PPProxyTensors(tensor_dict)
    if forwards:
        self._pp_commit_comm_work(self.send_output_work)
        with torch.profiler.record_function("send_res_dict_to_next_stage"):
            self.send_output_work = self._pp_send_dict_to_next_stage(
                dict(outputs.tensors), async_send=True, msg_type="output")
    self._pp_process_batch_result(batch, self._pp_prep_batch_result(batch, metadata, outputs))


def register(registry, around) -> None:
    from sglang_neuron.sglang_patches import source_guard

    source_guard.check_targets(_TARGETS_MD5, __name__)
    source_guard.check_targets(_AGREEMENT_MD5, __name__)
    registry.register(INIT, init_pp_loop_state, around)
    registry.register(DRAINED, pp_microbatches_drained, around)


def register_aggregated(registry, around) -> None:
    """A pipelined aggregated server: the loop state on SGLang's own PP loop, after checking the loop it relies on."""
    from sglang_neuron.sglang_patches import source_guard

    source_guard.check_targets(_TARGETS_MD5, __name__)
    source_guard.check_targets(_AGGREGATED_LOOP_MD5, __name__)
    registry.register(INIT, init_pp_loop_state, around)
    registry.register(DRAINED, pp_microbatches_drained, around)
