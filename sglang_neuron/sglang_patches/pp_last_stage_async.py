# SPDX-License-Identifier: Apache-2.0
"""Have a pipelined PD prefill's last stage send a micro-batch's sampled outputs once its forward has run, without
blocking on the forward at launch.

Upstream's last PP rank queues each micro-batch's outputs at launch (``_pp_launch_batch`` puts an event and
``PPProxyTensors(_pp_prepare_tensor_dict(result, batch))`` on ``last_rank_comm_queue``), and
``_pp_send_output_to_next_stage`` later sends them with that event as ``ready_event``. Under the plugin's async
execution the result is a pending sample (``async_execution.PendingSample``: tokens still on the device, the graph
maybe still running) and the CPU device's stream wait does nothing, so the queue would hold unfinished device tensors
that the gloo ring cannot send.

Here the queued event is a ``PendingOutputs``, and the send first fills the outputs from the sample's host copy
(``PendingSample.materialize``, which waits on the forward's launch event), rebuilt by SGLang's own
``_pp_prepare_tensor_dict``. The overlap-order prefill loop (sglang_patches/pp_prefill_overlap_loop) sends right after
launching the next micro-batch, so the wait is at most the previous forward's tail.

Upstream: generic for a backend whose forward returns at launch; upstream would take it behind that condition.

Targets, md5-pinned against upstream SGLang 43881a7: ``SchedulerPPMixin._pp_launch_batch`` and
``_pp_send_dict_to_next_stage`` (wrapped), ``_pp_send_output_to_next_stage`` (which passes the queued event and dict
to the send), ``_pp_prepare_tensor_dict`` (the outputs the send rebuilds) and ``init_pp_loop_state`` (the queue)."""

from __future__ import annotations

from types import SimpleNamespace

_MODULE = "sglang.srt.managers.scheduler_pp_mixin"
LAUNCH = f"{_MODULE}.SchedulerPPMixin._pp_launch_batch"
SEND = f"{_MODULE}.SchedulerPPMixin._pp_send_dict_to_next_stage"
# (module, qualified name): md5 of the source; the wrapped ones, then those the fill relies on
_TARGETS_MD5 = {
    (_MODULE, "SchedulerPPMixin._pp_launch_batch"): "59cebf2d14f04cc82dfbced48d974eff",
    (_MODULE, "SchedulerPPMixin._pp_send_dict_to_next_stage"): "a68ce1f2295efa4afa71e863298d8f9a",
    (_MODULE, "SchedulerPPMixin._pp_send_output_to_next_stage"): "e2025eddedf940579e994635b5553252",
    (_MODULE, "SchedulerPPMixin._pp_prepare_tensor_dict"): "d29fe72b57e526eda7c570e320d40051",
    (_MODULE, "SchedulerPPMixin.init_pp_loop_state"): "2deba5f7b367977316763050845e0288",
}


class PendingOutputs:
    """A last-rank micro-batch's queued outputs before its forward has run; ``fill`` writes them from the sample's host
    copy when SGLang pops them to send (once, as a queue entry is sent once)."""

    def __init__(self, result, batch):
        self.result = result
        # what _pp_prepare_tensor_dict reads of the batch, as of the launch (the micro-batch may be merged or filtered
        # before its outputs go)
        self.batch = SimpleNamespace(return_logprob=bool(batch.return_logprob), spec_algorithm=batch.spec_algorithm)

    def fill(self, scheduler, tensor_dict) -> None:
        from sglang_neuron.model_executor.async_execution import take_pending_sample

        take_pending_sample(self.result, self.batch.return_logprob)
        tensor_dict.clear()
        tensor_dict.update(scheduler._pp_prepare_tensor_dict(self.result, self.batch))
        self.result = None


def pp_launch_batch(original_fn, self, mb_id, cur_batch, pp_proxy_tensors, mb_metadata, last_rank_comm_queue):
    """AROUND SchedulerPPMixin._pp_launch_batch: the last rank queues a pending result's outputs behind its
    PendingOutputs; another rank sends the batch's last chunks at their forward with
    SGLANG_NEURON_PP_PREFILL_EARLY_LAST_SEND=1 (sglang_patches/pp_prefill_last_send)."""
    result, event = original_fn(self, mb_id, cur_batch, pp_proxy_tensors, mb_metadata, last_rank_comm_queue)
    if not self.pp_group.is_last_rank:
        # with SGLANG_NEURON_PP_PREFILL_EARLY_LAST_SEND=1: a last chunk's KV and state go out now, waiting on this
        # launch, with no metadata row (sglang_patches/pp_prefill_last_send)
        from sglang_neuron.sglang_patches import pp_prefill_last_send

        pp_prefill_last_send.send_after_launch(self, cur_batch)
    logits_output = getattr(result, "logits_output", None)
    if (self.pp_group.is_last_rank and last_rank_comm_queue
            and getattr(logits_output, "neuron_pending_sample", None) is not None):
        _, outputs = last_rank_comm_queue[-1]
        last_rank_comm_queue[-1] = (PendingOutputs(result, cur_batch), outputs)
    return result, event


def pp_send_dict_to_next_stage(original_fn, self, tensor_dict, async_send=True, msg_type="default",
                               ready_event=None):
    """AROUND SchedulerPPMixin._pp_send_dict_to_next_stage: fill queued pending outputs before SGLang sends them."""
    if isinstance(ready_event, PendingOutputs):
        ready_event.fill(self, tensor_dict)
        ready_event = None
    return original_fn(self, tensor_dict, async_send=async_send, msg_type=msg_type, ready_event=ready_event)


def register(registry, around) -> None:
    from sglang_neuron.sglang_patches import source_guard

    source_guard.check_targets(_TARGETS_MD5, __name__)
    registry.register(LAUNCH, pp_launch_batch, around)
    registry.register(SEND, pp_send_dict_to_next_stage, around)
