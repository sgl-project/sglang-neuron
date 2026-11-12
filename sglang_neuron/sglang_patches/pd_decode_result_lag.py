# SPDX-License-Identifier: Apache-2.0
"""The giga-kernel decode's overlap loop keeps as many launched steps unprocessed as the in-flight bound allows, so a
step's launch no longer waits for the result of the step before it (serving_policy.decode_result_lag).

Upstream's ``event_loop_overlap_disagg_decode`` launches step i + 1 and then processes step i, which gives the host one
device step for the result readback, the processing, the scheduling and the next launch's inputs. On the giga-kernel
decode every one of the 128 ranks meets the others at the kernel's in-NEFF barrier each step, so a rank whose host
needs longer than that for one step holds all of them. With a lag of L (device_io's in-flight bound,
``SGLANG_NEURON_MAX_INFLIGHT_LAUNCHES``, 2 to ``MAX_LAG``) the loop processes step i only after it launched step
i + L, and a rank whose host falls behind for one step runs from the steps already queued on its device instead of
stalling the others.

The loop is upstream's, rebuilt from its source with three changes:

1. After a launch it processes the oldest results until L stay unprocessed (``neuron_process``). An iteration that
   launches nothing processes every queued result. Where the iteration before launched nothing too, the queue is
   already empty, and the drain before upstream's ``on_idle`` only keeps it from ever running with results queued. A
   step whose result needs a delayed sample processes every older result and keeps its own, which upstream's
   ``launch_batch_sample_if_needed`` samples next, as upstream keeps its one.
2. A batch that runs without overlap processes every queued result before its launch (``NEURON_DRAIN``).
3. When the planning retracts a request that a queued result still holds, every queued result is processed before the
   launch, while the request is still marked retracted (``NEURON_PLAN``). This platform's retraction backup is
   ``none``, so SGLang aborts a retracted request rather than resuming it, and the drain only empties the queue on that
   rare step. Were a retracted request resumed, ``resume_retracted_reqs`` would clear ``is_retracted`` at the next
   iteration and the request would take a token from a step launched before its retraction.

A pause in ``retract`` mode also has to drain the queue. Upstream's ``Scheduler.pause_generation`` processes one queued
result, all an unlagged loop ever holds, so the wrapper processes every queued result but the newest first, and
upstream's then processes that one.

Everything else is upstream's overlap contract, which does not depend on how many results are queued: a request that
finished at step i is skipped by the processing of the steps launched after it (``req.finished()``), and each step's
input tokens come from the previous launch on the device. The one part of upstream's scheduling that assumes a single
queued result, the skip of requests finishing in flight (``Scheduler._reqs_finishing_inflight``, which asserts one), is
off on a disaggregated decode: only the aggregated ``event_loop_overlap`` turns ``enable_skip_finishing_decode`` on,
and ``_dispatch_event_loop_once`` turns it off before every loop it starts. Both are pinned.

The queued steps still read and write a finished request's pages on the device. On a disaggregated decode SGLang
stages every freed page until an allocation runs short, and the allocator hands a staged page out only once every
launch that may touch it has finished (mem_cache.allocator.NeuronPagedTokenToKVPoolAllocator.merge_and_sort_free), so
no page a queued step touches is reused before that step has run. Every decision follows the
scheduler's queue lengths and request states, which are the same on every rank of a DP group.

The policy turns the lag on for the giga-kernel decode only, the row it is measured on; the per-layer decodes run
upstream's loop.

Targets md5-pinned against upstream SGLang 43881a7:
``SchedulerDisaggregationDecodeMixin.event_loop_overlap_disagg_decode`` (decorated with ``@torch.no_grad()``, which only
wraps the call; the rebuild applies it again), ``Scheduler.pause_generation``, and the two that keep the skip of
requests finishing in flight off here, ``Scheduler._reqs_finishing_inflight`` and ``_dispatch_event_loop_once``.

Upstream: a generic option of the decode overlap loop (process a step's result L launches later). It pays where the
device can queue L steps and the host's step time varies, as on a decode of many ranks."""

from __future__ import annotations

_DECODE = "sglang.srt.disaggregation.decode"
TARGET = (_DECODE, "SchedulerDisaggregationDecodeMixin.event_loop_overlap_disagg_decode")
PAUSE = ("sglang.srt.managers.scheduler", "Scheduler.pause_generation")
_TARGETS_MD5 = {
    TARGET: "246b69291bfd47d63d0126dea073abb5",
    # processes one queued result when it retracts
    PAUSE: "dc4aefb416e83782c40da727908395df",
    # asserts one queued result, but only while enable_skip_finishing_decode is on
    ("sglang.srt.managers.scheduler", "Scheduler._reqs_finishing_inflight"): "49de36141874582bb409f54388251ca4",
    # turns enable_skip_finishing_decode off before every loop; only the aggregated event_loop_overlap turns it on
    ("sglang.srt.managers.scheduler", "_dispatch_event_loop_once"): "c16577b3a45ebae03284e5fbdaeb03f4",
}
DECORATOR = "@torch.no_grad()\n"

UPSTREAM_DRAIN = """        if disable_overlap_for_batch and self.last_batch:
            pop_and_process()
"""
NEURON_DRAIN = """        if disable_overlap_for_batch and self.last_batch:
            while self.result_queue:
                pop_and_process()
"""
UPSTREAM_PLAN = """        self.running_batch = plan.running_batch
        batch = plan.batch_to_run
"""
NEURON_PLAN = """        self.running_batch = plan.running_batch
        batch = plan.batch_to_run
        if any(req.is_retracted for queued, _ in self.result_queue for req in queued.reqs):
            while self.result_queue:
                pop_and_process()
"""
UPSTREAM_PROCESS = """        if self.last_batch:
            if not disable_overlap_for_batch:
                pop_and_process()
        elif batch is None:
            self.on_idle()
"""


def neuron_process(lag: int) -> str:
    """The result processing of the rebuilt loop: ``lag`` launched steps stay unprocessed after a launch.

    The queue is already empty when the idle branch runs (an iteration without a batch after one with a batch drains
    in the first branch); the idle branch drains anyway, so no result can outlive an idle step."""
    return f"""        if self.last_batch:
            if not disable_overlap_for_batch:
                keep = 0 if not batch else 1 if getattr(batch_result, "delay_sample_func", None) else {int(lag)}
                while len(self.result_queue) > keep:
                    pop_and_process()
        elif batch is None:
            while self.result_queue:
                pop_and_process()
            self.on_idle()
"""


# The validated range's top, and also the most a full-length request can ride: a finished request stays in the L
# steps launched after its last token, each writing one more KV slot, and upstream sizes every req_to_token row at the
# context plus 4 (allocation_sizing.get_req_to_token_extra_context_len, "the decode over-allocation").
MAX_LAG = 4


def lag() -> int:
    """The launched steps the loop keeps unprocessed: device_io's in-flight bound, refused outside 2 to MAX_LAG."""
    from sglang_neuron.backend import device_io

    value = device_io._max_inflight()
    if not 2 <= value <= MAX_LAG:
        raise ValueError(f"the giga-kernel decode keeps {device_io.MAX_INFLIGHT} launched steps unprocessed, and it "
                         f"is {value}: set it to 2 to {MAX_LAG}, or leave it unset")
    return value


def event_loop_overlap_disagg_decode():
    """event_loop_overlap_disagg_decode, upstream's with ``lag()`` launched steps left unprocessed after each launch,
    under upstream's own decorator."""
    import inspect

    import torch

    from sglang_neuron.sglang_patches import source_guard

    upstream = inspect.unwrap(source_guard.resolve(*TARGET))
    substitutions = [(UPSTREAM_PLAN, NEURON_PLAN), (UPSTREAM_DRAIN, NEURON_DRAIN),
                     (UPSTREAM_PROCESS, neuron_process(lag()))]
    rebuilt = source_guard.with_substitutions(upstream, substitutions, decorators=(DECORATOR,))
    return torch.no_grad()(rebuilt)


def pause_generation(original_fn, self, recv_req, *args, **kwargs):
    """AROUND Scheduler.pause_generation: a retracting pause processes every queued result but the newest, which
    upstream's processes."""
    if recv_req.mode == "retract" and self.enable_overlap and self.last_batch:
        while len(self.result_queue) > 1:
            batch, result = self.result_queue.popleft()
            self.process_batch_result(batch, result)
    return original_fn(self, recv_req, *args, **kwargs)


def register(registry, around, replace) -> None:
    from sglang_neuron.sglang_patches import source_guard

    source_guard.check_targets(_TARGETS_MD5, __name__)
    registry.register(".".join(TARGET), event_loop_overlap_disagg_decode(), replace)
    registry.register(".".join(PAUSE), pause_generation, around)
