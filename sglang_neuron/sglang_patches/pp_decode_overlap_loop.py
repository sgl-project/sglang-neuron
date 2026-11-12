# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""Upstream SGLang's PD decode PP loop in the overlap order, where the serving policy runs it.

Upstream's ``event_loop_pp_disagg_decode`` launches a micro-batch's next step only after processing the previous one,
whose outputs come back from the last stage a forward later, so the first stage's host waits a forward per step. In
the overlap order iteration i serves micro-batch i % N (N = pp_size), and every stage:

1. receives requests, takes this iteration's retract, prealloc and transfer ids (intersected along the stages, as
   upstream does), and schedules the micro-batch's next step while its previous step (iteration i - N) is still
   unprocessed; newly transferred requests join through a prebuilt batch merged into the running batch;
2. launches it on run_batch's overlap branch (sglang_patches/pp_overlap_schedule) and hands its proxy on at once,
   committing the previous proxy send only after the launch;
3. on the last rank, sends rank 0 the outputs of iteration i - 1 once their host copy is done;
4. processes iteration i - N - 1 (sglang_patches/pp_loop_state.process_step);
5. passes the consensus ids on, then receives and applies the next micro-batch's;
6. hands the next stage this iteration's requests and ids.

A send completes only when its receiver takes it, so a non-last stage must do step 4 before step 5: its previous stage
sends the consensus only after committing the output send, and the other order deadlocks. Every stage applies the same
consensus and processes the same iterations at the same points, so all stages schedule the same batches; a request
that finished runs one extra step whose token is discarded, as in SGLang's overlap scheduler. No host waits on a
forward with nothing queued behind it: the first stage reads iteration i - N's tokens from the last stage's relay over
the PP seam, and the last rank reads iteration i - 1's outputs only after launching iteration i. Launched, unprocessed
steps live in ``pp_overlap_records`` (sglang_patches/pp_loop_state), which the idle and drain checks count.

Where it departs from upstream's loop, and why that is safe on Neuron:

* Processing runs N + 1 iterations behind the launch; ``last_mbs`` holds a micro-batch's latest launch, like the
  overlap scheduler's ``last_batch``.
* It calls ``run_batch`` itself, with ``_pp_launch_batch``'s trace stamps and metadata but not its stream fences: the
  overlap branch waits for the schedule stream itself, and the other fences order a PP comm stream that a Neuron stage
  does not have (``init_pp_loop_state`` makes one on CUDA only).
* The proxy goes on right after the launch and the previous proxy send is committed after it (upstream commits before
  the launch and sends at the end of the step). The proxy is fresh host descriptor tensors built at every launch, so
  its send waits for no forward and no later launch overwrites it; on CUDA it is a view of buffers the next graph
  replay overwrites.
* Outputs are host tensors: the last rank sends them from its result's host copy, the other ranks process them without
  upstream's receive event and copy stream, and the last rank processes its own result instead of getting it back
  round the ring.
* No prebuilt checks: ``get_next_disagg_decode_batch_to_run`` merges a prebuilt batch itself, so the batch it plans is
  never one.
* The server counts as idle only once every launched step is processed. An async batch depth, which
  distributed.pipeline refuses at launch, is refused again here.

Upstream: the order is generic but relies on those Neuron properties (host-tensor PP messages, no PP comm stream, a
fresh proxy per launch); an upstream version would keep upstream's commit order and fences where they do not hold, as
on CUDA.

Targets, md5-pinned against upstream SGLang 43881a7: ``SchedulerPPMixin.event_loop_pp_disagg_decode`` (wrapped), and
the functions of ``_TARGETS_MD5`` the loop calls or relies on."""

from __future__ import annotations

from typing import Dict, List, Optional

_MIXIN = "sglang.srt.managers.scheduler_pp_mixin"
_SCHEDULER = "sglang.srt.managers.scheduler"
_DECODE = "sglang.srt.disaggregation.decode"
LOOP = f"{_MIXIN}.SchedulerPPMixin.event_loop_pp_disagg_decode"
# (module, qualified name): md5 of the source; the wrapped loop, then the functions it calls or relies on
_TARGETS_MD5 = {
    (_MIXIN, "SchedulerPPMixin.event_loop_pp_disagg_decode"): "3f82c7f86d5203839efa8ecac65a5628",
    (_SCHEDULER, "Scheduler.increment_scheduler_iterations"): "554e5c0901faedd818bfacbb343fff00",
    (_SCHEDULER, "Scheduler.ingest_requests"): "9519de2263bcfd52e322bce6372c0006",
    (_SCHEDULER, "Scheduler.is_fully_idle"): "28e67960ff8d30595eab2437fe29b45e",
    (_SCHEDULER, "Scheduler.on_idle"): "905885a538273f5271e11bf93c08848c",
    (_MIXIN, "SchedulerPPMixin._pp_pd_get_retract_ids"): "d74a5d2e2e653662a54672a0b540e712",
    (_MIXIN, "SchedulerPPMixin._pp_pd_get_prealloc_ids"): "62323f5f14a69689a7120e97f6c8c69a",
    (_MIXIN, "SchedulerPPMixin._pp_pd_get_decode_transferred_ids"): "492efd2231275e3582b3377d5cd24db9",
    (_MIXIN, "SchedulerPPMixin._pp_pd_send_consensus_bootstrapped_ids"): "7d8e1b4b14486de2728ce9da51243c6d",
    (_MIXIN, "SchedulerPPMixin._pp_pd_send_consensus_release_ids"): "d14e06df2f544720282bd7280956f0f4",
    (_MIXIN, "SchedulerPPMixin.process_retract_queue"): "a5f2e95eda63fe05abe46ab32f924d60",
    (_MIXIN, "SchedulerPPMixin.process_prealloc_queue"): "f952fddd0ba96eee5f13256db47cda80",
    (_MIXIN, "SchedulerPPMixin.process_decode_transfer_queue"): "e99cc696dba776b5fe5143bc9dedd4a4",
    (_MIXIN, "SchedulerPPMixin._pp_commit_comm_work"): "935f872964c601a49544236d3ae8604f",
    (_MIXIN, "SchedulerPPMixin._pp_send_pyobj_to_next_stage"): "d79ccfa66d862c8bba0470b4fb4e5fc0",
    (_MIXIN, "SchedulerPPMixin._pp_recv_pyobj_from_prev_stage"): "cbfe424507dbff5123a0c09d0af49fb7",
    (_MIXIN, "SchedulerPPMixin._pp_recv_proxy_tensors"): "6a8973312df0da42999f2585147e86c4",
    (_MIXIN, "SchedulerPPMixin._pp_send_dict_to_next_stage"): "a68ce1f2295efa4afa71e863298d8f9a",
    (_MIXIN, "SchedulerPPMixin._pp_prepare_tensor_dict"): "d29fe72b57e526eda7c570e320d40051",
    (_MIXIN, "_pp_snapshot_forward_batch"): "1d94eefb02e5fa3764def1edde59fb2b",
    # merges a prebuilt batch into the running batch, so the batch it plans is never a prebuilt one
    (_DECODE, "SchedulerDisaggregationDecodeMixin.get_next_disagg_decode_batch_to_run"):
        "3b90210c4c28d3c4cdb89f8f5e5a815e",
    # an empty list is still one message, so every stage's control sends and receives stay paired
    ("sglang.srt.utils.common", "point_to_point_pyobj"): "7ea53e9802a0b901ae3caa275a586d75",
}


def _overlap_order() -> bool:
    """Whether the serving policy runs the PD decode PP loop in the overlap order: a pipelined decode under SGLang's
    overlap schedule."""
    from sglang_neuron import serving_policy

    policy = serving_policy.current()
    return policy.role == "decode" and policy.pp_size > 1 and policy.overlap_schedule


def event_loop_pp_disagg_decode(original_fn, self):
    """AROUND SchedulerPPMixin.event_loop_pp_disagg_decode: the overlap order where the serving policy runs it."""
    if not _overlap_order():
        return original_fn(self)
    from sglang.srt.utils import DynamicGradMode

    # under upstream's decorator, like upstream's loop (inference or no-grad mode)
    return DynamicGradMode()(pp_decode_overlap_loop)(self)


def _send_outputs(self, batch, result) -> None:
    """The last rank: send rank 0 a step's outputs once its host copy is done."""
    import torch

    result.copy_done.synchronize()
    self._pp_commit_comm_work(self.send_output_work)
    with torch.profiler.record_function("send_res_dict_to_next_stage"):
        self.send_output_work = self._pp_send_dict_to_next_stage(
            self._pp_prepare_tensor_dict(result, batch), async_send=True, msg_type="output")


def pp_decode_overlap_loop(self) -> None:
    """The PD decode PP loop in the overlap order (the module docstring has its steps)."""
    import torch

    from sglang.srt.managers.scheduler_pp_mixin import PPBatchMetadata, _pp_snapshot_forward_batch
    from sglang.srt.observability.req_time_stats import set_time_batch
    from sglang.srt.runtime_context import get_disagg, get_parallel

    from sglang_neuron.sglang_patches import pp_loop_state

    self.init_pp_loop_state()
    pp_size, pp_rank = get_parallel().pp_size, get_parallel().pp_rank
    if self.pp_loop_size != pp_size:
        raise RuntimeError("the overlap-order PD decode PP loop runs pp_size micro-batches (no async batch depth)")
    n = self.pp_loop_size
    is_last = self.pp_group.is_last_rank
    forwards = pp_rank < pp_size - 2
    records = self.pp_overlap_records
    rmbs: List = [None] * n
    pmbs: List = [None] * n
    tmbs: List = [None] * n
    consensus_retract_rids: Optional[List[str]] = None
    consensus_prealloc_rids: Optional[List[str]] = None
    release_rids: Optional[Dict[str, bool]] = None  # consensus {rid: failed}
    send_retract_work, send_prealloc_work, send_transfer_work = [], [], []
    send_consensus_retract_work, send_consensus_prealloc_work, send_release_work = [], [], []

    while True:
        server_is_idle = True
        for mb_id in range(n):
            self.increment_scheduler_iterations()
            self.running_batch = self.running_mbs[mb_id]
            self.last_batch = self.last_mbs[mb_id]
            # the last rank's consensus of this iteration's ids goes to rank 0 now, and every stage receives the
            # consensus of the next micro-batch's ids
            next_first_rank_mb_id = (mb_id + pp_size) % n
            next_mb_id = (mb_id + 1) % n

            recv_reqs = self.ingest_requests()
            if not is_last:
                self._pp_commit_comm_work(self.send_req_work)

            retract_rids = self._pp_pd_get_retract_ids(mb_id)
            rmbs[mb_id] = retract_rids
            self._pp_commit_comm_work(send_retract_work)
            prealloc_rids = self._pp_pd_get_prealloc_ids()
            pmbs[mb_id] = prealloc_rids
            self._pp_commit_comm_work(send_prealloc_work)
            transferred_rids = self._pp_pd_get_decode_transferred_ids()
            tmbs[mb_id] = transferred_rids
            self._pp_commit_comm_work(send_transfer_work)

            with torch.profiler.record_function("get_next_batch_to_run"):
                plan = self.get_next_disagg_decode_batch_to_run(running_batch=self.running_batch)
            self.running_batch = plan.running_batch
            self.running_mbs[mb_id] = self.running_batch
            cur_batch = plan.batch_to_run
            self.mbs[mb_id] = cur_batch
            self.cur_batch_for_debug = cur_batch
            record = None
            if cur_batch:
                server_is_idle = False
                pp_proxy_tensors = self._pp_recv_proxy_tensors()
                with torch.profiler.record_function("run_batch"):
                    set_time_batch(cur_batch.reqs, "set_run_batch_cpu_start_time", trace_only=True)
                    result = self.run_batch(cur_batch, pp_proxy_tensors)
                    set_time_batch(cur_batch.reqs, "set_run_batch_cpu_end_time", trace_only=True,
                                   attrs={"pp_mb_id": mb_id})
                self.mb_metadata[mb_id] = PPBatchMetadata(
                    can_run_cuda_graph=result.can_run_cuda_graph,
                    fwd_batch=_pp_snapshot_forward_batch(cur_batch),
                    verify_out_cache_loc=result.spec_verify_out_cache_loc,
                    fpm_timing=result.fpm_timing,
                )
                record = (cur_batch.copy(), result, self.mb_metadata[mb_id])
                if not is_last:
                    # commit the previous proxy send after this launch, so a launch never waits for the next stage
                    # to take the last proxy; at most one proxy send is in flight per stage pair
                    self._pp_commit_comm_work(self.send_proxy_work)
                    with torch.profiler.record_function("send_proxy_dict_to_next_stage"):
                        self.send_proxy_work = self._pp_send_dict_to_next_stage(
                            result.pp_hidden_states_proxy_tensors.tensors, async_send=True, msg_type="proxy")
            if is_last and records and records[-1] is not None:
                _send_outputs(self, *records[-1][:2])
            records.append(record)
            if len(records) > n + 1:
                oldest = records.popleft()
                if oldest is not None:
                    with torch.profiler.record_function("process_batch_result"):
                        pp_loop_state.process_step(self, *oldest, is_last, forwards)

            send_consensus_retract_work, consensus_retract_rids = self._pp_pd_send_consensus_bootstrapped_ids(
                rmbs, next_first_rank_mb_id, consensus_retract_rids, retract_rids)
            send_consensus_prealloc_work, consensus_prealloc_rids = self._pp_pd_send_consensus_bootstrapped_ids(
                pmbs, next_first_rank_mb_id, consensus_prealloc_rids, prealloc_rids)
            send_release_work, release_rids = self._pp_pd_send_consensus_release_ids(
                tmbs, next_first_rank_mb_id, release_rids, transferred_rids)
            if get_disagg().disaggregation_decode_enable_offload_kvcache:
                self.decode_offload_manager.check_offload_progress()
            next_consensus_retract_rids = None
            next_consensus_prealloc_rids = None
            next_release_rids = None
            if rmbs[next_mb_id] is not None:
                next_consensus_retract_rids = self.process_retract_queue(self._pp_recv_pyobj_from_prev_stage())
            self._pp_commit_comm_work(send_consensus_retract_work)
            if pmbs[next_mb_id] is not None:
                next_consensus_prealloc_rids = self.process_prealloc_queue(self._pp_recv_pyobj_from_prev_stage())
            self._pp_commit_comm_work(send_consensus_prealloc_work)
            if tmbs[next_mb_id] is not None:
                next_release_rids = self.process_decode_transfer_queue(self._pp_recv_pyobj_from_prev_stage())
            self._pp_commit_comm_work(send_release_work)

            self.last_mbs[mb_id] = cur_batch

            if not is_last:
                self.send_req_work = self._pp_send_pyobj_to_next_stage(recv_reqs, async_send=True)
                send_retract_work = self._pp_send_pyobj_to_next_stage(retract_rids, async_send=True)
                send_prealloc_work = self._pp_send_pyobj_to_next_stage(prealloc_rids, async_send=True)
                send_transfer_work = self._pp_send_pyobj_to_next_stage(transferred_rids, async_send=True)

            release_rids = next_release_rids
            consensus_retract_rids = next_consensus_retract_rids
            consensus_prealloc_rids = next_consensus_prealloc_rids
            self.running_batch.batch_is_full = False

        queue_size = (len(self.waiting_queue) + len(self.disagg_decode_transfer_queue.queue)
                      + len(self.disagg_decode_prealloc_queue.queue))
        if get_disagg().disaggregation_decode_enable_offload_kvcache:
            queue_size += len(self.decode_offload_manager.ongoing_offload)
        if server_is_idle and not any(r is not None for r in records):
            self._sched_idled = True
            if queue_size == 0:
                self.on_idle()


def register(registry, around) -> None:
    from sglang_neuron.sglang_patches import source_guard

    source_guard.check_targets(_TARGETS_MD5, __name__)
    registry.register(LOOP, event_loop_pp_disagg_decode, around)
