# SPDX-License-Identifier: Apache-2.0
"""Compute ahead on a pipelined PD prefill: compute a request before its decode grants it a seat, keep its KV in the
prefill's pages, and push it once the decode publishes the destination.

SGLang's prefill starts a request only after its decode has a free seat, so with more requests than seats each seat
idles through a prefill; its optimistic prefill, the fix, is off under PP because a not-yet-bootstrapped request would
change state at different points on each stage. Here every such change happens at one decision point, the PP consensus
in ``pop_bootstrapped``, so all stages agree; and a row reserve (``pp_prefill_admission_floor``) keeps a row free for
seat-granted requests. It runs where the catalog declares it (``ServingPolicy.prefill_compute_ahead``)."""

from __future__ import annotations

_PREFILL = "sglang.srt.disaggregation.prefill"
_UTILS = "sglang.srt.disaggregation.utils"
_COMMON = "sglang.srt.disaggregation.common.conn"
_NIXL = "sglang.srt.disaggregation.nixl.conn"
_MIXIN = "sglang.srt.managers.scheduler_pp_mixin"
_SCHEDULER = "sglang.srt.managers.scheduler"
_POLICY = "sglang.srt.managers.schedule_policy"
POP = (_PREFILL, "PrefillBootstrapQueue.pop_bootstrapped")
ADMIT = (_PREFILL, "PrefillBootstrapQueue.can_prefill_before_bootstrap")
CHECK = (_PREFILL, "SchedulerDisaggregationPrefillMixin.check_bootstrap")
YIELD = (_PREFILL, "SchedulerDisaggregationPrefillMixin.has_bootstrapped_waiting_req")
ORDER = (_POLICY, "SchedulePolicy.calc_priority")
# (module, qualified name) of each upstream source, pinned in upstream_pins.json
_WRAPPED = (
    POP,
    ADMIT,
    CHECK,
    YIELD,
    ORDER,
)
_RELIED_ON = (
    # the bootstrap queue: a request leaves it pending and is finalized once, with its metadata buffer
    (_PREFILL, "PrefillBootstrapQueue.create_sender"),
    (_PREFILL, "PrefillBootstrapQueue.finalize_bootstrap"),
    (_PREFILL, "PrefillBootstrapQueue.ensure_metadata_buffer"),
    (_PREFILL, "_uses_write_through_cache"),
    (_PREFILL, "SchedulerDisaggregationPrefillMixin.handle_pending_bootstrap"),
    (_PREFILL, "SchedulerDisaggregationPrefillMixin.handle_bootstrap_failure"),
    # the chunk, result and push paths a pending request takes
    (_PREFILL, "SchedulerDisaggregationPrefillMixin.process_prefill_chunk"),
    (_PREFILL, "SchedulerDisaggregationPrefillMixin.process_batch_result_disagg_prefill"),
    (_PREFILL, "SchedulerDisaggregationPrefillMixin._retire_aborted_prefill_result"),
    (_PREFILL, "SchedulerDisaggregationPrefillMixin.process_disagg_prefill_inflight_queue"),
    (_PREFILL, "SchedulerDisaggregationPrefillMixin.send_kv_chunk"),
    (_PREFILL, "SchedulerDisaggregationPrefillMixin._send_kv_chunk"),
    # the PP consensus and the release consensus, each stage reading its own status
    (_UTILS, "poll_and_all_reduce_prefill_pp"),
    (_UTILS, "pp_sync_polls"),
    (_COMMON, "CommonKVSender.poll_pp_consensus"),
    (_COMMON, "CommonKVSender._poll_with_status"),
    (_COMMON, "CommonKVManager.check_status_pp_consensus"),
    (_COMMON, "CommonKVSender._check_bootstrap_timeout"),
    (_COMMON, "CommonKVManager.update_status"),
    (_COMMON, "CommonKVSender.abort"),
    (_MIXIN, "SchedulerPPMixin._pp_pd_get_prefill_transferred_ids"),
    (_MIXIN, "SchedulerPPMixin.get_rids"),
    (_COMMON, "CommonKVSender.poll"),
    (_NIXL, "NixlKVManager.check_status"),
    # the scheduler loop's order, and the abort path
    (_MIXIN, "SchedulerPPMixin.event_loop_pp_disagg_prefill"),
    (_MIXIN, "SchedulerPPMixin.process_bootstrapped_queue"),
    (_SCHEDULER, "Scheduler._get_new_batch_prefill_raw"),
    (_SCHEDULER, "Scheduler.abort_request"),
    (_SCHEDULER, "Scheduler.process_pending_chunked_abort"),
    (_SCHEDULER, "Scheduler.collect_inflight_reqs"),
    (_SCHEDULER, "Scheduler._collect_inflight_batches"),
)

# where a pending request is when the consensus decides it
WAITING, CHUNKED, LAUNCHED, PARKED = "waiting", "chunked", "launched", "parked"


def reserve(pp_size: int) -> int:
    """The prefill rows a not-yet-bootstrapped request may never take: the requests a pipelined prefill holds at once
    with its stages back to back (``pipeline.pp_prefill_admission_floor``)."""
    from sglang_neuron.distributed.pipeline import pp_prefill_admission_floor

    return pp_prefill_admission_floor(pp_size)


def ahead_room(max_running: int, pp_size: int, pending: int) -> int:
    """How many more Bootstrapping requests the prefill may take: its rows less the reserve, less the ``pending``
    requests already holding or about to hold one."""
    return max(int(max_running) - reserve(pp_size) - int(pending), 0)


def pending_places(scheduler) -> list:
    """``(place, req)`` of every pending request the consensus decides, in an order every stage shares: the waiting
    queue's, then the chunked request, then the launched batches' (``_collect_inflight_batches``, in its order), then
    the in-flight queue's parked ones, each request once at its first place. A chunked request whose client abort is
    pending is ``process_pending_chunked_abort``'s, in the chunked slot and in its launched batches alike."""
    places = [(WAITING, req) for req in scheduler.waiting_queue if req.pending_bootstrap]
    chunked = scheduler.chunked_req
    seen = {id(req) for _, req in places} | {id(req) for req in scheduler.disagg_prefill_inflight_queue}
    if chunked is not None:
        seen.add(id(chunked))
        if chunked.pending_bootstrap and getattr(scheduler, "_pending_chunked_abort_req", None) is not chunked:
            places.append((CHUNKED, chunked))
    for batch in scheduler._collect_inflight_batches():
        for req in batch.reqs if batch is not None else ():
            if req.pending_bootstrap and id(req) not in seen:
                seen.add(id(req))
                places.append((LAUNCHED, req))
    places += [(PARKED, req) for req in scheduler.disagg_prefill_inflight_queue if req.pending_bootstrap]
    return places


def pending_count(scheduler) -> int:
    """The pending requests holding or about to hold a prefill row: in the waiting queue, the chunked slot, a launched
    batch (``collect_inflight_reqs``, with the overlap loop's records) or parked."""
    reqs = [*scheduler.waiting_queue, *scheduler.disagg_prefill_inflight_queue, *scheduler.collect_inflight_reqs()]
    if scheduler.chunked_req is not None:
        reqs.append(scheduler.chunked_req)
    return len({id(req) for req in reqs if req.pending_bootstrap})


def fail_room(req) -> None:
    """Fail ``req``'s room in this stage's own status, as the consensus read it, so the release consensus, which reads
    each stage's own poll, names the room. A repeat, or a room already Failed here, is a
    no-op. No reason is recorded: where the room failed it has its own, and elsewhere SGLang reports a failure from
    another rank, as for a room the bootstrap queue's consensus fails."""
    from sglang.srt.disaggregation.base.conn import KVPoll

    req.disagg_kv_sender.kv_mgr.update_status(req.bootstrap_room, KVPoll.Failed)


def apply_poll(scheduler, place: str, req, poll) -> bool:
    """Apply one consensus poll to a pending request at ``place``; True if the request leaves the waiting queue."""
    from sglang.srt.disaggregation.base.conn import KVPoll

    if poll == KVPoll.WaitingForInput:
        if place in (WAITING, LAUNCHED):
            # its KV is not computed yet (waiting) or is being written (launched): only the finalize, with the metadata
            # buffer its admission took (as SGLang's resolve_waiting_queue_bootstrap); a launched one's result then
            # checkpoints and sends it
            scheduler.disagg_prefill_bootstrap_queue.finalize_bootstrap(req)
        elif scheduler.handle_pending_bootstrap(req, poll) and place == PARKED:
            # its last result is processed, so this stage's forwards of its KV have run
            scheduler.send_kv_chunk(req, last_chunk=True)
        return False
    if poll == KVPoll.Failed:
        if place in (LAUNCHED, PARKED):
            # its result parks it, and the release consensus concludes it once every stage's own poll reads it Failed
            fail_room(req)
            return False
        scheduler.handle_bootstrap_failure(req)
        if place == CHUNKED:
            scheduler.chunked_req = None
        return place == WAITING
    if poll != KVPoll.Bootstrapping:
        # a pending room has sent nothing, so no stage's status passes WaitingForInput
        raise RuntimeError(f"unexpected consensus poll {poll} for pending request {req.rid} ({place})")
    return False


def consensus_point(queue) -> None:
    """Poll the pending requests outside the bootstrap queue through the stages' consensus and apply the polls, on
    every stage at the same loop point, so all stages change a pending request's state together."""
    scheduler = queue.scheduler
    places = pending_places(scheduler)
    if not places:
        return
    from sglang.srt.disaggregation.utils import poll_and_all_reduce_prefill_pp

    work = getattr(queue, "compute_ahead_sync_work", None)
    if work is None:
        work = queue.compute_ahead_sync_work = []
    polls = poll_and_all_reduce_prefill_pp([req.disagg_kv_sender for _, req in places], scheduler.attn_cp_cpu_group,
                                           scheduler.attn_tp_cpu_group, scheduler.pp_group, queue.pp_rank,
                                           queue.pp_size, work)
    leaving = {id(req) for (place, req), poll in zip(places, polls, strict=True)
               if apply_poll(scheduler, place, req, poll)}
    if leaving:
        scheduler.waiting_queue = [req for req in scheduler.waiting_queue if id(req) not in leaving]


def pop_bootstrapped(original_fn, self, *args, **kwargs):
    """AROUND PrefillBootstrapQueue.pop_bootstrapped: the pending requests' consensus first, then SGLang's pop, which
    admits Bootstrapping requests up to the room the reserve leaves (``can_prefill_before_bootstrap``)."""
    consensus_point(self)
    scheduler = self.scheduler
    self.compute_ahead_room = ahead_room(scheduler.max_running_requests, self.pp_size, pending_count(scheduler))
    try:
        return original_fn(self, *args, **kwargs)
    finally:
        self.compute_ahead_room = None


def can_prefill_before_bootstrap(original_fn, self, req):
    """AROUND PrefillBootstrapQueue.can_prefill_before_bootstrap: inside the pop, a Bootstrapping request is taken
    while the pop's room lasts and upstream's write-through and SWA condition holds. Upstream's own answer (optimistic
    prefill, 0 attempts at PP) never applies.

    The pop's ensure_metadata_buffer, right after, may still keep the request in the queue when no buffer is free. The
    room then counts a request the pop did not take, which costs nothing: buffers are freed only outside the pop, and
    no queued request holds one (only a requeue leaves one with its buffer, and compute ahead never requeues), so no
    later request of that pop is taken either. The next pop counts the room afresh."""
    room = getattr(self, "compute_ahead_room", None)
    if room is None:
        # outside the wrapped pop, which in the pinned SGLang is its only caller
        return original_fn(self, req)
    from sglang.srt.disaggregation.prefill import _uses_write_through_cache

    if room <= 0:
        return False
    if (_uses_write_through_cache(self.scheduler.tree_cache) and req.swa_branching_seqlen is not None
            and req.swa_branching_seqlen > req.kv.cache_protected_len):
        return False
    self.compute_ahead_room = room - 1
    return True


def check_bootstrap(original_fn, self, req):
    """AROUND SchedulerDisaggregationPrefillMixin.check_bootstrap: a pending request reads not bootstrapped without a
    poll of the stage's own; the consensus point decides it."""
    if req.pending_bootstrap:
        return False
    return original_fn(self, req)


def has_bootstrapped_waiting_req(original_fn, self):
    """AROUND SchedulerDisaggregationPrefillMixin.has_bootstrapped_waiting_req: False, so a chunked request never
    yields to a bootstrapped one; the chunked yield is its only reader."""
    return False


def calc_priority(original_fn, self, waiting_queue, *args, **kwargs):
    """AROUND SchedulePolicy.calc_priority: after the policy's order, the seat-granted requests first, each part in
    that order, so a granted seat never waits behind compute-ahead work."""
    result = original_fn(self, waiting_queue, *args, **kwargs)
    granted = [req for req in waiting_queue if not req.pending_bootstrap]
    if len(granted) != len(waiting_queue):
        waiting_queue[:] = granted + [req for req in waiting_queue if req.pending_bootstrap]
    return result


HOOKS = {
    POP: pop_bootstrapped,
    ADMIT: can_prefill_before_bootstrap,
    CHECK: check_bootstrap,
    YIELD: has_bootstrapped_waiting_req,
    ORDER: calc_priority,
}


def register(registry, around) -> None:
    """A pipelined PD prefill with compute ahead (disaggregation.register_hooks, by the serving policy)."""
    from sglang_neuron.sglang_patches import source_guard

    source_guard.check_pinned((*_WRAPPED, *_RELIED_ON), __name__)
    for (module, qualname), hook in HOOKS.items():
        registry.register(f"{module}.{qualname}", hook, around)
