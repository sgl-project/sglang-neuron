# SPDX-License-Identifier: Apache-2.0
"""The DP identity of a disaggregated DP-attention decode, and the control fence that makes it safe.

Under DP attention, SGLang exchanges batch metadata across the DP ranks every step (the MLP sync), so that every rank
picks the same forward mode and graph shape. On a decode with one batch bucket, one context bucket and a fixed token
width, the exchange always gives the same answer. The DP identity takes that answer locally on each rank and skips the
exchange. Every rank then launches every iteration, a dummy step when it has no work, so the k-th graph of each rank
meets the k-th graph of every other rank in their collectives. A rank that enters a collective its peers skip waits
there until the watchdog fires, so ``proof_checks`` and ``fence_checks`` check the identity's conditions before the
first step, and a decode that misses one is refused.

A control message that stops launches or runs a host collective (pause, continue, memory release) must take effect at
the same iteration on every rank. Otherwise the ranks still launching wait in their next graph for the ones that
stopped. Every rank receives the same control sequence, but at different iterations. So every ``FENCE_EVERY``
iterations, and every iteration while paused or idle, the ranks exchange how many control messages each has received,
and every rank delivers up to the lowest count. Under the identity every rank runs one iteration per graph, so all ranks
reach that exchange at the same iteration.

The same exchange gates an idle fleet. When no rank has work for a graph, every rank stops launching dummies at that
iteration, and all resume together once any rank has work again. Without this, an idle fleet spins dummy graphs
forever, and a few ranks stalled on the host let the others queue graphs until the runtime's queue overflows. The
exchange also carries the latest deadline of a deferred ``flush_cache``, so every rank flushes, or gives the flush up,
at the same iteration.

Where it can, the exchange runs through one shared-memory segment of each node, joined across nodes by the node leaders
(node_shm_min). A gloo all-reduce is a TCP ring that releases its ranks one after another, so the first step after an
idle fleet would wait for the last rank the ring releases."""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_UNRESOLVED = object()
_proof = _UNRESOLVED  # the decode batch bucket, once proven

_DECODE = "sglang.srt.disaggregation.decode"
_DP_ATTN = "sglang.srt.managers.scheduler_components.dp_attn"
_RECEIVER = "sglang.srt.managers.scheduler_components.request_receiver"
_UPSTREAM_MD5 = {
    # md5-pinned against upstream SGLang 43881a7 (source_guard). First, the sources the identity and the fence replace:
    # the fence redoes the work/control split of _broadcast_reqs_across_ranks, and the identity writes the fields that
    # _update_gather_batch would set from the exchange.
    (_RECEIVER, "SchedulerRequestReceiver._broadcast_reqs_across_ranks"): "2c89050881db2b90eb39e316e32cd8c6",
    (_DP_ATTN, "prepare_mlp_sync_batch_raw"): "68b96f570ba15b939509a35f48793127",
    (_DP_ATTN, "_update_gather_batch"): "74fa1b927e958161b67d3cec74a0fb9f",
    # Then the decode loop the fence counts on: one ingest_requests -> recv_requests -> broadcast per iteration, a
    # `continue` before the batch is picked while paused, and at most one MLP sync per picked batch.
    (_DECODE, "SchedulerDisaggregationDecodeMixin.event_loop_overlap_disagg_decode"): "246b69291bfd47d63d0126dea073abb5",
    (_DECODE, "SchedulerDisaggregationDecodeMixin.get_next_disagg_decode_batch_to_run"):
        "3b90210c4c28d3c4cdb89f8f5e5a815e",
    # and the reason the MLP sync never sees a PREBUILT batch: get_next_disagg_decode_batch_to_run merges the landed
    # requests into the running batch first, and update_running_batch's prepare_for_decode makes it a DECODE batch
    ("sglang.srt.managers.scheduler", "Scheduler.update_running_batch"): "0112f5abf0ea516834bd424c362c8af4",
    ("sglang.srt.managers.scheduler", "Scheduler.ingest_requests"): "9519de2263bcfd52e322bce6372c0006",
    (_RECEIVER, "SchedulerRequestReceiver.recv_requests"): "da596a8026a2776644a4a07187b3c934",
    (_DP_ATTN, "SchedulerDPAttnAdapter.maybe_prepare_mlp_sync_batch"): "cf9d2b579621d4e6c55c129a445841ef",
}

FENCE_EVERY = 16
_RENDEZVOUS: dict = {}  # id(cpu group) -> its shared-memory segment, or None for the gloo all-reduce


class _FenceState:
    def __init__(self):
        self.it = 0           # decode-loop iterations, the same on every rank under the identity
        self.paused = False   # flips on every rank at the same delivery
        self.pending = []     # control requests received, not yet delivered
        self.received = 0
        self.delivered = 0    # the same on every rank after each fence
        self.idle = False     # no rank had work at the last rendezvous
        self.flush_deadline = 0  # 1 + the latest timed flush whose deadline marker reached this rank, 0 for none
        self.scheduler = None


_fence = _FenceState()


def _runtime(model_runner):
    """The runtime of the decode graphs, or None before the model has built it."""
    model = getattr(model_runner, "model", None)
    if getattr(model, "runtime", None) is None:
        return None
    return model._runtime_for("decode")


def proof_checks(server_args, rt, env=os.environ) -> dict:
    """The identity's static conditions, by name. They are the same on every rank, so all ranks pass or fail
    together."""
    from sglang_neuron import serving_policy
    from sglang_neuron.backend import device_io
    from sglang_neuron.speculative.neuron_mtp import is_neuron_mtp

    buckets = list(getattr(getattr(rt, "buckets", None), "num_seqs_buckets", ()) or ())
    max_queued = int(env.get("NEURON_RT_XU_COMPUTE_MAX_QUEUED_REQUESTS", "32"))
    spec = getattr(server_args, "speculative_algorithm", None)
    return {
        "pure_decode": str(getattr(server_args, "disaggregation_mode", "null")) == "decode"
        and getattr(rt, "graphs", None) == "decode",
        "one_seqs_bucket": len(buckets) == 1,
        "dp_lockstep": bool(getattr(rt, "dp_lockstep", False)),
        "fixed_token_width": bool(buckets)
        and int(getattr(rt, "decode_token_width", 0) or 0) >= buckets[-1],
        # no speculative decoding; a decode kernel's own MTP step, one fixed-shape graph a step as a plain decode
        # (speculative/neuron_mtp.py); or the algorithm the model's own speculative worker serves
        # (serving_policy.speculative_algorithm), which runs every step as a DECODE or IDLE batch at the MLP sync
        "no_spec": spec in (None, "NONE", "none") or is_neuron_mtp(spec) or (
            str(spec).upper() == serving_policy.current().speculative_algorithm),
        "no_tbo": not getattr(server_args, "enable_two_batch_overlap", False),
        # a rank keeps up to device_io's in-flight bound of graphs launched (SGLANG_NEURON_MAX_INFLIGHT_LAUNCHES,
        # 2 by default and the result lag's L on the giga-kernel decode), so the runtime must queue more than that
        "queue_headroom": device_io._max_inflight() < max_queued,
    }


def fence_checks(server_args) -> dict:
    """The launch flags the control fence needs, keyed by flag."""
    return {
        # every DP leader receives the control requests itself (by default only the first does), so every rank holds
        # the same sequence
        "--enable-dp-attention-local-control-broadcast": bool(server_args.enable_dp_attention_local_control_broadcast),
        "--scheduler-recv-interval 1": server_args.scheduler_recv_interval == 1,
        # with --sleep-on-idle, an idle leader naps in its poller while another rank's KV lands, and the fleet's resume
        # waits for it
        "no --sleep-on-idle": not server_args.sleep_on_idle,
    }


def _resolve_proof(model_runner):
    from sglang.srt.layers.dp_attention import world_dp_gather_enabled

    rt = _runtime(model_runner)
    checks = proof_checks(model_runner.server_args, rt)
    checks["no_elastic_world"] = not world_dp_gather_enabled()
    checks.update(fence_checks(model_runner.server_args))
    summary = ", ".join(f"{k}={v}" for k, v in checks.items())
    if not all(checks.values()):
        # static config, so every rank fails here at its first step and none mid-run
        raise RuntimeError(f"the serving policy runs the DP identity on this decode, but it cannot hold: {summary}")
    bucket = rt.buckets.num_seqs_buckets[0]
    rt.dp_identity_bucket = bucket
    logger.info("[neuron] DP identity on, bucket=%d (%s)", bucket, summary)
    return bucket


def check_step(local_batch, bucket: int) -> None:
    if local_batch is None:
        return
    mode = local_batch.forward_mode
    if not (mode.is_decode() or mode.is_idle()):
        raise RuntimeError(f"DP identity: {mode} batch on a pure decode server")
    if mode.is_decode() and local_batch.batch_size() > bucket:
        raise RuntimeError(f"DP identity: batch {local_batch.batch_size()} exceeds bucket {bucket}")


def apply_identity(batch, disable_cuda_graph: bool) -> None:
    """Set the fields _update_gather_batch would write from the exchange, as constants every rank agrees on."""
    batch.global_num_tokens = None  # ForwardBatch then skips the MLP-sync padding
    batch.global_num_tokens_for_logprob = None
    batch.is_extend_in_batch = False
    batch.tbo_split_seq_index = None
    batch.global_forward_mode = None
    batch.can_run_decode_cuda_graph = not disable_cuda_graph
    batch.can_run_dp_prefill_cuda_graph = False
    # the exchange also votes on replaying SGLang's draft and prefill graphs; neither runs here (no prefill, and a
    # speculative decode runs the model's own worker, which launches its own programs)
    batch.can_run_dp_draft_cuda_graph = False
    batch.dp_prefill_cuda_graph_max_prefix_len = 0


def identity_batch(local_batch, get_idle_batch, disable_cuda_graph: bool):
    """The batch this rank launches. Every rank launches every iteration: an IDLE batch when it has no work."""
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    if local_batch is None:
        local_batch = get_idle_batch()
    apply_identity(local_batch, disable_cuda_graph)
    # the same on every rank, for SchedulerRecvSkipper
    local_batch.recv_skipper_forward_mode = ForwardMode.DECODE
    local_batch.dp_cooperation_info = None
    return local_batch


def prepare_mlp_sync_identity(original_fn, local_batch, *args, **kwargs):
    """Replaces dp_attn.prepare_mlp_sync_batch_raw and never calls it: the batch takes the identity's constants instead
    of the exchange's result. The conditions are proven at the first call."""
    global _proof
    if _proof is _UNRESOLVED:
        _proof = _resolve_proof(kwargs["model_runner"])
    check_step(local_batch, _proof)
    if _fence.idle:
        # No rank had work at this iteration's rendezvous, so no rank launches. A rank that launched alone would wait
        # in its graph for peers that never come.
        if local_batch is not None and not local_batch.forward_mode.is_idle():
            raise RuntimeError("DP control fence: a rank built a batch while the fleet is idle")
        return None
    return identity_batch(local_batch, kwargs["get_idle_batch"], kwargs["disable_cuda_graph"])


def _loop_entry_barrier(original_fn, scheduler, *args, **kwargs):
    """Wait for every rank of the TP group at the decode loop's entry. Without this barrier, an early rank's first
    dummy graph waits in its collectives for the slowest scheduler to finish starting up, which can take longer than
    the runtime's execution timeout. After the barrier, every rank starts the fence's iteration count from 0. The loop
    thread first sets a short timer slack, since the shared-memory waits nap in short timed sleeps."""
    import torch.distributed as dist

    from sglang_neuron.distributed.node_shm_min import set_timer_slack

    set_timer_slack()
    dist.barrier(group=scheduler.tp_cpu_group)
    _fence.it = 0
    logger.info("[neuron] DP identity: all ranks entered the decode loop")
    return original_fn(scheduler, *args, **kwargs)


def _broadcast(group, obj, size: int):
    if size == 1:
        return obj
    if getattr(group, "mq_broadcaster", None) is not None:
        return group.broadcast_object(obj)
    from sglang.srt.utils.common import broadcast_pyobj

    return broadcast_pyobj(obj, group.rank, group.cpu_group, src=group.ranks[0])


def fence_due(state=_fence, every=None) -> bool:
    every = FENCE_EVERY if every is None else every
    return state.paused or state.idle or state.it % every == 0


def _rendezvous_shm(cpu_group):
    """The group's shared-memory reduce, or None when the rendezvous runs on the gloo all-reduce: the serving policy
    leaves the segment off, the host does not support it, or the group is not the whole server. On a server over
    several nodes, each node's ranks reduce through their node's segment and the node leaders over gloo
    (node_shm_min.group_reduce). It is built at the group's first rendezvous, which every rank of the group reaches at
    the same iteration."""
    key = id(cpu_group)
    if key in _RENDEZVOUS:
        return _RENDEZVOUS[key]
    import numpy as np
    import torch.distributed as dist

    from sglang_neuron import serving_policy
    from sglang_neuron.distributed.node_shm_min import group_reduce, shm_supported

    _RENDEZVOUS[key] = None
    policy = serving_policy.current()
    ranks = dist.get_process_group_ranks(cpu_group)
    if policy.dp_rendezvous_shm and shm_supported() and len(ranks) == dist.get_world_size():
        _RENDEZVOUS[key] = group_reduce(cpu_group, policy.nnodes, "rdv", capacity=3, dtype=np.int64, op="max")
        logger.info("[neuron] DP rendezvous through shared memory over %d ranks on %d node(s)", len(ranks),
                    policy.nnodes)
    return _RENDEZVOUS[key]


def _rendezvous(cpu_group, received: int, has_work: bool, flush_deadline: int = 0):
    """One MAX reduce over the ranks, returning (the fewest control messages any rank has received, whether any rank
    has work, the latest flush deadline marker). ``received`` goes in negated, so its MAX gives the MIN."""
    import torch
    import torch.distributed as dist

    shm = _rendezvous_shm(cpu_group)
    if shm is not None:
        m = shm.reduce([-received, int(has_work), int(flush_deadline)])
        return int(-m[0]), bool(m[1]), int(m[2])
    t = torch.tensor([-received, int(has_work), int(flush_deadline)], dtype=torch.int64)
    dist.all_reduce(t, op=dist.ReduceOp.MAX, group=cpu_group)
    return int(-t[0].item()), bool(t[1].item()), int(t[2].item())


def flush_deadline_marker() -> type:
    from sglang_neuron.sglang_patches.flush_deadline_markers import FlushDeadlineReached

    return FlushDeadlineReached


def _take_flush_markers(control_reqs):
    """Remove the flush deadline markers from ``control_reqs`` and return the other requests with 1 + the latest timed
    flush the markers name (0 for none). Each DP leader adds these markers by its own clock, so they would sit at
    different places in the ranks' sequences, which the fence delivers by count. Instead the fence takes the latest
    deadline at its rendezvous and delivers one marker for it on every rank."""
    if not control_reqs:
        return control_reqs, 0
    FlushDeadlineReached = flush_deadline_marker()
    kept, latest = [], 0
    for r in control_reqs:
        if isinstance(r, FlushDeadlineReached):
            latest = max(latest, r.flush + 1)
        else:
            kept.append(r)
    return kept, latest


def _controls_key(reqs) -> str:
    """The type and rid of each control request, in order: what the ranks compare at a delivery."""
    return "|".join(f"{type(r).__name__}:{getattr(r, 'rid', '')}" for r in reqs)


def _controls_digest(reqs) -> int:
    import zlib

    return zlib.crc32(_controls_key(reqs).encode()) & 0x7FFFFFFF


def _check_same_controls(cpu_group, due, delivered: int) -> None:
    """Raise unless every rank is about to deliver the same control requests. A request that reached only some DP
    groups would shift every later message by one. One MAX reduce of [digest, -digest] gives the highest and the lowest
    digest, and the ranks agree when the two are equal."""
    d = _controls_digest(due)
    shm = _rendezvous_shm(cpu_group)
    if shm is not None:
        high, neg_low = (int(v) for v in shm.reduce([d, -d]))
    else:
        import torch
        import torch.distributed as dist

        t = torch.tensor([d, -d], dtype=torch.int64)
        dist.all_reduce(t, op=dist.ReduceOp.MAX, group=cpu_group)
        high, neg_low = int(t[0]), int(t[1])
    if high != -neg_low:
        key = _controls_key(due[:32]) + (f"|... {len(due) - 32} more" if len(due) > 32 else "")
        raise RuntimeError(
            f"DP control fence: the ranks hold different control requests at delivery {delivered}+{len(due)}; this "
            f"rank holds [{key}]. Compare this line across the ranks' logs: a control request reached only some DP "
            "leaders, or reached them in another order, and the fence cannot order it fleet-wide")


def scheduler_has_work(scheduler) -> bool:
    """Whether this rank has work that needs a decode graph: a running batch, requests whose KV has landed, or
    retracted requests. Requests still preallocating or receiving their KV are not work yet. Before the scheduler is
    known, the rank counts as having work."""
    if scheduler is None:
        return True
    # upstream's attributes, read without defaults: a renamed one raises instead of reading as no work, which would
    # idle the fleet for good
    return (not scheduler.running_batch.is_empty() or bool(scheduler.waiting_queue)
            or bool(scheduler.disagg_decode_prealloc_queue.retracted_queue))


def hold_prebuilt(original_fn, scheduler, *args, **kwargs):
    """Build no new batch while the fleet is idle. A request whose KV lands meanwhile stays in the waiting queue, the
    next rendezvous counts it as work, and every rank resumes there together."""
    if _fence.idle:
        return None
    return original_fn(scheduler, *args, **kwargs)


def _refuse_rpc_requests(receiver, recv_reqs):
    """Answer every Engine RPC with a failure and drop it. An RPC reaches only one DP leader (the leaders share one
    RPC socket), so the fence could never deliver it on every rank. Refusing it at once keeps its caller from
    waiting."""
    if not recv_reqs:
        return recv_reqs
    from sglang.srt.managers.io_struct import RpcReqInput, RpcReqOutput, sock_send

    kept = []
    for req in recv_reqs:
        if isinstance(req, RpcReqInput):
            logger.error("[neuron] DP control fence: refusing Engine RPC %s (it reaches one DP leader only)",
                         req.method)
            sock_send(receiver.recv_from_rpc, RpcReqOutput(
                success=False, message=f"{req.method}: an Engine RPC is not supported while the DP control fence "
                                       "orders control requests (it reaches one DP leader only)"))
        else:
            kept.append(req)
    return kept


def _groups(receiver):
    """The ranks and groups of a request receiver, from SGLang's runtime context and group registry, since the
    receiver holds neither. Tests patch this function to read them from ``receiver`` instead."""
    from types import SimpleNamespace

    from sglang.srt.distributed.parallel_state import get_attn_tp_group, get_tp_group
    from sglang.srt.runtime_context import get_parallel

    return SimpleNamespace(ps=get_parallel(), tp_group=get_tp_group(), attn_tp_group=get_attn_tp_group())


def fenced_broadcast(original_fn, receiver, recv_reqs, local_reqs=None):
    """Replaces SchedulerRequestReceiver._broadcast_reqs_across_ranks, which it never calls. Work requests reach the DP
    group every iteration. Control requests wait for the next fence, which delivers the part of the control sequence
    that every rank already holds. A DP leader's own aborts (``local_reqs``) go with the work requests, as in SGLang."""
    g = _groups(receiver)
    ps = g.ps
    if ps.attn_tp_rank == 0:
        recv_reqs = _refuse_rpc_requests(receiver, recv_reqs)
        work_reqs, control_reqs = receiver._split_work_and_control_reqs(recv_reqs)
        reqs = ([*work_reqs, *(local_reqs or ())], control_reqs)
    else:
        reqs = None
    # both lists in one broadcast over the DP group
    work_reqs, control_reqs = _broadcast(g.attn_tp_group, reqs, ps.attn_tp_size)
    control_reqs, latest = _take_flush_markers(control_reqs)
    _fence.flush_deadline = max(_fence.flush_deadline, latest)
    _fence.pending.extend(control_reqs)
    _fence.received += len(control_reqs)
    _fence.it += 1
    if not fence_due():
        return work_reqs
    has_work = scheduler_has_work(_fence.scheduler)
    common, any_work, flush_deadline = _rendezvous(g.tp_group.cpu_group, _fence.received, has_work,
                                                   _fence.flush_deadline)
    if _fence.idle != (not any_work):
        logger.info("[neuron] DP control fence: fleet %s at iteration %d", "idle" if not any_work else "active",
                    _fence.it)
        if not any_work:
            # Every rank goes idle here. Retire the launches still queued, so the pages freed behind them return to
            # the pool now.
            from sglang_neuron.model_executor import async_execution

            async_execution.drain_launches()
    _fence.idle = not any_work
    expire = []
    if flush_deadline:
        # a deferred flush's deadline passed on some DP leader: every rank flushes, or gives the flush up, here
        expire, _fence.flush_deadline = [flush_deadline_marker()(flush=flush_deadline - 1)], 0
    n = common - _fence.delivered
    if n <= 0:
        return work_reqs + expire
    due, _fence.pending = _fence.pending[:n], _fence.pending[n:]
    _check_same_controls(g.tp_group.cpu_group, due, _fence.delivered)
    _fence.delivered = common
    logger.info("[neuron] DP control fence: delivering %d control request(s) at iteration %d (%d still held)",
                len(due), _fence.it, len(_fence.pending))
    return work_reqs + expire + due


def _track_pause(original_fn, scheduler, *args, **kwargs):
    _fence.scheduler = scheduler
    out = original_fn(scheduler, *args, **kwargs)
    paused = bool(getattr(scheduler, "_engine_paused", False))
    if paused != _fence.paused:
        logger.info("[neuron] DP control fence: %s at iteration %d (results queued %d)",
                    "paused" if paused else "resumed", _fence.it, len(getattr(scheduler, "result_queue", ()) or ()))
    _fence.paused = paused
    return out


def register_hooks(registry, around) -> None:
    """Register the identity and the control fence; the caller does this only where the serving policy runs the
    identity. The pinned upstream sources are checked first, so a changed one raises before any hook is in place."""
    from sglang_neuron.sglang_patches import source_guard

    source_guard.check_targets(_UPSTREAM_MD5, __name__)
    targets = {
        "sglang.srt.managers.scheduler_components.dp_attn.prepare_mlp_sync_batch_raw": prepare_mlp_sync_identity,
        "sglang.srt.disaggregation.decode.SchedulerDisaggregationDecodeMixin.event_loop_overlap_disagg_decode":
            _loop_entry_barrier,
        "sglang.srt.managers.scheduler_components.request_receiver.SchedulerRequestReceiver"
        "._broadcast_reqs_across_ranks": fenced_broadcast,
        "sglang.srt.managers.scheduler.Scheduler.process_input_requests": _track_pause,
        "sglang.srt.disaggregation.decode.SchedulerDisaggregationDecodeMixin.get_new_prebuilt_batch": hold_prebuilt,
    }
    for target, hook in targets.items():
        registry.register(target, hook, around)
