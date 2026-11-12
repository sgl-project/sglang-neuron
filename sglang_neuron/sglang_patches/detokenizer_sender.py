# SPDX-License-Identifier: Apache-2.0
"""Move a DP group's detokenizer push from its leader to attention-TP rank 1.

In upstream SGLang the DP leader pushes each step's outputs to the detokenizer, while every other rank of the group
builds the same payload and drops it. On a DP-attention decode server the leader streams between its step result and
the next request broadcast, which the rest of the group waits for, so on a CPU-saturated node the stream sits on the
step's critical path. Here rank 1 does the push. Every other rank, the leader included, only updates the request state
that SGLang reads outside the stream (finished output and length, the send offset, the sampling-mask rows, and the
once-per-request marks of the detokenize header and the request stats), under upstream's own skip, keep and stream
conditions.

Rank 1's payload is the leader's. Requests and control messages reach the whole group through one broadcast, and the
sampled tokens agree, so at each ``stream_output`` call every rank holds the same requests in the same state. Two
fields are rank-local. One is the scheduler time stats, so the policy keeps upstream's push when only the leader
records metrics, logs or traces requests. The other is the abort message of a PD transfer failure: the client reads
rank 1's copy of the same abort.

Rank 1 pushes to the endpoint the leader would have used: the detokenizer, or under ``--skip-tokenizer-init`` the
tokenizer. With ``--detokenizer-worker-num N`` above 1, it pushes straight to worker ``attn_dp_rank % N`` instead of
through upstream's ``MultiDetokenizerRouter``. That router unpickles and re-pickles every message in one process and
splits them by ``crc32`` of each request's id; at the giga-kernel decode's step rate its one process falls behind 16
groups' outputs. The workers' endpoints are named from the router's (``worker_ipc``),
so rank 1 finds its worker without new plumbing, and a request stays on one worker, which keeps its incremental decode
state. ``Engine._launch_detokenizer_subprocesses`` is upstream's with that one name changed. The launching process
starts the workers and registers its hooks before it knows the serving policy, so this hook is registered in every
process; an endpoint named this way also serves upstream's router, as the temporary file did. Rank 1's first push
refuses an endpoint with no listener rather than queue into nothing. The workers' endpoints are the launching node's,
so a server over several nodes (a model's declared decode layout) streams from rank 1 only with one detokenizer: rank
1 then pushes to the detokenizer's own endpoint, which SGLang serves over TCP on node 0 whenever ``--nnodes`` is above
1.

The serving policy enables this under the DP identity with an attention-TP width of at least 2, and
``init_ipc_channels`` refuses a runtime that does not match. Targets are md5-pinned against upstream SGLang 43881a7
(``_TARGETS_MD5``, ``_LAUNCH_MD5``).

Upstream: SGLang could let the scheduler choose which rank of a DP-attention group owns the detokenizer push, and
connect a rank to its own detokenizer worker; the module would then reduce to the platform's choice of rank 1."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# the attention-TP rank of each DP group that owns the group's detokenizer push
SENDER_RANK = 1

_SCHEDULER = "sglang.srt.managers.scheduler"
_COMPONENTS = "sglang.srt.managers.scheduler_components"
_STREAMER = f"{_COMPONENTS}.output_streamer"
IPC_TARGET = f"{_SCHEDULER}.Scheduler.init_ipc_channels"
STREAM_TARGET = f"{_STREAMER}.SchedulerOutputStreamer._stream_output_generation"
_ENGINE = "sglang.srt.entrypoints.engine"
LAUNCH_TARGET = (_ENGINE, "Engine._launch_detokenizer_subprocesses")
_LAUNCH_MD5 = {LAUNCH_TARGET: "f38d2dfcfa8cfb2b68b190e1bad50c6c"}
# upstream's worker endpoint, a fresh temporary file per worker, and the one named from the router's (worker_ipc)
UPSTREAM_WORKER_IPC = 'worker_ipc = f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}"'
NEURON_WORKER_IPC = ('worker_ipc = __import__("sglang_neuron.sglang_patches.detokenizer_sender", '
                     'fromlist=["worker_ipc"]).worker_ipc(router_ipc_name, i)')
_MIXIN = "sglang.srt.managers.multi_tokenizer_mixin"
# (module, qualified name): md5 of the source
_TARGETS_MD5 = {
    # wrapped: the leader alone gets the detokenizer push
    (_SCHEDULER, "Scheduler.init_ipc_channels"): "251c7e8b4e8e972d977d41e485ab8194",
    # the push's endpoint and socket, and SenderWrapper(None) on every rank but the leader
    (f"{_COMPONENTS}.ipc_channels", "SchedulerIpcChannels.create"): "612dc2808a1b3f0c96d974a7f3c48f56",
    # a push of SenderWrapper(None) sends nothing
    (f"{_COMPONENTS}.output_sender", "SenderWrapper.send_output"): "ce99d23b0c33b8ab07a2757171375151",
    # the streamer takes its push from the channels, after init_ipc_channels has run
    (_SCHEDULER, "Scheduler.init_output_streamer"): "6a6d7a49176bd75147390d926c6d03f3",
    # the control requests forwarded to the detokenizer through the push
    (_SCHEDULER, "Scheduler.handle_freeze_gc"): "02cd5faa26bb04c5fb80dd09afaa6dc6",
    (_SCHEDULER, "Scheduler.configure_logging"): "9d56a715a2aa28c1bfa842fab1c08654",
    # control requests reach every rank of the DP group under local control broadcast
    (f"{_COMPONENTS}.request_receiver", "SchedulerRequestReceiver._broadcast_reqs_across_ranks"):
        "2c89050881db2b90eb39e316e32cd8c6",
    # wrapped: the stream and the request state it changes (the bookkeeping reproduces it)
    (_STREAMER, "SchedulerOutputStreamer._stream_output_generation"): "052e5ca6d4626092ed4d4a3df3f6a948",
    (_STREAMER, "_GenerationStreamAccumulator.accept"): "9fafa03f0b7286a2268cc8c0277411e3",
    (_STREAMER, "_GenerationStreamAccumulator._beam_admits"): "1b623959101d2d4f3742634a14c5c9bc",
    # the leader alone logs time stats from the stream, under --enable-request-time-stats-logging (refused)
    (_STREAMER, "SchedulerOutputStreamer._maybe_log_time_stats"): "ec5744e95bf1c1fb11a9c1dc7cb5572a",
    # a request's stats carry its scheduler time stats only on a rank with metrics on
    (_STREAMER, "SchedulerOutputStreamer.build_request_stats"): "cb383a5b31e97e6131589cb81869e7e9",
    # an abort's release of the detokenizer's per-request state goes through the push, on the rank that streamed
    (_SCHEDULER, "Scheduler.send_abort_output"): "9ca91a856ffacd6f6485da6476cbeb98",
    (_STREAMER, "SchedulerOutputStreamer.release_detokenize_state"): "ddad2f95b9ef422abc5711dbcc495403",
    # bypassed: the router forwards a scheduler's output to the worker each request's id hashes to (a batch split by
    # worker, its items' tokenizer endpoints kept), and sends the control messages (FreezeGCReq, ConfigureLoggingReq) and an empty batch to every worker. Past it,
    # each DP group's rank 1 sends all of these to its own worker; every group receives the control requests (the DP
    # identity requires --enable-dp-attention-local-control-broadcast), so every worker that serves a group gets them
    (_MIXIN, "MultiDetokenizerRouter.event_loop"): "f0079a10fe0692b255b0068e16ff5ca5",
    (_MIXIN, "MultiDetokenizerRouter._send_batch"): "7023424323580fd5deaddcda1d149ef4",
    # a worker serves every message it receives and sends each output to its requests' tokenizer workers (the
    # output's http_worker_ipcs, carried from the scheduler's batch either way)
    (_MIXIN, "MultiHttpWorkerDetokenizerMixin.multi_http_worker_event_loop"): "f0a85e272dc4653daca0e1fd86f6ea48",
}


# the longest socket name Linux takes (sockaddr_un's sun_path, without its terminating NUL)
_MAX_SOCKET_NAME = 107


def worker_ipc(router_ipc_name: str, index: int) -> str:
    """Endpoint of detokenizer worker ``index`` for a router on ``router_ipc_name``. Workers are node-local either way:
    next to the router's socket file for an ``ipc://`` router, or in the abstract socket namespace for the ``tcp://``
    router a DP-attention launch uses, named from a digest of the router's endpoint so a long host name fits."""
    import hashlib

    if router_ipc_name.startswith("ipc://"):
        endpoint = f"{router_ipc_name}-worker{index}"
    elif router_ipc_name.startswith("tcp://") and router_ipc_name.rpartition(":")[2].isdigit() \
            and router_ipc_name[len("tcp://"):].rpartition(":")[0]:
        digest = hashlib.sha1(router_ipc_name.encode()).hexdigest()[:16]
        endpoint = f"ipc://@sglang-neuron-detokenizer-{digest}-worker{index}"
    else:
        raise ValueError(f"detokenizer router endpoint {router_ipc_name!r}: not an ipc:// path or a tcp://host:port")
    if len(endpoint) - len("ipc://") > _MAX_SOCKET_NAME:
        raise ValueError(f"detokenizer worker endpoint {endpoint!r} is longer than a socket name can be "
                         f"({_MAX_SOCKET_NAME} bytes): its router's socket file is too deep; set a shorter TMPDIR")
    return endpoint


def _listening(endpoint: str) -> bool:
    """Whether a socket listens on the IPC ``endpoint`` (a path, or an abstract name ``@...``)."""
    import os
    import stat

    path = endpoint[len("ipc://"):]
    if path.startswith("@"):
        with open("/proc/net/unix") as f:
            return any(line.split()[-1] == path for line in f.read().splitlines()[1:] if len(line.split()) > 7)
    try:
        return stat.S_ISSOCK(os.stat(path).st_mode)
    except FileNotFoundError:
        return False


# how long the first push waits for its worker's endpoint to have a listener (the workers start after the schedulers)
WORKER_WAIT_S = 120.0


def worker_sender(socket, endpoint: str, wait_s: float = WORKER_WAIT_S):
    """A SenderWrapper over ``socket`` whose first send refuses an ``endpoint`` with no listener: a worker the launching
    process started on another endpoint (its launch not the rebuilt one) would leave every message queued in the
    socket and every stream waiting."""
    import time

    from sglang.srt.managers.scheduler_components.output_sender import SenderWrapper

    class _WorkerSender(SenderWrapper):
        checked = False

        def send_output(self, output, recv_obj=None):
            if not self.checked:
                deadline = time.monotonic() + wait_s
                while not _listening(endpoint):
                    if time.monotonic() >= deadline:
                        raise RuntimeError(
                            f"detokenizer worker endpoint {endpoint} has no listener after {wait_s:.0f} s: the "
                            "launching process did not start its detokenizer workers where worker_ipc names them "
                            f"({'.'.join(LAUNCH_TARGET)} rebuilt by {__name__}), so this DP group's outputs would "
                            "never be streamed")
                    time.sleep(0.05)
                self.checked = True
            return super().send_output(output, recv_obj)

    return _WorkerSender(socket)


def launch_detokenizer_subprocesses():
    """Engine._launch_detokenizer_subprocesses, upstream's with each worker's endpoint named from the router's. The
    rebuilt function is the plain one: SGLang's registry applies the classmethod again."""
    from sglang_neuron.sglang_patches import source_guard

    upstream = source_guard.resolve(*LAUNCH_TARGET)
    return source_guard.with_substitutions(upstream, [(UPSTREAM_WORKER_IPC, NEURON_WORKER_IPC)],
                                           decorators=("@classmethod\n",))


def register_worker_launch(registry, replace) -> None:
    """The worker endpoints named from the router's, in every process (a launch hook)."""
    from sglang_neuron.sglang_patches import source_guard

    source_guard.check_targets(_LAUNCH_MD5, __name__)
    registry.register(".".join(LAUNCH_TARGET), launch_detokenizer_subprocesses(), replace)


def runtime_blockers(scheduler) -> list:
    """The hand-off's conditions that this scheduler's runtime does not meet; an empty list means it runs.
    serving_policy.detokenizer_sender_blockers checks the same from the launch's arguments."""
    from sglang.srt.environ import envs
    from sglang.srt.runtime_context import get_observability, get_parallel, get_serving

    p, obs = get_parallel(), get_observability()
    checks = {
        "DP attention": bool(p.attn_dp_enabled),
        f"an attention-TP width above {SENDER_RANK}": int(p.attn_tp_size) > SENDER_RANK,
        "--enable-dp-attention-local-control-broadcast": bool(p.enable_dp_attention_local_control_broadcast),
        "a generation model": bool(getattr(scheduler, "is_generation", True)),
        "no --enable-request-time-stats-logging": not obs.enable_request_time_stats_logging,
        "no --enable-metrics without --enable-metrics-for-all-schedulers":
            not obs.enable_metrics or bool(obs.enable_metrics_for_all_schedulers),
        "no --enable-trace": not obs.enable_trace,
        "SGLANG_RUST_SERVER off": not envs.SGLANG_RUST_SERVER.get(),
        # the workers' IPC endpoints are the launching node's (worker_ipc)
        "one node (nnodes 1) with --detokenizer-worker-num above 1":
            int(get_serving().detokenizer_worker_num or 1) <= 1 or int(p.nnodes or 1) == 1,
    }
    return [name for name, ok in checks.items() if not ok]


def init_ipc_channels(original_fn, self, port_args, *args, **kwargs):
    """AROUND Scheduler.init_ipc_channels: upstream's channels, with the detokenizer push moved from the DP leader to
    attention-TP rank 1."""
    import dataclasses

    import zmq
    from sglang.srt.managers.scheduler_components.output_sender import SenderWrapper
    from sglang.srt.runtime_context import get_parallel, get_serving
    from sglang.srt.utils.network import get_zmq_socket

    result = original_fn(self, port_args, *args, **kwargs)
    blockers = runtime_blockers(self)
    if blockers:
        # static configuration: every rank raises here at start, none mid-run
        raise RuntimeError("the serving policy moves the detokenizer push to attention-TP rank "
                           f"{SENDER_RANK} (serving_policy.detokenizer_sender_rank1), but this scheduler does not meet: "
                           + "; ".join(blockers))
    rank = get_parallel().attn_tp_rank
    channels = self.ipc_channels
    if rank == 0:
        socket = channels.send_to_detokenizer.socket
        if socket is None:
            raise RuntimeError("the DP leader has no detokenizer push to hand to attention-TP rank "
                               f"{SENDER_RANK}")
        self.ipc_channels = dataclasses.replace(channels, send_to_detokenizer=SenderWrapper(None))
        socket.close(linger=0)
    elif rank == SENDER_RANK:
        dp_rank, workers = get_parallel().attn_dp_rank, get_serving().detokenizer_worker_num
        if self.skip_tokenizer_init:
            endpoint, target = port_args.tokenizer_ipc_name, "the tokenizer"
        elif workers > 1:
            # the group's own detokenizer worker, past upstream's router (worker_ipc)
            endpoint, target = (worker_ipc(port_args.detokenizer_ipc_name, dp_rank % workers),
                                f"detokenizer worker {dp_rank % workers} of {workers}")
        else:
            endpoint, target = port_args.detokenizer_ipc_name, "the detokenizer"
        socket = get_zmq_socket(zmq.Context(1), zmq.PUSH, endpoint, False)
        sender = worker_sender(socket, endpoint) if workers > 1 and not self.skip_tokenizer_init else \
            SenderWrapper(socket)
        self.ipc_channels = dataclasses.replace(channels, send_to_detokenizer=sender)
        logger.info("[neuron] detokenizer push: attention-TP rank %d of DP group %s sends the group's outputs to %s; "
                    "its other ranks keep the stream's request bookkeeping", SENDER_RANK, dp_rank, target)
    return result


def stream_output_generation(original_fn, self, reqs, return_logprob, skip_req=None, is_idle_batch=False):
    """AROUND SchedulerOutputStreamer._stream_output_generation: a rank without the detokenizer push makes the request
    state changes upstream's stream makes, but builds no payload."""
    if (self.send_to_detokenizer.socket is not None or self.has_additional_customized_info
            or self.rust_server is not None):
        return original_fn(self, reqs, return_logprob, skip_req=skip_req, is_idle_batch=is_idle_batch)
    from sglang.srt.managers.scheduler_components import output_streamer
    from sglang.srt.runtime_context import get_serving

    stream_interval = get_serving().stream_interval
    force_interval = output_streamer.DEFAULT_FORCE_STREAM_INTERVAL
    for req in reqs:
        if req is skip_req:
            continue
        finished = req.finished()
        if finished and req.finished_output:
            continue
        if req.beam_group is not None and not (req.is_beam_leader and finished):
            continue
        if finished:
            req.finished_output = True
            if req.finished_len is None:
                req.finished_len = len(req.output_ids)
        elif req.stream:
            interval = req.sampling_params.stream_interval or stream_interval
            unsent = len(req.output_ids) - req.send_token_offset
            if not (interval == 1 or unsent >= interval or (req.send_token_offset == 0 and unsent > 0)):
                continue
            if req.check_match_stop_str_prefix():
                continue
        elif len(req.output_ids) % force_interval != 0:
            continue
        req.send_token_offset = len(req.output_ids_through_stop)
        if self.output_sink is output_streamer.OutputSink.DETOKENIZER:
            req.take_detokenize_header_due()
        req.take_request_stats_due()
        if req.return_sampling_mask:
            req.sampling_mask_rows.take()
    return None


def register(registry, around) -> None:
    from sglang_neuron.sglang_patches import source_guard

    source_guard.check_targets(_TARGETS_MD5, __name__)
    registry.register(IPC_TARGET, init_ipc_channels, around)
    registry.register(STREAM_TARGET, stream_output_generation, around)
