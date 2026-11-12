# SPDX-License-Identifier: Apache-2.0
"""Record the Neuron runtime's system profile through SGLang's /start_profile and /stop_profile.

SGLang's profiler drives torch.profiler for the host and CUDA, and neither sees a NeuronCore. A request with
``activities: ["NEURON"]`` makes every scheduler start torch-neuronx's runtime-mode profile instead. It records one
``nc_exec_running`` event per program execution and the collectives' ``cc_exec_barrier`` waits, on every core the rank
drives. /stop_profile writes one session directory per scheduler where SGLang puts its own traces: under the request's
``output_dir``, else SGLANG_TORCH_PROFILER_DIR, else /tmp. neuron-explorer opens the directories.

On the giga-kernel decode only DP group 0's first rank records (``_records``). Every rank meets the others at the
kernel's barrier inside each step, so that one rank's barrier waits already show when the last rank arrived, while a
session on each of the 128 ranks slows the run it records and writes a trace no viewer opens. The other ranks join the
group's counts below without a session of their own. Its event ring holds ``GIGA_DECODE_EVENTS_PER_NC`` events per
NeuronCore, against the runtime's default of about a million: a giga step records many more runtime events than a
per-layer step.

A session exports itself while it records: torch-neuronx's continuous export drains each core's event ring every
``EXPORT_INTERVAL_MS`` into a chunk in the session directory, and the stop joins the chunks into the session's
``ntrace.pb``, the same file a stop without exports writes. Without it the stop serializes the whole window on the
scheduler thread while every other rank waits; with it a drain can delay one program submission on its core per
interval. Since neither the runtime nor torch-neuronx reports a session it failed to write, the stop reads the
directory: it fails on chunks left unjoined, or on a window that ran a forward and has no ``ntrace.pb``.

A TP group succeeds or fails as a whole. If one rank's runtime starts no session, the start fails on every rank of the
group, and the group answers a stop only once every rank's session is written. SGLang's tokenizer answers the HTTP call
with the first group's reply only, so with several groups (DP attention, or a prefill of several replicas) a failed
group could read as success. The tokenizer half of this module therefore answers a NEURON start, and every stop, with
every group's replies, and when some groups fail to start it stops the ones that did. A start succeeds on every group
or on none. Every stop is answered this way because a launch of several tokenizer workers may take the stop on another
worker than the start, which keeps no record of it.

NEURON runs alone. torch-neuronx records through torch.profiler, which holds one session per process, so a request that
adds CPU, GPU or MEM is refused. So is one with start_step, num_steps or profile_by_stage: SGLang counts those windows
in forward steps through its own profiler, and this profile spans the two calls instead. A server launched with the
serve window (SGLANG_NEURON_PROFILE_SERVE_RANKS), which records through the same torch.profiler, refuses every
/start_profile, its own and SGLang's (model_executor/profiling.profile_refusal), before any count of its group.

Targets, md5-pinned against upstream SGLang 43881a7: in every scheduler, ``SchedulerProfilerManager._profile``
(wrapped) and its ``__post_init__``, which sets the in-progress flag and the TP group read here; in the process that
serves HTTP, ``TokenizerControlMixin._execute_profile`` (wrapped).

Upstream: an out-of-tree platform could give SGLang's profiler its own device session; this module then goes away.
"""

from __future__ import annotations

import glob
import logging
import os
import time

ACTIVITY = "NEURON"
TARGET = "sglang.srt.managers.scheduler_components.profiler_manager.SchedulerProfilerManager._profile"
_MODULE = "sglang.srt.managers.scheduler_components.profiler_manager"
_SOURCES_MD5 = {
    (_MODULE, "SchedulerProfilerManager._profile"): "a2e4d212f0d65321e19f2dbb67578ed7",
    (_MODULE, "SchedulerProfilerManager.__post_init__"): "ec314199c3a2bcfab089dadc8677d092",
}
_TOKENIZER_MODULE = "sglang.srt.managers.tokenizer_control_mixin"
TOKENIZER_TARGET = f"{_TOKENIZER_MODULE}.TokenizerControlMixin._execute_profile"
_TOKENIZER_SOURCES_MD5 = {
    (_TOKENIZER_MODULE, "TokenizerControlMixin._execute_profile"): "612486e86b17fe3e604a5a8172035bf4",
}
# the giga-kernel decode's per-NeuronCore event cap, the one its profiles of record were captured with
GIGA_DECODE_EVENTS_PER_NC = 32_000_000
# How often the export worker drains the rings: a stop exports at most one interval's events, and a longer interval
# makes fewer but longer stalls in the window.
EXPORT_INTERVAL_MS = 2_000
# the runtime's export chunks in a session directory, joined into ntrace.pb at the stop
EXPORT_CHUNKS = "ntrace.nc*.part"
# the session's events, joined from the chunks; a session that recorded none has none
TRACE = "ntrace.pb"
# set on SGLang's profiler manager: this rank's running session, its directory and the scheduler's forward count at its
# start
_SESSION = "_neuron_system_profile"

logger = logging.getLogger(__name__)


def output_dir(recv_req) -> str:
    """Where the session directories go: SGLang's own choice for its traces, with ``~`` expanded as SGLang does."""
    return os.path.expanduser(recv_req.output_dir or os.getenv("SGLANG_TORCH_PROFILER_DIR", "/tmp"))


def _refusal(manager, recv_req) -> str | None:
    others = [activity for activity in recv_req.activities if activity != ACTIVITY]
    if others:
        return (f"activities {recv_req.activities}: profile {ACTIVITY} alone, since torch-neuronx records it through "
                "torch.profiler, which holds one session per process")
    windows = [name for name in ("start_step", "num_steps", "profile_by_stage") if getattr(recv_req, name, None)]
    if windows:
        return f"{', '.join(windows)} with {ACTIVITY}: its profile spans /start_profile to /stop_profile"
    if getattr(manager, _SESSION, None) is not None or getattr(manager, "profile_in_progress", False):
        return "Profiling is already in progress. Call /stop_profile first."
    return None


def _start_session(directory: str):
    """Start this process's runtime-mode profile under ``directory``, exporting itself every EXPORT_INTERVAL_MS; return
    the profiler and its session directory."""
    from torch_neuronx.profiling import ContinuousExportConfig, InspectExportMask, TimeTrigger

    from sglang_neuron import serving_policy
    from sglang_neuron.model_executor import profiling

    cap = GIGA_DECODE_EVENTS_PER_NC if serving_policy.current().giga_decode else None
    # the system profile alone: this session records no device profile to export
    export = ContinuousExportConfig(policy=TimeTrigger(interval_ms=EXPORT_INTERVAL_MS,
                                                       mask=InspectExportMask.SYSTEM_PROFILE))
    return profiling.start_session(directory, ("RUNTIME",), max_events_per_nc=cap, continuous_export=export)


def _stop_session(session, session_dir: str, executed: bool) -> None:
    """Stop a session ``_start_session`` started: the runtime exports what its last export left and joins the export
    chunks into the session's ntrace.pb. ``executed``: this rank ran programs in the window, so the session holds their
    events. The stop fails on chunks still in the directory, which neuron-explorer would not read, and on an executed
    window without ntrace.pb, whose events are lost (module docstring); the runtime logged why in both cases."""
    session.stop()
    left = sorted(os.path.basename(p) for p in glob.glob(os.path.join(glob.escape(session_dir), EXPORT_CHUNKS)))
    if left:
        raise RuntimeError(f"the Neuron runtime left {len(left)} export chunk(s) ({', '.join(left)}) unjoined in "
                           f"{session_dir}")
    if executed and not os.path.isfile(os.path.join(session_dir, TRACE)):
        raise RuntimeError(f"the Neuron runtime wrote no {TRACE} in {session_dir}, though this rank ran programs in "
                           "the window: their events are lost (this scheduler's log holds the runtime's \"System "
                           "profile export failed for NeuronCore\" warning, or its error joining the part files)")


def _records() -> bool:
    """Whether this rank records a session: every rank, but DP group 0's first rank alone on the giga-kernel decode."""
    from sglang_neuron import serving_policy

    if not serving_policy.current().giga_decode:
        return True
    from sglang.srt.runtime_context import get_parallel

    parallel = get_parallel()
    return parallel.attn_dp_rank == 0 and parallel.attn_tp_rank == 0


def _group_failures(manager, failed: bool) -> int:
    """Count the ranks of the TP group that failed. Every rank reaches this all-reduce, because every rank of the group
    handles the requests in the same order."""
    import torch

    count = torch.tensor([int(failed)], dtype=torch.int32)
    torch.distributed.all_reduce(count, group=manager.dp_tp_cpu_group)
    return int(count.item())


def _start(manager, recv_req):
    from sglang.srt.managers.io_struct import ProfileReqOutput

    refusal = _refusal(manager, recv_req)
    if refusal is not None:
        return ProfileReqOutput(success=False, message=refusal)
    directory = output_dir(recv_req)
    # SGLang's own count of this scheduler's forwards, taken before anything starts: the stop requires the trace of a
    # window that ran one
    forward_ct = manager.get_forward_ct()
    started, error = None, None
    if not _records():
        started = (None, None)  # in the profile, without a session of its own
    else:
        try:
            started = _start_session(directory)
        except Exception as e:
            error = e
            logger.error("[neuron] system profile did not start (pid %d, into %s): %s", os.getpid(), directory, e)
    failed = _group_failures(manager, started is None)
    if failed:
        if started is not None and started[0] is not None:
            started[0].stop()
        return ProfileReqOutput(success=False, message=(
            f"the Neuron system profile did not start on {failed} rank(s) of the TP group"
            + (f": {error}" if error is not None else "; their scheduler logs say why")))
    setattr(manager, _SESSION, (*started, forward_ct))
    if started[0] is not None:
        logger.info("[neuron] system profile started: %s", started[1])
    return ProfileReqOutput(success=True, message=f"Neuron system profile started into {directory}")


def _stop(manager):
    from sglang.srt.managers.io_struct import ProfileReqOutput

    session, session_dir, started_forward_ct = getattr(manager, _SESSION)
    setattr(manager, _SESSION, None)
    error = None
    began = time.monotonic()
    try:
        if session is not None:
            # a window's forwards ran this rank's programs, whose submissions record events at once (a PD decode's
            # prebuilt batch runs none, but its requests' decode steps follow it at once)
            _stop_session(session, session_dir, executed=manager.get_forward_ct() > started_forward_ct)
    except Exception as e:
        error = e
        logger.error("[neuron] system profile %s was not written: %s", session_dir, e)
    # how long this rank's scheduler, and through the step's collectives every rank, waited on the stop
    took_s = time.monotonic() - began
    # Every rank joins the group's count, including one whose stop failed; otherwise the others would wait for it
    # forever. The group's first rank answers once every rank's session is on disk.
    failed = _group_failures(manager, error is not None)
    if failed:
        return ProfileReqOutput(success=False, message=(
            f"the Neuron system profile was not written on {failed} rank(s) of the TP group"
            + (f": {error}" if error is not None else "; their scheduler logs say why")))
    if session is not None:
        logger.info("[neuron] system profile written: %s (its stop took %.3f s)", session_dir, took_s)
    return ProfileReqOutput(success=True, message="Neuron system profile written")


def profile(original_fn, self, recv_req, *args, **kwargs):
    from sglang.srt.managers.io_struct import ProfileReqOutput, ProfileReqType

    from sglang_neuron.model_executor import profiling

    if recv_req.req_type == ProfileReqType.START_PROFILE:
        # a server launched with the serve window records it through this process's torch.profiler: every scheduler
        # of the server refuses here alike, before _records and the NEURON start's group count, and none starts
        # SGLang's profiler
        refusal = profiling.profile_refusal()
        if refusal is not None:
            return ProfileReqOutput(success=False, message=refusal)
    if recv_req.req_type == ProfileReqType.START_PROFILE and ACTIVITY in (recv_req.activities or ()):
        return _start(self, recv_req)
    if getattr(self, _SESSION, None) is not None:
        if recv_req.req_type != ProfileReqType.START_PROFILE:
            return _stop(self)
        # a start of SGLang's own profiler while a NEURON session runs: torch.profiler holds one session per process,
        # and every rank of the group holds the same session, so every rank refuses
        return ProfileReqOutput(success=False, message="A Neuron system profile is in progress. Call /stop_profile "
                                                        "first.")
    return original_fn(self, recv_req, *args, **kwargs)


async def execute_profile(original_fn, self, req):
    """The tokenizer's answer to a NEURON start, and to every stop: every group's replies, not the first group's
    alone."""
    from sglang.srt.managers.io_struct import ProfileReq, ProfileReqType

    start = req.req_type == ProfileReqType.START_PROFILE
    if start and ACTIVITY not in (req.activities or ()):
        return await original_fn(self, req)
    replies = await self.profile_communicator(req)
    failed = [reply for reply in replies if not reply.success]
    if start and failed and len(failed) < len(replies):
        # stop the groups that did start, so none records alone; the failed groups just answer that nothing runs
        await self.profile_communicator(ProfileReq(req_type=ProfileReqType.STOP_PROFILE))
    if failed:
        raise RuntimeError(f"{len(failed)} of {len(replies)} group(s): " + " | ".join(r.message for r in failed))
    return replies[0]


def register(registry, around) -> None:
    """Register the schedulers' half, which handles each scheduler's profile requests."""
    from sglang_neuron.sglang_patches import source_guard

    source_guard.check_targets(_SOURCES_MD5, __name__)
    registry.register(TARGET, profile, around)


def register_tokenizer(registry, around) -> None:
    """Register the HTTP process's half, which answers a NEURON start, and every stop, with every group's replies."""
    from sglang_neuron.sglang_patches import source_guard

    source_guard.check_targets(_TOKENIZER_SOURCES_MD5, __name__)
    registry.register(TOKENIZER_TARGET, execute_profile, around)
