# SPDX-License-Identifier: Apache-2.0
"""A bounded Neuron device-profile window over decode steps, after the compiled graphs have warmed (device_profile,
ServingCapture), recorded through torch-neuronx's torch.profiler integration (start_session)."""

from __future__ import annotations

import os
from contextlib import contextmanager

from sglang_neuron import envs

def profile_refusal(environ=None):
    """Why this process refuses every /start_profile, or None. torch.profiler runs one session per process, and a
    second start cancels the first without an error. A server launched with the serve window
    (SGLANG_NEURON_PROFILE_SERVE_RANKS) records it through that session, so every scheduler of the server, on every
    replica, refuses every /start_profile alike for as long as it runs (sglang_patches.neuron_system_profile asks)."""
    if serve_window(environ) is None:
        return None
    return ("this server was launched with the serve window (SGLANG_NEURON_PROFILE_SERVE_RANKS), which records through "
            "torch.profiler, one session per process: launch it without the window to use /start_profile")


def _session_state():
    """This process's Neuron profile session as torch-neuronx's export frontend sees it: None when none runs, else
    whether its continuous-export worker runs. torch-neuronx publishes a session to that frontend once the runtime has
    begun it, and withdraws it when the session stops. The question is an export request that selects nothing, which
    exports nothing: torch_neuronx.profiling.request_export answers it NO_ACTIVE_SESSION when no session runs (a failed
    begin included), EXPORT_NOT_ENABLED from a session without the worker, and UNSUPPORTED_MASK from the worker, which
    refuses an empty request before it takes one. A request for real data would make a running worker export."""
    from torch_neuronx.profiling import ExportStatus, InspectExportMask, NRTProfilerError, request_export

    try:
        request_export(InspectExportMask(0))
    except NRTProfilerError as e:
        status = getattr(e, "status", None)
        if status is ExportStatus.NO_ACTIVE_SESSION:
            return None
        if status is ExportStatus.EXPORT_NOT_ENABLED:
            return False
        if status is ExportStatus.UNSUPPORTED_MASK:
            return True
        raise
    raise RuntimeError("torch-neuronx accepted an export request that selects nothing; this plugin cannot tell whether "
                       "a Neuron profile session runs")


def _torch_profiler_running():
    """Whether torch.profiler records in this process, whatever its activities. torch sets this process-wide flag at a
    profiler's start and clears it at the end of its stop (torch.autograd.profiler._run_on_profiler_start and
    _run_on_profiler_stop), so a stop that raised before its end leaves it set. A torch without the flag raises here,
    so the check is never skipped."""
    from torch.autograd import profiler

    return bool(profiler._is_profiler_enabled)


def session_running():
    """Whether torch-neuronx's profiler runs a Neuron session in this process; asking exports nothing
    (_session_state)."""
    return _session_state() is not None


def start_session(directory, modes, max_events_per_nc=None, neff_cache_dir=None, continuous_export=None):
    """Start this process's torch.profiler session of the Neuron profile ``modes`` (torch_neuronx.profiling.ProfileMode
    names) into ``directory``, holding up to ``max_events_per_nc`` events per NeuronCore when set (else the runtime's
    own cap), copying the NEFFs from ``neff_cache_dir`` into the profile when set, and exporting as it records when
    ``continuous_export`` (a torch_neuronx.profiling.ContinuousExportConfig) is set. Returns the profiler and its
    session directory, which the runtime creates under ``directory``. The returned directory is the session's;
    NeuronConfig.get_session_dir() names the runtime's current session, so keep the returned one. Raises, and leaves the
    running session alone, while a Neuron session runs in this process (one whose stop raised before the runtime tore
    it down included) or torch.profiler records another one (of other activities, or one whose begin the runtime
    refused and whose stop raised); raises when the runtime did not begin the new one, and when a session asked to
    export runs no export worker."""
    from torch.profiler import ProfilerActivity, profile
    from torch_neuronx.profiling import NeuronConfig, ProfileMode

    if session_running():
        raise RuntimeError("a Neuron profile session already runs in this process, and torch.profiler runs one session "
                           "at a time")
    if _torch_profiler_running():
        raise RuntimeError("torch.profiler already records in this process (a profile without a Neuron session), and "
                           "a second start would end it")
    os.makedirs(directory, exist_ok=True)
    options = {name: value for name, value in (("max_events_per_nc", max_events_per_nc),
                                               ("neff_cache_dir", neff_cache_dir),
                                               ("continuous_export", continuous_export)) if value is not None}
    config = NeuronConfig(modes=[getattr(ProfileMode, mode) for mode in modes], profile_output_dir=directory, **options)
    session = profile(activities=[ProfilerActivity.PrivateUse1], experimental_config=config)
    session.start()
    try:
        # torch-neuronx only logs a begin the runtime refused, and an export worker that did not start (the session
        # then writes the whole window at its stop), and returns
        exports = _session_state()
        if exports is None:
            raise RuntimeError(f"the Neuron runtime did not begin a profile session into {directory} (torch-neuronx's "
                               "profiler logged why)")
        if continuous_export is not None and not exports:
            raise RuntimeError("the Neuron profile session runs no background export (torch-neuronx's profiler logged "
                               "why), so its stop would write the whole window on the scheduler thread")
        session_dir = config.get_session_dir()
    except BaseException:
        session.stop()
        raise
    return session, session_dir


@contextmanager
def device_profile(output_dir, mode=None):
    """Record the device and system profile of every execution while the context is open, and write it at exit.

    torch-neuronx's torch.profiler integration (torch_neuronx.profiling.NeuronConfig, its device and runtime modes)
    begins the Neuron runtime's inspect session at the start and writes it at the stop, with the NEFFs the device
    profile needs, copied from the NEFF cache (TORCH_NEURONX_NEFF_CACHE_DIR, /tmp/neff_cache by default). Opening it
    after warm-up keeps collective bootstrap and weight loading out. Raises when the runtime started no session, and
    when this process already runs the runtime's own profile from process start (NEURON_RT_INSPECT_ENABLE, which
    SGLANG_NEURON_PROFILE_RANKS sets): a new session would replace that one.

    ``mode``: ``session`` (every execution, what this window records) or None; SGLANG_NEURON_PROFILE_DEVICE_MODE sets
    only the SGLANG_NEURON_PROFILE_RANKS profile. ``model`` (one execution of each program) raises: NeuronConfig has
    no such mode.
    """
    if mode not in (None, "session"):
        raise ValueError(f"a device profile into {output_dir} with mode={mode!r}: this window records every execution "
                         "(session) through torch.profiler, which has no per-program (model) mode; pass session or "
                         "leave it unset")
    if _inspect_enabled(os.environ):
        raise RuntimeError(f"a device profile into {output_dir}: this rank's Neuron runtime already records its "
                           "profile from process start (NEURON_RT_INSPECT_ENABLE, set for "
                           "SGLANG_NEURON_PROFILE_RANKS), which a new session would replace; select the rank in one of "
                           "the two")
    neff_cache = os.environ.get("TORCH_NEURONX_NEFF_CACHE_DIR") or "/tmp/neff_cache"
    session, _ = start_session(output_dir, ("DEVICE", "RUNTIME"), neff_cache_dir=neff_cache)
    try:
        yield
    finally:
        session.stop()


def _inspect_enabled(environ):
    """Whether the Neuron runtime of a process with ``environ`` records its own profile from process start."""
    return environ.get("NEURON_RT_INSPECT_ENABLE", "0").strip() not in ("", "0")


def _window(names, defaults, environ=None):
    """A profile window's (RANKS, START, STEPS) variables as (ranks, start, steps), or None when RANKS is unset or
    empty. A malformed value raises naming the window; the platform calls this at launch
    (platform._check_plugin_env), so a bad window refuses the launch instead of failing every rank at its first
    decode step."""
    environ = os.environ if environ is None else environ
    from sglang_neuron import envs

    ranks_name, start_name, steps_name = names
    ranks = envs.ranks(ranks_name, "TP", environ)
    if ranks is None:
        return None
    bounds = []
    for name, default in ((start_name, defaults[0]), (steps_name, defaults[1])):
        raw = environ.get(name, default)
        try:
            n = int(raw)
        except ValueError:
            n = 0
        if n < 1:
            raise ValueError(f"{name}={raw!r} (the {ranks_name} window): give an integer >= 1")
        bounds.append(n)
    return ranks, bounds[0], bounds[1]


def refuse_overlap(tp_size, pp_size, environ=None):
    """Refuse a rank that SGLANG_NEURON_PROFILE_RANKS (local ranks, whose runtime records from process start) and the
    serve window (TP ranks) both select: the window would take the rank's one torch.profiler session from the running
    one, and its per-rank refusal at the window's first step would strand the other ranks of the window's barrier
    group, whose entry is a barrier. A server's ranks run on one node from GPU id 0 in steps of 1
    (platform.refuse_unserved), where SGLang gives TP rank r of pipeline stage s local rank s * tp_size + r in each
    replica (runtime_init.init_torch_distributed_hook), and the window counts TP ranks on each stage (ServingCapture),
    so local rank l is TP rank l % tp_size; a local rank at or above tp_size * pp_size runs no rank. The platform calls
    this at launch (platform._check_plugin_env) with the launch's --tp-size and --pp-size, so the overlap refuses the
    launch instead. A launch environment that enables the runtime's own profile (NEURON_RT_INSPECT_ENABLE) enables it
    on every rank, every rank of the window included, so the window refuses it too."""
    environ = os.environ if environ is None else environ
    window = serve_window(environ)
    if window is None:
        return
    if _inspect_enabled(environ):
        raise ValueError("NEURON_RT_INSPECT_ENABLE in the launch environment makes every rank's Neuron runtime record "
                         "its profile from process start, and SGLANG_NEURON_PROFILE_SERVE_RANKS's window refuses such "
                         "a rank at its first step, which would strand the rank's barrier group; select the "
                         "process-start ranks with SGLANG_NEURON_PROFILE_RANKS instead, or launch without the window")
    local = sorted(r for r in envs.profile_ranks(environ) if r < tp_size * pp_size and r % tp_size in window[0])
    if local:
        raise ValueError(f"SGLANG_NEURON_PROFILE_RANKS and SGLANG_NEURON_PROFILE_SERVE_RANKS both select TP rank(s) "
                         f"{sorted({r % tp_size for r in local})} (local rank(s) {local}; local rank s * {tp_size} + r "
                         "is TP rank r of pipeline stage s): a rank's one torch.profiler session serves one profile, "
                         "and the window's refusal at its first step would strand the rank's barrier group; select "
                         "each rank in one of the two")


def serve_label(owner):
    """This rank's serving-capture label (sglang_tree.rank_label of ``owner``'s parallel ranks)."""
    from sglang_neuron import sglang_tree

    return sglang_tree.rank_label(sglang_tree.parallel_of(owner))


def serve_window(environ=None):
    """The serving device-profile window (ServingCapture): (ranks, start, steps) or None."""
    return _window(("SGLANG_NEURON_PROFILE_SERVE_RANKS", "SGLANG_NEURON_PROFILE_SERVE_START",
                    "SGLANG_NEURON_PROFILE_SERVE_STEPS"), ("16", "4"), environ)


class ServingCapture:
    """One bounded device-profile window over decode steps, read from SGLANG_NEURON_PROFILE_SERVE_{RANKS,START,STEPS}
    and written to serve_rank<label> (``label``: sglang_tree.rank_label, the TP rank, dp<d>_tp<r> on a plain --dp-size
    server, as the host trace names its files); disabled unless requested. The window selects TP ranks, so on a plain
    --dp-size server each replica's selected ranks write their own directory. Its entry and exit are barriers, so every
    rank of the barrier's group must count the same steps. A server launched with it refuses /start_profile
    (profile_refusal)."""

    def __init__(self, rank, barrier, label=None):
        self.rank = rank
        self.barrier = barrier
        ranks, self.start, self.steps = serve_window()
        self.selected = rank in ranks
        self.directory = os.path.join(envs.profile_dir(), f"serve_rank{rank if label is None else label}")
        self.count = 0
        self.context = None

    def before(self):
        self.count += 1
        if self.count == self.start:
            _drain_device()
            self.barrier()
            if self.selected:
                self.context = device_profile(self.directory)
                self.context.__enter__()
            self.barrier()

    def after(self, completion_tensor):
        if self.count == self.start + self.steps - 1:
            # Wait for the final profiled execution before flushing, including
            # in overlap mode. Benchmark the subsequent unprofiled requests:
            # the window's boundary synchronization perturbs this request.
            completion_tensor.cpu()
            self.barrier()
            if self.context is not None:
                self.context.__exit__(None, None, None)
                self.context = None
            self.barrier()


def _drain_device():
    from sglang_neuron.model_executor.async_execution import synchronize_pending

    synchronize_pending()
