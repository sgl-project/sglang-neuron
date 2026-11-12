# SPDX-License-Identifier: Apache-2.0
"""Scheduler-loop hooks of the host trace (SGLANG_NEURON_HOST_TRACE). They record a per-rank timeline of the loop's
phases, so a launch delay on any rank can be traced to the phase that caused it. Nothing is registered unless the trace
is on.

Buffered rows are written right after an asynchronous launch, so the write adds no I/O to a step's critical path.
Whether a launch was asynchronous is read from the flag set when the step's outputs were deferred
(LogitsProcessorOutput.neuron_async), never from state the copy thread changes."""
from __future__ import annotations

import time

from sglang_neuron import envs
from sglang_neuron.model_executor import host_trace


def _rank(scheduler):
    """The rank in the scheduler's trace file name: its TP rank, prefixed by its DP replica on a --dp-size server
    without DP attention, where each replica numbers its TP ranks from 0."""
    from sglang_neuron import sglang_tree

    try:
        ps = sglang_tree.parallel_of(scheduler)
    except (AttributeError, RuntimeError):  # no ranks yet (a scheduler before its groups, a test double)
        return "?"
    return sglang_tree.rank_label(ps)


def _tp_rank(scheduler):
    """The scheduler's TP rank as an int, or None if it has none. SGLANG_NEURON_HOST_TRACE_RANKS selects ranks by it."""
    from sglang_neuron import sglang_tree

    try:
        tp = getattr(sglang_tree.parallel_of(scheduler), "tp_rank", None)
    except (AttributeError, RuntimeError):
        return None
    return int(tp) if tp is not None else None


def _pending(scheduler) -> bool:
    """Whether the idle loop still holds work: queued, bootstrapping, preallocating, transferring or chunked requests,
    or a running batch. Called only from on_idle, on the scheduler thread, for the trace's idle trim."""
    if len(getattr(scheduler, "waiting_queue", None) or ()) or getattr(scheduler, "chunked_req", None) is not None:
        return True
    if len(getattr(scheduler, "disagg_prefill_inflight_queue", None) or ()):
        return True
    for name in ("disagg_prefill_bootstrap_queue", "disagg_decode_prealloc_queue", "disagg_decode_transfer_queue"):
        if len(getattr(getattr(scheduler, name, None), "queue", None) or ()):
            return True
    running = getattr(scheduler, "running_batch", None)
    is_empty = getattr(running, "is_empty", None)
    return bool(running is not None and callable(is_empty) and not is_empty())


def dispatch_event_loop(original, scheduler, *args, **kwargs):
    """Open the trace file at the event loop's entry, once the model and the KV pools exist and the warm-up has run."""
    host_trace.start(_rank(scheduler), _tp_rank(scheduler))
    return original(scheduler, *args, **kwargs)


def _timed(kind, arg_of):
    def hook(original, *args, **kwargs):
        tr = host_trace.trace()
        if tr is None:
            return original(*args, **kwargs)
        t0 = time.time_ns()
        out = original(*args, **kwargs)
        tr.add(kind, t0, time.time_ns(), arg_of(args, kwargs, out))
        return out

    return hook


_finish = {"ns": 0, "n": 0}  # time in _handle_finish_state_updated_req and requests finished, within one result


def process_batch_result(original, scheduler, batch, result, *args, **kwargs):
    """Record a 'result' row and, inside it, a 'finish' row: the per-request finish-state handling summed over the
    batch, including the KV release of finished requests (arg: requests that finished). The 'finish' row starts with
    the result and is written only when a request finished or the handling took over 1 ms."""
    tr = host_trace.trace()
    if tr is None:
        return original(scheduler, batch, result, *args, **kwargs)
    _finish["ns"] = _finish["n"] = 0
    t0 = time.time_ns()
    out = original(scheduler, batch, result, *args, **kwargs)
    t1 = time.time_ns()
    tr.add("result", t0, t1, host_trace.describe(batch))
    if _finish["n"] or _finish["ns"] > 1_000_000:
        tr.add("finish", t0, t0 + _finish["ns"], _finish["n"])
    return out


def handle_finish_state(original, processor, req, *args, **kwargs):
    if host_trace.trace() is None:
        return original(processor, req, *args, **kwargs)
    t0 = time.time_ns()
    out = original(processor, req, *args, **kwargs)
    _finish["ns"] += time.time_ns() - t0
    if req.finished():
        _finish["n"] += 1
    return out


def run_batch(original, scheduler, batch, *args, **kwargs):
    """Record a 'launch' row. Right after an asynchronous launch, also write the buffered rows, once there are at
    least FLUSH_ROWS // 4 of them."""
    tr = host_trace.trace()
    if tr is None:
        return original(scheduler, batch, *args, **kwargs)
    t0 = time.time_ns()
    result = original(scheduler, batch, *args, **kwargs)
    # an async step, or a PP stage's boundary handed on while its forward still runs
    queued = bool(getattr(getattr(result, "logits_output", None), "neuron_async", False)
                  or getattr(getattr(result, "pp_hidden_states_proxy_tensors", None), "neuron_launched", False))
    tr.add("launch", t0, time.time_ns(), host_trace.describe(batch) + ("+a" if queued else ""))
    if queued:
        tr.flush(host_trace.FLUSH_ROWS // 4)
    return result


def on_idle(original, scheduler, *args, **kwargs):
    tr = host_trace.trace()
    t0 = time.time_ns()
    out = original(scheduler, *args, **kwargs)
    if tr is not None:
        tr.add("idle", t0, time.time_ns(), "-")
        tr.at_idle(_pending(scheduler))
    return out


def defining_target(target: str) -> str:
    """``target`` moved to the class that defines it, for an inherited method (Scheduler._pp_recv_pyobj_from_prev_stage
    is SchedulerPPMixin's): a hook on the subclass would shadow the defining class's hooks instead of wrapping them. A
    target that does not resolve is returned as is, for the registry to refuse by name."""
    import pkgutil

    owner_name, _, name = target.rpartition(".")
    try:
        owner = pkgutil.resolve_name(owner_name)
    except (ImportError, AttributeError, ValueError):
        return target
    if not isinstance(owner, type) or name in vars(owner):
        return target
    definer = next((cls for cls in owner.__mro__ if name in vars(cls)), None)
    return target if definer is None else f"{definer.__module__}.{definer.__qualname__}.{name}"


def register_hooks(registry, around) -> None:
    if not envs.host_trace_dir():
        return
    S = "sglang.srt.managers.scheduler.Scheduler"
    hooks = [
        ("sglang.srt.managers.scheduler.dispatch_event_loop", dispatch_event_loop),
        (f"{S}.run_batch", run_batch),
        (f"{S}.on_idle", on_idle),
        ("sglang.srt.managers.scheduler_components.request_receiver.SchedulerRequestReceiver.recv_requests",
         _timed("recv", lambda a, k, out: len(out) if out is not None else 0)),
        (f"{S}.process_input_requests", _timed("input", lambda a, k, out: len(a[1]) if len(a) > 1 else 0)),
        (f"{S}.get_next_batch_to_run",
         _timed("select", lambda a, k, out: host_trace.describe(getattr(out, "batch_to_run", None)))),
        (f"{S}.process_batch_result", process_batch_result),
        ("sglang.srt.managers.scheduler_components.batch_result_processor.SchedulerBatchResultProcessor."
         "_handle_finish_state_updated_req", handle_finish_state),
        ("sglang.srt.managers.scheduler_components.output_streamer.SchedulerOutputStreamer.stream_output",
         _timed("stream", lambda a, k, out: len(a[1]) if len(a) > 1 else 0)),
        ("sglang_neuron.model_executor.async_execution.CompletionEvent.synchronize",
         _timed("wait", lambda a, k, out: "-")),
    ]
    # one row per extra span, named by the target's last two components
    hooks += [(defining_target(target), _timed(".".join(target.split(".")[-2:]), lambda a, k, out: "-"))
              for target in envs.host_trace_spans()]
    for target, hook in hooks:
        registry.register(target, hook, around)
