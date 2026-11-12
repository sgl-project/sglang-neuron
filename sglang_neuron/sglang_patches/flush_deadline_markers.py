# SPDX-License-Identifier: Apache-2.0
"""Time out a deferred flush_cache at the same scheduler iteration on every rank.

SGLang's ``SchedulerFlushWrapper`` holds a flush_cache that has a timeout (a timed flush) until the scheduler is idle,
and times it out by each rank's own clock. On a scheduler of more than one rank, a flush that races the server going
idle can flush on some ranks and time out on others, which leaves their prefix caches different.

Here only the request leader's clock counts. The request leader is the rank that receives requests from the tokenizer
and broadcasts them. Once a timed flush's deadline passes by its clock, it puts a ``FlushDeadlineReached`` marker ahead
of the requests it receives. The broadcast delivers the marker to every rank in the same iteration, and each rank
expires the flush there: it flushes if the scheduler is idle, and times out otherwise.

Ranks number timed flushes in the order they receive them. A marker expires the flush it names or an earlier one, never
a later one. A timed flush that upstream refuses, because another is pending, sends no marker. A single-rank scheduler
keeps its own clock. The DP control fence (managers/dp_identity) keeps the markers out of its counted control sequence
and delivers one per rendezvous on every rank.

Targets, md5-pinned against upstream SGLang 43881a7:

- ``flush_wrapper.SchedulerFlushWrapper``, replaced by its subclass ``NeuronFlushWrapper``, which is also bound to the
  scheduler module's imported name;
- ``Scheduler.init_request_dispatcher``, for the marker's handler;
- ``Scheduler.init_request_receiver`` and ``SchedulerRequestReceiver._pull_raw_reqs``, for the leader's markers.

Upstream: a fix SGLang can take: a timed flush expires at an iteration every rank agrees on instead of on each rank's
own clock."""

from __future__ import annotations

import logging
import time
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# how long after its deadline a rank keeps a timed flush's record; the request leader sends its marker long before
_DEADLINE_RECORD_S = 600.0
_FLUSH_WRAPPER = "sglang.srt.managers.scheduler_components.flush_wrapper"
_SCHEDULER = "sglang.srt.managers.scheduler"
_RECEIVER = "sglang.srt.managers.scheduler_components.request_receiver"
# (module, qualified name): md5 of upstream's source
_TARGETS_MD5 = {
    (_FLUSH_WRAPPER, "SchedulerFlushWrapper"): "1cf9467ea03aac1291ea779ac4f877cf",
    (_SCHEDULER, "Scheduler.init_request_dispatcher"): "6f6131faf815fcc02360bb01d9530c4f",
    (_SCHEDULER, "Scheduler.init_request_receiver"): "82f952f443fbd61594c4a4b6dc1c8b27",
    (_RECEIVER, "SchedulerRequestReceiver._pull_raw_reqs"): "ada9a1afb295b795db184413c26c05e4",
}


class FlushDeadlineReached:
    """The request leader's clock has passed the deadline of timed flush number ``flush``. Ranks number the timed
    flushes they receive from 0, in order. A control request: it rides the control broadcast."""

    __slots__ = ("flush",)

    def __init__(self, flush: int):
        flush = int(flush)
        if flush < 0:
            raise ValueError(f"a flush deadline marker names a timed flush by its number from 0, not {flush}")
        self.flush = flush

    def __reduce__(self):
        return (FlushDeadlineReached, (self.flush,))

    def __repr__(self) -> str:
        return f"FlushDeadlineReached(flush={self.flush})"


def deadline_by_marker() -> bool:
    """True on a scheduler of more than one rank, whose flush decisions must agree."""
    from sglang.srt.runtime_context import get_parallel

    return get_parallel().tp_size > 1


def make_wrapper_class(base):
    from sglang.srt.managers.io_struct import FlushCacheReqOutput

    class NeuronFlushWrapper(base):
        """SGLang's flush wrapper, timed out by the request leader's deadline markers (see the module docstring)."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._by_marker: Optional[bool] = None  # resolved at the first flush, once the groups are built
            self._number: Optional[int] = None  # the pending flush's number among the timed flushes received
            self._timed_received = 0
            self._deadlines: List[Tuple[int, float]] = []

        def _marker_mode(self) -> bool:
            if self._by_marker is None:
                self._by_marker = deadline_by_marker()
            return self._by_marker

        def handle(self, recv_req):
            timeout_s = float(recv_req.timeout_s or 0.0)
            number = -1
            if timeout_s > 0.0 and self._marker_mode():
                now = time.monotonic()
                number = self._timed_received
                self._timed_received += 1
                self._deadlines = [d for d in self._deadlines if now < d[1] + _DEADLINE_RECORD_S]
                self._deadlines.append((number, now + timeout_s))
            pending_before = self._pending
            out = super().handle(recv_req)
            if self._pending is not None and self._pending is not pending_before:
                self._number = number
            elif number >= 0 and pending_before is not None:
                # Refused while another flush is pending. It keeps its number, so every rank still numbers alike, but
                # sends no marker: its marker could expire the pending flush before that flush's own deadline.
                self._deadlines = [d for d in self._deadlines if d[0] != number]
            return out

        def check_pending(self) -> None:
            if self._pending is None:
                return
            if not self._marker_mode():
                return super().check_pending()
            if self._is_fully_idle():
                super().check_pending()  # idle: flushes and answers
            # otherwise the request leader's marker decides (expire), never this rank's clock

        def deadline_markers(self) -> List[FlushDeadlineReached]:
            """On the request leader, one marker for each received timed flush whose deadline has passed, whether or
            not this rank still holds that flush. Each flush gets its marker once, ahead of the requests the leader
            receives."""
            if not self._deadlines or not self._marker_mode():
                return []
            now = time.monotonic()
            due = [number for number, deadline in self._deadlines if now >= deadline]
            if not due:
                return []
            self._deadlines = [d for d in self._deadlines if now < d[1]]
            return [FlushDeadlineReached(flush=number) for number in due]

        def expire(self, marker: FlushDeadlineReached) -> None:
            if self._pending is None:
                return None
            if self._number is not None and marker.flush < self._number:
                return None  # an earlier flush's marker leaves a later one pending
            if self._is_fully_idle():
                return super().check_pending()
            pending_req, _ = self._pending
            logging.warning("Deferred flush_cache timed out while waiting for idle state.")
            self._pending = None
            self._number = None
            self._send_output(FlushCacheReqOutput(success=False, message="Timed out waiting for idle state."),
                              pending_req)
            return None

    return NeuronFlushWrapper


def init_request_dispatcher(original_fn, self, *args, **kwargs):
    """AROUND Scheduler.init_request_dispatcher: the scheduler dispatches each marker to the flush wrapper's
    ``expire``."""
    out = original_fn(self, *args, **kwargs)
    from sglang.utils import TypeBasedDispatcher

    self._request_dispatcher += TypeBasedDispatcher([(FlushDeadlineReached, self.flush_wrapper.expire)])
    return out


def make_receiver_class(base):
    """SGLang's request receiver (a frozen, slotted dataclass) with the flush wrapper's markers as one more field."""
    import dataclasses
    from typing import Callable

    @dataclasses.dataclass(kw_only=True, slots=True, frozen=True)
    class NeuronRequestReceiver(base):
        lead_reqs: Optional[Callable[[], List[FlushDeadlineReached]]] = None

    return NeuronRequestReceiver


def pull_raw_reqs(original_fn, self, *args, **kwargs):
    """AROUND SchedulerRequestReceiver._pull_raw_reqs: the request leader (the rank whose pull returns a list from the
    tokenizer) puts the due deadline markers ahead of the requests."""
    reqs = original_fn(self, *args, **kwargs)
    lead = getattr(self, "lead_reqs", None)  # NeuronRequestReceiver's
    if reqs is None or lead is None:
        return reqs
    markers = lead()
    return markers + reqs if markers else reqs


_RECEIVER_CLASSES: dict = {}  # SGLang's receiver class -> NeuronRequestReceiver


def init_request_receiver(original_fn, self, *args, **kwargs):
    """AROUND Scheduler.init_request_receiver: rebuild the scheduler's receiver as a NeuronRequestReceiver with the
    same fields plus the flush wrapper's markers. SGLang's receiver is frozen, so it cannot take a new attribute."""
    import dataclasses

    out = original_fn(self, *args, **kwargs)
    receiver = self.request_receiver
    base = type(receiver)
    cls = _RECEIVER_CLASSES.get(base)
    if cls is None:
        cls = _RECEIVER_CLASSES[base] = make_receiver_class(base)
    fields = {f.name: getattr(receiver, f.name) for f in dataclasses.fields(receiver) if f.init}
    self.request_receiver = cls(**fields, lead_reqs=self.flush_wrapper.deadline_markers)
    return out


def register(registry, around) -> None:
    from sglang.srt.plugins.hook_registry import HookType

    from sglang_neuron.sglang_patches import source_guard

    source_guard.check_targets(_TARGETS_MD5, __name__)
    base = source_guard.resolve(_FLUSH_WRAPPER, "SchedulerFlushWrapper")
    registry.register(f"{_FLUSH_WRAPPER}.SchedulerFlushWrapper", make_wrapper_class(base), HookType.REPLACE)
    registry.register(f"{_SCHEDULER}.Scheduler.init_request_dispatcher", init_request_dispatcher, around)
    registry.register(f"{_SCHEDULER}.Scheduler.init_request_receiver", init_request_receiver, around)
    registry.register(f"{_RECEIVER}.SchedulerRequestReceiver._pull_raw_reqs", pull_raw_reqs, around)
