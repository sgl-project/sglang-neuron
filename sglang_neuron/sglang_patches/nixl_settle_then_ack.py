# SPDX-License-Identifier: Apache-2.0
"""Ack a decode's ABORT of a NIXL room whose transfers the prefill could not prove settled, once they settle.

When a decode aborts a room, each prefill rank of the room acks the ABORT once none of its WRITEs can still land in the
decode's pages. A post is one transfer request the worker hands NIXL, tracked by its handle. Upstream's transfer worker
poisons a room whose transfers it cannot prove settled, and a poisoned room is never acked. That happens in two cases:
NIXL rejected a post (NIXL may already have WRITEs of it on the wire), or a failed chunk's handles still run after the
settle timeout. A decode that keeps a failed room's pages until its acks arrive (pd_decode_kv_quarantine) would then
keep them until the prefill restarts.

Here both poisonings become a settle hold. The room stays poisoned while it is held, and the hold keeps:

- every handle of the chunk and of every post the worker made for it, accepted or rejected (upstream drops a rejected
  post's handle);
- the room's ack targets, including ABORTs that arrive during the hold.

A reaper thread reads the held handles. Once every handle of a room reads done or failed, it lifts the poison and acks
the targets through upstream's own path. A handle whose state cannot be read counts as running until ``unreadable_s()``
after the hold took it. A room with a rejected post is never acked before then either, since NIXL never re-checks a
failed post. Holding the handles also keeps them alive: dropping a handle's last reference releases its request, and
LIBFABRIC deletes a request in progress without cancelling its WRITEs.

The worker also tells the room's decode ranks of the failure at once, where upstream leaves them to their timeout. That
is safe because every decode this prefill serves keeps a failed room's pages until the ack: nixl_head_major admits only
decodes running this plugin, and the platform keeps deferred KV release on. A settle read is proof only on NIXL's
LIBFABRIC backend with its progress thread, which reports a transfer's error once every one of its requests has
completed (patches/nixl 0007). The platform refuses any other backend.

Upstream: generic for NIXL's LIBFABRIC backend with patches/nixl 0007; UCX's status read is no such proof."""

from __future__ import annotations

import contextlib
import logging
import threading
import time
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_NIXL = "sglang.srt.disaggregation.nixl.conn"
_COMMON = "sglang.srt.disaggregation.common.conn"
TRANSFER_WORKER = (_NIXL, "NixlKVManager.transfer_worker")
REGISTER_TARGET = (_COMMON, "CommonKVManager.register_deferred_ack_target")
POST_XFER = (_NIXL, "NixlKVManager._post_xfer")
MAYBE_ACK = (_COMMON, "CommonKVManager._maybe_ack_drained_abort")
SEND_ACKS = (_COMMON, "CommonKVManager._send_abort_acks")
_TARGETS_MD5 = {
    TRANSFER_WORKER: "aaccdcf4cb029cd5aa2f01d265e85c25",
    REGISTER_TARGET: "7629a27d0f8a6add3c70a2b245efd7e8",
    (_COMMON, "CommonKVManager.poison_deferred_ack_room"): "1713c79fd0024debb06c8de12b72d332",
    MAYBE_ACK: "3dd95713461d31328f5f707e657e67b8",
    (_COMMON, "CommonKVManager.update_status"): "3957aa78394858a23f22fb85ef4243b6",
    SEND_ACKS: "3d758824ad34d6c8fb963d13a4cd1e6e",
    (_COMMON, "CommonKVManager._send_abort_ack"): "1a1d21f00855fd549cbd34e08e8942dd",
    (_NIXL, "NixlKVManager._handle_abort_notification"): "e6b31f79d46246334b8b43f14b8bd88b",
    (_NIXL, "NixlKVManager._handle_deferred_abort_ack"): "7b52b036c71a6c7e1044dce801f4dd4a",
    POST_XFER: "98788e15a47fbf7631846e2a7b7fcd6a",
    (_NIXL, "NixlKVManager._await_handles"): "316621564891c5ed504bfd66b423d702",
    (_NIXL, "_NixlPostError"): "5ca0f9575ff7010e7e168cdf168b392e",
    (_COMMON, "CommonKVManager.send_kv_status_message"): "8aeaa35a093c57c15f7c5173261340e8",
    (_COMMON, "CommonKVManager.record_failure"): "c356974ace971d8311aeaf794a2548dd",
}

UPSTREAM_START = '''    staging_strategy = None
'''
NEURON_START = '''    from sglang_neuron.sglang_patches import nixl_settle_then_ack

    staging_strategy = None
'''
UPSTREAM_CHUNK = '''        handles: List[Any] = []
'''
NEURON_CHUNK = '''        handles: List[Any] = []
        nixl_settle_then_ack.begin_chunk()
'''
UPSTREAM_REJECTED_POST = '''                    # The rejected request is not in `handles`, so its writes
                    # may still land; leave the room to decode's release timeout.
                    self.poison_deferred_ack_room(room)
'''
NEURON_REJECTED_POST = '''                    # The rejected request is not in `handles`, so its writes
                    # may still land: the room's drain ack waits until its
                    # handle, kept by the settle hold, settles.
                    nixl_settle_then_ack.hold(
                        self, room, (), rejected_post=True,
                        settled=nixl_settle_then_ack.chunk_handles(handles),
                    )
'''
UPSTREAM_UNSETTLED = '''                # This chunk stays counted in _staging_outstanding, so no
                # drain ACK can follow; discard the target and fall back to
                # the timeout.
                self.poison_deferred_ack_room(room)
'''
NEURON_UNSETTLED = '''                # The handles stay referenced in the room's settle hold, which
                # keeps its drain ack until every one of them settled, so the
                # decode, which keeps the pages until that ack, is told now.
                nixl_settle_then_ack.hold(
                    self,
                    room,
                    handles,
                    rejected_post=isinstance(e, _NixlPostError),
                    kv_chunk=kv_chunk,
                )
                nixl_settle_then_ack.tell_decode(self, room, room_transfer_infos, e)
'''
UPSTREAM_FAILED_AWAIT = '''                notify, _ = self._await_handles(handles, failure_seen=True)
'''
NEURON_FAILED_AWAIT = '''                notify, _ = self._await_handles(
                    nixl_settle_then_ack.chunk_handles(handles), failure_seen=True
                )
'''
TRANSFER_WORKER_SUBSTITUTIONS = (
    (UPSTREAM_START, NEURON_START),
    (UPSTREAM_CHUNK, NEURON_CHUNK),
    (UPSTREAM_FAILED_AWAIT, NEURON_FAILED_AWAIT),
    (UPSTREAM_REJECTED_POST, NEURON_REJECTED_POST),
    (UPSTREAM_UNSETTLED, NEURON_UNSETTLED),
)

REAP_POLL_S = 0.01
REPORT_S = 60.0
_STATE = "_neuron_settle_holds"
_CREATE = threading.Lock()
# on a transfer worker's thread: the (accepted, rejected) handles of the posts it made for its current chunk
_CHUNK = threading.local()


class _Locked(threading.local):
    """How deep this thread is inside settle locks, and the abort acks it queued meanwhile."""

    def __init__(self):
        self.depth, self.queued = 0, []


_LOCKED = _Locked()


def unreadable_s() -> float:
    """How long a handle whose state cannot be read counts as running: the longer of upstream's settle window and how
    long upstream's decode holds a failed room's pages (its deferred KV release timeout)."""
    from sglang.srt.disaggregation.nixl import conn
    from sglang.srt.environ import envs

    return max(conn.NIXL_ERR_SETTLE_TIMEOUT_S, envs.SGLANG_DISAGGREGATION_DEFERRED_DECODE_KV_RELEASE_TIMEOUT.get())


class _Held:
    """A room held for its unsettled transfers. Each handle comes with the time until which an unreadable state counts
    as running."""

    __slots__ = ("since", "handles", "rejected", "not_before", "targets", "reported", "unreadable")

    def __init__(self, now: float):
        self.since, self.handles, self.rejected, self.not_before = now, [], [], now
        self.targets, self.reported = {}, False
        self.unreadable = 0  # handles let go because their state still could not be read after their window

    def unsettled(self) -> int:
        return len(self.handles) + len(self.rejected)


class _Settle:

    def __init__(self):
        self.lock = threading.RLock()  # taken through settle_locked only
        self.rooms: Dict[int, _Held] = {}
        self.reaper: Optional[threading.Thread] = None
        self.unreadable_logged: set = set()  # id() of the held handles whose unreadable state was logged


def settle_of(mgr) -> _Settle:
    settle = mgr.__dict__.get(_STATE)
    if settle is None:
        with _CREATE:
            settle = mgr.__dict__.setdefault(_STATE, _Settle())
    return settle


@contextlib.contextmanager
def settle_locked(settle: _Settle):
    """Take the settle lock. Acks sent under it are queued, and go out once the thread leaves its outermost settle
    lock, since each ack is a ZMQ send that can block."""
    outermost = _LOCKED.depth == 0
    _LOCKED.depth += 1
    try:
        with settle.lock:
            yield
    finally:
        _LOCKED.depth -= 1
        if outermost:
            queued, _LOCKED.queued = _LOCKED.queued, []
            for send, mgr, room, targets in queued:
                try:
                    send(mgr, room, targets)
                except Exception:  # noqa: BLE001 - one failed ack must not stop the others
                    logger.exception("NIXL room %s: its drain ack failed", room)


def send_abort_acks(original_fn, self, room, targets):
    if _LOCKED.depth:
        _LOCKED.queued.append((original_fn, self, room, targets))
        return None
    return original_fn(self, room, targets)


def begin_chunk() -> None:
    _CHUNK.posts = ([], [])


def post_xfer(original_fn, self, xfer_handle, error_msg):
    """Record every post's handle for the chunk's settle hold, accepted or rejected: NIXL may have WRITEs of either on
    the wire."""
    posts = getattr(_CHUNK, "posts", None)
    try:
        result = original_fn(self, xfer_handle, error_msg)
    except Exception:
        if posts is not None:
            posts[1].append(xfer_handle)
        raise
    if posts is not None:
        posts[0].append(xfer_handle)
    return result


def chunk_handles(handles):
    posts = getattr(_CHUNK, "posts", None)
    out, seen = [], set()
    for h in (*handles, *(posts[0] if posts else ())):
        if id(h) not in seen:
            seen.add(id(h))
            out.append(h)
    return out


def _take_posts():
    posts = getattr(_CHUNK, "posts", None)
    if posts is None:
        return [], []
    _CHUNK.posts = ([], [])
    return posts


def hold(mgr, room: int, handles, rejected_post: bool, kv_chunk=None, settled=()) -> None:
    """Hold a failed room the worker cannot prove settled until its transfers settle."""
    accepted, rejected = _take_posts()
    rejected_post = rejected_post or bool(rejected)
    now = time.monotonic()
    # Hold each handle once. A rejected post's handle that the chunk already owned is held as a rejected one.
    seen = {id(h) for h in (*rejected, *settled)}
    unsettled = []
    for h in (*handles, *accepted):
        if id(h) not in seen:
            seen.add(id(h))
            unsettled.append(h)
    settle = settle_of(mgr)
    with settle_locked(settle):
        held = settle.rooms.get(room)
        if held is None:
            held = settle.rooms[room] = _Held(now)
        unreadable_until = now + unreadable_s()
        held.handles.extend((h, unreadable_until) for h in unsettled)
        if rejected_post:
            # NIXL keeps a failed post's status and never asks its backend again. So no read of the rejected handle
            # shows whether the WRITEs the post submitted before it failed have stopped, and the room waits as long as
            # upstream's decode itself holds a failed room's pages.
            held.not_before = max(held.not_before, now + unreadable_s())
            held.rejected.extend((h, unreadable_until) for h in rejected)
        held.targets.update(mgr._deferred_ack_targets.get(room) or {})
        mgr.poison_deferred_ack_room(room)  # drops the room's ack targets, which the hold took above
        # Uncount the chunk before a reaper pass can release the room. A pass that found it still counted would leave
        # the room unacked until the decode sends its ABORT again.
        if kv_chunk is not None and kv_chunk.staging_counted:
            kv_chunk.staging_counted = False
            mgr._staging_outstanding[room] -= 1
        _ensure_reaper(mgr, settle)
    logger.warning(
        "NIXL room %s failed with %d transfer(s) that may still write%s: its drain ack waits until they settle",
        room, len(unsettled), f" and {len(rejected)} rejected post(s)" if rejected_post else "",
    )


def tell_decode(mgr, room: int, room_transfer_infos, error) -> None:
    """Tell the room's decode ranks it failed. The targets come from the worker's snapshot of the room, which survives
    the sender's clear."""
    from sglang.srt.disaggregation.base.conn import KVPoll

    targets = list(dict.fromkeys((info.endpoint, info.dst_port) for info in (room_transfer_infos or {}).values()
                                 if not info.is_dummy))
    with mgr.failure_lock:
        reason = mgr.failure_records.get(room, str(error))
    mgr.send_kv_status_message(targets=targets, bootstrap_room=room, status=KVPoll.Failed, failure_reason=reason)


def register_deferred_ack_target(original_fn, self, room, target):
    settle = settle_of(self)
    with settle_locked(settle):
        held = settle.rooms.get(room)
        if held is None:
            return original_fn(self, room, target)
        held.targets[(target.ip, target.port)] = target
        _ensure_reaper(self, settle)
        return None


def maybe_ack_drained_abort(original_fn, self, room):
    # under the settle lock, so no hold or reaper pass runs between reading the poison and taking the targets
    with settle_locked(settle_of(self)):
        return original_fn(self, room)


# The NIXL status errors that carry a request's final status. NIXL's Python binding raises on every negative status,
# so check_xfer_state never returns ERR.
_FINAL_STATUS_ERRORS = frozenset({"nixlBackendError", "nixlRemoteDisconnectError", "nixlCancelledError",
                                  "nixlNotPostedError"})


def _final_status_error(error: BaseException) -> bool:
    return type(error).__name__ in _FINAL_STATUS_ERRORS or (type(error) is RuntimeError and str(error) == "BAD_STATUS")


SETTLED, RUNNING, UNREADABLE = "settled", "running", "unreadable"


def _state(mgr, settle: _Settle, handle, unreadable_until: float, now: float) -> str:
    """A held handle's state:

    - SETTLED: DONE, ERR, or a status error that carries the request's final status;
    - RUNNING: in progress, or unreadable while its window has not passed;
    - UNREADABLE: still unreadable after its window, which proves nothing about its WRITEs.

    The first unreadable read of a handle is logged."""
    try:
        return SETTLED if mgr.agent.check_xfer_state(handle) in ("DONE", "ERR") else RUNNING
    except Exception as e:  # noqa: BLE001 - every read error is a state: settled, running, or unreadable
        if _final_status_error(e):
            return SETTLED
        if id(handle) not in settle.unreadable_logged:
            settle.unreadable_logged.add(id(handle))
            logger.warning("NIXL transfer state could not be read (%s: %s); it counts as running for up to %.0fs",
                           type(e).__name__, e, max(unreadable_until - now, 0.0))
        return UNREADABLE if now >= unreadable_until else RUNNING


def reap_once(mgr, settle: _Settle, now: float) -> List[int]:
    """One reaper pass: read the held handles outside the lock, then release every room whose handles all settled."""
    with settle_locked(settle):
        pending = [h for held in settle.rooms.values() for h in held.handles]
        rejected = [r for held in settle.rooms.values() for r in held.rejected]
    # the holds keep each handle alive, so no other object can take its id()
    states = {id(h): _state(mgr, settle, h, unreadable_until, now) for h, unreadable_until in (*pending, *rejected)}
    released = []
    with settle_locked(settle):
        for room, held in list(settle.rooms.items()):
            for handle, _ in (*held.handles, *held.rejected):
                if states.get(id(handle), RUNNING) != RUNNING:
                    held.unreadable += states[id(handle)] == UNREADABLE
                    settle.unreadable_logged.discard(id(handle))
            held.handles = [r for r in held.handles if states.get(id(r[0]), RUNNING) == RUNNING]
            held.rejected = [r for r in held.rejected if states.get(id(r[0]), RUNNING) == RUNNING]
            if held.unsettled() or now < held.not_before:
                if not held.reported and now - held.since >= REPORT_S:
                    held.reported = True
                    logger.error(
                        "NIXL room %s: %d transfer(s) still unsettled after %.0fs; its drain ack, and the decode's "
                        "pages, stay held until they settle", room, held.unsettled(), now - held.since,
                    )
                continue
            del settle.rooms[room]
            mgr._deferred_ack_poisoned_rooms.discard(room)
            for target in held.targets.values():
                mgr.register_deferred_ack_target(room, target)
            try:
                mgr._maybe_ack_drained_abort(room)  # its acks are sent once the lock is released
            except Exception:  # noqa: BLE001 - the decode asks again for a lost ack; the other rooms go on
                logger.exception("NIXL room %s: its drain ack failed", room)
            released.append((room, held))
        if not settle.rooms:
            settle.reaper = None
    for room, held in released:
        if held.unreadable:
            logger.error("NIXL room %s: released after %.1fs with %d transfer(s) whose state could not be read, without "
                         "proof that none still writes; its drain ack goes to %d decode rank(s)", room,
                         now - held.since, held.unreadable, len(held.targets))
        else:
            logger.warning("NIXL room %s: its transfers settled after %.1fs; its drain ack goes to %d decode rank(s)",
                           room, now - held.since, len(held.targets))
    return [room for room, _ in released]


def _reap(mgr, settle: _Settle) -> None:
    while True:
        try:
            reap_once(mgr, settle, time.monotonic())
            with settle_locked(settle):
                if settle.reaper is None or settle.reaper is not threading.current_thread():
                    return  # nothing held (a later hold starts a new reaper)
        except Exception:  # noqa: BLE001 - the holds wait for the next pass, never for a restart
            logger.exception("NIXL settle reaper pass failed")
        time.sleep(REAP_POLL_S)


def _ensure_reaper(mgr, settle: _Settle) -> None:
    """Start the manager's reaper unless one is running. The caller holds the settle lock. A start that fails is
    logged, and the next hold or ABORT of a held room tries again."""
    if settle.reaper is not None and settle.reaper.is_alive():
        return
    reaper = threading.Thread(target=_reap, args=(mgr, settle), name="nixl-settle-reaper", daemon=True)
    settle.reaper = reaper
    try:
        reaper.start()
    except Exception:  # noqa: BLE001 - a failed start must not end the transfer worker or the bootstrap thread
        settle.reaper = None
        logger.exception("NIXL settle reaper failed to start")


def settling_transfer_worker():
    """NixlKVManager.transfer_worker: upstream's, with the settle holds in place of its poisonings
    (TRANSFER_WORKER_SUBSTITUTIONS), and no aux write for a NO_AUX last chunk (pp_prefill_last_send)."""
    from sglang_neuron.sglang_patches import source_guard

    from sglang_neuron.sglang_patches import pp_prefill_last_send

    # + the aux write skipped for a last chunk that a prefill stage sent with no metadata row (NO_AUX)
    return source_guard.with_substitutions(source_guard.resolve(*TRANSFER_WORKER),
                                           TRANSFER_WORKER_SUBSTITUTIONS + pp_prefill_last_send.WORKER_SUBSTITUTIONS)


def register(registry, around, replace) -> None:
    from sglang_neuron.sglang_patches import source_guard

    source_guard.check_targets(_TARGETS_MD5, __name__)
    registry.register(".".join(TRANSFER_WORKER), settling_transfer_worker(), replace)
    registry.register(".".join(REGISTER_TARGET), register_deferred_ack_target, around)
    registry.register(".".join(POST_XFER), post_xfer, around)
    registry.register(".".join(MAYBE_ACK), maybe_ack_drained_abort, around)
    registry.register(".".join(SEND_ACKS), send_abort_acks, around)
