# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""Keep a NIXL decode's KV pages for a failed or aborted transfer until every prefill rank of the room has acked that
none of its WRITEs can still land in them.

Upstream's ``resolve_deferred_releases`` frees such pages at a deadline even without the acks, and an ABORT or ack lost
on the wire is never sent again. A WRITE that lands after the pages went to another request corrupts that request's KV
without any error.

Here a hold of device pages is released only when every prefill rank of the room has acked or, past its deadline, when
the heartbeat has declared that prefill dead (its registration was dropped). Releasing on a dead prefill is a
heuristic: an unreachable but live prefill is declared dead too. Without that release, a dead prefill's rooms would
stay held until the decode restarts.

Past its deadline a hold stays, and its ABORT goes again to every prefill rank of the room, at once and then with
doubling waits. A re-send never blocks. It carries the room's ack generation, so acks already counted still count.
Holds of host pages, and holds on another KV backend, keep upstream's behaviour.

Every rank of the attention-TP group holds the same entries in the same order. A hold is released only once every rank
reads it releasable, through upstream's MIN all-reduce, in the pass each rank runs at the same point of its loop. The
all-reduce runs every pass for the first ``FAST_PASSES`` passes after a hold is added, then every ``RESOLVE_EVERY``-th
pass, so an idle decode does not run a collective every pass for a quarantine. On a pipelined decode the group only
agrees: the stages release a hold together once every stage's group agreed it (sglang_patches/pp_decode_held_release,
``agreed_holds`` and ``release_agreed``). While anything is held, its pages stay allocated and the server does not
report idle. The quarantine needs SGLang's deferred KV release, which the platform turns on for every NIXL role.

Upstream: generic for NIXL; upstream's deferred release would stop freeing device pages at the deadline while acks
are missing."""

from __future__ import annotations

import logging
import time
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

_DECODE = "sglang.srt.disaggregation.decode"
_COMMON = "sglang.srt.disaggregation.common.conn"
_NIXL = "sglang.srt.disaggregation.nixl.conn"
_UTILS = "sglang.srt.disaggregation.utils"
_SCHEDULER = "sglang.srt.managers.scheduler"
TARGET = f"{_DECODE}.DecodeTransferQueue.resolve_deferred_releases"
_TARGETS_MD5 = {
    (_DECODE, "DecodeTransferQueue.resolve_deferred_releases"): "c2c2c335a27dad608da623b902715b98",
    # an entry is (decode_req, start, deadline, metadata slot, required acks); pop_transferred delegates to
    # pop_transferred_with_outcomes, which appends the agreed holds. A room whose destination no prefill rank was sent
    # (a deferred publish, a held send) is held for a peer rank only and waits for no ack
    (_DECODE, "DecodeTransferQueue.pop_transferred"): "734e67e1e066ea0df2d540fbb7c32fc2",
    (_DECODE, "DecodeTransferQueue.pop_transferred_with_outcomes"): "eefc7913311dd1a015b87b20faaf3de6",
    (_DECODE, "DecodeTransferQueue._defer_release"): "60935590b6a464dec14c18537964ef16",
    (_DECODE, "DecodeTransferQueue._do_release"): "8d3d458c315a9655797545f857552edd",
    (_DECODE, "DecodeTransferQueue._is_host_destination"): "b0b728ec184d5def1bfc29a701c0acd1",
    (_DECODE, "DecodeTransferQueue._observe_deferred_kv_release"): "ce9aac73dab17ec7799f4d0d61777ef2",
    (_DECODE, "DecodeTransferQueue.has_pending_deferred_releases"): "48cd0762b541a784b22f7e37b76a55b1",
    (_DECODE, "DecodeTransferQueue.release_memory_occupation"): "0fef59c463c16cff8ceac3bde0732244",
    (_COMMON, "CommonKVReceiver.ensure_abort_notified"): "e92d994aff6f8e1a224cc2fe03e612cc",
    (_UTILS, "all_reduce_flags"): "7a6299fea1c3571aba1d517d3feca38e",
    # the pass every rank runs at the same point of its loop
    (_DECODE, "SchedulerDisaggregationDecodeMixin.process_decode_queue"): "d1df3418bea2696da4c1ffacd3fa7bb0",
    # acks are counted per room under the generation its ABORT carried
    (_COMMON, "CommonKVManager.register_deferred_abort_room"): "034e96df7be819cb37e2cca35ac8ba32",
    (_COMMON, "CommonKVManager.note_abort_ack"): "b5e35b6e220f2f947246ebf9846f437b",
    (_COMMON, "CommonKVManager.handle_abort_ack_message"): "5a03f707892dcb1a275d9864e4138e4d",
    (_COMMON, "CommonKVManager.is_abort_release_safe"): "f7acf57ab690516719affe7683f61258",
    (_COMMON, "CommonKVManager.clear_deferred_abort_state"): "d6c18d09036971a0eca14a16d542830c",
    # a re-send goes on the socket the receiver's own ABORT went through
    (_COMMON, "CommonKVReceiver._send_abort_notification"): "a7fa2defc0059e56610adbec895e5c3b",
    (_COMMON, "CommonKVReceiver._connect_to_bootstrap_server"): "931f78e0263aeb8443eaa97ddeeab42a",
    (_COMMON, "CommonKVReceiver._connect"): "bfe8697f02c690a29300cf36a0b9c8f3",
    (_COMMON, "AbortNotification.to_zmq"): "137be85295de5eabf1942a2f09fd97c9",
    (_COMMON, "AbortNotification.from_zmq"): "63447705adae1a30796f92b28e7e085c",
    (_COMMON, "AbortNotification.deferred_ack_target"): "6aa51c5692951de80cf07c6a5b083de9",
    # the prefill acks every ABORT of a room it can prove drained
    (_NIXL, "NixlKVManager._handle_abort_notification"): "e6b31f79d46246334b8b43f14b8bd88b",
    (_NIXL, "NixlKVManager._handle_deferred_abort_ack"): "7b52b036c71a6c7e1044dce801f4dd4a",
    (_COMMON, "CommonKVManager.register_deferred_ack_target"): "7629a27d0f8a6add3c70a2b245efd7e8",
    (_COMMON, "CommonKVManager._maybe_ack_drained_abort"): "3dd95713461d31328f5f707e657e67b8",
    (_COMMON, "CommonKVManager.poison_deferred_ack_room"): "1713c79fd0024debb06c8de12b72d332",
    (_COMMON, "CommonKVManager._send_abort_acks"): "3d758824ad34d6c8fb963d13a4cd1e6e",
    (_COMMON, "CommonKVManager._send_abort_ack"): "1a1d21f00855fd549cbd34e08e8942dd",
    # a prefill's registration leaves the table only when the heartbeat declares it dead
    (_COMMON, "CommonKVManager._start_heartbeat_checker_thread"): "4b82f323219fc1b315be3ba5ee716cb4",
    (_COMMON, "CommonKVManager._handle_node_failure"): "6e9833cfe13f0603968dbeb2ff1bc091",
    (_COMMON, "CommonKVManager.try_ensure_parallel_info"): "15b2d05442275911ca74f44980700aac",
    (_COMMON, "CommonKVReceiver.init"): "fe1b27f354afbc1dfb46cf7dfa0be764",
    (_COMMON, "CommonKVReceiver._clear"): "8d04db4cf98d46c6677fd778efb08f4c",
    # not idle while anything is held, and no idle leak check
    (_SCHEDULER, "Scheduler.is_fully_idle"): "28e67960ff8d30595eab2437fe29b45e",
    (_SCHEDULER, "Scheduler.on_idle"): "905885a538273f5271e11bf93c08848c",
}

ABORT_RESEND_S = 5.0
ABORT_RESEND_MAX_S = 60.0
REPORT_S = 60.0
FAST_PASSES = 64
RESOLVE_EVERY = 16
DRAINED, DEAD, TIMEOUT = "drained", "dead", "timeout"


class _Held:
    """A hold past its deadline on this rank."""

    __slots__ = ("since", "tokens", "prefill", "resend_at", "wait")

    def __init__(self, now: float, tokens: int, prefill: str):
        self.since, self.tokens, self.prefill = now, tokens, prefill
        self.resend_at, self.wait = now, ABORT_RESEND_S


class _Quarantine:
    """A transfer queue's holds past their deadline, keyed by their entry's id(). Each entry is kept alongside, so no
    other object can take its id."""

    __slots__ = ("held", "report_at", "passes", "length", "agreed")

    def __init__(self):
        self.held: Dict[int, Tuple[tuple, _Held]] = {}
        self.report_at: Optional[float] = None
        self.passes = 0
        self.length = 0
        # pipelined decode only: holds this stage's group agreed releasable, hold_key -> reason (DRAINED, DEAD, TIMEOUT)
        self.agreed: Dict[Tuple[str, int], str] = {}


_STATE = "_neuron_kv_quarantine"


def _quarantine(queue) -> _Quarantine:
    quarantine = queue.__dict__.get(_STATE)
    if quarantine is None:
        quarantine = queue.__dict__[_STATE] = _Quarantine()
    return quarantine


def _kv_tokens(decode_req) -> int:
    kv = getattr(decode_req.req, "kv", None)
    return max(0, getattr(kv, "kv_allocated_len", 0) - getattr(kv, "cache_protected_len", 0))


def _acks(kv_mgr, room: int) -> int:
    state = kv_mgr._deferred_abort_ack_tracker.get(room)
    return len(state.prefill_ranks) if state is not None else 0


def prefill_declared_dead(receiver) -> bool:
    """Whether the heartbeat declared the room's prefill dead since the receiver was set up against it. A receiver
    that was never set up against a prefill has no prefill that knows its pages."""
    registration = getattr(receiver, "prefill_info", None)
    return registration is None or receiver.kv_mgr.prefill_info_table.get(receiver.bootstrap_addr) is not registration


def quarantines(receiver) -> bool:
    from sglang.srt.disaggregation.nixl.conn import NixlKVManager

    return isinstance(receiver.kv_mgr, NixlKVManager)


def resend_abort(receiver) -> None:
    """Send the room's ABORT again to every prefill rank without blocking. A busy socket or a full queue is skipped
    until the next re-send. No prefill acks an ABORT without an ack generation, so a room whose ABORT went out without
    one is given one first."""
    import zmq
    from sglang.srt.disaggregation.common.conn import AbortNotification

    kv_mgr = receiver.kv_mgr
    if receiver._abort_generation is None:
        receiver._abort_generation = kv_mgr.register_deferred_abort_room(receiver.bootstrap_room)
    frames = AbortNotification(
        room=receiver.bootstrap_room,
        decode_ip=kv_mgr.local_ip,
        decode_port=kv_mgr.rank_port,
        generation=receiver._abort_generation,
    ).to_zmq()
    for bootstrap_info in receiver.bootstrap_infos or ():
        try:
            sock, lock = receiver._connect_to_bootstrap_server(bootstrap_info)
            if not lock.acquire(blocking=False):
                continue
            try:
                sock.send_multipart(frames, flags=zmq.NOBLOCK)
            finally:
                lock.release()
        except Exception as e:  # noqa: BLE001 - zmq.Again (a full queue) or a lost peer: the next re-send tries again
            logger.debug(
                "Re-sending the ABORT of room %s to %s:%s failed: %s", receiver.bootstrap_room,
                bootstrap_info.get("rank_ip"), bootstrap_info.get("rank_port"), e,
            )


def _resolve_due(quarantine: _Quarantine, length: int) -> bool:
    # every rank of the group counts the same passes and sees the list grow at the same ones
    quarantine.passes = 1 if length > quarantine.length else quarantine.passes + 1
    quarantine.length = length
    return quarantine.passes <= FAST_PASSES or quarantine.passes % RESOLVE_EVERY == 0


def resolve_deferred_releases(self) -> None:
    if not self._deferred_releases:
        stale = self.__dict__.get(_STATE)
        if stale is not None:
            # reset, including holds dropped without a release (release_memory_occupation tears the pool down)
            stale.held, stale.report_at, stale.passes, stale.length = {}, None, 0, 0
            stale.agreed = {}
        return
    quarantine = _quarantine(self)
    if not _resolve_due(quarantine, len(self._deferred_releases)):
        return
    import torch
    from sglang.srt.disaggregation.utils import all_reduce_flags

    now = time.monotonic()
    # Why this rank may release each hold: DRAINED (every prefill rank acked), DEAD (past its deadline, its prefill
    # declared dead), TIMEOUT (past its deadline, on another backend than NIXL: upstream's timed release), or None.
    reasons = []
    for decode_req, _, deadline, _, required_acks in self._deferred_releases:
        receiver = decode_req.kv_receiver
        if self._is_host_destination(decode_req):
            # A timeout cannot prove that a remote write has stopped.
            reasons.append(DRAINED if receiver.is_abort_release_safe() else None)
            continue
        quarantined = quarantines(receiver)
        if quarantined and required_acks and receiver._abort_generation is None:
            # Rank-local: its ABORT went out without an ack generation, and no prefill rank acks it until it goes
            # again with one.
            resend_abort(receiver)
        if receiver.kv_mgr.is_abort_release_safe(decode_req.req.bootstrap_room, required_acks):
            reasons.append(DRAINED)
        elif now < deadline:
            reasons.append(None)
        elif not quarantined:
            reasons.append(TIMEOUT)
        else:
            reasons.append(DEAD if prefill_declared_dead(receiver) else None)
    # Acks, deadlines and death declarations are rank-local. Releasing only on some ranks would diverge admission and
    # the sizes of subsequent polling collectives. Entries are appended from agreed holds, so every rank reduces the
    # same list in the same order.
    ready = all_reduce_flags(
        flags=[reason is not None for reason in reasons],
        group=self.gloo_group,
        op=torch.distributed.ReduceOp.MIN,
    )
    held_before, quarantine.held = quarantine.held, {}
    still_held = []
    to_release = []
    across_stages = _pipelined()
    for entry, is_ready, reason in zip(self._deferred_releases, ready, reasons):
        decode_req, start_time, deadline, idx, required_acks = entry
        before = held_before.get(id(entry))
        held = before[1] if before is not None and before[0] is entry else None
        if is_ready and across_stages:
            # released with the other stages once all of them agreed (release_agreed); the first agreed reason sticks
            quarantine.agreed.setdefault(hold_key(entry), reason)
            still_held.append(entry)
            if held is not None:
                quarantine.held[id(entry)] = (entry, held)
            continue
        if is_ready:
            to_release.append((decode_req, start_time, deadline, idx, reason, held))
            continue
        still_held.append(entry)
        receiver = decode_req.kv_receiver
        if now < deadline or self._is_host_destination(decode_req) or not quarantines(receiver):
            # Not quarantined: a hold before its deadline, of host pages (only their receiver releases them), or on
            # another backend (upstream's timed release frees it once every rank of the group is past its deadline).
            continue
        room = decode_req.req.bootstrap_room
        if held is None:
            held = _Held(now, _kv_tokens(decode_req), receiver.bootstrap_addr)
            if reason == DRAINED:
                waits = "; this rank has every ack, another rank of its group still waits"
            elif reason == DEAD:
                waits = "; declared dead on this rank, another rank of its group still waits"
            else:
                waits = "; its ABORT is sent again"
            logger.error(
                "Deferred KV release for room %s: %d of %d prefill drain acks after %ss; its %d KV tokens stay held "
                "(quarantined) until every prefill rank of %s acks or that prefill is declared dead%s",
                room, _acks(receiver.kv_mgr, room), required_acks, self.deferred_kv_release_timeout, held.tokens,
                held.prefill, waits,
            )
        quarantine.held[id(entry)] = (entry, held)
        if reason is None and now >= held.resend_at:
            # rank-local, like the acks; a prefill declared dead is not asked, since it cannot answer
            resend_abort(receiver)
            held.resend_at = now + held.wait
            held.wait = min(2 * held.wait, ABORT_RESEND_MAX_S)
    # Commit the survivors before releasing so a _do_release exception can't leave a released entry in the list
    # (double-free / None receiver on retry).
    self._deferred_releases = still_held
    quarantine.length = len(still_held)
    _release(self, to_release, now)
    _report(self, quarantine, now)


def _pipelined() -> bool:
    from sglang_neuron import serving_policy

    return serving_policy.current().pp_size > 1


def hold_key(entry: tuple) -> Tuple[str, int]:
    """A hold's request as (rid, bootstrap room): the same on every stage whatever order it made its holds in, and never
    that of a later request reusing the rid, which comes with its own room."""
    req = entry[0].req
    return req.rid, int(req.bootstrap_room)


def agreed_holds(queue) -> list:
    """Pipelined decode: the hold_keys this stage's attention-TP group agreed releasable, in hold order."""
    quarantine = queue.__dict__.get(_STATE)
    if quarantine is None or not quarantine.agreed:
        return []
    keys = (hold_key(entry) for entry in queue._deferred_releases)
    return [key for key in keys if key in quarantine.agreed]


def release_agreed(queue, keys) -> list:
    """Release on this stage the holds named by ``keys``, which every stage's group agreed (the release consensus);
    returns the keys released. A key no longer held went at an earlier consensus, on every stage alike: a stage keeps
    sending its agreement until the first consensus naming the hold reaches it."""
    quarantine = _quarantine(queue)
    wanted = set(keys)
    released, to_release, still_held = [], [], []
    for entry in queue._deferred_releases:
        decode_req, start_time, deadline, idx, _ = entry
        key = hold_key(entry)
        if key not in wanted:
            still_held.append(entry)
            continue
        record = quarantine.held.pop(id(entry), None)
        held = record[1] if record is not None and record[0] is entry else None
        to_release.append((decode_req, start_time, deadline, idx, quarantine.agreed.pop(key), held))
        released.append(key)
    queue._deferred_releases = still_held
    # take only the released holds off the count the last resolve saw: a hold made since still counts as new
    quarantine.length -= len(to_release)
    _release(queue, to_release, time.monotonic())
    return released


def _release(self, to_release, now: float) -> None:
    for decode_req, start_time, deadline, idx, reason, held in to_release:
        room = decode_req.req.bootstrap_room
        if reason == DEAD:
            logger.warning(
                "Deferred KV release for room %s: released %.0fs past its deadline without every prefill drain ack, "
                "its prefill %s declared dead", room, now - deadline, decode_req.kv_receiver.bootstrap_addr,
            )
        elif reason == TIMEOUT:
            logger.warning(
                "Deferred KV release for room %s timed out after %ss without a full drain ack from prefill; releasing "
                "anyway (upstream's timed release: a %s decode is not quarantined)", room,
                self.deferred_kv_release_timeout, type(decode_req.kv_receiver.kv_mgr).__name__,
            )
        elif held is not None:
            logger.warning(
                "Deferred KV release for room %s: released after %.0fs in quarantine, every prefill rank acked", room,
                now - held.since,
            )
        outcome = "drained" if reason == DRAINED else "timeout"
        try:
            self._do_release(decode_req, idx)
        except Exception:
            # Isolate a failed release so the rest still run; entry already dropped.
            logger.exception("Deferred KV release failed for room %s", room)
            outcome = "error"
        self._observe_deferred_kv_release(duration_seconds=max(0.0, now - start_time), outcome=outcome)


def _report(queue, quarantine: _Quarantine, now: float) -> None:
    if not quarantine.held:
        quarantine.report_at = None
        return
    if quarantine.report_at is None:
        quarantine.report_at = now + REPORT_S
        return
    if now < quarantine.report_at:
        return
    quarantine.report_at = now + REPORT_S
    held = [record for _, record in quarantine.held.values()]
    tokens = sum(h.tokens for h in held)
    pool = getattr(getattr(queue.scheduler, "token_to_kv_pool_allocator", None), "size", 0)
    share = f" ({100.0 * tokens / pool:.1f}% of the KV pool)" if pool else ""
    logger.error(
        "KV quarantine: %d room(s) hold %d KV tokens%s without every prefill drain ack, the oldest for %.0fs, from "
        "prefill %s; their pages stay out of the pool until the acks arrive or the prefill is declared dead",
        len(held), tokens, share, now - min(h.since for h in held), ", ".join(sorted({h.prefill for h in held})),
    )


def register(registry, replace) -> None:
    import functools
    import importlib

    from sglang_neuron.sglang_patches import source_guard

    for (module, qualname), md5 in _TARGETS_MD5.items():
        obj = functools.reduce(getattr, qualname.split("."), importlib.import_module(module))
        source_guard.check(obj, md5, f"{module}.{qualname}", __name__)
    registry.register(TARGET, resolve_deferred_releases, replace)
