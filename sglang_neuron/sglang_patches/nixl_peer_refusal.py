# SPDX-License-Identifier: Apache-2.0
"""Keep a NIXL prefill's bootstrap thread alive when it refuses a decode peer, and fail each of that peer's rooms with
the reason on both servers.

The bootstrap thread receives the decode ranks' registrations, room messages and ABORTs for one prefill rank. Upstream
registers each decode rank through ``_add_remote_peer`` without a guard, so one raise there ends the thread. The raise
can come from upstream's own checks, from NIXL's ``add_remote_agent``, or from nixl_head_major's head-major refusal.
From then on every request from every decode to that prefill rank waits out its bootstrap timeout.

Here the thread is upstream's with three changes:

- A registration that raises refuses the peer. Its table entry, NIXL agent and prepared transfers are dropped, the
  reason is kept, and the thread goes on. A decode rank registers once per prefill: upstream registers again only after
  it drops its bootstrap cache, and a refused room does not drop it. So a refused rank stays refused until the decode or
  the prefill restarts. A rank that does register again and passes is served.
- A room message from a refused peer fails the room with that reason, and upstream's status message tells the decode
  rank at once, so the decode fails the request instead of waiting. Nothing was written to the decode's pages.
- Any other message that raises is logged with its traceback, and the thread goes on (``run_bootstrap_thread``). It
  has to: every later room and every ABORT ack of this prefill rank goes through it, and pd_decode_kv_quarantine holds
  a failed room's pages until its acks arrive. The thread ends only once its socket is closed.

A room can only fail once its sender exists. A refusal that arrives first is kept, and applied when the sender is
created. One lock per manager orders this against the sender's clear, so a room fails exactly once and never after it
was cleared. When the decode ABORTs a refused room, the room keeps the refusal's reason rather than upstream's generic
one. A refused room's entry lives until every decode rank's message for it has arrived, or for ``REFUSED_ROOM_TTL_S``.

Upstream: a generic fix: the bootstrap thread should never end on one peer's error."""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from typing import Dict, Optional, Set

logger = logging.getLogger(__name__)

_NIXL = "sglang.srt.disaggregation.nixl.conn"
_COMMON = "sglang.srt.disaggregation.common.conn"
_PREFILL = "sglang.srt.disaggregation.prefill"
REPLACED = (_NIXL, "NixlKVManager._start_bootstrap_thread")
_WRAPPED = {
    (_COMMON, "CommonKVSender.__init__"): "7dadb846afe3f233cbec620cf1d8eee9",
    (_COMMON, "CommonKVSender.clear"): "2f2d68d7514e9b7a3d60bbaf0c035050",
    (_NIXL, "NixlKVManager._handle_abort_notification"): "e6b31f79d46246334b8b43f14b8bd88b",
    (_COMMON, "CommonKVManager.record_failure"): "c356974ace971d8311aeaf794a2548dd",
}
# the upstream sources the module relies on
_TARGETS_MD5 = {
    REPLACED: "0ad34e79d2d77da1b2cb45c799e94748",
    **_WRAPPED,
    (_NIXL, "NixlKVSender"): "414c18889989c142c15cec6ed0417c16",
    (_COMMON, "AbortNotification.from_zmq"): "63447705adae1a30796f92b28e7e085c",
    (_NIXL, "NixlKVManager._add_remote_peer"): "e2ee9f36cd7849d884c49275e4c10bde",
    (_NIXL, "TransferInfo.from_zmq"): "7ba9ad4939368443fdd8574389bd4f7e",
    (_COMMON, "CommonKVManager.update_status"): "3957aa78394858a23f22fb85ef4243b6",
    (_COMMON, "CommonKVSender._poll_with_status"): "a0146ee1fd19a666fb61f4333523f996",
    (_COMMON, "CommonKVSender.failure_exception"): "3aa7eb415067d4cbb8a4c03a16f86a13",
    (_PREFILL, "SchedulerDisaggregationPrefillMixin.handle_bootstrap_failure"): "b3750a47fe171124d41dc4fd718316b5",
    (_COMMON, "CommonKVManager._room_notify_targets"): "4458842a2e19c357659231dc25d79d93",
    (_COMMON, "CommonKVManager.send_kv_status_message"): "8aeaa35a093c57c15f7c5173261340e8",
    (_COMMON, "CommonKVManager._encode_kv_status_message"): "03ef61a107cd8201bdb58bdd68f45374",
    (_COMMON, "CommonKVManager.parse_kv_status_message"): "cde7d580c2820e0be60b7d7ce769cba0",
    (_NIXL, "NixlKVManager._start_decode_listener_thread"): "fa4b0f0223692695150f86b6c9c3351b",
    (_COMMON, "CommonKVManager.apply_prefill_status"): "bc671b64e084a2655c125d8770ee410f",
}

UPSTREAM_LOOP = '''        """This thread recvs transfer info from the decode engine"""
        while True:
'''
NEURON_LOOP = '''        """This thread recvs transfer info from the decode engine"""
        from sglang_neuron.sglang_patches import nixl_peer_refusal

        while True:
'''
UPSTREAM_REGISTRATION = '''            if room == "None":
                # Register new peer and save KV base pointers.
                self._add_remote_peer(
                    KVArgsRegisterInfo.from_zmq(waiting_req_bytes)
                )
                logger.debug(f"Register KVArgs from {agent_name} successfully")
                continue
            room = int(room)
'''
NEURON_REGISTRATION = '''            if room == "None":
                # Register new peer and save KV base pointers, or refuse it.
                nixl_peer_refusal.register_or_refuse(self, waiting_req_bytes, agent_name)
                continue
            room = int(room)
            if nixl_peer_refusal.refused_room_message(self, room, agent_name, waiting_req_bytes):
                continue
'''
UPSTREAM_GUARDED = '''    def bootstrap_thread_guarded():
        try:
            bootstrap_thread()
        except Exception:
            logger.exception(
                "prefill bootstrap_thread died on engine_rank=%s; requests to "
                "this rank will time out in KVPoll.Bootstrapping",
                self.kv_args.engine_rank,
            )
            raise
'''
NEURON_GUARDED = '''    def bootstrap_thread_guarded():
        from sglang_neuron.sglang_patches import nixl_peer_refusal

        nixl_peer_refusal.run_bootstrap_thread(self, bootstrap_thread)
'''
BOOTSTRAP_THREAD_SUBSTITUTIONS = ((UPSTREAM_LOOP, NEURON_LOOP), (UPSTREAM_REGISTRATION, NEURON_REGISTRATION),
                                  (UPSTREAM_GUARDED, NEURON_GUARDED))

# longer than SGLang's default bootstrap and waiting timeouts
REFUSED_ROOM_TTL_S = 600.0
_STATE = "neuron_peer_refusals"
# (manager, room) while this thread runs upstream's ABORT handler on a room that already has its failure reason
_aborting = threading.local()


class _RefusedRoom:
    """A room that a refused peer sent for. ``agents`` are the decode ranks that have sent for it so far."""

    __slots__ = ("reason", "since", "expected", "agents", "failed")

    def __init__(self, reason: str, since: float, expected: int, agents: Set[str]):
        self.reason, self.since, self.expected, self.agents, self.failed = reason, since, expected, agents, False


class PeerRefusals:
    """One prefill manager's refused peers (agent name -> reason) and refused rooms (oldest first)."""

    def __init__(self):
        self.peers: Dict[str, str] = {}
        self.rooms: "OrderedDict[int, _RefusedRoom]" = OrderedDict()
        self.lock = threading.Lock()

    def expire(self, now: float) -> None:
        while self.rooms:
            room, entry = next(iter(self.rooms.items()))
            if now - entry.since <= REFUSED_ROOM_TTL_S:
                return  # entries are never re-inserted, so the rest are younger
            del self.rooms[room]

    def settle(self, mgr, room: int, entry: _RefusedRoom) -> None:
        # the caller holds self.lock
        if not entry.failed and room in mgr.request_status:
            entry.failed = True
            _fail_room(mgr, room, entry.reason)
        if entry.failed and len(entry.agents) >= entry.expected:
            del self.rooms[room]


def refusals_of(mgr) -> Optional[PeerRefusals]:
    return mgr.__dict__.get(_STATE)


def _fail_room(mgr, room: int, reason: str) -> None:
    from sglang.srt.disaggregation.base.conn import KVPoll

    status = mgr.request_status.get(room)
    if status is None or status in (KVPoll.Failed, KVPoll.Success):
        return  # cleared, or concluded already
    mgr.record_failure(room, reason)
    mgr.update_status(room, KVPoll.Failed)
    logger.debug("room %s failed: %s", room, reason)


def register_or_refuse(mgr, frames, agent_name: str) -> None:
    from sglang.srt.disaggregation.nixl.conn import KVArgsRegisterInfo

    # a peer already registered keeps its registration when a repeat registration is refused
    added = agent_name not in mgr.decode_kv_args_table
    try:
        mgr._add_remote_peer(KVArgsRegisterInfo.from_zmq(frames))
    except Exception as error:  # noqa: BLE001 - any refusal of the peer, reported on each of its rooms
        if added and mgr.decode_kv_args_table.pop(agent_name, None) is not None:
            _drop_remote_agent(mgr, agent_name)
        refusals = refusals_of(mgr) or mgr.__dict__.setdefault(_STATE, PeerRefusals())
        refusals.peers[agent_name] = f"{type(error).__name__}: {error}"
        logger.error("prefill rank %s refused decode peer %s at registration, and fails each of its rooms: %s",
                     mgr.kv_args.engine_rank, agent_name, refusals.peers[agent_name])
        return
    refusals = refusals_of(mgr)
    if refusals is not None:
        refusals.peers.pop(agent_name, None)
    logger.debug("Register KVArgs from %s successfully", agent_name)


def _drop_remote_agent(mgr, agent_name: str) -> None:
    """Undo what upstream's _add_remote_peer set up before it raised, so a later registration starts afresh."""
    for prepared in (mgr.prep_handles, mgr.prep_handles_slice_dst, mgr._prepared_state_plans):
        # the peer's handles: its own name, and the per-peer keys any patch stored under "<name>:<what>"
        for key in [key for key in prepared if key == agent_name or key.startswith(agent_name + ":")]:
            prepared.pop(key, None)
    try:
        mgr.agent.remove_remote_agent(agent_name)
    except Exception as error:  # noqa: BLE001 - NIXL never added it (add_remote_agent refused it), or cannot drop it
        logger.warning("prefill rank %s could not remove refused decode peer %s's NIXL agent: %s: %s",
                       mgr.kv_args.engine_rank, agent_name, type(error).__name__, error)


def refused_room_message(mgr, room: int, agent_name: str, frames) -> bool:
    """True for a message about a room a refused peer sent for: the room fails and its decode ranks are told. False for
    any other message, which upstream's loop goes on to handle."""
    refusals = refusals_of(mgr)
    if refusals is None or (agent_name not in refusals.peers and room not in refusals.rooms):
        return False
    from sglang.srt.disaggregation.base.conn import KVPoll
    from sglang.srt.disaggregation.nixl.conn import TransferInfo

    info = TransferInfo.from_zmq(frames)
    targets = [] if info.is_dummy else [(info.endpoint, info.dst_port)]
    now = time.monotonic()
    with refusals.lock:
        refusals.expire(now)
        entry = refusals.rooms.get(room)
        if entry is None:
            reason = refusals.peers.get(agent_name)
            if reason is None:
                return False
            # The entry goes in before settle looks the room up, and sender_init looks for the entry only after its
            # sender has set the room up, so one of the two always sees the other. Served peers that already sent for
            # this room are told too.
            entry = refusals.rooms[room] = _RefusedRoom(
                f"decode peer {agent_name} was refused at registration: {reason}", now,
                max(1, info.required_dst_info_num), set(mgr.transfer_infos.get(room, ())))
            targets += mgr._room_notify_targets(room)
        entry.agents.add(agent_name)
        refusals.settle(mgr, room, entry)
        reason = entry.reason
    mgr.send_kv_status_message(targets=list(dict.fromkeys(targets)), bootstrap_room=room, status=KVPoll.Failed,
                               failure_reason=reason)
    return True


def _aborted_room(msg) -> Optional[int]:
    """The room an ABORT message names. None for any other message, and for a malformed ABORT, which upstream's
    handler logs and drops."""
    from sglang.srt.disaggregation.common.conn import ABORT_TAG

    if len(msg) < 2 or msg[0] != ABORT_TAG:
        return None
    try:
        return int(msg[1].decode("ascii"))
    except (UnicodeDecodeError, ValueError):
        return None


def abort_keeps_first_reason(original_fn, self, msg, *args, **kwargs):
    """When the decode ABORTs a room that already failed, or a refused room, the room keeps that first reason instead
    of upstream's generic one."""
    room = _aborted_room(msg)
    if room is None:
        return original_fn(self, msg, *args, **kwargs)
    from sglang.srt.disaggregation.base.conn import KVPoll

    refusals = refusals_of(self)
    refused = refusals is not None and room in refusals.rooms
    if not refused and self.request_status.get(room) != KVPoll.Failed:
        return original_fn(self, msg, *args, **kwargs)
    _aborting.room = (self, room)
    try:
        return original_fn(self, msg, *args, **kwargs)
    finally:
        _aborting.room = None


def record_failure_keeps_first(original_fn, self, bootstrap_room, failure_reason, *args, **kwargs):
    aborting = getattr(_aborting, "room", None)
    if aborting is None or aborting[0] is not self or aborting[1] != bootstrap_room:
        return original_fn(self, bootstrap_room, failure_reason, *args, **kwargs)
    refusals = refusals_of(self)
    entry = refusals.rooms.get(bootstrap_room) if refusals is not None else None
    with self.failure_lock:
        self.failure_records.setdefault(bootstrap_room, entry.reason if entry is not None else failure_reason)


def sender_init(original_fn, self, *args, **kwargs):
    # a room that a refused peer sent for before this sender existed fails now
    original_fn(self, *args, **kwargs)
    refusals = refusals_of(self.kv_mgr)
    if refusals is not None and self.bootstrap_room in refusals.rooms:
        with refusals.lock:
            entry = refusals.rooms.get(self.bootstrap_room)
            if entry is not None:
                refusals.settle(self.kv_mgr, self.bootstrap_room, entry)


def sender_clear(original_fn, self, *args, **kwargs):
    # under the refusal lock, so the bootstrap thread never fails a room after its sender cleared it
    refusals = refusals_of(self.kv_mgr)
    if refusals is None:
        return original_fn(self, *args, **kwargs)
    with refusals.lock:
        return original_fn(self, *args, **kwargs)


def run_bootstrap_thread(mgr, bootstrap_thread) -> None:
    """Run upstream's bootstrap loop, and enter it again after a message raises. That is safe because the loop keeps
    nothing between messages. The thread ends once its socket is closed or its ZMQ context terminated: recv_multipart
    then raises ZMQError."""
    import zmq

    while True:
        try:
            bootstrap_thread()
            return
        except zmq.ZMQError:
            logger.info("prefill rank %s bootstrap thread ends: its socket is closed", mgr.kv_args.engine_rank)
            return
        except Exception:  # noqa: BLE001 - one message must not end the thread every later room and ABORT ack needs
            logger.exception("prefill rank %s bootstrap thread: a message failed; the thread goes on",
                             mgr.kv_args.engine_rank)


def bootstrap_thread_starter():
    from sglang_neuron.sglang_patches import source_guard

    return source_guard.with_substitutions(source_guard.resolve(*REPLACED), BOOTSTRAP_THREAD_SUBSTITUTIONS)


def check_status_message() -> None:
    """A refusal reaches the decode as NIXL's KV_STATUS message, which must carry the failure reason."""
    from sglang.srt.disaggregation.nixl.conn import NixlKVManager

    tag, carries_reason = NixlKVManager.kv_status_msg_tag, NixlKVManager.kv_status_msg_carries_reason
    if tag != b"KV_STATUS" or carries_reason is not True:
        raise RuntimeError(f"NixlKVManager's status message (kv_status_msg_tag {tag!r}, kv_status_msg_carries_reason "
                           f"{carries_reason!r}) is not the KV_STATUS message with its reason that {__name__} sends "
                           f"a refused room's decode rank; re-validate {__name__} against this SGLang before serving "
                           f"it")


def register(registry, around, replace) -> None:
    from sglang_neuron.sglang_patches import source_guard

    source_guard.check_targets(_TARGETS_MD5, __name__)
    check_status_message()
    registry.register(".".join(REPLACED), bootstrap_thread_starter(), replace)
    registry.register(f"{_COMMON}.CommonKVSender.__init__", sender_init, around)
    registry.register(f"{_COMMON}.CommonKVSender.clear", sender_clear, around)
    registry.register(f"{_NIXL}.NixlKVManager._handle_abort_notification", abort_keeps_first_reason, around)
    registry.register(f"{_COMMON}.CommonKVManager.record_failure", record_failure_keeps_first, around)
