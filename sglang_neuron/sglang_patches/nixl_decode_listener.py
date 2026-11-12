# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""Keep a NIXL decode's listener thread alive when it fails to handle a message.

The decode takes every control message from its prefills on one listener thread: chunk ready, staging requests, ABORT
acks and transfer status. Upstream's thread ends at the first message whose handling raises, and nothing restarts it.
Every later ack is then lost, so each later failed room keeps its pages until its prefill is declared dead, and each
later room waits out its timeout for a status that never comes.

Here each message runs upstream's dispatch under a guard that logs the failure and moves on. A failed receive, as when
the socket closes at shutdown, still ends the thread.

Upstream: a generic fix to the NIXL decode's listener."""

from __future__ import annotations

import logging
import threading

logger = logging.getLogger(__name__)

_NIXL = "sglang.srt.disaggregation.nixl.conn"
_COMMON = "sglang.srt.disaggregation.common.conn"
TARGET = (_NIXL, "NixlKVManager._start_decode_listener_thread")
_TARGETS_MD5 = {
    TARGET: "fa4b0f0223692695150f86b6c9c3351b",
    (_COMMON, "CommonKVManager.handle_chunk_ready"): "61b556a23302b895de9c4e2a0d60d8d1",
    (_COMMON, "CommonKVManager.handle_abort_ack_message"): "5a03f707892dcb1a275d9864e4138e4d",
    (_COMMON, "CommonKVManager.parse_kv_status_message"): "cde7d580c2820e0be60b7d7ce769cba0",
    (_COMMON, "CommonKVManager.apply_prefill_status"): "bc671b64e084a2655c125d8770ee410f",
}


def handle_decode_message(mgr, msg) -> None:
    if mgr.handle_chunk_ready(msg):
        return
    if msg[0] == b"STAGING_REQ":
        if mgr.enable_staging:
            mgr._handle_staging_req(msg)
        return
    if mgr.handle_abort_ack_message(msg):
        return
    parsed = mgr.parse_kv_status_message(msg)
    if parsed is not None:
        room, status, prefill_rank, reason = parsed
        mgr.apply_prefill_status(bootstrap_room=room, status=status, prefill_rank=prefill_rank, failure_reason=reason)
        return
    logger.warning("decode_listener_thread: unexpected message tag %s", msg[0][:20])


def start_decode_listener_thread(self) -> None:
    def decode_listener_thread():
        while True:
            msg = self.server_socket.recv_multipart()
            try:
                handle_decode_message(self, msg)
            except Exception:  # noqa: BLE001 - a message that fails must not end the thread every later ack needs
                logger.exception("decode_listener_thread: a %r message failed; the listener goes on",
                                 msg[0][:20] if msg else b"")

    threading.Thread(target=decode_listener_thread, daemon=True).start()


def register(registry, replace) -> None:
    from sglang_neuron.sglang_patches import source_guard

    source_guard.check_targets(_TARGETS_MD5, __name__)
    registry.register(".".join(TARGET), start_decode_listener_thread, replace)
