# SPDX-License-Identifier: Apache-2.0
"""A disaggregated KV manager changes a room's status in one step.

Upstream's ``CommonKVManager.update_status`` reads a room's status, compares it and writes the result without a lock,
and several threads call it. On a NIXL prefill the bootstrap thread fails a room on the decode's ABORT while a transfer
worker marks the same room Transferring. A worker that reads before the Failed and writes after it overwrites Failed.
The decode frees the pages once the prefill acks the ABORT, and the room's next chunk then writes into pages another
request now owns. A clear that runs between an update's read and write re-creates the cleared room, so a later request
on the same bootstrap room starts from a stale status.

Each manager gets one status lock. ``update_status`` runs under it, and so does each clear's drop of the room's status:
the clears are upstream's source with that one statement under the lock. Every other status write goes through
``update_status``. This is the innermost lock of the PD patches: nixl_peer_refusal's lock is always taken before it.

Upstream: a generic race fix in ``CommonKVManager``."""

from __future__ import annotations

import threading

_COMMON = "sglang.srt.disaggregation.common.conn"
_NIXL = "sglang.srt.disaggregation.nixl.conn"
UPDATE_STATUS = (_COMMON, "CommonKVManager.update_status")
SENDER_CLEAR = (_COMMON, "CommonKVSender.clear")
RECEIVER_CLEAR = (_COMMON, "CommonKVReceiver._clear")
_TARGETS_MD5 = {
    UPDATE_STATUS: "3957aa78394858a23f22fb85ef4243b6",
    SENDER_CLEAR: "2f2d68d7514e9b7a3d60bbaf0c035050",
    RECEIVER_CLEAR: "8d04db4cf98d46c6677fd778efb08f4c",
    # the NIXL classes override none of the methods above, which would otherwise run unlocked
    (_NIXL, "NixlKVReceiver"): "100db2679f035d31c9722068bf7d6fab",
    (_NIXL, "NixlKVSender"): "414c18889989c142c15cec6ed0417c16",
    (_NIXL, "NixlKVManager.update_status"): "3957aa78394858a23f22fb85ef4243b6",
    # these read a room's status and write it only through update_status
    (_NIXL, "NixlKVManager.transfer_worker"): "aaccdcf4cb029cd5aa2f01d265e85c25",
    (_NIXL, "NixlKVManager.check_status"): "2dfe12af40bd91d79bed451ce78f25fd",
    (_NIXL, "NixlKVManager._handle_abort_notification"): "e6b31f79d46246334b8b43f14b8bd88b",
    (_COMMON, "CommonKVManager.apply_prefill_status"): "bc671b64e084a2655c125d8770ee410f",
    (_COMMON, "CommonKVManager._handle_node_failure"): "6e9833cfe13f0603968dbeb2ff1bc091",
    (_COMMON, "CommonKVSender.abort"): "91c63738a82dd86a26a9b40102f3a918",
    (_COMMON, "CommonKVReceiver.abort"): "d1aaee19a0e575b463710144cf339d6c",
    (_COMMON, "CommonKVReceiver.clear"): "e2d8ccf453d085c88e12f5cd57c8ed56",
}

UPSTREAM_DROP = '''    self.kv_mgr.request_status.pop(self.bootstrap_room, None)
'''
NEURON_DROP = '''    from sglang_neuron.sglang_patches.pd_status_lock import status_lock

    with status_lock(self.kv_mgr):
        self.kv_mgr.request_status.pop(self.bootstrap_room, None)
'''
CLEAR_SUBSTITUTIONS = ((UPSTREAM_DROP, NEURON_DROP),)

_LOCK = "_neuron_status_lock"
_CREATE = threading.Lock()


def status_lock(mgr) -> threading.RLock:
    lock = mgr.__dict__.get(_LOCK)
    if lock is None:
        with _CREATE:
            lock = mgr.__dict__.setdefault(_LOCK, threading.RLock())
    return lock


def update_status(original_fn, self, *args, **kwargs):
    with status_lock(self):
        return original_fn(self, *args, **kwargs)


def locked_clear(target):
    from sglang_neuron.sglang_patches import source_guard

    return source_guard.with_substitutions(source_guard.resolve(*target), CLEAR_SUBSTITUTIONS)


def register(registry, around, replace) -> None:
    from sglang_neuron.sglang_patches import source_guard

    source_guard.check_targets(_TARGETS_MD5, __name__)
    registry.register(".".join(UPDATE_STATUS), update_status, around)
    registry.register(".".join(SENDER_CLEAR), locked_clear(SENDER_CLEAR), replace)
    registry.register(".".join(RECEIVER_CLEAR), locked_clear(RECEIVER_CLEAR), replace)
