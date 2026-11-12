# SPDX-License-Identifier: Apache-2.0
"""Fail a PD decode room whose metadata row (the first token) does not land after every prefill source reported its
transfer done, naming the cause, in place of holding the request on every rank of its replica for ever.

Upstream's decode polls a room Success once every prefill source's KV_STATUS arrived, and its metadata gate
(``disaggregation.utils._apply_metadata_gate``) holds the room as Transferring while the row's bootstrap room is still
0. A source sends its KV_STATUS only after its aux write, so on upstream's path the row lands first. A prefill whose
stages other than the last send their last chunk with no metadata row (sglang_patches/pp_prefill_last_send) leaves the
row to the last stage's sources. If none of them writes it, upstream's gate holds the room with no timeout, and the
MIN reduce over the attention-TP group holds it on every rank of the replica.

Here the gate is upstream's, and a room it holds for ``METADATA_WAIT_S`` after every status arrived is failed on this
rank (``record_failure`` + Failed): the reduce fails it on every rank, and the decode's failure path aborts the request
with the reason in its error.

Upstream: generic; upstream would put the bound in ``_apply_metadata_gate``."""

from __future__ import annotations

import logging
import time

logger = logging.getLogger(__name__)

_UTILS = "sglang.srt.disaggregation.utils"
GATE = f"{_UTILS}._apply_metadata_gate"
# (module, qualified name): md5 of the source
_TARGETS_MD5 = {(_UTILS, "_apply_metadata_gate"): "9320839bb34d08ea57103f6b99a60b00"}
METADATA_WAIT_S = 30.0
"""How long a decode holds a room whose every prefill status arrived but whose metadata row did not (as
pd_stripe_transfer's PD_WRITE_TIMEOUT_S bounds a WRITE)."""
# the decode requests held by the gate on this rank: id(decode request) -> (the request, when the hold began)
_HELD: dict = {}


def reason(room, waited_s: float) -> str:
    return (f"PD decode: room {room}: every prefill source reported its KV transfer done, but the metadata row (the "
            f"first token, from the last prefill stage) did not land in {waited_s:.0f} s. The request fails. Check "
            f"that the prefill's last stage sends its last chunks with the metadata row "
            f"(sglang_patches/pp_prefill_last_send).")


def apply_metadata_gate(original_fn, polls, decode_reqs, metadata_buffers, now=None):
    """AROUND _apply_metadata_gate: upstream's gate, then a room it has held for METADATA_WAIT_S fails."""
    from sglang.srt.disaggregation.base.conn import KVPoll

    before = list(polls)
    result = original_fn(polls, decode_reqs, metadata_buffers)
    now = time.monotonic() if now is None else now
    held = set()
    for i, (was, poll) in enumerate(zip(before, polls)):
        if was != int(KVPoll.Success) or poll != int(KVPoll.Transferring):
            continue
        decode_req = decode_reqs[i]
        key = id(decode_req)
        held.add(key)
        _, since = _HELD.setdefault(key, (decode_req, now))
        if now - since < METADATA_WAIT_S:
            continue
        room = decode_req.req.bootstrap_room
        message = reason(room, now - since)
        logger.error(message)
        kv_mgr = decode_req.kv_receiver.kv_mgr
        kv_mgr.record_failure(room, message)
        kv_mgr.update_status(room, KVPoll.Failed)
        polls[i] = int(KVPoll.Failed)
        held.discard(key)
    # forget the requests the gate no longer holds (landed, failed or gone)
    for key in [key for key in _HELD if key not in held]:
        del _HELD[key]
    return result


def register(registry, around) -> None:
    """On a PD decode."""
    from sglang_neuron.sglang_patches import source_guard

    source_guard.check_targets(_TARGETS_MD5, __name__)
    registry.register(GATE, apply_metadata_gate, around)
