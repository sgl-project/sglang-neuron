# SPDX-License-Identifier: Apache-2.0
"""Fail only the queued requests of a prefill a PD decode cannot pair with, and keep serving the other prefills.

Upstream resolves a prefill's parallel info at first contact (``DecodePreallocQueue._ensure_prefill_info`` calls
``CommonKVManager.try_ensure_parallel_info``, which caches it only once ``_resolve_rank_mapping`` returns). An error
there leaves the scheduler loop, so one incompatible prefill in a router's pool takes the decode down with every
request in flight.

The wrapper runs upstream's resolution one prefill at a time. A ``PrefillIncompatibleError``, raised by a first-contact
check hooked on ``_resolve_rank_mapping``, fails that prefill's queued requests with the reason, and the decode carries
on. The info stays uncached, so the prefill is checked again after upstream's 1 s retry interval
(``_ensure_retry_interval``) and a relaunched, matching prefill is served.

Such a check must decide only from the prefill's advertised info and the decode's launch config, never from a rank's
own history (say, which prefill it reached first). Then every rank refuses the same prefills, maybe in different
iterations, and the reduced poll (``_update_handshake_waiters``) fails the request on the whole group.

SGLang's own first-contact refusals (page size, KV dtype, KV layout, DCP, ``_resolve_rank_mapping``'s CP and PP
asserts, and others) still stop the decode: they raise RuntimeError or AssertionError, which cannot tell a bad prefill
from a bad decode config or a bug.

Registered only on the chunk-keyed decode, whose pairing check raises this error
(sglang_patches.decode_radix_prefill_chunks.check_pairing, serving_policy.decode_radix_prefill_chunks). It is the
decode-side counterpart of sglang_patches/nixl_peer_refusal: the receivers failed here were never initialized, so
their abort sends no ABORT, and nixl_peer_refusal's ``record_failure`` wrap passes through on a decode.

Upstream: generic; upstream would raise the same typed error from its own refusals of one prefill.

Targets are md5-pinned against upstream SGLang 43881a7: ``DecodePreallocQueue._ensure_prefill_info`` (wrapped) and the
sources the wrapper relies on."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_DECODE = "sglang.srt.disaggregation.decode"
_COMMON = "sglang.srt.disaggregation.common.conn"
_NIXL = "sglang.srt.disaggregation.nixl.conn"
TARGET = f"{_DECODE}.DecodePreallocQueue._ensure_prefill_info"
# (module, qualified name): md5 of upstream's source
_TARGETS_MD5 = {
    # sets the prefill's attempt time before try_ensure_parallel_info, which a raise leaves set: the prefill's later
    # requests stay pending for the retry interval
    (_DECODE, "DecodePreallocQueue._ensure_prefill_info"): "9bbdde803af402b44d086ee58dbbfda7",
    # caches the prefill's info only after _resolve_rank_mapping returns
    (_COMMON, "CommonKVManager.try_ensure_parallel_info"): "15b2d05442275911ca74f44980700aac",
    # keeps only the returned remaining requests pending and initializes only the ready prefills' receivers; a failed
    # request stays in the queue
    (_DECODE, "DecodePreallocQueue._resolve_pending_reqs"): "f77910873e66c71e120a97aa20b3a452",
    # the next poll of a failed receiver aborts its request with the room's recorded reason
    (_DECODE, "DecodePreallocQueue._update_handshake_waiters"): "558e493b485e756ea2b2941a460d0dbc",
    (_COMMON, "CommonKVReceiver.failure_exception"): "f568ed6f41876219b79a0ccd57a40b96",
    # fails the receiver and records its own reason first, which record_failure after it replaces
    (_COMMON, "CommonKVReceiver.abort"): "d1aaee19a0e575b463710144cf339d6c",
    (_COMMON, "CommonKVManager.record_failure"): "c356974ace971d8311aeaf794a2548dd",
    # abort's ABORT goes only from a receiver that init set up (its bootstrap infos), so one failed here sends none
    (_COMMON, "CommonKVReceiver.ensure_abort_notified"): "e92d994aff6f8e1a224cc2fe03e612cc",
    # the NIXL receiver overrides none of the receiver's methods above
    (_NIXL, "NixlKVReceiver"): "100db2679f035d31c9722068bf7d6fab",
}


class PrefillIncompatibleError(RuntimeError):
    """A prefill this decode cannot pair with: its queued requests fail and the decode serves the others. A
    RuntimeError like SGLang's own refusals, so without the wrapper it still stops the decode."""


def ensure_prefill_info(original_fn, self, addr_to_reqs):
    """AROUND DecodePreallocQueue._ensure_prefill_info: upstream's, one prefill at a time, so one refusal loses no other
    prefill's result. A prefill that raises PrefillIncompatibleError has its queued requests failed."""
    ready, remaining = {}, []
    for bootstrap_addr, reqs in addr_to_reqs.items():
        try:
            addr_ready, addr_remaining = original_fn(self, {bootstrap_addr: reqs})
        except PrefillIncompatibleError as error:
            _fail_queued(self, bootstrap_addr, reqs, f"Prefill {bootstrap_addr} cannot serve this decode: {error}")
            continue
        ready.update(addr_ready)
        remaining.extend(addr_remaining)
    return ready, remaining


def _fail_queued(queue, bootstrap_addr, reqs, reason: str) -> None:
    from sglang.srt.disaggregation.base.conn import KVPoll

    logger.error("%s; failing its %d queued request(s)", reason, len(reqs))
    for decode_req in reqs:
        receiver = decode_req.kv_receiver
        # None after a queue cleanup (upstream's retries-exhausted path). An already failed receiver keeps its reason:
        # a duplicate bootstrap room's receiver fails without owning the room's failure record.
        if receiver is None or receiver.conclude_state == KVPoll.Failed:
            continue
        receiver.abort()
        queue.kv_manager.record_failure(decode_req.req.bootstrap_room, reason)
    # upstream's count of failed fetches starts over; the attempt time it set before the raise stays
    queue._ensure_retry_count.pop(bootstrap_addr, None)


def register(registry, around) -> None:
    from sglang_neuron.sglang_patches import source_guard

    source_guard.check_targets(_TARGETS_MD5, __name__)
    registry.register(TARGET, ensure_prefill_info, around)
