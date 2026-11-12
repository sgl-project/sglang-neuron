# SPDX-License-Identifier: Apache-2.0
"""Have a pipelined PD prefill stage send a chunk's KV only after its own forward of that chunk has run, without the
scheduler thread waiting for it.

A non-last stage's forward returns at launch, so without the overlap schedule upstream's PD prefill can send a middle
chunk's KV from ``process_prefill_chunk`` at the next scheduling step, before the forward has written it, and NIXL
would read unwritten pages.

Upstream already has the hook: ``CommonKVSender.send`` moves the sender's ``_early_send_wait_event`` into the chunk's
``TransferKVChunk.wait_event``, which the NIXL transfer worker synchronizes before reading the chunk's KV. Here a
middle-chunk send sets that event to the newest launched forward (``backend.device_io.last_launch_event``, at that
point the chunk's own), so the wait happens on the transfer worker. A chunk sent in several segments carries the event
on its first segment only; a room's sends go to one transfer queue in order (``add_transfer_request`` shards by the
room's destination ports), so the later segments follow the first one's wait. A last chunk is sent when its result is
processed, after this stage's forward, so it gets no event (at most a finished one a skipped middle send left behind).

Upstream: generic for a backend whose forward returns at launch.

Targets, md5-pinned against upstream SGLang 43881a7: ``SchedulerDisaggregationPrefillMixin.send_kv_chunk`` (wrapped),
and the functions of the event's path (relied on, ``_TARGETS_MD5``)."""

from __future__ import annotations

_PREFILL = "sglang.srt.disaggregation.prefill"
_NIXL = "sglang.srt.disaggregation.nixl.conn"
_COMMON = "sglang.srt.disaggregation.common.conn"
SEND_KV_CHUNK = f"{_PREFILL}.SchedulerDisaggregationPrefillMixin.send_kv_chunk"
# (module, qualified name): md5 of the source; the wrapped one, then those the event's path relies on
_TARGETS_MD5 = {
    (_PREFILL, "SchedulerDisaggregationPrefillMixin.send_kv_chunk"): "0e5a574d819e4c08526f8751615bc139",
    # the middle-chunk send at the next scheduling step
    (_PREFILL, "SchedulerDisaggregationPrefillMixin.process_prefill_chunk"): "dca76f30fd359b20c4c3b2e861490e34",
    # skips a chunk that sends nothing, and splits the rest into segments, one CommonKVSender.send each
    (_PREFILL, "SchedulerDisaggregationPrefillMixin._send_kv_chunk"): "675d207208ef4fb85f58682044351166",
    # the field's one other writer, SGLang's early send, off on a Neuron prefill
    (_PREFILL, "SchedulerDisaggregationPrefillMixin.maybe_send_cached_prefix_chunk"):
        "4acc8992430a40d9359a3085d975d0fa",
    # takes _early_send_wait_event into the chunk and clears it, unless its indices send nothing
    (_COMMON, "CommonKVSender.send"): "180f9eab56ce26ca011f900d163a903d",
    (_COMMON, "CommonKVSender._prepare_send_indices"): "1ae8446b9ece6e6eec031197cd9a59b5",
    ("sglang.srt.disaggregation.common.utils", "TransferKVChunk"): "7332318fb16caa338519db926148cd01",
    # keeps the chunk's wait event on the room's one transfer queue, whose worker synchronizes it before reading the
    # chunk's KV
    (_NIXL, "NixlKVManager.add_transfer_request"): "7bd7b5f329968db10e3dd57ccebce041",
    (_NIXL, "NixlKVManager.transfer_worker"): "aaccdcf4cb029cd5aa2f01d265e85c25",
    ("sglang.srt.managers.scheduler_pp_mixin", "SchedulerPPMixin.event_loop_pp_disagg_prefill"):
        "8bbb0377fb4214f8ff289436470f2e7e",
}


def send_kv_chunk(original_fn, self, req, last_chunk: bool = False, end_idx=None):
    """AROUND SchedulerDisaggregationPrefillMixin.send_kv_chunk: a pipelined prefill's middle chunk waits on the newest
    launched forward.

    Overwriting the sender's event is fine: SGLang skips ``CommonKVSender.send``, its only reader, for a middle chunk
    that sends nothing, leaving an earlier chunk's event, and the newest launch covers every earlier one. SGLang's early
    send, the only other writer, is off on Neuron (``SGLANG_DISAGG_PREFILL_EARLY_SEND_CACHED_PREFIX=0``).

    A last chunk that the last stage already sent (sglang_patches/pp_prefill_last_send) is not sent again here."""
    from sglang_neuron.sglang_patches import pp_prefill_last_send

    if pp_prefill_last_send.skip_processing_send(req, last_chunk):
        # the last stage sent this last chunk when its outputs reached the host (pp_prefill_last_send)
        return None
    # a sender without the field (the fake sender of a warm-up request) transfers nothing
    if not last_chunk and hasattr(req.disagg_kv_sender, "_early_send_wait_event"):
        from sglang_neuron.backend import device_io

        req.disagg_kv_sender._early_send_wait_event = device_io.last_launch_event()
    return original_fn(self, req, last_chunk=last_chunk, end_idx=end_idx)


def register(registry, around) -> None:
    from sglang_neuron.sglang_patches import source_guard

    source_guard.check_targets(_TARGETS_MD5, __name__)
    registry.register(SEND_KV_CHUNK, send_kv_chunk, around)
