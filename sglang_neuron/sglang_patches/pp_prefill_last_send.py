# SPDX-License-Identifier: Apache-2.0
"""Send each request's last prefill chunk from every stage of a pipelined PD prefill at that stage's own forward, and
the first token from the last stage only (``SGLANG_NEURON_PP_PREFILL_EARLY_LAST_SEND=1``, off by default).

Without the knob, every stage sends a request's last chunk when the overlap-order loop (sglang_patches/
pp_prefill_overlap_loop) processes it. That send carries the chunk's KV, the request's state payload and the metadata
row (the first token, which only the last stage samples). A stage other than the last gets the first token only when
the outputs reach it round the ring: the last rank sends them to rank 0, and each stage hands them on at its
processing, one stage an iteration. Under load, stage s sends about s + 1 stage times after the last stage's forward
ends (measured: each stage adds to the delay, the last stage the most), and the decode waits for the latest. The send also
queues behind the next request's middle chunks on the room's transfer worker, each of which waits for its own forward.

With the knob:

* A stage other than the last sends a request's last chunk right after it launches the chunk's forward
  (``send_after_launch``, from pp_last_stage_async's launch hook): SGLang's ``send_kv_chunk(req, last_chunk=True)``
  with the forward's launch event as the chunk's wait event (the transfer worker waits for the forward before it
  reads the KV and the state), with no metadata row: the metadata buffer is not filled (``set_buf`` is skipped) and the
  sender's aux index is ``NO_AUX`` for the call, so the transfer worker writes no aux (``WORKER_SUBSTITUTIONS``, in
  nixl_settle_then_ack's rebuild of the worker). The stage still concludes the room with its KV_STATUS.
* The last stage sends a request's last chunk right before it sends that batch's outputs to rank 0 (``send_early``,
  from pp_prefill_overlap_loop.send_outputs, after it filled the outputs from their host copy): its forward has
  ended. It sends the KV, the state and the metadata row with the first token (appended to ``req.output_ids`` for
  the call only). Rank 0 then answers the request once it processed it (sglang_patches/pp_prefill_early_answer).
* Processing does not send a marked request's last chunk again (``skip_processing_send``, from
  pp_prefill_kv_after_forward.send_kv_chunk). Everything else of the processing (the in-flight queue, the cache
  checkpoint, the times, the release consensus) stays at its point on every stage, so the stages still schedule the
  same batches.

A decode rank's metadata row is written by its source ranks on every stage without the knob; with it, by its sources
on the last stage only (sglang_patches/dcp_page_transfer: every decode rank has a source on every stage). The decode
counts every source's KV_STATUS and gates the room on the row's bootstrap room (``_apply_metadata_gate``), so it admits
the request only when every stage's stripes and the last stage's row have landed; sglang_patches/pd_metadata_wait fails
a room whose row does not land after every status, naming the cause.

A request is sent at its forward only on the path where processing would send it at once: no pending bootstrap, not
aborted, no discarded output, no logprobs, no sampling mask, no grammar, no EAGLE draft data, no KV checksum, not the
``prefill_complete`` decode allocation policy, and not SGLang's forced-retry test hook. Every other request sends at
its processing, as without the knob.

Upstream: generic for a pipelined PD prefill whose stages each schedule; upstream would take it with the overlap-order
loop and a last-chunk flag for "no metadata row" in place of the NO_AUX index."""

from __future__ import annotations

import functools
import os

ENV = "SGLANG_NEURON_PP_PREFILL_EARLY_LAST_SEND"
# set on a request once its last chunk went out at its forward; pp_prefill_kv_after_forward skips the processing send
MARK = "sgn_early_last_send"
# set on a request for the duration of a stage's send with no metadata row: set_buf does not fill the buffer
NO_METADATA = "sgn_no_metadata_send"
# the aux index of a last chunk that writes no metadata row (a real index is a row of the metadata buffers, >= 0)
NO_AUX = -1
SET_BUF = "sglang.srt.disaggregation.utils.MetadataBuffers.set_buf"
# (module, qualified name): md5 of the source (the same pin as pd_stripe_transfer's)
_TARGETS_MD5 = {("sglang.srt.disaggregation.utils", "MetadataBuffers.set_buf"): "83e4b5cd93c678b1cde88c7b58a17e95"}

# the aux write of upstream's NixlKVManager.transfer_worker (dedented once, as source_guard rebuilds it), and the same
# write skipped for a NO_AUX chunk; nixl_settle_then_ack adds these to its rebuild of the worker
UPSTREAM_AUX_WRITE = """                    if kv_chunk.prefill_aux_index is None:
                        raise RuntimeError("Missing aux index for last chunk")
                    aux_xfer_handle = self.send_aux(
                        req.agent_name,
                        kv_chunk.prefill_aux_index,
                        dst_info.dst_aux_ptrs,
                        dst_info.dst_aux_item_lens,
                        req.dst_aux_index,
                    )
                    handles.append(aux_xfer_handle)
"""
NEURON_AUX_WRITE = """                    if kv_chunk.prefill_aux_index is None:
                        raise RuntimeError("Missing aux index for last chunk")
                    # a stage that sent its last chunk at its forward writes no metadata row
                    # (sglang_patches/pp_prefill_last_send): the last stage writes it with the first token
                    if kv_chunk.prefill_aux_index != -1:
                        aux_xfer_handle = self.send_aux(
                            req.agent_name,
                            kv_chunk.prefill_aux_index,
                            dst_info.dst_aux_ptrs,
                            dst_info.dst_aux_item_lens,
                            req.dst_aux_index,
                        )
                        handles.append(aux_xfer_handle)
"""
WORKER_SUBSTITUTIONS = ((UPSTREAM_AUX_WRITE, NEURON_AUX_WRITE),)


@functools.lru_cache(maxsize=1)
def enabled() -> bool:
    """``SGLANG_NEURON_PP_PREFILL_EARLY_LAST_SEND`` (read once): True for 1, False when unset or 0; another value is
    refused naming it.

    Example:
        >>> enabled()  # doctest: +SKIP
        False
    """
    raw = os.environ.get(ENV, "").strip()
    if raw in ("", "0"):
        return False
    if raw == "1":
        return True
    raise ValueError(f"{ENV}={raw!r}: set 1 (every stage sends a last chunk at its forward, the last stage with the "
                     f"first token) or 0.")


def _final_chunk(batch, index: int, req) -> bool:
    """Whether ``batch`` runs the last chunk of its request at ``index``: the chunk ends at the end of the prompt (a
    cached prefix counts in prefix_lens)."""
    return int(batch.prefix_lens[index]) + int(batch.extend_lens[index]) >= len(req.origin_input_ids)


def eligible(scheduler, req) -> bool:
    """Whether processing would send ``req``'s last chunk at once, with nothing else before the send that an early
    send cannot do."""
    from sglang.srt.disaggregation.utils import is_aborted
    from sglang.srt.environ import envs
    from sglang.srt.runtime_context import get_disagg

    if getattr(req, MARK, False) or req.pending_bootstrap or is_aborted(req):
        return False
    if req.discard_output_reason is not None or req.return_logprob or req.return_sampling_mask:
        return False
    if req.grammar is not None or scheduler.spec_algorithm.is_eagle():
        return False
    if getattr(scheduler, "kv_checksum_computer", None) is not None:
        return False  # one digest a metadata row: every stage would need to write it
    if get_disagg().disaggregation_decode_allocation_policy == "prefill_complete":
        return False
    return envs.SGLANG_TEST_FORCE_OPTIMISTIC_PREFILL_RETRY_PROB.get() <= 0


def _final_reqs(scheduler, batch):
    return [req for index, req in enumerate(batch.reqs)
            if _final_chunk(batch, index, req) and eligible(scheduler, req)]


def send_after_launch(scheduler, batch) -> list:
    """A stage other than the last, right after it launched ``batch``: send the last chunk of each eligible request of
    the batch, waiting on this launch, with no metadata row; mark the request. Returns the rids sent.

    Example:
        >>> send_after_launch(scheduler, batch)  # doctest: +SKIP
        ['rid-1']
    """
    if not enabled() or batch is None:
        return []
    from sglang_neuron.backend import device_io

    sent = []
    for req in _final_reqs(scheduler, batch):
        sender = req.disagg_kv_sender
        has_aux = hasattr(sender, "aux_index")  # SGLang's fake sender (a warm-up request) has none
        if hasattr(sender, "_early_send_wait_event"):
            sender._early_send_wait_event = device_io.last_launch_event()
        saved = sender.aux_index if has_aux else None
        setattr(req, NO_METADATA, True)
        try:
            if has_aux:
                sender.aux_index = NO_AUX
            scheduler.send_kv_chunk(req, last_chunk=True)
        finally:
            if has_aux:
                sender.aux_index = saved
            setattr(req, NO_METADATA, False)
        setattr(req, MARK, True)
        sent.append(req.rid)
    return sent


def send_early(scheduler, record) -> list:
    """The last stage, right before it sends ``record``'s outputs (on its host) to rank 0: send the last chunk of each
    eligible request of the batch with the first token, and mark the request. Returns the rids sent.

    Example:
        >>> send_early(scheduler, record)  # doctest: +SKIP
        ['rid-1']
    """
    if not enabled() or record is None or record.outputs is None:
        return []
    batch = record.batch
    tokens = None
    sent = []
    for index, req in enumerate(batch.reqs):
        if not _final_chunk(batch, index, req) or not eligible(scheduler, req):
            continue
        if tokens is None:
            tokens = record.outputs["next_token_ids"].tolist()
        # the metadata buffer reads the first token from output_ids; processing appends it as upstream does
        req.output_ids.append(tokens[index])
        try:
            scheduler.send_kv_chunk(req, last_chunk=True)
        finally:
            req.output_ids.pop()
        setattr(req, MARK, True)
        sent.append(req.rid)
    return sent


def skip_processing_send(req, last_chunk: bool) -> bool:
    """Whether the processing send of ``req``'s last chunk is the one this stage already did (called from
    pp_prefill_kv_after_forward.send_kv_chunk)."""
    return bool(last_chunk) and bool(getattr(req, MARK, False))


def set_buf(original_fn, self, req):
    """AROUND MetadataBuffers.set_buf: a stage's last-chunk send with no metadata row leaves the buffer as it is (the
    row is not sent, and the first token is not known there)."""
    if getattr(req, NO_METADATA, False):
        return None
    return original_fn(self, req)


def register(registry, around) -> None:
    """On a pipelined PD prefill: the set_buf hook (the loop hooks call this module directly, and
    nixl_settle_then_ack rebuilds the worker with WORKER_SUBSTITUTIONS), and the source check of the early answer
    (sglang_patches/pp_prefill_early_answer, which the loop calls)."""
    from sglang_neuron.sglang_patches import pp_prefill_early_answer, source_guard

    source_guard.check_targets(_TARGETS_MD5, __name__)
    pp_prefill_early_answer.check()
    registry.register(SET_BUF, set_buf, around)
