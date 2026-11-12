# SPDX-License-Identifier: Apache-2.0
"""Answer a pipelined PD prefill's HTTP request at the request's last-chunk sends, not at its release
(``SGLANG_NEURON_PP_PREFILL_EARLY_ANSWER``, on by default; it acts only on the requests that
sglang_patches/pp_prefill_last_send sends at their forward).

SGLang's prefill answers a PD request when it releases it: ``process_disagg_prefill_inflight_queue`` finishes the
request with ``FINISH_LENGTH(0)``, releases its KV to the tree cache, and then streams it to the tokenizer. A pipelined
prefill releases a request through the release consensus of iteration i - L (L = pp_size + 1, the overlap-order loop
sglang_patches/pp_prefill_overlap_loop), about L chunk times after its last chunk under load (measured on
Kimi-K3 PP8 with every stage sending at its forward). SGLang sends a streamed response's HTTP 200 only with its first
chunk, and the SGLang gateway (sgl-model-gateway pd_router.rs) reads the prefill's whole response before it uses the
decode's, so the client's first token waits for the release although the decode has it.

Here a stage answers a request right after it processes the request's last chunk, if the stage sent that chunk at its
forward (``pp_prefill_last_send.MARK``) and the request now waits in the in-flight queue. The answer is SGLang's own
``stream_output`` of the request with the release's reason, ``FINISH_LENGTH(0)``, set for the call only: the tokenizer
gets the same finished output (the first token, the counts) that the release would give, and ends the HTTP request.
The stream marks the request ``finished_output``, so the release's ``stream_output`` skips it (upstream's rule for an
output already sent), and nothing reaches the tokenizer twice.

The answer touches no KV, no tree cache entry, no lock, no metadata buffer and no transfer: the release consensus
still finishes the request, releases its KV (``release_kv_cache``, ``tree_cache.finish``), clears its sender and frees
its metadata buffer at the same iteration as without the answer, so the prompt cache (its keep and adopt) and its host
tier see the same calls in the same order. Once answered, the tokenizer has no state for the request, so a client that
closes its connection cannot abort it; the release consensus ends it as before.

Only PP rank 0's attention TP rank 0 has the tokenizer's socket, so rank 0's processing gives the answer; every other
stage makes the same call, whose send goes nowhere, and keeps the same request state. Rank 0 processes a batch only
after it takes the last stage's outputs, and the last stage sends a request's last chunk before those outputs
(pp_prefill_overlap_loop.send_outputs), so the answer follows every stage's last-chunk send. Exception: a stage whose
bootstrap is still pending at the last chunk sends it later (pp_prefill_last_send.eligible); the decode still waits for
it, and the KV lifetime does not change.

A transfer that fails after the answer is reported by the decode (its transfer fails, and pd_metadata_wait names a
missing metadata row); the client already has the prefill's answer, which only says that the prefill computed the
request.

Upstream: generic for a pipelined PD prefill whose release lags its last-chunk sends; upstream would take it with the
overlap-order loop.

Targets, md5-pinned against upstream SGLang: ``process_disagg_prefill_inflight_queue`` (the release: its reason, its
order of release then stream), ``SchedulerOutputStreamer._stream_output_generation`` (it skips a finished request
already output) and ``_GenerationStreamAccumulator.accept`` (it marks a finished request output)."""

from __future__ import annotations

import functools
import os
import time

ENV = "SGLANG_NEURON_PP_PREFILL_EARLY_ANSWER"
# set on a request once its answer went to the tokenizer
ANSWERED = "sgn_early_answered"
_STREAMER = "sglang.srt.managers.scheduler_components.output_streamer"
# (module, qualified name): md5 of the source the answer relies on
_TARGETS_MD5 = {
    ("sglang.srt.disaggregation.prefill", "SchedulerDisaggregationPrefillMixin.process_disagg_prefill_inflight_queue"):
        "5b49bcb8e522b632073c48f2c075fd7f",
    (_STREAMER, "SchedulerOutputStreamer._stream_output_generation"): "052e5ca6d4626092ed4d4a3df3f6a948",
    (_STREAMER, "_GenerationStreamAccumulator.accept"): "9fafa03f0b7286a2268cc8c0277411e3",
}


@functools.lru_cache(maxsize=1)
def enabled() -> bool:
    """``SGLANG_NEURON_PP_PREFILL_EARLY_ANSWER`` (read once): True when unset or 1, False for 0; another value is
    refused naming it.

    Example:
        >>> enabled()  # doctest: +SKIP
        True
    """
    raw = os.environ.get(ENV, "").strip()
    if raw in ("", "1"):
        return True
    if raw == "0":
        return False
    raise ValueError(f"{ENV}={raw!r}: set 1 (a pipelined PD prefill answers a request at its last-chunk sends) or 0 "
                     f"(it answers at the release).")


def _waiting(scheduler, batch) -> list:
    """The requests of ``batch`` that this stage sent at their forward, that now wait in the in-flight queue, and that
    have no answer and no finish yet."""
    from sglang_neuron.sglang_patches.pp_prefill_last_send import MARK

    queue = scheduler.disagg_prefill_inflight_queue
    return [req for req in batch.reqs
            if getattr(req, MARK, False) and not getattr(req, ANSWERED, False) and req.finished_reason is None
            and not req.finished_output and any(queued is req for queued in queue)]


def answer(scheduler, batch) -> list:
    """After a stage processed ``batch``: answer each request whose last chunk the stage sent at its forward and that
    waits in the in-flight queue, with SGLang's ``stream_output`` and the release's reason for the call only. Returns
    the rids answered.

    Example:
        >>> answer(scheduler, batch)  # doctest: +SKIP
        ['rid-1']
    """
    if batch is None or not enabled():
        return []
    reqs = _waiting(scheduler, batch)
    if not reqs:
        return []
    from sglang.srt.managers.schedule_batch import FINISH_LENGTH

    now = time.perf_counter()
    for req in reqs:
        # the reason the release gives (process_disagg_prefill_inflight_queue); the release sets it again
        req.finished_reason = FINISH_LENGTH(length=0)
        stats = getattr(req, "time_stats", None)
        if stats is not None:
            stats.completion_time = now  # the time the client gets its answer; the release sets it again
    try:
        # (b) sends no request with logprobs at its forward (pp_prefill_last_send.eligible)
        scheduler.output_streamer.stream_output(reqs, False, None)
    finally:
        for req in reqs:
            req.finished_reason = None
    answered = [req for req in reqs if req.finished_output]  # SGLang may hold an output (defer_output)
    for req in answered:
        setattr(req, ANSWERED, True)
    return [req.rid for req in answered]


def check() -> None:
    """Refuse an upstream whose release or output stream differs from the pinned source (called from
    pp_prefill_last_send.register, on every pipelined PD prefill)."""
    from sglang_neuron.sglang_patches import source_guard

    source_guard.check_targets(_TARGETS_MD5, __name__)
