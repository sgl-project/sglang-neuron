# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""Let a short prompt's prefill run between two chunks of a long one (``SGLANG_NEURON_PREFILL_YIELD=<tokens>``).

A Neuron prefill forward runs one request (general_plugin._one_request_per_prefill), and upstream continues a chunked
request at every pass until its last chunk. A long cold prompt then holds every waiting request for all of its chunks:
under the AgentX trace a 170k-token cold prompt prefills for 36 s, and the requests behind it, most of them a few
hundred to a few thousand new tokens over a cached prefix, wait that long for their first token (their TTFT p90 is
several times their own prefill).

Here, ``Scheduler._get_new_batch_prefill_raw`` (wrapped): when a request is chunked, the pass before the chunked
request's next chunk is given instead to the first waiting request (of the first ``_SCAN``, in queue order) whose
uncached tokens fit ``SGLANG_NEURON_PREFILL_YIELD`` (at most one chunk, so it prefills in this one forward and never
becomes a second chunked request). The pass runs upstream's builder with the chunked request set aside and the waiting
queue narrowed to that request, so the batch carries no chunked request (the chunked request counts no middle chunk
for it, and its result is a whole prompt's). The chunked request was checkpointed after its last chunk
(``checkpoint_kv_cache`` sets its prefix to its extend end), so setting it back resumes it where it stopped. The next
pass always continues the chunked request: a long prompt prefills at no less than half its speed.

A pass yields only with room in the KV pool for the short request and a chunk beside it. Off (0) by default; a PD
prefill server or a pipelined one never yields.

Target, md5-pinned against upstream SGLang 43881a7: ``Scheduler._get_new_batch_prefill_raw`` (wrapped)."""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

ENV = "SGLANG_NEURON_PREFILL_YIELD"
_SCHEDULER = "sglang.srt.managers.scheduler"
_TARGETS_MD5 = {(_SCHEDULER, "Scheduler._get_new_batch_prefill_raw"): "54e259e006cb8a1a3ff15ba2d412d430"}
# the waiting requests a pass looks at for one that fits
_SCAN = 16


def limit() -> int:
    """The largest uncached prompt a pass gives a chunked request's turn to; 0 (unset) is off."""
    raw = os.environ.get(ENV, "0").strip()
    if not raw.isdigit():
        raise ValueError(f"{ENV}={raw!r}: a token count (0 for off)")
    return int(raw)


def _short_request(scheduler, tokens: int):
    """The first of the first _SCAN waiting requests whose uncached tokens fit ``tokens`` (its prefix matched as the
    builder matches it, Req.init_next_round_input), or None."""
    for req in scheduler.waiting_queue[:_SCAN]:
        if getattr(req, "grammar", None) is not None or req.finished():
            continue
        req.init_next_round_input(scheduler.tree_cache)
        if 0 < _uncached(req) <= tokens:
            return req
    return None


def _uncached(req) -> int:
    """The tokens a request's next prefill computes: its fill ids past the prefix it matched."""
    return len(req.full_untruncated_fill_ids) - int(req.prefix_len)


def _room(scheduler, tokens: int) -> bool:
    allocator = getattr(scheduler, "token_to_kv_pool_allocator", None)
    available = getattr(allocator, "available_size", None)
    return available is None or available() >= tokens + 2 * int(scheduler.chunked_prefill_size or 0)


_YIELDS = [0]


def _pipelined() -> bool:
    from sglang.srt.runtime_context import get_parallel

    return get_parallel().pp_size > 1


def _get_new_batch_prefill_raw(original_fn, self, prefill_delayer_single_pass, running_batch):
    chunked = self.chunked_req
    tokens = min(limit(), int(self.chunked_prefill_size or 0))
    yielded_last = getattr(self, "_neuron_prefill_yielded", False)
    self._neuron_prefill_yielded = False
    mode = getattr(self.disaggregation_mode, "value", self.disaggregation_mode)
    if (chunked is None or tokens <= 0 or yielded_last or not self.waiting_queue or str(mode) != "null"
            or _pipelined()):
        return original_fn(self, prefill_delayer_single_pass, running_batch)
    req = _short_request(self, tokens)
    if req is None or not _room(self, _uncached(req)):
        return original_fn(self, prefill_delayer_single_pass, running_batch)
    queue = self.waiting_queue
    self.waiting_queue, self.chunked_req = [req], None
    try:
        batch, running_batch = original_fn(self, prefill_delayer_single_pass, running_batch)
    finally:
        started = self.chunked_req
        left = self.waiting_queue  # req if the pass did not admit it, and any request it requeued
        self.chunked_req = chunked
        kept = set(map(id, queue))
        self.waiting_queue = [r for r in queue if r is not req or req in left] + [r for r in left if id(r) not in kept]
    if started is not None:
        # req fit one chunk, so the builder cannot have chunked it; two chunked requests would lose one
        raise RuntimeError(f"prefill yield: request {req.rid} became a chunked request beside {chunked.rid}")
    if batch is not None:
        self._neuron_prefill_yielded = True
        _YIELDS[0] += 1
        if _YIELDS[0] in (1, 10, 100) or _YIELDS[0] % 1000 == 0:
            logger.info("[neuron] prefill yield %d: %s (%d new tokens) ran before chunked %s's next chunk", _YIELDS[0],
                        req.rid, _uncached(req), chunked.rid)
    return batch, running_batch


def register(registry, around) -> None:
    from sglang_neuron.sglang_patches import source_guard

    if limit() <= 0:
        return
    source_guard.check_targets(_TARGETS_MD5, __name__)
    registry.register(f"{_SCHEDULER}.Scheduler._get_new_batch_prefill_raw", _get_new_batch_prefill_raw, around)
