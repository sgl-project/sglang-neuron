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
uncached tokens fit ``SGLANG_NEURON_PREFILL_YIELD``. The pass runs upstream's builder with the chunked request set aside
and the waiting queue narrowed to that request. The chunked request was checkpointed after its last chunk
(``checkpoint_kv_cache`` sets its prefix to its extend end), so setting it back resumes it where it stopped.

- A request of at most one chunk prefills whole in that pass (the batch carries no chunked request, and its result is
  a whole prompt's); the next pass always continues the chunked request.
- A longer one (up to the limit, e.g. 8192 tokens = 4 chunks of 2048) takes over as the chunked request: the long one
  is suspended, the short one prefills chunk by chunk to its end, and the pass after its last chunk resumes the long
  one. No yield happens while one is suspended, so requests never nest deeper than one.

A pass yields only with room in the KV pool for the short request and a chunk beside it. Off (0) by default; a PD
decode server or a pipelined one never yields.

On a PD prefill server ``process_prefill_chunk`` (wrapped) runs after every pass and, while a request is chunked,
checkpoints it and sends its last chunk's KV to its decode. After a pass that yielded, the chunked request ran no chunk:
the wrapper hides it from that call, which then only retires the short request's batch, and the next pass sends the
chunked request's next chunk as usual.

Targets, md5-pinned against upstream SGLang 43881a7: ``Scheduler._get_new_batch_prefill_raw`` and
``SchedulerDisaggregationPrefillMixin.process_prefill_chunk`` (both wrapped)."""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

ENV = "SGLANG_NEURON_PREFILL_YIELD"
_SCHEDULER = "sglang.srt.managers.scheduler"
_PREFILL = "sglang.srt.disaggregation.prefill"
_TARGETS_MD5 = {(_SCHEDULER, "Scheduler._get_new_batch_prefill_raw"): "54e259e006cb8a1a3ff15ba2d412d430",
                (_PREFILL, "SchedulerDisaggregationPrefillMixin.process_prefill_chunk"):
                    "dca76f30fd359b20c4c3b2e861490e34"}
# the waiting requests a pass looks at for one that fits
_SCAN = 16


def limit() -> int:
    """The largest uncached prompt a pass gives a chunked request's turn to; 0 (unset) is off."""
    raw = os.environ.get(ENV, "0").strip()
    if not raw.isdigit():
        raise ValueError(f"{ENV}={raw!r}: a token count (0 for off)")
    return int(raw)


def _short_request(scheduler, tokens: int) -> int:
    """The queue index of the first of the first _SCAN waiting requests whose uncached tokens fit ``tokens`` (its prefix
    matched as the builder matches it, Req.init_next_round_input) and the KV pool's room (_room), or -1."""
    for i, req in enumerate(scheduler.waiting_queue[:_SCAN]):
        if getattr(req, "grammar", None) is not None or req.finished():
            continue
        req.init_next_round_input(scheduler.tree_cache)
        if 0 < _uncached(req) <= tokens:
            return i if _room(scheduler, _uncached(req)) else -1
    return -1


def _agreed(scheduler, index: int) -> int:
    """``index`` if every TP rank chose the same one, else -1. Each rank's choice reads its own KV pool, and under DCP the
    ranks' pools differ (a rank owns every D-th page), so ranks could choose differently and build different batches,
    which hangs the TP collectives (the scheduler watchdog fired with one rank's batch empty and another's not). One
    MIN all-reduce of (index, -index) over the TP CPU group gives the lowest and the highest choice."""
    import torch
    import torch.distributed as dist

    group = getattr(scheduler, "tp_cpu_group", None)
    if group is None or not dist.is_initialized() or dist.get_world_size(group) == 1:
        return index
    both = torch.tensor([index, -index], dtype=torch.int64)
    dist.all_reduce(both, op=dist.ReduceOp.MIN, group=group)
    low, high = int(both[0]), -int(both[1])
    return index if low == high else -1


def _uncached(req) -> int:
    """The tokens a request's next prefill computes: its fill ids past the prefix it matched, on the device or in
    HiCache's host tier (``num_matched_prefix_tokens`` = ``prefix_len`` + ``host_hit_length``, which the builder loads
    back rather than computes). Counting the device prefix only, a PD prefill whose prefixes sit mostly in L2 took
    nearly every multi-turn request for a long one and yielded a few times an hour."""
    matched = max(int(req.prefix_len), int(getattr(req, "num_matched_prefix_tokens", 0) or 0))
    return len(req.full_untruncated_fill_ids) - matched


def _room(scheduler, tokens: int) -> bool:
    """Whether the KV pool holds the short request and two chunks beside it: its free pages and the radix cache's
    evictable ones, which the batch builder evicts as it allocates. A server that keeps every finished prompt cached
    (a PD prefill under AgentX) has almost no free pages, only evictable ones."""
    allocator = getattr(scheduler, "token_to_kv_pool_allocator", None)
    available = getattr(allocator, "available_size", None)
    if available is None:
        return True
    evictable = getattr(getattr(scheduler, "tree_cache", None), "evictable_size", None)
    room = available() + (int(evictable()) if evictable is not None else 0)
    return room >= tokens + 2 * int(scheduler.chunked_prefill_size or 0)


_YIELDS = [0]


def _pipelined() -> bool:
    from sglang.srt.runtime_context import get_parallel

    return get_parallel().pp_size > 1


def _get_new_batch_prefill_raw(original_fn, self, prefill_delayer_single_pass, running_batch):
    suspended = getattr(self, "_neuron_prefill_suspended", None)
    if suspended is not None and self.chunked_req is None:
        # the turn that took over a chunked request's turn (below) prefilled its last chunk: the suspended request
        # continues, this pass
        self.chunked_req, self._neuron_prefill_suspended = suspended, None
        self._neuron_prefill_yielded = False
        if suspended.finished() or getattr(self, "_pending_chunked_abort_req", None) is suspended:
            # it was aborted (or its transfer failed) while suspended: upstream's chunked-request teardown releases it
            # (its KV, metadata buffer and pending sends), never a checkpoint of a finished request
            self._pending_chunked_abort_req = suspended
            self.process_pending_chunked_abort()
        return original_fn(self, prefill_delayer_single_pass, running_batch)
    chunked = self.chunked_req
    tokens = limit()
    chunk = int(self.chunked_prefill_size or 0)
    yielded_last = getattr(self, "_neuron_prefill_yielded", False)
    self._neuron_prefill_yielded = False
    mode = getattr(self.disaggregation_mode, "value", self.disaggregation_mode)
    if (chunked is None or suspended is not None or tokens <= 0 or chunk <= 0 or yielded_last or not self.waiting_queue
            or str(mode) not in ("null", "prefill") or _pipelined()):
        return original_fn(self, prefill_delayer_single_pass, running_batch)
    # every rank reaches this point together (the conditions above follow the broadcast requests and the agreed
    # history), so every rank takes part in the agreement
    index = _agreed(self, _short_request(self, tokens))
    if index < 0:
        return original_fn(self, prefill_delayer_single_pass, running_batch)
    req = self.waiting_queue[index]
    queue = self.waiting_queue
    self.waiting_queue, self.chunked_req = [req], None
    try:
        batch, running_batch = original_fn(self, prefill_delayer_single_pass, running_batch)
    except BaseException:
        self.chunked_req = chunked
        raise
    finally:
        left = self.waiting_queue  # req if the pass did not admit it, and any request it requeued
        kept = set(map(id, queue))
        self.waiting_queue = [r for r in queue if r is not req or req in left] + [r for r in left if id(r) not in kept]
    started = self.chunked_req
    if started is None:
        # req prefilled whole in this pass: the chunked request continues next pass
        self.chunked_req = chunked
        if batch is not None:
            self._neuron_prefill_yielded = True
            batch.neuron_prefill_yield = True
            _note(req, chunked, "ran before")
    elif started is req:
        # req needs more than one chunk: it takes over as the chunked request until its last chunk, the long one
        # suspended where its last chunk left it (checkpointed, its KV held) and continued right after
        self._neuron_prefill_suspended = chunked
        _note(req, chunked, "took over from")
    else:
        self.chunked_req = chunked
        raise RuntimeError(f"prefill yield: request {started.rid} became a chunked request beside {chunked.rid}")
    return batch, running_batch


def _note(req, chunked, what: str) -> None:
    _YIELDS[0] += 1
    if _YIELDS[0] in (1, 10, 100) or _YIELDS[0] % 1000 == 0:
        logger.info("[neuron] prefill yield %d: %s (%d new tokens) %s chunked %s", _YIELDS[0], req.rid, _uncached(req),
                    what, chunked.rid)


def _process_prefill_chunk(original_fn, self, last_batch, running_batch):
    """A PD prefill's per-pass chunk step: after a yielding pass the chunked request ran no chunk, so it is hidden from
    the step (no second checkpoint or send of its last chunk); the step still retires the short request's batch."""
    req = self.chunked_req
    if req is not None and req.finished() and getattr(self, "_pending_chunked_abort_req", None) is None:
        # a chunked request that finished outside a step (aborted, or its transfer failed, while suspended or set aside)
        # goes through upstream's teardown, never the step's checkpoint (which asserts on a finished request)
        self._pending_chunked_abort_req = req
        self.process_pending_chunked_abort()
    if last_batch is None or not getattr(last_batch, "neuron_prefill_yield", False) or self.chunked_req is None:
        return original_fn(self, last_batch=last_batch, running_batch=running_batch)
    chunked, self.chunked_req = self.chunked_req, None
    try:
        return original_fn(self, last_batch=last_batch, running_batch=running_batch)
    finally:
        if self.chunked_req is not None:
            raise RuntimeError(f"prefill yield: the chunk step set chunked request {self.chunked_req.rid} beside "
                               f"{chunked.rid}")
        self.chunked_req = chunked


def _abort_request(original_fn, self, recv_req):
    """Scheduler.abort_request: a suspended chunked request is in no queue upstream looks at, so an abort of it would be
    lost; mark it as the pending chunked abort, which the pass that resumes it tears down."""
    out = original_fn(self, recv_req)
    suspended = getattr(self, "_neuron_prefill_suspended", None)
    if suspended is not None and (getattr(recv_req, "abort_all", False) or suspended.rid.startswith(recv_req.rid)):
        self._pending_chunked_abort_req = suspended
    return out


def register(registry, around) -> None:
    from sglang_neuron.sglang_patches import source_guard

    if limit() <= 0:
        return
    source_guard.check_targets(_TARGETS_MD5, __name__)
    registry.register(f"{_SCHEDULER}.Scheduler._get_new_batch_prefill_raw", _get_new_batch_prefill_raw, around)
    registry.register(f"{_PREFILL}.SchedulerDisaggregationPrefillMixin.process_prefill_chunk", _process_prefill_chunk,
                      around)
    registry.register(f"{_SCHEDULER}.Scheduler.abort_request", _abort_request, around)
