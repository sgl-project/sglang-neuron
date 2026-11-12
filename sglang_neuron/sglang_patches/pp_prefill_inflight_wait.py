# SPDX-License-Identifier: Apache-2.0
"""Hold a waiting request on a pipelined PD prefill while a request whose whole prompt it extends is in flight, so the
waiting request takes the prefix-cache hit of that request's release (``SGLANG_NEURON_PP_PREFILL_INFLIGHT_WAIT_PASSES``,
default 1,000,000 forward passes; ``0``: no wait).

A PD prefill inserts the end of a request's prompt into the radix tree only at its release (``release_kv_cache``),
after the KV transfer and the PP release consensus, about pp_size + 1 chunk times after the last chunk under load. A
client can have its answer before that: the early answer (sglang_patches/pp_prefill_early_answer), or a balancer that
streams the decode's tokens without the prefill's answer. Its next turn can then reach the prefill before the insert
and compute again what the release would give it (measured with an insert only at the release: 4 next turns sent just
after their answers got no hits; with the answer at the release, every one hit exactly).

Here such a request stays out of the scheduling until that release, and then takes the hit. A request waits when:

* its prompt is longer than the prompt of an unreleased request and starts with all of it, with the same LoRA, cache
  salt and extra key. An unreleased request is one in the in-flight queue, or one whose last chunk is in a launched
  batch that this stage has not processed yet (the overlap-order loop's records, sglang_patches/
  pp_prefill_overlap_loop): a balancer that streams the decode's answer early can give the client its answer before
  the stage processes that chunk and moves the request to the in-flight queue;
* that request is not aborted and its release inserts into the tree (not ``skip_radix_cache_insert``);
* its release inserts more than the tree holds of it now: ``release_insert_len`` (the tree's own method when it has one,
  else the prompt floored to the tree page) is above the request's ``kv.cache_inserted_len``. A tree that inserts each
  whole chunk as the chunk is computed (the chunk-keyed tree) holds all but the last whole chunk already, so a request
  waits there only when the release adds a chunk.

The other requests schedule as before: the wait holds no queue. The limit counts only the passes in which this prefill
schedules a forward (every PP stage schedules the same batches, so every stage counts the same): an idle prefill counts
nothing, so a long transfer does not end the wait. A predecessor that fails or is aborted ends the wait, and so does
its release. After the limit the request schedules with the hit it has, and a warning names it.

Every PP stage holds the same requests at the same pass: the hold reads only the stage's waiting queue, its in-flight
queue (which the release consensus empties at the same pass on every stage), its launched records (every stage
launches and processes the same batches at the same loop points), the requests' prompts and the tree's inserted
lengths (the same inserts on every stage).

Upstream: generic for a PD prefill whose release lags its answer; upstream would hold such a request in its admission.

Target, md5-pinned against upstream SGLang 43881a7: ``Scheduler.get_new_batch_prefill`` (wrapped)."""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

ENV = "SGLANG_NEURON_PP_PREFILL_INFLIGHT_WAIT_PASSES"
DEFAULT_PASSES = 1_000_000
TARGET = "sglang.srt.managers.scheduler.Scheduler.get_new_batch_prefill"
_WAIT_ATTR = "_neuron_inflight_wait"
_LOG = "[neuron] PD prefill:"
# (module, qualified name): md5 of the source the hold wraps
_TARGETS_MD5 = {("sglang.srt.managers.scheduler", "Scheduler.get_new_batch_prefill"):
                "f26088aa5d6704902fea48c9b447a5b2"}


class _Wait:
    """A waiting request's state: the forward passes it was held, the predecessors it extends (rid -> bool, computed
    once), and the rid it waits for."""

    __slots__ = ("passes", "extends", "waits_for")

    def __init__(self) -> None:
        self.passes, self.extends, self.waits_for = 0, {}, None


def wait_passes(environ=os.environ) -> int:
    """``SGLANG_NEURON_PP_PREFILL_INFLIGHT_WAIT_PASSES``: the pass limit of the hold (default 1,000,000; 0: no hold).
    A value that is not a non-negative integer is refused by name.

    Example:
        >>> wait_passes({})
        1000000
    """
    from sglang_neuron import env_registry

    if environ.get(ENV) is None:
        return DEFAULT_PASSES
    return env_registry.non_negative_int(ENV, environ)


def _salt_key(req: Any) -> tuple:
    return (getattr(req, "lora_id", None), getattr(req, "cache_salt", None), getattr(req, "extra_key", None))


def _starts_with(tokens: Any, prefix: Any) -> bool:
    """Whether ``tokens`` is longer than ``prefix`` and starts with all of it (lists or arrays of token ids)."""
    n = len(prefix)
    if n == 0 or len(tokens) <= n:
        return False
    if tokens[0] != prefix[0] or tokens[n - 1] != prefix[n - 1] or tokens[n // 2] != prefix[n // 2]:
        return False
    if type(tokens) is type(prefix):
        return tokens[:n] == prefix
    return list(tokens[:n]) == list(prefix)


def release_insert_len(tree: Any, prior: Any) -> int:
    """The prompt length that ``prior``'s release inserts into ``tree``: the tree's own ``release_insert_len`` when it
    has one, else the prompt floored to the tree page."""
    own = getattr(tree, "release_insert_len", None)
    if callable(own):
        return int(own(prior))
    page = max(1, int(getattr(tree, "page_size", 1) or 1))
    return len(prior.origin_input_ids) // page * page


def _release_adds(tree: Any, prior: Any) -> bool:
    """Whether the release of in-flight ``prior`` inserts more than the tree holds of it now, and inserts at all (not
    aborted, not a request whose insert SGLang skips)."""
    from sglang.srt.managers.schedule_batch import FINISH_ABORT

    if isinstance(getattr(prior, "finished_reason", None), FINISH_ABORT) or getattr(prior, "skip_radix_cache_insert",
                                                                                      False):
        return False
    if tree is None or getattr(tree, "disable", False):
        return False
    kv = getattr(prior, "kv", None)
    inserted = int(getattr(kv, "cache_inserted_len", 0) or 0)
    return release_insert_len(tree, prior) > inserted


def _final_chunk(batch: Any, index: int, req: Any) -> bool:
    """Whether ``batch`` runs the last chunk of its request at ``index`` (a cached prefix counts in prefix_lens)."""
    return int(batch.prefix_lens[index]) + int(batch.extend_lens[index]) >= len(req.origin_input_ids)


def unreleased(scheduler: Any) -> list:
    """The requests that this stage has computed to their last chunk and not released: the in-flight queue, then the
    requests whose last chunk is in a launched, unprocessed batch (``scheduler.pp_overlap_records``), each once."""
    out, seen = [], set()
    for req in getattr(scheduler, "disagg_prefill_inflight_queue", None) or ():
        if id(req) not in seen:
            seen.add(id(req))
            out.append(req)
    for record in getattr(scheduler, "pp_overlap_records", None) or ():
        batch = getattr(record, "batch", None)
        for index, req in enumerate(getattr(batch, "reqs", None) or ()):
            if id(req) not in seen and _final_chunk(batch, index, req):
                seen.add(id(req))
                out.append(req)
    return out


def _extends(req: Any, prior: Any, state: _Wait) -> bool:
    known = state.extends.get(prior.rid)
    if known is None:
        known = _salt_key(req) == _salt_key(prior) and _starts_with(req.origin_input_ids, prior.origin_input_ids)
        state.extends[prior.rid] = known
    return known


def held_for_inflight(scheduler: Any, limit: Optional[int] = None) -> list:
    """The waiting requests that this scheduling pass holds back (the module docstring).

    Example:
        >>> held_for_inflight(scheduler)  # doctest: +SKIP
        [<Req rid=turn-2>]
    """
    limit = wait_passes() if limit is None else limit
    if not limit:
        return []
    tree = getattr(scheduler, "tree_cache", None)
    inflight = [prior for prior in unreleased(scheduler) if _release_adds(tree, prior)]
    if not inflight:
        return []
    held = []
    for req in scheduler.waiting_queue:
        state = getattr(req, _WAIT_ATTR, None)
        if state is not None and state.passes >= limit:
            continue
        if state is None:
            state = _Wait()
        prior = next((p for p in inflight if p is not req and _extends(req, p, state)), None)
        if prior is None:
            state.waits_for = None
            setattr(req, _WAIT_ATTR, state)
            continue
        if state.waits_for != prior.rid:
            _log_rank0("%s request %s extends request %s in flight (%d of its %d prompt tokens): it waits for that "
                       "release, then takes the hit", _LOG, req.rid, prior.rid, len(prior.origin_input_ids),
                       len(req.origin_input_ids))
        state.waits_for = prior.rid
        setattr(req, _WAIT_ATTR, state)
        held.append(req)
    return held


def count_forward_pass(held: list, limit: Optional[int] = None) -> None:
    """After a pass that scheduled a forward: count it for each request held in it. At the limit, warn that the request
    schedules next pass and computes the rest of its prompt without the hit."""
    limit = wait_passes() if limit is None else limit
    for req in held:
        state = getattr(req, _WAIT_ATTR)
        state.passes += 1
        if state.passes == limit:
            logger.warning("%s request %s was held %d forward passes for request %s in flight, and that request is not "
                           "released. The request schedules now without that hit. Check the KV transfer of request "
                           "%s, or raise %s.", _LOG, req.rid, state.passes, state.waits_for, state.waits_for, ENV)


def _log_rank0(msg: str, *args: Any) -> None:
    try:
        from sglang.srt.runtime_context import get_parallel

        parallel = get_parallel()
        if parallel.pp_rank != 0 or parallel.attn_tp_rank != 0:
            return
    except Exception:  # noqa: BLE001 - no runtime context (a test): log
        pass
    logger.info(msg, *args)


def _put_back(queue_after: list, queue_before: list, held: list) -> list:
    """``queue_after`` (what the scheduling left) with the held requests back at their places of ``queue_before``:
    each goes before the first request that followed it there and is still waiting, else at the end."""
    out = list(queue_after)
    held_ids = {id(req) for req in held}
    for i, req in enumerate(queue_before):
        if id(req) not in held_ids:
            continue
        later = {id(other) for other in queue_before[i + 1:] if id(other) not in held_ids}
        at = next((k for k, other in enumerate(out) if id(other) in later), len(out))
        out.insert(at, req)
    return out


def get_new_batch_prefill(original_fn, self, *args, **kwargs):
    """AROUND ``Scheduler.get_new_batch_prefill``: schedule without the requests that wait for an in-flight request
    they extend (``held_for_inflight``), then put them back in the waiting queue at their places, so an abort still
    finds them and the next pass sees them again. A pass that schedules a forward counts toward their limit."""
    held = held_for_inflight(self)
    if not held:
        return original_fn(self, *args, **kwargs)
    before = list(self.waiting_queue)
    held_ids = {id(req) for req in held}
    self.waiting_queue = [req for req in before if id(req) not in held_ids]
    try:
        plan = original_fn(self, *args, **kwargs)
    finally:
        self.waiting_queue = _put_back(self.waiting_queue, before, held)
    if getattr(plan, "batch_to_run", None) is not None:
        count_forward_pass(held)
    return plan


def register(registry, around) -> None:
    """On a pipelined PD prefill (general_plugin): the hold. Registers nothing at a limit of 0."""
    from sglang_neuron.sglang_patches import source_guard

    if not wait_passes():
        return
    source_guard.check_targets(_TARGETS_MD5, __name__)
    registry.register(TARGET, get_new_batch_prefill, around)
