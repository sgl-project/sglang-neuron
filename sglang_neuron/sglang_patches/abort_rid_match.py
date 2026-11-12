# SPDX-License-Identifier: Apache-2.0
"""Match an abort to its own request and to that request's expanded ``<rid>_<i>`` copies, not to every id that starts
with it.

Upstream's ``Scheduler.abort_request``, ``Scheduler._abort_deferred_responses`` and
``GrammarManager.abort_requests`` match by prefix in every queue, so aborting
``req_1`` (a disconnect, ``/abort_request``, a timeout) also ends ``req_10`` .. ``req_19`` without an error. The prefix
match exists for the copies a batched ``GenerateReqInput`` expands one id into. The three functions are rebuilt from
upstream's source with each prefix test replaced by ``reaches``: every request under ``abort_all``, otherwise the exact
id and ``<id>_<digits>``. An empty id reaches only the request with that id and its copies.

The match depends only on the abort and the request id, which every rank and stage holds alike, so all ranks abort the
same requests at the same point. Diffusion-LLM, encoder-disaggregation and unlimited-OCR document-window queues
keep upstream's match; no served configuration builds them. Targets are md5-pinned against upstream SGLang 43881a7.

Upstream: a fix SGLang can take as it is: match an abort by its exact id or an expanded <id>_<digits> copy, not by
prefix."""

from __future__ import annotations

import re

_SCHEDULER = "sglang.srt.managers.scheduler"
_GRAMMAR = "sglang.srt.constrained.grammar_manager"
_IO = "sglang.srt.managers.io_struct"
# (module, qualified name): md5 of the source
_TARGETS_MD5 = {
    (_SCHEDULER, "Scheduler.abort_request"): "0fed02d75e27c867928d017e68f7b601",
    # the finished responses held back until their completion (Scheduler.defer_response), which abort_request aborts
    # first
    (_SCHEDULER, "Scheduler._abort_deferred_responses"): "de60d9c6772e8dcff16274374e62c157",
    (_GRAMMAR, "GrammarManager.abort_requests"): "747820c210135ebc215f8c3ce4c8ceb5",
    # a batch with one string id expands it into f"{rid}_{i}"
    (_IO, "GenerateReqInput._normalize_rid"): "148e8216736aa1bee747c6d9bd69db73",
    # an AbortReq without an id has the empty id
    (_IO, "AbortReq.__post_init__"): "cd4bcfc9e04f7c9c807f4dde41969104",
}
_EXPANSION = re.compile(r"_[0-9]+")
_IMPORT = "    from sglang_neuron.sglang_patches.abort_rid_match import reaches\n\n"
# Upstream's prefix matches in Scheduler.abort_request and their replacements. Most matches also carry the line before
# the test, so their text occurs once. The first replacement also imports reaches.
SCHEDULER_MATCHES = (
    # the request being chunked
    ("    if (chunked_req := self.chunked_req) is not None:\n"
     "        if recv_req.abort_all or chunked_req.rid.startswith(recv_req.rid):\n",
     _IMPORT +
     "    if (chunked_req := self.chunked_req) is not None:\n"
     "        if reaches(recv_req, chunked_req.rid):\n"),
    # the waiting queue
    ("    for i, req in enumerate(self.waiting_queue):\n"
     "        if recv_req.abort_all or req.rid.startswith(recv_req.rid):\n",
     "    for i, req in enumerate(self.waiting_queue):\n"
     "        if reaches(recv_req, req.rid):\n"),
    # a PD prefill's bootstrap queue
    ("        for req in self.disagg_prefill_bootstrap_queue.queue:\n"
     "            if recv_req.abort_all or req.rid.startswith(recv_req.rid):\n",
     "        for req in self.disagg_prefill_bootstrap_queue.queue:\n"
     "            if reaches(recv_req, req.rid):\n"),
    # a PD prefill's in-flight queue
    ("        for req in self.disagg_prefill_inflight_queue:\n"
     "            if recv_req.abort_all or req.rid.startswith(recv_req.rid):\n",
     "        for req in self.disagg_prefill_inflight_queue:\n"
     "            if reaches(recv_req, req.rid):\n"),
    # a PD decode's prealloc queue
    ("        for decode_req in self.disagg_decode_prealloc_queue.queue:\n"
     "            if recv_req.abort_all or decode_req.req.rid.startswith(recv_req.rid):\n",
     "        for decode_req in self.disagg_decode_prealloc_queue.queue:\n"
     "            if reaches(recv_req, decode_req.req.rid):\n"),
    # a PD decode's transfer queue
    ("        for decode_req in self.disagg_decode_transfer_queue.queue:\n"
     "            if recv_req.abort_all or decode_req.req.rid.startswith(recv_req.rid):\n",
     "        for decode_req in self.disagg_decode_transfer_queue.queue:\n"
     "            if reaches(recv_req, decode_req.req.rid):\n"),
    # a PD decode's retracted queue
    ("                if recv_req.abort_all or decode_req.rid.startswith(recv_req.rid):\n",
     "                if reaches(recv_req, decode_req.rid):\n"),
    # the running requests
    ("        if not req.finished() and (\n"
     "            recv_req.abort_all or req.rid.startswith(recv_req.rid)\n"
     "        ):\n",
     "        if not req.finished() and reaches(recv_req, req.rid):\n"),
)
# upstream's prefix match in Scheduler._abort_deferred_responses and its replacement, which imports reaches
DEFERRED_MATCHES = (
    ("    for rid, (req, completion) in list((self._deferred_responses or {}).items()):\n"
     "        if recv_req.abort_all or rid.startswith(recv_req.rid):\n",
     _IMPORT +
     "    for rid, (req, completion) in list((self._deferred_responses or {}).items()):\n"
     "        if reaches(recv_req, rid):\n"),
)
# upstream's prefix match in GrammarManager.abort_requests and its replacement, which imports reaches
GRAMMAR_MATCHES = (
    ("    for req in self.grammar_queue:\n"
     "        if recv_req.abort_all or req.rid.startswith(recv_req.rid):\n",
     _IMPORT +
     "    for req in self.grammar_queue:\n"
     "        if reaches(recv_req, req.rid):\n"),
)


def reaches(abort, rid: str) -> bool:
    """Whether ``abort`` (an AbortReq) reaches the request ``rid``. Under abort_all it reaches every request. Otherwise
    it reaches the abort's own id and ``<id>_<digits>``, the copies GenerateReqInput._normalize_rid expands that id
    into."""
    if abort.abort_all:
        return True
    return rid == abort.rid or (rid.startswith(abort.rid) and _EXPANSION.fullmatch(rid, len(abort.rid)) is not None)


def _function(module: str, qualname: str):
    import functools
    import importlib

    return functools.reduce(getattr, qualname.split("."), importlib.import_module(module))


def abort_request():
    """Scheduler.abort_request, upstream's with each prefix match replaced by reaches."""
    from sglang_neuron.sglang_patches import source_guard

    return source_guard.with_substitutions(_function(_SCHEDULER, "Scheduler.abort_request"), SCHEDULER_MATCHES)


def abort_deferred_responses():
    """Scheduler._abort_deferred_responses, upstream's with its prefix match replaced by reaches."""
    from sglang_neuron.sglang_patches import source_guard

    return source_guard.with_substitutions(_function(_SCHEDULER, "Scheduler._abort_deferred_responses"),
                                           DEFERRED_MATCHES)


def grammar_abort_requests():
    """GrammarManager.abort_requests, upstream's with its prefix match replaced by reaches."""
    from sglang_neuron.sglang_patches import source_guard

    return source_guard.with_substitutions(_function(_GRAMMAR, "GrammarManager.abort_requests"), GRAMMAR_MATCHES)


def register(registry, replace) -> None:
    from sglang_neuron.sglang_patches import source_guard

    for (module, qualname), md5 in _TARGETS_MD5.items():
        source_guard.check(_function(module, qualname), md5, f"{module}.{qualname}", __name__)
    registry.register(f"{_SCHEDULER}.Scheduler.abort_request", abort_request(), replace)
    registry.register(f"{_SCHEDULER}.Scheduler._abort_deferred_responses", abort_deferred_responses(), replace)
    registry.register(f"{_GRAMMAR}.GrammarManager.abort_requests", grammar_abort_requests(), replace)
