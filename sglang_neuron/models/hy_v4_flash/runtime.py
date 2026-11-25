# SPDX-License-Identifier: Apache-2.0
"""Hy4-Flash's runtime: the platform runtime (model_executor/graph_runner.NeuronModelRuntime) over a ladder of context
rungs, so a server of the model's whole context (1M tokens) runs a short request in short graphs.

A sparse layer's work grows with the table it is handed: its IndexPool scores every 4-token group of the table, and a
prefill's attention reads the table's keys. So both families take the context in rungs (``contexts``: 16K, 64K, 256K
and 1M tokens by default, capped at ``--context-length``, which is always one; ``SGLANG_NEURON_HY4F_CONTEXTS`` lists
others, in tokens or with a K suffix):

* a decode step runs the graph of the smallest rung that holds its longest row (the platform's decode contexts);
* a prefill step runs the graph of its token bucket and of the smallest rung that holds the request's cached prior and
  the chunk, its tables (every layer's, the window pool's too) cut to the rung's pages. The warm-up compiles every
  token bucket at every rung.

The sliding layers read only their window in either family, and the window pool frees what is behind it."""

from __future__ import annotations

import dataclasses
import os

from sglang_neuron.model_executor.graph_runner import P_MAX, NeuronModelRuntime, _lnc_degree, _round_up

CONTEXTS_ENV = "SGLANG_NEURON_HY4F_CONTEXTS"
DEFAULT_CONTEXTS = (16 * 1024, 64 * 1024, 256 * 1024, 1024 * 1024)


def context_rungs(max_model_len: int, environ=os.environ) -> list:
    """The context rungs of a server of ``max_model_len`` tokens (module docstring), each whole decode tiles."""
    tile = _lnc_degree() * P_MAX
    listed = environ.get(CONTEXTS_ENV, "").strip()
    rungs = set()
    for item in (listed.split(",") if listed else DEFAULT_CONTEXTS):
        item = str(item).strip().lower()
        value = int(item[:-1]) * 1024 if item.endswith("k") else int(item)
        if value <= 0:
            raise ValueError(f"{CONTEXTS_ENV}={listed!r}: a rung must be above 0")
        rungs.add(_round_up(min(value, max_model_len), tile))
    rungs.add(_round_up(max_model_len, tile))
    return sorted(rungs)


class Hy4FlashRuntime(NeuronModelRuntime):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.contexts = context_rungs(int(self.buckets.max_model_len))
        self.buckets.decode_contexts = list(self.contexts)

    def _precompile(self, shapes) -> None:
        """Rank 0 traces every shape and queues its compiles before the other ranks start theirs.

        Under the precompile cache every rank traces the shapes in one order (graph_runner._precompile), so all of
        them would submit each shape's compile at the same moment, before any could find another's in flight: one
        compile per rank of a graph that takes the host's memory and cores many times over (a cold Hy4-Flash decode
        graph's 32 concurrent compiles ran the host out of memory). Rank 0 goes first; the others then trace from the
        StableHLO cache it filled and find its compiles."""
        from sglang_neuron.parallel.state import tp_barrier

        if self.tp_rank == 0:
            super()._precompile(shapes)
            tp_barrier()
        else:
            tp_barrier()
            super()._precompile(shapes)

    def _warm_shapes(self) -> list:
        if self.graphs == "prefill":
            return [("prefill", tb, c) for tb in self.buckets.prefill_token_buckets for c in self.contexts]
        return super()._warm_shapes()

    def prefill_context(self, md) -> int:
        """The smallest rung that holds the request's cached prior and this chunk."""
        need = int(md.cached_seq_len) + int(md.num_tokens)
        for c in self.contexts:
            if c >= need:
                return c
        return self.contexts[-1]

    def run_prefill(self, fb, md, bucket=None, pp_proxy=None):
        self._prefill_rung = self.prefill_context(md)
        try:
            return super().run_prefill(fb, md, bucket, pp_proxy=pp_proxy)
        finally:
            self._prefill_rung = None

    def build_model_inputs(self, md, token_ids, bucket, *args, context=None, **kwargs):
        """A prefill's inputs over its rung's table: the platform builds the host table to the bucket config's
        longest context, so it builds it here under a config whose longest context is the rung."""
        rung = context if context is not None else getattr(self, "_prefill_rung", None)
        if not md.is_prefill or rung is None:
            return super().build_model_inputs(md, token_ids, bucket, *args, context=context, **kwargs)
        whole = self.buckets
        self.buckets = dataclasses.replace(whole, max_model_len=int(rung))
        try:
            return super().build_model_inputs(md, token_ids, bucket, *args, context=context, **kwargs)
        finally:
            self.buckets = whole
