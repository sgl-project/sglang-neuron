# SPDX-License-Identifier: Apache-2.0
"""Runtime of a GLM-5.2 PD prefill stage: the platform's NeuronModelRuntime with the stage graph's own step inputs.

Token buckets, compile, warm-up with its SPMD NEFF check and in-place proof, and the pipeline loop are the platform's.
The model owns the step inputs (build_model_inputs: the stage graph takes the kernel's inputs, every token's owner slot,
the request's page ids over a fixed-width table and the prior as a [1, 1] tensor, where the platform's graphs take
per-layer attention metadata), the request's context bucket (prefill_context, and the warm-up shapes over it), the
logit test's readback in run_prefill, and the start-up checks of the served point and of the fleet.
"""

from __future__ import annotations

import dataclasses
import functools
from typing import Optional

import torch

from sglang_neuron.backend.device_io import upload_many
from sglang_neuron.model_executor.graph_runner import NeuronModelRuntime
from sglang_neuron.model_executor.runtime_init import neuron_device
from sglang_neuron.models.glm52.prefill import inputs as prefill_inputs
from sglang_neuron.models.glm52.launch import FIRST_TOKEN_SAMPLERS, context_buckets
from sglang_neuron.models.glm52.prefill.stage_layout import CP_INTERLEAVE, prefill_point


class Glm52PrefillRuntime(NeuronModelRuntime):
    """A PD prefill stage's runtime (the model host's runtime_class)."""

    def __init__(self, model_runner, sgl_model, phase=None):
        super().__init__(model_runner, sgl_model, phase)
        self._check_served()
        self._check_fleet()

    @functools.cached_property
    def context_buckets(self) -> tuple:
        """The stage graphs' context buckets up to the server's context (launch.context_buckets), taken once: the
        server's context is fixed for its life, and every step reads them."""
        return context_buckets(int(self.buckets.max_model_len))

    def prefill_context(self, tokens: int) -> int:
        """The smallest context bucket that holds a step whose request then holds ``tokens`` tokens (prior + new)."""
        for bucket in self.context_buckets:
            if tokens <= bucket:
                return bucket
        raise ValueError(f"a prefill step reaching {tokens} tokens is past the largest context bucket "
                         f"{self.context_buckets[-1]}")

    def _warm_shapes(self) -> list:
        """Every (token bucket, context bucket) stage graph: a step runs the graph of its tokens and its request's
        context (prefill_context), each compiled at warm-up."""
        return [("prefill", tb, c) for tb in self.buckets.prefill_token_buckets for c in self.context_buckets]

    def _check_served(self) -> None:
        """Refuse what the stage graph does not run: a decode family, buckets other than its point's
        (PrefillPoint.buckets), a page other than the stage's virtual page, and a sampler other than the first token's
        (launch.FIRST_TOKEN_SAMPLERS: the graph's greedy selection over the LM head's shards, or the GK tail's draw)."""
        layout = self.model_graph.layout
        problems = []
        if self.graphs != "prefill":
            problems.append(f"a {self.graphs} graph family: the GLM-5.2 stage graph is a PD prefill's")
        point = prefill_point(layout.pp_size)
        if tuple(self.buckets.prefill_token_buckets) != point.buckets:
            problems.append(f"the prefill buckets {self.buckets.prefill_token_buckets}: the stage graphs compile "
                            f"{list(point.buckets)} (--chunked-prefill-size {point.chunk} --cuda-graph-bs-prefill "
                            f"{' '.join(map(str, point.smaller))})")
        if self.block_size != layout.virtual_page:
            problems.append(f"--page-size {self.block_size}: a page is the stage's {layout.cp} ranks x "
                            f"{CP_INTERLEAVE} rows = {layout.virtual_page} tokens")
        if self.sampler_mode not in FIRST_TOKEN_SAMPLERS:
            problems.append(f"SGLANG_NEURON_SAMPLER={self.sampler_mode}: the last stage selects greedily or draws as "
                            f"the GK decode tail draws ({' or '.join(FIRST_TOKEN_SAMPLERS)})")
        if problems:
            raise ValueError("the GLM-5.2 prefill runtime cannot serve:\n  - " + "\n  - ".join(problems))

    def run_prefill(self, fb, md, bucket: Optional[int] = None, pp_proxy=None):
        """The platform's step; under the logit test (models/glm52/logit_probe.py) the last stage's CP rank 0 then
        reads the step's sampled row of logits back and writes it for the request (logit_probe.write_prefill_row),
        which makes the step synchronous."""
        res = super().run_prefill(fb, md, bucket, pp_proxy)
        graph = self.model_graph
        if not graph.probe_logits or int(graph.groups.cp.rank_in_group) != 0:
            return res
        from sglang_neuron.backend import device_io
        from sglang_neuron.models.glm52 import logit_probe

        rids = getattr(fb, "rids", None)
        if not rids or len(rids) != 1:
            raise RuntimeError(f"the logit test's prefill writes one request's row a step, and the step's rids are "
                               f"{rids}")
        out = logit_probe.outputs_of(res)
        pending = getattr(res, "neuron_pending_sample", None)
        logits = out["probe_logits"]
        if pending is not None:
            logits = device_io.readback(logits, pending.launch_event)
        end = int(md.cached_seq_len) + int(md.num_tokens)
        logit_probe.write_prefill_row(logit_probe.test_dir(), rids[0], end, logits[0], int(graph.config.vocab_size))
        return res

    def _check_fleet(self) -> None:
        """Every rank of the world, every stage, agrees on the model's own settings (the recipe, the compiler arguments,
        the toolchain, the kernel's digest, the placement and the split: the host's fleet_digest). The platform's check
        spans the stage's TP group only (its digest names the stage's layers), and stages of another recipe or compiler
        would run collectives the seam pairs against another program."""
        from sglang.srt.distributed.parallel_state import get_world_group

        from sglang_neuron.models.contract import assert_fleet_consistent

        contract = dataclasses.replace(self.contract, parallel_digest=("glm52-prefill-fleet",
                                                                       *self.sgl_model.fleet_digest()))
        assert_fleet_consistent(contract, get_world_group())

    def build_model_inputs(self, md, token_ids: torch.Tensor, bucket: int, temperatures: Optional[torch.Tensor] = None,
                           token_plan=None, seeds: Optional[torch.Tensor] = None,
                           filters: Optional[torch.Tensor] = None, *, context: Optional[int] = None) -> dict:
        """One request's prefill step as the stage graph's kwargs, uploaded in one upload_many; the last stage of a
        topk_topp server adds its three sampling uploads.

        SGLang's slots become the owner ranks' slots and its page ids the per-rank pages, over a table of the step's
        context bucket (the smallest that holds the request after the step; ``context`` at warm-up), -1 past the
        request. The kernel's static kv_ctx_local is the table's width, so a short request does not pay for a long
        context. The base run_prefill adds the landing slots and _run_boundary the send slots."""
        if not md.is_prefill or md.batch_size != 1:
            raise RuntimeError(f"the GLM-5.2 stage graph runs one request's prefill per forward, not a "
                               f"{'prefill' if md.is_prefill else 'decode'} of {md.batch_size} rows")
        layout = self.model_graph.layout
        device = neuron_device()
        n = int(md.num_tokens)
        slots = prefill_inputs.owner_slots(md.slot_mapping[:n], virtual_page=layout.virtual_page,
                                           interleave=CP_INTERLEAVE)
        cached = int(md.cached_seq_len)
        context = self.prefill_context(cached + n) if context is None else int(context)
        host = prefill_inputs.host_inputs(token_ids=token_ids[:n], positions=md.positions[:n], slot_mapping=slots,
                                          pages=md.block_table[0], cached=cached, bucket=bucket,
                                          table_width=context // layout.virtual_page,
                                          virtual_page=layout.virtual_page)
        out = prefill_inputs.step_kwargs(upload_many(host.tensors(), device), self.rank_tensor)
        if layout.samples and self.sampler_mode == "topk_topp":
            # the request's temperature, folded seed and (top_k, top_p, min_p), as the platform stages them for every
            # drawing graph; the draw's position key is the graph's own sampling position
            out.update(self._sampling_inputs(md, n, 1, temperatures, seeds, device, filters))
        return out
