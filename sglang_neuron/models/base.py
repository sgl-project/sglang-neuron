# SPDX-License-Identifier: Apache-2.0
"""The SGLang serving contract every Neuron model implements.

A model owns its compiled graphs, its weight loading and its KV specification; scheduling, sampling results and
disaggregation stay SGLang's.
"""
from __future__ import annotations

import torch.nn as nn


def segmented_prefill_cache_policy(phase: str) -> None:
    """Keep a prefilling role's forwards on the prefill chunk grid.

    The runtime's prefill runs the segmented kernel on compiled tiles of one chunk, so SGLang truncates every forward,
    a new request's first and each continuation, to a whole number of --chunked-prefill-size tokens unless it ends the
    prompt (sglang_patches.prefill_chunk_grid). On SGLang's default radix tree a hit at a whole KV page m then runs the
    forwards [m, m + C), [m + C, ...); the segmented kernel takes any block-multiple prior, so the hit's logits are
    close to a cold prefill's, not bit-identical. With --radix-cache-backend neuron-chunk the tree keys whole chunks
    (sglang_patches.chunk_keyed_tree), so a hit replays the cold forwards from a chunk boundary, bit-identical.

    Set in the scheduler process, which builds its tree and its truncation grid after the model. Streaming sessions are
    refused: a streaming turn resumes at the previous turn's whole KV, which is not on the kernel's block grid. A decode
    role never prefills and returns here."""
    if phase not in ("prefill", "all"):
        return
    from sglang.srt.runtime_context import get_memory, get_schedule, get_serving

    if get_serving().enable_streaming_session:
        raise ValueError(
            "--enable-streaming-session is not supported on a prefilling role: a streaming turn resumes at the "
            "previous turn's whole KV, which is not on the segmented kernel's block grid; launch without "
            "--enable-streaming-session")
    # without a tree or chunked prefill nothing is ever truncated or shared
    if not get_memory().disable_radix_cache or int(get_schedule().chunked_prefill_size or 0) > 0:
        from sglang_neuron.sglang_patches import prefill_chunk_grid

        prefill_chunk_grid.keep()


class NeuronModel(nn.Module):
    """A Neuron model: one compiled graph (``model_graph``, with ``execution_model`` its sampled execution), or
    one per phase on an aggregated server (``model_graphs`` and ``execution_models``), and its ``kv_spec``."""

    # The runtime the platform builds for each of the model's graphs (runtime_class(model_runner, model, phase=phase)),
    # a NeuronModelRuntime subclass whose steps need more than the platform runtime's step path (it keeps the generic
    # compilation, dispatch and async machinery and overrides the runtime's extension points;
    # graph_runner.runtime_class_of refuses any other class at load). None: the platform runtime. A megakernel's phase
    # program may name its own (megakernel.PhaseProgram.runtime_class).
    runtime_class = None

    # Pipeline parallelism (distributed/pipeline.py), only for models with ModelSpec.pipeline_stages. A stage sets its
    # global layers [start_layer, end_layer) (SGLang sizes the stage's KV pool from them), is_first_stage and
    # is_last_stage. A non-last stage's prefill graph writes the pp_prefill_boundary tensors, SP-sharded
    # [tokens / attention TP, pp_hidden_size] of pp_boundary_dtype, in place into the ``<name>_out`` slots it is
    # handed; the next stage takes them as inputs of the same names.
    pp_prefill_boundary = ("pipeline_hidden_states", "pipeline_residual")
    # A stage whose prefill boundary tensors differ in width, dtype or rows declares them here instead, as
    # {name: distributed.pp_seam.BoundaryKind} in the order its graph takes them (they replace pp_prefill_boundary). A
    # tensor of ``received`` row blocks is the input ``name`` and one of ``sent`` blocks the ``<name>_out`` slot, each
    # block [tokens / attention TP (SP-sharded) or tokens, hidden]; every stage declares the same ``blocks`` per kind.
    pp_prefill_boundary_kinds = None
    # the decode boundary, TP-replicated [batch rows, pp_hidden_size], passed the same way
    pp_decode_boundary = ("pipeline_hidden_states",)
    pp_boundary_dtype = None
    pp_hidden_size = None
    start_layer = 0
    end_layer = None
    is_first_stage = True
    is_last_stage = True

    def pp_boundary_spec(self, name: str) -> tuple:
        """The ``(width, dtype)`` of the boundary tensor ``name``'s rows: ``(pp_hidden_size, pp_boundary_dtype)``, a
        hidden-state row. A model whose boundary also carries another tensor (an indexer's top-k picks) names its
        own."""
        return int(self.pp_hidden_size), self.pp_boundary_dtype

    # Token context parallelism of the KV (model_executor/kv_layout.KVLayout.cp_size, which the model declares to the
    # pool while it is built): a page of P tokens is one P / kv_cp_size-token block on each attention-TP rank, rank r
    # holding the page's tokens [r * P / kv_cp_size, (r + 1) * P / kv_cp_size), every KV head on every rank. The
    # platform runtime addresses it on the PD prefill role only, over a pool that publishes the same layout, and the
    # model's prefill attention must read and write only its rank's share of each page (NeuronModelRuntime.__init__).
    # No model in the tree declares it on the platform runtime.
    kv_cp_size = 1

    # A model that builds its own KV pools defines configure_kv_cache(configurator, *, pre_model_load_memory) returning
    # SGLang's KVCacheConfigResult in place of KVCacheConfigurator.configure's (general_plugin._model_kv_pools).

    @property
    def config(self):
        """The checkpoint's HF config, under the name SGLang's in-tree models give it: SGLang's loader reads
        ``model.config`` under --enable-nvidia-megakernel (kernel_plugins.amd.binding.monokernel_weights_iterator reads
        its architectures and model_type). Every Neuron model stores it as ``hf_config``; this is a read-only view."""
        return self.hf_config

    # The recurrent-state pool class of a model with state layers (KDA, GDN, Mamba-2): a MambaPool subclass, normally
    # mem_cache.state_pool.NeuronStatePool.bind(spec, ops), set before SGLang builds the pools. SGLang's request tables
    # then build their state pool from it (sglang_patches.state_pool_class; a megakernel model declares it in its
    # PoolLayout instead); None keeps SGLang's MambaPool.
    state_pool_class = None

    def neuron_after_warmup(self, family: str, warmed_keys: dict) -> None:
        """Called on every rank of a device launch once a graph family (``"prefill"`` or ``"decode"``) has warmed and
        the ranks agreed on its programs, with the NEFF cache keys of every graph warmed so far (graph -> keys, as the
        SPMD check compares them); an aggregated server calls it once per phase. A model checks here what its programs
        compiled to; a raise fails the launch."""

    def fleet_digest(self) -> tuple:
        """Launch settings the model's traced programs depend on beyond the platform's own (say a graph variant picked
        from the environment), which every rank must resolve the same. Appended to the contract digest the ranks
        compare before compiling (models.contract.assert_fleet_consistent), so a mismatch fails there and not in a
        collective. JSON values only, no rank or device ids."""
        return ()

    def megakernel_spec(self):
        """The ``MegakernelSpec`` this model serves its steps through in this configuration, or None (the model's own
        host: the platform runtime per graph family). SGLang installs the spec registered for the model's architecture
        on the device target under --enable-nvidia-megakernel (sglang_neuron/megakernel/registry.py); its built
        object then plans and runs every step. A model that declares one refuses to serve without it, or with
        another (megakernel.dispatch.check_declared). Qwen3.5 declares its giga-kernel decode and prefill."""
        return None

    def bind_kv_cache(self, pool):
        # pools address layers by global id (the Neuron pool from its own start_layer, SGLang's hybrid SWA pool through
        # its layer mapping), and the model's KV layers are its stage's global layers from start_layer
        caches = {}
        for index, layer in enumerate(self.kv_spec.layers):
            k, v = pool.get_kv_buffer(self.start_layer + index)
            caches[layer.name] = [k, v]
        # an aggregated server built one graph per phase, and every graph reads and writes the same pool buffers
        for graph in getattr(self, "model_graphs", {"_": self.model_graph}).values():
            graph.bind_kv_cache(caches)
        # executors bound to the previous KV tensors prepare their inputs again at their next calls
        from sglang_neuron.backend import executor_cache

        executor_cache.invalidate()

    def _ensure_runtime(self, model_runner):
        if self.runtime is None:
            from sglang_neuron.megakernel import dispatch_of
            from sglang_neuron.megakernel.dispatch import check_declared

            check_declared(self)
            dispatch = dispatch_of(self)
            if dispatch is not None:
                # the built megakernel binds its pool layout, then builds its families' runtimes here (SGLang's
                # EagerRunner may have prepared it already; a second prepare is a no-op)
                dispatch.prepare(model_runner)
            else:
                self._build_runtimes(model_runner)
        return self.runtime

    def _build_runtimes(self, model_runner, runtime_classes=None, direct_submit=frozenset()) -> dict:
        """Build, compile and warm one runtime per graph family and return them by family name. ``runtime_classes``
        (a megakernel's phase programs) names a family's runtime class (None: the model's runtime_class);
        ``direct_submit`` the families whose decode steps go to the bound executor (backend.direct_submit)."""
        from sglang_neuron.model_executor.graph_runner import runtime_class_of

        graphs = getattr(self, "model_graphs", None)
        self.bind_kv_cache(model_runner.token_to_kv_pool)
        from sglang_neuron.mem_cache.mla_pool import attach_row_zeroing

        attach_row_zeroing(model_runner.token_to_kv_pool, getattr(model_runner, "req_to_token_pool", None))
        # An aggregated server builds one runtime per phase graph, dispatched by the step plan (a single graph builds
        # one, phase None). The warm-ups run one after another; their barriers span the engine group, and every rank
        # builds in the same order.
        aggregated = graphs is not None and len(graphs) > 1
        families = sorted(graphs) if graphs else [None]
        runtimes, by_family = {}, {}
        for family, phase in zip(families, sorted(graphs) if aggregated else (None,)):
            runtime = runtime_class_of(self, (runtime_classes or {}).get(family))(model_runner, self, phase=phase)
            if family in direct_submit:
                runtime.megakernel_direct_submit = True
            runtime.build_compiled()
            runtime.warmup_all()
            runtimes[phase] = by_family[family] = runtime
        if aggregated:
            self.phase_runtimes = runtimes
        self.runtime = next(iter(runtimes.values()))
        return by_family

    def _runtime_for(self, phase: str):
        """The runtime of a graph family. Callers pass the step plan's ``phase``, not its kind: an idle DP rank in a
        prefill step must dispatch the prefill graph whose collectives the busy ranks are running."""
        runtimes = getattr(self, "phase_runtimes", None)
        if runtimes is None:
            return self.runtime
        return runtimes[phase]

    def forward(self, input_ids, positions, forward_batch, input_embeds=None,
                pp_proxy_tensors=None, get_embedding=False):
        # text only: generating from token ids while silently dropping these would return fluent garbage
        if input_embeds is not None:
            raise ValueError("this Neuron model serves text token ids only; input_embeds are not consumed")
        if get_embedding:
            raise ValueError("embedding extraction is not supported by this Neuron model")
        from sglang.srt.model_executor.forward_context import get_forward_context

        backend = get_forward_context().attn_backend
        # a hybrid linear-attention model's backend is SGLang's wrapper over the platform's (full_attn_backend), which
        # builds the step metadata
        backend = getattr(backend, "full_attn_backend", backend)
        runtime = self._ensure_runtime(backend.model_runner)
        from sglang_neuron.megakernel import dispatch_of

        dispatch = dispatch_of(self)
        if dispatch is not None:
            # a megakernel plans and runs every step; a step it does not cover is refused there
            return dispatch(input_ids, positions, forward_batch, pp_proxy_tensors=pp_proxy_tensors)
        metadata = backend.build_metadata(forward_batch)
        plan = runtime.plan_step(forward_batch, metadata)
        runtime = self._runtime_for(plan.phase)
        if plan.kind == "idle" or metadata.batch_size == 0:
            if pp_proxy_tensors is not None:
                # a pipeline stage runs no idle step: a boundary always belongs to a batch
                raise RuntimeError("a pipeline boundary reached an idle step, which no stage runs")
            return runtime.run_idle(plan)
        # every runtime's step entry points take pp_proxy (graph_runner.runtime_class_of refuses one that drops it)
        if plan.kind == "decode":
            return runtime.run_decode(forward_batch, metadata, plan.bucket, plan.context, pp_proxy=pp_proxy_tensors)
        return runtime.run_prefill(forward_batch, metadata, plan.bucket, pp_proxy=pp_proxy_tensors)
