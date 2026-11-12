# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5's giga decode as a megakernel spec: ``QWEN35_GIGA_SPEC`` (``qwen35-giga-decode``).

The decode program is the model's decode graph family: one ``ops.qwen35_giga.qwen35_giga_decode`` call per step
(``qwen35_hybrid_multistep``, one step per launch), run by ``runtime.Qwen35DecodeRuntime`` (the slot, position and
validity tables of ``inputs.build_decode_rows``) with the direct submit. An idle DP rank runs the same graph with every
row padding (its collectives span every rank).

The prefill program is the prefill graph family (``runtime.Qwen35PrefillRuntime``). Each server role holds its own
programs (``phase_programs``, ``ROLE_FAMILIES``): an aggregated server both families, the PD decode role the decode
family with the direct submit, the PD prefill role the prefill family (its idle replicas pad the prefill).

The pool layout ``qwen35_giga`` (``pool_layout``) names what the kernel reads and writes in place, by global layer id:

- ``K`` / ``V``: the 15 full-attention layers' slot KV, one layer-major tensor each (``kv_pool.Qwen35SlotKVPool``, the
  ``mha`` pool class SGLang builds for the hybrid pool's attention layers), kind ``kv``;
- ``state.temporal`` / ``state.conv0``: the 45 GDN layers' recurrent and conv state, one layer-major tensor each
  (``state.Qwen35StatePool``, the state pool class SGLang builds into the request table), kinds ``recurrent_state``
  and ``conv_state``; its slot operations are ``state.GdnStateOps``.

``config_disable_reasons`` refuses the launches the kernels cannot serve (``config.Qwen35Layout.problems``): the
parallel layout (DP attention, EP over every rank, the replica's head split), the KV page (``--page-size``) and the KV
dtype. Two KV layouts are served:

- the slot pool (``--context-length`` at most ``--page-size``): one page per request, prompts of exactly the one
  prefill length (``--chunked-prefill-size``) from position 0, no radix cache (the slot shares nothing);
- the paged pool (``--context-length`` past ``--page-size``, docs/design/recurrent-state.md): a request's pages,
  chunked prefill on the chunk grid from the request's GDN state and cached pages (``prefix``), and the radix cache on
  the chunk-keyed tree (``--radix-cache-backend neuron-chunk``), whose every chunk boundary holds the GDN state the
  forward ending there left in SGLang's tracking slot (``ModelSpec.chunk_state_checkpoints``). Its chunk is whole KV
  pages and whole 512-token prefill blocks.

``request_disable_reasons`` refuses sampling the kernel's greedy argmax does not implement and logprobs.
"""
from __future__ import annotations

import functools
import logging

logger = logging.getLogger(__name__)
# the first plans each rank logs (its cohort view and the programs it runs)
_LOG_FIRST_PLANS = 8

ARCHITECTURE = "Qwen3_5MoeForCausalLM"
NAME = "qwen35-giga-decode"
MODEL_TYPE = "qwen3_5_moe_text"
# rows per replica the decode graph serves: GDN rows per group and attention rows x query heads stay within 128
MAX_ROWS = 64


def pool_layout(dims, layout, spec, ops=None):
    """The ``qwen35_giga`` pool layout (module docstring) of the model ``dims`` in ``layout`` with the GDN state
    ``spec``."""
    from sglang_neuron.megakernel import PoolEntry, PoolLayout

    from .kv_pool import Qwen35SlotKVPool, slot_kv
    from .state import layout_parts

    state_cls, state_entries = layout_parts(spec, ops)
    attn = dims.attention_layers
    return PoolLayout(
        family="qwen35_giga",
        entries=(
            PoolEntry("K", "kv", attn, slot_kv("k"), buffers=1, per_buffer=len(attn)),
            PoolEntry("V", "kv", attn, slot_kv("v"), buffers=1, per_buffer=len(attn)),
            *state_entries,
        ),
        page_size=layout.slot_tokens,
        pool_classes={"mha": Qwen35SlotKVPool},
        state_pool_class=state_cls,
    )


def _arg(server_args, name, default=None):
    return getattr(server_args, name, default)


def paged_launch(server_args) -> bool:
    """The launch serves the paged pool: a context longer than the KV page (module docstring)."""
    context = _arg(server_args, "context_length", None)
    page = int(_arg(server_args, "page_size", 1) or 1)
    return context is not None and int(context) > page


def _served_paged() -> bool:
    """Whether this process serves the paged pool (its launch's server arguments; none published: the slot pool)."""
    try:
        from sglang.srt.runtime_context import get_server_args

        args = get_server_args()
    except Exception:  # noqa: BLE001 -- no runtime context published in this process
        return False
    return args is not None and paged_launch(args)


def config_disable_reasons(server_args, hf_config) -> list:
    """Every reason the resolved server arguments cannot serve the giga decode (module docstring)."""
    from .config import Qwen35Dims, Qwen35Layout

    dims = Qwen35Dims.from_hf(hf_config)
    reasons = list(dims.problems())
    from sglang_neuron import sglang_tree

    world = int(_arg(server_args, "tp_size", 1) or 1)
    # SGLang resolves --dp-size N --enable-dp-attention to --attn-dp-size N (sglang_tree.dp_widths reads either)
    replicas, dp = sglang_tree.dp_widths(server_args)
    if replicas > 1:
        reasons.append(f"{replicas} model replicas (--dp-size without --enable-dp-attention): a replica of the "
                       "layout is an attention DP rank")
    page = int(_arg(server_args, "page_size", 1) or 1)
    paged = paged_launch(server_args)
    context = int(_arg(server_args, "context_length", 0) or 0) if paged else 0
    from .config import KV_CP_ENV, kv_cp_launch

    kv_cp = kv_cp_launch()
    if kv_cp > 1 and str(_arg(server_args, "disaggregation_mode", "null") or "null") != "decode":
        reasons.append(f"{KV_CP_ENV}={kv_cp}: decode context parallelism runs on the PD decode role "
                       "(--disaggregation-mode decode); a prefill or aggregated server holds whole pages")
    from .config import KV_TM_ENV, kv_tm_launch

    if kv_tm_launch():
        # token-major K pages are the kv_cp=1 PD prefill's half of the kv_cp scatter layout (sglang_model refuses
        # the same launches at build): the kv_cp decode's pool is token-major unconditionally, and no other role
        # may run the flag
        role = str(_arg(server_args, "disaggregation_mode", "null") or "null")
        if role != "prefill" or kv_cp != 1:
            reasons.append(f"{KV_TM_ENV}=1: token-major K pages run on the kv_cp=1 PD prefill role only "
                           f"(--disaggregation-mode prefill, no {KV_CP_ENV})")
        elif not context:
            reasons.append(f"{KV_TM_ENV}=1: token-major K pages serve the paged pool's PD scatter; a slot-pool "
                           "prefill (no --context-length) hands whole slots to a dim-major decode")
    layout = Qwen35Layout(world=world, dp=dp, slot_tokens=page, context=context, kv_cp=kv_cp)
    reasons += layout.problems(dims, ep_size=int(_arg(server_args, "ep_size", 1) or 1))
    radix = not _arg(server_args, "disable_radix_cache", False)
    if not paged:
        if radix:
            reasons.append("the radix cache needs the paged KV pool (--context-length past --page-size): the KV slot "
                           "page shares no prefix; launch with --disable-radix-cache or a paged pool")
    else:
        from sglang_neuron.sglang_patches import chunk_keyed_tree

        chunk = int(_arg(server_args, "chunked_prefill_size", 0) or 0)
        if chunk <= 0:
            reasons.append("the paged pool prefills on the chunk grid: launch with --chunked-prefill-size")
        else:
            from .prefix import prefill_buckets

            # SGLang has resolved the flag to the chunk of one attention-DP replica by now
            try:
                prefill_buckets(chunk, page)
            except ValueError as error:
                reasons.append(str(error))
            if chunk % page:
                reasons.append(f"the prefill chunk of {chunk} tokens per DP replica (--chunked-prefill-size over the "
                               f"DP size) is not whole {page}-token KV pages")
        if radix and not chunk_keyed_tree.selected(server_args):
            reasons.append(f"the radix cache keeps exact GDN checkpoints only on the chunk-keyed tree, every boundary "
                           f"a forward's end: launch with --radix-cache-backend {chunk_keyed_tree.BACKEND}")
    kv_dtype = str(_arg(server_args, "kv_cache_dtype", "auto") or "auto")
    if kv_dtype not in ("auto", "bfloat16", "bf16"):
        reasons.append(f"--kv-cache-dtype {kv_dtype}: the giga attention reads a bfloat16 KV cache")
    ssm = _arg(server_args, "mamba_ssm_dtype", None)
    if ssm not in (None, "float32"):
        reasons.append(f"--mamba-ssm-dtype {ssm}: the giga GDN keeps its recurrent state in float32")
    return reasons


def request_disable_reasons(tokenized) -> list:
    """The kernel samples greedily on the device and returns no logprobs."""
    reasons = []
    params = getattr(tokenized, "sampling_params", None)
    temperature = getattr(params, "temperature", 0.0)
    top_k = getattr(params, "top_k", 1)
    if temperature not in (0, 0.0) and top_k != 1:
        reasons.append("the Qwen3.5 giga decode samples greedily (temperature 0 or top_k 1)")
    if getattr(tokenized, "return_logprob", False):
        reasons.append("the Qwen3.5 giga decode returns no logprobs")
    ids = getattr(tokenized, "input_ids", None)
    if ids is not None and not _served_paged():
        # the one prefill bucket of a ladder-less launch is the chunk (--chunked-prefill-size)
        from sglang.srt.runtime_context import get_schedule

        length = get_schedule().chunked_prefill_size
        if length and length > 0 and len(ids) != int(length):
            reasons.append(f"the Qwen3.5 giga prefill runs prompts of exactly {int(length)} tokens "
                           f"(--chunked-prefill-size); this prompt has {len(ids)}")
    return reasons


# the graph families each server role builds and runs: an aggregated server both, a PD role its own phase only
ROLE_FAMILIES = {"all": ("decode", "prefill"), "decode": ("decode",), "prefill": ("prefill",)}
_GRAPH_MARKS = {"decode": "qwen35_giga_decode", "prefill": "qwen35_giga_prefill"}


def model_disable_reasons(model, hf_config) -> list:
    """The model must have built exactly its role's graph families (``ROLE_FAMILIES``), each the giga kernels' graph."""
    role = getattr(model, "phase", "all")
    families = ROLE_FAMILIES.get(role)
    if families is None:
        return [f"the Qwen3.5 giga host serves an aggregated server or a PD role, not {role!r}"]
    graphs = getattr(model, "model_graphs", None) or {}
    reasons = [f"the model's {f} graph is not the Qwen3.5 giga {f}" for f in families
               if not getattr(graphs.get(f), _GRAPH_MARKS[f], False)]
    extra = sorted(set(graphs) - set(families))
    if extra:
        reasons.append(f"the {role} role builds the {list(families)} graph families, and the model built {extra} too")
    return reasons


def phase_programs(role: str) -> list:
    """The phase programs of a server role: the decode family (decode and idle steps, direct submit) on an aggregated
    server and the PD decode, the prefill family (prefill steps; idle ones too on the PD prefill, where an idle replica
    pads the prefill every other replica runs) on an aggregated server and the PD prefill."""
    from sglang_neuron.megakernel import PhaseProgram

    from .runtime import Qwen35DecodeRuntime, Qwen35PrefillRuntime

    if role not in ROLE_FAMILIES:
        raise ValueError(f"the Qwen3.5 giga host serves an aggregated server or a PD role, not {role!r}")
    families, programs = ROLE_FAMILIES[role], []
    if "decode" in families:
        programs += [
            PhaseProgram("decode", "decode", Qwen35DecodeRuntime, direct_submit=True, samples_in_forward=True),
            PhaseProgram("idle", "decode", Qwen35DecodeRuntime, direct_submit=True, samples_in_forward=True),
        ]
    if "prefill" in families:
        programs.append(PhaseProgram("prefill", "prefill", Qwen35PrefillRuntime, samples_in_forward=True))
    if role == "prefill":
        programs.append(PhaseProgram("idle", "prefill", Qwen35PrefillRuntime, samples_in_forward=True))
    return programs


def build(model, hf_config, context):
    layout = pool_layout(model.qwen35_dims, model.qwen35_layout, model.state_spec, model.state_ops)
    programs = phase_programs(getattr(model, "phase", "all"))
    replica = model.qwen35_layout.replica
    widths = [w for w in context.widths if w <= MAX_ROWS and w % replica == 0]
    if not widths:
        raise ValueError(f"{NAME}: none of the decode widths {list(context.widths)} is a multiple of the "
                         f"{replica}-rank replica within {MAX_ROWS} rows")
    return qwen35_built_class()(name=NAME, model=model, programs=programs, pool_layout=layout, widths=widths)


@functools.lru_cache(maxsize=None)
def qwen35_built_class():
    """``Qwen35BuiltMegakernel``: the Neuron built megakernel with the cohort step plan of an aggregated DP server.

    Every rank of the world runs the same NEFFs in the same order (the MoE spans every replica), while SGLang's DP
    attention lets each replica prefill or decode on its own. A step's plan therefore comes from the DP-attention
    cohort snapshot (``mlp_sync_batch_metadata``, every participant's forward mode and request counts): when any replica
    prefills, every rank runs the prefill program (an idle or decoding replica a padding prompt), and then, when any
    replica decodes, every rank runs the decode program at the bucket of the largest decoding replica (the others
    padding). A rank returns its own step's output.

    A PD role holds one program family: the decode role's cohort only decodes or idles, the prefill role's only
    prefills or idles (an idle replica pads the prefill). A step whose cohort needs the other role's program is refused
    with the reason (its NEFF is not on this server)."""
    from sglang_neuron.megakernel import NeuronBuiltMegakernel
    from sglang_neuron.megakernel.built import step_class

    class Qwen35BuiltMegakernel(NeuronBuiltMegakernel):
        mlp_sync_cohort_metadata = True

        def plan(self, input_ids, forward_batch, *args, **kwargs):
            from sglang.srt.model_executor.forward_batch_info import ForwardMode

            from sglang_neuron.model_executor.graph_runner import StepPlan

            if self._runner is None:
                raise RuntimeError(f"{self.name} plans a step before it was prepared")
            md = self._attn.build_metadata(forward_batch)
            decode_rt, prefill_rt = self._runtimes.get("decode"), self._runtimes.get("prefill")
            if decode_rt is not None:
                decode_rt._q35_step_batch = forward_batch
            planner = decode_rt if decode_rt is not None else prefill_rt
            mode = forward_batch.forward_mode
            local = "prefill" if mode.is_extend() else "decode" if mode.is_decode() and md.batch_size else "idle"
            meta = getattr(forward_batch, "mlp_sync_batch_metadata", None)
            if meta is not None:
                modes = [ForwardMode(m) for m in meta.forward_modes]
                prefill = any(m.is_extend() for m in modes)
                decoding = [int(n or 0) for m, n in zip(modes, meta.num_decode_requests) if m.is_decode()]
            elif planner._q35_layout.dp > 1:
                # without the cohort every replica would plan only its own step, and the world's programs diverge
                from sglang_neuron.sglang_patches import mlp_sync_cohort_snapshot

                raise RuntimeError(f"{self.name}: a {mode.name} step without the DP-attention cohort snapshot "
                                   f"(mlp_sync_batch_metadata; the snapshot is "
                                   f"{'on' if mlp_sync_cohort_snapshot.enabled() else 'off'} in this process)")
            else:
                prefill = local == "prefill"
                decoding = [md.batch_size] if local == "decode" else []
            self._q35_plans = getattr(self, "_q35_plans", 0) + 1
            if self._q35_plans <= _LOG_FIRST_PLANS:
                logger.info("[qwen35] plan %d: local %s (%d rows), cohort %s, prefill %s, decoding %s",
                            self._q35_plans, local, md.batch_size,
                            None if meta is None else [ForwardMode(m).name for m in meta.forward_modes], prefill,
                            decoding)
            if prefill and prefill_rt is None:
                raise RuntimeError(f"{self.name}: a replica prefills in this step, and this server is the PD decode "
                                   "role, which holds no prefill program (the PD prefill role runs it)")
            if decoding and decode_rt is None:
                raise RuntimeError(f"{self.name}: a replica decodes in this step, and this server is the PD prefill "
                                   "role, which holds no decode program (the PD decode role runs it)")
            if local == "prefill" and md.batch_size != 1:
                raise RuntimeError(f"{self.name}: one prompt per replica per prefill step; this step has "
                                   f"{md.batch_size}")
            seq_len = None
            if prefill:
                seq_len = prefill_rt.seq_len
                if getattr(prefill_rt._q35_layout, "paged", False):
                    seq_len = _cohort_program(prefill_rt, meta, md, local)
            bucket = decode_rt.buckets.seqs_bucket(max(max(decoding), 1)) if decoding else None
            ctx = None
            if decode_rt is not None:
                ctx = decode_rt.buckets.decode_contexts[-1]
                if decoding and getattr(decode_rt._q35_layout, "paged", False):
                    ctx = _cohort_context(decode_rt, meta, md, local)
            phase = "prefill" if local == "prefill" or decode_rt is None else "decode"
            prefill_bucket = getattr(seq_len, "bucket", seq_len)
            plan = StepPlan(local, phase, prefill_bucket if local == "prefill" else (bucket or 0), ctx)
            plan.q35 = {"prefill": seq_len, "decode": bucket}
            step_mode = local if local != "decode" or bucket else "idle"
            return step_class()(mode=step_mode, rows=int(plan.bucket), phase=plan.phase, plan=plan, metadata=md)

        def forward(self, step, original, input_ids, positions, forward_batch, *args, **kwargs):
            from sglang_neuron.model_executor.graph_runner import StepPlan

            if self._pools is not None:
                self._pools.check(self._runner.token_to_kv_pool, self._runner.req_to_token_pool)
            plan = step.plan
            out = None
            if plan.q35["prefill"]:
                runtime = self._runtimes["prefill"]
                program = plan.q35["prefill"]
                if step.mode == "prefill":
                    out = runtime.run_prefill(forward_batch, step.metadata, program)
                else:
                    runtime.run_idle(StepPlan("idle", "prefill", getattr(program, "bucket", program),
                                              getattr(program, "tier", None)))
            if plan.q35["decode"]:
                runtime = self._runtimes["decode"]
                if step.mode == "decode":
                    out = runtime.run_decode(forward_batch, step.metadata, plan.q35["decode"], plan.context)
                else:
                    runtime.run_idle(StepPlan("idle", "decode", plan.q35["decode"], plan.context))
            if out is None:
                out = (self._runtimes.get("decode") or self._runtimes["prefill"]).empty_output()
            return out

    Qwen35BuiltMegakernel.__module__ = __name__
    return Qwen35BuiltMegakernel


def _cohort_context(decode_rt, meta, md, local: str) -> int:
    """The decode context rung every rank runs this step (paged pool): the smallest that holds the longest sequence of
    every decoding replica (the snapshot's ``max_prefix_tokens`` of a decode participant: its longest sequence plus
    the token it writes), or this replica's own without a cohort."""
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    if meta is None:
        lens = getattr(md, "seq_lens", None)
        need = int(lens.max()) + 1 if lens is not None and len(lens) else 1
        return decode_rt.context_for(need)
    prefixes = getattr(meta, "max_prefix_tokens", ())
    modes = [ForwardMode(m) for m in meta.forward_modes]
    if len(prefixes) != len(modes):
        return decode_rt.buckets.decode_contexts[-1]
    return decode_rt.context_for(max([int(p) for m, p in zip(modes, prefixes) if m.is_decode()] + [1]))


def _cohort_program(prefill_rt, meta, md, local: str):
    """The paged prefill program every rank runs this step: the largest chunk and the longest cached prefix of the
    prefilling replicas (the DP-attention snapshot's ``num_tokens`` and ``max_prefix_tokens``; this replica's own
    metadata without a cohort)."""
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    if meta is None:
        return prefill_rt.program([md.num_tokens], [int(getattr(md, "cached_seq_len", 0) or 0)])
    modes = [ForwardMode(m) for m in meta.forward_modes]
    prefixes = getattr(meta, "max_prefix_tokens", ())
    if len(prefixes) != len(modes):
        raise RuntimeError("the DP-attention cohort snapshot carries no cached prefix lengths (max_prefix_tokens): "
                           "the cohort cannot agree on a paged prefill's prior tier")
    tokens = [int(n) for m, n in zip(modes, meta.num_tokens) if m.is_extend()]
    prior = [int(p) for m, p in zip(modes, prefixes) if m.is_extend()]
    return prefill_rt.program(tokens, prior)


@functools.lru_cache(maxsize=None)
def spec():
    from sglang_neuron.megakernel import upstream
    from sglang_neuron.sglang_patches.megakernel_platform import PlatformMegakernelSpec

    _, Requirements, Numerics, NumericsKind, Feature, _, _ = upstream.contract()
    return PlatformMegakernelSpec(
        name=NAME,
        requires=Requirements(
            isas=frozenset({"trn3"}),
            tp_sizes=frozenset(),
            decode_graph_backends=frozenset(),
            dtypes=frozenset({"bfloat16"}),
            hf_config={"model_type": MODEL_TYPE},
        ),
        # the platform turns torch.compile on for every launch: SGLang builds an out-of-tree platform's graph runner
        # only then (platform.py), and the giga decode runs inside that runner
        # KV_OFFLOAD: HiCache L2 over the chunk-keyed tree (the paged KV pool's and the state pool's Neuron host pools;
        # a host hit's load completes before the prefill graph reads the pools)
        features=frozenset({Feature.DP_ATTENTION, Feature.EXPERT_PARALLEL, Feature.PD_DISAGGREGATION,
                            Feature.TORCH_COMPILE, Feature.KV_OFFLOAD}),
        # the kernels' MXFP8 experts round differently from the BF16 reference: first tokens against HF are the bound
        numerics=Numerics(kind=NumericsKind.TOLERANCE, tolerances={"first_token_match_min": 0.9}),
        max_rows=MAX_ROWS,
        build=build,
        api_version=upstream.API_VERSION,
        model_disable_reasons=model_disable_reasons,
        config_disable_reasons=config_disable_reasons,
        request_disable_reasons=request_disable_reasons,
    )


def __getattr__(name):
    if name == "QWEN35_GIGA_SPEC":
        return spec()
    raise AttributeError(name)
