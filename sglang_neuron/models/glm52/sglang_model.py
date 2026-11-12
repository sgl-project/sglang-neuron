# SPDX-License-Identifier: Apache-2.0
"""GlmMoeDsaForCausalLM: the host SGLang builds for a GLM-5.2 checkpoint (models/host.py), serving either PD role,
each on its own giga kernel (sglang_neuron/models/glm52/kernels):

- the PD prefill: one stage of the prefill giga kernel's pipeline at LNC1, at either prefill point
  (stage_layout.PREFILL_POINTS: PP4 x TP32 with EP32 and SGLANG_PP_LAYER_PARTITION=18,20,20,20, or PP16 x TP8 with
  EP8). The host wires the stage's parts to the platform: the layout (stage_layout) of the stage SGLang assigned, its
  weight tree (stage_weights) and stage graph (prefill.stage) over the plugin's groups, the stage's KV pool (kv_pool:
  configure_kv_cache builds it, bind_kv_cache binds its caches into the graph), the runtime (prefill.runtime) and the
  model's launch facts (launch: the config check and the fleet digest; compile_args: the compiler arguments).
- the PD decode: the decode giga kernel (GK) on one trn3 node, ``--tp-size 128`` at LNC1 with no DP attention (the
  kernel runs its own DP2 x CP4 x token16 attention and EP128 MoE inside), at the launch's decode point
  (points.py, SGLANG_NEURON_GLM52_POINT) with its 32 sequence slots, with or without the async MTP draft
  (--speculative-algorithm NEURON_MTP), sampling greedily or with each request's top_k / top_p / temperature: its graph
  (decode.graph, mtp.graph), runtime (decode.runtime) and the SGLang pools that only count its tokens
  (decode.pools). Every other decode launch is refused, naming what differs.

The two roles share the checkpoint and nothing else: each has its own graph, runtime, pools and compiler arguments. So
the host serves the role of its server's --disaggregation-mode (models.phase.model_phase) through every hook, and
refuses an aggregated server, which would build both. The launch rules dispatch on the same role (launch.py), and each
role's kernel build knobs are its launch settings (models/catalog.py).

The two roles pair over NIXL. The decode's KV is the GK's own caches, and the GLM PD KV hand-off
(models/glm52/pd/handoff.py, pd/placement.py) writes the prefill's KV into them. The decode also runs on
SGLang's fake KV transfer, measured alone (--disaggregation-transfer-backend fake), and the launch refuses every other
backend (launch._decode_problems).
"""

from __future__ import annotations

import dataclasses
import logging

import torch

from sglang_neuron.models.glm52 import launch
from sglang_neuron.models.glm52.decode.runtime import GlmGkDecodeRuntime
from sglang_neuron.models.glm52.options import Glm52PrefillOptions
from sglang_neuron.models.glm52.prefill.runtime import Glm52PrefillRuntime
from sglang_neuron.models.glm52.prefill.stage_layout import prefill_point, stage_layout
from sglang_neuron.models.host import ModelHost
from sglang_neuron.models.phase import model_phase

logger = logging.getLogger(__name__)

# the GK step's compiler arguments, the ones its accuracy and performance were measured with:
# -O1 with neuronx-cc's BIR verifier on,
# which the served decode GK passes at 128k and 1M and which changes none of its instruction bytes
GK_COMPILER_ARGS = [
    "--auto-cast=none",
    "-O1",
    "--internal-hlo2tensorizer-options=--modular-flow-mac-threshold=10",
]


def geometry(config):
    """The decode kernel's geometry for this checkpoint: the launch's point's (models/glm52/points.py,
    SGLANG_NEURON_GLM52_POINT, with its indexer pool), refused where the checkpoint's shapes or constants differ from
    the ones the kernel is traced with, or the geometry breaks one of the kernel's rules
    (layout.config.check_checkpoint). The kernel takes its constants (the routed scaling factor, the RoPE base, the
    indexer schedule, the router) from the geometry while the PD prefill reads them from the checkpoint, so a
    checkpoint that differs in one would decode finite, wrong tokens."""
    from sglang_neuron.models.glm52 import points
    from sglang_neuron.models.glm52.layout.config import check_checkpoint

    geo = points.current().decode_geometry()
    geo.validate()
    check_checkpoint(geo, config)
    return geo


def served_context(geo) -> int:
    """The longest sequence the kernel decodes exactly: its context less one pool block
    (layout.config.max_served_model_len). Its causal keep test reads the static position ``prior_len``, so a step from
    ``prior_len + 2`` on drops its newest keys and decodes wrong tokens without an error."""
    from sglang_neuron.models.glm52.layout.config import max_served_model_len

    return max_served_model_len(geo)


def context_problem(geo, context: int) -> str | None:
    """Why a launch whose sequences reach ``context`` tokens is not served, or None."""
    served = served_context(geo)
    if context <= served:
        return None
    return (f"a context of {context} (the kernel decodes {served} positions a slot exactly, its {geo.context} less a "
            f"pool block, and past them drops its newest keys; launch with --context-length {served})")


def decode_problems(config) -> list:
    """What of a decode launch the GK does not serve, one problem each ([] when it serves it), read where its values
    have resolved: the serving policy's layout, SGLang's parallel config, the checkpoint config with SGLang's context,
    and SGLang's decode request rows.

    Raises:
        ValueError: The checkpoint differs from the geometry the kernel is traced with, or the geometry breaks one of
            the kernel's rules (``geometry``): every difference, in one message.
    """
    from sglang.srt.runtime_context import get_disagg

    from sglang_neuron import serving_policy, sglang_tree
    from sglang_neuron.models.glm52.decode import pools as decode_pools

    policy = serving_policy.current()
    pc = sglang_tree.parallel_config()
    problems = []
    if not policy.giga_decode:
        problems.append("a layout other than the GK's (--tp-size 128, no DP attention)")
    if int(pc.pp_size) > 1 or int(pc.attn_dcp_size) > 1:
        problems.append("pipeline or decode context parallelism (the GK runs every layer and its own CP inside)")
    geo = geometry(config)
    # SGLang's resolved context (ModelConfig sets hf_config.context_len), the length every slot may reach
    too_long = context_problem(geo, int(config.context_len))
    if too_long:
        problems.append(too_long)
    extra_rows = decode_pools.extra_rows_problem(get_disagg().disaggregation_decode_extra_slots)
    if extra_rows:
        problems.append(extra_rows)
    return problems


class GlmMoeDsaForCausalLM(ModelHost):
    """One host for both PD roles (the module docstring): every hook is its server's role's, read as ``self.phase``
    (ModelHost's, set before the graph is built) and, in ``check_launch``, which runs before it, from model_phase."""

    name = "GLM-5.2"
    # neither role's attention is the platform's segmented kernel, whose KV tiles the prefill buckets would be
    segmented_prefill = False
    # a prefill stage hands the next one each rank's [tokens / 32, hidden] bf16 (h, r) shard (the kernel's residual
    # stream); the decode runs no pipeline
    pp_boundary_dtype = torch.bfloat16
    pp_hidden_size = 6144
    # at PP16 the stages also hand each other each rank's [tokens / 8, index_topk] int32 top-k picks: the stages that
    # start on a shared indexer layer take them (stage_layout picks_in), every stage carries them so all graphs and
    # seams are one shape (a stage that hands no picks on writes zeros); the checkpoint's index_topk must be this width
    # (_build_prefill_stage)
    pp_picks_width = 2048

    @property
    def pp_prefill_boundary(self) -> tuple:
        """The stage boundary's tensors: (h, r), and the top-k picks at a point whose boundary carries them
        (launch.boundary_carries_picks); a decode, which runs no pipeline, has the (h, r) names alone."""
        base = ("pipeline_hidden_states", "pipeline_residual")
        if self.stage is not None and launch.boundary_carries_picks(self.stage.pp_size):
            return base + ("pipeline_picks",)
        return base

    def pp_boundary_spec(self, name: str) -> tuple:
        if name == "pipeline_picks":
            return self.pp_picks_width, torch.int32
        return self.pp_hidden_size, self.pp_boundary_dtype

    @property
    def prefill_ladder(self) -> bool:
        """The prefill compiles the smaller steps of a prompt's last piece beside its chunk (PrefillPoint.buckets:
        1024 and 2048 beside 4096 at PP4, 1024 beside 2048 at PP16); the decode is one fixed shape, the kernel's slots
        at its context."""
        return self.phase == "prefill"

    @property
    def runtime_class(self):
        """The role's runtime: the platform's with the prefill stage graph's inputs, or with the GK's step tensors."""
        return GlmGkDecodeRuntime if self.phase == "decode" else Glm52PrefillRuntime

    def check_launch(self, config) -> None:
        """Refuse what the role's graphs do not serve, before any group is built: the checkpoint config facts the
        prefill stages of the launch's point need (launch.check_hf_config at SGLang's --pp-size, which reads the config
        alone and refuses a size no point serves), what of the decode launch the GK does not serve (decode_problems),
        and an aggregated server, which neither role's graphs serve."""
        role = model_phase()
        if role == "prefill":
            from sglang_neuron import sglang_tree

            launch.check_hf_config(config, int(sglang_tree.parallel_config().pp_size))
            return
        if role != "decode":
            raise ValueError("the GLM-5.2-family graphs serve the PD prefill and decode roles, not an aggregated "
                             "server: launch with --disaggregation-mode prefill or decode")
        problems = decode_problems(config)
        if problems:
            raise ValueError("GLM-5.2-family GK decode: not served with " + "; ".join(problems))

    def build_graph(self, phase: str, config):
        """The role's graph on the host: the decode's GK graph of this rank, or the prefill's stage graph, its weights
        on the meta device until load.

        Raises:
            ValueError: For the prefill, SGLang's split disagreeing with the stage layout's, a hidden size other than
                the boundary's, an index_topk other than the picks a PP16 boundary carries, or a stage group of
                another size.
        """
        if phase == "decode":
            from sglang.srt.runtime_context import get_spec

            from sglang_neuron.models.glm52.decode.graph import GlmGkDecodeGraph
            from sglang_neuron.models.glm52.mtp.graph import GlmGkMtpDecodeGraph
            from sglang_neuron.parallel.groups import get_neuron_attn_tp_group
            from sglang_neuron.speculative.neuron_mtp import is_neuron_mtp

            # --speculative-algorithm NEURON_MTP: the decode graph with the async MTP K7 step
            mtp = is_neuron_mtp(get_spec().speculative_algorithm)
            graph = GlmGkMtpDecodeGraph if mtp else GlmGkDecodeGraph
            from sglang_neuron.models.glm52 import logit_probe

            if logit_probe.test_dir() is not None:
                # the logit test's served MTP step with teacher forcing and raw logits (models/glm52/logit_probe.py)
                if not mtp:
                    raise ValueError(f"{logit_probe.LOGIT_TEST_DIR_ENV} tests the served MTP K7 decode: launch the "
                                     "decode with --speculative-algorithm NEURON_MTP")
                from sglang_neuron.models.glm52.mtp.logit_test_graph import GlmGkMtpLogitTestGraph

                graph = GlmGkMtpLogitTestGraph
            return graph(config, geometry(config), get_neuron_attn_tp_group().rank_in_group)
        return self._build_prefill_stage(config)

    def _build_prefill_stage(self, config):
        # a stage always exists here: check_launch refused a prefill without pipeline stages (prefill_point of its
        # --pp-size), and the host builds no stage only for --pp-size 1
        from sglang_neuron.models.glm52.prefill import stage_weights
        from sglang_neuron.models.glm52.layout.prefill.config import GlmMoeDsaConfig
        from sglang_neuron.models.glm52.layout.prefill.pipeline import stage_config
        from sglang_neuron.models.glm52.prefill.stage import Glm52PrefillStage
        from sglang_neuron.models.glm52.prefill.stage_groups import StageGroups

        full = GlmMoeDsaConfig.from_configs(config, Glm52PrefillOptions())
        layout = stage_layout(full, pp_rank=self.stage.pp_rank, pp_size=self.stage.pp_size)
        if (layout.start_layer, layout.end_layer) != (self.start_layer, self.end_layer):
            partition = prefill_point(self.stage.pp_size).partition
            raise ValueError(f"SGLang's split gives stage {layout.pp_rank} the layers [{self.start_layer}, "
                             f"{self.end_layer}), the stage layout [{layout.start_layer}, {layout.end_layer}): launch "
                             "with SGLANG_PP_LAYER_PARTITION=" + ",".join(map(str, partition)))
        if int(full.hidden_size) != self.pp_hidden_size:
            raise ValueError(f"the checkpoint's hidden size {full.hidden_size} is not the stage boundary's "
                             f"{self.pp_hidden_size}")
        if "pipeline_picks" in self.pp_prefill_boundary and int(full.index_topk) != self.pp_picks_width:
            # the boundary's picks slot is pp_picks_width wide, and the kernel writes index_topk picks into it
            raise ValueError(f"the checkpoint's index_topk {full.index_topk} is not the {self.pp_picks_width} picks "
                             f"the {self.stage.pp_size}-stage boundary carries")
        cfg = dataclasses.replace(stage_config(full, layout.start_layer, layout.end_layer, shared_start=layout.picks_in),
                                  options=Glm52PrefillOptions(glm52_mtp_kv=layout.mtp_rows, ep_degree=layout.cp))
        groups = StageGroups.from_plugin()
        if groups.cp.world_size != layout.cp or groups.ep.world_size != cfg.options.ep_degree:
            raise ValueError(f"the stage's groups hold {groups.cp.world_size} CP and {groups.ep.world_size} EP ranks; "
                             f"the stage graph runs CP{layout.cp} x EP{cfg.options.ep_degree}: launch with "
                             f"--tp-size {layout.cp} --ep-size {cfg.options.ep_degree}")
        weights = stage_weights.build_stage_weights(cfg, layout, cfg.options, groups)
        return Glm52PrefillStage(layout, cfg, weights, groups)

    def neuron_compiler_args(self, phase: str, default_args: list[str]) -> list[str]:
        """The role's own neuronx-cc arguments, in place of the platform's defaults (GPT-OSS's): the prefill's stage
        graphs' (compile_args, at its own scratchpad page), the decode's GK step's (GK_COMPILER_ARGS) at the
        launch's scratchpad page, which the runtime bills in (serving_policy.ServingPolicy.scratchpad_page_mb)."""
        if phase == "decode":
            page = [arg for arg in default_args if arg.startswith("--hbm-scratchpad-page-size=")]
            return list(GK_COMPILER_ARGS) + page
        from sglang_neuron.models.glm52.prefill import compile_args

        return compile_args.stage_compiler_args()

    def fleet_digest(self) -> tuple:
        """The prefill's launch facts every stage rank must hold the same (launch.fleet_digest); the decode states
        none beyond the platform's."""
        if self.phase != "prefill":
            return ()
        return tuple(launch.fleet_digest(int(self.stage.pp_size)))

    def configure_kv_cache(self, configurator, *, pre_model_load_memory):
        """The role's pools in place of SGLang's: the prefill stage's KV pool, allocator and request pool
        (kv_pool.configure_kv_cache), or the decode's request pool of the kernel's slots over a KV pool that only
        counts tokens (decode.pools.configure_kv_cache)."""
        if self.phase == "decode":
            from sglang_neuron.models.glm52.decode import pools as decode_pools

            graph = self.model_graph
            # the async MTP draft's caches, which only the MTP graph holds (GlmGkMtpDecodeGraph.pd_mtp_caches)
            return decode_pools.configure_kv_cache(configurator, graph.geo, pre_model_load_memory=pre_model_load_memory,
                                                   caches=graph.gk.caches, rank=graph.rank,
                                                   mtp_caches=getattr(graph, "pd_mtp_caches", None))
        from sglang_neuron.models.glm52.prefill import kv_pool

        return kv_pool.configure_kv_cache(configurator, self.model_graph.layout,
                                          pre_model_load_memory=pre_model_load_memory,
                                          cp_rank=int(self.model_graph.groups.cp.rank_in_group))

    def bind_kv_cache(self, pool) -> None:
        """Bind the pool's caches into the prefill stage graph (Glm52PrefillStage.bind_pool); executors bound to other
        caches prepare their inputs again at their next calls. The decode's GK reads no layer of SGLang's pool, which
        the platform's binding hands its graph (GlmGkDecodeGraph.bind_kv_cache)."""
        if self.phase == "decode":
            super().bind_kv_cache(pool)
            return
        from sglang_neuron.backend import executor_cache

        self.model_graph.bind_pool(pool)
        executor_cache.invalidate()

    def neuron_after_warmup(self, family: str, warmed_keys: dict) -> None:
        """Log a prefill stage's NEFF keys: the cache entries a warm start of the same tree reuses."""
        if self.phase != "prefill":
            return
        logger.info("GLM-5.2 %s stage %d NEFFs: %s", family, self.stage.pp_rank,
                    {graph: list(keys) for graph, keys in warmed_keys.items()})


EntryClass = GlmMoeDsaForCausalLM
