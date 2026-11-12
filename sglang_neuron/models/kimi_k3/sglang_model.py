# SPDX-License-Identifier: Apache-2.0
"""KimiK3ForConditionalGeneration: the host SGLang builds for a Kimi-K3 checkpoint (models/host.py), serving the PD
prefill of the text model. On the PD decode role the class builds the decode host instead (decode/host.py:
the verify and draft programs over 256 LNC1 ranks, with its own DSPARK worker).

Each pipeline stage builds one prefill graph (graph.KimiK3PrefillGraph) of its layers [start_layer, end_layer) over
SGLang's own pools for a hybrid MLA model: SGLang's ``HybridLinearKVPool`` holds the stage's MLA layers in the
platform's MLA pool (FP8 rows, DCP placement over the stage's TP group, every page zeroed as it is handed out), and
SGLang's request table holds the KDA layers' state in the platform's recurrent-state pool (upstream's layout, bf16).
The runtime (runtime.KimiK3PrefillRuntime) lowers each chunk to the graph's inputs. The stage boundary crosses the PP
seam as three kinds: the stream row block, the AttnRes bank (entry-major, one row block per entry, growing by one
entry a stage) and, with the drafter, its fp32 context accumulator.

The drafter context rows are written on the last stage when the launch names the drafter checkpoint
(--speculative-draft-model-path, as the decode takes it), into drafter row pools of the MLA pool's pages (the same slots
as the target rows), which the KV budget leaves out (``device_bytes_beside_pools``)."""

from __future__ import annotations

import copy
import logging
import os

import torch
from torch import nn

from sglang_neuron.models.host import ModelHost

logger = logging.getLogger(__name__)

ARCHITECTURE = "KimiK3ForConditionalGeneration"
DRAFTER_FLAG = "--speculative-draft-model-path"
STRIPE = 128


class _NoWrappedView:
    """The layer router SGLang's PD registration reads of a draft pool: the drafter row pools wrap no token KV view,
    so their entries take the draft band ids by layer count (disaggregation.utils.build_kv_layer_ids)."""

    @staticmethod
    def wrapped_view():
        return None

    @staticmethod
    def part_view(part):
        return None


class DrafterRowPool:
    """The drafter writer stage's drafter row pools (one ``[MLA pool pages, 128, row_width]`` uint8 tensor per drafter
    layer, at the MLA pool's pages) as the draft pool SGLang's PD prefill registers after the target's entries: one
    entry per drafter layer, one MLA pool page (a 128-token stripe) per item. A Kimi-K3 decode registers its rider pool
    the same way, so the two pair by the same draft band ids."""

    layer_router = _NoWrappedView()

    def __init__(self, rows, page_size: int):
        self.rows = list(rows)
        self.layer_num = len(self.rows)
        self.page_size = int(page_size)
        self.start_layer = 0

    def get_contiguous_buf_infos(self):
        from sglang_neuron.backend.device_io import device_address

        return ([device_address(t) for t in self.rows], [t.numel() * t.element_size() for t in self.rows],
                [t[0].numel() * t.element_size() for t in self.rows])

    def get_kv_size_bytes(self):
        return sum(t.numel() * t.element_size() for t in self.rows)


def text_config_of(config):
    """The text model's config of a composite (vision + text) checkpoint config, carrying the checkpoint path the
    host loads from; a text config as it is. SGLang's own config object is not modified."""
    text = getattr(config, "text_config", None)
    if text is None:
        return config
    text = copy.copy(text)
    text._name_or_path = getattr(config, "_name_or_path", "") or getattr(text, "_name_or_path", "")
    return text


def drafter_directory():
    """The drafter checkpoint directory of the launch (--speculative-draft-model-path), or None without the drafter
    rows."""
    from sglang.srt.runtime_context import get_spec

    path = get_spec().speculative_draft_model_path
    if not path:
        return None
    if not os.path.isdir(path):
        raise ValueError(f"{DRAFTER_FLAG} {path}: no such drafter checkpoint directory")
    return path


def stage_comm(device):
    """The collectives of this rank's stage group (the attention TP group, which is also the DCP and EP group)."""
    from sglang_neuron.models.kimi_k3.prefill.comm import GroupComm, SingleRankComm
    from sglang_neuron.parallel import groups

    group = groups.get_neuron_attn_tp_group()
    if group.world_size == 1:
        return SingleRankComm(device)
    comm = GroupComm(group.device_group, device)
    # the KDA block kernel's in-kernel collectives run over this group
    comm.replica_groups = (tuple(int(r) for r in group.ranks),)
    return comm


def kda_state_spec(geometry, tp_size: int):
    """The KDA state of every KDA layer per rank of an attention TP of ``tp_size``, bf16 conv and temporal
    (--mamba-ssm-dtype bfloat16): conv channel-major ``[3 hc D, K - 1]``, the layout the KDA kernels read and write,
    temporal upstream's ``[hc, D, D]``."""
    from sglang_neuron.mem_cache.state_spec import CHANNEL_MAJOR, kda_spec

    return kda_spec(layer_ids=geometry.kda_layers, tp_size=tp_size, num_heads=geometry.kda_num_heads,
                    head_dim=geometry.kda_head_dim, conv_kernel=geometry.kda_conv_kernel,
                    temporal_dtype="bfloat16", conv_dtype="bfloat16", conv_layout=CHANNEL_MAJOR)


def boundary_kinds(geometry, pp_rank: int, pp_size: int, world: int, drafter=None, environ=None) -> dict:
    """Stage ``pp_rank``'s prefill boundary, in the order the graph takes it (distributed.pp_seam.BoundaryKind): the
    stream row block, the bank (a slot of the largest boundary's entries, one row block each; the stage receives its
    input entries and sends its output entries) and, with the drafter, the accumulator (every row, this rank's
    columns). Every stage declares the same kinds and slots."""
    from sglang_neuron.distributed.pp_seam import BoundaryKind
    from sglang_neuron.models.kimi_k3.graph import BANK, DRAFT_ACC, HIDDEN, KINDS
    from sglang_neuron.models.kimi_k3.prefill.stage_plan import stage_plan

    targets = None if drafter is None else drafter.target_layer_ids
    plans = [stage_plan(geometry, s, pp_size, draft_target_layers=targets, environ=environ) for s in range(pp_size)]
    plan = plans[pp_rank]
    blocks = max([p.bank_out for p in plans[:-1]] or [1])
    first, last = plan.is_first, plan.is_last
    one_in, one_out = 0 if first else 1, 0 if last else 1
    kinds = {
        HIDDEN: BoundaryKind(KINDS[HIDDEN], geometry.hidden_size, torch.bfloat16, sent=one_out, received=one_in),
        BANK: BoundaryKind(KINDS[BANK], geometry.hidden_size, torch.bfloat16, blocks=blocks,
                           sent=0 if last else plan.bank_out, received=plan.bank_in),
    }
    if drafter is not None:
        kinds[DRAFT_ACC] = BoundaryKind(KINDS[DRAFT_ACC], drafter.hidden_size // world, torch.float32,
                                        sp_sharded=False, sent=one_out, received=one_in)
    return kinds


class KimiK3ForConditionalGeneration(ModelHost):
    name = "Kimi-K3"
    # the MLA attention is the stage's own DCP segment kernel, not the segmented prefill kernel
    segmented_prefill = False
    prefill_holds_prior = False

    def __new__(cls, config=None, *args, **kwargs):
        """On the PD decode role SGLang's loader gets the decode host (decode/host.py) for this architecture; every
        other role builds this prefill host."""
        from sglang_neuron.models import phase

        if config is not None and phase.model_phase() == "decode":
            from sglang_neuron.models.kimi_k3.decode.host import build_decode_host

            return build_decode_host(text_config_of(config), *args, **kwargs)
        return super().__new__(cls)

    @classmethod
    def speculative_worker_cls(cls, algorithm: str):
        """The decode's DSPARK worker (model_executor.spec_worker reads it from the registered class)."""
        from sglang_neuron.models.kimi_k3.decode.host import KimiK3DecodeHost

        return KimiK3DecodeHost.speculative_worker_cls(algorithm)

    def __init__(self, config, quant_config=None, prefix: str = "", **kwargs):
        super().__init__(text_config_of(config), quant_config, prefix, **kwargs)
        graph = self.model_graph
        self.geometry, self.plan = graph.geometry, graph.plan
        self.state_spec = kda_state_spec(self.geometry, graph.world)
        from sglang_neuron.mem_cache.state_pool import NeuronStatePool
        from sglang_neuron.model_executor import kv_layout

        # SGLang's request tables build the state pool from this class (sglang_patches.state_pool_class)
        self.state_pool_class = NeuronStatePool.bind(self.state_spec)
        if self.stage is not None:
            self.pp_prefill_boundary_kinds = boundary_kinds(self.geometry, self.stage.pp_rank, self.stage.pp_size,
                                                            graph.world, graph.drafter)
        # the MLA flash attention reads whole blocks past a request's end: every page reads zero when handed out
        kv_layout.declare(kv_layout.KVLayout(zero_fresh_pages=True))
        self.draft_row_pools = []

    def device_bytes_beside_pools(self, max_total_tokens) -> int:
        """The HBM the drafter row pools take beside SGLang's pools on the drafter writer stage (bind_kv_cache: one
        ``[pages, 128, row width]`` uint8 tensor a drafter layer at the MLA pool's pages and its null page), which the
        KV budget leaves out (general_plugin._less_the_model_pools_beside_sglang); 0 on the other stages."""
        graph = self.model_graph
        if not graph.drafter_writer or not max_total_tokens:
            return 0
        pages = int(max_total_tokens) // STRIPE + 1
        return pages * STRIPE * graph.drafter.row_width * graph.drafter.num_layers

    @property
    def pd_draft_kv_pool(self):
        """The drafter row pools as the draft pool SGLang's PD prefill registers after the target's entries
        (DrafterRowPool, sglang_patches.dcp_page_transfer.prefill_init_kv_manager) on the drafter writer stage; None on
        the other stages. Raises on the writer stage before bind_kv_cache made its pools."""
        graph = getattr(self, "model_graph", None)
        if graph is None or not graph.drafter_writer:
            return None
        if not self.draft_row_pools:
            raise RuntimeError(f"{self.name}'s drafter writer stage registers its PD KV before its drafter row pools "
                               "exist (bind_kv_cache)")
        return DrafterRowPool(self.draft_row_pools, STRIPE)

    @property
    def runtime_class(self):
        from sglang_neuron.models.kimi_k3.runtime import KimiK3PrefillRuntime

        return KimiK3PrefillRuntime

    def neuron_compiler_args(self, phase: str, default_args: list[str]) -> list[str]:
        """The stage graph's compile: the plugin's prefill arguments with the backend's nested dynamic loops (the
        blockwise MoE kernel's block loop inside the layer loop), the partition limit of a whole stage, the hierarchy
        splitter (each distinct layer module compiles once) and the scratchpad page the runtime starts with."""
        from sglang_neuron.backend.stack import SPLITTER_OPTION, with_hlo_options

        page = os.environ.get("NEURON_SCRATCHPAD_PAGE_SIZE")
        args = []
        for arg in default_args:
            if arg.startswith("--internal-backend-options="):
                arg = f"{arg} --enable-nested-dynamic-loop"
            elif page and arg.startswith("--hbm-scratchpad-page-size="):
                # the compiler's page and the runtime's must match
                arg = f"--hbm-scratchpad-page-size={page}"
            args.append(arg)
        return with_hlo_options(args, drop=(SPLITTER_OPTION,),
                                add=("--partitioner-max-disj-runs=64", SPLITTER_OPTION + "true"))

    def check_launch(self, config) -> None:
        """Refuse a launch the stage graphs cannot serve, naming every reason: another role than the PD prefill, DP
        attention, a head, expert or vocabulary count the stage group does not split, a DCP group other than the whole
        stage group, a chunk that is not whole stripe cycles, a radix cache other than the chunk-keyed tree or the
        kimi-prompt cache (the KDA state is kept at forward ends only), and a drafter checkpoint the context rows do not
        implement."""
        from sglang.srt.runtime_context import get_memory, get_schedule

        from sglang_neuron import sglang_tree
        from sglang_neuron.backend import envs as backend_envs
        from sglang_neuron.models.kimi_k3.prefill.draft_context import read_drafter_geometry
        from sglang_neuron.models.kimi_k3.prefill.geometry import read_geometry
        from sglang_neuron.models.phase import model_phase

        geometry = read_geometry(config)
        phase = model_phase()
        if phase != "prefill":
            raise ValueError(f"{self.name} on Neuron serves the PD prefill role, not {phase!r}: launch with "
                             "--disaggregation-mode prefill")
        pc = sglang_tree.parallel_config()
        world = int(pc.tp_size)
        problems = []
        if sglang_tree.attention_dp_size(pc) != 1:
            problems.append("DP attention: a stage's TP group is its attention group; launch without "
                            "--enable-dp-attention")
        intermediate = geometry.num_shared_experts * geometry.moe_intermediate_size
        for what, count in (("KDA heads", geometry.kda_num_heads), ("MLA heads", geometry.num_heads),
                            ("experts", geometry.num_experts), ("vocabulary rows", geometry.vocab_size),
                            ("dense MLP intermediate", geometry.intermediate_size),
                            ("shared expert intermediate", intermediate)):
            if count % world:
                problems.append(f"its {count} {what} do not split over --tp-size {world}")
        ep = int(pc.ep_size or 1)
        if ep not in (1, world):
            problems.append(f"--ep-size {ep}: the experts split over the stage's TP group; launch with --ep-size "
                            f"{world} or without it")
        dcp = int(getattr(pc, "attn_dcp_size", 1) or 1)
        if world > 1 and dcp != world:
            problems.append(f"--dcp-size {dcp}: the MLA rows are striped over the stage's whole TP group; launch "
                            f"with --dcp-size {world}")
        cycle = STRIPE * world
        chunk = int(get_schedule().chunked_prefill_size or 0)
        if chunk <= 0 or chunk % cycle:
            problems.append(f"--chunked-prefill-size {chunk}: a chunk is whole stripe cycles of {cycle} tokens "
                            f"({STRIPE} x --dcp-size); launch with a multiple of {cycle}")
        from sglang_neuron import serving_policy

        policy = serving_policy.current()
        if (not get_memory().disable_radix_cache and not policy.chunk_state_checkpoints
                and not policy.kimi_prompt_cache):
            problems.append("a radix cache keyed on KV pages: a hit inside a chunk would resume the KDA state where "
                            "the stage graphs keep no checkpoint (they keep one at each chunk's end); launch with "
                            "--radix-cache-backend kimi-prompt, --radix-cache-backend neuron-chunk or "
                            "--disable-radix-cache")
        directory = drafter_directory()
        if directory is not None:
            try:
                drafter = read_drafter_geometry(directory, target_hidden=geometry.hidden_size)
            except ValueError as error:
                problems.append(f"{DRAFTER_FLAG} {directory}: {error}")
            else:
                if drafter.hidden_size % world:
                    problems.append(f"the drafter's {drafter.hidden_size} context columns do not split over "
                                    f"--tp-size {world}")
                if not all(0 <= t < geometry.num_layers - 1 for t in drafter.target_layer_ids):
                    problems.append(f"the drafter's target layers {list(drafter.target_layer_ids)} are not layers "
                                    f"before the last of the {geometry.num_layers}")
        if not backend_envs.SGLANG_NEURON_CPU_MODE:
            from sglang_neuron.backend.stack import get_platform_target, normalize_target_family

            target = get_platform_target()
            if normalize_target_family(target) != "trn3":
                problems.append(f"the platform target {target!r}: the Kimi-K3 prefill kernels run on Trainium3")
        if problems:
            raise ValueError(f"{self.name} on Neuron cannot serve this launch: " + "; ".join(problems))

    def build_graph(self, phase: str, config) -> nn.Module:
        from sglang_neuron.backend import envs as backend_envs
        from sglang_neuron.models.kimi_k3.graph import KimiK3PrefillGraph
        from sglang_neuron.models.kimi_k3.prefill.draft_context import read_drafter_geometry
        from sglang_neuron.models.kimi_k3.prefill.geometry import read_geometry
        from sglang_neuron.models.kimi_k3.prefill.stage_plan import stage_plan
        from sglang_neuron.parallel import groups

        if phase != "prefill":
            raise ValueError(f"{self.name} on Neuron builds the prefill graph only, not the {phase} graph")
        geometry = read_geometry(config)
        directory = drafter_directory()
        drafter = None if directory is None else read_drafter_geometry(directory, target_hidden=geometry.hidden_size)
        pp_rank, pp_size = (0, 1) if self.stage is None else (self.stage.pp_rank, self.stage.pp_size)
        plan = stage_plan(geometry, pp_rank, pp_size,
                          draft_target_layers=None if drafter is None else drafter.target_layer_ids)
        if (plan.layers.start, plan.layers.stop) != (self.start_layer, self.end_layer):
            raise RuntimeError(f"the Kimi-K3 stage plan holds layers {list(plan.layers)}, SGLang's stage "
                               f"[{self.start_layer}, {self.end_layer}): both read SGLANG_PP_LAYER_PARTITION")
        group = groups.get_neuron_attn_tp_group()
        backend = "torch" if backend_envs.SGLANG_NEURON_CPU_MODE else "kernel"
        logger.info("%s prefill stage %d/%d: layers %s (MLA %s), %s backend, drafter %s", self.name, pp_rank, pp_size,
                    [plan.layers.start, plan.layers.stop], list(plan.mla_layers), backend, directory)
        return KimiK3PrefillGraph(geometry, plan, rank=int(group.rank_in_group), world=int(group.world_size),
                                  comm_factory=lambda device: stage_comm(device), drafter_dir=directory,
                                  drafter=drafter, backend=backend)

    def bind_kv_cache(self, pool):
        """Hand the stage graph its pools: SGLang's hybrid linear-attention pool's MLA leaf (each MLA layer's block
        view, by the pool's own index of the global layer), the recurrent-state pool's tensors of each KDA layer, and
        on the drafter writer stage the drafter row pools (one per drafter layer, of the MLA pool's pages)."""
        from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool

        from sglang_neuron.backend import executor_cache
        from sglang_neuron.distributed.decode_context_parallel import DcpLayout
        from sglang_neuron.mem_cache.memory_pool import NeuronMLATokenToKVPool
        from sglang_neuron.mem_cache.state_pool import check_model_state
        from sglang_neuron.model_executor.runtime_init import neuron_device
        from sglang_neuron.models.kimi_k3.graph import StageCaches

        graph, g = self.model_graph, self.geometry
        full = getattr(pool, "full_kv_pool", None)
        if not isinstance(pool, HybridLinearKVPool) or not isinstance(full, NeuronMLATokenToKVPool):
            raise ValueError(f"{self.name} runs over SGLang's hybrid linear-attention pool with the Neuron MLA pool as "
                             f"its full-attention leaf, not {type(pool).__name__}")
        problems = []
        if full.store_dtype != torch.uint8 or full.kv_cache_dim != g.mla_row_width:
            problems.append(f"MLA rows of {full.kv_cache_dim} {full.store_dtype} (the graph writes {g.mla_row_width} "
                            "FP8 bytes a row; --kv-cache-dtype fp8_e4m3)")
        if full.page_size != STRIPE:
            problems.append(f"{full.page_size}-token blocks (the MLA rows are {STRIPE}-token stripes)")
        if not full.zero_fresh_pages:
            problems.append("no fresh-page zeroing (the model declares KVLayout.zero_fresh_pages)")
        place = DcpLayout(graph.world, graph.rank, STRIPE) if graph.world > 1 else None
        if full.dcp != place:
            problems.append(f"the DCP placement {full.dcp} (the graph keeps the stripes of {place})")
        if problems:
            raise ValueError(f"{self.name}'s MLA pool does not hold the rows the stage graph writes: "
                             + "; ".join(problems))
        check_model_state(ARCHITECTURE, self.state_spec, pool)
        state = pool.mamba_pool
        mapping = pool.full_attention_layer_id_mapping
        mla = {i: full.get_block_buffer(mapping[i]) for i in self.plan.mla_layers}
        conv, temporal = state.spec_stage.convs[0].name, state.spec_stage.recurrent.name
        kda = {}
        for i in self.plan.kda_layers:
            index = state.spec_stage.layer_index(i)
            kda[i] = (state.tensors[conv][index], state.tensors[temporal][index])
        draft = []
        if graph.drafter_writer:
            width = graph.drafter.row_width
            draft = [torch.zeros((full.num_blocks, STRIPE, width), dtype=torch.uint8, device=neuron_device())
                     for _ in range(graph.drafter.num_layers)]
        self.draft_row_pools = draft
        # the drafter rows live at the MLA pool's pages: a HiCache host copy of a page carries them too
        # (mem_cache.hybrid_host_pool.NeuronHostMLAPool)
        full.page_companions = list(draft)
        graph.bind_kv_cache(StageCaches(mla=mla, kda=kda, draft=draft, sink_slot=state.sink_slot))
        # executors bound to the previous KV tensors prepare their inputs again at their next calls
        executor_cache.invalidate()


EntryClass = [KimiK3ForConditionalGeneration]
