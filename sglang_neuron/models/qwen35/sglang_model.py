# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5 (``Qwen3_5MoeForCausalLM``) for SGLang on Neuron: the giga decode behind its megakernel spec.

The model is served with DP attention over replicas of ``world / dp`` ranks, expert parallelism over every rank and
SGLang's hybrid pools (``config.Qwen35Layout``). The host builds:

- the graph families of its server role (``megakernel.ROLE_FAMILIES``): the decode family (``graph.Qwen35DecodeGraph``,
  ``runtime.Qwen35DecodeRuntime``) on an aggregated server and the PD decode role (``--disaggregation-mode decode``),
  the prefill family (``graph.Qwen35PrefillGraph``, ``runtime.Qwen35PrefillRuntime``) on an aggregated server and the PD
  prefill role (``--disaggregation-mode prefill``); ``megakernel.QWEN35_GIGA_SPEC`` serves them. The decode graph
  object holds the stacked operands on every role (``qwen35_decode_graph``): the prefill segments view their slices
  and the pools bind to it, and on the PD prefill it is not compiled. The model declares the spec
  (``megakernel_spec``), so it serves only with ``--enable-nvidia-megakernel``;
- the GDN state declaration (``state_spec``, ``state.state_spec``) and its slot operations (``state_ops``), which
  the spec's pool layout hands SGLang's request table;
- the weights: one rank's operands from a packed checkpoint (``weights.load_rank``), whose directory is
  ``SGLANG_NEURON_QWEN35_PACK`` (SGLang's lazy checkpoint iterator is not read).
"""
from __future__ import annotations

import logging
import os
from typing import Iterable, Tuple

import torch

from sglang_neuron.models.base import NeuronModel

logger = logging.getLogger(__name__)

PACK_ENV = "SGLANG_NEURON_QWEN35_PACK"

_OPT_LEVELS = ("-O0", "-O1", "-O2", "-O3")


def _with_hlo_options(default_args: list, *, drop: tuple = (), add: tuple = ()) -> list:
    from sglang_neuron.backend.stack import with_hlo_options

    return with_hlo_options(default_args, drop=drop, add=add)


def giga_compiler_args(default_args: list) -> list:
    """The decode family compiles as the GPT-OSS giga-kernel decode does: -O2 without the modular-flow threshold (the
    graph is one kernel call; no layer outlining) and without the FSDP annotation of collectives."""
    args = _with_hlo_options([a for a in default_args if a not in _OPT_LEVELS], drop=("--modular-flow-mac-threshold=",),
                             add=("--enable-annotate-fsdp-cc=false",))
    return args + ["-O2"]


def prefill_compiler_args(default_args: list) -> list:
    """The prefill family compiles in neuronx-cc's modular flow (the serving default arguments: -O1 and the forced
    modular-flow threshold) with the hierarchy splitter, without the FSDP annotation of collectives. The splitter
    partitions the graph by its module hierarchy: one partition per prefill segment (graph.Qwen35PrefillSegment),
    which are one module definition, so the segment compiles once and every segment runs it."""
    return _with_hlo_options(default_args, drop=("--hierarchy-splitter=",),
                             add=("--enable-annotate-fsdp-cc=false", "--hierarchy-splitter=true"))


class Qwen3_5MoeForCausalLM(NeuronModel):
    # the giga prefill does not run the segmented kernel; its paged pool keeps the chunk grid itself (__init__)
    prefill_chunk_grid = False
    # the giga prefill runs the chunk alone (one prefill bucket), not the segment tiles below it
    prefill_ladder = False
    # the giga decode reads the request's whole KV slot: no quarter/half-context graphs
    decode_context_ladder = False
    # the prefill attention is the giga kernel's, not the segmented kernel
    segmented_prefill = False
    # the prefill graph's window is the prompt alone, not the prompt and its cached prior
    prefill_holds_prior = False

    def __init__(self, config, quant_config=None, prefix: str = "", **kwargs):
        super().__init__()
        from sglang.srt.runtime_context import get_schedule

        from sglang_neuron import sglang_tree
        from sglang_neuron.models.phase import model_phase

        from .config import (KV_CP_ENV, KV_TM_ENV, Qwen35Dims, Qwen35Layout, kv_cp_launch, kv_tm_launch,
                             pool_context_problems)
        from .graph import Qwen35DecodeGraph
        from .state import GdnStateOps, state_spec

        self.hf_config = config
        self.phase = model_phase()
        from .megakernel import ROLE_FAMILIES

        if self.phase not in ROLE_FAMILIES:
            raise ValueError(f"the Qwen3.5 giga host serves an aggregated server or a PD role; this server's role is "
                             f"{self.phase}")
        pc = sglang_tree.parallel_config()
        world = int(pc.tp_size)
        dp = int(sglang_tree.attention_dp_size(pc))
        page = int(get_schedule().page_size or 1)
        from sglang.srt.runtime_context import get_server_args

        from .megakernel import paged_launch

        args = get_server_args()
        context = int(args.context_length) if paged_launch(args) else 0
        self.qwen35_dims = Qwen35Dims.from_hf(config)
        kv_cp = kv_cp_launch()
        if kv_cp > 1 and self.phase != "decode":
            raise ValueError(f"{KV_CP_ENV}={kv_cp}: decode context parallelism runs on the PD decode role; this "
                             f"server's role is {self.phase}")
        if kv_tm_launch():
            # token-major K pages are the kv_cp=1 PD prefill's half of the kv_cp scatter layout: the kv_cp
            # decode's pool is token-major unconditionally, and no other role may run the flag — the kv_cp=1
            # decode's attention kernel reads the dim-major layout, and a whole-page PD transfer (an equal-TP
            # pairing) is layout-blind
            if self.phase != "prefill" or kv_cp != 1:
                raise ValueError(f"{KV_TM_ENV}=1: token-major K pages run on the kv_cp=1 PD prefill role only; "
                                 f"this server's role is {self.phase} with {KV_CP_ENV}={kv_cp}")
            if not context:
                raise ValueError(f"{KV_TM_ENV}=1: token-major K pages serve the paged pool's PD scatter; a "
                                 "slot-pool prefill (no --context-length) hands whole slots to a dim-major decode")
        if kv_tm_launch():
            # the scatter handshake (sglang_patches/pd_kv_cp_layout) publishes the K layout so a dim-major peer
            # is refused at registration
            from sglang_neuron.model_executor import kv_layout

            kv_layout.declare(kv_layout.KVLayout(k_token_major=True))
        self.qwen35_layout = Qwen35Layout(world=world, dp=dp, slot_tokens=page, context=context, kv_cp=kv_cp)
        problems = self.qwen35_dims.problems() + self.qwen35_layout.problems(
            self.qwen35_dims, ep_size=int(pc.ep_size or 1))
        # a request near the context must fit the pool together with the page the KV allocator keeps back
        problems += pool_context_problems(context, page, getattr(args, "max_total_tokens", None))
        if problems:
            raise ValueError("the Qwen3.5 giga decode cannot serve this launch: " + "; ".join(problems))
        if self.qwen35_layout.paged:
            # the paged prefill runs one chunk per forward on the chunk grid (models/qwen35/prefix.py)
            from sglang_neuron.models.base import segmented_prefill_cache_policy

            segmented_prefill_cache_policy(self.phase)
        from sglang_neuron.parallel.groups import get_neuron_attn_tp_group, get_wide_ep_group
        from sglang_neuron.parallel.state import initialize_neuron_parallel_state

        from .graph import Qwen35PrefillGraph

        initialize_neuron_parallel_state(ep_degree=world, dp_size=dp)
        attn_tp = get_neuron_attn_tp_group()
        if attn_tp.world_size != self.qwen35_layout.replica:
            raise ValueError(f"the Neuron attention TP group has {attn_tp.world_size} ranks; the layout's replica "
                             f"{self.qwen35_layout.replica}")
        world_group = get_wide_ep_group() if dp > 1 else attn_tp
        if world_group.world_size != world:
            raise ValueError(f"the Neuron wide EP group has {world_group.world_size} ranks, the world {world}")
        # decode context parallelism (config.Qwen35Layout.kv_cp): declare it before the pools are built so the KV
        # pool and the runtime read it (memory_pool.NeuronMHATokenToKVPool, graph_runner.NeuronModelRuntime)
        self.kv_cp_size = int(self.qwen35_layout.kv_cp)
        self.kv_cp_decode = self.kv_cp_size > 1
        if self.kv_cp_decode:
            from sglang_neuron.model_executor import kv_layout

            # k_token_major: the decode pool's per-head K pages are [tokens, head_dim] like its V pages
            # (kv_pool.k_tm; attn_cp_core reads and writes them that way), so the PD scatter's K block of a
            # page is one contiguous run per layer instead of one per dim row
            kv_layout.declare(kv_layout.KVLayout(cp_size=self.kv_cp_size, cp_rank=attn_tp.rank_in_group,
                                                 kv_heads_per_rank=self.qwen35_dims.num_kv_heads,
                                                 k_token_major=True))
        self.state_spec = state_spec(self.qwen35_dims, self.qwen35_layout)
        self.state_ops = GdnStateOps()
        with torch.device("cpu"):
            graph = Qwen35DecodeGraph(self.qwen35_dims, self.qwen35_layout, tp_group=attn_tp,
                                      position=attn_tp.rank_in_group)
        # the stacked operands and the pools live on the decode graph object on every role (module docstring)
        self.qwen35_decode_graph = graph
        families = ROLE_FAMILIES[self.phase]
        self.model_graphs = {}
        if "decode" in families:
            self.model_graphs["decode"] = graph
        if "prefill" in families:
            from sglang.srt.distributed.parallel_state import get_tp_group

            self.model_graphs["prefill"] = Qwen35PrefillGraph(
                graph, world_group=world_group, replica=get_tp_group().rank_in_group // self.qwen35_layout.replica)
        # a one-family role's runtime takes model_graph (graph_runner.NeuronModelRuntime); the graphs sample in their
        # forward (greedy argmax on the device), so each is its own execution model
        self.model_graph = self.model_graphs[families[0]]
        self.execution_models = dict(self.model_graphs)
        self.execution_model = self.model_graph
        self.start_layer, self.end_layer, self.is_last_stage = 0, self.qwen35_dims.num_layers, True
        self.runtime = None
        self._weights_loaded = False
        logger.info("Qwen3.5 giga host (neuron, %s role: %s): world %d, DP %d x replica %d (GDN and attention TP), "
                    "EP %d, KV slot %d tokens", self.phase, "+".join(families), world, dp, self.qwen35_layout.replica,
                    self.qwen35_layout.ep, page)

    @property
    def runtime_class(self):
        from .runtime import Qwen35DecodeRuntime, Qwen35PrefillRuntime

        return Qwen35PrefillRuntime if self.phase == "prefill" else Qwen35DecodeRuntime

    def megakernel_spec(self):
        from .megakernel import spec

        return spec()

    def neuron_compiler_args(self, phase: str, default_args: list) -> list:
        """The decode family compiles as the GPT-OSS giga-kernel decode does (``giga_compiler_args``); the prefill
        family with ``prefill_compiler_args``."""
        if phase == "prefill":
            return prefill_compiler_args(default_args)
        return giga_compiler_args(default_args)

    # pools
    def bind_kv_cache(self, pool):
        """Bind the graph to the pools SGLang built from the spec's layout: the slot KV pool's layer-major K / V and
        the GDN state pool's layer-major conv / recurrent tensors (the hybrid pool's ``mamba_pool``)."""
        from sglang_neuron.backend import executor_cache

        from .kv_pool import Qwen35SlotKVPool
        from .state import Qwen35StatePool

        full = getattr(pool, "full_kv_pool", None)
        state = getattr(pool, "mamba_pool", None)
        if not isinstance(full, Qwen35SlotKVPool) or not isinstance(state, Qwen35StatePool):
            raise RuntimeError(f"the Qwen3.5 giga decode needs SGLang's hybrid pool over the slot KV pool and the GDN "
                               f"state pool; it got {type(full).__name__} and {type(state).__name__}")
        problems = self.qwen35_layout.problems(self.qwen35_dims, attention_layers=full.layer_num, slots=full.slots)
        if problems:
            raise ValueError("the Qwen3.5 KV pool: " + "; ".join(problems))
        # the CP write sink row of the KV stacks (kv_pool.Qwen35SlotKVPool.cp_sink_row), only under kv_cp
        sink = None
        if self.qwen35_layout.kv_cp > 1:
            sink = torch.tensor([[full.cp_sink_row]], dtype=torch.int32, device=full.k_stack.device)
        self.qwen35_decode_graph.bind_pools(state.kernel_tensors["conv0"], state.kernel_tensors["temporal"],
                                            full.k_stack, full.v_stack, cp_sink=sink)
        executor_cache.invalidate()

    # weights
    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """Load this rank's operands from the packed checkpoint (module docstring); an update after the load is
        refused (the operands are the kernel's stacked layouts)."""
        if self._weights_loaded:
            raise NotImplementedError("in-place weight updates are not supported by the Qwen3.5 giga decode")
        root = os.environ.get(PACK_ENV)
        if not root:
            raise ValueError(f"the Qwen3.5 giga decode loads a packed checkpoint: set {PACK_ENV} to its directory "
                             "(build it with python -m tools.qwen35.build_pack.pack, docs/model-recipes/qwen35-radix.md "
                             "'Build the inputs')")
        from sglang.srt.distributed.parallel_state import get_tp_group

        from sglang_neuron.model_executor.runtime_init import neuron_device

        from .weights import Pack, load_prefill_extra, load_rank

        rank = int(get_tp_group().rank_in_group)
        pack, dev = Pack(root), neuron_device()
        decode = self.qwen35_decode_graph
        decode.load_operands(load_rank(pack, self.qwen35_dims, self.qwen35_layout, rank), dev)
        decode.rank_pos = decode.rank_pos.to(dev)
        prefill = self.model_graphs.get("prefill")
        if prefill is not None:
            prefill.load_operands(load_prefill_extra(pack, self.qwen35_dims, self.qwen35_layout, rank), dev)
        self._weights_loaded = True
        logger.info("Qwen3.5 giga decode: rank %d operands loaded from %s", rank, root)


EntryClass = [Qwen3_5MoeForCausalLM]
