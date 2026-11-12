# SPDX-License-Identifier: Apache-2.0
"""The Kimi-K3 PD decode host: the model SGLang builds for a Kimi-K3 checkpoint on a decode role, which loads the
rank's weight stacks, binds the decode programs' pools over SGLang's own, builds the decode runtime (verify and draft
programs) and names its speculative worker.

The checkpoint's model class (``KimiK3ForConditionalGeneration``) builds this host on a decode role
(``build_decode_host``) and its prefill host otherwise; both answer the platform's speculative-worker seam
(model_executor.spec_worker) through ``speculative_worker_cls``.

What it declares before SGLang builds the pools: the request tables' state pool class (sgl_pools.state_pool_class: the
mamba slots of the 69 KDA layers, with no tensors of their own) and the paged allocator class (sgl_pools'
pair allocator, through model_executor.kv_layout). SGLang then builds its hybrid linear-attention pool over the
platform's MLA pool, whose 24 layer buffers are the decode programs' 24 shared buffers when ``--max-total-tokens`` is
``DecodePageLayout.max_total_tokens``.

``_ensure_runtime`` (SGLang's decode graph runner slot, NeuronGraphRunner) checks the pools and binds them
(``bind_pools``: ``pools.DecodePools`` of each MLA layer's block buffer as ``[1700, 73728]`` bf16, the static
``pools.hybrid_pool_of_layer``, the rider pool's rows), builds the rank's
``runtime.K3DecodeRuntime`` over its attention-TP group and warms it, and builds the step driver
(spec_worker.K3StepDriver) the worker runs every step through. On the host (the plugin's CPU mode) the runtime traces
its programs on the host (``CompileOptions.for_device``).
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
from torch import nn

from sglang_neuron import sglang_tree
from sglang_neuron.models.base import NeuronModel
from sglang_neuron.models.kimi_k3.decode import pools as pl
from sglang_neuron.models.kimi_k3.decode import sgl_pools as sp
from sglang_neuron.models.kimi_k3.decode import spec_worker as sw
from sglang_neuron.parallel.state import initialize_neuron_parallel_state

logger = logging.getLogger(__name__)

ALGORITHM = "DSPARK"
FORMS = {4: "k3", 8: "k7b2"}
"""Verify rows Q of a request -> the served form (kernel_calls.SERVED_FORMS)."""
FORM_BATCH = {"k3": 4, "k7b2": 2}
"""form -> the sequences B of a replica (kernel_calls.FORM_BATCH)."""


class _RuntimeSlot:
    """What SGLang's decode graph runner slot and the DP identity (managers.dp_identity.proof_checks) read of a runtime:
    the batch size the decode programs serve, that they are decode programs, that every replica launches every step in
    lockstep (an idle replica runs the programs on pad rows: the EP256 collectives span all 256 ranks), and that each
    step's token input is the fixed batch width."""

    graphs = "decode"
    dp_lockstep = True

    def __init__(self, batch: int):
        self.buckets = type("Buckets", (), {"num_seqs_buckets": (batch,)})()
        self.decode_token_width = int(batch)
        # set by the DP identity once its proof holds
        self.dp_identity_bucket: Optional[int] = None


def launch_reasons(*, role: str, policy, server_args, queries: int, layout_tokens: int) -> list[str]:
    """Why the decode host cannot serve this launch (empty: it can)."""
    reasons = []
    if role != "decode":
        reasons.append(f"the Kimi-K3 decode host serves the PD decode role, not {role!r}")
    if not getattr(policy, "decode_layout", None):
        reasons.append("no declared decode layout (catalog decode_layouts; launch the 2-node layout of "
                       "docs/model-recipes/kimi-k3.md)")
    if str(getattr(policy, "speculative_algorithm", "") or "").upper() != ALGORITHM:
        reasons.append("--speculative-algorithm DSPARK: the decode programs run every step as a DSPARK verify")
    if queries not in FORMS:
        reasons.append(f"--speculative-num-draft-tokens {queries}: the decode programs verify 4 (k 3) or 8 (k 7) rows")
    stated = getattr(server_args, "max_total_tokens", None)
    if stated != layout_tokens:
        reasons.append(f"--max-total-tokens {stated}: the decode programs' pools are {layout_tokens} tokens a rank "
                       f"(the 24 shared buffers of {sp.POOL_HOST_PAGES} host pages); launch with --max-total-tokens "
                       f"{layout_tokens}")
    if getattr(server_args, "kv_cache_dtype", None) not in ("fp8_e4m3",):
        reasons.append(f"--kv-cache-dtype {getattr(server_args, 'kv_cache_dtype', None)}: the MLA rows are FP8 "
                       "(--kv-cache-dtype fp8_e4m3)")
    if int(getattr(server_args, "page_size", 0) or 0) != sp.STRIPE:
        reasons.append(f"--page-size {getattr(server_args, 'page_size', None)}: the MLA rows are {sp.STRIPE}-token "
                       f"stripes (--page-size {sp.STRIPE})")
    if not getattr(server_args, "disable_radix_cache", False):
        reasons.append("a radix cache: the decode keeps no KDA checkpoint (--disable-radix-cache)")
    return reasons


class KimiK3DecodeHost(NeuronModel):
    """The Kimi-K3 PD decode host (the module docstring)."""

    name = "Kimi-K3 decode"
    # the DSPARK worker's drafter rows live in the decode programs' pools (the rider rings), so SGLang budgets and
    # builds no draft KV pool (general_plugin._without_draft_kv_pool)
    draft_kv_in_model_pools = True
    # every step runs through the DSPARK worker's decode programs and forward is refused, so SGLang's attention
    # backends serve nothing: no hybrid linear-attention wrapper (general_plugin._no_linear_attention_backend)
    runs_no_sglang_attention = True

    @classmethod
    def speculative_worker_cls(cls, algorithm: str):
        return sw.worker_class() if str(algorithm).upper() == ALGORITHM else None

    def set_dspark_layers_to_capture(self, layer_ids) -> None:
        """SGLang's DSPARK capture setting (model_runner_components.attention_backend_setup): the layers whose outputs
        the drafter reads. The decode programs capture the drafter's target layers themselves
        (weights_draft.DRAFT_TARGET_LAYERS, checked against the drafter checkpoint as it loads), so any other list is
        refused."""
        from sglang_neuron.models.kimi_k3.decode.weights_draft import DRAFT_TARGET_LAYERS

        asked = None if layer_ids is None else tuple(int(i) for i in layer_ids)
        if asked != DRAFT_TARGET_LAYERS:
            raise ValueError(f"{self.name}: SGLang asks the DSPARK capture of layers {asked}; the decode programs "
                             f"capture the drafter's target layers {list(DRAFT_TARGET_LAYERS)}")

    def __init__(self, config, quant_config=None, prefix: str = "", *, server_args=None, environ=None, **kwargs):
        super().__init__()
        import os

        from sglang_neuron import serving_policy
        from sglang_neuron.model_executor import kv_layout
        from sglang_neuron.models.kimi_k3.prefill.geometry import read_geometry

        server_args = server_args if server_args is not None else _server_args()
        environ = os.environ if environ is None else environ
        self.hf_config = config
        self.geometry = read_geometry(config)
        self.queries = int(getattr(server_args, "speculative_num_draft_tokens", 0) or 0)
        policy = serving_policy.current()
        layout_tokens = sp.DecodePageLayout(max(self.queries, 2), 1).max_total_tokens
        reasons = launch_reasons(role=policy.role, policy=policy, server_args=server_args, queries=self.queries,
                                 layout_tokens=layout_tokens)
        if reasons:
            raise ValueError(f"{self.name} on Neuron cannot serve this launch: " + "; ".join(reasons))
        # the plugin's groups, as every model host builds them before SGLang builds the pools: the Neuron MLA pool
        # takes its DCP placement from the plugin's DCP group (decode_context_parallel.runtime_layout), and SGLang
        # widens the paged allocator by the DCP size to one 128-token block on each of the 32 DCP ranks a page
        pc = sglang_tree.parallel_config()
        initialize_neuron_parallel_state(ep_degree=int(pc.ep_size or 1), dp_size=sglang_tree.attention_dp_size(pc))
        self.form = FORMS[self.queries]
        self.batch = FORM_BATCH[self.form]
        self.is_mla = tuple(self.geometry.is_mla(i) for i in range(self.geometry.num_layers))
        # one stage of every layer: SGLang's model runner reads the model's [start_layer, end_layer) for its layer
        # range (model_runner_components.layer_setup.resolve_layer_indices); NeuronModel leaves end_layer unset
        self.start_layer, self.end_layer = 0, len(self.is_mla)
        self.pool_of_layer = pl.hybrid_pool_of_layer(self.is_mla)
        self.server_args = server_args
        self.drafter_dir = getattr(server_args, "speculative_draft_model_path", None)
        self.synthetic_env = {sw.FORCED_AL_ENV: environ.get(sw.FORCED_AL_ENV, "")}
        # SGLang's request tables build their state pool from this class (sglang_patches.state_pool_class), and its
        # paged allocator is the platform's allocator class that the layout names
        self.state_pool_class = sp.state_pool_class(self.queries, self.geometry.kda_layers)
        kv_layout.declare(kv_layout.KVLayout(paged_allocator_class=sp.pair_page_allocator_class))
        self.weights: Optional[dict] = None
        self.runtime = None
        self.decode_runtime = None
        self.rider_pool: Optional[sp.RiderPool] = None
        self._driver: Optional[sw.K3StepDriver] = None
        logger.info("%s: form %s, %d sequences a replica, %d layers (%d MLA)", self.name, self.form, self.batch,
                    len(self.is_mla), sum(self.is_mla))

    # -- weights ----------------------------------------------------------------------------------------------------
    def load_weights(self, weights):
        """SGLang hands over a lazy checkpoint iterator, left unread: the rank's stacks come from the checkpoint
        directory (weights.load_decode_rank), on this rank's device."""
        from sglang_neuron import sglang_tree
        from sglang_neuron.model_executor.runtime_init import neuron_device
        from sglang_neuron.models.kimi_k3.decode.weights import load_decode_rank

        rank = int(sglang_tree.parallel_config().tp_rank)
        self.weights = load_decode_rank(self.hf_config._name_or_path, self.drafter_dir, self.geometry, rank,
                                        form=self.form, device=neuron_device())
        logger.info("%s weights of rank %d loaded (%d tensors)", self.name, rank, len(self.weights))

    # -- pools and runtime ------------------------------------------------------------------------------------------
    def bind_pools(self, token_to_kv_pool, req_to_token_pool, device=None) -> tuple[pl.DecodePools,
                                                                                     sp.DecodePageLayout]:
        """Check SGLang's pools and return the decode programs' pools over them, and the page layout."""
        from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool

        from sglang_neuron.backend.device_io import device_address
        from sglang_neuron.mem_cache.memory_pool import NeuronMLATokenToKVPool

        full = getattr(token_to_kv_pool, "full_kv_pool", None)
        if not isinstance(token_to_kv_pool, HybridLinearKVPool) or not isinstance(full, NeuronMLATokenToKVPool):
            raise ValueError(f"{self.name} runs over SGLang's hybrid linear-attention pool with the Neuron MLA pool as "
                             f"its leaf, not {type(token_to_kv_pool).__name__}")
        state = getattr(req_to_token_pool, "mamba_pool", None)
        layout = getattr(state, "layout", None)
        if not isinstance(layout, sp.DecodePageLayout):
            raise ValueError(f"{self.name}: the request table's state pool is {type(state).__name__}, not the "
                             "decode's (the model's state_pool_class)")
        problems = []
        if full.store_dtype != torch.uint8 or full.kv_cache_dim != sp.MLA_ROW_BYTES:
            problems.append(f"MLA rows of {full.kv_cache_dim} {full.store_dtype}, not {sp.MLA_ROW_BYTES} FP8 bytes")
        if full.page_size != sp.STRIPE or full.num_blocks != 2 * sp.POOL_HOST_PAGES:
            problems.append(f"{full.num_blocks} blocks of {full.page_size} tokens, not {2 * sp.POOL_HOST_PAGES} of "
                            f"{sp.STRIPE}")
        if getattr(full, "zero_fresh_pages", False):
            problems.append("fresh-page zeroing (a decode's pages receive their KV over NIXL)")
        mapping = token_to_kv_pool.full_attention_layer_id_mapping
        mla_layers = [i for i, mla in enumerate(self.is_mla) if mla]
        if sorted(mapping) != mla_layers:
            problems.append(f"the hybrid pool's full-attention layers {sorted(mapping)}, not the MLA layers "
                            f"{mla_layers}")
        if problems:
            raise ValueError(f"{self.name}'s pools do not hold what the decode programs read: " + "; ".join(problems))
        # buffer i is MLA layer i's (pools.hybrid_pool_of_layer numbers the MLA layers' buffers in layer order)
        kv = tuple(full.get_block_buffer(mapping[layer]).view(sp.POOL_HOST_PAGES, sp.HOST_PAGE_BYTES)
                   .view(torch.bfloat16) for layer in mla_layers)
        state.bind_buffers([device_address(buffer) for buffer in kv], self.pool_of_layer)
        self.rider_pool = sp.RiderPool(layout, device=device)
        return pl.DecodePools(kv=kv, pool_of_layer=self.pool_of_layer, rider=self.rider_pool.kernel_rows()), layout

    def build_runtime(self, pools: pl.DecodePools, rank: int, device=None, tp_group=None):
        """The decode runtime of global rank ``rank`` over ``pools`` (``runtime.K3DecodeRuntime``), warmed up."""
        from sglang_neuron.models.kimi_k3.decode import runtime as rt
        from sglang_neuron.models.kimi_k3.decode.layout import DP8_LAYOUT

        config = rt.program_config(self.form, geometry=self.geometry, drafter_dir=self.drafter_dir, layout=DP8_LAYOUT)
        runtime = rt.K3DecodeRuntime(layout=DP8_LAYOUT, form=self.form, rank=rank, weights=self.weights, pools=pools,
                                     compile_options=rt.CompileOptions.for_device(device or "cpu"), device=device,
                                     tp_group=tp_group, config=config)
        from sglang_neuron.models.kimi_k3 import logits_dump

        if logits_dump.from_env() is not None:
            runtime.enable_logits_dump()
        runtime.warmup()
        return runtime

    def _ensure_runtime(self, model_runner):
        if self.runtime is not None:
            return self.runtime
        from sglang_neuron import sglang_tree
        from sglang_neuron.model_executor.runtime_init import neuron_device
        from sglang_neuron.parallel import groups

        device = neuron_device()
        pools, layout = self.bind_pools(model_runner.token_to_kv_pool, model_runner.req_to_token_pool, device)
        rank = int(sglang_tree.parallel_config().tp_rank)
        # the embedding's all-reduce runs over this replica's attention-TP ranks
        tp_group = groups.get_neuron_attn_tp_group().device_group
        self.decode_runtime = self.build_runtime(pools, rank, device, tp_group)
        full = model_runner.token_to_kv_pool.full_kv_pool
        # the driver zeroes each new request's tail blocks with the pool's fill: compile the fill now, not at the first
        # request while the decode serves
        full.warm_zero_pages()
        self._driver = self.build_driver(self.decode_runtime, layout, rank, device, zero_blocks=full.zero_pages)
        self.runtime = _RuntimeSlot(self.batch)
        from sglang_neuron.model_executor import async_execution

        # A PD decode processes each step's result by synchronizing its CompletionEvent on the scheduler thread, so
        # the result is read there, inline, as the platform's graph runner configures a PD decode: no copy thread
        # waits on the step's launch event and harvests it beside the scheduler thread's own launch waits. The
        # decode programs' tokens never go through the device token relay (the worker returns host tensors), so the
        # relay's width stays 0.
        async_execution.configure(0, inline_results=True)
        return self.runtime

    def build_driver(self, runtime, layout: sp.DecodePageLayout, rank: int, device=None,
                     zero_blocks=None) -> sw.K3StepDriver:
        """The step driver of global rank ``rank`` (its replica ``rank // 32``, DCP rank ``rank % 32``)."""
        from sglang_neuron.models.kimi_k3.decode.layout import DP8_LAYOUT

        replica, cp_rank = DP8_LAYOUT.dp_rank(rank), DP8_LAYOUT.tp_rank(rank)
        seed = getattr(self.server_args, "random_seed", None)
        synthetic = sw.forced_acceptance(self.queries - 1, replica, seed, self.synthetic_env)
        from sglang_neuron.models.kimi_k3 import logits_dump

        dump = logits_dump.from_env()
        if dump is not None and not getattr(runtime.verify_program, "dump_logits", False):
            raise RuntimeError("The logits dump reads the verify's logits; build the runtime with the dump on.")
        return sw.K3StepDriver(runtime, runtime.config, layout, replica, cp_rank=cp_rank, device=device,
                               synthetic=synthetic, zero_blocks=zero_blocks, seed=seed,
                               max_model_len=getattr(self.server_args, "context_length", None), logits_dump=dump)

    def step_driver(self, required: bool = True) -> Optional[sw.K3StepDriver]:
        if self._driver is None and required:
            raise RuntimeError(f"{self.name}: the decode runtime is not built yet (SGLang's decode graph runner builds "
                               "it with the pools)")
        return self._driver

    def _runtime_for(self, phase: str):
        return self.runtime

    def bind_kv_cache(self, pool):
        raise RuntimeError(f"{self.name} binds its pools in _ensure_runtime (bind_pools)")

    def forward(self, *args, **kwargs):
        raise RuntimeError(f"{self.name} runs every step through its speculative worker (KimiK3DsparkWorker), not "
                           "the model runner's forward")


def _server_args():
    """The launch's resolved arguments, as SGLang's own readers see them: what the launch stated, then what resolution
    declared over it (the platform's defaults among them, models/kimi_k3/launch.apply_defaults). The published record's
    own fields keep the raw input."""
    from types import SimpleNamespace

    from sglang.srt.runtime_context import get_server_args

    return SimpleNamespace(**get_server_args().resolved_dict())


def build_decode_host(config, *args, **kwargs) -> nn.Module:
    """The checkpoint model class's decode host (``KimiK3ForConditionalGeneration.__new__`` on a decode role)."""
    return KimiK3DecodeHost(config, *args, **kwargs)
