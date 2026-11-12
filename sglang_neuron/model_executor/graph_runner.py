# SPDX-License-Identifier: Apache-2.0
"""The Neuron model runtime and SGLang's graph runner slot.

Neuron runs static shapes, one compiled graph per bucket. This module owns:

  * the buckets: the prefill token bucket, which is also the segmented-prefill KV tile, and the decode batch buckets,
    each over each decode context (BucketConfig.decode_contexts);
  * the step inputs: SGLang's exact metadata (block table, slot mapping, prefix) padded to a bucket and uploaded,
    including the sliding-window trim of the decode block table;
  * compile and warm-up: each bucket's graph compiles at its first warm-up call, under its family's compiler scope;
  * ``NeuronGraphRunner``: SGLang's decode graph path (can_run_graph / load_batch / execute), next to ``run_prefill``,
    which the model calls for extend batches;
  * a pipeline stage's prefill and decode: its layers, its boundary in from the previous stage's landing slots and out
    through claimed send slots of the PP seam (distributed/pp_seam.py), and the batch fingerprint each boundary is
    checked against.

SGLang's batch tensors live on the host. A step uploads its device inputs; an unchanged decode input reuses its device
copy, and a device token future passes through as it is.
"""

from __future__ import annotations

import contextlib
import logging
import os
import time
import weakref
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import torch
import torch.compiler

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.runner.base_runner import BaseRunner

from sglang_neuron import envs
from sglang_neuron.attention.backend import NeuronForwardMetadata
from sglang_neuron.model_executor.runtime_init import SCRATCHPAD_PAGE_MB, neuron_device

logger = logging.getLogger(__name__)


def pp_batch_fingerprint(md, token_ids, rids=None) -> int:
    """The batch identity two pipeline stages check each boundary against: phase, token count, cached prefix, each row's
    sequence length, input token ids and request ids. Each stage schedules on its own, and a boundary paired with
    another batch of the same row count would compute the wrong requests' tokens."""
    import zlib

    head = f"{int(md.is_prefill)}:{int(md.num_tokens)}:{int(md.cached_seq_len)}:".encode()
    ids = token_ids.reshape(-1)[: int(md.num_tokens)].to(torch.int64).contiguous().numpy().tobytes()
    requests = "\x00".join(rids or ()).encode()
    return zlib.crc32(head + md.seq_lens.to(torch.int64).contiguous().numpy().tobytes() + ids + requests)


def _output_tensor(out):
    """One output tensor of a step, for a host read that waits for it: the sampled tokens, a stage's boundary, or the
    host sampler's logits."""
    if isinstance(out, torch.Tensor):
        return out
    return out["token_ids"] if "token_ids" in out else next(iter(out.values()))


def _effective_temperatures(fb) -> Optional[torch.Tensor]:
    """Per-request temperature with greedy requests (SGLang: top_k == 1) mapped to 0."""
    si = getattr(fb, "sampling_info", None)
    if si is None or getattr(si, "temperatures", None) is None:
        return None
    t = si.temperatures.to("cpu", torch.float32).reshape(-1).clone()
    tk = getattr(si, "top_ks", None)
    if tk is not None:
        tk = tk.to("cpu").reshape(-1)
        t[tk[: t.numel()] == 1] = 0.0
    return t


P_MAX = 128  # attention kernel tile; a context must be a multiple of it


def _lnc_degree() -> int:
    """Logical NeuronCore degree of this process (2 for LNC2, 1 for LNC1). The attention kernels shard the prior over
    the LNC cores, so block tables align to LNC x P_MAX tokens."""
    from sglang_neuron.utils.hardware_config import get_logical_nc_config

    return int(get_logical_nc_config())


# The HBM scratchpad page (runtime_init.SCRATCHPAD_PAGE_MB): the compiler makes a private copy of any variable that
# crosses a page boundary, the aliased KV cache views included, so a page must be larger than the largest per-layer
# cache.
COMPILER_ARGS = [
    "--auto-cast=none",
    "--verbose=35",
    "-O1",
    "--internal-hlo2tensorizer-options=--modular-flow-mac-threshold=10",
    "--internal-backend-options=--enable-verifier=false",
    f"--hbm-scratchpad-page-size={SCRATCHPAD_PAGE_MB}",
]

# The live context of synthetic decode rows (warm-up, and an idle DP rank's dummy step), every page on the reserved
# block. A one-token context makes an idle DP rank's dummy step slower than the real ranks' steps, and every rank then
# waits for it.
DUMMY_LIVE_CTX = 1024

# torch-neuronx's backend option that makes a graph whose lowering or execution fails raise, where it would otherwise
# run the original GraphModule eagerly
_NO_EAGER_FALLBACK = {"fallback_execution": False}

# _check_boundary_in_slots' default: take this thread's writeback record itself
_TAKE = object()


def declared_in_place_targets(graph) -> Dict[str, torch.Tensor]:
    """The tensors a graph declares it writes in place outside its returned outputs, by name (its
    ``in_place_targets()``, e.g. KV caches a kernel writes directly), which the warm-up proves written in place; none
    for a graph that declares none."""
    declared = getattr(graph, "in_place_targets", None)
    return dict(declared()) if declared is not None else {}


def check_in_place(written, targets: Dict[str, torch.Tensor], kind: str) -> None:
    """Refuse a graph that did not write each of ``targets`` (name -> tensor) in place, from the executor's record of
    the call (executor_cache.take_writeback: each copy-back target, with whether its NEFF wrote it in place). Each
    target must be exactly one in-place entry: a target in no entry was never written by the NEFF (its rows would be
    stale), one copied back costs a device copy after every forward. The device harness checks its warm call with
    this too."""
    problems = []
    for name, target in targets.items():
        hits = [in_place for tensor, in_place in (written or ())
                if tensor.data_ptr() == target.data_ptr() and tuple(tensor.shape) == tuple(target.shape)]
        if not hits:
            problems.append(f"{name} (no NEFF write)")
        elif False in hits:
            problems.append(f"{name} (copied back, not written in place)")
        elif len(hits) > 1:
            problems.append(f"{name} (written {len(hits)} times)")
    if problems:
        raise RuntimeError(f"the {kind} graph did not write {len(problems)} of the tensors it declares in place: "
                           f"{', '.join(problems)}; their rows would be stale, or need a device copy after every "
                           "forward")


def _round_up(x: int, m: int) -> int:
    return (x + m - 1) // m * m


@dataclass
class BucketConfig:
    block_size: int
    # the whole context in whole kernel tiles of the LNC cores, over which the decode kernel splits a batch's prior: the
    # largest decode context and the prefill table's width
    max_model_len: int
    prefill_token_buckets: List[int]  # supported KV segment tiles, the largest the chunk; none on a decode role
    num_seqs_buckets: List[int]
    # the contexts the decode graphs cover, the largest the whole context: a decode step takes the smallest that holds
    # its longest row, since the decode attention kernel reads the whole context of its graph
    decode_contexts: List[int]
    # --dcp-size: a table column is a virtual page of kv_shards blocks, one per DCP rank
    # (distributed.decode_context_parallel)
    kv_shards: int = 1
    # the decode kernel's DCP context rule, (context, size, block, tile) -> the context a DCP decode graph needs
    # (NeuronModelRuntime.dcp_decode_context); None for decode_context_parallel.decode_context
    dcp_context_rule: Optional[Callable] = field(default=None, compare=False, repr=False)

    @classmethod
    def build(cls, *, block_size: int, max_model_len: int, attn_tp_size: int, max_running_requests: int,
              chunked_prefill_size: int, prefill: bool, decode_batch_sizes=None, prefill_token_counts=None,
              prefill_ladder: bool = True, decode_context_ladder: bool = False,
              kv_shards: int = 1, dcp_context_rule: Optional[Callable] = None) -> "BucketConfig":
        """The buckets of a launch, from its flags. A prefill runs a graph of the chunk (--chunked-prefill-size) and
        one of each smaller token count SGLang's --cuda-graph-bs-prefill lists, or without it of each smaller KV
        segment tile the attention TP splits, so the last piece of a prompt runs the smallest that holds it. A model
        whose prefill takes the chunk alone (``prefill_ladder`` false) runs the chunk's graph only. A decode runs one
        graph per batch size SGLang resolved (--cuda-graph-bs-decode, else the platform's default) up to the per-DP
        batch, and one of the per-DP batch itself, each over the whole context (--context-length), and with
        ``decode_context_ladder`` also over its quarter and its half. A disaggregated decode (``prefill`` false) has no
        prefill bucket. Under DCP (``kv_shards`` above 1) each context becomes its DCP decode context, by the decode
        kernel's rule (``dcp_context_rule``, default decode_context_parallel.decode_context)."""
        from sglang_neuron.utils.bucket_utils import SUPPORTED_KV_SEGMENT_SIZES

        tile = _lnc_degree() * P_MAX
        max_model_len = _round_up(max_model_len, tile)
        contexts = sorted({_round_up(max_model_len // q, tile) for q in ((4, 2, 1) if decode_context_ladder else (1,))})
        if kv_shards > 1:
            from sglang_neuron.distributed.decode_context_parallel import decode_context

            rule = dcp_context_rule or decode_context
            contexts = sorted({rule(c, kv_shards, block_size, 2 * tile) for c in contexts})
            max_model_len = contexts[-1]

        def fits(tokens):
            # whole P_MAX attention tiles that split evenly over the attention TP ranks, at least two tokens a rank
            return tokens > attn_tp_size and tokens % attn_tp_size == 0 and tokens % P_MAX == 0

        if not prefill:
            prefill_buckets = []
        elif not fits(chunked_prefill_size):
            raise ValueError(f"--chunked-prefill-size {chunked_prefill_size} is not a prefill bucket: it must be a "
                             f"multiple of {P_MAX} and of the attention TP size ({attn_tp_size}), and above it")
        elif prefill_token_counts is None:
            prefill_buckets = sorted(t for t in SUPPORTED_KV_SEGMENT_SIZES
                                     if prefill_ladder and t < chunked_prefill_size and fits(t))
            prefill_buckets.append(chunked_prefill_size)
        else:
            # counts above the chunk never run: every forward is at most one chunk
            below = sorted({int(t) for t in prefill_token_counts if int(t) < chunked_prefill_size})
            if below and not prefill_ladder:
                raise ValueError(f"--cuda-graph-bs-prefill {below}: this model prefills in whole chunks of "
                                 f"--chunked-prefill-size {chunked_prefill_size}; launch without the flag")
            bad = [t for t in below if t not in SUPPORTED_KV_SEGMENT_SIZES or not fits(t)]
            if bad:
                raise ValueError(f"--cuda-graph-bs-prefill {bad}: a prefill token count must be a KV segment tile "
                                 f"{sorted(SUPPORTED_KV_SEGMENT_SIZES)}, a multiple of the attention TP size "
                                 f"({attn_tp_size}) and above it")
            prefill_buckets = below + [chunked_prefill_size]
        return cls(block_size, max_model_len, prefill_buckets,
                   decode_batch_buckets(decode_batch_sizes, max_running_requests), contexts, kv_shards,
                   dcp_context_rule)

    def prefill_bucket(self, n: int) -> int:
        for b in self.prefill_token_buckets:
            if n <= b:
                return b
        if not self.prefill_token_buckets:
            raise RuntimeError(f"a prefill of {n} tokens on a disaggregated decode, which has no prefill graph")
        raise ValueError(f"prefill chunk of {n} tokens exceeds largest bucket {self.prefill_token_buckets[-1]}")

    def seqs_bucket(self, bs: int) -> int:
        for b in self.num_seqs_buckets:
            if bs <= b:
                return b
        raise ValueError(f"batch size {bs} exceeds largest bucket {self.num_seqs_buckets[-1]}")

    def decode_context(self, tokens: int) -> int:
        """The smallest decode context that holds a row of ``tokens`` tokens; under DCP, the table the decode kernel's
        rule needs for the row at a tile of one token (decode_context_parallel.decode_context: one virtual page past the
        row's last token)."""
        if self.kv_shards > 1:
            from sglang_neuron.distributed.decode_context_parallel import decode_context

            tokens = (self.dcp_context_rule or decode_context)(tokens, self.kv_shards, self.block_size, 1)
        for c in self.decode_contexts:
            if tokens <= c:
                return c
        raise ValueError(f"a decode row of {tokens} tokens exceeds the context {self.decode_contexts[-1]}")

    def max_blocks(self, context: Optional[int] = None) -> int:
        """Block table columns of ``context`` tokens, by default the longest context (virtual pages under DCP)."""
        return (context or self.max_model_len) // (self.block_size * self.kv_shards)


def decode_batch_buckets(listed, max_running_requests: int) -> List[int]:
    """The decode graphs' batch sizes: those of ``listed`` (SGLang's resolved decode batch sizes) up to the per-DP
    batch, as SGLang caps its own captured sizes, and the batch itself, so every admitted batch has a graph."""
    return sorted({int(b) for b in listed or () if 0 < int(b) <= max_running_requests} | {max_running_requests})


def check_prefill_tiles(token_buckets) -> None:
    """The plugin's segmented prefill kernel takes the prefill token bucket as its KV segment tile, and it compiles
    silently and generates garbage on a tile it does not support: refuse such a bucket before compiling."""
    from sglang_neuron.utils.bucket_utils import SUPPORTED_KV_SEGMENT_SIZES

    bad = sorted({tb for tb in token_buckets if tb not in SUPPORTED_KV_SEGMENT_SIZES})
    if bad:
        raise ValueError(
            f"prefill token bucket(s) {bad} are not supported KV segment tiles {sorted(SUPPORTED_KV_SEGMENT_SIZES)}: "
            "the segmented prefill kernel compiles silently and generates garbage on an unsupported tile; choose the "
            "chunk from that set (--chunked-prefill-size)"
        )


class NeuronModelRuntime:
    """Compiles and runs one graph family of a Neuron model for SGLang batches."""

    # the family's direct submit (backend.direct_submit), which build_compiled makes for a megakernel program that
    # declares it (megakernel.PhaseProgram.direct_submit, set here by NeuronModel._build_runtimes)
    direct_submit = None
    megakernel_direct_submit = False
    # SGLang's hybrid SWA window pool (mem_cache.hybrid_swa), which the sliding layers' tables and slots map into
    # (None on the one-pool layout)
    window_pool = None
    # the checkpoint's vocabulary size, which every sampled id is below (async_execution.refuse_invalid_token_ids), set
    # at construction (None, on a runtime not built from a model, bounds the ids by their sign alone)
    vocab_size = None

    # KV token context parallelism (set in __init__): the degree, this rank's place in it and its block of a page; a
    # runtime without it addresses a page as one block of its own
    kv_cp_size = 1
    kv_cp_rank = 0
    kv_cp_interleave = None
    # a megakernel runtime that takes a decode step's ids on the host (it embeds them there, or plans its calls from
    # them): under async execution its decode reads the ids from the previous step's lagged readback (async_execution's
    # host-ids mode, configure(host_ids=)) instead of the device token relay, and it has no fixed decode token width
    host_input_ids = False
    # a megakernel runtime whose idle calls (a DP replica's padding of the cohort's program) are launched and never read
    # under async execution, as a host-ids runtime's are
    idle_unread = False

    # The decode kernel's DCP limits (distributed.decode_context_parallel): the context a DCP decode graph needs, and
    # the largest decode batch its DCP path holds. Both are the per-layer decode attention kernel's rules; a model's
    # runtime_class whose decode runs another kernel states that kernel's own by overriding them. __init__ reads both.
    def dcp_decode_context(self, context: int, size: int, block: int, tile: int) -> int:
        from sglang_neuron.distributed.decode_context_parallel import decode_context

        return decode_context(context, size, block, tile)

    def check_dcp_decode_batch(self, q_heads_per_rank: int, size: int, batch: int) -> None:
        from sglang_neuron.distributed.decode_context_parallel import check_dcp_decode_batch

        check_dcp_decode_batch(q_heads_per_rank, size, batch)

    def __init__(self, model_runner, sgl_model, phase=None):
        self.model_runner = model_runner
        self.sgl_model = sgl_model
        if phase is not None:
            # an aggregated server: this runtime owns one phase's graph family
            self.model_graph = sgl_model.model_graphs[phase]
            self.execution_model = sgl_model.execution_models[phase]
        else:
            self.model_graph = sgl_model.model_graph
            self.execution_model = sgl_model.execution_model
        self.phase_override = phase
        self.hf_config = sgl_model.hf_config
        self.vocab_size = int(self.hf_config.vocab_size)
        from sglang_neuron.models import contract as model_contract

        # a graph that names its layers declares their names and windows (layer_names, sliding_windows); otherwise
        # layer_types decides each layer's window (a config without it is full attention everywhere; each model's host
        # refuses a sliding config it does not serve), and any other attention kind is refused
        architecture = type(sgl_model).__name__
        declares = [name for name in ("layer_names", "sliding_windows") if hasattr(self.model_graph, name)]
        if len(declares) == 1:
            raise ValueError(f"{architecture}'s graph declares {declares[0]} alone; a graph that names its layers "
                             "declares both layer_names and sliding_windows")
        # a hybrid graph that names its layers also declares their recurrent-state families (state_families)
        layers = (model_contract.declared_layer_kinds(self.hf_config, architecture, self.model_graph.layer_names,
                                                      self.model_graph.sliding_windows,
                                                      getattr(self.model_graph, "state_families", None))
                  if declares else None)
        self.contract = model_contract.derive(self.hf_config, architecture=architecture, layers=layers)
        # recurrent layers: the model's state declaration covers exactly them, and it is the one SGLang's state pool
        # was built from (mem_cache/state_pool.check_model_state)
        from sglang_neuron.mem_cache import state_pool

        state_pool.check_model_state(self.contract, getattr(sgl_model, "state_spec", None),
                                     getattr(model_runner, "req_to_token_pool", None))
        self.layer_names = self.contract.layer_names()
        self.sliding = self.contract.sliding_windows()
        from sglang_neuron.mem_cache import hybrid_swa

        self.window_pool = hybrid_swa.window_pool(model_runner.token_to_kv_pool)
        if self.window_pool is not None:
            hybrid_swa.check_window_pool(model_runner, max(w for w in self.sliding if w is not None))
        self._init_stage(sgl_model, phase)
        from sglang.srt.distributed.parallel_state import get_attn_tp_group, get_tp_group

        # engine-wide group: barriers and the cross-rank compile check
        self.tp_group = get_tp_group()
        self.tp_size = self.tp_group.world_size
        self.tp_rank = self.tp_group.rank_in_group
        # the model's SPMD rank input is its attention-TP rank
        self.model_tp_rank = get_attn_tp_group().rank_in_group
        self.model_tp_size = get_attn_tp_group().world_size
        self.rank_tensor = torch.tensor([self.model_tp_rank], dtype=torch.int32, device=neuron_device())
        sa = model_runner.server_args
        role = str(getattr(sa, "disaggregation_mode", "null"))
        # graph families: a disaggregated role builds its own phase only; an aggregated server builds one runtime per
        # phase and pins each here
        self.graphs = phase or role
        if self.graphs not in ("prefill", "decode"):
            raise ValueError(f"a runtime runs the prefill or the decode graphs, not {self.graphs!r}: an aggregated "
                             "server builds one runtime per phase")
        self.disaggregated_decode = role == "decode"
        model_phase = getattr(sgl_model, "phase", None)
        if model_phase is not None and model_phase != "all" and self.graphs != model_phase:
            raise ValueError(f"Runtime graphs={self.graphs} do not match model role={model_phase}")
        from sglang_neuron import serving_policy, sglang_tree

        self.dp_lockstep = sglang_tree.attention_dp_size(sa) > 1
        self.async_execution = serving_policy.current().async_execution
        # set by managers.dp_identity once its static proof holds: the decode bucket every DP rank dispatches without
        # exchanging per-step metadata
        self.dp_identity_bucket: Optional[int] = None
        from sglang.srt.arg_groups.overrides import resolution_result

        # the chunk the scheduler runs: under DP attention SGLang resolves --chunked-prefill-size / attn_dp_size
        chunk = int(resolution_result(sa, "chunked_prefill_size", 0) or 0)
        # SGLang's resolved decode batch sizes: --cuda-graph-bs-decode, or without it the platform's default
        # (platform.default_decode_batch_sizes)
        graph_config = resolution_result(sa, "cuda_graph_config", None)
        from sglang_neuron.distributed.decode_context_parallel import runtime_layout

        # this rank's DCP placement, or None (PD decode only: validate_dcp_launch refuses anything else at launch)
        self.dcp = runtime_layout(model_runner.page_size)
        self.buckets = BucketConfig.build(
            block_size=model_runner.page_size,
            max_model_len=model_runner.model_config.context_len,
            attn_tp_size=self.model_tp_size,
            max_running_requests=int(model_runner.max_running_requests or 64),
            chunked_prefill_size=chunk,
            # an aggregated server plans its prefill steps in either phase's runtime
            prefill=role != "decode",
            decode_batch_sizes=None if graph_config is None else graph_config.decode.bs,
            prefill_token_counts=resolution_result(sa, "cuda_graph_bs_prefill", None),
            prefill_ladder=sgl_model.prefill_ladder,
            # the DP ranks of a lockstep decode share no row lengths, so they could not agree on a context rung
            decode_context_ladder=sgl_model.decode_context_ladder and not self.dp_lockstep,
            kv_shards=1 if self.dcp is None else self.dcp.size,
            dcp_context_rule=None if self.dcp is None else self.dcp_decode_context,
        )
        if self.dcp is not None and role != "prefill":
            # the decode kernel's DCP batch tile; a DCP prefill role runs no decode graph
            self.check_dcp_decode_batch(int(model_runner.model_config.num_attention_heads) // self.model_tp_size,
                                        self.dcp.size, self.buckets.num_seqs_buckets[-1])
        self.prefill_holds_prior = sgl_model.prefill_holds_prior
        self.block_size = self.buckets.block_size
        # KV token context parallelism, which the model declares (models.base.NeuronModel.kv_cp_size, C): a page of
        # block_size tokens is one kv_cp_interleave-token block on each of the C attention-TP ranks, rank r holding
        # the page's tokens [r * interleave, (r + 1) * interleave); without it (C = 1) a page is one block of this rank's.
        # Served on the PD prefill role, and on the PD decode role of a model that serves a CP decode (the Qwen3.5
        # giga decode, models/qwen35: its kernel reads every rank's block of a page through the shared block table and
        # merges the ranks' split-K partials): a plain decode graph addresses a page as one block of the rank's, and
        # an aggregated server's decode reads the pages its own prefill writes. Refused besides: a degree that does
        # not divide the page or span the attention-TP group, the hybrid SWA pool (the CP prefill indexes the sliding
        # layers with full-pool pages), and a KV pool whose published layout (kv_cp_size, kv_cp_rank,
        # kv_cp_interleave; a pool that publishes none has no CP) is not the one the model declares.
        self.kv_cp_size = int(getattr(sgl_model, "kv_cp_size", 1) or 1)
        if self.block_size % self.kv_cp_size:
            raise ValueError(f"page size {self.block_size} is not a multiple of the KV CP degree {self.kv_cp_size}")
        self.kv_cp_interleave = self.block_size // self.kv_cp_size
        self.kv_cp_rank = self.model_tp_rank if self.kv_cp_size > 1 else 0
        if self.kv_cp_size > 1:
            if self.model_tp_size != self.kv_cp_size:
                raise ValueError(f"KV CP degree {self.kv_cp_size} must span the attention TP group "
                                 f"({self.model_tp_size})")
            decode_cp = (self.graphs == "decode" and self.phase_override is None
                         and bool(getattr(sgl_model, "kv_cp_decode", False)))
            if not decode_cp and (self.graphs != "prefill" or self.phase_override is not None):
                where = f"{self.graphs} graphs" + (" of an aggregated server" if self.phase_override else "")
                raise ValueError("KV context parallelism is served on the PD roles (a CP prefill, or a model's own "
                                 f"CP decode), not with {where}")
            if self.window_pool is not None:
                raise ValueError("KV context parallelism with the hybrid SWA pool: the CP prefill's inputs index the "
                                 "sliding layers with full-pool pages; launch with --disable-hybrid-swa-memory")
        kv_pool = getattr(model_runner, "token_to_kv_pool", None)
        kv_pool = getattr(kv_pool, "full_kv_pool", kv_pool)
        served_cp = int(getattr(kv_pool, "kv_cp_size", 1) or 1)
        served = ((served_cp, int(getattr(kv_pool, "kv_cp_rank", 0) or 0),
                   int(getattr(kv_pool, "kv_cp_interleave", 0) or 0))
                  if served_cp > 1 else (1, 0, self.block_size))
        if served != (self.kv_cp_size, self.kv_cp_rank, self.kv_cp_interleave):
            raise ValueError(f"the model declares KV CP degree {self.kv_cp_size} (rank {self.kv_cp_rank}, "
                             f"{self.kv_cp_interleave}-token blocks) but its KV pool ({type(kv_pool).__name__}) serves "
                             f"degree {served[0]} (rank {served[1]}, {served[2]}-token blocks)")
        from sglang_neuron.sglang_patches import prefill_chunk_grid

        chunk_grid = prefill_chunk_grid.kept()
        if chunk_grid and self.graphs == "prefill":
            largest = self.buckets.prefill_token_buckets[-1]
            if chunk != largest:
                flag = largest * sglang_tree.attention_dp_size(sa)
                raise ValueError(
                    f"--chunked-prefill-size and the prefill token buckets disagree: the chunk ({chunk}) must equal "
                    f"the largest prefill bucket ({largest}) wherever a prefill runs with a radix cache or chunked "
                    f"prefill, so a cold prompt runs one compiled tile per chunk; launch with --chunked-prefill-size "
                    f"{flag}{' (the chunk times the attention DP size)' if flag != largest else ''}")
        # The segmented kernel reads a forward's own blocks [P / bs, P / bs + T / bs) and pads the table it is given
        # only to the next multiple of T / bs, skipping any column past that pad: a hit on SGLang's page-keyed tree
        # whose prior P lies within T of the longest context would read columns that do not exist, so with that tree
        # the table has T / bs more columns (-1, like every column past a request's pages). The chunk-keyed tree
        # (--radix-cache-backend neuron-chunk) only has priors that are multiples of T, so it needs no extra columns.
        self.prefill_table_pad = bool(chunk_grid and self.graphs == "prefill"
                                      and not serving_policy.current().chunk_keyed_tree)
        # an idle DP rank's decode dummy completes through SGLang's result queue like a real decode step, on an
        # asynchronous DP-lockstep decode role
        self.async_idle_completion = bool(
            self.dp_lockstep and phase is None and self.graphs == "decode" and self.async_execution)
        from sglang_neuron.sampling import HOST_SAMPLER, sampler_mode

        # the launch checked the mode's settings (platform._check_plugin_env, check_host_sampler); the model must serve
        # a device mode. A stage before the last samples nothing, so its graph takes no sampling inputs
        self.sampler_mode = sampler_mode() if self.pp_is_last_stage else "greedy"
        if self.sampler_mode not in ("greedy", HOST_SAMPLER):
            supported = getattr(self.execution_model, "sampler_modes", ("greedy",))
            if self.sampler_mode not in supported:
                raise ValueError(f"SGLANG_NEURON_SAMPLER={self.sampler_mode}: this model's execution supports "
                                 f"{tuple(supported)}")
        self._init_decode_token_io()
        from sglang_neuron.model_executor import async_execution

        # the device token relay of this scheduler process plans its decode batches at the token I/O of the runtime
        # that serves them (an aggregated server's decode-phase runtime, not its prefill one)
        if self.graphs != "prefill":
            # a pipelined PD decode reads its results out of order, on the copy thread; its first stage takes tokens
            # from the last stage's relay and the others embed nothing. A pipelined aggregated server relays no tokens:
            # SGLang's PP loop hands them to every stage on the host
            stage_role = None if not self.pp_token_relay else "first" if self.pp_is_first_stage else "other"
            async_execution.configure(self.decode_token_width, inline_results=role == "decode" and not self.pp_active,
                                      pp=stage_role, host_ids=self.async_execution and self.host_input_ids)
        # every rank of the engine agrees on the contract's digest (models/contract.py) before it builds a graph, so a
        # rank that resolved another layout or model setting (NeuronModel.fleet_digest) fails here instead of desyncing
        # a collective while serving
        import dataclasses

        self.contract = dataclasses.replace(
            self.contract, phase=self.graphs,
            parallel_digest=(self.tp_size, self.model_tp_size, int(self.dp_lockstep), self.block_size,
                             self.sampler_mode, 1 if self.dcp is None else self.dcp.size, self.kv_cp_size,
                             self.pp_start_layer,
                             self.pp_end_layer, int(self.pp_is_last_stage),
                             *getattr(sgl_model, "fleet_digest", tuple)()))
        from sglang_neuron.models.contract import assert_fleet_consistent

        # the TP group is the stage's under PP: its ranks agree on the stage's layers too
        assert_fleet_consistent(self.contract, self.tp_group)
        self.compiled = None
        self._warmed = False
        self._decode_sampling_positions = {}
        self._decode_cached_zero = None
        self._token_identity_control = None
        self._token_placeholder = None
        # a pipelined decode's last stage: each token send slot's latest pending sample (_run_relaying)
        self._relay_samples = {}
        # diagnostics read once rather than on every step: the serving device profile window and the host trace
        self._profile_serve = bool(os.environ.get("SGLANG_NEURON_PROFILE_SERVE_RANKS"))
        self._host_trace = bool(envs.host_trace_dir())
        if self.pp_active:
            # collective over the world: every rank builds its runtime here, in the same order
            from sglang_neuron.distributed.pp_seam import get_seam

            get_seam(self, check=True)

    def _init_stage(self, sgl_model, phase) -> None:
        """The pipeline stage the model built (models.base.NeuronModel): its global layers [start, end), whose names key
        the attention metadata and the KV, and its boundary. Without PP the model is one stage of every layer."""
        self.pp_start_layer = int(getattr(sgl_model, "start_layer", 0) or 0)
        self.pp_end_layer = int(getattr(sgl_model, "end_layer", None) or len(self.layer_names))
        self.pp_is_first_stage = bool(getattr(sgl_model, "is_first_stage", True))
        self.pp_is_last_stage = bool(getattr(sgl_model, "is_last_stage", True))
        self.pp_active = not (self.pp_is_first_stage and self.pp_is_last_stage)
        self._seam_recv_live = False
        self.pp_token_relay = False
        self.pp_prefill_kinds = None
        if not self.pp_active:
            return
        stage_phase = phase or getattr(sgl_model, "phase", None)
        if stage_phase not in ("prefill", "decode"):
            raise ValueError("a pipeline stage runs the prefill or the decode graph of a pipelined PD role, or one of "
                             "them on a pipelined aggregated server")
        self.layer_names = self.layer_names[self.pp_start_layer:self.pp_end_layer]
        self.sliding = self.sliding[self.pp_start_layer:self.pp_end_layer]
        self.pp_prefill_boundary = tuple(sgl_model.pp_prefill_boundary)
        self.pp_decode_boundary = tuple(sgl_model.pp_decode_boundary)
        # a model that declares its prefill boundary's kinds names the boundary there (distributed.pp_seam.BoundaryKind)
        declared = getattr(sgl_model, "pp_prefill_boundary_kinds", None)
        if declared is not None:
            from sglang_neuron.distributed.pp_seam import check_prefill_kinds

            self.pp_prefill_kinds = check_prefill_kinds(declared, first=self.pp_is_first_stage,
                                                        last=self.pp_is_last_stage,
                                                        architecture=type(sgl_model).__name__)
            self.pp_prefill_boundary = tuple(self.pp_prefill_kinds)
        from sglang_neuron import serving_policy

        # a pipelined PD decode's first stage has no source of input ids but the last stage's relay over the seam. An
        # aggregated server's decode runtime (``phase``) gets them on the host from SGLang's PP loop
        self.pp_token_relay = bool(serving_policy.current().pp_token_relay)
        if stage_phase == "decode" and phase is None and not self.pp_token_relay:
            raise ValueError("a pipelined decode stage runs with the sampled-token relay (ServingPolicy.pp_token_relay)")
        self.pp_boundary_dtype = sgl_model.pp_boundary_dtype
        self.pp_hidden_size = int(sgl_model.pp_hidden_size or 0)
        # the one dtype and width of every boundary tensor the model does not declare a kind for: its decode boundary,
        # and its prefill boundary unless declared
        undeclared = declared is None or stage_phase == "decode" or phase is not None
        if undeclared and (self.pp_boundary_dtype is None or not self.pp_hidden_size):
            raise ValueError(f"{type(sgl_model).__name__} builds pipeline stages without declaring its boundary's "
                             "dtype and width (pp_boundary_dtype, pp_hidden_size)")
        # each boundary tensor's (row width, dtype): the hidden-state row unless the model names another
        # (ModelHost.pp_boundary_spec); none for a declared prefill boundary, whose kinds carry their own
        spec = getattr(sgl_model, "pp_boundary_spec", None)
        names = set(self.pp_prefill_boundary) | set(self.pp_decode_boundary)
        self.pp_boundary_specs = {} if not undeclared else {
            name: (tuple(spec(name)) if spec is not None else (self.pp_hidden_size, self.pp_boundary_dtype))
            for name in names}
        if self.pp_prefill_kinds is None:
            # one scheme for every model: a model that declares no kinds gets each name's fixed kind, SP-sharded rows
            # of its width (pp_boundary_specs), one row block each way, resolved here once (pp_seam.prefill_boundary)
            from sglang_neuron.distributed.pp_seam import default_prefill_kinds

            self.pp_prefill_kinds = default_prefill_kinds(self.pp_prefill_boundary, self.pp_hidden_size,
                                                          self.pp_boundary_dtype, self.pp_boundary_specs)

    def _init_decode_token_io(self) -> None:
        """Fixed-width decode token I/O with the in-graph token reorder, on every asynchronous decode runtime (a PD
        decode's, an aggregated server's decode phase): the width is the largest seqs bucket. Every decode that
        samples then returns the full width, so a batch below its bucket, or one that changed, still launches before
        the previous step's tokens reach the host."""
        on = self.graphs == "decode" and self.async_execution and not self.host_input_ids
        self.decode_token_width = int(self.buckets.num_seqs_buckets[-1]) if on else 0
        # a PP decode stage after the first embeds nothing and takes the placeholder ids as they are
        self.decode_token_assemble = not self.pp_active or self.pp_is_first_stage
        if self.decode_token_width:
            from sglang_neuron.model_executor.token_io import FixedDecodeTokenIO

            # under PP the first stage assembles its ids from the last stage's relayed tokens and the last stage samples
            # into the width it relays; a middle stage has no token I/O
            samples = not self.pp_active or self.pp_is_last_stage
            if self.decode_token_assemble or samples:
                self.execution_model = FixedDecodeTokenIO(self.execution_model, self.decode_token_width,
                                                          assemble=self.decode_token_assemble, samples=samples)

    def _compile_options(self):
        from sglang_neuron import serving_policy

        # the launch's scratchpad page, which the runtime bills in too (runtime_init.runtime_defaults)
        page = f"--hbm-scratchpad-page-size={serving_policy.current().scratchpad_page_mb}"
        compiler_args = [page if arg.startswith("--hbm-scratchpad-page-size=") else arg for arg in COMPILER_ARGS]
        customize = getattr(self.sgl_model, "neuron_compiler_args", None)
        if customize is not None:
            compiler_args = customize(self.graphs, compiler_args)
        return {"compiler_args": compiler_args}

    def build_compiled(self):
        import torch._dynamo

        torch._dynamo.config.cache_size_limit = 2**62
        torch._dynamo.config.accumulated_cache_size_limit = 2**62
        options = self._compile_options()
        # a megakernel's decode program that declares the direct submit (megakernel.PhaseProgram.direct_submit) on any
        # server, an aggregated one too: its calls differ between steps only in tensors (what the direct submit, which
        # takes tensor guards only, relies on)
        declared_direct = self.graphs == "decode" and getattr(self, "megakernel_direct_submit", False)
        if self.disaggregated_decode or declared_direct:
            # Every step re-evaluates dynamo's guards over a kwargs tree with one metadata dict per layer. A decode
            # role's calls differ between steps only in tensors, so keeping the tensor guards loses no check that
            # could fail, and the dictionary guards are cached by tag instead of walked.
            torch._dynamo.config.use_recursive_dict_tags_for_guards = True
            options = {**options, "guard_filter_fn": torch.compiler.keep_tensor_guards_unsafe}
        from sglang_neuron.backend import stack
        from sglang_neuron.backend import envs as backend_envs

        # torch-neuronx reads the compiler arguments from the process environment when a graph compiles, so the calls
        # that compile this family run under stack.compile_scope(self.compiler_args)
        self.compiler_args = options.pop("compiler_args")
        cpu_mode = backend_envs.SGLANG_NEURON_CPU_MODE
        if not cpu_mode:
            options = {**options, **_NO_EAGER_FALLBACK}
        # torch.compile pops the guard filter out of the options it is given, and the direct submit below must see it
        guard_filter_fn = options.get("guard_filter_fn")
        # every warmed bucket is its own static graph (dynamic=False): dynamo would otherwise mark a dim dynamic once
        # another family's call through the same code saw another size, and torch-neuronx's NKI wrapper refuses
        # symbolic shapes
        self.compiled = torch.compile(self.execution_model, backend=stack.compile_backend_name(cpu_mode=cpu_mode),
                                      fullgraph=True, dynamic=False, options=options)
        # the ranks meet before a model execution whose NEFF differs from the last one (program_fence), in place of the
        # runtime's per-execution barrier: on an aggregated server, and wherever this runtime has more than one shape
        from sglang_neuron.model_executor import program_fence

        shapes = len(self._warm_shapes())
        self.compiled = program_fence.wrap_compiled(self.compiled, programs=shapes)
        # a megakernel's decode program that declares it hands its steps straight to the bound executor, past the
        # layers above it that only cost host time on a graph of warmed shapes (backend.direct_submit)
        self.direct_submit = None
        if not cpu_mode and self.graphs == "decode" and getattr(self, "megakernel_direct_submit", False):
            from sglang_neuron.backend import direct_submit

            self.direct_submit = direct_submit.for_runtime(
                self.execution_model, guard_filter_fn=guard_filter_fn, shapes=shapes, name="megakernel decode graph")
        # a direct step's execution is a model execution of the program-switch fence like a compiled call's (a direct
        # step calls no compiled graph, so it enters the fence's model call itself): on an aggregated server a prefill
        # after direct decode steps must still meet the ranks at the switch
        fence = getattr(self.compiled, "program_switch_fence", None)
        self.direct_launch = None
        if self.direct_submit is not None:
            direct = self.direct_submit
            launch = lambda flat: direct.submit(flat)  # noqa: E731
            self.direct_launch = fence.model_call(launch) if fence is not None else launch

    def _warm_shapes(self) -> list:
        """The graphs warmup_all compiles and warms, as (kind, bucket, context) for _synthetic_inputs: the
        segmented-prefill graph of each token bucket (kv_segment_size is the fixed prior-KV tile, equal to the bucket;
        cached_seq_len is a runtime tensor, so fresh and prefix-cached prefills share the NEFF) and the decode graph of
        each batch bucket over each decode context."""
        shapes = []
        if self.graphs == "prefill":
            shapes += [("prefill", tb, None) for tb in self.buckets.prefill_token_buckets]
        else:
            shapes += [("decode", bs, c) for bs in self.buckets.num_seqs_buckets for c in self.buckets.decode_contexts]
        return shapes

    def warmup_all(self):
        """barrier -> warm each shape (it compiles at its first call) -> cross-rank NEFF check -> barrier, and on a
        pipelined server a barrier of every stage last.

        Runs under SGLang's DynamicGradMode so dynamo's global-state guards
        (grad mode) match the scheduler loop that later executes the graphs."""
        from sglang.srt.utils.common import DynamicGradMode

        with DynamicGradMode():
            self._warmup_all()

    def _warmup_all(self):
        from sglang_neuron.backend import envs as backend_envs
        from sglang_neuron.backend import executor_cache, spmd_check, stack
        from sglang_neuron.parallel.state import tp_barrier, world_barrier

        t0 = time.time()
        shapes = self._warm_shapes()
        if self.sgl_model.segmented_prefill:
            check_prefill_tiles(tb for kind, tb, _ in shapes if kind == "prefill")
        tp_barrier()
        # the ranks compare each executor's NEFF before its first submit (spmd_check.note_first_call)
        group = None
        if not backend_envs.SGLANG_NEURON_CPU_MODE:
            from sglang.srt.distributed import parallel_state as sgl_ps

            group = sgl_ps.get_tp_group().cpu_group
        for kind, bucket, context in shapes:
            logger.info("[neuron] warm %s bucket=%d%s", kind, bucket, f" context={context}" if context else "")
            kwargs = self._synthetic_inputs(kind, bucket, context)
            # torch-neuronx compiles the shape's graph at this call; its executors' NEFFs belong to the shape
            with stack.compile_scope(self.compiler_args), \
                    spmd_check.warming(f"{self.graphs}:{kind}:{bucket}:{context}", group, self.tp_size):
                # the direct submit's first bind call (the first serving step is its second)
                direct = getattr(self, "direct_submit", None)
                slots = ({k: v for k, v in kwargs.items() if k.endswith("_out")}
                         if getattr(self, "pp_active", False) else None)
                targets = declared_in_place_targets(getattr(self, "model_graph", None))
                armed = bool(slots or targets) and not backend_envs.SGLANG_NEURON_CPU_MODE
                if armed:
                    executor_cache.arm_writeback()  # the warm call's own record (_check_boundary_in_slots)
                out = self.compiled(**kwargs) if direct is None else direct.full(self.compiled, kwargs)
                # the first execution (its NEFF load on torch-neuronx's runtime worker) finishes under the compiler
                # environment that compiled it, and the scope restores the environment with nothing in flight
                _ = _output_tensor(out).cpu()
            written = executor_cache.take_writeback() if armed else None
            if slots:
                # the stage graph writes its boundary into the seam slots in place: refused at warm-up, not serving
                self._check_boundary_in_slots(out, slots, kind, written=written)
            if targets and armed:
                # and the tensors the graph declares it writes (a model's KV caches beside its kernels' outputs)
                check_in_place(written, targets, kind)
        if group is not None:
            # every shape warmed with a recorded NEFF on every rank, one program per shape on all of them
            spmd_check.check(group, self.tp_size)
            after_warmup = getattr(self.sgl_model, "neuron_after_warmup", None)
            if after_warmup is not None:
                after_warmup(self.graphs, spmd_check.warmed_keys())
        tp_barrier()
        if getattr(self, "pp_active", False):
            # Each stage warms its own graphs, some for minutes longer than others. A stage done first would block in
            # its scheduler loop on the stage before it with SGLang's 300 s watchdog armed, so every stage starts
            # serving together.
            world_barrier()
        self._warmed = True
        logger.info("[neuron] warmup done in %.1fs (%d shapes)", time.time() - t0, len(shapes))

    def _synthetic_inputs(self, kind: str, bucket: int, context: Optional[int] = None) -> dict:
        """Warm-up and idle-rank dummy inputs. KV writes land only in SGLang's reserved page 0 (slot 0) and reads only
        touch block 0, so a dummy never disturbs live requests."""
        if kind == "prefill":
            T = bucket
            # the table of the warmed context bucket (the server's longest context by default): a wider one would hold
            # more pages than that bucket's graph takes
            md = NeuronForwardMetadata(
                is_prefill=True, batch_size=1, num_tokens=T,
                block_table=torch.zeros((1, self.buckets.max_blocks(context)), dtype=torch.int32),
                slot_mapping=torch.zeros(T, dtype=torch.long),
                seq_lens=torch.tensor([T], dtype=torch.long),
                positions=torch.arange(T, dtype=torch.long),
                cached_seq_len=0,
            )
            # a runtime of prefill context buckets (_warm_shapes' third field) warms each; None is the server's own
            out = (self.build_model_inputs(md, token_ids=torch.ones(T, dtype=torch.int32), bucket=T) if context is None
                   else self.build_model_inputs(md, token_ids=torch.ones(T, dtype=torch.int32), bucket=T,
                                                context=context))
            if getattr(self, "pp_active", False):
                # a stage's boundary stand-ins: zeros in, and slots of the seam's shape out, so the warm-up compiles
                # the served graph and never writes a slot a send may be reading
                from sglang_neuron.distributed.pp_seam import prefill_boundary

                kinds, dev, tp = prefill_boundary(self), neuron_device(), self.model_tp_size
                if not self.pp_is_first_stage:
                    out.update({name: torch.zeros((spec.received_rows(T, tp), spec.hidden), dtype=spec.dtype,
                                                  device=dev) for name, spec in kinds.items()})
                if not self.pp_is_last_stage:
                    out.update({f"{name}_out": torch.zeros((spec.sent_rows(T, tp), spec.hidden), dtype=spec.dtype,
                                                           device=dev) for name, spec in kinds.items()})
            return out
        bs = bucket
        seq_len = min(DUMMY_LIVE_CTX, context or self.buckets.max_model_len)
        # laid out like a real row's table: the reserved block for the live pages, -1 after them (a table of block 0 in
        # every column makes the attention read all of it, and an idle rank then holds up every rank's step)
        live_pages = -(-seq_len // self.block_size)
        block_table = torch.full((bs, self.buckets.max_blocks(context)), -1, dtype=torch.int32)
        block_table[:, :live_pages] = 0
        md = NeuronForwardMetadata(
            is_prefill=False, batch_size=bs, num_tokens=bs,
            block_table=block_table,
            slot_mapping=torch.zeros(bs, dtype=torch.long),
            seq_lens=torch.full((bs,), seq_len, dtype=torch.long),
            positions=torch.full((bs,), seq_len - 1, dtype=torch.long),
            cached_seq_len=0,
        )
        out = self.build_model_inputs(md, token_ids=torch.ones(bs, dtype=torch.int32), bucket=bs, context=context)
        if getattr(self, "pp_active", False):
            # a decode stage's boundary stand-ins, in the served order (after the step's own inputs): zeros in, and slots
            # of the seam's shape out
            shape, dev = (bs, self.pp_hidden_size), neuron_device()
            if not self.pp_is_first_stage:
                out.update({name: torch.zeros(shape, dtype=self.pp_boundary_dtype, device=dev)
                            for name in self.pp_decode_boundary})
            if not self.pp_is_last_stage:
                out.update({f"{name}_out": torch.zeros(shape, dtype=self.pp_boundary_dtype, device=dev)
                            for name in self.pp_decode_boundary})
            elif self.pp_token_relay:
                # the token send slot the last stage samples into
                out["token_ids_out"] = torch.zeros(int(self.decode_token_width), dtype=torch.int32, device=dev)
        return out

    def plan_step(self, fb: ForwardBatch, md: Optional[NeuronForwardMetadata] = None) -> "StepPlan":
        """Pick (phase, bucket) for this step. Under DP attention the phase and
        bucket must agree across DP ranks: the phase is SGLang's globally
        synced ``is_extend_in_batch`` and the bucket is derived from the max
        token count in the group (idle ranks dispatch a dummy of that shape).
        A step that mixes prefill and decode across the EP-spanning group cannot
        be reconciled with static-shape NEFFs, so it raises."""
        mode = fb.forward_mode
        if self.dp_identity_bucket is not None:
            # No exchanged counts exist: the shape is a static constant, so this rank's
            # choice is the group's choice. A second bucket must never desync silently.
            if not (mode.is_decode() or mode.is_idle()):
                raise RuntimeError(f"DP identity: {mode} step on a pure decode server")
            n = 0 if mode.is_idle() else (md.batch_size if md is not None else int(fb.batch_size))
            if n > self.dp_identity_bucket:
                raise RuntimeError(f"DP identity: batch {n} exceeds bucket {self.dp_identity_bucket}")
            return StepPlan("idle" if n == 0 else "decode", "decode", self.dp_identity_bucket)
        g = getattr(fb, "original_global_num_tokens_cpu", None) or getattr(fb, "global_num_tokens_cpu", None)
        synced = bool(g) and len(g) > 1
        group_phase = None
        group_max = 0
        if synced:
            group_phase = "prefill" if fb.is_extend_in_batch else "decode"
            group_max = max(int(x) for x in g)
        if mode.is_idle():
            if not synced or group_phase == "decode":
                return StepPlan("idle", "decode", self.buckets.seqs_bucket(max(group_max, 1)))
            return StepPlan("idle", "prefill", self.buckets.prefill_bucket(max(group_max, 1)))
        if mode.is_decode():
            if synced and group_phase == "prefill":
                raise RuntimeError(
                    "DP ranks disagree on phase this step (this rank: decode, group: prefill); "
                    "mixed phases across an EP-spanning DP group need disaggregated prefill/decode fleets"
                )
            n = md.batch_size if md is not None else int(fb.batch_size)
            target = max(n, group_max) if synced else n
            context = None
            if len(self.buckets.decode_contexts) > 1 and md is not None and n:
                context = self.buckets.decode_context(int(md.seq_lens[:n].max()))
            return StepPlan("decode", "decode", self.buckets.seqs_bucket(max(target, 1)), context)
        # extend
        if synced and group_phase != "prefill":
            raise RuntimeError("DP ranks disagree on phase this step (this rank: prefill, group: decode)")
        n = md.num_tokens if md is not None else int(fb.input_ids.shape[0])
        if self.prefill_holds_prior and md is not None:
            n += int(md.cached_seq_len)
        target = max(n, group_max) if synced else n
        return StepPlan("prefill", "prefill", self.buckets.prefill_bucket(max(target, 1)))

    def _swa_num_blocks(self, sliding_window: int) -> int:
        """Table columns a decode row's sliding window spans: its W - 1 prior keys and the active token's page (under
        DCP virtual pages, plus one after the active token's, as in BucketConfig.decode_context)."""
        layout = getattr(self, "dcp", None)
        page = self.block_size if layout is None else layout.page
        min_blocks = -(-(sliding_window - 1) // page) + 1 + (0 if layout is None else 1)
        # the decode attention kernel shards the active KV across the LNC cores, each shard a multiple of P_MAX, so the
        # trimmed table aligns to LNC x P_MAX tokens; with the packed fp8 K cache the kernel also needs its resized
        # block length >= 2 (a multiple of 2 x LNC x P_MAX tokens), else it unpacks the cache on every step
        align_tokens = _lnc_degree() * P_MAX * (2 if self._kv_fp8_packed() else 1)
        n = _round_up(min_blocks, align_tokens // self.block_size)
        return min(n, self.buckets.max_blocks())

    def _dcp_bounds(self, md, n: int, rows: int, table_blocks: int) -> dict:
        """A DCP decode step's per-row inputs, as host tensors of the graph's ``rows``.

        Each row decodes position L (its prior's length; 0 on a padding row). ``("full", "end")``: how many prior tokens
        this rank keeps, where its prior ends in its own table. ``("full", "owner")``: 1 where this rank keeps the
        decoded token. A sliding layer of window W reads the table trimmed to virtual pages from ``("start_blocks", W)``
        (ending one past the decoded token's), and in that frame its window spans ``(W, "start")`` to ``(W, "end")``."""
        layout = self.dcp
        lengths = torch.zeros(rows, dtype=torch.long)
        lengths[:n] = md.seq_lens[:n] - 1
        end = layout.local_count(lengths)
        out = {("full", "end"): end.to(torch.int32),
               ("full", "owner"): (layout.owner(lengths) == layout.rank).to(torch.int32)}
        for sw in sorted({w for w in self.sliding if w is not None}):
            nb = min(self._swa_num_blocks(sw), table_blocks)
            start_blocks = torch.clamp(lengths // layout.page - nb + 2, min=0)
            frame = start_blocks * layout.block
            first = layout.local_count(torch.clamp(lengths - sw + 1, min=0))
            out[("start_blocks", sw)] = start_blocks
            out[(sw, "end")] = (end - frame).to(torch.int32)
            out[(sw, "start")] = torch.clamp(first - frame, min=0).to(torch.int32)
        return out

    def _cp_rank_slots(self, slot_mapping: torch.Tensor) -> torch.Tensor:
        """A page-aligned prefill chunk's global SGLang slots as this rank's compacted kernel slots.

        Rank r owns the chunk's tokens at in-page offsets [r * I, (r + 1) * I) (I the interleave), the static slice
        view(T // page, cp, I)[:, r]. A global slot s on page s // page lands in this rank's block s // page at row
        s % I. Padding slot 0 stays 0: page 0 is SGLang's reserved page, and block 0 every rank's null block."""
        page, cp, interleave = self.block_size, self.kv_cp_size, self.kv_cp_interleave
        if slot_mapping.shape[0] % page:
            raise ValueError(f"CP prefill chunk of {slot_mapping.shape[0]} tokens is not page ({page}) aligned")
        local = slot_mapping.view(-1, cp, interleave)[:, self.kv_cp_rank, :].reshape(-1)
        return (local // page) * interleave + local % interleave

    def _cp_prefill_swa_window(self, block_table: torch.Tensor, cached: int, sliding_window: int):
        """A CP prefill's static sliding-window table for window ``sliding_window`` and its frame offset (tokens).

        Columns are whole pages (cp x interleave tokens). Pages [start, cached // page] hold every key a window query
        of the active chunk can read: the W - 1 keys before the page-aligned ``cached`` span ceil((W - 1) / page) prior
        pages, then the first active page; the width rounds up to whole kernel row tiles (P_MAX rows of the
        interleave), and the last column is the first active page. A window past the table is refused."""
        page, interleave = self.block_size, self.kv_cp_interleave
        rows_per_tile = max(P_MAX // interleave, 1)
        prior_pages = (sliding_window - 1 + page - 1) // page
        width = _round_up(prior_pages + 1, rows_per_tile)
        start = max(cached // page - width + 1, 0)
        index = torch.arange(width, dtype=torch.long) + start
        if int(index[-1]) >= block_table.shape[1]:
            raise ValueError(f"SWA prefill window [{start}, {start + width}) exceeds the {block_table.shape[1]}-page "
                             "table")
        trimmed = block_table.index_select(1, index)
        offset = torch.full((block_table.shape[0],), start * page, dtype=torch.int32)
        return trimmed, offset

    def _kv_fp8_packed(self) -> bool:
        pool = self.model_runner.token_to_kv_pool
        # on the hybrid layout both sub-pools are Neuron pools of the launch's KV dtype
        return bool(getattr(pool, "full_kv_pool", pool).fp8_packed_k)

    def _decode_token_inputs(self, token_plan, token_ids, n, width):
        """(input_ids device vector, token_control) for the reordered decode graph (_decode_token_sources).

        Every decode dispatch (real, idle, warmup) passes the same two inputs with the same shapes,
        so all DP ranks keep launching the identical NEFF."""
        from sglang_neuron.backend.device_io import upload

        prev, control, host = self._decode_token_sources(token_plan, token_ids, n, width)
        return prev, control if host is None else upload(host, neuron_device())

    def _decode_token_sources(self, token_plan, token_ids, n, width):
        """The reordered decode graph's token inputs before their upload: the input_ids device vector, and the
        token_control as either its cached device identity or a host tensor to upload, the other None. Without PP it
        reads no pending device tokens, and resolves the device only for the two inputs uploaded once (the identity
        control and the placeholder future). On a pipelined first stage a relayed future resolves to its seam landing
        slot (async_execution.SeamTokenInbox.device), which may first drain earlier relayed steps to the host."""
        from sglang_neuron.backend.device_io import upload

        prev = token_plan.prev if token_plan is not None else None
        if prev is not None and hasattr(prev, "resolve"):
            # PP first stage: the last stage's relayed tokens, a seam landing slot whose WRITE the graph's submit waits
            # for (async_execution._SeamTokens)
            prev = prev.resolve()
        if token_plan is not None and token_plan.control is None:
            if tuple(prev.shape) != (width,) or prev.dtype != torch.int32:
                raise RuntimeError("Device token future does not match the decode token width")
            if self._token_identity_control is None:
                self._token_identity_control = upload(torch.stack(
                    [torch.zeros(width, dtype=torch.int32), torch.arange(width, dtype=torch.int32)]), neuron_device())
            return prev, self._token_identity_control, None
        if token_plan is not None:
            control = token_plan.control
        else:  # warm-up, idle dummy, host ids: every row from token_control[0]
            control = torch.zeros((2, width), dtype=torch.int32)
            control[1] = -1
            if n:
                control[0, :n] = token_ids[:n].to(torch.int32)
        if prev is None:
            if self._token_placeholder is None:
                # a distinct tensor: never alias another graph input (dynamo would dedup it)
                self._token_placeholder = upload(torch.zeros(width, dtype=torch.int32), neuron_device())
            prev = self._token_placeholder
        if tuple(prev.shape) != (width,) or prev.dtype != torch.int32:
            raise RuntimeError("Device token future does not match the decode token width")
        return prev, None, control

    def build_model_inputs(
        self, md: NeuronForwardMetadata, token_ids: torch.Tensor, bucket: int,
        temperatures: Optional[torch.Tensor] = None, token_plan=None,
        seeds: Optional[torch.Tensor] = None, filters: Optional[torch.Tensor] = None, *,
        context: Optional[int] = None,
    ) -> dict:
        """Pad exact SGLang metadata to a compiled bucket, a decode's table to its ``context`` (by default the whole
        context), and build the graph's device kwargs."""
        from sglang_neuron.backend.device_io import upload, upload_many
        from sglang_neuron.mem_cache import hybrid_swa

        bs_real = md.batch_size
        token_width = (self.decode_token_width or bucket) if not md.is_prefill else bucket
        device_token_ids = token_control = None
        if md.is_prefill:
            assert bs_real == 1, "Neuron prefill runs one request per forward"
            T = bucket
            n = md.num_tokens
            input_ids = torch.zeros(T, dtype=torch.int32)
            input_ids[:n] = token_ids.to(torch.int32)
            positions = torch.full((T,), int(md.positions[n - 1]) if n else 0, dtype=torch.long)
            positions[:n] = md.positions
            slot_mapping = torch.zeros(T, dtype=torch.long)
            slot_mapping[:n] = md.slot_mapping
            max_blocks = self.buckets.max_blocks()
            if self.prefill_table_pad:
                max_blocks += self.buckets.prefill_token_buckets[-1] // self.block_size
            block_table = torch.full((1, max_blocks), -1, dtype=torch.int32)
            w = min(md.block_table.shape[1], max_blocks)
            block_table[0, :w] = md.block_table[0, :w]
            cached = md.cached_seq_len
            assert cached % self.block_size == 0, (cached, self.block_size)
            if self.window_pool is not None and cached:
                hybrid_swa.check_window_resident(self.window_pool, block_table, cached,
                                                 max(w for w in self.sliding if w is not None), self.block_size)
            if self.kv_cp_size > 1:
                # this rank's own tokens of every page, as its compacted kernel slots
                slot_mapping = self._cp_rank_slots(slot_mapping)
            kv_segment = T  # the fixed segment tile, equal to the bucket
            sampling_positions = torch.tensor([n - 1], dtype=torch.long)
            padded_reqs = 1
            max_query_len = T
        else:
            B = bucket
            n = bs_real
            if self.decode_token_width and getattr(self, "decode_token_assemble", True):
                device_token_ids, token_control = self._decode_token_inputs(token_plan, token_ids, n, token_width)
            else:
                # the batch's host ids padded to the bucket: a decode without fixed-width token I/O, or a PP stage after
                # the first (zero placeholder ids it does not embed)
                input_ids = torch.zeros(B, dtype=torch.int32)
                input_ids[:n] = token_ids.to(torch.int32)
            positions = torch.zeros(B, dtype=torch.long)
            positions[:n] = md.positions[:n]
            slot_mapping = torch.zeros(B, dtype=torch.long)
            slot_mapping[:n] = md.slot_mapping[:n]
            bucket_blocks = self.buckets.max_blocks(context)
            block_table = torch.full((B, bucket_blocks), -1, dtype=torch.int32)
            w = min(md.block_table.shape[1], bucket_blocks)
            block_table[:n, :w] = md.block_table[:, :w]
            # padded rows: position 0 over SGLang's reserved page 0, so every row has exactly one valid key; a fully
            # masked row would fault the decode kernel. Under DCP only rank 0 holds that key, and the other ranks'
            # IDLE_SINK keeps their empty row finite and out of the merge (decode_context_parallel.group_sinks)
            block_table[n:, 0] = 0
            cached, kv_segment = 0, 0
            padded_reqs = B
            max_query_len = 1

        dev = neuron_device()
        # DCP: each row's bounds over this rank's share and the owner of its decoded token (_dcp_bounds)
        layout = getattr(self, "dcp", None)
        bounds = None if md.is_prefill or layout is None else self._dcp_bounds(md, n, padded_reqs,
                                                                                block_table.shape[1])
        # the step's host inputs in one upload (one switch to the side stream); a decode's block table is staged instead
        # (uploaded only when it changes)
        hosts = [slot_mapping, positions] + ([input_ids] if device_token_ids is None else [])
        if md.is_prefill:
            hosts += [block_table, torch.tensor([[cached]], dtype=torch.int32), sampling_positions]
        # hybrid layout: the sliding layers write the window pool's slots, and a prefill's read its pages through the
        # whole table (a decode's trimmed window table is staged below)
        window_hosts = []
        if self.window_pool is not None:
            window_hosts.append(hybrid_swa.window_slots(self.window_pool, slot_mapping))
            if md.is_prefill:
                window_hosts.append(hybrid_swa.window_blocks(self.window_pool, block_table, self.block_size))
        dcp_keys = [] if bounds is None else [key for key in bounds if key[0] != "start_blocks"]
        uploaded = upload_many(hosts + window_hosts + [bounds[key] for key in dcp_keys], dev)
        dcp_d = dict(zip(dcp_keys, uploaded[len(hosts) + len(window_hosts):]))
        window_d = uploaded[len(hosts):len(hosts) + len(window_hosts)]
        uploaded = uploaded[:len(hosts)]
        slot_mapping_d, positions_d = uploaded[0], uploaded[1]
        input_ids_d = uploaded[2] if device_token_ids is None else device_token_ids
        if md.is_prefill:
            block_table_d, cached_t, sampling_positions_d = uploaded[-3:]
        else:
            block_table_d = self._staged("bt", block_table, dev)
            if self._decode_cached_zero is None:
                self._decode_cached_zero = upload(torch.tensor([[0]], dtype=torch.int32), dev)
            cached_t = self._decode_cached_zero
            if bucket not in self._decode_sampling_positions:
                self._decode_sampling_positions[bucket] = upload(torch.arange(bucket, dtype=torch.long), dev)
            sampling_positions_d = self._decode_sampling_positions[bucket]
        attn_metadata: Dict[str, dict] = {}
        full_md = {
            "block_table_tensor": block_table_d, "slot_mapping": slot_mapping_d, "max_query_len": max_query_len,
            "block_size": self.kv_cp_interleave if self.kv_cp_size > 1 else self.block_size,
            "max_blocks_per_seq": block_table.shape[1],
            "decode_token_threshold": 1, "cached_seq_len": cached_t, "kv_segment_size": kv_segment,
        }
        if bounds is not None:
            full_md.update(dcp_kv_end=dcp_d[("full", "end")], dcp_active_owner=dcp_d[("full", "owner")])
        window_md: Dict[object, dict] = {}
        # the segmented prefill kernel masks a window over the untrimmed table, so a prefill's sliding layers read the
        # full layers' table, or on the hybrid layout the window pool's; a decode's read a table trimmed to the window
        prefill_window_md = full_md
        if md.is_prefill and window_d:
            prefill_window_md = {**full_md, "slot_mapping": window_d[0], "block_table_tensor": window_d[1]}
        for name, sw in zip(self.layer_names, self.sliding):
            if sw is not None and md.is_prefill and self.kv_cp_size > 1:
                # a CP prefill's sliding layers read their prior through a static table of whole pages
                # (_cp_prefill_swa_window); the active writes keep the chunk's slots
                if ("cp_prefill", sw) not in window_md:
                    trimmed, offset = self._cp_prefill_swa_window(block_table, cached, sw)
                    window_md[("cp_prefill", sw)] = {**full_md, "block_table_tensor": upload(trimmed, dev),
                                                     "max_blocks_per_seq": trimmed.shape[1],
                                                     "swa_kv_pos_offset": upload(offset, dev)}
                attn_metadata[name] = window_md[("cp_prefill", sw)]
                continue
            if sw is None or md.is_prefill:
                attn_metadata[name] = full_md if sw is None else prefill_window_md
                continue
            if sw not in window_md:
                ctx_lens = torch.zeros(padded_reqs, dtype=torch.long)
                ctx_lens[:n] = md.seq_lens[:n] - 1
                nb = min(self._swa_num_blocks(sw), block_table.shape[1])
                start_blocks = (torch.clamp(ctx_lens // self.block_size - nb + 1, min=0) if bounds is None
                                else bounds[("start_blocks", sw)])
                gather_idx = (start_blocks.unsqueeze(1) + torch.arange(nb, dtype=torch.long).unsqueeze(0)).clamp(
                    max=block_table.shape[1] - 1)
                trimmed = block_table.gather(1, gather_idx)
                # columns past the table's width carry the -1 sentinel, not a clamped duplicate
                valid = (torch.arange(nb).unsqueeze(0) + start_blocks.unsqueeze(1)) < block_table.shape[1]
                trimmed = torch.where(valid, trimmed, torch.full_like(trimmed, -1))
                offset = (start_blocks * (self.block_size if layout is None else layout.page)).to(torch.int32)
                if window_d:
                    trimmed = hybrid_swa.window_blocks(self.window_pool, trimmed, self.block_size)
                window_md[sw] = {**full_md,
                                 "block_table_tensor": self._staged(("swa_bt", sw), trimmed, dev),
                                 "max_blocks_per_seq": trimmed.shape[1],
                                 "swa_kv_pos_offset": self._staged(("swa_off", sw), offset, dev)}
                if window_d:
                    window_md[sw]["slot_mapping"] = window_d[0]
                if bounds is not None:
                    window_md[sw].update(dcp_kv_end=dcp_d[(sw, "end")], dcp_kv_start=dcp_d[(sw, "start")])
            attn_metadata[name] = window_md[sw]

        out = dict(
            input_ids=input_ids_d, positions=positions_d, attn_metadata=attn_metadata,
            sampling_positions=sampling_positions_d, rank=self.rank_tensor,
        )
        if token_control is not None:
            out["token_control"] = token_control
        from sglang_neuron.sampling import DRAWING_MODES

        if self.sampler_mode in DRAWING_MODES:
            out.update(self._sampling_inputs(md, n, padded_reqs, temperatures, seeds, dev, filters))
        return out

    def _sampling_inputs(self, md, n: int, rows: int, temperatures, seeds, dev, filters=None) -> dict:
        """Per-row inputs of the sampled device modes, padded to the graph's rows.

        Temperature 0 (greedy requests, padding rows and synthetic dispatches, which pass no
        temperatures) selects exactly; seeds are the requests' int64 seeds folded to int32;
        topk_topp adds the normalized (top_k, top_p, min_p) per row, unrestricted for padding.
        The draw's position key is the row's input position, which the graph reads from
        ``positions``, so these tensors change only with the batch: decode stages them."""
        from sglang_neuron.backend.device_io import upload

        from sglang_neuron.sampling import TOP_K_ALL, fold_seeds

        temps = torch.zeros(rows, dtype=torch.float32)
        keys = torch.zeros(rows, dtype=torch.int32)
        bounds = None
        if self.sampler_mode == "topk_topp":
            bounds = torch.tensor([[float(TOP_K_ALL), 1.0, 0.0]], dtype=torch.float32).repeat(rows, 1)
        if n > 0 and temperatures is not None:
            real = 1 if md.is_prefill else n
            tt = temperatures.to("cpu", torch.float32).reshape(-1)
            if tt.numel() < real:
                raise RuntimeError(f"sampling temperatures cover {tt.numel()} of {real} rows")
            temps[:real] = tt[:real]
            if seeds is not None:
                ss = seeds.to("cpu").reshape(-1)
                if ss.numel() < real:
                    raise RuntimeError(f"sampling seeds cover {ss.numel()} of {real} rows")
                keys[:real] = fold_seeds(ss[:real])
            if bounds is not None:
                if filters is None or filters.shape[0] < real:
                    raise RuntimeError(f"topk_topp sampling needs (top_k, top_p, min_p) for {real} rows")
                bounds[:real] = filters[:real].to("cpu", torch.float32)
        host = {"sampling_temperatures": temps, "sampling_seeds": keys}
        if bounds is not None:
            host["sampling_filters"] = bounds
        if md.is_prefill:
            return {name: upload(value, dev) for name, value in host.items()}
        return {name: self._staged(name, value, dev) for name, value in host.items()}

    def _staged(self, key, host: torch.Tensor, dev):
        """Reuse the device copy of a decode input whose host value is unchanged.

        A decode block table (and its sliding-window trim and offset) changes only when a row crosses a page
        or the batch changes, so most steps would re-upload identical bytes. A fresh device tensor is made for
        every change and an uploaded tensor is never written in place: a queued graph may still read the old copy
        (device_io.upload records it on the default stream).
        """
        from sglang_neuron.backend.device_io import upload

        cache = self.__dict__.setdefault("_stage_cache", {})
        hit = cache.get(key)
        if hit is not None and hit[0].shape == host.shape and hit[0].dtype == host.dtype and torch.equal(hit[0], host):
            self._stage_hits = getattr(self, "_stage_hits", 0) + 1
            return hit[1]
        d = upload(host, dev)
        cache[key] = (host.clone(), d)
        self._stage_misses = getattr(self, "_stage_misses", 0) + 1
        if self._stage_misses % 1000 == 0:
            logger.info("[neuron] staged decode inputs: %d reused, %d uploaded",
                        getattr(self, "_stage_hits", 0), self._stage_misses)
        return d

    def _decode_serving_capture(self, phase):
        """The serving device-profile window (SGLANG_NEURON_PROFILE_SERVE_*) a decode dispatch counts against, or None.
        Every decode dispatch counts, an idle DP rank's dummy and a non-last PP stage's too: the window's entry and exit
        are barriers every rank of the (stage's) TP group must reach on the same step."""
        if phase != "decode" or not self._profile_serve:
            return None
        if not hasattr(self, "_serving_capture"):
            from sglang_neuron.model_executor.profiling import ServingCapture, serve_label
            from sglang_neuron.parallel.state import tp_barrier

            self._serving_capture = ServingCapture(self.tp_rank, tp_barrier, serve_label(self.model_runner))
        return self._serving_capture

    def _run(self, kwargs: dict, n_real: int, phase=None, launch_only: bool = False) -> LogitsProcessorOutput:
        """Launch one call of the family's graph (or its bound executor) with ``kwargs`` and return its result: a pending
        result under async execution, else the outputs read back. ``launch_only`` (``_submit``): return its device
        outputs and launch event (backend.device_io.after_launch) instead, nothing read."""
        from sglang.srt.utils.common import DynamicGradMode

        from sglang_neuron.backend import device_io

        capture = self._decode_serving_capture(phase)
        if capture is not None:
            capture.before()
        tr = None
        if self._host_trace:
            from sglang_neuron.model_executor import host_trace

            tr = host_trace.trace()
        s0 = time.time_ns() if tr is not None else 0
        # a bound direct step: the executor's inputs built from the step's kwargs (None: the full call path, which the
        # direct submit observes while it binds and in its shadow steps)
        direct = getattr(self, "direct_submit", None) if phase == "decode" else None
        flat = direct.plan(kwargs) if direct is not None else None
        # a direct step calls no compiled graph, so it has nothing to recompile
        with DynamicGradMode(), (
            torch.compiler.set_stance("fail_on_recompile") if flat is None else contextlib.nullcontext()
        ):
            device_io.before_launch()
            with device_io.pre_execute_scope():
                if flat is not None:
                    out = (getattr(self, "direct_launch", None) or direct.submit)(flat)
                elif direct is not None:
                    out = direct.full(self.compiled, kwargs)
                else:
                    out = self.compiled(**kwargs)
            launch_event = device_io.after_launch()
        if tr is not None:
            tr.add("submit", s0, time.time_ns(), phase or "-")
        if capture is not None:
            capture.after(_output_tensor(out))
        if launch_only:
            return out, launch_event
        if isinstance(out, torch.Tensor):
            # the host sampler's logits (SGLANG_NEURON_SAMPLER=host), for SGLang's sampler in ModelRunner.sample
            logits = device_io.readback(out, launch_event)[:n_real].float()
            return LogitsProcessorOutput(next_token_logits=logits, hidden_states=None)
        fixed_tokens = bool(self.decode_token_width)
        # the host-ids mode (a megakernel runtime that takes its ids on the host): a step of any row count within its
        # outputs is pending (the pending sample reads its real rows), and an idle call is never read
        host_ids = self.async_execution and self.host_input_ids
        if self.async_execution and (
            n_real == out["token_ids"].shape[0]
            or (fixed_tokens or host_ids) and 0 < n_real <= out["token_ids"].shape[0]
            or self.async_idle_completion and phase == "decode" and n_real == 0
        ):
            from sglang_neuron.model_executor.async_execution import PendingSample, mark_pending

            res = LogitsProcessorOutput(next_token_logits=torch.zeros((n_real, 0), dtype=torch.float32),
                                        hidden_states=None)
            return mark_pending(res, PendingSample(out, n_real, keep_alive=kwargs, launch_event=launch_event,
                                                   phase=phase, vocab_size=self.vocab_size))
        if self.async_execution and (self.host_input_ids or self.idle_unread) and n_real == 0 \
                and not (self.async_idle_completion and phase == "decode"):
            self.retain_unread(out, keep_alive=kwargs)
            return LogitsProcessorOutput(next_token_logits=torch.zeros((0, 0), dtype=torch.float32),
                                         hidden_states=None)
        host = device_io.readback_many(list(out.values()), launch_event)  # one wait for every output
        sample = {key: value[:n_real] for key, value in zip(out, host)}
        res = LogitsProcessorOutput(next_token_logits=torch.zeros((n_real, 0), dtype=torch.float32),
                                    hidden_states=None)
        sample["tokens"] = sample.pop("token_ids").to(torch.int64)
        # an id no token has (negative, or at or above the vocabulary size) refuses the step; an asynchronous step's
        # rows are checked where the copy thread materializes them (PendingSample.materialize)
        from sglang_neuron.model_executor.async_execution import refuse_invalid_token_ids

        refuse_invalid_token_ids(sample["tokens"], self.vocab_size)
        res.neuron_sample = sample
        return res

    def _submit(self, kwargs: dict, phase=None):
        """Launch one call of the family's graph with ``kwargs``; its device outputs and launch event, nothing read (a
        megakernel step's calls, read together through step_output)."""
        return self._run(kwargs, 0, phase, launch_only=True)

    def step_output(self, parts, index, launch_event, phase=None, keep_alive=None, checks=(),
                    check=None) -> LogitsProcessorOutput:
        """The result of a step that ran as several calls (a megakernel step: prefill chunks, then lane calls), each
        launched with ``_submit``: ``parts`` are the calls' device token vectors in launch order, ``launch_event`` the
        last call's, and ``index`` (int64) picks each of the step's rows from the parts concatenated.

        Under async execution nothing is read here: the step's one pending result (async_execution.StepAssembly) reads
        every part in one readback when the step's result is processed, or when the next decode takes its ids (the
        host-ids mode), whichever comes first. Otherwise the parts are read here, in one readback. ``keep_alive`` holds
        the calls' inputs until the read. ``checks`` and ``check``: the calls' other outputs a check reads with the
        tokens, whose RuntimeError refuses the step (StepAssembly)."""
        from sglang_neuron.model_executor import async_execution

        index = torch.as_tensor(index, dtype=torch.int64).reshape(-1)
        n = int(index.shape[0])
        res = LogitsProcessorOutput(next_token_logits=torch.zeros((n, 0), dtype=torch.float32), hidden_states=None)
        assembly = async_execution.StepAssembly(parts, index, checks=checks, check=check)
        if self.async_execution:
            pending = async_execution.PendingSample({"token_ids": parts[-1]}, n, keep_alive=keep_alive,
                                                    launch_event=launch_event, phase=phase, assemble=assembly)
            return async_execution.mark_pending(res, pending)
        from sglang_neuron.backend import device_io

        tokens, refusal = assembly.read(device_io.readback_many(assembly.tensors(), launch_event))
        if refusal is not None:
            raise refusal
        async_execution.refuse_invalid_token_ids(tokens)
        res.neuron_sample = {"tokens": tokens}
        return res

    def retain_unread(self, outputs, keep_alive=None) -> None:
        """Hold a launch nothing reads (an idle step's padding calls under async execution) until later launches have
        retired it: its outputs and inputs stay referenced for the in-flight bound's worth of later steps
        (backend.device_io.before_launch waits for the oldest launch past the bound), so no buffer it writes or reads
        is released while it may still run."""
        held = getattr(self, "_unread", None)
        if held is None:
            import collections

            from sglang_neuron.backend import device_io

            held = self._unread = collections.deque(maxlen=device_io._max_inflight() + 2)
        held.append((outputs, keep_alive))

    def run_prefill(self, fb: ForwardBatch, md: NeuronForwardMetadata,
                    bucket: Optional[int] = None, pp_proxy=None) -> LogitsProcessorOutput:
        """One request's prefill forward. On a PP stage ``pp_proxy`` is the previous stage's step descriptor and batch
        fingerprint (None on the first stage), and a non-last stage returns its boundary's descriptor."""
        bucket = bucket or self.buckets.prefill_bucket(md.num_tokens)
        token_ids = fb.input_ids.to("cpu")[: md.num_tokens]
        if not getattr(self, "pp_active", False):
            if pp_proxy is not None:
                raise RuntimeError("a pipeline boundary reached a runtime that is not a pipeline stage")
            kwargs = self.build_model_inputs(md, token_ids=token_ids, bucket=bucket, **self._sampling_kwargs(fb))
            return self._run(kwargs, n_real=1, phase="prefill")
        fingerprint = pp_batch_fingerprint(md, token_ids, getattr(fb, "rids", None))
        try:
            kwargs = self.build_model_inputs(md, token_ids=token_ids, bucket=bucket, **self._sampling_kwargs(fb))
            if not self.pp_is_first_stage:
                kwargs.update(self._pp_boundary_inputs(fingerprint, bucket, pp_proxy))
        except BaseException as error:
            # a slot received and never consumed leaves the seam out of step with the previous stage
            self._poison_seam_after_receive(f"input build failed after a seam receive: {error!r}")
            raise
        return self._run_stage(kwargs, 1, "prefill", fingerprint)

    # pipeline stage
    def _poison_seam_after_receive(self, cause: str) -> None:
        """A step that received a boundary and failed before its forward never releases the slots: the seam is out of
        step with the previous stage for good."""
        if self._seam_recv_live:
            from sglang_neuron.distributed.pp_seam import get_seam

            self._seam_recv_live = False
            get_seam(self).poison(cause)

    def _pp_boundary_inputs(self, fingerprint: int, tokens: int, pp_proxy, phase: str = "prefill") -> dict:
        """A stage after the first: the landing slots of this step's boundary. A proxy without the seam's step
        descriptor, or for another batch, is refused and poisons the seam. On a device the WRITE is awaited right before
        the graph's submit (the seam's pre-execute hook), so dynamo's guards, the AOT wrapper and the executor's input
        preparation overlap the previous stage's WRITE."""
        from sglang_neuron.distributed.pp_seam import boundary_kinds, get_seam, prefill_boundary

        tensors = getattr(pp_proxy, "tensors", None)
        if not isinstance(tensors, dict) or "neuron_pp_step" not in tensors:
            raise RuntimeError(f"a pipeline stage after the first received no seam step descriptor for its {phase} "
                               f"(the proxy holds {sorted(tensors) if isinstance(tensors, dict) else pp_proxy!r})")
        seam = get_seam(self)
        sent = tensors.get("neuron_pp_batch")
        if sent is None or int(sent.reshape(-1)[0]) != fingerprint:
            reason = ("the pipeline stages scheduled different batches for one boundary (batch fingerprint "
                      f"{'missing' if sent is None else int(sent.reshape(-1)[0])} from the previous stage, "
                      f"{fingerprint} here)")
            seam.poison(reason)
            raise RuntimeError(reason)
        step = int(tensors["neuron_pp_step"].reshape(-1)[0])
        wait = not seam.landing_hook
        if phase == "prefill":
            # each kind's received row blocks of the step's bucket (SP-sharded over the attention TP ranks, or whole)
            out = {name: seam.recv(spec.kind, step, spec.received_rows(tokens, self.model_tp_size), wait=wait)
                   for name, spec in prefill_boundary(self).items()}
        else:
            # a decode boundary is TP-replicated: the batch bucket's rows
            names = self.pp_decode_boundary
            out = {name: seam.recv(kind, step, tokens, wait=wait)
                   for name, kind in zip(names, boundary_kinds(names, phase))}
        self._seam_recv_live = True
        return out

    def _check_boundary_in_slots(self, out, slots: dict, phase: str = "prefill", written=_TAKE) -> None:
        """Refuse a stage graph that did not write its boundary into its slots in place. The returned tensor can't tell,
        since AOTAutograd and torch-neuronx's writeback return the caller's tensor either way, so on a device each slot
        must be a copy-back target the NEFF wrote in place (executor_cache.take_writeback, or ``written`` when the
        caller took it). A non-target was never written, and a copy back is an extra device operation per forward. CPU
        mode checks the returned storage."""
        from sglang_neuron.backend import device_io, executor_cache

        if written is _TAKE:
            written = executor_cache.take_writeback() if device_io.on_device() else None
        for key, slot in slots.items():
            name = key[: -len("_out")]
            got = out.get(name) if isinstance(out, dict) else None
            if got is None or got.data_ptr() != slot.data_ptr() or tuple(got.shape) != tuple(slot.shape):
                raise RuntimeError(f"the {phase} stage graph returned its {name} outside the seam slot it was handed")
            if not device_io.on_device():
                continue
            hits = [in_place for target, in_place in (written or ())
                    if target.data_ptr() == slot.data_ptr() and tuple(target.shape) == tuple(slot.shape)]
            if hits != [True]:
                raise RuntimeError(
                    f"the {phase} stage graph's {name} reached its seam slot "
                    f"{'by no write of the NEFF' if not hits else 'by a copy back, not in place'}: the boundary "
                    "would be stale or need a device copy after every forward")

    def _run_stage(self, kwargs: dict, n_real: int, phase, fingerprint: int):
        """A pipeline stage's step. A stage before the last writes its boundary into claimed send slots and returns the
        step's descriptor for SGLang's proxy while the forward runs (``_run_boundary``). The last stage runs like any
        runtime (``_run``), a pipelined PD decode's also relaying its tokens to the first stage (``_run_relaying``), and
        hands its landing slots back once its forward has run. A failure poisons the seam, since the stages are then out
        of step for good."""
        from sglang_neuron.distributed.pp_seam import get_seam

        seam = get_seam(self)
        try:
            if self.pp_is_last_stage:
                if phase == "decode" and self.pp_token_relay:
                    return self._run_relaying(seam, kwargs, n_real)
                res = self._run(kwargs, n_real, phase)
                self._release_received(seam, res)
                return res
            return self._run_boundary(seam, kwargs, phase, fingerprint)
        except BaseException as error:
            seam.poison(f"stage forward: {type(error).__name__}: {error}")
            self._seam_recv_live = False
            raise

    def _release_received(self, seam, res) -> None:
        """The last stage: hand the landing slots its forward read back to the previous stage once that forward has run
        (on the seam's worker after the launch event for an async step, at once for a step read back here)."""
        if not self._seam_recv_live:
            return
        self._seam_recv_live = False
        pending = getattr(res, "neuron_pending_sample", None)
        if pending is None:
            seam.release_all()
            return
        seam.release_after(pending.outputs["token_ids"], keep_alive=pending.outputs, launch_event=pending.launch_event)

    def _run_relaying(self, seam, kwargs: dict, n_real: int):
        """A pipelined decode's last stage: claim the next token send slot, launch the graph that samples into it in
        place, check the write, and hand the step to the seam's worker, which WRITEs the slot to the first stage and
        then hands this stage's landing slots back. The slot is also the pending sample's token vector, so the slot's
        previous step is read back before the slot is reused."""
        from sglang_neuron.backend import device_io, executor_cache
        from sglang_neuron.distributed.pp_seam import TOKEN_KIND

        step, slot = seam.claim_send_slot(TOKEN_KIND, int(self.decode_token_width))
        earlier = self._relay_samples.pop(step % seam.capacity, None)
        earlier = earlier() if earlier is not None else None
        if earlier is not None:
            earlier.materialize(return_logprob=False)  # the slot's previous step on the host before it is written
        slots = {"token_ids_out": slot.reshape(-1)}
        armed = device_io.on_device()
        if armed:
            executor_cache.arm_writeback()  # this call's record only (_check_boundary_in_slots)
        try:
            res = self._run({**kwargs, **slots}, n_real, phase="decode")
        except BaseException:
            if armed:
                executor_cache.take_writeback()
            raise
        pending = getattr(res, "neuron_pending_sample", None)
        if pending is None:
            raise RuntimeError("a pipelined decode's last stage relays a pending sample's device tokens; this step was "
                               "read back synchronously")
        self._check_boundary_in_slots(pending.outputs, slots, "decode")
        self._relay_samples[step % seam.capacity] = weakref.ref(pending)
        self._seam_recv_live = False
        seam.send_claimed([TOKEN_KIND], keep_alive=pending.outputs, launch_event=pending.launch_event)
        return res

    def _run_boundary(self, seam, kwargs: dict, phase, fingerprint: int):
        """A stage before the last: claim the send slots (waiting only for their previous step's ack), launch the graph
        that writes the boundary into them, check it wrote in place, and hand the step to the seam's worker, which
        WRITEs the slots to the next stage once the forward has run. A prefill boundary is the last layer's output and
        residual, SP-sharded over the attention TP ranks; a decode boundary is the last layer's output, TP-replicated,
        one row per batch bucket row."""
        from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
        from sglang.srt.utils.common import DynamicGradMode

        from sglang_neuron.backend import device_io, executor_cache
        from sglang_neuron.distributed.pp_seam import boundary_kinds, prefill_boundary

        tr = None
        if self._host_trace:
            from sglang_neuron.model_executor import host_trace

            tr = host_trace.trace()
        s0 = time.time_ns() if tr is not None else 0
        if phase == "prefill":
            tokens = int(kwargs["input_ids"].shape[0])
            boundary = {name: (spec.kind, spec.sent_rows(tokens, self.model_tp_size))
                        for name, spec in prefill_boundary(self).items()}
        else:
            names, rows = self.pp_decode_boundary, int(kwargs["positions"].shape[0])
            boundary = {name: (kind, rows) for name, kind in zip(names, boundary_kinds(names, phase))}
        kinds = [kind for kind, _rows in boundary.values()]
        slots = {f"{name}_out": seam.claim_send_slot(kind, rows)[1] for name, (kind, rows) in boundary.items()}
        kwargs = {**kwargs, **slots}
        if tr is not None:  # the claim's wait for the slots' previous step, apart from the graph call
            tr.add("seam_claim", s0, time.time_ns(), phase)
            s0 = time.time_ns()
        capture = self._decode_serving_capture(phase)
        if capture is not None:
            capture.before()
        armed = device_io.on_device()
        try:
            with DynamicGradMode(), torch.compiler.set_stance("fail_on_recompile"):
                device_io.before_launch()
                if armed:
                    executor_cache.arm_writeback()  # this call's record only (_check_boundary_in_slots)
                with device_io.pre_execute_scope():  # the seam's landing wait, right before the runtime's submit
                    out = self.compiled(**kwargs)
                launch_event = device_io.after_launch()
        except BaseException:
            if armed:
                executor_cache.take_writeback()  # no stale record for a later call
            raise
        if tr is not None:
            tr.add("submit", s0, time.time_ns(), phase)
        if capture is not None:
            capture.after(next(iter(slots.values())))  # the window's last step waits for its boundary
        self._check_boundary_in_slots(out, slots, phase)
        # a middle stage's own landing slots go back once this forward has run, after its WRITEs (send_claimed)
        self._seam_recv_live = False
        step = seam.send_claimed(list(kinds), keep_alive=(kwargs, out), launch_event=launch_event)
        proxy = PPProxyTensors({"neuron_pp_step": torch.tensor([step], dtype=torch.int64),
                                "neuron_pp_batch": torch.tensor([fingerprint], dtype=torch.int64)})
        # handed on at the launch, while the forward runs: the host trace marks the launch asynchronous
        proxy.neuron_launched = True
        return proxy

    def run_decode(self, fb, md: NeuronForwardMetadata, bucket: int,
                   context: Optional[int] = None, pp_proxy=None) -> LogitsProcessorOutput:
        """A decode forward. On a PP stage ``pp_proxy`` is the previous stage's step descriptor and batch fingerprint
        (None on the first stage), and a non-last stage returns its boundary's descriptor."""
        token_ids = fb.input_ids.to("cpu")[: md.num_tokens]
        if getattr(self, "pp_active", False):
            return self._run_decode_stage(fb, md, bucket, context, token_ids, pp_proxy)
        if pp_proxy is not None:
            raise RuntimeError("a pipeline boundary reached a runtime that is not a pipeline stage")
        kwargs = self.build_model_inputs(md, token_ids=token_ids, bucket=bucket, **self._sampling_kwargs(fb),
                                         token_plan=getattr(fb, "neuron_token_plan", None), context=context)
        return self._run(kwargs, n_real=md.batch_size, phase="decode")

    def _run_decode_stage(self, fb, md, bucket: int, context, token_ids, pp_proxy):
        """A pipelined decode stage's step. Every stage hashes the same ids into the batch fingerprint: the relay's zero
        placeholder on a PD decode, the batch's own ids (which SGLang's PP loop gives every stage) on an aggregated
        server. A PD decode's first stage takes its input ids from the last stage's relay and, after the launch,
        registers the forward's rows as futures of the tokens relayed for it."""
        if not md.batch_size:
            # on a PD decode every stage counts each forward as one relayed token step, so an empty forward on one stage
            # would shift every later step (an idle step goes through run_idle, which a stage refuses)
            raise RuntimeError("a pipelined decode stage runs no forward of zero rows")
        fingerprint = pp_batch_fingerprint(md, token_ids, getattr(fb, "rids", None))
        try:
            kwargs = self.build_model_inputs(md, token_ids=token_ids, bucket=bucket, **self._sampling_kwargs(fb),
                                             token_plan=getattr(fb, "neuron_token_plan", None), context=context)
            if not self.pp_is_first_stage:
                kwargs.update(self._pp_boundary_inputs(fingerprint, bucket, pp_proxy, phase="decode"))
        except BaseException as error:
            self._poison_seam_after_receive(f"input build failed after a seam receive: {error!r}")
            raise
        res = self._run_stage(kwargs, md.batch_size, "decode", fingerprint)
        if self.pp_is_first_stage and self.pp_token_relay:
            from sglang_neuron.distributed.pp_seam import get_seam
            from sglang_neuron.model_executor import async_execution

            slots = fb.req_pool_indices.to("cpu").reshape(-1)[: md.batch_size].tolist()
            generations = self.model_runner.req_to_token_pool.req_generation[slots].tolist()
            async_execution.stash_seam_tokens(slots, generations, self.decode_token_width, get_seam(self))
        return res

    def _sampling_kwargs(self, fb) -> dict:
        """The batch's sampling parameters build_model_inputs consumes in this sampler mode."""
        from sglang_neuron.sampling import DRAWING_MODES

        if getattr(self, "sampler_mode", "greedy") in DRAWING_MODES:
            temperatures = _effective_temperatures(fb)
            seeds = getattr(getattr(fb, "sampling_info", None), "sampling_seed", None)
            if temperatures is None or (seeds is None and bool((temperatures > 0).any())):
                raise RuntimeError(
                    "temperature sampling: the batch carries no sampling temperatures or no per-request "
                    "seeds (SamplingBatchInfo.sampling_seed, set by the plugin's sampling-seed hook)"
                )
            kwargs = {"temperatures": temperatures, "seeds": seeds}
            if self.sampler_mode == "topk_topp":
                si = fb.sampling_info
                kwargs["filters"] = torch.stack([si.top_ks.to("cpu").reshape(-1).float(),
                                                 si.top_ps.to("cpu").reshape(-1).float(),
                                                 si.min_ps.to("cpu").reshape(-1).float()], dim=1)
            return kwargs
        return {}

    def run_idle(self, plan: "StepPlan") -> LogitsProcessorOutput:
        """DP lockstep: dispatch a dummy of the group's phase and bucket (every rank must step)."""
        if getattr(self, "pp_active", False):
            # a dummy would ship a boundary no batch scheduled and put the stages' seam out of step
            raise RuntimeError("a pipeline stage runs no idle step")
        kwargs = self._synthetic_inputs(plan.phase, plan.bucket)
        output = self._run(kwargs, n_real=0, phase=plan.phase)
        if self.async_idle_completion and plan.phase == "decode":
            # the pending output goes through SGLang's bounded result queue, whose copy_done event owns completion,
            # errors and input lifetimes
            return output
        return self.empty_output()

    def empty_output(self) -> LogitsProcessorOutput:
        return LogitsProcessorOutput(next_token_logits=torch.zeros((0, self.hf_config.vocab_size)), hidden_states=None)


@dataclass
class StepPlan:
    kind: str  # "prefill" | "decode" | "idle"
    phase: str  # graph family to dispatch: "prefill" | "decode"
    bucket: int
    context: Optional[int] = None  # a decode's context (BucketConfig.decode_contexts); None: the whole context


_RUNTIME_ENTRY_POINTS = ("__init__", "run_prefill", "run_decode", "run_idle")


def runtime_class_of(sgl_model, runtime_class=None) -> type:
    """The runtime class a model's graphs run under: ``runtime_class`` when given (a megakernel's phase program names
    it), else NeuronModelRuntime or the subclass the model names (models.base.NeuronModel.runtime_class). Refused at
    load: a class that is not a NeuronModelRuntime, and an override of an entry point that drops one of the base
    parameters (a step entry point would raise only on the first served step of its kind, which no warm-up runs)."""
    runtime_class = runtime_class or getattr(sgl_model, "runtime_class", None) or NeuronModelRuntime
    if not (isinstance(runtime_class, type) and issubclass(runtime_class, NeuronModelRuntime)):
        raise ValueError(f"{type(sgl_model).__name__}.runtime_class must be a NeuronModelRuntime subclass, got "
                         f"{runtime_class!r}")
    import inspect

    for name in _RUNTIME_ENTRY_POINTS:
        override, base = getattr(runtime_class, name), getattr(NeuronModelRuntime, name)
        if override is base:
            continue
        params = inspect.signature(override).parameters
        if any(p.kind is p.VAR_KEYWORD for p in params.values()):
            continue
        missing = [p for p in inspect.signature(base).parameters if p not in params]
        if missing:
            raise ValueError(f"{runtime_class.__name__}.{name} takes no {missing}, which the platform passes to every "
                             f"runtime's {name}")
    return runtime_class


class NeuronGraphRunner(BaseRunner):
    """SGLang's decode graph runner slot: precompiles all bucket NEFFs in __init__
    and serves decode via can_run_graph / load_batch / execute."""

    def __init__(self, model_runner):
        super().__init__(model_runner)
        sgl_model = model_runner.model
        ensure = getattr(sgl_model, "_ensure_runtime", None)
        if ensure is None:
            raise ValueError(f"{type(sgl_model).__name__} is not a sglang_neuron.models.base.NeuronModel: the Neuron "
                             "platform serves only the models sglang_neuron.models registers")
        # Dual-phase aware: an aggregated host builds one runtime per phase (compile + KV bind + warmup included,
        # NeuronModel._ensure_runtime); this runner serves decode.
        runtime = ensure(model_runner)
        select = getattr(sgl_model, "_runtime_for", None)
        self.runtime = select("decode") if select is not None else runtime
        self.attn_backend = model_runner.attn_backend
        self.bs = 0
        self.raw_bs = 0
        self.raw_num_token = 0
        self._md: Optional[NeuronForwardMetadata] = None
        self._plan: Optional[StepPlan] = None
        self._step = None  # a megakernel's planned step (load_batch), run by execute

    @property
    def capture_bs(self) -> list:
        """The batch sizes the decode graphs serve, as SGLang reads a graph runner's captured sizes (its startup log,
        the model runner's decode_cuda_graph_capture_bs): the runtime's seqs buckets."""
        return list(self.runtime.buckets.num_seqs_buckets)

    def can_run_graph(self, forward_batch: ForwardBatch) -> bool:
        if not (forward_batch.forward_mode.is_decode() or forward_batch.forward_mode.is_idle()):
            return False
        bs = int(forward_batch.batch_size)
        return bs <= self.runtime.buckets.num_seqs_buckets[-1]

    def load_batch(self, forward_batch: ForwardBatch, **kwargs):
        from sglang_neuron.megakernel import dispatch_of

        dispatch = dispatch_of(getattr(getattr(self, "model_runner", None), "model", None))
        if dispatch is not None:
            # a megakernel plans the step (host state only) and execute runs it
            step = dispatch.plan(forward_batch.input_ids, forward_batch)
            self._md, self._plan, self._step = step.metadata, step.plan, step
            self.raw_bs, self.raw_num_token, self.bs = step.metadata.batch_size, step.metadata.num_tokens, step.rows
            return self
        self._md = self.attn_backend.build_metadata(forward_batch)
        self.raw_bs = self._md.batch_size
        self.raw_num_token = self._md.num_tokens
        self._plan = self.runtime.plan_step(forward_batch, self._md)
        self.bs = self._plan.bucket
        return self

    def execute(self, forward_batch: ForwardBatch, **kwargs):
        if self._md is None:
            self.load_batch(forward_batch)
        md, plan = self._md, self._plan
        self._md = None
        step, self._step = getattr(self, "_step", None), None
        if step is not None:
            from sglang_neuron.megakernel import dispatch_of

            return dispatch_of(self.model_runner.model).run(
                step, forward_batch.input_ids, forward_batch.positions, forward_batch,
                pp_proxy_tensors=kwargs.get("pp_proxy_tensors"))
        proxy = kwargs.get("pp_proxy_tensors")
        if plan.kind == "idle" or md.batch_size == 0:  # idle rank in DP lockstep
            if proxy is not None:
                # a pipeline stage runs no idle step: its boundary belongs to a batch
                raise RuntimeError("a pipeline boundary reached an idle step, which no stage runs")
            # dispatch the group's graph family (an aggregated host's idle rank in a
            # prefill step runs the prefill graph the busy ranks run)
            model = getattr(getattr(self, "model_runner", None), "model", None)
            select = getattr(model, "_runtime_for", None)
            runtime = select(plan.phase) if select is not None else self.runtime
            return runtime.run_idle(plan)
        return self.runtime.run_decode(forward_batch, md, plan.bucket, plan.context, pp_proxy=proxy)
