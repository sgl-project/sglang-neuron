# SPDX-License-Identifier: Apache-2.0
"""The GLM-5.2-family decode graph: one decode step of the whole model in the GLM decode giga kernel (GK), on the
plugin's runtime, for the batch SGLang schedules.

The GK (vendored: sglang_neuron/models/glm52/kernels/decode, called through models/glm52/decode/model.py) is one SPMD program
over the node's 128 ranks: DP2 attention halves of 64 ranks, each a CP4 x token16 grid. It scores a fixed set of
``num_slots`` sequences (32 at the 128k point), each on the 4 CP ranks of one token group, each rank holding its CP
quarter of that sequence's KV in its own caches (``kv_cache_all``, ``idx_k_cache_all``) behind a static identity block
table. The geometry (models/glm52/layout/decode/gk_geometry.py) is the kernel's.

A step here:

1. The batch's tokens arrive in SGLang's row order (the platform's decode token I/O assembles them on the device from
   the step before). The embedding is sharded by row over the 128 ranks: each rank contributes the rows of the batch's
   tokens that it owns, and one all-reduce over the world gives every rank every row (a full table on every rank
   would cost 154,880 x 6,144 bf16, 1.9 GB of HBM a rank).
2. A request keeps its kernel sequence slot for its whole life (``slot = req_pool_idx - 1``, the runtime's
   ``gk_slot_rows`` / ``gk_row_slots``). A rank's entry residual is its sequence's token on the rank's own spec rows:
   rows ``t_shard * cp + t`` of the sequence; with no MTP draft only spec row 0 is a real token, which only the
   ``cp == 0`` rank of the token group owns.
3. The GK runs every layer, writes this step's KV rows into its caches in place, and its LM-head tail samples each
   token-table row on device: greedy ``[64, 1, 0, 0]`` rows under SGLANG_NEURON_SAMPLER=greedy; under topk_topp each
   request's ``[top_k, top_p, temperature, u]`` over 256 candidates, built in this graph from the platform's staged
   per-row sampling inputs (models/glm52/tail_sampling.py).
4. The tail's tokens, gathered over the world in rank order, are the token table ``[num_slots * spec]``; each batch
   row takes its slot's spec row 0.

The per-rank step tensors (RoPE rows at the sequence's position, the latent and indexer write rows, the residency
threshold) are the geometry's (``step_tensors``), built on the host by the runtime
(models/glm52/decode/runtime.py) from the positions SGLang already holds there, so a step reads nothing back from the
device."""

from __future__ import annotations

import logging

import numpy as np
import torch
from torch import nn

from sglang_neuron import sampling
from sglang_neuron.models.glm52 import tail_sampling
from sglang_neuron.models.glm52.layout.decode import idx_pool

logger = logging.getLogger(__name__)

# the LM-head tail's candidates per row when every request is greedy (a greedy row reads candidate 0 only; 64 is the
# tail every greedy accuracy gate ran); a sampling server's is sampling.GK_TAIL_SAMPLE_K (256), a different NEFF
LM_HEAD_K = 64
# the step tensors the runtime builds for this rank, in the kernel's names (decode.model.Glm52GkDecodeModel.forward)
STEP_TENSORS = ("cos", "sin", "idx_cos", "idx_sin", "kv_new_slot_mapping", "idx_new_slot_mapping", "idx_positions")
# and, at a point that deals its indexer pool in chunks (models/glm52/points.py), the pool's per-step tables, by the
# names the pool's builder returns them under (layout/decode/idx_pool.IdxChunkPool.tables)
POOL_TENSORS = idx_pool.CDP_TABLES


class _NoLmHead(nn.Module):
    """The LM head lives in the GK's tail; ModelHost sets ``gather_output`` on a graph's head."""

    gather_output = False


class GlmGkDecodeGraph(nn.Module):
    """One rank's decode graph (the module docstring)."""

    # the GK's tail samples: ModelHost runs the graph without the device sampler (models/host.sampled_execution), and
    # the runner hands it the drawing modes' per-row inputs (graph_runner._sampling_inputs) under topk_topp
    samples_in_forward = True
    sampler_modes = ("greedy", "topk_topp")

    def __init__(self, config, geo, rank: int):
        super().__init__()
        from sglang_neuron.models.glm52.decode import model as gk_model

        self.config = config
        self.geo = geo
        self.rank = int(rank)
        self.gk = gk_model.Glm52GkDecodeModel(geo, rank_id=self.rank, device="cpu")
        self.lm_head = _NoLmHead()
        # the layers it declares to the platform's model contract (models/contract.declared_layer_kinds): the
        # checkpoint's DSA layers are the kernel's, not a full or sliding attention the runtime builds metadata for
        self.layer_names = [f"model.layers.{i}.self_attn" for i in range(int(config.num_hidden_layers))]
        self.sliding_windows = [None] * int(config.num_hidden_layers)
        c = geo.coords(self.rank)
        self.coords = c
        self.num_slots = geo.dp_size * geo.num_seqs
        self.slot = geo.slot_of(self.rank)
        self.vocab_shard = geo.vocab_size // geo.ranks
        # this rank's spec rows of its sequence that carry a real token: spec row 0 alone without an MTP draft
        own = np.arange(geo.t_shard) + geo.t_shard * c["cp"]
        self.register_buffer("_my_slot", torch.tensor([self.slot], dtype=torch.int64), persistent=False)
        self.register_buffer("_own_live", torch.from_numpy((own == 0).astype(np.float32)).reshape(-1, 1),
                             persistent=False)
        self.register_buffer("_vocab_lo", torch.tensor(self.rank * self.vocab_shard, dtype=torch.int64),
                             persistent=False)
        # the token-table row of each slot's spec row 0
        self.register_buffer("_slot_rows0", torch.arange(self.num_slots, dtype=torch.int64) * geo.spec,
                             persistent=False)
        # the tail's rows: greedy [k, 1, 0, 0] for every row, or (topk_topp) each request's, built in forward
        self.samples = sampling.sampler_mode() == "topk_topp"
        self.lm_head_k = sampling.GK_TAIL_SAMPLE_K if self.samples else LM_HEAD_K
        self.register_buffer("_sampling_params", tail_sampling.filler_rows(geo.t_shard, self.lm_head_k),
                             persistent=False)
        if self.samples:
            # each own row's spec row (its input position past the sequence's) and the filter columns read
            self.register_buffer("_own_spec_rows", torch.from_numpy(own.astype(np.int32)), persistent=False)
            self.register_buffer("_filter_columns", torch.tensor([0, 1], dtype=torch.int64), persistent=False)
        # filled by load_weights: the embedding and lm_head row shards (with a zero row at 0 for out-of-shard tokens),
        # the final norm
        self.register_buffer("_embed_pad", torch.zeros(self.vocab_shard + 1, geo.hidden, dtype=torch.bfloat16),
                             persistent=False)
        self.register_buffer("_lm_head_wt", torch.zeros(geo.hidden, self.vocab_shard, dtype=torch.bfloat16),
                             persistent=False)
        self.register_buffer("_final_norm_w", torch.ones(1, geo.hidden, dtype=torch.float32), persistent=False)

    # the platform's model-graph interface
    def get_kv_spec(self):
        from sglang_neuron.models.glm52.decode import pools as decode_pools

        return decode_pools.accounting_kv_spec(self.config)

    def in_place_targets(self) -> dict:
        """The GK's caches, which every step writes in place and a PD prefill writes into (decode.pools): the warm-up
        proves the NEFF writes each in place, not into a copy past a scratchpad page (graph_runner.check_in_place)."""
        caches = getattr(self.gk, "caches", None)
        if caches is None:
            raise RuntimeError("the GK's caches are not allocated: load_weights first")
        targets = {f"kv_cache_{i}" if i else "kv_cache_all": stack for i, stack in enumerate(caches.latent_stacks)}
        targets["idx_k_cache_all"] = caches.idx_k_cache_all
        return targets

    def bind_kv_cache(self, caches: dict) -> None:
        """No layer reads SGLang's pool (``get_kv_spec``): the GK's caches are its own (``load_weights``)."""
        if caches:
            raise ValueError(f"the GLM GK decode reads no SGLang KV layer, but was handed {sorted(caches)}")

    def load_weights(self, path: str, device) -> None:
        """This rank's GK weight stacks from the weight store (``decode.pools.weight_store``), its embedding and lm_head
        row shards and the final norm from the checkpoint, and zeroed caches, all onto ``device``."""
        from sglang_neuron.models.glm52.decode import pools as decode_pools
        from sglang_neuron.models.glm52.layout.decode import weight_loaders_mxfp4 as w

        self.gk.device_str = str(device)
        store = decode_pools.weight_store(self.geo)
        if store == decode_pools.ZERO_WEIGHTS:
            # built on the host and copied over: torch-neuronx has no device factory of uint16, the MX tiles' dtype
            from sglang_neuron.models.glm52.decode import model as gk_model

            for name, tensor in w.zero_rank_weights(self.geo, device="cpu").as_kernel_kwargs().items():
                self.gk.register_buffer(gk_model.GK_STACK_ATTR_PREFIX + name, tensor.to(device), persistent=False)
            logger.warning("[glm52] rank %d runs the GK over zero weights (%s=%s)", self.rank,
                           decode_pools.WEIGHT_STORE_ENV, store)
        else:
            self.gk.load_weights(store, path, device=device)
        head = w.load_head_shards(path, self.geo, self.rank)
        embed = torch.from_numpy(np.ascontiguousarray(head["embed_tokens"])).view(torch.bfloat16)
        self._embed_pad = torch.cat([torch.zeros(1, self.geo.hidden, dtype=torch.bfloat16), embed]).to(device)
        lm_head = torch.from_numpy(np.ascontiguousarray(head["lm_head"])).view(torch.bfloat16)
        self._lm_head_wt = lm_head.t().contiguous().to(device)
        self._final_norm_w = torch.from_numpy(head["norm_weight"].astype(np.float32)).reshape(1, -1).to(device)
        self.gk.allocate_own_caches(indexer_fill="zeros")

    def forward(self, input_ids, positions, attn_metadata=None, sampling_positions=None, rank=None, *,
                gk_slot_rows, gk_row_slots, gk_cos, gk_sin, gk_idx_cos, gk_idx_sin, gk_kv_new_slot_mapping,
                gk_idx_new_slot_mapping, gk_idx_positions, sampling_temperatures=None, sampling_seeds=None,
                sampling_filters=None, gk_idx_pool_items=None, gk_idx_pool_n_items=None, gk_idx_pool_region_base=None,
                gk_idx_pool_new_key_rows=None, gk_idx_pool_unrot=None):
        from sglang_neuron.parallel import collectives
        from sglang_neuron.parallel.groups import get_neuron_attn_tp_group

        group = get_neuron_attn_tp_group()
        ids = input_ids.reshape(-1).to(torch.int64)
        # 1. the batch's embedding rows: this rank's shard's rows of the batch's tokens, all-reduced over the world
        local = ids - self._vocab_lo
        inside = (local >= 0) & (local < self.vocab_shard)
        rows = self._embed_pad.index_select(0, torch.where(inside, local + 1, torch.zeros_like(local)))
        batch = collectives.all_reduce(group, rows)
        # 2. this rank's sequence: its batch row, and its token on the rank's own spec rows
        row = gk_slot_rows.index_select(0, self._my_slot).to(torch.int64)
        live = (row >= 0).to(torch.float32)
        entry = batch.index_select(0, torch.clamp(row, min=0)).to(torch.float32)
        x_shard = (entry * live * self._own_live).to(torch.bfloat16)
        # 3. the whole stack, its tail sampling each of this rank's token-table rows: greedy, or each live row with
        # its request's parameters and its own draw (tail_sampling.gk_tail_rows)
        params = self._sampling_params
        if self.samples:
            if sampling_temperatures is None or sampling_seeds is None or sampling_filters is None:
                raise RuntimeError("the topk_topp GK decode needs the step's sampling_temperatures, sampling_seeds "
                                   "and sampling_filters (graph_runner._sampling_inputs)")
            params = tail_sampling.gk_tail_rows(row, self._own_live, positions.reshape(-1), self._own_spec_rows,
                                                sampling_temperatures, sampling_seeds, sampling_filters,
                                                self._sampling_params, self._filter_columns, self.lm_head_k)
        head = {"lm_head_weight_t": self._lm_head_wt, "final_norm_w": self._final_norm_w,
                "sampling_params": params, "k": self.lm_head_k}
        pool = None
        if self.geo.idx_pool_chunk:
            pool = {"idx_pool_items": gk_idx_pool_items, "idx_pool_n_items": gk_idx_pool_n_items,
                    "idx_pool_region_base": gk_idx_pool_region_base, "idx_pool_new_key_rows": gk_idx_pool_new_key_rows,
                    "idx_pool_unrot": gk_idx_pool_unrot}
        _, tokens = self.gk(x_shard, gk_cos, gk_sin, gk_idx_cos, gk_idx_sin, gk_kv_new_slot_mapping,
                            gk_idx_new_slot_mapping, gk_idx_positions, head=head, idx_pool=pool)
        # 4. the token table in rank order (gathered as float32, which holds every id below 2**24 exactly), then
        # each batch row's slot's spec row 0
        table = collectives.all_gather(group, tokens.to(torch.int32).reshape(-1, 1).to(torch.float32), dim=0)
        slot_tokens = table.reshape(-1).index_select(0, self._slot_rows0)
        token_ids = slot_tokens.index_select(0, gk_row_slots.to(torch.int64)).to(torch.int32)
        return {"token_ids": token_ids}
