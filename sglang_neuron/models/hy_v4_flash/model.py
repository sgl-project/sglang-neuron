# SPDX-License-Identifier: Apache-2.0
"""Hy4-Flash's role graphs (one class, ``phase`` prefill or decode) on attention TP with the routed experts over EP.

A layer, on the ``hc_mult`` residual streams [T, hc, H] (the reference's HYV4FlashDecoderLayer):

    streams = engram(streams)                                   # 4 of the layers
    post, x = attn_hc(streams);  streams += post x attention(input_layernorm(x))
    post, x = ffn_hc(streams);   streams += post x mlp(post_attention_layernorm(x))

then ``hc_head`` collapses the streams, the final norm and the vocabulary-sharded LM head. The embedding starts every
stream at the token's embedding. The streams, the iHC mixes and the Engram gates are replicated on every rank; the
attention (attention.py) and the MLPs (moe.py) shard over the TP and end in one all-reduce each.

The graph declares its layers (``layer_names``, ``sliding_windows``): the checkpoint's ``layer_types`` name sparse
layers ``deepseek_sparse_attention``, which the runtime's contract does not derive, and the declared windows give the
sliding layers a decode table trimmed to their window."""

from __future__ import annotations

import torch
from sglang.srt.distributed.parallel_state import get_attn_tp_group
from torch import nn

from sglang_neuron.layers import ColumnParallelLinear
from sglang_neuron.layers.embedding import VocabDimShardedEmbedding
from sglang_neuron.mem_cache.kv_cache import KVSpec, LayerSpec
from sglang_neuron.models.hy_v4_flash import ops, weights
from sglang_neuron.models.hy_v4_flash.attention import Attention, write_rows
from sglang_neuron.models.hy_v4_flash.config import INDEX_DIM, INDEX_RATIO, Hy4FlashConfig
from sglang_neuron.models.hy_v4_flash.engram import Engram, ngram_tokens
from sglang_neuron.models.hy_v4_flash.moe import DenseMLP, LatentMoE
from sglang_neuron.models.transformer import weights as dense_weights
from sglang_neuron.models.transformer.config import DTYPE
from sglang_neuron.models.transformer.model import RMSNorm
from sglang_neuron.parallel import groups
from sglang_neuron.utils.checkpoints import SafetensorsCheckpoint
from sglang_neuron.utils.neuron_utils import can_run_kernel
from sglang_neuron.utils.weight_loader import SafetensorsWeightLoader, set_weight_loader


# a decode of at least this many rows runs the MoE half kernel; one row keeps the graph's MoE, whose expert kernel
# reads only the experts it chose (moe_hits.py): the half kernel reads all 16 local experts (0.66 ms at one row)
KERNEL_MOE_MIN_ROWS = 2
# the decode kernels take at most this many rows (a tile of partitions)
KERNEL_MAX_ROWS = 128
# the sparse layers' decode attention kernel (kernels/attn_swa.attn_sparse) is off: its scores on the vector engine
# (one row's 2052 keys x 576 channels at a time) lost to the graph's attention at 1, 16 and 64 rows (six decode
# layers on one core: 5.03 / 10.73 / 20.50 ms against 4.91 / 9.65 / 17.42)
SPARSE_DECODE_KERNEL = False


class HyperConnection(nn.Module):
    """One block's iHC (``attn_hc`` / ``ffn_hc``): fn [2 hc, hc H], base [2 hc], scale [2], all fp32."""

    def __init__(self, config: Hy4FlashConfig):
        super().__init__()
        self.config = config
        hc, h = config.hc_mult, config.hidden_size
        self.fn = nn.Parameter(torch.empty(2 * hc, hc * h, dtype=torch.float32), requires_grad=False)
        self.base = nn.Parameter(torch.empty(2 * hc, dtype=torch.float32), requires_grad=False)
        self.scale = nn.Parameter(torch.empty(2, dtype=torch.float32), requires_grad=False)
        for p in (self.fn, self.base, self.scale):
            set_weight_loader(p, weights.vector(torch.float32))

    def forward(self, streams):
        c = self.config
        return ops.hc_mix(streams, self.fn, self.base, self.scale, c.hc_mult, c.hc_magnitude, c.hc_eps,
                          c.rms_norm_eps)

    def weight_mappings(self, src: str, dst: str) -> dict:
        return {f"{dst}.fn": f"{src}.hc_fn", f"{dst}.base": f"{src}.hc_base", f"{dst}.scale": f"{src}.hc_scale"}


class DecoderLayer(nn.Module):
    def __init__(self, config: Hy4FlashConfig, layer_idx: int, phase: str, engram: Engram | None):
        super().__init__()
        self.layer_idx = layer_idx
        self.engram = engram
        self.attn_hc = HyperConnection(config)
        self.ffn_hc = HyperConnection(config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = Attention(config, layer_idx)
        self.mlp = LatentMoE(config, phase) if config.moe_layers[layer_idx] else DenseMLP(config)
        # set by Hy4FlashGraph.prepare_kernels on a decode graph's MoE layers and sliding layers
        self.decode_kernel = False
        self.attn_kernel = False

    def forward(self, streams, attend, ngrams, real=None, attend_half=None):
        if self.engram is not None:
            streams = self.engram(streams, ngrams)
        if self.attn_kernel and streams.shape[0] <= KERNEL_MAX_ROWS and can_run_kernel(streams):
            # a sliding layer's mix, norm and attention up to its all-reduce in one kernel (kernels/attn_swa.py)
            partial, post = attend_half(streams, self.attn_hc, self.input_layernorm)
            streams = ops.hc_add(streams, post, self.self_attn.finish_half(partial))
        else:
            post, x = self.attn_hc(streams)
            streams = ops.hc_add(streams, post, attend(self.input_layernorm(x)))
        if self.decode_kernel and streams.shape[0] >= KERNEL_MOE_MIN_ROWS and can_run_kernel(streams):
            # the FFN block's mix, norm and MoE up to its all-reduce in one kernel (kernels/moe_half.py)
            partial, post = self.mlp.decode_half(streams, self.ffn_hc, self.post_attention_layernorm)
            return ops.hc_add(streams, post, self.mlp.finish_half(partial))
        post, x = self.ffn_hc(streams)
        return ops.hc_add(streams, post, self.mlp(self.post_attention_layernorm(x), real))

    def weight_mappings(self, src: str, dst: str) -> dict:
        m = {f"{dst}.input_layernorm.weight": f"{src}.input_layernorm.weight",
             f"{dst}.post_attention_layernorm.weight": f"{src}.post_attention_layernorm.weight"}
        m.update(self.attn_hc.weight_mappings(f"{src}.hc_attn_layer.hc_pre", f"{dst}.attn_hc"))
        m.update(self.ffn_hc.weight_mappings(f"{src}.hc_mlp_layer.hc_pre", f"{dst}.ffn_hc"))
        m.update(self.self_attn.weight_mappings(src, f"{dst}.self_attn"))
        m.update(self.mlp.weight_mappings(src, f"{dst}.mlp"))
        if self.engram is not None:
            m.update(self.engram.weight_mappings(src, f"{dst}.engram"))
        return m


class Hy4FlashGraph(nn.Module):
    """One role graph (``phase`` "prefill" or "decode")."""

    def __init__(self, config: Hy4FlashConfig, phase: str):
        super().__init__()
        if phase not in ("prefill", "decode"):
            raise ValueError(f"a role graph is a prefill or a decode, not {phase!r}")
        self.config = config
        self.phase = phase
        self.tp_group = get_attn_tp_group()
        self.lm_head_tp_group = group = groups.get_neuron_attn_tp_group()
        self.embed_tokens = VocabDimShardedEmbedding(config.vocab_size, config.hidden_size, dtype=DTYPE,
                                                     tp_group=group.device_group)
        engrams = {layer: Engram(config, i) for i, layer in enumerate(config.engram_layers)}
        self.layers = nn.ModuleList([DecoderLayer(config, i, phase, engrams.get(i))
                                     for i in range(config.num_hidden_layers)])
        hc, h = config.hc_mult, config.hidden_size
        self.hc_fn = nn.Parameter(torch.empty(hc, hc * h, dtype=torch.float32), requires_grad=False)
        self.hc_base = nn.Parameter(torch.empty(hc, dtype=torch.float32), requires_grad=False)
        self.hc_scale = nn.Parameter(torch.empty(1, dtype=torch.float32), requires_grad=False)
        for p in (self.hc_fn, self.hc_base, self.hc_scale):
            set_weight_loader(p, weights.vector(torch.float32))
        self.norm = RMSNorm(h, config.rms_norm_eps)
        # held [H, V / tp] (the matmul's layout): an [V / tp, H] weight is transposed on the device every call
        self.lm_head = ColumnParallelLinear(h, config.vocab_size, dtype=DTYPE, gather_output=True,
                                            tp_group=group.device_group, weight_layout="in_out")
        set_weight_loader(self.embed_tokens.weight,
                          dense_weights.vocab_rows_loader(self.embed_tokens.vocab_size_per_rank, h))
        rows = dense_weights.vocab_rows_loader(self.lm_head.out_features_per_rank, h)
        set_weight_loader(self.lm_head.weight, SafetensorsWeightLoader(
            transform=lambda slices, rank, rows=rows: rows.transform(slices, rank).t().contiguous()))
        # the runtime's layer contract (models/contract.declared_layer_kinds)
        self.layer_names = [config.layer_name(i) for i in range(config.num_hidden_layers)]
        self.sliding_windows = config.sliding_windows
        # bound by bind_kv_cache: each KV slot's token id, read by the Engram layers' n-grams
        self.token_history = None

    # -----------------------------------------------------------------------------------------------------------------

    @torch.no_grad()
    def forward(self, input_ids, positions, attn_metadata, sampling_positions, rank):
        cfg = self.config
        ids = input_ids.to(torch.int64)
        hidden = self.embed_tokens(input_ids, scatter_tokens=False, rank=rank)
        streams = hidden.unsqueeze(1).expand(-1, cfg.hc_mult, -1).contiguous()
        full = attn_metadata[self.layers[cfg.sparse_layer_ids[0]].self_attn.name]
        table, block = full["block_table_tensor"], int(full["block_size"])
        write_rows(self.token_history.view(-1, 1), ids.to(torch.int32).view(-1, 1), full["slot_mapping"])
        decode = self.phase == "decode"
        rows_table = torch.arange(ids.shape[0], device=ids.device) if decode else None
        ngrams = (ngram_tokens(self.token_history, ids, positions, table, block, rows_table, cfg.engram_max_ngram,
                               cfg.engram_pad_id) if cfg.engram_layers else None)
        prior = full["cached_seq_len"]
        # a prefill chunk's real rows: those up to its sampled (last real) one
        real = None if decode else torch.arange(ids.shape[0], device=ids.device) <= sampling_positions.reshape(())
        for layer in self.layers:
            attn = layer.self_attn
            md = attn_metadata[attn.name]
            half = None
            if decode:
                attend = lambda x, a=attn, m=md: a.decode(x, positions, m)  # noqa: E731
                half = lambda st, hc, ln, a=attn, m=md: a.decode_half(st, positions, m, hc, ln)  # noqa: E731
            else:
                attend = lambda x, a=attn, m=md: a.prefill(x, positions, m, prior)  # noqa: E731
            streams = layer(streams, attend, ngrams, real, half)
        streams = streams.index_select(0, sampling_positions)
        hidden = ops.hc_head(streams, self.hc_fn, self.hc_base, self.hc_scale, cfg.hc_eps, cfg.rms_norm_eps)
        return self.lm_head(self.norm(hidden))

    # -----------------------------------------------------------------------------------------------------------------

    def get_kv_spec(self) -> KVSpec:
        return KVSpec(layers=[LayerSpec(name=layer.self_attn.name) for layer in self.layers])

    def bind_kv_cache(self, kv_caches: dict) -> None:
        """Bind each layer's K and V pool buffers ([blocks, 1, block, 288], bf16 or FP8) as token rows; the host
        allocates the per-slot state beside them once for both role graphs (bind_state), one row per slot of the full
        pool, which the sparse layers and the token history address."""
        for layer in self.layers:
            k, v = kv_caches[layer.self_attn.name]
            if v.dim() != 4 or v.shape[1] != 1 or k.numel() != v.numel():
                raise ValueError(f"Hy4-Flash keeps one KV head a rank in the pool, not {tuple(k.shape)}: launch with "
                                 "--tp-size equal to the config's 32 KV heads")
            # an FP8 pool's K is laid out for the platform's kernels ([blocks, 1, block / 2, 288, 2]); the pool moves
            # and zeroes whole blocks, so the graphs keep their own token rows inside each block, as in V
            layer.self_attn.k_cache = k.view(-1, v.shape[-1])
            layer.self_attn.v_cache = v.view(-1, v.shape[-1])
        sparse = self.layers[self.config.sparse_layer_ids[0]].self_attn
        self.pool_slots = int(sparse.k_cache.shape[0])
        self.pool_device = sparse.k_cache.device

    def bind_state(self, slots: int, device, shared: dict | None = None) -> dict:
        """Allocate (or, ``shared`` from the other role graph, reuse) the per-slot state."""
        state = shared if shared is not None else {}
        if "history" not in state:
            state["history"] = torch.zeros(slots, dtype=torch.int32, device=device)
        self.token_history = state["history"]
        for layer in self.layers:
            idx = layer.self_attn.indexer
            if idx is None:
                continue
            key = ("index", layer.layer_idx)
            if key not in state:
                state[key] = (torch.zeros(slots, INDEX_DIM, dtype=DTYPE, device=device),
                              torch.zeros(slots // INDEX_RATIO, INDEX_DIM, dtype=DTYPE, device=device))
            idx.token_keys, idx.group_keys = state[key]
        return state

    def hf_weight_mappings(self, rank: int) -> dict:
        lm = "model.embed_tokens.weight" if self.config.tie_word_embeddings else "lm_head.weight"
        m = {"embed_tokens.weight": "model.embed_tokens.weight", "norm.weight": "model.norm.weight",
             "lm_head.weight": lm, "hc_fn": "model.hc_head.hc_head_fn", "hc_base": "model.hc_head.hc_head_base",
             "hc_scale": "model.hc_head.hc_head_scale"}
        for layer in self.layers:
            m.update(layer.weight_mappings(f"model.layers.{layer.layer_idx}", f"layers.{layer.layer_idx}"))
        return m

    def load_weights(self, checkpoint_path: str, device: torch.device) -> None:
        rank = self.tp_group.rank_in_group
        checkpoint = SafetensorsCheckpoint(checkpoint_path)
        state = checkpoint.load_sharded_pipelined(rank, self.tp_group.world_size, self, self.hf_weight_mappings(rank),
                                                  device).state_dict
        self.load_state_dict(state, strict=False, assign=True)
        self.bind_hashes()

    def prepare_kernels(self) -> None:
        """After the weights load: a decode graph's MoE layers take the MoE half kernel (kernels/moe_half.py) and its
        sliding layers the attention half kernel (kernels/attn_swa.py)."""
        if self.phase != "decode":
            return
        for layer in self.layers:
            if isinstance(layer.mlp, LatentMoE):
                layer.mlp.prepare_decode_kernel(layer.ffn_hc)
                layer.decode_kernel = True
            if layer.self_attn.sliding:
                layer.self_attn.prepare_decode_kernel(layer.attn_hc)
                layer.attn_kernel = True
            elif SPARSE_DECODE_KERNEL:
                layer.self_attn.prepare_decode_kernel()
                layer.self_attn.sparse_kernel = True

    def bind_hashes(self) -> None:
        for layer in self.layers:
            if layer.engram is not None:
                layer.engram.bind_hash()
