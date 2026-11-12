# SPDX-License-Identifier: Apache-2.0
"""A dense decoder-only transformer as the runtime's role graphs: one class serves the prefill graph (one
request's chunk per forward) and the decode graph (one token per row), on attention TP.

A prefill runs the QKV kernel, which writes the chunk's K/V into its pages, then the segmented attention kernel over
the request's prior and the chunk. A decode runs one fused attention_block_tkg kernel per layer, which writes the
row's K/V itself. The attention and the MLP each all-reduce their output over the attention TP; the embedding and the
LM head are sharded by vocabulary, and the device sampler reads each rank's shard of the logits."""

import torch
import torch.nn.functional as F
from sglang.srt.distributed.parallel_state import get_attn_tp_group
from torch import nn

import sglang_neuron.ops as NF
from sglang_neuron.layers import ColumnParallelLinear
from sglang_neuron.layers.embedding import VocabDimShardedEmbedding
from sglang_neuron.mem_cache.kv_cache import KVSpec, LayerSpec
from sglang_neuron.models.transformer import weights
from sglang_neuron.models.transformer.config import DTYPE, TransformerConfig
from sglang_neuron.ops.attention.attention_decode_paged import attention_decode_paged, paged_decode_inputs
from sglang_neuron.ops.attention.qkv_paged import qkv_paged
from sglang_neuron.parallel import collectives as neuron_collectives
from sglang_neuron.parallel import groups
from sglang_neuron.utils.checkpoints import SafetensorsCheckpoint
from sglang_neuron.utils.weight_loader import set_weight_loader


class RMSNorm(nn.Module):
    def __init__(self, size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size, dtype=DTYPE))
        self.eps = eps

    def forward(self, x):
        x32 = x.float()
        return self.weight * (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)).to(x.dtype)


class RotaryEmbedding(nn.Module):
    """cos and sin of each position over the rotary half-dimension, padded to half the head with cos 1 and sin 0 for
    the channels a partial RoPE passes through."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        rotary = config.rotary_dim
        inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, rotary, 2, dtype=torch.int64).float() / rotary))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.pass_through = (config.head_dim - rotary) // 2

    def forward(self, positions):
        """[T, head_dim / 2] cos and sin in DTYPE."""
        angles = positions.float()[:, None] * self.inv_freq[None, :]
        cos, sin = angles.cos(), angles.sin()
        if self.pass_through:
            cos = F.pad(cos, (0, self.pass_through), value=1.0)
            sin = F.pad(sin, (0, self.pass_through), value=0.0)
        return cos.to(DTYPE), sin.to(DTYPE)


class Attention(nn.Module):
    def __init__(self, config: TransformerConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.name = f"layers.{layer_idx}.self_attn"
        self.tp_group = get_attn_tp_group()
        self.attn_tp_group = groups.get_neuron_attn_tp_group()
        tp = self.tp_group.world_size
        if config.num_attention_heads % tp:
            raise ValueError(f"{config.num_attention_heads} attention heads do not shard over an attention TP of {tp}")
        if config.num_key_value_heads % tp and tp % config.num_key_value_heads:
            raise ValueError(f"{config.num_key_value_heads} KV heads neither shard over nor replicate across an "
                             f"attention TP of {tp}")
        self.head_dim = d = config.head_dim
        self.num_heads = config.num_attention_heads // tp
        self.num_kv_heads = max(1, config.num_key_value_heads // tp)
        self.scaling = d ** -0.5
        self.norm_eps = config.rms_norm_eps
        width = (self.num_heads + 2 * self.num_kv_heads) * d
        self.qkv_weight = nn.Parameter(torch.empty(config.hidden_size, width, dtype=DTYPE))
        self.qkv_bias = nn.Parameter(torch.empty(width, dtype=DTYPE)) if config.qkv_bias else None
        self.o_weight = nn.Parameter(torch.empty(self.num_heads * d, config.hidden_size, dtype=DTYPE))
        self.q_norm = RMSNorm(d, config.rms_norm_eps) if config.qk_norm else None
        self.k_norm = RMSNorm(d, config.rms_norm_eps) if config.qk_norm else None
        # bound by bind_kv_cache: [pages, kv_heads, page, d] each
        self.k_cache = None
        self.v_cache = None

        shard = (tp, self.num_heads * d, self.num_kv_heads * d, config.num_key_value_heads)
        set_weight_loader(self.qkv_weight, weights.fused_qkv_weight_loader(*shard))
        if self.qkv_bias is not None:
            set_weight_loader(self.qkv_bias, weights.fused_qkv_bias_loader(*shard))
        set_weight_loader(self.o_weight, weights.row_parallel_loader(self.num_heads * d))

    def prefill(self, x, cos, sin, md):
        page = md["block_size"]
        pages = self.k_cache.shape[0]
        tokens = x.shape[0]
        # The runtime pads with slot 0, on SGLang's reserved page. The QKV kernel writes K/V at its slots as DMA offsets
        # with no bounds check, so any slot outside the pool goes to slot 0 too instead of past the cache.
        slots = md["slot_mapping"]
        slots = torch.where((slots < 0) | (slots >= pages * page), torch.zeros_like(slots), slots)
        q, k_cache, v_cache = qkv_paged(
            x.unsqueeze(0),
            weights=self.qkv_weight,
            bias=None if self.qkv_bias is None else self.qkv_bias.unsqueeze(0),
            # the kernel's RoPE tables cover the whole head, the halves repeated
            cos=torch.cat([cos, cos], dim=-1).unsqueeze(0),
            sin=torch.cat([sin, sin], dim=-1).unsqueeze(0),
            q_norm=None if self.q_norm is None else self.q_norm.weight.unsqueeze(0),
            k_norm=None if self.k_norm is None else self.k_norm.weight.unsqueeze(0),
            norm_eps=self.norm_eps,
            d_head=self.head_dim,
            num_q_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            k_cache=self.k_cache,
            v_cache=self.v_cache,
            block_size=page,
            slot_mapping=slots.to(torch.int32).reshape(1, tokens),
        )
        out = NF.segmented_attention(
            q.view(tokens, self.num_heads, self.head_dim).transpose(0, 1),
            k_cache=k_cache,
            v_cache=v_cache,
            block_tables=md["block_table_tensor"],
            prior_tokens=md["cached_seq_len"],
            block_size=page,
            kv_segment_size=md["kv_segment_size"],
            scale=self.scaling,
            sliding_window=None,
            sink=None,
        )
        out = out.transpose(0, 1).reshape(tokens, -1) @ self.o_weight
        return neuron_collectives.all_reduce(self.attn_tp_group, out)

    def decode(self, x, cos, sin, inputs):
        out = attention_decode_paged(
            x.unsqueeze(1),
            w_qkv=self.qkv_weight,
            bias_qkv=None if self.qkv_bias is None else self.qkv_bias.unsqueeze(0),
            q_norm=None if self.q_norm is None else self.q_norm.weight.unsqueeze(0),
            k_norm=None if self.k_norm is None else self.k_norm.weight.unsqueeze(0),
            norm_eps=self.norm_eps,
            # [B, d / 2] -> the kernel's [d / 2, B, 1]
            cos=cos.t().unsqueeze(-1).contiguous(),
            sin=sin.t().unsqueeze(-1).contiguous(),
            k_cache=self.k_cache,
            v_cache=self.v_cache,
            inputs=inputs,
            softmax_scale=self.scaling,
            w_out=self.o_weight,
        )
        return neuron_collectives.all_reduce(self.attn_tp_group, out)


class MLP(nn.Module):
    """SiLU-gated: down(silu(gate(x)) * up(x)), the intermediate dimension sharded over the attention TP."""

    def __init__(self, config: TransformerConfig, fused_gate_up: bool):
        super().__init__()
        self.attn_tp_group = groups.get_neuron_attn_tp_group()
        tp = self.attn_tp_group.world_size
        if config.intermediate_size % tp:
            raise ValueError(f"an intermediate size of {config.intermediate_size} does not shard over {tp} ranks")
        rows = config.intermediate_size // tp
        self.gate_up_weight = nn.Parameter(torch.empty(config.hidden_size, 2 * rows, dtype=DTYPE))
        self.down_weight = nn.Parameter(torch.empty(rows, config.hidden_size, dtype=DTYPE))
        set_weight_loader(self.gate_up_weight, weights.gate_up_loader(rows, fused_gate_up))
        set_weight_loader(self.down_weight, weights.row_parallel_loader(rows))

    def forward(self, x):
        gate, up = (x @ self.gate_up_weight).chunk(2, dim=-1)
        return neuron_collectives.all_reduce(self.attn_tp_group, (F.silu(gate) * up) @ self.down_weight)


class DecoderLayer(nn.Module):
    def __init__(self, config: TransformerConfig, layer_idx: int, mlp: nn.Module):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = Attention(config, layer_idx)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = mlp
        self.post_self_attn_layernorm = (RMSNorm(config.hidden_size, config.rms_norm_eps)
                                         if config.post_attention_norm else None)
        self.post_mlp_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps) if config.post_mlp_norm else None

    def forward(self, x, attention):
        """``attention(normed)`` is this layer's attention on its phase's inputs."""
        h = attention(self.input_layernorm(x))
        if self.post_self_attn_layernorm is not None:
            h = self.post_self_attn_layernorm(h)
        x = x + h
        h = self.mlp(self.post_attention_layernorm(x))
        if self.post_mlp_layernorm is not None:
            h = self.post_mlp_layernorm(h)
        return x + h


class TransformerForCausalLM(nn.Module):
    """One role graph (``phase`` "prefill" or "decode"). A model on this core passes its config, its MLP, the
    checkpoint's name for each parameter (``hf_weight_mappings``) and whether its gate and up projections are fused."""

    fused_gate_up = False

    def __init__(self, config: TransformerConfig, phase: str):
        super().__init__()
        if phase not in ("prefill", "decode"):
            raise ValueError(f"a role graph is a prefill or a decode, not {phase!r}")
        self.config = config
        self.phase = phase
        self.tp_group = get_attn_tp_group()
        self.lm_head_tp_group = groups.get_neuron_attn_tp_group()
        group = self.lm_head_tp_group
        self.embed_tokens = VocabDimShardedEmbedding(config.vocab_size, config.hidden_size, dtype=DTYPE,
                                                     tp_group=group.device_group)
        self.layers = nn.ModuleList([DecoderLayer(config, i, self.build_mlp(config, i))
                                     for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(config)
        self.lm_head = ColumnParallelLinear(config.hidden_size, config.vocab_size, dtype=DTYPE, gather_output=True,
                                            tp_group=group.device_group)
        set_weight_loader(self.embed_tokens.weight,
                          weights.vocab_rows_loader(self.embed_tokens.vocab_size_per_rank, config.hidden_size))
        if config.tie_word_embeddings:
            # one device copy: the LM head's vocabulary shard is the embedding's (the TP size divides the vocabulary,
            # which the LM head refuses otherwise)
            self.lm_head.weight = self.embed_tokens.weight
        else:
            set_weight_loader(self.lm_head.weight,
                              weights.vocab_rows_loader(self.lm_head.out_features_per_rank, config.hidden_size))

    def build_mlp(self, config: TransformerConfig, layer_idx: int) -> nn.Module:
        return MLP(config, self.fused_gate_up)

    @torch.no_grad()
    def forward(self, input_ids, positions, attn_metadata, sampling_positions, rank):
        hidden = self.embed_tokens(input_ids, scatter_tokens=False, rank=rank)
        cos, sin = self.rotary_emb(positions)
        first = attn_metadata[self.layers[0].self_attn.name]
        if self.phase == "decode":
            attn = self.layers[0].self_attn
            inputs = paged_decode_inputs(first["block_table_tensor"], first["slot_mapping"], positions,
                                         kv_heads=attn.num_kv_heads, q_heads=attn.num_heads,
                                         page=first["block_size"])
            for layer in self.layers:
                hidden = layer(hidden, lambda x, a=layer.self_attn: a.decode(x, cos, sin, inputs))
        else:
            for layer in self.layers:
                md = attn_metadata[layer.self_attn.name]
                hidden = layer(hidden, lambda x, a=layer.self_attn, md=md: a.prefill(x, cos, sin, md))
        hidden = self.norm(hidden.index_select(0, sampling_positions))
        return self.lm_head(hidden)

    def get_kv_spec(self) -> KVSpec:
        return KVSpec(layers=[LayerSpec(name=layer.self_attn.name) for layer in self.layers])

    def bind_kv_cache(self, kv_caches: dict) -> None:
        for layer in self.layers:
            attn = layer.self_attn
            attn.k_cache, attn.v_cache = kv_caches[attn.name]

    def hf_weight_mappings(self) -> dict:
        """Parameter name to its checkpoint key, or the keys a fused parameter is built from."""
        mappings = {"embed_tokens.weight": "model.embed_tokens.weight", "norm.weight": "model.norm.weight"}
        if not self.config.tie_word_embeddings:
            mappings["lm_head.weight"] = "lm_head.weight"
        for i, layer in enumerate(self.layers):
            src, dst = f"model.layers.{i}", f"layers.{i}"
            attn = f"{src}.self_attn"
            mappings.update({
                f"{dst}.self_attn.qkv_weight": [f"{attn}.{n}_proj.weight" for n in "qkv"],
                f"{dst}.self_attn.o_weight": f"{attn}.o_proj.weight",
                f"{dst}.input_layernorm.weight": f"{src}.input_layernorm.weight",
                f"{dst}.post_attention_layernorm.weight": f"{src}.post_attention_layernorm.weight",
            })
            if layer.self_attn.qkv_bias is not None:
                mappings[f"{dst}.self_attn.qkv_bias"] = [f"{attn}.{n}_proj.bias" for n in "qkv"]
            if layer.self_attn.q_norm is not None:
                mappings[f"{dst}.self_attn.q_norm.weight"] = f"{attn}.q_norm.weight"
                mappings[f"{dst}.self_attn.k_norm.weight"] = f"{attn}.k_norm.weight"
            if layer.post_self_attn_layernorm is not None:
                mappings[f"{dst}.post_self_attn_layernorm.weight"] = f"{src}.post_self_attn_layernorm.weight"
            if layer.post_mlp_layernorm is not None:
                mappings[f"{dst}.post_mlp_layernorm.weight"] = f"{src}.post_mlp_layernorm.weight"
            mappings.update(self.mlp_weight_mappings(src, dst))
        return mappings

    def mlp_weight_mappings(self, src: str, dst: str) -> dict:
        gate_up = ([f"{src}.mlp.gate_up_proj.weight"] if self.fused_gate_up
                   else [f"{src}.mlp.gate_proj.weight", f"{src}.mlp.up_proj.weight"])
        return {f"{dst}.mlp.gate_up_weight": gate_up, f"{dst}.mlp.down_weight": f"{src}.mlp.down_proj.weight"}

    def load_weights(self, checkpoint_path: str, device: torch.device) -> None:
        """Read this rank's shard of every mapped parameter onto ``device``."""
        checkpoint = SafetensorsCheckpoint(checkpoint_path)
        state = checkpoint.load_sharded_pipelined(self.tp_group.rank_in_group, self.tp_group.world_size, self,
                                                  self.hf_weight_mappings(), device).state_dict
        self.load_state_dict(state, strict=False, assign=True)
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
