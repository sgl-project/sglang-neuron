# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V2 as the runtime's role graphs: one class serves the prefill graph (one request per forward, its
whole prompt in one window) and the decode graph (one token per row), on attention TP.

Attention is multi-head latent attention, absorbed: a token's KV is cached as one compact row, its normalized latent and
its rotated RoPE key, and each query head scores the latent through the key up-projection folded into it
(ops/attention/mla). The first first_k_dense_replace layers have a dense MLP; the others route each token to its top-k
experts, spread over the attention-TP ranks in contiguous blocks, beside the shared experts, which every rank holds a TP
shard of. Each attention and MLP output is all-reduced once over the attention TP; the embedding and the LM head are
sharded by vocabulary, and the device sampler reads each rank's shard of the logits."""

import torch
import torch.nn.functional as F
from sglang.srt.distributed.parallel_state import get_attn_tp_group
from torch import nn

from sglang_neuron.layers import ColumnParallelLinear
from sglang_neuron.layers.embedding import VocabDimShardedEmbedding
from sglang_neuron.mem_cache.kv_cache import KVSpec, LayerSpec
from sglang_neuron.models.deepseek_v2 import weights
from sglang_neuron.models.deepseek_v2.config import DTYPE, DeepseekV2Config
from sglang_neuron.models.transformer import weights as dense_weights
from sglang_neuron.models.transformer.model import RMSNorm
from sglang_neuron.ops.attention.mla import mla_decode, mla_prefill, token_slots
from sglang_neuron.ops.moe.experts import route, routed_experts
from sglang_neuron.parallel import collectives as neuron_collectives
from sglang_neuron.parallel import groups
from sglang_neuron.utils.checkpoints import SafetensorsCheckpoint
from sglang_neuron.utils.weight_loader import set_weight_loader


class RotaryEmbedding(nn.Module):
    """YaRN cos and sin of each position over the RoPE half-dimension, computed in fp32."""

    def __init__(self, config: DeepseekV2Config):
        super().__init__()
        inv_freq, self.scale = config.rope_inv_freq_and_scale()
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, positions):
        angles = positions.float()[:, None] * self.inv_freq[None, :]
        return (angles.cos() * self.scale).to(DTYPE), (angles.sin() * self.scale).to(DTYPE)


def _rotate(x, cos, sin):
    """RoPE over contiguous halves: x [T, heads, d], cos and sin [T, d / 2]."""
    half = x.shape[-1] // 2
    c = torch.cat([cos, cos], dim=-1)[:, None]
    s = torch.cat([sin, sin], dim=-1)[:, None]
    return x * c + torch.cat([-x[..., half:], x[..., :half]], dim=-1) * s


def compact_rows(cache):
    """A layer's cache as the graphs address it, one row per token slot [slots, 1, dim]: SGLang's MLA pool buffer as it
    is, the paged latent pool's [blocks, block, dim] viewed (no copy) as its rows."""
    if cache.dim() == 3 and cache.shape[1] == 1:
        return cache
    if cache.dim() != 3:
        raise ValueError(f"an MLA layer cache of shape {tuple(cache.shape)}: expected [slots, 1, dim] or "
                         "[blocks, block, dim]")
    return cache.view(-1, 1, cache.shape[-1])


def _write_rows(cache, rows, slots):
    """Write the compact ``rows`` [T, dim] into ``cache`` [slots, 1, dim] at ``slots``. A padding token's slot (0, -1
    or one past the pool) writes the reserved slot 0, which no attention reads."""
    flat = cache.view(cache.shape[0], -1)
    slots = slots.to(torch.int64)
    slots = torch.where((slots < 0) | (slots >= flat.shape[0]), torch.zeros_like(slots), slots)
    flat[slots] = rows.to(flat.dtype)


class Attention(nn.Module):
    def __init__(self, config: DeepseekV2Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.name = f"layers.{layer_idx}.self_attn"
        self.tp_group = get_attn_tp_group()
        self.attn_tp_group = groups.get_neuron_attn_tp_group()
        tp = self.tp_group.world_size
        if config.num_attention_heads % tp:
            raise ValueError(f"{config.num_attention_heads} attention heads do not shard over an attention TP of {tp}")
        self.num_heads = heads = config.num_attention_heads // tp
        hidden, latent = config.hidden_size, config.kv_lora_rank
        self.q_weight = nn.Parameter(torch.empty(hidden, heads * config.q_head_dim, dtype=DTYPE))
        self.kv_a_weight = nn.Parameter(torch.empty(hidden, config.compact_dim, dtype=DTYPE))
        self.kv_a_layernorm = RMSNorm(latent, config.rms_norm_eps)
        self.w_uk = nn.Parameter(torch.empty(heads, config.qk_nope_head_dim, latent, dtype=DTYPE))
        self.w_uv = nn.Parameter(torch.empty(heads, config.v_head_dim, latent, dtype=DTYPE))
        self.o_weight = nn.Parameter(torch.empty(heads * config.v_head_dim, hidden, dtype=DTYPE))
        self.scale = config.softmax_scale()
        # bound by bind_kv_cache: SGLang's MLA pool buffer of the layer, [slots, 1, latent + rope]
        self.cache = None

        set_weight_loader(self.q_weight, weights.q_loader(config, heads))
        set_weight_loader(self.kv_a_weight, weights.kv_a_loader(config))
        set_weight_loader(self.w_uk, weights.kv_b_loader(config, heads, values=False))
        set_weight_loader(self.w_uv, weights.kv_b_loader(config, heads, values=True))
        set_weight_loader(self.o_weight, dense_weights.row_parallel_loader(heads * config.v_head_dim))

    def _project(self, x, cos, sin):
        """(the absorbed query's latent part [T, heads, latent], its RoPE part [T, heads, rope], the tokens' compact
        rows [T, latent + rope])."""
        cfg = self.config
        q = (x @ self.q_weight).view(x.shape[0], self.num_heads, cfg.q_head_dim)
        q_nope, q_pe = q.split([cfg.qk_nope_head_dim, cfg.qk_rope_head_dim], dim=-1)
        latent, k_pe = (x @ self.kv_a_weight).split([cfg.kv_lora_rank, cfg.qk_rope_head_dim], dim=-1)
        q_pe = _rotate(q_pe, cos, sin)
        k_pe = _rotate(k_pe[:, None], cos, sin)[:, 0]
        compact = torch.cat([self.kv_a_layernorm(latent), k_pe], dim=-1)
        return torch.einsum("thd,hdr->thr", q_nope, self.w_uk), q_pe, compact

    def _output(self, attended):
        """The attention output of the attended latents [T, heads, latent]: each head's values, projected out and
        all-reduced over the attention TP."""
        values = torch.einsum("thr,hdr->thd", attended, self.w_uv)
        return neuron_collectives.all_reduce(self.attn_tp_group, values.reshape(values.shape[0], -1) @ self.o_weight)

    def prefill(self, x, cos, sin, md, length):
        ql_nope, q_pe, compact = self._project(x, cos, sin)
        width = x.shape[0]
        slots = token_slots(md["block_table_tensor"], md["block_size"], width)
        attended = mla_prefill(ql_nope.transpose(0, 1)[None], q_pe.transpose(0, 1)[None], compact, self.cache, slots,
                               md["cached_seq_len"], length, scale=self.scale)
        _write_rows(self.cache, compact, md["slot_mapping"])
        return self._output(attended[0].transpose(0, 1))

    def decode(self, x, cos, sin, md, slots, seq_lens):
        ql_nope, q_pe, compact = self._project(x, cos, sin)
        attended = mla_decode(ql_nope, q_pe, compact, self.cache, slots, seq_lens, scale=self.scale)
        _write_rows(self.cache, compact, md["slot_mapping"])
        return self._output(attended)


class MLP(nn.Module):
    """SiLU-gated: down(silu(gate(x)) * up(x)), the intermediate dimension sharded over the attention TP. ``reduce``
    all-reduces the output; the shared experts leave their partial to the MoE's one reduce."""

    def __init__(self, config: DeepseekV2Config, intermediate: int, reduce: bool = True):
        super().__init__()
        self.attn_tp_group = groups.get_neuron_attn_tp_group()
        tp = self.attn_tp_group.world_size
        if intermediate % tp:
            raise ValueError(f"an intermediate size of {intermediate} does not shard over {tp} ranks")
        rows = intermediate // tp
        self.reduce = reduce
        self.gate_up_weight = nn.Parameter(torch.empty(config.hidden_size, 2 * rows, dtype=DTYPE))
        self.down_weight = nn.Parameter(torch.empty(rows, config.hidden_size, dtype=DTYPE))
        set_weight_loader(self.gate_up_weight, dense_weights.gate_up_loader(rows, fused=False))
        set_weight_loader(self.down_weight, dense_weights.row_parallel_loader(rows))

    def forward(self, x):
        gate, up = (x @ self.gate_up_weight).chunk(2, dim=-1)
        out = (F.silu(gate) * up) @ self.down_weight
        return neuron_collectives.all_reduce(self.attn_tp_group, out) if self.reduce else out


class MoE(nn.Module):
    """The routed experts, each rank a contiguous block of them, and the shared experts; the routed partials, scaled,
    and the shared experts' TP partial are all-reduced together."""

    def __init__(self, config: DeepseekV2Config, phase: str):
        super().__init__()
        self.attn_tp_group = groups.get_neuron_attn_tp_group()
        tp = self.attn_tp_group.world_size
        if config.n_routed_experts % tp:
            raise ValueError(f"{config.n_routed_experts} routed experts do not split over {tp} ranks")
        self.num_local = config.n_routed_experts // tp
        self.top_k = config.num_experts_per_tok
        self.scaling = config.routed_scaling_factor
        # the decode runs moe_tkg on a NeuronCore; a prefill's tokens run each local expert in torch
        self.fused = phase == "decode"
        hidden, inner = config.hidden_size, config.moe_intermediate_size
        self.router_weight = nn.Parameter(torch.empty(config.n_routed_experts, hidden, dtype=DTYPE))
        self.gate_up = nn.Parameter(torch.empty(self.num_local, hidden, 2, inner, dtype=DTYPE))
        self.down = nn.Parameter(torch.empty(self.num_local, inner, hidden, dtype=DTYPE))
        self.shared_experts = (MLP(config, inner * config.n_shared_experts, reduce=False)
                               if config.n_shared_experts else None)
        set_weight_loader(self.gate_up, weights.experts_gate_up_loader())
        set_weight_loader(self.down, weights.experts_down_loader())

    def forward(self, x):
        expert_weights, experts = route(x, self.router_weight, self.top_k)
        first = self.attn_tp_group.rank_in_group_tensor * self.num_local
        out = (routed_experts(x, self.gate_up, self.down, expert_weights, experts, first, fused=self.fused)
               * self.scaling).to(x.dtype)
        if self.shared_experts is not None:
            out = out + self.shared_experts(x)
        return neuron_collectives.all_reduce(self.attn_tp_group, out)


class DecoderLayer(nn.Module):
    def __init__(self, config: DeepseekV2Config, layer_idx: int, phase: str):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = Attention(config, layer_idx)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = (MLP(config, config.intermediate_size) if layer_idx < config.first_k_dense_replace
                    else MoE(config, phase))

    def forward(self, x, attention):
        x = x + attention(self.input_layernorm(x))
        return x + self.mlp(self.post_attention_layernorm(x))


class DeepseekV2Graph(nn.Module):
    """One role graph (``phase`` "prefill" or "decode")."""

    def __init__(self, config: DeepseekV2Config, phase: str):
        super().__init__()
        if phase not in ("prefill", "decode"):
            raise ValueError(f"a role graph is a prefill or a decode, not {phase!r}")
        self.config = config
        self.phase = phase
        self.tp_group = get_attn_tp_group()
        self.lm_head_tp_group = group = groups.get_neuron_attn_tp_group()
        self.embed_tokens = VocabDimShardedEmbedding(config.vocab_size, config.hidden_size, dtype=DTYPE,
                                                     tp_group=group.device_group)
        self.layers = nn.ModuleList([DecoderLayer(config, i, phase) for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(config)
        self.lm_head = ColumnParallelLinear(config.hidden_size, config.vocab_size, dtype=DTYPE, gather_output=True,
                                            tp_group=group.device_group)
        set_weight_loader(self.embed_tokens.weight,
                          dense_weights.vocab_rows_loader(self.embed_tokens.vocab_size_per_rank, config.hidden_size))
        set_weight_loader(self.lm_head.weight,
                          dense_weights.vocab_rows_loader(self.lm_head.out_features_per_rank, config.hidden_size))

    @torch.no_grad()
    def forward(self, input_ids, positions, attn_metadata, sampling_positions, rank):
        hidden = self.embed_tokens(input_ids, scatter_tokens=False, rank=rank)
        cos, sin = self.rotary_emb(positions)
        md = attn_metadata[self.layers[0].self_attn.name]
        if self.phase == "decode":
            # every row's token slots over the graph's context (a rung of the decode context ladder), and its length
            # with the new token
            slots = token_slots(md["block_table_tensor"], md["block_size"],
                                md["block_table_tensor"].shape[1] * md["block_size"])
            seq_lens = positions + 1
            for layer in self.layers:
                hidden = layer(hidden, lambda x, a=layer.self_attn: a.decode(x, cos, sin, md, slots, seq_lens))
        else:
            # the request's length: its prior and its real tokens, the last of which the prefill samples
            length = md["cached_seq_len"].reshape(1) + sampling_positions.reshape(1) + 1
            for layer in self.layers:
                hidden = layer(hidden, lambda x, a=layer.self_attn: a.prefill(x, cos, sin, md, length))
        hidden = self.norm(hidden.index_select(0, sampling_positions))
        return self.lm_head(hidden)

    def get_kv_spec(self) -> KVSpec:
        return KVSpec(layers=[LayerSpec(name=layer.self_attn.name) for layer in self.layers])

    def bind_kv_cache(self, kv_caches: dict) -> None:
        # SGLang's MLA pool hands each layer (its compact rows, their latent view); the graphs read the rows. The paged
        # latent pool (SGLANG_NEURON_MLA_POOL=paged) hands [blocks, block, dim], block = page: the same rows, block 0
        # SGLang's reserved page (DeepseekV2ForCausalLM.bind_kv_cache checks the geometry)
        for layer in self.layers:
            layer.self_attn.cache = compact_rows(kv_caches[layer.self_attn.name][0])

    def hf_weight_mappings(self, rank: int) -> dict:
        """Parameter name to its checkpoint key, or the keys a parameter is built from, for attention-TP ``rank``,
        which holds the routed experts [rank x local, (rank + 1) x local)."""
        mappings = {"embed_tokens.weight": "model.embed_tokens.weight", "norm.weight": "model.norm.weight",
                    "lm_head.weight": ("model.embed_tokens.weight" if self.config.tie_word_embeddings
                                       else "lm_head.weight")}
        for i, layer in enumerate(self.layers):
            src, dst = f"model.layers.{i}", f"layers.{i}"
            attn = f"{src}.self_attn"
            mappings.update({
                f"{dst}.input_layernorm.weight": f"{src}.input_layernorm.weight",
                f"{dst}.post_attention_layernorm.weight": f"{src}.post_attention_layernorm.weight",
                f"{dst}.self_attn.q_weight": f"{attn}.q_proj.weight",
                f"{dst}.self_attn.kv_a_weight": f"{attn}.kv_a_proj_with_mqa.weight",
                f"{dst}.self_attn.kv_a_layernorm.weight": f"{attn}.kv_a_layernorm.weight",
                f"{dst}.self_attn.w_uk": f"{attn}.kv_b_proj.weight",
                f"{dst}.self_attn.w_uv": f"{attn}.kv_b_proj.weight",
                f"{dst}.self_attn.o_weight": f"{attn}.o_proj.weight",
            })
            mlp = layer.mlp
            if isinstance(mlp, MoE):
                local = range(rank * mlp.num_local, (rank + 1) * mlp.num_local)
                mappings[f"{dst}.mlp.router_weight"] = f"{src}.mlp.gate.weight"
                mappings[f"{dst}.mlp.gate_up"] = [f"{src}.mlp.experts.{e}.{p}_proj.weight" for e in local
                                                  for p in ("gate", "up")]
                mappings[f"{dst}.mlp.down"] = [f"{src}.mlp.experts.{e}.down_proj.weight" for e in local]
                if mlp.shared_experts is not None:
                    mappings.update(_mlp_mappings(f"{src}.mlp.shared_experts", f"{dst}.mlp.shared_experts"))
            else:
                mappings.update(_mlp_mappings(f"{src}.mlp", f"{dst}.mlp"))
        return mappings

    def load_weights(self, checkpoint_path: str, device: torch.device) -> None:
        """Read this rank's shard of every mapped parameter onto ``device``."""
        rank = self.tp_group.rank_in_group
        checkpoint = SafetensorsCheckpoint(checkpoint_path)
        state = checkpoint.load_sharded_pipelined(rank, self.tp_group.world_size, self, self.hf_weight_mappings(rank),
                                                  device).state_dict
        self.load_state_dict(state, strict=False, assign=True)


def _mlp_mappings(src: str, dst: str) -> dict:
    return {f"{dst}.gate_up_weight": [f"{src}.gate_proj.weight", f"{src}.up_proj.weight"],
            f"{dst}.down_weight": f"{src}.down_proj.weight"}
