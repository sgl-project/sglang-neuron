# SPDX-License-Identifier: Apache-2.0
"""The parts of the GPT-OSS MXFP4 graphs that prefill and decode share: parameters in their device layouts, their
checkpoint loaders, the norms, rotary embedding, the expert collectives and the KV cache contract. The role graphs
(model_prefill, model_decode) subclass these and add their forwards."""

import math

import torch
from sglang.srt.distributed.parallel_state import get_attn_tp_group
from torch import nn

import sglang_neuron.ops as NF
from sglang_neuron.layers import ColumnParallelLinear
from sglang_neuron.layers.embedding import VocabDimShardedEmbedding
from sglang_neuron.mem_cache.kv_cache import KVSpec, LayerSpec
from sglang_neuron.models.gpt_oss.config import DTYPE, GptOssConfig
from sglang_neuron.models.gpt_oss.weight_loaders_mxfp4 import (
    Q_HEIGHT,
    Q_WIDTH,
    _get_i_tiling_shard_i,
    decode_router_weight_loader,
    fused_qkv_bias_loader,
    fused_qkv_shuffling_weight_loader,
    mxfp4_down_bias_loader,
    mxfp4_down_scale_loader,
    mxfp4_down_weight_loader,
    mxfp4_gate_up_bias_loader,
    mxfp4_gate_up_blocks_loader,
    mxfp4_gate_up_scale_loader,
    natural_hidden_weight_loader,
    o_proj_weight_loader,
    shuffling_weight_loader,
    swizzle_router_weight_loader,
)
from sglang_neuron.ops.attention.swa_fused import SWA_FUSED_CORE_SHARDS
from sglang_neuron.parallel import collectives as neuron_collectives
from sglang_neuron.parallel import groups
from sglang_neuron.parallel.state import STATE
from sglang_neuron.utils.checkpoints import SafetensorsCheckpoint
from sglang_neuron.utils.weight_loader import (
    SafetensorsWeightLoader,
    ShardSpec,
    expert_parallel_tensor_dim_loader,
    set_weight_loader,
    with_rank_override,
)

# From this many tokens over the whole fleet, the decode's DP-to-EP transition is ReduceScatter + AllGather instead of
# AllReduce + slice.
DECODE_RSAG_TOKEN_THRESHOLD = 128


def _shuffled_hidden_loader(*shape):
    return shuffling_weight_loader(shuffle_dim=0, expected_shard_shape=shape, shard_spec=None)


class GptOssRMSNorm(nn.Module):
    """RMSNorm over the padded, shuffled hidden dimension: the mean is taken over the checkpoint's hidden size and the
    padding is zeroed after the scale."""

    def __init__(self, config: GptOssConfig):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(config.hidden_size, dtype=DTYPE))
        self.unpadded_hidden_size = config.unpadded_hidden_size
        self.hidden_size = config.hidden_size
        self.variance_epsilon = config.rms_norm_eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = hidden_states.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True) * (self.hidden_size / self.unpadded_hidden_size)
        out = self.weight * (x * torch.rsqrt(variance + self.variance_epsilon))
        # the shuffle puts the padding at the end of each of the four quarters
        quarters = out.view(*out.shape[:-1], 4, self.hidden_size // 4)
        quarters[..., self.unpadded_hidden_size // 4:] = 0.0
        return out.to(hidden_states.dtype)


class GptOssRotaryEmbedding(nn.Module):
    """YaRN rotary embedding; returns cos and sin of [T, head_dim // 2], scaled by YaRN's attention concentration."""

    def __init__(self, config: GptOssConfig):
        super().__init__()
        rope = config.rope_parameters
        head_dim, theta = config.head_dim, rope["rope_theta"]
        factor = rope.get("factor", 1.0)
        context = rope.get("original_max_position_embeddings", 8192)
        freq = theta ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim)
        d_half = head_dim / 2
        low = d_half * math.log(context / (rope.get("beta_fast", 32.0) * 2 * math.pi)) / math.log(theta)
        high = d_half * math.log(context / (rope.get("beta_slow", 1.0) * 2 * math.pi)) / math.log(theta)
        extrapolate = 1 - ((torch.arange(d_half, dtype=torch.float32) - low) / (high - low)).clamp(0, 1)
        inv_freq = 1.0 / (factor * freq) * (1 - extrapolate) + 1.0 / freq * extrapolate
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.concentration = 0.1 * math.log(factor) + 1.0  # a float: under SGLang's bf16 default a tensor rounds it

    def forward(self, positions: torch.Tensor, dtype: torch.dtype):
        freqs = positions[:, None].float() @ self.inv_freq[None, :].float()
        return (freqs.cos() * self.concentration).to(dtype), (freqs.sin() * self.concentration).to(dtype)


def _sinks_loader(heads_per_rank: int, rank: int):
    """This rank's heads of the per-head attention sinks, in float32."""

    def transform(slices, _rank):
        (sinks,) = slices
        return sinks[rank * heads_per_rank:(rank + 1) * heads_per_rank].to(torch.float32)

    return SafetensorsWeightLoader(transform=transform)


def _dcp_sinks_loader(heads_per_rank: int, rank: int, dcp_size: int, dcp_rank: int):
    """float32 sinks of the DCP group's gathered heads, real on DCP rank 0 only
    (decode_context_parallel.group_sinks)."""
    from sglang_neuron.distributed.decode_context_parallel import group_sinks

    def transform(slices, _rank):
        (sinks,) = slices
        return group_sinks(sinks, heads_per_rank, rank, dcp_size, dcp_rank)

    return SafetensorsWeightLoader(transform=transform)


class GptOssAttention(nn.Module):
    """Attention sharded by heads over attention TP: GQA with learned per-head sinks, a sliding window on the even
    layers and YaRN RoPE. The fused QKV and the output projection keep the shuffled hidden order."""

    def __init__(self, config: GptOssConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.unpadded_hidden_size = config.unpadded_hidden_size
        self.scaling = config.head_dim ** -0.5
        self.sliding_window = config.sliding_window if layer_idx % 2 == 0 else None

        self.tp_group = get_attn_tp_group()
        self.world_size = self.tp_group.world_size
        self.attn_tp_group = groups.get_neuron_attn_tp_group()
        rank = self.tp_group.rank_in_group
        # DCP group (None without DCP): its ranks hold this rank's KV head for the other tokens of each sequence; the
        # decode gathers the group's query heads, so this rank holds all their sinks
        dcp = groups.get_neuron_dcp_group()
        self.dcp_group = dcp if dcp is not None and dcp.world_size > 1 else None
        self.dcp_size = 1 if self.dcp_group is None else self.dcp_group.world_size

        self.num_attention_heads_per_rank = config.num_attention_heads // self.world_size
        # with more ranks than KV heads each rank holds one KV head, replicated across ranks
        self.num_key_value_heads_per_rank = max(1, config.num_key_value_heads // self.world_size)
        kv_replicas = max(1, self.world_size // config.num_key_value_heads)
        q_size = self.num_attention_heads_per_rank * self.head_dim
        kv_size = self.num_key_value_heads_per_rank * self.head_dim

        # The fused sliding-window prefill kernel splits a rank's KV heads over its two physical cores, so it needs at
        # least one KV head per core: with one KV head per rank both cores would write the same pages.
        self.fuse_swa = self.sliding_window is not None and self.num_key_value_heads_per_rank >= SWA_FUSED_CORE_SHARDS

        self.qkv_proj_weight = nn.Parameter(torch.empty(self.hidden_size, q_size + 2 * kv_size, dtype=DTYPE))
        self.qkv_proj_bias = nn.Parameter(torch.empty(q_size + 2 * kv_size, dtype=DTYPE))
        self.o_proj_weight = nn.Parameter(torch.empty(q_size, self.hidden_size, dtype=DTYPE))
        self.o_proj_bias = nn.Parameter(torch.empty(self.hidden_size, dtype=DTYPE))
        self.sinks = nn.Parameter(torch.empty(self.num_attention_heads_per_rank * self.dcp_size, dtype=torch.float32))

        # bound by bind_kv_cache: the FP8 V cache and the packed FP8 K cache
        self.k_cache = None
        self.v_cache = None
        # FP8 quantization scales (load_weights): buffers, so they move with the module, and their host floats, which
        # the kernels take as constants
        self.register_buffer("k_scale", None, persistent=False)
        self.register_buffer("v_scale", None, persistent=False)
        self.k_scale_float = 1.0
        self.v_scale_float = 1.0

        set_weight_loader(self.qkv_proj_weight, with_rank_override(fused_qkv_shuffling_weight_loader(
            q_size, kv_size, (self.hidden_size, q_size + 2 * kv_size), kv_replicas), rank=rank))
        set_weight_loader(self.qkv_proj_bias, with_rank_override(
            fused_qkv_bias_loader(q_size, kv_size, kv_replicas), rank=rank))
        set_weight_loader(self.o_proj_weight, with_rank_override(
            o_proj_weight_loader(q_size, (q_size, self.hidden_size), self.world_size), rank=rank))
        set_weight_loader(self.o_proj_bias, with_rank_override(
            shuffling_weight_loader(0, (self.hidden_size,), None, self.world_size), rank=rank))
        set_weight_loader(self.sinks, _sinks_loader(self.num_attention_heads_per_rank, rank) if self.dcp_group is None
                          else _dcp_sinks_loader(self.num_attention_heads_per_rank, rank, self.dcp_size,
                                                 self.dcp_group.rank_in_group))


class GptOssExperts(nn.Module):
    """The MoE of one layer with MXFP4 expert weights (uint16 blocks, uint8 scales) that the NKI kernels read as they
    are.

    Without EP every rank holds every expert and shards the expert intermediate dimension over attention TP. With EP
    a rank holds ``num_local_experts / ep_degree`` experts and shards their intermediate dimension over its EP-TP row
    (world / ep_degree ranks). Under DP attention with EP the experts span the DP replicas, so the tokens of every
    replica are gathered before the experts and the results reduced after them (cross_dp_ep).
    """

    # tokens per block of the prefill MoE kernel
    block_size = 256

    def __init__(self, config: GptOssConfig):
        super().__init__()
        self.tp_group = get_attn_tp_group()
        self.dp_size = STATE.dp_size
        self.ep_degree = STATE.ep_degree
        # ep_tp_group: the ranks that split one expert's intermediate dimension, which the blockwise token mapping
        # coordinates over; moe_group: the group the expert outputs are reduced over
        if self.ep_degree > 1:
            self.ep_rank = groups.get_neuron_ep_rank()
            # held for its rank tensor: the forward reads no group lookup, which dynamo cannot trace
            self.ep_group = groups.get_neuron_ep_group()
            self.ep_tp_group = groups.get_neuron_ep_tp_group()
            self.tp_degree = self.ep_tp_group.world_size
            self.expert_tp_rank = self.ep_tp_group.rank_in_group
            self.moe_group = self.tp_group
        else:
            self.ep_rank = 0
            self.ep_tp_group = self.tp_group
            self.moe_group = groups.get_neuron_attn_tp_group()
            self.tp_degree = self.moe_group.world_size
            self.expert_tp_rank = self.moe_group.rank_in_group
        self.cross_dp_ep = self.dp_size > 1 and self.ep_degree > 1
        if self.cross_dp_ep:
            self.dp_group = groups.get_dp_group()
            self.wide_ep_group = groups.get_wide_ep_group()
            self.attn_tp_group = groups.get_neuron_attn_tp_group()

        self.total_num_experts = config.num_local_experts
        self.num_local_experts = config.num_local_experts // self.ep_degree
        self.num_experts_per_token = config.num_experts_per_tok
        self.hidden_size = config.hidden_size
        self.unpadded_hidden_size = config.unpadded_hidden_size
        self.rms_norm_eps = config.rms_norm_eps
        self.intermediate_size_per_rank = config.intermediate_size // self.tp_degree
        self.limit = config.swiglu_limit

        self.post_attention_layernorm = GptOssRMSNorm(config)
        self._init_expert_weights()

    def _init_expert_weights(self):
        """The router and expert parameters with their loaders: MXFP4 tiles, as the kernels read them."""
        E, H, I = self.num_local_experts, self.hidden_size, self.intermediate_size_per_rank
        # The router is replicated. Decode reads it as [H, E] in the shuffled hidden order. Prefill holds it as [E, H]
        # in the swizzled order (the fused norm and router kernel reads its transpose) and its gamma in natural order:
        # its own copies, rather than permuting every forward.
        self.router_weight = nn.Parameter(torch.empty(H, self.total_num_experts, dtype=DTYPE))
        self.router_bias = nn.Parameter(torch.zeros(self.total_num_experts, dtype=DTYPE))
        self.router_weight_prefill = nn.Parameter(torch.empty(self.total_num_experts, H, dtype=DTYPE))
        self.gamma_prefill = nn.Parameter(torch.empty(H, dtype=DTYPE))
        self.gate_up_proj_weight = nn.Parameter(torch.empty(E, 128, 2, H // 512, I, dtype=torch.uint16),
                                                requires_grad=False)
        self.gate_up_proj_scale = nn.Parameter(torch.empty(E, 16, 2, H // 512, I, dtype=torch.uint8),
                                               requires_grad=False)
        self.gate_up_proj_bias = nn.Parameter(torch.zeros(E, I * 2, dtype=DTYPE))
        i_tiles, q_blocks_per_tile = _get_i_tiling_shard_i(I)
        self.down_proj_weight = nn.Parameter(torch.empty(E, q_blocks_per_tile * Q_HEIGHT, i_tiles, H,
                                                         dtype=torch.uint16), requires_grad=False)
        self.down_proj_scale = nn.Parameter(torch.empty(E, q_blocks_per_tile, i_tiles, H, dtype=torch.uint8),
                                            requires_grad=False)
        self.down_proj_bias = nn.Parameter(torch.zeros(E, H, dtype=DTYPE))
        self._set_weight_loaders()

    def _set_weight_loaders(self):
        E, H, I = self.num_local_experts, self.hidden_size, self.intermediate_size_per_rank
        local_experts = list(range(self.ep_rank * E, (self.ep_rank + 1) * E))
        set_weight_loader(self.router_weight, decode_router_weight_loader(self.total_num_experts, H))
        set_weight_loader(self.post_attention_layernorm.weight, _shuffled_hidden_loader(H))
        set_weight_loader(self.router_weight_prefill,
                          swizzle_router_weight_loader(expected_shard_shape=(self.total_num_experts, H)))
        set_weight_loader(self.gamma_prefill, natural_hidden_weight_loader(expected_shard_shape=(H,)))

        # the intermediate dimension is sharded by the expert TP rank: under DP attention an EP-TP row spans replicas,
        # so it is not the attention TP rank the checkpoint loader shards by
        rank = self.expert_tp_rank
        gate_up = ShardSpec(dim=1, size=I * 2, num_shards=self.tp_degree)
        down = ShardSpec(dim=2, size=I // 32, num_shards=self.tp_degree)
        mx_groups = H // 32
        loaders = {
            self.gate_up_proj_weight: mxfp4_gate_up_blocks_loader(gate_up, (E, I * 2, mx_groups, 16)),
            self.gate_up_proj_scale: mxfp4_gate_up_scale_loader(gate_up, (E, I * 2, mx_groups)),
            # GPT-OSS's SwiGLU multiplies by up + 1; the 1 is folded into the up bias
            self.gate_up_proj_bias: mxfp4_gate_up_bias_loader(gate_up, (E, I * 2), hidden_act_bias=1.0),
            self.down_proj_weight: mxfp4_down_weight_loader(down, (E, H, I // 32, 16)),
            self.down_proj_scale: mxfp4_down_scale_loader(down, (E, H, I // 32)),
            # summed over the expert TP ranks, so each holds its share of the bias
            self.down_proj_bias: mxfp4_down_bias_loader((E, H), scale=self.tp_degree),
        }
        for param, loader in loaders.items():
            # each read narrowed to this rank's experts first
            set_weight_loader(param, with_rank_override(expert_parallel_tensor_dim_loader(local_experts, loader),
                                                        rank=rank))

    def hf_weight_mappings(self, prefix: str, key: str) -> dict:
        """Router and expert parameter names under ``prefix`` to their checkpoint keys under the layer's ``key``."""
        experts, key_experts = f"{prefix}.mlp.experts", f"{key}.mlp.experts"
        return {
            f"{experts}.post_attention_layernorm.weight": f"{key}.post_attention_layernorm.weight",
            f"{experts}.gamma_prefill": f"{key}.post_attention_layernorm.weight",
            f"{experts}.router_weight": f"{key}.mlp.router.weight",
            f"{experts}.router_weight_prefill": f"{key}.mlp.router.weight",
            f"{experts}.router_bias": f"{key}.mlp.router.bias",
            f"{experts}.gate_up_proj_weight": f"{key_experts}.gate_up_proj_blocks",
            f"{experts}.gate_up_proj_scale": f"{key_experts}.gate_up_proj_scales",
            f"{experts}.gate_up_proj_bias": f"{key_experts}.gate_up_proj_bias",
            f"{experts}.down_proj_weight": f"{key_experts}.down_proj_blocks",
            f"{experts}.down_proj_scale": f"{key_experts}.down_proj_scales",
            f"{experts}.down_proj_bias": f"{key_experts}.down_proj_bias",
        }

    def _ep_rank_id(self) -> torch.Tensor:
        """The EP rank as a [1, 1] int32 graph input, read from the EP group's device rank tensor so that every rank
        traces the same graph; 0 without EP."""
        if self.ep_degree == 1:
            return torch.zeros((1, 1), dtype=torch.int32, device=self.gate_up_proj_weight.device)
        return self.ep_group.rank_in_group_tensor.reshape(1, 1).to(torch.int32)

    def _gate_up_bias_tiles(self) -> torch.Tensor:
        """The gate/up bias in the kernels' [E, I_block / Q_WIDTH, 2, I_blocks, Q_WIDTH] tiling."""
        i_blocks = math.ceil(self.intermediate_size_per_rank / 512)
        return self.gate_up_proj_bias.view(self.num_local_experts,
                                           self.intermediate_size_per_rank // i_blocks // Q_WIDTH,
                                           2, i_blocks, Q_WIDTH)

    def _clamps(self) -> dict:
        """The SwiGLU clamps as the kernels take them; the +1 folded into the up bias shifts the up clamp."""
        return dict(gate_clamp_upper_limit=self.limit, up_clamp_upper_limit=self.limit + 1,
                    up_clamp_lower_limit=-self.limit + 1)

    def _decode_expert_weights(self) -> dict:
        """The expert tensors for moe_block_tkg: MXFP4 tiles with their scales."""
        return dict(expert_gate_up_weights=self.gate_up_proj_weight,
                    expert_gate_up_weights_scale=self.gate_up_proj_scale,
                    expert_gate_up_bias=self._gate_up_bias_tiles(),
                    expert_down_weights=self.down_proj_weight,
                    expert_down_weights_scale=self.down_proj_scale,
                    expert_down_bias=self.down_proj_bias)

    def _run_moe_block_tkg(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """The decode MoE (post-attention norm, router, experts) in one kernel. While the batch's tokens times top-k
        stay under the rank's expert count the kernel loads only the routed experts; otherwise, and always with EP, it
        runs every local expert, which needs the EP rank."""
        all_experts = (hidden_states.shape[0] * self.num_experts_per_token >= self.num_local_experts
                       or self.ep_degree > 1)
        return NF.moe_block_tkg(
            hidden_states.unsqueeze(0),
            gamma=self.post_attention_layernorm.weight.unsqueeze(0).to(torch.float32),
            router_weights=self.router_weight,
            router_bias=self.router_bias.unsqueeze(0),
            eps=self.rms_norm_eps,
            top_k=self.num_experts_per_token,
            hidden_actual=self.unpadded_hidden_size,
            is_all_expert=all_experts,
            rank_id=self._ep_rank_id() if all_experts else None,
            **self._decode_expert_weights(),
            **self._clamps(),
        )

    def _wide_ep_all_reduce(self, output: torch.Tensor) -> torch.Tensor:
        """Sum ``output`` over the wide-EP world: two short rings (attention TP, then the DP column) where those groups
        tile the world, else one ring through every rank."""
        tp, dp = self.attn_tp_group.world_size, self.dp_group.world_size
        if tp > 1 and dp > 1 and tp * dp == self.wide_ep_group.world_size:
            output = neuron_collectives.all_reduce(self.attn_tp_group, output)
            return neuron_collectives.all_reduce(self.dp_group, output)
        return neuron_collectives.all_reduce(self.wide_ep_group, output)


class GptOssMLP(nn.Module):
    def __init__(self, config: GptOssConfig):
        super().__init__()
        self.experts = self.experts_type(config)


class GptOssDecoderLayer(nn.Module):
    def __init__(self, config: GptOssConfig, layer_idx: int):
        super().__init__()
        self.input_layernorm = GptOssRMSNorm(config)
        self.self_attn = self.attention_type(config, layer_idx=layer_idx)
        self.mlp = self.mlp_type(config)
        self.tp_group = get_attn_tp_group()
        self.world_size = self.tp_group.world_size
        set_weight_loader(self.input_layernorm.weight, _shuffled_hidden_loader(config.hidden_size))


class GptOssModel(nn.Module):
    """The decoder stack, or one pipeline stage of it (``stage``, a distributed.pipeline.StageSpec): global layers
    [start_layer, end_layer) keyed by global id, the embedding on the first stage and the final norm on the last."""

    def __init__(self, config: GptOssConfig, stage=None):
        super().__init__()
        self.config = config
        self.start_layer = 0 if stage is None else stage.start_layer
        self.end_layer = config.num_hidden_layers if stage is None else stage.end_layer
        self.is_first_stage = stage is None or stage.is_first
        self.is_last_stage = stage is None or stage.is_last
        self.embed_tokens = None
        if self.is_first_stage:
            embedding_group = groups.get_neuron_attn_tp_group()
            self.embed_tokens = VocabDimShardedEmbedding(vocab_size=config.vocab_size, embed_dim=config.hidden_size,
                                                         dtype=DTYPE, tp_group=embedding_group.device_group)
            rows = self.embed_tokens.vocab_size_per_rank
            set_weight_loader(self.embed_tokens.weight, with_rank_override(shuffling_weight_loader(
                shuffle_dim=1, expected_shard_shape=(rows, config.hidden_size),
                shard_spec=ShardSpec(dim=0, size=rows, num_shards=self.embed_tokens.tp_size),
            ), rank=embedding_group.rank_in_group))
        self.layers = nn.ModuleList([self.layer_type(config, i) for i in range(self.start_layer, self.end_layer)])
        self.norm = GptOssRMSNorm(config) if self.is_last_stage else None
        if self.norm is not None:
            set_weight_loader(self.norm.weight, _shuffled_hidden_loader(config.hidden_size))
        self.rotary_emb = GptOssRotaryEmbedding(config)


class GptOssForCausalLM(nn.Module):
    """A role graph: the model and the vocabulary-sharded LM head, with the checkpoint mappings and the KV cache
    contract."""

    def __init__(self, config: GptOssConfig, stage=None):
        super().__init__()
        self.config = config
        self.model = self.model_type(config) if stage is None else self.model_type(config, stage)
        self.is_first_stage, self.is_last_stage = self.model.is_first_stage, self.model.is_last_stage
        self.tp_group = get_attn_tp_group()
        self.world_size = self.tp_group.world_size
        self.lm_head_tp_group = groups.get_neuron_attn_tp_group()
        # only the last stage has an LM head; earlier stages return their boundary
        self.lm_head = None
        if self.is_last_stage:
            self.lm_head = ColumnParallelLinear(config.hidden_size, config.vocab_size, dtype=DTYPE,
                                                gather_output=True, tp_group=self.lm_head_tp_group.device_group)
            rows = self.lm_head.out_features_per_rank
            set_weight_loader(self.lm_head.weight, with_rank_override(shuffling_weight_loader(
                shuffle_dim=1, expected_shard_shape=(rows, config.hidden_size),
                shard_spec=ShardSpec(dim=0, size=rows, num_shards=self.lm_head.tp_size),
            ), rank=self.lm_head_tp_group.rank_in_group))

    @classmethod
    def from_hf_config(cls, hf_config, intermediate_size: int, stage=None):
        config = GptOssConfig.from_hf_config(hf_config, intermediate_size)
        return cls(config) if stage is None else cls(config, stage)

    def get_kv_spec(self) -> KVSpec:
        return KVSpec(layers=[LayerSpec(name=f"layers.{layer.self_attn.layer_idx}.self_attn")
                              for layer in self.model.layers])

    def bind_kv_cache(self, kv_caches: dict) -> None:
        for layer in self.model.layers:
            attn = layer.self_attn
            k_cache, v_cache = kv_caches[f"layers.{attn.layer_idx}.self_attn"]
            attn.k_cache, attn.v_cache = k_cache, v_cache

    def hf_weight_mappings(self) -> dict:
        """Parameter name to the checkpoint key (or the keys a fused parameter is built from). A stage names parameters
        by local layer index and maps them to the global layer's keys."""
        mappings = {}
        for i, layer in enumerate(self.model.layers):
            p, key = f"model.layers.{i}", f"model.layers.{layer.self_attn.layer_idx}"
            attn, key_attn = f"{p}.self_attn", f"{key}.self_attn"
            mappings.update({
                f"{attn}.qkv_proj_weight": [f"{key_attn}.{n}_proj.weight" for n in "qkv"],
                f"{attn}.qkv_proj_bias": [f"{key_attn}.{n}_proj.bias" for n in "qkv"],
                f"{attn}.o_proj_weight": f"{key_attn}.o_proj.weight",
                f"{attn}.o_proj_bias": f"{key_attn}.o_proj.bias",
                f"{attn}.sinks": f"{key_attn}.sinks",
                f"{p}.input_layernorm.weight": f"{key}.input_layernorm.weight",
                **layer.mlp.experts.hf_weight_mappings(p, key),
            })
        return mappings

    def load_weights(self, checkpoint_path: str, device: torch.device) -> None:
        """Read this rank's shard of every mapped parameter onto ``device`` and the FP8 KV scales."""
        checkpoint = SafetensorsCheckpoint(checkpoint_path)
        state = checkpoint.load_sharded_pipelined(
            self.tp_group.rank_in_group, self.world_size, self, self.hf_weight_mappings(), device).state_dict
        self._load_kv_cache_scales(checkpoint, device)
        self.load_state_dict(state, strict=False, assign=True)

    def _load_kv_cache_scales(self, checkpoint: SafetensorsCheckpoint, device: torch.device) -> None:
        """The checkpoint's k_scale/v_scale (1 where it has none). It stores the dequantization scale and the kernels
        quantize by multiplying, so the scales are inverted here."""
        for layer in self.model.layers:
            attn = layer.self_attn
            for name in ("k_scale", "v_scale"):
                key = f"model.layers.{attn.layer_idx}.self_attn.{name}"
                scale = (1.0 / checkpoint._get_slice(key)[:].to(dtype=torch.bfloat16, device=device)
                         if key in checkpoint._tensor_name_to_file
                         else torch.ones(1, dtype=torch.bfloat16, device=device))
                setattr(attn, name, scale.reshape(1, 1))
            attn.k_scale_float = attn.k_scale.item()
            attn.v_scale_float = attn.v_scale.item()
