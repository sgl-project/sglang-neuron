# SPDX-License-Identifier: Apache-2.0
"""MiniMax-H3's Qwen3-VL conditioning encoder on Neuron, TP-sharded on the ``vn`` NF ops: the
counterpart of diffusers' ``get_qwen3vl_prompt_embeds``, with the vision tower for keyframes."""

import logging
import os
from collections import namedtuple

import torch
import torch.distributed as dist
import torch.nn as nn
from nkilib.core.utils.common_types import NormType
from transformers import Qwen3VLConfig
from transformers.activations import ACT2FN
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLModel,
    Qwen3VLTextRMSNorm,
    Qwen3VLTextRotaryEmbedding,
    Qwen3VLVisionPatchEmbed,
    apply_rotary_pos_emb_vision,
)

import sglang_neuron.diffusion.vn.functional as NF
from sglang_neuron.diffusion.distributed.tp_loaders import cast_weight_loader
from sglang_neuron.diffusion.vn.functional.attention.attention_cte import flash_attention
from sglang_neuron.diffusion.vn.functional.attention.o_proj import o_proj
from sglang_neuron.diffusion.vn.functional.merge_multimodal_embeddings import (
    merge_multimodal_embeddings,
)
from sglang_neuron.diffusion.vn.functional.mlp import mlp
from sglang_neuron.diffusion.vn.functional.rmsnorm_residual_prefill import rmsnorm_residual_prefill
from sglang_neuron.diffusion.vn.model.qwen3_vl.utils.vision_preprocessing import (
    compute_position_indices_and_weights,
    compute_rotary_pos_emb,
)
from sglang_neuron.diffusion.vn.model.qwen3_vl.weight_loaders import (
    vis_qkv_bias_loader,
    vis_qkv_weight_loader,
)
from sglang_neuron.diffusion.vn.utils.shard_loaders import sharding_weight_loader
from sglang_neuron.layers.embedding import VocabDimShardedEmbedding
from sglang_neuron.layers.linear import ColumnParallelLinear, RowParallelLinear
from sglang_neuron.utils.checkpoints import SafetensorsCheckpoint
from sglang_neuron.utils.weight_loader import (
    SafetensorsWeightLoader,
    get_shard,
    get_weight_loader,
    set_weight_loader,
)

logger = logging.getLogger(__name__)

# H3 conditions on hidden_states[50]: the unnormalized output of decoder block 49.
MINIMAX_H3_TEXT_ENCODER_LAYER = 50

# Padding step of the presentation, not a length limit; each multiple is one compiled graph.
# Causal attention needs no padding mask: real tokens never attend the later pad tokens.
MINIMAX_H3_TEXT_SEQ_BUCKET = 512

_TextEncoderOutput = namedtuple("_TextEncoderOutput", ["last_hidden_state"])


def _tp_size_rank(tp_group) -> tuple[int, int]:
    if not dist.is_initialized():
        return 1, 0
    grp = tp_group if tp_group is not None else dist.group.WORLD
    return dist.get_world_size(grp), dist.get_rank(grp)


def _gqa_qkv_weight_loader(
    q_size: int, kv_size: int, num_shards: int, num_kv_replicas: int
) -> SafetensorsWeightLoader:
    """A rank's fused ``[hidden, Q | K | V]`` weight from the checkpoint's ``[out, in]`` Q, K and
    V: Q shard ``rank``, K / V shard ``rank // num_kv_replicas``, so ``num_kv_replicas``
    consecutive ranks share a KV head."""

    def transform(slices, rank: int) -> torch.Tensor:
        assert len(slices) == 3, "_gqa_qkv_weight_loader expects [Q, K, V] slices in order"
        q, k, v = slices
        kv_rank, kv_shards = rank // num_kv_replicas, num_shards // num_kv_replicas
        parts = [
            get_shard(q, 0, q_size, num_shards, rank),
            get_shard(k, 0, kv_size, kv_shards, kv_rank),
            get_shard(v, 0, kv_size, kv_shards, kv_rank),
        ]
        return torch.cat([part.T for part in parts], dim=1)

    return SafetensorsWeightLoader(transform=transform)


def _column_parallel(in_features, out_features, bias=False, dtype=torch.bfloat16, tp_group=None):
    """``ColumnParallelLinear`` with the ``[in, out / tp]`` weight the NF kernels take, on the
    meta device (``load_checkpoint`` assigns its parameters), loading this rank's shard."""
    with torch.device("meta"):
        layer = ColumnParallelLinear(
            in_features,
            out_features,
            dtype=dtype,
            gather_output=False,
            tp_group=tp_group,
            bias=bias,
            weight_layout="in_out",
        )
    shards = dict(shard_size=layer.out_features_per_rank, num_shards=layer.tp_size)
    set_weight_loader(
        layer.weight, sharding_weight_loader(shard_dim=1, is_storage_transposed=True, **shards)
    )
    if bias:
        set_weight_loader(layer.bias, sharding_weight_loader(shard_dim=0, **shards))
    return layer


def _row_parallel(in_features, out_features, bias=False, dtype=torch.bfloat16, tp_group=None):
    """``RowParallelLinear`` with the ``[in / tp, out]`` weight the NF kernels take, on the meta
    device (``load_checkpoint`` assigns its parameters), loading this rank's shard."""
    with torch.device("meta"):
        layer = RowParallelLinear(
            in_features,
            out_features,
            dtype=dtype,
            tp_group=tp_group,
            bias=bias,
            weight_layout="in_out",
        )
    set_weight_loader(
        layer.weight,
        sharding_weight_loader(
            shard_dim=0,
            shard_size=layer.in_features_per_rank,
            num_shards=layer.tp_size,
            is_storage_transposed=True,
        ),
    )
    return layer


def checkpoint_mappings(module: nn.Module, prefix: str = "") -> dict:
    """The HF checkpoint key(s) each of ``module``'s parameters loads from: ``prefix`` + its
    name, except the text attention's ``qkv_proj_weight``, which fuses HF's Q, K and V."""
    mappings = {}
    for name, _ in module.named_parameters():
        key = prefix + name
        if name.endswith("qkv_proj_weight"):
            key = [key.replace("qkv_proj_weight", f"{p}_proj.weight") for p in "qkv"]
        mappings[name] = key
    return mappings


def load_checkpoint(module: nn.Module, checkpoint_dir: str, prefix: str = "", tp_group=None):
    """Load ``module`` from the HF safetensors in ``checkpoint_dir``, sharded for this rank and
    cast to each parameter's dtype."""
    tp_size, tp_rank = _tp_size_rank(tp_group)
    logged_casts: set = set()
    for name, param in module.named_parameters():
        set_weight_loader(
            param,
            cast_weight_loader(get_weight_loader(param), param.dtype, name, logged_casts),
        )
    state_dict = (
        SafetensorsCheckpoint(checkpoint_dir)
        .load_sharded_pipelined(
            tp_rank,
            tp_size,
            module,
            checkpoint_mappings(module, prefix),
            torch.device("cpu"),
        )
        .state_dict
    )
    module.load_state_dict(state_dict, strict=True, assign=True)
    return len(state_dict)


def vision_inputs(pixel_values: torch.Tensor, image_grid_thw: torch.Tensor, vision_config):
    """Host-side inputs of ``NeuronQwen3VLVisionModel`` for one keyframe.

    One image is one attention segment, so the tower runs at its exact patch count; the keyframe
    is fit to the video's canvas, which already fixes every other graph's shape. The taps and
    rotary come from the ``vn`` preprocessing, which tracks no transformers version.

    Returns ``(pixel_values, pos_indices, pos_weights, cos, sin)``: the 4 bilinear taps of the
    position table and their weights, ``[4, patches]``, and the 2-D rotary, ``[patches, head_dim]``.
    """
    # One image per call: two in one unbounded segment would attend each other.
    if image_grid_thw.shape[0] != 1:
        raise NotImplementedError(f"one keyframe per call, got {image_grid_thw.shape[0]}")
    num_patches = int(image_grid_thw.prod())
    if num_patches != pixel_values.shape[0]:
        raise ValueError(
            f"image_grid_thw covers {num_patches} patches, pixel_values has {pixel_values.shape[0]}"
        )
    merge = vision_config.spatial_merge_size

    pos_indices, pos_weights = compute_position_indices_and_weights(
        image_grid_thw, int(vision_config.num_position_embeddings**0.5), merge
    )
    cos, sin = compute_rotary_pos_emb(
        image_grid_thw, vision_config.hidden_size // vision_config.num_heads, merge
    )
    return pixel_values, pos_indices, pos_weights, cos, sin


class NeuronQwen3VLVisionAttention(nn.Module):
    """Bidirectional attention over the image; TP-sharded heads, all-reduce after the
    projection."""

    def __init__(self, hidden_size, num_heads, dtype=torch.bfloat16, tp_group=None):
        super().__init__()
        tp_size, _ = _tp_size_rank(tp_group)
        if num_heads % tp_size:
            raise ValueError(f"num_heads ({num_heads}) must be divisible by tp_size ({tp_size})")
        self.tp_group = tp_group
        self.tp_size = tp_size
        self.num_heads = num_heads // tp_size
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = _column_parallel(hidden_size, 3 * hidden_size, True, dtype, tp_group)
        # HF fuses Q, K and V along the outputs, so each rank takes its heads of each.
        set_weight_loader(
            self.qkv.weight, vis_qkv_weight_loader(self.num_heads, self.head_dim, hidden_size)
        )
        set_weight_loader(
            self.qkv.bias, vis_qkv_bias_loader(self.num_heads, self.head_dim, hidden_size)
        )
        self.proj = _row_parallel(hidden_size, hidden_size, True, dtype, tp_group)

    def forward(self, hidden_states, cos, sin):
        S = hidden_states.shape[0]
        qkv = NF.qkv_proj(
            hidden=hidden_states.unsqueeze(0),
            qkv_weights=self.qkv.weight,
            bias=self.qkv.bias.unsqueeze(0),
        ).squeeze(0)
        q, k, v = qkv.view(S, 3, self.num_heads, self.head_dim).unbind(1)
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)
        q, k, v = (x.transpose(0, 1) for x in (q, k, v))
        attn_output = flash_attention(
            q,
            k,
            v,
            scale=self.scale,
            causal_mask=False,
            tp_q=True,
            tp_k=True,
        )
        # o_proj takes the heads d-major: [1, N, D, S].
        output = o_proj(attn_output.transpose(1, 2).unsqueeze(0), self.proj.weight).squeeze(0)
        if self.tp_size > 1:
            dist.all_reduce(output, group=self.tp_group)
        return output + self.proj.bias


class NeuronQwen3VLVisionMLP(nn.Module):
    """``linear_fc1 -> act -> linear_fc2`` with the inner dim sharded, all-reduce after fc2."""

    def __init__(
        self,
        in_features,
        hidden_features,
        out_features,
        hidden_act,
        dtype=torch.bfloat16,
        tp_group=None,
    ):
        super().__init__()
        tp_size, _ = _tp_size_rank(tp_group)
        if hidden_features % tp_size:
            raise ValueError(
                f"hidden_features ({hidden_features}) must be divisible by tp_size ({tp_size})"
            )
        self.act_fn = ACT2FN[hidden_act]
        self.linear_fc1 = _column_parallel(in_features, hidden_features, True, dtype, tp_group)
        self.linear_fc2 = _row_parallel(hidden_features, out_features, True, dtype, tp_group)

    def forward(self, x):
        return self.linear_fc2(self.act_fn(self.linear_fc1(x)))


class NeuronQwen3VLVisionBlock(nn.Module):
    def __init__(self, config, dtype=torch.bfloat16, tp_group=None):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6, dtype=dtype)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6, dtype=dtype)
        self.attn = NeuronQwen3VLVisionAttention(
            config.hidden_size, config.num_heads, dtype=dtype, tp_group=tp_group
        )
        self.mlp = NeuronQwen3VLVisionMLP(
            config.hidden_size,
            config.intermediate_size,
            config.hidden_size,
            hidden_act=config.hidden_act,
            dtype=dtype,
            tp_group=tp_group,
        )

    def forward(self, hidden_states, cos, sin):
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states), cos, sin)
        return hidden_states + self.mlp(self.norm2(hidden_states))


class NeuronQwen3VLVisionPatchMerger(NeuronQwen3VLVisionMLP):
    """Concatenate each 2x2 block of patches and project it to the text width (exact GELU).

    The final merger normalizes each patch before the concat; the deepstack mergers
    (``use_postshuffle_norm``) normalize the concatenated block.
    """

    def __init__(self, config, use_postshuffle_norm, dtype=torch.bfloat16, tp_group=None):
        merged_size = config.hidden_size * config.spatial_merge_size**2
        super().__init__(
            merged_size,
            merged_size,
            config.out_hidden_size,
            hidden_act="gelu",
            dtype=dtype,
            tp_group=tp_group,
        )
        self.merged_size = merged_size
        self.use_postshuffle_norm = use_postshuffle_norm
        norm_size = merged_size if use_postshuffle_norm else config.hidden_size
        self.norm = nn.LayerNorm(norm_size, eps=1e-6, dtype=dtype)

    def forward(self, x):
        if self.use_postshuffle_norm:
            x = x.view(-1, self.merged_size)
        return super().forward(self.norm(x).view(-1, self.merged_size))


class NeuronQwen3VLVisionModel(nn.Module):
    """Qwen3-VL vision tower over one keyframe's patches (see ``vision_inputs``).

    Returns ``[1, num_patches / merge**2, out_hidden * (1 + L)]``: per merged row the merged
    feature, then the L deepstack features -- the layout ``merge_multimodal_embeddings`` reads.
    """

    def __init__(self, config, dtype=torch.bfloat16, tp_group=None):
        super().__init__()
        self.config = config
        self.deepstack_visual_indexes = list(config.deepstack_visual_indexes)
        self.patch_embed = Qwen3VLVisionPatchEmbed(config).to(dtype)
        self.pos_embed = nn.Embedding(
            config.num_position_embeddings, config.hidden_size, dtype=dtype
        )
        self.blocks = nn.ModuleList(
            [NeuronQwen3VLVisionBlock(config, dtype, tp_group) for _ in range(config.depth)]
        )
        self.merger = NeuronQwen3VLVisionPatchMerger(config, False, dtype, tp_group)
        self.deepstack_merger_list = nn.ModuleList(
            [
                NeuronQwen3VLVisionPatchMerger(config, True, dtype, tp_group)
                for _ in self.deepstack_visual_indexes
            ]
        )

    def forward(self, pixel_values, pos_indices, pos_weights, cos, sin):
        hidden_states = self.patch_embed(pixel_values)
        # The taps arrive int32; the lookup takes int64.
        pos_embeds = (self.pos_embed(pos_indices.long()) * pos_weights[..., None]).sum(0)
        hidden_states = hidden_states + pos_embeds.to(hidden_states.dtype)

        features = []
        for layer_idx, block in enumerate(self.blocks):
            hidden_states = block(hidden_states, cos, sin)
            if layer_idx in self.deepstack_visual_indexes:
                merger = self.deepstack_merger_list[self.deepstack_visual_indexes.index(layer_idx)]
                features.append(merger(hidden_states))
        return torch.cat([self.merger(hidden_states)] + features, dim=-1).unsqueeze(0)


class NeuronQwen3VLAttention(nn.Module):
    """Qwen3-VL GQA self-attention: one QKV projection that fuses the per-head QK-RMSNorm and
    mRoPE, causal flash attention, the O projection and an all-reduce. Takes the normalized
    ``[T, hidden]`` and returns ``[T, hidden]``."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        eps: float,
        dtype: torch.dtype = torch.bfloat16,
        tp_group=None,
    ):
        super().__init__()
        tp_size, _ = _tp_size_rank(tp_group)
        if num_heads % tp_size:
            raise ValueError(f"num_heads ({num_heads}) must be divisible by tp_size ({tp_size})")
        # Past one KV head per rank, ranks share heads.
        if tp_size >= num_kv_heads:
            if tp_size % num_kv_heads:
                raise ValueError(
                    f"tp_size ({tp_size}) must be a multiple of num_kv_heads ({num_kv_heads})"
                )
            num_kv_heads_per_rank, num_kv_replicas = 1, tp_size // num_kv_heads
        else:
            if num_kv_heads % tp_size:
                raise ValueError(
                    f"num_kv_heads ({num_kv_heads}) must be divisible by tp_size ({tp_size})"
                )
            num_kv_heads_per_rank, num_kv_replicas = num_kv_heads // tp_size, 1

        self.tp_group = tp_group
        self.tp_size = tp_size
        self.head_dim = head_dim
        self.eps = eps
        self.scale = head_dim**-0.5
        self.num_heads = num_heads // tp_size
        self.num_kv_heads = num_kv_heads_per_rank

        q_size = self.num_heads * head_dim
        kv_size = self.num_kv_heads * head_dim
        self.qkv_split = [q_size, q_size + kv_size]
        self.qkv_proj_weight = nn.Parameter(
            torch.empty(hidden_size, q_size + 2 * kv_size, dtype=dtype)
        )
        set_weight_loader(
            self.qkv_proj_weight,
            _gqa_qkv_weight_loader(q_size, kv_size, tp_size, num_kv_replicas),
        )
        self.o_proj = _row_parallel(
            num_heads * head_dim, hidden_size, dtype=dtype, tp_group=tp_group
        )
        # Over head_dim, so replicated rather than sharded.
        self.q_norm = Qwen3VLTextRMSNorm(head_dim, eps=eps).to(dtype)
        self.k_norm = Qwen3VLTextRMSNorm(head_dim, eps=eps).to(dtype)

    def forward(self, hidden_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        """``hidden_states`` ``[T, hidden]``; ``cos`` / ``sin`` ``[T, head_dim]``."""
        T = hidden_states.shape[0]
        qkv = NF.qkv_proj(
            hidden=hidden_states.unsqueeze(0),
            qkv_weights=self.qkv_proj_weight,
            cos_cache=cos.unsqueeze(0),
            sin_cache=sin.unsqueeze(0),
            num_q_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            d_head=self.head_dim,
            qk_norm_pre_rope_q_norm=NormType.RMS_NORM,
            qk_norm_pre_rope_k_norm=NormType.RMS_NORM,
            qk_norm_pre_rope_eps=self.eps,
            qk_norm_pre_rope_q_gamma=self.q_norm.weight.unsqueeze(0),
            qk_norm_pre_rope_k_gamma=self.k_norm.weight.unsqueeze(0),
        ).squeeze(0)
        q, k, v = torch.tensor_split(qkv, self.qkv_split, dim=-1)

        # Flash attention takes Q and K d-major ([N, D, T]) and V token-major ([N, T, D]).
        n_rep = self.num_heads // self.num_kv_heads
        q = q.view(T, self.num_heads, self.head_dim).permute(1, 2, 0)
        k = k.view(T, self.num_kv_heads, self.head_dim).permute(1, 2, 0)
        v = v.view(T, self.num_kv_heads, self.head_dim).transpose(0, 1)
        attn_output = flash_attention(
            q,
            k.repeat_interleave(n_rep, dim=0),
            v.repeat_interleave(n_rep, dim=0),
            scale=self.scale,
            tp_q=False,
            tp_out=True,
        )
        output = o_proj(attn_output.unsqueeze(0), self.o_proj.weight).squeeze(0)
        if self.tp_size > 1:
            dist.all_reduce(output, group=self.tp_group)
        return output


class NeuronQwen3VLMLP(nn.Module):
    """SwiGLU MLP, intermediate dim sharded, all-reduce after the down projection."""

    def __init__(
        self, hidden_size: int, intermediate_size: int, dtype=torch.bfloat16, tp_group=None
    ):
        super().__init__()
        tp_size, _ = _tp_size_rank(tp_group)
        if intermediate_size % tp_size:
            raise ValueError(
                f"intermediate_size ({intermediate_size}) must be divisible by tp_size ({tp_size})"
            )
        self.tp_group = tp_group
        self.tp_size = tp_size
        self.gate_proj = _column_parallel(
            hidden_size, intermediate_size, dtype=dtype, tp_group=tp_group
        )
        self.up_proj = _column_parallel(
            hidden_size, intermediate_size, dtype=dtype, tp_group=tp_group
        )
        self.down_proj = _row_parallel(
            intermediate_size, hidden_size, dtype=dtype, tp_group=tp_group
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        output = mlp(
            hidden_states, self.gate_proj.weight, self.up_proj.weight, self.down_proj.weight
        )
        if self.tp_size > 1:
            dist.all_reduce(output, group=self.tp_group)
        return output


class NeuronQwen3VLDecoderLayer(nn.Module):
    """One Qwen3-VL text block, each residual add fused into the RMSNorm after it.

    ``forward`` takes and returns ``(hidden_states, residual)``, whose sum is the block's input /
    output: the add after the MLP is left to the next block's input norm, or to the caller.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        intermediate_size: int,
        eps: float,
        dtype=torch.bfloat16,
        tp_group=None,
    ):
        super().__init__()
        self.input_layernorm = Qwen3VLTextRMSNorm(hidden_size, eps=eps).to(dtype)
        self.self_attn = NeuronQwen3VLAttention(
            hidden_size, num_heads, num_kv_heads, head_dim, eps, dtype=dtype, tp_group=tp_group
        )
        self.post_attention_layernorm = Qwen3VLTextRMSNorm(hidden_size, eps=eps).to(dtype)
        self.mlp = NeuronQwen3VLMLP(hidden_size, intermediate_size, dtype=dtype, tp_group=tp_group)

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``[T, hidden]`` in and out; ``residual`` is None for the first block."""
        normed, residual = rmsnorm_residual_prefill(
            hidden_states,
            self.input_layernorm.weight,
            residual,
            eps=self.input_layernorm.variance_epsilon,
        )
        if residual is None:
            residual = hidden_states
        hidden_states = self.self_attn(normed, cos, sin)

        normed, residual = rmsnorm_residual_prefill(
            hidden_states,
            self.post_attention_layernorm.weight,
            residual,
            eps=self.post_attention_layernorm.variance_epsilon,
        )
        return self.mlp(normed), residual


class NeuronQwen3VLTextModel(nn.Module):
    """The vocab-sharded embedding and the first ``num_layers`` Qwen3-VL text blocks, with no
    final norm."""

    def __init__(self, config, num_layers: int, dtype=torch.bfloat16, tp_group=None):
        super().__init__()
        tp_size, _ = _tp_size_rank(tp_group)
        self._tp_size = tp_size
        if tp_size > 1:
            self.embed_tokens = VocabDimShardedEmbedding(
                config.vocab_size, config.hidden_size, dtype=dtype, tp_group=tp_group
            )
            set_weight_loader(
                self.embed_tokens.weight,
                sharding_weight_loader(
                    shard_dim=0,
                    shard_size=self.embed_tokens.vocab_size_per_rank,
                    num_shards=tp_size,
                    pad_shard=True,
                ),
            )
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, dtype=dtype)
        self.layers = nn.ModuleList(
            [
                NeuronQwen3VLDecoderLayer(
                    hidden_size=config.hidden_size,
                    num_heads=config.num_attention_heads,
                    num_kv_heads=config.num_key_value_heads,
                    head_dim=config.head_dim,
                    intermediate_size=config.intermediate_size,
                    eps=config.rms_norm_eps,
                    dtype=dtype,
                    tp_group=tp_group,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        rank: torch.Tensor,
        encoder_embeds: torch.Tensor | None = None,
        encoder_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Embedding -> vision splice -> causal blocks, on a bucketed presentation.

        ``input_ids`` ``[T]``; ``cos`` / ``sin`` ``[T, head_dim]`` mRoPE; ``rank`` ``[1]``, the TP
        rank as a tensor so every rank runs the same NEFF; ``encoder_embeds``
        ``[1, rows, hidden * (1 + L)]``, the vision tower output (None for text only), and
        ``encoder_positions`` ``[1, rows]`` int32, its rows' sequence positions. Returns
        ``[T, hidden]``.
        """
        if self._tp_size > 1:
            hidden_states = self.embed_tokens(input_ids, scatter_tokens=False, rank=rank)
        else:
            hidden_states = self.embed_tokens(input_ids)

        deepstack = None
        if encoder_embeds is not None:
            # One slot per token: vision row i for an image position, out of range (``rows``)
            # for text. The scatter needs distinct in-range positions, which
            # ``_encode_keyframes`` checks against the <|image_pad|> runs.
            rows = encoder_embeds.shape[1]
            encoder_slot_mapping = torch.full(
                (input_ids.shape[0],), rows, dtype=torch.int32, device=hidden_states.device
            ).scatter(
                0,
                encoder_positions[0].to(torch.int64),
                torch.arange(rows, dtype=torch.int32, device=hidden_states.device),
            )
            # No SP: every rank holds the whole sequence, so positions need no rank offset.
            hidden_states, deepstack = merge_multimodal_embeddings(
                hidden_states,
                encoder_embeds.to(hidden_states.dtype),
                encoder_slot_mapping,
                rank=torch.zeros_like(rank),
            )

        residual = None
        for layer_idx, layer in enumerate(self.layers):
            hidden_states, residual = layer(hidden_states, residual, cos, sin)
            # Deepstack slice i is added after block i. HF's hidden_states[k] is block k - 1's
            # output before its own deepstack add, so the last block built here gets none.
            if (
                deepstack is not None
                and layer_idx < deepstack.shape[0]
                and layer_idx < len(self.layers) - 1
            ):
                hidden_states = hidden_states + deepstack[layer_idx]
        return hidden_states + residual


class NeuronQwen3VLTextEncoder(nn.Module):
    """MiniMax-H3 conditioning encoder: the first 50 Qwen3-VL text blocks, no final norm, and
    the vision tower for keyframes. ``model_path`` holds it in ``text_encoder/``.

    ``language_model`` and ``visual`` mirror the HF ``model.language_model`` and
    ``model.visual``; blocks 50..63, the final norm and ``lm_head`` are never read.
    """

    # Class attribute so the module tests can retarget it on a reduced-depth checkpoint.
    num_conditioning_layers = MINIMAX_H3_TEXT_ENCODER_LAYER

    # transformers' mRoPE position builder, which reads only these two off ``self``: the method
    # below and ``config``.
    get_rope_index = Qwen3VLModel.get_rope_index
    get_vision_position_ids = Qwen3VLModel.get_vision_position_ids

    def __init__(self, model_path: str, dtype: torch.dtype = torch.bfloat16, tp_group=None):
        super().__init__()
        self._compiled = None
        self._compiled_visual = None
        self._dtype = dtype
        self._tp_group = tp_group
        self._tp_size, self._tp_rank = _tp_size_rank(tp_group)

        self.config = Qwen3VLConfig.from_pretrained(model_path, subfolder="text_encoder")
        tc = self.config.text_config
        if tc.num_hidden_layers <= self.num_conditioning_layers:
            raise ValueError(
                f"MiniMax-H3 conditions on hidden_states[{self.num_conditioning_layers}], which "
                f"needs more than {self.num_conditioning_layers} decoder layers, but the "
                f"checkpoint has {tc.num_hidden_layers}: the last hidden state of a stack "
                "truncated to exactly that depth is post-norm."
            )
        # FP32 vision tower; its rows round to BF16 once, where the text blocks take them.
        self.visual = NeuronQwen3VLVisionModel(
            self.config.vision_config, dtype=torch.float32, tp_group=tp_group
        )
        self.language_model = NeuronQwen3VLTextModel(
            tc, self.num_conditioning_layers, dtype=dtype, tp_group=tp_group
        )

    def load_weights(self, model_path: str):
        """Load the conditioner from ``<model_path>/text_encoder``, sharded for this rank."""
        num_params = load_checkpoint(
            self,
            os.path.join(model_path, "text_encoder"),
            prefix="model.",
            tp_group=self._tp_group,
        )
        logger.info(
            "Loaded Qwen3-VL conditioner: %d params (rank=%d, tp=%d)",
            num_params,
            self._tp_rank,
            self._tp_size,
        )

    @property
    def dtype(self):
        return self._dtype

    def _encode_keyframes(self, pixel_values, image_grid_thw, image_positions, device):
        """Run the vision tower once per keyframe; returns their output rows in order and the
        rows' sequence positions, ``image_positions`` (the ``<|image_pad|>`` rows)."""
        num_patches = image_grid_thw.prod(-1).tolist()
        merge = self.config.vision_config.spatial_merge_size**2
        num_rows = [n // merge for n in num_patches]
        # Each keyframe's <|image_pad|> run, in presentation order.
        breaks = (image_positions.diff() > 1).nonzero().flatten() + 1
        run_lengths = [len(run) for run in image_positions.tensor_split(breaks)]
        if run_lengths != num_rows:
            raise ValueError(
                f"input_ids has <|image_pad|> runs of {run_lengths} tokens but the keyframes "
                f"merge to {num_rows}"
            )
        fn = self._compiled_visual if self._compiled_visual is not None else self.visual
        outputs = []
        for pixels, grid in zip(pixel_values.split(num_patches), image_grid_thw):
            inputs = vision_inputs(pixels.float(), grid[None], self.config.vision_config)
            outputs.append(fn(*(x.to(device) for x in inputs)))
        # Joined on the host, where this encoder builds its graphs' inputs.
        encoder_embeds = (
            outputs[0]
            if len(outputs) == 1
            else torch.cat([out.cpu() for out in outputs], dim=1).to(device)
        )
        # Cast on the host: a host-to-device copy cannot cast.
        return encoder_embeds, image_positions[None].to(torch.int32).to(device)

    def compile(self, *args, **kwargs):
        """Compile the text and vision graphs, lazily: text-only use never builds the vision one.

        Static shapes: each text bucket, and each keyframe resolution, is a graph of its own.
        """
        kwargs.setdefault("dynamic", False)
        # Bound methods, not the modules: a compiled module assigned here would register as a
        # submodule and duplicate its weights in state_dict.
        self._compiled = torch.compile(self.language_model.forward, *args, **kwargs)
        self._compiled_visual = torch.compile(self.visual.forward, *args, **kwargs)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
    ) -> _TextEncoderOutput:
        """Encode a ``[1, bucket]`` presentation, right-padded to a multiple of
        ``MINIMAX_H3_TEXT_SEQ_BUCKET``; returns ``[1, T, hidden]`` for its T real tokens.

        The image rows, mRoPE positions and vision inputs are built on the host. ``pixel_values``
        / ``image_grid_thw`` are the image processor's outputs for the keyframes, in order.
        """
        if input_ids.shape[0] != 1:
            raise ValueError(
                "encodes one presentation at a time, as get_qwen3vl_prompt_embeds does; got "
                f"a batch of {input_ids.shape[0]}"
            )
        if attention_mask.shape != input_ids.shape:
            raise ValueError(
                f"attention_mask {tuple(attention_mask.shape)} does not match input_ids "
                f"{tuple(input_ids.shape)}"
            )
        # Any other length would compile a graph of its own.
        if input_ids.shape[1] % MINIMAX_H3_TEXT_SEQ_BUCKET:
            raise ValueError(
                f"input_ids has {input_ids.shape[1]} tokens, not a multiple of "
                f"MINIMAX_H3_TEXT_SEQ_BUCKET ({MINIMAX_H3_TEXT_SEQ_BUCKET})"
            )
        input_ids, attention_mask = input_ids.cpu(), attention_mask.cpu()
        T = int(attention_mask.sum())
        if not attention_mask[0, :T].all():
            raise ValueError("input_ids must be right-padded")
        device = next(self.parameters()).device
        if (input_ids == self.config.video_token_id).any():
            raise NotImplementedError("video vision blocks are not supported")
        image_rows = input_ids[0] == self.config.image_token_id

        # Before get_rope_index, which assumes the vision inputs match the image rows.
        vision = ()
        if image_rows.any():
            if pixel_values is None or image_grid_thw is None:
                raise ValueError(
                    "input_ids has <|image_pad|> tokens but no pixel_values / image_grid_thw"
                )
            image_grid_thw = image_grid_thw.cpu()
            vision = self._encode_keyframes(
                pixel_values.cpu(), image_grid_thw, image_rows.nonzero().flatten(), device
            )

        position_ids, _ = self.get_rope_index(
            input_ids, image_rows[None].long(), image_grid_thw=image_grid_thw
        )
        cos, sin = Qwen3VLTextRotaryEmbedding(self.config.text_config)(
            torch.empty(0, dtype=self._dtype), position_ids
        )

        rank = torch.tensor([self._tp_rank], dtype=torch.long, device=device)
        fn = self._compiled if self._compiled is not None else self.language_model
        hidden_states = fn(
            input_ids[0].to(device), cos[0].to(device), sin[0].to(device), rank, *vision
        )
        return _TextEncoderOutput(hidden_states[None, :T])
