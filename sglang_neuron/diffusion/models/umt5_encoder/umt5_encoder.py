# SPDX-License-Identifier: Apache-2.0
"""TP-sharded T5 / UMT5 encoder blocks for Neuron.

FLUX's T5-XXL text encoder (``models/flux/text_encoders.py``) and Wan2.2's UMT5-XXL text encoder
(``NeuronTextEncoderWrapper``) are built from them:
  - VocabDimShardedEmbedding2D: vocab-sharded embedding, all-reduce after lookup
  - NeuronUMT5RelativePositionBias: relative position bias sharded along the head dim
  - NeuronUMT5SelfAttention: attention heads sharded across ranks, all-reduce after O projection
  - NeuronUMT5GatedFFN: FFN intermediate dim sharded across ranks, all-reduce after down projection
  - NeuronUMT5EncoderBlock: one pre-norm UMT5 block (per-layer relative position bias)
  - NeuronTextEncoderWrapper: the whole UMT5 encoder, loaded from the checkpoint's safetensors
  - t5_block_mappings: checkpoint keys of one encoder block

No sequence parallelism: seq_len=512 is too short for an SP benefit.
"""

import logging
import math
import os
from collections import namedtuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from transformers import UMT5Config
from transformers.activations import NewGELUActivation
from transformers.models.umt5.modeling_umt5 import UMT5LayerNorm
from sglang_neuron.diffusion.distributed.tp_loaders import cast_weight_loader
from sglang_neuron.diffusion.vn.utils.shard_loaders import sharding_weight_loader
from sglang_neuron.layers.embedding import VocabDimShardedEmbedding
from sglang_neuron.layers.linear import ColumnParallelLinear, RowParallelLinear
from sglang_neuron.utils.checkpoints import SafetensorsCheckpoint
from sglang_neuron.utils.weight_loader import get_weight_loader, set_weight_loader

logger = logging.getLogger(__name__)

_TextEncoderOutput = namedtuple("_TextEncoderOutput", ["last_hidden_state"])


# =========================================================================
# Embedding
# =========================================================================


def _column_parallel(in_features: int, out_features: int, dtype, tp_group) -> ColumnParallelLinear:
    """The LLM path's column-parallel linear, without output gather, loading this rank's rows."""
    layer = ColumnParallelLinear(
        in_features, out_features, dtype=dtype, gather_output=False, tp_group=tp_group
    )
    set_weight_loader(
        layer.weight,
        sharding_weight_loader(
            shard_dim=0, shard_size=layer.out_features_per_rank, num_shards=layer.tp_size
        ),
    )
    return layer


def _row_parallel(in_features: int, out_features: int, dtype, tp_group) -> RowParallelLinear:
    """The LLM path's row-parallel linear (all-reduce after the product), loading this rank's
    columns."""
    layer = RowParallelLinear(in_features, out_features, dtype=dtype, tp_group=tp_group)
    set_weight_loader(
        layer.weight,
        sharding_weight_loader(
            shard_dim=1, shard_size=layer.in_features_per_rank, num_shards=layer.tp_size
        ),
    )
    return layer


class VocabDimShardedEmbedding2D(VocabDimShardedEmbedding):
    """The LLM path's vocab-sharded embedding, for [B, T] input like nn.Embedding.

    The base class takes 1D [T] ids, so this subclass flattens, delegates, and reshapes. With
    ``rank`` a tensor (the T5 encoder's graph input) the vocabulary range is computed from it,
    so every TP rank traces one graph; without one it is this process's rank in ``tp_group``.
    It loads this rank's rows of the table, the last shard padded.
    """

    def __init__(self, vocab_size: int, embed_dim: int, *, dtype, tp_group):
        super().__init__(vocab_size, embed_dim, dtype=dtype, tp_group=tp_group)
        set_weight_loader(
            self.weight,
            sharding_weight_loader(
                shard_dim=0,
                shard_size=self.vocab_size_per_rank,
                num_shards=self.tp_size,
                pad_shard=True,
            ),
        )

    def forward(self, input, scatter_tokens=False, rank=None):
        if rank is None:
            own = dist.get_rank(self.tp_group) if dist.is_initialized() else 0
            rank = torch.tensor([own], device=input.device)
        if input.dim() == 2:
            B, T = input.shape
            flat = input.reshape(-1)
            out = super().forward(flat, scatter_tokens=scatter_tokens, rank=rank)
            return out.reshape(B, T, -1)
        return super().forward(input, scatter_tokens=scatter_tokens, rank=rank)


# =========================================================================
# Relative Position Bias
# =========================================================================


class NeuronUMT5RelativePositionBias(nn.Module):
    """Computes T5-style relative position bias for self-attention.

    T5 uses learned relative position biases instead of RoPE.
    Rather than encoding "I am at position 5", it encodes
    "the key is 3 positions to the right of the query" by adding a learned
    bias to attention scores. This is a good fit for encoder models with short,
    fixed-length sequences (512 tokens for FLUX's T5-XXL). Modern decoder-only LLMs prefer
    RoPE because it has no learned parameters, is KV-cache friendly (the
    position info is baked into cached K vectors), and scales better to long
    sequences.

    To avoid learning O(T^2) parameters for every possible distance, distances
    are mapped to a fixed number of buckets (32 by default) via
    _relative_position_bucket: small distances get their own bucket for
    fine-grained local attention, while large distances are grouped on a log
    scale.

    Provenance: the bucketing logic and bias computation are copied from HF's
    UMT5Attention._relative_position_bucket and UMT5Attention.compute_bias
    (transformers.models.umt5.modeling_umt5.UMT5Attention). This module keeps a
    standalone copy rather than reusing HF directly because:
      1. HF's methods are instance methods on UMT5Attention, not static
         utilities — reusing them would require instantiating a full
         UMT5Attention with unused Q/K/V/O linear layers.
      2. The only semantic difference is the TP head slice at the end of
         forward(): HF returns [1, num_heads, Q, K] for all heads, while
         this version returns [1, num_heads_per_rank, Q, K] for the local
         rank's head range.

    HF's encoder has no TP support. This Neuron version exists to enable
    tensor-parallel inference.
    """

    def __init__(
        self,
        num_buckets: int,
        num_heads: int,
        num_heads_per_rank: int,
        tp_rank: int,
        max_distance: int = 128,
        dtype=None,
    ):
        super().__init__()
        self.num_buckets = num_buckets
        self.num_heads = num_heads
        self.num_heads_per_rank = num_heads_per_rank
        self.tp_rank = tp_rank
        self.max_distance = max_distance

        # Sharded [num_buckets, num_heads_per_rank] — each rank holds its head slice
        self.weight = nn.Parameter(torch.empty(num_buckets, num_heads_per_rank, dtype=dtype))

    def _relative_position_bucket(self, relative_position: torch.Tensor) -> torch.Tensor:
        """Translate relative position to bucket index (T5 bucketing scheme).

        Encoder-only (bidirectional) variant: half the buckets cover negative
        offsets, half cover positive. HF's decoder path (unidirectional) is
        omitted since this module is only used in an encoder.

        With defaults (num_buckets=32, max_distance=128):
          - 16 buckets per direction (positive / negative)
          - Of those 16: first 8 are exact (distance 0-7 each get own bucket),
            remaining 8 are log-spaced (distances 8-128+ grouped coarsely)
        """
        # Bidirectional: split buckets evenly between left/right directions
        num_buckets = self.num_buckets // 2
        # Positive offsets (key right of query) → use upper half of buckets
        relative_buckets = (relative_position > 0).to(torch.long) * num_buckets
        relative_position = torch.abs(relative_position)

        # Small distances (< max_exact) get their own bucket for fine-grained
        # local attention patterns
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # Large distances are mapped to log-spaced buckets, so the model can
        # still distinguish "far" from "very far" without wasting parameters
        log_ratio = torch.log(relative_position.float() / max_exact) / math.log(
            self.max_distance / max_exact
        )
        log_ratio = log_ratio * (num_buckets - max_exact)
        relative_position_if_large = max_exact + log_ratio.to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large,
            torch.full_like(relative_position_if_large, num_buckets - 1),
        )

        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets

    def forward(self, query_length: int, key_length: int, device=None) -> torch.Tensor:
        """Compute position bias: [1, num_heads_per_rank, query_length, key_length]."""
        if device is None:
            device = self.weight.device
        # Build the Q×K relative distance matrix
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        relative_position = memory_position - context_position

        # Map distances → bucket indices → learned bias per head
        buckets = self._relative_position_bucket(relative_position)
        values = F.embedding(buckets, self.weight)  # [Q, K, num_heads_per_rank]
        return values.permute(2, 0, 1).unsqueeze(0)  # [1, num_heads_per_rank, Q, K]

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        weight_key = prefix + "weight"
        if weight_key in state_dict:
            full_weight = state_dict[weight_key]
            if full_weight.shape != self.weight.data.shape:
                start_idx = self.tp_rank * self.num_heads_per_rank
                end_idx = start_idx + self.num_heads_per_rank
                state_dict[weight_key] = full_weight[:, start_idx:end_idx]
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )


# =========================================================================
# Self-Attention (TP-sharded heads, all-reduce after O projection)
# =========================================================================


class NeuronUMT5SelfAttention(nn.Module):
    """T5 self-attention with TP-sharded heads.

    Custom Neuron version: Q/K/V use ColumnParallelLinear, O uses
    RowParallelLinear with all-reduce. HF's T5Attention uses plain
    nn.Linear and has no TP support.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_kv: int,
        num_buckets: int,
        max_distance: int,
        eps: float,
        dtype=None,
        tp_group=None,
        has_relative_bias: bool = True,
    ):
        """Build the TP-sharded projections.

        Args:
            has_relative_bias: Build this layer's relative position bias table. False for
                callers that pass a shared ``position_bias`` to :meth:`forward`.
        """
        super().__init__()
        self.d_kv = d_kv
        inner_dim = num_heads * d_kv

        if dist.is_initialized():
            grp = tp_group if tp_group is not None else dist.group.WORLD
            tp_size = dist.get_world_size(grp)
            tp_rank = dist.get_rank(grp)
        else:
            tp_size = 1
            tp_rank = 0

        self.num_heads_per_rank = num_heads // tp_size
        self.scale = d_kv**-0.5

        # Q, K, V: column-parallel (shard output dim = inner_dim)
        self.q = _column_parallel(d_model, inner_dim, dtype, tp_group)
        self.k = _column_parallel(d_model, inner_dim, dtype, tp_group)
        self.v = _column_parallel(d_model, inner_dim, dtype, tp_group)

        # O: row-parallel (shard input dim = inner_dim, all-reduce output)
        self.o = _row_parallel(inner_dim, d_model, dtype, tp_group)

        # Per-layer relative position bias (sharded per rank)
        if has_relative_bias:
            self.relative_attention_bias = NeuronUMT5RelativePositionBias(
                num_buckets=num_buckets,
                num_heads=num_heads,
                num_heads_per_rank=self.num_heads_per_rank,
                tp_rank=tp_rank,
                max_distance=max_distance,
                dtype=dtype,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Attention over ``[B, T, d_model]``; ``position_bias`` overrides this layer's bias table.

        Args:
            attention_mask: Additive mask broadcast to ``[B, heads, T, T]``, added to the bias.
            position_bias: Precomputed ``[1, heads_per_rank, T, T]`` bias shared across layers.
        """
        batch_size, seq_length, _ = hidden_states.shape

        # Project Q, K, V → [B, T, num_heads_per_rank * d_kv]
        q = self.q(hidden_states)
        k = self.k(hidden_states)
        v = self.v(hidden_states)

        # Reshape to [B, num_heads_per_rank, T, d_kv]
        q = q.view(batch_size, seq_length, self.num_heads_per_rank, self.d_kv).transpose(1, 2)
        k = k.view(batch_size, seq_length, self.num_heads_per_rank, self.d_kv).transpose(1, 2)
        v = v.view(batch_size, seq_length, self.num_heads_per_rank, self.d_kv).transpose(1, 2)

        # Attention scores: [B, num_heads_per_rank, T, T]
        scores = torch.matmul(q, k.transpose(3, 2))

        # Add relative position bias
        if position_bias is None:
            position_bias = self.relative_attention_bias(
                seq_length, seq_length, device=scores.device
            )
        if attention_mask is not None:
            position_bias = position_bias + attention_mask
        scores = scores + position_bias

        # Softmax + weighted sum
        attn_weights = F.softmax(scores.float(), dim=-1).to(scores.dtype)
        attn_output = torch.matmul(attn_weights, v)

        # Reshape back: [B, T, num_heads_per_rank * d_kv]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_length, -1)

        # O projection with all-reduce
        return self.o(attn_output)


# =========================================================================
# Gated-GELU FFN (TP-sharded intermediate dim, all-reduce after down proj)
# =========================================================================


class NeuronUMT5GatedFFN(nn.Module):
    """T5 v1.1 gated-GELU feed-forward network with TP sharding.

    Custom Neuron version: wi_0/wi_1 use ColumnParallelLinear, wo uses
    RowParallelLinear with all-reduce.
    """

    def __init__(self, d_model: int, d_ff: int, dtype=None, tp_group=None, wo_dtype=None):
        """Build the TP-sharded projections.

        Args:
            wo_dtype: dtype of the down projection; defaults to ``dtype``. When it differs, the
                gated activation is cast to it before ``wo`` (HF T5 v1.1 ``_keep_in_fp32_modules``).
        """
        super().__init__()
        self.wi_0 = _column_parallel(d_model, d_ff, dtype, tp_group)
        self.wi_1 = _column_parallel(d_model, d_ff, dtype, tp_group)
        self.wo = _row_parallel(d_ff, d_model, dtype if wo_dtype is None else wo_dtype, tp_group)
        # The same activation as HF's T5DenseGatedActDense for T5 v1.1 (gelu_new)
        self.act = NewGELUActivation()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_gelu = self.act(self.wi_0(hidden_states))
        hidden_states = hidden_gelu * self.wi_1(hidden_states)
        if hidden_states.dtype != self.wo.weight.dtype:
            hidden_states = hidden_states.to(self.wo.weight.dtype)
        return self.wo(hidden_states)


# =========================================================================
# Encoder Block
# =========================================================================


class NeuronUMT5EncoderBlock(nn.Module):
    """Single UMT5 encoder block: self-attention + FFN with pre-norm residuals."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_kv: int,
        d_ff: int,
        num_buckets: int,
        max_distance: int,
        eps: float,
        dtype=None,
        tp_group=None,
    ):
        super().__init__()
        self.attn_norm = UMT5LayerNorm(d_model, eps=eps)
        self.self_attn = NeuronUMT5SelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            d_kv=d_kv,
            num_buckets=num_buckets,
            max_distance=max_distance,
            eps=eps,
            dtype=dtype,
            tp_group=tp_group,
        )
        self.ffn_norm = UMT5LayerNorm(d_model, eps=eps)
        self.ffn = NeuronUMT5GatedFFN(d_model=d_model, d_ff=d_ff, dtype=dtype, tp_group=tp_group)

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        # Self-attention with pre-norm and residual
        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask=attention_mask)
        hidden_states = residual + hidden_states

        # FFN with pre-norm and residual
        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states)
        hidden_states = self.ffn(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


# =========================================================================
# Full Encoder + Wrapper
# =========================================================================


def t5_block_mappings(i: int) -> dict[str, str]:
    """Parameter-to-checkpoint keys of encoder block ``i`` shared by the T5 and UMT5 loaders.

    Covers the attention projections, both layer norms and the gated FFN; the relative position
    bias is mapped by each loader (one table per layer in UMT5, one shared table in T5).
    """
    src = f"encoder.block.{i}"
    mappings = {
        f"blocks.{i}.self_attn.{proj}.weight": f"{src}.layer.0.SelfAttention.{proj}.weight"
        for proj in ("q", "k", "v", "o")
    }
    mappings[f"blocks.{i}.attn_norm.weight"] = f"{src}.layer.0.layer_norm.weight"
    for proj in ("wi_0", "wi_1", "wo"):
        mappings[f"blocks.{i}.ffn.{proj}.weight"] = f"{src}.layer.1.DenseReluDense.{proj}.weight"
    mappings[f"blocks.{i}.ffn_norm.weight"] = f"{src}.layer.1.layer_norm.weight"
    return mappings


class NeuronTextEncoderWrapper(nn.Module):
    """Standalone UMT5 text encoder for Neuron with full TP sharding (Wan2.2's text encoder).

    Constructed from config, not from the HF model. All weights load via SafetensorsCheckpoint in a
    single pass.

    Args:
        model_path: Path to the model directory (contains the text_encoder/ subfolder).
        dtype: Model dtype (default: bfloat16).
        tp_group: Tensor-parallel process group. ``None`` uses the world group
            (``groups.get_world_group().device_group``), which is how Wan2.2 shards it: one head
            and 1/world of the FFN and vocabulary per rank.
    """

    def __init__(self, model_path: str, dtype: torch.dtype = torch.bfloat16, tp_group=None):
        super().__init__()
        self._compiled = None
        self._dtype = dtype

        local_files_only = os.path.isdir(model_path)
        config = UMT5Config.from_pretrained(
            model_path, subfolder="text_encoder", local_files_only=local_files_only
        )
        self._config = config

        if dist.is_initialized():
            if tp_group is None:
                from sglang_neuron.diffusion.distributed.groups import get_world_group

                tp_group = get_world_group().device_group
            self._tp_size = dist.get_world_size(tp_group)
            self._tp_rank = dist.get_rank(tp_group)
            _register_world_spanning_group(tp_group)
        else:
            self._tp_size = 1
            self._tp_rank = 0
        self._tp_group = tp_group

        # Embedding: vocab-sharded if TP > 1, else plain nn.Embedding
        if self._tp_size > 1:
            self.embed_tokens = VocabDimShardedEmbedding2D(
                vocab_size=config.vocab_size,
                embed_dim=config.d_model,
                dtype=dtype,
                tp_group=tp_group,
            )
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)

        # Encoder blocks
        self.blocks = nn.ModuleList(
            [
                NeuronUMT5EncoderBlock(
                    d_model=config.d_model,
                    num_heads=config.num_heads,
                    d_kv=config.d_kv,
                    d_ff=config.d_ff,
                    num_buckets=config.relative_attention_num_buckets,
                    max_distance=config.relative_attention_max_distance,
                    eps=config.layer_norm_epsilon,
                    dtype=dtype,
                    tp_group=tp_group,
                )
                for _ in range(config.num_layers)
            ]
        )

        # Final layer norm
        self.final_norm = UMT5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)

    def load_weights(self, model_path: str):
        """Load all weights in one pass of the plugin's checkpoint loader
        (``SafetensorsCheckpoint.load_sharded_pipelined``), which every rank of the TP group runs.

        Args:
            model_path: Path to the text_encoder subfolder with .safetensors files.
        """
        tp_rank, tp_size = self._tp_rank, self._tp_size
        checkpoint = SafetensorsCheckpoint(model_path)

        # Build mappings: module param name -> checkpoint key
        # Checkpoint format (UMT5EncoderModel):
        #   shared.weight                                                    -> embedding
        #   encoder.block.{i}.layer.0.SelfAttention.{q,k,v,o}.weight       -> attention
        #   encoder.block.{i}.layer.0.SelfAttention.relative_attention_bias.weight
        #   encoder.block.{i}.layer.0.layer_norm.weight                     -> attn norm
        #   encoder.block.{i}.layer.1.DenseReluDense.{wi_0,wi_1,wo}.weight -> FFN
        #   encoder.block.{i}.layer.1.layer_norm.weight                     -> FFN norm
        #   encoder.final_layer_norm.weight
        mappings: dict[str, str] = {"embed_tokens.weight": "shared.weight"}

        # Attach a sharding weight loader to each relative_attention_bias.weight so the
        # checkpoint's [num_buckets, num_heads] is sliced to [num_buckets, num_heads_per_rank].
        num_heads_per_rank = self._config.num_heads // tp_size
        for i in range(len(self.blocks)):
            param = self.blocks[i].self_attn.relative_attention_bias.weight
            set_weight_loader(
                param,
                sharding_weight_loader(
                    shard_dim=1,
                    shard_size=num_heads_per_rank,
                    num_shards=tp_size,
                ),
            )

        for i in range(len(self.blocks)):
            mappings.update(t5_block_mappings(i))
            mappings[f"blocks.{i}.self_attn.relative_attention_bias.weight"] = (
                f"encoder.block.{i}.layer.0.SelfAttention.relative_attention_bias.weight"
            )
        mappings["final_norm.weight"] = "encoder.final_layer_norm.weight"

        # Every weight loads in the encoder's dtype, the UMT5LayerNorm weights (built in float32)
        # included, cast after the loader's transform as the DiT's are.
        logged_casts: set = set()
        for name, param in self.named_parameters():
            param.data = param.data.to(self._dtype)
            set_weight_loader(
                param,
                cast_weight_loader(get_weight_loader(param), self._dtype, name, logged_casts),
            )
        result = checkpoint.load_sharded_pipelined(
            tp_rank, tp_size, self, mappings, torch.device("cpu")
        )
        self.load_state_dict(result.state_dict, strict=False, assign=True)
        logger.info(
            "Loaded text encoder weights: %d params loaded, %d checkpoint keys unused (rank=%d, tp=%d)",
            len(result.state_dict),
            len(result.unexpected_keys),
            tp_rank,
            tp_size,
        )

    @property
    def dtype(self):
        return self._dtype

    def _encode(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, rank: torch.Tensor
    ) -> torch.Tensor:
        """Core forward: embedding -> blocks -> final norm.

        Separated from forward() so torch.compile wraps only the compute graph, not the output
        namedtuple construction.

        Args:
            input_ids: [B, T] token IDs.
            attention_mask: [B, T] binary mask (1 = valid, 0 = padding).
            rank: [1] tensor containing the TP rank, passed as a dynamic input so the compiled
                graph is rank-independent (same NEFF for all ranks).
        """
        if isinstance(self.embed_tokens, VocabDimShardedEmbedding2D):
            hidden_states = self.embed_tokens(input_ids, rank=rank)
        else:  # TP = 1: a plain nn.Embedding takes no rank
            hidden_states = self.embed_tokens(input_ids)

        # Build attention mask: [B, 1, 1, T] with -inf for padding
        extended_mask = attention_mask[:, None, None, :]
        extended_mask = extended_mask.to(dtype=hidden_states.dtype)
        extended_mask = (1.0 - extended_mask) * torch.finfo(hidden_states.dtype).min

        for block in self.blocks:
            hidden_states = block(hidden_states, attention_mask=extended_mask)

        hidden_states = self.final_norm(hidden_states)

        # Zero the padded positions in-graph. The encoder leaks nonzero values past the real
        # sequence length (attention only masks padded keys, not padded query rows), and the
        # downstream DiT cross-attention applies no key-padding mask over these positions, so the
        # pipeline must feed it zero-padded embeddings. Doing it here (inside the compiled graph)
        # keeps the whole post-process on-device with no eager round-trip.
        hidden_states = hidden_states * attention_mask[:, :, None].to(hidden_states.dtype)
        return hidden_states

    def compile(self, *args, **kwargs):
        """Compile the forward graph for Neuron (NEFF built on first call)."""
        self._compiled = torch.compile(self._encode, *args, **kwargs)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> _TextEncoderOutput:
        # Create the rank tensor on the input's device so it is a dynamic graph input rather than
        # a baked-in constant.
        rank = torch.tensor([self._tp_rank], dtype=torch.long, device=input_ids.device)
        fn = self._compiled if self._compiled is not None else self._encode
        hidden_state = fn(input_ids, attention_mask, rank)
        return _TextEncoderOutput(hidden_state)


def _register_world_spanning_group(group) -> None:
    """Register ``group``'s replica groups when it spans the world (the Wan text encoder's group).

    torch-neuronx resolves a collective traced on ``group`` through its mesh registry
    (``neuron_api.register_process_group_replica_groups``); a group over every rank is its own
    one-group partition. Groups that span part of the world are registered by their creator.
    """
    world = dist.get_world_size()
    if world <= 1 or dist.get_world_size(group) != world:
        return
    from sglang_neuron.diffusion.neuron_api import register_process_group_replica_groups

    register_process_group_replica_groups(group.group_name, [list(range(world))])
