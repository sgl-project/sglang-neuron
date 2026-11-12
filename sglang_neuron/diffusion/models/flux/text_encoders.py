# SPDX-License-Identifier: Apache-2.0
"""Neuron FLUX.1 text encoders: replicated CLIP-L and T5-XXL (T5 v1.1) TP-sharded over all ranks."""

import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPTextModel, T5Config
from transformers.models.t5.modeling_t5 import T5LayerNorm
from sglang_neuron.diffusion.distributed.tp_loaders import cast_weight_loader
from sglang_neuron.diffusion.vn.utils.shard_loaders import sharding_weight_loader
from sglang_neuron.utils.checkpoints import SafetensorsCheckpoint
from sglang_neuron.utils.weight_loader import get_weight_loader, set_weight_loader
from sglang_neuron.diffusion.distributed.groups import get_world_group

from sglang_neuron.diffusion.models.flux.flux_transformer import (
    T5_SEQ_LEN,
    TRANSFORMER_COMPILER_ARGS,
    compile_unit,
)
from sglang_neuron.diffusion.models.umt5_encoder.umt5_encoder import (
    NeuronUMT5GatedFFN,
    NeuronUMT5RelativePositionBias,
    NeuronUMT5SelfAttention,
    VocabDimShardedEmbedding2D,
    t5_block_mappings,
)
from sglang_neuron.diffusion.neuron_api import register_process_group_replica_groups

T5_PRECISIONS = ("bf16", "bf16_fp32_wo", "fp32")


class NeuronCLIPTextEncoder(nn.Module):
    """CLIP-L text encoder returning the EOS-pooled hidden state, with a host-built causal mask."""

    def __init__(self, model_path: str, dtype=torch.bfloat16, num_hidden_layers: int | None = None):
        super().__init__()
        overrides = {} if num_hidden_layers is None else {"num_hidden_layers": num_hidden_layers}
        self.model = CLIPTextModel.from_pretrained(
            model_path,
            subfolder="text_encoder",
            dtype=dtype,
            attn_implementation="eager",
            local_files_only=os.path.isdir(model_path),
            **overrides,
        )
        self.eos_token_id = self.model.config.eos_token_id
        n = self.model.config.max_position_embeddings
        self.max_length = n
        mask = torch.full((n, n), torch.finfo(dtype).min, dtype=dtype).triu(1)
        self.register_buffer("causal_mask", mask[None, None], persistent=False)
        self._compiled = None

    def eos_index(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Host-side HF pooling index for CPU ``[B, 77]`` ids, as an int64 ``[B]`` tensor."""
        ids = input_ids.to(torch.int)
        if self.eos_token_id == 2:
            return ids.argmax(dim=-1).to(torch.long)
        return (ids == self.eos_token_id).int().argmax(dim=-1).to(torch.long)

    def _encode(self, input_ids: torch.Tensor, eos_index: torch.Tensor) -> torch.Tensor:
        """Embeddings, causal encoder layers and final norm, then each prompt's EOS row ``[B, 768]``."""
        h = self.model.embeddings(input_ids=input_ids)
        for layer in self.model.encoder.layers:
            h = layer(h, self.causal_mask)
        h = self.model.final_layer_norm(h)
        index = eos_index.view(-1, 1, 1).expand(-1, 1, h.shape[-1])
        return h.gather(1, index).squeeze(1)

    def compile(self, *args, **kwargs):
        """Compile the encoder graph for Neuron (NEFF built on first call)."""
        self._compiled = compile_unit(
            self._encode, "flux_clip", TRANSFORMER_COMPILER_ARGS, *args, **kwargs
        )

    def forward(self, input_ids: torch.Tensor, eos_index: torch.Tensor) -> torch.Tensor:
        """Pooled ``[B, 768]`` output for ``[B, 77]`` ids; raises on any other sequence length."""
        if input_ids.shape[-1] != self.max_length:
            raise ValueError(
                f"CLIP input_ids must have {self.max_length} tokens (the causal mask is "
                f"{self.max_length}x{self.max_length}), got {input_ids.shape[-1]}"
            )
        fn = self._compiled if self._compiled is not None else self._encode
        return fn(input_ids, eos_index)


class _T5RelativeBias(NeuronUMT5RelativePositionBias):
    """Head-sharded T5 relative bias shared by all blocks; buckets are precomputed on the host."""

    def __init__(self, config: T5Config, tp_rank: int, tp_size: int, dtype):
        num_heads_per_rank = config.num_heads // tp_size
        super().__init__(
            num_buckets=config.relative_attention_num_buckets,
            num_heads=config.num_heads,
            num_heads_per_rank=num_heads_per_rank,
            tp_rank=tp_rank,
            max_distance=config.relative_attention_max_distance,
            dtype=dtype,
        )
        position = torch.arange(T5_SEQ_LEN, dtype=torch.long, device="cpu")
        buckets = self._relative_position_bucket(position[None, :] - position[:, None])
        self.register_buffer("buckets", buckets, persistent=False)
        set_weight_loader(
            self.weight,
            sharding_weight_loader(shard_dim=1, shard_size=num_heads_per_rank, num_shards=tp_size),
        )

    def forward(self) -> torch.Tensor:
        """Return the position bias ``[1, heads_per_rank, 512, 512]``."""
        return F.embedding(self.buckets, self.weight).permute(2, 0, 1).unsqueeze(0)


class NeuronT5Block(nn.Module):
    """Pre-norm T5 encoder block; the residual dtype follows type promotion as in HF."""

    def __init__(self, config: T5Config, dtype, wo_dtype, tp_group):
        super().__init__()
        eps = config.layer_norm_epsilon
        self.attn_norm = T5LayerNorm(config.d_model, eps=eps).to(dtype)
        self.self_attn = NeuronUMT5SelfAttention(
            d_model=config.d_model,
            num_heads=config.num_heads,
            d_kv=config.d_kv,
            num_buckets=config.relative_attention_num_buckets,
            max_distance=config.relative_attention_max_distance,
            eps=eps,
            dtype=dtype,
            tp_group=tp_group,
            has_relative_bias=False,
        )
        self.ffn_norm = T5LayerNorm(config.d_model, eps=eps).to(dtype)
        self.ffn = NeuronUMT5GatedFFN(
            config.d_model, config.d_ff, dtype=dtype, tp_group=tp_group, wo_dtype=wo_dtype
        )

    def forward(self, hidden_states: torch.Tensor, position_bias: torch.Tensor) -> torch.Tensor:
        attn = self.self_attn(self.attn_norm(hidden_states), position_bias=position_bias)
        hidden_states = hidden_states + attn
        return hidden_states + self.ffn(self.ffn_norm(hidden_states))


class NeuronT5Encoder(nn.Module):
    """T5-XXL encoder TP-sharded over all ranks (the world group), no mask, ``[1, 512]`` ids.

    ``precision``: ``bf16`` (all bf16), ``bf16_fp32_wo`` (HF ``_keep_in_fp32_modules`` recipe), ``fp32``.
    """

    def __init__(
        self,
        model_path: str,
        precision: str = "bf16",
        num_layers: int | None = None,
        output_dtype=torch.bfloat16,
    ):
        super().__init__()
        if precision not in T5_PRECISIONS:
            raise ValueError(f"t5_precision must be one of {T5_PRECISIONS}, got {precision!r}")
        self._dtype = torch.float32 if precision == "fp32" else torch.bfloat16
        wo_dtype = torch.bfloat16 if precision == "bf16" else torch.float32
        self.output_dtype = output_dtype
        config = T5Config.from_pretrained(
            model_path, subfolder="text_encoder_2", local_files_only=os.path.isdir(model_path)
        )
        if num_layers is not None:
            config.num_layers = num_layers

        self.tp_group = get_world_group().device_group
        self.tp_size = dist.get_world_size(self.tp_group)
        self.tp_rank = dist.get_rank(self.tp_group)
        for name in ("num_heads", "d_ff", "vocab_size"):
            if getattr(config, name) % self.tp_size:
                raise ValueError(
                    f"T5 {name}={getattr(config, name)} is not divisible by the "
                    f"tensor-parallel size {self.tp_size}"
                )
        if self.tp_size > 1:
            register_process_group_replica_groups(
                self.tp_group.group_name, [list(range(self.tp_size))]
            )
        self.register_buffer(
            "rank", torch.tensor([self.tp_rank], dtype=torch.long), persistent=False
        )

        self.embed_tokens = VocabDimShardedEmbedding2D(
            vocab_size=config.vocab_size,
            embed_dim=config.d_model,
            dtype=self._dtype,
            tp_group=self.tp_group,
        )
        self.relative_bias = _T5RelativeBias(config, self.tp_rank, self.tp_size, self._dtype)
        self.blocks = nn.ModuleList(
            NeuronT5Block(config, self._dtype, wo_dtype, self.tp_group)
            for _ in range(config.num_layers)
        )
        self.final_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon).to(self._dtype)
        self._compiled = None

    @property
    def dtype(self):
        return self._dtype

    def load_weights(self, model_path: str) -> None:
        """Load this rank's shards from the ``text_encoder_2`` safetensors; raises on any missing key."""
        mappings = {
            "embed_tokens.weight": "shared.weight",
            "relative_bias.weight": (
                "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"
            ),
            "final_norm.weight": "encoder.final_layer_norm.weight",
        }
        for i in range(len(self.blocks)):
            mappings.update(t5_block_mappings(i))

        logged_casts: set = set()
        for name, param in self.named_parameters():
            set_weight_loader(
                param,
                cast_weight_loader(get_weight_loader(param), param.dtype, name, logged_casts),
            )
        state_dict = (
            SafetensorsCheckpoint(model_path)
            .load_sharded_pipelined(self.tp_rank, self.tp_size, self, mappings, torch.device("cpu"))
            .state_dict
        )
        self.load_state_dict(state_dict, strict=True, assign=True)

    def _encode(self, input_ids: torch.Tensor, rank: torch.Tensor) -> torch.Tensor:
        """Embedding, 24 blocks sharing one position bias, final norm, cast to ``output_dtype``."""
        hidden_states = self.embed_tokens(input_ids, rank=rank)
        position_bias = self.relative_bias()
        for block in self.blocks:
            hidden_states = block(hidden_states, position_bias)
        return self.final_norm(hidden_states).to(self.output_dtype)

    def compile(self, *args, **kwargs):
        """Compile the encoder graph for Neuron (NEFF built on first call)."""
        self._compiled = compile_unit(
            self._encode, "flux_t5", TRANSFORMER_COMPILER_ARGS, *args, **kwargs
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """``[1, 512, 4096]`` hidden states for ``[1, 512]`` ids; raises on any other length."""
        if input_ids.shape[-1] != T5_SEQ_LEN:
            raise ValueError(
                f"T5 input_ids must have {T5_SEQ_LEN} tokens (the position bias is "
                f"{T5_SEQ_LEN}x{T5_SEQ_LEN}), got {input_ids.shape[-1]}"
            )
        fn = self._compiled if self._compiled is not None else self._encode
        return fn(input_ids, self.rank)
