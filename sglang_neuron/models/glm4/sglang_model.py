# SPDX-License-Identifier: Apache-2.0
"""Glm4ForCausalLM: the host SGLang builds for a GLM-4 checkpoint (models/host.py), on the shared dense transformer
(models/transformer). GLM-4 is the dense core with QKV biases, a fused gate and up projection, an RMSNorm after the
attention and after the MLP, and a partial RoPE whose interleaved channels its loader permutes into the core's
contiguous halves (rotary.py)."""

from torch import nn

from sglang_neuron.models.glm4 import rotary
from sglang_neuron.models.host import ModelHost
from sglang_neuron.models.transformer.config import (TransformerConfig, check_bf16_launch, check_no_expert_parallel,
                                                     rope_theta_of)
from sglang_neuron.models.transformer.model import TransformerForCausalLM
from sglang_neuron.utils.weight_loader import SafetensorsWeightLoader, get_weight_loader, set_weight_loader


def _partial_rotary_factor(hf_config) -> float:
    params = getattr(hf_config, "rope_parameters", None) or {}
    return float(params.get("partial_rotary_factor", getattr(hf_config, "partial_rotary_factor", 1.0)))


def glm4_config(hf_config) -> TransformerConfig:
    """The dense transformer of a GLM-4 checkpoint, refusing a quantized one, sliding-window layers and a scaled RoPE.
    GLM-4 has no sliding window in SGLang or transformers; the refusal catches a config.json that declares sliding
    layers (``layer_types``, which the runtime reads for each layer's window) or asks for one (``use_sliding_window``),
    which no GLM-4 implementation applies."""
    if getattr(hf_config, "quantization_config", None):
        raise ValueError("GLM-4: the graphs serve bf16 checkpoints; this one is quantized "
                         f"({hf_config.quantization_config})")
    if getattr(hf_config, "use_sliding_window", False) or "sliding_attention" in (getattr(hf_config, "layer_types",
                                                                                          None) or ()):
        raise ValueError("GLM-4: sliding-window attention layers are not served; the graphs attend the whole context")
    head_dim = getattr(hf_config, "head_dim", None) or hf_config.hidden_size // hf_config.num_attention_heads
    return TransformerConfig(
        vocab_size=hf_config.vocab_size,
        hidden_size=hf_config.hidden_size,
        intermediate_size=hf_config.intermediate_size,
        num_hidden_layers=hf_config.num_hidden_layers,
        num_attention_heads=hf_config.num_attention_heads,
        num_key_value_heads=hf_config.num_key_value_heads,
        head_dim=head_dim,
        rms_norm_eps=hf_config.rms_norm_eps,
        rope_theta=rope_theta_of(hf_config, "GLM-4"),
        rotary_dim=int(head_dim * _partial_rotary_factor(hf_config)),
        tie_word_embeddings=bool(getattr(hf_config, "tie_word_embeddings", False)),
        qkv_bias=bool(getattr(hf_config, "attention_bias", True)),
        post_attention_norm=True,
        post_mlp_norm=True,
    )


class Glm4Graph(TransformerForCausalLM):
    fused_gate_up = True

    def __init__(self, config: TransformerConfig, phase: str):
        super().__init__(config, phase)
        perm = rotary.half_split_permutation(config.head_dim, config.rotary_dim)
        for layer in self.layers:
            attn = layer.self_attn
            # Q and K rotate, V does not; the fused output columns are this rank's Q heads, then its K heads, then V
            rotated = (attn.num_heads + attn.num_kv_heads) * attn.head_dim
            for param, dim in ((attn.qkv_weight, 1), (attn.qkv_bias, 0)):
                if param is not None:
                    set_weight_loader(param, _rotated_columns(get_weight_loader(param), dim, rotated, attn.head_dim,
                                                              perm))


def _rotated_columns(loader, dim: int, rotated: int, head_dim: int, perm) -> SafetensorsWeightLoader:
    """``loader``'s tensor with GLM-4's interleaved RoPE channels of its first ``rotated`` entries along ``dim`` (the Q
    and K heads) permuted into contiguous halves."""

    def transform(slices, rank):
        t = loader.load(slices, rank).movedim(dim, 0)
        t[:rotated] = rotary.permute_heads(t[:rotated], head_dim, perm)
        return t.movedim(0, dim)

    return SafetensorsWeightLoader(transform=transform)


class Glm4ForCausalLM(ModelHost):
    name = "GLM-4"
    decode_context_ladder = True

    def check_launch(self, config) -> None:
        check_bf16_launch(config, self.name)
        check_no_expert_parallel(self.name)
        glm4_config(config)

    def build_graph(self, phase: str, config) -> nn.Module:
        return Glm4Graph(glm4_config(config), phase)


EntryClass = [Glm4ForCausalLM]
