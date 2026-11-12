# SPDX-License-Identifier: Apache-2.0
"""Qwen3ForCausalLM: the host SGLang builds for a Qwen3 dense checkpoint (models/host.py), on the shared dense
transformer (models/transformer)."""

from torch import nn

from sglang_neuron.models.host import ModelHost
from sglang_neuron.models.transformer.config import (TransformerConfig, check_bf16_launch, check_no_expert_parallel,
                                                     rope_theta_of)
from sglang_neuron.models.transformer.model import TransformerForCausalLM


def qwen3_config(hf_config) -> TransformerConfig:
    """The dense transformer of a Qwen3 checkpoint, refusing what its graphs do not serve: a quantized checkpoint,
    sliding-window layers, attention biases (Qwen3 puts one on o_proj too) and a scaled RoPE."""
    if getattr(hf_config, "quantization_config", None):
        raise ValueError("Qwen3: the graphs serve bf16 checkpoints; this one is quantized "
                         f"({hf_config.quantization_config})")
    if getattr(hf_config, "use_sliding_window", False) or "sliding_attention" in (getattr(hf_config, "layer_types",
                                                                                          None) or ()):
        raise ValueError("Qwen3: sliding-window attention layers are not served; the graphs attend the whole context")
    if getattr(hf_config, "attention_bias", False):
        raise ValueError("Qwen3: attention biases (attention_bias) are not served")
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
        rope_theta=rope_theta_of(hf_config, "Qwen3"),
        rotary_dim=head_dim,
        tie_word_embeddings=bool(hf_config.tie_word_embeddings),
        qk_norm=True,
    )


class Qwen3ForCausalLM(ModelHost):
    name = "Qwen3"
    decode_context_ladder = True

    def check_launch(self, config) -> None:
        check_bf16_launch(config, self.name)
        check_no_expert_parallel(self.name)
        qwen3_config(config)

    def build_graph(self, phase: str, config) -> nn.Module:
        return TransformerForCausalLM(qwen3_config(config), phase)


EntryClass = [Qwen3ForCausalLM]
