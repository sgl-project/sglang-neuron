# SPDX-License-Identifier: Apache-2.0
"""The dense transformer of a Llama checkpoint and its RoPE: the plain frequencies of Llama 3 and earlier, or the
``llama3`` rescaling Llama 3.1 and 3.2 train with (``rope_scaling`` in older configs, ``rope_parameters`` in current
transformers)."""

from __future__ import annotations

from dataclasses import dataclass

from sglang_neuron.models.transformer.config import TransformerConfig

NAME = "Llama"


@dataclass(frozen=True)
class Llama3Scaling:
    """The ``llama3`` RoPE parameters: frequencies with a wavelength above ``original_max_position_embeddings /
    low_freq_factor`` are divided by ``factor``, those below ``original_max_position_embeddings / high_freq_factor``
    are kept, and the band between is blended linearly (model.Llama3RotaryEmbedding)."""

    factor: float
    low_freq_factor: float
    high_freq_factor: float
    original_max_position_embeddings: int


def _rope_params(hf_config) -> dict:
    """The checkpoint's RoPE dict under either key: ``rope_parameters`` (current transformers) or ``rope_scaling``
    (configs written before it)."""
    return dict(getattr(hf_config, "rope_parameters", None) or getattr(hf_config, "rope_scaling", None) or {})


def rope_of(hf_config) -> tuple[float, Llama3Scaling | None]:
    """The RoPE base and, for a ``llama3`` checkpoint, its scaling. Any other scaling (yarn, linear, dynamic,
    longrope) is refused: the graphs rotate by plain or llama3-rescaled frequencies only."""
    params = _rope_params(hf_config)
    kind = params.get("rope_type") or params.get("type") or "default"
    theta = float(params.get("rope_theta", getattr(hf_config, "rope_theta", 10000.0)))
    if kind == "default":
        return theta, None
    if kind != "llama3":
        raise ValueError(f"{NAME}: a {kind!r} RoPE scaling is not served; the graphs rotate by plain or "
                         "llama3-rescaled frequencies")
    missing = [k for k in ("factor", "low_freq_factor", "high_freq_factor", "original_max_position_embeddings")
               if k not in params]
    if missing:
        raise ValueError(f"{NAME}: the llama3 RoPE scaling lacks {missing}")
    scaling = Llama3Scaling(factor=float(params["factor"]), low_freq_factor=float(params["low_freq_factor"]),
                            high_freq_factor=float(params["high_freq_factor"]),
                            original_max_position_embeddings=int(params["original_max_position_embeddings"]))
    if scaling.factor <= 0 or scaling.original_max_position_embeddings <= 0:
        raise ValueError(f"{NAME}: the llama3 RoPE scaling needs a positive factor and original context, not {scaling}")
    if scaling.high_freq_factor < scaling.low_freq_factor:
        raise ValueError(f"{NAME}: the llama3 RoPE scaling's high_freq_factor is below its low_freq_factor: {scaling}")
    return theta, scaling


def llama_config(hf_config) -> TransformerConfig:
    """The dense transformer of a Llama checkpoint, refusing what its graphs do not serve: a quantized checkpoint,
    sliding-window layers, attention or MLP biases, and a RoPE scaling other than llama3."""
    if getattr(hf_config, "quantization_config", None):
        raise ValueError(f"{NAME}: the graphs serve bf16 checkpoints; this one is quantized "
                         f"({hf_config.quantization_config})")
    if getattr(hf_config, "use_sliding_window", False) or "sliding_attention" in (getattr(hf_config, "layer_types",
                                                                                          None) or ()):
        raise ValueError(f"{NAME}: sliding-window attention layers are not served; the graphs attend the whole context")
    if getattr(hf_config, "attention_bias", False):
        raise ValueError(f"{NAME}: attention biases (attention_bias) are not served")
    if getattr(hf_config, "mlp_bias", False):
        raise ValueError(f"{NAME}: MLP biases (mlp_bias) are not served")
    theta, _ = rope_of(hf_config)
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
        rope_theta=theta,
        rotary_dim=head_dim,
        tie_word_embeddings=bool(getattr(hf_config, "tie_word_embeddings", False)),
    )
