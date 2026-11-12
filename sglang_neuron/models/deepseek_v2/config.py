# SPDX-License-Identifier: Apache-2.0
"""The shape of a DeepSeek-V2 checkpoint the graphs serve, and its YaRN rotary frequencies."""

import math
from dataclasses import dataclass

import torch

# the graphs compute in bfloat16
DTYPE = torch.bfloat16


@dataclass(frozen=True)
class DeepseekV2Config:
    vocab_size: int
    hidden_size: int
    # the dense MLP of the first first_k_dense_replace layers, and each expert's
    intermediate_size: int
    moe_intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    kv_lora_rank: int
    n_routed_experts: int
    num_experts_per_tok: int
    n_shared_experts: int
    first_k_dense_replace: int
    routed_scaling_factor: float
    rms_norm_eps: float
    rope_theta: float
    # YaRN: the context factor, the pre-training context, the ramp's rotation bounds and the attention-scale exponents
    rope_factor: float = 1.0
    rope_original_context: int = 4096
    rope_beta_fast: float = 32.0
    rope_beta_slow: float = 1.0
    rope_mscale: float = 1.0
    rope_mscale_all_dim: float = 0.0
    tie_word_embeddings: bool = False

    @property
    def q_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    @property
    def compact_dim(self) -> int:
        """A cached row: the latent, then the rotated RoPE key."""
        return self.kv_lora_rank + self.qk_rope_head_dim

    def softmax_scale(self) -> float:
        # DeepSeek's modeling code scales the head's attention by the squared YaRN mscale of mscale_all_dim, which
        # transformers' implementation of the model leaves out
        return self.q_head_dim ** -0.5 * _yarn_mscale(self.rope_factor, self.rope_mscale_all_dim) ** 2

    def rope_inv_freq_and_scale(self) -> tuple[torch.Tensor, float]:
        """The rotary inverse frequencies over the RoPE dimension, blended by YaRN's ramp between the extrapolated and
        the interpolated ones, and the factor the cos and sin are scaled by."""
        dim, base = self.qk_rope_head_dim, self.rope_theta
        extrapolated = 1.0 / base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        if self.rope_factor <= 1.0:
            return extrapolated, 1.0

        def correction(rotations):
            return dim * math.log(self.rope_original_context / (rotations * 2 * math.pi)) / (2 * math.log(base))

        low = max(math.floor(correction(self.rope_beta_fast)), 0)
        high = min(math.ceil(correction(self.rope_beta_slow)), dim - 1)
        ramp = torch.clamp((torch.arange(dim // 2, dtype=torch.float32) - low) / max(high - low, 0.001), 0, 1)
        inv_freq = extrapolated / self.rope_factor * ramp + extrapolated * (1 - ramp)
        scale = (_yarn_mscale(self.rope_factor, self.rope_mscale)
                 / _yarn_mscale(self.rope_factor, self.rope_mscale_all_dim))
        return inv_freq, scale


def _yarn_mscale(factor: float, mscale: float) -> float:
    return 1.0 if factor <= 1 else 0.1 * mscale * math.log(factor) + 1.0


def deepseek_v2_config(hf_config) -> DeepseekV2Config:
    """The served shape of a DeepSeek-V2 checkpoint, refusing what the graphs do not compute: a quantized checkpoint, a
    LoRA query, attention biases, group-limited or sigmoid routing, renormalized top-k weights and a RoPE scaling other
    than YaRN."""
    unserved = {
        "a quantized checkpoint": bool(getattr(hf_config, "quantization_config", None)),
        "a low-rank query (q_lora_rank)": getattr(hf_config, "q_lora_rank", None) is not None,
        "attention biases": bool(getattr(hf_config, "attention_bias", False)),
        "routing other than greedy top-k": (getattr(hf_config, "topk_method", None) or "greedy") != "greedy",
        "a scoring function other than softmax": (getattr(hf_config, "scoring_func", None) or "softmax") != "softmax",
        "renormalized top-k weights (norm_topk_prob)": bool(getattr(hf_config, "norm_topk_prob", False)),
    }
    scaling = getattr(hf_config, "rope_scaling", None) or {}
    kind = scaling.get("rope_type") or scaling.get("type") or "default"
    unserved[f"a {kind!r} RoPE scaling"] = kind not in ("default", "yarn")
    asked = [name for name, on in unserved.items() if on]
    if asked:
        raise ValueError(f"DeepSeek-V2: the graphs do not serve {', '.join(asked)}")
    params = getattr(hf_config, "rope_parameters", None) or {}
    yarn = {}
    if kind == "yarn":
        yarn = dict(rope_factor=float(scaling["factor"]),
                    rope_original_context=int(scaling.get("original_max_position_embeddings", 4096)),
                    rope_beta_fast=float(scaling.get("beta_fast", 32)),
                    rope_beta_slow=float(scaling.get("beta_slow", 1)),
                    rope_mscale=float(scaling.get("mscale", 1)),
                    rope_mscale_all_dim=float(scaling.get("mscale_all_dim", 0)))
    return DeepseekV2Config(
        vocab_size=hf_config.vocab_size,
        hidden_size=hf_config.hidden_size,
        intermediate_size=hf_config.intermediate_size,
        moe_intermediate_size=hf_config.moe_intermediate_size,
        num_hidden_layers=hf_config.num_hidden_layers,
        num_attention_heads=hf_config.num_attention_heads,
        qk_nope_head_dim=hf_config.qk_nope_head_dim,
        qk_rope_head_dim=hf_config.qk_rope_head_dim,
        v_head_dim=hf_config.v_head_dim,
        kv_lora_rank=hf_config.kv_lora_rank,
        n_routed_experts=hf_config.n_routed_experts,
        num_experts_per_tok=hf_config.num_experts_per_tok,
        n_shared_experts=hf_config.n_shared_experts or 0,
        first_k_dense_replace=hf_config.first_k_dense_replace,
        routed_scaling_factor=float(getattr(hf_config, "routed_scaling_factor", 1.0)),
        rms_norm_eps=hf_config.rms_norm_eps,
        rope_theta=float(params.get("rope_theta", getattr(hf_config, "rope_theta", 10000.0))),
        tie_word_embeddings=bool(getattr(hf_config, "tie_word_embeddings", False)),
        **yarn,
    )
