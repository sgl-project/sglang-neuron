# SPDX-License-Identifier: Apache-2.0
"""The shape of a MiMo-V2 checkpoint as the graphs serve it, and what they refuse.

MiMo-V2 attention differs per layer kind (``hybrid_layer_pattern``: 0 full, 1 sliding):

| | full (9 layers of V2.6-Flash) | sliding (39 layers) |
| --- | --- | --- |
| query / KV heads | 64 / 4 | 64 / 8 (``swa_*``) |
| Q/K, V head dim | 192, 128 | 192, 128 |
| RoPE base | ``rope_theta`` 1e7 | ``swa_rope_theta`` 1e4 |
| window, sink | none, ``add_full_attention_sink_bias`` | 128, ``add_swa_attention_sink_bias`` |

Both kinds rotate the first ``int(head_dim * partial_rotary_factor)`` = 64 channels as two contiguous halves
(``rotate_half``) and scale V by ``attention_value_scale`` before it is cached; the softmax scale is
``head_dim ** -0.5`` of the Q/K width.

The Neuron attention kernels take one head dim for Q, K and V, and above 128 only whole multiples of 128
(nkilib attention_tkg: ``d_head % 128 == 0 or d_head <= 128``). The graphs therefore run every head at
``KV_HEAD_DIM`` = 256: Q and K are their 192 channels re-ordered for the kernels' RoPE and zero-padded, V its 128
channels zero-padded (``weights.head_layout``). A zero channel adds nothing to a dot product, so the attention is
exact, and the output projection reads the first 128 channels of each head. The KV pool is sized for the same 256
(``sglang_patches/mimo_v2_launch.py``)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

NAME = "MiMo-V2"
ARCHITECTURES = ("MiMoV2ForCausalLM", "MiMoV2FlashForCausalLM")

# the head dim every attention kernel and the KV pool run at (module docstring)
KV_HEAD_DIM = 256


def _get(cfg, name: str, default=None):
    """``cfg.name`` of a config object or dict, ``default`` where it is missing or None."""
    value = cfg.get(name) if isinstance(cfg, dict) else getattr(cfg, name, None)
    return default if value is None else value


@dataclass(frozen=True)
class AttentionKind:
    """One layer kind's attention geometry."""

    num_heads: int
    num_kv_heads: int
    head_dim: int
    v_head_dim: int
    rope_theta: float
    sliding_window: Optional[int]
    sink: bool

    @property
    def scaling(self) -> float:
        return self.head_dim ** -0.5


@dataclass(frozen=True)
class MiMoV2Config:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    rms_norm_eps: float
    full: AttentionKind
    swa: AttentionKind
    rotary_dim: int
    v_scale: float
    # per layer: True for a sliding-window layer, True for an MoE layer
    swa_layers: tuple
    moe_layers: tuple
    num_experts: int
    top_k: int
    moe_intermediate_size: int
    norm_topk_prob: bool
    routed_scaling_factor: float
    tie_word_embeddings: bool
    # the checkpoint's quantization: "fp8" (128 x 128 blocks) or "" for bf16
    quant_method: str
    num_nextn_predict_layers: int
    max_position_embeddings: int

    def kind(self, layer_idx: int) -> AttentionKind:
        return self.swa if self.swa_layers[layer_idx] else self.full

    def layer_name(self, layer_idx: int) -> str:
        return f"layers.{layer_idx}.self_attn"

    @property
    def sliding_windows(self) -> list:
        return [self.swa.sliding_window if s else None for s in self.swa_layers]


def mimo_v2_config(hf_config) -> MiMoV2Config:
    """The graphs' config of a MiMo-V2 checkpoint (a PretrainedConfig or its dict), refusing what they do not serve."""
    problems = config_problems(hf_config)
    if problems:
        raise ValueError(f"{NAME}: " + "; ".join(problems))
    n = int(_get(hf_config, "num_hidden_layers"))
    pattern = list(_get(hf_config, "hybrid_layer_pattern"))[:n]
    freq = list(_get(hf_config, "moe_layer_freq", [1] * n))[:n]
    head_dim = int(_get(hf_config, "head_dim"))
    v_head_dim = int(_get(hf_config, "v_head_dim", head_dim))
    partial = float(_get(_get(hf_config, "rope_parameters", {}) or {}, "partial_rotary_factor",
                         _get(hf_config, "partial_rotary_factor", 1.0)))
    rope = _get(hf_config, "rope_parameters", {}) or {}
    rope_theta = float(_get(rope, "rope_theta", _get(hf_config, "rope_theta", 10000.0)))
    full = AttentionKind(
        num_heads=int(_get(hf_config, "num_attention_heads")),
        num_kv_heads=int(_get(hf_config, "num_key_value_heads")),
        head_dim=head_dim, v_head_dim=v_head_dim, rope_theta=rope_theta, sliding_window=None,
        sink=bool(_get(hf_config, "add_full_attention_sink_bias", False)))
    swa = AttentionKind(
        num_heads=int(_get(hf_config, "swa_num_attention_heads", full.num_heads)),
        num_kv_heads=int(_get(hf_config, "swa_num_key_value_heads", full.num_kv_heads)),
        head_dim=int(_get(hf_config, "swa_head_dim", head_dim)),
        v_head_dim=int(_get(hf_config, "swa_v_head_dim", v_head_dim)),
        rope_theta=float(_get(hf_config, "swa_rope_theta", rope_theta)),
        sliding_window=int(_get(hf_config, "sliding_window", _get(hf_config, "sliding_window_size", 0))),
        sink=bool(_get(hf_config, "add_swa_attention_sink_bias", False)))
    quant = _quantization(hf_config)
    return MiMoV2Config(
        vocab_size=int(_get(hf_config, "vocab_size")),
        hidden_size=int(_get(hf_config, "hidden_size")),
        intermediate_size=int(_get(hf_config, "intermediate_size")),
        num_hidden_layers=n,
        rms_norm_eps=float(_get(hf_config, "layernorm_epsilon", _get(hf_config, "rms_norm_eps", 1e-6))),
        full=full, swa=swa,
        rotary_dim=int(head_dim * partial),
        v_scale=float(_get(hf_config, "attention_value_scale", 1.0)),
        swa_layers=tuple(int(p) == 1 for p in pattern),
        moe_layers=tuple(bool(int(f)) for f in freq),
        num_experts=int(_get(hf_config, "n_routed_experts", 0)),
        top_k=int(_get(hf_config, "num_experts_per_tok", 0)),
        moe_intermediate_size=int(_get(hf_config, "moe_intermediate_size", 0)),
        norm_topk_prob=bool(_get(hf_config, "norm_topk_prob", True)),
        routed_scaling_factor=float(_get(hf_config, "routed_scaling_factor", 1.0)),
        tie_word_embeddings=bool(_get(hf_config, "tie_word_embeddings", False)),
        quant_method=str(quant.get("quant_method") or ""),
        num_nextn_predict_layers=int(_get(hf_config, "num_nextn_predict_layers", 0)),
        max_position_embeddings=int(_get(hf_config, "max_position_embeddings", 0)),
    )


def _quantization(hf_config) -> dict:
    quant = _get(hf_config, "quantization_config", {}) or {}
    return quant if isinstance(quant, dict) else quant.to_dict()


def config_problems(hf_config) -> list:
    """Everything in the checkpoint's config the graphs do not serve, each naming the field."""
    problems = []
    n = int(_get(hf_config, "num_hidden_layers", 0))
    pattern = _get(hf_config, "hybrid_layer_pattern")
    if not pattern or len(pattern) < n or any(int(p) not in (0, 1) for p in pattern[:n]):
        problems.append("hybrid_layer_pattern must give each layer 0 (full) or 1 (sliding window)")
    freq = _get(hf_config, "moe_layer_freq")
    if freq is not None and len(freq) < n:
        problems.append(f"moe_layer_freq has {len(freq)} entries for {n} layers")
    if _get(hf_config, "scoring_func", "sigmoid") != "sigmoid":
        problems.append(f"scoring_func {_get(hf_config, 'scoring_func')!r}: the router is served as a sigmoid")
    if _get(hf_config, "topk_method", "noaux_tc") != "noaux_tc":
        problems.append(f"topk_method {_get(hf_config, 'topk_method')!r}: the router is served as noaux_tc")
    if int(_get(hf_config, "n_group", 1)) != 1 or int(_get(hf_config, "topk_group", 1)) != 1:
        problems.append("grouped routing (n_group, topk_group above 1) is not served")
    if _get(hf_config, "n_shared_experts", None):
        problems.append("shared experts (n_shared_experts) are not served")
    layout = _get(hf_config, "attention_projection_layout", "split")
    if layout != "fused_qkv":
        problems.append(f"attention_projection_layout {layout!r}: the loaders read the fused qkv_proj (fused_qkv)")
    if _get(hf_config, "attention_bias", False):
        problems.append("attention biases (attention_bias) are not served")
    if _get(hf_config, "hidden_act", "silu") != "silu":
        problems.append(f"hidden_act {_get(hf_config, 'hidden_act')!r}: the MLPs are served SiLU-gated")
    rope = _get(hf_config, "rope_parameters", {}) or {}
    scaling = _get(hf_config, "rope_scaling", {}) or {}
    kind = _get(rope, "rope_type", None) or _get(scaling, "rope_type", None) or _get(scaling, "type", None) \
        or "default"
    if kind != "default":
        problems.append(f"a {kind!r} RoPE scaling is not served; the graphs rotate by plain frequencies")
    head_dim = int(_get(hf_config, "head_dim", 0))
    for prefix in ("", "swa_"):
        hd = int(_get(hf_config, f"{prefix}head_dim", head_dim))
        vhd = int(_get(hf_config, f"{prefix}v_head_dim", _get(hf_config, "v_head_dim", hd)))
        if hd > KV_HEAD_DIM or vhd > KV_HEAD_DIM:
            problems.append(f"{prefix}head_dim {hd} / {prefix}v_head_dim {vhd}: the graphs pad heads to "
                            f"{KV_HEAD_DIM} channels")
    partial = float(_get(rope, "partial_rotary_factor", _get(hf_config, "partial_rotary_factor", 1.0)))
    rotary = int(head_dim * partial)
    if rotary % 2 or not 0 < rotary <= head_dim:
        problems.append(f"a rotary dimension of {rotary} (head_dim x partial_rotary_factor) is not even and within "
                        "the head")
    swa_hd = int(_get(hf_config, "swa_head_dim", head_dim))
    if swa_hd != head_dim:
        problems.append(f"swa_head_dim {swa_hd} != head_dim {head_dim}: both kinds share one RoPE layout")
    quant = _quantization(hf_config)
    if quant:
        if quant.get("quant_method") != "fp8":
            problems.append(f"quantization {quant.get('quant_method')!r}: the graphs read bf16, or the FP8 "
                            "(128 x 128 blocks) checkpoint with bf16 or MXFP4 experts")
        elif list(quant.get("weight_block_size") or [128, 128]) != [128, 128]:
            problems.append(f"weight_block_size {quant.get('weight_block_size')}: the loaders read 128 x 128 FP8 "
                            "blocks")
        store = str(quant.get("store_dtype") or "").lower()
        if store not in ("", "fp8", "mxfp4"):
            problems.append(f"store_dtype {store!r}: the experts are read as FP8 blocks or MXFP4")
        if store == "mxfp4" and int(quant.get("mxfp4_block_size", 32)) != 32:
            problems.append(f"mxfp4_block_size {quant.get('mxfp4_block_size')}: MXFP4 blocks are 32 values")
    return problems
