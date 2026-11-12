# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3 model geometry for the prefill stage module, read from the HF checkpoint config.

The checkpoint ships a composite config (``model_type`` ``kimi_k3``) whose ``text_config`` holds the language model
(``model_type`` ``kimi_linear``). Either form is accepted. The prefill serves the language model only.

``read_geometry`` refuses a config whose semantics the prefill does not implement, naming every field that differs,
so a checkpoint with another router, activation or attention variant fails at load and never runs with the wrong
math. Layer lists are 0-indexed here; the checkpoint lists them 1-indexed.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Mapping

import torch

MLA_LORA_NORM_EPS = 1e-6
"""The eps of ``q_a_layernorm`` and ``kv_a_layernorm``: the checkpoint modeling code sets it, the config does not."""

TEXT_MODEL_TYPE = "kimi_linear"
COMPOSITE_MODEL_TYPE = "kimi_k3"
EXPERT_QUANT_FORMAT = "mxfp4-pack-quantized"
EXPERT_QUANT_GROUP = 32


@dataclass(frozen=True)
class KimiK3Geometry:
    """The model facts the prefill stage module reads. All layer indices are 0-indexed."""

    vocab_size: int
    hidden_size: int
    num_layers: int
    rms_norm_eps: float
    attn_res_block_size: int
    # MLA (gated, NoPE: the rope part of the cache row is stored raw)
    num_heads: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    # KDA
    kda_num_heads: int
    kda_head_dim: int
    kda_conv_kernel: int
    kda_gate_lower_bound: float
    # FFN
    num_experts: int
    top_k: int
    num_shared_experts: int
    moe_intermediate_size: int
    routed_expert_hidden_size: int
    intermediate_size: int
    first_k_dense_replace: int
    situ_beta: float
    situ_linear_beta: float
    # layer kinds
    mla_layers: tuple[int, ...]
    kda_layers: tuple[int, ...]
    # tokens
    pad_token_id: int | None
    bos_token_id: int | None
    eos_token_id: int | None
    max_position_embeddings: int
    expert_quant: str | None = "mxfp4"
    mla_lora_norm_eps: float = MLA_LORA_NORM_EPS
    dtype: torch.dtype = torch.bfloat16

    def is_mla(self, layer: int) -> bool:
        return layer in self._mla_set

    def is_kda(self, layer: int) -> bool:
        return not self.is_mla(layer)

    def is_moe(self, layer: int) -> bool:
        """Every layer from ``first_k_dense_replace`` on is a latent MoE; the earlier ones are dense MLPs."""
        return layer >= self.first_k_dense_replace

    def appends_bank(self, layer: int) -> bool:
        """The layer appends its input stream to the AttnRes bank (``layer % attn_res_block_size == 0``)."""
        return layer % self.attn_res_block_size == 0

    @property
    def mla_row_width(self) -> int:
        """Values of one MLA cache row: the normed latent, then the raw rope part (576 bytes in FP8)."""
        return self.kv_lora_rank + self.qk_rope_head_dim

    @property
    def _mla_set(self) -> frozenset:
        return frozenset(self.mla_layers)


def _load_config(source: Any) -> dict:
    """A config dict from a checkpoint directory, a ``config.json`` path, a dict, or a config object (``to_dict``)."""
    if isinstance(source, (str, os.PathLike)):
        path = os.fspath(source)
        if os.path.isdir(path):
            path = os.path.join(path, "config.json")
        with open(path) as f:
            return json.load(f)
    if isinstance(source, Mapping):
        return dict(source)
    to_dict = getattr(source, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    raise TypeError(f"read_geometry takes a checkpoint directory, a config.json path, a dict or a config object, "
                    f"not {type(source).__name__}")


def _text_config(config: dict) -> tuple[dict, dict]:
    """``(text config, top-level config)``: the composite config holds the language model in ``text_config``."""
    text = config.get("text_config")
    if text is None:
        return config, config
    if not isinstance(text, Mapping):
        text = _load_config(text)
    return dict(text), config


def _check(problems: list, name: str, value: Any, wanted: Any) -> None:
    if value != wanted:
        problems.append(f"{name} is {value!r}; the Kimi-K3 prefill implements {wanted!r}")


def _layer_list(problems: list, name: str, values: Any, num_layers: int) -> tuple[int, ...]:
    """A checkpoint 1-indexed layer list as sorted 0-indexed layers."""
    try:
        layers = sorted({int(v) - 1 for v in values})
    except (TypeError, ValueError):
        problems.append(f"linear_attn_config.{name} is {values!r}; give a list of 1-indexed layer numbers")
        return ()
    bad = [v + 1 for v in layers if not 0 <= v < num_layers]
    if bad:
        problems.append(f"linear_attn_config.{name} has layers {bad} outside 1..{num_layers}")
    return tuple(v for v in layers if 0 <= v < num_layers)


def _expert_quant(problems: list, text: dict, top: dict) -> str | None:
    quant = text.get("quantization_config", top.get("quantization_config"))
    if quant is None:
        problems.append(f"quantization_config is missing; the Kimi-K3 prefill loads the routed experts as "
                        f"{EXPERT_QUANT_FORMAT!r} (weight_packed), not unquantized experts")
        return None
    fmt = quant.get("format")
    if fmt != EXPERT_QUANT_FORMAT:
        problems.append(f"quantization_config.format is {fmt!r}; the Kimi-K3 prefill implements "
                        f"{EXPERT_QUANT_FORMAT!r} routed experts")
        return None
    for group_name, group in (quant.get("config_groups") or {}).items():
        weights = group.get("weights") or {}
        if weights.get("num_bits") != 4 or weights.get("group_size") != EXPERT_QUANT_GROUP or \
                weights.get("type") != "float":
            problems.append(f"quantization_config.config_groups.{group_name}.weights is {weights!r}; the Kimi-K3 "
                            f"prefill implements 4-bit float groups of {EXPERT_QUANT_GROUP}")
    return "mxfp4"


def read_geometry(source: Any) -> KimiK3Geometry:
    """Read and check the Kimi-K3 geometry of a checkpoint config.

    Args:
        source: The checkpoint directory, its ``config.json`` path, the config dict, or a config object with
            ``to_dict`` (the composite or the text config).

    Returns:
        The geometry.

    Raises:
        ValueError: The config has semantics the prefill does not implement; the message names every such field.
    """
    text, top = _text_config(_load_config(source))
    problems: list[str] = []
    model_type = top.get("model_type")
    if model_type not in (COMPOSITE_MODEL_TYPE, TEXT_MODEL_TYPE):
        problems.append(f"model_type is {model_type!r}; the Kimi-K3 prefill reads a {COMPOSITE_MODEL_TYPE!r} or "
                        f"{TEXT_MODEL_TYPE!r} config")
    if top is not text:
        _check(problems, "text_config.model_type", text.get("model_type"), TEXT_MODEL_TYPE)

    missing = [k for k in ("vocab_size", "hidden_size", "num_hidden_layers", "num_attention_heads", "q_lora_rank",
                           "kv_lora_rank", "qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim", "num_experts",
                           "num_experts_per_token", "num_shared_experts", "moe_intermediate_size",
                           "routed_expert_hidden_size", "intermediate_size", "attn_res_block_size",
                           "linear_attn_config") if text.get(k) is None]
    if missing:
        raise ValueError(f"The Kimi-K3 config has no {missing}; the prefill needs every one of them")

    num_layers = int(text["num_hidden_layers"])
    _check(problems, "hidden_act", text.get("hidden_act"), "situ")
    _check(problems, "mla_use_nope", bool(text.get("mla_use_nope", False)), True)
    _check(problems, "mla_use_output_gate", bool(text.get("mla_use_output_gate", False)), True)
    _check(problems, "num_key_value_heads", text.get("num_key_value_heads", text["num_attention_heads"]),
           text["num_attention_heads"])
    _check(problems, "topk_method", text.get("topk_method"), "noaux_tc")
    _check(problems, "moe_router_activation_func", text.get("moe_router_activation_func"), "sigmoid")
    _check(problems, "moe_renormalize", bool(text.get("moe_renormalize", False)), True)
    _check(problems, "routed_scaling_factor", float(text.get("routed_scaling_factor", 1.0)), 1.0)
    _check(problems, "num_expert_group", int(text.get("num_expert_group", 1)), 1)
    _check(problems, "topk_group", int(text.get("topk_group", 1)), 1)
    _check(problems, "moe_layer_freq", int(text.get("moe_layer_freq", 1)), 1)
    _check(problems, "latent_moe_use_norm", bool(text.get("latent_moe_use_norm", False)), True)
    _check(problems, "num_nextn_predict_layers", int(text.get("num_nextn_predict_layers", 0) or 0), 0)
    _check(problems, "tie_word_embeddings", bool(top.get("tie_word_embeddings", text.get("tie_word_embeddings",
                                                                                          False))), False)

    lin = dict(text["linear_attn_config"])
    _check(problems, "linear_attn_config.use_full_rank_gate", bool(lin.get("use_full_rank_gate", False)), True)
    _check(problems, "linear_attn_config.short_conv_kernel_size", lin.get("short_conv_kernel_size"), 4)
    mla_layers = _layer_list(problems, "full_attn_layers", lin.get("full_attn_layers", ()), num_layers)
    kda_layers = tuple(i for i in range(num_layers) if i not in set(mla_layers))
    if lin.get("kda_layers") is not None:
        listed = _layer_list(problems, "kda_layers", lin["kda_layers"], num_layers)
        if listed != kda_layers:
            problems.append("linear_attn_config.kda_layers and full_attn_layers do not partition the layers: every "
                            "layer must be in exactly one of them")
    if not mla_layers:
        problems.append("linear_attn_config.full_attn_layers is empty; the Kimi-K3 prefill needs MLA layers")
    # the math constants the graphs take from the config: a default in their place would change the numerics unseen
    unstated = [name for name, value in (("rms_norm_eps", text.get("rms_norm_eps")),
                                         ("activation_situ_beta", text.get("activation_situ_beta")),
                                         ("activation_situ_linear_beta", text.get("activation_situ_linear_beta")),
                                         ("linear_attn_config.gate_lower_bound", lin.get("gate_lower_bound")))
                if value is None]
    if unstated:
        problems.append(f"the config states no {', '.join(unstated)}; the Kimi-K3 prefill takes each from the "
                        "checkpoint config")
    expert_quant = _expert_quant(problems, text, top)
    if problems:
        raise ValueError("The checkpoint config is not the Kimi-K3 the prefill implements:\n  " + "\n  ".join(problems))

    return KimiK3Geometry(
        vocab_size=int(text["vocab_size"]),
        hidden_size=int(text["hidden_size"]),
        num_layers=num_layers,
        rms_norm_eps=float(text["rms_norm_eps"]),
        attn_res_block_size=int(text["attn_res_block_size"]),
        num_heads=int(text["num_attention_heads"]),
        q_lora_rank=int(text["q_lora_rank"]),
        kv_lora_rank=int(text["kv_lora_rank"]),
        qk_nope_head_dim=int(text["qk_nope_head_dim"]),
        qk_rope_head_dim=int(text["qk_rope_head_dim"]),
        v_head_dim=int(text["v_head_dim"]),
        kda_num_heads=int(lin.get("num_heads", text["num_attention_heads"])),
        kda_head_dim=int(lin.get("head_dim", 128)),
        kda_conv_kernel=int(lin["short_conv_kernel_size"]),
        kda_gate_lower_bound=float(lin["gate_lower_bound"]),
        num_experts=int(text["num_experts"]),
        top_k=int(text["num_experts_per_token"]),
        num_shared_experts=int(text["num_shared_experts"]),
        moe_intermediate_size=int(text["moe_intermediate_size"]),
        routed_expert_hidden_size=int(text["routed_expert_hidden_size"]),
        intermediate_size=int(text["intermediate_size"]),
        first_k_dense_replace=int(text.get("first_k_dense_replace", 0)),
        situ_beta=float(text["activation_situ_beta"]),
        situ_linear_beta=float(text["activation_situ_linear_beta"]),
        mla_layers=mla_layers,
        kda_layers=kda_layers,
        pad_token_id=text.get("pad_token_id", top.get("pad_token_id")),
        bos_token_id=text.get("bos_token_id", top.get("bos_token_id")),
        eos_token_id=text.get("eos_token_id", top.get("eos_token_id")),
        max_position_embeddings=int(text.get("max_position_embeddings", 0)),
        expert_quant=expert_quant,
    )
