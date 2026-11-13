# SPDX-License-Identifier: Apache-2.0
"""The shape of a Gemma 4 dense text model, read from a checkpoint's (text) config.

Gemma 4 has two layer kinds. Sliding layers: head_dim 256, 16 KV heads, a 1024-token window, RoPE theta 1e4.
Global layers: head_dim 512 (``global_head_dim``), 4 KV heads (``num_global_key_value_heads``), K reused as V
(``attention_k_eq_v``, no v_proj), a proportional RoPE over 25% of the head with theta 1e6. Some transformers releases
drop the two global fields when parsing (transformers 5.19 turns them into ``per_layer_config``), so they are read
back from the checkpoint's config.json."""

import json
import os
from dataclasses import dataclass

import torch

DTYPE = torch.bfloat16
FULL = "full_attention"
SLIDING = "sliding_attention"


def text_config(hf_config):
    """The language model's config: ``text_config`` of a multimodal config, else the config itself."""
    return getattr(hf_config, "text_config", None) or hf_config


def _raw_fields(hf_config) -> dict:
    """config.json as written (its text_config merged over the top level). transformers 5.19 refuses reads of
    per-layer attributes such as ``head_dim`` on a heterogeneous config, so the shape is read from the file."""
    path = getattr(hf_config, "_name_or_path", None)
    if not path or not os.path.isfile(os.path.join(path, "config.json")):
        return {}
    with open(os.path.join(path, "config.json")) as f:
        raw = json.load(f)
    return {**raw, **(raw.get("text_config") or {})}


@dataclass(frozen=True)
class Gemma4Config:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    global_head_dim: int
    num_global_key_value_heads: int
    attention_k_eq_v: bool
    rms_norm_eps: float
    sliding_window: int
    final_logit_softcapping: float | None
    layer_types: tuple
    sliding_rope_theta: float
    global_rope_theta: float
    global_partial_rotary_factor: float
    tie_word_embeddings: bool

    def is_global(self, i: int) -> bool:
        return self.layer_types[i] == FULL

    def layer_head_dim(self, i: int) -> int:
        return self.global_head_dim if self.is_global(i) else self.head_dim

    def layer_kv_heads(self, i: int) -> int:
        return self.num_global_key_value_heads if self.is_global(i) else self.num_key_value_heads


def gemma4_config(hf_config) -> Gemma4Config:
    """The model's shape, refusing what the graphs do not serve: a quantized checkpoint, per-layer input embeddings,
    shared KV layers, MoE blocks and a layer kind other than sliding or full attention."""
    cfg = text_config(hf_config)
    raw = {**_raw_fields(cfg), **_raw_fields(hf_config)}

    def get(name, default=None):
        if raw.get(name) is not None:
            return raw[name]
        try:
            value = getattr(cfg, name, None)
        except Exception:  # transformers' refusal of a per-layer attribute
            value = None
        return default if value is None else value

    if getattr(hf_config, "quantization_config", None) or getattr(cfg, "quantization_config", None):
        raise ValueError("Gemma 4: the graphs serve bf16 checkpoints; this one is quantized")
    if get("hidden_size_per_layer_input", 0):
        raise ValueError("Gemma 4: per-layer input embeddings (hidden_size_per_layer_input) are not served")
    if get("num_kv_shared_layers", 0):
        raise ValueError("Gemma 4: shared KV layers (num_kv_shared_layers) are not served")
    if get("enable_moe_block", False):
        raise ValueError("Gemma 4: MoE blocks are not served")
    n = int(get("num_hidden_layers"))
    layer_types = tuple(get("layer_types") or ())[:n]
    if len(layer_types) != n or any(t not in (FULL, SLIDING) for t in layer_types):
        raise ValueError(f"Gemma 4: layer_types must name {n} sliding or full attention layers, got {layer_types}")
    rope = get("rope_parameters") or {}
    full_rope, sliding_rope = rope.get(FULL, {}), rope.get(SLIDING, {})
    if sliding_rope.get("rope_type", "default") != "default":
        raise ValueError(f"Gemma 4: sliding RoPE type {sliding_rope.get('rope_type')!r} is not served")
    if full_rope.get("rope_type", "proportional") not in ("proportional", "default"):
        raise ValueError(f"Gemma 4: global RoPE type {full_rope.get('rope_type')!r} is not served")
    if get("use_double_wide_mlp", False):
        raise ValueError("Gemma 4: use_double_wide_mlp is not served")
    head_dim = int(get("head_dim"))
    global_head_dim, global_kv = get("global_head_dim"), get("num_global_key_value_heads")
    if global_head_dim is None:
        # transformers 5.19 saves the global layers' geometry as per-layer overrides
        overrides = {json.dumps(v, sort_keys=True) for k, v in (raw.get("per_layer_config") or {}).items()
                     if layer_types[int(k)] == FULL} if raw.get("per_layer_config") else set()
        if len(overrides) != 1:
            raise ValueError("Gemma 4: no global_head_dim, and per_layer_config does not give one geometry for every "
                             f"global layer ({overrides})")
        override = json.loads(overrides.pop())
        global_head_dim, global_kv = override.get("head_dim", head_dim), override.get("num_key_value_heads")
    return Gemma4Config(
        vocab_size=int(get("vocab_size")),
        hidden_size=int(get("hidden_size")),
        intermediate_size=int(get("intermediate_size")),
        num_hidden_layers=n,
        num_attention_heads=int(get("num_attention_heads")),
        num_key_value_heads=int(get("num_key_value_heads")),
        head_dim=head_dim,
        global_head_dim=int(global_head_dim),
        num_global_key_value_heads=int(global_kv or get("num_key_value_heads")),
        attention_k_eq_v=bool(get("attention_k_eq_v", False)),
        rms_norm_eps=float(get("rms_norm_eps", 1e-6)),
        sliding_window=int(get("sliding_window")),
        final_logit_softcapping=get("final_logit_softcapping"),
        layer_types=layer_types,
        sliding_rope_theta=float(sliding_rope.get("rope_theta", 10000.0)),
        global_rope_theta=float(full_rope.get("rope_theta", 1000000.0)),
        global_partial_rotary_factor=float(full_rope.get("partial_rotary_factor", 1.0)),
        tie_word_embeddings=bool(get("tie_word_embeddings", True)),
    )
