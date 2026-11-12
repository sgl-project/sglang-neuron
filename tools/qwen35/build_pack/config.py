# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5-397B-A17B text-model dimensions, read from the checkpoint's config.json.

The multimodal checkpoint nests the text model under ``text_config``; RoPE
parameters live either at top level or under ``rope_parameters`` /
``rope_scaling``. Defaults are the 397B values the kernels are built for.
"""

from __future__ import annotations

import dataclasses
import json
import os
from typing import Any, Optional


@dataclasses.dataclass(frozen=True)
class Qwen35Config:
    hidden: int = 4096
    n_layers: int = 60
    layer_types: tuple = ()
    n_q_heads: int = 32
    n_kv_heads: int = 2
    head_dim: int = 256
    gdn_k_heads: int = 16
    gdn_v_heads: int = 64
    gdn_k_dim: int = 128
    gdn_v_dim: int = 128
    conv_kernel: int = 4
    n_experts: int = 512
    top_k: int = 10
    moe_inter: int = 1024
    shared_inter: int = 1024
    vocab: int = 248320
    rope_theta: float = 1e7
    partial_rotary: float = 0.25
    eps: float = 1e-6

    def __post_init__(self):
        if not self.layer_types:
            object.__setattr__(
                self,
                "layer_types",
                tuple("full_attention" if i % 4 == 3 else "linear_attention" for i in range(self.n_layers)),
            )
        if len(self.layer_types) != self.n_layers:
            raise ValueError(f"{len(self.layer_types)} layer types for {self.n_layers} layers")

    # -- derived -------------------------------------------------------------
    @property
    def attn_layers(self) -> tuple:
        return tuple(i for i, t in enumerate(self.layer_types) if t == "full_attention")

    @property
    def gdn_layers(self) -> tuple:
        return tuple(i for i, t in enumerate(self.layer_types) if t == "linear_attention")

    @property
    def rot_dim(self) -> int:
        return int(self.head_dim * self.partial_rotary)

    @property
    def key_dim(self) -> int:
        return self.gdn_k_heads * self.gdn_k_dim

    @property
    def value_dim(self) -> int:
        return self.gdn_v_heads * self.gdn_v_dim

    @property
    def gqa(self) -> int:
        return self.gdn_v_heads // self.gdn_k_heads

    def kind_index(self, layer: int) -> tuple:
        """(is_attention, index within its kind) of a global layer id."""
        if self.layer_types[layer] == "full_attention":
            return True, self.attn_layers.index(layer)
        return False, self.gdn_layers.index(layer)

    def check_kernel_pattern(self) -> None:
        """The kernels hard-code 'layer i % 4 == 3 is attention'."""
        want = tuple(i for i in range(self.n_layers) if i % 4 == 3)
        if self.attn_layers != want:
            raise ValueError(f"attention layers {self.attn_layers} are not every 4th layer")

    # -- construction --------------------------------------------------------
    @classmethod
    def from_dict(cls, d: dict) -> "Qwen35Config":
        tc = d.get("text_config", d)
        rope = tc.get("rope_parameters") or tc.get("rope_scaling") or {}

        def get(key: str, default: Any, *alts: str) -> Any:
            for k in (key,) + alts:
                if k in tc and tc[k] is not None:
                    return tc[k]
            return default

        n_layers = int(get("num_hidden_layers", 60))
        theta = rope.get("rope_theta", get("rope_theta", 1e7))
        partial = rope.get("partial_rotary_factor", get("partial_rotary_factor", 0.25))
        return cls(
            hidden=int(get("hidden_size", 4096)),
            n_layers=n_layers,
            layer_types=tuple(get("layer_types", ()) or ()),
            n_q_heads=int(get("num_attention_heads", 32)),
            n_kv_heads=int(get("num_key_value_heads", 2)),
            head_dim=int(get("head_dim", 256)),
            gdn_k_heads=int(get("linear_num_key_heads", 16)),
            gdn_v_heads=int(get("linear_num_value_heads", 64)),
            gdn_k_dim=int(get("linear_key_head_dim", 128)),
            gdn_v_dim=int(get("linear_value_head_dim", 128)),
            conv_kernel=int(get("linear_conv_kernel_dim", 4)),
            n_experts=int(get("num_experts", 512)),
            top_k=int(get("num_experts_per_tok", 10)),
            moe_inter=int(get("moe_intermediate_size", 1024)),
            shared_inter=int(get("shared_expert_intermediate_size", 1024)),
            vocab=int(get("vocab_size", 248320)),
            rope_theta=float(theta),
            partial_rotary=float(partial),
            eps=float(get("rms_norm_eps", 1e-6)),
        )

    @classmethod
    def from_checkpoint(cls, ckpt_dir: Optional[str]) -> "Qwen35Config":
        if not ckpt_dir:
            return cls()
        with open(os.path.join(ckpt_dir, "config.json")) as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def from_saved(cls, d: dict) -> "Qwen35Config":
        """Inverse of to_dict (pack meta.json / NEFF manifests)."""
        d = dict(d)
        d["layer_types"] = tuple(d.get("layer_types", ()))
        return cls(**d)

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["layer_types"] = list(self.layer_types)
        return d


def rope_tables(cfg: Qwen35Config, positions: int):
    """cos / sin [positions, rot/2] fp32 (the kernels' [Lmax, 32] tables: HF default
    RoPE over the first ``rot_dim`` dims; text-only mRoPE equals plain RoPE)."""
    import numpy as np

    rot = cfg.rot_dim
    inv = 1.0 / cfg.rope_theta ** (np.arange(0, rot, 2, dtype=np.float32) / rot)
    ang = np.arange(positions, dtype=np.float32)[:, None] * inv[None, :]
    return np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)
