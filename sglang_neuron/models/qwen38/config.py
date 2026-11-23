# SPDX-License-Identifier: Apache-2.0
"""The shape of a Qwen3.8 dense hybrid checkpoint, and the layout its graphs run in.

Qwen3.8-27B is the dense member of the Qwen3.5 family: the same per-head geometry and the same config field names
(its architecture string is still ``Qwen3_5ForConditionalGeneration``), with a dense SwiGLU MLP in place of the MoE
block. 64 layers as ``[3 linear_attention + 1 full_attention] x 16``: 48 gated delta net layers (16 key heads and 48
value heads of 128, conv kernel 4) and 16 gated attention layers (24 query heads and 4 KV heads of 256, partial RoPE
over a quarter of the head, theta 1e7).

The graphs run the Qwen3.5 per-layer kernels (models/qwen35/kernels/giga: ``gdn_decode_tkg``, ``gdn_prefill``,
``attn_decode_tkg``), which are built for one rank per physical NeuronCore (LNC1) and for exactly this geometry: GDN
head dim 128 with conv kernel 4, attention head dim 256 with 64 rotary dims and one KV head per rank. The refusals
below name any config that leaves it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

# the kernels' fixed geometry (models/qwen35/kernels/giga: gdn_decode.HEAD, attn_decode.ROT)
GDN_HEAD_DIM = 128
GDN_CONV_KERNEL = 4
ATTENTION_HEAD_DIM = 256
ROTARY_DIMS = 64
FULL_ATTENTION = "full_attention"
LINEAR_ATTENTION = "linear_attention"


def text_config(hf_config):
    """The language model's config: ``text_config`` of a multimodal config, else the config itself."""
    return getattr(hf_config, "text_config", None) or hf_config


def _get(cfg, name, default=None):
    if isinstance(cfg, dict):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


@dataclass(frozen=True)
class Qwen38Dims:
    hidden: int
    num_layers: int
    layer_types: Tuple[str, ...]
    # gated delta net
    num_k_heads: int
    num_v_heads: int
    head_k_dim: int
    head_v_dim: int
    conv_kernel: int
    # gated attention
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    rotary_dim: int
    rope_theta: float
    # dense MLP and the rest
    intermediate: int
    vocab: int
    rms_norm_eps: float
    tie_word_embeddings: bool

    @classmethod
    def from_hf(cls, hf_config) -> "Qwen38Dims":
        t = text_config(hf_config)
        n = int(_get(t, "num_hidden_layers"))
        interval = int(_get(t, "full_attention_interval", 4) or 4)
        types = _get(t, "layer_types") or [
            FULL_ATTENTION if i % interval == interval - 1 else LINEAR_ATTENTION for i in range(n)]
        rope = _get(t, "rope_parameters") or {}
        head_dim = int(_get(t, "head_dim"))
        partial = float(_get(rope, "partial_rotary_factor", _get(t, "partial_rotary_factor", 0.25)))
        return cls(
            hidden=int(_get(t, "hidden_size")), num_layers=n, layer_types=tuple(types),
            num_k_heads=int(_get(t, "linear_num_key_heads")), num_v_heads=int(_get(t, "linear_num_value_heads")),
            head_k_dim=int(_get(t, "linear_key_head_dim")), head_v_dim=int(_get(t, "linear_value_head_dim")),
            conv_kernel=int(_get(t, "linear_conv_kernel_dim")),
            num_q_heads=int(_get(t, "num_attention_heads")), num_kv_heads=int(_get(t, "num_key_value_heads")),
            head_dim=head_dim, rotary_dim=int(head_dim * partial),
            rope_theta=float(_get(rope, "rope_theta", _get(t, "rope_theta", 1e7))),
            intermediate=int(_get(t, "intermediate_size")), vocab=int(_get(t, "vocab_size")),
            rms_norm_eps=float(_get(t, "rms_norm_eps", 1e-6)),
            tie_word_embeddings=bool(_get(t, "tie_word_embeddings", False)),
        )

    # the names the platform reads off a graph's config (models/host.SampledExecution, the logs)
    @property
    def vocab_size(self) -> int:
        return self.vocab

    @property
    def num_hidden_layers(self) -> int:
        return self.num_layers

    @property
    def hidden_size(self) -> int:
        return self.hidden

    @property
    def gdn_layers(self) -> Tuple[int, ...]:
        return tuple(i for i, kind in enumerate(self.layer_types) if kind == LINEAR_ATTENTION)

    @property
    def attention_layers(self) -> Tuple[int, ...]:
        return tuple(i for i, kind in enumerate(self.layer_types) if kind == FULL_ATTENTION)

    @property
    def conv_dim(self) -> int:
        """GDN conv channels over all heads, in the kernels' order q | k | v."""
        return 2 * self.num_k_heads * self.head_k_dim + self.num_v_heads * self.head_v_dim

    def problems(self, tp_size: int) -> list:
        """Every reason these graphs cannot serve this config at this attention TP, naming the field."""
        out = []
        if self.head_k_dim != GDN_HEAD_DIM or self.head_v_dim != GDN_HEAD_DIM:
            out.append(f"GDN head dims {self.head_k_dim} and {self.head_v_dim}: the kernels are built for "
                       f"{GDN_HEAD_DIM}")
        if self.conv_kernel != GDN_CONV_KERNEL:
            out.append(f"GDN conv kernel {self.conv_kernel}: the kernels keep a {GDN_CONV_KERNEL - 1}-token window")
        if self.head_dim != ATTENTION_HEAD_DIM:
            out.append(f"attention head dim {self.head_dim}: the attention kernel is built for "
                       f"{ATTENTION_HEAD_DIM}")
        if self.rotary_dim != ROTARY_DIMS:
            out.append(f"{self.rotary_dim} rotary dims: the attention kernel rotates the first {ROTARY_DIMS}")
        for name, value in (("GDN key heads", self.num_k_heads), ("GDN value heads", self.num_v_heads),
                            ("attention query heads", self.num_q_heads)):
            if value % tp_size:
                out.append(f"{value} {name} do not shard over attention TP {tp_size}")
        if self.num_kv_heads % tp_size and tp_size % self.num_kv_heads:
            out.append(f"{self.num_kv_heads} KV heads neither shard over nor replicate across attention TP "
                       f"{tp_size}")
        if self.intermediate % tp_size:
            out.append(f"an intermediate size of {self.intermediate} does not shard over {tp_size} ranks")
        if self.hidden % GDN_HEAD_DIM:
            out.append(f"a hidden size of {self.hidden} is not a multiple of {GDN_HEAD_DIM}, which the GDN "
                       "projection layout needs")
        unknown = sorted({k for k in self.layer_types} - {FULL_ATTENTION, LINEAR_ATTENTION})
        if unknown:
            out.append(f"layer kinds {unknown}: the graphs serve {LINEAR_ATTENTION} and {FULL_ATTENTION}")
        return out


@dataclass(frozen=True)
class Qwen38Layout:
    """What each rank holds. The kernels run LNC1, so a trn2.3xlarge serves this model at TP8 over the chip's eight
    physical NeuronCores."""

    tp_size: int
    dims: Qwen38Dims

    @property
    def gdn_v_heads(self) -> int:
        return self.dims.num_v_heads // self.tp_size

    @property
    def gdn_k_heads(self) -> int:
        return self.dims.num_k_heads // self.tp_size

    @property
    def q_heads(self) -> int:
        return self.dims.num_q_heads // self.tp_size

    @property
    def kernel_q_heads(self) -> int:
        """Query heads a rank runs: its own, plus one all-zero head when that count is odd. The attention prefill
        kernel requires an even count ("nq must be even"); Qwen3.8 at TP8 has 3 a rank. The extra head's output
        projection rows are zero, so it adds nothing to the layer's output."""
        return self.q_heads + self.q_heads % 2

    @property
    def kv_heads(self) -> int:
        """KV heads a rank holds; with more ranks than KV heads each head is replicated."""
        return max(1, self.dims.num_kv_heads // self.tp_size)

    @property
    def intermediate(self) -> int:
        return self.dims.intermediate // self.tp_size
