# SPDX-License-Identifier: Apache-2.0
"""The attention, norm and AttnRes weights of one Kimi-K3 decode layer for one rank (attention TP rank ``t``).

Each function reads the TP shard of rank ``t`` from the checkpoint, puts it into a zero global tensor at its position
(``weights_host.embed``) and calls the host builder of the vendored decode tree, which gives the kernel layout.

KDA shard of TP rank ``t`` (``hc = heads / tp`` heads, ``p = 128 hc`` channels): q, k, v, g, f_b, the three short
convolutions and dt_bias rows ``[p t, p t + p)``; b rows ``[hc t, hc t + hc)``; A_log values ``[hc t, hc t + hc)`` of
the first ``heads`` values; o_proj columns ``[p t, p t + p)``; f_a and o_norm whole. The ``kda_w_in`` linears (q, k,
v, g, f_a, b) go to the builder as MXFP8 hT_quad bytes and scales (``prefill.mx_linear.mx_quantize``, the
``nisa.quantize_mx`` rule); the other MX classes the builders quantize from BF16 themselves.

MLA shard of TP rank ``t`` (3 heads): q_a rows ``[48 t, 48 t + 48)``, q_b rows ``[576 t, + 576)``, kv_b rows
``[768 t, + 768)``, g rows ``[384 t, + 384)``, o_proj columns ``[384 t, + 384)``; kv_a and the two norms whole (the
builder picks the kv_a features ``32 c + t``).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from sglang_neuron.models.kimi_k3.decode.weights_host import (
    KDA_MX,
    MLA_MX,
    MX_ORDER,
    MX_SCALE_SUFFIX,
    embed,
    to_hT_order,
    to_numpy,
)
from sglang_neuron.models.kimi_k3.decode.weights_source import DecodeRank, RankCheckpoint
from sglang_neuron.models.kimi_k3.prefill.mx_linear import mx_quantize

BF16, FP32 = torch.bfloat16, torch.float32

KDA_CONV_DTYPE = "fp32"
"""The KDA short convolution of the served decode: FP32 conv weights in ``kda_params``, FP32 products and sum."""

VECTOR_NAMES = ("norm_weights", "attn_res_weights")


def kda_layer(ckpt: RankCheckpoint, layer: int, rank: DecodeRank, heads: int, head_dim: int, builder: Any,
              conv_dtype: str = KDA_CONV_DTYPE) -> dict[str, np.ndarray]:
    """The kernel KDA weights of one layer: ``kda_w_in`` (+ scale), ``kda_w_out`` (+ scale), ``kda_w_small``,
    ``kda_params`` (the plain width; the decode widens it on the device).

    Args:
        builder: The KDA host builder module (``kda_decode_torch``).
    """
    base = f"model.layers.{layer}.self_attn"
    tp, t = rank.tp, rank.tp_rank
    hc = heads // tp
    width = hc * head_dim
    c0, c1 = t * width, (t + 1) * width
    weights = {}

    def put_w_in(name: str, shard: torch.Tensor, dim: int | None) -> None:
        # The kda_w_in linears go to the builder as MXFP8 bytes + E8M0 scales (per row and group, so the shard's
        # bytes are those of the whole weight); the builder packs them.
        data, scale = mx_quantize(shard, MX_ORDER, what=f"{base}.{name}")
        weights[name] = embed(data, dim, tp, t)
        weights[f"{name}{MX_SCALE_SUFFIX}"] = embed(scale, dim, tp, t)

    for name in ("q_proj", "k_proj", "v_proj", "g_proj"):
        put_w_in(name, ckpt.rows(f"{base}.{name}.weight", c0, c1, BF16), 0)
    put_w_in("b_proj", ckpt.rows(f"{base}.b_proj.weight", t * hc, (t + 1) * hc, BF16), 0)
    put_w_in("f_a_proj", ckpt.full(f"{base}.f_a_proj.weight", BF16), None)
    weights["f_b_proj"] = embed(ckpt.rows(f"{base}.f_b_proj.weight", c0, c1, BF16), 0, tp, t)
    weights["o_proj"] = embed(ckpt.cols(f"{base}.o_proj.weight", c0, c1, BF16), 1, tp, t)
    for name in ("q_conv1d", "k_conv1d", "v_conv1d"):
        weights[name] = embed(ckpt.rows(f"{base}.{name}.weight", c0, c1, FP32), 0, tp, t)
    # The checkpoint A_log can hold more values than heads; the first ``heads`` are the heads.
    a_log = ckpt.full(f"{base}.A_log", FP32)
    if a_log.dim() != 1 or a_log.shape[0] < heads:
        raise ValueError(f"The KDA {base}.A_log is {tuple(a_log.shape)}; it needs {heads} values or more.")
    weights["a_log"] = embed(a_log.narrow(0, t * hc, hc).contiguous(), 0, tp, t)
    weights["dt_bias"] = embed(ckpt.rows(f"{base}.dt_bias", c0, c1, FP32), 0, tp, t)
    weights["o_norm"] = ckpt.full(f"{base}.o_norm.weight", FP32)
    return builder.build_kda_rank_weights(weights, tp, t, conv_dtype=conv_dtype, mx_weights=dict(KDA_MX))


def mla_layer(ckpt: RankCheckpoint, layer: int, rank: DecodeRank, builder: Any) -> dict[str, np.ndarray]:
    """The kernel MLA weights of one layer (``mla_decode_rank_weights``): ``mla_in_proj``, ``mla_q_a_norm``,
    ``mla_q_b_proj``, ``mla_kv_a_norm``, ``mla_kv_b_proj``, ``mla_o_proj`` and the scales of the four projections.

    Args:
        builder: The MLA host builder module (``mla_decode_torch``).
    """
    base = f"model.layers.{layer}.self_attn"
    tp, t = rank.tp, rank.tp_rank

    def column_shard(name: str):
        rows = ckpt.shape(f"{base}.{name}.weight")[0] // tp
        return embed(ckpt.rows(f"{base}.{name}.weight", t * rows, (t + 1) * rows, BF16), 0, tp, t)

    o_cols = ckpt.shape(f"{base}.o_proj.weight")[1] // tp
    weights = {
        "q_a_proj": column_shard("q_a_proj"),
        "q_a_layernorm": ckpt.full(f"{base}.q_a_layernorm.weight", BF16),
        "q_b_proj": column_shard("q_b_proj"),
        "kv_a_proj": ckpt.full(f"{base}.kv_a_proj_with_mqa.weight", BF16),
        "kv_a_layernorm": ckpt.full(f"{base}.kv_a_layernorm.weight", BF16),
        "kv_b_proj": column_shard("kv_b_proj"),
        "g_proj": column_shard("g_proj"),
        "o_proj": embed(ckpt.cols(f"{base}.o_proj.weight", t * o_cols, (t + 1) * o_cols, BF16), 1, tp, t),
    }
    return builder.mla_decode_rank_weights({k: to_numpy(v) for k, v in weights.items()}, t, mx_weights=dict(MLA_MX))


def vector_layer(ckpt: RankCheckpoint, layer: int) -> dict[str, torch.Tensor]:
    """The norms and AttnRes score weights of one layer, in hT order (whole on every rank).

    ``norm_weights`` BF16 ``[2, H]``: the input norm, the post-attention norm. ``attn_res_weights`` FP32 ``[2, H]``:
    the attention and the MLP mix score weights ``norm * proj``, products in FP32.
    """
    base = f"model.layers.{layer}"

    def read(name: str):
        return ckpt.full(f"{base}.{name}.weight", BF16)

    norms = torch.stack((read("input_layernorm"), read("post_attention_layernorm")))
    scores = torch.stack((
        read("self_attention_res_norm").float() * read("self_attention_res_proj").reshape(-1).float(),
        read("mlp_res_norm").float() * read("mlp_res_proj").reshape(-1).float(),
    ))
    return {"norm_weights": to_hT_order(norms), "attn_res_weights": to_hT_order(scores)}
