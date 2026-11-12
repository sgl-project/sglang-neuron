# SPDX-License-Identifier: Apache-2.0
"""The feed-forward weights of one Kimi-K3 decode layer for one rank of the 2-server world.

Dense layer 0 shards over the world: rank ``g`` owns the intermediate columns ``[I g, I g + I)``, ``I = 33792 / W``
(``dense_ffn_world_rank_weights``).

The MoE projections take the DP form: TP32 inside the replica, so rank ``g`` holds the shards of TP rank
``t = g % 32`` (``latent_moe_dp_rank_weights``): router rows ``[28 t, 28 t + 28)``, the 128-row latent-down block
``[128 t, 128 t + 128)`` (zeros for ``t >= 28``: 3584 = 28 x 128 latent rows), latent norm and latent-up columns
``[112 t, + 112)``, shared gate / up rows and shared down columns ``[192 t, + 192)``. ``moe_in_proj`` is MXFP8 and
``moe_router_weight`` the FP16 router rows of the TP rank. The out projection takes the column form
(``kernel_calls.MOE_OUT_PROJ_FORM``): ``moe_out_proj`` is BF16 ``[304, H]`` and holds, for the hidden rows
``128 h + 4 t + q`` of TP rank ``t``, the weights from all 3,584 latent and 6,144 shared inputs (``column_out_proj``).
``moe_vectors`` holds the router bias of every expert after the DP-form columns, then the 3,584 latent norm weights
(``column_vectors``).

The routed experts split EP x TP over the world (``moe_routed_layout`` served split): rank ``g`` holds the 7
experts of group ``g % 128`` and their intermediate part ``g // 128`` (1536 of 3072) in the MXFP4 static-loop layout
(``repack_part_experts``).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from sglang_neuron.models.kimi_k3.decode.kernel_calls import MOE_OUT_PROJ_FORM
from sglang_neuron.models.kimi_k3.decode.weights_host import (
    MOE_MX,
    embed,
    embed_row_block,
    router_rows,
    vectors_with_router_bias,
)
from sglang_neuron.models.kimi_k3.decode.weights_source import DecodeRank, RankCheckpoint

BF16, FP32 = torch.bfloat16, torch.float32

LATENT_BLOCK = 128
"""The latent-down rows of TP rank ``t`` in the MXFP8 latent router rows: ``[128 t, 128 t + 128)``."""

MOE_STACK_NAMES = ("moe_in_proj", "moe_out_proj", "moe_vectors", "moe_router_weight", "moe_in_proj_scale")
EXPERT_STACK_NAMES = ("expert_gate_up_weights", "expert_gate_up_weights_scale", "expert_down_weights",
                      "expert_down_weights_scale")
DENSE_STACK_NAMES = ("dense_gate_up", "dense_down")

_DP_NAMES = {"moe_in_proj_dp": "moe_in_proj", "moe_out_proj_dp": "moe_out_proj", "moe_vectors_dp": "moe_vectors",
             "moe_in_proj_scale": "moe_in_proj_scale"}

PARTITIONS = 128
"""The hT partitions: hidden row ``128 h + p`` is partition ``p`` of hidden tile ``h``."""

COLUMN_LATENT_TILES = 28
"""Latent K-tiles of the column out projection: partition ``p`` of K-tile ``c`` is latent input ``28 p + c``."""

COLUMN_SHARED_TILES = 48
"""Shared K-tiles of the column out projection: partition ``p`` of K-tile ``28 + c`` is shared input ``48 p + c``."""

COLUMN_OUT_ROWS = 304
"""Rows of ``moe_out_proj`` (``moe_out_proj_dp`` ``[304, H]``): the same tensor shape as the reduced layout."""


def column_hidden_rows(tp_rank: int, tp: int, hidden: int) -> Tensor:
    """The hidden rows of TP rank ``tp_rank`` in the column out projection, in the order of its columns: position
    ``(hidden // 128) q + h`` holds hidden row ``128 h + (128 // tp) tp_rank + q`` (``tkg_moe_2srv.COLUMN_K_TILES``).

    Example:
        >>> column_hidden_rows(1, 32, 7168)[:2].tolist(), column_hidden_rows(1, 32, 7168)[56].item()
        ([4, 132], 5)
    """
    if PARTITIONS % tp or hidden % PARTITIONS or not 0 <= tp_rank < tp:
        raise ValueError(f"The column out projection splits {PARTITIONS} partitions over TP {tp} and the hidden size "
                         f"{hidden} in tiles of {PARTITIONS}; TP rank {tp_rank} does not fit.")
    per = PARTITIONS // tp
    q = torch.arange(per).reshape(-1, 1)
    h = torch.arange(hidden // PARTITIONS).reshape(1, -1)
    return (PARTITIONS * h + per * tp_rank + q).reshape(-1)


def column_out_proj(latent_up_rows: Tensor, shared_down_rows: Tensor) -> Tensor:
    """The column out projection ``[304, H]`` BF16 of one TP rank from its hidden rows of the latent-up weight
    ``[R, 3584]`` and the shared-down weight ``[R, 6144]`` (``weight[out, in]``, the rows in ``column_hidden_rows``
    order, ``R = 4 H / 128``).

    Viewed as ``[128, 76, R]``, element ``(p, c, j)`` is ``latent_up_rows[j, 28 p + c]`` for ``c < 28`` and
    ``shared_down_rows[j, 48 p + c - 28]`` for ``c >= 28``: the layout of ``moe_out_proj_dp`` in
    ``tkg_moe_2srv.column_out_projection``. The values are copied exactly.

    Example:
        >>> column_out_proj(torch.zeros(16, 3584), torch.zeros(16, 6144)).shape
        torch.Size([304, 512])
    """
    rows = latent_up_rows.shape[0]
    if (tuple(latent_up_rows.shape) != (rows, PARTITIONS * COLUMN_LATENT_TILES)
            or tuple(shared_down_rows.shape) != (rows, PARTITIONS * COLUMN_SHARED_TILES)):
        raise ValueError(f"The column out projection takes [R, {PARTITIONS * COLUMN_LATENT_TILES}] latent-up and "
                         f"[R, {PARTITIONS * COLUMN_SHARED_TILES}] shared-down rows, got "
                         f"{tuple(latent_up_rows.shape)} and {tuple(shared_down_rows.shape)}.")
    latent = latent_up_rows.to(BF16).reshape(rows, PARTITIONS, COLUMN_LATENT_TILES).permute(1, 2, 0)
    shared = shared_down_rows.to(BF16).reshape(rows, PARTITIONS, COLUMN_SHARED_TILES).permute(1, 2, 0)
    tiles = torch.cat([latent, shared], dim=1).contiguous()
    return tiles.reshape(COLUMN_OUT_ROWS, -1)


def dense_layer(ckpt: RankCheckpoint, layer: int, rank: DecodeRank, builder: Any) -> dict[str, np.ndarray]:
    """``dense_gate_up`` ``[128, 56, 2 I]`` and ``dense_down`` ``[I, H]`` (BF16) of the world shard of the rank.

    Args:
        builder: The world dense host builder module (``ffn_world_torch``).
    """
    base = f"model.layers.{layer}.mlp"
    world, g = rank.world, rank.rank
    shard = ckpt.shape(f"{base}.gate_proj.weight")[0] // world
    r0, r1 = g * shard, (g + 1) * shard
    return builder.dense_ffn_world_rank_weights(
        g, world,
        gate_weight=embed(ckpt.rows(f"{base}.gate_proj.weight", r0, r1, BF16), 0, world, g),
        up_weight=embed(ckpt.rows(f"{base}.up_proj.weight", r0, r1, BF16), 0, world, g),
        down_weight=embed(ckpt.cols(f"{base}.down_proj.weight", r0, r1, BF16), 1, world, g),
    )


def moe_layer(ckpt: RankCheckpoint, layer: int, rank: DecodeRank, builder: Any) -> dict[str, Any]:
    """The MoE projection weights of one layer (``MOE_STACK_NAMES``).

    Args:
        builder: The DP MoE host builder module (``moe_decode_dp_torch``).
    """
    base = f"model.layers.{layer}.block_sparse_moe"
    tp, t = rank.tp, rank.tp_rank
    router = ckpt.full(f"{base}.gate.weight", BF16)
    bias = ckpt.full(f"{base}.gate.e_score_correction_bias", FP32)
    latent = ckpt.shape(f"{base}.routed_expert_down_proj.weight")[0]
    shared = ckpt.shape(f"{base}.shared_experts.gate_proj.weight")[0]
    if latent != LATENT_BLOCK * (latent // LATENT_BLOCK) or tp <= latent // LATENT_BLOCK:
        raise ValueError(f"The MXFP8 latent router rows give {latent // LATENT_BLOCK} TP ranks a {LATENT_BLOCK}-row "
                         f"latent block; the latent is {latent} rows and TP {tp}.")
    l0, l1 = t * (latent // tp), (t + 1) * (latent // tp)
    s0, s1 = t * (shared // tp), (t + 1) * (shared // tp)
    b0, b1 = min(t * LATENT_BLOCK, latent), min((t + 1) * LATENT_BLOCK, latent)
    down_block = torch.zeros((LATENT_BLOCK, router.shape[1]), dtype=BF16)
    if b1 > b0:
        down_block[: b1 - b0] = ckpt.rows(f"{base}.routed_expert_down_proj.weight", b0, b1, BF16)
    weights = {
        "router_weight": router,
        "router_bias": bias,
        "latent_down_weight": embed_row_block(down_block, latent, t * LATENT_BLOCK),
        "latent_norm_weight": embed(ckpt.rows(f"{base}.routed_expert_norm.weight", l0, l1, BF16), 0, tp, t),
        "latent_up_weight": embed(ckpt.cols(f"{base}.routed_expert_up_proj.weight", l0, l1, BF16), 1, tp, t),
        "shared_gate_weight": embed(ckpt.rows(f"{base}.shared_experts.gate_proj.weight", s0, s1, BF16), 0, tp, t),
        "shared_up_weight": embed(ckpt.rows(f"{base}.shared_experts.up_proj.weight", s0, s1, BF16), 0, tp, t),
        "shared_down_weight": embed(ckpt.cols(f"{base}.shared_experts.down_proj.weight", s0, s1, BF16), 1, tp, t),
    }
    with torch.device("cpu"):
        built = builder.latent_moe_dp_rank_weights(rank.rank, **weights, mx_weights=dict(MOE_MX),
                                                   latent_gather_mx=True)
    arrays = {_DP_NAMES[name]: value for name, value in built.items()}
    if MOE_OUT_PROJ_FORM != "column":
        raise ValueError(f"The decode MoE weights take the column out projection, not {MOE_OUT_PROJ_FORM!r}.")
    # the column form: the builder's reduced out projection is replaced, so its BF16 row shard is not used
    hidden = router.shape[1]
    rows = column_hidden_rows(t, tp, hidden)
    arrays["moe_out_proj"] = column_out_proj(_hidden_rows(ckpt, f"{base}.routed_expert_up_proj.weight", rows),
                                             _hidden_rows(ckpt, f"{base}.shared_experts.down_proj.weight", rows))
    norm = ckpt.full(f"{base}.routed_expert_norm.weight", BF16)
    arrays["moe_vectors"] = column_vectors(vectors_with_router_bias(arrays["moe_vectors"], bias), norm)
    arrays["moe_router_weight"] = router_rows(router, t, tp)
    return {name: arrays[name] for name in MOE_STACK_NAMES}


def _hidden_rows(ckpt: RankCheckpoint, key: str, rows: Tensor) -> Tensor:
    """The rows ``rows`` of ``key`` (BF16) in their order, read as runs of consecutive rows."""
    values = rows.tolist()
    parts, start = [], 0
    for i in range(1, len(values) + 1):
        if i == len(values) or values[i] != values[i - 1] + 1:
            parts.append(ckpt.rows(key, values[start], values[i - 1] + 1, BF16))
            start = i
    return torch.cat(parts, dim=0)


def column_vectors(vectors: Tensor, latent_norm_weight: Tensor) -> Tensor:
    """``moe_vectors`` of the column out projection: ``vectors`` (the DP columns and the router biases) and then the
    latent norm weights of all latent columns, in latent order, as the FP32 values of their BF16 rounding
    (``tkg_moe_2srv.column_norm_cols``).

    Example:
        >>> column_vectors(torch.zeros(1, 1036), torch.ones(3584)).shape
        torch.Size([1, 4620])
    """
    norm = latent_norm_weight.to(BF16).to(FP32).reshape(1, -1)
    return torch.cat([torch.as_tensor(vectors, dtype=FP32).reshape(1, -1), norm], dim=1)


def expert_split(rank: DecodeRank, routed: Any) -> tuple[dict, tuple[int, ...], slice]:
    """``(split, experts, intermediate rows)`` of the rank: the served EP x TP split of the world."""
    split = routed.routed_split(rank.world, rank.routed_tp)
    if split["layout_form"] != "served":
        raise ValueError(f"The routed split of world {rank.world} is {split['layout_tag']}; the decode stacks take "
                         "the served layout.")
    group, part = routed.rank_group_part(split, rank.rank)
    return split, routed.group_experts(split, group), routed.part_rows(split, part)


def expert_layer(ckpt: RankCheckpoint, layer: int, rank: DecodeRank, routed: Any) -> dict[str, np.ndarray]:
    """The four routed expert tensors of one layer (``EXPERT_STACK_NAMES``) of the rank's experts and part.

    Args:
        routed: The routed layout host module (``moe_routed_layout``).
    """
    split, experts, rows = expert_split(rank, routed)
    group_size = 32
    down_bytes = slice(rows.start // 2, rows.stop // 2)
    down_groups = slice(rows.start // group_size, rows.stop // group_size)
    parts = []
    for expert in experts:
        base = f"model.layers.{layer}.block_sparse_moe.experts.{expert}"

        def read(name: str, index: tuple) -> np.ndarray:
            tensor = ckpt.read(f"{base}.{name}", index)
            if tensor.dtype != torch.uint8:
                raise TypeError(f"The MXFP4 expert tensor {base}.{name} is {tensor.dtype}; it must be uint8.")
            return tensor.numpy()

        parts.append({
            "w1.weight_packed": read("w1.weight_packed", (rows,)),
            "w1.weight_scale": read("w1.weight_scale", (rows,)),
            "w3.weight_packed": read("w3.weight_packed", (rows,)),
            "w3.weight_scale": read("w3.weight_scale", (rows,)),
            "w2.weight_packed": read("w2.weight_packed", (slice(None), down_bytes)),
            "w2.weight_scale": read("w2.weight_scale", (slice(None), down_groups)),
        })
    tensors = routed.repack_part_experts(parts, split)
    return {name: tensors[name] for name in EXPERT_STACK_NAMES}
