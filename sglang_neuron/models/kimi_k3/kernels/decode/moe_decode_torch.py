# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Global torch references and host weight layouts of the Kimi-K3 latent MoE and dense MLP decode bodies.

The references take the global (unsharded) weights in the serving ``weight[out, in]`` orientation.
"""

from collections.abc import Callable, Mapping
from typing import Optional

import ml_dtypes
import numpy as np
import torch

from .moe.routed import moe_tkg_decode_torch_ref
from .tkg_moe import (
    COLUMN_FORMS,
    COLUMN_K_TILES,
    DENSE_SHARD,
    DP_SIZE,
    LATENT_NORM_EPS,
    LATENT_SHARD,
    LATENT_SIZE,
    LOCAL_EXPERTS,
    MOE_VECTOR_COLS,
    MX_BLOCK,
    MX_LATENT_BLOCKS,
    NUM_EXPERTS,
    OUT_PROJ_FORMS,
    ROUTER_SUM_EPS,
    SHARED_SHARD,
    SITU_BETA,
    SITU_LINEAR_BETA,
    TOP_K,
    WORLD_SIZE,
)
from .mx_weights_torch import mx_quantize, pack_mx_hT

MX_WEIGHT_CLASSES = ("moe_in_proj",)
"""MXFP8 weight classes of ``latent_moe_rank_weights`` (``mx_weights``): the latent down and the shared gate / up
columns of ``moe_in_proj``, with the FP16 router (``tkg_moe.MXFP8_IN_PROJ_COLS``)."""
MX_WEIGHT_ORDERS = {"moe_in_proj": ("hT_quad",)}
"""The group orders of each class of ``MX_WEIGHT_CLASSES``."""
MX_FROM_BF16 = ("moe_in_proj",)
"""The classes of ``MX_WEIGHT_CLASSES`` that the builder quantizes itself from the BF16 weights (a loader gives it BF16
weights, not MXFP8 linears)."""

_EXPERT_KEYS = (
    "expert_gate_up_weights",
    "expert_gate_up_weights_scale",
    "expert_down_weights",
    "expert_down_weights_scale",
)
# The positional order of the expert arrays in ``moe_tkg_decode`` and its reference.
_EXPERT_KEYS_KERNEL_ORDER = (
    "expert_gate_up_weights",
    "expert_down_weights",
    "expert_gate_up_weights_scale",
    "expert_down_weights_scale",
)
TP_SIZE = WORLD_SIZE // DP_SIZE


def latent_moe_decode_torch_ref(
    x,
    router_weight,
    router_bias,
    latent_down_weight,
    latent_norm_weight,
    latent_up_weight,
    shared_gate_weight,
    shared_up_weight,
    shared_down_weight,
    expert_pool: Mapping[str, np.ndarray] = None,
    rank_partials: Optional[Callable] = None,
    router_input=None,
    out_proj_form: str = "reduced",
) -> dict[str, torch.Tensor]:
    """Compute the Kimi-K3 latent MoE decode output of the T gathered tokens from the global weights.

    The math follows the serving decode layer. The bf16 boundaries follow the 128-rank kernel:
    each rank's routed partial and each rank's output partial are rounded to bf16 before the
    world sum. A rank's output partial is the FP32 sum of its latent up and its shared down
    partials (one K = 76 matmul in the kernel), rounded once. By default the routed experts are
    EP128 and use the ``moe_tkg_decode`` reference (MX activations), and only the selected experts
    are evaluated.

    Args:
        x: [T, 7168] BF16, the normalized hidden of the T = 4 B gathered tokens (token r B + b is
            token b of DP replica r).
        router_weight: [896, 7168] BF16.
        router_bias: [896] FP32 selection-only correction bias.
        latent_down_weight: [3584, 7168] BF16.
        latent_norm_weight: [3584] BF16.
        latent_up_weight: [7168, 3584] BF16.
        shared_gate_weight: [6144, 7168] BF16.
        shared_up_weight: [6144, 7168] BF16.
        shared_down_weight: [7168, 6144] BF16.
        expert_pool: packed MXFP4 serving arrays with a leading pool axis P (the four
            ``moe_tkg_decode`` weight names). Global expert e uses pool entry e % P, so P = 896
            gives every expert its own weights. Used without ``rank_partials``.
        rank_partials: optional ``f(latent, topk_indices, topk_weights)`` that returns the BF16
            [T, 3584] routed partials of the ranks (for example the TP128 slices of each rank).
            The routed latent is then their FP32 sum, rounded once to BF16.
        router_input: optional [T, 7168] FP16 router input (the FP16 router of the kernel): the logits
            are then ``router_input @ fp16(router_weight)^T`` in FP32.
        out_proj_form: ``reduced`` (default) or a column form (``tkg_moe.OUT_PROJ_FORMS``,
            ``COLUMN_FORMS``; see :func:`latent_moe_finish_torch`).

    Returns:
        dict with ``out`` [T, 7168] BF16, ``routed`` [T, 3584] BF16 (the reduced expert output),
        ``affinities`` [T, 896] FP32, ``topk_indices`` [T, 16] INT64 and ``topk_weights`` [T, 16] FP32.
    """
    route = latent_moe_route_torch(x, router_weight, router_bias, latent_down_weight, router_input)
    if rank_partials is None:
        partials = _ep128_partials(route, expert_pool)
    else:
        partials = rank_partials(route["latent"], route["topk_indices"], route["topk_weights"])
    routed_f32 = torch.zeros(route["latent"].shape, dtype=torch.float32)
    for partial in partials:
        routed_f32 = routed_f32 + partial.to(torch.bfloat16).float()
    routed = routed_f32.to(torch.bfloat16)
    out = latent_moe_finish_torch(
        routed,
        route["hidden_f32"],
        latent_norm_weight,
        latent_up_weight,
        shared_gate_weight,
        shared_up_weight,
        shared_down_weight,
        out_proj_form,
    )
    return {
        "out": out,
        "routed": routed,
        "affinities": route["affinities"],
        "topk_indices": route["topk_indices"],
        "topk_weights": route["topk_weights"],
    }


def latent_moe_route_torch(
    x, router_weight, router_bias, latent_down_weight, router_input=None
) -> dict[str, torch.Tensor]:
    """Route the T tokens and compute their latent: the first part of :func:`latent_moe_decode_torch_ref`.

    With ``router_input`` (FP16 [T, 7168]), the logits are ``router_input @ fp16(router_weight)^T`` in FP32 (the
    FP16 router); else ``x @ router_weight^T`` with the BF16 values.

    Returns:
        dict with ``hidden_f32`` [T, 7168], ``latent`` [T, 3584] BF16, ``topk_indices`` [T, 16] INT64
        (descending selection value), ``topk_weights`` [T, 16] FP32 (renormalized scores) and
        ``affinities`` [T, 896] FP32.
    """
    hidden = _to_torch(x).to(torch.bfloat16)
    hidden_f32 = hidden.float()
    if router_input is None:
        logits = hidden_f32 @ _to_torch(router_weight).float().T
    else:
        weight16 = _to_torch(router_weight).to(torch.float16).float()
        logits = _to_torch(router_input).to(torch.float16).float() @ weight16.T
    scores = torch.sigmoid(logits)
    selection = scores + _to_torch(router_bias).float().reshape(1, NUM_EXPERTS)
    topk_indices = torch.topk(selection, k=TOP_K, dim=1, sorted=True).indices
    topk_weights = torch.gather(scores, 1, topk_indices)
    topk_weights = topk_weights / (topk_weights.sum(dim=1, keepdim=True) + ROUTER_SUM_EPS)
    affinities = torch.zeros_like(scores).scatter(1, topk_indices, topk_weights)
    latent = (hidden_f32 @ _to_torch(latent_down_weight).float().T).to(torch.bfloat16)
    return {
        "hidden_f32": hidden_f32,
        "latent": latent,
        "topk_indices": topk_indices,
        "topk_weights": topk_weights,
        "affinities": affinities,
    }


def latent_moe_finish_torch(
    routed,
    hidden_f32,
    latent_norm_weight,
    latent_up_weight,
    shared_gate_weight,
    shared_up_weight,
    shared_down_weight,
    out_proj_form: str = "reduced",
) -> torch.Tensor:
    """Apply the latent norm, the latent up and the shared experts: the last part of :func:`latent_moe_decode_torch_ref`.

    ``reduced`` rounds each rank's partial (its latent-up and shared-down shards) to BF16 before the world sum, as the
    kernel's all-reduce does. ``column`` and ``column_shared_ag`` (``tkg_moe.COLUMN_FORMS``) sum all 9,728 inputs in
    FP32 and round once, as the column out-projection does (in another FP32 order).

    Returns:
        ``out`` [T, 7168] BF16.
    """
    if out_proj_form not in OUT_PROJ_FORMS:
        raise ValueError(f"out_proj_form must be one of {OUT_PROJ_FORMS}, got {out_proj_form!r}")
    routed_float = routed.float()
    inv_rms = torch.rsqrt(routed_float.square().sum(dim=1, keepdim=True) / LATENT_SIZE + LATENT_NORM_EPS)
    normed = (routed_float * inv_rms).to(torch.bfloat16)
    normed = (normed.float() * _to_torch(latent_norm_weight).float().reshape(1, LATENT_SIZE)).to(torch.bfloat16)

    shared_gate = (hidden_f32 @ _to_torch(shared_gate_weight).float().T).to(torch.bfloat16)
    shared_up = (hidden_f32 @ _to_torch(shared_up_weight).float().T).to(torch.bfloat16)
    shared_activation = _situ(shared_gate, shared_up)

    if out_proj_form in COLUMN_FORMS:
        out = normed.float() @ _to_torch(latent_up_weight).float().T
        out = out + shared_activation.float() @ _to_torch(shared_down_weight).float().T
        return out.to(torch.bfloat16)
    return _sum_of_rank_partials(
        [
            (normed, _to_torch(latent_up_weight), LATENT_SHARD),
            (shared_activation, _to_torch(shared_down_weight), SHARED_SHARD),
        ]
    )


def _ep128_partials(route: dict[str, torch.Tensor], expert_pool: Mapping[str, np.ndarray]) -> list[torch.Tensor]:
    """Return the EP128 routed partials of the owner ranks: each adds its local experts in FP32, in expert order."""
    latent = route["latent"]
    topk_indices = route["topk_indices"]
    affinities = route["affinities"]
    tokens = latent.shape[0]
    pool_size = expert_pool["expert_gate_up_weights"].shape[0]
    expert_outputs = {}
    for pool_idx in sorted({int(e) % pool_size for e in topk_indices.reshape(-1)}):
        expert_outputs[pool_idx] = moe_tkg_decode_torch_ref(
            latent,
            torch.ones((tokens, 1), dtype=torch.float32),
            *[expert_pool[key][pool_idx : pool_idx + 1] for key in _EXPERT_KEYS_KERNEL_ORDER],
            beta=SITU_BETA,
            linear_beta=SITU_LINEAR_BETA,
        )["out"].float()

    partials = []
    for rank in sorted({int(e) // LOCAL_EXPERTS for e in topk_indices.reshape(-1)}):
        rank_partial = torch.zeros((tokens, LATENT_SIZE), dtype=torch.float32)
        for expert in range(rank * LOCAL_EXPERTS, (rank + 1) * LOCAL_EXPERTS):
            if bool((affinities[:, expert] != 0).any()):
                rank_partial = rank_partial + expert_outputs[expert % pool_size] * affinities[:, expert : expert + 1]
        partials.append(rank_partial)
    return partials


def dense_mlp_decode_torch_ref(x, gate_weight, up_weight, down_weight) -> dict[str, torch.Tensor]:
    """Compute the Kimi-K3 dense SiTU MLP output of the T gathered tokens from the global weights.

    Each rank's TP128 down partial is rounded to bf16 before the world sum, as in the kernel.

    Args:
        x: [T, 7168] BF16, the normalized hidden of the T = 4 B gathered tokens.
        gate_weight: [33792, 7168] BF16.
        up_weight: [33792, 7168] BF16.
        down_weight: [7168, 33792] BF16.

    Returns:
        dict with ``out`` [T, 7168] BF16.
    """
    hidden_f32 = _to_torch(x).to(torch.bfloat16).float()
    gate = (hidden_f32 @ _to_torch(gate_weight).float().T).to(torch.bfloat16)
    up = (hidden_f32 @ _to_torch(up_weight).float().T).to(torch.bfloat16)
    activation = _situ(gate, up)
    return {"out": _sum_of_rank_partials([(activation, _to_torch(down_weight), DENSE_SHARD)])}


def latent_moe_rank_weights(
    rank: int,
    router_weight,
    router_bias,
    latent_down_weight,
    latent_norm_weight,
    latent_up_weight,
    shared_gate_weight,
    shared_up_weight,
    shared_down_weight,
    expert_weights: Optional[Mapping[str, np.ndarray]] = None,
    mx_weights: Optional[dict] = None,
) -> dict[str, np.ndarray]:
    """Build one rank's latent MoE kernel weights (``MOE_WEIGHT_NAMES``) from the global weights.

    Args:
        rank: Global rank g. It owns experts [7 g, 7 g + 7), latent columns [28 g, 28 g + 28),
            and shared intermediate columns [48 g, 48 g + 48).
        router_weight .. shared_down_weight: global weights, as in
            :func:`latent_moe_decode_torch_ref`.
        expert_weights: optional packed MXFP4 arrays of this rank's 7 experts, in the serving
            layout of ``moe_tkg_decode``. They are passed through unchanged.
        mx_weights: ``{"moe_in_proj": "hT_quad"}`` builds the MXFP8 ``moe_in_proj`` (the 124 latent down and shared
            gate / up columns, no router columns) and its ``moe_in_proj_scale`` (``tkg_mx_quant``) from the BF16 weights.
            The FP16 router gives the logits.

    Returns:
        ``moe_in_proj`` [128, 56, 131] BF16 (hT-tiled), ``moe_out_proj`` [76, 7168] BF16, ``moe_vectors``
        [1, 35] FP32, and the four expert arrays when ``expert_weights`` is given.
    """
    mx = dict(mx_weights or {})
    for name, order in mx.items():
        if order not in MX_WEIGHT_ORDERS.get(name, ()):
            raise ValueError(f"MXFP8 class {name!r} with order {order!r} is not in {MX_WEIGHT_ORDERS}")
    experts = slice(rank * LOCAL_EXPERTS, (rank + 1) * LOCAL_EXPERTS)
    latent = slice(rank * LATENT_SHARD, (rank + 1) * LATENT_SHARD)
    shared = slice(rank * SHARED_SHARD, (rank + 1) * SHARED_SHARD)
    in_proj = torch.cat(
        [
            _to_torch(router_weight)[experts].T,
            _to_torch(latent_down_weight)[latent].T,
            _to_torch(shared_gate_weight)[shared].T,
            _to_torch(shared_up_weight)[shared].T,
        ],
        dim=1,
    )
    out_proj = torch.cat(
        [_to_torch(latent_up_weight)[:, latent].T, _to_torch(shared_down_weight)[:, shared].T],
        dim=0,
    )
    bias = _to_torch(router_bias).float().reshape(NUM_EXPERTS)
    norm_weight = _to_torch(latent_norm_weight).to(torch.bfloat16).float().reshape(-1)
    vectors = torch.cat([bias[experts], norm_weight[latent]]).reshape(1, MOE_VECTOR_COLS)
    weights = {
        "moe_in_proj": _bf16_numpy(_hT_tiled(in_proj)),
        "moe_out_proj": _bf16_numpy(out_proj),
        "moe_vectors": vectors.numpy().astype(np.float32),
    }
    if "moe_in_proj" in mx:
        data, scale = mx_quantize(in_proj[:, LOCAL_EXPERTS:].T, mx["moe_in_proj"])
        weights["moe_in_proj"], weights["moe_in_proj_scale"] = pack_mx_hT(data.T, scale.T)
    if expert_weights is not None:
        for key in _EXPERT_KEYS:
            weights[key] = expert_weights[key]
    return weights


def latent_moe_drop_selection_vectors(moe_vectors: np.ndarray, router_bias) -> np.ndarray:
    """Return ``moe_vectors`` [1, 35] with the router correction bias of all 896 experts appended in expert order: the
    [1, 931] FP32 ``moe_vectors`` of ``router_rows_drop_selection`` (``tkg_moe.ALL_ROUTER_BIAS_COLS``). Every rank
    gets the same 896 values, the FP32 values of its own 7 bias columns."""
    bias = _to_torch(router_bias).float().reshape(1, NUM_EXPERTS).numpy().astype(np.float32)
    return np.concatenate([np.asarray(moe_vectors, dtype=np.float32).reshape(1, MOE_VECTOR_COLS), bias], axis=1)


def latent_moe_mx_rank_weights(
    rank: int,
    router_weight,
    router_bias,
    latent_down_weight,
    latent_norm_weight,
    latent_up_weight,
    shared_gate_weight,
    shared_up_weight,
    shared_down_weight,
    expert_weights: Optional[Mapping[str, np.ndarray]] = None,
) -> dict[str, np.ndarray]:
    """Build one rank's latent MoE kernel weights for ``latent_gather_mx`` (with ``router_rows_drop_selection``).

    The same arrays as :func:`latent_moe_rank_weights`, with two changes:

    - ``moe_in_proj`` [128, 56, 135] BF16 (hT-tiled): columns 0..6 the router rows 7 g .. 7 g + 6, columns 7..38 the
      latent-down rows [32 g, 32 g + 32) of the MXFP8 latent block of rank g < 112 (zeros for g >= 112), then the
      48 shared gate and the 48 shared up columns of rank g (``tkg_moe.MX_IN_PROJ_COLS``).
    - ``moe_vectors`` [1, 931] FP32: the 35 columns of the base layout (the 7 router biases of rank g and its 28
      latent norm weights, of the latent columns [28 g, 28 g + 28) of its latent up), then the router bias of all
      896 experts (:func:`latent_moe_drop_selection_vectors`).

    ``moe_out_proj`` and the expert arrays do not change: the latent up and the norm keep the 28-column shards.

    ``latent_down_weight`` must hold rows [32 g, 32 g + 32): the whole tensor, or a block of it. A tensor that holds only
    the rank's 28-row TP128 shard (rows 28 g .. 28 g + 27, zeros elsewhere) gives zeros for the rows outside the shard.
    """
    weights = latent_moe_rank_weights(
        rank,
        router_weight,
        router_bias,
        latent_down_weight,
        latent_norm_weight,
        latent_up_weight,
        shared_gate_weight,
        shared_up_weight,
        shared_down_weight,
        expert_weights,
    )
    experts = slice(rank * LOCAL_EXPERTS, (rank + 1) * LOCAL_EXPERTS)
    shared = slice(rank * SHARED_SHARD, (rank + 1) * SHARED_SHARD)
    down = _to_torch(latent_down_weight)
    if rank < MX_LATENT_BLOCKS:
        block = down[rank * MX_BLOCK : (rank + 1) * MX_BLOCK].T
    else:
        block = torch.zeros((down.shape[1], MX_BLOCK), dtype=down.dtype)
    in_proj = torch.cat(
        [
            _to_torch(router_weight)[experts].T,
            block,
            _to_torch(shared_gate_weight)[shared].T,
            _to_torch(shared_up_weight)[shared].T,
        ],
        dim=1,
    )
    weights["moe_in_proj"] = _bf16_numpy(_hT_tiled(in_proj))
    weights["moe_vectors"] = latent_moe_drop_selection_vectors(weights["moe_vectors"], router_bias)
    return weights


def latent_moe_dp_local_rank_weights(
    rank: int,
    router_weight,
    router_bias,
    latent_down_weight,
    latent_norm_weight,
    latent_up_weight,
    shared_gate_weight,
    shared_up_weight,
    shared_down_weight,
    expert_weights: Optional[Mapping[str, np.ndarray]] = None,
) -> dict[str, np.ndarray]:
    """Build one rank's latent MoE kernel weights for ``in_proj_form`` ``dp_local`` (``tkg_moe.IN_PROJ_FORMS``).

    Rank g = 32 d + t takes the router and latent-down columns of its 4 DP peer shards g' = 32 d' + t (d' = 0..3, in
    that order on every rank) and only its shared columns in ``moe_in_proj``. Every array is a slice of the arrays of
    :func:`latent_moe_rank_weights` of rank g or of a peer, so the bits are those of the gathered form.

    Args:
        rank: Global rank g.
        router_weight .. shared_down_weight: global weights, as in :func:`latent_moe_rank_weights`. Only these parts
            are read: the router rows and biases of the 28 peer experts, the latent-down rows of the 4 peer shards, and
            the rank's own latent-norm, latent-up and shared slices.
        expert_weights: optional expert arrays of this rank, passed through unchanged.

    Returns:
        ``moe_in_proj`` [128, 56, 96] BF16 (the shared gate and up columns, hT-tiled), ``moe_peer_in_proj``
        [128, 56, 140] BF16 (column 7 d' + e: router row 7 g' + e; column 28 + 28 d' + k: latent-down column 28 g' + k),
        ``moe_out_proj`` [76, 7168] BF16 (unchanged), ``moe_vectors`` [1, 63] FP32 (the 35 columns of the gathered form,
        then column 35 + 7 d' + e: the router bias of expert 7 g' + e), and the expert arrays when given.
    """
    global_weights = {
        "router_weight": router_weight,
        "router_bias": router_bias,
        "latent_down_weight": latent_down_weight,
        "latent_norm_weight": latent_norm_weight,
        "latent_up_weight": latent_up_weight,
        "shared_gate_weight": shared_gate_weight,
        "shared_up_weight": shared_up_weight,
        "shared_down_weight": shared_down_weight,
    }
    own = latent_moe_rank_weights(rank, expert_weights=expert_weights, **global_weights)
    tp_rank = rank % TP_SIZE
    peers = [latent_moe_rank_weights(TP_SIZE * dp_rank + tp_rank, **global_weights) for dp_rank in range(DP_SIZE)]
    router_cols, latent_cols = slice(0, LOCAL_EXPERTS), slice(LOCAL_EXPERTS, LOCAL_EXPERTS + LATENT_SHARD)
    weights = dict(own)
    weights["moe_peer_in_proj"] = np.ascontiguousarray(
        np.concatenate(
            [peer["moe_in_proj"][:, :, router_cols] for peer in peers]
            + [peer["moe_in_proj"][:, :, latent_cols] for peer in peers],
            axis=2,
        )
    )
    weights["moe_in_proj"] = np.ascontiguousarray(own["moe_in_proj"][:, :, LOCAL_EXPERTS + LATENT_SHARD :])
    weights["moe_vectors"] = np.ascontiguousarray(
        np.concatenate([own["moe_vectors"]] + [peer["moe_vectors"][:, router_cols] for peer in peers], axis=1)
    )
    return weights


def latent_moe_column_rank_weights(
    rank: int,
    router_weight,
    router_bias,
    latent_down_weight,
    latent_norm_weight,
    latent_up_weight,
    shared_gate_weight,
    shared_up_weight,
    shared_down_weight,
    expert_weights: Optional[Mapping[str, np.ndarray]] = None,
    in_proj_form: str = "gathered",
) -> dict[str, np.ndarray]:
    """Build one rank's latent MoE kernel weights for ``out_proj_form`` ``column`` (``tkg_moe.OUT_PROJ_FORMS``).

    The arrays of the in-projection form (:func:`latent_moe_rank_weights` or, with ``dp_local``,
    :func:`latent_moe_dp_local_rank_weights`), with two changes:

    - ``moe_out_proj`` [76, 7168] BF16 holds [128, 76, 56] (``tkg_moe.COLUMN_K_TILES``): element (p, c, j) is
      ``latent_up_weight[128 j + g, 28 p + c]`` for c < 28 and ``shared_down_weight[128 j + g, 48 p + c - 28]`` for
      c >= 28: the weights of all 9,728 inputs to the hidden columns 128 j + g of rank g. The bytes per rank do not
      change.
    - ``moe_vectors`` FP32 adds the 3,584 latent norm weights (the BF16 values, as the 28 of the rank) after the columns
      of the in-projection form: [1, 35 + 3584] (``gathered``) or [1, 63 + 3584] (``dp_local``).

    Args:
        rank: Global rank g.
        router_weight .. shared_down_weight: global weights, as in :func:`latent_moe_rank_weights`.
        expert_weights: optional expert arrays of this rank, passed through unchanged.
        in_proj_form: ``gathered`` or ``dp_local`` (``tkg_moe.IN_PROJ_FORMS``).
    """
    global_weights = {
        "router_weight": router_weight,
        "router_bias": router_bias,
        "latent_down_weight": latent_down_weight,
        "latent_norm_weight": latent_norm_weight,
        "latent_up_weight": latent_up_weight,
        "shared_gate_weight": shared_gate_weight,
        "shared_up_weight": shared_up_weight,
        "shared_down_weight": shared_down_weight,
    }
    if in_proj_form == "dp_local":
        weights = latent_moe_dp_local_rank_weights(rank, expert_weights=expert_weights, **global_weights)
    elif in_proj_form == "gathered":
        weights = latent_moe_rank_weights(rank, expert_weights=expert_weights, **global_weights)
    else:
        raise ValueError(f"in_proj_form must be 'gathered' or 'dp_local', got {in_proj_form!r}")
    hidden_tiles = _to_torch(latent_up_weight).shape[0] // WORLD_SIZE
    # Rows 128 j + g of the [out, in] weights: [56, in], then input i = (p, c) -> [128, c, 56].
    latent_rows = _to_torch(latent_up_weight)[rank::WORLD_SIZE].reshape(hidden_tiles, WORLD_SIZE, LATENT_SHARD)
    shared_rows = _to_torch(shared_down_weight)[rank::WORLD_SIZE].reshape(hidden_tiles, WORLD_SIZE, SHARED_SHARD)
    columns = torch.cat([latent_rows.permute(1, 2, 0), shared_rows.permute(1, 2, 0)], dim=1)
    weights["moe_out_proj"] = _bf16_numpy(columns.reshape(COLUMN_K_TILES, hidden_tiles * WORLD_SIZE))
    norm_weight = _to_torch(latent_norm_weight).to(torch.bfloat16).float().reshape(1, -1).numpy()
    weights["moe_vectors"] = np.ascontiguousarray(
        np.concatenate([weights["moe_vectors"], norm_weight.astype(np.float32)], axis=1)
    )
    return weights


def router_weight_fp16_rank(rank: int, router_weight) -> np.ndarray:
    """Return the FP16 router rows of rank g for the FP16 router (``moe_router_weight``), hT-tiled [128, 56, 7].

    Element (p, j, e) is ``fp16(router_weight[7 g + e, 128 j + p])``: the router columns of ``moe_in_proj``,
    cast to FP16.
    """
    experts = slice(rank * LOCAL_EXPERTS, (rank + 1) * LOCAL_EXPERTS)
    rows = _to_torch(router_weight)[experts].T.to(torch.float16)
    return _hT_tiled(rows).contiguous().numpy()


def router_weight_fp16_local_rank(rank: int, router_weight) -> np.ndarray:
    """Return the FP16 router rows of rank g for ``fp16_local`` (``moe_router_weight``), hT-tiled [128, 56, 28].

    Rank g = 32 r + t gets the rows of the 7 experts of each of its 4 DP peers 32 d + t, in DP rank order:
    element (p, j, 7 d + e) is ``fp16(router_weight[7 (32 d + t) + e, 128 j + p])``.
    """
    tp_rank = rank % TP_SIZE
    experts = [LOCAL_EXPERTS * (TP_SIZE * d + tp_rank) + e for d in range(DP_SIZE) for e in range(LOCAL_EXPERTS)]
    rows = _to_torch(router_weight)[experts].T.to(torch.float16)
    return _hT_tiled(rows).contiguous().numpy()


def dense_mlp_rank_weights(rank: int, gate_weight, up_weight, down_weight) -> dict[str, np.ndarray]:
    """Build one rank's dense MLP kernel weights (``DENSE_WEIGHT_NAMES``) from the global weights.

    Args:
        rank: Global rank g. It owns intermediate columns [264 g, 264 g + 264).
        gate_weight, up_weight, down_weight: global weights, as in :func:`dense_mlp_decode_torch_ref`.

    Returns:
        ``dense_gate_up`` [128, 56, 528] BF16 (hT-tiled) and ``dense_down`` [264, 7168] BF16.
    """
    shard = slice(rank * DENSE_SHARD, (rank + 1) * DENSE_SHARD)
    gate_up = torch.cat([_to_torch(gate_weight)[shard].T, _to_torch(up_weight)[shard].T], dim=1)
    return {
        "dense_gate_up": _bf16_numpy(_hT_tiled(gate_up)),
        "dense_down": _bf16_numpy(_to_torch(down_weight)[:, shard].T),
    }


def rank_info_array(rank: int) -> np.ndarray:
    """Return the INT32 [1, 4] ``rank_info`` (global rank, DP rank, TP rank, 0) of a global rank."""
    return np.array([[rank, rank // TP_SIZE, rank % TP_SIZE, 0]], dtype=np.int32)


def _hT_tiled(weight_in_out: torch.Tensor) -> torch.Tensor:
    """Tile an [H, N] ``[in, out]`` weight as [128, H/128, N]: element (p, j, n) is row j * 128 + p."""
    rows, cols = weight_in_out.shape
    return weight_in_out.reshape(rows // 128, 128, cols).permute(1, 0, 2).contiguous()


def _situ(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """``bf16(4 tanh(g/4) sigmoid(g) * 25 tanh(u/25))`` in FP32."""
    gate_f32 = gate.float()
    up_f32 = up.float()
    gate_term = SITU_BETA * torch.tanh(gate_f32 / SITU_BETA) * torch.sigmoid(gate_f32)
    up_term = SITU_LINEAR_BETA * torch.tanh(up_f32 / SITU_LINEAR_BETA)
    return (gate_term * up_term).to(torch.bfloat16)


# Ranks in each einsum of _sum_of_rank_partials: [16, T, 7168] FP32 is 7.3 MB at T = 16.
_RANK_CHUNK = 16


def _sum_of_rank_partials(terms: list[tuple[torch.Tensor, torch.Tensor, int]]) -> torch.Tensor:
    """Return ``bf16(sum_g bf16(sum_terms a[:, shard g] @ W[:, shard g]^T))``, shape [T, out].

    Each term is (activation [T, n], weight [out, n], shard). Rank g owns columns
    [shard g, shard (g + 1)) of each term; its partial is the FP32 sum of its terms, rounded once
    to BF16. The ranks run in chunks, so the memory stays bounded for any T.
    """
    tokens = terms[0][0].shape[0]
    out_features = terms[0][1].shape[0]
    total = torch.zeros((tokens, out_features), dtype=torch.float32)
    for first_rank in range(0, WORLD_SIZE, _RANK_CHUNK):
        ranks = slice(first_rank, first_rank + _RANK_CHUNK)
        partial = torch.zeros((_RANK_CHUNK, tokens, out_features), dtype=torch.float32)
        for activation, weight, shard in terms:
            activation_ranks = activation.float().reshape(tokens, WORLD_SIZE, shard)[:, ranks]
            weight_ranks = weight.float().reshape(out_features, WORLD_SIZE, shard)[:, ranks]
            partial += torch.einsum("tgc,ogc->gto", activation_ranks, weight_ranks)
        total += partial.to(torch.bfloat16).float().sum(dim=0)
    return total.to(torch.bfloat16)


def _bf16_numpy(value: torch.Tensor) -> np.ndarray:
    """Convert a torch tensor to a contiguous BF16 numpy array."""
    bits = value.to(torch.bfloat16).contiguous().view(torch.int16).numpy()
    return bits.view(ml_dtypes.bfloat16)


def _to_torch(value) -> torch.Tensor:
    """Convert a numpy array (BF16 included) or a torch tensor to a torch tensor."""
    if isinstance(value, torch.Tensor):
        return value
    array = np.asarray(value)
    if array.dtype == ml_dtypes.bfloat16:
        return torch.from_numpy(np.ascontiguousarray(array).view(np.int16)).view(torch.bfloat16)
    return torch.from_numpy(np.ascontiguousarray(array))
