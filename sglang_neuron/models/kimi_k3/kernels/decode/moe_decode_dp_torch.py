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

"""Host weight layout of the Kimi-K3 latent MoE decode body with DP4 x TP32 projections (``tkg_moe_dp``).

The global torch reference is ``moe_decode_torch.latent_moe_decode_torch_ref``: the sharding does not change it.
"""

from collections.abc import Mapping
from typing import Optional

import numpy as np
import torch

from .tkg_moe import NUM_EXPERTS
from .tkg_moe_dp import DP_EXPERTS, DP_LATENT, DP_SHARED, DP_VECTOR_COLS, TP_SIZE
from .moe_decode_torch import _EXPERT_KEYS, _bf16_numpy, _hT_tiled, _to_torch
from .multinode.tkg_moe_2srv import MX_LATENT_COLS, MX_LATENT_RANKS
from .tkg_mx_quant import MX_LANES
from .mx_weights_torch import mx_quantize, pack_mx_hT

MX_WEIGHT_CLASSES = ("moe_in_proj", "moe_out_proj")
"""MXFP8 weight classes of ``latent_moe_dp_rank_weights`` (``mx_weights``): the latent down and the shared gate / up
columns of ``moe_in_proj_dp``, with the FP16 router (``tkg_moe_2srv.MX_IN_PROJ_COLS``), and the padded-lane
``moe_out_proj_dp`` (``tkg_moe_2srv.MX_OUT_PROJ_SHAPES``)."""
MX_WEIGHT_ORDERS = {"moe_in_proj": ("hT_quad",), "moe_out_proj": ("hT_quad",)}
"""The group orders of each class of ``MX_WEIGHT_CLASSES``."""
MX_FROM_BF16 = ("moe_in_proj", "moe_out_proj")
"""The classes of ``MX_WEIGHT_CLASSES`` that the builder quantizes itself from the BF16 weights."""


def latent_moe_dp_rank_weights(
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
    latent_gather_mx: bool = False,
) -> dict[str, np.ndarray]:
    """Build one rank's DP4 x TP32 latent MoE weights from the global weights (``weight[out, in]`` orientation).

    Args:
        rank: Global rank g = 32 r + t. TP rank t owns router rows [28 t, 28 t + 28), latent columns
            [112 t, 112 t + 112), and shared columns [192 t, 192 t + 192). Rank g owns experts [7 g, 7 g + 7).
        router_weight .. shared_down_weight: global weights, as in ``latent_moe_decode_torch_ref``.
        expert_weights: optional packed MXFP4 arrays of this rank's 7 experts; passed through unchanged.
        mx_weights: ``{"moe_in_proj": "hT_quad"}`` builds the MXFP8 ``moe_in_proj_dp`` (the 496 latent down and shared
            gate / up columns, no router columns, or 512 with ``latent_gather_mx``) and its ``moe_in_proj_scale``
            (``tkg_mx_quant``) from the BF16 weights. The FP16 router gives the logits (``moe_router_weight``).
            ``{"moe_out_proj": "hT_quad"}`` builds the MXFP8 padded-lane ``moe_out_proj_dp`` (the 304 rows in the lanes
            of 512, the rows past them zero) and its ``moe_out_proj_scale``.
        latent_gather_mx: False (default) keeps the 112-column latent-down shard. True gives TP rank t < 28 the
            128-column block ``latent_down_weight[128 t : 128 t + 128]`` (zeros for t >= 28), so ``moe_in_proj_dp`` has
            the 540 columns of the block layout (``tkg_moe_2srv.G1A_IN_PROJ_COLS``), or 512 in the MXFP8 form. Only
            those 128 global latent-down rows are read: a zero global latent-down in which only rows [128 t, 128 t + 128)
            are filled gives the same weights as the full global latent-down.

    Returns:
        ``moe_in_proj_dp`` [128, 56, 524] BF16 hT-tiled (540 with ``latent_gather_mx``), ``moe_out_proj_dp``
        [304, 7168] BF16, ``moe_vectors_dp`` [1, 140] FP32, and the expert arrays when given.
    """
    tp_rank = rank % TP_SIZE
    experts = slice(tp_rank * DP_EXPERTS, (tp_rank + 1) * DP_EXPERTS)
    latent = slice(tp_rank * DP_LATENT, (tp_rank + 1) * DP_LATENT)
    shared = slice(tp_rank * DP_SHARED, (tp_rank + 1) * DP_SHARED)
    down = _to_torch(latent_down_weight)
    if latent_gather_mx:
        if tp_rank < MX_LATENT_RANKS:
            latent_cols = down[tp_rank * MX_LATENT_COLS : (tp_rank + 1) * MX_LATENT_COLS].T
        else:
            latent_cols = torch.zeros((down.shape[1], MX_LATENT_COLS), dtype=down.dtype)
    else:
        latent_cols = down[latent].T
    in_proj = torch.cat(
        [
            _to_torch(router_weight)[experts].T,
            latent_cols,
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
    vectors = torch.cat([bias[experts], norm_weight[latent]]).reshape(1, DP_VECTOR_COLS)
    mx = dict(mx_weights or {})
    for name, order in mx.items():
        if order not in MX_WEIGHT_ORDERS.get(name, ()):
            raise ValueError(f"MXFP8 class {name!r} with order {order!r} is not in {MX_WEIGHT_ORDERS}")
    weights = {
        "moe_in_proj_dp": _bf16_numpy(_hT_tiled(in_proj)),
        "moe_out_proj_dp": _bf16_numpy(out_proj),
        "moe_vectors_dp": vectors.numpy().astype(np.float32),
    }
    if "moe_in_proj" in mx:
        data, scale = mx_quantize(in_proj[:, DP_EXPERTS:].T, mx["moe_in_proj"])
        weights["moe_in_proj_dp"], weights["moe_in_proj_scale"] = pack_mx_hT(data.T, scale.T)
    if "moe_out_proj" in mx:
        padded = torch.zeros((MX_LANES * 128, out_proj.shape[1]), dtype=torch.float32)
        padded[: out_proj.shape[0]] = out_proj.float()
        data, scale = mx_quantize(padded.T, mx["moe_out_proj"])
        weights["moe_out_proj_dp"], weights["moe_out_proj_scale"] = pack_mx_hT(data.T, scale.T)
    if expert_weights is not None:
        for key in _EXPERT_KEYS:
            weights[key] = expert_weights[key]
    return weights
