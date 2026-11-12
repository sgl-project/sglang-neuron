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

"""Host functions of the Kimi-K3 multi-host feed-forward glue: the rank weights, the rank vector and the reference."""

import numpy as np
import torch

from ..moe_decode_torch import _bf16_numpy, _hT_tiled, _situ, _to_torch
from .tkg_ffn_world_cc import ATTN_TP, world_layout

_RANK_CHUNK = 16


def dense_ffn_world_rank_weights(rank: int, world_size: int, gate_weight, up_weight, down_weight) -> dict:
    """Build one rank's dense MLP weights of ``tkg_ffn_world_cc.dense_ffn_world`` from the global weights.

    Args:
        rank: global rank g of the world. It owns the intermediate columns ``[I g, I g + I)``, ``I = 33792 / world``.
        world_size: ranks of the world (256 or 384).
        gate_weight, up_weight: ``[33792, 7168]`` BF16 (``weight[out, in]``).
        down_weight: ``[7168, 33792]`` BF16.

    Returns:
        ``dense_gate_up`` ``[128, 56, 2 I]`` BF16 (hT-tiled: the gate shard, then the up shard) and ``dense_down``
        ``[I, 7168]`` BF16.
    """
    shard = world_layout(world_size)["dense_shard"]
    cols = slice(rank * shard, (rank + 1) * shard)
    gate_up = torch.cat([_to_torch(gate_weight)[cols].T, _to_torch(up_weight)[cols].T], dim=1)
    return {
        "dense_gate_up": _bf16_numpy(_hT_tiled(gate_up)),
        "dense_down": _bf16_numpy(_to_torch(down_weight)[:, cols].T),
    }


def rank_info_world(rank: int) -> np.ndarray:
    """Return the INT32 ``[1, 4]`` ``rank_info`` (global rank, DP rank, TP rank, 0) of a rank of a multi-host world."""
    return np.array([[rank, rank // ATTN_TP, rank % ATTN_TP, 0]], dtype=np.int32)


WORLD_SUMS = ("fp32", "bf16_rank_order")
"""How ``dense_ffn_world_torch_ref`` adds the BF16 rank partials: ``fp32`` (default) in FP32 with one BF16 rounding at
the end (the value of the layer); ``bf16_rank_order`` one partial at a time in rank order with a BF16 rounding after
each add, as the NKI simulator runs a BF16 world all-reduce (``functools.reduce`` of the rank arrays in the replica
group's rank order, in the arrays' dtype)."""


def dense_ffn_world_torch_ref(
    x, gate_weight, up_weight, down_weight, world_size: int, world_sum: str = "fp32"
) -> torch.Tensor:
    """Return the dense SiTU MLP output ``[T, 7168]`` BF16 of the gathered tokens ``x`` ``[T, 7168]``.

    Each rank's down partial is rounded to BF16 before the world sum, as in the kernel. ``world_sum``
    (``WORLD_SUMS``): ``fp32`` (default) adds the partials in FP32; ``bf16_rank_order`` adds them as the simulator's
    BF16 world all-reduce does, so a kernel output on the simulator can be held to a bar of its own roundings.
    """
    assert world_sum in WORLD_SUMS, f"world_sum must be one of {WORLD_SUMS}, got {world_sum!r}"  # noqa: S101
    shard = world_layout(world_size)["dense_shard"]
    hidden = _to_torch(x).to(torch.bfloat16).float()
    gate = (hidden @ _to_torch(gate_weight).float().T).to(torch.bfloat16)
    up = (hidden @ _to_torch(up_weight).float().T).to(torch.bfloat16)
    activation = _situ(gate, up).float()
    down = _to_torch(down_weight).float()
    tokens, out_features = activation.shape[0], down.shape[0]
    total = torch.zeros((tokens, out_features), dtype=torch.float32)
    running = None
    for first in range(0, world_size, _RANK_CHUNK):
        ranks = slice(first, min(world_size, first + _RANK_CHUNK))
        a = activation.reshape(tokens, world_size, shard)[:, ranks]
        w = down.reshape(out_features, world_size, shard)[:, ranks]
        partials = torch.einsum("tgc,ogc->gto", a, w).to(torch.bfloat16)
        if world_sum == "fp32":
            total += partials.float().sum(dim=0)
            continue
        for partial in partials:
            running = partial if running is None else (running.float() + partial.float()).to(torch.bfloat16)
    return total.to(torch.bfloat16) if world_sum == "fp32" else running
