# SPDX-License-Identifier: Apache-2.0
"""Qwen3-VL vision QKV loaders. The checkpoint fuses ``[Q | K | V]``, so each TP rank takes its
heads of each; a contiguous row slice would give rank 0 only Q.
"""

from __future__ import annotations

import torch

from sglang_neuron.utils.weight_loader import SafetensorsWeightLoader


def vis_qkv_weight_loader(
    num_heads_per_rank: int,
    head_dim: int,
    vis_hidden_size: int,
) -> SafetensorsWeightLoader:
    """Rank ``r``'s rows ``[r*hd, (r+1)*hd)`` of each of Q, K and V of the fused ``[3*H, H]``
    weight, ``hd = num_heads_per_rank * head_dim``, stored transposed as ``[H, 3*hd]``."""
    hd = num_heads_per_rank * head_dim
    H = vis_hidden_size

    def transform(slices, rank):
        w = slices[0][:]  # [3*H, H]
        q = w[rank * hd : (rank + 1) * hd]  # [hd, H]
        k = w[H + rank * hd : H + (rank + 1) * hd]  # [hd, H]
        v = w[2 * H + rank * hd : 2 * H + (rank + 1) * hd]  # [hd, H]
        fused = torch.cat([q, k, v], dim=0)  # [3*hd, H]
        return fused.T.contiguous()  # [H, 3*hd]

    return SafetensorsWeightLoader(transform=transform)


def vis_qkv_bias_loader(
    num_heads_per_rank: int,
    head_dim: int,
    vis_hidden_size: int,
) -> SafetensorsWeightLoader:
    """The ``[3*H]`` bias counterpart of :func:`vis_qkv_weight_loader`, giving ``[3*hd]``."""
    hd = num_heads_per_rank * head_dim
    H = vis_hidden_size

    def transform(slices, rank):
        b = slices[0][:]  # [3*H]
        bq = b[rank * hd : (rank + 1) * hd]
        bk = b[H + rank * hd : H + (rank + 1) * hd]
        bv = b[2 * H + rank * hd : 2 * H + (rank + 1) * hd]
        return torch.cat([bq, bk, bv], dim=0)  # [3*hd]

    return SafetensorsWeightLoader(transform=transform)
