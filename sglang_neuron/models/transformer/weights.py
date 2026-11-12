# SPDX-License-Identifier: Apache-2.0
"""The checkpoint loaders of the dense transformer: each builds this attention-TP rank's shard of a parameter from
the checkpoint's Hugging Face tensors, in the layout the graphs read.

Attention heads shard over the attention TP. With more ranks than KV heads each KV head is replicated: rank r holds KV
head ``r // (tp / kv_heads)``, the one its query heads group over."""

import torch

from sglang_neuron.utils.weight_loader import SafetensorsWeightLoader, pad_to_shape


def kv_rank(rank: int, tp_size: int, num_kv_heads: int) -> int:
    """The KV shard a rank holds: its own with at least one KV head per rank, else the one its query heads use."""
    return rank if num_kv_heads >= tp_size else rank // (tp_size // num_kv_heads)


def _qkv_rows(slices: list, rank: int, tp_size: int, q_rows: int, kv_rows: int, num_kv_heads: int) -> list:
    """This rank's Q rows and its KV shard's K and V rows (dim 0 of each of the three checkpoint tensors)."""
    q, k, v = slices
    kv = kv_rank(rank, tp_size, num_kv_heads)
    return [q[rank * q_rows:(rank + 1) * q_rows], k[kv * kv_rows:(kv + 1) * kv_rows], v[kv * kv_rows:(kv + 1) * kv_rows]]


def fused_qkv_weight_loader(tp_size: int, q_rows: int, kv_rows: int, num_kv_heads: int) -> SafetensorsWeightLoader:
    """q_proj, k_proj and v_proj ([out, H] each) as one [H, q + 2 kv], Q, K and V along the output dim."""
    return SafetensorsWeightLoader(transform=lambda slices, rank: torch.cat(
        _qkv_rows(slices, rank, tp_size, q_rows, kv_rows, num_kv_heads), dim=0).t())


def fused_qkv_bias_loader(tp_size: int, q_rows: int, kv_rows: int, num_kv_heads: int) -> SafetensorsWeightLoader:
    return SafetensorsWeightLoader(transform=lambda slices, rank: torch.cat(
        _qkv_rows(slices, rank, tp_size, q_rows, kv_rows, num_kv_heads), dim=0))


def row_parallel_loader(columns: int) -> SafetensorsWeightLoader:
    """A row-parallel projection ([H, in], o_proj or down_proj): this rank's input columns, as [in / tp, H]."""
    return SafetensorsWeightLoader(transform=lambda slices, rank: slices[0][:, rank * columns:(rank + 1) * columns].t())


def gate_up_loader(rows: int, fused: bool) -> SafetensorsWeightLoader:
    """The MLP's gate and up projections as one [H, 2 * I / tp], this rank's gate columns then its up columns. A fused
    checkpoint holds them as one [2 I, H], gate first."""

    def transform(slices: list, rank: int) -> torch.Tensor:
        if fused:
            (gate_up,) = slices
            full = gate_up.get_shape()[0] // 2
            gate = gate_up[rank * rows:(rank + 1) * rows]
            up = gate_up[full + rank * rows:full + (rank + 1) * rows]
        else:
            gate, up = (s[rank * rows:(rank + 1) * rows] for s in slices)
        return torch.cat([gate, up], dim=0).t()

    return SafetensorsWeightLoader(transform=transform)


def vocab_rows_loader(rows: int, hidden_size: int) -> SafetensorsWeightLoader:
    """A vocabulary-sharded [V, H] table (the embedding, the LM head): this rank's rows, the last shard padded."""
    return SafetensorsWeightLoader(transform=lambda slices, rank: pad_to_shape(
        slices[0][rank * rows:(rank + 1) * rows], (rows, hidden_size)))
