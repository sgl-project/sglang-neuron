# SPDX-License-Identifier: Apache-2.0
"""Weight and bias loaders for tensor-parallel (column/row) projections, including the fused QKV
loaders that de-interleave Q/K for the rotate-half RoPE kernels."""

import logging

import torch
import torch.nn.functional as F
from sglang_neuron.diffusion.vn.utils.shard_loaders import (
    fused_qkv_weight_loader,
    sharding_weight_loader,
)
from sglang_neuron.utils.weight_loader import SafetensorsWeightLoader

from sglang_neuron.diffusion.layers.rope_rotate_half import (
    deinterleave_gamma_row,
    deinterleave_qk_columns,
)

logger = logging.getLogger(__name__)


def cast_weight_loader(loader, dtype, name: str, logged_casts: set):
    """Cast a loader's output to dtype, logging each cast once per weight type.

    Args:
        logged_casts: Cast keys already logged, shared across this checkpoint's parameters.
    """
    weight_type = ".".join("*" if part.isdigit() else part for part in name.split("."))

    def transform(slices, rank):
        tensor = loader.load(slices, rank)
        cast_key = (weight_type, tensor.dtype, dtype)
        if tensor.dtype != dtype and cast_key not in logged_casts:
            logged_casts.add(cast_key)
            logger.info(
                "Casting weight %s from checkpoint dtype %s to parameter dtype %s", *cast_key
            )
        return tensor.to(dtype)

    return SafetensorsWeightLoader(transform=transform)


def fused_qkv_bias_loader(q_size, kv_size):
    """Fuse and shard Q, K, V bias tensors (1D variant of fused_qkv_weight_loader)."""

    def transform(slices, rank):
        assert len(slices) == 3
        parts = []
        for sl, size in zip(slices, [q_size, kv_size, kv_size]):
            start = rank * size
            parts.append(sl[start : start + size])
        return torch.cat(parts, dim=0)

    return SafetensorsWeightLoader(transform=transform)


def col_weight_loader(shard_size, num_shards):
    """Column-parallel weight loader: shard output dim (transposed storage)."""
    return sharding_weight_loader(
        shard_dim=1,
        shard_size=shard_size,
        num_shards=num_shards,
        is_storage_transposed=True,
    )


def col_bias_loader(shard_size, num_shards):
    """Column-parallel bias loader: shard along dim 0."""
    return sharding_weight_loader(shard_dim=0, shard_size=shard_size, num_shards=num_shards)


def row_weight_loader(shard_size, num_shards):
    """Row-parallel weight loader: shard input dim (transposed storage)."""
    return sharding_weight_loader(
        shard_dim=0,
        shard_size=shard_size,
        num_shards=num_shards,
        is_storage_transposed=True,
    )


def row_bias_loader(scale, size):
    """Row-parallel bias loader: divide by tp_size for pre-scaled bias pattern.

    Each rank adds bias/tp_size before all_reduce. After the sum-reduce across
    tp_size ranks, the net bias contribution is exactly the original bias.
    """

    def transform(slices, rank):
        assert len(slices) == 1
        tensor = slices[0][:] / scale
        pad_amount = size - tensor.shape[-1]
        if pad_amount > 0:
            tensor = F.pad(tensor, (0, pad_amount))
        return tensor

    return SafetensorsWeightLoader(transform=transform)


def qk_norm_gamma_loader(
    hidden_size: int,
    num_shards: int,
    head_dim: int,
    deinterleave: bool,
) -> SafetensorsWeightLoader:
    """Load a qk-norm (``DistributedRMSNorm``) gamma into the kernel's ``[1, hidden_size]`` row.

    TP-shards on dim 0 exactly like :class:`DistributedRMSNorm`, then shapes the gamma for the fused
    QKV kernel. When ``deinterleave`` (the fused rotate-half RoPE path, self-attention) each head's
    lanes are de-interleaved so the gamma stays aligned with the de-interleaved Q/K weight;
    otherwise (cross-attention, which has no RoPE) it is only row-reshaped. Applied once at
    checkpoint load, so the per-head permutation is not repeated every forward.
    """
    base = sharding_weight_loader(shard_dim=0, shard_size=hidden_size, num_shards=num_shards)
    base_transform = base.transform

    def transform(slices, rank):
        gamma = base_transform(slices, rank)  # [hidden_size], TP-sharded
        if deinterleave:
            return deinterleave_gamma_row(
                gamma, head_dim
            )  # [1, hidden_size], per-head de-interleaved
        return gamma.reshape(1, -1)  # [1, hidden_size]

    return SafetensorsWeightLoader(transform=transform)


def deinterleaved_fused_qkv_weight_loader(
    q_size: int,
    kv_size: int,
    num_shards: int,
    head_dim: int,
) -> SafetensorsWeightLoader:
    """Load the BF16 fused QKV weight like ``fused_qkv_weight_loader``, then de-interleave each Q/K
    head's lanes (the bias counterpart is :func:`deinterleaved_fused_qkv_bias_loader`)."""
    base = fused_qkv_weight_loader(
        q_size=q_size,
        kv_size=kv_size,
        shard_dim=1,
        num_shards=num_shards,
        is_storage_transposed=True,
    )
    base_transform = base.transform

    def transform(slices, rank):
        weight = base_transform(slices, rank)
        return deinterleave_qk_columns(weight, q_size, kv_size, head_dim)

    return SafetensorsWeightLoader(transform=transform)


def deinterleaved_fused_qkv_bias_loader(
    q_size: int,
    kv_size: int,
    head_dim: int,
) -> SafetensorsWeightLoader:
    """Fuse+shard the Q/K/V bias, de-interleaving Q/K per head to match the de-interleaved
    weight (so ``x·W_perm + bias_perm`` stays aligned). V is untouched."""

    def transform(slices, rank):
        assert len(slices) == 3, "fused QKV bias loader expects [qB, kB, vB]"
        parts = []
        for sl, size in zip(slices, [q_size, kv_size, kv_size]):
            start = rank * size
            parts.append(sl[start : start + size])
        bias = torch.cat(parts, dim=0)  # [I]
        return deinterleave_qk_columns(bias, q_size, kv_size, head_dim).contiguous()

    return SafetensorsWeightLoader(transform=transform)
