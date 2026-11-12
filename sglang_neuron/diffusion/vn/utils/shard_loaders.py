# SPDX-License-Identifier: Apache-2.0
"""Shard loaders of the diffusion layers, for the plugin's checkpoint loader (``sglang_neuron.utils``).

Each returns a ``SafetensorsWeightLoader`` whose transform reads only this rank's shard of the
checkpoint slices (``get_shard``).
"""

from typing import TYPE_CHECKING

import torch

from sglang_neuron.utils.weight_loader import SafetensorsWeightLoader, get_shard
# the layers' single-tensor shard loader is the plugin's own
from sglang_neuron.utils.weight_loader import sharding_weight_loader  # noqa: F401

if TYPE_CHECKING:
    from safetensors import PySafeSlice


def fused_qkv_weight_loader(
    q_size: int,
    kv_size: int,
    shard_dim: int,
    num_shards: int,
    is_storage_transposed: bool = False,
) -> SafetensorsWeightLoader:
    """This rank's Q, K and V shards (``q_size``, ``kv_size``, ``kv_size`` along ``shard_dim``),
    concatenated along ``shard_dim``; the mapping lists the three checkpoint keys in Q, K, V order.
    ``is_storage_transposed`` as in :func:`sharding_weight_loader`.
    """
    assert shard_dim in (0, 1), f"Shard dim must be 0 or 1, got {shard_dim}"
    storage_dim = 1 - shard_dim if is_storage_transposed else shard_dim

    def transform(slices: list["PySafeSlice"], rank: int) -> torch.Tensor:
        assert len(slices) == 3, "fused_qkv_weight_loader expects [Q, K, V] slices in order"
        for part, slice_obj in zip("QKV", slices):
            assert len(slice_obj.get_shape()) == 2, f"{part} slice must be 2 dimensional"
        parts = [
            get_shard(slice_obj, storage_dim, size, num_shards, rank)
            for slice_obj, size in zip(slices, (q_size, kv_size, kv_size))
        ]
        if is_storage_transposed:
            parts = [t.T for t in parts]
        return torch.cat(parts, dim=shard_dim)

    return SafetensorsWeightLoader(transform=transform)
