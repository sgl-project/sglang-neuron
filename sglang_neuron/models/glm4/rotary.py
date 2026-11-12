# SPDX-License-Identifier: Apache-2.0
"""GLM-4's partial rotary embedding in the half-split layout the attention kernels rotate.

GLM-4 rotates only the first ``rotary_dim`` channels of each head, as interleaved pairs (2j, 2j + 1), and passes the
rest through. The kernels rotate channel j with channel j + head_dim / 2. Permuting every head's q and k channels at
load turns one into the other: q.k is unchanged by a permutation applied to both, and v is never rotated. The rotary
pairs land at (j, j + head_dim / 2), and the pass-through channels at pairs the core's rotary cache gives cos 1 and
sin 0."""

from __future__ import annotations

import torch


def half_split_permutation(head_dim: int, rotary_dim: int) -> torch.Tensor:
    """The channel order of one head: [rotary evens | first pass-through half | rotary odds | second half]."""
    if rotary_dim % 2 or not 0 < rotary_dim <= head_dim or head_dim % 2:
        raise ValueError(f"rotary_dim {rotary_dim} must be even and in (0, head_dim {head_dim}]")
    half, pairs = head_dim // 2, rotary_dim // 2
    rest = torch.arange(rotary_dim, head_dim)
    return torch.cat([torch.arange(0, rotary_dim, 2), rest[:half - pairs], torch.arange(1, rotary_dim, 2),
                      rest[half - pairs:]])


def permute_heads(t: torch.Tensor, head_dim: int, perm: torch.Tensor) -> torch.Tensor:
    """``perm`` applied to every head's channels of a [heads * head_dim, ...] weight or bias."""
    return t.reshape(-1, head_dim, *t.shape[1:])[:, perm].reshape(t.shape)
