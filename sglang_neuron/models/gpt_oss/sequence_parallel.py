# SPDX-License-Identifier: Apache-2.0
"""The sampled rows of the prefill's sequence-parallel hidden states."""
import torch

from sglang_neuron.parallel import collectives as neuron_collectives


def select_sampled_rows(hidden_local: torch.Tensor, sampling_positions: torch.Tensor, group) -> torch.Tensor:
    """The rows ``sampling_positions`` of the sequence-parallel ``hidden_local`` shards, without gathering them.

    Each rank holds the contiguous rows [rank * L, (rank + 1) * L) of the sequence. Every rank picks, per sampling
    position, its own row at that position's local offset (clamped into its shard), all-gathers those S rows over
    ``group`` and keeps, per position, the row of the rank that owns it. The result equals
    ``all_gather(hidden_local).index_select(0, sampling_positions)`` while moving S rows per rank instead of L. The
    rank offset comes from the group's stamped rank tensor, so every rank traces one graph.
    """
    local_len = hidden_local.shape[0]
    num_samples = sampling_positions.shape[0]
    positions = sampling_positions.to(torch.int64)
    rank_offset = group.rank_in_group_tensor.to(torch.int64) * local_len
    local_index = torch.clamp(positions - rank_offset, min=0, max=local_len - 1)
    rows = hidden_local.index_select(0, local_index)
    gathered = neuron_collectives.all_gather(group, rows, dim=0)  # [world * S, H], rank-major
    owner = positions // local_len
    pick = owner * num_samples + torch.arange(num_samples, device=positions.device, dtype=torch.int64)
    return gathered.index_select(0, pick)
