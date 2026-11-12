# SPDX-License-Identifier: Apache-2.0
import math

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed._functional_collectives import reduce_scatter_tensor

from sglang_neuron.parallel import collectives as neuron_collectives


class VocabDimShardedEmbedding(nn.Module):
    """An embedding sharded by vocabulary over ``tp_group``: each rank holds the rows of one contiguous id range,
    padded to ``ceil(vocab / tp)`` rows, and looks up only the ids in it. The ranks' partial results are summed. The
    owner sets the weight's loader."""

    def __init__(self, vocab_size: int, embed_dim: int, *, dtype, tp_group):
        super().__init__()
        self.tp_group = tp_group
        self.tp_size = dist.get_world_size(tp_group) if dist.is_initialized() else 1
        self.vocab_size = vocab_size
        self.vocab_size_per_rank = math.ceil(vocab_size / self.tp_size)
        self.weight = nn.Parameter(torch.empty(self.vocab_size_per_rank, embed_dim, dtype=dtype))

    def forward(self, ids: torch.Tensor, *, scatter_tokens: bool, rank: torch.Tensor) -> torch.Tensor:
        """The embeddings of ``ids`` [T]: this rank's sequence shard [T / tp, D] with ``scatter_tokens``, else all
        [T, D]. ``rank`` is the group's rank tensor, from which the id range is computed so that every rank traces one
        graph."""
        start = rank * self.vocab_size_per_rank
        end = torch.clamp(start + self.vocab_size_per_rank, max=self.vocab_size)
        mine = (ids >= start) & (ids < end)
        # clamped as well as masked: the lookup may be issued before the mask resolves
        local = torch.where(mine, (ids - start).clamp(0, self.vocab_size_per_rank - 1), 0)
        rows = F.embedding(local, self.weight)
        out = torch.where(mine.unsqueeze(-1).expand_as(rows).contiguous(), rows, torch.zeros_like(rows))
        if self.tp_size == 1:
            return out
        if scatter_tokens:
            return reduce_scatter_tensor(out, "sum", scatter_dim=0, group=self.tp_group)
        return neuron_collectives.all_reduce_pg(self.tp_group, out, "sum")
