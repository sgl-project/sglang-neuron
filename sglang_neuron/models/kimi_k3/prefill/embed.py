# SPDX-License-Identifier: Apache-2.0
"""The vocab-sharded token embedding of the Kimi-K3 prefill (the first stage), in the sequence-parallel form.

Rank ``r`` holds the embedding rows ``r V / W .. (r + 1) V / W - 1``. Each rank looks up every token in its slice and
puts zero rows for the tokens outside it; a reduce-scatter by row adds the parts (exactly one rank holds each token,
so the sum is exact) and gives each rank its row block. The rank enters as tensor data, so every rank traces one graph.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class VocabEmbedding(nn.Module):
    """The embedding slice ``weight [V / W, H]`` of one rank.

    Example:
        >>> embed = VocabEmbedding(weight)
        >>> rows = embed(input_ids, comm)  # [T] int64 on every rank -> [T / W, H]
    """

    def __init__(self, weight: Tensor):
        super().__init__()
        self.register_buffer("weight", weight)
        self.vocab_local = int(weight.shape[0])

    def forward(self, input_ids: Tensor, comm, row_sharded: bool = True) -> Tensor:
        """Look up ``input_ids [T]`` and give the row block ``[T / W, H]`` (all rows with ``row_sharded=False``)."""
        rank = comm.rank_tensor.reshape(-1)[:1].to(torch.int64)
        local = input_ids.to(torch.int64) - rank * self.vocab_local
        valid = (local >= 0) & (local < self.vocab_local)
        emb = F.embedding(local.clamp(0, self.vocab_local - 1), self.weight)
        emb = emb * valid.unsqueeze(-1).to(emb.dtype)
        return comm.reduce_scatter(emb, 0) if row_sharded else comm.all_reduce(emb)
