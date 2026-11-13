# SPDX-License-Identifier: Apache-2.0
"""Block-sparse decode attention for Hy3 under DCP (``SGLANG_NEURON_HY3_SPARSE_DECODE=<fraction>``), Quest-style.

Each layer keeps, next to its FP8 cache, a summary of every local KV page: the per-dimension min and max of the page's
K (quantized units: the layer's one K scale is positive, so it does not change the ranking). The prefill writes the
summaries of the pages its chunk fills on this rank; every decode step refreshes the page its token lands in on the
token's owner. Every ``SELECT_EVERY`` layers a decode step scores each row's local pages with the bound

    max_k q . k <= relu(q) . max + min(q, 0) . min

(the max over the DCP group's query heads, all-gathered), keeps the ``sink`` first and ``recent`` last full pages, the
best of the rest, then the row's last (partial) page and one spare page (the decode kernel keeps the step's token in
the table's last row), and gives the kernel that narrower table: it reads only those pages. The layers until the next
selecting one reuse the table. A bucket whose table is not wider than the kept pages attends densely.

The query for the bound is computed in torch from this module's own copy of the rank's Q weight columns and QK-norm
gamma, with its own RoPE: torch ops reading the tensors the decode kernel reads gave wrong results in the vllm-neuron
port (its docs/SPARSE_DECODE.md)."""

import math
import os

import torch
import torch.nn.functional as F
from torch import nn

from sglang_neuron.ops.attention.attention_decode import unswizzle_packed_k
from sglang_neuron.parallel import collectives as neuron_collectives

FRACTION = float(os.environ.get("SGLANG_NEURON_HY3_SPARSE_DECODE", "0") or 0)
SELECT_EVERY = 4
SINK_PAGES = 1
RECENT_PAGES = 2
# the kernel reads the packed FP8 K cache in tiles of 2 x LNC x P_MAX tokens: 4 pages of 128
PAGE_QUANTUM = 4
_EMPTY = 3.0e38
# (layer name, its K cache's address) -> the layer's summaries, shared by the prefill and decode graphs
_SUMMARIES: dict = {}


def enabled() -> bool:
    return FRACTION > 0


def kept_pages(table_pages: int) -> int:
    """Pages a sparse step attends to for a ``table_pages``-wide decode table; ``table_pages`` (dense) when that is not
    narrower."""
    n = math.ceil(table_pages * FRACTION) + SINK_PAGES + RECENT_PAGES + 2
    n = -(-n // PAGE_QUANTUM) * PAGE_QUANTUM
    return min(n, table_pages)


def page_summary(k_pages: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """[n, 2, d] (min, max) over the first ``valid[i]`` tokens of each packed FP8 K page ``k_pages`` [n, page // 2, d, 2]
    (a page without tokens gets the empty summary, which scores lowest)."""
    k = unswizzle_packed_k(k_pages).to(torch.float32)  # [n, page, d]
    page = k.shape[1]
    live = (torch.arange(page, device=k.device).view(1, page) < valid.view(-1, 1)).unsqueeze(-1)
    kmin = torch.where(live, k, _EMPTY).amin(dim=1)
    kmax = torch.where(live, k, -_EMPTY).amax(dim=1)
    return torch.stack([kmin, kmax], dim=1).to(torch.bfloat16)


def local_count(lengths: torch.Tensor, size: int, rank: torch.Tensor, page: int) -> torch.Tensor:
    """How many of a sequence's first ``lengths`` tokens DCP rank ``rank`` keeps (DcpLayout.local_count, in graph)."""
    lengths = lengths.to(torch.int64)
    vpage = size * page
    full = torch.div(lengths, vpage, rounding_mode="floor")
    rem = lengths - full * vpage
    mine = torch.clamp(rem - rank.to(torch.int64).reshape(-1)[0] * page, min=0, max=page)
    return full * page + mine


class SparseState(nn.Module):
    """One layer's summaries and its own copy of the selection's weights."""

    def __init__(self, hidden: int, q_heads: int, head_dim: int, eps: float, rope_theta: float):
        super().__init__()
        self.q_heads, self.head_dim, self.eps = q_heads, head_dim, eps
        self.wq = nn.Parameter(torch.empty(hidden, q_heads * head_dim, dtype=torch.bfloat16), requires_grad=False)
        self.q_norm = nn.Parameter(torch.empty(head_dim, dtype=torch.bfloat16), requires_grad=False)
        inv = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float() / head_dim))
        self.register_buffer("inv_freq", inv, persistent=False)
        self.summary = None  # [pages, 2, d] bf16, bound with the cache (bind)

    def bind(self, name: str, k_cache: torch.Tensor) -> None:
        """The layer's summaries for its cache, one tensor for both graphs of an aggregated server."""
        key = (name, k_cache.data_ptr())
        if key not in _SUMMARIES:
            empty = torch.tensor([_EMPTY, -_EMPTY], dtype=torch.bfloat16).view(1, 2, 1)
            _SUMMARIES[key] = empty.expand(k_cache.shape[0], 2, self.head_dim).contiguous().to(k_cache.device)
        self.summary = _SUMMARIES[key]

    def query(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """[B, q_heads, d] fp32 decode queries after the QK-norm and RoPE, from x [B, H] (the attention's input)."""
        q = (x.to(torch.bfloat16) @ self.wq).float().view(x.shape[0], self.q_heads, self.head_dim)
        q = q * torch.rsqrt(q.pow(2).mean(-1, keepdim=True) + self.eps) * self.q_norm.float()
        angles = positions.float().view(-1, 1) * self.inv_freq.view(1, -1)
        cos, sin = angles.cos().unsqueeze(1), angles.sin().unsqueeze(1)
        half = self.head_dim // 2
        a, b = q[..., :half], q[..., half:]
        return torch.cat([a * cos - b * sin, b * cos + a * sin], dim=-1)

    def refresh(self, k_cache: torch.Tensor, blocks: torch.Tensor, valid: torch.Tensor) -> None:
        """Recompute the summaries of local pages ``blocks`` [n] (block ids; -1 for none) from the cache."""
        blocks = torch.where(blocks < 0, torch.zeros_like(blocks), blocks).to(torch.int64)
        summ = page_summary(k_cache.index_select(0, blocks).squeeze(1), valid)
        self.summary.index_copy_(0, blocks, summ)


def scores(state: SparseState, x, positions, table, dcp_group) -> torch.Tensor:
    """[B, pages] each local page's bound under the DCP group's query heads; ``table`` [B, pages] holds each row's
    local pages."""
    q = state.query(x, positions)  # [B, h, d], this rank's heads
    q = neuron_collectives.all_gather(dcp_group, q.contiguous(), dim=1)  # the group's heads
    rows, pages = table.shape
    blocks = torch.where(table < 0, torch.zeros_like(table), table).to(torch.int64)
    summ = state.summary.index_select(0, blocks.reshape(-1)).view(rows, pages, 2, -1).float()
    return (torch.einsum("bhd,bnd->bnh", F.relu(q), summ[:, :, 1])
            + torch.einsum("bhd,bnd->bnh", torch.clamp(q, max=0), summ[:, :, 0])).amax(dim=2)


def compact(bound, table, local_prior, page: int, n_keep: int):
    """The kept pages of each row (see the module docstring) and the prior tokens the compacted table holds."""
    rows, pages = table.shape
    n = local_prior.to(torch.int64).view(rows, 1)
    full = torch.div(n, page, rounding_mode="floor")  # whole prior pages
    idx = torch.arange(pages, device=table.device).view(1, pages)
    big = torch.finfo(torch.float32).max
    forced = (idx < SINK_PAGES) | (idx >= full - RECENT_PAGES)
    score = torch.where(idx < full, torch.where(forced, big, bound), -big)
    top = torch.topk(score, n_keep - 2, dim=1).indices
    tail = torch.clamp(torch.cat([full, full + 1], dim=1), max=pages - 1)
    dense = full + 2 <= n_keep
    pick = torch.where(dense, torch.arange(n_keep, device=table.device).view(1, n_keep), torch.cat([top, tail], dim=1))
    kept = table.gather(1, pick)
    prior = torch.where(dense, n, (n_keep - 2) * page + (n - full * page)).view(rows)
    return kept.to(torch.int32), prior
