# SPDX-License-Identifier: Apache-2.0
"""Block-sparse decode attention for Hy3 under DCP (``SGLANG_NEURON_HY3_SPARSE_DECODE=<fraction>``), Quest-style.

Each layer keeps, next to its FP8 cache, a summary of every local KV page: the per-dimension min and max of the page's
K (quantized units: the layer's one K scale is positive, so it does not change the ranking). The prefill writes the
summaries of the pages its chunk fills on this rank; every decode step folds its token into the summary of the page
it lands in, on the token's owner. Every ``SELECT_EVERY`` layers a decode step scores each row's local pages with the bound

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
# SGLANG_NEURON_HY3_SPARSE_SELECT_EVERY: layers 0, R, 2R, ... select, the rest reuse the selection (8, as the vllm-neuron
# Hy3 reference configuration's HY3_SPARSE_SELECT_EVERY)
SELECT_EVERY = int(os.environ.get("SGLANG_NEURON_HY3_SPARSE_SELECT_EVERY", "8") or 8)
if SELECT_EVERY < 1:
    raise ValueError(f"SGLANG_NEURON_HY3_SPARSE_SELECT_EVERY={SELECT_EVERY}: set a layer count of at least 1")
SINK_PAGES = 1
RECENT_PAGES = 2
# SGLANG_NEURON_HY3_SPARSE_PREFILL_KEEP=<tokens>: a DCP prefill chunk over a longer prior attends to that many of its
# prior tokens (whole virtual pages, a multiple of 8 local pages: the prefill kernel's prior segment), picked by the same
# bound with the chunk's queries; 0 (default) attends to the whole prior
PREFILL_KEEP = int(os.environ.get("SGLANG_NEURON_HY3_SPARSE_PREFILL_KEEP", "0") or 0)
# SGLANG_NEURON_HY3_SPAN_K=cache folds a decode span into the summaries from its K rows read back from the cache
# (add_span); the default computes the span's K beside the attention kernel (fold_span)
SPAN_K = os.environ.get("SGLANG_NEURON_HY3_SPAN_K", "computed")
if SPAN_K not in ("computed", "cache"):
    raise ValueError(f"SGLANG_NEURON_HY3_SPAN_K={SPAN_K!r}: computed or cache")
# the chunk's queries score the prior pages in groups of this many (prefill_select)
PREFILL_QUERY_GROUP = 128
# the kept pages are whole quanta of 32 pages (4096 tokens): the kernel reads the packed FP8 K cache in tiles of
# 2 x LNC x P_MAX tokens (4 pages), and a row of s_prior tokens in blocks of gcd(page, s_prior / 256), 16 at 4096
# (runtime.LOCAL_ALIGN)
PAGE_QUANTUM = 32
_EMPTY = 3.0e38
# (layer name, its K cache's address) -> the layer's summaries, shared by the prefill and decode graphs
_SUMMARIES: dict = {}
# (K cache, its summaries) in bind order, for the PD transfer's side buffers (transfer_summaries)
_BOUND: list = []


def enabled() -> bool:
    return FRACTION > 0


def pd_prefill() -> bool:
    """Whether this launch is a PD prefill: without DCP it keeps the summaries only to send them to its DCP decode."""
    from sglang.srt.runtime_context import get_server_args

    return str(getattr(get_server_args(), "disaggregation_mode", "null")) == "prefill"


def transfer_summaries(k_buffers) -> list:
    """(pool layer index, summaries) of the layers whose K cache is one of the pool's ``k_buffers``, in order: a PD
    transfer registers them next to K and V (memory_pool.page_side_buffers), page ``p`` at ``p * 2 * d`` elements, so a
    DCP decode receives each page's summary with its page."""
    out = []
    for i, k in enumerate(k_buffers):
        # data_ptr() is a placeholder on torch-neuronx, but each allocation's own: the cache tensor itself first
        hit = [summary for cache, summary in _BOUND if cache is k] or [
            summary for cache, summary in _BOUND if cache.data_ptr() == k.data_ptr()]
        if hit:
            out.append((i, hit[0]))
    return out


def _register_side_buffers() -> None:
    from sglang_neuron.mem_cache import memory_pool

    provider = lambda pool: transfer_summaries(pool.k_buffer)  # noqa: E731
    provider.__name__ = "hy3_sparse_summaries"
    if not any(getattr(p, "__name__", "") == provider.__name__ for p in memory_pool.PAGE_SIDE_BUFFERS):
        memory_pool.PAGE_SIDE_BUFFERS.append(provider)


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
            _BOUND.append((k_cache, _SUMMARIES[key]))
            _register_side_buffers()
        self.summary = _SUMMARIES[key]

    def query(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """[B, q_heads, d] fp32 decode queries after the QK-norm and RoPE, from x [B, H] (the attention's input)."""
        q = (x.to(torch.bfloat16) @ self.wq).float().view(x.shape[0], self.q_heads, self.head_dim)
        q = q * torch.rsqrt(q.pow(2).mean(-1, keepdim=True) + self.eps) * self.q_norm.float()
        return self.rotate(q, positions)

    def rotate(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """RoPE of x [n, heads, d] at ``positions`` [n] (the halves rotated, the decode kernel's contiguous layout)."""
        angles = positions.float().view(-1, 1) * self.inv_freq.view(1, -1)
        cos, sin = angles.cos().unsqueeze(1), angles.sin().unsqueeze(1)
        half = self.head_dim // 2
        a, b = x[..., :half], x[..., half:]
        return torch.cat([a * cos - b * sin, b * cos + a * sin], dim=-1)

    def refresh(self, k_cache: torch.Tensor, blocks: torch.Tensor, valid: torch.Tensor) -> None:
        """Recompute the summaries of local pages ``blocks`` [n] (block ids; -1 for none) from the cache."""
        blocks = torch.where(blocks < 0, torch.zeros_like(blocks), blocks).to(torch.int64)
        summ = page_summary(k_cache.index_select(0, blocks).squeeze(1), valid)
        self.summary.index_copy_(0, blocks, summ)

    def add_token(self, k_cache: torch.Tensor, blocks: torch.Tensor, slot: torch.Tensor) -> None:
        """Fold each row's decoded token, at ``slot`` [B] of local page ``blocks`` [B] (-1 for none), into its page's
        summary: the page's min and max so far with the token's K, or the token's K alone at a page's first slot. It
        reads the token's K row of the packed cache, not the page (refresh), which at a batch of 16 cost most of a
        decode step."""
        blocks = torch.where(blocks < 0, torch.zeros_like(blocks), blocks).to(torch.int64)
        slot = slot.to(torch.int64)
        half, d = k_cache.shape[2], k_cache.shape[3]
        pairs = k_cache.reshape(-1, d, 2).index_select(0, blocks * half + torch.div(slot, 2, rounding_mode="floor"))
        k = pairs.gather(2, (slot % 2).view(-1, 1, 1).expand(-1, d, 1)).squeeze(2).to(torch.float32)
        old = self.summary.index_select(0, blocks).to(torch.float32)
        first = (slot == 0).view(-1, 1)
        kmin = torch.where(first, k, torch.minimum(old[:, 0], k))
        kmax = torch.where(first, k, torch.maximum(old[:, 1], k))
        self.summary.index_copy_(0, blocks, torch.stack([kmin, kmax], dim=1).to(torch.bfloat16))


    def add_span(self, k_cache: torch.Tensor, table: torch.Tensor, local_pos: torch.Tensor, owner: torch.Tensor,
                 page: int) -> None:
        """Fold each row's span tokens this rank keeps (``owner`` [B, S] > 0, at local positions ``local_pos`` [B, S];
        ``table`` [B, pages] the rows' local pages) into their page's summary, in one pass. A rank keeps a run of
        consecutive tokens of a span, inside one page (a span is shorter than a DCP block, and the next block is
        another rank's), so each row folds the min and max of its run into one page, starting the page afresh when
        the run begins at its first slot. A row that keeps none rewrites block 0 with itself, as add_token's -1."""
        rows, span = local_pos.shape
        n = local_pos.to(torch.int64)
        own = owner.reshape(rows, span) > 0
        idx = torch.div(n, page, rounding_mode="floor")
        blocks = table.to(torch.int64).gather(1, torch.clamp(idx, max=table.shape[1] - 1))
        blocks = torch.where(own, blocks, torch.full_like(blocks, -1))
        slot = n - idx * page
        half, d = k_cache.shape[2], k_cache.shape[3]
        flat = k_cache.reshape(-1, d, 2)
        # each token's K row read on its own, as add_token's (one gather of all the span's rows at once hit
        # neuronx-cc NCC_ILLC051, a CoreBarrier mismatch between the logical core's two cores); the span folds
        # elementwise over its columns
        kmin = kmax = None
        row_block = torch.full_like(blocks[:, 0], -1)
        first = torch.zeros_like(own[:, 0])
        for j in range(span):
            at = torch.clamp(blocks[:, j], min=0) * half + torch.div(slot[:, j], 2, rounding_mode="floor")
            pairs = flat.index_select(0, at)
            kj = pairs.gather(2, (slot[:, j] % 2).view(-1, 1, 1).expand(-1, d, 1)).squeeze(2).to(torch.float32)
            oj = own[:, j].view(-1, 1)
            if kmin is None:
                kmin = torch.full_like(kj, _EMPTY)
                kmax = torch.full_like(kj, -_EMPTY)
            kmin = torch.where(oj, torch.minimum(kmin, kj), kmin)
            kmax = torch.where(oj, torch.maximum(kmax, kj), kmax)
            row_block = torch.maximum(row_block, blocks[:, j])
            first = first | (own[:, j] & slot[:, j].eq(0))
        row_block = torch.clamp(row_block, min=0)
        first = first.view(-1, 1)
        old = self.summary.index_select(0, row_block).to(torch.float32)
        kmin = torch.where(first, kmin, torch.minimum(old[:, 0], kmin))
        kmax = torch.where(first, kmax, torch.maximum(old[:, 1], kmax))
        self.summary.index_copy_(0, row_block, torch.stack([kmin, kmax], dim=1).to(torch.bfloat16))


    def fold_span(self, k: torch.Tensor, table: torch.Tensor, local_pos: torch.Tensor, owner: torch.Tensor,
                  page: int) -> None:
        """add_span from the span tokens' K values ``k`` [B, S, d] (fp32, as the cache stores them: _span_k) instead of
        their rows read back from the cache: a read of one K row is a gather of its own, and add_span's S of them a
        layer were about 8 ms of a 47 ms decode step (bs 8, MTP span 4, 80 layers)."""
        rows, span = local_pos.shape
        n = local_pos.to(torch.int64)
        own = owner.reshape(rows, span) > 0
        idx = torch.div(n, page, rounding_mode="floor")
        blocks = table.to(torch.int64).gather(1, torch.clamp(idx, max=table.shape[1] - 1))
        blocks = torch.where(own, blocks, torch.full_like(blocks, -1))
        slot = n - idx * page
        mine = own.unsqueeze(-1)
        kmin = torch.where(mine, k, _EMPTY).amin(dim=1)
        kmax = torch.where(mine, k, -_EMPTY).amax(dim=1)
        row_block = torch.clamp(blocks.amax(dim=1), min=0)
        first = (own & slot.eq(0)).any(dim=1).view(-1, 1)
        old = self.summary.index_select(0, row_block).to(torch.float32)
        kmin = torch.where(first, kmin, torch.minimum(old[:, 0], kmin))
        kmax = torch.where(first, kmax, torch.maximum(old[:, 1], kmax))
        self.summary.index_copy_(0, row_block, torch.stack([kmin, kmax], dim=1).to(torch.bfloat16))


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


def prefill_keep_pages(size: int, page: int) -> int:
    """Local pages a sparse prefill chunk keeps of its prior (0: dense); ``size`` 1 for a prefill without DCP."""
    if PREFILL_KEEP <= 0:
        return 0
    vpage, quantum = size * page, 8
    keep = PREFILL_KEEP // vpage
    if PREFILL_KEEP % vpage or keep % quantum or keep <= SINK_PAGES + RECENT_PAGES:
        raise ValueError(f"SGLANG_NEURON_HY3_SPARSE_PREFILL_KEEP={PREFILL_KEEP}: give a multiple of {vpage * quantum} "
                         f"tokens ({quantum} virtual pages of {vpage})")
    return keep


def prefill_select(state: SparseState, q, table_row, cached, size: int, page: int, keep: int, active: int, dcp_group):
    """The table, query offset and prior length of a DCP prefill chunk that attends to ``keep`` of its local prior
    pages. ``q`` [h, T, d] the chunk's queries on this rank (after the QK-norm and RoPE), ``table_row`` [pages] the
    request's local pages, ``cached`` [1, 1] the chunk's first position (whole virtual pages), ``active`` the pages the
    kernel loads for the chunk itself (its query segment / page).

    The kernel treats the prior as fully visible and masks only the chunk's own pages by position, so the kept pages
    are packed first and the chunk's pages follow, with the chunk's position moved to ``keep`` virtual pages: RoPE is
    already in Q and K, and the mask inside the chunk depends only on positions relative to it. A prior of at most
    ``keep`` pages keeps the request's own table, position and prior (dense). Without DCP (``size`` 1,
    ``dcp_group`` None) each rank scores by its own query heads."""
    pages = table_row.shape[0]
    full = torch.div(cached.reshape(1).to(torch.int64), size * page, rounding_mode="floor")  # local prior pages
    # each group of PREFILL_QUERY_GROUP queries scores by its per-dimension envelope: relu(max q) . max k +
    # min(min q, 0) . min k bounds q . k for every q of the group, so only the envelopes go over the DCP group
    h, t, d = q.shape
    groups = q.float().reshape(h, t // PREFILL_QUERY_GROUP, PREFILL_QUERY_GROUP, d)
    envelope = torch.cat([groups.amax(dim=2), groups.amin(dim=2)], dim=-1).reshape(-1, 2 * d)  # [h * G, 2d]
    if dcp_group is not None:
        envelope = neuron_collectives.all_gather(dcp_group, envelope.contiguous(), dim=0)  # the group's heads
    qmax, qmin = envelope[:, :d], envelope[:, d:]
    blocks = table_row.to(torch.int64)
    summ = state.summary.index_select(0, blocks).float()  # [pages, 2, d]
    bound = (F.relu(qmax) @ summ[:, 1].t() + torch.clamp(qmin, max=0) @ summ[:, 0].t()).amax(dim=0)  # [pages]
    idx = torch.arange(pages, device=table_row.device)
    big = torch.finfo(torch.float32).max
    forced = (idx < SINK_PAGES) | (idx >= full - RECENT_PAGES)
    score = torch.where(idx < full, torch.where(forced, big, bound), -big)
    top = torch.topk(score, keep).indices  # the kept prior pages (any order: the prior is fully visible)
    width = keep + active
    lane = torch.arange(width, device=table_row.device)
    top = torch.cat([top, torch.zeros(active, dtype=top.dtype, device=top.device)])  # lanes past keep: the chunk
    sparse_pick = torch.where(lane < keep, top, full + (lane - keep))
    sparse = full > keep
    pick = torch.clamp(torch.where(sparse, sparse_pick, lane), max=pages - 1)
    table = table_row.gather(0, pick).view(1, width).to(torch.int32)
    vpages = torch.where(sparse, torch.full_like(full, keep), full)
    offset = (vpages * size * page).to(torch.int32).view(1, 1)
    prior = (vpages * page).to(torch.int32).view(1, 1)
    return table, offset, prior
