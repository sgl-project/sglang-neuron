# SPDX-License-Identifier: Apache-2.0
"""Hy4-Flash's arithmetic as plain torch, one function per piece of the reference model (transformers'
``modeling_hy_v4_flash``), shared by the prefill and decode graphs and runnable on the CPU against the reference.

Every function takes rows as the graphs hold them (tokens first, no batch dimension) and computes where the reference
does: the iHC mixes, the IndexPool pooling and scores and the attention softmax in fp32, the projections in bf16.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

NEG = float("-inf")
# top_count_mask splits the threshold interval 16 ways a step up to this many rows, in halves above
WIDE_SEARCH_ROWS = 128


# ---------------------------------------------------------------------------------------------------------------------
# norms
# ---------------------------------------------------------------------------------------------------------------------

def rms_norm(x: torch.Tensor, weight: torch.Tensor | None, eps: float) -> torch.Tensor:
    """HYV4FlashRMSNorm: normalized in fp32, cast back to x's dtype, then times the weight (in the weight's dtype for
    a bf16 weight, as ``weight * hidden.to(input_dtype)``)."""
    dtype = x.dtype
    xf = x.float()
    y = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(dtype)
    if weight is None:
        return y
    return (weight * y) if weight.dtype == dtype else (weight.float() * y.float()).to(dtype)


def torch_rms_norm(x: torch.Tensor, weight: torch.Tensor | None, eps: float | None = None) -> torch.Tensor:
    """``torch.nn.RMSNorm`` (the Engram layer's norms): eps None means the machine epsilon of torch's compute type,
    fp32 for a bf16 input (``at::opmath_type``, not bf16's 2^-7, which would swamp the small residual streams the
    query norms read), the reduction in fp32 and the result in the input's dtype."""
    if eps is None:
        eps = torch.finfo(torch.float32).eps
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    if weight is not None:
        y = y * weight.float()
    return y.to(x.dtype)


# ---------------------------------------------------------------------------------------------------------------------
# RoPE: interleaved pairs (2j, 2j + 1) rotated by frequency j
# ---------------------------------------------------------------------------------------------------------------------

def inv_freq(theta: float, dim: int) -> torch.Tensor:
    return 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))


def rope_tables(positions: torch.Tensor, freq: torch.Tensor) -> tuple:
    """cos, sin [*positions.shape, dim / 2] of each position: computed in fp32 and rounded to bf16 as the reference's
    rotary embedding returns them (``cos.to(x.dtype)``, every rotated tensor being bf16), held in fp32."""
    angles = positions.to(torch.float32)[..., None] * freq.to(positions.device)
    return angles.cos().to(torch.bfloat16).float(), angles.sin().to(torch.bfloat16).float()


def rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Interleaved RoPE of x [..., dim] in fp32, cast back: cos / sin broadcast over x's leading dims as [..., dim/2]
    (the caller unsqueezes the head dimension)."""
    xf = x.float()
    even, odd = xf[..., 0::2], xf[..., 1::2]
    out = torch.stack((even * cos - odd * sin, odd * cos + even * sin), dim=-1).flatten(-2)
    return out.to(x.dtype)


# ---------------------------------------------------------------------------------------------------------------------
# iHC: independent hyper-connections over hc_mult residual streams
# ---------------------------------------------------------------------------------------------------------------------

def hc_mix(streams: torch.Tensor, fn: torch.Tensor, base: torch.Tensor, scale: torch.Tensor, hc_mult: int,
           magnitude: float, hc_eps: float, norm_eps: float) -> tuple:
    """(post [T, hc] fp32, the block input [T, H] in streams' dtype) of streams [T, hc, H]: a block reads the
    pre-gated sum of the streams and later adds its output to each stream times ``post``."""
    flat = streams.flatten(1).float()
    mixes = (flat @ fn.float().t()) * torch.rsqrt(flat.square().mean(-1, keepdim=True) + norm_eps)
    pre = torch.sigmoid(mixes[:, :hc_mult] * scale[0].float() + base[:hc_mult].float()) + hc_eps
    post = magnitude * torch.sigmoid(mixes[:, hc_mult:] * scale[1].float() + base[hc_mult:].float()) + hc_eps
    out = (pre.unsqueeze(-1) * streams.float()).sum(dim=1)
    return post, out.to(streams.dtype)


def hc_add(streams: torch.Tensor, post: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """streams + post x the block output, in fp32, cast back."""
    return (post.unsqueeze(-1) * out.float().unsqueeze(1) + streams.float()).to(streams.dtype)


def hc_head(streams: torch.Tensor, fn: torch.Tensor, base: torch.Tensor, scale: torch.Tensor, hc_eps: float,
            norm_eps: float) -> torch.Tensor:
    """The final collapse of the streams [T, hc, H] into [T, H]."""
    flat = streams.flatten(1).float()
    mixes = (flat @ fn.float().t()) * torch.rsqrt(flat.square().mean(-1, keepdim=True) + norm_eps)
    pre = torch.sigmoid(mixes * scale.float() + base.float()) + hc_eps
    return (pre.unsqueeze(-1) * streams.float()).sum(dim=1).to(streams.dtype)


# ---------------------------------------------------------------------------------------------------------------------
# attention
# ---------------------------------------------------------------------------------------------------------------------

def mla_attend(q_latent: torch.Tensor, q_rope: torch.Tensor, keys: torch.Tensor, allowed: torch.Tensor,
               sinks: torch.Tensor, scale: float, latent: int) -> torch.Tensor:
    """Absorbed MLA with a sink logit per head: [R, heads, latent] fp32 of the attended latents.

    q_latent [R, heads, latent] (the nope query through W_uk), q_rope [R, heads, rope], keys [R, K, latent + rope]
    the rows each query reads (a shared [K, latent + rope] broadcasts), allowed [R, K] bool, sinks [heads] fp32."""
    k_lat, k_rope = keys[..., :latent], keys[..., latent:]
    ql, qr = q_latent.to(keys.dtype), q_rope.to(keys.dtype)
    # the matmuls read the keys as they are (bf16, accumulated in fp32), as the reference's bf16 attention does:
    # widening them first doubles what a decode reads
    if keys.dim() == 2:
        scores = torch.einsum("rhd,kd->rhk", ql, k_lat).float() + torch.einsum("rhd,kd->rhk", qr, k_rope).float()
    else:
        scores = torch.einsum("rhd,rkd->rhk", ql, k_lat).float() + torch.einsum("rhd,rkd->rhk", qr, k_rope).float()
    scores = scores * scale
    scores = torch.where(allowed.unsqueeze(1), scores, torch.full_like(scores, NEG))
    sink = sinks.float().view(1, -1, 1)
    m = torch.maximum(scores.amax(dim=-1, keepdim=True), sink)
    p = torch.exp(scores - m)
    denom = p.sum(dim=-1, keepdim=True) + torch.exp(sink - m)
    p = p / denom
    # the reference casts the probabilities to the values' dtype before the value matmul
    p = p.to(keys.dtype)
    if keys.dim() == 2:
        return torch.einsum("rhk,kd->rhd", p, k_lat).float()
    return torch.einsum("rhk,rkd->rhd", p, k_lat).float()


# ---------------------------------------------------------------------------------------------------------------------
# IndexPool
# ---------------------------------------------------------------------------------------------------------------------

def pool_group_keys(members: torch.Tensor, value_w: torch.Tensor, gate_w: torch.Tensor, ape: torch.Tensor,
                    key_norm: torch.Tensor, eps: float, out_dtype) -> torch.Tensor:
    """Group keys [G, D] of the groups' member token keys [G, ratio, D]: the gated softmax pooling in fp32, the
    key norm, cast to the token keys' dtype (before the RoPE of the group's first position)."""
    grouped = members.float()
    values = grouped @ value_w.float().t()
    gates = grouped @ gate_w.float().t() + ape.float()
    pooled = (values * gates.softmax(dim=1)).sum(dim=1)
    return rms_norm(pooled, key_norm, eps).to(out_dtype)


def index_scores(q: torch.Tensor, head_w: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
    """[R, G] fp32 relu-scored logits of queries q [R, heads, D] (RoPE'd), head weights [R, heads] fp32 (already
    scaled) and group keys keys [R, G, D] (or a shared [G, D])."""
    if keys.dim() == 2:
        s = torch.relu(torch.einsum("rhd,gd->rhg", q.float(), keys.float()))
    else:
        s = torch.relu(torch.einsum("rhd,rgd->rhg", q.float(), keys.float()))
    # an elementwise weighting and a sum over the heads: as an einsum it lowers to one [1, heads] x [heads, G] matmul a
    # row
    return (head_w.float().unsqueeze(-1) * s).sum(dim=1)


def block_cumsum(x: torch.Tensor, block: int = 128) -> torch.Tensor:
    """Inclusive cumulative sum along the last dimension of x [R, N] (int32 or float) in blocks: within each block by a
    [block, block] triangular matmul, then each block's offset from the cumulative sum of the block totals.
    neuronx-cc lowers torch.cumsum over N to a matmul with an [N, N] triangular matrix (an IndexPool row over a 9K
    table is 2304 groups, 21 MB of weights a call, over a 1M table 262K: no fit); this reads block^2 + (N / block)^2."""
    rows, n = x.shape
    pad = (-n) % block
    xf = torch.nn.functional.pad(x.to(torch.float32), (0, pad)).view(rows, -1, block)
    idx = torch.arange(block, device=x.device)
    tri = (idx.unsqueeze(1) <= idx.unsqueeze(0)).to(torch.float32)
    within = xf @ tri
    totals = within[..., -1]
    nb = totals.shape[1]
    if nb > 1:
        bidx = torch.arange(nb, device=x.device)
        offsets = totals @ (bidx.unsqueeze(1) < bidx.unsqueeze(0)).to(torch.float32)
        within = within + offsets.unsqueeze(-1)
    out = within.reshape(rows, -1)[:, :n]
    return out.to(x.dtype) if x.dtype != torch.bool else out.to(torch.int32)


def top_count_mask(scores: torch.Tensor, valid: torch.Tensor, count: int, iters: int | None = None,
                   ways: int | None = None) -> torch.Tensor:
    """[R, G] bool: each row's ``count`` best valid entries (every valid one where a row has no more), by a search
    on the threshold instead of a sort: ``count`` above about a hundred is beyond what torch.topk lowers to well on a
    NeuronCore. Each step splits the row's interval into ``ways`` parts at once (``ways - 1`` thresholds compared in
    one pass), so ``iters`` steps resolve as finely as ``iters x log2(ways)`` halvings (32 by default) in a quarter of
    the passes; a prefill chunk's rows halve instead (15 thresholds a pass over 2048 rows cost more than the passes).
    Entries tied at the threshold (relu scores are often exactly 0) are kept in group order up to ``count`` after
    every entry above them (a group-order cap on all of them would drop a later positive score for an earlier 0 where
    fewer than ``count`` scores are positive)."""
    if ways is None:
        # a decode's few rows: few passes over a wide compare; a prefill chunk's rows: the compare itself dominates
        ways = 16 if scores.shape[0] <= WIDE_SEARCH_ROWS else 2
    if iters is None:
        iters = 32 // (ways.bit_length() - 1)
    big = torch.finfo(torch.float32).max
    s = torch.where(valid, scores, torch.full_like(scores, -big))
    lo = torch.where(valid, scores, torch.full_like(scores, big)).amin(dim=-1, keepdim=True)
    hi = s.amax(dim=-1, keepdim=True)
    n_valid = valid.sum(dim=-1, keepdim=True)
    frac = torch.arange(1, ways, device=scores.device, dtype=torch.float32) / ways
    slot = torch.arange(ways - 1, device=scores.device)
    for _ in range(iters):
        t = lo + (hi - lo) * frac  # [R, ways - 1], increasing
        # an invalid entry is -max, below every threshold (t >= lo, the least valid score): no mask a pass
        above = (s.unsqueeze(1) >= t.unsqueeze(-1)).sum(dim=-1)
        # the counts fall as t rises: the thresholds that keep at least ``count`` are a prefix
        n_ok = (above >= count).sum(dim=-1, keepdim=True)
        new_lo = (t * (slot == n_ok - 1)).sum(dim=-1, keepdim=True)
        new_hi = (t * (slot == n_ok)).sum(dim=-1, keepdim=True)
        lo = torch.where(n_ok > 0, new_lo, lo)
        hi = torch.where(n_ok < ways - 1, new_hi, hi)
    # s >= hi is fewer than ``count`` (unless hi never moved: ``count`` or more tied at the top), all kept; the band
    # [lo, hi) is the ties at the threshold, kept in group order up to ``count``
    n_hi = ((s >= hi) & valid).sum(dim=-1, keepdim=True)
    strict = (s >= hi) & valid & (n_hi < count)
    band = (s >= lo) & valid & ~strict
    room = count - strict.sum(dim=-1, keepdim=True)
    keep = strict | (band & (block_cumsum(band.to(torch.int32)) <= room))
    return torch.where(n_valid <= count, valid, keep)


def selected_table_tokens(group_mask: torch.Tensor, own_group: torch.Tensor, keys: int, query_positions: torch.Tensor,
                          ratio: int) -> torch.Tensor:
    """[R, keys] bool: which of a table's positions 0 .. keys - 1 a query attends after the IndexPool: a selected
    complete group's tokens or its own group's, causal. Each group's flag is repeated over its ``ratio`` tokens
    instead of gathered by position: neuronx-cc asserts (NCC_IDLO901, DataLocalityOpt) on the gather of the
    bisection's mask in a graph of two sparse layers."""
    pos = torch.arange(keys, device=group_mask.device)
    own = torch.div(pos, ratio, rounding_mode="floor").unsqueeze(0) == own_group.unsqueeze(1)
    chosen = group_mask.repeat_interleave(ratio, dim=1)[:, :keys] | own
    return chosen & (pos.unsqueeze(0) <= query_positions.unsqueeze(1))


def compact_groups(chosen: torch.Tensor, own_group: torch.Tensor, query_positions: torch.Tensor, budget: int,
                   ratio: int, block: int = 128) -> tuple:
    """(groups [R, budget + 1] int64, positions [R, (budget + 1) x ratio] int64, valid bool) of the tokens a query
    attends: its first ``budget`` chosen groups (chosen [R, G] bool) in group order, then its own group, each expanded
    to its ``ratio`` tokens; an empty group slot (-1), and a token after the query, is invalid.

    The j-th chosen group is found in two levels. The groups split into blocks of ``block``; the j-th chosen lies in
    the block whose count of chosen groups up to its end first exceeds j, and inside it at the number of groups whose
    count within the block is at most j minus the chosen before the block. The block's counts are picked by a one-hot
    matmul. This reads budget x (G / block + block) entries a row; the direct count of every group a slot (budget x
    G) and a scatter of each chosen group to its rank (one software DMA an element) took about 20 ms a sparse layer
    at 64 rows."""
    rows, groups = chosen.shape
    pad = (-groups) % block
    nb = (groups + pad) // block
    flags = F.pad(chosen.to(torch.float32), (0, pad)).view(rows, nb, block)
    idx = torch.arange(block, device=chosen.device)
    within = flags @ (idx.unsqueeze(1) <= idx.unsqueeze(0)).to(torch.float32)  # [R, nb, block] counts in a block
    totals = within[..., -1]
    bidx = torch.arange(nb, device=chosen.device)
    ends = totals @ (bidx.unsqueeze(1) <= bidx.unsqueeze(0)).to(torch.float32)  # [R, nb] counts to a block's end
    starts = ends - totals
    slots = torch.arange(budget, device=chosen.device, dtype=torch.float32)
    # the block of each slot: how many blocks end at or before it
    b = (ends.unsqueeze(1) <= slots.view(1, -1, 1)).to(torch.float32).sum(dim=-1).clamp(max=nb - 1)  # [R, budget]
    onehot = (b.unsqueeze(-1) == bidx.to(torch.float32)).to(torch.float32)  # [R, budget, nb]
    local = onehot @ within  # [R, budget, block]
    rank = slots.unsqueeze(0) - (onehot * starts.unsqueeze(1)).sum(dim=-1)  # [R, budget] rank within its block
    inside = (local <= rank.unsqueeze(-1)).to(torch.float32).sum(dim=-1)
    picked = (b * block + inside).to(torch.int64)
    picked = torch.where(slots.unsqueeze(0) < ends[:, -1:], picked, torch.full_like(picked, -1))
    picked = torch.cat([picked, own_group.to(torch.int64).unsqueeze(1)], dim=1)
    tokens = (picked.unsqueeze(-1) * ratio + torch.arange(ratio, device=chosen.device)).flatten(1)
    valid = (tokens >= 0) & (tokens <= query_positions.to(torch.int64).unsqueeze(1))
    return picked, tokens, valid

# ---------------------------------------------------------------------------------------------------------------------
# experts
# ---------------------------------------------------------------------------------------------------------------------

def swiglu(gate: torch.Tensor, up: torch.Tensor, limit: float) -> torch.Tensor:
    if limit > 0:
        gate = gate.clamp(max=limit)
        up = up.clamp(min=-limit, max=limit)
    return F.silu(gate) * up


def route(x: torch.Tensor, router_w: torch.Tensor, bias: torch.Tensor, top_k: int, norm: bool, scaling: float):
    """(affinities [T, E] fp32 zero off the chosen, chosen [T, k]): sigmoid scores, the top k chosen on score + bias,
    their weights the chosen scores renormalized and scaled (HYV4FlashTopkRouter with one group)."""
    scores = torch.sigmoid(x.float() @ router_w.float().t())
    _, chosen = torch.topk(scores + bias.float(), top_k, dim=-1)
    weights = scores.gather(1, chosen)
    if norm:
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
    weights = weights * scaling
    return torch.zeros_like(scores).scatter(1, chosen, weights), chosen
