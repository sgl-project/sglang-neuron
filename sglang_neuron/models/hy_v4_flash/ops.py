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
    return torch.einsum("rh,rhg->rg", head_w.float(), s)


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


def top_count_mask(scores: torch.Tensor, valid: torch.Tensor, count: int, iters: int = 32) -> torch.Tensor:
    """[R, G] bool: each row's ``count`` best valid entries (every valid one where a row has no more), by bisection
    on the threshold instead of a sort: ``count`` above about a hundred is beyond what torch.topk lowers to well on a
    NeuronCore. Entries tied at the threshold (relu scores are often exactly 0) are kept in group order up to
    ``count``, as the decode's compaction keeps them (compact_groups)."""
    big = torch.finfo(torch.float32).max
    s = torch.where(valid, scores, torch.full_like(scores, -big))
    lo = torch.where(valid, scores, torch.full_like(scores, big)).amin(dim=-1, keepdim=True)
    hi = s.amax(dim=-1, keepdim=True)
    n_valid = valid.sum(dim=-1, keepdim=True)
    for _ in range(iters):
        mid = lo + (hi - lo) * 0.5
        above = ((s >= mid) & valid).sum(dim=-1, keepdim=True)
        ok = above >= count
        lo = torch.where(ok, mid, lo)
        hi = torch.where(ok, hi, mid)
    keep = (s >= lo) & valid
    keep = keep & (block_cumsum(keep.to(torch.int32)) <= count)
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
                   ratio: int, chunk: int = 128) -> tuple:
    """(groups [R, budget + 1] int64, positions [R, (budget + 1) x ratio] int64, valid bool) of the tokens a query
    attends: its first ``budget`` chosen groups (chosen [R, G] bool) in group order, then its own group, each expanded
    to its ``ratio`` tokens; an empty group slot (-1), and a token after the query, is invalid.

    The j-th chosen group is the number of groups whose inclusive count of chosen groups is at most j (the count is
    non-decreasing and first reaches j + 1 there), a compare and a sum over the row, ``chunk`` slots at a time. A
    scatter of each chosen group to its rank lowers to one software DMA an element (about 22 ms a layer at 64 rows)."""
    rows, groups = chosen.shape
    counts = block_cumsum(chosen.to(torch.int32))
    total = counts[:, -1:]
    slots = torch.arange(budget, device=chosen.device, dtype=torch.int32)
    parts = []
    for lo in range(0, budget, chunk):
        j = slots[lo:lo + chunk].view(1, -1, 1)
        parts.append((counts.unsqueeze(1) <= j).to(torch.int32).sum(dim=-1))
    picked = torch.cat(parts, dim=1)
    picked = torch.where(slots.unsqueeze(0) < total, picked, torch.full_like(picked, -1)).to(torch.int64)
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
