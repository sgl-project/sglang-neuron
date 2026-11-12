# SPDX-License-Identifier: Apache-2.0
"""Absorbed multi-head latent attention (DeepSeek-V2) over a cache of compact rows: one row per token slot, the token's
normalized KV latent followed by its rotated RoPE key.

The key side of the KV up-projection is folded into the query, so a query scores the cached latent directly, and the
attention returns per head a latent that the caller projects to its values. On a NeuronCore the prefill and the decode
run the MLA kernels (nki/deepseek_mla_prefill, nki/deepseek_mla_decode); elsewhere the same attention runs in torch."""

import torch
import torch.nn.functional as F
from torch import Tensor

from sglang_neuron.utils.neuron_utils import can_run_kernel


def token_slots(block_table: Tensor, page: int, width: int) -> Tensor:
    """[rows, width] the token slot of each row's first ``width`` positions from its pages ``block_table`` [rows,
    pages] (-1 past a row's pages, which gives -1 slots)."""
    positions = torch.arange(width, device=block_table.device)
    pages = block_table[:, positions // page].to(torch.int64)
    return torch.where(pages >= 0, pages * page + positions % page, torch.full_like(pages, -1))


def _scores(ql_nope: Tensor, q_pe: Tensor, keys: Tensor, latent: int, scale: float) -> Tensor:
    """[..., heads, queries, keys] fp32 scores of the absorbed queries over the compact ``keys`` [..., keys, latent +
    rope]."""
    k_latent, k_pe = keys.float()[..., :latent], keys.float()[..., latent:]
    return (ql_nope.float() @ k_latent.transpose(-1, -2)[..., None, :, :]
            + q_pe.float() @ k_pe.transpose(-1, -2)[..., None, :, :]) * scale


def mla_decode(ql_nope: Tensor, q_pe: Tensor, compact_new: Tensor, pool: Tensor, slots: Tensor, seq_lens: Tensor, *,
               scale: float) -> Tensor:
    """[B, heads, latent] the attended latent of each row's query, ``ql_nope`` [B, heads, latent] and ``q_pe`` [B,
    heads, rope], over its history and its own new row ``compact_new`` [B, latent + rope]. The history is the row's
    positions below seq_len - 1, read from ``pool`` [slots, 1, latent + rope] at ``slots`` [B, context]. On a
    NeuronCore the kernel reads whole 512-token supertiles, so a context off them (a rung of the decode context ladder)
    pads with -1, no slot."""
    if can_run_kernel(ql_nope):
        from sglang_neuron.nki.deepseek_mla_decode import SUPERTILE, paged_absorbed_mla

        slots = F.pad(slots.to(torch.int32), (0, -slots.shape[1] % SUPERTILE), value=-1)
        return paged_absorbed_mla(ql_nope, q_pe, compact_new, pool, slots, seq_lens, scale=scale)
    latent = ql_nope.shape[-1]
    rows = pool.reshape(pool.shape[0], -1)
    history = rows[slots.clamp(min=0)]
    keys = torch.cat([history, compact_new.reshape(compact_new.shape[0], 1, -1).to(history.dtype)], dim=1)
    context = slots.shape[1]
    valid = torch.cat([torch.arange(context, device=slots.device)[None, :] < (seq_lens.to(torch.int64)[:, None] - 1),
                       torch.ones(slots.shape[0], 1, dtype=torch.bool, device=slots.device)], dim=1)
    scores = _scores(ql_nope[:, :, None], q_pe[:, :, None], keys, latent, scale)[:, :, 0]
    probs = torch.softmax(scores.masked_fill(~valid[:, None], float("-inf")), dim=-1)
    return (probs[:, :, None] @ keys.float()[:, None, :, :latent])[:, :, 0].to(ql_nope.dtype)


def mla_prefill(ql_nope: Tensor, q_pe: Tensor, compact_new: Tensor, pool: Tensor, slots: Tensor, prior: Tensor,
                length: Tensor, *, scale: float) -> Tensor:
    """[1, heads, W, latent] the attended latent of the window's W queries, ``ql_nope`` [1, heads, W, latent] and
    ``q_pe`` [1, heads, W, rope]: query i sits at position prior + i and attends causally to the positions up to
    min(prior + i, length - 1). Positions below ``prior`` are read from ``pool`` at ``slots`` [1, W]; positions in
    [prior, length) are the window's own rows ``compact_new`` [W, latent + rope], the first length - prior of them. The
    window holds the whole request: prior + new tokens <= W. A -1 slot (past the request's pages) reads nothing."""
    if can_run_kernel(ql_nope):
        from sglang_neuron.nki.deepseek_mla_prefill import prefill_absorbed_mla_cte

        return prefill_absorbed_mla_cte(ql_nope, q_pe, compact_new, pool, slots.to(torch.int32), prior, length,
                                        scale=scale)
    latent, width = ql_nope.shape[-1], ql_nope.shape[2]
    positions = torch.arange(width, device=slots.device)
    prior, length = prior.reshape(()).to(torch.int64), length.reshape(()).to(torch.int64)
    rows = pool.reshape(pool.shape[0], -1)
    new_rows = compact_new.reshape(width, -1)[(positions - prior).clamp(0, width - 1)]
    keys = torch.where((positions < prior)[:, None], rows[slots[0].clamp(min=0)], new_rows.to(rows.dtype))
    bound = torch.minimum(positions + prior, length - 1)
    allowed = positions[None, :] <= bound[:, None]
    scores = _scores(ql_nope[0], q_pe[0], keys, latent, scale)
    probs = torch.softmax(scores.masked_fill(~allowed[None], float("-inf")), dim=-1)
    return (probs @ keys.float()[:, :latent]).to(ql_nope.dtype)[None]
