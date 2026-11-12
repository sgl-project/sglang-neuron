# SPDX-License-Identifier: Apache-2.0
"""Segmented prefill attention: the bucket's queries attend to the request's prior over the paged cache, in segments
of ``kv_segment_size`` tokens, and to themselves causally. The runtime's warm-up refuses a bucket the kernel cannot
tile (check_prefill_tiles), so every device call reaches nkilib's attention_segmented_cte."""

from typing import Optional

import nki
import torch
from nkilib.core.attention.attention_segmented_cte import attention_segmented_cte
from torch import Tensor

from sglang_neuron.backend.stack import wrap_nki
from sglang_neuron.ops.attention.attention_decode import unswizzle_packed_k
from sglang_neuron.utils.hardware_config import get_logical_nc_config
from sglang_neuron.utils.neuron_utils import can_run_kernel

_attention_segmented_cte = nki.jit()(attention_segmented_cte)


def segmented_attention(q: Tensor, *, k_cache: Tensor, v_cache: Tensor, block_tables: Tensor, prior_tokens: Tensor,
                        block_size: int, kv_segment_size: int, scale: float, sliding_window: Optional[int],
                        sink: Optional[Tensor]) -> Tensor:
    """Attention of q [heads, S, d] over the prior ``prior_tokens`` (a whole number of pages) and the bucket itself.

    v_cache is [pages, kv_heads, page, d]; k_cache is the same in bf16, or packed as [pages, kv_heads, page // 2, d, 2]
    in FP8. sink is [heads, 1], or None for a model without sinks. ``scale`` is applied to q, since the kernel takes a
    unit scale with a sliding window or a prior. Returns [heads, S, d].
    """
    fp8_packed = k_cache.dim() == v_cache.dim() + 1
    if not can_run_kernel(q):
        return _torch_segmented_attention(q, unswizzle_packed_k(k_cache) if fp8_packed else k_cache, v_cache,
                                          block_tables, prior_tokens, block_size, scale, sliding_window, sink)
    return wrap_nki(_attention_segmented_cte)[get_logical_nc_config()](
        q=q * scale,
        k_cache=k_cache,
        v_cache=v_cache,
        block_tables=block_tables,
        prior_tokens=prior_tokens,
        block_size=block_size,
        prior_seg_size=kv_segment_size,
        scale=1.0,
        sliding_window=sliding_window,
        sink=sink,
        num_q_heads=q.shape[0],
        fp8_packed=fp8_packed,
    )


def _torch_segmented_attention(q, k_cache, v_cache, block_tables, prior_tokens, block_size, scale, sliding_window,
                               sink):
    """The kernel's result by plain attention over the row's whole block table, with positions past the request
    masked out. Static shapes and no ``.item()``, so it traces."""
    heads, s_q, d = q.shape
    kv_heads = k_cache.shape[1]
    span = block_tables.shape[1] * block_size
    pages = block_tables[0].clamp_min(0).long()
    k = k_cache[pages].permute(1, 0, 2, 3).reshape(kv_heads, span, d).repeat_interleave(heads // kv_heads, dim=0)
    v = v_cache[pages].permute(1, 0, 2, 3).reshape(kv_heads, span, d).repeat_interleave(heads // kv_heads, dim=0)

    prior = prior_tokens.reshape(-1)[0].long()
    q_pos = torch.arange(s_q, device=q.device).unsqueeze(1) + prior
    k_pos = torch.arange(span, device=q.device).unsqueeze(0)
    allowed = (k_pos < prior + s_q) & (k_pos <= q_pos)
    if sliding_window:
        allowed &= k_pos > q_pos - sliding_window

    scores = torch.bmm(q.float(), k.float().transpose(1, 2)) * scale
    scores = scores.masked_fill(~allowed.unsqueeze(0), float("-inf"))
    if sink is None:
        weights = torch.softmax(scores, dim=-1)
    else:
        scores = torch.cat([scores, sink.float().reshape(heads, 1, 1).expand(heads, s_q, 1)], dim=-1)
        weights = torch.softmax(scores, dim=-1)[:, :, :-1]
    return torch.bmm(weights, v.float()).to(q.dtype)
