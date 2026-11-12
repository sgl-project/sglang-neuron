# SPDX-License-Identifier: Apache-2.0
"""The fused sliding-window prefill of one layer over the packed FP8 cache: nkilib's swa_fused_cte runs the QKV
projection, RoPE, windowed attention over the prior and the bucket with per-head sinks, the output projection and the
K/V write in one kernel."""

import nki
import torch
import torch.nn.functional as F
from nkilib.experimental.attention.swa_fused_cte import swa_fused_cte
from torch import Tensor

from sglang_neuron.backend.stack import wrap_nki
from sglang_neuron.ops.attention.attention_decode_mask import P_MAX
from sglang_neuron.utils.dtype_utils import FP8_CLAMP_MAX
from sglang_neuron.utils.neuron_utils import can_run_kernel

# the kernel splits a rank's KV heads over the two physical cores of an LNC2 rank
SWA_FUSED_CORE_SHARDS = 2

_swa_fused_cte = nki.jit()(swa_fused_cte)


def swa_fused_attention(hidden_states: Tensor, *, qkv_weight: Tensor, op_weight: Tensor, k_cache: Tensor,
                        v_cache: Tensor, block_tables: Tensor, cos_cache: Tensor, sin_cache: Tensor, sink: Tensor,
                        prior_tokens: Tensor, qkv_bias: Tensor, op_bias: Tensor, scale: float, sliding_window: int,
                        block_size: int, num_q_heads: int, num_kv_heads: int, d_head: int, k_scale: Tensor,
                        v_scale: Tensor):
    """hidden [1, S, H] -> ``(out [1, S, H], k_cache, v_cache)``, the caches written in place.

    The K cache is packed [pages, kv_heads, page // 2, d, 2], V [pages, kv_heads, page, d], both FP8 with the
    per-tensor dequantization scales k_scale/v_scale [1, 1]. block_tables [1, blocks] (-1 past the request's pages),
    cos/sin [1, S, d] with the halves repeated, sink [1, num_q_heads], prior_tokens [1, 1] int32. The window must fit
    in a page and S be a multiple of 128.
    """
    # The kernel writes every 128-token tile of the bucket, padding included, to the logical blocks after the prior,
    # and a column past the table would read an uninitialized index there. A last chunk whose prior and bucket run past
    # the context reaches past the table, so it is extended with -1 columns, which the kernel skips.
    block_tables = F.pad(block_tables, (0, -(-hidden_states.shape[1] // block_size)), value=-1)
    if not can_run_kernel(hidden_states):
        return _torch_swa_fused(hidden_states, qkv_weight, op_weight, k_cache, v_cache, block_tables, cos_cache,
                                sin_cache, sink, prior_tokens, qkv_bias, op_bias, scale, sliding_window, block_size,
                                num_q_heads, num_kv_heads, d_head, k_scale, v_scale)
    return wrap_nki(_swa_fused_cte)[SWA_FUSED_CORE_SHARDS](
        hidden_states=hidden_states,
        qkv_weight=qkv_weight,
        op_weight=op_weight,
        k_cache=k_cache,
        v_cache=v_cache,
        block_tables=block_tables,
        cos_cache=cos_cache,
        sin_cache=sin_cache,
        sink=sink,
        prior_tokens=prior_tokens,
        qkv_bias=qkv_bias,
        op_bias=op_bias,
        scale=scale,
        sliding_window=sliding_window,
        block_size=block_size,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        d_head=d_head,
        # the kernel loads its scales as [P_MAX, 1]
        k_scale=k_scale.reshape(1, 1).expand(P_MAX, 1).contiguous(),
        v_scale=v_scale.reshape(1, 1).expand(P_MAX, 1).contiguous(),
    )


def _rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.shape[-1] // 2
    return x * cos + torch.cat([-x[..., half:], x[..., :half]], dim=-1) * sin


def _quantize(x: Tensor, scale: float, dtype: torch.dtype) -> Tensor:
    """The kernel's cache write: rounded to bf16, divided by the scale and clamped to the FP8 range."""
    return (x.to(torch.bfloat16).float() / scale).clamp(-FP8_CLAMP_MAX, FP8_CLAMP_MAX).to(dtype)


def _torch_swa_fused(hidden_states, qkv_weight, op_weight, k_cache, v_cache, block_tables, cos_cache, sin_cache, sink,
                     prior_tokens, qkv_bias, op_bias, scale, sliding_window, block_size, num_q_heads, num_kv_heads,
                     d_head, k_scale, v_scale):
    """The kernel in fp32: attention reads the bucket's own K/V unquantized and the prior dequantized, and the prior's
    bytes are never rewritten. CPU indexing has no fp8 kernels, so the cache is read and written through a uint8
    view."""
    k_s, v_s = float(k_scale.flatten()[0]), float(v_scale.flatten()[0])
    _, S, H = hidden_states.shape
    q_dim, kv_dim = num_q_heads * d_head, num_kv_heads * d_head
    prior = int(prior_tokens.flatten()[0])
    table = block_tables[0]

    qkv = hidden_states[0].float() @ qkv_weight.float() + qkv_bias.float()
    cos, sin = cos_cache[0].float().unsqueeze(1), sin_cache[0].float().unsqueeze(1)
    q = _rope(qkv[:, :q_dim].reshape(S, num_q_heads, d_head), cos, sin) * scale
    k = _rope(qkv[:, q_dim:q_dim + kv_dim].reshape(S, num_kv_heads, d_head), cos, sin)
    v = qkv[:, q_dim + kv_dim:].reshape(S, num_kv_heads, d_head)

    k_bytes, v_bytes = k_cache.view(torch.uint8), v_cache.view(torch.uint8)
    pos = torch.arange(prior)
    page, offset = table[pos // block_size].long(), pos % block_size
    k_prior = k_bytes[page, :, offset // 2, :, offset % 2].view(k_cache.dtype).float() * k_s
    v_prior = v_bytes[page, :, offset, :].view(v_cache.dtype).float() * v_s

    # the bucket's tokens go to the pages after the prior's; a -1 page (a padding tile past the request) is skipped
    pos = torch.arange(S)
    page, offset = table[-(-prior // block_size) + pos // block_size].long(), pos % block_size
    live = (page >= 0) & (page < k_cache.shape[0])
    page, offset = page[live], offset[live]
    k_bytes[page, :, offset // 2, :, offset % 2] = _quantize(k[live], k_s, k_cache.dtype).view(torch.uint8)
    v_bytes[page, :, offset, :] = _quantize(v[live], v_s, v_cache.dtype).view(torch.uint8)

    group = num_q_heads // num_kv_heads
    keys = torch.cat([k_prior, k]).repeat_interleave(group, dim=1)
    values = torch.cat([v_prior, v]).repeat_interleave(group, dim=1)
    q_pos = (torch.arange(S) + prior).unsqueeze(1)
    k_pos = torch.arange(prior + S).unsqueeze(0)
    hidden = (k_pos > q_pos) | (k_pos <= q_pos - sliding_window)
    scores = torch.einsum("shd,thd->hst", q, keys).masked_fill(hidden.unsqueeze(0), float("-inf"))
    scores = torch.cat([scores, sink[0].float().reshape(num_q_heads, 1, 1).expand(-1, S, 1)], dim=-1)
    attended = torch.einsum("hst,thd->shd", torch.softmax(scores, dim=-1)[..., :-1], values)
    out = attended.reshape(S, q_dim) @ op_weight.float() + op_bias.float()
    return out.to(hidden_states.dtype).unsqueeze(0), k_cache, v_cache
