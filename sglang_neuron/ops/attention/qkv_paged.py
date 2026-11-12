# SPDX-License-Identifier: Apache-2.0
"""The prefill QKV of the dense models over a bf16 paged cache: nkilib's qkv_cte projects the chunk (with its bias),
RMSNorms each Q and K head before RoPE, rotates them and writes K and V into the cache's [pages, kv_heads, page, d]
layout in place (head-split), returning Q. The torch path is the reference the CPU tests run."""

import nki
import torch
from nkilib.core.qkv.qkv_cte import qkv_cte
from nkilib.core.utils.common_types import NormType, QKNormConfig
from torch import Tensor

from sglang_neuron.backend.stack import wrap_nki
from sglang_neuron.utils.hardware_config import get_logical_nc_config
from sglang_neuron.utils.neuron_utils import can_run_kernel


@nki.jit
def _qkv_head_split(hidden, weights, bias, cos, sin, q_norm, k_norm, norm_eps, d_head, num_q_heads, num_kv_heads,
                    k_cache, v_cache, block_size, slot_mapping):
    qk_norm = None if q_norm is None else QKNormConfig(q_norm=NormType.RMS_NORM, k_norm=NormType.RMS_NORM,
                                                       eps=norm_eps, q_gamma_norm_weights=q_norm,
                                                       k_gamma_norm_weights=k_norm)
    return qkv_cte(input=hidden, fused_qkv_weights=weights, bias=bias, fused_rope=True, cos_cache=cos, sin_cache=sin,
                   d_head=d_head, num_q_heads=num_q_heads, num_kv_heads=num_kv_heads, k_cache=k_cache, v_cache=v_cache,
                   use_block_kv=True, head_split_kv_cache=True, block_size=block_size, slot_mapping=slot_mapping,
                   qk_norm_pre_rope=qk_norm)


def qkv_paged(hidden: Tensor, *, weights: Tensor, bias: Tensor | None, cos: Tensor, sin: Tensor, q_norm: Tensor | None,
              k_norm: Tensor | None, norm_eps: float, d_head: int, num_q_heads: int, num_kv_heads: int, k_cache: Tensor,
              v_cache: Tensor, block_size: int, slot_mapping: Tensor):
    """Q of hidden [1, T, H] as [1, T, q * d], with the chunk's K and V written at ``slot_mapping`` [1, T] int32 into
    the [pages, kv_heads, page, d] caches; returns ``(q, k_cache, v_cache)``, the caches returned so the attention that
    reads them depends on the write. ``weights`` [H, (q + 2 kv) * d], ``bias`` [1, (q + 2 kv) * d], ``q_norm`` /
    ``k_norm`` [1, d] the per-head RMSNorm gammas (None without QK norm), cos/sin [1, T, d] over contiguous halves."""
    args = dict(weights=weights, bias=bias, cos=cos, sin=sin, q_norm=q_norm, k_norm=k_norm, norm_eps=norm_eps,
                d_head=d_head, num_q_heads=num_q_heads, num_kv_heads=num_kv_heads, k_cache=k_cache, v_cache=v_cache,
                block_size=block_size, slot_mapping=slot_mapping)
    if not can_run_kernel(hidden):
        return _torch_qkv_paged(hidden, **args)
    return wrap_nki(_qkv_head_split)[get_logical_nc_config()](hidden, **args)


def _rms(x: Tensor, gamma: Tensor, eps: float) -> Tensor:
    x32 = x.float()
    return gamma.reshape(-1) * (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype)


def _torch_qkv_paged(hidden, *, weights, bias, cos, sin, q_norm, k_norm, norm_eps, d_head, num_q_heads, num_kv_heads,
                     k_cache, v_cache, block_size, slot_mapping):
    tokens = hidden.shape[1]
    qkv = hidden[0] @ weights
    if bias is not None:
        qkv = qkv + bias.reshape(-1)
    q, k, v = qkv.split([num_q_heads * d_head, num_kv_heads * d_head, num_kv_heads * d_head], dim=-1)
    q, k, v = q.view(tokens, -1, d_head), k.view(tokens, -1, d_head), v.view(tokens, -1, d_head)
    if q_norm is not None:
        q, k = _rms(q, q_norm, norm_eps), _rms(k, k_norm, norm_eps)
    c, s = cos[0][:, None], sin[0][:, None]

    def rope(t):
        return t * c + torch.cat([-t[..., d_head // 2:], t[..., :d_head // 2]], dim=-1) * s

    q, k = rope(q), rope(k)
    slots = slot_mapping.reshape(-1).to(torch.int64)
    k_cache[slots // block_size, :, slots % block_size] = k.to(k_cache.dtype)
    v_cache[slots // block_size, :, slots % block_size] = v.to(v_cache.dtype)
    return q.reshape(1, tokens, -1), k_cache, v_cache
