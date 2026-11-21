# SPDX-License-Identifier: Apache-2.0
"""The prefill QKV of the dense models over a paged cache: nkilib's qkv_cte projects the chunk (with its bias),
RMSNorms each Q and K head before RoPE, rotates them and writes K and V into the cache in place, returning Q. A bf16
cache is [pages, kv_heads, page, d] for both (head-split). An FP8 cache (``--kv-cache-dtype fp8_e4m3``) keeps V in that
layout and K packed as [pages, kv_heads, page // 2, d, 2], at unit scale. The torch path is the reference the CPU tests
run."""

import nki
import nki.language as nl
import torch
from nkilib.core.qkv.qkv_cte import qkv_cte
from nkilib.core.utils.common_types import NormType, QKNormConfig
from torch import Tensor

from sglang_neuron.backend.stack import wrap_nki
from sglang_neuron.ops.attention.attention_decode import swizzle_packed_k, unswizzle_packed_k
from sglang_neuron.ops.attention.attention_decode_mask import P_MAX
from sglang_neuron.utils.dtype_utils import FP8_CLAMP_MAX
from sglang_neuron.utils.hardware_config import get_logical_nc_config
from sglang_neuron.utils.neuron_utils import can_run_kernel


def is_fp8_cache(cache: Tensor) -> bool:
    return cache.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)


def _qk_norm(q_norm, k_norm, norm_eps):
    return None if q_norm is None else QKNormConfig(q_norm=NormType.RMS_NORM, k_norm=NormType.RMS_NORM, eps=norm_eps,
                                                    q_gamma_norm_weights=q_norm, k_gamma_norm_weights=k_norm)


@nki.jit
def _qkv_head_split(hidden, weights, bias, cos, sin, q_norm, k_norm, norm_eps, d_head, num_q_heads, num_kv_heads,
                    k_cache, v_cache, block_size, slot_mapping):
    return qkv_cte(input=hidden, fused_qkv_weights=weights, bias=bias, fused_rope=True, cos_cache=cos, sin_cache=sin,
                   d_head=d_head, num_q_heads=num_q_heads, num_kv_heads=num_kv_heads, k_cache=k_cache, v_cache=v_cache,
                   use_block_kv=True, head_split_kv_cache=True, block_size=block_size, slot_mapping=slot_mapping,
                   qk_norm_pre_rope=_qk_norm(q_norm, k_norm, norm_eps))


@nki.jit
def _qkv_fp8_packed(hidden, weights, bias, cos, sin, q_norm, k_norm, norm_eps, d_head, num_q_heads, num_kv_heads,
                        k_cache, v_cache, block_size, slot_mapping, k_scale, v_scale, fp8_max, fp8_min, kv_dtype):
    return qkv_cte(input=hidden, fused_qkv_weights=weights, bias=bias, fused_rope=True, cos_cache=cos, sin_cache=sin,
                   d_head=d_head, num_q_heads=num_q_heads, num_kv_heads=num_kv_heads, k_cache=k_cache, v_cache=v_cache,
                   use_block_kv=True, fp8_packed=True, k_scale=k_scale, v_scale=v_scale,
                   fp8_max=fp8_max, fp8_min=fp8_min, kv_dtype=kv_dtype, block_size=block_size,
                   slot_mapping=slot_mapping, qk_norm_pre_rope=_qk_norm(q_norm, k_norm, norm_eps))


def qkv_paged(hidden: Tensor, *, weights: Tensor, bias: Tensor | None, cos: Tensor, sin: Tensor, q_norm: Tensor | None,
              k_norm: Tensor | None, norm_eps: float, d_head: int, num_q_heads: int, num_kv_heads: int, k_cache: Tensor,
              v_cache: Tensor, block_size: int, slot_mapping: Tensor):
    """Q of hidden [1, T, H] as [1, T, q * d], with the chunk's K and V written at ``slot_mapping`` [1, T] int32 into
    the caches; returns ``(q, k_cache, v_cache)``, the caches returned so the attention that reads them depends on the
    write. ``weights`` [H, (q + 2 kv) * d], ``bias`` [1, (q + 2 kv) * d], ``q_norm`` / ``k_norm`` [1, d] the per-head
    RMSNorm gammas (None without QK norm), cos/sin [1, T, d] over contiguous halves. An FP8 ``v_cache`` selects the
    FP8 write, K packed."""
    args = dict(weights=weights, bias=bias, cos=cos, sin=sin, q_norm=q_norm, k_norm=k_norm, norm_eps=norm_eps,
                d_head=d_head, num_q_heads=num_q_heads, num_kv_heads=num_kv_heads, k_cache=k_cache, v_cache=v_cache,
                block_size=block_size, slot_mapping=slot_mapping)
    fp8 = is_fp8_cache(v_cache)
    if not can_run_kernel(hidden):
        return _torch_qkv_paged(hidden, **args)
    if not fp8:
        return wrap_nki(_qkv_head_split)[get_logical_nc_config()](hidden, **args)
    # the kernel loads its scales as [P_MAX, 1] and recognizes FP8 by the NKI dtype
    unit = torch.ones((P_MAX, 1), dtype=torch.float32, device=hidden.device)
    return wrap_nki(_qkv_fp8_packed)[get_logical_nc_config()](
        hidden, k_scale=unit, v_scale=unit, fp8_max=FP8_CLAMP_MAX, fp8_min=-FP8_CLAMP_MAX,
        kv_dtype=nl.float8_e4m3fn, **args)


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
    index = (slots // block_size, slice(None), slots % block_size)
    if not is_fp8_cache(v_cache):
        k_cache[index] = k.to(k_cache.dtype)
        v_cache[index] = v.to(v_cache.dtype)
        return q.reshape(1, tokens, -1), k_cache, v_cache
    k, v = k.float().clamp(-FP8_CLAMP_MAX, FP8_CLAMP_MAX), v.float().clamp(-FP8_CLAMP_MAX, FP8_CLAMP_MAX)
    # through the compute dtype: CPU indexing has no fp8 kernels
    unpacked = unswizzle_packed_k(k_cache).to(hidden.dtype)
    unpacked[index] = k.to(k_cache.dtype).to(hidden.dtype)
    k_cache.copy_(swizzle_packed_k(unpacked.to(k_cache.dtype)))
    values = v_cache.to(hidden.dtype)
    values[index] = v.to(v_cache.dtype).to(hidden.dtype)
    v_cache.copy_(values.to(v_cache.dtype))
    return q.reshape(1, tokens, -1), k_cache, v_cache
