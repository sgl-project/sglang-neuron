# SPDX-License-Identifier: Apache-2.0
"""The prefill QKV projection: nkilib's qkv kernel projects, applies RoPE to Q and K and writes K/V into the paged FP8
cache, K in its packed layout, returning Q."""

import nki
import nki.language as nl
import torch
from nkilib.core.qkv.qkv import qkv
from nkilib.core.utils.common_types import DtypeMode
from torch import Tensor

from sglang_neuron.backend.stack import wrap_nki
from sglang_neuron.ops.attention.attention_decode import swizzle_packed_k, unswizzle_packed_k
from sglang_neuron.ops.attention.attention_decode_mask import P_MAX
from sglang_neuron.utils.dtype_utils import FP8_CLAMP_MAX
from sglang_neuron.utils.hardware_config import get_logical_nc_config
from sglang_neuron.utils.neuron_utils import can_run_kernel

# the widest fused QKV the kernel computes correctly; it must also not exceed the hidden size
MAX_FUSED_QKV = 4096


@nki.jit
def _qkv(hidden, qkv_weights, bias, d_head, cos_cache, sin_cache, num_q_heads, num_kv_heads, k_cache, v_cache,
         k_scale, v_scale, fp8_max, fp8_min, kv_dtype, block_size, slot_mapping):
    return qkv(input=hidden, fused_qkv_weights=qkv_weights, bias=bias, fused_rope=True, cos_cache=cos_cache,
               sin_cache=sin_cache, d_head=d_head, num_q_heads=num_q_heads, num_kv_heads=num_kv_heads,
               k_cache=k_cache, v_cache=v_cache, k_scale=k_scale, v_scale=v_scale, fp8_max=fp8_max, fp8_min=fp8_min,
               kv_dtype=kv_dtype, use_block_kv=True, fp8_packed=True, block_size=block_size,
               slot_mapping=slot_mapping, dtype_mode=DtypeMode.AUTO)


def qkv_proj(hidden: Tensor, *, qkv_weights: Tensor, bias: Tensor, d_head: int, cos_cache: Tensor, sin_cache: Tensor,
             num_q_heads: int, num_kv_heads: int, k_cache: Tensor, v_cache: Tensor, k_scale: Tensor, v_scale: Tensor,
             block_size: int, slot_mapping: Tensor):
    """Project hidden [1, T, H], rotate Q and K by cos/sin [1, T, d] and write K/V at ``slot_mapping`` [1, T].

    The caches are FP8: V [pages, kv_heads, page, d], K packed as [pages, kv_heads, page // 2, d, 2]. K and V are
    divided by k_scale/v_scale [1, 1] and clamped to the FP8 range. Returns ``(q, k_cache, v_cache)``, q as [1, T,
    num_q_heads * d]; the caches are written in place and returned so that the attention that reads them depends on the
    write.
    """
    if not can_run_kernel(hidden):
        return _torch_qkv(hidden, qkv_weights, bias, d_head, cos_cache, sin_cache, num_q_heads, num_kv_heads,
                          k_cache, v_cache, k_scale, v_scale, block_size, slot_mapping)
    # the packed write loads its scales as [P_MAX, 1]; the kernel recognizes FP8 by its NKI dtype, where a torch dtype
    # would size the packed store as bf16
    return wrap_nki(_qkv)[get_logical_nc_config()](
        hidden=hidden, qkv_weights=qkv_weights, bias=bias, d_head=d_head, cos_cache=cos_cache, sin_cache=sin_cache,
        num_q_heads=num_q_heads, num_kv_heads=num_kv_heads, k_cache=k_cache, v_cache=v_cache,
        k_scale=k_scale.reshape(1, 1).expand(P_MAX, 1).contiguous(),
        v_scale=v_scale.reshape(1, 1).expand(P_MAX, 1).contiguous(), fp8_max=FP8_CLAMP_MAX, fp8_min=-FP8_CLAMP_MAX,
        kv_dtype=nl.float8_e4m3fn, block_size=block_size, slot_mapping=slot_mapping,
    )


def _torch_qkv(hidden, qkv_weights, bias, d_head, cos_cache, sin_cache, num_q_heads, num_kv_heads, k_cache, v_cache,
               k_scale, v_scale, block_size, slot_mapping):
    out = hidden @ qkv_weights + bias  # [1, T, (q + 2 kv) * d]
    half = d_head // 2
    for head in range(num_q_heads + num_kv_heads):
        x = out[:, :, head * d_head:(head + 1) * d_head]
        x1, x2 = x[..., :half].clone(), x[..., half:].clone()
        sin = sin_cache[..., :half]
        x[..., :half] = x1 * cos_cache[..., :half] - x2 * sin
        x[..., half:] = x2 * cos_cache[..., half:] + x1 * sin
    q_dim, kv_dim = num_q_heads * d_head, num_kv_heads * d_head
    q, k, v = out[..., :q_dim], out[..., q_dim:q_dim + kv_dim], out[..., q_dim + kv_dim:]
    k = (k.float() / k_scale.float().flatten()[0]).clamp(-FP8_CLAMP_MAX, FP8_CLAMP_MAX)
    v = (v.float() / v_scale.float().flatten()[0]).clamp(-FP8_CLAMP_MAX, FP8_CLAMP_MAX)

    # token-major, head-minor rows, matched by the (page, head, offset) index
    tokens = k.shape[0] * k.shape[1]
    slots = slot_mapping.reshape(tokens).long()
    index = (
        (slots // block_size).repeat_interleave(num_kv_heads),
        torch.arange(num_kv_heads, device=k_cache.device).repeat(tokens),
        (slots % block_size).repeat_interleave(num_kv_heads),
    )
    # through the compute dtype: CPU indexing has no fp8 kernels
    unpacked = unswizzle_packed_k(k_cache).to(hidden.dtype)
    unpacked.index_put_(index, k.reshape(tokens * num_kv_heads, d_head).to(k_cache.dtype).to(hidden.dtype))
    k_cache.copy_(swizzle_packed_k(unpacked.to(k_cache.dtype)))
    v_cache.index_put_(index, v.reshape(tokens * num_kv_heads, d_head).to(v_cache.dtype))
    return q, k_cache, v_cache
