# SPDX-License-Identifier: Apache-2.0
"""Decode attention of the dense models over a bf16 paged cache: nkilib's attention_block_tkg runs the QKV projection
(with its bias), the per-head RMSNorm of Q and K before RoPE, RoPE, causal attention over the row's pages and its own
token, the K/V write and the output projection in one call per layer.

A rank may hold several KV heads. The cache is [pages, kv_heads, page, d], which the kernel reads as kv_heads blocks per
page: a row's block table and its write slot are folded per KV head (``page * kv_heads + head``). The kernel masks the
prior by position (``pos_ids``) over every block of the graph's table. On a CPU the torch path computes the same
attention.

``max_context_len`` is not passed. On nkilib's flash-attention path (a prior above its 8K tile) it ends the tile loop
early after the tiles that hold the batch's longest context, but that dynamic early exit gives wrong attention: a
decode whose context crosses 8K tokens on a long ``--context-length`` (Qwen3-30B-A3B-Instruct-2507 at 262,144, TP8)
degenerates after its first token, and with the early exit off the output is right up to 240K tokens. Without it the
kernel reads every tile of the graph's prior, which the decode context ladder bounds."""

from dataclasses import dataclass

import nki
import torch
from nkilib.core.utils.common_types import QuantizationType
from nkilib.experimental.transformer.attention_block_tkg import attention_block_tkg
from torch import Tensor

from sglang_neuron.backend.stack import wrap_nki
from sglang_neuron.utils.hardware_config import get_logical_nc_config
from sglang_neuron.utils.neuron_utils import can_run_kernel


@dataclass
class PagedDecodeInputs:
    """A decode step's kernel inputs, shared by every layer: ``table`` [B, kv_heads, blocks] folded block ids (-1 for a
    block that starts at or after the row's new token), ``update_idx`` the new token's folded slot row per KV head
    ([B, kv_heads, 1], or [B, 1] with one KV head, as the kernel takes them), ``pos_ids`` [B, 1] fp32 its position and
    ``active_mask`` [1, B, q_heads, 1]."""

    table: Tensor
    update_idx: Tensor
    pos_ids: Tensor
    active_mask: Tensor


def paged_decode_inputs(block_table: Tensor, slots: Tensor, positions: Tensor, *, kv_heads: int, q_heads: int,
                        page: int) -> PagedDecodeInputs:
    """The kernel inputs of a decode step from the runtime's metadata: ``block_table`` [B, blocks] int32 (-1 past a
    row's pages), ``slots`` [B] the new token's slot, ``positions`` [B] its position. A padding row (slot 0, position 0)
    writes SGLang's reserved page 0 and attends nothing before itself."""
    rows, blocks = block_table.shape
    heads = torch.arange(kv_heads, device=block_table.device, dtype=torch.int64)
    pages = block_table.to(torch.int64)
    folded = pages[:, None, :] * kv_heads + heads[None, :, None]
    starts = torch.arange(blocks, device=block_table.device, dtype=torch.int64) * page
    positions = positions.to(torch.int64)
    # a block holds prior keys while it starts before the row's new token; the kernel masks the rest by position
    prior = starts[None, None, :] < positions[:, None, None]
    table = torch.where(prior, folded, torch.full_like(folded, -1)).to(torch.int32)
    slots = torch.clamp(slots.to(torch.int64), min=0)
    update = ((slots // page)[:, None] * kv_heads + heads[None, :]) * page + (slots % page)[:, None]
    return PagedDecodeInputs(
        table=table,
        update_idx=(update.unsqueeze(-1) if kv_heads > 1 else update).to(torch.int32),
        pos_ids=positions.to(torch.float32).view(rows, 1),
        active_mask=torch.ones((1, rows, q_heads, 1), dtype=torch.uint8, device=block_table.device),
    )


@nki.jit
def _attention_block_tkg(X, W_qkv, bias_qkv, q_norm, k_norm, norm_eps, cos, sin, active_blocks_table, K_cache,
                         V_cache, attention_mask, kv_cache_update_idx, pos_ids, softmax_scale, W_out):
    qk_norm = q_norm is not None
    return attention_block_tkg(
        X=X,
        X_hidden_dim_actual=None,
        rmsnorm_X_enabled=False,
        rmsnorm_X_eps=None,
        rmsnorm_X_gamma=None,
        W_qkv=W_qkv,
        bias_qkv=bias_qkv,
        quantization_type_qkv=QuantizationType.NONE,
        weight_dequant_scale_qkv=None,
        input_dequant_scale_qkv=None,
        rmsnorm_QK_pre_rope_enabled=qk_norm,
        rmsnorm_QK_pre_rope_eps=norm_eps if qk_norm else 0.0,
        rmsnorm_QK_pre_rope_W_Q=q_norm,
        rmsnorm_QK_pre_rope_W_K=k_norm,
        cos=cos,
        sin=sin,
        rope_contiguous_layout=True,
        rmsnorm_QK_post_rope_enabled=False,
        rmsnorm_QK_post_rope_eps=0.0,
        rmsnorm_QK_post_rope_W_Q=None,
        rmsnorm_QK_post_rope_W_K=None,
        K_cache_transposed=False,
        active_blocks_table=active_blocks_table,
        K_cache=K_cache,
        V_cache=V_cache,
        attention_mask=attention_mask,
        sink=None,
        update_cache=True,
        kv_cache_update_idx=kv_cache_update_idx,
        W_out=W_out,
        bias_out=None,
        quantization_type_out=QuantizationType.NONE,
        weight_dequant_scale_out=None,
        input_dequant_scale_out=None,
        transposed_out=False,
        out_in_sb=False,
        softmax_scale=softmax_scale,
        pos_ids=pos_ids,
    )


def attention_decode_paged(x: Tensor, *, w_qkv: Tensor, bias_qkv: Tensor | None, q_norm: Tensor | None,
                           k_norm: Tensor | None, norm_eps: float, cos: Tensor, sin: Tensor, k_cache: Tensor,
                           v_cache: Tensor, inputs: PagedDecodeInputs, softmax_scale: float, w_out: Tensor) -> Tensor:
    """One decode attention layer of x [B, 1, H] (already normed): returns [B, H], this rank's partial of the output
    projection. ``w_qkv`` [H, (q + 2 kv) * d] packs Q, K and V on its output dim, ``bias_qkv`` [1, (q + 2 kv) * d],
    ``q_norm``/``k_norm`` [1, d] the per-head RMSNorm gammas (None for a model without QK norm), cos/sin [d / 2, B, 1]
    the contiguous-halves RoPE tables (a partial RoPE pads its tail with cos 1 and sin 0), ``w_out`` [q * d, H]. The
    caches are written in place."""
    args = dict(W_qkv=w_qkv, bias_qkv=bias_qkv, q_norm=q_norm, k_norm=k_norm, norm_eps=norm_eps, cos=cos, sin=sin,
                active_blocks_table=inputs.table, K_cache=k_cache, V_cache=v_cache,
                attention_mask=inputs.active_mask, kv_cache_update_idx=inputs.update_idx, pos_ids=inputs.pos_ids,
                softmax_scale=softmax_scale, W_out=w_out)
    if not can_run_kernel(x):
        return _torch_attention_decode_paged(x, **args)
    out, _, _ = wrap_nki(_attention_block_tkg)[get_logical_nc_config()](x, **args)
    return out


def _rms(x: Tensor, gamma: Tensor, eps: float) -> Tensor:
    x32 = x.float()
    return (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps) * gamma.float().reshape(-1)).to(x.dtype)


def _torch_attention_decode_paged(x, *, W_qkv, bias_qkv, q_norm, k_norm, norm_eps, cos, sin, active_blocks_table,
                                  K_cache, V_cache, attention_mask, kv_cache_update_idx, pos_ids, softmax_scale,
                                  W_out) -> Tensor:
    """The kernel's result by plain attention over every block the table lists, keys masked by position. Static shapes
    and no ``.item()``, so it traces."""
    del attention_mask  # the reference masks by position over every listed block
    rows = x.shape[0]
    pages, kv_heads, page, d = K_cache.shape
    q_heads = W_out.shape[0] // d
    group = q_heads // kv_heads
    qkv = x.reshape(rows, -1) @ W_qkv
    if bias_qkv is not None:
        qkv = qkv + bias_qkv.reshape(-1)
    q = qkv[:, :q_heads * d].view(rows, q_heads, d)
    k = qkv[:, q_heads * d:(q_heads + kv_heads) * d].view(rows, kv_heads, d)
    v = qkv[:, (q_heads + kv_heads) * d:].view(rows, kv_heads, d)
    if q_norm is not None:
        q, k = _rms(q, q_norm, norm_eps), _rms(k, k_norm, norm_eps)
    c = cos.permute(1, 2, 0).reshape(rows, 1, d // 2).repeat(1, 1, 2).to(x.dtype)
    s = sin.permute(1, 2, 0).reshape(rows, 1, d // 2).repeat(1, 1, 2).to(x.dtype)

    def rope(t):
        return t * c + torch.cat((-t[..., d // 2:], t[..., :d // 2]), dim=-1) * s

    q, k = rope(q), rope(k)
    table = active_blocks_table.long()                                         # [B, kv, blocks]
    blocks = table.shape[-1]
    keys = K_cache.reshape(pages * kv_heads, page, d)[table.clamp(min=0)]      # [B, kv, blocks, page, d]
    values = V_cache.reshape(pages * kv_heads, page, d)[table.clamp(min=0)]
    keys = keys.reshape(rows, kv_heads, blocks * page, d).to(x.dtype)
    values = values.reshape(rows, kv_heads, blocks * page, d).to(x.dtype)
    key_pos = torch.arange(blocks * page, device=x.device)
    visible = (table.repeat_interleave(page, dim=-1) >= 0) & (key_pos[None, None, :] < pos_ids.reshape(rows, 1, 1))
    keys = torch.cat([keys, k.unsqueeze(2)], dim=2)                            # the row's own token last
    values = torch.cat([values, v.unsqueeze(2)], dim=2)
    visible = torch.cat([visible, torch.ones(rows, kv_heads, 1, dtype=torch.bool, device=x.device)], dim=-1)
    q = q.view(rows, kv_heads, group, d)
    scores = torch.einsum("bhgd,bhsd->bhgs", q.float(), keys.float()) * softmax_scale
    scores = scores.masked_fill(~visible[:, :, None, :], float("-inf"))
    out = torch.einsum("bhgs,bhsd->bhgd", torch.softmax(scores, dim=-1), values.float()).to(x.dtype)
    index = kv_cache_update_idx.reshape(-1).long()
    K_cache.view(-1, d).index_copy_(0, index, k.reshape(-1, d).to(K_cache.dtype))
    V_cache.view(-1, d).index_copy_(0, index, v.reshape(-1, d).to(V_cache.dtype))
    return out.reshape(rows, q_heads * d) @ W_out
