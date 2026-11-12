# SPDX-License-Identifier: Apache-2.0
"""GPT-OSS decode attention: nkilib's attention_block_tkg kernel runs the input RMSNorm, the QKV projection, RoPE,
attention with sinks over the paged cache, the K/V write and the output projection in one call. The torch path is the
reference the CPU tests run.

Under DCP the ranks of a group hold the same KV head and different tokens of each sequence
(distributed.decode_context_parallel). The kernel gathers the group's query heads, attends over this rank's share of
the prior and merges the partial softmaxes, handing each rank back its own heads; the torch path does the same with an
all-gather and a reduce-scatter."""


import nki
import nki.collectives as ncc
import torch
from nkilib.core.utils.common_types import QuantizationType
from nkilib.experimental.transformer.attention_block_tkg import attention_block_tkg
from nkilib.experimental.transformer.attention_block_tkg_sharding import CPCollectiveMode
from torch import Tensor

from sglang_neuron.backend.stack import wrap_nki
from sglang_neuron.ops.attention.attention_decode_mask import P_MAX, resize_block_len
from sglang_neuron.utils.dtype_utils import FP8_CLAMP_MAX
from sglang_neuron.utils.hardware_config import get_logical_nc_config
from sglang_neuron.utils.neuron_utils import can_run_kernel


@nki.jit
def _attention_block_tkg(X, X_hidden_dim_actual, rmsnorm_X_eps, rmsnorm_X_gamma, W_qkv, bias_qkv, cos, sin,
                         active_blocks_table, K_cache, V_cache, attention_mask, sink, softmax_scale,
                         kv_cache_update_idx, k_scale, v_scale, fp8_packed, W_out, bias_out):
    return attention_block_tkg(
        X=X,
        X_hidden_dim_actual=X_hidden_dim_actual,
        rmsnorm_X_enabled=True,
        rmsnorm_X_eps=rmsnorm_X_eps,
        rmsnorm_X_gamma=rmsnorm_X_gamma,
        W_qkv=W_qkv,
        bias_qkv=bias_qkv,
        quantization_type_qkv=QuantizationType.NONE,
        weight_dequant_scale_qkv=None,
        input_dequant_scale_qkv=None,
        rmsnorm_QK_pre_rope_enabled=False,
        rmsnorm_QK_pre_rope_eps=1e-6,
        rmsnorm_QK_pre_rope_W_Q=None,
        rmsnorm_QK_pre_rope_W_K=None,
        cos=cos,
        sin=sin,
        rope_contiguous_layout=True,
        rmsnorm_QK_post_rope_enabled=False,
        rmsnorm_QK_post_rope_eps=1e-6,
        rmsnorm_QK_post_rope_W_Q=None,
        rmsnorm_QK_post_rope_W_K=None,
        K_cache_transposed=False,
        active_blocks_table=active_blocks_table,
        K_cache=K_cache,
        V_cache=V_cache,
        attention_mask=attention_mask,
        sink=sink,
        update_cache=True,
        kv_cache_update_idx=kv_cache_update_idx,
        W_out=W_out,
        bias_out=bias_out,
        quantization_type_out=QuantizationType.NONE,
        weight_dequant_scale_out=None,
        input_dequant_scale_out=None,
        transposed_out=False,
        out_in_sb=False,
        softmax_scale=softmax_scale,
        fp8_packed=fp8_packed,
        k_scale=k_scale,
        v_scale=v_scale,
    )


@nki.jit
def _attention_block_tkg_dcp(X, X_hidden_dim_actual, rmsnorm_X_eps, rmsnorm_X_gamma, W_qkv, bias_qkv, cos, sin,
                             active_blocks_table, K_cache, V_cache, attention_mask, sink, softmax_scale,
                             kv_cache_update_idx, k_scale, v_scale, fp8_packed, W_out, bias_out, dcp_size, dcp_mesh,
                             dcp_mode):
    """_attention_block_tkg over a DCP group. ``dcp_mesh`` holds every DCP group's ranks as a tuple of tuples (a
    ReplicaGroup cannot cross into the trace), so all ranks trace one SPMD program. A separate kernel keeps the non-DCP
    trace unchanged."""
    cells = []
    for cell in dcp_mesh:  # NKI's tracer takes no list comprehension
        cells.append(list(cell))
    return attention_block_tkg(
        X=X,
        X_hidden_dim_actual=X_hidden_dim_actual,
        rmsnorm_X_enabled=True,
        rmsnorm_X_eps=rmsnorm_X_eps,
        rmsnorm_X_gamma=rmsnorm_X_gamma,
        W_qkv=W_qkv,
        bias_qkv=bias_qkv,
        quantization_type_qkv=QuantizationType.NONE,
        weight_dequant_scale_qkv=None,
        input_dequant_scale_qkv=None,
        rmsnorm_QK_pre_rope_enabled=False,
        rmsnorm_QK_pre_rope_eps=1e-6,
        rmsnorm_QK_pre_rope_W_Q=None,
        rmsnorm_QK_pre_rope_W_K=None,
        cos=cos,
        sin=sin,
        rope_contiguous_layout=True,
        rmsnorm_QK_post_rope_enabled=False,
        rmsnorm_QK_post_rope_eps=1e-6,
        rmsnorm_QK_post_rope_W_Q=None,
        rmsnorm_QK_post_rope_W_K=None,
        K_cache_transposed=False,
        active_blocks_table=active_blocks_table,
        K_cache=K_cache,
        V_cache=V_cache,
        attention_mask=attention_mask,
        sink=sink,
        update_cache=True,
        kv_cache_update_idx=kv_cache_update_idx,
        W_out=W_out,
        bias_out=bias_out,
        quantization_type_out=QuantizationType.NONE,
        weight_dequant_scale_out=None,
        input_dequant_scale_out=None,
        transposed_out=False,
        out_in_sb=False,
        softmax_scale=softmax_scale,
        fp8_packed=fp8_packed,
        k_scale=k_scale,
        v_scale=v_scale,
        CP=dcp_size,
        CP_replica_group=ncc.ReplicaGroup(cells),
        CP_collective_mode=dcp_mode,
    )


def dcp_collective_mode(size: int) -> CPCollectiveMode:
    """The collective that hands each rank its heads back after the DCP merge: all-to-all from 4 ranks (nkilib's
    minimum at LNC2), reduce-scatter below. LNC1 would need 8, but DCP runs at LNC2 only (the giga-kernel decode, the
    one LNC1 graph, is refused under DCP)."""
    return CPCollectiveMode.ALL_TO_ALL if size >= 4 else CPCollectiveMode.REDUCE_SCATTER


def attention_decode(
    X: Tensor,
    *,
    X_hidden_dim_actual: int,
    rmsnorm_X_gamma: Tensor,
    rmsnorm_X_eps: float,
    W_qkv: Tensor,
    bias_qkv: Tensor,
    cos: Tensor,
    sin: Tensor,
    active_blocks_table: Tensor,
    K_cache: Tensor,
    V_cache: Tensor,
    attention_mask: Tensor,
    sink: Tensor,
    softmax_scale: float,
    kv_cache_update_idx: Tensor,
    k_scale: Tensor,
    v_scale: Tensor,
    fp8_packed: bool,
    W_out: Tensor,
    bias_out: Tensor,
    return_k_cache: bool = False,
    dcp_group=None,
):
    """One decode attention layer for one KV head per rank; writes the step's K/V into the caches.

    Shapes: X [B, S, H] (H zero-padded past ``X_hidden_dim_actual``), cos/sin [d_head // 2, B, S],
    active_blocks_table [B, blocks] int32 with -1 past each row's pages, the FP8 V_cache [pages, 1, page, d_head], the
    FP8 K_cache the same or, with ``fp8_packed``, [pages, 1, page // 2, d_head, 2], attention_mask [blocks * page, B,
    q_heads, S] from gen_attention_decode_mask, sink [q_heads, 1], kv_cache_update_idx [B, S] uint32 slots. Returns
    [B * S, H].

    ``return_k_cache`` also returns the K cache the kernel wrote. A caller that passes a K cache computed in the graph
    needs it: the kernel's in-place write to a tensor that is not a graph input is not threaded back.

    ``dcp_group`` (D ranks): the caches hold this rank's share of the prior, indexed by virtual page in
    ``active_blocks_table``; ``attention_mask`` covers the gathered query heads ([blocks * page, B, q_heads * D, S],
    the active token on its owner only) and ``sink`` their sinks (decode_context_parallel.group_sinks);
    ``kv_cache_update_idx`` is this rank's slot per token, the reserved slot where another rank keeps it. Returns this
    rank's heads, as without DCP.
    """
    dcp = dcp_group is not None and dcp_group.world_size > 1
    args = dict(X_hidden_dim_actual=X_hidden_dim_actual, rmsnorm_X_eps=rmsnorm_X_eps, rmsnorm_X_gamma=rmsnorm_X_gamma,
                W_qkv=W_qkv, bias_qkv=bias_qkv, cos=cos, sin=sin, active_blocks_table=active_blocks_table,
                K_cache=K_cache, V_cache=V_cache, sink=sink, softmax_scale=softmax_scale,
                kv_cache_update_idx=kv_cache_update_idx, k_scale=k_scale, v_scale=v_scale, fp8_packed=fp8_packed,
                W_out=W_out, bias_out=bias_out)
    if not can_run_kernel(X):
        output = _torch_attention_decode(X, attention_mask=attention_mask, dcp_group=dcp_group if dcp else None,
                                         **args)
        return (output, K_cache) if return_k_cache else output
    # the mask's [S_ctx, B, q_heads, S] is the non-swapped QK path's layout, the one the model requires
    # (sglang_model._check_target)
    if dcp:
        from sglang_neuron.backend.stack import resolve_full_mesh

        mesh = tuple(tuple(int(r) for r in cell) for cell in resolve_full_mesh(dcp_group))
        output, k_written, _ = wrap_nki(_attention_block_tkg_dcp)[get_logical_nc_config()](
            X, attention_mask=attention_mask, dcp_size=dcp_group.world_size, dcp_mesh=mesh,
            dcp_mode=dcp_collective_mode(dcp_group.world_size), **args)
    else:
        output, k_written, _ = wrap_nki(_attention_block_tkg)[get_logical_nc_config()](
            X, attention_mask=attention_mask, **args)
    return (output, k_written) if return_k_cache else output


def packed_fp8_viable_for_bucket(block_len: int, bs: int, q_head: int, s_active: int, s_prior: int) -> bool:
    """Whether the kernel can read the packed FP8 K cache at this decode bucket. The kernel shrinks the page so the
    pages of a row tile its partitions (resize_block_len), and the packed layout holds token pairs, so it needs the
    shrunk page to keep at least two tokens."""
    return block_len > 0 and s_prior > 0 and resize_block_len(block_len, bs, q_head, s_active, s_prior) >= 2


def unswizzle_packed_k(k_packed: Tensor) -> Tensor:
    """[..., page // 2, d_head, 2] packed K (token pairs on the last dim) to [..., page, d_head]."""
    *lead, half, d_head, _ = k_packed.shape
    return k_packed.transpose(-1, -2).reshape(*lead, half * 2, d_head).contiguous()


def swizzle_packed_k(k: Tensor) -> Tensor:
    """[..., page, d_head] K to the packed [..., page // 2, d_head, 2]; the inverse of unswizzle_packed_k."""
    *lead, page, d_head = k.shape
    return k.reshape(*lead, page // 2, 2, d_head).transpose(-1, -2).contiguous()


def _torch_attention_decode(X, *, X_hidden_dim_actual, rmsnorm_X_eps, rmsnorm_X_gamma, W_qkv, bias_qkv, cos, sin,
                            active_blocks_table, K_cache, V_cache, attention_mask, sink, softmax_scale,
                            kv_cache_update_idx, k_scale, v_scale, fp8_packed, W_out, bias_out,
                            dcp_group=None) -> Tensor:
    B, S, _ = X.shape
    pages, _, page, d_head = V_cache.shape
    q_heads = W_qkv.shape[1] // d_head - 2
    blocks = active_blocks_table.shape[-1]
    s_ctx = blocks * page

    x = X.to(torch.float32)
    x = x * torch.rsqrt(x.pow(2).sum(-1, keepdim=True) / X_hidden_dim_actual + rmsnorm_X_eps)
    qkv = (x * rmsnorm_X_gamma.to(torch.float32)).to(X.dtype) @ W_qkv + bias_qkv
    q = qkv[..., :q_heads * d_head].view(B, S, q_heads, d_head).transpose(1, 2)
    k = qkv[..., q_heads * d_head:(q_heads + 1) * d_head].view(B, S, 1, d_head).transpose(1, 2)
    v = qkv[..., (q_heads + 1) * d_head:].view(B, S, 1, d_head).transpose(1, 2)

    # RoPE over contiguous halves; cos/sin [d_head // 2, B, S] -> [B, 1, S, d_head]
    cos_full = cos.permute(1, 2, 0).unsqueeze(1).repeat(1, 1, 1, 2).to(q.dtype)
    sin_full = sin.permute(1, 2, 0).unsqueeze(1).repeat(1, 1, 1, 2).to(q.dtype)

    def rope(t):
        half = t.shape[-1] // 2
        return t * cos_full + torch.cat((-t[..., half:], t[..., :half]), dim=-1) * sin_full

    q, k = rope(q), rope(k)
    if dcp_group is not None:
        # gather the DCP group's query heads in rank order; they attend this rank's share of the prior
        from sglang_neuron.parallel import collectives

        q = collectives.all_gather(dcp_group, q.contiguous(), dim=1)
        q_heads = q.shape[1]
    # the cache holds fp8 of x * scale; the caller folds the dequantization into softmax_scale and W_out
    k = (k * k_scale).clamp(-FP8_CLAMP_MAX, FP8_CLAMP_MAX).to(torch.float8_e4m3fn)
    v = (v * v_scale).clamp(-FP8_CLAMP_MAX, FP8_CLAMP_MAX).to(torch.float8_e4m3fn)

    # the prior from the row's pages (-1 past its pages reads page 0, which the mask hides), with the step's tokens in
    # the last S slots, where the mask's active part addresses them
    rows = active_blocks_table.clamp(min=0).long().reshape(-1)
    k_unpacked = unswizzle_packed_k(K_cache) if fp8_packed else K_cache
    k_ctx = k_unpacked.to(X.dtype)[rows].reshape(B, s_ctx, d_head).unsqueeze(1)
    v_ctx = V_cache.to(X.dtype)[rows].reshape(B, s_ctx, d_head).unsqueeze(1)
    k_ctx[:, :, -S:] = k.to(X.dtype)
    v_ctx[:, :, -S:] = v.to(X.dtype)

    scores = (q @ k_ctx.transpose(-2, -1) * softmax_scale).float()  # [B, q_heads, S, s_ctx]
    scores = scores.masked_fill(_mask_in_sequence_order(attention_mask, page, B, q_heads, S, s_ctx) == 0,
                                float("-inf"))
    scores = torch.cat([scores, sink.float().view(1, q_heads, 1, 1).expand(B, -1, S, 1)], dim=-1)
    weights = torch.softmax(scores, dim=-1)[..., :-1].to(q.dtype)
    attended = weights @ v_ctx  # [B, q_heads, S, d_head]
    if dcp_group is not None:
        # rescale each rank's partial softmax to the group's log-sum-exp, then sum, each rank keeping its own heads
        from sglang_neuron.parallel import collectives

        partial = torch.logsumexp(scores, dim=-1)  # [B, q_heads, S]
        merged = torch.logsumexp(collectives.all_gather(dcp_group, partial.unsqueeze(0).contiguous(), dim=0), dim=0)
        attended = (attended.float() * torch.exp(partial - merged).unsqueeze(-1)).to(attended.dtype)
        attended = collectives.reduce_scatter(dcp_group, attended.contiguous(), dim=1)
        q_heads = attended.shape[1]
    attended = attended.transpose(1, 2).reshape(B * S, q_heads * d_head)

    # the step's K/V into their slots; a slot outside the pool (-1, or its uint32 cast) goes to page 0, SGLang's
    # reserved page, which no table reads
    slots = kv_cache_update_idx.to(torch.int64).reshape(-1)
    slots = torch.where((slots < 0) | (slots >= pages * page), torch.zeros_like(slots), slots)
    head = torch.zeros_like(slots)
    index = (slots // page, head, slots % page)
    if fp8_packed:
        k_written = k_unpacked.to(X.dtype)
        k_written.index_put_(index, k.reshape(-1, d_head).to(X.dtype))
        K_cache.copy_(swizzle_packed_k(k_written.to(K_cache.dtype)))
    else:
        K_cache.index_put_(index, k.reshape(-1, d_head).to(K_cache.dtype))
    V_cache.index_put_(index, v.reshape(-1, d_head).to(V_cache.dtype))

    return attended @ W_out + bias_out


def _mask_in_sequence_order(mask: Tensor, page: int, B: int, q_heads: int, S: int, s_ctx: int) -> Tensor:
    """gen_attention_decode_mask's [s_ctx, B, q_heads, S] in the kernel's partition-major order, as [B, q_heads, S,
    s_ctx] in sequence order."""
    block = resize_block_len(page, B, q_heads, S, s_ctx)
    return mask.permute(1, 2, 3, 0).reshape(-1, block, P_MAX).swapaxes(-1, -2).reshape(B, q_heads, S, s_ctx)
