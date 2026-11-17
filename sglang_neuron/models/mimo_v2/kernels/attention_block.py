# SPDX-License-Identifier: Apache-2.0
"""nkilib's attention_block_tkg as MiMo-V2's decode runs it, for heads of 256 channels (config.KV_HEAD_DIM).

Above 128 channels nkilib takes the attention only: the in-kernel output projection and block-KV cache update need
``d_head <= 128``, so both wrappers pass ``W_out=None`` and ``update_cache=False``; the kernel returns each head's
attention output [B, heads, d, S] and the new token's K [d, B, S] and V [B, 1, S, d], which the graph projects and
scatters itself (attention.MiMoAttention.decode). ``decode_masked`` takes a pre-built mask
(ops/attention/attention_decode_mask, in the layout qk_swap_layout gives); ``decode_posid`` masks by position. It
passes no ``max_context_len``: the flash-attention early exit it enables is wrong past an 8K prior
(sgl-project/sglang-neuron#4)."""

import nki
from nkilib.core.utils.common_types import QuantizationType
from nkilib.experimental.transformer.attention_block_tkg import attention_block_tkg

# each wrapper spells its call out: NKI's tracer takes a plain call, not a helper's keyword dict


@nki.jit
def decode_masked(X, X_hidden_dim_actual, rmsnorm_X_eps, rmsnorm_X_gamma, W_qkv, cos, sin, active_blocks_table,
                  K_cache, V_cache, attention_mask, sink, softmax_scale):
    return attention_block_tkg(
        X=X, X_hidden_dim_actual=X_hidden_dim_actual, rmsnorm_X_enabled=True, rmsnorm_X_eps=rmsnorm_X_eps,
        rmsnorm_X_gamma=rmsnorm_X_gamma, W_qkv=W_qkv, bias_qkv=None, quantization_type_qkv=QuantizationType.NONE,
        weight_dequant_scale_qkv=None, input_dequant_scale_qkv=None, rmsnorm_QK_pre_rope_enabled=False,
        rmsnorm_QK_pre_rope_eps=0.0, rmsnorm_QK_pre_rope_W_Q=None, rmsnorm_QK_pre_rope_W_K=None, cos=cos, sin=sin,
        rope_contiguous_layout=True, rmsnorm_QK_post_rope_enabled=False, rmsnorm_QK_post_rope_eps=0.0,
        rmsnorm_QK_post_rope_W_Q=None, rmsnorm_QK_post_rope_W_K=None, K_cache_transposed=False,
        active_blocks_table=active_blocks_table, K_cache=K_cache, V_cache=V_cache, attention_mask=attention_mask,
        sink=sink, update_cache=False, kv_cache_update_idx=None, W_out=None, bias_out=None,
        quantization_type_out=QuantizationType.NONE, weight_dequant_scale_out=None, input_dequant_scale_out=None,
        transposed_out=False, out_in_sb=False, softmax_scale=softmax_scale,
    )


@nki.jit
def decode_posid(X, X_hidden_dim_actual, rmsnorm_X_eps, rmsnorm_X_gamma, W_qkv, cos, sin, active_blocks_table,
                 K_cache, V_cache, attention_mask, sink, softmax_scale, pos_ids):
    return attention_block_tkg(
        X=X, X_hidden_dim_actual=X_hidden_dim_actual, rmsnorm_X_enabled=True, rmsnorm_X_eps=rmsnorm_X_eps,
        rmsnorm_X_gamma=rmsnorm_X_gamma, W_qkv=W_qkv, bias_qkv=None, quantization_type_qkv=QuantizationType.NONE,
        weight_dequant_scale_qkv=None, input_dequant_scale_qkv=None, rmsnorm_QK_pre_rope_enabled=False,
        rmsnorm_QK_pre_rope_eps=0.0, rmsnorm_QK_pre_rope_W_Q=None, rmsnorm_QK_pre_rope_W_K=None, cos=cos, sin=sin,
        rope_contiguous_layout=True, rmsnorm_QK_post_rope_enabled=False, rmsnorm_QK_post_rope_eps=0.0,
        rmsnorm_QK_post_rope_W_Q=None, rmsnorm_QK_post_rope_W_K=None, K_cache_transposed=False,
        active_blocks_table=active_blocks_table, K_cache=K_cache, V_cache=V_cache, attention_mask=attention_mask,
        sink=sink, update_cache=False, kv_cache_update_idx=None, W_out=None, bias_out=None,
        quantization_type_out=QuantizationType.NONE, weight_dequant_scale_out=None, input_dequant_scale_out=None,
        transposed_out=False, out_in_sb=False, softmax_scale=softmax_scale, pos_ids=pos_ids,
    )


@nki.jit
def decode_masked_fp8(X, X_hidden_dim_actual, rmsnorm_X_eps, rmsnorm_X_gamma, W_qkv, cos, sin, active_blocks_table,
                      K_cache, V_cache, attention_mask, sink, softmax_scale, k_scale, v_scale):
    """``decode_masked`` over an FP8 cache (token-major K and V, ``k_scale`` / ``v_scale`` [1, 1] fp32): the kernel
    quantizes the new token's K/V (clip(x * scale) to e4m3), so the K [d, B, S] and V it returns are FP8; the caller
    folds 1 / k_scale into ``softmax_scale`` and divides the output by v_scale."""
    return attention_block_tkg(
        X=X, X_hidden_dim_actual=X_hidden_dim_actual, rmsnorm_X_enabled=True, rmsnorm_X_eps=rmsnorm_X_eps,
        rmsnorm_X_gamma=rmsnorm_X_gamma, W_qkv=W_qkv, bias_qkv=None, quantization_type_qkv=QuantizationType.NONE,
        weight_dequant_scale_qkv=None, input_dequant_scale_qkv=None, rmsnorm_QK_pre_rope_enabled=False,
        rmsnorm_QK_pre_rope_eps=0.0, rmsnorm_QK_pre_rope_W_Q=None, rmsnorm_QK_pre_rope_W_K=None, cos=cos, sin=sin,
        rope_contiguous_layout=True, rmsnorm_QK_post_rope_enabled=False, rmsnorm_QK_post_rope_eps=0.0,
        rmsnorm_QK_post_rope_W_Q=None, rmsnorm_QK_post_rope_W_K=None, K_cache_transposed=False,
        active_blocks_table=active_blocks_table, K_cache=K_cache, V_cache=V_cache, attention_mask=attention_mask,
        sink=sink, update_cache=False, kv_cache_update_idx=None, W_out=None, bias_out=None,
        quantization_type_out=QuantizationType.NONE, weight_dequant_scale_out=None, input_dequant_scale_out=None,
        transposed_out=False, out_in_sb=False, softmax_scale=softmax_scale, k_scale=k_scale, v_scale=v_scale,
    )
