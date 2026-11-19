# SPDX-License-Identifier: Apache-2.0
"""nkilib's attention_block_tkg as MiMo-V2's decode runs it, for heads of 256 channels (config.KV_HEAD_DIM).

Above 128 channels nkilib takes the attention only: the in-kernel output projection and block-KV cache update need
``d_head <= 128``, so both wrappers pass ``W_out=None`` and ``update_cache=False``; the kernel returns each head's
attention output [B, heads, d, S] and the new token's K [d, B, S] and V [B, 1, S, d], which the graph projects and
scatters itself (attention.MiMoAttention.decode). ``decode_masked`` takes a pre-built mask
(ops/attention/attention_decode_mask, in the layout attention.qk_swap_layout gives); ``decode_posid`` masks by position. It
passes no ``max_context_len``: the flash-attention early exit it enables is wrong past an 8K prior
(sgl-project/sglang-neuron#4)."""

import nki
import nki.collectives as ncc
from nkilib.core.attention.attention_kv_parallel_segmented_cte import attention_kv_parallel_segmented_cte
from nkilib.core.utils.common_types import QuantizationType
from nkilib.experimental.transformer.attention_block_tkg import attention_block_tkg
from sglang_neuron.models.mimo_v2.kernels.cp.attention_block_tkg_sharding import KVDPCollectiveMode

from sglang_neuron.models.mimo_v2.kernels.cp.attention_block_tkg import attention_block_tkg as attention_block_tkg_cp

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


@nki.jit
def decode_masked_dcp(X, X_hidden_dim_actual, rmsnorm_X_eps, rmsnorm_X_gamma, W_qkv, cos, sin, active_blocks_table,
                      K_cache, V_cache, attention_mask, sink, softmax_scale, k_scale, v_scale, dcp_size, dcp_mesh,
                      dcp_mode):
    """``decode_masked`` over the DCP group's KV shards (``CP = dcp_size``): the kernel gathers the group's query heads,
    attends over this rank's share (the group's mask: models/gpt_oss/attention_masks, the decoded token counted on its
    owner), merges the partial softmaxes across the group and hands each rank its own heads back. ``sink`` is the
    group's (distributed.decode_context_parallel.group_sinks), ``k_scale`` / ``v_scale`` None over a bf16 cache."""
    cells = []
    for cell in dcp_mesh:  # NKI's tracer takes no list comprehension
        cells.append(list(cell))
    # nkilib's CP path holds Q and the output as [d_head, rows] tiles (heads of at most 128 channels); the vendored copy
    # d-tiles it for MiMo's 256 (kernels/cp)
    return attention_block_tkg_cp(
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
        CP=dcp_size, CP_replica_group=ncc.ReplicaGroup(cells), CP_collective_mode=dcp_mode,
    )


@nki.jit
def prefill_kv_parallel(q, k_cache, v_cache, block_tables, kvp_q_offset, kvp_rank_id, valid_num_prior_tokens, k_scale,
                        v_scale, scale, block_size, seg_size, dcp_size, dcp_mesh, fp8_packed):
    """A full layer's prefill over the DCP group's KV shards (nkilib's KV-parallel segmented attention, Hy3's
    models/hy_v3/attention._prefill_kv_parallel): no sink, no window."""
    cells = []
    for cell in dcp_mesh:
        cells.append(list(cell))
    return attention_kv_parallel_segmented_cte(
        q=q, k_cache=k_cache, v_cache=v_cache, block_tables=block_tables, kvp_q_offset=kvp_q_offset,
        replica_groups=ncc.ReplicaGroup(cells), group_size=dcp_size, block_size=block_size, seg_size=seg_size,
        scale=scale, global_q_offset=0, kvp_rank_id=kvp_rank_id, kvp_group_size=dcp_size, apc_mode=True,
        valid_num_prior_tokens=valid_num_prior_tokens, fp8_packed=fp8_packed, k_scale=k_scale, v_scale=v_scale,
    )


@nki.jit
def decode_posid_kvdp(X, X_hidden_dim_actual, rmsnorm_X_eps, rmsnorm_X_gamma, W_qkv, cos, sin, active_blocks_table,
                      K_cache, V_cache, attention_mask, sink, softmax_scale, pos_ids, kvdp_rank, kvdp_mesh, kvdp,
                      k_scale=None, v_scale=None):
    """``decode_posid`` with the batch's attention split over the ``kvdp`` ranks of each group of ``kvdp_mesh`` that
    hold the same KV head (nkilib's KV data parallelism): the group's query heads are gathered, each rank attends
    ``B / kvdp`` rows (``active_blocks_table`` / ``pos_ids`` / ``attention_mask`` its rows'), and every rank gets its
    own heads back over all B rows, with every row's new K [d, B, S] / V [B, 1, S, d] (the vendored kernel's d-tiled
    KVDP path, kernels/cp: nkilib's takes heads of at most 128 channels)."""
    cells = []
    for cell in kvdp_mesh:  # NKI's tracer takes no list comprehension
        cells.append(list(cell))
    return attention_block_tkg_cp(
        X=X, X_hidden_dim_actual=X_hidden_dim_actual, rmsnorm_X_enabled=True, rmsnorm_X_eps=rmsnorm_X_eps,
        rmsnorm_X_gamma=rmsnorm_X_gamma, W_qkv=W_qkv, bias_qkv=None, quantization_type_qkv=QuantizationType.NONE,
        weight_dequant_scale_qkv=None, input_dequant_scale_qkv=None, rmsnorm_QK_pre_rope_enabled=False,
        rmsnorm_QK_pre_rope_eps=0.0, rmsnorm_QK_pre_rope_W_Q=None, rmsnorm_QK_pre_rope_W_K=None, cos=cos, sin=sin,
        rope_contiguous_layout=True, rmsnorm_QK_post_rope_enabled=False, rmsnorm_QK_post_rope_eps=0.0,
        rmsnorm_QK_post_rope_W_Q=None, rmsnorm_QK_post_rope_W_K=None, K_cache_transposed=False,
        active_blocks_table=active_blocks_table, K_cache=K_cache, V_cache=V_cache, attention_mask=attention_mask,
        sink=sink, update_cache=False, kv_cache_update_idx=None, W_out=None, bias_out=None,
        quantization_type_out=QuantizationType.NONE, weight_dequant_scale_out=None, input_dequant_scale_out=None,
        transposed_out=False, out_in_sb=False, softmax_scale=softmax_scale, pos_ids=pos_ids, k_scale=k_scale,
        v_scale=v_scale, KVDP=kvdp, KVDP_replica_group=ncc.ReplicaGroup(cells),
        KVDP_collective_mode=KVDPCollectiveMode.ALL_TO_ALL, KVDP_rank=kvdp_rank,
    )
