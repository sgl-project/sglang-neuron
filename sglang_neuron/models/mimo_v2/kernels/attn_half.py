# SPDX-License-Identifier: Apache-2.0
"""MiMo-V2's decode attention half in one NKI kernel: from the residual stream to this rank's partial of the output
projection, with the new token's K/V written into its pages, and no XLA op in between.

nkilib's ``attention_block_tkg`` runs the input RMSNorm, the QKV projection, RoPE and the attention (with the layer's
sinks) for heads of config.KV_HEAD_DIM channels; above 128 channels it takes neither the output projection nor the
cache write (``W_out=None``, ``update_cache=False``), so this kernel does both after it:

- the K/V write (after the attention, whose mask never reads the new tokens' slots from the cache, so the write cannot
  change what they attend to; a span's tokens see each other through the kernel's active K/V): the new token's K (``[d, B, 1]``, the kernel's layout for one KV head) is read
  token-major through a transposing access pattern, its V (``[B, 1, 1, d]``) as it is, and both rows go to each
  token's slot by an indirect DMA (with one KV head per rank a slot is its row of the flattened ``[pages x page, d]``
  cache); the two programs take V and K. A padding row's slot is 0, SGLang's reserved page;
- the output projection: the first ``v_head_dim`` channels of each head (the rest are the padding's zeros) against
  ``W_o`` ``[heads x v_head_dim, H]``, the two programs taking one half of H each.

The kernel returns the partial ``[B, H]`` and the two caches, so the caller's graph takes the caches as written in
place (an NKI kernel that writes an input and does not return it reports no alias, and the write can be dropped)."""

import nki
import nki.collectives as nccl
import nki.isa as nisa
import nki.language as nl
from nki.isa import oob_mode
from nkilib.core.utils.common_types import QuantizationType
from nkilib.experimental.transformer.attention_block_tkg import attention_block_tkg

P = 128
COLS = 512


def _attention(X, X_hidden_dim_actual, rmsnorm_X_eps, rmsnorm_X_gamma, W_qkv, cos, sin, active_blocks_table, K_cache,
               V_cache, attention_mask, sink, softmax_scale, k_scale=None, v_scale=None):
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


def _attention_posid(X, H, eps, gamma, W_qkv, cos, sin, table, K_cache, V_cache, active_mask, sink, softmax_scale,
                     k_scale, v_scale, pos, start=None):
    """_attention masked by position: ``pos`` [B, S] fp32 each token's prior length (the prior mask is the kernel's
    own), ``start`` [B, S] fp32 or None each token's window start (a sliding layer's, in its trimmed table's frame),
    ``active_mask`` [S, B, heads, S] uint8 the span's active part."""
    return attention_block_tkg(
        X=X, X_hidden_dim_actual=H, rmsnorm_X_enabled=True, rmsnorm_X_eps=eps, rmsnorm_X_gamma=gamma, W_qkv=W_qkv,
        bias_qkv=None, quantization_type_qkv=QuantizationType.NONE, weight_dequant_scale_qkv=None,
        input_dequant_scale_qkv=None, rmsnorm_QK_pre_rope_enabled=False, rmsnorm_QK_pre_rope_eps=0.0,
        rmsnorm_QK_pre_rope_W_Q=None, rmsnorm_QK_pre_rope_W_K=None, cos=cos, sin=sin, rope_contiguous_layout=True,
        rmsnorm_QK_post_rope_enabled=False, rmsnorm_QK_post_rope_eps=0.0, rmsnorm_QK_post_rope_W_Q=None,
        rmsnorm_QK_post_rope_W_K=None, K_cache_transposed=False, active_blocks_table=table, K_cache=K_cache,
        V_cache=V_cache, attention_mask=active_mask, sink=sink, update_cache=False, kv_cache_update_idx=None,
        W_out=None, bias_out=None, quantization_type_out=QuantizationType.NONE, weight_dequant_scale_out=None,
        input_dequant_scale_out=None, transposed_out=False, out_in_sb=False, softmax_scale=softmax_scale,
        pos_ids=pos, swa_start_pos_ids=start, k_scale=k_scale, v_scale=v_scale,
    )


def _attention_kvdp(X, H, eps, gamma, W_qkv, cos, sin, table, K_cache, V_cache, active_mask, sink, softmax_scale,
                    k_scale, v_scale, pos, rank, kvdp, kvdp_mesh):
    """The attention of this rank's B / kvdp rows of the batch (attn_half's ``kvdp``): its place in the group, its
    rows of the table and of ``pos`` gathered here, then the vendored kernel's KVDP path."""
    from sglang_neuron.models.mimo_v2.kernels.cp.attention_block_tkg import attention_block_tkg as attention_tkg_cp
    from sglang_neuron.models.mimo_v2.kernels.cp.attention_block_tkg_sharding import KVDPCollectiveMode

    B, S, _ = X.shape
    Ba = B // kvdp
    blocks = table.shape[1]
    # this rank's place in its group, rank mod kvdp (the group's ranks are consecutive; kvdp a power of two)
    r_i = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=r_i, src=rank.ap(pattern=[[1, 1], [1, 1]], offset=0))
    m_i = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=m_i, data=r_i, op0=nl.bitwise_and, operand0=kvdp - 1)
    m_u = nl.ndarray((1, 1), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=m_u, src=m_i)
    kvdp_rank = nl.ndarray((1,), dtype=nl.uint32, buffer=nl.shared_hbm, name="mimo_kvdp_rank")
    nisa.dma_copy(dst=kvdp_rank.reshape((1, 1)), src=m_u)
    # this rank's rows b = place x Ba + j of the table and of the positions
    pl_u = nl.ndarray((P, 1), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.dma_copy(dst=pl_u[0:Ba, 0:1], src=kvdp_rank.ap(pattern=[[0, Ba], [1, 1]], offset=0))
    base = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=base[0:Ba, 0:1], src=pl_u[0:Ba, 0:1])
    nisa.tensor_scalar(dst=base[0:Ba, 0:1], data=base[0:Ba, 0:1], op0=nl.multiply, operand0=float(Ba))
    row_f = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(row_f[0:Ba, 0:1], pattern=[[0, 1]], offset=0, channel_multiplier=1)
    nisa.tensor_tensor(dst=row_f[0:Ba, 0:1], data1=row_f[0:Ba, 0:1], data2=base[0:Ba, 0:1], op=nl.add)
    row = nl.ndarray((P, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=row[0:Ba, 0:1], src=row_f[0:Ba, 0:1])
    t_sb = nl.ndarray((P, blocks), dtype=table.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=t_sb[0:Ba, 0:blocks],
                  src=table.ap(pattern=[[blocks, Ba], [1, blocks]], offset=0, vector_offset=row[0:Ba, 0:1],
                               indirect_dim=0))
    table_mine = nl.ndarray((Ba, blocks), dtype=table.dtype, buffer=nl.shared_hbm, name="mimo_kvdp_table")
    nisa.dma_copy(dst=table_mine, src=t_sb[0:Ba, 0:blocks])
    p_sb = nl.ndarray((P, S), dtype=pos.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=p_sb[0:Ba, 0:S],
                  src=pos.ap(pattern=[[S, Ba], [1, S]], offset=0, vector_offset=row[0:Ba, 0:1], indirect_dim=0))
    pos_mine = nl.ndarray((Ba, S), dtype=pos.dtype, buffer=nl.shared_hbm, name="mimo_kvdp_pos")
    nisa.dma_copy(dst=pos_mine, src=p_sb[0:Ba, 0:S])
    cells = []
    for cell in kvdp_mesh:  # NKI's tracer takes no list comprehension
        cells.append(list(cell))
    return attention_tkg_cp(
        X=X, X_hidden_dim_actual=H, rmsnorm_X_enabled=True, rmsnorm_X_eps=eps, rmsnorm_X_gamma=gamma, W_qkv=W_qkv,
        bias_qkv=None, quantization_type_qkv=QuantizationType.NONE, weight_dequant_scale_qkv=None,
        input_dequant_scale_qkv=None, rmsnorm_QK_pre_rope_enabled=False, rmsnorm_QK_pre_rope_eps=0.0,
        rmsnorm_QK_pre_rope_W_Q=None, rmsnorm_QK_pre_rope_W_K=None, cos=cos, sin=sin, rope_contiguous_layout=True,
        rmsnorm_QK_post_rope_enabled=False, rmsnorm_QK_post_rope_eps=0.0, rmsnorm_QK_post_rope_W_Q=None,
        rmsnorm_QK_post_rope_W_K=None, K_cache_transposed=False, active_blocks_table=table_mine, K_cache=K_cache,
        V_cache=V_cache, attention_mask=active_mask, sink=sink, update_cache=False, kv_cache_update_idx=None,
        W_out=None, bias_out=None, quantization_type_out=QuantizationType.NONE, weight_dequant_scale_out=None,
        input_dequant_scale_out=None, transposed_out=False, out_in_sb=False, softmax_scale=softmax_scale,
        pos_ids=pos_mine, k_scale=k_scale, v_scale=v_scale, KVDP=kvdp, KVDP_replica_group=nccl.ReplicaGroup(cells),
        KVDP_collective_mode=KVDPCollectiveMode.ALL_TO_ALL, KVDP_rank=kvdp_rank,
    )


def _attention_dcp(X, H, eps, gamma, W_qkv, cos, sin, table, K_cache, V_cache, active_mask, sink, softmax_scale,
                   k_scale, v_scale, pos, start, dcp, dcp_mesh, dcp_mode):
    """The attention over the DCP group's KV shards (attn_half's ``dcp``): masked by position over this rank's share of
    the prior (``pos`` the runtime's dcp_kv_end, ``start`` its dcp_kv_start on a sliding layer), the span's active part
    gated by the row's owner in ``active_mask``, the group's query heads gathered and the partial softmaxes merged
    across it by the vendored kernel's CP path (kernels/cp: nkilib's takes heads of at most 128 channels)."""
    from sglang_neuron.models.mimo_v2.kernels.cp.attention_block_tkg import attention_block_tkg as attention_tkg_cp

    cells = []
    for cell in dcp_mesh:  # NKI's tracer takes no list comprehension
        cells.append(list(cell))
    return attention_tkg_cp(
        X=X, X_hidden_dim_actual=H, rmsnorm_X_enabled=True, rmsnorm_X_eps=eps, rmsnorm_X_gamma=gamma, W_qkv=W_qkv,
        bias_qkv=None, quantization_type_qkv=QuantizationType.NONE, weight_dequant_scale_qkv=None,
        input_dequant_scale_qkv=None, rmsnorm_QK_pre_rope_enabled=False, rmsnorm_QK_pre_rope_eps=0.0,
        rmsnorm_QK_pre_rope_W_Q=None, rmsnorm_QK_pre_rope_W_K=None, cos=cos, sin=sin, rope_contiguous_layout=True,
        rmsnorm_QK_post_rope_enabled=False, rmsnorm_QK_post_rope_eps=0.0, rmsnorm_QK_post_rope_W_Q=None,
        rmsnorm_QK_post_rope_W_K=None, K_cache_transposed=False, active_blocks_table=table, K_cache=K_cache,
        V_cache=V_cache, attention_mask=active_mask, sink=sink, update_cache=False, kv_cache_update_idx=None,
        W_out=None, bias_out=None, quantization_type_out=QuantizationType.NONE, weight_dequant_scale_out=None,
        input_dequant_scale_out=None, transposed_out=False, out_in_sb=False, softmax_scale=softmax_scale,
        pos_ids=pos, swa_start_pos_ids=start, k_scale=k_scale, v_scale=v_scale, CP=dcp,
        CP_replica_group=nccl.ReplicaGroup(cells), CP_collective_mode=dcp_mode,
    )


@nki.jit
def attn_half(X, rmsnorm_X_gamma, W_qkv, cos, sin, active_blocks_table, K_cache, V_cache, attention_mask, sink,
              slots, W_o, k_scale=None, v_scale=None, rmsnorm_X_eps=1e-6, softmax_scale=1.0, v_head_dim=128,
              out_in_sbuf=False, pos=None, rank=None, kvdp=1, kvdp_mesh=None, start=None, dcp=1, dcp_mesh=None,
              dcp_mode=None):
    """X [B, S, H] bf16 (S tokens a row: 1, or a NEURON_MTP span, which attend each other causally under the span
    mask); W_qkv [H, (heads + 2) x d]; cos / sin [d / 2, B, S]; the caches [pages, 1, page, d]; the mask in the
    kernel's layout; sink [heads, 1] fp32 or None; slots [B x S, 1] int32 (row-major: token s of row b at b x S + s);
    W_o [heads x v_head_dim, H] bf16. Over an FP8 cache (``k_scale`` / ``v_scale`` [1, 1] fp32) the attention
    quantizes the new tokens' K/V, which are written as they come; the caller folds 1 / k_scale into
    ``softmax_scale`` and 1 / v_scale into ``W_o``. Returns (partial [B x S, H] bf16, K_cache, V_cache); with
    ``out_in_sbuf`` (inside another kernel: layer.py) the partial is this program's half of H in SBUF instead, rows
    ``0 .. B x S - 1`` of a [128, H / programs] tile, row-major.

    ``kvdp`` > 1 (the full layers at a batch of at least 4: model.MiMoV2Graph._layer_call) splits the batch's attention
    over the ``kvdp`` ranks of each group of ``kvdp_mesh`` that hold the same KV head (kernels/attention_block.
    decode_posid_kvdp): each attends B / kvdp rows, masked by position (``pos`` [B, S] fp32, each token's prior length;
    ``attention_mask`` then the span's active part, [S, B / kvdp, heads x kvdp, S] uint8), and this rank's table rows
    and place in its group come from ``rank`` [1] int32, its attention-TP rank, inside the kernel. Every row's new K / V
    come back, so every rank writes them all as before.

    ``dcp`` > 1 attends over the DCP group's KV shards (_attention_dcp): ``pos`` / ``start`` this rank's share's bounds
    and ``attention_mask`` the owner-gated active part; the new K / V come back for every row, and ``slots`` (0 where
    another rank keeps the token) writes each on its owner."""
    B, S, H = X.shape
    R = B * S
    pages, _, page, d = K_cache.shape
    heads = W_o.shape[0] // v_head_dim
    n_prg = nl.num_programs(axes=0)
    pid = nl.program_id(axis=0)
    HH = H // n_prg
    h0 = pid * HH

    # the output projection's weights for this program's half of H, loading while the attention runs (each one's
    # load sat between the attention and its matmul)
    w_all = nl.ndarray((P, heads, HH), dtype=W_o.dtype, buffer=nl.sbuf)
    for h in range(heads):
        nisa.dma_copy(dst=w_all[0:v_head_dim, h, 0:HH],
                      src=W_o.ap(pattern=[[H, v_head_dim], [1, HH]], offset=h * v_head_dim * H + h0),
                      dge_mode=nisa.dge_mode.hwdge)

    if dcp > 1:
        heads_out, k_new, v_new = _attention_dcp(X, H, rmsnorm_X_eps, rmsnorm_X_gamma, W_qkv, cos, sin,
                                                 active_blocks_table, K_cache, V_cache, attention_mask, sink,
                                                 softmax_scale, k_scale, v_scale, pos, start, dcp, dcp_mesh, dcp_mode)
    elif kvdp > 1:
        heads_out, k_new, v_new = _attention_kvdp(X, H, rmsnorm_X_eps, rmsnorm_X_gamma, W_qkv, cos, sin,
                                                  active_blocks_table, K_cache, V_cache, attention_mask, sink,
                                                  softmax_scale, k_scale, v_scale, pos, rank, kvdp, kvdp_mesh)
    elif pos is not None:
        heads_out, k_new, v_new = _attention_posid(X, H, rmsnorm_X_eps, rmsnorm_X_gamma, W_qkv, cos, sin,
                                                   active_blocks_table, K_cache, V_cache, attention_mask, sink,
                                                   softmax_scale, k_scale, v_scale, pos, start)
    else:
        heads_out, k_new, v_new = _attention(X, H, rmsnorm_X_eps, rmsnorm_X_gamma, W_qkv, cos, sin,
                                             active_blocks_table, K_cache, V_cache, attention_mask, sink,
                                             softmax_scale, k_scale, v_scale)
    # the two programs each wrote a share of the rows, and each reads them all below (nkilib's transformer_tkg fences
    # its attention output the same way); the new K / V are fenced and written after the output projection, which
    # does not read them, so their DMAs overlap its matmuls and the all-reduce after it
    if n_prg > 1:
        nisa.core_barrier(data=heads_out, cores=(0, 1))

    # ---- the output projection of this program's half of H ----
    # heads_out [B, heads, d, S]: head h's channel j of token s of row b at ((b x heads + h) x d + j) x S + s; lhsT's
    # columns are the tokens, row-major (column b x S + s)
    lhsT = nl.ndarray((P, heads, P), dtype=heads_out.dtype, buffer=nl.sbuf)
    for h in range(heads):
        if S == 1:
            nisa.dma_copy(dst=lhsT[0:v_head_dim, h, 0:B],
                          src=heads_out.ap(pattern=[[1, v_head_dim], [heads * d, B]], offset=h * d))
        else:
            for b in range(B):
                nisa.dma_copy(dst=lhsT[0:v_head_dim, h, b * S:(b + 1) * S],
                              src=heads_out.ap(pattern=[[S, v_head_dim], [1, S]], offset=(b * heads + h) * d * S))
    o_sb = nl.ndarray((P, HH), dtype=X.dtype, buffer=nl.sbuf)
    for c in range(HH // COLS):
        ps = nl.ndarray((P, COLS), dtype=nl.float32, buffer=nl.psum)
        for h in range(heads):
            nisa.nc_matmul(dst=ps[0:R, 0:COLS], stationary=lhsT[0:v_head_dim, h, 0:R],
                           moving=w_all[0:v_head_dim, h, c * COLS:(c + 1) * COLS])
        nisa.tensor_copy(dst=o_sb[0:R, c * COLS:(c + 1) * COLS], src=ps[0:R, 0:COLS])

    if n_prg > 1:
        nisa.core_barrier(data=k_new, cores=(0, 1))
        nisa.core_barrier(data=v_new, cores=(0, 1))
    # ---- the new tokens' K / V into their slots: program 0 writes V, program 1 K (both on one program at LNC1) ----
    # K [d, B, S]: token r = b x S + s, channel j at j x R + r; V [B, 1, S, d]: token r at r x d
    idx = nl.ndarray((P, 1), dtype=slots.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=idx[0:R, 0:1], src=slots.ap(pattern=[[1, R], [1, 1]], offset=0))
    k_flat = K_cache.reshape((pages * page, d))
    v_flat = V_cache.reshape((pages * page, d))
    # every buffer before the branches on the program, so it sits at one address in both programs' traces
    v_rows = nl.ndarray((P, d), dtype=V_cache.dtype, buffer=nl.sbuf)
    k_rows = nl.ndarray((P, d), dtype=K_cache.dtype, buffer=nl.sbuf)
    if n_prg == 1 or pid == 0:
        nisa.dma_copy(dst=v_rows[0:R, 0:d], src=v_new.ap(pattern=[[d, R], [1, d]], offset=0))
        nisa.dma_copy(dst=v_flat.ap(pattern=[[d, R], [1, d]], offset=0, vector_offset=idx[0:R, 0:1], indirect_dim=0),
                      src=v_rows[0:R, 0:d], oob_mode=oob_mode.skip)
    if n_prg == 1 or pid == 1:
        nisa.dma_copy(dst=k_rows[0:R, 0:d], src=k_new.ap(pattern=[[1, R], [R, d]], offset=0))
        nisa.dma_copy(dst=k_flat.ap(pattern=[[d, R], [1, d]], offset=0, vector_offset=idx[0:R, 0:1], indirect_dim=0),
                      src=k_rows[0:R, 0:d], oob_mode=oob_mode.skip)

    if out_in_sbuf:
        return o_sb, K_cache, V_cache
    out = nl.ndarray((R, H), dtype=X.dtype, buffer=nl.shared_hbm, name="attn_half_out")
    nisa.dma_copy(dst=out.ap(pattern=[[H, R], [1, HH]], offset=h0), src=o_sb[0:R, 0:HH])
    return out, K_cache, V_cache
