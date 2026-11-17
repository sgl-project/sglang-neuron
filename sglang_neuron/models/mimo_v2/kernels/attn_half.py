# SPDX-License-Identifier: Apache-2.0
"""MiMo-V2's decode attention half in one NKI kernel: from the residual stream to this rank's partial of the output
projection, with the new token's K/V written into its pages, and no XLA op in between.

nkilib's ``attention_block_tkg`` runs the input RMSNorm, the QKV projection, RoPE and the attention (with the layer's
sinks) for heads of config.KV_HEAD_DIM channels; above 128 channels it takes neither the output projection nor the
cache write (``W_out=None``, ``update_cache=False``), so this kernel does both after it:

- the K/V write (after the attention, whose mask never reads the new token's slot from the cache, so the write cannot
  change what it attends to): the new token's K (``[d, B, 1]``, the kernel's layout for one KV head) is read
  token-major through a transposing access pattern, its V (``[B, 1, 1, d]``) as it is, and both rows go to each
  token's slot by an indirect DMA (with one KV head per rank a slot is its row of the flattened ``[pages x page, d]``
  cache); the two programs take V and K. A padding row's slot is 0, SGLang's reserved page;
- the output projection: the first ``v_head_dim`` channels of each head (the rest are the padding's zeros) against
  ``W_o`` ``[heads x v_head_dim, H]``, the two programs taking one half of H each.

The kernel returns the partial ``[B, H]`` and the two caches, so the caller's graph takes the caches as written in
place (an NKI kernel that writes an input and does not return it reports no alias, and the write can be dropped)."""

import nki
import nki.isa as nisa
import nki.language as nl
from nki.isa import oob_mode
from nkilib.core.utils.common_types import QuantizationType
from nkilib.experimental.transformer.attention_block_tkg import attention_block_tkg

P = 128
COLS = 512


def _attention(X, X_hidden_dim_actual, rmsnorm_X_eps, rmsnorm_X_gamma, W_qkv, cos, sin, active_blocks_table, K_cache,
               V_cache, attention_mask, sink, softmax_scale):
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
def attn_half(X, rmsnorm_X_gamma, W_qkv, cos, sin, active_blocks_table, K_cache, V_cache, attention_mask, sink,
              slots, W_o, rmsnorm_X_eps=1e-6, softmax_scale=1.0, v_head_dim=128):
    """X [B, 1, H] bf16; W_qkv [H, (heads + 2) x d]; cos / sin [d / 2, B, 1]; the caches [pages, 1, page, d]; the mask
    in the kernel's layout; sink [heads, 1] fp32 or None; slots [B, 1] int32; W_o [heads x v_head_dim, H] bf16.
    Returns (partial [B, H] bf16, K_cache, V_cache)."""
    B, _, H = X.shape
    pages, _, page, d = K_cache.shape
    heads = W_o.shape[0] // v_head_dim
    n_prg = nl.num_programs(axes=0)
    pid = nl.program_id(axis=0)
    HH = H // n_prg
    h0 = pid * HH

    heads_out, k_new, v_new = _attention(X, H, rmsnorm_X_eps, rmsnorm_X_gamma, W_qkv, cos, sin, active_blocks_table,
                                         K_cache, V_cache, attention_mask, sink, softmax_scale)
    # the two programs each wrote a share of the rows, and each reads them all below (nkilib's transformer_tkg fences
    # its attention output the same way)
    if n_prg > 1:
        nisa.core_barrier(data=heads_out, cores=(0, 1))
        nisa.core_barrier(data=k_new, cores=(0, 1))
        nisa.core_barrier(data=v_new, cores=(0, 1))

    # ---- the new token's K / V into its slot: program 0 writes V, program 1 K (both on one program at LNC1) ----
    idx = nl.ndarray((P, 1), dtype=slots.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=idx[0:B, 0:1], src=slots.ap(pattern=[[1, B], [1, 1]], offset=0))
    k_flat = K_cache.reshape((pages * page, d))
    v_flat = V_cache.reshape((pages * page, d))
    # every buffer before the branches on the program, so it sits at one address in both programs' traces
    v_rows = nl.ndarray((P, d), dtype=V_cache.dtype, buffer=nl.sbuf)
    k_rows = nl.ndarray((P, d), dtype=K_cache.dtype, buffer=nl.sbuf)
    if n_prg == 1 or pid == 0:
        nisa.dma_copy(dst=v_rows[0:B, 0:d], src=v_new.ap(pattern=[[d, B], [1, d]], offset=0))
        nisa.dma_copy(dst=v_flat.ap(pattern=[[d, B], [1, d]], offset=0, vector_offset=idx[0:B, 0:1], indirect_dim=0),
                      src=v_rows[0:B, 0:d], oob_mode=oob_mode.skip)
    if n_prg == 1 or pid == 1:
        # K [d, B, 1]: element (b, j) at j * B + b
        nisa.dma_copy(dst=k_rows[0:B, 0:d], src=k_new.ap(pattern=[[1, B], [B, d]], offset=0))
        nisa.dma_copy(dst=k_flat.ap(pattern=[[d, B], [1, d]], offset=0, vector_offset=idx[0:B, 0:1], indirect_dim=0),
                      src=k_rows[0:B, 0:d], oob_mode=oob_mode.skip)

    # ---- the output projection of this program's half of H ----
    # heads_out [B, heads, d, 1]: head h's channel j of row b at (b * heads + h) * d + j
    lhsT = nl.ndarray((P, heads, P), dtype=heads_out.dtype, buffer=nl.sbuf)
    for h in range(heads):
        nisa.dma_copy(dst=lhsT[0:v_head_dim, h, 0:B],
                      src=heads_out.ap(pattern=[[1, v_head_dim], [heads * d, B]], offset=h * d))
    out = nl.ndarray((B, H), dtype=X.dtype, buffer=nl.shared_hbm, name="attn_half_out")
    o_sb = nl.ndarray((P, HH), dtype=X.dtype, buffer=nl.sbuf)
    for c in range(HH // COLS):
        ps = nl.ndarray((P, COLS), dtype=nl.float32, buffer=nl.psum)
        for h in range(heads):
            w = nl.ndarray((P, COLS), dtype=W_o.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=w[0:v_head_dim, 0:COLS],
                          src=W_o.ap(pattern=[[H, v_head_dim], [1, COLS]], offset=h * v_head_dim * H + h0 + c * COLS))
            nisa.nc_matmul(dst=ps[0:B, 0:COLS], stationary=lhsT[0:v_head_dim, h, 0:B], moving=w[0:v_head_dim, 0:COLS])
        nisa.tensor_copy(dst=o_sb[0:B, c * COLS:(c + 1) * COLS], src=ps[0:B, 0:COLS])
    nisa.dma_copy(dst=out.ap(pattern=[[H, B], [1, HH]], offset=h0), src=o_sb[0:B, 0:HH])
    return out, K_cache, V_cache
