# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GLM-5.2 DSA indexer: key projection, RoPE, cache write and the head-reduced relevance score for one rank's key range.

    scores[t, e] = sum_h w[t, h] * relu(<q[t, h, :], k[e, :]> / sqrt(head_dim))

The relu sits between the head scores and the head sum, so the head weight cannot fold into ``q``, and
``weights_proj`` has either sign, so a masked slot is clamped rather than zeroed. The scored range stops before the
step's own tokens, which the attention core adds itself (see ``indexer_kv_score``).
"""

import os

import nki
import nki.isa as nisa
import nki.language as nl

from ....utils.kernel_assert import kernel_assert
from ....utils.kernel_helpers import div_ceil
from ....utils.scratch_hbm import layer_scratch_hbm
from ....utils.tiled_range import TiledRange
from ..mla.mla_proj_primitives import linear, load_rope_tables, rope_inplace, transpose_into_k_tiles
from .indexer_constants import (
    IDX_KEY_FP8_MAX,
    P_MAX,
    SCORE_PSUM_DTYPE,
    SNAKE_PARTITIONS,
    IndexerScoreShape,
    get_indexer_score_shape,
    indexer_cache_is_blocked,
    indexer_cache_keys,
    snake_score_layout,
)

SCORE_HWDGE = os.environ.get("GLM52K_GLM52_IDX_SCORE_HWDGE", "0") == "1"
"""Transpose each score block back to token-major on the hardware DGE, ``HWDGE_KEYS`` keys at a time (default off).

The static ``dma_transpose`` has about one descriptor per key (5.5 M per NEFF at 1M with cp 4); the hardware DGE
builds them at run time and moves the same bytes, so the scores are bit-identical. Only the HBM-score path
transposes; the snake layout is unchanged. Read from the environment because it sizes the whole NEFF, which the
deployment sets for the giga kernel and the chain together.
"""

HWDGE_KEYS = 16
"""Keys (source partitions) per hardware-DGE score transpose: the DGE's limit on ``src.shape[0]``."""


@nki.jit
def indexer_kv_score(
    q: nl.NkiTensor,
    hidden_states: nl.NkiTensor,
    wk: nl.NkiTensor,
    k_norm_weight: nl.NkiTensor,
    k_norm_bias: nl.NkiTensor,
    weights_proj: nl.NkiTensor,
    cos: nl.NkiTensor,
    sin: nl.NkiTensor,
    k_cache: nl.NkiTensor,
    new_slot_mapping: nl.NkiTensor,
    mask_add: nl.NkiTensor,
    num_heads: int,
    rope_dim: int,
    key_base: int,
    norm_eps: float = 1e-6,
    block_rows: nl.NkiTensor = None,
    snake_out: bool = False,
    prologue: tuple | None = None,
) -> tuple[nl.NkiTensor, nl.NkiTensor]:
    """Score this rank's key range against every indexer head, reduce the heads, and write the step's new keys.

    Args:
        q: [T_q, num_heads * head_dim] the complete indexer query (the head axis is reduced here, so it is asserted
            complete).
        hidden_states: [T_q, hidden] the step's normed rows, input to the key projection and the head weights.
        wk: [hidden, head_dim] the key projection: one key head shared by every query head.
        k_norm_weight, k_norm_bias: [1, head_dim] fp32 key LayerNorm.
        weights_proj: [hidden, num_heads] per-head mixing weights.
        cos, sin: [T_q, rope_dim // 2] fp32 interleaved RoPE tables.
        k_cache: [n_blocks, 1, block_size, head_dim] this rank's key cache, updated in place: bf16, or uint8 for fp8
            keys (``IDX_KEY_FP8_MAX``). Position-ordered (local row ``i`` is position ``key_base + i``) unless
            ``block_rows`` pages it.
        new_slot_mapping: [T_q] int32 physical cache row of each new key, negative for keys another rank owns.
        mask_add: the step's additive residency mask (``indexer_residency_mask``): 0 for a key below the step's base
            position, ``-MASK_BOUND`` otherwise.
        num_heads, rope_dim: indexer heads and the rotated width, the leading slice of each head.
        key_base: global position of local row 0.
        norm_eps: epsilon of the key LayerNorm.
        block_rows: [P_MAX, num_keys / P_MAX] uint32 physical row of every logical key (``indexer_block_rows``), or
            None for a position-ordered cache. The paged read is an indirect transpose, so prefer None when the cache
            is in position order.
        snake_out: return the scores as ``nisa.topk``'s snake tile in SBUF instead of HBM (``snake_score_layout``).
        prologue: :func:`indexer_key_prologue`'s result; the key side then already ran and only the cache write and
            the score remain.

    Returns:
        ``(index_scores, k_cache)``: [T_q, num_keys] bf16 scores in HBM (``nisa.topk`` ranks bf16 only) and the updated
        cache.

    The scored range stops at the step's base position. The attention core adds this step's own keys separately, so
    selecting them here would count them twice. Below the base the cut is one threshold on the key position, the same
    for every indexer layer of a step, which is why the mask is an input. The cache stays token-major and the DMA
    transposes each key tile on the way in, which measured faster than a feature-major cache.
    """

    shapes = get_indexer_score_shape(
        q, hidden_states, wk, weights_proj, k_cache, mask_add, num_heads, rope_dim, key_base
    )
    T, NH, HD = shapes.num_tokens, shapes.num_heads, shapes.head_dim
    n_keys = shapes.num_keys

    # The cache arrives BLOCKED, [n_blocks, 1, block_size, head_dim] (or legacy flat, used as is), as MLA's kv_cache and
    # the engine's paged convention use. Both the append and the score address individual ROWS, so flatten once here
    # and keep the rest per-row: the indirect DMA's offset is scaled by the stride of dim 0, so a row
    # offset needs a row-major-2D view. Same storage, so slot_mapping and block_rows are unchanged.
    cache_rows = k_cache.reshape((n_keys, HD)) if indexer_cache_is_blocked(k_cache) else k_cache
    # The score loop reads an fp8 cache as [n_keys, head_dim / 2] bf16 WORDS (IndexerScoreShape.key_fp8); the
    # write keeps the byte rows. Same storage either way.
    key_rows = cache_rows.view(nl.bfloat16) if shapes.key_fp8 else cache_rows

    if prologue is not None:
        head_w_sbuf, rope_rows, k_new_sbuf = prologue
        # The cache write stays here: the score loop's key loads are ordered after it, and issued early they would
        # queue ahead of the up-projection's weights on the Sync FIFO.
        _write_key_cache(cache_rows, k_new_sbuf, new_slot_mapping, shapes)
        q_t_sbuf = _load_query_feature_major(q, None, None, shapes, rope_rows=rope_rows)
    else:
        cos_sbuf, sin_sbuf = load_rope_tables(cos, sin, T)

        hidden_sbuf = nl.ndarray((P_MAX, shapes.hidden), dtype=shapes.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=hidden_sbuf[0:T, 0 : shapes.hidden], src=hidden_states[0:T, 0 : shapes.hidden])
        hidden_tiles_sbuf = transpose_into_k_tiles(hidden_sbuf[0:P_MAX, 0 : shapes.hidden], T)

        k_new_sbuf = _project_and_norm_key(hidden_tiles_sbuf, wk, k_norm_weight, k_norm_bias, norm_eps, shapes)
        rope_inplace(k_new_sbuf, cos_sbuf, sin_sbuf, 0, HD, 1, T)
        _write_key_cache(cache_rows, k_new_sbuf, new_slot_mapping, shapes)

        head_w_sbuf = _head_sum_weights(hidden_tiles_sbuf, weights_proj, shapes)
        q_t_sbuf = _load_query_feature_major(q, cos_sbuf, sin_sbuf, shapes)

    # One DMA for the whole row table, not one per chunk: it is [128, 256] uint32 = 128 KB of SBUF at
    # the production shape, layer-invariant, and the score loop reads a 4-column window of it per chunk.
    rows_sbuf = None
    if block_rows is not None:
        rows_sbuf = nl.ndarray((P_MAX, shapes.num_key_tiles), dtype=nl.uint32, buffer=nl.sbuf)
        nisa.dma_copy(dst=rows_sbuf, src=block_rows[0:P_MAX, 0 : shapes.num_key_tiles])

    # Under snake_score_layout the mask arrives in the snake loop's column order, so that loop runs for every
    # caller; one that did not ask for the snake gets the HBM scores from it.
    if snake_score_layout(n_keys, T):
        snake_sbuf = nl.ndarray((T * SNAKE_PARTITIONS, n_keys // SNAKE_PARTITIONS), dtype=nl.bfloat16, buffer=nl.sbuf)
        _score_all_keys_snake(snake_sbuf, key_rows, q_t_sbuf, head_w_sbuf, mask_add, shapes, rows_sbuf)
        if snake_out:
            return snake_sbuf, k_cache
        return indexer_scores_from_snake(snake_sbuf), k_cache
    kernel_assert(not snake_out, f"snake_out needs snake_score_layout({n_keys}, {T})")

    scores_out = nl.ndarray((T, n_keys), dtype=nl.bfloat16, buffer=layer_scratch_hbm())
    _score_all_keys(scores_out, key_rows, q_t_sbuf, head_w_sbuf, mask_add, shapes, rows_sbuf)

    # k_cache, not cache_rows: the caller's tensor is what the input/output alias names.
    return scores_out, k_cache


def indexer_key_prologue(
    hidden_tiles_sbuf: nl.NkiTensor,
    stationary_fn,
    wk: nl.NkiTensor,
    k_norm_weight: nl.NkiTensor,
    k_norm_bias: nl.NkiTensor,
    weights_proj: nl.NkiTensor,
    cos: nl.NkiTensor,
    sin: nl.NkiTensor,
    k_cache: nl.NkiTensor,
    num_heads: int,
    rope_dim: int,
    norm_eps: float = 1e-6,
) -> tuple:
    """
    The query-independent front of :func:`indexer_kv_score` (the attention layer's indexer front), for a caller that already
    holds hidden's ``linear`` K-tiles (the down-projection's): the key projection, LayerNorm and RoPE, the head-sum
    weights, and every token's RoPE row replicated over its head block -- one broadcast DMA per (token, table) from
    HBM instead of :func:`_broadcast_rope_rows`' transposes. The same values as the kernel computes, so
    ``indexer_kv_score(prologue=)`` is bit-identical; it still does the cache write. Traced ahead of the query, it
    fills the PE gap of the latent AllGather.

    Returns:
        tuple: ``(head_w_sbuf, rope_rows, k_new_sbuf)`` for ``indexer_kv_score(prologue=)``.
    """
    T, HD = cos.shape[0], wk.shape[1]
    shapes = IndexerScoreShape(
        num_tokens=T,
        num_new=T,
        num_heads=num_heads,
        head_dim=HD,
        rope_dim=rope_dim,
        hidden=wk.shape[0],
        num_keys=indexer_cache_keys(k_cache),
        key_base=0,
        dtype=hidden_tiles_sbuf.dtype,
        key_fp8=k_cache.dtype == nl.uint8,
    )
    cos_sbuf, sin_sbuf = load_rope_tables(cos, sin, T)
    k_new_sbuf = _project_and_norm_key(
        hidden_tiles_sbuf, wk, k_norm_weight, k_norm_bias, norm_eps, shapes, stationary_fn=stationary_fn
    )
    rope_inplace(k_new_sbuf, cos_sbuf, sin_sbuf, 0, HD, 1, T)
    head_w_sbuf = _head_sum_weights(hidden_tiles_sbuf, weights_proj, shapes, stationary_fn=stationary_fn)

    rope_half = cos.shape[1]
    rope_rows = []
    for group in range(shapes.num_token_groups):
        t0, gt = group * shapes.tokens_per_group, shapes.group_tokens(group)
        pair = []
        for table in (cos, sin):
            rows_sbuf = nl.ndarray((P_MAX, rope_half), dtype=nl.float32, buffer=nl.sbuf)
            for i in range(gt):
                nisa.dma_copy(
                    dst=rows_sbuf[i * num_heads : (i + 1) * num_heads, 0:rope_half],
                    src=table[t0 + i : t0 + i + 1, 0:rope_half].broadcast(0, num_heads),
                )
            pair.append(rows_sbuf)
        rope_rows.append((pair[0], pair[1]))
    return head_w_sbuf, rope_rows, k_new_sbuf


def indexer_scores_from_snake(snake_sbuf: nl.NkiTensor) -> nl.NkiTensor:
    """The ``[T_q, num_keys]`` HBM ``index_scores`` of a ``snake_out`` tile, for taps and non-snake consumers.

    Row ``t * 16 + q`` of the snake is exactly the ``q``-th sixteenth of row ``t``, so this is one DMA."""
    rows, n_cols = snake_sbuf.shape[0], snake_sbuf.shape[1]
    scores = nl.ndarray(
        (rows // SNAKE_PARTITIONS, n_cols * SNAKE_PARTITIONS), dtype=nl.bfloat16, buffer=layer_scratch_hbm()
    )
    nisa.dma_copy(dst=scores.reshape((rows, n_cols)), src=snake_sbuf[0:rows, 0:n_cols])
    return scores


def _project_and_norm_key(
    hidden_tiles_sbuf: nl.NkiTensor,
    wk: nl.NkiTensor,
    k_norm_weight: nl.NkiTensor,
    k_norm_bias: nl.NkiTensor,
    eps: float,
    shapes: IndexerScoreShape,
    stationary_fn=None,
) -> nl.NkiTensor:
    """
    Projects this step's hidden states to one key per token and applies the LayerNorm.

    Returns ``[P_MAX, head_dim]``. The norm is a TRUE LayerNorm -- it subtracts the mean and adds a
    bias -- so ``mla_proj_primitives.rmsnorm_inplace`` is not a substitute: that one never centres and
    has no bias term.
    """

    T, HD = shapes.num_tokens, shapes.head_dim

    k_sbuf = linear(hidden_tiles_sbuf, wk, T, stationary_fn=stationary_fn)

    gain_sbuf = nl.ndarray((P_MAX, HD), dtype=nl.float32, buffer=nl.sbuf)
    bias_sbuf = nl.ndarray((P_MAX, HD), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=gain_sbuf[0:T, 0:HD], src=k_norm_weight[0:1, 0:HD].broadcast(0, T))
    nisa.dma_copy(dst=bias_sbuf[0:T, 0:HD], src=k_norm_bias[0:1, 0:HD].broadcast(0, T))

    _layernorm_inplace(k_sbuf[0:P_MAX, 0:HD], gain_sbuf, bias_sbuf, T, eps)

    return k_sbuf


def _layernorm_inplace(
    x_sbuf: nl.NkiTensor,
    gain_sbuf: nl.NkiTensor,
    bias_sbuf: nl.NkiTensor,
    P: int,
    eps: float,
) -> None:
    """
    Applies ``x = (x - mean(x)) * rsqrt(var(x) + eps) * gain + bias`` in place, in fp32.

    Ported from the reference DSA indexer rather than derived: the variance is taken about the
    already-centred data, so it never forms ``E[x^2] - E[x]^2`` and cannot lose precision to
    cancellation when the projection output is far from zero-mean. The mean carries its own
    ``1 / head_dim``; the variance's rides the rsqrt's scale.
    """

    F = x_sbuf.shape[1]

    scratch = nl.ndarray((P_MAX, F), dtype=nl.float32, buffer=nl.sbuf)
    stat = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)

    nisa.activation_reduce(
        dst=scratch[0:P, 0:F],
        op=nl.copy,
        data=x_sbuf[0:P, 0:F],
        reduce_op=nl.add,
        reduce_res=stat[0:P, 0:1],
    )
    nisa.tensor_scalar(dst=stat[0:P, 0:1], data=stat[0:P, 0:1], op0=nl.multiply, operand0=1.0 / F)
    nisa.tensor_scalar(dst=x_sbuf[0:P, 0:F], data=x_sbuf[0:P, 0:F], op0=nl.subtract, operand0=stat[0:P, 0:1])

    nisa.activation_reduce(
        dst=scratch[0:P, 0:F],
        op=nl.square,
        data=x_sbuf[0:P, 0:F],
        reduce_op=nl.add,
        reduce_res=stat[0:P, 0:1],
    )
    nisa.activation(stat[0:P, 0:1], op=nl.rsqrt, data=stat[0:P, 0:1], scale=1.0 / F, bias=eps)

    nisa.scalar_tensor_tensor(
        dst=x_sbuf[0:P, 0:F],
        data=x_sbuf[0:P, 0:F],
        op0=nl.multiply,
        operand0=stat[0:P, 0:1],
        op1=nl.multiply,
        operand1=gain_sbuf[0:P, 0:F],
    )
    nisa.tensor_tensor(dst=x_sbuf[0:P, 0:F], data1=x_sbuf[0:P, 0:F], data2=bias_sbuf[0:P, 0:F], op=nl.add)


def _write_key_cache(
    k_cache: nl.NkiTensor,
    k_new_sbuf: nl.NkiTensor,
    new_slot_mapping: nl.NkiTensor,
    shapes: IndexerScoreShape,
) -> None:
    """
    Scatters this step's keys into the cache, skipping any token this rank does not own.

    The write is pure bookkeeping for LATER steps: this step's own keys are excluded from the score
    (see the kernel's ``base_position`` note), so there is no write-then-read ordering hazard here at
    all -- unlike the MLA cache, which must reason about it.

    An fp8 cache (``shapes.key_fp8``) takes ``clamp(key, +-IDX_KEY_FP8_MAX)`` cast to float8_e4m3fn, the bytes the
    host's prefill writes: the clamp and the cast are one ``tensor_scalar`` into a byte tile, so the scatter itself
    is unchanged.
    """

    T, HD = shapes.num_tokens, shapes.head_dim

    slot_sbuf = nl.ndarray((T, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=slot_sbuf[0:T, 0:1], src=new_slot_mapping[0:T].reshape((T, 1)))

    src_sbuf = k_new_sbuf
    if shapes.key_fp8:
        src_sbuf = nl.ndarray((P_MAX, HD), dtype=nl.uint8, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=src_sbuf.view(nl.float8_e4m3fn)[0:T, 0:HD],
            data=k_new_sbuf[0:T, 0:HD],
            op0=nl.minimum,
            operand0=IDX_KEY_FP8_MAX,
            op1=nl.maximum,
            operand1=-IDX_KEY_FP8_MAX,
        )

    nisa.dma_copy(
        dst=k_cache.ap(
            pattern=[[HD, T], [1, HD]],
            offset=0,
            vector_offset=slot_sbuf.ap(pattern=[[1, T], [1, 1]], offset=0),
            indirect_dim=0,
        ),
        src=src_sbuf[0:T, 0:HD],
        oob_mode=nisa.oob_mode.skip,
    )


def _head_sum_weights(
    hidden_tiles_sbuf: nl.NkiTensor,
    weights_proj: nl.NkiTensor,
    shapes: IndexerScoreShape,
    stationary_fn=None,
) -> list[nl.NkiTensor]:
    """The head-sum stationaries: one ``[P_MAX, T_q]`` tile per token group, entry ``[i * num_heads + h, t]`` equal to
    ``w[t, h] * (num_heads * head_dim) ** -0.5`` when ``t`` is the group's ``i``-th token and zero elsewhere, so one
    matmul weights a token's heads and sums them. The PE is the only engine that reduces across partitions, and folding
    the weight into the stationary keeps the relu a plain Scalar activation. The scale is a power of two (1/64 here), so
    it is exact. The partition move goes through PE transposes: the equivalent small strided DMAs were far slower.
    """

    T, NH = shapes.num_tokens, shapes.num_heads
    dtype = shapes.dtype
    fold = float(NH * shapes.head_dim) ** -0.5

    w_sbuf = linear(hidden_tiles_sbuf, weights_proj, T, stationary_fn=stationary_fn)

    w_scaled_sbuf = nl.ndarray((P_MAX, NH), dtype=dtype, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=w_scaled_sbuf[0:T, 0:NH], data=w_sbuf[0:T, 0:NH], op0=nl.multiply, operand0=fold)

    # One transpose per token slot in a group, each landing [head, token] at that slot's partition
    # block, so stacked[i * num_heads + h, t] = w[t, h] for every slot i. The block start is
    # 32-aligned by tokens_per_group, which is what makes the offset slice legal.
    stacked_psum = nl.ndarray((P_MAX, T), dtype=dtype, buffer=nl.psum)
    for i in range(shapes.tokens_per_group):
        nisa.nc_transpose(stacked_psum[i * NH : (i + 1) * NH, 0:T], w_scaled_sbuf[0:T, 0:NH])

    # Keep only each token's OWN block. Without this mask a token's head sum would also pick up the
    # other tokens sharing its group, because every block holds every token's weights.
    weights_sbuf = []
    for group in range(shapes.num_token_groups):
        block = nl.ndarray((P_MAX, T), dtype=dtype, buffer=nl.sbuf)
        nisa.memset(dst=block[0:P_MAX, 0:T], value=0.0)
        for i in range(shapes.group_tokens(group)):
            token = group * shapes.tokens_per_group + i
            # Same partition range on both sides, so this is a lane-wise copy, not a partition move.
            nisa.tensor_copy(
                dst=block[i * NH : (i + 1) * NH, token : token + 1],
                src=stacked_psum[i * NH : (i + 1) * NH, token : token + 1],
            )
        weights_sbuf.append(block)

    return weights_sbuf


def _load_query_feature_major(
    q: nl.NkiTensor,
    cos_sbuf: nl.NkiTensor,
    sin_sbuf: nl.NkiTensor,
    shapes: IndexerScoreShape,
    rope_rows: list | None = None,
) -> nl.NkiTensor | tuple[nl.NkiTensor, nl.NkiTensor]:
    """Load the query one (token, head) pair per partition, rotate it, and transpose it to ``[head_dim, T_q * num_heads]``
    with the head innermost, so a token group's columns are one contiguous run (a matmul stationary takes one free
    dimension). One pair per partition spreads the rotation over all 128 lanes and needs one transpose per token group.
    RoPE runs here because it needs whole heads; the rotated slice leads each head. For an fp8 cache it returns the
    even and odd feature halves as a pair.
    """

    T, NH, HD = shapes.num_tokens, shapes.num_heads, shapes.head_dim
    dtype = shapes.dtype

    if shapes.key_fp8:
        q_halves = [nl.ndarray((P_MAX, T * NH), dtype=dtype, buffer=nl.sbuf) for _parity in range(2)]
    else:
        q_t_sbuf = nl.ndarray((P_MAX, T * NH), dtype=dtype, buffer=nl.sbuf)

    for group in range(shapes.num_token_groups):
        gp, gt = shapes.group_partitions(group), shapes.group_tokens(group)
        t0 = group * shapes.tokens_per_group

        # Partition i * num_heads + h holds token t0 + i, head h -- a contiguous run of gt * num_heads
        # rows of q's row-major (token, head, head_dim) layout, so it is one plain DMA.
        q_g_sbuf = nl.ndarray((P_MAX, HD), dtype=dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=q_g_sbuf[0:gp, 0:HD],
            src=q.ap(pattern=[[NH * HD, gt], [HD, NH], [1, HD]], offset=t0 * NH * HD),
        )

        if rope_rows is not None:
            cos_g, sin_g = rope_rows[group]
        else:
            cos_g, sin_g = _broadcast_rope_rows(cos_sbuf, sin_sbuf, t0, gt, shapes)
        rope_inplace(q_g_sbuf, cos_g, sin_g, 0, HD, 1, gp)

        if shapes.key_fp8:
            # Features 2j + parity of every (token, head) row: the halves the two fp8 score matmuls contract.
            half = HD // 2
            for parity, q_half in enumerate(q_halves):
                g_psum = nl.ndarray((P_MAX, P_MAX), dtype=dtype, buffer=nl.psum)
                nisa.nc_transpose(g_psum[0:half, 0:gp], q_g_sbuf.ap(pattern=[[HD, gp], [2, half]], offset=parity))
                nisa.tensor_copy(dst=q_half[0:half, t0 * NH : t0 * NH + gp], src=g_psum[0:half, 0:gp])
        else:
            # One transpose per group: [(token, head), head_dim] -> [head_dim, (token, head)], which is
            # already the returned layout's column order.
            g_psum = nl.ndarray((P_MAX, P_MAX), dtype=dtype, buffer=nl.psum)
            nisa.nc_transpose(g_psum[0:HD, 0:gp], q_g_sbuf[0:gp, 0:HD])
            nisa.tensor_copy(dst=q_t_sbuf[0:HD, t0 * NH : t0 * NH + gp], src=g_psum[0:HD, 0:gp])

    return tuple(q_halves) if shapes.key_fp8 else q_t_sbuf


def _broadcast_rope_rows(
    cos_sbuf: nl.NkiTensor,
    sin_sbuf: nl.NkiTensor,
    t0: int,
    n_tokens: int,
    shapes: IndexerScoreShape,
) -> tuple[nl.NkiTensor, nl.NkiTensor]:
    """Copy each token's RoPE row onto that token's block of head partitions: transpose the table so tokens are on the
    free axis, broadcast the token's column, transpose back. A partition-strided DMA would be much slower, and slicing the
    token's row directly is not allowed (a partition range must start on a multiple of 32).
    """

    NH = shapes.num_heads
    T = shapes.num_tokens
    rope_half = cos_sbuf.shape[1]
    out = []

    for table in (cos_sbuf, sin_sbuf):
        # [rope_half, T_q]: tokens onto the free axis, so selecting one is a free-axis slice.
        t_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(t_psum[0:rope_half, 0:T], table[0:T, 0:rope_half])
        t_sbuf = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=t_sbuf[0:rope_half, 0:T], src=t_psum[0:rope_half, 0:T])

        dst = nl.ndarray((P_MAX, rope_half), dtype=nl.float32, buffer=nl.sbuf)
        for i in range(n_tokens):
            token = t0 + i
            wide = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=wide[0:rope_half, 0:NH], src=t_sbuf[0:rope_half, token : token + 1].broadcast(1, NH))
            psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_transpose(psum[0:NH, 0:rope_half], wide[0:rope_half, 0:NH])
            nisa.tensor_copy(dst=dst[i * NH : (i + 1) * NH, 0:rope_half], src=psum[0:NH, 0:rope_half])
        out.append(dst)

    return out[0], out[1]


def _score_all_keys(
    scores_out: nl.NkiTensor,
    k_cache: nl.NkiTensor,
    q_t_sbuf: nl.NkiTensor,
    head_w_sbuf: list[nl.NkiTensor],
    mask_add: nl.NkiTensor,
    shapes: IndexerScoreShape,
    rows_sbuf: nl.NkiTensor = None,
) -> None:
    """The plain score loop: score every local key against every (token, head), relu, and reduce the heads on the PE
    with the keys on the output partitions, so the mask add and the eviction run on 128 lanes once per block of
    ``512 / T`` key tiles instead of on ``T`` lanes per chunk. A transposing DMA per block returns the ``[T, keys]``
    layout the top-k reads. Scalar and Vector split each relu; Vector evicts. An fp8 cache scores in two half-depth
    matmuls (``_score_chunk``). ``k_cache`` is the row view of the cache.
    """

    T, n_keys = shapes.num_tokens, shapes.num_keys
    chunk = shapes.key_chunk
    blk_tiles, blk_keys = shapes.head_sum_block_tiles, shapes.head_sum_block_keys

    for blk in TiledRange(n_keys, blk_keys):
        n_tiles = div_ceil(blk.size, P_MAX)
        tile0 = blk.start_offset // P_MAX

        # The residency mask for this block, in the head sum's own [128 keys, (tile, token)] layout
        # (see indexer_residency_mask), so the eviction below adds it elementwise on 128 lanes.
        mask_sbuf = nl.ndarray((P_MAX, blk_tiles * T), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=mask_sbuf[0:P_MAX, 0 : n_tiles * T],
            src=mask_add[0:P_MAX, tile0 * T : (tile0 + n_tiles) * T],
        )

        # One bank of head sums for the whole block: column (tile, token), partition = key within tile.
        head_sum_psum = nl.ndarray((P_MAX, blk_tiles * T), dtype=nl.float32, buffer=nl.psum)

        for key_chunk in TiledRange(blk.size, chunk):
            _score_chunk(
                head_sum_psum,
                k_cache,
                q_t_sbuf,
                head_w_sbuf,
                rows_sbuf,
                blk.start_offset + key_chunk.start_offset,
                key_chunk.size,
                key_chunk.start_offset // P_MAX,
                shapes,
            )

        # Drop every key at or above this step's base position, and narrow to bfloat16, in ONE pass
        # over the block on all 128 lanes: the precomputed additive mask (0.0 keep / -MASK_BOUND drop)
        # was DMA'd in above in the same layout. Vector: it reads PSUM, which GpSimd cannot.
        scores_km_sbuf = nl.ndarray((P_MAX, blk_tiles * T), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=scores_km_sbuf[0:P_MAX, 0 : n_tiles * T],
            data1=head_sum_psum[0:P_MAX, 0 : n_tiles * T],
            data2=mask_sbuf[0:P_MAX, 0 : n_tiles * T],
            op=nl.add,
            engine=nisa.engine.vector,
        )

        # Back to the [T, keys] layout the top-k reads: [key, tile, 1, token] -> [token, tile, 1, key]
        # is dma_transpose's fixed 4-D permutation [3, 1, 2, 0], with the tile order UNCHANGED, so
        # column tile * 128 + key of the result is exactly global key blk.start + tile * 128 + key.
        scores_tm_sbuf = nl.ndarray((T, blk_keys), dtype=nl.bfloat16, buffer=nl.sbuf)
        if SCORE_HWDGE:
            # The same move in HWDGE_KEYS-key pieces: piece p takes keys [p * 16, (p + 1) * 16) of every tile, the
            # [16 keys, tile, 1, token] -> [token, tile, 1, 16 keys] slice of the permutation above.
            for piece in range(P_MAX // HWDGE_KEYS):
                k0 = piece * HWDGE_KEYS
                nisa.dma_transpose(
                    scores_tm_sbuf[0:T, 0 : n_tiles * P_MAX]
                    .reshape_dim(1, [n_tiles, P_MAX // HWDGE_KEYS, HWDGE_KEYS])
                    .slice(2, start=piece, end=piece + 1),
                    scores_km_sbuf[k0 : k0 + HWDGE_KEYS, 0 : n_tiles * T].reshape_dim(1, [n_tiles, 1, T]),
                    dge_mode=nisa.dge_mode.hwdge,
                )
        else:
            nisa.dma_transpose(
                scores_tm_sbuf[0:T, 0 : n_tiles * P_MAX].reshape_dim(1, [n_tiles, 1, P_MAX]),
                scores_km_sbuf[0:P_MAX, 0 : n_tiles * T].reshape_dim(1, [n_tiles, 1, T]),
            )
        nisa.dma_copy(
            dst=scores_out[0:T, blk.start_offset : blk.end_offset],
            src=scores_tm_sbuf[0:T, 0 : blk.size],
        )


def _score_chunk(
    head_sum_psum: nl.NkiTensor,
    k_cache: nl.NkiTensor,
    q_t_sbuf: nl.NkiTensor,
    head_w_sbuf: list[nl.NkiTensor],
    rows_sbuf: nl.NkiTensor,
    base: int,
    csz: int,
    chunk_tile0: int,
    shapes: IndexerScoreShape,
    tile_stride: int = 1,
) -> None:
    """
    Scores keys ``[base, base + csz)`` and accumulates the head sums of its ``tile``-th 128 keys into
    ``head_sum_psum`` columns ``(chunk_tile0 + tile * tile_stride) * T + [0, T)``. One iteration of
    ``_score_all_keys``.

    ``k_cache`` is the ROW view ``_score_all_keys`` describes; under ``shapes.key_fp8`` it is bf16 words of fp8 pairs
    and ``q_t_sbuf`` is ``_load_query_feature_major``'s ``(even, odd)`` pair.
    """

    T, NH, HD = shapes.num_tokens, shapes.num_heads, shapes.head_dim
    dtype = shapes.dtype
    chunk = shapes.key_chunk
    kw = shapes.key_words

    # The cache is token-major, so the DMA moves the contraction onto partitions on the way in: one dma_transpose in
    # place of a load, four nc_transpose and a PSUM eviction, which takes five steps out of the per-chunk chain. The 4-D
    # permutation [3, 1, 2, 0] keeps the key order. An fp8 cache moves as head_dim / 2 bf16 words per key (the transpose
    # needs a 2-byte dtype): word j of key e lands at [j, e], so the fp8 view has feature 2j + parity of key e at [j, 2e
    # + parity].
    k_t_sbuf = nl.ndarray((kw, chunk), dtype=nl.bfloat16 if shapes.key_fp8 else dtype, buffer=nl.sbuf)
    if rows_sbuf is None:
        nisa.dma_transpose(
            k_t_sbuf.slice(1, start=0, end=csz).reshape_dim(1, [1, 1, csz]),
            k_cache.slice(0, start=base, end=base + csz).reshape_dim(1, [1, 1, kw]),
        )
    else:
        # Paged: the same transpose with rows from the block table (the indirect form needs a 2-byte source and
        # src.shape[-1] <= 128, which a 128-wide bf16 or fp8-as-words cache meets). The table is in natural order, entry
        # [p, i] for key i * P_MAX + p, the order the score matmul and mask_add assume; a snake table would permute the
        # keys away from the mask and could still pass a loose tolerance.
        nisa.dma_transpose(
            dst=k_t_sbuf[0:kw, 0:csz],
            src=k_cache.ap(
                pattern=[[kw, csz], [1, kw]],
                offset=0,
                vector_offset=rows_sbuf[0:P_MAX, base // P_MAX : (base + csz) // P_MAX],
                indirect_dim=0,
            ),
            axes=(1, 0),
        )

    for group in range(shapes.num_token_groups):
        gp = shapes.group_partitions(group)
        t0 = group * shapes.tokens_per_group

        # dst[(token, head), key] = <q[token, head], k[key]>. The contraction is
        # head_dim = 128, filling the PE rows exactly; the stationary is this group's 128
        # (token, head) columns and the moving axis is the whole key chunk. NOT double-buffered
        # by chunk parity: that measured no faster and wrong on the full-shard accuracy check.
        raw = nl.ndarray((P_MAX, chunk), dtype=SCORE_PSUM_DTYPE, buffer=nl.psum)
        if shapes.key_fp8:
            # Two kw-deep halves accumulating in the bank: the even features (a stride-2 slice of the fp8 view
            # from byte 0) against the query's even half, then the odd ones. bf16 x e4m3 products are exact
            # in the fp32 accumulation, so only the summation order differs from one 128-deep pass.
            k_f8 = k_t_sbuf.view(nl.float8_e4m3fn)
            for parity, q_half in enumerate(q_t_sbuf):
                nisa.nc_matmul(
                    dst=raw[0:gp, 0:csz],
                    stationary=q_half[0:kw, t0 * NH : t0 * NH + gp],
                    moving=k_f8.slice(1, start=parity, end=parity + 2 * (csz - 1) + 1, step=2),
                    accumulate=parity == 1,
                )
        else:
            nisa.nc_matmul(
                dst=raw[0:gp, 0:csz],
                stationary=q_t_sbuf[0:HD, t0 * NH : t0 * NH + gp],
                moving=k_t_sbuf[0:HD, 0:csz],
            )

        # Plain relu, evicting PSUM fp32 to SBUF bf16, the only elementwise pass over the score (the scales and the head
        # weight ride the head-sum weights, see _head_sum_weights). Each group's relu is split by key halves, the first
        # on Scalar's activation and the second as max(x, 0) on Vector, so the stage every head-sum matmul waits on
        # is shorter. Splitting by group instead left the PE idle in each layer at 128 ranks.
        relu_sbuf = nl.ndarray((P_MAX, chunk), dtype=dtype, buffer=nl.sbuf)
        half = div_ceil(csz, 2 * P_MAX) * P_MAX  # tile-aligned so a head-sum tile is on one half
        nisa.activation(relu_sbuf[0:gp, 0:half], op=nl.relu, data=raw[0:gp, 0:half])
        if half < csz:
            nisa.tensor_scalar(
                dst=relu_sbuf[0:gp, half:csz],
                data=raw[0:gp, half:csz],
                op0=nl.maximum,
                operand0=0.0,
                engine=nisa.engine.vector,
            )

        # Weight and sum each token's own heads across partitions, KEYS OUT ON PARTITIONS:
        # out[key, t] = sum_(i,h) relu(score)[(i,h), key] * W[(i,h), t]. One matmul per 128-key
        # tile, into that tile's T columns of the block's bank; the groups accumulate, so a
        # key's row is complete after the last group.
        for tile in range(div_ceil(csz, P_MAX)):
            k0 = tile * P_MAX
            ksz = min(P_MAX, csz - k0)
            c0 = (chunk_tile0 + tile * tile_stride) * T
            nisa.nc_matmul(
                dst=head_sum_psum[0:ksz, c0 : c0 + T],
                stationary=relu_sbuf[0:gp, k0 : k0 + ksz],
                moving=head_w_sbuf[group][0:gp, 0:T],
                accumulate=group > 0,
            )


def _score_all_keys_snake(
    snake_sbuf: nl.NkiTensor,
    k_cache: nl.NkiTensor,
    q_t_sbuf: nl.NkiTensor,
    head_w_sbuf: list[nl.NkiTensor],
    mask_add: nl.NkiTensor,
    shapes: IndexerScoreShape,
    rows_sbuf: nl.NkiTensor = None,
) -> None:
    """
    ``_score_all_keys``, landing the scores straight in ``nisa.topk``'s snake ``[T * 16, n_keys / 16]``.

    Block ``b`` scores chunk ``b`` of every snake row ``q``, i.e. keys ``q * n_cols + b * chunk + [0, chunk)``,
    so its head-sum bank holds columns ``(j, q, t)`` (tile ``j`` of the chunk).
    The eviction writes them as ``(j, t, q)``, which makes each 128-key column ``j`` a contiguous
    ``[128 keys, (t, q)]`` tile whose PE transpose IS snake rows ``t * 16 + q``, columns
    ``b * chunk + j * 128 + [0, 128)``. Per key the arithmetic is unchanged; only the order the chunks are
    visited in differs. A block's transposes are traced after the next block's first chunk, so the PE never
    waits on the eviction.
    """

    T, Q = shapes.num_tokens, SNAKE_PARTITIONS
    chunk = shapes.key_chunk
    tpc = chunk // P_MAX
    n_cols = shapes.num_keys // Q
    blk_cols = Q * tpc * T

    pending_km = None
    pending_b = 0
    for b in range(n_cols // chunk):
        mask_sbuf = nl.ndarray((P_MAX, blk_cols), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=mask_sbuf[0:P_MAX, 0:blk_cols], src=mask_add[0:P_MAX, b * blk_cols : (b + 1) * blk_cols])

        head_sum_psum = nl.ndarray((P_MAX, blk_cols), dtype=nl.float32, buffer=nl.psum)
        for q in range(Q):
            # The bank is laid out (j, q, t): tile j of chunk q at column (j * Q + q) * T.
            _score_chunk(
                head_sum_psum,
                k_cache,
                q_t_sbuf,
                head_w_sbuf,
                rows_sbuf,
                q * n_cols + b * chunk,
                chunk,
                q,
                shapes,
                Q,
            )
            if q == 0 and pending_km is not None:
                _transpose_into_snake(snake_sbuf, pending_km, pending_b, shapes)
                pending_km = None

        scores_km = nl.ndarray((P_MAX, blk_cols), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=scores_km.reshape_dim(1, [tpc, T, Q]),
            data1=head_sum_psum.reshape_dim(1, [tpc, Q, T]).permute((0, 1, 3, 2)),
            data2=mask_sbuf.reshape_dim(1, [tpc, T, Q]),
            op=nl.add,
            engine=nisa.engine.vector,
        )
        pending_km = scores_km
        pending_b = b

    _transpose_into_snake(snake_sbuf, pending_km, pending_b, shapes)


def _transpose_into_snake(
    snake_sbuf: nl.NkiTensor,
    scores_km: nl.NkiTensor,
    b: int,
    shapes: IndexerScoreShape,
) -> None:
    """Block ``b``'s ``(j, t, q)``-ordered scores -> snake columns ``[b * chunk, (b + 1) * chunk)``: one PE
    transpose per 128 keys into one PSUM bank, one Vector copy out. Bit-exact: both move bf16 unchanged."""

    TQ = shapes.num_tokens * SNAKE_PARTITIONS
    chunk = shapes.key_chunk
    tp_psum = nl.ndarray((P_MAX, chunk), dtype=nl.bfloat16, buffer=nl.psum)
    for j in range(chunk // P_MAX):
        nisa.nc_transpose(
            dst=tp_psum[0:TQ, j * P_MAX : (j + 1) * P_MAX],
            data=scores_km[0:P_MAX, j * TQ : (j + 1) * TQ],
            engine=nisa.engine.tensor,
        )
    nisa.tensor_copy(
        dst=snake_sbuf[0:TQ, b * chunk : (b + 1) * chunk], src=tp_psum[0:TQ, 0:chunk], engine=nisa.engine.vector
    )
