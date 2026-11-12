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

from ...utils.kernel_assert import kernel_assert
from ...utils.kernel_helpers import div_ceil
from ...utils.scratch_hbm import layer_scratch_hbm
from ...utils.tiled_range import TiledRange
from ...utils.trace_names import unique_name
from ..mla.mla_proj_primitives import linear, load_rope_tables, rope_inplace, transpose_into_k_tiles
from .indexer_constants import (
    DYNAMIC_RELU_POOL,
    HEADSUM_COL_TILE,
    HEADSUM_PACKED,
    HEADSUM_SLICE,
    IDX_DYNAMIC_LOOP,
    IDX_KEY_FP8_MAX,
    IDX_CHUNK_PEEL,
    IDX_KEY_HWDGE,
    IDX_KEY_PREFETCH,
    key_block_uses_hwdge,
    IDX_HEADSUM_LAG,
    IDX_RELU_BY_GROUP,
    IDX_SCORE_KEYPAIR,
    IDX_SCORE_ROWTILE,
    IDX_SELECT_2Q,
    MASK_BOUND,
    P_MAX,
    SCORE_PSUM_DTYPE,
    SNAKE_PARTITIONS,
    IndexerScoreShape,
    get_indexer_score_shape,
    headsum_packed_geometry,
    indexer_cache_is_blocked,
    indexer_cache_keys,
    snake_score_layout,
    MASK_BOUND,
)

SCORE_HWDGE = os.environ.get("GLM52K_GLM52_IDX_SCORE_HWDGE", "0") == "1"
"""
Opt-in (default OFF): transpose each score block back to token-major in ``HWDGE_KEYS``-key pieces on the hardware DGE.

The default per-block ``dma_transpose`` has static DMA descriptors, about one per key: 21 x 262144 = 5.5 M per NEFF at
1M with cp 4 (~84 MiB of model code at 16 B each). The hardware DGE builds its descriptors at run time, and moves the
same bytes to the same places, so the scores are bit-identical. It takes a source of at most 16 partitions, hence 8
pieces per block where the static form issues one. Only the HBM-score path has the transpose; the snake layout
(``snake_score_layout``) moves its scores through the PE and is unchanged. Read at import, like the other knobs.
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
    store_keys: int = 0,
    positions: nl.NkiTensor = None,
    rank_offset: nl.NkiTensor = None,
    shard_degree: int = 1,
    block_size: int = 0,
    chunk_select=None,
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
        block_rows: [P_MAX, num_keys / P_MAX] uint32 physical row of every logical key (``indexer_block_rows``), or
            None for a position-ordered cache. The paged read is an indirect transpose, so prefer None when the cache
            is in position order.
        snake_out: return the scores as ``nisa.topk``'s snake tile in SBUF instead of HBM (``snake_score_layout``).
        prologue: :func:`indexer_key_prologue`'s result; the key side then already ran and only the cache write and
            the score remain.
        store_keys, chunk_select, positions, rank_offset, shard_degree, block_size: the packed head sum's chunk loop
            (the MTP draft at 1M). ``chunk_select(chunk_scores, slot, label_p)`` is traced inside the loop over the
            live top-k chunks of ``store_keys`` keys; ``positions`` and the deal arguments give the live length.
        num_heads, rope_dim: indexer heads and the rotated width, the leading slice of each head.
        key_base: global position of local row 0.
        norm_eps: epsilon of the key LayerNorm.

    Returns:
        ``(index_scores, k_cache)``: [T_q, num_keys] bf16 scores in HBM (``nisa.topk`` ranks bf16 only; the chunk loop
        hands its scores to ``chunk_select`` and leaves this unwritten), and the updated cache.

    The scored range stops at the step's base position. The attention core adds this step's own keys separately, so
    selecting them here would count them twice. Below the base the cut is one threshold on the key position, the
    same for every indexer layer of a step, which is why the mask is an input. The cache stays token-major and the
    DMA transposes each key tile on the way in, which measured faster than a feature-major cache.
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
    if HEADSUM_PACKED:
        # The packed head sum serves one form: the chunk loop over the live top-k chunks of the position-ordered read
        # (the MTP draft's indexer at 1M), each chunk's scores handed to ``chunk_select``.
        kernel_assert(
            IDX_DYNAMIC_LOOP and positions is not None and rows_sbuf is None and chunk_select is not None,
            "the packed head sum scores only through the chunk loop: GLM52K_IDX_DYNAMIC_LOOP, positions, a "
            "position-ordered cache and chunk_select",
        )
        _DynamicLoop(key_rows, shapes).score_chunks(
            q_t_sbuf,
            head_w_sbuf,
            mask_add,
            positions,
            rank_offset,
            shard_degree,
            block_size,
            store_keys,
            chunk_select,
        )
        return scores_out, k_cache  # unwritten: the chunk loop hands its scores to chunk_select
    kernel_assert(chunk_select is None and not store_keys, "store_keys / chunk_select need the packed head sum")
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
                if IDX_SCORE_ROWTILE:
                    # The half twice along the free axis, so the transpose lands it on partitions [0, half) AND
                    # [half, 2 * half): the stationary of both PE row tiles (see _score_relu).
                    dup = nl.ndarray((P_MAX, 2 * half), dtype=dtype, buffer=nl.sbuf)
                    for r in range(2):
                        nisa.tensor_copy(
                            dst=dup[0:gp, r * half : (r + 1) * half],
                            src=q_g_sbuf.ap(pattern=[[HD, gp], [2, half]], offset=parity),
                        )
                    nisa.nc_transpose(g_psum[0 : 2 * half, 0:gp], dup[0:gp, 0 : 2 * half])
                    nisa.tensor_copy(dst=q_half[0 : 2 * half, t0 * NH : t0 * NH + gp], src=g_psum[0 : 2 * half, 0:gp])
                    continue
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

    T = shapes.num_tokens
    chunk = shapes.key_chunk

    k_t_sbuf = nl.ndarray((shapes.key_words, chunk), dtype=_key_tile_dtype(shapes), buffer=nl.sbuf)
    _load_key_chunk(k_t_sbuf, k_cache, rows_sbuf, base, csz, shapes)

    for group in range(shapes.num_token_groups):
        gp = shapes.group_partitions(group)
        raw = nl.ndarray((P_MAX, chunk), dtype=SCORE_PSUM_DTYPE, buffer=nl.psum)
        relu_sbuf = _score_relu(raw, None, k_t_sbuf, q_t_sbuf, group, csz, shapes)

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


def _key_tile_dtype(shapes: IndexerScoreShape):
    """dtype of a transposed key tile: bf16 words of fp8 pairs, else the kernel dtype."""
    return nl.bfloat16 if shapes.key_fp8 else shapes.dtype


def _load_key_chunk(
    k_t_sbuf: nl.NkiTensor,
    k_cache: nl.NkiTensor,
    rows_sbuf: nl.NkiTensor,
    base: int,
    csz: int,
    shapes: IndexerScoreShape,
    name: str = None,
) -> None:
    """Keys ``[base, base + csz)`` into ``k_t_sbuf`` ``[key_words, csz]``, feature-major; through ``rows_sbuf``
    (entry ``[p, tile]`` the row of key ``tile * P_MAX + p``) when given."""

    kw = shapes.key_words

    # The cache is token-major, so the DMA moves the contraction onto partitions on the way in: one dma_transpose in
    # place of a load, four nc_transpose and a PSUM eviction, which takes five steps out of the per-chunk chain. The 4-D
    # permutation [3, 1, 2, 0] keeps the key order. An fp8 cache moves as head_dim / 2 bf16 words per key (the transpose
    # needs a 2-byte dtype): word j of key e lands at [j, e], so the fp8 view has feature 2j + parity of key e at [j, 2e
    # + parity].
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
            name=name,
        )


_HWDGE_ROWS = 16
"""Source rows one Hardware-DGE transpose moves onto its last destination axis (``src.shape[0] <= 16``)."""


def _load_key_chunk_hwdge(
    k_t_sbuf: nl.NkiTensor,
    k_rows: nl.NkiTensor,
    key_reg: nl.NkiTensor,
    base: int,
    csz: int,
    shapes: IndexerScoreShape,
    name: str = None,
    words: int = 0,
) -> None:
    """``IDX_KEY_HWDGE``: keys ``[key_reg + base, key_reg + base + csz)`` into ``k_t_sbuf`` ``[key_words, csz]``,
    feature-major, in ONE Hardware-DGE transpose at the runtime row ``key_reg`` ([1, 1] int32 SBUF).

    The 4-D transpose permutes ``[A, B, C, D] -> [D, B, C, A]``. The source is a strided view of the rows with
    ``A = 16`` consecutive keys (stride ``kw``), ``B = csz / 16`` groups of them (stride ``16 * kw``), ``C = 1`` and
    ``D = kw`` words, so destination ``[d, b, 0, a]`` holds word ``d`` of key ``16 b + a``: the identity key order on
    the free axis, as the score matmul and the mask assume. The runtime offset rides the key axis (stride ``kw``).
    """
    # words: the stage row width when it is not key_words (IDX_SCORE_KEYPAIR: a key pair a row, 2 * key_words).
    kw = words or shapes.key_words
    A = _HWDGE_ROWS
    kernel_assert(csz % A == 0, f"key chunk {csz} is not a multiple of the {A}-row HWDGE transpose")
    B = csz // A
    pattern = [[kw, A], [A * kw, B], [A * kw, 1], [1, kw]]
    if key_reg is None:
        src = k_rows.ap(pattern=pattern, offset=base * kw)
    else:
        src = k_rows.ap(pattern=pattern, offset=base * kw, scalar_offset=key_reg, indirect_dim=0)
    return nisa.dma_transpose(
        dst=k_t_sbuf[0:kw, 0:csz].reshape_dim(1, [B, 1, A]), src=src, dge_mode=nisa.dge_mode.hwdge, name=name
    )


def _stage_key_block(stage: nl.NkiTensor, k_rows: nl.NkiTensor, key_reg: nl.NkiTensor, blk_keys: int, shapes,
                     name: str = None) -> list:
    """``IDX_KEY_HWDGE``: copy keys ``[key_reg, key_reg + blk_keys)`` of the cache rows into ``stage`` ([blk_keys,
    key_words] bf16 @ HBM) with one Hardware-DGE copy at the runtime row.

    The HWDGE transpose needs a 2-byte SOURCE MEMREF: a bf16 view or ``.ap(dtype=)`` of the uint8 (fp8) cache is
    refused ("requires 2-byte dtype, got 1-byte type"). A copy has no such rule, and the stage is a real bf16 tensor,
    so ``_load_key_chunk_hwdge`` then transposes from it at static offsets."""
    kw = shapes.key_words
    return [nisa.dma_copy(
        dst=stage[0:blk_keys, 0:kw],
        src=k_rows.ap(pattern=[[kw, blk_keys], [1, kw]], offset=0, scalar_offset=key_reg, indirect_dim=0,
                      dtype=_key_tile_dtype(shapes)),
        dge_mode=nisa.dge_mode.hwdge,
        name=name,
    )]


def _score_relu(
    raw: nl.NkiTensor,
    relu_sbuf: nl.NkiTensor,
    k_t_sbuf: nl.NkiTensor,
    q_t_sbuf: nl.NkiTensor,
    group: int,
    csz: int,
    shapes: IndexerScoreShape,
    rowtile: bool = False,
    by_group: bool = False,
) -> nl.NkiTensor:
    """Token group ``group``'s scores of the ``csz`` keys in ``k_t_sbuf`` into ``raw`` (PSUM), then
    ``relu_sbuf[(token, head), key]`` = relu of them in the kernel dtype (allocated here when None, after the
    score matmul). Returns ``relu_sbuf``. See ``_score_chunk``."""

    NH, HD = shapes.num_heads, shapes.head_dim
    kw = shapes.key_words
    gp = shapes.group_partitions(group)
    t0 = group * shapes.tokens_per_group

    # dst[(token, head), key] = <q[token, head], k[key]>. The contraction is
    # head_dim = 128, filling the PE rows exactly; the stationary is this group's 128
    # (token, head) columns and the moving axis is the whole key chunk. NOT double-buffered
    # by chunk parity: that measured no faster and wrong on the full-shard accuracy check.
    if rowtile:
        # IDX_SCORE_ROWTILE: k_t_sbuf is [2 * kw, csz / 2], keys [r * csz / 2, (r + 1) * csz / 2) of the chunk on
        # partitions [r * kw, (r + 1) * kw). PE row tile r scores them into raw[:, r * csz / 2 + m], i.e. in natural
        # key order: per parity one kw-deep matmul per row tile, the two tiles in flight at once, the odd features
        # accumulating onto the even ones as below. The relu after it is unchanged.
        kernel_assert(shapes.key_fp8 and csz % (2 * P_MAX) == 0, f"a row-tiled score needs fp8 keys, got {csz} keys")
        hc = csz // 2
        k_f8 = k_t_sbuf.view(nl.float8_e4m3fn)
        for parity, q_half in enumerate(q_t_sbuf):
            for r in range(2):
                p0 = r * kw
                nisa.nc_matmul(
                    dst=raw[0:gp, r * hc : (r + 1) * hc],
                    stationary=q_half[p0 : p0 + kw, t0 * NH : t0 * NH + gp],
                    moving=k_f8[p0 : p0 + kw, parity : parity + 2 * (hc - 1) + 1 : 2],
                    tile_position=(p0, 0),
                    tile_size=(kw, P_MAX),
                    accumulate=parity == 1,
                )
    elif shapes.key_fp8:
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
    # weight ride the head-sum weights, see _head_sum_weights). Each group's relu is split by key halves, the first on
    # Scalar's activation and the second as max(x, 0) on Vector, so the stage every head-sum matmul waits on is
    # shorter. Splitting by group instead left the PE idle in each layer at 128 ranks.
    if relu_sbuf is None:
        relu_sbuf = nl.ndarray((P_MAX, shapes.key_chunk), dtype=shapes.dtype, buffer=nl.sbuf)
    if by_group:
        _relu_one_engine(relu_sbuf, raw, gp, csz, group)
        return relu_sbuf
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
    return relu_sbuf


def _relu_one_engine(relu_sbuf: nl.NkiTensor, raw: nl.NkiTensor, gp: int, csz: int, group: int) -> None:
    """``IDX_RELU_BY_GROUP``: token group ``group``'s whole relu on one engine, Scalar for even groups, Vector for odd."""
    if group % 2 == 0:
        nisa.activation(relu_sbuf[0:gp, 0:csz], op=nl.relu, data=raw[0:gp, 0:csz])
    else:
        nisa.tensor_scalar(
            dst=relu_sbuf[0:gp, 0:csz], data=raw[0:gp, 0:csz], op0=nl.maximum, operand0=0.0, engine=nisa.engine.vector
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


def _packed_head_weights(head_w_sbuf: list[nl.NkiTensor], shapes: IndexerScoreShape) -> list[list[nl.NkiTensor]]:
    """Zero-padded 32-wide stationaries for the packed head sum: ``[group][slot] -> [P_MAX, 32]``.

    Slot ``j`` carries the group's ``[P_MAX, T]`` weights at columns ``j*S .. j*S+T`` and zeros elsewhere,
    so a matmul with it writes only its own ``S`` rows of a 32-row PSUM band and adds exact zeros to the
    others -- the same staggering ``attention_tkg`` uses for ``q_pad_sb``.
    """
    T, dtype = shapes.num_tokens, shapes.dtype
    slot_w, slots, _, _ = headsum_packed_geometry(T, shapes.num_keys)
    out = []
    for group in range(shapes.num_token_groups):
        per_slot = []
        for j in range(slots):
            w = nl.ndarray((P_MAX, HEADSUM_COL_TILE), dtype=dtype, buffer=nl.sbuf)
            nisa.memset(dst=w[0:P_MAX, 0:HEADSUM_COL_TILE], value=0.0)
            nisa.tensor_copy(dst=w[0:P_MAX, j * slot_w : j * slot_w + T], src=head_w_sbuf[group][0:P_MAX, 0:T])
            per_slot.append(w)
        out.append(per_slot)
    return out


def _packed_head_sum(head_sum_psum, w_pad, relu_sbuf, group: int, sl: int, j: int, gp: int) -> None:
    """Slice ``sl`` of a block (relu columns ``[j*512, (j+1)*512)``) into its band of the packed bank.

    Slice ``s`` uses column tile ``s % 4`` and the zero-padded stationary of slot ``s // 4``; the first write to
    a band (slot 0, group 0) initialises all 32 rows, since its stationary is zero outside its own slot."""
    tile, slot = sl % 4, sl // 4
    nisa.nc_matmul(
        dst=head_sum_psum[tile * HEADSUM_COL_TILE : (tile + 1) * HEADSUM_COL_TILE, 0:HEADSUM_SLICE],
        stationary=w_pad[group][slot][0:gp, 0:HEADSUM_COL_TILE],
        moving=relu_sbuf[0:gp, j * HEADSUM_SLICE : (j + 1) * HEADSUM_SLICE],
        tile_position=(0, tile * HEADSUM_COL_TILE),
        tile_size=(P_MAX, HEADSUM_COL_TILE),
        accumulate=not (slot == 0 and group == 0),
    )


def _is_pow2(n: int) -> bool:
    return n > 0 and n & (n - 1) == 0


def _i32_scalar(value: int = 0) -> nl.NkiTensor:
    """A ``[1, 1]`` int32 SBUF scalar, the only kind of state a ``fori_loop`` body carries."""
    out = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.memset(out, value)
    return out


def live_blocks(
    positions, rank_offset, num_keys: int, blk_keys: int, shard_degree: int, block_size: int
) -> nl.NkiTensor:
    """``[1, 1]`` int32 ``max(ceil(live / blk_keys), 1)``: the head-sum blocks holding a key below ``positions[0]``. The
    live keys are a logical prefix of the shard under either deal::

        live = (base // (shard_degree * block_size)) * block_size + clamp(base % round - rank_offset, 0, block_size)
        live = base                                                                 # contiguous

    At least one block, so no DMA has a zero count.
    """
    kernel_assert(_is_pow2(blk_keys), f"head-sum block {blk_keys} must be a power of two")
    base = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=base, src=positions[0:1].reshape((1, 1)))
    # A slot with no request carries a NEGATIVE cut (-1, "nothing resident"), and nl.right_shift is a LOGICAL
    # shift, so clamp first: a dead rank then takes the one-block floor like a just-started sequence.
    nisa.tensor_scalar(base, base, op0=nl.maximum, operand0=0)
    live = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    if shard_degree == 1:
        nisa.tensor_scalar(live, base, op0=nl.add, operand0=0)
    else:
        kernel_assert(rank_offset is not None, "a dealt shard needs rank_offset = shard_rank * block_size")
        rnd = shard_degree * block_size
        kernel_assert(_is_pow2(rnd), f"shard_degree * block_size = {rnd} must be a power of two")
        roff = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=roff, src=rank_offset[0:1, 0:1])
        whole = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_scalar(whole, base, op0=nl.right_shift, operand0=rnd.bit_length() - 1)
        part = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_scalar(part, base, op0=nl.bitwise_and, operand0=rnd - 1)
        nisa.scalar_tensor_tensor(dst=part, data=roff, op0=nl.multiply, operand0=-1.0, op1=nl.add, operand1=part)
        nisa.tensor_scalar(part, part, op0=nl.maximum, operand0=0, op1=nl.minimum, operand1=block_size)
        nisa.scalar_tensor_tensor(
            dst=live, data=whole, op0=nl.multiply, operand0=float(block_size), op1=nl.add, operand1=part
        )
    nisa.tensor_scalar(live, live, op0=nl.maximum, operand0=0, op1=nl.minimum, operand1=num_keys)
    n_blk = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(n_blk, live, op0=nl.add, operand0=blk_keys - 1)
    nisa.tensor_scalar(n_blk, n_blk, op0=nl.right_shift, operand0=blk_keys.bit_length() - 1)
    nisa.tensor_scalar(n_blk, n_blk, op0=nl.maximum, operand0=1)
    return n_blk


class _DynamicLoop:
    """The packed score's loop over the live head-sum blocks, its trip count computed on chip (``IDX_DYNAMIC_LOOP``).

    Every SBUF tile the body touches is allocated here, outside the loop (a tile allocated in a ``fori_loop`` body can
    land on a live one); the key tiles form a ring so the next iteration's loads wait only for the matching matmul. The
    per-block offsets all come from one block cursor. A block's keys are read through the HWDGE stage (``IDX_KEY_HWDGE``)
    or the row-window gather; a direct transpose cannot take a runtime offset in a loop body.
    """

    def __init__(self, key_rows: nl.NkiTensor, shapes: IndexerScoreShape):
        self.key_rows, self.shapes = key_rows, shapes
        T, chunk = shapes.num_tokens, shapes.key_chunk
        self.slot_w, _, self.blk_keys, self.n_blocks = headsum_packed_geometry(T, shapes.num_keys)
        self.n_slices = self.blk_keys // HEADSUM_SLICE
        self.rows = min(4, self.n_slices) * HEADSUM_COL_TILE
        kernel_assert(shapes.num_keys % self.blk_keys == 0, f"num_keys={shapes.num_keys} vs block {self.blk_keys}")
        kernel_assert(chunk % HEADSUM_SLICE == 0, f"key_chunk={chunk} must be a multiple of {HEADSUM_SLICE}")
        kernel_assert(self.n_slices <= 4 or self.n_slices % 4 == 0, f"{self.n_slices} slices do not tile 4 bands")
        # IDX_SCORE_ROWTILE (fp8): a row-tiled block's key tile holds its chunk's two halves on the two partition halves,
        # [2 * key_words, chunk / 2], a view of the same tile as the [key_words, chunk] of a plain block.
        self.rowtile = IDX_SCORE_ROWTILE and shapes.key_fp8
        kt_shape = (2 * shapes.key_words, chunk) if self.rowtile else (shapes.key_words, chunk)
        self.k_t = [
            nl.ndarray(kt_shape, dtype=_key_tile_dtype(shapes), buffer=nl.sbuf)
            for _ in range(self.blk_keys // chunk)
        ]
        # IDX_SCORE_KEYPAIR: the loop's keys row-tiled by pairs (even keys on partitions 0-63, odd on 64-127).
        self.keypair = IDX_SCORE_KEYPAIR and self.rowtile and IDX_KEY_HWDGE == 1
        # IDX_KEY_HWDGE: two bf16 HBM stages of one block's keys, alternated by an unrolled block's parity.
        self.key_stage = (
            [nl.ndarray((self.blk_keys, shapes.key_words), dtype=nl.bfloat16, buffer=layer_scratch_hbm()) for _ in range(2)]
            if IDX_KEY_HWDGE
            else None
        )
        self.relu = [nl.ndarray((P_MAX, chunk), dtype=shapes.dtype, buffer=nl.sbuf) for _ in range(DYNAMIC_RELU_POOL)]
        # The row window: rows_base[p, tile] = tile * 128 + p, plus the block's first key (cur_p, every partition).
        blk_tiles = self.blk_keys // P_MAX
        self.rows_base = nl.ndarray((P_MAX, blk_tiles), dtype=nl.int32, buffer=nl.sbuf)
        nisa.iota(self.rows_base, pattern=[[P_MAX, blk_tiles]], offset=0, channel_multiplier=1)
        self.rows_win = nl.ndarray((P_MAX, blk_tiles), dtype=nl.uint32, buffer=nl.sbuf)
        self.cur_p = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        self.mask = nl.ndarray((P_MAX, HEADSUM_SLICE), dtype=nl.float32, buffer=nl.sbuf)
        self.ev = nl.ndarray((P_MAX, HEADSUM_SLICE), dtype=nl.bfloat16, buffer=nl.sbuf)
        self.cur = _i32_scalar(0)  # the block of this iteration
        # IDX_KEY_PREFETCH: the HWDGE stage row of the block after a chunk's last one.
        self.prefetch = IDX_KEY_PREFETCH and IDX_KEY_HWDGE == 1 and IDX_CHUNK_PEEL
        self.keys_nxt = _i32_scalar(0) if self.prefetch else None
        self.keys_cur = _i32_scalar(0)  # cur * blk_keys: key rows and score columns
        self.cols_cur = _i32_scalar(0)  # cur * 512: mask columns
        # The giga kernel emits this body once per indexer layer; DMA names must not repeat within a trace.
        self.name = unique_name("idx_dyn")

    def _block(
        self, q_t_sbuf, w_pad, mask_add, scores_out, tag: str, out_col0: int = None, stage_idx: int = 0,
        static_base: int = None, prefetched: bool = False, k_tiles: list = None, mask_tile: nl.NkiTensor = None,
        mask_ready: bool = False,
    ) -> None:
        """One block at the cursor: mask window, key loads, score, relu, packed head sum, eviction, stores.

        ``out_col0`` stores the block at that STATIC column of ``scores_out`` (a chunk scratch) instead of at
        the cursor's runtime column (the whole-capacity ``scores_out``). ``prefetched``: the key tiles already hold
        this block's keys (``IDX_KEY_PREFETCH``), so it neither stages nor loads them."""
        shapes = self.shapes
        T, chunk = shapes.num_tokens, shapes.key_chunk
        kt = self.k_t if k_tiles is None else k_tiles  # _PooledChunkLoop's own-chunk rings
        nisa.tensor_scalar(self.keys_cur, self.cur, op0=nl.multiply, operand0=self.blk_keys)
        nisa.tensor_scalar(self.cols_cur, self.cur, op0=nl.multiply, operand0=HEADSUM_SLICE)
        if static_base is None and not key_block_uses_hwdge(stage_idx):  # the gather's row window
            nisa.tensor_scalar(self.rows_win, self.rows_base, op0=nl.add, operand0=self.cur_p[0:P_MAX, 0:1])
        mt = self.mask if mask_tile is None else mask_tile  # _PooledChunkLoop: the own block's preloaded mask tile
        if not mask_ready:
            nisa.dma_copy(
                dst=mt,
                src=mask_add.ap(
                    pattern=[[shapes.mask_cols, P_MAX], [1, HEADSUM_SLICE]],
                    offset=0,
                    scalar_offset=self.cols_cur,
                    indirect_dim=1,
                ),
                name=f"{self.name}_{tag}_mask",
            )
        head_sum_psum = nl.ndarray((P_MAX, HEADSUM_SLICE), dtype=nl.float32, buffer=nl.psum)
        # static_base: a peeled block at a trace-time key offset reads its keys with the static direct transpose.
        use_hw = static_base is None and key_block_uses_hwdge(stage_idx)
        if use_hw and not prefetched:
            stage = self.key_stage[stage_idx % 2]
            _stage_key_block(stage, self.key_rows, self.keys_cur, self.blk_keys, shapes, name=f"{self.name}_{tag}_stage")
        pending = None
        for key_chunk in TiledRange(self.blk_keys, chunk):
            csz = key_chunk.size
            kw = shapes.key_words
            # IDX_SCORE_ROWTILE, peeled (static) blocks only: the chunk's two halves, each into its own partition half
            # of the key tile.
            rowtile = self.rowtile and static_base is not None
            if rowtile:
                k_t_full = kt[key_chunk.index][0 : 2 * kw, 0 : csz // 2]
                halves = [(k_t_full[h * kw : (h + 1) * kw, 0 : csz // 2], h * (csz // 2), csz // 2) for h in range(2)]
            elif self.keypair:
                # IDX_SCORE_KEYPAIR: 256 key pairs, even keys on partitions [0, kw), odd ones on [kw, 2 kw).
                k_t_full = kt[key_chunk.index][0 : 2 * kw, 0 : csz // 2]
                halves = [(k_t_full, 0, csz)]
            else:
                k_t_full = kt[key_chunk.index][0:kw, 0:chunk] if self.rowtile else kt[key_chunk.index]
                halves = [(k_t_full, 0, csz)]
            for h, (k_t_sbuf, off, hsz) in enumerate(halves):
                kname = f"{self.name}_{tag}_k{key_chunk.index}" + (f"h{h}" if rowtile else "")
                if prefetched:
                    pass
                elif static_base is not None:
                    _load_key_chunk(
                        k_t_sbuf, self.key_rows, None, static_base + key_chunk.start_offset + off, hsz, shapes,
                        name=kname,
                    )
                elif use_hw and self.keypair:
                    # The stage rows as [pairs, 2 * kw]: pair base start / 2, 256 pairs.
                    _load_key_chunk_hwdge(
                        k_t_sbuf, stage, None, (key_chunk.start_offset + off) // 2, hsz // 2, shapes, name=kname,
                        words=2 * kw,
                    )
                elif use_hw:
                    _load_key_chunk_hwdge(
                        k_t_sbuf,
                        stage,
                        None,
                        key_chunk.start_offset + off,
                        hsz,
                        shapes,
                        name=kname,
                    )
                else:
                    _load_key_chunk(
                        k_t_sbuf,
                        self.key_rows,
                        self.rows_win,
                        key_chunk.start_offset + off,
                        hsz,
                        shapes,
                        name=kname,
                    )
            k_t_sbuf = k_t_full
            relus = []
            for group in range(shapes.num_token_groups):
                gp = shapes.group_partitions(group)
                raw = nl.ndarray((P_MAX, chunk), dtype=SCORE_PSUM_DTYPE, buffer=nl.psum)
                relu_sbuf = self.relu[(key_chunk.index * shapes.num_token_groups + group) % len(self.relu)]
                _score_relu(
                    raw, relu_sbuf, k_t_sbuf, q_t_sbuf, group, csz, shapes, rowtile=rowtile or (self.keypair and static_base is None),
                    by_group=IDX_RELU_BY_GROUP,
                )
                if IDX_HEADSUM_LAG:
                    relus.append((group, gp, relu_sbuf))
                else:
                    self._head_sums(head_sum_psum, w_pad, key_chunk.start_offset, csz, [(group, gp, relu_sbuf)])
            # IDX_HEADSUM_LAG: a key chunk's head sums are traced after the NEXT chunk's score matmuls, so the PE scores
            # while Scalar/Vector run the relu instead of waiting for it. Same matmuls, same order per band.
            if IDX_HEADSUM_LAG:
                if pending is not None:
                    self._head_sums(head_sum_psum, w_pad, *pending)
                pending = (key_chunk.start_offset, csz, relus)
        if pending is not None:
            self._head_sums(head_sum_psum, w_pad, *pending)
        if self.keypair and static_base is None:
            # IDX_SCORE_KEYPAIR: bank column r * 256 + m holds key 2 m + r of its slice; back to natural order.
            hc = HEADSUM_SLICE // 2
            for r in range(2):
                self.last_evict = nisa.tensor_tensor(
                    dst=self.ev[0 : self.rows, r : r + 2 * (hc - 1) + 1 : 2],
                    data1=head_sum_psum[0 : self.rows, r * hc : (r + 1) * hc],
                    data2=mt[0 : self.rows, r : r + 2 * (hc - 1) + 1 : 2],
                    op=nl.add,
                    engine=nisa.engine.vector,
                )
        else:
            # The eviction's handle is kept for a subclass's scheduling edges (_PooledChunkLoop.stage).
            self.last_evict = nisa.tensor_tensor(
                dst=self.ev[0 : self.rows, 0:HEADSUM_SLICE],
                data1=head_sum_psum[0 : self.rows, 0:HEADSUM_SLICE],
                data2=mt[0 : self.rows, 0:HEADSUM_SLICE],
                op=nl.add,
                engine=nisa.engine.vector,
            )
        # Row (tile, slot, t) holds slice s = slot*4 + tile: -> scores_out[t, cur * block + s * 512 + c].
        self._block_stores(scores_out, out_col0, tag)
        nisa.tensor_scalar(dst=self.cur, data=self.cur, op0=nl.add, operand0=1)
        nisa.tensor_scalar(dst=self.cur_p, data=self.cur_p, op0=nl.add, operand0=float(self.blk_keys))

    def _block_stores(self, scores_out, out_col0, tag: str) -> None:
        """A block's score stores out of ``self.ev``; see ``_block``."""
        shapes = self.shapes
        T = shapes.num_tokens
        for sl in range(self.n_slices):
            p0 = (sl % 4) * HEADSUM_COL_TILE + (sl // 4) * self.slot_w
            if out_col0 is None:
                dst = scores_out.ap(
                    pattern=[[shapes.num_keys, T], [1, HEADSUM_SLICE]],
                    offset=sl * HEADSUM_SLICE,
                    scalar_offset=self.keys_cur,
                    indirect_dim=1,
                )
            else:
                c0 = out_col0 + sl * HEADSUM_SLICE
                dst = scores_out[0:T, c0 : c0 + HEADSUM_SLICE]
            # GLM52K_IDX_SELECT_2Q: odd slices' stores from the Sync queue, so the two halves issue in parallel.
            queue = (
                {"dge_mode": nisa.dge_mode.hwdge, "engine": nisa.engine.sync}
                if IDX_SELECT_2Q and sl % 2
                else {}
            )
            nisa.dma_copy(
                dst=dst,
                src=self.ev[p0 : p0 + T, 0:HEADSUM_SLICE],
                name=f"{self.name}_{tag}_out{sl}",
                **queue,
            )

    def _prefetch_block(self, tag: str, k_tiles: list = None, stage_idx: int = 0) -> list:
        """``IDX_KEY_PREFETCH``: stage and transpose the keys of the block at the cursor into stage 0 and the key tiles,
        as that block's own reads would (``_block`` with ``prefetched``). The row is clamped into the cache, so the
        prefetch after the last live chunk reads real rows that nothing consumes."""
        shapes = self.shapes
        chunk, kw = shapes.key_chunk, shapes.key_words
        stage = self.key_stage[stage_idx]
        dmas = []  # the stage copy and transposes, for a caller's scheduling edge (the CDP preload)
        kt = self.k_t if k_tiles is None else k_tiles
        nisa.tensor_scalar(
            self.keys_nxt, self.cur, op0=nl.multiply, operand0=self.blk_keys, op1=nl.minimum,
            operand1=shapes.num_keys - self.blk_keys,
        )
        dmas += _stage_key_block(stage, self.key_rows, self.keys_nxt, self.blk_keys, shapes, name=f"{self.name}_{tag}_pstage")
        for key_chunk in TiledRange(self.blk_keys, chunk):
            if self.keypair:
                dmas.append(_load_key_chunk_hwdge(
                    kt[key_chunk.index][0 : 2 * kw, 0 : key_chunk.size // 2], stage, None,
                    key_chunk.start_offset // 2, key_chunk.size // 2, shapes, name=f"{self.name}_{tag}_pk{key_chunk.index}",
                    words=2 * kw,
                ))
                continue
            k_t = kt[key_chunk.index][0:kw, 0:chunk] if self.rowtile else kt[key_chunk.index]
            dmas.append(_load_key_chunk_hwdge(
                k_t, stage, None, key_chunk.start_offset, key_chunk.size, shapes, name=f"{self.name}_{tag}_pk{key_chunk.index}",
            ))
        return dmas

    @staticmethod
    def _head_sums(head_sum_psum, w_pad, start: int, csz: int, relus) -> None:
        """The packed head-sum matmuls of one key chunk (``relus``: ``(group, partitions, relu tile)`` per group)."""
        for group, gp, relu_sbuf in relus:
            for j in range(csz // HEADSUM_SLICE):
                sl = (start + j * HEADSUM_SLICE) // HEADSUM_SLICE
                _packed_head_sum(head_sum_psum, w_pad, relu_sbuf, group, sl, j, gp)

    def score_chunks(
        self,
        q_t_sbuf,
        head_w_sbuf,
        mask_add,
        positions,
        rank_offset,
        shard_degree: int,
        block_size: int,
        seg_keys: int,
        chunk_select,
    ) -> None:
        """``IDX_CHUNK_LOOP``: one ``nl.fori_loop`` over the live top-k chunks of ``seg_keys`` keys. Iteration ``c`` scores
        the chunk's head-sum blocks, unrolled, into one chunk-wide scratch and calls ``chunk_select(chunk_scores, slot,
        label_p)``, which selects from it into slot ``c``. ``label_p`` is the chunk's term of the global position. A chunk
        past the live length runs nothing.
        """
        shapes = self.shapes
        T = shapes.num_tokens
        blk_keys = self.blk_keys
        kernel_assert(
            seg_keys >= blk_keys and seg_keys % blk_keys == 0 and shapes.num_keys % seg_keys == 0,
            f"top-k chunk of {seg_keys} keys vs {blk_keys}-key blocks and {shapes.num_keys} keys",
        )
        seg_blocks = seg_keys // blk_keys
        chunk_scores = nl.ndarray((T, seg_keys), dtype=nl.bfloat16, buffer=layer_scratch_hbm())
        n_live = live_blocks(positions, rank_offset, shapes.num_keys, seg_keys, shard_degree, block_size)
        w_pad = _packed_head_weights(head_w_sbuf, shapes)
        slot = _i32_scalar(0)
        label_p = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(label_p, 0.0)
        nisa.memset(self.cur, 0)
        nisa.memset(self.cur_p, 0.0)
        if IDX_CHUNK_PEEL:
            # Chunk 0 always runs: outside the loop, at static offsets (IDX_CHUNK_PEEL).
            for j in range(seg_blocks):
                self._block(
                    q_t_sbuf, w_pad, mask_add, chunk_scores, f"cp{j}", out_col0=j * blk_keys, stage_idx=j,
                    static_base=j * blk_keys,
                )
            chunk_select(chunk_scores, slot, label_p)
            if self.prefetch:
                self._prefetch_block("cpp")  # chunk 1's first block, for the loop's first iteration
            nisa.tensor_scalar(dst=slot, data=slot, op0=nl.add, operand0=1)
            nisa.tensor_scalar(dst=label_p, data=label_p, op0=nl.add, operand0=float(shard_degree * seg_keys))
            nisa.tensor_scalar(dst=n_live, data=n_live, op0=nl.add, operand0=-1)
        def one_chunk(scores, tag):
            for j in range(seg_blocks):
                self._block(
                    q_t_sbuf, w_pad, mask_add, scores, f"{tag}{j}", out_col0=j * blk_keys, stage_idx=j,
                    prefetched=self.prefetch and j == 0,
                )
            if self.prefetch:
                self._prefetch_block(f"{tag}p")  # the next chunk's first block, under this chunk's select

        def advance():
            nisa.tensor_scalar(dst=slot, data=slot, op0=nl.add, operand0=1)
            nisa.tensor_scalar(dst=label_p, data=label_p, op0=nl.add, operand0=float(shard_degree * seg_keys))

        trip_reg = nisa.register_alloc()
        nisa.register_load(trip_reg, n_live)

        def body(_i):
            one_chunk(chunk_scores, "cl")
            chunk_select(chunk_scores, slot, label_p)
            advance()

        nl.fori_loop(0, trip_reg, body)


POOL_ROW_ALIGN = 16


# ── The chunk-dealt pool (the 1M point's indexer) ──────────────────────────────────────────────────────────────
#
# The pool deals whole top-k chunks of 32,768 keys: chunk q of the sequence in pool slot s lives on member (s + q) % E,
# so region r of member e is chunk r of owner (e - r) % E and region 0 is the member's own chunk 0. Each member
# publishes its query halves, head weights and new keys in one tile for the pool all_gather
# (``indexer_pool_cdp_publish``), scores its own chunk 0 with the static blocks, runs one loop over its live foreign
# regions (the chunk loop's body plus the chunk select, the 2,048 candidates going to the owner's part of the send
# buffer), hands the send buffer to the pool all_to_all, and writes the pool's new keys that land in its cache. Every
# 32,768-key select therefore sees the same bytes in the same order as the unpooled chunk loop.
#
# Host tables (``layout/decode/idx_pool.py`` ``IdxChunkPool.tables``):
#   items [R, 8] int32       row i < n: the i-th live foreign region (region, owner, block cursor, the send offsets of
#                            its values and positions, label); rows >= n repeat a valid row for the last prefetch
#   n_items [1, 1] int32     n, the foreign loop's trip count
#   new_key_rows [E*T, 1]    this rank's cache row of pool slot s's new key t, or -1
# The send buffer is [E * T, 3 * topk / 2] uint32: row s * T + t holds owner s's token t, its topk bf16 values then
# its topk int32 positions.

CDP_ITEM_REGION, CDP_ITEM_OWNER, CDP_ITEM_CURSOR, CDP_ITEM_SEND_V, CDP_ITEM_SEND_P, CDP_ITEM_LABEL = range(6)
CDP_ITEM_COLS = 8


def cdp_publish_layout(num_tokens: int, num_heads: int, num_groups: int) -> tuple[int, int, int]:
    """``(w_col, k_col, width)`` of the CDP publish tile ``[128, width]`` (and of each 128-row member of ``pool_qw``).

    Columns ``[0, T*NH)``: the fp8 score's query halves, the even features on rows ``[0, HD/2)`` and the odd ones on
    ``[HD/2, HD)`` (column ``t * NH + h``, as ``_load_query_feature_major``); ``[w_col, w_col + G*T)``: token group
    ``g``'s ``_head_sum_weights`` tile at ``w_col + g*T`` (all 128 rows); ``[k_col, k_col + T)``: the new keys,
    feature-major (``k_new^T``); padded to a multiple of 16. 288 at 8 tokens x 32 heads (256 + 16 + 8 -> 288)."""
    w_col = num_tokens * num_heads
    k_col = w_col + num_groups * num_tokens
    return w_col, k_col, div_ceil(k_col + num_tokens, POOL_ROW_ALIGN) * POOL_ROW_ALIGN


def cdp_candidate_words(topk: int) -> int:
    """uint32 words of one token's candidate slot in the CDP send / recv buffers: topk bf16 values, topk int32."""
    return 3 * topk // 2


def indexer_pool_cdp_publish(
    q_t_sbuf: tuple, head_w_sbuf: list, k_new_sbuf: nl.NkiTensor, shapes: IndexerScoreShape, name: str = None
) -> nl.NkiTensor:
    """This rank's share of the CDP ``pool_qw``: the tiles the front already holds, in one ``[128, W]`` private-HBM tile.

    No recompute: byte copies of ``q_t_sbuf`` (the two fp8 query halves, ``IDX_SCORE_ROWTILE`` so each half is on both
    partition halves), ``head_w_sbuf`` (one ``[128, T]`` per token group) and ``k_new_sbuf`` (``[T, HD]``, transposed
    on the PE, which moves bf16 exactly), assembled lane-wise in SBUF and stored with ONE DMA on the Sync queue, so the
    pool all_gather that reads it can be triggered at once. ``cdp_publish_layout`` gives the columns.
    """
    T, NH, HD = shapes.num_tokens, shapes.num_heads, shapes.head_dim
    kw, G = shapes.key_words, shapes.num_token_groups
    kernel_assert(
        shapes.key_fp8 and 2 * kw == HD and HD == P_MAX and IDX_SCORE_ROWTILE,
        f"the CDP publish packs the fp8 query halves on the two partition halves (head_dim {HD}, fp8 {shapes.key_fp8}, "
        f"GLM52K_IDX_SCORE_ROWTILE={int(IDX_SCORE_ROWTILE)})",
    )
    w_col, k_col, W = cdp_publish_layout(T, NH, G)
    tnh = T * NH
    pub = nl.ndarray((P_MAX, W), dtype=shapes.dtype, buffer=nl.sbuf)
    # Even half's rows [0, kw) and odd half's rows [kw, 2 kw): both lane-wise (the odd half's copy on [kw, 2 kw)).
    nisa.tensor_copy(dst=pub[0:kw, 0:tnh], src=q_t_sbuf[0][0:kw, 0:tnh])
    nisa.tensor_copy(dst=pub[kw : 2 * kw, 0:tnh], src=q_t_sbuf[1][kw : 2 * kw, 0:tnh])
    for g in range(G):
        nisa.tensor_copy(dst=pub[0:P_MAX, w_col + g * T : w_col + (g + 1) * T], src=head_w_sbuf[g][0:P_MAX, 0:T])
    # private_hbm: the source of the pool all_gather (at LNC=1 a collective's source may not be shared).
    out = nl.ndarray((P_MAX, W), dtype=shapes.dtype, buffer=nl.private_hbm)
    # The keys token-major by their own DMA (Scalar queue, in parallel with the Sync one): row 16 t + j, columns
    # [k_col, k_col + 8) hold k_new[t, 8 j : 8 j + 8] (cdp_key_rows). Columns past them stay unwritten (never read).
    kr, kc = cdp_key_rows(T, HD)
    nisa.dma_copy(
        dst=out[0:P_MAX, 0:k_col], src=pub[0:P_MAX, 0:k_col], dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync,
        name=f"{name or unique_name('idx_cdp')}_publish",
    )
    nisa.dma_copy(
        dst=out.ap(pattern=[[kr * W, T], [W, kr], [1, kc]], offset=k_col),
        src=k_new_sbuf[0:T, 0:HD].reshape_dim(1, [kr, kc]), dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.scalar,
        name=f"{name or unique_name('idx_cdp')}_publish_k",
    )
    return out


def cdp_key_rows(num_tokens: int, head_dim: int) -> tuple[int, int]:
    """``(rows per token, features per row)`` of the token-major new keys in the publish tile."""
    kr = P_MAX // num_tokens
    kernel_assert(P_MAX % num_tokens == 0 and head_dim % kr == 0, f"{num_tokens} tokens x {head_dim} features")
    return kr, head_dim // kr


def indexer_cdp_front(
    q, hidden_states, wk, k_norm_weight, k_norm_bias, weights_proj, cos, sin, shapes: IndexerScoreShape, norm_eps: float,
    prologue: tuple | None = None,
) -> tuple:
    """``(q_t_sbuf, head_w_sbuf, k_new_sbuf)``: ``indexer_kv_score``'s front, the same instructions, minus the cache write
    (with ``prologue``, only the query load runs here). What :func:`indexer_pool_cdp_publish` packs."""
    T, HD = shapes.num_tokens, shapes.head_dim
    if prologue is not None:
        head_w_sbuf, rope_rows, k_new_sbuf = prologue
        return _load_query_feature_major(q, None, None, shapes, rope_rows=rope_rows), head_w_sbuf, k_new_sbuf
    cos_sbuf, sin_sbuf = load_rope_tables(cos, sin, T)
    hidden_sbuf = nl.ndarray((P_MAX, shapes.hidden), dtype=shapes.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=hidden_sbuf[0:T, 0 : shapes.hidden], src=hidden_states[0:T, 0 : shapes.hidden])
    hidden_tiles_sbuf = transpose_into_k_tiles(hidden_sbuf[0:P_MAX, 0 : shapes.hidden], T)
    k_new_sbuf = _project_and_norm_key(hidden_tiles_sbuf, wk, k_norm_weight, k_norm_bias, norm_eps, shapes)
    rope_inplace(k_new_sbuf, cos_sbuf, sin_sbuf, 0, HD, 1, T)
    head_w_sbuf = _head_sum_weights(hidden_tiles_sbuf, weights_proj, shapes)
    q_t_sbuf = _load_query_feature_major(q, cos_sbuf, sin_sbuf, shapes)
    return q_t_sbuf, head_w_sbuf, k_new_sbuf


class _PooledChunkLoop(_DynamicLoop):
    """The chunk-dealt pool's score loop, built on ``_DynamicLoop``: :meth:`own_blocks` scores this member's own chunk 0
    with the static (peeled) blocks; :meth:`foreign` is one ``nl.fori_loop`` over the live foreign regions, each the chunk
    loop's body followed by the chunk select into the owner's part of the send buffer; :meth:`stage` loads the next
    item's query halves and head weights from the gathered ``pool_qw`` under the current select. All SBUF state is
    allocated before the loop, and nothing the loop writes in SBUF is read after it.
    """

    def __init__(self, key_rows: nl.NkiTensor, shapes: IndexerScoreShape, pool_qw: nl.NkiTensor, items: nl.NkiTensor,
                 seg_keys: int):
        super().__init__(key_rows, shapes)
        T, NH = shapes.num_tokens, shapes.num_heads
        G = shapes.num_token_groups
        kernel_assert(
            shapes.key_fp8 and self.rowtile and IDX_KEY_HWDGE == 1 and IDX_CHUNK_PEEL and HEADSUM_PACKED,
            "the CDP loop is the served chunk loop: fp8 keys, GLM52K_IDX_SCORE_ROWTILE=1, "
            "GLM52K_IDX_KEY_HWDGE=1, GLM52K_IDX_CHUNK_PEEL=1, GLM52K_IDX_HEADSUM_PACKED=1",
        )
        kernel_assert(
            seg_keys % self.blk_keys == 0 and shapes.num_keys % seg_keys == 0,
            f"top-k chunk of {seg_keys} keys vs {self.blk_keys}-key blocks and {shapes.num_keys} keys",
        )
        self.seg_keys, self.seg_blocks = seg_keys, seg_keys // self.blk_keys
        self.pool_qw, self.items = pool_qw, items
        self.n_regions = items.shape[0]
        kernel_assert(
            tuple(items.shape) == (shapes.num_keys // seg_keys, CDP_ITEM_COLS),
            f"items is {tuple(items.shape)}, expected ({shapes.num_keys // seg_keys}, {CDP_ITEM_COLS}): one row per region",
        )
        self.w_col, self.k_col, self.width = cdp_publish_layout(T, NH, G)
        if pool_qw is not None:
            self.bind_pool(pool_qw)
        self.tnh = T * NH
        # The staged owner's query: both halves on both partition halves (the rows the row-tiled score reads), in one
        # tile, parity p at columns [p * T*NH, (p + 1) * T*NH): one DMA per partition half stages both parities.
        self.q_f2 = nl.ndarray((P_MAX, 2 * self.tnh), dtype=shapes.dtype, buffer=nl.sbuf)
        self.q_f = tuple(self.q_f2[0:P_MAX, p * self.tnh : (p + 1) * self.tnh] for p in range(2))
        # The head-sum stationaries of _packed_head_weights, [group][slot] -> [128, 32], as windows of ONE tile (zeros
        # once): slot j of group g at columns (g * slots + j) * 32, its weights at + j * slot_w. One DMA fills all of
        # them per item (a stride-0 source over the slots) instead of a weight load and 8 lane-wise copies.
        self.slots = slots = HEADSUM_COL_TILE // self.slot_w
        self.wb_cols = G * slots * HEADSUM_COL_TILE
        self.w_big = nl.ndarray((P_MAX, self.wb_cols), dtype=shapes.dtype, buffer=nl.sbuf)
        nisa.memset(dst=self.w_big[0:P_MAX, 0 : self.wb_cols], value=0.0)
        self.w_pad_f = [
            [
                self.w_big[0:P_MAX, (g * slots + j) * HEADSUM_COL_TILE : (g * slots + j + 1) * HEADSUM_COL_TILE]
                for j in range(slots)
            ]
            for g in range(G)
        ]
        self.last_evict = None
        self.k_t2 = None  # the own chunk's second key-tile ring (own_ring)
        self.loop_kt = None  # the foreign loop's own key-tile ring
        self.item0_early = False  # item 0's row and keys loaded by early_item0
        self.own_masks = None  # the own blocks' two mask tiles (own_mask)
        self.qrow = _i32_scalar(0)  # P_MAX * owner: the staged owner's first row of pool_qw
        self.row = nl.ndarray((1, CDP_ITEM_COLS), dtype=nl.int32, buffer=nl.sbuf)  # this item's row
        self.row_n = nl.ndarray((1, CDP_ITEM_COLS), dtype=nl.int32, buffer=nl.sbuf)  # the next item's row
        self.label_f = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)  # this item's chunk term
        nisa.memset(self.cur_p, 0.0)  # read only by the gather path, which KEY_HWDGE=1 never takes
        self.name = unique_name("idx_cdp")

    def bind_pool(self, pool_qw: nl.NkiTensor) -> None:
        """The gathered ``pool_qw`` [E * 128, W], bound after the loop object is built (the preload runs first)."""
        kernel_assert(
            pool_qw.shape[1] == self.width and pool_qw.shape[0] % P_MAX == 0,
            f"pool_qw is {tuple(pool_qw.shape)}, expected (E * {P_MAX}, {self.width})",
        )
        self.pool_qw = pool_qw

    def preload(self, blocks, mask_add: nl.NkiTensor = None) -> None:
        """Stage and transpose own blocks ``blocks`` (one or two) into the key-tile rings
        (:meth:`own_ring`), at the block cursor, as the loop's prefetch does; :meth:`own_blocks` then runs them
        ``prefetched``."""
        self.preload_dmas = []
        for j in blocks:
            nisa.memset(self.cur, j)
            self.preload_dmas += self._prefetch_block(f"cpl{j}", k_tiles=self.own_ring(j), stage_idx=j % 2)
            if mask_add is not None:
                # The block's mask window (static: own block j is mask columns [512 j, 512 (j + 1))).
                self.preload_dmas.append(nisa.dma_copy(
                    dst=self.own_mask(j), src=mask_add[0:P_MAX, j * HEADSUM_SLICE : (j + 1) * HEADSUM_SLICE],
                    name=f"{self.name}_cpl{j}_mask",
                ))

    def own_mask(self, j: int) -> nl.NkiTensor:
        """Own block ``j``'s mask tile (two, by block parity), allocated on first use."""
        if self.own_masks is None:
            self.own_masks = [nl.ndarray((P_MAX, HEADSUM_SLICE), dtype=nl.float32, buffer=nl.sbuf) for _ in range(2)]
        return self.own_masks[j % 2]

    def own_ring(self, j: int) -> list:
        """The key-tile ring of own block ``j``: odd blocks a second ring, even ones the loop's (the foreign loop has a
        ring of its own, so block 2 may hold the base one across it)."""
        if j % 2 == 0:
            return self.k_t
        shapes = self.shapes
        kt_shape = (2 * shapes.key_words, shapes.key_chunk) if self.rowtile else (shapes.key_words, shapes.key_chunk)
        if self.k_t2 is None:
            self.k_t2 = [
                nl.ndarray(kt_shape, dtype=_key_tile_dtype(shapes), buffer=nl.sbuf) for _ in range(len(self.k_t))
            ]
        return self.k_t2

    def own_blocks(self, j0: int, j1: int, q_t_sbuf, w_pad, mask_add, scores) -> None:
        """Blocks ``[j0, j1)`` of the own chunk 0, static, into ``scores`` (``score_chunks``' peel, from block ``j0``)."""
        if j1 <= j0:
            return
        nisa.memset(self.cur, j0)
        nisa.memset(self.cur_p, float(j0 * self.blk_keys))
        for j in range(j0, j1):
            # The dynamic form at the cursor (memset above), as the loop body (HWDGE key stage, KEYPAIR tiles), its key
            # tiles and mask already loaded (preload).
            self._block(
                q_t_sbuf, w_pad, mask_add, scores, f"cd{j}", out_col0=j * self.blk_keys, stage_idx=j,
                prefetched=1, k_tiles=self.own_ring(j), mask_tile=self.own_mask(j), mask_ready=1,
            )

    def stage(self, row: nl.NkiTensor, tag: str, after=None, queue: str = "pool") -> None:
        """The query halves and head weights of ``row``'s owner (``row`` [1, 8] int32 SBUF), from ``pool_qw``, in 3 DMAs.

        On GpSimd's software DGE: GpSimd is idle between a block's eviction and the select's top-k, and before the
        foreign loop a stage that waits for the pool all_gather stalls only GpSimd (on the Scalar queue it would stall
        the relus of the own chunk's blocks). ``after`` (an instruction at the same loop nesting: the last block's
        eviction): the DMAs depend on it, so the scheduler cannot hoist them into that block, where their register
        loads (each waits for the PE to release the tile) stalled their engine between the block's relus and its
        eviction (measured on each item). ``queue="act"``: the Scalar engine's hardware DGE instead (the pre-loop
        stage)."""
        shapes = self.shapes
        T, kw, W = shapes.num_tokens, shapes.key_words, self.width
        G = shapes.num_token_groups
        nisa.tensor_scalar(self.qrow, row[0:1, CDP_ITEM_OWNER : CDP_ITEM_OWNER + 1], op0=nl.multiply, operand0=P_MAX)
        if queue == "act":
            act = {"dge_mode": nisa.dge_mode.hwdge, "engine": nisa.engine.scalar}
        else:
            act = {"dge_mode": nisa.dge_mode.swdge}
        dmas = []
        # Partition half h of q_f2 = the owner's rows [0, kw) (the even half) then [kw, 2 kw) (the odd one), side by
        # side: a scalar-offset DMA costs its engine a register load and the address arithmetic (a measured cost on the Scalar
        # engine), so 2 for the query instead of 4.
        for h in range(2):
            dmas.append(nisa.dma_copy(
                dst=self.q_f2[h * kw : (h + 1) * kw, 0 : 2 * self.tnh].reshape_dim(1, [2, self.tnh]),
                src=self.pool_qw.ap(
                    pattern=[[W, kw], [kw * W, 2], [1, self.tnh]], offset=0, scalar_offset=self.qrow, indirect_dim=0
                ),
                name=f"{self.name}_{tag}_q{h}", **act,
            ))
        # Group g's weights [128, T] into every slot window j of its stationaries (stride 0 over j on the source).
        slots, ct = self.slots, HEADSUM_COL_TILE
        dmas.append(nisa.dma_copy(
            dst=self.w_big.ap(pattern=[[self.wb_cols, P_MAX], [slots * ct, G], [ct + self.slot_w, slots], [1, T]], offset=0),
            src=self.pool_qw.ap(
                pattern=[[W, P_MAX], [T, G], [0, slots], [1, T]], offset=self.w_col, scalar_offset=self.qrow, indirect_dim=0
            ),
            name=f"{self.name}_{tag}_w", **act,
        ))
        if after is not None:
            for d in dmas:
                d.depends_on(after)

    def early_item0(self) -> None:
        """Item 0's row and first key block (into the loop's own ring), before the publish."""
        kernel_assert(self.prefetch, "the CDP loop's early first item needs GLM52K_IDX_KEY_PREFETCH=1")
        shapes = self.shapes
        kt_shape = (2 * shapes.key_words, shapes.key_chunk) if self.rowtile else (shapes.key_words, shapes.key_chunk)
        self.loop_kt = [nl.ndarray(kt_shape, dtype=_key_tile_dtype(shapes), buffer=nl.sbuf) for _ in range(len(self.k_t))]
        nisa.dma_copy(dst=self.row_n, src=self.items[0:1, 0:CDP_ITEM_COLS], name=f"{self.name}_row0e")
        nisa.tensor_scalar(self.cur, self.row_n[0:1, CDP_ITEM_CURSOR : CDP_ITEM_CURSOR + 1], op0=nl.add, operand0=0)
        self._prefetch_block("cfie", k_tiles=self.loop_kt)
        self.item0_early = True

    def prefetch_item(self, tag: str, stage: bool = True, after=None, queue: str = "pool", keys_done: bool = False) -> None:
        """Point the block cursor at the item in ``row_n`` and, with ``IDX_KEY_PREFETCH``, prefetch its first key block
        (``_prefetch_block`` at the cursor) before its select, as the chunk loop does; ``stage`` also stages its owner
        (:meth:`stage`)."""
        nisa.tensor_scalar(self.cur, self.row_n[0:1, CDP_ITEM_CURSOR : CDP_ITEM_CURSOR + 1], op0=nl.add, operand0=0)
        if self.prefetch and not keys_done:
            self._prefetch_block(f"{tag}p", k_tiles=self.loop_kt)
        if stage:
            self.stage(self.row_n, tag, after=after, queue=queue)

    def _row_ap(self, reg, col: int, cols: int, parts: int = 1, next_row: bool = False):
        """Columns ``[col, col + cols)`` of items row ``reg`` (``+ 1`` with ``next_row``), on ``parts`` partitions
        (stride 0 down them)."""
        return self.items.ap(
            pattern=[[0 if parts > 1 else CDP_ITEM_COLS, parts], [1, cols]],
            offset=col + (CDP_ITEM_COLS if next_row else 0), scalar_offset=reg, indirect_dim=0,
        )

    def foreign(self, n_items, mask_add, chunk_scores, send_v, send_p, select_to, after=None) -> None:
        """The loop over the live foreign regions: item ``i`` scores region ``items[i]`` into ``chunk_scores`` and
        selects it with ``select_to(chunk_scores, send_v, send_p, off_v, off_p, label_p)``, ``off_v`` / ``off_p`` the
        item's send offsets ([1, 1] int32 SBUF) and ``label_p`` its chunk term ([128, 1] fp32 SBUF). ``after``: the
        eviction of the last own block before the loop, which item 0's stage (the first reader of the pool all_gather)
        waits for."""
        n_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=n_sb, src=n_items.reshape((1, 1))[0:1, 0:1], name=f"{self.name}_n")
        nisa.tensor_scalar(n_sb, n_sb, op0=nl.maximum, operand0=0, op1=nl.minimum, operand1=self.n_regions - 1)
        # Item 0: its cursor, owner and first key block, before the loop (row 0 is valid even when n is 0).
        # Row and keys already there (early_item0): the cursor is re-pointed, the owner staged.
        if not self.item0_early:
            nisa.dma_copy(dst=self.row_n, src=self.items[0:1, 0:CDP_ITEM_COLS], name=f"{self.name}_row0")
        # On GpSimd after the last own eviction before the loop: a stage hoisted into the own blocks would wait there
        # for the pool all_gather on the engine that runs half the relus.
        self.prefetch_item("cfi", after=after, queue="pool", keys_done=self.item0_early)
        trip_reg = nisa.register_alloc()
        nisa.register_load(trip_reg, n_sb)
        seg_blocks, blk_keys = self.seg_blocks, self.blk_keys

        def body(i):
            # This item's select operands and the next item's row, by the loop register (latency under the blocks).
            nisa.dma_copy(dst=self.row, src=self._row_ap(i, 0, CDP_ITEM_COLS), name=f"{self.name}_row")
            nisa.dma_copy(
                dst=self.label_f, src=self._row_ap(i, CDP_ITEM_LABEL, 1, parts=P_MAX), name=f"{self.name}_label"
            )
            nisa.dma_copy(dst=self.row_n, src=self._row_ap(i, 0, CDP_ITEM_COLS, next_row=True), name=f"{self.name}_rown")
            for j in range(seg_blocks):
                self._block(
                    self.q_f, self.w_pad_f, mask_add, chunk_scores, f"cf{j}", out_col0=j * blk_keys, stage_idx=j,
                    prefetched=self.prefetch and j == 0, k_tiles=self.loop_kt,
                )
            # The next item's cursor and first key block, under this item's select (the chunk loop's prefetch point); its
            # owner's query and weights after the last block's eviction (stage's ``after``), under the select too.
            self.prefetch_item("cfn", stage=False)
            select_to(
                chunk_scores, send_v, send_p, self.row[0:1, CDP_ITEM_SEND_V : CDP_ITEM_SEND_V + 1],
                self.row[0:1, CDP_ITEM_SEND_P : CDP_ITEM_SEND_P + 1], self.label_f,
            )
            self.stage(self.row_n, "cfs", after=self.last_evict)

        nl.fori_loop(0, trip_reg, body)

    def write_new_keys(self, cache_rows: nl.NkiTensor, new_key_rows: nl.NkiTensor) -> None:
        """Scatter the pool's new keys (``pool_qw`` columns ``[k_col, k_col + T)`` of each member) into this rank's
        cache rows ``new_key_rows`` (-1: not here, skipped), as ``_PooledLoop.write_new_keys``: per 128-row tile of
        (slot, token) pairs one strided read, one PE transpose, the fp8 clamp-cast and one indirect scatter."""
        shapes = self.shapes
        HD, T, W = shapes.head_dim, shapes.num_tokens, self.width
        E = self.pool_qw.shape[0] // P_MAX
        es = min(E, P_MAX // T)
        kernel_assert(E % es == 0, f"{E} slots do not split into tiles of {es} ({T} tokens)")
        for i in range(E // es):
            e0, n = i * es, es * T
            k_new = nl.ndarray((P_MAX, HD), dtype=shapes.dtype, buffer=nl.sbuf)
            # Token-major already (row 16 t + j of member e): partition (e, t) = e * T + t reads 16 rows x 8.
            kr, kc = cdp_key_rows(T, HD)
            nisa.dma_copy(
                dst=k_new[0:n, 0:HD].reshape_dim(1, [kr, kc]),
                src=self.pool_qw.ap(pattern=[[kr * W, n], [W, kr], [1, kc]], offset=e0 * P_MAX * W + self.k_col),
                name=f"{self.name}_newkey_rows_tm{i}",
            )
            if shapes.key_fp8:
                k_u8 = nl.ndarray((P_MAX, HD), dtype=nl.uint8, buffer=nl.sbuf)
                nisa.tensor_scalar(
                    dst=k_u8.view(nl.float8_e4m3fn)[0:n, 0:HD], data=k_new[0:n, 0:HD],
                    op0=nl.minimum, operand0=IDX_KEY_FP8_MAX, op1=nl.maximum, operand1=-IDX_KEY_FP8_MAX,
                )
                k_new = k_u8
            rows_sb = nl.ndarray((P_MAX, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.dma_copy(dst=rows_sb[0:n, 0:1], src=new_key_rows[e0 * T : e0 * T + n, 0:1], name=f"{self.name}_newkey_rows{i}")
            nisa.dma_copy(
                dst=cache_rows.ap(pattern=[[HD, n], [1, HD]], offset=0, vector_offset=rows_sb[0:n, 0:1], indirect_dim=0),
                src=k_new[0:n, 0:HD],
                oob_mode=nisa.oob_mode.skip,
                name=f"{self.name}_newkey_write{i}",
            )


@nki.jit
def indexer_kv_score_cdp(
    q: nl.NkiTensor,
    hidden_states: nl.NkiTensor,
    wk: nl.NkiTensor,
    k_norm_weight: nl.NkiTensor,
    k_norm_bias: nl.NkiTensor,
    weights_proj: nl.NkiTensor,
    cos: nl.NkiTensor,
    sin: nl.NkiTensor,
    k_cache: nl.NkiTensor,
    mask_add: nl.NkiTensor,
    num_heads: int,
    rope_dim: int,
    norm_eps: float,
    prologue: tuple | None,
    items: nl.NkiTensor,
    n_items: nl.NkiTensor,
    new_key_rows: nl.NkiTensor,
    shard_degree: int,
    block_size: int,
    store_keys: int,
    gather_pool,
    select_to,
    exchange,
    own_slot: tuple,
    index_topk: int = 2048,
    send: nl.NkiTensor = None,
) -> nl.NkiTensor:
    """The chunk-dealt pool's score for one member: its own chunk 0 and its live foreign regions, each selected by the
    chunk select (see the comment above).

    Args:
        q, hidden_states, wk, k_norm_weight, k_norm_bias, weights_proj, cos, sin, num_heads, rope_dim, norm_eps,
        prologue: as in ``indexer_kv_score``.
        k_cache: this member's fp8 key cache; region ``r`` is chunk ``r`` of owner ``(e - r) % E``. Updated with the
            pool's new keys.
        mask_add: the residency mask in the packed layout, each region's columns built from its owner's base.
        items, n_items, new_key_rows: the host's pool tables.
        shard_degree, block_size: the CP deal.
        store_keys: keys per top-k chunk, the region width.
        gather_pool: ``gather_pool(publish)`` issues the pool all_gather of the publish tile and returns ``pool_qw``.
        select_to: the chunk select, ``select_to(chunk_scores, dst_v, dst_p, off_v, off_p, label_p)``: once per foreign
            item into the send buffer, and once for the own chunk into ``own_slot`` after the all_to_all.
        exchange: ``exchange(send)`` issues the pool all_to_all of the send buffer.
        own_slot: the phase-1 ``(values, positions)`` buffers; slot 0 is the own chunk.
        index_topk: candidates per chunk.
        send: an optional preallocated send buffer.

    Returns:
        ``k_cache``.
    """
    shapes = get_indexer_score_shape(q, hidden_states, wk, weights_proj, k_cache, mask_add, num_heads, rope_dim, 0)
    T, HD = shapes.num_tokens, shapes.head_dim
    n_keys = shapes.num_keys
    cache_rows = k_cache.reshape((n_keys, HD)) if indexer_cache_is_blocked(k_cache) else k_cache
    key_rows = cache_rows.view(nl.bfloat16) if shapes.key_fp8 else cache_rows

    q_t_sbuf, head_w_sbuf, k_new_sbuf = indexer_cdp_front(
        q, hidden_states, wk, k_norm_weight, k_norm_bias, weights_proj, cos, sin, shapes, norm_eps, prologue
    )

    # 1. Publish, and the pool all_gather (the layer's), in flight under the own chunk's first blocks.
    name = unique_name("idx_cdp_k")
    # The loop object before the publish: its pool_qw is bound after the all_gather is issued. The own chunk's first two
    # blocks are preloaded into rings of their own (``preload``), and the foreign loop's first item's row and keys into
    # the loop's own ring (``early_item0``), all under the all_gather.
    loop = _PooledChunkLoop(key_rows, shapes, None, items, store_keys)
    split = min(2, loop.seg_blocks)
    loop.preload(range(0, split), mask_add)
    loop.early_item0()
    publish = indexer_pool_cdp_publish(q_t_sbuf, head_w_sbuf, k_new_sbuf, shapes, name)
    pool_qw = gather_pool(publish)
    loop.bind_pool(pool_qw)
    seg_keys, seg_blocks = loop.seg_keys, loop.seg_blocks
    w_pad = _packed_head_weights(head_w_sbuf, shapes)
    own_scores = nl.ndarray((T, seg_keys), dtype=nl.bfloat16, buffer=layer_scratch_hbm())
    f_scores = nl.ndarray((T, seg_keys), dtype=nl.bfloat16, buffer=layer_scratch_hbm())
    E = pool_qw.shape[0] // P_MAX
    words = cdp_candidate_words(index_topk)
    if send is None:
        send = nl.ndarray((E * T, words), dtype=nl.uint32, buffer=nl.private_hbm)
    kernel_assert(tuple(send.shape) == (E * T, words), f"send is {tuple(send.shape)}, expected ({E * T}, {words})")
    send_v, send_p = send.view(nl.bfloat16), send.view(nl.int32)
    # The dropped-slot sentinel in every part's value region (a dead part reaches its owner as sentinel candidates, as
    # the chunk loop's sentinel-filled phase-1 slots); the loop's selects overwrite the live parts.
    fill = nl.ndarray((E * T, index_topk), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.memset(fill, -MASK_BOUND)
    nisa.dma_copy(
        dst=send_v[0 : E * T, 0:index_topk], src=fill[0 : E * T, 0:index_topk], dge_mode=nisa.dge_mode.hwdge,
        engine=nisa.engine.sync, name=f"{name}_sendfill",
    )

    # 2. Own chunk 0, blocks [0, split): the all_gather runs under them.
    loop.own_blocks(0, split, q_t_sbuf, w_pad, mask_add, own_scores)
    # 3. The foreign regions, selected into the send buffer.
    loop.foreign(n_items, mask_add, f_scores, send_v, send_p, select_to, after=loop.last_evict)
    # 4. The rest of the own chunk preloaded, then the all_to_all once those DMAs have landed (a DMA in flight under a
    # pool collective is much slower), and the own chunk's last blocks under it.
    kernel_assert(seg_blocks - split <= 2, f"the own chunk preloads two blocks: {seg_blocks - split} after the loop")
    loop.preload(range(split, seg_blocks), mask_add)
    exchange(send, after=loop.preload_dmas)
    loop.own_blocks(split, seg_blocks, q_t_sbuf, w_pad, mask_add, own_scores)
    zero = _i32_scalar(0)
    label0 = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(label0, 0.0)
    select_to(own_scores, own_slot[0], own_slot[1], zero, zero, label0)
    # 5. The pool's new keys that land here: excluded from this step's score (the residency cut is the step's base), so
    # traced last, off the selects' path.
    loop.write_new_keys(cache_rows, new_key_rows)
    return k_cache
