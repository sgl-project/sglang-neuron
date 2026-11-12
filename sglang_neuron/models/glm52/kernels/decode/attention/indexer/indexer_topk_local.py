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

"""
GLM-5.2 DSA indexer: stage 1 of the context-parallel top-k.

Each rank selects the top ``topk`` of its OWN key quarter and publishes them as candidates with
GLOBAL positions. The caller gathers the four ranks' candidate lists; stage 2 merges them.

The two-stage merge is EXACT, not approximate: if a key is in the global top-K then fewer than K keys
globally exceed it, hence fewer than K on its own shard, so it survives its shard's local top-K. The
local count must be the FULL K -- with a smaller local budget, random data at four shards already
loses true winners.
"""


import nki
import nki.isa as nisa
import nki.language as nl

from sglang_neuron.models.glm52.kernels import topk_select

from ...utils.kernel_assert import kernel_assert
from .indexer_constants import (
    IDX_SELECT_2Q,
    MASK_BOUND,
    SNAKE_PARTITIONS,
    IndexerTopkShape,
    assert_topk_call_n,
    get_indexer_topk_shape,
)
from .indexer_topk_common import deal_positions, load_snake, snake_position_to_candidate

@nki.jit
def indexer_topk_local_packed(
    index_scores: nl.NkiTensor,
    topk: int,
    key_offset: nl.NkiTensor,
    shard_degree: int,
    block_size: int,
    src_snake: nl.NkiTensor = None,
) -> nl.NkiTensor:
    """The local select of a dealt shard, emitting the packed merge tile straight from SBUF.

    One ``nisa.topk`` over the whole shard, so ``num_keys`` is at most ``IDX_TOPK_MAX_CALL_N`` and ``topk`` a multiple
    of 32; wider shards go through ``indexer_topk_local_carry_index_packed``. ``key_offset`` ([1, 1] int32, the rank term
    of the deal) is a tensor because it differs per rank under one compiled NEFF, and is added to the winners after the
    select. ``src_snake`` passes scores already in the snake (``indexer_kv_score(snake_out=True)``).

    Returns:
        [T * 16, 2 * (topk // 16)] uint32, the tile ``indexer_topk_merge_carry_index_packed`` reads: per snake row the
        selected bf16 scores, their payload codes (``topk_select.encode_local_codes``; ``DEAD_KEY`` for a dead slot) and
        their global positions.
    """
    if shard_degree < 2:
        raise ValueError(
            f"indexer_topk_local_packed needs the keys dealt over the key group (shard_degree > 1 and the deal "
            f"block_size): the packed merge rebuilds each position from its shard's deal; got "
            f"shard_degree={shard_degree}"
        )
    if src_snake is not None:
        shapes = IndexerTopkShape(
            num_tokens=src_snake.shape[0] // SNAKE_PARTITIONS,
            num_candidates=src_snake.shape[1] * SNAKE_PARTITIONS,
            topk=topk,
            dtype=src_snake.dtype,
        )
    else:
        shapes = get_indexer_topk_shape(index_scores, topk)
    n_keys = shapes.num_candidates
    n_cols, k_cols = shapes.snake_columns, topk // SNAKE_PARTITIONS
    P = shapes.snake_partitions
    kernel_assert(
        k_cols % 2 == 0,
        f"the packed tile stores two bf16 scores per uint32 word, so k_cols must be even; got {k_cols}",
    )

    if src_snake is None:
        src_snake = load_snake(index_scores, shapes)

    val_snake = nl.ndarray((P, topk), dtype=nl.bfloat16, buffer=nl.sbuf)
    idx_snake = nl.ndarray((P, topk), dtype=nl.uint32, buffer=nl.sbuf)
    assert_topk_call_n(n_keys, "indexer_topk_local_packed")
    nisa.topk(val_dst=val_snake[:, 0:topk], idx_dst=idx_snake[:, 0:topk], src=src_snake, n=n_keys)

    # The value tile is read through a DMA, never a compute op. That is the same ordering the index tile
    # gets in snake_position_to_candidate, and it is the form that was gated at 128 ranks (2097152/2097152).
    val_evac = nl.ndarray((P, k_cols), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=val_evac[0:P, 0:k_cols], src=val_snake[0:P, 0:k_cols])

    # The label is the local index mapped through the deal, plus the rank term, in the same three steps as the dealt
    # arm of indexer_topk_local_chunk: the local index, deal_positions on the k_cols winners, then key_offset as an
    # fp32 per-partition operand on int32 data. The payload code is the local index, before the deal.
    pos_i32 = snake_position_to_candidate(idx_snake, n_cols, P, k_cols, offset=0)
    codes = nl.ndarray((P, k_cols), dtype=nl.uint16, buffer=nl.sbuf)
    topk_select.encode_local_codes(pos_i32, val_evac, codes, P, k_cols, live_floor=-0.5 * MASK_BOUND)
    deal_positions(pos_i32, P, k_cols, shard_degree, block_size)
    off_f = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=off_f[0:P, 0:1], src=key_offset[0:1, 0:1].broadcast(0, P))
    nisa.tensor_scalar(
        dst=pos_i32[0:P, 0:k_cols],
        data=pos_i32[0:P, 0:k_cols],
        op0=nl.add,
        operand0=off_f[0:P, 0:1],
    )

    # private_hbm: the layer all_gathers this tile directly, and at LNC=1 the runtime rejects a collective
    # whose sources are shared_hbm.
    packed = nl.ndarray((P, 2 * k_cols), dtype=nl.uint32, buffer=nl.private_hbm)
    nisa.dma_copy(dst=packed[:, nl.ds(0, k_cols // 2)].view(nl.bfloat16), src=val_evac[0:P, 0:k_cols])
    nisa.dma_copy(dst=packed[:, nl.ds(k_cols, k_cols)].view(nl.int32), src=pos_i32[0:P, 0:k_cols])
    nisa.dma_copy(dst=packed[:, nl.ds(k_cols // 2, k_cols // 2)].view(nl.uint16), src=codes[0:P, 0:k_cols])
    return packed


def indexer_topk_local_chunk(
    index_scores: nl.NkiTensor,
    topk: int,
    key_base: int,
    key_offset: nl.NkiTensor,
    shard_degree: int = 1,
    block_size: int = 0,
    extra_offset: nl.NkiTensor = None,
    vector_inverse: bool = False,
    src_snake: nl.NkiTensor = None,
) -> tuple[nl.NkiTensor, nl.NkiTensor]:
    """The builtin select over one sub-cap chunk, traced into a chunk loop (phase 1 of a shard wider than one select).

    Positions are ``key_offset + key_base + local``: ``key_base``, the chunk's base, folds into the position arithmetic
    for free, ``key_offset`` is the per-rank tensor, and ``extra_offset`` ([P, 1] fp32 SBUF) carries a runtime chunk term
    when the chunk index is a loop register. The scores are read from the select's own output by slice, with no
    re-gather. ``src_snake`` passes a snake the caller already loaded.

    Returns:
        ``(cand_values, cand_positions)``, [T, 16, topk // 16] each in private HBM, in ``nisa.topk``'s snake layout
        (-1 for a masked slot).
        Dead slots keep the score sentinel; the wide rounds treat a slot at the liveness floor as dead.
    """
    shapes = get_indexer_topk_shape(index_scores, topk)
    T, n_keys = shapes.num_tokens, shapes.num_candidates
    n_cols, k_cols = shapes.snake_columns, topk // SNAKE_PARTITIONS
    P = shapes.snake_partitions

    if src_snake is None:
        src_snake = load_snake(index_scores, shapes, two_queues=IDX_SELECT_2Q)

    val_snake = nl.ndarray((P, topk), dtype=nl.bfloat16, buffer=nl.sbuf)
    idx_snake = nl.ndarray((P, topk), dtype=nl.uint32, buffer=nl.sbuf)
    assert_topk_call_n(n_keys, "indexer_topk_local_chunk")
    nisa.topk(val_dst=val_snake[:, 0:topk], idx_dst=idx_snake[:, 0:topk], src=src_snake, n=n_keys)

    # The value tile is read through a DMA, never a compute op -- the same ordering the index tile gets
    # in snake_position_to_candidate, and the form that was gated at 128 ranks (2097152/2097152).
    val_evac = nl.ndarray((P, k_cols), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=val_evac[0:P, 0:k_cols], src=val_snake[0:P, 0:k_cols])

    # key_base rides the inverse-snake's spare op1 slot, so the chunk base is free.
    pos_i32 = snake_position_to_candidate(
        idx_snake, n_cols, P, k_cols, offset=key_base, engine=nisa.engine.vector if vector_inverse else None
    )
    if shard_degree > 1:
        # INTERLEAVED: pos_i32 is the shard-local index (chunk base included), so deal it before the rank
        # term is added -- on the k_cols winners, not the chunk.
        deal_positions(pos_i32, P, k_cols, shard_degree, block_size)
    """
    The per-rank shard base. int32 data and int32 dst with an fp32 per-partition operand: the DMA
    converts the int32 scalar, the engine computes in fp32, and a position is far inside fp32's
    exactly-representable integer range. The offset broadcasts per PARTITION, so it is loaded stride-0
    down every one.
    """
    off_f = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=off_f[0:P, 0:1], src=key_offset[0:1, 0:1].broadcast(0, P))
    nisa.tensor_scalar(
        dst=pos_i32[0:P, 0:k_cols],
        data=pos_i32[0:P, 0:k_cols],
        op0=nl.add,
        operand0=off_f[0:P, 0:1],
    )
    if extra_offset is not None:
        nisa.tensor_scalar(
            dst=pos_i32[0:P, 0:k_cols],
            data=pos_i32[0:P, 0:k_cols],
            op0=nl.add,
            operand0=extra_offset[0:P, 0:1],
        )

    """
    De-snake into [T, 16, k_cols] (partition p -> token p // 16, snake partition p % 16), the blocked
    convention the local stage publishes. P == T * 16 and both destinations are freshly allocated and
    contiguous, so collapsing the leading two axes is stride-exact. Order-preserving; NO gather.
    private_hbm: the layer AllGathers these, and at LNC=1 the runtime rejects a collective whose
    sources are shared_hbm.
    """
    cand_values = nl.ndarray((T, SNAKE_PARTITIONS, k_cols), dtype=nl.bfloat16, buffer=nl.private_hbm)
    cand_positions = nl.ndarray((T, SNAKE_PARTITIONS, k_cols), dtype=nl.int32, buffer=nl.private_hbm)
    nisa.dma_copy(
        dst=cand_values.reshape((T * SNAKE_PARTITIONS, k_cols)),
        src=val_evac[0 : T * SNAKE_PARTITIONS, 0:k_cols],
    )
    nisa.dma_copy(
        dst=cand_positions.reshape((T * SNAKE_PARTITIONS, k_cols)),
        src=pos_i32[0 : T * SNAKE_PARTITIONS, 0:k_cols],
    )
    return cand_values, cand_positions

