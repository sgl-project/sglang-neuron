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
GLM-5.2 DSA indexer stage-1 LOCAL for a shard wider than one select may take (``IDX_TOPK_MAX_CALL_N`` keys).

The shard is ranked in chunks, a builtin ``nisa.topk`` each (``indexer_topk_local_chunk``) carrying each winner's
global position, and the key group's wide rounds (``sglang_neuron.models.glm52.kernels.topk_select``) then keep this rank's share
of the global top-k among the chunk winners.
"""

import nki
import nki.isa as nisa
import nki.language as nl

from sglang_neuron.models.glm52.kernels import topk_select

from ...utils.kernel_assert import kernel_assert
from .indexer_constants import (
    IDX_TOPK_MAX_CALL_N,
    MASK_BOUND,
    P_MAX,
    SNAKE_PARTITIONS,
    get_indexer_topk_shape,
    local_topk_chunked,
)
from .indexer_topk_common import load_snake
from .indexer_topk_local import indexer_topk_local_chunk


def _pick_local_chunks(n_keys: int, topk: int) -> tuple[int, int]:
    """Chunk count C and per-chunk width W for a local key range wider than one select may take.

    C is the smallest power of two such that ``W = n_keys / C`` divides evenly, is at most
    ``IDX_TOPK_MAX_CALL_N`` and a multiple of ``SNAKE_PARTITIONS`` (the snake load's row/partition constraint),
    AND ``C * topk <= IDX_TOPK_MAX_CALL_N`` so phase 2 over the ``C*topk`` candidates is itself one select of a
    safe width. 512k/CP4 (n_keys = 131072, topk = 2048) gets C = 4, W = 32768 and a phase 2 over
    8192; 1M/CP4 (262144) gets C = 8. A shard that is not a power-of-two multiple of 32768 gets narrower
    chunks: 98304 gets C = 4, W = 24576, where ``W < TOPK_MAX_N`` alone would have given C = 2, W = 49152.
    """
    # C = 1, then every 2**k <= n_keys, in ascending order; ``max(n_keys, 1)`` keeps C = 1 a candidate at n_keys = 0.
    for k in range(max(n_keys, 1).bit_length()):
        C = 1 << k
        W = n_keys // C
        if (
            n_keys % C == 0
            and W <= IDX_TOPK_MAX_CALL_N
            and W % SNAKE_PARTITIONS == 0
            and C * topk <= IDX_TOPK_MAX_CALL_N
        ):
            return C, W
    raise ValueError(
        f"no legal local chunking for n_keys={n_keys}, topk={topk}: needs some C = 2**k with "
        f"n_keys % C == 0, (n_keys // C) <= {IDX_TOPK_MAX_CALL_N}, (n_keys // C) % {SNAKE_PARTITIONS} == 0 and "
        f"C * topk <= {IDX_TOPK_MAX_CALL_N}. The supported ceiling is 524288 keys per rank at topk=2048 (C=16, "
        f"W=32768); no wider shard has a solution"
    )


def _wide_publish(cand_values, cand_positions, topk, C, W, key_offset, shard_degree, block_size, wide_gather):
    """The chunked shard's phase 2, the key group's wide rounds (``topk_select``): this rank's share of the GLOBAL top-k
    in the packed tile's position words (-1 elsewhere; the value words zero). A shard-local key is 18 bits at 1M, more
    than one bf16 code carries, so a merge cannot compact the shards' own top-k; instead the key group trades its top-k
    scores first and each rank keeps exactly its share."""
    shapes = get_indexer_topk_shape(cand_values, topk)
    T, n_cand = shapes.num_tokens, shapes.num_candidates
    n_cols, k_cols = shapes.snake_columns, topk // SNAKE_PARTITIONS
    P = shapes.snake_partitions
    kernel_assert(P == P_MAX and n_cand == C * topk and SNAKE_PARTITIONS % C == 0,
                  f"the wide select takes {C} whole chunks of {topk} on the full {P_MAX}-partition tile")
    src = load_snake(cand_values, shapes)
    pos = nl.ndarray((P, n_cols), dtype=nl.int32, buffer=nl.sbuf)
    for row in range(T):
        base = row * SNAKE_PARTITIONS
        nisa.dma_copy(dst=pos[base : base + SNAKE_PARTITIONS, 0:n_cols],
                      src=cand_positions[row, 0:n_cand].reshape((SNAKE_PARTITIONS, n_cols)))
    # round 1: this rank's top-k scores, traded over the key group
    val1 = topk_select.wide_local_values(src, n_cand, topk)
    val1_hbm = nl.ndarray((P, k_cols), dtype=nl.bfloat16, buffer=nl.private_hbm)
    nisa.dma_copy(dst=val1_hbm, src=val1)
    all_vals = wide_gather(val1_hbm, "cand_vals1_all")
    gathered = nl.ndarray((P, shard_degree * k_cols), dtype=nl.bfloat16, buffer=nl.sbuf)
    for s in range(shard_degree):
        nisa.dma_copy(dst=gathered[0:P, s * k_cols : (s + 1) * k_cols], src=all_vals[s * P : (s + 1) * P, 0:k_cols])
    # this rank's key rank and rank term, from key_offset = rank * block_size
    rank_term = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=rank_term[0:P, 0:1], src=key_offset[0:1, 0:1].broadcast(0, P))
    own = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=own, data=rank_term, op0=nl.multiply, operand0=1.0 / block_size)
    tau, room = topk_select.wide_global_cut(gathered, topk, shard_degree, own)
    member = topk_select.wide_members(src, tau, room)
    out = topk_select.wide_publish(src, pos, member, n_cand, topk, C, W, shard_degree=shard_degree,
                                   block_size=block_size, rank_term=rank_term, live_floor=-0.5 * MASK_BOUND)
    packed = nl.ndarray((P, 2 * k_cols), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.memset(dst=packed[0:P, 0:k_cols], value=0)
    nisa.tensor_copy(dst=packed[0:P, k_cols : 2 * k_cols].view(nl.int32), src=out[0:P, 0:k_cols])
    return packed, P, k_cols


def _chunk_phase1(
    index_scores: nl.NkiTensor,
    topk: int,
    key_base: int,
    key_offset: nl.NkiTensor,
    C: int,
    W: int,
    T: int,
    shard_degree: int,
    block_size: int,
) -> tuple[nl.NkiTensor, nl.NkiTensor]:
    """Phase 1: each of ``C`` chunks' top-``topk`` candidates, concatenated into ``(cand_values_all [T, C*topk] bf16,
    cand_positions_all [T, C*topk] int32)``. A candidate's GLOBAL position is the deal of ``key_base + c*W + local``
    (``deal_positions``) plus the runtime ``key_offset``."""
    cand_values_all, cand_positions_all = chunk_phase1_alloc(T, C, topk)
    for c in range(C):
        chunk_phase1_one(
            index_scores, c, W, topk, key_base, key_offset, shard_degree, block_size, cand_values_all, cand_positions_all
        )
    return cand_values_all, cand_positions_all


def chunk_phase1_alloc(T: int, C: int, topk: int, sentinel: bool = False) -> tuple[nl.NkiTensor, nl.NkiTensor]:
    """The phase-1 candidate buffers ``(values [T, C*topk] bf16, positions [T, C*topk] int32)`` @ private_hbm.

    ``sentinel`` fills the values with the dropped-slot sentinel, for a caller that may skip a chunk's select (the
    chunk loop past the live length): its slots are then dead to the wide rounds, as a select's own dead slots are, whatever
    their positions hold."""
    cand_values_all = nl.ndarray((T, C * topk), dtype=nl.bfloat16, buffer=nl.private_hbm)
    cand_positions_all = nl.ndarray((T, C * topk), dtype=nl.int32, buffer=nl.private_hbm)
    if sentinel:
        fill = nl.ndarray((T, topk), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.memset(fill, -MASK_BOUND)
        for c in range(C):
            nisa.dma_copy(dst=cand_values_all[0:T, c * topk : (c + 1) * topk], src=fill)
    return cand_values_all, cand_positions_all


def chunk_phase1_one(
    index_scores: nl.NkiTensor,
    c: int,
    W: int,
    topk: int,
    key_base: int,
    key_offset: nl.NkiTensor,
    shard_degree: int,
    block_size: int,
    cand_values_all: nl.NkiTensor,
    cand_positions_all: nl.NkiTensor,
) -> None:
    """Chunk ``c`` of :func:`_chunk_phase1`: its top-``topk`` (builtin ``nisa.topk``, ``indexer_topk_local_chunk``)
    into slot ``c`` of the candidate buffers.
    """
    T = index_scores.shape[0]
    k_cols = topk // SNAKE_PARTITIONS
    chunk = index_scores[:, c * W : (c + 1) * W]
    cv, cp = indexer_topk_local_chunk(chunk, topk, key_base + c * W, key_offset, shard_degree, block_size)
    nisa.dma_copy(
        dst=cand_values_all[:, c * topk : (c + 1) * topk].reshape((T, SNAKE_PARTITIONS, k_cols)),
        src=cv,
    )
    nisa.dma_copy(
        dst=cand_positions_all[:, c * topk : (c + 1) * topk].reshape((T, SNAKE_PARTITIONS, k_cols)),
        src=cp,
    )


@nki.jit
def indexer_topk_local_carry_index_packed(
    index_scores: nl.NkiTensor,
    topk: int,
    key_offset: nl.NkiTensor,
    shard_degree: int,
    block_size: int,
    phase1: tuple = None,
    wide_gather=None,
) -> nl.NkiTensor:
    """This rank's share of the global top-k of a key shard wider than one select may take, as a packed tile.

    The shard is ranked in ``C`` chunks of ``W`` keys (``_pick_local_chunks``: 512k/CP4, 131072 keys, is C=4 x 32768;
    1M/CP4, 262144 keys, is C=8 x 32768), then the key group's wide rounds (``_wide_publish``) keep exactly this
    rank's share of the global top-k among the chunk winners. The layer takes the tile as the rank's own selection
    (``indexer_topk_own_selection_packed``); the MTP draft's merge takes the key group's tiles with ``wide=True``.

    Args:
        index_scores: [T, num_keys] bfloat16 @ HBM, token-major (as ``indexer_kv_score`` emits it), ``num_keys``
            above ``IDX_TOPK_MAX_CALL_N``.
        topk: candidates to publish (the FULL global budget).
        key_offset: [1, 1] int32 @ HBM, this rank's rank term ``rank * block_size`` of the deal.
        shard_degree: Degree the keys are DEALT over, above 1 (see ``deal_positions``).
        block_size: The deal unit in keys, a power of two.
        phase1: The ``chunk_phase1_alloc`` buffers, already filled by the caller's chunk selects (the chunk-dealt
            pool, the MTP draft's chunk loop), or None to run phase 1 here.
        wide_gather: The key group's all_gather of a ``[T * 16, cols]`` HBM tile,
            ``(tile, name) -> [degree * T * 16, cols]``, which the wide rounds trade the ranks' top-k scores over.

    Returns:
        [T * 16, 2 * (topk // 16)] uint32 @ private_hbm: row ``t*16 + q`` holds token ``t``'s snake partition ``q``;
        words ``[0, topk//16)`` zero, words ``[topk//16, 2*topk//16)`` this rank's share's global positions, -1 past
        the share.
    """
    T = index_scores.shape[0]
    n_keys = index_scores.shape[1]
    if not local_topk_chunked(n_keys):
        raise ValueError(
            f"{n_keys} keys a rank fit one select: indexer_topk_local_packed ranks them; the carry-index local "
            f"takes only a chunked shard"
        )
    if wide_gather is None or shard_degree < 2:
        raise ValueError(
            "a chunked shard publishes its share of the global top-k, which needs dealt keys (shard_degree > 1) and "
            "the key group's gather (wide_gather)"
        )
    C, W = _pick_local_chunks(n_keys, topk)
    if phase1 is None:
        phase1 = _chunk_phase1(index_scores, topk, 0, key_offset, C, W, T, shard_degree, block_size)
    cand_values_all, cand_positions_all = phase1
    packed, P, k_cols = _wide_publish(cand_values_all, cand_positions_all, topk, C, W, key_offset,
                                      shard_degree, block_size, wide_gather)

    # private_hbm: the layer all_gathers this directly, and at LNC=1 the runtime rejects a collective
    # whose sources are shared_hbm. One order-preserving DMA of the whole tile.
    packed_hbm = nl.ndarray((P, 2 * k_cols), dtype=nl.uint32, buffer=nl.private_hbm)
    nisa.dma_copy(dst=packed_hbm, src=packed[0:P, 0 : 2 * k_cols])
    return packed_hbm
