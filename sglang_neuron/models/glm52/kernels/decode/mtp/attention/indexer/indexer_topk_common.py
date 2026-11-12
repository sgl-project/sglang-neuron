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
Snake layout helpers shared by both stages of the GLM-5.2 DSA indexer top-k.

``nisa.topk`` describes its tile in terms of SNAKE POSITION -- position ``s`` sits at partition
``s % 16``, free column ``s // 16``. Read the other way round, in candidate order, that same tile is
BLOCKED: partition ``p`` of a row holds candidates ``[p * n_cols, (p + 1) * n_cols)``. Which is the
useful reading, because it makes the load one contiguous reshape per row instead of a strided gather,
and it pushes the whole permutation into an integer inverse applied only to the K returned positions.
"""

import nki.isa as nisa
import nki.language as nl

from ....utils.kernel_assert import kernel_assert
from .indexer_constants import SNAKE_PARTITIONS, IndexerTopkShape


def load_snake(scores: nl.NkiTensor, shapes: IndexerTopkShape) -> nl.NkiTensor:
    """
    Loads each row of ``scores`` ``[T, num_candidates]`` into the 16-partition snake tile.

    Returns ``[T * 16, ceil(num_candidates / 16)]`` bfloat16.

    ONE DMA for the whole tile whenever the row is exact, because the ``T`` per-row reshapes are
    literally one global reshape. Row ``r``, snake partition ``q``, column ``c`` reads flat HBM offset
    ``r*num_candidates + q*n_cols + c``, and when ``num_candidates == 16 * n_cols`` that is
    ``(16r + q)*n_cols + c`` -- exactly the destination's own partition index. Issuing it once matters
    because each ``dma_copy`` pays a fixed semaphore-to-start cost on this target, so ``T`` of them
    cost far more in launch than in bytes (the merge stage moves only 57 K elements).

    A padded row (``num_candidates < 16 * n_cols``) breaks the identity, since the tail of each row is
    then not where the next row's head is, so that case keeps the per-row loop.
    """

    n_cand, n_cols = shapes.num_candidates, shapes.snake_columns

    src_snake = nl.ndarray((shapes.snake_partitions, n_cols), dtype=nl.bfloat16, buffer=nl.sbuf)
    # One DMA per row. Folding all T rows into one ``scores.reshape((T * 16, n_cols))`` is only right when the row
    # stride is exactly num_candidates. On the layer's shared_hbm score buffer it was not, and at 128 ranks the real
    # token's candidates came back as 3 of the true 2048 (invisible to a test that fills every row the same). Per-row
    # addressing holds for any stride, at the cost of one launch a row.
    for row in range(shapes.num_tokens):
        base = row * SNAKE_PARTITIONS
        nisa.dma_copy(
            dst=src_snake[base : base + SNAKE_PARTITIONS, 0:n_cols],
            src=scores[row, 0:n_cand].reshape((SNAKE_PARTITIONS, n_cols)),
        )

    return src_snake


def snake_position_to_candidate(
    idx_snake: nl.NkiTensor,
    n_cols: int,
    partitions: int,
    k_cols: int,
    offset: int = 0,
) -> nl.NkiTensor:
    """
    Inverts the snake to recover each winner's candidate index, plus a constant ``offset``.

    The blocked load put candidate ``v`` at snake position ``s = 16 * (v % n_cols) + (v // n_cols)``,
    so ``v = (s % 16) * n_cols + (s // 16)``.

    Done with an integer shift and mask: there is no ``floor`` activation on this hardware, and a shift
    is exact by construction where a float divide would need an argument about representable range.
    """

    idx_i32 = nl.ndarray((partitions, k_cols), dtype=nl.int32, buffer=nl.sbuf)
    # SBUF->SBUF DMA, NOT nisa.tensor_copy. ``nisa.topk`` writes ``idx_snake`` on the GpSimd engine while
    # the bitvec ``tensor_scalar``s below are PINNED to Vector, so a COMPUTE read here is an unordered
    # cross-engine read of a GpSimd-written tile: it returns stale SBUF, the inverse maps it to an
    # enormous candidate index, and the caller's indirect position gather then reads arbitrary HBM. That
    # was observed on device as selections containing memset fill (0x01010101) and ASCII rather than
    # positions, while ``cand_values`` stayed correct because its only consumer is a DMA. A DMA is
    # semaphore-ordered against the GpSimd write and also performs the uint32->int32 conversion. Same
    # hazard and same fix as the kernel library's GpSimd top-k kernel, which never
    # compute-reads a topk output at all.
    nisa.dma_copy(dst=idx_i32[0:partitions, 0:k_cols], src=idx_snake[0:partitions, 0:k_cols])

    hi_i32 = nl.ndarray((partitions, k_cols), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=hi_i32[0:partitions, 0:k_cols],
        data=idx_i32[0:partitions, 0:k_cols],
        op0=nl.right_shift,
        operand0=4,
    )
    lo_i32 = nl.ndarray((partitions, k_cols), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=lo_i32[0:partitions, 0:k_cols],
        data=idx_i32[0:partitions, 0:k_cols],
        op0=nl.bitwise_and,
        operand0=SNAKE_PARTITIONS - 1,
    )

    out_i32 = nl.ndarray((partitions, k_cols), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=out_i32[0:partitions, 0:k_cols],
        data=lo_i32[0:partitions, 0:k_cols],
        op0=nl.multiply,
        operand0=n_cols,
        op1=nl.add,
        operand1=offset,
    )
    nisa.tensor_tensor(
        dst=out_i32[0:partitions, 0:k_cols],
        data1=out_i32[0:partitions, 0:k_cols],
        data2=hi_i32[0:partitions, 0:k_cols],
        op=nl.add,
    )

    return out_i32


def deal_positions(pos: nl.NkiTensor, parts: int, cols: int, shard_degree: int, block_size: int) -> None:
    """In place: a shard's LOCAL key index -> its position under an INTERLEAVED deal, minus the rank term.

    Local key ``v`` sits in global block ``shard_degree * (v >> log2(block_size)) + rank``. This writes
    ``shard_degree * block_size * (v >> shift) + (v & (block_size - 1))``; the caller adds the rank's
    ``rank * block_size`` as its runtime ``key_offset``, because one NEFF serves every rank. See
    ``indexer_global_positions_np`` for the host twin. ``pos`` is int32 ``[parts, cols]``.
    """
    kernel_assert(
        block_size > 0 and block_size & (block_size - 1) == 0, f"block_size={block_size} must be a power of two"
    )
    shift = block_size.bit_length() - 1
    blk = nl.ndarray((parts, cols), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(blk, pos[0:parts, 0:cols], op0=nl.right_shift, operand0=shift)
    intra = nl.ndarray((parts, cols), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(intra, pos[0:parts, 0:cols], op0=nl.bitwise_and, operand0=block_size - 1)
    nisa.scalar_tensor_tensor(
        dst=pos[0:parts, 0:cols],
        data=blk,
        op0=nl.multiply,
        operand0=float(shard_degree * block_size),
        op1=nl.add,
        operand1=intra,
    )
