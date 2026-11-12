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

"""Token-pair layout of the Kimi-K3 FP8 MLA cache pages.

A pair-layout page holds the 128 rows of a page as 64 pair rows. Pair row ``j`` is ``[576, 2]`` FP8: element
``(j, d, k)`` is feature ``d`` of row ``2 j + k``. So one 2-byte element holds feature ``d`` of two adjacent
rows, and a 2-byte transpose of a pair row block gives the keys feature-major in row order. The bytes are the
same FP8 values as in a row-major page; only their order changes.

This module has the kernel side of the layout; the host layout functions are not in this package.
"""

import nki.isa as nisa
import nki.language as nl

from .utils.kernel_assert import kernel_assert

PAGE_ROWS = 128
"""Rows of one kernel page (one CP stripe of 128 positions)."""
CACHE_WIDTH = 576
"""FP8 values of one cache row: the normalized latent (512), then k_pe (64)."""
PAIR_ROWS = PAGE_ROWS // 2
"""Pair rows of one kernel page."""
PAIR_WIDTH = 2 * CACHE_WIDTH
"""Bytes of one pair row."""
_BYTE = 256
_LOW_BYTE = 255


def pair_rows_view(kv_pool):
    """Return the ``[P * 64, 576]`` uint16 view of a pair-layout FP8 pool ``[P, 73728]``: element ``(i, d)`` holds
    feature ``d`` of rows ``2 i`` and ``2 i + 1`` (low byte, high byte)."""
    kernel_assert(kv_pool.shape[1] == PAGE_ROWS * CACHE_WIDTH, f"a page holds {PAGE_ROWS * CACHE_WIDTH} bytes")
    kernel_assert(kv_pool.dtype != nl.bfloat16, "give the FP8 pool, not its bf16 view")
    return kv_pool.view(nl.uint16).reshape((kv_pool.shape[0] * PAIR_ROWS, CACHE_WIDTH))


def write_pair_rows(kv_pool, rows, write_row, seq_rows, tag, sb):
    """Write each FP8 row of ``rows`` into its byte lane of a pair-layout pool, and keep the other lane.

    Row ``r = j * seq_rows + b`` is position ``j`` of sequence ``b``; the positions of one sequence follow each
    other. A pool row ``s`` (``write_row``) is lane ``s & 1`` of pair row ``s >> 1``. The other lane comes from row
    ``r - seq_rows`` or ``r + seq_rows`` when that row holds the other position of the pair, else from the pool.
    So the rows of one pair store the same bytes, and no store depends on another store. The merge runs on
    integers, so every byte stays exact. All rows of the null pool row 0 store to pair row 0, which no read
    trusts.

    Args:
        kv_pool: ``[P, 73728]`` pair-layout FP8 HBM pool.
        rows: ``[R, 576]`` FP8 SBUF rows.
        write_row: ``[R, 1]`` INT32 SBUF pool row of each row: ``page * 128 + offset``, 0 for no write.
        seq_rows: sequences per position (``B``); ``R == seq_rows`` is one position per sequence.
        tag: unique name prefix of the DMAs.
        sb: the SBUF allocator.

    Returns:
        The store DMAs.
    """
    count = rows.shape[0]
    kernel_assert(rows.shape[1] == CACHE_WIDTH, f"rows must be [R, {CACHE_WIDTH}]")
    kernel_assert(count % seq_rows == 0, "the rows must hold whole positions of seq_rows sequences")
    int32 = nl.int32
    fp32 = nl.float32
    hwdge = nisa.dge_mode.hwdge
    sync = nisa.engine.sync
    pairs = pair_rows_view(kv_pool)
    width = CACHE_WIDTH
    pair = sb.tile((count, 1), int32)
    nisa.tensor_scalar(dst=pair, data=write_row, op0=nl.right_shift, operand0=1)
    lane_i = sb.tile((count, 1), int32)
    nisa.tensor_scalar(dst=lane_i, data=write_row, op0=nl.bitwise_and, operand0=1)
    # A per-partition scalar operand of an arithmetic op must be FP32; 0 and 1 are exact.
    lane = sb.tile((count, 1), fp32)
    nisa.tensor_copy(dst=lane, src=lane_i)
    stored = sb.tile((count, width), nl.uint16)
    for r in range(count):
        nisa.dma_copy(
            dst=stored[r : r + 1, 0:width],
            src=pairs.ap(
                pattern=[[width, 1], [1, width]], offset=0, scalar_offset=pair[r : r + 1, 0:1], indirect_dim=0
            ),
            dge_mode=hwdge,
            engine=sync,
            name=f"{tag}_pair_load_{r}",
        )
    old = sb.tile((count, width), int32)
    nisa.tensor_copy(dst=old, src=stored)
    new = sb.tile((count, width), int32)
    nisa.tensor_copy(dst=new, src=rows.view(nl.uint8))
    # The stored byte of the other lane: the high byte for lane 0, the low byte for lane 1.
    high = sb.tile((count, width), int32)
    nisa.tensor_scalar(dst=high, data=old, op0=nl.right_shift, operand0=8)
    low = sb.tile((count, width), int32)
    nisa.tensor_scalar(dst=low, data=old, op0=nl.bitwise_and, operand0=_LOW_BYTE)
    nisa.tensor_tensor(dst=low, data1=low, data2=high, op=nl.subtract)
    other = sb.tile((count, width), int32)
    nisa.scalar_tensor_tensor(dst=other, data=low, op0=nl.multiply, operand0=lane, op1=nl.add, operand1=high)
    if count > seq_rows:
        _take_partner(other, new, lane, seq_rows, tag, sb)
    # pair = new * 256**lane + other * 256**(1 - lane)
    new_scale = sb.tile((count, 1), fp32)
    nisa.tensor_scalar(dst=new_scale, data=lane, op0=nl.multiply, operand0=_BYTE - 1, op1=nl.add, operand1=1)
    other_scale = sb.tile((count, 1), fp32)
    nisa.tensor_scalar(dst=other_scale, data=lane, op0=nl.multiply, operand0=1 - _BYTE, op1=nl.add, operand1=_BYTE)
    nisa.tensor_scalar(dst=new, data=new, op0=nl.multiply, operand0=new_scale)
    nisa.scalar_tensor_tensor(dst=old, data=other, op0=nl.multiply, operand0=other_scale, op1=nl.add, operand1=new)
    merged = sb.tile((count, width), nl.uint16)
    nisa.tensor_copy(dst=merged, src=old)
    stores = []
    for r in range(count):
        stores.append(
            nisa.dma_copy(
                dst=pairs.ap(
                    pattern=[[width, 1], [1, width]], offset=0, scalar_offset=pair[r : r + 1, 0:1], indirect_dim=0
                ),
                src=merged[r : r + 1, 0:width],
                dge_mode=hwdge,
                engine=sync,
                name=f"{tag}_pair_store_{r}",
            )
        )
    return stores


def _take_partner(other, new, lane, seq_rows, tag, sb):
    """Replace ``other`` with the new byte of the partner row where the partner row is in the batch.

    Lane 1 at position ``j > 0`` pairs with row ``r - seq_rows``; lane 0 at ``j < Q - 1`` with ``r + seq_rows``.
    """
    count, width = new.shape
    int32 = nl.int32
    hwdge = nisa.dge_mode.hwdge
    sync = nisa.engine.sync
    tail = count - seq_rows
    previous = sb.tile((count, width), int32)
    following = sb.tile((count, width), int32)
    nisa.memset(dst=previous, value=0)
    nisa.memset(dst=following, value=0)
    nisa.dma_copy(dst=previous[seq_rows:count], src=new[0:tail], dge_mode=hwdge, engine=sync, name=f"{tag}_pair_prev")
    nisa.dma_copy(dst=following[0:tail], src=new[seq_rows:count], dge_mode=hwdge, engine=sync, name=f"{tag}_pair_next")
    index = sb.tile((count, 1), int32)
    nisa.iota(dst=index, pattern=[[0, 1]], offset=0, channel_multiplier=1)
    has_previous = sb.tile((count, 1), nl.float32)
    nisa.tensor_scalar(dst=has_previous, data=index, op0=nl.greater_equal, operand0=seq_rows)
    nisa.tensor_tensor(dst=has_previous, data1=has_previous, data2=lane, op=nl.multiply)
    has_following = sb.tile((count, 1), nl.float32)
    nisa.tensor_scalar(dst=has_following, data=index, op0=nl.less, operand0=tail)
    not_lane = sb.tile((count, 1), nl.float32)
    nisa.tensor_scalar(dst=not_lane, data=lane, op0=nl.multiply, operand0=-1, op1=nl.add, operand1=1)
    nisa.tensor_tensor(dst=has_following, data1=has_following, data2=not_lane, op=nl.multiply)
    # At most one of the two flags is 1: other += flag_p * (previous - other) + flag_f * (following - other).
    for flag, partner in ((has_previous, previous), (has_following, following)):
        nisa.tensor_tensor(dst=partner, data1=partner, data2=other, op=nl.subtract)
        nisa.scalar_tensor_tensor(dst=other, data=partner, op0=nl.multiply, operand0=flag, op1=nl.add, operand1=other)


def pair_rows_tile(sb, blocks: int, dtype):
    """Return the SBUF pair-row tile of ``blocks`` kernel pages: ``[128, ceil(blocks / 2), 1152]``.

    Column ``q`` holds pages ``2 q`` (partitions 0-63) and ``2 q + 1`` (partitions 64-127). Partition ``p`` of
    column ``q`` holds keys ``256 q + 2 p`` and ``256 q + 2 p + 1`` of the group.
    """
    return sb.tile((2 * PAIR_ROWS, (blocks + 1) // 2, PAIR_WIDTH), dtype)


def load_pair_page(kv_pool, pairs, block: int, entry, tag: str):
    """Load kernel page ``entry`` (``[1, 1]`` INT32 SBUF page id) as block ``block`` of the pair-row tile ``pairs``."""
    half = (block % 2) * PAIR_ROWS
    return nisa.dma_copy(
        dst=pairs[half : half + PAIR_ROWS, block // 2, 0:PAIR_WIDTH],
        src=kv_pool.ap(
            pattern=[[PAIR_WIDTH, PAIR_ROWS], [1, PAIR_WIDTH]], offset=0, scalar_offset=entry, indirect_dim=0
        ),
        dge_mode=nisa.dge_mode.hwdge,
        engine=nisa.engine.sync,
        name=tag,
    )


def _pair_columns(blocks: int):
    """Yield ``(column, partitions)`` of the pair-row tile of ``blocks`` pages."""
    for q in range((blocks + 1) // 2):
        yield q, 2 * PAIR_ROWS if 2 * q + 1 < blocks else PAIR_ROWS


def pair_key_tiles(pairs, blocks: int, latent_keys, rope_keys, latent: int = 512, rope: int = 64):
    """Write the feature-major BF16 keys of ``blocks`` pages from the pair rows, as ``_transpose_group`` does.

    A transpose-mode 2-byte TensorE transpose of ``[partitions, 128 features]`` gives the FP8 keys of 2 pages in
    key order; the PSUM copy reads them through an FP8 view, an exact upcast. ``latent_keys`` is
    ``[128, latent / 128, keys]`` and ``rope_keys`` ``[>= rope, keys]``, both BF16. The mask rows are the caller's.
    """
    bf16 = nl.bfloat16
    keys = blocks * PAGE_ROWS
    tiles = latent // 128
    for t in range(tiles + 1):
        rows = 128 if t < tiles else rope
        psum = nl.ndarray((rows, keys // 2), dtype=bf16, buffer=nl.psum)
        for q, parts in _pair_columns(blocks):
            nisa.nc_transpose(
                dst=psum[0:rows, q * 2 * PAIR_ROWS : q * 2 * PAIR_ROWS + parts],
                data=pairs[0:parts, q, 2 * t * 128 : 2 * t * 128 + 2 * rows].view(bf16),
            )
        if t < tiles:
            nisa.tensor_copy(
                dst=latent_keys[0:128, t, 0:keys],
                src=psum.view(nl.float8_e4m3fn),
                engine=nisa.engine.scalar if t % 2 else nisa.engine.vector,
            )
        else:
            nisa.tensor_copy(dst=rope_keys[0:rope, 0:keys], src=psum.view(nl.float8_e4m3fn), engine=nisa.engine.scalar)


def pair_numerator(weights, pairs, blocks: int, heads: int, sb, latent: int = 512):
    """Return the FP32 PSUM numerator ``[heads, latent]`` of the BF16 weights ``[heads, keys]`` over the pair rows.

    The contraction runs over the pairs of each column, one matmul for each token parity: lane ``k`` of partition
    ``p`` of column ``q`` is key ``256 q + 2 p + k``. So the weights of lane ``k`` are a stride-2 column view, and
    the values are a stride-2 view of the pair rows. The fp32 sum order differs from the row form.
    """
    bf16 = nl.bfloat16
    columns = list(_pair_columns(blocks))
    weights_psum = nl.ndarray((2 * PAIR_ROWS, 2 * len(columns), heads), dtype=bf16, buffer=nl.psum)
    for q, parts in columns:
        for k in range(2):
            nisa.nc_transpose(
                dst=weights_psum[0:parts, 2 * q + k, 0:heads],
                data=weights[0:heads, q * 4 * PAIR_ROWS : q * 4 * PAIR_ROWS + 2 * parts].slice(1, k, 2 * parts, step=2),
            )
    weights_t = sb.tile((2 * PAIR_ROWS, 2 * len(columns), heads), bf16)
    nisa.tensor_copy(dst=weights_t, src=weights_psum, engine=nisa.engine.scalar)
    numerator = nl.ndarray((heads, latent), dtype=nl.float32, buffer=nl.psum)
    first = True
    for q, parts in columns:
        for k in range(2):
            nisa.nc_matmul(
                dst=numerator,
                stationary=weights_t[0:parts, 2 * q + k, 0:heads],
                moving=pairs[0:parts, q, 0 : 2 * latent].slice(1, k, 2 * latent, step=2),
                accumulate=not first,
            )
            first = False
    return numerator
