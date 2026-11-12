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


"""GLM-5.2 DSA indexer: the additive key-residency mask, built once per decode step.

The score kernel drops keys at or past the step's base position by adding ``-MASK_BOUND`` as the head sums leave
PSUM. The cut depends only on ``positions[0] - key_base``, the same for every indexer layer of a step, so it is built
once here, on all 128 lanes, in the layout the head sum lands in: ``[128, (num_keys / 128) * num_tokens]`` fp32,
element ``[p, tile * num_tokens + t]`` for key ``tile * 128 + p``. A kept key adds 0.0 and a dropped one rounds to
exactly ``-MASK_BOUND``.
"""

import nki
import nki.isa as nisa
import nki.language as nl

from ...utils.kernel_assert import kernel_assert
from ...utils.scratch_hbm import layer_scratch_hbm
from .indexer_constants import (
    HEADSUM_PACKED,
    HEADSUM_SLICE,
    MASK_BOUND,
    P_MAX,
    SCORE_KEY_CHUNK,
    SNAKE_PARTITIONS,
    headsum_packed_geometry,
    snake_score_layout,
)


@nki.jit
def indexer_residency_mask(
    positions: nl.NkiTensor,
    num_keys: int,
    num_tokens: int,
    key_base: int,
    shard_degree: int = 1,
    block_size: int = 0,
    rank_offset: nl.NkiTensor = None,
    region_bases: nl.NkiTensor = None,
) -> nl.NkiTensor:
    """Build the additive residency mask for one rank's key slice.

    Args:
        positions: [>= 1] int32; element 0 is the step's base position (unused with ``region_bases``).
        num_keys: keys in this rank's slice, a multiple of ``P_MAX``.
        num_tokens: query rows of the score kernel; each key's value repeats per token.
        key_base: global position of local key 0.
        region_bases: [R] int32 for the chunk-dealt pool, or None: the cache is R regions, each a chunk of another
            sequence, and region ``r`` keeps keys below ``region_bases[r]``, so it equals the mask its owner builds.
            Packed head sum only.

    Returns:
        [P_MAX, (num_keys / P_MAX) * num_tokens] fp32, key-major: 0 where the key is below the cut, ``-MASK_BOUND``
        elsewhere.
    """
    kernel_assert(num_keys % P_MAX == 0, f"num_keys={num_keys} must be a multiple of {P_MAX}")
    kernel_assert(1 <= num_tokens <= P_MAX, f"num_tokens={num_tokens} must be in [1, {P_MAX}]")
    n_tiles = num_keys // P_MAX

    def load_base(src):
        """[P_MAX, 1] fp32 copy of one int32 cut, broadcast down the partitions."""
        b_i32 = nl.ndarray((P_MAX, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=b_i32[0:P_MAX, 0:1], src=src.reshape((1, 1)).broadcast(0, P_MAX))
        b_f32 = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=b_f32, src=b_i32)
        return b_f32

    if region_bases is not None:
        kernel_assert(HEADSUM_PACKED, "region_bases (the chunk-dealt pool) needs GLM52K_IDX_HEADSUM_PACKED=1")
        return _packed_region_mask(
            region_bases, load_base, num_keys, num_tokens, key_base, shard_degree, block_size, rank_offset
        )

    base_sbuf = load_base(positions[0:1])

    if HEADSUM_PACKED:
        # PACKED layout, matching the head sum's PSUM bank: partition p = tile*32 + slot*S + t holds
        # 512-key slice s = slot*4 + tile, so column b*512 + c is local key b*block + s*512 + c. The slice
        # term is partition-dependent and not affine in p, so it is built from a partition iota.
        slot_w, _, blk_keys, n_blocks = headsum_packed_geometry(num_tokens, num_keys)
        kernel_assert(num_keys % blk_keys == 0, f"num_keys={num_keys} must be a multiple of {blk_keys}")
        p_i32 = nl.ndarray((P_MAX, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.iota(p_i32, pattern=[[0, 1]], offset=0, channel_multiplier=1)
        tile_i32 = nl.ndarray((P_MAX, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_scalar(tile_i32, p_i32, op0=nl.right_shift, operand0=5)
        slot_i32 = nl.ndarray((P_MAX, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            slot_i32, p_i32, op0=nl.bitwise_and, operand0=31, op1=nl.right_shift, operand1=slot_w.bit_length() - 1
        )
        slice_i32 = nl.ndarray((P_MAX, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.scalar_tensor_tensor(
            dst=slice_i32, data=slot_i32, op0=nl.multiply, operand0=4.0, op1=nl.add, operand1=tile_i32
        )
        slice_off_f32 = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(slice_off_f32, slice_i32, op0=nl.multiply, operand0=float(HEADSUM_SLICE))
        mask_add = nl.ndarray((P_MAX, n_blocks * HEADSUM_SLICE), dtype=nl.float32, buffer=layer_scratch_hbm())
        # In chunks of whole blocks: at 1M the full width is 16384 columns, and the five working tiles at
        # that width overflow SBUF.
        chunk_blocks = max(1, _PACKED_CHUNK_COLS // HEADSUM_SLICE)
        for b0 in range(0, n_blocks, chunk_blocks):
            nb = min(chunk_blocks, n_blocks - b0)
            n_cols = nb * HEADSUM_SLICE
            key_pos_i32 = nl.ndarray((P_MAX, n_cols), dtype=nl.int32, buffer=nl.sbuf)
            nisa.iota(
                key_pos_i32,
                pattern=[[blk_keys, nb], [1, HEADSUM_SLICE]],
                offset=key_base + b0 * blk_keys,
                channel_multiplier=0,
            )
            nisa.tensor_scalar(key_pos_i32, key_pos_i32, op0=nl.add, operand0=slice_off_f32[0:P_MAX, 0:1])
            mask_tile_sbuf = _keep_mask(key_pos_i32, n_cols, base_sbuf, num_keys, shard_degree, block_size, rank_offset)
            nisa.dma_copy(dst=mask_add[0:P_MAX, b0 * HEADSUM_SLICE : b0 * HEADSUM_SLICE + n_cols], src=mask_tile_sbuf)
        return mask_add

    # local[p, tile] = tile * P_MAX + p: partition = key within its 128-key tile.
    key_pos_i32 = nl.ndarray((P_MAX, n_tiles), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(key_pos_i32, pattern=[[P_MAX, n_tiles]], offset=key_base, channel_multiplier=1)

    mask_tile_sbuf = _keep_mask(key_pos_i32, n_tiles, base_sbuf, num_keys, shard_degree, block_size, rank_offset)

    # Repeat each tile's column once per token so the loop's eviction is a same-shape add: column
    # tile * num_tokens + t. num_tokens tiny copies of n_tiles elements per partition.
    if snake_score_layout(num_keys, num_tokens):
        # The snake score loop's column order (indexer_kv_score._score_all_keys_snake): block b, snake row q
        # and chunk tile j is key tile q * (n_blk * tpc) + b * tpc + j.
        tpc = SCORE_KEY_CHUNK // P_MAX
        n_blk = n_tiles // (SNAKE_PARTITIONS * tpc)
        # (block, tile j, token, snake row q): the order the eviction writes.
        tiles_bjq = mask_tile_sbuf.reshape_dim(1, [SNAKE_PARTITIONS, n_blk, tpc]).permute((0, 2, 3, 1))
        mask_sbuf = nl.ndarray((P_MAX, n_blk, tpc, num_tokens, SNAKE_PARTITIONS), dtype=nl.float32, buffer=nl.sbuf)
        for t in range(num_tokens):
            nisa.tensor_copy(dst=mask_sbuf[0:P_MAX, 0:n_blk, 0:tpc, t, 0:SNAKE_PARTITIONS], src=tiles_bjq)
    else:
        mask_sbuf = nl.ndarray((P_MAX, n_tiles, num_tokens), dtype=nl.float32, buffer=nl.sbuf)
        for t in range(num_tokens):
            nisa.tensor_copy(dst=mask_sbuf[0:P_MAX, 0:n_tiles, t], src=mask_tile_sbuf)

    mask_add = nl.ndarray((P_MAX, n_tiles * num_tokens), dtype=nl.float32, buffer=layer_scratch_hbm())
    nisa.dma_copy(dst=mask_add, src=mask_sbuf.reshape((P_MAX, n_tiles * num_tokens)))
    return mask_add


_PACKED_CHUNK_COLS = 2048
"""Columns per working chunk of the packed mask: 8 KB per fp32 tile, five live at once."""


def _packed_slice_offsets(slot_w: int) -> nl.NkiTensor:
    """[P_MAX, 1] fp32 key offset of each packed partition's 512-key slice (see the packed branch above)."""
    p_i32 = nl.ndarray((P_MAX, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(p_i32, pattern=[[0, 1]], offset=0, channel_multiplier=1)
    tile_i32 = nl.ndarray((P_MAX, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(tile_i32, p_i32, op0=nl.right_shift, operand0=5)
    slot_i32 = nl.ndarray((P_MAX, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        slot_i32, p_i32, op0=nl.bitwise_and, operand0=31, op1=nl.right_shift, operand1=slot_w.bit_length() - 1
    )
    slice_i32 = nl.ndarray((P_MAX, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(dst=slice_i32, data=slot_i32, op0=nl.multiply, operand0=4.0, op1=nl.add, operand1=tile_i32)
    slice_off_f32 = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(slice_off_f32, slice_i32, op0=nl.multiply, operand0=float(HEADSUM_SLICE))
    return slice_off_f32


def _packed_region_mask(
    region_bases, load_base, num_keys, num_tokens, key_base, shard_degree, block_size, rank_offset
) -> nl.NkiTensor:
    """The packed mask with one cut per REGION (``indexer_residency_mask(region_bases=)``, the chunk-dealt pool).

    Region ``r`` covers columns ``[r * region_cols, (r + 1) * region_cols)``, i.e. local keys ``[r * region_keys,
    (r + 1) * region_keys)``. Each region is built exactly as the packed branch builds those columns -- the same iota,
    slice offset, deal, ``rank_offset`` and compare, in chunks of at most ``_PACKED_CHUNK_COLS`` columns -- only the
    cut is ``region_bases[r]``. 8 regions of 32,768 keys at 1M with cp 4: each region is one 2048-column chunk.
    """
    R = region_bases.shape[0]
    slot_w, _, blk_keys, n_blocks = headsum_packed_geometry(num_tokens, num_keys)
    kernel_assert(
        num_keys % R == 0 and (num_keys // R) % blk_keys == 0,
        f"{R} regions of {num_keys} keys must each be a whole number of {blk_keys}-key head-sum blocks",
    )
    region_blocks = (num_keys // R) // blk_keys
    slice_off_f32 = _packed_slice_offsets(slot_w)
    mask_add = nl.ndarray((P_MAX, n_blocks * HEADSUM_SLICE), dtype=nl.float32, buffer=layer_scratch_hbm())
    chunk_blocks = max(1, min(_PACKED_CHUNK_COLS // HEADSUM_SLICE, region_blocks))
    kernel_assert(region_blocks % chunk_blocks == 0, f"{region_blocks} blocks per region vs chunks of {chunk_blocks}")
    for r in range(R):
        base_sbuf = load_base(region_bases[r : r + 1])
        for b0 in range(r * region_blocks, (r + 1) * region_blocks, chunk_blocks):
            nb = chunk_blocks
            n_cols = nb * HEADSUM_SLICE
            key_pos_i32 = nl.ndarray((P_MAX, n_cols), dtype=nl.int32, buffer=nl.sbuf)
            nisa.iota(
                key_pos_i32,
                pattern=[[blk_keys, nb], [1, HEADSUM_SLICE]],
                offset=key_base + b0 * blk_keys,
                channel_multiplier=0,
            )
            nisa.tensor_scalar(key_pos_i32, key_pos_i32, op0=nl.add, operand0=slice_off_f32[0:P_MAX, 0:1])
            mask_tile_sbuf = _keep_mask(key_pos_i32, n_cols, base_sbuf, num_keys, shard_degree, block_size, rank_offset)
            nisa.dma_copy(dst=mask_add[0:P_MAX, b0 * HEADSUM_SLICE : b0 * HEADSUM_SLICE + n_cols], src=mask_tile_sbuf)
    return mask_add


def _keep_mask(key_pos_i32, n_cols, base_sbuf, num_keys, shard_degree, block_size, rank_offset):
    """``[P_MAX, n_cols]`` fp32 additive mask from LOCAL key indices: deal them, then 0 / -MASK_BOUND.

    ``key_pos_i32`` is overwritten. See ``indexer_global_positions_np`` for the deal.
    """
    # INTERLEAVED: the local index is no longer the position offset, so rebuild the global position on
    # the iota -- see indexer_global_positions_np. rank_offset is RUNTIME data because one NEFF serves
    # every rank.
    if shard_degree > 1:
        bs = block_size or num_keys
        kernel_assert(bs > 0 and bs & (bs - 1) == 0, f"block_size={bs} must be a power of two")
        kernel_assert(num_keys % bs == 0, f"num_keys={num_keys} must be a multiple of block_size={bs}")
        kernel_assert(rank_offset is not None, "an interleaved shard needs rank_offset = shard_rank * block_size")
        shift = bs.bit_length() - 1
        blk = nl.ndarray((P_MAX, n_cols), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_scalar(blk, key_pos_i32, op0=nl.right_shift, operand0=shift)
        intra = nl.ndarray((P_MAX, n_cols), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_scalar(intra, key_pos_i32, op0=nl.bitwise_and, operand0=bs - 1)
        nisa.scalar_tensor_tensor(
            dst=key_pos_i32,
            data=blk,
            op0=nl.multiply,
            operand0=float(shard_degree * bs),
            op1=nl.add,
            operand1=intra,
        )
        # FP32, not int32: tensor_scalar rejects an i32 operand0. Exact for a position either way.
        off_f32 = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=off_f32[0:P_MAX, 0:1], src=rank_offset[0:1, 0:1].broadcast(0, P_MAX))
        nisa.tensor_scalar(key_pos_i32, key_pos_i32, op0=nl.add, operand0=off_f32[0:P_MAX, 0:1])

    key_pos_sbuf = nl.ndarray((P_MAX, n_cols), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=key_pos_sbuf, src=key_pos_i32)

    # (key_pos >= base) * -MASK_BOUND: exactly 0.0 for a kept key, exactly -MASK_BOUND for a dropped one.
    mask_tile_sbuf = nl.ndarray((P_MAX, n_cols), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=mask_tile_sbuf,
        data=key_pos_sbuf,
        op0=nl.greater_equal,
        operand0=base_sbuf,
        op1=nl.multiply,
        operand1=-MASK_BOUND,
        engine=nisa.engine.vector,
    )
    return mask_tile_sbuf
