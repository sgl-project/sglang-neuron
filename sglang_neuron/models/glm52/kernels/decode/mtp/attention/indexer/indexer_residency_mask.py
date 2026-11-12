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
GLM-5.2 DSA indexer: the additive key-residency mask, built ONCE per decode step.

``indexer_kv_score`` scores every key of this rank's contiguous cache slice, live or not, and drops the
keys at or beyond this step's base position by ADDING ``-MASK_BOUND`` to them on the way out of PSUM
(see that module for why the range is cut at ``positions[0]`` and why the mask is an add, not a clamp).
The mask depends only on ``positions[0] - key_base``, which is the same for every indexer layer of a
step, so building it inside the score loop -- one ``[T_q, 512]`` compare per chunk on 8 of the Vector
engine's 128 lanes in every layer -- repeated the same 262k-element computation 21 times per step.

This kernel builds it once, on all 128 lanes, in the layout the score loop's head sum lands in --
``[128, (num_keys / 128) * num_tokens]`` fp32, element ``[p, tile * num_tokens + t]`` belonging to key
``tile * 128 + p`` (the same value repeated for every token ``t``) -- so the loop DMAs one block of it
and adds it to the head-sum PSUM tile elementwise, on 128 lanes. Bit-identical to the in-loop form: a
kept key adds ``0.0``, a dropped key adds ``-MASK_BOUND`` and rounds back to exactly ``-MASK_BOUND``.
"""

import nki
import nki.isa as nisa
import nki.language as nl

from ....utils.kernel_assert import kernel_assert
from ....utils.scratch_hbm import layer_scratch_hbm
from .indexer_constants import MASK_BOUND, P_MAX, SCORE_KEY_CHUNK, SNAKE_PARTITIONS, snake_score_layout


@nki.jit
def indexer_residency_mask(
    positions: nl.NkiTensor,
    num_keys: int,
    num_tokens: int,
    key_base: int,
    shard_degree: int = 1,
    block_size: int = 0,
    rank_offset: nl.NkiTensor = None,
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

    base_i32 = nl.ndarray((P_MAX, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=base_i32[0:P_MAX, 0:1], src=positions[0:1].reshape((1, 1)).broadcast(0, P_MAX))
    base_sbuf = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=base_sbuf, src=base_i32)

    # local[p, tile] = tile * P_MAX + p: partition = key within its 128-key tile.
    key_pos_i32 = nl.ndarray((P_MAX, n_tiles), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(key_pos_i32, pattern=[[P_MAX, n_tiles]], offset=key_base, channel_multiplier=1)

    # INTERLEAVED: the local index is no longer the position offset, so rebuild the global position on
    # the iota -- see indexer_global_positions_np. rank_offset is RUNTIME data because one NEFF serves
    # every rank.
    if shard_degree > 1:
        bs = block_size or num_keys
        kernel_assert(bs > 0 and bs & (bs - 1) == 0, f"block_size={bs} must be a power of two")
        kernel_assert(num_keys % bs == 0, f"num_keys={num_keys} must be a multiple of block_size={bs}")
        kernel_assert(rank_offset is not None, "an interleaved shard needs rank_offset = shard_rank * block_size")
        shift = bs.bit_length() - 1
        blk = nl.ndarray((P_MAX, n_tiles), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_scalar(blk, key_pos_i32, op0=nl.right_shift, operand0=shift)
        intra = nl.ndarray((P_MAX, n_tiles), dtype=nl.int32, buffer=nl.sbuf)
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

    key_pos_sbuf = nl.ndarray((P_MAX, n_tiles), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=key_pos_sbuf, src=key_pos_i32)

    # (key_pos >= base) * -MASK_BOUND: exactly 0.0 for a kept key, exactly -MASK_BOUND for a dropped one.
    mask_tile_sbuf = nl.ndarray((P_MAX, n_tiles), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=mask_tile_sbuf,
        data=key_pos_sbuf,
        op0=nl.greater_equal,
        operand0=base_sbuf,
        op1=nl.multiply,
        operand1=-MASK_BOUND,
        engine=nisa.engine.vector,
    )

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
