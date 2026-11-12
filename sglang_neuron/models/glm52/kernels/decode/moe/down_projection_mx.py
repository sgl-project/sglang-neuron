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
Down projection sub-kernels for hardware MX (MXFP8 activations, MXFP4 / MXFP8 weights).

Each NeuronCore computes the full down projection for one expert: the whole intermediate dimension is
contracted locally and the whole hidden dimension is produced locally, so there is no cross-core
reduction. Results are accumulated across experts either in SBUF or directly in HBM, selected by an
SBUF-pressure heuristic. PSUM, the evicted expert output and the accumulator are all bfloat16.

These sub-kernels can be used by any algorithm that requires an MX down projection.
"""

import os
from typing import Optional

import nki.isa as nisa
import nki.language as nl

# Common utils
from ..utils.common_types import ExpertAffinityScaleMode
from ..utils.kernel_assert import kernel_assert
from ..utils.kernel_helpers import div_ceil

# Shared MX constants
from .projection_mx_constants import (
    NUM_QUADRANTS_IN_SBUF,
    SBUF_QUADRANT_SIZE,
    SCALE_P_ELEM_PER_QUADRANT,
    SCALE_PACK64_N_PARTITIONS,
    SCALE_PACK64_P_PER_QUADRANT,
)

MOE_DOWN_PSUM_SCALE = os.environ.get("GLM52K_MOE_DOWN_PSUM_SCALE", "0")
"""Post-scale every down-projection tile straight from its bf16 PSUM tile, without the PSUM -> SBUF copy first
(``1``: every expert; ``last``: the last expert only). The copy is a bf16 -> bf16 identity, so the post-scale reads
the same values: bit-identical. It shortens the Vector engine's tail after the last routed matmul. Off by
default."""
MOE_DOWN_LAST_SPLIT = int(os.environ.get("GLM52K_MOE_DOWN_LAST_SPLIT", "1"))
"""Compute the last expert's LAST H tile (SBUF accumulation) as this many column pieces, each with its own PSUM tile,
matmuls, post-scale and store, so that only the last piece's post-scale and store follow the last routed matmul. Each
output column sees the same matmuls over the same I tiles in the same order: bit-identical. Must divide the
1024-column tile. ``1`` (default) is off."""

# PACK64 down-scale consumer extent. Real scale rows sit at partition 32q + 4*tile_i + r, so an AP
# starting at this tile's slot (4*tile_i) must reach 32*3 + 4*tile_i + 3 to span every quadrant --
# a constant 100 partitions regardless of tile_i, which keeps the extent static inside the traced
# tile_i loop. See the extent note in the gate/up PACK64 path: too small silently mis-scales the
# quadrants it fails to name.
_SCALE_PACK64_DOWN_AP_EXTENT = SBUF_QUADRANT_SIZE * (NUM_QUADRANTS_IN_SBUF - 1) + SCALE_P_ELEM_PER_QUADRANT


def down_h_tile_size(H: int) -> int:
    """Columns per H tile of ``down_projection_mx`` (one bf16 PSUM tile: 2 x psum_fmax)."""
    return min(H, nl.tile_size.psum_fmax * 2)


def _load_down_scale_unpacked(
    scale: nl.NkiTensor,
    scale_sb: nl.NkiTensor,
    expert_idx: int,
    H: int,
    tile_I: int,
    n_I512_tiles: int,
    I_p_scale_in_hbm: int,
) -> None:
    """Load the unpacked [E_L, 16_I, I/512, H] down scale, 4 partitions per SBUF quadrant.

    The 16 HBM scale rows map to SBUF partitions [0-3, 32-35, 64-67, 96-99]: a scale must sit in the
    quadrant its scaling group came from, and a 32-partition destination stride is inexpressible, so
    one DMA per quadrant is the floor for this layout.

    Args:
        scale (nl.NkiTensor): [E_L, 16_I, I/512, H] unpacked MX scales in HBM.
        scale_sb (nl.NkiTensor): [128_I, n_I512_tiles, H] destination tile in SBUF.
        expert_idx (int): Index of the current expert to load.
        H (int): Hidden dimension size.
        tile_I (int): Tile size for the I dimension (typically 128).
        n_I512_tiles (int): Number of I/512 tiles to load.
        I_p_scale_in_hbm (int): Scale partition count present in HBM (16 when I >= 512).

    Returns:
        None. Scales are written into scale_sb.
    """
    n_quadrants_needed = div_ceil(I_p_scale_in_hbm, SCALE_P_ELEM_PER_QUADRANT)

    if I_p_scale_in_hbm < tile_I // 8:
        for quadrant_idx in nl.affine_range(NUM_QUADRANTS_IN_SBUF):
            nisa.memset(
                dst=scale_sb[nl.ds(SBUF_QUADRANT_SIZE * quadrant_idx, SCALE_P_ELEM_PER_QUADRANT), :, :], value=0.0
            )

    for quadrant_idx in nl.affine_range(n_quadrants_needed):
        actual_scale_p = min(SCALE_P_ELEM_PER_QUADRANT, I_p_scale_in_hbm - SCALE_P_ELEM_PER_QUADRANT * quadrant_idx)
        if actual_scale_p > 1:
            scale_view = (
                scale.select(dim=0, index=expert_idx)
                .slice(
                    dim=0,
                    start=SCALE_P_ELEM_PER_QUADRANT * quadrant_idx,
                    end=SCALE_P_ELEM_PER_QUADRANT * quadrant_idx + actual_scale_p,
                )
                .slice(dim=1, start=0, end=n_I512_tiles)
            )
            nisa.dma_copy(
                src=scale_view,
                dst=scale_sb[nl.ds(SBUF_QUADRANT_SIZE * quadrant_idx, actual_scale_p), :, :],
                dge_mode=nisa.dge_mode.none,
            )
        else:
            hbm_p_idx = SCALE_P_ELEM_PER_QUADRANT * quadrant_idx
            sb_p_idx = SBUF_QUADRANT_SIZE * quadrant_idx
            scale_f_per_partition = n_I512_tiles * H
            expert_stride = I_p_scale_in_hbm * scale_f_per_partition
            hbm_offset = expert_idx * expert_stride + hbm_p_idx * scale_f_per_partition
            nisa.dma_copy(
                src=scale.ap(
                    pattern=[[scale_f_per_partition, 1], [1, scale_f_per_partition]],
                    offset=hbm_offset,
                ),
                dst=scale_sb.ap(
                    pattern=[[scale_f_per_partition, 1], [1, scale_f_per_partition]],
                    offset=sb_p_idx * scale_f_per_partition,
                ),
                dge_mode=nisa.dge_mode.none,
            )


def down_scale_is_pack64(scale: nl.NkiTensor) -> bool:
    """True when the down MX scale tensor in HBM is in the PACK64 layout.

    Unpacked is 4D ``[E_L, 16_I, I/512, H]``. PACK64 folds the I/512 contraction-tile axis into
    partitions -- 16 rows per SBUF quadrant instead of 4 -- giving 3D ``[E_L, 64_I, H]``, so the
    rank identifies the layout. Read off the tensor rather than a flag so the caller's layout and
    the kernel's load cannot disagree.

    Unlike the gate/up scale, the folded factor is the contraction-tile axis, not a free-dim factor:
    the down projection issues one matmul per I/512 tile over the whole H tile, so tile_i is the
    only 4-way grouping available to map onto the ISA's four scale_pidx slots.

    Args:
        scale (nl.NkiTensor): Down projection MX scale tensor from HBM.

    Returns:
        bool: True for the PACK64 layout, False for the unpacked layout.
    """
    return len(scale.shape) == 3 and scale.shape[1] == SCALE_PACK64_N_PARTITIONS


def load_broadcast_down_weight_scale_bias(
    weight: nl.NkiTensor,
    scale: nl.NkiTensor,
    bias: Optional[nl.NkiTensor],
    expert_idx: int,
    H: int,
    tile_I: int,
    n_I512_tiles: int,
    tile_T: int,
    dma_handles: Optional[list] = None,
) -> tuple[nl.NkiTensor, nl.NkiTensor, Optional[nl.NkiTensor]]:
    """
    Load down projection weight, scale, and bias (optional) for one expert using static DMA.

    The bias is loaded into the first partition and then broadcast over the T partitions on the PE
    (matmul against a ones vector). Activations are computed in bfloat16, so the bias buffer is
    bfloat16 too and the broadcast matmul uses the doubled 16-bit PSUM free dim.

    Args:
        weight (nl.NkiTensor): [E_L, 128_I, I/512, H], Down projection weight tensor from HBM (4_I packed in x4 dtype).
        scale (nl.NkiTensor): Down projection MX scale tensor from HBM (uint8 MX scales), in either
            layout: unpacked [E_L, 16_I, I/512, H] or PACK64 [E_L, 64_I, H]. PACK64 folds the I/512
            axis into partitions, so each DMA moves 16 partitions of H bytes instead of 4 partitions
            of n_I512_tiles*H; it requires n_I512_tiles <= 4 and a full 16-row scale.
        bias (Optional[nl.NkiTensor]): [E_L, H], Optional down projection bias tensor from HBM.
        expert_idx (int): Index of the current expert to load.
        H (int): Hidden dimension size.
        tile_I (int): Tile size for I dimension (typically 128).
        n_I512_tiles (int): Number of I/512 tiles to load.
        tile_T (int): Tile size for T dimension (for bias broadcast).
        dma_handles (Optional[list]): When given, the weight DMA's instruction is appended (for ordering).

    Returns:
        weight_sb (nl.NkiTensor): [128_I, n_I512_tiles, H], Weight in SBUF (4_I packed in x4 dtype).
        scale_sb (nl.NkiTensor): Scales in SBUF. [128_I, n_I512_tiles, H] with 4 rows per quadrant for
            the unpacked layout, [128_I, H] with 16 rows per quadrant for PACK64.
        bias_sb (Optional[nl.NkiTensor]): [tile_T, H], Broadcasted bias in SBUF (None when bias=None).

    Notes:
        - Based on experiments, static DMA demonstrates better performance
        - Can revert to DGE if HBM out-of-memory (OOM) issues occur
    """

    # Calculate shapes / tiling
    weight_sb_shape = (tile_I, n_I512_tiles, H)
    bias_sb_shape = (tile_T, H)

    # PACK64 keeps the scale in 64 partitions with the I/512 axis folded in, so the tile drops that
    # axis entirely: [128_I, H] instead of [128_I, n_I512_tiles, H] (n_I512_tiles x less SBUF).
    is_pack64 = down_scale_is_pack64(scale)
    scale_sb_shape = (tile_I, H) if is_pack64 else weight_sb_shape

    # Allocate buffers
    base_weight = weight
    weight_sb = nl.ndarray(weight_sb_shape, dtype=base_weight.dtype, buffer=nl.sbuf)
    scale_sb = nl.ndarray(scale_sb_shape, dtype=scale.dtype, buffer=nl.sbuf)
    bias_sb: Optional[nl.NkiTensor] = None

    I_p_in_hbm = base_weight.shape[1]

    # Load weight: index expert, then slice I/512 tiles
    # Shape: [E_L, I_p, I/512, H] -> [I_p, n_I512_tiles, H] -> padded to [128_I, n_I512_tiles, H]
    if I_p_in_hbm < tile_I:
        nisa.memset(dst=weight_sb[...], value=0)
        weight_view = base_weight.select(dim=0, index=expert_idx).slice(dim=1, start=0, end=n_I512_tiles)
        weight_dma = nisa.dma_copy(src=weight_view, dst=weight_sb[:I_p_in_hbm, :, :], dge_mode=nisa.dge_mode.none)
    else:
        weight_view = base_weight.select(dim=0, index=expert_idx).slice(dim=1, start=0, end=n_I512_tiles)
        weight_dma = nisa.dma_copy(src=weight_view, dst=weight_sb[...], dge_mode=nisa.dge_mode.none)
    if dma_handles != None:
        dma_handles.append(weight_dma)
    weight_sb = weight_sb.view(weight.dtype)

    """
    Load scale: index expert, then slice I/512 tiles.
    Shape: [E_L, I_p//8, I/512, H] -> [I_p//8, n_I512_tiles, H] -> padded to [128_I, n_I512_tiles, H]
    Note: scales have I_p//8 (not 128), need to map to first 4 partitions of each quadrant.
    Scale layout: 16 partitions map to partitions [0-3, 32-35, 64-67, 96-99] in 128-partition buffer.
    """
    I_p_scale_in_hbm = scale.shape[1]

    if is_pack64:
        # PACK64 maps HBM row 16q + 4*tile_i + r onto SBUF partition 32q + 4*tile_i + r, so one DMA
        # per quadrant copies 16 contiguous rows -- same bytes as the 4 x [4 P, n_I512*H] it
        # replaces, but 4x more partitions and n_I512 x narrower.
        kernel_assert(
            n_I512_tiles <= NUM_QUADRANTS_IN_SBUF,
            f"PACK64 down scale folds the I/512 axis into the ISA's {NUM_QUADRANTS_IN_SBUF} scale_pidx "
            f"slots, so it requires n_I512_tiles <= {NUM_QUADRANTS_IN_SBUF} (I <= 2048), got {n_I512_tiles=}. "
            f"Supply the unpacked [E_L, 16_I, I/512, H] scale instead.",
        )
        for quadrant_idx in nl.affine_range(NUM_QUADRANTS_IN_SBUF):
            pack_view = scale.select(dim=0, index=expert_idx).slice(
                dim=0,
                start=SCALE_PACK64_P_PER_QUADRANT * quadrant_idx,
                end=SCALE_PACK64_P_PER_QUADRANT * (quadrant_idx + 1),
            )
            nisa.dma_copy(
                src=pack_view,
                dst=scale_sb[nl.ds(SBUF_QUADRANT_SIZE * quadrant_idx, SCALE_PACK64_P_PER_QUADRANT), :],
                dge_mode=nisa.dge_mode.none,
            )
    else:
        _load_down_scale_unpacked(
            scale=scale,
            scale_sb=scale_sb,
            expert_idx=expert_idx,
            H=H,
            tile_I=tile_I,
            n_I512_tiles=n_I512_tiles,
            I_p_scale_in_hbm=I_p_scale_in_hbm,
        )

    """
    Load + broadcast bias over the T partitions.
    Shape: [E_L, H] -> [1, H] -> broadcast to [tile_T, H]
    """
    if bias != None:
        bias_sb = nl.ndarray(bias_sb_shape, dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.memset(dst=bias_sb[...], value=0.0, engine=nisa.gpsimd_engine)
        bias_view = bias.slice(dim=0, start=expert_idx, end=expert_idx + 1)
        nisa.dma_copy(src=bias_view, dst=bias_sb[0:1, :], dge_mode=nisa.dge_mode.none)

        # Broadcast bias using PE (matmul against a ones vector). bf16 PSUM doubles the free dim.
        H_tile_size_local = min(H, nl.tile_size.psum_fmax * 2)
        ones_mask = nl.ndarray((1, tile_T), dtype=bias_sb.dtype, buffer=nl.sbuf)
        nisa.memset(dst=ones_mask[...], value=1.0, engine=nisa.gpsimd_engine)
        n_H_tiles_local = div_ceil(H, H_tile_size_local)
        for h_tile_idx in nl.affine_range(n_H_tiles_local):
            h_tile_actual = min(H_tile_size_local, H - h_tile_idx * H_tile_size_local)
            bias_bc_psum = nl.ndarray((tile_T, h_tile_actual), dtype=nl.bfloat16, buffer=nl.psum)
            H_tile_slice = nl.ds(h_tile_idx * H_tile_size_local, h_tile_actual)
            nisa.nc_matmul(
                dst=bias_bc_psum,
                stationary=ones_mask[...],
                moving=bias_sb[0:1, H_tile_slice],
                is_stationary_onezero=True,
            )
            nisa.tensor_copy(
                dst=bias_sb[:, H_tile_slice],
                src=bias_bc_psum[...],
                engine=nisa.scalar_engine,
            )

    return weight_sb, scale_sb, bias_sb


def down_projection_mx(
    act_sb: nl.NkiTensor,
    act_scale_sb: nl.NkiTensor,
    weight_sb: nl.NkiTensor,
    weight_scale_sb: nl.NkiTensor,
    bias_sb: Optional[nl.NkiTensor],
    expert_affinities_masked_sb: nl.NkiTensor,
    expert_idx: int,
    out_sb: nl.NkiTensor,
    out_hbm: nl.NkiTensor,
    expert_affinities_scaling_mode: ExpertAffinityScaleMode = ExpertAffinityScaleMode.POST_SCALE,
    is_first_expert: bool = False,
    is_last_expert: bool = False,
    T_physical: int = None,
) -> nl.NkiTensor:
    """
    Computes down projection, expert affinity scaling, expert add, and SB->HBM spill.

    Usage:
        Tuned for: mx all-expert MoE algorithm
        Applicable to: any algorithm requiring an mx down projection

    Args:
        act_sb (nl.NkiTensor): [16_I * 8_I, I/512, T], Activation tensor in SBUF (4_I packed in x4 dtype).
        act_scale_sb (nl.NkiTensor): [16_I * 8_I, I/512, T], Activation scales in SBUF
            (in leading 4P of each SBUF quadrant).
        weight_sb (nl.NkiTensor): [16_I * 8_I, I/512, H], Weight tensor in SBUF (4_I packed in x4 dtype).
        weight_scale_sb (nl.NkiTensor): [16_I * 8_I, I/512, H], Weight scales in SBUF
            (in leading 4P of each SBUF quadrant).
        bias_sb (Optional[nl.NkiTensor]): [tile_T, H], Optional broadcasted bias tensor in SBUF.
        expert_affinities_masked_sb (nl.NkiTensor): [T, E_L] or [128_T, T/128, E_L],
            Expert affinity scores in SBUF. None scales by an immediate 1.0 (unit affinities).
        expert_idx (int): Index of the current expert.
        out_sb (nl.NkiTensor): [min(T, 128), ⌈T/128⌉, H], Output tensor in SBUF.
        out_hbm (nl.NkiTensor): [T, H], Output tensor in HBM for spill.
        expert_affinities_scaling_mode (ExpertAffinityScaleMode): Scaling mode for expert affinities.
        is_first_expert (bool): Whether the current expert is the first expert.
        is_last_expert (bool): Whether the current expert is the last expert.
        T_physical (int): Number of real (unpadded) tokens, when T is padded. None when T is exact.

    Returns:
        out_sb (nl.NkiTensor): [min(T, 128), ⌈T/128⌉, H], Output tensor in SBUF with accumulated results.
    """

    # Extract / validate shapes
    TILE_I, n_I512_tiles, T = act_sb.shape
    T_out = T_physical if T_physical is not None else T
    TILE_I_, n_I512_tiles_, H = weight_sb.shape
    kernel_assert(
        TILE_I == TILE_I_, f"Expected same number of partitions in activation and weight, got {TILE_I}, {TILE_I_}"
    )
    kernel_assert(
        n_I512_tiles == n_I512_tiles_,
        f"Expected same number of I tiles in activation and weight, got {n_I512_tiles}, {n_I512_tiles_}",
    )
    kernel_assert(H % 512 == 0, f"Expected H divisible by 512, got {H=}")
    kernel_assert(
        expert_affinities_scaling_mode == ExpertAffinityScaleMode.POST_SCALE,
        f"Expected expert_affinities_scaling_mode={ExpertAffinityScaleMode.POST_SCALE}, "
        f"got: {expert_affinities_scaling_mode=}",
    )

    # Algorithm + tiling strategy
    pmax = nl.tile_size.pmax
    TILE_T = min(T, pmax)  # T will be partition dim in output
    TILE_H = down_h_tile_size(H)  # use 2 * fmax with bf16 PSUM
    n_T128_tiles = div_ceil(T, pmax)
    n_H1024_tiles = H // TILE_H

    # Cast expert affinities to fp32 for tensor_scalar on scalar engine
    # FIXME[perf]: skip TensorCopy and use strided AP in below TensorScalar when affinities are already fp32
    expert_affinities_masked_fp32_sb = None
    if expert_affinities_masked_sb != None:
        is_3D_affinities = len(expert_affinities_masked_sb.shape) == 3
        expert_affinities_masked_fp32_sb = nl.ndarray((TILE_T, n_T128_tiles), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(
            dst=expert_affinities_masked_fp32_sb[...],
            src=expert_affinities_masked_sb[:, :, expert_idx]
            if is_3D_affinities
            else expert_affinities_masked_sb[:, expert_idx],
            engine=nisa.scalar_engine,
        )

    # HBM accumulation strategy: when out_sb exceeds 64KB per SBUF partition, it consumes
    # a significant fraction of usable SBUF (trn3: 240KB, trn2: 213KB), constraining the
    # compiler's ability to double-buffer weight/activation loads alongside compute.
    # Switching to HBM accumulation (dma_compute atomic add) eliminates out_sb entirely,
    # freeing SBUF for better DMA/compute overlap.
    #
    # Profiled crossover:
    #   out_sb=28KB  (T=256, H=7168): HBM accum regresses (overhead > benefit)
    #   out_sb=96KB  (T=2048, H=3072): HBM accum improves
    #   out_sb=224KB (T=2048, H=7168): HBM accum improves
    #
    # When out_sb is small, SBUF accumulation is cheaper (no HBM round-trips).
    _HBM_ACCUM_THRESHOLD_BYTES = 64 * 1024
    out_sb_per_partition_bytes = n_T128_tiles * H * 2  # bf16 = 2 bytes
    use_hbm_accumulation = out_sb_per_partition_bytes > _HBM_ACCUM_THRESHOLD_BYTES

    # The last expert's SBUF accumulator is stored to HBM tile by tile, each (H tile, T tile) right after its
    # post-scale, so only the last tile's store follows the last matmul before the ReduceScatter that consumes the
    # output. A single deferred spill after the loop left the store stream bandwidth-bound in the tail.
    tile_store = is_last_expert and not use_hbm_accumulation

    # MOE_DOWN_LAST_SPLIT: the last H tile of those stores as column pieces
    last_tile_split = MOE_DOWN_LAST_SPLIT if tile_store else 1
    kernel_assert(
        last_tile_split >= 1 and TILE_H % last_tile_split == 0,
        f"GLM52K_MOE_DOWN_LAST_SPLIT={last_tile_split} must divide TILE_H={TILE_H}",
    )
    # MOE_DOWN_PSUM_SCALE: post-scale from PSUM (no eviction copy)
    psum_scale_all = MOE_DOWN_PSUM_SCALE == "1" or (MOE_DOWN_PSUM_SCALE == "last" and is_last_expert)

    # Unified loop: always for_h{for_t}. Accumulation strategy selected by heuristic above.
    for tile_h in nl.sequential_range(n_H1024_tiles):
        is_last_h = tile_h == n_H1024_tiles - 1
        n_pieces = last_tile_split if is_last_h else 1
        piece_H = TILE_H // n_pieces

        for tile_t in nl.sequential_range(n_T128_tiles):
            tile_T_offset = TILE_T * tile_t
            tile_T_actual = min(TILE_T, T - tile_T_offset)
            tile_T_slice = nl.ds(tile_T_offset, tile_T_actual)

            for piece in range(n_pieces):
                # One piece per tile unless last_tile_split: the whole H tile
                tile_H_offset = TILE_H * tile_h + piece_H * piece
                weight_H_slice = nl.ds(tile_H_offset, piece_H)
                expert_out_tile_sb = _down_proj_tile_compute(
                    act_sb=act_sb,
                    act_scale_sb=act_scale_sb,
                    weight_sb=weight_sb,
                    weight_scale_sb=weight_scale_sb,
                    bias_sb=bias_sb,
                    expert_affinities_masked_fp32_sb=expert_affinities_masked_fp32_sb,
                    t_tile_idx=tile_t,
                    tile_T_actual=tile_T_actual,
                    tile_T_slice=tile_T_slice,
                    tile_H_offset=tile_H_offset,
                    weight_H_slice=weight_H_slice,
                    TILE_T=TILE_T,
                    TILE_H=piece_H,
                    n_I512_tiles=n_I512_tiles,
                    perform_accumulation=not use_hbm_accumulation,
                    accumulation_result_tile=out_sb[
                        :tile_T_actual, tile_t : tile_t + 1, tile_H_offset : tile_H_offset + piece_H
                    ],
                    overwrite_accum_result=is_first_expert,
                    psum_postscale=psum_scale_all,
                )
                if n_pieces > 1:
                    # MOE_DOWN_LAST_SPLIT: store each piece of the last H tile right after its post-scale
                    T_out_piece = min(tile_T_actual, T_out - tile_T_offset)
                    nisa.dma_copy(
                        src=out_sb[:T_out_piece, tile_t : tile_t + 1, tile_H_offset : tile_H_offset + piece_H],
                        dst=out_hbm[nl.ds(tile_T_offset, T_out_piece), tile_H_offset : tile_H_offset + piece_H],
                        dge_mode=nisa.dge_mode.none,
                    )
            tile_H_offset = TILE_H * tile_h
            tile_T_out_actual = min(tile_T_actual, T_out - tile_T_offset)

            if use_hbm_accumulation:
                # HBM accumulation: write per tile per expert
                hbm_T_slice = nl.ds(TILE_T * tile_t, tile_T_out_actual)
                hbm_dst = out_hbm[hbm_T_slice, nl.ds(tile_H_offset, TILE_H)]
                if is_first_expert:
                    nisa.dma_copy(
                        src=expert_out_tile_sb[:tile_T_out_actual, :], dst=hbm_dst, dge_mode=nisa.dge_mode.none
                    )
                else:
                    nisa.dma_compute(
                        dst=hbm_dst,
                        srcs=[hbm_dst, expert_out_tile_sb[:tile_T_out_actual, :]],
                        scales=[1.0, 1.0],
                        reduce_op=nl.add,
                    )
            # SBUF accumulation: the accumulation into out_sb is fused into the expert affinity scaling inside
            # _down_proj_tile_compute; the last expert stores this (H tile, T tile) now
            elif tile_store and n_pieces == 1:
                store_src = out_sb[:tile_T_out_actual, tile_t : tile_t + 1, tile_H_offset : tile_H_offset + TILE_H]
                store_dst = out_hbm[nl.ds(TILE_T * tile_t, tile_T_out_actual), tile_H_offset : tile_H_offset + TILE_H]
                nisa.dma_copy(src=store_src, dst=store_dst, dge_mode=nisa.dge_mode.none)

    return out_sb


def _down_proj_tile_compute(
    act_sb: nl.NkiTensor,
    act_scale_sb: nl.NkiTensor,
    weight_sb: nl.NkiTensor,
    weight_scale_sb: nl.NkiTensor,
    bias_sb: Optional[nl.NkiTensor],
    expert_affinities_masked_fp32_sb: nl.NkiTensor,
    t_tile_idx: int,
    tile_T_actual: int,
    tile_T_slice: nl.DynamicSlice,
    tile_H_offset: int,
    weight_H_slice: nl.DynamicSlice,
    TILE_T: int,
    TILE_H: int,
    n_I512_tiles: int,
    perform_accumulation: bool = False,
    accumulation_result_tile: Optional[nl.NkiTensor] = None,
    overwrite_accum_result: bool = False,
    psum_postscale: bool = False,
) -> nl.NkiTensor:
    """Compute one (T-tile, H-tile) of the down projection for a single expert.

    Performs: matmul over I tiles → optional bias add → expert affinity scaling.

    Steps:
    1. MX matmul: act[I, T_tile] × weight[I, H_tile] → out[T_tile, H_tile] in PSUM
    2. PSUM eviction to SBUF, with the bias add fused into the eviction when a bias is present
    3. Expert affinity scaling: element-wise multiply by per-token affinity score, fused with the
       cross-expert accumulation into `accumulation_result_tile` when `perform_accumulation` is set

    Args:
        act_sb: Activation in SBUF [128_I, n_I512_tiles, T].
        act_scale_sb: Activation MX scales in SBUF [128_I, n_I512_tiles, T].
        weight_sb: Weight in SBUF [128_I, n_I512_tiles, H].
        weight_scale_sb: Weight MX scales in SBUF [128_I, n_I512_tiles, H].
        bias_sb: Optional broadcasted bias [TILE_T, H].
        expert_affinities_masked_fp32_sb: Per-token affinity scores [T, n_T_tiles], or None for 1.0.
        t_tile_idx: Index of this T-tile (used for affinity indexing).
        tile_T_actual: Number of valid T elements (handles remainder).
        tile_T_slice: nl.ds slice for T dimension in act/scale tensors.
        tile_H_offset: Absolute H offset of this tile.
        weight_H_slice: nl.ds slice for H dimension in weight tensors.
        TILE_T: Full T tile size.
        TILE_H: Full H tile size.
        n_I512_tiles: Number of I-dimension tiles to contract over.
        perform_accumulation (bool): When True, the affinity-scaled result is written into
            `accumulation_result_tile` instead of `expert_out_tile_sb` (SBUF accumulation path).
        accumulation_result_tile (Optional[nl.NkiTensor]): [tile_T_actual, 1, TILE_H] destination slice of the
            caller's cross-expert accumulator. Required when perform_accumulation=True.
        overwrite_accum_result (bool): When True, `accumulation_result_tile` is overwritten (first expert);
            otherwise this expert's contribution is added to it.
        psum_postscale (bool): Post-scale straight from the bf16 PSUM tile (no eviction copy; the copy is an
            identity, so the result is bit-identical). Ignored with a bias (the bias add is fused into the eviction).

    Returns:
        expert_out_tile_sb: Result in SBUF [TILE_T, TILE_H]. Affinity-scaled only when
            perform_accumulation=False; otherwise the affinity-scaled result lands in
            `accumulation_result_tile` (None with psum_postscale: nothing is evicted).
    """
    kernel_assert(
        not perform_accumulation or accumulation_result_tile != None,
        "accumulation_result_tile is required when perform_accumulation=True",
    )
    out_psum = nl.ndarray((TILE_T, TILE_H), dtype=nl.bfloat16, buffer=nl.psum)
    # psum_postscale: no eviction, so no SBUF tile when the result goes to the accumulator
    from_psum = psum_postscale and bias_sb == None
    expert_out_tile_sb = None
    if not (from_psum and perform_accumulation):
        expert_out_tile_sb = nl.ndarray((TILE_T, TILE_H), dtype=nl.bfloat16, buffer=nl.sbuf)
    # PACK64 down scale is 2D [128_I, H]: the I/512 axis moved into partitions, so tile_i selects its
    # scale group by base partition 4*tile_i rather than by indexing a middle axis.
    scale_is_pack64 = len(weight_scale_sb.shape) == 2
    for tile_i in nl.sequential_range(n_I512_tiles):
        if scale_is_pack64:
            # 4*tile_i is only the WITHIN-quadrant start offset that the ISA's scale_pidx field
            # consumes, so the AP must still span every quadrant -- see the extent note on
            # _SCALE_PACK64_DOWN_AP_EXTENT. The extent is constant so it stays static in this
            # traced loop.
            moving_scale = weight_scale_sb[
                nl.ds(SCALE_P_ELEM_PER_QUADRANT * tile_i, _SCALE_PACK64_DOWN_AP_EXTENT), weight_H_slice
            ]
        else:
            moving_scale = weight_scale_sb[:, tile_i, weight_H_slice]
        nisa.nc_matmul_mx(
            dst=out_psum[:tile_T_actual, :],
            stationary=act_sb[:, tile_i, tile_T_slice],
            moving=weight_sb[:, tile_i, weight_H_slice],
            stationary_scale=act_scale_sb[:, tile_i, tile_T_slice],
            moving_scale=moving_scale,
        )

    if bias_sb != None:
        # Fuse the bias add into the PSUM eviction
        nisa.tensor_tensor(
            dst=expert_out_tile_sb[:tile_T_actual, :],
            data1=out_psum[:tile_T_actual, :],
            op=nl.add,
            data2=bias_sb[:tile_T_actual, tile_H_offset : tile_H_offset + TILE_H],
        )
    elif not from_psum:
        nisa.tensor_copy(dst=expert_out_tile_sb[:tile_T_actual, :], src=out_psum[:tile_T_actual, :])

    # Expert affinity scaling, fused with cross-expert accumulation when accumulating in SBUF
    if expert_affinities_masked_fp32_sb == None:
        affinity_vec = 1.0
    else:
        affinity_vec = expert_affinities_masked_fp32_sb[:tile_T_actual, t_tile_idx : t_tile_idx + 1]
    # psum_postscale: the post-scale reads the bf16 PSUM tile itself (the eviction above was an identity copy)
    tile_ap = (
        out_psum[:tile_T_actual, :] if from_psum else expert_out_tile_sb.ap([[TILE_H, tile_T_actual], [1, TILE_H]])
    )
    if perform_accumulation and not overwrite_accum_result:
        nisa.scalar_tensor_tensor(
            dst=accumulation_result_tile,
            data=tile_ap,
            op0=nl.multiply,
            operand0=affinity_vec,
            op1=nl.add,
            operand1=accumulation_result_tile,
        )
    else:
        nisa.tensor_scalar(
            dst=accumulation_result_tile if perform_accumulation else expert_out_tile_sb[:tile_T_actual, :],
            data=tile_ap,
            op0=nl.multiply,
            operand0=affinity_vec,
            engine=nisa.scalar_engine,
        )
    return expert_out_tile_sb
