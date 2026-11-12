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
Gate/Up projection sub-kernels for hardware MX (MXFP4 / MXFP8 weights).

Each NeuronCore computes the full gate/up projection for one expert: the weights, scales and
biases of the whole intermediate dimension are resident in SBUF and the result is quantized to
MXFP8 for the down projection. PSUM and the evicted activation tile are bfloat16.

These sub-kernels can be used by any algorithm that requires an MX gate/up projection.
"""

from typing import Optional

import nki.isa as nisa
import nki.language as nl

# Common utils
from ..utils.common_types import ActFnType
from ..utils.kernel_assert import kernel_assert
from ..utils.kernel_helpers import div_ceil, get_nl_act_fn_from_type

# Shared MX constants
from .projection_mx_constants import (
    MAX_MATMULT_MX_UNPACKED_CONTRACT_DIM,
    MIN_MATMULT_MX_P_DIM,
    SBUF_QUADRANT_SIZE,
    SCALE_P_ELEM_PER_QUADRANT,
    SCALE_PACK64_N_PARTITIONS,
    SCALE_PACK64_P_PER_QUADRANT,
    _psum_fmax,
    _q_height,
    _q_width,
    pad_to_valid_qmx_partitions,
)


def gate_up_projection_mx(
    input_quant_sb: nl.NkiTensor,
    input_scale_sb: nl.NkiTensor,
    gate_weight_sb: nl.NkiTensor,
    up_weight_sb: nl.NkiTensor,
    gate_weight_scale_sb: nl.NkiTensor,
    up_weight_scale_sb: nl.NkiTensor,
    gate_bias_sb: Optional[nl.NkiTensor],
    up_bias_sb: Optional[nl.NkiTensor],
    gate_clamp_upper_limit: Optional[float] = None,
    gate_clamp_lower_limit: Optional[float] = None,
    up_clamp_upper_limit: Optional[float] = None,
    up_clamp_lower_limit: Optional[float] = None,
    hidden_act_fn: ActFnType = ActFnType.Swish,
    n_tokens=None,
    n_matmul_tokens=None,
) -> tuple[nl.NkiTensor, nl.NkiTensor]:
    """
    Compute gate and up projections with clamping, activation function, and MX quantization.

    Usage:
        Tuned for: mx all-expert MoE algorithm
        Applicable to: any algorithm requiring an mx gate/up projection

    Args:
        input_quant_sb (nl.NkiTensor): [16_H * 8_H, H/512, T], Quantized input in SBUF (4_H packed in x4 dtype).
        input_scale_sb (nl.NkiTensor): [16_H * 8_H, H/512, T], Input scales in SBUF (in leading 4P of each quadrant).
        gate_weight_sb (nl.NkiTensor): [16_H * 8_H, H/512, I/512 * 4_I * 16_I * 8_I], Gate weights in SBUF
            (4_H packed in x4 dtype).
        up_weight_sb (nl.NkiTensor): [16_H * 8_H, H/512, I/512 * 4_I * 16_I * 8_I], Up weights in SBUF
            (4_H packed in x4 dtype).
        gate_weight_scale_sb (nl.NkiTensor): Gate weight scales in SBUF. Unpacked layout is
            [16_H * 8_H, H/512, I/512 * 4_I * 16_I * 8_I] with 4 scale rows in the leading 4P of each
            quadrant; PACK64 is [16_H * 8_H, H/512, I/512 * 16_I * 8_I] with 16 rows per quadrant.
            The layout is detected from the tile shape.
        up_weight_scale_sb (nl.NkiTensor): Up weight scales in SBUF, same layout as
            gate_weight_scale_sb.
        gate_bias_sb (Optional[nl.NkiTensor]): [16_I * 8_I, I/512, 4_I], Gate bias in SBUF.
        up_bias_sb (Optional[nl.NkiTensor]): [16_I * 8_I, I/512, 4_I], Up bias in SBUF.
        gate_clamp_upper_limit (Optional[float]): Upper clamp limit for gate projection.
        gate_clamp_lower_limit (Optional[float]): Lower clamp limit for gate projection.
        up_clamp_upper_limit (Optional[float]): Upper clamp limit for up projection.
        up_clamp_lower_limit (Optional[float]): Lower clamp limit for up projection.
        hidden_act_fn (ActFnType): Activation function type (default: Swish).
        n_tokens (Optional[VirtualRegister]): Runtime token count (1 <= n_tokens <= T). When set, the
            matmuls and PSUM evictions only process the first n_tokens tokens (hardware shape
            registers); the remaining output tokens are zero.
        n_matmul_tokens (Optional[VirtualRegister]): Matmul free size, n_tokens <= n_matmul_tokens <= T.
            Defaults to n_tokens; only the first n_tokens PSUM columns are evicted.

    Returns:
        out_quant_sb (nl.NkiTensor): [16_I * 8_I, I/512, T], Quantized output in SBUF (4_I packed in x4 dtype).
        out_scale_sb (nl.NkiTensor): [16_I * 8_I, I/512, T], Output scales in SBUF (in leading 4P of each quadrant).
    """

    # Step 1: Input validation
    TILE_H, n_H512_tiles, T = input_quant_sb.shape
    TILE_H_, n_H512_tiles_, I_padded = gate_weight_sb.shape
    kernel_assert(
        gate_weight_sb.shape == up_weight_sb.shape,
        f"expected gate and up weights to have the same shapes, got {gate_weight_sb.shape=}, {up_weight_sb.shape=}",
    )
    kernel_assert(
        gate_weight_scale_sb.shape == up_weight_scale_sb.shape,
        f"expected gate and up scales to have the same shapes, "
        f"got {gate_weight_scale_sb.shape=}, {up_weight_scale_sb.shape=}",
    )
    # Validate bias consistency: both must be None or both must have matching shapes
    if gate_bias_sb != None and up_bias_sb != None:
        kernel_assert(
            gate_bias_sb.shape == up_bias_sb.shape,
            f"expected gate and up biases to have the same shapes, got {gate_bias_sb.shape=}, {up_bias_sb.shape=}",
        )
    elif (
        gate_bias_sb != None or up_bias_sb != None
    ):  # pragma: no cover - validation guard; gate/up biases are always both-None or both-present
        kernel_assert(
            False,
            f"expected gate and up biases to be both None or both not None",
        )
    kernel_assert(TILE_H == TILE_H_, f"Expected same number of partitions in input and weight, got {TILE_H}, {TILE_H_}")
    kernel_assert(
        n_H512_tiles == n_H512_tiles_,
        f"Expected same number of H tiles in input and weight, got {n_H512_tiles}, {n_H512_tiles_}",
    )

    # Whether the caller's scale tiles carry the 4_I factor in partitions (PACK64) or in the free
    # dim. Read off the tile shapes so this tracks whatever load_gate_up_weight_scale_bias produced.
    is_pack64 = gate_up_scale_sb_is_pack64(gate_weight_sb, gate_weight_scale_sb)

    # Tiling strategies for T, I
    TILE_T = min(_psum_fmax * 2 // _q_width, T)  # I_4 * TILE_T <= psum_fmax * 2 for bf16 PSUM
    n_T256_tiles = div_ceil(T, TILE_T)
    n_total_I512_tiles = div_ceil(I_padded, MAX_MATMULT_MX_UNPACKED_CONTRACT_DIM)
    I_4, TILE_I = _q_width, nl.tile_size.pmax

    act_fn = get_nl_act_fn_from_type(hidden_act_fn) if hidden_act_fn != None else None
    fold_gate_multiply = _is_gate_multiply_foldable(up_bias_sb, up_clamp_upper_limit, up_clamp_lower_limit)

    # Step 2: Allocate output buffers
    out_shape = (TILE_I, n_total_I512_tiles, T, I_4)
    out_quant_shape = (TILE_I, n_total_I512_tiles, T)
    # A register-sized access is costed against its whole tensor by the SBUF allocator, so the dynamic
    # path keeps one [TILE_I, T, 4_I] tensor per I tile (the extent each eviction actually touches)
    # instead of slicing a single [TILE_I, I/512, T, 4_I] tensor.
    out_sb = None
    out_tile_lst = None
    if n_tokens != None:
        out_tile_lst = [
            nl.ndarray((TILE_I, T, I_4), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(n_total_I512_tiles)
        ]
    else:
        out_sb = nl.ndarray(out_shape, dtype=nl.bfloat16, buffer=nl.sbuf)
    out_quant_sb = nl.ndarray(out_quant_shape, dtype=nl.float8_e4m3fn_x4, buffer=nl.sbuf)
    out_scale_sb = nl.ndarray(out_quant_shape, dtype=nl.uint8, buffer=nl.sbuf)

    if n_tokens != None:
        kernel_assert(
            n_T256_tiles == 1,
            f"dynamic token count supports a single T tile (T <= {TILE_T}), got {T=}",
        )

    # only memset in I padding case
    last_tile_I_size = I_padded - (n_total_I512_tiles - 1) * MAX_MATMULT_MX_UNPACKED_CONTRACT_DIM
    last_I_pdim_sz = last_tile_I_size // _q_width
    if last_I_pdim_sz < TILE_I:
        if out_sb != None:  # the dynamic path zeroes each out_tile_lst tile before its eviction
            nisa.memset(dst=out_sb[...], value=0.0, engine=nisa.gpsimd_engine)
        nisa.memset(dst=out_quant_sb[...], value=0.0, engine=nisa.gpsimd_engine)
        nisa.memset(dst=out_scale_sb[...], value=0.0, engine=nisa.gpsimd_engine)

    """
    Step 3: Fused gate projection, projection clamping (optional), activation function.
    Step 3.1: Compute W_mxfp4/8 (stationary) @ input_mxfp8 (moving).
    """
    for tile_t in nl.sequential_range(n_T256_tiles):
        # T dim slicing, handling case when T tile < 256_T
        tile_T_offset = TILE_T * tile_t
        tile_T_actual = min(TILE_T, T - tile_T_offset)
        tile_T_slice = nl.ds(tile_T_offset, tile_T_actual)

        # Pre-allocate PSUM for all I tiles upfront
        out_psum_lst = []
        for tile_i in range(n_total_I512_tiles):
            out_psum_lst.append(nl.ndarray((TILE_I, I_4, TILE_T), dtype=nl.bfloat16, buffer=nl.psum))

        _projection_matmul_mx(
            out_psum_lst=out_psum_lst,
            weight_sb=gate_weight_sb,
            weight_scale_sb=gate_weight_scale_sb,
            input_quant_sb=input_quant_sb,
            input_scale_sb=input_scale_sb,
            tile_T_slice=tile_T_slice,
            tile_T_actual=tile_T_actual,
            n_H512_tiles=n_H512_tiles,
            n_total_I512_tiles=n_total_I512_tiles,
            I_padded=I_padded,
            is_pack64=is_pack64,
            n_tokens=n_matmul_tokens if n_matmul_tokens != None else n_tokens,
        )

        # Step 3.2: PSUM eviction + bias + clamp + activation (after all H tiles complete)
        for tile_i in range(n_total_I512_tiles):
            cur_tile_I_size = min(
                MAX_MATMULT_MX_UNPACKED_CONTRACT_DIM, I_padded - tile_i * MAX_MATMULT_MX_UNPACKED_CONTRACT_DIM
            )
            cur_I_pdim_sz = cur_tile_I_size // _q_width

            """
            Evict PSUM, applying the bias, clamp and activation function in as few instructions as
            possible (a single instruction without bias or clamp).
            out_sb shape: [TILE_I, n_total_I512_tiles, T, I_4]
            out_psum shape: [TILE_I, I_4, TILE_T]
            gate_bias_sb shape: [TILE_I, n_total_I512_tiles, I_4]
            Use strided access pattern to reorder from [TILE_I, I_4, TILE_T] to [TILE_I, TILE_T, I_4].
            """
            n_evict = tile_T_actual
            if n_tokens != None:
                # Tokens past n_tokens are never written by the evictions; zero them so the static
                # quantize and down projection see zeros rather than stale SBUF contents. Vector engine:
                # it idles until the first eviction, while GpSimd is issuing the weight DMAs.
                nisa.memset(dst=out_tile_lst[tile_i][...], value=0.0, engine=nisa.vector_engine)
                n_evict = n_tokens
            _evict_projection_tile(
                dst=_token_tile_access(out_sb, out_tile_lst, cur_I_pdim_sz, tile_i, tile_T_slice, n_tokens),
                psum_access=out_psum_lst[tile_i].ap([[I_4 * TILE_T, cur_I_pdim_sz], [1, n_evict], [TILE_T, I_4]]),
                bias_ap=(
                    _col_broadcast_access(gate_bias_sb, cur_I_pdim_sz, tile_T_actual, I_4, tile_i)
                    if gate_bias_sb != None
                    else None
                ),
                clamp_upper_limit=gate_clamp_upper_limit,
                clamp_lower_limit=gate_clamp_lower_limit,
                act_fn=act_fn,
            )

    """
    Step 4: Fused up projection, projection clamp (optional), gate * up, MX quantization.
    Step 4.1: Compute W_mxfp4/8 (stationary) @ input_mxfp8 (moving).
    """
    for tile_t in nl.sequential_range(n_T256_tiles):
        # T dim slicing, handling case when T tile < 256_T
        tile_T_offset = TILE_T * tile_t
        tile_T_actual = min(TILE_T, T - tile_T_offset)
        tile_T_slice = nl.ds(tile_T_offset, tile_T_actual)

        # Pre-allocate PSUM for all I tiles upfront (enables T → H → I → 4_I loop order)
        up_psum_lst = []
        for tile_i in range(n_total_I512_tiles):
            up_psum_lst.append(nl.ndarray((TILE_I, I_4, TILE_T), dtype=nl.bfloat16, buffer=nl.psum))

        # Matmul compute with loop order H → I → 4_I (same input H-slice reused across all I tiles)
        _projection_matmul_mx(
            out_psum_lst=up_psum_lst,
            weight_sb=up_weight_sb,
            weight_scale_sb=up_weight_scale_sb,
            input_quant_sb=input_quant_sb,
            input_scale_sb=input_scale_sb,
            tile_T_slice=tile_T_slice,
            tile_T_actual=tile_T_actual,
            n_H512_tiles=n_H512_tiles,
            n_total_I512_tiles=n_total_I512_tiles,
            I_padded=I_padded,
            is_pack64=is_pack64,
            n_tokens=n_matmul_tokens if n_matmul_tokens != None else n_tokens,
        )

        # Step 4.2: PSUM eviction + bias + clamp + gate*up + quantize (after all H tiles complete)
        for tile_i in range(n_total_I512_tiles):
            cur_tile_I_size = min(
                MAX_MATMULT_MX_UNPACKED_CONTRACT_DIM, I_padded - tile_i * MAX_MATMULT_MX_UNPACKED_CONTRACT_DIM
            )
            cur_I_pdim_sz = cur_tile_I_size // _q_width

            """
            Evict PSUM, applying the bias, clamp and the gate * up multiply in as few instructions as
            possible (a single instruction without bias or clamp).
            intermediate_tile_sb shape: [TILE_I, 1, TILE_T, I_4]
            out_psum shape: [TILE_I, I_4, TILE_T]
            up_bias_sb shape: [TILE_I, n_total_I512_tiles, I_4]
            Use strided access pattern to reorder from [TILE_I, I_4, TILE_T] to [TILE_I, TILE_T, I_4].
            """
            # Scratch is only needed when the gate * up multiply cannot be folded into the eviction instruction
            intermediate_tile_sb = None
            if not fold_gate_multiply:
                intermediate_tile_sb = nl.ndarray((TILE_I, 1, TILE_T, I_4), dtype=nl.bfloat16, buffer=nl.sbuf)

            n_evict = tile_T_actual if n_tokens == None else n_tokens
            _evict_projection_tile(
                dst=_token_tile_access(out_sb, out_tile_lst, cur_I_pdim_sz, tile_i, tile_T_slice, n_tokens),
                psum_access=up_psum_lst[tile_i].ap([[I_4 * TILE_T, cur_I_pdim_sz], [1, n_evict], [TILE_T, I_4]]),
                bias_ap=(
                    _col_broadcast_access(up_bias_sb, cur_I_pdim_sz, tile_T_actual, I_4, tile_i)
                    if up_bias_sb != None
                    else None
                ),
                clamp_upper_limit=up_clamp_upper_limit,
                clamp_lower_limit=up_clamp_lower_limit,
                mul_operand=_token_tile_access(out_sb, out_tile_lst, cur_I_pdim_sz, tile_i, tile_T_slice, n_tokens),
                scratch=(
                    intermediate_tile_sb[:cur_I_pdim_sz, 0, :tile_T_actual, :] if intermediate_tile_sb != None else None
                ),
            )

            # Step 4.3: MX quantize combined gate * up tile
            # Pad partition count to valid quantize_mx size {32, 64, 96, 128}.
            # Extra zero-padded partitions are harmless: downstream weight is zero-padded.
            qmx_I_pdim_sz = pad_to_valid_qmx_partitions(cur_I_pdim_sz)
            nisa.quantize_mx(
                src=(
                    out_sb[:qmx_I_pdim_sz, tile_i, tile_T_slice, :]
                    if n_tokens == None
                    else out_tile_lst[tile_i][:qmx_I_pdim_sz, tile_T_slice, :]
                ),
                dst=out_quant_sb[:qmx_I_pdim_sz, tile_i, tile_T_slice],
                dst_scale=out_scale_sb[:qmx_I_pdim_sz, tile_i, tile_T_slice],
            )

    return out_quant_sb, out_scale_sb


def gate_up_scale_is_pack64(scale: nl.NkiTensor) -> bool:
    """True when the gate/up MX scale tensor in HBM is in the PACK64 layout.

    Unpacked is ``[E_L, 16, 2, H/512, I]`` and PACK64 is ``[E_L, 2, 64, H/512, I/4]``, so dims 1-2
    identify the layout unambiguously. The layout is read off the tensor rather than a flag so the
    caller's layout and the kernel's load can never disagree: whichever one the caller supplies is
    the one the kernel loads.

    Args:
        scale (nl.NkiTensor): Gate/up MX scale tensor from HBM.

    Returns:
        bool: True for the PACK64 layout, False for the unpacked layout.
    """
    return scale.shape[1] == 2 and scale.shape[2] == SCALE_PACK64_N_PARTITIONS


def gate_up_scale_sb_is_pack64(weight_sb: nl.NkiTensor, scale_sb: nl.NkiTensor) -> bool:
    """True when an SBUF gate/up scale tile holds the PACK64 layout.

    PACK64 moves the ``4_I`` factor into partitions, so the scale tile's free dim is ``_q_width``
    times narrower than the weight tile's; the unpacked layout matches the weight tile exactly.

    Args:
        weight_sb (nl.NkiTensor): [128_H, H/512, I_padded] weight tile in SBUF.
        scale_sb (nl.NkiTensor): Scale tile in SBUF, either layout.

    Returns:
        bool: True for the PACK64 layout, False for the unpacked layout.
    """
    return scale_sb.shape[-1] * _q_width == weight_sb.shape[-1]


def load_gate_up_weight_scale_bias(
    weight: nl.NkiTensor,
    scale: nl.NkiTensor,
    bias: Optional[nl.NkiTensor],
    expert_idx: int,
    gate_or_up_idx: int,
    H: int,
    n_I512_tiles: int,
    I: int,
    I_padded: int,
    dma_handles: Optional[list] = None,
) -> tuple[nl.NkiTensor, nl.NkiTensor, Optional[nl.NkiTensor]]:
    """
    Load gate or up projection weight, scale, and bias (optional) for one expert using static DMA.

    Args:
        weight (nl.NkiTensor): [E_L, 128_H, 2, H/512, I], Gate or up projection weight tensor from HBM
            (fused gate/up weights), 4_H packed in x4 dtype.
        scale (nl.NkiTensor): Gate or up projection MX scale tensor from HBM (fused gate/up scales),
            uint8 MX scales, in either layout: unpacked [E_L, 16_H, 2, H/512, I] or PACK64
            [E_L, 2, 64_H, H/512, I/4]. PACK64 loads 4x fewer, 4x narrower partitions per DMA and
            reserves 4x less SBUF; it requires I to be a multiple of 512.
        bias (Optional[nl.NkiTensor]): [E_L, 128_I, 2, I/512, 4_I], Optional gate or up projection bias
            tensor from HBM (fused gate/up biases).
        expert_idx (int): Index of the current expert to load.
        gate_or_up_idx (int): Index to select gate (0) or up (1) projection from fused tensor.
        H (int): Hidden dimension size.
        n_I512_tiles (int): Number of I/512 tiles.
        I (int): Intermediate dimension size.
        I_padded (int): I padded up to a multiple of 8 for the nc_matmul_mx even free-dim constraint.
        dma_handles (Optional[list]): When given, the weight DMA's instruction is appended (for ordering).

    Returns:
        weight_sb (nl.NkiTensor): [128_H, H/512, I_padded], Weight in SBUF (4_H packed in x4 dtype).
        scale_sb (nl.NkiTensor): Scales in SBUF. [128_H, H/512, I_padded] with 4 rows per quadrant
            for the unpacked layout, [128_H, H/512, I_padded/4] with 16 rows per quadrant for PACK64.
        bias_sb (Optional[nl.NkiTensor]): [128_I, n_I512_tiles, 4_I], Bias in SBUF (None when bias not provided).

    Notes:
        - Based on experiments, static DMA demonstrates better performance
    """

    # Calculate shapes / tiling
    I_buf = I_padded
    kernel_assert(
        I_buf % MIN_MATMULT_MX_P_DIM == 0,
        f"Expected padded I divisible by {MIN_MATMULT_MX_P_DIM} for the nc_matmul_mx free-dim constraint, {I_buf=}.",
    )
    needs_padding = I_buf > I
    pmax = nl.tile_size.pmax
    TILE_H, n_H512_tiles = pmax, H // MAX_MATMULT_MX_UNPACKED_CONTRACT_DIM
    TILE_I, I_4 = pmax, _q_width
    weight_sb_shape = (TILE_H, n_H512_tiles, I_buf)
    bias_sb_shape = (TILE_I, n_I512_tiles, I_4)
    is_bias = bias != None

    # Allocate buffers. PACK64 keeps the scale in 64 partitions with the 4_I factor folded out of
    # the free dim, so its tile is _q_width narrower than the weight tile (4x less SBUF/partition).
    is_pack64 = gate_up_scale_is_pack64(scale)
    scale_sb_shape = (TILE_H, n_H512_tiles, I_buf // _q_width) if is_pack64 else weight_sb_shape
    base_weight = weight
    weight_sb = nl.ndarray(weight_sb_shape, dtype=base_weight.dtype, buffer=nl.sbuf)
    scale_sb = nl.ndarray(scale_sb_shape, dtype=scale.dtype, buffer=nl.sbuf)
    bias_sb = nl.ndarray(bias_sb_shape, dtype=bias.dtype, buffer=nl.sbuf) if is_bias else None

    # Load weight: index expert and gate/up, then slice the I dimension
    # Shape: [E_L, 128_H, 2, H/512, I] -> [128_H, H/512, I] -> padded to [128_H, H/512, I_buf]
    weight_view = (
        base_weight.select(dim=0, index=expert_idx).select(dim=1, index=gate_or_up_idx).slice(dim=2, start=0, end=I)
    )
    if needs_padding:
        nisa.memset(dst=weight_sb[...], value=0, engine=nisa.gpsimd_engine)
        weight_dma = nisa.dma_copy(src=weight_view, dst=weight_sb[:, :, :I], dge_mode=nisa.dge_mode.none)
    else:
        weight_dma = nisa.dma_copy(src=weight_view, dst=weight_sb[...], dge_mode=nisa.dge_mode.none)
    if dma_handles != None:
        dma_handles.append(weight_dma)
    weight_sb = weight_sb.view(weight.dtype)

    """
    Load scale: index expert and gate/up, then slice the I dimension.
    Shape (unpacked): [E_L, 16_H, 2, H/512, I]      -> [16_H, H/512, I], 4 rows per quadrant
    Shape (PACK64):   [E_L, 2, 64_H, H/512, I/4]    -> [64_H, H/512, I/4], 16 rows per quadrant
    A scale must sit in the quadrant its scaling group came from (ISA rule), and a 32-partition
    destination stride is inexpressible, so one DMA per quadrant is the floor for either layout.
    """
    n_scale_partitions = TILE_H // _q_height
    n_quadrants_needed = div_ceil(n_scale_partitions, SCALE_P_ELEM_PER_QUADRANT)

    if needs_padding:
        nisa.memset(dst=scale_sb[...], value=0.0, engine=nisa.gpsimd_engine)

    if is_pack64:
        # PACK64 folds 4_I into partitions by splitting I into (I/512) tiles of 4x128, so it only
        # applies when I fills whole 512-element tiles.
        kernel_assert(
            I % MAX_MATMULT_MX_UNPACKED_CONTRACT_DIM == 0,
            f"PACK64 gate/up scale requires I divisible by {MAX_MATMULT_MX_UNPACKED_CONTRACT_DIM}, got {I=}. "
            f"Supply the unpacked [E_L, 16, 2, H/512, I] scale instead.",
        )
        # Copy 16 contiguous HBM rows into partitions 32q..32q+15 -- the same bytes as the
        # 4 x [4 P, I-wide] tiles it replaces, but 4x fewer partitions and 4x narrower.
        I_pack = I // _q_width
        for quadrant_idx in nl.affine_range(n_quadrants_needed):
            pack_view = (
                scale.select(dim=0, index=expert_idx)
                .select(dim=0, index=gate_or_up_idx)
                .slice(
                    dim=0,
                    start=SCALE_PACK64_P_PER_QUADRANT * quadrant_idx,
                    end=SCALE_PACK64_P_PER_QUADRANT * (quadrant_idx + 1),
                )
            )
            nisa.dma_copy(
                src=pack_view,
                dst=scale_sb[
                    nl.ds(SBUF_QUADRANT_SIZE * quadrant_idx, SCALE_PACK64_P_PER_QUADRANT),
                    :,
                    :I_pack,
                ],
                dge_mode=nisa.dge_mode.none,
            )
    else:
        DMA_FREE_DIM_TILE = 1024
        n_I_tiles = div_ceil(I, DMA_FREE_DIM_TILE)

        for quadrant_idx in nl.affine_range(n_quadrants_needed):
            for i_tile_idx in nl.affine_range(n_I_tiles):
                i_start = i_tile_idx * DMA_FREE_DIM_TILE
                i_size = min(DMA_FREE_DIM_TILE, I - i_start)
                scale_view = (
                    scale.select(dim=0, index=expert_idx)
                    .slice(
                        dim=0,
                        start=SCALE_P_ELEM_PER_QUADRANT * quadrant_idx,
                        end=SCALE_P_ELEM_PER_QUADRANT * (quadrant_idx + 1),
                    )
                    .select(dim=1, index=gate_or_up_idx)
                    .slice(dim=2, start=i_start, end=i_start + i_size)
                )
                nisa.dma_copy(
                    src=scale_view,
                    dst=scale_sb[
                        nl.ds(SBUF_QUADRANT_SIZE * quadrant_idx, SCALE_P_ELEM_PER_QUADRANT),
                        :,
                        i_start : i_start + i_size,
                    ],
                    dge_mode=nisa.dge_mode.none,
                )

    # Load bias: index expert and gate/up
    # Shape: [E_L, I_p, 2, I/512, 4_I] -> [I_p, n_I512_tiles, 4_I] -> padded to [128_I, n_I512_tiles, 4_I]
    if is_bias:
        I_p_bias_in_hbm = bias.shape[1]

        if I_p_bias_in_hbm < TILE_I:
            nisa.memset(dst=bias_sb[...], value=0.0, engine=nisa.gpsimd_engine)

        bias_view = bias.select(dim=0, index=expert_idx).select(dim=1, index=gate_or_up_idx)
        nisa.dma_copy(
            src=bias_view,
            dst=bias_sb[:I_p_bias_in_hbm, :, :],
            dge_mode=nisa.dge_mode.none,
        )

    return weight_sb, scale_sb, bias_sb


def _partition_pitch(tensor: nl.NkiTensor) -> int:
    """Number of elements per SBUF partition, i.e. the partition-dim stride of an access pattern on `tensor`."""
    pitch = 1
    for dim_size in tensor.shape[1:]:
        pitch *= dim_size
    return pitch


def _col_broadcast_access(
    tensor: nl.NkiTensor,
    n_partitions: int,
    n_tokens: int,
    i_4: int,
    tile_i: int,
) -> nl.NkiTensor:
    """
    Access pattern for a per-output-column bias tile, broadcast over the T dim.

    Columns of one I512 tile are contiguous, so the tile is traversed as [TILE_I, T, 4_I] with a stride-0 T dim
    to match the [TILE_I, T, 4_I] layout of the evicted projection output.
    """
    return tensor.ap([[_partition_pitch(tensor), n_partitions], [0, n_tokens], [1, i_4]], offset=tile_i * i_4)


def _token_tile_access(out_sb, out_tile_lst, I_pdim_sz: int, tile_i: int, tile_T_slice, n_tokens) -> nl.NkiTensor:
    """
    [I_pdim_sz, T, 4_I] view of I tile `tile_i`: a slice of `out_sb` ([TILE_I, n_I512_tiles, T, 4_I])
    on the static path, or the first `n_tokens` tokens of `out_tile_lst[tile_i]` ([TILE_I, T, 4_I])
    when `n_tokens` is a runtime register.
    """
    if n_tokens == None:
        return out_sb[:I_pdim_sz, tile_i, tile_T_slice, :]
    tile = out_tile_lst[tile_i]
    _, T, I_4 = tile.shape
    return tile.ap(pattern=[[T * I_4, I_pdim_sz], [I_4, n_tokens], [1, I_4]], offset=tile_T_slice.start * I_4)


def _evict_projection_tile(
    dst: nl.NkiTensor,
    psum_access: nl.NkiTensor,
    bias_ap: Optional[nl.NkiTensor] = None,
    clamp_upper_limit: Optional[float] = None,
    clamp_lower_limit: Optional[float] = None,
    act_fn=None,
    mul_operand: Optional[nl.NkiTensor] = None,
    scratch: Optional[nl.NkiTensor] = None,
) -> None:
    """
    Evict one projection PSUM tile to SBUF, computing `dst = act_fn(clamp(psum + bias)) * mul_operand` in as
    few instructions as possible.

    `psum_access` must already carry the [TILE_I, 4_I, T] -> [TILE_I, T, 4_I] reorder access pattern, and the
    bias access pattern must already broadcast over the dims it is constant in, so that every instruction
    below covers a full tile instead of looping over the 4_I quad lanes.

    With no bias and no clamping this is a single instruction (reorder + activation function, or reorder +
    gate * up multiply).

    Args:
        dst (nl.NkiTensor): [TILE_I, T, 4_I] SBUF tile to write.
        psum_access (nl.NkiTensor): [TILE_I, T, 4_I] reordered view of the projection PSUM tile.
        bias_ap (Optional[nl.NkiTensor]): Projection bias access.
        clamp_upper_limit (Optional[float]): Upper clamp limit, applied before `act_fn`.
        clamp_lower_limit (Optional[float]): Lower clamp limit, applied before `act_fn`.
        act_fn: Activation function applied last (None to skip).
        mul_operand (Optional[nl.NkiTensor]): Tile the result is multiplied by, used to fold the gate * up
            multiply into the up projection eviction. Must alias `dst`.
        scratch (Optional[nl.NkiTensor]): Tile with the same shape as `dst`, required when `mul_operand` is set
            and the multiply cannot be folded into the eviction (see `_is_gate_multiply_foldable`).

    Returns:
        None. Result is written to `dst`.
    """
    kernel_assert(
        act_fn == None or mul_operand == None,
        f"expected at most one of act_fn / mul_operand to be fused into the eviction",
    )
    has_clamp = clamp_upper_limit != None or clamp_lower_limit != None
    # `dst` still holds the gate result when a gate * up multiply is pending, so build the up tile in `scratch`
    work = scratch if scratch != None else dst
    # The trailing activation function or gate * up multiply only folds into the eviction when the eviction
    # itself collapses to a single instruction with a free operand slot
    is_single_instruction = bias_ap == None and not has_clamp

    if is_single_instruction and act_fn != None:
        nisa.activation(dst=work, op=act_fn, data=psum_access, scale=1.0)
    elif is_single_instruction and mul_operand != None and scratch == None:
        nisa.tensor_tensor(dst=work, data1=psum_access, op=nl.multiply, data2=mul_operand)
        return
    else:
        # Apply the bias, then clamp and apply the activation function separately
        if bias_ap != None:
            nisa.tensor_tensor(dst=work, data1=psum_access, op=nl.add, data2=bias_ap)
        else:
            nisa.tensor_copy(dst=work, src=psum_access)

        _clamp_tensor(tensor=work, clamp_upper_limit=clamp_upper_limit, clamp_lower_limit=clamp_lower_limit)

        if act_fn != None:
            nisa.activation(dst=work, op=act_fn, data=work)

    if mul_operand != None:
        nisa.tensor_tensor(dst=dst, data1=mul_operand, op=nl.multiply, data2=work)


def _is_gate_multiply_foldable(
    up_bias_sb: Optional[nl.NkiTensor],
    up_clamp_upper_limit: Optional[float],
    up_clamp_lower_limit: Optional[float],
) -> bool:
    """
    Whether the gate * up multiply can be folded into the up projection PSUM eviction instruction.

    A bias or a clamp each need an extra full-tile operand, forcing the up tile to be materialized in
    scratch first.
    """
    return up_bias_sb == None and up_clamp_upper_limit == None and up_clamp_lower_limit == None


def _clamp_tensor(
    tensor: nl.NkiTensor,
    clamp_upper_limit: Optional[float],
    clamp_lower_limit: Optional[float],
) -> None:
    """
    Apply optional clamping to a tensor in-place.

    Args:
        tensor (nl.NkiTensor): Tensor slice to clamp.
        clamp_upper_limit (Optional[float]): Upper clamp limit (None to skip).
        clamp_lower_limit (Optional[float]): Lower clamp limit (None to skip).

    Returns:
        None. Tensor is clamped in-place.
    """
    if clamp_upper_limit != None or clamp_lower_limit != None:
        nisa.tensor_scalar(
            dst=tensor,
            data=tensor,
            op0=nl.minimum if clamp_upper_limit != None else None,
            operand0=clamp_upper_limit,
            op1=nl.maximum if clamp_lower_limit != None else None,
            operand1=clamp_lower_limit,
        )


def _projection_matmul_mx(
    out_psum_lst: list,
    weight_sb: nl.NkiTensor,
    weight_scale_sb: nl.NkiTensor,
    input_quant_sb: nl.NkiTensor,
    input_scale_sb: nl.NkiTensor,
    tile_T_slice,
    tile_T_actual: int,
    n_H512_tiles: int,
    n_total_I512_tiles: int,
    I_padded: int,
    is_pack64: bool,
    n_tokens=None,
) -> None:
    """
    Perform MX matmul accumulation over H tiles and I tiles.
    Loop order: H → I → 4_I for better input tensor reuse with larger T.

    Args:
        out_psum_lst (list): List of PSUM buffers, one per I512 tile.
        weight_sb (nl.NkiTensor): [128_H, H/512, I_padded], Weight tensor in SBUF.
        weight_scale_sb (nl.NkiTensor): [128_H, H/512, I_padded], Weight MX scales in SBUF.
        input_quant_sb (nl.NkiTensor): [128_H, H/512, T], Quantized input in SBUF.
        input_scale_sb (nl.NkiTensor): [128_H, H/512, T], Input scale in SBUF.
        tile_T_slice: T dimension slice descriptor.
        tile_T_actual (int): Actual T tile size (may be < 256 for last tile).
        n_H512_tiles (int): Number of H/512 tiles.
        n_total_I512_tiles (int): Number of I/512 tiles.
        I_padded (int): Intermediate dimension size, padded to a multiple of 8.
        is_pack64 (bool): True when weight_scale_sb holds the PACK64 layout, i.e. the 4_I factor is
            in partitions at within-quadrant offsets {0, 4, 8, 12} rather than in the free dim.
        n_tokens (Optional[VirtualRegister]): Runtime token count; when set the moving and PSUM
            free size come from a shape register. The moving scale stays static: the hardware
            walks it alongside the moving data.

    Returns:
        None. Results are accumulated in-place into out_psum_lst buffers.
    """
    for tile_h in range(n_H512_tiles):
        for tile_i in range(n_total_I512_tiles):
            cur_tile_I_size = min(
                MAX_MATMULT_MX_UNPACKED_CONTRACT_DIM, I_padded - tile_i * MAX_MATMULT_MX_UNPACKED_CONTRACT_DIM
            )
            cur_I128_tile_sz = cur_tile_I_size // _q_width
            for q_width_I_idx in range(_q_width):
                weight_I_offset = tile_i * MAX_MATMULT_MX_UNPACKED_CONTRACT_DIM + q_width_I_idx * cur_I128_tile_sz
                weight_I_slice = nl.ds(weight_I_offset, cur_I128_tile_sz)
                if is_pack64:
                    # PACK64: the 4_I factor lives in PARTITIONS at within-quadrant offsets
                    # {0, 4, 8, 12}, so this group is selected by base partition 4*q_width_I_idx and
                    # the free index collapses from (tile_i*512 + q_width_I_idx*128) to
                    # tile_i*cur_I128_tile_sz.
                    #
                    # The partition EXTENT is load-bearing: 4*q_width_I_idx is only the
                    # WITHIN-quadrant start offset that the ISA's scale_pidx field consumes, so the
                    # AP must still span all four quadrants. An extent of 4 names one quadrant's
                    # rows, compiles clean, and silently mis-scales the other three quadrants'
                    # contraction blocks -- it cannot be caught by a compile-only test.
                    p_base = SCALE_P_ELEM_PER_QUADRANT * q_width_I_idx
                    scale_ap = weight_scale_sb[
                        nl.ds(p_base, nl.tile_size.pmax - p_base),
                        tile_h,
                        nl.ds(tile_i * cur_I128_tile_sz, cur_I128_tile_sz),
                    ]
                else:
                    scale_ap = weight_scale_sb[:, tile_h, weight_I_slice]
                dst = out_psum_lst[tile_i][:cur_I128_tile_sz, q_width_I_idx, :tile_T_actual]
                moving = input_quant_sb[:, tile_h, tile_T_slice]
                if n_tokens != None:
                    psum_I_4, psum_T = out_psum_lst[tile_i].shape[1:]
                    dst = out_psum_lst[tile_i].ap(
                        pattern=[[psum_I_4 * psum_T, cur_I128_tile_sz], [1, n_tokens]],
                        offset=q_width_I_idx * psum_T,
                    )
                    in_H512_tiles, in_T = input_quant_sb.shape[1:]
                    moving = input_quant_sb.ap(
                        pattern=[[in_H512_tiles * in_T, input_quant_sb.shape[0]], [1, n_tokens]],
                        offset=tile_h * in_T + tile_T_slice.start,
                    )
                nisa.nc_matmul_mx(
                    dst=dst,
                    stationary=weight_sb[:, tile_h, weight_I_slice],
                    moving=moving,
                    stationary_scale=scale_ap,
                    moving_scale=input_scale_sb[:, tile_h, tile_T_slice],
                )
