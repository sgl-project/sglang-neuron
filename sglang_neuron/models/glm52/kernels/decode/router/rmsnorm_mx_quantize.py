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

"""Fused RMSNorm + hardware MX quantization sub-kernel for MoE token generation.

Single configuration: RMSNorm (no residual add), hardware ``nisa.quantize_mx`` only, one
NeuronCore (LNC=1, no token sharding), and the unpacked ``(quant, scale)`` output pair in HBM
that the MoE expert kernel consumes pre-quantized.
"""

from dataclasses import dataclass

import nki
import nki.isa as nisa
import nki.language as nl

from ..mlp.projection_mx_constants import _q_width
from ..utils.kernel_assert import kernel_assert

_ERROR_PREFIX = "[RMSNorm MX Quantize TKG Kernel]"

# One uint8 hardware micro-scale per MX block.
_MX_SCALE_DTYPE = nl.uint8

# nisa.quantize_mx packs 4 values into one element of these dtypes.
_SUPPORTED_QMX_OUTPUT_DTYPES = [nl.float8_e4m3fn_x4, nl.float8_e5m2_x4]


def get_token_tile_size(num_tokens: int) -> int:
    """
    Determine the tile size for processing tokens.

    Finds the largest power-of-2 tile size (8-64) that evenly divides num_tokens.
    Used to tile the token (batch * sequence) dimension for efficient processing.

    Args:
        num_tokens (int): Number of tokens to process (BxS).

    Returns:
        int: Tile size for iterating over tokens.

    Notes:
        - Falls back to num_tokens if no power-of-2 tile size in [8, 64] divides evenly
    """
    MIN_TILE_SIZE = 8
    MAX_TILE_SIZE = 64

    tile_size = MAX_TILE_SIZE
    while tile_size >= MIN_TILE_SIZE:
        if num_tokens % tile_size == 0:
            return tile_size
        tile_size //= 2

    return num_tokens


@dataclass
class RmsNormMXQuantizeDims(nl.NKIObject):
    """Dimension and hardware parameters derived from the input shape."""

    input_shape: tuple

    # Computed fields
    H: int = None
    H0: int = None
    H1: int = None
    BxS: int = None
    num_H512_tiles: int = None
    inter_dtype: object = None

    def __post_init__(self):
        self.H0 = nl.tile_size.pmax
        self.inter_dtype = nl.float32

        # NOTE: NKI does not support unpacking complex var into multiple class var assignments
        B, S, H = self.input_shape
        self.H = H
        self.H1 = self.H // self.H0
        self.BxS = B * S
        # nisa.quantize_mx consumes 512 hidden elements per call: H0 partitions of _q_width values.
        self.num_H512_tiles = self.H1 // _q_width


@dataclass
class RmsNormMXQuantizeConfig(nl.NKIObject):
    """Hyperparameter and tiling configuration for the RMSNorm + MX quantization TKG kernel."""

    eps: float
    BxS_tile_size: int
    num_BxS_tiles: int


def validate_rmsnorm_mx_quantize_tkg(
    input_shape,
    gamma_shape,
    output_shape,
    output_quant_dtype,
    output_quant_shape,
    output_quant_buffer,
    output_scale_shape,
    output_scale_buffer,
    output_dtype,
    eps,
):
    """
    Reject the launches and inputs this single-configuration kernel does not handle, and build
    RmsNormMXQuantizeDims / RmsNormMXQuantizeConfig.

    Returns:
        Tuple[RmsNormMXQuantizeDims, RmsNormMXQuantizeConfig]: (dims, cfg)
    """
    dims = RmsNormMXQuantizeDims(input_shape=input_shape)
    H0, H1, BxS, H = dims.H0, dims.H1, dims.BxS, dims.H

    # Single NeuronCore only: there is no work split, so a second program would recompute the
    # whole result and race with the first one on the shared-HBM outputs.
    kernel_assert(
        nl.program_ndim() == 0 or nl.program_ndim() == 1,
        f"{_ERROR_PREFIX} only no specialization or specialization along one axis is supported, got "
        f"nl.program_ndim()={nl.program_ndim()}",
    )
    n_programs = nl.num_programs(axes=0) if nl.program_ndim() != 0 else 1
    kernel_assert(
        n_programs == 1,
        f"{_ERROR_PREFIX} RMSNorm MX quantize runs on a single NeuronCore (LNC=1), got {n_programs} programs",
    )

    # HW validation
    kernel_assert(
        nisa.get_nc_version() >= nisa.nc_version.gen4,
        f"{_ERROR_PREFIX} hardware MX quantization only supports gen4+ (Trn3+) but got {nisa.get_nc_version()=}",
    )

    # Shape validation
    kernel_assert(H % H0 == 0, f"{_ERROR_PREFIX} H must be divisible by {H0}, got H={H}")
    kernel_assert(
        H1 % _q_width == 0,
        f"{_ERROR_PREFIX} the MX swizzle tiles H by {H0 * _q_width}, so H1={H1} must be divisible by {_q_width}",
    )
    kernel_assert(gamma_shape == (1, H), f"{_ERROR_PREFIX} malformed shape of gamma {gamma_shape}")
    kernel_assert(output_shape == (H0, BxS, H1), f"{_ERROR_PREFIX} expected output.shape = {(H0, BxS, H1)}")
    kernel_assert(
        output_dtype in [nl.float16, nl.bfloat16], f"{_ERROR_PREFIX} output.dtype must be float16 or bfloat16"
    )

    # Quantized output pair: unpacked [H0, H/512, BxS] quant and uint8 scale, both in HBM. This is
    # the layout the MoE expert kernel consumes pre-quantized; the packed [BxS, H + H/4] A2A-v form
    # and the SBUF-resident form are not supported.
    kernel_assert(
        output_quant_shape == (H0, dims.num_H512_tiles, BxS),
        f"{_ERROR_PREFIX} expected output_quant.shape = {(H0, dims.num_H512_tiles, BxS)}, got {output_quant_shape=}",
    )
    kernel_assert(
        output_quant_shape == output_scale_shape,
        f"{_ERROR_PREFIX} expected same shape for output_quant and output_scale, but got {output_quant_shape=}, "
        f"{output_scale_shape=}",
    )
    kernel_assert(
        (output_quant_buffer == nl.sbuf) == (output_scale_buffer == nl.sbuf),
        f"{_ERROR_PREFIX} output_quant and output_scale must both be in HBM or both in SBUF, but got "
        f"{output_quant_buffer=}, {output_scale_buffer=}",
    )
    kernel_assert(
        output_quant_dtype in _SUPPORTED_QMX_OUTPUT_DTYPES,
        f"{_ERROR_PREFIX} expected output_quant.dtype in {_SUPPORTED_QMX_OUTPUT_DTYPES}, but got {output_quant_dtype=}",
    )

    # Tiling strategy: one token tile, which the swizzle moves with one copy and quantizes with one quantize_mx
    BxS_tile_size = get_token_tile_size(BxS)
    num_BxS_tiles = BxS // BxS_tile_size
    kernel_assert(BxS % BxS_tile_size == 0, f"{_ERROR_PREFIX} BxS must be divisible by BxS_tile_size")
    kernel_assert(
        num_BxS_tiles == 1,
        f"{_ERROR_PREFIX} the MX swizzle takes one token tile, got BxS={BxS} in {num_BxS_tiles} tiles",
    )

    cfg = RmsNormMXQuantizeConfig(
        eps=eps,
        BxS_tile_size=BxS_tile_size,
        num_BxS_tiles=num_BxS_tiles,
    )

    return dims, cfg


@nki.jit
def rmsnorm_mx_quantize_tkg(
    input: nl.NkiTensor,
    gamma: nl.NkiTensor,
    output: nl.NkiTensor,
    output_quant: nl.NkiTensor,
    output_scale: nl.NkiTensor,
    eps: float = 1e-6,
    input_parts: tuple = None,
):
    """
    Fused RMSNorm + hardware MX quantization for token generation.

    Dimensions:
        B: Batch size
        S: Sequence length
        H: Hidden dimension size (must be a multiple of 512)
        H0: Partition dimension (128)
        H1: H // H0

    Args:
        input (nl.NkiTensor): [B, S, H], input tensor on HBM.
        gamma (nl.NkiTensor): [1, H], RMSNorm scaling weights on HBM.
        output (nl.NkiTensor): [H0, B*S, H1], FP16/BF16 normalized output tensor in SBUF. This is
            the router's input, in the layout the MX swizzle emits.
        output_quant (nl.NkiTensor): [H0, H/512, B*S], FP8x4 MX-quantized output tensor in HBM, or in
            SBUF (together with output_scale) to skip the HBM spill.
        output_scale (nl.NkiTensor): [H0, H/512, B*S], uint8 MX scale output tensor in HBM (or SBUF). Only 16
            of the 128 partitions carry a scale -- rows [0-3, 32-35, 64-67, 96-99], the leading 4
            rows of each SBUF quadrant, which is what nisa.quantize_mx writes and nc_matmul_mx
            reads. The other 112 partitions are zeroed.
        eps (float): Epsilon for numerical stability. Default is 1e-6.
        input_parts (tuple): Optional ``(a, b)``, two [B, S, H] HBM tensors of ``input``'s dtype whose elementwise
            sum is ``input``: both are transpose-loaded and added (``a + b``, one Vector op) in place of loading
            ``input``, which is then read for its shape only. The giga stack's residual add computes the same
            ``a + b``, so the tile is identical. (default: None)

    Returns:
        list: [output, output_quant, output_scale]. Results are written into the pre-allocated
            output tensors; the return value is for convenience.

    Notes:
        - Runs on a single NeuronCore (LNC=1); a multi-program launch is rejected.
        - Output tensors must be pre-allocated.

    Pseudocode:
        # For each token tile:
        squared = input ** 2
        rms = sqrt(mean(squared) + eps)
        output = (input * gamma) / rms
        output_quant, output_scale = quantize_mx(swizzle(output))
    """

    # Step 1: Configuration, validation
    dims, cfg = validate_rmsnorm_mx_quantize_tkg(
        input_shape=input.shape,
        gamma_shape=gamma.shape,
        output_shape=output.shape,
        output_quant_dtype=output_quant.dtype,
        output_quant_shape=output_quant.shape,
        output_quant_buffer=output_quant.buffer,
        output_scale_shape=output_scale.shape,
        output_scale_buffer=output_scale.buffer,
        output_dtype=output.dtype,
        eps=eps,
    )

    # Step 2: Allocate buffers, load weights/constants
    gamma_sb = nl.ndarray((dims.H0, dims.H1), dtype=gamma.dtype, buffer=nl.sbuf)
    zero_bias = nl.ndarray((dims.H0, 1), dtype=dims.inter_dtype, buffer=nl.sbuf)
    reduction_const_matrix = nl.ndarray((dims.H0, dims.H0), dtype=dims.inter_dtype, buffer=nl.sbuf)
    eps_loaded = nl.ndarray((dims.H0, 1), dtype=dims.inter_dtype, buffer=nl.sbuf)

    # SBUF outputs are written in place and the HBM spill (step 4) is skipped: a caller that packs the
    # pair for a collective straight from SBUF saves the round trip.
    in_sbuf = output_quant.buffer == nl.sbuf
    if in_sbuf:
        kernel_assert(
            output_scale.dtype == _MX_SCALE_DTYPE,
            f"{_ERROR_PREFIX} an SBUF output_scale must be {_MX_SCALE_DTYPE}, got {output_scale.dtype}",
        )
        output_quant_sb = output_quant
        output_scale_sb = output_scale
    else:
        output_quant_sb = nl.ndarray((dims.H0, dims.num_H512_tiles, dims.BxS), dtype=output_quant.dtype, buffer=nl.sbuf)
        output_scale_sb = nl.ndarray((dims.H0, dims.num_H512_tiles, dims.BxS), dtype=_MX_SCALE_DTYPE, buffer=nl.sbuf)
    # nisa.quantize_mx writes only 4 of 32 scale partitions per quadrant; zero-init so the
    # unwritten padding reads as 0 (matching the golden reference) rather than an uninitialized value.
    nisa.memset(output_scale_sb, value=0)

    nisa.memset(zero_bias, value=0.0)
    nisa.memset(reduction_const_matrix, value=1.0)
    nisa.memset(eps_loaded, value=cfg.eps)

    # Load gamma with transpose: (1, H) -> (H) -> (H1, H0) -> (H0, H1)
    gamma_hbm = gamma.flatten_dims(start_dim=0, end_dim=1)
    gamma_hbm_view = gamma_hbm.reshape_dim(dim=0, shape=[dims.H1, dims.H0]).expand_dim(dim=1).expand_dim(dim=1)
    gamma_sb_view = gamma_sb.expand_dim(dim=1).expand_dim(dim=1)
    nisa.dma_transpose(dst=gamma_sb_view, src=gamma_hbm_view)

    # Reshape the HBM input for the transpose load
    input_hbm_view = input.reshape((dims.BxS * dims.H1, dims.H0))

    # Step 3: Process tiles - RMSNorm + MX quantization
    for bxs_tile_idx in range(cfg.num_BxS_tiles):
        # Step 3.1: Indexing, allocations
        tile_BxS_start_idx = bxs_tile_idx * cfg.BxS_tile_size
        hbm_tile_offset = tile_BxS_start_idx * dims.H1
        tile_BxS_slice = nl.ds(tile_BxS_start_idx, cfg.BxS_tile_size)

        hidden_sb = nl.ndarray((dims.H0, cfg.BxS_tile_size, dims.H1), dtype=input.dtype, buffer=nl.sbuf)
        output_tile_swizzled = nl.ndarray(
            (dims.H0, dims.num_H512_tiles, cfg.BxS_tile_size, _q_width), dtype=output.dtype, buffer=nl.sbuf
        )

        # Step 3.2: Transpose load the input tile: (BxS_tile_size * H1, H0) -> (H0, BxS_tile_size, H1)
        input_src_view = (
            input_hbm_view.slice(dim=0, start=hbm_tile_offset, end=hbm_tile_offset + cfg.BxS_tile_size * dims.H1)
            .expand_dim(dim=1)
            .expand_dim(dim=1)
        )
        input_dst_view = hidden_sb.flatten_dims(start_dim=1, end_dim=2).expand_dim(dim=1).expand_dim(dim=1)
        if input_parts is None:
            nisa.dma_transpose(dst=input_dst_view, src=input_src_view)
        else:
            part_sbs = []
            for part in input_parts:
                part_sb = nl.ndarray((dims.H0, cfg.BxS_tile_size, dims.H1), dtype=input.dtype, buffer=nl.sbuf)
                part_src = (
                    part.reshape((dims.BxS * dims.H1, dims.H0))
                    .slice(dim=0, start=hbm_tile_offset, end=hbm_tile_offset + cfg.BxS_tile_size * dims.H1)
                    .expand_dim(dim=1)
                    .expand_dim(dim=1)
                )
                _PART_LOADS.append(
                    nisa.dma_transpose(
                        dst=part_sb.flatten_dims(start_dim=1, end_dim=2).expand_dim(dim=1).expand_dim(dim=1),
                        src=part_src,
                    )
                )
                part_sbs.append(part_sb)
            nisa.tensor_tensor(dst=hidden_sb, data1=part_sbs[0], data2=part_sbs[1], op=nl.add)

        # Step 3.3: Compute RMSNorm
        _compute_rmsnorm(
            dims=dims,
            cfg=cfg,
            hidden_sb=hidden_sb,
            output=output,
            gamma_sb=gamma_sb,
            zero_bias=zero_bias,
            reduction_const_matrix=reduction_const_matrix,
            eps_loaded=eps_loaded,
            tile_BxS_slice=tile_BxS_slice,
        )

        # Step 3.4: Swizzle and MX quantize
        _swizzle_quantize_mx(
            dims=dims,
            cfg=cfg,
            output=output,
            output_tile_swizzled=output_tile_swizzled,
            output_quant_sb=output_quant_sb,
            output_scale_sb=output_scale_sb,
            tile_BxS_slice=tile_BxS_slice,
        )

    # Step 4: Spill the quantized pair to HBM (SBUF outputs were written in place above)
    if not in_sbuf:
        nisa.dma_copy(
            src=output_quant_sb[0 : dims.H0, 0 : dims.num_H512_tiles, 0 : dims.BxS],
            dst=output_quant[0 : dims.H0, 0 : dims.num_H512_tiles, 0 : dims.BxS],
        )
        nisa.dma_copy(
            src=output_scale_sb[0 : dims.H0, 0 : dims.num_H512_tiles, 0 : dims.BxS],
            dst=output_scale[0 : dims.H0, 0 : dims.num_H512_tiles, 0 : dims.BxS],
        )

    return [output, output_quant, output_scale]


def _compute_rmsnorm(
    dims,
    cfg,
    hidden_sb,
    output,
    gamma_sb,
    zero_bias,
    reduction_const_matrix,
    eps_loaded,
    tile_BxS_slice,
):
    """Compute RMSNorm on one token tile: output = (hidden * gamma) / RMS(hidden).

    hidden_sb is overwritten in place with (hidden * gamma) after the squares have been taken.
    """

    # Allocations
    square = nl.ndarray((dims.H0, cfg.BxS_tile_size, dims.H1), dtype=dims.inter_dtype, buffer=nl.sbuf)
    reduced = nl.ndarray((dims.H0, cfg.BxS_tile_size), dtype=dims.inter_dtype, buffer=nl.sbuf)
    final_reduced = nl.ndarray((dims.H0, cfg.BxS_tile_size), dtype=nl.float32, buffer=nl.psum)
    sqrt = nl.ndarray((dims.H0, cfg.BxS_tile_size), dtype=dims.inter_dtype, buffer=nl.sbuf)

    # Input ^2
    nisa.activation(dst=square, op=nl.square, data=hidden_sb, bias=zero_bias)

    # Input * gamma (broadcast gamma across BxS_tile_size)
    gamma_sb_view = gamma_sb.expand_dim(dim=1).broadcast(dim=1, size=cfg.BxS_tile_size)
    nisa.tensor_tensor(hidden_sb, hidden_sb, gamma_sb_view, nl.multiply)

    # Reduce squared input along H1 dimension (last free dimension)
    nisa.tensor_reduce(dst=reduced, op=nl.add, data=square, axis=2)

    # Complete reduction across H0 dimension using matmul
    nisa.nc_matmul(dst=final_reduced, stationary=reduction_const_matrix, moving=reduced)

    # Compute 1/sqrt(mean(x^2) + eps)
    nisa.activation(dst=sqrt, op=nl.rsqrt, data=final_reduced, scale=(1.0 / dims.H), bias=eps_loaded)

    # Compute input * 1/RMS(input)
    sqrt_view = sqrt.expand_dim(dim=2).broadcast(dim=2, size=dims.H1)
    nisa.tensor_tensor(output[:, tile_BxS_slice, :], hidden_sb, sqrt_view, nl.multiply)


def _swizzle_quantize_mx(
    dims,
    cfg,
    output,
    output_tile_swizzled,
    output_quant_sb,
    output_scale_sb,
    tile_BxS_slice,
):
    """Swizzle the normalized output of the token tile and quantize it to MX format.

    The swizzle is ONE copy (a 3-free-dim source pattern), and ONE ``quantize_mx`` covers every H/512 tile, whose
    outputs are contiguous with a single token tile. Each 32-value group is quantized on its own, so the bytes are
    those of ``H/512 * 4`` copies and a ``quantize_mx`` per H/512 tile.
    """
    # swizzled[p, h, t, q] = output[p, t, q * num_H512_tiles + h]
    n_h, T, H1 = dims.num_H512_tiles, cfg.BxS_tile_size, dims.H1
    nisa.tensor_copy(
        dst=output_tile_swizzled[0 : dims.H0, 0:n_h, 0:T, 0:_q_width],
        src=output.ap(pattern=[[T * H1, dims.H0], [1, n_h], [H1, T], [n_h, _q_width]], offset=0),
        engine=nisa.vector_engine,
    )
    nisa.quantize_mx(
        src=output_tile_swizzled[0 : dims.H0, 0:n_h, 0:T, 0:_q_width],
        dst=output_quant_sb[0 : dims.H0, 0:n_h, tile_BxS_slice],
        dst_scale=output_scale_sb[0 : dims.H0, 0:n_h, tile_BxS_slice],
    )


_PART_LOADS = []


def take_part_loads() -> list:
    """The ``input_parts`` loads of the latest :func:`rmsnorm_mx_quantize_for_router` calls (they wait for the
    attention exit's ReduceScatter; the MoE layer holds routed expert 0's gate weight load behind them), then forget
    them."""
    out = list(_PART_LOADS)
    _PART_LOADS.clear()
    return out
