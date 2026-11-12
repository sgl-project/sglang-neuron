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

"""Input load and PSUM drain helpers for the bf16 dense MLP TKG kernel."""

import nki.isa as nisa
import nki.language as nl

from .mlp_tkg_constants import MLPTKGConstantsDimensionSizes

_DGE_MODE_NONE = 3
"""Use STATIC DMA mode, which is faster than letting the compiler choose for these loads."""


def input_load(
    input: nl.NkiTensor,
    dims: MLPTKGConstantsDimensionSizes,
    output_dtype,
) -> nl.NkiTensor:
    """
    Load the hidden activations from HBM into an ``[H0, T, H1]`` SBUF tile.

    There is no normalization: in the model the RMSNorm lives outside the MLP module, so the
    caller applies it and hands this kernel the already-normed activations.

    Args:
        input (nl.NkiTensor): [B, S, H] hidden state in HBM.
        dims (MLPTKGConstantsDimensionSizes): Dimension data.
        output_dtype: dtype of the SBUF tile. The load casts to it, which is what makes a
            float32 ``output_dtype`` against a bfloat16 input widen at the load rather than later.

    Returns:
        nl.NkiTensor: SBUF [H0, T, H1].
    """
    H0 = dims.H0
    T = dims.T
    H1 = dims.H1

    output = nl.ndarray((H0, T, H1), dtype=output_dtype, buffer=nl.sbuf)

    input_view = input
    if len(input_view.shape) == 3:
        input_view = input_view.flatten_dims(start_dim=0, end_dim=1)

    # [T, H] -> [H0, T, H1]
    input_view = input_view.reshape_dim(dim=1, shape=[H0, H1]).permute(dims=[1, 0, 2])

    nisa.dma_copy(
        src=input_view,
        dst=output,
        dge_mode=_DGE_MODE_NONE,
    )

    return output


def interleave_copy(dst: nl.NkiTensor, src: nl.NkiTensor, index: int, scale: nl.NkiTensor) -> None:
    """
    Drain one PSUM tile into SBUF times a per-partition factor, alternating engines so consecutive drains overlap.

    Even ``index`` emits on the Scalar (activation) engine, the factor its ``scale``; odd ``index`` on the Vector
    (DVE) engine, a ``tensor_scalar`` multiply. Callers pass a monotonically increasing ``index``.

    Args:
        dst (nl.NkiTensor): Destination tile in SBUF.
        src (nl.NkiTensor): Source tile in PSUM.
        index (int): Engine-selection counter.
        scale (nl.NkiTensor): [T, 1] fp32 per-partition factor.
    """
    if index % 2 == 0:
        nisa.activation(dst=dst, data=src, op=nl.copy, scale=scale)
    else:
        nisa.tensor_scalar(dst=dst, data=src, op0=nl.multiply, operand0=scale, engine=nisa.vector_engine)


def column_tiled_matmul(
    dst: nl.NkiTensor,
    stationary: nl.NkiTensor,
    moving: nl.NkiTensor,
    column_idx: int,
    dims: MLPTKGConstantsDimensionSizes,
) -> None:
    """
    One token-stationary ``nc_matmul`` on PE column tile ``column_idx``.

    With a column-tiling factor above 1 the matmul is pinned to the ``column_tiling_dim``-wide
    PE column tile starting at column ``column_tiling_dim * column_idx``, so up to ``factor``
    such matmuls with different operands run concurrently; ``dst`` must then be the matching
    PSUM row window ``[column_tiling_dim * column_idx, +T)``. With a factor of 1 the matmul uses
    the whole array and ``column_idx`` must be 0.

    Args:
        dst (nl.NkiTensor): [T, N] PSUM destination window.
        stationary (nl.NkiTensor): [K, T] SBUF tile, K <= 128.
        moving (nl.NkiTensor): [K, N] SBUF tile, N <= 512.
        column_idx (int): PE column tile index, ``< column_tiling_factor``.
        dims (MLPTKGConstantsDimensionSizes): Dimension constants.
    """
    if dims.column_tiling_factor == 1:
        nisa.nc_matmul(dst=dst, stationary=stationary, moving=moving)
    else:
        nisa.nc_matmul(
            dst=dst,
            stationary=stationary,
            moving=moving,
            tile_position=(0, dims.column_tiling_dim * column_idx),
            tile_size=(dims._pmax, dims.column_tiling_dim),
        )
