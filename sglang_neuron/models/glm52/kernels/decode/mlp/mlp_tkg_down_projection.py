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

"""Down projection sub-kernel for the bf16 dense MLP TKG kernel (column tiling)."""

import nki.isa as nisa
import nki.language as nl

from ..utils.kernel_assert import kernel_assert
from ..utils.kernel_helpers import div_ceil
from ..utils.tiled_range import TiledRange
from .mlp_parameters_bf16 import MLPParameters
from .mlp_tkg_constants import MLPTKGConstantsDimensionSizes, MLPTKGConstantsWeightRing
from .mlp_tkg_utils import column_tiled_matmul, interleave_copy

_DGE_MODE_NONE = 3  # Use STATIC DMA mode


def down_projection(
    hidden: nl.NkiTensor,
    weight: nl.NkiTensor,
    output_tile: nl.NkiTensor,
    weight_ring: list[nl.NkiTensor],
    ring: MLPTKGConstantsWeightRing,
    ring_base: int,
    dims: MLPTKGConstantsDimensionSizes,
):
    """
    Performs the Down projection.

    Computes: Hidden[I, T] @ Weight[I, H] -> [T, H]
    - Hidden is the stationary tensor, Weight is the moving tensor.

    Tiled computation:
    H/512 * [ I/128 * (Hidden[128, T] @ Weight[128, 512]) ]

    Tile Load:
    One weight load is ``[128, i1, H_tile]``: ``i1`` 128-row tiles of I, partition ``p`` reading row
    ``j * 128 + p`` of each -- the same ``i = j * 128 + p`` placement the gate/up transpose gave the
    stationary ``hidden``. ``i1`` is the most that fits a ring slot (4 at GLM-5.2's H = 6144 -> 6 MB
    per load, 4 loads for I = 2048). The whole H_tile accumulates in PSUM across all I loads, so the
    loop is I-outer, H-compute-tile inner:
    H/H_tile * [ I/(128*i1) * [ i1 * [ H_tile/512 * (Hidden[128, T] @ Weight[128, 512]) ] ] ]

    Column Tiling:
    A decode step's hidden tile is [128, T] with T <= 32, so it occupies only 32 of the PE array's
    128 columns. Column tiling puts four consecutive 512-column slices of the H tile side by side:

    ---------------------------------------------------
    | col_tile_0 | col_tile_1 | col_tile_2 | col_tile_3 |
    | 32 columns | 32 columns | 32 columns | 32 columns |
    ---------------------------------------------------
    - `column_tiling_dim` = 32, `column_tiling_factor` = 4 -> a speedup.
    - Slice ``k`` of an H tile lands in PSUM bank ``k // factor`` at row window
      ``[dim*(k % factor), +T)``: H = 6144 is 12 slices over 3 banks at the 4-way tile, 6 banks at
      the 2-way tile, and two H tiles of 8 slices / 8 banks untiled (``ring.down_h_tile``).

    Args:
        hidden (nl.NkiTensor): [I0, I1, T] hidden activations in SBUF, ``i = i1 * 128 + i0``.
        weight (nl.NkiTensor): [I, H] weight matrix in HBM.
        output_tile (nl.NkiTensor): [T, H] output buffer in SBUF.
        weight_ring (list[nl.NkiTensor]): The shared ring of [128, slot_elems] SBUF slots.
        ring (MLPTKGConstantsWeightRing): Ring geometry and this projection's load shape.
        ring_base (int): Slot the first load takes (continues the gate/up rotation).
        dims (MLPTKGConstantsDimensionSizes): Dimension constants.
    """

    # ---------- Configuration and Dimension Setup ----------
    I0, I1, T = hidden.shape
    I = weight.shape[0]
    H = dims.H
    kernel_assert(
        I % I0 == 0,
        f"down_projection loads [{I0}, i1, H] weight tiles, so I must be a multiple of {I0}, got I={I}",
    )

    # [I0, I1, H]: element (p, j, h) is weight[j * I0 + p, h], matching the stationary layout.
    weight_i1 = weight.reshape_dim(dim=0, shape=(I1, I0)).permute(dims=[1, 0, 2])

    # ---------- Compute matmul  ----------
    for h_tiles in TiledRange(H, ring.down_h_tile):
        h_offset = h_tiles.start_offset
        num_compute_tiles = div_ceil(h_tiles.size, dims._psum_fmax)
        num_psums = div_ceil(num_compute_tiles, dims.column_tiling_factor)

        # Allocate PSUM: the whole H tile accumulates here across every I load.
        result_psums = []
        for _ in range(num_psums):
            result_psum = nl.ndarray(
                shape=(dims._pmax, dims._psum_fmax),
                dtype=nl.float32,
                buffer=nl.psum,
            )
            result_psums.append(result_psum)

        for i_load in TiledRange(I1, ring.down_i1_per_load):
            i1_tiles = i_load.size
            slot = (ring_base + h_tiles.index * ring.down_loads_per_h_tile + i_load.index) % ring.depth

            # Load weight [i1 * 128, H_tile] from HBM into the SBUF slot as [I0, i1, H_tile]
            weight_sb = weight_ring[slot][0:I0, 0 : i1_tiles * h_tiles.size].reshape_dim(
                dim=1, shape=(i1_tiles, h_tiles.size)
            )
            weight_view = weight_i1.slice(dim=1, start=i_load.start_offset, end=i_load.end_offset).slice(
                dim=2, start=h_offset, end=h_offset + h_tiles.size
            )
            nisa.dma_copy(
                dst=weight_sb,
                src=weight_view,
                dge_mode=_DGE_MODE_NONE,
            )

            # Matmul with column tiling: distribute H slices across columns and PSUM banks
            for j in range(i1_tiles):
                i1_idx = i_load.start_offset + j
                hidden_sb_tile_slice = hidden.slice(dim=0, start=0, end=I0).slice(dim=1, start=i1_idx, end=i1_idx + 1)
                for compute_tile in TiledRange(h_tiles.size, dims._psum_fmax):
                    column_tile_index = compute_tile.index % dims.column_tiling_factor
                    psum_bank_index = compute_tile.index // dims.column_tiling_factor
                    column_tiled_matmul(
                        dst=result_psums[psum_bank_index][
                            nl.ds(dims.column_tiling_dim * column_tile_index, T),
                            0 : compute_tile.size,
                        ],
                        stationary=hidden_sb_tile_slice,
                        moving=weight_sb.select(dim=1, index=j).slice(
                            dim=1, start=compute_tile.start_offset, end=compute_tile.end_offset
                        ),
                        column_idx=column_tile_index,
                        dims=dims,
                    )

        # ---------- Copy PSUM output to SBUF ----------
        for compute_tile in TiledRange(h_tiles.size, dims._psum_fmax):
            column_tile_index = compute_tile.index % dims.column_tiling_factor
            psum_bank_index = compute_tile.index // dims.column_tiling_factor
            dst_offset = h_offset + compute_tile.start_offset
            interleave_copy(
                index=compute_tile.index,
                dst=output_tile.slice(dim=1, start=dst_offset, end=dst_offset + compute_tile.size),
                src=result_psums[psum_bank_index][
                    nl.ds(dims.column_tiling_dim * column_tile_index, T),
                    0 : compute_tile.size,
                ],
            )


def process_down_projection(
    hidden: nl.NkiTensor,
    output: nl.NkiTensor,
    params: MLPParameters,
    dims: MLPTKGConstantsDimensionSizes,
    weight_ring: list[nl.NkiTensor],
    ring: MLPTKGConstantsWeightRing,
) -> None:
    """
    Performs the Down projection for the bf16 MLP (T = BxS).

    Expected hidden tensor shape is [128(I0), I/128, T].

    Overview:
    ---------
    hidden @ down_weight
    # [T, H] = [T, I] @ [I, H]

    Hardware constraints (max partition size of 128) require tiling along the I dimension:
    # hidden [128, I//128, T] @ down_weight [128, I//128, H]

    ``down_weight`` is row-sharded across tensor-parallel ranks, so the [T, H] result is this
    rank's PARTIAL SUM; the cross-rank reduction happens outside the kernel.

    DMA mode:
    ---------
    Based on experiments, Static DMA provides better performance.
    The MLP TKG implementation therefore uses Static DMA for tensor loads.
    If HBM out-of-memory (OOM) issues arise, we can fall back to DGE mode.

    Args:
        hidden (nl.NkiTensor): [I0, I/I0, T] activations in SBUF.
        output (nl.NkiTensor): [T, H] output buffer in SBUF.
        params (MLPParameters): MLP configuration parameters.
        dims (MLPTKGConstantsDimensionSizes): Dimension constants.
        weight_ring (list[nl.NkiTensor]): The shared ring of [128, slot_elems] SBUF slots.
        ring (MLPTKGConstantsWeightRing): Ring geometry and load shapes.
    """
    down_projection(
        hidden=hidden,
        weight=params.down_proj_weights_tensor,
        output_tile=output,
        weight_ring=weight_ring,
        ring=ring,
        # Continue the rotation after the gate and up loads.
        ring_base=(2 * ring.gate_up_loads) % ring.depth,
        dims=dims,
    )
