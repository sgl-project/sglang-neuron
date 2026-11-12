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

"""Gate and Up projection sub-kernel for the bf16 dense MLP TKG kernel (column tiling)."""

import nki.isa as nisa
import nki.language as nl

from ...utils.kernel_assert import kernel_assert
from ...utils.kernel_helpers import get_nl_act_fn_from_type
from ...utils.tiled_range import TiledRange
from .mlp_parameters_bf16 import MLPParameters
from .mlp_tkg_constants import MLPTKGConstantsDimensionSizes, MLPTKGConstantsWeightRing
from .mlp_tkg_utils import column_tiled_matmul

_DGE_MODE_NONE = 3  # Use STATIC DMA mode


def gate_up_projection(
    hidden: nl.NkiTensor,
    weight: nl.NkiTensor,
    output_tile: nl.NkiTensor,
    weight_ring: list[nl.NkiTensor],
    ring: MLPTKGConstantsWeightRing,
    ring_base: int,
    dims: MLPTKGConstantsDimensionSizes,
    op_name: str,
    weight_dmas: list,
    preloaded: nl.NkiTensor = None,
):
    """
    Performs a single Gate or Up projection.

    Computes: Hidden[H, T] @ Weight[H, I] -> [T, I]
    - Hidden is the stationary tensor, Weight is the moving tensor.

    Tiled computation:
    H/128 * [ I/512 * (Hidden[128, T] @ Weight[128, 512]) ]

    Tile Load:
    One weight load is ``[128, h1, I]``: ``h1`` consecutive 128-row tiles of H, each partition
    reading ``h1`` rows of the weight. ``h1`` is the most that fits a ring slot (16 at GLM-5.2's
    I = 2048 -> 8 MB per load, 3 loads for H = 6144):
    H/(128*h1) * [ h1 * [ I/512 * (Hidden[128, T] @ Weight[128, 512]) ] ]

    Column Tiling:
    A decode step's hidden tile is [128, T] with T <= 32, so it occupies only 32 of the PE array's
    128 columns. Column tiling puts the H1 tiles of one weight load side by side instead:

    ---------------------------------------------------
    | col_tile_0 | col_tile_1 | col_tile_2 | col_tile_3 |
    | 32 columns | 32 columns | 32 columns | 32 columns |
    ---------------------------------------------------
    - `column_tiling_dim` = 32, `column_tiling_factor` = 4 -> a speedup.
    - H1 tile ``k`` of a load runs on column tile ``k % factor`` and accumulates in PSUM row window
      [32*(k % factor), +T) with every other H1 tile of the same residue, across all loads. The
      ``factor`` windows are summed into SBUF afterwards. Untiled, everything accumulates in window
      [0, T) and there is nothing to sum.

    Key Points:
    -----------
    - Intermediate projection tensors are always fp32 for better numerical accuracy
    - Matrix multiplication is tiled along H and I

    Args:
        hidden (nl.NkiTensor): [H0, T, H1] hidden activations in SBUF.
        weight (nl.NkiTensor): [H, I] weight matrix in HBM.
        output_tile (nl.NkiTensor): [T, I] fp32 output buffer in SBUF.
        weight_ring (list[nl.NkiTensor]): The shared ring of [128, slot_elems] SBUF slots.
        ring (MLPTKGConstantsWeightRing): Ring geometry and this projection's load shape.
        ring_base (int): Slot the first load of this projection takes (continues the rotation).
        dims (MLPTKGConstantsDimensionSizes): Dimension constants.
        op_name (str): "gate" or "up", for messages.
        weight_dmas (list): Receives the weight DMA instructions in issue order.
        preloaded (nl.NkiTensor): Optional [H0, H1, I] SBUF view of the whole weight, already loaded (the
            projection is then one load and issues none).
    """

    # ---------- Configuration and Dimension Setup ----------
    H0 = dims.H0
    T = dims.T
    H = dims.H
    I = weight.shape[1]
    H1 = H // H0
    h_tile = ring.gate_up_h1_per_load * H0

    # Sanity check: the projection issues one PSUM bank per 512 columns of I and does not tile I.
    kernel_assert(
        I <= dims.max_I_shard_size,
        f"{op_name}_projection supports I <= {dims.max_I_shard_size}, got I={I}",
    )

    # ---------- Allocate PSUM buffers ----------
    result_psums = []
    for _ in TiledRange(I, dims._psum_fmax):
        result_psum = nl.ndarray(
            shape=(dims._pmax, dims._psum_fmax),
            dtype=nl.float32,
            buffer=nl.psum,
        )
        result_psums.append(result_psum)

    # ---------- Matrix multiplication ----------
    # H1 tile k of a load runs on PE column tile k % factor; the windows accumulate across loads.
    used_columns = 0

    for hidden_tiles in TiledRange(H, h_tile):
        h_offset = hidden_tiles.index * ring.gate_up_h1_per_load
        h1_tiles = hidden_tiles.size // H0

        if preloaded is not None:
            kernel_assert(h1_tiles == H1, f"{op_name}_projection: a preloaded weight must be one load")
            weight_sb = preloaded
        else:
            # Load weight [h1 * 128, I] -> SBUF layout [H0, h1, I]: partition h0 reads rows h0 * H1 + h1.
            slot = (ring_base + hidden_tiles.index) % ring.depth
            weight_view = weight.reshape_dim(dim=0, shape=(H0, H1)).slice(
                dim=1, start=h_offset, end=h_offset + h1_tiles
            )
            weight_sb = weight_ring[slot][0:H0, 0 : h1_tiles * I].reshape_dim(dim=1, shape=(h1_tiles, I))

            weight_dmas.append(
                nisa.dma_copy(
                    dst=weight_sb,
                    src=weight_view,
                    dge_mode=_DGE_MODE_NONE,
                )
            )

        # Matmul
        for i_tile in TiledRange(I, dims._psum_fmax):
            for h1_idx in range(h1_tiles):
                column_idx = h1_idx % dims.column_tiling_factor
                column_tiled_matmul(
                    dst=result_psums[i_tile.index][
                        nl.ds(dims.column_tiling_dim * column_idx, T),
                        0 : i_tile.size,
                    ],
                    # hidden is [H0, T, H1], so the H1 tile index selects on dim 2.
                    stationary=hidden.select(dim=2, index=h_offset + h1_idx),
                    moving=weight_sb.select(dim=1, index=h1_idx).slice(
                        dim=1, start=i_tile.start_offset, end=i_tile.end_offset
                    ),
                    column_idx=column_idx,
                    dims=dims,
                )
            # Update used column numbers
            used_columns = max(used_columns, min(h1_tiles, dims.column_tiling_factor))

    _accumulate_psum_windows(output_tile, result_psums, used_columns, I, T, dims)


def _accumulate_psum_windows(
    output_tile: nl.NkiTensor,
    result_psums: list,
    used_columns: int,
    I: int,
    T: int,
    dims: MLPTKGConstantsDimensionSizes,
) -> None:
    """A projection's PSUM row windows summed into its [T, I] fp32 SBUF output: window 0 copied, the others added."""
    # ---------- Accumulate PSUMs into output ----------
    for i_tile in TiledRange(I, dims._psum_fmax):
        # Create sliced output view for this I tile
        sliced_output = output_tile.slice(dim=1, start=i_tile.start_offset, end=i_tile.end_offset)
        # Copy PSUM to SBUF
        nisa.activation(
            dst=sliced_output,
            data=result_psums[i_tile.index][0:T, 0 : i_tile.size],
            op=nl.copy,
        )

        # Accumulate PSUMs to SBUF
        for factor_idx in range(1, used_columns):
            nisa.tensor_tensor(
                dst=sliced_output,
                data1=result_psums[i_tile.index][nl.ds(dims.column_tiling_dim * factor_idx, T), 0 : i_tile.size],
                data2=sliced_output,
                op=nl.add,
            )


def _fused_split_loads(dims: MLPTKGConstantsDimensionSizes, ring: MLPTKGConstantsWeightRing) -> int:
    """Loads of ``[H0, H1 / n, 2I]`` that carry a fused [H, 2I] gate/up when the whole does not fit one slot: the
    fewest ``n`` whose load fits a slot and keeps a multiple of the column-tiling factor of H tiles; 0 if none."""
    for n in range(2, dims.H1 + 1):
        h1 = dims.H1 // n
        if dims.H1 % n == 0 and h1 % dims.column_tiling_factor == 0 and h1 * 2 * dims.I <= ring.slot_elems:
            return n
    return 0


def _gate_up_fused_split(
    hidden: nl.NkiTensor,
    fused: nl.NkiTensor,
    gate_sb: nl.NkiTensor,
    up_sb: nl.NkiTensor,
    weight_ring: list[nl.NkiTensor],
    ring: MLPTKGConstantsWeightRing,
    ring_start: int,
    dims: MLPTKGConstantsDimensionSizes,
    n_loads: int,
    weight_dmas: list,
) -> None:
    """
    Gate and up from ONE [H, 2I] tensor in ``n_loads`` ring loads of ``[H0, H1 / n_loads, 2I]``: each partition
    reads ``H1 / n_loads`` whole rows, one contiguous run, where separate gate and up loads read ``I``-element runs at
    a ``2I`` stride. Per load both projections' matmuls run on their halves; every H tile ``k`` still goes to PSUM
    row window ``k % factor`` (a load holds a multiple of ``factor`` tiles) and each window accumulates over ``k`` in
    increasing order, so the sums are ``gate_up_projection``'s.
    """
    H0, H1, I, T = dims.H0, dims.H1, dims.I, dims.T
    h1_per = H1 // n_loads
    psums = []
    for _ in range(2):
        psums.append(
            [
                nl.ndarray(shape=(dims._pmax, dims._psum_fmax), dtype=nl.float32, buffer=nl.psum)
                for _ in TiledRange(I, dims._psum_fmax)
            ]
        )
    src = fused.reshape_dim(dim=0, shape=(H0, H1))
    for load in range(n_loads):
        slot = (ring_start + load) % ring.depth
        weight_sb = weight_ring[slot][0:H0, 0 : h1_per * 2 * I].reshape_dim(dim=1, shape=(h1_per, 2 * I))
        weight_dmas.append(
            nisa.dma_copy(
                dst=weight_sb,
                src=src.slice(dim=1, start=load * h1_per, end=(load + 1) * h1_per),
                dge_mode=_DGE_MODE_NONE,
            )
        )
        for proj in range(2):
            for i_tile in TiledRange(I, dims._psum_fmax):
                for h1_idx in range(h1_per):
                    column_idx = h1_idx % dims.column_tiling_factor
                    column_tiled_matmul(
                        dst=psums[proj][i_tile.index][nl.ds(dims.column_tiling_dim * column_idx, T), 0 : i_tile.size],
                        stationary=hidden.select(dim=2, index=load * h1_per + h1_idx),
                        moving=weight_sb.select(dim=1, index=h1_idx).slice(
                            dim=1, start=proj * I + i_tile.start_offset, end=proj * I + i_tile.end_offset
                        ),
                        column_idx=column_idx,
                        dims=dims,
                    )
    used_columns = min(h1_per, dims.column_tiling_factor)
    _accumulate_psum_windows(gate_sb, psums[0], used_columns, I, T, dims)
    _accumulate_psum_windows(up_sb, psums[1], used_columns, I, T, dims)


def process_gate_up_projection(
    hidden: nl.NkiTensor,
    output: nl.NkiTensor,
    params: MLPParameters,
    dims: MLPTKGConstantsDimensionSizes,
    weight_ring: list[nl.NkiTensor],
    ring: MLPTKGConstantsWeightRing,
    factors_sb: nl.NkiTensor,
    ring_start: int,
    weight_dmas: list,
) -> int:
    """
    Performs the Gate/Up projection for the bf16 MLP (T = BxS).

    Runs both projections into fp32 SBUF tiles, applies the activation to gate, multiplies the
    two, then transposes the [T, I] product into the [I0, I/I0, T] layout the down projection
    consumes as its stationary operand. The up projection's loads continue the weight ring where
    the gate projection's stopped, so the first up load overlaps the last gate matmuls.

    Args:
        hidden (nl.NkiTensor): [H0, T, H1] hidden activations in SBUF.
        output (nl.NkiTensor): [I0, I/I0, T] output buffer in SBUF, written by the transpose.
        params (MLPParameters): MLP configuration parameters.
        dims (MLPTKGConstantsDimensionSizes): Dimension constants.
        weight_ring (list[nl.NkiTensor]): The shared ring of [128, slot_elems] SBUF slots.
        ring (MLPTKGConstantsWeightRing): Ring geometry and load shapes.
        factors_sb (nl.NkiTensor): [T, 3] fp32 FP8 tensor factors (gate, up, down) on every token partition: the
            gate's goes into the activation's ``scale``, the up's into the product.
        ring_start (int): Ring position of the first load (the caller's ring continues its rotation).
        weight_dmas (list): Receives the weight DMA instructions in issue order.

    Returns:
        int: Weight loads issued (the down projection's loads continue the rotation after them).
    """
    gate_w, up_w = params.gate_proj_weights_tensor, params.up_proj_weights_tensor

    # gate and up in ONE load when they are the halves of one [H, 2I] tensor and each is a single load anyway ...
    fused_sb = None
    n_loads = 2 * ring.gate_up_loads
    if params.gate_up_fused is not None and ring.gate_up_loads == 1 and dims.H1 * 2 * dims.I <= ring.slot_elems:
        n_loads = 1
        H0, H1, I = dims.H0, dims.H1, dims.I
        slot = ring_start % ring.depth
        fused_sb = weight_ring[slot][0:H0, 0 : H1 * 2 * I].reshape_dim(dim=1, shape=(H1, 2 * I))
        weight_dmas.append(
            nisa.dma_copy(
                dst=fused_sb,
                src=params.gate_up_fused.reshape_dim(dim=0, shape=(H0, H1)),
                dge_mode=_DGE_MODE_NONE,
            )
        )

    # ... or in a few loads of whole rows, when it does not fit one slot.
    split_loads = 0
    if params.gate_up_fused is not None and fused_sb is None and ring.gate_up_loads == 1:
        split_loads = _fused_split_loads(dims, ring)

    # Intermediate projection results are fp32 for accuracy: PSUM accumulates in fp32 and the
    # activation and the element-wise product both run before the down projection rounds back.
    tile_shape = (dims.T, dims.I)
    gate_sb = nl.ndarray(tile_shape, dtype=nl.float32, buffer=nl.sbuf)
    up_sb = nl.ndarray(tile_shape, dtype=nl.float32, buffer=nl.sbuf)

    # ---------------- Gate/Up Projection (column tiling) ----------------
    if split_loads > 0:
        n_loads = split_loads
        _gate_up_fused_split(
            hidden, params.gate_up_fused, gate_sb, up_sb, weight_ring, ring, ring_start, dims, split_loads, weight_dmas
        )
    else:
        gate_up_projection(
            hidden=hidden,
            weight=gate_w,
            output_tile=gate_sb,
            weight_ring=weight_ring,
            ring=ring,
            ring_base=ring_start,
            dims=dims,
            op_name="gate",
            weight_dmas=weight_dmas,
            preloaded=None if fused_sb is None else fused_sb.slice(dim=2, start=0, end=dims.I),
        )
        gate_up_projection(
            hidden=hidden,
            weight=up_w,
            output_tile=up_sb,
            weight_ring=weight_ring,
            ring=ring,
            ring_base=(ring_start + ring.gate_up_loads) % ring.depth,
            dims=dims,
            op_name="up",
            weight_dmas=weight_dmas,
            preloaded=None if fused_sb is None else fused_sb.slice(dim=2, start=dims.I, end=2 * dims.I),
        )

    # ---------------- Activation and element-wise product ----------------
    # FP8 tensor factors: act(s_g * gate) and (s_u * up) * act, two instructions.
    nisa.activation(
        dst=gate_sb,
        op=get_nl_act_fn_from_type(params.activation_fn),
        data=gate_sb,
        scale=factors_sb[0 : dims.T, 0:1],
    )
    nisa.scalar_tensor_tensor(
        dst=up_sb,
        data=up_sb,
        op0=nl.multiply,
        operand0=factors_sb[0 : dims.T, 1:2],
        op1=nl.multiply,
        operand1=gate_sb,
    )

    # ---------------- Transpose [T, I] -> [I0, I/I0, T] for the down projection ----------------
    for i_tile in TiledRange(dims.I, dims.I0):
        tp_psum = nl.ndarray(
            (i_tile.size, dims.T),
            dtype=up_sb.dtype,
            buffer=nl.psum,
        )
        nisa.nc_transpose(
            dst=tp_psum,
            data=up_sb.slice(dim=0, start=0, end=dims.T).slice(
                dim=1, start=i_tile.index * dims.I0, end=i_tile.index * dims.I0 + i_tile.size
            ),
        )
        nisa.tensor_copy(
            dst=output.slice(dim=0, start=0, end=i_tile.size)
            .slice(dim=1, start=i_tile.index, end=i_tile.index + 1)
            .slice(dim=2, start=0, end=dims.T),
            src=tp_psum,
        )
    return n_loads
