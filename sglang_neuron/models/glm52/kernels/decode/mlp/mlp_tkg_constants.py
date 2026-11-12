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

"""Tiling constants for the bf16 dense MLP TKG kernel.

SBUF tiles are allocated with plain ``nl.ndarray``, so the compiler's allocator places them and
there is no free-space budget to divide. The weight ring below is therefore a fixed double-buffer
of fixed-size slots rather than "as many tiles as fit".

Weight loads
------------
The kernel is weight-DMA bound (a decode step's matmuls take a fraction of the time the weights
take to arrive), so each weight load should be as large as SBUF allows: a DMA trigger carries a
fixed setup cost, and a 1-2 MB load spends a large part of its time in it. All three projections
stream their weights through ONE ring of ``_W_RING_DEPTH`` slots of ``_W_RING_SLOT_BYTES`` per
partition, rotating continuously from gate through up to down so the next projection's first load
overlaps the previous projection's last matmuls. A gate/up load is ``[128, h1, I]`` (``h1`` rows of
H per partition), a down load ``[128, i1, H]`` (``i1`` rows of I per partition); ``h1`` / ``i1`` are
the largest counts whose tile fits the slot and divides the projection evenly. At GLM-5.2's expert
shape (H = 6144, I = 2048, bf16) that is 3 loads of 8 MB per gate/up projection and 4 loads of 6 MB
for down, 10 triggers per expert.
"""

from dataclasses import dataclass

import nki.language as nl

from ..utils.kernel_helpers import div_ceil, sizeinbytes
from .mlp_parameters_bf16 import MLPParameters

_W_RING_DEPTH = 2
"""Weight ring depth: double-buffered, so a load overlaps the matmuls on the previous slot."""

_W_RING_SLOT_BYTES = 64 * 1024
"""Bytes per partition of one weight-ring slot (two slots: 128 KB of the 240 KB usable on Trn3).

Sized so the MoE layer that calls this kernel per expert still fits its own tiles beside the ring;
a 96 KB slot (2 loads per projection) does not.
"""

_DENSE_RING_DEPTH = 3
"""The dense weight ring's depth: 48 KB slots, so at the GLM-5.2 dense shape (I = 1536) every load keeps its 6.29 MB
size and a third load is in flight while the PE drains a slot."""

_W_RING_DEEP_SLOT_BYTES = 48 * 1024
"""Slot bytes per partition of the depth-3 ring (144 KB in total, 16 KB more than ``_W_RING_DEPTH``'s)."""

_COLUMN_TILING_DIMS = ((32, 32), (64, 64))
"""``(max T, PE column-tile width)`` steps for column tiling; T above the last step runs untiled (128).

Column tiling packs ``128 / dim`` independent token tiles side by side in the PE array, so a
decode step that occupies only T of 128 columns still fills it. Each column tile writes PSUM rows
``[dim*i, dim*i + T)``, so the width is the smallest power of two >= T and the tiles stay inside
the 128 partitions. GLM-5.2's C32 x spec-7 decode step is 14 real tokens (16 padded slots) per
rank, and the MoE layer's compacted routed path hands this kernel <= 32 live tokens, so both take
the 32-wide, 4-way tile. Gen2 cannot express a 32-wide tile (neither the row nor the column size
of a PE tile may be 32 there); this kernel is gen4-only, so that case does not arise.
"""


def column_tiling_dim(T: int, column_tiling: bool) -> int:
    """
    PE column-tile width for ``T`` tokens, or the full 128 columns when tiling is off.

    Args:
        T (int): Tokens per decode step (the stationary operand's free size).
        column_tiling (bool): False runs every matmul on the full PE array.

    Returns:
        int: 32, 64 or 128.
    """
    if column_tiling:
        for max_tokens, dim in _COLUMN_TILING_DIMS:
            if T <= max_tokens:
                return dim
    return nl.tile_size.pmax


def tiles_per_load(n_tiles: int, cap: int) -> int:
    """
    Number of 128-row tiles per weight load: the largest count <= ``cap`` that divides ``n_tiles``.

    Even loads keep every trigger the same size; a divisor always exists (1), and the cap itself
    is returned when it covers the whole projection.

    Args:
        n_tiles (int): 128-row tiles in the projection's contraction dimension (H1 or I1).
        cap (int): Most tiles whose rows fit one ring slot.

    Returns:
        int: Tiles per load, in ``[1, min(cap, n_tiles)]``.
    """
    cap = max(1, min(cap, n_tiles))
    for d in range(cap, 0, -1):
        if n_tiles % d == 0:
            return d
    return cap


@dataclass
class MLPTKGConstantsDimensionSizes(nl.NKIObject):
    """
    Dimension sizes for the bf16 MLP TKG computation.

    ``H`` is whole, not sharded: this kernel runs on a single NeuronCore, so there is no
    cross-core split of the hidden dimension.
    """

    _pmax: int
    _psum_fmax: int
    _psum_bmax: int
    T: int
    H: int
    I: int
    H0: int
    H1: int
    I0: int
    I1: int
    column_tiling_dim: int
    column_tiling_factor: int
    max_I_shard_size: int


@dataclass
class MLPTKGConstantsWeightRing(nl.NKIObject):
    """The weight ring shared by the gate, up and down projections."""

    depth: int
    """Slots."""
    slot_elems: int
    """Elements per partition per slot (``_W_RING_SLOT_BYTES`` over the weight dtype's size)."""
    gate_up_h1_per_load: int
    """128-row H tiles per gate/up load: one load is ``[128, h1, I]``."""
    gate_up_loads: int
    """Loads per gate/up projection, ``ceil(H1 / h1)``."""
    down_i1_per_load: int
    """128-row I tiles per down load: one load is ``[128, i1, H_tile]``."""
    down_loads_per_h_tile: int
    """Loads per down H tile, ``ceil(I1 / i1)``."""
    down_h_tile: int
    """Columns of H the down projection accumulates per PSUM pass (whole H at the 4-way tile)."""


class MLPTKGConstants(nl.NKIObject):
    """Constants for the bf16 MLP TKG kernel implementation."""

    @staticmethod
    def calculate_constants(params: MLPParameters) -> MLPTKGConstantsDimensionSizes:
        """
        Calculate the dimension constants needed for the bf16 MLP TKG kernel.

        Args:
            params (MLPParameters): MLP configuration parameters.

        Returns:
            MLPTKGConstantsDimensionSizes: Dataclass with all computed dimension constants.
        """
        # --- Tile size constants ---
        _pmax = nl.tile_size.pmax  # Max partition dimension in SBUF
        _psum_fmax = nl.tile_size.psum_fmax  # Max free dim for psum
        _psum_bmax = 8  # Max batch dimension for psum

        T = params.batch_size * params.sequence_len
        H = params.hidden_size
        I = params.intermediate_size

        H0 = _pmax
        I0 = _pmax
        H1 = H // H0
        I1 = div_ceil(I, I0)

        # Hardware restriction: the moving tensor processes 512 elements per PSUM bank, with 8
        # PSUM banks. The gate/up projection issues one PSUM per 512 columns of I and does not
        # tile I further, so this is a hard upper bound on I (asserted by the projection).
        max_I_shard_size = _psum_fmax * _psum_bmax

        dim = column_tiling_dim(T, params.column_tiling)

        return MLPTKGConstantsDimensionSizes(
            _pmax=_pmax,
            _psum_fmax=_psum_fmax,
            _psum_bmax=_psum_bmax,
            T=T,
            H=H,
            I=I,
            H0=H0,
            H1=H1,
            I0=I0,
            I1=I1,
            column_tiling_dim=dim,
            column_tiling_factor=_pmax // dim,
            max_I_shard_size=max_I_shard_size,
        )

    @staticmethod
    def calculate_weight_ring(dims: MLPTKGConstantsDimensionSizes, weight_dtype) -> MLPTKGConstantsWeightRing:
        """
        Size the shared weight ring and each projection's load.

        The down projection accumulates ``H_tile / 512`` PSUM column tiles over ``column_tiling_factor``
        banks each; ``down_h_tile`` is the most H that fits the 8 banks, so the whole H = 6144 of GLM-5.2
        is one pass at the 4-way tile (3 banks) and the 2-way tile (6 banks), and two passes untiled.

        Args:
            dims (MLPTKGConstantsDimensionSizes): Precomputed dimension constants.
            weight_dtype: dtype of the three weight tensors (they must agree).

        Returns:
            MLPTKGConstantsWeightRing: Ring depth, slot size and per-projection load shapes.
        """
        depth, slot_bytes = _DENSE_RING_DEPTH, _W_RING_DEEP_SLOT_BYTES
        slot_elems = slot_bytes // sizeinbytes(weight_dtype)
        h1 = tiles_per_load(dims.H1, slot_elems // dims.I)
        h_tile = min(dims.H, dims.column_tiling_factor * dims._psum_bmax * dims._psum_fmax)
        i1 = tiles_per_load(dims.I1, slot_elems // h_tile)
        return MLPTKGConstantsWeightRing(
            depth=depth,
            slot_elems=slot_elems,
            gate_up_h1_per_load=h1,
            gate_up_loads=div_ceil(dims.H1, h1),
            down_i1_per_load=i1,
            down_loads_per_h_tile=div_ceil(dims.I1, i1),
            down_h_tile=h_tile,
        )
