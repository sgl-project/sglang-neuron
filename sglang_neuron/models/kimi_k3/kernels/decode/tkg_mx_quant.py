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

"""MXFP8 operands of the Kimi-K3 decode projections (Trn3).

A decode projection takes a token-major hT activation ``x_hT`` ``[128, T, J]``: element ``(p, t, j)`` is
``x[t, 128 j + p]``. Its MXFP8 form packs the 4 hidden tiles ``4 g .. 4 g + 3`` of one partition into the 4 lanes of
one x4 element, so a 512-row contraction group ``g`` needs only a free-axis copy of the activation, no transpose.
The 32 values of one MX scale are then the rows ``{128 (4 g + q) + 8 s + r : q < 4, r < 8}`` (the "hT_quad" order
of the host weights, ``mx_weights_torch``).

HBM layouts of an MXFP8 weight ``[in, out]`` (``in = 512 G``):

* data ``[G, 128, 4 out]`` float8_e4m3fn (or uint8): byte ``(g, p, 4 n + q)`` is the e4m3 value of row
  ``128 (4 g + q) + p``, column ``n``. One partition loads one contiguous run of each group.
* scales ``[128, S, out]`` uint8, ``S = ceil(G / 8)``: the 16 scale rows of group ``g = 8 v + u`` sit in slot ``v``
  at partitions ``32 c + 4 u + i`` (scale row ``s = 4 c + i``), the quadrant layout of ``nc_matmul_mx``. Eight
  groups share one slot, so one DMA loads all the scales of a weight.

A scale operand of nc_matmul_mx can start at partition 4 u only up to ``MX_SCALE_START_MAX`` (the Trn3 ISA checks
refuse starts of 16 and more). ``load_mx_scales`` therefore loads the scales into ``MX_SCALE_COPIES`` SBUF copies:
copy k holds the partitions from ``k * MX_SCALE_SHIFT`` on, at partition 0. A moving operand must also start on an
even x4 column.
"""

import nki.isa as nisa
import nki.language as nl

from .utils.kernel_assert import kernel_assert
from .utils.kernel_helpers import div_ceil

P_MAX = 128
MX_LANES = 4
"""Contraction values of one x4 element: the hidden tiles of one MX group."""
MX_GROUP_ROWS = P_MAX * MX_LANES
"""Contraction rows of one MX group (one nc_matmul_mx)."""
MX_SCALE_ROWS = 4
"""Scale rows of one MX group in each 32-partition quadrant."""
MX_SCALE_SLOTS = 8
"""MX groups whose scales share one ``[128, out]`` slot."""
MX_SCALE_START_MAX = 12
"""Largest scale partition start of an nc_matmul_mx operand (MEASURED on Trn3: 0 .. 12 pass, 16 .. 28 fail)."""
_GROUPS_PER_COPY = MX_SCALE_START_MAX // MX_SCALE_ROWS + 1
MX_SCALE_COPIES = (MX_SCALE_SLOTS + _GROUPS_PER_COPY - 1) // _GROUPS_PER_COPY
"""SBUF copies of the scales of one MXFP8 weight."""
MX_SCALE_SHIFT = MX_SCALE_ROWS * _GROUPS_PER_COPY
"""Partitions between two scale copies."""


def mx_hT_shapes(rows: int, cols: int, fp4: bool = False) -> tuple[tuple, tuple]:
    """Return the HBM shapes ``(data, scales)`` of an MX weight ``[rows, cols]`` in the hT_quad layout.

    MXFP8 data is uint8 ``[G, 128, 4 cols]``; MXFP4 data (``fp4``) is uint16 ``[G, 128, cols]``, one
    ``float4_e2m1fn_x4`` element (4 lanes, lane ``q`` in bits ``4 q .. 4 q + 3``) per column.
    """
    kernel_assert(rows % MX_GROUP_ROWS == 0, f"an MX weight needs rows % {MX_GROUP_ROWS} == 0, got {rows}")
    groups = rows // MX_GROUP_ROWS
    width = cols if fp4 else MX_LANES * cols
    return (groups, P_MAX, width), (P_MAX, div_ceil(groups, MX_SCALE_SLOTS), cols)


def mx_weight_format(w: nl.NkiTensor) -> tuple[bool, int, object]:
    """Return ``(fp4, cols, x4 dtype)`` of an hT_quad MX weight: uint16 data is MXFP4, uint8 / float8 data MXFP8."""
    fp4 = w.dtype == nl.uint16
    cols = w.shape[2] if fp4 else w.shape[2] // MX_LANES
    return fp4, cols, nl.float4_e2m1fn_x4 if fp4 else nl.float8_e4m3fn_x4


def load_mx_scales(w_scale: nl.NkiTensor, alloc, load) -> list:
    """Load the scales ``[128, S, N]`` of an MXFP8 weight into its ``MX_SCALE_COPIES`` SBUF copies.

    ``alloc(shape, copy)`` returns the SBUF tile of a copy; ``load(dst, src, copy)`` issues its DMA.
    """
    copies = []
    for copy in range(MX_SCALE_COPIES):
        shift = copy * MX_SCALE_SHIFT
        tile = alloc(tuple(w_scale.shape), copy)
        rows, slots, cols = w_scale.shape
        load(tile[0 : rows - shift, 0:slots, 0:cols], w_scale[shift:rows, 0:slots, 0:cols], copy)
        copies.append(tile)
    return copies


def mx_scale_view(scales: list, group: int, col: int, cols: int) -> nl.NkiTensor:
    """Return the stationary or moving scale operand of MX group ``group``, columns ``col .. col + cols``."""
    lane = group % MX_SCALE_SLOTS
    first = MX_SCALE_ROWS * (lane % _GROUPS_PER_COPY)
    return scales[lane // _GROUPS_PER_COPY][first:P_MAX, group // MX_SCALE_SLOTS, col : col + cols]


def _sbuf_tile(shape, dtype, name):
    return nl.ndarray(shape, dtype=dtype, buffer=nl.sbuf, name=name)


def quantize_hT_mx(
    x_hT: nl.NkiTensor,
    tag: str,
    padded: int = None,
    alloc=_sbuf_tile,
    tile_major: bool = False,
    lanes_view: bool = False,
    done: list = None,
) -> tuple[nl.NkiTensor, nl.NkiTensor]:
    """Return the MXFP8 operand ``(data [128, G, Tp] float8_e4m3fn_x4, scales [128, G, Tp] uint8)`` of ``x_hT``.

    VectorE copies lane ``q`` of each group (hidden tiles ``4 g + q``) next to the other lanes, then quantize_mx
    takes one scale per 8 partitions x 4 lanes. ``padded`` (default T) adds zero token columns, for a stationary
    operand (its free size must be even). A tile count that is not a multiple of 4 gets zero lanes after the last
    tile (the padded-lane form of the KDA output projection: 3 heads and a zero lane). ``alloc(shape, dtype, name)``
    returns the SBUF tiles. ``tile_major``: ``x_hT`` is ``[128, J, T]`` (element ``(p, j, t)`` is
    ``x[t, 128 j + p]``), as the gathered MoE tokens.
    ``lanes_view``: quantize_mx reads the lanes of ``x_hT`` through a strided view (groups, tokens, lanes), with no
    VectorE copies, when no zero token column or zero lane is needed (``padded`` = T and a tile count that is a
    multiple of 4; else the copies stay). The same values in the same order, so the result does not change.
    ``done``: optional list; the quantize_mx instruction is appended.
    """
    n_tok, tiles = (x_hT.shape[2], x_hT.shape[1]) if tile_major else (x_hT.shape[1], x_hT.shape[2])
    padded = n_tok if padded is None else padded
    groups = div_ceil(tiles, MX_LANES)
    if lanes_view and not tile_major and padded == n_tok and tiles % MX_LANES == 0:
        data = alloc((P_MAX, groups, padded), nl.float8_e4m3fn_x4, f"{tag}_mx_x")
        scales = alloc((P_MAX, groups, padded), nl.uint8, f"{tag}_mx_x_scale")
        lanes_of_x = x_hT[0:P_MAX, 0:n_tok, 0:tiles].reshape_dim(2, (groups, MX_LANES)).permute((0, 2, 1, 3))
        quantize = nisa.quantize_mx(dst=data, src=lanes_of_x, dst_scale=scales)
        if done is not None:
            done.append(quantize)
        return data, scales
    lanes = alloc((P_MAX, groups, padded, MX_LANES), x_hT.dtype, f"{tag}_mx_lanes")
    if padded > n_tok or tiles % MX_LANES:
        nisa.memset(dst=lanes, value=0.0)
    for lane in range(min(MX_LANES, tiles)):
        lane_groups = len(range(lane, tiles, MX_LANES))
        if tile_major:
            src = x_hT[0:P_MAX, lane:tiles:MX_LANES, 0:n_tok]
        else:
            src = x_hT[0:P_MAX, 0:n_tok, lane:tiles:MX_LANES].permute((0, 2, 1))
        nisa.tensor_copy(dst=lanes[0:P_MAX, 0:lane_groups, 0:n_tok, lane], src=src, engine=nisa.engine.vector)
    data = alloc((P_MAX, groups, padded), nl.float8_e4m3fn_x4, f"{tag}_mx_x")
    scales = alloc((P_MAX, groups, padded), nl.uint8, f"{tag}_mx_x_scale")
    quantize = nisa.quantize_mx(dst=data, src=lanes, dst_scale=scales)
    if done is not None:
        done.append(quantize)
    return data, scales


MX_QUADRANT_ROWS = 32
"""PE rows (partitions) of one MX quadrant row tile."""
MX_COMPACT_ROWS = (P_MAX // MX_QUADRANT_ROWS) * MX_SCALE_ROWS
"""Scale rows of one MX group, 4 in each quadrant: the compact HBM scales ``[16, N]`` of a one-group weight."""


def load_mx_compact_scales(compact: nl.NkiTensor, tile: nl.NkiTensor, load) -> list:
    """Load the compact scales ``[16, N]`` of a one-group MXFP8 weight into the quadrant rows of ``tile`` ``[128, N]``.

    Compact row ``4 c + i`` (quadrant ``c``, partition group ``i``) goes to partition ``32 c + i``, the quadrant layout
    of nc_matmul_mx (``mx_scale_view`` of group 0). The other partitions of ``tile`` are not read. The values equal
    those of the ``[128, 1, N]`` form, at 1/8 of its HBM bytes. ``load(dst, src, quadrant)`` issues one DMA a quadrant
    and returns it; the DMAs are returned.
    """
    rows, cols = compact.shape
    kernel_assert(rows == MX_COMPACT_ROWS, f"compact MX scales need {MX_COMPACT_ROWS} rows, got {rows}")
    loads = []
    for quadrant in range(P_MAX // MX_QUADRANT_ROWS):
        first = quadrant * MX_QUADRANT_ROWS
        row = quadrant * MX_SCALE_ROWS
        loads.append(
            load(tile[first : first + MX_SCALE_ROWS, 0:cols], compact[row : row + MX_SCALE_ROWS, 0:cols], quadrant)
        )
    return loads


def quantize_quadrant_rows_mx(rows_src: nl.NkiTensor, rows: int, heads: int, padded: int, tag: str, alloc, copy=None):
    """Return the MXFP8 quadrant operand ``(data [32 heads, Tp] x4, scales [32 heads, Tp] uint8)`` of ``rows_src``.

    ``rows_src`` is ``[T, heads * 128]`` in PSUM or SBUF (row-major, 128 contraction values a head). Head ``h`` takes
    the partitions ``32 h .. 32 h + 31`` of one quadrant: partition ``32 h + p``, lane ``l`` holds value ``32 l + p`` of
    the head. One copy (the read-out of a PSUM source) writes the BF16 rows in lane-major order ``[T, lane, head, 32]``.
    The TensorE transpose of a lane then reads one contiguous run of ``32 heads`` values (an LdWeights operand must be
    one strided run) and writes the heads side by side at PSUM partition 0 (no offset destination). One copy puts the
    lanes next to each other (zero columns past ``rows`` up to ``padded``), and quantize_mx takes one scale per 8
    partitions x 4 lanes (``mx_quadrant_quantize`` on the host). ``copy`` (optional) wraps the two copies (an order
    hook); ``alloc(shape, dtype, name)`` returns the SBUF tiles.
    """
    bf16 = nl.bfloat16
    width = heads * MX_QUADRANT_ROWS
    kernel_assert(heads * MX_QUADRANT_ROWS <= P_MAX and padded % 2 == 0, "the quadrant operand needs <= 4 heads")
    kernel_assert(
        tuple(rows_src.shape) == (rows, MX_LANES * width),
        f"the quadrant rows must be [{rows}, {MX_LANES * width}], got {tuple(rows_src.shape)}",
    )
    lane_rows = alloc((rows, MX_LANES * width), bf16, f"{tag}_mxq_rows")
    rows_copy = nisa.tensor_copy(
        dst=lane_rows.reshape_dim(1, (MX_LANES, heads, MX_QUADRANT_ROWS)),
        src=rows_src.reshape_dim(1, (heads, MX_LANES, MX_QUADRANT_ROWS)).permute((0, 2, 1, 3)),
    )
    if copy is not None:
        copy(rows_copy)
    lanes_psum = nl.ndarray((width, MX_LANES, padded), dtype=bf16, buffer=nl.psum)
    for lane in range(MX_LANES):
        nisa.nc_transpose(
            dst=lanes_psum[0:width, lane, 0:rows], data=lane_rows[0:rows, lane * width : (lane + 1) * width]
        )
    lanes = alloc((width, padded, MX_LANES), bf16, f"{tag}_mxq_lanes")
    if padded > rows:
        nisa.memset(dst=lanes, value=0.0)
    lane_copy = nisa.tensor_copy(
        dst=lanes[0:width, 0:rows, 0:MX_LANES], src=lanes_psum[0:width, 0:MX_LANES, 0:rows].permute((0, 2, 1))
    )
    if copy is not None:
        copy(lane_copy)
    data = alloc((width, padded), nl.float8_e4m3fn_x4, f"{tag}_mxq_x")
    scales = alloc((width, padded), nl.uint8, f"{tag}_mxq_x_scale")
    nisa.quantize_mx(dst=data, src=lanes, dst_scale=scales)
    return data, scales


def quadrant_absorb_mx(dsts: list, data, scales, w_x4, w_scales, cols: int, first: int, count: int):
    """Write head ``h`` of ``data`` (``quantize_quadrant_rows_mx``) times the quadrant MXFP8 weight ``w_x4`` (columns
    ``0 .. cols``) to ``dsts[h]`` ``[>= count, cols]`` (FP32 PSUM, one bank each, cols <= 512), for the columns
    ``first .. first + count`` of ``data``.

    Head ``h`` is one nc_matmul_mx row tile at PE rows ``32 h`` (``tile_size (32, 128)``), column position 0 and PSUM
    partition 0, so the heads run at the same time. Head 0 runs in two halves of N, first and last, so the group of
    tiles starts and ends on quadrant 0, the row group of the partial BF16 transposes nearby: those keep the
    compiler's same-row-group edge, and no partial MX tile streams beside a partial non-MX tile on another row
    group (a hazard MEASURED wrong on Trn3). All tiles are in one ``nl.no_reorder`` block with no other TensorE op.
    The split moves no value: each output column is the same dot product. A scale operand is the 4 scale rows at the
    start of its quadrant (the row-tile form of the library's MXFP8 decode attention).
    """
    heads = len(dsts)
    kernel_assert(cols % 4 == 0 and count % 2 == 0, "the quadrant matmuls need cols % 4 == 0 and an even count")
    half = cols // 2
    parts = [(0, 0, half)] + [(head, 0, cols) for head in range(1, heads)] + [(0, half, cols)]
    with nl.no_reorder():
        for head, lo, hi in parts:
            row = head * MX_QUADRANT_ROWS
            nisa.nc_matmul_mx(
                dst=dsts[head][0:count, lo:hi],
                stationary=data[row : row + MX_QUADRANT_ROWS, first : first + count],
                moving=w_x4[row : row + MX_QUADRANT_ROWS, lo:hi],
                stationary_scale=scales[row : row + MX_SCALE_ROWS, first : first + count],
                moving_scale=w_scales[row : row + MX_SCALE_ROWS, lo:hi],
                tile_position=(row, 0),
                tile_size=(MX_QUADRANT_ROWS, P_MAX),
            )


def quantize_heads_mx(latent_t: nl.NkiTensor, rows: int, padded: int, tag: str, alloc):
    """Return the MXFP8 operand ``(data [128, H, Tp] x4, scales [128, H, Tp] uint8)`` of the transposed latent
    ``latent_t`` ``[128, 4, H, T]`` (element ``(p, k, h, t)`` is latent ``128 k + p`` of head ``h``, row ``t``).

    Group ``h`` holds head ``h``: lane ``k`` is latent tile ``k`` (the hT_quad order of one 512-row group), one copy
    and one quantize_mx for all heads.
    """
    _, tiles, heads, _ = latent_t.shape
    kernel_assert(tiles == MX_LANES, f"the MXFP8 latent needs {MX_LANES} latent tiles, got {tiles}")
    lanes = alloc((P_MAX, heads, padded, MX_LANES), latent_t.dtype, f"{tag}_mxv_lanes")
    if padded > rows:
        nisa.memset(dst=lanes, value=0.0)
    nisa.tensor_copy(
        dst=lanes[0:P_MAX, 0:heads, 0:rows, 0:MX_LANES],
        src=latent_t[0:P_MAX, 0:MX_LANES, 0:heads, 0:rows].permute((0, 2, 3, 1)),
    )
    data = alloc((P_MAX, heads, padded), nl.float8_e4m3fn_x4, f"{tag}_mxv_x")
    scales = alloc((P_MAX, heads, padded), nl.uint8, f"{tag}_mxv_x_scale")
    nisa.quantize_mx(dst=data, src=lanes, dst_scale=scales)
    return data, scales
