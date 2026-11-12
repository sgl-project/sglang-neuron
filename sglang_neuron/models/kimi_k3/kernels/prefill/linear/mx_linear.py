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

"""Kimi-K3 prefill MXFP8 linear (Trn3): ``out = x @ W`` with an MXFP8 weight and an on-chip MXFP8 quantize of x."""

import nki
import nki.isa as nisa
import nki.language as nl

from ..utils.kernel_assert import kernel_assert
from ..utils.kernel_helpers import div_ceil

P_MAX = 128
MX_LANES = 4
"""FP8 values in one float8_e4m3fn_x4 element: the 4 lanes of one contraction row."""
K_TILE = P_MAX * MX_LANES
"""Contraction values of one nc_matmul_mx (128 partitions x 4 lanes)."""
SCALE_GROUP = 32
"""Contraction values per E8M0 scale (8 partitions x 4 lanes)."""
ORDERS = ("hT_quad", "consecutive")
"""hT_quad: partition p, lane q of K-tile g hold input 128 (4 g + q) + p. consecutive: input 512 g + 4 p + q."""
MAX_ROWS = 8192
MAX_CONTRACT = 16384
MAX_COLS = 16384

_GROUPS_PER_TILE = P_MAX // 8
_QUADRANT = 32
_SCALE_ROWS_PER_QUADRANT = 4
_OUT_CHUNK = 512  # FP32 columns of one 2 KiB PSUM bank: one matmul output never crosses a bank
_CHUNKS_PER_STATIONARY = 2  # output banks that reuse one stationary load
_RESIDENT_WEIGHT_BYTES = 112 * 1024  # per partition: weights + scales of all columns stay in SBUF up to this
_BLOCK_WEIGHT_BYTES = 64 * 1024  # per partition: one streamed column block (two are live)
_RESIDENT_PART_COLS = 2048  # column split of the resident weight DMAs, so the first matmuls start early
_X_GROUP_BYTES = 48 * 1024  # per partition: quantized row tiles kept across the column blocks
_QUANT_BYTES_PER_TILE = P_MAX * 4 + P_MAX  # x4 data + scales of one 128-row stationary per K-tile


@nki.jit
def prefill_mx_linear(
    x: nl.NkiTensor,
    w_mx: nl.NkiTensor,
    w_scale: nl.NkiTensor,
    order: str = "hT_quad",
    feature_major_x: bool = False,
) -> nl.NkiTensor:
    """
    MXFP8 linear of the Kimi-K3 prefill: ``out = x @ W`` for M = 1 .. 8192 rows, on one NeuronCore-v4 (LNC1).

    The kernel quantizes x to MXFP8 on chip (``quantize_mx``, one E8M0 scale per 32 contraction values in the
    groups of ``order``) and runs ``nc_matmul_mx`` against the MXFP8 weight with the activation as the stationary
    operand (128 rows) and the weight as the moving operand (512 columns), with FP32 PSUM accumulation over K.
    Built for the MLA o_proj (K 768, N 7168), the MoE latent projections (K 7168 / 3584, M 128 / 256) and the
    shared expert projections (K 7168 / 384).

    Dimensions:
        M: rows (tokens), 1 .. 8192
        K: contraction (input features); hT_quad K % 128 == 0, consecutive K % 32 == 0, 32 .. 16384
        N: output features, N % 32 == 0, 32 .. 16384
        G: K-tiles of 512, ceil(K / 512)

    Args:
        x (nl.NkiTensor): bfloat16 [M, K] @HBM, row-major; or [K, M] @HBM with ``feature_major_x``.
        w_mx (nl.NkiTensor): uint8 [R, 4 N] or [G, 128, 4 N] @HBM. Byte (r, 4 n + q) is the e4m3fn value of
            W[k, n], with r = 128 g + p and k = 128 (4 g + q) + p (hT_quad; R = 128 G, missing lanes zero) or
            k = 4 r + q (consecutive; R >= K / 4).
        w_scale (nl.NkiTensor): uint8 [S, N] @HBM. Row 16 g + s is the E8M0 scale of group s (partitions
            8 s .. 8 s + 7, 4 lanes) of K-tile g; S >= 16 G (hT_quad) or K / 32 (consecutive).
        order (str): "hT_quad" or "consecutive", the group order of the weight.
        feature_major_x (bool): x is [K, M] (element (k, m) = x[m, k]); the kernel then needs no transposes.

    Returns:
        out (nl.NkiTensor): bfloat16 [M, N] @HBM.

    Notes:
        - Accuracy class: MXFP8 x MXFP8 (e4m3fn + E8M0, the hardware quantize_mx rule), FP32 PSUM, BF16 output.
        - Row-major x: 4 PE transposes per (128 rows, K-tile), ScalarE lane interleave, VectorE quantize_mx.
          The transposes cost about K PE cycles per 128 rows against (K / 512) N for the MX matmuls.
        - Rows past M and contraction values past K are zero in SBUF, so they add nothing.

    Pseudocode:
        for each 128-row tile of x (one tile ahead of its matmuls):
            xq, xs = quantize_mx(interleave(transpose(x_tile)))       # [128 K/4, 128 rows] per K-tile
        for each column block (resident, or streamed one block ahead):
            w, ws = load(w_mx, w_scale)                                # [128, G, cols]
            for each row tile, for each pair of 512-column chunks:
                psum = sum_g nc_matmul_mx(xq[g], w[g, chunk], xs[g], ws[g, chunk])
                out[rows, chunk] = bf16(psum)
    """
    if feature_major_x:
        contract, rows = x.shape
    else:
        rows, contract = x.shape
    cols = w_scale.shape[1]
    out = nl.ndarray((rows, cols), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="prefill_mx_linear_out")
    prefill_mx_linear_body(x, w_mx, w_scale, out, order=order, feature_major_x=feature_major_x)
    return out


def prefill_mx_linear_body(
    x: nl.NkiTensor,
    w_mx: nl.NkiTensor,
    w_scale: nl.NkiTensor,
    out: nl.NkiTensor,
    order: str = "hT_quad",
    feature_major_x: bool = False,
) -> None:
    """The body of ``prefill_mx_linear`` for a caller kernel that inlines it (not ``@nki.jit``).

    Args:
        x, w_mx, w_scale, order, feature_major_x: as in ``prefill_mx_linear``.
        out (nl.NkiTensor): bfloat16 [M, N] @HBM, written in full.
    """
    kernel_assert(nisa.get_nc_version() >= nisa.nc_version.gen4, "prefill_mx_linear requires Trn3 or newer")
    kernel_assert(
        nl.program_ndim() == 0 or (nl.program_ndim() == 1 and nl.num_programs(0) == 1),
        "prefill_mx_linear supports LNC1 only",
    )
    kernel_assert(order in ORDERS, f"order must be one of {ORDERS}, got {order!r}")
    kernel_assert(x.dtype == nl.bfloat16, f"x must be bfloat16, got {x.dtype}")
    if feature_major_x:
        contract, rows = x.shape
    else:
        rows, contract = x.shape
    cols = w_scale.shape[1]
    ht_quad = order == "hT_quad"
    k_align = P_MAX if ht_quad else SCALE_GROUP
    kernel_assert(1 <= rows <= MAX_ROWS, f"M={rows} must be in 1 .. {MAX_ROWS}")
    kernel_assert(
        SCALE_GROUP <= contract <= MAX_CONTRACT and contract % k_align == 0,
        f"K={contract} must be a multiple of {k_align} in {SCALE_GROUP} .. {MAX_CONTRACT} for order {order}",
    )
    kernel_assert(cols % SCALE_GROUP == 0 and cols <= MAX_COLS, f"N={cols} must be a multiple of 32, <= {MAX_COLS}")
    k_tiles = div_ceil(contract, K_TILE)
    data_rows_needed = P_MAX * k_tiles if ht_quad else contract // MX_LANES
    scale_rows_needed = _GROUPS_PER_TILE * k_tiles if ht_quad else contract // SCALE_GROUP

    kernel_assert(w_mx.dtype == nl.uint8, f"w_mx must be uint8, got {w_mx.dtype}")
    if len(w_mx.shape) == 3:
        kernel_assert(w_mx.shape[1] == P_MAX, f"a 3-D w_mx must be [G, 128, 4N], got {w_mx.shape}")
        w_mx = w_mx.reshape((w_mx.shape[0] * P_MAX, w_mx.shape[2]))
    kernel_assert(
        len(w_mx.shape) == 2 and w_mx.shape[1] == MX_LANES * cols and w_mx.shape[0] >= data_rows_needed,
        f"w_mx must be [R >= {data_rows_needed}, {MX_LANES * cols}] uint8 for K={contract}, got {w_mx.shape}",
    )
    kernel_assert(
        w_scale.dtype == nl.uint8 and len(w_scale.shape) == 2 and w_scale.shape[0] >= scale_rows_needed,
        f"w_scale must be uint8 [S >= {scale_rows_needed}, N], got {w_scale.dtype} {w_scale.shape}",
    )
    kernel_assert(tuple(out.shape) == (rows, cols), f"out must be [{rows}, {cols}], got {out.shape}")
    # One x4 element (uint32) per weight row and column: same-width DMA, then an x4 view of the SBUF tile.
    w_words = w_mx.view(nl.uint32)

    bytes_per_col = k_tiles * (MX_LANES + 1)
    if cols * bytes_per_col <= _RESIDENT_WEIGHT_BYTES:
        block_cols = cols
    else:
        block_cols = max(_OUT_CHUNK, (_BLOCK_WEIGHT_BYTES // bytes_per_col) // _OUT_CHUNK * _OUT_CHUNK)
    n_blocks = div_ceil(cols, block_cols)
    m_tiles = div_ceil(rows, P_MAX)
    tiles = _k_tile_shapes(contract, k_tiles, ht_quad)

    def load_block(block_idx):
        col0 = block_idx * block_cols
        ncols = min(block_cols, cols - col0)
        parts = _RESIDENT_PART_COLS if n_blocks == 1 else ncols
        return _load_weight_block(w_words, w_scale, tiles, col0, ncols, parts)

    def quantize(m_tile):
        return _quantize_row_tile(x, m_tile, rows, contract, tiles, ht_quad, feature_major_x)

    evictions = [0]
    if n_blocks == 1:
        # Weights resident: one row tile is quantized ahead of the matmuls of the previous one.
        pending = quantize(0)
        weights = load_block(0)
        for m_tile in range(m_tiles):
            following = quantize(m_tile + 1) if m_tile + 1 < m_tiles else None
            _matmul_store(pending, weights, out, m_tile, rows, 0, cols, tiles, evictions)
            pending = following
        return

    # Weights streamed in column blocks, one block ahead. A group of row tiles keeps its quantized x over the blocks.
    group = max(1, min(m_tiles, _X_GROUP_BYTES // (k_tiles * _QUANT_BYTES_PER_TILE)))
    for first_tile in range(0, m_tiles, group):
        group_tiles = list(range(first_tile, min(m_tiles, first_tile + group)))
        quantized = [quantize(m_tile) for m_tile in group_tiles]
        pending = load_block(0)
        for block_idx in range(n_blocks):
            following = load_block(block_idx + 1) if block_idx + 1 < n_blocks else None
            col0 = block_idx * block_cols
            ncols = min(block_cols, cols - col0)
            for tile_idx, m_tile in enumerate(group_tiles):
                _matmul_store(quantized[tile_idx], pending, out, m_tile, rows, col0, ncols, tiles, evictions)
            pending = following


def _k_tile_shapes(contract: int, k_tiles: int, ht_quad: bool) -> list:
    """Return ``(valid partitions, padded partitions, valid lanes, scale groups)`` of each K-tile.

    hT_quad: every K-tile holds 128 partitions; the last one holds K / 128 - 4 (G - 1) valid lanes.
    consecutive: every K-tile holds 4 lanes; the last one holds (K - 512 (G - 1)) / 4 valid partitions, padded
    to 32, 64 or 128 for nc_matmul_mx (96 is padded to 128).
    """
    shapes = []
    for tile_idx in range(k_tiles):
        rest = contract - tile_idx * K_TILE
        if ht_quad:
            shapes.append((P_MAX, P_MAX, min(MX_LANES, rest // P_MAX), _GROUPS_PER_TILE))
            continue
        valid = min(P_MAX, rest // MX_LANES)
        padded = _QUADRANT if valid <= _QUADRANT else (2 * _QUADRANT if valid <= 2 * _QUADRANT else P_MAX)
        shapes.append((valid, padded, MX_LANES, valid // 8))
    return shapes


def _load_weight_block(
    w_words: nl.NkiTensor,
    w_scale: nl.NkiTensor,
    tiles: list,
    col0: int,
    ncols: int,
    part_cols: int,
) -> tuple:
    """Load the MXFP8 data and scales of columns ``col0 .. col0 + ncols`` of every K-tile into SBUF.

    Returns ``(data, scales)``: data float8_e4m3fn_x4 [128, G, ncols] and scales uint8 [128, G, ncols], with the
    scales of group s of K-tile g at partition 32 (s // 4) + s % 4 (the nc_matmul_mx quadrant layout, start 0).
    Padded partitions hold zero data and zero scales.
    """
    k_tiles = len(tiles)
    cols = w_scale.shape[1]
    data = nl.ndarray((P_MAX, k_tiles, ncols), dtype=nl.uint32, buffer=nl.sbuf)
    scales = nl.ndarray((P_MAX, k_tiles, ncols), dtype=nl.uint8, buffer=nl.sbuf)
    for tile_idx in range(k_tiles):
        valid, padded, _, _ = tiles[tile_idx]
        if valid < padded:
            nisa.memset(dst=data[0:padded, tile_idx, 0:ncols], value=0, engine=nisa.engine.gpsimd)
        for part0 in range(0, ncols, part_cols):
            part = min(part_cols, ncols - part0)
            nisa.dma_copy(
                dst=data[0:valid, tile_idx, part0 : part0 + part],
                src=w_words[P_MAX * tile_idx : P_MAX * tile_idx + valid, col0 + part0 : col0 + part0 + part],
            )

    # Scales: one DMA per quadrant for all K-tiles with 16 groups, then the quadrants of a partial last tile.
    full_tiles = sum(1 for shape in tiles if shape[3] == _GROUPS_PER_TILE)
    for quadrant in range(P_MAX // _QUADRANT):
        if full_tiles > 0:
            nisa.dma_copy(
                dst=scales[_QUADRANT * quadrant : _QUADRANT * quadrant + _SCALE_ROWS_PER_QUADRANT, 0:full_tiles, :],
                src=w_scale.ap(
                    pattern=[[cols, _SCALE_ROWS_PER_QUADRANT], [_GROUPS_PER_TILE * cols, full_tiles], [1, ncols]],
                    offset=_SCALE_ROWS_PER_QUADRANT * quadrant * cols + col0,
                ),
            )
    for tile_idx in range(full_tiles, k_tiles):
        _, padded, _, groups = tiles[tile_idx]
        for quadrant in range(padded // _QUADRANT):
            first = _SCALE_ROWS_PER_QUADRANT * quadrant
            count = max(0, min(_SCALE_ROWS_PER_QUADRANT, groups - first))
            row0 = _QUADRANT * quadrant
            if count < _SCALE_ROWS_PER_QUADRANT:
                nisa.memset(
                    dst=scales[row0 : row0 + _SCALE_ROWS_PER_QUADRANT, tile_idx, :],
                    value=0,
                    engine=nisa.engine.gpsimd,
                )
            if count > 0:
                scale_row = _GROUPS_PER_TILE * tile_idx + first
                nisa.dma_copy(
                    dst=scales[row0 : row0 + count, tile_idx, :],
                    src=w_scale[scale_row : scale_row + count, col0 : col0 + ncols],
                )
    return data.view(nl.float8_e4m3fn_x4), scales


def _quantize_row_tile(
    x: nl.NkiTensor,
    m_tile: int,
    rows: int,
    contract: int,
    tiles: list,
    ht_quad: bool,
    feature_major_x: bool,
) -> tuple:
    """Quantize rows ``128 m_tile ..`` of x into the MXFP8 stationary operand of every K-tile.

    Returns ``(data, scales)``: float8_e4m3fn_x4 [128, G, 128] and uint8 [128, G, 128]. Element (p, g, m) holds the
    4 lanes of contraction row p of K-tile g for row m of the tile. Rows past M and missing lanes or partitions
    are zero before the quantize.
    """
    k_tiles = len(tiles)
    row0 = m_tile * P_MAX
    row_count = min(P_MAX, rows - row0)
    data = nl.ndarray((P_MAX, k_tiles, P_MAX), dtype=nl.float8_e4m3fn_x4, buffer=nl.sbuf)
    scales = nl.ndarray((P_MAX, k_tiles, P_MAX), dtype=nl.uint8, buffer=nl.sbuf)

    if feature_major_x:
        for tile_idx in range(k_tiles):
            valid, padded, lanes, _ = tiles[tile_idx]
            # [p, lane, m]: lane q of partition p is contraction row 512 g + 128 q + p (hT) or 512 g + 4 p + q.
            lanes_sb = nl.ndarray((P_MAX, MX_LANES, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
            if valid < padded or lanes < MX_LANES or row_count < P_MAX:
                nisa.memset(dst=lanes_sb[0:padded, :, :], value=0.0, engine=nisa.engine.gpsimd)
            partition_step = rows if ht_quad else MX_LANES * rows
            lane_step = P_MAX * rows if ht_quad else rows
            nisa.dma_copy(
                dst=lanes_sb[0:valid, 0:lanes, 0:row_count],
                src=x.ap(
                    pattern=[[partition_step, valid], [lane_step, lanes], [1, row_count]],
                    offset=tile_idx * K_TILE * rows + row0,
                ),
            )
            interleaved = nl.ndarray((P_MAX, P_MAX, MX_LANES), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(
                dst=interleaved[0:padded, :, :],
                src=lanes_sb[0:padded, :, :].permute((0, 2, 1)),
                engine=nisa.engine.scalar,
            )
            nisa.quantize_mx(
                dst=data[0:padded, tile_idx, :],
                src=interleaved[0:padded, :, :],
                dst_scale=scales[0:padded, tile_idx, :],
            )
        return data, scales

    x_sb = nl.ndarray((P_MAX, contract), dtype=nl.bfloat16, buffer=nl.sbuf)
    if row_count < P_MAX:
        nisa.memset(dst=x_sb, value=0.0, engine=nisa.engine.gpsimd)
    nisa.dma_copy(dst=x_sb[0:row_count, :], src=x[row0 : row0 + row_count, :])

    def interleave_and_quantize(tile_idx, transposed, half):
        """Copy the transposed lanes [p, lane, m] of one K-tile to [p, m, lane] and quantize them."""
        valid, padded, lanes, _ = tiles[tile_idx]
        interleaved = nl.ndarray((P_MAX, P_MAX, MX_LANES), dtype=nl.bfloat16, buffer=nl.sbuf)
        if valid < padded or lanes < MX_LANES:
            nisa.memset(dst=interleaved[0:padded, :, :], value=0.0, engine=nisa.engine.gpsimd)
        nisa.tensor_copy(
            dst=interleaved[0:valid, :, 0:lanes],
            src=transposed.ap(
                pattern=[[2 * MX_LANES * P_MAX, valid], [1, P_MAX], [P_MAX, lanes]],
                offset=half * MX_LANES * P_MAX,
            ),
            engine=nisa.engine.scalar,
        )
        nisa.quantize_mx(
            dst=data[0:padded, tile_idx, :],
            src=interleaved[0:padded, :, :],
            dst_scale=scales[0:padded, tile_idx, :],
        )

    # One 2 KiB PSUM bank holds the transposes of 2 K-tiles. The copies of a bank follow the transposes of the
    # next bank, so the PE does not wait for them.
    pending = []
    for pair0 in range(0, k_tiles, 2):
        transposed = nl.ndarray((P_MAX, 2, MX_LANES, P_MAX), dtype=nl.bfloat16, buffer=nl.psum)
        for tile_idx in range(pair0, min(k_tiles, pair0 + 2)):
            valid, _, lanes, _ = tiles[tile_idx]
            half = tile_idx - pair0
            for lane in range(lanes):
                if ht_quad:
                    start = tile_idx * K_TILE + lane * P_MAX
                    lane_cols = x_sb[:, start : start + P_MAX]
                else:
                    start = tile_idx * K_TILE + lane
                    lane_cols = x_sb.slice(dim=1, start=start, end=start + MX_LANES * (valid - 1) + 1, step=MX_LANES)
                nisa.nc_transpose(dst=transposed[0:valid, half, lane, :], data=lane_cols, engine=nisa.engine.tensor)
        for args in pending:
            interleave_and_quantize(*args)
        pending = [(tile_idx, transposed, tile_idx - pair0) for tile_idx in range(pair0, min(k_tiles, pair0 + 2))]
    for args in pending:
        interleave_and_quantize(*args)
    return data, scales


def _matmul_store(
    quantized: tuple,
    weights: tuple,
    out: nl.NkiTensor,
    m_tile: int,
    rows: int,
    col0: int,
    ncols: int,
    tiles: list,
    evictions: list,
) -> None:
    """Multiply one quantized row tile by one weight column block and store its BF16 rows of ``out``.

    Each 512-column chunk accumulates over the K-tiles in its own FP32 PSUM bank at offset 0. Two chunks share each
    stationary load. The evictions of a chunk pair follow the matmuls of the next pair, and take VectorE for two of
    every three chunks and ScalarE for the third.
    """
    x_data, x_scales = quantized
    w_data, w_scales = weights
    row0 = m_tile * P_MAX
    row_count = min(P_MAX, rows - row0)
    out_sb = nl.ndarray((P_MAX, ncols), dtype=nl.bfloat16, buffer=nl.sbuf)
    n_chunks = div_ceil(ncols, _OUT_CHUNK)

    def evict(chunk_psums):
        for chunk_idx, psum in chunk_psums:
            chunk0 = chunk_idx * _OUT_CHUNK
            width = min(_OUT_CHUNK, ncols - chunk0)
            engine = nisa.engine.scalar if evictions[0] % 3 == 2 else nisa.engine.vector
            evictions[0] += 1
            nisa.tensor_copy(dst=out_sb[:, chunk0 : chunk0 + width], src=psum[:, 0:width], engine=engine)

    pending = []
    for first_chunk in range(0, n_chunks, _CHUNKS_PER_STATIONARY):
        chunk_ids = list(range(first_chunk, min(n_chunks, first_chunk + _CHUNKS_PER_STATIONARY)))
        psums = [nl.ndarray((P_MAX, _OUT_CHUNK), dtype=nl.float32, buffer=nl.psum) for _ in chunk_ids]
        for tile_idx in range(len(tiles)):
            _, padded, _, _ = tiles[tile_idx]
            for chunk_pos, chunk_idx in enumerate(chunk_ids):
                chunk0 = chunk_idx * _OUT_CHUNK
                width = min(_OUT_CHUNK, ncols - chunk0)
                nisa.nc_matmul_mx(
                    dst=psums[chunk_pos][:, 0:width],
                    stationary=x_data[0:padded, tile_idx, :],
                    moving=w_data[0:padded, tile_idx, chunk0 : chunk0 + width],
                    stationary_scale=x_scales[0:padded, tile_idx, :],
                    moving_scale=w_scales[0:padded, tile_idx, chunk0 : chunk0 + width],
                    accumulate=tile_idx > 0,
                )
        evict(pending)
        pending = list(zip(chunk_ids, psums))
    evict(pending)
    nisa.dma_copy(dst=out[row0 : row0 + row_count, col0 : col0 + ncols], src=out_sb[0:row_count, :])
