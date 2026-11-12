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

"""Kimi-K3 routed-expert tile for intermediate slices below the 512-row MX tile (Trn3, LNC1).

An EP_g x TP_r split gives each rank the intermediate slice I_s = 3072 / r of its experts. The served EP128 x TP2 body
tiles the down contraction in 512-row MX tiles, so it needs I_s % 512 == 0 (``moe_2srv_split``). This module runs one
local expert (gate/up -> SiTU -> down) on a compact tile of at most 128 tokens for I_s in ``TILE_WIDTHS``:

    I_s   TP   down MX tiles (contraction rows)   MX matmul partitions
    768    4   512 + 256                           128 + 64
    384    8   384                                 96
    192   16   192 (rows 48..63 zero)              64
     96   32    96 (rows 24..31 zero)              32

The MX matmul contracts over partitions x 4 lanes, and its partition count must be a multiple of 32 (128 values). Each
down tile holds whole 32-element E8M0 groups of the checkpoint, so every weight keeps its checkpoint scale (exact
dequant, no requantize), and the SiTU output is quantized to MXFP8 on the same 32-element groups as the served body.

Operand orientation (as the served ``moe_tkg_decode``): the tokens are the stationary operand and the weights stream as
the moving operand. A matmul costs about its moving columns (the same per column on trn3, independent of the contraction
size), so one call costs one pass of the expert's weights for any token count up to 128.

MX matmuls a call: I_s 768: 21 gate/up + 14 down; 384: 14 + 7; 192: 7 + 7; 96: 7 + 7. A 768 down tile split as
3 x 256 would take 21 down matmuls at the same cost each, so the 512 + 256 split is used.

Layouts (per rank, per I-slice; ``tile_weight_shapes``):

- Kernel I order. The I_s slice rows (checkpoint intermediate rows of the rank, in order) split into the down tiles of
  ``tile_plan``. Inside a tile of ``rows`` = width / 4 partitions that starts at slice row f, kernel I column
  f + q * rows + p holds slice row f + 4 p + q (p < rows, q < 4). Partition p, lane q of the tile's down operand is
  then the same row, and partitions 8k .. 8k + 7 (4 lanes) hold slice rows f + 32k .. f + 32k + 31: checkpoint group
  f / 32 + k.
- Gate/up weights: uint16 (float4_e2m1fn_x4) [E_L, 128, 7, 2 I_s]. Partition p, H tile hb and lane l hold hidden index
  512 hb + 4 p + l (the served layout). The 2 I_s columns of an H tile are the gate/up chunks of ``tile_plan``: chunk
  (c0, w) takes columns 2 c0 .. 2 c0 + w - 1 for gate kernel columns c0 .. c0 + w - 1 and the next w columns for up.
- Gate/up scales: uint8 [E_L, 16, 7, 2 I_s]: row s is the E8M0 scale of partitions 8s .. 8s + 7 (checkpoint group
  16 hb + s of the column's row).
- Down weights: uint16 (float4_e2m1fn_x4) [E_L, I_s / 4, H]: row m holds slice rows 4m .. 4m + 3 (lane q = row
  4m + q) of every hidden column: the checkpoint down bytes (2m, 2m + 1) of the slice.
- Down scales: uint8 [E_L, I_s / 32, H]: row g is checkpoint group g of the slice.

The SBUF forms the tile reads come from ``alloc_tile_weights`` and ``load_tile_weights`` (the expert index may be a
runtime value, so a dynamic loop can stream only its active experts).

Packed form (``routed_expert_tile_packed``, I_s in ``PACKED_WIDTHS``): one down contraction of 128 rows holds 1.3 to
5.3 slices of 96 / 48 / 24 rows, so K experts share each down matmul. The tile rows are (token, expert) pairs, each
expert's activation is zero outside its own rows, and the contraction is block-diagonal: every pair row gets the
FP32 down output of its own expert, then its FP32 affinity, as in ``routed_expert_tile``. At I_s 96 and K 16: 112
gate/up + 21 down MX matmuls for 16 experts, against 16 x 14 for 16 single calls. The weights of the K experts
share one SBUF set (``alloc_packed_weights``, ``load_packed_weights``), so the down rows of one load fill up to 128
partitions.
"""

from typing import NamedTuple

import nki.isa as nisa
import nki.language as nl

from ..utils.kernel_assert import kernel_assert
from ..utils.kernel_helpers import div_ceil
from ..tkg_moe import LATENT_SIZE, SITU_BETA, SITU_LINEAR_BETA

TILE_WIDTHS = (768, 384, 192, 96)
"""The intermediate slices of the tile (I_s = 3072 / routed_tp for routed_tp 4, 8, 16, 32)."""
MX_BLOCK = 32
"""Elements of one E8M0 scale group along the contraction."""
N_H512 = LATENT_SIZE // 512
"""Gate/up contraction tiles of 512 hidden values (7)."""
MAX_TILE_TOKENS = 128
"""Most tokens of one tile: the stationary free size of an MX matmul."""
GATE_UP_SCALE_ROWS = 16
"""Dense E8M0 rows of one gate/up column: one for each 8 partitions."""

_PMAX = 128
_Q = 4  # x4 lanes of an MX element
_MX_TILE = 512  # contraction of one full MX matmul (128 partitions x 4 lanes)
_BLOCK_ROWS = 8  # partitions of one E8M0 group
_QUADRANT = 32
_SCALE_ROWS = 4  # E8M0 rows of one SBUF quadrant
_GATE_UP_CHUNK = 256  # I columns of a gate/up chunk: its gate and up columns fill one 512-column FP32 PSUM bank
_DOWN_COLS = 512  # hidden columns of one down matmul (one FP32 PSUM bank)
_DOWN_GROUP = 4  # down PSUM banks live at once: the activation tile stays stationary over 4 matmuls
_GROUP_BLOCKS = _PMAX // _BLOCK_ROWS  # E8M0 groups of one 128-row down group
ROW_TILED_TAIL_WIDTHS = (768,)
"""The I-slices with a row-tiled tail option (``alloc_tile_weights`` ``row_tiled_tail``): the 256-row down tile."""
_TAIL_ROWS = 64  # partitions of the 768 tail tile: one 64-row PE tile


class DownTile(NamedTuple):
    """One down MX tile of an I-slice."""

    first: int
    """First slice row, and first kernel I column, of the tile."""
    width: int
    """Slice rows of the tile (its contraction)."""
    rows: int
    """Partitions with data: width / 4."""
    partitions: int
    """Partitions of the MX matmul: rows rounded up to a multiple of 32; the rows past ``rows`` are zero."""


class TilePlan(NamedTuple):
    """The down tiles and gate/up chunks of an I-slice (``tile_plan``)."""

    i_slice: int
    down_tiles: tuple
    """``DownTile`` of each down MX tile, in slice-row order."""
    chunks: tuple
    """(first kernel I column, width) of each gate/up chunk: at most 256 columns, inside one down tile."""


def tile_plan(i_slice: int) -> TilePlan:
    """Return the down tiles and the gate/up chunks of I-slice ``i_slice`` (one of ``TILE_WIDTHS``).

    Example:
        >>> [tile.width for tile in tile_plan(768).down_tiles]
        [512, 256]
    """
    kernel_assert(
        i_slice in TILE_WIDTHS,
        f"The routed-expert tile takes an intermediate slice in {TILE_WIDTHS}, got {i_slice}",
    )
    tiles = []
    for first in range(0, i_slice, _MX_TILE):
        width = min(_MX_TILE, i_slice - first)
        rows = width // _Q
        tiles.append(DownTile(first, width, rows, div_ceil(rows, _QUADRANT) * _QUADRANT))
    chunks = tuple(
        (tile.first + col, min(_GATE_UP_CHUNK, tile.width - col))
        for tile in tiles
        for col in range(0, tile.width, _GATE_UP_CHUNK)
    )
    return TilePlan(i_slice, tuple(tiles), chunks)


def tile_matmul_counts(i_slice: int) -> dict:
    """Return the MX matmuls and the PE transposes of one ``routed_expert_tile`` call.

    Example:
        >>> tile_matmul_counts(768)
        {'gate_up': 21, 'down': 14, 'transposes': 8}
    """
    plan = tile_plan(i_slice)
    return {
        "gate_up": N_H512 * len(plan.chunks),
        "down": (LATENT_SIZE // _DOWN_COLS) * len(plan.down_tiles),
        "transposes": _Q * len(plan.down_tiles),
    }


def kernel_column_rows(i_slice: int) -> tuple:
    """Return the slice row of each kernel I column: column f + q * rows + p of a down tile holds row f + 4p + q."""
    rows = []
    for tile in tile_plan(i_slice).down_tiles:
        rows.extend(tile.first + 4 * p + q for q in range(_Q) for p in range(tile.rows))
    return tuple(rows)


def gate_up_columns(i_slice: int) -> tuple:
    """Return (projection, kernel I column) of each of the 2 I_s columns of an H tile: projection 0 gate, 1 up."""
    columns = []
    for first, width in tile_plan(i_slice).chunks:
        columns.extend((projection, first + col) for projection in range(2) for col in range(width))
    return tuple(columns)


def tile_weight_shapes(i_slice: int, local_experts: int) -> dict:
    """Return the HBM shapes of one rank's expert tensors for I-slice ``i_slice`` (the module docstring layout).

    Example:
        >>> tile_weight_shapes(768, 14)["expert_down_weights"]
        (14, 192, 3584)
    """
    tile_plan(i_slice)
    return {
        "expert_gate_up_weights": (local_experts, _PMAX, N_H512, 2 * i_slice),
        "expert_gate_up_weights_scale": (local_experts, GATE_UP_SCALE_ROWS, N_H512, 2 * i_slice),
        "expert_down_weights": (local_experts, i_slice // _Q, LATENT_SIZE),
        "expert_down_weights_scale": (local_experts, i_slice // MX_BLOCK, LATENT_SIZE),
    }


def tail_halves(n_chunks: int = LATENT_SIZE // _DOWN_COLS) -> tuple:
    """Return (half, SBUF column) of each 512-column hidden chunk of the row-tiled tail.

    In each group of 4 chunks (the down PSUM banks live at once) the first half of the chunks goes to partitions
    0 .. 63 (half 0) and the rest to partitions 64 .. 127 (half 1), so each pair of a group can run on the two
    64-row PE tiles at once. A half packs its chunks side by side.

    Example:
        >>> tail_halves()
        ((0, 0), (0, 512), (1, 0), (1, 512), (0, 1024), (0, 1536), (1, 1024))
    """
    out, used = [], [0, 0]
    for group_first in range(0, n_chunks, _DOWN_GROUP):
        group = range(group_first, min(group_first + _DOWN_GROUP, n_chunks))
        split = div_ceil(len(group), 2)
        for position in range(len(group)):
            half = 0 if position < split else 1
            out.append((half, used[half] * _DOWN_COLS))
            used[half] += 1
    return tuple(out)


def _tail_order(group: range) -> list:
    """The chunks of a group in the issue order of the row-tiled tail: half 0 and half 1 alternate."""
    halves = tail_halves()
    first = [chunk for chunk in group if halves[chunk][0] == 0]
    second = [chunk for chunk in group if halves[chunk][0] == 1]
    order = []
    for index in range(max(len(first), len(second))):
        order.extend(chunks[index] for chunks in (first, second) if index < len(chunks))
    return order


def alloc_tile_weights(i_slice: int, name: str, zero_engine=nisa.engine.gpsimd, row_tiled_tail: bool = False) -> dict:
    """Allocate the SBUF weights of one expert for ``routed_expert_tile`` and zero the padding of the narrow tiles.

    Returns a dict of 2-D SBUF views:
        - gate_up_w: uint16 [128, 7 * 2 I_s], column hb * 2 I_s + c = column c of H tile hb.
        - gate_up_s: uint8 [128, 7 * 2 I_s], E8M0 row 4q + r at partition 32q + r.
        - down_w: uint16 [128, n_tiles * H], tile t in columns t * H .. (t + 1) * H, its rows on partitions
          0 .. rows - 1, zero on partitions rows .. partitions - 1.
        - down_s: uint8 [124, n_tiles * H], a view at partition 4 of the gate/up scale buffer (quadrant rows 4 .. 7),
          group k of tile t at partition 32 (k // 4) + k % 4 of the view.
        - scales, down_scale_offset: the scale buffer and the partition of the down view in it (for the loads).
        - tail_s (with ``row_tiled_tail``): uint8 [128, 2048], the tail scales of both halves.
    The loads (``load_tile_weights``) write only the data rows, so the zero rows stay zero for every expert.

    ``row_tiled_tail`` (I-slice 768 only, default False): the 64-row tail tile of each hidden chunk goes to partitions
    0 .. 63 or 64 .. 127 (``tail_halves``), in columns H .. H + 2048 of down_w, with its scales in tail_s. Its loads
    then fill all 128 partitions (two 64-partition halves), and its down matmuls use the 64-row PE tiles, two at a
    time when the PE runs them concurrently. The values do not change.
    """
    plan = tile_plan(i_slice)
    kernel_assert(
        not row_tiled_tail or i_slice in ROW_TILED_TAIL_WIDTHS,
        f"row_tiled_tail takes an I-slice in {ROW_TILED_TAIL_WIDTHS}, got {i_slice}",
    )
    gate_up_cols = N_H512 * 2 * i_slice
    down_cols = len(plan.down_tiles) * LATENT_SIZE
    gate_up_w = nl.ndarray((_PMAX, gate_up_cols), dtype=nl.uint16, buffer=nl.sbuf, name=f"{name}_gate_up_w")
    down_w = nl.ndarray((_PMAX, down_cols), dtype=nl.uint16, buffer=nl.sbuf, name=f"{name}_down_w")
    # The gate/up scales take quadrant rows 0..3 and the down scales rows 4..7 of one buffer (MX scale offsets 0, 4).
    scales = nl.ndarray((_PMAX, max(gate_up_cols, down_cols)), dtype=nl.uint8, buffer=nl.sbuf, name=f"{name}_scales")
    for index, tile in enumerate(plan.down_tiles):
        if tile.partitions == tile.rows:
            continue
        cols = slice(index * LATENT_SIZE, (index + 1) * LATENT_SIZE)
        nisa.memset(
            dst=down_w[0 : tile.partitions, cols], value=0, engine=zero_engine, name=f"{name}_zero_down_w{index}"
        )
        nisa.memset(
            dst=scales[0 : tile.partitions, cols], value=0, engine=zero_engine, name=f"{name}_zero_down_s{index}"
        )
    weights = {
        "gate_up_w": gate_up_w,
        "gate_up_s": scales,
        "down_w": down_w,
        "down_s": scales[_SCALE_ROWS:, :],
        "scales": scales,
        "down_scale_offset": _SCALE_ROWS,
    }
    if row_tiled_tail:
        cols = max(column for _, column in tail_halves()) + _DOWN_COLS
        weights["tail_s"] = nl.ndarray((_PMAX, cols), dtype=nl.uint8, buffer=nl.sbuf, name=f"{name}_tail_s")
    return weights


def _check_stacks(i_slice: int, gate_up, gate_up_scale, down, down_scale) -> tuple:
    """Check the four HBM expert stacks (``tile_weight_shapes``); return them with the FP4 stacks as uint16."""
    stacks = (gate_up, gate_up_scale, down, down_scale)
    shapes = tile_weight_shapes(i_slice, gate_up.shape[0])
    for stack, shape in zip(stacks, shapes.values()):
        kernel_assert(
            tuple(stack.shape) == shape,
            f"Expected the expert stack {shape} for I-slice {i_slice}, got {tuple(stack.shape)}",
        )
    return tuple(stack.view(nl.uint16) if stack.dtype == nl.float4_e2m1fn_x4 else stack for stack in stacks)


def _expert_source(stack: nl.NkiTensor, expert, pattern: list, offset: int) -> nl.NkiTensor:
    """The access pattern of expert ``expert`` of an expert-major HBM stack: a trace-time offset for an int, else a
    runtime offset along the expert axis (``scalar_offset``)."""
    if isinstance(expert, int):
        per_expert = 1
        for size in stack.shape[1:]:
            per_expert *= size
        return stack.ap(pattern=pattern, offset=expert * per_expert + offset)
    return stack.ap(pattern=pattern, offset=offset, scalar_offset=expert, indirect_dim=0)


LOADS_SKIP_OOB = True
"""The expert loads with a runtime expert id skip their DMAs when the id lies past the stack (``oob_mode`` skip):
the buffers keep the expert loaded before, so a skip id (``_stack_skip_id`` of the served loop: E_L for a stack of
its own) reruns it, and a padding slot loads nothing. A trace-time id in range takes static descriptors,
with no skip (``_dma``)."""


def _dma(dst: nl.NkiTensor, src: nl.NkiTensor, engine, name: str, expert=0):
    """One load of the expert loads. A trace-time expert id (an int) takes static descriptors (DGE mode none, as the
    served static loop's loads), so it holds no DGE table entry. A runtime id takes the hardware DGE on ``engine`` and
    skips an out-of-bounds read (``LOADS_SKIP_OOB``)."""
    if isinstance(expert, int):
        return nisa.dma_copy(dst=dst, src=src, dge_mode=nisa.dge_mode.none, name=name)
    return nisa.dma_copy(
        dst=dst, src=src, dge_mode=nisa.dge_mode.hwdge, engine=engine, oob_mode=nisa.oob_mode.skip, name=name
    )


def load_tile_weights(
    weights: dict,
    expert_gate_up_weights: nl.NkiTensor,
    expert_gate_up_weights_scale: nl.NkiTensor,
    expert_down_weights: nl.NkiTensor,
    expert_down_weights_scale: nl.NkiTensor,
    expert,
    i_slice: int,
    name: str,
    engine=nisa.engine.sync,
) -> list:
    """Load expert ``expert`` of the HBM stacks (``tile_weight_shapes``) into ``weights`` (``alloc_tile_weights``).

    ``expert``: a trace-time int, or an INT32 [1, 1] SBUF tensor (or a register) that the DMAs take as a runtime
    offset along the expert axis, so a dynamic loop can load only its active experts. A runtime id past the stack
    skips every DMA (``LOADS_SKIP_OOB``): the buffers keep the expert loaded before. A trace-time id loads with static
    descriptors, a runtime id with the hardware DGE on ``engine``. One DMA for the gate/up weights,
    4 for their scales, and per down tile one for the weights and one for each scale row of a quadrant (at most 4).
    Returns the DMA instructions.
    """
    plan = tile_plan(i_slice)
    cols = N_H512 * 2 * i_slice
    # The down scale rows go to absolute partitions of the scale buffer (no strided slice of a view).
    down_scales, down_offset = weights.get("scales", weights["down_s"]), weights.get("down_scale_offset", 0)
    gate_up, gate_up_scale, down, down_scale = _check_stacks(
        i_slice, expert_gate_up_weights, expert_gate_up_weights_scale, expert_down_weights, expert_down_weights_scale
    )
    loads = [
        _dma(
            weights["gate_up_w"][:, 0:cols],
            _expert_source(gate_up, expert, [[cols, _PMAX], [1, cols]], 0),
            engine,
            f"{name}_load_gate_up",
            expert=expert,
        )
    ]
    # E8M0 row 4q + r goes to partition 32q + r: one DMA for each r over the 4 quadrants.
    for row in range(_SCALE_ROWS):
        loads.append(
            _dma(
                weights["gate_up_s"][row : row + 3 * _QUADRANT + 1 : _QUADRANT, 0:cols],
                _expert_source(
                    gate_up_scale, expert, [[_SCALE_ROWS * cols, _PMAX // _QUADRANT], [1, cols]], row * cols
                ),
                engine,
                f"{name}_load_gate_up_s{row}",
                expert=expert,
            )
        )
    for index, tile in enumerate(plan.down_tiles):
        if "tail_s" in weights and index == len(plan.down_tiles) - 1:
            loads.extend(_load_row_tiled_tail(weights, down, down_scale, expert, tile, index, engine, name))
            continue
        tile_cols = slice(index * LATENT_SIZE, (index + 1) * LATENT_SIZE)
        loads.append(
            _dma(
                weights["down_w"][0 : tile.rows, tile_cols],
                _expert_source(
                    down, expert, [[LATENT_SIZE, tile.rows], [1, LATENT_SIZE]], tile.first // _Q * LATENT_SIZE
                ),
                engine,
                f"{name}_load_down{index}",
                expert=expert,
            )
        )
        groups, first_group = tile.rows // _BLOCK_ROWS, tile.first // MX_BLOCK
        for row in range(min(_SCALE_ROWS, groups)):
            quadrants = len(range(row, groups, _SCALE_ROWS))
            first = down_offset + row
            loads.append(
                _dma(
                    down_scales[first : first + (quadrants - 1) * _QUADRANT + 1 : _QUADRANT, tile_cols],
                    _expert_source(
                        down_scale,
                        expert,
                        [[_SCALE_ROWS * LATENT_SIZE, quadrants], [1, LATENT_SIZE]],
                        (first_group + row) * LATENT_SIZE,
                    ),
                    engine,
                    f"{name}_load_down_s{index}_{row}",
                    expert=expert,
                )
            )
    return loads


def _load_row_tiled_tail(weights: dict, down, down_scale, expert, tile: DownTile, index: int, engine, name: str):
    """The loads of the row-tiled tail (``alloc_tile_weights`` ``row_tiled_tail``): per half and run of consecutive
    chunks, one DMA for the weights and one for the scales of each quadrant (2 quadrants a half): 12 DMAs."""
    halves, loads = tail_halves(), []
    groups, first_group = tile.rows // _BLOCK_ROWS, tile.first // MX_BLOCK
    for half in range(2):
        runs = []  # [first hidden column, first SBUF column, columns]
        for chunk, (chunk_half, column) in enumerate(halves):
            if chunk_half != half:
                continue
            if runs and runs[-1][0] + runs[-1][2] == chunk * _DOWN_COLS and runs[-1][1] + runs[-1][2] == column:
                runs[-1][2] += _DOWN_COLS
            else:
                runs.append([chunk * _DOWN_COLS, column, _DOWN_COLS])
        partition = half * _TAIL_ROWS
        for hidden, column, cols in runs:
            sbuf = slice(index * LATENT_SIZE + column, index * LATENT_SIZE + column + cols)
            loads.append(
                _dma(
                    weights["down_w"][partition : partition + tile.rows, sbuf],
                    _expert_source(
                        down, expert, [[LATENT_SIZE, tile.rows], [1, cols]], tile.first // _Q * LATENT_SIZE + hidden
                    ),
                    engine,
                    f"{name}_load_tail{half}_h{hidden}",
                    expert=expert,
                )
            )
            # Groups 4q .. 4q + 3 of the tile go to 4 neighboring partitions of quadrant q of the half.
            for quadrant in range(div_ceil(groups, _SCALE_ROWS)):
                first = partition + quadrant * _QUADRANT
                count = min(_SCALE_ROWS, groups - quadrant * _SCALE_ROWS)
                loads.append(
                    _dma(
                        weights["tail_s"][first : first + count, column : column + cols],
                        _expert_source(
                            down_scale,
                            expert,
                            [[LATENT_SIZE, count], [1, cols]],
                            (first_group + quadrant * _SCALE_ROWS) * LATENT_SIZE + hidden,
                        ),
                        engine,
                        f"{name}_load_tail_s{half}_h{hidden}_q{quadrant}",
                        expert=expert,
                    )
                )
    return loads


def _instruction(handle):
    """Return an ISA call result as an NkiInstruction, for ``depends_on``."""
    return handle if isinstance(handle, nisa.NkiInstruction) else nisa.NkiInstruction(handle)


def _after(later, earlier):
    """Keep ``later`` after ``earlier`` on TensorE: a schedule edge, no semaphore wait."""
    _instruction(later).depends_on(_instruction(earlier), kind=nisa.dependency.schedule)


def _check_tile_inputs(x_mx: nl.NkiTensor, x_scale: nl.NkiTensor, n_tok: int):
    kernel_assert(
        isinstance(n_tok, int) and 4 <= n_tok <= MAX_TILE_TOKENS and n_tok % _Q == 0,
        f"n_tok must be a trace-time multiple of {_Q} in [4, {MAX_TILE_TOKENS}] (an MX matmul of a runtime size hangs "
        f"the PE), got {n_tok}",
    )
    kernel_assert(
        len(x_mx.shape) == 3
        and x_mx.shape[0] == _PMAX
        and x_mx.shape[1] == N_H512
        and x_mx.shape[2] >= n_tok
        and x_mx.dtype == nl.float8_e4m3fn_x4
        and len(x_scale.shape) == 3
        and tuple(x_scale.shape[:2]) == (_PMAX, N_H512)
        and x_scale.shape[2] >= n_tok
        and x_scale.dtype == nl.uint8,
        f"x_mx / x_scale must be the MXFP8 latent [128, {N_H512}, >= n_tok] float8_e4m3fn_x4 / uint8 SBUF tiles, got "
        f"{x_mx.dtype} {tuple(x_mx.shape)} and {x_scale.dtype} {tuple(x_scale.shape)}",
    )


def _situ_terms(gate_up_ps: nl.NkiTensor, width: int, n_tok: int, cfg: tuple, name: str) -> tuple:
    """Return (up tanh, gate term) FP32 [n_tok, width] of one gate/up chunk: the served SiTU op order up to its
    product. The product (``_situ_product``) is linear_beta * up tanh * gate term."""
    beta, linear_beta = cfg
    gate, up = gate_up_ps[:, 0:width], gate_up_ps[:, width : 2 * width]
    sigmoid = nl.ndarray((n_tok, width), dtype=nl.float32, buffer=nl.sbuf, name=f"{name}_sigmoid_t")
    gate_term = nl.ndarray((n_tok, width), dtype=nl.float32, buffer=nl.sbuf, name=f"{name}_gate_term_t")
    up_term = nl.ndarray((n_tok, width), dtype=nl.float32, buffer=nl.sbuf, name=f"{name}_up_term_t")
    nisa.activation(dst=sigmoid, op=nl.sigmoid, data=gate, engine=nisa.engine.scalar, name=f"{name}_sigmoid")
    nisa.activation(
        dst=gate_term, op=nl.tanh, data=gate, scale=1.0 / beta, engine=nisa.engine.scalar, name=f"{name}_tanh_gate"
    )
    nisa.scalar_tensor_tensor(
        dst=gate_term,
        data=gate_term,
        op0=nl.multiply,
        operand0=beta,
        op1=nl.multiply,
        operand1=sigmoid,
        name=f"{name}_gate_term",
    )
    nisa.activation(
        dst=up_term, op=nl.tanh, data=up, scale=1.0 / linear_beta, engine=nisa.engine.scalar, name=f"{name}_tanh_up"
    )
    return up_term, gate_term


def _situ_product(dst: nl.NkiTensor, up_term: nl.NkiTensor, gate_term: nl.NkiTensor, scale, name: str):
    """dst (FP16) = (up_term * scale) * gate_term: ``scale`` is linear_beta, or an FP32 [n_tok, 1] column of
    linear_beta * row mask (exactly linear_beta or 0 a row)."""
    nisa.scalar_tensor_tensor(
        dst=dst,
        data=up_term,
        op0=nl.multiply,
        operand0=scale,
        op1=nl.multiply,
        operand1=gate_term,
        name=name,
    )


def _gate_up_chunk(x_mx, x_scale, weights, scales, col: int, width: int, n_tok: int, after, name: str) -> tuple:
    """Return (FP32 PSUM [n_tok, 2 width], last matmul) of one gate/up chunk: its 7 MX matmuls over the H tiles, with
    the weights of H tile hb at columns hb * row + col of ``weights`` (row = the gate/up columns of one H tile)."""
    gate_up_ps = nl.ndarray((n_tok, 2 * width), dtype=nl.float32, buffer=nl.psum, name=f"{name}_ps")
    row = weights.shape[1] // N_H512
    last = None
    for tile_h in range(N_H512):
        first_col = tile_h * row + col
        matmul = nisa.nc_matmul_mx(
            dst=gate_up_ps,
            stationary=x_mx[:, tile_h, 0:n_tok],
            moving=weights[:, first_col : first_col + 2 * width],
            stationary_scale=x_scale[:, tile_h, 0:n_tok],
            moving_scale=scales[:, first_col : first_col + 2 * width],
            accumulate=tile_h > 0,
            name=f"{name}_mm_h{tile_h}",
        )
        if last is None and after is not None:
            _after(matmul, after)
        last = matmul
    return gate_up_ps, last


def _activation_tile(activation: nl.NkiTensor, first: int, stride: int, rows: int, partitions: int, n_tok: int, name):
    """Return (codes, scales, transposes): the MXFP8 down operand [partitions, n_tok] whose lane q, partition p
    (p < rows) is activation column first + q * stride + p, zero on partitions rows .. partitions - 1."""
    transposed = nl.ndarray((rows, _Q, n_tok), dtype=nl.float16, buffer=nl.psum, name=f"{name}_t")
    transposes = []
    for lane in range(_Q):
        col = first + lane * stride
        transposes.append(
            nisa.nc_transpose(
                dst=transposed[:, lane, :],
                data=activation[:, col : col + rows],
                engine=nisa.engine.tensor,
                name=f"{name}_transpose_q{lane}",
            )
        )
    # [partitions, token, lane] FP16, the quantize input; the zero rows give zero codes on their own groups.
    lanes = nl.ndarray((partitions, n_tok, _Q), dtype=nl.float16, buffer=nl.sbuf, name=f"{name}_f16")
    if partitions > rows:
        nisa.memset(dst=lanes, value=0.0, engine=nisa.vector_engine, name=f"{name}_zero")
    nisa.tensor_copy(
        dst=lanes[0:rows, :, :],
        src=transposed.ap(pattern=[[_Q * n_tok, rows], [1, n_tok], [n_tok, _Q]], offset=0),
        engine=nisa.scalar_engine,
        name=f"{name}_evict",
    )
    codes = nl.ndarray((partitions, n_tok), dtype=nl.float8_e4m3fn_x4, buffer=nl.sbuf, name=f"{name}_mx")
    scales = nl.ndarray((partitions, n_tok), dtype=nl.uint8, buffer=nl.sbuf, name=f"{name}_s")
    nisa.quantize_mx(src=lanes, dst=codes, dst_scale=scales, name=f"{name}_quantize")
    return codes, scales, transposes


def _check_gate_up(gate_up_w: nl.NkiTensor, gate_up_s: nl.NkiTensor, cols: int, scale_cols: int):
    kernel_assert(
        tuple(gate_up_w.shape) == (_PMAX, cols) and gate_up_w.dtype in (nl.uint16, nl.float4_e2m1fn_x4),
        f"gate_up_w must be [128, {cols}] uint16 or float4_e2m1fn_x4, got {gate_up_w.dtype} {tuple(gate_up_w.shape)}",
    )
    kernel_assert(
        gate_up_s.dtype == nl.uint8 and gate_up_s.shape[1] >= scale_cols,
        f"gate_up_s must be uint8 with >= {scale_cols} columns, got {gate_up_s.dtype} {tuple(gate_up_s.shape)}",
    )


def _fp4(weights: nl.NkiTensor) -> nl.NkiTensor:
    return weights.view(nl.float4_e2m1fn_x4) if weights.dtype == nl.uint16 else weights


def expert_tile_gate_up(
    x_mx: nl.NkiTensor,
    x_scale: nl.NkiTensor,
    gate_up_w: nl.NkiTensor,
    gate_up_s: nl.NkiTensor,
    n_tok: int,
    i_slice: int,
    name: str,
    after=None,
    beta: float = SITU_BETA,
    linear_beta: float = SITU_LINEAR_BETA,
) -> tuple:
    """Run the gate/up matmuls and the SiTU of one expert on ``n_tok`` tokens (the first stage of
    ``expert_tile_front``). Returns (the FP16 token-major activation [n_tok, I_s] in kernel I order, the last gate/up
    matmul). ``after``: a TensorE instruction that the first matmul follows, or None.

    With the stages split, a loop can issue gate/up(e + 1), then down(e), then the activation of e + 1
    (``expert_tile_activation``), so TensorE runs the down of e while the SiTU of e + 1 runs on ScalarE / VectorE."""
    plan = tile_plan(i_slice)
    _check_tile_inputs(x_mx, x_scale, n_tok)
    _check_gate_up(gate_up_w, gate_up_s, N_H512 * 2 * i_slice, N_H512 * 2 * i_slice)
    weights = _fp4(gate_up_w)
    activation = nl.ndarray((n_tok, i_slice), dtype=nl.float16, buffer=nl.sbuf, name=f"{name}_activation")
    last = None
    for chunk, (first, width) in enumerate(plan.chunks):
        gate_up_ps, last = _gate_up_chunk(
            x_mx,
            x_scale,
            weights,
            gate_up_s,
            2 * first,
            width,
            n_tok,
            after if chunk == 0 else None,
            f"{name}_gate_up{chunk}",
        )
        up_term, gate_term = _situ_terms(gate_up_ps, width, n_tok, (beta, linear_beta), f"{name}_situ{chunk}")
        _situ_product(
            activation[:, first : first + width], up_term, gate_term, linear_beta, f"{name}_situ{chunk}_product"
        )
    return activation, last


def expert_tile_activation(
    activation: nl.NkiTensor, n_tok: int, i_slice: int, name: str, after=None, row_tiled_tail: bool = False
) -> tuple:
    """Transpose and quantize an ``expert_tile_gate_up`` activation to the down operands (the second stage of
    ``expert_tile_front``). Returns (activation tiles, last transpose). ``after``: a TensorE instruction that the
    first transpose follows (for example the last down matmul of the expert before), or None."""
    plan = tile_plan(i_slice)
    kernel_assert(
        not row_tiled_tail or i_slice in ROW_TILED_TAIL_WIDTHS,
        f"row_tiled_tail takes an I-slice in {ROW_TILED_TAIL_WIDTHS}, got {i_slice}",
    )
    kernel_assert(
        tuple(activation.shape) == (n_tok, i_slice) and activation.dtype == nl.float16,
        f"activation must be FP16 [{n_tok}, {i_slice}], got {activation.dtype} {tuple(activation.shape)}",
    )
    tiles, last = [], None
    for index, tile in enumerate(plan.down_tiles):
        if row_tiled_tail and index == len(plan.down_tiles) - 1:
            codes, scales, transposes = _row_tiled_tail_activation(activation, tile, n_tok, f"{name}_act{index}")
        else:
            codes, scales, transposes = _activation_tile(
                activation, tile.first, tile.rows, tile.rows, tile.partitions, n_tok, f"{name}_act{index}"
            )
        if last is None and after is not None:
            _after(transposes[0], after)
        last = transposes[-1]
        tiles.append((codes, scales))
    return tuple(tiles), last


def expert_tile_front(
    x_mx: nl.NkiTensor,
    x_scale: nl.NkiTensor,
    gate_up_w: nl.NkiTensor,
    gate_up_s: nl.NkiTensor,
    n_tok: int,
    i_slice: int,
    name: str,
    after=None,
    beta: float = SITU_BETA,
    linear_beta: float = SITU_LINEAR_BETA,
    row_tiled_tail: bool = False,
) -> tuple:
    """Run the gate/up matmuls, the SiTU, the transposes and the MXFP8 quantize of one expert on ``n_tok`` tokens
    (``expert_tile_gate_up``, then ``expert_tile_activation``).

    Returns (activation tiles, last TensorE instruction). Activation tile t is (codes float8_e4m3fn_x4
    [partitions, n_tok], E8M0 scales uint8 [partitions, n_tok]) of down tile t of ``tile_plan(i_slice)``: partition p,
    lane q of token j is slice row first + 4p + q of the expert's SiTU output for token j (zero past ``rows``).
    ``after``: a TensorE instruction that the first gate/up matmul follows (a schedule edge), or None.
    ``row_tiled_tail`` (I-slice 768): the tail tile is [128, n_tok] with the 64 rows on partitions 0 .. 63 and again
    on 64 .. 127 (the stationary of both 64-row PE tiles).
    """
    activation, _ = expert_tile_gate_up(
        x_mx, x_scale, gate_up_w, gate_up_s, n_tok, i_slice, name, after=after, beta=beta, linear_beta=linear_beta
    )
    return expert_tile_activation(activation, n_tok, i_slice, name, row_tiled_tail=row_tiled_tail)


def _row_tiled_tail_activation(activation: nl.NkiTensor, tile: DownTile, n_tok: int, name: str) -> tuple:
    """The tail operand of the row-tiled tail: each lane's 64 tail columns twice side by side (two FP16 copies), so
    one transpose a lane puts the rows on partitions 0 .. 63 and 64 .. 127."""
    doubled = nl.ndarray((n_tok, _Q * 2 * tile.rows), dtype=nl.float16, buffer=nl.sbuf, name=f"{name}_doubled")
    for copy, engine in enumerate((nisa.vector_engine, nisa.scalar_engine)):
        nisa.tensor_copy(
            dst=doubled.ap(
                pattern=[[_Q * 2 * tile.rows, n_tok], [2 * tile.rows, _Q], [1, tile.rows]], offset=copy * tile.rows
            ),
            src=activation.ap(
                pattern=[[activation.shape[1], n_tok], [tile.rows, _Q], [1, tile.rows]], offset=tile.first
            ),
            engine=engine,
            name=f"{name}_double{copy}",
        )
    return _activation_tile(doubled, 0, 2 * tile.rows, 2 * tile.rows, 2 * tile.rows, n_tok, name)


def _down(
    acc_fp32, tiles: tuple, weights, down_s, n_tok: int, name: str, after, affinity, accumulate, consume, tail_s=None
):
    """The down matmuls of the activation tiles (tile t: moving columns t * H .. of ``weights``) and the eviction of
    each 512-column hidden chunk. With ``tail_s`` the last tile is the row-tiled tail (``tail_halves``): its chunks
    run on the 64-row PE tiles in ``_tail_order``. Returns the last matmul."""
    halves = tail_halves()
    if tail_s is not None:
        tiles, tail = tiles[:-1], (len(tiles) - 1, *tiles[-1])
    n_chunks = LATENT_SIZE // _DOWN_COLS
    last = after
    for group_first in range(0, n_chunks, _DOWN_GROUP):
        group = range(group_first, min(group_first + _DOWN_GROUP, n_chunks))
        out_ps = {
            chunk: nl.ndarray((n_tok, _DOWN_COLS), dtype=nl.float32, buffer=nl.psum, name=f"{name}_down_ps{chunk}")
            for chunk in group
        }
        for index, (codes, scales) in enumerate(tiles):
            partitions = codes.shape[0]
            for chunk in group:
                col = index * LATENT_SIZE + chunk * _DOWN_COLS
                matmul = nisa.nc_matmul_mx(
                    dst=out_ps[chunk],
                    stationary=codes[:, 0:n_tok],
                    moving=weights[0:partitions, col : col + _DOWN_COLS],
                    stationary_scale=scales[:, 0:n_tok],
                    moving_scale=down_s[:, col : col + _DOWN_COLS],
                    accumulate=index > 0,
                    name=f"{name}_mm_down{index}_h{chunk}",
                )
                if last is after and after is not None:
                    _after(matmul, after)
                last = matmul
        if tail_s is not None:
            index, codes, scales = tail
            for chunk in _tail_order(group):
                half, column = halves[chunk]
                rows = slice(half * _TAIL_ROWS, (half + 1) * _TAIL_ROWS)
                col = index * LATENT_SIZE + column
                last = nisa.nc_matmul_mx(
                    dst=out_ps[chunk],
                    stationary=codes[rows, 0:n_tok],
                    moving=weights[rows, col : col + _DOWN_COLS],
                    stationary_scale=scales[rows, 0:n_tok],
                    moving_scale=tail_s[rows, column : column + _DOWN_COLS],
                    tile_position=(half * _TAIL_ROWS, 0),
                    tile_size=(_TAIL_ROWS, _PMAX),
                    accumulate=index > 0,
                    name=f"{name}_mm_tail{half}_h{chunk}",
                )
        for chunk in group:
            _consume_chunk(acc_fp32, out_ps[chunk], chunk, n_tok, affinity, accumulate, consume, name)
    return last


def _check_down_inputs(acc_fp32, down_w, down_s, cols: int, n_tok: int, affinity, consume):
    kernel_assert(
        tuple(down_w.shape) == (_PMAX, cols) and down_w.dtype in (nl.uint16, nl.float4_e2m1fn_x4),
        f"down_w must be [128, {cols}] uint16 or float4_e2m1fn_x4, got {down_w.dtype} {tuple(down_w.shape)}",
    )
    kernel_assert(
        down_s.dtype == nl.uint8 and down_s.shape[1] >= cols,
        f"down_s must be uint8 with >= {cols} columns, got {down_s.dtype} {tuple(down_s.shape)}",
    )
    if consume is None:
        kernel_assert(
            acc_fp32 is not None
            and acc_fp32.dtype == nl.float32
            and acc_fp32.shape[0] >= n_tok
            and acc_fp32.shape[1] == LATENT_SIZE,
            f"acc_fp32 must be an FP32 SBUF tile [>= {n_tok}, {LATENT_SIZE}], got "
            f"{None if acc_fp32 is None else (acc_fp32.dtype, tuple(acc_fp32.shape))}",
        )
    if affinity is not None:
        kernel_assert(
            affinity.dtype == nl.float32 and affinity.shape[0] >= n_tok,
            f"affinity must be FP32 [>= {n_tok}, 1], got {affinity.dtype} {tuple(affinity.shape)}",
        )


def expert_tile_down(
    acc_fp32: nl.NkiTensor,
    activation_tiles: tuple,
    down_w: nl.NkiTensor,
    down_s: nl.NkiTensor,
    n_tok: int,
    i_slice: int,
    name: str,
    after=None,
    affinity: nl.NkiTensor = None,
    accumulate: bool = True,
    consume=None,
    tail_s: nl.NkiTensor = None,
):
    """Run the down matmuls of one expert on its ``expert_tile_front`` activation and add the result into acc_fp32.

    Each 512-column hidden chunk accumulates the down tiles in FP32 PSUM in ascending order, then one VectorE
    instruction reads the PSUM chunk: acc[j] = a[j] * y[j] + acc[j] (``accumulate``) or a[j] * y[j], with a = 1 when
    ``affinity`` is None. ``consume(first_col, cols, psum)``, when given, takes each FP32 PSUM chunk [n_tok, 512] in
    place of that (acc_fp32 may then be None). ``tail_s``: the tail scales of a row-tiled tail (``alloc_tile_weights``
    ``row_tiled_tail``, with the activation of ``expert_tile_front`` ``row_tiled_tail``), or None. Returns the last
    down matmul.
    """
    plan = tile_plan(i_slice)
    kernel_assert(
        tail_s is None or i_slice in ROW_TILED_TAIL_WIDTHS,
        f"a row-tiled tail takes an I-slice in {ROW_TILED_TAIL_WIDTHS}, got {i_slice}",
    )
    kernel_assert(
        len(activation_tiles) == len(plan.down_tiles),
        f"Expected {len(plan.down_tiles)} activation tiles for I-slice {i_slice}, got {len(activation_tiles)}",
    )
    _check_down_inputs(acc_fp32, down_w, down_s, len(plan.down_tiles) * LATENT_SIZE, n_tok, affinity, consume)
    return _down(
        acc_fp32, activation_tiles, _fp4(down_w), down_s, n_tok, name, after, affinity, accumulate, consume, tail_s
    )


def _consume_chunk(acc_fp32, out_ps, chunk: int, n_tok: int, affinity, accumulate: bool, consume, name: str):
    """Add one FP32 down PSUM chunk into the accumulator (or hand it to ``consume``)."""
    first_col = chunk * _DOWN_COLS
    if consume is not None:
        consume(first_col, _DOWN_COLS, out_ps)
        return
    dst = acc_fp32[0:n_tok, first_col : first_col + _DOWN_COLS]
    label = f"{name}_acc_h{chunk}"
    if affinity is None and accumulate:
        nisa.tensor_tensor(dst=dst, data1=out_ps, data2=dst, op=nl.add, engine=nisa.vector_engine, name=label)
    elif affinity is None:
        nisa.tensor_copy(dst=dst, src=out_ps, engine=nisa.vector_engine, name=label)
    elif accumulate:
        nisa.scalar_tensor_tensor(
            dst=dst,
            data=out_ps,
            op0=nl.multiply,
            operand0=affinity[0:n_tok, 0:1],
            op1=nl.add,
            operand1=dst,
            name=label,
        )
    else:
        nisa.tensor_scalar(
            dst=dst,
            data=out_ps,
            op0=nl.multiply,
            operand0=affinity[0:n_tok, 0:1],
            engine=nisa.vector_engine,
            name=label,
        )


def routed_expert_tile(
    acc_fp32: nl.NkiTensor,
    x_mx: nl.NkiTensor,
    x_scale: nl.NkiTensor,
    gate_up_w: nl.NkiTensor,
    gate_up_s: nl.NkiTensor,
    down_w: nl.NkiTensor,
    down_s: nl.NkiTensor,
    n_tok: int,
    i_slice: int,
    name: str,
    after=None,
    affinity: nl.NkiTensor = None,
    accumulate: bool = True,
    consume=None,
    beta: float = SITU_BETA,
    linear_beta: float = SITU_LINEAR_BETA,
    tail_s: nl.NkiTensor = None,
):
    """Run one local expert (gate/up -> SiTU -> down) on a compact tile of ``n_tok`` tokens; add it into acc_fp32.

    The numerics of the served ``moe_tkg_decode``: MX matmuls with FP32 PSUM accumulation, the two-sided SiTU in FP32,
    the SiTU output rounded to FP16, then MXFP8 on the checkpoint's 32-element down groups, the down accumulated in
    FP32 PSUM over the down tiles in ascending order, the affinity applied after the down in FP32. A fixed order for
    a fixed input, so same-batch runs are bitwise.

    Args:
        acc_fp32: FP32 SBUF [>= n_tok, 3584], the caller's accumulator in compact token order (row j = token j of the
            tile). acc[j] = a[j] * y[j] + acc[j], or a[j] * y[j] without ``accumulate``. None with ``consume``.
        x_mx, x_scale: the MXFP8 latent of the gathered tokens: float8_e4m3fn_x4 / uint8 SBUF [128, 7, >= n_tok] (the
            ``quantized_latent`` layout of ``moe_tkg_decode``, e.g. from its ``_quantize_natural_latent``).
        gate_up_w, gate_up_s, down_w, down_s: the expert's SBUF weights (``alloc_tile_weights`` /
            ``load_tile_weights``). The scale views may start at an MX scale offset (0, 4, 8 or 12) of a quadrant.
        n_tok: trace-time tile rows, a multiple of 4 in [4, 128]. Rows past the live count must hold finite MXFP8
            values (e.g. zero codes and scales) and an affinity of 0; their outputs add 0.
        i_slice: the intermediate slice, one of ``TILE_WIDTHS``.
        name: a prefix for the instruction names, unique for each call.
        after: a TensorE instruction that the first matmul follows (a schedule edge), or None.
        affinity: FP32 SBUF [>= n_tok, 1], the gate weight of each token; None for 1. Same base partition as acc_fp32.
        accumulate: add into acc_fp32 (True, default) or overwrite it (False, the first expert of a token tile).
        consume: None, or a function (first_col, cols, psum) that takes each FP32 PSUM down chunk [n_tok, 512].
        beta, linear_beta: the SiTU gate and up soft-clipping factors.
        tail_s: None (default), or the ``tail_s`` of ``alloc_tile_weights(768, ..., row_tiled_tail=True)``: the 64-row
            tail tile runs on the 64-row PE tiles from 128-partition loads (opt-in, I-slice 768; the same values).

    Returns:
        The last down matmul (for the next call's ``after``).
    """
    activation_tiles, front_last = expert_tile_front(
        x_mx,
        x_scale,
        gate_up_w,
        gate_up_s,
        n_tok,
        i_slice,
        name,
        after=after,
        beta=beta,
        linear_beta=linear_beta,
        row_tiled_tail=tail_s is not None,
    )
    return expert_tile_down(
        acc_fp32,
        activation_tiles,
        down_w,
        down_s,
        n_tok,
        i_slice,
        name,
        after=front_last,
        affinity=affinity,
        accumulate=accumulate,
        consume=consume,
        tail_s=tail_s,
    )


# ---- Packed tile: several experts of a narrow I-slice in one down contraction ----

PACKED_WIDTHS = (384, 192, 96)
"""The I-slices of ``routed_expert_tile_packed`` (one down tile each)."""
MAX_PACKED_ROWS = 4 * _PMAX
"""Most down rows of one packed call (K experts x I_s / 4): at most 4 groups of 128."""


class PackedPlan(NamedTuple):
    """The down groups of ``experts`` packed experts of one I-slice (``packed_plan``)."""

    i_slice: int
    experts: int
    rows: int
    """Down rows of one expert: I_s / 4. Expert k takes packed rows k * rows .. (k + 1) * rows - 1."""
    groups: tuple
    """(first packed row, data rows, matmul partitions) of each group of 128 packed rows."""


def packed_plan(i_slice: int, experts: int) -> PackedPlan:
    """Return the down groups of ``experts`` experts of I-slice ``i_slice`` (one of ``PACKED_WIDTHS``).

    Example:
        >>> packed_plan(96, 16).groups
        ((0, 128, 128), (128, 128, 128), (256, 128, 128))
    """
    kernel_assert(
        i_slice in PACKED_WIDTHS, f"The packed tile takes an intermediate slice in {PACKED_WIDTHS}, got {i_slice}"
    )
    rows = i_slice // _Q
    kernel_assert(
        isinstance(experts, int) and 1 <= experts and experts * rows <= MAX_PACKED_ROWS,
        f"The packed tile takes 1 to {MAX_PACKED_ROWS // rows} experts of I-slice {i_slice}, got {experts}",
    )
    total = experts * rows
    groups = tuple(
        (first, min(_PMAX, total - first), div_ceil(min(_PMAX, total - first), _QUADRANT) * _QUADRANT)
        for first in range(0, total, _PMAX)
    )
    return PackedPlan(i_slice, experts, rows, groups)


def packed_matmul_counts(i_slice: int, experts: int) -> dict:
    """Return the MX matmuls and the PE transposes of one ``routed_expert_tile_packed`` call.

    Example:
        >>> packed_matmul_counts(96, 16)
        {'gate_up': 112, 'down': 21, 'transposes': 12}
    """
    plan = packed_plan(i_slice, experts)
    return {
        "gate_up": experts * N_H512 * len(tile_plan(i_slice).chunks),
        "down": (LATENT_SIZE // _DOWN_COLS) * len(plan.groups),
        "transposes": _Q * len(plan.groups),
    }


def alloc_packed_weights(
    i_slice: int, experts: int, name: str, zero_scales: bool = False, zero_engine=nisa.engine.gpsimd
) -> dict:
    """Allocate the SBUF weights of ``experts`` packed experts and zero the padding rows of the last down group.

    Returns a dict of 2-D SBUF views and the expert count:
        - gate_up_w: uint16 [128, K * G] (G = 7 * 2 I_s), expert k in columns k * G .. (k + 1) * G.
        - gate_up_s: uint8 [128, ceil(K / 4) * G], expert k at quadrant scale offset 4 (k % 4), columns (k // 4) * G.
        - down_w: uint16 [128, n_groups * H], packed row r on partition r % 128 of group r // 128.
        - down_s: uint8 [128, n_groups * H], packed group b (rows 8b .. 8b + 7) at partition 32 (j // 4) + j % 4 of
          group b // 16 (j = b % 16).
        - experts: K.
    Each call reads every slot: load all K slots before each call (an unused slot may repeat any expert, with an
    all-zero mask column), or set ``zero_scales`` to zero both scale buffers once here (about 16 KB a partition at
    I_s 96, K 16), so a slot that holds no loaded expert reads finite scales and adds exact zeros.
    """
    plan = packed_plan(i_slice, experts)
    cols = N_H512 * 2 * i_slice
    down_cols = len(plan.groups) * LATENT_SIZE
    gate_up_w = nl.ndarray((_PMAX, experts * cols), dtype=nl.uint16, buffer=nl.sbuf, name=f"{name}_gate_up_w")
    gate_up_s = nl.ndarray(
        (_PMAX, div_ceil(experts, _SCALE_ROWS) * cols), dtype=nl.uint8, buffer=nl.sbuf, name=f"{name}_gate_up_s"
    )
    down_w = nl.ndarray((_PMAX, down_cols), dtype=nl.uint16, buffer=nl.sbuf, name=f"{name}_down_w")
    down_s = nl.ndarray((_PMAX, down_cols), dtype=nl.uint8, buffer=nl.sbuf, name=f"{name}_down_s")
    if zero_scales:
        nisa.memset(dst=gate_up_s, value=0, engine=zero_engine, name=f"{name}_zero_gate_up_s")
        nisa.memset(dst=down_s, value=0, engine=zero_engine, name=f"{name}_zero_all_down_s")
    group = len(plan.groups) - 1
    _, rows, partitions = plan.groups[group]
    if partitions > rows:
        last = slice(group * LATENT_SIZE, (group + 1) * LATENT_SIZE)
        nisa.memset(dst=down_w[0:partitions, last], value=0, engine=zero_engine, name=f"{name}_zero_down_w")
        nisa.memset(dst=down_s[0:partitions, last], value=0, engine=zero_engine, name=f"{name}_zero_down_s")
    return {"gate_up_w": gate_up_w, "gate_up_s": gate_up_s, "down_w": down_w, "down_s": down_s, "experts": experts}


def load_packed_weights(
    weights: dict,
    expert_gate_up_weights: nl.NkiTensor,
    expert_gate_up_weights_scale: nl.NkiTensor,
    expert_down_weights: nl.NkiTensor,
    expert_down_weights_scale: nl.NkiTensor,
    slot: int,
    expert,
    i_slice: int,
    name: str,
    engine=nisa.engine.sync,
) -> list:
    """Load expert ``expert`` of the HBM stacks (``tile_weight_shapes``) into packed slot ``slot`` of ``weights``
    (``alloc_packed_weights``). ``expert``: a trace-time int, or an INT32 [1, 1] SBUF tensor (a runtime offset).
    Returns the DMA instructions."""
    plan = packed_plan(i_slice, weights["experts"])
    kernel_assert(0 <= slot < plan.experts, f"slot {slot} is not one of the {plan.experts} packed slots")
    cols = N_H512 * 2 * i_slice
    gate_up, gate_up_scale, down, down_scale = _check_stacks(
        i_slice, expert_gate_up_weights, expert_gate_up_weights_scale, expert_down_weights, expert_down_weights_scale
    )
    loads = [
        _dma(
            weights["gate_up_w"][:, slot * cols : (slot + 1) * cols],
            _expert_source(gate_up, expert, [[cols, _PMAX], [1, cols]], 0),
            engine,
            f"{name}_load_gate_up",
            expert=expert,
        )
    ]
    offset, scale_cols = (
        _SCALE_ROWS * (slot % _SCALE_ROWS),
        slice((slot // _SCALE_ROWS) * cols, (slot // _SCALE_ROWS + 1) * cols),
    )
    for row in range(_SCALE_ROWS):
        first = offset + row
        loads.append(
            _dma(
                weights["gate_up_s"][first : first + 3 * _QUADRANT + 1 : _QUADRANT, scale_cols],
                _expert_source(
                    gate_up_scale, expert, [[_SCALE_ROWS * cols, _PMAX // _QUADRANT], [1, cols]], row * cols
                ),
                engine,
                f"{name}_load_gate_up_s{row}",
                expert=expert,
            )
        )
    lo, hi = slot * plan.rows, (slot + 1) * plan.rows
    for group in range(lo // _PMAX, (hi - 1) // _PMAX + 1):
        first, last = max(lo, group * _PMAX), min(hi, (group + 1) * _PMAX)
        cols_g = slice(group * LATENT_SIZE, (group + 1) * LATENT_SIZE)
        loads.append(
            _dma(
                weights["down_w"][first - group * _PMAX : last - group * _PMAX, cols_g],
                _expert_source(
                    down, expert, [[LATENT_SIZE, last - first], [1, LATENT_SIZE]], (first - lo) * LATENT_SIZE
                ),
                engine,
                f"{name}_load_down_g{group}",
                expert=expert,
            )
        )
        # Packed 8-row blocks of the expert in this group, one DMA for each run inside a quadrant.
        blocks = range(first // _BLOCK_ROWS, last // _BLOCK_ROWS)
        for quadrant in range(_SCALE_ROWS):
            run = [b for b in blocks if (b % _GROUP_BLOCKS) // _SCALE_ROWS == quadrant]
            if not run:
                continue
            partition = _QUADRANT * quadrant + run[0] % _SCALE_ROWS
            loads.append(
                _dma(
                    weights["down_s"][partition : partition + len(run), cols_g],
                    _expert_source(
                        down_scale,
                        expert,
                        [[LATENT_SIZE, len(run)], [1, LATENT_SIZE]],
                        (run[0] - lo // _BLOCK_ROWS) * LATENT_SIZE,
                    ),
                    engine,
                    f"{name}_load_down_s_g{group}_q{quadrant}",
                    expert=expert,
                )
            )
    return loads


def routed_expert_tile_packed(
    acc_fp32: nl.NkiTensor,
    x_mx: nl.NkiTensor,
    x_scale: nl.NkiTensor,
    weights: dict,
    row_masks: nl.NkiTensor,
    n_tok: int,
    i_slice: int,
    name: str,
    after=None,
    affinity: nl.NkiTensor = None,
    accumulate: bool = True,
    consume=None,
    beta: float = SITU_BETA,
    linear_beta: float = SITU_LINEAR_BETA,
):
    """Run K packed experts of a narrow I-slice on one tile of ``n_tok`` pair rows; add the result into acc_fp32.

    Row j of the tile is one (token, expert) pair: it belongs to the packed expert k with ``row_masks[j, k]`` = 1 (at
    most one k a row; padding rows none). Each expert's gate/up runs over every row, and its SiTU product is multiplied
    by its row mask in the last SiTU op (operand linear_beta * mask: exactly the served op order for the kept rows, 0
    for the others). Expert k's activation fills packed down rows k * I_s / 4 .. (k + 1) * I_s / 4 - 1, so the down
    contraction is block-diagonal: output row j is the FP32 down output of its own expert, the value of
    ``routed_expert_tile`` for that pair (an expert across two 128-row groups adds its two partial sums in FP32 PSUM).
    Then the affinity of the pair in FP32, as in ``routed_expert_tile``. The cross-expert sum of a token is the
    caller's scatter of the pair rows.

    Args:
        acc_fp32, x_mx, x_scale, n_tok, name, after, affinity, accumulate, consume, beta, linear_beta: as in
            ``routed_expert_tile``, with pair rows in place of tokens.
        weights: ``alloc_packed_weights`` (K experts) loaded by ``load_packed_weights``.
        row_masks: FP32 SBUF [>= n_tok, >= K], 0 or 1.
        i_slice: one of ``PACKED_WIDTHS``.

    Returns:
        The last down matmul.
    """
    experts = weights["experts"]
    plan = packed_plan(i_slice, experts)
    rows = plan.rows
    _check_tile_inputs(x_mx, x_scale, n_tok)
    cols = N_H512 * 2 * i_slice
    _check_gate_up(weights["gate_up_w"], weights["gate_up_s"], experts * cols, div_ceil(experts, _SCALE_ROWS) * cols)
    _check_down_inputs(
        acc_fp32, weights["down_w"], weights["down_s"], len(plan.groups) * LATENT_SIZE, n_tok, affinity, consume
    )
    kernel_assert(
        row_masks.dtype == nl.float32 and row_masks.shape[0] >= n_tok and row_masks.shape[1] >= experts,
        f"row_masks must be FP32 [>= {n_tok}, >= {experts}], got {row_masks.dtype} {tuple(row_masks.shape)}",
    )
    total = experts * rows
    masked_scale = nl.ndarray((n_tok, experts), dtype=nl.float32, buffer=nl.sbuf, name=f"{name}_mask_scale")
    nisa.tensor_scalar(
        dst=masked_scale,
        data=row_masks[0:n_tok, 0:experts],
        op0=nl.multiply,
        operand0=linear_beta,
        engine=nisa.vector_engine,
        name=f"{name}_mask_scale_op",
    )
    # Lane-major packed activation: column q * total + r holds lane q of packed row r (expert r // rows, row r % rows).
    activation = nl.ndarray((n_tok, _Q * total), dtype=nl.float16, buffer=nl.sbuf, name=f"{name}_activation")
    gate_up = _fp4(weights["gate_up_w"])
    last = None
    for slot in range(experts):
        slot_weights = gate_up[:, slot * cols : (slot + 1) * cols]
        slot_scales = weights["gate_up_s"][
            _SCALE_ROWS * (slot % _SCALE_ROWS) :, (slot // _SCALE_ROWS) * cols : (slot // _SCALE_ROWS + 1) * cols
        ]
        for chunk, (first, width) in enumerate(tile_plan(i_slice).chunks):
            label = f"{name}_e{slot}_c{chunk}"
            gate_up_ps, last = _gate_up_chunk(
                x_mx,
                x_scale,
                slot_weights,
                slot_scales,
                2 * first,
                width,
                n_tok,
                after if last is None else None,
                label,
            )
            up_term, gate_term = _situ_terms(gate_up_ps, width, n_tok, (beta, linear_beta), label)
            for lane in range(first // rows, (first + width - 1) // rows + 1):
                lo, hi = max(first, lane * rows), min(first + width, (lane + 1) * rows)
                dst = lane * total + slot * rows + lo - lane * rows
                _situ_product(
                    activation[:, dst : dst + hi - lo],
                    up_term[:, lo - first : hi - first],
                    gate_term[:, lo - first : hi - first],
                    masked_scale[:, slot : slot + 1],
                    f"{label}_product_q{lane}",
                )
    tiles = []
    for group, (first, group_rows, partitions) in enumerate(plan.groups):
        codes, scales, transposes = _activation_tile(
            activation, first, total, group_rows, partitions, n_tok, f"{name}_act_g{group}"
        )
        last = transposes[-1]
        tiles.append((codes, scales))
    return _down(
        acc_fp32, tiles, _fp4(weights["down_w"]), weights["down_s"], n_tok, name, last, affinity, accumulate, consume
    )
