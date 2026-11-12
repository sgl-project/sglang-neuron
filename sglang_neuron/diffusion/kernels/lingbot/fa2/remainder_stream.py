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
"""Block streaming for a dimension whose extent is not a whole number of tiles."""

from typing import Any, Optional

import nki.language as nl

from nkilib.core.utils.kernel_assert import kernel_assert
from nkilib.core.utils.kernel_helpers import div_ceil
from nkilib.experimental import neurotile as nt


def block_table(extent: int, tile: int, tiles_per_blk: int) -> list[tuple[int, int]]:
    """block_table(extent, tile, tiles_per_blk) -> list

    Describe how ``nt.blocks`` groups ``extent`` into blocks.

    The trace-time answer to "what does each block actually contain", which a kernel needs in order to
    size its compute operands against a partial final tile. Each entry is
    ``(tile_count, last_tile_extent)``; ``last_tile_extent`` is the full tile granule for every block
    but the final one, where it is whatever the extent leaves over.

    .. warning::

       This API is experimental and may change in future releases.

    Args:
        extent (int): Number of elements along the dimension being grouped. Required.
        tile (int): The per-tile extent along that dimension. Required.
        tiles_per_blk (int): Tiles grouped into one block along that dimension. Required.

    Returns:
        list[tuple[int, int]]: One ``(tile_count, last_tile_extent)`` per block, in index order.
        Empty when ``extent`` is zero.

    Example:
        ``block_table(896, 512, 4)`` is ``[(2, 384)]`` -- two tiles in one block, second holding 384.
    """
    if extent <= 0:
        return []
    num_tiles = div_ceil(extent, tile)
    last_extent = extent - (num_tiles - 1) * tile
    table = []
    for blk in range(div_ceil(num_tiles, tiles_per_blk)):
        count = min(tiles_per_blk, num_tiles - blk * tiles_per_blk)
        is_final = blk * tiles_per_blk + count == num_tiles
        table.append((count, last_extent if is_final else tile))
    return table


class TransposedBlock:
    """A DMA-transposed block, indexed ``[row_tile, col_tile]`` like the axis-swapped source grid.

    Transposing a ``[P, F]`` block whose F spans several tiles packs one transposed chunk per F tile
    along the free axis, so the loaded grid is ``(1, row_tiles * col_tiles)``, chunk-major. This
    regroups it as ``(row_tiles, col_tiles)``; ``data`` is the whole packed SBUF buffer.
    """

    def __init__(self, loaded: Any, row_tiles: int):
        kernel_assert(
            loaded.shape[0] == 1 and loaded.shape[1] % row_tiles == 0,
            f"transposed block grid {loaded.shape} cannot be split into {row_tiles} row tiles",
        )
        self._loaded = loaded
        self.data = loaded.data
        self.shape = (row_tiles, loaded.shape[1] // row_tiles)

    def __getitem__(self, key: tuple) -> Any:
        row_tile_idx, col_tile_idx = key
        return self._loaded[0, row_tile_idx * self.shape[1] + col_tile_idx]


class RemainderBlockStream(nl.NKIObject):
    """One stream interface over a dimension cut into a tile-aligned body and a remainder block.

    Returned by :func:`remainder_block_stream`; not constructed directly. Local to the FlashAttention-2
    kernels here; it can move to neurotile, with its tests, if it gets wider use. The body keeps a
    normal rotating-buffer ``nt.blocks(...).stream()``, so its DMA still overlaps compute, while the
    remainder is a single block loaded on its own. ``load(index)`` dispatches on the block index, so a
    compute primitive walking blocks never learns the dimension was cut.

    With ``transpose``, :meth:`load` returns a :class:`TransposedBlock`. :meth:`store` is the
    store-side mirror of :meth:`load`.

    ``block_table`` gives ``(tile_count, last_tile_extent)`` per block, in ``load`` index order, for
    sizing compute operands against the partial tile.
    """

    def __init__(
        self,
        body_stream: Optional[nt.BlockStream],
        body_block_count: int,
        remainder_block: Optional[nt.NDSlice],
        table: list[tuple[int, int]],
        transpose: bool = False,
        transposed_row_tiles: int = 1,
    ):
        self._body_stream = body_stream
        self._body_block_count = body_block_count
        self._remainder_block = remainder_block
        self.block_table = table
        self.num_blocks = len(table)
        self._transpose = transpose
        self._transposed_row_tiles = transposed_row_tiles

    def load(self, index: int, dge_mode: Optional[int] = None, engine: Optional[int] = None) -> Any:
        """Load block ``index``, from the body stream or the remainder block as appropriate.

        Args:
            index (int): Block index, ``0 <= index < num_blocks``.
            dge_mode: Descriptor-generation mode, passed through to the underlying load. Dropped with
                ``transpose``.
            engine: Engine to issue the DMA on, passed through to the underlying load. Dropped with
                ``transpose``, whose DMA-transpose path has no engine parameter.

        Returns:
            NDSlice: The loaded block's tile grid, in SBUF. With ``transpose``, a
            :class:`TransposedBlock` indexed like the axis-swapped grid.
        """
        if self._transpose:
            dge_mode = None
            engine = None
        if index < self._body_block_count:
            loaded = self._body_stream.load(index, transpose=self._transpose, dge_mode=dge_mode, engine=engine)
        else:
            loaded = self._remainder_block.load(transpose=self._transpose, dge_mode=dge_mode, engine=engine)
        if self._transpose:
            return TransposedBlock(loaded, self._transposed_row_tiles)
        return loaded

    def store(
        self,
        index: int,
        data: nl.NkiTensor,
        dge_mode: Optional[int] = None,
        engine: Optional[int] = None,
    ) -> None:
        """Store ``data`` (SBUF) into block ``index``'s HBM view -- the store-side mirror of :meth:`load`.

        Args:
            index (int): Block index, ``0 <= index < num_blocks``.
            data: SBUF source (an ``nl.NkiTensor`` or a loaded view's ``.data``).
            dge_mode: Descriptor-generation mode, passed through to the underlying store.
            engine: Engine to issue the DMA on, passed through to the underlying store.
        """
        if self._body_stream != None and index < self._body_block_count:
            target = self._body_stream._child(index)
        else:
            target = self._remainder_block
        target.store(data, dge_mode=dge_mode, engine=engine)


def remainder_block_stream(
    source: Any,
    block_size: tuple,
    tile_size: tuple,
    dim: int,
    buffer_count: int = 2,
    align: Optional[int] = None,
    transpose: bool = False,
) -> RemainderBlockStream:
    """remainder_block_stream(source, block_size, tile_size, dim, buffer_count=2, align=None, transpose=False) -> RemainderBlockStream

    Stream ``source`` in blocks along ``dim`` when that extent is not a whole number of tiles.

    ``blocks(...).stream()`` groups tiles into blocks by index, so a partial final tile lands in a
    block alongside whole ones. On the **partition axis** that block is not a rectangle and cannot be
    transferred as one DMA -- it is rejected at trace time. This factory cuts the dimension at the
    last ``align`` boundary first, so every block is either all-whole tiles or a lone partial tile,
    and hands back both pieces behind one ``load(index)`` interface.

    Cutting also keeps a partial tile from being *paired* with whole ones inside a block, which
    matters even on the free axis: a kernel that narrows its operands per block only has to handle
    "this block is short" rather than "one tile inside this block is short".

    .. warning::

       This API is experimental and may change in future releases.

    Args:
        source (HBM tensor | nl.ndarray | NDSlice): The tensor to stream, already indexed down to the
            axes named by ``tile_size``. Required.
        block_size (tuple[int, ...]): Tiles per block along each dimension, as for
            ``nt.blocks``. Required.
        tile_size (tuple[int, ...]): The per-tile compute-grain shape. Required here, unlike
            ``blocks()``, because the cut point is defined in tiles.
        dim (int): The dimension to stream and cut. Required.
        buffer_count (int): Rotating-buffer depth of the body stream. Defaults to 2.
        align (int | None): Cut granularity along ``dim``. Defaults to ``tile_size[dim]``, the last
            whole tile boundary. Pass an explicit value when several streams over the same logical
            sequence must agree on block boundaries -- for attention, K is tiled 512-wide on its free
            axis but must be cut at V's 128 partition granule so that block ``i`` means the same
            keys in both.
        transpose (bool): DMA-transpose every loaded tile (requires a 2-byte dtype). ``block_size`` /
            ``tile_size`` / ``dim`` describe the untransposed HBM source; each loaded block arrives
            axis-swapped. Defaults to ``False``.

    Returns:
        RemainderBlockStream: Streams ``num_blocks`` blocks; ``block_table`` describes their extents.

    Example:
        Sk=2880 token-major keys with 512-wide tiles, 4 tiles per block, cut at V's 128 granule::

            k_stream = remainder_block_stream(
                k, block_size=(4, 1), tile_size=(512, D), dim=0, align=128, transpose=True
            )
            # block_table == [(4, 512), (2, 256), (1, 64)] -- 2048 + 768 whole, then a 64-row tail
            for b in range(k_stream.num_blocks):
                k_blk = k_stream.load(b)
    """
    kernel_assert(0 <= dim < len(tile_size), f"dim {dim} is out of range for tile_size {tile_size}")
    grain = tile_size[dim] if align == None else align
    kernel_assert(grain > 0, f"align must be positive, got {grain}")

    extent = source.shape[dim]
    remainder = extent % grain
    body = extent - remainder

    def blocks_over(lo: int, hi: int):
        # Slice `dim` to [lo, hi) and leave the other axes whole, then reduce the block grid to the
        # streamed axis so `load(i)` walks blocks along `dim` alone.
        view_index = []
        for axis in range(len(tile_size)):
            view_index.append(slice(lo, hi) if axis == dim else slice(None))
        grid = nt.blocks(source[tuple(view_index)], block_size=block_size, tile_size=tile_size)
        grid_index = []
        for axis in range(len(block_size)):
            grid_index.append(slice(None) if axis == dim else 0)
        return grid[tuple(grid_index)]

    table = block_table(body, tile_size[dim], block_size[dim])
    if remainder > 0:
        table.append((1, remainder))

    # A transpose swaps the block's axes, so its second-axis tile count becomes the row count.
    transposed_row_tiles = block_size[1] if transpose else 1
    if remainder == 0:
        # Nothing to split: an ordinary block stream over the whole dimension.
        body_stream = blocks_over(0, extent).stream(buffer_count=buffer_count, transpose=transpose)
        return RemainderBlockStream(
            body_stream, len(table), None, table, transpose=transpose, transposed_row_tiles=transposed_row_tiles
        )
    remainder_block = blocks_over(body, extent)[0]
    if body == 0:
        # Nothing but the remainder: one block, so there is no body to stream.
        return RemainderBlockStream(
            None, 0, remainder_block, table, transpose=transpose, transposed_row_tiles=transposed_row_tiles
        )
    body_stream = blocks_over(0, body).stream(buffer_count=buffer_count)
    return RemainderBlockStream(
        body_stream,
        len(table) - 1,
        remainder_block,
        table,
        transpose=transpose,
        transposed_row_tiles=transposed_row_tiles,
    )
