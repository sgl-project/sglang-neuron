# SPDX-License-Identifier: Apache-2.0
"""Tile-parallel video VAE decode and encode on the Neuron executor.

The merges stream their payload, so no rank holds every gathered tile: ``gather_tiles`` gathers
to rank 0's host (encode); the column merge stitches each rank's columns on its device and sends
rank 0 only the kept pixels (decode).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn.functional as F

from sglang_neuron.diffusion.distributed.autoencoders.vae_executor import (
    GridSpec,
    NeuronDistributedVaeExecutor,
)


@dataclass
class TileGridSpec(GridSpec):
    """A :class:`GridSpec` with the dtype of the merged output (None: the input's)."""

    output_dtype: torch.dtype | None = None


@dataclass
class TileTask:
    tile_id: int
    grid_coord: tuple[int, ...]  # the tile's coordinate in TileGridSpec.grid_shape
    tensor: torch.Tensor | list[torch.Tensor]  # the tile's input
    workload: int | float = 1


@dataclass
class DistributedOperator:
    """How a tiled VAE call splits its input, runs one tile and merges the results.

    ``merge(local_tile_tensor, meta_gather, grid_spec, tid_coord_map)`` receives this rank's
    padded tiles and the gathered tile metadata (on rank 0) and gathers the payload itself.
    ``balance(tasks, num_rank)`` deals the tiles out to the ranks, one list per rank; the default
    is the executor's largest-first balance.
    """

    split: Callable[[torch.Tensor], tuple[list[TileTask], TileGridSpec]]
    exec: Callable[[TileTask], torch.Tensor]
    merge: Callable[..., torch.Tensor]
    balance: Callable[[list[TileTask], int], list[list[TileTask]]] | None = None


class TiledVaeExecutor(NeuronDistributedVaeExecutor):
    """The shared Neuron executor with a tile split / run / streamed merge on top."""

    # Ranks that take the column merge's columns first, when that costs no rank an extra column
    # (None: every rank).
    tile_ranks: list[int] | None = None

    @contextmanager
    def _sync_errors(self, stage: str):
        """Agree on local-stage failures over the Gloo error group before the next collective,
        so a rank that fails does not leave the others waiting in it."""
        status = 0
        message = ""
        try:
            yield
        except torch.OutOfMemoryError as error:
            status = 1
            message = str(error)
        except Exception as error:  # noqa: BLE001 - reported to every rank, then raised
            # A non-retryable error takes precedence over OOM on another rank.
            status = 2
            message = f"{type(error).__name__}: {error}"

        # Reporting an OOM must not need another device allocation: a CPU tensor.
        error_status = torch.tensor(status, dtype=torch.int32)
        dist.all_reduce(error_status, op=dist.ReduceOp.MAX, group=self._error_group)
        global_status = error_status.item()
        if global_status == 2:
            detail = message if status == 2 else "another rank failed"
            raise RuntimeError(f"Distributed VAE {stage} failed: {detail}")
        if global_status == 1:
            detail = message if status == 1 else "another rank ran out of memory"
            raise torch.OutOfMemoryError(f"Distributed VAE {stage} failed: {detail}")

    def _balance_tasks(self, task_list, num_rank):
        """Largest workload first, each to the least loaded rank."""
        workloads = [0] * num_rank
        assigned: list[list[TileTask]] = [[] for _ in range(num_rank)]
        for task in sorted(task_list, key=lambda t: t.workload, reverse=True):
            rank = workloads.index(min(workloads))
            assigned[rank].append(task)
            workloads[rank] += task.workload
        return assigned

    def _compute_global_padding_shape(self, local_results, output_ndim: int, device):
        """``[max tiles per rank, *max tile dims]`` over the group: the slot shape every rank
        pads its tiles to."""
        with self._sync_errors("shape preparation"):
            local_tile_max_dims = [0] * output_ndim
            for _, tile_tensor in local_results:
                for dim_idx, dim_size in enumerate(tile_tensor.shape):
                    local_tile_max_dims[dim_idx] = max(local_tile_max_dims[dim_idx], dim_size)
            local_shape_stat = torch.tensor(
                [len(local_results), *local_tile_max_dims], device=device
            )
        dist.all_reduce(local_shape_stat, op=dist.ReduceOp.MAX, group=self.group)
        return local_shape_stat.tolist()

    def _unpack_tiles(self, meta_gather, tile_gather, grid_spec, tid_coord_map):
        """``{grid_coord: tile}`` from the gathered padded tiles, each cropped to its size."""
        coord_tensor_map = {}
        for r in range(self.world_size):
            meta_src = meta_gather[r]
            tiles_src = tile_gather[r]
            for idx in range(meta_src.shape[0]):
                tid = int(meta_src[idx, 0])
                if tid < 0:
                    continue
                slices_list = [slice(None)] * tiles_src[idx].ndim
                for i, dim in enumerate(grid_spec.split_dims):
                    slices_list[dim] = slice(0, int(meta_src[idx, i + 1]))
                coord_tensor_map[tid_coord_map[tid]] = tiles_src[idx][tuple(slices_list)]
        return coord_tensor_map

    def execute(self, z, operator: DistributedOperator):
        """Tiled VAE decode / encode: the merged result on group rank 0, an empty tensor elsewhere.

        Only the tile metadata is gathered here; ``operator.merge`` streams the payload itself.
        A failure while splitting, running or packing this rank's tiles is reported to every
        rank before the next collective. The merge is not wrapped: it runs its own collectives,
        whose device gathers check their own errors.
        """
        pp_size = min(self.parallel_size, self.world_size)

        with self._sync_errors("tile decode"):
            tiletask_list, grid_spec = operator.split(z)
            tid_coord_map = {task.tile_id: task.grid_coord for task in tiletask_list}
            balance = operator.balance or self._balance_tasks
            assigned = balance(tiletask_list, pp_size)
            local_tasks = assigned[self.rank] if self.rank < pp_size else []
            local_results = [(t.tile_id, operator.exec(t)) for t in local_tasks]

        global_padding_shape = self._compute_global_padding_shape(local_results, z.ndim, z.device)
        output_dtype = grid_spec.output_dtype if grid_spec.output_dtype is not None else z.dtype
        with self._sync_errors("tile packing"):
            local_tile_tensor, local_meta_tensor = self._pack_local_tiles(
                local_results, global_padding_shape, grid_spec, z.device, output_dtype
            )

        meta_gather = self.gather_tensors(local_meta_tensor)

        result = operator.merge(local_tile_tensor, meta_gather, grid_spec, tid_coord_map)
        if self.rank != 0:
            result = torch.empty(0, device=z.device)
        return result

    def gather_tiles(self, local_tile_tensor, meta_gather, grid_spec, tid_coord_map):
        """All-gather the padded tiles to rank 0's host in plane chunks: the ``{grid_coord: tile}``
        map on rank 0, ``None`` elsewhere. For merges ``gather_and_blend_tiles`` cannot express.
        """
        slots = local_tile_tensor.shape[0]
        slot_shape = tuple(local_tile_tensor.shape[1:])
        slot_height, slot_width = slot_shape[-2], slot_shape[-1]
        planes = math.prod(slot_shape[:-2])
        local_planes = local_tile_tensor.reshape(slots, planes, slot_height, slot_width)
        plane_bytes = (
            self.world_size * slots * slot_height * slot_width * local_tile_tensor.element_size()
        )
        budget_planes = (
            max(1, self.MAX_DEVICE_GATHER_BYTES // plane_bytes) if plane_bytes else planes
        )
        chunk_planes = max(1, min(planes, budget_planes))

        # Filled chunk by chunk: concatenating the host chunks would hold the gathered tiles twice.
        tile_gather = None
        if self.rank == 0:
            tile_gather = torch.empty(
                (self.world_size, slots, planes, slot_height, slot_width),
                dtype=local_tile_tensor.dtype,
            )
        start = 0
        for gathered in self._stream_gather_planes(local_planes, chunk_planes, planes):
            if self.rank != 0:
                continue
            stack = self._compile_device_graph(
                "stack",
                tuple(tuple(chunk.shape) for chunk in gathered),
                lambda *chunks: torch.stack(chunks),
            )
            stacked = stack(*gathered)
            size = stacked.shape[2]
            tile_gather[:, :, start : start + size] = stacked.to("cpu")
            start += size
            del gathered, stacked  # free the device buffers before the next chunk

        if self.rank != 0:
            return None
        tile_gather = tile_gather.reshape(self.world_size, slots, *slot_shape)
        return self._unpack_tiles(meta_gather, tile_gather, grid_spec, tid_coord_map)

    def column_layout(self, num_groups: int, num_columns: int) -> list[tuple[int, int]]:
        """The ``(rank, slot)`` of column ``group * num_columns + column``: round-robin over
        ``tile_ranks``, or over every rank when they would hold more columns each."""
        num_rank = min(self.parallel_size, self.world_size)
        num_runs = num_groups * num_columns
        ranks = list(range(num_rank))
        if self.tile_ranks is not None:
            preferred = [rank for rank in self.tile_ranks if rank < num_rank]
            if preferred and -(-num_runs // len(preferred)) <= -(-num_runs // num_rank):
                ranks = preferred
        return [(ranks[run % len(ranks)], run // len(ranks)) for run in range(num_runs)]

    def balance_columns(self, task_list, num_rank):
        """Tiles at ``(*group, row, column)``, by ``column_layout``: a rank's columns in slot
        order, each one's rows in order."""
        groups = sorted({task.grid_coord[:-2] for task in task_list})
        num_columns = 1 + max(task.grid_coord[-1] for task in task_list)
        layout = self.column_layout(len(groups), num_columns)
        group_index = {group: index for index, group in enumerate(groups)}
        assigned = [[] for _ in range(num_rank)]

        def place(task):
            rank, slot = layout[
                group_index[task.grid_coord[:-2]] * num_columns + task.grid_coord[-1]
            ]
            return rank, slot, task.grid_coord[-2]

        for task in sorted(task_list, key=place):
            assigned[place(task)[0]].append(task)
        return assigned

    def run_graph(self, name, key, fn, *args):
        """``fn(*args)`` as a cached compiled graph on the device, eagerly on the host."""
        if args[0].device.type == "cpu":
            return fn(*args)
        return self._compile_device_graph(name, key, fn)(*args)

    def all_gather_pieces(self, tensor: torch.Tensor):
        """Every rank's ``tensor`` on every rank, in pieces of its leading dim within
        ``MAX_DEVICE_GATHER_BYTES``: yields ``(start, [world, piece, *tensor.shape[1:]])`` in
        order. Collective.

        The pieces share one length (each length recompiles the shared gather graph), so the
        last one may start inside the one before."""
        length = tensor.shape[0]
        piece_bytes = math.prod(tensor.shape[1:]) * tensor.element_size() * self.world_size
        size = max(1, min(length, self.MAX_DEVICE_GATHER_BYTES // piece_bytes))
        starts = [min(start, length - size) for start in range(0, length, size)]
        if tensor.device.type == "cpu":
            for start in starts:
                piece = tensor[start : start + size].contiguous()
                gathered = [torch.empty_like(piece) for _ in range(self.world_size)]
                torch.distributed.all_gather(gathered, piece, group=self.group)
                yield start, torch.stack(gathered)
            return

        setup_error = None
        try:
            compiled_gather = self._get_compiled_device_gather()
        except Exception as error:
            setup_error = error
        self._raise_if_device_gather_failed(setup_error, "setup")
        # torch-neuronx queues each gather and returns, so no local error is caught per piece;
        # one check after the last reports the other ranks' failures (as _gather_chunk).
        for start in starts:
            piece = tensor[start : start + size]
            gathered = compiled_gather(piece.reshape(-1))
            yield start, gathered.reshape(self.world_size, *piece.shape)
            del gathered  # free the buffer before the next gather
        self._raise_if_device_gather_failed(None, "pieces")

    def gather_slots(self, tensor: torch.Tensor, sources: list[tuple[int, int]]) -> torch.Tensor:
        """``tensor[:, slot]`` of rank ``rank`` for each ``(rank, slot)`` in ``sources``, stacked
        on dim 1. Every rank holds ``tensor`` in one ``(length, slots, ...)`` shape and names its
        own ``sources``. Collective."""
        slots = tensor.shape[1]
        index = torch.tensor([rank * slots + slot for rank, slot in sources], device=tensor.device)
        pieces, starts = [], []
        for start, gathered in self.all_gather_pieces(tensor):
            key = (tuple(gathered.shape), gathered.dtype, len(sources))
            pieces.append(self.run_graph("pick_slots", key, _pick_slots, gathered, index))
            starts.append(start)
        if len(pieces) == 1:
            return pieces[0]
        # The last piece's overlap with the one before.
        skip = starts[-2] + pieces[0].shape[0] - starts[-1]

        def join(*pieces):
            return torch.cat([*pieces[:-1], pieces[-1][skip:]])

        key = (tuple(tensor.shape), tensor.dtype, len(sources), len(pieces))
        return self.run_graph("join_slots", key, join, *pieces)

    def stitch_columns(
        self,
        tiles: torch.Tensor,
        layout: list[tuple[int, int]],
        num_columns: int,
        height_overlaps: list[int],
        width_overlaps: list[int],
        blend,
        select: list[int] | None = None,
        epilogue=None,
        key=(),
    ):
        """This rank's columns from ``column_layout``, stitched on its device: ``(columns,
        *epilogue(columns))`` with ``columns`` in slot order on a rank that holds any, None
        elsewhere. Collective.

        ``tiles`` are this rank's ``(slots * rows, B, C, F, H, W)`` tiles, ``select`` the frames
        they keep, and ``blend(a, b, extent, dim)`` the model's overlap blend. Each tile blends
        the original tile above it, then the original tile to its left (its right edge comes from
        an all-gather), and keeps the rows and columns before the next tile's overlap.
        ``epilogue`` adds outputs to the same cached graph; ``key`` must name everything it
        reads."""
        rank = self.rank
        runs = [run for run, (owner, _) in enumerate(layout) if owner == rank]
        num_slots = 1 + max(slot for _, slot in layout)
        num_rows = len(height_overlaps) + 1
        _, batch, channels, num_frames, tile_h, tile_w = tiles.shape
        tiles = tiles.reshape(-1, num_rows, *tiles.shape[1:])
        keep_h = [tile_h - overlap for overlap in height_overlaps] + [tile_h]
        keep_w = [tile_w - overlap for overlap in width_overlaps] + [tile_w]
        tops = [sum(keep_h[:row]) for row in range(num_rows)]
        select = list(range(num_frames)) if select is None else list(select)
        index = torch.tensor(select, device=tiles.device)
        columns = [run % num_columns for run in runs]
        key = (
            tuple(tiles.shape),
            tiles.dtype,
            tuple(select),
            tuple(height_overlaps),
            tuple(width_overlaps),
            tuple(columns),
            num_slots,
            blend,
            epilogue is not None,
            key,
        )

        halo = max(width_overlaps, default=0)
        left = ()
        if num_columns > 1:
            edge_shape = (len(select), num_slots, batch, channels, sum(keep_h), halo)
            if runs:

                def right_edges(tiles, index):
                    tiles = tiles.index_select(4, index)
                    edges = [
                        torch.cat(
                            [
                                tiles[slot, row, ..., : keep_h[row], tile_w - halo :]
                                for row in range(num_rows)
                            ],
                            dim=-2,
                        )
                        for slot in range(len(runs))
                    ]
                    edges = torch.stack(edges, dim=0).permute(3, 0, 1, 2, 4, 5)
                    return F.pad(edges, (0, 0, 0, 0, 0, 0, 0, 0, 0, num_slots - len(runs)))

                edges = self.run_graph("right_edges", key, right_edges, tiles, index)
            else:
                edges = torch.zeros(edge_shape, dtype=tiles.dtype, device=tiles.device)
            # A run in the first column reads its own slot, unused.
            sources = [layout[run - 1] if run % num_columns else (rank, 0) for run in runs]
            received = self.gather_slots(edges, sources or [(rank, 0)])
            left = (received,)
        if not runs:
            return None

        def stitch(tiles, index, *left):
            tiles = tiles.index_select(4, index)
            stitched = []
            for slot, column in enumerate(columns):
                rows = []
                for row in range(num_rows):
                    tile = tiles[slot, row]
                    if row > 0:
                        tile = blend(tiles[slot, row - 1], tile, height_overlaps[row - 1], dim=-2)
                    tile = tile[..., : keep_h[row], :]
                    if column > 0:
                        edge = left[0][:, slot, :, :, tops[row] : tops[row] + keep_h[row]]
                        tile = blend(
                            edge.permute(1, 2, 0, 3, 4), tile, width_overlaps[column - 1], dim=-1
                        )
                    rows.append(tile[..., : keep_w[column]])
                stitched.append(torch.cat(rows, dim=-2))
            return (*stitched, *(epilogue(stitched) if epilogue is not None else ()))

        outputs = self.run_graph("stitch", key, stitch, tiles, index, *left)
        return (outputs[: len(runs)], *outputs[len(runs) :])

    def gather_columns(
        self, blocks: torch.Tensor, layout: list[tuple[int, int]], num_columns, widths
    ):
        """Every column's block on rank 0's device, joined across each group's columns: yields
        ``(start, [groups, piece, ...])`` per piece of the leading dim on rank 0, ``(start,
        None)`` elsewhere. Collective.

        ``blocks`` is each rank's ``(length, slots, ..., width, C)``, padded to the widest
        column; column ``j`` keeps ``widths[j]``."""
        num_groups = len(layout) // num_columns

        def join(gathered):
            return torch.stack(
                [
                    torch.cat(
                        [
                            gathered[rank, :, slot, ..., : widths[j], :]
                            for j in range(num_columns)
                            for rank, slot in (layout[group * num_columns + j],)
                        ],
                        dim=-2,
                    )
                    for group in range(num_groups)
                ]
            )

        for start, gathered in self.all_gather_pieces(blocks):
            if self.rank != 0:
                yield start, None
                continue
            key = (tuple(gathered.shape), gathered.dtype, tuple(widths), tuple(layout))
            yield start, self.run_graph("join_columns", key, join, gathered)


def _pick_slots(gathered: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """``[world, piece, slots, ...]`` to ``[piece, len(index), ...]``, picking flat
    ``rank * slots + slot`` entries.

    The pick is on dim 0, then transposed: on dim 1, neuronx-cc's verifier rejects some odd
    pieces of 9 or more rows with NCC_EBIR026."""
    world, piece, slots = gathered.shape[:3]
    flat = gathered.transpose(1, 2).reshape(world * slots, piece, *gathered.shape[3:])
    return flat.index_select(0, index).transpose(0, 1)
