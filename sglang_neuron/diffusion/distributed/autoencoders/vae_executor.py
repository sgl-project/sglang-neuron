# SPDX-License-Identifier: Apache-2.0
"""Neuron distributed VAE executor, decoder tile geometry and the VAE attention kernel."""

import math
from dataclasses import dataclass, field

import nki
import nki.language as nl
import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
import torch.nn.functional as F
from nkilib.core.attention.attention_cte import attention_cte

from sglang_neuron.diffusion import compile_backend
from sglang_neuron.diffusion.distributed.groups import get_world_group
from sglang_neuron.diffusion.neuron_api import (
    register_process_group_replica_groups,
    wrap_nki_kernel,
)


def decoder_tile_geometry(
    dimension: int, tile_size: int, stride: int
) -> tuple[tuple[int, int, int, int], ...]:
    """Map logical tiles to full-size input windows and output crops.

    Each entry is ``(logical_start, input_start, crop_offset, valid_size)``.
    Duplicate terminal windows are omitted.
    """
    if tile_size <= 0 or stride <= 0:
        raise ValueError("tile_size and stride must be positive")
    if stride > tile_size:
        raise ValueError("stride cannot exceed tile_size without leaving gaps")
    if dimension <= 0:
        raise ValueError("dimension must be positive")

    padded_dimension = max(dimension, tile_size)
    geometry = []
    seen_input_starts = set()
    for logical_start in range(0, dimension, stride):
        input_start = min(logical_start, padded_dimension - tile_size)
        if input_start in seen_input_starts:
            continue
        seen_input_starts.add(input_start)
        geometry.append(
            (
                logical_start,
                input_start,
                logical_start - input_start,
                min(tile_size, dimension - logical_start),
            )
        )
    return tuple(geometry)


@nki.jit
def _vae_attention_kernel(q, k, v):
    return attention_cte(
        q=q,
        k=k,
        v=v,
        scale=1.0,  # or pass scale=C**-0.5, see below
        causal_mask=False,
        tp_q=True,
        tp_k=True,
        tp_out=False,
        cache_softmax=False,
        softmax_dtype=nl.float32,
        mm_out_dtype=nl.float32,
    )


def vae_nki_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Run the VAE attention kernel through ``wrap_nki``, the traceable NKI HOP.

    ``wrap_nki`` shape-infers in its meta impl, and every NKI kernel here uses it.
    """
    return wrap_nki_kernel(_vae_attention_kernel, supported_lnc=(2,))(q, k, v)


def new_vae_parallel_groups(ranks: list[int]) -> tuple[dist.ProcessGroup, dist.ProcessGroup]:
    """Create the VAE-parallel device group and its Gloo twin. Collective: every rank calls it.

    Returns:
        ``(device_group, error_group)`` over ``ranks``.

    The executor's payload collectives run on the device group; its host-gather error check
    all-reduces a CPU tensor, so it needs a Gloo group over the same ranks.
    """
    return dist.new_group(ranks=ranks), dist.new_group(ranks=ranks, backend="gloo")


@dataclass
class GridSpec:
    """The tile grid of one decode: the split dims, the grid shape and the tile geometry."""

    split_dims: tuple[int, ...]
    grid_shape: tuple[int, ...]
    tile_spec: dict = field(default_factory=dict)


class NeuronDistributedVaeExecutor:
    """Route control metadata over CPU/Gloo and payload collectives over the device.

    Until ``bind_group`` is called the executor spans the whole worker world.
    """

    MAX_DEVICE_GATHER_BYTES = 512 * 1024 * 1024

    def __init__(self):
        world = get_world_group()
        self.group = world.device_group
        self._error_group = world.cpu_group
        self.world_size = dist.get_world_size(self.group)
        self.rank = dist.get_rank(self.group)
        self.parallel_size = 1
        self.parallel_mode = "tile"

    def set_parallel_size(self, parallel_size: int, mode: str = "tile") -> None:
        self.parallel_size = parallel_size
        self.parallel_mode = mode

    def gather_tensors(self, tensor: torch.Tensor) -> list[torch.Tensor] | None:
        """Host all-gather of an equal-shape ``tensor`` over the group: the list on group rank 0,
        ``None`` elsewhere.

        Every rank first reports whether its receive buffers could be allocated (over the Gloo
        error group), so a rank that fails does not leave the others waiting in the gather.
        """
        error = None
        try:
            gathered = [torch.empty_like(tensor) for _ in range(self.world_size)]
        except Exception as exc:  # noqa: BLE001 - reported to every rank, then raised
            error = exc
        failed = torch.tensor([int(error is not None)], dtype=torch.int32)
        dist.all_reduce(failed, op=dist.ReduceOp.MAX, group=self._error_group)
        if failed.item():
            where = "" if error is not None else " on another rank"
            raise RuntimeError(f"VAE host gather allocation failed{where}") from error
        dist.all_gather(gathered, tensor, group=self.group)
        return gathered if self.rank == 0 else None

    def bind_group(self, group: dist.ProcessGroup, error_group: dist.ProcessGroup | None) -> None:
        """Run every collective over the VAE-parallel subgroup only.

        Only the subgroup's ranks run the executor, so a world-wide error check would wait on
        ranks that never join it; bind the error check to the subgroup's Gloo group as well.
        """
        if error_group is None:
            raise ValueError(
                "bind_group needs the subgroup's Gloo error group; create both with "
                "new_vae_parallel_groups()"
            )
        self.group = group
        self.world_size = dist.get_world_size(group)
        self.rank = dist.get_rank(group)
        self._error_group = error_group

    def _compile_device_graph(self, name, key, fn):
        """Compile and cache a fixed-shape VAE payload graph."""
        graphs = getattr(self, "_device_graphs", None)
        if graphs is None:
            graphs = {}
            self._device_graphs = graphs
        graph = graphs.get((name, key))
        if graph is None:
            graph = torch.compile(
                fn,
                backend=compile_backend.register(),
                fullgraph=True,
                dynamic=False,
                options={"model_name": f"vae_{name}"},
            )
            graphs[(name, key)] = graph
        return graph

    def _build_coord_index(self, meta_gather, grid_spec, tid_coord_map):
        """Map each grid coordinate to the ``(rank, slot)`` holding its tile."""
        coord_index = {}
        for rank in range(self.world_size):
            meta_src = meta_gather[rank]
            for idx in range(meta_src.shape[0]):
                tid = int(meta_src[idx, 0])
                if tid < 0:
                    continue
                coord_index[tid_coord_map[tid]] = (rank, idx)
        return coord_index

    def _compiled_blend_graph(
        self,
        slot_shapes,
        grid_height,
        grid_width,
        heights,
        widths,
        crop_tops,
        crop_lefts,
        full_height,
        full_width,
        stride_height,
        stride_width,
        blend_height,
        blend_width,
        clamp,
    ):
        """Compile (cached) the tile blend for one plane chunk. Slots arrive
        pre-sliced to ``[chunk_planes, H, W]``; keyed on shapes + geometry."""

        def blend_vertical(above, tile):
            weights = (
                torch.arange(blend_height, device=tile.device, dtype=tile.dtype) / blend_height
            ).reshape(blend_height, 1)
            blended = (
                above[..., -blend_height:, :] * (1.0 - weights)
                + tile[..., :blend_height, :] * weights
            )
            return torch.cat((blended, tile[..., blend_height:, :]), dim=-2)

        def blend_horizontal(left, tile):
            weights = (
                torch.arange(blend_width, device=tile.device, dtype=tile.dtype) / blend_width
            ).reshape(1, blend_width)
            blended = left[..., -blend_width:] * (1.0 - weights) + tile[..., :blend_width] * weights
            return torch.cat((blended, tile[..., blend_width:]), dim=-1)

        def merge(*slots):
            blended_tiles = []
            rows = []
            for row in range(grid_height):
                row_tiles = []
                for column in range(grid_width):
                    index = row * grid_width + column
                    crop_top = crop_tops[row]
                    crop_left = crop_lefts[column]
                    tile = slots[index][
                        ...,
                        crop_top : crop_top + heights[row],
                        crop_left : crop_left + widths[column],
                    ]
                    if row > 0:
                        tile = blend_vertical(blended_tiles[index - grid_width], tile)
                        if column > 0:
                            tile = tile.clone(memory_format=torch.contiguous_format)
                    if column > 0:
                        tile = blend_horizontal(blended_tiles[index - 1], tile)
                    blended_tiles.append(tile)
                    output_height = heights[row] if row == grid_height - 1 else stride_height
                    output_width = widths[column] if column == grid_width - 1 else stride_width
                    row_tiles.append(tile[..., :output_height, :output_width])
                rows.append(torch.cat(row_tiles, dim=-1))
            merged = torch.cat(rows, dim=-2)[..., :full_height, :full_width]
            return torch.clamp(merged, min=-1.0, max=1.0) if clamp else merged

        key = (
            slot_shapes,
            (grid_height, grid_width),
            heights,
            widths,
            crop_tops,
            crop_lefts,
            (
                full_height,
                full_width,
                stride_height,
                stride_width,
                blend_height,
                blend_width,
                clamp,
            ),
        )
        # Name stays "merge" so its compilation remains observable to tests.
        return self._compile_device_graph("merge", key, merge)

    def _gather_chunk(self, compiled_gather, chunk, chunk_index):
        """Run one device all-gather, then the cross-rank error sync; returns the gathered
        ``[world, *chunk.shape]`` tensor on rank 0, ``None`` elsewhere.

        torch-neuronx queues the gather and returns, so no local error is caught here; the
        sync only reports failures the other ranks raised before it.
        """
        gathered = compiled_gather(chunk)
        result = gathered if self.rank == 0 else None
        del gathered
        self._raise_if_device_gather_failed(None, f"post-gather chunk {chunk_index}")
        return result

    def _plane_chunk(self, local_planes, start, size):
        """``local_planes[:, start:start + size]`` as a contiguous device tensor.

        With one tile slot per rank the slice is already contiguous. With more it is strided,
        and the copy runs as a fixed-shape compiled graph rather than an eager ``.contiguous()``
        on the strided device view. The plane indices are a graph input, so chunks of one size
        share a graph: a graph per ``start`` fails past Dynamo's recompile limit (8).
        """
        chunk = local_planes.narrow(1, start, size)
        if chunk.is_contiguous():
            return chunk

        def take(planes_tensor, plane_indices):
            return planes_tensor.index_select(1, plane_indices)

        graph = self._compile_device_graph(
            "plane_chunk", (tuple(local_planes.shape), local_planes.dtype, size), take
        )
        return graph(local_planes, torch.arange(start, start + size, device=local_planes.device))

    def _stream_gather_planes(self, local_planes, chunk_planes, planes):
        """Yield the plane-chunk gathers, with the setup error check hoisted out of the
        per-chunk loop (only the post-gather check stays per chunk)."""
        starts = list(range(0, planes, chunk_planes))

        if local_planes.device.type == "cpu":
            for start in starts:
                size = min(chunk_planes, planes - start)
                yield self.gather_tensors(local_planes.narrow(1, start, size).contiguous())
            return

        setup_error = None
        try:
            compiled_gather = self._get_compiled_device_gather()
        except Exception as error:
            setup_error = error
        self._raise_if_device_gather_failed(setup_error, "setup")

        for chunk_index, start in enumerate(starts):
            size = min(chunk_planes, planes - start)
            local_chunk = self._plane_chunk(local_planes, start, size)
            gathered = self._gather_chunk(compiled_gather, local_chunk.reshape(-1), chunk_index)
            if gathered is None:
                yield None
                continue
            chunk_views = list(gathered.reshape(self.world_size, *local_chunk.shape).unbind(0))
            del gathered  # drop the alias; chunk_views still pins the storage
            yield chunk_views
            del chunk_views  # on resume, free the buffer before the next gather

    def gather_and_blend_tiles(
        self,
        local_tile_tensor,
        meta_gather,
        grid_spec,
        tid_coord_map,
        *,
        full_height,
        full_width,
        stride_height,
        stride_width,
        blend_height,
        blend_width,
        clamp,
        crop_tops=None,
        crop_lefts=None,
    ):
        """All-gather and blend the padded tiles, streamed over plane chunks.

        Instead of gathering the whole ``[world, slots, C, T, H, W]`` payload to
        rank 0 (a large payload runs the rank out of memory) then merging, this gathers
        only one channel*frame ("planes") slice at a time, blends it on rank 0, and
        frees it before the next. Peak is one plane chunk plus the full output.
        Every rank drives the per-chunk gather (collective); only rank 0 blends and
        returns a tensor, others return ``None``.
        """
        slots = local_tile_tensor.shape[0]
        slot_shape = tuple(local_tile_tensor.shape[1:])
        slot_height, slot_width = slot_shape[-2], slot_shape[-1]
        frame_shape = slot_shape[:-2]
        planes = math.prod(frame_shape)
        local_planes = local_tile_tensor.reshape(slots, planes, slot_height, slot_width)

        # Take the largest plane chunk whose gathered form
        # ([world, slots, chunk_planes, H, W]) still fits the collective budget, so
        # each per-chunk gather is single-shot (no internal element-chunk join) and
        # the chunk count matches the element chunks of one full-payload gather --
        # no extra collectives. A smaller payload gathers whole in one chunk.
        plane_bytes = (
            self.world_size * slots * slot_height * slot_width * local_tile_tensor.element_size()
        )
        budget_planes = (
            max(1, self.MAX_DEVICE_GATHER_BYTES // plane_bytes) if plane_bytes else planes
        )
        chunk_planes = max(1, min(planes, budget_planes))

        grid_height, grid_width = grid_spec.grid_shape
        coord_index = (
            self._build_coord_index(meta_gather, grid_spec, tid_coord_map)
            if self.rank == 0
            else None
        )
        heights = tuple(
            min(slot_height, full_height - row * stride_height) for row in range(grid_height)
        )
        widths = tuple(
            min(slot_width, full_width - column * stride_width) for column in range(grid_width)
        )
        crop_tops = tuple(0 for _ in range(grid_height)) if crop_tops is None else tuple(crop_tops)
        crop_lefts = (
            tuple(0 for _ in range(grid_width)) if crop_lefts is None else tuple(crop_lefts)
        )
        if len(crop_tops) != grid_height or len(crop_lefts) != grid_width:
            raise ValueError("Tile crop offsets must match the tile grid")

        # With several plane chunks, joining the merged chunks on device needs a second buffer of
        # the full output, which for a large output rank 0 cannot afford next to the resident
        # DiT. The decoded frames go to the host anyway, so copy each chunk there as soon as it
        # is blended and join on the host; a single chunk stays on device.
        chunks_to_host = chunk_planes < planes
        merged_chunks = []
        for gathered in self._stream_gather_planes(local_planes, chunk_planes, planes):
            if self.rank != 0:
                continue
            ordered = [
                gathered[coord_index[(row, column)][0]][coord_index[(row, column)][1]]
                for row in range(grid_height)
                for column in range(grid_width)
            ]
            blend = self._compiled_blend_graph(
                tuple(tuple(tile.shape) for tile in ordered),
                grid_height,
                grid_width,
                heights,
                widths,
                crop_tops,
                crop_lefts,
                full_height,
                full_width,
                stride_height,
                stride_width,
                blend_height,
                blend_width,
                clamp,
            )
            merged = blend(*ordered)
            merged_chunks.append(merged.to("cpu") if chunks_to_host else merged)
            del gathered, ordered, merged  # free the device buffers before the next chunk

        if self.rank != 0:
            return None
        if len(merged_chunks) == 1:
            result = merged_chunks[0]
        else:
            result = torch.cat(merged_chunks, dim=0)
        return result.reshape(*frame_shape, full_height, full_width)

    def _get_compiled_device_gather(self):
        compiled_gather = getattr(self, "_compiled_device_gather", None)
        if compiled_gather is None:
            group = self.group
            world_size = self.world_size

            group_name = getattr(group, "group_name", None)
            if group_name is None:
                raise RuntimeError("VAE process group has no group_name for compilation")
            group_ranks = torch.distributed.get_process_group_ranks(group)
            global_world_size = torch.distributed.get_world_size()
            if group_ranks != list(range(world_size)) or global_world_size % world_size != 0:
                raise RuntimeError(
                    "VAE process group must be a contiguous rank-0 group whose size "
                    "evenly divides the global world for SPMD compilation"
                )
            register_process_group_replica_groups(
                group_name,
                [
                    list(range(start, start + world_size))
                    for start in range(0, global_world_size, world_size)
                ],
            )

            def device_gather(tensor):
                gathered = funcol.all_gather_tensor(tensor, gather_dim=0, group=group)
                return gathered.reshape(world_size, *tensor.shape)

            compiled_gather = torch.compile(
                device_gather,
                backend=compile_backend.register(),
                fullgraph=True,
                dynamic=False,
                options={"model_name": "vae_gather"},
            )
            self._compiled_device_gather = compiled_gather
        return compiled_gather

    def _raise_if_device_gather_failed(self, local_error, phase):
        failure = torch.tensor(
            [int(local_error is not None)],
            device="cpu",
            dtype=torch.int32,
        )
        torch.distributed.all_reduce(
            failure,
            op=torch.distributed.ReduceOp.MAX,
            group=self.group,
        )
        if not failure.item():
            return

        message = f"VAE device gather failed during {phase}"
        if local_error is not None:
            raise RuntimeError(message) from local_error
        raise RuntimeError(message)

    def _pack_local_tiles(self, local_results, global_padding_shape, grid_spec, device, dtype):
        """Pad this rank's tiles to the slot shape and stack them, on device.

        Writing each tile into a preallocated tensor by strided assignment is not something
        the device kernels can do, so the pad-and-stack runs as a compiled graph. The metadata
        tensor stays on CPU: rank 0 reads tile ids out of it as host ints.
        """
        slots = global_padding_shape[0]
        meta_tensor = torch.full(
            (slots, len(grid_spec.split_dims) + 1),
            -1,
            device="cpu",
            dtype=torch.int64,
        )
        for idx, (tid, t_tensor) in enumerate(local_results):
            meta_tensor[idx, 0] = tid
            for i, dim in enumerate(grid_spec.split_dims):
                meta_tensor[idx, i + 1] = t_tensor.shape[dim]

        if not local_results:
            return torch.zeros(global_padding_shape, device=device, dtype=dtype), meta_tensor

        slot_shape = tuple(global_padding_shape[1:])
        tiles = [tile for _, tile in local_results]
        key = (slot_shape, tuple(tuple(tile.shape) for tile in tiles), slots)

        def pack(*tile_tensors):
            padded = []
            for tile in tile_tensors:
                pad = []
                for axis in range(len(slot_shape) - 1, -1, -1):
                    pad.extend((0, slot_shape[axis] - tile.shape[axis]))
                padded.append(F.pad(tile, pad) if any(pad) else tile)
            stacked = torch.stack(padded, dim=0)
            if stacked.shape[0] < slots:
                empty = torch.zeros(
                    (slots - stacked.shape[0], *slot_shape),
                    dtype=stacked.dtype,
                    device=stacked.device,
                )
                stacked = torch.cat((stacked, empty), dim=0)
            return stacked

        return self._compile_device_graph("pack", key, pack)(*tiles), meta_tensor
