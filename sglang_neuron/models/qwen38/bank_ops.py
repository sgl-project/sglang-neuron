# SPDX-License-Identifier: Apache-2.0
"""The working bank's row programs (``StateOps.load_rows`` / ``publish_rows``) for Qwen3.8's GDN state.

A publish scatters the bank's rows into the canonical slots. As eager torch indexing on a NeuronCore it costs work in
proportion to the whole state tensor, not to the rows it writes: measured on one trn2 core at Qwen3.8's per-rank
shapes, one row took 1.1 ms into a pool of 4 slots and 6.1 ms into one of 84 (the radix cache's pool at 4 rows). Here
the publish is one compiled graph per state tensor and row bucket whose scatter is an indirect-DMA NKI kernel that
writes the rows in place (``_rows_write``), so its cost follows the rows. The load, a gather from the canonical slots
into the small bank, stays torch indexing in a compiled graph of its own.

Every graph is compiled once per (tensor, bucket) when the pool is bound (runtime.warm_bank_moves) and takes its slot
and row ids through ``device_io.UploadRing`` buffers, as ``CompiledStateOps`` does. On the host (CPU mode, tests) the
operations are the reference's eager indexing.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import torch

_TILE = 128  # rows per indirect DMA (one per SBUF partition)
# the compiler pins the dynamic-DMA scratch at SBUF [0, 16 KB); the kernel's tiles start above it
_SBUF_BASE = 17 * 1024
_MAX_ROW_BYTES = 8192


def _kernel():
    import nki
    import nki.isa as nisa
    import nki.language as nl

    @nki.jit
    def _rows_write(pool, rows, row_index):
        """Write ``rows[128 * t + p]`` to row ``row_index[p, t]`` of ``pool`` viewed as ``[R, width]``, in place; an
        index outside ``[0, R)`` writes nothing. Returns ``(pool, done)``, ``done`` a ``[128, 1]`` zero token that
        completes after the writes."""
        parts, tiles = row_index.shape
        width = rows.shape[1]
        total = 1
        for size in pool.shape:
            total *= int(size)
        assert parts == _TILE and rows.shape[0] == _TILE * tiles  # noqa: S101
        assert total % width == 0  # noqa: S101
        view = pool.reshape((total // width, width))
        done = nl.ndarray((_TILE, 1), dtype=nl.int32, buffer=nl.shared_hbm)
        idx_bytes = (tiles * 4 + 63) // 64 * 64
        at_base = _SBUF_BASE + idx_bytes
        buf_base = at_base + 256
        idx = nl.ndarray((_TILE, tiles), dtype=nl.int32, buffer=nl.sbuf, name="q38w_idx", address=(0, _SBUF_BASE))
        nisa.dma_copy(dst=idx, src=row_index)
        n_buf = min(2, tiles)
        ats = [nl.ndarray((_TILE, 1), dtype=nl.int32, buffer=nl.sbuf, name=f"q38w_at{k}",
                          address=(0, at_base + 64 * k)) for k in range(n_buf)]
        bufs = [nl.ndarray((_TILE, width), dtype=rows.dtype, buffer=nl.sbuf, name=f"q38w_rows{k}",
                           address=(0, buf_base + _MAX_ROW_BYTES * k)) for k in range(n_buf)]
        for t in range(tiles):
            at, buf = ats[t % 2], bufs[t % 2]
            # the DMA index must come from an engine-written tile, not straight from a DMA load
            nisa.tensor_scalar(dst=at, data=idx[0:_TILE, t:t + 1], op0=nl.add, operand0=0)
            nisa.dma_copy(dst=buf, src=rows[nl.ds(t * _TILE, _TILE), :])
            nisa.dma_copy(dst=view.ap(pattern=[[width, _TILE], [1, width]], offset=0, vector_offset=at[0:_TILE, 0:1],
                                      indirect_dim=0),
                          src=buf, oob_mode=nisa.oob_mode.skip)
        zero = nl.ndarray((_TILE, 1), dtype=nl.int32, buffer=nl.sbuf, name="q38w_zero", address=(0, at_base + 128))
        nisa.tensor_scalar(dst=zero, data=idx[0:_TILE, 0:1], op0=nl.multiply, operand0=0)
        nisa.dma_copy(dst=done, src=zero)
        return pool, done

    return _rows_write


def row_width(per_slot: int, elem_bytes: int) -> int:
    """The widest divisor of a slot's elements per layer whose row fits one kernel row buffer."""
    for width in range(min(per_slot, _MAX_ROW_BYTES // elem_bytes), 0, -1):
        if per_slot % width == 0:
            return width
    raise ValueError(f"no row width for {per_slot} elements")


def publish_index(slots: Sequence[int], layers: int, slot_count: int, per: int) -> np.ndarray:
    """``[128, tiles]`` int32: the pool row of each source row ``(layer, i, j)`` (layer-major, then the batch row, then
    the slot's sub-row), tiled one index per partition; padding indexes past the pool, which the kernel skips."""
    s = np.asarray(slots, dtype=np.int64)
    rows = ((np.arange(layers)[:, None, None] * slot_count + s[None, :, None]) * per
            + np.arange(per)[None, None, :]).reshape(-1)
    tiles = -(-rows.size // _TILE)
    out = np.full(tiles * _TILE, layers * slot_count * per, dtype=np.int64)
    out[:rows.size] = rows
    return out.reshape(tiles, _TILE).T.astype(np.int32).copy()


def _publish_body(kernel, lnc: int, width: int):
    from sglang_neuron.backend.stack import wrap_nki

    def body(pool, work, rows, index):
        src = work.index_select(1, rows).reshape(-1, width)
        pad = index.shape[0] * index.shape[1] - src.shape[0]
        if pad:
            src = torch.nn.functional.pad(src, (0, 0, 0, pad))
        _, done = wrap_nki(kernel)[lnc](pool=pool, rows=src, row_index=index)
        return done

    return body


def _load_body(tensor, work, slots, rows):
    work.index_copy_(1, rows, tensor.index_select(1, slots))
    return work.reshape(-1)[-1:].clone()


class KernelBankOps:
    """``load_rows`` / ``publish_rows`` / ``drain`` of a working-bank pool (module docstring); WorkingBankOps adds the
    canonical-slot operations."""

    name = "qwen38 kernel rows"

    def __init__(self):
        self._graphs = {}
        self._rings = {}

    def _ring(self, key, shape, dtype, device):
        from sglang_neuron.backend import device_io

        ring = self._rings.get(key)
        if ring is None:
            ring = self._rings[key] = device_io.UploadRing(shape, dtype, device)
        return ring

    def _graph(self, key, build):
        graph = self._graphs.get(key)
        if graph is None:
            from sglang_neuron.mem_cache.state_ops import compile_op

            graph = self._graphs[key] = compile_op(build())
        return graph

    def load_rows(self, tensors: Dict[str, torch.Tensor], slots: Sequence[int], rows: Sequence[int]) -> None:
        for name, tensor in tensors.items():
            if name.startswith("work."):
                continue
            work = tensors["work." + name]
            if tensor.device.type != "neuron":
                work[:, list(rows)] = tensor[:, list(slots)]
                continue
            n = len(slots)
            graph = self._graph(("load", name, n), lambda: _load_body)
            ids = [self._ring(("load", name, n, k), (n,), torch.int64, tensor.device) for k in range(2)]
            args = [ring.put(torch.tensor(list(v), dtype=torch.int64)) for ring, v in zip(ids, (slots, rows))]
            graph(tensor, work, *args)
            for ring in ids:
                ring.used()

    def publish_rows(self, tensors: Dict[str, torch.Tensor], slots: Sequence[int], rows: Sequence[int]) -> None:
        for name, tensor in tensors.items():
            if name.startswith("work."):
                continue
            work = tensors["work." + name]
            if tensor.device.type != "neuron":
                tensor[:, list(slots)] = work[:, list(rows)]
                continue
            from sglang_neuron.utils.hardware_config import get_logical_nc_config

            layers, slot_count = tensor.shape[0], tensor.shape[1]
            per_slot = tensor[0, 0].numel()
            width = row_width(per_slot, tensor.element_size())
            n = len(slots)
            index = torch.from_numpy(publish_index(slots, layers, slot_count, per_slot // width))
            graph = self._graph(("publish", name, n),
                                lambda: _publish_body(_kernel(), get_logical_nc_config(), width))
            rows_ring = self._ring(("publish rows", name, n), (n,), torch.int64, tensor.device)
            index_ring = self._ring(("publish index", name, n), tuple(index.shape), torch.int32, tensor.device)
            graph(tensor, work, rows_ring.put(torch.tensor(list(rows), dtype=torch.int64)), index_ring.put(index))
            rows_ring.used()
            index_ring.used()

    def drain(self):
        return None
