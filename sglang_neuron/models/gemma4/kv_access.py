# SPDX-License-Identifier: Apache-2.0
"""Paged KV access for Gemma 4 with indirect DMA: an in-place row write and a block gather.

Each layer's K and V caches are separate pool tensors (``[pages, kv_heads, page, head_dim]`` as the pool allocates
them, the same for every layer). A global layer stores its wider rows in the same bytes: the kernels view a pool tensor
as ``[R, width]`` inside the kernel, so the graph never holds two differently shaped views of one buffer.

The kernels follow the vLLM Neuron Gemma 4 contrib (kv_write.py), where torch scatters into the cache lowered to work
proportional to the whole cache. Here a single decode step took the same time with a 2.5 times larger pool.
"""

import torch
import torch.nn.functional as F

import nki
import nki.isa as nisa
import nki.language as nl

from sglang_neuron.backend.stack import wrap_nki
from sglang_neuron.utils.hardware_config import get_logical_nc_config
from sglang_neuron.utils.neuron_utils import can_run_kernel

_TILE = 128  # rows per indirect DMA (one per SBUF partition)
# the compiler pins the dynamic-DMA scratch at SBUF [0, 16 KB); user tiles start above it
_SBUF_BASE = 17 * 1024
_MAX_ROW_BYTES = 8192


def _elem_bytes(dtype) -> int:
    return 1 if dtype in (nl.float8_e4m3fn, nl.float8_e4m3, nl.uint8, nl.int8) else 2


@nki.jit
def _rows_write_kernel(pool, rows, row_index):
    """Write ``rows[128 * t + p]`` to row ``row_index[p, t]`` of ``pool`` viewed as ``[R, width]``; an index outside
    ``[0, R)`` writes nothing. Returns ``(pool, done)``, ``done`` a ``[128, 1]`` zero token that completes after the
    writes."""
    parts, tiles = row_index.shape
    width = rows.shape[1]
    total = 1
    for size in pool.shape:
        total *= int(size)
    assert parts == _TILE and rows.shape[0] == _TILE * tiles  # noqa: S101
    assert total % width == 0  # noqa: S101
    assert width * _elem_bytes(rows.dtype) <= _MAX_ROW_BYTES  # noqa: S101

    view = pool.reshape((total // width, width))
    done = nl.ndarray((_TILE, 1), dtype=nl.int32, buffer=nl.shared_hbm)
    idx_bytes = (tiles * 4 + 63) // 64 * 64
    at_base = _SBUF_BASE + idx_bytes
    buf_base = at_base + 256

    idx = nl.ndarray((_TILE, tiles), dtype=nl.int32, buffer=nl.sbuf, name="g4w_idx", address=(0, _SBUF_BASE))
    nisa.dma_copy(dst=idx, src=row_index)
    n_buf = min(2, tiles)
    ats = [nl.ndarray((_TILE, 1), dtype=nl.int32, buffer=nl.sbuf, name=f"g4w_at{k}", address=(0, at_base + 64 * k))
           for k in range(n_buf)]
    bufs = [nl.ndarray((_TILE, width), dtype=rows.dtype, buffer=nl.sbuf, name=f"g4w_rows{k}",
                       address=(0, buf_base + _MAX_ROW_BYTES * k)) for k in range(n_buf)]
    for t in range(tiles):
        at, buf = ats[t % 2], bufs[t % 2]
        # the DMA index must come from an engine-written tile, not straight from a DMA load
        nisa.tensor_scalar(dst=at, data=idx[0:_TILE, t:t + 1], op0=nl.add, operand0=0)
        nisa.dma_copy(dst=buf, src=rows[nl.ds(t * _TILE, _TILE), :])
        nisa.dma_copy(dst=view.ap(pattern=[[width, _TILE], [1, width]], offset=0, vector_offset=at[0:_TILE, 0:1],
                                  indirect_dim=0),
                      src=buf, oob_mode=nisa.oob_mode.skip)
    zero = nl.ndarray((_TILE, 1), dtype=nl.int32, buffer=nl.sbuf, name="g4w_zero", address=(0, at_base + 128))
    nisa.tensor_scalar(dst=zero, data=idx[0:_TILE, 0:1], op0=nl.multiply, operand0=0)
    nisa.dma_copy(dst=done, src=zero)
    return pool, done


@nki.jit
def _rows_gather_kernel(pool, row_index, after, width: int):
    """``out[128 * t + p] = pool_view[row_index[p, t]]`` with ``pool`` viewed as ``[R, width]``. Read-only on
    ``pool``; indices must be in range. ``after`` is a ``[128, 1]`` int32 token added (times 0) to every index, so the
    gather starts after the kernel that produced it."""
    parts, tiles = row_index.shape
    total = 1
    for size in pool.shape:
        total *= int(size)
    assert parts == _TILE and total % width == 0  # noqa: S101
    assert width * _elem_bytes(pool.dtype) <= _MAX_ROW_BYTES  # noqa: S101

    view = pool.reshape((total // width, width))
    out = nl.ndarray((_TILE * tiles, width), dtype=pool.dtype, buffer=nl.shared_hbm)
    idx_bytes = (tiles * 4 + 63) // 64 * 64
    at_base = _SBUF_BASE + idx_bytes
    buf_base = at_base + 256

    idx = nl.ndarray((_TILE, tiles), dtype=nl.int32, buffer=nl.sbuf, name="g4g_idx", address=(0, _SBUF_BASE))
    nisa.dma_copy(dst=idx, src=row_index)
    edge = nl.ndarray((_TILE, 1), dtype=nl.int32, buffer=nl.sbuf, name="g4g_edge", address=(0, at_base + 128))
    nisa.dma_copy(dst=edge, src=after)
    zero = nl.ndarray((_TILE, 1), dtype=nl.int32, buffer=nl.sbuf, name="g4g_zero", address=(0, at_base + 192))
    nisa.tensor_scalar(dst=zero, data=edge, op0=nl.multiply, operand0=0)
    n_buf = min(2, tiles)
    ats = [nl.ndarray((_TILE, 1), dtype=nl.int32, buffer=nl.sbuf, name=f"g4g_at{k}", address=(0, at_base + 64 * k))
           for k in range(n_buf)]
    bufs = [nl.ndarray((_TILE, width), dtype=pool.dtype, buffer=nl.sbuf, name=f"g4g_rows{k}",
                       address=(0, buf_base + _MAX_ROW_BYTES * k)) for k in range(n_buf)]
    for t in range(tiles):
        at, buf = ats[t % 2], bufs[t % 2]
        nisa.tensor_tensor(dst=at, data1=idx[0:_TILE, t:t + 1], data2=zero, op=nl.add)
        nisa.dma_copy(dst=buf, src=view.ap(pattern=[[width, _TILE], [1, width]], offset=0,
                                           vector_offset=at[0:_TILE, 0:1], indirect_dim=0))
        nisa.dma_copy(dst=out[nl.ds(t * _TILE, _TILE), :], src=buf)
    return out


def no_token(like: torch.Tensor) -> torch.Tensor:
    return torch.zeros((_TILE, 1), dtype=torch.int32, device=like.device)


def _tiled(index: torch.Tensor, fill: int) -> torch.Tensor:
    n = index.shape[0]
    tiles = -(-n // _TILE)
    return F.pad(index, (0, tiles * _TILE - n), value=fill).to(torch.int32).view(tiles, _TILE).t().contiguous()


def token_rows(page_idx: torch.Tensor, offset: torch.Tensor, kv_heads: int, heads: int, page: int) -> torch.Tensor:
    """Row number (in ``[pages, heads, page]`` rows) of each (kv head, token), head-major: ``h * T + t``."""
    h = torch.arange(kv_heads, dtype=page_idx.dtype, device=page_idx.device)
    return ((page_idx.view(1, -1) * heads + h.view(-1, 1)) * page + offset.view(1, -1)).reshape(-1)


def write_rows(pool: torch.Tensor, rows: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Write ``rows`` ``[N, width]`` at row ``index`` of ``pool`` viewed as ``[R, width]``, in place; returns the
    ordering token. An index of ``R`` or more is skipped."""
    width = rows.shape[1]
    if not can_run_kernel(pool):
        # the runtime pads with slot 0 (SGLang's reserved page), so every index is in range
        pool.view(-1, width).index_copy_(0, index, rows)
        return no_token(pool)
    n = rows.shape[0]
    tiled = _tiled(index, pool.numel() // width)
    _, token = wrap_nki(_rows_write_kernel)[get_logical_nc_config()](
        pool=pool, rows=F.pad(rows, (0, 0, 0, tiled.numel() - n)), row_index=tiled)
    return token


def gather_pages(pool: torch.Tensor, page_ids: torch.Tensor, dims: tuple, after: torch.Tensor | None = None):
    """``[len(page_ids), heads, page, head_dim]``: the pages ``page_ids`` of ``pool`` viewed as ``dims`` =
    ``(pages, heads, page, head_dim)``, read after the write that produced ``after``."""
    pages, heads, page, head_dim = dims
    nb = page_ids.shape[0]
    if not can_run_kernel(pool):
        return torch.index_select(pool.view(pages, heads, page, head_dim), 0, page_ids)
    page_elems = heads * page * head_dim
    width = page_elems
    while width * pool.element_size() > _MAX_ROW_BYTES:
        width //= 2
    per_page = page_elems // width
    rows = (page_ids.view(-1, 1) * per_page
            + torch.arange(per_page, dtype=page_ids.dtype, device=page_ids.device).view(1, -1)).reshape(-1)
    out = wrap_nki(_rows_gather_kernel)[get_logical_nc_config()](
        pool=pool, row_index=_tiled(rows, 0), after=no_token(pool) if after is None else after, width=width)
    return out[:rows.shape[0]].view(nb, heads, page, head_dim)
