"""Reads and writes of the MLA row pool ``[num_blocks, STRIPE, width]`` uint8.

The torch forms run on any device; the kernel form writes each tile of 128 rows with one indirect DMA. A row whose
slot is negative or whose ``keep`` is False writes nothing.
"""

from __future__ import annotations

import torch

from sglang_neuron.models.kimi_k3.prefill import kernel_imports
from sglang_neuron.models.kimi_k3.prefill.fp8_rows import FP8_DTYPE, STRIPE

ROW_TILE = 128
_GRID = 1


def gather_pages(pool: torch.Tensor, block_table: torch.Tensor) -> torch.Tensor:
    """Give the logical rank-local view ``[NBT * STRIPE, width]`` of the pool; a negative entry reads block 0."""
    table = block_table.reshape(-1).to(torch.int64).clamp_min(0)
    return pool.index_select(0, table).reshape(-1, pool.shape[-1])


def write_rows(pool: torch.Tensor, rows: torch.Tensor, slots: torch.Tensor, keep: torch.Tensor) -> None:
    """Write ``rows [N, width]`` uint8 at the flat pool rows ``slots [N]``, in place, where ``keep`` and ``slots >= 0``.

    Each skipped row writes back the bytes that its (clamped) slot holds, so a skip changes no byte.
    """
    width = pool.shape[-1]
    flat = pool.view(-1)
    slots = slots.reshape(-1).to(torch.int64)
    keep = keep.reshape(-1) & (slots >= 0)
    safe = torch.where(keep, slots, torch.zeros_like(slots))
    elems = (safe[:, None] * width + torch.arange(width, device=pool.device)[None, :]).reshape(-1)
    old = flat.index_select(0, elems).view(-1, width)
    flat.index_copy_(0, elems, torch.where(keep[:, None], rows, old).reshape(-1))


def row_index_tile(slots: torch.Tensor, keep: torch.Tensor, rows: torch.Tensor, num_rows: int):
    """Give the ``[128, n]`` int32 index tile and the ``[128 n, width]`` rows of one kernel row write.

    Column ``t`` holds the slots of rows ``128 t .. 128 t + 127``. A skipped row gets ``num_rows``, an index the
    kernel does not write.
    """
    index = slots.reshape(-1).to(torch.int64)
    valid = keep.reshape(-1) & (index >= 0) & (index < num_rows)
    safe = torch.where(valid, index, torch.full_like(index, num_rows)).to(torch.int32)
    pad = -index.shape[0] % ROW_TILE
    if pad:
        safe = torch.cat((safe, safe.new_full((pad,), num_rows)))
        rows = torch.cat((rows, rows.new_zeros(pad, rows.shape[1])))
    return safe.reshape(-1, ROW_TILE).transpose(0, 1).contiguous(), rows.contiguous()


_ROWS_SHIM = None
_SLOT_READ_SHIM = None
_SLOT_WRITE_SHIM = None


def prepare_rows_kernel() -> None:
    """Build the row writes' trace shims (bytes, FP8 rows); call it outside any traced forward (``prepare_kernel``)."""
    global _ROWS_SHIM, _FP8_ROWS_SHIM
    if _ROWS_SHIM is None or _FP8_ROWS_SHIM is None:
        _ROWS_SHIM, _FP8_ROWS_SHIM = _rows_write_shim(), _fp8_rows_write_shim()


def prepare_slot_kernels() -> None:
    """Build the trace shims of the state slot read and write; call it outside any traced forward."""
    global _SLOT_READ_SHIM, _SLOT_WRITE_SHIM
    if _SLOT_READ_SHIM is None:
        _SLOT_READ_SHIM = _slot_read_shim()
    if _SLOT_WRITE_SHIM is None:
        _SLOT_WRITE_SHIM = _slot_write_shim()


def _slot_write_shim():
    """One slot of a state pool ``[slots, *shape]`` written in place with one DMA at the runtime slot index; the shim
    flattens the pool inside the kernel trace and returns the pool it was given, so the write aliases the graph
    input."""
    import nki

    kernel = kernel_imports.page_write()

    @nki.jit(mode="trace")
    def kimi_k3_slot_write(pool, payload, slot, slot_elems: int):
        kernel(pool.reshape((pool.shape[0] * slot_elems,)), payload, slot, slot_elems, 0)
        return pool

    return kimi_k3_slot_write


def write_slot_kernel(pool: torch.Tensor, slot: torch.Tensor, value: torch.Tensor) -> None:
    """``pool[slot] = value`` in place for a state pool ``[slots, *shape]`` and a runtime ``slot`` (``[1]`` int), by
    the page write kernel: one contiguous run at the slot's offset."""
    from sglang_neuron.backend import stack

    if _SLOT_WRITE_SHIM is None:
        raise kernel_imports.KernelUnavailable("The slot write shim is not built. Call page_io.prepare_slot_kernels() "
                                               "before the trace.")
    elems = pool[0].numel()
    payload = value.reshape(1, elems).to(pool.dtype)
    stack.wrap_nki(_SLOT_WRITE_SHIM)[_GRID](pool, payload, slot.reshape(1, 1).to(torch.int32),
                                            slot_elems=int(elems))


def _slot_read_shim():
    """One slot of a state pool ``[slots, *shape]`` read with one DMA at the runtime slot index; the flat view the
    kernel takes is made inside the kernel trace. A torch ``index_select`` of the slot is one indirect DMA descriptor
    an element, which neuronx-cc may fold into the DMAs of the kernel that reads the slot."""
    import nki

    kernel = kernel_imports.page_read()

    @nki.jit(mode="trace")
    def kimi_k3_slot_read(pool, slot, slot_elems: int):
        return kernel(pool.reshape((pool.shape[0] * slot_elems,)), slot, slot_elems, 0, slot_elems)

    return kimi_k3_slot_read


def read_slot_kernel(pool: torch.Tensor, slot: torch.Tensor) -> torch.Tensor:
    """``pool[slot]`` of a state pool ``[slots, *shape]`` at a runtime ``slot`` (``[1]`` int), by the page read
    kernel: the slot's contiguous run."""
    from sglang_neuron.backend import stack

    if _SLOT_READ_SHIM is None:
        raise kernel_imports.KernelUnavailable("The slot read shim is not built. Call "
                                               "page_io.prepare_slot_kernels() before the trace.")
    elems = pool[0].numel()
    out = stack.wrap_nki(_SLOT_READ_SHIM)[_GRID](pool, slot.reshape(1, 1).to(torch.int32), slot_elems=int(elems))
    return out.reshape(pool.shape[1:])


def _rows_write_shim():
    """The row write kernel on the block pool itself: the flat view the kernel takes is made inside the kernel trace,
    and the shim returns the pool it was given. A torch view of the pool between the graph input and the kernel
    makes neuronx-cc keep the written pool as a new value (a whole-pool copy) instead of an in-place update."""
    import nki

    kernel = kernel_imports.rows_write_after()

    @nki.jit(mode="trace")
    def kimi_k3_rows_write_blocks(pool, rows, row_index, after, width: int):
        kernel(pool.reshape((pool.shape[0] * pool.shape[1] * pool.shape[2],)), rows, row_index, after, width=width)
        return pool

    return kimi_k3_rows_write_blocks


def write_rows_kernel(
    pool: torch.Tensor, rows: torch.Tensor, slots: torch.Tensor, keep: torch.Tensor, after: torch.Tensor
) -> torch.Tensor:
    """Kernel form of ``write_rows``: the writes wait for ``after`` (``[1, 1]`` int32 data edge, value unused).

    The pool and the rows are uint8, so the kernel moves bytes and the graph has no dtype view.
    """
    from sglang_neuron.backend import stack

    width = pool.shape[-1]
    if pool.shape[-2] != STRIPE:
        raise ValueError(f"The MLA pool has blocks of {pool.shape[-2]} rows, not {STRIPE}.")
    if _ROWS_SHIM is None:
        raise kernel_imports.KernelUnavailable("The row write shim is not built. Call page_io.prepare_rows_kernel() "
                                               "(MlaAttention.prepare_kernel) before the trace.")
    index, rows = row_index_tile(slots, keep, rows, pool.numel() // width)
    return stack.wrap_nki(_ROWS_SHIM)[_GRID](pool, rows, index, after.reshape(1, 1).to(torch.int32), width=int(width))


def kernel_row_writer(pool: torch.Tensor, slots: torch.Tensor, rows: torch.Tensor) -> None:
    """The drafter row writer of the kernel backend (``DraftContext.write(..., writer=...)``): FP8 rows
    ``[T, width]`` (``float8_e4m3fn``) at the local slots ``slots [T]`` of the uint8 ``pool`` in place; a negative
    slot writes nothing. The rows reach the kernel as FP8 and the kernel trace views the pool as FP8
    (``_fp8_rows_write_shim``), so the graph holds no dtype view of the rows."""
    from sglang_neuron.backend import stack

    width = pool.shape[-1]
    if rows.dtype != FP8_DTYPE or pool.dtype != torch.uint8 or pool.shape[-2] != STRIPE:
        raise ValueError(f"The FP8 row write takes {FP8_DTYPE} rows into a uint8 pool of {STRIPE}-row blocks, not "
                         f"{rows.dtype} rows into a {pool.dtype} pool of {tuple(pool.shape)}.")
    if _FP8_ROWS_SHIM is None:
        raise kernel_imports.KernelUnavailable("The FP8 row write shim is not built. Call "
                                               "page_io.prepare_rows_kernel() before the trace.")
    after = torch.zeros((1, 1), dtype=torch.int32, device=rows.device)
    index, rows = row_index_tile(slots, slots.reshape(-1) >= 0, rows, pool.numel() // width)
    stack.wrap_nki(_FP8_ROWS_SHIM)[_GRID](pool, rows, index, after, width=int(width))


_FP8_ROWS_SHIM = None


def _fp8_rows_write_shim():
    """The row write kernel of FP8 rows into a uint8 block pool: the trace views the pool as the rows' FP8 type and
    flattens it, so the kernel moves the FP8 bytes as they are (rows in the dtype of the pool), and returns the pool
    it was given. A torch uint8 view of the rows is not used: the device compile lowers a uint8 view of a cast FP8
    value with no other consumer as a value conversion to uint8."""
    import nki

    kernel = kernel_imports.rows_write_after()

    @nki.jit(mode="trace")
    def kimi_k3_fp8_rows_write_blocks(pool, rows, row_index, after, width: int):
        flat = pool.view(rows.dtype).reshape((pool.shape[0] * pool.shape[1] * pool.shape[2],))
        kernel(flat, rows, row_index, after, width=width)
        return pool

    return kimi_k3_fp8_rows_write_blocks
