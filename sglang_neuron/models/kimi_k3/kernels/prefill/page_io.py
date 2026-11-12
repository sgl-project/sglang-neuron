# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3 prefill cache page I/O: copy one page row with a dynamic offset.

The prefill reads and writes whole cache pages at a page index that is
tensor data (the block table). ``index_select`` and ``index_copy_`` on the
cache lower to one indirect DMA descriptor for each element on Neuron. These
kernels copy one contiguous run of a page row with one DMA, at the runtime
page index (``scalar_offset``):

* ``kimi_k3_page_read``: read the run of ``n`` elements from element
  ``col`` of page ``page``.
* ``kimi_k3_page_write``: write ``payload`` to the run. The kernel returns
  ``pool``, so the output aliases the input and the write is in place.
* ``kimi_k3_page_write_after``: the same write, with one more input
  ``after``. The kernel reads ``after`` and adds 0 times its value to the
  page offset, so the write depends on ``after`` and the values do not change.
  A deferred page write uses it to wait for the last reads of the pool.
* ``kimi_k3_rows_write_after``: write rows of the pool, viewed as rows of
  ``width``, at a runtime index for each row (one indirect DMA for each tile
  of 128 rows), after ``after``. An index out of range writes nothing.

``pool`` is the flat cache base ``[num_pages * row_elems]``, as the runner
binds it. Page ``p`` is the run ``pool[p * row_elems : (p + 1) * row_elems]``.
The caller passes the base itself, not a view, so the in-place write of
``kimi_k3_page_write`` aliases a graph input.

An FP8 MLA cache shares a bf16 base with the KDA states. Both kernels can
reinterpret the bytes of a bf16 ``pool`` as FP8 (``NkiTensor.view``). The
torch graph then has no dtype view: the neuron backend does not keep the
bytes of a torch dtype view that changes the element size and feeds an NKI
kernel. With the FP8 view, ``row_elems``, ``col`` and ``n`` count FP8
elements (bytes).

An fp32 KDA recurrent state (``kda_state_dtype`` fp32) is
also in the bf16 base. The kernels can reinterpret the bytes of a bf16
``pool`` as fp32 in the same way. With the fp32 view, ``row_elems``, ``col``
and ``n`` count fp32 elements (4 bytes).
"""

import nki
import nki.isa as nisa
import nki.language as nl


def _fp8_type():
    """Give the FP8 e4m3 type of the trace target.

    Trn3 (gen4) has the OCP ``float8_e4m3fn``. Trn2 (gen3) has the legacy
    ``float8_e4m3`` only. The prefill clamps the cache rows to +-240 on
    Trn2, so each byte has the same value in both types.
    """
    return nl.float8_e4m3 if nisa.get_nc_version() == nisa.nc_version.gen3 else nl.float8_e4m3fn


def _fp8_payload_view(pool, payload):
    """Give the view of a bf16 ``pool`` in the FP8 type of ``payload``."""
    assert payload.dtype in (nl.float8_e4m3fn, nl.float8_e4m3) and pool.dtype == nl.bfloat16, (  # noqa: S101
        "a payload in another dtype must be FP8 into a bf16 pool"
    )
    return pool.view(payload.dtype)


def _payload_view(pool, payload):
    """Give the view of a bf16 ``pool`` in the dtype of ``payload``: fp32 for an fp32 KDA state, else FP8."""
    if payload.dtype == nl.float32 and pool.dtype == nl.bfloat16:
        return pool.view(nl.float32)
    return _fp8_payload_view(pool, payload)


def _names():
    """Return a name factory that is unique in one kernel trace.

    A name depends only on the order of the buffers in the trace. It does not
    depend on the earlier traces of the process. Thus two processes that
    trace the same call get the same IR, also when one of them gets the
    trace from the NKI trace cache.
    """
    prefix = "kpio"
    count = [0]

    def name(tag):
        count[0] += 1
        return f"{prefix}_{tag}_{count[0]}"

    return name


# The largest DMA of a page run. neuronx-cc splits a larger DMA into pieces, and a reader then does not wait for
# the last piece (MEASURED on the served 1M prefill NEFF: a 147,456 B read split at 131,072 B).
PIECE_BYTES = 65536


def _elem_bytes(dtype) -> int:
    """Give the bytes of one element of an FP8, 8-bit integer, fp32 or bf16 tensor."""
    if dtype in (nl.float8_e4m3fn, nl.float8_e4m3, nl.uint8, nl.int8):
        return 1
    return 4 if dtype == nl.float32 else 2


def _pieces(n, dtype):
    """Give ``(start, count)`` for the DMAs of a run of ``n`` elements: at most ``PIECE_BYTES`` each."""
    step = max(1, PIECE_BYTES // _elem_bytes(dtype))
    return [(p0, min(step, n - p0)) for p0 in range(0, n, step)]


def _offset_sb(page, row_elems, nm):
    """Load the page index and return ``page * row_elems`` in SBUF."""
    idx = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf, name=nm("page"))
    nisa.dma_copy(dst=idx[0:1, 0:1], src=page[0:1, 0:1])
    off = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf, name=nm("offset"))
    nisa.tensor_scalar(dst=off, data=idx, op0=nl.multiply, operand0=row_elems)
    return off


@nki.jit
def kimi_k3_page_read(pool, page, row_elems: int, col: int, n: int, fp8: bool = False, fp32: bool = False):
    """Read ``n`` elements of one page row at a runtime page index.

    Args:
        pool (nl.NkiTensor): [num_pages * row_elems] HBM flat cache base.
        page (nl.NkiTensor): [1, 1] int32 HBM page index, in range.
        row_elems (int): The element count of one page.
        col (int): The first element of the run in the page.
        n (int): The element count of the run, ``col + n <= row_elems``.
        fp8 (bool): Read the bytes of a bf16 ``pool`` as FP8: ``float8_e4m3fn``
            on Trn3, ``float8_e4m3`` on Trn2. ``row_elems``, ``col`` and ``n``
            then count FP8 elements.
        fp32 (bool): Read the bytes of a bf16 ``pool`` as fp32 (an fp32 KDA
            state). ``row_elems``, ``col`` and ``n`` then count fp32 elements.

    Returns:
        out (nl.NkiTensor): [1, n] HBM, the elements
        ``pool[page * row_elems + col :][:n]``, in FP8 with ``fp8``, in fp32
        with ``fp32``.

    Example:
        >>> # out = kimi_k3_page_read(base, page, 73728, 0, 52608)
    """
    src = pool.view(_fp8_type()) if fp8 else pool
    if fp32:
        assert not fp8 and pool.dtype == nl.bfloat16, "an fp32 read needs a bf16 pool"  # noqa: S101
        src = pool.view(nl.float32)
    assert col + n <= row_elems, "the run must fit in one page"  # noqa: S101
    assert src.shape[0] % row_elems == 0, "the pool must hold whole pages"  # noqa: S101
    nm = _names()
    out = nl.ndarray((1, n), dtype=src.dtype, buffer=nl.shared_hbm, name=nm("out"))
    off = _offset_sb(page, row_elems, nm)
    for p0, m in _pieces(n, src.dtype):
        nisa.dma_copy(
            dst=out[0:1, p0 : p0 + m],
            src=src.ap(pattern=[[m, 1], [1, m]], offset=col + p0, scalar_offset=off[0:1, 0:1], indirect_dim=0),
        )
    return out


@nki.jit
def kimi_k3_page_write(pool, payload, page, row_elems: int, col: int):
    """Write a run of one page at a runtime page index, in place.

    Args:
        pool (nl.NkiTensor): [num_pages * row_elems] HBM flat cache base.
        payload (nl.NkiTensor): [1, n] HBM values, in the dtype of ``pool``,
            or FP8 (``float8_e4m3fn``, or ``float8_e4m3`` on Trn2) for a bf16
            ``pool``. The kernel then writes
            the FP8 bytes into the bytes of ``pool``, and ``row_elems`` and
            ``col`` count FP8 elements. An fp32 payload for a bf16 ``pool``
            (an fp32 KDA state) writes fp32 bytes, and ``row_elems`` and
            ``col`` count fp32 elements.
        page (nl.NkiTensor): [1, 1] int32 HBM page index, in range.
        row_elems (int): The element count of one page.
        col (int): The first element of the run, ``col + n <= row_elems``.

    Returns:
        pool (nl.NkiTensor): The input ``pool``, with the run written.

    Example:
        >>> # base = kimi_k3_page_write(base, payload, page, 73728, 0)
    """
    n = payload.shape[1]
    dst = pool
    if payload.dtype != pool.dtype:
        dst = _payload_view(pool, payload)
    assert col + n <= row_elems, "the run must fit in one page"  # noqa: S101
    assert dst.shape[0] % row_elems == 0, "the pool must hold whole pages"  # noqa: S101
    nm = _names()
    off = _offset_sb(page, row_elems, nm)
    for p0, m in _pieces(n, dst.dtype):
        nisa.dma_copy(
            dst=dst.ap(pattern=[[m, 1], [1, m]], offset=col + p0, scalar_offset=off[0:1, 0:1], indirect_dim=0),
            src=payload[0:1, p0 : p0 + m],
        )
    return pool


@nki.jit
def kimi_k3_page_write_after(pool, payload, page, after, row_elems: int, col: int):
    """Write a run of one page at a runtime page index, in place, after ``after``.

    The write is the same as ``kimi_k3_page_write``. The kernel also reads
    ``after`` and adds ``0 * after`` to the page offset, so the offset stays
    the same. The data edge orders the write after the op that gives
    ``after``. A stage that defers its page writes to its last layer gives an
    ``after`` from the output of that layer. Then the compiler sees that the
    write runs after the last reads of the pool in the subgraph, and it keeps
    the pool in place (no copy of the pool for the reads).

    Args:
        pool (nl.NkiTensor): [num_pages * row_elems] HBM flat cache base.
        payload (nl.NkiTensor): [1, n] HBM values, as for ``kimi_k3_page_write``
            (also fp32 for a bf16 ``pool``).
        page (nl.NkiTensor): [1, 1] int32 HBM page index, in range.
        after (nl.NkiTensor): [1, 1] int32 HBM value. Only its data edge counts.
        row_elems (int): The element count of one page.
        col (int): The first element of the run, ``col + n <= row_elems``.

    Returns:
        pool (nl.NkiTensor): The input ``pool``, with the run written.

    Example:
        >>> # base = kimi_k3_page_write_after(base, payload, page, after, 73728, 0)
    """
    n = payload.shape[1]
    dst = pool
    if payload.dtype != pool.dtype:
        dst = _payload_view(pool, payload)
    assert col + n <= row_elems, "the run must fit in one page"  # noqa: S101
    assert dst.shape[0] % row_elems == 0, "the pool must hold whole pages"  # noqa: S101
    nm = _names()
    off = _offset_sb(page, row_elems, nm)
    edge = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf, name=nm("after"))
    nisa.dma_copy(dst=edge[0:1, 0:1], src=after[0:1, 0:1])
    zero = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf, name=nm("zero"))
    nisa.tensor_scalar(dst=zero, data=edge, op0=nl.multiply, operand0=0)
    at = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf, name=nm("at"))
    nisa.tensor_tensor(dst=at, data1=off, data2=zero, op=nl.add)
    for p0, m in _pieces(n, dst.dtype):
        nisa.dma_copy(
            dst=dst.ap(pattern=[[m, 1], [1, m]], offset=col + p0, scalar_offset=at[0:1, 0:1], indirect_dim=0),
            src=payload[0:1, p0 : p0 + m],
        )
    return pool


ROW_TILE = 128
# The row kernel puts its SBUF tiles at fixed addresses from 17 KB: the
# compiler pins the dynamic DMA scratch at SBUF [0, 16 KB).
ROW_SBUF_BASE = 17 * 1024
ROW_MAX_BYTES = 8192


@nki.jit
def kimi_k3_rows_write_after(pool, rows, row_index, after, width: int):
    """Write rows of a flat cache base at runtime row indices, in place, after ``after``.

    The kernel views ``pool`` as ``[R, width]`` and writes row ``128 t + p``
    of ``rows`` to the row ``row_index[p, t]``, with one indirect DMA for each
    tile of 128 rows. An index outside ``[0, R)`` writes nothing, so a pad row
    keeps the value of the pool. The kernel adds 0 times ``after`` to each
    index, as ``kimi_k3_page_write_after`` does to the page offset, and to
    the offset of each row load: the loads and the writes depend on
    ``after``, and the indices do not change.

    Args:
        pool (nl.NkiTensor): [R * width] HBM flat cache base, bf16 or in
            the dtype of ``rows`` (for example a uint8 row pool).
        rows (nl.NkiTensor): [128 T, width] HBM rows in the dtype of
            ``pool``, or FP8 (``float8_e4m3fn``, or ``float8_e4m3`` on
            Trn2) for a bf16 ``pool``. The kernel then writes the FP8
            bytes into the bytes of ``pool``, and ``width`` counts FP8
            elements.
        row_index (nl.NkiTensor): [128, T] int32 HBM row index. Column ``t``
            holds the indices of the rows ``128 t`` to ``128 t + 127``.
        after (nl.NkiTensor): [1, 1] int32 HBM value. Only its data edge counts.
        width (int): The elements of one row, at most 8 KB.

    Returns:
        pool (nl.NkiTensor): The input ``pool``, with the rows written.

    Example:
        >>> # base = kimi_k3_rows_write_after(base, rows, index, after, 288)
    """
    parts, tiles = row_index.shape
    # Rows in the dtype of the pool move as they are; FP8 rows go into the bytes of a bf16 pool.
    dst = pool if rows.dtype == pool.dtype else _fp8_payload_view(pool, rows)
    assert dst.shape[0] % width == 0, "the pool must hold whole rows"  # noqa: S101
    assert parts == ROW_TILE and tuple(rows.shape) == (ROW_TILE * tiles, width), (  # noqa: S101
        "the rows must be [128 T, width] for a [128, T] index"
    )
    row_bytes = width * _elem_bytes(dst.dtype)
    assert row_bytes <= ROW_MAX_BYTES, "a row must have 8 KB or less"  # noqa: S101
    nm = _names()
    view = dst.reshape((dst.shape[0] // width, width))
    idx_bytes = (tiles * 4 + 63) // 64 * 64
    at_base = ROW_SBUF_BASE + idx_bytes + 128
    buf_base = at_base + 128
    idx = nl.ndarray(
        (ROW_TILE, tiles),
        dtype=nl.int32,
        buffer=nl.sbuf,
        name=nm("index"),
        address=(0, ROW_SBUF_BASE),
    )
    nisa.dma_copy(dst=idx, src=row_index)
    edge = nl.ndarray(
        (ROW_TILE, 1),
        dtype=nl.int32,
        buffer=nl.sbuf,
        name=nm("after"),
        address=(0, ROW_SBUF_BASE + idx_bytes),
    )
    nisa.dma_copy(dst=edge, src=after.ap(pattern=[[0, ROW_TILE], [1, 1]], offset=0))
    zero = nl.ndarray(
        (ROW_TILE, 1),
        dtype=nl.int32,
        buffer=nl.sbuf,
        name=nm("zero"),
        address=(0, ROW_SBUF_BASE + idx_bytes + 64),
    )
    nisa.tensor_scalar(dst=zero, data=edge, op0=nl.multiply, operand0=0)
    # Two index and row buffers alternate, so the load of the next tile overlaps the write of this one.
    ats = [
        nl.ndarray(
            (ROW_TILE, 1),
            dtype=nl.int32,
            buffer=nl.sbuf,
            name=nm("at"),
            address=(0, at_base + 64 * k),
        )
        for k in range(min(2, tiles))
    ]
    bufs = [
        nl.ndarray(
            (ROW_TILE, width),
            dtype=rows.dtype,
            buffer=nl.sbuf,
            name=nm("rows"),
            address=(0, buf_base + ROW_MAX_BYTES * k),
        )
        for k in range(min(2, tiles))
    ]
    for t in range(tiles):
        at, buf = ats[t % 2], bufs[t % 2]
        nisa.tensor_tensor(
            dst=at, data1=idx[0:ROW_TILE, t : t + 1], data2=zero, op=nl.add
        )
        # The load also waits for the edge: a value that a DMA read gives is then complete (offset 0).
        nisa.dma_copy(
            dst=buf,
            src=rows.ap(
                pattern=[[width, ROW_TILE], [1, width]],
                offset=t * ROW_TILE * width,
                scalar_offset=zero[0:1, 0:1],
                indirect_dim=0,
            ),
        )
        nisa.dma_copy(
            dst=view.ap(
                pattern=[[width, ROW_TILE], [1, width]],
                offset=0,
                vector_offset=at[0:ROW_TILE, 0:1],
                indirect_dim=0,
            ),
            src=buf,
            oob_mode=nisa.oob_mode.skip,
        )
    return pool


def _numel(shape) -> int:
    """Return the element count of a shape."""
    n = 1
    for size in shape:
        n *= int(size)
    return n


@nki.jit
def kimi_k3_page_pack(first, second):
    """Put two HBM tensors back to back in one run, in row-major order.

    The kernel copies the elements of ``first`` and then the elements of
    ``second`` into one ``[1, n1 + n2]`` HBM run, with one DMA for each
    tensor. The two dtypes are the same, so each DMA is a byte copy: no
    engine instruction reads or writes the values. The run equals
    ``torch.cat([first.reshape(-1), second.reshape(-1)])``.

    A KDA state page holds the conv state and then the recurrent state. The
    compiler lowers the torch ``cat`` of the two states to many small
    TensorE transposes and copies. This kernel gives the same payload with
    two DMAs.

    Args:
        first (nl.NkiTensor): HBM tensor of any shape, the first part.
        second (nl.NkiTensor): HBM tensor of any shape, in the dtype of
            ``first``, the second part.

    Returns:
        out (nl.NkiTensor): [1, n1 + n2] HBM, the elements of ``first`` and
        then of ``second``.

    Example:
        >>> # payload = kimi_k3_page_pack(conv_state, recurrent_state)  # [1, 52608]
    """
    assert first.dtype == second.dtype, "the two parts must have one dtype"  # noqa: S101
    n1 = _numel(first.shape)
    n2 = _numel(second.shape)
    nm = _names()
    out = nl.ndarray((1, n1 + n2), dtype=first.dtype, buffer=nl.shared_hbm, name=nm("pack"))
    nisa.dma_copy(dst=out[0:1, 0:n1], src=first.ap(pattern=[[n1, 1], [1, n1]], offset=0))
    nisa.dma_copy(dst=out[0:1, n1:n1 + n2], src=second.ap(pattern=[[n2, 1], [1, n2]], offset=0))
    return out
