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

"""Explicit SBUF addresses for the tiles that live across the loops of an MLA layer.

The MLA body of a long context loops over its key groups with ``nl.fori_loop``. The loops split the
kernel into basic blocks. Some neuronx-cc builds compute the SB liveness of a tile from its
accesses inside each basic block. The compiler can thus give the address of a tile that lives across a
block boundary to another tile. An explicit address alone does not protect a tile: the compiler treats
it as a fixed node with the same liveness, and it can still put another tile on it.

The compiler places only the tiles without an address. So a tile at an explicit address is safe when
every tile that is live at the same time across a block boundary also has an explicit address. With
the MLA loop path, the block gives explicit addresses to:

- its long-lived tiles: the all-ones tile, the norm and AttnRes weights, the stream, the bank, and the
  rank selectors of the feed-forward bodies;
- the identity matrices of the TensorE transposes;
- every SBUF tile of an MLA layer, from the attention mix to the residual add.

The other tiles of the kernel live inside one basic block, so the compiler places them, with no explicit address. The
compiler can also reuse the addresses of the MLA tiles when no MLA layer is live.

Above 8 sequences per replica, the MLA layer tiles do not fit the region beside the long-lived tiles
(NCC_INKI016), so the block gives explicit addresses only to the long-lived tiles and the identities, and
the compiler places the MLA layer tiles. This breaks the rule above: each build of that path must show
that no compiler-placed tile takes the bytes of a tile that is still live (an overlap check of the BIR
after SB allocation, on the compiler of the build).

A kernel with an explicit address is an "allocated kernel" for neuronx-cc, which then does not spill
SBUF. A compile that needs a spill fails with NCC_EGCA111 instead.
"""

import math
from contextlib import contextmanager

import nki.isa as nisa
import nki.language as nl
from nki.compiler._builtins import get_backend

from .utils.allocator import SbufManager, sizeinbytes
from .utils.kernel_assert import kernel_assert
from .utils.logging import get_logger

SB_DGE_SCRATCH = 16384
"""Bytes per partition of the DGE scratch at the bottom of SBUF. Explicit address 0 is SB byte 16384."""
SB_RESERVED_TAIL = 8
"""Bytes per partition that the compiler reserves at the top of SBUF."""
SB_PARTITION_BYTES = 262144
"""SBUF bytes per partition (trn3)."""
EXPLICIT_BYTES = SB_PARTITION_BYTES - SB_DGE_SCRATCH - SB_RESERVED_TAIL
"""Bytes per partition of the explicit address space."""
SB_PARTITION_BYTES_TRN2 = 229376
"""SBUF bytes per partition (trn2)."""
EXPLICIT_BYTES_TRN2 = SB_PARTITION_BYTES_TRN2 - SB_DGE_SCRATCH - SB_RESERVED_TAIL
"""Bytes per partition of the explicit address space on trn2."""
SB_TILE_ALIGN = 32
"""Byte alignment of every tile of an explicit SBUF region or zone."""
_ALIGN = SB_TILE_ALIGN
_PREFIX = "xsb_"


class ExplicitSbuf:
    """A region of explicit SBUF addresses: long-lived tiles on the heap, scoped tiles in stack frames.

    The heap grows down from the top of the region, and the stack grows up from the bottom. A frame
    frees its stack tiles when it closes.

    Example:
        >>> region = ExplicitSbuf()
        >>> ones = region.heap((128, 128), nl.float32, "ones")
        >>> with region.frame("L3"):
        ...     tile = region.stack((128, 8), nl.float32, "scores")
    """

    def __init__(self, limit: int | None = None):
        if limit is None:
            limit = EXPLICIT_BYTES if nisa.get_nc_version() >= nisa.nc_version.gen4 else EXPLICIT_BYTES_TRN2
        self.sbm = SbufManager(0, limit, get_logger("kimi_k3_sbuf"), use_auto_alloc=False)
        self.sbm.open_scope(name="kernel")
        self.count = 0
        self.heap_marks = []
        """For each open frame: the heap depth to restore at its end, or None to keep the heap."""
        self.ones = None
        """The all-ones tile of the region (``decode_block.region_ones``)."""
        self.streams = None
        """The two stream tiles of the region (``decode_block.region_streams``)."""
        self.identities = {}
        """The identity matrices of the region, by dtype (``install_identities``)."""
        self.null_rows = None
        """The int32 ``[128, 1]`` partition index column of the region (``install_null_rows``), or None."""
        self.spans = {}
        """kind -> the (stack top, heap bottom) of each span of that kind (``open_span``)."""
        self._open_spans = []

    def open_span(self, kind: str):
        """Start a span of ``kind``: until ``close_span``, record the highest stack top and the lowest heap bottom.

        Spans can nest. A span adds no instruction. ``free_ranges`` returns the bytes that no span of the kinds used.
        """
        sbm = self.sbm
        self._open_spans.append((kind, sbm.max_stack_usage, sbm.max_heap_usage))
        sbm.max_stack_usage = sbm.stack_curr_addr - sbm.lower_bound
        sbm.max_heap_usage = sbm.upper_bound - sbm.heap_curr_addr

    def close_span(self):
        """End the innermost span and record its range; the manager keeps its own maxima."""
        sbm = self.sbm
        kind, stack_max, heap_max = self._open_spans.pop()
        span_stack, span_heap = sbm.max_stack_usage, sbm.max_heap_usage
        self.spans.setdefault(kind, []).append((sbm.lower_bound + span_stack, sbm.upper_bound - span_heap))
        sbm.max_stack_usage = max(stack_max, span_stack)
        sbm.max_heap_usage = max(heap_max, span_heap)

    def free_ranges(self, *kinds) -> list:
        """Return ``[(start, size)]``: the explicit addresses (SB byte = start + ``SB_DGE_SCRATCH``) that no explicit
        tile took in any recorded span of ``kinds``, per partition. The compiler-placed tiles of a span also take bytes
        of these ranges, so a tile placed there must be explicit, and the compile must show that the rest still fits.

        Example:
            >>> region.free_ranges("kda_attention", "kda_ffn")  # the bytes free across the KDA layers so far
        """
        records = [record for kind in kinds for record in self.spans.get(kind, ())]
        if not records:
            return []
        start = max(top for top, _ in records)
        end = min(bottom for _, bottom in records)
        return [(start, end - start)] if end > start else []

    def _name(self, name) -> str:
        self.count += 1
        return f"{_PREFIX}{name or 't'}_{self.count}"

    def heap(self, shape, dtype, name=None):
        """Return an SBUF tile at a fixed address for the rest of the kernel."""
        return self.sbm.alloc_heap(tuple(shape), dtype, name=self._name(name), align=_ALIGN)

    def stack(self, shape, dtype, name=None):
        """Return an SBUF tile at an address of the innermost open frame."""
        return self.sbm.alloc_stack(tuple(shape), dtype, name=self._name(name), align=_ALIGN)

    def release_heap(self, tiles):
        """Return ``tiles`` (the newest heap tiles, in the order of their allocation) to the region before the end of
        their frame, so later stack tiles can take their bytes. Fail at trace time if another heap tile is newer."""
        for tile in reversed(list(tiles)):
            heap = self.sbm.heap
            kernel_assert(heap and heap[-1] is tile, "release_heap frees only the newest heap tiles, newest first")
            self.sbm.pop_heap()

    def open_frame(self, name: str = "", release_heap: bool = False):
        """Open a stack frame. Its tiles return to the region at ``close_frame``.

        With ``release_heap``, the heap tiles that the frame allocates also return at its end. Use it for
        a frame that allocates heap tiles only for its own work (``SbufTiles.tile(keep=True)``).
        """
        self.sbm.open_scope(name=name)
        self.heap_marks.append(len(self.sbm.heap) if release_heap else None)

    def close_frame(self):
        """Close the innermost frame and free its stack tiles (and its heap tiles with ``release_heap``)."""
        mark = self.heap_marks.pop()
        while mark is not None and len(self.sbm.heap) > mark:
            self.sbm.pop_heap()
        self.sbm.close_scope()

    @contextmanager
    def frame(self, name: str = "", release_heap: bool = False):
        """Open a stack frame for the duration of the context."""
        self.open_frame(name, release_heap)
        try:
            yield self
        finally:
            self.close_frame()


class SbufTiles:
    """The SBUF tile allocator that a body takes: explicit addresses of a region, or compiler placement.

    With a region, ``tile`` returns a tile at a stack address of the innermost open frame of the region
    (a heap address with ``heap=True`` or ``keep=True``), and ``scope`` opens a nested frame. Without a
    region, ``tile`` returns a tile that the compiler places, as ``nl.ndarray`` does, and ``scope`` does
    nothing.

    Example:
        >>> sb = SbufTiles()  # compiler placement
        >>> scores = sb.tile((96, 512), nl.float32)
        >>> with sb.scope("group"):
        ...     weights = sb.tile((96, 512), nl.bfloat16)
    """

    def __init__(self, region: ExplicitSbuf = None, heap: bool = False):
        self.region = region
        self.heap = heap

    @property
    def explicit(self) -> bool:
        """True when the tiles take explicit addresses."""
        return self.region is not None

    def tile(self, shape, dtype, name=None, keep: bool = False):
        """Return an SBUF tile of ``shape`` and ``dtype``.

        ``keep=True`` puts the tile on the heap of the region, so a nested scope does not free it. The
        tile then returns to the region at the end of the innermost frame with ``release_heap``. Use it
        for a tile that lives across nested scopes, so the stack of the scopes can free the other tiles.
        """
        if self.region is None:
            return nl.ndarray(shape, dtype=dtype, buffer=nl.sbuf)
        if self.heap or keep:
            return self.region.heap(shape, dtype, name)
        return self.region.stack(shape, dtype, name)

    def release(self, tiles):
        """Free heap tiles of ``tile(keep=True)`` before the end of their frame (``ExplicitSbuf.release_heap``; the
        newest heap tiles, in the order of their allocation). Without a region, or with ``heap``, it does nothing."""
        if self.region is not None and not self.heap:
            self.region.release_heap(tiles)

    def open_scope(self, name: str = ""):
        """Open a nested frame (with a region). ``close_scope`` frees the stack tiles of the frame."""
        if self.region is not None and not self.heap:
            self.region.open_frame(name)

    def close_scope(self):
        """Close the frame of ``open_scope`` (with a region)."""
        if self.region is not None and not self.heap:
            self.region.close_frame()

    @contextmanager
    def scope(self, name: str = ""):
        """Free the stack tiles that the context allocates at its end (with a region).

        Use it only for tiles that are dead at the end of the context.
        """
        if self.region is None or self.heap:
            yield self
            return
        with self.region.frame(name):
            yield self


class SbufZone:
    """A block of explicit SBUF bytes that a region reserves on its heap, for tiles that live from the first frame that
    uses them to the end of the kernel.

    A tile that a frame puts on the region heap returns to the region at the end of that frame (``frame`` with
    ``release_heap``), so a tile that later frames still read must not take a heap address in that frame. A zone
    reserves its bytes before the frames, and ``tile`` gives each tile the next 32-byte aligned bytes of the zone. The
    trace fails if a tile does not fit. ``scope`` does nothing: zone tiles are never freed.

    Example:
        >>> zone = SbufZone(region, 2048, "mla_pack_shared")
        >>> with region.frame("L3", release_heap=True):
        ...     ti = zone.tile((128, 33), nl.uint16, "pack_ti_0")  # still valid in the frames after L3
    """

    explicit = True
    heap = True

    def __init__(self, region: ExplicitSbuf, nbytes: int, name: str):
        self.region = region
        self.size = nbytes
        region.heap((128, max(nbytes, _ALIGN)), nl.uint8, name)
        self.start = region.sbm.heap_curr_addr
        self.next = self.start

    def tile(self, shape, dtype, name=None, keep: bool = False):
        """Return a tile at the next 32-byte aligned bytes of the zone (``keep`` has no effect: zone tiles stay)."""
        size = math.prod(shape[1:]) * sizeinbytes(dtype)
        start = -(-self.next // _ALIGN) * _ALIGN
        kernel_assert(
            start + size <= self.start + self.size,
            f"SBUF zone of {self.size} B is full: a tile of {size} B per partition does not fit",
        )
        self.next = start + size
        return nl.ndarray(tuple(shape), dtype=dtype, buffer=nl.sbuf, name=self.region._name(name), address=(0, start))

    def release(self, tiles):
        """Do nothing: zone tiles stay for the rest of the kernel."""

    def open_scope(self, name: str = ""):
        """Do nothing (zone tiles are never freed)."""

    def close_scope(self):
        """Do nothing (zone tiles are never freed)."""

    @contextmanager
    def scope(self, name: str = ""):
        """Do nothing (zone tiles are never freed)."""
        yield self


AUTO_SBUF = SbufTiles()
"""The allocator of a body that lets the compiler place its SBUF tiles."""


def install_identities(region: ExplicitSbuf, dtypes=(nl.bfloat16, nl.float32, nl.float16)):
    """Put the identity matrices of the TensorE transposes at explicit heap addresses.

    ``nisa.nc_transpose`` and ``nl.shared_identity_matrix`` take the identity of a dtype from a cache
    of the kernel. The function fills that cache with heap tiles, so no transpose creates an identity
    that the compiler places. Call it before the first transpose of the kernel.

    The dtypes cover every transpose of the decode: BF16 for the keys and the weights, FP32 for the KDA
    state, and FP16 for the MoE activation of the FP32 accumulation. The compiler places the identity of a
    dtype that this function leaves out, and on the loop path the region leaves so little room that it puts
    that identity past the end of SBUF: the 93-layer 1M form fails with NCC_INLA001 "out of bound
    {_nki_identity_float16}@SB<0,262112>(128x256)".
    """
    cache = get_backend().identity_matrix_cache()
    for dtype in dtypes:
        kernel_assert(dtype not in cache, f"The {dtype} identity exists already. Install it first.")
        tile = region.heap((128, 128), dtype, f"identity_{dtype}")
        nl.shared_identity_matrix(128, dtype=dtype, dst=tile)
        cache[dtype] = tile
        region.identities[dtype] = tile


def install_null_rows(region: ExplicitSbuf):
    """Put the int32 ``[128, 1]`` column of the partition indices (row p holds p) at an explicit heap address for the
    rest of the kernel, in ``region.null_rows``. The MLA cache write takes its null-page rows from it
    (``mla_block_attention`` ``batched_cache_write``), so no layer builds the column again. Call it at the kernel level,
    outside any frame that frees its heap tiles."""
    kernel_assert(region.null_rows is None, "The null rows column exists already.")
    region.null_rows = region.heap((128, 1), nl.int32, "null_rows")
    nisa.iota(dst=region.null_rows, pattern=[[0, 1]], offset=0, channel_multiplier=1)


def explicit_ones(region: ExplicitSbuf, dtype=nl.float32):
    """Return an all-ones ``[128, 128]`` heap tile for cross-partition sums."""
    ones = region.heap((128, 128), dtype, "ones")
    nisa.memset(dst=ones, value=1.0)
    return ones


def check_identities(region: ExplicitSbuf):
    """Fail the trace if a transpose made an identity matrix outside the explicit SBUF region.

    ``install_identities`` fills the identity cache of the kernel before the first transpose. A transpose
    of another dtype adds its own identity to that cache, and the compiler places that tile above the
    region. The loop path fills SBUF, so the tile goes past the end of a partition, and the bytes land in
    the DMA descriptor scratch of another engine. The kernel then gives a semaphore error on some cores.
    Call this function at the end of the kernel, after the last transpose.

    Example:
        >>> region = ExplicitSbuf()
        >>> install_identities(region)
        >>> check_identities(region)  # after the last transpose of the kernel
    """
    cache = get_backend().identity_matrix_cache()
    outside = sorted(str(dtype) for dtype in cache if region.identities.get(dtype) is not cache[dtype])
    kernel_assert(
        not outside,
        f"The identity matrix of {outside} is outside the explicit SBUF region. "
        f"Add the dtype to the install_identities call of the kernel.",
    )
