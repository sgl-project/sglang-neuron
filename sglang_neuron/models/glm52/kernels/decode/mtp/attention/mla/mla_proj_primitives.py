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

"""Primitives shared by the GLM-5.2 MLA decode projection kernels: linear, RMSNorm, RoPE, transpose."""

import nki.isa as nisa
import nki.language as nl

from ....attention.mla.mla_decode_constants import F_MAX, P_MAX
from ....utils.kernel_assert import kernel_assert
from ....utils.kernel_helpers import div_ceil
from ....utils.tiled_range import TiledRange

PSUM_BANKS = 8
"""PSUM banks available to one kernel. A ``[P_MAX, F_MAX]`` fp32 tile occupies exactly one."""

BATCH_LOAD_MAX_N = 256
"""
Widest weight (columns) that ``linear`` loads whole with one strided DMA instead of one DMA per k-tile.

A per-k-tile load of a narrow weight moves ``P_MAX`` runs of at most 512 B, so it is descriptor-bound
(far below the bandwidth of the wide weights), and each load adds a semaphore the k-tile's matmuls
wait on. One access-pattern DMA into ``[P_MAX, K / P_MAX, N]`` removes that issue and dependency chain,
and the tile stays small (<= 14 KB per partition at these shapes). Wide weights keep per-tile loads:
their DMAs are already efficient, and loading tile by tile overlaps the k-loop with the transfer.
"""


def pe_column_band(num_tokens: int) -> int:
    """Narrowest PE column tile (32, 64 or 128) that holds a ``num_tokens``-wide stationary."""
    for band in (P_MAX // 4, P_MAX // 2, P_MAX):
        if num_tokens <= band:
            return band
    kernel_assert(False, f"num_tokens={num_tokens} exceeds the {P_MAX}-column PE array")
    return P_MAX


def linear(
    stationary_tiles_sbuf: nl.NkiTensor,
    w_hbm: nl.NkiTensor,
    P: int,
    stationary_fn=None,
    out_hbm: nl.NkiTensor | None = None,
    out_col0: int = 0,
    banded: bool = False,
    evict_engine=None,
) -> nl.NkiTensor:
    """
    Computes ``y[P, N] = x[P, K] @ w_hbm[K, N]`` from an already K-tiled stationary operand.

    The stationary is only ``P`` tokens wide, so a plain matmul would drive only ``P`` of the PE array's 128
    columns. The array is split into ``P_MAX / band`` column bands (``band`` = 32 at ``P <= 32``); band
    ``b`` holds its own copy of the stationary (``tile_position=(0, b * band)``), accumulates its chain in
    PSUM partitions ``[b * band, b * band + P)``, and bands run concurrently. The chains are the
    ``F_MAX``-wide n tiles; with fewer n tiles than bands (``wq_a``: N = 512) each is split into ``subs``
    column ranges. Every output element keeps its fp32 K accumulation order, so the result is
    bit-identical to a single-band walk. Bands other than 0 are moved down to rows ``[0, P)`` by one
    SBUF->SBUF DMA per chain, since only a DMA crosses partitions.

    Every chain holds its PSUM region across the whole K loop, so N is bounded by ``ceil(chains / bands)
    <= PSUM_BANKS``; wider callers tile N themselves (see ``mla_decode_out_proj``).

    Weights up to ``BATCH_LOAD_MAX_N`` columns are loaded whole; wider ones one k-tile at a time.

    Optional, all off by default:
        stationary_fn: ``stationary_fn(k_index)`` returns k-tile ``k_index``'s ``[k_size, P]`` stationary
            from any other packing.
        out_hbm, out_col0: store every chain from the partitions it was evicted on straight to
            ``out_hbm[0:P, out_col0 + c0 : out_col0 + c1]``, alternating the Sync and ACT HWDGE queues,
            with no band DMA and no second SBUF -> HBM hop.
        banded: return the banded layout ``[P_MAX, n_banks * F_MAX]`` (chain ``c`` at partitions
            ``[b * band, b * band + P)``, columns of its bank) with one eviction per bank and no band DMA.
            Needs every chain a whole ``F_MAX`` n tile. Band rows past ``P`` hold stale PSUM: move them
            only lane-wise or through pass-through transposes, never read them.
        evict_engine: engine for the banded evictions (the compiler picks by default).
    """
    K, N = w_hbm.shape[0], w_hbm.shape[1]
    if stationary_fn is None:
        kernel_assert(
            stationary_tiles_sbuf.shape[1] == div_ceil(K, P_MAX) * P,
            f"stationary packing {stationary_tiles_sbuf.shape[1]} must equal div_ceil(K={K}, {P_MAX}) * P={P}",
        )
    band = pe_column_band(P)
    n_bands = P_MAX // band
    n_tiles = list(TiledRange(N, F_MAX))
    subs = max(1, n_bands // len(n_tiles))
    # Chain c covers columns [c0, c1) of one n tile and lives in band c % n_bands of bank c // n_bands.
    chains = []
    for n_tile in n_tiles:
        width = div_ceil(n_tile.size, subs)
        for sub in range(subs):
            c0 = n_tile.start_offset + sub * width
            c1 = min(n_tile.end_offset, c0 + width)
            if c0 < c1:
                chains.append((c0, c1))
    n_banks = div_ceil(len(chains), n_bands)
    kernel_assert(
        n_banks <= PSUM_BANKS,
        f"N={N} needs {n_banks} PSUM banks but only {PSUM_BANKS} exist; tile N in the caller",
    )
    if banded:
        kernel_assert(
            subs == 1 and N % (F_MAX * n_bands) == 0 and out_hbm is None,
            f"banded linear needs N={N} a multiple of {F_MAX * n_bands}",
        )
    acc_psums = []
    for _ in range(n_banks):
        acc_psums.append(nl.ndarray((P_MAX, F_MAX), dtype=nl.float32, buffer=nl.psum))

    n_k_tiles = div_ceil(K, P_MAX)
    batched = N <= BATCH_LOAD_MAX_N and K % P_MAX == 0
    w_all_sbuf = None
    if batched:
        # Partition p receives rows p, p + P_MAX, ... as k-tile columns. The reshape doubles as a contiguity
        # check: ``.ap`` addresses storage, so a column-sliced view must not reach it, and reshape refuses one.
        w_tiled = w_hbm.reshape((n_k_tiles, P_MAX, N))
        w_all_sbuf = nl.ndarray((P_MAX, n_k_tiles, N), dtype=w_hbm.dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=w_all_sbuf,
            src=w_tiled.ap(pattern=[[N, P_MAX], [P_MAX * N, n_k_tiles], [1, N]], offset=0),
            dge_mode=nisa.dge_mode.none,
        )

    is_first_k = True
    for k_tile in TiledRange(K, P_MAX):
        k_start, k_end, k_size = k_tile.start_offset, k_tile.end_offset, k_tile.size

        if batched:
            w_moving = w_all_sbuf[0:k_size, k_tile.index, 0:N]
        else:
            w_moving_sbuf = nl.ndarray((P_MAX, N), dtype=w_hbm.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=w_moving_sbuf[0:k_size, 0:N], src=w_hbm[k_start:k_end, 0:N], dge_mode=nisa.dge_mode.none)
            w_moving = w_moving_sbuf[0:k_size, 0:N]

        stationary_tile_start = k_tile.index * P
        if stationary_fn is None:
            stationary = stationary_tiles_sbuf[0:k_size, stationary_tile_start : stationary_tile_start + P]
        else:
            stationary = stationary_fn(k_tile.index)
        for c, (c0, c1) in enumerate(chains):
            b, bank = c % n_bands, c // n_bands
            rows = slice(b * band, b * band + P)
            nisa.nc_matmul(
                dst=acc_psums[bank][rows, 0 : c1 - c0],
                stationary=stationary,
                moving=w_moving[:, c0:c1],
                accumulate=not is_first_k,
                tile_position=(0, b * band),
                tile_size=(P_MAX, band),
            )
        is_first_k = False

    if banded:
        # One eviction per bank, over the band rows up to the last token row.
        band_rows = (n_bands - 1) * band + P
        banded_sbuf = nl.ndarray((P_MAX, n_banks * F_MAX), dtype=stationary_tiles_sbuf.dtype, buffer=nl.sbuf)
        engine_kw = {} if evict_engine is None else dict(engine=evict_engine)
        for bank in range(n_banks):
            nisa.tensor_copy(
                dst=banded_sbuf[0:band_rows, bank * F_MAX : (bank + 1) * F_MAX],
                src=acc_psums[bank][0:band_rows, 0:F_MAX],
                **engine_kw,
            )
        return banded_sbuf

    # Evict each chain (with the fp32 -> output cast) on the partitions its band wrote, then move bands
    # 1.. down to rows [0, P), or with ``out_hbm`` store every chain to HBM.
    out_sbuf = nl.ndarray((P_MAX, N), dtype=stationary_tiles_sbuf.dtype, buffer=nl.sbuf)
    band_sbuf = nl.ndarray((P_MAX, N), dtype=stationary_tiles_sbuf.dtype, buffer=nl.sbuf)
    store_engines = (nisa.engine.sync, nisa.engine.scalar)
    for c, (c0, c1) in enumerate(chains):
        b, bank = c % n_bands, c // n_bands
        rows = slice(b * band, b * band + P)
        evicted_sbuf = out_sbuf if b == 0 else band_sbuf
        nisa.tensor_copy(dst=evicted_sbuf[rows, c0:c1], src=acc_psums[bank][rows, 0 : c1 - c0])
        if out_hbm is not None:
            nisa.dma_copy(
                dst=out_hbm[0:P, out_col0 + c0 : out_col0 + c1],
                src=evicted_sbuf[rows, c0:c1],
                dge_mode=nisa.dge_mode.hwdge,
                engine=store_engines[c % 2],
            )
        elif b > 0:
            nisa.dma_copy(dst=out_sbuf[0:P, c0:c1], src=band_sbuf[rows, c0:c1])
    return out_sbuf


def transpose_into_k_tiles(x_sbuf: nl.NkiTensor, P: int) -> nl.NkiTensor:
    """
    Tiles ``[P, K]`` across K, transposed so K lands on partitions.

    Returns ``[P_MAX, div_ceil(K, P_MAX) * P]``, the packing ``linear`` expects: k-tile ``i``
    occupies columns ``[i * P, (i + 1) * P)``.
    """

    dtype, K = x_sbuf.dtype, x_sbuf.shape[1]
    n_k_tiles = div_ceil(K, P_MAX)

    tiles_sbuf = nl.ndarray((P_MAX, n_k_tiles * P), dtype=dtype, buffer=nl.sbuf)

    for tile in TiledRange(K, P_MAX):
        tile_psum = nl.ndarray((P_MAX, P_MAX), dtype=dtype, buffer=nl.psum)
        nisa.nc_transpose(tile_psum[0 : tile.size, 0:P], x_sbuf[0:P, tile.start_offset : tile.end_offset])

        tile_start = tile.index * P
        nisa.tensor_copy(dst=tiles_sbuf[0 : tile.size, tile_start : tile_start + P], src=tile_psum[0 : tile.size, 0:P])

    return tiles_sbuf


def rmsnorm_inplace(x_sbuf: nl.NkiTensor, gamma_sbuf: nl.NkiTensor, P: int, eps: float) -> None:
    """
    Applies ``x *= rsqrt(mean(x^2) + eps) * gamma`` in place over ``x_sbuf[P, F]``.

    F is the tile's full free width: to normalize one field of a wider row, pass a slice of exactly
    that field, or the rest of the row folds into the mean.
    """

    rmsnorm_into(x_sbuf, x_sbuf, gamma_sbuf, P, eps)


def rmsnorm_into(dst_sbuf: nl.NkiTensor, x_sbuf: nl.NkiTensor, gamma_sbuf: nl.NkiTensor, P: int, eps: float) -> None:
    """
    :func:`rmsnorm_inplace` into a separate ``dst``, bit-identical, leaving ``x`` intact for a later reader
    (the residual the block adds back onto).
    """

    F = x_sbuf.shape[1]

    squares_discard = nl.ndarray((P_MAX, F), dtype=nl.float32, buffer=nl.sbuf)
    inv_rms = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation_reduce(
        dst=squares_discard[0:P, 0:F],
        op=nl.square,
        data=x_sbuf[0:P, 0:F],
        reduce_op=nl.add,
        reduce_res=inv_rms[0:P, 0:1],
    )

    # The mean's 1/F and the eps fold into the activation's scale and bias: rsqrt(sum/F + eps).
    nisa.activation(inv_rms[0:P, 0:1], op=nl.rsqrt, data=inv_rms[0:P, 0:1], scale=1.0 / F, bias=eps)

    nisa.scalar_tensor_tensor(
        dst=dst_sbuf[0:P, 0:F],
        data=x_sbuf[0:P, 0:F],
        op0=nl.multiply,
        operand0=inv_rms[0:P, 0:1],
        op1=nl.multiply,
        operand1=gamma_sbuf[0:P, 0:F],
    )


def rope_inplace(
    x_sbuf: nl.NkiTensor,
    cos_sbuf: nl.NkiTensor,
    sin_sbuf: nl.NkiTensor,
    col_offset: int,
    block_stride: int,
    n_rope_blocks: int,
    P: int,
) -> None:
    """
    Applies interleaved RoPE in place across ``n_rope_blocks`` equally strided blocks at once.

    Interleaved: element ``2i`` pairs with ``2i + 1``, so even and odd lanes are stride-2 access patterns
    and ``even' = even * cos - odd * sin``, ``odd' = odd * cos + even * sin``. Head ``h``'s rope field sits
    at ``h * qk_head_dim + qk_nope_dim``, so one access pattern covers every head in a single pass.
    """

    rope_dim_half = cos_sbuf.shape[1]
    partition_stride = x_sbuf.shape[1]

    # [stride, count] per axis: partition, then block, then the stride-2 even or odd lane.
    block_pattern = [[partition_stride, P], [block_stride, n_rope_blocks], [2, rope_dim_half]]

    even = x_sbuf.ap(pattern=block_pattern, offset=col_offset)
    odd = x_sbuf.ap(pattern=block_pattern, offset=col_offset + 1)

    cos_blocks = cos_sbuf[0:P, 0:rope_dim_half].expand_dim(1).broadcast(1, n_rope_blocks)
    sin_blocks = sin_sbuf[0:P, 0:rope_dim_half].expand_dim(1).broadcast(1, n_rope_blocks)

    even_out = nl.ndarray((P_MAX, n_rope_blocks, rope_dim_half), dtype=nl.float32, buffer=nl.sbuf)
    odd_out = nl.ndarray((P_MAX, n_rope_blocks, rope_dim_half), dtype=nl.float32, buffer=nl.sbuf)
    scratch = nl.ndarray((P_MAX, n_rope_blocks, rope_dim_half), dtype=nl.float32, buffer=nl.sbuf)

    nisa.tensor_tensor(even_out[0:P], even, cos_blocks, nl.multiply)
    nisa.tensor_tensor(scratch[0:P], odd, sin_blocks, nl.multiply)
    nisa.tensor_tensor(even_out[0:P], even_out[0:P], scratch[0:P], nl.subtract)

    nisa.tensor_tensor(odd_out[0:P], odd, cos_blocks, nl.multiply)
    nisa.tensor_tensor(scratch[0:P], even, sin_blocks, nl.multiply)
    nisa.tensor_tensor(odd_out[0:P], odd_out[0:P], scratch[0:P], nl.add)

    nisa.tensor_copy(dst=even, src=even_out[0:P])
    nisa.tensor_copy(dst=odd, src=odd_out[0:P])


def load_gamma_broadcast(gamma_hbm: nl.NkiTensor, P: int) -> nl.NkiTensor:
    """Loads an RMSNorm gamma ``[1, D]`` broadcast to ``[P, D]``."""

    D = gamma_hbm.shape[1]

    gamma_sbuf = nl.ndarray((P_MAX, D), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=gamma_sbuf[0:P, 0:D], src=gamma_hbm[0:1, 0:D].broadcast(0, P))

    return gamma_sbuf


def load_rope_tables(cos: nl.NkiTensor, sin: nl.NkiTensor, P: int) -> tuple[nl.NkiTensor, nl.NkiTensor]:
    """Loads the interleaved-RoPE ``cos`` and ``sin`` tables, each ``[P, rope_dim // 2]``."""

    rope_dim_half = cos.shape[1]

    cos_sbuf = nl.ndarray((P_MAX, rope_dim_half), dtype=nl.float32, buffer=nl.sbuf)
    sin_sbuf = nl.ndarray((P_MAX, rope_dim_half), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=cos_sbuf[0:P, 0:rope_dim_half], src=cos[0:P, 0:rope_dim_half])
    nisa.dma_copy(dst=sin_sbuf[0:P, 0:rope_dim_half], src=sin[0:P, 0:rope_dim_half])

    return cos_sbuf, sin_sbuf
