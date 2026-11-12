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

"""
GLM-5.2 MLA decode: the W_UV absorb fused with the row-parallel output projection.

The mirror of the front's W_UK absorb: the query expands into the cache's latent space on the way in,
and the value contracts back out of it here.

o_proj is row-parallel because its contraction, ``num_heads * v_head_dim``, is the axis KVDP already
shards. The output is therefore a partial sum over heads; the caller's ReduceScatter completes it and
re-shards tokens onto the MLP's layout.

The absorb and projection bodies are ported from the kernel library's MLA decode attention core.
"""

import nki
import nki.isa as nisa
import nki.language as nl

from ....attention.mla.mla_decode_constants import F_MAX, P_MAX
from ....attention.mla.mla_proj_constants import OutProjShape, get_out_proj_shape
from ....utils.kernel_assert import kernel_assert
from ....utils.kernel_helpers import div_ceil
from ....utils.tiled_range import TiledRange

# The o-projection evicts one PSUM bank per copy (all column bands at once) rather than one ``[T, F_MAX]`` copy per
# hidden tile: 3 copies instead of 12 at the production shape, since Vector cost follows the free size, not the
# partition count. Its output DMAs go on the Sync HWDGE queue; left to the compiler they land on ACT behind the routed
# experts' affinity-offset multiply, which can wait on a small SWDGE load queued behind the router weights.

_O_PROJ_PSUM_BANKS = 6
"""PSUM banks the o-projection may hold live at once (of 8). At T <= 32 four hidden tiles share a bank in
PE column bands, so 3 banks hold the 6144-wide hidden and the cap is not reached; it bounds the pre-CP
shape (T = 112, one tile per bank), which walks the hidden in groups of 6 tiles."""


@nki.jit
def mla_decode_out_proj(
    ctx: nl.NkiTensor,
    out_absorb_w: nl.NkiTensor,
    o_proj_w: nl.NkiTensor,
) -> nl.NkiTensor:
    """
    Absorb W_UV into this rank's latent attention output and project it through o_proj.

    Usage:
        Tuned for: GLM-5.2 MLA decode, head-shard-8 o_proj contraction, post-CP-combine, LNC=1
        Applicable to: any absorbed-MLA output path whose o_proj is row-parallel on heads

    Args:
        ctx (nl.NkiTensor): [T, num_heads, C] @ HBM, this rank's latent attention output, already
            CP-combined. T is the post-ReduceScatter token count, not the count the attention core scored.
        out_absorb_w (nl.NkiTensor): [C, num_heads, v_head_dim] @ HBM, the value half of ``wkv_b`` for this
            rank's heads, stored latent-major so it loads as ``C / 128`` contiguous DMAs with no on-chip
            transpose (the checkpoint's ``[num_heads, v_head_dim, C]`` order would cost 128 small DMAs and
            128 PE transposes per call; the weight store transposes once, offline). Per-head, so KVDP
            already shards it.
        o_proj_w (nl.NkiTensor): [num_heads * v_head_dim, hidden_shard] @ HBM, this rank's rows of the
            output projection. The row count is asserted, since it is what makes the output a partial sum.

    Returns:
        nl.NkiTensor: [T, hidden_shard] @ HBM, a partial sum over heads. A ReduceScatter over the
        head-sharded ranks completes it and re-shards tokens in one collective.

    Notes:
        ``hidden_shard`` is read off ``o_proj_w``, so one kernel serves both schemes: the full hidden for
        pure row-parallel (exit is one ReduceScatter), or a slice for the 2-D variant (ReduceScatter over
        heads and AllGather over hidden).

        The o-projection weight streams once, contiguously: one DMA per 128-row chunk over the whole
        hidden width, 32 per call. That needs every hidden tile's accumulator live at once, which fits
        because the T-wide stationary leaves room for four copies in 32-column PE bands: four hidden
        tiles accumulate in four partition bands of one PSUM bank, three banks hold 6144 hidden, and the
        bands stream their weight tiles concurrently. See ``_o_proj``.
    """

    shapes = get_out_proj_shape(ctx, out_absorb_w, o_proj_w)
    T, hidden_shard = shapes.num_tokens, shapes.hidden_shard

    ctx_t_sbuf = _load_ctx_transposed(ctx, shapes)
    out_absorb_t_sbuf = _load_out_absorb(out_absorb_w, shapes)

    attn_sbuf = _out_absorb(ctx_t_sbuf, out_absorb_t_sbuf, shapes)

    # private_hbm, not shared: the caller feeds this partial sum straight into a ReduceScatter, and at
    # LNC=1 the runtime rejects a collective whose sources are shared_hbm.
    attn_out = nl.ndarray((T, hidden_shard), dtype=shapes.dtype, buffer=nl.private_hbm)
    _o_proj(attn_out, attn_sbuf, o_proj_w, shapes)

    return attn_out


class _CtxSpec:
    """The ``shape`` / ``dtype`` of a token-major ``[T, num_heads, C]`` context that only exists on chip."""

    def __init__(self, shape: tuple, dtype):
        self.shape, self.dtype = shape, dtype


def mla_decode_out_proj_sbuf(
    ctx_t_sbuf: nl.NkiTensor,
    num_tokens: int,
    out_absorb_w: nl.NkiTensor,
    o_proj_w: nl.NkiTensor,
    rows_table: tuple = None,
) -> nl.NkiTensor:
    """
    :func:`mla_decode_out_proj` for a context that is already on chip, latent-major.

    Args:
        ctx_t_sbuf (nl.NkiTensor): [P_MAX, num_latent_tiles * num_heads * num_tokens] @ SBUF, in the layout
            ``_load_ctx_transposed`` builds: latent chunk ``chunk`` of head ``head`` at column
            ``(chunk * num_heads + head) * num_tokens``.
        num_tokens (int): T, the tokens on the context's free axis.
        out_absorb_w, o_proj_w (nl.NkiTensor): As for :func:`mla_decode_out_proj`.
        rows_table (tuple): Optional ``(table, offsets)``: write token ``t``'s row straight from the bank copies into
            row ``exit_rows[t]`` of ``table`` ([n_seq, hidden_shard] HBM) by one indirect DMA, with ``offsets`` from
            :func:`o_proj_rows_offsets`; other rows of ``table`` are left untouched. (default: None)

    Returns:
        nl.NkiTensor: [T, hidden_shard] @ private HBM, the partial sum over this rank's heads; ``table`` with
        ``rows_table``.
    """
    C, Nh = out_absorb_w.shape[0], out_absorb_w.shape[1]
    shapes = get_out_proj_shape(_CtxSpec((num_tokens, Nh, C), ctx_t_sbuf.dtype), out_absorb_w, o_proj_w)
    out_absorb_t_sbuf = _load_out_absorb(out_absorb_w, shapes)
    attn_sbuf = _out_absorb(ctx_t_sbuf, out_absorb_t_sbuf, shapes)
    if rows_table is not None:
        _o_proj(None, attn_sbuf, o_proj_w, shapes, rows_table=rows_table)
        return rows_table[0]
    attn_out = nl.ndarray((num_tokens, shapes.hidden_shard), dtype=shapes.dtype, buffer=nl.private_hbm)
    _o_proj(attn_out, attn_sbuf, o_proj_w, shapes)
    return attn_out


def _o_proj_bank_layout(num_tokens: int, hidden_shard: int) -> tuple:
    """``(band, n_bands, tiles_per_band)`` when :func:`_o_proj` evicts ``hidden_shard`` as one group of full banks."""
    band = _pe_column_band(num_tokens)
    n_bands = P_MAX // band
    n_tiles = div_ceil(hidden_shard, F_MAX)
    tiles_per_band = div_ceil(n_tiles, n_bands)
    fits = hidden_shard % F_MAX == 0 and n_tiles == n_bands * tiles_per_band and tiles_per_band <= _O_PROJ_PSUM_BANKS
    return (band, n_bands, tiles_per_band) if fits else None


def o_proj_rows_offsets(exit_rows: nl.NkiTensor, n_seq: int, hidden_shard: int) -> nl.NkiTensor:
    """Per-partition row offsets for the ``rows_table`` write of :func:`mla_decode_out_proj_sbuf`.

    The bank copies leave token ``t``'s hidden span ``[k * span, (k + 1) * span)`` on partition ``k * band + t``
    (``span = tiles_per_band * F_MAX``), so the table viewed as ``[n_seq * n_bands, span]`` takes that partition at
    row ``exit_rows[t] * n_bands + k``. Every other partition gets ``n_seq * n_bands + k``, past the end, which the
    indirect write skips. Built from the runtime row indices, since one program serves every rank.

    Args:
        exit_rows (nl.NkiTensor): [T, 1] int32 HBM, the table row of each of the o-projection's tokens.
        n_seq (int): Table rows.
        hidden_shard (int): Table width, the o-projection's output width.

    Returns:
        nl.NkiTensor: [P_MAX, 1] uint32 SBUF.
    """
    T = exit_rows.shape[0]
    layout = _o_proj_bank_layout(T, hidden_shard)
    kernel_assert(layout is not None, f"o_proj rows table: hidden {hidden_shard} at T={T} is not one group of banks")
    band, n_bands, _ = layout
    seq_i = nl.ndarray((P_MAX, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(seq_i, pattern=[[0, 1]], offset=n_seq, channel_multiplier=0)
    for k in range(n_bands):
        nisa.dma_copy(dst=seq_i[k * band : k * band + T, 0:1], src=exit_rows[0:T, 0:1])
    band_i = nl.ndarray((P_MAX, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(band_i, pattern=[[0, 1]], offset=0, channel_multiplier=1)
    nisa.tensor_scalar(dst=band_i, data=band_i, op0=nl.right_shift, operand0=band.bit_length() - 1)
    # seq * n_bands + k, exact in fp32.
    seq_f = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=seq_f, src=seq_i)
    band_f = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=band_f, src=band_i)
    nisa.scalar_tensor_tensor(
        dst=seq_f, data=seq_f, op0=nl.multiply, operand0=float(n_bands), op1=nl.add, operand1=band_f
    )
    offsets = nl.ndarray((P_MAX, 1), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=offsets, src=seq_f)
    return offsets


def _load_ctx_transposed(ctx: nl.NkiTensor, shapes: OutProjShape) -> nl.NkiTensor:
    """
    Loads the latent attention output with the latent axis on partitions, which the absorb contracts.

    Returns ``[P_MAX, num_latent_tiles * num_heads * T]``, holding latent chunk ``chunk`` of head
    ``head`` at column ``(chunk * num_heads + head) * T``. That is the layout ``_out_absorb`` indexes.
    """

    T, Nh, C = shapes.num_tokens, shapes.num_heads, shapes.kv_lora_dim
    nC = shapes.num_latent_tiles

    ctx_t_sbuf = nl.ndarray((P_MAX, nC * Nh * T), dtype=shapes.dtype, buffer=nl.sbuf)
    for head in range(Nh):
        head_sbuf = nl.ndarray((P_MAX, C), dtype=shapes.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=head_sbuf[0:T, 0:C], src=ctx[0:T, head, 0:C])

        for latent_tile in TiledRange(C, P_MAX):
            chunk_psum = nl.ndarray((P_MAX, P_MAX), dtype=shapes.dtype, buffer=nl.psum)
            nisa.nc_transpose(
                chunk_psum[0 : latent_tile.size, 0:T],
                head_sbuf[0:T, latent_tile.start_offset : latent_tile.end_offset],
            )

            base = (latent_tile.index * Nh + head) * T
            nisa.tensor_copy(
                dst=ctx_t_sbuf[0 : latent_tile.size, base : base + T],
                src=chunk_psum[0 : latent_tile.size, 0:T],
            )

    return ctx_t_sbuf


def _load_out_absorb(out_absorb_w: nl.NkiTensor, shapes: OutProjShape) -> nl.NkiTensor:
    """
    Loads the out-absorb weight with the latent axis on partitions, straight from HBM.

    Returns ``[P_MAX, num_latent_tiles, num_heads * v_head_dim]``: latent tile ``chunk`` at
    ``[.., chunk, ..]``, head ``head``'s v width contiguous at column ``head * v_head_dim``. The stored
    ``[C, num_heads, v_head_dim]`` layout makes each latent tile one contiguous DMA, with no transpose.
    """

    Nh, v_head_dim, C = shapes.num_heads, shapes.v_head_dim, shapes.kv_lora_dim
    nC = shapes.num_latent_tiles
    width = Nh * v_head_dim

    out_absorb_flat = out_absorb_w.reshape((C, width))
    out_absorb_t_sbuf = nl.ndarray((P_MAX, nC, width), dtype=shapes.dtype, buffer=nl.sbuf)
    for latent_tile in TiledRange(C, P_MAX):
        nisa.dma_copy(
            dst=out_absorb_t_sbuf[0 : latent_tile.size, latent_tile.index, 0:width],
            src=out_absorb_flat[latent_tile.start_offset : latent_tile.end_offset, 0:width],
        )

    return out_absorb_t_sbuf


def _out_absorb(
    ctx_t_sbuf: nl.NkiTensor,
    out_absorb_t_sbuf: nl.NkiTensor,
    shapes: OutProjShape,
) -> nl.NkiTensor:
    """
    Projects every token's latent attention output back out of the cache's latent space.

    Returns ``[P_MAX, num_v_tiles, T * num_heads]``, v-tile ``vt`` of head ``head`` for all tokens at
    ``[.., vt, head * T]`` with the v index on partitions: already the o-projection's contraction layout.

    The v axis is the stationary free dim, hence the PSUM partition dim, and GLM-5.2's 256 exceeds 128
    partitions, so each v tile is its own matmul chain over the latent.

    Computes ``attn = einsum("thc,hvc->thv", ctx, out_absorb_w)``.
    """

    T, Nh = shapes.num_tokens, shapes.num_heads
    v_head_dim, C, nC = shapes.v_head_dim, shapes.kv_lora_dim, shapes.num_latent_tiles
    nV = shapes.num_v_tiles

    attn_sbuf = nl.ndarray((P_MAX, nV, T * Nh), dtype=shapes.dtype, buffer=nl.sbuf)
    for head in range(Nh):
        for v_tile in TiledRange(v_head_dim, P_MAX):
            v_size = v_tile.size

            attn_psum = nl.ndarray((P_MAX, T), dtype=nl.float32, buffer=nl.psum)
            is_first_matmul = True
            for latent_tile in TiledRange(C, P_MAX):
                latent_size = latent_tile.size
                weight_base = head * v_head_dim + v_tile.start_offset
                latent_base = (latent_tile.index * Nh + head) * T

                nisa.nc_matmul(
                    dst=attn_psum[0:v_size, 0:T],
                    stationary=out_absorb_t_sbuf[0:latent_size, latent_tile.index, weight_base : weight_base + v_size],
                    moving=ctx_t_sbuf[0:latent_size, latent_base : latent_base + T],
                    accumulate=not is_first_matmul,
                )
                is_first_matmul = False

            nisa.tensor_copy(
                dst=attn_sbuf[0:v_size, v_tile.index, head * T : (head + 1) * T],
                src=attn_psum[0:v_size, 0:T],
            )

    return attn_sbuf


def _pe_column_band(num_tokens: int) -> int:
    """Narrowest PE column tile (32, 64 or 128) that holds a ``num_tokens``-wide stationary."""
    for band in (P_MAX // 4, P_MAX // 2, P_MAX):
        if num_tokens <= band:
            return band
    kernel_assert(False, f"num_tokens={num_tokens} exceeds the {P_MAX}-column PE array")
    return P_MAX


def _o_proj(
    output: nl.NkiTensor,
    attn_sbuf: nl.NkiTensor,
    o_proj_w: nl.NkiTensor,
    shapes: OutProjShape,
    rows_table: tuple = None,
) -> None:
    """
    Contracts every head's attention output through this rank's o-projection rows and evicts it.

    The contraction is walked as ``num_heads * num_v_tiles`` 128-row chunks: row ``head * v_head_dim + v``
    of ``o_proj_w`` pairs with v-tile ``v // P_MAX`` of head ``head``, the reference's flattening order.

    Chunk-outer, hidden-inner: each chunk's rows load as one contiguous DMA over the hidden group, and
    every hidden tile's accumulator stays live. The PE array is split into ``P_MAX / band`` column bands
    (``band`` = 32 for T <= 32); band ``k`` holds its own stationary copy and writes PSUM partitions
    ``[k * band, k * band + T)``, so one bank accumulates ``P_MAX / band`` hidden tiles. Tile ``j`` of a
    group goes to band ``j // tiles_per_band``, bank ``j % tiles_per_band``: band-major, so each band
    covers a contiguous hidden span and evicts with one DMA. Bands run concurrently.

    A hidden needing more than ``_O_PROJ_PSUM_BANKS`` banks (only the pre-CP T = 112 shape) is walked in
    groups, each re-reading its own weight columns.

    ``rows_table`` (see :func:`mla_decode_out_proj_sbuf`) needs one group of full banks: one copy per
    bank, then one indirect DMA writes every band's rows into the table; ``output`` is unused.
    """

    T, Nh = shapes.num_tokens, shapes.num_heads
    v_head_dim, hidden_shard = shapes.v_head_dim, shapes.hidden_shard

    band = _pe_column_band(T)
    n_bands = P_MAX // band
    tiles = list(TiledRange(hidden_shard, F_MAX))
    chunks = [(head, v_tile) for head in range(Nh) for v_tile in TiledRange(v_head_dim, P_MAX)]
    group_tiles = n_bands * _O_PROJ_PSUM_BANKS
    kernel_assert(
        rows_table is None or _o_proj_bank_layout(T, hidden_shard) is not None,
        f"o_proj rows table: hidden {hidden_shard} at T={T} is not one group of full PSUM banks",
    )

    out_sbuf = nl.ndarray((P_MAX, hidden_shard), dtype=shapes.dtype, buffer=nl.sbuf)
    for g0 in range(0, len(tiles), group_tiles):
        gtiles = tiles[g0 : g0 + group_tiles]
        tiles_per_band = div_ceil(len(gtiles), n_bands)
        cols0, cols1 = gtiles[0].start_offset, gtiles[-1].end_offset
        group_width = cols1 - cols0

        o_psum = [nl.ndarray((P_MAX, F_MAX), dtype=nl.float32, buffer=nl.psum) for _ in range(tiles_per_band)]
        for chunk_idx, (head, v_tile) in enumerate(chunks):
            v_size = v_tile.size
            row_start = head * v_head_dim + v_tile.start_offset

            # The chunk's rows are adjacent in HBM: one contiguous read with pre-built descriptors.
            w_sbuf = nl.ndarray((P_MAX, group_width), dtype=o_proj_w.dtype, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=w_sbuf[0:v_size, 0:group_width],
                src=o_proj_w[row_start : row_start + v_size, cols0:cols1],
                dge_mode=nisa.dge_mode.none,
            )

            for j, tile in enumerate(gtiles):
                bank, k = j % tiles_per_band, j // tiles_per_band
                rows = slice(k * band, k * band + T)
                nisa.nc_matmul(
                    dst=o_psum[bank][rows, 0 : tile.size],
                    stationary=attn_sbuf[0:v_size, v_tile.index, head * T : (head + 1) * T],
                    moving=w_sbuf[0:v_size, tile.start_offset - cols0 : tile.end_offset - cols0],
                    accumulate=chunk_idx > 0,
                    tile_position=(0, k * band),
                    tile_size=(P_MAX, band),
                )

        dma_kw = dict(dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)
        full_banks = len(gtiles) == n_bands * tiles_per_band and all(t.size == F_MAX for t in gtiles)
        if full_banks or rows_table is not None:
            # One copy per bank over every band's partitions: bank b's band k rows land at columns
            # [b * F_MAX, (b + 1) * F_MAX) of band_sbuf, so band k's rows hold its tiles in hidden order.
            n_rows = (n_bands - 1) * band + T
            band_sbuf = nl.ndarray((P_MAX, tiles_per_band * F_MAX), dtype=shapes.dtype, buffer=nl.sbuf)
            for bank in range(tiles_per_band):
                nisa.tensor_copy(
                    dst=band_sbuf[0:n_rows, bank * F_MAX : (bank + 1) * F_MAX],
                    src=o_psum[bank][0:n_rows, 0:F_MAX],
                    engine=nisa.vector_engine,
                )
            if rows_table is not None:
                # Partition k * band + t holds token t's band-k span: one indirect write moves every band's rows
                # to their table rows, and the partitions between the bands are skipped as out of range.
                table, offsets = rows_table
                span = tiles_per_band * F_MAX
                n_seq = table.shape[0]
                nisa.dma_copy(
                    dst=table.reshape((n_seq * n_bands, span)).ap(
                        pattern=[[span, n_rows], [1, span]],
                        offset=0,
                        vector_offset=offsets[0:n_rows, 0:1],
                        indirect_dim=0,
                    ),
                    src=band_sbuf[0:n_rows, 0:span],
                    oob_mode=nisa.oob_mode.skip,
                )
                continue
            for k in range(n_bands):
                c0 = gtiles[k * tiles_per_band].start_offset
                nisa.dma_copy(
                    dst=output[0:T, c0 : c0 + tiles_per_band * F_MAX],
                    src=band_sbuf[k * band : k * band + T, 0 : tiles_per_band * F_MAX],
                    **dma_kw,
                )
            continue
        # Evict on the partitions the band wrote, then one DMA per band moves its hidden span to token
        # rows [0, T) of the output; only a DMA crosses partitions.
        for j, tile in enumerate(gtiles):
            bank, k = j % tiles_per_band, j // tiles_per_band
            rows = slice(k * band, k * band + T)
            nisa.tensor_copy(
                dst=out_sbuf[rows, tile.start_offset : tile.end_offset],
                src=o_psum[bank][rows, 0 : tile.size],
            )
        for k in range(n_bands):
            band_tiles = gtiles[k * tiles_per_band : (k + 1) * tiles_per_band]
            if not band_tiles:
                break
            c0, c1 = band_tiles[0].start_offset, band_tiles[-1].end_offset
            nisa.dma_copy(dst=output[0:T, c0:c1], src=out_sbuf[k * band : k * band + T, c0:c1], **dma_kw)
