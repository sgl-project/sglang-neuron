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

The mirror of the W_UK absorb: the query went into the cache's latent space on the way in, and the
value comes back out of it here.

o_proj is row-parallel because its contraction, ``num_heads * v_head_dim``, is the axis the head
shard already splits. The output is a partial sum over heads; the caller's ReduceScatter completes
it and re-shards tokens for the MLP.

The absorb and projection bodies come from the kernel library's MLA decode attention core.
"""

import os

import nki
import nki.isa as nisa
import nki.language as nl

from ...utils.kernel_assert import kernel_assert
from ...utils.kernel_helpers import div_ceil
from ...utils.tiled_range import TiledRange
from .mla_decode_constants import F_MAX, P_MAX
from .mla_proj_constants import OutProjShape, get_out_proj_shape

# With full banks the o-projection evicts one PSUM bank per copy (every column band at once) instead of one copy
# per hidden tile; Vector cost follows the free size, not the partition count. Its output DMAs go on the Sync
# HWDGE queue: left to the compiler they land on ACT, behind the routed experts' affinity-offset multiply, which
# waits on a small SWDGE load queued behind the router weights.
LAT_OPROJ_EVICT_ALT = os.environ.get("GLM52K_LAT_OPROJ_EVICT_ALT", "0") == "1"
"""Evict the full banks alternately on Vector and ACT instead of all on Vector. At 128 tokens the evictions, not
the matmuls, are the o-projection's tail. With ``LAT_OPROJ_STORE_EACH`` each bank's store goes on its evicting
engine's queue. Same cast on either engine. Off by default."""
LAT_OPROJ_STORE_EACH = os.environ.get("GLM52K_LAT_OPROJ_STORE_EACH", "0") == "1"
"""Store each full bank as soon as it is evicted, alternating the Sync and ACT HWDGE queues, instead of one DMA
per band after the group's last eviction, so the ReduceScatter source (1.5 MB at 128 tokens) is not stored all
at the end. Same bytes. Off by default."""

_O_PROJ_PSUM_BANKS = 6
"""PSUM banks (of 8) the o-projection may hold live at once, each one ``[P_MAX, F_MAX]`` fp32 accumulator. At
T <= 32 four hidden tiles share a bank in PE column bands, so 3 banks hold the 6144-wide hidden. Wider T gets one
tile per bank and walks the hidden in groups of 6 tiles."""


@nki.jit
def mla_decode_out_proj(
    ctx: nl.NkiTensor,
    out_absorb_w: nl.NkiTensor,
    o_proj_w: nl.NkiTensor,
    weight_gate=None,
) -> nl.NkiTensor:
    """
    Absorb W_UV into this rank's latent attention output and project it through o_proj.

    Usage:
        Tuned for: GLM-5.2 MLA decode, o_proj row-sharded on heads, after the CP combine, LNC=1
        Applicable to: any absorbed-MLA output path whose o_proj is row-parallel on heads

    ``hidden_shard`` comes off ``o_proj_w``, so one kernel serves both schemes: the full hidden width for
    pure row parallelism, or a slice for the 2-D variant. The o_proj weight streams once, one contiguous
    DMA per 128-row chunk across the whole hidden width, with every hidden tile's accumulator live in PE
    column bands (see ``_o_proj``).

    Args:
        ctx (nl.NkiTensor): [T, num_heads, C] @ HBM. This rank's CP-combined latent attention output; T
            is the token count after the ReduceScatter, not the count the core scored.
        out_absorb_w (nl.NkiTensor): [C, num_heads, v_head_dim] @ HBM. The value half of ``wkv_b`` for
            this rank's heads, latent-major: row ``c`` holds every head's ``v`` weights for latent ``c``.
            That is the contraction layout the absorb wants on partitions, so it loads as ``C / 128``
            contiguous DMAs with no on-chip transpose (the weight store transposes the checkpoint's
            ``[num_heads, v_head_dim, C]`` once, offline).
        o_proj_w (nl.NkiTensor): [num_heads * v_head_dim, hidden_shard] @ HBM. This rank's rows of the
            output projection; the row count is asserted, since it is what makes the output a partial sum.
        weight_gate: An instruction the o_proj weight loads wait on (a runtime dependency), or None.

    Returns:
        nl.NkiTensor: [T, hidden_shard] @ HBM, a partial sum over heads. A ReduceScatter over the
        head-sharded ranks completes the sum and re-shards tokens in one collective.
    """

    shapes = get_out_proj_shape(ctx, out_absorb_w, o_proj_w)
    T, hidden_shard = shapes.num_tokens, shapes.hidden_shard

    ctx_t_sbuf = _load_ctx_transposed(ctx, shapes)
    out_absorb_t_sbuf = _load_out_absorb(out_absorb_w, shapes)

    attn_sbuf = _out_absorb(ctx_t_sbuf, out_absorb_t_sbuf, shapes)

    # private_hbm: the caller feeds this straight into a ReduceScatter, and at LNC=1 the runtime rejects
    # a collective whose sources are shared_hbm.
    attn_out = nl.ndarray((T, hidden_shard), dtype=shapes.dtype, buffer=nl.private_hbm)
    _o_proj(attn_out, attn_sbuf, o_proj_w, shapes, weight_gate=weight_gate)

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
) -> nl.NkiTensor:
    """
    :func:`mla_decode_out_proj` for a context that is already on chip, latent-major.

    Args:
        ctx_t_sbuf (nl.NkiTensor): [P_MAX, num_latent_tiles * num_heads * num_tokens] @ SBUF, in the layout
            ``_load_ctx_transposed`` builds: latent chunk ``chunk`` of head ``head`` at column
            ``(chunk * num_heads + head) * num_tokens``.
        num_tokens (int): T, the tokens on the context's free axis.
        out_absorb_w, o_proj_w (nl.NkiTensor): As for :func:`mla_decode_out_proj`.

    Returns:
        nl.NkiTensor: [T, hidden_shard] @ private HBM, the partial sum over this rank's heads.
    """
    C, Nh = out_absorb_w.shape[0], out_absorb_w.shape[1]
    shapes = get_out_proj_shape(_CtxSpec((num_tokens, Nh, C), ctx_t_sbuf.dtype), out_absorb_w, o_proj_w)
    out_absorb_t_sbuf = _load_out_absorb(out_absorb_w, shapes)
    attn_sbuf = _out_absorb(ctx_t_sbuf, out_absorb_t_sbuf, shapes)
    attn_out = nl.ndarray((num_tokens, shapes.hidden_shard), dtype=shapes.dtype, buffer=nl.private_hbm)
    _o_proj(attn_out, attn_sbuf, o_proj_w, shapes)
    return attn_out


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
    ``[.., chunk, ..]``, head ``head``'s v width contiguous from column ``head * v_head_dim``.

    The weight arrives ``[C, num_heads, v_head_dim]``, so latent tile ``chunk`` is rows
    ``[chunk * 128, (chunk + 1) * 128)``: one DMA per tile, contiguous per partition, no transpose.
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

    Computes ``attn = einsum("thc,hvc->thv", ctx, out_absorb_w)``.

    Returns ``[P_MAX, num_v_tiles, T * num_heads]``, v-tile ``vt`` of head ``head`` for all tokens at
    ``[.., vt, head * T]`` with the v index on partitions, which is already the o-projection's
    contraction layout. The v axis is tiled because it is the stationary's free dim, hence the PSUM
    partition dim, and 256 exceeds 128 partitions; each v tile is its own chain over the latent.
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
    weight_gate=None,
) -> None:
    """
    Contracts every head's attention output through this rank's o-projection rows and evicts it.

    The contraction walks ``num_heads * num_v_tiles`` 128-row chunks: row ``head * v_head_dim + v`` of
    ``o_proj_w`` pairs with v-tile ``v // P_MAX`` of head ``head``, the reference's flatten order.

    The loop is chunk-outer, hidden-inner, so each chunk's rows load as one contiguous DMA over the
    hidden group and every hidden tile's accumulator stays live for the whole stream. They fit because
    the stationary is only T columns wide: band ``k`` of the PE array (``band`` = 32 at T <= 32) holds
    its own copy and writes PSUM partitions ``[k * band, k * band + T)``, so one bank accumulates
    ``P_MAX / band`` hidden tiles, and the bands run concurrently. Hidden tile ``j`` goes to band
    ``j // tiles_per_band`` and bank ``j % tiles_per_band``, so a band's tiles are a contiguous hidden
    span that stores with one DMA. Past ``_O_PROJ_PSUM_BANKS`` the hidden is walked in groups, each
    re-reading its own weight columns.
    """

    T, Nh = shapes.num_tokens, shapes.num_heads
    v_head_dim, hidden_shard = shapes.v_head_dim, shapes.hidden_shard

    band = _pe_column_band(T)
    n_bands = P_MAX // band
    tiles = list(TiledRange(hidden_shard, F_MAX))
    chunks = [(head, v_tile) for head in range(Nh) for v_tile in TiledRange(v_head_dim, P_MAX)]
    group_tiles = n_bands * _O_PROJ_PSUM_BANKS

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

            # The chunk's rows are adjacent in HBM: one contiguous read with static descriptors.
            w_sbuf = nl.ndarray((P_MAX, group_width), dtype=o_proj_w.dtype, buffer=nl.sbuf)
            w_load = nisa.dma_copy(
                dst=w_sbuf[0:v_size, 0:group_width],
                src=o_proj_w[row_start : row_start + v_size, cols0:cols1],
                dge_mode=nisa.dge_mode.none,
            )
            if weight_gate is not None:
                w_load.depends_on(weight_gate, kind=nisa.dependency.runtime)

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
        if full_banks:
            # One copy per bank over every band's partitions: bank b's band k rows land at columns
            # [b * F_MAX, (b + 1) * F_MAX) of band_sbuf, so band k's rows hold its tiles in hidden order.
            n_rows = (n_bands - 1) * band + T
            band_sbuf = nl.ndarray((P_MAX, tiles_per_band * F_MAX), dtype=shapes.dtype, buffer=nl.sbuf)
            store_engines = (nisa.engine.sync, nisa.engine.scalar)
            for bank in range(tiles_per_band):
                bank_engine = nisa.vector_engine
                if LAT_OPROJ_EVICT_ALT:
                    bank_engine = (nisa.vector_engine, nisa.scalar_engine)[bank % 2]
                nisa.tensor_copy(
                    dst=band_sbuf[0:n_rows, bank * F_MAX : (bank + 1) * F_MAX],
                    src=o_psum[bank][0:n_rows, 0:F_MAX],
                    engine=bank_engine,
                )
                if LAT_OPROJ_STORE_EACH:
                    # Band k of this bank holds hidden tile k * tiles_per_band + bank.
                    for k in range(n_bands):
                        c0 = gtiles[k * tiles_per_band + bank].start_offset
                        store_engine = store_engines[(bank * n_bands + k) % 2]
                        if LAT_OPROJ_EVICT_ALT:
                            store_engine = store_engines[bank % 2]
                        nisa.dma_copy(
                            dst=output[0:T, c0 : c0 + F_MAX],
                            src=band_sbuf[k * band : k * band + T, bank * F_MAX : (bank + 1) * F_MAX],
                            dge_mode=nisa.dge_mode.hwdge,
                            engine=store_engine,
                        )
            if LAT_OPROJ_STORE_EACH:
                continue
            for k in range(n_bands):
                c0 = gtiles[k * tiles_per_band].start_offset
                nisa.dma_copy(
                    dst=output[0:T, c0 : c0 + tiles_per_band * F_MAX],
                    src=band_sbuf[k * band : k * band + T, 0 : tiles_per_band * F_MAX],
                    **dma_kw,
                )
            continue
        # Evict on the partitions the band wrote, then one DMA per band moves its contiguous hidden
        # span to token rows [0, T) of the output (only a DMA crosses partitions).
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
