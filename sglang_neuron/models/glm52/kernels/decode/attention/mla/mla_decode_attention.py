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
Absorbed-MLA decode attention over selector-chosen keys, for one context-parallel rank (GLM-5.2).

MLA's compressed cache holds one latent per token, so with W_UK folded into the query and W_UV into
the output, decode is MQA over a 576-wide row and no per-head K or V is ever materialized:

    scores[t,h,s] = scale * ( q_absorbed[t,h,:C] . kv_latent[s,:C]
                            + q_pe[t,h,:R]      . k_pe[s,:R] )
    ctx[t,h,:C]   = softmax_s(scores[t,h,:]) . kv_latent[:,:C]

The rope half has no W_UK to fold through, so a cache row is ``cat([k_pe(R), kv_latent(C)])`` and
two matmuls accumulate into one PSUM tile per key tile.

The kernel stops at ``ctx`` in the latent space and also returns the log-sum-exp. Under context
parallelism the ranks combine partials (LSE all-gather plus a reduce-scatter) and W_UV runs on the
post-scatter head slice, so each rank holds only its share of W_UV; o_proj also runs later, after the
head/token all-to-all. Both orders are exact: W_UV is linear and the combine is a weighted sum.

The token axis is flat over speculation (``num_tokens = num_seqs * spec``), each token with its own
position and selected-key row. The step's ``spec`` new rows are appended in band rather than
gathered, so no sibling is read back through HBM from a row this kernel just wrote.
"""

import math
import os

import nki
import nki.isa as nisa
import nki.language as nl
from nki.isa import dge_mode

from ...utils.kernel_assert import kernel_assert
from ...utils.kernel_helpers import div_ceil
from ...utils.tiled_range import TiledRange
from .kv_row_layout import E4M3FN_MAX, latent_write_slices
from .mla_decode_constants import F_MAX, MASK_BOUND, P_MAX, MlaDecodeShape, get_mla_decode_shape
from .prior_index import (
    build_prior_index,
    build_resident_index,
    export_prior_index,
    gather_prior_rows,
    import_prior_index,
)

# Schedule. This step's rows are written into the cache after the pair loop, so the prior gathers do not depend on the
# scatter (which sits behind the query exchange's trigger on GpSimd). That is safe because no live selection reads a
# row written this step: the indexer masks keys at or after ``positions[0]``, dead slots resolve to NON_RESIDENT_POS,
# and this step's rows enter only through the in-band sibling append. Tokens 0-1 gather once this step's rows are
# staged and tokens 2+ once the queries have loaded, so the early gathers do not starve q_kv_up's weight stream. The
# softmax scale is applied at each score bank's eviction (exact for a power-of-two scale, asserted at trace time), and
# the pair loop is software-pipelined two pairs at a time so one pair's softmax overlaps the other's PE work.

ATTN_RESIDENT_TILES = 6
"""Key tiles per token of the resident-only core: it processes only the keys this rank keeps (selected, resident and
causal, about 1/cp_degree of the selection; the rest have weight exactly 0), compacted to this many tiles by
``prior_index.build_resident_index``. The math is exact, but the sums group keys into fewer tiles, so fp32 rounding
differs from the full core. A token keeping more than ``ATTN_RESIDENT_TILES * 128`` keys sets an overflow flag, and
the full core then runs in a fenced dynamic loop (zero trips otherwise) and overwrites the outputs."""

_RESIDENT_TABLES = {}
"""The resident tables handed from a full layer to its group's shared layers in SBUF, together with the full
tables the fallback reads, so no table crosses the fallback's dynamic loop through HBM:
``id(snake_hbm) -> (snake_hbm, rows_c, key_pos_c, flag, snake_sb, key_pos_sb)``."""

_LOOP_DGE = []
"""Non-empty while tracing the resident fallback's body, whose plain DMAs then use the hardware DGE (a DMA inside a
dynamic loop must use a DGE queue)."""


def _loop_dge() -> dict:
    return {"dge_mode": dge_mode.hwdge} if _LOOP_DGE else {}


class _Stub:
    """Stands in for every instruction handle while ``_arena_record`` traces without instructions."""

    def __getattr__(self, name):
        return lambda *a, **k: self

    def __call__(self, *a, **k):
        return self


def _arena_record(fn) -> list:
    """
    Trace ``fn`` with every ``nki.isa`` call stubbed out, recording the SBUF / HBM tensors it allocates.

    ``_arena_replay`` then traces ``fn`` inside the dynamic loop on these tensors, because a tile allocated inside a
    ``fori_loop`` body can be placed over a tile still live outside it and corrupt it. PSUM tiles are still allocated
    by the body itself.
    """
    stub = _Stub()
    names = [n for n in dir(nisa) if not n.startswith("_") and callable(getattr(nisa, n))
             and not isinstance(getattr(nisa, n), type)]
    saved = {n: getattr(nisa, n) for n in names}
    real = nl.ndarray
    tiles = []

    def record(shape, dtype, buffer=nl.sbuf, name="", address=None):
        t = real(shape, dtype=dtype, buffer=buffer, name=name, address=address)
        if buffer is not nl.psum:
            tiles.append(t)
        return t

    try:
        nl.ndarray = record
        for n in names:
            setattr(nisa, n, lambda *a, **k: stub)
        fn()
    finally:
        nl.ndarray = real
        for n, f in saved.items():
            setattr(nisa, n, f)
    return tiles


def _arena_replay(tiles: list, fn) -> None:
    """Trace ``fn`` with its non-PSUM allocations served, in order, from ``_arena_record``'s tensors."""
    real = nl.ndarray
    it = iter(tiles)

    def replay(shape, dtype, buffer=nl.sbuf, name="", address=None):
        if buffer is nl.psum:
            return real(shape, dtype=dtype, buffer=buffer, name=name, address=address)
        t = next(it)
        kernel_assert(tuple(t.shape) == tuple(shape) and t.dtype == dtype, f"arena replay {t.shape} vs {shape}")
        return t

    try:
        nl.ndarray = replay
        fn()
    finally:
        nl.ndarray = real
    kernel_assert(next(it, None) is None, "arena replay left tiles unused")


ATTN_Q_HOOK = int(os.environ.get("GLM52K_ATTN_Q_HOOK", "0"))
"""Shared layers: the caller passes a ``q_hook`` instead of the queries, and the core emits pair 0's gathers and key
transposes before the query exchange's receive (the hook), so they run while the all_to_all is in flight and the PE
would otherwise idle. Same instructions reordered, so bit-identical. 0 = off."""

_KEY_EVICT_GROUP = 6
"""Number of consecutive full prior tiles transposed into one PSUM tile and evicted with one copy per part (rope, two
latent halves) instead of one per tile, since Vector, the bit-accurate PSUM reader, bounds the key build."""

ATTN_SIB_ONCE = os.environ.get("GLM52K_ATTN_SIB_ONCE", "0") == "1"
"""With one sequence per call (``num_seqs == 1``) the ``spec`` sibling rows are the same for every token, so token 0
transposes the sibling tile once and evicts it into every key-buffer slot, and a token reusing a slot an earlier
token filled skips re-appending them. Bit-identical. 0 = off."""

_RANK_MAJOR_HEADS = {}
"""Absorbed queries handed over from the head gather (``mla_decode_q_kv_up._gather_absorbed_heads``) in its
all-gather's rank-major layout (``heads_blocked``) rather than relaid out HBM-to-HBM in 32-byte elements; the core
loads the blocks in 1 KB runs and puts heads innermost on chip.
``id(q_absorbed_out) -> (q_absorbed_out, q_absorbed_all, q_pe_all, degree)``."""


def publish_rank_major_heads(
    q_absorbed_out: nl.NkiTensor, q_absorbed_all: nl.NkiTensor, q_pe_all: nl.NkiTensor, degree: int
) -> None:
    """Record that ``q_absorbed_out`` was left unwritten and its heads live rank-major in the gathered blocks."""
    _RANK_MAJOR_HEADS[id(q_absorbed_out)] = (q_absorbed_out, q_absorbed_all, q_pe_all, degree)


def take_rank_major_heads(q_absorbed_out: nl.NkiTensor):
    """``(q_absorbed_all, q_pe_all, degree)`` published for ``q_absorbed_out``, or None."""
    entry = _RANK_MAJOR_HEADS.pop(id(q_absorbed_out), None)
    if entry is None or entry[0] is not q_absorbed_out:
        return None
    return entry[1], entry[2], entry[3]


_PAIR_SLOTS = 2
"""
Token pairs in flight. Two let pair p+1's gathers and key transposes overlap pair p's softmax and
context sum; SBUF caps it, since a pair slot holds two tokens' gathered rows and feature-major keys
plus the pair's shared fp32 scores and bf16 weights.

Tokens are processed in pairs on the two halves of the PE array and the partition axis: token A's
queries are the stationary of PE column band 0 and its scores land on partitions [0, Nh), token B's
use band 1 and land on [64, 64 + Nh). Column tiling lets each band stream its own keys (selections
differ per token) and the two run concurrently. Everything after the scores is per partition, so one
instruction serves both tokens (eviction, max, exp+sum, log, reciprocal), and the weight transposes
are one full 128-row transpose per key tile per pair (a sub-128 transpose trips the di/dt throttle).
"""

_PAIR_BAND = 64
"""Width of one PE column band, and so token B's partition offset. Fixed by the array (2 x 64 column
tiles); ``num_heads`` must fit in it."""


@nki.jit
def mla_decode_attention(
    q_absorbed: nl.NkiTensor,
    q_pe: nl.NkiTensor,
    kv_cache: nl.NkiTensor,
    kv_row_new: nl.NkiTensor,
    new_slot_mapping: nl.NkiTensor,
    gather_indices: nl.NkiTensor,
    block_table: nl.NkiTensor,
    positions: nl.NkiTensor,
    softmax_scale: float,
    kv_scale: float,
    spec: int,
    kv_lora_rank: int,
    cp_degree: int,
    cp_rank: int,
    prior_snake_rows: nl.NkiTensor = None,
    prior_key_pos: nl.NkiTensor = None,
    prior_natural_rows: nl.NkiTensor = None,
    pos_offset: nl.NkiTensor = None,
    positions_km: nl.NkiTensor = None,
    ctx_lse_out: nl.NkiTensor = None,
    prior_tables_out: list = None,
    heads_blocked: int = 1,
    ctx_lse_token_split: int = 1,
    resident_tables: tuple = None,
    kv_update: bool = True,
    q_hook=None,
    q_shape: tuple = None,
) -> tuple[nl.NkiTensor, nl.NkiTensor, nl.NkiTensor]:
    """
    Score one decode step's queries against this rank's share of the selected keys.

    Usage:
        Tuned for: GLM-5.2 absorbed-MLA decode, one CP rank, LNC=1
        Applicable to: any absorbed-MLA decode over an externally selected, globally indexed key set

    Args:
        q_absorbed (nl.NkiTensor): [P_MAX, C/P_MAX, T, Nh] @ HBM or SBUF, the query already pushed
            through W_UK, feature-major: feature ``c`` of token ``t`` head ``h`` is at
            ``[c % P_MAX, c // P_MAX, t, h]``, the layout the query fold leaves in SBUF.
        q_pe (nl.NkiTensor): [P_MAX, T, Nh] @ HBM or SBUF, the RoPE'd query half, feature-major,
            valid in the leading ``R`` partitions.
        kv_cache (nl.NkiTensor): [num_blocks, 1, block_size, 640] uint8 @ HBM, this rank's local
            paged cache of packed rows (see ``kv_row_layout``). Mutated: this step's rows are
            quantized and written after every gather, and the tensor is returned.
        kv_row_new (nl.NkiTensor): [num_seqs, spec, R + C] bf16 @ HBM, this step's new rows,
            unquantized. The kernel quantizes and packs them.
        new_slot_mapping (nl.NkiTensor): [num_seqs, spec] int32 @ HBM, the flat local cache row each
            new row is written to, or negative to skip it.
        gather_indices (nl.NkiTensor): [num_seqs, spec, num_prior] int32 @ HBM, the global positions
            the selector chose, one row per (sequence, speculative token) as in the reference model's
            ``[B, S, topk]``; negative means not selected. Since each token has its own selection, its
            gathered key tile (the score matmul's stationary) differs, so speculative tokens cannot
            share a matmul.
        block_table (nl.NkiTensor): [num_seqs, max_blocks] int32 @ HBM, this rank's local table.
        positions (nl.NkiTensor): [T] int32 @ HBM, each query token's absolute position. A
            sequence's ``spec`` tokens must have consecutive positions, which lets the sibling causal
            mask be a position-free iota.
        softmax_scale (float): Score scale, ``qk_head_dim ** -0.5`` = 0.0625 for GLM-5.2 (default
            RoPE, no yarn mscale).
        kv_scale (float): Dequantization scale of the packed cache, which stores ``latent / kv_scale``
            and ``k_pe / kv_scale``. Folded into the score scale and the softmax denominator, so it
            costs no instructions. Must be a power of two, so pre-dividing the bf16 k_pe is exact and
            1.0 is a true no-op; 1.0 suits a layernorm-output latent, well inside e4m3fn's range.
        spec (int): Speculative tokens per sequence. 1 is plain decode.
        kv_lora_rank (int): Latent width C.
        cp_degree (int): Context-parallel degree; this rank holds every cp_degree'th logical block.
        cp_rank (int): This rank's index within its CP group.
        prior_snake_rows (nl.NkiTensor): [P_MAX, T, num_prior_tiles] uint32 @ HBM, or None. With
            ``prior_key_pos`` and ``prior_natural_rows``, the index tables ``build_prior_index``
            would compute from ``gather_indices`` and ``block_table`` (see ``export_prior_index``).
            Layers sharing one selection build them once and pass them here; ``gather_indices`` is
            then unused. None builds them in the kernel.
        prior_key_pos (nl.NkiTensor): [P_MAX, T, num_prior_tiles] fp32 @ HBM, or None. See above.
        prior_natural_rows (nl.NkiTensor): [P_MAX, T, num_prior_tiles] uint32 @ HBM, or None. Read by
            the resident core (``build_resident_index``).
        pos_offset (nl.NkiTensor): [1, 1] int32 @ HBM, or None. A per-rank shift added to every live
            selected position when the index tables are built; ignored with prebuilt tables.
        positions_km (nl.NkiTensor): [num_seqs * P_MAX, spec * num_prior_tiles] int32 @ HBM, or None.
            The selection already key-major, loaded instead of ``gather_indices`` when building the
            tables. Ignored with prebuilt tables.
        ctx_lse_out (nl.NkiTensor): [T * Nh, chunk] bf16 @ HBM, or None. If given, the context and
            lse go straight into this head-major cell table, the fused CP combine's all_to_all
            payload, which saves re-staging the context through HBM: row ``n * T + t`` holds token
            ``t``'s head ``n``, its ``C`` context values, then its fp32 lse as raw bits in bf16 slots
            ``[C, C + 2)``, then padding. ``chunk >= C + 2``. ``ctx`` and ``lse`` are then None.
        prior_tables_out (list): Optional. When the kernel builds the index tables itself, they are
            also exported to HBM and appended as ``(snake_hbm, key_pos_hbm, natural_hbm)`` for the
            layers sharing this selection (see ``build_shared_prior_index``). The export is three
            small DMAs off the critical path.
        heads_blocked (int): 1 (default) takes the layout above. ``g > 1`` takes the queries
            rank-major as the head gather left them: ``q_absorbed`` [g * P_MAX, C/P_MAX, T, Nh/g] and
            ``q_pe`` [g * R, T, Nh/g], head ``r * Nh/g + h`` in block ``r`` column ``h``.
        ctx_lse_token_split (int): ``g > 1`` lays ``ctx_lse_out`` out in ``g`` token pieces for the
            two-stage CP combine: row ``(t // Tq) * Nh * Tq + n * Tq + t % Tq`` (``Tq = T / g``)
            holds token ``t``'s head ``n``. 1 (default) is the layout above.
        resident_tables (tuple): Internal. None is the entry, which builds the resident tables and runs both
            calls. ``(rows_c, key_pos_c, n_tiles_c)`` runs the resident-only core over these SBUF tables; the
            fallback passes the full SBUF tables ``(snake, key_pos, num_prior_tiles)`` the same way.
        kv_update (bool): Internal. False skips the cache write (the fallback; the first call wrote it).
        q_hook (callable): ATTN_Q_HOOK. With ``q_absorbed = q_pe = None``, called after pair 0's
            gathers and key transposes are emitted; returns ``(q_absorbed, q_pe, heads_blocked)``.
            ``q_shape`` then gives the head-innermost ``q_absorbed`` shape for the extents.

    Returns:
        ctx (nl.NkiTensor): [T, Nh, C] @ HBM, the softmax-normalized latent context over this rank's
            keys. Normalized so it stays bounded in bf16; the CP combine reweights it by
            ``exp(lse_r - lse_max)``.
        lse (nl.NkiTensor): [T, Nh] fp32 @ HBM, ``max + log(sum(exp(score - max)))`` over this
            rank's keys. A rank that keeps no key of a token yields a large negative but finite lse
            (the mask is the finite bias -MASK_BOUND); the combine weights it by 0, since the rank
            owning the token's position always keeps its own row.
        kv_cache (nl.NkiTensor): the input cache, returned so its mutation is part of the result.
    """
    if resident_tables is None:
        return _resident_attention(
            q_absorbed, q_pe, kv_cache, kv_row_new, new_slot_mapping, gather_indices, block_table, positions,
            softmax_scale, kv_scale, spec, kv_lora_rank, cp_degree, cp_rank, prior_snake_rows, prior_key_pos,
            prior_natural_rows, pos_offset, positions_km, ctx_lse_out, prior_tables_out, heads_blocked,
            ctx_lse_token_split, q_hook, q_shape,
        )
    hb = heads_blocked
    # The shape helper reads the head count off a head-innermost query: give it that view of a blocked one.
    q_shape_src = q_absorbed
    if q_hook is not None:
        kernel_assert(q_absorbed is None and q_pe is None and q_shape is not None and resident_tables is not False,
                      "q_hook replaces q_absorbed / q_pe and needs q_shape (not in the resident fallback)")
        q_shape_src = nl.ndarray(q_shape, dtype=kv_row_new.dtype, buffer=nl.sbuf)  # extents only, never accessed
    elif hb > 1:
        q_shape_src = q_absorbed.reshape(
            (q_absorbed.shape[0] // hb, q_absorbed.shape[1], q_absorbed.shape[2], hb * q_absorbed.shape[3])
        )
    mla = get_mla_decode_shape(
        kv_cache,
        block_table,
        gather_indices,
        q_shape_src,
        positions,
        kv_row_new,
        kv_lora_rank,
        spec,
        cp_degree,
        cp_rank,
    )
    if resident_tables:
        mla = _resident_shape(mla, resident_tables[2])

    C, R = mla.kv_lora_dim, mla.qk_rope_dim
    T, Nh = mla.num_tokens, mla.num_heads
    Sk = mla.num_keys
    kv_row = mla.kv_row_dim
    row_b, rope_b = mla.kv_row_bytes, mla.rope_bytes
    nC, nS = mla.num_latent_tiles, mla.num_key_tiles
    n_prior_tiles = mla.num_prior_tiles
    io_dtype = q_shape_src.dtype

    def check_q(q_absorbed, q_pe, hb):
        if hb == 1:
            kernel_assert(
                q_absorbed.shape == (P_MAX, nC, T, Nh),
                f"q_absorbed must be ({P_MAX}, {nC}, {T}, {Nh}), got {q_absorbed.shape}",
            )
            kernel_assert(
                q_pe.shape == (P_MAX, T, Nh),
                f"q_pe must be ({P_MAX}, {T}, {Nh}), got {q_pe.shape}",
            )
        else:
            hl = Nh // hb
            kernel_assert(
                Nh % hb == 0 and q_absorbed.shape == (hb * P_MAX, nC, T, hl) and q_pe.shape == (hb * R, T, hl),
                f"heads_blocked={hb}: q_absorbed must be ({hb * P_MAX}, {nC}, {T}, {hl}) and q_pe ({hb * R}, {T}, {hl}), "
                f"got {q_absorbed.shape} / {q_pe.shape}",
            )

    if q_hook is None:
        check_q(q_absorbed, q_pe, hb)
    kernel_assert(
        kv_row_new.shape == (mla.num_seqs, spec, kv_row),
        f"kv_row_new must be ({mla.num_seqs}, {spec}, {kv_row}), got {kv_row_new.shape}",
    )

    if ctx_lse_out is None:
        ctx_out = nl.ndarray((T, Nh, C), dtype=io_dtype, buffer=nl.shared_hbm)
        # private_hbm: the CP layer all-gathers this directly, and at LNC=1 the runtime rejects a
        # collective whose source is shared_hbm.
        lse_out = nl.ndarray((T, Nh), dtype=nl.float32, buffer=nl.private_hbm)
        cell = 0
    else:
        ctx_out = lse_out = None
        cell = ctx_lse_out.shape[1]
        kernel_assert(
            tuple(ctx_lse_out.shape[:1]) == (T * Nh,)
            and cell >= C + 2
            and cell % 2 == 0
            and ctx_lse_out.dtype == io_dtype,
            f"ctx_lse_out must be [{T * Nh}, >= {C + 2} (even)] {io_dtype}, got {ctx_lse_out.shape} "
            f"{ctx_lse_out.dtype}",
        )
        kernel_assert(
            ctx_lse_token_split >= 1 and T % ctx_lse_token_split == 0,
            f"ctx_lse_token_split={ctx_lse_token_split} must divide the {T} tokens",
        )

    # Quantize this step's new rows; they are written into the cache after the pair loop.
    packed_new = _quantize_new_rows(kv_row_new, kv_scale, mla)
    # Stage them in HBM for the in-band append, which needs sibling ``i`` on partition ``i`` while
    # ``packed_new`` holds sequence ``s``'s sibling ``i`` on partition ``s * spec + i``: a partition
    # shift the lane-locked compute engines cannot do. One DMA per call gives the append a source it
    # can read at any partition offset.
    n_new = mla.num_new_rows
    new_rows_hbm = nl.ndarray((mla.num_seqs, spec, row_b), dtype=nl.uint8, buffer=nl.private_hbm)
    rows_staged = nisa.dma_copy(
        dst=new_rows_hbm.reshape((n_new, row_b))[0:n_new, 0:row_b], src=packed_new[0:n_new, 0:row_b], **_loop_dge()
    )

    # Resolve the global selection to local rows, or take the caller's tables.
    if resident_tables:
        snake_rows, key_pos_f, natural_rows = resident_tables[0], resident_tables[1], None
    elif prior_snake_rows is None:
        snake_rows, key_pos_f, natural_rows = build_prior_index(
            gather_indices, block_table, mla, pos_offset=pos_offset, positions_km=positions_km
        )
        if prior_tables_out is not None:
            prior_tables_out.append(export_prior_index(snake_rows, key_pos_f, natural_rows, name="layer_prior_index"))
    else:
        kernel_assert(prior_key_pos is not None, "prior_snake_rows and prior_key_pos must be passed together")
        snake_rows, key_pos_f, natural_rows = import_prior_index(
            prior_snake_rows, prior_key_pos, prior_natural_rows, mla, need_natural=False, **_loop_dge()
        )

    # The queries land in SBUF once, in the feature-major layout the score matmuls index in place.
    Q = {}  # q_absorbed_sb, q_pe_sb and the load handle, once loaded (load_q)

    def load_q(q_absorbed, q_pe, hb):
        Q["abs"], Q["pe"], Q["load"] = _load_queries(q_absorbed, q_pe, hb, nC, T, Nh, R, io_dtype)

    def ensure_q():
        if not Q:
            qa, qp, hq = q_hook()
            check_q(qa, qp, hq)
            load_q(qa, qp, hq)

    if q_hook is None:
        load_q(q_absorbed, q_pe, hb)

    # Token-major gathered rows, two slots per pair slot (token A, token B). uint8 because a packed row
    # mixes dtypes.
    kernel_assert(Nh <= _PAIR_BAND, f"num_heads={Nh} must fit one PE column band of {_PAIR_BAND}")
    n_pairs = div_ceil(T, 2)
    n_pair_slots = min(_PAIR_SLOTS, n_pairs)
    n_slots = 2 * n_pair_slots
    rows_sb = nl.ndarray((P_MAX, n_slots, nS, row_b), dtype=nl.uint8, buffer=nl.sbuf)
    # Only the last (sibling) tile needs zeroing. The prior tiles are fully overwritten for every
    # token, since _resolve_local_rows clamps every row into range and the gather never skips a slot.
    # The sibling tile is written on partitions [0, spec) only, but the context matmul streams all 128
    # rows, where a stale NaN/Inf byte pair would give 0 * Inf = NaN; 0x00 is +0.0 as fp8 and bf16.
    nisa.memset(dst=rows_sb[0:P_MAX, 0:n_slots, nS - 1, 0:row_b], value=0.0)
    # Both views are taken once on the whole allocation: the narrowing uint8 -> bf16 view has offset
    # and stride divisibility rules a strided sub-slice would have to satisfy again.
    rows_bf = rows_sb.view(nl.bfloat16)  # [P_MAX, n_slots, nS, row_b // 2]
    rows_f8 = rows_sb.view(nl.float8_e4m3fn)  # [P_MAX, n_slots, nS, row_b]

    # Feature-major keys for the score matmul, one buffer per token in flight so the next pair's
    # transposes do not wait on this pair's last matmul. The latent buffer is bf16-typed but holds fp8
    # bytes: the transpose moves 16-bit words, so 512 latent bytes travel as 256 words in two 128-column
    # tiles (see kv_row_layout.latent_byte_offset), and the score matmul reads them through an fp8 view.
    # Each half is ``Sk_full = nS * P_MAX`` words wide, every tile at full width so the ragged sibling
    # tile is scored full width too; latent half ``jw`` is words [jw * Sk_full, (jw + 1) * Sk_full).
    Sk_full = nS * P_MAX
    n_bufs = n_slots
    kv_res_all = nl.ndarray((P_MAX, n_bufs, 2 * Sk_full), dtype=nl.bfloat16, buffer=nl.sbuf)

    def lat_off(jw: int, key: int) -> int:
        """Word offset within a slot of (latent half ``jw``, key ``key``)."""
        return jw * Sk_full + key

    kpe_res_all = nl.ndarray((P_MAX, n_bufs, Sk_full), dtype=nl.bfloat16, buffer=nl.sbuf)
    # ATTN_SIB_ONCE needs one sequence (shared sibling rows) and a sibling tile of its own.
    sib_once = ATTN_SIB_ONCE and mla.num_seqs == 1 and nS > n_prior_tiles
    # Key columns past the last real key are never transposed into, but the score matmuls stream every
    # bank at full width, so zero that tail once (rope and both latent halves) and let the bias row
    # alone decide those columns.
    if Sk_full > Sk:
        nisa.memset(dst=kpe_res_all[0:R, 0:n_bufs, Sk:Sk_full], value=0.0)
        for jw in range(2):
            nisa.memset(
                dst=kv_res_all[0:P_MAX, 0:n_bufs, lat_off(jw, Sk) : lat_off(jw, Sk) + (Sk_full - Sk)], value=0.0
            )
    # Rope key rows above the bias row are never written: zero them so the rope+bias matmul contracts
    # over all 128 partitions (ones on the query side, so the padding is exact) instead of R + 1. A
    # sub-128 contraction is reported to trip the 50%-duty di/dt throttle. The memset starts at R, a
    # legal partition start; row R is the bias row, which each token's bias DMA rewrites later.
    kernel_assert(R % 32 == 0, f"rope_dim={R} must be a multiple of 32 to zero the rows above it in one access")
    nisa.memset(dst=kpe_res_all[R:P_MAX, 0:n_bufs, 0:Sk_full], value=0.0)

    # Head-major scores and weights: head ``h`` of key ``si * P_MAX + p`` sits at ``[h, si * P_MAX + p]``,
    # one tile per pair slot (token A on partitions [0, Nh), token B on [64, 64 + Nh)). With heads on
    # partitions the query is the score matmul's stationary, loaded once per contraction chunk rather
    # than per key tile, the keys stream as the moving operand at full tile width, and the softmax is a
    # per-partition free-axis reduction. The cost is transposing the weights to key-major for the
    # context sum; they are real bf16 values, so those evictions need no bit-accurate engine. Every key
    # tile is written at full width, so the ragged sibling tile's unused columns are defined.
    w_hm = nl.ndarray((P_MAX, n_pair_slots, Sk_full), dtype=io_dtype, buffer=nl.sbuf)
    if Nh < _PAIR_BAND or T % 2:
        # The transposes read all 128 rows, but the rows above Nh, and token B's half of an odd T's last pair, are
        # never written. Zero the whole tile once, since a memset must start on partition 0 or 64.
        nisa.memset(dst=w_hm[0:P_MAX, 0:n_pair_slots, 0:Sk_full], value=0.0)
    # fp32 scores in SBUF, one pair per slot on the two partition halves: PSUM holds only the bank
    # being accumulated, so the next pair's matmuls can start while this pair's softmax reads from here.
    s_sb = nl.ndarray((P_MAX, n_pair_slots, Sk_full), dtype=nl.float32, buffer=nl.sbuf)
    # Key tiles per PSUM bank (2 KB = 512 fp32 per partition). No bank holds a single tile unless the
    # token has only one: that would be a score matmul with a moving free dim of exactly 128, also
    # reported to trip the di/dt throttle, so a trailing 4 + 1 becomes 3 + 2.
    tiles_per_bank = F_MAX // P_MAX
    bank_ranges = _split_banks(nS, tiles_per_bank)
    n_banks = len(bank_ranges)

    # Query positions replicated down all partitions (stride-0 partition pattern), so the key-major
    # keep test subtracts one without a per-token broadcast.
    pos_i = nl.ndarray((P_MAX, T), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=pos_i[0:P_MAX, 0:T],
        src=positions[0:T].reshape((1, T)).ap(pattern=[[0, P_MAX], [1, T]], offset=0),
        **_loop_dge(),
    )
    pos_f = nl.ndarray((P_MAX, T), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=pos_f[0:P_MAX, 0:T], src=pos_i[0:P_MAX, 0:T])

    # Sibling keep, built once: sibling key ``i`` of query token ``tok``'s sequence is kept iff
    #   causal: sibling ``i`` is at ``base + i`` and token ``j = tok % spec`` at ``base + j``, so
    #     ``key_pos <= pos`` is ``i <= j``;
    #   owned: the siblings are appended on every CP rank, and the combine is the global softmax only
    #     if the ranks' key sets are disjoint, so only the rank owning ``base + i`` keeps it, as for a
    #     gathered key.
    # Ownership depends on the sequence's base, hence ``[spec, num_tokens]``. One iota (partition ramp
    # +1, free ramp -1 restarting every ``spec``) gives ``i - j``; adding the query position gives
    # ``base + i``.
    n_seqs = mla.num_seqs
    delta_3d = nl.ndarray((P_MAX, n_seqs, spec), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(
        delta_3d[0:spec, 0:n_seqs, 0:spec],
        pattern=[[0, n_seqs], [-1, spec]],
        offset=0,
        channel_multiplier=1,
    )
    delta = delta_3d.reshape((P_MAX, T))

    sib_keep = nl.ndarray((P_MAX, T), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=sib_keep[0:spec, 0:T], data=delta[0:spec, 0:T], op0=nl.less, operand0=0.5)

    # sib_pos = positions[tok] + (i - j) = base + i, since positions[tok] is base + j.
    sib_pos_f = nl.ndarray((P_MAX, T), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=sib_pos_f[0:spec, 0:T], data1=pos_f[0:spec, 0:T], data2=delta[0:spec, 0:T], op=nl.add)
    sib_pos_i = nl.ndarray((P_MAX, T), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=sib_pos_i[0:spec, 0:T], src=sib_pos_f[0:spec, 0:T])

    # owner = (position >> block_shift) & (cp_degree - 1), the same round-robin rule the prior index
    # uses, so the two paths agree on who owns a position.
    owner_i = nl.ndarray((P_MAX, T), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=owner_i[0:spec, 0:T],
        data=sib_pos_i[0:spec, 0:T],
        op0=nl.right_shift,
        operand0=mla.block_shift,
        op1=nl.bitwise_and,
        operand1=mla.cp_degree - 1,
    )
    owner_f = nl.ndarray((P_MAX, T), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=owner_f[0:spec, 0:T], src=owner_i[0:spec, 0:T])
    own_keep = nl.ndarray((P_MAX, T), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=own_keep[0:spec, 0:T], data=owner_f[0:spec, 0:T], op0=nl.equal, operand0=float(mla.cp_rank))

    nisa.tensor_tensor(
        dst=sib_keep[0:spec, 0:T], data1=sib_keep[0:spec, 0:T], data2=own_keep[0:spec, 0:T], op=nl.multiply
    )

    # Additive mask, one bf16 row of ``Sk_full`` biases per token. Keys are on the free axis, where a
    # per-key operand would need a [Nh, keys] broadcast tile, so the mask rides the score's PSUM
    # accumulation at no extra instructions: the rope matmul's extra contraction row is a one in the
    # query (partition R of ``q_pe_sb``) against the token's bias row (partition R of ``kpe_res``),
    # 0 for a kept key and -MASK_BOUND for a dropped one. That matmul runs first for each bank at full
    # width, so the sibling tile's ragged tail is a defined -MASK_BOUND.
    #
    # Adding is safe because every score is finite and bounded (fp8 keys, bf16 queries, a 576-long
    # contraction): ``score - 1e30`` is ``-1e30`` in fp32 and its exp is exactly 0. The bias is bf16
    # because an fp32 matmul runs the array in a slower multi-pass mode. The rows depend only on
    # ``key_pos``, ``positions`` and the sibling keep, so they are built once here: keep test
    # key-major, bf16 conversion, PE transposes to row order, and a copy to HBM from which each token
    # loads its row with one contiguous DMA.
    keep_km = nl.ndarray((P_MAX, T, nS), dtype=nl.float32, buffer=nl.sbuf)
    for tok in range(T):
        # keep = (key_pos <= positions[tok]) for the prior tiles. Non-resident and unselected slots
        # carry NON_RESIDENT_POS, so they always fail; integers are exact in fp32 at these magnitudes,
        # so ``diff < 0.5`` is the reference's non-strict ``<=``.
        nisa.tensor_scalar(
            dst=keep_km[0:P_MAX, tok, 0:n_prior_tiles],
            data=key_pos_f[0:P_MAX, tok, 0:n_prior_tiles],
            op0=nl.subtract,
            operand0=pos_f[0:P_MAX, tok : tok + 1],
            op1=nl.less,
            operand1=0.5,
        )
    # The sibling column: this token's sibling keep on partitions [0, spec), zero past it so the ragged
    # tail of that tile is dropped.
    nisa.memset(dst=keep_km[0:P_MAX, 0:T, n_prior_tiles:nS], value=0.0)
    nisa.tensor_copy(dst=keep_km[0:spec, 0:T, n_prior_tiles:nS], src=sib_keep[0:spec, 0:T].reshape((spec, T, 1)))
    # keep * MASK_BOUND - MASK_BOUND is exactly 0 for a kept key and -MASK_BOUND (rounded to bf16, still
    # ~1e30) otherwise; the arithmetic runs in fp32 and only the result is narrowed.
    bias_km = nl.ndarray((P_MAX, T * nS), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=bias_km[0:P_MAX, 0 : T * nS],
        data=keep_km.reshape((P_MAX, T * nS))[0:P_MAX, 0 : T * nS],
        op0=nl.multiply,
        operand0=MASK_BOUND,
        op1=nl.subtract,
        operand1=MASK_BOUND,
    )
    # Transpose to row order, (tok, tile) on partitions, then copy to HBM as ``bias_rows[tok, si, p]`` so
    # a token's Sk_full biases are contiguous.
    bias_rows = nl.ndarray((T * nS, P_MAX), dtype=nl.bfloat16, buffer=nl.private_hbm)
    for ch in TiledRange(T * nS, P_MAX):
        cs, ce, csz = ch.start_offset, ch.end_offset, ch.size
        bT_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.psum)
        nisa.nc_transpose(bT_psum[0:csz, 0:P_MAX], bias_km[0:P_MAX, cs:ce])
        bT_sb = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=bT_sb[0:csz, 0:P_MAX], src=bT_psum[0:csz, 0:P_MAX], engine=nisa.engine.scalar)
        nisa.dma_copy(dst=bias_rows[cs:ce, 0:P_MAX], src=bT_sb[0:csz, 0:P_MAX], **_loop_dge())
    bias_rows_flat = bias_rows.reshape((T, Sk_full))

    # The dequant rides the softmax scale: the cache holds latent/kv_scale and k_pe/kv_scale, so one
    # scalar undoes both halves of the score.
    scale = softmax_scale * kv_scale
    # Scaling the scores at their eviction is exact only for a power-of-two scale: then
    # scale * max(s) == max(scale * s), and scale * s - scale * m is what an unfused exp would see.
    kernel_assert(
        scale > 0 and math.frexp(scale)[0] == 0.5,
        f"softmax_scale * kv_scale = {scale} must be a power of two (the eviction-folded softmax is exact only then)",
    )

    # Per pair of query tokens (see _PAIR_SLOTS): each token gathers its own selection and builds its
    # own feature-major keys; the pair then shares the score PSUM tile, one softmax, one set of weight
    # transposes and one column-banded context sum.
    def prep_token(tok: int, slot: int) -> None:
        """Gather token ``tok``'s rows into slot ``slot`` and build its feature-major keys + bias row there."""
        # Tokens 0-1 lead the core once this step's rows are staged; the rest wait for the query.
        gather = gather_prior_rows(rows_sb, slot, kv_cache, snake_rows, tok, mla)
        gather.depends_on(Q["load"] if tok >= 2 else rows_staged)
        # ATTN_SIB_ONCE: a reused slot already holds this sequence's sibling rows.
        if not (sib_once and tok >= n_slots):
            _append_spec_rows(rows_sb, slot, new_rows_hbm, tok, mla)
        kpe_res = kpe_res_all[0:P_MAX, slot]

        # Feature-major keys. The gather leaves rows token-major, as the context sum wants, so this is
        # the only transpose of key data. The latent moves as bf16 words over fp8 bytes, which can form
        # bf16 subnormals (see E4M3FN_MAX), so the transpose and eviction must be bit-accurate: the PE
        # transpose is, and the eviction is pinned to Vector, the documented bit-accurate PSUM reader.
        # Full prior tiles go in groups of _KEY_EVICT_GROUP, one eviction per group and part.
        G = _KEY_EVICT_GROUP
        grouped_upto = (n_prior_tiles // G) * G
        for g0 in range(0, grouped_upto, G):
            c0, c1 = g0 * P_MAX, (g0 + G) * P_MAX
            kpe_psum_g = nl.ndarray((P_MAX, G * P_MAX), dtype=nl.bfloat16, buffer=nl.psum)
            ckv_psum_g = [nl.ndarray((P_MAX, G * P_MAX), dtype=nl.bfloat16, buffer=nl.psum) for _ in range(2)]
            for k in range(G):
                si = g0 + k
                nisa.nc_transpose(kpe_psum_g[0:R, k * P_MAX : (k + 1) * P_MAX], rows_bf[0:P_MAX, slot, si, 0:R])
                for jw in range(2):
                    w0 = rope_b // 2 + 128 * jw
                    nisa.nc_transpose(
                        ckv_psum_g[jw][0:P_MAX, k * P_MAX : (k + 1) * P_MAX], rows_bf[0:P_MAX, slot, si, w0 : w0 + 128]
                    )
            nisa.tensor_copy(dst=kpe_res_all[0:R, slot, c0:c1], src=kpe_psum_g[0:R, 0 : G * P_MAX])
            for jw in range(2):
                nisa.tensor_copy(
                    dst=kv_res_all[0:P_MAX, slot, lat_off(jw, c0) : lat_off(jw, c0) + G * P_MAX],
                    src=ckv_psum_g[jw][0:P_MAX, 0 : G * P_MAX],
                    engine=nisa.engine.vector,
                )
        for st in TiledRange(Sk, P_MAX):
            ss, se, ssz = st.start_offset, st.end_offset, st.size
            si = st.index
            if si < grouped_upto:
                continue
            # ATTN_SIB_ONCE: the sibling tile is transposed by token 0 alone and evicted into every key slot.
            sib = sib_once and si == n_prior_tiles
            if sib and tok > 0:
                continue
            ev_slots = range(n_bufs) if sib else [slot]
            kpe_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.psum)
            nisa.nc_transpose(kpe_psum[0:R, 0:ssz], rows_bf[0:ssz, slot, si, 0:R])
            for es in ev_slots:
                nisa.tensor_copy(dst=kpe_res_all[0:R, es, ss:se], src=kpe_psum[0:R, 0:ssz])
            for jw in range(2):
                w0 = rope_b // 2 + 128 * jw  # bf16 word offset of this half's first word
                ckv_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.psum)
                nisa.nc_transpose(ckv_psum[0:P_MAX, 0:ssz], rows_bf[0:ssz, slot, si, w0 : w0 + 128])
                for es in ev_slots:
                    nisa.tensor_copy(
                        dst=kv_res_all[0:P_MAX, es, lat_off(jw, ss) : lat_off(jw, ss) + ssz],
                        src=ckv_psum[0:P_MAX, 0:ssz],
                        engine=nisa.engine.vector,
                    )

        # This token's bias row goes to partition R of its rope key buffer, the extra contraction row of
        # the rope matmul.
        nisa.dma_copy(
            dst=kpe_res[R : R + 1, 0:Sk_full], src=bias_rows_flat[tok : tok + 1, 0:Sk_full], dge_mode=dge_mode.hwdge
        )

    band_a = dict(tile_position=(0, 0), tile_size=(P_MAX, _PAIR_BAND))
    band_b = dict(tile_position=(0, _PAIR_BAND), tile_size=(P_MAX, _PAIR_BAND))
    lse_flat = lse_out.reshape((T * Nh, 1)) if lse_out is not None else None

    def pair_begin(pair: int) -> dict:
        """Query-token pair ``pair``: its tokens, buffer slots, partition bases, PE bands and partition ranges."""
        pslot = pair % n_pair_slots
        tok_a = 2 * pair
        has_b = tok_a + 1 < T
        toks = [tok_a, tok_a + 1] if has_b else [tok_a]
        # Partition base of each token and the PE column band it computes in. A lone trailing token
        # (odd T) uses the whole array and partitions [0, Nh), untiled.
        bases = [0, _PAIR_BAND]
        # Partition ranges for the per-partition stages: one range when the two tokens' heads abut
        # (Nh == 64), else one per token.
        abut = has_b and Nh == _PAIR_BAND
        return dict(
            pslot=pslot,
            tok_a=tok_a,
            toks=toks,
            slots=[2 * pslot, 2 * pslot + 1],
            bases=bases,
            bands=[band_a, band_b] if has_b else [{}],
            abut=abut,
            ranges=[(0, 2 * Nh)] if abut else [(bases[k], bases[k] + Nh) for k in range(len(toks))],
        )

    def emit_prep(P: dict) -> None:
        for k, tok in enumerate(P["toks"]):
            prep_token(tok, P["slots"][k])

    def emit_scores(P: dict) -> None:
        """
        Head-major scores for both tokens of the pair, one PSUM bank at a time: ``s[b * 64 + h, key]``.

        The query is the stationary (M = Nh) and the feature-major keys stream as the moving operand,
        one matmul per (token, contraction chunk, bank) over the whole bank, so no matmul has a moving
        free dim of exactly 128 or a contraction under 128 (both di/dt throttle triggers). The two
        tokens' matmuls run concurrently in the two PE column bands and write the two partition halves
        of the same PSUM tile. The rope+bias matmul initialises each half at full width, the latent
        chunks accumulate on top, and the finished bank is evicted to ``s_sb`` with its max in one
        instruction while the next bank computes. Holding one bank at a time leaves PSUM room for the
        next pair to start before this pair's softmax has drained.

        The latent is read as fp8 through a view of the bf16 word buffer; thanks to the write-time byte
        permutation, partition p of chunk ``ci`` is latent dim ``128 * ci + p``. A bf16 stationary with
        an fp8 moving is allowed (only fp32 must pair with fp32). The view is taken on the per-token
        2-D slice: on the 3-D buffer a free-axis slice would index slots rather than words and read
        another token's keys.
        """
        ensure_q()
        q_absorbed_sb, q_pe_sb = Q["abs"], Q["pe"]
        pslot, toks, slots, bases, bands = P["pslot"], P["toks"], P["slots"], P["bases"], P["bands"]
        kv_f8 = [kv_res_all[0:P_MAX, slots[k]].view(nl.float8_e4m3fn) for k in range(len(toks))]
        m_parts = nl.ndarray((P_MAX, n_banks), dtype=nl.float32, buffer=nl.sbuf)
        for bank, (t0, t1) in enumerate(bank_ranges):
            c0 = t0 * P_MAX
            c1 = t1 * P_MAX
            ncol = c1 - c0
            s_psum = nl.ndarray((P_MAX, F_MAX), dtype=nl.float32, buffer=nl.psum)
            for k, tok in enumerate(toks):
                pb = bases[k]
                # Rope + bias first, over the whole bank and all 128 partitions: rows [0, R) are the rope
                # features, row R is ones against the bias row, rows above are ones against zeros.
                nisa.nc_matmul(
                    dst=s_psum[pb : pb + Nh, 0:ncol],
                    stationary=q_pe_sb[0:P_MAX, tok, 0:Nh],
                    moving=kpe_res_all[0:P_MAX, slots[k], c0:c1],
                    accumulate=False,
                    **bands[k],
                )
                for ci in range(nC):
                    jw, parity = ci // 2, ci % 2
                    base = 2 * lat_off(jw, c0) + parity
                    nisa.nc_matmul(
                        dst=s_psum[pb : pb + Nh, 0:ncol],
                        stationary=q_absorbed_sb[0:P_MAX, ci, tok, 0:Nh],
                        moving=kv_f8[k].slice(1, start=base, end=base + (ncol - 1) * 2 + 1, step=2),
                        accumulate=True,
                        **bands[k],
                    )
            # Evict the finished bank (both tokens), scaled, and take its max in the same Vector instruction, off
            # the PE's critical path.
            for p0, p1 in P["ranges"]:
                nisa.tensor_scalar_reduce(
                    dst=s_sb[p0:p1, pslot, c0:c1],
                    data=s_psum[p0:p1, 0:ncol],
                    op0=nl.multiply,
                    operand0=scale,
                    reduce_op=nl.maximum,
                    reduce_res=m_parts[p0:p1, bank : bank + 1],
                    reduce_cmd=nisa.reduce_cmd.reset_reduce,
                )
        P["m_parts"] = m_parts

    def emit_lse_fold(P: dict, p0: int, p1: int) -> None:
        # lse = scale * m + log(l), for the cross-rank combine (m_col holds -scale * m).
        nisa.scalar_tensor_tensor(
            dst=P["lse_col"][p0:p1, 0:1],
            data=P["m_col"][p0:p1, 0:1],
            op0=nl.multiply,
            operand0=-1.0,
            op1=nl.add,
            operand1=P["lse_col"][p0:p1, 0:1],
        )

    def emit_softmax(P: dict) -> None:
        """
        Per-partition softmax, both tokens in each instruction.

        -scale * m comes from a negating max-reduce of the per-bank maxima, then
        w = exp(scale * s - scale * m) with the row sum fused into the same Scalar instruction.
        ``reduce_cmd`` must be passed: its default ``idle`` would leave ``reduce_res`` reading a stale
        accumulator. Masked scores are the finite -MASK_BOUND, so m, l and the lse stay finite even on a
        rank that keeps no key of the token.
        """
        pslot = P["pslot"]
        m_col = P["m_col"] = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        l_col = P["l_col"] = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        lse_col = P["lse_col"] = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        for p0, p1 in P["ranges"]:
            nisa.tensor_reduce(m_col[p0:p1, 0:1], nl.maximum, P["m_parts"][p0:p1, 0:n_banks], 1, negate=True)
            nisa.activation(
                w_hm[p0:p1, pslot, 0:Sk_full],
                nl.exp,
                s_sb[p0:p1, pslot, 0:Sk_full],
                bias=m_col[p0:p1, 0:1],
                scale=1.0,
                reduce_op=nl.add,
                reduce_res=l_col[p0:p1, 0:1],
                reduce_cmd=nisa.reduce_cmd.reset_reduce,
            )
            nisa.activation(lse_col[p0:p1, 0:1], nl.log, l_col[p0:p1, 0:1])

    def emit_lse_out(P: dict) -> None:
        toks, bases, lse_col = P["toks"], P["bases"], P["lse_col"]
        if ctx_lse_out is not None:
            # Into the cell table (see _ctx_lse_row): the lse's fp32 bits go, viewed as bf16, into the two
            # slots at column C.
            for k, tok in enumerate(toks):
                pb = bases[k]
                row, stride = _ctx_lse_row(tok, T, Nh, ctx_lse_token_split)
                nisa.dma_copy(
                    dst=ctx_lse_out.ap(pattern=[[stride * cell, Nh], [1, 2]], offset=row * cell + C),
                    src=lse_col[pb : pb + Nh, 0:1].view(io_dtype),
                    dge_mode=dge_mode.hwdge,
                )
        # Written as the pair's 2 * Nh-row slice when the halves abut, else one Nh-row slice per token.
        elif P["abut"]:
            tok_a = P["tok_a"]
            nisa.dma_copy(
                dst=lse_flat[tok_a * Nh : (tok_a + 2) * Nh, 0:1], src=lse_col[0 : 2 * Nh, 0:1], dge_mode=dge_mode.hwdge
            )
        else:
            for k, tok in enumerate(toks):
                pb = bases[k]
                nisa.dma_copy(
                    dst=lse_flat[tok * Nh : (tok + 1) * Nh, 0:1],
                    src=lse_col[pb : pb + Nh, 0:1],
                    dge_mode=dge_mode.hwdge,
                )

    def emit_softmax_tail(P: dict) -> None:
        """The lse fold and write, and the denominator's reciprocal, ahead of the pair's context sum."""
        for p0, p1 in P["ranges"]:
            emit_lse_fold(P, p0, p1)
        emit_lse_out(P)
        recip_col = P["recip_col"] = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        for p0, p1 in P["ranges"]:
            nisa.reciprocal(recip_col[p0:p1, 0:1], P["l_col"][p0:p1, 0:1])

    def emit_context(P: dict, wT_sb: nl.NkiTensor) -> None:
        _emit_context_sum(
            ctx_out,
            rows_f8,
            wT_sb,
            P["l_col"],
            P["toks"],
            P["slots"],
            P["bases"],
            P["bands"],
            P["ranges"],
            mla,
            io_dtype,
            kv_scale,
            ctx_lse_out,
            recip_col=P.get("recip_col"),
            ctx_lse_token_split=ctx_lse_token_split,
        )

    # Software-pipelined in groups of two pairs (A, B) so the PE never waits on a softmax:
    #
    #   prep(A) score(A) smax(A) prep(B) wT(A) score(B) smax(B) tail(A) ctx(A) wT(B) tail(B) ctx(B)
    #
    # A's softmax runs under B's key transposes and B's under A's context sum. Groups of two keep ctx(A)
    # two stages ahead of the next group's prep, whose gathers reuse A's row slots. A lone last pair
    # runs in plain order.
    for g in range(0, n_pairs, 2):
        if g + 1 == n_pairs:
            P = pair_begin(g)
            emit_prep(P)
            emit_scores(P)
            emit_softmax(P)
            emit_softmax_tail(P)
            emit_context(P, _emit_weight_transposes(w_hm, P["pslot"], mla, io_dtype))
            continue
        A, B = pair_begin(g), pair_begin(g + 1)
        emit_prep(A)
        emit_scores(A)
        emit_softmax(A)
        emit_prep(B)
        wT_a = _emit_weight_transposes(w_hm, A["pslot"], mla, io_dtype)
        emit_scores(B)
        emit_softmax(B)
        emit_softmax_tail(A)
        emit_context(A, wT_a)
        wT_b = _emit_weight_transposes(w_hm, B["pslot"], mla, io_dtype)
        emit_softmax_tail(B)
        emit_context(B, wT_b)

    if kv_update:
        # After every gather; no live selection reads these rows (see the schedule note at the top).
        _update_kv_cache(kv_cache, packed_new, new_slot_mapping, mla)
    return ctx_out, lse_out, kv_cache


def _load_queries(q_absorbed, q_pe, hb, nC, T, Nh, R, io_dtype):
    """Load the queries into SBUF in the score matmuls' layout; returns (q_absorbed_sb, q_pe_sb, load)."""
    if hb == 1:
        q_absorbed_sb = nl.ndarray((P_MAX, nC, T, Nh), dtype=io_dtype, buffer=nl.sbuf)
        # A caller-built SBUF query is copied on chip, not re-loaded.
        if q_absorbed.buffer == nl.sbuf:
            q_load = nisa.tensor_copy(
                dst=q_absorbed_sb[0:P_MAX, 0:nC, 0:T, 0:Nh], src=q_absorbed[0:P_MAX, 0:nC, 0:T, 0:Nh]
            )
        else:
            q_load = nisa.dma_copy(
                dst=q_absorbed_sb[0:P_MAX, 0:nC, 0:T, 0:Nh], src=q_absorbed[0:P_MAX, 0:nC, 0:T, 0:Nh], **_loop_dge()
            )
        q_pe_sb = nl.ndarray((P_MAX, T, Nh), dtype=io_dtype, buffer=nl.sbuf)
        # Ones everywhere, then the rope query over partitions [0, R): partition R stays ones, the extra
        # contraction row through which the mask bias enters the score.
        nisa.memset(dst=q_pe_sb[0:P_MAX, 0:T, 0:Nh], value=1.0)
        if q_pe.buffer == nl.sbuf:
            nisa.tensor_copy(dst=q_pe_sb[0:R, 0:T, 0:Nh], src=q_pe[0:R, 0:T, 0:Nh])
        else:
            nisa.dma_copy(dst=q_pe_sb[0:R, 0:T, 0:Nh], src=q_pe[0:R, 0:T, 0:Nh], **_loop_dge())
        return q_absorbed_sb, q_pe_sb, q_load
    # Rank-major (block r = heads [r * hl, (r + 1) * hl)): one DMA of 1 KB runs per tensor, then heads innermost on
    # chip, since a score stationary must be one free dim (the BIR verifier rejects the (r, h) view). Same-dtype
    # Vector copies, so bit-exact.
    hl = Nh // hb
    blk = nC * T * hl
    q_blk = nl.ndarray((P_MAX, hb, nC, T, hl), dtype=io_dtype, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=q_blk[0:P_MAX, 0:hb, 0:nC, 0:T, 0:hl],
        src=q_absorbed.ap(pattern=[[blk, P_MAX], [P_MAX * blk, hb], [1, blk]], offset=0),
        dge_mode=dge_mode.hwdge,
    )
    pe_blk = nl.ndarray((R, hb, T, hl), dtype=io_dtype, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=pe_blk[0:R, 0:hb, 0:T, 0:hl],
        src=q_pe.ap(pattern=[[T * hl, R], [R * T * hl, hb], [1, T * hl]], offset=0),
        dge_mode=dge_mode.hwdge,
    )
    q_absorbed_sb = nl.ndarray((P_MAX, nC, T, Nh), dtype=io_dtype, buffer=nl.sbuf)
    q_pe_sb = nl.ndarray((P_MAX, T, Nh), dtype=io_dtype, buffer=nl.sbuf)
    nisa.memset(dst=q_pe_sb[0:P_MAX, 0:T, 0:Nh], value=1.0)
    for r in range(hb):
        h0, h1 = r * hl, (r + 1) * hl
        nisa.tensor_copy(dst=q_pe_sb[0:R, 0:T, h0:h1], src=pe_blk[0:R, r, 0:T, 0:hl], engine=nisa.engine.vector)
        q_load = nisa.tensor_copy(
            dst=q_absorbed_sb[0:P_MAX, 0:nC, 0:T, h0:h1],
            src=q_blk[0:P_MAX, r, 0:nC, 0:T, 0:hl],
            engine=nisa.engine.vector,
        )
    return q_absorbed_sb, q_pe_sb, q_load


def _resident_shape(mla: MlaDecodeShape, n_tiles_c: int) -> MlaDecodeShape:
    """The shape of the resident-only call: the same step with ``n_tiles_c`` prior key tiles."""
    return MlaDecodeShape(
        num_seqs=mla.num_seqs, spec=mla.spec, num_heads=mla.num_heads, kv_lora_dim=mla.kv_lora_dim,
        qk_rope_dim=mla.qk_rope_dim, num_prior=n_tiles_c * P_MAX, num_blocks=mla.num_blocks,
        block_size=mla.block_size, max_blocks=mla.max_blocks, block_shift=mla.block_shift,
        cp_degree=mla.cp_degree, cp_rank=mla.cp_rank, dtype=mla.dtype,
    )


def _resident_attention(
    q_absorbed, q_pe, kv_cache, kv_row_new, new_slot_mapping, gather_indices, block_table, positions, softmax_scale,
    kv_scale, spec, kv_lora_rank, cp_degree, cp_rank, prior_snake_rows, prior_key_pos, prior_natural_rows, pos_offset,
    positions_km, ctx_lse_out, prior_tables_out, heads_blocked, ctx_lse_token_split, q_hook=None, q_shape=None,
):
    """
    The resident-only call, then the full core in a dynamic loop that runs only on overflow (a
    token keeping more than ``n_tiles_c * 128`` keys) and overwrites the outputs. The loop is fenced on every DMA of
    the first call, so the full core never reads a row or table that call is still writing.

    A full layer builds and exports the full tables here; a shared layer is handed them. The resident tables are
    built once per selection in the full layer and handed to the group's shared layers in SBUF through
    ``_RESIDENT_TABLES``, keyed by the full snake table the layer passes.
    """
    kernel_assert(ctx_lse_out is not None, "the resident-only core needs the fused ctx_lse_out payload")
    hb = heads_blocked
    q_shape_src = q_absorbed
    if q_hook is not None:
        q_shape_src = nl.ndarray(q_shape, dtype=kv_row_new.dtype, buffer=nl.sbuf)  # extents only, never accessed
    elif hb > 1:
        q_shape_src = q_absorbed.reshape(
            (q_absorbed.shape[0] // hb, q_absorbed.shape[1], q_absorbed.shape[2], hb * q_absorbed.shape[3])
        )
    mla = get_mla_decode_shape(
        kv_cache, block_table, gather_indices, q_shape_src, positions, kv_row_new, kv_lora_rank, spec, cp_degree,
        cp_rank,
    )
    n_tc = min(ATTN_RESIDENT_TILES, mla.num_prior_tiles)
    shape = (P_MAX, mla.num_tokens, mla.num_prior_tiles)
    shape_c = (P_MAX, mla.num_tokens, n_tc)
    compact = None
    if prior_snake_rows is None:
        snake_sb, key_pos_sb, natural_sb = build_prior_index(
            gather_indices, block_table, mla, pos_offset=pos_offset, positions_km=positions_km
        )
        snake_h, key_pos_h, natural_h = export_prior_index(snake_sb, key_pos_sb, natural_sb, name="layer_prior_index")
        if prior_tables_out is not None:
            prior_tables_out.append((snake_h, key_pos_h, natural_h))
    else:
        snake_h, key_pos_h, natural_h = prior_snake_rows, prior_key_pos, prior_natural_rows
        entry = _RESIDENT_TABLES.get(id(snake_h))
        if entry is not None and entry[0] is snake_h:
            compact = entry[1:4]
            snake_sb, key_pos_sb = entry[4], entry[5]  # the full tables in SBUF, for the fallback
        else:
            snake_sb, key_pos_sb, natural_sb = import_prior_index(snake_h, key_pos_h, natural_h, mla, need_natural=True)
    if compact is None:
        rows_c, key_pos_c, flag = build_resident_index(key_pos_sb, natural_sb, positions, mla, n_tc)
        _RESIDENT_TABLES[id(snake_h)] = (snake_h, rows_c, key_pos_c, flag, snake_sb, key_pos_sb)
    else:
        rows_c, key_pos_c, flag = compact
    kernel_assert(tuple(rows_c.shape) == shape_c and tuple(key_pos_h.shape) == shape, "resident table shapes")

    common = dict(
        q_absorbed=q_absorbed, q_pe=q_pe, kv_cache=kv_cache, kv_row_new=kv_row_new, new_slot_mapping=new_slot_mapping,
        gather_indices=gather_indices, block_table=block_table, positions=positions, softmax_scale=softmax_scale,
        kv_scale=kv_scale, spec=spec, kv_lora_rank=kv_lora_rank, cp_degree=cp_degree, cp_rank=cp_rank,
        ctx_lse_out=ctx_lse_out, heads_blocked=heads_blocked, ctx_lse_token_split=ctx_lse_token_split,
    )
    # Record every DMA of the resident-only call, so the loop below starts only after all of them complete.
    dmas = []
    real_dma = nisa.dma_copy

    def rec_dma(*a, **k):
        h = real_dma(*a, **k)
        dmas.append(h)
        return h

    nisa.dma_copy = rec_dma
    try:
        q_cap = []
        if q_hook is not None:
            # ATTN_Q_HOOK: the resident-only call runs the hook; the fallback reuses the queries it returned.
            def hook():
                q_cap.append(q_hook())
                return q_cap[-1]

            common.update(q_absorbed=None, q_pe=None)
            ctx_out, lse_out, kv_cache = mla_decode_attention(
                **common, resident_tables=(rows_c, key_pos_c, n_tc), q_hook=hook, q_shape=q_shape
            )
            common.update(q_absorbed=q_cap[0][0], q_pe=q_cap[0][1], heads_blocked=q_cap[0][2])
        else:
            ctx_out, lse_out, kv_cache = mla_decode_attention(**common, resident_tables=(rows_c, key_pos_c, n_tc))
    finally:
        nisa.dma_copy = real_dma
    fence_t = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    fence = nisa.tensor_copy(dst=fence_t[0:1, 0:1], src=flag[0:1, 0:1])
    for h in dmas:
        fence.depends_on(h)
    flag_reg = nisa.register_alloc()
    nisa.register_load(flag_reg, fence_t)

    # The fallback reads the full tables from SBUF, so no table read crosses the loop through HBM.
    sb_full = (snake_sb, key_pos_sb, mla.num_prior_tiles)

    def full_core():
        _LOOP_DGE.append(1)
        mla_decode_attention(
            **common, prior_snake_rows=snake_h, prior_key_pos=key_pos_h, prior_natural_rows=natural_h,
            resident_tables=sb_full, kv_update=False,
        )
        _LOOP_DGE.pop()

    tiles = _arena_record(full_core)  # outside the loop: the body allocates no SBUF / HBM tile of its own
    nl.fori_loop(0, flag_reg, lambda _i: _arena_replay(tiles, full_core))
    return ctx_out, lse_out, kv_cache


def _split_banks(n_tiles: int, tiles_per_bank: int) -> list:
    """Split ``n_tiles`` key tiles into PSUM banks of at most ``tiles_per_bank`` tiles, with no one-tile bank unless
    there is only one tile (e.g. 17 -> [4, 4, 4, 3, 2])."""
    ranges = []
    t = 0
    while t < n_tiles:
        rem = n_tiles - t
        n = min(tiles_per_bank, rem)
        if rem - n == 1 and n > 1:
            n -= 1
        ranges.append((t, t + n))
        t += n
    return ranges


def _emit_weight_transposes(w_hm: nl.NkiTensor, pslot: int, mla: MlaDecodeShape, io_dtype: nl.DType) -> nl.NkiTensor:
    """
    The pair's weights key-major, ``[P_MAX, nS, P_MAX]`` @ SBUF, for :func:`_emit_context_sum`.

    One full 128-row PE transpose per key tile per pair: columns [0, Nh) of the result are token A's
    weights and [64, 64 + Nh) token B's, each the stationary of that token's context matmul. They are
    real bf16 values, so any PSUM reader is exact; the eviction goes to Scalar, which the softmax
    leaves almost idle.
    """
    nS = mla.num_key_tiles
    # Full 128 keys per tile (the last tile's extra columns are zero weights), so the context matmul
    # contracts over K = 128 on every tile.
    wT_sb = nl.ndarray((P_MAX, nS, P_MAX), dtype=io_dtype, buffer=nl.sbuf)
    for si in range(nS):
        wT_psum = nl.ndarray((P_MAX, P_MAX), dtype=io_dtype, buffer=nl.psum)
        nisa.nc_transpose(wT_psum[0:P_MAX, 0:P_MAX], w_hm[0:P_MAX, pslot, si * P_MAX : (si + 1) * P_MAX])
        nisa.tensor_copy(dst=wT_sb[0:P_MAX, si, 0:P_MAX], src=wT_psum[0:P_MAX, 0:P_MAX], engine=nisa.engine.scalar)
    return wT_sb


def _ctx_lse_row(tok: int, T: int, Nh: int, split: int) -> tuple[int, int]:
    """Row of token ``tok``'s head 0 in the ``ctx_lse_out`` cell table and the row stride between its heads.

    ``split`` 1: ``(tok, T)``, the head-major table. ``split`` g: ``g`` token pieces of ``Tq = T / g`` tokens, each
    head-major: ``((tok // Tq) * Nh * Tq + tok % Tq, Tq)``.
    """
    Tq = T // split
    return (tok // Tq) * Nh * Tq + tok % Tq, Tq


def _emit_context_sum(
    ctx_out: nl.NkiTensor,
    rows_f8: nl.NkiTensor,
    wT_sb: nl.NkiTensor,
    l_col: nl.NkiTensor,
    toks: list,
    slots: list,
    bases: list,
    bands: list,
    ranges: list,
    mla: MlaDecodeShape,
    io_dtype: nl.DType,
    kv_scale: float,
    ctx_lse_out: nl.NkiTensor = None,
    recip_col: nl.NkiTensor = None,
    ctx_lse_token_split: int = 1,
) -> None:
    """
    ctx[Nh, C] = w[Nh, Sk] . kv_latent[Sk, C], normalized, for the one or two tokens of a pair.

    The contraction is over keys, so the latent is wanted token-major, as the gather left it: the fp8
    bytes feed the matmul straight from the packed row, with no transpose or dequantization pass (one
    over nS * C elements per token would cost as much as the rest of the kernel). x_psum's free axis is
    therefore in the row's permuted byte order; the denominator multiply un-permutes it.

    ``wT_sb`` is the key-major weights of :func:`_emit_weight_transposes`. ``recip_col`` is the
    denominator's reciprocal if the caller already took it; otherwise it is taken here.
    """
    C, Nh = mla.kv_lora_dim, mla.num_heads
    rope_b = mla.rope_bytes
    nS = mla.num_key_tiles
    T = mla.num_tokens

    x_psum = nl.ndarray((P_MAX, C), dtype=nl.float32, buffer=nl.psum)
    for si in range(nS):
        for k, tok in enumerate(toks):
            pb = bases[k]
            # K = 128 on every tile, the ragged last one included: past the real keys the weights are
            # exact zeros and the rows finite (the sibling tile is zeroed once), so the padding adds
            # nothing and avoids a sub-128 contraction.
            nisa.nc_matmul(
                dst=x_psum[pb : pb + Nh, 0:C],
                stationary=wT_sb[0:P_MAX, si, pb : pb + Nh],
                moving=rows_f8[0:P_MAX, slots[k], si, rope_b : rope_b + C],
                accumulate=(si > 0),
                **bands[k],
            )

    # Apply the deferred softmax denominator to x[Nh, C] rather than to w[Nh, Sk]: far fewer elements,
    # since Sk greatly exceeds C.
    x_sb = nl.ndarray((P_MAX, C), dtype=io_dtype, buffer=nl.sbuf)
    if recip_col is None:
        recip_col = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        for p0, p1 in ranges:
            nisa.reciprocal(recip_col[p0:p1, 0:1], l_col[p0:p1, 0:1])

    # Un-permute while normalizing: dims ``256a + 128b .. +127`` sit at byte-order positions from
    # ``256a + b`` with stride 2, so restoring latent order costs three extra instructions rather than a
    # data pass. The partition slice on ``data`` is required: tensor_scalar does not broadcast
    # partitions, and ``operand0`` must have one element per partition of ``data``.
    for p0, p1 in ranges:
        for a in range(2):
            for b in range(2):
                dst_off = 256 * a + 128 * b
                src_off = 256 * a + b
                # The latent dequant uses op1's free slot (the matmul contracted latent / kv_scale);
                # tensor_scalar costs the same with one operator or two.
                nisa.tensor_scalar(
                    dst=x_sb[p0:p1, dst_off : dst_off + 128],
                    data=x_psum.slice(1, start=src_off, end=src_off + 255, step=2)[p0:p1],
                    op0=nl.multiply,
                    operand0=recip_col[p0:p1, 0:1],
                    op1=nl.multiply,
                    operand1=kv_scale,
                )
    # Plain 2-D writes pinned to the hardware DGE; by default they lower to a software descriptor build
    # on GpSimd, which the indirect gathers already saturate. One write for the pair when the halves
    # abut, since ``ctx_out[tok_a : tok_a + 2]`` is ``[2 * Nh, C]`` contiguous.
    if ctx_lse_out is not None:
        # Cell table (see _ctx_lse_row), context in each cell's first C columns. One DMA per token.
        cell = ctx_lse_out.shape[1]
        for k, tok in enumerate(toks):
            pb = bases[k]
            row, stride = _ctx_lse_row(tok, T, Nh, ctx_lse_token_split)
            nisa.dma_copy(
                dst=ctx_lse_out.ap(pattern=[[stride * cell, Nh], [1, C]], offset=row * cell),
                src=x_sb[pb : pb + Nh, 0:C],
                dge_mode=dge_mode.hwdge,
            )
    elif len(toks) == 2 and Nh == _PAIR_BAND:
        nisa.dma_copy(
            dst=ctx_out.reshape((T * Nh, C))[toks[0] * Nh : (toks[0] + 2) * Nh, 0:C],
            src=x_sb[0 : 2 * Nh, 0:C],
            dge_mode=dge_mode.hwdge,
        )
    else:
        for k, tok in enumerate(toks):
            pb = bases[k]
            nisa.dma_copy(dst=ctx_out[tok, 0:Nh, 0:C], src=x_sb[pb : pb + Nh, 0:C], dge_mode=dge_mode.hwdge)


def _quantize_new_rows(kv_row_new: nl.NkiTensor, kv_scale: float, mla: MlaDecodeShape) -> nl.NkiTensor:
    """
    Pack this step's bf16 rows into the cache's byte layout, quantizing the latent to fp8 (clamped, see
    ``E4M3FN_MAX``). A handful of Vector ops once per call.

    Returns [P_MAX, kv_row_bytes] uint8 @ SBUF, row ``s * spec + i`` on partition ``s * spec + i``.
    """
    n_new, C, R = mla.num_new_rows, mla.kv_lora_dim, mla.qk_rope_dim
    row_b = mla.kv_row_bytes
    inv_scale = 1.0 / kv_scale

    src = nl.ndarray((P_MAX, R + C), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=src[0:n_new, 0 : R + C], src=kv_row_new.reshape((n_new, R + C))[0:n_new, 0 : R + C], **_loop_dge())

    packed = nl.ndarray((P_MAX, row_b), dtype=nl.uint8, buffer=nl.sbuf)
    # Partitions past n_new are never scattered, but zeroing keeps them finite in every view.
    nisa.memset(dst=packed[0:P_MAX, 0:row_b], value=0.0)
    packed_bf = packed.view(nl.bfloat16)
    packed_f8 = packed.view(nl.float8_e4m3fn)

    # k_pe, pre-divided by kv_scale so one score scalar undoes both halves. Exact in bf16 because
    # kv_scale is a power of two.
    nisa.activation(dst=packed_bf[0:n_new, 0:R], op=nl.copy, data=src[0:n_new, 0:R], scale=inv_scale)

    # Scale and clamp the upper end in one pass over the whole latent.
    tmp = nl.ndarray((P_MAX, C), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=tmp[0:n_new, 0:C],
        data=src[0:n_new, R : R + C],
        op0=nl.multiply,
        operand0=inv_scale,
        op1=nl.minimum,
        operand1=E4M3FN_MAX,
    )

    # Clamp the lower end and cast to fp8 in the permuted byte order: each op covers 128 latent dims at
    # destination stride 2, with offsets from kv_row_layout.
    for dst_byte, src_dim in latent_write_slices():
        nisa.tensor_scalar(
            dst=packed_f8.slice(1, start=dst_byte, end=dst_byte + 255, step=2)[0:n_new],
            data=tmp[0:n_new, src_dim : src_dim + 128],
            op0=nl.maximum,
            operand0=-E4M3FN_MAX,
        )
    return packed


def _update_kv_cache(
    kv_cache: nl.NkiTensor,
    packed_new: nl.NkiTensor,
    new_slot_mapping: nl.NkiTensor,
    mla: MlaDecodeShape,
) -> None:
    """
    Write this step's packed rows into the paged cache. A negative slot skips its write.

    This step's scoring takes the rows from the in-band append, never from this write, so no ordering
    between an indirect write and an indirect read of the same HBM tensor is relied on.
    """
    n_new = mla.num_new_rows
    row_b = mla.kv_row_bytes

    slots_i32 = nl.ndarray((P_MAX, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=slots_i32[0:n_new, 0:1], src=new_slot_mapping.reshape((n_new, 1))[0:n_new, 0:1])

    # A negative slot must become an out-of-range row before the indirect DMA: the int32 -> uint32
    # cast goes through fp32 and saturates, so -1 would clobber row 0. That is the common case: the
    # newest positions' block belongs to one CP rank, so on every other rank no new row has a home.
    #
    # The sentinel must be out of range for the allocation, not just this view: nki bounds an indirect
    # DMA by the parent allocation's storage shape, which ``[i]``, ``reshape`` and ``ap`` propagate. For
    # one layer of a stacked cache (``kv_cache_all[i]`` in the multi-layer decode kernel) the bound is
    # ``num_layers * num_cache_rows``, so ``mla.oob_row`` would land in the next layer's row 0. The
    # gather keeps ``mla.oob_row``, since its snake table is staged through uint16.
    storage_rows = int(math.prod(kv_cache._storage_shape)) // row_b
    oob_scatter = max(mla.oob_row, storage_rows)
    invalid = nl.ndarray((P_MAX, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(invalid[0:n_new, 0:1], slots_i32[0:n_new, 0:1], op0=nl.less, operand0=0.0)
    nisa.scalar_tensor_tensor(
        dst=slots_i32[0:n_new, 0:1],
        data=invalid[0:n_new, 0:1],
        op0=nl.multiply,
        operand0=float(oob_scatter),
        op1=nl.maximum,
        operand1=slots_i32[0:n_new, 0:1],
    )
    slots = nl.ndarray((P_MAX, 1), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=slots[0:n_new, 0:1], src=slots_i32[0:n_new, 0:1])

    # Both sides are uint8, so no cast: this 1-D indirect form, unlike the batched 2-D one, does not
    # require matching dtypes and would silently cast through fp32.
    cache_rows = kv_cache.reshape((mla.num_cache_rows, row_b))
    nisa.dma_copy(
        dst=cache_rows.ap(
            pattern=[[row_b, n_new], [1, row_b]],
            offset=0,
            vector_offset=slots[0:n_new, 0:1],
            indirect_dim=0,
        ),
        src=packed_new[0:n_new, 0:row_b],
        oob_mode=nisa.oob_mode.skip,
    )


def _append_spec_rows(
    rows_sb: nl.NkiTensor,
    slot: int,
    new_rows_hbm: nl.NkiTensor,
    tok: int,
    mla: MlaDecodeShape,
) -> None:
    """
    Append the ``spec`` packed new rows of token ``tok``'s sequence after its gathered prior.

    They land on partitions ``[0, spec)`` of key tile ``num_prior_tiles``, a tile of its own since
    num_prior is a whole number of tiles; the sibling keep then masks them. The source is the staged
    packed copy, so appended rows have exactly the layout of gathered ones.
    """
    spec, row_b = mla.spec, mla.kv_row_bytes
    seq = tok // spec
    tile = mla.num_prior_tiles
    nisa.dma_copy(
        dst=rows_sb[0:spec, slot, tile, 0:row_b],
        src=new_rows_hbm[seq, 0:spec, 0:row_b],
        **_loop_dge(),
    )


__all__ = ["mla_decode_attention"]
