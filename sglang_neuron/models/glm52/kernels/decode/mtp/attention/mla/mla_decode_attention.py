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
the output, decode is MQA over a 576-wide row and no per-head K or V is materialized:

    scores[t,h,s] = scale * ( q_absorbed[t,h,:C] . kv_latent[s,:C]
                            + q_pe[t,h,:R]      . k_pe[s,:R] )
    ctx[t,h,:C]   = softmax_s(scores[t,h,:]) . kv_latent[:,:C]

The rope half has no W_UK to fold through, so a cache row is ``cat([k_pe(R), kv_latent(C)])`` and
two matmuls accumulate into one PSUM tile per key tile.

The kernel stops at the latent ``ctx`` and its log-sum-exp. The CP ranks combine partials with the
LSE, and W_UV then runs on the post-combine head slice, so each rank holds only its share of W_UV;
o_proj also runs later. Both commute with the combine (W_UV is linear, the combine a weighted sum).

The token axis is flat over speculation, ``num_tokens = num_seqs * spec``, each token with its own
position and selection. The step's ``spec`` new latents are appended in band rather than gathered, so
no sibling is read back from a row this kernel just wrote.
"""

import math

import nki
import nki.isa as nisa
import nki.language as nl
from nki.isa import dge_mode

from ....attention.mla.kv_row_layout import E4M3FN_MAX, ROW_BYTES, latent_write_slices
from ....attention.mla.mla_decode_constants import F_MAX, MASK_BOUND, P_MAX, MlaDecodeShape, get_mla_decode_shape
from ....utils.kernel_assert import kernel_assert
from ....utils.kernel_helpers import div_ceil
from ....utils.tiled_range import TiledRange
from .prior_index import build_prior_index, export_prior_index, gather_prior_rows, import_prior_index

# Schedule: this step's rows are written into the cache after the pair loop, so the prior gathers do not depend on
# the scatter (which sits behind the query exchange's trigger on GpSimd). That is safe because no live selection
# reads a row written this step: the indexer masks keys at or after ``positions[0]``, dead slots resolve to
# NON_RESIDENT_POS, and this step's rows enter only through the in-band sibling append. Tokens 0-1 gather once this
# step's rows are staged, tokens 2+ once the queries have loaded. The softmax scale rides each score bank's eviction
# (exact for a power-of-two scale, asserted), and the pair loop is software-pipelined two pairs at a time so one
# pair's softmax overlaps the other's PE work.

_RANK_MAJOR_HEADS = {}
"""Hand-over of the absorbed query from the head gather (``mla_decode_q_kv_up._gather_absorbed_heads``), which leaves
it rank-major in the all-gather destination (``heads_blocked``) rather than relaying it out in 32-byte elements; the
core loads the blocks in 1 KB runs and puts heads innermost on chip.
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
Token pairs kept resident (gathered rows and feature-major keys in flight). Two lets pair p+1's gathers
and key transposes overlap pair p's softmax and context sum.

Tokens run two at a time on the two halves of the PE array and partition axis: token A's queries are
the stationary of PE column band 0 and its scores land on partitions [0, Nh), token B's use band 1 and
partitions [64, 64 + Nh). Each band streams its own token's keys, and the two bands run concurrently.
Everything after the scores is per partition, so one instruction serves both tokens, and the weight
transposes are one 128-row transpose per key tile per pair (a sub-128 transpose can trip the di/dt
throttle).

Bounded by SBUF: a pair slot holds two tokens' gathered rows (21.8 KB per partition at the production
shape) and feature-major keys (13 KB), plus the pair's fp32 scores (8.5 KB) and bf16 weights (4.3 KB).
"""

_PAIR_BAND = 64
"""Width of one PE column band = the partition offset of token B. Fixed by the array (2 x 64 column
tiles); ``num_heads`` must fit in it."""


class _PriorRowsShape:
    """The extents ``import_prior_index`` and ``gather_prior_rows`` read, without a full :class:`MlaDecodeShape`."""

    def __init__(self, num_tokens: int, num_prior_tiles: int, num_cache_rows: int):
        self.num_tokens = num_tokens
        self.num_prior_tiles = num_prior_tiles
        self.num_prior = num_prior_tiles * P_MAX
        self.num_cache_rows = num_cache_rows
        self.kv_row_bytes = ROW_BYTES


def _kv_free_slots(num_tokens: int) -> int:
    """Gathered-row slots of a ``kv_free`` call: the pair slots, but never more than there are tokens."""
    return min(2 * min(_PAIR_SLOTS, div_ceil(num_tokens, 2)), num_tokens)


def mla_decode_attention_prefetch(
    kv_cache: nl.NkiTensor, prior_snake_rows: nl.NkiTensor, prior_key_pos: nl.NkiTensor
) -> dict:
    """
    The query-independent head of a ``kv_free=2`` :func:`mla_decode_attention`, traced earlier so the prior tables'
    loads and the first tokens' gathers overlap the query projection instead of following its AllGather.

    Args:
        kv_cache (nl.NkiTensor): As for :func:`mla_decode_attention`.
        prior_snake_rows, prior_key_pos (nl.NkiTensor): [P_MAX, T, num_prior_tiles] @ HBM, as for
            :func:`mla_decode_attention` (the shared selection's tables).

    Returns:
        dict: Pass as ``prefetched`` to the :func:`mla_decode_attention` call with the same tables.
    """
    _, T, n_tiles = prior_snake_rows.shape
    shape = _PriorRowsShape(T, n_tiles, kv_cache.shape[0] * kv_cache.shape[2])
    snake_rows, key_pos_f, _ = import_prior_index(prior_snake_rows, prior_key_pos, None, shape, need_natural=False)
    n_slots = _kv_free_slots(T)
    # No sibling tile (kv_free=2), and every prior tile is fully overwritten by its gather: no memset.
    rows_sb = nl.ndarray((P_MAX, n_slots, n_tiles, ROW_BYTES), dtype=nl.uint8, buffer=nl.sbuf)
    # Token tok < n_slots takes slot tok in the pair loop (pair tok // 2, member tok % 2).
    for tok in range(n_slots):
        gather_prior_rows(rows_sb, tok, kv_cache, snake_rows, tok, shape)
    return dict(
        tables=(prior_snake_rows, prior_key_pos),
        snake_rows=snake_rows,
        key_pos_f=key_pos_f,
        rows_sb=rows_sb,
        gathered=n_slots,
    )


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
    attend_new_keys: bool = True,
    ctx_lse_out: nl.NkiTensor = None,
    prior_tables_out: list = None,
    heads_blocked: int = 1,
    kv_free: int = 0,
    prefetched: dict = None,
    query_rows: nl.NkiTensor = None,
) -> tuple[nl.NkiTensor, nl.NkiTensor, nl.NkiTensor]:
    """
    Score one decode step's queries against this rank's share of the selected keys.

    Usage:
        Tuned for: GLM-5.2 absorbed-MLA decode, one CP rank, LNC=1
        Applicable to: any absorbed-MLA decode over an externally selected, globally indexed key set

    Args:
        q_absorbed (nl.NkiTensor): [P_MAX, C/P_MAX, T, Nh] @ HBM, the query already through W_UK,
            feature-major: latent feature ``c`` of token ``t`` head ``h`` is at ``[c % P_MAX, c // P_MAX, t, h]``,
            the layout a fused front leaves in SBUF.
        q_pe (nl.NkiTensor): [P_MAX, T, Nh] @ HBM, the RoPE'd query half, feature-major, valid in the
            leading ``R`` partitions.
        kv_cache (nl.NkiTensor): [num_blocks, 1, block_size, 640] uint8 @ HBM, this rank's local paged cache
            of packed rows (bf16 k_pe then fp8 latent, see ``kv_row_layout``). Mutated: this step's rows are
            quantized and written after every gather, and the tensor is returned.
        kv_row_new (nl.NkiTensor): [num_seqs, spec, R + C] bf16 @ HBM, this step's new rows, unquantized;
            the kernel quantizes and packs them.
        new_slot_mapping (nl.NkiTensor): [num_seqs, spec] int32 @ HBM, the flat local cache row each new
            row is written to, or negative to skip the write.
        gather_indices (nl.NkiTensor): [num_seqs, spec, num_prior] int32 @ HBM, global positions the
            selector chose, one row per (sequence, speculative token) as in the reference. Negative means
            not selected. Because each token has its own selection, the speculative tokens cannot share
            the score matmul's moving operand.
        block_table (nl.NkiTensor): [num_seqs, max_blocks] int32 @ HBM, this rank's local table.
        positions (nl.NkiTensor): [T] int32 @ HBM, each query token's absolute position. A sequence's
            ``spec`` tokens must occupy consecutive positions, which lets the sibling causal mask be an iota.
        softmax_scale (float): Score scale; GLM-5.2 uses ``qk_head_dim ** -0.5`` = 0.0625 and default RoPE,
            so there is no yarn mscale.
        kv_scale (float): Dequantization scale of the packed cache, which stores ``latent / kv_scale`` and
            ``k_pe / kv_scale``. Folded into ``softmax_scale`` for the scores and into the denominator for
            the context, so it costs no instructions. Must be a power of two so pre-dividing the bf16 k_pe
            is exact; 1.0 suits a layernorm-output latent, well inside fp8 e4m3's range.
        spec (int): Speculative tokens per sequence; 1 is plain decode.
        kv_lora_rank (int): Latent width C.
        cp_degree (int): Context-parallel degree; this rank holds every cp_degree'th logical block.
        cp_rank (int): This rank's index within its CP group.
        prior_snake_rows (nl.NkiTensor): [P_MAX, T, num_prior_tiles] uint32 @ HBM, or None. With
            ``prior_key_pos``, the index tables ``build_prior_index`` would build (see
            ``prior_index.export_prior_index``). They depend only on the selection, block table and CP
            shift, so layers sharing one selection build them once; ``gather_indices`` is then unused.
            None builds them here.
        prior_key_pos (nl.NkiTensor): [P_MAX, T, num_prior_tiles] fp32 @ HBM, or None. See above.
        prior_natural_rows (nl.NkiTensor): [P_MAX, T, num_prior_tiles] uint32 @ HBM, or None. Not read; the
            key path gathers through the snake table.
        pos_offset (nl.NkiTensor): [1, 1] int32 @ HBM, or None. Per-rank shift added to every live selected
            position when building the tables; ignored with prebuilt tables.
        positions_km (nl.NkiTensor): [num_seqs * P_MAX, spec * num_prior_tiles] int32 @ HBM, or None. The
            selection already key-major, read instead of ``gather_indices`` when building the tables (see
            ``build_prior_index``); ignored with prebuilt tables.
        attend_new_keys (bool): Whether this step's ``spec`` appended rows are scored. True is the causal
            decoder rule. False scores only the gathered prior selection: the GLM5.2 MTP ``KVShare`` rule
            for draft steps after the first, which reuse the first step's selection and must not attend
            their own key. The rows are still written unless ``new_slot_mapping`` is negative too.
        ctx_lse_out (nl.NkiTensor): [T * Nh, chunk] bf16 @ HBM, ``chunk >= C + 2``. The head-major cell table
            the fused CP combine sends: row ``n * T + t`` holds token ``t``'s head ``n``, its ``C`` context
            values, then its fp32 lse as raw bits in bf16 slots ``[C, C + 2)``, then padding.
        prior_tables_out (list): Optional. When the kernel builds the tables itself, they are also staged to
            HBM with ``export_prior_index`` and appended as ``(snake_hbm, key_pos_hbm, natural_hbm)`` for later
            layers sharing this selection (see ``build_shared_prior_index``). Three small DMAs, off the
            critical path.
        heads_blocked (int): 1 (default) takes the layout above. ``g > 1`` takes the queries rank-major as the
            head gather left them: ``q_absorbed`` [g * P_MAX, C/P_MAX, T, Nh/g] and ``q_pe`` [g * R, T, Nh/g],
            head ``r * Nh/g + h`` in block ``r`` column ``h`` (see ``_RANK_MAJOR_HEADS``).
        kv_free (int): 2 for a KVShare step (``attend_new_keys=False``) with no new rows: no quantize, cache
            write, staging or append (``kv_row_new`` supplies only its shape, ``new_slot_mapping`` is unused),
            and no sibling key tile, so ``Sk = num_prior``. Every key of that tile was masked to
            ``-MASK_BOUND``, so the scores, max, sum, lse and context of any row with a kept key are unchanged;
            a rank with no kept key keeps its lse, so its CP combine weight stays 0. 0 (default) is the decoder.
        prefetched (dict): Optional, with ``kv_free=2`` and prebuilt tables: the result of
            :func:`mla_decode_attention_prefetch`, whose loads and gathers are not repeated.
        query_rows (nl.NkiTensor): Must be None (every token is queried); kept for the draft layer's call.

    Returns:
        ctx, lse: None; the normalized latent context of this rank's key share and its lse,
            ``max + log(sum(exp(score - max)))``, are written to ``ctx_lse_out``. A rank that keeps no key of
            a token still yields a finite lse (about -6e28, from the finite -MASK_BOUND bias), and the combine
            weights it 0 because the rank owning the token's position keeps its own row.
        kv_cache (nl.NkiTensor): The input cache, returned so its mutation is part of the result.
    """
    kernel_assert(
        query_rows is None and ctx_lse_out is not None,
        "the draft core queries every token and writes the fused ctx_lse_out payload: query_rows=None and a "
        "ctx_lse_out",
    )
    hb = heads_blocked
    # The shape helper reads the head count off a head-innermost query: give it that view of a blocked one.
    q_shape_src = q_absorbed
    if hb > 1:
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

    C, R = mla.kv_lora_dim, mla.qk_rope_dim
    T, Nh = mla.num_tokens, mla.num_heads
    Sk = mla.num_keys
    num_prior = mla.num_prior
    kv_row = mla.kv_row_dim
    row_b, rope_b = mla.kv_row_bytes, mla.rope_bytes
    nC, nS = mla.num_latent_tiles, mla.num_key_tiles
    n_prior_tiles = mla.num_prior_tiles
    io_dtype = q_absorbed.dtype

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
    kernel_assert(
        kv_row_new.shape == (mla.num_seqs, spec, kv_row),
        f"kv_row_new must be ({mla.num_seqs}, {spec}, {kv_row}), got {kv_row_new.shape}",
    )
    kernel_assert(
        kv_free in (0, 2) and (not kv_free or not attend_new_keys),
        f"kv_free={kv_free} must be 0 or 2, and nonzero only with attend_new_keys=False",
    )
    kernel_assert(
        prefetched is None
        or (kv_free == 2 and prefetched["tables"][0] is prior_snake_rows and prefetched["tables"][1] is prior_key_pos),
        "prefetched needs kv_free=2 and the same prior tables it was built from",
    )
    if kv_free == 2:
        # No sibling tile: the key axis is the prior selection alone.
        Sk, nS = num_prior, n_prior_tiles
    cell = ctx_lse_out.shape[1]
    kernel_assert(
        tuple(ctx_lse_out.shape[:1]) == (T * Nh,) and cell >= C + 2 and cell % 2 == 0 and ctx_lse_out.dtype == io_dtype,
        f"ctx_lse_out must be [{T * Nh}, >= {C + 2} (even)] {io_dtype}, got {ctx_lse_out.shape} {ctx_lse_out.dtype}",
    )

    # 0) Quantize this step's new rows; they are written into the cache after the pair loop.
    packed_new = new_rows_hbm = rows_staged = None
    if not kv_free:
        packed_new = _quantize_new_rows(kv_row_new, kv_scale, mla)
    # Stage the packed rows to HBM for the in-band append: it needs sibling ``i`` on partition ``i``, but
    # ``packed_new`` holds it on partition ``s * spec + i``, a shift the lane-locked engines cannot do.
    if not kv_free:
        new_rows_hbm = nl.ndarray((mla.num_seqs, spec, row_b), dtype=nl.uint8, buffer=nl.private_hbm)
        rows_staged = nisa.dma_copy(dst=new_rows_hbm.reshape((T, row_b))[0:T, 0:row_b], src=packed_new[0:T, 0:row_b])

    # 1) Resolve the global selection to local rows, or take the caller's tables.
    if prefetched is not None:
        snake_rows, key_pos_f, natural_rows = prefetched["snake_rows"], prefetched["key_pos_f"], None
    elif prior_snake_rows is None:
        snake_rows, key_pos_f, natural_rows = build_prior_index(
            gather_indices, block_table, mla, pos_offset=pos_offset, positions_km=positions_km
        )
        if prior_tables_out is not None:
            prior_tables_out.append(export_prior_index(snake_rows, key_pos_f, natural_rows, name="layer_prior_index"))
    else:
        kernel_assert(prior_key_pos is not None, "prior_snake_rows and prior_key_pos must be passed together")
        snake_rows, key_pos_f, natural_rows = import_prior_index(
            prior_snake_rows, prior_key_pos, prior_natural_rows, mla, need_natural=False
        )

    # 2) Buffers. The queries land in SBUF once, in the feature-major layout the score matmuls index.
    if hb == 1:
        q_absorbed_sb = nl.ndarray((P_MAX, nC, T, Nh), dtype=io_dtype, buffer=nl.sbuf)
        q_load = nisa.dma_copy(dst=q_absorbed_sb[0:P_MAX, 0:nC, 0:T, 0:Nh], src=q_absorbed[0:P_MAX, 0:nC, 0:T, 0:Nh])
        q_pe_sb = nl.ndarray((P_MAX, T, Nh), dtype=io_dtype, buffer=nl.sbuf)
        # Ones first, then the rope query over partitions [0, R): partition R stays a row of ones, the
        # contraction row through which the mask bias enters the score (see below).
        nisa.memset(dst=q_pe_sb[0:P_MAX, 0:T, 0:Nh], value=1.0)
        nisa.dma_copy(dst=q_pe_sb[0:R, 0:T, 0:Nh], src=q_pe[0:R, 0:T, 0:Nh])
    else:
        # Rank-major (block r = heads [r * hl, (r + 1) * hl)): one DMA of 1 KB runs per tensor, then heads
        # innermost on chip, since a score stationary must be one free dim (the verifier rejects an (r, h)
        # view). Same-dtype Vector copies, so bit-exact.
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

    # The token-major gather tile: two slots per pair slot (token A, token B), the pipelining depth.
    kernel_assert(Nh <= _PAIR_BAND, f"num_heads={Nh} must fit one PE column band of {_PAIR_BAND}")
    n_pairs = div_ceil(T, 2)
    n_pair_slots = min(_PAIR_SLOTS, n_pairs)
    n_slots = _kv_free_slots(T) if kv_free else 2 * n_pair_slots
    if prefetched is not None:
        rows_sb = prefetched["rows_sb"]
        kernel_assert(
            tuple(rows_sb.shape) == (P_MAX, n_slots, nS, row_b),
            f"prefetched rows must be {(P_MAX, n_slots, nS, row_b)}",
        )
    else:
        rows_sb = nl.ndarray((P_MAX, n_slots, nS, row_b), dtype=nl.uint8, buffer=nl.sbuf)
    # Only the sibling tile needs zeroing: every prior row index is in range, so the prior tiles are
    # fully overwritten, while the sibling tile is written only on partitions [0, spec). The memset
    # buys finite bytes, not zeros (0x00 is +0 as fp8 and bf16): the context matmul reads this tile
    # directly, where 0 * Inf is NaN whatever the mask. The packed row is heterogeneous, so the tile is
    # uint8 and its views are taken once on the whole buffer, where the narrowing view's offset and
    # stride requirements are known to hold.
    if nS > n_prior_tiles:
        nisa.memset(dst=rows_sb[0:P_MAX, 0:n_slots, nS - 1, 0:row_b], value=0.0)
    rows_bf = rows_sb.view(nl.bfloat16)  # [P_MAX, n_slots, nS, row_b // 2]
    rows_f8 = rows_sb.view(nl.float8_e4m3fn)  # [P_MAX, n_slots, nS, row_b]

    # Feature-major keys for the score matmul, one buffer per token in flight. The latent buffer is bf16
    # but holds fp8 bytes: the transpose moves 16-bit words, so 512 fp8 bytes travel as 256 words in two
    # 128-column tiles instead of four, and the score matmul reads them back through an fp8 view. Every
    # key tile is full width (``Sk_full = nS * P_MAX`` words) so the ragged sibling tile can be scored
    # full width too; latent half ``jw`` of a slot is words [jw * Sk_full, (jw + 1) * Sk_full).
    Sk_full = nS * P_MAX
    n_bufs = n_slots
    kv_res_all = nl.ndarray((P_MAX, n_bufs, 2 * Sk_full), dtype=nl.bfloat16, buffer=nl.sbuf)

    def lat_off(jw: int, key: int) -> int:
        """Word offset within a slot of (latent half ``jw``, key ``key``)."""
        return jw * Sk_full + key

    kpe_res_all = nl.ndarray((P_MAX, n_bufs, Sk_full), dtype=nl.bfloat16, buffer=nl.sbuf)
    # Columns past the last real key are never written by a transpose but are read, since the score
    # matmuls stream every bank at full width: zero them once (rope buffer and both latent halves) so the
    # bias row alone decides them.
    if Sk_full > Sk:
        nisa.memset(dst=kpe_res_all[0:R, 0:n_bufs, Sk:Sk_full], value=0.0)
        for jw in range(2):
            nisa.memset(
                dst=kv_res_all[0:P_MAX, 0:n_bufs, lat_off(jw, Sk) : lat_off(jw, Sk) + (Sk_full - Sk)], value=0.0
            )
    # Rope key rows above the bias row are never written: zero them so the rope+bias matmul contracts over
    # all 128 partitions (ones on the query side, so the padding is exact) rather than 65, a sub-128 shape
    # that can trip the di/dt throttle. The memset starts at R, a legal partition start; row R is the bias
    # row, which each token's bias DMA rewrites later.
    kernel_assert(R % 32 == 0, f"rope_dim={R} must be a multiple of 32 to zero the rows above it in one access")
    nisa.memset(dst=kpe_res_all[R:P_MAX, 0:n_bufs, 0:Sk_full], value=0.0)

    # Head-major scores and weights: head ``h`` of key ``si * P_MAX + p`` at ``[h, si * P_MAX + p]``. The
    # query is then the score matmul's stationary, loaded once per contraction chunk, and the keys stream
    # as the moving operand at full tile width. The softmax becomes a per-partition free-axis reduction,
    # and the denominator lands where the deferred reciprocal wants it. The cost is one transpose of the
    # weights per key tile on the way into the context sum, which wants them key-major.
    #
    # One tile per pair slot: token A's heads on partitions [0, Nh), token B's on [64, 64 + Nh). The
    # transposes read all 128 rows, so rows no token writes are zeroed once.
    w_hm = nl.ndarray((P_MAX, n_pair_slots, Sk_full), dtype=io_dtype, buffer=nl.sbuf)
    if Nh < _PAIR_BAND or T % 2:
        # The whole tile: rows above Nh, or the missing token B of an odd last pair, are never written (and
        # a memset must start on partition 0 or 64, which the rows above Nh do not).
        nisa.memset(dst=w_hm[0:P_MAX, 0:n_pair_slots, 0:Sk_full], value=0.0)
    # fp32 scores in SBUF, one pair per slot: PSUM holds only the bank being accumulated, so the next
    # pair's matmuls can start while this pair's softmax reads from here.
    s_sb = nl.ndarray((P_MAX, n_pair_slots, Sk_full), dtype=nl.float32, buffer=nl.sbuf)
    # Key tiles per PSUM bank (512 fp32 per partition). No bank holds a single tile unless there is only
    # one: a moving free dim of exactly 128 can trip the di/dt throttle, so a trailing 4 + 1 becomes 3 + 2.
    tiles_per_bank = F_MAX // P_MAX
    bank_ranges = _split_banks(nS, tiles_per_bank)
    n_banks = len(bank_ranges)

    # Query positions replicated down all partitions, so the key-major keep test needs no broadcast.
    pos_i = nl.ndarray((P_MAX, T), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=pos_i[0:P_MAX, 0:T],
        src=positions[0:T].reshape((1, T)).ap(pattern=[[0, P_MAX], [1, T]], offset=0),
    )
    pos_f = nl.ndarray((P_MAX, T), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=pos_f[0:P_MAX, 0:T], src=pos_i[0:P_MAX, 0:T])

    # The sibling keep, for sibling key ``i`` of query token ``tok`` (speculative index ``j = tok % spec``):
    # causality reduces to ``i <= j``, since sibling ``i`` sits at ``base + i``. CP ownership is also
    # required: siblings are appended on every CP rank, and the combine is the global softmax only when
    # the ranks' key sets are disjoint, so sibling ``i`` is kept only by the rank owning ``base + i``, as a
    # gathered key would be. That depends on each sequence's base, hence ``[spec, num_tokens]``.
    n_seqs = mla.num_seqs
    if not kv_free:
        sib_keep = _sibling_keep(pos_f, n_seqs, spec, T, mla)

    # The additive mask, one bf16 row of ``Sk_full`` biases per token. With keys on the free axis a per-key
    # operand would need a full [Nh, keys] tile, so the mask rides the score's PSUM accumulation at no extra
    # instructions: the rope matmul contracts one more row, a one in the query (partition R of ``q_pe_sb``)
    # against the token's bias row (partition R of ``kpe_res``), 0 for a kept key and -MASK_BOUND for a
    # dropped one. That matmul is issued first for each key tile, at full width, so the sibling tile's
    # ragged tail is a defined -MASK_BOUND; the latent chunks accumulate on top.
    #
    # Additive is safe because every score is finite and bounded (fp8 keys, bf16 queries, a 576-long
    # contraction), so a dropped score is -MASK_BOUND exactly in fp32 and its exp is exactly 0. The bias is
    # bf16 because an fp32 matmul runs the array in a slower multi-pass mode.
    #
    # The rows depend only on ``key_pos``, ``positions`` and the sibling keep, so they are built once here:
    # the keep test key-major, one bf16 conversion, PE transposes to row order, and a copy to HBM from which
    # each token's row comes back as one contiguous DMA.
    keep_km = nl.ndarray((P_MAX, T, nS), dtype=nl.float32, buffer=nl.sbuf)
    for tok in range(T):
        # keep = (key_pos <= positions[tok]) for the prior tiles; NON_RESIDENT_POS always fails. Integers are
        # exact in fp32 here, so ``diff < 0.5`` is the reference's ``<=``.
        nisa.tensor_scalar(
            dst=keep_km[0:P_MAX, tok, 0:n_prior_tiles],
            data=key_pos_f[0:P_MAX, tok, 0:n_prior_tiles],
            op0=nl.subtract,
            operand0=pos_f[0:P_MAX, tok : tok + 1],
            op1=nl.less,
            operand1=0.5,
        )
    # The sibling column: the sibling keep on partitions [0, spec), zero past it. Left all zero under
    # ``attend_new_keys=False`` (KVShare), so the whole sibling tile is dropped.
    if nS > n_prior_tiles:
        nisa.memset(dst=keep_km[0:P_MAX, 0:T, n_prior_tiles:nS], value=0.0)
    if attend_new_keys:
        nisa.tensor_copy(dst=keep_km[0:spec, 0:T, n_prior_tiles:nS], src=sib_keep[0:spec, 0:T].reshape((spec, T, 1)))
    # keep * MASK_BOUND - MASK_BOUND: exactly 0 when kept, else -MASK_BOUND rounded to bf16 (computed in fp32).
    bias_km = nl.ndarray((P_MAX, T * nS), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=bias_km[0:P_MAX, 0 : T * nS],
        data=keep_km.reshape((P_MAX, T * nS))[0:P_MAX, 0 : T * nS],
        op0=nl.multiply,
        operand0=MASK_BOUND,
        op1=nl.subtract,
        operand1=MASK_BOUND,
    )
    # Flip to row order, 128 (tok, tile) columns per transpose, and copy to ``bias_rows[tok, si, p]`` in HBM
    # so a token's biases are contiguous.
    bias_rows = nl.ndarray((T * nS, P_MAX), dtype=nl.bfloat16, buffer=nl.private_hbm)
    for ch in TiledRange(T * nS, P_MAX):
        cs, ce, csz = ch.start_offset, ch.end_offset, ch.size
        bT_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.psum)
        nisa.nc_transpose(bT_psum[0:csz, 0:P_MAX], bias_km[0:P_MAX, cs:ce])
        bT_sb = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=bT_sb[0:csz, 0:P_MAX], src=bT_psum[0:csz, 0:P_MAX], engine=nisa.engine.scalar)
        nisa.dma_copy(dst=bias_rows[cs:ce, 0:P_MAX], src=bT_sb[0:csz, 0:P_MAX])
    bias_rows_flat = bias_rows.reshape((T, Sk_full))

    # The dequant rides the softmax scale, undoing both halves of the score with one scalar.
    scale = softmax_scale * kv_scale
    # Scaling at the eviction is exact only for a power-of-two scale: then scale * max(s) == max(scale * s),
    # and scale * s - scale * m is what an unfused exp would see.
    kernel_assert(
        scale > 0 and math.frexp(scale)[0] == 0.5,
        f"softmax_scale * kv_scale = {scale} must be a power of two (the eviction-folded softmax is exact only then)",
    )

    # Per pair of query tokens: each token gathers its own selection (one top-k row per token, as in the
    # reference) and builds its own feature-major keys; the pair shares the score PSUM tile (token A on
    # partitions [0, Nh) from PE band 0, token B on [64, 64 + Nh) from band 1), one softmax, one set of
    # weight transposes and one banded context sum.
    def prep_token(tok: int, slot: int) -> None:
        """Gather token ``tok``'s rows into ``slot`` and build its feature-major keys and bias row there."""
        if prefetched is not None and tok < prefetched["gathered"]:
            pass  # gathered by mla_decode_attention_prefetch
        elif tok >= 2 or not kv_free:
            # Tokens 0-1 lead the core once this step's rows are staged; the rest wait for the query.
            gather = gather_prior_rows(rows_sb, slot, kv_cache, snake_rows, tok, mla)
            gather.depends_on(q_load if tok >= 2 else rows_staged)
        else:
            # A kv_free call stages no rows, so its first tokens' gathers have nothing to follow.
            gather_prior_rows(rows_sb, slot, kv_cache, snake_rows, tok, mla)
        if not kv_free:
            _append_spec_rows(rows_sb, slot, new_rows_hbm, tok, mla)
        kv_res = kv_res_all[0:P_MAX, slot]
        kpe_res = kpe_res_all[0:P_MAX, slot]

        # Feature-major keys: the kernel's only transpose of key data (the weighted sum uses the gathered
        # token-major rows). The latent moves as bf16 words over fp8 bytes, which are arbitrary bit patterns
        # (e.g. bf16 subnormals), so the transpose and eviction must be bit-accurate: the PE transpose is,
        # and the eviction is pinned to Vector, the bit-accurate PSUM reader.
        for st in TiledRange(Sk, P_MAX):
            ss, se, ssz = st.start_offset, st.end_offset, st.size
            si = st.index
            kpe_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.psum)
            nisa.nc_transpose(kpe_psum[0:R, 0:ssz], rows_bf[0:ssz, slot, si, 0:R])
            nisa.tensor_copy(dst=kpe_res[0:R, ss:se], src=kpe_psum[0:R, 0:ssz])
            for jw in range(2):
                w0 = rope_b // 2 + 128 * jw  # bf16 word offset of this half's first word
                ckv_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.psum)
                nisa.nc_transpose(ckv_psum[0:P_MAX, 0:ssz], rows_bf[0:ssz, slot, si, w0 : w0 + 128])
                nisa.tensor_copy(
                    dst=kv_res[0:P_MAX, lat_off(jw, ss) : lat_off(jw, ss) + ssz],
                    src=ckv_psum[0:P_MAX, 0:ssz],
                    engine=nisa.engine.vector,
                )

        # This token's bias row lands on partition R of its rope key buffer, the rope matmul's extra row.
        nisa.dma_copy(
            dst=kpe_res[R : R + 1, 0:Sk_full], src=bias_rows_flat[tok : tok + 1, 0:Sk_full], dge_mode=dge_mode.hwdge
        )

    band_a = dict(tile_position=(0, 0), tile_size=(P_MAX, _PAIR_BAND))
    band_b = dict(tile_position=(0, _PAIR_BAND), tile_size=(P_MAX, _PAIR_BAND))

    def pair_begin(pair: int) -> dict:
        """Query-token pair ``pair``: its tokens, buffer slots, partition bases, PE bands and partition ranges."""
        pslot = pair % n_pair_slots
        tok_a = 2 * pair
        has_b = tok_a + 1 < T
        toks = [tok_a, tok_a + 1] if has_b else [tok_a]
        # Partition base and PE column band of each token. A lone trailing token (odd T) takes the whole
        # array and partitions [0, Nh) untiled.
        bases = [0, _PAIR_BAND]
        # Partition ranges of the per-partition stages: one when the tokens' heads abut (Nh == 64), else one each.
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
        Head-major scores, one PSUM bank at a time, both tokens of the pair: ``s[b * 64 + h, key]``.

        The query is the stationary (M = Nh) and the feature-major keys the moving operand. One matmul
        per (token, contraction chunk, bank) streams the whole bank, up to 512 keys, so no matmul has a
        moving free dim of exactly 128 or a contraction under 128 (both di/dt throttle triggers). The two
        tokens issue into the two PE column bands and write the two halves of one PSUM tile; the bands
        run concurrently. The rope+bias matmul initializes each half, the nC latent chunks accumulate on
        top, and the finished bank is evicted and max-reduced while the next bank computes.

        Keeping one bank in PSUM lets consecutive pairs overlap; a whole pair's scores would occupy five
        of eight banks and block the next pair until this softmax drained.

        The latent is read as fp8 through a view of the bf16 word buffer; partition p of chunk
        ``ci = 2j + parity`` holds latent dim ``128 * ci + p`` because the cache's byte order is permuted
        at write time. A bf16 stationary with an fp8 moving is allowed, and PSUM accumulates in fp32. The
        fp8 view is taken per token on the 2-D slice: on the 3-D buffer a free-axis slice would index
        slots rather than words and read the wrong token's keys.
        """
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
                # rope + bias first, over the whole bank and all 128 partitions: rows [0, R) are rope, row R
                # is ones against the bias row, and the rows above are ones against zeros.
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
            # Evict the finished bank scaled and take its max in the same Vector instruction, off the PE's path.
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

        -scale * m comes from a negating max-reduce of the per-bank maxima; then
        w = exp(scale * s - scale * m) with the sum fused into the same instruction. ``reduce_cmd`` must be
        passed, since its default ``idle`` would leave ``reduce_res`` reading a stale accumulator. Masked
        scores are the finite -MASK_BOUND, so m, l and the lse stay finite on a rank with no kept key.
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
        # Token ``tok``'s head ``n`` is row ``n * T + tok``; its lse is the fp32 bits in the two bf16 slots at column C.
        toks, bases, lse_col = P["toks"], P["bases"], P["lse_col"]
        for k, tok in enumerate(toks):
            pb = bases[k]
            nisa.dma_copy(
                dst=ctx_lse_out.ap(pattern=[[T * cell, Nh], [1, 2]], offset=tok * cell + C),
                src=lse_col[pb : pb + Nh, 0:1].view(io_dtype),
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
            nS=nS,
            num_tokens=T,
        )

    # The pair loop is software-pipelined in groups of two pairs (A, B) so the PE never idles on a softmax:
    #
    #     prep(A) score(A) smax(A) prep(B) wT(A) score(B) smax(B) tail(A) ctx(A) wT(B) tail(B) ctx(B)
    #
    # A's softmax runs under B's key transposes and B's under A's context sum. Groups of two keep ctx(A)
    # ahead of the next group's prep(A'), whose gathers reuse A's row slots. A lone last pair runs in order.
    for g in range(0, n_pairs, 2):
        if g + 1 == n_pairs:
            P = pair_begin(g)
            emit_prep(P)
            emit_scores(P)
            emit_softmax(P)
            emit_softmax_tail(P)
            emit_context(P, _emit_weight_transposes(w_hm, P["pslot"], mla, io_dtype, nS))
            continue
        A, B = pair_begin(g), pair_begin(g + 1)
        emit_prep(A)
        emit_scores(A)
        emit_softmax(A)
        emit_prep(B)
        wT_a = _emit_weight_transposes(w_hm, A["pslot"], mla, io_dtype, nS)
        emit_scores(B)
        emit_softmax(B)
        emit_softmax_tail(A)
        emit_context(A, wT_a)
        wT_b = _emit_weight_transposes(w_hm, B["pslot"], mla, io_dtype, nS)
        emit_softmax_tail(B)
        emit_context(B, wT_b)

    if not kv_free:
        # After every gather: no live selection reads these rows (see the schedule note at the top).
        _update_kv_cache(kv_cache, packed_new, new_slot_mapping, mla)
    return None, None, kv_cache


def _split_banks(n_tiles: int, tiles_per_bank: int) -> list:
    """Split ``n_tiles`` key tiles into PSUM banks of at most ``tiles_per_bank`` tiles, never leaving a
    one-tile bank unless there is only one tile (e.g. 17 -> [4, 4, 4, 3, 2])."""
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


def _sibling_keep(pos_f: nl.NkiTensor, n_seqs: int, spec: int, T: int, mla: MlaDecodeShape) -> nl.NkiTensor:
    """The sibling keep ``[P_MAX, T]`` fp32 (rows ``[0, spec)``): causality times CP ownership, see the caller."""
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

    # owner = (position >> block_shift) & (cp_degree - 1), the prior index's round-robin rule, so the two
    # paths agree on who owns a position.
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
    return sib_keep


def _emit_weight_transposes(
    w_hm: nl.NkiTensor, pslot: int, mla: MlaDecodeShape, io_dtype: nl.DType, nS: int = None
) -> nl.NkiTensor:
    """
    The pair's weights key-major, ``[P_MAX, nS, P_MAX]`` @ SBUF, for :func:`_emit_context_sum`.

    One PE transpose per key tile per pair, over all 128 rows (token A's heads on rows [0, Nh), token B's
    on [64, 64 + Nh)). Columns [0, Nh) of the result are token A's key-major weights and [64, 64 + Nh)
    token B's, each the stationary of that token's context matmul. These are real bf16 values, so any
    PSUM reader is exact; the eviction goes on Scalar, which the softmax leaves mostly idle. ``nS``
    defaults to ``mla.num_key_tiles``.
    """
    nS = mla.num_key_tiles if nS is None else nS
    # Full 128 keys per tile (the last tile's extras are zero weights), so the context matmul contracts K = 128.
    wT_sb = nl.ndarray((P_MAX, nS, P_MAX), dtype=io_dtype, buffer=nl.sbuf)
    for si in range(nS):
        wT_psum = nl.ndarray((P_MAX, P_MAX), dtype=io_dtype, buffer=nl.psum)
        nisa.nc_transpose(wT_psum[0:P_MAX, 0:P_MAX], w_hm[0:P_MAX, pslot, si * P_MAX : (si + 1) * P_MAX])
        nisa.tensor_copy(dst=wT_sb[0:P_MAX, si, 0:P_MAX], src=wT_psum[0:P_MAX, 0:P_MAX], engine=nisa.engine.scalar)
    return wT_sb


def _emit_context_sum(
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
    ctx_lse_out: nl.NkiTensor,
    recip_col: nl.NkiTensor = None,
    nS: int = None,
    num_tokens: int = None,
) -> None:
    """
    ctx[Nh, C] = w[Nh, Sk] . kv_latent[Sk, C], normalized, for the one or two tokens of a pair, into the
    ``ctx_lse_out`` cell table.

    The contraction is over keys, so the latent is wanted token-major, as the gather left it: the fp8
    bytes feed the matmul straight from the packed row, with no transpose and no dequantization pass
    (which would cost about as much as the rest of the kernel). ``wT_sb`` is the key-major weights of
    :func:`_emit_weight_transposes`. x_psum's free index is the row's permuted byte order; the
    un-permute happens in the denominator multiply.

    ``recip_col`` is the denominator's reciprocal if the caller already took it, else it is taken here.
    ``nS`` defaults to ``mla.num_key_tiles`` and ``num_tokens`` (the output's token count) to ``mla.num_tokens``.
    """
    C, Nh = mla.kv_lora_dim, mla.num_heads
    rope_b = mla.rope_bytes
    nS = mla.num_key_tiles if nS is None else nS
    T = mla.num_tokens if num_tokens is None else num_tokens

    x_psum = nl.ndarray((P_MAX, C), dtype=nl.float32, buffer=nl.psum)
    for si in range(nS):
        for k, tok in enumerate(toks):
            pb = bases[k]
            # K = 128 on every tile, the ragged last one included: rows past the real keys are zero weights
            # (exp of a masked score) against finite zeroed rows, so the padding adds exactly nothing.
            nisa.nc_matmul(
                dst=x_psum[pb : pb + Nh, 0:C],
                stationary=wT_sb[0:P_MAX, si, pb : pb + Nh],
                moving=rows_f8[0:P_MAX, slots[k], si, rope_b : rope_b + C],
                accumulate=(si > 0),
                **bands[k],
            )

    # The deferred softmax denominator is applied to x[Nh, C] rather than w[Nh, Sk], far fewer elements;
    # it is already one element per partition (head).
    x_sb = nl.ndarray((P_MAX, C), dtype=io_dtype, buffer=nl.sbuf)
    if recip_col is None:
        recip_col = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        for p0, p1 in ranges:
            nisa.reciprocal(recip_col[p0:p1, 0:1], l_col[p0:p1, 0:1])

    # Un-permute while normalizing: dims ``256a + 128b .. +127`` sit at byte-order positions ``256a + b``
    # with stride 2, so latent order costs four strided instructions, not a data pass. The partition slice
    # on ``data`` is required: tensor_scalar needs the partition dims to match ``operand0`` exactly.
    for p0, p1 in ranges:
        for a in range(2):
            for b in range(2):
                dst_off = 256 * a + 128 * b
                src_off = 256 * a + b
                # The latent dequant rides op1 at no cost: the matmul contracted the raw fp8 latent, in
                # units of latent / kv_scale.
                nisa.tensor_scalar(
                    dst=x_sb[p0:p1, dst_off : dst_off + 128],
                    data=x_psum.slice(1, start=src_off, end=src_off + 255, step=2)[p0:p1],
                    op0=nl.multiply,
                    operand0=recip_col[p0:p1, 0:1],
                    op1=nl.multiply,
                    operand1=kv_scale,
                )
    # Head-major cell table: token ``tok``'s head ``n`` at row ``n * T + tok``, context in the first C
    # columns. One DMA per token, pinned to the hardware DGE; by default it lowers to a software
    # descriptor build on GpSimd, which the indirect gathers already saturate.
    cell = ctx_lse_out.shape[1]
    for k, tok in enumerate(toks):
        pb = bases[k]
        nisa.dma_copy(
            dst=ctx_lse_out.ap(pattern=[[T * cell, Nh], [1, C]], offset=tok * cell),
            src=x_sb[pb : pb + Nh, 0:C],
            dge_mode=dge_mode.hwdge,
        )


def _quantize_new_rows(kv_row_new: nl.NkiTensor, kv_scale: float, mla: MlaDecodeShape) -> nl.NkiTensor:
    """
    Pack this step's logical bf16 rows into the cache's byte layout, quantizing the latent to fp8.

    Returns [P_MAX, kv_row_bytes] uint8 @ SBUF, row ``s * spec + i`` on partition ``s * spec + i``.

    The range is clamped rather than assumed: the simulator's fp8 cast yields NaN for an out-of-range
    e4m3fn value, and one NaN latent would poison a whole score row and softmax denominator.
    """
    n_new, C, R = mla.num_tokens, mla.kv_lora_dim, mla.qk_rope_dim
    row_b, rope_b = mla.kv_row_bytes, mla.rope_bytes
    inv_scale = 1.0 / kv_scale

    src = nl.ndarray((P_MAX, R + C), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=src[0:n_new, 0 : R + C], src=kv_row_new.reshape((n_new, R + C))[0:n_new, 0 : R + C])

    packed = nl.ndarray((P_MAX, row_b), dtype=nl.uint8, buffer=nl.sbuf)
    # Partitions past n_new are never scattered; zeroing keeps them finite in every view.
    nisa.memset(dst=packed[0:P_MAX, 0:row_b], value=0.0)
    packed_bf = packed.view(nl.bfloat16)
    packed_f8 = packed.view(nl.float8_e4m3fn)

    # k_pe pre-divided by kv_scale, exact in bf16 since kv_scale is a power of two.
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

    # Clamp the lower end and cast to fp8 in the permuted byte order, 128 dims per op at stride 2; the
    # offsets come from kv_row_layout, shared with the reader.
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
    Write this step's packed rows into the paged cache.

    The rows are also appended in band for this step's scoring, so the gather never reads them back:
    an indirect write then an indirect read of the same HBM tensor has no ordering guarantee to rely on.
    A negative slot skips its write.
    """
    n_new = mla.num_tokens
    row_b = mla.kv_row_bytes

    slots_i32 = nl.ndarray((P_MAX, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=slots_i32[0:n_new, 0:1], src=new_slot_mapping.reshape((n_new, 1))[0:n_new, 0:1])

    # A negative slot must become an explicit out-of-range row: the int32 -> uint32 cast goes through fp32
    # and saturates, so -1 would become row 0 and clobber it. That is the common case, since on all but
    # one CP rank every new row has no home.
    #
    # The sentinel must be out of range for the allocation, not just this view: nki bounds an indirect
    # DMA by the parent storage shape, so for one layer of a stacked cache ``mla.oob_row`` is a valid
    # index into the next layer, and the write would corrupt a row that layer attends to. The gather keeps
    # ``mla.oob_row``, since its snake table is staged through uint16.
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

    # Both sides are uint8 so the scatter does not cast: unlike the batched 2-D form, this 1-D form has no
    # same-dtype guard, and a casting dma_copy would write unscaled, unclamped values into the cache.
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
    Append the ``spec`` packed new rows of this token's sequence after the gathered prior, in band.

    They land on partitions ``[0, spec)`` of key tile ``num_prior_tiles`` (num_prior is a whole number
    of tiles), and the mask keeps sibling ``i <= j`` for query token ``j``. The source is the staged
    packed copy, since appended rows must match the cache's byte layout exactly.
    """
    spec, row_b = mla.spec, mla.kv_row_bytes
    seq = tok // spec
    tile = mla.num_prior_tiles
    nisa.dma_copy(
        dst=rows_sb[0:spec, slot, tile, 0:row_b],
        src=new_rows_hbm[seq, 0:spec, 0:row_b],
    )


__all__ = ["mla_decode_attention", "mla_decode_attention_prefetch"]
