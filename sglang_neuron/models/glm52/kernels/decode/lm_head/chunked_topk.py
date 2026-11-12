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

"""GLM-5.2 fork of ``lm_head/chunked_topk.py``: identical except that every ``nc_matmul`` carries an explicit
``accumulate=False`` (see ``tail_chunked``).

A top-k over a vocab wider than ``nisa.topk``'s 65,535-element limit, on one core and with no collective: stage 1
splits each row into ``P`` contiguous chunks and takes a top-k of each, stage 2 takes the top-k of the ``P * k``
candidates.

* The chunk split comes from the instruction itself: ``nisa.topk`` ranks each group of 16 partitions independently,
  so loading a row as one contiguous sweep over 128 partitions makes every group a contiguous vocab range. The
  reshapes between the stages are views (rows are token-major, ``j * P + c``).
* A stage-1 index is chunk-local; the chunk offset folds into the snake-to-vocab remap, so global ids cost nothing
  extra on the fast path.
* Each stage de-snakes its results once per tile (``nisa.topk`` spreads a row's results over 16 partitions).
* Stage 1 is never sorted; only the final reduce sorts, with the hardware sort (``sorted=True``). Cost follows the
  number of top-k calls times ``k``, not the elements scanned.
* ``P`` is the smallest value that fills a whole tile and keeps each chunk legal: stage-1 cost does not depend on
  ``P``, the reduce grows with it.
* The position-to-id lookup is a one-hot contraction on the PE rather than an ``nc_n_gather``, whose stores were
  observed to land late for a consumer that waited on it correctly.
"""

import inspect
from dataclasses import dataclass
from typing import Tuple

import nki
import nki.isa as nisa
import nki.language as nl
import numpy as np

from .tail_helpers import get_verified_program_sharding_info
from .topk_layout import (
    BFLOAT16_MIN,
    GROUPS_PER_TILE,
    PARTS_PER_GROUP,
    PMAX,
    _valid_desnake_col_runs,
    _validate_topk,
    create_topk_config,
)
from ..utils.kernel_assert import kernel_assert
from ..utils.kernel_helpers import div_ceil
from ..utils.scratch_hbm import scratch_hbm

# nisa.topk reduces n < _TOPK_N_MAX elements per call. This is the instruction's own
# bound (the returned index is a snake POSITION), not a kernel choice -- which is why
# P >= ceil(V / (_TOPK_N_MAX - 1)) is a hardware floor rather than a tuning knob.
_TOPK_N_MAX = 65536
# nisa.topk's k bound, mirrored so an out-of-envelope k fails here with a clear message.
_TOPK_K_MAX = 32768

# --- Hardware sort capability ---
# nisa.topk(sorted=True) emits rank j at snake position j, which turns the descending sort into a fixed column
# permutation (the hw_sort branch of _inline_topk). Probed from the signature rather than a version check: passing
# sorted= to an nki without it fails at trace time, and the flag needs both the nki argument and the compiler's BIR
# field.
_HW_SORT_AVAILABLE = "sorted" in inspect.signature(nisa.topk).parameters
# The ISA caps the sorted mode at k <= 8192. Sorted requests above it are REFUSED: no
# software sort serves them.
_HW_SORT_MAX_K = 8192


@dataclass(frozen=True, eq=True)
class ChunkedTopkConfig(nl.NKIObject):
    """Configuration for ``chunked_topk``.

    Attributes:
        BxS: rows (tokens).
        vocab_size: the full vocab reduced over; may exceed nisa.topk's 65,535.
        k: values kept per row.
        chunks: P, the vocab chunks stage 1 reduces independently; 1 runs a single inlined top-k.
        sorted: whether the output must be descending.
        inp_dtype: bfloat16 (nisa.topk's requirement).
        index_dtype: uint32.
        out_shape: the output shape (input leading dims + k).
        debug_taps: append the diagnostic taps of ``TAP_NAMES`` to the outputs (this changes the kernel's arity).
        skip_nan_fold: skip stage 1's NaN fold, a compare and a predicated copy over the full width per tile. Only for
            input known to hold no NaN (+/-inf are fine), such as the sampling tail's own GEMM output. The reduce always
            skips it, since its input is already folded.
        interleaved_peer_width: 0 for an ordinary ``[BxS, vocab]`` input. ``w > 0`` takes the all-to-all receive layout
            instead, ``[BxS * (vocab / w), w]`` with row ``p * BxS + j`` holding token ``j``'s columns
            ``[p * w, (p + 1) * w)``; stage 1 folds the de-interleave into its loads, bitwise the same as
            de-interleaving first. Needs the desnake-only stage-1 shape and ``w | (vocab / P) / 16``, else it is refused
            at trace time.
    """

    BxS: int
    vocab_size: int
    k: int
    chunks: int
    sorted: bool
    inp_dtype: np.dtype
    index_dtype: np.dtype
    out_shape: tuple
    debug_taps: bool = False
    skip_nan_fold: bool = False
    interleaved_peer_width: int = 0


def pick_chunks(BxS: int, vocab: int, k: int) -> int:
    """Smallest legal P that fills a whole 128-partition tile.

    Ranked by: fills a tile (``BxS * P % 8 == 0``), then takes the coalesced load path (``(vocab / P) % 16 == 0``),
    then the smallest P. That picks ``P = 8`` at ``BxS = 1`` and ``P = 4`` at ``BxS >= 4``. Returns 1 when the vocab
    fits one nisa.topk call.
    """
    if vocab < _TOPK_N_MAX:
        return 1
    best_key, best_p = None, None
    p = 1
    while p <= PMAX:
        if vocab % p == 0:
            width = vocab // p
            legal = (8 <= width < _TOPK_N_MAX) and (k <= width) and (8 <= p * k < _TOPK_N_MAX)
            if legal:
                key = (
                    0 if (BxS * p) % GROUPS_PER_TILE == 0 else 1,
                    0 if width % PARTS_PER_GROUP == 0 else 1,
                    p,
                )
                if best_key is None or key < best_key:
                    best_key, best_p = key, p
        p *= 2
    if best_p is None:
        raise ValueError(
            f"no legal chunk count for vocab={vocab}, k={k}: need a power-of-two P dividing "
            f"vocab with 8 <= vocab/P < {_TOPK_N_MAX}, k <= vocab/P and 8 <= P*k < {_TOPK_N_MAX}"
        )
    return best_p


def interleaved_fold_supported(bxs: int, vocab: int, k: int, peer_width: int) -> bool:
    """Whether the kernel accepts ``interleaved_peer_width=peer_width`` at this geometry.

    Callers use this to choose between the fold and an explicit de-interleave, so they apply the kernel's own rules
    rather than a subset of them (a vocab small enough for ``pick_chunks`` to return 1 takes the single-call path,
    which has no fold). It mirrors the p == 1 refusal, the desnake-only stage-1 requirements and the peer-width
    divisibility asserts.
    """
    if peer_width <= 0:
        return False
    p = pick_chunks(bxs, vocab, k)
    if p == 1:
        return False
    width = vocab // p
    s1_rows = bxs * p
    if not (k % PARTS_PER_GROUP == 0 and s1_rows % GROUPS_PER_TILE == 0 and width % PARTS_PER_GROUP == 0):
        return False
    n_cols_s1 = div_ceil(width, PARTS_PER_GROUP)
    return vocab % peer_width == 0 and n_cols_s1 % peer_width == 0 and width % peer_width == 0


def create_chunked_topk_config(
    inp_shape: Tuple,
    inp_dtype: np.dtype,
    k: int,
    chunks: int = 0,
    sorted: bool = True,
    debug_taps: bool = False,
    skip_nan_fold: bool = False,
    interleaved_peer_width: int = 0,
) -> ChunkedTopkConfig:
    """Build a ChunkedTopkConfig. ``chunks=0`` (default) auto-picks via ``pick_chunks``.

    ``inp_shape`` is always the LOGICAL ``[..., vocab]`` shape, including when
    ``interleaved_peer_width`` declares the receive layout (the physical tensor then has the
    same element count in ``[BxS*(vocab/w), w]`` order; every internal access is a reshape or
    an explicit pattern over the same flat buffer).
    """
    BxS = 1
    for d in inp_shape[:-1]:
        BxS *= d
    vocab_size = inp_shape[-1]
    return ChunkedTopkConfig(
        BxS=BxS,
        vocab_size=vocab_size,
        k=k,
        chunks=chunks if chunks else pick_chunks(BxS, vocab_size, k),
        sorted=sorted,
        inp_dtype=inp_dtype,
        index_dtype=nl.uint32,
        out_shape=tuple(list(inp_shape[:-1]) + [k]),
        debug_taps=debug_taps,
        skip_nan_fold=skip_nan_fold,
        interleaved_peer_width=interleaved_peer_width,
    )


#: The diagnostic taps ``chunked_topk`` appends to its outputs under ``config.debug_taps``, in order, so a caller that
#: forwards them (the sampling tail does) can name them.
TAP_NAMES = ("s1_pos", "s1_gid", "s1_late", "cand_i", "r_pos", "gathered", "s1_val")


def _validate_chunked_topk(config: ChunkedTopkConfig) -> None:
    """Validate the requested shape against every nisa.topk constraint, per stage."""
    v, k, p = config.vocab_size, config.k, config.chunks
    kernel_assert(config.inp_dtype == nl.bfloat16, "chunked_topk requires bfloat16 input")
    kernel_assert(1 <= k < _TOPK_K_MAX, f"chunked_topk requires 1 <= k ({k}) < {_TOPK_K_MAX}")
    # Sorted output needs the hardware sort (there is no software fallback), so refuse it at trace time without one.
    if config.sorted:
        kernel_assert(
            _HW_SORT_AVAILABLE,
            "chunked_topk sorted=True requires nisa.topk(sorted=True); this toolchain does not "
            "have it and the software sort "
            "was removed. Pass sorted=False or upgrade the toolchain.",
        )
        kernel_assert(k <= _HW_SORT_MAX_K, f"sorted top-k is capped at k <= {_HW_SORT_MAX_K} by the ISA, got k={k}")
    kernel_assert(p >= 1, f"chunks ({p}) must be >= 1")
    kernel_assert(v % p == 0, f"vocab ({v}) must be divisible by chunks ({p})")
    width = v // p
    # Stage 1 sees [BxS*P, V/P], so the CHUNK width is what faces the instruction bound.
    kernel_assert(8 <= width < _TOPK_N_MAX, f"chunk width V/P ({width}) must be in [8, {_TOPK_N_MAX})")
    kernel_assert(k <= width, f"k ({k}) must be <= chunk width ({width})")
    if p > 1:
        # The reduce sees [BxS, P*k]; P*k is a SECOND, independent bound. Batch sharding
        # keeps it small (2,048 at P=8) but a large P inflates it past the ceiling.
        red_w = p * k
        kernel_assert(8 <= red_w < _TOPK_N_MAX, f"reduce width P*k ({red_w}) must be in [8, {_TOPK_N_MAX})")
    if config.interleaved_peer_width:
        w = config.interleaved_peer_width
        n_cols_s1 = div_ceil(width, PARTS_PER_GROUP)
        kernel_assert(
            v % w == 0 and n_cols_s1 % w == 0,
            f"interleaved_peer_width ({w}) must divide the vocab ({v}) and the stage-1 snake "
            f"row width ({n_cols_s1}); the receive-layout fold needs whole peer blocks per "
            f"snake partition",
        )


# _inline_topk is an inline copy of the @nki.jit body of the kernel library's gpsimd_topk: the same instructions, order,
# buffers and DMA modes. Inlining puts both nisa.topk calls in one traced body, so the candidates can stay in SBUF and
# stage 1's snake layout can go straight to the reduce instead of through a [BxS, k] HBM interface.

# Several non-obvious lines (the DMA modes on the de-snake round trip, the NaN folds, the index clamps) work around
# measured hardware defects, explained beside the same lines in the library kernel. Read those before simplifying any of
# them.

#: Slots of _inline_topk's ``fused`` argument. A tuple rather than a dataclass because nki hashes config objects
#: for its trace cache, and these hold live tensor handles.
_FUSE_CAND_I = 0  # [rows, P*k] uint32 HBM: stage-1 chunk-local ids, the gather's data
_FUSE_CHUNKS = 1  # P
_FUSE_WIDTH = 2  # vocab // P, the per-chunk id offset
_FUSE_OUT_VALS = 3  # [rows, k] bf16 shared_hbm destination
_FUSE_OUT_IDX = 4  # [rows, k] uint32 shared_hbm destination
_FUSE_N_ROWS = 5  # rows to emit; the reduce's own row count may be padded above it
_FUSE_TAP_CAND = 6
_FUSE_TAP_POS = 7
_FUSE_TAP_GOT = 8
_FUSE_IDS_GLOBAL = 9  # cand_i already holds float32 GLOBAL ids: skip the cast and the offsets


def _stage1_desnake_only(
    inp: nl.NkiTensor,
    config,
    chunks: int,
    chunk_width: int,
    pad_rows: int,
    tap_pos=None,
    skip_nan_fold=False,
    peer_width=0,
):
    """Stage 1 without its phase 2: the de-snake buffers are the candidate arrays.

    The library top-k's phase 2 reloads the de-snaked ``[BxS, k_pad]`` buffers, compacts them to ``k`` columns, remaps
    the positions and stores ``[BxS, k]``, which ``chunked_topk`` then views as ``[BxS / P, P * k]``. With
    ``k_pad == k`` that pass is a copy: ``[BxS * P, k]`` viewed as ``[BxS, P * k]`` already is the candidate value
    array, matched slot for slot by the index buffer, and the reduce does not care about candidate order. So phase 2
    goes, along with the caller's chunk-offset adds and casts. The snake-to-vocab remap and the chunk offset are
    applied to ``idx_snake`` before the de-snake store instead, on a tile 16x narrower.

    Args:
        inp: ``[BxS * P, V / P]`` bf16 view of the vocab, one chunk per row.
        config: a ``TopkConfig`` for that view. Requires ``k % 16 == 0`` (so ``k_pad == k``) and a shard of
            whole 128-partition tiles (which keeps the value buffer bf16, as the reduce needs).
        chunks: P; row ``r`` of ``inp`` is chunk ``r % P``.
        chunk_width: ``V / P``, the chunk offset unit.
        pad_rows: rows to allocate, ``red_rows * P``, so the reduce's ``[red_rows, P * k]`` view exists. The tail rows
            hold the sentinel and are whole reduce rows whose results are never read.

    Returns:
        ``(cand_v [pad_rows, k] bf16, cand_i [pad_rows, k] f32)``, whose ``[pad_rows / P, P * k]`` views are the
        reduce's input and the gather's data. Indices are global vocab ids.
    """
    _validate_topk(config)

    BxS = config.BxS
    vocab = config.vocab_size
    k = config.k

    shard_info = get_verified_program_sharding_info("chunked_topk_stage1", (0, 1), 2)
    if BxS > 1:
        n_prgs = shard_info[1]
        prg_id = shard_info[2]
    else:
        n_prgs = 1
        prg_id = 0
    per_lnc_BxS = div_ceil(BxS, n_prgs)

    n_cols = div_ceil(vocab, PARTS_PER_GROUP)
    full_n = PARTS_PER_GROUP * n_cols
    has_pad = full_n != vocab
    k_cols = div_ceil(k, PARTS_PER_GROUP)
    k_pad = k_cols * PARTS_PER_GROUP
    fast_dma_safe = (per_lnc_BxS % GROUPS_PER_TILE == 0) and (BxS == n_prgs * per_lnc_BxS)
    # The two preconditions of the whole scheme, asserted rather than assumed: the caller picks
    # this path on exactly these conditions, and if either were false the buffers below would not
    # be the candidate arrays (k_pad > k leaves unwritten columns interleaved with the valid ones)
    # or would not be bfloat16 (which nisa.topk requires of the reduce's source).
    kernel_assert(k_pad == k, f"_stage1_desnake_only needs k % 16 == 0, got k={k}")
    kernel_assert(fast_dma_safe, "_stage1_desnake_only needs whole 128-partition tiles")
    kernel_assert(pad_rows % chunks == 0, f"pad_rows ({pad_rows}) must be a multiple of chunks ({chunks})")
    # chunk_width == 16*n_cols is what makes the chunk offset a whole number of snake
    # partitions, which is what lets it fold into the remap's multiply (see chunk_base).
    kernel_assert(
        not has_pad and chunk_width == PARTS_PER_GROUP * n_cols,
        f"_stage1_desnake_only needs 16 | chunk width, got {chunk_width}",
    )

    inp_flat = inp.reshape((BxS * vocab,))
    # --- The interleaved (all-to-all receive) source layout ---
    # peer_width > 0: inp is the receive layout, where token j's vocab column p*w + q sits at flat address (p*BxS_tok +
    # j)*w + q. Since w divides n_cols, each snake partition covers whole peer blocks and the load stays a three-level
    # pattern with a unit innermost stride:
    #     [[ppp*BxS_tok*w, 16], [BxS_tok*w, ppp], [1, w]]   ppp = n_cols // w
    #     offset(j, c) = c*(chunk_width//w)*BxS_tok*w + j*w
    # The snake contents are those of the de-interleaved [BxS_tok, vocab], so everything downstream is bitwise
    # unchanged; the caller's de-interleave (an HBM round trip) goes away, at one DMA per 16-partition group instead of
    # one per tile.
    BxS_tok = BxS // chunks
    ppp = ppc = 0
    if peer_width:
        kernel_assert(
            not has_pad and n_cols % peer_width == 0 and chunk_width % peer_width == 0,
            f"interleaved_peer_width ({peer_width}) must divide the snake row width "
            f"({n_cols}) and the chunk width ({chunk_width})",
        )
        ppp = n_cols // peer_width
        ppc = chunk_width // peer_width
    # See the desnake_dge comment in _inline_topk: unconditional, and the reason is a measured
    # silent index corruption, not tidiness.
    desnake_dge = nisa.dge_mode.unknown
    cand_v = nl.ndarray((pad_rows, k), dtype=nl.bfloat16, buffer=nl.private_hbm)
    cand_i = nl.ndarray((pad_rows, k), dtype=nl.float32, buffer=nl.private_hbm)
    cand_v_flat = cand_v.reshape((pad_rows * k,))
    cand_i_flat = cand_i.reshape((pad_rows * k,))

    # Sentinel the pad rows once. They are whole reduce rows, so BFLOAT16_MIN here cannot
    # outrank a real -inf in a REAL row -- it only has to lose within its own row, and that row's
    # result is never read. cand_i's pad rows are left as they are for the same reason.
    if pad_rows > BxS:
        n_pad = pad_rows - BxS
        for t in range(div_ceil(n_pad, PMAX)):
            rows = min(PMAX, n_pad - t * PMAX)
            pad_tile = nl.ndarray((PMAX, k), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.memset(dst=pad_tile[0:rows, :], value=BFLOAT16_MIN)
            nisa.dma_copy(dst=cand_v[nl.ds(BxS + t * PMAX, rows), :], src=pad_tile[0:rows, :])

    lnc_row_start = prg_id * per_lnc_BxS
    n_tiles = div_ceil(per_lnc_BxS, GROUPS_PER_TILE)
    lnc_row_end = min(lnc_row_start + per_lnc_BxS, BxS)

    # Coalesced + double-buffered load, exactly as in _inline_topk's phase 1.
    fast_coalesced = (not has_pad) and (per_lnc_BxS % GROUPS_PER_TILE == 0) and (lnc_row_start + per_lnc_BxS <= BxS)
    kernel_assert(
        peer_width == 0 or fast_coalesced,
        "interleaved_peer_width requires the coalesced whole-tile load path",
    )
    CHUNK_TILES = 1
    n_chunks = div_ceil(n_tiles, CHUNK_TILES)
    _load_engines = [nisa.engine.sync, nisa.engine.scalar]
    # Interleaved path: one buffer PER TILE (four at the production shapes, ~3.1 MB of SBUF
    # that is entirely free post-collective). With buffer REUSE, a reused buffer's load
    # must wait for the sort still reading it, and everything queued behind that descriptor
    # stalls too -- measured as a gap between the first two sorts at gbs=1024. With no
    # reuse every load free-runs from the collective's completion. The contiguous path keeps
    # its historical 3 (it serves every other suite; don't perturb those schedules).
    N_PREFETCH_BUFS = min(4 if peer_width else 3, n_chunks)
    chunk_bufs = []

    def _load_chunk(dst3, first_s1_row, tiles, engine_phase):
        """One chunk-buffer fill from either source layout.

        Contiguous [BxS_tok, vocab]: the whole 128-partition tile is ONE run-per-partition
        DMA. Interleaved receive layout: the partition stride is uniform only WITHIN a
        16-partition group (each group is a different (token, chunk) with its own base), so
        it is one DMA per group -- 8 descriptors instead of 1, all unit-innermost-stride,
        ALTERNATING engines per group: descriptor generation (a fixed cost per dma_copy on one
        engine) is what paces these loads, so splitting a tile's eight across both load
        engines shortens its ready time.
        """
        if not peer_width:
            nisa.dma_copy(
                dst=dst3[nl.ds(0, PMAX), nl.ds(0, tiles), nl.ds(0, n_cols)],
                src=inp_flat.ap(
                    pattern=[[n_cols, PMAX], [PMAX * n_cols, tiles], [1, n_cols]],
                    offset=first_s1_row * vocab,
                ),
                dge_mode=nisa.dge_mode.hwdge,
                engine=_load_engines[engine_phase % 2],
            )
            return
        for tt in range(tiles):
            for grp in range(GROUPS_PER_TILE):
                r = first_s1_row + tt * GROUPS_PER_TILE + grp
                j = r // chunks
                c = r % chunks
                nisa.dma_copy(
                    dst=dst3[nl.ds(grp * PARTS_PER_GROUP, PARTS_PER_GROUP), nl.ds(tt, 1), nl.ds(0, n_cols)],
                    src=inp_flat.ap(
                        pattern=[
                            [ppp * BxS_tok * peer_width, PARTS_PER_GROUP],
                            [BxS_tok * peer_width, ppp],
                            [1, peer_width],
                        ],
                        offset=c * ppc * BxS_tok * peer_width + j * peer_width,
                    ),
                    dge_mode=nisa.dge_mode.hwdge,
                    engine=_load_engines[(engine_phase + grp) % 2],
                )

    if fast_coalesced:
        for _b in range(N_PREFETCH_BUFS):
            chunk_bufs.append(nl.ndarray((PMAX, CHUNK_TILES, n_cols), dtype=nl.bfloat16, buffer=nl.sbuf))
        for pc in range(min(N_PREFETCH_BUFS - 1, n_chunks)):
            prime_row_start = lnc_row_start + pc * CHUNK_TILES * GROUPS_PER_TILE
            prime_tiles = min(CHUNK_TILES, n_tiles - pc * CHUNK_TILES)
            _load_chunk(chunk_bufs[pc % N_PREFETCH_BUFS], prime_row_start, prime_tiles, pc)

    # The constant NaN floor, hoisted OUT of the tile loop: it is the same tile every time, and
    # at the GPT-OSS chunk width the memset covers 128 x 3,142 bf16, which is not a rounding
    # error when there are four tiles. Not built at all under the caller's finite-input
    # contract (config.skip_nan_fold) -- see the fold site in the tile loop.
    _nan_floor = None
    if not skip_nan_fold:
        _nan_floor = nl.ndarray((PMAX, n_cols), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.memset(dst=_nan_floor, value=BFLOAT16_MIN)

    # --- The per-chunk id offset, (r % P) * chunk_width for s1 row r, as a per-partition constant ---
    # One add per 16-partition group is not expressible (a 16-partition access must start at partition 0, 32, 64 or 96),
    # so it is built as a tile with iota's channel_multiplier and folded into the remap. chunk_width == 16 * n_cols, so
    #     global = (r % P)*chunk_width + (s % 16)*n_cols + (s // 16) = (16*(r % P) + (s % 16)) * n_cols + (s // 16)
    # iota seeds 16*(first_row % P) + pp and the AND with (16P - 1) & ~15 (P is a power of two) reduces it to
    # 16*((first_row + pp//16) % P). When P divides GROUPS_PER_TILE every tile starts at a multiple of P and one hoisted
    # tile serves them all; a larger P rebuilds the base per tile (reusing it gives ids that are in range but wrong).
    _base_mask = (PARTS_PER_GROUP * chunks - 1) & ~(PARTS_PER_GROUP - 1)

    def _build_chunk_base(first_row):
        cb = nl.ndarray((PMAX, k_cols), dtype=nl.uint32, buffer=nl.sbuf)
        nisa.iota(
            dst=cb,
            pattern=[[0, k_cols]],
            offset=PARTS_PER_GROUP * (first_row % chunks),
            channel_multiplier=1,
        )
        nisa.tensor_scalar(dst=cb, data=cb, op0=nl.bitwise_and, operand0=_base_mask)
        return cb

    chunk_base = _build_chunk_base(0) if GROUPS_PER_TILE % chunks == 0 else None

    for tile_idx in nl.sequential_range(n_tiles):
        tile_row_start = lnc_row_start + tile_idx * GROUPS_PER_TILE
        tile_row_end = min(tile_row_start + GROUPS_PER_TILE, lnc_row_end)
        n_rows = tile_row_end - tile_row_start
        if n_rows <= 0:
            continue
        par_dim = n_rows * PARTS_PER_GROUP

        if fast_coalesced:
            chunk_idx = tile_idx // CHUNK_TILES
            tile_in_chunk = tile_idx % CHUNK_TILES
            prefetch_chunk = chunk_idx + (N_PREFETCH_BUFS - 1)
            if tile_in_chunk == 0 and prefetch_chunk < n_chunks:
                pf_first_tile = prefetch_chunk * CHUNK_TILES
                pf_row_start = lnc_row_start + pf_first_tile * GROUPS_PER_TILE
                pf_tiles = min(CHUNK_TILES, n_tiles - pf_first_tile)
                _load_chunk(
                    chunk_bufs[prefetch_chunk % N_PREFETCH_BUFS],
                    pf_row_start,
                    pf_tiles,
                    prefetch_chunk,
                )
            src_snake = chunk_bufs[chunk_idx % N_PREFETCH_BUFS][nl.ds(0, PMAX), tile_in_chunk, nl.ds(0, n_cols)]
        elif not has_pad:
            src_snake = nl.ndarray((par_dim, n_cols), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=src_snake[nl.ds(0, par_dim), nl.ds(0, n_cols)],
                src=inp_flat.ap(pattern=[[n_cols, par_dim], [1, n_cols]], offset=tile_row_start * vocab),
                dge_mode=nisa.dge_mode.hwdge,
                engine=nisa.engine.sync,
            )
        else:
            src_snake = nl.ndarray((par_dim, n_cols), dtype=nl.bfloat16, buffer=nl.sbuf)
            # -inf, NOT the most negative finite bf16: a finite sentinel outranks a real -inf
            # logit and pulls padding into the top-k. See _inline_topk's memset comment.
            nisa.memset(dst=src_snake, value=float("-inf"))
            for g in nl.affine_range(n_rows):
                row_off = (tile_row_start + g) * vocab
                base = g * PARTS_PER_GROUP
                q_full = vocab // n_cols
                r_partial = vocab % n_cols
                if q_full > 0:
                    nisa.dma_copy(
                        dst=src_snake[nl.ds(base, q_full), nl.ds(0, n_cols)],
                        src=inp_flat.ap(pattern=[[n_cols, q_full], [1, n_cols]], offset=row_off),
                    )
                if r_partial != 0:
                    nisa.dma_copy(
                        dst=src_snake[nl.ds(base + q_full, 1), nl.ds(0, r_partial)],
                        src=inp_flat.ap(pattern=[[1, r_partial]], offset=row_off + q_full * n_cols),
                    )

        # Fold NaN out of the snake BEFORE nisa.topk selects from it: afterwards the instruction
        # has already emitted a (value, position) pair that does not correspond, and no later
        # clamp can undo that. (x != x) is true only for NaN. Skipped under the caller's
        # finite-input contract (config.skip_nan_fold): with no NaN the predicate can never
        # fire, yet the compare + predicated copy span the full chunk width -- the widest
        # non-topk work in this loop. +/-inf are orderable and never needed the fold.
        if not skip_nan_fold:
            _nan_mask = nl.ndarray((par_dim, n_cols), dtype=nl.uint8, buffer=nl.sbuf)
            nisa.tensor_tensor(
                dst=_nan_mask[nl.ds(0, par_dim), :],
                data1=src_snake[nl.ds(0, par_dim), nl.ds(0, n_cols)],
                data2=src_snake[nl.ds(0, par_dim), nl.ds(0, n_cols)],
                op=nl.not_equal,
            )
            nisa.tensor_copy_predicated(
                dst=src_snake[nl.ds(0, par_dim), nl.ds(0, n_cols)],
                src=_nan_floor[nl.ds(0, par_dim), :],
                predicate=_nan_mask[nl.ds(0, par_dim), :],
            )

        val_snake = nl.ndarray((par_dim, k_pad), dtype=nl.bfloat16, buffer=nl.sbuf)
        idx_snake = nl.ndarray((par_dim, k_pad), dtype=nl.uint32, buffer=nl.sbuf)
        nisa.topk(val_dst=val_snake[:, nl.ds(0, k)], idx_dst=idx_snake[:, nl.ds(0, k)], src=src_snake, n=full_n)

        # Stage the positions through a dge_mode.unknown DMA before any compute engine reads them. Reading idx_snake
        # directly on Vector (a GpSIMD-write to Vector-read edge nothing else here has) read it before nisa.topk's
        # stores were visible on hardware: wrong ids with the right values on one row of a 128-row sweep. The staging
        # DMA is the same handoff the de-snake stores use.
        idx_stage = nl.ndarray((PMAX, k_cols), dtype=nl.uint32, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=idx_stage[nl.ds(0, par_dim), :],
            src=idx_snake[nl.ds(0, par_dim), nl.ds(0, k_cols)],
            dge_mode=desnake_dge,
        )

        # --- The remap, on the snake tile (k_cols wide, not k) ---
        # The blocked load put chunk-local id p*n_cols + c at snake position s = p + 16*c, and the chunk offset folds
        # into the same multiply (see _build_chunk_base):
        #     global = (16*(g % P) + (s % 16)) * n_cols + (s // 16)
        # 16 is a power of two, so the split is exact with bit ops on the uint32 positions.
        c_u32 = nl.ndarray((PMAX, k_cols), dtype=nl.uint32, buffer=nl.sbuf)
        q_u32 = nl.ndarray((PMAX, k_cols), dtype=nl.uint32, buffer=nl.sbuf)
        c_f32 = nl.ndarray((PMAX, k_cols), dtype=nl.float32, buffer=nl.sbuf)
        gid = nl.ndarray((PMAX, k_cols), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=c_u32[nl.ds(0, par_dim), :],
            data=idx_stage[nl.ds(0, par_dim), :],
            op0=nl.right_shift,
            operand0=4,  # log2(PARTS_PER_GROUP); an int method call is rejected by the parser
        )
        # Read from idx_stage, NOT from c_u32: two ops reading one source and writing two
        # distinct destinations have no hazard between them, whereas chaining them through a
        # shared buffer creates exactly the write-after-read pair that has bitten this kernel.
        nisa.tensor_scalar(
            dst=q_u32[nl.ds(0, par_dim), :],
            data=idx_stage[nl.ds(0, par_dim), :],
            op0=nl.bitwise_and,
            operand0=PARTS_PER_GROUP - 1,
        )
        # + 16*(row % P), the chunk offset expressed in snake partitions. Both operands are
        # uint32 and bounded by 15 and 16*(P-1) <= 2032, so the sum cannot overflow anything
        # (and the fp32 gid below stays exact: ids are < 2^24 by kernel contract).
        _cb = chunk_base if chunk_base is not None else _build_chunk_base(tile_row_start)
        nisa.tensor_tensor(
            dst=q_u32[nl.ds(0, par_dim), :],
            data1=q_u32[nl.ds(0, par_dim), :],
            data2=_cb[nl.ds(0, par_dim), :],
            op=nl.add,
        )
        nisa.tensor_copy(dst=c_f32[nl.ds(0, par_dim), :], src=c_u32[nl.ds(0, par_dim), :])
        nisa.tensor_scalar(
            dst=gid[nl.ds(0, par_dim), :],
            data=q_u32[nl.ds(0, par_dim), :],
            op0=nl.multiply,
            operand0=float(n_cols),
        )
        nisa.tensor_tensor(
            dst=gid[nl.ds(0, par_dim), :],
            data1=gid[nl.ds(0, par_dim), :],
            data2=c_f32[nl.ds(0, par_dim), :],
            op=nl.add,
        )
        # Keep every id in [0, vocab): one can only escape when its position was garbage (unorderable values), and an
        # out-of-range id faults a later table gather. The clamp is to the full vocab rather than the chunk because the
        # offset is already folded in; such an id is unpaired from its value either way.
        nisa.tensor_scalar(
            dst=gid[nl.ds(0, par_dim), :],
            data=gid[nl.ds(0, par_dim), :],
            op0=nl.minimum,
            operand0=float(chunks * vocab - 1),
        )

        # TAP: the RAW snake positions, de-snaked through the same pattern as everything else so
        # the tap means exactly what it means on the general path. It reads idx_stage -- the
        # staged buffer the remap consumed -- so an L2 break really does mean the remap's own
        # arithmetic disagreed with its own input, not that the two taps read one racy buffer
        # at different times.
        if tap_pos is not None:
            nisa.dma_copy(
                dst=tap_pos.reshape((BxS * k,)).ap(
                    pattern=[[k_cols, par_dim], [1, k_cols]], offset=tile_row_start * k_pad
                ),
                src=idx_stage[nl.ds(0, par_dim), :],
                dge_mode=desnake_dge,
            )

        # De-snake straight into the candidate arrays. One contiguous par_dim-partition DMA per
        # tensor per tile, as in _inline_topk -- the only difference is where it lands.
        tile_out_off = tile_row_start * k_pad
        nisa.dma_copy(
            dst=cand_v_flat.ap(pattern=[[k_cols, par_dim], [1, k_cols]], offset=tile_out_off),
            src=val_snake[nl.ds(0, par_dim), nl.ds(0, k_cols)],
            dge_mode=desnake_dge,
        )
        nisa.dma_copy(
            dst=cand_i_flat.ap(pattern=[[k_cols, par_dim], [1, k_cols]], offset=tile_out_off),
            src=gid[nl.ds(0, par_dim), nl.ds(0, k_cols)],
            dge_mode=desnake_dge,
        )

    return cand_v, cand_i


def _inline_topk(
    inp: nl.NkiTensor, config, tap_pos=None, fused=None, skip_nan_fold=False
) -> Tuple[nl.NkiTensor, nl.NkiTensor, nl.NkiTensor]:
    """Top-k over ``inp[BxS, V]`` with ``nisa.topk``, an inlined copy of the kernel library's ``gpsimd_topk``.

    Args:
        inp: ``[BxS, V]`` bf16 in HBM, ``8 <= V < 65536``.
        config: a ``TopkConfig`` (``create_topk_config``).
        tap_pos: optional ``[BxS, k]`` fp32 HBM tap for the raw snake positions phase 2 worked from.
        fused: optional ``_FUSE_*`` tuple that replaces the plain output store with the reduce's tail: translate the
            positions through the caller's candidate ids and write the final values and global ids directly. Only for
            the second stage, whose "vocab index" is a candidate position.

    Returns:
        ``(values, indices, desnaked_idx [BxS, k_pad] f32)``. Without ``fused`` the first two are this call's own
        ``[BxS, k]`` buffers, indices into ``inp``'s last dimension, descending when ``config.sorted``; with ``fused``
        they are the caller's destinations holding global ids. The third is the phase-1 de-snake index buffer, for a
        tap that re-reads it.
    """
    _validate_topk(config)

    BxS = config.BxS
    vocab = config.vocab_size
    k = config.k
    index_dtype = config.index_dtype

    # Resolve runtime sharding (LNC). BxS == 1 collapses to a single program. chunked_topk has
    # already asserted n_prgs == 1, so this only ever resolves to (1, 0) here; it is kept
    # verbatim so the inline stays a faithful copy.
    shard_info = get_verified_program_sharding_info("topk", (0, 1), 2)
    if BxS > 1:
        n_prgs = shard_info[1]
        prg_id = shard_info[2]
    else:
        n_prgs = 1
        prg_id = 0

    per_lnc_BxS = div_ceil(BxS, n_prgs)

    # Without fusion this stage publishes its own [BxS, k] results; with it the caller's
    # destinations are written in place and these are never allocated.
    if fused is None:
        topk_values = nl.ndarray((BxS, k), dtype=inp.dtype, buffer=nl.private_hbm)
        topk_indices = nl.ndarray((BxS, k), dtype=index_dtype, buffer=nl.private_hbm)
    else:
        topk_values = fused[_FUSE_OUT_VALS]
        topk_indices = fused[_FUSE_OUT_IDX]

    # Snake free-dim sizes (BLOCKED layout: snake[p, c] = inp[row, p*n_cols + c], so snake
    # position s = p + 16*c holds vocab index p*n_cols + c -- inverted by the remap below).
    n_cols = div_ceil(vocab, PARTS_PER_GROUP)
    full_n = PARTS_PER_GROUP * n_cols  # the n passed to nisa.topk; padded slots hold -inf
    q_full = vocab // n_cols  # fully-valid partitions
    r_partial = vocab % n_cols  # valid elements in the partial partition q_full
    has_pad = full_n != vocab  # some snake slots are padding (vocab % 16 != 0)

    inp_flat = inp.reshape((BxS * vocab,))

    k_cols = div_ceil(k, PARTS_PER_GROUP)
    k_pad = k_cols * PARTS_PER_GROUP  # >= k; padded snake-column width

    # fast_dma_safe: this shard's phase-1 tiles are ALL full 128-partition tiles, which is what
    # makes a bf16 de-snake round trip safe against the narrow-dtype strided store. It does NOT
    # select the DGE mode -- see the desnake_dge comment.
    fast_dma_safe = (per_lnc_BxS % GROUPS_PER_TILE == 0) and (BxS == n_prgs * per_lnc_BxS)
    # _validate_chunked_topk already refused a sorted request without the hardware sort; this guards direct callers.
    if config.sorted:
        kernel_assert(
            _HW_SORT_AVAILABLE and k <= _HW_SORT_MAX_K,
            "sorted top-k requires nisa.topk(sorted=True) "
            f"and k <= {_HW_SORT_MAX_K}; the software sort was removed",
        )
    hw_sort = bool(config.sorted)
    # dge_mode.unknown always: with dge_mode.none the de-snake store's descriptors are generated out of band and nothing
    # orders it against the reload of the same buffer (repeated ids with the right values on 2 of 128 row blocks at 128
    # ranks). It costs time; see the store/reload race in the library's gpsimd_topk.
    desnake_dge = nisa.dge_mode.unknown
    asc_val_dtype = nl.bfloat16 if fast_dma_safe else nl.float32
    asc_val_hbm = nl.ndarray((BxS, k_pad), dtype=asc_val_dtype, buffer=nl.private_hbm)
    asc_idx_hbm = nl.ndarray((BxS, k_pad), dtype=nl.float32, buffer=nl.private_hbm)
    asc_val_flat = asc_val_hbm.reshape((BxS * k_pad,))
    asc_idx_flat = asc_idx_hbm.reshape((BxS * k_pad,))

    lnc_row_start = prg_id * per_lnc_BxS
    n_tiles = div_ceil(per_lnc_BxS, GROUPS_PER_TILE)
    lnc_row_end = min(lnc_row_start + per_lnc_BxS, BxS)

    # --- Coalesced + double-buffered phase-1 load ------------------------------------------
    # When 16 | vocab and every tile is full, CHUNK_TILES tiles are ONE contiguous HBM block
    # (partition stride n_cols, tile stride 128*n_cols, free stride 1), so the load is a few
    # large HWDGE transfers into rolling buffers instead of one latency-bound transfer per
    # tile. N_PREFETCH_BUFS distinct buffers keep loads in flight while GpSIMD drains another.
    fast_coalesced = (not has_pad) and (per_lnc_BxS % GROUPS_PER_TILE == 0) and (lnc_row_start + per_lnc_BxS <= BxS)
    CHUNK_TILES = 1
    n_chunks = div_ceil(n_tiles, CHUNK_TILES)
    # Round-robin the loads across two DGE sequencer queues so consecutive loads run
    # concurrently instead of serializing on one sync queue.
    _load_engines = [nisa.engine.sync, nisa.engine.scalar]
    N_PREFETCH_BUFS = min(3, n_chunks)
    chunk_bufs = []
    if fast_coalesced:
        for _b in range(N_PREFETCH_BUFS):
            chunk_bufs.append(nl.ndarray((PMAX, CHUNK_TILES, n_cols), dtype=nl.bfloat16, buffer=nl.sbuf))
        n_prime = min(N_PREFETCH_BUFS - 1, n_chunks)
        for pc in range(n_prime):
            prime_first_tile = pc * CHUNK_TILES
            prime_row_start = lnc_row_start + prime_first_tile * GROUPS_PER_TILE
            # The last chunk may hold fewer tiles; clamp so the DMA never over-reads.
            prime_tiles = min(CHUNK_TILES, n_tiles - prime_first_tile)
            nisa.dma_copy(
                dst=chunk_bufs[pc % N_PREFETCH_BUFS][nl.ds(0, PMAX), nl.ds(0, prime_tiles), nl.ds(0, n_cols)],
                src=inp_flat.ap(
                    pattern=[[n_cols, PMAX], [PMAX * n_cols, prime_tiles], [1, n_cols]],
                    offset=prime_row_start * vocab,
                ),
                dge_mode=nisa.dge_mode.hwdge,
                engine=_load_engines[pc % 2],
            )

    # Phase 1, per 8-row tile: blocked load, nisa.topk, and the de-snake of the k (value, snake position) pairs into the
    # [BxS, k_pad] HBM buffers. Phase 2 orders them once, one row per partition.
    for tile_idx in nl.sequential_range(n_tiles):
        tile_row_start = lnc_row_start + tile_idx * GROUPS_PER_TILE
        tile_row_end = min(tile_row_start + GROUPS_PER_TILE, lnc_row_end)
        n_rows = tile_row_end - tile_row_start
        if n_rows <= 0:
            continue

        par_dim = n_rows * PARTS_PER_GROUP  # multiple of 16

        if fast_coalesced:
            chunk_idx = tile_idx // CHUNK_TILES
            tile_in_chunk = tile_idx % CHUNK_TILES
            # Keep the pipeline full: at the first tile of a chunk, prefetch the chunk
            # (N_PREFETCH_BUFS-1) ahead into a slot nothing is still consuming, so no
            # anti-dependency forces the load to wait.
            prefetch_chunk = chunk_idx + (N_PREFETCH_BUFS - 1)
            if tile_in_chunk == 0 and prefetch_chunk < n_chunks:
                pf_first_tile = prefetch_chunk * CHUNK_TILES
                pf_row_start = lnc_row_start + pf_first_tile * GROUPS_PER_TILE
                pf_tiles = min(CHUNK_TILES, n_tiles - pf_first_tile)
                nisa.dma_copy(
                    dst=chunk_bufs[prefetch_chunk % N_PREFETCH_BUFS][
                        nl.ds(0, PMAX), nl.ds(0, pf_tiles), nl.ds(0, n_cols)
                    ],
                    src=inp_flat.ap(
                        pattern=[[n_cols, PMAX], [PMAX * n_cols, pf_tiles], [1, n_cols]],
                        offset=pf_row_start * vocab,
                    ),
                    dge_mode=nisa.dge_mode.hwdge,
                    engine=_load_engines[prefetch_chunk % 2],
                )
            src_snake = chunk_bufs[chunk_idx % N_PREFETCH_BUFS][nl.ds(0, PMAX), tile_in_chunk, nl.ds(0, n_cols)]
        elif not has_pad:
            src_snake = nl.ndarray((par_dim, n_cols), dtype=nl.bfloat16, buffer=nl.sbuf)
            # 16 | vocab but the coalesced precondition does not hold. The whole-tile blocked
            # load is still ONE contiguous HBM run (partition stride n_cols, free stride 1),
            # emitted as a single par_dim-partition DMA. HWDGE on Sync keeps descriptor
            # generation off GpSIMD so the load overlaps the previous tile's top-k.
            nisa.dma_copy(
                dst=src_snake[nl.ds(0, par_dim), nl.ds(0, n_cols)],
                src=inp_flat.ap(pattern=[[n_cols, par_dim], [1, n_cols]], offset=tile_row_start * vocab),
                dge_mode=nisa.dge_mode.hwdge,
                engine=nisa.engine.sync,
            )
        else:
            src_snake = nl.ndarray((par_dim, n_cols), dtype=nl.bfloat16, buffer=nl.sbuf)
            # vocab % 16 != 0: some snake slots are padding. The sentinel is -inf, NOT the
            # most-negative FINITE bf16: a finite sentinel outranks a real -inf logit, so on a
            # masked row padding wins the top-k and the kernel returns a value the input never
            # contained paired with a clamped index. See BFLOAT16_MIN in decode/lm_head/topk_layout.py.
            nisa.memset(dst=src_snake, value=float("-inf"))

            for g in nl.affine_range(n_rows):
                row = tile_row_start + g
                base = g * PARTS_PER_GROUP
                row_off = row * vocab
                if q_full > 0:
                    nisa.dma_copy(
                        dst=src_snake[nl.ds(base, q_full), nl.ds(0, n_cols)],
                        src=inp_flat.ap(pattern=[[n_cols, q_full], [1, n_cols]], offset=row_off),
                    )
                if r_partial != 0:
                    nisa.dma_copy(
                        dst=src_snake[nl.ds(base + q_full, 1), nl.ds(0, r_partial)],
                        src=inp_flat.ap(pattern=[[1, r_partial]], offset=row_off + q_full * n_cols),
                    )

        # Fold NaN to BFLOAT16_MIN before nisa.topk selects (it is undefined on NaN; x != x only for NaN). BFLOAT16_MIN
        # rather than NEG_INF_FLOOR because the snake is bf16. The reduce skips this: its input is stage 1's folded
        # output.
        if not skip_nan_fold:
            _nan_mask = nl.ndarray((par_dim, n_cols), dtype=nl.uint8, buffer=nl.sbuf)
            nisa.tensor_tensor(
                dst=_nan_mask[nl.ds(0, par_dim), :],
                data1=src_snake[nl.ds(0, par_dim), nl.ds(0, n_cols)],
                data2=src_snake[nl.ds(0, par_dim), nl.ds(0, n_cols)],
                op=nl.not_equal,
            )
            _nan_floor = nl.ndarray((par_dim, n_cols), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.memset(dst=_nan_floor[nl.ds(0, par_dim), :], value=BFLOAT16_MIN)
            nisa.tensor_copy_predicated(
                dst=src_snake[nl.ds(0, par_dim), nl.ds(0, n_cols)],
                src=_nan_floor[nl.ds(0, par_dim), :],
                predicate=_nan_mask[nl.ds(0, par_dim), :],
            )

        # n = full_n (NOT vocab): with the blocked layout the valid vocab elements are not a
        # contiguous prefix of snake positions, so all full_n slots are active.
        val_snake = nl.ndarray((par_dim, k_pad), dtype=nl.bfloat16, buffer=nl.sbuf)
        idx_snake = nl.ndarray((par_dim, k_pad), dtype=nl.uint32, buffer=nl.sbuf)
        if hw_sort:
            # sorted=True puts rank j at snake position j, which is what lets phase 2 replace
            # the entire descending sort with a fixed column permutation. Two call forms because
            # passing `sorted=` to an nki without it is a trace-time TypeError.
            nisa.topk(
                val_dst=val_snake[:, nl.ds(0, k)],
                idx_dst=idx_snake[:, nl.ds(0, k)],
                src=src_snake,
                n=full_n,
                sorted=True,
            )
        else:
            nisa.topk(val_dst=val_snake[:, nl.ds(0, k)], idx_dst=idx_snake[:, nl.ds(0, k)], src=src_snake, n=full_n)
        # k_pad > k leaves snake positions [k, k_pad) unwritten. They live at a PARTITION
        # SUBRANGE here, which a full-partition memset cannot target, so they are masked after
        # the de-snake reload in phase 2 where the layout makes it a legal free-dim write.
        # (The hw_sort path needs no masking at all: it reads BY POSITION, and positions
        # [k, k_pad) are never among the ranks it asks for.)

        # --- De-snake the topk output into a CONTIGUOUS [rows, k_pad] HBM tile ---
        # asc[row, p*k_cols + c] = val_snake[base + p, c]. Because k_pad == 16*k_cols,
        # consecutive row-groups are exactly k_pad apart, so the WHOLE par_dim-partition tile
        # is ONE contiguous HBM run: a single DMA per tile rather than n_rows of them (the
        # de-snake stores were the top serial contributors in the HW profile).
        tile_out_off = tile_row_start * k_pad
        nisa.dma_copy(
            dst=asc_val_flat.ap(pattern=[[k_cols, par_dim], [1, k_cols]], offset=tile_out_off),
            src=val_snake[nl.ds(0, par_dim), nl.ds(0, k_cols)],
            dge_mode=desnake_dge,
        )
        nisa.dma_copy(
            dst=asc_idx_flat.ap(pattern=[[k_cols, par_dim], [1, k_cols]], offset=tile_out_off),
            src=idx_snake[nl.ds(0, par_dim), nl.ds(0, k_cols)],
            dge_mode=desnake_dge,
        )

    # =====================================================================
    # PHASE 2 -- batch-parallel descending sort, ONE invocation over up to 128
    # rows (one row per partition) rather than one per 8-row tile.
    # =====================================================================
    n_pass = div_ceil(k, 8)
    log2_ppg = 4  # == log2(PARTS_PER_GROUP)
    n_sort_tiles = div_ceil(per_lnc_BxS, PMAX)

    for sort_idx in nl.sequential_range(n_sort_tiles):
        sort_row_start = lnc_row_start + sort_idx * PMAX
        sort_row_end = min(sort_row_start + PMAX, lnc_row_end)
        n_srows = sort_row_end - sort_row_start
        if n_srows <= 0:
            continue

        out_v = nl.ndarray((PMAX, n_pass * 8), dtype=nl.float32, buffer=nl.sbuf)
        out_i = nl.ndarray((PMAX, n_pass * 8), dtype=nl.float32, buffer=nl.sbuf)

        if hw_sort:
            # The hardware sort. nisa.topk(sorted=True) puts rank j at snake position j and the de-snake puts position s
            # at column (s % 16)*k_cols + (s // 16), so descending order is a fixed column permutation: with j = a +
            # 16b, rank j sits at column a*k_cols + b. The reload stays one contiguous run per row, and 16 tensor_copy
            # ops per tensor move each k_cols-wide block for one a into the strided [:, :, a] slice. Folding the
            # permutation into the reload's access pattern instead makes every element its own 4-byte transfer (much
            # slower at gbs=128).

            # The instruction reports the index of each value it selected, so there is no value-matched index recovery:
            # ties and -inf need no sort-key clamps, and the unwritten positions [k, k_pad) are never read. The phase-1
            # NaN fold still applies.
            pv = nl.ndarray((PMAX, k_cols, PARTS_PER_GROUP), dtype=nl.float32, buffer=nl.sbuf)
            pi = nl.ndarray((PMAX, k_cols, PARTS_PER_GROUP), dtype=nl.float32, buffer=nl.sbuf)
            lv = nl.ndarray((PMAX, k_pad), dtype=nl.float32, buffer=nl.sbuf)
            li = nl.ndarray((PMAX, k_pad), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=lv[nl.ds(0, n_srows), :],
                src=asc_val_hbm[nl.ds(sort_row_start, n_srows), :],
                dge_mode=desnake_dge,
            )
            nisa.dma_copy(
                dst=li[nl.ds(0, n_srows), :],
                src=asc_idx_hbm[nl.ds(sort_row_start, n_srows), :],
                dge_mode=desnake_dge,
            )
            for _a in range(PARTS_PER_GROUP):
                nisa.tensor_copy(
                    dst=pv[nl.ds(0, n_srows), :, _a], src=lv[nl.ds(0, n_srows), nl.ds(_a * k_cols, k_cols)]
                )
                nisa.tensor_copy(
                    dst=pi[nl.ds(0, n_srows), :, _a], src=li[nl.ds(0, n_srows), nl.ds(_a * k_cols, k_cols)]
                )
            # The 3D tile's free dims are contiguous, so this view is the [n_srows, k_pad]
            # descending array the shared tail below expects. k <= k_pad, so the tail's
            # [:, 0:k] slice takes ranks 0..k-1.
            out_v = pv.reshape((PMAX, k_pad))
            out_i = pi.reshape((PMAX, k_pad))
        else:
            # ===== Unsorted fast path (config.sorted == False) ================
            # Skip the sort; COMPACT the valid de-snaked columns into the output prefix
            # [0, k). nisa.topk writes only snake positions [0, k), and the de-snake scatters
            # those among the k_pad columns when k % 16 != 0; for k % 16 == 0 this is the
            # single full-width run (0, 0, k).
            cv = nl.ndarray((PMAX, k_pad), dtype=nl.float32, buffer=nl.sbuf)
            ci = nl.ndarray((PMAX, k_pad), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=cv[nl.ds(0, n_srows), :],
                src=asc_val_hbm[nl.ds(sort_row_start, n_srows), :],
                dge_mode=desnake_dge,
            )
            nisa.dma_copy(
                dst=ci[nl.ds(0, n_srows), :],
                src=asc_idx_hbm[nl.ds(sort_row_start, n_srows), :],
                dge_mode=desnake_dge,
            )
            _valid_runs = _valid_desnake_col_runs(k, k_cols, k_pad)
            # The parser frontend rejects tuple-unpacking in a for-target, so index explicitly.
            for _run_idx in range(len(_valid_runs)):
                _src_col = _valid_runs[_run_idx][0]
                _dst_col = _valid_runs[_run_idx][1]
                _run_len = _valid_runs[_run_idx][2]
                nisa.tensor_copy(
                    dst=out_v[nl.ds(0, n_srows), nl.ds(_dst_col, _run_len)],
                    src=cv[nl.ds(0, n_srows), nl.ds(_src_col, _run_len)],
                )
                nisa.tensor_copy(
                    dst=out_i[nl.ds(0, n_srows), nl.ds(_dst_col, _run_len)],
                    src=ci[nl.ds(0, n_srows), nl.ds(_src_col, _run_len)],
                )

        # TAP: the raw snake positions phase 2 ended up with, before the remap. Reads out_i
        # full-width, which the remap immediately below does too, so it adds no ordering edge
        # that is not already there -- the property that keeps a tap from hiding the race it is
        # meant to catch (see DEBUG TAPS).
        if tap_pos is not None:
            nisa.dma_copy(dst=tap_pos[nl.ds(sort_row_start, n_srows), :], src=out_i[nl.ds(0, n_srows), nl.ds(0, k)])

        # --- Snake position to vocab index ---
        # The blocked load put vocab index p*n_cols + c at snake position s = p + 16*c, so v = (s % 16)*n_cols + (s //
        # 16), split exactly with uint32 bit ops and recombined in fp32 (exact below 2^24). k columns wide: only [0, k)
        # is stored, and the branches above leave out_i at different widths.
        s_u32 = nl.ndarray((PMAX, k), dtype=nl.uint32, buffer=nl.sbuf)
        c_u32 = nl.ndarray((PMAX, k), dtype=nl.uint32, buffer=nl.sbuf)
        p_u32 = nl.ndarray((PMAX, k), dtype=nl.uint32, buffer=nl.sbuf)
        c_f32 = nl.ndarray((PMAX, k), dtype=nl.float32, buffer=nl.sbuf)
        remapped_i = nl.ndarray((PMAX, k), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=s_u32[nl.ds(0, n_srows), :], src=out_i[nl.ds(0, n_srows), nl.ds(0, k)])
        nisa.tensor_scalar(
            dst=c_u32[nl.ds(0, n_srows), :],
            data=s_u32[nl.ds(0, n_srows), :],
            op0=nl.right_shift,
            operand0=log2_ppg,
        )
        nisa.tensor_scalar(
            dst=p_u32[nl.ds(0, n_srows), :],
            data=s_u32[nl.ds(0, n_srows), :],
            op0=nl.bitwise_and,
            operand0=PARTS_PER_GROUP - 1,
        )
        nisa.tensor_copy(dst=c_f32[nl.ds(0, n_srows), :], src=c_u32[nl.ds(0, n_srows), :])
        nisa.tensor_scalar(
            dst=remapped_i[nl.ds(0, n_srows), :],
            data=p_u32[nl.ds(0, n_srows), :],
            op0=nl.multiply,
            operand0=float(n_cols),
        )
        nisa.tensor_tensor(
            dst=remapped_i[nl.ds(0, n_srows), :],
            data1=remapped_i[nl.ds(0, n_srows), :],
            data2=c_f32[nl.ds(0, n_srows), :],
            op=nl.add,
        )
        # Every returned index must be in [0, vocab). Two ways it could escape: a padding slot
        # that tied real -inf data and remapped up to full_n - 1, or a gathered snake position
        # that was itself garbage. An out-of-range index breaks the contract and, if it feeds a
        # table gather, faults the device. Unconditional -- one Vector op on k columns.
        nisa.tensor_scalar(
            dst=remapped_i[nl.ds(0, n_srows), :],
            data=remapped_i[nl.ds(0, n_srows), :],
            op0=nl.minimum,
            operand0=float(vocab - 1),
        )

        if fused is None:
            nisa.dma_copy(dst=topk_values[nl.ds(sort_row_start, n_srows), :], src=out_v[nl.ds(0, n_srows), nl.ds(0, k)])
            nisa.dma_copy(
                dst=topk_indices[nl.ds(sort_row_start, n_srows), :], src=remapped_i[nl.ds(0, n_srows), nl.ds(0, k)]
            )
            continue

        # The fused reduce tail. For the second stage remapped_i is a position in the candidate array, not a vocab id,
        # so it is translated through the caller's candidate ids here, from SBUF, and the values go straight to the
        # caller's output; unfused that is a store, reload and cast of the positions plus a round trip of the values.
        # The positions are already fp32 and clamped to [0, red_w).
        n_out = fused[_FUSE_N_ROWS]
        rows = min(n_srows, n_out - sort_row_start)
        if rows <= 0:
            continue
        p_chunks = fused[_FUSE_CHUNKS]
        chunk_width = fused[_FUSE_WIDTH]
        red_w = p_chunks * k

        # Values: ONE fp32 SBUF -> bf16 HBM DMA, the cast riding on the transfer.
        nisa.dma_copy(dst=fused[_FUSE_OUT_VALS][nl.ds(sort_row_start, rows), :], src=out_v[nl.ds(0, rows), nl.ds(0, k)])

        # Index normalisation. fp32 throughout: every global id is < vocab <= 2^24, so the
        # uint32 -> fp32 -> uint32 round trip is exact.
        if fused[_FUSE_IDS_GLOBAL]:
            # _stage1_desnake_only already remapped and offset the ids, on a k_cols-wide tile,
            # and wrote them as float32 -- and the Tensor-Engine lookup below reads them
            # STRAIGHT FROM HBM (the d_all loads), so no SBUF copy of the full candidate row
            # exists on this path at all. Only the cand_i tap, which captures exactly what the
            # lookup consumes, materialises one.
            i_f = None
            if fused[_FUSE_TAP_CAND] is not None:
                i_f = nl.ndarray((PMAX, red_w), dtype=nl.float32, buffer=nl.sbuf)
                nisa.dma_copy(dst=i_f[0:rows, :], src=fused[_FUSE_CAND_I][nl.ds(sort_row_start, rows), :])
        else:
            i_f = nl.ndarray((PMAX, red_w), dtype=nl.float32, buffer=nl.sbuf)
            iu = nl.ndarray((PMAX, red_w), dtype=nl.uint32, buffer=nl.sbuf)
            nisa.dma_copy(dst=iu[0:rows, :], src=fused[_FUSE_CAND_I][nl.ds(sort_row_start, rows), :])
            nisa.tensor_copy(dst=i_f[0:rows, :], src=iu[0:rows, :])
            # global = c*width + chunk_local. c is constant per column block, so this is a
            # free-dim slice add per chunk. Chunk 0 needs no offset.
            for c in range(1, p_chunks):
                nisa.tensor_scalar(
                    dst=i_f[0:rows, nl.ds(c * k, k)],
                    data=i_f[0:rows, nl.ds(c * k, k)],
                    op0=nl.add,
                    operand0=float(c * chunk_width),
                )
        if fused[_FUSE_TAP_CAND] is not None:
            nisa.dma_copy(dst=fused[_FUSE_TAP_CAND][nl.ds(sort_row_start, rows), :], src=i_f[0:rows, :])
        if fused[_FUSE_TAP_POS] is not None:
            nisa.dma_copy(dst=fused[_FUSE_TAP_POS][nl.ds(sort_row_start, rows), :], src=remapped_i[0:rows, nl.ds(0, k)])

        # The position-to-id lookup, got[r, j] = i_f[r, remapped_i[r, j]], is a one-hot contraction on the PE rather
        # than an nc_n_gather. On hardware the gather's consumer read stale zeros from the tail of its output despite a
        # correct dependency edge, and neither a DMA fence nor a GpSIMD copy or cast fence closed the window. The PE's
        # PSUM handoff to Vector is the well-trodden edge. With pos = q*c_w + c and c_w = min(128, red_w):
        #     D[(g, q), c] = i_f[r0+g, q*c_w + c]                 the group's candidates, fp32
        #     B[(g, q), (g', j)] = (g*n_q + q == key[g', j])      key = g'*n_q + q_{g',j}
        #     M[c, (g', j)] = sum_p D[p, c] * B[p, (g', j)]       = i_f[r0+g', q_j*c_w + c]
        #     got[g', j] = sum_c M[c, (g', j)] * (c == c_{g',j})  the c-selection and a ones-reduce
        # All of it is exact in fp32: ids are below 2^24 and every one-hot column matches one partition, so each PSUM
        # sum is one product plus zeros. A padded tail chunk is zeroed so B's zeros never meet uninitialized bits (0 *
        # NaN is NaN).
        c_w = min(PMAX, red_w)
        n_q = div_ceil(red_w, c_w)
        # Rows per matmul group: the block-diagonal key trick batches g rows into one contraction
        # so long as their K fits the partition dim and their flattened outputs fit one PSUM-bank
        # moving tile. k > _PSUM_MOVING falls back to one row per group, tiled along k.
        _PSUM_MOVING = 512
        g_max = max(1, min(PMAX // n_q, _PSUM_MOVING // k if k <= _PSUM_MOVING else 1))
        # Shared small constants, sliced per use: one integer ramp down the partitions (serves
        # both the K-side and the c-side one-hot compares) and a ones vector for the broadcasts
        # and the partition reduction.
        iota_pf = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        _iota_pu = nl.ndarray((PMAX, 1), dtype=nl.uint32, buffer=nl.sbuf)
        nisa.iota(dst=_iota_pu, pattern=[[0, 1]], channel_multiplier=1)
        nisa.tensor_copy(dst=iota_pf, src=_iota_pu)
        ones_row = nl.ndarray((1, PMAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=ones_row, value=1.0)
        ones_col = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=ones_col, value=1.0)
        # bf16 twin of the ones vector, for the two BROADCAST matmuls (key_b, c_b) below.
        # Those carry only small integers (keys < 128, c < c_w <= 128 -- all exact in bf16's
        # 8 significand bits) and every PSUM element is ONE exact product, so running them at
        # the PE's bf16 rate instead of the quarter-rate fp32 mode is BITWISE free. The id
        # contraction (m_ps) and the ones-reduce (sel) carry vocab ids up to 2^24 and STAY
        # fp32 -- see the exactness paragraph above.
        ones_row_bf = nl.ndarray((1, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.memset(dst=ones_row_bf, value=1.0)

        for r0 in range(0, rows, g_max):
            g = min(g_max, rows - r0)
            n_flat = g * k
            k_g = g * n_q
            # The group's positions, flattened onto one partition, and split into (q, c). The
            # per-row DMAs are partition moves; everything after them is exact fp32 arithmetic.
            pos_flat = nl.ndarray((1, n_flat), dtype=nl.float32, buffer=nl.sbuf)
            for gi in range(g):
                nisa.dma_copy(
                    dst=pos_flat[0:1, nl.ds(gi * k, k)],
                    src=remapped_i[nl.ds(r0 + gi, 1), nl.ds(0, k)],
                )
            key_f = nl.ndarray((1, n_flat), dtype=nl.float32, buffer=nl.sbuf)
            c_f = nl.ndarray((1, n_flat), dtype=nl.float32, buffer=nl.sbuf)
            key_bf = nl.ndarray((1, n_flat), dtype=nl.bfloat16, buffer=nl.sbuf)
            c_bf = nl.ndarray((1, n_flat), dtype=nl.bfloat16, buffer=nl.sbuf)
            if n_q == 1:
                # red_w <= 128: q is identically zero and c is the position itself.
                nisa.tensor_copy(dst=c_f[0:1, :], src=pos_flat[0:1, :])
                nisa.memset(dst=key_f, value=0.0)
            else:
                # c_w == 128 here (red_w > 128), so the split is exact bit arithmetic.
                pos_u = nl.ndarray((1, n_flat), dtype=nl.uint32, buffer=nl.sbuf)
                q_u = nl.ndarray((1, n_flat), dtype=nl.uint32, buffer=nl.sbuf)
                c_u = nl.ndarray((1, n_flat), dtype=nl.uint32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=pos_u[0:1, :], src=pos_flat[0:1, :])
                nisa.tensor_scalar(dst=q_u[0:1, :], data=pos_u[0:1, :], op0=nl.right_shift, operand0=7)
                nisa.tensor_scalar(dst=c_u[0:1, :], data=pos_u[0:1, :], op0=nl.bitwise_and, operand0=PMAX - 1)
                nisa.tensor_copy(dst=key_f[0:1, :], src=q_u[0:1, :])
                nisa.tensor_copy(dst=c_f[0:1, :], src=c_u[0:1, :])
            if g > 1:
                # key = g'*n_q + q: fold the row-within-group block into the key so ONE compare
                # builds the block-diagonal one-hot.
                for gi in range(g):
                    nisa.tensor_scalar(
                        dst=key_f[0:1, nl.ds(gi * k, k)],
                        data=key_f[0:1, nl.ds(gi * k, k)],
                        op0=nl.add,
                        operand0=float(gi * n_q),
                    )

            # bf16 twins of the finished key/c vectors, for the broadcast matmuls: keys and
            # c are integers below 128/c_w, exact in bf16, so this is a pure rate change.
            nisa.tensor_copy(dst=key_bf[0:1, :], src=key_f[0:1, :])
            nisa.tensor_copy(dst=c_bf[0:1, :], src=c_f[0:1, :])

            # The group's candidate ids, reshaped one row -> n_q partition chunks of c_w. A DMA
            # cannot fan one SBUF partition out to n_q partitions, so the reshape happens where
            # each path can afford it: the bypass path's ids are already global float32 in HBM,
            # where an access pattern reshapes for free (contiguous c_w-runs); the general path
            # computed i_f in SBUF, so it moves one single-partition chunk at a time.
            d_all = nl.ndarray((PMAX, c_w), dtype=nl.float32, buffer=nl.sbuf)
            if n_q * c_w != red_w:
                nisa.memset(dst=d_all[0:k_g, :], value=0.0)
            q_full = red_w // c_w
            rem = red_w % c_w
            if fused[_FUSE_IDS_GLOBAL]:
                cand_flat = fused[_FUSE_CAND_I].reshape((fused[_FUSE_CAND_I].shape[0] * red_w,))
                for gi in range(g):
                    row_off = (sort_row_start + r0 + gi) * red_w
                    if q_full > 0:
                        nisa.dma_copy(
                            dst=d_all[nl.ds(gi * n_q, q_full), nl.ds(0, c_w)],
                            src=cand_flat.ap(pattern=[[c_w, q_full], [1, c_w]], offset=row_off),
                        )
                    if rem:
                        nisa.dma_copy(
                            dst=d_all[nl.ds(gi * n_q + q_full, 1), nl.ds(0, rem)],
                            src=cand_flat.ap(pattern=[[1, rem]], offset=row_off + q_full * c_w),
                        )
            else:
                for gi in range(g):
                    for q in range(q_full):
                        nisa.dma_copy(
                            dst=d_all[nl.ds(gi * n_q + q, 1), nl.ds(0, c_w)],
                            src=i_f[nl.ds(r0 + gi, 1), nl.ds(q * c_w, c_w)],
                        )
                    if rem:
                        nisa.dma_copy(
                            dst=d_all[nl.ds(gi * n_q + q_full, 1), nl.ds(0, rem)],
                            src=i_f[nl.ds(r0 + gi, 1), nl.ds(q_full * c_w, rem)],
                        )

            got_flat = nl.ndarray((1, n_flat), dtype=nl.uint32, buffer=nl.sbuf)
            got_tap = None
            if fused[_FUSE_TAP_GOT] is not None:
                got_tap = nl.ndarray((1, n_flat), dtype=nl.float32, buffer=nl.sbuf)
            for t0 in range(0, n_flat, _PSUM_MOVING):
                n_t = min(_PSUM_MOVING, n_flat - t0)
                tile = nl.ds(t0, n_t)
                # Broadcast the tile's keys down K_g partitions and compare against the ramp:
                # the block-diagonal one-hot, one column of which selects one (row, q) chunk.
                key_b = nl.ndarray((PMAX, _PSUM_MOVING), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(
                    dst=key_b[0:k_g, 0:n_t],
                    stationary=ones_row_bf[0:1, 0:k_g],
                    moving=key_bf[0:1, tile],
                    accumulate=False,
                )
                b_hot = nl.ndarray((PMAX, _PSUM_MOVING), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(
                    dst=b_hot[0:k_g, 0:n_t], data=key_b[0:k_g, 0:n_t], op0=nl.equal, operand0=iota_pf[0:k_g, 0:1]
                )
                # M[c, col] = i_f[row_col, q_col*c_w + c] for every c at once.
                m_ps = nl.ndarray((PMAX, _PSUM_MOVING), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(
                    dst=m_ps[0:c_w, 0:n_t], stationary=d_all[0:k_g, 0:c_w], moving=b_hot[0:k_g, 0:n_t], accumulate=False
                )
                # The c-selection: broadcast c down c_w partitions, one-hot it against the same
                # ramp, zero everything but row c_col, and sum the partitions with a ones matmul.
                c_b = nl.ndarray((PMAX, _PSUM_MOVING), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(
                    dst=c_b[0:c_w, 0:n_t], stationary=ones_row_bf[0:1, 0:c_w], moving=c_bf[0:1, tile], accumulate=False
                )
                m_sel = nl.ndarray((PMAX, _PSUM_MOVING), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(
                    dst=m_sel[0:c_w, 0:n_t], data=c_b[0:c_w, 0:n_t], op0=nl.equal, operand0=iota_pf[0:c_w, 0:1]
                )
                nisa.tensor_tensor(
                    dst=m_sel[0:c_w, 0:n_t], data1=m_ps[0:c_w, 0:n_t], data2=m_sel[0:c_w, 0:n_t], op=nl.multiply
                )
                sel = nl.ndarray((1, _PSUM_MOVING), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(
                    dst=sel[0:1, 0:n_t], stationary=ones_col[0:c_w, 0:1], moving=m_sel[0:c_w, 0:n_t], accumulate=False
                )
                nisa.tensor_copy(dst=got_flat[0:1, nl.ds(t0, n_t)], src=sel[0:1, 0:n_t])
                if got_tap is not None:
                    nisa.tensor_copy(dst=got_tap[0:1, nl.ds(t0, n_t)], src=sel[0:1, 0:n_t])

            # The destination rows are contiguous in HBM, so the whole group's ids leave in
            # ONE flat DMA rather than one per row.
            out_idx_flat = fused[_FUSE_OUT_IDX].reshape((fused[_FUSE_OUT_IDX].shape[0] * k,))
            nisa.dma_copy(
                dst=out_idx_flat.ap(pattern=[[1, n_flat]], offset=(sort_row_start + r0) * k),
                src=got_flat[0:1, :],
            )
            if got_tap is not None:
                tap_flat = fused[_FUSE_TAP_GOT].reshape((fused[_FUSE_TAP_GOT].shape[0] * k,))
                nisa.dma_copy(
                    dst=tap_flat.ap(pattern=[[1, n_flat]], offset=(sort_row_start + r0) * k),
                    src=got_tap[0:1, :],
                )

    return topk_values, topk_indices, asc_idx_hbm


@nki.jit
def chunked_topk(inp: nl.NkiTensor, config: ChunkedTopkConfig) -> Tuple[nl.NkiTensor, nl.NkiTensor]:
    """Top-k over ``inp[BxS, vocab]`` for a vocab wider than nisa.topk's limit.

    Args:
        inp: ``[BxS, vocab]`` bf16 in HBM.
        config: a ``ChunkedTopkConfig``; ``chunks == 1`` runs one inlined top-k over the whole vocab.

    Returns:
        ``(topk_values [BxS, k] bf16, topk_indices [BxS, k] uint32)`` in HBM, with global vocab ids, descending when
        ``config.sorted``. Under ``config.debug_taps`` the ``TAP_NAMES`` taps follow.

    In outline, with ``W = vocab // P``::

        s1_v, s1_i = topk(inp.reshape(BxS * P, W), k)        # a view: one chunk per row
        cand_v, cand_i = s1_v.reshape(BxS, P * k), s1_i + c * W  # the offset folds into stage 1's remap
        r_v, r_pos = topk(cand_v, k)                          # local reduce, no collective
        out_i = cand_i[r_pos]                                 # one-hot matmul on the PE, from SBUF
    """
    _validate_chunked_topk(config)

    bxs = config.BxS
    vocab = config.vocab_size
    k = config.k
    p = config.chunks

    # LNC=1 only, asserted before the p == 1 shortcut. num_programs=1 on the inner calls pins nothing (the inlined top-k
    # reads the live grid), so at LNC=2 stage 1 would shard its rows while the reduce does not, and both programs would
    # read each other's rows with no barrier and write the same outputs.
    _, n_prgs, _ = get_verified_program_sharding_info("chunked_topk", (0, 1), 2)
    kernel_assert(n_prgs == 1, f"chunked_topk is LNC=1 only, got n_prgs={n_prgs}")

    # A vocab that fits one call needs none of the chunking: it runs this kernel's inlined top-k (hardware sort
    # only), unfused, whose private buffers are staged into the kernel's shared outputs.
    if p == 1:
        # Every tap point is a link of the CHUNKING chain, none of which exists here, so asking
        # for taps on this path is a caller mistake rather than something to silently return
        # fewer outputs for.
        kernel_assert(not config.debug_taps, "debug_taps has nothing to observe when chunks == 1")
        kernel_assert(
            config.interleaved_peer_width == 0,
            "interleaved_peer_width is not implemented on the chunks == 1 delegate path",
        )
        p1_vals, p1_idx, _ = _inline_topk(
            inp,
            create_topk_config(
                inp_shape=(bxs, vocab), inp_dtype=config.inp_dtype, k=k, sorted=config.sorted, num_programs=1
            ),
            skip_nan_fold=config.skip_nan_fold,
        )
        p1_out_v = nl.ndarray((bxs, k), dtype=config.inp_dtype, buffer=nl.shared_hbm)
        p1_out_i = nl.ndarray((bxs, k), dtype=config.index_dtype, buffer=nl.shared_hbm)
        for t in range(div_ceil(bxs, PMAX)):
            r0 = t * PMAX
            rows = min(PMAX, bxs - r0)
            sv = nl.ndarray((PMAX, k), dtype=config.inp_dtype, buffer=nl.sbuf)
            si = nl.ndarray((PMAX, k), dtype=config.index_dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=sv[0:rows, :], src=p1_vals[r0 : r0 + rows, :])
            nisa.dma_copy(dst=si[0:rows, :], src=p1_idx[r0 : r0 + rows, :])
            nisa.dma_copy(dst=p1_out_v[nl.ds(r0, rows), :], src=sv[0:rows, :])
            nisa.dma_copy(dst=p1_out_i[nl.ds(r0, rows), :], src=si[0:rows, :])
        return p1_out_v, p1_out_i

    width = vocab // p
    s1_rows = bxs * p
    red_w = p * k
    # Stage 1's de-snake width, needed here only to shape the s1_late tap.
    s1_k_pad = div_ceil(k, PARTS_PER_GROUP) * PARTS_PER_GROUP

    # ---- Diagnostic taps (config.debug_taps) ----
    # Six extra outputs that make the index chain observable on hardware, in a normally overlapped run.
    tap_s1_pos = None
    tap_s1_gid = None
    tap_s1_late = None
    tap_cand_i = None
    tap_r_pos = None
    tap_gathered = None
    tap_s1_val = None
    if config.debug_taps:
        tap_s1_pos = nl.ndarray((s1_rows, k), dtype=nl.float32, buffer=nl.shared_hbm)
        # float32 on both paths so the reader never has to know which one produced it: the
        # general path's uint32 ids cast exactly on the way out (every id is < 2^24).
        tap_s1_gid = nl.ndarray((s1_rows, k), dtype=nl.float32, buffer=nl.shared_hbm)
        tap_s1_late = nl.ndarray((s1_rows, s1_k_pad), dtype=nl.float32, buffer=nl.shared_hbm)
        tap_cand_i = nl.ndarray((bxs, red_w), dtype=nl.float32, buffer=nl.shared_hbm)
        tap_r_pos = nl.ndarray((bxs, k), dtype=nl.float32, buffer=nl.shared_hbm)
        tap_gathered = nl.ndarray((bxs, k), dtype=nl.float32, buffer=nl.shared_hbm)
        # The VALUE side of stage 1, appended LAST so every earlier tap keeps its position.
        # It makes the intermediate candidate VALUES checkable on hardware -- per-chunk pairing
        # against the caller's own input row, and per-chunk top-k multiset equality -- closing
        # the one link the index chain does not cover.
        tap_s1_val = nl.ndarray((s1_rows, k), dtype=nl.float32, buffer=nl.shared_hbm)

    # ---- Which stage-1 shape can skip its phase 2 ----
    # With k % 16 == 0 the de-snake buffers are the candidate arrays (_stage1_desnake_only), provided every stage-1 tile
    # is a whole 128-partition tile so the value buffer stays bf16 (the reduce needs bf16). Every auto-picked P meets
    # both at the served shapes; the general path below stays covered by the tests.
    red_rows = max(GROUPS_PER_TILE, div_ceil(bxs, GROUPS_PER_TILE) * GROUPS_PER_TILE)
    bypass_s1_phase2 = (
        (k % PARTS_PER_GROUP == 0) and (s1_rows % GROUPS_PER_TILE == 0) and (width % PARTS_PER_GROUP == 0)
    )

    # ---- Stage 1: one top-k per vocab chunk ----
    # [BxS, vocab] viewed as [BxS*P, vocab/P] puts chunk c of token j at row j*P + c, so the blocked load is the flat
    # vocab sweep and the instruction's 16-partition groups are the chunks. Never sorted: the reduce re-sorts, so stage
    # 1's order cannot reach the output.
    s1_cfg = create_topk_config(
        inp_shape=(s1_rows, width),
        inp_dtype=config.inp_dtype,
        k=k,
        sorted=False,
        num_programs=1,
    )
    kernel_assert(
        config.interleaved_peer_width == 0 or bypass_s1_phase2,
        "interleaved_peer_width is only implemented on the desnake-only stage-1 path",
    )
    if bypass_s1_phase2:
        cand_v, cand_i = _stage1_desnake_only(
            inp.reshape((s1_rows, width)),
            s1_cfg,
            chunks=p,
            chunk_width=width,
            pad_rows=red_rows * p,
            tap_pos=tap_s1_pos,
            skip_nan_fold=config.skip_nan_fold,
            peer_width=config.interleaved_peer_width,
        )
        # The de-snake buffers, viewed as the reduce wants them. Both views are free, and they
        # agree slot for slot because both were written by the same de-snake pattern.
        red_in = cand_v.reshape((red_rows, red_w))
        cand_i_view = cand_i.reshape((red_rows, red_w))
        # On this path stage 1 publishes ONE index buffer, already global, so it is what both
        # the "what stage 1 said" tap and the second-read tap look at -- which makes the s1_late
        # comparison a three-read agreement test on the de-snake store rather than two.
        s1_published = cand_i
        s1_desnaked = cand_i
        # First s1_rows rows of the [pad_rows, k] buffer are the real candidates; the rest pad.
        s1_vals_pub = cand_v
    else:
        s1_vals, s1_idx, s1_desnaked = _inline_topk(
            inp.reshape((s1_rows, width)), s1_cfg, tap_pos=tap_s1_pos, skip_nan_fold=config.skip_nan_fold
        )
        s1_published = s1_idx
        s1_vals_pub = s1_vals
        # ---- Reduce input: another free reshape, row-padded to a whole tile ---------------
        # [BxS*P, k] viewed as [BxS, P*k] places chunk c at columns [c*k, (c+1)*k) of row j.
        cand_v = s1_vals.reshape((bxs, red_w))
        cand_i_view = s1_idx.reshape((bxs, red_w))
        # Pad the row count up to a multiple of GROUPS_PER_TILE so the reduce stays on its
        # fast_dma_safe path (bf16 rather than fp32 de-snake buffers). This is a PERFORMANCE
        # pad, not a correctness one: a partial tile is handled correctly. Pad rows carry the
        # most-negative FINITE bf16, so they can never win a slot within their own row, and
        # their row's results are simply never read.
        if red_rows > bxs:
            red_in = nl.ndarray((red_rows, red_w), dtype=config.inp_dtype, buffer=nl.private_hbm)
            n_pad = red_rows - bxs
            pad_tile = nl.ndarray((PMAX, red_w), dtype=config.inp_dtype, buffer=nl.sbuf)
            # Only the n_pad rows that are actually stored need the sentinel. Memsetting the
            # whole [PMAX, red_w] allocation wrote 128 partitions x 2,048 bf16 to fill 7 rows.
            nisa.memset(dst=pad_tile[0:n_pad, :], value=BFLOAT16_MIN)
            nisa.dma_copy(dst=red_in[nl.ds(bxs, n_pad), :], src=pad_tile[0:n_pad, :])
            for t in range(div_ceil(bxs, PMAX)):
                r0 = t * PMAX
                rows = min(PMAX, bxs - r0)
                stage = nl.ndarray((PMAX, red_w), dtype=config.inp_dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=stage[0:rows, :], src=cand_v[r0 : r0 + rows, :])
                nisa.dma_copy(dst=red_in[r0 : r0 + rows, :], src=stage[0:rows, :])
        else:
            red_in = cand_v

    # ---- Reduce: top-k over the P*k candidates, entirely local ----
    # Its tail is fused (see _inline_topk): positions become global ids from the same SBUF tile and the values go
    # straight to out_vals.
    out_vals = nl.ndarray((bxs, k), dtype=config.inp_dtype, buffer=scratch_hbm())
    out_idx = nl.ndarray((bxs, k), dtype=config.index_dtype, buffer=scratch_hbm())
    _inline_topk(
        red_in,
        create_topk_config(
            inp_shape=(red_rows, red_w),
            inp_dtype=config.inp_dtype,
            k=k,
            sorted=config.sorted,
            num_programs=1,
        ),
        # The reduce input is stage 1's own (already NaN-folded) candidates plus finite
        # sentinel pad rows: the fold can never fire here. See the fold site for the argument.
        skip_nan_fold=True,
        fused=(
            cand_i_view,
            p,
            width,
            out_vals,
            out_idx,
            bxs,
            tap_cand_i,
            tap_r_pos,
            tap_gathered,
            bypass_s1_phase2,
        ),
    )

    if config.debug_taps:
        # A second, independent read of stage 1's de-snake buffer plus a copy of its remapped ids. Two reads of one
        # address can only disagree while a write is in flight: s1_pos wrong and s1_late right means the de-snake store
        # landed after phase 2 read it; both wrong means the top-k itself reported those positions.
        for t in range(div_ceil(s1_rows, PMAX)):
            r0 = t * PMAX
            rows = min(PMAX, s1_rows - r0)
            lt = nl.ndarray((PMAX, s1_k_pad), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=lt[0:rows, :], src=s1_desnaked[r0 : r0 + rows, :])
            nisa.dma_copy(dst=tap_s1_late[r0 : r0 + rows, :], src=lt[0:rows, :])
            gt = nl.ndarray((PMAX, k), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=gt[0:rows, :], src=s1_published[r0 : r0 + rows, :])
            nisa.dma_copy(dst=tap_s1_gid[r0 : r0 + rows, :], src=gt[0:rows, :])
            vt = nl.ndarray((PMAX, k), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=vt[0:rows, :], src=s1_vals_pub[r0 : r0 + rows, :])
            nisa.dma_copy(dst=tap_s1_val[r0 : r0 + rows, :], src=vt[0:rows, :])
        return out_vals, out_idx, tap_s1_pos, tap_s1_gid, tap_s1_late, tap_cand_i, tap_r_pos, tap_gathered, tap_s1_val

    return out_vals, out_idx
