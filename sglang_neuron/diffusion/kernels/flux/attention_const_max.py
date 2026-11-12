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
Constant-Max Attention Kernel

Computes the UNNORMALIZED partial attention

    [Σ P·V | Σ P·1]     where    P = exp(Q @ K.T * softmax_scale - c)

and folds it into a caller-owned SBUF accumulator. The two matmuls are referred to throughout as:
  - MM1: Q @ K.T (scores)
  - MM2: P @ [V | 1]  (weighted value accumulation, plus the sum_exp column)

`c` is the per-ROW softmax bound c_i = softmax_scale·‖q_i‖·max_j‖k_j‖, supplied by the caller in
``c_row_hbm`` in natural token order. Taking the max as given rather than deriving it online
eliminates:
  1. The DMA transpose between MM1 and MM2, which serializes with collective
     communication (CC) and blocks CC overlap.
  2. The online softmax max-reduction pass.

With no online max the scores stay K-stationary ([Sk-partition, Sq-free]), which is what lets a
ring driver's KV-rotation collective overlap this compute. The accumulator is SBUF-resident across
ring steps and the caller normalizes once at the end, so there is no HBM round-trip per step and no
separate cross-step reduction pass. Where the accumulators do not fit SBUF next to a head's
resident K/V, _attention_const_max_hbm_acc keeps them in HBM instead.
"""

from typing import Optional

import nki.isa as nisa
import nki.language as nl

from nkilib.core.utils.kernel_assert import kernel_assert
from .. import neurotile as nt

# Tiling constants
_P = nl.tile_size.pmax  # 128: partition dimension
_SK_BLOCK = 2048  # KV streaming block size (16 tiles of 128)
_O_TILE_SIZE = nl.tile_size.gemm_stationary_fmax  # 128: output tile size along Sq (MM2 stationary)
_KV_TILES_PER_BLOCK = _SK_BLOCK // _P  # 16: number of K/V tiles per KV block

# PSUM bank allocation:
#   _MM1_BUFFER_BANKS (2) banks hold the MM1 score tiles, so the next KV tile's MM1 can start while
#   the current scores are still being read. The rest, at most _MAX_O_AUG_BANKS_CAP (8), hold the
#   O_aug accumulators: _max_o_aug_banks() = min(psum_num_banks - 2, 8).
#
# O_aug is the "augmented" output tile: 128 query rows by d+1 columns, [Σ P·V | Σ P·1]. MM2
# multiplies the probabilities P by [V | 1], so columns 0..d-1 accumulate the weighted output and
# column d accumulates the softmax denominator (sum_exp), which the final normalization divides by.
#
# Q and O blocks share the same Sq-dimension size. O tiles are staged in PSUM and
# accumulate across all KV blocks. The number of available O_aug banks is derived at
# trace time from nl.tile_size.psum_num_banks. Each O_aug tile takes its own bank and covers
# gemm_stationary_fmax (128) query rows, so the maximum O block is _max_o_aug_banks() × 128 Sq elements.
#
# On trn2 (gen3), MM1 produces f32 PSUM and the moving operand (Q) is limited to
# gemm_moving_fmax = 512, so we choose max Q/O block = 512. On trn3 (gen4), MM1
# produces bf16 PSUM with no such limit, so the full O_aug capacity is used.
#
# Q stays resident in SBUF while we stream all KV blocks, prefetching the next block
# behind the current block's compute.

# Trace-size budget for the statically unrolled partial attention, in emitted instructions summed
# over every head and ring step of one kernel. Every loop here is unrolled at trace time, so the
# emitted BIR grows as heads x ring steps x Q-blocks x KV tiles, and the serialized BIR body's size
# is a uint32 ("colz: body blob exceeds uint32 offset range" past 4 GiB). Measured on trn2
# (nki 0.6.0): ~950 bytes of body per instruction, so 1M instructions is ~22% of the limit. At 10
# heads, CP8 and 21105 tokens/rank the fully unrolled kernel emits ~4.4M — 98%; at 9450 tokens/rank
# it emits ~0.9M and stays unrolled.
_MAX_STATIC_ATTN_INSTRUCTIONS = 1_000_000
# Instructions emitted per (Q-block, KV tile) pair, not counting the MM2s (one per 128-row output
# tile): V copy, MM1, subtract and exp. choose_q_group multiplies it out to estimate the fully
# unrolled kernel's instruction count, which decides between the unrolled and rolled paths and sets
# the rolled group size. It counts NKI calls, not hardware instructions; the ~950 bytes per
# instruction above was calibrated with it.
_INSTRUCTIONS_PER_KV_TILE = 4
# Query rows per head that the rolled path emits statically ahead of each ring step's Q-block
# loops (the "head start"). Entering an nl.fori_loop waits for every DMA and collective still in
# flight, including ones the loop never touches such as the ring step's K/V collective-permute and
# the next step's private K/V copy (loop back-edges are cheap). Without a head start the Tensor
# engine sits in one semaphore wait from the first loop entry until the permute pair completes.
_HEAD_START_ROWS = 2560
# The first ring step overlaps two steps' transfers (the driver hoists the first kv_prefetch_depth
# permutes ahead of it), so it gets a larger head start. One step of num_steps, so the extra code is
# small next to the budget.
_FIRST_STEP_HEAD_START_ROWS = 2 * _HEAD_START_ROWS
# Smallest Q-block group per rolled loop iteration that choose_q_group trades head start for.
_MIN_ROLLED_GROUP = 3
# MM1→exp tiles ahead of the first MM2 in the rolled loop body (8 on the unrolled path; see
# _process_kv_tiles_pipelined).
_ROLLED_PROLOGUE_DEPTH = 4
# Rotating [V|1] block slots for the direct-V path (streamed K/V in _attention_const_max_hbm_acc),
# plus one dedicated slot for the last block so its pad mask is never overwritten. MM2 reads a
# block's slot prologue_depth tiles after its MM1, so three rotating slots keep a block's load from
# waiting on the previous use's MM2s.
_V_DIRECT_ROTATING_SLOTS = 3
# Per-partition SBUF one head's resident K + [V|1] pair may take (see _resident_kv_fits). The rest
# of the Q-block work (Q block, bias, probabilities, shifted scores, accumulator staging) needs
# ~40 KB more of the partition.
_RESIDENT_KV_SBUF_BUDGET_BYTES = 120 * 1024
# Pieces each resident K/V load is split into, so it spreads across DMA queues.
_RESIDENT_KV_PIECES = 8

# PSUM banks reserved for the MM1 score tiles, and the cap on O_aug banks (see the PSUM notes above).
_MM1_BUFFER_BANKS = 2
_MAX_O_AUG_BANKS_CAP = 8


def _max_o_aug_banks():
    """PSUM banks available for O_aug accumulators: 6 on trn2 and trn3 (8 banks - 2 for MM1)."""
    return min(nl.tile_size.psum_num_banks - _MM1_BUFFER_BANKS, _MAX_O_AUG_BANKS_CAP)


def _max_sq_block_size():
    """Q/O block size along Sq: 512 on trn2 (gen3, f32 PSUM caps the MM1 moving free size),
    _max_o_aug_banks() x 128 = 768 on trn3 (gen4, bf16 PSUM). See the PSUM bank notes above."""
    has_bf16_psum = nisa.get_nc_version() >= nisa.nc_version.gen4
    return _max_o_aug_banks() * _O_TILE_SIZE if has_bf16_psum else nl.tile_size.gemm_moving_fmax


def rolled_blocks_for(num_full, q_group, head_start):
    """Full Q-blocks the rolled loop covers: whole groups of ``q_group`` from block 0, leaving at
    least ``head_start`` full blocks static, and 0 (no loop) when that is under 2 iterations."""
    num_iters = max(num_full - head_start, 0) // q_group
    return num_iters * q_group if num_iters >= 2 else 0


def choose_q_group(num_heads, num_steps, sq, sk):
    """(Q-blocks per rolled loop iteration, head-start blocks, first-step head-start blocks), or None
    to keep the unrolled path.

    Estimate. For every head and ring step, one Q-block of max_sq rows emits, per KV tile,
    _INSTRUCTIONS_PER_KV_TILE instructions plus one MM2 per 128-row output tile. So one Q-block
    costs

        per_q_block = num_heads x num_steps x ceil(sk / 128) x (4 + max_sq / 128)

    and the fully unrolled kernel costs ceil(sq / max_sq) x per_q_block. The budget allows
    budget_blocks = _MAX_STATIC_ATTN_INSTRUCTIONS // per_q_block statically emitted Q-blocks.

    Selection. If all Q-blocks fit, return None. Otherwise return
    ``(group, head_start, first_step_head_start)``. The head-start blocks are emitted statically
    before any rolled loops so ring transfers can overlap compute instead of being drained at the
    first loop entry. Prefer groups of at least _MIN_ROLLED_GROUP to amortize the loop overhead.

    The budget check counts every ring step at head_start. Ring step 0 is emitted with
    first_step_head_start instead, which also moves the loop remainder out of the loop, so it emits
    (static blocks at first_step_head_start - static blocks at head_start) more Q-blocks for that
    one step: that many x per_q_block / num_steps instructions on top of the checked estimate. The
    total therefore runs slightly past _MAX_STATIC_ATTN_INSTRUCTIONS, which is a soft target (~22%
    of the uint32 BIR limit), not the limit itself.

    Values at 10 heads and CP8 (max_sq is 512 on trn2 and 768 on trn3, see _max_sq_block_size):
        9450 tokens/rank: ~0.77M (trn3) and ~0.90M (trn2) unrolled, so None.
        21105 tokens/rank, trn3: 28 blocks x 132K = ~3.7M, budget 7 blocks,
            (group, head_start, first) = (3, 3, 7): 3 + 3 + 1 = 7 blocks emitted per step
            (~0.92M), plus 6 more on step 0 (+99K) = ~1.02M total, 8 iterations.
        21105/21420 tokens/rank, trn2: 42 blocks x ~106K = ~4.4M, budget 9
            blocks, (3, 5, 10): 3 + 5 + 1 = 9 blocks emitted per step (~0.96M), plus 6 more on
            step 0 (+80K) = ~1.04M total (~23% of the BIR limit), 12 iterations.
    """
    max_sq = _max_sq_block_size()
    num_q_blocks = (sq + max_sq - 1) // max_sq
    kv_tiles = (sk + _P - 1) // _P
    per_q_block = num_heads * num_steps * kv_tiles * (_INSTRUCTIONS_PER_KV_TILE + max_sq // _O_TILE_SIZE)
    budget_blocks = max(1, _MAX_STATIC_ATTN_INSTRUCTIONS // per_q_block)
    if num_q_blocks <= budget_blocks:
        return None
    num_full = sq // max_sq
    has_tail = 1 if sq % max_sq else 0
    full_head_start = min((_HEAD_START_ROWS + max_sq - 1) // max_sq, num_full)
    first_head_start = min((_FIRST_STEP_HEAD_START_ROWS + max_sq - 1) // max_sq, num_full)
    fallback = None
    for head_start in range(full_head_start, -1, -1):
        for group in range(min(budget_blocks, num_full), 0, -1):
            rolled = rolled_blocks_for(num_full, group, head_start)
            if rolled > 0 and group + (num_full - rolled) + has_tail <= budget_blocks:
                if group >= _MIN_ROLLED_GROUP:
                    return group, head_start, max(head_start, first_head_start)
                if fallback is None:
                    fallback = (group, head_start, max(head_start, first_head_start))
                break
    return fallback if fallback is not None else (1, 0, 0)


def _attention_const_max_partial(
    q_hbm,
    k_hbm,
    v_hbm,
    o_aug_acc_sb,
    c_row_hbm,
    softmax_scale=None,
    aug_col_base=0,
    is_accumulate_step=False,
    kv_pad_rows=0,
    prologue_depth=8,
    sbuf_allocator=None,
    sbuf_unit_bases=None,
    sbuf_unit_bytes=None,
    num_shifted_buffers=2,
    kv_roots=None,
):
    """Partial (unnormalized) attention folding into a resident SBUF accumulator.

    Runs the K-stationary MM1 → exp(scores − c) → MM2([V|1]) pipeline and folds each output
    tile's [Σ P·V | Σ P·1] straight from PSUM into the caller's resident SBUF accumulator —
    is_accumulate_step selects init (ring step 0) vs += (later steps), so there is no HBM
    round-trip and no separate cross-step reduction pass. Does NO LNC sharding of its own: the
    caller passes the head slice this shard owns.

    Args:
        q_hbm: [N, d, Sq] bf16 @ HBM.
        k_hbm: [N, d, Sk] bf16 @ HBM (arbitrary Sk; a trailing sub-128 group is handled).
        v_hbm: [N, Sk, d] bf16 @ HBM.
        o_aug_acc_sb: resident SBUF accumulator this step folds into. Layout mirrors the PSUM
            source: each (head, group) tile occupies (d+1) contiguous columns — d output columns
            followed by 1 sum column.
        c_row_hbm: [N, Sq] bf16 @ HBM holding the per-ROW bound c_i = softmax_scale·‖q_i‖·max_j‖k_j‖
            in natural token order. Each Q-block loads its contiguous [1, sq_block] slice onto the
            free axis and broadcasts it across the 128 partitions.
        softmax_scale: scaling factor for Q. Defaults to 1/sqrt(d).
        aug_col_base: base column of this call's first (head, group) tile (stride d+1 between tiles).
        is_accumulate_step: True folds (+=); False initializes (first ring step).
        kv_pad_rows: Number of trailing Sk tokens that are caller-supplied ZERO padding (used when
            the caller rounded Sk up to a 128-multiple to keep the KV loop uniform). Those rows are
            masked out of the sums exactly; 0 means Sk is entirely real tokens.
        prologue_depth: KV tiles whose MM1 and exp run before the first MM2 of each Q-block
            (capped at the tile count); see _process_kv_tiles_pipelined.
        sbuf_allocator: Optional ModularAllocator. With sbuf_unit_bases, every SBUF tile buffer of
            a Q-block unit is allocated from it at a fixed address instead of compiler-placed.
        sbuf_unit_bases: SBUF byte addresses of the rotating per-unit regions (Q-block unit u
            starts at sbuf_unit_bases[u % len]); used only with sbuf_allocator.
        sbuf_unit_bytes: Size of each region at sbuf_unit_bases; a unit whose tiles end past its
            region fails at trace time. None skips the check.
        num_shifted_buffers: slots in the fp16 `shifted` pool (see _process_kv_tiles_pipelined).
        kv_roots: Optional ``(k_root, v_root, first_head)``: the root [H, d, Sk] K and [H, Sk, d] V
            tensors k_hbm / v_hbm slice, and the root index of their head 0. When given (Sk a
            multiple of 128), each head's K and [V|1] are loaded into SBUF once and every Q-block
            reads them there instead of streaming them from HBM (see _alloc_resident_kv), so none
            of the KV split's sources below is used.

    KV split. Sk is cut into up to three sources that the pipeline consumes in order:

    * blocks (``k_stream``/``v_stream``): the first ``sk_block_rows`` rows, streamed in blocks of
      ``_SK_BLOCK`` = 2048 rows (16 tiles of 128) with double-buffered prefetch;
    * leftover tiles (``k_leftover``/``v_leftover``): the whole 128-row tiles after the last full
      block, loaded once as one multi-tile grid (``sk_leftover_rows`` rows);
    * tail (``k_tail``/``v_tail``): the final ``Sk % 128`` rows as one partial tile.

    A neurotile block stream sizes its rotating buffers from its first block, so it can carry
    either several full 16-tile blocks or a single block of any size up to 16 tiles, but not
    full blocks followed by a shorter one (the DMA into the 16-tile buffer then fails to trace:
    "dma_copy requires src and dst to have the same number of elements"). Hence the rule
    ``sk_block_rows = sk_whole`` when all whole tiles fit in one block (``sk_whole <= _SK_BLOCK``)
    and ``sk_block_rows = (sk_whole // _SK_BLOCK) * _SK_BLOCK`` otherwise; the rest of the whole
    tiles become the leftover source. Examples, with ``sk_whole = (Sk // 128) * 128``:

    * Sk = 1000: 7 whole tiles fit one block, so the stream carries one 7-tile block, there are
      no leftover tiles, and the tail has 104 rows (7 + 1 = 8 tiles).
    * Sk = 2176 (FLUX.1 2048 px at CP8, 17 tiles): one 16-tile block, one leftover tile, no tail.
    * Sk = 4700: 36 whole tiles do not fit one block, so the stream carries two 16-tile blocks
      (4096 rows), the leftover source holds 4 tiles (512 rows), and the tail has 92 rows
      (16 + 16 + 4 + 1 = 37 tiles).

    An Sk of at most 16 tiles, or of a multiple of 16 tiles, has no leftover source:
    ``sk_leftover_rows`` is 0 for it.

    Pseudocode:
        for each head:
            for each Q-block:
                Q = load(q_hbm[head, Q-block]) * softmax_scale
                c = load(c_row_hbm[head, Q-block])       # per-row bound, broadcast over partitions
                for each KV-block:
                    for each kv tile:
                        S = Q @ K_tile.T                 # MM1: scores
                        P = exp(S - c)                   # evict + subtract + exp
                        O_aug += P @ [V_tile | 1]        # MM2: output + sum_exp
                fold O_aug into o_aug_acc_sb
    """
    Nq, d, Sq = q_hbm.shape
    Sk = k_hbm.shape[2]
    softmax_scale = d ** (-0.5) if softmax_scale == None else softmax_scale
    # Groups (128-row Sq output tiles) per head — the accumulator stride between heads.
    groups_per_head = (Sq + _O_TILE_SIZE - 1) // _O_TILE_SIZE

    kernel_assert(nl.tile_size.psum_num_banks >= 6, f"need at least 6 PSUM banks, have {nl.tile_size.psum_num_banks}")
    max_o_aug_banks = _max_o_aug_banks()
    max_sq_block_size = _max_sq_block_size()

    sk_whole = (Sk // _P) * _P
    r_tail = Sk % _P
    total_kv_tiles = (Sk + _P - 1) // _P
    sk_block_rows = sk_whole if sk_whole <= _SK_BLOCK else (sk_whole // _SK_BLOCK) * _SK_BLOCK
    sk_leftover_rows = sk_whole - sk_block_rows

    unit_idx = 0
    kv_resident = None
    if kv_roots is not None:
        kernel_assert(Sk % _P == 0, f"resident K/V expects Sk padded to a multiple of {_P}, got {Sk}")
        k_root, v_root, first_head = kv_roots
        kv_resident = _alloc_resident_kv(Sk, d, kv_pad_rows, k_hbm.dtype)

    for head_idx in range(Nq):
        # Q is [d, Sq] per head, tiled along Sq at max_sq_block_size; each Q block loaded once and
        # kept resident while all KV blocks stream through it.
        q_block_grid = nt.tiles(q_hbm[head_idx], tile_size=(_P, max_sq_block_size))
        if kv_resident is not None:
            _load_resident_kv(k_root, v_root, first_head + head_idx, kv_resident, Sk, d)

        for q_block_idx in range(q_block_grid.shape[1]):
            unit_base = None
            if sbuf_allocator is not None:
                unit_base = sbuf_unit_bases[unit_idx % len(sbuf_unit_bases)]
                sbuf_allocator.set_current_address(unit_base)
            unit_idx += 1
            # Load Q block into SBUF — stays resident for all KV iterations. Q is pre-scaled here
            # rather than folding softmax_scale into the exp, because the bound is
            # c = softmax_scale·‖q‖·max‖k‖: the scores it is subtracted from must already carry
            # the scale.
            q_loaded = q_block_grid[0, q_block_idx].load(sbuf_allocator=sbuf_allocator)
            q_block = q_loaded.data
            nisa.tensor_scalar(dst=q_block, data=q_block, op0=nl.multiply, operand0=softmax_scale)

            sq_block_size = q_loaded.element_shape[1]

            # c_row_hbm holds c_i in natural token order, so this Q-block's contiguous
            # [1, sq_block] slice IS c_i along the score tile's free (Sq) axis.
            # Replicate it down all 128 partitions with ONE partition-stride-0 DMA.
            #
            # Every partition must hold the IDENTICAL c for a query column: softmax is invariant to c only
            # then, since a column whose partitions disagree scales its Sk terms inconsistently and the
            # factor stops cancelling in o = ΣPV/ΣP. 128 reads of the same immutable HBM row give exactly
            # that, and they cost no Vector time. Deriving all 128 partitions from one loaded row with a
            # stream shuffle instead is the same guarantee, but it puts serialized Vector work on
            # every Q-block boundary AHEAD of the Q pre-scale in the in-order Vector queue — and the
            # pre-scale gates this block's first MM1, so the Tensor engine waits out both.
            sq0 = q_block_idx * max_sq_block_size
            if sbuf_allocator is None:
                per_row_bias = nl.ndarray(shape=(_P, sq_block_size), dtype=c_row_hbm.dtype, buffer=nl.sbuf)
            else:
                per_row_bias = sbuf_allocator.alloc_sbuf_tensor(
                    shape=(_P, sq_block_size), dtype=c_row_hbm.dtype, align_to=64
                )
            nisa.dma_copy(
                dst=per_row_bias,
                src=c_row_hbm[head_idx].ap(pattern=[[0, _P], [1, sq_block_size]], offset=sq0),
            )

            # Allocate output accumulator in PSUM — one tile per Sq output slice, each on its own
            # bank. Accumulate across all KV blocks, then fold into the SBUF accumulator.
            num_output_tiles = (sq_block_size + _O_TILE_SIZE - 1) // _O_TILE_SIZE
            kernel_assert(
                num_output_tiles <= max_o_aug_banks,
                "Q block too large: O_aug tiles exceed available PSUM banks.",
            )
            # Output accumulator width is d+1: columns 0..d-1 hold the weighted output (P @ V), and
            # column d accumulates sum_exp (P @ 1) for the caller's normalize.
            o_aug_psums = nt.psum_pool(
                tile_size=(_O_TILE_SIZE, d + 1),
                element_shape=(sq_block_size, d + 1),
            )
            # Continuous software pipeline across all KV tiles keeps TensorE fed (no per-block
            # boundary bubbles). nt.blocks handles a non-_SK_BLOCK-multiple prefix: the last block
            # has fewer tiles.
            k_stream = None
            v_stream = None
            # Resident K/V needs no stream, leftover or tail source (its Sk is a 128-multiple).
            if kv_resident is None and sk_block_rows > 0:
                k_view = nt.blocks(
                    k_hbm[head_idx, :, :sk_block_rows],
                    tile_size=(_P, d),
                    block_size=(1, _KV_TILES_PER_BLOCK),
                )
                v_view = nt.blocks(
                    v_hbm[head_idx, :sk_block_rows, :],
                    tile_size=(_P, d),
                    block_size=(_KV_TILES_PER_BLOCK, 1),
                )
                k_stream = k_view[0, :].stream(buffer_count=2, sbuf_allocator=sbuf_allocator)
                v_stream = v_view[:, 0].stream(buffer_count=2, sbuf_allocator=sbuf_allocator)

            k_leftover = None
            v_leftover = None
            if kv_resident is None and sk_leftover_rows > 0:
                k_leftover = nt.tiles(k_hbm[head_idx, :, sk_block_rows:sk_whole], tile_size=(_P, d))[
                    0, :
                ].load(sbuf_allocator=sbuf_allocator)
                v_leftover = nt.tiles(v_hbm[head_idx, sk_block_rows:sk_whole, :], tile_size=(_P, d))[
                    :, 0
                ].load(sbuf_allocator=sbuf_allocator)

            # The sub-128 tail as its own single partial tile — a 1-tile grid passes the neurotile
            # load, and it indexes exactly like a block-stream tile grid.
            k_tail = None
            v_tail = None
            if kv_resident is None and r_tail > 0:
                k_tail = nt.tiles(k_hbm[head_idx, :, sk_whole:Sk], tile_size=(_P, d))[0, :].load(
                    sbuf_allocator=sbuf_allocator
                )
                v_tail = nt.tiles(v_hbm[head_idx, sk_whole:Sk, :], tile_size=(_P, d))[:, 0].load(
                    sbuf_allocator=sbuf_allocator
                )

            _process_kv_tiles_pipelined(
                k_stream=k_stream,
                v_stream=v_stream,
                q_block=q_block,
                o_aug_psums=o_aug_psums,
                per_row_bias=per_row_bias,
                k_leftover=k_leftover,
                v_leftover=v_leftover,
                leftover_tiles=sk_leftover_rows // _P if kv_resident is None else 0,
                k_tail=k_tail,
                v_tail=v_tail,
                total_kv_tiles=total_kv_tiles,
                kv_pad_rows=kv_pad_rows,
                prologue_depth=prologue_depth,
                sbuf_allocator=sbuf_allocator,
                num_shifted_buffers=num_shifted_buffers,
                kv_resident=kv_resident,
            )
            if unit_base is not None and sbuf_unit_bytes is not None:
                unit_end = sbuf_allocator.get_current_address()
                kernel_assert(
                    unit_end <= unit_base + sbuf_unit_bytes,
                    f"Q-block unit tiles end at SBUF byte {unit_end}, past their "
                    f"{sbuf_unit_bytes}-byte region at {unit_base}",
                )

            # Fold each output tile's accumulated o_aug = [Σ P·V | Σ P·1] straight from PSUM into the
            # resident SBUF accumulator (init on the first ring step, += after).
            o_block_start = q_block_idx * (max_sq_block_size // _O_TILE_SIZE)
            for output_tile_idx in range(num_output_tiles):
                g = head_idx * groups_per_head + (o_block_start + output_tile_idx)
                _accumulate_partial(
                    o_aug_psums[output_tile_idx].data,
                    o_aug_acc_sb,
                    aug_col_base + g * (d + 1),
                    is_accumulate_step,
                )


def _process_kv_tiles_pipelined(
    k_stream: Optional[nt.BlockStream],
    v_stream: Optional[nt.BlockStream],
    q_block: nl.NkiTensor,
    o_aug_psums: nt.NDSlice,
    per_row_bias: nl.NkiTensor,
    total_kv_tiles: int,
    k_leftover: Optional[nt.NDSlice] = None,
    v_leftover: Optional[nt.NDSlice] = None,
    leftover_tiles: int = 0,
    k_tail: Optional[nt.NDSlice] = None,
    v_tail: Optional[nt.NDSlice] = None,
    kv_pad_rows: int = 0,
    inside_fori_loop: bool = False,
    prologue_depth: int = 8,
    sbuf_allocator=None,
    v_direct=None,
    num_shifted_buffers: int = 2,
    kv_resident=None,
) -> None:
    """Pipelined MM1→exp→MM2 across all KV tiles with a prologue/epilogue split.

    The bound c_i varies along the free (Sq) axis, so it cannot ride the exp's per-partition
    max_value. It is subtracted explicitly (Vector) into an fp16 tile, which the exp then reads.
    The exp is issued via nisa.activation (Scalar engine) so the Vector engine does only the
    subtract and the two pipeline across tiles; the [V|1] copy goes to whichever of the two is
    lighter (see copy_engine below), and all of them pipeline against the Tensor engine's MMs.

    Pipeline (P = prologue_depth):
        Prologue  (kv_tile_idx 0..P-1):        v_copy, MM1, exp
        Steady-state (kv_tile_idx P..total-1):  v_copy, MM1, exp, MM2(i-P)
        Epilogue  (drain last P):               MM2

    A trailing partial tile (the peeled sub-128 KV tail) has r < 128 valid Sk rows. Every
    Sk-indexed extent — the V copy, MM1 output partitions, exp, and the MM2 contraction on both
    operands — is clamped to that tile's real row count r (== v_block[i, 0].data.shape[0]), so
    padded positions contribute nothing to Σ P·V or Σ P·1 and no explicit tail mask is needed.

    Args:
        k_stream/v_stream: block streams over the first sk_block_rows Sk rows: full 16-tile
            blocks, or one block of at most 16 tiles when all whole 128-row tiles fit in one block
            (see the KV split in _attention_const_max_partial). None when Sk < 128 or with
            kv_resident.
        k_leftover/v_leftover: Optional pre-loaded multi-tile grids holding the whole 128-row
            tiles after the last streamed block and before the sub-128 tail
            (K (128, leftover_tiles*128), V (leftover_tiles*128, d)). None when the streamed
            blocks reach the tail.
        leftover_tiles: Tiles in k_leftover/v_leftover; 0 when they are None.
        per_row_bias: [_P, sq_block_size] per-row bound c_i replicated down all 128 partitions
            (uniform per free column = per query row).
        k_tail/v_tail: Optional pre-loaded single-tile grids holding the sub-128 Sk tail
            (K tile (128, r_tail), V tile (r_tail, d)). Consumed as the pipeline's last tile.
        kv_pad_rows: Trailing partitions of the LAST tile that are caller-supplied ZERO padding
            (Sk already rounded up to a 128-multiple). Their ones-column is zeroed so they add
            nothing to Σ P·1; Σ P·V is already zero because the padded V tokens are zero.
        total_kv_tiles: ceil(Sk / 128) — the exact flat tile count, including the tail tile. Sets
            the pipeline depth and the epilogue drain range; the last prefix block may be partial,
            so this cannot be derived from num_blocks.
        inside_fori_loop: True when called from an nl.fori_loop body. There the K/V stream loads
            use hwdge, which DMAs inside the loop require, and the pipeline is not wrapped in
            nl.no_reorder(), which is not allowed inside the loop. Otherwise the loads use the
            default DGE mode, and each block's tiles and the epilogue run under nl.no_reorder().
        prologue_depth: MM1+exp tiles issued before the first MM2 (capped at total_kv_tiles). The
            Tensor engine runs these MM1s ahead of the Vector subtract, then idles until the
            first MM2's probabilities are ready, so a smaller depth shortens that idle gap at the
            start of each Q-block. MM2 order, and therefore the result, does not depend on it.
        sbuf_allocator: Optional ModularAllocator for the V_aug, probability and shifted tiles
            (fixed SBUF addresses); None lets the compiler place them.
        num_shifted_buffers: slots in the fp16 `shifted` pool (see shifted_all below).
        v_direct: Optional ``(v_root, element_offset)`` of this head's [Sk, d] V in HBM. When given,
            each V block is DMAed straight into the MM2 moving operand [128 keys, tile, d+1], whose
            ones column is written once, so there is no per-tile [V|1] copy on the Vector engine
            (a large part of that engine's ring work). Requires a 128-multiple, block-streamed Sk.
        kv_resident: Optional ``(k_sb, v_sb)`` from _alloc_resident_kv: this head's whole K
            ([d, Sk]) and [V|1] ([128 keys, Sk/128 tiles, d+1], ones column and pad mask already
            set) in SBUF. The pipeline then reads every tile there and issues no K/V DMA; pass no
            stream, leftover or tail source with it.

    Sources are visited in KV order: source indices 0..num_blocks-1 are the streamed blocks,
    then the leftover grid when leftover_tiles > 0, then the tail when k_tail is not None. The
    tail is therefore the last source only when it exists: with leftover tiles and no tail (any
    128-multiple Sk above 16 tiles that is not a multiple of 16, such as the ring driver's padded
    2176), the last source is the leftover grid, which is why is_tail also checks k_tail.
    """
    d = _P
    num_blocks = k_stream.count if k_stream != None else 0
    sq_block_size = q_block.shape[1]
    num_output_tiles = o_aug_psums.shape[0]

    scores_dtype = nl.bfloat16 if nisa.get_nc_version() >= nisa.nc_version.gen4 else nl.float32
    # V-augmentation copy engine: route to whichever engine is NOT running exp so the copy overlaps.
    # NEVER GpSimd. The compiler triggers the ring's KV collective-permute from GpSimd, and that queue
    # is in-order, so a per-tile copy stream there defers every permute trigger until the previous ring
    # step's copies drain: the driver's kv_prefetch_depth hoist stops taking effect on hardware and the
    # hops go one-per-step, with a long channel idle in each hop. Where the collective cannot hide
    # behind per-core compute, that directly stalls the next step's MM1, so moving this copy off
    # GpSimd gives a shorter step (trn3 cp4, LNC1 and LNC2, on a 32,760-token sequence).
    #
    # The subtract runs on Vector and the exp on Scalar, so pick the lighter of those two. The
    # subtract's cost tracks the `shifted` precision, and `shifted` is fp16, which leaves Vector the
    # lighter of the pair -- so the copy goes there. It is cheaper there than on GpSimd.
    copy_engine = nisa.engine.vector

    # Number of MM1+exp tiles computed before first MM2 fires. Gives TensorE
    # a buffer of ready probabilities so it can always issue MM2 without waiting. Bounded by the
    # exact tile count (not num_blocks * tiles-per-block, which over-counts a partial last block
    # and would make the epilogue drain tiles that were never issued).
    #
    # Inside an nl.fori_loop body nothing pins MM1 to program order, so with
    # depth 8 the compiler lets MM1 run far ahead and a Q-block ends with ~18 KV tiles of MM2 still
    # queued; that backlog drains at the exp's pace without overlapping the next block. A shallower
    # pipeline there bounds the run-ahead.
    prologue_depth = min(_ROLLED_PROLOGUE_DEPTH if inside_fori_loop else prologue_depth, total_kv_tiles)

    # Rotating V_aug buffers: need prologue_depth+1 (v_copy runs at MM1 time,
    # MM2 consumes prologue_depth iterations later; +1 to avoid WAR)
    num_v_aug_buffers = prologue_depth + 1
    v_aug_all = None
    v_blk = None
    if kv_resident is not None:
        k_res, v_res = kv_resident
    elif v_direct is None:
        if sbuf_allocator is None:
            v_aug_all = nl.ndarray(shape=(_P, num_v_aug_buffers, d + 1), dtype=nl.bfloat16, buffer=nl.sbuf)
        else:
            v_aug_all = sbuf_allocator.alloc_sbuf_tensor(
                shape=(_P, num_v_aug_buffers, d + 1), dtype=nl.bfloat16, align_to=64
            )
        nisa.memset(v_aug_all[:, :, d : d + 1], 1.0)
    else:
        kernel_assert(k_tail is None and k_leftover is None, "direct V needs a 128-multiple, block-streamed Sk")
        last_slot = _V_DIRECT_ROTATING_SLOTS
        # One ndarray per slot, so a block's load waits only on the MM2s of that slot's previous block.
        v_blk = []
        for slot_idx in range(_V_DIRECT_ROTATING_SLOTS + 1):
            slot_buf = nl.ndarray(shape=(_P, _KV_TILES_PER_BLOCK, d + 1), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.memset(slot_buf[:, :, d : d + 1], 1.0)
            v_blk.append(slot_buf)
        if kv_pad_rows > 0:
            _mask_pad_keys(v_blk[last_slot], (total_kv_tiles - 1) % _KV_TILES_PER_BLOCK, kv_pad_rows, d)

    num_prob_buffers = prologue_depth + 1
    if sbuf_allocator is None:
        probs_all = nl.ndarray(shape=(_P, num_prob_buffers, sq_block_size), dtype=nl.bfloat16, buffer=nl.sbuf)
    else:
        probs_all = sbuf_allocator.alloc_sbuf_tensor(
            shape=(_P, num_prob_buffers, sq_block_size), dtype=nl.bfloat16, align_to=64
        )

    # `shifted = scores - c_i`, consumed by the exp. fp16 (not bf16) because the exp needs ABSOLUTE
    # precision in the exponent: bf16's 8-bit mantissa leaves ~0.004*|shifted| (~4% on a prob) and
    # fails accuracy, while fp16's 11 bits leave ~0.0005*|shifted| — below the error the bf16 PSUM
    # scores already carry, and finest exactly where |shifted| ~ 0 (the probs that dominate the sum).
    # Halving this tile vs fp32 also halves the subtract's write and the exp's read, the SBUF traffic
    # that slows both of them and, through them, the Tensor engine.
    # Lifetime is one tile (subtract → exp). NKI 0.7 no longer splits this pool, so with two slots
    # the subtract waits on the exp for almost every tile; the ring driver passes more on trn3
    # (ring_attention_const_max_fwd._num_shifted_buffers). The rolled path keeps two: a deeper pool is
    # slower at 21105 tokens/rank.
    # (One buffer per in-flight tile does not fix the head-0 miscompute: it makes that failure more
    # frequent and total rather than partial, so the defect is not this aliasing.)
    if sbuf_allocator is None:
        shifted_all = nl.ndarray(
            shape=(_P, num_shifted_buffers, sq_block_size),
            dtype=nl.float16,
            buffer=nl.sbuf,
        )
    else:
        shifted_all = sbuf_allocator.alloc_sbuf_tensor(
            shape=(_P, num_shifted_buffers, sq_block_size), dtype=nl.float16, align_to=64
        )

    # Per-tile valid-Sk-row slice, recorded as each tile is issued so the deferred MM2 (fired
    # prologue_depth iterations later) and the epilogue drain can reproduce that tile's extent.
    # Every whole-128 prefix tile gets the full _P-partition slice (the extent of every tile `rows`
    # indexes), so the prefix emits exactly the accesses it would for a 128-multiple Sk; only the
    # peeled tail tile is clamped.
    full_rows = slice(0, _P)
    tile_rows = []
    tile_vaug = []  # per tile, the [V|1] MM2 operand (rotating copy buffer, or the direct-V slot)

    kv_tile_idx = 0  # flat index across all KV tiles

    num_leftover_sources = 1 if leftover_tiles > 0 else 0
    num_sources = num_blocks + num_leftover_sources + (1 if k_tail is not None else 0)
    dge_mode = nisa.dge_mode.hwdge if inside_fori_loop else None

    def _issue_tiles(k_block, v_block, tiles_in_block, is_tail, kv_tile_idx, v_slot=0):
        """v_copy, MM1, exp (and the deferred MM2) for one source's (or one resident group's)
        tiles; returns the next flat index."""
        for local_tile in range(tiles_in_block):
            v_buf = kv_tile_idx % num_v_aug_buffers
            prob_buf = kv_tile_idx % num_prob_buffers
            # Rows valid in this tile: all 128 for a prefix tile, r_tail for the tail tile.
            rows = slice(0, v_block[local_tile, 0].data.shape[0]) if is_tail else full_rows
            tile_rows.append(rows)

            if kv_resident is not None:
                tile_vaug.append(v_res[:, kv_tile_idx, :])
            elif v_direct is not None:
                tile_vaug.append(v_blk[v_slot][:, local_tile, :])
            else:
                tile_vaug.append(v_aug_all[:, v_buf, :])
                # v_copy(i) — capture V tile now while block buffer is valid
                nisa.tensor_copy(
                    dst=v_aug_all[rows, v_buf, :d],
                    src=v_block[local_tile, 0].data,
                    engine=copy_engine,
                )
                # The resident and direct-V operands carry their pad mask from setup; this
                # rotating buffer needs it on the last tile.
                if kv_pad_rows > 0 and kv_tile_idx == total_kv_tiles - 1:
                    _mask_pad_keys(v_aug_all, v_buf, kv_pad_rows, d)

            # MM1(i) — the K tile's free extent is this tile's row count, so the matmul
            # writes exactly that many score partitions.
            scores = nl.ndarray(shape=(_P, sq_block_size), dtype=scores_dtype, buffer=nl.psum)
            if kv_resident is not None:
                k_stationary = k_res[:, kv_tile_idx * _P : (kv_tile_idx + 1) * _P]
            else:
                k_stationary = k_block[local_tile].data
            nisa.nc_matmul(dst=scores[rows, :], stationary=k_stationary, moving=q_block)

            # exp(i) — subtract the per-row bound, then exponentiate.
            # c_i varies along the free (Sq) axis, so it can't ride the exp's per-partition
            # max_value. Subtract it explicitly (Vector; scores PSUM - c SBUF into an fp16 tile,
            # see shifted_all), then exp on the SCALAR engine (nisa.activation) so the Vector
            # engine does ONLY the subtract and the two pipeline across tiles (the [V|1] copy
            # rides the lighter of the two). Running exp on Vector too is slower.
            # (Folding -c into the score bank ahead of an accumulating MM1 would remove this
            # pass, but nc_matmul cannot target an fp16 PSUM dst and a bf16 bank would round
            # scores-c, whose absolute error ~0.004*c is what the per-row bound exists to avoid.)
            shifted_buf = kv_tile_idx % num_shifted_buffers
            nisa.tensor_tensor(
                dst=shifted_all[rows, shifted_buf, :],
                data1=scores[rows, :],
                data2=per_row_bias[rows, :],
                op=nl.subtract,
                engine=nisa.engine.vector,
            )
            nisa.activation(
                dst=probs_all[rows, prob_buf, :],
                data=shifted_all[rows, shifted_buf, :],
                op=nl.exp,
                scale=1.0,
                bias=0.0,
            )

            # MM2(i-P) — only after prologue is filled
            prologue_filled = kv_tile_idx >= prologue_depth
            if prologue_filled:
                consume_idx = kv_tile_idx - prologue_depth
                _issue_mm2(
                    o_aug_psums=o_aug_psums,
                    probs_all=probs_all,
                    v_aug=tile_vaug[consume_idx],
                    tile_idx=consume_idx,
                    rows=tile_rows[consume_idx],
                    num_prob_buffers=num_prob_buffers,
                    num_output_tiles=num_output_tiles,
                )

            kv_tile_idx += 1
        return kv_tile_idx

    if kv_resident is not None:
        # Everything is already in SBUF, so there are no sources to load and the source loop below
        # is empty. Outside a loop the tiles are issued in _KV_TILES_PER_BLOCK groups under
        # nl.no_reorder(), like the streamed blocks.
        kernel_assert(num_sources == 0, "resident K/V takes no stream, leftover or tail source")
        for first_tile in range(0, total_kv_tiles, _KV_TILES_PER_BLOCK):
            group_tiles = min(_KV_TILES_PER_BLOCK, total_kv_tiles - first_tile)
            if inside_fori_loop:
                kv_tile_idx = _issue_tiles(None, None, group_tiles, False, kv_tile_idx)
            else:
                with nl.no_reorder():
                    kv_tile_idx = _issue_tiles(None, None, group_tiles, False, kv_tile_idx)
    for source_idx in range(num_sources):
        is_tail = k_tail is not None and source_idx == num_sources - 1
        if is_tail:
            k_block = k_tail
            v_block = v_tail
            tiles_in_block = 1
        elif source_idx >= num_blocks:
            k_block = k_leftover
            v_block = v_leftover
            tiles_in_block = leftover_tiles
        else:
            k_block = k_stream.load(source_idx, dge_mode=dge_mode)[0]
            tiles_in_block = k_block.shape[0]
            # Direct V never has a tail tile, the only place _issue_tiles reads v_block.
            v_block = k_block if v_direct is not None else v_stream.load(source_idx, dge_mode=dge_mode)

        v_slot = 0
        if v_direct is not None:
            v_root, v_base = v_direct
            v_slot = last_slot if source_idx == num_sources - 1 else source_idx % _V_DIRECT_ROTATING_SLOTS
            v_src = v_root.ap(pattern=[[d, _P], [_P * d, tiles_in_block], [1, d]], offset=v_base + kv_tile_idx * _P * d)
            if dge_mode is None:
                nisa.dma_copy(dst=v_blk[v_slot][:, :tiles_in_block, :d], src=v_src)
            else:
                nisa.dma_copy(dst=v_blk[v_slot][:, :tiles_in_block, :d], src=v_src, dge_mode=dge_mode)

        if inside_fori_loop:
            kv_tile_idx = _issue_tiles(k_block, v_block, tiles_in_block, is_tail, kv_tile_idx, v_slot)
        else:
            with nl.no_reorder():
                kv_tile_idx = _issue_tiles(k_block, v_block, tiles_in_block, is_tail, kv_tile_idx, v_slot)

    # === Epilogue: drain remaining prologue_depth MM2s ===
    def _drain_epilogue():
        for i in range(prologue_depth):
            drain_idx = total_kv_tiles - prologue_depth + i
            _issue_mm2(
                o_aug_psums=o_aug_psums,
                probs_all=probs_all,
                v_aug=tile_vaug[drain_idx],
                tile_idx=drain_idx,
                rows=tile_rows[drain_idx],
                num_prob_buffers=num_prob_buffers,
                num_output_tiles=num_output_tiles,
            )

    if inside_fori_loop:
        _drain_epilogue()
    else:
        with nl.no_reorder():
            _drain_epilogue()


def _issue_mm2(
    o_aug_psums,
    probs_all,
    v_aug,
    tile_idx,
    rows,
    num_prob_buffers,
    num_output_tiles,
) -> None:
    """MM2 for one KV tile: P @ [V|1] accumulated into the O_aug PSUM tiles (tiled along Sq).

    Both operands clamp their contraction (Sk) to `rows`, so a peeled sub-128 tail tile's padded
    rows add nothing to Σ P·V or Σ P·1. Shared by the steady-state and the epilogue drain, which
    issue the same MM2 for different tile indices.
    """
    prob_buf = tile_idx % num_prob_buffers
    should_accumulate = tile_idx != 0
    for ot_idx in range(num_output_tiles):
        sq_tile_size = o_aug_psums[ot_idx].data.shape[0]
        nisa.nc_matmul(
            dst=o_aug_psums[ot_idx].data,
            stationary=probs_all[rows, prob_buf, nl.ds(ot_idx * _O_TILE_SIZE, sq_tile_size)],
            moving=v_aug[rows, :],
            accumulate=should_accumulate,
        )


def _accumulate_partial(o_aug_psum, o_aug_acc_sb, aug_col, is_accumulate_step):
    """Fold one output tile's [Σ P·V | Σ P·1] from PSUM into a resident SBUF accumulator.

    Replaces the store-to-HBM + reload-and-add cross-step reduction with a single PSUM→SBUF op per
    tile: init (tensor_copy) on the first ring step, add (tensor_tensor) after.

    o_aug_psum: [sq_tile_size, d+1] PSUM — cols 0..d-1 are Σ P·V, col d is Σ P·1 (sum_exp).
    o_aug_acc_sb: [_P, ...] SBUF accumulator (this run's view); this tile lands at
        o_aug_acc_sb[:, aug_col:aug_col+d+1] (d output cols followed by 1 sum col).
    is_accumulate_step: True => += (fold), False => = (initialize).
    """
    sq_tile_size = o_aug_psum.shape[0]
    d = o_aug_psum.shape[1] - 1
    aug_dst = o_aug_acc_sb[:sq_tile_size, aug_col : aug_col + d + 1]
    if is_accumulate_step:
        nisa.tensor_tensor(dst=aug_dst, data1=aug_dst, data2=o_aug_psum, op=nl.add)
    else:
        nisa.tensor_copy(dst=aug_dst, src=o_aug_psum)


def _attention_const_max_hbm_acc(
    q_root,
    k_root,
    v_root,
    acc_root,
    c_root,
    o_root,
    slots,
    q_group,
    softmax_scale=None,
    is_first_step=True,
    is_last_step=True,
    kv_pad_rows=0,
    sum_clamp=1e-37,
    num_pad_keys=0,
    head_start=0,
    emit_static=True,
    emit_loops=True,
    tp_out=False,
):
    """Partial attention with the cross-step accumulator in HBM, the Q-block loop optionally rolled.

    Same math as _attention_const_max_partial; the ring driver uses it at inference whenever the
    SBUF accumulators do not fit next to a resident K/V pair or the Q-block loop is rolled (10
    heads at 9450 or 21105 tokens/rank). Each Q-block's [Σ P·V | Σ P·1] accumulator persists
    across ring steps in HBM, and every fold DMAs it into a fixed SBUF staging tile and back: ring
    step 0 writes it, later steps read-add-write it, and the last step normalizes it and writes the
    output instead (so there is no separate normalize pass). Keeping the accumulators out of SBUF
    (~54 MB per core at 21105 tokens/rank, ~24 MB at 9450) needs no head chunking, and leaves SBUF
    for each head's K and [V|1]: when one head's pair fits (_resident_kv_fits), it is loaded once
    per head per ring step and every Q-block reads it from SBUF instead of re-streaming the step's
    K/V from HBM.

    Shapes whose fully unrolled trace would overflow the BIR body (see choose_q_group) roll the
    Q-block loop. The nl.fori_loop body is traced once, so a Q-block's position is a runtime value
    (the loop register); the per-block DMAs take it as a scalar offset, and the compute reads fixed
    SBUF tiles. Per head, the leading full Q-blocks run as nl.fori_loop iterations of ``q_group``
    blocks each, unrolled inside the body; the remaining full blocks (at least ``head_start``) and
    the partial tail block are emitted statically. With ``head_start`` at least the number of full
    blocks nothing is rolled: every block is static, and each head's blocks follow that head's
    resident K/V load. With a loop, each head's loop follows its resident load, and the static
    blocks stream their K/V (they run for every head before any loop, see below).

    Every head's static blocks are emitted before any head's loop: entering a loop waits for
    in-flight transfers the loop does not use (the ring's K/V permute, the next step's K/V copy), so
    the static blocks are the only compute those transfers can overlap. ``emit_static`` /
    ``emit_loops`` select the two parts, so the ring driver can issue transfers between them that
    must not delay the static blocks. Every tensor is a ROOT HBM tensor indexed by the caller's head
    slot, and DMAs index the root (so indirect_dim is the root's row-major dimension). When there is
    a loop, nothing its body reads may be shared HBM (a shared read inside the loop needs a core
    barrier there), so the rolled driver passes private copies.

    Args:
        q_root: [H, d, Sq] bf16 @ HBM.
        k_root: [H, d, Sk] bf16 @ HBM, Sk a multiple of 128 (the ring driver pads it).
        v_root: [H, Sk, d] bf16 @ HBM.
        acc_root: [H, S_rows, d+1] fp32 @ private HBM, S_rows >= Sq; row i holds query i's
            [Σ P·V | Σ P·1].
        c_root: [H, S_rows] per-row softmax bound (see _attention_const_max_partial).
        o_root: [H, Sq, d] output, written on the last step; [H, d, Sq] with tp_out.
        slots: head slots (indices into dim 0 of every root tensor) to process.
        q_group: full Q-blocks per loop iteration (from choose_q_group).
        softmax_scale: scaling factor for Q. Defaults to 1/sqrt(d).
        is_first_step / is_last_step: position in the ring; both True for a single-step ring.
        kv_pad_rows: trailing zero-padding tokens in Sk (see _attention_const_max_partial).
        sum_clamp: floor on Σ P·1 before the reciprocal.
        num_pad_keys: zero-K/V keys in the whole ring (see ring_attention_const_max_fwd).
        head_start: minimum full Q-blocks per head emitted statically (from choose_q_group).
        emit_static / emit_loops: emit the static blocks / the loops (both by default).
        tp_out: write the output d-major, [H, d, Sq] (see _normalize_rows_to_hbm).
    """
    _, d, Sq = q_root.shape
    Sk = k_root.shape[2]
    softmax_scale = d ** (-0.5) if softmax_scale == None else softmax_scale
    kernel_assert(Sk % _P == 0, f"HBM-accumulator attention expects Sk padded to a multiple of {_P}, got {Sk}")
    max_sq = _max_sq_block_size()
    num_full = Sq // max_sq
    tail = Sq - num_full * max_sq
    rolled_blocks = rolled_blocks_for(num_full, q_group, head_start)
    shape = _HbmAccShape(d=d, sq=Sq, sk=Sk, s_rows=acc_root.shape[1])

    # Arguments every Q-block shares; the per-block ones are slot, start, size and the register.
    common = (q_root, k_root, v_root, acc_root, c_root, o_root, shape)
    flags = (softmax_scale, is_first_step, is_last_step, kv_pad_rows, sum_clamp, num_pad_keys, tp_out)
    # Resident K/V saves re-streaming the head's whole K/V from HBM for every Q-block (at 21420
    # tokens/rank, 5 heads x 36 looped blocks x 11 MB per ring step), which shortens the trn2 end-to-end time.
    kv_fits = _resident_kv_fits(Sk, d)
    if emit_static:
        kv_resident = None
        if rolled_blocks == 0 and kv_fits:
            kv_resident = _alloc_resident_kv(Sk, d, kv_pad_rows, k_root.dtype)
        for slot in slots:
            if kv_resident is not None:
                _load_resident_kv(k_root, v_root, slot, kv_resident, Sk, d)
            for block_idx in range(rolled_blocks, num_full):
                _hbm_acc_q_block(*common, slot, block_idx * max_sq, max_sq, None, *flags, kv_resident=kv_resident)
            if tail > 0:
                _hbm_acc_q_block(*common, slot, num_full * max_sq, tail, None, *flags, kv_resident=kv_resident)
    if emit_loops and rolled_blocks > 0:
        kv_resident = None
        if kv_fits:
            kv_resident = _alloc_resident_kv(Sk, d, kv_pad_rows, k_root.dtype)
        for slot in slots:
            if kv_resident is not None:
                _load_resident_kv(k_root, v_root, slot, kv_resident, Sk, d)

            def _group_body(sq_dyn, slot=slot):
                for local_block in range(q_group):
                    _hbm_acc_q_block(*common, slot, local_block * max_sq, max_sq, sq_dyn, *flags, kv_resident=kv_resident)

            nl.fori_loop(0, rolled_blocks * max_sq, _group_body, step=q_group * max_sq)


def _resident_kv_bytes(Sk, d):
    """Per-partition bytes of one head's resident K ([d, Sk] bf16) and [V|1] ([Sk/128, d+1] bf16)."""
    return Sk * 2 + (Sk // _P) * (d + 1) * 2


def _resident_kv_fits(Sk, d):
    """Whether one head's K and [V|1] fit the resident SBUF budget."""
    return _resident_kv_bytes(Sk, d) <= _RESIDENT_KV_SBUF_BUDGET_BYTES


def _alloc_resident_kv(Sk, d, kv_pad_rows, k_dtype):
    """One resident (K [d, Sk], [V|1] [128 keys, Sk/128 tiles, d+1]) pair in SBUF.

    The ones column is written once here, and masked on the final tile's pad keys; the per-head
    loads (_load_resident_kv) fill only the K and V columns. One pair is reused head after head:
    two alternating pairs are no faster on trn2 and corrupt the output of a 75,600-token
    sequence on trn3.
    """
    kv_tiles = Sk // _P
    k_res = nl.ndarray(shape=(_P, Sk), dtype=k_dtype, buffer=nl.sbuf)
    v_res = nl.ndarray(shape=(_P, kv_tiles, d + 1), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.memset(v_res[:, :, d : d + 1], 1.0)
    if kv_pad_rows > 0:
        _mask_pad_keys(v_res, kv_tiles - 1, kv_pad_rows, d)
    return k_res, v_res


def _load_resident_kv(k_root, v_root, slot, kv_resident, Sk, d):
    """Load head ``slot``'s K ([d, Sk]) and V ([Sk, d], straight into the [V|1] layout) into the
    resident pair, K in _RESIDENT_KV_PIECES pieces and V one 128-key tile per DMA.

    V is read through tensor slices, not one raw ``v_root.ap(...)`` per piece: with the raw access
    pattern, the ring's HBM-accumulator path gave Inf at a 32,760-token sequence, batch 2 (10 heads
    per core). The same raw read was correct when the buffer was filled by a copy rather than a
    ring permute.
    """
    k_res, v_res = kv_resident
    kv_tiles = Sk // _P
    piece = (kv_tiles + _RESIDENT_KV_PIECES - 1) // _RESIDENT_KV_PIECES
    for first_tile in range(0, kv_tiles, piece):
        end_tile = min(first_tile + piece, kv_tiles)
        nisa.dma_copy(dst=k_res[:, first_tile * _P : end_tile * _P], src=k_root[slot][:, first_tile * _P : end_tile * _P])
        for tile in range(first_tile, end_tile):
            nisa.dma_copy(dst=v_res[:, tile, :d], src=v_root[slot][tile * _P : (tile + 1) * _P, :])


def _mask_pad_keys(v_aug, tile, kv_pad_rows, d):
    """Zero the [V|1] ones column of ``v_aug[:, tile, :]`` on its trailing kv_pad_rows keys.

    When the caller padded Sk up to a 128-multiple with ZEROED tokens, the last tile's trailing
    kv_pad_rows partitions are padding. Their V data columns are zero, so Σ P·V is already exact —
    but the ones column would still fold those rows' exp into Σ P·1. Zeroing it there makes BOTH
    terms exactly zero, so no masked bias is needed. Both memsets start at partition 0 (a non-zero
    partition start offset is rejected), hence zero-the-column-then-rewrite-the-real-rows rather
    than memsetting the pad rows.
    """
    nisa.memset(v_aug[:, tile, d : d + 1], 0.0)
    nisa.memset(v_aug[0 : _P - kv_pad_rows, tile, d : d + 1], 1.0)


class _HbmAccShape(nl.NKIObject):
    """Root-tensor extents shared by every Q-block of one _attention_const_max_hbm_acc call."""

    def __init__(self, d, sq, sk, s_rows):
        self.d = d
        self.sq = sq
        self.sk = sk
        self.s_rows = s_rows


def _hbm_acc_q_block(
    q_root,
    k_root,
    v_root,
    acc_root,
    c_root,
    o_root,
    shape,
    slot,
    sq_start,
    sq_size,
    sq_dyn,
    softmax_scale,
    is_first_step,
    is_last_step,
    kv_pad_rows,
    sum_clamp,
    num_pad_keys,
    tp_out,
    kv_resident=None,
):
    """One Q-block of _attention_const_max_hbm_acc: MM1 → exp → MM2 over all KV tiles, then fold.

    Covers query rows sq_dyn + sq_start .. + sq_size of head ``slot``, where sq_dyn is the
    fori_loop register (the group's first query row) or None for a statically placed block. Every
    query-indexed DMA adds it as a scalar offset along the root tensor's query-row dimension
    (stride 1 for q and c, d+1 for the accumulator, d for a token-major output and 1 for a d-major
    one); nothing else depends on it. ``kv_resident`` is the head's resident K/V pair
    (_alloc_resident_kv), or None to stream them. ``tp_out`` stores the output d-major (see
    _normalize_rows_to_hbm).
    """
    d, Sq, Sk, s_rows = shape.d, shape.sq, shape.sk, shape.s_rows
    hwdge = nisa.dge_mode.hwdge

    q_block = nl.ndarray(shape=(d, sq_size), dtype=q_root.dtype, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=q_block,
        src=_row_ap(sq_dyn, q_root, [[Sq, d], [1, sq_size]], slot * d * Sq + sq_start, 2),
        dge_mode=hwdge,
    )
    nisa.tensor_scalar(dst=q_block, data=q_block, op0=nl.multiply, operand0=softmax_scale)

    # Per-row bound c_i on the free (Sq) axis, replicated down all 128 partitions by a
    # partition-stride-0 DMA (see _attention_const_max_partial).
    per_row_bias = nl.ndarray(shape=(_P, sq_size), dtype=c_root.dtype, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=per_row_bias,
        src=_row_ap(sq_dyn, c_root, [[0, _P], [1, sq_size]], slot * s_rows + sq_start, 1),
        dge_mode=hwdge,
    )

    num_output_tiles = (sq_size + _O_TILE_SIZE - 1) // _O_TILE_SIZE
    o_aug_psums = nt.psum_pool(tile_size=(_O_TILE_SIZE, d + 1), element_shape=(sq_size, d + 1))
    # Resident K/V, or K streamed in blocks with V DMAed straight into the [V|1] operand.
    k_stream = None
    v_direct = None
    if kv_resident is None:
        k_view = nt.blocks(k_root[slot], tile_size=(_P, d), block_size=(1, _KV_TILES_PER_BLOCK))
        k_stream = k_view[0, :].stream(buffer_count=2)
        v_direct = (v_root, slot * Sk * d)
    _process_kv_tiles_pipelined(
        k_stream=k_stream,
        v_stream=None,
        q_block=q_block,
        o_aug_psums=o_aug_psums,
        per_row_bias=per_row_bias,
        total_kv_tiles=Sk // _P,
        kv_pad_rows=kv_pad_rows,
        inside_fori_loop=sq_dyn is not None,
        v_direct=v_direct,
        kv_resident=kv_resident,
    )

    for output_tile_idx in range(num_output_tiles):
        o_aug_psum = o_aug_psums[output_tile_idx].data  # [rows, d+1]: Σ P·V cols, then Σ P·1
        rows = o_aug_psum.shape[0]
        row0 = sq_start + output_tile_idx * _O_TILE_SIZE
        acc_rows = _row_ap(sq_dyn, acc_root, [[d + 1, rows], [1, d + 1]], (slot * s_rows + row0) * (d + 1), 1)
        if is_first_step and is_last_step:
            folded = o_aug_psum
        else:
            stage = nl.ndarray(shape=(_P, d + 1), dtype=nl.float32, buffer=nl.sbuf)
            if is_first_step:
                nisa.tensor_copy(dst=stage[:rows, :], src=o_aug_psum)
            else:
                nisa.dma_copy(dst=stage[:rows, :], src=acc_rows, dge_mode=hwdge)
                nisa.tensor_tensor(dst=stage[:rows, :], data1=stage[:rows, :], data2=o_aug_psum, op=nl.add)
            folded = stage[:rows, :]

        if is_last_step:
            if tp_out:
                # o_root is [H, d, Sq]: partitions stride Sq, this tile's query rows contiguous.
                o_rows = _row_ap(sq_dyn, o_root, [[Sq, d], [1, rows]], slot * d * Sq + row0, 2)
            else:
                o_rows = _row_ap(sq_dyn, o_root, [[d, rows], [1, d]], (slot * Sq + row0) * d, 1)
            c_rows = _row_ap(sq_dyn, c_root, [[1, rows], [1, 1]], slot * s_rows + row0, 1)
            _normalize_rows_to_hbm(
                folded, rows, o_rows, d, sum_clamp, c_rows, num_pad_keys, tp_out=tp_out
            )
        else:
            nisa.dma_copy(dst=acc_rows, src=folded, dge_mode=hwdge)


def _pad_key_sum(c, rows, cols, num_pad_keys):
    """num_pad_keys·P_pad [_P, cols] fp32 for the per-row bounds c [rows, cols] @ SBUF.

    A zero-K/V key scores 0, so the pipeline adds P_pad = exp(0 - c_i) to the row's Σ P·1 and nothing
    to Σ P·V. P_pad is rebuilt as the pipeline rounds it (an fp16 shift, a bf16 P), so subtracting
    num_pad_keys of it drops those keys exactly.
    """
    shifted = nl.ndarray(shape=(_P, cols), dtype=nl.float16, buffer=nl.sbuf)
    probs = nl.ndarray(shape=(_P, cols), dtype=nl.bfloat16, buffer=nl.sbuf)
    pad_sum = nl.ndarray(shape=(_P, cols), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=shifted[:rows, :], data=c, op0=nl.multiply, operand0=-1.0)
    nisa.activation(dst=probs[:rows, :], data=shifted[:rows, :], op=nl.exp, scale=1.0, bias=0.0)
    nisa.tensor_scalar(dst=pad_sum[:rows, :], data=probs[:rows, :], op0=nl.multiply, operand0=float(num_pad_keys))
    return pad_sum


def _row_ap(sq_dyn, root, pattern, offset, dim):
    """Access pattern on a root HBM tensor, shifted by the fori_loop register ``sq_dyn`` (query rows,
    scaled by the stride of the root's dimension ``dim``) when the block is inside the loop."""
    if sq_dyn is None:
        return root.ap(pattern=pattern, offset=offset)
    return root.ap(pattern=pattern, offset=offset, scalar_offset=sq_dyn, indirect_dim=dim)


def _normalize_rows_to_hbm(
    acc_rows, rows, o_rows, d, sum_clamp, c_rows=None, num_pad_keys=0, tp_out=False
):
    """o = Σ P·V / max(Σ P·1 - num_pad_keys·P_pad, sum_clamp) for one output tile, stored to o_rows
    (HBM). c_rows is the tile's c_i @ HBM.

    With ``tp_out`` the tile is transposed on the PE and stored d-major — ``[d, rows]`` into a
    ``[d, Sq]`` output row — which is the layout the o-projection kernels consume, so the caller
    does not transpose the whole attention output in HBM afterwards. The store stays one DMA per
    tile: the tile's query rows are adjacent along the output's free axis. The transpose costs one
    128xrows ``nc_transpose`` and a PSUM bank per output tile.
    """
    sum_clamped = nl.ndarray(shape=(_P, 1), dtype=nl.float32, buffer=nl.sbuf)
    inv_sum = nl.ndarray(shape=(_P, 1), dtype=nl.float32, buffer=nl.sbuf)
    o_tile = nl.ndarray(shape=(_P, d), dtype=o_rows.dtype, buffer=nl.sbuf)
    sum_src = acc_rows[:, d : d + 1]
    if num_pad_keys:
        c_col = nl.ndarray(shape=(_P, 1), dtype=c_rows.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=c_col[:rows, :], src=c_rows, dge_mode=nisa.dge_mode.hwdge)
        pad_sum = _pad_key_sum(c_col[:rows, :], rows, 1, num_pad_keys)
        nisa.tensor_tensor(dst=sum_clamped[:rows, :], data1=sum_src, data2=pad_sum[:rows, :], op=nl.subtract)
        sum_src = sum_clamped[:rows, :]
    nisa.tensor_scalar(dst=sum_clamped[:rows, :], data=sum_src, op0=nl.maximum, operand0=sum_clamp)
    nisa.reciprocal(dst=inv_sum[:rows, :], data=sum_clamped[:rows, :])
    nisa.tensor_scalar(dst=o_tile[:rows, :], data=acc_rows[:, :d], op0=nl.multiply, operand0=inv_sum[:rows, :])
    if tp_out:
        # [rows, d] -> [d, rows] on the PE. A transposing DMA would move the tile as 2-byte
        # elements; the PE transpose keeps the store one descriptor per output partition.
        o_psum = nl.ndarray(shape=(d, _P), dtype=o_rows.dtype, buffer=nl.psum)
        nisa.nc_transpose(o_psum[:, :rows], o_tile[:rows, :])
        o_t = nl.ndarray(shape=(d, _P), dtype=o_rows.dtype, buffer=nl.sbuf)
        # Evict on the Scalar engine: Vector has the next tile's fold and scale.
        nisa.tensor_copy(o_t[:, :rows], o_psum[:, :rows], engine=nisa.engine.scalar)
        nisa.dma_copy(dst=o_rows, src=o_t[:, :rows], dge_mode=nisa.dge_mode.hwdge)
        return
    nisa.dma_copy(dst=o_rows, src=o_tile[:rows, :], dge_mode=nisa.dge_mode.hwdge)
