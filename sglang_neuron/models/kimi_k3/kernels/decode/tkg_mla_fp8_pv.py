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

"""The FP8 P@V pass of the packed p2 MLA verify (``tkg_mla_block`` ``packed_prior_options`` ``fp8_pv``, opt-in).

One pass attends the prior keys of S sequences x Q query rows. A sub-slot is one item (4 pages, 512 keys) x one group
of 128 (row, head) columns, as in the p2 loop. Per sub-slot:

1. Scores with the keys on partitions. For page t of the item, ``S^T_t [128 keys, 128 columns]`` is one MX latent QK
   (the page's key words as the stationary, the stage's MX query words as the moving data) plus one BF16 rope matmul
   of 98 contraction rows: rope 0 .. 63, the query bias rows 64 .. 64 + Q - 1 (the masks of the item), zeros, and the
   fixed softmax reference of each column at rows 96 (BF16 hi) and 97 (BF16 lo) against a key value of 1.0. The 4
   pages write the 4 contiguous 128-column slices of one FP32 PSUM bank.
2. One exp reads the 4 slices in (column, page) order and writes BF16 ``P^T [128 rows, (column, page)]``
   contiguously: the MX x4 layout (lane t = page t). ``nisa.quantize_mx`` gives the MXFP8 P^T with one E8M0 scale
   for each 32 keys.
3. One MX P@V (P^T as the stationary, the value words as the moving data) and one MX weight sum on 2 ones columns of
   the same FP8 P^T, so the numerator and the sum take the same rounded weights.

The numerator and the sum of 4 items of one (sequence, group) accumulate in PSUM before one state add. No P
transpose, no row max and no scale broadcast. The softmax frame is the served fixed reference, so the state of the
pass is in the frame of the p2 state and its store, reduce-scatter and normalization do not change. ``capped`` takes
the served score cap as one constant minimum of the shifted scores (scale x S' <= 60, the served cap in the shifted
frame); capless leaves it out (the overflow check and redo of the block guard that form).

Item layout (``fp8_pv_items``): each sequence has g key groups (``_pack_items`` step P1). It runs its groups last to
first. Its first chunk of 4 items holds p = (-g) mod 4 null items, its last group, and the next 3 - p groups. Only the
last group can hold keys at or past the position of a row (the CP32 key layout leaves 3,969 local offsets between
groups, more than Q - 1), so only the chunk writes mask rows and clamps its pages. The other groups of a sequence
follow as a stream whose length is a multiple of 4, in loops of 32, 16, 8 and 4 items.

Accuracy class: FP8 P (MXFP8 with E8M0 block scales), not the bits of the BF16-P loop.
"""

from contextlib import nullcontext

import nki.experimental as nke
import nki.isa as nisa
import nki.language as nl

from .utils.kernel_assert import kernel_assert
from .tkg_weight_dma_gate import pool_after, wait_for_completion
from .tkg_mla_paged import (
    _BIAS_ROW,
    _BLOCK,
    _CACHE_WIDTH,
    _GROUP_BLOCKS,
    _GROUP_STRIDE,
    _KV_LORA,
    _KV_LORA_TILES,
    _MASK_COLUMN_SCALE,
    _MX_PACK,
    _MX_UNIT_SCALE,
    _P2_STAGE_MX_SCALE,
    _P2_STAGE_REF_WORD,
    _P2_STAGE_WIDTH,
    _P_MAX,
    _ROPE,
    _SOFTMAX_SCALE,
    _STATE_WIDTH,
    _TI_COLS,
    _instruction,
    _pack_ti,
    _sanitize_cache_rows,
    _stage_query_rows_p2,
    pack_p2_groups,
    pack_p2_runs,
)

_KEYS = _GROUP_BLOCKS * _P_MAX
ITEM_FIELDS = 8
"""Fields of an item row (INT32): page ids 0 .. 3, sequence, mask threshold, valid, key group."""
CHUNK_ITEMS = 4
"""Items of the first chunk of a sequence (a PSUM accumulation block)."""
STREAM_BODIES = ((32, 1, None), (16, 2, 5), (8, 3, 6), (4, 4, 7))
"""The stream loops: (items, trip-count column, start-row column of ``seq_info``; None: the 32-item loop steps by
32 rows)."""
MASK_FP8 = -448.0
"""The FP8 key value of a masked key in its row's bias row (the query bias row holds 2^100, as ``_mask_column``)."""
FRAME_ROW = 96
"""Contraction rows 96 (hi) and 97 (lo) carry the fixed softmax reference; the rows past the bias rows are zero."""
_ROWS = FRAME_ROW + 2
CAP_SHIFTED = 60.0
"""The served score cap in the shifted frame: scale x (s + reference / scale) <= 60."""
_SCORE_BANKS = 3
_P_RING = 4
LOOKAHEAD_ITEMS = (0, 2, 4)
STREAM_FORMS = ("loops", "ifs", "ifs_grouped", "straight", "rolled", "batched", "exact")
"""``fp8_pv_loop`` ``stream``: the stream bodies as 0/1-trip ``fori_loop``s (the program), as forward then-only IFs on
their trip registers (also inside the IFs of their zero-together groups: "ifs_grouped"), or straight-line (every
body runs: a 1M test form only, NOT servable). "rolled" (no lookahead): one loop of 4-item bodies over the stream rows of
a sequence, a small program for a pass that seldom runs (the capless redo). "batched" (with ``lookahead``): one 0/1-trip
loop for each body size of the pass, with the bodies of all sequences in it. The trips are the bits of the longest
stream of the pass, and the exp bias of each item past the stream of its own sequence is ``-MASK_BIAS``
(``fp8_pv_loop``). "exact" (with ``lookahead`` 4): the rows that every sequence of the pass has in shared rolled loops
of 16-item and 4-item bodies, then the other rows of each sequence in rolled loops of its own inside one loop of 0 or 1
trips; no item runs past the stream of its own sequence (``fp8_pv_loop``)."""
MASK_BIAS = 30000.0
"""``fp8_pv_loop`` ``stream`` "batched": the negative exp bias of an item past the stream of its sequence. The exp of a
capped score minus this value is 0 in FP32, so the item adds 0 to the numerator and to the weight sum."""
_BATCHED_NO_MASK_TEST = False
"""Test only (the must-fail control of the "batched" stream): True gives every batched item the zero exp bias, so the
items past the stream of a short sequence add their weights. Keep False."""
_EXACT_NO_OWN_TEST = False
"""Test only (the must-fail control of the "exact" stream): True leaves out the own parts of the sequences (the rows
past the shortest stream of the pass), so a pass of mixed contexts drops rows. Keep False."""
"""``fp8_pv_loop`` ``lookahead``: the items of the next stream body whose pages the tail of a body loads (0: off)."""
QK_GROUP_PAGES = (0, 2, 4)
"""``fp8_pv_loop`` ``qk_group``: the pages of a sub-slot whose MX QK run before their rope matmuls (0: off)."""


def cap_weight_tile(tile, bias):
    """Return the capped weight of ``fp8_pv_loop`` ``cap_pool``: an FP32 ``[128, 1]`` SBUF tile of the BF16 exp of the
    score cap.

    The tile is the exp of the capped score that the capped loop gives: the same ScalarE exp, scale and ``bias`` as
    the P^T of a sub-slot, on the cap ``CAP_SHIFTED / scale`` (the FP32 operand of the VectorE cap), widened to FP32
    (exact). ``tile(shape, dtype)`` gives an SBUF tile. ``bias`` is the FP32 ``[128, 1]`` zero bias of that exp.

    Example:
        >>> cap = cap_weight_tile(sb.tile, zero[0:128, 8:9])  # doctest: +SKIP
    """
    P = _P_MAX
    vector = nisa.engine.vector
    score = tile((P, 1), nl.float32)
    nisa.memset(dst=score[0:P, 0:1], value=CAP_SHIFTED / _SOFTMAX_SCALE, engine=vector)
    weight = tile((P, 1), nl.bfloat16)
    nisa.activation(dst=weight[0:P, 0:1], op=nl.exp, data=score[0:P, 0:1], scale=_SOFTMAX_SCALE, bias=bias)
    cap = tile((P, 1), nl.float32)
    nisa.tensor_copy(dst=cap[0:P, 0:1], src=weight[0:P, 0:1], engine=vector)
    return cap


def cap_weights(p_t, cap):
    """Cap the BF16 P^T weights ``p_t`` ``[128, N]`` (SBUF) at ``cap`` (``cap_weight_tile``), in place on GpSimdE.

    The exp is monotonic, so ``min(exp(scale x s), exp(scale x c))`` is ``exp(scale x min(s, c))``: the weights of the
    VectorE score cap of ``stage_a``. The minimum of two BF16 values is exact in FP32.

    Example:
        >>> cap_weights(p_t[r], cap)  # doctest: +SKIP
    """
    nisa.tensor_scalar(
        dst=p_t, data=p_t, op0=nl.minimum, operand0=cap[0:_P_MAX, 0:1], engine=nisa.engine.gpsimd
    )


def fp8_pv_items(
    position,
    rank_info,
    table,
    first,
    seqs,
    queries,
    num_pages,
    tag,
    sb,
    page_runs=0,
    run_scale_dst=None,
    dma_engine=None,
    null_last=False,
    plan_reduce=False,
    pad_longest=False,
):
    """Return the item rows of one pass in HBM and the trip counts of its stream.

    ``position`` ``[B, 1]`` INT32 HBM, ``rank_info`` ``[1, >= 3]`` INT32 HBM (TP rank at column 2), ``table``
    ``[S, local_blocks]`` INT32 HBM (the pass rows of the block table), ``first`` the first sequence of the pass.
    Returns a dict:

    - ``chunk_hbm`` ``[4 S, 8]`` INT32: the first chunk of sequence b at rows 4 b .. 4 b + 3 (null items first).
    - ``rest_hbm`` ``[S x groups, 8]`` INT32: the stream of the other groups.
    - ``seq_info`` ``[1, 8 S]`` INT32 SBUF, per sequence b at columns 8 b ..: its stream start row E_b, its trip counts
      of loops of 32, 16, 8 and 4 items (r_b = 32 T32 + 16 T16 + 8 T8 + 4 T4), and the rows of its 16, 8 and 4 item
      loops relative to E_b.

    A row holds the 4 page ids of the group (clamped into the pool), the sequence, the mask threshold
    ``local - 16384 g`` of a real chunk item or ``-Q`` of a null item (key ``4096 b + p - j`` of row j is masked at or
    past it), the valid bit and the group. ``dma_engine`` (``fp8_pv_setup`` ``plan_first``): the hardware DGE queue of
    the plan's input loads and of its HBM round trip (None, the default: the default queue of each load).

    ``page_runs`` 2 (``fp8_pv_loop`` ``page_runs``): fields 0 and 1 hold the first pages of the group's two page runs
    (pages 0, 1 and 2, 3; each kept a run in the pool) and field 2 the broken bit of the row (its pages are not two runs
    of 2; an item of null pages only is not broken). ``run_scale`` ``[128, S]`` FP32 SBUF (in the dict) is then 1.0 for
    a sequence with no broken row and a quiet NaN otherwise, as the packed loop's ``page_runs``; ``run_scale_dst`` (a
    ``[128, S]`` FP32 tile, ``fp8_pv_setup`` ``step``) takes it in place of a tile of its own. ``null_last``
    (``fp8_pv_loop`` ``null_skip``): the chunk holds its real items first and its null items after them.

    ``plan_reduce`` (``fp8_pv_setup`` ``plan_reduce``): every sum over the S sequences of the plan is one
    ``nisa.tensor_reduce``. That is the field select of each row's sequence, the sequence of a stream row, the start of
    its sequence, and the E_b of the seq info (the stream starts of ``_rest_rows`` are one reduce each in every form).
    ``pad_longest`` (``fp8_pv_loop`` ``stream`` "batched"; False keeps the program): each row past the stream of the
    pass holds group 0 of the longest sequence (``_rest_rows``), so a batched body that reads the rows past the stream
    of a short sequence reads only full key groups.
    One op writes the S terms into a tile of their own, or the reduce reads ``seq_all``. No add then reads and writes a
    tile that a memset or an earlier add wrote. The compiler takes a memset and a chain of adds on one tile as an
    order-free reduction: each add waits only for the memset, so the chain form depends on the engine to keep the adds
    in order. The terms are the same small integers (one nonzero term in a select, 0 / 1 terms in a count, exact in
    FP32), so the item rows and the seq info are the same.
    """
    fp32 = nl.float32
    int32 = nl.int32
    vector = nisa.engine.vector
    local_blocks = table.shape[1]
    kernel_assert(
        local_blocks % _GROUP_BLOCKS == 0,
        f"fp8_pv needs whole key groups of {_GROUP_BLOCKS} blocks, got {local_blocks} local blocks",
    )
    groups = local_blocks // _GROUP_BLOCKS
    chunk_rows = CHUNK_ITEMS * seqs
    rest_rows = seqs * groups
    chunk_hbm = nl.ndarray((chunk_rows, ITEM_FIELDS), dtype=int32, buffer=nl.hbm, name=f"{tag}_fp8pv_chunk")
    rest_hbm = nl.ndarray((rest_rows, ITEM_FIELDS), dtype=int32, buffer=nl.hbm, name=f"{tag}_fp8pv_rest")
    seq_hbm = nl.ndarray((1, 4 * seqs), dtype=fp32, buffer=nl.hbm, name=f"{tag}_fp8pv_seq")
    seq_info = sb.tile((1, 8 * seqs), int32)
    # page_runs: the run scale outlives the scope (the loop applies it after its last state add), and so do the SBUF
    # sources of the item-row stores: a tile of a later scope could take their bytes while a store still reads them (a
    # DGE store's completion orders no later compute write).
    run_scale = (sb.tile((_P_MAX, seqs), fp32) if run_scale_dst is None else run_scale_dst) if page_runs else None
    store_srcs = (
        [sb.tile((_P_MAX, ITEM_FIELDS), int32) for _ in range(1 + -(-rest_rows // _P_MAX))] if page_runs else None
    )
    early = {} if dma_engine is None else {"dge_mode": nisa.dge_mode.hwdge, "engine": dma_engine}
    with sb.scope("fp8_pv_items"):
        # Per sequence (partition b): local position, group count g, null items p, stream groups r.
        pos_i = sb.tile((seqs, 1), int32)
        nisa.dma_copy(dst=pos_i, src=position[first : first + seqs, 0:1], name=f"{tag}_fp8pv_pos", **early)
        rank_i = sb.tile((seqs, 1), int32)
        nisa.dma_copy(dst=rank_i, src=rank_info[0:1, 2:3].broadcast(0, seqs), name=f"{tag}_fp8pv_rank", **early)
        seq_f = sb.tile((seqs, 4), fp32)
        rank_f = sb.tile((seqs, 1), fp32)
        nisa.tensor_copy(dst=rank_f, src=rank_i, engine=vector)
        pos_f = sb.tile((seqs, 1), fp32)
        nisa.tensor_copy(dst=pos_f, src=pos_i, engine=vector)
        # local = position - 128 rank; g = min((local + 16383 + Q - 1) >> 14, groups), as _pack_items step P1.
        nisa.scalar_tensor_tensor(
            dst=seq_f[0:seqs, 0:1],
            data=rank_f,
            op0=nl.multiply,
            operand0=-float(_BLOCK),
            op1=nl.add,
            operand1=pos_f,
        )
        g_i = sb.tile((seqs, 3), int32)
        g_f = sb.tile((seqs, 1), fp32)
        nisa.tensor_scalar(
            dst=g_f,
            data=seq_f[0:seqs, 0:1],
            op0=nl.add,
            operand0=float(_GROUP_STRIDE - 1 + queries - 1),
            engine=vector,
        )
        nisa.tensor_copy(dst=g_i[0:seqs, 0:1], src=g_f, engine=vector)
        nisa.tensor_scalar(dst=g_i[0:seqs, 0:1], data=g_i[0:seqs, 0:1], op0=nl.right_shift, operand0=14, engine=vector)
        nisa.tensor_scalar(
            dst=g_i[0:seqs, 0:1], data=g_i[0:seqs, 0:1], op0=nl.minimum, operand0=float(groups), engine=vector
        )
        # p = 4 - (g mod 4) where g mod 4 > 0: m = g & 3 in INT32, the rest in FP32 (exact small integers).
        nisa.tensor_scalar(
            dst=g_i[0:seqs, 1:2],
            data=g_i[0:seqs, 0:1],
            op0=nl.bitwise_and,
            operand0=CHUNK_ITEMS - 1,
            engine=vector,
        )
        nisa.tensor_copy(dst=seq_f[0:seqs, 1:3], src=g_i[0:seqs, 0:2], engine=vector)
        m_pos = sb.tile((seqs, 1), fp32)
        nisa.tensor_scalar(dst=m_pos, data=seq_f[0:seqs, 2:3], op0=nl.greater, operand0=0.0, engine=vector)
        nisa.tensor_scalar(
            dst=seq_f[0:seqs, 2:3],
            data=seq_f[0:seqs, 2:3],
            op0=nl.multiply,
            operand0=-1.0,
            op1=nl.add,
            operand1=float(CHUNK_ITEMS),
            engine=vector,
        )
        nisa.tensor_tensor(dst=seq_f[0:seqs, 2:3], data1=seq_f[0:seqs, 2:3], data2=m_pos, op=nl.multiply, engine=vector)
        # r = max(g + p - 4, 0): the stream groups of the sequence.
        nisa.scalar_tensor_tensor(
            dst=seq_f[0:seqs, 3:4],
            data=seq_f[0:seqs, 1:2],
            op0=nl.add,
            operand0=-float(CHUNK_ITEMS),
            op1=nl.add,
            operand1=seq_f[0:seqs, 2:3],
        )
        nisa.tensor_scalar(dst=seq_f[0:seqs, 3:4], data=seq_f[0:seqs, 3:4], op0=nl.maximum, operand0=0.0, engine=vector)
        # Every partition gets [local, g, p, r] of every sequence (an HBM round trip and a broadcast DMA).
        nisa.dma_copy(dst=seq_hbm.reshape((seqs, 4)), src=seq_f, name=f"{tag}_fp8pv_seq_store", **early)
        seq_all = sb.tile((_P_MAX, seqs, 4), fp32)
        nisa.dma_copy(
            dst=seq_all.reshape((_P_MAX, 4 * seqs)),
            src=seq_hbm.broadcast(0, _P_MAX),
            name=f"{tag}_fp8pv_seq_load",
            **early,
        )
        runs = _runs_state(seqs, sb, store_srcs) if page_runs else None
        if plan_reduce:
            ids = _seq_ids(seqs, sb)
            _chunk_rows(
                seq_all, table, chunk_hbm, seqs, queries, groups, num_pages, tag, sb, runs, null_last=null_last, ids=ids
            )
            _rest_rows(seq_all, table, rest_hbm, seqs, groups, num_pages, tag, sb, runs, ids=ids, pad_longest=pad_longest)
            _seq_info(seq_all, seq_info, seqs, sb, reduce=True)
        else:
            _chunk_rows(seq_all, table, chunk_hbm, seqs, queries, groups, num_pages, tag, sb, runs, null_last=null_last)
            _rest_rows(seq_all, table, rest_hbm, seqs, groups, num_pages, tag, sb, runs, pad_longest=pad_longest)
            _seq_info(seq_all, seq_info, seqs, sb)
        if page_runs:
            _run_scale(runs, run_scale, seqs, sb)
    out = {"chunk_hbm": chunk_hbm, "rest_hbm": rest_hbm, "seq_info": seq_info}
    if page_runs:
        out["run_scale"] = run_scale
    return out


def _runs_state(seqs, sb, store_srcs):
    """``page_runs``: the broken-row counts of the pass, ``[128, S]`` FP32 PSUM: each row tile adds its broken rows of
    sequence b to column b of every partition (a ones stationary over the row partitions, exact small integers); and
    ``store_srcs``, the item-row store sources of the pass (one a row tile, allocated outside the items scope)."""
    fp32 = nl.float32
    vector = nisa.engine.vector
    ones = sb.tile((_P_MAX, _P_MAX), nl.bfloat16)
    nisa.memset(dst=ones, value=1.0, engine=vector)
    seq_ids = sb.tile((_P_MAX, seqs), nl.int32)
    nisa.iota(dst=seq_ids, pattern=[[1, seqs]], offset=0, channel_multiplier=0)
    seq_f = sb.tile((_P_MAX, seqs), fp32)
    nisa.tensor_copy(dst=seq_f, src=seq_ids, engine=vector)
    counts = nl.ndarray((_P_MAX, seqs), dtype=fp32, buffer=nl.psum)
    return {"ones": ones, "seq": seq_f, "counts": counts, "tiles": 0, "srcs": list(store_srcs)}


def _runs_count(runs, item_f, rows, sb):
    """``page_runs``: add the broken bits (field 2) of ``rows`` item rows to their sequences' counts (field 4)."""
    seqs = runs["seq"].shape[1]
    hit = sb.tile((_P_MAX, seqs), nl.bfloat16)
    nisa.tensor_scalar(
        dst=hit[0:rows],
        data=runs["seq"][0:rows],
        op0=nl.equal,
        operand0=item_f[0:rows, 4:5],
        op1=nl.multiply,
        operand1=item_f[0:rows, 2:3],
        engine=nisa.engine.vector,
    )
    nisa.nc_matmul(
        dst=runs["counts"], stationary=runs["ones"][0:rows], moving=hit[0:rows], accumulate=runs["tiles"] > 0
    )
    runs["tiles"] += 1


def _run_scale(runs, run_scale, seqs, sb):
    """``page_runs``: ``run_scale`` ``[128, S]`` FP32 = 1.0 where sequence b has no broken row, else a quiet NaN (the
    bits 0x3F800000 / 0x7FC00000, exact in FP32), on every partition."""
    fp32 = nl.float32
    int32 = nl.int32
    vector = nisa.engine.vector
    scale_f = sb.tile((_P_MAX, seqs), fp32)
    nisa.tensor_scalar(
        dst=scale_f,
        data=runs["counts"],
        op0=nl.minimum,
        operand0=1.0,
        op1=nl.multiply,
        operand1=float(0x7FC00000 - 0x3F800000),
        engine=vector,
    )
    nisa.tensor_scalar(dst=run_scale.view(int32), data=scale_f, op0=nl.add, operand0=float(0x3F800000), engine=vector)


def _seq_info(seq_all, seq_info, seqs, sb, reduce=False):
    """Per sequence on partition 0: E_b, T32, T16, T8, T4 of r_b, and the relative rows 32 T32, + 16 T16, + 8 T8.
    ``reduce`` (``fp8_pv_items`` ``plan_reduce``): each E_b is one reduce of r_0 .. r_{b-1} from ``seq_all``, not an add
    to E_{b-1} (the same sums of small integers)."""
    fp32 = nl.float32
    int32 = nl.int32
    vector = nisa.engine.vector
    info_f = sb.tile((1, 8 * seqs), fp32)
    nisa.memset(dst=info_f, value=0.0, engine=vector)
    for b in range(1, seqs):
        col = 8 * b
        if reduce:
            nisa.tensor_reduce(dst=info_f[0:1, col : col + 1], data=seq_all[0:1, 0:b, 3], op=nl.add, axis=1)
            continue
        nisa.tensor_tensor(
            dst=info_f[0:1, col : col + 1],
            data1=info_f[0:1, col - 8 : col - 7],
            data2=seq_all[0:1, b - 1, 3:4],
            op=nl.add,
            engine=vector,
        )
    info_i = sb.tile((1, 8 * seqs), int32)
    r_i = sb.tile((1, seqs), int32)
    for b in range(seqs):
        nisa.tensor_copy(dst=r_i[0:1, b : b + 1], src=seq_all[0:1, b, 3:4], engine=vector)
    nisa.tensor_copy(dst=info_i, src=info_f, engine=vector)
    for b in range(seqs):
        col = 8 * b
        for k, shift in enumerate((5, 4, 3, 2)):
            dst = info_i[0:1, col + 1 + k : col + 2 + k]
            nisa.tensor_scalar(dst=dst, data=r_i[0:1, b : b + 1], op0=nl.right_shift, operand0=shift, engine=vector)
            if k:
                nisa.tensor_scalar(dst=dst, data=dst, op0=nl.bitwise_and, operand0=1, engine=vector)
    nisa.tensor_copy(dst=info_f, src=info_i, engine=vector)
    for b in range(seqs):
        col = 8 * b
        # rel16 = 32 T32; rel8 = rel16 + 16 T16; rel4 = rel8 + 8 T8.
        nisa.tensor_scalar(
            dst=info_f[0:1, col + 5 : col + 6],
            data=info_f[0:1, col + 1 : col + 2],
            op0=nl.multiply,
            operand0=32.0,
            engine=vector,
        )
        for k, size in ((6, 16.0), (7, 8.0)):
            nisa.scalar_tensor_tensor(
                dst=info_f[0:1, col + k : col + k + 1],
                data=info_f[0:1, col + k - 4 : col + k - 3],
                op0=nl.multiply,
                operand0=size,
                op1=nl.add,
                operand1=info_f[0:1, col + k - 1 : col + k],
            )
    nisa.tensor_copy(dst=seq_info, src=info_f, engine=vector)


def _rows_store(group_f, seq_f, thr_f, valid_f, table, dst, rows, groups, num_pages, tag, sb, label, runs=None):
    """Gather the page ids of ``rows`` items (one a partition: group, sequence, threshold, valid as FP32 columns) and
    store their item rows to ``dst`` (``runs``: ``_runs_state``, the run fields of ``fp8_pv_items`` page_runs)."""
    fp32 = nl.float32
    int32 = nl.int32
    vector = nisa.engine.vector
    row_f = sb.tile((_P_MAX, 1), fp32)
    # Table row of the group: sequence x groups + group, on the [S x groups, 4] view of the pass table.
    nisa.scalar_tensor_tensor(
        dst=row_f[0:rows],
        data=seq_f[0:rows],
        op0=nl.multiply,
        operand0=float(groups),
        op1=nl.add,
        operand1=group_f[0:rows],
    )
    row_i = sb.tile((_P_MAX, 1), int32)
    nisa.tensor_copy(dst=row_i[0:rows], src=row_f[0:rows], engine=vector)
    pages_i = sb.tile((_P_MAX, _GROUP_BLOCKS), int32)
    nisa.dma_copy(
        dst=pages_i[0:rows],
        src=table.reshape((table.shape[0] * groups, _GROUP_BLOCKS)).ap(
            pattern=[[_GROUP_BLOCKS, rows], [1, _GROUP_BLOCKS]],
            offset=0,
            vector_offset=row_i[0:rows, 0:1],
            indirect_dim=0,
        ),
        name=f"{tag}_fp8pv_pages_{label}",
    )
    item_f = sb.tile((_P_MAX, ITEM_FIELDS), fp32)
    nisa.tensor_copy(dst=item_f[0:rows, 0:_GROUP_BLOCKS], src=pages_i[0:rows], engine=vector)
    nisa.tensor_scalar(
        dst=item_f[0:rows, 0:_GROUP_BLOCKS],
        data=item_f[0:rows, 0:_GROUP_BLOCKS],
        op0=nl.minimum,
        operand0=float(num_pages - 1),
        op1=nl.maximum,
        operand1=0.0,
        engine=vector,
    )
    nisa.tensor_copy(dst=item_f[0:rows, 4:5], src=seq_f[0:rows], engine=vector)
    nisa.tensor_copy(dst=item_f[0:rows, 5:6], src=thr_f[0:rows], engine=vector)
    nisa.tensor_copy(dst=item_f[0:rows, 6:7], src=valid_f[0:rows], engine=vector)
    nisa.tensor_copy(dst=item_f[0:rows, 7:8], src=group_f[0:rows], engine=vector)
    if runs is not None:
        _run_fields(item_f, rows, num_pages, sb)
        _runs_count(runs, item_f, rows, sb)
    item_i = sb.tile((_P_MAX, ITEM_FIELDS), int32) if runs is None else runs["srcs"].pop(0)
    nisa.tensor_copy(dst=item_i[0:rows], src=item_f[0:rows], engine=vector)
    nisa.dma_copy(dst=dst[0:rows, 0:ITEM_FIELDS], src=item_i[0:rows], name=f"{tag}_fp8pv_rows_store_{label}")


def _run_fields(item_f, rows, num_pages, sb):
    """``page_runs`` 2: rewrite fields 0 .. 2 of ``rows`` item rows (FP32, pages 0 .. 3 clamped into the pool): the first
    pages of the runs (pages 0 and 2, each at most the pool's last page - 1) and the broken bit: a valid row (field 6)
    whose pages 1, 3 are not pages 0, 2 + 1, unless every page is 0 (null pages only). A row that is not valid (a null
    chunk item: its keys are masked, and the chunk clamps its pages) or not loaded counts no break. Small integers,
    exact in FP32."""
    fp32 = nl.float32
    vector = nisa.engine.vector
    step = sb.tile((_P_MAX, 2), fp32)
    nisa.tensor_tensor(
        dst=step[0:rows],
        data1=item_f[0:rows, 1:_GROUP_BLOCKS:2],
        data2=item_f[0:rows, 0:_GROUP_BLOCKS:2],
        op=nl.subtract,
        engine=vector,
    )
    nisa.tensor_scalar(dst=step[0:rows], data=step[0:rows], op0=nl.not_equal, operand0=1.0, engine=vector)
    broken = sb.tile((_P_MAX, 1), fp32)
    nisa.tensor_reduce(dst=broken[0:rows], op=nl.maximum, data=step[0:rows], axis=1)
    used = sb.tile((_P_MAX, 1), fp32)
    nisa.tensor_reduce(dst=used[0:rows], op=nl.maximum, data=item_f[0:rows, 0:_GROUP_BLOCKS], axis=1)
    nisa.scalar_tensor_tensor(
        dst=broken[0:rows],
        data=used[0:rows],
        op0=nl.minimum,
        operand0=1.0,
        op1=nl.multiply,
        operand1=broken[0:rows],
    )
    # Field 1 = the first page of run 1 (page 2) before field 2 takes the broken bit; field 0 in place.
    last_first = float(num_pages - 2)
    nisa.tensor_scalar(
        dst=item_f[0:rows, 1:2], data=item_f[0:rows, 2:3], op0=nl.minimum, operand0=last_first, engine=vector
    )
    nisa.tensor_scalar(
        dst=item_f[0:rows, 0:1], data=item_f[0:rows, 0:1], op0=nl.minimum, operand0=last_first, engine=vector
    )
    nisa.tensor_tensor(
        dst=item_f[0:rows, 2:3], data1=broken[0:rows], data2=item_f[0:rows, 6:7], op=nl.multiply, engine=vector
    )


def _seq_ids(seqs, sb):
    """``plan_reduce``: ``[128, S]`` FP32 with b in column b on every partition (exact small integers)."""
    ids_i = sb.tile((_P_MAX, seqs), nl.int32)
    nisa.iota(dst=ids_i, pattern=[[1, seqs]], offset=0, channel_multiplier=0)
    ids = sb.tile((_P_MAX, seqs), nl.float32)
    nisa.tensor_copy(dst=ids, src=ids_i, engine=nisa.engine.vector)
    return ids


def _seq_pick(ids, seq_col, values, seqs, rows, sb):
    """``plan_reduce``: ``[rows, 1]`` FP32 ``values[p, seq_col[p]]`` of each partition. One op writes the S terms
    ``(b == seq_col) x values[:, b]`` into a tile of its own, then one reduce sums them: the terms of ``_seq_value``
    (one nonzero term), with no add into a tile that an earlier op wrote."""
    fp32 = nl.float32
    terms = sb.tile((_P_MAX, seqs), fp32)
    nisa.scalar_tensor_tensor(
        dst=terms[0:rows],
        data=ids[0:rows],
        op0=nl.equal,
        operand0=seq_col[0:rows],
        op1=nl.multiply,
        operand1=values,
    )
    out = sb.tile((_P_MAX, 1), fp32)
    nisa.tensor_reduce(dst=out[0:rows], op=nl.add, data=terms[0:rows], axis=1)
    return out


def _seq_value(seq_all, seq_col, field, seqs, rows, sb, ids=None):
    """``[rows, 1]`` FP32: field ``field`` of the sequence in ``seq_col`` of each partition (a select over the S).
    ``ids`` (``_seq_ids``, ``fp8_pv_items`` ``plan_reduce``): the select is one reduce (``_seq_pick``)."""
    if ids is not None:
        return _seq_pick(ids, seq_col, seq_all[0:rows, 0:seqs, field], seqs, rows, sb)
    fp32 = nl.float32
    vector = nisa.engine.vector
    out = sb.tile((_P_MAX, 1), fp32)
    nisa.memset(dst=out[0:rows], value=0.0, engine=vector)
    pick = sb.tile((_P_MAX, 1), fp32)
    for b in range(seqs):
        # pick = (sequence == b) x field of b; out += pick.
        nisa.tensor_scalar(
            dst=pick[0:rows],
            data=seq_col[0:rows],
            op0=nl.equal,
            operand0=float(b),
            op1=nl.multiply,
            operand1=seq_all[0:rows, b, field : field + 1],
            engine=vector,
        )
        nisa.tensor_tensor(dst=out[0:rows], data1=out[0:rows], data2=pick[0:rows], op=nl.add, engine=vector)
    return out


def _select(dst, keep, other, rows, sb):
    """``dst = keep ? dst : other`` (keep 1.0 or 0.0, exact in FP32)."""
    vector = nisa.engine.vector
    diff = sb.tile((_P_MAX, 1), nl.float32)
    nisa.tensor_tensor(dst=diff[0:rows], data1=dst[0:rows], data2=other[0:rows], op=nl.subtract, engine=vector)
    nisa.tensor_tensor(dst=diff[0:rows], data1=diff[0:rows], data2=keep[0:rows], op=nl.multiply, engine=vector)
    nisa.tensor_tensor(dst=dst[0:rows], data1=other[0:rows], data2=diff[0:rows], op=nl.add, engine=vector)


def _chunk_rows(
    seq_all, table, chunk_hbm, seqs, queries, groups, num_pages, tag, sb, runs=None, null_last=False, ids=None
):
    """The first chunk of each sequence: item 4 b + k is null for k < p, else group g - 1 + p - k (null too if < 0).
    ``null_last`` (``fp8_pv_loop`` ``null_skip``): the real items first, item 4 b + k is group g - 1 - k for k < 4 - p
    (null too if < 0), else null: the same groups in the same order, the null items after them. ``ids``
    (``fp8_pv_items`` ``plan_reduce``): the selects of ``_seq_value`` take one reduce each."""
    fp32 = nl.float32
    int32 = nl.int32
    vector = nisa.engine.vector
    rows = CHUNK_ITEMS * seqs
    idx_i = sb.tile((_P_MAX, 1), int32)
    nisa.iota(dst=idx_i[0:rows], pattern=[[0, 1]], offset=0, channel_multiplier=1)
    # Sequence b = idx >> 2 and chunk index k = idx & 3.
    sk_i = sb.tile((_P_MAX, 2), int32)
    nisa.tensor_scalar(dst=sk_i[0:rows, 0:1], data=idx_i[0:rows], op0=nl.right_shift, operand0=2, engine=vector)
    nisa.tensor_scalar(
        dst=sk_i[0:rows, 1:2], data=idx_i[0:rows], op0=nl.bitwise_and, operand0=CHUNK_ITEMS - 1, engine=vector
    )
    sk_f = sb.tile((_P_MAX, 2), fp32)
    nisa.tensor_copy(dst=sk_f[0:rows], src=sk_i[0:rows], engine=vector)
    seq_f = sk_f[0:rows, 0:1]
    local = _seq_value(seq_all, sk_f[:, 0:1], 0, seqs, rows, sb, ids)
    g = _seq_value(seq_all, sk_f[:, 0:1], 1, seqs, rows, sb, ids)
    p = _seq_value(seq_all, sk_f[:, 0:1], 2, seqs, rows, sb, ids)
    group_f = sb.tile((_P_MAX, 1), fp32)
    valid_f = sb.tile((_P_MAX, 1), fp32)
    if null_last:
        # group = g - 1 - k; valid = (k + p < 4) and (group >= 0).
        nisa.tensor_scalar(dst=group_f[0:rows], data=g[0:rows], op0=nl.add, operand0=-1.0, engine=vector)
        nisa.tensor_tensor(
            dst=group_f[0:rows], data1=group_f[0:rows], data2=sk_f[0:rows, 1:2], op=nl.subtract, engine=vector
        )
        nisa.tensor_tensor(dst=valid_f[0:rows], data1=sk_f[0:rows, 1:2], data2=p[0:rows], op=nl.add, engine=vector)
        nisa.tensor_scalar(
            dst=valid_f[0:rows], data=valid_f[0:rows], op0=nl.less, operand0=float(CHUNK_ITEMS), engine=vector
        )
    else:
        # group = g - 1 + p - k; valid = (k >= p) and (group >= 0).
        nisa.scalar_tensor_tensor(
            dst=group_f[0:rows], data=g[0:rows], op0=nl.add, operand0=-1.0, op1=nl.add, operand1=p[0:rows]
        )
        nisa.tensor_tensor(
            dst=group_f[0:rows], data1=group_f[0:rows], data2=sk_f[0:rows, 1:2], op=nl.subtract, engine=vector
        )
        nisa.tensor_tensor(
            dst=valid_f[0:rows], data1=sk_f[0:rows, 1:2], data2=p[0:rows], op=nl.greater_equal, engine=vector
        )
    real_f = sb.tile((_P_MAX, 1), fp32)
    nisa.tensor_scalar(dst=real_f[0:rows], data=group_f[0:rows], op0=nl.greater_equal, operand0=0.0, engine=vector)
    nisa.tensor_tensor(dst=valid_f[0:rows], data1=valid_f[0:rows], data2=real_f[0:rows], op=nl.multiply, engine=vector)
    # A null item takes the pages of group max(g - 1, 0) (allocated, finite after the clamp).
    null_group = sb.tile((_P_MAX, 1), fp32)
    nisa.tensor_scalar(
        dst=null_group[0:rows], data=g[0:rows], op0=nl.add, operand0=-1.0, op1=nl.maximum, operand1=0.0, engine=vector
    )
    _select(group_f, valid_f, null_group, rows, sb)
    # threshold = valid ? local - 16384 group : -Q.
    thr_f = sb.tile((_P_MAX, 1), fp32)
    nisa.scalar_tensor_tensor(
        dst=thr_f[0:rows],
        data=group_f[0:rows],
        op0=nl.multiply,
        operand0=-float(_GROUP_STRIDE),
        op1=nl.add,
        operand1=local[0:rows],
    )
    null_thr = sb.tile((_P_MAX, 1), fp32)
    nisa.memset(dst=null_thr[0:rows], value=-float(queries), engine=vector)
    _select(thr_f, valid_f, null_thr, rows, sb)
    _rows_store(group_f, seq_f, thr_f, valid_f, table, chunk_hbm, rows, groups, num_pages, tag, sb, "chunk", runs)


def _rest_rows(seq_all, table, rest_hbm, seqs, groups, num_pages, tag, sb, runs=None, ids=None, pad_longest=False):
    """The stream: item j of sequence b (j - E_{b-1} < r_b) holds group g - 1 + p - 4 - (j - E_{b-1}). ``ids``
    (``fp8_pv_items`` ``plan_reduce``): each count and select below is one reduce, as each start is in every form.
    ``pad_longest`` (``fp8_pv_items``; False keeps the program): a row past the stream of the pass takes group 0 of the
    longest sequence of the pass, not group 0 of the last sequence. The row stays not valid (valid 0)."""
    fp32 = nl.float32
    int32 = nl.int32
    vector = nisa.engine.vector
    rows_total = seqs * groups
    # Stream starts E_{b-1} (exclusive prefix sums of r) on every partition: [P, S].
    # Each start is summed directly from seq_all (one reduce per column), so no column reads a
    # value a prior instruction wrote. The chained form gave a wrong start on device for the
    # sequence after one with stream rows (0 in place of the row count), so a row was tagged with
    # the next sequence and that sequence read the next sequence's pages; the direct sum keeps
    # every start correct on device. The simulator runs the adds in order, so it never saw this.
    starts = sb.tile((_P_MAX, seqs), fp32)
    nisa.memset(dst=starts, value=0.0, engine=vector)
    for b in range(1, seqs):
        nisa.tensor_reduce(
            dst=starts[0:_P_MAX, b : b + 1],
            data=seq_all[0:_P_MAX, 0:b, 3],
            op=nl.add,
            axis=1,
        )
    if pad_longest:
        # The total stream rows of the pass and the index of a longest sequence (the largest index of the maximum r),
        # on every partition. The longest sequence has 5 or more groups when its r is 4 or more, so its group 0 is full.
        total = sb.tile((_P_MAX, 1), fp32)
        nisa.tensor_reduce(dst=total, data=seq_all[0:_P_MAX, 0:seqs, 3], op=nl.add, axis=1)
        r_max = sb.tile((_P_MAX, 1), fp32)
        nisa.tensor_reduce(dst=r_max, data=seq_all[0:_P_MAX, 0:seqs, 3], op=nl.maximum, axis=1)
        pad_ids = ids if ids is not None else _seq_ids(seqs, sb)
        hit_ids = sb.tile((_P_MAX, seqs), fp32)
        nisa.scalar_tensor_tensor(
            dst=hit_ids,
            data=seq_all[0:_P_MAX, 0:seqs, 3],
            op0=nl.equal,
            operand0=r_max,
            op1=nl.multiply,
            operand1=pad_ids,
        )
        longest = sb.tile((_P_MAX, 1), fp32)
        nisa.tensor_reduce(dst=longest, data=hit_ids, op=nl.maximum, axis=1)
    for tile_start in range(0, rows_total, _P_MAX):
        rows = min(_P_MAX, rows_total - tile_start)
        j_i = sb.tile((_P_MAX, 1), int32)
        nisa.iota(dst=j_i[0:rows], pattern=[[0, 1]], offset=tile_start, channel_multiplier=1)
        j_f = sb.tile((_P_MAX, 1), fp32)
        nisa.tensor_copy(dst=j_f[0:rows], src=j_i[0:rows], engine=vector)
        # Sequence = the number of stream starts past the first at or below j.
        seq_f = sb.tile((_P_MAX, 1), fp32)
        if ids is not None and seqs > 1:
            # The S - 1 compares (start <= j) into a tile of their own, then one reduce counts them.
            hits = sb.tile((_P_MAX, seqs - 1), fp32)
            nisa.tensor_scalar(
                dst=hits[0:rows], data=starts[0:rows, 1:seqs], op0=nl.less_equal, operand0=j_f[0:rows], engine=vector
            )
            nisa.tensor_reduce(dst=seq_f[0:rows], op=nl.add, data=hits[0:rows], axis=1)
        else:
            nisa.memset(dst=seq_f[0:rows], value=0.0, engine=vector)
            hit = sb.tile((_P_MAX, 1), fp32)
            for b in range(1, seqs):
                nisa.tensor_tensor(
                    dst=hit[0:rows],
                    data1=j_f[0:rows],
                    data2=starts[0:rows, b : b + 1],
                    op=nl.greater_equal,
                    engine=vector,
                )
                nisa.tensor_tensor(dst=seq_f[0:rows], data1=seq_f[0:rows], data2=hit[0:rows], op=nl.add, engine=vector)
        if pad_longest:
            # A row past the stream (j >= total) takes the longest sequence: seq = seq + pad x (longest - seq).
            pad = sb.tile((_P_MAX, 1), fp32)
            nisa.tensor_scalar(
                dst=pad[0:rows], data=j_f[0:rows], op0=nl.greater_equal, operand0=total[0:rows], engine=vector
            )
            move = sb.tile((_P_MAX, 1), fp32)
            nisa.tensor_tensor(
                dst=move[0:rows], data1=longest[0:rows], data2=seq_f[0:rows], op=nl.subtract, engine=vector
            )
            nisa.tensor_tensor(dst=move[0:rows], data1=move[0:rows], data2=pad[0:rows], op=nl.multiply, engine=vector)
            padded = sb.tile((_P_MAX, 1), fp32)
            nisa.tensor_tensor(dst=padded[0:rows], data1=seq_f[0:rows], data2=move[0:rows], op=nl.add, engine=vector)
            seq_f = padded
        g = _seq_value(seq_all, seq_f, 1, seqs, rows, sb, ids)
        p = _seq_value(seq_all, seq_f, 2, seqs, rows, sb, ids)
        if ids is not None:
            start_b = _seq_pick(ids, seq_f, starts[0:rows, 0:seqs], seqs, rows, sb)
        else:
            start_b = sb.tile((_P_MAX, 1), fp32)
            nisa.memset(dst=start_b[0:rows], value=0.0, engine=vector)
            pick = sb.tile((_P_MAX, 1), fp32)
            for b in range(seqs):
                nisa.tensor_scalar(
                    dst=pick[0:rows],
                    data=seq_f[0:rows],
                    op0=nl.equal,
                    operand0=float(b),
                    op1=nl.multiply,
                    operand1=starts[0:rows, b : b + 1],
                    engine=vector,
                )
                nisa.tensor_tensor(
                    dst=start_b[0:rows], data1=start_b[0:rows], data2=pick[0:rows], op=nl.add, engine=vector
                )
        # group = g + p - 5 - (j - start); rows past the stream get group 0 of their sequence (valid 0).
        group_f = sb.tile((_P_MAX, 1), fp32)
        nisa.scalar_tensor_tensor(
            dst=group_f[0:rows],
            data=g[0:rows],
            op0=nl.add,
            operand0=-float(CHUNK_ITEMS + 1),
            op1=nl.add,
            operand1=p[0:rows],
        )
        nisa.tensor_tensor(dst=group_f[0:rows], data1=group_f[0:rows], data2=start_b[0:rows], op=nl.add, engine=vector)
        nisa.tensor_tensor(dst=group_f[0:rows], data1=group_f[0:rows], data2=j_f[0:rows], op=nl.subtract, engine=vector)
        valid_f = sb.tile((_P_MAX, 1), fp32)
        nisa.tensor_scalar(dst=valid_f[0:rows], data=group_f[0:rows], op0=nl.greater_equal, operand0=0.0, engine=vector)
        nisa.tensor_scalar(dst=group_f[0:rows], data=group_f[0:rows], op0=nl.maximum, operand0=0.0, engine=vector)
        thr_f = sb.tile((_P_MAX, 1), fp32)
        nisa.memset(dst=thr_f[0:rows], value=float(_GROUP_STRIDE), engine=vector)
        _rows_store(
            group_f,
            seq_f,
            thr_f,
            valid_f,
            table,
            rest_hbm[tile_start : tile_start + rows],
            rows,
            groups,
            num_pages,
            tag,
            sb,
            f"rest{tile_start}",
            runs,
        )


def _mask_pattern(queries, sb):
    """``[64 + Q, 512]`` FP32: rows 64 + j hold ``4096 t + w - j`` for key column ``128 t + w`` (row w of page t)."""
    fp32 = nl.float32
    rows = _ROPE + queries
    pat = sb.tile((rows, _KEYS), fp32)
    pool_after(nisa.iota(dst=pat[0:rows, 0:_KEYS], pattern=[[4096, _GROUP_BLOCKS], [1, _P_MAX]], offset=0), "mask_pat")
    row_i = sb.tile((rows, 1), nl.int32)
    pool_after(nisa.iota(dst=row_i, pattern=[[0, 1]], offset=-_ROPE, channel_multiplier=1), "mask_row")
    row_f = sb.tile((rows, 1), fp32)
    nisa.tensor_copy(dst=row_f, src=row_i, engine=nisa.engine.vector)
    nisa.tensor_scalar(
        dst=pat[_ROPE:rows, 0:_KEYS],
        data=pat[_ROPE:rows, 0:_KEYS],
        op0=nl.subtract,
        operand0=row_f[_ROPE:rows, 0:1],
        engine=nisa.engine.vector,
    )
    return pat


def _prep_item(
    pg,
    kw_psum,
    rope_psum,
    key_words,
    rope_keys,
    value_words,
    pat,
    identity,
    queries,
    thr_col,
    clamp,
    clamp_split=False,
    after=(),
    rope_engine=nisa.engine.scalar,
):
    """The operands of one item from its 4 FP8 pages ``pg`` ``[128, 4, 576]`` (row w of page t on partition w), in
    the key order 128 t + w:

    - ``key_words`` ``[128, 512]`` FP32: word c of key column 128 t + w = latent features 4 c .. 4 c + 3 of the key
      (four 32-bit TensorE transposes, then one contiguous uint16 ScalarE copy);
    - ``rope_keys`` ``[98, 512]`` FP8: the rope features (FP8 transposes, a ScalarE byte copy) and, with ``thr_col``
      (``[64 + Q, 1]`` FP32 threshold of the item on rows 64 ..), the mask rows (-448 at or past it);
    - ``value_words`` ``[128, 512, 4]`` uint8: lane t of word (w, f) = feature f of row w of page t (one VectorE copy).

    ``clamp``: the pages are clamped to the FP8 bound first (stale bytes of masked rows, ``_sanitize_cache_rows``);
    ``clamp_split`` clamps each page alone (the same elementwise op on the same bytes), so the transposes of a page wait
    for its clamp only.
    ``value_words`` None leaves the value words to ``_value_words`` (the caller places it).
    ``rope_engine``: the engine of the rope byte copy (ScalarE by default, VectorE with ``fp8_pv_loop``
    ``rope_copy_vector``). The copy moves uint8 values with no conversion to another type, so either engine writes the
    same bytes."""
    fp32 = nl.float32
    fp8 = nl.float8_e4m3fn
    u8 = nl.uint8
    u16 = nl.uint16
    if clamp and clamp_split:
        for page in range(_GROUP_BLOCKS):
            _sanitize_cache_rows(pg[0:_P_MAX, page, 0:_CACHE_WIDTH])
    elif clamp:
        _sanitize_cache_rows(pg[0:_P_MAX, 0:_GROUP_BLOCKS, 0:_CACHE_WIDTH])
    for page in range(_GROUP_BLOCKS):
        transpose = nisa.nc_transpose(
            dst=kw_psum[0:_P_MAX, page * _P_MAX : (page + 1) * _P_MAX],
            data=pg.view(fp32)[0:_P_MAX, page, 0 : _KV_LORA // 4],
            engine=nisa.engine.tensor,
        )
        if after:
            _instruction(transpose).depends_on(*after, kind=nisa.dependency.schedule)
    nisa.tensor_copy(
        dst=key_words.view(u16)[0:_P_MAX, 0 : 2 * _KEYS],
        src=kw_psum.view(u16)[0:_P_MAX, 0 : 2 * _KEYS],
        engine=nisa.engine.scalar,
    )
    for page in range(_GROUP_BLOCKS):
        transpose = nisa.nc_matmul(
            dst=rope_psum[0:_ROPE, 2 * page * _P_MAX : 2 * (page + 1) * _P_MAX : 2],
            stationary=pg.view(fp8)[0:_P_MAX, page, _KV_LORA:_CACHE_WIDTH],
            moving=identity,
            is_transpose=True,
            accumulate=False,
        )
        if after:
            _instruction(transpose).depends_on(*after, kind=nisa.dependency.schedule)
    nisa.tensor_copy(
        dst=rope_keys.view(u8)[0:_ROPE, 0:_KEYS],
        src=rope_psum.view(u8).ap(pattern=[[4 * _KEYS, _ROPE], [2, _KEYS]], offset=0, dtype=u8),
        engine=rope_engine,
    )
    if thr_col is not None:
        rows = _ROPE + queries
        nisa.tensor_scalar(
            dst=rope_keys[_ROPE:rows, 0:_KEYS],
            data=pat[_ROPE:rows, 0:_KEYS],
            op0=nl.greater_equal,
            operand0=thr_col[_ROPE:rows, 0:1],
            op1=nl.multiply,
            operand1=MASK_FP8,
            engine=nisa.engine.vector,
        )
    if value_words is not None:
        _value_words(pg, value_words, 0, None)


VALUE_FORMS = (0, 4, 5)
"""``fp8_pv_loop`` ``value_form``: 0 one VectorE copy that walks the words in order (a 576-byte source stride, at the
prep); 4 one copy a page (a contiguous source, a 4-byte destination stride), placed in the loop after the quantize of a
sub-slot; 5 the copy of form 0 in 4 pieces of 128 features (words in order), placed as form 4."""
VALUE_PIECE_FEATURES = _KV_LORA // _GROUP_BLOCKS


def _value_words(pg, value_words, form, page):
    """Write the value words ``[128, 512, 4]`` uint8 of one item (lane t of word (w, f) = feature f of row w of page t)
    from its pages ``pg`` ``[128, 4, 576]``: byte moves only, so every form writes the same bytes. ``form`` 0 walks the
    words in order; with ``page``, ``form`` 4 writes only lane ``page`` of every word, ``form`` 5 only the words of
    features 128 ``page`` .. 128 ``page`` + 127."""
    u8 = nl.uint8
    row = _GROUP_BLOCKS * pg.shape[2]
    word = 4 * _KV_LORA
    if form == 0:
        dst = value_words
        src = pg.ap(pattern=[[row, _P_MAX], [1, _KV_LORA], [pg.shape[2], _GROUP_BLOCKS]], offset=0, dtype=u8)
    elif form == 5:
        f0 = page * VALUE_PIECE_FEATURES
        dst = value_words.ap(pattern=[[word, _P_MAX], [1, 4 * VALUE_PIECE_FEATURES]], offset=4 * f0, dtype=u8)
        src = pg.ap(
            pattern=[[row, _P_MAX], [1, VALUE_PIECE_FEATURES], [pg.shape[2], _GROUP_BLOCKS]], offset=f0, dtype=u8
        )
    else:
        dst = value_words.ap(pattern=[[word, _P_MAX], [4, _KV_LORA]], offset=page, dtype=u8)
        src = pg.ap(pattern=[[row, _P_MAX], [1, _KV_LORA]], offset=page * pg.shape[2], dtype=u8)
    nisa.tensor_copy(dst=dst, src=src, engine=nisa.engine.vector)


def _last_read(prep_at, value_form):
    """The sub-slot of an item's last read of its page buffer in a body's event order (``fp8_pv_loop``): its prep, or
    with ``value_form`` its last value-word piece (events before sub-slot 0 sit at sub-slot -1 at the latest)."""
    if not value_form:
        return prep_at
    at = prep_at + _GROUP_BLOCKS - 1
    return at if prep_at >= 0 else min(at, -1)


def _page_plan(items, groups, nb, carried, held, value_form, tail, buffers):
    """The page buffer and the load sub-slot of each item ``carried`` .. ``items`` - 1 of a body (``fp8_pv_loop``
    ``page_buffers`` 3). ``held``: buffer index -> the carried item that holds it at the body start. Each item tries a
    load one item ahead (2 ``groups`` sub-slots before its prep), then the load place of 2 buffers (``groups``
    sub-slots before), and takes the free buffer whose last read comes first; a buffer is free when its holder's last
    read comes strictly earlier in the event order (a load sorts before the reads of its own sub-slot). With ``tail``
    the last item does not take buffer 0, which the tail of the item before reloads for the next body. Returns
    {item: (buffer, sub-slot)}."""
    last = {i: _last_read(groups * k - groups * nb + 3, value_form) for i, k in held.items()}
    plan = {}
    for k in range(carried, items):
        prep_at = groups * k - groups * nb + 3
        choice = None
        for at in (prep_at - 2 * groups, prep_at - groups):
            free = [i for i in range(buffers) if i not in last or last[i] < at]
            if tail and k == items - 1:
                free = [i for i in free if i != 0]
            if free:
                choice = (min(free, key=lambda i: (last.get(i, float("-inf")), i)), at)
                break
        kernel_assert(choice is not None, f"fp8_pv_loop page_buffers {buffers}: no free page buffer for item {k}")
        plan[k] = choice
        last[choice[0]] = _last_read(prep_at, value_form)
    return plan


def fp8_pv_prior(
    kv_pool,
    table,
    position,
    rank_info,
    q_nat_all,
    q_rope_all,
    softmax,
    state_flat,
    seqs,
    queries,
    first,
    tag,
    sb,
    capped=True,
    clamp_pages=True,
    after_writes=(),
    lookahead=0,
    trips_ahead=False,
    chunk_ahead=False,
    qk_group=0,
    value_form=0,
    page_runs=0,
    stream="loops",
):
    """Attend the prior keys of one pass of ``seqs`` sequences x ``queries`` rows in the FP8 P@V form (module
    docstring) and add them into the p2 state ``state_flat`` ``[128, (G S + 1) x 513]``: ``fp8_pv_setup``, then one
    ``fp8_pv_loop``. ``capped``: the served score cap (a constant minimum of the shifted scores); False is the capless
    form. ``clamp_pages``, ``after_writes``, ``lookahead``, ``trips_ahead``, ``chunk_ahead``, ``qk_group``,
    ``value_form``, ``page_runs`` and ``stream`` as ``fp8_pv_loop`` (the setup takes ``page_runs`` too).
    """
    with sb.scope("fp8_pv"):
        setup = fp8_pv_setup(
            kv_pool,
            table,
            position,
            rank_info,
            q_nat_all,
            q_rope_all,
            softmax,
            seqs,
            queries,
            first,
            tag,
            sb,
            page_runs=page_runs,
        )
        with sb.scope("fp8_pv_loop"):
            fp8_pv_loop(
                kv_pool,
                setup,
                state_flat,
                seqs,
                queries,
                tag,
                sb,
                capped,
                clamp_pages,
                after_writes,
                lookahead=lookahead,
                trips_ahead=trips_ahead,
                chunk_ahead=chunk_ahead,
                qk_group=qk_group,
                value_form=value_form,
                page_runs=page_runs,
                stream=stream,
            )


def _fp8_pv_seq_rows(plan, seqs, tag):
    """The stream rows of each sequence of ``plan`` (``fp8_pv_items``) in HBM ``[S, rows_alloc, 8]``, padded to whole
    bodies of 32 rows for the 32-item loop's row view; return ``(rows_alloc, seq_rows)``."""
    int32 = nl.int32
    fields = ITEM_FIELDS
    rows_per_seq = plan["rest_hbm"].shape[0] // seqs
    rows_alloc = (rows_per_seq // 32 + 1) * 32
    seq_rows = nl.ndarray((seqs, rows_alloc, fields), dtype=int32, buffer=nl.hbm, name=f"{tag}_fp8pv_seq_rows")
    info = plan["seq_info"]
    for b in range(seqs):
        nisa.dma_copy(
            dst=seq_rows[b, 0:rows_per_seq, 0:fields],
            src=plan["rest_hbm"].ap(
                pattern=[[fields, rows_per_seq], [1, fields]],
                offset=0,
                scalar_offset=info[0:1, 8 * b : 8 * b + 1],
                indirect_dim=0,
            ),
            name=f"{tag}_fp8pv_seq_rows_{b}",
        )
    return rows_alloc, seq_rows


FRAME_ROUND = 4
"""``fp8_pv_setup``: (sequence, group) columns of one round of reference transposes (one PSUM bank)."""
_FRAME_STRIDE = 16
"""``_frame_rows_batched``: FP32 columns between the transpose windows of two columns (64 bytes: aligned TensorE reads)."""


def _frame_masks(sb):
    """``_frame_rows_batched``: the row masks ``[98, 1]`` FP32 of the reference rows, m_lo (row 96: +0, row 97: 1) and
    m_hi = 1 - m_lo."""
    vector = nisa.engine.vector
    sel_i = sb.tile((_ROWS, 1), nl.int32)
    pool_after(nisa.iota(dst=sel_i, pattern=[[0, 1]], offset=-FRAME_ROW, channel_multiplier=1), "frame_sel")
    m_lo = sb.tile((_ROWS, 1), nl.float32)
    nisa.tensor_copy(dst=m_lo, src=sel_i, engine=vector)
    m_hi = sb.tile((_ROWS, 1), nl.float32)
    nisa.tensor_scalar(dst=m_hi, data=m_lo, op0=nl.multiply, operand0=-1.0, op1=nl.add, operand1=1.0, engine=vector)
    return m_lo, m_hi


def _frame_rows_batched(stage, qpe, fr_ps, m_lo, m_hi, seqs, groups, rope_rows_q, sb, refs=None):
    """``fp8_pv_setup``: write the stage rows (rope + bias rows) and the reference rows 96 / 97 of every
    (sequence, group) column of ``qpe``, batched over the columns.

    - The stage rows of all columns: one strided VectorE copy (column c of the stage is c x width + 512 in each row).
    - The reference / scale of every column: two VectorE products into one tile ``V`` ``[128, 16 (C - 1) + 98]``,
      column 96 + j + 16 c (j = 0, 1). Column c's transpose reads the window ``V[:, 16 c : 16 c + 98]``: its columns 96
      and 97 hold the column's values, the others only reach PSUM rows 0 .. 95, which nothing reads.
    - Rounds of ``FRAME_ROUND`` columns: the TensorE transposes of the round into the 4 slices of the one PSUM bank, then
      4 VectorE operations on the round's rows 96 / 97 (hi, lo, hi x m_hi, lo x m_lo + that).

    Each output element takes the same operations on the same values as a column-at-a-time form (one transpose and
    those 4 operations a column), so the bits are the same; the setup runs ~16 VectorE instructions in place of 7 a
    column, with no VectorE <-> TensorE round trip a column. ``refs`` (``fp8_pv_setup`` ``stage_free``; ``stage`` None):
    the references ``[128, S, G]`` FP32 of the columns on their partitions, and the rope + bias rows already in
    ``qpe`` (``_rope_rows_direct``): the same values, so the same bits."""
    fp32 = nl.float32
    bf16 = nl.bfloat16
    vector = nisa.engine.vector
    P, R = _P_MAX, _ROWS
    width = _P2_STAGE_WIDTH
    cols = seqs * groups
    kernel_assert(
        width % 2 == 0, f"the batched reference rows read FP32 words of the stage rows: width {width} must be even"
    )
    if refs is None:
        stage_row = stage.shape[1] * stage.shape[2]
        nisa.tensor_copy(
            dst=qpe.ap(pattern=[[qpe.shape[1], rope_rows_q], [P, cols], [1, P]], offset=0),
            src=stage.ap(pattern=[[stage_row, rope_rows_q], [width, cols], [1, P]], offset=_KV_LORA),
            engine=vector,
        )
        ref_words = stage.ap(pattern=[[stage_row // 2, P], [width // 2, cols]], offset=_P2_STAGE_REF_WORD, dtype=fp32)
    else:
        ref_words = refs.ap(pattern=[[cols, P], [1, cols]], offset=0)
    span = _FRAME_STRIDE * (cols - 1) + R
    with sb.scope("fp8_pv_frame"):
        v_t = sb.tile((P, span), fp32)
        nisa.memset(dst=v_t, value=0.0, engine=vector)
        for j in range(2):
            nisa.tensor_scalar(
                dst=v_t.ap(pattern=[[span, P], [_FRAME_STRIDE, cols]], offset=FRAME_ROW + j),
                data=ref_words,
                op0=nl.multiply,
                operand0=1.0 / _SOFTMAX_SCALE,
                engine=vector,
            )
        fr_hi = sb.tile((R, FRAME_ROUND * P), bf16)
        fr_lo = sb.tile((R, FRAME_ROUND * P), fp32)
        fr_t = sb.tile((R, FRAME_ROUND * P), fp32)
        for c0 in range(0, cols, FRAME_ROUND):
            n = min(FRAME_ROUND, cols - c0)
            for i in range(n):
                c = c0 + i
                nisa.nc_transpose(
                    dst=fr_ps[0:R, i * P : (i + 1) * P],
                    data=v_t[0:P, _FRAME_STRIDE * c : _FRAME_STRIDE * c + R],
                    engine=nisa.engine.tensor,
                )
            w = n * P
            fr = fr_ps[FRAME_ROW:R, 0:w]
            # Row 96 = hi = BF16(ref / scale); row 97 = lo = BF16(ref / scale - hi).
            nisa.tensor_copy(dst=fr_hi[FRAME_ROW:R, 0:w], src=fr, engine=vector)
            nisa.tensor_tensor(
                dst=fr_lo[FRAME_ROW:R, 0:w], data1=fr, data2=fr_hi[FRAME_ROW:R, 0:w], op=nl.subtract, engine=vector
            )
            nisa.tensor_scalar(
                dst=fr_t[FRAME_ROW:R, 0:w],
                data=fr_hi[FRAME_ROW:R, 0:w],
                op0=nl.multiply,
                operand0=m_hi[FRAME_ROW:R, 0:1],
                engine=vector,
            )
            nisa.scalar_tensor_tensor(
                dst=qpe[FRAME_ROW:R, c0 * P : c0 * P + w],
                data=fr_lo[FRAME_ROW:R, 0:w],
                op0=nl.multiply,
                operand0=m_lo[FRAME_ROW:R, 0:1],
                op1=nl.add,
                operand1=fr_t[FRAME_ROW:R, 0:w],
            )


def _frame_ref_rows(qpe, fr_ps, ref_words, cols, sb, keep_frame=False):
    """``fp8_pv_setup``: the reference rows 96 / 97 of ``cols`` (sequence, group) columns of ``qpe`` from the references
    ``ref_words`` ``[128, cols]`` FP32 of the columns on their partitions, with the arithmetic of the batched form
    (``_frame_rows_batched``) done before the transposes.

    ``_frame_rows_batched`` transposes f = reference / scale (FP32) of each column to PSUM rows 96 / 97 and then runs 4
    VectorE operations there a round of ``FRAME_ROUND`` columns, each on 2 partitions x 512 columns: hi = BF16(f),
    lo = f - hi, t = hi x m_hi, and BF16(lo x m_lo + t) into ``qpe`` (row 96 + j: m_hi = 1 - j, m_lo = j). Here the
    same 4 operations, in the same order and with the same operand kinds (per-partition FP32 scalars m_hi / m_lo of the
    same values), run once on ``[128, cols]`` before any transpose, and write the BF16 result of row 96 + j into the
    low half of word ``16 c + 96 + j`` of a zero window tile. The TensorE transposes of the same 98-word FP32 windows
    (32-bit words: the bits move unchanged) bring the words to PSUM rows 96 / 97, and one ScalarE copy a round moves
    their low halves to ``qpe`` through a UINT16 view (an integer copy: every bit pattern stays). The transposes take
    the same values in, so every element gets the bits of ``_frame_rows_batched``, non-finite and zero values included:
    ~9 small VectorE operations in place of 14 (12 of them 512 wide), and the PSUM rows read by ScalarE. ``keep_frame``
    (``fp8_pv_setup`` ``pages_first``): the tiles take the caller's frame, not a frame of their own, so the loop's page
    tiles after the setup do not take their bytes (and their first loads do not wait for the frame rows)."""
    fp32 = nl.float32
    bf16 = nl.bfloat16
    vector = nisa.engine.vector
    P, R = _P_MAX, _ROWS
    span = _FRAME_STRIDE * (cols - 1) + R
    with nullcontext() if keep_frame else sb.scope("fp8_pv_frame"):
        # m[:, j] = m_hi of row 96 + j (1, +0) and m[:, 1 - j] its m_lo (+0, 1): the values of the setup's m_hi / m_lo.
        m = sb.tile((P, 2), fp32)
        nisa.memset(dst=m, value=0.0, engine=vector)
        nisa.memset(dst=m[0:P, 0:1], value=1.0, engine=vector)
        f = sb.tile((P, cols), fp32)
        nisa.tensor_scalar(dst=f, data=ref_words, op0=nl.multiply, operand0=1.0 / _SOFTMAX_SCALE, engine=vector)
        hi = sb.tile((P, cols), bf16)
        nisa.tensor_copy(dst=hi, src=f, engine=vector)
        lo = sb.tile((P, cols), fp32)
        nisa.tensor_tensor(dst=lo, data1=f, data2=hi, op=nl.subtract, engine=vector)
        # The BF16 halves of the window words: word w of a row is halves 2 w (low) and 2 w + 1. The products below run on
        # VectorE, as in _frame_rows_batched (scalar_tensor_tensor is a VectorE instruction and takes no engine).
        halves = sb.tile((P, 2 * span), bf16)
        nisa.memset(dst=halves, value=0.0, engine=vector)
        for j in range(2):
            t = sb.tile((P, cols), fp32)
            nisa.tensor_scalar(dst=t, data=hi, op0=nl.multiply, operand0=m[0:P, j : j + 1], engine=vector)
            nisa.scalar_tensor_tensor(
                dst=halves.ap(pattern=[[2 * span, P], [2 * _FRAME_STRIDE, cols]], offset=2 * (FRAME_ROW + j)),
                data=lo,
                op0=nl.multiply,
                operand0=m[0:P, 1 - j : 2 - j],
                op1=nl.add,
                operand1=t,
            )
        words = halves.view(fp32)
        ps_u16 = fr_ps.view(nl.uint16)
        qpe_u16 = qpe.view(nl.uint16)
        for c0 in range(0, cols, FRAME_ROUND):
            n = min(FRAME_ROUND, cols - c0)
            for i in range(n):
                c = c0 + i
                nisa.nc_transpose(
                    dst=fr_ps[0:R, i * P : (i + 1) * P],
                    data=words[0:P, _FRAME_STRIDE * c : _FRAME_STRIDE * c + R],
                    engine=nisa.engine.tensor,
                )
            w = n * P
            nisa.tensor_copy(
                dst=qpe_u16[FRAME_ROW:R, c0 * P : c0 * P + w],
                src=ps_u16[FRAME_ROW:R, 0 : 2 * w : 2],
                engine=nisa.engine.scalar,
            )


def _frame_rows_split(qpe, fr_ps, seqs, groups, refs, sb, keep_frame=False):
    """``fp8_pv_setup`` ``stage_free``: the reference rows 96 / 97 of every (sequence, group) column of ``qpe`` from
    ``refs`` ``[128, S, G]`` FP32 (``_frame_ref_rows``; the bits of ``_frame_rows_batched`` on the same references).
    ``keep_frame``: ``_frame_ref_rows`` ``keep_frame``."""
    cols = seqs * groups
    _frame_ref_rows(qpe, fr_ps, refs.ap(pattern=[[cols, _P_MAX], [1, cols]], offset=0), cols, sb, keep_frame=keep_frame)


def _query_words_direct(
    q_nat_all, softmax, seqs, queries, q_words, q_scales, refs, tag, sb, after=(), sched_after=(), refs_engine=None
):
    """``fp8_pv_setup`` ``stage_free``: the MX query words and scales and the references of a pass straight into the
    tiles that read them, with no p2 stage tile.

    The ops and their order are those of ``_stage_query_rows_p2`` "block32" on the runs of ``pack_p2_runs``: an exact
    interleaved BF16 copy of each run (a ring of 2), and one ``nisa.quantize_mx`` a group and sequence into
    ``q_words`` ``[128, S, G x 128]`` FP32 (``float8_e4m3fn_x4`` words) and ``q_scales`` ``[128, S, G x 128]`` UINT8,
    which start at zero as the stage does (``q_scales``: the bytes that no quantize writes). The negative reference of
    each column moves to its partition in ``refs`` ``[128, S, G]`` FP32 (SBUF DMAs, as the stage's reference words);
    ``after``: the first of those DMAs waits for each of them (the state stores of the pass before, whose sources these
    tiles may reuse) and the others for it. The score cap word of the stage is not written: the FP8 P@V loop reads no cap column. The same
    bytes reach the same instructions, so the bits are the same. ``q_words`` None (the gather words of ``q_mx_gather``):
    the reference DMAs only."""
    bf16 = nl.bfloat16
    vector = nisa.engine.vector
    P = _P_MAX
    words = q_words is not None
    if words:
        nisa.memset(dst=q_scales, value=0, engine=vector)
        q_il = [sb.tile((P, seqs, P, _MX_PACK), bf16) for _ in range(2)]
    gate = None
    for group, row, head, column, count in pack_p2_runs(queries):
        rows = slice(row * seqs, (row + 1) * seqs)
        dma = nisa.dma_copy(
            dst=refs[column : column + count, 0:seqs, group],
            src=softmax.negative_reference[head : head + count, rows],
            dge_mode=nisa.dge_mode.hwdge,
            # ``refs_engine`` (``tkg_mla_block`` ``fp8_pv_refs_act``): the hardware DGE queue of that engine in place of SP.
            engine=nisa.engine.sync if refs_engine is None else refs_engine,
            name=f"{tag}_fp8pv_ref_{group}_{column}",
        )
        if after:
            # The first reference DMA waits for every store; the others wait for it (the same order, fewer edges).
            _instruction(dma).depends_on(*(after if gate is None else (gate,)), kind=nisa.dependency.runtime)
            gate = dma if gate is None else gate
        if sched_after:
            # ``sched_after`` (``tkg_mla_block`` ``fp8_pv_refs_after_q``): the reference DMAs follow these on the SP queue (a
            # schedule edge, no wait), so the pass's MX query loads go first.
            _instruction(dma).depends_on(*sched_after, kind=nisa.dependency.schedule)
        if not words:
            continue
        interleaved = q_il[group % 2]
        nisa.tensor_copy(
            dst=interleaved[0:P, 0:seqs, column : column + count, 0:_KV_LORA_TILES],
            src=q_nat_all[0:P, rows, 0:_KV_LORA_TILES, head : head + count].permute((0, 1, 3, 2)),
            engine=vector,
        )
        if column + count == P:
            for token in range(seqs):
                nisa.quantize_mx(
                    dst=q_words[0:P, token : token + 1, group * P : (group + 1) * P].view(nl.float8_e4m3fn_x4),
                    src=interleaved[0:P, token : token + 1, 0:P, 0:_MX_PACK],
                    dst_scale=q_scales[0:P, token : token + 1, group * P : (group + 1) * P],
                )


def _rope_rows_direct(q_rope_all, qpe, seqs, queries, pat):
    """``fp8_pv_setup`` ``stage_free``: the rope query rows 0 .. 63 of every (sequence, row, head) column straight from
    ``q_rope_all`` (row j S + b) into ``qpe`` (zero; column b G 128 + 96 j + h) in one VectorE copy, and 2^100 on query
    bias row 64 + j of each column of row j in one VectorE op from the mask pattern ``pat`` (``_mask_pattern``: row
    64 + i, column j < 128 holds j - i, so no GpSimd iota of its own): the values the stage rows carry (the stage's zero
    + the bias, exact), so the same BF16 bits."""
    vector = nisa.engine.vector
    heads = q_rope_all.shape[2]
    kernel_assert(
        heads * queries == pack_p2_groups(queries) * _P_MAX and q_rope_all.shape[1] >= queries * seqs,
        f"stage_free rope rows need [128, >= Q S, Q x heads / Q] rope queries, got {q_rope_all.shape}",
    )
    bias_end = _BIAS_ROW + queries
    qpe4 = qpe.reshape((qpe.shape[0], seqs, queries, heads))
    nisa.tensor_copy(
        dst=qpe4[0:_ROPE, 0:seqs, 0:queries, 0:heads],
        src=q_rope_all.ap(
            pattern=[[q_rope_all.shape[1] * heads, _ROPE], [heads, seqs], [seqs * heads, queries], [1, heads]], offset=0
        ),
        engine=vector,
    )
    kernel_assert(_BIAS_ROW == _ROPE and queries <= _P_MAX, "the bias rows read the mask pattern's rows 64 + j")
    nisa.tensor_scalar(
        dst=qpe4[_BIAS_ROW:bias_end, 0:seqs, 0:queries, 0:heads],
        data=pat.ap(pattern=[[_KEYS, queries], [0, seqs], [1, queries], [0, heads]], offset=_BIAS_ROW * _KEYS),
        op0=nl.equal,
        operand0=0.0,
        op1=nl.multiply,
        operand1=_MASK_COLUMN_SCALE,
        engine=vector,
    )


def fp8_pv_setup(
    kv_pool,
    table,
    position,
    rank_info,
    q_nat_all,
    q_rope_all,
    softmax,
    seqs,
    queries,
    first,
    tag,
    sb,
    page_runs=0,
    step=None,
    stage_free=False,
    after=(),
    query_words=None,
    plan_first=False,
    pages_first=False,
    tables=None,
    refs_sched_after=(),
    refs_engine=None,
    null_last=False,
    plan_reduce=False,
    pad_longest=False,
):
    """Build what every loop of a pass reads, once (so a capped redo of the pass runs only its loop): the p2 stage, the
    item plan and the stream rows in HBM, the chunk entries and mask thresholds in SBUF, the rope query rows with the
    reference rows, the mask pattern and the constants. Every DMA of a kernel input is here. ``plan_first``: the item
    plan's input loads and HBM round trip go on the ACT queue, not behind the query loads of the in-order SP queue (the
    same plan).

    ``kv_pool`` ``[NP, 128 x 576]`` FP8 kernel pages, ``table`` ``[S, local_blocks]`` INT32 HBM (the pass rows of the
    block table), ``position`` ``[B, 1]`` and ``rank_info`` as ``_attend_prior_packed``, ``first`` the first sequence of
    the pass. ``page_runs`` (0 or 2, ``fp8_pv_loop`` ``page_runs``): the item rows of the run form (``fp8_pv_items``).
    ``q_nat_all`` holds the natural latent query tiles ``[128, Q S, 4, 96]`` BF16 of the pass rows
    (``tkg_mla_block._mx_natural_query_tiles``), from which the stage takes the MXFP8 query words in scale blocks of 32
    (``_stage_query_rows_p2`` ``mx_qk`` "block32"), and ``softmax`` the fixed reference of each column.
    The rope query rows of all (sequence, group) columns are written in rounds of ``FRAME_ROUND`` columns
    (``_frame_rows_batched``).
    ``step`` (the ``mla_pass_plans`` step store of this pass with ``step_items``, a dict that every MLA layer of the step
    shares; None builds the plan in every layer): the first MLA layer of the step builds the item rows, the stream rows
    and the run scale and keeps them in the store (the seq info also in HBM, the run scale in the store's persistent
    tile); every later layer reads them and skips the item plan (its DMA round trips and ~140 VectorE instructions a
    pass). Every MLA layer of a step reads the same positions and table, so the rows are the same bytes and the bits
    are the same.
    ``stage_free`` (opt-in; False keeps the program): no p2 stage tile (``[128, S + 1, G x 680]`` BF16, ~24 KB a
    partition, its zero fill, TI copies and null row, none of which the loop reads): the MX query words and scales go
    straight into ``q_words`` / ``q_scales`` and the references into a ``[128, S, G]`` tile
    (``_query_words_direct``; its reference DMAs wait for ``after``), the rope + bias rows straight into ``qpe``
    (``_rope_rows_direct``). The same values reach the same instructions, so the bits are the same.

    ``query_words`` (``tkg_mla_block`` ``q_mx_gather``): the MX query words ``[128, S G 128]`` FP32 and E8M0 scales (u8) of
    the pass, loaded from the MX query gather (column (b G + g) 128 + c: column c of group g of sequence b); the loops
    read them in place of the stage's words and scales. ``q_nat_all`` is then None, and the stage takes no latent query
    (``_stage_query_rows_p2`` ``latent`` False). The gather words take precedence over ``stage_free``: with both,
    the stage-free setup builds only the references and the rope + bias rows (``_query_words_direct`` with no words
    tiles, ``_rope_rows_direct``), and the loops read the gather words. ``pages_first`` (``tkg_mla_block``
    ``fp8_pv_pages_first``; with ``stage_free``): the chunk entries and thresholds land in heap tiles (``setup["heap"]``,
    which the caller releases after the loops) and the frame rows' tiles stay in the setup's frame, so neither the chunk
    loads nor the loop's first page loads wait for readers of re-used bytes (the own-key state, the frame rows).
    ``refs_sched_after`` (``tkg_mla_block`` ``fp8_pv_refs_after_q``, with ``stage_free``): instructions that the reference
    DMAs follow in the schedule (no wait): the pass's MX query loads, so they take the SP queue first. ``refs_engine``
    (``tkg_mla_block`` ``fp8_pv_refs_act``, with ``stage_free``): the reference DMAs go on that engine's hardware DGE queue
    (ScalarE) in place of SP, so they leave the SP queue to the query loads and the loop's register loads.
    ``tables`` (``tkg_mla_block`` ``fp8_pv_tables_ahead``; ``fp8_pv_tables_ahead`` of the pass before): the chunk entries and
    thresholds of this pass, already loaded into the caller's heap tiles from the same step store, so the setup issues no
    chunk load and keeps no heap tile of its own; the same bytes, so the same bits.
    ``plan_reduce`` (``tkg_mla_block`` ``fp8_pv_plan_reduce``; False keeps the program): the item plan sums over the S
    sequences with one reduce each, not with a chain of adds after a memset (``fp8_pv_items`` ``plan_reduce``; the same
    item rows, so the same bits). With ``step`` it acts where the plan is built (the first MLA layer of the step).
    ``pad_longest`` (``tkg_mla_block`` ``fp8_pv_batched``; False keeps the program): the stream rows past the stream of the
    pass hold group 0 of the longest sequence (``fp8_pv_items`` ``pad_longest``), for the "batched" stream.
    """
    fp32 = nl.float32
    bf16 = nl.bfloat16
    int32 = nl.int32
    u8 = nl.uint8
    fp8 = nl.float8_e4m3fn
    vector = nisa.engine.vector
    P, K, R = _P_MAX, _KEYS, _ROWS
    groups = pack_p2_groups(queries)
    nb = CHUNK_ITEMS
    fields = ITEM_FIELDS
    rope_rows_q = _ROPE + queries
    p2_columns = groups * seqs
    kernel_assert(not after or stage_free, "fp8_pv_setup after orders the stage-free reference DMAs: set stage_free")
    kernel_assert(not pages_first or stage_free, "fp8_pv_setup pages_first is a stage-free setup form: set stage_free")
    if query_words is not None:
        cols_q = seqs * groups * P
        kernel_assert(
            all(tuple(t.shape) == (P, cols_q) for t in query_words),
            f"fp8_pv query_words must be [{P}, {cols_q}] words and scales",
        )
    stage = q_words = q_scales = refs = None
    if stage_free:
        if query_words is None:
            q_words = sb.tile((_P_MAX, seqs, groups * _P_MAX), fp32)
            q_scales = sb.tile((_P_MAX, seqs, groups * _P_MAX), u8)
        refs = sb.tile((_P_MAX, seqs, groups), fp32)
        _query_words_direct(
            q_nat_all, softmax, seqs, queries, q_words, q_scales, refs, tag, sb, after, refs_sched_after, refs_engine
        )
    else:
        ti_u = sb.tile((_P_MAX, p2_columns + 1, _TI_COLS), nl.uint16)
        _pack_ti(ti_u, p2_columns, sb)
        stage = sb.tile((_P_MAX, seqs + 1, groups * _P2_STAGE_WIDTH), bf16)
        _stage_query_rows_p2(
            None,
            ti_u,
            q_nat_all,
            q_rope_all,
            softmax,
            seqs,
            queries,
            tag,
            sb,
            stage_sb=stage,
            mx_qk="block32",
            hbm_store=False,
            **({} if query_words is None else {"latent": False}),
        )
    form = (
        (seqs, queries, table.shape[1], kv_pool.shape[0], page_runs, first)
        + ((True,) if null_last else ())
        + (("pad_longest",) if pad_longest else ())
    )
    store = None if step is None else step.setdefault("fp8_pv", {})
    if store is not None and "chunk_hbm" in store:
        # A later MLA layer of the step: the item rows, stream rows and run scale of the first one.
        kernel_assert(
            store["form"] == form,
            f"fp8_pv step items of {store['form']} (sequences, Q, blocks, pages, page runs, first), this pass {form}",
        )
        seq_info = sb.tile((1, 8 * seqs), int32)
        nisa.dma_copy(dst=seq_info, src=store["info_hbm"], name=f"{tag}_fp8pv_step_info")
        plan = {"chunk_hbm": store["chunk_hbm"], "rest_hbm": store["rest_hbm"], "seq_info": seq_info}
        if page_runs:
            plan["run_scale"] = step["run_scale"]
        rows_alloc, seq_rows = store["rows_alloc"], store["seq_rows"]
    else:
        run_scale_dst = step["run_scale"] if step is not None and page_runs else None
        plan = fp8_pv_items(
            position,
            rank_info,
            table,
            first,
            seqs,
            queries,
            kv_pool.shape[0],
            tag,
            sb,
            page_runs,
            run_scale_dst,
            **({"dma_engine": nisa.engine.scalar} if plan_first else {}),
            **({"null_last": True} if null_last else {}),
            **({"plan_reduce": True} if plan_reduce else {}),
            **({"pad_longest": True} if pad_longest else {}),
        )
        rows_alloc, seq_rows = _fp8_pv_seq_rows(plan, seqs, tag)
        if store is not None:
            info_hbm = nl.ndarray((1, 8 * seqs), dtype=int32, buffer=nl.hbm, name=f"{step['tag']}_fp8pv_info")
            nisa.dma_copy(dst=info_hbm, src=plan["seq_info"], name=f"{tag}_fp8pv_step_info_store")
            store.update(
                form=form,
                chunk_hbm=plan["chunk_hbm"],
                rest_hbm=plan["rest_hbm"],
                info_hbm=info_hbm,
                rows_alloc=rows_alloc,
                seq_rows=seq_rows,
            )
    # The chunk items of every sequence (entries of sequence b at 32 b ..) and their mask thresholds (column 4 b + k).
    if tables is not None:
        # ``tables``: the pass before loaded them from this pass's step store (``fp8_pv_tables_ahead``).
        kernel_assert(
            store is not None and tables["chunk_hbm"] is store.get("chunk_hbm") and tables["form"] == form,
            "fp8_pv_setup tables are the chunk tables of this pass's step store (fp8_pv_tables_ahead)",
        )
        chunk_ents, thr_i = tables["chunk_ents"], tables["thr_i"]
    else:
        chunk_ents = sb.tile((1, seqs * nb * fields), int32, keep=pages_first)
        nisa.dma_copy(
            dst=chunk_ents,
            src=plan["chunk_hbm"].reshape((1, nb * seqs * fields)),
            name=f"{tag}_fp8pv_chunk_ents",
        )
        thr_i = sb.tile((rope_rows_q, seqs * nb), int32, keep=pages_first)
        nisa.dma_copy(
            dst=thr_i,
            src=plan["chunk_hbm"].ap(pattern=[[0, rope_rows_q], [fields, seqs * nb]], offset=5),
            name=f"{tag}_fp8pv_chunk_thr",
        )
    thr_all = sb.tile((rope_rows_q, seqs * nb), fp32)
    nisa.tensor_copy(dst=thr_all, src=thr_i, engine=vector)

    unit = sb.tile((P, K), u8)
    nisa.memset(dst=unit, value=_MX_UNIT_SCALE, engine=vector)
    identity = sb.tile((P, P), fp8)
    nl.shared_identity_matrix(P, dtype=fp8, dst=identity)
    # Two e4m3 ones columns (0x38 = 1.0) as one x4 word each: the moving data of the weight sums.
    ones_b = sb.tile((P, 8), u8)
    nisa.memset(dst=ones_b, value=0x38, engine=vector)
    zero = sb.tile((P, 16), fp32)
    nisa.memset(dst=zero, value=0.0, engine=vector)

    # The rope query of every (sequence, group) in 98 contraction rows: the stage rows (rope + bias rows), zeros, and
    # the reference / scale of each column at rows 96 (BF16 hi) and 97 (BF16 lo), from a TensorE transpose of the
    # stage's reference column.
    fr_ps = nl.ndarray((P, K), dtype=fp32, buffer=nl.psum)
    cols_all = seqs * groups * P
    qpe = sb.tile((R, cols_all), bf16)
    nisa.memset(dst=qpe, value=0.0, engine=vector)
    if stage_free:
        # The mask pattern first, for the bias rows of ``_rope_rows_direct`` (and the loop).
        pat = _mask_pattern(queries, sb)
        _rope_rows_direct(q_rope_all, qpe, seqs, queries, pat)
        _frame_rows_split(qpe, fr_ps, seqs, groups, refs, sb, keep_frame=pages_first)
    else:
        pat = None
        m_lo, m_hi = _frame_masks(sb)
        _frame_rows_batched(stage, qpe, fr_ps, m_lo, m_hi, seqs, groups, rope_rows_q, sb)
    return {
        "stage": stage,
        "q_words": q_words,
        "q_scales": q_scales,
        "plan": plan,
        "seq_rows": seq_rows,
        "rows_alloc": rows_alloc,
        "chunk_ents": chunk_ents,
        "thr": thr_all,
        "heap": [chunk_ents, thr_i] if pages_first and tables is None else [],
        "unit": unit,
        "identity": identity,
        "ones": ones_b,
        "zero": zero,
        "qpe": qpe,
        "pat": pat if pat is not None else _mask_pattern(queries, sb),
        "page_runs": page_runs,
        "words": query_words,
    }


def fp8_pv_tables_ahead(step, seqs, queries, tag, sb):
    """``tkg_mla_block`` ``fp8_pv_tables_ahead``: the chunk entries and mask thresholds of a later pass (the ``fp8_pv_setup``
    ``step`` store of that pass, which the first MLA layer of the step filled) in heap tiles, loaded now on the SP
    hardware DGE queue, and two heap page buffers for the pages of chunk items 0 and 1 of its first sequence (loaded by
    this pass's last chunk, ``fp8_pv_loop`` ``next_chunk``). The later pass's setup then takes the tables
    (``fp8_pv_setup`` ``tables``) and its first chunk the buffers (``fp8_pv_loop`` ``first_carry``), so neither waits
    behind this pass's state stores on the in-order SP queue. The same bytes, so the same bits. Return None when the
    store is not filled yet (the first MLA layer of the step: the later pass builds its own tables); else the tiles and
    ``heap`` (all four, in allocation order), which the caller releases after the later pass's loops."""
    store = None if step is None else step.get("fp8_pv")
    if not store or "chunk_hbm" not in store:
        return None
    kernel_assert(
        store["form"][0] == seqs and store["form"][1] == queries,
        f"fp8_pv_tables_ahead: the later pass has the form {store['form']}, this pass {seqs} sequences of Q {queries}",
    )
    int32, fp8 = nl.int32, nl.float8_e4m3fn
    nb, fields, rope_rows_q = CHUNK_ITEMS, ITEM_FIELDS, _ROPE + queries
    queue = {"dge_mode": nisa.dge_mode.hwdge, "engine": nisa.engine.sync}
    chunk_ents = sb.tile((1, seqs * nb * fields), int32, keep=True)
    nisa.dma_copy(
        dst=chunk_ents,
        src=store["chunk_hbm"].reshape((1, nb * seqs * fields)),
        name=f"{tag}_fp8pv_ahead_chunk_ents",
        **queue,
    )
    thr_i = sb.tile((rope_rows_q, seqs * nb), int32, keep=True)
    nisa.dma_copy(
        dst=thr_i,
        src=store["chunk_hbm"].ap(pattern=[[0, rope_rows_q], [fields, seqs * nb]], offset=5),
        name=f"{tag}_fp8pv_ahead_chunk_thr",
        **queue,
    )
    bufs = [sb.tile((_P_MAX, _GROUP_BLOCKS, _CACHE_WIDTH), fp8, keep=True) for _ in range(2)]
    return {
        "chunk_ents": chunk_ents,
        "thr_i": thr_i,
        "bufs": bufs,
        "chunk_hbm": store["chunk_hbm"],
        "form": store["form"],
        "heap": [chunk_ents, thr_i, *bufs],
    }


def fp8_pv_loop(
    kv_pool,
    setup,
    state_flat,
    seqs,
    queries,
    tag,
    sb,
    capped,
    clamp_pages,
    after_writes,
    lookahead=0,
    trips_ahead=False,
    chunk_ahead=False,
    qk_group=0,
    value_form=0,
    page_runs=0,
    stream="loops",
    clamp_split=False,
    first_carry=(),
    next_chunk=None,
    null_skip=False,
    prep_late=False,
    page_buffers=2,
    chunks_first=False,
    cap_pool=False,
    rope_copy_vector=False,
):
    """The key loop of one pass (module docstring) on its ``fp8_pv_setup``; adds into ``state_flat``.

    Each sequence runs its masked first chunk, then its stream in loops of 32 (``nl.fori_loop``), 16, 8 and 4 items.
    The sub-slots of a body run group-major (the 4 items of a block for group 0, then group 1, ...) in a 3-stage
    pipeline: the scores of sub-slot s, the exp and quantize of s - 1, and P@V of s - 2. The pages of item k load 4
    sub-slots of a group before its prep, and its prep runs 3 sub-slots before its first scores. ``capped``: the served
    score cap; ``clamp_pages``: the chunk clamps its pages (``_sanitize_cache_rows``); ``after_writes``: every chunk page
    load waits for each of them (the owner writes of the call's current rows, and in a redo its entry barrier). The
    loop reads no kernel input but the pages (dynamic DMAs), so it can run in the then-region of an IF.

    ``lookahead`` (opt-in, 2 or 4; 0 keeps the program): every stream loop of a sequence runs 0 or 1 trips (at most 64
    stream rows), so the item rows of each loop of the sequence load once, before its chunk, on the SP HWDGE queue (not
    one Pool SWDGE load at each body start). The tail of the chunk and of each stream body but the last loads the
    pages of the first ``lookahead`` items of the next stream body: the rows right after its own, since the bodies of
    a sequence walk its stream rows in order whichever of them run. A body takes those items as loaded and loads the
    rest. The pages land in the page queue's idle window before the body's end, so the next body starts its prep
    right after the fence. Items 0 and 1 ride the page buffers, items 2 and 3 two carry buffers. The same pages reach
    the same instructions, so the bits are the same. ``trips_ahead`` (opt-in) loads the four trip counts of a sequence
    before its chunk, so the block between two loops holds only the zero-trip branch (no register load, so the
    compiler leaves it without a fence); the same trips, the same bits. ``chunk_ahead`` (opt-in) loads the pages of the
    first 2 items of the next sequence's chunk at the end of a chunk's preps, into 2 buffers of their own that the
    stream between does not touch (the chunk rows are static), so that chunk starts its prep right after the boundary;
    the same pages (after the same owner writes), the same bits. ``stream`` (with ``lookahead``): "loops" keeps the
    0/1-trip ``fori_loop``s; "ifs" runs each stream body as a forward then-only IF on its trip register, loaded before
    the chunk (no induction register, no back edge; the same bodies, the same bits); "ifs_grouped" nests those IFs in
    an IF on any stream body and, inside it, an IF on any body but the last, so the zero-trip bodies of a short
    context (all of them, or the prefix ahead of the last) take one forward branch, not one each (the same bits); "straight" emits every stream body
    unconditionally, with no boundary at all: valid only when every sequence runs all of its stream loops (r = 60 of 64
    groups, the 1M test form), so a TEST form, NOT servable. "rolled" (without ``lookahead``) runs the
    stream of a sequence as one ``fori_loop`` of 4-item bodies, r / 4 trips (8 T32 + 4 T16 + 2 T8 + T4): trip j takes
    stream rows 4 j .. 4 j + 3. Each body adds its PSUM block into the state after its 4 items, as the 4-item blocks of
    the other forms do, in the same row order, so the bits are the same. The program holds one 4-item body for each
    sequence in place of the 32, 16, 8 and 4-item bodies, so it is small but slow (a loop boundary every 4 items). Use it
    for a pass that seldom runs: the capless redo (``tkg_mla_block`` ``fp8_pv_redo_rolled``). "exact" (with ``lookahead``
    4; ``tkg_mla_block`` ``fp8_pv_batched_exact``) runs the chunks of all sequences, then the stream rows 0 .. r_min - 1
    that every sequence has (r_min: the shortest stream of the pass) for all sequences in shared rolled loops of 16-item
    and 4-item bodies, then the rows r_min .. r_b - 1 of each sequence in rolled loops of its own, inside one loop of 0
    or 1 trips. No item runs past the stream of its own sequence, so a pass of mixed contexts does no masked work. A
    pass has at most 3 + 2 S loops, in place of the 4 S loops of "loops" (``exact_stream``). The same blocks in the same
    row order: the same bits.

    ``value_form`` (``VALUE_FORMS``, opt-in; 0 keeps the program) splits each item's value-word copy into 4 VectorE
    pieces and places piece t of item k in sub-slot prep + t, after the quantize and the state adds, so a quantize does
    not wait behind a whole copy (4: a page a piece; 5: 128 features a piece in the order of form 0). The same byte
    moves, the same bits. ``page_runs`` 2 (opt-in, with a setup of the same ``page_runs``): the 4 pages of an item load
    as 2 runs of 2 consecutive kernel pages (one DMA and one page-id register each, not 4), the bytes of the 4 page
    DMAs when the table's groups are runs of 2 (the 2 kernel pages of a 256-row page). A sequence with a broken run gets
    a NaN weight sum (the state sums times ``run_scale``: 1.0 keeps every bit), as the packed loop's ``page_runs``.
    ``clamp_split`` (opt-in) clamps each page of a chunk item alone (``_prep_item``), so the transposes of a page wait
    for that page's clamp, not the item's; the same elementwise op on the same bytes, so the same bits.

    ``qk_group`` (opt-in, 2 or 4 pages; 0 keeps the program): the scores of a sub-slot issue the MX QK of ``qk_group``
    pages, then their rope matmuls, in program order (``nl.no_reorder``), so fewer QK weight loads follow a rope
    matmul. With 2, the exp and quantize of sub-slot s - 1 and the P@V and weight sum of s - 2 run between the two
    halves of the scores of s, so the QK of the second half follow the P@V and the weight sum. Each page slice still
    takes its MX QK, then its rope matmul, with no other write between them: the same bits.

    ``first_carry`` (``tkg_mla_block`` ``fp8_pv_tables_ahead``): two page buffers that hold the pages of chunk items 0 and 1
    of the first sequence, loaded by the pass before (its ``next_chunk``), so the first chunk loads only the rest.
    ``next_chunk`` (``fp8_pv_tables_ahead`` of the next pass: its chunk entries and two page buffers): the tail of the
    last sequence's chunk also loads the pages of chunk items 0 and 1 of the next pass's first sequence into the buffers,
    in the page queue's window before the stream (the same pages, so the same bits).

    ``null_skip`` (opt-in, with a setup of ``null_last``: the chunk's real items first): a sequence whose chunk holds
    one real item (3 null items: g = 1 mod 4, every sequence at 16K) runs a chunk of that one item, in a forward
    then-only IF on its valid count; any other sequence runs the chunk of 4 items, in the IF on the other count. 2 (the
    merged form, with the lookahead and the grouped IF stream): the 4-item chunk's IF goes inside the stream's any-IF,
    whose condition takes (count != 1) too, so a short context (count 1, no stream body) skips the 4-item chunk and the
    stream in one not-taken branch (one IF a sequence more than without the null skip, in place of two). The
    null items add only zero weights, so each numerator and sum takes the same terms in the same order: the same bits
    up to the sign of an exact zero (a column whose every real term is -0 takes +0 from the null items without the null
    skip).

    ``prep_late`` (opt-in; False keeps the program): the item preps of a sub-slot (the key and rope transposes and
    their PSUM copies) run after its scores and P@V, not before its scores (a schedule edge from the sub-slot's last
    TensorE instruction to each transpose pins the order), so TensorE does that work while the
    previous prep's ScalarE copies free the transpose banks. The page loads keep their place. An item still preps
    two sub-slots before its first scores. The same instructions on the same operands: the same bits.

    ``page_buffers`` (2 or 3; 2 keeps the program): the page buffers of the items a body loads itself. With 2, item k
    takes buffer k mod 2, so its page load waits for the last read of item k - 2 (its prep, or its last value-word
    piece on VectorE), and the load sits ``groups`` sub-slots before its prep. With 3, each such item takes the buffer
    whose last read comes first in program order (``_page_plan``), so a load waits for the item three back, and it
    sits one item earlier (2 ``groups`` sub-slots before its prep) when a buffer is free there (else at the place of
    2). The carry of a body (items 0 and 1 in buffers 0 and 1 with ``lookahead``) and the tails (the next body's items
    0 and 1 into buffers 0 and 1) keep their buffers: the last item of a body never takes buffer 0, which the tail of
    its item before reloads. The same pages reach the same instructions, so the bits are the same.

    ``chunks_first`` (with the lookahead and the grouped IF stream; False keeps the program): the trip registers, the
    item rows and the chunk of every sequence run ahead of the first sequence's IF tree, so the block after the join of
    one sequence's any-IF holds only the next any-IF's branch, not the next sequence's chunk. Each sequence after the
    first takes item-row tiles and lookahead page buffers of its own, which the chunk tail of the sequence before
    loads (the last chunk's tail loads the first sequence's). A chunk and a stream body of another sequence share no
    state column, and each column takes the same terms in the same order: the same bits.

    ``rope_copy_vector`` (opt-in; False keeps the program): the rope byte copy (uint8, PSUM to SBUF) of each item prep
    runs on VectorE, not on ScalarE (``_prep_item``). ScalarE runs the exp of every sub-slot and is the busiest engine
    of the capless loop with 4 sequences a pass. A copy of uint8 values writes the same bytes on either engine, so the
    bits are the same.

    ``cap_pool`` (opt-in; False keeps the program): the score cap of a capped pass moves off VectorE. The VectorE
    minimum on the scores of each sub-slot (``stage_a``) goes out, and GpSimdE caps the BF16 weights after the exp
    (``cap_weights``) at the exp of the cap (``cap_weight_tile``, from the same ScalarE exp). The exp is monotonic, so
    each weight is the weight of the capped score: the same bits. A capless pass takes no cap.
    """
    kernel_assert(
        value_form in VALUE_FORMS and page_runs in (0, 2),
        f"fp8_pv_loop takes value_form in {VALUE_FORMS} and page_runs 0 or 2, got {value_form} / {page_runs}",
    )
    kernel_assert(
        setup.get("page_runs", 0) == page_runs,
        f"fp8_pv_loop page_runs {page_runs} needs the item rows of a setup with the same page_runs",
    )
    fp32 = nl.float32
    bf16 = nl.bfloat16
    int32 = nl.int32
    u8 = nl.uint8
    fp8 = nl.float8_e4m3fn
    x4 = nl.float8_e4m3fn_x4
    vector = nisa.engine.vector
    P, K, R = _P_MAX, _KEYS, _ROWS
    groups = pack_p2_groups(queries)
    width = _P2_STAGE_WIDTH
    nb = CHUNK_ITEMS
    fields = ITEM_FIELDS
    rope_rows_q = _ROPE + queries
    stage = setup["stage"]
    q_words, q_scales = setup.get("q_words"), setup.get("q_scales")
    stage_f = None if stage is None else stage.view(fp32)
    stage_u8 = None if stage is None else stage.view(u8)
    plan = setup["plan"]
    rows_per_seq = plan["rest_hbm"].shape[0] // seqs
    rows_alloc = setup["rows_alloc"]
    seq_rows = setup["seq_rows"]
    info = plan["seq_info"]
    unit, identity, ones_b, zero = setup["unit"], setup["identity"], setup["ones"], setup["zero"]
    qpe, pat, chunk_ents, thr = setup["qpe"], setup["pat"], setup["chunk_ents"], setup["thr"]
    words = setup["words"]

    # PSUM: 8 banks, each matmul output inside one: 3 score banks (4 page slices of 128 columns each), 2 numerators,
    # the key-word transposes, the FP8 rope transposes and the weight sums.
    scores = [nl.ndarray((P, K), dtype=fp32, buffer=nl.psum) for _ in range(_SCORE_BANKS)]
    numer = [nl.ndarray((P, _KV_LORA), dtype=fp32, buffer=nl.psum) for _ in range(2)]
    kw_psum = nl.ndarray((P, K), dtype=fp32, buffer=nl.psum)
    rope_psum = nl.ndarray((P, 4 * K), dtype=fp8, buffer=nl.psum)
    l_sc = nl.ndarray((P, K), dtype=fp32, buffer=nl.psum)

    kernel_assert(page_buffers in (2, 3), f"fp8_pv_loop page_buffers takes 2 or 3, got {page_buffers!r}")
    slots = 2 * nb
    raw = [sb.tile((P, _GROUP_BLOCKS, _CACHE_WIDTH), fp8) for _ in range(page_buffers)]
    key_words = [sb.tile((P, K), fp32) for _ in range(slots)]
    value_words = [sb.tile((P, _KV_LORA, 4), u8) for _ in range(slots)]
    # The key side of the 98 rows: zero mask rows (the stream), zeros, and 1.0 in the reference rows.
    rope_keys = [sb.tile((R, K), fp8) for _ in range(slots)]
    rope_keys_m = [sb.tile((R, K), fp8) for _ in range(nb)]
    for rk0 in rope_keys + rope_keys_m:
        nisa.memset(dst=rk0[0:R, 0:K], value=0.0, engine=vector)
        nisa.memset(dst=rk0[FRAME_ROW:R, 0:K], value=1.0, engine=vector)
    p_t = [sb.tile((P, K), bf16) for _ in range(_P_RING)]
    p_q = [sb.tile((P, P), fp32) for _ in range(_P_RING)]
    p_s = [sb.tile((P, P), u8) for _ in range(_P_RING)]
    # ``cap_pool``: the weight cap of the P^T after the exp (GpSimdE), in place of the VectorE score cap.
    cap = cap_weight_tile(sb.tile, zero[0:P, 8:9]) if cap_pool and capped else None
    kernel_assert(lookahead in LOOKAHEAD_ITEMS, f"fp8_pv lookahead takes {LOOKAHEAD_ITEMS} items, got {lookahead!r}")
    kernel_assert(
        stream in STREAM_FORMS and (stream in ("loops", "rolled") or lookahead) and not (stream == "rolled" and lookahead),
        f"fp8_pv stream takes {STREAM_FORMS} (all but 'loops' and 'rolled' with lookahead, 'rolled' without it), got "
        f"{stream!r}",
    )
    kernel_assert(qk_group in QK_GROUP_PAGES, f"fp8_pv qk_group takes {QK_GROUP_PAGES} pages, got {qk_group!r}")
    bodies = [body for body in STREAM_BODIES if body[0] <= rows_per_seq]
    ahead = lookahead if bodies else 0
    # ``chunks_first`` changes nothing without a stream (no IF tree to run the chunks ahead of).
    first = bool(chunks_first and ahead)
    kernel_assert(
        not first or (stream == "ifs_grouped" and not null_skip),
        f"fp8_pv chunks_first takes the grouped IF stream without null_skip, got {stream!r} / {null_skip!r}",
    )
    if ahead:
        # Every stream loop runs 0 or 1 trips (T32 = r >> 5 <= 1), and the rows a tail reads (at most r + 3) are
        # stream rows of the pass (rows_per_seq a multiple of 4).
        kernel_assert(
            rows_per_seq <= 64 and rows_per_seq % CHUNK_ITEMS == 0,
            f"fp8_pv lookahead needs at most 64 key groups a sequence in whole chunks, got {rows_per_seq}",
        )
        # The "exact" stream runs bodies of 16 and 4 items only.
        ent_sizes = [s for s in (16, CHUNK_ITEMS) if s <= rows_per_seq] if stream == "exact" else [b[0] for b in bodies]
        ents_of = {size: sb.tile((1, size * fields), int32) for size in ent_sizes}
        next_ids = sb.tile((1, ahead * fields), int32)
        raw_c = [sb.tile((P, _GROUP_BLOCKS, _CACHE_WIDTH), fp8) for _ in range(ahead - 2)]
    else:
        ents = sb.tile((1, 32 * fields), int32)
    raw_n = [sb.tile((P, _GROUP_BLOCKS, _CACHE_WIDTH), fp8) for _ in range(2)] if chunk_ahead and seqs > 1 else []
    # The item rows, the tail page ids and the lookahead page buffers of each sequence: one set for all of them, or
    # with ``chunks_first`` a set of its own for each sequence after the first (live across the IF trees before it).
    seq_sets = [(ents_of, next_ids, (raw[0], raw[1], *raw_c))] if ahead else []
    for _ in range(1, seqs) if first else ():
        seq_sets.append(
            (
                {size: sb.tile((1, size * fields), int32) for size, _, _ in bodies},
                sb.tile((1, ahead * fields), int32),
                tuple(sb.tile((P, _GROUP_BLOCKS, _CACHE_WIDTH), fp8) for _ in range(ahead)),
            )
        )

    def seq_set(b):
        return seq_sets[b if first else 0]

    if null_skip:
        # The chunk IFs load these page buffers and later IFs read them first. One word of each, written here above
        # every IF, starts the compiler's live range of each before all of them (the loads overwrite it before a read).
        for buf in raw + (raw_c if ahead else []) + raw_n:
            nisa.memset(dst=buf[0:1, 0, 0:4], value=0.0, engine=vector)

    def place(j, bi=nb):
        # ``bi``: the items of a block (``nb``; fewer for a ``null_skip`` chunk of fewer items).
        blk, rem = divmod(j, bi * groups)
        g, i = divmod(rem, bi)
        return blk * bi + i, g, i

    def load_pages(k, masked, b, ents_t=None, buf=None, chunk_src=None):
        # ``chunk_src``: the chunk entries of another pass (``next_chunk``) in place of this pass's.
        chunk_t = chunk_ents if chunk_src is None else chunk_src
        src_ents, base = (chunk_t, nb * fields * b) if masked else (ents if ents_t is None else ents_t, 0)
        buf = raw[k % 2] if buf is None else buf
        per = page_runs or 1
        for t in range(_GROUP_BLOCKS // per):
            # With page runs, run t (pages 2 t, 2 t + 1) from its first page in field t.
            if page_runs:
                dst = buf[0:P, per * t : per * (t + 1), 0:_CACHE_WIDTH]
                pattern = [[_CACHE_WIDTH, P], [kv_pool.shape[1], per], [1, _CACHE_WIDTH]]
            else:
                dst = buf[0:P, t, 0:_CACHE_WIDTH]
                pattern = [[_CACHE_WIDTH, P], [1, _CACHE_WIDTH]]
            dma = nisa.dma_copy(
                dst=dst,
                src=kv_pool.ap(
                    pattern=pattern,
                    offset=0,
                    scalar_offset=src_ents[0:1, base + fields * k + t : base + fields * k + t + 1],
                    indirect_dim=0,
                ),
                dge_mode=nisa.dge_mode.hwdge,
                engine=nisa.engine.sync,
            )
            if masked and after_writes:
                # The chunk holds the current block: its pages wait for the owner writes of this call. The compiler
                # orders this pool read after each owner write (record_pool_writes); the others take a runtime edge.
                wait_for_completion(dma, after_writes, pool_reads=True)

    def emit_body(b, items, masked, ents_t=None, carry=(), tail=None, bias_t=None):
        """``carry``: the buffers that hold the pages of items 0 .. len(carry) - 1, loaded by the body before
        (``lookahead``, ``chunk_ahead``); ``tail(0)`` / ``tail(1)`` run right after the preps of items ``items`` - 2 /
        ``items`` - 1 (the last reads of the page buffers): the next body's loads. ``bias_t`` (``stream`` "batched"):
        a ``[128, items]`` FP32 tile, the exp bias of item k in column k (0, or ``-MASK_BIAS`` for an item past the
        stream of the sequence); None keeps the zero bias."""
        total = items * groups
        lag = 2
        bi = min(nb, items)

        def rope_of(k):
            return rope_keys_m[k % nb] if masked else rope_keys[k % slots]

        # ``page_buffers`` 3: the buffer and the load sub-slot of each item the body loads (``_page_plan``).
        plan = None
        if page_buffers == 3:
            held = {i: k for k, buf in enumerate(carry) for i, r in enumerate(raw) if buf is r}
            plan = _page_plan(items, groups, nb, len(carry), held, value_form, tail is not None, len(raw))

        def raw_of(k):
            if k < len(carry):
                return carry[k]
            return raw[plan[k][0]] if plan is not None else raw[k % 2]

        def prep(k, after=()):
            sl = k % slots
            _prep_item(
                raw_of(k),
                kw_psum,
                rope_psum,
                key_words[sl],
                rope_of(k),
                value_words[sl] if value_form == 0 else None,
                pat,
                identity,
                queries,
                thr[0:rope_rows_q, nb * b + k : nb * b + k + 1] if masked else None,
                masked and clamp_pages,
                clamp_split,
                after=after,
                rope_engine=vector if rope_copy_vector else nisa.engine.scalar,
            )

        def value_piece(k, piece):
            _value_words(raw_of(k), value_words[k % slots], value_form, piece)

        # The last TensorE instruction of the sub-slot's scores and P@V: ``prep_late`` preps follow it in the schedule.
        pe_last = [None]

        def stage_a(j, first=0, last=_GROUP_BLOCKS):
            """The scores of sub-slot ``j`` on pages ``first`` .. ``last`` - 1 (the cap after the last page)."""
            k, g, i = place(j, bi)
            base = (g * width) // 2
            sc0 = 2 * (g * width + _P2_STAGE_MX_SCALE)
            c0 = (b * groups + g) * P
            s_ps = scores[j % _SCORE_BANKS]
            # The query words and scales of (b, g): with ``query_words`` the bytes from the gather, else the stage-free
            # tiles, else the stage's.
            if words is not None:
                q_mv, q_sc = words[0][0:P, c0 : c0 + P].view(x4), words[1][0:P, c0 : c0 + P]
            elif stage is None:
                q_mv, q_sc = q_words[0:P, b, g * P : (g + 1) * P].view(x4), q_scales[0:P, b, g * P : (g + 1) * P]
            else:
                q_mv, q_sc = stage_f[0:P, b, base : base + P].view(x4), stage_u8[0:P, b, sc0 : sc0 + P]

            def score_pages(pages):
                for t in pages:
                    nisa.nc_matmul_mx(
                        dst=s_ps[0:P, t * P : (t + 1) * P],
                        stationary=key_words[k % slots][0:P, t * P : (t + 1) * P].view(x4),
                        moving=q_mv,
                        stationary_scale=unit[0:P, 0:P],
                        moving_scale=q_sc,
                        accumulate=False,
                    )
                for t in pages:
                    pe_last[0] = nisa.nc_matmul(
                        dst=s_ps[0:P, t * P : (t + 1) * P],
                        stationary=rope_of(k)[0:R, t * P : (t + 1) * P],
                        moving=qpe[0:R, c0 : c0 + P],
                        accumulate=True,
                    )

            for t0 in range(first, last, qk_group or 1):
                if not qk_group:
                    score_pages((t0,))
                    continue
                # The scheduler would put each rope right behind its own QK: keep the group in program order.
                with nl.no_reorder():
                    score_pages(range(t0, t0 + qk_group))
            if capped and last == _GROUP_BLOCKS and cap is None:
                nisa.tensor_scalar(
                    dst=s_ps, data=s_ps, op0=nl.minimum, operand0=CAP_SHIFTED / _SOFTMAX_SCALE, engine=vector
                )

        def stage_b(j):
            r = j % _P_RING
            s_ps = scores[j % _SCORE_BANKS]
            # P^T [row w, 4 c + t] = exp(scale x S'_t[w, c]): the 4 page slices read in (column, page) order.
            src = s_ps.ap(pattern=[[K, P], [1, P], [P, _GROUP_BLOCKS]], offset=0)
            if bias_t is None:
                bias = zero[0:P, 8:9]
            else:
                # "batched": the bias column of the sub-slot's item (0 keeps the bits; -MASK_BIAS gives 0 weights).
                k_item = place(j, bi)[0]
                bias = bias_t[0:P, k_item : k_item + 1]
            nisa.activation(dst=p_t[r], op=nl.exp, data=src, scale=_SOFTMAX_SCALE, bias=bias)
            if cap is not None:
                cap_weights(p_t[r], cap)
            nisa.quantize_mx(dst=p_q[r].view(x4), src=p_t[r], dst_scale=p_s[r])

        def stage_c(j):
            k, g, i = place(j, bi)
            r = j % _P_RING
            col = g * seqs + b
            v_words = value_words[k % slots].view(fp32)[0:P, 0:_KV_LORA].view(x4)
            o_ps = numer[g % 2]
            l_ps = l_sc[0:P, 2 * (g % 2) : 2 * (g % 2) + 2]
            acc = i > 0
            nisa.nc_matmul_mx(
                dst=o_ps,
                stationary=p_q[r].view(x4),
                moving=v_words,
                stationary_scale=p_s[r],
                moving_scale=unit[0:P, 0:_KV_LORA],
                accumulate=acc,
            )
            pe_last[0] = nisa.nc_matmul_mx(
                dst=l_ps,
                stationary=p_q[r].view(x4),
                moving=ones_b.view(fp32).view(x4),
                stationary_scale=p_s[r],
                moving_scale=unit[0:P, 0:2],
                accumulate=acc,
            )
            if i == bi - 1:
                st = state_flat[0:P, col * _STATE_WIDTH : col * _STATE_WIDTH + _KV_LORA]
                sm = state_flat[0:P, col * _STATE_WIDTH + _KV_LORA : (col + 1) * _STATE_WIDTH]
                nisa.tensor_tensor(dst=st, data1=st, data2=o_ps, op=nl.add, engine=vector)
                nisa.tensor_tensor(dst=sm, data1=sm, data2=l_ps[0:P, 0:1], op=nl.add, engine=vector)

        # Events (sub-slot, kind, item, piece): 0 the page loads, 1 the prep, 2 a value-word piece (value_form 4 / 5,
        # placed after the sub-slot's quantize and state adds). Events before sub-slot 0 run first, in order.
        events = []
        for k in range(items):
            prep_at = groups * k - groups * nb + 3
            if k >= len(carry):
                events.append((plan[k][1] if plan is not None else prep_at - groups, 0, k, 0))
            events.append((prep_at, 1, k, 0))
            if value_form:
                kernel_assert(groups >= _GROUP_BLOCKS, "value_form 4 / 5 places a piece a sub-slot: needs 4+ groups")
                for piece in range(_GROUP_BLOCKS):
                    at = prep_at + piece
                    events.append((at if prep_at >= 0 else min(at, -1), 2, k, piece))
        events.sort()

        def run(kind, k, piece, after=()):
            # The tail loads the next body's pages into the buffers of items items - 2 / items - 1: after their last
            # read, the prep (value_form 0) or the last value piece.
            if kind == 0:
                load_pages(k, masked, b, ents_t, buf=raw_of(k) if plan is not None else None)
                return
            if kind == 2:
                value_piece(k, piece)
                if tail is not None and k >= items - 2 and piece == _GROUP_BLOCKS - 1:
                    run_tail(k)
                return
            prep(k, after)
            if tail is not None and k >= items - 2 and not value_form:
                run_tail(k)

        def run_tail(k):
            # A 1-item chunk (``null_skip``) runs both tails after its one item (both page buffers are free then).
            for which in (0, 1) if items == 1 else (k - (items - 2),):
                tail(which)

        timed = {}
        for e in events:
            if e[0] < 0:
                run(*e[1:])
            else:
                timed.setdefault(e[0], []).append(e)
        half = _GROUP_BLOCKS // 2
        for s in range(total + lag):
            pe_last[0] = None
            for _, kind, k, piece in timed.get(s, []):
                if kind == 0 or (kind == 1 and not prep_late):
                    run(kind, k, piece)
            if s < total:
                stage_a(s, 0, half if qk_group == 2 else _GROUP_BLOCKS)
            if 1 <= s <= total:
                stage_b(s - 1)
            if lag <= s < total + lag:
                stage_c(s - lag)
            if s < total and qk_group == 2:
                stage_a(s, half)
            for _, kind, k, piece in timed.get(s, []):
                # ``prep_late``: the item preps of the sub-slot after its scores and P@V (a schedule edge from the
                # sub-slot's last TensorE instruction, so the compiler keeps the order), ahead of its value pieces.
                if kind == 1 and prep_late:
                    run(kind, k, piece, (pe_last[0],) if pe_last[0] is not None else ())
            for _, kind, k, piece in timed.get(s, []):
                if kind == 2:
                    run(kind, k, piece)

    def stream_rows(b, row_col, skip, count):
        """``count`` stream rows of sequence ``b`` from ``skip`` rows past the start of the loop whose start-row column
        is ``row_col`` (None: the 32-item loop, at row 0 with lookahead)."""
        if row_col is None:
            flat = seq_rows[b].reshape((1, rows_alloc * fields))
            return flat[0:1, skip * fields : (skip + count) * fields]
        return seq_rows[b].ap(
            pattern=[[fields, 1], [1, count * fields]],
            offset=skip * fields,
            scalar_offset=info[0:1, 8 * b + row_col : 8 * b + row_col + 1],
            indirect_dim=0,
        )

    def next_tail(b, row_col, size):
        """The tail of a body that ends ``size`` rows past the start of the loop of ``row_col`` (the chunk: None, 0):
        the page ids, then the pages, of the first ``ahead`` rows after it."""

        _, ids, bufs = seq_set(b)

        def tail(which):
            if which == 0:
                nisa.dma_copy(
                    dst=ids[0:1, 0 : ahead * fields],
                    src=stream_rows(b, row_col, size, ahead),
                    dge_mode=nisa.dge_mode.hwdge,
                    engine=nisa.engine.sync,
                )
                load_pages(0, False, b, ids, bufs[0])
                for j in range(2, ahead):
                    load_pages(j, False, b, ids, bufs[j])
            else:
                load_pages(1, False, b, ids, bufs[1])

        return tail

    def chunk_tail(b, stream_tail):
        """The tail of chunk ``b``: the next stream body's loads (``lookahead``), then with ``chunk_ahead`` the first
        pages of the next sequence's chunk, and after the last sequence's chunk with ``next_chunk`` the first pages of
        the next pass's first chunk."""
        to_next = next_chunk is not None and b == seqs - 1
        if stream_tail is None and not (raw_n and b + 1 < seqs) and not to_next:
            return None

        def tail(which):
            if stream_tail is not None:
                stream_tail(which)
            if which == 1 and raw_n and b + 1 < seqs:
                for j in range(2):
                    load_pages(j, True, b + 1, buf=raw_n[j])
            if which == 1 and to_next:
                # ``next_chunk`` (``fp8_pv_tables_ahead``): chunk items 0 and 1 of the next pass's first sequence, with
                # the owner writes of the call (``after_writes``) as every chunk page load.
                for j in range(2):
                    load_pages(j, True, 0, buf=next_chunk["bufs"][j], chunk_src=next_chunk["chunk_ents"])

        return tail

    def chunk_flags(b):
        """``null_skip``: the FP32 and INT32 [1, 2] tiles of (count == 1, count != 1) for the valid count of the chunk
        rows of sequence ``b`` (field 6 of rows 4 b .. 4 b + 3)."""
        valid = chunk_ents.ap(pattern=[[seqs * nb * fields, 1], [fields, nb]], offset=nb * fields * b + 6)
        v_f = sb.tile((1, nb), fp32)
        nisa.tensor_copy(dst=v_f, src=valid, engine=vector)
        c_f = sb.tile((1, 2), fp32)
        nisa.tensor_reduce(dst=c_f[0:1, 0:1], op=nl.add, data=v_f, axis=1)
        nisa.tensor_scalar(dst=c_f[0:1, 0:1], data=c_f[0:1, 0:1], op0=nl.equal, operand0=1.0, engine=vector)
        nisa.tensor_scalar(
            dst=c_f[0:1, 1:2],
            data=c_f[0:1, 0:1],
            op0=nl.multiply,
            operand0=-1.0,
            op1=nl.add,
            operand1=1.0,
            engine=vector,
        )
        c_i = sb.tile((1, 2), int32)
        nisa.tensor_copy(dst=c_i, src=c_f, engine=vector)
        return c_f, c_i

    def flag_regs(c_i):
        regs = []
        for col in range(2):
            regs.append(nisa.register_alloc())
            nisa.register_load(regs[-1], c_i[0:1, col : col + 1])
        return regs

    def emit_chunk(b, carry, tail, flags=None):
        """The chunk of sequence ``b``; with ``null_skip`` a 1-item chunk when its valid count is 1, else the 4 items
        (forward then-only IFs on the two counts). ``flags`` (``null_skip`` 2, the merged form): the count registers
        of ``chunk_flags``; only the 1-item IF is traced here (the 4-item chunk goes in the stream's any-IF)."""
        if not null_skip:
            emit_body(b, nb, True, carry=carry, tail=tail)
            return
        regs = flags if flags is not None else flag_regs(chunk_flags(b)[1])
        nke.if_else(regs[0], lambda: emit_body(b, 1, True, carry=carry, tail=tail))
        if flags is None:
            nke.if_else(regs[1], lambda: emit_body(b, nb, True, carry=carry, tail=tail))

    def rolled_stream(b):
        """``stream`` "rolled": the stream rows of sequence ``b`` as one loop of 4-item bodies. The trip count is the
        stream blocks of 4 rows, 8 T32 + 4 T16 + 2 T8 + T4 (exact small integers in FP32)."""
        sizes = [(size, trip_col) for size, trip_col, _ in STREAM_BODIES if size <= rows_per_seq]
        if not sizes:
            return
        c0, n = sizes[0][1], len(sizes)
        t_f = sb.tile((1, len(STREAM_BODIES)), fp32)
        nisa.tensor_copy(dst=t_f[0:1, 0:n], src=info[0:1, 8 * b + c0 : 8 * b + c0 + n], engine=vector)
        for i, (size, _) in enumerate(sizes):
            nisa.tensor_scalar(
                dst=t_f[0:1, i : i + 1], data=t_f[0:1, i : i + 1], op0=nl.multiply, operand0=float(size // nb), engine=vector
            )
        blocks_f = sb.tile((1, 1), fp32)
        nisa.tensor_reduce(dst=blocks_f, op=nl.add, data=t_f[0:1, 0:n], axis=1)
        blocks_i = sb.tile((1, 1), int32)
        nisa.tensor_copy(dst=blocks_i, src=blocks_f, engine=vector)
        rows4 = seq_rows[b].reshape((rows_alloc // nb, nb * fields))

        def body(step, b=b):
            src = rows4.ap(pattern=[[nb * fields, 1], [1, nb * fields]], offset=0, scalar_offset=step, indirect_dim=0)
            nisa.dma_copy(dst=ents[0:1, 0 : nb * fields], src=src)
            emit_body(b, nb, False)

        nl.fori_loop(0, nisa.scalar_load(blocks_i), body)

    kernel_assert(
        not (first_carry or next_chunk) or nb >= 2,
        "fp8_pv_loop first_carry / next_chunk carry chunk items 0 and 1: the chunk needs at least 2 items",
    )
    kernel_assert(
        not first_carry or len(first_carry) == 2, "fp8_pv_loop first_carry holds the pages of chunk items 0, 1"
    )
    def batched_stream():
        """``stream`` "batched": the chunks of all sequences, then one 0/1-trip loop for each body size, with the body
        of each sequence in it, in sequence order. The trips are the bits of the longest stream r_max of the pass, and
        every sequence walks its stream rows 0 .. r_max - 1 in the same bodies. An item at row j >= r_b gets the exp
        bias ``-MASK_BIAS``, so it adds 0. Its pages are full key groups (``fp8_pv_items`` ``pad_longest``). The valid
        items of a sequence add into its state in the same 4-item blocks and in the same row order as the "loops"
        form, so the bits are the same (up to the sign of an exact zero). At 1M each sequence runs every body, so the
        pass takes 4 loop boundaries in place of 4 for each sequence (16 for 4 sequences)."""
        gpsimd = nisa.engine.gpsimd
        info_f = sb.tile((1, 8 * seqs), fp32)
        nisa.tensor_copy(dst=info_f, src=info, engine=vector)
        # r_b = rel4 + 4 T4 (columns 8 b + 7 and 8 b + 4 of the seq info), then r_max.
        r_f = sb.tile((1, seqs + 1), fp32)
        for b in range(seqs):
            nisa.scalar_tensor_tensor(
                dst=r_f[0:1, b : b + 1],
                data=info_f[0:1, 8 * b + 4 : 8 * b + 5],
                op0=nl.multiply,
                operand0=float(CHUNK_ITEMS),
                op1=nl.add,
                operand1=info_f[0:1, 8 * b + 7 : 8 * b + 8],
            )
        nisa.tensor_reduce(dst=r_f[0:1, seqs : seqs + 1], op=nl.maximum, data=r_f[0:1, 0:seqs], axis=1)
        # The shared trips T32, T16, T8, T4 of r_max (columns 0 .. 3) and the start row of each body (columns 4 .. 7):
        # 0, 32 T32, + 16 T16, + 8 T8 (exact small integers).
        m_i = sb.tile((1, 1), int32)
        nisa.tensor_copy(dst=m_i, src=r_f[0:1, seqs : seqs + 1], engine=vector)
        t_i = sb.tile((1, len(STREAM_BODIES)), int32)
        for k, shift in enumerate((5, 4, 3, 2)):
            nisa.tensor_scalar(dst=t_i[0:1, k : k + 1], data=m_i, op0=nl.right_shift, operand0=shift, engine=vector)
            if k:
                nisa.tensor_scalar(
                    dst=t_i[0:1, k : k + 1], data=t_i[0:1, k : k + 1], op0=nl.bitwise_and, operand0=1, engine=vector
                )
        sh_f = sb.tile((1, 2 * len(STREAM_BODIES)), fp32)
        nisa.tensor_copy(dst=sh_f[0:1, 0:4], src=t_i, engine=vector)
        nisa.memset(dst=sh_f[0:1, 4:5], value=0.0, engine=vector)
        nisa.tensor_scalar(dst=sh_f[0:1, 5:6], data=sh_f[0:1, 0:1], op0=nl.multiply, operand0=32.0, engine=vector)
        for k, size in ((6, 16.0), (7, 8.0)):
            nisa.scalar_tensor_tensor(
                dst=sh_f[0:1, k : k + 1],
                data=sh_f[0:1, k - 5 : k - 4],
                op0=nl.multiply,
                operand0=size,
                op1=nl.add,
                operand1=sh_f[0:1, k - 1 : k],
            )
        sh_i = sb.tile((1, 2 * len(STREAM_BODIES)), int32)
        nisa.tensor_copy(dst=sh_i, src=sh_f, engine=vector)
        # Every partition gets r_b of each sequence and the start rows (an HBM round trip and a broadcast DMA).
        n_bc = seqs + len(STREAM_BODIES)
        bc_src = sb.tile((1, n_bc), fp32)
        nisa.tensor_copy(dst=bc_src[0:1, 0:seqs], src=r_f[0:1, 0:seqs], engine=vector)
        nisa.tensor_copy(dst=bc_src[0:1, seqs:n_bc], src=sh_f[0:1, 4:8], engine=vector)
        bc_hbm = nl.ndarray((1, n_bc), dtype=fp32, buffer=nl.hbm, name=f"{tag}_fp8pv_batched")
        nisa.dma_copy(dst=bc_hbm, src=bc_src, name=f"{tag}_fp8pv_batched_store")
        bc = sb.tile((P, n_bc), fp32)
        nisa.dma_copy(dst=bc, src=bc_hbm.broadcast(0, P), name=f"{tag}_fp8pv_batched_load")
        # The exp bias of item k of the body of each (size, sequence): -MASK_BIAS where start + k >= r_b, else 0.
        widest = max(size for size, _, _ in bodies)
        k_i = sb.tile((P, widest), int32)
        nisa.iota(dst=k_i, pattern=[[1, widest]], offset=0, channel_multiplier=0)
        k_f = sb.tile((P, widest), fp32)
        nisa.tensor_copy(dst=k_f, src=k_i, engine=gpsimd)
        flag = sb.tile((P, widest), fp32)
        biases = {}
        for size, trip_col, row_col in bodies:
            k = STREAM_BODIES.index((size, trip_col, row_col))
            for b in range(seqs):
                nisa.tensor_scalar(
                    dst=flag[0:P, 0:size],
                    data=k_f[0:P, 0:size],
                    op0=nl.add,
                    operand0=bc[0:P, seqs + k : seqs + k + 1],
                    op1=nl.greater_equal,
                    operand1=bc[0:P, b : b + 1],
                    engine=gpsimd,
                )
                bias = sb.tile((P, size), fp32)
                nisa.tensor_scalar(
                    dst=bias, data=flag[0:P, 0:size], op0=nl.multiply, operand0=-MASK_BIAS, engine=gpsimd
                )
                biases[(size, b)] = bias

        def shared_rows(b, k, skip, count):
            """``count`` stream rows of sequence ``b`` from ``skip`` rows past the shared start row of body ``k``."""
            if STREAM_BODIES[k][2] is None:
                flat = seq_rows[b].reshape((1, rows_alloc * fields))
                return flat[0:1, skip * fields : (skip + count) * fields]
            return seq_rows[b].ap(
                pattern=[[fields, 1], [1, count * fields]],
                offset=skip * fields,
                scalar_offset=sh_i[0:1, 4 + k : 5 + k],
                indirect_dim=0,
            )

        _, ids_t, bufs = seq_set(0)

        def batched_tail(b, k, skip, items):
            """The tail of a body of ``items`` items: the page ids, then the pages, of the first ``ahead`` rows of
            sequence ``b`` from ``skip`` rows past the shared start row of body ``k`` (the next body that the pass
            runs). ``tail(0)`` runs after the prep of item ``items`` - 2. A body of ``ahead`` items or fewer takes every
            item from the carry buffers, so its last item still reads carry buffer ``items`` - 1 then: ``tail(1)``
            loads that buffer, after the last prep."""
            late = items - 1 if items <= ahead else None

            def tail(which):
                if which == 0:
                    nisa.dma_copy(
                        dst=ids_t[0:1, 0 : ahead * fields],
                        src=shared_rows(b, k, skip, ahead),
                        dge_mode=nisa.dge_mode.hwdge,
                        engine=nisa.engine.sync,
                    )
                    load_pages(0, False, b, ids_t, bufs[0])
                    for j in range(2, ahead):
                        if j != late:
                            load_pages(j, False, b, ids_t, bufs[j])
                else:
                    load_pages(1, False, b, ids_t, bufs[1])
                    if late is not None and late >= 2:
                        load_pages(late, False, b, ids_t, bufs[late])

            return tail

        k_first = STREAM_BODIES.index(bodies[0])
        for b in range(seqs):
            chunk_carry = tuple(raw_n) if raw_n and b > 0 else ()
            if b == 0 and first_carry:
                chunk_carry = tuple(first_carry)
            # The last chunk's tail loads the first rows of the first body that runs (row 0 of sequence 0).
            emit_chunk(
                b, chunk_carry, chunk_tail(b, batched_tail(0, k_first, 0, nb + ahead) if b == seqs - 1 else None)
            )
        # Two sets of item-row tiles (sequence parity), so the rows of a body load while the body before reads its own.
        ent_sets = (seq_set(0)[0], {size: sb.tile((1, size * fields), int32) for size, _, _ in bodies})
        for i, (size, trip_col, row_col) in enumerate(bodies):
            k = STREAM_BODIES.index((size, trip_col, row_col))

            def body(step, size=size, k=k, last=i == len(bodies) - 1):
                for b in range(seqs):
                    ents_t = ent_sets[b % 2][size]
                    nisa.dma_copy(
                        dst=ents_t[0:1, 0 : size * fields],
                        src=shared_rows(b, k, 0, size),
                        dge_mode=nisa.dge_mode.hwdge,
                        engine=nisa.engine.sync,
                    )
                    if b + 1 < seqs:
                        tail = batched_tail(b + 1, k, 0, size)
                    else:
                        tail = None if last else batched_tail(0, k, size, size)
                    bias_t = None if _BATCHED_NO_MASK_TEST else biases[(size, b)]
                    emit_body(b, size, False, ents_t, carry=bufs, tail=tail, bias_t=bias_t)

            nl.fori_loop(0, nisa.scalar_load(sh_i[0:1, k : k + 1]), body)

    def exact_stream():
        """``stream`` "exact": the chunks of all sequences, then the stream rows 0 .. r_min - 1 that every sequence of
        the pass has (r_min: the shortest stream of the pass) in shared rolled loops, then the rows r_min .. r_b - 1 of
        each sequence in rolled loops of its own, inside one outer loop of 0 or 1 trips (1 when a stream is longer than
        r_min). Each rolled part is a loop of 16-item bodies and a loop of 4-item bodies, so a pass has at most
        3 + 2 S loops (2 + 2 S at a short context), not 4 S. No item runs past the stream of its own sequence. The
        items of each sequence add into its state in the same 4-item blocks, in the same row order, as the "loops"
        form, so the bits are the same.

        The loop counters count blocks of 4 stream rows (``rows_blk``). The tail of each body loads the first
        ``ahead`` rows after it: in the shared part the rows of the next sequence of the trip (the last sequence: of
        sequence 0 at the next trip, which after the last trip is the start r_min of its own part); in the part of one
        sequence its own next rows. Sequence 0 then starts its own part on the pages of the shared part's last tail
        (or of the last chunk's tail when r_min is 0). Each later sequence takes the other of two carry sets. A
        sequence in set 0 loads the first rows of the next sequence into set 1 before its loops; a sequence in set 1
        loads them into set 0 after its 16-item loop (its 4-item bodies read only set 1). Set 1 holds no buffer of
        the item loads of a 16-item body (``raw`` 0 / 1), so no body overwrites a carried page."""
        int32_ = nl.int32
        sizes = [size for size in (16, CHUNK_ITEMS) if size <= rows_per_seq]
        if not sizes:
            # No stream row (a table of fewer than 4 key groups a sequence after the chunk): the chunks only.
            for b in range(seqs):
                emit_chunk(b, tuple(raw_n) if raw_n and b > 0 else tuple(first_carry) if b == 0 else (), chunk_tail(b, None))
            return
        big = 16 in sizes
        info_f = sb.tile((1, 8 * seqs), fp32)
        nisa.tensor_copy(dst=info_f, src=info, engine=vector)
        # Columns of the bounds (blocks of 4 rows, exact small integers): 0 the end of the shared 16-item loop
        # (4 (m >> 2)), 1 m = r_min / 4, 2 the trip of the outer loop (max > min), 3 + b the end of the 16-item loop of
        # sequence b (m + 4 ((r4_b - m) >> 2)), 3 + S + b the end of its stream r4_b = r_b / 4 = T4 + rel4 / 4.
        n_v = 3 + 2 * seqs
        v_f = sb.tile((1, n_v), fp32)
        r_cols = v_f[0:1, 3 + seqs : n_v]
        for b in range(seqs):
            nisa.scalar_tensor_tensor(
                dst=v_f[0:1, 3 + seqs + b : 4 + seqs + b],
                data=info_f[0:1, 8 * b + 7 : 8 * b + 8],
                op0=nl.multiply,
                operand0=0.25,
                op1=nl.add,
                operand1=info_f[0:1, 8 * b + 4 : 8 * b + 5],
            )
        mm = sb.tile((1, 2), fp32)
        nisa.tensor_reduce(dst=mm[0:1, 0:1], op=nl.minimum, data=r_cols, axis=1)
        nisa.tensor_reduce(dst=mm[0:1, 1:2], op=nl.maximum, data=r_cols, axis=1)
        nisa.tensor_copy(dst=v_f[0:1, 1:2], src=mm[0:1, 0:1], engine=vector)
        nisa.tensor_tensor(dst=v_f[0:1, 2:3], data1=mm[0:1, 1:2], data2=mm[0:1, 0:1], op=nl.greater, engine=vector)
        if big:
            # The 16-row parts: 4 (x >> 2) of x = m and of x = r4_b - m (shift in INT32, the rest in FP32).
            d_f = sb.tile((1, seqs + 1), fp32)
            nisa.tensor_copy(dst=d_f[0:1, 0:1], src=mm[0:1, 0:1], engine=vector)
            nisa.tensor_scalar(
                dst=d_f[0:1, 1 : seqs + 1], data=r_cols, op0=nl.subtract, operand0=mm[0:1, 0:1], engine=vector
            )
            d_i = sb.tile((1, seqs + 1), int32_)
            nisa.tensor_copy(dst=d_i, src=d_f, engine=vector)
            nisa.tensor_scalar(dst=d_i, data=d_i, op0=nl.right_shift, operand0=2, engine=vector)
            q_f = sb.tile((1, seqs + 1), fp32)
            nisa.tensor_copy(dst=q_f, src=d_i, engine=vector)
            nisa.tensor_scalar(dst=v_f[0:1, 0:1], data=q_f[0:1, 0:1], op0=nl.multiply, operand0=4.0, engine=vector)
            nisa.tensor_scalar(
                dst=v_f[0:1, 3 : 3 + seqs],
                data=q_f[0:1, 1 : seqs + 1],
                op0=nl.multiply,
                operand0=4.0,
                op1=nl.add,
                operand1=mm[0:1, 0:1],
                engine=vector,
            )
        else:
            nisa.memset(dst=v_f[0:1, 0:1], value=0.0, engine=vector)
            nisa.tensor_scalar(
                dst=v_f[0:1, 3 : 3 + seqs], data=r_cols, op0=nl.minimum, operand0=mm[0:1, 0:1], engine=vector
            )
        bounds = sb.tile((1, n_v), int32_)
        nisa.tensor_copy(dst=bounds, src=v_f, engine=vector)

        def bound(col):
            return nisa.scalar_load(bounds[0:1, col : col + 1])

        def rows_blk(b, blk, skip, count):
            """``count`` stream rows of sequence ``b`` from ``skip`` rows past block ``blk`` (a loop counter or a
            bounds column; None: row 0)."""
            if blk is None:
                flat = seq_rows[b].reshape((1, rows_alloc * fields))
                return flat[0:1, skip * fields : (skip + count) * fields]
            rows4 = seq_rows[b].reshape((rows_alloc // nb, nb * fields))
            return rows4.ap(
                pattern=[[nb * fields, 1], [1, count * fields]], offset=skip * fields, scalar_offset=blk, indirect_dim=0
            )

        ents0, ids0, bufs0 = seq_set(0)
        # Carry set 1: the chunk-ahead buffers (free after the chunks) and new tiles.
        extra = ahead - len(raw_n)
        bufs1 = tuple(raw_n) + tuple(sb.tile((P, _GROUP_BLOCKS, _CACHE_WIDTH), fp8) for _ in range(extra))
        sets = ((ids0, bufs0), (sb.tile((1, ahead * fields), int32_), bufs1))

        def row_tail(b, blk, skip, items, carry_set):
            """The tail of a body of ``items`` items: the page ids, then the pages, of the first ``ahead`` rows of
            sequence ``b`` from ``skip`` rows past block ``blk``, into ``carry_set``. ``tail(0)`` runs after the prep of
            item ``items`` - 2. A body of ``ahead`` items or fewer reads carry buffer ``items`` - 1 at its last prep:
            ``tail(1)`` loads that buffer."""
            ids_t, bufs_t = carry_set
            late = items - 1 if items <= ahead else None

            def tail(which):
                if which == 0:
                    nisa.dma_copy(
                        dst=ids_t[0:1, 0 : ahead * fields],
                        src=rows_blk(b, blk, skip, ahead),
                        dge_mode=nisa.dge_mode.hwdge,
                        engine=nisa.engine.sync,
                    )
                    load_pages(0, False, b, ids_t, bufs_t[0])
                    for j in range(2, ahead):
                        if j != late:
                            load_pages(j, False, b, ids_t, bufs_t[j])
                else:
                    load_pages(1, False, b, ids_t, bufs_t[1])
                    if late is not None and late >= 2:
                        load_pages(late, False, b, ids_t, bufs_t[late])

            return tail

        def preload(b, carry_set):
            """The first ``ahead`` rows of the own part of sequence ``b`` (block m) into ``carry_set``."""
            ids_t, bufs_t = carry_set
            nisa.dma_copy(
                dst=ids_t[0:1, 0 : ahead * fields],
                src=rows_blk(b, bounds[0:1, 1:2], 0, ahead),
                dge_mode=nisa.dge_mode.hwdge,
                engine=nisa.engine.sync,
            )
            for j in range(ahead):
                load_pages(j, False, b, ids_t, bufs_t[j])

        for b in range(seqs):
            chunk_carry = tuple(raw_n) if raw_n and b > 0 else ()
            if b == 0 and first_carry:
                chunk_carry = tuple(first_carry)
            # The last chunk's tail loads the first rows of sequence 0 (row 0).
            emit_chunk(
                b, chunk_carry, chunk_tail(b, row_tail(0, None, 0, nb + ahead, sets[0]) if b == seqs - 1 else None)
            )
        # Two sets of item-row tiles (sequence parity), so the rows of a body load while the body before reads its own.
        ent_sets = (ents0, {size: sb.tile((1, size * fields), int32_) for size in sizes})
        for size in sizes:

            def body(i, size=size):
                for b in range(seqs):
                    ents_t = ent_sets[b % 2][size]
                    nisa.dma_copy(
                        dst=ents_t[0:1, 0 : size * fields],
                        src=rows_blk(b, i, 0, size),
                        dge_mode=nisa.dge_mode.hwdge,
                        engine=nisa.engine.sync,
                    )
                    if b + 1 < seqs:
                        tail = row_tail(b + 1, i, 0, size, sets[0])
                    else:
                        tail = row_tail(0, i, size, size, sets[0])
                    emit_body(b, size, False, ents_t, carry=sets[0][1], tail=tail)

            # The shared 16-item loop: blocks 0 .. 4 (m >> 2) - 1; the shared 4-item loop: the rest of blocks 0 .. m - 1.
            if size == 16:
                nl.fori_loop(0, bound(0), body, step=16 // nb)
            else:
                nl.fori_loop(bound(0) if big else 0, bound(1), body, step=1)
        if seqs == 1 or _EXACT_NO_OWN_TEST:
            return

        def own_loop(b, size, lo, hi, carry_set):
            """The rows of blocks ``lo`` .. ``hi`` - 1 of sequence ``b`` in bodies of ``size`` items."""

            def body(i, b=b, size=size):
                ents_t = ent_sets[0][size]
                nisa.dma_copy(
                    dst=ents_t[0:1, 0 : size * fields],
                    src=rows_blk(b, i, 0, size),
                    dge_mode=nisa.dge_mode.hwdge,
                    engine=nisa.engine.sync,
                )
                emit_body(b, size, False, ents_t, carry=carry_set[1], tail=row_tail(b, i, size, size, carry_set))

            nl.fori_loop(lo, hi, body, step=size // nb)

        def own_parts(_):
            for b in range(seqs):
                # Sequence b runs in carry set b mod 2 (sequence 0 on the pages of the shared part's last tail).
                mine = sets[b % 2]
                other = sets[(b + 1) % 2]
                nxt = b + 1 < seqs
                if nxt and b % 2 == 0:
                    preload(b + 1, other)
                if big:
                    own_loop(b, 16, bound(1), bound(3 + b), mine)
                if nxt and b % 2 == 1:
                    preload(b + 1, other)
                own_loop(b, CHUNK_ITEMS, bound(3 + b) if big else bound(1), bound(3 + seqs + b), mine)

        nl.fori_loop(0, bound(2), own_parts)

    if stream == "batched":
        kernel_assert(
            ahead and not first and not null_skip,
            "fp8_pv stream 'batched' needs the lookahead, and no chunks_first or null_skip",
        )
        batched_stream()
    if stream == "exact":
        # Lookahead 4: a 4-item body then takes every item from its carry set, so the pre-load of the next sequence
        # (exact_stream) can go into the buffers of the 16-item item loads while a sequence of set 1 runs.
        kernel_assert(
            (ahead == 4 or not bodies) and not first and not null_skip,
            f"fp8_pv stream 'exact' needs the lookahead of 4 items, and no chunks_first or null_skip, got lookahead "
            f"{ahead}",
        )
        exact_stream()
    # ``chunks_first``: the IF tree of each sequence, traced after every chunk.
    trees = []
    for b in range(seqs) if stream not in ("batched", "exact") else ():
        chunk_carry = tuple(raw_n) if raw_n and b > 0 else ()
        if b == 0 and first_carry:
            # ``first_carry`` (``fp8_pv_tables_ahead``): the pass before loaded the pages of items 0 and 1.
            chunk_carry = tuple(first_carry)
        trips = {}
        groups_ifs = {}
        # ``null_skip`` 2 (the merged form): the 4-item chunk runs inside the stream's any-IF, whose condition becomes
        # (any stream body) or (count != 1), so a short context skips the 4-item chunk and its stream in one branch.
        merged = null_skip == 2 and ahead and stream == "ifs_grouped" and len(bodies) > 1
        m_flags = chunk_flags(b) if merged else None
        if stream in ("ifs", "ifs_grouped"):
            # The IF condition registers of the sequence's bodies (each trip count is 0 or 1), before its chunk.
            for size, trip_col, _ in bodies:
                trips[size] = nisa.register_alloc()
                nisa.register_load(trips[size], info[0:1, 8 * b + trip_col : 8 * b + trip_col + 1])
            if stream == "ifs_grouped" and len(bodies) > 1:
                # The group conditions: any body but the last (the prefix the short contexts skip together), and any body.
                c0, c1 = bodies[0][1], bodies[-1][1]
                t_f = sb.tile((1, 4), fp32)
                nisa.tensor_copy(
                    dst=t_f[0:1, 0 : c1 - c0 + 1], src=info[0:1, 8 * b + c0 : 8 * b + c1 + 1], engine=vector
                )
                g_f = sb.tile((1, 2), fp32)
                nisa.tensor_reduce(dst=g_f[0:1, 0:1], op=nl.add, data=t_f[0:1, 0 : c1 - c0], axis=1)
                nisa.tensor_tensor(
                    dst=g_f[0:1, 1:2],
                    data1=g_f[0:1, 0:1],
                    data2=t_f[0:1, c1 - c0 : c1 - c0 + 1],
                    op=nl.add,
                    engine=vector,
                )
                if merged:
                    # any' = (sum of the trips) + (count != 1): nonzero when a stream body runs or the chunk is 4 items.
                    nisa.tensor_tensor(
                        dst=g_f[0:1, 1:2],
                        data1=g_f[0:1, 1:2],
                        data2=m_flags[0][0:1, 1:2],
                        op=nl.add,
                        engine=vector,
                    )
                g_i = sb.tile((1, 2), int32)
                nisa.tensor_copy(dst=g_i, src=g_f, engine=vector)
                for k, name in enumerate(("prefix", "any")):
                    groups_ifs[name] = nisa.register_alloc()
                    nisa.register_load(groups_ifs[name], g_i[0:1, k : k + 1])
        elif trips_ahead:
            for size, trip_col, _ in bodies:
                trips[size] = nisa.scalar_load(info[0:1, 8 * b + trip_col : 8 * b + trip_col + 1])
        if ahead:
            b_ents, _, b_carry = seq_set(b)
            for size, _, row_col in bodies:
                nisa.dma_copy(
                    dst=b_ents[size][0:1, 0 : size * fields],
                    src=stream_rows(b, row_col, 0, size),
                    dge_mode=nisa.dge_mode.hwdge,
                    engine=nisa.engine.sync,
                )
            # ``chunks_first``: the chunk ahead of a stream is the next sequence's, or the last chunk ahead of the
            # first stream: its tail loads that stream's first pages.
            c_tail = chunk_tail(b, next_tail((b + 1) % seqs if first else b, None, 0))
            m_regs = flag_regs(m_flags[1]) if merged else None
            emit_chunk(b, chunk_carry, c_tail, flags=m_regs)
            thunks = []
            for i, (size, trip_col, row_col) in enumerate(bodies):

                def body(
                    step, size=size, row_col=row_col, b=b, last=i == len(bodies) - 1, b_ents=b_ents, b_carry=b_carry
                ):
                    emit_body(
                        b,
                        size,
                        False,
                        b_ents[size],
                        carry=b_carry,
                        tail=None if last else next_tail(b, row_col, size),
                    )

                if stream == "straight":
                    body(0)
                elif stream in ("ifs", "ifs_grouped"):
                    thunks.append(lambda body=body, reg=trips[size]: nke.if_else(reg, lambda: body(0)))
                else:
                    bound = (
                        trips[size]
                        if trips_ahead
                        else nisa.scalar_load(info[0:1, 8 * b + trip_col : 8 * b + trip_col + 1])
                    )
                    nl.fori_loop(0, bound, body)
            if stream == "ifs" or (stream == "ifs_grouped" and not groups_ifs):
                trees.append(lambda thunks=thunks: [thunk() for thunk in thunks])
                if not first:
                    trees.pop()()
            elif stream == "ifs_grouped":
                # IF any { IF prefix { the IFs of every body but the last }; IF last }: at short contexts the zero-trip
                # bodies are skipped in one branch (the whole stream, or the prefix ahead of the last body).

                def prefix_ifs(thunks=thunks):
                    for thunk in thunks[:-1]:
                        thunk()

                def stream_ifs(
                    b=b, carry=chunk_carry, tail=c_tail, regs=m_regs, thunks=thunks, ifs=groups_ifs, prefix=prefix_ifs
                ):
                    if regs is not None:
                        # The merged form: the 4-item chunk, then the stream, in the any-IF.
                        nke.if_else(regs[1], lambda: emit_body(b, nb, True, carry=carry, tail=tail))
                    nke.if_else(ifs["prefix"], prefix)
                    thunks[-1]()

                trees.append(lambda reg=groups_ifs["any"], stream_ifs=stream_ifs: nke.if_else(reg, stream_ifs))
                if not first:
                    trees.pop()()
            continue
        emit_chunk(b, chunk_carry, chunk_tail(b, None))
        if stream == "rolled":
            rolled_stream(b)
            continue
        for size, trip_col, row_col in STREAM_BODIES:
            if size > rows_per_seq:
                continue

            def body(step, size=size, row_col=row_col, b=b):
                if row_col is None:
                    rows32 = seq_rows[b].reshape((rows_alloc // 32, 32 * fields))
                    src = rows32.ap(
                        pattern=[[32 * fields, 1], [1, 32 * fields]], offset=0, scalar_offset=step, indirect_dim=0
                    )
                else:
                    src = seq_rows[b].ap(
                        pattern=[[fields, 1], [1, size * fields]],
                        offset=0,
                        scalar_offset=info[0:1, 8 * b + row_col : 8 * b + row_col + 1],
                        indirect_dim=0,
                    )
                nisa.dma_copy(dst=ents[0:1, 0 : size * fields], src=src)
                emit_body(b, size, False)

            bound = trips[size] if trips_ahead else nisa.scalar_load(info[0:1, 8 * b + trip_col : 8 * b + trip_col + 1])
            nl.fori_loop(0, bound, body)
    for tree in trees:
        tree()
    if page_runs:
        # sum x 1.0 keeps every bit; sum x NaN fails the sequences whose groups are not page runs (column g S + b).
        run_scale = plan["run_scale"]
        state_cols = state_flat.reshape((P, groups * seqs + 1, _STATE_WIDTH))
        for g in range(groups):
            sums = state_cols[0:P, g * seqs : (g + 1) * seqs, _KV_LORA:_STATE_WIDTH]
            nisa.tensor_tensor(
                dst=sums, data1=sums, data2=run_scale.reshape((P, seqs, 1)), op=nl.multiply, engine=vector
            )
