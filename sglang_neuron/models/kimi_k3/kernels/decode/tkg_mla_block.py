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

"""MLA multi-query attention body for Kimi-K3 speculative decode (TP32 = CP32, LNC1).

``mla_block_attention`` generalizes ``mla_decode_attention``: it runs one MLA attention layer for
``Q`` query rows of each of ``B`` sequences on one TP32 rank, in one call. It serves the DSpark
drafter (``local_heads=2``, 64 heads, GPT-J RoPE, a non-causal block of query rows) and the target
verify (``local_heads=3``, 96 heads, NoPE, a causal block with the sigmoid output gate).

The body keeps the phases of ``mla_decode_attention``: the input projection, the TP gathers, the
owner cache write, the paged attention over the prior cache rows, the CP state merge, the value and
output projections, and the TP all-reduce. The generalizations are:

- ``local_heads`` sets the head count (``32 * local_heads``), so the weight widths scale.
- ``rope_rows`` rotates ``q_pe`` and the current rows' ``k_pe`` with host cos/sin rows (GPT-J pairs).
- ``causal`` masks the ``Q x Q`` current block: a block mask (every query sees every current row) or
  a causal mask (query row ``j`` sees current rows ``0 .. j``). The prior cache rows are the pool rows
  below ``position[b]`` (row 0 of sequence ``b``), shared by every query row of the sequence.
- ``output_gate`` applies the sigmoid gate of the target; the drafter has no gate region.
- The ``Q`` current rows come from SBUF, never read back from the pool, so one call never writes a
  cache row that it later reads, so a read never races the owner write.

The attention reuses the shift of ``mla_decode_attention``: every rank shifts the scores of a query
row by its own current-key score plus 64.46, so the softmax needs no maximum, and each rank adds
``1/32`` of the current block, which the CP reduce-scatter sums back to one copy.
"""

import itertools
import math
from contextlib import ExitStack, nullcontext
from typing import NamedTuple

import nki.collectives as ncc
import nki.experimental as nke
import nki.isa as nisa
import nki.language as nl

from .utils.kernel_assert import kernel_assert
from .common import (
    StateParts,
    dma_priority,
    fp8_kernel_pages,
    fp8_page_dtype,
    fp8_page_max,
    is_trn3,
    load_state_bf16_o,
    mla_state_bf16_o_views,
    state_rows_plain,
    store_state_bf16_o,
    token_reduce_scatter,
)
from .tkg_weight_dma_gate import gate_weight_dma, record_pool_writes, release_weight_gate
from .tkg_sbuf_alloc import AUTO_SBUF, SbufTiles, SbufZone
from .tkg_mla_fp8_pv import fp8_pv_loop, fp8_pv_setup, fp8_pv_tables_ahead
from .tkg_mla_paged import (
    _DR_BLOCK_WORDS,
    _DR_BLOCKS,
    _E4M3_MAX,
    _GATHER_CARRY_GROUP,
    _GATHER_LEAD,
    _IN_WIDTH,
    _MX_PACK,
    _MX_UNIT_SCALE,
    _SOFTMAX_SCALE,
    LOOP_PAGE_GATHERS,
    MxKvB,
    MxOChunk,
    _absorb_mx_operand,
    _clamped_table,
    _dr_gather_tile_queries,
    _e4m3_query_rounding,
    _gather_constants,
    _gather_index,
    _instruction,
    _load_mx_kv_b,
    _o_proj_mx_chunk,
    _o_proj_mx_quantize,
    _pack_items,
    _packed_key_loop,
    _qkv_projection_mx,
    _value_mx,
    pack_p2_groups,
    pack_p2_runs,
    pack_shared_zone_bytes,
    pack_tail_chain,
)
from .tkg_mx_quant import (
    MX_COMPACT_ROWS,
    MX_LANES,
    MX_QUADRANT_ROWS,
    load_mx_compact_scales,
    load_mx_scales,
    mx_hT_shapes,
    mx_scale_view,
    quadrant_absorb_mx,
    quantize_hT_mx,
)

MLA_WEIGHT_NAMES = (
    "mla_in_proj",
    "mla_q_a_norm",
    "mla_q_b_proj",
    "mla_kv_a_norm",
    "mla_kv_b_proj",
    "mla_o_proj",
)
"""Per-layer kernel weights of the block body, in argument order. All are BF16.

The layout matches ``tkg_mla_paged.MLA_WEIGHT_NAMES`` for ``local_heads`` heads, except that
``mla_in_proj`` has no gate region when ``output_gate`` is off (the drafter). See
``mla_block_torch.mla_block_rank_weights`` for the host builder.
"""

_P_MAX = 128
_HIDDEN = 7168
_HIDDEN_TILES = _HIDDEN // _P_MAX
_TP = 32
_Q_LORA = 1536
_Q_LORA_TILES = _Q_LORA // _P_MAX
_Q_LORA_LOCAL = _Q_LORA // _TP
_KV_LORA = 512
_KV_LORA_TILES = _KV_LORA // _P_MAX
_ROPE = 64
_NOPE = 128
_V_HEAD = 128
_Q_HEAD = _NOPE + _ROPE
_CACHE_WIDTH = _KV_LORA + _ROPE
_CACHE_LOCAL = _CACHE_WIDTH // _TP
_KV_LATENT_LOCAL = _KV_LORA // _TP
_QKV_LOCAL = _Q_LORA_LOCAL + _CACHE_LOCAL
_GATE_LOCAL_MX = _IN_WIDTH - _QKV_LOCAL
"""The gate columns of the MXFP8 ``mla_in_proj`` of the target verify (3 local heads, the decode's layout)."""
_BLOCK = 128
_GROUP_BLOCKS = 4
"""Local blocks of one prior-key group: the scores of a group fill one FP32 PSUM bank."""
_MAX_STATIC_BLOCKS = _GROUP_BLOCKS
"""Local blocks of the static path, one key group per sequence."""
_MAX_UNROLLED_BLOCKS = 16
"""Local blocks of the unrolled dynamic path, at most. A larger table takes the loop path, which the
compiler can mis-allocate without explicit SBUF (see ``tkg_sbuf_alloc`` and ``tkg_mla_paged``)."""
_TRIMMED_WINDOW_BLOCKS = 3 * _GROUP_BLOCKS
"""Local blocks of the trimmed draft window (``trimmed_window``): 3 key groups of the static path."""
_BF16_KEY_MAX = 65536.0
"""Clamp bound of a BF16 cache row, so a stale masked key is finite (see ``tkg_mla_paged``)."""
_PAGE_WIDTH = _BLOCK * _CACHE_WIDTH
_GLOBAL_BLOCK_STRIDE = _TP * _BLOCK
"""Global positions between two local blocks of one rank."""
_LOOP_GROUPS = 8
"""Prior-key groups of one loop iteration, at most, with an FP8 pool."""
_BF16_LOOP_GROUPS = 4
"""Prior-key groups of one loop iteration with a BF16 pool."""
# ``loop_cast_engines`` (the page forms of ``tkg_mla_paged``): the most BF16 group tiles of one loop iteration (0: off).
ROWS_LOOP_CASTS = {"off": 0, "dma": _LOOP_GROUPS, "dma6": 6, "dma4": 4}
_TAIL_SIZES = (4, 2, 1)
"""The key groups of a tail iteration of the plain prior-key loop, largest first (``loop_tails``)."""
_FULL_TAIL = 8
"""The ``loop_tail_groups`` bit of the full-iteration tail: the last iteration of every sequence runs in a loop of its own
(all the groups of an iteration, or the smaller tail that holds them), so the main loop reads only prior keys and skips
the page clamp (``loop_tail_sizes``)."""
LOOP_TAILS = {"off": 0, "tail4": 4, "tail2": 2, "tail1": 1, "tail421": 7, "tail21": 3, "tail8": 8, "tail8421": 15}
"""The tail iteration forms of the plain prior-key loop (``mla_block_attention`` ``loop_tail_groups``): the sum of the
key groups of its tail iterations (``loop_tails``), one tail loop each; with the ``_FULL_TAIL`` bit ("tail8",
"tail8421") the last iteration of each sequence also takes a loop of its own when no smaller tail holds it."""


def loop_tails(tail_groups: int) -> tuple[int, ...]:
    """Return the key groups of each tail iteration of ``loop_tail_groups`` ``tail_groups`` (``LOOP_TAILS``) below a
    full iteration, largest first: 7 -> (4, 2, 1), 3 -> (2, 1), 4 -> (4,), 15 -> (4, 2, 1), 8 -> (), 0 -> ()."""
    return tuple(t for t in _TAIL_SIZES if tail_groups & t)


def loop_tail_sizes(tail_groups: int, loop_groups: int) -> tuple[int, ...]:
    """Return the key groups of each tail loop of ``tail_groups`` for iterations of ``loop_groups`` groups, largest first:
    the full iteration (``loop_groups``) with the ``_FULL_TAIL`` bit, then ``loop_tails``: 15 with 8 groups ->
    (8, 4, 2, 1), 8 -> (8,), 7 -> (4, 2, 1)."""
    return ((loop_groups,) if tail_groups & _FULL_TAIL else ()) + loop_tails(tail_groups)


def rows_cast_groups(form: str, group_count: int, loop_group_cap: int = 0) -> int:
    """Return the key groups of one prior-loop iteration with the page cast form ``form`` (``loop_cast_engines``): the
    largest count of at most ``ROWS_LOOP_CASTS[form]`` (and ``loop_group_cap``, if set) that divides ``group_count``,
    the key groups of the table, and is even or the whole table, so the iterations cover the table and keep the group
    pairs of the loop with the cast off. The 1M table (256 local blocks, 64 groups) gives 8 for "dma" and 4 for "dma6"
    and "dma4"."""
    cap = ROWS_LOOP_CASTS[form]
    cap = min(cap, loop_group_cap) if loop_group_cap else cap
    for groups in range(min(cap, group_count), 0, -1):
        if group_count % groups == 0 and (groups % 2 == 0 or groups == group_count):
            return groups
    return 0


_GROUP_STRIDE = _GROUP_BLOCKS * _GLOBAL_BLOCK_STRIDE
"""Global positions between the starts of two key groups of one rank."""
_LORA_NORM_EPS = 1e-5
"""Default epsilon of the q_a and kv_a RMSNorms (``lora_norm_eps`` of ``mla_block_attention``)."""
_MASK_BIAS = -1.0e30
"""Score bias of a masked key. The exponential of a biased score is exactly 0."""
_CURRENT_WEIGHT_LOG2 = -93
_MERGE_OFFSET = -_CURRENT_WEIGHT_LOG2 * math.log(2.0)
"""Offset of the softmax reference below the scaled current score, so the own current key weighs
``2**-93`` (see ``tkg_mla_paged``)."""
_WEIGHT_CAP = 60.0
"""Upper bound of the exponent of a key weight, so no weight overflows."""
_CAP_WEIGHT = math.exp(_WEIGHT_CAP)
"""The weight at the cap: the capped score gives ``exp(_WEIGHT_CAP)`` (see ``_Softmax.set_reference``)."""
_TS_GROUPS = 2
"""Key groups that one call of ``_attend_group_rows`` attends (the numerator adds once for all their blocks)."""
_SUMS_LAG = 3
"""Key blocks whose score matmuls run before the sums matmul of a block (``rows_pair_lag``)."""
_REF_PARTS = 3
"""BF16 parts of the folded reference (``folded_reference``): about 24 bits of each FP32 reference value."""
_REF_ROW = 96
"""First partition of the folded reference rows: a quadrant start, so one 3-partition op can write them."""
_RANK_SCALE = 2.0**-5
"""Each rank adds ``1/32`` of the current block; the CP reduce-scatter sums the 32 identical copies."""
_RANK_CURRENT_WEIGHT = 2.0 ** (_CURRENT_WEIGHT_LOG2 - 5)
"""Capped own-key weight of the bitwise verify, ``1/32`` of the ``2**-93`` current weight (see ``tkg_mla_paged``)."""
_QUADRANT = 32
_BIAS_ROW = _ROPE
"""Contraction row of the rope score matmul that adds the prior-key mask; the rope query holds 1.0."""
_MAX_TOKENS = _P_MAX // _TP
"""Query rows of one gather tile, at most: 32 ranks of 4 rows fill the 128 partitions."""
_TAIL_ROWS = 16
"""Query rows of one output-tail pass, at most. The state and its merge fit SBUF at B up to 8 when the
tail runs in passes of this many rows. It also caps the state reduce-scatter message at
``16 * 513 * 4 = 32832`` bytes: a message of 64 KiB or more makes the runtime mesh builder exceed its
reduce-scatter subtype limit (``32 * 513 * 4 = 65664`` bytes failed on Trn3 at verify B=4)."""
_DEBUG_ROW_SUMS = None
"""Test hook (None in a served program): a list. When set, each packed-prior pass appends ``(tag, sums)``: an HBM tensor
``[parts, columns]`` FP32 with the weight sum of each state column of the pass after its loop (the state's sum column).
The caller returns the tensors as extra kernel outputs; their maximum is the margin of the capless check (a pass redoes
its loop when a sum exceeds ``_CAPLESS_SUM_LIMIT``)."""
_TABLES_AHEAD_TEST = None
"""Test hook (a list, or None): the tag of every pass that loads the next pass's tables (``fp8_pv_tables_ahead``)."""
_PACKED_PRIOR_OPTIONS = {
    "slots": 32,
    "tail_slots": 4,
    "prefetch": 2,
    "page_runs": 2,
    "latent_act": 2,
    "carry_main_only": True,
    "sum_after_copy": True,
    "dr_keys": False,
    "item_rope_ahead": False,
    "p2": False,
    "psum_per_block": False,
    "own_key_select": True,
    "hoist_plans": True,
    "q_fp8": False,
    "pad_skip": False,
    "null_copy_vector": True,
    "null_runs": True,
    "deep_pipe": False,
    "mx_keys": False,
    "sum_batch": True,
    "pad_runs": False,
    "step_items": False,
    "stage_rows_sbuf": False,
    "own_key_selects": False,
    "cap_first": True,
    "clamp_pin": True,
    "numerator_ring": 0,
    "clamp_tail_only": False,
    "row_depth": None,
    "word_keys": False,
    "slot_rows_sbuf": False,
    "keys_ahead": True,
    "mx_qk": False,
    "p2_entry_act": False,
    "setup_off_pool": False,
    "skip_stage_store": False,
    "mx_natural": False,
    "act_spread": False,
    "kv_pitch576": False,
    "kv_gather": False,
    "kv_rem4": False,
    "kv_group": 0,
    "fp8_pv": False,
    "own_state_stack": False,
    "fp8_pv_lookahead": 0,
    "fp8_pv_trips_ahead": False,
    "fp8_pv_chunk_ahead": False,
    "q_mx_gather": False,
    "q_mx_pass_prefetch": False,
    "q_mx_scale_words": False,
    "own_rows_ahead": False,
    "own_state_first": False,
    "fp8_pv_qk_group": 0,
    "fp8_pv_null_skip": False,
    "tail_o_proj_first": False,
    "state_sums_act": False,
    "state_o_act": False,
    "state_o_heap": False,
    "state_sums_heap": False,
    "fp8_pv_refs_after_q": False,
    "fp8_pv_refs_act": False,
    "state_sum_blocks": False,
    "state_sums_pre_if": False,
    "state_sum_blocks_pass": False,
    "state_dense_tail": False,
    "fp8_pv_value_form": 0,
    "fp8_pv_page_runs": 0,
    "fp8_pv_stage_free": False,
    "fp8_pv_clamp_split": False,
    # True: the score cap of the capped FP8 P@V loop runs on GpSimdE after the exp, not on VectorE (fp8_pv_loop
    # cap_pool; the same bits). VectorE is the busiest MLA engine of the no_if form.
    "fp8_pv_cap_pool": False,
    "fp8_pv_prep_late": False,
    "fp8_pv_page_buffers": 2,
    "fp8_pv_chunks_first": False,
    "fp8_pv_step_store": False,
    "fp8_pv_pages_first": False,
    "fp8_pv_tables_ahead": False,
    "q_pass_rows": False,
    "q_prep_act": False,
    "fp8_pv_stream": "loops",
    "fp8_pv_plan_first": False,
    "fp8_pv_plan_reduce": False,
    "q_b_scales_scoped": False,
    # True: no runtime IF in the packed loop. The p2 passes run the capped loop only, with no capless loop and no redo IF
    # (bitwise the same). verify_model_2srv.served_pack_options also gives the FP8 P@V stream its loops and no null
    # skip. The released neuronx-cc 2.28 stops on a runtime IF ([NCC_ICFG019]).
    "no_if": False,
    # True (with no_if): the p2 passes run the capless loop and the weight-sum check again, and the redo is a loop of 0
    # or 1 trips on the check flag, not an IF (_capless_redo as_loop). The released neuronx-cc 2.28 compiles a loop with
    # a runtime trip count. Bitwise the capped loop. The loop without the score cap has no VectorE minimum in each
    # sub-slot, and VectorE is the busiest MLA engine of the capped loop.
    "no_if_loops": False,
    # True (with fp8_pv): the capless redo of the FP8 P@V pass runs its stream as one loop of 4-item bodies
    # (fp8_pv_loop stream "rolled", no lookahead), not the 32-, 16-, 8- and 4-item bodies of the pass. The redo seldom
    # runs, so its program can be small: the verify NEFF and its compile time grow less. Bitwise the same.
    "fp8_pv_redo_rolled": False,
    # True (with no_if): verify_model_2srv.served_pack_options gives the FP8 P@V stream the "batched" form in place of
    # its per-sequence loops: one 0/1-trip loop for each body size of the pass, with the bodies of all sequences in it
    # (fp8_pv_loop stream "batched"; the same bits up to the sign of an exact zero). Each loop boundary of the released
    # neuronx-cc 2.28 is 4 all-engine barriers. At 1M a pass then has 4 loop boundaries in place of 16.
    "fp8_pv_batched": False,
    # True (with no_if): verify_model_2srv.served_pack_options gives the FP8 P@V stream the "exact" form: the stream
    # rows that every sequence of the pass has in shared rolled loops (16-item and 4-item bodies of all sequences), then
    # the other rows of each sequence in rolled loops of its own, inside one loop of 0 or 1 trips (fp8_pv_loop stream
    # "exact"; the same bits). No item runs past the stream of its own sequence, so a pass of mixed contexts does no
    # masked work. A pass has at most 3 + 2 S loops in place of 4 S. Not with fp8_pv_batched.
    "fp8_pv_batched_exact": False,
    # True (with fp8_pv): the rope byte copy (uint8) of each FP8 P@V item prep runs on VectorE, not on ScalarE
    # (fp8_pv_loop rope_copy_vector). ScalarE runs the exp of every sub-slot and is the busiest MLA engine of the capless
    # loop with 4 sequences a pass; VectorE has slack there. The same bytes on either engine.
    "fp8_pv_rope_copy_vector": False,
}
"""The packed-loop form of ``packed_prior`` (the slot form of the served 1M-context batch-16 decode): 32 items an iteration with the rounded
tail loops of ``tkg_mla_paged.pack_tail_chain`` for a tail of 4, a prefetch of 2, page runs of 2, 2 ScalarE latent key
copies, the pipeline carried in the main loop only, and the sum add after the tile-0 key copy. ``packed_prior_options``
overrides these keys only. On by default: ``own_key_select`` and ``hoist_plans`` (below), ``cap_first`` and
``clamp_pin`` (below; where the loop takes them: Q > 1 rows, for the pin Q >= 3 and the page clamp, also with ``p2``),
and ``null_copy_vector`` (the
null-entry page fix of the zeroed pages on VectorE, ``tkg_mla_paged._pack_layer_items``) and ``null_runs`` (a padded sequence
counts as page runs). Opt-in keys (default off): ``dr_keys`` (the GPU decode's e4m3 query with FP8 double_row QK
keys, as the decode's ``pack_dr_keys``: the rows' queries, rope queries and own-key scores take the e4m3 query, so each
row is bitwise its form-B decode step, not the step without ``dr_keys``), ``item_rope_ahead`` (the rope keys of the next
item one sub-slot early; bitwise the same) and ``p2`` (the sub-slots of an item take 128 (row, head) columns, not the 96
heads of one row: Q x 96 / 128 sub-slots an item, ``tkg_mla_paged._packed_key_loop`` ``p2``; the state of the pass lives
in that layout and its store to HBM puts each column back at its head and row, so the bits are the same; with
``dr_keys`` each column takes the double_row stationary of its row, so each row is bitwise its ``dr_keys`` form; with
``item_rope_ahead`` the rope keys of the next item transpose in sub-slot G - 2 of an item), ``psum_per_block`` (the
loop PSUM tensors allocated in each top-level block that uses them, ``tkg_mla_paged._packed_key_loop`` ``psum_per_block``;
bitwise the same), ``own_key_select`` (the own-key numerator of each row comes from one matmul over all current rows
with a one-hot stationary, not a DMA that shifts the row to partition 0; the other rows add exact zeros, so the bits are
the same except the sign of an exact-zero numerator entry) and ``hoist_plans`` (the caller builds the item plan of each
pass once for the MLA layers of a step, ``mla_pass_plans``, and gives it as ``mla_block_attention`` ``pass_plans``; the
plans are those each pass builds, so the bits are the same). ``q_fp8`` (with ``dr_keys`` or ``p2``, and ``q_hbm``)
rounds the queries of the call to e4m3 (clamped to +-448, the rounding of ``dr_keys``) before the one query all-gather
of the call, so the gather moves half the bytes; the tile loads cast the e4m3 rows back to BF16 (exact). With
``dr_keys`` every query consumer takes the e4m3 query already, so the bits are those of ``dr_keys``. With ``p2`` every
query consumer (the latent and rope QK, the own-key score, ``mx_qk``'s query words) takes the e4m3 query at unit scale,
the query form of the GPU decode (accuracy class). The caller's ``q_call`` HBM pair is e4m3
(``mla_block_collective_hbm`` ``q_fp8``). ``q_fp8`` "latent" (with ``p2``, ``mx_qk`` and ``q_hbm``, no ``dr_keys``) rounds only
the latent (absorbed) query to e4m3 that way and keeps the rope query BF16: the rope QK of the p2 loop is a BF16 matmul, so
its e4m3 query was the error of ``q_fp8`` True, and the latent query of ``mx_qk`` is e4m3 already. The gather pair is
uint16 (``_q_call_layout``: the e4m3 latent bytes, then the BF16 rope words), so the all-gather moves the bytes as they
are, 1,920 bytes a row in place of 1,728 (e4m3) or 3,456 (BF16); each query load is a casting e4m3 load of the latent
part and a BF16 load of the rope part. ``pad_skip`` (opt-in) gives a padding sequence no item, also with
``clamp_tail_only``: its state keeps its own-key term only, a finite row; the other sequences keep their items and bits
(``tkg_mla_paged._pack_items``). With the runner's ``live_rows`` (``mla_block_attention``) the padding sequences are those
with mask 0 (the served form); without it, those at position 0 (TEST ONLY, NOT SERVABLE: a live row can sit at
position 0 in serving). ``step_items`` (with ``hoist_plans``) keeps
the item rows of each pass for the MLA layers of a step too: the first MLA layer of the step writes them (and the
page-run scale of each pass), every later one reads them and skips its page-id gathers (``tkg_mla_paged._packed_key_loop``
``step_items``; the caller gives every MLA layer the table of the first one, so the rows are the same bytes).
``stage_rows_sbuf`` loads the slot rows of the loop prologue from the SBUF stage tile, not from its HBM store
(``tkg_mla_paged._packed_key_loop`` ``stage_rows_sbuf``; the same bytes). ``slot_rows_sbuf`` (opt-in; with
``stage_rows_sbuf``) loads every later slot row of the loops from that tile too (``tkg_mla_paged._packed_key_loop``
``slot_rows_sbuf``; the same bytes). ``own_key_selects``
(with ``own_key_select``) writes the one-hot stationary of each row of a pass into its own slice of one pass tile, not
into one tile that every row reuses: the same values, so the bits are the same, and the select of a row does not wait
for the own-key matmul of the row before it. ``cap_first`` traces the state adds of each sub-slot's P@V after the score
cap of the next sub-slot, so on VectorE the cap does not queue behind them and the exp, which the weight transposes
wait for, starts earlier (``tkg_mla_paged._packed_key_loop`` ``cap_first``; the same adds in the same order, so the bits
are the same). ``clamp_pin`` (on by default) clamps the pages of the next item in two VectorE instructions of 2 pages, issued after the score caps
of rows 0 and 1 of the current item (``tkg_mla_paged._packed_key_loop`` ``clamp_pin``), not just before the last row, where
the score cap queues behind the clamp; bitwise the same. ``deep_pipe`` (opt-in; Q > 1 rows, not with ``p2``) runs the
weight transposes of sub-slot s - 2 and P@V of s - 3 after the QK of sub-slot s, not those of s - 1 and s - 2, so each
score cap and exp hides behind two sub-slots of TensorE work (``tkg_mla_paged._packed_key_loop`` ``deep_pipe``); the same
instructions on the same operands in the same order on each engine's state, so the bits are the same. ``mx_keys`` (opt-in; with ``dr_keys``, Q > 1 rows, not with ``p2``) runs the latent QK of each
sub-slot as one MX matmul with unit scales on the 32-bit key words of 4 page transposes an item, not 2 double_row
matmuls on the 2-byte words of 8 (``tkg_mla_paged._packed_key_loop`` ``mx_keys`` "single" with ``mx_q_rows``): the pass rows
take the MX query words of the e4m3 query from each gather tile (``_mx_query_words``) in place of the double_row
stationaries, and every other consumer of the query takes the e4m3 query of ``dr_keys``. The products are those of
``dr_keys``; the FP32 accumulation order differs, so the bits are not those of ``dr_keys`` (an accuracy-class form). With Q > 1 rows (not with ``p2``) the loop adds
the Q weight sums of an item with one VectorE TI add on its last row instead of one a sub-slot wherever it can
(``tkg_mla_paged._packed_key_loop`` ``sum_batch``); each sum element gets the same one add in the same order, so the bits are
the same. ``sum_batch`` (default True) is its kill switch: False keeps one sum add a sub-slot. ``pad_runs`` (opt-in) gives each padding sequence of a pass
the page-run scale 1.0 whatever its pages, by a select on its pad bit (``live_rows`` 0; without ``live_rows``, position
0), not by the pages: the runner's row-runs fill gives a pad row one page in every entry, which is no run, so without
it the sums of the pad row take NaN (``tkg_mla_paged._pack_layer_items``). A live sequence keeps its scale, so its bits
and its NaN check do not change. ``numerator_ring`` (opt-in, 0 or 3; the default key form, no ``dr_keys`` or
``mx_keys``): P@V of sub-slot s writes buffer s % 3 of a ring of 3 numerator PSUM tensors in place of a new tensor that
the compiler places, so it does not wait for the state adds of sub-slot s - 2 (``tkg_mla_paged._packed_key_loop``
``numerator_ring``); bitwise the same. ``clamp_tail_only`` (opt-in; with the page clamp): the plan moves the last key
group of each sequence into one block after the other items (each state column adds its groups in the same order), and
only that block's loop clamps its pages (``tkg_mla_paged._pack_items`` ``clamp_block``): every key of an earlier group is
below the sequence's position, so it is a row that an MLA cache writer wrote, which the clamp does not change; bitwise
the same. ``row_depth`` (opt-in, 8) gives the slot rows a ring of 8 tiles, not 4 (``tkg_mla_paged._packed_key_loop``
``row_depth``), so the row DMAs on the SP queue wait for the readers of 8 slots back; the same rows, bitwise the same.
``word_keys`` (opt-in; Q > 1 rows on FP8 pages, the default BF16 query: no
``dr_keys`` or ``mx_keys``, no ``deep_pipe`` or ``numerator_ring``) transposes the latent keys of an item with one 32-bit
TensorE transpose a page of byte-permuted rows (one ScalarE byte permute a page) in place of 4 FP8 transpose-mode
transposes a page, and each latent QK reads its byte lane at element step 4 (``tkg_mla_paged._packed_key_loop``
``word_keys``): the same keys and query tiles in each matmul, so the bits are the same. ``keys_ahead`` (on by default;
it applies with ``word_keys`` at 6 or more sub-slots an item, as ``p2`` at Q = 8) runs the next item's key-word
transposes and their ScalarE copy one sub-slot earlier, so the next item's first latent QK does not wait for them
(``tkg_mla_paged._packed_key_loop`` ``keys_ahead``); the same matmuls on the same operands, bitwise the same. False (not the
default) gives the program without it. ``mx_qk`` (opt-in, accuracy
class; with ``p2`` and ``word_keys``) runs the latent QK of each sub-slot as one MX matmul with unit scales on the
``word_keys`` key words in place of 4 BF16 matmuls (``tkg_mla_paged._packed_key_loop`` ``mx_qk``): "unit" (or True) rounds
the latent query to e4m3 at unit scales (saturated at 448, the GPU decode's rounding), "block32" quantizes it to MXFP8
with one E8M0 scale a 32-value block (``nisa.quantize_mx``, the flash prefill's ``fp8_query`` "mx"), in the stage rows;
the rope query and the own-key score keep the BF16 query. ``p2_entry_act`` (opt-in; with ``p2``) moves two kinds of
loop-entry DMAs to the ScalarE HWDGE queue: the softmax references of the stage rows go straight into their stage words
by 12 DMAs before the stage tile's zero fill (which skips them), not by 2 SBUF DMAs a run on the SP queue that alternate
with the VectorE copies of the runs; and with ``stage_rows_sbuf`` the prologue's slot rows do not queue behind the
stage rows' HBM store (``tkg_mla_paged._packed_key_loop`` ``p2_entry_act``); the same bytes, bitwise the same.
``setup_off_pool`` (opt-in; with ``p2``) moves pass-setup work off GpSimd: the passes and layers share one tile of the
batched weight-sum offsets, and the prologue's sum-tile zero fills and the stage rows' TI offset copies run on VectorE
(``tkg_mla_paged._packed_key_loop`` ``setup_off_pool``); the same values, bitwise the same.
``skip_stage_store`` (opt-in; with ``slot_rows_sbuf`` and ``p2``) drops the HBM store of the stage rows of each pass,
because every slot row then reads the SBUF stage tile (``tkg_mla_paged._packed_key_loop`` ``skip_stage_store``); the same
bytes, bitwise the same.
``mx_natural`` (opt-in, accuracy class; with ``mx_qk``) takes the key words of the FP8 key rows as they are (word ``c``:
features ``4c .. 4c + 3``), so the loop runs no ScalarE byte permute (4 an item) and has no permute buffers, and builds
the MX query words in that grouping from natural latent query tiles (``_mx_natural_query_tiles``: 4 strided TensorE
transposes and a copy a head of each gather tile; ``tkg_mla_paged._packed_key_loop`` ``mx_natural``). The feature-major
query tiles then serve only the own-key scores, in a frame that ends before the loop. The same products ("unit": the
same e4m3 query bytes; "block32": scale blocks of 32 consecutive features), summed in other 4-value groups and partition
order: not bitwise the permuted form.
``act_spread`` (opt-in; with ``word_keys``) orders the ScalarE work of each sub-slot of the loops of 4 or more items:
the weight copy of the sub-slot before it ahead of the exp, then at most one group of the item's byte permutes,
key-word copy and rope key copy after the exp (``tkg_mla_paged._packed_key_loop`` ``act_spread``; with ``mx_natural``
there are no byte permutes to place); the same instructions, bitwise the same.
With ``p2`` and ``word_keys`` (no ``mx_keys`` or ``dr_keys``) each pass runs its packed loop without the score cap
(``tkg_mla_paged._packed_key_loop`` ``capless``), then checks every column's weight sum and redoes the pass on the capped
loop when one is not at most e^59 (``_capless_redo``): bitwise the same in every case.
``kv_pitch576`` (opt-in; Q > 1 rows with ``clamp_tail_only``) gives the page buffers the 576 bytes of a cache row,
without the mask column: the main and tail loops, whose items are full key groups or null items, run without the mask
(64 rope rows; the null items take a negative reference that makes their weights 0), and the loop of the last key groups
writes its mask into a small tile that its rope transposes read (``tkg_mla_paged._packed_key_loop`` ``kv_pitch576``);
bitwise the same. ``kv_gather`` (opt-in; with ``kv_pitch576``) loads the pages of the main loop's items after the first
``prefetch`` with one GpSimd software DGE gather (DMA batching) for each 2 items, in place of their SP page DMAs
(``tkg_mla_paged._packed_key_loop`` ``kv_gather``); the same rows in the same tiles, bitwise the same. ``kv_rem4`` (opt-in;
with ``kv_pitch576``; accuracy class) loads each page onto 32 partitions, 4 consecutive rows (2,304 contiguous bytes) on
each, so every DMA piece is 2,304 bytes whatever the block table; the keys of each 128-key block change, so the PE
partition sums change (``tkg_mla_paged._packed_key_loop`` ``kv_rem4``). ``kv_group`` (opt-in, 0, 2, 4 or 8 = G; with
``kv_rem4``) gathers the pages of the main loop's items G .. in groups of G items (4 G pages, one GpSimd software DGE
gather each) into a ring of 2 G page buffers, 2 slots ahead (``tkg_mla_paged._packed_key_loop`` ``kv_group``; G 2 keeps the
4 page buffers of Q > 1 rows); the bits of ``kv_rem4``. ``fp8_pv`` (opt-in, accuracy class; with ``p2``, ``mx_qk``
"block32" and ``mx_natural``, Q > 1 rows, Trn3) attends the prior keys of a pass with the FP8 P@V pass
(``tkg_mla_fp8_pv.fp8_pv_prior``) in place of the packed loop: the scores with the keys on partitions, BF16 P
from one exp in the MX x4 layout, ``nisa.quantize_mx`` (an E8M0 scale for each 32 keys), one MX P@V and one MX weight sum
on the same FP8 P a sub-slot, 4 items a (sequence, group) accumulated in PSUM, in the fixed-reference frame of the p2
state. Where the packed loop runs capless, the pass runs without the score cap, then the same weight-sum check and redo
(``_capless_redo``) redoes the pass with the cap (a constant minimum of the shifted scores); elsewhere it takes the cap.
The pass builds its own item plan; the hoisted plans and step items stay unused. False keeps the program.
``own_state_stack`` (opt-in; with ``p2``, no ``pass_rows_ahead``) writes the own-key state of each group of 128 (row,
head) columns with one matmul over the 4 current rows of its period, not one matmul for each run (``_own_key_state_p2``
``stack``): 12 TensorE passes in place of 24 for a pass of 2 sequences of 8 rows. Each column gets the one product of
its row plus exact zeros from the other rows, and the state copy adds +0.0, so a sum of zeros is +0 as with the per-run
matmuls: the bits are the same.
With ``fp8_pv``: ``fp8_pv_lookahead`` (opt-in, 2 or 4 items; 0 keeps the program) loads the item rows of each stream
loop of a sequence once, before its chunk, on the SP HWDGE queue, and the tail of each body loads the pages of the
next stream body's first items, so they are in SBUF before the loop boundary's fence (``fp8_pv_loop`` ``lookahead``;
the same pages, the same bits); ``fp8_pv_trips_ahead`` (opt-in) loads the trip counts of a sequence's stream loops
before its chunk, so each block between two loops holds only its zero-trip branch (``fp8_pv_loop`` ``trips_ahead``;
the same bits); ``fp8_pv_chunk_ahead`` (opt-in) loads the first pages of the next sequence's chunk during a chunk, into
buffers of their own (``fp8_pv_loop`` ``chunk_ahead``; the same bits); ``fp8_pv_qk_group`` (opt-in, 2 or 4 pages; 0
keeps the program) issues the MX QK of that many pages of a sub-slot before their rope matmuls, in program order, so
fewer QK weight loads follow a rope matmul (``fp8_pv_loop`` ``qk_group``; the same matmuls in each PSUM accumulation,
the same bits); ``fp8_pv_null_skip`` (opt-in) orders each chunk's real items before its null items and runs a 1-item
chunk for a sequence with 3 null items (every sequence at 16K) in place of the 4-item chunk (``fp8_pv_setup``
``null_last``, ``fp8_pv_loop`` ``null_skip``; forward then-only IFs on the valid count, 2: the merged form, the 4-item
chunk inside the stream's any-IF; the same terms in the same
order, the same bits up to the sign of an exact-zero numerator).
``fp8_pv_value_form`` (opt-in, 4 or 5; 0 keeps the
program) splits each item's value-word copy into 4 VectorE pieces placed after the quantize of 4 sub-slots
(``fp8_pv_loop`` ``value_form``; the same bits); ``fp8_pv_page_runs`` (opt-in, 2; 0 keeps the program; with
``page_runs`` 2, the same table contract) loads each item's 4 pages as 2 runs of 2 consecutive kernel pages, and a
sequence with a group that is not made of runs gets a NaN weight sum, as the packed loop's ``page_runs``
(``fp8_pv_loop`` ``page_runs``); ``fp8_pv_stream`` (with ``fp8_pv_lookahead``):
"loops" (the program), "ifs" (each 0/1-trip stream loop as a forward then-only IF on its trip register; the same bits),
"ifs_grouped" (those IFs nested in IFs on their zero-together groups, so a short context skips them in one branch)
or "straight" (every stream body unconditionally: a 1M test form only, NOT servable) (``fp8_pv_loop``
``stream``). With ``fp8_pv`` and ``step_items`` the FP8 P@V item plan of each pass (item rows,
stream rows, run scale) is built in the first MLA layer of the step only and read in every later one, from the step store
of ``mla_pass_plans`` (``fp8_pv_setup`` ``step``; every MLA layer of a step reads the same table, so the same bits).
``fp8_pv_plan_first`` (opt-in) puts the input loads of the pass's item plan and its HBM round trip on the ACT
hardware DGE queue, so they do not queue behind the query loads of the in-order SP queue, which wait for the query
all-gather (``fp8_pv_setup`` ``plan_first``; the same plan, the same bits). With ``step_items`` it acts where the plan
is built (the first MLA layer of the step).
``fp8_pv_plan_reduce`` (opt-in, with ``fp8_pv``; False keeps the program) sums each value of the item plan over the S
sequences of the pass with one reduce: a field select of a row's sequence, the sequence of a stream row, the start of
its sequence and the E_b of the seq info (the stream starts are one reduce each in every form). The plan then has no
memset of a tile followed by a chain of adds into that tile, which the compiler takes as an order-free reduction (no
wait between the adds) (``fp8_pv_setup`` ``plan_reduce``; the same small-integer terms, so the same item rows and the
same bits).
``fp8_pv_stage_free`` (opt-in, with ``fp8_pv``; False keeps the program) builds the FP8 P@V setup without the p2 stage
tile (~24 KB a partition, its zero fill, TI copies and null row): the MX query words and scales, the references and the
rope + bias rows go straight into the tiles that read them, and in a pass after the first the reference DMAs wait for
the state stores of the pass before (``fp8_pv_setup`` ``stage_free``; the same values, the same bits).
``fp8_pv_clamp_split`` (opt-in, with ``fp8_pv``) clamps each page of a chunk item alone, so the key transposes of a page
wait for its clamp only (``fp8_pv_loop`` ``clamp_split``; the same bits). ``fp8_pv_prep_late`` (opt-in, with ``fp8_pv``)
runs each sub-slot's item preps after its scores and P@V (``fp8_pv_loop`` ``prep_late``; the same bits).
``fp8_pv_page_buffers`` (opt-in 3, with ``fp8_pv``; 2 keeps the program) gives the FP8 P@V pass a third page buffer, so
an item's page load waits for the item three back, not two, and issues one item earlier (``fp8_pv_loop``
``page_buffers``; the same bits).
``fp8_pv_chunks_first`` (opt-in, with the lookahead and the grouped IF stream) runs the chunk of every sequence of an
FP8 P@V pass ahead of the first sequence's IF tree, so the block after each any-IF join holds only the next branch
(``fp8_pv_loop`` ``chunks_first``; the same bits).
``fp8_pv_step_store`` (opt-in, with ``fp8_pv``)
builds the FP8 P@V item plan of each pass once a step without the hoisted plans of the packed loop: ``pass_plans`` holds
only the step store of each pass (``mla_fp8_pv_step_stores``), which the first MLA layer of the step fills and the later
ones read (``fp8_pv_setup`` ``step``; every MLA layer of a step reads the same table, so the same bits).
With ``fp8_pv_plan_first`` the plan-first queue acts in the first MLA layer, where the plan is built.
``fp8_pv_pages_first`` (opt-in, with ``fp8_pv_stage_free``) puts the FP8 P@V setup's chunk entries and thresholds in
heap tiles (released after the pass's loops) and keeps the frame rows' tiles in the setup's frame (``fp8_pv_setup``
``pages_first``): the chunk loads do not wait for the readers of the own-key state's freed bytes, and the loop's first
page loads do not wait for the frame rows, whose freed bytes the page tiles took. The same instructions on other
bytes, so the bits are the same.
``fp8_pv_tables_ahead`` (opt-in, with ``fp8_pv_pages_first`` and ``fp8_pv_step_store``; a call of two full passes, in
the MLA layers after the first of the step) loads the second pass's chunk entries and thresholds from its step store at
the first pass's setup, and the pages of chunk items 0 and 1 of its first sequence at the tail of the first pass's last
chunk, into heap tiles (``fp8_pv_tables_ahead``, ``fp8_pv_loop`` ``next_chunk``), all on the SP hardware DGE queue in
the first pass's static code, before its stream's IF fences. The second pass's setup and first chunk take them
(``fp8_pv_setup`` ``tables``, ``fp8_pv_loop`` ``first_carry``), so its chunk tables and first pages do not queue
behind the first pass's state stores on the in-order SP queue; the tiles return after the second pass's loops (release
order: its setup heap, the ``state_o_heap`` tile, then these). The same bytes reach the same instructions, so the bits
are the same.
The MLA front and tail (opt-in, bitwise): ``q_pass_rows`` (with the split query gather, ``q_split``) orders the rows
of the q path by pass at the q_a copy, so the rows of a pass sit on adjacent partitions and the q-call store of a pass
is one DMA, not one for each query row; the same per-row arithmetic and the same gathered rows (not with
``q_mx_gather``, whose query store takes the call's row order). ``q_prep_act`` moves
query-prep copies of the gathered tiles from VectorE to ScalarE: the natural MX copies of a head take the engine
opposite to its latent copy, and the rope copy takes ScalarE; the same values (exact BF16 copies).
``q_mx_gather`` (opt-in; with ``fp8_pv`` and ``q_hbm``): each rank makes the MX query words and E8M0 scales of its
own heads before the query all-gather, and the gather moves them (``_q_call_layout`` "mx": 1,980 bytes a row in place of
3,456) with the BF16 rope query and the own-key score of each (row, head) (``_q_mx_store``); each pass loads them as
bytes (``_q_mx_load``) and takes no gather tile, PE transpose, interleave copy or ``nisa.quantize_mx`` of the query. A
scale block is 32 latent features of one (row, head) column, so the words and scales are those that the receiver made
from the gathered BF16 rows, and the owner's own-key score takes the operands and accumulation order of ``_own_key``:
the bits are the same. The KV path then runs before the query store (the own-key scores need the current keys).
``pass_q_prefetch`` and ``q_gather_buffers`` act on the BF16 gather tiles, which this form does not have, so the key
refuses them (0 and 1 only), and ``pass_rows_ahead`` too. Every write of a later pass's MX query loads waits for the
state stores of the pass before (a runtime dependency, as ``_own_key_state_p2``). ``q_mx_pass_prefetch`` (opt-in, with
``q_mx_gather``) loads the MX query bytes of each later full pass after the loop of the pass before, ahead of its state
stores, into one set of heap stage tiles (``_q_mx_stage``): the pass then runs only the copies of ``_q_mx_load`` from
them, and its first copy waits for the state stores of the pass before (a runtime dependency) in place of the first
load. The same DMAs and copies on the same bytes, so the bits are the same. ``q_mx_scale_words`` (opt-in, with
``q_mx_gather``) packs the E8M0 scales into the word records of the gather pair (``_q_call_layout`` "mx_packed",
``_q_mx_regions`` ``packed``): each (partition, head) record holds the partition's 4 n word bytes, then n scale bytes
(the scale row of that partition, pad bytes on the partitions without one), so one DMA loads the words and the scales
of a pass; the four scale DMAs of 384 16-byte elements, which the DGE puts on the first four engines of the queue ahead
of the words, go away. The gather moves 2,316 bytes a row in place of 1,980. The same bytes land at the same positions
of the stage tiles' words and scales; a pad byte reaches only a partition of the query scales that holds no scale row,
which the MX matmul does not read (the stage bytes there that stay unwritten without this option), so the bits are
the same.
``q_b_scales_scoped`` (opt-in, with ``mla_q_b_proj_scale``) puts the scale copies of the MXFP8 q_b
(``_load_mx_q_b``) in the projection scope (stack tiles), not on the heap of the region: Step 13 is their only reader
and it runs in that scope, so they free before the attention with the q_b weight. The loop frame then has 1,152 bytes a
partition more (2 copies of ``[128, 1, 576]`` uint8). The same DMAs and matmuls on the same bytes at other addresses, so
the bits are the same.
``own_rows_ahead`` (opt-in, with
``q_mx_gather``, ``p2`` and ``own_state_stack``) loads the stack-form current rows of every pass
(``_own_rows_stack_p2``) once a call, after the query store and before the passes, into heap tiles, and each pass's
``_own_key_state_p2`` takes them (``stack_rows``) in place of its own loads: no pass waits on a current-row load behind
the query gathers' DMAs. The same rows into the same operands, so the bits are the same.
``own_state_first`` (opt-in, with ``own_rows_ahead``; False keeps the program) traces each pass's own-key state
(``_own_key_state_p2`` on the rows loaded ahead: the run masks, the matmuls and the state copies) right after the pass's
state tile, before its MX query loads and copies: it reads only the current rows and the own-key weight, so the
engines can run it during the query all-gather, not behind the query copies on the in-order VectorE queue. The same
instructions on the same bytes in another trace order, so the bits are the same.
With one state reduce-scatter a call and ``tail_weights_after_rs``: ``tail_o_proj_first`` (opt-in) puts the gate exp
of each tail pass after the last o_proj weight DMA in the schedule (a schedule edge, no wait), so all o_proj chunk DMAs
issue at the end of the numerator reduce-scatter. Without it the scheduler can put the later chunk DMAs behind the gate
exp on the ScalarE queue, which waits on the state loads, and the DMA engines idle between the chunks. The same
instructions, so the bits are the same. With ``p2`` and the BF16-o state: ``state_sums_act`` (opt-in) stores the sums of
each pass on the ScalarE hardware DGE queue and the numerators on the SP queue (``_store_state_p2_bf16_o``
``sums_engine``), so the numerator stores do not wait for the semaphores of the sum stores, and the sums
reduce-scatter can start when the sums land. The same bytes, so the bits are the same. ``state_o_act`` (opt-in, with
``p2`` and the BF16-o state) stores the numerators of each pass that another pass follows on the ScalarE hardware DGE
queue too (``_store_state_p2_bf16_o`` ``o_engine``): on the SP queue they hold its completion semaphores, so the next
pass's query bytes, item tables and first pages, which that queue loads right after them, issue and land
later (MEASURED on the served k7 form). The same bytes, so the bits are the same. ``state_o_heap`` (opt-in, with ``p2``
and the BF16-o state) puts the BF16 numerator tile of a pass that another pass follows on the heap of the region
(12 KiB a partition at C = 16), released after the next pass's loops: on the stack the next pass's setup tiles and page
buffers take its bytes, so every one of them waits for the numerator stores. The next pass then waits only for the
numerator cast and the sum stores (the state tile, whose bytes its own state tile takes). The same bytes, so the bits
are the same. ``state_sums_heap`` (opt-in, with ``state_o_heap``) copies that pass's FP32 sums into a heap tile of
their own (VectorE, right after the cast; released with the numerator tile) and stores them from it, so the next pass
waits for the cast and the copy, not for the sum stores (on the ScalarE queue with ``state_sums_act``, which its own-key
state copies and reference loads otherwise wait to drain). The same words, so the bits are the same.
``fp8_pv_refs_after_q`` (opt-in, with ``q_mx_gather`` and ``fp8_pv_stage_free``) orders the stage-free setup's reference
DMAs after the pass's MX query loads on the SP queue (a schedule edge, no wait): the scheduler had put them between the
query words and the query scales, so the scale loads that the first QK waits for issued later. The same
DMAs, so the bits are the same. ``fp8_pv_refs_act`` (opt-in, with ``fp8_pv_stage_free``; with
``fp8_pv_refs_after_q`` or without it) issues those reference DMAs on the ScalarE hardware DGE queue in place of SP: in
the first pass's entry ScalarE's stream is idle while SP issues the query scale loads (MEASURED), and on SP the
scheduler had put the pass's loop register loads ahead of the last reference DMAs. The same DMAs on another queue, so
the bits are the same. ``state_sum_blocks`` (opt-in, with ``p2``, the BF16-o state, the call-level state reduce-scatter and its
sums pair) stores the sums of a call once, in whole 64-byte lines: the sums of a (partition, head slot) of the p2 layout
(head h, row j % 4; rows j and j + 4 share the partition) go to one 16-word block of head h's sums region
(``_sum_block_word``), each pass's VectorE copy puts its sums into a block tile, and the last pass stores the blocks with
one DMA a run of 4 rows (6 in all) in place of 12 sum DMAs a pass of 8-byte runs, each a read-modify-write of its line.
The tails read each head's rows from the blocks. Each rank's part of the sums reduce-scatter keeps its heads, so it adds
the same values; the bits are the same. The key applies to the calls with ``p2``, the BF16-o state and the call-level
state reduce-scatter with its sums pair padded to 64 words a head (``cc_pad256``), whole passes, Q a multiple of 4 and
no split or ``query_rows``; it leaves every other call as it is. ``state_sums_pre_if`` (opt-in, with
``state_sum_blocks``) moves the last pass's sums copy into the call's blocks and the block stores to right after its
uncapped FP8 P@V loop, before the capless check and its redo IF, and traces them again at the end of the redo body
(after the capped loop; DMAs and a VectorE copy, no collective). The sums reduce-scatter still issues after the IF, so it
reads the stores of the path taken: the same words as the stores after the IF without this option, so the bits are the
same. MEASURED (a profile of the served 93-layer C16 verify, 78 MLA calls): after the IF the block stores wait for
ScalarE's skip of the redo body, and the sums reduce-scatter starts late after the check (its last store descriptor
lands well after the first issue), with the numerator reduce-scatter queued behind it on the same collective stream.
It applies to the passes that store the blocks (the last pass of a call with ``state_sum_blocks``) in the capless form
of the FP8 P@V pass; it leaves every other pass as it is. ``state_sum_blocks_pass`` (opt-in, with ``p2``, the BF16-o
state and its sums pair padded to 64 words a head, ``cc_pad256``) gives a call of ONE pass with the per-pass state
reduce-scatter (no ``state_call``) the sum blocks of ``state_sum_blocks``: the pass copies its sums into the block
tile and stores the blocks with one DMA a run of 4 rows (6 in all) in place of one 8-byte read-modify-write DMA a run
(12 a pass), and the tail reads each head's rows from the reduced blocks. Each rank's part of the sums reduce-scatter
keeps its heads, so it adds the same values; the bits are the same. MEASURED (a profile of the 93-layer C8 verify, 138
MLA calls): those 12 sum stores take SP issue time after the capless IF, and the numerator stores wait behind them.
With ``state_sums_pre_if`` the blocks are stored before the capless check, as on the call-level form.
``state_dense_tail`` (opt-in, with the BF16-o state) gives the output tail of a call of ONE pass with the per-pass state
reduce-scatter the dense local state of the call-level tail (``_output_tail`` ``dense_state``: one row of a head on
each partition), so the normalize runs on heads x rows partitions in place of the heads and the latent takes 4
transposes in place of 64 (``_value_projection_dense``: the same multiply on the same values, then moves, so the same
bits). MEASURED (a 12-layer C = 8 profile with the sum blocks, 1M and 16K, 3 MLA layers): the head-major normalize is one
VectorE op on 3 partitions and its 128 transpose instructions take TensorE time, both between the end of
the numerator reduce-scatter and the value matmuls; the call-level dense normalize takes much less time."""
_STATE = _KV_LORA + 1
"""FP32 words of one state column: the 512 numerator words and the sum."""
MLA_PASS_ROWS = _TAIL_ROWS
"""Query rows of one attention pass of ``mla_block_attention``: a call with more rows runs in passes."""
_MAX_SEQS = 8
_MAX_FIT_SEQS = 16
"""Sequences of one call with ``fit_passes``: its sequence-major passes size the state by rows, not sequences."""
PASS_Q_PREFETCH_MAX_TILES = 2
"""The most prefetch tiles of ``pass_q_prefetch`` (``mla_block_attention``). Each tile holds one query gather tile (4 rows,
3,456 bytes a partition at 3 local heads) for the whole call. In the served k7 verify (C = 16, two passes, with the
step items) the second pass's packed loop has 9,692 bytes a partition free: 2 tiles leave 2,780 bytes, and 3 tiles
(10,368 bytes) do not fit."""
_Q_SPLIT_TEST_OFF = False
"""TEST ONLY: True keeps one query gather a call where ``q_split`` would split it (the A side of an A / B
test of the query split). False, the default, keeps the program."""
GATE_BEFORE_STATE = True
"""The call-level state reduce-scatter tail (``state_call``: C = 16, two passes) traces the output-gate projection of
every tail pass right after the tail weight loads, before any state load, and pins the first TensorE instruction of
each pass's value projection after the last gate matmul of the call (a scheduling-only edge), so the gate's MX matmuls
run while the sum reduce-scatter flies, not after the state loads. The same instructions on the same operands, so the
bits are the same. MEASURED on a 12-layer C = 16 verify with the "E" options (1M and 16K): without the pin, the gate ran after the end of the
reduce-scatters, after the value transposes, on the post-loop path (a longer post-loop path than without those options,
where it ran under the sum reduce-scatter). False (the kill switch) gives the program without the pin."""
GATE_PIN_ALL_TILES = True
"""With ``GATE_BEFORE_STATE``, every TensorE transpose of a pass's value projection (all latent tiles, all rows) waits
for the last gate matmul of the call (scheduling-only edges), not only the first one. With the first one pinned alone,
the transposes of latent tiles 1 to 3 queued ahead of the gate on the in-order TensorE queue and waited for the state
load, so the gate ran after the state reduce-scatters, on the post-loop path (MEASURED, a C = 16 12L profile, layers 3 /
7 / 11: TensorE idle under the reduce-scatters, gate after them). The same instructions on the same
operands, so the bits are the same. False (the kill switch) pins the first transpose only."""
SUM_RS_FIRST = True
"""The BF16-o state store of a ``p2`` pass (``_store_state_p2_bf16_o``) issues every sums run before the numerator runs,
and ``_bf16_o_reduce_scatters`` issues the small FP32 sums reduce-scatter before the BF16 numerator one. The two run on
one collective stream in issue order; with the sums stored first, the sums reduce-scatter flies while the numerators
still store, instead of queuing behind the numerator reduce-scatter. The same bytes and the same two reductions, so the
bits are the same. MEASURED (12L E 2-node profiles, rank 0): the sums reduce-scatter starts after the numerator
one ends and adds time (C = 16 and C = 8) to every MLA post-loop. False (the kill switch) gives the
program without the sums-first order."""
GATE_WEIGHTS_AFTER_SUMS = True
"""With ``SUM_RS_FIRST`` (the sums reduce-scatter first, the numerator one last), the output-gate weight DMAs of the
call-level state reduce-scatter tail (``tail_weights_after_rs``) wait for the sums reduce-scatter, not the numerator one,
so they load and the gate runs (``GATE_BEFORE_STATE``) while the numerator reduce-scatter flies; the o_proj weights still
wait for the numerator reduce-scatter. The same bytes into the same tiles, so the bits are the same. False (the kill
switch) keeps every tail weight after the numerator reduce-scatter."""
STACK_WEIGHTS_ON_ACT = True
"""``stack_rows`` (``_attend_group_stacked``): the transposed weights of each unit and key group move out of PSUM on
ScalarE in place of VectorE, as ``weights_on_act`` does on the loop path. A plain copy of the BF16 weights, so the bits are
the same, and VectorE, the busiest engine of the stacked draft attention, loses one copy a unit and group. MEASURED (a
profile of the gated served C16 drafter with ``merged_out_ag``, ``stack_rows`` and the query tiles, rank 0): that copy was on the
critical chain on VectorE. False (the kill switch) keeps it on VectorE."""
STACK_STATE_UPPER = True
"""``stack_rows``: the FP32 state of each row ``u > 0`` of a unit runs on the partitions ``step u ..`` of the pass state
tile, in the column of the unit's row 0 (the tile takes 128 partitions), in place of a shadow tile of its own. With
``step == heads`` the rows of a unit sit on adjacent partitions of one column, so one VectorE add a unit and key group
(``[width, 512]`` and ``[width, 1]``) does the state adds of all its rows. Each partition adds the same two FP32 values as
the per-row add, so the bits are the same. MEASURED (a profile of the gated C16 drafter with ``stack_rows``, rank 0): the
per-row state adds were a large part of the VectorE work of each MLA pass, VectorE the busiest engine of the pass.
False (the kill switch) keeps the shadow tiles and one add a row."""
STACK_DENSE_TAIL = True
"""``stack_rows`` with the FP32 HBM state pair: the output tail of each pass loads its local state dense (one row of a
head on each partition, ``dense_state``), so the normalize runs on 32 partitions in place of 2 and the latent takes 4
transposes in place of 64 (``_value_projection_dense``, the same values). MEASURED (a profile of the gated C16 drafter
with ``stack_rows``, rank 0): the head-major normalize was one VectorE op and its transposes took PE time,
both between the state reduce-scatter and the output reduce-scatter of the last pass of each layer. False (the kill
switch) keeps the head-major tail."""
_QKV_CHUNK_TILES = 28
_GATE_CHUNK_TILES = 8
_O_PROJ_CHUNK_TILES = 14
_PSUM_BANK_FP32 = 512
CP_SIZES = (32, 128)
"""CP group sizes of the body: 32 (TP32 = CP32 in a DP4 replica), 128 (one replica on all ranks, one head per rank)."""


class _Layout(NamedTuple):
    """The CP-dependent sizes of one call (see ``cp_layout``)."""

    cp: int
    heads: int
    local_heads: int
    q_lora_local: int
    kv_latent_local: int
    kpe_local: int
    qkv_local: int
    block_stride: int
    group_stride: int
    rank_scale: float
    rank_current_weight: float

    @property
    def kpe_ranks(self) -> int:
        """Ranks that hold k_pe columns: column ``c`` of rank ``r < kpe_ranks`` is k_pe feature ``kpe_ranks * c + r``."""
        return _ROPE // self.kpe_local

    @property
    def one_head(self) -> bool:
        """One query and output head on each rank, with zero pad heads on the ranks at and above ``heads``."""
        return self.cp != _TP


def cp_layout(cp: int, local_heads: int, heads: int = None) -> _Layout:
    """Return the CP-dependent sizes of a call with ``cp`` CP ranks.

    CP32: ``local_heads`` heads on each rank, ``32 * local_heads`` gathered heads (the DP4 replica). CP128:
    rank ``g`` owns the query and the output of head ``g`` for ``g < heads`` (``local_heads`` is 1), the q_a
    columns ``12 g .. 12 g + 11``, the kv latent features ``128 c + g`` and the k_pe feature ``g`` for ``g < 64``.

    Example:
        >>> cp_layout(128, 1, 96).qkv_local
        17
    """
    kernel_assert(cp in CP_SIZES, f"cp must be one of {CP_SIZES}, got {cp}")
    if cp == _TP:
        return _Layout(
            _TP,
            _TP * local_heads,
            local_heads,
            _Q_LORA_LOCAL,
            _KV_LATENT_LOCAL,
            _CACHE_LOCAL - _KV_LATENT_LOCAL,
            _QKV_LOCAL,
            _GLOBAL_BLOCK_STRIDE,
            _GROUP_STRIDE,
            _RANK_SCALE,
            _RANK_CURRENT_WEIGHT,
        )
    kernel_assert(
        local_heads == 1 and heads is not None and heads <= cp, f"CP{cp} takes one local head and heads <= {cp}"
    )
    q_lora_local, kv_latent_local, kpe_local = _Q_LORA // cp, _KV_LORA // cp, 1
    block_stride = cp * _BLOCK
    rank_log2 = int(math.log2(cp))
    return _Layout(
        cp,
        heads,
        1,
        q_lora_local,
        kv_latent_local,
        kpe_local,
        q_lora_local + kv_latent_local + kpe_local,
        block_stride,
        _GROUP_BLOCKS * block_stride,
        2.0**-rank_log2,
        2.0 ** (_CURRENT_WEIGHT_LOG2 - rank_log2),
    )


class _ArenaPlan:
    """Column ranges of two BF16 SBUF arenas, 32-byte aligned: one for the collective destinations, one for
    the sources. A slot named ``*_dst`` goes to the destination arena (index 1), everything else to arena 0.

    A single mixed arena gets random corruption from a collective whose source and destination share one
    SBUF tensor, and from gather destinations a peer can write before this rank reaches the gather. So the
    sources and the destinations go in different arenas, and both arenas are reserved to the kernel end.
    """

    def __init__(self):
        self.slots = []
        self.widths = [0, 0]

    def slot(self, shape, name):
        """Record a tile of ``shape`` at the next 32-byte-aligned columns of its arena; return its index."""
        arena = 1 if name.endswith("_dst") else 0
        self.slots.append((arena, self.widths[arena], tuple(shape)))
        self.widths[arena] += -(-math.prod(shape[1:]) // 16) * 16
        return len(self.slots) - 1

    def view(self, arenas, index):
        """Return the arena view of slot ``index`` (a BF16 ``nl.NkiTensor``)."""
        arena, offset, shape = self.slots[index]
        view = arenas[arena][0 : shape[0], offset : offset + math.prod(shape[1:])]
        return view.reshape_dim(1, shape[1:]) if len(shape) > 2 else view


def mla_block_collective_sources(
    total_rows: int,
    local_heads: int,
    sb: SbufTiles,
    *,
    with_rs: bool = False,
    cp: int = _TP,
    heads: int = None,
    gather_sb: SbufTiles = None,
    q_hbm: bool = False,
    out_ag_call: bool = False,
    rs_call: bool = False,
    out_ar_call: bool = False,
    qkv_extra_rows: int = 0,
    out_rows: int = None,
    y_dst_passes: int = 0,
) -> dict:
    """Return the persistent SBUF tiles of the MLA collectives of one layer (sources and destinations).

    An integration caller that invokes ``mla_block_attention`` many times (one per layer) allocates these
    once from a persistent ``sb`` (a region heap or an arena) and passes the dict as ``sources=`` to every
    call. The tiles then stay reserved to the kernel end, so no later tile takes an all-gather source's bytes
    while a CP peer still reads them, and no earlier tile shares a gather destination's bytes a peer can
    write early. The ``*_dst`` keys name the gather destinations, which the caller's plan puts
    in a separate arena. The state reduce-scatter runs in HBM (see ``mla_block_collective_hbm``). With
    ``cp=128`` the query gather runs in HBM (``mla_block_collective_hbm``), so there are no ``q`` tiles.
    ``gather_sb``, when given, holds the qkv gather pair (for example the caller's reserved arenas).

    ``with_rs`` adds the output reduce-scatter sources (two buffers, alternated by chunk parity like ``ag``):
    a caller that opts into ``persistent_rs`` needs them reserved, so a later tile (a weight load, a mask
    scratch) cannot take a reduce-scatter source's bytes while a CP peer still reads them for the sum. A
    reserved ``sb`` tile only stays reserved when ``sb`` is a region or an arena; a plain fresh tile is not,
    which is why the reduce-scatter source must come from this persistent dict, not from the body.

    ``q_hbm`` drops the ``q`` tiles: the call gathers the queries of all rows once, in the HBM pair of
    ``mla_block_collective_hbm(q_hbm=True)``. ``out_ag_call`` replaces the two ``ag`` parity tiles with one
    ``ag_call`` tile that holds the output reduce-scatter slice of every pass, for one output all-gather a call.
    ``rs_call`` (with ``with_rs``) replaces the two ``rs`` tiles with one ``rs_call`` tile that holds the output
    partial of every pass, for one output reduce-scatter a call (``mla_block_attention`` ``fit_passes``).
    ``out_ar_call`` drops the ``ag`` tiles: the output sum is an all-reduce in the HBM pair of
    ``mla_block_collective_hbm(out_ar_call=True)``. ``qkv_extra_rows`` (the split v5 fold): the qkv gather pair holds
    that many more rows after the call's rows (the foreign rows of ``mla_block_attention`` ``split`` ``fold``).
    ``out_rows`` (the split v5 extended batch: Q x B request rows of the Q x B' call rows): the ``rs_call`` tile holds
    that many output rows (the call returns the request rows only); None keeps the call's rows. ``y_dst_passes``
    (``mla_block_attention`` ``pass_tail_overlap``) adds ``y_dst``: one output all-gather destination ``[128, 16, 56]``
    for each of that many passes; 0 (default) keeps the tiles.
    """
    lay = cp_layout(cp, local_heads, heads)
    tr = _TAIL_ROWS if total_rows > _TAIL_ROWS else total_rows
    gather_sb = sb if gather_sb is None else gather_sb
    sources = {
        "qkv": gather_sb.tile((total_rows + qkv_extra_rows, lay.qkv_local), nl.bfloat16, "mla_qkv_src"),
        "qkv_dst": gather_sb.tile((total_rows + qkv_extra_rows, cp * lay.qkv_local), nl.bfloat16, "mla_qkv_dst"),
    }
    # The ag tiles go last, after the q tiles, to keep the CP32 arena layout.
    if not lay.one_head and not q_hbm:
        gather_width = local_heads * _KV_LORA + local_heads * _ROPE
        n_q_tiles = sum(
            (min(_TAIL_ROWS, total_rows - cb) + _MAX_TOKENS - 1) // _MAX_TOKENS
            for cb in range(0, total_rows, _TAIL_ROWS)
        )
        sources["q"] = [sb.tile((_MAX_TOKENS, gather_width), nl.bfloat16, f"mla_q_src_{i}") for i in range(n_q_tiles)]
        sources["q_dst"] = [sb.tile((_P_MAX, gather_width), nl.bfloat16, f"mla_q_dst_{i}") for i in range(n_q_tiles)]
    passes = -(-total_rows // _TAIL_ROWS)
    if with_rs and rs_call:
        rs_rows = passes * tr if out_rows is None else out_rows
        sources["rs_call"] = sb.tile((_P_MAX, rs_rows, _HIDDEN_TILES), nl.bfloat16, "mla_rs_call_src")
    elif with_rs:
        sources["rs"] = [sb.tile((_P_MAX, tr, _HIDDEN_TILES), nl.bfloat16, f"mla_rs_src_{i}") for i in range(2)]
    if out_ar_call:
        pass
    elif out_ag_call:
        sources["ag_call"] = sb.tile((_P_MAX // cp, passes * tr * _HIDDEN_TILES), nl.bfloat16, "mla_ag_call_src")
    else:
        sources["ag"] = [sb.tile((_P_MAX // cp, tr, _HIDDEN_TILES), nl.bfloat16, f"mla_ag_src_{i}") for i in range(2)]
    if y_dst_passes:
        kernel_assert(not lay.one_head and not out_ag_call, "y_dst_passes is for the per-pass CP32 output all-gathers")
        sources["y_dst"] = [
            sb.tile((_P_MAX, tr, _HIDDEN_TILES), nl.bfloat16, f"mla_y_dst_{i}") for i in range(y_dst_passes)
        ]
    return sources


def _q_call_layout(local_heads: int, q_fp8) -> tuple:
    """Return the row width and dtype of the call query gather pair (``q_hbm``): BF16 rows of the absorbed and rope
    queries; e4m3 rows with ``q_fp8`` True; with ``q_fp8`` "latent", uint16 rows of the e4m3 latent query (two bytes a
    word) and then the BF16 rope query (an integer dtype, so the all-gather keeps every byte).

    With ``q_fp8`` "mx" (``packed_prior_options`` ``q_mx_gather``): uint16 rows of ``_Q_MX_HEAD_BYTES`` a local head;
    the block of each pass is laid out by ``_q_mx_regions``. "mx_packed" (with ``q_mx_scale_words``): rows of
    ``_Q_MX_PACKED_HEAD_BYTES`` a local head (``_q_mx_regions`` ``packed``).

    Example:
        >>> _q_call_layout(3, False)[0], _q_call_layout(3, True)[0], _q_call_layout(3, "latent")[0]
        (1728, 1728, 960)
        >>> _q_call_layout(3, "mx")[0], _q_call_layout(3, "mx_packed")[0]
        (990, 1158)
    """
    if q_fp8 == "mx":
        return local_heads * _Q_MX_HEAD_BYTES // 2, nl.uint16
    if q_fp8 == "mx_packed":
        return local_heads * _Q_MX_PACKED_HEAD_BYTES // 2, nl.uint16
    if q_fp8 == "latent":
        return local_heads * (_KV_LORA // 2 + _ROPE), nl.uint16
    return local_heads * (_KV_LORA + _ROPE), nl.float8_e4m3fn if q_fp8 else nl.bfloat16


def check_q_fp8_options(pack_options: dict, q_hbm: bool):
    """Refuse a ``q_fp8`` of the packed-loop options ``pack_options`` (``_PACKED_PRIOR_OPTIONS`` filled in) that the call
    cannot take: True needs ``dr_keys`` or ``p2`` and ``q_hbm``; "latent" needs ``p2`` with ``mx_qk`` (whose latent QK
    takes an e4m3 latent query already: with the BF16 latent QK the e4m3 latent query costs accuracy, 1.22x the error of
    the BF16 gather in the 32-rank k7b2 check) and ``q_hbm``, and no ``dr_keys``.

    Example:
        >>> check_q_fp8_options({**_PACKED_PRIOR_OPTIONS, "p2": True, "mx_qk": "block32", "q_fp8": "latent"}, True)
    """
    q_fp8 = pack_options["q_fp8"]
    kernel_assert(
        q_fp8 in (False, True, "latent"), f"packed_prior_options q_fp8 is False, True or 'latent', got {q_fp8!r}"
    )
    kernel_assert(
        not q_fp8 or ((pack_options["dr_keys"] or pack_options["p2"]) and q_hbm),
        "packed_prior_options q_fp8 needs dr_keys or p2 (the e4m3 query) and q_hbm (the one query gather of the call)",
    )
    kernel_assert(
        q_fp8 != "latent" or (pack_options["p2"] and pack_options["mx_qk"] and not pack_options["dr_keys"]),
        "packed_prior_options q_fp8 'latent' (the e4m3 latent query, the BF16 rope query) needs p2 with mx_qk (the MX "
        "latent QK, whose latent query is e4m3 already) and no dr_keys",
    )
    q_mx = pack_options.get("q_mx_gather", False)
    kernel_assert(q_mx in (False, True), f"packed_prior_options q_mx_gather is False or True, got {q_mx!r}")
    kernel_assert(
        not q_mx
        or (
            pack_options["fp8_pv"]
            and q_hbm
            and not q_fp8
            and not pack_options["dr_keys"]
            and not pack_options["mx_keys"]
        ),
        "packed_prior_options q_mx_gather needs fp8_pv (the MX latent QK of the natural grouping) and q_hbm (the query "
        "gather of the call), and takes no q_fp8, dr_keys or mx_keys",
    )


def _q_latent_store(q_local: nl.NkiTensor, total_rows: int, local_heads: int, sb: SbufTiles) -> nl.NkiTensor:
    """Return the uint16 rows of the ``q_fp8`` "latent" query gather pair (``_q_call_layout``) from the BF16 query rows
    ``q_local`` ``[Q*B, local_heads x 576]``: the latent query clamped to +-448 into e4m3 on VectorE (the rounding of
    ``q_fp8``) in the first local_heads x 512 bytes, then the BF16 rope query, copied on ScalarE."""
    absorb = local_heads * _KV_LORA
    width, dtype = _q_call_layout(local_heads, "latent")
    q_store = sb.tile((total_rows, width), dtype)
    nisa.tensor_scalar(
        dst=q_store.view(nl.float8_e4m3fn)[0:total_rows, 0:absorb],
        data=q_local[0:total_rows, 0:absorb],
        op0=nl.minimum,
        operand0=_E4M3_MAX,
        op1=nl.maximum,
        operand1=-_E4M3_MAX,
        engine=nisa.engine.vector,
    )
    nisa.tensor_copy(
        dst=q_store.view(nl.bfloat16)[0:total_rows, absorb // 2 : width],
        src=q_local[0:total_rows, absorb : absorb + local_heads * _ROPE],
        engine=nisa.engine.scalar,
    )
    return q_store


def _q_latent_load(
    dst: nl.NkiTensor, pair: nl.NkiTensor, row: int, total_rows: int, local_heads: int, name: str, rope_dma: dict
):
    """Load query row ``row`` of every rank from the ``q_fp8`` "latent" gather pair ``pair`` (rank-major, ``[32 x Q*B,
    local_heads x 320]`` uint16) into ``dst`` ``[32, local_heads x 576]`` BF16 (with the split query gather, ``row`` is the
    slot and ``total_rows`` the block rows of ``q_slot``): the e4m3 latent part with a casting load
    (exact; neuronx-cc gives a casting load no HWDGE, NCC_IBIR098, so it takes the default queue), and the BF16 rope part
    with ``rope_dma``."""
    width = _q_call_layout(local_heads, "latent")[0]
    absorb = local_heads * _KV_LORA
    rope = local_heads * _ROPE
    nisa.dma_copy(
        dst=dst[0:_TP, 0:absorb],
        src=pair.view(nl.float8_e4m3fn).ap(
            pattern=[[2 * total_rows * width, _TP], [1, absorb]], offset=2 * row * width
        ),
        name=f"{name}_latent",
    )
    nisa.dma_copy(
        dst=dst[0:_TP, absorb : absorb + rope],
        src=pair.view(nl.bfloat16).ap(pattern=[[total_rows * width, _TP], [1, rope]], offset=row * width + absorb // 2),
        name=f"{name}_rope",
        **rope_dma,
    )


_Q_MX_HEAD_BYTES = _KV_LORA + _KV_LORA // 32 + 2 * _ROPE + 4
"""Bytes of one (row, local head) in the ``q_mx_gather`` query pair: the e4m3 latent query (512), its E8M0 block scales
(16), the BF16 rope query (128) and the FP32 own-key score (4)."""
_Q_MX_SCALE_ROWS = 4
"""Partitions of each quadrant of a 128-partition MX tile that hold the E8M0 scales (``nisa.quantize_mx``)."""
_Q_MX_PACKED_HEAD_BYTES = 5 * _P_MAX + 2 * _ROPE + 4
"""Bytes of one (row, local head) in the ``q_mx_scale_words`` pair: a word record byte set of 5 bytes for each of the
128 partitions (4 e4m3 word bytes, then the E8M0 scale byte of a scale partition or a pad byte), the BF16 rope query
(128) and the FP32 own-key score (4)."""


_Q_MX_SCORES_ONE_DMA = True
"""``q_mx_gather`` (True in a served program): the receiver loads the own-key scores of all ranks in one DMA. False
(the A / B switch of the timing rows): one DMA a rank. The same bytes either way."""

_Q_MX_PAD_POISON_TEST = None
"""Test hook (``q_mx_scale_words``): None in a served program. An integer 0 .. 255 makes ``_q_mx_store`` fill the pad
bytes of its word records with that byte (one more DMA a block), so a check can show that no output takes a pad byte."""


def _q_mx_regions(rows: int, local_heads: int, packed: bool = False) -> dict:
    """Return the layout of one rank's block of ``rows`` rows of the ``q_mx_gather`` pair (``_q_call_layout`` "mx"),
    head-major: local head h takes bytes h ``head`` .. (h + 1) ``head`` - 1 of the block, and in its part, at the byte
    offsets returned here, its MX query words ``[128, rows]`` FP32 (word c holds the e4m3 latent features 4c .. 4c + 3),
    its E8M0 scale rows ``[16, rows]`` (row 4q + m: partition 32q + m of the MX tile), its rope query ``[64, rows]``
    BF16 and its own-key scores ``[rows]`` FP32. ``cols`` = local_heads x rows (head, row) columns (column h rows + i:
    local head h of block row i). A block is local_heads ``head`` bytes, so the scores of head 3 g + h of the gather
    (rank g) sit at one stride, ``head``: one DMA loads them onto the head partitions.

    ``packed`` (``q_mx_scale_words``, ``_q_call_layout`` "mx_packed"): the words and the scales of a head share one
    region of 128 records of ``record`` = 5 rows bytes, record c the 4 rows word bytes of partition c, then at
    ``record_scales`` the rows scale bytes of partition c (partition 32q + m, m < 4: scale row 4q + m; any other
    partition: pad bytes, never written by the owner); then the rope query and the scores as above.

    Example:
        >>> _q_mx_regions(16, 3)
        {'cols': 48, 'head': 10560, 'words': 0, 'scales': 8192, 'rope': 8448, 'scores': 10496, 'end': 31680}
        >>> _q_mx_regions(16, 3, packed=True)
        {'cols': 48, 'head': 12352, 'words': 0, 'record': 80, 'record_scales': 64, 'rope': 10240, 'scores': 12288, 'end': 37056}
    """
    cols = local_heads * rows
    if packed:
        head = rows * _Q_MX_PACKED_HEAD_BYTES
        rope = _P_MAX * 5 * rows
        scores = rope + _ROPE * rows * 2
        return {
            "cols": cols,
            "head": head,
            "words": 0,
            "record": 5 * rows,
            "record_scales": 4 * rows,
            "rope": rope,
            "scores": scores,
            "end": local_heads * head,
        }
    head = rows * _Q_MX_HEAD_BYTES
    scales = _P_MAX * rows * 4
    rope = scales + 4 * _Q_MX_SCALE_ROWS * rows
    scores = rope + _ROPE * rows * 2
    return {
        "cols": cols,
        "head": head,
        "words": 0,
        "scales": scales,
        "rope": rope,
        "scores": scores,
        "end": local_heads * head,
    }


def _q_mx_store(q_local, cur_keys, pair, blocks, q_rows, tokens, local_heads, tag, sb, dma, packed=False):
    """``q_mx_gather``: store block p of ``blocks`` (the rows of each pass, in pass order; n rows a block) at rows
    p n .. p n + n - 1 of the uint16 query pair ``pair`` in the layout of ``_q_mx_regions`` (``packed``:
    ``q_mx_scale_words``, the words and scale rows in the word records), from this rank's BF16 query
    rows ``q_local`` ``[Q B, H x 576]`` (H local heads) and the current keys ``cur_keys`` ``[128, 5, Q B]``. The bytes are
    those that each receiver made from the gathered BF16 rows: the natural-grouping lanes come from exact PE transposes
    (``_mx_natural_query_tiles``), and one ``nisa.quantize_mx`` makes the words and scales (a scale block is 32 latent
    features of one (row, head) column); the rope query is an exact transpose. The own-key score of each (head, row) is
    the ``_own_key`` matmul set (4 latent tiles, then rope, into one PSUM tile) on the same operands, with the rows of the
    block as the moving columns; bit masks take its diagonal (each row's own key), so every bit stays. The source tiles
    of the store DMAs are heap tiles, so no later tile takes their bytes while a DMA reads them. Return the store DMAs."""
    bf16 = nl.bfloat16
    fp32 = nl.float32
    u16 = nl.uint16
    int32 = nl.int32
    vector = nisa.engine.vector
    tensor = nisa.engine.tensor
    total_rows = q_rows * tokens
    n = len(blocks[0])
    nb = n // q_rows
    reg = _q_mx_regions(n, local_heads, packed)
    cols = reg["cols"]
    part = reg["head"] // 2
    width = pair.shape[1]
    quadrant = _P_MAX // 4
    kernel_assert(
        2 * width * n == reg["end"] and cols <= _P_MAX and n % 2 == 0 and n % q_rows == 0,
        f"q_mx_gather: a block of {n} rows takes {reg['end']} bytes and at most {_P_MAX} (head, row) columns",
    )
    firsts = []
    for p, rows in enumerate(blocks):
        first = rows[0] % tokens
        kernel_assert(
            list(rows) == [j * tokens + first + bb for j in range(q_rows) for bb in range(nb)],
            f"q_mx_gather: block {p} is not the sequence-major rows of {nb} sequences",
        )
        firsts.append(first)
    # The diagonal mask: element (h n + i, i') is all ones where i' = i (block row i's own key), else zero.
    rel = sb.tile((cols, n), int32)
    nisa.iota(dst=rel, pattern=[[-1, n]], offset=0, channel_multiplier=1)
    mask = sb.tile((cols, n), int32)
    band = sb.tile((cols, n), int32)
    for h in range(local_heads):
        nisa.tensor_scalar(
            dst=band if h else mask,
            data=rel,
            op0=nl.equal,
            operand0=float(h * n),
            op1=nl.multiply,
            operand1=-1.0,
            engine=vector,
        )
        if h:
            nisa.tensor_tensor(dst=mask, data1=mask, data2=band, op=nl.add, engine=vector)
    lanes_sb = [sb.tile((_P_MAX, cols, _MX_PACK), bf16) for _ in blocks]
    tiles_sb = [sb.tile((_P_MAX, _KV_LORA_TILES, cols), bf16) for _ in blocks]
    words = [sb.tile((_P_MAX, cols), fp32, keep=True) for _ in blocks]
    scales = [sb.tile((_P_MAX, cols), nl.uint8, keep=True) for _ in blocks]
    rope = [sb.tile((_ROPE, cols), bf16, keep=True) for _ in blocks]
    diag = [sb.tile((cols, 1), int32, keep=True) for _ in blocks]
    for h in range(local_heads):
        engine = nisa.engine.scalar if h % 2 else vector
        heads_cols = slice(h * n, (h + 1) * n)
        # Lane t: every 4th latent feature from feature t (the step-4 stationary read of _mx_natural_query_tiles).
        lanes = nl.ndarray((_P_MAX, _MX_PACK, total_rows), dtype=bf16, buffer=nl.psum)
        for lane in range(_MX_PACK):
            nisa.nc_transpose(
                dst=lanes[0:_P_MAX, lane, 0:total_rows],
                data=q_local[0:total_rows, h * _KV_LORA + lane : (h + 1) * _KV_LORA : _MX_PACK],
                engine=tensor,
            )
        tiles = nl.ndarray((_P_MAX, _KV_LORA_TILES, total_rows), dtype=bf16, buffer=nl.psum)
        for latent_tile in range(_KV_LORA_TILES):
            col = h * _KV_LORA + latent_tile * _P_MAX
            nisa.nc_transpose(
                dst=tiles[0:_P_MAX, latent_tile, 0:total_rows],
                data=q_local[0:total_rows, col : col + _P_MAX],
                engine=tensor,
            )
        for p, first in enumerate(firsts):
            nisa.tensor_copy(
                dst=lanes_sb[p][0:_P_MAX, heads_cols, 0:_MX_PACK].reshape_dim(1, (q_rows, nb)),
                src=lanes.reshape_dim(2, (q_rows, tokens))[0:_P_MAX, 0:_MX_PACK, 0:q_rows, first : first + nb].permute(
                    (0, 2, 3, 1)
                ),
                engine=engine,
            )
            nisa.tensor_copy(
                dst=tiles_sb[p][0:_P_MAX, 0:_KV_LORA_TILES, heads_cols].reshape_dim(2, (q_rows, nb)),
                src=tiles.reshape_dim(2, (q_rows, tokens))[0:_P_MAX, 0:_KV_LORA_TILES, 0:q_rows, first : first + nb],
                engine=engine,
            )
    rope_psum = nl.ndarray((_ROPE, local_heads, total_rows), dtype=bf16, buffer=nl.psum)
    for h in range(local_heads):
        col = local_heads * _KV_LORA + h * _ROPE
        nisa.nc_transpose(
            dst=rope_psum[0:_ROPE, h, 0:total_rows], data=q_local[0:total_rows, col : col + _ROPE], engine=tensor
        )
    for p, first in enumerate(firsts):
        nisa.tensor_copy(
            dst=rope[p].reshape((_ROPE, local_heads, q_rows, nb)),
            src=rope_psum.reshape_dim(2, (q_rows, tokens))[0:_ROPE, 0:local_heads, 0:q_rows, first : first + nb],
            engine=vector,
        )
    keys = cur_keys.reshape_dim(2, (q_rows, tokens))
    stores = []
    for p, first in enumerate(firsts):
        nisa.quantize_mx(
            dst=words[p][0:_P_MAX, 0:cols].view(nl.float8_e4m3fn_x4),
            src=lanes_sb[p][0:_P_MAX, 0:cols, 0:_MX_PACK],
            dst_scale=scales[p][0:_P_MAX, 0:cols],
        )
        score_psum = nl.ndarray((cols, n), dtype=fp32, buffer=nl.psum)
        for latent_tile in range(_KV_LORA_TILES):
            nisa.nc_matmul(
                dst=score_psum.reshape_dim(1, (q_rows, nb)),
                stationary=tiles_sb[p][0:_P_MAX, latent_tile, 0:cols],
                moving=keys[0:_P_MAX, latent_tile, 0:q_rows, first : first + nb],
                accumulate=latent_tile > 0,
            )
        nisa.nc_matmul(
            dst=score_psum.reshape_dim(1, (q_rows, nb)),
            stationary=rope[p][0:_ROPE, 0:cols],
            moving=keys[0:_ROPE, _KV_LORA_TILES, 0:q_rows, first : first + nb],
            accumulate=True,
        )
        hits = sb.tile((cols, n), int32)
        nisa.tensor_tensor(dst=hits, data1=score_psum.view(int32), data2=mask, op=nl.bitwise_and, engine=vector)
        nisa.tensor_reduce(dst=diag[p], data=hits, op=nl.bitwise_or, axis=(1,), keepdims=True)
        base = p * n * width
        # Head-major (``_q_mx_regions``): the (head, row) columns h n .. h n + n - 1 of each tile go to head h's part.
        # ``packed``: partition c's words start word record c (``record`` bytes apart), its scale row at ``record_scales``.
        word_step = reg["record"] // 2 if packed else 2 * n
        stores.append(
            nisa.dma_copy(
                dst=pair.ap(
                    pattern=[[word_step, _P_MAX], [part, local_heads], [1, 2 * n]], offset=base + reg["words"] // 2
                ),
                src=words[p].view(u16).reshape((_P_MAX, local_heads, 2 * n)),
                name=f"{tag}_q_mx_words_p{p}",
                **dma,
            )
        )
        if packed and _Q_MX_PAD_POISON_TEST is not None:
            # Test hook: the pad bytes of the word records (partitions 32q + 4 .. 32q + 31) take the poison byte.
            pad_rows = quadrant - _Q_MX_SCALE_ROWS
            poison = sb.tile((pad_rows, cols // 2), u16)
            nisa.memset(dst=poison, value=_Q_MX_PAD_POISON_TEST * 257, engine=vector)
            for quad in range(_P_MAX // quadrant):
                stores.append(
                    nisa.dma_copy(
                        dst=pair.ap(
                            pattern=[[word_step, pad_rows], [part, local_heads], [1, n // 2]],
                            offset=base
                            + ((quadrant * quad + _Q_MX_SCALE_ROWS) * reg["record"] + reg["record_scales"]) // 2,
                        ),
                        src=poison.reshape((pad_rows, local_heads, n // 2)),
                        name=f"{tag}_q_mx_pad_poison_p{p}_{quad}",
                        **dma,
                    )
                )
        for quad in range(_P_MAX // quadrant):
            scale_dst = (
                pair.ap(
                    pattern=[[word_step, _Q_MX_SCALE_ROWS], [part, local_heads], [1, n // 2]],
                    offset=base + (quadrant * quad * reg["record"] + reg["record_scales"]) // 2,
                )
                if packed
                else pair.ap(
                    pattern=[[n // 2, _Q_MX_SCALE_ROWS], [part, local_heads], [1, n // 2]],
                    offset=base + (reg["scales"] + _Q_MX_SCALE_ROWS * quad * n) // 2,
                )
            )
            stores.append(
                nisa.dma_copy(
                    dst=scale_dst,
                    src=scales[p]
                    .view(u16)[quadrant * quad : quadrant * quad + _Q_MX_SCALE_ROWS, 0 : cols // 2]
                    .reshape((_Q_MX_SCALE_ROWS, local_heads, n // 2)),
                    name=f"{tag}_q_mx_scales_p{p}_{quad}",
                    **dma,
                )
            )
        stores.append(
            nisa.dma_copy(
                dst=pair.ap(pattern=[[n, _ROPE], [part, local_heads], [1, n]], offset=base + reg["rope"] // 2),
                src=rope[p].view(u16).reshape((_ROPE, local_heads, n)),
                name=f"{tag}_q_mx_rope_p{p}",
                **dma,
            )
        )
        for h in range(local_heads):
            stores.append(
                nisa.dma_copy(
                    dst=pair.ap(pattern=[[2, n], [1, 2]], offset=base + h * part + reg["scores"] // 2),
                    src=diag[p].view(u16)[h * n : (h + 1) * n, 0:2],
                    name=f"{tag}_q_mx_scores_p{p}_{h}",
                    **dma,
                )
            )
    return stores


def _q_mx_stage(n, local_heads, sb, name=None, keep=False, packed=False):
    """``q_mx_gather``: the stage tiles of one pass of ``n`` rows (words FP32, scales u8, rope BF16), into which
    ``_q_mx_loads`` puts the gathered bytes and from which ``_q_mx_copies`` places them. ``keep``: heap tiles named
    ``name`` (``q_mx_pass_prefetch``). ``packed`` (``q_mx_scale_words``): one u8 tile of the word records ``[128, 32,
    local_heads x 5 n]`` (rank g, head h: the 4 n word bytes, then the n scale bytes of the partition) in place of the
    words and the scales tiles (the same bytes a partition), None for the scales, then the rope tile."""
    cols = _q_mx_regions(n, local_heads)["cols"]
    if packed:
        return (
            sb.tile((_P_MAX, _TP, 5 * cols), nl.uint8, **({"name": f"{name}_ws", "keep": True} if keep else {})),
            None,
            sb.tile((_ROPE, _TP, cols), nl.bfloat16, **({"name": f"{name}_r", "keep": True} if keep else {})),
        )
    if not keep:
        return (
            sb.tile((_P_MAX, _TP, cols), nl.float32),
            sb.tile((_P_MAX, _TP, cols), nl.uint8),
            sb.tile((_ROPE, _TP, cols), nl.bfloat16),
        )
    return (
        sb.tile((_P_MAX, _TP, cols), nl.float32, name=f"{name}_w", keep=True),
        sb.tile((_P_MAX, _TP, cols), nl.uint8, name=f"{name}_s", keep=True),
        sb.tile((_ROPE, _TP, cols), nl.bfloat16, name=f"{name}_r", keep=True),
    )


def _q_mx_loads(pair, base_row, n, local_heads, stage, scores, tag, dma, packed=False):
    """``q_mx_gather``: the DMAs of ``_q_mx_load`` (the own-key scores first, then the words, the scales and the rope
    query) into the stage tiles ``stage`` (``_q_mx_stage``) and ``scores``. Return them in that order. ``packed``
    (``q_mx_scale_words``): one DMA of the word records (words and scales) in place of the words and scales DMAs, traced
    after the rope query's: the rope DMA then does not queue behind the 80-byte records on the rings (the words' copies
    have slack, the rope's had a small slack in the c0 entry, MEASURED)."""
    u16 = nl.uint16
    reg = _q_mx_regions(n, local_heads, packed)
    cols = reg["cols"]
    part = reg["head"] // 2
    width = pair.shape[1]
    block = n * width
    base = base_row * width
    quadrant = _P_MAX // 4
    kernel_assert(block == local_heads * part, "q_mx_gather: a rank's block is its local heads' parts")
    stage_w, stage_s, stage_r = stage
    # The gate first (NKI takes a dependency on an instruction traced before only): the scores of head 3 g + h (rank
    # g's part h) at offset (3 g + h) part, onto partition 3 g + h.
    ranks = [None] if _Q_MX_SCORES_ONE_DMA else list(range(_TP))
    score_loads = [
        nisa.dma_copy(
            dst=scores.view(u16)[0 : _TP * local_heads, 0 : 2 * n]
            if g is None
            else scores.view(u16)[local_heads * g : local_heads * (g + 1), 0 : 2 * n],
            src=pair.ap(
                pattern=[[part, _TP * local_heads if g is None else local_heads], [1, 2 * n]],
                offset=base + (0 if g is None else g * block) + reg["scores"] // 2,
            ),
            name=f"{tag}_q_mx_scores" + ("" if g is None else f"_{g}"),
            **dma,
        )
        for g in ranks
    ]
    loads = score_loads[:1]

    def rope_load():
        return nisa.dma_copy(
            dst=stage_r.view(u16).reshape((_ROPE, _TP, local_heads, n)),
            src=pair.ap(
                pattern=[[n, _ROPE], [block, _TP], [part, local_heads], [1, n]], offset=base + reg["rope"] // 2
            ),
            name=f"{tag}_q_mx_rope",
            **dma,
        )

    if packed:
        loads.append(rope_load())
    # ``packed``: the whole record of each (partition, rank, head), words and scale bytes, is one element.
    word_step, word_len = (reg["record"] // 2, reg["record"] // 2) if packed else (2 * n, 2 * n)
    load = nisa.dma_copy(
        dst=stage_w.view(u16).reshape((_P_MAX, _TP, local_heads, word_len)),
        src=pair.ap(
            pattern=[[word_step, _P_MAX], [block, _TP], [part, local_heads], [1, word_len]],
            offset=base + reg["words"] // 2,
        ),
        name=f"{tag}_q_mx_words",
        **dma,
    )
    loads.append(load)
    for quad in range(0 if packed else _P_MAX // quadrant):
        load = nisa.dma_copy(
            dst=stage_s.view(u16)[quadrant * quad : quadrant * quad + _Q_MX_SCALE_ROWS, 0:_TP, 0 : cols // 2].reshape(
                (_Q_MX_SCALE_ROWS, _TP, local_heads, n // 2)
            ),
            src=pair.ap(
                pattern=[[n // 2, _Q_MX_SCALE_ROWS], [block, _TP], [part, local_heads], [1, n // 2]],
                offset=base + (reg["scales"] + _Q_MX_SCALE_ROWS * quad * n) // 2,
            ),
            name=f"{tag}_q_mx_scales_{quad}",
            **dma,
        )
        loads.append(load)
    if not packed:
        loads.append(rope_load())
    loads += score_loads[1:]
    return loads


def _q_mx_copies(n, q_rows, local_heads, stage, q_words, q_scales, q_rope_all):
    """``q_mx_gather``: the copies of ``_q_mx_load`` that place the stage tiles ``stage`` (``_q_mx_stage``) in the
    pass operands through integer views (VectorE). Return them in trace order. A ``stage`` without a scales tile
    (``q_mx_scale_words``) holds the word records: the words and the scales are views of its records, at the same
    (partition, rank, head, row) positions as the two tiles hold them."""
    u16 = nl.uint16
    vector = nisa.engine.vector
    nb = n // q_rows
    stage_w, stage_s, stage_r = stage
    copies = []
    # u16 element ((b Q + j) 32 + g) 6 + 2 h + half of q_words; ((g 3 + h) Q + j) 2 nb + 2 b + half of the staged words.
    words_u = q_words.view(u16).reshape((_P_MAX, nb * q_rows, _TP, 2 * local_heads))
    scales_v = q_scales.reshape((_P_MAX, nb * q_rows, _TP, local_heads))
    if stage_s is None:
        # Record (g, h) of a partition: u16 words 0 .. 2 n - 1, then bytes 4 n .. 5 n - 1 the scales.
        records_u = stage_w.view(u16).reshape((_P_MAX, _TP, local_heads, 5 * n // 2))
        stage_wu = records_u[0:_P_MAX, 0:_TP, 0:local_heads, 0 : 2 * n].reshape_dim(3, (q_rows, 2 * nb))
        records = stage_w.reshape((_P_MAX, _TP, local_heads, 5 * n))
        stage_sv = records[0:_P_MAX, 0:_TP, 0:local_heads, 4 * n : 5 * n].reshape_dim(3, (q_rows, nb))
    else:
        stage_wu = stage_w.view(u16).reshape((_P_MAX, _TP, local_heads, q_rows, 2 * nb))
        stage_sv = stage_s.reshape((_P_MAX, _TP, local_heads, q_rows, nb))
    rope_u = q_rope_all.view(u16)
    stage_ru = stage_r.view(u16).reshape((_ROPE, _TP, local_heads, n))
    for h in range(local_heads):
        for bb in range(nb):
            seq_rows = slice(bb * q_rows, (bb + 1) * q_rows)
            copies.append(
                nisa.tensor_copy(
                    dst=words_u[0:_P_MAX, seq_rows, 0:_TP, 2 * h : 2 * h + 2],
                    src=stage_wu[0:_P_MAX, 0:_TP, h, 0:q_rows, 2 * bb : 2 * bb + 2].permute((0, 2, 1, 3)),
                    engine=vector,
                )
            )
            copies.append(
                nisa.tensor_copy(
                    dst=scales_v[0:_P_MAX, seq_rows, 0:_TP, h],
                    src=stage_sv[0:_P_MAX, 0:_TP, h, 0:q_rows, bb].permute((0, 2, 1)),
                    engine=vector,
                )
            )
        copies.append(
            nisa.tensor_copy(
                dst=rope_u[0:_ROPE, 0:n, 0 : _TP * local_heads].reshape_dim(2, (_TP, local_heads))[
                    0:_ROPE, 0:n, 0:_TP, h
                ],
                src=stage_ru[0:_ROPE, 0:_TP, h, 0:n].permute((0, 2, 1)),
                engine=vector,
            )
        )
    return copies


def _q_mx_load(
    pair, base_row, n, q_rows, local_heads, q_words, q_scales, q_rope_all, scores, tag, sb, dma, after=(), packed=False
):
    """``q_mx_gather``: load the pass of ``n`` rows whose rank-g block sits at rows ``base_row + g n`` of the gathered
    uint16 pair ``pair`` (``_q_mx_regions``) into the operands of the pass: the MX query words ``q_words`` ``[128, S G
    128]`` FP32 and scales ``q_scales`` (u8) of its p2 columns (column b Q 96 + 96 j + 3 g + h: head 3 g + h of row j
    of sequence b), rows 0 .. 63 of the rope query ``q_rope_all`` ``[128, n, 96]`` and the own-key scores ``scores``
    ``[96, n]`` FP32. The DMAs move bytes (no dtype change, so the hardware DGE takes them), and the copies place them in
    the p2 columns through integer views, so every bit stays. The head-major block (``_q_mx_regions``) puts the scores of
    head 3 g + h at one stride, so one DMA lands all of them on their head partitions. ``after`` (the state store DMAs
    of the pass before, whose source bytes this pass's tiles can take): the first DMA (the 6 KB of scores) waits for
    their completion (a runtime dependency), and every other DMA here waits for it, so each write waits on the store
    semaphores once and the large DMAs start when the small gate lands. The copies read the tiles of these DMAs, so
    they follow them. ``packed``: the ``q_mx_scale_words`` layout (``_q_mx_regions``)."""
    stage = _q_mx_stage(n, local_heads, sb, packed=packed)
    loads = _q_mx_loads(pair, base_row, n, local_heads, stage, scores, tag, dma, packed)
    _q_mx_copies(n, q_rows, local_heads, stage, q_words, q_scales, q_rope_all)
    if after:
        _instruction(loads[0]).depends_on(*after, kind=nisa.dependency.runtime)
        for load in loads[1:]:
            _instruction(load).depends_on(loads[0], kind=nisa.dependency.runtime)
    return tuple(loads)


def mla_block_collective_hbm(
    total_rows: int,
    local_heads: int,
    cp: int = _TP,
    heads: int = None,
    name: str = "",
    reduce_rows: int = None,
    q_hbm: bool = False,
    out_ag_call: bool = False,
    state_call: bool = False,
    out_ar_call: bool = False,
    state_sums: bool = False,
    q_fp8: bool | str = False,
    state_pair: bool = True,
    out_rows: int = None,
    cc_pad256: bool = False,
    state_pairs: int = 1,
) -> dict:
    """Return the persistent HBM tensors of the MLA collectives of one layer: the FP32 state reduce-scatter pair,
    and with ``cp=128`` one query gather pair for each pass of 16 rows.

    The state reduce-scatter runs in HBM, not SBUF: an SBUF reduce-scatter gets a compiler scratch tile, which the
    Neuron runtime writes only on Trn2, after a forced global handshake. The caller passes these as ``sources``;
    they are not cleared at the kernel start. With ``cp=128`` the state source has one row for each rank
    (rank ``g`` gets head ``g``), each pass has a query gather pair, and the output all-reduce runs in an HBM pair.
    ``reduce_rows`` sizes the state reduce-scatter and the output all-reduce for that many rows (see
    ``mla_block_attention``). ``q_hbm`` (TP32 heads) adds the query gather pair of all rows: ``q_call_src_hbm``
    ``[Q*B, width]`` and ``q_call_dst_hbm`` ``[cp*Q*B, width]`` (rank-major, the dim-0 all-gather). ``out_ag_call``
    (TP32 heads) adds the output all-gather pair of all passes: ``out_ag_src_hbm`` ``[128/cp, cols]`` and
    ``out_ag_dst_hbm`` ``[128, cols]``, ``cols`` = passes x 16 rows x 56 hidden tiles. ``state_call`` (TP32 heads,
    more than 16 rows) gives the state pair one row for each head and pass, row ``head * passes + pass``, for one
    state reduce-scatter a call; each row keeps the 16-row message of a pass. ``out_ar_call`` (TP32 heads) adds the
    output all-reduce pair of all passes: ``out_ar_src_hbm`` and ``out_ar_dst_hbm`` ``[passes * 128, 16 rows * 56]``,
    rows ``128 p .. 128 p + 127`` for pass ``p``. ``state_sums`` (TP32 heads, ``state_bf16_o``) adds the FP32 pair of
    the BF16-o sums: ``state_sum_hbm`` / ``state_sum_dst_hbm``, the rows of the state pair x 16 rows, so the sums
    reduce-scatter writes a tensor of its own. ``q_fp8`` (with ``q_hbm``) makes the query gather pair e4m3, or with
    "latent" the uint16 pair of the e4m3 latent and BF16 rope query (``_q_call_layout``; ``packed_prior_options``
    ``q_fp8``), "mx" the uint16 pair of ``q_mx_gather`` and "mx_packed" that of ``q_mx_scale_words``
    (``q_call_fp8``). ``state_pair=False`` (TP32 heads, one pass) leaves out the state pair, so the
    state reduce-scatter of ``mla_block_attention`` reads the SBUF state, as ``tkg_mla_paged`` does without sources: on
    Trn2 a 16-row FP32 state of 96 heads (3,151,872 bytes) exceeds the 2 MiB HBM limit of the mesh reduce-scatter of 32
    ranks, while an SBUF source takes the mesh up to 4 MiB. ``state_sums`` needs the state pair. ``out_rows`` (the split
    v5 extended batch) sizes the output all-reduce pair for that many rows (the request rows); None keeps the call's.
    ``cc_pad256`` pads the columns of the FP32 pair of the state reduce-scatter (the sums pair with ``state_sums``,
    else the state pair) so that each rank's part is a multiple of 256 bytes (``pad256_columns``); the pad columns are
    not read. False (default) keeps the tensors. ``state_pairs`` 2 adds a second state pair of the same shapes,
    ``state_hbm_1`` / ``state_dst_hbm_1``, for the odd passes of ``mla_block_attention`` ``pass_tail_overlap``; 1
    (default) keeps the tensors.
    """
    kernel_assert(state_pairs in (1, 2), f"state_pairs is 1 or 2, got {state_pairs}")
    kernel_assert(
        state_pairs == 1 or (state_pair and not state_sums and not state_call),
        "state_pairs 2 is the second FP32 state pair of pass_tail_overlap: the state pair, not state_sums or state_call",
    )
    kernel_assert(
        state_pair or not (cp_layout(cp, local_heads, heads).one_head or state_call or total_rows > _TAIL_ROWS),
        "state_pair=False is the one-pass TP32 form: the CP128 and multi-pass forms reduce the state in HBM",
    )
    kernel_assert(
        state_pair or not state_sums, "state_sums adds the sums pair of the HBM state pair: not state_pair=False"
    )
    lay = cp_layout(cp, local_heads, heads)
    tr = _TAIL_ROWS if total_rows > _TAIL_ROWS else total_rows
    q_tr = tr
    if reduce_rows is not None:
        tr = min(_TAIL_ROWS, max(tr, reduce_rows))
    state_rows = cp if lay.one_head else lay.heads
    state_passes = -(-total_rows // _TAIL_ROWS) if (state_call and not lay.one_head) else 1
    rank_rows = local_heads * state_passes
    state_cols = tr * (_KV_LORA + 1)
    if cc_pad256 and not state_sums:
        state_cols = pad256_columns(rank_rows, state_cols)
    shapes = {
        "state_hbm": (state_rows * state_passes, state_cols),
        "state_dst_hbm": (rank_rows, state_cols),
    }
    if state_pairs == 2:
        shapes["state_hbm_1"] = shapes["state_hbm"]
        shapes["state_dst_hbm_1"] = shapes["state_dst_hbm"]
    if state_sums and not lay.one_head:
        sum_cols = pad256_columns(rank_rows, tr) if cc_pad256 else tr
        shapes["state_sum_hbm"] = (state_rows * state_passes, sum_cols)
        shapes["state_sum_dst_hbm"] = (rank_rows, sum_cols)
    tensors = {
        key: nl.ndarray(shape, dtype=nl.float32, buffer=nl.private_hbm, name=f"mla{name}_{key}")
        for key, shape in shapes.items()
        if state_pair
    }
    if lay.one_head:
        for key in ("out_src_hbm", "out_dst_hbm"):
            tensors[key] = nl.ndarray(
                (_P_MAX, tr * _HIDDEN_TILES), dtype=nl.bfloat16, buffer=nl.private_hbm, name=f"mla{name}_{key}"
            )
        width = _KV_LORA + _ROPE
        passes = range(-(-total_rows // _TAIL_ROWS))
        tensors["q_src_hbm"] = [
            nl.ndarray((q_tr, width), dtype=nl.bfloat16, buffer=nl.private_hbm, name=f"mla{name}_q_src_hbm_{i}")
            for i in passes
        ]
        tensors["q_dst_hbm"] = [
            nl.ndarray((cp * q_tr, width), dtype=nl.bfloat16, buffer=nl.private_hbm, name=f"mla{name}_q_dst_hbm_{i}")
            for i in passes
        ]
    kernel_assert(not q_fp8 or (q_hbm and not lay.one_head), "q_fp8 needs q_hbm with the 3-head layout")
    kernel_assert(
        q_fp8 in (False, True, "latent", "mx", "mx_packed"),
        f"q_fp8 is False, True, 'latent', 'mx' or 'mx_packed', got {q_fp8!r}",
    )
    if q_hbm and not lay.one_head:
        width, q_dtype = _q_call_layout(local_heads, q_fp8)
        for key, rows in (("q_call_src_hbm", total_rows), ("q_call_dst_hbm", cp * total_rows)):
            tensors[key] = nl.ndarray((rows, width), dtype=q_dtype, buffer=nl.private_hbm, name=f"mla{name}_{key}")
    if out_ag_call and not lay.one_head:
        cols = -(-total_rows // _TAIL_ROWS) * q_tr * _HIDDEN_TILES
        for key, rows in (("out_ag_src_hbm", _P_MAX // cp), ("out_ag_dst_hbm", _P_MAX)):
            tensors[key] = nl.ndarray((rows, cols), dtype=nl.bfloat16, buffer=nl.private_hbm, name=f"mla{name}_{key}")
    if out_ar_call and not lay.one_head:
        # ``out_rows`` (the split v5 extended batch): the pair holds the request rows only.
        tensors.update(_out_ar_pair(total_rows if out_rows is None else out_rows, tr, name))
    return tensors


def _transposed_scores_on(
    transposed_scores: bool, bitwise: bool, causal: bool, exact_softmax: bool, n_mask: int, q_rows: int
) -> bool:
    """Return True when ``mla_block_attention`` attends the prior keys in the rows form (``_attend_group_rows``).

    The rows form needs the capped softmax. The bitwise verify needs one mask row for each query row (``fused_masks``)
    or one query row. The non-causal draft form needs neither: every row of its block attends the same prior keys, so
    the rows share one mask row.

    Example:
        >>> _transposed_scores_on(True, False, False, False, 1, 7), _transposed_scores_on(True, True, True, False, 1, 4)
        (True, False)
    """
    if not transposed_scores or exact_softmax:
        return False
    if bitwise:
        return n_mask > 1 or q_rows == 1
    return not causal


def pad256_columns(rows: int, columns: int) -> int:
    """Return the least FP32 column count at or above ``columns`` that makes ``rows`` rows a multiple of 256 bytes: the
    per-rank part of a reduce-scatter of a ``[ranks * rows, columns]`` FP32 pair (``cc_pad256``).

    Example:
        >>> pad256_columns(3, 16), pad256_columns(6, 16), pad256_columns(2, 7182), pad256_columns(2, 8208)
        (64, 32, 7200, 8224)
    """
    step = 64 // math.gcd(rows, 64)
    return -(-columns // step) * step


def _out_ar_pair(total_rows: int, tr: int, name: str) -> dict:
    """Return the HBM output all-reduce pair of ``out_ar_call``: ``[passes * 128, tr * 56]`` BF16 each."""
    shape = (-(-total_rows // _TAIL_ROWS) * _P_MAX, tr * _HIDDEN_TILES)
    return {
        key: nl.ndarray(shape, dtype=nl.bfloat16, buffer=nl.private_hbm, name=f"mla{name}_{key}")
        for key in ("out_ar_src_hbm", "out_ar_dst_hbm")
    }


def mla_block_attention(
    x_hT: nl.NkiTensor,
    mla_in_proj: nl.NkiTensor,
    mla_q_a_norm: nl.NkiTensor,
    mla_q_b_proj: nl.NkiTensor,
    mla_kv_a_norm: nl.NkiTensor,
    mla_kv_b_proj: nl.NkiTensor,
    mla_o_proj: nl.NkiTensor,
    kv_pool: nl.NkiTensor,
    block_table: nl.NkiTensor,
    slot: nl.NkiTensor,
    position: nl.NkiTensor,
    rank_info: nl.NkiTensor,
    tp_group: ncc.ReplicaGroup,
    tag: str,
    local_heads: int,
    q_rows: int,
    rope_rows: nl.NkiTensor = None,
    causal: bool = False,
    output_gate: bool = False,
    write_cache: bool = True,
    softmax_scale: float = None,
    exact_softmax: bool = False,
    bf16_state: bool = False,
    loop_path: bool = False,
    fp8_pages: bool = False,
    sbuf: SbufTiles = None,
    key_start: nl.NkiTensor = None,
    bitwise: bool = False,
    sources: dict | None = None,
    window_base: nl.NkiTensor = None,
    lora_norm_eps: float = _LORA_NORM_EPS,
    cp: int = _TP,
    heads: int = None,
    reduce_rows: int = None,
    fused_masks: bool = False,
    transposed_scores: bool = False,
    persistent_rs: bool = False,
    loop_group_cap: int = 0,
    q_hbm: bool = False,
    out_ag_call: bool = False,
    fit_passes: bool = False,
    zeroed_pages: bool = False,
    batched_cache_write: bool = False,
    late_tail_weights: bool = False,
    packed_prior: bool = False,
    packed_prior_options: dict | None = None,
    out_rs_tp: int = 0,
    critical_dma_priority: bool | str = False,
    cache_write_off_sp: bool = False,
    own_rows_ahead: bool = False,
    o_proj_ahead: bool = False,
    gemv_bands: bool = False,
    late_o_proj_weights: bool = False,
    late_tail_frames: bool = False,
    out_ar_call: bool = False,
    state_bf16_o: bool = False,
    pass_plans: dict | None = None,
    own_key_select: bool = False,
    mla_in_proj_scale: nl.NkiTensor = None,
    mla_q_b_proj_scale: nl.NkiTensor = None,
    mla_kv_b_proj_scale: nl.NkiTensor = None,
    mla_o_proj_scale: nl.NkiTensor = None,
    fused_single_pass: bool = False,
    loop_first_static: bool = False,
    scores_ahead: bool = False,
    folded_reference: bool = False,
    trimmed_window: bool = False,
    query_rows: int = 0,
    rows_v_stationary: bool = False,
    rows_dr: bool = False,
    rows_prefetch: bool = False,
    rows_dr_act_copies: bool = False,
    rows_dr_inplace: bool = False,
    rows_pair_lag: bool = False,
    rows_pad_skip: bool = False,
    live_rows: nl.NkiTensor = None,
    rows_seq_hoist: bool = False,
    rows_setup_ahead: bool = False,
    rows_q_fp8: bool = False,
    state_rs_hook=None,
    out_ar_hook=None,
    rows_mx_keys: str = None,
    loop_p2: bool = False,
    loop_preload: bool = False,
    rows_pair_tail: bool = False,
    rows_unroll2: bool = False,
    loop_cast_engines: str = "off",
    pass_q_prefetch: int = 0,
    q_gather_buffers: int = 1,
    tail_weights_after_rs: bool = False,
    sums_rs_first: bool = None,
    pass_rows_ahead: bool = False,
    loop_page_gather: str = "off",
    weights_on_act: bool = False,
    loop_tail_groups: int = 0,
    loop_row_pipeline: bool = False,
    split: dict | None = None,
    stack_rows: bool = False,
    q_prefetch_after: tuple = (),
    pass_tail_overlap: bool = False,
    seq_passes: bool = False,
    state_store_split: bool = False,
    prior_hoist: int = 0,
) -> nl.NkiTensor:
    """Run one Kimi-K3 MLA attention layer for ``Q`` query rows of each of ``B`` sequences on one TP32
    rank.

    The rank owns heads ``local_heads t .. local_heads t + local_heads - 1`` and the CP32 cache rows of
    the global blocks ``b`` with ``b % 32 == t``. Query row ``r = j * B + b`` (token-major) is query row
    ``j`` of sequence ``b``.

    Dimensions:
        B: sequences of the DP replica, 1 to 4.
        Q: query rows of each sequence (``q_rows``): 7 for the drafter, 8 for the verify.
        NB: local blocks of the block table.
        Nh: ``32 * local_heads`` gathered heads.

    Args:
        x_hT: ``[128, Q*B, 56]`` BF16 SBUF normalized hidden in token-major hT layout: element
            ``(p, r, k)`` is ``x[r, k * 128 + p]`` of query row ``r = j * B + b``.
        mla_in_proj, mla_q_a_norm, mla_q_b_proj, mla_kv_a_norm, mla_kv_b_proj, mla_o_proj: BF16 HBM
            weights of the layer for ``local_heads`` heads, see ``MLA_WEIGHT_NAMES``.
        kv_pool: ``[P, 73728]`` BF16 or FLOAT8_E4M3FN HBM page pool, as ``mla_decode_attention``.
        block_table: ``[B, NB]`` INT32 HBM local block table of each sequence.
        slot: ``[Q*B, 1]`` INT32 HBM write slot of each query row (0 off the owner rank).
        position: ``[B, 1]`` INT32 HBM global position of query row 0 of each sequence.
        rank_info: ``[1, 4]`` INT32 HBM (global rank, DP rank, TP rank, 0).
        tp_group: the TP32 groups, one list of 32 ranks per DP replica.
        tag: unique per-layer name prefix of the collectives and DMAs.
        local_heads: heads of this rank: 2 (drafter, 64 heads) or 3 (target verify, 96 heads).
        q_rows: query rows ``Q`` of each sequence.
        rope_rows: ``[Q*B, 128]`` FP32 HBM GPT-J cos/sin rows ``[cos (64) | sin (64)]`` of each query
            row, pre-expanded for the pairs. ``None`` runs NoPE (the target).
        causal: the current-block mask. ``False`` is the non-causal block (the drafter); ``True`` is the
            causal mask of the verify (query row ``j`` sees current rows ``0 .. j``).
        output_gate: apply the sigmoid gate of the target. The drafter has no gate region.
        write_cache: owner-write each current row to the pool. ``True`` for the verify; a caller whose
            next stage overwrites the rows before any read (the drafter block) can set ``write_cache=False`` to
            skip the write and the null-page write.
        softmax_scale: the softmax scale. ``None`` uses ``192**-0.5``.
        exact_softmax, bf16_state, loop_path, fp8_pages, sbuf, lora_norm_eps: as ``mla_decode_attention``.
        key_start: ``[B, 1]`` INT32 HBM lower bound. A prior pool key of sequence ``b`` with a global
            position below ``key_start[b]`` is masked, so the call sees only prior keys in
            ``[key_start[b], position[b])``. ``None`` or all-zero is bitwise identical to no bound. The
            current rows always stay visible. Use it for a decode after a prefill prefix-cache hit.
        bitwise: verify-only bitwise-identity mode. Each query row ``r = j * B + b`` becomes the step ``j``
            of sequence ``b`` of ``Q`` sequential ``mla_decode_attention`` calls, so the output is bitwise
            equal to that sequence of decode calls. The current rows are owner-written to the pool and the
            current block is attended as a normal prior group (its page load waits for the write), with the
            per-row mask threshold ``position[b] + j``. The own key ``position[b] + j`` is added through the
            ``2**-98`` current-weight term first, then the prior groups in the decode's group order. Forces
            ``write_cache`` and orders each current-block read after the owner-write. ``False`` keeps the
            capped-tolerance verify path.
        sources: None, or the caller's persistent collective-source tiles (see ``mla_collective_sources``),
            for a layer loop that keeps one tile per site reserved across layers. When None on the standalone
            path, the q, qkv, and tail all-gather sources become columns of one SBUF arena whose last use is
            the tail all-gather, so no later tile takes their bytes while a CP peer still reads them.
        cp: CP ranks of ``tp_group`` (``CP_SIZES``). 32 is the TP32 = CP32 replica. 128 is one replica on all
            ranks: global block ``b`` lives on rank ``b % 128``, and rank ``g`` owns the query and the output of
            head ``g`` (``local_heads`` is 1; the ranks at and above ``heads`` hold zero pad weights).
        heads: gathered heads with ``cp=128`` (96 for the target, 64 for the drafter); None for CP32.
        reduce_rows: ``cp=128``: run the state reduce-scatter and the output all-reduce on buffers of this many rows
            (up to 16) when the call has fewer. The runtime picks the collective algorithm, and so the summation
            order, by the message size: a decode call that sizes its reductions as the verify does gets the same
            bits as the verify rows.
        fused_masks: ``bitwise`` with ``Q > 1``: the prior-key mask of each query row ``j`` is mask row ``j`` of
            the key group, below the rope keys, and a one-hot row of the query selects it in the rope matmul. A
            group then takes one mask pass for all rows, in place of one for each row. The masked keys get the
            same zero weights, so the values do not change.
        transposed_scores: the capped softmax (``cp=128``, or the CP32 heads layout): the keys of a block are the
            stationary operand of the score matmuls of all rows of a sequence, and the weights stay key-major, so the
            numerator needs no transpose (``_attend_group_rows``). With ``bitwise`` it needs ``fused_masks`` for more
            than one row, and the decode and the verify must both take it, so that their rows stay bitwise equal (at
            CP32 the decode form is ``mla_rows_decode_attention``). Without ``bitwise`` it applies only to the
            non-causal draft form, whose rows share one mask row (``_transposed_scores_on``).
        persistent_rs: take the output reduce-scatter sources from ``sources["rs"]`` (reserved tiles).
        loop_group_cap: the most prior-key groups of one loop iteration on the loop path (0 = the pool-dtype
            default). A smaller value frees group buffers of the loop frame.
        q_hbm: CP32 heads: gather the queries of all rows with one all-gather, in the HBM pair
            ``q_call_src_hbm`` / ``q_call_dst_hbm`` (``sources``, or the standalone arena path), in place of one SBUF
            all-gather for each 4-token tile. Each tile then loads its 32 x 4 gathered rows from HBM. The loaded
            queries are the same bytes, so the outputs do not change.
        out_ag_call: CP32 heads: each pass writes its output reduce-scatter slice to its own columns of one
            ``ag_call`` buffer (``sources``, or the standalone arena), and one all-gather after the last pass
            gathers them all, in place of one all-gather for each 16-row pass. The bytes do not change.
        out_ar_call: CP32 heads, not with ``out_ag_call``: the output sum of each pass is one all-reduce in the HBM
            pair ``out_ar_src_hbm`` / ``out_ar_dst_hbm`` (``sources``, the standalone arena, or a pair of this call),
            in place of the reduce-scatter and the all-gather; with ``fit_passes`` one all-reduce sums the partials of
            all passes. The runtime picks the reduction order of each collective, so whether the output keeps its bits
            is a device check.
        mla_in_proj_scale: None (default, a BF16 ``mla_in_proj``), or the scales of an MXFP8 ``mla_in_proj`` in the
            hT_quad layout of the decode (``mla_decode_attention`` ``mla_in_proj_scale``, ``mla_decode_torch``
            ``mx_weights``): the q_a / kv_a and the gate projections run MXFP8 x MXFP8, with the hidden quantized
            per token on VectorE, so each row gets the bits of the decode. CP32 with 3 local heads and the output
            gate (the target verify) on Trn3 only.
        mla_q_b_proj_scale: None (default, a BF16 ``mla_q_b_proj``), or the scales of an MXFP8 ``mla_q_b_proj`` in the
            hT_quad layout of the decode (``mla_decode_attention`` ``mla_q_b_proj_scale``, ``mla_decode_torch``
            ``mx_weights`` ``{"mla_q_b_proj": "hT_quad"}``): the q_b projection (Step 13) runs MXFP8 x MXFP8, with
            the q_latent quantized per query row on VectorE, so each row gets the bits of the decode. The q_b weight
            is the moving operand (Step 13 keeps q_latent stationary), so the MX form keeps that orientation. CP32
            with 3 local heads on Trn3 only. Independent of ``mla_in_proj_scale``.
        mla_kv_b_proj_scale: None (default, a BF16 ``mla_kv_b_proj``), or the compact ``[16, 512 + 384]`` scales of an
            MXFP8 ``mla_kv_b_proj`` ``[128, 4 * (512 + 384)]`` (``mla_decode_torch`` ``mx_weights``
            ``{"mla_kv_b_proj": "hT_quad"}``). Step 14 (the key absorb) then runs the 3 heads as MXFP8 quadrant row
            tiles at the same time (head ``h`` at PE rows ``32 h``, ``tkg_mx_quant.quadrant_absorb_mx``), with the q_nope
            quantized per query row; the value projection of the tail runs one MXFP8 group a head, with the latent
            quantized per row. The decode twin takes the same key, so each row gets the bits of the decode. CP32 with
            3 local heads on Trn3 only.
        mla_o_proj_scale: None (default, a BF16 ``mla_o_proj``), or the compact ``[16, 7168]`` scales of the MXFP8
            padded-lane ``mla_o_proj`` ``[1, 128, 4 * 7168]`` (``mx_weights`` ``{"mla_o_proj": "hT_quad"}``; lanes 0 ..
            2 the heads, lane 3 zero): each hidden tile of the o_proj is one MXFP8 matmul (the weight stationary, as
            the BF16 form), with the gated value quantized per row. The compact scales are what frees the HBM (the
            [128, 1, 7168] form would hold about as many bytes as the saved weight bytes). CP32 with 3 local heads and
            the output gate on Trn3 only.
        state_bf16_o: the state reduce-scatter of the FP32 HBM pair carries the numerators in BF16 and the sums in
            FP32 (``common.mla_state_bf16_o_views``): two reduce-scatters a pass (or a call, with the one state
            reduce-scatter of ``fit_passes``), issued together, about half the bytes. VectorE casts the numerators
            before the store. The output changes (the numerators lose bits), so a decode twin takes the same key
            (``mla_decode_attention`` ``state_bf16_o``). It needs the FP32 HBM state pair; not with ``bf16_state``
            or one head a rank (``cp`` 128). False (default) keeps the program.
        fit_passes: the bitwise loop path above 16 rows loads the tail weights in each pass after its
            prior-key loop and keeps the local state of the tail dense, so a 256K-table call fits the explicit
            region. With ``sources["rs_call"]`` and ``out_ag_call`` the passes write their output partials to it,
            and one output reduce-scatter a call runs after the last pass. With the state HBM pair of each pass too
            (one state reduce-scatter a call), the passes attend whole sequences (sequence-major), so each sequence's
            keys load once a call. The output is bitwise the same.
        zeroed_pages: the loop path reads its cache pages with no clamp (``_sanitize_cache_rows``), as the decode
            (``zeroed_pages`` of ``tkg_mla_paged.mla_decode_attention``). The static path keeps the clamp. Set it only
            when every row that the loop path can read is finite: each MLA cache writer clamps a valid row to +-448
            (FP8) or +-65536 (BF16), the cache zeroes each MLA page when it allocates it, and an invalid table entry
            points to a page of the same sequence. The result is then the same as with the clamp.
        batched_cache_write: write the current rows with one vector-indirect DMA (SWDGE) in place of one DMA per
            row. The compiler orders every two dynamic-offset writes of the pool, so the per-row writes run one
            after another. Each row that this rank does not own goes to its own row of the null page (row ``r`` for
            query row ``r``), so no two rows of the one write share a target. The values read do not change;
            the null page holds other bytes. ``False`` (default) keeps the program.
        late_tail_weights: the bitwise loop path loads the output gate and o_proj weights in each pass after its
            prior-key loop, as ``fit_passes`` does, also for one pass of up to 16 rows. Held across the loop, they
            leave little room for its key buffers when another body holds part of the explicit region. The output
            is bitwise the same.
        packed_prior: the bitwise loop path attends the prior keys of each pass with one packed loop
            (``_attend_prior_packed``) in place of one loop for each sequence: an item is a key group of a sequence
            with all its Q rows of the pass, and each item runs Q sub-slots, each the slot body of the packed decode
            loop (``tkg_mla_paged._packed_key_loop`` ``queries``) on its row's query, softmax reference and state column.
            Each row thus takes the instructions of its decode step and adds its key groups in the same order after
            its own-key state, so the output is bitwise the same. The pass must hold whole sequences (one pass of up
            to 16 rows, or the sequence-major passes of ``fit_passes``). Trn3, CP32 with 3 local heads, FP8 pages, the
            capped softmax and an FP32 state, an explicit SBUF region, no ``key_start`` or ``transposed_scores``.
            The page loads of the first items wait for the owner writes of the call (a runtime dependency): the later
            rows read the earlier current rows back as prior keys. Default off (the per-sequence loops).
        packed_prior_options: ``packed_prior``: overrides of the keys of ``_PACKED_PRIOR_OPTIONS`` (the packed-loop
            form; every form gives the same bits).
        pass_plans: ``packed_prior`` with ``hoist_plans``: the item plans of the passes of the step and the tiles the
            passes share (``mla_pass_plans``, built once for the MLA layers of a step). Each pass then gathers only its
            page ids. None (default) builds the plan in each pass.
        own_key_select: with ``bitwise``, the own-key numerator of each row comes from one matmul over all current rows
            with a one-hot stationary, not a DMA that shifts the row to partition 0 (as ``packed_prior_options``
            ``own_key_select``, also without ``packed_prior``). The bits are the same except the sign of an exact-zero
            numerator entry. False (default) keeps the program.
        loop_first_static: the loop path (not ``packed_prior``) with at most 2 loop iterations a sequence loads and
            clamps the page-table entries of a sequence once and runs its iteration 0 as static code, so its page loads
            and key transposes can start under the query chain; the loop runs the other iteration. More iterations
            keep the loop. Each row keeps its instructions and group order, so the bits do not change on finite pages.
            False (default) keeps the program; the verify and decode models set it on their CP128 layers.
        scores_ahead: the transposed-scores loop path computes the weights of each pair of key groups before the
            sums of the pair before it, so the PE runs the next score matmuls while the vector and scalar engines
            finish the weights of a pair. Each sum keeps its order, so the bits do not change. False (default)
            keeps the program; the verify and decode models set it on their CP128 layers.
        folded_reference: the transposed-scores loop path adds the softmax reference in the rope matmul (3 BF16
            parts against 3 ones rows of the keys), so one exponential of the scaled scores writes the BF16 weights
            and the vector-engine score scale goes away. Other rounding: the verify and its decode must both take
            it. False (default) keeps the program.
        trimmed_window: ``block_table`` may also have 12 local blocks (3 key groups), the trimmed draft window
            (``kimi_k3_draft_mla_step`` ``win_trim``). The static path runs one key group for each 4 local blocks, so
            the 12 blocks need no other change. False (default) keeps the width rule.
        query_rows: run only the attention passes that start below this token-major row (the rows that feed an
            output of the caller), and give the rows of the later passes a zero attention output. The passes that
            run keep their rows, instructions and collectives, so their rows do not change. Every row still writes its
            cache row (``write_cache``) and takes the query gather, which run before the passes. Only the per-pass
            output path takes it (no output-call, state-call or sequence-major form). 0 (default) runs every pass.
        rows_v_stationary: with the rows form: the values are the stationary operand of the numerator matmuls, one
            matmul for each 128 latent dims and key block with the weights of all rows of the sequence moving (4
            matmuls of R x 96 columns a block, in place of R matmuls of 512). The numerators add up transposed in an
            FP32 tile of the sequence, and PE transposes add them to the state after its last key group. The state then
            takes one add for each row and sequence, so the sums round in another order: the decode and the verify
            must both take it (``mla_rows_decode_attention`` ``rows_v_stationary``). Default off.
        rows_dr: with the rows form on FP8 pages (Trn3): the GPU decode's e4m3 query with FP8 double_row QK keys, as
            ``packed_prior_options`` ``dr_keys`` (form B), with the keys as the stationary: the latent part of the scores
            of a key block is 2 double_row matmuls of the e4m3 key planes and the e4m3 queries of the rows, the key
            planes come from 2-byte transposes of the FP8 rows (2 a block, in place of 4 BF16 ones), and the rope queries
            and own-key scores take the e4m3 query. It changes the bits, so the decode and the verify must both take it
            (``mla_rows_decode_attention`` ``rows_dr``). Default off.
        rows_dr_act_copies: with ``rows_dr``: the 2 byte copies of the key planes of each key group run on ScalarE, not
            VectorE (the same bytes). Default off.
        rows_dr_inplace: with ``rows_dr``: the key tile is the BF16 word tile of the 2-byte key words (the same bytes per
            partition as the planes), so one exact 16-bit word copy of the transposed words replaces the 2 byte copies of
            the planes, and the double_row stationary is a step-2 FP8 view of the word tile. The scores do not change.
            False (default) keeps the program.
        rows_prefetch: with the rows form on the loop path (not ``loop_first_static`` or ``scores_ahead``): the pages of
            the first key groups of each loop iteration load during the iteration before, and those of a sequence's
            first iteration before the loop of the sequence before it, so an iteration does not start by waiting for
            its pages. The same pages and the same instructions on them, so the bits are the same. Default off.
        rows_pair_lag: with the rows form and ``rows_v_stationary`` on the loop path (not ``scores_ahead``): two
            program-order changes of each iteration, so the engines do not wait on the score chain (score matmul,
            exponential, cap, sums). The ones-column sums matmul of a key block runs after the score matmuls of the
            next ``_SUMS_LAG`` blocks; the value matmuls and state adds of a group pair run after the transposes and
            masks of the next pair (the pairs take turns in two weight tiles). The same instructions on the same
            operands, and each accumulation keeps its order, so the bits are the same. Default off.
        rows_pad_skip: TEST ONLY, NOT SERVABLE: with the rows form on the loop path (not
            ``loop_first_static``) a sequence at position 0 is taken as a padding row: it runs no loop iteration (its
            prefetch prologue still loads its head pages), and its weight sums take 1 on each rank, so its output is a
            finite 0. The other
            sequences keep their bits (their trip counts and sums go through selects). In serving a live row can sit
            at position 0 (the first decode step of a 1-token prompt), so a served form needs the runner's live-row
            mask, not the position: ``live_rows``. Default off.
        live_rows: None, or the INT32 ``[B, 1]`` HBM live-row mask of the runner (1: a live request in sequence slot b,
            0: an empty slot). With ``rows_pad_skip`` or the packed loop's ``pad_skip`` the padding sequences are those
            with mask 0, not those at position 0 (the served form of the pad skip).
        rows_pair_tail: with the rows form on the loop path (fused masks, ``rows_prefetch``, iterations of 2 group
            pairs, as the k1 rows verify runs them): a sequence whose last iteration would hold only masked keys in its
            second pair (ceil(g / 2) group pairs odd) runs that iteration as its first pair alone, so at 16K (one group
            per rank) 1 group pair runs, not 2. The pair keeps its instructions and order, and the dropped pair added
            exact zero weights, so the state keeps every bit (an exact-zero numerator entry can keep a -0 that the zero
            adds made +0). Not with ``rows_mx_keys`` (an unmeasured pair, refused). Default off.
        rows_unroll2: with ``rows_pair_tail``: each trip of the loop of a sequence runs two of its iterations (8 key
            groups), so the loop takes half the back edges and their end-of-body syncs; an odd iteration left over runs
            after the loop, then the pair tail. Every iteration keeps its instructions on the same operands in the same
            order, so the bits are those of ``rows_pair_tail``. Applies where the table holds whole trips. Default off.
        rows_seq_hoist: with the rows form on the loop path: one set of group buffers serves every sequence of the
            layer, set up before the first loop, so the constant rows of the folded reference are written once and
            not at each sequence start. The same instructions on the same operands (the constant rows once), so the
            bits are the same. Default off.
        rows_setup_ahead: with the rows form on the loop path and ``folded_reference``: the reference columns, the
            folded reference rows of the queries and the latent query tiles (the double_row query planes) of every
            sequence are made before the first loop, not at the start of each sequence's loop, so a sequence start
            waits less for VectorE. The query tiles of the later sequences stay live until their loops (about 1.5 KB
            a partition each with ``rows_dr``). The same instructions on the same operands, so the bits are the same.
            Default off.
        rows_q_fp8: with ``rows_dr`` (or ``rows_mx_keys`` "single") and ``q_hbm`` (the one query gather of the call):
            each rank rounds its queries to e4m3 before the gather, so the query gather pair is e4m3 (half the bytes of
            BF16) and the load casts the rows back to BF16 (exact), as the packed loop's ``q_fp8`` does. ``rows_dr``
            rounds every use of the query (the double_row query planes, the rope queries and the own-key scores) to
            e4m3 with the same clamp, and the rounding of an e4m3 value is the value, so the bits are the same. The
            ``q_call`` HBM pair must be e4m3. Default off.
        state_rs_hook: a function of the layer's allocator (``sb``), traced once in its own scope right after the
            issue of the last state reduce-scatter of the call (the BF16-o form of each pass, the one call-level
            reduce-scatter, or the FP32 reduce-scatter of a one-pass call), before the state load that waits for it. A caller gives it work that does not depend on
            the layer (``decode_block_body`` ``attn_res_mla_mlp_early``), so VectorE runs it in the flight of the
            collective. None (default) keeps the program.
        out_ar_hook: a function ``hook(sb, after)`` of the layer's allocator, traced once in its own scope right after the
            issue of the call's output all-reduce (``out_ar_call`` with the call-level partials, ``rs_call``), before
            the load that waits for it. ``after`` holds the store of the tail's partials into the all-reduce source:
            the hook's work follows it in the schedule (an edge, not a wait), so it runs after the tail, in the flight
            of the all-reduce. A caller gives it work that does not depend on the layer (``decode_block_body``
            ``attn_res_mla_mlp_tail_entries``). None (default) keeps the program.
        rows_mx_keys: None (default, no change) or "single": the rows form on FP8 pages (Trn3) runs the latent part of
            each score as an MX matmul (``float8_e4m3fn_x4``, unit E8M0 scales, ``_KeyPath`` "mx_single") of FP8 keys and
            an FP8 query, in place of the default BF16 key transposes. One 32-bit PE transpose a key block gives the MX
            keys (``_mx_key_prep``); the query words come from the gather (``_mx_query_words``). "single" rounds only the
            latent query to e4m3 (the MX QK); the rope query and the own-key score stay BF16 (unless ``rows_q_fp8``
            rounds the whole gather), so it is more precise than the packed form B, which rounds the whole query to
            e4m3. Accuracy class (not bitwise). Not with ``rows_dr`` / ``rows_dr_inplace`` / ``rows_dr_act_copies``.
            Default off.
        loop_p2: the transposed-scores loop path keeps the state of the pass in the (row, head) layout of
            ``packed_prior_options`` ``p2``: state column ``g S + s`` holds the 128 (row, head) columns
            ``128 g .. 128 g + 127`` of sequence ``s`` (``pack_p2_runs``). The numerators of each key block then take
            one 128-column stationary of the weights for each such group, Q x 96 / 128 matmuls in place of Q (3 in
            place of 4 at Q = 4, 6 in place of 8 at Q = 8), and the tail stores the state head-major
            (``_store_state_loop_p2``, with the pad heads of CP128). Each state value takes the same products and adds in
            the same order (the own-key state too, in the form of ``own_key_select``), so the output is bitwise the
            same. Needs ``bitwise`` on the loop path with transposed scores, Q x 96 a multiple of 128, one pass of at
            most 16 rows, an FP32 HBM state reduce-scatter for each pass, and no ``packed_prior``, ``bf16_state``,
            ``state_bf16_o``, ``rows_v_stationary``, ``rows_pair_lag`` or ``rows_mx_keys``. False (default) keeps the
            program.
        loop_preload: with ``loop_first_static`` and 2 loop iterations a sequence, the static iteration 0 loads the
            pages of the first key groups of iteration 1 into their group buffers after its own last read of each, and
            the loop does not load them again (``_attend_prior_loop`` ``preload``), so the loop does not start by waiting
            for its pages. The same pages and instructions, so the output is bitwise the same. Not with
            ``rows_mx_keys``. False (default) keeps the program.
        critical_dma_priority: give the small critical DMAs of the layer the trn3 DMA QoS priority 0
            (``common.dma_priority``): the per-row token-info loads, the owner cache write, and the state
            reduce-scatter source store and result load. Without a priority the compiler can queue these
            tiny latency-critical transfers behind the expert-weight stream. The bytes do not change, so the
            output is bitwise the same (``None`` priority on trn2, which has no DMA QoS). Default off.
            ``"query"``: also the query path of the k1 call (the query store of the call-level query gather and the
            per-token loads of the gathered queries) and the call-level state store and load around the one state
            reduce-scatter of the call, each with priority 0 on the SP hardware DGE queue in place of the default
            queue. The same bytes, so the output is bitwise the same.
        cache_write_off_sp: issue the per-row owner cache write on the Activation HWDGE queue, off the SP
            queue that the prior-page loads use, so the first page load does not wait behind the owner write.
            A data dependency, not the queue, orders a current-block read after the write (``bitwise``), so the
            output is bitwise the same. Ignored with ``batched_cache_write`` (SWDGE, already off SP). Default off.
        own_rows_ahead: with ``bitwise`` (not ``own_key_select``, ``packed_prior`` or ``loop_p2``), each pass loads the
            current rows of its query rows at its start, on the Activation HWDGE queue, into one tile: row ``i`` of the
            pass on partition ``32 (i % 4)`` of 512-column block ``i // 4``, so one DMA moves 4 evenly spaced rows. The
            own-key weight of the pass is set once on those partitions. ``_own_key`` then reads its row and weight there
            in place of a DMA that shifts the row to partition 0 and a memset, so the rows of a pass do not wait for
            each other's DMA. The one-partition matmul reads the same row and weight values, so the output is bitwise the
            same. Default off.
        o_proj_ahead: with ``late_tail_weights`` on the loop path (one pass, not ``late_o_proj_weights`` or
            ``late_tail_frames``) and the BF16 call query gather, the o_proj weight chunks load before the prior-key
            loop, on the Activation HWDGE queue, after the pass's query loads (so after the flight of the q-call
            gather), into heap tiles that live through the loop; the tail reads them in place of loads after the loop,
            which share the DMA rings with the state reduce-scatter. The same bytes into the same matmuls, so the
            output is bitwise the same. Default off.
        gemv_bands: run the narrow x-stationary GEMVs (``q_b`` and the key absorb) in two PE column bands
            (``tile_position`` / ``tile_size``), so two matmuls with a stationary free dimension of at most 64
            share the 128 systolic-array columns. Column tiling does not change the accumulation, so the output
            is bitwise the same. Applied only when ``Q*B <= 64`` (else the full array). Default off.
        fused_single_pass: ``fused_masks`` also for a B that does not divide 16 (B = 3, 5, 6, 7), when the call is one
            pass of at most 16 rows: the one-hot of query row ``j = cq // B`` then covers all rows of the pass. The
            masked keys get the same zero weights, so the values do not change. False (default) keeps the program.
        late_o_proj_weights: with ``late_tail_weights`` (one tail for each pass), each pass loads its o_proj weights
            after its value projection, the last reader of the pass state, instead of with the gate weights before
            the state load: the 43 KB of o_proj weights and the pass state are then never live together. The output
            is bitwise the same. False (default) keeps the program.
        late_tail_frames: with ``late_o_proj_weights`` on the FP32 HBM state reduce-scatter (no fit passes), each pass
            loads its gate weights and runs its gate projection in a frame of its own, which ends before the state
            load, and loads its local state dense (``dense_state``, 2 KB per partition). In an explicit SBUF region
            the state, value and o_proj tiles of the tail then take the bytes of the gate weights, so the tail stays
            under the top of the prior-key loop's tiles. The output is bitwise the same. False (default) keeps the
            program.
        loop_cast_engines: Trn2 FP8 pages, the plain prior-key loop of the loop path (the page forms of ``tkg_mla_paged``
            ``loop_cast_engines``): "dma", "dma6" or "dma4". The page loads of each key group write BF16 group tiles
            (the DMA casts the FP8 bytes exactly), the clamp keeps the FP8 bound, and the transposes and the value
            matmuls read the BF16 rows, so the 20 FP8 -> BF16 copies of each group (``_transpose_rows``) go. The loop
            then holds at most 8 / 6 / 4 groups an iteration (and at most ``loop_group_cap``): the largest even count
            that divides the key groups of the table (``rows_cast_groups``; 8 / 4 / 4 on the 1M table). BF16 tiles take
            twice the FP8 bytes, so 4 groups take the SBUF of 8 FP8 groups. The groups keep their pairs and
            their order, so the output is bitwise the same. "off" (default) keeps the program. Not with
            ``packed_prior``, ``rows_prefetch``, ``rows_seq_hoist``, ``rows_dr`` or ``rows_mx_keys``.
        pass_q_prefetch: with the call query gather (``q_hbm``) and more than one pass, the first ``pass_q_prefetch``
            gather tiles (4 rows each) of each later pass load into heap tiles on the ScalarE hardware DGE queue
            right after the query gather of the pass before, so they do not wait for that pass's loop, its state
            store or the SP queue. The pass then transposes them as without the prefetch. The same bytes and the same
            instructions, so the output is bitwise the same. The tiles take ``pass_q_prefetch`` x 3,456 bytes a
            partition for the whole call. At most ``PASS_Q_PREFETCH_MAX_TILES`` (2): a larger value fails at trace time.
            A form without the room for the tiles fails at trace time in the region. 0 (default) keeps the program.
        q_gather_buffers: with the call query gather (``q_hbm``), the stack tiles that the query gather tiles of a pass
            load into. 1 (default) keeps the program: each gather tile takes a new tile in its own scope, so it takes
            the bytes of the tile before, and its loads wait until TensorE has read that tile. 2: two tiles in turn,
            in a scope of the gather only, so the loads of a tile wait only for the reads of the tile two before. The
            loads then overlap the transposes of the tile before. The same bytes and the same instructions, so the
            output is bitwise the same. The second tile takes 3,456 bytes a partition during the gather only; the
            packed loop does not hold it. A pass with fewer than 2 loaded gather tiles keeps one tile.
        tail_weights_after_rs: with the call-level state reduce-scatter (``fit_passes``, the BF16-o state and the call
            sources), the output gate and o_proj weight DMAs of the tail wait for the numerator state reduce-scatter
            (a runtime dependency), and they load on the ScalarE hardware DGE queue. Without it they load on the SP
            queue during the reduce-scatter, and the weight transfer delays the reduce-scatter on the shared DMA
            engines. The state loads of the tail stay on their queues, so they are not queued behind the weights. The
            same bytes and the same tiles, so the output is bitwise the same. False (default) keeps the program.
        sums_rs_first: the BF16-o state's sums before its numerators, in the state stores and the reduce-scatters
            (and with ``tail_weights_after_rs`` the gate weights after the sums one): None (default) takes
            ``SUM_RS_FIRST``; False gives the numerators first. The same bits.
        pass_rows_ahead: with ``packed_prior`` ``p2``, the call query gather (``q_hbm``), one state store a call
            (``state_call``) and more than one pass: after the loop of each pass and before its state stores, the next
            pass's current-row loads of its own-key state and its query gather tiles past the ``pass_q_prefetch`` tiles
            load into heap tiles, on the queues that they take without this option. So the stores of a pass do not hold
            those loads back on the SP queue. The next pass frees the tiles after its own-key state, before its loop.
            With ``p2`` (any number of passes) the current rows of a pass load 4 rows a DMA (rows evenly spaced in the
            current rows; ``_own_rows_p2``), in place of one DMA a row. The same bytes, so the output is bitwise the
            same. The heap tiles live from the end of a pass's loop to the next pass's own-key state, not in a loop.
            False (default) keeps the program.
        loop_page_gather: the page loads of the plain prior-key loop of the loop path (``tkg_mla_paged``
            ``LOOP_PAGE_GATHERS``, as its per-token ``loop_page_gather``): "off" (default) keeps the program (one SP DMA a
            page); "group" loads the 4 pages of each key group with one GpSimd software DGE gather (512 rows of 576 bytes
            into the group's own page tile, the same bytes in the same places); "carry" also gathers group 0 of the next
            iteration inside each iteration, once group 0's tile has no reader left; "carry_pool" is "carry" with the
            gather indices of the next iteration built on GpSimd. The first indices of every sequence (and with "carry"
            its group 0, into a tile of its own) come before the first loop. With ``loop_cast_engines`` the gathers
            write the BF16 group tiles (the DMA casts the FP8 bytes, as the page loads of that form do). FP8 pages, an
            explicit SBUF region, the plain loop (no transposed scores, ``loop_first_static`` or ``packed_prior``). The
            output is bitwise the same.
        weights_on_act: the plain prior-key loop of the loop path (not transposed scores or ``packed_prior``) copies
            the transposed weights of each query row and key group out of PSUM on ScalarE in place of VectorE, as the
            loop does with ``fit_passes`` above 16 rows. A plain copy, so the output is bitwise the same. False (default)
            keeps the program.
        stack_rows: the draft form of the static path (not bitwise, not causal, the capped softmax, one shared mask row,
            the BF16 key path, no transposed scores) attends each prior-key group of a sequence for several of its rows of
            the pass at once (``_attend_group_stacked``; 2 rows of up to 64 heads, 4 of up to 32): row ``u`` of a unit
            takes the stationary columns (and the PSUM partitions) ``step u .. step u + heads``, so one set of score matmuls, score cap, exp, weight transposes, weight
            copy and value matmuls serves the unit in place of one set for each row. Each output column of a matmul
            reads only its own stationary column, and the cap, exp and copy work per partition, so each row gets the
            numerator and the weight sum of the per-row form, and its state takes the same adds in the same order. The
            pad columns are zero and their partitions are never read. False (default) keeps the program.
        loop_row_pipeline: the plain prior-key loop of the loop path with fused masks and the capped softmax runs the
            rows of each iteration as a two-stage software pipeline: the scores, cap and exponentials of a (group, row)
            item come before the weight transposes, value matmuls and state add of the item before it, in two weight
            tile sets (``_attend_prior_loop`` ``row_pipeline``). Each row keeps its instructions and group order, so the
            output is bitwise the same. False (default) keeps the program.
        q_prefetch_after: with ``pass_q_prefetch``, instructions (of the caller) that every prefetch load follows in the
            schedule: a schedule edge, no hardware wait, so the loads and their bytes do not change. () (default) keeps
            the program.
        pass_tail_overlap: the CP32 static path with the FP32 HBM state pair runs the output tail of each pass after the
            attention of the next pass: pass ``p`` stores its state and issues its state reduce-scatter, pass ``p + 1``
            attends, then pass ``p`` loads its reduced state and computes its output partial, and after pass ``p + 1``
            issues its state store and reduce-scatter, pass ``p`` runs its output reduce-scatter and all-gather. The
            passes alternate between the state pair and the second pair of ``mla_block_collective_hbm``
            (``state_pairs`` 2, in ``sources``), so a pass's reduce-scatter never writes the state that the pass before
            still has to load. The state reduce-scatter of pass ``p`` follows the final all-gather of pass ``p - 2`` and
            the partial copies of pass ``p - 1`` (its output reduce-scatter source), and the output reduce-scatter of
            pass ``p`` the final all-gather of pass ``p - 1``, so a buffer of a parity is reused only after a later
            all-gather has completed, as in the default order, and no local write lands in a collective's window. Each
            pass's output all-gather lands in its own persistent tile (``sources["y_dst"]``). The same instructions and the same
            collective sizes, so the outputs do not change. A call of one pass (16 rows or fewer) keeps its program.
            False (default) keeps the program.
        seq_passes: the CP32 static path with the layer query gather in HBM (``q_hbm``) runs its passes on whole
            sequences: pass ``p`` holds the ``Q`` rows of each of the ``16 // Q`` sequences ``p * 16 // Q ..`` (14 rows
            for the ``Q`` = 7 drafter, padded to the 16-row tail), in place of 16 token-major rows that hold rows of
            every sequence. Each sequence's prior keys are then loaded and prepared in one pass a layer, not in every
            pass. Each row takes the same scores, weights, value matmuls and state adds in the same key-group order,
            the state reduce-scatter keeps its 16-row message, and the output reduce-scatter sums each element over the
            same ranks, so the outputs do not change. The pass inputs and outputs move by row runs, as on the CP128
            sequence-major path (with ``out_ag_call``, from the one gathered tile of the call). The batch must split into
            whole passes (``B`` a multiple of ``16 // Q`` with ``ceil(Q B / 16)`` passes). A call of one pass keeps its
            program. False (default) keeps the program.
        state_store_split: the FP32 HBM state store of each pass (the store before the state reduce-scatter, without a
            ``p2`` store or the BF16-o state) runs as two DMAs of the column halves, on the SP and the ScalarE HWDGE
            queues, in place of one DMA. The same bytes land, so the outputs do not change. False (default) keeps the
            program.
        loop_tail_groups: the plain prior-key loop of the loop path (not transposed scores, ``packed_prior``,
            ``loop_first_static``, ``loop_preload``, ``rows_prefetch``, ``rows_seq_hoist`` or ``rows_pair_tail``) runs the
            last iteration of a sequence in a loop of its own with only its first 4, 2 or 1 key groups (``LOOP_TAILS``)
            when the iteration would hold 1 to that many groups with a prior key: one table serves every context length,
            and each iteration attends all of its groups, masked or not. The cascades 7 ("tail421": tails of 4, 2 and 1
            groups) and 3 ("tail21": 2 and 1) take one tail loop for each tail and run that iteration in the smallest
            tail that holds its groups ("tail421": 1 group in the 1-group tail, 2 in the 2-group tail, 3 or 4 in the
            4-group tail). The groups dropped hold only masked keys (exact zero weights), so the output is bitwise the
            same (an exact-zero state entry can keep a -0 that the zero adds made +0). Each tail applies where an
            iteration holds more groups than it; 0 (default) keeps the program. The full-iteration forms 8 ("tail8")
            and 15 ("tail8421": the cascade too) also run the last iteration of every other sequence with a key group in
            a loop of its own, with all its groups, so every masked row (the rows past each sequence's keys, its null
            page 0 entries, and a padding or empty sequence, whose only iteration is its last) is read in a tail. The
            main loop then reads only prior keys of its sequence and skips the page clamp (``_sanitize_cache_rows``;
            the tails keep it). On finite cache rows the clamp leaves the prior keys unchanged, so the output is
            bitwise the same.
        prior_hoist: with ``stack_rows``, the page loads and the TensorE key transposes of the first ``prior_hoist`` key
            groups of each sequence of a pass are traced before that pass's sequence loop (``_attend_prior_sequences``),
            so they can run while the call's query gather is in flight. The loop then attends those groups from the
            hoisted tiles. The same loads and transposes on the same values, so the outputs do not change. 0 (default)
            keeps the program.

    Returns:
        y_hT: ``[128, Q*B, 56]`` BF16 SBUF attention output after the TP all-reduce, token-major hT.
        Its ``side_effects`` holds the cache-row write DMAs and ``side_effect_sources`` their SBUF source.
        split: None (default; the program does not change), or the long-sequence split across two DP replicas
            ``{"group": the pair ReplicaGroup of the TP ranks t of the replicas (r, r xor 1), "info": [1, 4] INT32 HBM
            (role, X, 0, 0) of this replica}``. The caller gives the batch of B' = B + 2 slots: the B request slots,
            then the foreign slot B' - 2 and a pad slot B' - 1 (zero inputs, slot 0 writes). The donor sequence of an
            owner (role 1) sits in slot B' - 3; it attends its key groups X / 16384 .. (X a multiple of 16384), and a
            helper (role 2) attends groups 0 .. X / 16384 - 1 of the same sequence in its foreign slot, from its own
            pages in that block-table row. The helper's foreign rows take the owner's q_a | kv_a (a pair all-reduce
            after the projection), so its softmax reference is the owner's; their own-key weight is 0 (the owner adds
            the current key). After the call-level state reduce-scatter a pair all-reduce gives the owner the helper's
            state of its donor rows, added before ``num / sum``. Role 0 on both: every output keeps the bits of the
            call without the split. The outputs of the foreign and the pad slot are not meaningful. Needs the
            bitwise packed loop (``packed_prior``) with the call-level state reduce-scatter and the sequence-major
            passes; v1: B' = 4 (C=8, the k7b2 batch), Q = 8. Accuracy class for the donor rows (the FP32 sum order).
            v5 (``{"form": "v5", "dp_group": the DP ReplicaGroup, "dp": its ranks, "foreign_slots": F}``, the CC
            skeleton of the split program): no extended batch; one DP all-gather of the call's q_a | kv_a rows after
            the projection (the import) and one DP all-gather of an ``[F Q, 3 x 514]`` BF16-o state block after the
            call-level state reduce-scatter (the export). Their results are not read yet: the outputs keep the bits of
            the call without the split.
    """
    total_rows = x_hT.shape[1]
    tokens = position.shape[0]
    local_blocks = block_table.shape[1]
    lay = cp_layout(cp, local_heads, heads)
    heads = lay.heads
    gate_local = local_heads * _V_HEAD
    absorb_width = local_heads * _KV_LORA
    gather_width = absorb_width + local_heads * _ROPE
    qkv_region = _HIDDEN_TILES * lay.qkv_local
    scale = softmax_scale if softmax_scale is not None else _Q_HEAD**-0.5
    multi_pass = total_rows > _TAIL_ROWS
    tr = _TAIL_ROWS if multi_pass else total_rows
    arena_qkv_row = None
    arena_q_tiles = None
    arena_ag_bufs = None
    arena_qkv_all = None
    arena_q_dsts = None
    arena_state_hbm = None
    arena_state_dst_hbm = None
    arena_q_call = None
    arena_out_ag = None
    arena_out_ar = None
    q_tile_idx = 0
    kernel_assert(lora_norm_eps > 0, f"lora_norm_eps must be positive, got {lora_norm_eps}")
    kernel_assert(
        not out_ar_call or not (out_ag_call or lay.one_head),
        "out_ar_call is for CP32 heads without out_ag_call (one head per rank already all-reduces its output)",
    )
    if bitwise:
        # The bitwise verify reads the current block back from the pool, so it must write the rows. A
        # runtime edge in _attend_prior orders each current-block read after the owner-write.
        kernel_assert(causal and rope_rows is None, "bitwise is the verify mode: causal and NoPE")
        write_cache = True
    # Mask rows of each key group (``fused_masks``): one for each query row of a sequence, or the one shared row.
    fuse = (
        fused_masks
        and bitwise
        and q_rows > 1
        and window_base is None
        and (_TAIL_ROWS % tokens == 0 or (fused_single_pass and total_rows <= _TAIL_ROWS))
    )
    n_mask = q_rows if fuse else 1
    ts = _transposed_scores_on(transposed_scores, bitwise, causal, exact_softmax, n_mask, q_rows)
    kernel_assert(_BIAS_ROW + n_mask <= _P_MAX, f"{n_mask} mask rows do not fit below the rope keys")
    sb = AUTO_SBUF if sbuf is None else sbuf
    if fp8_pages and kv_pool.dtype == nl.bfloat16:
        kv_pool = fp8_kernel_pages(kv_pool.view(fp8_page_dtype()), _PAGE_WIDTH)
    elif kv_pool.dtype == nl.float8_e4m3fn and not is_trn3():
        kv_pool = fp8_kernel_pages(kv_pool.view(fp8_page_dtype()), _PAGE_WIDTH)
    kernel_assert(
        loop_cast_engines in ROWS_LOOP_CASTS,
        f"loop_cast_engines must be one of {tuple(ROWS_LOOP_CASTS)}, got {loop_cast_engines!r}",
    )
    bf16_pages = loop_cast_engines != "off"
    if bf16_pages:
        kernel_assert(
            loop_path
            and not is_trn3()
            and kv_pool.dtype != nl.bfloat16
            and not (packed_prior or rows_prefetch or rows_seq_hoist or rows_dr or rows_mx_keys),
            "loop_cast_engines is a form of the plain prior-key loop on Trn2 FP8 pages (loop_path; not packed_prior, "
            "rows_prefetch, rows_seq_hoist, rows_dr or rows_mx_keys)",
        )
        cast_groups = rows_cast_groups(loop_cast_engines, local_blocks // _GROUP_BLOCKS, loop_group_cap)
        kernel_assert(
            cast_groups > 0,
            f"loop_cast_engines {loop_cast_engines!r}: no even group count of at most "
            f"{ROWS_LOOP_CASTS[loop_cast_engines]} divides the {local_blocks // _GROUP_BLOCKS} key groups of the table",
        )
        loop_group_cap = cast_groups
    kernel_assert(
        not weights_on_act or (loop_path and not packed_prior),
        "weights_on_act is a form of the plain prior-key loop of the loop path (not packed_prior)",
    )
    kernel_assert(
        not stack_rows
        or (
            not loop_path
            and not bitwise
            and not causal
            and not exact_softmax
            and not ts
            and n_mask == 1
            and not packed_prior
            and not (rows_dr or rows_dr_inplace or rows_mx_keys)
        ),
        "stack_rows is a form of the draft static path: not loop_path, bitwise, causal, exact_softmax, transposed "
        "scores, fused masks, packed_prior or a rows key form",
    )
    kernel_assert(
        prior_hoist >= 0 and (not prior_hoist or stack_rows),
        f"prior_hoist (key groups hoisted a sequence) needs stack_rows; got {prior_hoist}",
    )
    kernel_assert(
        not loop_row_pipeline or (loop_path and bitwise and not packed_prior and not exact_softmax),
        "loop_row_pipeline is a form of the bitwise plain prior-key loop of the loop path with the capped softmax (not "
        "packed_prior)",
    )
    kernel_assert(
        loop_tail_groups in LOOP_TAILS.values(),
        f"loop_tail_groups must be one of {sorted(LOOP_TAILS.values())}, got {loop_tail_groups!r}",
    )
    kernel_assert(
        not loop_tail_groups
        or (
            loop_path
            and not ts
            and not (packed_prior or loop_first_static or loop_preload or rows_prefetch or rows_seq_hoist)
            and not rows_pair_tail
        ),
        "loop_tail_groups is a form of the plain prior-key loop of the loop path (no transposed scores, packed_prior, "
        "loop_first_static, loop_preload, rows_prefetch, rows_seq_hoist or rows_pair_tail)",
    )
    kernel_assert(
        loop_page_gather in LOOP_PAGE_GATHERS,
        f"loop_page_gather must be one of {LOOP_PAGE_GATHERS}, got {loop_page_gather!r}",
    )
    kernel_assert(
        loop_page_gather == "off"
        or (
            loop_path
            and kv_pool.dtype == fp8_page_dtype()
            and sb.explicit
            and not (packed_prior or loop_first_static or loop_preload)
            and kv_pool.shape[0] * _BLOCK <= 2**24
        ),
        "loop_page_gather changes the page loads of the plain prior-key loop: it needs the loop path on FP8 pages, an "
        "explicit SBUF region (its index tiles live across the back edge), a pool of at most 2**17 pages, and not "
        "packed_prior, loop_first_static or loop_preload",
    )
    kernel_assert(
        not rows_dr or (is_trn3() and kv_pool.dtype == nl.float8_e4m3fn and not packed_prior),
        "rows_dr needs the rows form on FP8 pages on Trn3, not packed_prior",
    )
    kernel_assert(rows_mx_keys in (None, "single"), "rows_mx_keys must be None or 'single'")
    kernel_assert(
        rows_mx_keys is None or (is_trn3() and kv_pool.dtype == nl.float8_e4m3fn and not packed_prior),
        "rows_mx_keys needs the rows form on FP8 pages on Trn3, not packed_prior",
    )
    kernel_assert(
        rows_mx_keys is None or not (rows_dr or rows_dr_inplace or rows_dr_act_copies),
        "rows_mx_keys does not combine with rows_dr, rows_dr_inplace or rows_dr_act_copies",
    )
    _validate_inputs(
        x_hT,
        mla_in_proj,
        mla_q_b_proj,
        mla_kv_b_proj,
        mla_o_proj,
        kv_pool,
        block_table,
        slot,
        position,
        rope_rows,
        lay,
        q_rows,
        output_gate,
        total_rows,
        tokens,
        _MAX_FIT_SEQS if fit_passes else _MAX_SEQS,
        mla_in_proj_scale,
        trimmed_window,
        mla_q_b_proj_scale=mla_q_b_proj_scale,
        mla_kv_b_proj_scale=mla_kv_b_proj_scale,
        mla_o_proj_scale=mla_o_proj_scale,
    )
    kernel_assert(
        local_blocks <= _MAX_UNROLLED_BLOCKS or loop_path,
        f"block_table has {local_blocks} local blocks, which needs the loop path. Use at most "
        f"{_MAX_UNROLLED_BLOCKS} local blocks, or set loop_path=True.",
    )
    kernel_assert(not (trimmed_window and loop_path), "trimmed_window takes the static path: set loop_path=False")
    pack_options = None
    if packed_prior:
        kernel_assert(
            bitwise
            and is_trn3()
            and cp == _TP
            and local_heads == 3
            and loop_path
            and local_blocks > _MAX_UNROLLED_BLOCKS
            and kv_pool.dtype == nl.float8_e4m3fn
            and not exact_softmax
            and not bf16_state
            and key_start is None
            and window_base is None
            and not transposed_scores
            and sb.explicit,
            "packed_prior needs the bitwise loop path (more than 16 local blocks) on Trn3 with CP32, 3 local heads, FP8 "
            "pages, the capped softmax, an FP32 state and an explicit SBUF region, and no key_start, window_base or "
            "transposed_scores",
        )
        _check_packed_softmax_scale(scale)
        unknown = set(packed_prior_options or {}) - set(_PACKED_PRIOR_OPTIONS)
        kernel_assert(not unknown, f"packed_prior_options takes only {sorted(_PACKED_PRIOR_OPTIONS)}, got {unknown}")
        pack_options = {**_PACKED_PRIOR_OPTIONS, **(packed_prior_options or {})}
        check_no_if_loops_options(pack_options)
        kernel_assert(
            pass_plans is None or pack_options["hoist_plans"] or pack_options["fp8_pv_step_store"],
            "pass_plans is the plan of packed_prior_options hoist_plans (or the FP8 P@V step stores of fp8_pv_step_store)",
        )
        kernel_assert(
            not pack_options["fp8_pv_step_store"] or (pass_plans is not None and "items" in pass_plans),
            "packed_prior_options fp8_pv_step_store needs the step stores of mla_fp8_pv_step_stores as pass_plans",
        )
        kernel_assert(
            not pack_options["step_items"] or (pass_plans is not None and "items" in pass_plans),
            "packed_prior_options step_items needs the pass_plans of hoist_plans with their step items "
            "(mla_pass_plans with step_items)",
        )
        kernel_assert(
            not pack_options["dr_keys"] or not lay.one_head,
            "packed_prior_options dr_keys needs the gathered queries of 3 local heads (not the one-head layout)",
        )
        kernel_assert(
            not pack_options["mx_keys"] or (pack_options["dr_keys"] and not pack_options["p2"] and q_rows > 1),
            "packed_prior_options mx_keys needs dr_keys (the e4m3 query) and Q > 1 rows, and no p2",
        )
        kernel_assert(
            not pack_options["mx_qk"] or (pack_options["p2"] and pack_options["word_keys"] and q_rows > 1),
            "packed_prior_options mx_qk needs p2 and word_keys (its key words) with Q > 1 rows",
        )
        kernel_assert(
            not pack_options["mx_natural"]
            or (pack_options["mx_qk"] and not lay.one_head and bitwise and not pack_options["own_key_selects"]),
            "packed_prior_options mx_natural needs mx_qk (its MX query words), the gather tiles of the queries and the "
            "bitwise own-key scores without own_key_selects",
        )
        kernel_assert(
            not pack_options["own_state_stack"] or (pack_options["p2"] and q_rows % 4 == 0 and not pass_rows_ahead),
            "packed_prior_options own_state_stack needs p2 with Q a multiple of 4, and no pass_rows_ahead (it loads its "
            "own rows)",
        )
        check_q_fp8_options(pack_options, q_hbm)
    kernel_assert(packed_prior or packed_prior_options is None, "packed_prior_options needs packed_prior")
    kernel_assert(
        not rows_q_fp8 or ((rows_dr or rows_mx_keys == "single") and q_hbm and not packed_prior and not lay.one_head),
        "rows_q_fp8 needs rows_dr or rows_mx_keys 'single' and q_hbm (the one query gather) with the 3-head layout",
    )
    kernel_assert(not (loop_first_static and packed_prior), "loop_first_static is a form of the per-sequence loop path")
    kernel_assert(
        ts
        or not (
            rows_v_stationary
            or rows_dr
            or rows_prefetch
            or rows_dr_inplace
            or rows_pair_lag
            or rows_pad_skip
            or rows_seq_hoist
            or rows_setup_ahead
            or rows_q_fp8
        ),
        "the rows keys need the rows form",
    )
    kernel_assert(not rows_dr_inplace or rows_dr, "rows_dr_inplace is a form of rows_dr")
    kernel_assert(
        not loop_preload or (loop_first_static and not packed_prior and rows_mx_keys is None),
        "loop_preload is a form of loop_first_static, without packed_prior or rows_mx_keys",
    )
    kernel_assert(
        not loop_p2
        or (
            bitwise
            and ts
            and loop_path
            and local_blocks > _MAX_UNROLLED_BLOCKS
            and not packed_prior
            and not bf16_state
            and not state_bf16_o
            and not rows_v_stationary
            and not rows_pair_lag
            and rows_mx_keys is None
            and total_rows <= _TAIL_ROWS
            and q_rows * lay.heads % _P_MAX == 0
        ),
        f"loop_p2 needs the bitwise transposed-scores loop path (more than {_MAX_UNROLLED_BLOCKS} local blocks) on one "
        f"pass of at most {_TAIL_ROWS} rows with Q x heads a multiple of {_P_MAX}, and no packed_prior, bf16_state, "
        f"state_bf16_o, rows_v_stationary, rows_pair_lag or rows_mx_keys: got Q {q_rows}, heads {lay.heads}, "
        f"rows {total_rows}, transposed scores {ts}",
    )
    kernel_assert(
        not rows_pad_skip or (loop_path and not loop_first_static), "rows_pad_skip is a form of the rows loop path"
    )
    kernel_assert(
        rows_mx_keys is None or (ts and not lay.one_head),
        "rows_mx_keys needs the rows form (transposed_scores) with the CP32 gathered queries (not the one-head layout)",
    )
    kernel_assert(
        not rows_pair_tail
        or (loop_path and transposed_scores and rows_prefetch and not loop_first_static and rows_mx_keys is None),
        "rows_pair_tail is a form of the rows loop path with rows_prefetch, not with rows_mx_keys",
    )
    kernel_assert(not rows_unroll2 or rows_pair_tail, "rows_unroll2 is a form of rows_pair_tail")
    split_v1 = split is not None and "info" in split
    split_v5 = split is not None and split.get("form") == "v5"
    kernel_assert(
        not (split_v5 and split.get("ext") is not None)
        or (
            packed_prior
            and bitwise
            and cp == _TP
            and local_heads == 3
            and q_rows == 8
            and split["foreign_slots"] in (2, 4)
            and tokens == split["ext"] + split["foreign_slots"]
            and split["ext"] % 2 == 0
            and not split.get("fold")
        ),
        "split (v5) ext needs the bitwise packed loop on CP32 with 3 local heads, Q = 8 and B + F slots (F = 2 or 4: "
        f"whole passes of 2 foreign slots after the even B), without the fold; got B' = {tokens}, Q = {q_rows}",
    )
    # The v5 extended batch (the DEV split program): B = split["ext"] request slots, then the F foreign slots, 2 a pass.
    split_ext = split_v5 and split.get("ext") is not None
    kernel_assert(split is None or split_v1 or split_v5, f"split: the v1 or the v5 form, got {split}")
    kernel_assert(
        not q_prefetch_after or pass_q_prefetch,
        "q_prefetch_after orders the pass_q_prefetch loads: set pass_q_prefetch",
    )
    kernel_assert(
        not pass_tail_overlap
        or (
            not loop_path
            and sbuf is None
            and cp == _TP
            and not exact_softmax
            and not bf16_state
            and not packed_prior
            and not loop_p2
            and split is None
            and not state_bf16_o
            and not out_ar_call
            and not out_rs_tp
            and not late_tail_frames
            and sources is not None
            and "state_hbm_1" in sources
        ),
        "pass_tail_overlap is a form of the CP32 static path with the FP32 HBM state pairs (sources with state_hbm_1): "
        "not loop_path, an explicit sbuf, exact_softmax, bf16_state, packed_prior, loop_p2, split, state_bf16_o, "
        "out_ar_call, out_rs_tp or late_tail_frames",
    )
    kernel_assert(
        not seq_passes
        or (
            not loop_path
            and sbuf is None
            and cp == _TP
            and q_hbm
            and not bitwise
            and not exact_softmax
            and not packed_prior
            and not loop_p2
            and split is None
            and not query_rows
            and not out_ar_call
            and not out_rs_tp
            and not late_tail_frames
            and n_mask == 1
        ),
        "seq_passes is a form of the CP32 static draft path with the layer query gather in HBM (q_hbm): not loop_path, "
        "an explicit sbuf, bitwise, exact_softmax, packed_prior, loop_p2, split, query_rows, out_ar_call, out_rs_tp, "
        "late_tail_frames or fused masks",
    )
    kernel_assert(
        pass_q_prefetch in range(PASS_Q_PREFETCH_MAX_TILES + 1),
        f"pass_q_prefetch {pass_q_prefetch!r} is not a tile count from 0 to {PASS_Q_PREFETCH_MAX_TILES}. Each tile takes "
        "3,456 bytes a partition of the explicit MLA region for the call, and 3 tiles leave the second pass's packed "
        f"loop too little SBUF in the k7 verify. Use 0 to {PASS_Q_PREFETCH_MAX_TILES}.",
    )
    kernel_assert(
        q_gather_buffers in (1, 2), f"q_gather_buffers {q_gather_buffers!r} is not 1 or 2. Use 1 or 2 gather tiles."
    )
    kernel_assert(
        not (pass_rows_ahead and sb.heap),
        "pass_rows_ahead frees its heap tiles before the next pass's loop: it needs the stack frames of a region "
        "(an SbufTiles without heap) or compiler placement",
    )
    kernel_assert(
        not split_v5
        or (
            {"form", "dp_group", "dp", "foreign_slots"}
            <= set(split)
            <= {"form", "dp_group", "dp", "foreign_slots", "checks", "fold", "ext", "guards", "ag_priority"}
            and split["foreign_slots"] in (1, 2, 4)
        ),
        f"split (v5) is {{form, dp_group, dp, foreign_slots 1 or 2}}, got {split}",
    )
    # The split passes start with own-key weight 0 on the packed loop only; the loop_p2 path has no split pass.
    kernel_assert(not (loop_p2 and split is not None), "split needs the packed loop (packed_prior), not loop_p2")
    kernel_assert(
        not split_v1
        or (
            packed_prior
            and bitwise
            and cp == _TP
            and local_heads == 3
            and output_gate
            and write_cache
            and rope_rows is None
            and tokens == 4
            and q_rows == 8
            and set(split) == {"group", "info"}
        ),
        "split (v1) needs the bitwise packed loop on CP32 with 3 local heads and the output gate, no rope rows, the "
        f"extended batch of B' = 4 slots and Q = 8, and split = {{group, info}}; got B' = {tokens}, Q = {q_rows}",
    )
    q_fp8 = (packed_prior and pack_options["q_fp8"]) or rows_q_fp8
    q_mx = bool(packed_prior and pack_options["q_mx_gather"])
    # ``q_mx_scale_words``: the word records of the gather pair hold the scales too (``_q_mx_regions`` ``packed``).
    q_mx_packed = bool(q_mx and pack_options["q_mx_scale_words"])
    q_dtype = _q_call_layout(local_heads, "mx" if q_mx else q_fp8)[1]
    num_pages = kv_pool.shape[0]
    bf16 = nl.bfloat16
    fp32 = nl.float32
    hwdge = nisa.dge_mode.hwdge
    sync = nisa.engine.sync
    # The MLA DMA and GEMV options (all bitwise, default off): priority 0 on the small critical DMAs; the owner write
    # off the SP queue; and two PE column bands for the narrow x-stationary GEMVs.
    kernel_assert(
        critical_dma_priority in (False, True, "query"),
        f"critical_dma_priority must be False, True or 'query', got {critical_dma_priority!r}",
    )
    crit_dma = {"priority": dma_priority(0)} if critical_dma_priority else {}
    # "query": the query path and the call-level state store / load on the SP hardware DGE queue at priority 0.
    query_dma = {**crit_dma, "dge_mode": hwdge, "engine": sync} if critical_dma_priority == "query" else {}
    cache_write_engine = nisa.engine.scalar if cache_write_off_sp else sync
    kernel_assert(
        not own_rows_ahead or (bitwise and not (own_key_select or packed_prior or loop_p2)),
        "own_rows_ahead is a form of the bitwise own-key DMA path (not own_key_select, packed_prior or loop_p2)",
    )
    bands_ok = gemv_bands and total_rows <= _P_MAX // 2

    def gemv_band(index: int) -> dict:
        """Column-band options of a narrow x-stationary matmul, or {} for the full array (keys off / wide)."""
        if not bands_ok:
            return {}
        return {"tile_position": (0, (index % 2) * (_P_MAX // 2)), "tile_size": (_P_MAX, _P_MAX // 2)}

    # Reserve every static-path collective tile to the kernel end. On the standalone
    # region=None path the q, qkv and tail all-gather sources and destinations are columns of two SBUF arenas
    # (destinations separate from sources, so a collective's source and destination never share one tensor,
    # and a peer that writes a gather destination early does not corrupt a reused source). The state
    # reduce-scatter runs in persistent HBM, so it needs no compiler scratch in SBUF. The loop path
    # keeps its tiles in the explicit SBUF region; an integration caller passes the same tiles via sources=,
    # so the arenas are built only when neither applies. The HBM tensors are not cleared at the kernel start.
    q_gather_hbm = None
    if sources is None and not sb.explicit:
        n_q_tiles = 0
        if not lay.one_head and not q_hbm:
            n_q_tiles = sum(
                (min(_TAIL_ROWS, total_rows - cb) + _MAX_TOKENS - 1) // _MAX_TOKENS
                for cb in range(0, total_rows, _TAIL_ROWS)
            )
        plan = _ArenaPlan()
        qkv_i = plan.slot((total_rows, lay.qkv_local), "qkv_src")
        q_i = [plan.slot((_MAX_TOKENS, gather_width), f"q_src_{i}") for i in range(n_q_tiles)]
        if out_ar_call:
            ag_i = []
        elif out_ag_call:
            ag_i = [plan.slot((_P_MAX // cp, -(-total_rows // _TAIL_ROWS) * tr * _HIDDEN_TILES), "ag_call_src")]
        else:
            ag_i = [plan.slot((_P_MAX // cp, tr, _HIDDEN_TILES), f"ag_src_{i}") for i in range(2)]
        qkv_dst_i = plan.slot((total_rows, cp * lay.qkv_local), "qkv_dst")
        q_dst_i = [plan.slot((_P_MAX, gather_width), f"q_dst_{i}") for i in range(n_q_tiles)]
        arenas = tuple(
            nl.ndarray((_P_MAX, w), dtype=bf16, buffer=nl.sbuf, name=f"{tag}_mla_coll_arena_{k}")
            for k, w in enumerate(plan.widths)
        )
        for a in arenas:
            nisa.memset(dst=a, value=0.0)
        arena_qkv_row = plan.view(arenas, qkv_i)
        arena_q_tiles = [plan.view(arenas, i) for i in q_i]
        arena_ag_bufs = [plan.view(arenas, i) for i in ag_i]
        arena_qkv_all = plan.view(arenas, qkv_dst_i)
        arena_q_dsts = [plan.view(arenas, i) for i in q_dst_i]
        state_hbm_tiles = mla_block_collective_hbm(
            total_rows,
            local_heads,
            cp,
            heads,
            name=f"_{tag}" if lay.one_head else "",
            reduce_rows=reduce_rows,
            q_hbm=q_hbm,
            out_ag_call=out_ag_call,
            out_ar_call=out_ar_call,
            **({"q_fp8": q_fp8} if q_fp8 else {}),
        )
        arena_state_hbm = state_hbm_tiles["state_hbm"]
        arena_state_dst_hbm = state_hbm_tiles["state_dst_hbm"]
        arena_q_call = state_hbm_tiles if "q_call_src_hbm" in state_hbm_tiles else None
        arena_out_ag = state_hbm_tiles if "out_ag_src_hbm" in state_hbm_tiles else None
        arena_out_ar = state_hbm_tiles if "out_ar_src_hbm" in state_hbm_tiles else None
        q_gather_hbm = state_hbm_tiles
    if lay.one_head and q_gather_hbm is None:
        q_gather_hbm = sources if sources is not None and "q_src_hbm" in sources else None
    if lay.one_head and q_gather_hbm is None:
        q_gather_hbm = mla_block_collective_hbm(
            total_rows, local_heads, cp, heads, name=f"_{tag}", reduce_rows=reduce_rows
        )
    use_rope = rope_rows is not None

    # Step 1: build the query-row metadata. token_meta holds, per query row, its own position
    # (position[b] + j), its write slot, and the TP rank. bias_meta holds the base position of each
    # sequence (position[b], the prior-key mask threshold) and the TP rank on the bias partition.
    pos_row = sb.tile((1, tokens), nl.int32)
    nisa.dma_copy(
        dst=pos_row,
        src=position.ap(pattern=[[tokens, 1], [1, tokens]], offset=0),
        dge_mode=hwdge,
        engine=sync,
        **crit_dma,
        name=f"{tag}_mla_position_row",
    )
    j3 = sb.tile((1, q_rows, tokens), nl.int32)
    nisa.iota(dst=j3, pattern=[[1, q_rows], [0, tokens]], offset=0)
    own3 = sb.tile((1, q_rows, tokens), nl.int32)
    nisa.tensor_tensor(dst=own3, data1=j3, data2=pos_row.reshape((1, 1, tokens)).broadcast(1, q_rows), op=nl.add)
    own_row_f = sb.tile((1, total_rows), nl.float32)
    nisa.tensor_copy(dst=own_row_f, src=own3.reshape((1, total_rows)))
    own_psum = nl.ndarray((total_rows, 1), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=own_psum, data=own_row_f)
    # token_meta is padded to 8 int32 (two 16-byte ECC blocks per 32-byte row) so the engine-written own
    # position (element 0, block 0) shares no 16-byte block with the DMA-written slot and rank (elements 4
    # and 5, block 1). SBUF ECC is 16 bytes, so a sub-16-byte engine write is a read-modify-write of its
    # whole block; keeping the engine and DMA writes in different blocks means neither can lose the other's
    # bytes with no ordering edge. The 32-byte row is 16-byte aligned, so adjacent rows share no block.
    token_meta = sb.tile((total_rows, 8), nl.int32)
    nisa.tensor_copy(dst=token_meta[0:total_rows, 0:1], src=own_psum)
    nisa.dma_copy(
        dst=token_meta[0:total_rows, 4:5],
        src=slot[0:total_rows, 0:1],
        dge_mode=hwdge,
        engine=sync,
        **crit_dma,
        name=f"{tag}_mla_slot_load",
    )
    nisa.dma_copy(
        dst=token_meta[0:total_rows, 5:6],
        src=rank_info[0:1, 2:3].broadcast(0, total_rows),
        dge_mode=hwdge,
        engine=sync,
        **crit_dma,
        name=f"{tag}_mla_rank_load",
    )
    # With fused masks, the thresholds sit on each mask row (one partition for each query row).
    bias_end = _BIAS_ROW + n_mask
    bias_meta = sb.tile((_P_MAX, tokens + 1), nl.int32)
    nisa.dma_copy(
        dst=bias_meta[_BIAS_ROW:bias_end, 0:tokens],
        src=position.ap(pattern=[[0 if n_mask > 1 else tokens, n_mask], [1, tokens]], offset=0),
        dge_mode=hwdge,
        engine=sync,
        **crit_dma,
        name=f"{tag}_mla_base_position_row",
    )
    nisa.dma_copy(
        dst=bias_meta[_BIAS_ROW:bias_end, tokens : tokens + 1],
        src=rank_info[0:1, 2:3].broadcast(0, n_mask) if n_mask > 1 else rank_info[0:1, 2:3],
        dge_mode=hwdge,
        engine=sync,
        **crit_dma,
        name=f"{tag}_mla_rank_row_load",
    )

    # The projections run in one SBUF scope that frees before the attention. The attention then runs in
    # row chunks (each gathers, attends, and tails its own rows), so the explicit-SBUF region of the
    # loop path fits at B up to 8: the projection scratch and transient weights free before the chunks,
    # and each chunk's queries, key-group buffers, and state free before the next.
    sb.open_scope("mla_proj")

    # Step 3: load the head weights (see ``tkg_mla_paged``). The gate region loads only with output_gate.
    qkv_chunks = []
    mx_scales = None
    if mla_in_proj_scale is not None:
        qkv_chunks.append(_load_mx_qkv(mla_in_proj, mla_in_proj_scale, sb, tag))
        mx_scales = qkv_chunks[0][2]
    for chunk_start in range(0, _HIDDEN_TILES if mla_in_proj_scale is None else 0, _QKV_CHUNK_TILES):
        chunk_tiles = min(_QKV_CHUNK_TILES, _HIDDEN_TILES - chunk_start)
        qkv_w = sb.tile((_P_MAX, chunk_tiles, lay.qkv_local), bf16)
        gate_weight_dma(
            nisa.dma_copy(
                dst=qkv_w,
                src=mla_in_proj[0:_P_MAX, chunk_start * lay.qkv_local : (chunk_start + chunk_tiles) * lay.qkv_local],
                dge_mode=hwdge,
                engine=sync,
                name=f"{tag}_mla_w_qkv_{chunk_start}",
            )
        )
        qkv_chunks.append((chunk_start, chunk_tiles, qkv_w))
    q_a_norm_w = sb.tile((_P_MAX, _Q_LORA_TILES), bf16)
    gate_weight_dma(
        nisa.dma_copy(
            dst=q_a_norm_w,
            src=mla_q_a_norm.ap(pattern=[[_Q_LORA_TILES, _P_MAX], [1, _Q_LORA_TILES]], offset=0),
            dge_mode=hwdge,
            engine=sync,
            name=f"{tag}_mla_w_q_a_norm",
        )
    )
    kv_a_norm_w = sb.tile((total_rows, _KV_LORA), bf16)
    gate_weight_dma(
        nisa.dma_copy(
            dst=kv_a_norm_w,
            src=mla_kv_a_norm.ap(pattern=[[0, total_rows], [1, _KV_LORA]], offset=0),
            dge_mode=hwdge,
            engine=sync,
            name=f"{tag}_mla_w_kv_a_norm",
        )
    )
    # ``q_b_scales_scoped``: the q_b scale copies in the projection scope (Step 13 reads them there), not on the heap.
    q_b_scoped = bool(pack_options and pack_options["q_b_scales_scoped"])
    kernel_assert(
        not q_b_scoped or mla_q_b_proj_scale is not None, "q_b_scales_scoped needs the MXFP8 mla_q_b_proj_scale"
    )
    if mla_q_b_proj_scale is not None:
        q_b_w, q_b_scales = _load_mx_q_b(mla_q_b_proj, mla_q_b_proj_scale, sb, tag, keep=not q_b_scoped)
    else:
        q_b_w = sb.tile((_P_MAX, _Q_LORA_TILES, local_heads * _Q_HEAD), bf16)
        gate_weight_dma(
            nisa.dma_copy(dst=q_b_w, src=mla_q_b_proj, dge_mode=hwdge, engine=sync, name=f"{tag}_mla_w_q_b")
        )
        q_b_scales = None
    if mla_kv_b_proj_scale is not None:
        kv_b_w = _load_mx_kv_b(mla_kv_b_proj, mla_kv_b_proj_scale, sb, tag, keep=True)
    else:
        kv_b_w = sb.tile((_P_MAX, 2 * absorb_width), bf16, keep=True)
        gate_weight_dma(
            nisa.dma_copy(dst=kv_b_w, src=mla_kv_b_proj, dge_mode=hwdge, engine=sync, name=f"{tag}_mla_w_kv_b")
        )
    # ``o_proj_ahead``: the heap tiles of the tail's o_proj weight chunks, live through the loop; the pass loads them
    # (``load_o_proj_ahead``).
    o_ahead_chunks = None
    if o_proj_ahead and mla_o_proj_scale is not None:
        # The MXFP8 chunks (``MxOChunk``) and the scale tile they share, on the heap the same way.
        o_ahead_scale = sb.tile((_P_MAX, _HIDDEN), nl.uint8, keep=True)
        o_ahead_chunks = []
        for chunk_start in range(0, _HIDDEN_TILES, _O_PROJ_CHUNK_TILES):
            chunk_tiles = min(_O_PROJ_CHUNK_TILES, _HIDDEN_TILES - chunk_start)
            o_w = sb.tile((_P_MAX, chunk_tiles * _P_MAX * MX_LANES), nl.uint8, keep=True)
            o_ahead_chunks.append((chunk_start, chunk_tiles, MxOChunk(o_w, o_ahead_scale)))
    elif o_proj_ahead:
        o_ahead_chunks = []
        for chunk_start in range(0, _HIDDEN_TILES, _O_PROJ_CHUNK_TILES):
            chunk_tiles = min(_O_PROJ_CHUNK_TILES, _HIDDEN_TILES - chunk_start)
            o_w = sb.tile((_P_MAX, local_heads, chunk_tiles * _P_MAX), bf16, keep=True)
            o_ahead_chunks.append((chunk_start, chunk_tiles, o_w))

    # Step 4: the write row of each query row. owner = ((own position >> 7) & (cp - 1)) == t. The write row is
    # the slot on the owner rank with a valid slot, else row 0 of the null page.
    owner_block = sb.tile((total_rows, 1), nl.int32)
    nisa.tensor_scalar(
        dst=owner_block,
        data=token_meta[0:total_rows, 0:1],
        op0=nl.right_shift,
        operand0=7,
        op1=nl.bitwise_and,
        operand1=cp - 1,
    )
    # Read after the projection scope (loop metadata, attention), so it stays on a region's heap. With fused masks only
    # the loop metadata reads it after the scope, so it stays on the heap on the loop path.
    token_f = sb.tile((total_rows, 3), fp32, keep=n_mask == 1 or (loop_path and local_blocks > _MAX_UNROLLED_BLOCKS))
    nisa.tensor_copy(dst=token_f[0:total_rows, 0:1], src=owner_block)
    nisa.tensor_copy(dst=token_f[0:total_rows, 1:3], src=token_meta[0:total_rows, 4:6])
    owner = sb.tile((total_rows, 1), fp32)
    nisa.tensor_scalar(dst=owner, data=token_f[0:total_rows, 0:1], op0=nl.equal, operand0=token_f[0:total_rows, 2:3])
    write_ok = sb.tile((total_rows, 1), fp32)
    nisa.tensor_scalar(
        dst=write_ok,
        data=token_f[0:total_rows, 1:2],
        op0=nl.greater_equal,
        operand0=float(_BLOCK),
        op1=nl.multiply,
        operand1=owner,
    )
    write_mask = sb.tile((total_rows, 1), nl.int32)
    nisa.tensor_scalar(dst=write_mask, data=write_ok, op0=nl.multiply, operand0=-1.0)
    # Keep ``write_row`` on the heap: the bitwise read-after-write edge anchors the owner-write it indexes,
    # and that edge's topological order is sensitive to this tile's liveness.
    write_row = sb.tile((total_rows, 1), nl.int32, keep=True)
    nisa.tensor_tensor(dst=write_row, data1=token_meta[0:total_rows, 4:5], data2=write_mask, op=nl.bitwise_and)
    if batched_cache_write:
        # Row r of the null page for each query row r that writes no owned slot: one target per row.
        kernel_assert(total_rows <= _BLOCK, f"batched_cache_write writes at most {_BLOCK} rows, got {total_rows}")
        # The region's partition index column (``install_null_rows``), when the kernel built it: no iota a layer.
        region_rows = getattr(sb.region, "null_rows", None) if sb.explicit else None
        if region_rows is not None:
            null_row = region_rows[0:total_rows, 0:1]
        else:
            null_row = sb.tile((total_rows, 1), nl.int32)
            nisa.iota(dst=null_row, pattern=[[0, 1]], offset=0, channel_multiplier=1)
        null_off = sb.tile((total_rows, 1), fp32)
        nisa.tensor_scalar(dst=null_off, data=write_ok, op0=nl.multiply, operand0=-1.0, op1=nl.add, operand1=1.0)
        nisa.tensor_tensor(dst=null_off, data1=null_off, data2=null_row, op=nl.multiply)
        nisa.tensor_tensor(dst=write_row, data1=write_row, data2=null_off, op=nl.add)

    # Step 5: the prior-key mask threshold of each sequence on the bias partition (base position - 128 t), and on
    # each mask row with fused masks.
    def bias_rows(src):
        return src.ap(pattern=[[0 if n_mask > 1 else tokens, n_mask], [1, tokens]], offset=0)

    bias_f = sb.tile((_P_MAX, tokens + 1), fp32)
    nisa.tensor_copy(dst=bias_f[_BIAS_ROW:bias_end, 0 : tokens + 1], src=bias_meta[_BIAS_ROW:bias_end])
    rank_offset = sb.tile((_P_MAX, 1), fp32)
    nisa.tensor_scalar(
        dst=rank_offset[_BIAS_ROW:bias_end, 0:1],
        data=bias_f[_BIAS_ROW:bias_end, tokens : tokens + 1],
        op0=nl.multiply,
        operand0=-float(_BLOCK),
    )
    mask_threshold = sb.tile((_P_MAX, tokens), fp32, keep=n_mask == 1)
    nisa.tensor_scalar(
        dst=mask_threshold[_BIAS_ROW:bias_end, 0:tokens],
        data=bias_f[_BIAS_ROW:bias_end, 0:tokens],
        op0=nl.add,
        operand0=rank_offset[_BIAS_ROW:bias_end, 0:1],
    )
    if window_base is not None:
        # Context window: slot_offset numbers prior keys from block column 0 of the window-sliced block_table, so
        # the prior-key mask threshold (position[b] - 128 t) must be relative to the window base
        # first_block[b] * 4096. position stays absolute for RoPE, the owner rank and the write slot, so the
        # shift is applied here on the mask path only (row_threshold below inherits it).
        window_i = sb.tile((_P_MAX, tokens), nl.int32)
        nisa.dma_copy(
            dst=window_i[_BIAS_ROW:bias_end, 0:tokens],
            src=bias_rows(window_base),
            dge_mode=hwdge,
            engine=sync,
            name=f"{tag}_mla_window_base_row",
        )
        window_row = sb.tile((_P_MAX, tokens), fp32)
        nisa.tensor_copy(dst=window_row[_BIAS_ROW:bias_end, 0:tokens], src=window_i[_BIAS_ROW:bias_end, 0:tokens])
        nisa.tensor_tensor(
            dst=mask_threshold[_BIAS_ROW:bias_end, 0:tokens],
            data1=mask_threshold[_BIAS_ROW:bias_end, 0:tokens],
            data2=window_row[_BIAS_ROW:bias_end, 0:tokens],
            op=nl.subtract,
        )
    # The bitwise verify masks the prior keys of row r = j * B + b at position[b] + j, so each row sees the
    # current rows 0 .. j-1 (written to the pool) as prior keys. row_threshold[r] = mask_threshold[b] + j. With
    # fused masks, mask row j holds the thresholds of query row j: row_threshold[j, b] = mask_threshold[b] + j.
    row_threshold = None
    if n_mask > 1:
        row_index = sb.tile((_P_MAX, 1), fp32)
        nisa.iota(dst=row_index[_BIAS_ROW:bias_end, 0:1], pattern=[[0, 1]], offset=0, channel_multiplier=1)
        # The attention reads the fused thresholds after the projection scope, so they stay on the heap.
        row_threshold = sb.tile((_P_MAX, tokens), fp32, keep=True)
        nisa.tensor_scalar(
            dst=row_threshold[_BIAS_ROW:bias_end, 0:tokens],
            data=mask_threshold[_BIAS_ROW:bias_end, 0:tokens],
            op0=nl.add,
            operand0=row_index[_BIAS_ROW:bias_end, 0:1],
        )
    elif bitwise:
        row_threshold = sb.tile((_P_MAX, total_rows), fp32, keep=True)
        for r in range(total_rows):
            nisa.tensor_scalar(
                dst=row_threshold[_BIAS_ROW : _BIAS_ROW + 1, r : r + 1],
                data=mask_threshold[_BIAS_ROW : _BIAS_ROW + 1, r % tokens : r % tokens + 1],
                op0=nl.add,
                operand0=float(r // tokens),
            )
    # The optional lower bound of each sequence (key_start[b] - 128 t). A prior key below it is masked.
    lower_threshold = None
    if key_start is not None:
        kernel_assert(tuple(key_start.shape) == (tokens, 1), f"key_start must be [B, 1], got {tuple(key_start.shape)}")
        key_i = sb.tile((_P_MAX, tokens), nl.int32)
        nisa.dma_copy(
            dst=key_i[_BIAS_ROW:bias_end, 0:tokens],
            src=bias_rows(key_start),
            dge_mode=hwdge,
            engine=sync,
            **crit_dma,
            name=f"{tag}_mla_key_start",
        )
        key_f = sb.tile((_P_MAX, tokens), fp32)
        nisa.tensor_copy(dst=key_f[_BIAS_ROW:bias_end, 0:tokens], src=key_i[_BIAS_ROW:bias_end, 0:tokens])
        lower_threshold = sb.tile((_P_MAX, tokens), fp32, keep=True)
        nisa.tensor_scalar(
            dst=lower_threshold[_BIAS_ROW:bias_end, 0:tokens],
            data=key_f[_BIAS_ROW:bias_end, 0:tokens],
            op0=nl.add,
            operand0=rank_offset[_BIAS_ROW:bias_end, 0:1],
        )

    # Step 6: the q_a and kv_a projections, x stationary and the weight columns moving.
    x_mx = None
    mx_rows = total_rows + total_rows % 2
    if mla_in_proj_scale is not None:
        x_mx = quantize_hT_mx(x_hT, f"{tag}_qkv", mx_rows, lambda shape, dtype, name: sb.tile(shape, dtype, name))
    qkv_psum = nl.ndarray((mx_rows if x_mx else total_rows, lay.qkv_local), dtype=fp32, buffer=nl.psum)
    if x_mx:
        _qkv_projection_mx(qkv_psum, x_mx, qkv_chunks[0], mx_rows)
    for chunk_start, chunk_tiles, qkv_w in qkv_chunks if x_mx is None else ():
        for chunk_tile in range(chunk_tiles):
            hidden_tile = chunk_start + chunk_tile
            nisa.nc_matmul(
                dst=qkv_psum[0:total_rows, 0 : lay.qkv_local],
                stationary=x_hT[0:_P_MAX, 0:total_rows, hidden_tile],
                moving=qkv_w[0:_P_MAX, chunk_tile, 0 : lay.qkv_local],
                accumulate=hidden_tile > 0,
            )
    # Hold-free qkv all-gather source: the arena column (standalone) or a resident tile, so no later tile
    # takes its bytes while a CP peer still reads them (the scan flags a reused all-gather source).
    # The split v5 fold: the qkv gather pair holds the F x Q foreign rows after the call's rows (B + F row slots), so the
    # main gather moves them and the foreign pass needs no gather of its own.
    fold_rows = split["foreign_slots"] * q_rows if split_v5 and split.get("fold") else 0
    kernel_assert(
        not fold_rows or (sources is not None and sources["qkv"].shape[0] == total_rows + fold_rows),
        f"split (v5) fold needs the qkv sources of {total_rows} + {fold_rows} rows (qkv_extra_rows)",
    )
    qkv_full = None
    if sources is not None and "qkv" in sources:
        qkv_full = sources["qkv"]
        qkv_row = qkv_full[0:total_rows] if fold_rows else qkv_full
    elif arena_qkv_row is not None:
        qkv_row = arena_qkv_row
    else:
        qkv_row = sb.tile((total_rows, lay.qkv_local), bf16, keep=True)
    nisa.tensor_copy(dst=qkv_row, src=qkv_psum[0:total_rows, 0 : lay.qkv_local])
    if split_v1:
        # The long-sequence split: the helper's foreign rows take the owner's q_a | kv_a before the TP gather.
        _split_import(qkv_row, split, tokens, q_rows, lay.qkv_local, tag, sb)
    if split_v5:
        if split_ext:
            _split_v5_import_ext(qkv_row, split, q_rows, tokens, lay.qkv_local, tag, sb)
        else:
            imp_dst = _split_v5_import(qkv_row, split, total_rows, lay.qkv_local, tag)
        if not split_ext and fold_rows:
            _split_v5_select(split, imp_dst, qkv_full, total_rows, q_rows, tokens, lay.qkv_local, tag, sb)

    # Step 7: gather q_a | kv_a over the CP group along the free axis. The destination is a distinct persistent tile.
    # A CP peer can write it before this rank reaches the gather, so no earlier tile may share its bytes.
    qkv_all_full = None
    if sources is not None and "qkv_dst" in sources:
        qkv_all_full = sources["qkv_dst"]
        qkv_all = qkv_all_full[0:total_rows] if fold_rows else qkv_all_full
    elif arena_qkv_all is not None:
        qkv_all = arena_qkv_all
    else:
        qkv_all = sb.tile((total_rows, cp * lay.qkv_local), bf16)
    ncc.all_gather(
        srcs=[qkv_full if fold_rows else qkv_row],
        dsts=[qkv_all_full if fold_rows else qkv_all],
        replica_group=tp_group,
        collective_dim=1,
        name=f"{tag}_mla_qkv_all_gather",
    )
    qkv_ranks = qkv_all.reshape_dim(1, (cp, lay.qkv_local))

    # ``q_first`` (the call query gather, no rope rows): the q path, the q-call store and its gather trigger first,
    # then the KV path, so the c0 entry's wait for the gather starts about the KV path's time sooner. The same ops on
    # the same operands, so the output is bitwise the same. Without it (the drafter's rope rows feed the q path) the
    # KV path runs first.
    q_first = q_hbm and not lay.one_head and not use_rope
    # ``q_split`` (with ``q_first``): the passes of the sequence-major fit form with one state reduce-scatter a call, so
    # that each pass gathers only its own query rows; the c0 pass then waits for half the bytes. The same bytes in the
    # same rows of the gathered queries, so the output is bitwise the same. The pass rows below must equal these.
    early_loop = loop_path and local_blocks > _MAX_UNROLLED_BLOCKS
    early_state = sources["state_hbm"] if sources is not None and "state_hbm" in sources else arena_state_hbm
    early_passes = -(-total_rows // _TAIL_ROWS)
    early_state_call = (
        fit_passes
        and bitwise
        and early_loop
        and total_rows > _TAIL_ROWS
        and sources is not None
        and sources.get("rs_call") is not None
        and early_state is not None
        and early_state.shape[0] == heads * early_passes > heads
    )
    early_seq_shape = (
        bitwise and not exact_softmax and tr % q_rows == 0 and tokens % (tr // q_rows) == 0 and tr % tokens == 0
    )
    q_split = None
    if q_first and early_state_call and early_seq_shape and not _Q_SPLIT_TEST_OFF:
        q_split = [
            [j * tokens + p * (tr // q_rows) + bb for j in range(q_rows) for bb in range(tr // q_rows)]
            for p in range(early_passes)
        ]
    # Step 9: q_a in the feature-major column layout of the q_b contraction.
    padded = total_rows + total_rows % 2
    q_lora_local = lay.q_lora_local
    q_a_rows = sb.tile((total_rows, cp, q_lora_local), bf16)
    half = cp // 2
    nisa.tensor_copy(
        dst=q_a_rows[0:total_rows, 0:half, 0:q_lora_local],
        src=qkv_ranks[0:total_rows, 0:half, 0:q_lora_local],
        engine=nisa.engine.vector,
    )
    nisa.tensor_copy(
        dst=q_a_rows[0:total_rows, half:cp, 0:q_lora_local],
        src=qkv_ranks[0:total_rows, half:cp, 0:q_lora_local],
        engine=nisa.engine.scalar,
    )
    q_a_flat = q_a_rows.reshape((total_rows, _Q_LORA))
    q_a_psum = nl.ndarray((_P_MAX, _Q_LORA_TILES, padded), dtype=bf16, buffer=nl.psum)
    for q_lora_tile in range(_Q_LORA_TILES):
        nisa.nc_transpose(
            dst=q_a_psum[0:_P_MAX, q_lora_tile, 0:total_rows],
            data=q_a_flat[0:total_rows, q_lora_tile * _P_MAX : (q_lora_tile + 1) * _P_MAX],
        )
    q_a = sb.tile((_P_MAX, total_rows, _Q_LORA_TILES), fp32)
    # ``q_pass_rows`` (where the call takes q_split): column p * R + j * nb + bb of q_a (R rows a pass, nb sequences
    # a pass) takes row j * B + p * nb + bb, the q_split order, so every row-indexed tile of the q path holds the rows
    # of a pass on adjacent partitions and the q-call store of a pass is one DMA. The same per-row arithmetic: the
    # same bits. Not with ``q_mx_gather``: its query store (``_q_mx_store``) and own-key scores take the rows of q_local
    # and of the current keys in the call's row order.
    q_pass_rows = packed_prior and pack_options["q_pass_rows"] and q_split is not None and not q_mx
    # ``q_prep_act``: the natural MX copies of a head on the engine opposite to its latent copy, the rope copy on
    # ScalarE (the query prep of each gathered tile; exact BF16 copies, the same bits).
    q_prep_act = packed_prior and pack_options["q_prep_act"]
    if q_pass_rows:
        nb_pass = tr // q_rows
        nisa.tensor_copy(
            dst=q_a.ap(
                pattern=[
                    [total_rows * _Q_LORA_TILES, _P_MAX],
                    [len(q_split[0]) * _Q_LORA_TILES, len(q_split)],
                    [nb_pass * _Q_LORA_TILES, q_rows],
                    [_Q_LORA_TILES, nb_pass],
                    [1, _Q_LORA_TILES],
                ],
                offset=0,
            ),
            src=q_a_psum.ap(
                pattern=[
                    [_Q_LORA_TILES * padded, _P_MAX],
                    [nb_pass, len(q_split)],
                    [tokens, q_rows],
                    [1, nb_pass],
                    [padded, _Q_LORA_TILES],
                ],
                offset=0,
            ),
        )
    else:
        nisa.tensor_copy(dst=q_a, src=q_a_psum[0:_P_MAX, 0:_Q_LORA_TILES, 0:total_rows].permute((0, 2, 1)))

    # Step 10: q_latent = rms(q_a).
    ones = sb.tile((_P_MAX, _P_MAX), fp32)
    nisa.memset(dst=ones, value=1.0)
    q_a_square = sb.tile((_P_MAX, total_rows, _Q_LORA_TILES), fp32)
    nisa.tensor_tensor(dst=q_a_square, data1=q_a, data2=q_a, op=nl.multiply)
    q_a_partial = sb.tile((_P_MAX, total_rows), fp32)
    nisa.tensor_reduce(dst=q_a_partial, op=nl.add, data=q_a_square, axis=2)
    q_a_total = nl.ndarray((_P_MAX, total_rows), dtype=fp32, buffer=nl.psum)
    nisa.nc_matmul(dst=q_a_total, stationary=ones, moving=q_a_partial, is_stationary_onezero=True)
    q_a_rstd = sb.tile((_P_MAX, total_rows), fp32)
    nisa.tensor_scalar(
        dst=q_a_rstd,
        data=q_a_total,
        op0=nl.multiply,
        operand0=1.0 / float(_Q_LORA),
        op1=nl.add,
        operand1=lora_norm_eps,
    )
    nisa.tensor_scalar(dst=q_a_rstd, data=q_a_rstd, op0=nl.rsqrt, operand0=0.0, engine=nisa.engine.gpsimd)
    q_a_normed = sb.tile((_P_MAX, total_rows, _Q_LORA_TILES), bf16)
    nisa.tensor_tensor(
        dst=q_a_normed, data1=q_a, data2=q_a_rstd.expand_dim(2).broadcast(2, _Q_LORA_TILES), op=nl.multiply
    )
    q_latent = sb.tile((_P_MAX, total_rows, _Q_LORA_TILES), bf16)
    nisa.tensor_tensor(
        dst=q_latent, data1=q_a_normed, data2=q_a_norm_w.expand_dim(1).broadcast(1, total_rows), op=nl.multiply
    )

    def _kv_part(after=None):
        """Steps 11-12: the current cache rows, their owner writes and the feature-major current keys. Return
        ``(cur_rows, write_rows, cache_writes, cur_keys, cos_x, sin_x)``. ``after`` (the q-call gather): the KV path
        issues after it."""
        cos_x = sin_x = None
        # Step 11: the current cache rows [rms(latent) | k_pe]. kv_a column c of rank r is feature cp c + r.
        kv_end = q_lora_local + lay.kv_latent_local
        kv_latent = qkv_ranks[0:total_rows, 0:cp, q_lora_local:kv_end]
        kv_square = sb.tile((total_rows, cp, lay.kv_latent_local), fp32)
        kv_partial = sb.tile((total_rows, 1), fp32)
        kv_root = nisa.scalar_tensor_tensor(
            dst=kv_square,
            data=kv_latent,
            op0=nl.multiply,
            operand0=1.0,
            op1=nl.multiply,
            operand1=kv_latent,
            reduce_op=nl.add,
            reduce_res=kv_partial,
        )
        kv_rstd = sb.tile((total_rows, 1), fp32)
        nisa.tensor_scalar(
            dst=kv_rstd,
            data=kv_partial,
            op0=nl.multiply,
            operand0=1.0 / float(_KV_LORA),
            op1=nl.add,
            operand1=lora_norm_eps,
        )
        nisa.tensor_scalar(dst=kv_rstd, data=kv_rstd, op0=nl.rsqrt, operand0=0.0, engine=nisa.engine.gpsimd)
        kv_normed = sb.tile((total_rows, lay.kv_latent_local, cp), bf16)
        nisa.tensor_scalar(dst=kv_normed, data=kv_latent.permute((0, 2, 1)), op0=nl.multiply, operand0=kv_rstd)
        cur_rows = sb.tile((total_rows, _CACHE_WIDTH), bf16, keep=True)
        nisa.tensor_tensor(
            dst=cur_rows[0:total_rows, 0:_KV_LORA],
            data1=kv_normed.reshape((total_rows, _KV_LORA)),
            data2=kv_a_norm_w,
            op=nl.multiply,
        )
        kpe_root = nisa.tensor_copy(
            dst=cur_rows[0:total_rows, _KV_LORA:_CACHE_WIDTH].reshape_dim(1, (lay.kpe_local, lay.kpe_ranks)),
            src=qkv_ranks[0:total_rows, 0 : lay.kpe_ranks, kv_end : lay.qkv_local].permute((0, 2, 1)),
        )
        if after is not None:
            # The two roots of the KV path (every later KV op depends on them) issue after the q-call gather trigger, so
            # the KV path does not take its engines ahead of the q path (scheduling-only dependencies).
            for root in (kv_root, kpe_root):
                _instruction(root).depends_on(after, kind=nisa.dependency.schedule)
        # RoPE the current k_pe of each query row (the drafter). The cache stores the rotated k_pe.
        if use_rope:
            rope_sbuf = sb.tile((total_rows, 2 * _ROPE), fp32)
            nisa.dma_copy(
                dst=rope_sbuf,
                src=rope_rows[0:total_rows, 0 : 2 * _ROPE],
                dge_mode=hwdge,
                engine=sync,
                name=f"{tag}_mla_rope_rows",
            )
            cos_x = rope_sbuf[0:total_rows, 0:_ROPE]
            sin_x = rope_sbuf[0:total_rows, _ROPE : 2 * _ROPE]
            _apply_rope(cur_rows[0:total_rows, _KV_LORA:_CACHE_WIDTH], cos_x, sin_x, total_rows, 1, sb)

        # The row is clamped (FP8 to +-448, rounded to nearest even; BF16 to +-_BF16_KEY_MAX): the current block
        # mixes the rows of all sequences, and its mask does not stop a NaN.
        write_rows = cur_rows
        if kv_pool.dtype != bf16:
            write_rows = sb.tile((total_rows, _CACHE_WIDTH), kv_pool.dtype, keep=True)
            nisa.tensor_scalar(
                dst=write_rows,
                data=cur_rows,
                op0=nl.minimum,
                operand0=fp8_page_max(),
                op1=nl.maximum,
                operand1=-fp8_page_max(),
            )
            cur_rows = sb.tile((total_rows, _CACHE_WIDTH), bf16, keep=True)
            nisa.tensor_copy(dst=cur_rows, src=write_rows)
        else:
            nisa.tensor_scalar(
                dst=cur_rows,
                data=cur_rows,
                op0=nl.minimum,
                operand0=_BF16_KEY_MAX,
                op1=nl.maximum,
                operand1=-_BF16_KEY_MAX,
            )

        # Owner-write the current rows at the write rows, with no branch. A caller that overwrites the rows
        # before any read (the drafter block) skips the write.
        pool_rows = kv_pool.reshape((num_pages * _BLOCK, _CACHE_WIDTH))
        cache_writes = []
        if write_cache and batched_cache_write:
            kernel_assert(pool_rows.shape[0] < 1 << 24, "batched_cache_write needs fewer than 2**24 pool rows")
            cache_writes.append(
                nisa.dma_copy(
                    dst=pool_rows.ap(
                        pattern=[[_CACHE_WIDTH, total_rows], [1, _CACHE_WIDTH]],
                        offset=0,
                        vector_offset=write_row[0:total_rows, 0:1],
                        indirect_dim=0,
                    ),
                    src=write_rows[0:total_rows, 0:_CACHE_WIDTH],
                    dge_mode=nisa.dge_mode.swdge,
                    name=f"{tag}_mla_cache_write",
                )
            )
        elif write_cache:
            for row in range(total_rows):
                if split_ext and row % tokens >= split["ext"]:
                    # A foreign row of the split program writes no cache row (its slot is the null page's): no DMA.
                    continue
                cache_write = nisa.dma_copy(
                    dst=pool_rows.ap(
                        pattern=[[_CACHE_WIDTH, 1], [1, _CACHE_WIDTH]],
                        offset=0,
                        scalar_offset=write_row[row : row + 1, 0:1],
                        indirect_dim=0,
                    ),
                    src=write_rows[row : row + 1, 0:_CACHE_WIDTH],
                    dge_mode=hwdge,
                    engine=cache_write_engine,
                    **crit_dma,
                    name=f"{tag}_mla_cache_write_{row}",
                )
                cache_writes.append(cache_write)

        # Step 12: transpose the current keys into feature-major columns [128, tile, Q*B] and [64, Q*B].
        cur_psum = nl.ndarray((_P_MAX, _KV_LORA_TILES + 1, padded), dtype=bf16, buffer=nl.psum)
        for latent_tile in range(_KV_LORA_TILES):
            nisa.nc_transpose(
                dst=cur_psum[0:_P_MAX, latent_tile, 0:total_rows],
                data=cur_rows[0:total_rows, latent_tile * _P_MAX : (latent_tile + 1) * _P_MAX],
            )
        nisa.nc_transpose(
            dst=cur_psum[0:_ROPE, _KV_LORA_TILES, 0:total_rows], data=cur_rows[0:total_rows, _KV_LORA:_CACHE_WIDTH]
        )
        cur_keys = sb.tile((_P_MAX, _KV_LORA_TILES + 1, total_rows), bf16, keep=True)
        nisa.tensor_copy(
            dst=cur_keys[0:_P_MAX, 0:_KV_LORA_TILES, 0:total_rows],
            src=cur_psum[0:_P_MAX, 0:_KV_LORA_TILES, 0:total_rows],
        )
        nisa.tensor_copy(
            dst=cur_keys[0:_ROPE, _KV_LORA_TILES, 0:total_rows], src=cur_psum[0:_ROPE, _KV_LORA_TILES, 0:total_rows]
        )
        # The owner writes of the pool: a later pool read at a runtime offset takes a compiler RAW edge on them.
        record_pool_writes(cache_writes)
        return cur_rows, write_rows, cache_writes, cur_keys, cos_x, sin_x

    def _q_call_gather(split=None):
        """The q-call store and its all-gather (``q_hbm``). Return ``(q_call, the first gather, q_slot)``:
        ``q_slot(row)`` is ``(slot, block rows)``: the 32 rank rows of query row ``row`` sit at rows ``slot + g x block
        rows`` (rank ``g``) of the gathered pair (``q_pair_src`` and ``_q_latent_load`` take it). ``(None, None,
        None)`` without the call gather. ``split`` (``q_split``: the rows of each pass): the store writes the rows pass by
        pass, and each pass gets its own all-gather of its rows, so a pass waits only for the gather of its own rows."""
        q_ag = None
        q_call = None
        q_slot = None
        if q_hbm and not lay.one_head:
            q_pool = sources if sources is not None and "q_call_src_hbm" in sources else arena_q_call
            kernel_assert(q_pool is not None, "q_hbm needs the q_call HBM pair in sources (or the standalone arena)")
            q_call = (q_pool["q_call_src_hbm"], q_pool["q_call_dst_hbm"])
            kernel_assert(
                all(t.dtype == q_dtype for t in q_call),
                f"the q_call HBM pair must be {q_dtype} (packed_prior_options q_fp8, rows_q_fp8)",
            )
            if q_mx:
                # The MX query bytes of each pass's block (``_q_mx_store``), then the block's all-gather.
                blocks = split if split is not None else [list(range(total_rows))]
                pair_width = q_call[0].shape[1]
                _q_mx_store(
                    q_local,
                    cur_keys,
                    q_call[0],
                    blocks,
                    q_rows,
                    tokens,
                    local_heads,
                    f"{tag}_mla",
                    sb,
                    query_dma,
                    packed=q_mx_packed,
                )
                n = len(blocks[0])
                gathers = [
                    ncc.all_gather(
                        srcs=[q_call[0][p * n : (p + 1) * n, 0:pair_width]],
                        dsts=[q_call[1][p * _TP * n : (p + 1) * _TP * n, 0:pair_width]],
                        replica_group=tp_group,
                        collective_dim=0,
                        name=f"{tag}_mla_q_call_all_gather" + (f"_p{p}" if split is not None else ""),
                    )
                    for p in range(len(blocks))
                ]
                where = {row: (p, i) for p, rows_p in enumerate(blocks) for i, row in enumerate(rows_p)}

                def q_slot(row):
                    p, i = where[row]
                    return p * _TP * n + i, n

                return q_call, gathers[0], q_slot
            # One gather of all rows (rank-major: rank g's rows at g * Q*B); each tile loads the 32 rank rows of a token.
            q_store = q_local[0:total_rows, 0:gather_width]
            if q_fp8 == "latent":
                q_store = _q_latent_store(q_local, total_rows, local_heads, sb)
            elif q_fp8:
                # The e4m3 rounding of dr_keys / rows_dr (_dr_gather_tile_queries, _e4m3_query_rounding), before the gather.
                q_store = sb.tile((total_rows, gather_width), nl.float8_e4m3fn)
                nisa.tensor_scalar(
                    dst=q_store,
                    data=q_local[0:total_rows, 0:gather_width],
                    op0=nl.minimum,
                    operand0=_E4M3_MAX,
                    op1=nl.maximum,
                    operand1=-_E4M3_MAX,
                    engine=nisa.engine.vector,
                )
            if split is None:
                nisa.dma_copy(dst=q_call[0], src=q_store, name=f"{tag}_mla_q_call_store", **query_dma)
                q_ag = ncc.all_gather(
                    srcs=[q_call[0]],
                    dsts=[q_call[1]],
                    replica_group=tp_group,
                    collective_dim=0,
                    name=f"{tag}_mla_q_call_all_gather",
                )

                def q_slot(row):
                    return row, total_rows

            else:
                # Row j * B + p * nb + bb (query row j of sequence p * nb + bb) goes to row j * nb + bb of pass p's
                # block. The nb rows of query row j of a pass sit on adjacent partitions, so pass p stores Q DMAs of nb
                # rows; the gather of pass p follows the stores of its rows.
                n_split, split_rows = len(split), len(split[0])
                nb_split = split_rows // q_rows
                pair_width = q_call[0].shape[1]
                kernel_assert(
                    cp == _TP and n_split * split_rows == total_rows and n_split * nb_split == tokens,
                    "the split query gather needs TP32 heads and passes that hold every row",
                )
                gathers = []
                for p in range(n_split):
                    if q_pass_rows:
                        # The q path holds the rows in pass order (``q_pass_rows``): one DMA of the pass's rows.
                        nisa.dma_copy(
                            dst=q_call[0][p * split_rows : (p + 1) * split_rows, 0:pair_width],
                            src=q_store[p * split_rows : (p + 1) * split_rows, 0:pair_width],
                            name=f"{tag}_mla_q_call_store_p{p}",
                            **query_dma,
                        )
                    for j in range(0 if q_pass_rows else q_rows):
                        first = j * tokens + p * nb_split
                        nisa.dma_copy(
                            dst=q_call[0][
                                p * split_rows + j * nb_split : p * split_rows + (j + 1) * nb_split, 0:pair_width
                            ],
                            src=q_store[first : first + nb_split, 0:pair_width],
                            name=f"{tag}_mla_q_call_store_p{p}_{j}",
                            **query_dma,
                        )
                    gathers.append(
                        ncc.all_gather(
                            srcs=[q_call[0][p * split_rows : (p + 1) * split_rows, 0:pair_width]],
                            dsts=[q_call[1][p * _TP * split_rows : (p + 1) * _TP * split_rows, 0:pair_width]],
                            replica_group=tp_group,
                            collective_dim=0,
                            name=f"{tag}_mla_q_call_all_gather_p{p}",
                        )
                    )
                q_ag = gathers[0]
                where = {row: (p, i) for p, rows_p in enumerate(split) for i, row in enumerate(rows_p)}

                def q_slot(row):
                    p, i = where[row]
                    return p * _TP * split_rows + i, split_rows

        return q_call, q_ag, q_slot

    kernel_assert(
        not q_mx
        or (
            q_first
            and bitwise
            and not exact_softmax
            and not pass_rows_ahead
            and pass_q_prefetch == 0
            and q_gather_buffers == 1
            and (q_split is not None or total_rows <= _TAIL_ROWS)
        ),
        "packed_prior_options q_mx_gather needs the call query gather with the 3-head layout and no rope rows, the "
        "bitwise capped softmax, one pass or the split query gather (one gather a pass), and no BF16 gather-tile lever: "
        f"pass_q_prefetch 0 (got {pass_q_prefetch}), q_gather_buffers 1 (got {q_gather_buffers}), no pass_rows_ahead "
        f"(got {pass_rows_ahead})",
    )
    q_call = q_ag = q_slot = None

    def q_pair_src(row):
        """The source of the 32 rank rows of query row ``row`` in the BF16 (or e4m3) gather pair (``q_slot``)."""
        slot, block = q_slot(row)
        return q_call[1].ap(pattern=[[block * gather_width, _TP], [1, gather_width]], offset=slot * gather_width)

    if not q_first:
        cur_rows, write_rows, cache_writes, cur_keys, cos_x, sin_x = _kv_part()

    # Step 13: q = q_latent @ q_b, one row per query row.
    if mla_q_b_proj_scale is not None:
        # MXFP8 q_b: q_latent stationary (quantized per row), the q_b weight moving (the Step 13 orientation). The
        # psum takes an even partition count (the stationary free size); the body reads only the live rows.
        padded = total_rows + total_rows % 2
        q_nope_psum = nl.ndarray((padded, local_heads * _NOPE), dtype=fp32, buffer=nl.psum)
        q_rope_psum = nl.ndarray((padded, local_heads * _ROPE), dtype=fp32, buffer=nl.psum)
        _q_b_projection_mx(q_latent, q_b_w, q_b_scales, q_nope_psum, q_rope_psum, total_rows, local_heads, tag, sb)
        q_nope_psum = q_nope_psum[0:total_rows]
        q_rope_psum = q_rope_psum[0:total_rows]
    else:
        q_nope_psum = nl.ndarray((total_rows, local_heads * _NOPE), dtype=fp32, buffer=nl.psum)
        q_rope_psum = nl.ndarray((total_rows, local_heads * _ROPE), dtype=fp32, buffer=nl.psum)
        q_b_heads = q_b_w.reshape_dim(2, (local_heads, _Q_HEAD))
        for q_lora_tile in range(_Q_LORA_TILES):
            nisa.nc_matmul(
                dst=q_nope_psum.reshape_dim(1, (local_heads, _NOPE)),
                stationary=q_latent[0:_P_MAX, 0:total_rows, q_lora_tile],
                moving=q_b_heads[0:_P_MAX, q_lora_tile, 0:local_heads, 0:_NOPE],
                accumulate=q_lora_tile > 0,
                **gemv_band(0),
            )
            nisa.nc_matmul(
                dst=q_rope_psum.reshape_dim(1, (local_heads, _ROPE)),
                stationary=q_latent[0:_P_MAX, 0:total_rows, q_lora_tile],
                moving=q_b_heads[0:_P_MAX, q_lora_tile, 0:local_heads, _NOPE:_Q_HEAD],
                accumulate=q_lora_tile > 0,
                **gemv_band(1),
            )
    q_local = sb.tile((total_rows, gather_width), bf16, keep=True)
    q_rope_local = sb.tile((total_rows, local_heads, _ROPE), bf16)
    nisa.tensor_copy(dst=q_rope_local, src=q_rope_psum.reshape_dim(1, (local_heads, _ROPE)))
    # RoPE the q_pe of each query row (the drafter).
    if use_rope:
        _apply_rope(q_rope_local, cos_x, sin_x, total_rows, local_heads, sb)
    nisa.tensor_copy(
        dst=q_local[0:total_rows, absorb_width:gather_width],
        src=q_rope_local.reshape((total_rows, local_heads * _ROPE)),
    )
    if isinstance(kv_b_w, MxKvB):
        # Step 14 MXFP8: the q_nope rows quantized in quadrant row tiles (head h at partitions 32 h), the 3 heads as
        # concurrent MXFP8 row tiles against the quadrant key weight (the weight moving, as the BF16 form). The
        # quantize reads the q_nope PSUM itself (one lane-major read-out, ``quantize_quadrant_rows_mx``).
        qn_data, qn_scale = _absorb_mx_operand(q_nope_psum, total_rows, padded, f"{tag}_qn", sb)
        q_abs_psums = [nl.ndarray((padded, _KV_LORA), dtype=fp32, buffer=nl.psum) for _ in range(local_heads)]
        quadrant_absorb_mx(q_abs_psums, qn_data, qn_scale, kv_b_w.absorb, kv_b_w.absorb_scale, _KV_LORA, 0, padded)
        for local_head in range(local_heads):
            nisa.tensor_copy(
                dst=q_local[0:total_rows, local_head * _KV_LORA : (local_head + 1) * _KV_LORA],
                src=q_abs_psums[local_head][0:total_rows],
                engine=nisa.engine.vector if local_head % 2 else nisa.engine.scalar,
            )
    else:
        q_nope_row = sb.tile((total_rows, local_heads * _NOPE), bf16)
        nisa.tensor_copy(dst=q_nope_row, src=q_nope_psum)
        q_nope_t_psum = nl.ndarray((_P_MAX, local_heads, padded), dtype=bf16, buffer=nl.psum)
        for local_head in range(local_heads):
            nisa.nc_transpose(
                dst=q_nope_t_psum[0:_P_MAX, local_head, 0:total_rows],
                data=q_nope_row[0:total_rows, local_head * _NOPE : (local_head + 1) * _NOPE],
            )
        q_nope = sb.tile((_P_MAX, local_heads, total_rows), bf16)
        nisa.tensor_copy(dst=q_nope, src=q_nope_t_psum[0:_P_MAX, 0:local_heads, 0:total_rows])

        # Step 14: absorb the key projection, q_abs[h] = q_nope[h] @ key[h].
        for local_head in range(local_heads):
            q_abs_psum = nl.ndarray((total_rows, _KV_LORA), dtype=fp32, buffer=nl.psum)
            nisa.nc_matmul(
                dst=q_abs_psum,
                stationary=q_nope[0:_P_MAX, local_head, 0:total_rows],
                moving=kv_b_w[0:_P_MAX, local_head * _KV_LORA : (local_head + 1) * _KV_LORA],
                **gemv_band(local_head),
            )
            nisa.tensor_copy(
                dst=q_local[0:total_rows, local_head * _KV_LORA : (local_head + 1) * _KV_LORA],
                src=q_abs_psum,
                engine=nisa.engine.vector if local_head % 2 else nisa.engine.scalar,
            )

    # ``own_rows_ahead``: the stack-form current rows of every pass, in heap tiles (``own_rows_pre``; the rows of each
    # sequence-major pass in ``own_rows_seq``, checked against the passes below).
    own_rows_on = (
        q_mx
        and pack_options["own_rows_ahead"]
        and pack_options["p2"]
        and pack_options["own_state_stack"]
        and not (split_ext and split.get("guards", False))
    )
    own_rows_pre, own_rows_seq = {}, None
    if q_first:
        if q_mx:
            # ``q_mx_gather``: the own-key scores of the query store read the current keys.
            cur_rows, write_rows, cache_writes, cur_keys, cos_x, sin_x = _kv_part()
            if own_rows_on:
                # The current rows load here, before the query store and the query gathers, so no load of them
                # shares the DMA engines with a gather in flight. The k7 call attends sequence-major passes.
                seq_nb = tr // q_rows
                own_rows_seq = [
                    [j * tokens + p * seq_nb + bb for j in range(q_rows) for bb in range(seq_nb)]
                    for p in range(-(-total_rows // _TAIL_ROWS))
                ]
                for row_cb in range(0, total_rows, _TAIL_ROWS):
                    if not (query_rows and row_cb >= query_rows):
                        own_rows_pre[row_cb] = _own_rows_stack_p2(
                            cur_rows, own_rows_seq[row_cb // _TAIL_ROWS], q_rows, f"{tag}_c{row_cb}", sb, keep=True
                        )[0]
        q_call, q_ag, q_slot = _q_call_gather(q_split)
        if not q_mx:
            cur_rows, write_rows, cache_writes, cur_keys, cos_x, sin_x = _kv_part(after=q_ag)

    # Close the projection scope: the projection scratch and the transient weights free here, before
    # the attention. The attention keeps resident only q_local, cur_rows, cur_keys, and kv_b (heap).
    sb.close_scope()

    # Steps 15-17: attend the query rows in chunks of at most 16. Each chunk gathers only its own
    # queries over CP32, attends the current block and the prior-key groups, and writes its slice of the
    # full state. The prior keys reload per chunk (cheap vs the Q*B-row attention compute), which keeps
    # the resident set bounded so the loop path fits the explicit SBUF region at B up to 8.
    group_blocks = min(local_blocks, _GROUP_BLOCKS)
    num_keys = group_blocks * _BLOCK
    use_loop = loop_path and local_blocks > _MAX_UNROLLED_BLOCKS
    slot_offset = sb.tile((_P_MAX, _GROUP_BLOCKS * _BLOCK), fp32)
    if not use_loop:
        nisa.iota(
            dst=slot_offset[_BIAS_ROW:bias_end, 0:num_keys],
            pattern=[[lay.block_stride, group_blocks], [1, _BLOCK]],
            offset=0,
        )
    loop_row = pads = None
    # rows_pair_tail applies where an iteration holds 2 group pairs; elsewhere the program is the knob-off one.
    pair_tail = False
    unroll = 1
    loop_tail = 0
    # The packed prior loop builds its own items (``_attend_prior_packed``).
    if use_loop and not packed_prior:
        group_count = local_blocks // _GROUP_BLOCKS
        loop_groups = _loop_groups(kv_pool, group_count, loop_group_cap)
        pair_tail = rows_pair_tail and loop_groups == 2 * _TS_GROUPS
        # rows_unroll2 applies where the table holds whole trips of 2 iterations.
        if pair_tail and rows_unroll2 and group_count % (2 * loop_groups) == 0:
            unroll = 2
        # Each tail of loop_tail_groups applies where an iteration holds more groups than it; without one the program
        # is the knob-off one.
        loop_tail = sum(t for t in loop_tails(loop_tail_groups) if t < loop_groups) + (loop_tail_groups & _FULL_TAIL)
        # Bitwise visits the current rows' group too, so count iterations from position[b] + Q - 1.
        iter_extra = (q_rows - 1) if bitwise else 0
        loop_row, pads = _loop_metadata(
            token_meta,
            token_f,
            key_start,
            tokens,
            group_count,
            loop_groups,
            iter_extra,
            tag,
            sb,
            lay,
            pad_skip=rows_pad_skip,
            live_rows=live_rows,
            **({"pair_tail": True} if pair_tail else {}),
            **({"unroll": 2} if unroll == 2 else {}),
            **({"tail_groups": loop_tail} if loop_tail else {}),
        )
    cur_latent_keys = cur_keys[0:_P_MAX, 0:_KV_LORA_TILES, 0:total_rows]
    cur_rope_keys = cur_keys[0:_ROPE, _KV_LORA_TILES, 0:total_rows]
    # With fit_passes the multi-pass bitwise loop path loads the output and gate weights in each pass after its
    # prior-key loop: held across the loop, they leave no room for its key buffers. Otherwise they load once here.
    late_weights = fit_passes and bitwise and use_loop and total_rows > _TAIL_ROWS
    # late_tail_weights moves only the tail weight loads after the loop (the fit form's other changes stay off).
    late_load = late_weights or (late_tail_weights and bitwise and use_loop)
    kernel_assert(
        not o_proj_ahead
        or (
            late_load
            and not (late_weights or late_tail_frames or late_o_proj_weights or tail_weights_after_rs)
            and total_rows <= _TAIL_ROWS
        ),
        "o_proj_ahead is a form of late_tail_weights on the loop path with one pass (not fit_passes, late_tail_frames, "
        "late_o_proj_weights or tail_weights_after_rs: the tail's o_proj_dmas, which tail_o_proj_first orders, hold "
        "no ahead load)",
    )
    kernel_assert(
        not late_tail_frames or (late_load and late_o_proj_weights and not late_weights and not bf16_state),
        "late_tail_frames needs late_tail_weights and late_o_proj_weights on the bitwise loop path, with an FP32 state "
        "and no fit passes",
    )

    def load_tail_weights(suffix="", o_proj=True, engine=None, after=(), gate_after=None):
        """Load the output gate and o_proj weight chunks of the tail (``o_proj`` False: a loader of the o_proj
        chunks in their place, which the tail calls after its value projection). ``engine`` (default SP) is the DMA
        queue engine, and each DMA waits for ``after`` (a runtime dependency; ``tail_weights_after_rs``); the MXFP8
        gate DMAs wait for ``gate_after`` in its place when given (``GATE_WEIGHTS_AFTER_SUMS``)."""
        kernel_assert(not after or mx_scales is not None, "tail weights after the reduce-scatter need the MXFP8 gate")
        gate_chunks = []
        if output_gate and mx_scales is not None:
            gate_dep = after if gate_after is None else gate_after
            gate_chunks.append(_load_mx_gate_cols(mla_in_proj, mx_scales, sb, f"{tag}{suffix}", engine, gate_dep))
        if output_gate and mx_scales is None:
            for chunk_start in range(0, _HIDDEN_TILES, _GATE_CHUNK_TILES):
                chunk_tiles = min(_GATE_CHUNK_TILES, _HIDDEN_TILES - chunk_start)
                gate_w = sb.tile((_P_MAX, chunk_tiles, gate_local), bf16)
                region_start = qkv_region + chunk_start * gate_local
                gate_weight_dma(
                    nisa.dma_copy(
                        dst=gate_w,
                        src=mla_in_proj[0:_P_MAX, region_start : region_start + chunk_tiles * gate_local],
                        dge_mode=hwdge,
                        engine=sync,
                        name=f"{tag}_mla_w_gate_{chunk_start}{suffix}",
                    )
                )
                gate_chunks.append((chunk_start, chunk_tiles, gate_w))
        if not o_proj:
            return gate_chunks, lambda: load_o_proj_weights(suffix, engine, after)
        return gate_chunks, load_o_proj_weights(suffix, engine, after)

    def load_o_proj_weights(suffix="", engine=None, after=()):
        """Load the o_proj weight chunks of the tail (``engine`` and ``after`` as ``load_tail_weights``)."""
        if mla_o_proj_scale is not None:
            return load_mx_o_proj_weights(suffix, engine, after)
        o_w_chunks = []
        for chunk_start in range(0, _HIDDEN_TILES, _O_PROJ_CHUNK_TILES):
            chunk_tiles = min(_O_PROJ_CHUNK_TILES, _HIDDEN_TILES - chunk_start)
            o_w = sb.tile((_P_MAX, local_heads, chunk_tiles * _P_MAX), bf16)
            cols = mla_o_proj[0:gate_local, chunk_start * _P_MAX : (chunk_start + chunk_tiles) * _P_MAX]
            dma = gate_weight_dma(
                nisa.dma_copy(
                    dst=o_w,
                    src=cols.reshape_dim(0, (local_heads, _P_MAX)).permute((1, 0, 2)),
                    dge_mode=hwdge,
                    engine=sync if engine is None else engine,
                    name=f"{tag}_mla_w_o_{chunk_start}{suffix}",
                )
            )
            if after:
                _instruction(dma).depends_on(*after, kind=nisa.dependency.runtime)
            o_proj_dmas.append(dma)
            o_w_chunks.append((chunk_start, chunk_tiles, o_w))
        return o_w_chunks

    def mx_o_proj_dma(dst, src, name, engine, after=()):
        """One DMA of the MXFP8 o_proj weights (``after`` as ``load_tail_weights``), kept in ``o_proj_dmas``."""
        dma = gate_weight_dma(nisa.dma_copy(dst=dst, src=src, dge_mode=hwdge, engine=engine, name=name))
        if after:
            _instruction(dma).depends_on(*after, kind=nisa.dependency.runtime)
        o_proj_dmas.append(dma)
        return dma

    def load_mx_o_proj_weights(suffix="", engine=None, after=(), chunks=None):
        """Load the MXFP8 o_proj (``mla_o_proj_scale``): the compact scales into one ``[128, 7168]`` tile, then the
        chunks of ``[128, 4 * chunk_tiles * 128]`` bytes (``MxOChunk``). ``chunks`` (``o_proj_ahead``): the heap
        chunks to load into; None allocates them."""
        dma_engine = sync if engine is None else engine
        o_scale = sb.tile((_P_MAX, _HIDDEN), nl.uint8) if chunks is None else chunks[0][2].scale

        def load_scale(dst, src, quadrant):
            return mx_o_proj_dma(dst, src, f"{tag}_mla_w_o_scale_{quadrant}{suffix}", dma_engine, after)

        load_mx_compact_scales(mla_o_proj_scale, o_scale, load_scale)
        o_w_chunks = []
        for chunk_idx, chunk_start in enumerate(range(0, _HIDDEN_TILES, _O_PROJ_CHUNK_TILES)):
            chunk_tiles = min(_O_PROJ_CHUNK_TILES, _HIDDEN_TILES - chunk_start)
            width = chunk_tiles * _P_MAX * MX_LANES
            if chunks is None:
                o_w = sb.tile((_P_MAX, width), nl.uint8)
            else:
                o_w = chunks[chunk_idx][2].data
            first = chunk_start * _P_MAX * MX_LANES
            src = mla_o_proj[0, 0:_P_MAX, first : first + width]
            mx_o_proj_dma(o_w, src, f"{tag}_mla_w_o_{chunk_start}{suffix}", dma_engine, after)
            o_w_chunks.append((chunk_start, chunk_tiles, MxOChunk(o_w, o_scale)))
        return o_w_chunks

    # The o_proj weight DMAs in issue order (``tail_o_proj_first`` pins the tail gate exps after the last one).
    o_proj_dmas = []

    def load_o_proj_ahead(q_loads):
        """``o_proj_ahead``: load the o_proj chunks into their heap tiles on the Activation HWDGE queue. Each DMA waits
        for the last query load of the pass (a semaphore wait), which waits for the q-call gather, so the chunks
        stream after its flight, in the pre-loop chain and the loop, which run no collective. The schedule edges keep
        the cache writes ahead of the chunks on a shared queue and keep the DMA prefetch pass from hoisting the chunks
        into the flight of the gather."""
        if mla_o_proj_scale is not None:
            first_dma = len(o_proj_dmas)
            load_mx_o_proj_weights("_ahead", nisa.engine.scalar, chunks=o_ahead_chunks)
            for dma in o_proj_dmas[first_dma:]:
                _instruction(dma).depends_on(q_ag, *cache_writes, kind=nisa.dependency.schedule)
                _instruction(dma).depends_on(q_loads[-1], kind=nisa.dependency.runtime)
            # As the BF16 form, the ahead loads are not tail o_proj DMAs.
            del o_proj_dmas[first_dma:]
            return
        for chunk_start, chunk_tiles, o_w in o_ahead_chunks:
            cols = mla_o_proj[0:gate_local, chunk_start * _P_MAX : (chunk_start + chunk_tiles) * _P_MAX]
            dma = _instruction(
                gate_weight_dma(
                    nisa.dma_copy(
                        dst=o_w,
                        src=cols.reshape_dim(0, (local_heads, _P_MAX)).permute((1, 0, 2)),
                        dge_mode=hwdge,
                        engine=nisa.engine.scalar,
                        name=f"{tag}_mla_w_o_ahead_{chunk_start}",
                    )
                )
            )
            dma.depends_on(q_ag, *cache_writes, kind=nisa.dependency.schedule)
            dma.depends_on(q_loads[-1], kind=nisa.dependency.runtime)

    gate_chunks, o_w_chunks = ([], []) if late_load else load_tail_weights()
    # The split program's extended batch returns the rows of its B request slots only (j B + b): the foreign rows have no
    # output.
    out_rows = q_rows * split["ext"] if split_ext else total_rows
    y_hT = sb.tile((_P_MAX, out_rows, _HIDDEN_TILES), bf16)
    prev_tail = None
    # Every multi-chunk pass runs its tail on the same number of rows (the last chunk is padded), so the
    # tail reduce-scatter and all-gather have one shape per kernel. A different shape would let the
    # runtime pick a different mesh algorithm and summation order, which flips the last bits run to run.
    # Hold-free o_proj all-gather source: two persistent buffers, alternating by chunk parity. Chunk k+2
    # reuses chunk k's buffer only after chunk k+1's all-gather (the inter-pass chaining below), by when every
    # CP peer has finished reading chunk k's source. On the standalone path these are the arena columns whose
    # last use (the tail all-gather) reserves the whole arena, so the q and qkv sources above also stay live.
    ag_call = None
    if out_ag_call and not lay.one_head:
        if sources is not None and "ag_call" in sources:
            ag_call = sources["ag_call"]
        elif arena_ag_bufs is not None:
            ag_call = arena_ag_bufs[0]
        else:
            ag_call = sb.tile((_P_MAX // cp, -(-total_rows // _TAIL_ROWS) * tr * _HIDDEN_TILES), bf16, keep=True)
        ag_bufs = None
    elif out_ar_call:
        # The output sum is an all-reduce in HBM: no all-gather source.
        ag_bufs = None
    elif sources is not None and "ag" in sources:
        ag_bufs = sources["ag"]
    elif arena_ag_bufs is not None:
        ag_bufs = arena_ag_bufs
    else:
        ag_bufs = [sb.tile((_P_MAX // cp, tr, _HIDDEN_TILES), bf16, keep=True) for _ in range(2)]
    # The state reduce-scatter runs in persistent HBM. CP128 always has the pair: an SBUF reduce-scatter of one row for
    # each of 128 ranks is not a mesh form of the runtime.
    if sources is not None and "state_hbm" in sources:
        state_pair = (sources["state_hbm"], sources["state_dst_hbm"])
    elif arena_state_hbm is not None:
        state_pair = (arena_state_hbm, arena_state_dst_hbm)
    elif lay.one_head:
        state_pair = (q_gather_hbm["state_hbm"], q_gather_hbm["state_dst_hbm"])
    else:
        state_pair = (None, None)
    # BF16-o: the sums reduce-scatter takes a pair of its own when the sources give one (``state_sums`` of
    # ``mla_block_collective_hbm``), so the two reduce-scatters of the state never write views of one tensor.
    sum_pair = (
        (sources["state_sum_hbm"], sources["state_sum_dst_hbm"])
        if sources is not None and "state_sum_hbm" in sources
        else (None, None)
    )
    kernel_assert(
        not state_bf16_o or (state_pair[0] is not None and not bf16_state and not lay.one_head),
        "state_bf16_o needs the FP32 HBM state pair (sources, the arena or CP128), not bf16_state, and more than one "
        "head a rank",
    )
    # The served CP32 drafter passes reserved reduce-scatter sources (with_rs), alternated by chunk parity.
    rs_bufs = sources["rs"] if (persistent_rs and sources is not None and "rs" in sources) else None
    # One output reduce-scatter a call: the passes write their partials to the reserved rs_call tile.
    rs_call = sources.get("rs_call") if (late_weights and sources is not None) else None
    passes = -(-total_rows // _TAIL_ROWS)
    # ``pass_tail_overlap``: the tail of pass p runs after the attention of pass p + 1; the odd passes take the second
    # state pair. A call of one pass keeps its program.
    tail_overlap = pass_tail_overlap and passes > 1
    kernel_assert(not tail_overlap or rs_call is None, "pass_tail_overlap runs the output collectives of each pass")
    state_pair_odd = (sources["state_hbm_1"], sources["state_dst_hbm_1"]) if tail_overlap else None
    kernel_assert(
        not tail_overlap or out_ag_call or len(sources.get("y_dst", ())) >= passes,
        "pass_tail_overlap needs one persistent output all-gather tile a pass (sources y_dst, "
        "mla_block_collective_sources y_dst_passes)",
    )
    # The final all-gather of each pass by its index, the last one, and the deferred tail of the pass before (a function
    # that computes it and returns its partial copies and its output collectives).
    overlap = {"ags": {}, "last": None, "pending": None}

    def _run_pending_tail():
        pending, overlap["pending"] = overlap["pending"], None
        if pending is not None:
            _copies, cc = pending()
            cc()

    out_ar_pair = None
    if out_ar_call:
        out_ar_pair = sources if sources is not None and "out_ar_src_hbm" in sources else arena_out_ar
        if out_ar_pair is None:
            out_ar_pair = _out_ar_pair(total_rows, tr, f"_{tag}")

    def out_ar_pass(pass_idx: int):
        """Return the ``out_hbm`` of ``_output_tail`` for pass ``pass_idx`` (``out_ar_call``, one pass a call output
        sum), or None."""
        if out_ar_pair is None or rs_call is not None:
            return None
        src, dst = out_ar_pair["out_ar_src_hbm"], out_ar_pair["out_ar_dst_hbm"]
        if passes == 1:
            return {"out_src_hbm": src, "out_dst_hbm": dst}
        rows = slice(pass_idx * _P_MAX, (pass_idx + 1) * _P_MAX)
        return {"out_src_hbm": src[rows, 0 : src.shape[1]], "out_dst_hbm": dst[rows, 0 : dst.shape[1]]}

    # One state reduce-scatter a call too, when the HBM pair holds a row for each head and pass.
    state_call = rs_call is not None and state_pair[0] is not None and state_pair[0].shape[0] == heads * passes > heads
    kernel_assert(
        not tail_weights_after_rs or (state_call and state_bf16_o and mx_scales is not None),
        "tail_weights_after_rs needs the call-level BF16-o state reduce-scatter (fit_passes with the call sources) "
        "and the MXFP8 mla_in_proj. Turn it off for this call.",
    )
    kernel_assert(
        out_ar_hook is None or (rs_call is not None and out_ar_call and not split_ext and not out_rs_tp),
        "out_ar_hook follows the call-level output all-reduce (out_ar_call with the rs_call partials), not the split "
        "export or out_rs_tp",
    )
    kernel_assert(
        state_rs_hook is None or state_call or state_bf16_o or passes == 1,
        "state_rs_hook follows the BF16-o state reduce-scatter of a pass, the call-level state reduce-scatter or the "
        "state reduce-scatter of a one-pass call",
    )
    if not q_first:
        q_call, q_ag, q_slot = _q_call_gather()
    kernel_assert(
        not o_proj_ahead or (q_call is not None and not q_fp8),
        "o_proj_ahead loads after the query loads of the BF16 call query gather (q_hbm, not q_fp8)",
    )
    # The fit form with one state reduce-scatter attends sequence-major passes: pass p holds the Q rows of the
    # sequences p * nb .. p * nb + nb - 1, so each sequence's keys load once a call (a token-major pass holds rows of
    # every sequence and reloads all keys). Each row's state goes to its token-major pass rows of the HBM pair, so
    # the state reduce-scatter and the tails do not change.
    seq_shape = bitwise and not exact_softmax and tr % q_rows == 0 and tokens % (tr // q_rows) == 0 and tr % tokens == 0
    # CP128 runs a reduce-scatter and a tail per pass, so its sequence-major passes move the rows of each pass. The
    # static path (up to 16 local blocks) loads each sequence's blocks once a pass too, so both paths take it. Each
    # row keeps its instructions, so the output is bitwise the same; one pass of up to 16 rows does not change.
    seq_rows = lay.one_head and total_rows > _TAIL_ROWS and seq_shape
    seq_major = seq_shape and ((use_loop and state_call and q_call is not None) or seq_rows)
    kernel_assert(tokens <= _MAX_SEQS or seq_major, f"B={tokens} > {_MAX_SEQS} needs the sequence-major fit passes")
    kernel_assert(
        not split_v5 or state_call or state_bf16_o,
        "split (v5) needs the call-level state reduce-scatter or the BF16-o pass reduce-scatter (its hook)",
    )
    kernel_assert(
        not split_ext or (state_call and state_bf16_o and seq_major and rs_call is not None and out_ar_call),
        "split (v5) ext needs the sequence-major passes, the call-level BF16-o state reduce-scatter and the call-level "
        "output all-reduce (fit_passes, q_hbm, state_bf16_o, out_ar_call, the state and rs_call sources): its export, "
        "merge and tails run the sequence-major passes",
    )
    if split_ext:
        kernel_assert(
            tuple(rs_call.shape[1:]) == (out_rows, _HIDDEN_TILES),
            f"split (v5) ext: the rs_call source holds the {out_rows} request rows (mla_block_collective_sources "
            f"out_rows), got {tuple(rs_call.shape)}",
        )
    if split_v5 and not state_call:
        # One pass set without the call-level reduce-scatter (k7b2): the export follows the state reduce-scatter of
        # the last pass, as its state_rs_hook (after the caller's hook, if any).
        caller_hook = state_rs_hook

        def state_rs_hook(hook_sb, caller_hook=caller_hook):
            if caller_hook is not None:
                caller_hook(hook_sb)
            _split_v5_export(split, q_rows, local_heads, tag, hook_sb)

    kernel_assert(
        not split_v1 or (seq_major and state_call),
        "split needs the sequence-major passes and the call-level state reduce-scatter (fit_passes, q_hbm, the state "
        "and rs_call sources)",
    )
    # ``seq_passes``: whole-sequence passes (a call of one pass keeps the token-major rows).
    draft_seq = seq_passes and passes > 1
    if draft_seq:
        seq_nb = _TAIL_ROWS // q_rows
        kernel_assert(
            not seq_major and not state_call and tokens % seq_nb == 0 and tokens // seq_nb == passes,
            f"seq_passes needs B = {tokens} in whole passes of {seq_nb} sequences, {passes} passes",
        )
    if seq_major:
        nb = tr // q_rows
        pass_rows = [[j * tokens + p * nb + bb for j in range(q_rows) for bb in range(nb)] for p in range(passes)]
    elif draft_seq:
        pass_rows = [
            [j * tokens + p * seq_nb + bb for j in range(q_rows) for bb in range(seq_nb)] for p in range(passes)
        ]
    else:
        pass_rows = [list(range(cb, min(cb + _TAIL_ROWS, total_rows))) for cb in range(0, total_rows, _TAIL_ROWS)]
    kernel_assert(
        q_split is None or (seq_major and q_split == pass_rows),
        "the split query gather stored the rows of other passes than the sequence-major passes of this call",
    )
    # ``state_sum_blocks``: the sum blocks of the call (``_store_state_p2_bf16_o`` ``sum_blocks``), stored by its last pass.
    sum_block_tile = None
    block_seqs = tr // q_rows if q_rows else 0
    if (
        packed_prior
        and pack_options["state_sum_blocks"]
        and pack_options["p2"]
        and state_bf16_o
        and state_call
        and sum_pair[0] is not None
        and split is None
        and not query_rows
        and total_rows % _TAIL_ROWS == 0
        and q_rows % 4 == 0
        and block_seqs > 0
        and q_rows // 4 <= _SUM_BLOCK // (passes * block_seqs)
        # 64 sums words a head: the sums pair padded to 256 bytes a rank (``cc_pad256``).
        and passes * sum_pair[0].shape[1] >= 4 * _SUM_BLOCK
    ):
        sum_block_tile = sb.tile((_P_MAX, 3, _SUM_BLOCK), fp32)
        # The pad words of each block go to HBM with it (never read): zeros, so every byte is known.
        nisa.memset(dst=sum_block_tile, value=0.0, engine=nisa.engine.vector)
    # ``state_sum_blocks_pass``: the sum blocks of a call of one pass with the per-pass state reduce-scatter.
    sum_blocks_pass = False
    if (
        sum_block_tile is None
        and packed_prior
        and pack_options["state_sum_blocks_pass"]
        and pack_options["p2"]
        and state_bf16_o
        and not state_call
        and passes == 1
        and sum_pair[0] is not None
        and split is None
        and not query_rows
        and total_rows == _TAIL_ROWS
        and q_rows % 4 == 0
        and block_seqs > 0
        and q_rows // 4 <= _SUM_BLOCK // block_seqs
        and sum_pair[0].shape[1] >= 4 * _SUM_BLOCK
    ):
        sum_blocks_pass = True
        sum_block_tile = sb.tile((_P_MAX, 3, _SUM_BLOCK), fp32)
        nisa.memset(dst=sum_block_tile, value=0.0, engine=nisa.engine.vector)
    # ``state_dense_tail``: the dense output tail of a call of one pass with the BF16-o per-pass state reduce-scatter.
    dense_tail_pass = bool(
        packed_prior and pack_options["state_dense_tail"] and state_bf16_o and not state_call and passes == 1
    )
    # ``p2``: the state store DMAs of the pass before. The next pass's state tile takes their source bytes, so its own-key
    # writes wait for their completion (a runtime dependency).
    p2_stores = ()
    # ``state_o_heap``: the heap numerator tile of the pass before, released after this pass's loops.
    o16_heap = []
    # ``fp8_pv_tables_ahead``: the heap tiles that the first pass loaded for the second (keyed by the second pass's row).
    tables_ahead = {}
    kernel_assert(
        not query_rows
        or (0 < query_rows <= total_rows and not seq_major and rs_call is None and ag_call is None and not out_ar_call),
        f"query_rows is 1 to Q*B = {total_rows} rows of the per-pass output path; got {query_rows}",
    )
    # ``rows_mx_keys``: the unit E8M0 MX scales of the latent QK, one tile for the whole call (the loop only reads them).
    mx_scale = None
    if rows_mx_keys is not None:
        mx_scale = sb.tile((_P_MAX, _GROUP_BLOCKS * _BLOCK), nl.uint8)
        nisa.memset(dst=mx_scale, value=_MX_UNIT_SCALE)
    # ``pass_q_prefetch``: the heap tiles of the next pass's first gather tiles, and the pass whose queries they hold.
    q_prefetch_tiles = []
    q_prefetch_pass = [None]
    if pass_q_prefetch and q_call is not None and not lay.one_head and total_rows > _TAIL_ROWS:
        q_prefetch_tiles = [
            sb.tile((_P_MAX, gather_width), bf16, name=f"{tag}_mla_q_prefetch_{i}", keep=True)
            for i in range(pass_q_prefetch)
        ]
    # ``q_mx_pass_prefetch``: the heap stage tiles and own-key scores of the next full pass's MX query bytes (loaded after
    # the loop of the pass before, ahead of its state stores), and the pass whose bytes they hold.
    q_mx_pre = None
    q_mx_pre_pass = [None]
    if (
        q_mx
        and pack_options["q_mx_pass_prefetch"]
        and total_rows > _TAIL_ROWS
        and not (split_ext and split.get("guards", False))
    ):
        q_mx_pre = (
            _q_mx_stage(_TAIL_ROWS, local_heads, sb, name=f"{tag}_mla_q_mx_pre", keep=True, packed=q_mx_packed),
            sb.tile((heads, _TAIL_ROWS), fp32, name=f"{tag}_mla_q_mx_pre_scores", keep=True),
        )

    # ``own_rows_ahead``: the rows loaded before the query store must be the rows of these passes; without the q-first
    # path the current rows load now (after the query store of the call, before the passes) into heap tiles.
    kernel_assert(
        own_rows_seq is None or own_rows_seq == pass_rows,
        "own_rows_ahead loads the current rows of sequence-major passes before the query store: these passes differ",
    )
    if own_rows_on and own_rows_seq is None:
        for row_cb in range(0, total_rows, _TAIL_ROWS):
            if not (query_rows and row_cb >= query_rows):
                own_rows_pre[row_cb] = _own_rows_stack_p2(
                    cur_rows, pass_rows[row_cb // _TAIL_ROWS], q_rows, f"{tag}_c{row_cb}", sb, keep=True
                )[0]

    def prefetch_queries(next_cb):
        """``pass_q_prefetch``: load the first gather tiles of the pass at row ``next_cb`` into the prefetch tiles, on the
        ScalarE hardware DGE queue (the same source rows and bytes as the pass's own query loads)."""
        next_rows = pass_rows[next_cb // _TAIL_ROWS]
        next_cr = len(next_rows)
        queue = {
            **{k: v for k, v in query_dma.items() if k == "priority"},
            "dge_mode": hwdge,
            "engine": nisa.engine.scalar,
        }
        for tile, q_pre in enumerate(q_prefetch_tiles):
            tile_base = tile * _MAX_TOKENS
            if tile_base >= next_cr:
                break
            for tok in range(min(_MAX_TOKENS, next_cr - tile_base)):
                if q_fp8 == "latent":
                    # The casting latent load takes no HWDGE (NCC_IBIR098): the default queue; the rope part the queue.
                    _q_latent_load(
                        q_pre[tok * _TP : (tok + 1) * _TP, 0:gather_width],
                        q_call[1],
                        *q_slot(next_rows[tile_base + tok]),
                        local_heads,
                        f"{tag}_mla_q_prefetch_{next_cb}_{tile}_{tok}",
                        queue,
                    )
                    continue
                prefetch = nisa.dma_copy(
                    dst=q_pre[tok * _TP : (tok + 1) * _TP, 0:gather_width],
                    src=q_pair_src(next_rows[tile_base + tok]),
                    name=f"{tag}_mla_q_prefetch_{next_cb}_{tile}_{tok}",
                    **queue,
                )
                if q_prefetch_after:
                    _instruction(prefetch).depends_on(*q_prefetch_after, kind=nisa.dependency.schedule)
        q_prefetch_pass[0] = next_cb

    # ``pass_rows_ahead``: the next pass's own-key current rows and its query gather tiles past the prefetch tiles, loaded
    # after this pass's loop and before its state stores (``issue_rows_ahead``), and the heap tiles that hold them.
    rows_batched_on = pass_rows_ahead and packed_prior and pack_options["p2"]
    rows_ahead_on = (
        rows_batched_on and state_call and q_call is not None and not lay.one_head and total_rows > _TAIL_ROWS
    )
    rows_ahead = {}

    def issue_rows_ahead(next_cb):
        """``pass_rows_ahead``: load the pass at row ``next_cb``'s query gather tiles past the ``pass_q_prefetch`` tiles
        (its own query loads, on the query queue) and its own-key current rows (``_own_rows_p2``) into heap tiles."""
        next_cr = min(_TAIL_ROWS, total_rows - next_cb)
        next_rows = pass_rows[next_cb // _TAIL_ROWS]
        q_tiles, heap_tiles, loads = {}, [], []
        for tile in range(len(q_prefetch_tiles), (next_cr + _MAX_TOKENS - 1) // _MAX_TOKENS):
            tile_base = tile * _MAX_TOKENS
            tile_tokens = min(_MAX_TOKENS, next_cr - tile_base)
            q_ahead = sb.tile(
                (_TP * tile_tokens, gather_width), bf16, name=f"{tag}_mla_q_ahead_{next_cb}_{tile}", keep=True
            )
            heap_tiles.append(q_ahead)
            for tok in range(tile_tokens):
                load = nisa.dma_copy(
                    dst=q_ahead[tok * _TP : (tok + 1) * _TP, 0:gather_width],
                    src=q_pair_src(next_rows[tile_base + tok]),
                    name=f"{tag}_mla_q_load_{next_cb}_{tile}_{tok}",
                    **query_dma,
                )
                loads.append(_instruction(load))
            q_tiles[tile] = q_ahead
        own_rows, row_tiles, row_loads = _own_rows_p2(cur_rows, next_rows, q_rows, f"{tag}_c{next_cb}", sb, keep=True)
        loads += row_loads
        # The loads keep their order (schedule edges, no wait), and the state stores issue after the last one.
        for before, after in zip(loads, loads[1:]):
            after.depends_on(before, kind=nisa.dependency.schedule)
        rows_ahead.clear()
        rows_ahead.update(cb=next_cb, q=q_tiles, own_rows=own_rows, tiles=heap_tiles + row_tiles, last=loads[-1])

    def _attend_pass(cb):
        """One attention pass of the rows ``pass_rows[cb // 16]`` (the body of the pass loop below). The split v5 "one"
        program runs its foreign passes inside the foreign-compute guards."""
        nonlocal q_tile_idx, p2_stores, prev_tail, gate_chunks, o_w_chunks
        rows = pass_rows[cb // _TAIL_ROWS]
        cr = len(rows)
        if query_rows and cb >= query_rows:
            # No output of the caller reads the rows of this pass (query_rows): their attention output is 0.
            nisa.memset(dst=y_hT[0:_P_MAX, cb : cb + cr, 0:_HIDDEN_TILES], value=0.0)
            return
        # ``prior_hoist``: the first key groups of each sequence of the pass, traced before the pass's query loads.
        hoisted_keys = None
        if prior_hoist:
            hoisted_keys = _hoist_prior_keys(
                rows,
                kv_pool,
                block_table,
                row_threshold if bitwise else mask_threshold,
                lower_threshold,
                slot_offset,
                tokens,
                local_blocks,
                prior_hoist,
                bitwise,
                lay,
                n_mask,
                _KeyPath(
                    ("mx_" + rows_mx_keys)
                    if rows_mx_keys
                    else (("dr_inplace" if rows_dr_inplace else "dr") if rows_dr else "bf16"),
                    nisa.engine.scalar if rows_dr_act_copies else nisa.engine.vector,
                    mx_scale=mx_scale,
                ),
                f"{tag}_c{cb}",
                sb,
            )
        with sb.scope("row_chunk"):
            state_flat = None
            stack_upper = False
            p2 = packed_prior and pack_options["p2"]
            # ``o_proj_ahead``: the query loads of the pass, which the o_proj chunk loads wait for.
            pass_q_loads = []
            # The split's last sequence-major pass holds the foreign and the pad slot: no own-key term (the owner adds
            # the current key of the donor rows).
            # The foreign passes (the last of v1; those past the B request slots of the v5 extended batch).
            split_pass = (split_v1 and cb // _TAIL_ROWS == passes - 1) or (
                split_ext and cb // _TAIL_ROWS >= split["ext"] // 2
            )
            own_weight = 0.0 if split_pass else (lay.rank_scale if exact_softmax else lay.rank_current_weight)
            if loop_p2:
                # ``loop_p2``: the (row, head) state of ``p2``; the tail stores it head-major with the pad heads.
                state_flat = sb.tile((_P_MAX, (pack_p2_groups(q_rows) * (cr // q_rows) + 1) * _STATE), fp32)
                state = None
            elif p2:
                # ``p2``: the state of the pass in the (row, head) layout of the packed loop: column g S + s holds group
                # g of sequence s on its 128 partitions, and column G S takes the null items.
                p2_columns = pack_p2_groups(q_rows) * (cr // q_rows)
                state_flat = sb.tile((_P_MAX, (p2_columns + 1) * _STATE), fp32)
                state = None
                nisa.memset(
                    dst=state_flat.reshape((_P_MAX, p2_columns + 1, _STATE))[0:_P_MAX, p2_columns, 0:_STATE],
                    value=0.0,
                    engine=nisa.engine.vector,
                )
            elif packed_prior:
                # The packed loop adds through TI to the 2D storage of the state; column tr takes its null items.
                state_flat = sb.tile((heads, (tr + 1) * (_KV_LORA + 1)), fp32)
                state = state_flat.reshape((heads, tr + 1, _KV_LORA + 1))
                nisa.memset(dst=state[0:heads, tr, 0 : _KV_LORA + 1], value=0.0, engine=nisa.engine.vector)
            else:
                # ``stack_rows`` (``STACK_STATE_UPPER``): the shadow rows of the stacked units take the upper partitions.
                stack_upper = stack_rows and STACK_STATE_UPPER and not lay.one_head and not bf16_state
                state_parts = cp if lay.one_head else (_P_MAX if stack_upper else heads)
                state = sb.tile((state_parts, tr, _KV_LORA + 1), fp32)
            if lay.one_head and state is not None:
                # The rows of the pad heads reduce-scatter to the ranks at and above ``heads``: zero numerator, unit sum.
                nisa.memset(dst=state[heads:cp, 0:tr, 0:_KV_LORA], value=0.0)
                nisa.memset(dst=state[heads:cp, 0:tr, _KV_LORA : _KV_LORA + 1], value=1.0)
            kernel_assert(not (p2 or loop_p2) or cr == tr, "pack p2 and loop_p2 need full passes")
            if cr < tr:
                # Padding rows: zero numerator, unit sum (finite, discarded after the tail).
                nisa.memset(dst=state[0:heads, cr:tr, 0:_KV_LORA], value=0.0)
                nisa.memset(dst=state[0:heads, cr:tr, _KV_LORA : _KV_LORA + 1], value=1.0)
            softmax = _Softmax(exact_softmax, scale, tr, heads, sb)
            if cr < tr and exact_softmax:
                nisa.memset(dst=softmax.running_max[0:heads, cr:tr], value=0.0)
            # ``own_state_first``: the own-key state of the pass from the rows loaded ahead, before the query loads.
            own_first = (
                p2
                and pack_options["own_state_first"]
                and pack_options["own_state_stack"]
                and cb in own_rows_pre
                and not rows_batched_on
            )
            if own_first:
                _own_key_state_p2(
                    state_flat,
                    rows,
                    cur_rows,
                    q_rows,
                    own_weight,
                    f"{tag}_c{cb}",
                    sb,
                    after=p2_stores,
                    row_after=p2_stores,
                    stack=True,
                    stack_rows=own_rows_pre[cb],
                )
            # ``own_rows_ahead``: the current rows and the own-key weight of the pass rows, before the query loads.
            own_rows = None
            if own_rows_ahead:
                own_rows = _own_rows_ahead(cur_rows, rows[0:cr], heads, own_weight, f"{tag}_c{cb}", sb)
            # ``mx_natural``: the loop takes the latent query tiles of the natural MX grouping (``q_nat_all``, built from
            # each gather tile with the feature-major tiles); the feature-major tiles serve only the own-key scores, in a
            # frame that ends before the loop.
            natural = packed_prior and pack_options["mx_natural"]
            # ``q_mx_gather``: the MX query words and scales of the pass come from the gather (``_q_mx_load``), in place
            # of the natural tiles; there are no feature-major tiles (the own-key scores come from the gather too).
            q_mx_words = None
            if q_mx:
                p2_cols = pack_p2_groups(q_rows) * (cr // q_rows) * _P_MAX
                q_mx_words = (sb.tile((_P_MAX, p2_cols), fp32), sb.tile((_P_MAX, p2_cols), nl.uint8))
            q_nat_all = sb.tile((_P_MAX, cr, _KV_LORA_TILES, heads), bf16) if natural and not q_mx else None
            if not natural:
                q_nope_all = sb.tile((_P_MAX, cr, _KV_LORA_TILES, heads), bf16)
            q_rope_all = sb.tile((_P_MAX, cr, heads), bf16)
            if natural:
                sb.open_scope("q_own")
                q_nope_all = None if q_mx else sb.tile((_P_MAX, cr, _KV_LORA_TILES, heads), bf16)
            if folded_reference and ts:
                # The rope matmul reads up to the folded reference rows: the rows below them stay zero.
                nisa.memset(dst=q_rope_all[_BIAS_ROW:_REF_ROW, 0:cr, 0:heads], value=0.0)
            # ``dr_keys``: the e4m3 double_row stationaries of the pass rows, from each gather tile (with ``mx_keys`` the
            # MX query words below in their place; the other query consumers take the e4m3 query of ``dr_keys``).
            pack_mx = packed_prior and pack_options["mx_keys"]
            e4m3_query = (packed_prior and pack_options["dr_keys"]) or rows_dr
            dr_rows = None
            if e4m3_query and not pack_mx:
                dr_rows = sb.tile((_P_MAX, cr, 2, _DR_BLOCKS, heads), nl.float8_e4m3fn)
                dr_identity = sb.tile((_P_MAX, _P_MAX), bf16)
                nl.shared_identity_matrix(_P_MAX, dtype=bf16, dst=dr_identity)
            # ``rows_mx_keys``: the MX query words (hi) of the pass rows, from each gather tile.
            # (``mx_q_rows`` to avoid the in-projection's ``mx_rows`` row count above.)
            mx_q_rows = None
            if rows_mx_keys is not None or pack_mx:
                mx_q_rows = sb.tile((_P_MAX, cr, 1, heads), fp32)
            if n_mask > 1:
                # Query row j selects mask row j of each key group: a one-hot over the mask rows (j = r // B). A
                # sequence-major pass holds the rows (j, bb) of nb sequences, j = 0 .. Q - 1.
                row_of_query = sb.tile((_P_MAX, cr, heads), fp32)
                nisa.iota(
                    dst=row_of_query[_BIAS_ROW:bias_end, 0:cr, 0:heads],
                    pattern=[[-1, cr // nb], [0, nb], [0, heads]]
                    if seq_major
                    else [[-1, cr // tokens], [0, tokens], [0, heads]],
                    offset=0 if seq_major else -(cb // tokens),
                    channel_multiplier=1,
                )
                nisa.tensor_scalar(
                    dst=q_rope_all[_BIAS_ROW:bias_end, 0:cr, 0:heads],
                    data=row_of_query[_BIAS_ROW:bias_end, 0:cr, 0:heads],
                    op0=nl.equal,
                    operand0=0.0,
                )
            else:
                nisa.memset(dst=q_rope_all[_BIAS_ROW : _BIAS_ROW + 1, 0:cr, 0:heads], value=1.0)
            n_tiles = 0 if lay.one_head or q_mx else (cr + _MAX_TOKENS - 1) // _MAX_TOKENS
            q_load = None
            if lay.one_head:
                pass_index = cb // _TAIL_ROWS
                q_load = _gather_head_queries(
                    q_local,
                    cb,
                    cr,
                    q_gather_hbm["q_src_hbm"][pass_index],
                    q_gather_hbm["q_dst_hbm"][pass_index],
                    q_nope_all,
                    q_rope_all,
                    heads,
                    tp_group,
                    f"{tag}_c{cb}",
                    sb,
                    row_ids=rows if seq_rows else None,
                    **({"dr_rows": dr_rows, "dr_identity": dr_identity} if dr_rows is not None else {}),
                )
            # ``q_gather_buffers`` 2: two gather tiles in turn, in a scope of the gather only (the loop does not hold
            # them), for a pass that loads 2 or more gather tiles from the call query gather.
            q_pair = None
            ahead_q = rows_ahead["q"] if rows_ahead.get("cb") == cb else {}
            first_loaded = len(q_prefetch_tiles) if q_prefetch_pass[0] == cb else 0
            loaded = [tile for tile in range(first_loaded, n_tiles) if tile not in ahead_q]
            loaded_tiles = len(loaded)
            if q_gather_buffers == 2 and q_call is not None and loaded_tiles > 1:
                sb.open_scope("gather_pair")
                q_pair = [sb.tile((_P_MAX, gather_width), bf16, name=f"{tag}_mla_q_pair_{cb}_{i}") for i in range(2)]
            for tile in range(n_tiles):
                tile_base = tile * _MAX_TOKENS
                tile_tokens = min(_MAX_TOKENS, cr - tile_base)
                gather_rows = _TP * tile_tokens
                # Scope each tile's gather scratch; the gathered queries land in the chunk's q_nope_all.
                with sb.scope("gather_tile"):
                    tile_idx = q_tile_idx
                    q_tile_idx += 1
                    # The gather source must start at partition 0, so a DMA shifts the tile's rows down. The
                    # source and destination are distinct persistent tiles. A peer can read the source
                    # after the local output lands and write the destination before this rank arrives, so no
                    # later tile may take either's bytes early. The loop path keeps the local scratch tiles.
                    if q_call is not None and tile in ahead_q:
                        # ``pass_rows_ahead``: the pass before loaded this tile's rows into a heap tile.
                        q_all = ahead_q[tile][0:gather_rows, 0:gather_width]
                    elif q_call is not None and q_prefetch_pass[0] == cb and tile < len(q_prefetch_tiles):
                        # ``pass_q_prefetch``: the pass before loaded this tile's rows into a prefetch tile.
                        q_all = q_prefetch_tiles[tile][0:gather_rows, 0:gather_width]
                    elif q_call is not None:
                        # Token-major tile: partition t * 32 + g holds rank g's queries of token t (one load a token).
                        # With q_fp8 the HBM rows are e4m3 and the load casts them to BF16 (exact). neuronx-cc takes
                        # no HWDGE for that casting load (NCC_IBIR098: DGE None or SWDGE), so it keeps the default
                        # queue, as the loads do without the query priority.
                        q_load_dma = {} if q_fp8 else query_dma
                        if q_pair is not None:
                            # ``q_gather_buffers`` 2: the loaded gather tiles take the two tiles in turn.
                            q_all = q_pair[loaded.index(tile) % 2][0:gather_rows, 0:gather_width]
                        else:
                            q_all = sb.tile((gather_rows, gather_width), bf16)
                        for tok in range(tile_tokens):
                            if q_fp8 == "latent":
                                # The casting latent part on the default queue, the BF16 rope part with the priority.
                                _q_latent_load(
                                    q_all[tok * _TP : (tok + 1) * _TP, 0:gather_width],
                                    q_call[1],
                                    *q_slot(rows[tile_base + tok]),
                                    local_heads,
                                    f"{tag}_mla_q_load_{cb}_{tile}_{tok}",
                                    query_dma,
                                )
                                continue
                            q_load = nisa.dma_copy(
                                dst=q_all[tok * _TP : (tok + 1) * _TP, 0:gather_width],
                                src=q_pair_src(rows[tile_base + tok]),
                                name=f"{tag}_mla_q_load_{cb}_{tile}_{tok}",
                                **q_load_dma,
                            )
                            if o_ahead_chunks is not None:
                                pass_q_loads.append(q_load)
                    else:
                        if sources is not None and "q" in sources:
                            q_local_tile = sources["q"][tile_idx]
                            q_all = sources["q_dst"][tile_idx]
                        elif arena_q_tiles is not None:
                            q_local_tile = arena_q_tiles[tile_idx]
                            q_all = arena_q_dsts[tile_idx]
                        else:
                            q_local_tile = sb.tile((_MAX_TOKENS, gather_width), bf16)
                            q_all = sb.tile((gather_rows, gather_width), bf16)
                        nisa.dma_copy(
                            dst=q_local_tile[0:tile_tokens, 0:gather_width],
                            src=q_local[cb + tile_base : cb + tile_base + tile_tokens, 0:gather_width],
                            dge_mode=hwdge,
                            engine=sync,
                            name=f"{tag}_mla_q_tile_{cb}_{tile}",
                        )
                        ncc.all_gather(
                            srcs=[q_local_tile[0:tile_tokens, 0:gather_width]],
                            dsts=[q_all[0:gather_rows, 0:gather_width]],
                            replica_group=tp_group,
                            collective_dim=0,
                            name=f"{tag}_mla_q_all_gather_{cb}_{tile}",
                        )
                    if dr_rows is not None:
                        _dr_gather_tile_queries(
                            q_all, tile_tokens, q_call is not None, dr_rows, tile_base, dr_identity, sb
                        )
                    # The gathered rows are (rank, token) on the per-tile path and (token, rank) on the call path.
                    rows_split = (tile_tokens, _TP) if q_call is not None else (_TP, tile_tokens)
                    nope_order = (0, 2, 1, 3) if q_call is not None else (0, 3, 1, 2)
                    rope_order = (0, 2, 3, 1) if q_call is not None else (0, 3, 2, 1)
                    for local_head in range(local_heads):
                        q_head_psum = nl.ndarray((_P_MAX, _KV_LORA_TILES, gather_rows), dtype=bf16, buffer=nl.psum)
                        for latent_tile in range(_KV_LORA_TILES):
                            col = local_head * _KV_LORA + latent_tile * _P_MAX
                            nisa.nc_transpose(
                                dst=q_head_psum[0:_P_MAX, latent_tile, 0:gather_rows],
                                data=q_all[0:gather_rows, col : col + _P_MAX],
                            )
                        nisa.tensor_copy(
                            dst=q_nope_all[:, tile_base : tile_base + tile_tokens, :, :].reshape_dim(
                                3, (_TP, local_heads)
                            )[:, :, :, :, local_head],
                            src=q_head_psum.reshape_dim(2, rows_split).permute(nope_order),
                            engine=nisa.engine.scalar if local_head % 2 else nisa.engine.vector,
                        )
                    q_rope_psum_all = nl.ndarray((_ROPE, local_heads, gather_rows), dtype=bf16, buffer=nl.psum)
                    for local_head in range(local_heads):
                        col = absorb_width + local_head * _ROPE
                        nisa.nc_transpose(
                            dst=q_rope_psum_all[0:_ROPE, local_head, 0:gather_rows],
                            data=q_all[0:gather_rows, col : col + _ROPE],
                        )
                    nisa.tensor_copy(
                        dst=q_rope_all[0:_ROPE, tile_base : tile_base + tile_tokens, 0:heads].reshape_dim(
                            2, (_TP, local_heads)
                        ),
                        src=q_rope_psum_all.reshape_dim(2, rows_split).permute(rope_order),
                        **({"engine": nisa.engine.scalar} if q_prep_act else {}),
                    )
                    if mx_q_rows is not None:
                        _mx_query_words(
                            q_all,
                            mx_q_rows,
                            tile_base,
                            tile_tokens,
                            local_heads,
                            rows_split,
                            q_call is not None,
                            sb,
                        )
                    if q_nat_all is not None:
                        _mx_natural_query_tiles(
                            q_all, q_nat_all, tile_base, tile_tokens, local_heads, rows_split, nope_order, q_prep_act
                        )
            q_mx_scores = None
            # ``fp8_pv_refs_after_q``: this pass's MX query loads, which its reference DMAs follow on the SP queue.
            q_mx_pass_loads = ()
            if q_mx and q_mx_pre_pass[0] == cb:
                # ``q_mx_pass_prefetch``: the pass before loaded this pass's bytes into the heap stage tiles; the first
                # copy waits for the state stores of the pass before (their source bytes can be this pass's tiles).
                kernel_assert(cr == _TAIL_ROWS, "q_mx_pass_prefetch loads full passes only")
                q_mx_scores = q_mx_pre[1]
                copies = _q_mx_copies(cr, q_rows, local_heads, q_mx_pre[0], *q_mx_words, q_rope_all)
                if p2_stores:
                    _instruction(copies[0]).depends_on(*p2_stores, kind=nisa.dependency.runtime)
                q_mx_pre_pass[0] = None
            elif q_mx:
                q_mx_scores = sb.tile((heads, cr), fp32)
                slot, block_rows = q_slot(rows[0])
                kernel_assert(block_rows == cr, "q_mx_gather: the gather block of a pass holds the rows of the pass")
                q_mx_pass_loads = _q_mx_load(
                    q_call[1],
                    slot,
                    cr,
                    q_rows,
                    local_heads,
                    *q_mx_words,
                    q_rope_all,
                    q_mx_scores,
                    f"{tag}_c{cb}_mla",
                    sb,
                    query_dma,
                    after=p2_stores,
                    packed=q_mx_packed,
                )
            if q_pair is not None:
                # ``q_gather_buffers`` 2: free the two gather tiles before the loop of the pass.
                sb.close_scope()
            if q_prefetch_tiles and cb + _TAIL_ROWS < total_rows:
                # ``pass_q_prefetch``: the next pass's first gather tiles load now, before this pass's loop.
                prefetch_queries(cb + _TAIL_ROWS)
            # ``dr_keys``: the rope queries (in place) and the own-key scores take the e4m3 query, as the decode's
            # pack_dr_keys does; the e4m3 latent query tiles live only for the own-key scores.
            q_own = q_nope_all
            if e4m3_query:
                sb.open_scope("dr_reference")
                q_own = _e4m3_query_rounding(q_nope_all, q_rope_all, sb)
            row_select = None
            if bitwise and (own_key_select or (packed_prior and pack_options["own_key_select"])):
                # The partition index of each current row, for the one-hot stationaries of ``_own_key``.
                row_select_i = sb.tile((total_rows, heads), nl.int32)
                nisa.iota(dst=row_select_i, pattern=[[0, heads]], offset=0, channel_multiplier=1)
                row_select = sb.tile((total_rows, heads), fp32)
                nisa.tensor_copy(dst=row_select, src=row_select_i)
            # ``own_key_selects``: the one-hot stationaries of the rows of the pass, one slice a row.
            selects = None
            if row_select is not None and packed_prior and pack_options["own_key_selects"]:
                selects = sb.tile((total_rows, cr, heads), nl.bfloat16)
            if q_mx:
                # ``q_mx_gather``: the owners' own-key scores of the pass rows (``_q_mx_store``), one SBUF tile: the
                # references and caps of all rows in one op each.
                softmax.set_reference(q_mx_scores[0:heads, 0:cr], 0, count=cr)
            for cq in range(0 if q_mx else cr):
                r = rows[cq]
                b = r % tokens
                j = r // tokens
                if bitwise:
                    # The own key position[b] + j, with the 2**-98 current weight, is the whole current
                    # contribution; the current rows 0 .. j-1 are prior keys read from the pool.
                    _own_key(
                        cq,
                        r,
                        q_own,
                        q_rope_all,
                        cur_latent_keys,
                        cur_rope_keys,
                        cur_rows,
                        state,
                        softmax,
                        heads,
                        own_weight,
                        f"{tag}_c{cb}",
                        sb,
                        tied_sum=not lay.one_head,
                        row_select=row_select,
                        **({"select_dst": selects[0:total_rows, cq, 0:heads]} if selects is not None else {}),
                        **({"own_row": own_rows[cq]} if own_rows is not None else {}),
                    )
                else:
                    _current_block(
                        cq,
                        r,
                        b,
                        j,
                        q_nope_all,
                        q_rope_all,
                        cur_latent_keys,
                        cur_rope_keys,
                        cur_rows,
                        state,
                        softmax,
                        heads,
                        q_rows,
                        causal,
                        lay.rank_scale,
                        sb,
                    )
            if e4m3_query:
                sb.close_scope()
            if natural:
                # The feature-major latent query tiles end with the own-key scores.
                sb.close_scope()
            if p2:
                # The own-key state of each column, from the current row of its row (the numerators of ``_own_key``);
                # with ``own_state_first`` it is traced before the query loads.
                if not own_first:
                    _own_key_state_p2(
                        state_flat,
                        rows,
                        cur_rows,
                        q_rows,
                        own_weight,
                        f"{tag}_c{cb}",
                        sb,
                        after=p2_stores,
                        row_after=p2_stores,
                        **(
                            {
                                "own_rows": rows_ahead["own_rows"] if rows_ahead.get("cb") == cb else None,
                                "batched": True,
                            }
                            if rows_batched_on
                            else {}
                        ),
                        **({"stack": True} if pack_options["own_state_stack"] else {}),
                        **({"stack_rows": own_rows_pre[cb]} if cb in own_rows_pre else {}),
                    )
                if rows_ahead.get("cb") == cb:
                    # ``pass_rows_ahead``: the gather tiles and the own-key rows are read; free them before the loop.
                    sb.release(rows_ahead["tiles"])
                    rows_ahead.clear()
            elif loop_p2:
                # ``loop_p2``: the own-key state of each (row, head) column, in the form of the head-major state.
                weight = lay.rank_scale if exact_softmax else lay.rank_current_weight
                if row_select is not None:
                    _own_key_state_select_p2(state_flat, rows, cur_rows, q_rows, weight, row_select, f"{tag}_c{cb}", sb)
                else:
                    _own_key_state_p2(state_flat, rows, cur_rows, q_rows, weight, f"{tag}_c{cb}", sb)
            if o_ahead_chunks is not None:
                kernel_assert(pass_q_loads, "o_proj_ahead needs the pass to load its queries from the call gather")
                load_o_proj_ahead(pass_q_loads)
            # ``state_sums_pre_if``: the pass that stores the call's sum blocks traces them before its capless check.
            sums_pre_if = None
            if (
                packed_prior
                and pack_options["state_sums_pre_if"]
                and (state_call or sum_blocks_pass)
                and p2
                and state_bf16_o
                and sum_block_tile is not None
                and cb + _TAIL_ROWS >= total_rows
            ):
                sums_pre_if = {
                    "out": None,
                    "trace": (
                        # ``state_sum_blocks_pass``: the arguments of the single-pass p2 store below.
                        lambda suffix, cb=cb, cr=cr, state_flat=state_flat: _store_sum_blocks_pre(
                            state_flat,
                            state_pair[0],
                            sum_pair[0],
                            list(range(cr)),
                            None,
                            1,
                            q_rows,
                            f"{tag}_t{cb}_mla_state_store{suffix}",
                            {"tile": sum_block_tile, "pass": 0, "passes": 1, "store": True},
                            sums_engine=nisa.engine.scalar if pack_options["state_sums_act"] else None,
                        )
                    )
                    if sum_blocks_pass
                    else lambda suffix, cb=cb, rows=rows, state_flat=state_flat: _store_sum_blocks_pre(
                        state_flat,
                        state_pair[0],
                        sum_pair[0],
                        rows,
                        tr,
                        passes,
                        q_rows,
                        f"{tag}_t{cb}{suffix}",
                        {"tile": sum_block_tile, "pass": cb // _TAIL_ROWS, "passes": passes, "store": True},
                        dma_options=query_dma,
                        sums_engine=nisa.engine.scalar if pack_options["state_sums_act"] else None,
                    ),
                }
            if packed_prior:
                # ``fp8_pv_tables_ahead``: a call of two full passes (with the FP8 P@V step stores of the step); the first
                # pass loads the second's chunk tables and first pages, which the second takes.
                tables_next = None
                if (
                    pack_options["fp8_pv_tables_ahead"]
                    and cb == 0
                    and total_rows == 2 * _TAIL_ROWS
                    and not query_rows
                    and not (split_v1 or split_v5)
                    and pass_plans is not None
                    and "items" in pass_plans
                ):
                    tables_next = pass_plans["items"][1]
                tables_in = tables_ahead.pop(cb, None)
                ahead_out = _attend_prior_packed(
                    rows,
                    kv_pool,
                    block_table,
                    position,
                    rank_info,
                    q_nope_all if q_nat_all is None else q_nat_all,
                    q_rope_all,
                    slot_offset,
                    state_flat,
                    softmax,
                    tokens,
                    q_rows,
                    local_blocks,
                    f"{tag}_c{cb}",
                    sb,
                    cache_writes,
                    pack_options,
                    clamp_pages=not zeroed_pages,
                    dr_rows=dr_rows,
                    live_rows=live_rows,
                    mx_rows=mx_q_rows,
                    split_info=split["info"] if split_v1 else None,
                    split_lo=_split_v5_lo(split) if split_v5 else None,
                    **({"query_words": q_mx_words} if q_mx else {}),
                    # q_mx_gather: the pass frame drops its BF16 query tiles, so the loop's page tiles can take the
                    # freed bytes of the pass before's store sources; their first loads wait for those stores.
                    **({"loop_after": p2_stores} if q_mx else {}),
                    hoisted=None
                    if pass_plans is None
                    else {
                        **pass_plans,
                        "plan": pass_plans["plans"][cb // _TAIL_ROWS] if "plans" in pass_plans else None,
                        **({"step": pass_plans["items"][cb // _TAIL_ROWS]} if "items" in pass_plans else {}),
                    },
                    # The capless redo's state of the pass start: the own-key state again (current rows from HBM: the
                    # bits of every own-key form), with the pass's own-key weight (0 on a split pass). None (pack
                    # option no_if) gives the capped loop only. With no_if_loops the capless loop runs again, and
                    # its redo is a loop of 0 or 1 trips (no IF).
                    state_reinit=(
                        lambda after=(), state_flat=state_flat, rows=rows, cb=cb, weight=own_weight: _own_key_state_p2(
                            state_flat,
                            rows,
                            cur_rows,
                            q_rows,
                            weight,
                            f"{tag}_c{cb}r",
                            sb,
                            after=after,
                        )
                    )
                    if p2 and (not pack_options["no_if"] or pack_options["no_if_loops"])
                    else None,
                    setup_after=p2_stores if pack_options.get("fp8_pv_stage_free") else (),
                    next_step=tables_next,
                    tables_in=tables_in,
                    refs_sched_after=q_mx_pass_loads if pack_options["fp8_pv_refs_after_q"] else (),
                    **({"sums_pre_if": sums_pre_if} if sums_pre_if is not None else {}),
                )
                if ahead_out is not None:
                    tables_ahead[cb + _TAIL_ROWS] = ahead_out
                if o16_heap:
                    # state_o_heap: the pass before's numerator tile returns to the region after this pass's loops
                    # (release_heap fails at trace time unless it is the newest heap tile).
                    sb.release(o16_heap)
                    o16_heap.clear()
                if tables_in is not None:
                    # ``fp8_pv_tables_ahead``: the tiles that the pass before loaded for this pass, after its loops (after
                    # the ``state_o_heap`` tile: newest first).
                    sb.release(tables_in["heap"])
            else:
                _attend_prior(
                    rows,
                    kv_pool,
                    block_table,
                    q_nope_all,
                    q_rope_all,
                    row_threshold if bitwise else mask_threshold,
                    lower_threshold,
                    slot_offset,
                    loop_row,
                    state,
                    softmax,
                    tokens,
                    q_rows,
                    local_blocks,
                    heads,
                    use_loop,
                    bitwise,
                    f"{tag}_c{cb}",
                    sb,
                    lay,
                    n_mask,
                    ts,
                    loop_group_cap=loop_group_cap,
                    clamp_pages=not zeroed_pages,
                    weights_on_act=late_weights or weights_on_act,
                    **({"row_pipeline": True} if loop_row_pipeline else {}),
                    **({"first_static": True, "first_static_after": q_load} if loop_first_static else {}),
                    **({"scores_ahead": True} if scores_ahead else {}),
                    **({"folded_reference": True} if folded_reference else {}),
                    rows_v_stationary=rows_v_stationary,
                    dr_rows=dr_rows,
                    rows_prefetch=rows_prefetch,
                    key_path=_KeyPath(
                        ("mx_" + rows_mx_keys)
                        if rows_mx_keys
                        else (("dr_inplace" if rows_dr_inplace else "dr") if rows_dr else "bf16"),
                        nisa.engine.scalar if rows_dr_act_copies else nisa.engine.vector,
                        mx_scale=mx_scale,
                    ),
                    pair_lag=rows_pair_lag,
                    pads=pads,
                    seq_hoist=rows_seq_hoist,
                    setup_ahead=rows_setup_ahead,
                    mx_rows=mx_q_rows,
                    p2_state=(
                        (
                            state_flat.reshape((_P_MAX, pack_p2_groups(q_rows) * (cr // q_rows) + 1, _STATE)),
                            cr // q_rows,
                        )
                        if loop_p2
                        else None
                    ),
                    preload=loop_preload,
                    **({"pair_tail": True} if pair_tail else {}),
                    **({"unroll": 2} if unroll == 2 else {}),
                    **({"bf16_pages": True} if bf16_pages else {}),
                    **({"page_gather": loop_page_gather} if loop_page_gather != "off" else {}),
                    **({"tail_groups": loop_tail} if loop_tail else {}),
                    **({"stack_rows": True} if stack_rows else {}),
                    **({"prior_keys": hoisted_keys} if hoisted_keys is not None else {}),
                )
            # The rows of the pass (the packed state has one more column, for its null items; the state of
            # ``STACK_STATE_UPPER`` has the shadow rows of the stacked units on its upper partitions).
            pass_state = (
                None
                if (p2 or loop_p2)
                else (state[0:heads, 0:tr, 0 : _KV_LORA + 1] if (packed_prior or stack_upper) else state)
            )
            if exact_softmax:
                softmax.merge_scale(state, tp_group, f"{tag}_c{cb}")
            next_cb = cb + _TAIL_ROWS
            if rows_ahead_on and next_cb < total_rows and not (query_rows and next_cb >= query_rows):
                # ``pass_rows_ahead``: the next pass's loads go on their queues before this pass's state stores.
                issue_rows_ahead(next_cb)
            if (
                q_mx_pre is not None
                and total_rows - next_cb >= _TAIL_ROWS
                and not (query_rows and next_cb >= query_rows)
            ):
                # ``q_mx_pass_prefetch``: the next pass's MX query bytes go on the query queue now, before this pass's
                # state stores, into the heap stage tiles (no byte of a store source).
                next_slot, next_block = q_slot(pass_rows[next_cb // _TAIL_ROWS][0])
                kernel_assert(
                    next_block == _TAIL_ROWS, "q_mx_gather: the gather block of a pass holds the rows of the pass"
                )
                _q_mx_loads(
                    q_call[1],
                    next_slot,
                    _TAIL_ROWS,
                    local_heads,
                    q_mx_pre[0],
                    q_mx_pre[1],
                    f"{tag}_c{next_cb}_mla",
                    # The SP hardware DGE queue when the call gives no query queue (critical_dma_priority off).
                    query_dma or {"dge_mode": hwdge, "engine": sync},
                    packed=q_mx_packed,
                )
                q_mx_pre_pass[0] = next_cb
            if state_call:
                # The tail of every pass runs after the one state reduce-scatter of the call.
                if p2 and state_bf16_o:
                    # The BF16-o state of the call: the numerators cast to BF16 and the FP32 sums, at the tensor rows
                    # and columns of the p2-off BF16-o store, so the one BF16-o reduce-scatter reads its own form.
                    follows = next_cb < total_rows and not (query_rows and next_cb >= query_rows)
                    heap_out = {}
                    p2_stores = _store_state_p2_bf16_o(
                        state_flat,
                        state_pair[0],
                        sum_pair[0],
                        rows,
                        tr,
                        passes,
                        q_rows,
                        sb,
                        f"{tag}_t{cb}",
                        dma_options=query_dma,
                        sums_first=sums_rs_first,
                        sums_engine=nisa.engine.scalar if pack_options["state_sums_act"] else None,
                        # state_o_act: a pass that another pass follows stores its numerators on the ScalarE queue.
                        **({"o_engine": nisa.engine.scalar} if pack_options["state_o_act"] and follows else {}),
                        # state_o_heap: its numerator tile on the heap, out of the next pass's stack bytes.
                        **({"o_heap": True, "out": heap_out} if pack_options["state_o_heap"] and follows else {}),
                        # state_sums_heap: the sum stores read a heap copy, so the next pass waits for the copy.
                        **(
                            {"sums_heap": True}
                            if pack_options["state_sums_heap"]
                            and pack_options["state_o_heap"]
                            and follows
                            and sum_block_tile is None
                            else {}
                        ),
                        # state_sum_blocks: the sums go to the call's blocks; the last pass stores them.
                        **(
                            {
                                "sum_blocks": {
                                    "tile": sum_block_tile,
                                    "pass": cb // _TAIL_ROWS,
                                    "passes": passes,
                                    "store": next_cb >= total_rows,
                                }
                            }
                            if sum_block_tile is not None
                            else {}
                        ),
                        # state_sums_pre_if: the sum blocks were traced before the capless check.
                        **(
                            {"sums_pre": sums_pre_if["out"]}
                            if sums_pre_if is not None and sums_pre_if["out"] is not None
                            else {}
                        ),
                    )
                    if heap_out:
                        # The next pass waits only for the cast and the sum stores (its stack does not hold the
                        # numerator tile); the tile returns to the region after its loops. With state_sums_heap it
                        # waits for the cast and the sums copy, and the sums tile returns with the numerator tile.
                        o16_heap.append(heap_out["o16"])
                        if "s32" in heap_out:
                            o16_heap.append(heap_out["s32"])
                        p2_stores = tuple(heap_out["next_waits"])
                elif p2:
                    # _store_state_p2 already takes the SP hardware DGE queue: the query level adds its priority.
                    query_priority = crit_dma if query_dma else {}
                    p2_stores = _store_state_p2(
                        state_flat, state_pair[0], rows, tr, passes, q_rows, f"{tag}_t{cb}", dma_options=query_priority
                    )
                elif state_bf16_o:
                    _store_call_state_bf16_o(
                        pass_state,
                        state_pair[0],
                        sum_pair[0],
                        rows if seq_major else None,
                        tr,
                        passes,
                        cb // _TAIL_ROWS,
                        sb,
                        f"{tag}_t{cb}",
                        dma_options=query_dma,
                    )
                elif seq_major:
                    _store_rows_state(state, state_pair[0], rows, tr, passes, f"{tag}_t{cb}", dma_options=query_dma)
                else:
                    _store_pass_state(
                        pass_state, state_pair[0], cb // _TAIL_ROWS, passes, f"{tag}_t{cb}", dma_options=query_dma
                    )
                if p2 and rows_ahead.get("cb") == next_cb:
                    # ``pass_rows_ahead``: the state stores issue after the next pass's loads (schedule edges, no wait).
                    for store in p2_stores:
                        store.depends_on(rows_ahead["last"], kind=nisa.dependency.schedule)
                return
            if late_load and not late_tail_frames and o_ahead_chunks is not None:
                # ``o_proj_ahead``: the gate weights load here; the o_proj chunks loaded before the loop.
                gate_chunks, _ = load_tail_weights(f"_p{cb}", o_proj=False)
                o_w_chunks = o_ahead_chunks
            elif late_load and not late_tail_frames:
                gate_chunks, o_w_chunks = load_tail_weights(f"_p{cb}", o_proj=not late_o_proj_weights)
            # Steps 18-21: the output tail for this chunk, on ``tr`` rows. Reduce-scatter the state,
            # normalize, project the value and the gate, run o_proj, and TP-sum the hT partial. Chaining
            # each chunk's first collective after the previous chunk's last one keeps a later chunk from
            # reusing an earlier chunk's collective buffers while a slower rank still reads or writes them.
            rs_pass = pass_state
            if bf16_state:
                rs_pass = sb.tile(state.shape, bf16)
                nisa.tensor_copy(dst=rs_pass, src=state)
            if draft_seq and not output_gate:
                # ``seq_passes``: only the output gate reads the pass inputs, and the drafter has none.
                x_pass = None
            elif seq_rows or draft_seq:
                # A sequence-major pass takes the layer inputs of its rows, run by run.
                x_pass = sb.tile((_P_MAX, tr, _HIDDEN_TILES), bf16)
                if cr < tr:
                    nisa.memset(dst=x_pass[0:_P_MAX, cr:tr, 0:_HIDDEN_TILES], value=0.0)
                for dst_row, src_row, n in _row_runs(rows):
                    nisa.tensor_copy(
                        dst=x_pass[0:_P_MAX, dst_row : dst_row + n, 0:_HIDDEN_TILES],
                        src=x_hT[0:_P_MAX, src_row : src_row + n, 0:_HIDDEN_TILES],
                    )
            elif cr < tr:
                # Pad x over the padding rows (x_hT is read-only here, so a view would also be fine).
                x_pass = sb.tile((_P_MAX, tr, _HIDDEN_TILES), bf16)
                nisa.memset(dst=x_pass[0:_P_MAX, cr:tr, 0:_HIDDEN_TILES], value=0.0)
                nisa.tensor_copy(
                    dst=x_pass[0:_P_MAX, 0:cr, 0:_HIDDEN_TILES], src=x_hT[0:_P_MAX, cb : cb + cr, 0:_HIDDEN_TILES]
                )
            else:
                x_pass = x_hT[0:_P_MAX, cb : cb + tr, 0:_HIDDEN_TILES]
            # The output all-gather writes a fresh y_pass and the engine copy-back below writes y_hT, so y_hT
            # is engine-written only. A y_hT view as the all-gather destination ends on a byte that is not
            # 16-byte aligned (``cb * _HIDDEN_TILES`` is not a multiple of 8 bf16 elements for every chunk),
            # so the next chunk's copy-back into y_hT starts in that same 16-byte ECC block with no ordering
            # edge, and the engine read-modify-write could lose the collective's bytes. Keeping y_hT off every
            # collective removes that pair (both copy-backs are engine writes, which the ECC race cannot lose).
            if tail_overlap and ag_call is None:
                # ``pass_tail_overlap``: the output all-gather of each pass lands in its own persistent tile, which no
                # other tile shares, since the next pass's tail tiles live in its window.
                y_pass = sources["y_dst"][cb // _TAIL_ROWS]
            else:
                y_pass = None if ag_call is not None else sb.tile((_P_MAX, tr, _HIDDEN_TILES), bf16)
            pass_cols = (cb // _TAIL_ROWS) * tr * _HIDDEN_TILES
            ag_buf = (
                ag_call[0 : _P_MAX // cp, pass_cols : pass_cols + tr * _HIDDEN_TILES]
                if ag_call is not None
                else (None if ag_bufs is None else ag_bufs[(cb // _TAIL_ROWS) % 2])
            )
            # ``p2``: the state store of the tail puts each (row, head) column at its head and row of the HBM state.
            p2_store = None
            if loop_p2:
                kernel_assert(not state_bf16_o, "loop_p2 does not take state_bf16_o")
                p2_store = lambda tag_s, dma_options, state_flat=state_flat, cr=cr: _store_state_loop_p2(  # noqa: E731
                    state_flat, state_pair[0], list(range(cr)), tr, q_rows, heads, tag_s, sb, dma_options
                )
            elif p2:
                p2_store = lambda tag_s, dma_options, state_flat=state_flat, cr=cr: _store_state_p2(  # noqa: E731
                    state_flat, state_pair[0], list(range(cr)), tr, 1, q_rows, tag_s, dma_options
                )
                if state_bf16_o:
                    # ``p2`` with the BF16-o state of a pass: the numerators cast to BF16 and the FP32 sums of each
                    # (row, head) column, at the head and row of the pass's BF16-o store (one pass of the call pair).
                    p2_store = lambda tag_s, sums, view_rows, state_flat=state_flat, cr=cr: (  # noqa: E731
                        _store_state_p2_bf16_o(
                            state_flat,
                            state_pair[0],
                            sums,
                            list(range(cr)),
                            view_rows,
                            1,
                            q_rows,
                            sb,
                            tag_s,
                            sums_first=sums_rs_first,
                            sums_engine=nisa.engine.scalar if pack_options["state_sums_act"] else None,
                            # state_sum_blocks_pass: the sums of the one pass in the call's blocks, stored here
                            # (or before the capless check with state_sums_pre_if).
                            **(
                                {
                                    "sum_blocks": {"tile": sum_block_tile, "pass": 0, "passes": 1, "store": True},
                                    **(
                                        {"sums_pre": sums_pre_if["out"]}
                                        if sums_pre_if is not None and sums_pre_if["out"] is not None
                                        else {}
                                    ),
                                }
                                if sum_blocks_pass
                                else {}
                            ),
                        )
                    )
            gate_psum = None
            if late_tail_frames:
                # The gate weights die with this frame, so the tiles of the tail below take their bytes.
                with sb.scope("tail_gate"):
                    gate_chunks, o_w_chunks = load_tail_weights(f"_p{cb}", o_proj=False)
                    if output_gate:
                        gate_psum = _gate_projection(x_pass, gate_chunks, tr, local_heads)
            pass_index = cb // _TAIL_ROWS
            pass_pair = state_pair_odd if (tail_overlap and pass_index % 2) else state_pair
            # ``pass_tail_overlap``: the pass before computes its tail now, after this pass's attention and before this
            # pass's state reduce-scatter, which waits on its partial copies (its output reduce-scatter source).
            prev_copies, prev_cc = (), None
            if tail_overlap and overlap["pending"] is not None:
                prev_copies, prev_cc = overlap["pending"]()
                overlap["pending"] = None
            tail = _output_tail(
                rs_pass,
                x_pass,
                gate_chunks,
                o_w_chunks,
                kv_b_w,
                y_pass,
                tr,
                local_heads,
                absorb_width,
                output_gate,
                tp_group,
                overlap["ags"].get(pass_index - 2) if tail_overlap else prev_tail,
                ag_buf,
                pass_pair[0],
                pass_pair[1],
                f"{tag}_t{cb}",
                sb,
                cp,
                q_gather_hbm if lay.one_head else out_ar_pass(cb // _TAIL_ROWS),
                rs_buf=rs_bufs[(cb // _TAIL_ROWS) % 2] if rs_bufs else None,
                gather=ag_call is None,
                dense_state=late_weights
                or late_tail_frames
                or (
                    stack_rows
                    and STACK_DENSE_TAIL
                    and pass_pair[0] is not None
                    and not bf16_state
                    and not state_bf16_o
                    and p2_store is None
                )
                or dense_tail_pass,
                partial_out=None if rs_call is None else rs_call[0:_P_MAX, cb : cb + tr, 0:_HIDDEN_TILES],
                crit_dma=crit_dma,
                state_store=p2_store,
                state_rows=heads,
                gate_psum=gate_psum,
                state_bf16_o=state_bf16_o,
                state_sums=sum_pair,
                # state_sum_blocks_pass: each head's rows from the reduced sum blocks of the one pass.
                **(
                    {
                        "state_sum_aps": lambda sum_dst, dense: _sum_block_loads(
                            sum_dst, pass_rows, 0, tr, 1, q_rows, head_major=not dense
                        )
                    }
                    if sum_blocks_pass
                    else {}
                ),
                rs_hook=None if state_call or cb + _TAIL_ROWS < total_rows else state_rs_hook,
                sums_first=sums_rs_first,
                defer=tail_overlap,
                state_rs_after=tuple(prev_copies),
                state_store_split=state_store_split,
            )

            def copy_back(y_pass=y_pass, rows=rows, cb=cb, cr=cr):
                if y_pass is not None and (seq_rows or draft_seq):
                    for dst_row, src_row, n in _row_runs(rows):
                        nisa.tensor_copy(
                            dst=y_hT[0:_P_MAX, src_row : src_row + n, 0:_HIDDEN_TILES],
                            src=y_pass[0:_P_MAX, dst_row : dst_row + n, 0:_HIDDEN_TILES],
                        )
                elif y_pass is not None:
                    nisa.tensor_copy(
                        dst=y_hT[0:_P_MAX, cb : cb + cr, 0:_HIDDEN_TILES], src=y_pass[0:_P_MAX, 0:cr, 0:_HIDDEN_TILES]
                    )

            if not tail_overlap:
                prev_tail = tail
                copy_back()
                return
            compute, _state_rs = tail
            if prev_cc is not None:
                # This pass's state store and reduce-scatter are issued: the output collectives of the pass before.
                prev_cc()

            def pending(compute=compute, copy_back=copy_back, pass_index=pass_index):
                # The partial copies of this pass follow the final collective of the pass before (a runtime wait), so
                # they are in none of that pass's output collective windows.
                copies, collectives = compute(() if overlap["last"] is None else (overlap["last"],))

                def cc():
                    final = collectives(after=overlap["last"])
                    overlap["ags"][pass_index] = final
                    overlap["last"] = final
                    copy_back()

                return copies, cc

            overlap["pending"] = pending

    split_guards = split_ext and split.get("guards", False)
    foreign_cbs = []
    for cb in range(0, total_rows, _TAIL_ROWS):
        if split_guards and cb // _TAIL_ROWS >= split["ext"] // (tr // q_rows):
            foreign_cbs.append(cb)
            continue
        _attend_pass(cb)
    if tail_overlap:
        _run_pending_tail()
        prev_tail = overlap["last"]
    # "one": the foreign passes attend only on a step with an active piece of this replica (CC-free: the q gather and
    # the state reduce-scatter of the call stay outside). Inside that IF, each later foreign pass takes its own IF on the
    # active pieces of its slots: the planner takes the low slot of each parity first, so a helper with one piece runs
    # one foreign pass. A pass waits on the stores of the pass before it, so the later IFs nest in the first.

    def foreign_passes():
        _attend_pass(foreign_cbs[0])
        for cb in foreign_cbs[1:]:
            first_slot = cb // _TAIL_ROWS * (tr // q_rows) - split["ext"]
            pass_flag = _split_v5_pass_mine(split, first_slot, tr // q_rows, f"{tag}_c{cb}", sb)
            _split_v5_if(pass_flag, lambda cb=cb: _attend_pass(cb), f"{tag}_mla_split_g2_{cb}", sb)

    if foreign_cbs:
        _split_v5_if(split["checks"]["any_mine"], foreign_passes, f"{tag}_mla_split_g2", sb)
    kernel_assert(not o16_heap, "state_o_heap: the numerator heap tile of a pass was not released after the next pass")
    kernel_assert(not tables_ahead, "fp8_pv_tables_ahead: the tiles loaded for a pass were not taken by that pass")
    if state_call and state_bf16_o:
        prev_tail, call_views, sums_rs_call = _state_rs_bf16_o_call(
            state_pair, tp_group, tag, sum_pair, sums_first=sums_rs_first
        )
    elif state_call:
        prev_tail = ncc.reduce_scatter(
            srcs=[state_pair[0]],
            dsts=[state_pair[1]],
            replica_group=tp_group,
            collective_dim=0,
            op=nl.add,
            name=f"{tag}_state_rs_call",
        )
    if state_call:
        _run_hook(state_rs_hook, sb, "state_rs_hook")
    split_merge = None
    exp_dst = None
    if state_call and split_ext:
        kernel_assert(state_bf16_o, "split (v5) ext exports the BF16-o state of the foreign pass")
        # The export, the merge and the tails read the sequence-major passes: a sequence-major copy of the call state.
        seq_views = _split_v5_seq_state(call_views, passes, tokens, tr // q_rows, local_heads, tag)
        exp_dst = _split_v5_export_ext(split, seq_views, passes, local_heads, tag)
    elif state_call and split_v5:
        _split_v5_export(split, q_rows, local_heads, tag, sb)
    if state_call and split_v1:
        # The long-sequence split: each rank stores its donor-row contribution (the helper's foreign states), one pair
        # all-reduce, then the owner adds it in the tails below.
        merge_src = nl.ndarray(state_pair[1].shape, dtype=fp32, buffer=nl.hbm, name=f"{tag}_mla_split_merge_src")
        merge_dst = nl.ndarray(state_pair[1].shape, dtype=fp32, buffer=nl.hbm, name=f"{tag}_mla_split_merge_dst")
        for cb in range(0, total_rows, _TAIL_ROWS):
            with sb.scope("split_contrib"):
                p = cb // _TAIL_ROWS
                if state_bf16_o:
                    o_dst, sum_dst, view_rows = call_views
                    pass_state = load_state_bf16_o(
                        o_dst,
                        sum_dst,
                        view_rows,
                        lambda h, p=p: h * passes + p,
                        local_heads,
                        tr,
                        _KV_LORA,
                        True,
                        sb,
                        f"{tag}_sc{cb}_mla_state",
                    )
                else:
                    pass_state = _load_pass_state(state_pair[1], p, passes, tr, local_heads, sb, f"{tag}_sc{cb}")
                _split_contribution(pass_state, merge_src, p, passes, tr, local_heads, tokens, f"{tag}_sc{cb}", sb)
        split_merge = ncc.all_reduce(
            srcs=[merge_src],
            dsts=[merge_dst],
            replica_group=split["group"],
            op=nl.add,
            name=f"{tag}_mla_split_merge_ar",
        )
    if state_call:
        # No prior-key buffer is live any more, so the tail weights load once for all passes. With
        # ``tail_weights_after_rs`` they wait for the numerator reduce-scatter and load on the ScalarE queue.
        if tail_weights_after_rs:
            # ``GATE_WEIGHTS_AFTER_SUMS``: with the sums reduce-scatter first, the gate weights wait for it only.
            early = (
                (SUM_RS_FIRST if sums_rs_first is None else sums_rs_first) and GATE_WEIGHTS_AFTER_SUMS and state_bf16_o
            )
            gate_after = (sums_rs_call,) if early else None
            gate_chunks, o_w_chunks = load_tail_weights(
                "_call", engine=nisa.engine.scalar, after=(prev_tail,), gate_after=gate_after
            )
        else:
            gate_chunks, o_w_chunks = load_tail_weights("_call")
        # ``tail_o_proj_first``: every gate exp of the tail follows the last o_proj weight DMA of the call on ScalarE.
        o_proj_first = packed_prior and pack_options["tail_o_proj_first"] and tail_weights_after_rs
        act_after = _instruction(o_proj_dmas[-1]) if o_proj_first and o_proj_dmas else None
        # ``GATE_BEFORE_STATE``: the gate of each full tail pass traced before any state load (bitwise the same).
        early_gates = {}
        if GATE_BEFORE_STATE and output_gate:
            for cb in range(0, total_rows, _TAIL_ROWS):
                if min(_TAIL_ROWS, total_rows - cb) == tr:
                    early_gates[cb] = _gate_projection(
                        x_hT[0:_P_MAX, cb : cb + tr, 0:_HIDDEN_TILES], gate_chunks, tr, local_heads
                    )
            # Every pass's value projection waits for the last gate matmul of the call, so all gate matmuls run first.
            last_gate = getattr(early_gates[max(early_gates)], "last_matmul", None) if early_gates else None
            for early_psum in early_gates.values():
                object.__setattr__(early_psum, "last_matmul", last_gate)
        for cb in range(0, total_rows, _TAIL_ROWS):
            cr = min(_TAIL_ROWS, total_rows - cb)
            if split_ext and cb // _TAIL_ROWS >= split["ext"] // (tr // q_rows):
                # The foreign passes have no output: their reduced state went to the export (above); no tail.
                continue
            with sb.scope("tail_pass"):
                if split_ext:
                    # The split program's tails run the sequence-major request passes (the merge adds by slot): the
                    # rows of pass p, their inputs, their state rows, and their partial rows of the request-row output.
                    _split_v5_seq_tail(
                        split,
                        exp_dst,
                        seq_views,
                        x_hT,
                        rs_call,
                        pass_rows[cb // _TAIL_ROWS],
                        cb // _TAIL_ROWS,
                        passes,
                        tokens,
                        q_rows,
                        tr,
                        local_heads,
                        lambda local_state, x_pass, partial: _tail_partial(
                            local_state,
                            x_pass,
                            gate_chunks,
                            o_w_chunks,
                            kv_b_w,
                            tr,
                            local_heads,
                            absorb_width,
                            output_gate,
                            sb,
                            dense_state=True,
                            partial=partial,
                            act_after=act_after,
                        ),
                        f"{tag}_t{cb}",
                        sb,
                    )
                    continue
                if cr < tr:
                    x_pass = sb.tile((_P_MAX, tr, _HIDDEN_TILES), bf16)
                    nisa.memset(dst=x_pass[0:_P_MAX, cr:tr, 0:_HIDDEN_TILES], value=0.0)
                    nisa.tensor_copy(
                        dst=x_pass[0:_P_MAX, 0:cr, 0:_HIDDEN_TILES], src=x_hT[0:_P_MAX, cb : cb + cr, 0:_HIDDEN_TILES]
                    )
                else:
                    x_pass = x_hT[0:_P_MAX, cb : cb + tr, 0:_HIDDEN_TILES]
                if state_bf16_o:
                    o_dst, sum_dst, view_rows = call_views
                    p = cb // _TAIL_ROWS
                    local_state = load_state_bf16_o(
                        o_dst,
                        sum_dst,
                        view_rows,
                        lambda h, p=p: h * passes + p,
                        local_heads,
                        tr,
                        _KV_LORA,
                        True,
                        sb,
                        f"{tag}_t{cb}_mla_state",
                        dma=query_dma,
                        # state_sum_blocks: each head's rows from the reduced sum blocks.
                        **(
                            {"sum_aps": _sum_block_loads(sum_dst, pass_rows, p, tr, passes, q_rows)}
                            if sum_block_tile is not None
                            else {}
                        ),
                    )
                else:
                    local_state = _load_pass_state(
                        state_pair[1], cb // _TAIL_ROWS, passes, tr, local_heads, sb, f"{tag}_t{cb}", query_dma
                    )
                if split_merge is not None:
                    local_state = _split_merge(
                        local_state,
                        merge_dst,
                        cb // _TAIL_ROWS,
                        passes,
                        tr,
                        local_heads,
                        tokens,
                        split["info"],
                        f"{tag}_sm{cb}",
                        sb,
                    )
                _tail_partial(
                    local_state,
                    x_pass,
                    gate_chunks,
                    o_w_chunks,
                    kv_b_w,
                    tr,
                    local_heads,
                    absorb_width,
                    output_gate,
                    sb,
                    dense_state=True,
                    partial=rs_call[0:_P_MAX, cb : cb + tr, 0:_HIDDEN_TILES],
                    gate_psum=early_gates.get(cb),
                    early_gate=cb in early_gates,
                    act_after=act_after,
                )
    if out_rs_tp:
        # attn_rs_moe_input (reading b): token reduce-scatter the pre-reduce partial so this rank keeps its one token,
        # in place of the output reduce-scatter (partition) + all-gather that rebuild the full reduced rows.
        kernel_assert(rs_call is not None, "mla out_rs_tp needs the ag_call / rs_call output path")
        kernel_assert(not out_ar_call, "mla out_rs_tp replaces the output collectives: not with out_ar_call")
        kept = token_reduce_scatter(rs_call[0:_P_MAX, 0:total_rows, 0:_HIDDEN_TILES], tp_group, tag, out_rs_tp, sb)
        object.__setattr__(kept, "side_effects", tuple(cache_writes))
        object.__setattr__(
            kept, "side_effect_sources", (write_rows, write_row) if batched_cache_write else (write_rows,)
        )
        return kept
    if rs_call is not None and out_ar_call:
        # One all-reduce of every pass's partial through the HBM pair, then one load of the Q*B rows into y_hT (the split
        # program: the request rows only).
        cols = out_rows * _HIDDEN_TILES if split_ext else passes * tr * _HIDDEN_TILES
        src, dst = (out_ar_pair[key] for key in ("out_ar_src_hbm", "out_ar_dst_hbm"))
        kernel_assert(src.shape[0] * src.shape[1] == _P_MAX * cols, "out_ar_call: the HBM pair must hold every pass")
        src, dst = src.reshape((_P_MAX, cols)), dst.reshape((_P_MAX, cols))
        store = nisa.dma_copy(dst=src, src=rs_call.reshape((_P_MAX, cols)), name=f"{tag}_out_ar_call_store")
        out_ar = release_weight_gate(
            ncc.all_reduce(srcs=[src], dsts=[dst], replica_group=tp_group, op=nl.add, name=f"{tag}_out_ar_call")
        )
        out_ar.depends_on(prev_tail)
        if out_ar_hook is not None:
            # ``out_ar_hook``: work in the flight of the all-reduce, after the store of the tail's partials.
            with sb.scope("out_ar_hook"):
                out_ar_hook(sb, (_instruction(store),))
        nisa.dma_copy(
            dst=y_hT.reshape((_P_MAX, out_rows * _HIDDEN_TILES)),
            src=dst[0:_P_MAX, 0 : out_rows * _HIDDEN_TILES],
            name=f"{tag}_out_ar_call_load",
        )
    elif rs_call is not None:
        kernel_assert(ag_call is not None, "rs_call needs the ag_call output all-gather")
        cols = -(-total_rows // _TAIL_ROWS) * tr * _HIDDEN_TILES
        out_rs = release_weight_gate(
            ncc.reduce_scatter(
                srcs=[rs_call.reshape((_P_MAX, cols))],
                dsts=[ag_call[0 : _P_MAX // cp, 0:cols]],
                replica_group=tp_group,
                collective_dim=0,
                op=nl.add,
                name=f"{tag}_out_rs_call",
            )
        )
        out_rs.depends_on(prev_tail)
    out_ag_pair = sources if sources is not None and "out_ag_src_hbm" in sources else arena_out_ag
    kernel_assert(
        not (draft_seq and ag_call is not None and out_ag_pair is not None),
        "seq_passes with out_ag_call gathers into SBUF (no HBM output all-gather pair)",
    )
    if ag_call is not None and out_ag_pair is not None:
        # One all-gather of every pass's slice through persistent HBM (no SBUF gather destination),
        # then one load of the Q*B rows into y_hT.
        cols = total_rows * _HIDDEN_TILES
        out_src, out_dst = out_ag_pair["out_ag_src_hbm"], out_ag_pair["out_ag_dst_hbm"]
        nisa.dma_copy(dst=out_src, src=ag_call, name=f"{tag}_out_call_store")
        release_weight_gate(
            ncc.all_gather(
                srcs=[out_src], dsts=[out_dst], replica_group=tp_group, collective_dim=0, name=f"{tag}_out_call_ag"
            )
        )
        nisa.dma_copy(dst=y_hT.reshape((_P_MAX, cols)), src=out_dst[0:_P_MAX, 0:cols], name=f"{tag}_out_call_load")
    elif ag_call is not None:
        # One all-gather of every pass's slice into a tile that only the collective writes (no ECC block shared with
        # an engine write), then one copy into y_hT.
        gathered_rows = -(-total_rows // _TAIL_ROWS) * tr
        y_all = sb.tile((_P_MAX, gathered_rows, _HIDDEN_TILES), bf16)
        release_weight_gate(
            ncc.all_gather(
                srcs=[ag_call],
                dsts=[y_all.reshape((_P_MAX, gathered_rows * _HIDDEN_TILES))],
                replica_group=tp_group,
                collective_dim=0,
                name=f"{tag}_out_call_ag",
            )
        )
        if draft_seq:
            # ``seq_passes``: pass p's slice holds its rows in pass order; they go back to y_hT by row runs.
            for p, p_rows in enumerate(pass_rows):
                for dst_row, src_row, n in _row_runs(p_rows):
                    nisa.tensor_copy(
                        dst=y_hT[0:_P_MAX, src_row : src_row + n, 0:_HIDDEN_TILES],
                        src=y_all[0:_P_MAX, p * tr + dst_row : p * tr + dst_row + n, 0:_HIDDEN_TILES],
                    )
        else:
            nisa.tensor_copy(
                dst=y_hT[0:_P_MAX, 0:total_rows, 0:_HIDDEN_TILES], src=y_all[0:_P_MAX, 0:total_rows, 0:_HIDDEN_TILES]
            )
    object.__setattr__(y_hT, "side_effects", tuple(cache_writes))
    object.__setattr__(y_hT, "side_effect_sources", (write_rows, write_row) if batched_cache_write else (write_rows,))
    return y_hT


mla_block_attention.token_major = True
"""The body takes and returns token-major ``[128, Q*B, 56]`` hT tiles."""

CP_TARGET_HEADS = 96
"""Heads of the target MLA layers."""


def mla_cp_decode_attention(
    x_hT: nl.NkiTensor,
    mla_in_proj: nl.NkiTensor,
    mla_q_a_norm: nl.NkiTensor,
    mla_q_b_proj: nl.NkiTensor,
    mla_kv_a_norm: nl.NkiTensor,
    mla_kv_b_proj: nl.NkiTensor,
    mla_o_proj: nl.NkiTensor,
    kv_pool: nl.NkiTensor,
    block_table: nl.NkiTensor,
    slot: nl.NkiTensor,
    position: nl.NkiTensor,
    rank_info: nl.NkiTensor,
    cp_group: ncc.ReplicaGroup,
    tag: str,
    loop_path: bool = False,
    sbuf: SbufTiles = None,
    sources: dict | None = None,
    fp8_pages: bool = False,
    cp: int = 128,
    reduce_rows: int = None,
    transposed_scores: bool = True,
    lora_norm_eps: float = _LORA_NORM_EPS,
    zeroed_pages: bool = False,
    loop_first_static: bool = False,
    scores_ahead: bool = False,
    folded_reference: bool = False,
    rows_dr: bool = False,
    rows_dr_act_copies: bool = False,
    rows_dr_inplace: bool = False,
) -> nl.NkiTensor:
    """Run one target MLA decode layer (one token for each of B sequences) with the CP``cp`` layout.

    The decode is the one-row bitwise form of ``mla_block_attention`` (``q_rows=1``, ``bitwise=True``), so the
    rows of a verify step with ``bitwise=True`` are bitwise equal to the same number of decode steps. The
    arguments are those of ``tkg_mla_paged.mla_decode_attention``; ``sources`` holds the persistent tiles of the
    layer collectives under the key ``mla_cp`` (see ``decode_block.collective_sources``). ``reduce_rows`` is the row
    count of the verify step of the deployment (see ``mla_block_attention``). ``transposed_scores`` must match the
    verify of the deployment. ``zeroed_pages``: the loop path reads its cache pages with no clamp
    (``mla_block_attention`` ``zeroed_pages``; set it only when every row that the loop path can read is finite).
    ``loop_first_static``: ``mla_block_attention`` ``loop_first_static``; the verify of the deployment must match it.
    ``scores_ahead`` / ``folded_reference``: the ``mla_block_attention`` keys. ``rows_dr`` (it changes the bits, so the
    verify of the deployment must match it), ``rows_dr_act_copies`` and ``rows_dr_inplace``: the ``mla_block_attention``
    keys of the double_row keys (``CP_ROWS_OPTIONS``).
    """
    return mla_block_attention(
        x_hT,
        mla_in_proj,
        mla_q_a_norm,
        mla_q_b_proj,
        mla_kv_a_norm,
        mla_kv_b_proj,
        mla_o_proj,
        kv_pool,
        block_table,
        slot,
        position,
        rank_info,
        cp_group,
        tag,
        local_heads=1,
        q_rows=1,
        causal=True,
        output_gate=True,
        write_cache=True,
        loop_path=loop_path,
        fp8_pages=fp8_pages,
        sbuf=sbuf,
        bitwise=True,
        sources=None if sources is None else sources.get("mla_cp"),
        cp=cp,
        heads=CP_TARGET_HEADS,
        reduce_rows=reduce_rows,
        transposed_scores=transposed_scores,
        lora_norm_eps=lora_norm_eps,
        zeroed_pages=zeroed_pages,
        **({"loop_first_static": True} if loop_first_static else {}),
        **({"scores_ahead": True} if scores_ahead else {}),
        **({"folded_reference": True} if folded_reference else {}),
        **({"rows_dr": True} if rows_dr else {}),
        **({"rows_dr_act_copies": True} if rows_dr_act_copies else {}),
        **({"rows_dr_inplace": True} if rows_dr_inplace else {}),
    )


mla_cp_decode_attention.token_major = True

ROWS_LOCAL_HEADS = 3
"""Heads of a rank in the CP32 rows form (``mla_rows_decode_attention``)."""

ROWS_SEQUENCES = (1, 2, 4, 8)
"""Sequences per replica of a CP32 rows-form call: each divides the rows of a pass (``ROWS_PASS_ROWS``), so the masks of
a Q-row call fuse, and a decode call of that many rows needs no fit passes."""

ROWS_PASS_ROWS = _TAIL_ROWS
"""Rows of one pass of ``mla_block_attention``; a rows-form k1 verify above it needs the sequence-major fit passes."""

ROWS_GROUP_ROWS = _MAX_SEQS
"""Rows of one call of the rows-form decode twin; more sequences run in groups of this many."""

ROWS_OPTIONS = (
    "folded_reference",
    "rows_v_stationary",
    "rows_dr",
    "rows_dr_act_copies",
    "rows_dr_inplace",
    "rows_prefetch",
    "rows_pair_lag",
    "rows_pad_skip",
    "rows_seq_hoist",
    "rows_setup_ahead",
    "rows_q_fp8",
    "rows_mx_keys",
    "rows_pair_tail",
    "rows_unroll2",
)
"""Keys of the rows form that a rows-form k1 verify can add (``verify_model`` ``mla_k1_rows_options``)."""


def q_call_fp8(pack_options: dict = None, rows: bool = False, rows_options: dict = None) -> bool | str:
    """Return the ``q_fp8`` of the query gather pair of the k1 MLA call (``mla_block_collective_hbm``): True when it must
    be e4m3 (the packed loop's ``q_fp8`` True, ``pack_options``, or the rows form, ``rows``, with ``rows_q_fp8``,
    ``rows_options``), "latent" for the packed loop's ``q_fp8`` "latent" (the uint16 pair of the e4m3 latent and BF16
    rope query), "mx" for ``q_mx_gather`` ("mx_packed" with ``q_mx_scale_words``), else False.

    Example:
        >>> q_call_fp8(None, True, {"rows_dr": True, "rows_q_fp8": True}), q_call_fp8({"q_fp8": "latent"})
        (True, 'latent')
    """
    if (pack_options or {}).get("q_mx_gather"):
        return "mx_packed" if (pack_options or {}).get("q_mx_scale_words") else "mx"
    if (pack_options or {}).get("q_fp8") == "latent":
        return "latent"
    return bool((pack_options or {}).get("q_fp8") or (rows and (rows_options or {}).get("rows_q_fp8")))


ROWS_TWIN_OPTIONS = ("folded_reference", "rows_v_stationary", "rows_dr", "rows_mx_keys")
"""Keys of ``ROWS_OPTIONS`` that can change the bits of a row: the decode twin takes the values of the verify
(``rows_mx_keys`` is accuracy class, so the twin shares its key to stay bitwise against the verify's rows)."""

CP_ROWS_OPTIONS = ("rows_dr", "rows_dr_act_copies", "rows_dr_inplace")
"""Keys of the rows form that the CP128 MLA calls (the 1M-context batch-1 verify and its CP decode) take, as
``mla_cp_rows_options`` of the verify and decode models: the double_row keys (one head on each rank,
``_dr_head_queries``). ``rows_dr`` changes the bits, so the verify and its decode must both take it."""

ROWS_TWIN_SEQUENCES = (*ROWS_SEQUENCES, 16)
"""Sequences per replica of the rows-form decode twin (``mla_rows_decode_attention``): above ``_MAX_SEQS`` it runs
groups of ``_MAX_SEQS`` sequences (the 1M-context batch-64 decode, 16 sequences for each replica)."""


def mla_rows_decode_attention(
    x_hT: nl.NkiTensor,
    mla_in_proj: nl.NkiTensor,
    mla_q_a_norm: nl.NkiTensor,
    mla_q_b_proj: nl.NkiTensor,
    mla_kv_a_norm: nl.NkiTensor,
    mla_kv_b_proj: nl.NkiTensor,
    mla_o_proj: nl.NkiTensor,
    kv_pool: nl.NkiTensor,
    block_table: nl.NkiTensor,
    slot: nl.NkiTensor,
    position: nl.NkiTensor,
    rank_info: nl.NkiTensor,
    tp_group: ncc.ReplicaGroup,
    tag: str,
    loop_path: bool = False,
    sbuf: SbufTiles = None,
    sources: dict | None = None,
    fp8_pages: bool = False,
    zeroed_pages: bool = False,
    lora_norm_eps: float = _LORA_NORM_EPS,
    state_bf16_o: bool = False,
    folded_reference: bool = False,
    rows_v_stationary: bool = False,
    rows_dr: bool = False,
    rows_prefetch: bool = False,
    rows_dr_act_copies: bool = False,
    rows_dr_inplace: bool = False,
    rows_pair_lag: bool = False,
    rows_pad_skip: bool = False,
    live_rows: nl.NkiTensor = None,
    rows_seq_hoist: bool = False,
    mla_in_proj_scale: nl.NkiTensor = None,
    mla_q_b_proj_scale: nl.NkiTensor = None,
    mla_kv_b_proj_scale: nl.NkiTensor = None,
    mla_o_proj_scale: nl.NkiTensor = None,
    rows_setup_ahead: bool = False,
    rows_mx_keys: str = None,
) -> nl.NkiTensor:
    """Run one target MLA decode layer (one token for each sequence) in the CP32 rows form.

    This is the identity twin of the CP32 rows-form verify (``verify_model`` ``mla_k1_rows``): the one-row bitwise
    form of ``mla_block_attention`` with ``transposed_scores``. In the rows form the result of a row does not depend
    on the other rows of its sequence, so each row of a rows-form verify step is bitwise equal to this decode.
    ``sources`` holds the persistent tiles of the call under the key ``mla_rows`` (see ``decode_model``). On the
    loop path the call takes the forms of the loop-path verify: the query and output gathers in persistent HBM and
    the late tail weights (the output does not change). Test only: the served decode runs
    ``tkg_mla_paged.mla_decode_attention``.

    Above ``_MAX_SEQS`` sequences the layer runs one call for each group of ``_MAX_SEQS`` sequences, each on its own
    copy of the group's block table, slots and positions and with its own source set (``sources["mla_rows"]`` is then
    a list of sets). A row's result does not depend on the other sequences, so the groups keep the bits of the rows.
    ``state_bf16_o``: the BF16-o state reduce-scatter of ``mla_block_attention``, the twin of a BF16-o rows verify.
    ``folded_reference``, ``rows_v_stationary``, ``rows_dr``, ``rows_prefetch``, ``rows_dr_act_copies``,
    ``rows_dr_inplace``, ``rows_pair_lag``, ``rows_pad_skip``, ``rows_seq_hoist``, ``rows_setup_ahead``: the keys of
    ``mla_block_attention``, the twin of a verify that takes them (``rows_prefetch``, ``rows_dr_act_copies``,
    ``rows_dr_inplace``, ``rows_pair_lag``, ``rows_pad_skip``, ``rows_seq_hoist`` and ``rows_setup_ahead`` keep the bits
    of the live rows, so the twin may take them or not). ``rows_mx_keys`` ("single"): the MX latent QK of
    ``mla_block_attention``, the twin of a verify that takes it (accuracy class; the decode and the verify take the
    same key, so each verify row stays equal to this decode). ``live_rows``: the runner's
    ``[B, 1]`` live-row mask of ``rows_pad_skip`` (``mla_block_attention``). ``mla_in_proj_scale``: None (a BF16
    ``mla_in_proj``), or the scales of an MXFP8 ``mla_in_proj`` (``mla_block_attention`` ``mla_in_proj_scale``), as
    the MXFP8 verify takes them.
    """
    rows_sources = None if sources is None else sources.get("mla_rows")
    tokens = slot.shape[0]
    if tokens > _MAX_SEQS:
        sb = AUTO_SBUF if sbuf is None else sbuf
        y_hT = sb.tile((_P_MAX, tokens, _HIDDEN_TILES), x_hT.dtype)
        effects, effect_sources = (), ()
        for g, g0 in enumerate(range(0, tokens, _MAX_SEQS)):
            n = min(_MAX_SEQS, tokens - g0)
            table = nl.ndarray(
                (n, block_table.shape[1]), dtype=block_table.dtype, buffer=nl.private_hbm, name=f"{tag}_g{g}_table"
            )
            group_slot = nl.ndarray((n, 1), dtype=slot.dtype, buffer=nl.private_hbm, name=f"{tag}_g{g}_slot")
            group_position = nl.ndarray(
                (n, 1), dtype=position.dtype, buffer=nl.private_hbm, name=f"{tag}_g{g}_position"
            )
            nisa.dma_copy(dst=table, src=block_table[g0 : g0 + n, 0 : block_table.shape[1]])
            nisa.dma_copy(dst=group_slot, src=slot[g0 : g0 + n, 0:1])
            nisa.dma_copy(dst=group_position, src=position[g0 : g0 + n, 0:1])
            group_live = None
            if live_rows is not None:
                group_live = nl.ndarray((n, 1), dtype=live_rows.dtype, buffer=nl.private_hbm, name=f"{tag}_g{g}_live")
                nisa.dma_copy(dst=group_live, src=live_rows[g0 : g0 + n, 0:1])
            with sb.scope(f"{tag}_rows_g{g}"):
                x_group = sb.tile((_P_MAX, n, _HIDDEN_TILES), x_hT.dtype)
                nisa.tensor_copy(dst=x_group, src=x_hT[0:_P_MAX, g0 : g0 + n, 0:_HIDDEN_TILES])
                y_group = mla_rows_decode_attention(
                    x_group,
                    mla_in_proj,
                    mla_q_a_norm,
                    mla_q_b_proj,
                    mla_kv_a_norm,
                    mla_kv_b_proj,
                    mla_o_proj,
                    kv_pool,
                    table,
                    group_slot,
                    group_position,
                    rank_info,
                    tp_group,
                    f"{tag}_g{g}",
                    loop_path=loop_path,
                    sbuf=sbuf,
                    sources=None if rows_sources is None else {"mla_rows": rows_sources[g]},
                    fp8_pages=fp8_pages,
                    zeroed_pages=zeroed_pages,
                    lora_norm_eps=lora_norm_eps,
                    state_bf16_o=state_bf16_o,
                    folded_reference=folded_reference,
                    rows_v_stationary=rows_v_stationary,
                    rows_dr=rows_dr,
                    rows_prefetch=rows_prefetch,
                    rows_dr_act_copies=rows_dr_act_copies,
                    rows_dr_inplace=rows_dr_inplace,
                    rows_pair_lag=rows_pair_lag,
                    rows_pad_skip=rows_pad_skip,
                    live_rows=group_live,
                    rows_seq_hoist=rows_seq_hoist,
                    mla_in_proj_scale=mla_in_proj_scale,
                    mla_q_b_proj_scale=mla_q_b_proj_scale,
                    mla_kv_b_proj_scale=mla_kv_b_proj_scale,
                    mla_o_proj_scale=mla_o_proj_scale,
                    rows_setup_ahead=rows_setup_ahead,
                    rows_mx_keys=rows_mx_keys,
                )
                nisa.tensor_copy(dst=y_hT[0:_P_MAX, g0 : g0 + n, 0:_HIDDEN_TILES], src=y_group)
            effects += y_group.side_effects
            effect_sources += y_group.side_effect_sources
        object.__setattr__(y_hT, "side_effects", effects)
        object.__setattr__(y_hT, "side_effect_sources", effect_sources)
        return y_hT
    return mla_block_attention(
        x_hT,
        mla_in_proj,
        mla_q_a_norm,
        mla_q_b_proj,
        mla_kv_a_norm,
        mla_kv_b_proj,
        mla_o_proj,
        kv_pool,
        block_table,
        slot,
        position,
        rank_info,
        tp_group,
        tag,
        local_heads=ROWS_LOCAL_HEADS,
        q_rows=1,
        causal=True,
        output_gate=True,
        write_cache=True,
        loop_path=loop_path,
        fp8_pages=fp8_pages,
        sbuf=sbuf,
        bitwise=True,
        sources=rows_sources,
        persistent_rs=rows_sources is not None and "rs" in rows_sources,
        transposed_scores=True,
        zeroed_pages=zeroed_pages,
        lora_norm_eps=lora_norm_eps,
        **({"q_hbm": True, "out_ag_call": True, "late_tail_weights": True} if loop_path else {}),
        **({"state_bf16_o": True} if state_bf16_o else {}),
        **({"folded_reference": True} if folded_reference else {}),
        **({"rows_v_stationary": True} if rows_v_stationary else {}),
        **({"rows_dr": True} if rows_dr else {}),
        **({"rows_prefetch": True} if rows_prefetch else {}),
        **({"rows_dr_act_copies": True} if rows_dr_act_copies else {}),
        **({"rows_dr_inplace": True} if rows_dr_inplace else {}),
        **({"rows_pair_lag": True} if rows_pair_lag else {}),
        **({"rows_pad_skip": True} if rows_pad_skip else {}),
        **({"live_rows": live_rows} if live_rows is not None else {}),
        **({"rows_seq_hoist": True} if rows_seq_hoist else {}),
        **({"mla_in_proj_scale": mla_in_proj_scale} if mla_in_proj_scale is not None else {}),
        **({"mla_q_b_proj_scale": mla_q_b_proj_scale} if mla_q_b_proj_scale is not None else {}),
        **({"mla_kv_b_proj_scale": mla_kv_b_proj_scale} if mla_kv_b_proj_scale is not None else {}),
        **({"mla_o_proj_scale": mla_o_proj_scale} if mla_o_proj_scale is not None else {}),
        **({"rows_setup_ahead": True} if rows_setup_ahead else {}),
        **({"rows_mx_keys": rows_mx_keys} if rows_mx_keys is not None else {}),
    )


mla_rows_decode_attention.token_major = True


def _row_runs(rows):
    """Return the runs ``(pass row, global row, count)`` of consecutive global ``rows`` (a pass in its row order)."""
    runs = []
    for dst_row, src_row in enumerate(rows):
        if runs and runs[-1][1] + runs[-1][2] == src_row:
            runs[-1][2] += 1
        else:
            runs.append([dst_row, src_row, 1])
    return [tuple(run) for run in runs]


def _gather_head_queries(
    q_local,
    cb,
    rows,
    q_src_hbm,
    q_dst_hbm,
    q_nope_all,
    q_rope_all,
    heads,
    cp_group,
    tag,
    sb,
    row_ids=None,
    dr_rows=None,
    dr_identity=None,
):
    """Gather the queries of query rows ``cb .. cb + rows - 1`` (or of the global rows ``row_ids``, in their order) over
    the CP group, one head on each rank.

    Rank ``g`` holds head ``g`` (``q_local`` ``[Q*B, 576]``: the absorbed query, then the rope query). It stores its
    rows in a persistent HBM source ``[tr, 576]``; the gather gives the ``tr`` rows of each rank (head) one after
    the other, so one DMA loads the rows of rank ``g`` onto partition ``g``. TensorE transposes then put the latent tiles and the
    rope query of the ``heads`` heads in feature-major columns (``q_nope_all`` ``[128, rows, 4, heads]``,
    ``q_rope_all`` ``[128, rows, heads]``); the rows of the pad ranks are not read. Return the DMA that loads the
    gathered rows. ``dr_rows`` (``rows_dr``): also write the e4m3 double_row stationaries of the rows from the gathered
    rows (``_dr_head_queries``, with the BF16 identity ``dr_identity``).
    """
    bf16 = nl.bfloat16
    width = _KV_LORA + _ROPE
    tr = q_src_hbm.shape[0]
    ranks = q_dst_hbm.shape[0] // tr
    if row_ids is None:
        nisa.dma_copy(dst=q_src_hbm[0:rows, 0:width], src=q_local[cb : cb + rows, 0:width], name=f"{tag}_mla_q_store")
    else:
        for dst_row, src_row, n in _row_runs(row_ids):
            nisa.dma_copy(
                dst=q_src_hbm[dst_row : dst_row + n, 0:width],
                src=q_local[src_row : src_row + n, 0:width],
                name=f"{tag}_mla_q_store_{dst_row}",
            )
    ncc.all_gather(srcs=[q_src_hbm], dsts=[q_dst_hbm], replica_group=cp_group, collective_dim=0, name=f"{tag}_mla_q_ag")
    q_all = sb.tile((ranks, rows * width), bf16)
    gathered = q_dst_hbm.reshape((ranks, tr * width))
    q_load = nisa.dma_copy(dst=q_all, src=gathered[0:ranks, 0 : rows * width], name=f"{tag}_mla_q_load")
    for row in range(rows):
        base = row * width
        latent_psum = nl.ndarray((_P_MAX, _KV_LORA_TILES, ranks), dtype=bf16, buffer=nl.psum)
        for latent_tile in range(_KV_LORA_TILES):
            col = base + latent_tile * _P_MAX
            nisa.nc_transpose(dst=latent_psum[0:_P_MAX, latent_tile, 0:ranks], data=q_all[0:ranks, col : col + _P_MAX])
        nisa.tensor_copy(
            dst=q_nope_all[0:_P_MAX, row, 0:_KV_LORA_TILES, 0:heads],
            src=latent_psum[0:_P_MAX, 0:_KV_LORA_TILES, 0:heads],
            engine=nisa.engine.scalar if row % 2 else nisa.engine.vector,
        )
        rope_psum = nl.ndarray((_ROPE, ranks), dtype=bf16, buffer=nl.psum)
        nisa.nc_transpose(dst=rope_psum, data=q_all[0:ranks, base + _KV_LORA : base + width])
        nisa.tensor_copy(
            dst=q_rope_all[0:_ROPE, row, 0:heads],
            src=rope_psum[0:_ROPE, 0:heads],
            engine=nisa.engine.vector if row % 2 else nisa.engine.scalar,
        )
    if dr_rows is not None:
        _dr_head_queries(q_all, rows, dr_rows, heads, dr_identity, sb)
    return q_load


def _dr_head_queries(q_all, rows, dr_rows, heads, identity16, sb):
    """Write the e4m3 double_row stationaries ``dr_rows`` ``[128, rows, 2, 2, heads]`` FP8 of the CP128 query gather
    ``q_all`` ``[ranks, rows x 576]`` (rank ``g`` holds head ``g``, ``_gather_head_queries``), as
    ``tkg_mla_paged._dr_gather_tile_queries`` writes them from the 3-head gather tiles (``rows_dr`` with one head on each
    rank). The latent query of each row rounds to e4m3 (clamped to +-448, the rounding of the decode's ``pack_dr_keys``),
    a 2-byte transpose-mode transpose of each 256-feature block moves the feature pairs ``(2c, 2c + 1)`` of every head to
    partition ``c`` (it moves the bits), and one copy for each byte takes byte ``i`` of the words: element
    ``(c, row, i, k, h)`` is latent feature ``256 k + 2c + i`` of head ``h``."""
    fp8 = nl.float8_e4m3fn
    ranks = q_all.shape[0]
    width = _KV_LORA + _ROPE
    with sb.scope("dr_head_queries"):
        for row in range(rows):
            q8 = sb.tile((ranks, _KV_LORA), fp8)
            nisa.tensor_scalar(
                dst=q8,
                data=q_all[0:ranks, row * width : row * width + _KV_LORA],
                op0=nl.minimum,
                operand0=_E4M3_MAX,
                op1=nl.maximum,
                operand1=-_E4M3_MAX,
                engine=nisa.engine.vector,
            )
            q8_words = q8.view(nl.bfloat16)
            words_psum = nl.ndarray((_P_MAX, _DR_BLOCKS, ranks), dtype=nl.bfloat16, buffer=nl.psum)
            for block in range(_DR_BLOCKS):
                col = block * _DR_BLOCK_WORDS
                nisa.nc_matmul(
                    dst=words_psum[0:_P_MAX, block, 0:ranks],
                    stationary=q8_words[0:ranks, col : col + _DR_BLOCK_WORDS],
                    moving=identity16[0:ranks, 0:ranks],
                    is_transpose=True,
                    accumulate=False,
                )
            words = words_psum.view(fp8).reshape_dim(2, (ranks, 2))
            for byte in range(2):
                nisa.tensor_copy(
                    dst=dr_rows[0:_P_MAX, row, byte, 0:_DR_BLOCKS, 0:heads],
                    src=words[0:_P_MAX, 0:_DR_BLOCKS, 0:heads, byte],
                    engine=nisa.engine.vector,
                )


def _apply_rope(x, cos_x, sin_x, rows, n_heads, sb):
    """Rotate the GPT-J pairs of ``x`` (``[rows, n_heads * 64]`` BF16) in place, in FP32.

    ``cos_x`` and ``sin_x`` are ``[rows, 64]`` FP32, pre-expanded for the pairs (element ``2i`` and
    ``2i+1`` both hold the frequency ``i``). The rotation is ``x * cos + rot(x) * sin`` with
    ``rot(x)[2i] = -x[2i+1]`` and ``rot(x)[2i+1] = x[2i]``.
    """
    fp32 = nl.float32
    x3 = x.reshape((rows, n_heads, _ROPE))
    xf = sb.tile((rows, n_heads, _ROPE), fp32)
    nisa.tensor_copy(dst=xf, src=x3)
    xf_pairs = xf.reshape((rows, n_heads, _ROPE // 2, 2))
    rot = sb.tile((rows, n_heads, _ROPE // 2, 2), fp32)
    nisa.tensor_scalar(dst=rot[:, :, :, 0:1], data=xf_pairs[:, :, :, 1:2], op0=nl.multiply, operand0=-1.0)
    nisa.tensor_copy(dst=rot[:, :, :, 1:2], src=xf_pairs[:, :, :, 0:1])
    rot_flat = rot.reshape((rows, n_heads, _ROPE))
    cos_b = cos_x.reshape((rows, 1, _ROPE)).broadcast(1, n_heads)
    sin_b = sin_x.reshape((rows, 1, _ROPE)).broadcast(1, n_heads)
    out = sb.tile((rows, n_heads, _ROPE), fp32)
    nisa.tensor_tensor(dst=out, data1=xf, data2=cos_b, op=nl.multiply)
    nisa.tensor_tensor(dst=rot_flat, data1=rot_flat, data2=sin_b, op=nl.multiply)
    nisa.tensor_tensor(dst=out, data1=out, data2=rot_flat, op=nl.add)
    nisa.tensor_copy(dst=x3, src=out)


class _Softmax:
    """The softmax reference of one query row, capped (default) or exact.

    The capped reference of a row is its own current-key score scaled plus 64.46, and a score more
    than 124.46 above it is capped. The exact reference is the running maximum of the raw scores on
    this rank, which starts at the maximum over the row's visible current keys.
    """

    def __init__(self, exact, scale, total_rows, heads, sb):
        self.exact = exact
        self.scale = scale
        self.heads = heads
        self.sb = sb
        fp32 = nl.float32
        if exact:
            self.running_max = sb.tile((heads, total_rows), fp32)
            self.alpha = sb.tile((heads, 1), fp32)
        else:
            self.negative_reference = sb.tile((heads, total_rows), fp32)
            self.score_cap = sb.tile((heads, total_rows), fp32)

    def set_reference(self, current_score, row, count=1):
        """Set the capped reference of ``row`` from its own current-key raw score ``[heads, 1]`` (``count`` rows
        ``row`` .. ``row + count - 1`` from ``[heads, count]`` scores: the same elementwise ops, so the same bits)."""
        heads = self.heads
        nisa.tensor_scalar(
            dst=self.negative_reference[0:heads, row : row + count],
            data=current_score,
            op0=nl.multiply,
            operand0=-self.scale,
            op1=nl.subtract,
            operand1=_MERGE_OFFSET,
        )
        nisa.tensor_scalar(
            dst=self.score_cap[0:heads, row : row + count],
            data=self.negative_reference[0:heads, row : row + count],
            op0=nl.multiply,
            operand0=-1.0 / self.scale,
            op1=nl.add,
            operand1=_WEIGHT_CAP / self.scale,
        )

    def update(self, state, row, numerator_psum, group_sum):
        """Add one prior-key group to the FP32 state of ``row``."""
        heads = self.heads
        if self.exact:
            nisa.scalar_tensor_tensor(
                dst=state[0:heads, row, 0:_KV_LORA],
                data=state[0:heads, row, 0:_KV_LORA],
                op0=nl.multiply,
                operand0=self.alpha,
                op1=nl.add,
                operand1=numerator_psum,
            )
            nisa.scalar_tensor_tensor(
                dst=state[0:heads, row, _KV_LORA : _KV_LORA + 1],
                data=state[0:heads, row, _KV_LORA : _KV_LORA + 1],
                op0=nl.multiply,
                operand0=self.alpha,
                op1=nl.add,
                operand1=group_sum,
            )
        else:
            nisa.tensor_tensor(
                dst=state[0:heads, row, 0:_KV_LORA],
                data1=state[0:heads, row, 0:_KV_LORA],
                data2=numerator_psum,
                op=nl.add,
            )
            nisa.tensor_tensor(
                dst=state[0:heads, row, _KV_LORA : _KV_LORA + 1],
                data1=state[0:heads, row, _KV_LORA : _KV_LORA + 1],
                data2=group_sum,
                op=nl.add,
            )

    def merge_scale(self, state, tp_group, tag):
        """Scale the state of each head by ``exp(scale * (rank maximum - maximum over the ranks))``."""
        fp32 = nl.float32
        heads = self.heads
        total_rows = state.shape[1]
        max_row_psum = nl.ndarray((1, total_rows * heads), dtype=fp32, buffer=nl.psum)
        for row in range(total_rows):
            nisa.nc_transpose(
                dst=max_row_psum[0:1, row * heads : (row + 1) * heads], data=self.running_max[0:heads, row : row + 1]
            )
        max_row = self.sb.tile((1, total_rows * heads), fp32)
        nisa.tensor_copy(dst=max_row, src=max_row_psum)
        global_row = self.sb.tile((1, total_rows * heads), fp32)
        ncc.all_reduce(
            srcs=[max_row], dsts=[global_row], replica_group=tp_group, op=nl.maximum, name=f"{tag}_mla_max_all_reduce"
        )
        global_psum = nl.ndarray((heads, total_rows), dtype=fp32, buffer=nl.psum)
        for row in range(total_rows):
            nisa.nc_transpose(
                dst=global_psum[0:heads, row : row + 1], data=global_row[0:1, row * heads : (row + 1) * heads]
            )
        negative_global = self.sb.tile((heads, total_rows), fp32)
        nisa.tensor_scalar(dst=negative_global, data=global_psum, op0=nl.multiply, operand0=-self.scale)
        merge = self.sb.tile((heads, total_rows), fp32)
        for row in range(total_rows):
            nisa.activation(
                dst=merge[0:heads, row : row + 1],
                op=nl.exp,
                data=self.running_max[0:heads, row : row + 1],
                scale=self.scale,
                bias=negative_global[0:heads, row : row + 1],
            )
            nisa.tensor_scalar(
                dst=state[0:heads, row, 0 : _KV_LORA + 1],
                data=state[0:heads, row, 0 : _KV_LORA + 1],
                op0=nl.multiply,
                operand0=merge[0:heads, row : row + 1],
            )


def _current_block(
    q_row,
    r,
    b,
    j,
    q_nope_all,
    q_rope_all,
    cur_latent_keys,
    cur_rope_keys,
    cur_rows,
    state,
    softmax,
    heads,
    q_rows,
    causal,
    rank_scale,
    sb,
):
    """Attend the current block of query row ``r = j * B + b`` and write the ``1/cp`` state.

    ``q_row`` indexes the row in the gathered query tile of the current chunk; ``r`` is the global row,
    which indexes the softmax reference, the state, and the own-key column of the all-rows current keys.
    The query attends every current key of its own sequence (block mask) or keys ``0 .. j`` (causal).
    The softmax reference is the row's own current key. Each rank writes ``rank_scale = 1/cp`` of the state;
    the CP reduce-scatter sums the cp identical copies.
    """
    bf16 = nl.bfloat16
    fp32 = nl.float32
    total_rows = cur_rows.shape[0]
    tokens = total_rows // q_rows
    cq = q_row
    sb.open_scope("current")
    scores_psum = nl.ndarray((heads, total_rows), dtype=fp32, buffer=nl.psum)
    for latent_tile in range(_KV_LORA_TILES):
        nisa.nc_matmul(
            dst=scores_psum,
            stationary=q_nope_all[0:_P_MAX, q_row, latent_tile, 0:heads],
            moving=cur_latent_keys[0:_P_MAX, latent_tile, 0:total_rows],
            accumulate=latent_tile > 0,
        )
    nisa.nc_matmul(
        dst=scores_psum,
        stationary=q_rope_all[0:_ROPE, q_row, 0:heads],
        moving=cur_rope_keys[0:_ROPE, 0:total_rows],
        accumulate=True,
    )
    scores = sb.tile((heads, total_rows), fp32)
    nisa.tensor_copy(dst=scores, src=scores_psum)
    # Mask every key outside the row's own sequence, and the future keys of the causal mask.
    mask = sb.tile((heads, q_rows, tokens), fp32)
    nisa.memset(dst=mask, value=_MASK_BIAS)
    visible = (j + 1) if causal else q_rows
    nisa.memset(dst=mask[0:heads, 0:visible, b : b + 1], value=0.0)
    group_sum = sb.tile((heads, 1), fp32)
    weights = sb.tile((heads, total_rows), bf16)
    if softmax.exact:
        masked = sb.tile((heads, total_rows), fp32)
        nisa.tensor_tensor(dst=masked, data1=scores, data2=mask.reshape((heads, total_rows)), op=nl.add)
        running_max = softmax.running_max[0:heads, cq : cq + 1]
        nisa.tensor_reduce(dst=running_max, op=nl.maximum, data=masked, axis=1)
        bias = sb.tile((heads, 1), fp32)
        nisa.tensor_scalar(dst=bias, data=running_max, op0=nl.multiply, operand0=-softmax.scale)
        nisa.activation(
            dst=weights,
            op=nl.exp,
            data=masked,
            scale=softmax.scale,
            bias=bias,
            reduce_op=nl.add,
            reduce_res=group_sum,
            reduce_cmd=nisa.reduce_cmd.reset_reduce,
        )
    else:
        softmax.set_reference(scores[0:heads, r : r + 1], cq)
        nisa.tensor_scalar(dst=scores, data=scores, op0=nl.minimum, operand0=softmax.score_cap[0:heads, cq : cq + 1])
        nisa.tensor_tensor(dst=scores, data1=scores, data2=mask.reshape((heads, total_rows)), op=nl.add)
        nisa.activation(
            dst=weights,
            op=nl.exp,
            data=scores,
            scale=softmax.scale,
            bias=softmax.negative_reference[0:heads, cq : cq + 1],
            reduce_op=nl.add,
            reduce_res=group_sum,
            reduce_cmd=nisa.reduce_cmd.reset_reduce,
        )
    weights_t_psum = nl.ndarray((total_rows, heads), dtype=bf16, buffer=nl.psum)
    nisa.nc_transpose(dst=weights_t_psum, data=weights)
    weights_t = sb.tile((total_rows, heads), bf16)
    nisa.tensor_copy(dst=weights_t, src=weights_t_psum)
    numerator_psum = nl.ndarray((heads, _KV_LORA), dtype=fp32, buffer=nl.psum)
    nisa.nc_matmul(dst=numerator_psum, stationary=weights_t, moving=cur_rows[0:total_rows, 0:_KV_LORA])
    nisa.tensor_scalar(dst=state[0:heads, cq, 0:_KV_LORA], data=numerator_psum, op0=nl.multiply, operand0=rank_scale)
    nisa.tensor_scalar(
        dst=state[0:heads, cq, _KV_LORA : _KV_LORA + 1], data=group_sum, op0=nl.multiply, operand0=rank_scale
    )
    sb.close_scope()


def _own_rows_ahead(cur_rows, rows, heads, weight, tag, sb):
    """``own_rows_ahead``: load the current row of each pass row and set the own-key weight on quadrant partitions, for
    ``_own_key`` ``own_row``. Pass row ``i`` (``rows[i]``, a row of ``cur_rows``) goes to partition ``32 (i % 4)`` of the
    512-column block ``i // 4`` of one tile; one DMA moves 4 evenly spaced rows (else one DMA a row), on the Activation
    HWDGE queue. The weight tile holds ``weight`` on every partition. Return the ``(row [1, 512], weight [1, heads])`` views
    of each pass row in order."""
    bf16 = nl.bfloat16
    quads = _P_MAX // _OWN_ROW_QUADRANT
    count_rows = len(rows)
    own_tile = sb.tile((_P_MAX, -(-count_rows // quads) * _KV_LORA), bf16)
    own_w = sb.tile((_P_MAX, heads), bf16)
    nisa.memset(dst=own_w, value=weight, engine=nisa.engine.gpsimd)
    for first in range(0, count_rows, quads):
        block_rows = list(rows[first : first + quads])
        count = len(block_rows)
        step = block_rows[1] - block_rows[0] if count > 1 else 1
        columns = slice((first // quads) * _KV_LORA, (first // quads + 1) * _KV_LORA)
        if step > 0 and all(later - row == step for row, later in zip(block_rows, block_rows[1:])):
            nisa.dma_copy(
                dst=own_tile[0 : _OWN_ROW_QUADRANT * (count - 1) + 1 : _OWN_ROW_QUADRANT, columns],
                src=cur_rows[block_rows[0] : block_rows[-1] + 1 : step, 0:_KV_LORA],
                dge_mode=nisa.dge_mode.hwdge,
                engine=nisa.engine.scalar,
                name=f"{tag}_mla_own_rows_{first}",
            )
            continue
        for index, r in enumerate(block_rows):
            partition = _OWN_ROW_QUADRANT * index
            nisa.dma_copy(
                dst=own_tile[partition : partition + 1, columns],
                src=cur_rows[r : r + 1, 0:_KV_LORA],
                dge_mode=nisa.dge_mode.hwdge,
                engine=nisa.engine.scalar,
                name=f"{tag}_mla_own_row_{r}",
            )
    views = []
    for index in range(count_rows):
        partition = _OWN_ROW_QUADRANT * (index % quads)
        block = index // quads
        views.append(
            (
                own_tile[partition : partition + 1, block * _KV_LORA : (block + 1) * _KV_LORA],
                own_w[partition : partition + 1, 0:heads],
            )
        )
    return views


def _own_key(
    cq,
    r,
    q_nope_all,
    q_rope_all,
    cur_latent_keys,
    cur_rope_keys,
    cur_rows,
    state,
    softmax,
    heads,
    weight,
    tag,
    sb,
    tied_sum=False,
    row_select=None,
    select_dst=None,
    own_row=None,
):
    """Initialize the bitwise state of query row ``r`` (chunk index ``cq``) with its own current key.

    This matches ``tkg_mla_paged``: the softmax reference is the own-key raw score, and the state numerator is
    ``weight * current_row`` on every head (``weight`` is the per-rank current weight, ``2**-98`` capped or
    ``2**-5`` exact), with the sum ``weight``. The current rows ``0 .. j-1`` are prior keys from the pool.
    ``tied_sum`` writes the sum as ``0 * numerator + weight`` (same bits, the clamped row is finite): an input-free
    memset can be scheduled early and keep the state tile live through the feed-forward layer before.
    ``select_dst`` (``packed_prior_options`` ``own_key_selects``): the ``[rows, heads]`` tile of the one-hot stationary of
    this row (its slice of the pass tile), not a tile of the frame that the next row reuses.
    ``own_row`` (``own_rows_ahead``): ``(row, weight)``, the ``[1, 512]`` current row and the ``[1, heads]`` weight on one
    quadrant partition of the pass tiles, loaded and set before the rows of the pass: no row DMA and no memset here.
    """
    bf16 = nl.bfloat16
    fp32 = nl.float32
    sb.open_scope("own")
    score_psum = nl.ndarray((heads, 1), dtype=fp32, buffer=nl.psum)
    for latent_tile in range(_KV_LORA_TILES):
        nisa.nc_matmul(
            dst=score_psum,
            stationary=q_nope_all[0:_P_MAX, cq, latent_tile, 0:heads],
            moving=cur_latent_keys[0:_P_MAX, latent_tile, r : r + 1],
            accumulate=latent_tile > 0,
        )
    nisa.nc_matmul(
        dst=score_psum,
        stationary=q_rope_all[0:_ROPE, cq, 0:heads],
        moving=cur_rope_keys[0:_ROPE, r : r + 1],
        accumulate=True,
    )
    if softmax.exact:
        nisa.tensor_copy(dst=softmax.running_max[0:heads, cq : cq + 1], src=score_psum)
    else:
        softmax.set_reference(score_psum, cq)
    if state is None:
        # ``p2``: the state of the pass is in the (row, head) layout (``_own_key_state_p2`` writes its own-key state).
        sb.close_scope()
        return
    numerator_psum = nl.ndarray((heads, _KV_LORA), dtype=fp32, buffer=nl.psum)
    if row_select is not None:
        # ``own_key_select``: weight on the partition of row r, 0 on the others; the matmul over all current rows.
        rows_all = row_select.shape[0]
        select = sb.tile((rows_all, heads), bf16) if select_dst is None else select_dst
        nisa.tensor_scalar(
            dst=select,
            data=row_select,
            op0=nl.equal,
            operand0=float(r),
            op1=nl.multiply,
            operand1=weight,
            **({} if select_dst is None else {"engine": nisa.engine.vector}),
        )
        nisa.nc_matmul(
            dst=numerator_psum, stationary=select[0:rows_all, 0:heads], moving=cur_rows[0:rows_all, 0:_KV_LORA]
        )
    elif own_row is not None:
        # Numerator = weight * current_row on every head, from the pass tiles of ``own_rows_ahead`` (the row and the
        # weight on the same quadrant partition).
        row_view, weight_view = own_row
        nisa.nc_matmul(dst=numerator_psum, stationary=weight_view, moving=row_view)
    else:
        # Numerator = weight * current_row on every head. The one-partition matmul needs the row on a quadrant
        # partition, so a DMA shifts it to partition 0.
        row0 = sb.tile((1, _KV_LORA), bf16)
        nisa.dma_copy(
            dst=row0[0:1, 0:_KV_LORA],
            src=cur_rows[r : r + 1, 0:_KV_LORA],
            dge_mode=nisa.dge_mode.hwdge,
            engine=nisa.engine.sync,
            name=f"{tag}_mla_own_row_{r}",
        )
        weight_tile = sb.tile((1, heads), bf16)
        nisa.memset(dst=weight_tile, value=weight)
        nisa.nc_matmul(dst=numerator_psum, stationary=weight_tile[0:1, 0:heads], moving=row0[0:1, 0:_KV_LORA])
    nisa.tensor_copy(dst=state[0:heads, cq, 0:_KV_LORA], src=numerator_psum)
    sum_column = state[0:heads, cq, _KV_LORA : _KV_LORA + 1]
    if tied_sum:
        nisa.tensor_scalar(
            dst=sum_column,
            data=numerator_psum[0:heads, 0:1],
            op0=nl.multiply,
            operand0=0.0,
            op1=nl.add,
            operand1=weight,
        )
    else:
        nisa.memset(dst=sum_column, value=weight)
    sb.close_scope()


def _attend_prior(
    rows,
    kv_pool,
    block_table,
    q_nope_all,
    q_rope_all,
    mask_threshold,
    lower_threshold,
    slot_offset,
    loop_row,
    state,
    softmax,
    tokens,
    q_rows,
    local_blocks,
    heads,
    use_loop,
    bitwise,
    tag,
    sb,
    lay,
    n_mask=1,
    ts=False,
    loop_group_cap=0,
    clamp_pages=True,
    weights_on_act=False,
    first_static=False,
    scores_ahead=False,
    folded_reference=False,
    first_static_after=None,
    rows_v_stationary=False,
    dr_rows=None,
    rows_prefetch=False,
    key_path=None,
    pair_lag=False,
    pads=None,
    seq_hoist=False,
    setup_ahead=False,
    mx_rows=None,
    p2_state=None,
    preload=False,
    pair_tail=False,
    unroll=1,
    bf16_pages=False,
    page_gather="off",
    tail_groups=0,
    stack_rows=False,
    row_pipeline=False,
    prior_keys=None,
):
    """Attend the prior cache rows (position below ``position[b]``) of this row chunk into ``state``.

    The chunk is the global rows ``rows``; ``q_nope_all`` holds the chunk's gathered queries at the local index of
    each row in ``rows``. The prior keys of sequence ``b`` are shared by its rows in the
    chunk; only the query differs. The static path loads every block of the sequence; the loop path has
    one loop per sequence over its key groups and reloads the keys per chunk.

    In bitwise mode ``mask_threshold`` is the per-row threshold ``position[b] + j - 128 t``, the current
    block is attended as a prior group (the current-block read follows the owner-write through the kv_pool
    tensor order), and the mask is rewritten per row because each row of a sequence has its own threshold.
    With ``n_mask`` mask rows (fused masks), ``mask_threshold[j, b]`` is the threshold of query row ``j`` on mask
    row ``j``, and each key group writes the masks of all rows at once. With ``ts`` (transposed scores), the rows
    of a sequence attend each group together (``_attend_group_rows``). ``key_path``: the latent key form
    (``_KeyPath``; None = BF16). ``p2_state`` (``loop_p2``, the loop path with transposed scores): ``(state, seqs)``,
    the (row, head) state ``[128, G seqs + 1, 513]`` of the pass and its sequences; sequence ``s`` of the pass adds into
    its columns ``g seqs + s`` (``_attend_group_rows``), and ``state`` is unused. ``page_gather``
    (``loop_page_gather``): the page loads of the plain loop path (``_PageGather``). ``tail_groups``
    (``loop_tail_groups``): the tail iterations of each loop of the plain loop path (``_attend_prior_loop``).
    ``stack_rows``: the static path attends each key group for 2 (or 4) rows of a sequence at once
    (``_attend_group_stacked``; the draft form only, see ``mla_block_attention``).
    """
    kernel_assert(p2_state is None or (use_loop and ts), "loop_p2 is a form of the transposed-scores loop path")
    kernel_assert(
        page_gather == "off" or (use_loop and not ts and not rows_prefetch and not seq_hoist and unroll == 1),
        "loop_page_gather is a form of the plain loop path (no transposed scores, rows_prefetch, rows_seq_hoist or "
        "rows_unroll2)",
    )
    kernel_assert(
        ts or not (rows_v_stationary or dr_rows is not None or rows_prefetch),
        "rows_v_stationary, rows_dr and rows_prefetch need the rows form (ts)",
    )
    kernel_assert(not rows_prefetch or use_loop, "rows_prefetch is a form of the loop path")
    kernel_assert(
        not pair_lag or (use_loop and rows_v_stationary and not scores_ahead),
        "rows_pair_lag is a form of the rows loop path with rows_v_stationary, not scores_ahead",
    )
    kernel_assert(not seq_hoist or (use_loop and ts), "rows_seq_hoist is a form of the rows loop path")
    kernel_assert(
        not setup_ahead or (use_loop and ts and folded_reference),
        "rows_setup_ahead is a form of the rows loop path with folded_reference",
    )
    key_path = key_path or _KeyPath()
    sequences = [b for b in range(tokens) if any(r % tokens == b for r in rows)]
    prefetch = None
    if rows_prefetch:
        loop_groups = _loop_groups(kv_pool, local_blocks // _GROUP_BLOCKS, loop_group_cap)
        # unroll 2 reads the table in rows of 2 iterations: one more zero row keeps the row count even.
        prefetch = _RowsPrefetch(
            kv_pool, block_table, local_blocks, loop_groups, sequences, tag, sb, pad_rows=2 if unroll == 2 else 1
        )
        prefetch.load(sequences[0])
    gather = None
    gather_scope = ExitStack()
    if page_gather != "off" and bf16_pages:
        # The BF16 gather tiles free before the output tail: they and the tail's frame do not fit the region together.
        gather_scope.enter_context(sb.scope("prior_gather"))
    if page_gather != "off":
        # The gather tiles and the first indices of every sequence, before the first loop (they outlive each loop).
        gather = _PageGather(
            page_gather,
            kv_pool,
            block_table,
            local_blocks,
            _loop_groups(kv_pool, local_blocks // _GROUP_BLOCKS, loop_group_cap),
            sequences,
            tag,
            sb,
            page_dtype=nl.bfloat16 if bf16_pages else None,
        )
    with gather_scope, sb.scope("prior_hoist") if seq_hoist else nullcontext():
        hoisted = None
        if seq_hoist:
            hoisted = _rows_seq_setup(
                kv_pool,
                _loop_groups(kv_pool, local_blocks // _GROUP_BLOCKS, loop_group_cap),
                0 if prefetch is None else prefetch.groups,
                key_path,
                folded_reference,
                _BIAS_ROW + n_mask,
                sb,
            )
        ahead = None
        if setup_ahead:
            # Every sequence's reference columns (freed after the fold), folded query rows and query tile.
            ahead = {}
            for b in sequences:
                cqs = [cq for cq, r in enumerate(rows) if r % tokens == b]
                with sb.scope("prior_ahead_reference"):
                    reference = _reference_columns(softmax, cqs, heads, sb)
                    _fold_reference(q_rope_all, reference, cqs, heads, softmax.scale, sb)
                ahead[b] = key_path.query_tile(dr_rows, cqs, heads, sb, mx_rows)
        _attend_prior_sequences(
            rows,
            kv_pool,
            block_table,
            q_nope_all,
            q_rope_all,
            mask_threshold,
            lower_threshold,
            slot_offset,
            loop_row,
            state,
            softmax,
            tokens,
            q_rows,
            local_blocks,
            heads,
            use_loop,
            bitwise,
            tag,
            sb,
            lay,
            n_mask,
            ts,
            loop_group_cap,
            clamp_pages,
            weights_on_act,
            first_static,
            scores_ahead,
            folded_reference,
            first_static_after,
            rows_v_stationary,
            dr_rows,
            key_path,
            pair_lag,
            pads,
            sequences,
            prefetch,
            hoisted,
            ahead,
            mx_rows,
            p2_state=p2_state,
            preload=preload,
            pair_tail=pair_tail,
            unroll=unroll,
            bf16_pages=bf16_pages,
            gather=gather,
            tail_groups=tail_groups,
            stack_rows=stack_rows,
            row_pipeline=row_pipeline,
            prior_keys=prior_keys,
        )


def _prior_block_ids(block_table, b, local_blocks, num_pages, tag, sb, engine=None):
    """Return the ``[1, local_blocks]`` INT32 clamped block ids of sequence ``b`` (the static prior-key path)."""
    block_ids = sb.tile((1, local_blocks), nl.int32)
    nisa.dma_copy(
        dst=block_ids,
        src=block_table.ap(pattern=[[local_blocks, 1], [1, local_blocks]], offset=b * local_blocks),
        dge_mode=nisa.dge_mode.hwdge,
        engine=nisa.engine.sync if engine is None else engine,
        name=f"{tag}_mla_prior_table_{b}",
    )
    nisa.tensor_scalar(
        dst=block_ids, data=block_ids, op0=nl.minimum, operand0=float(num_pages - 1), op1=nl.maximum, operand1=0.0
    )
    return block_ids


def _prior_group_keys(
    b,
    g,
    gb,
    kv_pool,
    block_ids,
    mask_threshold,
    lower_threshold,
    slot_offset,
    thr_row,
    lay,
    n_mask,
    bias_end,
    key_path,
    tag,
    sb,
    engine=None,
):
    """Load and transpose key group ``g`` (``gb`` blocks) of sequence ``b`` (the static prior-key path); return the
    ``(key_rows, latent_keys, rope_keys)`` tiles of ``_transpose_group``."""
    fp32 = nl.float32
    group_keys = gb * _BLOCK
    key_rows = sb.tile((_P_MAX, gb, _CACHE_WIDTH), kv_pool.dtype)
    for lb in range(gb):
        entry_block = g * _GROUP_BLOCKS + lb
        nisa.dma_copy(
            dst=key_rows[0:_P_MAX, lb, 0:_CACHE_WIDTH],
            src=kv_pool.ap(
                pattern=[[_CACHE_WIDTH, _BLOCK], [1, _CACHE_WIDTH]],
                offset=0,
                scalar_offset=block_ids[0:1, entry_block : entry_block + 1],
                indirect_dim=0,
            ),
            dge_mode=nisa.dge_mode.hwdge,
            engine=nisa.engine.sync if engine is None else engine,
            name=f"{tag}_mla_prior_page_{b}_{g}_{lb}",
        )
        _sanitize_cache_rows(key_rows[0:_P_MAX, lb, 0:_CACHE_WIDTH])
    latent_keys = key_path.key_tile(group_keys, sb)
    rope_keys = sb.tile((_P_MAX, group_keys), nl.bfloat16)
    threshold = sb.tile((_P_MAX, 1), fp32)
    nisa.tensor_scalar(
        dst=threshold[_BIAS_ROW:bias_end, 0:1],
        data=mask_threshold[_BIAS_ROW:bias_end, thr_row : thr_row + 1],
        op0=nl.subtract,
        operand0=float(g * lay.group_stride),
    )
    lower = None
    if lower_threshold is not None:
        lower = sb.tile((_P_MAX, 1), fp32)
        nisa.tensor_scalar(
            dst=lower[_BIAS_ROW:bias_end, 0:1],
            data=lower_threshold[_BIAS_ROW:bias_end, b : b + 1],
            op0=nl.subtract,
            operand0=float(g * lay.group_stride),
        )
        lower = lower[_BIAS_ROW:bias_end, 0:1]
    _transpose_group(
        key_rows,
        latent_keys,
        rope_keys,
        slot_offset,
        threshold[_BIAS_ROW:bias_end, 0:1],
        lower,
        sb,
        n_mask,
        key_path=key_path,
    )
    return key_rows, latent_keys, rope_keys


def _hoist_prior_keys(
    rows,
    kv_pool,
    block_table,
    mask_threshold,
    lower_threshold,
    slot_offset,
    tokens,
    local_blocks,
    n_hoist,
    bitwise,
    lay,
    n_mask,
    key_path,
    tag,
    sb,
):
    """``prior_hoist``: the block ids and the first ``n_hoist`` key groups of each sequence of the pass rows ``rows``
    (the loads and transposes of ``_attend_prior_sequences``'s static path), for its ``prior_keys``. The page loads take
    the ScalarE HWDGE queue, so they do not wait behind the pass's query loads on the SP queue."""
    kernel_assert(sb.region is None, "prior_hoist is a form of the compiler-placed (AUTO_SBUF) static path")
    num_pages = kv_pool.shape[0]
    bias_end = _BIAS_ROW + n_mask
    hoisted_ids, hoisted = {}, {}
    for b in range(tokens):
        hoist_rows = [(cq, r) for cq, r in enumerate(rows) if r % tokens == b]
        if not hoist_rows:
            continue
        hoisted_ids[b] = _prior_block_ids(block_table, b, local_blocks, num_pages, tag, sb, engine=nisa.engine.scalar)
        thr_row = b if n_mask > 1 or not bitwise else hoist_rows[0][1]
        for g in range(min(n_hoist, (local_blocks + _GROUP_BLOCKS - 1) // _GROUP_BLOCKS)):
            hoisted[(b, g)] = _prior_group_keys(
                b,
                g,
                min(_GROUP_BLOCKS, local_blocks - g * _GROUP_BLOCKS),
                kv_pool,
                hoisted_ids[b],
                mask_threshold,
                lower_threshold,
                slot_offset,
                thr_row,
                lay,
                n_mask,
                bias_end,
                key_path,
                tag,
                sb,
                engine=nisa.engine.scalar,
            )
    return hoisted_ids, hoisted


def _attend_prior_sequences(
    rows,
    kv_pool,
    block_table,
    q_nope_all,
    q_rope_all,
    mask_threshold,
    lower_threshold,
    slot_offset,
    loop_row,
    state,
    softmax,
    tokens,
    q_rows,
    local_blocks,
    heads,
    use_loop,
    bitwise,
    tag,
    sb,
    lay,
    n_mask,
    ts,
    loop_group_cap,
    clamp_pages,
    weights_on_act,
    first_static,
    scores_ahead,
    folded_reference,
    first_static_after,
    rows_v_stationary,
    dr_rows,
    key_path,
    pair_lag,
    pads,
    sequences,
    prefetch,
    hoisted,
    ahead=None,
    mx_rows=None,
    p2_state=None,
    preload=False,
    pair_tail=False,
    unroll=1,
    bf16_pages=False,
    gather=None,
    tail_groups=0,
    stack_rows=False,
    row_pipeline=False,
    prior_keys=None,
):
    """The sequence loop of ``_attend_prior``: one ``_attend_prior_loop`` for each sequence on the loop path, else the
    static attend of each sequence. ``hoisted``: None, or the ``_rows_seq_setup`` group buffers (``rows_seq_hoist``).
    ``ahead``: None, or the query tile of each sequence, made before the first loop with its folded reference rows
    (``rows_setup_ahead``)."""
    fp32 = nl.float32
    num_pages = kv_pool.shape[0]
    bias_end = _BIAS_ROW + n_mask
    # ``prior_hoist``: the block ids and the first key groups of each sequence, traced before the pass (``_hoist_prior_keys``).
    # Their own names: ``hoisted`` is the loop path's ``rows_seq_hoist`` group buffers (None when off), which an empty dict
    # here would turn into zero group buffers (no prior-key loop at all).
    prior_ids, prior_groups = prior_keys if prior_keys is not None else ({}, {})
    seq = 0
    for b in range(tokens):
        chunk_rows = [(cq, r, r // tokens) for cq, r in enumerate(rows) if r % tokens == b]
        if not chunk_rows:
            continue
        seq_p2 = None if p2_state is None else (p2_state[0], seq, p2_state[1])
        seq += 1
        if use_loop:
            # Scope each sequence's loop buffers so they free before the next sequence and the tail.
            with sb.scope("prior_seq"):
                _attend_prior_loop(
                    b,
                    chunk_rows,
                    kv_pool,
                    block_table,
                    loop_row,
                    q_nope_all,
                    q_rope_all,
                    mask_threshold,
                    slot_offset,
                    state,
                    softmax,
                    tokens,
                    q_rows,
                    local_blocks,
                    heads,
                    lower_threshold,
                    bitwise,
                    tag,
                    sb,
                    lay,
                    n_mask,
                    ts,
                    loop_group_cap,
                    clamp_pages,
                    weights_on_act,
                    first_static,
                    scores_ahead,
                    folded_reference,
                    first_static_after,
                    rows_v_stationary,
                    dr_rows,
                    prefetch,
                    _next_sequence(sequences, b),
                    key_path,
                    pair_lag,
                    pads,
                    hoisted,
                    None if ahead is None else (ahead[b],),
                    mx_rows,
                    p2=seq_p2,
                    preload=preload,
                    pair_tail=pair_tail,
                    unroll=unroll,
                    bf16_pages=bf16_pages,
                    gather=gather,
                    tail_groups=tail_groups,
                    row_pipeline=row_pipeline,
                )
            continue
        cqs = [cq for cq, _r, _j in chunk_rows]
        reference = _reference_columns(softmax, cqs, heads, sb) if ts else None
        row_sums = _new_row_sums(len(cqs), heads, sb) if ts else None
        state_t = _new_state_t(len(cqs), heads, sb) if rows_v_stationary else None
        q_tile = key_path.query_tile(dr_rows, cqs, heads, sb, mx_rows)
        units = _stack_units(cqs, q_nope_all, q_rope_all, softmax, heads, bias_end, sb) if stack_rows else None
        if units is not None:
            _stack_state_in(units, state, heads)
        block_ids = (
            prior_ids[b] if b in prior_ids else _prior_block_ids(block_table, b, local_blocks, num_pages, tag, sb)
        )
        # Bitwise reads the current block back from the pool. No explicit edge orders the read after the
        # owner-write: both the owner-write and the page load access kv_pool, and the compiler orders a
        # read after a write of the same tensor (indirect accesses serialize conservatively). An explicit
        # runtime edge on the owner-write is fragile (NCC_IXDR001 as the owner-writes schedule late), so
        # the kernel relies on the tensor order.
        n_groups = (local_blocks + _GROUP_BLOCKS - 1) // _GROUP_BLOCKS
        pending = []
        pair = _TS_GROUPS if sb.region is None else 1
        for g in range(n_groups):
            gb = min(_GROUP_BLOCKS, local_blocks - g * _GROUP_BLOCKS)
            group_keys = gb * _BLOCK
            with sb.scope("prior_group"):
                if (b, g) in prior_groups:
                    key_rows, latent_keys, rope_keys = prior_groups[(b, g)]
                else:
                    # The mask threshold of the first chunk row; bitwise rewrites it per row below. Fused masks take the
                    # thresholds of all rows of the sequence, one on each mask row.
                    thr_row = b if n_mask > 1 or not bitwise else chunk_rows[0][1]
                    key_rows, latent_keys, rope_keys = _prior_group_keys(
                        b,
                        g,
                        gb,
                        kv_pool,
                        block_ids,
                        mask_threshold,
                        lower_threshold,
                        slot_offset,
                        thr_row,
                        lay,
                        n_mask,
                        bias_end,
                        key_path,
                        tag,
                        sb,
                    )
                if ts:
                    # Compiler-placed tiles outlive the scope, so two groups attend together (fewer state adds).
                    pending.append((latent_keys, rope_keys, key_rows))
                    if len(pending) == pair or g == n_groups - 1:
                        _attend_group_rows(
                            cqs,
                            q_nope_all,
                            q_rope_all,
                            pending,
                            softmax,
                            state,
                            reference,
                            row_sums,
                            sb,
                            n_mask,
                            state_t=state_t,
                            key_path=key_path,
                            q_tile=q_tile,
                        )
                        pending = []
                    continue
                if units is not None:
                    for unit in units:
                        _attend_group_stacked(unit, latent_keys, rope_keys, key_rows, softmax, state, heads, sb, n_mask)
                    continue
                for cq, r, _j in chunk_rows:
                    if bitwise and n_mask == 1:
                        _row_mask(rope_keys, slot_offset, mask_threshold, r, g * lay.group_stride, group_keys, sb)
                    local_sum = sb.tile((heads, 1), fp32)
                    numerator_psum = _attend_group(
                        cq,
                        q_nope_all,
                        q_rope_all,
                        latent_keys,
                        rope_keys,
                        key_rows,
                        softmax,
                        local_sum,
                        heads,
                        sb,
                        n_mask,
                    )
                    softmax.update(state, cq, numerator_psum, local_sum)
        if units is not None:
            _stack_state_out(units, state, heads)
        if ts:
            _add_row_sums(state, row_sums, cqs, heads)
        if state_t is not None:
            _add_state_t(state, state_t, cqs, heads)


_OWN_ROW_QUADRANT = 32
"""The partition step of the current rows of a sequence in ``_own_rows_p2``: an operand of one partition starts at
partition 0, 32, 64 or 96, so a row and its mask sit at one of those partitions."""


def _own_rows_p2(cur_rows, rows, q_rows, tag, sb, keep=False):
    """``pass_rows_ahead``: load the current rows of each sequence of a pass for ``_own_key_state_p2``. Row ``j`` of a
    sequence goes to partition ``32 (j % 4)`` and the 512 columns of block ``j // 4`` of the sequence's tile, so one DMA
    moves 4 rows (rows evenly spaced in ``cur_rows``, the sequence-major passes; else one DMA a row): ``ceil(q_rows / 4)``
    DMAs a sequence in place of ``q_rows``. ``keep`` puts the tiles on the heap (the loads of the next pass). Return,
    for each sequence, the ``(row view [1, 512], partition)`` of each of its rows in row order, and the tiles in the
    order of their allocation, and the load DMAs in issue order."""
    bf16 = nl.bfloat16
    seqs = len(rows) // q_rows
    quads = _P_MAX // _OWN_ROW_QUADRANT
    blocks = -(-q_rows // quads)
    views, tiles, loads = [], [], []
    for seq in range(seqs):
        seq_rows = [rows[row * seqs + seq] for row in range(q_rows)]
        seq_tile = sb.tile((_P_MAX, blocks * _KV_LORA), bf16, keep=keep)
        tiles.append(seq_tile)
        for block in range(blocks):
            block_rows = seq_rows[block * quads : (block + 1) * quads]
            count = len(block_rows)
            step = block_rows[1] - block_rows[0] if count > 1 else 1
            columns = slice(block * _KV_LORA, (block + 1) * _KV_LORA)
            if step > 0 and all(later - row == step for row, later in zip(block_rows, block_rows[1:])):
                load = nisa.dma_copy(
                    dst=seq_tile[0 : _OWN_ROW_QUADRANT * (count - 1) + 1 : _OWN_ROW_QUADRANT, columns],
                    src=cur_rows[block_rows[0] : block_rows[-1] + 1 : step, 0:_KV_LORA],
                    dge_mode=nisa.dge_mode.hwdge,
                    engine=nisa.engine.sync,
                    name=f"{tag}_mla_own_rows_p2_{block_rows[0]}",
                )
                loads.append(_instruction(load))
                continue
            for index, r in enumerate(block_rows):
                partition = _OWN_ROW_QUADRANT * index
                load = nisa.dma_copy(
                    dst=seq_tile[partition : partition + 1, columns],
                    src=cur_rows[r : r + 1, 0:_KV_LORA],
                    dge_mode=nisa.dge_mode.hwdge,
                    engine=nisa.engine.sync,
                    name=f"{tag}_mla_own_row_p2_{r}",
                )
                loads.append(_instruction(load))
        placed = []
        for row in range(q_rows):
            partition = _OWN_ROW_QUADRANT * (row % quads)
            block = row // quads
            placed.append((seq_tile[partition : partition + 1, block * _KV_LORA : (block + 1) * _KV_LORA], partition))
        views.append(placed)
    return views, tiles, loads


_OWN_STACK_ROWS = 4
"""``own_state_stack``: the current rows of one period of ``pack_p2``. 4 rows of 96 heads fill 3 groups of 128 columns
(``tkg_mla_paged.pack_p2_ref_pieces``), so the runs of a group always come from the 4 rows of its period."""


def _own_stack_masks(weight, sb):
    """``own_state_stack``: the stationaries of one period of 4 rows, ``[4, 3 x 128]`` BF16. Partition ``p`` holds
    ``weight`` on the 96 columns of row ``p`` (columns ``96 p .. 96 p + 95`` of the 3 groups, group-major) and 0 on the
    others: the bytes of the run masks of ``_own_key_state_p2``. Every memset starts at partition 0: the weight of row
    ``p`` goes to partitions ``0 .. p``, then a zero memset clears partitions ``0 .. p - 1`` (a column belongs to one
    row). VectorE only."""
    vector = nisa.engine.vector
    width = 3 * _P_MAX
    row_width = width // _OWN_STACK_ROWS  # the 96 heads of one row
    masks = sb.tile((_OWN_STACK_ROWS, width), nl.bfloat16)
    nisa.memset(dst=masks[0:_OWN_STACK_ROWS, 0:width], value=0.0, engine=vector)
    for row in range(_OWN_STACK_ROWS):
        columns = slice(row * row_width, (row + 1) * row_width)
        nisa.memset(dst=masks[0 : row + 1, columns], value=weight, engine=vector)
        if row:
            nisa.memset(dst=masks[0:row, columns], value=0.0, engine=vector)
    return masks


def _own_rows_stack_p2(cur_rows, rows, q_rows, tag, sb, keep=False):
    """``own_state_stack``: load the current rows of each sequence of a pass for ``_own_key_state_p2`` ``stack``. Row
    ``j`` of a sequence goes to partition ``j % 4`` and the 512 columns of period ``j // 4`` of the sequence's tile
    ``[4, (Q / 4) x 512]`` BF16, so one DMA moves the 4 rows of a period (rows evenly spaced in ``cur_rows``; else one
    DMA a row). ``keep``: heap tiles (``own_rows_ahead``). Return the tile of each sequence and the loads."""
    bf16 = nl.bfloat16
    seqs = len(rows) // q_rows
    periods = q_rows // _OWN_STACK_ROWS
    tiles, loads = [], []
    for seq in range(seqs):
        seq_rows = [rows[row * seqs + seq] for row in range(q_rows)]
        seq_tile = (
            sb.tile((_OWN_STACK_ROWS, periods * _KV_LORA), bf16, name=f"{tag}_mla_own_rows_{seq}", keep=True)
            if keep
            else sb.tile((_OWN_STACK_ROWS, periods * _KV_LORA), bf16)
        )
        tiles.append(seq_tile)
        for period in range(periods):
            block_rows = seq_rows[period * _OWN_STACK_ROWS : (period + 1) * _OWN_STACK_ROWS]
            step = block_rows[1] - block_rows[0]
            columns = slice(period * _KV_LORA, (period + 1) * _KV_LORA)
            if step > 0 and all(later - row == step for row, later in zip(block_rows, block_rows[1:])):
                load = nisa.dma_copy(
                    dst=seq_tile[0:_OWN_STACK_ROWS, columns],
                    src=cur_rows[block_rows[0] : block_rows[-1] + 1 : step, 0:_KV_LORA],
                    dge_mode=nisa.dge_mode.hwdge,
                    engine=nisa.engine.sync,
                    name=f"{tag}_mla_own_row_p2_{block_rows[0]}",
                )
                loads.append(_instruction(load))
                continue
            for index, r in enumerate(block_rows):
                load = nisa.dma_copy(
                    dst=seq_tile[index : index + 1, columns],
                    src=cur_rows[r : r + 1, 0:_KV_LORA],
                    dge_mode=nisa.dge_mode.hwdge,
                    engine=nisa.engine.sync,
                    name=f"{tag}_mla_own_row_p2_{r}",
                )
                loads.append(_instruction(load))
    return tiles, loads


def _own_key_state_p2(
    state_flat,
    rows,
    cur_rows,
    q_rows,
    weight,
    tag,
    sb,
    after=(),
    own_rows=None,
    batched=False,
    stack=False,
    row_after=(),
    stack_rows=None,
):
    """``p2``: write the own-key state of every column of the pass into ``state_flat`` ``[128, (G S + 1) x 513]`` (the
    (row, head) layout of ``tkg_mla_paged._packed_key_loop`` ``p2``): column ``g S + s`` gets ``weight * current_row`` of
    the row of each of its partitions and the sum ``weight``, the values that ``_own_key`` writes for each head of the
    row. For each run of one row inside one group (``tkg_mla_paged.pack_p2_runs``) the one-partition matmul of ``_own_key``
    (``weight`` on the stationary columns of the run, 0 on the others; the current row on partition 0) accumulates into
    the group's PSUM tile, so each column gets the one product of its run plus exact zeros from the other run of the group
    (the current rows are finite): the bits of ``_own_key``. One copy of all 128 partitions moves a group to the state (an
    engine access of 96 partitions must start at partition 0, so a copy of each run alone is not legal). ``after`` (the
    state store DMAs of the pass before, whose source bytes this pass's state tile takes) gives every write of the state a
    runtime dependency on each of them: a wait on their completion. ``row_after`` (the same stores) orders the
    current-row loads here: their stack tiles can take the bytes of the stores' source (the BF16 copy of
    ``_store_state_p2_bf16_o``, freed when it returns), and one SP queue alone does not order a load's write after a
    store's read. The first load waits for the landing of each store (the stores count on several DMA queue
    semaphores), and each later load issues after the one before it on SP, behind those waits (a scheduling edge, no
    wait). With ``stack`` the row loads of ``_own_rows_stack_p2`` take the same order. ``own_rows`` (``_own_rows_p2``,
    loaded by the pass before with ``pass_rows_ahead``) gives the current rows of each sequence, in place of their
    loads here; ``batched`` loads them here with ``_own_rows_p2`` (4 rows a DMA). With either, row ``j`` and the masks
    of its runs sit at partition ``32 (j % 4)``: each matmul still contracts over one partition with the same
    operands, so the bits are those of the loads at partition 0. ``stack`` (``packed_prior_options``
    ``own_state_stack``; no ``own_rows`` or ``batched``): one matmul for each group, over the 4 current rows of its
    period (``_own_rows_stack_p2``, rows at partitions 0 .. 3) with the stationary of the period's masks
    (``_own_stack_masks``). Each column gets the one product of its row plus exact zeros (the current rows are finite).
    The copy to the state adds +0.0: a nonzero value keeps its bits, and a sum that rounds to -0 (an underflowed
    negative product) becomes +0, the value of the per-run PSUM accumulation. G passes in place of one for each run (12
    in place of 24 for 2 sequences of 8 rows). ``stack_rows`` (with ``stack``; ``own_rows_ahead``): the tiles of
    ``_own_rows_stack_p2`` loaded before the passes, in place of the loads here."""
    bf16 = nl.bfloat16
    fp32 = nl.float32
    vector = nisa.engine.vector
    groups = pack_p2_groups(q_rows)
    seqs = len(rows) // q_rows
    state = state_flat.reshape((_P_MAX, groups * seqs + 1, _STATE))
    runs = pack_p2_runs(q_rows)
    sb.open_scope("own_p2")
    previous_load = []

    def ordered_load(load):
        """``row_after``: the first current-row load waits for every store; each later one follows the one before."""
        if row_after:
            if previous_load:
                load.depends_on(previous_load[-1], kind=nisa.dependency.schedule)
            else:
                load.depends_on(*row_after, kind=nisa.dependency.runtime)
            previous_load.append(load)

    if stack:
        kernel_assert(
            own_rows is None and not batched and q_rows % _OWN_STACK_ROWS == 0,
            "own_state_stack loads its own rows: use no pass_rows_ahead rows, and Q a multiple of 4",
        )
        masks = _own_stack_masks(weight, sb)
        # The rows take the frame bytes that the run masks leave to the current rows without ``stack`` (an SBUF
        # tile of 1 x 128 BF16 for each run), so each byte keeps the writer engine that it has without ``stack``.
        if stack_rows is None:
            gap = len(runs) * _P_MAX - 3 * _P_MAX
            if gap > 0:
                sb.tile((1, gap), bf16)
            row_tiles, loads = _own_rows_stack_p2(cur_rows, rows, q_rows, tag, sb)
            for load in loads:
                ordered_load(load)
        else:
            row_tiles = stack_rows
        for seq in range(seqs):
            for group in range(groups):
                period, piece = divmod(group, 3)
                numerator_psum = nl.ndarray((_P_MAX, _KV_LORA), dtype=fp32, buffer=nl.psum)
                nisa.nc_matmul(
                    dst=numerator_psum,
                    stationary=masks[0:_OWN_STACK_ROWS, piece * _P_MAX : (piece + 1) * _P_MAX],
                    moving=row_tiles[seq][0:_OWN_STACK_ROWS, period * _KV_LORA : (period + 1) * _KV_LORA],
                    accumulate=False,
                )
                # x + 0.0 keeps every nonzero value and turns -0 into +0: a sum of the group matmul that is an
                # underflowed negative product rounds to -0, where the per-run PSUM accumulation gives +0.
                copy = nisa.tensor_scalar(
                    dst=state[0:_P_MAX, group * seqs + seq, 0:_KV_LORA],
                    data=numerator_psum,
                    op0=nl.add,
                    operand0=0.0,
                    engine=nisa.engine.scalar if (seq * groups + group) % 2 else vector,
                )
                if after:
                    _instruction(copy).depends_on(*after, kind=nisa.dependency.runtime)
        sums = nisa.memset(dst=state[0:_P_MAX, 0 : groups * seqs, _KV_LORA:_STATE], value=weight, engine=vector)
        if after:
            _instruction(sums).depends_on(*after, kind=nisa.dependency.runtime)
        sb.close_scope()
        return
    placed = own_rows is not None or batched
    masks = []
    for _group, run_row, _head, column, count in runs:
        # With the rows of ``_own_rows_p2``, the mask of a run sits at the partition of its row.
        part = _OWN_ROW_QUADRANT * (run_row % (_P_MAX // _OWN_ROW_QUADRANT)) if placed else 0
        mask = sb.tile((_P_MAX if placed else 1, _P_MAX), bf16)
        nisa.memset(dst=mask[part : part + 1, 0:_P_MAX], value=0.0, engine=vector)
        nisa.memset(dst=mask[part : part + 1, column : column + count], value=weight, engine=vector)
        masks.append((mask, part))
    if own_rows is None and batched:
        own_rows, _, loads = _own_rows_p2(cur_rows, rows, q_rows, tag, sb)
        for load in loads:
            ordered_load(load)
    for seq in range(seqs):
        row_tiles = [] if own_rows is None else own_rows[seq]
        for row in range(q_rows if own_rows is None else 0):
            r = rows[row * seqs + seq]
            row0 = sb.tile((1, _KV_LORA), bf16)
            load = nisa.dma_copy(
                dst=row0[0:1, 0:_KV_LORA],
                src=cur_rows[r : r + 1, 0:_KV_LORA],
                dge_mode=nisa.dge_mode.hwdge,
                engine=nisa.engine.sync,
                name=f"{tag}_mla_own_row_p2_{r}",
            )
            ordered_load(_instruction(load))
            row_tiles.append((row0[0:1, 0:_KV_LORA], 0))
        for group in range(groups):
            numerator_psum = nl.ndarray((_P_MAX, _KV_LORA), dtype=fp32, buffer=nl.psum)
            group_runs = [(run, mask) for run, mask in zip(runs, masks, strict=True) if run[0] == group]
            for index, ((_group, run_row, _head, _column, _count), (mask, part)) in enumerate(group_runs):
                row_view, row_part = row_tiles[run_row]
                kernel_assert(row_part == part, "a run's mask and its current row sit at one partition")
                nisa.nc_matmul(
                    dst=numerator_psum,
                    stationary=mask[part : part + 1, 0:_P_MAX],
                    moving=row_view,
                    accumulate=index > 0,
                )
            copy = nisa.tensor_copy(
                dst=state[0:_P_MAX, group * seqs + seq, 0:_KV_LORA],
                src=numerator_psum,
                engine=nisa.engine.scalar if (seq * groups + group) % 2 else vector,
            )
            if after:
                _instruction(copy).depends_on(*after, kind=nisa.dependency.runtime)
    sums = nisa.memset(dst=state[0:_P_MAX, 0 : groups * seqs, _KV_LORA:_STATE], value=weight, engine=vector)
    if after:
        _instruction(sums).depends_on(*after, kind=nisa.dependency.runtime)
    sb.close_scope()


def _own_key_state_select_p2(state_flat, rows, cur_rows, q_rows, weight, row_select, tag, sb):
    """``loop_p2`` with ``own_key_select``: write the own-key state of every column of the pass into the (row, head)
    state ``state_flat`` ``[128, (G S + 1) x 513]`` (``_own_key_state_p2``) with the one-hot matmul of ``_own_key``
    ``row_select``: for each group of 128 (row, head) columns, one matmul over all current rows whose stationary column of
    head h of row r holds ``weight`` on partition r and 0 on the others, the column that ``_own_key`` gives head h of row
    r. Each state column thus takes the products of its head in the same order: the bits of ``_own_key``."""
    bf16 = nl.bfloat16
    fp32 = nl.float32
    vector = nisa.engine.vector
    groups = pack_p2_groups(q_rows)
    seqs = len(rows) // q_rows
    state = state_flat.reshape((_P_MAX, groups * seqs + 1, _STATE))
    rows_all = row_select.shape[0]
    runs = pack_p2_runs(q_rows)
    sb.open_scope("own_select_p2")
    for seq in range(seqs):
        for group in range(groups):
            select = sb.tile((rows_all, _P_MAX), bf16)
            for _group, row, _head, column, count in (run for run in runs if run[0] == group):
                nisa.tensor_scalar(
                    dst=select[0:rows_all, column : column + count],
                    data=row_select[0:rows_all, 0:count],
                    op0=nl.equal,
                    operand0=float(rows[row * seqs + seq]),
                    op1=nl.multiply,
                    operand1=weight,
                    engine=vector,
                )
            numerator_psum = nl.ndarray((_P_MAX, _KV_LORA), dtype=fp32, buffer=nl.psum)
            nisa.nc_matmul(
                dst=numerator_psum, stationary=select[0:rows_all, 0:_P_MAX], moving=cur_rows[0:rows_all, 0:_KV_LORA]
            )
            nisa.tensor_copy(
                dst=state[0:_P_MAX, group * seqs + seq, 0:_KV_LORA],
                src=numerator_psum,
                engine=nisa.engine.scalar if (seq * groups + group) % 2 else vector,
            )
    nisa.memset(dst=state[0:_P_MAX, 0 : groups * seqs, _KV_LORA:_STATE], value=weight, engine=vector)
    sb.close_scope()


def _store_state_loop_p2(state_flat, state_hbm, rows, tr, q_rows, heads, tag, sb, dma_options=None):
    """``loop_p2`` (CP128): store the (row, head) state ``state_flat`` to the head-major HBM state of the pass
    (``_store_state_p2``), and rows ``heads ..`` of the pad heads (one for each rank at and above ``heads``) as the
    head-major state holds them: a zero numerator and a unit sum for each row."""
    pad_rows = state_hbm.shape[0] - heads
    if pad_rows:
        pad = sb.tile((pad_rows, _STATE), nl.float32)
        nisa.memset(dst=pad[0:pad_rows, 0:_KV_LORA], value=0.0, engine=nisa.engine.vector)
        nisa.memset(dst=pad[0:pad_rows, _KV_LORA:_STATE], value=1.0, engine=nisa.engine.vector)
        row_width = state_hbm.shape[1]
        nisa.dma_copy(
            dst=state_hbm.ap(pattern=[[row_width, pad_rows], [_STATE, tr], [1, _STATE]], offset=heads * row_width),
            src=pad.ap(pattern=[[_STATE, pad_rows], [0, tr], [1, _STATE]], offset=0),
            dge_mode=nisa.dge_mode.hwdge,
            engine=nisa.engine.sync,
            name=f"{tag}_mla_state_pad_store_p2",
            **(dma_options or {}),
        )
    return _store_state_p2(state_flat, state_hbm, rows, tr, 1, q_rows, tag, dma_options)


def _store_state_p2(state_flat, state_hbm, rows, rows_per_pass, passes, q_rows, tag, dma_options=None):
    """``p2``: store the (row, head) state ``state_flat`` ``[128, (G S + 1) x 513]`` of a pass to the head-major HBM state:
    the column of head h of chunk row i (row ``i // S`` of sequence ``i % S``; global row ``rows[i]``) goes to HBM row
    ``h * passes + rows[i] // rows_per_pass``, column ``rows[i] % rows_per_pass`` (``_store_rows_state``; one pass:
    ``_store_pass_state`` with ``rows`` the chunk rows). One DMA for each run of one row inside one group
    (``tkg_mla_paged.pack_p2_runs``) and each run of consecutive global rows in one pass: the same bytes as the store of
    the head-major state, in pieces of the S x 2 KB columns of a run (column g S + s of sequence s is next to s + 1).
    ``dma_options`` (the tail's ``crit_dma``) go to every piece."""
    groups = pack_p2_groups(q_rows)
    seqs = len(rows) // q_rows
    state = state_flat.reshape((_P_MAX, groups * seqs + 1, _STATE))
    row_width = state_hbm.shape[1]
    stores = []
    for group, row, head, column, count in pack_p2_runs(q_rows):
        seq = 0
        while seq < seqs:
            first_row = rows[row * seqs + seq]
            n = 1
            while seq + n < seqs and rows[row * seqs + seq + n] == first_row + n and (first_row + n) % rows_per_pass:
                n += 1
            pass_index, state_column = divmod(first_row, rows_per_pass)
            store = nisa.dma_copy(
                dst=state_hbm.ap(
                    pattern=[[passes * row_width, count], [_STATE, n], [1, _STATE]],
                    offset=head * passes * row_width + pass_index * row_width + state_column * _STATE,
                ),
                src=state[column : column + count, group * seqs + seq : group * seqs + seq + n, 0:_STATE],
                dge_mode=nisa.dge_mode.hwdge,
                engine=nisa.engine.sync,
                name=f"{tag}_mla_state_store_p2_{group}_{column}_{seq}",
                **(dma_options or {}),
            )
            stores.append(_instruction(store))
            seq += n
    return tuple(stores)


_SUM_BLOCK = 16
"""FP32 words of one ``state_sum_blocks`` block: 64 bytes, a whole line of the HBM sums tensor."""


def _sum_block_spacing(passes: int, seqs: int, q_rows: int) -> int:
    """``state_sum_blocks``: the word step of the (pass, sequence) values of a block, for ``passes`` passes of ``seqs``
    sequences of ``q_rows`` rows; ``j // 4`` of row j adds 0 .. Q / 4 - 1 to it.

    Example:
        >>> _sum_block_spacing(2, 2, 8), _sum_block_spacing(1, 2, 8)
        (4, 8)
    """
    spacing = _SUM_BLOCK // (passes * seqs)
    kernel_assert(
        q_rows % 4 == 0 and q_rows // 4 <= spacing,
        f"state_sum_blocks needs Q % 4 == 0 and the {passes} x {seqs} x {q_rows // 4} sums of a block in {_SUM_BLOCK} "
        "words",
    )
    return spacing


def _sum_block_word(j: int, pass_index: int, seq: int, passes: int, seqs: int, q_rows: int) -> int:
    """``state_sum_blocks``: the word of the sum of row ``j`` (of Q) of sequence ``seq`` of pass ``pass_index`` in its
    head's sums region: block ``j % 4`` (4 blocks a head), word ``(pass_index * seqs + seq) * spacing + j // 4``.

    Example:
        >>> [_sum_block_word(j, 0, 0, 2, 2, 8) for j in (0, 1, 4)], _sum_block_word(0, 1, 1, 2, 2, 8)
        ([0, 16, 1], 12)
    """
    spacing = _sum_block_spacing(passes, seqs, q_rows)
    return (j % 4) * _SUM_BLOCK + (pass_index * seqs + seq) * spacing + j // 4


def _sum_block_loads(
    sum_dst, pass_rows: list, tail_pass: int, rows: int, passes: int, q_rows: int, head_major: bool = False
):
    """``state_sum_blocks``: the dense sums loads of tail pass ``tail_pass`` (global rows ``16 tail_pass ..``, ``rows``
    of them) from the reduced blocks ``sum_dst`` ``[local heads x passes, columns]``: a function of the local head that
    returns ``(first row, count, source AP)`` for each run of rows whose words step evenly (``common.load_state_bf16_o``
    ``sum_aps``). ``pass_rows``: the global rows of each pass, row-major over (row j, sequence). ``head_major``: each
    AP takes the ``[1, n, 1]`` shape of a run of the head-major sums tile (one partition a head), not ``[n, 1]``."""
    seqs = len(pass_rows[0]) // q_rows
    where = {r: (p, i) for p, prow in enumerate(pass_rows) for i, r in enumerate(prow)}
    words = []
    for c in range(rows):
        p, i = where[tail_pass * _TAIL_ROWS + c]
        j, seq = divmod(i, seqs)
        words.append(_sum_block_word(j, p, seq, len(pass_rows), seqs, q_rows))
    runs = []
    c = 0
    while c < rows:
        n = 1
        step = words[c + 1] - words[c] if c + 1 < rows else 1
        while c + n < rows and words[c + n] - words[c + n - 1] == step:
            n += 1
        runs.append((c, n, words[c], step if n > 1 else 1))
        c += n
    head_words = passes * sum_dst.shape[1]

    def aps(h):
        return [
            (
                c0,
                n,
                sum_dst.ap(
                    pattern=([[head_words, 1]] if head_major else []) + [[step, n], [1, 1]], offset=h * head_words + w0
                ),
            )
            for c0, n, w0, step in runs
        ]

    return aps


def _sum_blocks_p2(state_flat, sums, sum_view, passes, seqs, q_rows, sum_blocks, sum_queue, tag):
    """``_store_state_p2_bf16_o`` ``sum_blocks``: the VectorE copy of a pass's sums into the call's block tile and, with
    ``"store"`` (the last pass), the block stores to the sums view ``sum_view`` on ``sum_queue``. Return the copy and the
    stores (a list)."""
    groups = pack_p2_groups(q_rows)
    sum_stride = sum_view.shape[1]
    kernel_assert(
        sums is not None and groups == 3 * (q_rows // 4),
        "state_sum_blocks needs the sums tensor of its own and Q a multiple of 4",
    )
    block = sum_blocks["tile"]
    spacing = _sum_block_spacing(sum_blocks["passes"], seqs, q_rows)
    head_words = passes * sum_stride
    kernel_assert(
        head_words >= 4 * _SUM_BLOCK, f"state_sum_blocks needs 64 sums words a head, the sums view has {head_words}"
    )
    # The sums of column (group 3 jj + slot, sequence) at word (pass seqs + sequence) spacing + jj of block slot.
    block_cols = 3 * _SUM_BLOCK
    sums_copy = nisa.tensor_copy(
        dst=block.ap(
            pattern=[[block_cols, _P_MAX], [1, q_rows // 4], [_SUM_BLOCK, 3], [spacing, seqs]],
            offset=sum_blocks["pass"] * seqs * spacing,
        ),
        src=state_flat.ap(
            pattern=[
                [state_flat.shape[1], _P_MAX],
                [3 * seqs * _STATE, q_rows // 4],
                [seqs * _STATE, 3],
                [_STATE, seqs],
            ],
            offset=_KV_LORA,
        ),
        engine=nisa.engine.vector,
    )
    block_stores = []
    if sum_blocks["store"]:
        for group, row, head, column, count in pack_p2_runs(q_rows):
            if group >= 3:
                # Rows 4 .. Q - 1 share the partitions and blocks of rows 0 .. 3.
                continue
            block_stores.append(
                _instruction(
                    nisa.dma_copy(
                        dst=sum_view.ap(
                            pattern=[[head_words, count], [1, _SUM_BLOCK]],
                            offset=head * head_words + row * _SUM_BLOCK,
                        ),
                        src=block[column : column + count, group, 0:_SUM_BLOCK],
                        name=f"{tag}_mla_state_store_p2b_{group}_{column}",
                        **sum_queue,
                    )
                )
            )
    return sums_copy, block_stores


def _store_sum_blocks_pre(
    state_flat,
    state_hbm,
    sums,
    rows,
    rows_per_pass,
    passes,
    q_rows,
    tag,
    sum_blocks,
    dma_options=None,
    sums_engine=None,
):
    """``state_sums_pre_if``: the sums copy and the block stores of the pass that stores the sum blocks, on the queue and
    the views of ``_store_state_p2_bf16_o`` (the same arguments; ``rows_per_pass`` None: the pair's own rows, as the
    single-pass store gives them), traced before the pass's capless check; the result is its ``sums_pre``."""
    queue = {"dge_mode": nisa.dge_mode.hwdge, "engine": nisa.engine.sync, **(dma_options or {})}
    sum_queue = queue if sums_engine is None else {**queue, "engine": sums_engine}
    _o_view, sum_view, view_rows = mla_state_bf16_o_views(state_hbm, _KV_LORA, sums=sums)
    kernel_assert(
        rows_per_pass is None or view_rows == rows_per_pass,
        f"the call state pair holds {view_rows} rows a pass, not {rows_per_pass}",
    )
    kernel_assert(sum_blocks["store"], "state_sums_pre_if traces the sum blocks of the pass that stores them")
    return _sum_blocks_p2(state_flat, sums, sum_view, passes, len(rows) // q_rows, q_rows, sum_blocks, sum_queue, tag)


def _store_state_p2_bf16_o(
    state_flat,
    state_hbm,
    sums,
    rows,
    rows_per_pass,
    passes,
    q_rows,
    sb,
    tag,
    dma_options=None,
    sums_first=None,
    sums_engine=None,
    o_engine=None,
    o_heap=False,
    out=None,
    sums_heap=False,
    sum_blocks=None,
    sums_pre=None,
):
    """``p2`` with ``state_bf16_o``: store the (row, head) state ``state_flat`` ``[128, (G S + 1) x 513]`` of a pass in the
    BF16-o form of the call-level state pair (``common.mla_state_bf16_o_views``; ``sums`` the sums tensor when the sums
    take a tensor of their own): the numerators of head h of chunk row i (global row ``rows[i]``) go, cast to BF16 on
    VectorE as ``common.store_state_bf16_o`` casts them, to tensor row ``h * passes + rows[i] // rows_per_pass``, column
    ``rows[i] % rows_per_pass`` of the numerator view, and their FP32 sum to the same row and column of the sums view:
    the words of ``_store_call_state_bf16_o`` for the head-major state. One DMA of each view for each run of one row
    inside one group (``tkg_mla_paged.pack_p2_runs``) and each run of consecutive global rows in one pass, on the SP
    hardware DGE queue (``dma_options``, if given, adds to it: a priority). ``sums_engine`` (``state_sums_act``) puts the
    sums DMAs on the hardware DGE queue of that engine in place of SP, ``o_engine`` (``state_o_act``) the numerator
    DMAs. ``o_heap`` (``state_o_heap``): the BF16 numerator tile is a heap tile (``SbufTiles.tile`` ``keep``), and
    ``out`` (a dict) gets it (``"o16"``) and the instructions that the next pass must wait for, the cast and the sum
    stores (``"next_waits"``). ``sums_heap`` (``state_sums_heap``, with ``o_heap``): the sum stores read a heap copy of the
    sums (``out["s32"]``, one VectorE copy right after the cast), so the next pass waits for the cast and that copy, not
    for the sum stores. ``sum_blocks`` (``state_sum_blocks``, a dict; not with ``sums_heap``): the sums go to the
    ``"tile"`` of the call's sum blocks ``[128, 3, 16]`` FP32 (one VectorE copy, at pass ``"pass"`` of ``"passes"``),
    and with ``"store"`` (the last pass) the blocks go to the sums view, one DMA for each run of the first 4 rows
    (``_sum_block_word``). ``sums_pre`` (``state_sums_pre_if``, with ``sum_blocks`` ``"store"``): the sums copy and the
    block stores of ``_sum_blocks_p2``, already traced before the capless check; they are not traced again. Return the
    store DMAs and the sums copy (the next pass's state tile takes the bytes of their sources)."""
    queue = {"dge_mode": nisa.dge_mode.hwdge, "engine": nisa.engine.sync, **(dma_options or {})}
    sum_queue = queue if sums_engine is None else {**queue, "engine": sums_engine}
    o_queue = queue if o_engine is None else {**queue, "engine": o_engine}
    groups = pack_p2_groups(q_rows)
    seqs = len(rows) // q_rows
    columns = groups * seqs
    state = state_flat.reshape((_P_MAX, columns + 1, _STATE))
    o_view, sum_view, view_rows = mla_state_bf16_o_views(state_hbm, _KV_LORA, sums=sums)
    kernel_assert(view_rows == rows_per_pass, f"the call state pair holds {view_rows} rows a pass, not {rows_per_pass}")
    sum_stride = sum_view.shape[1]
    kernel_assert(not sums_heap or o_heap, "_store_state_p2_bf16_o sums_heap goes with o_heap (state_o_heap)")
    kernel_assert(not (sums_heap and sum_blocks is not None), "_store_state_p2_bf16_o takes sums_heap or sum_blocks")
    o16 = sb.tile((_P_MAX, columns, _KV_LORA), nl.bfloat16, keep=o_heap)
    cast = nisa.tensor_copy(dst=o16, src=state[0:_P_MAX, 0:columns, 0:_KV_LORA], engine=nisa.engine.vector)
    s32 = None
    if sums_heap:
        # The sums, copied out of the state tile (the same FP32 words), so the next pass's tiles can take its bytes.
        s32 = sb.tile((_P_MAX, columns, 1), nl.float32, keep=True)
        sums_copy = nisa.tensor_copy(
            dst=s32, src=state[0:_P_MAX, 0:columns, _KV_LORA : _KV_LORA + 1], engine=nisa.engine.vector
        )
    sum_src = state if s32 is None else None
    runs = []
    for group, row, head, column, count in pack_p2_runs(q_rows):
        seq = 0
        while seq < seqs:
            first_row = rows[row * seqs + seq]
            n = 1
            while seq + n < seqs and rows[row * seqs + seq + n] == first_row + n and (first_row + n) % rows_per_pass:
                n += 1
            pass_index, state_column = divmod(first_row, rows_per_pass)
            runs.append((group, column, count, seq, n, head * passes + pass_index, state_column))
            seq += n

    def o_store(group, column, count, seq, n, tensor_row, state_column):
        return nisa.dma_copy(
            dst=o_view.ap(
                pattern=[[passes * view_rows * _KV_LORA, count], [_KV_LORA, n], [1, _KV_LORA]],
                offset=(tensor_row * view_rows + state_column) * _KV_LORA,
            ),
            src=o16[column : column + count, slice(group * seqs + seq, group * seqs + seq + n), 0:_KV_LORA],
            name=f"{tag}_mla_state_store_p2o_{group}_{column}_{seq}",
            **o_queue,
        )

    if sum_blocks is not None:
        kernel_assert(
            sums_pre is None or sum_blocks["store"],
            "state_sums_pre_if: only the pass that stores the sum blocks traces them before its capless check",
        )
        sums_copy, block_stores = (
            sums_pre
            if sums_pre is not None
            else _sum_blocks_p2(state_flat, sums, sum_view, passes, seqs, q_rows, sum_blocks, sum_queue, tag)
        )
        block_stores = list(block_stores)
        o_stores = [_instruction(o_store(*run)) for run in runs]
        stores = (
            block_stores + o_stores if (SUM_RS_FIRST if sums_first is None else sums_first) else o_stores + block_stores
        )
        if out is not None:
            out["o16"] = o16
            out["next_waits"] = (_instruction(cast), _instruction(sums_copy), *block_stores)
        return (_instruction(sums_copy), *stores)

    def sum_store(group, column, count, seq, n, tensor_row, state_column):
        return nisa.dma_copy(
            dst=sum_view.ap(
                pattern=[[passes * sum_stride, count], [1, n], [1, 1]],
                offset=tensor_row * sum_stride + state_column,
            ),
            src=(
                sum_src[
                    column : column + count, slice(group * seqs + seq, group * seqs + seq + n), _KV_LORA : _KV_LORA + 1
                ]
                if s32 is None
                else s32[column : column + count, slice(group * seqs + seq, group * seqs + seq + n), 0:1]
            ),
            name=f"{tag}_mla_state_store_p2s_{group}_{column}_{seq}",
            **sum_queue,
        )

    if SUM_RS_FIRST if sums_first is None else sums_first:
        # Every sums DMA first, so the sums land (and their reduce-scatter can start) before the numerators store.
        sum_stores = [_instruction(sum_store(*run)) for run in runs]
        stores = sum_stores + [_instruction(o_store(*run)) for run in runs]
    else:
        pairs = [(_instruction(o_store(*run)), _instruction(sum_store(*run))) for run in runs]
        stores = [store for pair in pairs for store in pair]
        sum_stores = [pair[1] for pair in pairs]
    if out is not None:
        out["o16"] = o16
        out["next_waits"] = (_instruction(cast), *sum_stores)
        if s32 is not None:
            out["s32"] = s32
            out["next_waits"] = (_instruction(cast), _instruction(sums_copy))
    return tuple(stores)


def _check_packed_softmax_scale(scale: float):
    """Refuse a softmax scale other than the packed loop's (``packed_prior``): the loop's exp takes
    ``tkg_mla_paged._SOFTMAX_SCALE`` while the reference and the cap of the rows come from ``scale``, so a call with another
    scale would mix the two. The served calls give no scale (``192**-0.5``, the same value). A scale within a relative
    1e-12 of the loop's is the same scale in another rounding (``1 / math.sqrt(192)`` is one ulp above ``192**-0.5``)
    and passes.

    Example:
        >>> _check_packed_softmax_scale(_Q_HEAD**-0.5) is None, _check_packed_softmax_scale(1 / math.sqrt(192)) is None
        (True, True)
    """
    kernel_assert(
        math.isclose(scale, _SOFTMAX_SCALE, rel_tol=1e-12),
        f"packed_prior takes the packed loop's softmax scale {_SOFTMAX_SCALE!r} (mla_decode._SOFTMAX_SCALE), got {scale!r}",
    )


_CAPLESS_SUM_LIMIT = math.exp(_WEIGHT_CAP - 1.0)
"""The largest weight sum of a column that keeps the loop without the score cap (e^59). A score above its cap gives a
weight of at least about ``exp(_WEIGHT_CAP)`` (on Trn3 the ScalarE exp is finite and within ~20 ulp of exp up to 88.72,
Inf from 88.73, NaN for NaN), and the terms of a sum are >= 0, so its column's sum is above this limit, Inf or NaN."""


_CAPLESS_REDO_COUNT_TEST = None
"""Test hook (None in a served program): an HBM ``[1, 1]`` int32 tensor. When set, the capless redo IF takes its register
straight from it (one DMA load, no compute) in place of the check's flag, so a checker that drives the inputs takes the
redo on any nonzero value; the check still runs."""


def _capless_scratch(state_flat, sb):
    """``capless``: the tiles of the check and of the redo's entry barrier, allocated before the first loop so that no
    instruction after it writes a byte that the loop's last work may still read."""
    fp32 = nl.float32
    parts = state_flat.shape[0]
    columns = state_flat.shape[1] // _STATE - 1
    return {
        "passed": sb.tile((parts, columns), fp32),
        "least": sb.tile((parts, 1), fp32),
        "flag": sb.tile((1, 1), fp32),
        "redo": sb.tile((1, 1), nl.int32),
        "pad": sb.tile((1, 8), fp32),
        **({"forced": sb.tile((1, 1), nl.int32)} if _CAPLESS_REDO_COUNT_TEST is not None else {}),
    }


def _capless_redo(state_flat, scratch, state_reinit, redo_loop, redo_tail=None, as_loop=False):
    """``capless``: after the loop without the score cap, check the weight sum of every column of ``state_flat`` (its
    sum column; not the null items' column) and redo the pass on the capped loop (the loop without ``capless``) when any
    sum is not at most ``_CAPLESS_SUM_LIMIT`` (a NaN sum fails the compare). While every sum passes, every weight was at
    most e^59, so every score was below its cap (weight e^60): the min was a no-op and the state is bitwise that of the
    capped loop. The check writes only the tiles of ``scratch`` (``_capless_scratch``). The redo is a forward then-only
    IF on the flag. Its entry is a full barrier: a chain of small instructions on VectorE, GpSimd, ScalarE and TensorE
    (each after all the loop's work in its queue), whose last, a VectorE copy into the state (which the re-init
    overwrites), each engine (VectorE, GpSimd, ScalarE, TensorE and the SP queue) then waits on before its first redo
    instruction. Then ``state_reinit(after=...)`` writes the state of the pass's start again and ``redo_loop(after)``
    traces the capped loop (its prologue loads after ``after`` too), whose state is that of the loop without
    ``capless``. ``redo_tail`` (``state_sums_pre_if``): traced at the end of the redo body, after the capped loop (the
    pass's sum blocks again, from the capped state).

    ``as_loop`` (``packed_prior_options`` ``no_if_loops``): the redo body is a loop of 0 or 1 trips on the same flag
    (``nl.fori_loop``), not an IF. The released neuronx-cc 2.28 has no runtime IF ([NCC_ICFG019]), but it compiles a
    loop with a runtime trip count. The body is the same, so the bits are the same. The released neuronx-cc 2.28 counts
    a runtime edge on a DMA producer twice in its dependency reduction pass (``tkg_weight_dma_gate.wait_for_completion``). So the
    loop form has no SP-queue DMA in the barrier: the redo instructions wait on the barrier (a VectorE instruction) and
    on the GpSimd, ScalarE and TensorE gates. The redo page loads on the SP queue wait on the barrier directly."""
    fp32 = nl.float32
    vector, gpsimd, scalar, tensor = nisa.engine.vector, nisa.engine.gpsimd, nisa.engine.scalar, nisa.engine.tensor
    parts = state_flat.shape[0]
    columns = state_flat.shape[1] // _STATE - 1
    sums = state_flat.reshape_dim(1, (columns + 1, _STATE))[0:parts, 0:columns, _KV_LORA]
    passed, least, flag, redo_i, pad = (scratch[k] for k in ("passed", "least", "flag", "redo", "pad"))
    nisa.tensor_scalar(dst=passed, data=sums, op0=nl.less_equal, operand0=_CAPLESS_SUM_LIMIT, engine=vector)
    nisa.tensor_reduce(dst=least, op=nl.minimum, data=passed, axis=1)
    # Across the partitions: a 32-bit TensorE transpose (exact) and a minimum on one partition.
    least_t = nl.ndarray((1, parts), dtype=fp32, buffer=nl.psum)
    nisa.nc_transpose(dst=least_t, data=least, engine=tensor)
    nisa.tensor_reduce(dst=flag, op=nl.minimum, data=least_t, axis=1)
    # The redo count, 1 - the least pass: 1 when a sum failed, else 0.
    nisa.tensor_scalar(dst=redo_i, data=flag, op0=nl.multiply, operand0=-1.0, op1=nl.add, operand1=1.0, engine=vector)
    redo = nisa.register_alloc()
    if _CAPLESS_REDO_COUNT_TEST is None:
        nisa.register_load(redo, redo_i[0:1, 0:1])
    else:
        # The gate twin: the IF register straight from the input cell (a DMA, no compute), so a simulator that drives
        # the inputs takes the then-only IF on any nonzero value.
        forced = scratch["forced"]
        nisa.dma_copy(dst=forced, src=_CAPLESS_REDO_COUNT_TEST, dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)
        nisa.register_load(redo, forced[0:1, 0:1])

    def body():
        # The chain moves its value with tensor_scalar adds of 0, not copies, so the compiler's copy elimination leaves it
        # alone.
        nisa.memset(dst=pad[0:1, 0:1], value=0.0, engine=vector)
        nisa.tensor_scalar(dst=pad[0:1, 1:2], data=pad[0:1, 0:1], op0=nl.add, operand0=0.0, engine=gpsimd)
        nisa.tensor_scalar(dst=pad[0:1, 2:3], data=pad[0:1, 1:2], op0=nl.add, operand0=0.0, engine=scalar)
        pe_mark = nl.ndarray((1, 1), dtype=fp32, buffer=nl.psum)
        nisa.nc_transpose(dst=pe_mark, data=pad[0:1, 2:3], engine=tensor)
        barrier = nisa.tensor_scalar(dst=state_flat[0:1, 0:1], data=pe_mark, op0=nl.add, operand0=0.0, engine=vector)
        # Each engine's queue waits on the barrier before its redo work.
        gates = [
            nisa.memset(dst=pad[0:1, 3:4], value=0.0, engine=gpsimd),
            nisa.tensor_scalar(dst=pad[0:1, 4:5], data=pad[0:1, 3:4], op0=nl.add, operand0=0.0, engine=scalar),
        ]
        if not as_loop:
            gates.append(
                nisa.dma_copy(
                    dst=pad[0:1, 5:6], src=pad[0:1, 4:5], dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync
                )
            )
        pe_gate = nl.ndarray((1, 1), dtype=fp32, buffer=nl.psum)
        gates.append(nisa.nc_transpose(dst=pe_gate, data=pad[0:1, 4:5] if as_loop else pad[0:1, 5:6], engine=tensor))
        for gate in gates:
            _instruction(gate).depends_on(barrier, kind=nisa.dependency.runtime)
        after = (barrier, *gates)
        state_reinit(after=after)
        redo_loop(after)
        if redo_tail is not None:
            redo_tail()

    if as_loop:
        # A loop of 0 or 1 trips on the flag register (the body does not read the induction value).
        nl.fori_loop(0, redo, lambda _trip: body())
        return
    # A forward then-only IF on the flag register (no loop: no induction register, nothing on the not-taken path past
    # the branch).
    nke.if_else(redo, body)


def check_no_if_loops_options(options: dict):
    """Refuse ``packed_prior_options`` ``no_if_loops`` without ``no_if``: ``no_if_loops`` changes only the form of the
    no-IF loop (the capless loop with a redo loop of 0 or 1 trips, in place of the capped loop alone).

    Example:
        >>> check_no_if_loops_options({"no_if": True, "no_if_loops": True}) is None
        True
    """
    kernel_assert(
        isinstance(options.get("no_if_loops", False), bool),
        f"packed_prior_options no_if_loops is True or False, got {options.get('no_if_loops')!r}",
    )
    kernel_assert(
        not options.get("no_if_loops") or options.get("no_if"),
        "packed_prior_options no_if_loops is a form of no_if (the redo as a loop of 0 or 1 trips): set no_if too",
    )


def check_fp8_pv_options(options: dict, q_rows: int, split: bool = False):
    """Refuse ``packed_prior_options`` ``fp8_pv`` without the form it reads: the p2 sub-slots of Q > 1 rows, the
    "block32" MX latent query words (``mx_qk``) in the natural grouping (``mx_natural``, the grouping of the key words of
    an FP8 key row as it is), on Trn3 (the MX matmuls); and with ``split`` (a split group range of the pass): the pass
    builds its items from the positions only, so it would attend the groups that the split's helpers attend.

    Example:
        >>> check_fp8_pv_options({"fp8_pv": False}, 1) is None
        True
    """
    if not options.get("fp8_pv"):
        return
    kernel_assert(
        not split,
        "packed_prior_options fp8_pv does not take the long-sequence split (mla_block_attention split): turn one off",
    )
    kernel_assert(
        options.get("fp8_pv_plan_first", False) in (False, True),
        f"packed_prior_options fp8_pv_plan_first is a bool, got {options.get('fp8_pv_plan_first')!r}",
    )
    kernel_assert(
        options.get("p2") and options.get("mx_qk") == "block32" and options.get("mx_natural") and q_rows > 1,
        "packed_prior_options fp8_pv needs p2, mx_qk 'block32' and mx_natural with Q > 1 rows, got "
        f"p2={options.get('p2')!r}, mx_qk={options.get('mx_qk')!r}, mx_natural={options.get('mx_natural')!r}, "
        f"Q={q_rows}",
    )
    kernel_assert(is_trn3(), "packed_prior_options fp8_pv runs MX matmuls: Trn3 only")


def check_fp8_pv_batched_options(options: dict):
    """Refuse an ``fp8_pv_batched`` or ``fp8_pv_batched_exact`` that is not a bool, both True together, and either
    True without ``no_if`` and its FP8 P@V stream ("batched" or "exact": ``verify_model_2srv.served_pack_options`` gives
    that stream to ``no_if``).

    Example:
        >>> check_fp8_pv_batched_options({"fp8_pv_batched": False, "fp8_pv_batched_exact": False}) is None
        True
    """
    kernel_assert(
        not (options.get("fp8_pv_batched") and options.get("fp8_pv_batched_exact")),
        "packed_prior_options fp8_pv_batched and fp8_pv_batched_exact are both true. They are two forms of the FP8 P@V "
        "stream. Set one of them to false.",
    )
    for key, stream in (("fp8_pv_batched", "batched"), ("fp8_pv_batched_exact", "exact")):
        value = options.get(key, False)
        kernel_assert(isinstance(value, bool), f"packed_prior_options {key} is a bool, got {value!r}")
        kernel_assert(
            not value or (options.get("no_if") and options.get("fp8_pv_stream") == stream),
            f"packed_prior_options {key} needs no_if and fp8_pv_stream {stream!r} (served_pack_options), got no_if "
            f"{options.get('no_if')!r} and fp8_pv_stream {options.get('fp8_pv_stream')!r}",
        )


def check_fp8_pv_loop_options(options: dict):
    """Refuse ``fp8_pv_value_form`` / ``fp8_pv_page_runs`` / ``fp8_pv_stage_free`` / ``fp8_pv_clamp_split`` off their
    defaults without ``fp8_pv``, values the loop does not take, and ``fp8_pv_page_runs`` without the ``page_runs`` 2
    table contract.

    Example:
        >>> check_fp8_pv_loop_options({"fp8_pv": False, "fp8_pv_value_form": 0, "fp8_pv_page_runs": 0}) is None
        True
    """
    form = options.get("fp8_pv_value_form", 0)
    runs = options.get("fp8_pv_page_runs", 0)
    kernel_assert(
        options.get("fp8_pv")
        or (
            form == 0
            and not runs
            and not options.get("fp8_pv_stage_free")
            and not options.get("fp8_pv_clamp_split")
            and not options.get("fp8_pv_cap_pool")
            and not options.get("fp8_pv_prep_late")
            and options.get("fp8_pv_page_buffers", 2) == 2
            and not options.get("fp8_pv_step_store")
            and not options.get("fp8_pv_redo_rolled")
            and not options.get("fp8_pv_rope_copy_vector")
        ),
        "packed_prior_options fp8_pv_value_form / fp8_pv_page_runs / fp8_pv_stage_free / fp8_pv_clamp_split / "
        "fp8_pv_cap_pool / fp8_pv_prep_late / fp8_pv_page_buffers / fp8_pv_step_store / fp8_pv_redo_rolled / "
        "fp8_pv_rope_copy_vector change the FP8 P@V pass: they need fp8_pv",
    )
    kernel_assert(
        isinstance(options.get("fp8_pv_rope_copy_vector", False), bool),
        f"packed_prior_options fp8_pv_rope_copy_vector is True or False, got {options.get('fp8_pv_rope_copy_vector')!r}",
    )
    kernel_assert(
        isinstance(options.get("fp8_pv_redo_rolled", False), bool),
        f"packed_prior_options fp8_pv_redo_rolled is True or False, got {options.get('fp8_pv_redo_rolled')!r}",
    )
    kernel_assert(
        options.get("fp8_pv_page_buffers", 2) in (2, 3),
        f"packed_prior_options fp8_pv_page_buffers takes 2 or 3, got {options.get('fp8_pv_page_buffers')!r}",
    )
    kernel_assert(
        form in (0, 4, 5) and runs in (0, 2),
        f"packed_prior_options fp8_pv_value_form takes 0, 4 or 5 and fp8_pv_page_runs 0 or 2, got {form} / {runs}",
    )
    kernel_assert(
        options.get("fp8_pv_null_skip", False) in (False, True, 2),
        f"packed_prior_options fp8_pv_null_skip is False, True or 2, got {options.get('fp8_pv_null_skip')!r}",
    )
    kernel_assert(
        not options.get("fp8_pv_null_skip") or options.get("fp8_pv"),
        "packed_prior_options fp8_pv_null_skip changes the FP8 P@V chunk: set fp8_pv",
    )
    kernel_assert(
        options.get("fp8_pv_plan_reduce", False) in (False, True),
        f"packed_prior_options fp8_pv_plan_reduce is a bool, got {options.get('fp8_pv_plan_reduce')!r}",
    )
    kernel_assert(
        not options.get("fp8_pv_plan_reduce") or options.get("fp8_pv"),
        "packed_prior_options fp8_pv_plan_reduce changes the FP8 P@V item plan: set fp8_pv",
    )
    kernel_assert(
        not runs or options.get("page_runs") == 2,
        "packed_prior_options fp8_pv_page_runs 2 takes the table contract of page_runs 2: set page_runs 2",
    )
    kernel_assert(
        not options.get("fp8_pv_pages_first") or options.get("fp8_pv_stage_free"),
        "packed_prior_options fp8_pv_pages_first is a stage-free setup form: set fp8_pv_stage_free",
    )
    kernel_assert(
        not options.get("fp8_pv_refs_after_q")
        or (options.get("fp8_pv") and options.get("fp8_pv_stage_free") and options.get("q_mx_gather")),
        "packed_prior_options fp8_pv_refs_after_q orders the stage-free setup's reference DMAs after the MX query loads: "
        "set fp8_pv, fp8_pv_stage_free and q_mx_gather",
    )
    kernel_assert(
        not options.get("own_state_first")
        or (options.get("own_rows_ahead") and options.get("own_state_stack") and options.get("p2")),
        "packed_prior_options own_state_first traces the own-key state of the rows loaded ahead first: set "
        "own_rows_ahead, own_state_stack and p2",
    )
    kernel_assert(
        not options.get("q_mx_scale_words") or options.get("q_mx_gather"),
        "packed_prior_options q_mx_scale_words packs the MX query scales into the word records of the q_mx_gather pair: "
        "set q_mx_gather",
    )
    kernel_assert(
        not options.get("fp8_pv_refs_act") or (options.get("fp8_pv") and options.get("fp8_pv_stage_free")),
        "packed_prior_options fp8_pv_refs_act puts the stage-free setup's reference DMAs on the ScalarE queue: set fp8_pv "
        "and fp8_pv_stage_free",
    )
    kernel_assert(
        not options.get("fp8_pv_tables_ahead")
        or (options.get("fp8_pv") and options.get("fp8_pv_pages_first") and options.get("fp8_pv_step_store")),
        "packed_prior_options fp8_pv_tables_ahead reads the step stores of the pages-first FP8 P@V pass: set fp8_pv, "
        "fp8_pv_pages_first and fp8_pv_step_store",
    )


def _attend_prior_packed(
    rows,
    kv_pool,
    block_table,
    position,
    rank_info,
    q_nope_all,
    q_rope_all,
    slot_offset,
    state_flat,
    softmax,
    tokens,
    q_rows,
    local_blocks,
    tag,
    sb,
    cache_writes,
    options,
    clamp_pages=True,
    dr_rows=None,
    hoisted=None,
    live_rows=None,
    mx_rows=None,
    sum_batch=True,
    state_reinit=None,
    split_info=None,
    split_lo=None,
    setup_after=(),
    query_words=None,
    loop_after=(),
    next_step=None,
    tables_in=None,
    refs_sched_after=(),
    sums_pre_if=None,
):
    """Attend the prior keys of the pass rows ``rows`` with one packed loop (``mla_block_attention`` ``packed_prior``).

    The pass holds whole sequences: ``rows`` are the rows ``j B + b`` of the sequences ``b0 .. b0 + S - 1`` for each
    query row j, in that order (chunk index ``j S + (b - b0)``), as one pass of up to 16 rows or a sequence-major pass of
    ``fit_passes`` has them. The S sequences of the pass are the sequences of ``tkg_mla_paged._packed_key_loop`` with
    ``queries`` Q: its state column ``j S + (b - b0)`` is the chunk row of row j of sequence b. Its items count the key
    groups of a sequence from ``position[b] + Q - 1``, so they hold the keys of every row, and a null item masks every
    key of every row (``_pack_items`` ``group_extra`` and ``null_threshold``). The row mask of row j is the mask at
    ``position[b] + j`` of the decode step j. ``state_flat`` ``[96, (Q S + 1) x 513]`` holds the own-key state of each
    row, and column Q S takes the null items. The first item row and the page loads of the prologue wait for every owner
    write of the call (``cache_writes``): the rows of the current block that a row reads were written by this call.
    ``options`` is the packed-loop form (``_PACKED_PRIOR_OPTIONS``). With its ``dr_keys``, ``dr_rows`` holds the e4m3
    double_row stationaries ``[128, Q S, 2, 2, 96]`` FP8 of the pass rows (``tkg_mla_paged._dr_gather_tile_queries``); with
    its ``mx_keys``, ``mx_rows`` holds the MX query words ``[128, Q S, 1, 96]`` FP32 of the pass rows (``_mx_query_words``)
    instead. ``hoisted`` (``mla_block_attention`` ``pass_plans``): the plan of this pass, built once for the MLA layers of the
    step, and the tiles the passes share; with ``options`` ``step_items`` also the step store of the item rows of this
    pass (``step``). With its ``mx_natural``, ``q_nope_all`` holds the latent query tiles ``[128, Q S, 4, 96]`` BF16 of
    the pass rows in the natural MX grouping (``_mx_natural_query_tiles``).
    ``split_info`` (``mla_block_attention`` ``split``): the plan of the pass takes the group range of the
    split slot B - 1 (``_split_group_range``); a hoisted plan was built with it already. ``split_lo`` (split v5): the
    ``[1, B]`` INT32 HBM lo_eff of the step check; sequence b of the pass takes its groups lo_eff[b] .. (0 = every
    group).
    ``setup_after`` (with ``fp8_pv_stage_free``): the state stores of the pass before, which the stage-free setup's
    reference DMAs wait for (the tiles of this pass's setup may take the bytes of their sources).
    ``query_words`` (``q_mx_gather``, with ``fp8_pv``): the MX query words and scales of the pass (``_q_mx_load``);
    ``q_nope_all`` is then None. ``loop_after`` (with ``query_words``): the first page loads of the loop (not the capless
    redo's) wait for these instructions too: the state stores of the pass before.
    ``next_step`` (``fp8_pv_tables_ahead``): the step store of the next pass; the pass loads its chunk tables now and the
    pages of its first chunk items 0 and 1 at the tail of its own last chunk, and returns those heap tiles
    (``fp8_pv_tables_ahead``; None in the first MLA layer of the step, before the store is filled). ``tables_in``: the
    tiles that the pass before returned, for this pass's setup and first chunk; the caller releases them after this
    pass's loops. ``refs_sched_after`` (``fp8_pv_refs_after_q``): the pass's MX query loads, which the setup's reference
    DMAs follow in the schedule (``fp8_pv_setup`` ``refs_sched_after``). ``sums_pre_if`` (``state_sums_pre_if``, a dict):
    in the capless form, ``sums_pre_if["trace"](suffix)`` traces the pass's sum blocks right after the uncapped loop (its
    result goes to ``sums_pre_if["out"]``) and again at the end of the redo body.
    """
    seqs = len(rows) // q_rows
    first = rows[0] % tokens
    kernel_assert(
        len(rows) == q_rows * seqs
        and first + seqs <= tokens
        and list(rows) == [j * tokens + first + s for j in range(q_rows) for s in range(seqs)],
        "packed_prior needs passes of whole sequences (one pass of up to 16 rows, or the sequence-major passes of "
        f"fit_passes), got the pass rows {list(rows)} of {tokens} sequences",
    )
    table = block_table if seqs == tokens else block_table[first : first + seqs, 0:local_blocks]
    kernel_assert(
        not (options["step_items"] or options["fp8_pv_step_store"]) or (hoisted is not None and "step" in hoisted),
        "packed_prior_options step_items / fp8_pv_step_store needs the step store of the pass",
    )
    kernel_assert(hoisted is None or split_lo is None, "split v5 owner ranges need the per-pass plans (no hoist_plans)")
    check_fp8_pv_loop_options(options)
    check_fp8_pv_batched_options(options)
    kernel_assert(query_words is None or options.get("fp8_pv"), "query_words (q_mx_gather) needs fp8_pv")
    if options.get("fp8_pv"):
        check_fp8_pv_options(options, q_rows, split=split_lo is not None or split_info is not None)
        with sb.scope("prior_fp8_pv"):
            # What every loop of the pass reads, built once: the redo below then holds only its loop, which reads no
            # kernel input but the pages (an IF block takes no static DMA of an input).
            # ``fp8_pv_tables_ahead``: the next pass's chunk tables now, in heap tiles older than this pass's setup heap
            # (the next pass releases them after its loops).
            ahead = (
                fp8_pv_tables_ahead(next_step, seqs, q_rows, tag, sb)
                if options.get("fp8_pv_tables_ahead") and next_step is not None
                else None
            )
            if ahead is not None and _TABLES_AHEAD_TEST is not None:
                _TABLES_AHEAD_TEST.append(tag)
            setup = fp8_pv_setup(
                kv_pool,
                table,
                position,
                rank_info,
                q_nope_all,
                q_rope_all,
                softmax,
                seqs,
                q_rows,
                first,
                tag,
                sb,
                page_runs=options["fp8_pv_page_runs"],
                step=hoisted["step"] if options["step_items"] or options["fp8_pv_step_store"] else None,
                plan_first=bool(options.get("fp8_pv_plan_first")),
                **({"stage_free": True, "after": tuple(setup_after)} if options.get("fp8_pv_stage_free") else {}),
                **({"query_words": query_words} if query_words is not None else {}),
                **({"pages_first": True} if options.get("fp8_pv_pages_first") else {}),
                **({"tables": tables_in} if tables_in is not None else {}),
                **({"refs_sched_after": tuple(refs_sched_after)} if refs_sched_after else {}),
                **({"refs_engine": nisa.engine.scalar} if options.get("fp8_pv_refs_act") else {}),
                **({"null_last": True} if options.get("fp8_pv_null_skip") else {}),
                **({"plan_reduce": True} if options.get("fp8_pv_plan_reduce") else {}),
                # The "batched" stream reads rows past the stream of a short sequence: they hold full key groups.
                **({"pad_longest": True} if options["fp8_pv_stream"] == "batched" else {}),
            )

            def fp8_pv_pass(pass_tag, capped, extra_after=(), next_chunk=None, redo=False):
                """Trace the loop of the FP8 P@V pass (``capped``: with the served score cap) in a stack scope of its
                own, so a redo reuses the bytes of the loop before. ``redo`` with ``fp8_pv_redo_rolled``: the stream
                as one loop of 4-item bodies, without the lookahead, the trips ahead, the chunk-ahead pages and the
                chunks-first order (they change no bit). A null skip of the merged form takes its two-IF form."""
                rolled = redo and options.get("fp8_pv_redo_rolled", False)
                stream_options = (
                    {"lookahead": 0, "trips_ahead": False, "chunk_ahead": False, "stream": "rolled"}
                    if rolled
                    else {
                        "lookahead": options["fp8_pv_lookahead"],
                        "trips_ahead": options["fp8_pv_trips_ahead"],
                        "chunk_ahead": options["fp8_pv_chunk_ahead"],
                        "stream": options["fp8_pv_stream"],
                    }
                )
                with sb.scope(f"{pass_tag}_fp8_pv_loop"):
                    fp8_pv_loop(
                        kv_pool,
                        setup,
                        state_flat,
                        seqs,
                        q_rows,
                        pass_tag,
                        sb,
                        capped,
                        clamp_pages,
                        # The redo (``extra_after``: its entry barrier) is ordered after the stores already.
                        tuple(cache_writes) + (tuple(extra_after) if extra_after else tuple(loop_after)),
                        **stream_options,
                        qk_group=options["fp8_pv_qk_group"],
                        **({"null_skip": options["fp8_pv_null_skip"]} if options.get("fp8_pv_null_skip") else {}),
                        value_form=options["fp8_pv_value_form"],
                        page_runs=options["fp8_pv_page_runs"],
                        **({"clamp_split": True} if options.get("fp8_pv_clamp_split") else {}),
                        **({"cap_pool": True} if options.get("fp8_pv_cap_pool") else {}),
                        **({"prep_late": True} if options.get("fp8_pv_prep_late") else {}),
                        **({"page_buffers": 3} if options.get("fp8_pv_page_buffers", 2) == 3 else {}),
                        **({"first_carry": tuple(tables_in["bufs"])} if tables_in is not None else {}),
                        **({"next_chunk": next_chunk} if next_chunk is not None else {}),
                        **({"chunks_first": True} if options.get("fp8_pv_chunks_first") and not rolled else {}),
                        **({"rope_copy_vector": True} if options.get("fp8_pv_rope_copy_vector") else {}),
                    )

            # The capless form of the packed loop takes the pass too: without the cap, then the check of the weight
            # sums and, when one is not at most e^59, the state again from its start and the loop with the cap.
            if (
                options["p2"]
                and options["word_keys"]
                and not options["mx_keys"]
                and not options["dr_keys"]
                and state_reinit is not None
            ):
                scratch = _capless_scratch(state_flat, sb)
                # ``fp8_pv_tables_ahead``: only the first loop loads the next pass's first pages (the redo's IF does not).
                fp8_pv_pass(tag, False, next_chunk=ahead)
                if sums_pre_if is not None:
                    # ``state_sums_pre_if``: the pass's sum blocks before the check, and again after a capped redo.
                    sums_pre_if["out"] = sums_pre_if["trace"]("")
                _capless_redo(
                    state_flat,
                    scratch,
                    state_reinit,
                    lambda after: fp8_pv_pass(f"{tag}r", True, after, redo=True),
                    redo_tail=None if sums_pre_if is None else (lambda: sums_pre_if["trace"]("r")),
                    as_loop=options.get("no_if_loops", False),
                )
            else:
                fp8_pv_pass(tag, True, next_chunk=ahead)
        # ``fp8_pv_pages_first``: the setup's heap tiles (the chunk entries and thresholds) return after the loops.
        sb.release(setup["heap"])
        if _DEBUG_ROW_SUMS is not None:
            # Test hook: the weight sums of the pass's columns after the pass (the margin of the capless check).
            columns, heads = state_flat.shape[1] // _STATE - 1, state_flat.shape[0]
            sums_hbm = nl.ndarray((heads, columns), dtype=nl.float32, buffer=nl.shared_hbm)
            nisa.dma_copy(
                dst=sums_hbm,
                src=state_flat.reshape_dim(1, (columns + 1, _STATE))[0:heads, 0:columns, _KV_LORA],
                dge_mode=nisa.dge_mode.hwdge,
                engine=nisa.engine.sync,
            )
            _DEBUG_ROW_SUMS.append((tag, sums_hbm))
        return ahead
    with sb.scope("prior_packed"):
        if hoisted is None:
            plan = _pass_plan(
                position,
                rank_info,
                table,
                first,
                seqs,
                q_rows,
                options,
                tag,
                sb,
                live_rows=live_rows,
                split_info=split_info,
                total_seqs=tokens,
                split_lo=split_lo,
            )
            share = {}
        else:
            plan = hoisted["plan"]
            share = {"pack_sb": hoisted["pack_sb"], "shared": hoisted["shared"], "pack_hbm": hoisted["pack_hbm"]}
            if options["step_items"]:
                share["step_items"] = hoisted["step"]
        # cap_first and the pinned page clamp, on by default, apply to Q > 1 rows (the pin: Q >= 3 with the page clamp),
        # also with p2 (the same instructions in a new order on each engine, so the bits are the same); elsewhere the
        # loop keeps its program.
        cap_first = options["cap_first"] and q_rows > 1
        clamp_pin = options["clamp_pin"] and clamp_pages and q_rows >= 3
        deep_pipe = options["deep_pipe"] and q_rows > 1 and not options["p2"]
        # The batched weight-sum adds (bitwise the same) wherever the loop takes them; ``sum_batch`` False (the
        # options key, a kill switch, or the test hook) keeps one sum add a sub-slot.
        sum_batch = sum_batch and options["sum_batch"] and q_rows > 1

        def trace_loop(loop_tag, capless_now, extra_after=()):
            """Trace the packed loop of the pass (``capless_now``: without the score cap). With the capless check each loop
            takes a stack scope of its own, so the redo reuses the bytes of the loop before."""
            if capless:
                with sb.scope(f"{loop_tag}_loop"):
                    trace_loop_now(loop_tag, capless_now, extra_after)
            else:
                trace_loop_now(loop_tag, capless_now, extra_after)

        def trace_loop_now(loop_tag, capless_now, extra_after):
            _packed_key_loop(
                kv_pool,
                table,
                None,
                None,
                q_nope_all,
                q_rope_all,
                softmax,
                slot_offset,
                state_flat,
                seqs,
                local_blocks,
                loop_tag,
                sb,
                plan,
                clamp_pages=clamp_pages,
                prefetch=options["prefetch"],
                page_runs=options["page_runs"],
                null_copy_vector=options["null_copy_vector"],
                null_runs=options["null_runs"],
                key_transpose="transpose",
                latent_act=options["latent_act"],
                mask_column=True,
                bg_rope=True,
                bg_interleave=True,
                carry=True,
                carry_main_only=options["carry_main_only"],
                sum_after_copy=options["sum_after_copy"],
                queries=q_rows,
                after_writes=tuple(cache_writes) + tuple(extra_after),
                **({"dr_keys": True, "dr_q_rows": dr_rows} if options["dr_keys"] and not options["mx_keys"] else {}),
                **(
                    {"mx_keys": "single", "mx_q_rows": mx_rows.reshape((_P_MAX, len(rows), mx_rows.shape[3]))}
                    if options["mx_keys"]
                    else {}
                ),
                **({"item_rope_ahead": True} if options["item_rope_ahead"] else {}),
                **({"p2": True} if options["p2"] else {}),
                **({"psum_per_block": True} if options["psum_per_block"] else {}),
                **({"stage_rows_sbuf": True} if options["stage_rows_sbuf"] else {}),
                **({"cap_first": True} if cap_first else {}),
                **({"clamp_pin": True} if clamp_pin else {}),
                **({"deep_pipe": True} if deep_pipe else {}),
                **({"sum_batch": True} if sum_batch else {}),
                **({"numerator_ring": options["numerator_ring"]} if options["numerator_ring"] else {}),
                **({"row_depth": options["row_depth"]} if options["row_depth"] else {}),
                **({"word_keys": True} if options["word_keys"] else {}),
                **({"slot_rows_sbuf": True} if options["slot_rows_sbuf"] else {}),
                **({"keys_ahead": True} if options["word_keys"] and options["keys_ahead"] else {}),
                **({"mx_qk": options["mx_qk"]} if options["mx_qk"] else {}),
                **({"p2_entry_act": True} if options["p2_entry_act"] else {}),
                **({"setup_off_pool": True} if options["setup_off_pool"] else {}),
                **({"skip_stage_store": True} if options["skip_stage_store"] else {}),
                **({"mx_natural": True} if options["mx_natural"] else {}),
                **({"act_spread": True} if options["act_spread"] else {}),
                **({"capless": True} if capless_now else {}),
                **({"kv_pitch576": True} if options["kv_pitch576"] else {}),
                **({"kv_gather": True} if options["kv_gather"] else {}),
                **({"kv_rem4": True} if options["kv_rem4"] else {}),
                **({"kv_group": options["kv_group"]} if options["kv_group"] else {}),
                **share,
            )

        capless = (
            options["p2"]
            and options["word_keys"]
            and not options["mx_keys"]
            and not options["dr_keys"]
            and state_reinit is not None
        )
        if capless:
            # The loop without the score cap, then the check of the weight sums and, when a sum is not at most e^59 (a
            # score above its cap, or Inf / NaN), the state again from its start and the capped loop (``_capless_redo``).
            scratch = _capless_scratch(state_flat, sb)
            trace_loop(tag, True)
            _capless_redo(
                state_flat,
                scratch,
                state_reinit,
                lambda after: trace_loop(f"{tag}r", False, after),
                as_loop=options.get("no_if_loops", False),
            )
        else:
            trace_loop(tag, False)
        if _DEBUG_ROW_SUMS is not None:
            # Test hook: the weight sums of the pass's columns after the loop (the margin of the capless check).
            # The state's columns (the last takes the null items).
            columns, heads = state_flat.shape[1] // _STATE - 1, state_flat.shape[0]
            sums_hbm = nl.ndarray((heads, columns), dtype=nl.float32, buffer=nl.shared_hbm)
            nisa.dma_copy(
                dst=sums_hbm,
                src=state_flat.reshape_dim(1, (columns + 1, _STATE))[0:heads, 0:columns, _KV_LORA],
                dge_mode=nisa.dge_mode.hwdge,
                engine=nisa.engine.sync,
            )
            _DEBUG_ROW_SUMS.append((tag, sums_hbm))


SPLIT_TAIL_SLOTS = 2
"""Slots that the long-sequence split appends to a replica's batch (``mla_block_attention`` ``split``): the foreign slot
B' - 2 (the helper's copy of the partner's donor sequence) and a pad slot B' - 1 (whole sequence-major passes)."""


def _split_group_range(split_info, first, seqs, total_seqs, local_blocks, tag, sb):
    """Return the FP32 ``[seqs, 2]`` group range (lo, hi) of the sequences ``first .. first + seqs - 1`` of a pass for
    the long-sequence split (``mla_block_attention`` ``split``) of a batch of ``total_seqs`` = B' slots. ``split_info``
    ``[1, 4]`` INT32 HBM is (role, X, 0, 0) of this replica, X a multiple of 16384:
    - the donor slot B' - 3 keeps groups X / 16384 .. on an owner (role 1), every group otherwise;
    - the foreign slot B' - 2 takes groups 0 .. X / 16384 - 1 on a helper (role 2), no group otherwise;
    - the pad slot B' - 1 takes no group;
    - every other slot keeps every group.
    Small integers, exact in FP32."""
    fp32 = nl.float32
    int32 = nl.int32
    vector = nisa.engine.vector
    groups = float(local_blocks // _GROUP_BLOCKS)
    info_i = sb.tile((seqs, 4), int32)
    nisa.dma_copy(
        dst=info_i,
        src=split_info[0:1, 0:4].broadcast(0, seqs),
        dge_mode=nisa.dge_mode.hwdge,
        engine=nisa.engine.sync,
        name=f"{tag}_mla_split_info_load",
    )
    info_f = sb.tile((seqs, 4), fp32)
    nisa.tensor_copy(dst=info_f, src=info_i, engine=vector)
    seq_i = sb.tile((seqs, 1), int32)
    nisa.iota(dst=seq_i, pattern=[[0, 1]], offset=first, channel_multiplier=1)
    seq_f = sb.tile((seqs, 1), fp32)
    nisa.tensor_copy(dst=seq_f, src=seq_i, engine=vector)

    def slot_is(slot):
        flag = sb.tile((seqs, 1), fp32)
        nisa.tensor_scalar(dst=flag, data=seq_f, op0=nl.equal, operand0=float(slot), engine=vector)
        return flag

    def role_and(flag, role):
        out = sb.tile((seqs, 1), fp32)
        nisa.tensor_scalar(dst=out, data=info_f[0:seqs, 0:1], op0=nl.equal, operand0=float(role), engine=vector)
        nisa.tensor_tensor(dst=out, data1=out, data2=flag, op=nl.multiply, engine=vector)
        return out

    donor = role_and(slot_is(total_seqs - 3), 1)
    foreign = slot_is(total_seqs - 2)
    helper = role_and(foreign, 2)
    pad = slot_is(total_seqs - 1)
    x_groups = sb.tile((seqs, 1), fp32)
    nisa.tensor_scalar(
        dst=x_groups, data=info_f[0:seqs, 1:2], op0=nl.multiply, operand0=1.0 / _GROUP_STRIDE, engine=vector
    )
    group_range = sb.tile((seqs, 2), fp32)
    # lo = donor-of-an-owner * X / 16384.
    nisa.tensor_tensor(dst=group_range[0:seqs, 0:1], data1=donor, data2=x_groups, op=nl.multiply, engine=vector)
    # hi = groups * (1 - foreign - pad) + helper * X / 16384 (the foreign and the pad slots are never both).
    keep = sb.tile((seqs, 1), fp32)
    nisa.tensor_tensor(dst=keep, data1=foreign, data2=pad, op=nl.add, engine=vector)
    nisa.tensor_scalar(
        dst=keep, data=keep, op0=nl.multiply, operand0=-groups, op1=nl.add, operand1=groups, engine=vector
    )
    nisa.tensor_tensor(dst=group_range[0:seqs, 1:2], data1=helper, data2=x_groups, op=nl.multiply, engine=vector)
    nisa.tensor_tensor(
        dst=group_range[0:seqs, 1:2], data1=group_range[0:seqs, 1:2], data2=keep, op=nl.add, engine=vector
    )
    return group_range


def _split_role_flags(split_info, parts, tag, sb):
    """Return the FP32 ``[parts, 1]`` (owner, helper) flags of ``split_info`` (role 1 / role 2), on every partition."""
    fp32 = nl.float32
    vector = nisa.engine.vector
    info_i = sb.tile((parts, 4), nl.int32)
    nisa.dma_copy(
        dst=info_i,
        src=split_info[0:1, 0:4].broadcast(0, parts),
        dge_mode=nisa.dge_mode.hwdge,
        engine=nisa.engine.sync,
        name=f"{tag}_mla_split_role_load",
    )
    info_f = sb.tile((parts, 4), fp32)
    nisa.tensor_copy(dst=info_f, src=info_i, engine=vector)
    owner = sb.tile((parts, 1), fp32)
    nisa.tensor_scalar(dst=owner, data=info_f[0:parts, 0:1], op0=nl.equal, operand0=1.0, engine=vector)
    helper = sb.tile((parts, 1), fp32)
    nisa.tensor_scalar(dst=helper, data=info_f[0:parts, 0:1], op0=nl.equal, operand0=2.0, engine=vector)
    return owner, helper


def _split_row_mask(parts, tokens, slot, flag, width, dtype, tag, sb):
    """Return a ``[parts, width]`` ``dtype`` mask: ``flag`` (``[parts, 1]`` FP32, 0 / 1) on the partitions p with
    ``p % tokens == slot`` (``tokens`` a power of two), 0 on the others."""
    fp32 = nl.float32
    vector = nisa.engine.vector
    part_i = sb.tile((parts, 1), nl.int32)
    nisa.iota(dst=part_i, pattern=[[0, 1]], offset=0, channel_multiplier=1)
    nisa.tensor_scalar(dst=part_i, data=part_i, op0=nl.bitwise_and, operand0=tokens - 1)
    part_f = sb.tile((parts, 1), fp32)
    nisa.tensor_copy(dst=part_f, src=part_i, engine=vector)
    nisa.tensor_scalar(
        dst=part_f, data=part_f, op0=nl.equal, operand0=float(slot), op1=nl.multiply, operand1=flag, engine=vector
    )
    ones = sb.tile((parts, width), fp32)
    nisa.memset(dst=ones, value=0.0, engine=vector)
    nisa.tensor_scalar(dst=ones, data=ones, op0=nl.add, operand0=part_f, engine=vector)
    mask = sb.tile((parts, width), dtype)
    nisa.tensor_copy(dst=mask, src=ones, engine=vector)
    return mask


def _split_v5_lo(split):
    """Return the ``split_lo`` of the split v5 owner ranges: ``(tensor, stride)``, the step check's ``range_ext``
    ``[1, B', 2]`` (lo, hi) of the extended batch (stride 2), or its ``lo_mine`` ``[1, B]`` (stride 1); None before the
    step check (no checks)."""
    checks = split.get("checks") or {}
    if split.get("ext") is not None and "range_ext" in checks:
        return checks["range_ext"], 2
    return (checks["lo_mine"], 1) if "lo_mine" in checks else None


def _split_v5_group_range(split_lo, first, seqs, local_blocks, tag, sb):
    """Return the FP32 ``[seqs, 2]`` group range (lo, hi) of the sequences ``first ..`` of a pass for the split v5
    ranges: ``split_lo`` = ``(tensor, stride)`` (``_split_v5_lo``). Stride 1: the ``[1, B]`` INT32 HBM lo_eff of this
    replica's slots (in key groups; 0 = every group, the items without the split), with hi the group count. Stride 2:
    the ``[1, B', 2]`` (lo, hi) of each slot of the extended batch. Small integers, exact in FP32."""
    lo_hbm, stride = split_lo
    lo_i = sb.tile((seqs, stride), nl.int32)
    nisa.dma_copy(
        dst=lo_i,
        src=lo_hbm.ap(pattern=[[stride, seqs], [1, stride]], offset=first * stride),
        dge_mode=nisa.dge_mode.hwdge,
        engine=nisa.engine.sync,
        name=f"{tag}_mla_split_lo_load",
    )
    group_range = sb.tile((seqs, 2), nl.float32)
    nisa.tensor_copy(dst=group_range[0:seqs, 0:stride], src=lo_i, engine=nisa.engine.vector)
    if stride == 1:
        nisa.memset(dst=group_range[0:seqs, 1:2], value=float(local_blocks // _GROUP_BLOCKS), engine=nisa.engine.vector)
    return group_range


def _split_v5_if(flag, then_branch, name, sb):
    """Split v5 "one" (``split`` ``guards``): run ``then_branch`` in a CC-free runtime then-only IF on the INT32 ``[1, 1]``
    ``flag`` (HBM or SBUF; nonzero = taken), so a step with no split skips it. The flag is a verdict of the step check
    on the gathered records, the same on every rank of a DP group."""
    flag_sb = sb.tile((1, 1), nl.int32)
    nisa.dma_copy(dst=flag_sb, src=flag, name=f"{name}_flag_load")
    reg = nisa.register_alloc()
    nisa.register_load(reg, flag_sb)
    nke.if_else(reg, then_branch)


def _split_v5_split_mine(split, tag, sb):
    """Split v5 "one": the INT32 ``[1, 1]`` SBUF flag of a confirmed split owned by this replica (any of the step check's
    ``lo_mine`` ``[1, B]`` above 0): the guard of the owner merge."""
    B = split["ext"]
    lo_i = sb.tile((1, B), nl.int32)
    nisa.dma_copy(dst=lo_i, src=split["checks"]["lo_mine"].reshape((1, B)), name=f"{tag}_mla_split_lo_mine")
    lo_f = sb.tile((1, B), nl.float32)
    nisa.tensor_copy(dst=lo_f, src=lo_i, engine=nisa.engine.vector)
    top = sb.tile((1, 1), nl.float32)
    nisa.tensor_reduce(dst=top, op=nl.maximum, data=lo_f, axis=1)
    nisa.tensor_scalar(dst=top, data=top, op0=nl.greater, operand0=0.0, engine=nisa.engine.vector)
    flag = sb.tile((1, 1), nl.int32)
    nisa.tensor_copy(dst=flag, src=top, engine=nisa.engine.vector)
    return flag


def _split_v5_pass_mine(split, first, count, tag, sb):
    """Split v5 "one": the INT32 ``[1, 1]`` SBUF flag of a foreign pass with an active piece (any active word of the
    step check's ``pieces_mine`` ``[1, F, 5]`` in the foreign slots ``first .. first + count - 1``): the guard of that
    pass."""
    F = split["foreign_slots"]
    act_i = sb.tile((1, count), nl.int32)
    nisa.dma_copy(
        dst=act_i,
        src=split["checks"]["pieces_mine"].ap(pattern=[[5 * F, 1], [5, count]], offset=5 * first),
        name=f"{tag}_mla_split_pass_active",
    )
    act_f = sb.tile((1, count), nl.float32)
    nisa.tensor_copy(dst=act_f, src=act_i, engine=nisa.engine.vector)
    top = sb.tile((1, 1), nl.float32)
    nisa.tensor_reduce(dst=top, op=nl.maximum, data=act_f, axis=1)
    nisa.tensor_scalar(dst=top, data=top, op0=nl.greater, operand0=0.0, engine=nisa.engine.vector)
    flag = sb.tile((1, 1), nl.int32)
    nisa.tensor_copy(dst=flag, src=top, engine=nisa.engine.vector)
    return flag


def _split_v5_ag_kwargs(split):
    """The ``priority`` of the split's DP all-gathers (``split`` ``ag_priority``, 0-3, lower = higher; TEST arm)."""
    return {} if split.get("ag_priority") is None else {"priority": split["ag_priority"]}


def _split_v5_import_ext(qkv_row, split, q_rows, tokens, cols, tag, sb):
    """Split v5 extended batch, the import: one DP all-gather of this replica's request rows (the B of the B' = B + F
    slots, token-major ``j B + b``, compacted from ``qkv_row`` ``[Q B', cols]``), then the Q rows of each foreign slot's
    source sequence (``pieces_mine[f, 4]`` + j B) into its rows ``j B' + B + f`` of ``qkv_row``, before the TP gather.
    An inactive foreign slot copies the rows at 0 (its group range is empty)."""
    bf16 = nl.bfloat16
    B, F, dp = split["ext"], split["foreign_slots"], split["dp"]
    rows = q_rows * B
    src = nl.ndarray((rows, cols), dtype=bf16, buffer=nl.hbm, name=f"{tag}_mla_split_imp_src")
    dst = nl.ndarray((dp * rows, cols), dtype=bf16, buffer=nl.hbm, name=f"{tag}_mla_split_imp_dst")
    for j in range(q_rows):
        nisa.dma_copy(
            dst=src[j * B : (j + 1) * B, 0:cols],
            src=qkv_row[j * tokens : j * tokens + B, 0:cols],
            name=f"{tag}_mla_split_imp_store_{j}",
        )
    ncc.all_gather(
        srcs=[src],
        dsts=[dst],
        replica_group=split["dp_group"],
        collective_dim=0,
        name=f"{tag}_mla_split_imp_ag",
        **_split_v5_ag_kwargs(split),
    )

    def select():
        pm = sb.tile((1, F * 5), nl.int32)
        nisa.dma_copy(dst=pm, src=split["checks"]["pieces_mine"].reshape((1, F * 5)), name=f"{tag}_mla_split_pm_load")
        for f in range(F):
            for j in range(q_rows):
                row = j * tokens + B + f
                nisa.dma_copy(
                    dst=qkv_row[row : row + 1, 0:cols],
                    src=dst.ap(
                        pattern=[[cols, 1], [1, cols]],
                        offset=j * B * cols,
                        scalar_offset=pm[0:1, f * 5 + 4 : f * 5 + 5],
                        indirect_dim=0,
                    ),
                    name=f"{tag}_mla_split_select_{f}_{j}",
                )

    if split.get("guards"):
        # "one": the foreign rows take their source rows only on a step with an active piece of this replica.
        _split_v5_if(split["checks"]["any_mine"], select, f"{tag}_mla_split_g1", sb)
    else:
        select()


def _split_v5_seq_pass_aps(call_views, passes, tokens, nb, head, seq_pass):
    """Split v5 extended batch: the HBM access patterns ``(o, sums)`` of the ``Q nb`` rows of local head ``head`` in the
    sequence-major pass ``seq_pass`` (the sequences ``seq_pass nb ..``; row ``j nb + s`` = global row ``j B' + seq_pass
    nb + s``) in the call-level BF16-o state ``call_views`` (``_state_rs_bf16_o_call``), which keeps each global row r at
    the token-major pass row: tensor row ``head x passes + r // view_rows``, column ``r % view_rows``
    (``_store_state_p2_bf16_o``). ``o`` covers ``Q nb x 512`` BF16 values, ``sums`` ``Q nb`` FP32 values, in row order.
    Sources of HBM-to-HBM DMAs (a DMA to SBUF does not lower these 4-level patterns: MEASURED trace)."""
    o_view, sum_view, view_rows = call_views
    per = view_rows // tokens
    kernel_assert(per * tokens == view_rows, f"split (v5) ext: B' = {tokens} must divide the {view_rows} pass rows")
    sum_stride = sum_view.shape[1]
    o_ap = o_view.ap(
        pattern=[[view_rows * _KV_LORA, passes], [tokens * _KV_LORA, per], [_KV_LORA, nb], [1, _KV_LORA]],
        offset=(head * passes * view_rows + seq_pass * nb) * _KV_LORA,
    )
    sum_ap = sum_view.ap(
        pattern=[[sum_stride, passes], [tokens, per], [1, nb], [1, 1]],
        offset=head * passes * sum_stride + seq_pass * nb,
    )
    return o_ap, sum_ap


def _split_v5_seq_state(call_views, passes, tokens, nb, local_heads, tag):
    """Split v5 extended batch: copy the call-level BF16-o state (token-major passes) to a sequence-major copy in HBM:
    tensor row ``head x passes + p`` holds the sequence-major pass p (rows ``j nb + s`` of the sequences ``p nb ..``,
    ``_split_v5_seq_pass_aps``), so the export, the merge and the tails read whole passes of 2 sequences. One HBM-to-HBM
    DMA of each view for each head and pass. Return the views ``(o, sums, view_rows)`` of the copy."""
    o_view, sum_view, view_rows = call_views
    o_seq = nl.ndarray(tuple(o_view.shape), dtype=o_view.dtype, buffer=nl.hbm, name=f"{tag}_mla_split_seq_o")
    sum_seq = nl.ndarray(
        (sum_view.shape[0], view_rows), dtype=sum_view.dtype, buffer=nl.hbm, name=f"{tag}_mla_split_seq_s"
    )
    o_words = view_rows * _KV_LORA
    for h in range(local_heads):
        for p in range(passes):
            row = h * passes + p
            o_ap, sum_ap = _split_v5_seq_pass_aps(call_views, passes, tokens, nb, h, p)
            nisa.dma_copy(dst=o_seq[row : row + 1, 0:o_words], src=o_ap, name=f"{tag}_seq_o{h}_{p}")
            nisa.dma_copy(dst=sum_seq[row : row + 1, 0:view_rows], src=sum_ap, name=f"{tag}_seq_s{h}_{p}")
    return o_seq, sum_seq, view_rows


def _split_v5_export_ext(split, seq_views, passes, local_heads, tag):
    """Split v5 extended batch, the export: this rank's reduced BF16-o state of the foreign passes (the sequence-major
    passes past the B request slots, 2 foreign slots each: per local head and foreign pass, its numerators ``[16 x 512]``
    BF16 and sums ``[16]`` FP32 as BF16 words), from the sequence-major copy ``seq_views`` (``_split_v5_seq_state``), in
    one DP all-gather; return the destination ``[dp x local_heads x FP, 16 x 512 + 32]`` BF16 (FP = F / 2; replica h's
    row of head hd and foreign pass fp is (h x local_heads + hd) x FP + fp)."""
    bf16, fp32 = nl.bfloat16, nl.float32
    o_dst, sum_dst, view_rows = seq_views
    width = view_rows * _KV_LORA + 2 * view_rows
    dp, first = split["dp"], split["ext"] // 2
    fpasses = passes - first
    src = nl.ndarray((local_heads * fpasses, width), dtype=bf16, buffer=nl.hbm, name=f"{tag}_mla_split_exp_src")
    dst = nl.ndarray((dp * local_heads * fpasses, width), dtype=bf16, buffer=nl.hbm, name=f"{tag}_mla_split_exp_dst")
    src_f = src.view(fp32)
    o_words = view_rows * _KV_LORA
    for hd in range(local_heads):
        for fp in range(fpasses):
            row, out = hd * passes + first + fp, hd * fpasses + fp
            nisa.dma_copy(
                dst=src[out : out + 1, 0:o_words], src=o_dst[row : row + 1, 0:o_words], name=f"{tag}_exp_o{hd}_{fp}"
            )
            nisa.dma_copy(
                dst=src_f[out : out + 1, o_words // 2 : o_words // 2 + view_rows],
                src=sum_dst[row : row + 1, 0:view_rows],
                name=f"{tag}_exp_s{hd}_{fp}",
            )
    ncc.all_gather(
        srcs=[src],
        dsts=[dst],
        replica_group=split["dp_group"],
        collective_dim=0,
        name=f"{tag}_mla_split_exp_ag",
        **_split_v5_ag_kwargs(split),
    )
    return dst


def _split_v5_seq_tail(
    split, exp_dst, seq_views, x_hT, rs_call, rows, seq_pass, passes, tokens, q_rows, tr, local_heads, tail, tag, sb
):
    """Split v5 extended batch: the call-level tail of the request pass ``seq_pass`` (a sequence-major pass of 2 request
    slots; global ``rows``, row ``j nb + s`` = global row ``j B' + seq_pass nb + s``): its layer inputs (run by run), its
    state from the sequence-major copy ``seq_views`` (``_split_v5_seq_state``; ``load_state_bf16_o`` as the token-major
    tail loads a pass) with the owner merge (``_split_v5_merge``), the tail ``tail(local_state, x_pass, partial)`` into a
    pass partial, and the partial rows to their rows ``j B + seq_pass nb + s`` of the request-row output tile
    ``rs_call`` ``[128, Q B, 56]`` (``_split_v5_store_seq_partial``), which the call-level output all-reduce reads. The
    foreign passes take no tail (their rows have no output). Each row takes the instructions of the token-major tail,
    so a row the merge does not touch keeps its bits."""
    bf16 = nl.bfloat16
    vector = nisa.engine.vector
    nb = tr // q_rows
    x_pass = sb.tile((_P_MAX, tr, _HIDDEN_TILES), bf16)
    for dst_row, src_row, n in _row_runs(rows):
        nisa.tensor_copy(
            dst=x_pass[0:_P_MAX, dst_row : dst_row + n, 0:_HIDDEN_TILES],
            src=x_hT[0:_P_MAX, src_row : src_row + n, 0:_HIDDEN_TILES],
            engine=vector,
        )
    o_seq, sum_seq, view_rows = seq_views
    local_state = load_state_bf16_o(
        o_seq,
        sum_seq,
        view_rows,
        lambda h: h * passes + seq_pass,
        local_heads,
        tr,
        _KV_LORA,
        True,
        sb,
        f"{tag}_mla_state",
    )
    local_state = _split_v5_merge(local_state, split, exp_dst, seq_pass, tr, local_heads, f"{tag}_sm", sb)
    partial = sb.tile((_P_MAX, tr, _HIDDEN_TILES), bf16)
    tail(local_state, x_pass, partial)
    _split_v5_store_seq_partial(rs_call, partial, q_rows, split["ext"], nb, seq_pass)


def _split_v5_store_seq_partial(rs_call, partial, q_rows, tokens, nb, seq_pass):
    """Split v5 extended batch: copy the partial ``[128, Q nb, 56]`` of the sequence-major pass ``seq_pass`` (row ``j nb +
    s``) to its token-major rows ``j T + seq_pass nb + s`` of ``rs_call`` ``[128, Q T, 56]`` (``tokens`` T; the request
    slots B for the request-row output tile; one VectorE copy)."""
    nisa.tensor_copy(
        dst=rs_call.reshape((_P_MAX, q_rows, tokens, _HIDDEN_TILES))[
            0:_P_MAX, 0:q_rows, seq_pass * nb : (seq_pass + 1) * nb, 0:_HIDDEN_TILES
        ],
        src=partial.reshape((_P_MAX, q_rows, nb, _HIDDEN_TILES)),
        engine=nisa.engine.vector,
    )


def _split_v5_merge(local_state, split, exp_dst, pass_index, rows, local_heads, tag, sb):
    """Split v5 extended batch, the owner merge of one pass (2 slots): add, in FP32 before num / sum, the foreign states of
    the active pieces of each of this replica's slots (``merge_mine`` (count, row_lo, row_hi): the export row h x 3 FP +
    fp of each piece's head 0, helper h, foreign pass fp = f // 2;
    a piece in foreign slot f serves owner slots of parity f % 2, so its rows are the slot's pass rows) to the slot's rows
    of the dense pass state ``[local_heads x rows, 513]``, and return it (``_tail_partial`` ``dense_state``). A request
    pass only: the foreign passes take no tail. Both helpers' terms add in a fixed order (h_lo, then h_hi)."""
    fp32, int32 = nl.float32, nl.int32
    vector = nisa.engine.vector
    n = local_heads * rows
    B, nb = split["ext"], 2
    fpasses = split["foreign_slots"] // 2
    kernel_assert(pass_index * nb < B, "split (v5) ext: the merge runs on the request passes (no foreign-pass tail)")
    dense = _split_dense_fp32(local_state, rows, local_heads, sb)

    def add_pieces():
        merge = split["checks"]["merge_mine"]
        width = exp_dst.shape[1]
        exp_f = exp_dst.view(fp32)
        o_words = rows * _KV_LORA
        for s_slot in range(nb):
            b = pass_index * nb + s_slot
            mi = sb.tile((1, 3), int32)
            nisa.dma_copy(dst=mi, src=merge[b : b + 1, 0:3], name=f"{tag}_mi{s_slot}")
            cnt_i = sb.tile((n, 1), int32)
            nisa.dma_copy(dst=cnt_i, src=merge.ap(pattern=[[0, n], [1, 1]], offset=b * 3), name=f"{tag}_mc{s_slot}")
            cnt = sb.tile((n, 1), fp32)
            nisa.tensor_copy(dst=cnt, src=cnt_i, engine=vector)
            for c in range(2):
                flag = sb.tile((n, 1), fp32)
                nisa.tensor_scalar(dst=flag, data=cnt, op0=nl.greater, operand0=float(c), engine=vector)
                contrib = sb.tile((n, _STATE), fp32)
                for hd in range(local_heads):
                    nisa.dma_copy(
                        dst=contrib[hd * rows : (hd + 1) * rows, 0:_KV_LORA],
                        src=exp_dst.ap(
                            pattern=[[_KV_LORA, rows], [1, _KV_LORA]],
                            offset=hd * fpasses * width,
                            scalar_offset=mi[0:1, 1 + c : 2 + c],
                            indirect_dim=0,
                        ),
                        name=f"{tag}_mo{s_slot}{c}{hd}",
                    )
                    nisa.dma_copy(
                        dst=contrib[hd * rows : (hd + 1) * rows, _KV_LORA:_STATE],
                        src=exp_f.ap(
                            pattern=[[1, rows], [1, 1]],
                            offset=hd * fpasses * (width // 2) + o_words // 2,
                            scalar_offset=mi[0:1, 1 + c : 2 + c],
                            indirect_dim=0,
                        ),
                        name=f"{tag}_ms{s_slot}{c}{hd}",
                    )
                merged = sb.tile((n, _STATE), fp32)
                nisa.tensor_tensor(dst=merged, data1=dense, data2=contrib, op=nl.add, engine=vector)
                nisa.tensor_copy_predicated(
                    dst=dense,
                    src=merged,
                    predicate=_split_row_mask(n, nb, s_slot, flag, _STATE, nl.uint8, f"{tag}_mm{s_slot}{c}", sb),
                )

    if split.get("guards"):
        # "one": only a step with a confirmed split of this replica merges (the guard of the export's readers).
        _split_v5_if(_split_v5_split_mine(split, tag, sb), add_pieces, f"{tag}_mla_split_g3", sb)
    else:
        add_pieces()
    return dense


def _split_v5_import(qkv_row, split, rows, cols, tag):
    """Split v5 (``mla_block_attention`` ``split`` v5): the import all-gather of the call's q_a | kv_a rows
    ``qkv_row`` ``[rows, cols]`` BF16 over the DP group (one HBM block per replica). The CC skeleton: the gathered rows
    are not read yet."""
    bf16 = nl.bfloat16
    dp = split["dp"]
    src = nl.ndarray((rows, cols), dtype=bf16, buffer=nl.hbm, name=f"{tag}_mla_split_imp_src")
    dst = nl.ndarray((dp * rows, cols), dtype=bf16, buffer=nl.hbm, name=f"{tag}_mla_split_imp_dst")
    nisa.dma_copy(dst=src, src=qkv_row[0:rows, 0:cols], name=f"{tag}_mla_split_imp_store")
    ncc.all_gather(
        srcs=[src], dsts=[dst], replica_group=split["dp_group"], collective_dim=0, name=f"{tag}_mla_split_imp_ag"
    )
    return dst


def _split_v5_select(split, imp_dst, qkv_full, rows, q_rows, tokens, cols, tag, sb):
    """Split v5 fold, the import row select (guard 1): when this replica has an active foreign piece (the step check's
    ``any_mine``), copy the Q token-major rows of each foreign slot's source sequence from the DP gather of the request
    rows ``imp_dst`` ``[dp x rows, cols]`` (row j of the source = its first row ``pieces_mine[f, 4]`` + j B) into the
    foreign rows ``rows + f Q ..`` of ``qkv_full``, before the main TP gather. A CC-free runtime then-only IF, skipped on
    a step with no active piece, so the gather does not wait for the import all-gather then. An inactive slot of an
    active replica copies the rows at 0 (never merged)."""
    F = split["foreign_slots"]
    checks = split["checks"]
    any_i = sb.tile((1, 1), nl.int32)
    nisa.dma_copy(dst=any_i, src=checks["any_mine"], name=f"{tag}_mla_split_any_load")
    reg = nisa.register_alloc()
    nisa.register_load(reg, any_i)

    def then_branch():
        pm = sb.tile((1, F * 5), nl.int32)
        nisa.dma_copy(dst=pm, src=checks["pieces_mine"].reshape((1, F * 5)), name=f"{tag}_mla_split_pm_load")
        for f in range(F):
            nisa.dma_copy(
                dst=qkv_full[rows + f * q_rows : rows + (f + 1) * q_rows, 0:cols],
                src=imp_dst.ap(
                    pattern=[[tokens * cols, q_rows], [1, cols]],
                    offset=0,
                    scalar_offset=pm[0:1, f * 5 + 4 : f * 5 + 5],
                    indirect_dim=0,
                ),
                name=f"{tag}_mla_split_select_{f}",
            )

    nke.if_else(reg, then_branch)


def _split_v5_export(split, q_rows, local_heads, tag, sb):
    """Split v5: the export all-gather of the reduced foreign states over the DP group, in the BF16-o form of the main
    state reduce-scatter: per foreign row, the BF16 numerators ``[local_heads, 512]`` then the FP32 sums
    ``[local_heads]`` as their BF16 view, ``[F Q, local_heads x 514]`` BF16 (24,672 B a rank at F = 1, Q = 8). The CC
    skeleton: the block is zeros and the gathered states are not read yet."""
    bf16 = nl.bfloat16
    dp = split["dp"]
    rows, cols = split["foreign_slots"] * q_rows, local_heads * (_KV_LORA + 2)
    src = nl.ndarray((rows, cols), dtype=bf16, buffer=nl.hbm, name=f"{tag}_mla_split_exp_src")
    dst = nl.ndarray((dp * rows, cols), dtype=bf16, buffer=nl.hbm, name=f"{tag}_mla_split_exp_dst")
    with sb.scope("split_v5_export"):
        zeros = sb.tile((rows, cols), bf16)
        nisa.memset(dst=zeros, value=0.0, engine=nisa.engine.vector)
        nisa.dma_copy(dst=src, src=zeros, name=f"{tag}_mla_split_exp_store")
    ncc.all_gather(
        srcs=[src], dsts=[dst], replica_group=split["dp_group"], collective_dim=0, name=f"{tag}_mla_split_exp_ag"
    )


def _split_import(qkv_row, split, tokens, q_rows, cols, tag, sb):
    """Long-sequence split (``mla_block_attention`` ``split``): give a helper the q_a | kv_a slice of its partner's
    donor rows. The owner (role 1) sends the rows ``j B' + B' - 3`` of ``qkv_row`` ``[Q B', cols]`` BF16 at the rows
    ``j B' + B' - 2`` of a zero tile (single-row DMAs; an engine access must start on a quadrant); every other rank sends
    zeros. One all-reduce over the pair group (``split["group"]``, an HBM pair) then gives each rank of a pair the
    owner's rows, and a helper (role 2) copies them into its foreign rows ``j B' + B' - 2``. Adding zeros keeps the
    bytes (the slice is finite), so the helper's rows are the owner's: the same query and current key bits."""
    bf16 = nl.bfloat16
    rows = q_rows * tokens
    donor, foreign = tokens - 3, tokens - 2
    owner, helper = _split_role_flags(split["info"], rows, f"{tag}_imp", sb)
    moved = sb.tile((rows, cols), bf16)
    nisa.memset(dst=moved, value=0.0, engine=nisa.engine.vector)
    for j in range(q_rows):
        nisa.dma_copy(
            dst=moved[j * tokens + foreign : j * tokens + foreign + 1, 0:cols],
            src=qkv_row[j * tokens + donor : j * tokens + donor + 1, 0:cols],
            name=f"{tag}_mla_split_export_{j}",
        )
    send = sb.tile((rows, cols), bf16)
    nisa.memset(dst=send, value=0.0, engine=nisa.engine.vector)
    nisa.tensor_copy_predicated(
        dst=send, src=moved, predicate=_split_row_mask(rows, tokens, foreign, owner, cols, nl.uint8, f"{tag}_imo", sb)
    )
    src_hbm = nl.ndarray((rows, cols), dtype=bf16, buffer=nl.hbm, name=f"{tag}_mla_split_imp_src")
    dst_hbm = nl.ndarray((rows, cols), dtype=bf16, buffer=nl.hbm, name=f"{tag}_mla_split_imp_dst")
    nisa.dma_copy(dst=src_hbm, src=send, name=f"{tag}_mla_split_imp_store")
    ncc.all_reduce(
        srcs=[src_hbm], dsts=[dst_hbm], replica_group=split["group"], op=nl.add, name=f"{tag}_mla_split_imp_ar"
    )
    recv = sb.tile((rows, cols), bf16)
    nisa.dma_copy(dst=recv, src=dst_hbm, name=f"{tag}_mla_split_imp_load")
    nisa.tensor_copy_predicated(
        dst=qkv_row,
        src=recv,
        predicate=_split_row_mask(rows, tokens, foreign, helper, cols, nl.uint8, f"{tag}_imh", sb),
    )


def _split_dense_fp32(local_state, rows, local_heads, sb):
    """Return the dense FP32 ``[local_heads * rows, 513]`` state of a pass: ``local_state`` itself, or for the BF16-o
    form (``StateParts``) its BF16 numerators widened (exact) beside its FP32 sums."""
    if not isinstance(local_state, StateParts):
        return local_state
    dense = sb.tile((local_heads * rows, _STATE), nl.float32)
    nisa.tensor_copy(dst=dense[0 : local_heads * rows, 0:_KV_LORA], src=local_state.o, engine=nisa.engine.vector)
    nisa.tensor_copy(
        dst=dense[0 : local_heads * rows, _KV_LORA:_STATE], src=local_state.sums, engine=nisa.engine.vector
    )
    return dense


def _split_contribution(local_state, merge_src, pass_index, passes, rows, local_heads, tokens, tag, sb):
    """Long-sequence split: store this rank's merge contribution of one token-major pass to ``merge_src`` (the pass-minor
    HBM rows of ``_load_pass_state``): the state of each foreign row ``j B' + B' - 2`` at its donor row ``j B' + B' - 3``
    (one partition down: the donor slot is the foreign slot - 1), zero elsewhere. The foreign rows hold zeros unless
    this rank is a helper (no key, own-key weight 0), so only a helper sends a state."""
    fp32 = nl.float32
    n = local_heads * rows
    dense = _split_dense_fp32(local_state, rows, local_heads, sb)
    shifted = sb.tile((n, _STATE), fp32)
    nisa.memset(dst=shifted, value=0.0, engine=nisa.engine.vector)
    nisa.dma_copy(dst=shifted[0 : n - 1, 0:_STATE], src=dense[1:n, 0:_STATE], name=f"{tag}_mla_split_shift")
    ones = sb.tile((n, 1), fp32)
    nisa.memset(dst=ones, value=1.0, engine=nisa.engine.vector)
    donor_rows = _split_row_mask(n, tokens, tokens - 3, ones, _STATE, nl.uint8, f"{tag}_cm", sb)
    contribution = sb.tile((n, _STATE), fp32)
    nisa.memset(dst=contribution, value=0.0, engine=nisa.engine.vector)
    nisa.tensor_copy_predicated(dst=contribution, src=shifted, predicate=donor_rows)
    row_width = merge_src.shape[1]
    for local_head in range(local_heads):
        nisa.dma_copy(
            dst=merge_src.ap(
                pattern=[[_STATE, rows], [1, _STATE]], offset=(local_head * passes + pass_index) * row_width
            ),
            src=contribution[local_head * rows : (local_head + 1) * rows, 0:_STATE],
            name=f"{tag}_mla_split_contrib_{local_head}",
        )


def _split_merge(local_state, merge_dst, pass_index, passes, rows, local_heads, tokens, split_info, tag, sb):
    """Long-sequence split: on an owner (role 1), add the helper's state of its donor rows (``merge_dst``, the pair
    all-reduce of ``_split_contribution``) to the donor rows ``j B' + B' - 3`` of the pass state before ``num / sum``,
    and return the dense FP32 state ``[local_heads * rows, 513]`` that the tail reads (``_tail_partial``
    ``dense_state``). The FP32 add is the cross-replica term of the CP merge. With the BF16-o form the merged donor
    numerators stay FP32 into ``num / sum`` (no second BF16 rounding), and every other row holds its BF16 numerators
    widened (exact), so its ``num / sum`` keeps its bits; every other real row keeps its bits (a predicated copy).
    Every rank then gives the foreign and the pad rows a unit sum (no own key there), so their tail stays finite."""
    fp32 = nl.float32
    vector = nisa.engine.vector
    n = local_heads * rows
    helper_state = _load_pass_state(merge_dst, pass_index, passes, rows, local_heads, sb, f"{tag}_mla_split_recv")
    owner, _ = _split_role_flags(split_info, n, f"{tag}_mg", sb)
    dense = _split_dense_fp32(local_state, rows, local_heads, sb)
    merged = sb.tile((n, _STATE), fp32)
    nisa.tensor_tensor(dst=merged, data1=dense, data2=helper_state, op=nl.add, engine=vector)
    nisa.tensor_copy_predicated(
        dst=dense,
        src=merged,
        predicate=_split_row_mask(n, tokens, tokens - 3, owner, _STATE, nl.uint8, f"{tag}_mf", sb),
    )
    # The foreign and the pad rows carry no own key: a unit sum keeps their tail finite (their output is not read).
    ones = sb.tile((n, 1), fp32)
    nisa.memset(dst=ones, value=1.0, engine=vector)
    for slot in (tokens - 2, tokens - 1):
        nisa.tensor_copy_predicated(
            dst=dense[0:n, _KV_LORA:_STATE],
            src=ones,
            predicate=_split_row_mask(n, tokens, slot, ones, 1, nl.uint8, f"{tag}_mu{slot}", sb),
        )
    return dense


def _pass_plan(
    position,
    rank_info,
    table,
    first,
    seqs,
    q_rows,
    options,
    tag,
    sb,
    plan_sb=None,
    live_rows=None,
    split_info=None,
    total_seqs=None,
    split_lo=None,
):
    """Build the item plan of the packed loop of one pass (the ``seqs`` sequences ``first ..`` of ``position``
    ``[B, 1]`` HBM, ``table`` their block table) for ``_attend_prior_packed``. ``plan_sb`` (default ``sb``) allocates
    the plan tiles. ``live_rows``: the runner's ``[B, 1]`` live-row mask of ``options`` ``pad_skip`` and ``pad_runs``
    (the plan's ``pad_src``: the HBM pad source of the pass, ``live_rows`` or else ``position``, and its first row).
    ``split_info`` (``mla_block_attention`` ``split``) with ``total_seqs`` B: the items of the split slot B - 1 take its
    group range (``_split_group_range``); ``split_lo`` (split v5, the ``[1, B]`` INT32 HBM lo_eff): sequence b takes its
    groups lo_eff[b] .. (``_split_v5_group_range``)."""
    int32 = nl.int32
    fp32 = nl.float32
    hwdge = nisa.dge_mode.hwdge
    sync = nisa.engine.sync
    local_blocks = table.shape[1]
    position_i = sb.tile((seqs, 1), int32)
    nisa.dma_copy(
        dst=position_i,
        src=position[first : first + seqs, 0:1],
        dge_mode=hwdge,
        engine=sync,
        name=f"{tag}_mla_prior_position_load",
    )
    rank_i = sb.tile((seqs, 1), int32)
    nisa.dma_copy(
        dst=rank_i,
        src=rank_info[0:1, 2:3].broadcast(0, seqs),
        dge_mode=hwdge,
        engine=sync,
        name=f"{tag}_mla_prior_rank_load",
    )
    rank_f = sb.tile((seqs, 1), fp32)
    nisa.tensor_copy(dst=rank_f, src=rank_i, engine=nisa.engine.vector)
    live_i = None
    if options["pad_skip"] and live_rows is not None:
        live_i = sb.tile((seqs, 1), int32)
        nisa.dma_copy(
            dst=live_i,
            src=live_rows[first : first + seqs, 0:1],
            dge_mode=hwdge,
            engine=sync,
            name=f"{tag}_mla_live_load",
        )
    slots = options["slots"]
    tails = pack_tail_chain(local_blocks, slots, options["tail_slots"], seqs) if options["tail_slots"] else None
    plan = _pack_items(
        position_i,
        rank_f,
        table,
        tag,
        sb,
        slots,
        plan_sb=plan_sb,
        tail_slots=tails,
        tail_round=tails is not None,
        with_ti=False,
        group_extra=q_rows - 1,
        null_threshold=-float(q_rows),
        pad_skip=options["pad_skip"],
        live_i=live_i,
        clamp_block=max(2, 1 << (seqs - 1).bit_length()) if options["clamp_tail_only"] else 0,
        **(
            {}
            if split_info is None
            else {
                "group_range": _split_group_range(
                    split_info, first, seqs, total_seqs, table.shape[1], f"{tag}_split", sb
                )
            }
        ),
        **(
            {}
            if split_lo is None
            else {
                "group_range": _split_v5_group_range(split_lo, first, seqs, table.shape[1], f"{tag}_v5", sb),
                "group_lo_hbm": (split_lo[0], first, split_lo[1]),
            }
        ),
    )
    if options["pad_runs"]:
        plan["pad_src"] = (position if live_rows is None else live_rows, first)
    return plan


def packed_prior_form(packed_prior_options: dict | None) -> dict:
    """Return the packed-loop form of ``packed_prior``: ``_PACKED_PRIOR_OPTIONS`` with the given overrides."""
    return {**_PACKED_PRIOR_OPTIONS, **(packed_prior_options or {})}


def mla_pass_plans(
    position: nl.NkiTensor,
    rank_info: nl.NkiTensor,
    block_table: nl.NkiTensor,
    q_rows: int,
    packed_prior_options: dict | None,
    sb: SbufTiles,
    tag: str = "mla_pass_plan",
    live_rows: nl.NkiTensor = None,
    split_info: nl.NkiTensor = None,
) -> dict:
    """Return ``mla_block_attention`` ``pass_plans`` for the MLA layers of a step: the item plan of each pass of up
    to 16 rows (``Q`` rows of ``16 / Q`` sequences, or all ``B`` sequences in one pass) of the sequences of
    ``position`` ``[B, 1]`` (HBM), as ``_attend_prior_packed`` builds it, on ``sb`` (a persistent heap: the plans live
    through the MLA layers), and the tile caches the passes share. ``block_table`` ``[B, NB]`` gives the shape only.
    With ``packed_prior_options`` ``step_items`` it also returns the step store of the item rows of each pass
    (``items``: its name prefix and, with page runs, its persistent run-scale tile on ``sb``), which the first MLA layer
    of the step fills. ``split_info`` (``mla_block_attention`` ``split``): the plans take the group range of the split
    slot B - 1."""
    options = {**_PACKED_PRIOR_OPTIONS, **(packed_prior_options or {})}
    # The first MLA layer of the step puts the tiles that the layers share (``shared``) on this heap inside its frame,
    # which returns its heap tiles at its end; the per-layer ``kv_group`` constants of the later layers then take the
    # bytes of the shared TI offsets that those layers still read.
    kernel_assert(
        not (options["kv_group"] and sb.explicit and sb.heap),
        "packed_prior_options kv_group corrupts the shared MLA tiles of hoisted pass plans on an SBUF region heap",
    )
    tokens, local_blocks = block_table.shape
    nb = min(tokens, _TAIL_ROWS // q_rows)
    kernel_assert(tokens % nb == 0, f"mla_pass_plans needs whole passes of {nb} sequences, got {tokens}")
    plans = []
    items = []
    # ``step_items``: the run scale of a pass is one value for each state partition (96 heads, 128 columns with p2).
    scale_parts = _P_MAX if options["p2"] else CP_TARGET_HEADS
    # With a region heap, the scratch tiles of each plan build take a stack frame, so only the plan tiles stay on ``sb``.
    scratch = SbufTiles(sb.region) if sb.explicit and sb.heap else sb
    for p in range(tokens // nb):
        first = p * nb
        with scratch.scope(f"pass_plan_{p}"):
            plans.append(
                _pass_plan(
                    position,
                    rank_info,
                    block_table[first : first + nb, 0:local_blocks],
                    first,
                    nb,
                    q_rows,
                    options,
                    f"{tag}_{p}",
                    scratch,
                    plan_sb=sb,
                    live_rows=live_rows,
                    split_info=split_info,
                    total_seqs=tokens,
                )
            )
        if options["step_items"]:
            items.append(
                {
                    "tag": f"{tag}_{p}_step",
                    **({"run_scale": sb.tile((scale_parts, nb), nl.float32)} if options["page_runs"] else {}),
                }
            )
    # The tiles that the packed loops of the step share (``shared``) take a zone of the persistent heap: the first MLA
    # layer allocates them in its frame, which returns its own heap tiles at its end, and the later layers read them.
    pack_sb = sb
    if sb.explicit and sb.heap:
        pack_sb = SbufZone(sb.region, pack_shared_zone_bytes(options, q_rows, nb), f"{tag}_pack_shared")
    return {
        "plans": plans,
        "shared": {},
        "pack_hbm": {},
        "pack_sb": pack_sb,
        **({"items": items} if options["step_items"] else {}),
    }


def mla_fp8_pv_step_stores(
    block_table: nl.NkiTensor,
    q_rows: int,
    packed_prior_options: dict | None,
    sb: SbufTiles,
    tag: str = "mla_pass_plan",
) -> dict:
    """Return ``mla_block_attention`` ``pass_plans`` for ``packed_prior_options`` ``fp8_pv_step_store``: only the step
    store of each pass (``mla_pass_plans`` ``items``: its name prefix and, with ``fp8_pv_page_runs``, its persistent
    run-scale tile on ``sb``), with no hoisted item plans of the packed loop (the FP8 P@V pass builds its own). The
    first MLA layer of the step fills each store (``fp8_pv_setup`` ``step``) and the later ones read it.
    ``block_table`` ``[B, NB]`` gives the shape only."""
    options = {**_PACKED_PRIOR_OPTIONS, **(packed_prior_options or {})}
    kernel_assert(
        options["fp8_pv"] and options["fp8_pv_step_store"],
        "mla_fp8_pv_step_stores is the pass_plans of packed_prior_options fp8_pv with fp8_pv_step_store",
    )
    tokens = block_table.shape[0]
    nb = min(tokens, _TAIL_ROWS // q_rows)
    kernel_assert(tokens % nb == 0, f"mla_fp8_pv_step_stores needs whole passes of {nb} sequences, got {tokens}")
    items = [
        {
            "tag": f"{tag}_{p}_step",
            **({"run_scale": sb.tile((_P_MAX, nb), nl.float32)} if options["fp8_pv_page_runs"] else {}),
        }
        for p in range(tokens // nb)
    ]
    return {"items": items}


def _loop_metadata(
    token_meta,
    token_f,
    key_start,
    tokens,
    group_count,
    loop_groups,
    iter_extra,
    tag,
    sb,
    lay,
    pad_skip=False,
    live_rows=None,
    pair_tail=False,
    unroll=1,
    tail_groups=0,
):
    """Return the ``[1, cols*tokens]`` loop metadata row of each sequence, and the padding rows (None without
    ``pad_skip``).

    The columns are (iterations, local start), plus the local lower bound ``key_start[b] - 128 t`` when
    ``key_start`` (the ``[B, 1]`` INT32 HBM bound) is given (``cols`` is 2 or 3). Local block j holds a prior
    key when ``128 cp j + 128 t < position``, so a sequence has ``ceil((position - 128 t) / group_stride)``
    key groups and ``ceil(groups / loop_groups)`` iterations. The row lives on partition 0 so a per-sequence
    register load reads it (see ``tkg_mla_paged``).

    ``iter_extra`` extends the iteration count: the bitwise verify passes ``Q - 1`` so the loop visits the
    group that holds the current rows ``position[b] .. position[b] + Q - 1`` (prior keys of the later rows),
    which the count from ``position[b]`` alone misses when ``position[b]`` sits on a group boundary. The
    local start (the mask base) stays ``position[b]``; the per-row mask adds ``j``.

    With ``pad_skip`` a padding sequence (``live_rows`` 0, or without ``live_rows`` a sequence at position 0, the
    test form) has 0 iterations, and the second return value holds on
    partition 0 the FP32 row ``pad`` (column ``2 b``: 1 for a padding sequence, else 0; column ``2 b + 1``: 1 for a
    live sequence, else 0).

    With ``pair_tail`` (``loop_groups`` 4: two group pairs an iteration) a sequence of g groups has P = ceil(g / 2) group
    pairs, its iteration column holds the full iterations P // 2, and the row gains ``tokens`` columns after
    ``cols * tokens``: the tail count P % 2 of each sequence (``_attend_prior_loop`` ``pair_tail``). With ``unroll`` 2
    (two iterations a loop trip) the iteration column holds the trips F // 2 of the F = P // 2 full iterations, and each
    sequence takes 2 columns after ``cols * tokens``: its tail count P % 2, then its iteration left over F % 2.

    With ``tail_groups`` (``_attend_prior_loop`` ``tail_groups``; the tails t_0 > t_1 > ... of ``loop_tails``) a
    sequence of g groups (at most ``group_count``) with r = g mod L in 1..t_0 (L = ``loop_groups``) has ceil(g / L) - 1
    iterations and the tail count 1 in the smallest tail t_k >= r (0 in the others), else ceil(g / L) iterations and
    the tail counts 0; with n tails the row gains n x ``tokens`` columns after ``cols * tokens``, and column
    ``cols * tokens + n b + k`` holds the count of tail t_k of sequence b. With the ``_FULL_TAIL`` bit the tails start
    with the full iteration (t_0 = L, ``loop_tail_sizes``): a sequence with at least one key group has ceil(g / L) - 1
    iterations and its last iteration in exactly one tail (the full one when no smaller tail holds it); a sequence with
    no key group has 0 iterations and no tail.
    """
    fp32 = nl.float32
    int32 = nl.int32
    hwdge = nisa.dge_mode.hwdge
    sync = nisa.engine.sync
    cols = 3 if key_start is not None else 2
    position_f = sb.tile((tokens, 1), fp32)
    nisa.tensor_copy(dst=position_f, src=token_meta[0:tokens, 0:1])
    local_position = sb.tile((tokens, 1), fp32)
    nisa.scalar_tensor_tensor(
        dst=local_position,
        data=token_f[0:tokens, 2:3],
        op0=nl.multiply,
        operand0=-float(_BLOCK),
        op1=nl.add,
        operand1=position_f,
    )
    loop_meta_f = sb.tile((tokens, cols), fp32)
    nisa.tensor_scalar(
        dst=loop_meta_f[0:tokens, 0:1],
        data=local_position,
        op0=nl.add,
        operand0=float(lay.group_stride - 1 + iter_extra),
    )
    nisa.tensor_copy(dst=loop_meta_f[0:tokens, 1:2], src=local_position)
    pads = None
    if pad_skip:
        flags = sb.tile((tokens, 2), fp32)
        if live_rows is None:
            # Test only: a sequence at position 0 is taken as a padding row.
            nisa.tensor_scalar(dst=flags[0:tokens, 1:2], data=position_f, op0=nl.greater, operand0=0.0)
        else:
            live_i = sb.tile((tokens, 1), int32)
            nisa.dma_copy(
                dst=live_i, src=live_rows[0:tokens, 0:1], dge_mode=hwdge, engine=sync, name=f"{tag}_mla_live_load"
            )
            live_f = sb.tile((tokens, 1), fp32)
            nisa.tensor_copy(dst=live_f, src=live_i)
            nisa.tensor_scalar(dst=flags[0:tokens, 1:2], data=live_f, op0=nl.greater, operand0=0.0)
        nisa.tensor_scalar(
            dst=flags[0:tokens, 0:1],
            data=flags[0:tokens, 1:2],
            op0=nl.multiply,
            operand0=-1.0,
            op1=nl.add,
            operand1=1.0,
        )
        # A select, not arithmetic: a padding sequence takes 0 groups, a live one keeps its value bit for bit.
        pad_u8 = sb.tile((tokens, 1), nl.uint8)
        nisa.tensor_copy(dst=pad_u8, src=flags[0:tokens, 0:1])
        zero_f = sb.tile((tokens, 1), fp32)
        nisa.memset(dst=zero_f, value=0.0)
        nisa.tensor_copy_predicated(dst=loop_meta_f[0:tokens, 0:1], src=zero_f, predicate=pad_u8)
        flags_hbm = nl.ndarray((tokens, 2), dtype=fp32, buffer=nl.hbm, name=f"{tag}_mla_pad_flags")
        nisa.dma_copy(dst=flags_hbm, src=flags, dge_mode=hwdge, engine=sync, name=f"{tag}_mla_pad_store")
        pad_row = sb.tile((1, 2 * tokens), fp32)
        nisa.dma_copy(
            dst=pad_row, src=flags_hbm.reshape((1, 2 * tokens)), dge_mode=hwdge, engine=sync, name=f"{tag}_mla_pad_load"
        )
        pads = {"pad": pad_row}
    if key_start is not None:
        # Load the bound per sequence from HBM. A TensorE transpose off the bias partition fails with explicit SBUF.
        key_i = sb.tile((tokens, 1), int32)
        nisa.dma_copy(dst=key_i, src=key_start[0:tokens, 0:1], dge_mode=hwdge, engine=sync, name=f"{tag}_mla_loop_key")
        key_f = sb.tile((tokens, 1), fp32)
        nisa.tensor_copy(dst=key_f, src=key_i)
        nisa.scalar_tensor_tensor(
            dst=loop_meta_f[0:tokens, 2:3],
            data=token_f[0:tokens, 2:3],
            op0=nl.multiply,
            operand0=-float(_BLOCK),
            op1=nl.add,
            operand1=key_f,
        )
    loop_meta = sb.tile((tokens, cols), int32)
    nisa.tensor_copy(dst=loop_meta, src=loop_meta_f)
    nisa.tensor_scalar(
        dst=loop_meta[0:tokens, 0:1],
        data=loop_meta[0:tokens, 0:1],
        op0=nl.right_shift,
        operand0=lay.group_stride.bit_length() - 1,
    )
    tail_i = None
    # The tail columns of each sequence after ``cols * tokens``.
    tail_width = unroll
    kernel_assert(
        not (tail_groups and pair_tail)
        and (
            not tail_groups
            or (
                loop_groups & (loop_groups - 1) == 0
                and tail_groups in LOOP_TAILS.values()
                and max(loop_tails(tail_groups), default=0) < loop_groups
            )
        ),
        f"tail_groups needs a form of LOOP_TAILS, iterations of a power-of-2 group count above its tails and no "
        f"pair_tail, got {tail_groups} tail groups and {loop_groups} groups",
    )
    if pair_tail:
        kernel_assert(loop_groups == 2 * _TS_GROUPS, f"pair_tail needs iterations of 2 group pairs, got {loop_groups}")
        # P = (min(g, groups) + 1) >> 1 group pairs: P >> 1 full iterations, then P & 1 tail pairs.
        nisa.tensor_scalar(
            dst=loop_meta[0:tokens, 0:1],
            data=loop_meta[0:tokens, 0:1],
            op0=nl.minimum,
            operand0=float(group_count),
            op1=nl.add,
            operand1=1.0,
            engine=nisa.engine.vector,
        )
        nisa.tensor_scalar(
            dst=loop_meta[0:tokens, 0:1],
            data=loop_meta[0:tokens, 0:1],
            op0=nl.right_shift,
            operand0=1,
            engine=nisa.engine.vector,
        )
        tail_i = sb.tile((tokens, unroll), int32)
        for k in range(unroll):
            # k 0: the tail pair P & 1, then P >> 1 full iterations; k 1 (unroll 2): the iteration left over, then trips.
            nisa.tensor_scalar(
                dst=tail_i[0:tokens, k : k + 1],
                data=loop_meta[0:tokens, 0:1],
                op0=nl.bitwise_and,
                operand0=1,
                engine=nisa.engine.vector,
            )
            nisa.tensor_scalar(
                dst=loop_meta[0:tokens, 0:1],
                data=loop_meta[0:tokens, 0:1],
                op0=nl.right_shift,
                operand0=1,
                engine=nisa.engine.vector,
            )
    elif tail_groups:
        vector = nisa.engine.vector
        tails = loop_tail_sizes(tail_groups, loop_groups)
        full_tail = bool(tail_groups & _FULL_TAIL)
        last = len(tails) - 1
        # g = min(groups, group_count) + loop_groups - 1: (g mod L) - 1 (mod L) is below t exactly when 1 <= g mod L <= t.
        nisa.tensor_scalar(
            dst=loop_meta[0:tokens, 0:1],
            data=loop_meta[0:tokens, 0:1],
            op0=nl.minimum,
            operand0=float(group_count),
            op1=nl.add,
            operand1=float(loop_groups - 1),
            engine=vector,
        )
        tail_i = sb.tile((tokens, len(tails)), int32)
        # (g - 1) mod L in the last column; each column k then takes [1 <= g mod L <= t_k], the last one in place.
        nisa.tensor_scalar(
            dst=tail_i[0:tokens, last : last + 1],
            data=loop_meta[0:tokens, 0:1],
            op0=nl.bitwise_and,
            operand0=loop_groups - 1,
            engine=vector,
        )
        for k, t in enumerate(tails):
            nisa.tensor_scalar(
                dst=tail_i[0:tokens, k : k + 1],
                data=tail_i[0:tokens, last : last + 1],
                op0=nl.less,
                operand0=float(t),
                engine=vector,
            )
        nisa.tensor_scalar(
            dst=loop_meta[0:tokens, 0:1],
            data=loop_meta[0:tokens, 0:1],
            op0=nl.right_shift,
            operand0=loop_groups.bit_length() - 1,
            engine=vector,
        )
        if full_tail:
            # The full-iteration tail takes the last iteration of every sequence with a key group: [ceil(g / L) >= 1].
            nisa.tensor_scalar(
                dst=tail_i[0:tokens, 0:1],
                data=loop_meta[0:tokens, 0:1],
                op0=nl.minimum,
                operand0=1.0,
                engine=vector,
            )
        nisa.tensor_tensor(
            dst=loop_meta[0:tokens, 0:1],
            data1=loop_meta[0:tokens, 0:1],
            data2=tail_i[0:tokens, 0:1],
            op=nl.subtract,
            engine=vector,
        )
        for k in range(last):
            # [t_(k+1) < g mod L <= t_k]: the smallest tail that holds the groups of the iteration.
            nisa.tensor_tensor(
                dst=tail_i[0:tokens, k : k + 1],
                data1=tail_i[0:tokens, k : k + 1],
                data2=tail_i[0:tokens, k + 1 : k + 2],
                op=nl.subtract,
                engine=vector,
            )
        tail_width = len(tails)
    else:
        nisa.tensor_scalar(
            dst=loop_meta[0:tokens, 0:1],
            data=loop_meta[0:tokens, 0:1],
            op0=nl.minimum,
            operand0=float(group_count),
            op1=nl.add,
            operand1=float(loop_groups - 1),
        )
        nisa.tensor_scalar(
            dst=loop_meta[0:tokens, 0:1],
            data=loop_meta[0:tokens, 0:1],
            op0=nl.right_shift,
            operand0=loop_groups.bit_length() - 1,
        )
    loop_hbm = nl.ndarray((tokens, cols), dtype=int32, buffer=nl.hbm, name=f"{tag}_mla_loop_meta")
    nisa.dma_copy(dst=loop_hbm, src=loop_meta, dge_mode=hwdge, engine=sync, name=f"{tag}_mla_loop_store")
    if tail_i is None:
        loop_row = sb.tile((1, cols * tokens), int32)
        nisa.dma_copy(
            dst=loop_row,
            src=loop_hbm.reshape((1, cols * tokens)),
            dge_mode=hwdge,
            engine=sync,
            name=f"{tag}_mla_loop_load",
        )
    else:
        tail_cols = tail_width * tokens
        loop_row = sb.tile((1, cols * tokens + tail_cols), int32)
        nisa.dma_copy(
            dst=loop_row[0:1, 0 : cols * tokens],
            src=loop_hbm.reshape((1, cols * tokens)),
            dge_mode=hwdge,
            engine=sync,
            name=f"{tag}_mla_loop_load",
        )
        tail_hbm = nl.ndarray((tokens, tail_width), dtype=int32, buffer=nl.hbm, name=f"{tag}_mla_loop_tail")
        nisa.dma_copy(dst=tail_hbm, src=tail_i, dge_mode=hwdge, engine=sync, name=f"{tag}_mla_loop_tail_store")
        nisa.dma_copy(
            dst=loop_row[0:1, cols * tokens : cols * tokens + tail_cols],
            src=tail_hbm.reshape((1, tail_cols)),
            dge_mode=hwdge,
            engine=sync,
            name=f"{tag}_mla_loop_tail_load",
        )
    return loop_row, pads


def _loop_groups(kv_pool, group_count: int, cap: int = 0) -> int:
    """Return the prior-key groups of one loop iteration: the pool-dtype default, at most ``cap`` (0 = no cap)."""
    default = _BF16_LOOP_GROUPS if kv_pool.dtype == nl.bfloat16 else _LOOP_GROUPS
    return min(min(default, cap) if cap else default, group_count)


def _values_stationary_numerators(groups, blocks, weights, pieces, heads, state_t):
    """Add the transposed numerators of the rows of one ``_attend_group_rows`` call to ``state_t``.

    For each piece of rows and latent tile ``lt``, the values ``v[keys, 128 lt .. 128 lt + 127]`` of each block are the
    stationary operand and the weights of the piece move, so PSUM holds ``sum_keys v[key, d] w[key, (row, head)]``.
    """
    fp32 = nl.float32
    for r0, r1 in pieces:
        cols = slice(r0 * heads, r1 * heads)
        for lt in range(_KV_LORA_TILES):
            numerator_t = nl.ndarray((_P_MAX, (r1 - r0) * heads), dtype=fp32, buffer=nl.psum)
            for index, (g, block) in enumerate(blocks):
                nisa.nc_matmul(
                    dst=numerator_t,
                    stationary=groups[g][2][0:_P_MAX, block, lt * _P_MAX : (lt + 1) * _P_MAX],
                    moving=weights[0:_P_MAX, index, cols],
                    accumulate=index > 0,
                )
            nisa.tensor_tensor(
                dst=state_t[0:_P_MAX, lt, cols], data1=state_t[0:_P_MAX, lt, cols], data2=numerator_t, op=nl.add
            )


def _new_state_t(count: int, heads: int, sb):
    """Return a zero FP32 ``[128, 4, count * heads]`` tile of transposed numerators (``rows_v_stationary``)."""
    state_t = sb.tile((_P_MAX, _KV_LORA_TILES, count * heads), nl.float32)
    nisa.memset(dst=state_t, value=0.0)
    return state_t


def _add_state_t(state, state_t, cqs, heads):
    """Add the transposed numerators ``state_t`` of each row to the numerator of the state of the row (PE transposes)."""
    for i, cq in enumerate(cqs):
        numerator_psum = nl.ndarray((heads, _KV_LORA), dtype=nl.float32, buffer=nl.psum)
        for lt in range(_KV_LORA_TILES):
            nisa.nc_transpose(
                dst=numerator_psum[0:heads, lt * _P_MAX : (lt + 1) * _P_MAX],
                data=state_t[0:_P_MAX, lt, i * heads : (i + 1) * heads],
            )
        nisa.tensor_tensor(
            dst=state[0:heads, cq, 0:_KV_LORA], data1=state[0:heads, cq, 0:_KV_LORA], data2=numerator_psum, op=nl.add
        )


def _latent_key_tile(group_keys: int, dr: bool, sb, words: bool = False):
    """Return the latent key tile of one key group: BF16 ``[128, 4, keys]``, the FP8 double_row key planes
    ``[128, 2, 2, keys]`` (``rows_dr``, ``_dr_key_planes``), or the 2-byte key words ``[128, 2, keys]`` BF16
    (``words``, ``rows_dr_inplace``, ``_dr_key_words``: the same bytes per partition as the planes, read as a step-2 FP8
    view). ``words`` keeps the planes allocation unchanged."""
    if words:
        return sb.tile((_P_MAX, _DR_BLOCKS, group_keys), nl.bfloat16)
    if dr:
        return sb.tile((_P_MAX, 2, _DR_BLOCKS, group_keys), nl.float8_e4m3fn)
    return sb.tile((_P_MAX, _KV_LORA_TILES, group_keys), nl.bfloat16)


def _dr_key_transposes(rows, words_psum, identity16):
    """Transpose the FP8 cache rows ``rows`` ``[128, blocks, 576]`` into the 2-byte key words ``words_psum`` ``[128, 2,
    keys]`` BF16 (shared by ``_dr_key_planes`` and ``_dr_key_words``): a 2-byte transpose-mode transpose of each
    256-feature block of each 128 keys moves the feature pairs ``(2c, 2c + 1)`` to partition ``c`` (it moves the bits),
    so word ``(c, block, key)`` holds features ``256 block + 2c`` (low byte) and ``+ 1`` (high byte)."""
    group_blocks = rows.shape[1]
    row_words = rows.view(nl.bfloat16)
    for local_block in range(group_blocks):
        keys = slice(local_block * _BLOCK, (local_block + 1) * _BLOCK)
        for block in range(_DR_BLOCKS):
            col = block * _DR_BLOCK_WORDS
            nisa.nc_matmul(
                dst=words_psum[0:_P_MAX, block, keys],
                stationary=row_words[0:_P_MAX, local_block, col : col + _DR_BLOCK_WORDS],
                moving=identity16[0:_P_MAX, 0:_P_MAX],
                is_transpose=True,
                accumulate=False,
            )


def _dr_key_planes(rows, planes, engine=nisa.engine.vector):
    """Write the e4m3 double_row key planes ``planes`` ``[128, 2, 2, keys]`` FP8 of the FP8 cache rows ``rows`` (``rows_dr``).

    Element ``(c, i, k, key)`` is latent feature ``256 k + 2c + i`` of the key, so ``planes[:, i, :, keys]`` is the
    stationary ``[128, 2, 128]`` of double_row pass ``i`` of a block. The transposes of ``_dr_key_transposes`` move the
    feature pairs ``(2c, 2c + 1)`` to partition ``c``, and one copy for each byte on ``engine`` (VectorE by default, as
    ``tkg_mla_paged._dr_queries`` does; ``rows_dr_act_copies``: ScalarE) takes byte ``i`` of the words; one engine reads the
    bank.
    """
    group_blocks = rows.shape[1]
    num_keys = group_blocks * _BLOCK
    identity16 = nl.shared_identity_matrix(_P_MAX, dtype=nl.bfloat16)
    words_psum = nl.ndarray((_P_MAX, _DR_BLOCKS, num_keys), dtype=nl.bfloat16, buffer=nl.psum)
    _dr_key_transposes(rows, words_psum, identity16)
    words = words_psum.view(nl.float8_e4m3fn).reshape_dim(2, (num_keys, 2))
    for byte in range(2):
        nisa.tensor_copy(
            dst=planes[0:_P_MAX, byte, 0:_DR_BLOCKS, 0:num_keys],
            src=words[0:_P_MAX, 0:_DR_BLOCKS, 0:num_keys, byte],
            engine=engine,
        )


def _dr_key_words(rows, words, engine=nisa.engine.vector):
    """Write the 2-byte key words ``words`` ``[128, 2, keys]`` BF16 of the FP8 cache rows ``rows`` (``rows_dr_inplace``).

    The transposes of ``_dr_key_transposes`` (the same ones ``_dr_key_planes`` runs) put the feature pair ``(2c, 2c +
    1)`` on partition ``c`` as BF16 words in PSUM, then one exact 16-bit integer copy on ``engine`` (VectorE by default;
    ``rows_dr_act_copies``: ScalarE) writes the words in place: a BF16 copy would flush a subnormal word to zero and drop
    its low byte. Byte ``i`` of word ``(c, block, key)`` is latent feature ``256 block + 2c + i``, the plane element of
    ``_dr_key_planes`` that ``_dr_word_stationary`` reads as a step-2 FP8 view.
    """
    group_blocks = rows.shape[1]
    num_keys = group_blocks * _BLOCK
    identity16 = nl.shared_identity_matrix(_P_MAX, dtype=nl.bfloat16)
    words_psum = nl.ndarray((_P_MAX, _DR_BLOCKS, num_keys), dtype=nl.bfloat16, buffer=nl.psum)
    _dr_key_transposes(rows, words_psum, identity16)
    nisa.tensor_copy(dst=words.view(nl.uint16), src=words_psum.view(nl.uint16), engine=engine)


def _dr_query_planes(dr_rows, cqs, heads, sb):
    """Return the e4m3 double_row query planes ``[128, 2, 2, R * heads]`` FP8 of the chunk rows ``cqs`` (``rows_dr``):
    the moving operand of double_row pass ``i`` is ``[:, i]``, from the stationaries ``dr_rows`` ``[128, C, 2, 2,
    heads]`` of the gather (``_dr_gather_tile_queries``), row after row. Exact FP8 copies."""
    planes = sb.tile((_P_MAX, 2, _DR_BLOCKS, len(cqs) * heads), nl.float8_e4m3fn)
    for i, cq in enumerate(cqs):
        for byte in range(2):
            nisa.tensor_copy(
                dst=planes[0:_P_MAX, byte, 0:_DR_BLOCKS, i * heads : (i + 1) * heads],
                src=dr_rows[0:_P_MAX, cq, byte, 0:_DR_BLOCKS, 0:heads],
                engine=nisa.engine.vector,
            )
    return planes


def _dr_key_stationary(latent_keys, byte: int, keys: slice):
    """Return the stationary ``[128, 2, 128]`` of double_row pass ``byte`` of key block ``keys`` in the key planes."""
    return latent_keys[0:_P_MAX, byte, 0:_DR_BLOCKS, keys]


def _dr_word_stationary(latent_keys, byte: int, keys: slice):
    """Return the stationary ``[128, 2, 128]`` of double_row pass ``byte`` of key block ``keys`` in the key words
    (``rows_dr_inplace``): byte ``byte`` of each word, a step-2 FP8 view of the word tile along the keys (offset =
    ``byte``), which is feature ``256 k + 2c + byte`` of the key, the plane element of ``_dr_key_stationary``."""
    return latent_keys.view(nl.float8_e4m3fn)[0:_P_MAX, 0:_DR_BLOCKS, 2 * keys.start + byte : 2 * keys.stop : 2]


def _mx_key_prep(rows, latent_keys):
    """Write the MX key words of the FP8 cache rows ``rows`` ``[128 keys, blocks, 576]`` into ``latent_keys``
    ``[128 words, keys]`` FP32 (``rows_mx_keys``): word ``(c, key)`` holds latent features ``4c .. 4c + 3`` of the key
    (the FP32 view of ``float8_e4m3fn_x4``). This replaces the 4 BF16 latent transposes of the ``bf16`` form.

    One 32-bit TensorE transpose of each 128 keys of the latent words (``[128 keys, 128 words] -> [128 words, 128
    keys]``) moves the bits into one PSUM bank, then one byte-exact copy through a uint16 view on ScalarE (an FP32
    copy of the 4 packed FP8 bytes could flush an FP32 subnormal, so K would not stay exact).
    """
    group_blocks = rows.shape[1]
    num_keys = group_blocks * _BLOCK
    words = _KV_LORA // _MX_PACK
    row_words = rows.view(nl.float32)
    words_psum = nl.ndarray((_P_MAX, num_keys), dtype=nl.float32, buffer=nl.psum)
    for local_block in range(group_blocks):
        nisa.nc_transpose(
            dst=words_psum[0:_P_MAX, local_block * _BLOCK : (local_block + 1) * _BLOCK],
            data=row_words[0:_P_MAX, local_block, 0:words],
            engine=nisa.engine.tensor,
        )
    nisa.tensor_copy(
        dst=latent_keys.view(nl.uint16)[0:_P_MAX, 0 : 2 * num_keys],
        src=words_psum.view(nl.uint16)[0:_P_MAX, 0 : 2 * num_keys],
        engine=nisa.engine.scalar,
    )


def _mx_key_stationary(latent_keys, keys: slice):
    """Return the MX stationary ``[128 words, 128 keys]`` ``float8_e4m3fn_x4`` of key block ``keys`` (``rows_mx_keys``)."""
    return latent_keys[0:_P_MAX, keys].view(nl.float8_e4m3fn_x4)


def _mx_query_planes(mx_rows, cqs, heads, sb, engine=nisa.engine.vector):
    """Return the contiguous MX query planes ``[128 words, parts, R * heads]`` FP32 of the rows ``cqs`` of the per-call
    gather tile ``mx_rows`` ``[128 words, cr, parts, heads]`` (``rows_mx_keys``): plane ``part`` holds the query words
    of the rows ``cqs`` row after row, so a block's moving is the contiguous slice
    ``planes[:, part, r0 * heads : r1 * heads]`` (``nc_matmul_mx`` needs each operand to be one contiguous run).

    One copy a (row, part), each a plain slice of a uint16 view of the FP32 words (an FP32 copy of the 4 packed FP8
    bytes could flush an FP32 subnormal, and the row stride makes a strided ``.ap`` on the uint16 view mismatch the FP32
    free size), as ``_dr_query_planes`` copies the double_row planes row after row."""
    parts = mx_rows.shape[2]
    planes = sb.tile((_P_MAX, parts, len(cqs) * heads), nl.float32)
    dst = planes.view(nl.uint16)
    src = mx_rows.view(nl.uint16)
    for i, cq in enumerate(cqs):
        for part in range(parts):
            nisa.tensor_copy(
                dst=dst[0:_P_MAX, part, 2 * i * heads : 2 * (i + 1) * heads],
                src=src[0:_P_MAX, cq, part, 0 : 2 * heads],
                engine=engine,
            )
    return planes


def _mx_query_words(q_all, q_mx_all, tile_base, tile_tokens, local_heads, rows_split, token_major, sb):
    """Round the latent query of one gather tile ``q_all`` to e4m3 and transpose it as 32-bit words into ``q_mx_all``
    ``[128 words, cr, 1, heads]`` FP32 (``rows_mx_keys``): word ``(c, r, 0, 3t + h)`` holds latent features
    ``4c .. 4c + 3`` of head ``3t + h`` of row ``r``.

    hi = e4m3(q), nearest even with saturation at 448 (unit scale), as the GPU decode rounds its query. The 32-bit
    transposes move the bits, and the PSUM -> SBUF copies run through a uint16 view (an FP32 copy of the packed FP8
    bytes could flush an FP32 subnormal, which small queries hit: an e4m3 zero in the top byte makes the word an FP32
    subnormal).
    """
    fp8 = nl.float8_e4m3fn
    gather_rows = rows_split[0] * rows_split[1]
    absorb = local_heads * _KV_LORA
    words = _KV_LORA // _MX_PACK
    q_rows = q_all[0:gather_rows, 0:absorb]
    q8 = sb.tile((gather_rows, absorb), fp8)
    nisa.tensor_scalar(
        dst=q8,
        data=q_rows,
        op0=nl.minimum,
        operand0=_E4M3_MAX,
        op1=nl.maximum,
        operand1=-_E4M3_MAX,
        engine=nisa.engine.vector,
    )
    # The uint16 view keeps the (word, row) pair contiguous, so the reorder is a byte move: (0, token, rank, pair).
    mx_order = (0, 1, 2, 3) if token_major else (0, 2, 1, 3)
    dst_u16 = q_mx_all.view(nl.uint16)
    q8_words = q8.view(nl.float32)
    for local_head in range(local_heads):
        q_mx_psum = nl.ndarray((_P_MAX, gather_rows), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(
            dst=q_mx_psum,
            data=q8_words[0:gather_rows, local_head * words : (local_head + 1) * words],
            engine=nisa.engine.tensor,
        )
        # dst_u16: [128, cr, 1, 2 * heads] -> slice part 0 -> [128, tile_tokens, _TP, local_heads, 2] pick local_head.
        dst = dst_u16[:, tile_base : tile_base + tile_tokens, 0, :].reshape_dim(2, (_TP, local_heads, 2))[
            :, :, :, local_head, :
        ]
        src = q_mx_psum.view(nl.uint16).reshape_dim(1, (gather_rows, 2)).reshape_dim(1, rows_split).permute(mx_order)
        nisa.tensor_copy(
            dst=dst,
            src=src,
            engine=nisa.engine.scalar if local_head % 2 else nisa.engine.vector,
        )


def _mx_natural_query_tiles(
    q_all, q_nat_all, tile_base, tile_tokens, local_heads, rows_split, nope_order, opposite=False
):
    """Write the latent queries of one gather tile ``q_all`` into ``q_nat_all`` ``[128, cr, 4, heads]`` BF16 in the
    natural MX grouping (``packed_prior_options`` ``mx_natural``): element ``(c, r, t, 3g + h)`` is latent feature
    ``4c + t`` of head ``3g + h`` of row ``r``, so the MX query words of the stage rows (``tkg_mla_paged._stage_query_rows_p2``
    on these tiles) hold features ``4c .. 4c + 3`` in word ``c``, the grouping of the 32-bit key words of an FP8 key row.

    Lane ``t`` of a head is one TensorE transpose of every 4th latent feature of the rows from feature ``t`` (a step-4
    stationary read; exact), and one copy a head moves the 4 lanes to the rows' columns on the engine of that head's
    ``q_nope_all`` copy in the gather (``rows_split`` and ``nope_order``: the row order of the tile,
    ``mla_block_attention``); ``opposite`` (``q_prep_act``): on the other engine, so the copies of a tile split evenly.
    """
    gather_rows = rows_split[0] * rows_split[1]
    for local_head in range(local_heads):
        nat_psum = nl.ndarray((_P_MAX, _MX_PACK, gather_rows), dtype=nl.bfloat16, buffer=nl.psum)
        for lane in range(_MX_PACK):
            nisa.nc_transpose(
                dst=nat_psum[0:_P_MAX, lane, 0:gather_rows],
                data=q_all[0:gather_rows, local_head * _KV_LORA + lane : (local_head + 1) * _KV_LORA : _MX_PACK],
                engine=nisa.engine.tensor,
            )
        nisa.tensor_copy(
            dst=q_nat_all[:, tile_base : tile_base + tile_tokens, :, :].reshape_dim(3, (_TP, local_heads))[
                :, :, :, :, local_head
            ],
            src=nat_psum.reshape_dim(2, rows_split).permute(nope_order),
            engine=nisa.engine.scalar if (local_head % 2) != bool(opposite) else nisa.engine.vector,
        )


KEY_FORMS = ("bf16", "dr", "dr_inplace", "mx_single")
"""The latent key forms of the rows form (``_KeyPath``): BF16 feature tiles, the FP8 double_row planes of ``rows_dr``,
the 2-byte key words of ``rows_dr_inplace`` (the same scores as ``dr``, read as a step-2 FP8 view), and the MX key
words of ``rows_mx_keys`` ("mx_single")."""


class _KeyPath:
    """The latent key path of the rows form: the key tile of a key group, its prep from the cache rows, the query tile
    of the rows of a sequence, and the latent score matmuls of a key block.

    The rows paths call only these four, so a new key form changes this class and the choice of the form in
    ``mla_block_attention``. ``bf16``: 4 BF16 feature tiles ``[128, 4, keys]`` from PE transposes, with the BF16 queries
    of ``q_nope_all`` moving. ``dr`` (``rows_dr``): the e4m3 double_row key planes of ``_dr_key_planes`` (the byte
    copies on ``copy_engine``) as the stationary, with the query planes of ``_dr_query_planes`` moving. ``dr_inplace``
    (``rows_dr_inplace``): the 2-byte key words of ``_dr_key_words`` (one word copy on ``copy_engine``, in place of the
    byte copies) read by ``_dr_word_stationary`` as a step-2 FP8 view; every score is bitwise equal to ``dr``.
    ``mx_single`` (``rows_mx_keys``): the MX key words of ``_mx_key_prep`` (``float8_e4m3fn_x4``) as the stationary,
    with the contiguous MX query planes of ``_mx_query_planes`` (the per-call gather words of ``_mx_query_words``)
    moving; the latent part is 1 ``nc_matmul_mx``, with the unit E8M0 scales ``mx_scale``.
    """

    def __init__(self, form: str = "bf16", copy_engine=nisa.engine.vector, mx_scale=None):
        kernel_assert(form in KEY_FORMS, f"the key form must be one of {KEY_FORMS}, got {form!r}")
        self.form = form
        self.copy_engine = copy_engine
        self._dr = form in ("dr", "dr_inplace")
        self.mx_scale = mx_scale
        self.mx = form == "mx_single"

    def key_tile(self, group_keys: int, sb):
        """Return the latent key tile of one key group of ``group_keys`` keys."""
        if self.mx:
            return sb.tile((_P_MAX, group_keys), nl.float32)
        return _latent_key_tile(group_keys, self.form == "dr", sb, words=self.form == "dr_inplace")

    def query_tile(self, dr_rows, cqs, heads, sb, mx_rows=None):
        """Return the latent query tile of the rows ``cqs`` (``dr`` and ``dr_inplace``: the query planes of the gather
        stationaries ``dr_rows``; ``mx``: the contiguous MX query planes of the per-call words ``mx_rows``), or None
        (``bf16`` reads ``q_nope_all``)."""
        if self.mx:
            return _mx_query_planes(mx_rows, cqs, heads, sb, self.copy_engine)
        return _dr_query_planes(dr_rows, cqs, heads, sb) if self._dr else None

    def prep(self, rows, latent_keys):
        """Write the latent key tile ``latent_keys`` of the cache rows ``rows`` ``[128, blocks, 576]`` of a key group."""
        if self.mx:
            _mx_key_prep(rows, latent_keys)
            return
        if self.form == "dr":
            _dr_key_planes(rows, latent_keys, self.copy_engine)
            return
        if self.form == "dr_inplace":
            _dr_key_words(rows, latent_keys, self.copy_engine)
            return
        group_blocks = rows.shape[1]
        num_keys = group_blocks * _BLOCK
        for latent_tile in range(_KV_LORA_TILES):
            latent_psum = nl.ndarray((_P_MAX, num_keys), dtype=nl.bfloat16, buffer=nl.psum)
            for local_block in range(group_blocks):
                _transpose_rows(
                    dst=latent_psum[0:_P_MAX, local_block * _BLOCK : (local_block + 1) * _BLOCK],
                    data=rows[0:_P_MAX, local_block, latent_tile * _P_MAX : (latent_tile + 1) * _P_MAX],
                )
            nisa.tensor_copy(
                dst=latent_keys[0:_P_MAX, latent_tile, 0:num_keys],
                src=latent_psum,
                engine=nisa.engine.scalar if latent_tile % 2 else nisa.engine.vector,
            )

    def scores(self, scores_psum, pieces, latent_keys, keys, q_nope_all, q_tile, cqs, stride, heads):
        """Start the scores of key block ``keys`` (a slice of the group keys) of the rows ``cqs`` (row step ``stride``):
        the latent matmuls of each row piece of ``pieces`` into its PSUM tile of ``scores_psum``. The caller's rope
        matmul accumulates onto them."""
        if self.mx:
            for (r0, r1), psum in zip(pieces, scores_psum):
                nisa.nc_matmul_mx(
                    dst=psum,
                    stationary=_mx_key_stationary(latent_keys, keys),
                    moving=q_tile[0:_P_MAX, 0, r0 * heads : r1 * heads].view(nl.float8_e4m3fn_x4),
                    stationary_scale=self.mx_scale[0:_P_MAX, 0:_BLOCK],
                    moving_scale=self.mx_scale[0:_P_MAX, 0 : (r1 - r0) * heads],
                    accumulate=False,
                )
            return
        if self._dr:
            stationary = _dr_word_stationary if self.form == "dr_inplace" else _dr_key_stationary
            for byte in range(2):
                for (r0, r1), psum in zip(pieces, scores_psum):
                    nisa.nc_matmul(
                        dst=psum,
                        stationary=stationary(latent_keys, byte, keys),
                        moving=q_tile[0:_P_MAX, byte, 0:_DR_BLOCKS, r0 * heads : r1 * heads],
                        accumulate=byte > 0,
                        perf_mode=nisa.matmul_perf_mode.double_row,
                    )
            return
        tiles = q_nope_all.shape[2]
        # Each key tile serves the pieces back to back, so its stationary load can serve all rows.
        for latent_tile in range(_KV_LORA_TILES):
            for (r0, r1), psum in zip(pieces, scores_psum):
                nisa.nc_matmul(
                    dst=psum,
                    stationary=latent_keys[0:_P_MAX, latent_tile, keys],
                    moving=_row_view(q_nope_all, cqs[r0], r1 - r0, stride, tiles * heads, latent_tile * heads, _P_MAX),
                    accumulate=latent_tile > 0,
                )


def _set_fold_key_rows(buffers):
    """Write the constant rows of the folded reference in the rope key tile of each group buffer: the group transposes
    rewrite the mask rows, the rows above them stay zero, then come the ones rows."""
    group_keys = _GROUP_BLOCKS * _BLOCK
    for _key_rows, _latent_keys, rope_keys, _threshold, _lower in buffers:
        nisa.memset(dst=rope_keys[_BIAS_ROW:_REF_ROW, 0:group_keys], value=0.0)
        nisa.memset(dst=rope_keys[_REF_ROW : _REF_ROW + _REF_PARTS, 0:group_keys], value=1.0)


def _rows_seq_setup(kv_pool, loop_groups: int, head_groups: int, key_path, folded_reference: bool, bias_end: int, sb):
    """Return the group buffers of the rows loop that every sequence of the layer takes (``rows_seq_hoist``).

    One ``(key rows, latent keys, rope keys, threshold, lower)`` set for each group, as ``_attend_prior_loop`` makes
    for one sequence (no key rows for the first ``head_groups`` groups, which take the prefetch head tiles), with the
    constant rows of the folded reference written once.
    """
    fp32 = nl.float32
    group_keys = _GROUP_BLOCKS * _BLOCK
    buffers = [
        (
            None if g < head_groups else sb.tile((_P_MAX, _GROUP_BLOCKS, _CACHE_WIDTH), kv_pool.dtype),
            key_path.key_tile(group_keys, sb),
            sb.tile((_P_MAX, group_keys), nl.bfloat16),
            sb.tile((_P_MAX, 1), fp32),
            sb.tile((_P_MAX, 1), fp32),
        )
        for g in range(loop_groups)
    ]
    if folded_reference:
        kernel_assert(bias_end <= _REF_ROW, f"folded_reference takes at most {_REF_ROW - _BIAS_ROW} mask rows")
        _set_fold_key_rows(buffers)
    return buffers


def _next_sequence(sequences, b):
    """Return the sequence after ``b`` in ``sequences``, or None."""
    i = sequences.index(b)
    return sequences[i + 1] if i + 1 < len(sequences) else None


class _RowsPrefetch:
    """The first key groups of each loop iteration of the rows form, loaded one iteration early (``rows_prefetch``).

    The pages of the first ``groups`` key groups of an iteration (those of its first attend call) go to one of two
    head tile sets, by the order of the sequence. Before a sequence's loop, its prologue loads the head groups of the
    next sequence; inside the loop, after the first attend call (the last reader of the head pages), each iteration
    loads those of the next iteration. A padded copy of each block table row gives the iteration after the last one
    a row of page 0, so the last load stays in range. The loads read the same pages as the loop does, so the bits do
    not change.
    """

    def __init__(self, kv_pool, block_table, local_blocks, loop_groups, sequences, tag, sb, pad_rows=1):
        int32 = nl.int32
        self.kv_pool = kv_pool
        self.groups = min(_TS_GROUPS, loop_groups)
        self.loop_blocks = loop_groups * _GROUP_BLOCKS
        self.tag = tag
        self.order = {b: i for i, b in enumerate(sequences)}
        rows = local_blocks // self.loop_blocks
        zero_row = sb.tile((1, self.loop_blocks), int32)
        nisa.memset(dst=zero_row, value=0)
        self.tables = {}
        for b in sequences:
            table = nl.ndarray(
                (rows + pad_rows, self.loop_blocks),
                dtype=int32,
                buffer=nl.private_hbm,
                name=f"{tag}_mla_pre_table_{b}",
            )
            nisa.dma_copy(
                dst=table[0:rows, 0 : self.loop_blocks],
                src=block_table[b : b + 1, 0:local_blocks].reshape((rows, self.loop_blocks)),
            )
            for pad in range(rows, rows + pad_rows):
                nisa.dma_copy(dst=table[pad : pad + 1, 0 : self.loop_blocks], src=zero_row)
            self.tables[b] = table
        self.heads = [
            [sb.tile((_P_MAX, _GROUP_BLOCKS, _CACHE_WIDTH), kv_pool.dtype) for _ in range(self.groups)]
            for _ in range(2)
        ]
        self.entries = [sb.tile((1, self.groups * _GROUP_BLOCKS), int32) for _ in range(2)]

    def head_tiles(self, b):
        """Return the head key-row tiles of sequence ``b``."""
        return self.heads[self.order[b] % 2]

    def load(self, b, step=None, row=1, pitch=1, suffix=""):
        """Load the head groups of iteration 0 of ``b`` (``step`` None), or of iteration ``pitch * step + row``
        (default ``step + 1``); ``suffix`` ends the instruction names."""
        parity = self.order[b] % 2
        entries = self.entries[parity]
        width = self.groups * _GROUP_BLOCKS
        pattern = [[self.loop_blocks, 1], [1, width]]
        table = self.tables[b]
        if step is None:
            src = table.ap(pattern=pattern, offset=0)
        else:
            # The scalar offset steps by the row of the view (``ap`` ``indirect_dim``): rows of ``pitch`` iterations.
            view = table if pitch == 1 else table.reshape((table.shape[0] // pitch, pitch * self.loop_blocks))
            pattern = [[pitch * self.loop_blocks, 1], [1, width]]
            src = view.ap(pattern=pattern, offset=row * self.loop_blocks, scalar_offset=step, indirect_dim=0)
        name = f"{self.tag}_mla_pre_table_load_{b}" + ("" if step is None else "_next") + suffix
        nisa.dma_copy(dst=entries, src=src, dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync, name=name)
        num_pages = self.kv_pool.shape[0]
        nisa.tensor_scalar(
            dst=entries, data=entries, op0=nl.minimum, operand0=float(num_pages - 1), op1=nl.maximum, operand1=0.0
        )
        for g, key_rows in enumerate(self.heads[parity]):
            for lb in range(_GROUP_BLOCKS):
                entry = g * _GROUP_BLOCKS + lb
                nisa.dma_copy(
                    dst=key_rows[0:_P_MAX, lb, 0:_CACHE_WIDTH],
                    src=self.kv_pool.ap(
                        pattern=[[_CACHE_WIDTH, _BLOCK], [1, _CACHE_WIDTH]],
                        offset=0,
                        scalar_offset=entries[0:1, entry : entry + 1],
                        indirect_dim=0,
                    ),
                    dge_mode=nisa.dge_mode.hwdge,
                    engine=nisa.engine.sync,
                    name=f"{self.tag}_mla_pre_page_{b}_{g}_{lb}" + ("" if step is None else "_next") + suffix,
                )


class _PageGather:
    """The page gathers of the plain prior-key loop (``loop_page_gather``; the loop form of ``tkg_mla_paged``'s per-token
    ``loop_page_gather``).

    One GpSimd software DGE gather loads the 4 pages of a key group (512 rows of 576 bytes, an index tile ``[128, 4]``
    int32) into the group's own page tile, with the same bytes in the same places as the 4 page loads: row ``4 p + j``
    of the tile (page ``j``, row ``p``) is index ``[k % 128, k // 128]``, so the index of partition ``q`` and column
    ``c`` is ``128 clamp(e[4 g + q % 4]) + 32 c + q // 4`` (``tkg_mla_paged._gather_index``). Before the first loop every
    sequence gets the indices of its first iteration (and with ``carry`` its group 0, into a tile of its own); each
    iteration loads the next table row onto every partition and builds the next iteration's indices after group
    ``_GATHER_CARRY_GROUP`` (with ``carry`` it also gathers the next group 0 then), and copies them to the current
    indices after its last gather. The tiles live across the back edge, so they need an explicit SBUF region.
    ``page_dtype`` (default: the pool's): the dtype of the page tiles, BF16 with ``loop_cast_engines`` (the gathers then
    cast the FP8 bytes, as the casting page loads do).
    """

    def __init__(self, form, kv_pool, block_table, local_blocks, loop_groups, sequences, tag, sb, page_dtype=None):
        int32 = nl.int32
        self.form = form
        self.carry = form in ("carry", "carry_pool")
        self.pool = form == "carry_pool"
        self.tag = tag
        self.groups = loop_groups
        self.loop_blocks = loop_groups * _GROUP_BLOCKS
        self.table_rows = local_blocks // self.loop_blocks
        num_pages = kv_pool.shape[0]
        self.num_pages = num_pages
        self.kv_rows = kv_pool.reshape((num_pages * _BLOCK, _CACHE_WIDTH))
        self.row_table = sb.tile((_P_MAX, self.loop_blocks), int32)
        self.page_select = sb.tile((_P_MAX, loop_groups), nl.uint32)
        self.row_offsets = sb.tile((_P_MAX, _GROUP_BLOCKS), int32)
        self.group_pages = sb.tile((_P_MAX, loop_groups), int32)
        self.index_next = sb.tile((_P_MAX, loop_groups, _GROUP_BLOCKS), int32)
        self.index_cur = {b: sb.tile((_P_MAX, loop_groups, _GROUP_BLOCKS), int32) for b in sequences}
        self.group0 = {
            b: sb.tile((_P_MAX, _GROUP_BLOCKS, _CACHE_WIDTH), page_dtype or kv_pool.dtype)
            for b in (sequences if self.carry else ())
        }
        _gather_constants(self.page_select, self.row_offsets, sb)
        # The GpSimd build reads the table clamped once (GpSimd has no minimum / maximum on Trn2).
        clamped = (
            _clamped_table(block_table, block_table.shape[0], local_blocks, num_pages, tag, sb) if self.pool else None
        )
        self.index_rows = {
            b: (clamped if self.pool else block_table)[b : b + 1, 0:local_blocks].reshape(
                (self.table_rows, self.loop_blocks)
            )
            for b in sequences
        }
        with sb.scope("gather_first"):
            first = sb.tile((_P_MAX, self.loop_blocks), int32)
            for b in sequences:
                nisa.dma_copy(
                    dst=first,
                    src=block_table[b : b + 1, 0 : self.loop_blocks].ap(
                        pattern=[[0, _P_MAX], [1, self.loop_blocks]], offset=0
                    ),
                    dge_mode=nisa.dge_mode.hwdge,
                    engine=nisa.engine.sync,
                    name=f"{tag}_mla_gather_table_first_{b}",
                )
                self.build(self.index_cur[b], first, pool=False)
                if self.carry:
                    self.gather(self.group0[b], self.index_cur[b], 0, f"{tag}_mla_gather_first_{b}")

    def build(self, dst, table, pool=None):
        """The gather indices of the table row ``table`` (on every partition) into ``dst``."""
        return _gather_index(
            dst,
            table,
            self.group_pages,
            self.page_select,
            self.row_offsets,
            self.num_pages,
            self.pool if pool is None else pool,
        )

    def gather(self, rows, index, group, name):
        """Gather the 4 pages of key group ``group`` of ``index`` into the page tile ``rows``."""
        return nisa.dma_copy(
            dst=rows,
            src=self.kv_rows.ap(
                pattern=[[_CACHE_WIDTH, _GROUP_BLOCKS * _BLOCK], [1, _CACHE_WIDTH]],
                offset=0,
                vector_offset=index[0:_P_MAX, group, 0:_GROUP_BLOCKS],
                indirect_dim=0,
            ),
            dge_mode=nisa.dge_mode.swdge,
            engine=nisa.engine.dma,
            name=name,
        )

    def load_next_row(self, b, step):
        """Load the table row of the iteration after ``step`` of ``b`` onto every partition (the row after the last
        one is row 0: its gathers read pages in range and nothing reads them)."""
        nisa.dma_copy(
            dst=self.row_table,
            src=self.index_rows[b].ap(
                pattern=[[0, _P_MAX], [1, self.loop_blocks]],
                offset=0,
                scalar_offset=(step + 1) % self.table_rows,
                indirect_dim=0,
            ),
            dge_mode=nisa.dge_mode.hwdge,
            engine=nisa.engine.sync,
            name=f"{self.tag}_mla_gather_table_next_{b}",
        )


def _attend_prior_loop(
    b,
    chunk_rows,
    kv_pool,
    block_table,
    loop_row,
    q_nope_all,
    q_rope_all,
    mask_threshold,
    slot_offset,
    state,
    softmax,
    tokens,
    q_rows,
    local_blocks,
    heads,
    lower_threshold,
    bitwise,
    tag,
    sb,
    lay,
    n_mask=1,
    ts=False,
    loop_group_cap=0,
    clamp_pages=True,
    weights_on_act=False,
    first_static=False,
    scores_ahead=False,
    folded_reference=False,
    first_static_after=None,
    rows_v_stationary=False,
    dr_rows=None,
    prefetch=None,
    next_b=None,
    key_path=None,
    pair_lag=False,
    pads=None,
    hoisted=None,
    ahead=None,
    mx_rows=None,
    p2=None,
    preload=False,
    pair_tail=False,
    unroll=1,
    bf16_pages=False,
    gather=None,
    tail_groups=0,
    row_pipeline=False,
):
    """Attend the prior-key groups of sequence ``b`` with one runtime loop over its key groups.

    In bitwise mode the per-group threshold gains the row index ``j`` so each row of the sequence masks at
    its own position ``position[b] + j``; the current-block read follows the owner-write through the
    kv_pool tensor order. The local threshold comes from ``loop_row``, and ``mask_threshold`` is unused. With
    ``n_mask`` mask rows (fused masks), the thresholds of the rows come from ``mask_threshold[j, b]`` (one on
    each mask row) and move by the key span of each iteration, so the loop stores no register.

    With ``first_static`` and at most 2 iterations (``local_blocks`` up to 2 x the blocks of one iteration) the table
    entries of the sequence load and clamp once, iteration 0 runs as static code, and the loop runs the other iteration,
    if any; with more iterations the loop runs them all. A sequence with no prior key group then attends iteration 0
    too, whose keys are all masked: its weights are exact zeros, so its state does not change on finite pages. Each row
    keeps its instructions and group order. The page loads of iteration 0 after its first group pair follow
    ``first_static_after`` (the query load) in the schedule, so they do not hold the queue of that load.

    With ``scores_ahead`` (transposed scores) the weights of each group pair come before the sums of the pair before
    it, so the PE runs the next score matmuls while the vector and scalar engines finish the weights of a pair. Each
    sum and numerator keeps its order. The pairs of one iteration take turns in two weight tiles.
    ``folded_reference`` writes the reference into the query rope tile once and 3 ones rows into each key buffer.
    ``rows_v_stationary`` and ``dr_rows`` (the double_row query stationaries of ``rows_dr``) are the
    ``mla_block_attention`` keys. With ``prefetch`` (``_RowsPrefetch``; ``rows_prefetch``) the head key groups of each
    iteration come one iteration early, and before the loop the head groups of sequence ``next_b`` (if any) start to
    load; not with ``first_static`` or ``scores_ahead``. With ``pair_lag`` (``rows_pair_lag``) the sums matmul of each
    block runs ``_SUMS_LAG`` blocks late, and the value matmuls of each group pair run after the transposes of the next
    pair (the last pair's at the end of the iteration); the pairs take turns in two weight tiles, and the head pages of
    the next iteration load after the value matmuls of the head pair, their last reader. With ``pads``
    (``_loop_metadata`` ``pad_skip``) a padding sequence has no iterations and its weight sums take 1; the prologue
    still loads its head pages (page loads only: a loop around them would cost each sequence a scheduling wall).
    ``hoisted``: None, or the group buffers of every sequence from ``_rows_seq_setup`` (``rows_seq_hoist``), set up
    before the first loop with the constant rows of the folded reference. ``key_path``: the latent key form
    (``_KeyPath``; None = BF16). ``ahead``: None, or ``(q_tile,)``, the query tile of the sequence, made with its folded
    reference rows before the first loop (``rows_setup_ahead``; the reference columns are then not made here).
    ``p2`` (``loop_p2``, transposed scores): ``(state, seq, seqs)``, the (row, head) state of the sequence that the
    numerators and sums add into (``_attend_group_rows``); ``state`` is then unused.
    ``preload`` (``loop_preload``, with ``first_static`` and 2 iterations): the static iteration 0 also loads the pages
    of the first ``loop_groups - 2`` key groups of iteration 1 into their group buffers, after its own last read of
    each buffer (the buffer's write follows that read), and the loop body does not load them again: the loop then
    starts on pages that landed during iteration 0, not on pages that it issues at its start. The same pages in the same
    buffers, read by the same instructions, so the bits do not change. The fence of the loop entry waits for the loads.

    With ``pair_tail`` (``rows_pair_tail``; transposed scores, fused masks, the head-page prefetch, iterations of 2 group
    pairs) ``loop_row`` holds the full iterations of the sequence (``_loop_metadata`` ``pair_tail``) and, at column
    ``cols x tokens + b``, its tail count (0 or 1): the iteration after the full ones whose second pair would hold only
    masked keys runs as its first pair alone. That pair is the head pair, which the last full iteration (or the prologue)
    already loaded, so the tail loads no page and loads no next head pair. Its pair call is that of the iteration it
    replaces; the dropped pair's masked keys only added exact zero weights after it, so the state keeps every bit (an
    exact-zero numerator entry can keep a -0 that the zero adds made +0).

    With ``bf16_pages`` (``loop_cast_engines``) the group buffers are BF16: the page loads cast the FP8 bytes, the clamp
    keeps the FP8 bound (a NaN byte gives the same bound as on the FP8 tile), and the transposes and the value matmuls
    read the BF16 rows in place of the FP8 rows and their BF16 copies.

    With ``gather`` (``_PageGather``; ``loop_page_gather``) one GpSimd gather loads each key group's 4 pages: the gather of
    group g + ``_GATHER_LEAD`` issues before the clamps of group g (the first groups at the iteration start; with
    ``carry`` group 0 comes from the iteration before, or from before the loop), and each iteration loads the next table
    row, builds the next indices after group ``_GATHER_CARRY_GROUP`` (with ``carry`` it gathers the next group 0 then:
    group 0's tile has no reader left) and copies them to the current indices after its last gather. The same pages in
    the same tiles, read by the same instructions, so the bits do not change.

    With ``tail_groups`` (``loop_tail_groups``; the plain loop; n tails t_0 > ... of ``loop_tails``) ``loop_row`` holds
    the full iterations of the sequence and, at column ``cols x tokens + n b + k``, the count of tail t_k
    (``_loop_metadata`` ``tail_groups``): 1 when its last iteration would hold 1 to t_k key groups with a prior key and
    no smaller tail holds them, else 0. A loop after the first for each tail, largest first, runs that iteration with
    only its first t_k groups (its table row and, with ``gather``, the indices and the carried group 0 the loop already
    loaded; it loads no next row); at most one of them runs. The dropped groups hold only masked keys: their weights are
    exact zeros and the running max stays, so the state keeps every bit (an exact-zero state entry can keep a -0 that
    the zero adds made +0).

    With ``row_pipeline`` (``loop_row_pipeline``; the plain loop with fused masks and the capped softmax) the rows of the
    iteration run as a two-stage software pipeline over the (group, row) items: the scores, cap and exponentials of an
    item (``_attend_group_scores``) come before the weight transposes, value matmuls and state add of the item before it
    (``_attend_group_values``), so TensorE runs the next scores while the vector and scalar engines finish an item. The
    items take turns in two weight tiles. Each row keeps its instructions and its group order, so the state keeps every
    bit.
    """
    kernel_assert(
        not row_pipeline
        or (
            n_mask > 1
            and not softmax.exact
            and not (ts or rows_v_stationary or pair_lag or scores_ahead)
            and p2 is None
        ),
        "loop_row_pipeline is a form of the plain loop with fused masks and the capped softmax (no transposed scores, "
        "rows_v_stationary, rows_pair_lag or scores_ahead)",
    )
    kernel_assert(
        not tail_groups
        or (
            tail_groups in LOOP_TAILS.values()
            and max(loop_tails(tail_groups), default=0)
            < _loop_groups(kv_pool, local_blocks // _GROUP_BLOCKS, loop_group_cap)
            and not (ts or first_static or preload or pair_tail or unroll != 1)
            and prefetch is None
            and hoisted is None
        ),
        f"loop_tail_groups is a form of the plain loop (no transposed scores, first_static, rows_prefetch, rows_seq_hoist, "
        f"rows_pair_tail or rows_unroll2) with tails of 4, 2 or 1 groups (LOOP_TAILS) below the groups of an "
        f"iteration and the full-iteration tail, got {tail_groups}",
    )
    kernel_assert(
        gather is None
        or not (ts or first_static or preload or pair_tail or unroll != 1)
        and prefetch is None
        and hoisted is None,
        "loop_page_gather is a form of the plain loop (no transposed scores, first_static, rows_prefetch, rows_seq_hoist, "
        "rows_pair_tail or rows_unroll2)",
    )
    kernel_assert(
        prefetch is None or not (first_static or scores_ahead),
        "rows_prefetch is a form of the plain loop, not first_static or scores_ahead",
    )
    kernel_assert(
        p2 is None or (ts and not rows_v_stationary and not pair_lag),
        "loop_p2 is a form of the transposed-scores loop, not rows_v_stationary or rows_pair_lag",
    )
    kernel_assert(not preload or (first_static and prefetch is None), "loop_preload is a form of loop_first_static")
    kernel_assert(
        not bf16_pages or (prefetch is None and hoisted is None and dr_rows is None and kv_pool.dtype != nl.bfloat16),
        "loop_cast_engines is a form of the plain loop on FP8 pages (no rows_prefetch, rows_seq_hoist or rows_dr)",
    )
    kernel_assert(
        not pair_tail
        or (
            ts
            and n_mask > 1
            and prefetch is not None
            and prefetch.groups == _TS_GROUPS
            and _loop_groups(kv_pool, local_blocks // _GROUP_BLOCKS, loop_group_cap) == 2 * _TS_GROUPS
            and not (first_static or scores_ahead or pair_lag)
        ),
        "rows_pair_tail needs the transposed scores with fused masks, the head-page prefetch of one pair, iterations of "
        "2 group pairs, and no first_static, scores_ahead or rows_pair_lag",
    )
    kernel_assert(
        unroll == 1 or (unroll == 2 and pair_tail and local_blocks % (4 * _TS_GROUPS * _GROUP_BLOCKS) == 0),
        f"rows_unroll2 runs 2 iterations a loop trip with rows_pair_tail on whole loop trips, got unroll {unroll}",
    )
    fp32 = nl.float32
    int32 = nl.int32
    hwdge = nisa.dge_mode.hwdge
    sync = nisa.engine.sync
    num_pages = kv_pool.shape[0]
    group_count = local_blocks // _GROUP_BLOCKS
    group_keys = _GROUP_BLOCKS * _BLOCK
    loop_groups = _loop_groups(kv_pool, group_count, loop_group_cap)
    loop_blocks = loop_groups * _GROUP_BLOCKS
    # A BF16 tile of FP8 pages keeps the FP8 clamp bound (``bf16_pages``).
    page_bound = fp8_page_max() if bf16_pages else None
    # With the full-iteration tail every iteration of the main loop holds only prior keys of the sequence (its last
    # iteration, with every masked row, runs in a tail), so the main loop reads its pages without the clamp.
    main_clamp = clamp_pages and not tail_groups & _FULL_TAIL
    use_lower = lower_threshold is not None
    cols = 3 if use_lower else 2
    fused = n_mask > 1
    bias_end = _BIAS_ROW + n_mask

    head_groups = 0 if prefetch is None else prefetch.groups
    # ``preload``: the next iteration's groups whose buffers iteration 0 frees before its last group pair.
    preload_groups = max(loop_groups - _TS_GROUPS, 0)
    head_rows = [] if prefetch is None else prefetch.head_tiles(b)
    cqs = [cq for cq, _r, _j in chunk_rows]
    key_path = key_path or _KeyPath()
    if hoisted is not None:
        buffers = [(head_rows[g] if g < head_groups else rows_g, *rest) for g, (rows_g, *rest) in enumerate(hoisted)]
    else:
        buffers = [
            (
                head_rows[g]
                if g < head_groups
                else gather.group0[b]
                if gather is not None and gather.carry and g == 0
                else sb.tile((_P_MAX, _GROUP_BLOCKS, _CACHE_WIDTH), nl.bfloat16 if bf16_pages else kv_pool.dtype),
                key_path.key_tile(group_keys, sb),
                sb.tile((_P_MAX, group_keys), nl.bfloat16),
                sb.tile((_P_MAX, 1), fp32),
                sb.tile((_P_MAX, 1), fp32),
            )
            for g in range(loop_groups)
        ]
    entries = None if gather is not None else sb.tile((1, loop_blocks), int32)
    entries2 = sb.tile((1, 2 * loop_blocks), int32) if unroll == 2 else None
    reference = _reference_columns(softmax, cqs, heads, sb) if ts and ahead is None else None
    row_sums = _new_row_sums(len(cqs), heads, sb) if ts else None
    state_t = _new_state_t(len(cqs), heads, sb) if rows_v_stationary else None
    q_tile = key_path.query_tile(dr_rows, cqs, heads, sb, mx_rows) if ahead is None else ahead[0]
    if ts and folded_reference:
        kernel_assert(bias_end <= _REF_ROW, f"folded_reference takes at most {_REF_ROW - _BIAS_ROW} mask rows")
        if ahead is None:
            _fold_reference(q_rope_all, reference, cqs, heads, softmax.scale, sb)
        if hoisted is None:
            _set_fold_key_rows(buffers)
    lag_weights = None
    if pair_lag:
        lag_weights = [sb.tile((_P_MAX, _TS_GROUPS * _GROUP_BLOCKS, len(cqs) * heads), nl.bfloat16) for _ in range(2)]
    pair_weights = ones = None
    if ts and scores_ahead:
        pair_weights = [sb.tile((_P_MAX, _TS_GROUPS * _GROUP_BLOCKS, len(cqs) * heads), nl.bfloat16) for _ in range(2)]
        ones = sb.tile((_P_MAX, 1), nl.bfloat16)
        nisa.memset(dst=ones, value=1.0)

    def pair_sums(pair, weights):
        _attend_group_rows(
            cqs,
            q_nope_all,
            q_rope_all,
            pair,
            softmax,
            state,
            reference,
            row_sums,
            sb,
            n_mask,
            weights=weights,
            ones=ones,
            phase="sums",
            state_t=state_t,
            p2=p2,
        )

    if fused:
        row_start = sb.tile((_P_MAX, 1), fp32)
        nisa.tensor_copy(dst=row_start[_BIAS_ROW:bias_end, 0:1], src=mask_threshold[_BIAS_ROW:bias_end, b : b + 1])
        if use_lower:
            low_start = sb.tile((_P_MAX, 1), fp32)
            nisa.tensor_copy(dst=low_start[_BIAS_ROW:bias_end, 0:1], src=lower_threshold[_BIAS_ROW:bias_end, b : b + 1])
    else:
        threshold_i = sb.tile((_P_MAX, loop_groups), int32)
        lower_i = sb.tile((_P_MAX, loop_groups), int32)
        local_start = nisa.scalar_load(loop_row[0:1, cols * b + 1 : cols * b + 2])
        key_start_local = nisa.scalar_load(loop_row[0:1, cols * b + 2 : cols * b + 3]) if use_lower else 0
    table_row = block_table[b : b + 1, 0:local_blocks].reshape((local_blocks // loop_blocks, loop_blocks))
    table_row2 = None
    if unroll == 2:
        table_row2 = block_table[b : b + 1, 0:local_blocks].reshape(
            (local_blocks // (2 * loop_blocks), 2 * loop_blocks)
        )

    def slot_offsets():
        nisa.iota(
            dst=slot_offset[_BIAS_ROW:bias_end, 0:group_keys],
            pattern=[[lay.block_stride, _GROUP_BLOCKS], [1, _BLOCK]],
            offset=0,
        )

    def body(step):
        slot_offsets()
        if gather is not None:
            # The next iteration's table row on every partition, for its gather indices.
            gather.load_next_row(b, step)
            attend_iteration(step, None, 0, clamp=main_clamp)
            return
        table_src = table_row.ap(
            pattern=[[loop_blocks, 1], [1, loop_blocks]], offset=0, scalar_offset=step, indirect_dim=0
        )
        nisa.dma_copy(dst=entries, src=table_src, dge_mode=hwdge, engine=sync, name=f"{tag}_mla_table_load_{b}")
        nisa.tensor_scalar(
            dst=entries, data=entries, op0=nl.minimum, operand0=float(num_pages - 1), op1=nl.maximum, operand1=0.0
        )
        attend_iteration(step, entries, 0, clamp=main_clamp)

    def load_entries2(trip, name):
        """``unroll`` 2: load and clamp the pages of the two iterations of loop trip ``trip``."""
        table_src = table_row2.ap(
            pattern=[[2 * loop_blocks, 1], [1, 2 * loop_blocks]], offset=0, scalar_offset=trip, indirect_dim=0
        )
        nisa.dma_copy(dst=entries2, src=table_src, dge_mode=hwdge, engine=sync, name=f"{tag}_mla_{name}_{b}")
        nisa.tensor_scalar(
            dst=entries2,
            data=entries2,
            op0=nl.minimum,
            operand0=float(num_pages - 1),
            op1=nl.maximum,
            operand1=0.0,
            engine=nisa.engine.vector,
        )

    def body2(trip):
        """``unroll`` 2: iterations 2 trip and 2 trip + 1; each half loads the head pair of the iteration after it."""
        load_entries2(trip, "table_load2")
        for half in range(2):
            slot_offsets()
            attend_iteration(trip, entries2, half * loop_blocks, prefetch_at=(half + 1, 2), name_suffix=f"_u{half}")

    def leftover(_step):
        """``unroll`` 2: iteration 2 F after the F loop trips, when the full iterations are odd."""
        trips = nisa.scalar_load(loop_row[0:1, cols * b : cols * b + 1])
        load_entries2(trips, "table_load_lo")
        slot_offsets()
        attend_iteration(trips, entries2, 0, prefetch_at=(1, 2), name_suffix="_lo")

    def attend_iteration(
        step,
        entries,
        base,
        after=None,
        loaded=0,
        preload_base=None,
        groups=None,
        prefetch_next=True,
        prefetch_at=(1, 1),
        name_suffix=None,
        next_row=True,
        clamp=None,
    ):
        """The page loads and the attention of one iteration; ``entries[0, base + i]`` is the page of its block i.

        ``after``: the page loads after the first group pair follow this instruction in the schedule. ``loaded``: the
        pages of the first ``loaded`` groups are in their buffers already (``preload``). ``preload_base``: after the
        attention, load the pages of the next iteration's first ``preload_groups`` groups, block ``i`` from
        ``entries[0, preload_base + i]``. ``groups``: the first groups of the iteration only (``pair_tail``: the head
        pair); ``prefetch_next`` False: no head-pair load of the next iteration. ``prefetch_at``: the (row, pitch) of
        the next head pair (``_RowsPrefetch.load``); ``name_suffix``: the suffix of the instruction names (default: by
        ``base``). ``next_row`` False (``tail_groups``): with ``gather``, no next indices, carried group 0 or index copy.
        ``clamp``: whether the pages take the clamp (None: ``clamp_pages``; False: the main loop of a full-iteration
        tail, whose iterations hold only prior keys).
        """
        group_buffers = buffers if groups is None else buffers[:groups]
        # The static first iteration and the loop body are both traced: their page loads take other names.
        suffix = name_suffix if name_suffix is not None else (f"_at{base}" if base else "")
        pre_suffix = "" if name_suffix is None else name_suffix
        after_instr = after if after is None or hasattr(after, "depends_on") else nisa.NkiInstruction(after)
        # Bitwise orders the current-block read after the owner-write through the kv_pool tensor order (both
        # access kv_pool), not an explicit edge, which is fragile in a larger graph.
        # Prefetch pass: issue every group's page loads first (DMA only), so a group's attention overlaps
        # the loads of the later groups instead of serializing behind them. The buffers are pre-allocated
        # per group, so all loads of one iteration are in flight at once.
        for g, (key_rows, _latent_keys, _rope_keys, _threshold, _lower) in enumerate(group_buffers):
            for lb in range(_GROUP_BLOCKS if g >= max(head_groups, loaded) and gather is None else 0):
                entry = base + g * _GROUP_BLOCKS + lb
                load = nisa.dma_copy(
                    dst=key_rows[0:_P_MAX, lb, 0:_CACHE_WIDTH],
                    src=kv_pool.ap(
                        pattern=[[_CACHE_WIDTH, _BLOCK], [1, _CACHE_WIDTH]],
                        offset=0,
                        scalar_offset=entries[0:1, entry : entry + 1],
                        indirect_dim=0,
                    ),
                    dge_mode=hwdge,
                    engine=sync,
                    name=f"{tag}_mla_group_page_{b}_{g}_{lb}{suffix}",
                )
                if after is not None and g >= _TS_GROUPS:
                    load = load if hasattr(load, "depends_on") else nisa.NkiInstruction(load)
                    load.depends_on(after_instr, kind=nisa.dependency.schedule)
        # Compute pass: sanitize (unless the pages are zeroed), transpose, and attend each group; its load is
        # already in flight.
        pending = []
        # ``row_pipeline``: two weight tile sets in turn, and the (group, row) item whose values are still to come.
        pipe_tiles = (
            [
                (
                    sb.tile((heads, group_keys), nl.bfloat16),
                    sb.tile((_P_MAX, _GROUP_BLOCKS, heads), nl.bfloat16),
                    sb.tile((heads, 1), fp32),
                )
                for _ in range(2)
            ]
            if row_pipeline
            else None
        )
        row_pending = []

        def finish_row(item, after=None):
            # ``row_pipeline``: the transposes, value matmuls and state add of one (group, row) item; on TensorE after
            # ``after``, the last score matmul of the next item.
            cq_p, key_rows_p, (weights_p, weights_t_p, sum_p) = item
            numerator_p = _attend_group_values(
                key_rows_p, weights_p, weights_t_p, heads, weights_on_act=weights_on_act, after=after
            )
            softmax.update(state, cq_p, numerator_p, sum_p)

        # With scores_ahead: the pair whose sums wait for the weights of the next pair, and its weight tile.
        behind = None
        # With pair_lag: the value matmuls of the pair before (after this pair's transposes), and whether that pair
        # holds the head groups (the next iteration's head pages load after it).
        lagged = None
        pairs = 0

        def run_lagged(lagged):
            numerators, head_pair = lagged
            numerators()
            if head_pair and prefetch is not None:
                prefetch.load(b, step, *prefetch_at, suffix=pre_suffix)

        for g, (key_rows, latent_keys, rope_keys, threshold, lower) in enumerate(group_buffers):
            if gather is not None:
                # Group g + _GATHER_LEAD's gather before the clamps of group g; the first groups at the iteration start
                # (group 0 carried in with "carry").
                first = (1 if gather.carry else 0) if g == 0 else g + _GATHER_LEAD
                for gathered in range(first, min(g + _GATHER_LEAD, len(group_buffers) - 1) + 1):
                    gather.gather(
                        buffers[gathered][0],
                        gather.index_cur[b],
                        gathered,
                        f"{tag}_mla_gather_{b}_{gathered}{pre_suffix}",
                    )
            if clamp_pages if clamp is None else clamp:
                for lb in range(_GROUP_BLOCKS):
                    _sanitize_cache_rows(key_rows[0:_P_MAX, lb, 0:_CACHE_WIDTH], bound=page_bound)
            lower_arg = None
            if fused:
                nisa.tensor_scalar(
                    dst=threshold[_BIAS_ROW:bias_end, 0:1],
                    data=row_start[_BIAS_ROW:bias_end, 0:1],
                    op0=nl.subtract,
                    operand0=float(g * lay.group_stride),
                )
                if use_lower:
                    nisa.tensor_scalar(
                        dst=lower[_BIAS_ROW:bias_end, 0:1],
                        data=low_start[_BIAS_ROW:bias_end, 0:1],
                        op0=nl.subtract,
                        operand0=float(g * lay.group_stride),
                    )
                    lower_arg = lower[_BIAS_ROW:bias_end, 0:1]
            else:
                nisa.scalar_store(
                    threshold_i[_BIAS_ROW : _BIAS_ROW + 1, g : g + 1],
                    local_start - step * (loop_groups * lay.group_stride) - g * lay.group_stride,
                )
                nisa.tensor_copy(
                    dst=threshold[_BIAS_ROW : _BIAS_ROW + 1, 0:1], src=threshold_i[_BIAS_ROW : _BIAS_ROW + 1, g : g + 1]
                )
                if use_lower:
                    nisa.scalar_store(
                        lower_i[_BIAS_ROW : _BIAS_ROW + 1, g : g + 1],
                        key_start_local - step * (loop_groups * lay.group_stride) - g * lay.group_stride,
                    )
                    nisa.tensor_copy(
                        dst=lower[_BIAS_ROW : _BIAS_ROW + 1, 0:1], src=lower_i[_BIAS_ROW : _BIAS_ROW + 1, g : g + 1]
                    )
                    lower_arg = lower[_BIAS_ROW : _BIAS_ROW + 1, 0:1]
            _transpose_group(
                key_rows,
                latent_keys,
                rope_keys,
                slot_offset,
                threshold[_BIAS_ROW:bias_end, 0:1],
                lower_arg,
                sb,
                n_mask,
                key_path=key_path,
            )
            if ts:
                pending.append((latent_keys, rope_keys, key_rows))
                if (len(pending) == _TS_GROUPS or g == len(group_buffers) - 1) and pair_weights is None:
                    if lagged is not None:
                        run_lagged(lagged)
                    numerators = _attend_group_rows(
                        cqs,
                        q_nope_all,
                        q_rope_all,
                        pending,
                        softmax,
                        state,
                        reference,
                        row_sums,
                        sb,
                        n_mask,
                        weights=None if lag_weights is None else lag_weights[pairs % 2],
                        folded=folded_reference,
                        state_t=state_t,
                        key_path=key_path,
                        q_tile=q_tile,
                        sums_lag=_SUMS_LAG if pair_lag else 0,
                        defer_numerators=pair_lag,
                        p2=p2,
                    )
                    pending.clear()
                    pairs += 1
                    if pair_lag:
                        lagged = (numerators, g == head_groups - 1)
                    elif prefetch is not None and prefetch_next and g == head_groups - 1:
                        # The first call was the last reader of the head pages: load those of the next iteration.
                        prefetch.load(b, step, *prefetch_at, suffix=pre_suffix)
                elif len(pending) == _TS_GROUPS or g == len(group_buffers) - 1:
                    weights = pair_weights[g // _TS_GROUPS % 2]
                    _attend_group_rows(
                        cqs,
                        q_nope_all,
                        q_rope_all,
                        list(pending),
                        softmax,
                        state,
                        reference,
                        row_sums,
                        sb,
                        n_mask,
                        weights=weights,
                        phase="scores",
                        folded=folded_reference,
                        key_path=key_path,
                        q_tile=q_tile,
                    )
                    if behind is not None:
                        pair_sums(*behind)
                    behind = (list(pending), weights)
                    pending.clear()
                continue
            for cq, _r, j in chunk_rows:
                if bitwise and not fused:
                    # Per-row threshold in the local frame: add the row index j so row j masks at
                    # position[b] + j. Rewrite the group mask for this row.
                    nisa.scalar_store(
                        threshold_i[_BIAS_ROW : _BIAS_ROW + 1, g : g + 1],
                        local_start - step * (loop_groups * lay.group_stride) - g * lay.group_stride + j,
                    )
                    nisa.tensor_copy(
                        dst=threshold[_BIAS_ROW : _BIAS_ROW + 1, 0:1],
                        src=threshold_i[_BIAS_ROW : _BIAS_ROW + 1, g : g + 1],
                    )
                    nisa.tensor_scalar(
                        dst=rope_keys[_BIAS_ROW : _BIAS_ROW + 1, 0:group_keys],
                        data=slot_offset[_BIAS_ROW : _BIAS_ROW + 1, 0:group_keys],
                        op0=nl.greater_equal,
                        operand0=threshold[_BIAS_ROW : _BIAS_ROW + 1, 0:1],
                        op1=nl.multiply,
                        operand1=_MASK_BIAS,
                    )
                if row_pipeline:
                    # The scores of this item, then the values of the item before it.
                    tiles = pipe_tiles[1] if row_pending and row_pending[-1][2] is pipe_tiles[0] else pipe_tiles[0]
                    scores_end = _attend_group_scores(
                        cq, q_nope_all, q_rope_all, latent_keys, rope_keys, softmax, tiles[2], tiles[0], heads, n_mask
                    )
                    if row_pending:
                        finish_row(row_pending.pop(), after=scores_end)
                    row_pending.append((cq, key_rows, tiles))
                    continue
                local_sum = sb.tile((heads, 1), fp32)
                numerator_psum = _attend_group(
                    cq,
                    q_nope_all,
                    q_rope_all,
                    latent_keys,
                    rope_keys,
                    key_rows,
                    softmax,
                    local_sum,
                    heads,
                    sb,
                    n_mask,
                    weights_on_act=weights_on_act,
                )
                softmax.update(state, cq, numerator_psum, local_sum)
            if gather is not None and next_row and g == min(_GATHER_CARRY_GROUP, len(group_buffers) - 1):
                gather.build(gather.index_next, gather.row_table)
                if gather.carry:
                    # Group 0's page tile has no reader left in this iteration.
                    gather.gather(buffers[0][0], gather.index_next, 0, f"{tag}_mla_gather_next_{b}")
        if row_pending:
            # ``row_pipeline``: the values of the last item of the iteration (the loop is a scheduling wall).
            finish_row(row_pending.pop())
        if gather is not None and next_row:
            # After every gather of this iteration has read its indices (the same GpSimd queue).
            nisa.tensor_copy(dst=gather.index_cur[b], src=gather.index_next, engine=nisa.engine.gpsimd)
        if behind is not None:
            # The loop is a scheduling wall: the last pair of the iteration adds its sums here.
            pair_sums(*behind)
        if lagged is not None:
            # The value matmuls of the last pair of the iteration.
            run_lagged(lagged)
        for g in range(preload_groups if preload_base is not None else 0):
            # ``preload``: the next iteration's pages of group g, into its buffer after this iteration's last read.
            for lb in range(_GROUP_BLOCKS):
                entry = preload_base + g * _GROUP_BLOCKS + lb
                nisa.dma_copy(
                    dst=buffers[g][0][0:_P_MAX, lb, 0:_CACHE_WIDTH],
                    src=kv_pool.ap(
                        pattern=[[_CACHE_WIDTH, _BLOCK], [1, _CACHE_WIDTH]],
                        offset=0,
                        scalar_offset=entries[0:1, entry : entry + 1],
                        indirect_dim=0,
                    ),
                    dge_mode=hwdge,
                    engine=sync,
                    name=f"{tag}_mla_group_page_{b}_{g}_{lb}_pre",
                )
        if fused:
            # The next iteration starts loop_groups key groups lower in the local frame.
            span = float(loop_groups * lay.group_stride)
            nisa.tensor_scalar(
                dst=row_start[_BIAS_ROW:bias_end, 0:1],
                data=row_start[_BIAS_ROW:bias_end, 0:1],
                op0=nl.subtract,
                operand0=span,
            )
            if use_lower:
                nisa.tensor_scalar(
                    dst=low_start[_BIAS_ROW:bias_end, 0:1],
                    data=low_start[_BIAS_ROW:bias_end, 0:1],
                    op0=nl.subtract,
                    operand0=span,
                )

    if first_static and local_blocks <= 2 * loop_blocks:
        # The entries of both iterations, loaded and clamped once.
        entries_all = sb.tile((1, local_blocks), int32)
        nisa.dma_copy(
            dst=entries_all,
            src=block_table[b : b + 1, 0:local_blocks],
            dge_mode=hwdge,
            engine=sync,
            name=f"{tag}_mla_table_load_all_{b}",
        )
        nisa.tensor_scalar(
            dst=entries_all,
            data=entries_all,
            op0=nl.minimum,
            operand0=float(num_pages - 1),
            op1=nl.maximum,
            operand1=0.0,
        )
        slot_offsets()
        two = local_blocks > loop_blocks
        attend_iteration(0, entries_all, 0, first_static_after, preload_base=loop_blocks if preload and two else None)
        if local_blocks > loop_blocks:
            # The other iteration runs when the sequence has more than one iteration of key groups.
            rest = sb.tile((1, 1), int32)
            nisa.tensor_scalar(
                dst=rest,
                data=loop_row[0:1, cols * b : cols * b + 1],
                op0=nl.subtract,
                operand0=1.0,
                op1=nl.maximum,
                operand1=0.0,
            )

            def second(_step):
                slot_offsets()
                attend_iteration(1, entries_all, loop_blocks, loaded=preload_groups if preload else 0)

            nl.fori_loop(0, nisa.scalar_load(rest[0:1, 0:1]), second)
    else:
        if prefetch is not None and next_b is not None:
            prefetch.load(next_b)
        iterations = nisa.scalar_load(loop_row[0:1, cols * b : cols * b + 1])
        if unroll == 2:
            nl.fori_loop(0, iterations, body2)
            left_col = cols * tokens + 2 * b + 1
            nl.fori_loop(0, nisa.scalar_load(loop_row[0:1, left_col : left_col + 1]), leftover)
        else:
            nl.fori_loop(0, iterations, body)
        if pair_tail:
            tail_col = cols * tokens + unroll * b
            tail_trips = nisa.scalar_load(loop_row[0:1, tail_col : tail_col + 1])

            def tail_body(_step):
                slot_offsets()
                attend_iteration(0, entries, 0, groups=_TS_GROUPS, prefetch_next=False)

            nl.fori_loop(0, tail_trips, tail_body)
        tails = loop_tail_sizes(tail_groups, loop_groups)
        for k, tail in enumerate(tails):
            tail_col = cols * tokens + len(tails) * b + k
            tail_trips = nisa.scalar_load(loop_row[0:1, tail_col : tail_col + 1])
            # One tail keeps the names of the single tail form; a cascade names each tail by its groups.
            tail_suffix = "_tail" if len(tails) == 1 else f"_tail{tail}"

            def tail_iteration(_step, tail=tail, tail_suffix=tail_suffix):
                """The first ``tail`` groups of the iteration after the full ones."""
                full = nisa.scalar_load(loop_row[0:1, cols * b : cols * b + 1])
                slot_offsets()
                if gather is not None:
                    attend_iteration(full, None, 0, groups=tail, name_suffix=tail_suffix, next_row=False)
                    return
                table_src = table_row.ap(
                    pattern=[[loop_blocks, 1], [1, loop_blocks]], offset=0, scalar_offset=full, indirect_dim=0
                )
                nisa.dma_copy(
                    dst=entries,
                    src=table_src,
                    dge_mode=hwdge,
                    engine=sync,
                    name=f"{tag}_mla_table_load{tail_suffix}_{b}",
                )
                nisa.tensor_scalar(
                    dst=entries,
                    data=entries,
                    op0=nl.minimum,
                    operand0=float(num_pages - 1),
                    op1=nl.maximum,
                    operand1=0.0,
                    engine=nisa.engine.vector,
                )
                attend_iteration(full, entries, 0, groups=tail, name_suffix=tail_suffix)

            nl.fori_loop(0, tail_trips, tail_iteration)
    if ts:
        if pads is not None:
            # A padding sequence has no keys: its sums take 1 on each rank (a finite 0 output); a select, so the sums
            # of a live sequence keep every bit.
            width = row_sums.shape[1]
            one = sb.tile((1, width), nl.float32)
            nisa.memset(dst=one, value=1.0)
            pad_f = sb.tile((1, width), nl.float32)
            nisa.tensor_scalar(dst=pad_f, data=one, op0=nl.multiply, operand0=pads["pad"][0:1, 2 * b : 2 * b + 1])
            pad_u8 = sb.tile((1, width), nl.uint8)
            nisa.tensor_copy(dst=pad_u8, src=pad_f)
            nisa.tensor_copy_predicated(dst=row_sums, src=one, predicate=pad_u8)
        _add_row_sums(state, row_sums, cqs, heads, p2=p2)
    if state_t is not None:
        _add_state_t(state, state_t, cqs, heads)


def _reference_columns(softmax, cqs, heads, sb):
    """Return the FP32 ``[128, R * heads]`` tile of the negative reference of the rows ``cqs``, on every partition.

    A matmul of the ``heads x heads`` diagonal of each row's reference with an all-ones stationary moves the values
    from the head partitions to the free axis. The diagonal comes from the kernel's FP32 identity, so no layer runs a
    Pool iota that the scheduler can put ahead of the weight gathers of the layer before.
    """
    fp32 = nl.float32
    eye = nl.shared_identity_matrix(_P_MAX, dtype=fp32)[0:heads, 0:heads]
    ones = sb.tile((heads, _P_MAX), fp32)
    nisa.memset(dst=ones, value=1.0)
    reference = sb.tile((_P_MAX, len(cqs) * heads), fp32)
    for i, cq in enumerate(cqs):
        diag = sb.tile((heads, heads), fp32)
        nisa.tensor_scalar(
            dst=diag, data=eye, op0=nl.multiply, operand0=softmax.negative_reference[0:heads, cq : cq + 1]
        )
        reference_psum = nl.ndarray((_P_MAX, heads), dtype=fp32, buffer=nl.psum)
        nisa.nc_matmul(dst=reference_psum, stationary=ones, moving=diag, is_stationary_onezero=True)
        nisa.tensor_copy(dst=reference[0:_P_MAX, i * heads : (i + 1) * heads], src=reference_psum)
    return reference


def _row_view(tile, cq0: int, count: int, stride: int, inner: int, inner_offset: int, partitions: int):
    """Return the ``[partitions, count * heads]`` view of the chunk rows ``cq0, cq0 + stride, ...`` of a query tile.

    ``tile`` is a whole ``[128, cr, ..., heads]`` tile; ``inner`` is the element count of one chunk row and
    ``inner_offset`` the offset of the head columns in it.
    """
    heads = tile.shape[-1]
    free = int(math.prod(tile.shape[1:]))
    return tile.ap(pattern=[[free, partitions], [stride * inner, count], [1, heads]], offset=cq0 * inner + inner_offset)


def _attend_group_rows(
    cqs,
    q_nope_all,
    q_rope_all,
    groups,
    softmax,
    state,
    reference,
    row_sums,
    sb,
    mask_rows=1,
    weights=None,
    ones=None,
    phase=None,
    folded=False,
    state_t=None,
    key_path=None,
    q_tile=None,
    sums_lag=0,
    defer_numerators=False,
    p2=None,
):
    """Attend the chunk rows ``cqs`` of one sequence over prior-key groups, with the keys as the stationary operand.

    ``groups`` holds the ``(latent_keys, rope_keys, key_rows)`` of each group; the numerator of a row accumulates
    over all their blocks, so the state takes one add for each row and call. The scores of all rows of a key block
    take one pass, and the weights stay key-major (``[128 keys, R * heads]``), so the numerator of each row needs no
    transpose. The weight is ``min(exp(scale * score + reference), e**cap)``, the capped softmax of ``_Softmax``. A
    ones-column matmul adds the weight sums of all rows to the FP32 ``[1, R * heads]`` row ``row_sums``;
    ``_add_row_sums`` moves them into the state after the last group.

    ``phase="scores"`` only writes the capped weights of the blocks into ``weights`` (a ``[128, blocks, R * heads]``
    tile of the caller); ``phase="sums"`` then adds their weight sums and numerators, with the caller's ones column
    ``ones``. The call of the next groups can run between the two phases, and each accumulation keeps its order. None
    runs both phases in one pass. ``folded``: the rope matmul adds the reference (``_fold_reference``).

    With ``state_t`` (``_new_state_t``; ``rows_v_stationary``) the values are the stationary operand of the numerator
    matmuls, with the weights of all rows moving, and the transposed numerators add up in ``state_t``; ``_add_state_t``
    moves them into the state after the last group. ``key_path`` (``_KeyPath``, None = BF16) holds the latent key form of
    the groups and issues the latent part of the scores of each block, with the query tile ``q_tile`` of the rows
    (``_KeyPath.query_tile``).

    ``sums_lag``: the sums matmul of block ``i`` runs after the score matmuls of block ``i + sums_lag`` (the sums keep
    their order). ``defer_numerators`` (with ``state_t`` and the caller's ``weights``): return the value matmuls and
    state adds as a function for the caller to run later, in place of running them (``rows_pair_lag``).

    ``p2`` (``loop_p2``): ``(state, seq, seqs)``, the (row, head) state ``[128, G seqs + 1, 513]`` of ``p2`` and the
    index of the sequence in the pass. The numerators then take one 128-column stationary of the weights for each group
    ``g`` of 128 (row, head) columns (``pack_p2_runs``), in place of one 96-column stationary for each row, and add into
    state column ``g seqs + seq``. Each numerator column takes the same products in the same order, so the bits do not
    change. Not with ``state_t``.
    """
    kernel_assert(
        not defer_numerators or (state_t is not None and weights is not None and phase is None),
        "defer_numerators needs state_t, the caller's weights and one pass",
    )
    kernel_assert(p2 is None or state_t is None, "loop_p2 and rows_v_stationary are two forms of the numerators")
    groups = list(groups)
    sb.open_scope("group_rows")
    bf16 = nl.bfloat16
    fp32 = nl.float32
    heads = q_rope_all.shape[-1]
    blocks = [(g, block) for g, (_lk, _rk, key_rows) in enumerate(groups) for block in range(key_rows.shape[1])]
    count = len(cqs)
    width = count * heads
    bank_rows = _PSUM_BANK_FP32 // heads
    pieces = [(r0, min(count, r0 + bank_rows)) for r0 in range(0, count, bank_rows)]
    rope_rows = _BIAS_ROW + mask_rows
    if ones is None and phase != "scores":
        ones = sb.tile((_P_MAX, 1), bf16)
        nisa.memset(dst=ones, value=1.0)
    if weights is None:
        weights = sb.tile((_P_MAX, len(blocks), width), bf16)
    if phase != "scores":
        sums_psum = [nl.ndarray((1, (r1 - r0) * heads), dtype=fp32, buffer=nl.psum) for r0, r1 in pieces]

    def block_sums(index):
        for (r0, r1), psum in zip(pieces, sums_psum):
            nisa.nc_matmul(
                dst=psum,
                stationary=ones,
                moving=weights[0:_P_MAX, index, r0 * heads : r1 * heads],
                accumulate=index > 0,
            )

    waiting = []
    for index, (g, block) in enumerate(blocks):
        if phase != "sums":
            _block_row_weights(
                cqs,
                q_nope_all,
                q_rope_all,
                groups[g],
                block,
                softmax,
                reference,
                weights,
                index,
                sb,
                rope_rows,
                folded,
                key_path,
                q_tile,
            )
        if phase == "scores":
            continue
        waiting.append(index)
        if len(waiting) > sums_lag:
            block_sums(waiting.pop(0))
    for index in waiting:
        block_sums(index)
    if phase == "scores":
        sb.close_scope()
        return
    for (r0, r1), psum in zip(pieces, sums_psum):
        cols = slice(r0 * heads, r1 * heads)
        nisa.tensor_tensor(dst=row_sums[0:1, cols], data1=row_sums[0:1, cols], data2=psum, op=nl.add)
    if p2 is not None:
        p2_state, seq, seqs = p2
        for group in range(width // _P_MAX):
            numerator_psum = nl.ndarray((_P_MAX, _KV_LORA), dtype=fp32, buffer=nl.psum)
            for index, (g, block) in enumerate(blocks):
                nisa.nc_matmul(
                    dst=numerator_psum,
                    stationary=weights[0:_P_MAX, index, group * _P_MAX : (group + 1) * _P_MAX],
                    moving=groups[g][2][0:_P_MAX, block, 0:_KV_LORA],
                    accumulate=index > 0,
                )
            column = group * seqs + seq
            nisa.tensor_tensor(
                dst=p2_state[0:_P_MAX, column, 0:_KV_LORA],
                data1=p2_state[0:_P_MAX, column, 0:_KV_LORA],
                data2=numerator_psum,
                op=nl.add,
                engine=nisa.engine.vector,
            )
        sb.close_scope()
        return None
    if state_t is not None:
        if defer_numerators:
            sb.close_scope()
            return lambda: _values_stationary_numerators(groups, blocks, weights, pieces, heads, state_t)
        _values_stationary_numerators(groups, blocks, weights, pieces, heads, state_t)
        sb.close_scope()
        return None
    for i, cq in enumerate(cqs):
        numerator_psum = nl.ndarray((heads, _KV_LORA), dtype=fp32, buffer=nl.psum)
        for index, (g, block) in enumerate(blocks):
            nisa.nc_matmul(
                dst=numerator_psum,
                stationary=weights[0:_P_MAX, index, i * heads : (i + 1) * heads],
                moving=groups[g][2][0:_P_MAX, block, 0:_KV_LORA],
                accumulate=index > 0,
            )
        nisa.tensor_tensor(
            dst=state[0:heads, cq, 0:_KV_LORA], data1=state[0:heads, cq, 0:_KV_LORA], data2=numerator_psum, op=nl.add
        )
    sb.close_scope()


def _block_row_weights(
    cqs,
    q_nope_all,
    q_rope_all,
    group,
    block,
    softmax,
    reference,
    weights,
    index,
    sb,
    rope_rows,
    folded,
    key_path=None,
    q_tile=None,
):
    """Write the capped weights of key block ``block`` of ``group`` for the rows ``cqs`` into ``weights[:, index]``.

    ``folded``: the rope matmul also adds the reference (``_fold_reference``), so one exponential of the scaled scores
    writes the BF16 weights and the cap runs on them in place. ``key_path`` (``_KeyPath``, None = BF16) issues the
    latent part of the scores, with the query tile ``q_tile`` of the rows (``_KeyPath.query_tile``).
    """
    fp32 = nl.float32
    latent_keys, rope_keys, _key_rows = group
    heads = q_rope_all.shape[-1]
    count = len(cqs)
    stride = cqs[1] - cqs[0] if count > 1 else 1
    width = count * heads
    bank_rows = _PSUM_BANK_FP32 // heads
    pieces = [(r0, min(count, r0 + bank_rows)) for r0 in range(0, count, bank_rows)]
    keys = slice(block * _BLOCK, (block + 1) * _BLOCK)
    scores_psum = [nl.ndarray((_P_MAX, (r1 - r0) * heads), dtype=fp32, buffer=nl.psum) for r0, r1 in pieces]
    (key_path or _KeyPath()).scores(scores_psum, pieces, latent_keys, keys, q_nope_all, q_tile, cqs, stride, heads)
    if folded:
        contraction = _REF_ROW + _REF_PARTS
        for (r0, r1), psum in zip(pieces, scores_psum):
            nisa.nc_matmul(
                dst=psum,
                stationary=rope_keys[0:contraction, keys],
                moving=_row_view(q_rope_all, cqs[r0], r1 - r0, stride, heads, 0, contraction),
                accumulate=True,
            )
            nisa.activation(
                dst=weights[0:_P_MAX, index, r0 * heads : r1 * heads], op=nl.exp, data=psum, scale=softmax.scale
            )
        nisa.tensor_scalar(
            dst=weights[0:_P_MAX, index, 0:width],
            data=weights[0:_P_MAX, index, 0:width],
            op0=nl.minimum,
            operand0=_CAP_WEIGHT,
        )
        return
    exponent = sb.tile((_P_MAX, width), fp32)
    for (r0, r1), psum in zip(pieces, scores_psum):
        nisa.nc_matmul(
            dst=psum,
            stationary=rope_keys[0:rope_rows, keys],
            moving=_row_view(q_rope_all, cqs[r0], r1 - r0, stride, heads, 0, rope_rows),
            accumulate=True,
        )
        nisa.scalar_tensor_tensor(
            dst=exponent[0:_P_MAX, r0 * heads : r1 * heads],
            data=psum,
            op0=nl.multiply,
            operand0=softmax.scale,
            op1=nl.add,
            operand1=reference[0:_P_MAX, r0 * heads : r1 * heads],
        )
    nisa.activation(dst=exponent, op=nl.exp, data=exponent)
    nisa.tensor_scalar(dst=weights[0:_P_MAX, index, 0:width], data=exponent, op0=nl.minimum, operand0=_CAP_WEIGHT)


def _fold_reference(q_rope_all, reference, cqs, heads, scale, sb):
    """Write ``reference / scale`` of the rows ``cqs`` into partitions ``_REF_ROW ..`` of ``q_rope_all`` as BF16 parts.

    With ``_REF_PARTS`` ones rows in the keys, the rope matmul then adds the reference to every score, so
    ``exp(scale * psum)`` is the weight of ``_attend_group_rows``. Part ``k`` (on partition ``_REF_ROW + k``) is the
    BF16 rounding of what the parts before it leave. The reference tile holds the same values on every partition, so
    each partition computes every part and keeps its own with a column of the identity.
    """
    fp32 = nl.float32
    width = len(cqs) * heads
    rows = slice(_REF_ROW, _REF_ROW + _REF_PARTS)
    eye = nl.shared_identity_matrix(_P_MAX, dtype=fp32)
    rest = sb.tile((_P_MAX, width), fp32)
    nisa.tensor_scalar(dst=rest[rows, 0:width], data=reference[rows, 0:width], op0=nl.multiply, operand0=1.0 / scale)
    part = sb.tile((_P_MAX, width), nl.bfloat16)
    kept = sb.tile((_P_MAX, width), fp32)
    for k in range(_REF_PARTS):
        nisa.tensor_copy(dst=part[rows, 0:width], src=rest[rows, 0:width])
        own = eye[rows, _REF_ROW + k : _REF_ROW + k + 1]
        if k == 0:
            nisa.tensor_scalar(dst=kept[rows, 0:width], data=part[rows, 0:width], op0=nl.multiply, operand0=own)
        else:
            nisa.scalar_tensor_tensor(
                dst=kept[rows, 0:width],
                data=part[rows, 0:width],
                op0=nl.multiply,
                operand0=own,
                op1=nl.add,
                operand1=kept[rows, 0:width],
            )
        if k < _REF_PARTS - 1:
            nisa.tensor_tensor(
                dst=rest[rows, 0:width], data1=rest[rows, 0:width], data2=part[rows, 0:width], op=nl.subtract
            )
    for i, cq in enumerate(cqs):
        nisa.tensor_copy(dst=q_rope_all[rows, cq, 0:heads], src=kept[rows, i * heads : (i + 1) * heads])


def _new_row_sums(count: int, heads: int, sb):
    """Return a zero FP32 ``[1, count * heads]`` row for the weight sums of ``_attend_group_rows``."""
    row_sums = sb.tile((1, count * heads), nl.float32)
    nisa.memset(dst=row_sums, value=0.0)
    return row_sums


def _add_row_sums(state, row_sums, cqs, heads, p2=None):
    """Add the weight sums of each row (``row_sums``, ``[1, R * heads]``) to the sum column of the state of the row.

    ``p2`` (``loop_p2``, see ``_attend_group_rows``): the sums of each group of 128 (row, head) columns go to the sum
    column of state column ``g seqs + seq`` of the (row, head) state, the same values."""
    if p2 is not None:
        p2_state, seq, seqs = p2
        for group in range(len(cqs) * heads // _P_MAX):
            sum_psum = nl.ndarray((_P_MAX, 1), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_transpose(dst=sum_psum, data=row_sums[0:1, group * _P_MAX : (group + 1) * _P_MAX])
            column = group * seqs + seq
            nisa.tensor_tensor(
                dst=p2_state[0:_P_MAX, column, _KV_LORA : _KV_LORA + 1],
                data1=p2_state[0:_P_MAX, column, _KV_LORA : _KV_LORA + 1],
                data2=sum_psum,
                op=nl.add,
                engine=nisa.engine.vector,
            )
        return
    for i, cq in enumerate(cqs):
        sum_psum = nl.ndarray((heads, 1), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=sum_psum, data=row_sums[0:1, i * heads : (i + 1) * heads])
        nisa.tensor_tensor(
            dst=state[0:heads, cq, _KV_LORA : _KV_LORA + 1],
            data1=state[0:heads, cq, _KV_LORA : _KV_LORA + 1],
            data2=sum_psum,
            op=nl.add,
        )


def _attend_group(
    cq,
    q_nope_all,
    q_rope_all,
    latent_keys,
    rope_keys,
    key_rows,
    softmax,
    group_sum,
    heads,
    sb,
    mask_rows=1,
    weights_on_act=False,
):
    """Attend query ``cq`` (chunk-local index) over one prior-key group.

    Return the FP32 PSUM numerator ``[heads, 512]`` and write the FP32 sum of the weights to
    ``group_sum`` (``[heads, 1]``). The rope matmul also adds the ``mask_rows`` mask rows below the rope keys,
    selected by the query rows below the rope queries. ``weights_on_act`` copies the transposed weights out of
    PSUM on ScalarE (as the decode does) in place of VectorE; the copy is exact.
    """
    sb.open_scope("group")
    bf16 = nl.bfloat16
    fp32 = nl.float32
    group_blocks = key_rows.shape[1]
    num_keys = group_blocks * _BLOCK
    scores_psum = nl.ndarray((heads, num_keys), dtype=fp32, buffer=nl.psum)
    for latent_tile in range(_KV_LORA_TILES):
        nisa.nc_matmul(
            dst=scores_psum,
            stationary=q_nope_all[0:_P_MAX, cq, latent_tile, 0:heads],
            moving=latent_keys[0:_P_MAX, latent_tile, 0:num_keys],
            accumulate=latent_tile > 0,
        )
    nisa.nc_matmul(
        dst=scores_psum,
        stationary=q_rope_all[0 : _BIAS_ROW + mask_rows, cq, 0:heads],
        moving=rope_keys[0 : _BIAS_ROW + mask_rows, 0:num_keys],
        accumulate=True,
    )
    if softmax.exact:
        block_max = sb.tile((heads, 1), fp32)
        nisa.tensor_reduce(dst=block_max, op=nl.maximum, data=scores_psum, axis=1)
        running_max = softmax.running_max[0:heads, cq : cq + 1]
        nisa.tensor_tensor(dst=block_max, data1=block_max, data2=running_max, op=nl.maximum)
        bias = sb.tile((heads, 1), fp32)
        nisa.tensor_scalar(dst=bias, data=block_max, op0=nl.multiply, operand0=-softmax.scale)
        nisa.activation(dst=softmax.alpha, op=nl.exp, data=running_max, scale=softmax.scale, bias=bias)
        nisa.tensor_copy(dst=running_max, src=block_max)
    else:
        nisa.tensor_scalar(
            dst=scores_psum, data=scores_psum, op0=nl.minimum, operand0=softmax.score_cap[0:heads, cq : cq + 1]
        )
        bias = softmax.negative_reference[0:heads, cq : cq + 1]
    weights = sb.tile((heads, num_keys), bf16)
    nisa.activation(
        dst=weights,
        op=nl.exp,
        data=scores_psum,
        scale=softmax.scale,
        bias=bias,
        reduce_op=nl.add,
        reduce_res=group_sum,
        reduce_cmd=nisa.reduce_cmd.reset_reduce,
    )
    weights_psum = nl.ndarray((_P_MAX, group_blocks, heads), dtype=bf16, buffer=nl.psum)
    for local_block in range(group_blocks):
        nisa.nc_transpose(
            dst=weights_psum[0:_P_MAX, local_block, 0:heads],
            data=weights[0:heads, local_block * _BLOCK : (local_block + 1) * _BLOCK],
        )
    weights_t = sb.tile((_P_MAX, group_blocks, heads), bf16)
    if weights_on_act:
        nisa.tensor_copy(dst=weights_t, src=weights_psum, engine=nisa.engine.scalar)
    else:
        nisa.tensor_copy(dst=weights_t, src=weights_psum)
    numerator_psum = nl.ndarray((heads, _KV_LORA), dtype=fp32, buffer=nl.psum)
    for local_block in range(group_blocks):
        nisa.nc_matmul(
            dst=numerator_psum,
            stationary=weights_t[0:_P_MAX, local_block, 0:heads],
            moving=key_rows[0:_P_MAX, local_block, 0:_KV_LORA],
            accumulate=local_block > 0,
        )
    sb.close_scope()
    return numerator_psum


def _stack_step(heads: int) -> int:
    """``stack_rows``: the partition (and stationary column) step of the rows of a unit for ``heads`` heads a row. An
    engine access of at most 32 partitions may move between quadrants, and one of at most 64 between the halves, so the
    state add of row ``u`` reads its partitions ``step u ..`` and writes the per-row state at partition 0."""
    kernel_assert(heads <= _P_MAX // 2, f"stack_rows needs at most {_P_MAX // 2} heads a row, got {heads}")
    return _P_MAX // 4 if heads <= _P_MAX // 4 else _P_MAX // 2


class _StackUnit:
    """``stack_rows``: the rows of one sequence of a pass that attend each key group together (``128 / step`` at most:
    4 rows of up to 32 heads, 2 rows of up to 64). Row ``u`` (chunk-local index ``cqs[u]``) holds the stationary columns
    ``step u .. step u + heads`` of ``q_nope`` ``[128, 4, width]`` and ``q_rope`` ``[bias_end, width]`` BF16, and the
    partitions ``step u .. step u + heads`` of the capped-softmax operands ``cap`` and ``bias`` ``[width, 1]`` FP32. The
    other columns and partitions are zero. A two-input engine op takes both inputs at one partition start, so the FP32
    state of each row ``u > 0`` runs on the partitions ``step u ..`` of ``shadow`` ``[128, 513]`` during the key groups
    (``_stack_state_in`` / ``_stack_state_out`` copy it from and back to the per-row state, as it is)."""

    def __init__(self, cqs, q_nope, q_rope, cap, bias, width, step, shadow):
        self.cqs = cqs
        self.step = step
        self.q_nope = q_nope
        self.q_rope = q_rope
        self.cap = cap
        self.bias = bias
        self.width = width
        self.shadow = shadow

    def state_view(self, state, u, heads):
        """The FP32 state ``[heads, 513]`` that row ``u`` adds into during the key groups."""
        if u == 0:
            return state[0:heads, self.cqs[0], 0:_STATE]
        p0 = self.step * u
        if self.shadow is None:
            # ``STACK_STATE_UPPER``: the partitions ``step u ..`` of the column of row 0.
            kernel_assert(state.shape[0] >= p0 + heads, "STACK_STATE_UPPER needs a 128-partition pass state")
            return state[p0 : p0 + heads, self.cqs[0], 0:_STATE]
        return self.shadow[p0 : p0 + heads, 0:_STATE]

    def merged_view(self, state, heads):
        """``STACK_STATE_UPPER`` with ``step == heads``: the FP32 state ``[width, 513]`` of all the rows of the unit (row
        ``u`` on the partitions ``heads u ..``), or None (one add a row)."""
        if self.shadow is not None or len(self.cqs) < 2 or self.step != heads or state.shape[0] < self.width:
            return None
        return state[0 : self.width, self.cqs[0], 0:_STATE]


def _stack_units(cqs, q_nope_all, q_rope_all, softmax, heads, bias_end, sb):
    """``stack_rows``: the units of the rows ``cqs`` of one sequence (in row order, ``128 / step`` a unit). The copies
    move the BF16 query words and the FP32 cap and reference words as they are, so every operand of a row keeps its
    bits."""
    kernel_assert(not softmax.exact, "stack_rows needs the capped softmax")
    step = _stack_step(heads)
    per_unit = _P_MAX // step
    bf16 = nl.bfloat16
    fp32 = nl.float32
    vector = nisa.engine.vector
    scalar = nisa.engine.scalar
    units = []
    for u0 in range(0, len(cqs), per_unit):
        unit_cqs = cqs[u0 : u0 + per_unit]
        width = step * (len(unit_cqs) - 1) + heads
        q_nope = sb.tile((_P_MAX, _KV_LORA_TILES, width), bf16)
        q_rope = sb.tile((bias_end, width), bf16)
        cap = sb.tile((width, 1), fp32)
        bias = sb.tile((width, 1), fp32)
        shadow = sb.tile((_P_MAX, _STATE), fp32) if len(unit_cqs) > 1 and not STACK_STATE_UPPER else None
        nisa.memset(dst=q_nope, value=0.0, engine=vector)
        nisa.memset(dst=q_rope, value=0.0, engine=vector)
        nisa.memset(dst=cap, value=0.0, engine=vector)
        nisa.memset(dst=bias, value=0.0, engine=vector)
        for u, cq in enumerate(unit_cqs):
            c0 = step * u
            nisa.tensor_copy(
                dst=q_nope[0:_P_MAX, 0:_KV_LORA_TILES, c0 : c0 + heads],
                src=q_nope_all[0:_P_MAX, cq, 0:_KV_LORA_TILES, 0:heads],
                engine=scalar,
            )
            nisa.tensor_copy(
                dst=q_rope[0:bias_end, c0 : c0 + heads], src=q_rope_all[0:bias_end, cq, 0:heads], engine=scalar
            )
            nisa.tensor_copy(dst=cap[c0 : c0 + heads, 0:1], src=softmax.score_cap[0:heads, cq : cq + 1], engine=vector)
            nisa.tensor_copy(
                dst=bias[c0 : c0 + heads, 0:1], src=softmax.negative_reference[0:heads, cq : cq + 1], engine=vector
            )
        units.append(_StackUnit(unit_cqs, q_nope, q_rope, cap, bias, width, step, shadow))
    return units


def _stack_state_in(units, state, heads):
    """``stack_rows``: before the key groups of a sequence, copy the FP32 state of each row ``u > 0`` of each unit to its
    shadow partitions (a one-input copy may move between quadrants or halves). The copy keeps every bit."""
    for unit in units:
        for u, cq in enumerate(unit.cqs[1:], start=1):
            nisa.tensor_copy(
                dst=unit.state_view(state, u, heads), src=state[0:heads, cq, 0:_STATE], engine=nisa.engine.vector
            )


def _stack_state_out(units, state, heads):
    """``stack_rows``: after the key groups of a sequence, copy the shadow state of each row ``u > 0`` back to the
    per-row state."""
    for unit in units:
        for u, cq in enumerate(unit.cqs[1:], start=1):
            nisa.tensor_copy(
                dst=state[0:heads, cq, 0:_STATE], src=unit.state_view(state, u, heads), engine=nisa.engine.vector
            )


def _attend_group_stacked(unit, latent_keys, rope_keys, key_rows, softmax, state, heads, sb, mask_rows=1):
    """``stack_rows``: attend the rows of ``unit`` over one prior-key group, then add each row's numerator and weight sum
    to its FP32 state, as ``_attend_group`` and ``_Softmax.update`` (capped softmax) do for one row. The matmuls take the
    unit's stationary columns, the cap, the exp and the weight copy work on all its partitions, and the state add of row
    ``u`` reads partitions ``step u .. step u + heads``."""
    sb.open_scope("group_stacked")
    bf16 = nl.bfloat16
    fp32 = nl.float32
    vector = nisa.engine.vector
    tensor = nisa.engine.tensor
    width = unit.width
    group_blocks = key_rows.shape[1]
    num_keys = group_blocks * _BLOCK
    scores_psum = nl.ndarray((width, num_keys), dtype=fp32, buffer=nl.psum)
    for latent_tile in range(_KV_LORA_TILES):
        nisa.nc_matmul(
            dst=scores_psum,
            stationary=unit.q_nope[0:_P_MAX, latent_tile, 0:width],
            moving=latent_keys[0:_P_MAX, latent_tile, 0:num_keys],
            accumulate=latent_tile > 0,
        )
    nisa.nc_matmul(
        dst=scores_psum,
        stationary=unit.q_rope[0 : _BIAS_ROW + mask_rows, 0:width],
        moving=rope_keys[0 : _BIAS_ROW + mask_rows, 0:num_keys],
        accumulate=True,
    )
    nisa.tensor_scalar(
        dst=scores_psum, data=scores_psum, op0=nl.minimum, operand0=unit.cap[0:width, 0:1], engine=vector
    )
    weights = sb.tile((width, num_keys), bf16)
    group_sum = sb.tile((width, 1), fp32)
    nisa.activation(
        dst=weights,
        op=nl.exp,
        data=scores_psum,
        scale=softmax.scale,
        bias=unit.bias[0:width, 0:1],
        reduce_op=nl.add,
        reduce_res=group_sum,
        reduce_cmd=nisa.reduce_cmd.reset_reduce,
        engine=nisa.engine.scalar,
    )
    weights_psum = nl.ndarray((_P_MAX, group_blocks, width), dtype=bf16, buffer=nl.psum)
    for local_block in range(group_blocks):
        nisa.nc_transpose(
            dst=weights_psum[0:_P_MAX, local_block, 0:width],
            data=weights[0:width, local_block * _BLOCK : (local_block + 1) * _BLOCK],
            engine=tensor,
        )
    weights_t = sb.tile((_P_MAX, group_blocks, width), bf16)
    nisa.tensor_copy(dst=weights_t, src=weights_psum, engine=nisa.engine.scalar if STACK_WEIGHTS_ON_ACT else vector)
    numerator_psum = nl.ndarray((width, _KV_LORA), dtype=fp32, buffer=nl.psum)
    for local_block in range(group_blocks):
        nisa.nc_matmul(
            dst=numerator_psum,
            stationary=weights_t[0:_P_MAX, local_block, 0:width],
            moving=key_rows[0:_P_MAX, local_block, 0:_KV_LORA],
            accumulate=local_block > 0,
        )
    merged = unit.merged_view(state, heads)
    for u in range(1 if merged is not None else len(unit.cqs)):
        p0 = unit.step * u
        acc = merged if merged is not None else unit.state_view(state, u, heads)
        parts = width if merged is not None else heads
        nisa.tensor_tensor(
            dst=acc[0:parts, 0:_KV_LORA],
            data1=acc[0:parts, 0:_KV_LORA],
            data2=numerator_psum[p0 : p0 + parts, 0:_KV_LORA],
            op=nl.add,
            engine=vector,
        )
        nisa.tensor_tensor(
            dst=acc[0:parts, _KV_LORA : _KV_LORA + 1],
            data1=acc[0:parts, _KV_LORA : _KV_LORA + 1],
            data2=group_sum[p0 : p0 + parts, 0:1],
            op=nl.add,
            engine=vector,
        )
    sb.close_scope()


def _attend_group_scores(
    cq, q_nope_all, q_rope_all, latent_keys, rope_keys, softmax, group_sum, weights, heads, mask_rows=1
):
    """``row_pipeline``, stage 1 of ``_attend_group`` (capped softmax): the scores of query ``cq`` over one key group,
    capped, and their BF16 exponentials into ``weights`` ``[heads, keys]`` with the FP32 sum into ``group_sum``. The
    instructions of ``_attend_group``. Return the handle of the rope matmul that closes the score group."""
    fp32 = nl.float32
    num_keys = weights.shape[1]
    scores_psum = nl.ndarray((heads, num_keys), dtype=fp32, buffer=nl.psum)
    for latent_tile in range(_KV_LORA_TILES):
        nisa.nc_matmul(
            dst=scores_psum,
            stationary=q_nope_all[0:_P_MAX, cq, latent_tile, 0:heads],
            moving=latent_keys[0:_P_MAX, latent_tile, 0:num_keys],
            accumulate=latent_tile > 0,
        )
    scores_end = nisa.nc_matmul(
        dst=scores_psum,
        stationary=q_rope_all[0 : _BIAS_ROW + mask_rows, cq, 0:heads],
        moving=rope_keys[0 : _BIAS_ROW + mask_rows, 0:num_keys],
        accumulate=True,
    )
    nisa.tensor_scalar(
        dst=scores_psum, data=scores_psum, op0=nl.minimum, operand0=softmax.score_cap[0:heads, cq : cq + 1]
    )
    nisa.activation(
        dst=weights,
        op=nl.exp,
        data=scores_psum,
        scale=softmax.scale,
        bias=softmax.negative_reference[0:heads, cq : cq + 1],
        reduce_op=nl.add,
        reduce_res=group_sum,
        reduce_cmd=nisa.reduce_cmd.reset_reduce,
    )
    return _instruction(scores_end)


def _attend_group_values(key_rows, weights, weights_t, heads, weights_on_act=False, after=None):
    """``row_pipeline``, stage 2 of ``_attend_group``: transpose the weights ``[heads, keys]`` of one query into
    ``weights_t`` and return the FP32 PSUM numerator ``[heads, 512]`` of the key group ``key_rows``. The instructions of
    ``_attend_group``. ``after``: a TensorE instruction that the first transpose follows on TensorE (a scheduling edge,
    no wait), so the compiler keeps the score matmuls of the next item ahead of these transposes."""
    bf16 = nl.bfloat16
    fp32 = nl.float32
    group_blocks = key_rows.shape[1]
    weights_psum = nl.ndarray((_P_MAX, group_blocks, heads), dtype=bf16, buffer=nl.psum)
    for local_block in range(group_blocks):
        transpose = nisa.nc_transpose(
            dst=weights_psum[0:_P_MAX, local_block, 0:heads],
            data=weights[0:heads, local_block * _BLOCK : (local_block + 1) * _BLOCK],
        )
        if local_block == 0 and after is not None:
            _instruction(transpose).depends_on(after, kind=nisa.dependency.schedule)
    if weights_on_act:
        nisa.tensor_copy(dst=weights_t, src=weights_psum, engine=nisa.engine.scalar)
    else:
        nisa.tensor_copy(dst=weights_t, src=weights_psum)
    numerator_psum = nl.ndarray((heads, _KV_LORA), dtype=fp32, buffer=nl.psum)
    for local_block in range(group_blocks):
        nisa.nc_matmul(
            dst=numerator_psum,
            stationary=weights_t[0:_P_MAX, local_block, 0:heads],
            moving=key_rows[0:_P_MAX, local_block, 0:_KV_LORA],
            accumulate=local_block > 0,
        )
    return numerator_psum


def _sanitize_cache_rows(rows, bound=None):
    """Clamp each cache row of ``rows`` (SBUF) to a finite bound, in place (see ``tkg_mla_paged``). ``bound`` (default: by
    the dtype of ``rows``) is the FP8 bound for FP8 pages in BF16 tiles (``loop_cast_engines``)."""
    if bound is None:
        bound = _BF16_KEY_MAX if rows.dtype == nl.bfloat16 else fp8_page_max()
    nisa.tensor_scalar(
        dst=rows, data=rows, op0=nl.maximum, operand0=-bound, op1=nl.minimum, operand1=bound, engine=nisa.engine.vector
    )


def _transpose_rows(dst, data):
    """Transpose cache rows ``data`` (``[128 keys, F]`` SBUF) into ``dst`` (``[F, 128]`` BF16 PSUM).

    Trn3 upcasts FP8 rows with a matmul into BF16 PSUM. Trn2 has no BF16 matmul output: the rows upcast exactly to a
    BF16 tile first, and a BF16 transpose moves it.
    """
    if data.dtype == nl.bfloat16:
        nisa.nc_transpose(dst=dst, data=data)
    elif is_trn3():
        identity = nl.shared_identity_matrix(_P_MAX, dtype=nl.bfloat16)
        nisa.nc_matmul(dst=dst, stationary=data, moving=identity[0:_P_MAX, 0:_P_MAX])
    else:
        rows = nl.ndarray(tuple(data.shape), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=rows, src=data)
        nisa.nc_transpose(dst=dst, data=rows)


def _row_mask(rope_keys, slot_offset, row_threshold, r, group_base, group_keys, sb):
    """Rewrite the prior-key mask of ``rope_keys`` for query row ``r`` (bitwise mode).

    A key with a slot offset at or above ``row_threshold[r] - group_base`` gets the mask bias (``group_base`` is
    the group index times the group stride). Each row of a sequence has its own threshold ``position[b] + j``,
    so the mask is rewritten per row.
    """
    thr = sb.tile((_P_MAX, 1), nl.float32)
    nisa.tensor_scalar(
        dst=thr[_BIAS_ROW : _BIAS_ROW + 1, 0:1],
        data=row_threshold[_BIAS_ROW : _BIAS_ROW + 1, r : r + 1],
        op0=nl.subtract,
        operand0=float(group_base),
    )
    nisa.tensor_scalar(
        dst=rope_keys[_BIAS_ROW : _BIAS_ROW + 1, 0:group_keys],
        data=slot_offset[_BIAS_ROW : _BIAS_ROW + 1, 0:group_keys],
        op0=nl.greater_equal,
        operand0=thr[_BIAS_ROW : _BIAS_ROW + 1, 0:1],
        op1=nl.multiply,
        operand1=_MASK_BIAS,
    )


def _transpose_group(
    rows, latent_keys, rope_keys, slot_offset, threshold, lower=None, sb=None, mask_rows=1, key_path=None
):
    """Transpose the cache rows of one key group into feature-major tiles, and write the mask row below
    the rope keys: a key with a slot offset of at least ``threshold`` gets the mask bias. With ``lower``,
    a key with a slot offset below ``lower`` is also masked (the ``key_start`` lower bound). With ``mask_rows``,
    ``threshold`` and ``lower`` hold one value on each mask row, and the call writes all mask rows. ``key_path``
    (``_KeyPath``, None = BF16) writes the latent keys."""
    mask_end = _BIAS_ROW + mask_rows
    group_blocks = rows.shape[1]
    num_keys = group_blocks * _BLOCK
    (key_path or _KeyPath()).prep(rows, latent_keys)
    rope_psum = nl.ndarray((_ROPE, num_keys), dtype=nl.bfloat16, buffer=nl.psum)
    for local_block in range(group_blocks):
        _transpose_rows(
            dst=rope_psum[0:_ROPE, local_block * _BLOCK : (local_block + 1) * _BLOCK],
            data=rows[0:_P_MAX, local_block, _KV_LORA:_CACHE_WIDTH],
        )
    nisa.tensor_copy(dst=rope_keys[0:_ROPE, 0:num_keys], src=rope_psum)
    nisa.tensor_scalar(
        dst=rope_keys[_BIAS_ROW:mask_end, 0:num_keys],
        data=slot_offset[_BIAS_ROW:mask_end, 0:num_keys],
        op0=nl.greater_equal,
        operand0=threshold,
        op1=nl.multiply,
        operand1=_MASK_BIAS,
    )
    if lower is not None:
        low_mask = sb.tile((_P_MAX, num_keys), nl.float32)
        nisa.tensor_scalar(
            dst=low_mask[_BIAS_ROW:mask_end, 0:num_keys],
            data=slot_offset[_BIAS_ROW:mask_end, 0:num_keys],
            op0=nl.less,
            operand0=lower,
            op1=nl.multiply,
            operand1=_MASK_BIAS,
        )
        nisa.tensor_tensor(
            dst=rope_keys[_BIAS_ROW:mask_end, 0:num_keys],
            data1=rope_keys[_BIAS_ROW:mask_end, 0:num_keys],
            data2=low_mask[_BIAS_ROW:mask_end, 0:num_keys],
            op=nl.add,
        )


def _output_tail(
    rs_pass,
    x_hT_pass,
    gate_chunks,
    o_w_chunks,
    kv_b_w,
    y_hT_pass,
    rows,
    local_heads,
    absorb_width,
    output_gate,
    tp_group,
    prev_tail,
    ag_buf,
    state_hbm,
    state_dst_hbm,
    tag,
    sb,
    cp=_TP,
    out_hbm=None,
    rs_buf=None,
    gather=True,
    dense_state=False,
    partial_out=None,
    crit_dma=None,
    state_store=None,
    state_rows=None,
    gate_psum=None,
    state_bf16_o=False,
    state_sums=(None, None),
    rs_hook=None,
    sums_first=None,
    defer=False,
    state_rs_after=(),
    state_store_split=False,
    state_sum_aps=None,
):
    """Run the output tail for one pass of ``rows`` query rows and write ``y_hT_pass``.

    The pass reduce-scatters the softmax state by heads, normalizes, projects the value (and the
    sigmoid gate for the target), runs ``o_proj``, and TP-sums the hT partial with a reduce-scatter and
    an all-gather. ``prev_tail`` is the previous pass's final all-gather; the state reduce-scatter waits
    for it, so a pass never reuses the previous pass's collective buffers before it completes. Returns
    this pass's final all-gather.

    With ``out_hbm`` (``cp=128``, or ``out_ar_call``) the output sum is one all-reduce in its persistent HBM pair.

    ``ag_buf`` is this pass's all-gather source buffer, one of two the caller alternates by chunk parity.
    The reduce-scatter sources rely on the NRT end handshake, so they need no such buffer; the all-gather
    source does, because a CP peer can still read it after the local output lands. The parity split plus
    the ``prev_tail`` chaining means a buffer is reused only after a later all-gather has completed.

    ``dense_state`` (HBM state reduce-scatter only) loads the local state with one row of a head on each partition,
    ``[local_heads * rows, 513]``, in place of ``[local_heads, rows, 513]``, so it takes 2 KB per partition, not 32 KB.
    The values are the same.

    ``partial_out`` takes the output partial of the pass, and the caller runs the output reduce-scatter of all
    passes; the pass then returns its state reduce-scatter. ``gate_psum`` is the gate projection of the pass when the
    caller ran it (``late_tail_frames``). ``state_bf16_o``: the BF16-o state reduce-scatter (``_state_rs_bf16_o``),
    with the sums in the pair ``state_sums`` when it is given. ``rs_hook``: traced right after the issue of the state
    reduce-scatter (``mla_block_attention`` ``state_rs_hook``).

    ``defer`` (the FP32 HBM state of ``pass_tail_overlap``) issues only the state store and the state reduce-scatter,
    and returns ``(compute, state_rs)``: ``compute(copy_after=())`` runs the state load, the value and output
    projections and the partial copies (the output reduce-scatter source; they wait on ``copy_after``), and returns
    ``(copies, collectives)``; ``collectives(after=None)`` runs the output reduce-scatter (after ``after``) and
    all-gather and returns the final all-gather.
    ``state_rs_after``: instructions that the state reduce-scatter waits on (a runtime dependency).
    ``state_store_split``: the FP32 HBM state store (no ``state_store``) runs as two DMAs of the column halves, on the SP
    and the ScalarE HWDGE queues. The same bytes land, so the output does not change.
    """
    bf16 = nl.bfloat16
    fp32 = nl.float32
    crit_dma = crit_dma or {}
    # ``state_store`` (a function of the tag and the DMA options; the state of ``p2``): the state is not an SBUF tile of
    # this layout, and the function stores it to ``state_hbm``. ``state_rows`` then gives its rows.
    kernel_assert(state_store is None or state_hbm is not None, "state_store needs the HBM state reduce-scatter")
    # With ``state_bf16_o`` the ``state_store`` of p2 takes the tag, the sums tensor and the rows of a pass of the pair.
    state_dtype = fp32 if state_store is not None else rs_pass.dtype
    state_rows = state_rows if state_store is not None else rs_pass.shape[0]
    kernel_assert(
        not defer
        or (
            not state_bf16_o
            and state_store is None
            and state_hbm is not None
            and partial_out is None
            and out_hbm is None
        ),
        "defer is the FP32 HBM state tail with its own output collectives (pass_tail_overlap)",
    )

    def _compute(local_state, copies_out=None, copy_after=()):
        return _tail_partial(
            local_state,
            x_hT_pass,
            gate_chunks,
            o_w_chunks,
            kv_b_w,
            rows,
            local_heads,
            absorb_width,
            output_gate,
            sb,
            dense_state,
            partial=partial_out,
            rs_buf=rs_buf,
            gate_psum=gate_psum,
            copies_out=copies_out,
            copy_after=copy_after,
        )

    def _collectives(partial, after=None):
        if partial_out is not None:
            return state_rs
        if out_hbm is not None:
            out_src, out_dst = out_hbm["out_src_hbm"], out_hbm["out_dst_hbm"]
            cols = rows * _HIDDEN_TILES
            nisa.dma_copy(dst=out_src[0:_P_MAX, 0:cols], src=partial.reshape((_P_MAX, cols)), name=f"{tag}_out_store")
            out_ar = release_weight_gate(
                ncc.all_reduce(srcs=[out_src], dsts=[out_dst], replica_group=tp_group, op=nl.add, name=f"{tag}_out_ar")
            )
            nisa.dma_copy(dst=y_hT_pass.reshape((_P_MAX, cols)), src=out_dst[0:_P_MAX, 0:cols], name=f"{tag}_out_load")
            return out_ar
        # The all-gather source is the caller's persistent parity buffer, not a fresh tile.
        partial_sum = ag_buf
        out_rs = release_weight_gate(
            ncc.reduce_scatter(
                srcs=[partial.reshape((_P_MAX, rows * _HIDDEN_TILES))],
                dsts=[partial_sum.reshape((_P_MAX // cp, rows * _HIDDEN_TILES))],
                replica_group=tp_group,
                collective_dim=0,
                op=nl.add,
                name=f"{tag}_out_rs",
            )
        )
        if after is not None:
            # ``defer``: the parity buffers of the output collectives are reused after the pass before's all-gather.
            out_rs.depends_on(after)
        if not gather:
            # The caller gathers the reduce-scatter slices of all passes with one all-gather.
            if not nl.is_hbm(getattr(partial_sum, "buffer", None)):
                # A runtime edge on a collective breaks the released neuronx-cc 2.28 (tkg_weight_dma_gate.wait_for_completion).
                # So a GpSimd x * 1 of one element of each partition of the slice follows the reduce-scatter (compiler
                # RAW and WAW edges), and the all-gather reads it. A later wait on this reduce-scatter waits on this
                # relay (``_tail_partial`` ``copy_after``). The values do not change.
                view = partial_sum.reshape((_P_MAX // cp, rows * _HIDDEN_TILES))[0 : _P_MAX // cp, 0:1]
                relay = nisa.tensor_scalar(
                    dst=view,
                    data=view,
                    op0=nl.multiply,
                    operand0=1.0,
                    engine=nisa.engine.gpsimd,
                    name=f"{tag}_rs_relay",
                )
                _RS_RELAY_OF[id(out_rs)] = (out_rs, relay)
            return out_rs
        out_ag = release_weight_gate(
            ncc.all_gather(
                srcs=[partial_sum.reshape((_P_MAX // cp, rows * _HIDDEN_TILES))],
                dsts=[y_hT_pass.reshape((_P_MAX, rows * _HIDDEN_TILES))],
                replica_group=tp_group,
                collective_dim=0,
                name=f"{tag}_out_ag",
            )
        )
        return out_ag

    if state_bf16_o:
        local_state, state_rs = _state_rs_bf16_o(
            rs_pass,
            state_hbm,
            state_dst_hbm,
            tp_group,
            prev_tail,
            local_heads,
            rows,
            dense_state,
            sb,
            tag,
            state_sums,
            rs_hook,
            store=state_store,
            store_heads=state_rows,
            sums_first=sums_first,
            **({"sum_aps_of": state_sum_aps} if state_sum_aps is not None else {}),
        )
    else:
        # The dense tile is not a reduce-scatter destination, so the SBUF form keeps the head-major tile.
        local_state = None if dense_state else sb.tile((local_heads, rows, _KV_LORA + 1), state_dtype)
        # The state reduce-scatter runs in persistent HBM when it is given (fp32 state): an SBUF reduce-scatter
        # gets a compiler scratch, which the Neuron runtime writes only on Trn2. Store the state to
        # HBM, reduce-scatter HBM to HBM, then load the local shard. BF16 state keeps the SBUF reduce-scatter.
        rs_src = None if state_store is not None else rs_pass.reshape((state_rows, rows * (_KV_LORA + 1)))
        rs_dst = None if dense_state else local_state.reshape((local_heads, rows * (_KV_LORA + 1)))
        in_hbm = state_hbm is not None and state_dtype == fp32
        kernel_assert(in_hbm or not dense_state, "dense_state needs the FP32 HBM state reduce-scatter")
        width = rows * (_KV_LORA + 1)
        if in_hbm:
            # A wider buffer (``reduce_rows``) keeps the reduction size of the verify; the rows past ``rows`` are not
            # read.
            state_dst = state_hbm if state_hbm.shape[1] == width else state_hbm[0:state_rows, 0:width]
            if state_store is not None:
                state_store(tag, crit_dma)
            elif state_store_split:
                half = width // 2
                for part, (c0, c1), engine in (
                    (0, (0, half), nisa.engine.sync),
                    (1, (half, width), nisa.engine.scalar),
                ):
                    nisa.dma_copy(
                        dst=state_dst[0:state_rows, c0:c1],
                        src=rs_src[0:state_rows, c0:c1],
                        dge_mode=nisa.dge_mode.hwdge,
                        engine=engine,
                        name=f"{tag}_mla_state_store_{part}",
                        **crit_dma,
                    )
            else:
                nisa.dma_copy(dst=state_dst, src=rs_src, name=f"{tag}_mla_state_store", **crit_dma)
            rs_src, rs_dst = state_hbm, state_dst_hbm
        state_rs = ncc.reduce_scatter(
            srcs=[rs_src],
            dsts=[rs_dst],
            replica_group=tp_group,
            collective_dim=0,
            op=nl.add,
            name=f"{tag}_state_rs",
        )
        if prev_tail is not None:
            state_rs.depends_on(prev_tail)
        # ``rs_hook`` (``state_rs_hook``, a one-pass call): work in the flight of the reduce-scatter.
        _run_hook(rs_hook, sb, "state_rs_hook")
        if state_rs_after:
            # ``pass_tail_overlap``: the partial copies of the pass before (its output reduce-scatter source) come
            # before this reduce-scatter, the collective before that output reduce-scatter in the group.
            state_rs.depends_on(*state_rs_after, kind=nisa.dependency.runtime)

        def load_state(local_state=local_state, rs_dst=rs_dst):
            if in_hbm and dense_state:
                local_state = sb.tile((local_heads * rows, _KV_LORA + 1), fp32)
                for local_head in range(local_heads):
                    nisa.dma_copy(
                        dst=local_state[local_head * rows : (local_head + 1) * rows, 0 : _KV_LORA + 1],
                        src=rs_dst.ap(
                            pattern=[[_KV_LORA + 1, rows], [1, _KV_LORA + 1]], offset=local_head * rs_dst.shape[1]
                        ),
                        name=f"{tag}_mla_state_load_{local_head}",
                        **crit_dma,
                    )
            elif in_hbm:
                state_src = rs_dst if rs_dst.shape[1] == width else rs_dst[0:local_heads, 0:width]
                nisa.dma_copy(
                    dst=local_state.reshape((local_heads, rows * (_KV_LORA + 1))),
                    src=state_src,
                    name=f"{tag}_mla_state_load",
                    **crit_dma,
                )
            return local_state

        if defer:

            def compute(copy_after=()):
                copies = []
                partial = _compute(load_state(), copies, copy_after)
                return copies, lambda after=None: _collectives(partial, after)

            return compute, state_rs
        local_state = load_state()
    return _collectives(_compute(local_state))


def _state_rs_bf16_o(
    state,
    state_hbm,
    state_dst_hbm,
    tp_group,
    prev_tail,
    local_heads,
    rows,
    dense_state,
    sb,
    tag,
    sums=(None, None),
    rs_hook=None,
    store=None,
    store_heads=None,
    sums_first=None,
    sum_aps_of=None,
):
    """The BF16-o state reduce-scatter of one pass (``mla_block_attention`` ``state_bf16_o``).

    The FP32 state ``[heads, rows, 513]`` goes to the BF16-o views of the HBM pair (``common.mla_state_bf16_o_views``):
    BF16 numerators and FP32 sums, each reduce-scattered by heads, the two collectives issued together. A wider pair
    (``reduce_rows``) keeps its reduction size, as the FP32 form. ``sums``: the (source, destination) pair of the sums
    when they take tensors of their own. ``rs_hook``: traced in a scope of ``sb`` right after the issue of the two
    collectives. Return the local ``StateParts`` and the numerator reduce-scatter. ``store`` (the state of ``p2``,
    ``_store_state_p2_bf16_o``; ``store_heads`` its heads): a function of the tag, the sums tensor and the rows of the
    pair that writes the same words, in place of the store of the head-major ``state``. ``sum_aps_of``
    (``state_sum_blocks_pass``): a function of the destination sums view and ``dense_state`` that gives the loads of each
    head's rows from the reduced sum blocks (``_sum_block_loads``; ``common.load_state_bf16_o`` ``sum_aps``).
    """
    kernel_assert(
        state_hbm is not None and (store is not None or state.dtype == nl.float32),
        "state_bf16_o needs the FP32 HBM state pair",
    )
    o_src, sum_src, view_rows = mla_state_bf16_o_views(state_hbm, _KV_LORA, sums=sums[0])
    o_dst, sum_dst, _ = mla_state_bf16_o_views(state_dst_hbm, _KV_LORA, sums=sums[1])
    kernel_assert(view_rows >= rows, f"the state pair holds {view_rows} rows, the pass {rows}")
    if store is not None:
        kernel_assert(store_heads is not None, "the p2 state store of state_bf16_o needs its heads")
        store(f"{tag}_mla_state_store", sums[0], view_rows)
    else:
        heads = state.shape[0]
        store_state_bf16_o(
            state,
            state_rows_plain(o_src, view_rows, heads, _KV_LORA),
            state_rows_plain(sum_src, sum_src.shape[1], heads, 0),
            _KV_LORA,
            sb,
            f"{tag}_mla_state_store",
        )
    o_rs, sum_rs = _bf16_o_reduce_scatters(
        o_src, sum_src, o_dst, sum_dst, tp_group, f"{tag}_state_rs", sums_first=sums_first
    )
    if prev_tail is not None:
        o_rs.depends_on(prev_tail)
        sum_rs.depends_on(prev_tail)
    _run_hook(rs_hook, sb, "state_rs_hook")
    parts = load_state_bf16_o(
        o_dst,
        sum_dst,
        view_rows,
        lambda h: h,
        local_heads,
        rows,
        _KV_LORA,
        dense_state,
        sb,
        f"{tag}_mla_state",
        **({"sum_aps": sum_aps_of(sum_dst, dense_state)} if sum_aps_of is not None else {}),
    )
    return parts, o_rs


def _run_hook(hook, sb, name: str):
    """Trace ``hook(sb)`` in a scope of its own, or nothing when ``hook`` is None."""
    if hook is None:
        return
    with sb.scope(name):
        hook(sb)


def _bf16_o_reduce_scatters(o_src, sum_src, o_dst, sum_dst, tp_group, name, sums_first=None):
    """Issue the two reduce-scatters of the BF16-o state by heads (``sums_first``, None: ``SUM_RS_FIRST``: the sums
    first); return (numerators, sums)."""

    def sums_rs():
        return ncc.reduce_scatter(
            srcs=[sum_src], dsts=[sum_dst], replica_group=tp_group, collective_dim=0, op=nl.add, name=f"{name}_sum"
        )

    sum_rs = sums_rs() if (SUM_RS_FIRST if sums_first is None else sums_first) else None
    o_rs = ncc.reduce_scatter(
        srcs=[o_src], dsts=[o_dst], replica_group=tp_group, collective_dim=0, op=nl.add, name=name
    )
    if sum_rs is None:
        sum_rs = sums_rs()
    return o_rs, sum_rs


def _store_call_state_bf16_o(
    state, state_hbm, sums, rows, rows_per_pass, passes, pass_index, sb, tag, dma_options=None
):
    """Store the state of one pass of the one state reduce-scatter a call in the BF16-o form.

    The pass-minor layout of ``_store_pass_state`` (tensor row ``head * passes + pass_index``), or with ``rows`` (the
    global rows of a sequence-major pass) the columns of ``_store_rows_state``. ``sums``: the sums source tensor when
    the sums take tensors of their own, else None.
    """
    heads = state.shape[0]
    o_view, sum_view, view_rows = mla_state_bf16_o_views(state_hbm, _KV_LORA, sums=sums)
    kernel_assert(view_rows == rows_per_pass, f"the call state pair holds {view_rows} rows a pass, not {rows_per_pass}")
    # The sums of a tensor of their own can be wider than the pass rows (``cc_pad256``): their row is the tensor's row.
    sum_width = sum_view.shape[1]
    if rows is None:
        o_rows = state_rows_plain(o_view, view_rows, heads, _KV_LORA, row0=pass_index, stride=passes)
        sum_rows = state_rows_plain(sum_view, sum_width, heads, 0, row0=pass_index, stride=passes)
    else:

        def runs(view, width, row_width):
            def rows_of(r0, r):
                pieces = []
                i = r0
                while i < r0 + r:
                    n = 1
                    while i + n < r0 + r and rows[i + n] == rows[i] + n and (rows[i] + n) % rows_per_pass:
                        n += 1
                    pass_of, column = divmod(rows[i], rows_per_pass)
                    ap = view.ap(
                        pattern=[[passes * row_width * width, heads], [width, n], [1, width]],
                        offset=(pass_of * row_width + column) * width,
                    )
                    pieces.append((ap, i, n))
                    i += n
                return pieces

            return rows_of

        o_rows, sum_rows = runs(o_view, _KV_LORA, view_rows), runs(sum_view, 1, sum_width)
    store_state_bf16_o(state, o_rows, sum_rows, _KV_LORA, sb, f"{tag}_mla_state_store", dma=dma_options)


def _state_rs_bf16_o_call(state_pair, tp_group, tag, sum_pair=(None, None), sums_first=None):
    """The one BF16-o state reduce-scatter pair of a call; return (the numerator one, the destination views, the sums
    one). ``sum_pair``: the sums tensors when they take tensors of their own."""
    o_src, sum_src, view_rows = mla_state_bf16_o_views(state_pair[0], _KV_LORA, sums=sum_pair[0])
    o_dst, sum_dst, _ = mla_state_bf16_o_views(state_pair[1], _KV_LORA, sums=sum_pair[1])
    o_rs, sum_rs = _bf16_o_reduce_scatters(
        o_src, sum_src, o_dst, sum_dst, tp_group, f"{tag}_state_rs_call", sums_first=sums_first
    )
    return o_rs, (o_dst, sum_dst, view_rows), sum_rs


# id(output reduce-scatter) -> (the reduce-scatter, its GpSimd relay) of ``_collectives`` (no gather). The entries keep
# the handles, so an id cannot be reused by another instruction while it is a key.
_RS_RELAY_OF = {}


def _tail_partial(
    local_state,
    x_hT_pass,
    gate_chunks,
    o_w_chunks,
    kv_b_w,
    rows,
    local_heads,
    absorb_width,
    output_gate,
    sb,
    dense_state=False,
    partial=None,
    rs_buf=None,
    gate_psum=None,
    early_gate=False,
    act_after=None,
    copies_out=None,
    copy_after=(),
):
    """Normalize the local state, project the value and the gate, run o_proj, and write the hT partial of one pass to
    ``partial`` (default ``rs_buf``, else a fresh tile). Return the partial. A given ``gate_psum`` is the gate
    projection of the pass, which the caller ran; with ``early_gate`` (``GATE_BEFORE_STATE``) it ran before the state
    loads, and the value projection's first TensorE instruction is pinned after its last matmul. ``act_after``
    (``tail_o_proj_first``): the gate exp follows that instruction in the schedule (no wait). ``copies_out``: a list that
    takes the copies of the partial out of PSUM; ``copy_after``: instructions those copies wait on (a runtime
    dependency)."""
    bf16 = nl.bfloat16
    fp32 = nl.float32
    if output_gate and gate_psum is None:
        gate_psum = _gate_projection(x_hT_pass, gate_chunks, rows, local_heads)
    value_all = sb.tile((_P_MAX, local_heads, rows), bf16)
    project = _value_projection_dense if dense_state else _value_projection
    # ``GATE_BEFORE_STATE``: a gate traced before the state loads (``early_gate``) runs first on TensorE.
    after = getattr(gate_psum, "last_matmul", None) if early_gate else None
    pin = {"after": after} if after is not None else {}
    project(local_state, kv_b_w, value_all, rows, local_heads, absorb_width, sb, **pin)
    # A loader (late_o_proj_weights): the o_proj weights load after the last reader of the pass state.
    if callable(o_w_chunks):
        o_w_chunks = o_w_chunks()
    if output_gate:
        gate_bf16 = sb.tile((_P_MAX, local_heads, rows), bf16)
        nisa.tensor_copy(dst=gate_bf16, src=gate_psum)
        gate_exp = sb.tile((_P_MAX, local_heads, rows), fp32)
        gate_act = nisa.activation(dst=gate_exp, op=nl.exp, data=gate_bf16, scale=-1.0)
        if act_after is not None:
            _instruction(gate_act).depends_on(act_after, kind=nisa.dependency.schedule)
        nisa.tensor_scalar(dst=gate_exp, data=gate_exp, op0=nl.add, operand0=1.0)
        gate = sb.tile((_P_MAX, local_heads, rows), bf16)
        nisa.reciprocal(dst=gate, data=gate_exp)
        gated = sb.tile((_P_MAX, local_heads, rows), bf16)
        nisa.tensor_tensor(dst=gated, data1=value_all, data2=gate, op=nl.multiply)
    else:
        gated = value_all

    tiles_per_bank = min(_HIDDEN_TILES, _PSUM_BANK_FP32 // rows)
    out_banks = []
    for bank_start in range(0, _HIDDEN_TILES, tiles_per_bank):
        bank_tiles = min(tiles_per_bank, _HIDDEN_TILES - bank_start)
        out_banks.append((bank_start, bank_tiles, nl.ndarray((_P_MAX, bank_tiles, rows), dtype=fp32, buffer=nl.psum)))
    if o_w_chunks and isinstance(o_w_chunks[0][2], MxOChunk):
        # MXFP8 o_proj: the gated value quantized per row (padded-lane order), one MXFP8 matmul a hidden tile.
        o_data, o_scale = _o_proj_mx_quantize(gated, rows, f"mla_o{next(_O_PROJ_CALLS)}", sb)
        for chunk_start, chunk_tiles, chunk in o_w_chunks:
            _o_proj_mx_chunk(out_banks, tiles_per_bank, chunk_start, chunk_tiles, chunk, o_data, o_scale, rows)
        o_w_chunks = ()
    for chunk_start, chunk_tiles, o_w in o_w_chunks:
        for chunk_tile in range(chunk_tiles):
            hidden_tile = chunk_start + chunk_tile
            bank_start, _bank_tiles, out_psum = out_banks[hidden_tile // tiles_per_bank]
            for local_head in range(local_heads):
                nisa.nc_matmul(
                    dst=out_psum[0:_P_MAX, hidden_tile - bank_start, 0:rows],
                    stationary=o_w[0:_P_MAX, local_head, chunk_tile * _P_MAX : (chunk_tile + 1) * _P_MAX],
                    moving=gated[0:_P_MAX, local_head, 0:rows],
                    accumulate=local_head > 0,
                )
    # rs_buf is the persistent reduce-scatter source of the served drafter; None keeps a fresh tile.
    if partial is None:
        partial = rs_buf if rs_buf is not None else sb.tile((_P_MAX, rows, _HIDDEN_TILES), bf16)
    for bank_start, bank_tiles, out_psum in out_banks:
        copy = nisa.tensor_copy(
            dst=partial[0:_P_MAX, 0:rows, bank_start : bank_start + bank_tiles], src=out_psum.permute((0, 2, 1))
        )
        if copy_after:
            # An output reduce-scatter with a relay (``_collectives``): the copy follows it in the schedule and waits
            # for its relay, which completes after it. A runtime edge on a collective breaks the released neuronx-cc
            # 2.28 (tkg_weight_dma_gate.wait_for_completion). Each other instruction keeps the runtime edge.
            relayed = [p for p in copy_after if _RS_RELAY_OF.get(id(p), (None, None))[0] is p]
            others = [p for p in copy_after if not any(p is q for q in relayed)]
            if relayed:
                _instruction(copy).depends_on(*relayed, kind=nisa.dependency.schedule)
                relays = [_instruction(_RS_RELAY_OF[id(p)][1]) for p in relayed]
                _instruction(copy).depends_on(*relays, kind=nisa.dependency.runtime)
            if others:
                _instruction(copy).depends_on(*others, kind=nisa.dependency.runtime)
        if copies_out is not None:
            copies_out.append(_instruction(copy))
    return partial


def _store_pass_state(state, state_hbm, pass_index, passes, tag, dma_options=None):
    """Store the ``[heads, rows, 513]`` state of one pass to HBM row ``head * passes + pass_index`` of each head."""
    heads, rows, width = state.shape
    nisa.dma_copy(
        dst=state_hbm.ap(
            pattern=[[passes * state_hbm.shape[1], heads], [1, rows * width]], offset=pass_index * state_hbm.shape[1]
        ),
        src=state.reshape((heads, rows * width)),
        name=f"{tag}_mla_state_store",
        **(dma_options or {}),
    )


def _store_rows_state(state, state_hbm, rows, rows_per_pass, passes, tag, dma_options=None):
    """Store the ``[heads, len(rows), 513]`` state of a sequence-major pass: local row ``i`` (global row ``rows[i]``)
    goes to column ``rows[i] % rows_per_pass`` of its token-major pass ``rows[i] // rows_per_pass``, one DMA for
    each run of consecutive global rows in one pass."""
    heads, _, width = state.shape
    row_width = state_hbm.shape[1]
    i = 0
    while i < len(rows):
        n = 1
        while i + n < len(rows) and rows[i + n] == rows[i] + n and (rows[i] + n) % rows_per_pass:
            n += 1
        pass_index, column = divmod(rows[i], rows_per_pass)
        nisa.dma_copy(
            dst=state_hbm.ap(
                pattern=[[passes * row_width, heads], [width, n], [1, width]],
                offset=pass_index * row_width + column * width,
            ),
            src=state[0:heads, i : i + n, 0:width],
            name=f"{tag}_mla_state_store_{i}",
            **(dma_options or {}),
        )
        i += n


def _load_pass_state(state_dst_hbm, pass_index, passes, rows, local_heads, sb, tag, dma_options=None):
    """Load the dense local state ``[local_heads * rows, 513]`` of one pass from the pass-minor HBM rows."""
    local_state = sb.tile((local_heads * rows, _KV_LORA + 1), nl.float32)
    row_width = state_dst_hbm.shape[1]
    for local_head in range(local_heads):
        nisa.dma_copy(
            dst=local_state[local_head * rows : (local_head + 1) * rows, 0 : _KV_LORA + 1],
            src=state_dst_hbm.ap(
                pattern=[[_KV_LORA + 1, rows], [1, _KV_LORA + 1]], offset=(local_head * passes + pass_index) * row_width
            ),
            name=f"{tag}_mla_state_load_{local_head}",
            **(dma_options or {}),
        )
    return local_state


def _load_mx_qkv(mla_in_proj, mla_in_proj_scale, sb, tag):
    """Load the MXFP8 q_a / kv_a columns of ``mla_in_proj`` ``[14, 128, 4 * 450]`` and the copies of its scales.

    The entry of ``tkg_mla_paged._qkv_projection_mx``. The scales stay live past the projection scope (``keep``): the
    tail's gate projection reads them.
    """
    groups = mla_in_proj.shape[0]
    qkv_w = sb.tile((_P_MAX, groups, MX_LANES * _QKV_LOCAL), mla_in_proj.dtype)
    gate_weight_dma(
        nisa.dma_copy(
            dst=qkv_w,
            src=mla_in_proj[0:groups, 0:_P_MAX, 0 : MX_LANES * _QKV_LOCAL].permute((1, 0, 2)),
            dge_mode=nisa.dge_mode.hwdge,
            engine=nisa.engine.sync,
            name=f"{tag}_mla_w_qkv_mx",
        )
    )

    def load(dst, src, copy):
        gate_weight_dma(
            nisa.dma_copy(
                dst=dst,
                src=src,
                dge_mode=nisa.dge_mode.hwdge,
                engine=nisa.engine.sync,
                name=f"{tag}_mla_w_in_scale_{copy}",
            )
        )

    scales = load_mx_scales(mla_in_proj_scale, lambda shape, copy: sb.tile(shape, nl.uint8, keep=True), load)
    return ("mx", qkv_w, scales)


_GATE_CALLS = itertools.count()
"""Unique name suffixes of the per-pass MXFP8 hidden of the gate projections."""
_VALUE_CALLS = itertools.count()
"""Unique name suffixes of the per-pass MXFP8 latent of the value projections."""
_O_PROJ_CALLS = itertools.count()
"""Unique name suffixes of the per-pass MXFP8 gated value of the o_proj."""


def _load_mx_gate_cols(mla_in_proj, scales, sb, tag, engine=None, after=()):
    """Load the MXFP8 gate columns of ``mla_in_proj`` ``[14, 128, 4 * 450]`` (columns ``4 * 66 ..``) in 7 DMAs.

    Returns the MXFP8 gate entry of ``_gate_projection``: ``("mx", gate_w, scales, sb, tag)``, with the scales of the
    q_a / kv_a load (one scale set for all columns). ``engine`` (default SP) is the DMA queue engine, and each DMA waits
    for the instructions ``after`` (a runtime dependency; ``mla_block_attention`` ``tail_weights_after_rs``).
    """
    groups = mla_in_proj.shape[0]
    gate_w = sb.tile((_P_MAX, groups, MX_LANES * _GATE_LOCAL_MX), mla_in_proj.dtype)
    for first in range(0, groups, 2):
        count = min(2, groups - first)
        dma = gate_weight_dma(
            nisa.dma_copy(
                dst=gate_w[0:_P_MAX, first : first + count, 0 : MX_LANES * _GATE_LOCAL_MX],
                src=mla_in_proj[first : first + count, 0:_P_MAX, MX_LANES * _QKV_LOCAL : MX_LANES * _IN_WIDTH].permute(
                    (1, 0, 2)
                ),
                dge_mode=nisa.dge_mode.hwdge,
                engine=nisa.engine.sync if engine is None else engine,
                name=f"{tag}_mla_w_gate_mx_{first}",
            )
        )
        if after:
            _instruction(dma).depends_on(*after, kind=nisa.dependency.runtime)
    return ("mx", gate_w, scales, sb, tag)


def _load_mx_q_b(mla_q_b_proj, mla_q_b_proj_scale, sb, tag, keep=True):
    """Load the MXFP8 q_b weight ``[G, 128, 4 * 576]`` as ``[128, G, 576]`` x4 and the copies of its scales.

    The q_b entry of ``tkg_mla_paged._query_columns`` MXFP8. The scales keep past the projection scope (``keep``, the
    default); ``keep=False`` puts them in the current scope with the weight (``q_b_scales_scoped``).
    """
    groups = mla_q_b_proj.shape[0]
    width = mla_q_b_proj.shape[2]
    q_b_w = sb.tile((_P_MAX, groups, width), mla_q_b_proj.dtype)
    gate_weight_dma(
        nisa.dma_copy(
            dst=q_b_w,
            src=mla_q_b_proj[0:groups, 0:_P_MAX, 0:width].permute((1, 0, 2)),
            dge_mode=nisa.dge_mode.hwdge,
            engine=nisa.engine.sync,
            name=f"{tag}_mla_w_q_b_mx",
        )
    )

    def load(dst, src, copy):
        gate_weight_dma(
            nisa.dma_copy(
                dst=dst,
                src=src,
                dge_mode=nisa.dge_mode.hwdge,
                engine=nisa.engine.sync,
                name=f"{tag}_mla_w_q_b_scale_{copy}",
            )
        )

    scales = load_mx_scales(mla_q_b_proj_scale, lambda shape, copy: sb.tile(shape, nl.uint8, keep=keep), load)
    return q_b_w.view(nl.float8_e4m3fn_x4), scales


def _q_b_projection_mx(q_latent, q_b_x4, q_b_scales, q_nope_psum, q_rope_psum, total_rows, local_heads, tag, sb):
    """Write the MXFP8 q_b projection into ``q_nope_psum`` ``[padded, local_heads * NOPE]`` and ``q_rope_psum``
    ``[padded, local_heads * ROPE]``.

    The activation ``q_latent`` ``[128, total_rows, _Q_LORA_TILES]`` is the stationary operand, quantized per query
    row on VectorE; the MXFP8 q_b weight ``[128, G, 576]`` x4 is moving (the Step 13 orientation). The weight columns
    of head ``h`` are the contiguous run ``h * Q_HEAD .. (h + 1) * Q_HEAD`` (nope then rope), an even x4 start, so one
    scale view covers the nope columns and one the rope columns of each head. The order matches the decode, so each
    query row gets the bits of the decode.
    """
    padded = total_rows + total_rows % 2
    q_data, q_scale = quantize_hT_mx(
        q_latent, f"{tag}_qb", padded, lambda shape, dtype, name: sb.tile(shape, dtype, name)
    )
    groups = q_b_x4.shape[1]
    nope_view = q_nope_psum.reshape_dim(1, (local_heads, _NOPE))
    rope_view = q_rope_psum.reshape_dim(1, (local_heads, _ROPE))
    for local_head in range(local_heads):
        base = local_head * _Q_HEAD
        for group in range(groups):
            nisa.nc_matmul_mx(
                dst=nope_view[0:padded, local_head, 0:_NOPE],
                stationary=q_data[0:_P_MAX, group, 0:padded],
                moving=q_b_x4[0:_P_MAX, group, base : base + _NOPE],
                stationary_scale=q_scale[0:_P_MAX, group, 0:padded],
                moving_scale=mx_scale_view(q_b_scales, group, base, _NOPE),
                accumulate=group > 0,
            )
    for local_head in range(local_heads):
        base = local_head * _Q_HEAD
        for group in range(groups):
            nisa.nc_matmul_mx(
                dst=rope_view[0:padded, local_head, 0:_ROPE],
                stationary=q_data[0:_P_MAX, group, 0:padded],
                moving=q_b_x4[0:_P_MAX, group, base + _NOPE : base + _Q_HEAD],
                stationary_scale=q_scale[0:_P_MAX, group, 0:padded],
                moving_scale=mx_scale_view(q_b_scales, group, base + _NOPE, _ROPE),
                accumulate=group > 0,
            )


def _gate_projection(x_hT, gate_chunks, total_rows, local_heads):
    """Return the FP32 PSUM gate projection ``[128, local_heads, Q*B]``.

    With the MXFP8 gate entry (``_load_mx_gate_cols``) the rows of ``x_hT`` are quantized here, per token, so a pass
    of the rows gets the bits of a whole-batch quantize (the decode's).
    """
    gate_psum = nl.ndarray((_P_MAX, local_heads, total_rows), dtype=nl.float32, buffer=nl.psum)
    if gate_chunks and gate_chunks[0][0] == "mx":
        _, gate_w, scales, sb, tag = gate_chunks[0]
        rows = total_rows + total_rows % 2
        x_data, x_scale = quantize_hT_mx(
            x_hT, f"{tag}_g{next(_GATE_CALLS)}", rows, lambda shape, dtype, name: sb.tile(shape, dtype, name)
        )
        w_x4 = gate_w.view(nl.float8_e4m3fn_x4)
        for local_head in range(local_heads):
            cols = slice(local_head * _V_HEAD, (local_head + 1) * _V_HEAD)
            for group in range(gate_w.shape[1]):
                last = nisa.nc_matmul_mx(
                    dst=gate_psum[0:_P_MAX, local_head, 0:total_rows],
                    stationary=w_x4[0:_P_MAX, group, cols],
                    moving=x_data[0:_P_MAX, group, 0:total_rows],
                    stationary_scale=mx_scale_view(scales, group, _QKV_LOCAL + cols.start, _V_HEAD),
                    moving_scale=x_scale[0:_P_MAX, group, 0:total_rows],
                    accumulate=group > 0,
                )
        object.__setattr__(gate_psum, "last_matmul", last)
        return gate_psum
    for local_head in range(local_heads):
        for chunk_start, chunk_tiles, gate_w in gate_chunks:
            for chunk_tile in range(chunk_tiles):
                hidden_tile = chunk_start + chunk_tile
                last = nisa.nc_matmul(
                    dst=gate_psum[0:_P_MAX, local_head, 0:total_rows],
                    stationary=gate_w[0:_P_MAX, chunk_tile, local_head * _V_HEAD : (local_head + 1) * _V_HEAD],
                    moving=x_hT[0:_P_MAX, 0:total_rows, hidden_tile],
                    accumulate=hidden_tile > 0,
                )
    object.__setattr__(gate_psum, "last_matmul", last)
    return gate_psum


def _value_projection(local_state, kv_b_w, value_all, total_rows, local_heads, absorb_width, sb, after=None):
    """Normalize the merged state and write ``latent @ value`` to ``value_all`` (``[128, local_heads, Q*B]``)."""
    bf16 = nl.bfloat16
    fp32 = nl.float32
    parts = local_state if isinstance(local_state, StateParts) else None
    inverse_sum = sb.tile((local_heads, total_rows, 1), fp32)
    sums = local_state[0:local_heads, 0:total_rows, _KV_LORA : _KV_LORA + 1] if parts is None else parts.sums
    nisa.reciprocal(dst=inverse_sum, data=sums)
    latent = sb.tile((local_heads, total_rows, _KV_LORA), bf16)
    nisa.tensor_tensor(
        dst=latent,
        data1=local_state[0:local_heads, 0:total_rows, 0:_KV_LORA] if parts is None else parts.o,
        data2=inverse_sum.broadcast(2, _KV_LORA),
        op=nl.multiply,
    )
    latent_psum = nl.ndarray((_P_MAX, _KV_LORA_TILES, total_rows, 4), dtype=bf16, buffer=nl.psum)
    for row in range(total_rows):
        for latent_tile in range(_KV_LORA_TILES):
            transpose = nisa.nc_transpose(
                dst=latent_psum[0:_P_MAX, latent_tile, row, 0:local_heads],
                data=latent[0:local_heads, row, latent_tile * _P_MAX : (latent_tile + 1) * _P_MAX],
            )
            if after is not None and (GATE_PIN_ALL_TILES or (row == 0 and latent_tile == 0)):
                _instruction(transpose).depends_on(after, kind=nisa.dependency.schedule)
    latent_t = sb.tile((_P_MAX, _KV_LORA_TILES, local_heads, total_rows), bf16)
    nisa.tensor_copy(
        dst=latent_t.permute((0, 1, 3, 2)), src=latent_psum[0:_P_MAX, 0:_KV_LORA_TILES, 0:total_rows, 0:local_heads]
    )
    _value_matmuls(latent_t, kv_b_w, value_all, total_rows, local_heads, absorb_width, sb)


def _value_matmuls(latent_t, kv_b_w, value_all, total_rows, local_heads, absorb_width, sb=None):
    """Write ``latent @ value`` of the transposed latent ``[128, 4, local_heads, Q*B]`` to ``value_all``.

    With the MXFP8 ``kv_b_w`` (``MxKvB``) the latent of each (head, row) is quantized (one hT_quad group, the 4
    latent tiles in the lanes) and each head is one MXFP8 matmul, the value weight stationary as the BF16 form.
    """
    value_psum = nl.ndarray((_P_MAX, local_heads, total_rows), dtype=nl.float32, buffer=nl.psum)
    if isinstance(kv_b_w, MxKvB):
        _value_mx(latent_t, kv_b_w, value_psum, total_rows, f"mla_v{next(_VALUE_CALLS)}", sb)
        nisa.tensor_copy(dst=value_all, src=value_psum)
        return
    for local_head in range(local_heads):
        for latent_tile in range(_KV_LORA_TILES):
            value_col = absorb_width + (latent_tile * local_heads + local_head) * _V_HEAD
            nisa.nc_matmul(
                dst=value_psum[0:_P_MAX, local_head, 0:total_rows],
                stationary=kv_b_w[0:_P_MAX, value_col : value_col + _V_HEAD],
                moving=latent_t[0:_P_MAX, latent_tile, local_head, 0:total_rows],
                accumulate=latent_tile > 0,
            )
    nisa.tensor_copy(dst=value_all, src=value_psum)


def _value_projection_dense(local_state, kv_b_w, value_all, total_rows, local_heads, absorb_width, sb, after=None):
    """``_value_projection`` on the dense state ``[local_heads * Q*B, 513]`` (row ``h * Q*B + r``); the same values."""
    bf16 = nl.bfloat16
    fp32 = nl.float32
    state_rows = local_heads * total_rows
    parts = local_state if isinstance(local_state, StateParts) else None
    inverse_sum = sb.tile((state_rows, 1), fp32)
    sums = local_state[0:state_rows, _KV_LORA : _KV_LORA + 1] if parts is None else parts.sums
    nisa.reciprocal(dst=inverse_sum, data=sums)
    latent = sb.tile((state_rows, _KV_LORA), bf16)
    nisa.tensor_tensor(
        dst=latent,
        data1=local_state[0:state_rows, 0:_KV_LORA] if parts is None else parts.o,
        data2=inverse_sum.broadcast(1, _KV_LORA),
        op=nl.multiply,
    )
    latent_psum = nl.ndarray((_P_MAX, _KV_LORA_TILES, state_rows), dtype=bf16, buffer=nl.psum)
    for latent_tile in range(_KV_LORA_TILES):
        transpose = nisa.nc_transpose(
            dst=latent_psum[0:_P_MAX, latent_tile, 0:state_rows],
            data=latent[0:state_rows, latent_tile * _P_MAX : (latent_tile + 1) * _P_MAX],
        )
        if after is not None and (GATE_PIN_ALL_TILES or latent_tile == 0):
            _instruction(transpose).depends_on(after, kind=nisa.dependency.schedule)
    latent_t = sb.tile((_P_MAX, _KV_LORA_TILES, local_heads, total_rows), bf16)
    nisa.tensor_copy(dst=latent_t.reshape((_P_MAX, _KV_LORA_TILES, state_rows)), src=latent_psum)
    _value_matmuls(latent_t, kv_b_w, value_all, total_rows, local_heads, absorb_width, sb)


def _validate_inputs(
    x_hT,
    mla_in_proj,
    mla_q_b_proj,
    mla_kv_b_proj,
    mla_o_proj,
    kv_pool,
    block_table,
    slot,
    position,
    rope_rows,
    lay,
    q_rows,
    output_gate,
    total_rows,
    tokens,
    max_tokens,
    mla_in_proj_scale=None,
    trimmed_window=False,
    mla_q_b_proj_scale=None,
    mla_kv_b_proj_scale=None,
    mla_o_proj_scale=None,
):
    """Check the configuration and the static shapes of the body inputs."""
    local_heads = lay.local_heads
    gate_local = local_heads * _V_HEAD
    absorb_width = local_heads * _KV_LORA
    in_width = _HIDDEN_TILES * (lay.qkv_local + (gate_local if output_gate else 0))
    kernel_assert(
        nl.program_ndim() == 0 or (nl.program_ndim() == 1 and nl.num_programs(0) == 1),
        "mla_block_attention supports LNC1 only",
    )
    kernel_assert(nisa.get_nc_version() >= nisa.nc_version.gen3, "mla_block_attention requires Trn2 or newer")
    kernel_assert(
        local_heads in ((1,) if lay.one_head else (2, 3)),
        f"local_heads must be 2 (drafter) or 3 (target) at CP32 and 1 at CP{lay.cp}, got {local_heads}",
    )
    kernel_assert(total_rows == q_rows * tokens, f"x_hT must have Q*B={q_rows * tokens} rows, got {total_rows}")
    kernel_assert(
        1 <= tokens <= max_tokens, f"mla_block_attention supports 1 to {max_tokens} sequences, got B={tokens}"
    )
    kernel_assert(
        tuple(x_hT.shape) == (_P_MAX, total_rows, _HIDDEN_TILES),
        f"x_hT must be [128, Q*B, 56], got {tuple(x_hT.shape)}",
    )
    kernel_assert(x_hT.dtype == nl.bfloat16, "x_hT must be BF16")
    if mla_in_proj_scale is None:
        kernel_assert(
            tuple(mla_in_proj.shape) == (_P_MAX, in_width),
            f"mla_in_proj must be [128, {in_width}], got {tuple(mla_in_proj.shape)}",
        )
    else:
        kernel_assert(
            is_trn3() and not lay.one_head and local_heads == 3 and output_gate,
            "the MXFP8 mla_in_proj needs Trn3, CP32, 3 local heads and the output gate (the target verify)",
        )
        data_shape, scale_shape = mx_hT_shapes(_HIDDEN, _IN_WIDTH)
        kernel_assert(
            tuple(mla_in_proj.shape) == data_shape and tuple(mla_in_proj_scale.shape) == scale_shape,
            f"the MXFP8 mla_in_proj / scales must be {data_shape} / {scale_shape}, got "
            f"{tuple(mla_in_proj.shape)} / {tuple(mla_in_proj_scale.shape)}",
        )
    if mla_q_b_proj_scale is None:
        q_b_row = _Q_LORA_TILES * local_heads * _Q_HEAD
        kernel_assert(
            tuple(mla_q_b_proj.shape) == (_P_MAX, q_b_row),
            f"mla_q_b_proj must be [128, {q_b_row}], got {tuple(mla_q_b_proj.shape)}",
        )
    else:
        kernel_assert(
            is_trn3() and not lay.one_head and local_heads == 3,
            "the MXFP8 mla_q_b_proj needs Trn3, CP32 and 3 local heads (the target verify / its decode twin)",
        )
        q_b_data_shape, q_b_scale_shape = mx_hT_shapes(_Q_LORA, local_heads * _Q_HEAD)
        kernel_assert(
            tuple(mla_q_b_proj.shape) == q_b_data_shape and tuple(mla_q_b_proj_scale.shape) == q_b_scale_shape,
            f"the MXFP8 mla_q_b_proj / scales must be {q_b_data_shape} / {q_b_scale_shape}, got "
            f"{tuple(mla_q_b_proj.shape)} / {tuple(mla_q_b_proj_scale.shape)}",
        )
    if mla_kv_b_proj_scale is None:
        kernel_assert(
            tuple(mla_kv_b_proj.shape) == (_P_MAX, 2 * absorb_width),
            f"mla_kv_b_proj must be [128, {2 * absorb_width}], got {tuple(mla_kv_b_proj.shape)}",
        )
    else:
        kernel_assert(
            is_trn3() and not lay.one_head and local_heads == 3,
            "the MXFP8 mla_kv_b_proj needs Trn3, CP32 and 3 local heads (the target verify / its decode twin)",
        )
        kv_b_cols = _KV_LORA + gate_local
        kernel_assert(
            tuple(mla_kv_b_proj.shape) == (_P_MAX, MX_LANES * kv_b_cols)
            and tuple(mla_kv_b_proj_scale.shape) == (MX_COMPACT_ROWS, kv_b_cols)
            and local_heads * MX_QUADRANT_ROWS <= _P_MAX,
            f"the MXFP8 mla_kv_b_proj / scales must be [128, {MX_LANES * kv_b_cols}] / [{MX_COMPACT_ROWS}, "
            f"{kv_b_cols}], got {tuple(mla_kv_b_proj.shape)} / {tuple(mla_kv_b_proj_scale.shape)}",
        )
    if mla_o_proj_scale is None:
        kernel_assert(
            tuple(mla_o_proj.shape) == (gate_local, _HIDDEN),
            f"mla_o_proj must be [{gate_local}, {_HIDDEN}], got {tuple(mla_o_proj.shape)}",
        )
    else:
        kernel_assert(
            is_trn3() and not lay.one_head and local_heads == 3 and output_gate,
            "the MXFP8 mla_o_proj needs Trn3, CP32, 3 local heads and the output gate (the target verify)",
        )
        o_data_shape, _ = mx_hT_shapes(MX_LANES * _V_HEAD, _HIDDEN)
        kernel_assert(
            tuple(mla_o_proj.shape) == o_data_shape and tuple(mla_o_proj_scale.shape) == (MX_COMPACT_ROWS, _HIDDEN),
            f"the MXFP8 mla_o_proj / scales must be {o_data_shape} / [{MX_COMPACT_ROWS}, {_HIDDEN}], got "
            f"{tuple(mla_o_proj.shape)} / {tuple(mla_o_proj_scale.shape)}",
        )
    kernel_assert(
        len(kv_pool.shape) == 2 and kv_pool.shape[1] == _PAGE_WIDTH,
        f"kv_pool must be [P, {_PAGE_WIDTH}], got {tuple(kv_pool.shape)}",
    )
    kernel_assert(
        kv_pool.dtype == nl.bfloat16 or kv_pool.dtype == fp8_page_dtype(),
        f"kv_pool must be BF16 or {fp8_page_dtype()}, got {kv_pool.dtype}",
    )
    kernel_assert(
        len(block_table.shape) == 2 and block_table.shape[0] == tokens,
        f"block_table must be [B, NB], got {tuple(block_table.shape)}",
    )
    kernel_assert(
        block_table.dtype == nl.int32,
        f"block_table must be INT32 page ids (the MLA loops gather its rows with DMAs; an INT64 table gives a "
        f"64-bit DMA dtype, which the ISA does not allow), got {block_table.dtype}",
    )
    local_blocks = block_table.shape[1]
    kernel_assert(
        1 <= local_blocks <= _MAX_STATIC_BLOCKS
        or local_blocks in (8, 16, 32, 64, 128, 256)
        or (trimmed_window and local_blocks == _TRIMMED_WINDOW_BLOCKS),
        f"block_table must have 1 to 4 local blocks, or a power of two of 8 to 256 (or {_TRIMMED_WINDOW_BLOCKS} with "
        f"trimmed_window), got {local_blocks}",
    )
    kernel_assert(tuple(slot.shape) == (total_rows, 1), f"slot must be [Q*B, 1], got {tuple(slot.shape)}")
    kernel_assert(tuple(position.shape) == (tokens, 1), f"position must be [B, 1], got {tuple(position.shape)}")
    if rope_rows is not None:
        kernel_assert(
            tuple(rope_rows.shape) == (total_rows, 2 * _ROPE),
            f"rope_rows must be [Q*B, {2 * _ROPE}], got {tuple(rope_rows.shape)}",
        )
