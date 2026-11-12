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

"""MLA decode attention body of the Kimi-K3 decode giga-kernel (TP32 = CP32, LNC1, B sequences).

The body runs the projections, the TP gathers, the cache writes, the paged attention over the local
cache rows, the CP merge, the value and output projections, and the TP all-reduce of one MLA layer
for the B sequences of one DP replica, one token per sequence.
"""

import functools
import itertools
import math
from contextlib import nullcontext
from typing import NamedTuple

import nki.collectives as ncc
import nki.isa as nisa
import nki.language as nl

from .utils.kernel_assert import kernel_assert
from .common import (
    StateParts,
    fp8_kernel_pages,
    fp8_page_dtype,
    fp8_page_max,
    is_trn3,
    load_state_bf16_o,
    mla_state_bf16_o_views,
    state_rows_plain,
    store_state_bf16_o,
)
from .tkg_weight_dma_gate import gate_weight_dma, pool_after, release_weight_gate
from .tkg_sbuf_alloc import AUTO_SBUF, SB_TILE_ALIGN, SbufTiles
from .tkg_mx_quant import (
    MX_COMPACT_ROWS,
    MX_LANES,
    MX_QUADRANT_ROWS,
    load_mx_compact_scales,
    load_mx_scales,
    mx_hT_shapes,
    mx_scale_view,
    quadrant_absorb_mx,
    quantize_heads_mx,
    quantize_hT_mx,
    quantize_quadrant_rows_mx,
)
from .tkg_kda_state_pairs import load_pair_page, pair_key_tiles, pair_numerator, pair_rows_tile, write_pair_rows

MLA_WEIGHT_NAMES = (
    "mla_in_proj",
    "mla_q_a_norm",
    "mla_q_b_proj",
    "mla_kv_a_norm",
    "mla_kv_b_proj",
    "mla_o_proj",
)
"""Per-layer kernel weights of the MLA body, in argument order. All are BF16.

- ``mla_in_proj`` ``[128, 25200]``: the fused input projection of TP rank ``t`` in SBUF order, so each
  partition loads long contiguous runs. Row ``p`` holds the qkv region, element ``j * 66 + c`` = weight
  ``[in = 128 j + p, out = c]`` for the 66 qkv columns, then the gate region, element
  ``3696 + j * 384 + g`` = weight ``[128 j + p, 66 + g]``. The qkv columns are the q_a columns
  ``48t .. 48t+47`` and 18 kv_a columns: kv_a column ``c`` is kv_a feature ``32c + t``, so a strided view
  of the gathered kv_a is in feature order. The 384 gate columns are the local g_proj columns.
- ``mla_q_a_norm`` ``[1536]``: the q_a_layernorm weight in hT order (element ``p * 12 + k`` is weight
  ``k * 128 + p``).
- ``mla_q_b_proj`` ``[128, 6912]``: the local q_b projection in SBUF order: element ``(p, k * 576 + o)`` is
  weight ``[in = 128 k + p, out = o]``. Output ``o = h * 192 + d`` is dim ``d`` (nope 0..127, rope 128..191)
  of local head ``h``.
- ``mla_kv_a_norm`` ``[512]``: the kv_a_layernorm weight.
- ``mla_kv_b_proj`` ``[128, 3072]``: the local kv_b projection, arranged for the two matmuls. Columns
  ``0..1535`` are the absorb weight: element ``(d, h * 512 + c)`` is ``key[h, d, c]``. Columns
  ``1536..3071`` are the value weight: element ``(p, 1536 + (k * 3 + h) * 128 + v)`` is
  ``value[h, v, k * 128 + p]``.
- ``mla_o_proj`` ``[384, 7168]``: the local o_proj rows ``[in, out]``; row ``h * 128 + v`` is dim ``v`` of
  local head ``h``.
"""

MLA_MX_SCALES = (
    ("mla_in_proj_scale", 0),
    ("mla_q_b_proj_scale", 2),
    ("mla_kv_b_proj_scale", 4),
    ("mla_o_proj_scale", 5),
)
"""The MXFP8 scale keys of the MLA body with the index of their weight in ``MLA_WEIGHT_NAMES``, in the order of the
scale stacks that the models append after the six weights (one for each class whose weight is MXFP8)."""

_P_MAX = 128
_HIDDEN = 7168
_HIDDEN_TILES = _HIDDEN // _P_MAX
_TP = 32
_LOCAL_HEADS = 3
_HEADS = _TP * _LOCAL_HEADS
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
_GATE_LOCAL = _LOCAL_HEADS * _V_HEAD
_IN_WIDTH = _QKV_LOCAL + _GATE_LOCAL
_BLOCK = 128
_GROUP_BLOCKS = 4
"""Local blocks of one key group: the scores of a group fill one FP32 PSUM bank."""
_MAX_STATIC_BLOCKS = _GROUP_BLOCKS
"""Local blocks of the static path, one key group per sequence."""
_MAX_UNROLLED_BLOCKS = 16
"""Local blocks of the unrolled path, at most (a 64K context). A larger table takes the loop path. The
loops split the kernel into basic blocks, and neuronx-cc computes the SB liveness of a tile inside each
basic block. Another tile can then overwrite a tile that lives across a block boundary at runtime, or
take the pinned DGE scratch and fail the compile with NCC_INLA001. The loop path is thus correct only
with explicit addresses for those tiles (``sbuf``, see ``tkg_sbuf_alloc``)."""
_FP8_TRANSPOSE_STEP = 2
"""PSUM element step of the output of an FP8 TensorE transpose in transpose mode."""
_BF16_KEY_MAX = 65536.0
"""Clamp bound of a BF16 cache row. A masked row of a reused block or a null page can hold stale bytes
that read as +-Inf or a value up to about 3.4e38 in BF16. A key of +-Inf makes the masked product
``0 * Inf`` a NaN, and a huge key overflows the score, so the score cap turns a masked key into the cap
weight. The clamp keeps every normalized cache value (far below the bound) and makes a stale key finite,
so a masked key weighs 0 and adds 0. The score of a clamped key stays far below the mask bias."""
_PAGE_WIDTH = _BLOCK * _CACHE_WIDTH
_GLOBAL_BLOCK_STRIDE = _TP * _BLOCK
"""Global positions between two local blocks of one rank."""
_LOOP_GROUPS = 8
"""Key groups of one iteration of the dynamic path, at most: the page loads of the later groups overlap
the matmuls of the earlier ones. Each iteration ends with a barrier of all engines, so a larger iteration
hides more DMA latency and has fewer barriers. A table of 8 blocks takes 2 groups per iteration."""
_BF16_LOOP_GROUPS = 4
"""Key groups of one iteration with BF16 pages: 8 groups do not fit the explicit SBUF stack of a 1M B=4 block."""
LOOP_RESTARTS = ("off", "table")
"""Forms of the iteration restart of the per-sequence key loop (``mla_decode_attention`` ``loop_restart``)."""
LOOP_GROUP_COUNTS = (0, 16)
"""Key groups per iteration of the per-sequence key loop (``mla_decode_attention`` ``loop_groups``); 0 keeps
``_LOOP_GROUPS`` / ``_BF16_LOOP_GROUPS``."""
O_PROJ_RINGS = (0, 2, 4)
"""o_proj weight chunks in flight with ``o_proj_stream`` (``mla_decode_attention`` ``o_proj_ring``); 0 keeps one chunk
at a time."""
LOOP_PAGE_GATHERS = ("off", "group", "carry", "carry_pool")
"""Forms of the page loads of the per-sequence key loop (``mla_decode_attention`` ``loop_page_gather``): "off" loads each
page with its own DMA on the SP queue; "group" loads the 4 pages of each key group with one GpSimd software DGE gather;
"carry" also gathers group 0 of the next iteration inside each iteration; "carry_pool" is "carry" with the gather indices
of the next iteration built on GpSimd."""
_GATHER_LEAD = 2
"""Key groups between the gather of a group and its first use (``loop_page_gather``): the gather of group g + 2 issues
before the casts of group g."""
_GATHER_CARRY_GROUP = 1
"""With ``loop_page_gather`` "carry", the gather indices of the next iteration build, and its group 0 gathers, after the
softmax update of this group: group 0's page tile has no reader left in the iteration by then."""
_GATHER_ROW_BITS = 7
"""log2 of the rows of a page: the gather index of row p of page e is (e << 7) + p."""
_KEY_SETS = 4
"""Transposed-key tiles of the dynamic path, at most. Group g of an iteration uses set g % 4, so 8 groups
need the SBUF of 8 row tiles and only 4 key tiles."""
_DEEP_WEIGHTS = 4
"""Weight tiles of the packed loop with ``deep_pipe`` (the weights of sub-slot s live until its transposes in s + 2)."""
_ACT_SPREAD_ITEMS = 4
"""``_packed_key_loop`` ``act_spread``: the fewest items of a loop that takes the ScalarE order (MEASURED on trn3, 1-core: a 2-item
block loop is slower with it, loops of 8 or more items gain time on each item)."""
_DEEP_SUMS = 8
"""Weight-sum tiles of the packed loop with ``deep_pipe`` (the sum of sub-slot s lives until its add after cap s + 4)."""
_SUM_BATCH_RING = 2
"""``sum_batch``: the dense weight-sum tiles of the items, a ring of this many items (an item's batched sum add runs
before the exps of the item two later rewrite its tile; every loop's item count is even, so item -1 of a carried
iteration is the last item of the one before)."""
_GROUP_STRIDE = _GROUP_BLOCKS * _GLOBAL_BLOCK_STRIDE
"""Global positions between the starts of two key groups of one rank."""
_ABSORB_WIDTH = _LOCAL_HEADS * _KV_LORA
LORA_NORM_EPS = 1e-5
"""Default epsilon of the q_a and kv_a RMSNorms (``lora_norm_eps`` of ``mla_decode_attention``)."""
_PACK_SLOTS_SMALL = 8
"""Key groups (items) of one iteration of the packed loop for a table below _PACK_WIDE_BLOCKS local blocks."""
_PACK_SLOTS_WIDE = 16
"""Key groups of one iteration for a table of _PACK_WIDE_BLOCKS local blocks or more (256K and up) and at least
_PACK_WIDE_TOKENS sequences: half the iteration barriers, at most 15 null items."""
_PACK_WIDE_BLOCKS = 64
_PACK_WIDE_TOKENS = 4
"""Below this batch the 16-item iteration is slower than two 8-item ones (B=1 at 256K, in each layer)."""
_PACK_BUFFERS = 8
"""Page buffers of the packed loop; slot j of an iteration uses buffer j % _PACK_BUFFERS."""
_PACK_CARRY_PREFETCH_LOOKAHEAD = 4
"""The most prefetched slots of a carried packed loop with ``pack_lookahead`` 2 (2 with ``pack_lookahead`` 1)."""
_PACK_LOOKAHEADS = (1, 2)
"""Slots ahead of the current one whose slot row and pages a packed slot loads (``pack_lookahead``)."""
_ITEM_FIELDS = 8
"""Fields of an item row of the packed loop: 4 page ids, the sequence, the mask threshold, 2 unused."""
_TI_GROUP = 16
"""Partitions of one VectorE Tensor Indirection offset group: offset i sits at [i % 16, i // 16]."""
_STATE_WIDTH = _KV_LORA + 1
_TI_COLS = -(-_STATE_WIDTH // _TI_GROUP)
_STAGE_REF_WORD = (_KV_LORA_TILES + 1) * _HEADS // 2
"""FP32 word of the negative reference in a merged slot row; the cap follows it."""
_STAGE_TI = (_KV_LORA_TILES + 1) * _HEADS + 4
"""uint16 column of the TI offsets in a merged slot row."""
_STAGE_WIDTH = _STAGE_TI + _TI_COLS + 3
"""BF16 columns of a merged slot row (q, rope, 2 FP32, 33 uint16, pad to an even count)."""
_P2_STAGE_REF_WORD = (_KV_LORA_TILES + 1) * _P_MAX // 2
"""``pack_p2``: FP32 word of the negative reference in a stage row of 128 (row, head) columns; the cap follows it."""
_P2_STAGE_TI = (_KV_LORA_TILES + 1) * _P_MAX + 4
"""``pack_p2``: uint16 column of the TI offsets in a stage row of 128 (row, head) columns."""
_P2_STAGE_WIDTH = _P2_STAGE_TI + _TI_COLS + 3
"""``pack_p2``: BF16 columns of a stage row of 128 (row, head) columns (q 4 x 128, rope 128, 2 FP32, 33 uint16, pad)."""
_P2_STAGE_FILL_TEST = None
"""TEST ONLY: a value that ``_stage_query_rows_p2`` writes over its whole stage tile first (None, the default: no write),
so a test can check that every byte of the stage rows is written again (``p2_entry_act`` zero-fills only the bytes that
no copy writes)."""
CLAMP_BLOCK_ONCE = True
"""``_packed_key_loop`` with ``clamp_block``: the block's loop always runs one iteration (``_pack_items``: the bounds
``[R / clamp_block, R / clamp_block + 1)``), so its body runs once in place of the loop, without the loads of a next
iteration (the item row after the block is the padding of that prefetch, and nothing reads the slot rows and pages it
would load). The values do not change. False keeps the loop (the switch for a test)."""
_SOFTMAX_SCALE = _Q_HEAD**-0.5
_MASK_BIAS = -1.0e30
"""Score bias of a masked key. The exponential of a biased score is exactly 0."""
_CURRENT_WEIGHT_LOG2 = -93
_MERGE_OFFSET = -_CURRENT_WEIGHT_LOG2 * math.log(2.0)
"""Offset of the softmax reference below the scaled current score. Every key weighs
``exp(scaled score - scaled current score - 64.46)``, so the current key weighs exactly ``2**-93``. A key
22.8 below the current key underflows to 0, which leaves out less than 1.3e-10 of the current weight."""
_WEIGHT_CAP = 60.0
"""Upper bound of the exponent of a key weight. A key whose scaled score exceeds the scaled current score
by more than 124.46 gets the weight e**60, so a weight, a sum, and a numerator stay finite."""
_RANK_CURRENT_WEIGHT = 2.0 ** (_CURRENT_WEIGHT_LOG2 - 5)
"""The weight of the current key that each of the 32 ranks adds. The CP reduce-scatter sums it to
``2**-93``. A power of two is exact in BF16, so the numerator and the sum add the same weight."""
_QUADRANT = 32
"""Partition quadrant of the engines. An operand of fewer than 32 partitions starts at a multiple of it."""
_BIAS_ROW = _ROPE
"""Contraction row of the rope score matmul that adds the key mask; the rope query holds 1.0 in this row."""
_GATHER_WIDTH = _ABSORB_WIDTH + _LOCAL_HEADS * _ROPE
"""Width of one query gather row: the absorbed queries of 3 heads, then their rope queries."""
QUERY_SET_TOKENS = _P_MAX // _TP
"""Sequences of one query set: the query gather of 32 ranks of 4 rows fills the 128 partitions."""
MAX_TOKENS = 2 * QUERY_SET_TOKENS
"""Sequences of the DP replica: two query sets."""
_QKV_CHUNK_TILES = 28
"""Hidden tiles of one qkv weight DMA: 3.7 KB per partition."""
_GATE_CHUNK_TILES = 8
"""Hidden tiles of one gate weight DMA: 6 KB per partition."""
_WGATE_EARLY_FORMS = ("off", "sync", "scalar")
"""Forms of ``wgate_early``: where the gate weight DMAs issue."""
_MX_QUERY_FORMS = ("off", "emulate")
"""Forms of ``mx_query``: the BF16 absorbed query, or its MXFP8 values (an accuracy form)."""
WGATE_EARLY_FORMS = _WGATE_EARLY_FORMS
"""Forms of ``mla_decode_attention`` ``wgate_early`` (the entries' ``mla_wgate_early``)."""
_ROW_DMA_POOL_PERIODS = (0, 2, 3, 4)
"""Periods of ``pack_row_dma_pool``: 0 keeps every slot row DMA on SP; k moves that of every k-th slot to GpSimd."""
_QKV_REGION = _HIDDEN_TILES * _QKV_LOCAL
"""Elements of the qkv region of one row of mla_in_proj."""
_IN_PROJ_ROW = _HIDDEN_TILES * _IN_WIDTH
_O_PROJ_CHUNK_TILES = 14
"""Hidden tiles of one output-projection weight DMA."""
_PSUM_BANK_FP32 = 512
_PSUM_BANK_BYTES = 2048
"""Bytes of one PSUM bank in a partition."""
_MX_PACK = 4
"""FP8 values in one element of an MX matmul operand (``float8_e4m3fn_x4``)."""
_MX_UNIT_SCALE = 127
"""E8M0 exponent of the unit scale of an MX matmul."""
_MX_KEY_FORMS = (None, "single", "split")
"""Forms of ``pack_mx_keys``: BF16 queries (the default program), one e4m3 query, or an e4m3 query and its e4m3 rest."""
_MX_QK_FORMS = (False, "unit", "block32")
"""Forms of ``pack_mx_qk``: BF16 latent queries (the default program), the e4m3 query at unit scales (the GPU decode's
query), or MXFP8 with one E8M0 scale a 32-value block (``nisa.quantize_mx``, the flash prefill's ``fp8_query`` "mx")."""
_P2_STAGE_MX_SCALE = 2 * _P_MAX
"""``pack_p2`` with ``pack_mx_qk`` "block32": the BF16 column of a stage row where the 128 E8M0 scale bytes of its
columns start (after the 128 FP32 MX query words, in the rest of the BF16 latent query region)."""
_E4M3_MAX = 448.0
"""Largest finite float8_e4m3fn value: the saturation bound of the e4m3 queries of ``pack_mx_keys``."""
_MX_SCALE_PLAN_SLOT = 4
"""Plan tile slot of the unit MX scales (``_plan_tile``); the plan of the packed loop takes slots 0 to 3."""
_VOG5_PLAN_SLOTS = (5, 6, 7)
"""The plan tile slots of the pass-invariant constants of ``kv_group`` (slots 5 and 7 unused; slot 6 the group ``sel``
and ``const``), built once a staged layer."""
_DR_BLOCKS = 2
"""256-feature blocks of the latent keys of ``pack_dr_keys``: the 2 rows of a double_row matmul."""
_DR_BLOCK_WORDS = _KV_LORA // (2 * _DR_BLOCKS)
"""2-byte words of one block of a key (2 FP8 features each): one 2-byte transpose of a page per block."""
_DR_QUERY_WORDS = 2 * _DR_BLOCKS * _HEADS // 2
"""2-byte words of the e4m3 double_row query stationaries of one sequence on each partition: [byte, block, head]."""


def mla_runs_loops(local_blocks: int) -> bool:
    """Return True when ``mla_decode_attention`` runs its key loops (``nl.fori_loop``) for ``local_blocks``.

    The loop path runs one loop per sequence above ``_MAX_UNROLLED_BLOCKS`` local blocks. The loops split
    the kernel into basic blocks.

    Example:
        >>> mla_runs_loops(256)
        True
    """
    return local_blocks > _MAX_UNROLLED_BLOCKS


def _mla_decode_body(
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
    exact_softmax: bool = False,
    bf16_state: bool = False,
    loop_path: bool = False,
    fp8_pages: bool = False,
    sbuf: SbufTiles = None,
    sources: dict | None = None,
    packed_loop: bool = False,
    pack_plan: dict | None = None,
    pack_slots: int | None = None,
    pack_tail_slots: int | tuple | None = None,
    pack_tail_round: bool = False,
    zeroed_pages: bool = False,
    loop_cast_engines: str = "off",
    loop_pad_stationary: bool = False,
    lora_norm_eps: float = LORA_NORM_EPS,
    pack_sbuf: SbufTiles = None,
    pair_pages: bool = False,
    batched_cache_write: bool = False,
    pack_prefetch: int = 1,
    o_proj_stream: bool = False,
    pack_row_dma_act: bool = False,
    pack_page_runs: int = 0,
    pack_null_copy_vector: bool = False,
    pack_null_runs: bool = False,
    wgate_early: str = "off",
    pack_key_transpose: str = "matmul",
    pack_latent_act: int = 2,
    pack_mask_column: bool = False,
    pack_bg_rope: bool = False,
    pack_bg_interleave: bool = False,
    pack_carry: bool = False,
    pack_carry_main_only: bool = False,
    pack_sum_after_copy: bool = False,
    pack_pair: dict | None = None,
    pack_setup_ahead: bool = False,
    pack_state_store_queues: int = 1,
    pack_current_after_rs: bool = False,
    stage_post: bool = False,
    stage_pre: bool = False,
    pack_mx_keys: str | None = None,
    pack_rope_ahead: bool = False,
    pack_dr_keys: bool = False,
    mla_in_proj_scale: nl.NkiTensor = None,
    mla_q_b_proj_scale: nl.NkiTensor = None,
    mla_kv_b_proj_scale: nl.NkiTensor = None,
    mla_o_proj_scale: nl.NkiTensor = None,
    mx_query: str = "off",
    pack_row_dma_pool: int = 0,
    pack_lookahead: int = 1,
    head_prefetch=None,
    pack_psum_per_block: bool = False,
    loop_restart: str = "off",
    pack_step_items: dict | None = None,
    loop_groups: int = 0,
    o_proj_ring: int = 0,
    loop_page_gather: str = "off",
    pack_stage_rows_sbuf: bool = False,
    o_proj_preload: bool = False,
    o_proj_early: bool = False,
    loop_inputs_ahead: bool = False,
    loop_dr_keys: bool = False,
    state_bf16_o: bool = False,
    pack_mask_pool: bool = False,
) -> nl.NkiTensor:
    """
    Run one Kimi-K3 MLA decode attention layer for B sequences on one TP32 rank.

    The rank owns heads ``3t .. 3t+2`` and the CP32 cache rows of the global blocks ``b`` with
    ``b % 32 == t``. Local block ``j`` of a sequence holds its global block ``32j + t``.

    Dimensions:
        B: sequences of the DP replica, one token each, 1 to 8. More than 4 sequences run as query sets of 4.
        NB: local blocks of the block table: 1 to 4 (16K context), or a power of two from 8 to 256 (1M).
            More than 16 (64K) needs ``loop_path=True``.
        H: 7168 hidden size.
        Nh: 96 gathered heads (3 local heads on each of 32 ranks).
        C: 512 latent features. R: 64 rope features.
        P: pages of the cache pool.

    Args:
        x_hT (nl.NkiTensor): [128, B, 56] BF16 SBUF normalized hidden in token-major hT layout:
            element ``(p, b, j)`` is ``x[b, j * 128 + p]``.
        mla_in_proj, mla_q_a_norm, mla_q_b_proj, mla_kv_a_norm, mla_kv_b_proj, mla_o_proj
            (nl.NkiTensor): BF16 HBM weights of the layer, see ``MLA_WEIGHT_NAMES``.
        kv_pool (nl.NkiTensor): [P, 73728] BF16 or FLOAT8_E4M3FN HBM page pool. An MLA page is 128 rows
            of ``[normalized latent 512 | k_pe 64]``. Page 0 is the null page. An FP8 pool holds the rows
            with a per-tensor scale of 1.0: the body writes the current row clamped to +-448 and rounded to
            nearest even, and reads the rows unscaled. The keys upcast exactly to BF16 on TensorE, so the
            query, the softmax weights and the accumulation are those of a BF16 pool.
        block_table (nl.NkiTensor): [B, NB] INT32 HBM local block table. A block without a page holds 0.
        slot (nl.NkiTensor): [B, 1] INT32 HBM slot ``page * 128 + offset`` of the current row on this rank,
            0 when the rank does not own the row.
        position (nl.NkiTensor): [B, 1] INT32 HBM global position of the current token.
        rank_info (nl.NkiTensor): [1, 4] INT32 HBM (global rank, DP rank, TP rank, 0).
        tp_group (ncc.ReplicaGroup): the TP32 groups, one list of 32 ranks per DP replica.
        tag (str): unique per-layer name prefix of the collectives and DMAs.
        exact_softmax (bool): use the exact maximum of the scores of each head as the softmax reference.
            This adds a [1, 96 B] maximum all-reduce, and an online softmax over the key groups. The
            default uses the capped reference of the current key (see Notes). 1 to 4 sequences only.
        bf16_state (bool): reduce-scatter the softmax state in BF16, which halves the bytes of the CP
            merge. The default reduces an FP32 state. On trn3 at B=4 a BF16 state saves time in each
            layer, but the relative L2 error grows from 5.4e-3 to 7.0e-3 per
            layer, and over 4 layers of the test it reaches 6.8e-2 against 5.1e-2.
        state_bf16_o (bool): the state reduce-scatter of the FP32 HBM pair (``sources`` ``state_hbm``) carries the
            numerators in BF16 and the sums in FP32 (``common.mla_state_bf16_o_views``): two reduce-scatters a
            layer, issued together, about half the bytes, as ``mla_block_attention`` ``state_bf16_o`` (its verify
            twin). VectorE casts the numerators before the store; the output changes. Not with ``bf16_state``,
            ``pack_pair``, ``stage_post`` or the staged groups. False (default) keeps the program.
        fp8_pages (bool): the pool holds FP8 pages, but it comes as BF16 ``[P', 73728]`` (147456 bytes
            per page), for example a buffer that the KDA layers share. The body then views it, with no
            copy, as an FP8 pool ``[2P', 73728]`` of 128-row pages, which the block table and the slot
            index. A BF16 pool of 512-row pages ``[P', 147456]`` (the pages of an fp32 KDA state) gives
            ``[4P', 73728]`` (``common.fp8_kernel_pages``). An FLOAT8_E4M3FN pool needs no flag.
        zeroed_pages (bool): the loop path reads its BF16 or FP8 cache pages with no clamp
            (``_sanitize_cache_rows``). The static and the unrolled paths keep the clamp. Set it only
            when every row that the loop path can read is finite: each MLA cache writer clamps a valid
            row to +-fp8_page_max(), the cache zeroes each MLA page when it allocates it, and an invalid table entry
            points to a page of the same sequence. The result is then the same as with the clamp.
        loop_cast_engines (str): the engines of the FP8 -> BF16 copies of the page tiles before their transposes in the
            per-sequence key loops of the dynamic path (Trn2 FP8 pages; 20 a key group: 16 latent tiles and 4 rope
            tiles), one of ``_LOOP_CAST_FORMS``. "off" (default) leaves them to the compiler, which keeps the program
            (the compiler puts all of them on VectorE, the engine that paces the loop). The other forms set
            ``engine=`` on each copy: "dve" all on VectorE, "split86" ScalarE 8 / VectorE 6 / GpSimd 6 a group (the
            split the compiler picks itself), "pool8" ScalarE 4 / VectorE 8 / GpSimd 8, "pool12" VectorE 8 /
            GpSimd 12, "act12" ScalarE 12 (latent tiles 1-3) / VectorE 8, "act8" ScalarE 8 (latent tiles 0-1) / VectorE 4
            (the rope tiles) / GpSimd 8. Each is an exact copy, so the output is
            bitwise the same. The page forms (``_LOOP_PAGE_CASTS``) leave no per-tile copy: the transposes and the
            value sums read BF16 pages. "dma" / "dma6" / "dma4": the page loads write BF16 page tiles (the DMA casts
            the FP8 bytes exactly; one tile per group of an iteration, or a ring of 6 / 4 that the groups take in turn;
            with the carried gathers of ``loop_page_gather`` group 0 keeps its own tile and the ring holds the others),
            and the page clamp keeps the FP8 bound. "page2" / "page3": one VectorE op
            per page (the clamp when the loop clamps, else a copy) writes the page into a ring of 2 / 3 BF16 group
            tiles. A clamped NaN byte gives the same bound as with the FP8 pages, so these are bitwise the same
            too. Not with ``packed_loop``, ``pair_pages`` or (the page forms) ``loop_dr_keys``; the static path has
            no key loops.
        loop_pad_stationary (bool): the QK, rope and P@V matmuls of the per-sequence key loops of the dynamic path (Trn2)
            take 128-column stationaries in place of the 96 heads: the 96 heads and the next 32 columns of their tile
            (the next latent tile or sequence of the queries, the next block of the transposed weights; each tile has
            one spare slot at its end, zeroed: the query slot once, the 32 read columns of the weight block by GpSimd
            in each group). TensorE loads a 128-column BF16 stationary on its fast path. Rows 96-127 of the
            scores and the numerator PSUM tiles are never read, and each output row depends on its own column only,
            so the output is bitwise the same. Not with ``exact_softmax``, ``packed_loop`` or ``pair_pages``. False
            (default) keeps the program.
        loop_path (bool): allow a block table of more than 16 local blocks, which takes the loop path.
            WARNING: some neuronx-cc builds can place another SBUF tile over a tile that lives
            through the loops, and the output is then wrong with no compile error. Give ``sbuf`` with
            it (see ``tkg_sbuf_alloc``), or set it only for a kernel that you check on hardware.
        sbuf (SbufTiles): the allocator of the SBUF tiles of the body. With a region
            (``SbufTiles(region)``), every SBUF tile of the body takes a stack address in the innermost
            open frame of the region, and ``y_hT`` too, so the caller closes that frame after it reads
            ``y_hT``. None lets the compiler place the tiles.
        sources (dict): None, or the persistent SBUF tiles of the gathers (``mla_collective_sources``). The
            values do not change.
        packed_loop (bool): the loop path runs one loop over the key groups of all sequences, which mix in
            iterations of 8 groups (``_packed_key_loop``), not one loop per sequence. The output is bitwise
            the same. Capped softmax only.
        pack_plan (dict): None, or the items of the step from ``mla_pack_plan`` (the same for every MLA
            layer), so the layer builds only its page rows.
        pack_slots (int): items of one packed iteration (``pack_slots``: a power of two up to 64; 16 at 256K+ tables
            with B >= 4, else 8). A plan fixes it.
        pack_tail_slots (int | tuple): None, or the items of one iteration of a second (tail) packed loop: the main
            loop then runs only full iterations and the tail loop the rest of the items, so fewer slots hold null
            items. A power of two up to ``pack_slots``. A pair ``(t1, t2)`` of powers of two with ``pack_slots`` >
            t1 > t2 (for example 64 slots with (32, 8)) adds a third loop: the first tail loop then runs only full
            iterations of t1 items and the second the rest, so fewer iterations pay the boundary cost; more counts
            (largest first) add more loops in the same way. Not with ``pack_plan``, which carries its own
            (``mla_pack_plan``). The output is bitwise the same.
        pack_tail_round (bool): with ``pack_tail_slots``, round the item count up to whole iterations of the last
            tail loop before it splits over the loops (``_pack_items(tail_round=True)``), so no tail loop runs as
            many iterations as one iteration of the loop before it holds. Not with ``pack_plan``. Bitwise the same.
        lora_norm_eps (float): the epsilon of the q_a and kv_a RMSNorms, which also make the cache rows that
            the body writes. The default is 1e-5 (``LORA_NORM_EPS``); the KimiRMSNorm of the checkpoint uses
            1e-6.
        pair_pages (bool): the FP8 pool holds each page in the token-pair layout (``pair_pages``): row pair
            ``(2 j, 2 j + 1)`` as ``[576, 2]``. The body takes the keys with 2-byte TensorE transposes, runs P@V
            by token parity, and writes the current row into its byte lane. The keys and the stored FP8 values
            are those of a row-major pool; the fp32 sum order of P@V differs.
        batched_cache_write (bool): write the current rows with one vector-indirect DMA (SWDGE) in place of one DMA
            per sequence. The compiler orders every two dynamic-offset writes of the pool, so the per-sequence writes
            run one after another, and the next staged group's kv norm waits for them. Each row that this rank does
            not own goes to its own row of the null page (row ``stage_row + b`` for sequence ``b`` of a staged group,
            row ``b`` otherwise), so no two rows of the layer share a target. The outputs and the owned rows do not
            change; only the null page holds other bytes. Not with ``pair_pages``. False (default) keeps the program.
        pack_sbuf (SbufTiles): None, or a region allocator for the tiles of the packed loop that need explicit
            addresses (the state, the TI offsets and the plan), when the other tiles of the body have compiler
            placement (``sbuf`` None). Without it and without ``sbuf``, those tiles take fixed addresses
            (160 KB and 200 KB), which other tiles can also take. A staged layer (``stage_shared``) gives each
            group the same tiles: the groups run their loops one after the other.
        pack_prefetch (int): slots of the next packed iteration whose slot rows and pages load at the end of an
            iteration, 1 (the default) to 8 and at most ``pack_slots``. More than 4 adds a slot row tile for each
            slot above 4 (1040 bytes a partition). With ``pack_carry`` at most 2, or at most 4 with ``pack_lookahead``
            2 (its ring of 8 slot rows): then slots 2 and 3 of the next iteration also load before the iteration
            fence, so the first slots after the fence wait less for their page DMAs. The DMAs and their data are the
            same, so the output is bitwise the same.
        o_proj_stream (bool): load and consume one o_proj weight chunk at a time, so only one chunk is
            resident, instead of prefetching every chunk. The matmul accumulation is unchanged, so the
            output is bitwise identical. It frees SBUF for a caller whose region is otherwise full (the
            verify step on the loop path, which keeps Q*B rows), at the cost of the weight prefetch overlap.
        pack_row_dma_act (bool): the packed loop issues the slot row DMA of each slot on the ScalarE (Activation)
            HWDGE queue instead of the SP queue, which otherwise issues all 5 DMAs of a slot. The DMA and its bytes
            are the same, so the output is bitwise the same. It moves the issue work of one DMA per slot from SP to
            ScalarE, so it pays only where SP paces the loop (32 sequences per replica).
        pack_page_runs (int): 0 (default), or the pages of one page-run DMA of the packed loop, 2 or 4: a slot then
            loads the 4 pages of its key group with 4 / ``pack_page_runs`` DMAs, each from the first page of its run.
            The table entries ``4 g + i`` of each group must be runs of pages: ``t[4 g + i] = t[4 g + i - i % R] +
            i % R`` for ``R = pack_page_runs`` (the 128-row view of 256-row FP8 pages gives runs of 2, of 512-row
            pages runs of 4). The DMAs read the bytes of the 4 page DMAs, so the output is bitwise the same. A
            sequence with a group that is not made of runs gets a NaN softmax sum, so its output is NaN. Row-major
            pages only. 4 needs a pool of 512-row FP8 pages (``fp8_pages`` with a BF16 ``[P, 147456]`` pool, or pages
            of a multiple of 512 rows): each key group is then the 4 kernel pages of one page, a run by construction.
            The 2 pages of a group of 256-row pages need not be consecutive, so a pool of 256-row or 128-row pages
            refuses 4.
        pack_null_copy_vector (bool): with ``zeroed_pages``, the packed loop copies the first page of each key group
            for its null table entries on VectorE, so that page fix waits on no other engine (with the copy on
            ScalarE the trn3 2-server verify hung at that wait). The values do not change. False (default) keeps the
            program.
        pack_null_runs (bool): with ``pack_page_runs``, an item of null pages only (a padded sequence: position 0
            and a zero block table row) counts as runs, so a padded sequence gets finite rows, not NaN. The rows of
            real sequences do not change. False (default) keeps the program.
        wgate_early (str): where the gate projection weights (the ``g`` columns of ``mla_in_proj``, 7 DMAs, 5.5 MB)
            load. "off" (the default) loads them on the SP queue after the query gather, where they share the DMA
            queue with the loads of the key loop and the fence before the loop waits for them. "sync" and "scalar"
            load them right after the q_a | kv_a gather, when the DMA rings are mostly idle up to the query gather,
            on the SP or the ScalarE hardware DMA queue. Each load also waits for the first read of the gather
            result (a semaphore wait), so it does not share the rings with the gather. On the ScalarE queue the
            loads take no SP queue slots, so the loads of the key loop do not wait behind them, and the TensorE
            read-outs up to the query gather follow the loads in the schedule, so ScalarE issues all 7 first.
            Their tiles (42 KB a partition) are allocated from that point on. The DMAs and their data are the same,
            so the output is bitwise the same.
        pack_key_transpose (str): how the packed loop transposes the latent keys of FP8 row-major pages: "matmul"
            (the default, a matmul with a BF16 identity) or "transpose" (TensorE transpose mode,
            ``_transpose_mode_tile``), which trn3 can run in the TensorE background beside the matmuls. The keys and
            the scores are the same, so the output is bitwise the same. Other page forms already transpose in
            transpose mode.
        pack_latent_act (int): latent key tiles of each packed slot (2 or 3 of the 4) whose PSUM-to-SBUF copy runs on
            ScalarE; the others run on VectorE. Exact copies, so the output is bitwise the same.
        pack_mask_column (bool): the packed loop writes the key mask of each slot on GpSimd as an FP8 column of its page
            tile (-448 for a masked key, 0 else; the page rows get a 608-byte pitch), which the rope transposes carry into
            the bias row, and the bias row of the query is 2^100 instead of 1.0. This replaces the VectorE mask row. A
            masked score is then about -5.7e32 instead of -1e30 and its weight is 0 in both, and an unmasked key adds
            +-0, so the output is bitwise the same. FP8 row-major pages only.
        pack_bg_rope (bool): with ``pack_mask_column``, the 4 rope transposes of each packed slot run in TensorE
            transpose mode, as the latent keys of ``pack_key_transpose`` "transpose" do: the same key values, so the
            output is bitwise the same.
        pack_bg_interleave (bool): with ``pack_key_transpose`` "transpose", the packed loop emits latent tile k of the
            next slot's key transposes after QK matmul k of a slot, and pins them between QK matmuls k and k + 1 (the
            rope matmul after tile 3) with scheduling-only dependencies, so the background transposes run under the QK
            matmuls instead of as one block. The same instructions in another order, so the output is bitwise the
            same.
        pack_carry (bool): with ``pack_bg_interleave``, the 2-deep slot pipeline of the packed loop carries across the
            iteration fence: the last slot of an iteration transposes the latent keys of the next iteration's slot 0, and
            the weight transposes and P@V of the last two slots run in slots 0 and 1 of the next iteration (or of the
            next loop). The prologue transposes slot 0 of the first iteration and primes the carried buffers with a
            null slot (zero weights and pages, the state column of the null items), and one drain after the last loop
            runs the last two slots. The same instructions and state adds in slot order, so the output is bitwise the
            same.
        pack_carry_main_only (bool): with ``pack_carry``, carry the pipeline in the main packed loop only and drain it
            right after the main loop, before the tail loops, which keep their own per-iteration form. When the tail
            loops run no iteration, TensorE then reaches the drain before their forward skips (instruction fetches of
            far targets) instead of after them. The same instructions and state adds in slot order, so the output is
            bitwise the same.
        pack_sum_after_copy (bool): with ``pack_bg_interleave``, the VectorE weight sum add of slot j - 2 (issued with
            the P@V of slot j) issues after the VectorE copy of latent key tile 0 of slot j + 2 (issued in slot j + 1),
            not before it (trace order and a scheduling-only dependency). That copy then does not queue behind the
            sum add, and the TensorE transposes that reuse its PSUM bank wait less. The sum adds keep the slot order
            on their own state column, so the output is bitwise the same. False (the default) keeps the order.
        pack_pair (dict): None, or the packed loop pair of this staged group (``decode_block`` ``mla_pack_pairs``):
            ``half`` 0 or 1 of two groups of B sequences that run one packed loop, ``tokens`` 2B, the ``block_table`` and
            ``position`` of the 2B sequences, ``rank_info``, and ``state_hbm``, the 2B columns of the pair in the HBM
            state of the stage. The groups share the state tile ``[96, (2B + 1) x 513]`` (``stage_shared``), each half
            in its own B columns. Half 0 stages its slot rows and returns from the loop, half 1 runs the loop of the pair
            and stores the state of both halves (``_packed_key_loop`` ``pair``). Each sequence adds its own key groups in
            the same order, so the output is bitwise the same. Staged packed loop with an FP32 HBM state only. Half 1
            takes a ``pack_plan`` of the 2B sequences without TI offsets (``decode_block`` ``mla_pack_pair_plans``),
            else it builds the items of the pair in each layer; half 0 takes none.
        pack_setup_ahead (bool): a staged group of the packed loop (``decode_block`` ``mla_pack_setup_ahead``) with its own
            plan (``pack_plan``; none for half 0 of a pair) and its own HBM rows and items (``sources["mla_pack_hbm"]``):
            the body yields a sync site (``_SETUP_SITE`` of ``MLA_SYNC_SITES``) once its items and slot rows are stored and
            before its loop (``_packed_key_loop_steps`` ``setup_ahead``), so ``decode_block._run_staged`` runs that part
            of every group of the layer before the first loop. The state of the group starts after the site, as the
            groups share the state tile (the state of a later unit starts after the loops of the units before it). The
            same instructions on the same data, so the output is bitwise the same. False (default) keeps the program.
        pack_state_store_queues (int): the packed loop with the FP32 HBM state of a staged group: the HBM store of the
            state before its reduce-scatter (of the group, or of both halves of a pair) runs as this many DMAs of
            consecutive sequence blocks, each on all 96 partitions, on the ScalarE and SP hardware DGE queues and (3)
            the GpSimd software DGE queue (``_STATE_STORE_QUEUES``), not as one DMA. The same bytes, so the output is
            bitwise the same. 1 (default) keeps the program.
        pack_current_after_rs (bool): the packed loop with an FP32 HBM state: the state of each sequence starts at 0, and
            the work after the state reduce-scatter adds the current key once, with its whole weight ``2**-93``, to the
            merged state of each local head, from an HBM copy of the current rows of the stage (``_add_current_rows``).
            Without it each of the 32 ranks starts its state with ``2**-98`` of the current key (``_CurrentKey``: one
            matmul and one state copy for each sequence), which the reduce-scatter sums. The current key then adds once,
            as in the GPU decode, but the FP32 order of the state sums changes, so the output is not bitwise the same.
            False (the default) keeps the program.
        stage_post (bool): a staged group (``decode_block`` ``mla_stage_post``, ``sources["post_stage"]`` the HBM tensors
            of the stage) runs the work after the state reduce-scatter once for all groups of the stage, not once for
            each group. The first group loads the merged state of every group, one partition per (head, sequence), and
            normalizes it and runs the value projection for all of them (``_value_projection_rows``: 4 transposes and
            one VectorE pass on 3 S partitions, for S sequences of the stage, in place of 4 B transposes and a pass on 3
            partitions for each group). Each group still runs its own gate projection and writes its sigmoid into the
            columns of the stage. The last group runs o_proj once on all S sequences (moving S columns, so each o_proj
            weight tile loads once, not once for each group) and stores the partial of the stage; the others store
            nothing. Every output column takes the same element operations and the same matmul column, so the output
            is bitwise the same. FP32 HBM state, HBM out reduce-scatter, no ``o_proj_stream``. False (default) keeps
            the program.
        stage_pre (bool): a staged group (``decode_block`` ``mla_stage_pre``, ``sources["pre_stage"]`` the HBM tensors
            of the stage and the ``[128, S, 56]`` input of its S sequences) runs the work before the query gather once
            for the S sequences of the stage, not once for each group: the first group runs the qkv projection, one
            qkv gather store and load, Steps 9 to 14 and one query gather store for all of them, and each group takes
            its rows of the current cache rows and keys for its cache writes, its softmax reference and its state. Each
            matmul column and each row or column of the norms takes the same values, so the output is bitwise the same.
            The HBM qkv and query gathers of an unsplit stage, at most 128 sequences in whole query sets, no pair
            pages, a BF16 ``mla_in_proj`` (no ``mla_in_proj_scale``). False (default) keeps the program.
        pack_mx_keys (str): None (the default, keeps the program), "single" or "split": the packed loop takes the latent
            part of each score as an MX matmul (``float8_e4m3fn_x4``, unit E8M0 scales) of FP8 queries and FP8 keys
            instead of 4 BF16 x FP8 matmuls. Each page takes one 32-bit TensorE transpose, which gives partition ``c``
            the latent features ``4c .. 4c + 3`` of every key, and two FP32 copies (ScalarE, VectorE) move a slot's
            words out of their PSUM bank: 4 transposes and 2 copies a slot instead of 16 transposes and 4 copies.
            "single" rounds the whole query ``[q_latent | q_rope]`` once to e4m3 (unit scale, nearest even, saturated at
            448), as the GPU decode does: one MX matmul, the rope matmul with the e4m3 rope values, and the current
            score of the softmax reference with the e4m3 query. "split" takes hi = e4m3(q) and lo = e4m3(q - hi) of the
            latent query in two MX matmuls, so the scores keep about the BF16 query precision; the rope query and the
            current score stay BF16. The weights and the value sums do not change. The output is not bitwise the same.
            It needs the packed loop (more than 16 local blocks) on Trn3 FP8 row-major pages with
            ``pack_bg_interleave``, ``pack_mask_column`` and ``pack_bg_rope``, and no ``pack_pair``; it refuses every
            other form.
        pack_rope_ahead (bool): with the packed loop, ``pack_carry`` (not ``pack_carry_main_only``) and
            ``pack_bg_rope``, each packed slot j traces the transpose-mode rope key transposes and the ScalarE rope key
            copy of slot j + 1 right after its rope matmul: two transposes in each of the first two gaps between the
            P@V matmuls of slot j - 2 (the weight transposes of slot j - 1 in the last two), and the copy before the exp
            of slot j, with the weight copy of slot j - 1 after the exp and before the first ScalarE key copy of the
            next slot (scheduling-only dependencies); the prologue does those of slot 0. The rope matmul of slot j + 1
            then does not wait for a copy that queues behind the exp of slot j and the weight copy of slot j - 1.
            The same instructions on the same pages, so the output is bitwise the same. Not with ``pack_mx_keys``.
            False (the default) keeps the program.
        pack_dr_keys (bool): with the packed loop on Trn3 FP8 row-major pages with ``pack_key_transpose`` "transpose",
            ``pack_bg_interleave``, ``pack_mask_column`` and ``pack_bg_rope``, the latent part of each score takes 2
            double_row matmuls of e4m3 queries and FP8 keys instead of 4 BF16 x FP8 matmuls. Each page takes 2 2-byte
            transpose-mode transposes (one for each 256-feature block), which give partition ``c`` the features
            ``2c`` and ``2c + 1`` of the block of every key; double_row pass ``i`` then reads byte ``i`` of the words of
            both blocks: 8 transposes and one ScalarE copy a slot instead of 16 transposes and 4 copies (2 on VectorE).
            With ``pack_rope_ahead`` the weight copy of slot j - 1 also goes before the exp of slot j. The whole query
            ``[q_latent | q_rope]`` is rounded once to e4m3 (unit scale, nearest even, saturated at 448), as the GPU
            decode rounds it, and the current score of the softmax reference takes the same query. The weights and
            the value sums do not change. The output is not bitwise the same. Not with ``pack_mx_keys``; with
            ``pack_rope_ahead`` and ``pack_pair``. False (the default) keeps the program.
        mla_in_proj_scale (nl.NkiTensor): None (default, a BF16 ``mla_in_proj``), or the ``[128, 2, 450]`` uint8 scales
            of an MXFP8 ``mla_in_proj`` ``[14, 128, 1800]`` (``tkg_mx_quant``; columns 0..65 the qkv columns, then the 384
            gate columns): the q_a | kv_a and the gate projections run MXFP8 x MXFP8 with the hidden quantized on the
            VectorE. The values change (MXFP8 rounding of both operands).
        mla_q_b_proj_scale (nl.NkiTensor): None (default, a BF16 ``mla_q_b_proj``), or the ``[128, 1, 576]`` uint8
            scales of an MXFP8 ``mla_q_b_proj`` ``[3, 128, 2304]`` (``tkg_mx_quant`` hT_quad): the q_b projection (Step 13)
            runs MXFP8 x MXFP8, the q_latent quantized per sequence on the VectorE and the q_b weight the moving operand
            (the Step 13 orientation). The values change (MXFP8 rounding of both operands). The verify twin
            (``mla_block_attention`` ``mla_q_b_proj_scale``) takes the same key, so verify rows stay bitwise equal to
            the decode steps. CP32, 3 local heads, Trn3 only; independent of ``mla_in_proj_scale``.
        mla_kv_b_proj_scale (nl.NkiTensor): None (default, a BF16 ``mla_kv_b_proj``), or the compact ``[16, 896]``
            uint8 scales of an MXFP8 ``mla_kv_b_proj`` ``[128, 3584]`` (``mla_decode_torch`` ``mx_weights``
            ``{"mla_kv_b_proj": "hT_quad"}``): Step 14 (the key absorb) runs the 3 heads as MXFP8 quadrant row tiles
            (``tkg_mx_quant.quadrant_absorb_mx``), the q_nope quantized per sequence, and the value projection runs one
            MXFP8 group a head, the latent quantized per sequence. The values change. The verify twin takes the same
            key. CP32, 3 local heads, Trn3 only.
        mla_o_proj_scale (nl.NkiTensor): None (default, a BF16 ``mla_o_proj``), or the compact ``[16, 7168]`` uint8
            scales of the MXFP8 padded-lane ``mla_o_proj`` ``[1, 128, 28672]`` (``mx_weights``
            ``{"mla_o_proj": "hT_quad"}``): each hidden tile of the o_proj is one MXFP8 matmul, the weight stationary,
            the gated value quantized per sequence. The values change. The verify twin takes the same key. CP32, 3
            local heads, Trn3 only; not with ``o_proj_ring``.
        mx_query (str): "off" (default) or "emulate": round the absorbed latent query of each head to its MXFP8 values
            (the quantize_mx rule, scale groups of 8 rows x 4 latent tiles, ``_mx_emulate_rows``) before the query
            gather. The scores then see the query of an MXFP8 x FP8 QK matmul (the keys are FP8 already); the matmuls
            stay BF16, so this is an accuracy form, not a perf form. The values change.
        pack_row_dma_pool (int): 0 (the default, keeps the program), or a period k of 2 to 4: with the packed loop
            (more than 16 local blocks, one query row per item), slot j of a packed iteration with j % k == k - 1
            issues its slot row DMA on the GpSimd software DGE queue instead of the SP queue (``_packed_key_loop``
            ``row_dma_pool``). The SP queue issues the 3 DMAs of every other slot (the row and the 2 page runs with
            ``pack_page_runs`` 2), and these SP instructions pace the form-B slot (they take longer than its TensorE
            work); GpSimd has the slack. The DMA and its bytes are the same, so the output is bitwise the same.
            Not with ``pack_row_dma_act``.
        pack_lookahead (int): 1 (the default, keeps the program) or 2: with ``pack_bg_interleave`` and a
            ``pack_prefetch`` of at least 2, packed slot j issues the slot row and page DMAs of slot j + 2 instead of
            those of slot j + 1 (with ``pack_carry`` the last two slots of an iteration load slots 0 and 1 of the next
            one), and the loop keeps 8 slot rows instead of 4 (4 more tiles of 1040 bytes a partition), so a row DMA
            does not wait for the state add of the slot 4 before it. The SP load DMAs of an iteration keep the slot
            order (the pages of a slot, then its row; scheduling-only dependencies). The slot rows then load one slot
            earlier. The DMAs and their data are the same, so the output is bitwise the same.
        head_prefetch (MlaHeadTiles): None (the default, keeps the program), or the BF16 qkv and q_b weight tiles of
            this call from ``prefetch_mla_head``, whose loads a caller issued earlier (for example inside the routed
            all-reduce of the feed-forward body before); the body then loads only its other weights. The tiles hold the
            same bytes, so the output is bitwise the same. Not with ``mla_in_proj_scale``, ``stage_pre`` or a staged
            layer (``stage_shared``).
        pack_psum_per_block (bool): False (the default, keeps the program) or True: the packed loop allocates its loop
            PSUM tensors (the score buffers, and the key-word, numerator, weight and rope banks of ``pack_dr_keys`` /
            ``pack_mx_keys``) in each top-level block that uses them (each loop body and each drain) instead of once
            before the loops, so no PSUM tensor is used in more than one top-level block, as the BIR verifier requires
            (NCC_IBIR092 otherwise; the served compile turns the verifier off). Each block writes them before it reads
            them, so the same instructions on the same data: the output is bitwise the same.
        loop_restart (str): the iteration restart of the per-sequence key loops of the loop path (``LOOP_RESTARTS``,
            ``_key_loops`` ``restart``): "off" (the default) keeps the program; "table" loads each iteration's table
            entries one iteration ahead. Needs an explicit ``sbuf`` (a region), no ``pair_pages`` and no
            ``packed_loop``. The output is bitwise the same.
        pack_step_items (dict): None (the default, keeps the program), or the step store of this layer's packed loop unit
            (``decode_block`` ``mla_options`` ``pack_step_items``: one per pair or group, kept for the whole step): the
            first MLA layer of the step that runs the unit writes its item rows (page ids) into the store and every later
            layer reuses them instead of gathering its own (``_packed_key_loop`` ``step_items``). Valid only when every
            MLA layer of the step has the same block table, which the decode builds once for all MLA layers; the item
            rows are then the same bytes, so the output is bitwise the same. Needs the packed loop with one query row
            per item, and no ``pack_setup_ahead``.
        loop_groups (int): the key groups of one iteration of the per-sequence key loops (``LOOP_GROUP_COUNTS``,
            ``_key_loops`` ``groups``): 0 (the default) keeps the program (8 groups, 4 with BF16 pages). 16: each
            sequence runs 16-group iterations while they hold key groups, then the rest in iterations of the default
            groups, so it attends the groups of the default loop in about half the iterations (half the back edges of
            a long sequence). Loop path only, no ``packed_loop``. The output is bitwise the same.
        o_proj_ring (int): with ``o_proj_stream``, the o_proj weight chunks in flight (``O_PROJ_RINGS``): 0 (the
            default) keeps the program (one chunk tile, so each chunk's load waits for the previous chunk's
            matmuls). 2 or 4: that many chunk tiles; the first ones load where the non-streamed form loads its
            chunks, and chunk c + ring loads into chunk c's tile after chunk c's matmuls, so a chunk's load runs
            during the earlier chunks' matmuls. The same bytes and the same accumulation order: bitwise the same.
            Needs ``o_proj_stream`` and no shared stage weights.
        loop_page_gather (str): the page loads of the per-sequence key loops (``LOOP_PAGE_GATHERS``, ``_key_loops``
            ``page_gather``): "off" (the default) keeps the program (one SP DMA and one table load per page); "group"
            loads the 4 pages of each key group with one GpSimd software DGE gather, from indices built one iteration
            ahead; "carry" also gathers group 0 of the next iteration inside each iteration; "carry_pool" is "carry" with
            the next iteration's gather indices built on GpSimd, so that gather issues mid-iteration and not right before
            the back edge. Needs ``loop_path`` with more than 16 local blocks, FP8 row-major pages, an explicit ``sbuf``
            (a region), and no ``packed_loop``. The output is bitwise the same.
        pack_stage_rows_sbuf (bool): False (the default, keeps the program) or True: with the packed loop (more than 16
            local blocks), the slot rows of the first ``pack_prefetch`` slots, which the loop prologue loads, come from
            the SBUF stage tile of the slot rows (the same item field) instead of from its HBM store, so the first QK of
            each loop entry does not wait for the store to land and the row to come back (``_packed_key_loop_steps``
            ``stage_rows_sbuf``). The stage tile then lives until those loads (with ``sbuf``, outside the frame of the
            staging: ``(B + 1) x 1040`` more bytes a partition in the frame of the loop). The later slot rows still load
            from the HBM store. The same bytes, so the output is bitwise the same. Row-major pages; not with
            ``pack_pair`` or ``pack_setup_ahead``.
        o_proj_preload (bool): False (the default, keeps the program) or True: the 4 o_proj weight DMAs (5.5 MB on the SP
            queue, 43008 bytes a partition) issue at the first read of the q_a | kv_a gather result, each with a
            semaphore wait on that read (as the loads of ``wgate_early``), not after the state reduce-scatter. They
            stream while the DMA rings are mostly idle up to the query gather, so the tail of the layer does not wait
            for them. The tiles live from there to o_proj (a staged layer loads them once, in its first group). The same
            DMAs and matmuls, so the output is bitwise the same. Not with ``o_proj_stream`` or ``stage_pre``, and only
            with compiler-placed tiles (``sbuf`` None, as the staged MLA groups above 8 sequences): in an explicit SBUF frame
            the tiles stay on the stack through the key loops, and the 1M low-batch frames (4 and 8 sequences) then
            overflow (NCC_INKI016).
        o_proj_early (bool): with ``o_proj_ring``, the ring's first chunks load with the head weights of the call,
            before its key loops, into tiles that stay resident until the o_proj (the loads must complete before the
            first key-loop fence; ``o_proj_ring`` x 10,752 bytes a partition stay live through the loops). The other
            chunks keep the ring. The same bytes in the same tiles and the same accumulation order: bitwise the same.
            False (the default) keeps the program.
        loop_inputs_ahead (bool): the loop inputs of the per-sequence key loops (``_key_loops`` ``inputs_ahead``):
            False (the default) keeps the program (each sequence's register loads and counts, and with
            ``loop_restart`` "table" its first table row, issue right before its loop). True: those of every sequence
            issue before the first sequence's loop, so only branches lie between two sequences' loops and the compiler
            puts no fence there. Needs ``loop_path`` with more than 16 local blocks and no ``packed_loop``; with
            ``loop_restart`` "table" an explicit ``sbuf`` (a region). The output is bitwise the same.
        loop_dr_keys (bool): the double_row keys of ``pack_dr_keys`` on the per-sequence loop path of Trn2 (more than
            16 local blocks, ``loop_path``, FP8 row-major pages, the capped softmax). Each page takes 2 2-byte
            transpose-mode transposes (one for each 256-feature block) in place of 4 FP8 -> BF16 copies and 4 BF16
            transposes, one ScalarE copy moves the key words of a group, and the latent part of each score takes 2
            double_row matmuls of e4m3 queries and the FP8 keys in place of 4 BF16 matmuls. The rope keys keep their
            BF16 transposes. The whole query ``[q_latent | q_rope]`` is rounded once to the FP8 page format (nearest
            even, saturated at ``fp8_page_max``), and the current score of the softmax reference takes the same
            query. The weights and the value sums do not change. The output is not bitwise the same. False (the
            default) keeps the program.
        pack_mask_pool (bool): the packed loop writes the key mask row of each slot on GpSimd, not on the engine that
            the compiler selects (the released neuronx-cc selects VectorE). The mask row is one compare and one multiply
            on 512 FP32 slot offsets in SBUF, so GpSimd can do it. On BF16 pages VectorE paces the slot (the page clamps,
            two latent key copies, the score cap and the state adds), and GpSimd has slack. The same operation on the
            same values, so the output is bitwise the same. It needs the packed loop and no ``pack_mask_column`` (which
            writes no mask row). False (the default) keeps the program.

    Returns:
        y_hT (nl.NkiTensor): [128, B, 56] BF16 SBUF attention output after the TP all-reduce, in
        token-major hT layout. Its attribute ``side_effects`` holds the cache-row write DMAs, which the
        output does not depend on.

    Notes:
        - The owner rank ``(position // 128) % 32 == t`` writes the current row at ``slot``. Another rank,
          or the owner of a slot in the null page, writes the row to row 0 of the null page 0 instead, so
          page 0 holds finite data that no read may trust.
        - The prior keys are the local rows with a global position below ``position``. Unused rows can
          hold any finite data. A padded sequence uses position 0, slot 0, and a zero block table row.
        - The softmax weights of the value sum are BF16. The softmax sums and the CP merge are FP32.
        - Every rank shifts the scores by the scaled current score plus 64.46, which all ranks compute
          alike, so the softmax and the CP merge need no maximum. The current key then weighs ``2**-93``,
          and each rank adds 1/32 of it to every head. A key that weighs less than 1.3e-10 of the
          current key underflows to 0.
        - The scores are capped at 124.46 above the scaled current score, so no weight overflows. Below the
          cap the softmax is exact. Above it, the keys share one weight, e**60. One dominant key still
          gets all the weight. The default is thus not exact only when a key beats the current key by more
          than 124.46 (scaled). Use exact_softmax=True to remove this limit.
        - A table of up to 4 blocks takes the static path, which attends every block. A larger table
          takes the dynamic path, which attends key groups of 4 blocks into one FP32 state. A table of 8
          or 16 blocks attends every group in straight-line code (the unrolled path). A larger table has
          one loop per sequence over its groups that hold a prior key, so the cost follows the context of
          each sequence. Each loop adds a fixed cost. With such a neuronx-cc build, use more
          than 16 blocks only with ``sbuf`` and the long-lived tiles of the caller at explicit addresses.
        - More than 4 sequences run as two query sets of up to 4 (sequences 0-3 and 4-7). The projections
          and the collectives take all sequences. The query gather sends the two sets side by side, so its
          32 x 4 rows fill the 128 partitions. The static path keeps the key tiles of 4 sequences: it loads
          the pages of sequence b + 4 when sequence b is done.

    Pseudocode:
        q_a | kv_a | g = x @ mla_in_proj
        q_latent = rms(all_gather(q_a)); kv = all_gather(kv_a); row = [rms(kv[:512]) | kv[512:]]
        write row at slot on the owner rank, else at row 0 of the null page
        q = q_latent @ q_b; q_abs = q_nope @ key; q_all = all_gather(q_abs | q_pe)
        reference = scaled current score + 64.46
        weights = exp(q_all . [local rows] * 192**-0.5 - reference), masked
        state = [weights @ latents + current / 32 | sum(weights) + 2**-98]
        latent = reduce_scatter_heads(state); latent = latent.num / latent.sum
        o = (latent @ value) * sigmoid(g)
        y = all_gather(reduce_scatter(o @ mla_o_proj))
    """
    tokens = x_hT.shape[1]
    local_blocks = block_table.shape[1]
    sb = AUTO_SBUF if sbuf is None else sbuf
    kernel_assert(
        not (o_proj_preload and sb.explicit),
        "o_proj_preload keeps the o_proj weight tiles (43008 bytes a partition) from the query chain to o_proj; in an explicit "
        "SBUF frame (sbuf) they stay on the stack through the key loops, which then overflow the 1M low-batch frames "
        "(NCC_INKI016). Use it with compiler-placed tiles (sbuf None: the staged MLA groups above 8 sequences)",
    )
    # Kernel pages of 128 rows in one page of the pool: 2 for 256-row FP8 pages (a BF16 [P, 73728] pool with
    # fp8_pages), 4 for 512-row ones ([P, 147456]), 1 for a pool of kernel pages.
    page_blocks = kv_pool.shape[1] // _PAGE_WIDTH
    if kv_pool.dtype == nl.bfloat16:
        page_blocks = 2 * page_blocks if fp8_pages else 1
    kernel_assert(
        pack_page_runs != _GROUP_BLOCKS or page_blocks % _GROUP_BLOCKS == 0,
        f"pack_page_runs {_GROUP_BLOCKS} needs 512-row FP8 pages (fp8_pages with a BF16 [P, {2 * _PAGE_WIDTH}] pool): "
        f"each key group is then one page, so its pages are a run. This pool holds {page_blocks} kernel page(s) of "
        f"{_BLOCK} rows a page, whose groups need not be runs; use pack_page_runs 0, or 2 on 256-row pages",
    )
    if fp8_pages and kv_pool.dtype == nl.bfloat16:
        kv_pool = fp8_kernel_pages(kv_pool.view(fp8_page_dtype()), _PAGE_WIDTH)
    elif kv_pool.dtype == nl.float8_e4m3fn and not is_trn3():
        kv_pool = fp8_kernel_pages(kv_pool.view(fp8_page_dtype()), _PAGE_WIDTH)
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
        tokens,
        mla_in_proj_scale,
        mla_q_b_proj_scale,
        mla_kv_b_proj_scale,
        mla_o_proj_scale,
    )
    kernel_assert(not pair_pages or kv_pool.dtype == nl.float8_e4m3fn, "pair_pages needs an FP8 pool on Trn3")
    kernel_assert(
        not (batched_cache_write and pair_pages), "batched_cache_write writes row-major pages, not pair_pages"
    )
    kernel_assert(lora_norm_eps > 0, f"lora_norm_eps must be positive, got {lora_norm_eps}")
    kernel_assert(mx_query in _MX_QUERY_FORMS, f"mx_query must be one of {_MX_QUERY_FORMS}, got {mx_query!r}")
    kernel_assert(loop_restart in LOOP_RESTARTS, f"loop_restart must be one of {LOOP_RESTARTS}, got {loop_restart!r}")
    kernel_assert(
        loop_groups in LOOP_GROUP_COUNTS, f"loop_groups must be one of {LOOP_GROUP_COUNTS}, got {loop_groups!r}"
    )
    kernel_assert(o_proj_ring in O_PROJ_RINGS, f"o_proj_ring must be one of {O_PROJ_RINGS}, got {o_proj_ring!r}")
    kernel_assert(
        not o_proj_ring or o_proj_stream, "o_proj_ring sets the chunks in flight of o_proj_stream: it needs it"
    )
    kernel_assert(
        not o_proj_early or o_proj_ring, "o_proj_early loads the first chunks of the o_proj ring: it needs o_proj_ring"
    )
    kernel_assert(
        not loop_groups or (loop_path and not packed_loop),
        "loop_groups sets the iterations of the per-sequence key loops of the loop path: it needs loop_path and no "
        "packed_loop",
    )
    kernel_assert(
        loop_restart == "off" or (loop_path and not packed_loop and not pair_pages),
        "loop_restart changes the per-sequence key loops of the loop path: it needs loop_path, and no packed_loop "
        "or pair_pages",
    )
    kernel_assert(
        pack_step_items is None or (isinstance(pack_step_items, dict) and packed_loop and not pack_setup_ahead),
        "pack_step_items takes the step store of the decode block (a dict) and needs the packed loop without "
        "pack_setup_ahead",
    )
    kernel_assert(
        loop_page_gather in LOOP_PAGE_GATHERS,
        f"loop_page_gather must be one of {LOOP_PAGE_GATHERS}, got {loop_page_gather!r}",
    )
    kernel_assert(
        loop_page_gather == "off"
        or (
            loop_path
            and local_blocks > _MAX_UNROLLED_BLOCKS
            and not packed_loop
            and not pair_pages
            and sbuf is not None
            and sbuf.explicit
        ),
        "loop_page_gather changes the page loads of the per-sequence key loops of the loop path: it needs loop_path "
        f"with more than {_MAX_UNROLLED_BLOCKS} local blocks, an explicit sbuf (a region), and no packed_loop or "
        "pair_pages",
    )
    kernel_assert(
        not loop_inputs_ahead or (loop_path and local_blocks > _MAX_UNROLLED_BLOCKS and not packed_loop),
        "loop_inputs_ahead moves the loop inputs of the per-sequence key loops of the loop path: it needs loop_path "
        f"with more than {_MAX_UNROLLED_BLOCKS} local blocks and no packed_loop",
    )
    kernel_assert(
        wgate_early in _WGATE_EARLY_FORMS, f"wgate_early must be one of {_WGATE_EARLY_FORMS}, got {wgate_early!r}"
    )
    kernel_assert(
        not (o_proj_preload and (o_proj_stream or stage_pre)),
        "o_proj_preload loads every o_proj weight chunk early: not with o_proj_stream or stage_pre",
    )
    kernel_assert(
        local_blocks <= _MAX_UNROLLED_BLOCKS or loop_path,
        f"block_table has {local_blocks} local blocks, which needs the loop path. neuronx-cc can corrupt SBUF "
        f"in that path. Use at most {_MAX_UNROLLED_BLOCKS} local blocks, or set loop_path=True.",
    )
    kernel_assert(not (packed_loop and exact_softmax), "packed_loop supports the capped softmax only")
    kernel_assert(pack_mx_keys in _MX_KEY_FORMS, f"pack_mx_keys must be one of {_MX_KEY_FORMS}, got {pack_mx_keys!r}")
    kernel_assert(
        pack_mx_keys is None
        or (
            packed_loop
            and local_blocks > _MAX_UNROLLED_BLOCKS
            and kv_pool.dtype == nl.float8_e4m3fn
            and not pair_pages
            and pack_pair is None
            and pack_bg_interleave
            and pack_mask_column
            and pack_bg_rope
        ),
        f"pack_mx_keys needs the packed loop (more than {_MAX_UNROLLED_BLOCKS} local blocks) on Trn3 FP8 row-major "
        "pages with pack_bg_interleave, pack_mask_column and pack_bg_rope, and no pack_pair",
    )
    kernel_assert(
        not pack_rope_ahead
        or (
            packed_loop
            and local_blocks > _MAX_UNROLLED_BLOCKS
            and pack_carry
            and not pack_carry_main_only
            and pack_bg_rope
        ),
        f"pack_rope_ahead needs the packed loop (more than {_MAX_UNROLLED_BLOCKS} local blocks) with pack_carry (not "
        "pack_carry_main_only) and pack_bg_rope",
    )
    kernel_assert(
        not pack_dr_keys
        or (
            packed_loop
            and local_blocks > _MAX_UNROLLED_BLOCKS
            and kv_pool.dtype == nl.float8_e4m3fn
            and not pair_pages
            and pack_key_transpose == "transpose"
            and pack_bg_interleave
            and pack_mask_column
            and pack_bg_rope
            and pack_mx_keys is None
        ),
        f"pack_dr_keys needs the packed loop (more than {_MAX_UNROLLED_BLOCKS} local blocks) on Trn3 FP8 row-major "
        'pages with pack_key_transpose "transpose", pack_bg_interleave, pack_mask_column and pack_bg_rope, and no '
        "pack_mx_keys",
    )
    kernel_assert(
        pack_row_dma_pool in _ROW_DMA_POOL_PERIODS
        and (
            pack_row_dma_pool == 0
            or (packed_loop and local_blocks > _MAX_UNROLLED_BLOCKS and not pack_row_dma_act and not pair_pages)
        ),
        f"pack_row_dma_pool must be one of {_ROW_DMA_POOL_PERIODS}; a period needs the packed loop (more than "
        f"{_MAX_UNROLLED_BLOCKS} local blocks) on row-major pages, and not pack_row_dma_act, got {pack_row_dma_pool}",
    )
    kernel_assert(
        not pack_mask_pool or (packed_loop and local_blocks > _MAX_UNROLLED_BLOCKS and not pack_mask_column),
        f"pack_mask_pool moves the mask row of the packed loop to GpSimd. It needs the packed loop (more than "
        f"{_MAX_UNROLLED_BLOCKS} local blocks) and no pack_mask_column. Turn off pack_mask_pool, or turn off "
        "pack_mask_column.",
    )
    kernel_assert(
        pack_lookahead in _PACK_LOOKAHEADS
        and (
            pack_lookahead == 1
            or (packed_loop and local_blocks > _MAX_UNROLLED_BLOCKS and pack_bg_interleave and pack_prefetch >= 2)
        ),
        f"pack_lookahead must be one of {_PACK_LOOKAHEADS}; 2 needs the packed loop (more than {_MAX_UNROLLED_BLOCKS} "
        f"local blocks) with pack_bg_interleave and pack_prefetch 2 or more, got {pack_lookahead}",
    )
    kernel_assert(
        not pack_stage_rows_sbuf
        or (
            packed_loop
            and local_blocks > _MAX_UNROLLED_BLOCKS
            and not pair_pages
            and pack_pair is None
            and not pack_setup_ahead
        ),
        f"pack_stage_rows_sbuf needs the packed loop (more than {_MAX_UNROLLED_BLOCKS} local blocks) on row-major pages, "
        "and no pack_pair or pack_setup_ahead",
    )
    kernel_assert(
        pack_pair is None
        or (
            packed_loop
            and loop_path
            and local_blocks > _MAX_UNROLLED_BLOCKS
            and (pack_plan is None or (pack_pair["half"] and pack_plan.get("ti") is None))
            and not bf16_state
            and sources is not None
            and sources.get("stage_shared") is not None
            and sources.get("state_hbm") is not None
            and pack_pair["half"] in (0, 1)
            and pack_pair["tokens"] == 2 * tokens
        ),
        "pack_pair needs the packed loop of a staged group (more than 16 local blocks) with an FP32 HBM state, and no "
        "pack_plan, or (half 1) a plan of the pair without TI offsets",
    )
    kernel_assert(
        not pack_setup_ahead
        or (
            packed_loop
            and loop_path
            and local_blocks > _MAX_UNROLLED_BLOCKS
            and sbuf is None
            and sources is not None
            and sources.get("stage_shared") is not None
            and sources.get("mla_pack_hbm") is not None
            and (pack_plan is not None or (pack_pair is not None and not pack_pair["half"]))
        ),
        "pack_setup_ahead needs the packed loop of a staged group (more than 16 local blocks, compiler-placed tiles) with "
        "its own plan (none for half 0 of a pair) and its own HBM rows and items",
    )
    kernel_assert(
        not pack_current_after_rs
        or (
            packed_loop
            and loop_path
            and local_blocks > _MAX_UNROLLED_BLOCKS
            and not exact_softmax
            and not bf16_state
            and sources is not None
            and sources.get("state_hbm") is not None
        ),
        f"pack_current_after_rs needs the packed loop (more than {_MAX_UNROLLED_BLOCKS} local blocks) with the capped "
        "softmax and an FP32 HBM state",
    )
    kernel_assert(
        pack_state_store_queues == 1
        or (
            1 < pack_state_store_queues <= len(_STATE_STORE_QUEUES)
            and packed_loop
            and loop_path
            and local_blocks > _MAX_UNROLLED_BLOCKS
            and not bf16_state
            and sources is not None
            and sources.get("state_hbm") is not None
        ),
        f"pack_state_store_queues must be 1 to {len(_STATE_STORE_QUEUES)}, above 1 with the packed loop of a staged group "
        f"(an FP32 HBM state), got {pack_state_store_queues}",
    )
    kernel_assert(
        not packed_loop or is_trn3(),
        "packed_loop needs Trn3 (SBUF tensor indirection). On Trn2 use the per-sequence loop path, which gives the "
        "same output.",
    )
    kernel_assert(
        loop_cast_engines in LOOP_CAST_ENGINES
        and (
            loop_cast_engines == "off"
            or (
                not packed_loop
                and not pair_pages
                and local_blocks > _MAX_STATIC_BLOCKS
                and kv_pool.dtype != nl.bfloat16
                and not is_trn3()
            )
        ),
        f"loop_cast_engines must be one of {LOOP_CAST_ENGINES}; a form other than off needs the per-sequence key loops "
        f"of the dynamic path (more than {_MAX_STATIC_BLOCKS} local blocks, no packed_loop or pair_pages) on Trn2 FP8 "
        f"pages, got {loop_cast_engines!r}",
    )
    kernel_assert(
        loop_cast_engines not in _LOOP_PAGE_CASTS or not loop_dr_keys,
        "the page forms of loop_cast_engines read BF16 pages, so not with loop_dr_keys (FP8 rows)",
    )
    kernel_assert(
        not loop_pad_stationary
        or (
            not packed_loop
            and not pair_pages
            and not exact_softmax
            and local_blocks > _MAX_STATIC_BLOCKS
            and not is_trn3()
        ),
        "loop_pad_stationary needs the per-sequence key loops of the dynamic path (more than "
        f"{_MAX_STATIC_BLOCKS} local blocks, no packed_loop or pair_pages) with the capped softmax on Trn2",
    )
    kernel_assert(
        not loop_dr_keys
        or (
            not is_trn3()
            and loop_path
            and local_blocks > _MAX_UNROLLED_BLOCKS
            and not packed_loop
            and not exact_softmax
            and not pair_pages
            and kv_pool.dtype == fp8_page_dtype()
        ),
        f"loop_dr_keys needs the Trn2 per-sequence loop path (loop_path, more than {_MAX_UNROLLED_BLOCKS} local "
        "blocks, no packed_loop) on FP8 row-major pages, with the capped softmax",
    )
    kernel_assert(
        not (loop_pad_stationary and loop_dr_keys),
        "loop_pad_stationary does not combine with loop_dr_keys; set one of them",
    )
    # The maximum row of the exact softmax, [1, 96 B] FP32, must fit one PSUM bank.
    kernel_assert(
        not exact_softmax or tokens <= QUERY_SET_TOKENS, f"exact_softmax supports 1 to {QUERY_SET_TOKENS} sequences"
    )
    num_pages = kv_pool.shape[0]
    # A table of more than 4 local blocks takes the dynamic path: one loop per sequence over the key groups
    # that hold a prior key. The static path attends every block of the table.
    dynamic = local_blocks > _MAX_STATIC_BLOCKS
    group_blocks = min(local_blocks, _GROUP_BLOCKS)
    num_keys = group_blocks * _BLOCK
    sets = -(-tokens // QUERY_SET_TOKENS)
    set_rows = min(tokens, QUERY_SET_TOKENS)
    key_buffers = 0 if dynamic else set_rows
    # A BF16 PSUM column block starts on a 4-byte boundary, so odd token counts are padded.
    padded = tokens + tokens % 2
    bf16 = nl.bfloat16
    fp32 = nl.float32
    hwdge = nisa.dge_mode.hwdge
    sync = nisa.engine.sync

    # Step 1: load the sequence metadata. One sequence per partition: position, slot, TP rank. The bias
    # partition holds the positions as a row and the TP rank, for the key masks.
    token_meta = sb.tile((tokens, 3), nl.int32)
    meta_sources = (
        (position[0:tokens, 0:1], "position"),
        (slot[0:tokens, 0:1], "slot"),
        (rank_info[0:1, 2:3].broadcast(0, tokens), "rank"),
    )
    for meta_col, (meta_src, meta_name) in enumerate(meta_sources):
        nisa.dma_copy(
            dst=token_meta[0:tokens, meta_col : meta_col + 1],
            src=meta_src,
            dge_mode=hwdge,
            engine=sync,
            name=f"{tag}_mla_{meta_name}_load",
        )
    bias_meta = sb.tile((_P_MAX, tokens + 1), nl.int32)
    nisa.dma_copy(
        dst=bias_meta[_BIAS_ROW : _BIAS_ROW + 1, 0:tokens],
        src=position.ap(pattern=[[tokens, 1], [1, tokens]], offset=0),
        dge_mode=hwdge,
        engine=sync,
        name=f"{tag}_mla_position_row_load",
    )
    nisa.dma_copy(
        dst=bias_meta[_BIAS_ROW : _BIAS_ROW + 1, tokens : tokens + 1],
        src=rank_info[0:1, 2:3],
        dge_mode=hwdge,
        engine=sync,
        name=f"{tag}_mla_rank_row_load",
    )
    if not dynamic:
        table_size = tokens * local_blocks
        block_ids = sb.tile((1, table_size), nl.int32)
        nisa.dma_copy(
            dst=block_ids[0:1, 0:table_size],
            src=block_table.ap(pattern=[[table_size, 1], [1, table_size]], offset=0),
            dge_mode=hwdge,
            engine=sync,
            name=f"{tag}_mla_block_table_load",
        )
        # A padded table entry is clamped into the pool, so its masked rows are still finite reads.
        nisa.tensor_scalar(
            dst=block_ids[0:1, 0:table_size],
            data=block_ids[0:1, 0:table_size],
            op0=nl.minimum,
            operand0=float(num_pages - 1),
            op1=nl.maximum,
            operand1=0.0,
        )

    # Step 2: on the static path, load the pages of every sequence, one contiguous DMA per page with the
    # page id as a dynamic offset. Partition p of block tile j is local slot 128 j + p.
    key_rows = [_page_tile(sb, local_blocks, kv_pool.dtype, pair_pages) for _ in range(key_buffers)]

    def load_pages(token):
        rows = key_rows[token % key_buffers]
        for local_block in range(local_blocks):
            entry = token * local_blocks + local_block
            if pair_pages:
                load_pair_page(
                    kv_pool,
                    rows,
                    local_block,
                    block_ids[0:1, entry : entry + 1],
                    f"{tag}_mla_page_load_{token}_{local_block}",
                )
                _sanitize_cache_rows(_page_view(rows, local_block, True))
                continue
            nisa.dma_copy(
                dst=rows[0:_P_MAX, local_block, 0:_CACHE_WIDTH],
                src=kv_pool.ap(
                    pattern=[[_CACHE_WIDTH, _BLOCK], [1, _CACHE_WIDTH]],
                    offset=0,
                    scalar_offset=block_ids[0:1, entry : entry + 1],
                    indirect_dim=0,
                ),
                dge_mode=hwdge,
                engine=sync,
                name=f"{tag}_mla_page_load_{token}_{local_block}",
            )
            _sanitize_cache_rows(rows[0:_P_MAX, local_block, 0:_CACHE_WIDTH])

    for token in range(key_buffers):
        load_pages(token)

    # Step 3: load the head weights on the SP queue. In a decode block they wait for the weight gate of the
    # block (tkg_weight_dma_gate.py). In a block the gate and o_proj weights also wait for the query gather and the
    # state reduce-scatter, so they do not slow those collectives.
    # A staged layer (``stage_shared``) loads the weights with its first group and shares them with the others.
    shared = sources.get("stage_shared") if sources else None
    # stage_pre: the first group runs Steps 6 to 14 for the pre_rows sequences of the stage; the others take their rows.
    pre_stage = sources.get("pre_stage") if stage_pre and sources else None
    kernel_assert(
        not stage_pre
        or (
            pre_stage is not None
            and shared is not None
            and "qkv_ag_src_hbm" in sources
            and "q_ag_src_hbm" in sources
            and "qkv" not in sources
            and "q" not in sources
            and "split_sites" not in pre_stage["stage"]
            and not pair_pages
            and tokens % QUERY_SET_TOKENS == 0
            and sources["stage_rows"] % QUERY_SET_TOKENS == 0
            and sources["stage_rows"] <= _P_MAX
        ),
        f"stage_pre needs a staged group (sources pre_stage) with the HBM qkv and query gathers of an unsplit stage of at "
        f"most {_P_MAX} sequences in whole query sets, and no pair_pages",
    )
    kernel_assert(
        not stage_pre or mla_in_proj_scale is None,
        "stage_pre runs the BF16 qkv projection of the stage; it does not take an MXFP8 mla_in_proj (mla_in_proj_scale)",
    )
    pre_rows = sources["stage_rows"] if pre_stage is not None else tokens
    pre_run = pre_stage is None or sources["stage_row"] == 0
    pre_x = pre_stage["x_hT"] if pre_stage is not None else x_hT
    # head_prefetch: the qkv and q_b tiles of this call, loaded by a caller ahead of the body (prefetch_mla_head).
    head_pre = head_prefetch
    kernel_assert(
        head_pre is None
        or (mla_in_proj_scale is None and mla_q_b_proj_scale is None and not stage_pre and shared is None),
        "head_prefetch gives the BF16 qkv and q_b tiles of one unstaged call: not with mla_in_proj_scale, "
        "mla_q_b_proj_scale, stage_pre or stage_shared",
    )

    def load_head_weights():
        qkv_chunks = [] if head_pre is None else list(head_pre.qkv_chunks)
        if mla_in_proj_scale is not None:
            qkv_chunks.append(_load_mx_in_proj(mla_in_proj, mla_in_proj_scale, sb, tag))
        own_qkv = mla_in_proj_scale is None and head_pre is None
        for chunk_start in range(0, _HIDDEN_TILES if own_qkv else 0, _QKV_CHUNK_TILES):
            chunk_tiles = min(_QKV_CHUNK_TILES, _HIDDEN_TILES - chunk_start)
            qkv_w = sb.tile((_P_MAX, chunk_tiles, _QKV_LOCAL), bf16)
            gate_weight_dma(
                nisa.dma_copy(
                    dst=qkv_w,
                    src=mla_in_proj[0:_P_MAX, chunk_start * _QKV_LOCAL : (chunk_start + chunk_tiles) * _QKV_LOCAL],
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
        kv_a_norm_w = sb.tile((pre_rows, _KV_LORA), bf16)
        gate_weight_dma(
            nisa.dma_copy(
                dst=kv_a_norm_w,
                src=mla_kv_a_norm.ap(pattern=[[0, pre_rows], [1, _KV_LORA]], offset=0),
                dge_mode=hwdge,
                engine=sync,
                name=f"{tag}_mla_w_kv_a_norm",
            )
        )
        q_b_scales = None
        if head_pre is None:
            if mla_q_b_proj_scale is not None:
                q_b_w, q_b_scales = _load_mx_q_b(mla_q_b_proj, mla_q_b_proj_scale, sb, tag)
            else:
                q_b_w = sb.tile((_P_MAX, _Q_LORA_TILES, _LOCAL_HEADS * _Q_HEAD), bf16)
                gate_weight_dma(
                    nisa.dma_copy(
                        dst=q_b_w,
                        src=mla_q_b_proj,
                        dge_mode=hwdge,
                        engine=sync,
                        name=f"{tag}_mla_w_q_b",
                    )
                )
        else:
            q_b_w = head_pre.q_b
        if mla_kv_b_proj_scale is not None:
            kv_b_w = _load_mx_kv_b(mla_kv_b_proj, mla_kv_b_proj_scale, sb, tag)
            return qkv_chunks, q_a_norm_w, kv_a_norm_w, q_b_w, kv_b_w, q_b_scales
        kv_b_w = sb.tile((_P_MAX, 2 * _ABSORB_WIDTH), bf16)
        gate_weight_dma(
            nisa.dma_copy(
                dst=kv_b_w,
                src=mla_kv_b_proj,
                dge_mode=hwdge,
                engine=sync,
                name=f"{tag}_mla_w_kv_b",
            )
        )
        return qkv_chunks, q_a_norm_w, kv_a_norm_w, q_b_w, kv_b_w, q_b_scales

    qkv_chunks, q_a_norm_w, kv_a_norm_w, q_b_w, kv_b_w, q_b_scales = _shared(shared, "head", load_head_weights)
    # o_proj_early: the ring's first o_proj chunks load with the head weights, before the key loops, and stay resident
    # until the o_proj (no state reduce-scatter gate: that collective has not run yet).
    kernel_assert(not o_proj_early or shared is None, "o_proj_early keeps tiles of this call: no shared stage weights")
    kernel_assert(
        mla_o_proj_scale is None or not (o_proj_early or o_proj_ring), "the MXFP8 o_proj takes no o_proj_ring"
    )
    early_o_chunks = _o_proj_chunks(mla_o_proj, tag, None, sb, True, o_proj_ring) if o_proj_early else None

    # Step 4: the write row of each sequence. owner = ((position >> 7) & 31) == t. The write row is the
    # slot when this rank owns a valid slot, and row 0 of the null page otherwise: page 0 holds finite data
    # that no read trusts. write_mask is all ones or zero, so the AND is exact.
    owner_block = sb.tile((tokens, 1), nl.int32)
    nisa.tensor_scalar(
        dst=owner_block,
        data=token_meta[0:tokens, 0:1],
        op0=nl.right_shift,
        operand0=7,
        op1=nl.bitwise_and,
        operand1=_TP - 1,
    )
    token_f = sb.tile((tokens, 3), fp32)
    nisa.tensor_copy(dst=token_f[0:tokens, 0:1], src=owner_block)
    nisa.tensor_copy(dst=token_f[0:tokens, 1:3], src=token_meta[0:tokens, 1:3])
    owner = sb.tile((tokens, 1), fp32)
    nisa.tensor_scalar(dst=owner, data=token_f[0:tokens, 0:1], op0=nl.equal, operand0=token_f[0:tokens, 2:3])
    write_ok = sb.tile((tokens, 1), fp32)
    nisa.tensor_scalar(
        dst=write_ok,
        data=token_f[0:tokens, 1:2],
        op0=nl.greater_equal,
        operand0=float(_BLOCK),
        op1=nl.multiply,
        operand1=owner,
    )
    write_mask = sb.tile((tokens, 1), nl.int32)
    nisa.tensor_scalar(dst=write_mask, data=write_ok, op0=nl.multiply, operand0=-1.0)
    write_row = sb.tile((tokens, 1), nl.int32)
    nisa.tensor_tensor(dst=write_row, data1=token_meta[0:tokens, 1:2], data2=write_mask, op=nl.bitwise_and)
    if batched_cache_write:
        # Row stage_row + b of the null page for each sequence b that writes no owned slot: one target per row of
        # the layer, so the one write of each group does not depend on the order of its rows.
        null_base = sources.get("stage_row", 0) if sources else 0
        kernel_assert(
            null_base + tokens <= _BLOCK,
            f"batched_cache_write needs at most {_BLOCK} null rows, got {null_base + tokens}",
        )
        null_row = sb.tile((tokens, 1), nl.int32)
        nisa.iota(dst=null_row, pattern=[[0, 1]], offset=null_base, channel_multiplier=1)
        # All ones where the row writes no owned slot, else 0. The write row is 0 where the null row is kept, so
        # the OR is exact for any slot. VectorE writes the index tile that the DMA reads.
        vector = nisa.engine.vector
        null_mask = sb.tile((tokens, 1), nl.int32)
        nisa.tensor_scalar(dst=null_mask, data=write_ok, op0=nl.add, operand0=-1.0, engine=vector)
        nisa.tensor_tensor(dst=null_row, data1=null_row, data2=null_mask, op=nl.bitwise_and, engine=vector)
        nisa.tensor_tensor(dst=write_row, data1=write_row, data2=null_row, op=nl.bitwise_or, engine=vector)

    # Step 5: the key mask inputs on the bias partition. Local slot s = 128 j + o has the global position
    # 4096 j + 128 t + o, so it is a prior key when 4096 j + o < position - 128 t.
    bias_f = sb.tile((_P_MAX, tokens + 1), fp32)
    nisa.tensor_copy(dst=bias_f[_BIAS_ROW : _BIAS_ROW + 1, 0 : tokens + 1], src=bias_meta[_BIAS_ROW : _BIAS_ROW + 1])
    rank_offset = sb.tile((_P_MAX, 1), fp32)
    nisa.tensor_scalar(
        dst=rank_offset[_BIAS_ROW : _BIAS_ROW + 1, 0:1],
        data=bias_f[_BIAS_ROW : _BIAS_ROW + 1, tokens : tokens + 1],
        op0=nl.multiply,
        operand0=-float(_BLOCK),
    )
    mask_threshold = sb.tile((_P_MAX, tokens), fp32)
    nisa.tensor_scalar(
        dst=mask_threshold[_BIAS_ROW : _BIAS_ROW + 1, 0:tokens],
        data=bias_f[_BIAS_ROW : _BIAS_ROW + 1, 0:tokens],
        op0=nl.add,
        operand0=rank_offset[_BIAS_ROW : _BIAS_ROW + 1, 0:1],
    )
    slot_offset = sb.tile((_P_MAX, num_keys), fp32)
    # The loop path writes the slot offsets inside its loop body instead (_dynamic_attention).
    if not mla_runs_loops(local_blocks):
        nisa.iota(
            dst=slot_offset[_BIAS_ROW : _BIAS_ROW + 1, 0:num_keys],
            pattern=[[_GLOBAL_BLOCK_STRIDE, group_blocks], [1, _BLOCK]],
            offset=0,
        )

    # Step 6: the q_a and kv_a projections, x stationary and the weight columns moving.
    x_mx = None
    if mla_in_proj_scale is not None:
        x_mx = quantize_hT_mx(x_hT, tag, padded, lambda shape, dtype, name: sb.tile(shape, dtype, name, keep=True))
    qkv_psum = nl.ndarray((padded if x_mx else pre_rows, _QKV_LOCAL), dtype=fp32, buffer=nl.psum)
    if x_mx:
        _qkv_projection_mx(qkv_psum, x_mx, qkv_chunks[0], padded)
    for chunk_start, chunk_tiles, qkv_w in qkv_chunks if x_mx is None and pre_run else ():
        for chunk_tile in range(chunk_tiles):
            hidden_tile = chunk_start + chunk_tile
            nisa.nc_matmul(
                dst=qkv_psum[0:pre_rows, 0:_QKV_LOCAL],
                stationary=pre_x[0:_P_MAX, 0:pre_rows, hidden_tile],
                moving=qkv_w[0:_P_MAX, chunk_tile, 0:_QKV_LOCAL],
                accumulate=hidden_tile > 0,
            )
    qkv_row = sources["qkv"] if sources and "qkv" in sources else sb.tile((pre_rows, _QKV_LOCAL), bf16)
    if pre_run:
        nisa.tensor_copy(dst=qkv_row, src=qkv_psum[0:pre_rows, 0:_QKV_LOCAL])

    # Step 7: gather q_a | kv_a over TP32 along the free axis: row b holds the 66 values of every rank.
    qkv_all = sources["qkv_dst"] if sources and "qkv_dst" in sources else sb.tile((pre_rows, _TP * _QKV_LOCAL), bf16)
    if sources and "qkv_ag_src_hbm" in sources:
        # The HBM all-gather stacks the ranks on dim 0: row r * B + b is sequence b of rank r.
        if pre_run:
            qkv_src = sources["qkv_ag_src_hbm"] if pre_stage is None else pre_stage["stage"]["qkv_ag_src_hbm"]
            nisa.dma_copy(dst=qkv_src, src=qkv_row, name=f"{tag}_mla_qkv_store")
        yield _Cc(
            "qkv",
            ncc.all_gather,
            dict(
                srcs=[sources["qkv_ag_src_hbm"]],
                dsts=[sources["qkv_ag_dst_hbm"]],
                replica_group=tp_group,
                collective_dim=0,
                name=f"{tag}_mla_qkv_all_gather",
            ),
        )
        # In a staged layer the destination stacks the rows of all groups: B rows of each rank from row offset.
        rows, offset = sources.get("stage_rows", tokens), sources.get("stage_row", 0)
        if pre_run:
            nisa.dma_copy(
                dst=qkv_all.reshape((pre_rows, _TP, _QKV_LOCAL)),
                src=sources["qkv_ag_dst_hbm"].ap(
                    pattern=[[_QKV_LOCAL, pre_rows], [rows * _QKV_LOCAL, _TP], [1, _QKV_LOCAL]],
                    offset=(0 if pre_stage is not None else offset) * _QKV_LOCAL,
                ),
                name=f"{tag}_mla_qkv_load",
            )
    else:
        yield _Cc(
            "qkv",
            ncc.all_gather,
            dict(
                srcs=[qkv_row],
                dsts=[qkv_all],
                replica_group=tp_group,
                collective_dim=1,
                name=f"{tag}_mla_qkv_all_gather",
            ),
        )
    qkv_ranks = qkv_all.reshape_dim(1, (_TP, _QKV_LOCAL))

    # Step 8: while the gather is in flight, transpose the prior keys into feature-major tiles, and
    # write the mask row below the rope keys.
    latent_keys = [sb.tile((_P_MAX, _KV_LORA_TILES, num_keys), bf16) for _ in range(key_buffers)]
    rope_keys = [sb.tile((_P_MAX, num_keys), bf16) for _ in range(key_buffers)]

    def transpose_keys(token):
        buffer = token % key_buffers
        _transpose_group(
            key_rows[buffer],
            latent_keys[buffer],
            rope_keys[buffer],
            slot_offset,
            mask_threshold[_BIAS_ROW : _BIAS_ROW + 1, token : token + 1],
            pair_blocks=local_blocks if pair_pages else 0,
        )

    for token in range(key_buffers):
        transpose_keys(token)

    def static_keys(token):
        """Return the key tiles of ``token`` on the static path; a sequence of the second set loads them now."""
        if token >= key_buffers:
            load_pages(token)
            transpose_keys(token)
        buffer = token % key_buffers
        return latent_keys[buffer], rope_keys[buffer], key_rows[buffer]

    gate_loads = []

    def after_gate_loads(instr):
        """With wgate_early "scalar", order ``instr`` after the gate weight loads in the schedule (no semaphore wait).

        The PSUM read-outs of Steps 12-14 wait for TensorE, and the compiler may put any of them on ScalarE. Without
        the edges it puts some of the loads after them, so ScalarE issues those loads at the query gather.
        """
        if gate_loads:
            handle = instr if hasattr(instr, "depends_on") else nisa.NkiInstruction(instr)
            handle.depends_on(*gate_loads, kind=nisa.dependency.schedule)
        return instr

    def load_gate_weights(after):
        """Load the gate projection weights; in a decode block they follow ``after`` in the schedule.

        With ``wgate_early`` each load also waits for ``after`` on hardware: the issuing engine holds no other wait for
        the gather, so a schedule edge alone lets it trigger the first load while the gather is in flight.
        """
        engine = nisa.engine.scalar if wgate_early == "scalar" else sync
        if mla_in_proj_scale is not None:
            return [_load_mx_gate(mla_in_proj, qkv_chunks[0], sb, tag, engine, after, wgate_early, gate_loads)]
        gate_chunks = []
        for chunk_start in range(0, _HIDDEN_TILES, _GATE_CHUNK_TILES):
            chunk_tiles = min(_GATE_CHUNK_TILES, _HIDDEN_TILES - chunk_start)
            gate_w = sb.tile((_P_MAX, chunk_tiles, _GATE_LOCAL), bf16)
            region_start = _QKV_REGION + chunk_start * _GATE_LOCAL
            dma = gate_weight_dma(
                nisa.dma_copy(
                    dst=gate_w,
                    src=mla_in_proj[0:_P_MAX, region_start : region_start + chunk_tiles * _GATE_LOCAL],
                    dge_mode=hwdge,
                    engine=engine,
                    name=f"{tag}_mla_w_gate_{chunk_start}",
                ),
                after,
            )
            if wgate_early != "off":
                dma.depends_on(after, kind=nisa.dependency.runtime)
            if wgate_early == "scalar":
                gate_loads.append(dma)
            gate_chunks.append((chunk_start, chunk_tiles, gate_w))
        return gate_chunks

    def load_o_weights(after, early=False):
        """Load the o_proj weight chunks (``_o_proj_chunks``) after ``after`` in the schedule (the state
        reduce-scatter). With ``early`` (``o_proj_preload``: ``after`` is the first read of the q_a | kv_a gather
        result) each load also waits for ``after`` on hardware, as the loads of ``wgate_early`` do."""
        if early_o_chunks is not None:
            return early_o_chunks
        return _o_proj_chunks(
            mla_o_proj,
            tag,
            after,
            sb,
            o_proj_stream,
            o_proj_ring,
            early=early,
            **({"mla_o_proj_scale": mla_o_proj_scale} if mla_o_proj_scale is not None else {}),
        )

    # Steps 9 to 11: the latent queries and the current cache rows.
    gate_box = []
    # o_proj_preload: the o_proj weight chunks, loaded at the first read of the q_a | kv_a gather result.
    o_box = []

    def first_read(read):
        if wgate_early != "off":
            gate_box.append(_shared(shared, "gate", lambda: load_gate_weights(read)))
        if o_proj_preload:
            o_box.append(_shared(shared, "o", lambda: load_o_weights(read, early=True)))

    if pre_run:
        q_latent, cur_rows, write_rows = _query_rows(
            qkv_ranks,
            pre_rows,
            q_a_norm_w,
            kv_a_norm_w,
            lora_norm_eps,
            kv_pool.dtype,
            sb,
            first_read=first_read if wgate_early != "off" or o_proj_preload else None,
        )
    if gate_box:
        gate_chunks = gate_box[0]
    elif pre_stage is not None and wgate_early != "off":
        gate_chunks = shared["gate"]
    if pre_stage is not None:
        # stage_pre: the rows of the stage, from the first group; this group's rows from its stage row.
        if pre_run:
            shared["pre_rows"] = (q_latent, cur_rows, write_rows)
        q_latent, stage_cur_rows, stage_write_rows = shared["pre_rows"]
        pre_row = sources["stage_row"]
        cur_rows = stage_cur_rows[pre_row : pre_row + tokens]
        write_rows = stage_write_rows[pre_row : pre_row + tokens]

    # pack_current_after_rs: the current rows of the stage in HBM, for the add after the state reduce-scatter.
    current_hbm = None
    if pack_current_after_rs:
        stage_rows = sources.get("stage_rows", tokens)
        current_hbm = _shared(
            shared,
            "current_rows",
            lambda: nl.ndarray((stage_rows, _KV_LORA), dtype=bf16, buffer=nl.hbm, name=f"{tag}_mla_current_rows"),
        )
        row0 = sources.get("stage_row", 0)
        nisa.dma_copy(
            dst=current_hbm[row0 : row0 + tokens, 0:_KV_LORA],
            src=cur_rows[0:tokens, 0:_KV_LORA],
            dge_mode=hwdge,
            engine=sync,
            name=f"{tag}_mla_current_rows_store",
        )

    # Write the current rows at the write rows, with no branch. The row index stays inside the pool, so
    # the write works on a view of a larger tensor. The writes follow the first page loads, and they read
    # write_rows here, far from the end of the body, so a later inlined body cannot reuse it before the
    # DMAs read it.
    pool_rows = kv_pool.reshape((num_pages * _BLOCK, _CACHE_WIDTH))
    cache_writes = []
    if pair_pages:
        cache_writes = write_pair_rows(kv_pool, write_rows, write_row, tokens, f"{tag}_mla_cache_write", sb)
    if batched_cache_write:
        cache_writes.append(
            nisa.dma_copy(
                dst=pool_rows.ap(
                    pattern=[[_CACHE_WIDTH, tokens], [1, _CACHE_WIDTH]],
                    offset=0,
                    vector_offset=write_row[0:tokens, 0:1],
                    indirect_dim=0,
                ),
                src=write_rows[0:tokens, 0:_CACHE_WIDTH],
                dge_mode=nisa.dge_mode.swdge,
                name=f"{tag}_mla_cache_write",
            )
        )
    for token in range(0 if pair_pages or batched_cache_write else tokens):
        cache_write = nisa.dma_copy(
            dst=pool_rows.ap(
                pattern=[[_CACHE_WIDTH, 1], [1, _CACHE_WIDTH]],
                offset=0,
                scalar_offset=write_row[token : token + 1, 0:1],
                indirect_dim=0,
            ),
            src=write_rows[token : token + 1, 0:_CACHE_WIDTH],
            dge_mode=hwdge,
            engine=sync,
            name=f"{tag}_mla_cache_write_{token}",
        )
        cache_writes.append(cache_write)

    # Steps 12 to 14: the current keys and the query gather source.
    if pre_stage is None:
        cur_keys, q_local = _query_columns(
            cur_rows, q_latent, tokens, q_b_w, kv_b_w, sources, after_gate_loads, sb, mx_query, q_b_scales, tag
        )
    else:
        if pre_run:
            shared["pre_columns"] = _query_columns(
                stage_cur_rows,
                q_latent,
                pre_rows,
                q_b_w,
                kv_b_w,
                sources,
                after_gate_loads,
                sb,
                mx_query,
                q_b_scales,
                tag,
            )
        stage_cur_keys, q_local = shared["pre_columns"]
        cur_keys = stage_cur_keys[0:_P_MAX, 0 : _KV_LORA_TILES + 1, pre_row : pre_row + tokens]

    # Step 15: gather the queries of all 96 heads over CP32. Row 4 r + b of block s of the gather holds the 3
    # heads of rank r for sequence 4 s + b (row B r + b for B <= 4). Transposes put each head's latent tiles
    # and rope query in feature-major columns, with the heads in global order 3r + h.
    gather_rows = _TP * set_rows
    q_all = sources["q_dst"] if sources and "q_dst" in sources else sb.tile((gather_rows, sets * _GATHER_WIDTH), bf16)
    if sources and "q_ag_src_hbm" in sources:
        if pre_run:
            q_src = sources["q_ag_src_hbm"] if pre_stage is None else pre_stage["stage"]["q_ag_src_hbm"]
            nisa.dma_copy(dst=q_src, src=q_local, name=f"{tag}_mla_q_store")
        q_gather = yield _Cc(
            "q",
            ncc.all_gather,
            dict(
                srcs=[sources["q_ag_src_hbm"]],
                dsts=[sources["q_ag_dst_hbm"]],
                replica_group=tp_group,
                collective_dim=0,
                name=f"{tag}_mla_q_all_gather",
            ),
        )
        nisa.dma_copy(dst=q_all, src=sources["q_ag_dst_hbm"], name=f"{tag}_mla_q_load")
    else:
        q_gather = yield _Cc(
            "q",
            ncc.all_gather,
            dict(
                srcs=[q_local], dsts=[q_all], replica_group=tp_group, collective_dim=0, name=f"{tag}_mla_q_all_gather"
            ),
        )

    if wgate_early == "off":
        gate_chunks = _shared(shared, "gate", lambda: load_gate_weights(q_gather))
    # loop_pad_stationary: one spare sequence slot at the end of each query tile, so a 128-column stationary of the last
    # sequence stays in its tile (zeroed once; its columns reach only PSUM rows 96-127, which nothing reads).
    query_slots = tokens + 1 if loop_pad_stationary else tokens
    q_nope_all = sb.tile((_P_MAX, query_slots, _KV_LORA_TILES, _HEADS), bf16)
    q_rope_all = sb.tile((_P_MAX, query_slots, _HEADS), bf16)
    if loop_pad_stationary:
        nisa.memset(dst=q_nope_all[0:_P_MAX, tokens, 0:_KV_LORA_TILES, 0:_HEADS], value=0.0, engine=nisa.engine.gpsimd)
        nisa.memset(dst=q_rope_all[0:_P_MAX, tokens, 0:_HEADS], value=0.0, engine=nisa.engine.gpsimd)
    nisa.memset(dst=q_rope_all[_BIAS_ROW : _BIAS_ROW + 1, 0:tokens, 0:_HEADS], value=1.0)
    for query_set in range(sets):
        _set_queries(q_all, query_set, tokens, q_nope_all, q_rope_all)
    # loop_dr_keys: the double_row query stationaries of every sequence live through the key loops.
    loop_dr_rows = _loop_dr_query_rows(q_all, tokens, sb) if loop_dr_keys else None

    # Step 16: the softmax reference of each sequence, and the operands that add 1/32 of the current key to
    # every head: a row of the per-rank current weight and the current row of the sequence, on partition
    # 32 (b % 4). The capped softmax gives the current key the weight 2**-93; the exact softmax starts each head
    # with the current key as its maximum, weight 1.
    softmax = _Softmax(exact_softmax, tokens, sb)
    # With pack_mx_keys "single", pack_dr_keys or loop_dr_keys the rope queries (in place) and the current score take the
    # e4m3 query too, as the GPU decode rounds its whole query; the e4m3 latent query tiles live only for the current score.
    mx_single = pack_mx_keys == "single" or pack_dr_keys or loop_dr_keys
    with sb.scope("mx_reference") if mx_single else nullcontext():
        q_reference = _e4m3_query_rounding(q_nope_all, q_rope_all, sb) if mx_single else q_nope_all
        for token in range(tokens):
            _reference(token, q_reference, q_rope_all, cur_keys, softmax)
    # pack_current_after_rs adds the current key after the state reduce-scatter (``_add_current_rows``).
    current = None if pack_current_after_rs else _CurrentKey(cur_rows, tokens, softmax.rank_current_weight, tag, sb)

    # Step 17: the softmax state [numerator | sum] of each sequence, which starts with 1/32 of its current
    # key and the weight of that key. The static path attends every local block. The dynamic path attends
    # only the key groups that hold a prior key. The capped static path writes the state of the
    # reduce-scatter directly; the other paths accumulate in FP32.
    state_dtype = bf16 if bf16_state else fp32
    # The state reduce-scatter destination takes its SBUF before the key groups, so it shares no bytes with
    # their per-group scratch (weights/weights_t). Otherwise a fast CP peer's reduce-scatter write can land
    # in this rank's local_state while a slow rank (one extra loop iteration at an iteration crossing) still
    # uses that SBUF as loop scratch. The BF16-o state reduce-scatters into HBM and loads its own parts after it, so it
    # needs no such tile.
    local_state = None if state_bf16_o else sb.tile((_LOCAL_HEADS, tokens, _KV_LORA + 1), state_dtype)
    if not dynamic and not exact_softmax:
        rs_state = sb.tile((_HEADS, tokens, _KV_LORA + 1), state_dtype)
        prior_sum = sb.tile((_HEADS, tokens), fp32)
        for token in range(tokens):
            numerator_psum = _attend_group(
                token,
                q_nope_all,
                q_rope_all,
                *static_keys(token),
                softmax,
                prior_sum[0:_HEADS, token : token + 1],
                sb,
                pair_blocks=local_blocks if pair_pages else 0,
            )
            current.add(numerator_psum, token, accumulate=True)
            nisa.tensor_copy(dst=rs_state[0:_HEADS, token, 0:_KV_LORA], src=numerator_psum)
            nisa.tensor_scalar(
                dst=rs_state[0:_HEADS, token, _KV_LORA : _KV_LORA + 1],
                data=prior_sum[0:_HEADS, token : token + 1],
                op0=nl.add,
                operand0=softmax.rank_current_weight,
            )
    else:
        packed = packed_loop and local_blocks > _MAX_UNROLLED_BLOCKS
        # A pair of staged groups shares one state: the columns of half h start at h x B, the null items at 2B.
        first_column = 0 if pack_pair is None else pack_pair["half"] * tokens
        state_tokens = tokens if pack_pair is None else pack_pair["tokens"]
        # The packed loop adds its null items to column B, through TI on the 2D storage of the state.
        pack_sb = sb if pack_sbuf is None else pack_sbuf
        state_flat = (
            _shared(
                shared,
                "pack_state",
                lambda: _ti_tile(
                    pack_sb, (_HEADS, (state_tokens + 1) * (_KV_LORA + 1)), fp32, f"{tag}_mla_pack_state", 0
                ),
            )
            if packed
            else None
        )
        state = (
            state_flat.reshape((_HEADS, state_tokens + 1, _KV_LORA + 1))
            if packed
            else sb.tile((_HEADS, tokens, _KV_LORA + 1), fp32)
        )

        def start_state():
            """Start the state of the sequences of this group with their current keys (and the null column)."""
            # The ops keep their default engines: without pack_setup_ahead this is the program of the base.
            if packed and first_column == 0:
                nisa.memset(dst=state[0:_HEADS, state_tokens, 0 : _KV_LORA + 1], value=0.0)
            if pack_current_after_rs:
                # The state starts at 0: the current key adds after the reduce-scatter.
                nisa.memset(
                    dst=state[0:_HEADS, first_column : first_column + tokens, 0 : _KV_LORA + 1],
                    value=0.0,
                    engine=nisa.engine.gpsimd,
                )
                return
            for token in range(tokens):
                current_psum = nl.ndarray((_HEADS, _KV_LORA), dtype=fp32, buffer=nl.psum)
                current.add(current_psum, token, accumulate=False)
                nisa.tensor_copy(dst=state[0:_HEADS, first_column + token, 0:_KV_LORA], src=current_psum)
            nisa.memset(
                dst=state[0:_HEADS, first_column : first_column + tokens, _KV_LORA : _KV_LORA + 1],
                value=softmax.rank_current_weight,
            )

        # pack_setup_ahead: the state starts after the setup point of the packed loop (``_packed_key_loop_steps``).
        if not pack_setup_ahead:
            start_state()
        if not dynamic:
            group_sum = sb.tile((_HEADS, 1), fp32)
            for token in range(tokens):
                numerator_psum = _attend_group(
                    token,
                    q_nope_all,
                    q_rope_all,
                    *static_keys(token),
                    softmax,
                    group_sum,
                    sb,
                    pair_blocks=local_blocks if pair_pages else 0,
                )
                softmax.update(state, token, numerator_psum, group_sum)
        elif packed:
            # The loop tiles get their own scope at every B, so the o_proj after the loop has their bytes.
            with sb.scope("keys"):
                steps = _packed_key_loop_steps(
                    kv_pool,
                    block_table,
                    token_meta,
                    token_f,
                    q_nope_all,
                    q_rope_all,
                    softmax,
                    slot_offset,
                    state_flat,
                    tokens,
                    local_blocks,
                    tag,
                    sb,
                    pack_plan,
                    pack_slots,
                    tail_slots=pack_tail_slots,
                    tail_round=pack_tail_round,
                    clamp_pages=not zeroed_pages,
                    pack_sb=pack_sb,
                    shared=shared,
                    pack_hbm=sources.get("mla_pack_hbm") if sources else None,
                    pair_pages=pair_pages,
                    prefetch=pack_prefetch,
                    row_dma_act=pack_row_dma_act,
                    row_dma_pool=pack_row_dma_pool,
                    page_runs=pack_page_runs,
                    null_copy_vector=pack_null_copy_vector,
                    null_runs=pack_null_runs,
                    key_transpose=pack_key_transpose,
                    latent_act=pack_latent_act,
                    mask_column=pack_mask_column,
                    **({"mask_pool": True} if pack_mask_pool else {}),
                    bg_rope=pack_bg_rope,
                    bg_interleave=pack_bg_interleave,
                    carry=pack_carry,
                    carry_main_only=pack_carry_main_only,
                    sum_after_copy=pack_sum_after_copy,
                    rope_ahead=pack_rope_ahead,
                    pair=pack_pair,
                    **({"setup_ahead": True, "after_setup": start_state} if pack_setup_ahead else {}),
                    lookahead=pack_lookahead,
                    **({"psum_per_block": True} if pack_psum_per_block else {}),
                    **({"step_items": pack_step_items} if pack_step_items is not None else {}),
                    **({"stage_rows_sbuf": True} if pack_stage_rows_sbuf else {}),
                    **({"mx_keys": pack_mx_keys, "mx_q_all": q_all} if pack_mx_keys is not None else {}),
                    **({"dr_keys": True, "mx_q_all": q_all} if pack_dr_keys else {}),
                )

                for _ in steps:
                    # pack_setup_ahead: the setup point of the loop (a sync site, no collective).
                    yield _Cc(_SETUP_SITE, None, {})
                if pack_setup_ahead and pack_pair is not None and not pack_pair["half"]:
                    # Half 0 of a pair returns from the loop call once its slot rows are stored.
                    yield _Cc(_SETUP_SITE, None, {})
                    start_state()
        else:
            _dynamic_attention(
                kv_pool,
                block_table,
                token_meta,
                token_f,
                q_nope_all,
                q_rope_all,
                softmax,
                slot_offset,
                state,
                tokens,
                local_blocks,
                tag,
                sb,
                clamp_pages=not zeroed_pages,
                pair_pages=pair_pages,
                dr_rows=loop_dr_rows,
                **({"cast_engines": loop_cast_engines} if loop_cast_engines != "off" else {}),
                **({"pad_stationary": True} if loop_pad_stationary else {}),
                **({"restart": loop_restart} if loop_restart != "off" else {}),
                **({"groups": loop_groups} if loop_groups else {}),
                **({"page_gather": loop_page_gather} if loop_page_gather != "off" else {}),
                **({"inputs_ahead": True} if loop_inputs_ahead else {}),
            )
        if exact_softmax:
            softmax.merge_scale(state, tp_group, tag)
        rs_state = state[0:_HEADS, first_column : first_column + tokens, 0 : _KV_LORA + 1] if packed else state
        if bf16_state:
            rs_state = sb.tile((_HEADS, tokens, _KV_LORA + 1), state_dtype)
            nisa.tensor_copy(dst=rs_state, src=state[0:_HEADS, 0:tokens, 0 : _KV_LORA + 1])

    if state_bf16_o:
        # The BF16-o form: the numerators reduce-scatter in BF16 and the sums in FP32, in the views of the FP32 pair.
        kernel_assert(
            not bf16_state
            and pack_pair is None
            and not stage_post
            and not pack_current_after_rs
            and pack_state_store_queues <= 1,
            "state_bf16_o is not with bf16_state, pack_pair, stage_post, pack_current_after_rs or "
            "pack_state_store_queues",
        )
        state_reduce_scatter, local_state = yield from _state_rs_bf16_o(rs_state, sources, tp_group, tokens, sb, tag)
        post_stage = post = None
        stage_tokens, stage_row = tokens, 0
        first_group = last_group = True
    else:
        # Step 18: reduce-scatter the state by heads, then latent = numerator / sum. With persistent HBM tensors the
        # reduce-scatter runs in HBM: an SBUF reduce-scatter gets a compiler scratch (the runtime writes it on Trn2 only).
        rs_src = rs_state.reshape((_HEADS, tokens * (_KV_LORA + 1)))
        rs_dst = local_state.reshape((_LOCAL_HEADS, tokens * (_KV_LORA + 1)))
        state_hbm = sources.get("state_hbm") if sources and state_dtype == fp32 else None
        if state_hbm is not None:
            if pack_state_store_queues > 1 and (pack_pair is None or pack_pair["half"]):
                # The state store on pack_state_store_queues DMA queues (the state of both halves with a pair).
                store_tokens = tokens if pack_pair is None else pack_pair["tokens"]
                store_src = (
                    rs_src
                    if pack_pair is None
                    else state[0:_HEADS, 0:store_tokens, 0 : _KV_LORA + 1].reshape(
                        (_HEADS, store_tokens * (_KV_LORA + 1))
                    )
                )
                _store_state(
                    state_hbm if pack_pair is None else pack_pair["state_hbm"],
                    store_src,
                    store_tokens,
                    pack_state_store_queues,
                    f"{tag}_mla_state_store",
                )
            elif pack_pair is None:
                nisa.dma_copy(dst=state_hbm, src=rs_src, name=f"{tag}_mla_state_store")
            elif pack_pair["half"]:
                # The loop of the pair ran in this half: store the states of both halves.
                pair_tokens = pack_pair["tokens"]
                nisa.dma_copy(
                    dst=pack_pair["state_hbm"],
                    src=state[0:_HEADS, 0:pair_tokens, 0 : _KV_LORA + 1].reshape(
                        (_HEADS, pair_tokens * (_KV_LORA + 1))
                    ),
                    name=f"{tag}_mla_state_store",
                )
            rs_src, rs_dst = state_hbm, sources["state_dst_hbm"]
        state_reduce_scatter = yield _Cc(
            "state",
            ncc.reduce_scatter,
            dict(
                srcs=[rs_src],
                dsts=[rs_dst],
                replica_group=tp_group,
                collective_dim=0,
                op=nl.add,
                name=f"{tag}_mla_state_reduce_scatter",
            ),
        )
        # stage_post: the work after the reduce-scatter runs once for the S sequences of the stage (see the docstring).
        post_stage = sources.get("post_stage") if stage_post and sources else None
        kernel_assert(
            not stage_post
            or (
                post_stage is not None
                and state_hbm is not None
                and sources.get("out_rs_src_hbm") is not None
                and sources.get("out_in_hbm")
                and not o_proj_stream
            ),
            "stage_post needs a staged MLA group (sources post_stage) with the FP32 HBM state, the HBM out reduce-scatter "
            "and no o_proj_stream",
        )
        stage_tokens = sources["stage_rows"] if post_stage is not None else tokens
        stage_row = sources["stage_row"] if post_stage is not None else 0
        first_group, last_group = stage_row == 0, stage_row + tokens == stage_tokens
        post = None
        if post_stage is not None:
            post = _shared(
                shared,
                "post",
                lambda: {
                    "gate": sb.tile((_P_MAX, _LOCAL_HEADS, stage_tokens), bf16),
                    "value": sb.tile((_P_MAX, _LOCAL_HEADS, stage_tokens), bf16),
                },
            )
            if first_group:
                # Row h S + b holds the merged state of local head h for sequence b of the stage.
                state_rows = sb.tile((_LOCAL_HEADS * stage_tokens, _KV_LORA + 1), fp32)
                row_width = stage_tokens * (_KV_LORA + 1)
                for local_head in range(_LOCAL_HEADS):
                    nisa.dma_copy(
                        dst=state_rows[local_head * stage_tokens : (local_head + 1) * stage_tokens, 0 : _KV_LORA + 1],
                        src=post_stage["state_dst_hbm"].ap(
                            pattern=[[_KV_LORA + 1, stage_tokens], [1, _KV_LORA + 1]], offset=local_head * row_width
                        ),
                        name=f"{tag}_mla_state_load_{local_head}",
                    )
                if pack_current_after_rs:
                    _add_current_rows(state_rows, current_hbm, stage_tokens, True, tag, sb)
        elif state_hbm is not None:
            nisa.dma_copy(
                dst=local_state.reshape((_LOCAL_HEADS, tokens * (_KV_LORA + 1))),
                src=rs_dst,
                name=f"{tag}_mla_state_load",
            )
            if pack_current_after_rs:
                row0 = sources.get("stage_row", 0)
                _add_current_rows(local_state, current_hbm[row0 : row0 + tokens, 0:_KV_LORA], tokens, False, tag, sb)
    gate_psum = _gate_projection(x_hT, gate_chunks, tokens, x_mx)
    if post is not None:
        value_all = post["value"]
        if first_group:
            _value_projection_rows(state_rows, kv_b_w, value_all, stage_tokens, sb)
    else:
        value_all = sb.tile((_P_MAX, _LOCAL_HEADS, tokens), bf16)
        _value_projection(local_state, kv_b_w, value_all, tokens, sb)

    kernel_assert(not o_proj_ring or shared is None, "o_proj_ring reuses its chunk tiles: no shared stage weights")

    # With o_proj_stream each weight chunk loads just before its matmuls; the accumulation order is the same.
    if o_box:
        o_w_chunks = o_box[0]
    else:
        o_w_chunks = _shared(shared, "o", lambda: load_o_weights(state_reduce_scatter))

    # Step 19: o = value * sigmoid(g), with sigmoid(g) = 1 / (1 + exp(-g)) so ScalarE stays on its exp table.
    gate_bf16 = sb.tile((_P_MAX, _LOCAL_HEADS, tokens), bf16)
    nisa.tensor_copy(dst=gate_bf16, src=gate_psum)
    gate_exp = sb.tile((_P_MAX, _LOCAL_HEADS, tokens), fp32)
    nisa.activation(dst=gate_exp, op=nl.exp, data=gate_bf16, scale=-1.0)
    nisa.tensor_scalar(dst=gate_exp, data=gate_exp, op0=nl.add, operand0=1.0)
    if post is not None:
        gate = post["gate"][0:_P_MAX, 0:_LOCAL_HEADS, stage_row : stage_row + tokens]
    else:
        gate = sb.tile((_P_MAX, _LOCAL_HEADS, tokens), bf16)
    nisa.reciprocal(dst=gate, data=gate_exp)
    # With stage_post only the last group runs o_proj, on the S sequences of the stage.
    out_tokens = stage_tokens if post is not None else tokens
    partial = None
    if post is None or last_group:
        gated = sb.tile((_P_MAX, _LOCAL_HEADS, out_tokens), bf16)
        nisa.tensor_tensor(dst=gated, data1=value_all, data2=gate if post is None else post["gate"], op=nl.multiply)

        # Step 20: the o_proj partial in hT layout, the weight tiles stationary. One PSUM tile holds the hidden
        # tiles that fit one bank.
        partial = _o_proj_partial(
            o_w_chunks, gated, out_tokens, mla_o_proj, tag, state_reduce_scatter, sb, o_proj_stream
        )

    # Step 21: sum the BF16 partials over TP32. A reduce-scatter over the partitions and an all-gather
    # replace the all-reduce: an SBUF all-reduce over many partitions is not reliable.
    partial_flat = None if partial is None else partial.reshape((_P_MAX, out_tokens * _HIDDEN_TILES))
    out_rs_src = sources.get("out_rs_src_hbm") if sources else None
    if out_rs_src is not None:
        # In SBUF the out reduce-scatter scratch can share a 32-byte block with an engine write, so it runs in HBM.
        rs_dst, ag_dst = sources["out_rs_dst_hbm"], sources.get("out_ag_dst_hbm")
        y_hT = None if sources.get("out_in_hbm") else _out_tile(sources, sb, tokens)
        if post is None:
            nisa.dma_copy(dst=out_rs_src, src=partial_flat, name=f"{tag}_mla_out_store")
        elif last_group:
            # The partial of every group of the stage: the stage's reduce-scatter source.
            nisa.dma_copy(dst=post_stage["out_rs_src_hbm"], src=partial_flat, name=f"{tag}_mla_out_store")
        release_weight_gate(
            (
                yield _Cc(
                    "out_rs",
                    ncc.reduce_scatter,
                    dict(
                        srcs=[out_rs_src],
                        dsts=[rs_dst],
                        replica_group=tp_group,
                        collective_dim=0,
                        op=nl.add,
                        name=f"{tag}_mla_out_reduce_scatter",
                    ),
                )
            )
        )
        if ag_dst is None:
            # Verify: load the reduce-scatter result, then all-gather in SBUF into the persistent mla_out_dst.
            kernel_assert(
                y_hT is not None,
                "out_in_hbm needs the HBM out all-gather destination out_ag_dst_hbm; the SBUF out all-gather has no tile",
            )
            partial_sum = sources["out_gather"]
            nisa.dma_copy(
                dst=partial_sum.reshape((_P_MAX // _TP, tokens * _HIDDEN_TILES)),
                src=rs_dst,
                name=f"{tag}_mla_out_load",
            )
            release_weight_gate(
                (
                    yield _Cc(
                        "out_ag",
                        ncc.all_gather,
                        dict(
                            srcs=[partial_sum.reshape((_P_MAX // _TP, tokens * _HIDDEN_TILES))],
                            dsts=[y_hT.reshape((_P_MAX, tokens * _HIDDEN_TILES))],
                            replica_group=tp_group,
                            collective_dim=0,
                            name=f"{tag}_mla_out_all_gather",
                        ),
                    )
                )
            )
        else:
            release_weight_gate(
                (
                    yield _Cc(
                        "out_ag",
                        ncc.all_gather,
                        dict(
                            srcs=[rs_dst],
                            dsts=[ag_dst],
                            replica_group=tp_group,
                            collective_dim=0,
                            name=f"{tag}_mla_out_all_gather",
                        ),
                    )
                )
            )
            # The caller loads the output from HBM after the last group (``out_in_hbm``, see ``decode_block``).
            if y_hT is None:
                return ag_dst
            nisa.dma_copy(dst=y_hT.reshape((_P_MAX, tokens * _HIDDEN_TILES)), src=ag_dst, name=f"{tag}_mla_out_load")
        object.__setattr__(y_hT, "side_effects", tuple(cache_writes))
        return y_hT
    partial_sum = sb.tile((_P_MAX // _TP, tokens, _HIDDEN_TILES), bf16) if sources is None else sources["out_gather"]
    release_weight_gate(
        (
            yield _Cc(
                "out_rs",
                ncc.reduce_scatter,
                dict(
                    srcs=[partial_flat],
                    dsts=[partial_sum.reshape((_P_MAX // _TP, tokens * _HIDDEN_TILES))],
                    replica_group=tp_group,
                    collective_dim=0,
                    op=nl.add,
                    name=f"{tag}_mla_out_reduce_scatter",
                ),
            )
        )
    )
    y_hT = (
        sources["mla_out_dst"]
        if sources and "mla_out_dst" in sources
        else sb.tile((_P_MAX, tokens, _HIDDEN_TILES), bf16)
    )
    release_weight_gate(
        (
            yield _Cc(
                "out_ag",
                ncc.all_gather,
                dict(
                    srcs=[partial_sum.reshape((_P_MAX // _TP, tokens * _HIDDEN_TILES))],
                    dsts=[y_hT.reshape((_P_MAX, tokens * _HIDDEN_TILES))],
                    replica_group=tp_group,
                    collective_dim=0,
                    name=f"{tag}_mla_out_all_gather",
                ),
            )
        )
    )
    object.__setattr__(y_hT, "side_effects", tuple(cache_writes))
    return y_hT


class MlaHeadTiles:
    """The BF16 qkv and q_b weight tiles of one MLA call whose loads a caller issued before the body.

    ``prefetch_mla_head`` builds it. ``qkv_chunks``: ``(first hidden tile, tiles, tile)`` for each chunk of the qkv
    region of ``mla_in_proj``, as the body loads them; ``q_b``: the ``mla_q_b_proj`` tile; ``dmas``: the loads, in
    issue order. The body takes it as ``head_prefetch``.
    """

    def __init__(self, qkv_chunks: tuple, q_b: nl.NkiTensor, dmas: tuple):
        self.qkv_chunks = qkv_chunks
        self.q_b = q_b
        self.dmas = dmas


def prefetch_mla_head(
    mla_in_proj: nl.NkiTensor, mla_q_b_proj: nl.NkiTensor, tag: str, after: tuple = (), wait: tuple = ()
) -> MlaHeadTiles:
    """Issue the qkv and q_b weight loads of one BF16 MLA call ahead of its body; return the tiles.

    ``mla_in_proj`` and ``mla_q_b_proj`` are the layer's HBM weights (``MLA_WEIGHT_NAMES``). The loads take the tiles,
    queue (SP hardware DGE) and DMA form of the body's own loads, so the body reads the same bytes; the gate region of
    ``mla_in_proj``, the norms and ``mla_kv_b_proj`` stay with the body. Each load follows the instructions ``after``
    and the load before it in the schedule (the compiler can neither hoist a load above them nor reorder the loads on
    the in-order queue) and waits for the instructions ``wait`` (a semaphore wait). The loads are not weight-gated
    (``tkg_weight_dma_gate``): the caller picks the place. Give the result to the body of the same call (``head_prefetch``). The
    output of the body does not change.

    The tiles are compiler-placed: they must be read before the first loop of the body (the qkv and query projections
    are), so no tile lives across a basic-block boundary (``tkg_sbuf_alloc``).
    """
    kernel_assert(
        tuple(mla_in_proj.shape) == (_P_MAX, _HIDDEN_TILES * (_QKV_LOCAL + _GATE_LOCAL))
        and tuple(mla_q_b_proj.shape) == (_P_MAX, _Q_LORA_TILES * _LOCAL_HEADS * _Q_HEAD),
        f"prefetch_mla_head takes the BF16 mla_in_proj and mla_q_b_proj of one rank, got {tuple(mla_in_proj.shape)} and "
        f"{tuple(mla_q_b_proj.shape)}",
    )
    hwdge, sync, bf16 = nisa.dge_mode.hwdge, nisa.engine.sync, nl.bfloat16
    chunks, loads = [], []

    def place(dma):
        handle = dma if hasattr(dma, "depends_on") else nisa.NkiInstruction(dma)
        # In issue order on the in-order queue: after the caller's instructions and after the load before it.
        before = list(after) + [
            load if hasattr(load, "depends_on") else nisa.NkiInstruction(load) for load in loads[-1:]
        ]
        if before:
            handle.depends_on(*before, kind=nisa.dependency.schedule)
        if wait:
            handle.depends_on(*wait, kind=nisa.dependency.runtime)
        loads.append(dma)

    for chunk_start in range(0, _HIDDEN_TILES, _QKV_CHUNK_TILES):
        chunk_tiles = min(_QKV_CHUNK_TILES, _HIDDEN_TILES - chunk_start)
        qkv_w = nl.ndarray((_P_MAX, chunk_tiles, _QKV_LOCAL), dtype=bf16, buffer=nl.sbuf)
        place(
            nisa.dma_copy(
                dst=qkv_w,
                src=mla_in_proj[0:_P_MAX, chunk_start * _QKV_LOCAL : (chunk_start + chunk_tiles) * _QKV_LOCAL],
                dge_mode=hwdge,
                engine=sync,
                name=f"{tag}_mla_w_qkv_prefetch_{chunk_start}",
            )
        )
        chunks.append((chunk_start, chunk_tiles, qkv_w))
    q_b_w = nl.ndarray((_P_MAX, _Q_LORA_TILES, _LOCAL_HEADS * _Q_HEAD), dtype=bf16, buffer=nl.sbuf)
    place(nisa.dma_copy(dst=q_b_w, src=mla_q_b_proj, dge_mode=hwdge, engine=sync, name=f"{tag}_mla_w_q_b_prefetch"))
    return MlaHeadTiles(tuple(chunks), q_b_w, tuple(loads))


def _shared(shared, key, load):
    """Return ``load()``, once per staged layer when ``shared`` is a dict (the groups share the weight tiles)."""
    if shared is None:
        return load()
    if key not in shared:
        shared[key] = load()
    return shared[key]


_STATE_STORE_QUEUES = (
    {"engine": nisa.engine.scalar, "dge_mode": nisa.dge_mode.hwdge},
    {"engine": nisa.engine.sync, "dge_mode": nisa.dge_mode.hwdge},
    {"dge_mode": nisa.dge_mode.swdge},
)
"""The DMA queues of the parts of the state store with ``pack_state_store_queues`` (``_store_state``), in order. The third
is the GpSimd software DGE queue: ``nisa.dma_copy`` takes no engine with ``dge_mode.swdge``."""


def _store_state(dst, src, tokens: int, queues: int, name: str):
    """Store the FP32 state ``src`` ``[96, tokens x 513]`` to ``dst`` as ``queues`` DMAs of consecutive sequence blocks,
    each on all 96 partitions, on the first ``queues`` of ``_STATE_STORE_QUEUES``."""
    bounds = [tokens * part // queues for part in range(queues + 1)]
    for part in range(queues):
        first, last = bounds[part] * _STATE_WIDTH, bounds[part + 1] * _STATE_WIDTH
        nisa.dma_copy(
            dst=dst[0:_HEADS, first:last],
            src=src[0:_HEADS, first:last],
            name=f"{name}_{part}",
            **_STATE_STORE_QUEUES[part],
        )


_SETUP_SITE = "setup"
"""The sync site of ``pack_setup_ahead``: a ``_Cc`` without a collective (``MLA_SYNC_SITES``)."""
MLA_SYNC_SITES = (_SETUP_SITE,)
"""The sites of the MLA body that issue no collective: the groups of a staged layer only meet there."""


_STATE_SUM_ROWS = 16
"""The values a head of the padded sums of the BF16-o decode state (``mla_state_bf16_o_views`` ``sum_rows``)."""


def mla_state_sum_hbm(tokens: int, name: str = "") -> dict:
    """Return the FP32 HBM pair of the BF16-o sums of a decode MLA call of ``tokens`` sequences: ``state_sum_hbm``
    ``[96, S]`` and ``state_sum_dst_hbm`` ``[3, S]``, S = max(tokens, 16), so the sums reduce-scatter writes a tensor of
    its own and moves at least 192 B a rank. ``name`` tells the tensors apart."""
    cols = max(tokens, _STATE_SUM_ROWS)
    return {
        key: nl.ndarray((heads, cols), dtype=nl.float32, buffer=nl.private_hbm, name=f"mla{name}_{key}")
        for key, heads in (("state_sum_hbm", _HEADS), ("state_sum_dst_hbm", _LOCAL_HEADS))
    }


def _state_rs_bf16_o(rs_state, sources, tp_group, tokens, sb, tag):
    """The BF16-o state reduce-scatter of a decode layer (``state_bf16_o``), as a generator of ``_mla_decode_body``.

    The FP32 state ``[96, tokens, 513]`` goes to the BF16-o views of the HBM pair: BF16 numerators and FP32 sums, each
    reduce-scattered by heads at the site ``state``, the two collectives issued together. Return the numerator
    reduce-scatter and the local ``StateParts`` ``[3, tokens, 512]`` / ``[3, tokens, 1]``.
    """
    kernel_assert(
        sources is not None and sources.get("state_hbm") is not None and rs_state.dtype == nl.float32,
        "state_bf16_o needs the FP32 HBM state pair in sources",
    )
    # The sums of a decode layer go padded to _STATE_SUM_ROWS values a head (192 B a rank at TP32, the verify's size):
    # 4 sequences would make a 48 B a rank reduce-scatter.
    # With the pair of mla_state_sum_hbm in the sources, the sums take tensors of their own, so the two reduce-scatters
    # never write views of one tensor.
    o_src, sum_src, view_rows = mla_state_bf16_o_views(
        sources["state_hbm"], _KV_LORA, _STATE_SUM_ROWS, sums=sources.get("state_sum_hbm")
    )
    o_dst, sum_dst, _ = mla_state_bf16_o_views(
        sources["state_dst_hbm"], _KV_LORA, _STATE_SUM_ROWS, sums=sources.get("state_sum_dst_hbm")
    )
    kernel_assert(view_rows >= tokens, f"the state pair holds {view_rows} rows, the layer {tokens}")
    sum_rows = sum_src.shape[1]
    store_state_bf16_o(
        rs_state,
        state_rows_plain(o_src, view_rows, _HEADS, _KV_LORA),
        state_rows_plain(sum_src, sum_rows, _HEADS, 0),
        _KV_LORA,
        sb,
        f"{tag}_mla_state_store",
    )

    def both(name):
        o_rs = ncc.reduce_scatter(
            srcs=[o_src], dsts=[o_dst], replica_group=tp_group, collective_dim=0, op=nl.add, name=name
        )
        ncc.reduce_scatter(
            srcs=[sum_src], dsts=[sum_dst], replica_group=tp_group, collective_dim=0, op=nl.add, name=f"{name}_sum"
        )
        return o_rs

    o_rs = yield _Cc("state", both, {"name": f"{tag}_mla_state_reduce_scatter"})
    parts = load_state_bf16_o(
        o_dst, sum_dst, view_rows, lambda h: h, _LOCAL_HEADS, tokens, _KV_LORA, False, sb, f"{tag}_mla_state", sum_rows
    )
    return o_rs, parts


class _Cc:
    """One collective of the MLA body: ``fn(**kwargs)``, at the sync point ``site`` of a staged layer."""

    def __init__(self, site: str, fn, kwargs: dict):
        self.site, self.fn, self.kwargs = site, fn, kwargs


@functools.wraps(_mla_decode_body)
def mla_decode_attention(*args, **kwargs) -> nl.NkiTensor:
    body = _mla_decode_body(*args, **kwargs)
    try:
        cc = next(body)
        while True:
            cc = body.send(None if cc.site in MLA_SYNC_SITES else cc.fn(**cc.kwargs))
    except StopIteration as done:
        return done.value


mla_decode_attention.token_major = True
"""The body takes and returns token-major ``[128, B, 56]`` hT tiles (see ``decode_block``)."""
mla_decode_attention.stages = _mla_decode_body
"""The generator form: it yields each collective (``_Cc``), so a caller can run the groups of a layer in lockstep
and issue one collective for all of them (``decode_block``)."""


def _out_tile(sources, sb, tokens):
    """Return the SBUF tile of the MLA output: the persistent source tile, else a tile of this layer."""
    if "mla_out_dst" in sources:
        return sources["mla_out_dst"]
    return sb.tile((_P_MAX, tokens, _HIDDEN_TILES), nl.bfloat16)


def mla_collective_sources(
    tokens: int,
    sb=AUTO_SBUF,
    static: bool = True,
    name_prefix: str = "",
    q_hbm: bool = False,
) -> dict:
    """Return the persistent SBUF tiles of the MLA collectives (see ``decode_block.collective_sources``).

    ``static`` adds the tiles of the input gathers and the output all-gather destination, for MLA layers without
    loops. The query all-gather source is persistent on both paths: a peer can read it after the local completion
    of the gather, and a per-call tile can give its bytes to a later tile that waits for no collective of the
    group. The next writer of the persistent tile is the query of the next MLA layer, which follows the state
    reduce-scatter and the output collectives of this layer on the same group. ``name_prefix`` makes the tile names
    unique when a kernel keeps one tile set per MLA group of a layer (``decode_block`` runs more than 8 sequences as
    groups of 8, each with its own collective tiles).
    ``q_hbm`` (verify parity sets only) drops the q input-gather tiles, which then run in HBM and keep their
    SBUF source and destination transient (freed per layer) instead of in the collective arenas.
    """
    out_gather = sb.tile((_P_MAX // _TP, tokens, _HIDDEN_TILES), nl.bfloat16, f"{name_prefix}mla_out_gather_src")
    sources = {"out_gather": out_gather}
    sets, set_rows = -(-tokens // QUERY_SET_TOKENS), min(tokens, QUERY_SET_TOKENS)
    if not static:
        sources["q"] = sb.tile((set_rows, sets * _GATHER_WIDTH), nl.bfloat16, f"{name_prefix}mla_q_src")
    if static:
        sources["qkv"] = sb.tile((tokens, _QKV_LOCAL), nl.bfloat16, f"{name_prefix}mla_qkv_src")
        sources["qkv_dst"] = sb.tile((tokens, _TP * _QKV_LOCAL), nl.bfloat16, f"{name_prefix}mla_qkv_dst")
        if not q_hbm:
            sources["q"] = sb.tile((set_rows, sets * _GATHER_WIDTH), nl.bfloat16, f"{name_prefix}mla_q_src")
            sources["q_dst"] = sb.tile((_TP * set_rows, sets * _GATHER_WIDTH), nl.bfloat16, f"{name_prefix}mla_q_dst")
        sources["mla_out_dst"] = sb.tile((_P_MAX, tokens, _HIDDEN_TILES), nl.bfloat16, f"{name_prefix}mla_out_dst")
    return sources


def mla_collective_hbm(
    tokens: int,
    name: str = "",
    out_rs: bool = False,
    gathers: bool = False,
    q_hbm: bool = False,
    out_ag_sbuf: bool = False,
    qkv_hbm: bool = False,
    keys: tuple | None = None,
    skip: tuple = (),
) -> dict:
    """Return the persistent HBM tensors of the MLA collectives.

    The FP32 source and destination of the state reduce-scatter, and with ``out_rs`` the BF16 source and
    destination of the output reduce-scatter and the destination of its all-gather. An SBUF reduce-scatter gets
    a compiler scratch, which the Neuron runtime writes only on Trn2. ``name`` tells the tensors
    apart when one kernel builds more than one set. ``gathers`` adds the HBM source and destination of the qkv and
    query all-gathers, for more than 8 sequences, where the SBUF collective tiles of 2 groups do not fit.
    ``q_hbm`` (verify parity sets only) adds the BF16 source and destination of the q input all-gather, so it
    runs in HBM and its SBUF tiles leave the collective arenas. ``out_ag_sbuf`` (verify parity sets only) drops
    the output all-gather HBM destination, so only the output reduce-scatter runs in HBM and the all-gather runs
    in SBUF into the persistent mla_out_dst. ``qkv_hbm`` (loop path only) adds the BF16 source and partition-axis
    destination of the qkv input all-gather, so each per-call gather runs in HBM into its own destination
    (collective_dim=0), which a peer cannot alias to region-heap bytes; a strided load then restores the layout.
    ``keys`` (None: all) keeps only the named tensors and ``skip`` drops the named ones (``mla_stage_hbm``).
    """
    state = tokens * (_KV_LORA + 1)
    rows = tokens * _HIDDEN_TILES
    shapes = [("state_hbm", (_HEADS, state), nl.float32), ("state_dst_hbm", (_LOCAL_HEADS, state), nl.float32)]
    if out_rs:
        shapes += [
            ("out_rs_src_hbm", (_P_MAX, rows), nl.bfloat16),
            ("out_rs_dst_hbm", (_P_MAX // _TP, rows), nl.bfloat16),
        ]
        if not out_ag_sbuf:
            shapes += [("out_ag_dst_hbm", (_P_MAX, rows), nl.bfloat16)]
    if q_hbm:
        set_rows = min(tokens, QUERY_SET_TOKENS)
        qw = -(-tokens // QUERY_SET_TOKENS) * _GATHER_WIDTH
        shapes += [
            ("q_ag_src_hbm", (set_rows, qw), nl.bfloat16),
            ("q_ag_dst_hbm", (_TP * set_rows, qw), nl.bfloat16),
        ]
    if qkv_hbm:
        shapes += [
            ("qkv_ag_src_hbm", (tokens, _QKV_LOCAL), nl.bfloat16),
            ("qkv_ag_dst_hbm", (_TP * tokens, _QKV_LOCAL), nl.bfloat16),
        ]
    if gathers:
        sets, set_rows = -(-tokens // QUERY_SET_TOKENS), min(tokens, QUERY_SET_TOKENS)
        shapes += [
            ("qkv_ag_src_hbm", (tokens, _QKV_LOCAL), nl.bfloat16),
            ("qkv_ag_dst_hbm", (_TP * tokens, _QKV_LOCAL), nl.bfloat16),
            ("q_ag_src_hbm", (set_rows, sets * _GATHER_WIDTH), nl.bfloat16),
            ("q_ag_dst_hbm", (_TP * set_rows, sets * _GATHER_WIDTH), nl.bfloat16),
        ]
    return {
        key: nl.ndarray(shape, dtype=dtype, buffer=nl.private_hbm, name=f"mla{name}_{key}")
        for key, shape, dtype in shapes
        if key not in skip and (keys is None or key in keys)
    }


def mla_stage_hbm(tokens: int, name: str = "", split_sites: tuple = (), unit_tokens: int = 0) -> dict:
    """Return the HBM tensors of the MLA collectives of a staged layer: one set for all its groups of 8.

    The groups store and load their slices (``mla_stage_views``), and the caller issues each collective once for
    all groups (``mla_stage_collective``). The layers reuse the set: world collectives separate two MLA layers.
    ``split_sites`` (sites of ``MLA_STAGE_SPLIT_SITES``) run their collective once for each unit of ``unit_tokens``
    sequences instead (``decode_block`` ``mla_stage_cc_split``): each unit has its own source and destination
    (``"units"``) in place of the tensors of the whole stage, so the bytes are the same.
    """
    if not split_sites:
        return mla_collective_hbm(tokens, name=name, out_rs=True, gathers=True)
    kernel_assert(
        set(split_sites) <= set(MLA_STAGE_SPLIT_SITES)
        and unit_tokens > 0
        and unit_tokens % QUERY_SET_TOKENS == 0
        and tokens % unit_tokens == 0,
        f"split_sites must be sites of {MLA_STAGE_SPLIT_SITES} with units of whole query sets that divide the "
        f"{tokens} sequences; got {split_sites} and units of {unit_tokens}",
    )
    split = tuple(key for site in split_sites for key in _STAGE_KEYS[site])
    stage = mla_collective_hbm(tokens, name=name, out_rs=True, gathers=True, skip=split)
    stage["units"] = [
        mla_collective_hbm(unit_tokens, name=f"{name}_u{unit}", gathers=True, keys=split)
        for unit in range(tokens // unit_tokens)
    ]
    stage["split_sites"] = tuple(split_sites)
    stage["unit_tokens"] = unit_tokens
    return stage


def _stage_unit(stage: dict, site: str, g0: int, count: int) -> tuple:
    """Return the tensors of ``site`` that hold sequences ``g0 .. g0 + count - 1``, and the row of ``g0`` in them."""
    if site not in stage.get("split_sites", ()):
        return stage, g0
    unit, first = divmod(g0, stage["unit_tokens"])
    kernel_assert(
        first + count <= stage["unit_tokens"],
        f"the {count} sequences from {g0} span two units of {stage['unit_tokens']}",
    )
    return stage["units"][unit], first


def mla_stage_views(stage: dict, g0: int, count: int) -> dict:
    """Return the sources of the group of ``count`` sequences from sequence ``g0``: its slices of ``stage``.

    A split site (``mla_stage_hbm`` ``split_sites``) takes its slices of the tensors of the unit of the group.
    """
    tokens = stage["qkv_ag_src_hbm"].shape[0]
    q_stage, q0 = _stage_unit(stage, "q", g0, count)
    state_stage, s0 = _stage_unit(stage, "state", g0, count)
    sets0, sets = q0 // QUERY_SET_TOKENS, -(-count // QUERY_SET_TOKENS)
    q_cols = slice(sets0 * _GATHER_WIDTH, (sets0 + sets) * _GATHER_WIDTH)
    state_cols = slice(s0 * (_KV_LORA + 1), (s0 + count) * (_KV_LORA + 1))
    out_cols = slice(g0 * _HIDDEN_TILES, (g0 + count) * _HIDDEN_TILES)
    return {
        "qkv_ag_src_hbm": stage["qkv_ag_src_hbm"][g0 : g0 + count, 0:_QKV_LOCAL],
        "qkv_ag_dst_hbm": stage["qkv_ag_dst_hbm"],
        "stage_rows": tokens,
        "stage_row": g0,
        "q_ag_src_hbm": q_stage["q_ag_src_hbm"][0:QUERY_SET_TOKENS, q_cols],
        "q_ag_dst_hbm": q_stage["q_ag_dst_hbm"][0 : _TP * QUERY_SET_TOKENS, q_cols],
        "state_hbm": state_stage["state_hbm"][0:_HEADS, state_cols],
        "state_dst_hbm": state_stage["state_dst_hbm"][0:_LOCAL_HEADS, state_cols],
        "out_rs_src_hbm": stage["out_rs_src_hbm"][0:_P_MAX, out_cols],
        "out_rs_dst_hbm": stage["out_rs_dst_hbm"][0 : _P_MAX // _TP, out_cols],
        "out_ag_dst_hbm": stage["out_ag_dst_hbm"][0:_P_MAX, out_cols],
        "out_in_hbm": True,
    }


_STAGE_KEYS = {
    "qkv": ("qkv_ag_src_hbm", "qkv_ag_dst_hbm"),
    "q": ("q_ag_src_hbm", "q_ag_dst_hbm"),
    "state": ("state_hbm", "state_dst_hbm"),
    "out_rs": ("out_rs_src_hbm", "out_rs_dst_hbm"),
    "out_ag": ("out_rs_dst_hbm", "out_ag_dst_hbm"),
}


MLA_STAGE_SPLIT_SITES = ("q", "state")
"""The sites of a staged layer whose collective can run once for each unit of groups (``mla_stage_hbm``)."""


def mla_stage_collective(stage: dict, cc, tag: str, unit: int | None = None):
    """Issue the collective ``cc`` that every group of a staged layer yields, once, on the tensors of ``stage``.

    With ``unit``, issue it for the groups of that unit of a split site, on the tensors of the unit.
    """
    src, dst = _STAGE_KEYS[cc.site]
    if unit is not None:
        tensors = stage["units"][unit]
        return cc.fn(
            **{**cc.kwargs, "srcs": [tensors[src]], "dsts": [tensors[dst]], "name": f"{tag}_mla_{cc.site}_u{unit}"}
        )
    return cc.fn(**{**cc.kwargs, "srcs": [stage[src]], "dsts": [stage[dst]], "name": f"{tag}_mla_{cc.site}"})


class _Softmax:
    """The softmax reference of one layer, capped (default) or exact.

    The capped reference of a head is the scaled current score plus 64.46, and a score more than 124.46
    above the scaled current score is capped. The exact reference is the running maximum of the raw
    scores on this rank, which starts at the current score; ``merge_scale`` then moves every rank to the
    maximum over the ranks.
    """

    def __init__(self, exact: bool, tokens: int, sb: SbufTiles):
        self.sb = sb
        self.exact = exact
        self.tokens = tokens
        fp32 = nl.float32
        if exact:
            self.rank_current_weight = 2.0**-5
            self.running_max = self.sb.tile((_HEADS, tokens), fp32)
            self.alpha = self.sb.tile((_HEADS, 1), fp32)
        else:
            self.rank_current_weight = _RANK_CURRENT_WEIGHT
            self.negative_reference = self.sb.tile((_HEADS, tokens), fp32)
            self.score_cap = self.sb.tile((_HEADS, tokens), fp32)

    def update(self, state, token, numerator_psum, group_sum):
        """Add one key group to the FP32 state of ``token``: ``state * alpha + [numerator | sum]``."""
        if self.exact:
            nisa.scalar_tensor_tensor(
                dst=state[0:_HEADS, token, 0:_KV_LORA],
                data=state[0:_HEADS, token, 0:_KV_LORA],
                op0=nl.multiply,
                operand0=self.alpha,
                op1=nl.add,
                operand1=numerator_psum,
            )
            nisa.scalar_tensor_tensor(
                dst=state[0:_HEADS, token, _KV_LORA : _KV_LORA + 1],
                data=state[0:_HEADS, token, _KV_LORA : _KV_LORA + 1],
                op0=nl.multiply,
                operand0=self.alpha,
                op1=nl.add,
                operand1=group_sum,
            )
        else:
            nisa.tensor_tensor(
                dst=state[0:_HEADS, token, 0:_KV_LORA],
                data1=state[0:_HEADS, token, 0:_KV_LORA],
                data2=numerator_psum,
                op=nl.add,
            )
            nisa.tensor_tensor(
                dst=state[0:_HEADS, token, _KV_LORA : _KV_LORA + 1],
                data1=state[0:_HEADS, token, _KV_LORA : _KV_LORA + 1],
                data2=group_sum,
                op=nl.add,
            )

    def merge_scale(self, state, tp_group, tag):
        """Scale the state of each head by exp(scale * (rank maximum - maximum over the ranks)).

        The all-reduce runs on one [1, 96 B] row: an SBUF all-reduce over many partitions is not reliable.
        """
        fp32 = nl.float32
        tokens = self.tokens
        max_row_psum = nl.ndarray((1, tokens * _HEADS), dtype=fp32, buffer=nl.psum)
        for token in range(tokens):
            nisa.nc_transpose(
                dst=max_row_psum[0:1, token * _HEADS : (token + 1) * _HEADS],
                data=self.running_max[0:_HEADS, token : token + 1],
            )
        max_row = self.sb.tile((1, tokens * _HEADS), fp32)
        nisa.tensor_copy(dst=max_row, src=max_row_psum)
        global_row = self.sb.tile((1, tokens * _HEADS), fp32)
        ncc.all_reduce(
            srcs=[max_row], dsts=[global_row], replica_group=tp_group, op=nl.maximum, name=f"{tag}_mla_max_all_reduce"
        )
        global_psum = nl.ndarray((_HEADS, tokens), dtype=fp32, buffer=nl.psum)
        for token in range(tokens):
            nisa.nc_transpose(
                dst=global_psum[0:_HEADS, token : token + 1],
                data=global_row[0:1, token * _HEADS : (token + 1) * _HEADS],
            )
        negative_global = self.sb.tile((_HEADS, tokens), fp32)
        nisa.tensor_scalar(dst=negative_global, data=global_psum, op0=nl.multiply, operand0=-_SOFTMAX_SCALE)
        merge = self.sb.tile((_HEADS, tokens), fp32)
        for token in range(tokens):
            nisa.activation(
                dst=merge[0:_HEADS, token : token + 1],
                op=nl.exp,
                data=self.running_max[0:_HEADS, token : token + 1],
                scale=_SOFTMAX_SCALE,
                bias=negative_global[0:_HEADS, token : token + 1],
            )
            nisa.tensor_scalar(
                dst=state[0:_HEADS, token, 0 : _KV_LORA + 1],
                data=state[0:_HEADS, token, 0 : _KV_LORA + 1],
                op0=nl.multiply,
                operand0=merge[0:_HEADS, token : token + 1],
            )


def _reference(token, q_nope_all, q_rope_all, cur_keys, softmax):
    """Compute the current score of one sequence and set its softmax reference.

    The capped softmax writes ``-(scaled current score + 64.46)`` and the largest raw score that the weight
    cap allows. The exact softmax starts the running maximum at the raw current score. Every rank computes
    the same current score from the gathered queries and the gathered kv_a.
    """
    current_score_psum = nl.ndarray((_HEADS, 1), dtype=nl.float32, buffer=nl.psum)
    for latent_tile in range(_KV_LORA_TILES):
        nisa.nc_matmul(
            dst=current_score_psum,
            stationary=q_nope_all[0:_P_MAX, token, latent_tile, 0:_HEADS],
            moving=cur_keys[0:_P_MAX, latent_tile, token : token + 1],
            accumulate=latent_tile > 0,
        )
    nisa.nc_matmul(
        dst=current_score_psum,
        stationary=q_rope_all[0:_ROPE, token, 0:_HEADS],
        moving=cur_keys[0:_ROPE, _KV_LORA_TILES, token : token + 1],
        accumulate=True,
    )
    if softmax.exact:
        nisa.tensor_copy(dst=softmax.running_max[0:_HEADS, token : token + 1], src=current_score_psum)
        return
    nisa.tensor_scalar(
        dst=softmax.negative_reference[0:_HEADS, token : token + 1],
        data=current_score_psum,
        op0=nl.multiply,
        operand0=-_SOFTMAX_SCALE,
        op1=nl.subtract,
        operand1=_MERGE_OFFSET,
    )
    # A raw score s keeps scale * s + reference <= _WEIGHT_CAP when s <= (_WEIGHT_CAP - reference) / scale.
    nisa.tensor_scalar(
        dst=softmax.score_cap[0:_HEADS, token : token + 1],
        data=softmax.negative_reference[0:_HEADS, token : token + 1],
        op0=nl.multiply,
        operand0=-1.0 / _SOFTMAX_SCALE,
        op1=nl.add,
        operand1=_WEIGHT_CAP / _SOFTMAX_SCALE,
    )


def _attend_group(
    token,
    q_nope_all,
    q_rope_all,
    latent_keys,
    rope_keys,
    key_rows,
    softmax,
    group_sum,
    sb,
    q_tiles=None,
    q_rope=None,
    pair_blocks=0,
    pad_stationary=False,
    dr=None,
):
    """Attend one sequence over the up to 4 local blocks of one key group.

    Return the FP32 PSUM numerator ``[96, 512]`` (the accumulation group stays open), and write the FP32
    sum of the weights to ``group_sum`` (``[96, 1]``). The exact softmax also moves the running maximum
    of the sequence and writes the scale of the previous state to ``softmax.alpha``. The SBUF tiles of
    the group return to ``sb`` at the end, so the groups reuse their addresses. With ``pad_stationary``
    (``loop_pad_stationary``) the QK, rope and P@V matmuls take 128-column stationaries (the 96 heads and the next
    32 columns of their tile) and write 128-row PSUM tiles, of which only rows 0-95 are read. With ``dr`` (``loop_dr_keys``: the 2
    e4m3 double_row stationaries ``[128, 2, 96]`` of the sequence), ``latent_keys`` ``[128, 2, 512]`` holds the 2-byte
    key words (``_dr_group_keys``) and the latent part of the scores is 2 double_row matmuls.
    """
    sb.open_scope("group")
    bf16 = nl.bfloat16
    fp32 = nl.float32
    group_blocks = pair_blocks or key_rows.shape[1]
    num_keys = group_blocks * _BLOCK

    # Raw scores of the prior keys with the mask bias.
    if pad_stationary:
        q_flat = q_nope_all.reshape((_P_MAX, q_nope_all.shape[1] * _KV_LORA_TILES * _HEADS))
        rope_flat = q_rope_all.reshape((_P_MAX, q_rope_all.shape[1] * _HEADS))
        wide_psum = nl.ndarray((_P_MAX, num_keys), dtype=fp32, buffer=nl.psum)
        for latent_tile in range(_KV_LORA_TILES):
            column = (token * _KV_LORA_TILES + latent_tile) * _HEADS
            nisa.nc_matmul(
                dst=wide_psum,
                stationary=q_flat[0:_P_MAX, column : column + _P_MAX],
                moving=latent_keys[0:_P_MAX, latent_tile, 0:num_keys],
                accumulate=latent_tile > 0,
            )
        nisa.nc_matmul(
            dst=wide_psum,
            stationary=rope_flat[0 : _BIAS_ROW + 1, token * _HEADS : token * _HEADS + _P_MAX],
            moving=rope_keys[0 : _BIAS_ROW + 1, 0:num_keys],
            accumulate=True,
        )
        scores_psum = wide_psum[0:_HEADS, 0:num_keys]
    else:
        scores_psum = nl.ndarray((_HEADS, num_keys), dtype=fp32, buffer=nl.psum)
        if dr is None:
            for latent_tile in range(_KV_LORA_TILES):
                nisa.nc_matmul(
                    dst=scores_psum,
                    stationary=q_nope_all[0:_P_MAX, token, latent_tile, 0:_HEADS]
                    if q_tiles is None
                    else q_tiles[latent_tile],
                    moving=latent_keys[0:_P_MAX, latent_tile, 0:num_keys],
                    accumulate=latent_tile > 0,
                )
        else:
            # Pass ``byte`` contracts byte ``byte`` of the key words of both 256-feature blocks.
            for byte in range(2):
                nisa.nc_matmul(
                    dst=scores_psum,
                    stationary=dr[byte],
                    moving=_dr_word_keys(latent_keys, byte),
                    accumulate=byte > 0,
                    perf_mode=nisa.matmul_perf_mode.double_row,
                )
        nisa.nc_matmul(
            dst=scores_psum,
            stationary=q_rope_all[0 : _BIAS_ROW + 1, token, 0:_HEADS] if q_rope is None else q_rope,
            moving=rope_keys[0 : _BIAS_ROW + 1, 0:num_keys],
            accumulate=True,
        )
    if softmax.exact:
        # The new running maximum, the scale of the previous state, and the bias of this group. A group
        # without a valid key keeps the maximum, because its masked scores are -1e30.
        block_max = sb.tile((_HEADS, 1), fp32)
        nisa.tensor_reduce(dst=block_max, op=nl.maximum, data=scores_psum, axis=1)
        running_max = softmax.running_max[0:_HEADS, token : token + 1]
        nisa.tensor_tensor(dst=block_max, data1=block_max, data2=running_max, op=nl.maximum)
        bias = sb.tile((_HEADS, 1), fp32)
        nisa.tensor_scalar(dst=bias, data=block_max, op0=nl.multiply, operand0=-_SOFTMAX_SCALE)
        nisa.activation(dst=softmax.alpha, op=nl.exp, data=running_max, scale=_SOFTMAX_SCALE, bias=bias)
        nisa.tensor_copy(dst=running_max, src=block_max)
    else:
        # Cap the scores, so a key far above the current key cannot overflow the weights. A masked score
        # stays masked.
        nisa.tensor_scalar(
            dst=scores_psum,
            data=scores_psum,
            op0=nl.minimum,
            operand0=softmax.score_cap[0:_HEADS, token : token + 1],
        )
        bias = softmax.negative_reference[0:_HEADS, token : token + 1]
    weights = sb.tile((_HEADS, num_keys), bf16)
    nisa.activation(
        dst=weights,
        op=nl.exp,
        data=scores_psum,
        scale=_SOFTMAX_SCALE,
        bias=bias,
        reduce_op=nl.add,
        reduce_res=group_sum,
        reduce_cmd=nisa.reduce_cmd.reset_reduce,
    )

    # The weighted latent sum, the keys on the contraction axis.
    if pair_blocks:
        numerator_psum = pair_numerator(weights, key_rows, pair_blocks, _HEADS, sb)
        sb.close_scope()
        return numerator_psum
    weights_psum = nl.ndarray((_P_MAX, group_blocks, _HEADS), dtype=bf16, buffer=nl.psum)
    for local_block in range(group_blocks):
        nisa.nc_transpose(
            dst=weights_psum[0:_P_MAX, local_block, 0:_HEADS],
            data=weights[0:_HEADS, local_block * _BLOCK : (local_block + 1) * _BLOCK],
        )
    if pad_stationary:
        # One spare block at the end, so the 128-column stationary of the last block stays in the tile. GpSimd zeroes
        # the 32 columns of it that the stationary reads (they reach only PSUM rows 96-127) in each group, so the loop
        # carries no tile into its next iteration.
        weights_flat = sb.tile((_P_MAX, (group_blocks + 1) * _HEADS), bf16)
        weights_t = weights_flat.reshape((_P_MAX, group_blocks + 1, _HEADS))
        nisa.memset(dst=weights_t[0:_P_MAX, group_blocks, 0 : _P_MAX - _HEADS], value=0.0, engine=nisa.engine.gpsimd)
        nisa.tensor_copy(dst=weights_t[0:_P_MAX, 0:group_blocks, 0:_HEADS], src=weights_psum, engine=nisa.engine.scalar)
        wide_numerator = nl.ndarray((_P_MAX, _KV_LORA), dtype=fp32, buffer=nl.psum)
        for local_block in range(group_blocks):
            nisa.nc_matmul(
                dst=wide_numerator,
                stationary=weights_flat[0:_P_MAX, local_block * _HEADS : local_block * _HEADS + _P_MAX],
                moving=key_rows[0:_P_MAX, local_block, 0:_KV_LORA],
                accumulate=local_block > 0,
            )
        sb.close_scope()
        return wide_numerator[0:_HEADS, 0:_KV_LORA]
    weights_t = sb.tile((_P_MAX, group_blocks, _HEADS), bf16)
    # VectorE limits the loop path (the page clamps, the score cap and the state adds), so ScalarE takes
    # this exact BF16 copy.
    nisa.tensor_copy(dst=weights_t, src=weights_psum, engine=nisa.engine.scalar)
    numerator_psum = nl.ndarray((_HEADS, _KV_LORA), dtype=fp32, buffer=nl.psum)
    for local_block in range(group_blocks):
        nisa.nc_matmul(
            dst=numerator_psum,
            stationary=weights_t[0:_P_MAX, local_block, 0:_HEADS],
            moving=key_rows[0:_P_MAX, local_block, 0:_KV_LORA],
            accumulate=local_block > 0,
        )
    sb.close_scope()
    return numerator_psum


class _CurrentKey:
    """The operands that add the per-rank weight of the current key of a sequence to every head.

    The matmul of sequence b contracts over one partition: the current row of b and a row of the weight.
    A one-hot over the rows of all sequences would add a 0 x row term for each other sequence, and
    0 x NaN is NaN, so a NaN current row of one sequence would reach every sequence of the replica. An
    operand of one partition starts at partition 0, 32, 64 or 96, so the row of sequence b moves to
    partition 32 (b % 4) of column block b // 4 with one DMA. A batch of one sequence reads its row in place.
    """

    def __init__(self, cur_rows, tokens: int, weight: float, tag: str, sb):
        self.tokens = tokens
        self.rows = cur_rows
        if tokens > 1:
            self.rows = sb.tile((_P_MAX, -(-tokens // QUERY_SET_TOKENS) * _KV_LORA), nl.bfloat16)
            for token in range(tokens):
                row, col = self._place(token)
                nisa.dma_copy(
                    dst=self.rows[row : row + 1, col : col + _KV_LORA],
                    src=cur_rows[token : token + 1, 0:_KV_LORA],
                    dge_mode=nisa.dge_mode.hwdge,
                    engine=nisa.engine.sync,
                    name=f"{tag}_mla_current_row_{token}",
                )
        self.weight = sb.tile((_P_MAX, _HEADS), nl.bfloat16)
        nisa.memset(dst=self.weight, value=weight)

    @staticmethod
    def _place(token: int) -> tuple[int, int]:
        """Return the partition and the first column of the current row of ``token``."""
        return _QUADRANT * (token % QUERY_SET_TOKENS), (token // QUERY_SET_TOKENS) * _KV_LORA

    def add(self, numerator_psum, token: int, accumulate: bool):
        """Add the weighted current key of ``token`` to every head of ``numerator_psum`` (``[96, 512]`` FP32)."""
        row, col = self._place(token) if self.tokens > 1 else (0, 0)
        nisa.nc_matmul(
            dst=numerator_psum,
            stationary=self.weight[row : row + 1, 0:_HEADS],
            moving=self.rows[row : row + 1, col : col + _KV_LORA],
            accumulate=accumulate,
        )


def _query_rows(qkv_ranks, tokens, q_a_norm_w, kv_a_norm_w, lora_norm_eps, kv_dtype, sb, first_read=None):
    """Steps 9 to 11 of ``_mla_decode_body`` for ``tokens`` sequences: the normed latent queries and the current cache rows
    from the qkv gather ``qkv_ranks`` ``[B, 32, 66]``.

    ``first_read`` (None, or a function) gets the first read of the gather result. Returns ``q_latent``
    ``[128, B, 12]``, the BF16 current rows ``cur_rows`` ``[B, 576]`` and the cache rows ``write_rows`` (the dtype of the
    pool: with an FP8 pool the rounded rows, which ``cur_rows`` then holds as BF16).
    """
    bf16 = nl.bfloat16
    fp32 = nl.float32
    padded = tokens + tokens % 2
    # Step 9: q_a in the feature-major column layout of the q_b contraction. A TensorE transpose reads one
    # free dimension, so two copies first pack the q_a columns of all ranks into feature order.
    q_a_rows = sb.tile((tokens, _TP, _Q_LORA_LOCAL), bf16)
    half = _TP // 2
    qkv_read = nisa.tensor_copy(
        dst=q_a_rows[0:tokens, 0:half, 0:_Q_LORA_LOCAL],
        src=qkv_ranks[0:tokens, 0:half, 0:_Q_LORA_LOCAL],
        engine=nisa.engine.vector,
    )
    nisa.tensor_copy(
        dst=q_a_rows[0:tokens, half:_TP, 0:_Q_LORA_LOCAL],
        src=qkv_ranks[0:tokens, half:_TP, 0:_Q_LORA_LOCAL],
        engine=nisa.engine.scalar,
    )
    if first_read is not None:
        # The gate weights follow the first read of the gather result, so they are a descendant of the gather (the
        # compiler does not move them above it), and they stream while the rings are idle up to the query gather.
        first_read(qkv_read)
    q_a_flat = q_a_rows.reshape((tokens, _Q_LORA))
    q_a_psum = nl.ndarray((_P_MAX, _Q_LORA_TILES, padded), dtype=bf16, buffer=nl.psum)
    for q_lora_tile in range(_Q_LORA_TILES):
        nisa.nc_transpose(
            dst=q_a_psum[0:_P_MAX, q_lora_tile, 0:tokens],
            data=q_a_flat[0:tokens, q_lora_tile * _P_MAX : (q_lora_tile + 1) * _P_MAX],
        )
    q_a = sb.tile((_P_MAX, tokens, _Q_LORA_TILES), fp32)
    nisa.tensor_copy(dst=q_a, src=q_a_psum[0:_P_MAX, 0:_Q_LORA_TILES, 0:tokens].permute((0, 2, 1)))

    # Step 10: q_latent = rms(q_a). The norms run on VectorE and GpSimd, so ScalarE keeps one activation
    # table (exp) for the layer.
    ones = sb.tile((_P_MAX, _P_MAX), fp32)
    nisa.memset(dst=ones, value=1.0)
    q_a_square = sb.tile((_P_MAX, tokens, _Q_LORA_TILES), fp32)
    nisa.tensor_tensor(dst=q_a_square, data1=q_a, data2=q_a, op=nl.multiply)
    q_a_partial = sb.tile((_P_MAX, tokens), fp32)
    nisa.tensor_reduce(dst=q_a_partial, op=nl.add, data=q_a_square, axis=2)
    q_a_total = nl.ndarray((_P_MAX, tokens), dtype=fp32, buffer=nl.psum)
    nisa.nc_matmul(dst=q_a_total, stationary=ones, moving=q_a_partial, is_stationary_onezero=True)
    q_a_rstd = sb.tile((_P_MAX, tokens), fp32)
    nisa.tensor_scalar(
        dst=q_a_rstd,
        data=q_a_total,
        op0=nl.multiply,
        operand0=1.0 / float(_Q_LORA),
        op1=nl.add,
        operand1=lora_norm_eps,
    )
    nisa.tensor_scalar(dst=q_a_rstd, data=q_a_rstd, op0=nl.rsqrt, operand0=0.0, engine=nisa.engine.gpsimd)
    q_a_normed = sb.tile((_P_MAX, tokens, _Q_LORA_TILES), bf16)
    nisa.tensor_tensor(
        dst=q_a_normed,
        data1=q_a,
        data2=q_a_rstd.expand_dim(2).broadcast(2, _Q_LORA_TILES),
        op=nl.multiply,
    )
    q_latent = sb.tile((_P_MAX, tokens, _Q_LORA_TILES), bf16)
    nisa.tensor_tensor(
        dst=q_latent,
        data1=q_a_normed,
        data2=q_a_norm_w.expand_dim(1).broadcast(1, tokens),
        op=nl.multiply,
    )

    # Step 11: the current cache rows [rms(latent) | k_pe], one sequence per partition. kv_a column c of
    # rank r is feature 32 c + r, so the permuted view of the gather is in feature order.
    kv_latent = qkv_ranks[0:tokens, 0:_TP, _Q_LORA_LOCAL : _Q_LORA_LOCAL + _KV_LATENT_LOCAL]
    kv_square = sb.tile((tokens, _TP, _KV_LATENT_LOCAL), fp32)
    kv_partial = sb.tile((tokens, 1), fp32)
    nisa.scalar_tensor_tensor(
        dst=kv_square,
        data=kv_latent,
        op0=nl.multiply,
        operand0=1.0,
        op1=nl.multiply,
        operand1=kv_latent,
        reduce_op=nl.add,
        reduce_res=kv_partial,
    )
    kv_rstd = sb.tile((tokens, 1), fp32)
    nisa.tensor_scalar(
        dst=kv_rstd,
        data=kv_partial,
        op0=nl.multiply,
        operand0=1.0 / float(_KV_LORA),
        op1=nl.add,
        operand1=lora_norm_eps,
    )
    nisa.tensor_scalar(dst=kv_rstd, data=kv_rstd, op0=nl.rsqrt, operand0=0.0, engine=nisa.engine.gpsimd)
    kv_normed = sb.tile((tokens, _KV_LATENT_LOCAL, _TP), bf16)
    nisa.tensor_scalar(dst=kv_normed, data=kv_latent.permute((0, 2, 1)), op0=nl.multiply, operand0=kv_rstd)
    cur_rows = sb.tile((tokens, _CACHE_WIDTH), bf16)
    nisa.tensor_tensor(
        dst=cur_rows[0:tokens, 0:_KV_LORA],
        data1=kv_normed.reshape((tokens, _KV_LORA)),
        data2=kv_a_norm_w,
        op=nl.multiply,
    )
    nisa.tensor_copy(
        dst=cur_rows[0:tokens, _KV_LORA:_CACHE_WIDTH].reshape_dim(1, (_CACHE_LOCAL - _KV_LATENT_LOCAL, _TP)),
        src=qkv_ranks[0:tokens, 0:_TP, _Q_LORA_LOCAL + _KV_LATENT_LOCAL : _QKV_LOCAL].permute((0, 2, 1)),
    )

    # An FP8 pool stores the row clamped to +-fp8_page_max() (448 on Trn3, 240 on Trn2) and rounded to nearest
    # even (a cast alone gives NaN above the bound). The body then attends the stored row, as a GPU decode reads it.
    write_rows = cur_rows
    if kv_dtype != bf16:
        write_rows = sb.tile((tokens, _CACHE_WIDTH), kv_dtype)
        nisa.tensor_scalar(
            dst=write_rows,
            data=cur_rows,
            op0=nl.minimum,
            operand0=fp8_page_max(),
            op1=nl.maximum,
            operand1=-fp8_page_max(),
        )
        cur_rows = sb.tile((tokens, _CACHE_WIDTH), bf16)
        nisa.tensor_copy(dst=cur_rows, src=write_rows)

    return q_latent, cur_rows, write_rows


def _query_columns(
    cur_rows, q_latent, tokens, q_b_w, kv_b_w, sources, after_gate_loads, sb, mx_query="off", q_b_scales=None, tag=""
):
    """Steps 12 to 14 of ``_mla_decode_body`` for ``tokens`` sequences: the feature-major current keys and the query
    gather source.

    Returns ``cur_keys`` ``[128, 5, B]`` and ``q_local`` ``[min(B, 4), sets x 1728]`` (``sources["q"]`` when given).
    ``after_gate_loads`` orders the PSUM read-outs after the gate weight loads (``wgate_early``). ``mx_query`` as in
    ``_mla_decode_body``. ``q_b_scales`` (not None) runs the MXFP8 q_b of Step 13: ``q_b_w`` is then the ``[128, G,
    576]`` x4 weight (``_load_mx_q_b``) and the q_latent is quantized per sequence; the values match the verify twin.
    """
    bf16 = nl.bfloat16
    fp32 = nl.float32
    sets = -(-tokens // QUERY_SET_TOKENS)
    set_rows = min(tokens, QUERY_SET_TOKENS)
    padded = tokens + tokens % 2
    # Step 12: transpose the current keys into feature-major columns.
    cur_psum = nl.ndarray((_P_MAX, _KV_LORA_TILES + 1, padded), dtype=bf16, buffer=nl.psum)
    for latent_tile in range(_KV_LORA_TILES):
        nisa.nc_transpose(
            dst=cur_psum[0:_P_MAX, latent_tile, 0:tokens],
            data=cur_rows[0:tokens, latent_tile * _P_MAX : (latent_tile + 1) * _P_MAX],
        )
    nisa.nc_transpose(dst=cur_psum[0:_ROPE, _KV_LORA_TILES, 0:tokens], data=cur_rows[0:tokens, _KV_LORA:_CACHE_WIDTH])
    cur_keys = sb.tile((_P_MAX, _KV_LORA_TILES + 1, tokens), bf16)
    after_gate_loads(
        nisa.tensor_copy(
            dst=cur_keys[0:_P_MAX, 0:_KV_LORA_TILES, 0:tokens], src=cur_psum[0:_P_MAX, 0:_KV_LORA_TILES, 0:tokens]
        )
    )
    after_gate_loads(
        nisa.tensor_copy(
            dst=cur_keys[0:_ROPE, _KV_LORA_TILES, 0:tokens], src=cur_psum[0:_ROPE, _KV_LORA_TILES, 0:tokens]
        )
    )

    # Step 13: q = q_latent @ q_b, one row per sequence, q_latent stationary and the weight columns moving. The
    # rope queries come out for each query set, so that each set starts at partition 0 of the gather source.
    if q_b_scales is not None:
        # MXFP8 q_b: q_latent stationary (quantized per sequence), the q_b weight moving (the same orientation). Each
        # PSUM takes an even partition count (the stationary free size); the body reads only the live rows. The
        # per-head per-group order matches the verify twin, so the decode steps stay bitwise equal to the verify rows.
        q_nope_psum = nl.ndarray((padded, _LOCAL_HEADS * _NOPE), dtype=fp32, buffer=nl.psum)
        q_rope_psums = [
            nl.ndarray((set_rows + set_rows % 2, _LOCAL_HEADS * _ROPE), dtype=fp32, buffer=nl.psum) for _ in range(sets)
        ]
        q_data, q_scale = quantize_hT_mx(
            q_latent, f"{tag}_qb", padded, lambda shape, dtype, name: sb.tile(shape, dtype, name)
        )
        groups = q_b_w.shape[1]
        nope_view = q_nope_psum.reshape_dim(1, (_LOCAL_HEADS, _NOPE))
        for local_head in range(_LOCAL_HEADS):
            base = local_head * _Q_HEAD
            for group in range(groups):
                nisa.nc_matmul_mx(
                    dst=nope_view[0:padded, local_head, 0:_NOPE],
                    stationary=q_data[0:_P_MAX, group, 0:padded],
                    moving=q_b_w[0:_P_MAX, group, base : base + _NOPE],
                    stationary_scale=q_scale[0:_P_MAX, group, 0:padded],
                    moving_scale=mx_scale_view(q_b_scales, group, base, _NOPE),
                    accumulate=group > 0,
                )
        for query_set in range(sets):
            first, count = _set_span(query_set, tokens)
            set_padded = count + count % 2
            rope_view = q_rope_psums[query_set].reshape_dim(1, (_LOCAL_HEADS, _ROPE))
            for local_head in range(_LOCAL_HEADS):
                base = local_head * _Q_HEAD
                for group in range(groups):
                    nisa.nc_matmul_mx(
                        dst=rope_view[0:set_padded, local_head, 0:_ROPE],
                        stationary=q_data[0:_P_MAX, group, first : first + set_padded],
                        moving=q_b_w[0:_P_MAX, group, base + _NOPE : base + _Q_HEAD],
                        stationary_scale=q_scale[0:_P_MAX, group, first : first + set_padded],
                        moving_scale=mx_scale_view(q_b_scales, group, base + _NOPE, _ROPE),
                        accumulate=group > 0,
                    )
        q_nope_psum = q_nope_psum[0:tokens]
    else:
        q_nope_psum = nl.ndarray((tokens, _LOCAL_HEADS * _NOPE), dtype=fp32, buffer=nl.psum)
        q_rope_psums = [nl.ndarray((set_rows, _LOCAL_HEADS * _ROPE), dtype=fp32, buffer=nl.psum) for _ in range(sets)]
        q_b_heads = q_b_w.reshape_dim(2, (_LOCAL_HEADS, _Q_HEAD))

        def rope_matmul(query_set, q_lora_tile):
            first, count = _set_span(query_set, tokens)
            nisa.nc_matmul(
                dst=q_rope_psums[query_set].reshape_dim(1, (_LOCAL_HEADS, _ROPE))[0:count],
                stationary=q_latent[0:_P_MAX, first : first + count, q_lora_tile],
                moving=q_b_heads[0:_P_MAX, q_lora_tile, 0:_LOCAL_HEADS, _NOPE:_Q_HEAD],
                accumulate=q_lora_tile > 0,
            )

        for q_lora_tile in range(_Q_LORA_TILES):
            nisa.nc_matmul(
                dst=q_nope_psum.reshape_dim(1, (_LOCAL_HEADS, _NOPE)),
                stationary=q_latent[0:_P_MAX, 0:tokens, q_lora_tile],
                moving=q_b_heads[0:_P_MAX, q_lora_tile, 0:_LOCAL_HEADS, 0:_NOPE],
                accumulate=q_lora_tile > 0,
            )
            rope_matmul(0, q_lora_tile)
        # The second set accumulates after the first, so no two accumulation groups of a PSUM bank interleave.
        for query_set in range(1, sets):
            for q_lora_tile in range(_Q_LORA_TILES):
                rope_matmul(query_set, q_lora_tile)
    # The gather source holds the rows of one query set, and the sets side by side. A row holds q_abs of the 3
    # local heads (3 x 512), then their rope queries (3 x 64).
    q_local = sources["q"] if sources and "q" in sources else sb.tile((set_rows, sets * _GATHER_WIDTH), bf16)
    if tokens % set_rows and not (sources and "q" in sources):
        # The last set sends zero rows for its missing sequences; a persistent tile is cleared at the kernel start.
        last = (sets - 1) * _GATHER_WIDTH
        nisa.memset(dst=q_local[0:set_rows, last : last + _GATHER_WIDTH], value=0.0)
    for query_set in range(sets):
        _, count = _set_span(query_set, tokens)
        base = query_set * _GATHER_WIDTH
        after_gate_loads(
            nisa.tensor_copy(
                dst=q_local[0:count, base + _ABSORB_WIDTH : base + _GATHER_WIDTH], src=q_rope_psums[query_set][0:count]
            )
        )
    if isinstance(kv_b_w, MxKvB):
        # The MXFP8 key absorb reads the q_nope PSUM itself (one lane-major read-out, quantize_quadrant_rows_mx).
        _absorb_query_sets_mx(q_nope_psum, kv_b_w, q_local, tokens, sets, set_rows, after_gate_loads, sb, mx_query, tag)
        return cur_keys, q_local
    q_nope_row = sb.tile((tokens, _LOCAL_HEADS * _NOPE), bf16)
    after_gate_loads(nisa.tensor_copy(dst=q_nope_row, src=q_nope_psum))
    q_nope_t_psum = nl.ndarray((_P_MAX, _LOCAL_HEADS, padded), dtype=bf16, buffer=nl.psum)
    for local_head in range(_LOCAL_HEADS):
        nisa.nc_transpose(
            dst=q_nope_t_psum[0:_P_MAX, local_head, 0:tokens],
            data=q_nope_row[0:tokens, local_head * _NOPE : (local_head + 1) * _NOPE],
        )
    q_nope = sb.tile((_P_MAX, _LOCAL_HEADS, tokens), bf16)
    after_gate_loads(nisa.tensor_copy(dst=q_nope, src=q_nope_t_psum[0:_P_MAX, 0:_LOCAL_HEADS, 0:tokens]))

    # Step 14: absorb the key projection, q_abs[h] = q_nope[h] @ key[h], one row per sequence of each set.
    for local_head in range(_LOCAL_HEADS):
        for query_set in range(sets):
            first, count = _set_span(query_set, tokens)
            col = query_set * _GATHER_WIDTH + local_head * _KV_LORA
            q_abs_psum = nl.ndarray((set_rows, _KV_LORA), dtype=fp32, buffer=nl.psum)
            nisa.nc_matmul(
                dst=q_abs_psum[0:count],
                stationary=q_nope[0:_P_MAX, local_head, first : first + count],
                moving=kv_b_w[0:_P_MAX, local_head * _KV_LORA : (local_head + 1) * _KV_LORA],
            )
            if mx_query == "emulate":
                # The emulation scratch returns at the end of the scope (the packed loop needs the SBUF).
                with sb.scope("mx_query"):
                    emulated = _mx_emulate_rows(q_local[0:count, col : col + _KV_LORA], q_abs_psum[0:count], sb)
                after_gate_loads(emulated)
                continue
            after_gate_loads(
                nisa.tensor_copy(
                    dst=q_local[0:count, col : col + _KV_LORA],
                    src=q_abs_psum[0:count],
                    engine=nisa.engine.vector if local_head % 2 else nisa.engine.scalar,
                )
            )

    return cur_keys, q_local


def _absorb_query_sets_mx(q_nope_src, kv_b_w, q_local, tokens, sets, set_rows, after_gate_loads, sb, mx_query, tag):
    """Step 14 of ``_query_columns`` with the MXFP8 ``kv_b_w`` (``MxKvB``): the q_nope rows ``q_nope_src``
    ``[tokens, 3 * 128]`` (the FP32 PSUM of Step 13, or SBUF) quantized in quadrant row tiles (head h at partitions
    32 h), then for each query set the 3 heads as concurrent MXFP8 row tiles (``quadrant_absorb_mx``), one
    bank-aligned PSUM output a head. Each output row reads only its own q_nope column, so a sequence gets the bits of
    its verify row (``mla_block_attention``)."""
    fp32 = nl.float32
    padded = tokens + tokens % 2
    set_padded = set_rows + set_rows % 2
    qn_data, qn_scale = _absorb_mx_operand(q_nope_src, tokens, padded, f"{tag}_qn", sb, after_gate_loads)
    for query_set in range(sets):
        first, count = _set_span(query_set, tokens)
        q_abs_psums = [nl.ndarray((set_padded, _KV_LORA), dtype=fp32, buffer=nl.psum) for _ in range(_LOCAL_HEADS)]
        quadrant_absorb_mx(
            q_abs_psums, qn_data, qn_scale, kv_b_w.absorb, kv_b_w.absorb_scale, _KV_LORA, first, count + count % 2
        )
        for local_head in range(_LOCAL_HEADS):
            col = query_set * _GATHER_WIDTH + local_head * _KV_LORA
            q_abs_psum = q_abs_psums[local_head]
            if mx_query == "emulate":
                # The emulation scratch returns at the end of the scope (the packed loop needs the SBUF).
                with sb.scope("mx_query"):
                    emulated = _mx_emulate_rows(q_local[0:count, col : col + _KV_LORA], q_abs_psum[0:count], sb)
                after_gate_loads(emulated)
                continue
            after_gate_loads(
                nisa.tensor_copy(
                    dst=q_local[0:count, col : col + _KV_LORA],
                    src=q_abs_psum[0:count],
                    engine=nisa.engine.vector if local_head % 2 else nisa.engine.scalar,
                )
            )


def _set_span(query_set: int, tokens: int) -> tuple[int, int]:
    """Return the first sequence and the sequence count of query set ``query_set``."""
    first = query_set * QUERY_SET_TOKENS
    return first, min(QUERY_SET_TOKENS, tokens - first)


def _set_queries(q_all, query_set: int, tokens: int, q_nope_all, q_rope_all):
    """Transpose the gathered queries of one query set into the feature-major query tiles of its sequences.

    ``q_all`` is the query gather ``[32 R, S * 1728]``: row ``R r + b`` of column block ``s`` holds the 3 heads of
    rank ``r`` for sequence ``4 s + b``. The latent tiles go to ``q_nope_all`` ``[128, B, 4, 96]`` and the rope
    queries to ``q_rope_all`` ``[128, B, 96]``, with the heads in global order 3r + h.
    """
    bf16 = nl.bfloat16
    first, count = _set_span(query_set, tokens)
    gather_rows = q_all.shape[0]
    set_rows = gather_rows // _TP
    base = query_set * _GATHER_WIDTH
    for local_head in range(_LOCAL_HEADS):
        q_head_psum = nl.ndarray((_P_MAX, _KV_LORA_TILES, gather_rows), dtype=bf16, buffer=nl.psum)
        for latent_tile in range(_KV_LORA_TILES):
            col = base + local_head * _KV_LORA + latent_tile * _P_MAX
            nisa.nc_transpose(
                dst=q_head_psum[0:_P_MAX, latent_tile, 0:gather_rows], data=q_all[0:gather_rows, col : col + _P_MAX]
            )
        # (k, r, b) of the transposes goes to sequence first + b, tile k, head 3r + h.
        nisa.tensor_copy(
            dst=q_nope_all.reshape_dim(3, (_TP, _LOCAL_HEADS))[
                0:_P_MAX, first : first + count, 0:_KV_LORA_TILES, 0:_TP, local_head
            ],
            src=q_head_psum.reshape_dim(2, (_TP, set_rows))[0:_P_MAX, 0:_KV_LORA_TILES, 0:_TP, 0:count].permute(
                (0, 3, 1, 2)
            ),
            engine=nisa.engine.scalar if local_head % 2 else nisa.engine.vector,
        )
    q_rope_psum_all = nl.ndarray((_ROPE, _LOCAL_HEADS, gather_rows), dtype=bf16, buffer=nl.psum)
    for local_head in range(_LOCAL_HEADS):
        col = base + _ABSORB_WIDTH + local_head * _ROPE
        nisa.nc_transpose(
            dst=q_rope_psum_all[0:_ROPE, local_head, 0:gather_rows], data=q_all[0:gather_rows, col : col + _ROPE]
        )
    nisa.tensor_copy(
        dst=q_rope_all[0:_ROPE, first : first + count, 0:_HEADS].reshape_dim(2, (_TP, _LOCAL_HEADS)),
        src=q_rope_psum_all.reshape_dim(2, (_TP, set_rows))[0:_ROPE, 0:_LOCAL_HEADS, 0:_TP, 0:count].permute(
            (0, 3, 2, 1)
        ),
    )


def _mx_query_words(mx_keys: str) -> int:
    """Return the FP32 words of the MX query stationaries of one sequence on each partition (``pack_mx_keys``)."""
    return _HEADS * (2 if mx_keys == "split" else 1)


def _mx_queries(q_all, query_set: int, tokens: int, q_mx_all, sb, split: bool = False):
    """Round the gathered latent queries of one query set to e4m3 and transpose them as 32-bit words into the MX
    stationaries of its sequences (``pack_mx_keys``).

    ``q_all`` is the query gather (``_set_queries``). ``q_mx_all`` ``[128, B, 96]`` (``[128, B, 192]`` with ``split``) is
    the FP32 view of ``float8_e4m3fn_x4``: word ``(c, b, h)`` holds latent features ``4c .. 4c + 3`` of head ``h`` (global
    order 3r + h) of sequence ``b``. hi = e4m3(q), nearest even with saturation at 448 (unit scale), as the GPU decode
    rounds its query; with ``split`` words ``96 + h`` hold lo = e4m3(q - hi), also saturated (q - hi is exact in BF16).
    The 32-bit TensorE transposes and the FP32 copies keep the bits.
    """
    first, count = _set_span(query_set, tokens)
    gather_rows = q_all.shape[0]
    set_rows = gather_rows // _TP
    base = query_set * _GATHER_WIDTH
    words = _KV_LORA // _MX_PACK
    q_rows = q_all[0:gather_rows, base : base + _ABSORB_WIDTH]
    parts = [sb.tile((gather_rows, _ABSORB_WIDTH), nl.float8_e4m3fn)]
    nisa.tensor_scalar(
        dst=parts[0],
        data=q_rows,
        op0=nl.minimum,
        operand0=_E4M3_MAX,
        op1=nl.maximum,
        operand1=-_E4M3_MAX,
        engine=nisa.engine.vector,
    )
    if split:
        rest = sb.tile((gather_rows, _ABSORB_WIDTH), nl.bfloat16)
        nisa.tensor_tensor(dst=rest, data1=q_rows, data2=parts[0], op=nl.subtract, engine=nisa.engine.vector)
        parts.append(sb.tile((gather_rows, _ABSORB_WIDTH), nl.float8_e4m3fn))
        nisa.tensor_scalar(
            dst=parts[1],
            data=rest,
            op0=nl.minimum,
            operand0=_E4M3_MAX,
            op1=nl.maximum,
            operand1=-_E4M3_MAX,
            engine=nisa.engine.vector,
        )
    for part, q8_rows in enumerate(parts):
        q8_words = q8_rows.view(nl.float32)
        q_mx_psum = nl.ndarray((_P_MAX, _LOCAL_HEADS, gather_rows), dtype=nl.float32, buffer=nl.psum)
        for local_head in range(_LOCAL_HEADS):
            nisa.nc_transpose(
                dst=q_mx_psum[0:_P_MAX, local_head, 0:gather_rows],
                data=q8_words[0:gather_rows, local_head * words : (local_head + 1) * words],
                engine=nisa.engine.tensor,
            )
        # (h, r, b) of the transposes goes to sequence first + b, head 3r + h of the part.
        nisa.tensor_copy(
            dst=q_mx_all[0:_P_MAX, first : first + count, part * _HEADS : (part + 1) * _HEADS].reshape_dim(
                2, (_TP, _LOCAL_HEADS)
            ),
            src=q_mx_psum.reshape_dim(2, (_TP, set_rows))[0:_P_MAX, 0:_LOCAL_HEADS, 0:_TP, 0:count].permute(
                (0, 3, 2, 1)
            ),
            engine=nisa.engine.vector,
        )


def _dr_queries(q_all, query_set: int, tokens: int, q_dr_all, identity16, sb, q_dr_psum=None):
    """Round the gathered latent queries of one query set to e4m3 and lay them out as the double_row stationaries of
    its sequences (``pack_dr_keys``).

    ``q_all`` is the query gather (``_set_queries``) and ``identity16`` a BF16 identity ``[128, 128]``. ``q_dr_all``
    ``[128, B, 2, 2, 96]`` FP8: element ``(c, b, i, k, h)`` is latent feature ``256 k + 2c + i`` of head ``h`` (global order
    3r + h) of sequence ``b``, so ``q_dr_all[:, b, i]`` is the stationary ``[128, 2, 96]`` of double_row pass ``i``. The
    rounding is e4m3(q), nearest even with saturation (unit scale), as ``_mx_queries`` rounds it. A 2-byte
    transpose-mode transpose of each 256-feature block of a head moves the feature pairs (2c, 2c + 1) to partition c,
    and one FP8 copy for each (byte, block) takes byte i of the words: every step keeps the bits. The FP8 format is
    that of the pages (``fp8_page_dtype``: float8_e4m3fn on Trn3, the legacy float8_e4m3 on Trn2, saturated at
    ``fp8_page_max``). ``q_dr_psum`` (None: a new tile) is the PSUM tile ``[128, 3, 2, 32 R]`` BF16 of the transposes.
    """
    fp8 = fp8_page_dtype()
    first, count = _set_span(query_set, tokens)
    gather_rows = q_all.shape[0]
    set_rows = gather_rows // _TP
    base = query_set * _GATHER_WIDTH
    q8_rows = sb.tile((gather_rows, _ABSORB_WIDTH), fp8)
    nisa.tensor_scalar(
        dst=q8_rows,
        data=q_all[0:gather_rows, base : base + _ABSORB_WIDTH],
        op0=nl.minimum,
        operand0=fp8_page_max(),
        op1=nl.maximum,
        operand1=-fp8_page_max(),
        engine=nisa.engine.vector,
    )
    q8_words = q8_rows.view(nl.bfloat16)
    if q_dr_psum is None:
        q_dr_psum = nl.ndarray((_P_MAX, _LOCAL_HEADS, _DR_BLOCKS, gather_rows), dtype=nl.bfloat16, buffer=nl.psum)
    for local_head in range(_LOCAL_HEADS):
        for block in range(_DR_BLOCKS):
            col = (local_head * _DR_BLOCKS + block) * _DR_BLOCK_WORDS
            nisa.nc_matmul(
                dst=q_dr_psum[0:_P_MAX, local_head, block, 0:gather_rows],
                stationary=q8_words[0:gather_rows, col : col + _DR_BLOCK_WORDS],
                moving=identity16[0:gather_rows, 0:gather_rows],
                is_transpose=True,
                accumulate=False,
            )
    # (h, k, r, b, i) of the transposes (byte i of the word) goes to sequence first + b, pass i, block k, head 3r + h.
    words = q_dr_psum.view(fp8).reshape_dim(3, (_TP, set_rows, 2))
    heads = q_dr_all.reshape_dim(4, (_TP, _LOCAL_HEADS))
    for byte in range(2):
        for block in range(_DR_BLOCKS):
            nisa.tensor_copy(
                dst=heads[0:_P_MAX, first : first + count, byte, block, 0:_TP, 0:_LOCAL_HEADS],
                src=words[0:_P_MAX, 0:_LOCAL_HEADS, block, 0:_TP, 0:count, byte].permute((0, 3, 2, 1)),
                engine=nisa.engine.vector,
            )


def _dr_stationary(slot_row, byte: int, word_offset: int = 0, cols: int = _HEADS):
    """Return the e4m3 double_row stationary ``[128, 2, 96]`` of pass ``byte`` in the first words of ``slot_row``
    (``pack_dr_keys``: ``[byte, block, head]`` FP8, ``_dr_queries``), or in those at 2-byte word ``word_offset`` on (the
    stage row of a query row of the multi-query packed loop). ``cols`` 128: the ``[128, 2, 128]`` stationary of the 128
    (row, head) columns of a ``pack_p2`` sub-slot (``[byte, block, column]`` FP8, ``_stage_query_rows_p2``)."""
    row_bytes = slot_row.shape[1] * 2
    return slot_row.ap(
        pattern=[[row_bytes, _P_MAX], [cols, _DR_BLOCKS], [1, cols]],
        offset=2 * word_offset + byte * _DR_BLOCKS * cols,
        dtype=nl.float8_e4m3fn,
    )


def _dr_query_rows(q_all, tokens: int, sb):
    """Return the e4m3 double_row stationaries ``[128, B, 2, 2, 96]`` FP8 of the ``tokens`` sequences of the query gather
    ``q_all`` (``_dr_queries``), with their BF16 identity, in the current scope of ``sb`` (``pack_dr_keys``)."""
    identity16 = sb.tile((_P_MAX, _P_MAX), nl.bfloat16)
    nl.shared_identity_matrix(_P_MAX, dtype=nl.bfloat16, dst=identity16)
    rows = sb.tile((_P_MAX, tokens, 2, _DR_BLOCKS, _HEADS), nl.float8_e4m3fn)
    for query_set in range(-(-tokens // QUERY_SET_TOKENS)):
        _dr_queries(q_all, query_set, tokens, rows, identity16, sb)
    return rows


def _loop_dr_query_rows(q_all, tokens: int, sb):
    """Return the e4m3 double_row stationaries ``[128, B, 2, 2, 96]`` (the FP8 page format) of the ``tokens`` sequences of
    the query gather ``q_all`` for the per-sequence key loops (``loop_dr_keys``), as ``_dr_queries`` writes them.

    The rounding scratch returns to ``sb`` at the end. The transposes write a whole PSUM bank, so no open accumulation
    group shares their 2 KB zone, and they take the BF16 identity of the kernel (``nl.shared_identity_matrix``).
    """
    rows = sb.tile((_P_MAX, tokens, 2, _DR_BLOCKS, _HEADS), fp8_page_dtype())
    identity16 = nl.shared_identity_matrix(_P_MAX, dtype=nl.bfloat16)
    gather_rows = q_all.shape[0]
    with sb.scope("dr_queries"):
        for query_set in range(-(-tokens // QUERY_SET_TOKENS)):
            # [128, 4, 2, 128] BF16 is one 2 KB bank; the transposes take its first 3 heads and 32 R columns.
            bank = nl.ndarray((_P_MAX, _LOCAL_HEADS + 1, _DR_BLOCKS, _P_MAX), dtype=nl.bfloat16, buffer=nl.psum)
            q_dr_psum = bank[0:_P_MAX, 0:_LOCAL_HEADS, 0:_DR_BLOCKS, 0:gather_rows]
            _dr_queries(q_all, query_set, tokens, rows, identity16, sb, q_dr_psum=q_dr_psum)
    return rows


def _dr_group_keys(rows, latent_keys, identity16):
    """Write the 2-byte key words of the FP8 row-major page tile ``rows`` (``[128, 4, 576]``) into ``latent_keys``
    ``[128, 2, 512]`` for the double_row keys of the per-sequence loop (``loop_dr_keys``): 8 transpose-mode transposes
    (``_dr_key_words``, 2 blocks x 4 pages) into one whole PSUM bank, then one exact ScalarE copy (``_dr_key_copies``)."""
    num_keys = rows.shape[1] * _BLOCK
    words_psum = nl.ndarray((_P_MAX, _DR_BLOCKS * num_keys), dtype=nl.bfloat16, buffer=nl.psum)
    last = None
    for block in range(_DR_BLOCKS):
        last = _dr_key_words(rows, words_psum, identity16, block)
    _dr_key_copies(latent_keys, words_psum, after=last)


def _dr_gather_tile_queries(q_all, count: int, token_major: bool, q_dr_all, first: int, identity16, sb):
    """Write the e4m3 double_row stationaries of the ``count`` rows of one query gather tile ``q_all``
    ``[32 count, >= 1536]`` (the 3 local heads' latent queries of 32 ranks) to columns ``first .. first + count - 1`` of
    ``q_dr_all`` ``[128, C, 2, 2, 96]`` FP8, as ``_dr_queries`` does for a query set of the decode (the multi-query
    packed loop: ``tkg_mla_block.mla_block_attention`` ``packed_prior_options`` ``dr_keys``). The rows of ``q_all`` are
    (token, rank) with ``token_major``, else (rank, token). The same rounding, transposes and byte copies, so the
    stationaries of a row are bitwise those of its decode step."""
    fp8 = nl.float8_e4m3fn
    gather_rows = _TP * count
    q8_rows = sb.tile((gather_rows, _ABSORB_WIDTH), fp8)
    nisa.tensor_scalar(
        dst=q8_rows,
        data=q_all[0:gather_rows, 0:_ABSORB_WIDTH],
        op0=nl.minimum,
        operand0=_E4M3_MAX,
        op1=nl.maximum,
        operand1=-_E4M3_MAX,
        engine=nisa.engine.vector,
    )
    q8_words = q8_rows.view(nl.bfloat16)
    q_dr_psum = nl.ndarray((_P_MAX, _LOCAL_HEADS, _DR_BLOCKS, gather_rows), dtype=nl.bfloat16, buffer=nl.psum)
    for local_head in range(_LOCAL_HEADS):
        for block in range(_DR_BLOCKS):
            col = (local_head * _DR_BLOCKS + block) * _DR_BLOCK_WORDS
            nisa.nc_matmul(
                dst=q_dr_psum[0:_P_MAX, local_head, block, 0:gather_rows],
                stationary=q8_words[0:gather_rows, col : col + _DR_BLOCK_WORDS],
                moving=identity16[0:gather_rows, 0:gather_rows],
                is_transpose=True,
                accumulate=False,
            )
    # (h, k, rows, i) of the transposes (byte i of the word) goes to column first + t, pass i, block k, head 3r + h.
    split = (count, _TP, 2) if token_major else (_TP, count, 2)
    words = q_dr_psum.view(fp8).reshape_dim(3, split)
    heads = q_dr_all.reshape_dim(4, (_TP, _LOCAL_HEADS))
    for byte in range(2):
        for block in range(_DR_BLOCKS):
            if token_major:
                src = words[0:_P_MAX, 0:_LOCAL_HEADS, block, 0:count, 0:_TP, byte].permute((0, 2, 3, 1))
            else:
                src = words[0:_P_MAX, 0:_LOCAL_HEADS, block, 0:_TP, 0:count, byte].permute((0, 3, 2, 1))
            nisa.tensor_copy(
                dst=heads[0:_P_MAX, first : first + count, byte, block, 0:_TP, 0:_LOCAL_HEADS],
                src=src,
                engine=nisa.engine.vector,
            )


def _dr_stage_queries(stage_u, row: int, count: int, dr_rows):
    """Copy the double_row stationaries of the first ``count`` sequences of ``dr_rows`` (``_dr_query_rows``) into the
    first words of stage rows ``row`` on of ``stage_u`` (the 2-byte view of the stage tile): an exact 2-byte copy."""
    words = dr_rows.view(nl.uint16).reshape((_P_MAX, dr_rows.shape[1], _DR_QUERY_WORDS))
    nisa.tensor_copy(
        dst=stage_u[0:_P_MAX, row : row + count, 0:_DR_QUERY_WORDS],
        src=words[0:_P_MAX, 0:count, 0:_DR_QUERY_WORDS],
        engine=nisa.engine.vector,
    )


def _e4m3_query_rounding(q_nope_all, q_rope_all, sb):
    """Round the rope queries ``q_rope_all`` ``[128, B, 96]`` (rows 0..63) to e4m3 in place, and return the latent query
    tiles ``q_nope_all`` ``[128, B, 4, 96]`` rounded to e4m3, as ``_mx_queries`` rounds them (``pack_mx_keys``
    "single").

    The GPU decode rounds its whole query ``[q_latent | q_rope]`` to e4m3 once. The rope queries stay BF16 tiles that hold
    the e4m3 values, so the rope matmul and its mask row do not change. The current score takes both roundings, so the
    current key keeps its weight relative to the prior keys as on the GPU. The FP8 format is that of the pages
    (``fp8_page_dtype``), saturated at ``fp8_page_max``.
    """
    tokens = q_rope_all.shape[1]
    fp8, bound = fp8_page_dtype(), fp8_page_max()
    rope8 = sb.tile((_ROPE, tokens, _HEADS), fp8)
    nisa.tensor_scalar(
        dst=rope8,
        data=q_rope_all[0:_ROPE, 0:tokens, 0:_HEADS],
        op0=nl.minimum,
        operand0=bound,
        op1=nl.maximum,
        operand1=-bound,
        engine=nisa.engine.vector,
    )
    nisa.tensor_copy(dst=q_rope_all[0:_ROPE, 0:tokens, 0:_HEADS], src=rope8, engine=nisa.engine.vector)
    q_reference = sb.tile(tuple(q_nope_all.shape), fp8)
    nisa.tensor_scalar(
        dst=q_reference,
        data=q_nope_all,
        op0=nl.minimum,
        operand0=bound,
        op1=nl.maximum,
        operand1=-bound,
        engine=nisa.engine.vector,
    )
    return q_reference


def _dynamic_attention(
    kv_pool,
    block_table,
    token_meta,
    token_f,
    q_nope_all,
    q_rope_all,
    softmax,
    slot_offset,
    state,
    tokens,
    local_blocks,
    tag,
    sb,
    clamp_pages=True,
    pair_pages=False,
    dr_rows=None,
    cast_engines="off",
    pad_stationary=False,
    restart="off",
    groups=0,
    page_gather="off",
    inputs_ahead=False,
):
    """Add the key groups of the dynamic path to the FP32 softmax state ``state`` ``[96, B, 513]``.

    A table of up to 16 local blocks attends every key group of every sequence in straight-line code (the
    unrolled path). The masks hide the groups after the position. A larger table has one loop per sequence
    over its key groups of 4 local blocks that hold a prior key, ``_LOOP_GROUPS`` groups per iteration
    (``_BF16_LOOP_GROUPS`` with BF16 pages), so the cost follows the context of each sequence and not the
    capacity of the table. The last iteration of a sequence can attend more groups, which their masks hide
    completely. ``dr_rows`` (``loop_dr_keys``): the double_row query stationaries ``[128, B, 2, 2, 96]``
    (``_loop_dr_query_rows``) of the double_row keys.
    """
    # Above 4 sequences the tiles of the key loops return to the region at the end, so the projections after
    # them reuse them; B <= 4 keeps the SBUF layout of 4 sequences. The page gathers add tiles of each sequence, so
    # with them the tiles return at any B, and so do the first-row table tiles of each sequence of ``inputs_ahead``
    # with the table restart.
    own_tiles = page_gather != "off" or (inputs_ahead and restart != "off")
    with sb.scope("keys") if tokens > QUERY_SET_TOKENS or own_tiles else nullcontext():
        _key_loops(
            kv_pool,
            block_table,
            token_meta,
            token_f,
            q_nope_all,
            q_rope_all,
            softmax,
            slot_offset,
            state,
            tokens,
            local_blocks,
            tag,
            sb,
            clamp_pages,
            pair_pages,
            dr_rows,
            restart=restart,
            groups=groups,
            **({"cast_engines": cast_engines} if cast_engines != "off" else {}),
            **({"pad_stationary": True} if pad_stationary else {}),
            **({"page_gather": page_gather} if page_gather != "off" else {}),
            **({"inputs_ahead": True} if inputs_ahead else {}),
        )


def _key_loops(
    kv_pool,
    block_table,
    token_meta,
    token_f,
    q_nope_all,
    q_rope_all,
    softmax,
    slot_offset,
    state,
    tokens,
    local_blocks,
    tag,
    sb,
    clamp_pages=True,
    pair_pages=False,
    dr_rows=None,
    cast_engines="off",
    pad_stationary=False,
    restart="off",
    groups=0,
    page_gather="off",
    inputs_ahead=False,
):
    """Run the key loops of ``_dynamic_attention``; the caller frees their SBUF tiles. ``cast_engines``: the
    ``loop_cast_engines`` form of the page-tile casts; ``pad_stationary``: ``loop_pad_stationary``.

    ``groups`` (``LOOP_GROUP_COUNTS``): 0 keeps one loop per sequence of ``_LOOP_GROUPS`` (or ``_BF16_LOOP_GROUPS``
    with BF16 pages) groups per iteration, G. 16: the loop inputs still count the c iterations of G groups of a
    sequence; the sequence runs c // (16 / G) iterations of 16 groups, then c % (16 / G) iterations of G groups on the
    first G page tiles (the tail), from the table row of G groups after the 16-group ones. So it attends the groups of
    the default loop, in about G / 16 of its iterations (each back edge is a barrier of every engine). The same groups
    in the same order give the same state bits.

    ``restart`` (``LOOP_RESTARTS``, loop path only): "off" keeps the program. "table": each iteration loads the table
    entries of the next iteration (row ``(step + 1) % rows``, a valid row on the last one) into a second entries tile
    after the page loads of its group 0, clamps them after group 2, and copies them to the entries tile at its end; the
    first row loads before the loop. So after the back edge the page loads start at once. The same bytes reach the same
    tiles and every compute op is the same, so the output is bitwise the same. The carried tiles live across the back
    edge, so "table" needs explicit tiles (``sb`` with a region).

    ``page_gather`` (``LOOP_PAGE_GATHERS``, loop path on FP8 row-major pages with explicit tiles): "off" keeps the program.
    "group": the 4 pages of key group g load with one GpSimd software DGE gather of 512 rows of 576 bytes into the page
    tile of the group (row k of the gather is row k // 4 of page k % 4, and its index sits at [k % 128, k // 128] of the
    ``[128, 4]`` index tile, as the hardware reads the index tile column-major), in place of 4 SP DMAs and their 4 table
    loads; the gather of group g + 2 issues before the casts of group g. Each iteration loads the table row of the next
    iteration (row ``(step + 1) % rows``, on every partition) on the SP queue and builds its gather indices: the page of
    each partition (``nc_n_gather``), clamped into the pool as the table entries are, shifted by 7, plus the row. The
    first rows of every sequence load and their indices build before the first loop, into index tiles of each sequence,
    so nothing loads between two loops. "carry": also the gather of the next iteration's group 0, after the softmax
    update of group 1, into group 0's page tile, which the next iteration (or the tail loop) reads without loading; each
    sequence has its own group-0 tile, whose first gather issues before the first loop. The R1 entries of ``restart``
    are not needed then. The same clamped pages reach the same tiles and every compute op is the same, so the output is
    bitwise the same.
    "carry_pool": "carry" with the gather indices of the next iteration built on GpSimd alone (``* 128`` for the shift
    and ``+`` for the row, exact on integers below 2**24), from the table rows of every sequence clamped once before the
    loops (``_clamped_table``: GpSimd has no minimum / maximum on Trn2): GpSimd idles after the gathers of the iteration,
    while the compiler puts the VectorE build at the end of the iteration, so with it the group-0 gather and its DMA run
    right before the back edge, which waits for them. The same indices, so the same bits.

    ``inputs_ahead`` (loop path only): False keeps the program. True: the loop inputs of every sequence (the register
    loads of its key start and iteration count, with ``groups`` 16 its main and tail counts, and with ``restart``
    "table" its first table row, into a table tile of its own) issue before the first sequence's loop. A block between
    two loops then holds only branches, and the compiler puts no fence there (each fence drains and stops every engine).
    The same registers and table bytes reach the same uses, so the output is bitwise the same.
    """
    page_cast = _LOOP_PAGE_CASTS.get(cast_engines)
    cast_kw = {} if cast_engines == "off" or page_cast else {"cast_engines": _loop_cast_engines(cast_engines)}
    bf16 = nl.bfloat16
    fp32 = nl.float32
    int32 = nl.int32
    hwdge = nisa.dge_mode.hwdge
    sync = nisa.engine.sync
    num_pages = kv_pool.shape[0]
    group_count = local_blocks // _GROUP_BLOCKS
    group_keys = _GROUP_BLOCKS * _BLOCK
    kernel_assert(groups in LOOP_GROUP_COUNTS, f"loop_groups must be one of {LOOP_GROUP_COUNTS}, got {groups!r}")
    default_groups = _BF16_LOOP_GROUPS if kv_pool.dtype == nl.bfloat16 else _LOOP_GROUPS
    loop_groups = min(groups or default_groups, group_count)
    loop_blocks = loop_groups * _GROUP_BLOCKS
    unrolled = not mla_runs_loops(local_blocks)
    # With 16 groups, the loop inputs count iterations of the default groups (the tail's).
    tail_groups = default_groups if not unrolled and loop_groups > default_groups else 0
    meta_groups = tail_groups or loop_groups

    # Step D1: the loop inputs of each sequence. Local block j holds a prior key when
    # 4096 j + 128 t < position, so a sequence has ceil((position - 128 t) / 16384) key groups, and
    # ceil(groups / loop_groups) loop iterations. Key group g keeps the slot offsets below
    # position - 128 t - 16384 g.
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
    loop_meta_f = sb.tile((tokens, 2), fp32)
    nisa.tensor_scalar(
        dst=loop_meta_f[0:tokens, 0:1],
        data=local_position,
        op0=nl.add,
        operand0=float(_GROUP_STRIDE - 1),
    )
    nisa.tensor_copy(dst=loop_meta_f[0:tokens, 1:2], src=local_position)
    loop_meta = sb.tile((tokens, 2), int32)
    nisa.tensor_copy(dst=loop_meta, src=loop_meta_f)
    # groups = (local position + 16383) >> 14, at most the table capacity; iterations = ceil(groups / G)
    # for G groups per iteration, a power of two.
    nisa.tensor_scalar(dst=loop_meta[0:tokens, 0:1], data=loop_meta[0:tokens, 0:1], op0=nl.right_shift, operand0=14)
    nisa.tensor_scalar(
        dst=loop_meta[0:tokens, 0:1],
        data=loop_meta[0:tokens, 0:1],
        op0=nl.minimum,
        operand0=float(group_count),
        op1=nl.add,
        operand1=float(meta_groups - 1),
    )
    nisa.tensor_scalar(
        dst=loop_meta[0:tokens, 0:1],
        data=loop_meta[0:tokens, 0:1],
        op0=nl.right_shift,
        operand0=meta_groups.bit_length() - 1,
    )
    # A register load reads partition 0, so the loop inputs move to one row through HBM.
    loop_hbm = nl.ndarray((tokens, 2), dtype=int32, buffer=nl.hbm, name=f"{tag}_mla_loop_meta")
    nisa.dma_copy(dst=loop_hbm, src=loop_meta, dge_mode=hwdge, engine=sync, name=f"{tag}_mla_loop_store")
    loop_row = sb.tile((1, 2 * tokens), int32)
    nisa.dma_copy(
        dst=loop_row, src=loop_hbm.reshape((1, 2 * tokens)), dge_mode=hwdge, engine=sync, name=f"{tag}_mla_loop_load"
    )

    # Step D2: one loop per sequence. Each iteration loads the table entries of its key groups, then attends
    # each group with its own buffers and adds it to the state of the sequence.
    # The double_row keys hold the 2-byte key words of the 2 blocks of a key (``_dr_group_keys``).
    latent_parts = _KV_LORA_TILES if dr_rows is None else _DR_BLOCKS
    identity16 = None if dr_rows is None else nl.shared_identity_matrix(_P_MAX, dtype=bf16)
    key_sets = [
        (
            sb.tile((_P_MAX, latent_parts, group_keys), bf16),
            sb.tile((_P_MAX, group_keys), bf16),
            sb.tile((_P_MAX, 1), fp32),
            sb.tile((_HEADS, 1), fp32),
        )
        for _ in range(min(loop_groups, _KEY_SETS))
    ]
    # The page forms of ``cast_engines``: "dma" loads BF16 page tiles (a ring of ``page_cast[1]`` if set), "page" casts
    # each page into a ring of BF16 group tiles that the transposes and the value sums read.
    dma_cast = page_cast is not None and page_cast[0] == "dma"
    page_dtype = bf16 if dma_cast else kv_pool.dtype
    ring_size = page_cast[1] if dma_cast else 0
    kernel_assert(
        not ring_size or page_gather == "off" or ring_size > _GATHER_LEAD,
        f"a dma ring with loop_page_gather needs more than {_GATHER_LEAD} tiles (the gathers lead by {_GATHER_LEAD} groups)",
    )
    # With the carried gathers each sequence keeps its own group-0 tile across the back edge, so the ring holds the groups
    # from 1 on.
    ring_start = 1 if ring_size and page_gather in ("carry", "carry_pool") else 0
    page_count = min(loop_groups, ring_start + ring_size) if ring_size else loop_groups
    page_tiles = [_page_tile(sb, _GROUP_BLOCKS, page_dtype, pair_pages) for _ in range(page_count)]

    def page_index(group):
        """The page tile of ``group`` (its own one, or its ring slot)."""
        return group if group < ring_start or not ring_size else ring_start + (group - ring_start) % ring_size

    buffers = [(page_tiles[page_index(group)], *key_sets[group % len(key_sets)]) for group in range(loop_groups)]
    cast_tiles = (
        [_page_tile(sb, _GROUP_BLOCKS, bf16, False) for _ in range(min(page_cast[1], loop_groups))]
        if page_cast is not None and page_cast[0] == "page"
        else None
    )
    # A BF16 tile of FP8 pages keeps the FP8 clamp bound, so a NaN byte gives the same value as in an FP8 page tile.
    page_bound = fp8_page_max() if page_cast is not None else None
    gather = page_gather != "off"
    kernel_assert(
        page_gather in LOOP_PAGE_GATHERS, f"loop_page_gather must be one of {LOOP_PAGE_GATHERS}, got {page_gather!r}"
    )
    kernel_assert(
        not gather
        or (
            not unrolled
            and not pair_pages
            and sb.explicit
            and kv_pool.dtype == fp8_page_dtype()
            and (not tail_groups or loop_groups == 2 * tail_groups)
        ),
        "loop_page_gather needs the loop path (more than 16 local blocks) on FP8 row-major pages and explicit tiles (a "
        "region): its index tiles live across the back edge",
    )
    kernel_assert(
        not gather or num_pages * _BLOCK <= 2**24,
        "loop_page_gather needs a pool of at most 2**17 pages (exact indices)",
    )
    entries = None if gather else sb.tile((1, loop_blocks), int32)
    kernel_assert(restart in LOOP_RESTARTS, f"loop_restart must be one of {LOOP_RESTARTS}, got {restart!r}")
    # The gathers carry their own indices across the back edge, so the R1 entries are not loaded with them.
    carry = restart != "off" and not unrolled and not gather
    kernel_assert(
        not carry or (sb.explicit and not pair_pages),
        "loop_restart carries the entries across the back edge: it needs explicit tiles (a region) and no pair_pages",
    )
    entries_next = sb.tile((1, loop_blocks), int32) if carry else None
    threshold_i = sb.tile((_P_MAX, 1), int32)
    threshold_base = sb.tile((_P_MAX, 1), fp32)
    group_offsets = sb.tile((_P_MAX, loop_groups), fp32)
    thresholds = sb.tile((_P_MAX, loop_groups), fp32)
    table_rows = local_blocks // loop_blocks
    vector = nisa.engine.vector
    gpsimd = nisa.engine.gpsimd
    carry_gather = page_gather in ("carry", "carry_pool")
    # The GpSimd build of the next iteration's gather indices.
    pool_index = page_gather == "carry_pool"
    if gather:
        # The table row of an iteration on every partition, the page of each (partition, group) of the row, and the
        # gather indices of the next iteration and of this one ([128, groups, 4] int32).
        row_table = sb.tile((_P_MAX, loop_blocks), int32)
        page_select = sb.tile((_P_MAX, loop_groups), nl.uint32)
        row_offsets = sb.tile((_P_MAX, _GROUP_BLOCKS), int32)
        group_pages = sb.tile((_P_MAX, loop_groups), int32)
        index_next = sb.tile((_P_MAX, loop_groups, _GROUP_BLOCKS), int32)
        # Each sequence's loop has its own current indices (and with "carry" its own group-0 page tile), so the first
        # iteration of every sequence builds (and gathers) before the first loop, not between two loops.
        first_tables = [sb.tile((_P_MAX, loop_blocks), int32) for _ in range(tokens)]
        index_curs = [sb.tile((_P_MAX, loop_groups, _GROUP_BLOCKS), int32) for _ in range(tokens)]
        group0_rows = [buffers[0][0]] + [
            _page_tile(sb, _GROUP_BLOCKS, page_dtype, pair_pages) for _ in range(tokens - 1 if carry_gather else 0)
        ]
        kv_rows = kv_pool.reshape((num_pages * _BLOCK, _CACHE_WIDTH))
        _gather_constants(page_select, row_offsets, sb)
        # With the GpSimd build ("carry_pool") each iteration reads the next table row from the table clamped once before
        # the loops: GpSimd has no minimum / maximum on Trn2.
        clamped_table = _clamped_table(block_table, tokens, local_blocks, num_pages, tag, sb) if pool_index else None

    def build_index(dst, table, pool=False):
        """``page_gather``: the gather indices of the table row in ``table`` into ``dst`` (``_gather_index``)."""
        return _gather_index(dst, table, group_pages, page_select, row_offsets, num_pages, pool)

    def gather_group(rows, index, name):
        """``page_gather``: gather the 4 pages of one key group (512 rows of 576 bytes, ``index`` ``[128, 4]``) into its
        page tile ``rows``. Return the DMA."""
        return nisa.dma_copy(
            dst=rows,
            src=kv_rows.ap(
                pattern=[[_CACHE_WIDTH, _GROUP_BLOCKS * _BLOCK], [1, _CACHE_WIDTH]],
                offset=0,
                vector_offset=index,
                indirect_dim=0,
            ),
            dge_mode=nisa.dge_mode.swdge,
            engine=nisa.engine.dma,
            name=name,
        )

    def load_table(dst, src, name):
        """Load one table row into ``dst`` and clamp it into the pool (the body's own form)."""
        nisa.dma_copy(dst=dst, src=src, dge_mode=hwdge, engine=sync, name=name)
        nisa.tensor_scalar(
            dst=dst,
            data=dst,
            op0=nl.minimum,
            operand0=float(num_pages - 1),
            op1=nl.maximum,
            operand1=0.0,
            engine=vector,
        )

    if gather:
        # The first iteration of every sequence (a tail-only sequence takes the first groups of its first row too):
        # its table row, its indices, and with "carry" its group 0, all before the first loop.
        for token in range(tokens):
            nisa.dma_copy(
                dst=first_tables[token],
                src=block_table[token : token + 1, 0:loop_blocks].ap(pattern=[[0, _P_MAX], [1, loop_blocks]], offset=0),
                dge_mode=hwdge,
                engine=sync,
                name=f"{tag}_mla_gather_table_first_{token}",
            )
        for token in range(tokens):
            build_index(index_curs[token], first_tables[token])
        for token in range(tokens if carry_gather else 0):
            gather_group(
                group0_rows[token],
                index_curs[token][0:_P_MAX, 0, 0:_GROUP_BLOCKS],
                f"{tag}_mla_gather_first_{token}",
            )

    kernel_assert(
        not inputs_ahead or not unrolled,
        "loop_inputs_ahead moves the inputs of the per-sequence key loops: it needs the loop path (more than 16 local "
        "blocks)",
    )
    kernel_assert(
        not (inputs_ahead and carry) or sb.explicit,
        "loop_inputs_ahead with loop_restart keeps a first table row of each sequence across the loops: it needs "
        "explicit tiles (a region)",
    )
    ratio = loop_groups // tail_groups if tail_groups else 1
    tail_blocks = tail_groups * _GROUP_BLOCKS
    ahead = {}
    if inputs_ahead:
        # Every sequence's loop inputs, before the first loop: then only branches lie between two loops (a register op
        # or a load between them would add a fence of every engine).
        ahead["start"] = [nisa.scalar_load(loop_row[0:1, 2 * t + 1 : 2 * t + 2]) for t in range(tokens)]
        ahead["count"] = [nisa.scalar_load(loop_row[0:1, 2 * t : 2 * t + 1]) for t in range(tokens)]
        if tail_groups:
            ahead["main"] = [count // ratio for count in ahead["count"]]
            ahead["tail"] = [count % ratio for count in ahead["count"]]
            ahead["row"] = [main * ratio for main in ahead["main"]]
        if carry:
            # The first iteration's entries of each sequence, each into its own tile.
            ahead["entries"] = [sb.tile((1, loop_blocks), int32) for _ in range(tokens)]
            for t in range(tokens):
                first_row = block_table[t : t + 1, 0:local_blocks].reshape((table_rows, loop_blocks))
                load_table(ahead["entries"][t], first_row[0:1, 0:loop_blocks], f"{tag}_mla_table_first_{t}")

    for token in range(tokens):
        local_start = (
            ahead["start"][token] if inputs_ahead else nisa.scalar_load(loop_row[0:1, 2 * token + 1 : 2 * token + 2])
        )
        table_row = block_table[token : token + 1, 0:local_blocks].reshape((table_rows, loop_blocks))
        token_entries = ahead["entries"][token] if "entries" in ahead else entries
        if carry and not inputs_ahead:
            # The first iteration's entries load before the loop.
            load_table(entries, table_row[0:1, 0:loop_blocks], f"{tag}_mla_table_first_{token}")
        if gather:
            index_cur = index_curs[token]
            # The rows of the next iteration's gather indices: with the GpSimd build, from the clamped table.
            index_rows = (
                clamped_table[token : token + 1, 0:local_blocks].reshape((table_rows, loop_blocks))
                if pool_index
                else table_row
            )
            # The page tiles of this sequence's groups: with "carry" its own group-0 tile.
            token_buffers = [(group0_rows[token if carry_gather else 0], *buffers[0][1:]), *buffers[1:]]
        else:
            token_buffers = buffers

        def body(
            step,
            token=token,
            local_start=local_start,
            table_row=table_row,
            part=None,
            index_cur=index_cur if gather else None,
            index_rows=index_rows if gather else None,
            buffers=token_buffers,
            entries=token_entries,
        ):
            # The unrolled path gives a static step, and the loop path gives the loop register. ``part``: the tail
            # loop's (groups, table view, row blocks, first row register).
            groups_n, row_view, row_blocks, row = (
                (loop_groups, table_row, loop_blocks, step) if part is None else (*part[:3], part[3] + step)
            )
            carry_on = carry and part is None
            if not unrolled:
                # Write the slot offsets in the loop body, so the tile lives inside one basic block. If the
                # block before the loop writes the tile and only the loop reads it, the tile is live across a
                # block boundary. Some neuronx-cc builds compute the SB liveness of a tile inside each block,
                # so the block before the loop can give its bytes to another tile. A MoE weight scale buffer
                # of the layer before did that: it overwrote the slot offsets of keys 472-511 of every group,
                # and the mask row of those keys then read weight scales.
                nisa.iota(
                    dst=slot_offset[_BIAS_ROW : _BIAS_ROW + 1, 0:group_keys],
                    pattern=[[_GLOBAL_BLOCK_STRIDE, _GROUP_BLOCKS], [1, _BLOCK]],
                    offset=0,
                )
            if unrolled:
                table_src = table_row[step : step + 1, 0:loop_blocks]
            else:
                table_src = row_view.ap(
                    pattern=[[row_blocks, 1], [1, row_blocks]], offset=0, scalar_offset=row, indirect_dim=0
                )
            part_entries = None if gather else (entries if part is None else entries[0:1, 0:row_blocks])
            if gather and part is None:
                # The next iteration's table row on every partition, for its gather indices.
                nisa.dma_copy(
                    dst=row_table,
                    src=index_rows.ap(
                        pattern=[[0, _P_MAX], [1, loop_blocks]],
                        offset=0,
                        scalar_offset=(step + 1) % table_rows,
                        indirect_dim=0,
                    ),
                    dge_mode=hwdge,
                    engine=sync,
                    name=f"{tag}_mla_gather_table_next_{token}",
                )
            if not carry_on and not gather:
                nisa.dma_copy(
                    dst=part_entries,
                    src=table_src,
                    dge_mode=hwdge,
                    engine=sync,
                    name=f"{tag}_mla_table_load_{token}_{step}"
                    if unrolled
                    else f"{tag}_mla_table_{'load' if part is None else 'tail'}_{token}",
                )
                # A padded table entry is clamped into the pool, so its masked rows are still finite reads.
                nisa.tensor_scalar(
                    dst=part_entries,
                    data=part_entries,
                    op0=nl.minimum,
                    operand0=float(num_pages - 1),
                    op1=nl.maximum,
                    operand1=0.0,
                )
            # The mask thresholds of the key groups of this step go through one register store to the bias
            # row. Each scalar store runs on every engine, so one store per step in place of one per group
            # saves time on every engine.
            nisa.scalar_store(
                threshold_i[_BIAS_ROW : _BIAS_ROW + 1, 0:1], local_start - row * (groups_n * _GROUP_STRIDE)
            )
            nisa.tensor_copy(
                dst=threshold_base[_BIAS_ROW : _BIAS_ROW + 1, 0:1],
                src=threshold_i[_BIAS_ROW : _BIAS_ROW + 1, 0:1],
                # With the gathers the copy runs on VectorE, so it does not hold the GpSimd queue (the gathers and
                # the GpSimd casts) while it waits for the register stores of every engine.
                **({"engine": vector} if gather else {}),
            )
            nisa.iota(
                dst=group_offsets[_BIAS_ROW : _BIAS_ROW + 1, 0:groups_n],
                pattern=[[-_GROUP_STRIDE, groups_n]],
                offset=0,
            )
            # Group g gets base - 16384 g, as the int32 register expression gave it. Each value is an integer
            # below 2**24, so the FP32 sum is exact.
            nisa.tensor_scalar(
                dst=thresholds[_BIAS_ROW : _BIAS_ROW + 1, 0:groups_n],
                data=group_offsets[_BIAS_ROW : _BIAS_ROW + 1, 0:groups_n],
                op0=nl.add,
                operand0=threshold_base[_BIAS_ROW : _BIAS_ROW + 1, 0:1],
                engine=nisa.engine.vector,
            )
            for group, (rows, latent_keys, rope_keys, _, group_sum) in enumerate(buffers[:groups_n]):
                # The rows the transposes and the value sums read: the group's BF16 tile with the "page" casts.
                key_rows = rows if cast_tiles is None else cast_tiles[group % len(cast_tiles)]
                if gather:
                    # Group g + 2's gather before the casts of group g; the first groups at the iteration start
                    # (group 0 carried in from the iteration before with "carry").
                    first = (1 if carry_gather else 0) if group == 0 else group + _GATHER_LEAD
                    last = min(group + _GATHER_LEAD, groups_n - 1)
                    for gathered in range(first, last + 1):
                        gather_group(
                            buffers[gathered][0],
                            index_cur[0:_P_MAX, gathered, 0:_GROUP_BLOCKS],
                            f"{tag}_mla_gather_{token}_{gathered}" + ("" if part is None else "_tail"),
                        )
                for local_block in range(_GROUP_BLOCKS):
                    entry = group * _GROUP_BLOCKS + local_block
                    name = f"{tag}_mla_group_page_{token}_{group}_{local_block}" + (
                        f"_{step}" if unrolled else ("" if part is None else "_tail")
                    )
                    if gather:
                        pass  # the group's gather loaded the page
                    elif pair_pages:
                        load_pair_page(kv_pool, rows, local_block, entries[0:1, entry : entry + 1], name)
                    else:
                        nisa.dma_copy(
                            dst=rows[0:_P_MAX, local_block, 0:_CACHE_WIDTH],
                            src=kv_pool.ap(
                                pattern=[[_CACHE_WIDTH, _BLOCK], [1, _CACHE_WIDTH]],
                                offset=0,
                                scalar_offset=entries[0:1, entry : entry + 1],
                                indirect_dim=0,
                            ),
                            dge_mode=hwdge,
                            engine=sync,
                            name=name,
                        )
                    # One clamp per page, so the transposes of a page start when its clamp ends. With
                    # zeroed pages (clamp_pages False) the loop path reads finite rows only, so it skips
                    # the clamp. The unrolled path always clamps. The "page" casts write the clamped (or copied)
                    # page into the group's BF16 tile.
                    page = _page_view(rows, local_block, pair_pages)
                    if cast_tiles is not None:
                        _cast_page(_page_view(key_rows, local_block, False), page, clamp_pages or unrolled)
                    elif clamp_pages or unrolled:
                        _sanitize_cache_rows(page, bound=page_bound)
                if carry_on and group == 0:
                    # The next iteration's entries, on the SP queue right after group 0's page loads.
                    nisa.dma_copy(
                        dst=entries_next,
                        src=table_row.ap(
                            pattern=[[loop_blocks, 1], [1, loop_blocks]],
                            offset=0,
                            scalar_offset=(step + 1) % table_rows,
                            indirect_dim=0,
                        ),
                        dge_mode=hwdge,
                        engine=sync,
                        name=f"{tag}_mla_table_next_{token}",
                    )
                pair_blocks = _GROUP_BLOCKS if pair_pages else 0
                if dr_rows is not None:
                    _dr_group_keys(rows, latent_keys, identity16)
                _transpose_group(
                    key_rows,
                    latent_keys,
                    rope_keys,
                    slot_offset,
                    thresholds[_BIAS_ROW : _BIAS_ROW + 1, group : group + 1],
                    pair_blocks=pair_blocks,
                    skip_latent=dr_rows is not None,
                    **cast_kw,
                )
                numerator_psum = _attend_group(
                    token,
                    q_nope_all,
                    q_rope_all,
                    latent_keys,
                    rope_keys,
                    key_rows,
                    softmax,
                    group_sum,
                    sb,
                    pair_blocks=pair_blocks,
                    dr=None
                    if dr_rows is None
                    else [dr_rows[0:_P_MAX, token, byte, 0:_DR_BLOCKS, 0:_HEADS] for byte in range(2)],
                    **({"pad_stationary": True} if pad_stationary else {}),
                )
                softmax.update(state, token, numerator_psum, group_sum)
                if gather and part is None and group == min(_GATHER_CARRY_GROUP, groups_n - 1):
                    build_index(index_next, row_table, pool=pool_index)
                    if carry_gather:
                        # Group 0's page tile has no reader left in this iteration.
                        gather_group(
                            buffers[0][0],
                            index_next[0:_P_MAX, 0, 0:_GROUP_BLOCKS],
                            f"{tag}_mla_gather_next_{token}",
                        )
                if carry_on and group == min(2, loop_groups - 1):
                    # The clamp of the next entries, long after their load (so VectorE does not wait for it).
                    nisa.tensor_scalar(
                        dst=entries_next,
                        data=entries_next,
                        op0=nl.minimum,
                        operand0=float(num_pages - 1),
                        op1=nl.maximum,
                        operand1=0.0,
                        engine=vector,
                    )
            if carry_on:
                # After every page load of this iteration has read the entries.
                nisa.tensor_copy(dst=entries, src=entries_next, engine=vector)
            if gather and part is None:
                # After every gather of this iteration has read its indices (the same GpSimd queue).
                nisa.tensor_copy(dst=index_cur, src=index_next, engine=gpsimd)

        if unrolled:
            for step in range(local_blocks // loop_blocks):
                body(step)
        else:
            iterations = (
                ahead["count"][token] if inputs_ahead else nisa.scalar_load(loop_row[0:1, 2 * token : 2 * token + 1])
            )
            if not tail_groups:
                nl.fori_loop(0, iterations, body)
            else:
                # ``iterations`` counts default-group iterations: ratio of them make one iteration of the main loop.
                main_iterations = ahead["main"][token] if inputs_ahead else iterations // ratio
                # The tail count before the main loop: a register op after the loop would add a fence of every engine.
                tail_iterations = ahead["tail"][token] if inputs_ahead else iterations % ratio
                tail_part = (
                    tail_groups,
                    block_table[token : token + 1, 0:local_blocks].reshape((local_blocks // tail_blocks, tail_blocks)),
                    tail_blocks,
                    ahead["row"][token] if inputs_ahead else main_iterations * ratio,
                )

                def tail_body(step, body=body, tail_part=tail_part):
                    body(step, part=tail_part)

                nl.fori_loop(0, main_iterations, body)
                nl.fori_loop(0, tail_iterations, tail_body)


# The latent key tiles of a slot whose PSUM copy runs on ScalarE, by their count (``pack_latent_act``); tile 0, which
# QK reads first, stays on VectorE.
_LATENT_ACT_TILES = {2: (1, 3), 3: (1, 2, 3)}
# The engines of the 20 FP8 -> BF16 page-tile casts of a key group of the per-sequence key loops (``loop_cast_engines``), by
# form: the latent key tiles 0-3 (4 pages each) and the rope tiles (4 pages). ScalarE takes the latent tiles whose PSUM copy
# is on ScalarE already (1 and 3); GpSimd (idle in the loop) takes the rest of the moved casts.
_LOOP_CAST_FORMS = {
    "off": None,
    "dve": {"latent": ("dve", "dve", "dve", "dve"), "rope": ("dve", "dve", "dve", "dve")},
    "split86": {"latent": ("dve", "act", "pool", "act"), "rope": ("dve", "dve", "pool", "pool")},
    "pool8": {"latent": ("dve", "act", "pool", "pool"), "rope": ("dve", "dve", "dve", "dve")},
    "pool12": {"latent": ("dve", "pool", "pool", "pool"), "rope": ("dve", "dve", "dve", "dve")},
    "act12": {"latent": ("dve", "act", "act", "act"), "rope": ("dve", "dve", "dve", "dve")},
    "act8": {"latent": ("act", "act", "pool", "pool"), "rope": ("dve", "dve", "dve", "dve")},
}
# The page forms of ``loop_cast_engines``: (where the cast runs, tiles). "dma": the page loads write BF16 page tiles, one a
# group of an iteration (0) or a ring of that many. "page": one VectorE op a page (the clamp, or a copy) writes a ring of
# that many BF16 group tiles, which the transposes and the value sums read.
_LOOP_PAGE_CASTS = {
    "dma": ("dma", 0),
    "dma6": ("dma", 6),
    "dma4": ("dma", 4),
    "page2": ("page", 2),
    "page3": ("page", 3),
}
LOOP_CAST_ENGINES = (*_LOOP_CAST_FORMS, *_LOOP_PAGE_CASTS)


def _loop_cast_engines(form: str) -> dict:
    """The ``_transpose_group`` cast engines of ``loop_cast_engines`` form ``form``: {(latent tile, block) or ("rope",
    block): engine} for every page tile of a key group."""
    names = {"dve": nisa.engine.vector, "act": nisa.engine.scalar, "pool": nisa.engine.gpsimd}
    table = _LOOP_CAST_FORMS[form]
    engines = {
        (tile, block): names[table["latent"][tile]] for tile in range(_KV_LORA_TILES) for block in range(_GROUP_BLOCKS)
    }
    engines.update({("rope", block): names[table["rope"][block]] for block in range(_GROUP_BLOCKS)})
    return engines


# The page tile pitch of ``pack_mask_column``: 576 bytes of the row and the mask column in the next 32-byte granule.
_MASK_PAGE_PITCH = _CACHE_WIDTH + 32
# The query bias row of ``pack_mask_column``: -448 x 2^100 is exact in FP32 and far below any capped score.
_MASK_COLUMN_SCALE = 2.0**100
# ``kv_gather``: the pages of the 2 items of one gather (the elements of each partition of its destination).
_KV_PAIR_PAGES = 2 * _GROUP_BLOCKS
# The negative reference of the null items with ``kv_pitch576``: exp(scale x capped score - 2^100) is 0 in FP32, as for a
# masked score.
_NULL_ITEM_REFERENCE = 2.0**100
# ``kv_rem4``: the consecutive rows of a page on one partition, the partitions of a page and the bytes of a piece.
_REM4_ROWS = 4
_REM4_PARTS = _BLOCK // _REM4_ROWS
_REM4_PIECE = _REM4_ROWS * _CACHE_WIDTH
# ``kv_group``: the item counts of one group gather (its ring has the page buffers of two groups).
_KV_GROUP_SIZES = (2, 4, 8)
_TI_TILE_BASE = 160 * 1024
"""Explicit SBUF byte of the first TI tile when the body has no region (compiler placement)."""
_TI_TILE_SLOT = 36 * 1024
"""Explicit bytes per partition of the state slot (slot 0): the state of 17 columns (B = 16), 34884 bytes, fits."""
_TI_INDEX_BYTES = 128
"""Explicit bytes per partition of a TI index slot (slots 1, 2, ...): a [96, 33] uint16 tile takes 66."""


def _ti_tile(sb, shape, dtype, name, slot):
    """Return an SBUF tile for a TI operand of the packed loop.

    With compiler placement, neuronx-cc fails on a TI operand of a placed tile once the kernel has more than
    one MLA layer (``NCC_IBIR370`` in ``dma_optimization_sb``), so the TI tiles take explicit addresses. With
    a region, they take region addresses as every other tile.
    """
    if sb.explicit:
        return sb.tile(shape, dtype, name)
    kernel_assert(slot > 0 or math.prod(shape[1:]) * 4 <= _TI_TILE_SLOT, "the packed state does not fit its slot")
    offset = _TI_TILE_BASE if slot == 0 else _TI_TILE_BASE + _TI_TILE_SLOT + (slot - 1) * _TI_INDEX_BYTES
    return nl.ndarray(shape, dtype=dtype, buffer=nl.sbuf, name=name, address=(0, offset))


_PLAN_TILE_BASE = 200 * 1024
"""Explicit SBUF byte of the first plan tile when the caller has no region."""
_PLAN_TILE_BYTES = 4096
"""Explicit bytes per partition of one plan tile slot."""


def _plan_tile(sb, shape, dtype, name, slot):
    """Return a plan tile: on the heap of the region, or at an explicit address without one."""
    if sb.explicit:
        return sb.tile(shape, dtype, name, keep=True)
    return nl.ndarray(
        shape, dtype=dtype, buffer=nl.sbuf, name=name, address=(0, _PLAN_TILE_BASE + slot * _PLAN_TILE_BYTES)
    )


def pack_slots(local_blocks: int, slots: int | None = None, tokens: int | None = None) -> int:
    """Return the items of one packed-loop iteration for a table of ``local_blocks`` and ``tokens`` sequences:
    ``slots`` when given (a power of two from 1 to 64), else 16 at 64 local blocks and up with 4 sequences and up,
    and 8 otherwise."""
    if slots is None:
        wide = local_blocks >= _PACK_WIDE_BLOCKS and (tokens is None or tokens >= _PACK_WIDE_TOKENS)
        return _PACK_SLOTS_WIDE if wide else _PACK_SLOTS_SMALL
    kernel_assert(slots in (1, 2, 4, 8, 16, 32, 64), f"pack slots must be a power of two from 1 to 64, got {slots}")
    return slots


def _tail_slots(slots: int, tail_slots: int | tuple | None) -> tuple | None:
    """Return the items of one iteration of each tail loop (``pack_tail_slots``: one count, or two or more counts for
    as many tail loops, such as the rounded chain of ``pack_tail_chain``), largest first, or None for one loop."""
    if tail_slots is None:
        return None
    tails = tuple(tail_slots) if isinstance(tail_slots, (tuple, list)) else (tail_slots,)
    powers = (1, 2, 4, 8, 16, 32)
    if len(tails) == 1:
        kernel_assert(
            tails[0] in powers and tails[0] <= slots,
            f"pack tail slots must be a power of two up to the {slots} pack slots, got {tail_slots}",
        )
    else:
        kernel_assert(
            all(tail in powers for tail in tails)
            and slots > tails[0]
            and all(wide > narrow for wide, narrow in zip(tails, tails[1:])),
            f"pack tail slots must be powers of two below the {slots} pack slots, largest first, got {tail_slots}",
        )
    return tails


def pack_tail_chain(local_blocks: int, slots: int | None, tail_slots: int | tuple, tokens: int) -> tuple:
    """Return the tail loops of a rounded packed loop (``tail_round``) for ``tail_slots``: a single count t takes the
    loops of halving width from half the slots down to t (32 slots with 8: (16, 8); 64 slots with 8: (32, 16, 8)), so
    that each tail loop runs at most one iteration and a layer holds fewer than t null items; a tuple stays as given."""
    if isinstance(tail_slots, (tuple, list)):
        return tuple(tail_slots)
    chain = []
    width = pack_slots(local_blocks, slots, tokens) // 2
    while width > tail_slots:
        chain.append(width)
        width //= 2
    return (*chain, tail_slots)


class _SlotSoftmax:
    """The capped softmax reference of one packed slot: ``[96, 1]`` columns staged for the slot's sequence."""

    def __init__(self, negative_reference, score_cap):
        self.exact = False
        self.negative_reference = negative_reference
        self.score_cap = score_cap


def _per_partition_values(column, tokens, sb):
    """Return the FP32 ``[128, B]`` tile whose every partition holds the B values of ``column`` (FP32 ``[B, 1]``, small
    integers): a matmul of the values replicated along the free axis with a B x B identity (exact in BF16 / FP32)."""
    fp32 = nl.float32
    bf16 = nl.bfloat16
    ident_i = sb.tile((tokens, tokens), nl.int32)
    nisa.iota(dst=ident_i, pattern=[[1, tokens]], offset=0, channel_multiplier=-1)
    ident = sb.tile((tokens, tokens), bf16)
    nisa.tensor_scalar(dst=ident, data=ident_i, op0=nl.equal, operand0=0.0)
    value_row = sb.tile((tokens, _P_MAX), bf16)
    nisa.memset(dst=value_row, value=0.0)
    nisa.tensor_scalar(dst=value_row, data=value_row, op0=nl.add, operand0=column)
    value_psum = nl.ndarray((_P_MAX, tokens), dtype=fp32, buffer=nl.psum)
    nisa.nc_matmul(dst=value_psum, stationary=value_row, moving=ident)
    values = sb.tile((_P_MAX, tokens), fp32)
    nisa.tensor_copy(dst=values, src=value_psum)
    return values


def _seq_value(seq_iota, seq, values, rows, tokens, sb):
    """Return the FP32 ``[rows, 1]`` value ``values[p, seq[p]]`` of each partition (0 when ``seq[p]`` is past the B
    columns): a one-hot of the sequence against ``seq_iota`` ``[128, B]``, times ``values`` ``[128, B]``, summed."""
    fp32 = nl.float32
    onehot = sb.tile((_P_MAX, tokens), fp32)
    nisa.tensor_scalar(dst=onehot[0:rows], data=seq_iota[0:rows], op0=nl.equal, operand0=seq)
    nisa.tensor_tensor(dst=onehot[0:rows], data1=onehot[0:rows], data2=values[0:rows], op=nl.multiply)
    out = sb.tile((_P_MAX, 1), fp32)
    nisa.tensor_reduce(dst=out[0:rows], op=nl.add, data=onehot[0:rows], axis=1)
    return out[0:rows]


def _clamp_block_fields(fields, item_f, rows, tokens, clamp_block, seq_iota, full_groups, start_f, sb, lo_groups=None):
    """``_pack_items`` ``clamp_block``: in the block of ``clamp_block`` items at row R (``start_f``), item R + b takes the
    last key group g_b - 1 of sequence b (``full_groups`` ``[128, B]``: g_b on every partition) when g_b >= 1, else it is
    a null item; the item fields [sequence, start, valid, group] of the other rows stay. Small integers, exact in FP32.
    ``lo_groups`` (``_pack_items`` ``group_range``): the first group of each sequence on every partition; the block item
    then takes group lo_b + g_b - 1."""
    fp32 = nl.float32
    vector = nisa.engine.vector
    k = sb.tile((_P_MAX, 1), fp32)
    nisa.tensor_tensor(dst=k[0:rows], data1=item_f[0:rows], data2=start_f[0:rows], op=nl.subtract)
    in_block = sb.tile((_P_MAX, 1), fp32)
    nisa.tensor_scalar(dst=in_block[0:rows], data=k[0:rows], op0=nl.greater_equal, operand0=0.0, engine=vector)
    below = sb.tile((_P_MAX, 1), fp32)
    nisa.tensor_scalar(dst=below[0:rows], data=k[0:rows], op0=nl.less, operand0=float(clamp_block), engine=vector)
    nisa.tensor_tensor(dst=in_block[0:rows], data1=in_block[0:rows], data2=below[0:rows], op=nl.multiply)
    # g_k: the full group count of sequence k (0 past the sequences).
    onehot = sb.tile((_P_MAX, tokens), fp32)
    nisa.tensor_scalar(dst=onehot[0:rows], data=seq_iota[0:rows], op0=nl.equal, operand0=k[0:rows], engine=vector)
    nisa.tensor_tensor(dst=onehot[0:rows], data1=onehot[0:rows], data2=full_groups[0:rows], op=nl.multiply)
    g_k = sb.tile((_P_MAX, 1), fp32)
    nisa.tensor_reduce(dst=g_k[0:rows], op=nl.add, data=onehot[0:rows], axis=1)
    # valid: in the block, a sequence (k < B, via g_k = 0 past them) with a key group on this rank.
    valid = sb.tile((_P_MAX, 1), fp32)
    nisa.tensor_scalar(dst=valid[0:rows], data=g_k[0:rows], op0=nl.greater_equal, operand0=1.0, engine=vector)
    nisa.tensor_tensor(dst=valid[0:rows], data1=valid[0:rows], data2=in_block[0:rows], op=nl.multiply)
    # sequence = valid ? k : B; group = valid ? g_k - 1 : 0.
    seq_b = sb.tile((_P_MAX, 1), fp32)
    nisa.tensor_scalar(dst=seq_b[0:rows], data=k[0:rows], op0=nl.add, operand0=-float(tokens), engine=vector)
    nisa.tensor_tensor(dst=seq_b[0:rows], data1=seq_b[0:rows], data2=valid[0:rows], op=nl.multiply)
    nisa.tensor_scalar(dst=seq_b[0:rows], data=seq_b[0:rows], op0=nl.add, operand0=float(tokens), engine=vector)
    group_b = sb.tile((_P_MAX, 1), fp32)
    nisa.tensor_scalar(dst=group_b[0:rows], data=g_k[0:rows], op0=nl.add, operand0=-1.0, engine=vector)
    if lo_groups is not None:
        nisa.tensor_tensor(
            dst=group_b[0:rows],
            data1=group_b[0:rows],
            data2=_seq_value(seq_iota, k[0:rows], lo_groups, rows, tokens, sb),
            op=nl.add,
        )
    nisa.tensor_tensor(dst=group_b[0:rows], data1=group_b[0:rows], data2=valid[0:rows], op=nl.multiply)
    # fields = fields + in_block * (block fields - fields), for the sequence, valid and group fields.
    for column, value in ((0, seq_b), (2, valid), (3, group_b)):
        delta = sb.tile((_P_MAX, 1), fp32)
        nisa.tensor_tensor(
            dst=delta[0:rows], data1=value[0:rows], data2=fields[0:rows, column : column + 1], op=nl.subtract
        )
        nisa.tensor_tensor(dst=delta[0:rows], data1=delta[0:rows], data2=in_block[0:rows], op=nl.multiply)
        nisa.tensor_tensor(
            dst=fields[0:rows, column : column + 1],
            data1=fields[0:rows, column : column + 1],
            data2=delta[0:rows],
            op=nl.add,
        )


def _pack_items(
    position_i,
    rank_f,
    block_table,
    tag,
    sb,
    slots=None,
    plan_sb=None,
    shared=None,
    tail_slots=None,
    tail_round=False,
    with_ti=True,
    group_extra=0,
    null_threshold=0.0,
    pad_skip=False,
    live_i=None,
    clamp_block=0,
    group_range=None,
    group_lo_hbm=None,
):
    """Build the item table of the packed loop (``_packed_key_loop``) for one decode step.

    ``position_i`` ``[B, 1]`` INT32 and ``rank_f`` ``[B, 1]`` FP32 (SBUF) hold the position and the TP rank of
    each sequence on its partition. Return the plan: the SBUF ``[1, 1]`` iteration count, the SBUF
    ``[96, B + 1, 33]`` uint16 TI offsets of the state columns, and per item the block-table row (sequence,
    group) and the row template [-, sequence, mask threshold, valid]. Every MLA layer of a step has the same
    items, so a caller can build the plan once (``mla_pack_plan``); each layer gathers its own page ids.
    ``plan_sb`` (default ``sb``) allocates the plan tiles; the groups of a staged layer (``shared``) share them.
    With ``tail_slots`` the iteration tile holds [main iterations | first tail row | end tail row], the tail rows
    in units of ``tail_slots`` items; with two or more tail loops, a [first | end] pair for each, in its own units.
    With ``tail_round`` the loops split u = ceil(items / t) units of the last tail count t: the main loop runs
    u >> log2(slots / t) iterations and a tail loop of w items ends at row u >> log2(w / t). A remainder that would
    run as many iterations of a tail loop as one iteration of the loop before it holds then runs as one iteration of
    the larger loop: the same slots in fewer iterations. The rows past the items are null items, so the state is
    bitwise the same. Without ``with_ti`` the plan has no TI offsets: they depend on B only, so the packed loop builds
    them (``_pack_ti``).

    The verify of Q rows per sequence (``_packed_key_loop`` ``queries``) counts the key groups of a sequence from its
    position plus ``group_extra`` (Q - 1, so the items hold the keys of its last row), and gives each null item the mask
    threshold ``null_threshold`` (-Q) in place of 0, so that the row thresholds ``threshold + j`` of a null item still
    mask every key. The defaults keep the program.

    ``pad_skip``: a padding sequence has no key group, so it takes no item; with ``clamp_block`` its block item is a null
    item too. The other sequences keep their group counts (a select), so their items and bits do not change. ``live_i``:
    the INT32 ``[B, 1]`` SBUF live-row mask of the runner; with it a sequence with mask 0 is the padding row (the served
    form). Without it a sequence at position 0 is the padding row (TEST ONLY, NOT SERVABLE: a live row can sit
    at position 0 in serving). Default off.

    ``clamp_block`` (``_packed_key_loop`` tail-only page clamp; a power of two from 2 to ``slots``, at least the
    sequences): the items hold the key groups of every sequence but its last one, in the same order, then from row
    ``R`` (the end of the main and tail loops, a multiple of the last loop's width) one block of ``clamp_block`` items:
    the last key group of sequence b in item ``R + b`` (a null item when the sequence has no key group on this rank),
    then null items. Each state column still adds its groups in order, so the state is bitwise the same. The iteration
    tile then ends with the first and end rows of the block's loop, in units of ``clamp_block``. Only the last key group
    of a sequence holds keys at or past its position (every key of an earlier group is below ``position + Q - 3969``), so
    with the page clamp the loop clamps the pages of that block only.

    ``group_range`` (the long-sequence split, ``mla_block_attention`` ``split``): None, or the FP32 ``[B, 2]`` SBUF tile
    (lo, hi) of each sequence in key groups: sequence b takes only its groups lo .. min(g_b, hi) - 1 (g_b counted from
    its position as above), so the items of a sequence start at group lo. Small integers, exact in FP32. (0, group
    count) for every sequence gives the items without ``group_range``. ``group_lo_hbm`` (with ``group_range``): None, or
    ``(lo, first, stride)``, an INT32 HBM copy of the lo column (one value every ``stride`` words) from entry ``first``;
    its values reach every partition by one broadcast DMA, in place of the GpSimd iota identity and the matmul of
    ``_per_partition_values``.
    """
    bf16 = nl.bfloat16
    fp32 = nl.float32
    int32 = nl.int32
    hwdge = nisa.dge_mode.hwdge
    sync = nisa.engine.sync
    tokens, local_blocks = block_table.shape
    group_count = local_blocks // _GROUP_BLOCKS
    slots = pack_slots(local_blocks, slots, tokens)
    tail = _tail_slots(slots, tail_slots)
    kernel_assert(not tail_round or tail is not None, "pack tail_round needs pack tail slots")
    last_width = tail[-1] if tail else slots
    kernel_assert(
        not clamp_block or (clamp_block in (2, 4, 8, 16, 32, 64) and tokens <= clamp_block <= last_width),
        f"pack clamp_block is a power of two from {tokens} sequences to {last_width}, got {clamp_block}",
    )
    # Without a tail loop the iteration count goes to the whole iteration tile, whose clamp_block columns follow it.
    kernel_assert(not clamp_block or tail is not None, "pack clamp_block needs pack tail slots")
    # With ``clamp_block`` the rows also hold the block and its next row (the prefetch of its loop).
    extra_rows = last_width + 2 * clamp_block if clamp_block else 0
    max_iterations = -(-(tokens * group_count + extra_rows) // slots)
    item_rows = max_iterations * slots
    item_tiles = -(-item_rows // _P_MAX)
    # The plan tiles live through the loops of every MLA layer: explicit addresses without a region.
    psb = sb if plan_sb is None else plan_sb

    def plan_tile(name, shape, dtype, slot):
        return _shared(shared, f"plan_{name}", lambda: _plan_tile(psb, shape, dtype, f"{tag}_mla_plan_{name}", slot))

    iteration_cols = (1 if tail is None else 1 + 2 * len(tail)) + (2 if clamp_block else 0)
    iteration_i = plan_tile("iterations", (1, iteration_cols), int32, 0)
    ti_u = plan_tile("ti", (_HEADS, tokens + 1, _TI_COLS), nl.uint16, 1) if with_ti else None
    gather_rows = plan_tile("gather", (_P_MAX, item_tiles), int32, 2)
    row_template = plan_tile("rows", (_P_MAX, item_tiles, _ITEM_FIELDS), fp32, 3)
    sb.open_scope("pack_items")

    # Step P1: the key groups and the local position of each sequence, as in _key_loops.
    position_f = sb.tile((tokens, 1), fp32)
    nisa.tensor_copy(dst=position_f, src=position_i)
    seq_f = sb.tile((tokens, 2), fp32)
    nisa.scalar_tensor_tensor(
        dst=seq_f[0:tokens, 1:2],
        data=rank_f,
        op0=nl.multiply,
        operand0=-float(_BLOCK),
        op1=nl.add,
        operand1=position_f,
    )
    if group_extra:
        nisa.tensor_scalar(
            dst=seq_f[0:tokens, 0:1],
            data=seq_f[0:tokens, 1:2],
            op0=nl.add,
            operand0=float(_GROUP_STRIDE - 1 + group_extra),
            engine=nisa.engine.vector,
        )
    else:
        nisa.tensor_scalar(
            dst=seq_f[0:tokens, 0:1], data=seq_f[0:tokens, 1:2], op0=nl.add, operand0=float(_GROUP_STRIDE - 1)
        )
    seq_i = sb.tile((tokens, 2), int32)
    nisa.tensor_copy(dst=seq_i, src=seq_f)
    nisa.tensor_scalar(dst=seq_i[0:tokens, 0:1], data=seq_i[0:tokens, 0:1], op0=nl.right_shift, operand0=14)
    nisa.tensor_scalar(dst=seq_i[0:tokens, 0:1], data=seq_i[0:tokens, 0:1], op0=nl.minimum, operand0=float(group_count))
    nisa.tensor_copy(dst=seq_f[0:tokens, 0:1], src=seq_i[0:tokens, 0:1])
    lo_groups = None
    if group_range is not None:
        # The split range: g_b = max(min(g_b, hi) - lo, 0) groups from group lo.
        nisa.tensor_tensor(
            dst=seq_f[0:tokens, 0:1], data1=seq_f[0:tokens, 0:1], data2=group_range[0:tokens, 1:2], op=nl.minimum
        )
        nisa.tensor_tensor(
            dst=seq_f[0:tokens, 0:1], data1=seq_f[0:tokens, 0:1], data2=group_range[0:tokens, 0:1], op=nl.subtract
        )
        nisa.tensor_scalar(dst=seq_f[0:tokens, 0:1], data=seq_f[0:tokens, 0:1], op0=nl.maximum, operand0=0.0)
        nisa.tensor_copy(dst=seq_i[0:tokens, 0:1], src=seq_f[0:tokens, 0:1])
        # lo of every sequence on every partition ([128, B]), for the item groups below.
        if group_lo_hbm is None:
            lo_groups = _per_partition_values(group_range[0:tokens, 0:1], tokens, sb)
        else:
            lo_hbm, lo_first, lo_stride = group_lo_hbm
            lo_all_i = sb.tile((_P_MAX, tokens), int32)
            nisa.dma_copy(
                dst=lo_all_i,
                src=lo_hbm.ap(pattern=[[0, _P_MAX], [lo_stride, tokens]], offset=lo_first * lo_stride),
                dge_mode=hwdge,
                engine=sync,
                name=f"{tag}_mla_pack_lo_all",
            )
            lo_groups = sb.tile((_P_MAX, tokens), fp32)
            nisa.tensor_copy(dst=lo_groups, src=lo_all_i, engine=nisa.engine.vector)
    groups_full = None
    if clamp_block:
        # The main and tail loops take every key group but the last of each sequence: max(g - 1, 0) groups.
        groups_full = sb.tile((tokens, 1), fp32)
        nisa.tensor_copy(dst=groups_full, src=seq_f[0:tokens, 0:1])
        nisa.tensor_scalar(
            dst=seq_f[0:tokens, 0:1],
            data=seq_f[0:tokens, 0:1],
            op0=nl.add,
            operand0=-1.0,
            op1=nl.maximum,
            operand1=0.0,
            engine=nisa.engine.vector,
        )
    if pad_skip:
        # A select: a sequence at position 0 takes 0 groups, a live one keeps its count.
        pad_f = sb.tile((tokens, 1), fp32)
        if live_i is None:
            nisa.tensor_scalar(dst=pad_f, data=position_f, op0=nl.less_equal, operand0=0.0)
        else:
            live_f = sb.tile((tokens, 1), fp32)
            nisa.tensor_copy(dst=live_f, src=live_i)
            nisa.tensor_scalar(dst=pad_f, data=live_f, op0=nl.less_equal, operand0=0.0)
        pad_u8 = sb.tile((tokens, 1), nl.uint8)
        nisa.tensor_copy(dst=pad_u8, src=pad_f)
        zero_f = sb.tile((tokens, 1), fp32)
        nisa.memset(dst=zero_f, value=0.0)
        nisa.tensor_copy_predicated(dst=seq_f[0:tokens, 0:1], src=zero_f, predicate=pad_u8)
        nisa.tensor_copy(dst=seq_i[0:tokens, 0:1], src=seq_f[0:tokens, 0:1])
        if groups_full is not None:
            # The clamp block: a padding sequence takes a null item there too (g_b = 0).
            nisa.tensor_copy_predicated(dst=groups_full, src=zero_f, predicate=pad_u8)
    # The local positions go to HBM, so each item partition gathers the one of its sequence.
    position_hbm = nl.ndarray((tokens, 1), dtype=int32, buffer=nl.hbm, name=f"{tag}_mla_pack_position")
    nisa.dma_copy(
        dst=position_hbm, src=seq_i[0:tokens, 1:2], dge_mode=hwdge, engine=sync, name=f"{tag}_mla_pack_position_store"
    )

    # Step P2: the ends c_b = g_0 + ... + g_b and the starts c_b - g_b of the sequences on every partition: a
    # matmul with an upper-triangular 0/1 operand over the sequences (small integers, exact in BF16 and FP32).
    tri_i = sb.tile((tokens, 2 * tokens), int32)
    nisa.iota(dst=tri_i, pattern=[[0, 2], [1, tokens]], offset=0, channel_multiplier=-1)
    tri = sb.tile((tokens, 2 * tokens), bf16)
    nisa.tensor_scalar(dst=tri[0:tokens, 0:tokens], data=tri_i[0:tokens, 0:tokens], op0=nl.greater_equal, operand0=0.0)
    nisa.tensor_scalar(
        dst=tri[0:tokens, tokens : 2 * tokens],
        data=tri_i[0:tokens, tokens : 2 * tokens],
        op0=nl.greater_equal,
        operand0=1.0,
    )
    groups_row = sb.tile((tokens, _P_MAX), bf16)
    nisa.memset(dst=groups_row, value=0.0)
    nisa.tensor_scalar(dst=groups_row, data=groups_row, op0=nl.add, operand0=seq_f[0:tokens, 0:1])
    bounds_psum = nl.ndarray((_P_MAX, 2 * tokens), dtype=fp32, buffer=nl.psum)
    nisa.nc_matmul(dst=bounds_psum, stationary=groups_row, moving=tri)
    bounds = sb.tile((_P_MAX, 2 * tokens), fp32)
    nisa.tensor_copy(dst=bounds, src=bounds_psum)
    # Iterations = (item count + slots - 1) >> log2(slots), from partition 0.
    if tail is None:
        nisa.tensor_copy(dst=iteration_i, src=bounds[0:1, tokens - 1 : tokens])
        nisa.tensor_scalar(dst=iteration_i, data=iteration_i, op0=nl.add, operand0=float(slots - 1))
        nisa.tensor_scalar(dst=iteration_i, data=iteration_i, op0=nl.right_shift, operand0=slots.bit_length() - 1)
    elif tail_round:
        # u = ceil(items / t) in the end column of the last tail loop; main M = u >> log2(slots / t), and each tail
        # loop runs from the end of the loop before it to u >> log2(width / t), in units of its own width.
        last = tail[-1]
        u_i = iteration_i[0:1, 2 * len(tail) : 2 * len(tail) + 1]
        nisa.tensor_copy(dst=u_i, src=bounds[0:1, tokens - 1 : tokens])
        nisa.tensor_scalar(dst=u_i, data=u_i, op0=nl.add, operand0=float(last - 1))
        nisa.tensor_scalar(dst=u_i, data=u_i, op0=nl.right_shift, operand0=last.bit_length() - 1)
        main_i = iteration_i[0:1, 0:1]
        nisa.tensor_scalar(dst=main_i, data=u_i, op0=nl.right_shift, operand0=slots.bit_length() - last.bit_length())
        width, last_i = slots, main_i
        for level, tail_count in enumerate(tail):
            first_i, end_i = (iteration_i[0:1, column : column + 1] for column in (2 * level + 1, 2 * level + 2))
            nisa.tensor_scalar(
                dst=first_i, data=last_i, op0=nl.left_shift, operand0=width.bit_length() - tail_count.bit_length()
            )
            if level < len(tail) - 1:
                shift = tail_count.bit_length() - last.bit_length()
                nisa.tensor_scalar(dst=end_i, data=u_i, op0=nl.right_shift, operand0=shift)
            width, last_i = tail_count, end_i
    else:
        # main M = items >> log2(slots); the rows of a tail loop run from the end of the loop before it, in units of
        # its own slots, to items >> log2(tail), or to ceil(items / tail) for the last tail loop.
        for column in range(0, 2 * len(tail) + 1, 2):
            nisa.tensor_copy(dst=iteration_i[0:1, column : column + 1], src=bounds[0:1, tokens - 1 : tokens])
        main_i = iteration_i[0:1, 0:1]
        nisa.tensor_scalar(dst=main_i, data=main_i, op0=nl.right_shift, operand0=slots.bit_length() - 1)
        width, last_i = slots, main_i
        for level, tail_count in enumerate(tail):
            first_i, end_i = (iteration_i[0:1, column : column + 1] for column in (2 * level + 1, 2 * level + 2))
            shift = width.bit_length() - tail_count.bit_length()
            nisa.tensor_scalar(dst=first_i, data=last_i, op0=nl.left_shift, operand0=shift)
            if level == len(tail) - 1:
                nisa.tensor_scalar(dst=end_i, data=end_i, op0=nl.add, operand0=float(tail_count - 1))
            nisa.tensor_scalar(dst=end_i, data=end_i, op0=nl.right_shift, operand0=tail_count.bit_length() - 1)
            width, last_i = tail_count, end_i

    clamp_rows = None
    if clamp_block:
        # The full group count g_b of every sequence on every partition (a ones-identity matmul), and the first row R of
        # the block: the item count rounded up to the width of the last loop, on every partition.
        ident_i = sb.tile((tokens, tokens), int32)
        nisa.iota(dst=ident_i, pattern=[[1, tokens]], offset=0, channel_multiplier=-1)
        ident = sb.tile((tokens, tokens), bf16)
        nisa.tensor_scalar(dst=ident, data=ident_i, op0=nl.equal, operand0=0.0)
        full_row = sb.tile((tokens, _P_MAX), bf16)
        nisa.memset(dst=full_row, value=0.0)
        nisa.tensor_scalar(dst=full_row, data=full_row, op0=nl.add, operand0=groups_full[0:tokens, 0:1])
        full_psum = nl.ndarray((_P_MAX, tokens), dtype=fp32, buffer=nl.psum)
        nisa.nc_matmul(dst=full_psum, stationary=full_row, moving=ident)
        full_groups = sb.tile((_P_MAX, tokens), fp32)
        nisa.tensor_copy(dst=full_groups, src=full_psum)
        start_i = sb.tile((_P_MAX, 1), int32)
        nisa.tensor_copy(dst=start_i, src=bounds[0:_P_MAX, tokens - 1 : tokens])
        nisa.tensor_scalar(dst=start_i, data=start_i, op0=nl.add, operand0=float(last_width - 1))
        nisa.tensor_scalar(dst=start_i, data=start_i, op0=nl.right_shift, operand0=last_width.bit_length() - 1)
        nisa.tensor_scalar(dst=start_i, data=start_i, op0=nl.left_shift, operand0=last_width.bit_length() - 1)
        start_f = sb.tile((_P_MAX, 1), fp32)
        nisa.tensor_copy(dst=start_f, src=start_i)
        # The block's loop: [R / clamp_block, R / clamp_block + 1) in its own units, from partition 0.
        first_col = iteration_cols - 2
        nisa.tensor_scalar(
            dst=iteration_i[0:1, first_col : first_col + 1],
            data=start_i[0:1, 0:1],
            op0=nl.right_shift,
            operand0=clamp_block.bit_length() - 1,
        )
        nisa.tensor_scalar(
            dst=iteration_i[0:1, first_col + 1 : first_col + 2],
            data=iteration_i[0:1, first_col : first_col + 1],
            op0=nl.add,
            operand0=1.0,
        )
        clamp_rows = (full_groups, start_f)

    # Step P3: the item rows [4 page ids | sequence | mask threshold | 0 | 0], 128 items per partition tile. The
    # sequence of item i is the count of the sequences that end at or before i (B for a null item); its group
    # is i minus the start of the sequence.
    first_ended = sb.tile((_P_MAX, tokens), fp32)
    nisa.tensor_scalar(dst=first_ended, data=bounds[0:_P_MAX, 0:tokens], op0=nl.less_equal, operand0=0.0)
    first_row = sb.tile((_P_MAX, 1), fp32)
    nisa.tensor_reduce(dst=first_row, op=nl.add, data=first_ended, axis=1)
    first_seq = None
    if lo_groups is not None:
        first_seq = sb.tile((_P_MAX, 1), fp32)
        nisa.tensor_scalar(dst=first_seq, data=first_row, op0=nl.minimum, operand0=float(tokens - 1))
    nisa.tensor_scalar(
        dst=first_row,
        data=first_row,
        op0=nl.minimum,
        operand0=float(tokens - 1),
        op1=nl.multiply,
        operand1=float(group_count),
    )
    seq_iota_i = sb.tile((_P_MAX, tokens), int32)
    nisa.iota(dst=seq_iota_i, pattern=[[1, tokens]], offset=0)
    seq_iota = sb.tile((_P_MAX, tokens), fp32)
    nisa.tensor_copy(dst=seq_iota, src=seq_iota_i)
    if lo_groups is not None:
        # A null item reads group lo of the first sequence with a key group: a page of the sequence's range.
        nisa.tensor_tensor(
            dst=first_row,
            data1=first_row,
            data2=_seq_value(seq_iota, first_seq, lo_groups, _P_MAX, tokens, sb),
            op=nl.add,
        )
    for item_tile in range(item_tiles):
        rows = min(_P_MAX, item_rows - item_tile * _P_MAX)
        item_i = sb.tile((_P_MAX, 1), int32)
        nisa.iota(dst=item_i[0:rows], pattern=[[0, 1]], offset=item_tile * _P_MAX, channel_multiplier=1)
        item_f = sb.tile((_P_MAX, 1), fp32)
        nisa.tensor_copy(dst=item_f[0:rows], src=item_i[0:rows])
        ended = sb.tile((_P_MAX, tokens), fp32)
        nisa.tensor_scalar(dst=ended[0:rows], data=bounds[0:rows, 0:tokens], op0=nl.less_equal, operand0=item_f[0:rows])
        # fields: [sequence, start, valid, group]; a null item has valid 0 and group 0.
        fields = sb.tile((_P_MAX, 4), fp32)
        nisa.tensor_reduce(dst=fields[0:rows, 0:1], op=nl.add, data=ended[0:rows], axis=1)
        onehot = sb.tile((_P_MAX, tokens), fp32)
        nisa.tensor_scalar(dst=onehot[0:rows], data=seq_iota[0:rows], op0=nl.equal, operand0=fields[0:rows, 0:1])
        nisa.tensor_tensor(
            dst=onehot[0:rows], data1=onehot[0:rows], data2=bounds[0:rows, tokens : 2 * tokens], op=nl.multiply
        )
        nisa.tensor_reduce(dst=fields[0:rows, 1:2], op=nl.add, data=onehot[0:rows], axis=1)
        nisa.tensor_scalar(dst=fields[0:rows, 2:3], data=fields[0:rows, 0:1], op0=nl.less, operand0=float(tokens))
        nisa.tensor_tensor(dst=fields[0:rows, 3:4], data1=item_f[0:rows], data2=fields[0:rows, 1:2], op=nl.subtract)
        nisa.tensor_tensor(
            dst=fields[0:rows, 3:4], data1=fields[0:rows, 3:4], data2=fields[0:rows, 2:3], op=nl.multiply
        )
        if lo_groups is not None:
            # group += lo of the item's sequence (0 for a null item: its sequence is B, past every one-hot column).
            nisa.tensor_tensor(
                dst=fields[0:rows, 3:4],
                data1=fields[0:rows, 3:4],
                data2=_seq_value(seq_iota, fields[0:rows, 0:1], lo_groups, rows, tokens, sb),
                op=nl.add,
            )
        if clamp_rows is not None:
            _clamp_block_fields(
                fields, item_f, rows, tokens, clamp_block, seq_iota, *clamp_rows, sb, lo_groups=lo_groups
            )
        # Gather indices: the table row (sequence, group) and the sequence, clamped into range for a null item.
        # A null item reads the table row of item 0 (group 0 of the first sequence with a key group), so every
        # page that a slot reads is a page of a sequence (written, or zeroed by a caller that zeroes its pages).
        gather_f = sb.tile((_P_MAX, 2), fp32)
        nisa.tensor_scalar(
            dst=gather_f[0:rows, 1:2], data=fields[0:rows, 0:1], op0=nl.minimum, operand0=float(tokens - 1)
        )
        nisa.scalar_tensor_tensor(
            dst=gather_f[0:rows, 0:1],
            data=gather_f[0:rows, 1:2],
            op0=nl.multiply,
            operand0=float(group_count),
            op1=nl.add,
            operand1=fields[0:rows, 3:4],
        )
        # row = (row - row of item 0) * valid + row of item 0.
        nisa.scalar_tensor_tensor(
            dst=gather_f[0:rows, 0:1],
            data=gather_f[0:rows, 0:1],
            op0=nl.subtract,
            operand0=first_row[0:rows, 0:1],
            op1=nl.multiply,
            operand1=fields[0:rows, 2:3],
        )
        nisa.tensor_scalar(
            dst=gather_f[0:rows, 0:1], data=gather_f[0:rows, 0:1], op0=nl.add, operand0=first_row[0:rows, 0:1]
        )
        gather_i = sb.tile((_P_MAX, 2), int32)
        nisa.tensor_copy(dst=gather_i[0:rows], src=gather_f[0:rows])
        nisa.tensor_copy(dst=gather_rows[0:rows, item_tile : item_tile + 1], src=gather_i[0:rows, 0:1])
        local_i = sb.tile((_P_MAX, 1), int32)
        nisa.dma_copy(
            dst=local_i[0:rows],
            src=position_hbm.ap(
                pattern=[[1, rows], [1, 1]], offset=0, vector_offset=gather_i[0:rows, 1:2], indirect_dim=0
            ),
            name=f"{tag}_mla_pack_local_{item_tile}",
        )
        # The row template [pages (per layer) | sequence | threshold | valid | 0]; threshold = local position -
        # 16384 g, 0 for a null item.
        row_f = row_template[0:_P_MAX, item_tile, 0:_ITEM_FIELDS]
        nisa.memset(dst=row_f[0:rows], value=0.0)
        nisa.tensor_copy(dst=row_f[0:rows, 6:7], src=fields[0:rows, 2:3])
        nisa.tensor_copy(dst=row_f[0:rows, 4:5], src=fields[0:rows, 0:1])
        nisa.tensor_copy(dst=row_f[0:rows, 5:6], src=local_i[0:rows])
        nisa.scalar_tensor_tensor(
            dst=row_f[0:rows, 5:6],
            data=fields[0:rows, 3:4],
            op0=nl.multiply,
            operand0=-float(_GROUP_STRIDE),
            op1=nl.add,
            operand1=row_f[0:rows, 5:6],
        )
        nisa.tensor_tensor(dst=row_f[0:rows, 5:6], data1=row_f[0:rows, 5:6], data2=fields[0:rows, 2:3], op=nl.multiply)
        if null_threshold:
            # threshold += (1 - valid) * null_threshold: a null item takes null_threshold, a real item keeps its own.
            null_f = sb.tile((_P_MAX, 1), fp32)
            nisa.tensor_scalar(
                dst=null_f[0:rows],
                data=fields[0:rows, 2:3],
                op0=nl.multiply,
                operand0=-float(null_threshold),
                op1=nl.add,
                operand1=float(null_threshold),
                engine=nisa.engine.vector,
            )
            nisa.tensor_tensor(
                dst=row_f[0:rows, 5:6],
                data1=row_f[0:rows, 5:6],
                data2=null_f[0:rows],
                op=nl.add,
                engine=nisa.engine.vector,
            )

    if with_ti:
        _pack_ti(ti_u, tokens, sb)
    sb.close_scope()
    return {
        "iterations": iteration_i,
        **({"ti": ti_u} if with_ti else {}),
        "gather": gather_rows,
        "rows": row_template,
        "item_rows": item_rows,
        "group_count": group_count,
        "slots": slots,
        "tail_slots": tail,
        **({"clamp_block": clamp_block} if clamp_block else {}),
    }


def _pack_ti(ti_u, tokens, sb):
    """Write the TI offsets of the packed loop state into ``ti_u`` ``[96, B + 1, 33]`` uint16: b * 513 + i in the snake
    layout (offset i at [i % 16, i // 16], the same for each group of 16). ``sb`` allocates the scratch tiles."""
    int32 = nl.int32
    fp32 = nl.float32
    # The partitions of the state (96 heads, or 128 (row, head) columns with ``pack_p2``).
    parts = ti_u.shape[0]
    ti_i = sb.tile((parts, tokens + 1, _TI_COLS), int32)
    nisa.iota(dst=ti_i, pattern=[[_STATE_WIDTH, tokens + 1], [_TI_GROUP, _TI_COLS]], offset=0, channel_multiplier=1)
    group_base_i = sb.tile((parts, 1), int32)
    nisa.iota(dst=group_base_i, pattern=[[0, 1]], offset=0, channel_multiplier=1)
    nisa.tensor_scalar(
        dst=group_base_i, data=group_base_i, op0=nl.right_shift, operand0=4, op1=nl.left_shift, operand1=4
    )
    group_base = sb.tile((parts, 1), fp32)
    nisa.tensor_copy(dst=group_base, src=group_base_i)
    ti_f = sb.tile((parts, tokens + 1, _TI_COLS), fp32)
    nisa.tensor_copy(dst=ti_f, src=ti_i)
    nisa.tensor_scalar(dst=ti_f, data=ti_f, op0=nl.subtract, operand0=group_base)
    nisa.tensor_copy(dst=ti_u, src=ti_f)


def _pack_layer_items(
    plan,
    block_table,
    num_pages,
    tag,
    sb,
    skip_null_page=False,
    item_hbm=None,
    page_runs=0,
    run_scale=None,
    null_copy_vector=False,
    null_runs=False,
):
    """Write the item rows of one MLA layer: the page ids of its block table into the rows of the plan.

    ``item_hbm`` (default a new HBM tensor) takes the rows.

    With ``skip_null_page`` (zeroed pages, no clamp), a table entry 0 (an unallocated block: the null page, which
    other layers may write, for example the KDA state of a padded sequence) takes the first page of the item's own
    group. That page holds the first key of the group, so it is allocated, and its masked rows are finite.

    With ``page_runs`` R (2 or 4) the rows take the page-run form of ``_packed_key_loop``: the first page of each
    run of R pages, then the sequence ([page 0 | sequence | pages 2, 3 | ...] for 4, [page 0 | page 2 | sequence |
    page 3 | ...] for 2), each first page at most the pool size minus R, so a run stays in the pool. ``run_scale``
    (an SBUF ``[96, B]`` FP32 tile) gets 1.0 for each sequence whose items hold runs of R pages (page ``i`` = page
    ``i - i % R`` + ``i % R``, after the clamp), else NaN. With ``null_runs`` an item of null pages only (a padded
    sequence: position 0 and a zero block table row) counts as runs, so a padded sequence gets 1.0 and finite rows.
    With ``plan["pad_src"]`` (``tkg_mla_block`` ``pad_runs``: the ``[B, 1]`` INT32 HBM pad source, the runner's live rows or
    the positions, and the first row of the pass) a sequence whose source value is at most 0 (a padding row) gets 1.0
    whatever its pages, and its items take kernel page 0 in every entry: selects, so a live sequence keeps its pages and
    its scale. The runner's row-runs fill gives a padding row page 1 in every entry, so its runs would read pages (1, 2),
    and page 2 starts the KDA null block, whose FP32 state bytes hold FP8 NaN codes; with page 0 the runs read the zeroed
    null block. Null items have the pages of item 0.
    """
    fp32 = nl.float32
    int32 = nl.int32
    # The pages DMA below gathers block_table rows into an INT32 tile: an INT64 table would give it a 64-bit dtype.
    kernel_assert(block_table.dtype == int32, f"block_table must be INT32 page ids, got {block_table.dtype}")
    tokens = block_table.shape[0]
    runs = _GROUP_BLOCKS // page_runs if page_runs else 0
    item_rows = plan["item_rows"]
    group_count = plan["group_count"]
    # One more iteration of rows, a copy of the first one, so the prefetch of the last iteration reads valid pages.
    pad_rows = plan["slots"]
    if item_hbm is None:
        item_hbm = nl.ndarray(
            (item_rows + pad_rows, _ITEM_FIELDS), dtype=int32, buffer=nl.hbm, name=f"{tag}_mla_pack_items"
        )
    table_rows = block_table.reshape((tokens * group_count, _GROUP_BLOCKS))
    sb.open_scope("pack_layer")
    if runs:
        kernel_assert(num_pages >= page_runs, f"pack_page_runs needs at least {page_runs} pages")
        # Per sequence, the items whose pages are not one run: a ones matmul sums the item partitions.
        # The scale goes to every partition of the state (96 heads, or 128 (row, head) columns with ``pack_p2``).
        scale_parts = run_scale.shape[0]
        broken_psum = nl.ndarray((scale_parts, tokens), dtype=fp32, buffer=nl.psum)
        ones = sb.tile((_P_MAX, scale_parts), nl.bfloat16)
        nisa.memset(dst=ones, value=1.0)
        steps_i = sb.tile((_P_MAX, _GROUP_BLOCKS + tokens), int32)
        nisa.iota(dst=steps_i[0:_P_MAX, 0:_GROUP_BLOCKS], pattern=[[1, _GROUP_BLOCKS]], offset=0)
        nisa.iota(dst=steps_i[0:_P_MAX, _GROUP_BLOCKS : _GROUP_BLOCKS + tokens], pattern=[[1, tokens]], offset=0)
        steps = sb.tile((_P_MAX, _GROUP_BLOCKS + tokens), fp32)
        nisa.tensor_copy(dst=steps, src=steps_i)
        pad_src = plan.get("pad_src")
        if pad_src is not None:
            # ``pad_runs``: the pad bit of each sequence on every scale partition (source value <= 0).
            pad_hbm, pad_first = pad_src
            pad_i = sb.tile((scale_parts, tokens), int32)
            nisa.dma_copy(
                dst=pad_i,
                src=pad_hbm.ap(pattern=[[0, scale_parts], [1, tokens]], offset=pad_first),
                dge_mode=nisa.dge_mode.hwdge,
                engine=nisa.engine.sync,
                name=f"{tag}_mla_pad_runs_load",
            )
            # The pad bit of each sequence on every item partition (1.0 for a padding sequence), for the page select.
            pad_rows_i = sb.tile((_P_MAX, tokens), int32)
            nisa.dma_copy(
                dst=pad_rows_i,
                src=pad_hbm.ap(pattern=[[0, _P_MAX], [1, tokens]], offset=pad_first),
                dge_mode=nisa.dge_mode.hwdge,
                engine=nisa.engine.sync,
                name=f"{tag}_mla_pad_pages_load",
            )
            pad_rows_f = sb.tile((_P_MAX, tokens), fp32)
            nisa.tensor_copy(dst=pad_rows_f, src=pad_rows_i, engine=nisa.engine.vector)
            nisa.tensor_scalar(
                dst=pad_rows_f, data=pad_rows_f, op0=nl.less_equal, operand0=0.0, engine=nisa.engine.vector
            )
    for item_tile in range(-(-item_rows // _P_MAX)):
        rows = min(_P_MAX, item_rows - item_tile * _P_MAX)
        pages_i = sb.tile((_P_MAX, _GROUP_BLOCKS), int32)
        nisa.dma_copy(
            dst=pages_i[0:rows],
            src=table_rows.ap(
                pattern=[[_GROUP_BLOCKS, rows], [1, _GROUP_BLOCKS]],
                offset=0,
                vector_offset=plan["gather"][0:rows, item_tile : item_tile + 1],
                indirect_dim=0,
            ),
            name=f"{tag}_mla_pack_pages_{item_tile}",
        )
        # Pages clamped into the pool (a null item has the pages of item 0).
        row_f = sb.tile((_P_MAX, _ITEM_FIELDS), fp32)
        nisa.tensor_copy(dst=row_f[0:rows], src=plan["rows"][0:rows, item_tile, 0:_ITEM_FIELDS])
        nisa.tensor_copy(dst=row_f[0:rows, 0:_GROUP_BLOCKS], src=pages_i[0:rows])
        nisa.tensor_scalar(
            dst=row_f[0:rows, 0:_GROUP_BLOCKS],
            data=row_f[0:rows, 0:_GROUP_BLOCKS],
            op0=nl.minimum,
            operand0=float(num_pages - 1),
            op1=nl.maximum,
            operand1=0.0,
        )
        if runs and pad_src is not None:
            # ``pad_runs``: the items of a padding sequence take page 0 (pages x (1 - pad bit of the item's sequence),
            # exact in FP32); a null item (sequence B) keeps its pages.
            item_seq = sb.tile((_P_MAX, tokens), fp32)
            nisa.tensor_scalar(
                dst=item_seq[0:rows],
                data=steps[0:rows, _GROUP_BLOCKS : _GROUP_BLOCKS + tokens],
                op0=nl.equal,
                operand0=row_f[0:rows, 4:5],
                engine=nisa.engine.vector,
            )
            nisa.tensor_tensor(
                dst=item_seq[0:rows],
                data1=item_seq[0:rows],
                data2=pad_rows_f[0:rows],
                op=nl.multiply,
                engine=nisa.engine.vector,
            )
            keep_f = sb.tile((_P_MAX, 1), fp32)
            # (tensor_reduce takes no engine argument; it runs on VectorE.)
            nisa.tensor_reduce(dst=keep_f[0:rows], op=nl.maximum, data=item_seq[0:rows], axis=1)
            nisa.tensor_scalar(
                dst=keep_f[0:rows],
                data=keep_f[0:rows],
                op0=nl.multiply,
                operand0=-1.0,
                op1=nl.add,
                operand1=1.0,
                engine=nisa.engine.vector,
            )
            nisa.tensor_scalar(
                dst=row_f[0:rows, 0:_GROUP_BLOCKS],
                data=row_f[0:rows, 0:_GROUP_BLOCKS],
                op0=nl.multiply,
                operand0=keep_f[0:rows, 0:1],
                engine=nisa.engine.vector,
            )
        if skip_null_page:
            # page = page + (page == 0) * first page of the group (page ids are exact in FP32). With null_copy_vector
            # all on VectorE: with the copy on ScalarE the 2-server verify hung on trn3 at this cross-engine wait.
            first_f = sb.tile((_P_MAX, 1), fp32)
            nisa.tensor_copy(
                dst=first_f[0:rows],
                src=row_f[0:rows, 0:1],
                **({"engine": nisa.engine.vector} if null_copy_vector else {}),
            )
            null_f = sb.tile((_P_MAX, _GROUP_BLOCKS), fp32)
            nisa.tensor_scalar(dst=null_f[0:rows], data=row_f[0:rows, 0:_GROUP_BLOCKS], op0=nl.equal, operand0=0.0)
            nisa.scalar_tensor_tensor(
                dst=row_f[0:rows, 0:_GROUP_BLOCKS],
                data=null_f[0:rows],
                op0=nl.multiply,
                operand0=first_f[0:rows, 0:1],
                op1=nl.add,
                operand1=row_f[0:rows, 0:_GROUP_BLOCKS],
            )
        row_i = sb.tile((_P_MAX, _ITEM_FIELDS), int32)
        if runs:
            # The run form: the first page of each run, which keeps the run in the pool, then the sequence.
            nisa.tensor_copy(dst=row_i[0:rows, runs + 1 : _ITEM_FIELDS], src=row_f[0:rows, runs + 1 : _ITEM_FIELDS])
            nisa.tensor_copy(dst=row_i[0:rows, runs : runs + 1], src=row_f[0:rows, 4:5])
            for run in range(runs):
                first = run * page_runs
                nisa.tensor_scalar(
                    dst=row_i[0:rows, run : run + 1],
                    data=row_f[0:rows, first : first + 1],
                    op0=nl.minimum,
                    operand0=float(num_pages - page_runs),
                )
        else:
            nisa.tensor_copy(dst=row_i[0:rows], src=row_f[0:rows])
        nisa.dma_copy(
            dst=item_hbm[item_tile * _P_MAX : item_tile * _P_MAX + rows, 0:_ITEM_FIELDS],
            src=row_i[0:rows],
            dge_mode=nisa.dge_mode.hwdge,
            engine=nisa.engine.sync,
            name=f"{tag}_mla_pack_items_store_{item_tile}",
        )
        if item_tile == 0:
            nisa.dma_copy(
                dst=item_hbm[item_rows : item_rows + pad_rows, 0:_ITEM_FIELDS],
                src=row_i[0:pad_rows],
                dge_mode=nisa.dge_mode.hwdge,
                engine=nisa.engine.sync,
                name=f"{tag}_mla_pack_items_pad",
            )
        if runs:
            # After the store: page_i - i is the same for the clamped pages of a run; a broken item counts for its
            # sequence (none for a null item, whose sequence is B).
            base_f = sb.tile((_P_MAX, _GROUP_BLOCKS), fp32)
            nisa.tensor_tensor(
                dst=base_f[0:rows],
                data1=row_f[0:rows, 0:_GROUP_BLOCKS],
                data2=steps[0:rows, 0:_GROUP_BLOCKS],
                op=nl.subtract,
            )
            off_f = sb.tile((_P_MAX, _GROUP_BLOCKS), fp32)
            for first in range(0, _GROUP_BLOCKS, page_runs):
                nisa.tensor_scalar(
                    dst=off_f[0:rows, first : first + page_runs],
                    data=base_f[0:rows, first : first + page_runs],
                    op0=nl.not_equal,
                    operand0=base_f[0:rows, first : first + 1],
                )
            broken_f = sb.tile((_P_MAX, 1), fp32)
            nisa.tensor_reduce(dst=broken_f[0:rows], op=nl.maximum, data=off_f[0:rows], axis=1)
            if null_runs:
                # An item of null pages only (a padded sequence) is not a broken run: it counts min(max page, 1) = 0.
                used_f = sb.tile((_P_MAX, 1), fp32)
                nisa.tensor_reduce(dst=used_f[0:rows], op=nl.maximum, data=row_f[0:rows, 0:_GROUP_BLOCKS], axis=1)
                nisa.scalar_tensor_tensor(
                    dst=broken_f[0:rows],
                    data=used_f[0:rows],
                    op0=nl.minimum,
                    operand0=1.0,
                    op1=nl.multiply,
                    operand1=broken_f[0:rows],
                )
            seq_broken = sb.tile((_P_MAX, tokens), nl.bfloat16)
            nisa.tensor_scalar(
                dst=seq_broken[0:rows],
                data=steps[0:rows, _GROUP_BLOCKS : _GROUP_BLOCKS + tokens],
                op0=nl.equal,
                operand0=row_f[0:rows, 4:5],
                op1=nl.multiply,
                operand1=broken_f[0:rows, 0:1],
            )
            nisa.nc_matmul(
                dst=broken_psum, stationary=ones[0:rows], moving=seq_broken[0:rows], accumulate=item_tile > 0
            )
    if runs:
        broken = broken_psum
        if pad_src is not None:
            # A padding sequence counts no broken item: a select, so the counts of a live sequence keep every bit.
            pad_f = sb.tile((scale_parts, tokens), fp32)
            nisa.tensor_copy(dst=pad_f, src=pad_i, engine=nisa.engine.vector)
            nisa.tensor_scalar(dst=pad_f, data=pad_f, op0=nl.less_equal, operand0=0.0, engine=nisa.engine.vector)
            pad_u8 = sb.tile((scale_parts, tokens), nl.uint8)
            nisa.tensor_copy(dst=pad_u8, src=pad_f, engine=nisa.engine.vector)
            zero_f = sb.tile((scale_parts, tokens), fp32)
            nisa.memset(dst=zero_f, value=0.0, engine=nisa.engine.vector)
            broken = sb.tile((scale_parts, tokens), fp32)
            nisa.tensor_copy(dst=broken, src=broken_psum, engine=nisa.engine.vector)
            nisa.tensor_copy_predicated(dst=broken, src=zero_f, predicate=pad_u8)
        # The bits of 1.0 (0x3F800000) for no broken item, else of a quiet NaN (0x7FC00000): exact in FP32.
        scale_f = sb.tile((scale_parts, tokens), fp32)
        nisa.tensor_scalar(
            dst=scale_f,
            data=broken,
            op0=nl.minimum,
            operand0=1.0,
            op1=nl.multiply,
            operand1=float(0x7FC00000 - 0x3F800000),
        )
        nisa.tensor_scalar(dst=run_scale.view(int32), data=scale_f, op0=nl.add, operand0=float(0x3F800000))
    sb.close_scope()
    return item_hbm


def mla_pack_step_unit(region, name: str, tokens: int, page_runs: int) -> dict:
    """Return the step store of one packed-loop unit (a pair, a group, or the one loop of a layer) for
    ``_mla_decode_body`` ``pack_step_items``: its name prefix and, with ``page_runs``, the ``[96, tokens]`` FP32 run-scale
    tile on the heap of ``region``, which lives for the rest of the kernel. The first layer that runs the unit adds its
    item rows (``_packed_key_loop`` ``step_items``)."""
    run_scale = region.heap((_HEADS, tokens), nl.float32, f"mla_step_{name}_run_scale") if page_runs else None
    return {"tag": f"mla_step_{name}", "run_scale": run_scale}


def mla_pack_plan(
    block_table,
    position,
    rank_info,
    tag,
    sbuf=None,
    slots=None,
    tail_slots=None,
    tail_round=False,
    plan_sbuf=None,
    with_ti=True,
    shared=None,
):
    """Build the item table of ``mla_decode_attention(packed_loop=True)`` once for the MLA layers of a step.

    Args:
        block_table, position, rank_info (nl.NkiTensor): the HBM inputs of ``mla_decode_attention``.
        tag (str): unique name prefix.
        sbuf (SbufTiles): the allocator; with a region the plan tiles go on its heap.
        slots (int): items of one iteration (``pack_slots``); None picks it from the table width.
        tail_slots (int | tuple): None, or the items of one iteration of the tail loop (``pack_tail_slots``), or two
            or more counts for as many tail loops, which the plan then carries.
        tail_round (bool): round the items up to whole iterations of the last tail loop before they split over the
            loops (``_pack_items``); bitwise the same.
        plan_sbuf (SbufTiles): None, or the allocator of the plan tiles (with a region, on its heap); the scratch
            tiles then take ``sbuf``. None gives the plan tiles ``sbuf`` too.
        with_ti (bool): the plan holds the TI offsets of the state, which depend on B only. False leaves them out,
            so the plan takes less SBUF, and the packed loop of each layer builds them (``_pack_ti``).
        shared (dict): None, or the tiles that the calls of a staged layer share (``_shared``): the plan tiles then
            take the same SBUF in each call.

    Returns:
        dict: ``pack_plan`` of ``mla_decode_attention``.
    """
    sb = AUTO_SBUF if sbuf is None else sbuf
    tokens = position.shape[0]
    position_i = sb.tile((tokens, 1), nl.int32)
    rank_i = sb.tile((tokens, 1), nl.int32)
    for dst, src, name in (
        (position_i, position[0:tokens, 0:1], "position"),
        (rank_i, rank_info[0:1, 2:3].broadcast(0, tokens), "rank"),
    ):
        nisa.dma_copy(
            dst=dst, src=src, dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync, name=f"{tag}_mla_plan_{name}"
        )
    rank_f = sb.tile((tokens, 1), nl.float32)
    nisa.tensor_copy(dst=rank_f, src=rank_i)
    return _pack_items(
        position_i,
        rank_f,
        block_table,
        tag,
        sb,
        slots,
        plan_sbuf,
        shared=shared,
        tail_slots=tail_slots,
        tail_round=tail_round,
        with_ti=with_ti,
    )


def pack_p2_runs(queries: int) -> list[tuple[int, int, int, int, int]]:
    """Return the runs of the (row, head) layout of ``pack_p2`` for ``queries`` Q rows: the Q x 96 (row, head) columns of
    a sequence, row-major (column 96 j + h is head h of row j), split into groups of 128, as (group, row, first head,
    first column in the group, count) for each run of one row inside one group.

    Example:
        >>> pack_p2_runs(4)[:3]
        [(0, 0, 0, 0, 96), (0, 1, 0, 96, 32), (1, 1, 32, 0, 64)]
    """
    kernel_assert(queries * _HEADS % _P_MAX == 0, f"pack_p2 needs Q x 96 to be a multiple of 128, got Q = {queries}")
    runs = []
    for flat in range(0, queries * _HEADS):
        group, column = divmod(flat, _P_MAX)
        row, head = divmod(flat, _HEADS)
        if runs and runs[-1][0] == group and runs[-1][1] == row:
            runs[-1][4] += 1
        else:
            runs.append([group, row, head, column, 1])
    return [tuple(run) for run in runs]


def pack_shared_zone_bytes(options: dict, queries: int, tokens: int) -> int:
    """Return the bytes per partition of the step zone of the hoisted packed loops (``mla_pass_plans``): the tiles that
    ``_packed_key_loop_steps`` shares across passes and layers (``shared`` on ``pack_sb``) for passes of ``tokens``
    sequences of ``queries`` rows, at 32-byte alignment: the TI offsets of the plan, the ``_KEY_SETS`` TI index tiles,
    the unit MX scales (``mx_keys`` / ``mx_qk``), and with ``setup_off_pool`` the batched weight-sum offsets of the stage
    rows and of the null row. ``options``: the packed-loop form (``tkg_mla_block`` ``_PACKED_PRIOR_OPTIONS`` with overrides).
    A tile past the zone fails the trace (``SbufZone``), so a change of these tiles must change this size too."""

    def aligned(nbytes):
        return -(-nbytes // SB_TILE_ALIGN) * SB_TILE_ALIGN

    subs = pack_p2_groups(queries) if options["p2"] else queries
    size = aligned((subs * tokens + 1) * _TI_COLS * 2) + _KEY_SETS * aligned(_TI_COLS * 2)
    if options["mx_keys"] or options["mx_qk"]:
        size += aligned(_GROUP_BLOCKS * _BLOCK)
    if options["setup_off_pool"] and options["sum_batch"] and queries > 1:
        size += aligned((tokens + 1) * 4) + aligned(4)
    return size


def pack_p2_groups(queries: int) -> int:
    """Return the sub-slots of one item with ``pack_p2``: the groups of 128 (row, head) columns of Q rows."""
    return queries * _HEADS // _P_MAX


def pack_p2_ref_pieces(queries: int) -> list[tuple[int, int, int, int, int]]:
    """Return the runs of ``pack_p2_runs`` as periodic pieces: the (row, head) layout repeats every 4 rows, which fill
    3 groups of 128 columns (4 x 96 = 3 x 128), so run (g, j, h, c, n) of Q rows is piece (g % 3, j % 4, h, c, n) of the
    first 4 rows in period j // 4 (row j, group g = g % 3 + 3 (j // 4)). The 6 pieces of ``pack_p2_runs(4)``, each
    repeated over the Q / 4 periods, are the runs of Q rows, in the same order.

    Example:
        >>> len(pack_p2_ref_pieces(8)), pack_p2_ref_pieces(8)[1]
        (6, (0, 1, 0, 96, 32))
    """
    kernel_assert(queries % 4 == 0, f"pack_p2 needs Q x 96 to be a multiple of 128, got Q = {queries}")
    pieces = pack_p2_runs(4)
    expanded = [(g + 3 * k, j + 4 * k, h, c, n) for k in range(queries // 4) for g, j, h, c, n in pieces]
    kernel_assert(expanded == pack_p2_runs(queries), "pack_p2: the runs of Q rows are the 4-row pieces in each period")
    return pieces


def _stage_query_rows_p2(
    stage_hbm,
    plan_ti,
    q_nope_all,
    q_rope_all,
    softmax,
    tokens,
    queries,
    tag,
    sb,
    dr_rows=None,
    stage_sb=None,
    sum_batch=False,
    mx_qk=False,
    refs_act=False,
    sums_f=None,
    ti_engine=None,
    hbm_store=True,
    null_reference=False,
    latent=True,
):
    """``pack_p2``: store the slot rows of the multi-query packed loop to ``stage_hbm`` ``[B + 1, 128, G x 680]``: row b
    holds G = Q x 96 / 128 stage rows of sequence b, stage row g the 128 (row, head) columns 128 g .. 128 g + 127 of its Q
    rows (``pack_p2_runs``): the latent queries (4 x 128), the rope queries (rows 0 .. 63) with 2^100 on query bias row
    64 + j of each column of row j, the softmax reference and cap of each column on its partition, and the TI offsets of
    state column ``g B + b``. Row B (the null items) holds those of sequence 0 with the TI offsets of the null column
    ``B G``. The queries, rope queries and TI offsets stay on their partitions (engine copies); the references move to
    the partition of their column (SBUF DMAs). The values are those of ``_stage_query_rows``, in other columns. With
    ``dr_rows`` (``pack_dr_keys``: the double_row stationaries ``[128, Q B, 2, 2, 96]`` FP8 of the rows,
    ``_dr_gather_tile_queries``) the first 512 bytes of stage row g hold the ``[128, 2, 2, 128]`` stationaries of its
    columns (``[byte, block, column]``, ``_dr_stationary`` with ``cols`` 128) in place of the BF16 latent query tiles: an
    exact byte copy of each run. Return the store DMA: the loop buffers take the bytes of the stage tile after it, so their
    first writes wait for its completion (``_packed_key_loop`` ``p2``). ``stage_sb``: the stage tile kept by the caller
    (``stage_rows_sbuf``). ``sum_batch``: the sum-offset column of stage row G - 1 (TI column ``_TI_COLS - 1``) holds the
    TI offsets of the item's batched weight-sum add, ``(g B + b) 513 + 512`` for group g < G at partition g of each group
    of 16, as ``_stage_query_rows`` does for the Q rows (``_sum_batch_offsets`` on the 128 columns). With ``mx_qk``
    (``pack_mx_qk``) the first 512 bytes of stage row g hold the MX query words of its 128 columns in place of the BF16
    latent query tiles: word ``(c, column)`` (FP32, ``float8_e4m3fn_x4``) holds the e4m3 latent features ``c``, ``128 + c``,
    ``256 + c`` and ``384 + c`` of the column in bytes 0 to 3 (byte t from query tile t of partition c, the grouping of the
    ``word_keys`` key words). "unit": rounded to nearest even and saturated at 448 (unit scale), as the GPU decode rounds
    its query; one VectorE op a run, on the partitions of the tiles. "block32": MXFP8, one E8M0 scale for each block of
    8 partitions x 4 bytes of a column (features ``8m + i + 128 t``, i < 8, t < 4), from ``nisa.quantize_mx`` on an
    exact interleaved BF16 copy of the run (VectorE), one a group and sequence; the 128 scale bytes of the columns sit at
    BF16 column ``_P2_STAGE_MX_SCALE`` of the stage row (the quadrant rows of the MX scale layout). The null row is then
    copied through a uint16 view, so no engine reads the FP8 bytes as BF16. With ``_packed_key_loop`` ``mx_natural``
    ``q_nope_all`` holds the latent query tiles of the natural MX grouping (element ``(c, r, t, h)`` is feature
    ``4c + t``): the same ops then give word ``(c, column)`` features ``4c .. 4c + 3`` in bytes 0 to 3 (the grouping of the
    key words of an FP8 key row as it is), and a "block32" scale block the 32 features ``32m .. 32m + 31``. ``refs_act``
    (``_packed_key_loop`` ``p2_entry_act``): the references move to the partitions of their columns first, by 2 x 6 DMAs
    on the ScalarE queue straight into their stage words (``pack_p2_ref_pieces``, each over the Q / 4 periods), so the
    copies of the runs do not alternate with SBUF DMAs on the SP queue; and the zero fill covers only the bytes that
    no copy writes (the rope columns of partitions 64 .. 127 and the 3 pad columns of the B sequence rows; with
    ``dr_rows`` every byte but the references), about a fifth of the tile. The same bytes in the stage rows.
    ``hbm_store`` False (``_packed_key_loop`` ``skip_stage_store``): the stage rows stay in ``stage_sb`` only, with no
    store to ``stage_hbm``; return None. ``latent`` False (``mx_qk`` "block32" in SBUF only: a reader that takes the MX
    query words and scales from elsewhere, the FP8 P@V setup with ``q_mx_gather``): no latent query copies and no
    ``nisa.quantize_mx``; the rope rows, the references and the rest are as with ``latent``, and the words and scales of
    the stage keep what the tile held. ``q_nope_all`` may then be None."""
    bf16 = nl.bfloat16
    fp32 = nl.float32
    u16 = nl.uint16
    vector = nisa.engine.vector
    kernel_assert(
        mx_qk in _MX_QK_FORMS and not (mx_qk and dr_rows is not None),
        f"pack_mx_qk must be one of {_MX_QK_FORMS} and does not take the double_row stationaries, got {mx_qk!r}",
    )
    groups = pack_p2_groups(queries)
    width = _P2_STAGE_WIDTH
    q_width = _KV_LORA_TILES * _P_MAX
    bias_end = _BIAS_ROW + queries
    stage_sb = sb.tile((_P_MAX, tokens + 1, groups * width), bf16) if stage_sb is None else stage_sb
    stage_f = stage_sb.view(fp32)
    if _P2_STAGE_FILL_TEST is not None:
        nisa.memset(dst=stage_sb, value=_P2_STAGE_FILL_TEST, engine=vector)
    kernel_assert(
        latent or (mx_qk == "block32" and not hbm_store and not refs_act and dr_rows is None),
        "_stage_query_rows_p2 latent False is the SBUF stage of mx_qk block32 without its words: no hbm_store, "
        "refs_act or dr_rows",
    )
    if refs_act:
        # The zero fill first: it has no input and skips the reference words, so it can run before the query
        # rows exist, off the entry path of the loop. The reference DMAs then write the words that it skips. The
        # bytes do not overlap, so the stage rows hold the same bytes in either order.
        stage_g = stage_sb.reshape((_P_MAX, tokens + 1, groups, width))
        if dr_rows is None and not mx_qk:
            # Only the bytes that no copy writes: the rope columns of partitions 64 .. 127 (the bias rows 64 .. 64 + Q - 1
            # add their bias to this zero) and the 3 pad columns, in the B sequence rows; the latent and rope copies,
            # the references, the TI copies and the null row (a copy of row 0) write every other byte.
            zero_parts = (
                (slice(_BIAS_ROW, _P_MAX), slice(q_width, q_width + _P_MAX)),
                (slice(0, _P_MAX), slice(_P2_STAGE_TI + _TI_COLS, width)),
            )
            for parts, cols in zero_parts:
                nisa.memset(dst=stage_g[parts, 0:tokens, 0:groups, cols], value=0.0, engine=vector)
        else:
            # The double_row stationaries and the MX query words fill only part of the latent columns: every byte but
            # the references, in the B sequence rows (the null row is a copy of row 0, which writes every byte of it).
            ref_col = 2 * _P2_STAGE_REF_WORD
            for first, end in ((0, ref_col), (ref_col + 4, width)):
                nisa.memset(dst=stage_g[0:_P_MAX, 0:tokens, 0:groups, first:end], value=0.0, engine=vector)
        # The reference and cap of column c of group g = g0 + 3 k of sequence b land on partition c, in FP32 words
        # 320 and 321 of the stage row, one DMA a piece of each period (the words that the zero fill skips).
        periods = queries // 4
        refs_dst = stage_f.reshape((_P_MAX, tokens + 1, periods, 3, width // 2))
        for offset, source in enumerate((softmax.negative_reference, softmax.score_cap)):
            source_rows = source[0:_HEADS, 0 : queries * tokens].reshape((_HEADS, periods, 4, tokens))
            for piece, (group, row, head, column, count) in enumerate(pack_p2_ref_pieces(queries)):
                nisa.dma_copy(
                    dst=refs_dst[column : column + count, 0:tokens, 0:periods, group, _P2_STAGE_REF_WORD + offset],
                    src=source_rows[head : head + count, 0:periods, row, 0:tokens].permute((0, 2, 1)),
                    dge_mode=nisa.dge_mode.hwdge,
                    engine=nisa.engine.scalar,
                    name=f"{tag}_mla_pack_p2_refs_{piece}_{offset}",
                )
    else:
        # The B sequence rows: the null row is a copy of row 0, which writes every byte of it.
        nisa.memset(dst=stage_sb[0:_P_MAX, 0:tokens, 0 : groups * width], value=0.0, engine=vector)
    stage_u = stage_sb.view(u16)
    # bias[64 + i, j] = 2^100 for i == j, else 0: the query bias rows of the columns of row j.
    bias_i = sb.tile((_P_MAX, queries), nl.int32)
    pool_after(
        nisa.iota(
            dst=bias_i[_BIAS_ROW:bias_end, 0:queries],
            pattern=[[-1, queries]],
            offset=0,
            channel_multiplier=1,
        ),
        "p2_stage_bias",
    )
    bias = sb.tile((_P_MAX, queries), fp32)
    nisa.tensor_scalar(
        dst=bias[_BIAS_ROW:bias_end, 0:queries],
        data=bias_i[_BIAS_ROW:bias_end, 0:queries],
        op0=nl.equal,
        operand0=0.0,
        op1=nl.multiply,
        operand1=_MASK_COLUMN_SCALE,
        engine=vector,
    )
    # ``dr_rows``: the stationaries of a run as bytes, the (byte, block) pairs of a column merged into 4 rows of 128.
    dr_parts = 2 * _DR_BLOCKS
    stage_b = None if dr_rows is None else stage_sb.view(nl.uint8)
    dr_u = None if dr_rows is None else dr_rows.view(nl.uint8).reshape((_P_MAX, dr_rows.shape[1], dr_parts, _HEADS))
    # ``mx_qk``: the e4m3 bytes of the MX query words, byte t of word (c, column) from query tile t; "block32" quantizes
    # an interleaved BF16 copy of each group (a ring of 2), whose runs fill its 128 columns in order.
    stage_8 = stage_sb.view(nl.float8_e4m3fn) if mx_qk else None
    q_il = None
    if mx_qk == "block32" and latent:
        q_il = [sb.tile((_P_MAX, tokens, _P_MAX, _MX_PACK), bf16) for _ in range(2)]
    for group, row, head, column, count in pack_p2_runs(queries):
        base = group * width
        rows = slice(row * tokens, (row + 1) * tokens)
        if not latent:
            # The MX words and scales come from elsewhere (``q_mx_gather``).
            pass
        elif dr_u is not None:
            nisa.tensor_copy(
                dst=stage_b[0:_P_MAX, 0:tokens, 2 * base : 2 * base + dr_parts * _P_MAX].reshape(
                    (_P_MAX, tokens, dr_parts, _P_MAX)
                )[0:_P_MAX, 0:tokens, 0:dr_parts, column : column + count],
                src=dr_u[0:_P_MAX, rows, 0:dr_parts, head : head + count],
                engine=vector,
            )
        elif mx_qk == "block32":
            interleaved = q_il[group % 2]
            nisa.tensor_copy(
                dst=interleaved[0:_P_MAX, 0:tokens, column : column + count, 0:_KV_LORA_TILES],
                src=q_nope_all[0:_P_MAX, rows, 0:_KV_LORA_TILES, head : head + count].permute((0, 1, 3, 2)),
                engine=vector,
            )
            if column + count == _P_MAX:
                stage_b8 = stage_sb.view(nl.uint8)
                for token in range(tokens):
                    nisa.quantize_mx(
                        dst=stage_f[0:_P_MAX, token : token + 1, base // 2 : base // 2 + _P_MAX].view(
                            nl.float8_e4m3fn_x4
                        ),
                        src=interleaved[0:_P_MAX, token : token + 1, 0:_P_MAX, 0:_MX_PACK],
                        dst_scale=stage_b8[
                            0:_P_MAX,
                            token : token + 1,
                            2 * (base + _P2_STAGE_MX_SCALE) : 2 * (base + _P2_STAGE_MX_SCALE) + _P_MAX,
                        ],
                    )
        elif mx_qk:
            nisa.tensor_scalar(
                dst=stage_8[0:_P_MAX, 0:tokens, 2 * base : 2 * base + _MX_PACK * _P_MAX].reshape(
                    (_P_MAX, tokens, _P_MAX, _MX_PACK)
                )[0:_P_MAX, 0:tokens, column : column + count, 0:_KV_LORA_TILES],
                data=q_nope_all[0:_P_MAX, rows, 0:_KV_LORA_TILES, head : head + count].permute((0, 1, 3, 2)),
                op0=nl.minimum,
                operand0=_E4M3_MAX,
                op1=nl.maximum,
                operand1=-_E4M3_MAX,
                engine=vector,
            )
        else:
            nisa.tensor_copy(
                dst=stage_sb[0:_P_MAX, 0:tokens, base : base + q_width].reshape(
                    (_P_MAX, tokens, _KV_LORA_TILES, _P_MAX)
                )[0:_P_MAX, 0:tokens, 0:_KV_LORA_TILES, column : column + count],
                src=q_nope_all[0:_P_MAX, rows, 0:_KV_LORA_TILES, head : head + count],
                engine=vector,
            )
        rope_cols = slice(base + q_width + column, base + q_width + column + count)
        nisa.tensor_copy(
            dst=stage_sb[0:_ROPE, 0:tokens, rope_cols],
            src=q_rope_all[0:_ROPE, rows, head : head + count],
            engine=vector,
        )
        nisa.tensor_scalar(
            dst=stage_sb[_BIAS_ROW:bias_end, 0:tokens, rope_cols],
            data=stage_sb[_BIAS_ROW:bias_end, 0:tokens, rope_cols],
            op0=nl.add,
            operand0=bias[_BIAS_ROW:bias_end, row : row + 1],
            engine=vector,
        )
        if refs_act:
            continue
        for offset, source in enumerate((softmax.negative_reference, softmax.score_cap)):
            word = base // 2 + _P2_STAGE_REF_WORD + offset
            nisa.dma_copy(
                dst=stage_f[column : column + count, 0:tokens, word],
                src=source[head : head + count, rows],
                dge_mode=nisa.dge_mode.hwdge,
                engine=nisa.engine.sync,
                name=f"{tag}_mla_pack_p2_ref_{group}_{column}_{offset}",
            )
    # ``ti_engine`` (``_packed_key_loop`` ``setup_off_pool``: VectorE) is the engine of the TI offset copies (GpSimd).
    ti_engine = nisa.engine.gpsimd if ti_engine is None else ti_engine
    for group in range(groups):
        ti_cols = slice(group * width + _P2_STAGE_TI, group * width + _P2_STAGE_TI + _TI_COLS)
        nisa.tensor_copy(
            dst=stage_u[0:_P_MAX, 0:tokens, ti_cols],
            src=plan_ti[0:_P_MAX, group * tokens : (group + 1) * tokens, 0:_TI_COLS],
            engine=ti_engine,
        )
    # The null row: sequence 0's stage rows (after their references land), then the TI offsets of the null column.
    # ``mx_qk``: through the uint16 view (an exact copy of the FP8 query bytes and scales).
    null_view = stage_u if mx_qk else stage_sb
    nisa.tensor_copy(
        dst=null_view[0:_P_MAX, tokens, 0 : groups * width],
        src=null_view[0:_P_MAX, 0, 0 : groups * width],
        engine=vector,
    )
    for group in range(groups):
        ti_cols = slice(group * width + _P2_STAGE_TI, group * width + _P2_STAGE_TI + _TI_COLS)
        nisa.tensor_copy(
            dst=stage_u[0:_P_MAX, tokens, ti_cols],
            src=plan_ti[0:_P_MAX, groups * tokens, 0:_TI_COLS],
            engine=ti_engine,
        )
    if sum_batch:
        # A given ``sums_f`` (``_packed_key_loop`` ``setup_off_pool``) is the offset tile that the passes share.
        sums_f = _sum_batch_offsets(groups, tokens, sb, parts=_P_MAX) if sums_f is None else sums_f
        sum_col = (groups - 1) * width + _P2_STAGE_TI + _TI_COLS - 1
        nisa.tensor_copy(
            dst=stage_u[0:_P_MAX, 0 : tokens + 1, sum_col], src=sums_f[0:_P_MAX, 0 : tokens + 1], engine=vector
        )
    if null_reference:
        # ``kv_pitch576``: every weight of a null item is 0 without its mask (``_NULL_ITEM_REFERENCE``), copied from an
        # FP32 tile (a memset through the FP32 view of the BF16 stage tile would take its BF16 element type).
        null_ref = sb.tile((_P_MAX, groups), fp32)
        nisa.memset(dst=null_ref, value=-_NULL_ITEM_REFERENCE, engine=vector)
        nisa.tensor_copy(
            dst=stage_f[0:_P_MAX, tokens, _P2_STAGE_REF_WORD : groups * width // 2 : width // 2],
            src=null_ref,
            engine=vector,
        )
    if not hbm_store:
        # ``skip_stage_store``: every slot row of the loop reads the kept stage tile, so no DMA reads the HBM store.
        return None
    store = nisa.dma_copy(
        dst=stage_hbm.permute((1, 0, 2)),
        src=stage_sb,
        dge_mode=nisa.dge_mode.hwdge,
        engine=nisa.engine.sync,
        name=f"{tag}_mla_pack_rows_store",
    )
    return _instruction(store)


def _stage_pair_rows(
    stage_hbm, plan_ti, q_nope_all, q_rope_all, softmax, tokens, half, mask_column, tag, sb, dr_q_all=None
):
    """Store the slot rows of the ``tokens`` sequences of half ``half`` of a packed loop pair (``_packed_key_loop``
    ``pair``) to rows ``half x tokens`` on of ``stage_hbm`` ``[2 tokens + 1, 128, W]``, with the TI offsets of their state
    columns; half 0 also stores row ``2 tokens`` (null items), a copy of its sequence 0 with the TI offsets of the null
    column, as the loop of one group does with its row B. With ``dr_q_all`` (the query gather of the half,
    ``pack_dr_keys``) the rows hold the e4m3 double_row stationaries (``_dr_query_rows``) in place of the BF16 latent
    queries."""
    bf16 = nl.bfloat16
    fp32 = nl.float32
    u16 = nl.uint16
    first = half * tokens
    null_row = 2 * tokens
    rows = tokens + (0 if half else 1)
    q_width = _KV_LORA_TILES * _HEADS
    sb.open_scope("pack")
    q_rows = q_nope_all.reshape((_P_MAX, tokens, q_width))
    stage_sb = sb.tile((_P_MAX, rows, _STAGE_WIDTH), bf16)
    nisa.memset(dst=stage_sb, value=0.0)
    stage_f = stage_sb.view(fp32)
    stage_u = stage_sb.view(u16)
    dr_rows = None if dr_q_all is None else _dr_query_rows(dr_q_all, tokens, sb)
    for row, count in ((0, tokens), (tokens, 1))[: rows - tokens + 1]:
        if dr_rows is None:
            nisa.tensor_copy(
                dst=stage_sb[0:_P_MAX, row : row + count, 0:q_width], src=q_rows[0:_P_MAX, 0:count, 0:q_width]
            )
        else:
            _dr_stage_queries(stage_u, row, count, dr_rows)
        nisa.tensor_copy(
            dst=stage_sb[0 : _BIAS_ROW + 1, row : row + count, q_width : q_width + _HEADS],
            src=q_rope_all[0 : _BIAS_ROW + 1, 0:count, 0:_HEADS],
        )
        for column, source in enumerate((softmax.negative_reference, softmax.score_cap)):
            word = _STAGE_REF_WORD + column
            nisa.tensor_copy(dst=stage_f[0:_HEADS, row : row + count, word : word + 1], src=source[0:_HEADS, 0:count])
    ti_cols = slice(_STAGE_TI, _STAGE_TI + _TI_COLS)
    nisa.tensor_copy(
        dst=stage_u[0:_HEADS, 0:tokens, ti_cols], src=plan_ti[0:_HEADS, first : first + tokens, 0:_TI_COLS]
    )
    if not half:
        nisa.tensor_copy(dst=stage_u[0:_HEADS, tokens, ti_cols], src=plan_ti[0:_HEADS, null_row, 0:_TI_COLS])
    if mask_column:
        nisa.memset(
            dst=stage_sb[_BIAS_ROW : _BIAS_ROW + 1, 0:rows, q_width : q_width + _HEADS], value=_MASK_COLUMN_SCALE
        )
    stage_rows = stage_hbm.permute((1, 0, 2))
    hwdge = nisa.dge_mode.hwdge
    sync = nisa.engine.sync
    nisa.dma_copy(
        dst=stage_rows[0:_P_MAX, first : first + tokens, 0:_STAGE_WIDTH],
        src=stage_sb[0:_P_MAX, 0:tokens, 0:_STAGE_WIDTH],
        dge_mode=hwdge,
        engine=sync,
        name=f"{tag}_mla_pack_rows_store",
    )
    if not half:
        nisa.dma_copy(
            dst=stage_rows[0:_P_MAX, null_row, 0:_STAGE_WIDTH],
            src=stage_sb[0:_P_MAX, tokens, 0:_STAGE_WIDTH],
            dge_mode=hwdge,
            engine=sync,
            name=f"{tag}_mla_pack_null_row_store",
        )
    sb.close_scope()


def _sum_batch_offsets(queries: int, tokens: int, sb, null_only: bool = False, parts: int = _HEADS, out_sb=None):
    """Return the batched weight-sum TI offsets of the multi-query packed loop (``_packed_key_loop`` ``sum_batch``), FP32
    ``[128, B + 1]`` (``null_only``: ``[128, 1]``, the null row), for the 16 elements of the batched add (``_TI_GROUP``;
    element e of a TI index is partition e of each group of 16): on partition p, with e = p % 16, column b < B holds
    ``(e B + b) 513 + 512`` (row e's sum) for e < Q and ``Q B 513 + 512 - e`` for e >= Q, and column B (the null row)
    ``Q B 513 + 512 - e`` for every e. The null-column words are distinct (its sum and the last numerator words), so no
    batched add repeats an offset; the elements past Q and the null row add zero sums. Exact in FP32. ``parts``: the
    partitions of the sums (96 heads; with ``pack_p2`` the 128 (row, head) columns, and ``queries`` the G groups of an
    item, whose state column ``g B + b`` takes the same offsets). ``out_sb`` (default ``sb``) allocates the result tile,
    ``sb`` the scratch tiles."""
    int32 = nl.int32
    fp32 = nl.float32
    vector = nisa.engine.vector
    columns = queries * tokens
    part_i = sb.tile((_P_MAX, 1), int32)
    nisa.iota(dst=part_i[0:parts, 0:1], pattern=[[0, 1]], offset=0, channel_multiplier=1)
    nisa.tensor_scalar(
        dst=part_i[0:parts, 0:1], data=part_i[0:parts, 0:1], op0=nl.bitwise_and, operand0=_TI_GROUP - 1, engine=vector
    )
    elem_f = sb.tile((_P_MAX, 1), fp32)
    nisa.tensor_copy(dst=elem_f[0:parts, 0:1], src=part_i[0:parts, 0:1], engine=vector)
    width = 1 if null_only else tokens + 1
    out = (sb if out_sb is None else out_sb).tile((_P_MAX, width), fp32)
    # The null words: Q B 513 + 512 - e (the null row, and the elements past Q).
    null_f = out[0:parts, width - 1 : width]
    nisa.tensor_scalar(
        dst=null_f,
        data=elem_f[0:parts, 0:1],
        op0=nl.multiply,
        operand0=-1.0,
        op1=nl.add,
        operand1=float(columns * _STATE_WIDTH + _KV_LORA),
        engine=vector,
    )
    if not null_only:
        # 1.0 for the elements of the Q rows, else 0.0.
        row_mask = sb.tile((_P_MAX, 1), fp32)
        nisa.tensor_scalar(
            dst=row_mask[0:parts, 0:1],
            data=elem_f[0:parts, 0:1],
            op0=nl.less_equal,
            operand0=float(queries - 1),
            engine=vector,
        )
        seq_i = sb.tile((_P_MAX, tokens), int32)
        nisa.iota(dst=seq_i[0:parts, 0:tokens], pattern=[[_STATE_WIDTH, tokens]], offset=_KV_LORA, channel_multiplier=0)
        row_off = sb.tile((_P_MAX, 1), fp32)
        nisa.tensor_scalar(
            dst=row_off[0:parts, 0:1],
            data=elem_f[0:parts, 0:1],
            op0=nl.multiply,
            operand0=float(tokens * _STATE_WIDTH),
            engine=vector,
        )
        rows_f = out[0:parts, 0:tokens]
        nisa.tensor_copy(dst=rows_f, src=seq_i[0:parts, 0:tokens], engine=vector)
        # (e B + b) 513 + 512 - null, masked to the Q rows, plus null: the row offset, or the null word past Q.
        nisa.tensor_scalar(
            dst=rows_f,
            data=rows_f,
            op0=nl.add,
            operand0=row_off[0:parts, 0:1],
            op1=nl.subtract,
            operand1=null_f,
            engine=vector,
        )
        nisa.tensor_scalar(
            dst=rows_f,
            data=rows_f,
            op0=nl.multiply,
            operand0=row_mask[0:parts, 0:1],
            op1=nl.add,
            operand1=null_f,
            engine=vector,
        )
    return out


def _shared_sum_offsets(queries: int, tokens: int, sb, pack_sb, null_only: bool = False, parts: int = _HEADS):
    """``_sum_batch_offsets`` for a tile that the passes and layers share (``shared``): the result on ``pack_sb``. When
    ``pack_sb`` is another allocator than ``sb`` (a zone of the step), the scratch tiles take a frame of ``sb``."""
    if pack_sb is sb:
        return _sum_batch_offsets(queries, tokens, sb, null_only=null_only, parts=parts)
    with sb.scope("sum_offsets"):
        return _sum_batch_offsets(queries, tokens, sb, null_only=null_only, parts=parts, out_sb=pack_sb)


def _stage_query_rows(
    stage_hbm,
    plan_ti,
    q_nope_all,
    q_rope_all,
    softmax,
    tokens,
    queries,
    tag,
    sb,
    dr_rows=None,
    stage_sb=None,
    mx_rows=None,
    sum_batch=False,
    null_reference=False,
):
    """Store the slot rows of the multi-query packed loop (``_packed_key_loop`` ``queries``) to ``stage_hbm``
    ``[B + 1, 128, Q x 520]``: row b holds the stage rows of the Q query rows of sequence b one after the other, stage row
    j the slot row of the decode for state column ``j B + b`` (the query, the rope query, the softmax reference and cap,
    and the TI offsets of the column), with 2^100 on query bias row 64 + j and 0 on the other bias rows. Row B (the null
    items) holds those of sequence 0 with the TI offsets of the null column ``Q B``. Each partition stores runs of
    ``Q x 1040`` bytes. With ``dr_rows`` (``pack_dr_keys``: the double_row stationaries ``[128, Q B, 2, 2, 96]`` FP8 of
    the rows) the first words of stage row j hold those of row j (an exact 2-byte copy), in place of the BF16 latent
    query tiles. With ``mx_rows`` (``pack_mx_keys`` "single": the MX query words ``[128, Q B, 96]`` FP32 of the rows) the
    first 96 words of stage row j hold those of row j, also an exact 2-byte copy (an FP32 copy of the packed FP8 bytes
    could flush an FP32 subnormal word). ``stage_sb`` (``_packed_key_loop`` ``stage_rows_sbuf``): the ``[128, B + 1, Q x
    520]`` stage tile, which the caller keeps for the prologue's slot rows (default a tile of ``sb``). With ``sum_batch``
    (``_packed_key_loop`` ``sum_batch``) the sum-offset column of stage row Q - 1 (TI column ``_TI_COLS - 1``, which no
    add of that row reads then) holds the TI offsets of the item's batched weight-sum add, ``(e B + b) 513 + 512`` for row
    e < Q at partition e of each group of 16 (``_TI_GROUP``), else and on row B distinct elements of the null column
    ``Q B`` (``_sum_batch_offsets``)."""
    bf16 = nl.bfloat16
    fp32 = nl.float32
    u16 = nl.uint16
    vector = nisa.engine.vector
    columns = queries * tokens
    width = _STAGE_WIDTH
    q_width = _KV_LORA_TILES * _HEADS
    bias_end = _BIAS_ROW + queries
    q_cols = q_nope_all.reshape((_P_MAX, columns, q_width))
    stage_sb = sb.tile((_P_MAX, tokens + 1, queries * width), bf16) if stage_sb is None else stage_sb
    nisa.memset(dst=stage_sb, value=0.0, engine=vector)
    stage_f = stage_sb.view(fp32)
    stage_u = stage_sb.view(u16)
    # bias[64 + i, j] = 2^100 for i == j, else 0: the query bias rows of the stage row of row j.
    bias_i = sb.tile((_P_MAX, queries), nl.int32)
    pool_after(
        nisa.iota(dst=bias_i[_BIAS_ROW:bias_end, 0:queries], pattern=[[-1, queries]], offset=0, channel_multiplier=1),
        "p2_stage_bias_b",
    )
    bias = sb.tile((_P_MAX, queries), fp32)
    nisa.tensor_scalar(
        dst=bias[_BIAS_ROW:bias_end, 0:queries],
        data=bias_i[_BIAS_ROW:bias_end, 0:queries],
        op0=nl.equal,
        operand0=0.0,
        op1=nl.multiply,
        operand1=_MASK_COLUMN_SCALE,
        engine=vector,
    )
    for row in range(queries):
        base = row * width
        rope_cols = slice(base + q_width, base + q_width + _HEADS)
        for first, count in ((0, tokens), (tokens, 1)):
            column = row * tokens
            if dr_rows is not None:
                dr_words = dr_rows.view(u16).reshape((_P_MAX, columns, _DR_QUERY_WORDS))
                nisa.tensor_copy(
                    dst=stage_u[0:_P_MAX, first : first + count, base : base + _DR_QUERY_WORDS],
                    src=dr_words[0:_P_MAX, column : column + count, 0:_DR_QUERY_WORDS],
                    engine=vector,
                )
            elif mx_rows is not None:
                mx_words = mx_rows.view(u16)
                nisa.tensor_copy(
                    dst=stage_u[0:_P_MAX, first : first + count, base : base + 2 * _HEADS],
                    src=mx_words[0:_P_MAX, column : column + count, 0 : 2 * _HEADS],
                    engine=vector,
                )
            else:
                nisa.tensor_copy(
                    dst=stage_sb[0:_P_MAX, first : first + count, base : base + q_width],
                    src=q_cols[0:_P_MAX, column : column + count, 0:q_width],
                    engine=vector,
                )
            nisa.tensor_copy(
                dst=stage_sb[0:_ROPE, first : first + count, rope_cols],
                src=q_rope_all[0:_ROPE, column : column + count, 0:_HEADS],
                engine=vector,
            )
            for offset, source in enumerate((softmax.negative_reference, softmax.score_cap)):
                word = base // 2 + _STAGE_REF_WORD + offset
                nisa.tensor_copy(
                    dst=stage_f[0:_HEADS, first : first + count, word : word + 1],
                    src=source[0:_HEADS, column : column + count],
                    engine=vector,
                )
        nisa.tensor_scalar(
            dst=stage_sb[_BIAS_ROW:bias_end, 0 : tokens + 1, rope_cols],
            data=stage_sb[_BIAS_ROW:bias_end, 0 : tokens + 1, rope_cols],
            op0=nl.add,
            operand0=bias[_BIAS_ROW:bias_end, row : row + 1],
            engine=vector,
        )
        ti_cols = slice(base + _STAGE_TI, base + _STAGE_TI + _TI_COLS)
        nisa.tensor_copy(
            dst=stage_u[0:_HEADS, 0:tokens, ti_cols],
            src=plan_ti[0:_HEADS, row * tokens : (row + 1) * tokens, 0:_TI_COLS],
            engine=nisa.engine.gpsimd,
        )
        nisa.tensor_copy(
            dst=stage_u[0:_HEADS, tokens, ti_cols],
            src=plan_ti[0:_HEADS, columns, 0:_TI_COLS],
            engine=nisa.engine.gpsimd,
        )
    if sum_batch:
        sums_f = _sum_batch_offsets(queries, tokens, sb)
        sum_col = (queries - 1) * width + _STAGE_TI + _TI_COLS - 1
        nisa.tensor_copy(
            dst=stage_u[0:_HEADS, 0 : tokens + 1, sum_col], src=sums_f[0:_HEADS, 0 : tokens + 1], engine=vector
        )
    if null_reference:
        # ``kv_pitch576``: every weight of a null item is 0 without its mask (``_NULL_ITEM_REFERENCE``), copied from an
        # FP32 tile (a memset through the FP32 view of the BF16 stage tile would take its BF16 element type).
        null_ref = sb.tile((_HEADS, queries), fp32)
        nisa.memset(dst=null_ref, value=-_NULL_ITEM_REFERENCE, engine=vector)
        nisa.tensor_copy(
            dst=stage_f[0:_HEADS, tokens, _STAGE_REF_WORD : queries * width // 2 : width // 2],
            src=null_ref,
            engine=vector,
        )
    nisa.dma_copy(
        dst=stage_hbm.permute((1, 0, 2)),
        src=stage_sb,
        dge_mode=nisa.dge_mode.hwdge,
        engine=nisa.engine.sync,
        name=f"{tag}_mla_pack_rows_store",
    )


def _packed_key_loop_steps(
    kv_pool,
    block_table,
    token_meta,
    token_f,
    q_nope_all,
    q_rope_all,
    softmax,
    slot_offset,
    state_flat,
    tokens,
    local_blocks,
    tag,
    sb,
    pack_plan=None,
    slots=None,
    tail_slots=None,
    tail_round=False,
    clamp_pages=True,
    pack_sb=None,
    shared=None,
    pack_hbm=None,
    pair_pages=False,
    prefetch=1,
    row_dma_act=False,
    page_runs=0,
    null_copy_vector=False,
    null_runs=False,
    key_transpose="matmul",
    latent_act=2,
    mask_column=False,
    bg_rope=False,
    bg_interleave=False,
    carry=False,
    carry_main_only=False,
    sum_after_copy=False,
    pair=None,
    setup_ahead=False,
    after_setup=None,
    mx_keys=None,
    mx_q_all=None,
    mx_q_rows=None,
    rope_ahead=False,
    queries=1,
    after_writes=(),
    dr_keys=False,
    dr_q_rows=None,
    item_rope_ahead=False,
    p2=False,
    row_dma_pool=0,
    lookahead=1,
    psum_per_block=False,
    step_items=None,
    stage_rows_sbuf=False,
    cap_first=False,
    clamp_pin=False,
    deep_pipe=False,
    sum_batch=False,
    numerator_ring=0,
    row_depth=None,
    word_keys=False,
    slot_rows_sbuf=False,
    keys_ahead=False,
    mx_qk=False,
    p2_entry_act=False,
    setup_off_pool=False,
    skip_stage_store=False,
    mx_natural=False,
    act_spread=False,
    capless=False,
    kv_pitch576=False,
    kv_gather=False,
    kv_rem4=False,
    kv_group=0,
    mask_pool=False,
):
    """Add the key groups of all sequences to ``state_flat`` ``[96, (B + 1) x 513]`` in one loop.

    Item i is key group g of sequence b, in sequence-major order: the groups of sequence 0, then those of
    sequence 1, and so on. Each iteration attends ``slots`` items, so the replica rounds its item count
    once, not each sequence, and runs one loop, not one loop per sequence. Slot j of an iteration loads the
    slot row of its sequence (query, softmax reference and Tensor Indirection (TI) offsets) with one DMA, and
    adds its group to the state column of the sequence through TI on VectorE. The state starts with the
    current key, as in the loop path, and each sequence adds its groups in the same order, so the state is
    bitwise the same. All updates run on VectorE in program order, so two slots of one sequence need no other
    ordering. The items after the last real one (null items) read the table row and pages of item 0 with every key
    masked, and add to column B. A dynamic DMA costs about as much as its bytes, so a slot has one row DMA and its 4
    page DMAs. With ``tail_slots`` the loop runs only the full iterations, and a second loop of ``tail_slots`` items
    per iteration runs the rest: the same items in the same order, so the state is bitwise the same. With a pair
    ``(t1, t2)`` the second loop runs only full iterations of ``t1`` items, and a third loop of ``t2`` items the rest.
    The slot rows and pages of the first ``prefetch`` slots of an iteration load during the iteration before; the
    DMAs and the data are the same, so the state is bitwise the same. ``row_dma_act`` moves the slot row DMA of each
    slot from the SP queue to the ScalarE queue: the same DMA, so the state is bitwise the same. ``row_dma_pool`` k
    (2 to 4) moves that of slot j with j % k == k - 1 to the GpSimd software DGE queue (j counts from the first slot of
    its iteration, in every loop, the prologue and the prefetched slots): the same DMA, so the state is bitwise the
    same. With ``page_runs`` R (2 or 4) a slot loads its 4 pages with 4 / R DMAs, each from the first page of its run
    of R, which read the bytes of the 4 page DMAs when the pages of the group are runs of R, so the state is bitwise
    the same; the softmax sum of a sequence with a group that is not made of runs is multiplied by NaN before the loop
    (``_pack_layer_items``), and 1.0 otherwise. With ``key_transpose`` "transpose" the latent keys of FP8 row-major
    pages transpose in TensorE transpose mode (``_transpose_mode_tile``): the keys and the scores are the same, so the
    state is bitwise the same.
    ``latent_act`` 3 moves one more of the 4 exact latent key copies of a slot from VectorE
    to ScalarE. ``mask_column`` writes the key mask on GpSimd as an FP8 column of the page tile (``_mask_column``),
    which the rope transposes move into the bias row, with a query bias of 2^100: masked weights stay 0 and unmasked
    scores keep their value, so the state is bitwise the same. ``bg_rope`` transposes the rope keys in transpose mode:
    the same values, so the state is bitwise the same.
    ``bg_interleave`` emits the latent key transposes of slot j + 1 between the QK matmuls of slot j (scheduling-only
    dependencies): the same instructions, so the state is bitwise the same. ``carry`` runs that pipeline across the
    iteration fence and the loops (``interleaved_slots``): the same instructions and state adds in slot order, so the
    state is bitwise the same. With ``carry_main_only`` only the main loop carries, and its drain runs before the tail
    loops. With ``sum_after_copy`` (with ``bg_interleave``) the weight sum add of slot j - 2
    issues after the VectorE copy of latent key tile 0 of slot j + 2, in the hook of slot j + 1 (the last one of an
    iteration at its end): the sum adds keep the slot order on their own state column, so the state is bitwise the same.
    With ``rope_ahead`` (with ``carry``, not ``carry_main_only``, and ``bg_rope``) slot j traces the rope key transposes
    and the rope key copy of slot j + 1 (of the next iteration's slot 0 in the last slot, of slot 0 in the prologue) right
    after its rope matmul: on TensorE two of them in each of the first two gaps between the P@V matmuls of slot j - 2 and
    the weight transposes of slot j - 1 in the last two, and on ScalarE the copy before the exp of slot j, the weight copy
    of slot j - 1 after the exp and before the first ScalarE key copy of slot j + 1 (scheduling-only dependencies). The
    same instructions on the same pages, one slot earlier, so the state is bitwise the same.

    With ``pair`` (``_mla_decode_body`` ``pack_pair``: half 0 or 1 of two staged groups of B sequences, the 2B sequences,
    block table and positions of both, and ``rank_info``) the two groups run one loop over ``state_flat``
    ``[96, (2B + 1) x 513]``, whose columns ``half x B`` on hold the sequences of each half. The call of half 0 stages the
    slot rows of its sequences and the row of the null items (column 2B), and returns; the call of half 1 stages its own
    slot rows, builds the items of the 2B sequences (or takes ``pack_plan``, the plan of the pair built once for the
    block) and runs the loop. Each sequence adds its own groups in the same order, so its state is bitwise the same as
    in the loop of its group.

    The function is a generator (``_packed_key_loop`` runs it in one call). With ``setup_ahead`` (``_mla_decode_body``
    ``pack_setup_ahead``: a staged unit with its own plan, ``pack_plan``, and its own HBM rows and items, ``pack_hbm``) it
    yields once, after the items and the slot rows of its sequences are stored and before the loop: the caller runs that
    part of every unit of the layer first. It then calls ``after_setup``, which starts the state of the unit (the units
    of a layer share the state tile, so it must follow the loops of the units before it), then runs the sum scale of the
    page runs and the loop. The same instructions on the same data, so the state is bitwise the same.

    With ``mx_keys`` (``_mla_decode_body`` ``pack_mx_keys``: "single" or "split", with ``bg_interleave``, ``mask_column``
    and ``bg_rope``, and no ``pair``)
    the slot rows hold the e4m3 MX stationaries of the latent queries (``_mx_queries`` of the query gather ``mx_q_all``)
    in their first 96 words ("single") or 192 words ("split": hi, then lo), and the latent part of each score is one or
    two MX matmuls (``_slot_scores``) on the keys of 4 32-bit page transposes (``_mx_key_words``), which the QK hook of
    slot j issues for slot j + 1 right after its MX matmul. The loop's PSUM tensors are then 8 whole banks, allocated once
    before the loop: 2 score buffers, 2 key-word buffers, 2 numerators, the weight transposes and the 2 rope-key word
    buffers (one bank), so no transpose tile shares a bank with the scores, the weights or a numerator.

    With ``dr_keys`` (``_mla_decode_body`` ``pack_dr_keys``, with ``bg_interleave``, ``mask_column`` and ``bg_rope``) the slot
    rows hold the e4m3 double_row stationaries of the latent queries (``_dr_queries`` of the query gather ``mx_q_all``) in
    their first 192 words, and the latent part of each score is 2 double_row matmuls (``_slot_scores``) on the 2-byte
    key words of 8 transposes a slot (``_dr_key_words``): the QK hook of slot j issues those of block 0 of slot j + 1 after
    its first matmul and those of block 1 and their ScalarE copy after its second. The loop's PSUM tensors are 8 whole
    banks, as with ``mx_keys`` (the key-word buffers BF16), so no transpose tile shares a bank with the scores, the weights
    or P@V. With ``rope_ahead`` the weight copy of slot j - 1 issues before the exp of slot j (after the rope copy of slot
    j + 1), so P@V of slot j - 1 does not wait for the softmax of slot j. With ``pair`` the call of each half stages the
    stationaries of its own sequences.

    With ``queries`` Q > 1 (the multi-query verify, ``tkg_mla_block.mla_block_attention`` ``packed_prior``) an item holds the
    key group of all Q query rows of its sequence, and each slot runs Q sub-slots, one for each row: sub-slot j of the
    slot of sequence b is the slot body of the decode on the query, softmax reference and state column ``j B + b`` of
    row j, so ``state_flat`` is ``[96, (Q B + 1) x 513]`` (column ``Q B`` takes the null items), ``q_nope_all``,
    ``q_rope_all`` and the softmax reference have Q B columns, and the slot row of a sequence holds the Q rows of it
    (``[128, Q x 520]``). The key tiles of an item serve its Q sub-slots. The key mask of row j is mask column j of the
    page tile (bytes 576 to 576 + Q - 1), which compares the slot offset of a key minus j with the threshold of the item,
    and query bias row 64 + j selects it in the rope matmul (the bias rows of the other rows are 0, and an exact 0 times
    a mask value adds an exact 0). So each row takes the instructions of its decode step on the same pages and adds its
    groups in the same order: its state is bitwise that of the decode step. The 2-deep pipeline of ``bg_interleave``
    runs on the sub-slots; the latent keys of the next item transpose in the QK gaps of the last sub-slot of an item.
    An item lasts Q sub-slots, so 4 page buffers (not 8) serve the items, which leaves SBUF for the Q-row slot rows.
    The plan must count the groups of a sequence from its position plus Q - 1 and give the null items the threshold
    -Q (``_pack_items`` ``group_extra`` and ``null_threshold``). Only the served slot form: FP8 row-major pages with
    ``key_transpose`` "transpose", ``mask_column``, ``bg_rope``, ``bg_interleave`` and ``carry``; no MX keys, no pair,
    no ``rope_ahead``. ``after_writes`` (instructions, the owner writes of the current rows of the call, which later
    rows read back as prior keys) gives the first item-row load and the page loads of the prologue a runtime
    dependency on each of them; every later page load of the loop issues after those on the same SP queue.

    With Q > 1 rows, ``dr_keys`` takes the e4m3 double_row stationaries of the Q B rows in ``dr_q_rows``
    ``[128, Q B, 2, 2, 96]`` FP8 (``_dr_gather_tile_queries``) in place of the query gather: stage row j of a slot row
    holds those of row j, and the key words of the next item transpose in the 2 QK gaps of the last sub-slot of an item
    (one ScalarE copy an item). With Q > 1 rows, ``mx_keys`` "single" (FP8 pages, not with ``p2``) takes the MX query
    words ``[128, Q B, 96]`` FP32 of the rows in ``mx_q_rows`` (``tkg_mla_block`` ``_mx_query_words``: e4m3(q), saturated at
    448, as ``dr_keys`` rounds it) in place of the query gather: stage row j holds those of row j in its first 96 words,
    each sub-slot runs one MX matmul (unit scales) and the rope matmul, and the key words of the next item are 4 32-bit
    page transposes in the QK gap of the last sub-slot of an item and one ScalarE copy (``_mx_item_key_words``). The
    products are those of ``dr_keys``; the accumulation order differs, so the bits are not those of ``dr_keys``. It
    takes ``cap_first``, ``clamp_pin``, ``deep_pipe`` and ``item_rope_ahead`` as ``dr_keys`` does.
    ``item_rope_ahead`` (Q > 1) traces the rope key transposes of the next item between the
    P@V matmuls of the second-to-last sub-slot of an item (sub-slot Q - 2, or G - 2 with ``p2``) and their ScalarE copy
    after them (after the weight copy of that sub-slot), so the first sub-slot of the next item finds its rope keys done
    and its rope matmul does not wait for the rope copy behind the softmax of the sub-slot before it. The same
    instructions on the same pages, one sub-slot earlier, so the state is bitwise the same.

    With ``p2`` (Q > 1 rows, ``tkg_mla_block`` ``packed_prior_options`` "p2") the sub-slots of an item hold 128 (row, head)
    columns, not the 96 heads of one row: the Q x 96 columns of a sequence (column 96 j + h is head h of row j) split
    into G = Q x 96 / 128 groups (``pack_p2_runs``), and sub-slot g of an item runs the slot body on the 128 columns of
    group g. Each column is a stationary column of the QK matmuls, a partition of the scores, the cap, the exp, the
    weights and P@V, and a partition of ``state_flat`` ``[128, (G B + 1) x 513]`` (column ``g B + b``; the null items
    take column ``G B``), with the softmax reference, cap and query bias row of its own row. TensorE computes each
    stationary column on its own, and every other op acts on each partition on its own, so each column takes the
    instructions of its decode step and its state is bitwise that of the sub-slot of its row without ``p2``. Only the
    served slot form. With ``dr_keys`` stage row g holds the ``[128, 2, 2, 128]`` double_row stationaries of its 128
    columns (the bytes of ``dr_q_rows``, ``_stage_query_rows_p2``) and sub-slot g runs the 2 double_row matmuls on them
    (``_dr_stationary`` with ``cols`` 128): each column is a stationary column of the bytes of its row, so its state is
    bitwise that of the ``dr_keys`` sub-slot of its row. With ``item_rope_ahead`` the rope keys of the next item
    transpose in sub-slot G - 2, and the prologue's rope key copy also waits for the store of the slot rows, whose
    source bytes the loop buffers take.

    ``lookahead`` 2 (``_mla_decode_body`` ``pack_lookahead``, with ``bg_interleave``, one query row per item and ``prefetch``
    of at least 2): slot j issues the slot row and page DMAs of slot j + 2, not those of slot j + 1 (with ``carry`` the
    last two slots of an iteration load slots 0 and 1 of the next one, which are its first ``prefetch`` = 2 slots), and
    the slot rows take a ring of 8 (``_PACK_BUFFERS``) instead of 4. The row of slot j is read until its state add in
    slot j + 2, so with 4 rows the row DMA of slot j + 4 waits for that add; with 8 it overwrites the row of slot j - 4,
    whose add ran 2 slots before. The SP load DMAs of an iteration then keep the slot order (the page DMAs of a slot, then
    its row DMA, then the page DMAs of the next slot; scheduling-only dependencies): without them the compiler issues the
    rows, which do not wait with 8 rows, ahead of the pages of earlier slots. The same DMAs and data, so the state is
    bitwise the same.

    ``psum_per_block`` (``_mla_decode_body`` ``pack_psum_per_block``): the PSUM tensors of the loop that are allocated before
    it (the 2 score buffers of ``bg_interleave`` and, with ``dr_keys`` or ``mx_keys``, the 6 more tensors of ``banks``) are
    allocated again at the start of each loop body and before each drain, so no PSUM tensor is used in more than one
    top-level block (the verifier's rule: a memory location used in more than one top-level block must be in DRAM or
    SBUF; without the key the loop programs break it, NCC_IBIR092). Every block writes each tensor before it reads it (each
    first write is a start: a transpose, or the first matmul of a group with ``accumulate=False``), and every value that
    crosses a block boundary (the key words, rope keys and weights of the next slots, the slot rows and pages) is in SBUF,
    so the same instructions on the same data: the state is bitwise the same. Only PSUM tensor identities change.

    ``step_items`` (``_mla_decode_body`` ``pack_step_items``: the step store of this loop unit, a dict that the decode block
    keeps for the step, with ``tag`` and, with ``page_runs``, the persistent ``run_scale`` tile): the first layer of the
    step that runs the unit writes its item rows to an HBM tensor of the store and its run scale into the store tile; every
    later layer reads them and skips its own page-row gathers (``_pack_layer_items``). Only for MLA layers that share one
    block table (the decode builds one table for every MLA layer of a step): the item rows are then the same bytes, so
    the state is bitwise the same. With Q > 1 rows (the verify: ``tkg_mla_block`` ``pass_plans`` with ``packed_prior_options``
    ``step_items``, one store for each pass, whose run-scale tile is ``[sub_cols, B]``; the caller gives every MLA layer
    of a step the table of the first one) and with ``p2`` the same holds.

    ``stage_rows_sbuf`` (``_mla_decode_body`` ``pack_stage_rows_sbuf``, one query row per item, no ``pair``, no
    ``setup_ahead``, row-major pages): the slot rows of the first ``prefetch`` slots, which the prologue loads before the
    first loop, come from the SBUF stage tile (row b of ``[128, B + 1, 520]``, at the same item field) instead of from its
    HBM store, so they do not wait for the store to land. The stage tile then takes its SBUF outside the frame of the
    staging, so the loop tiles do not take its bytes while those DMAs read it. Every later slot row still loads from the
    HBM store. The same bytes, so the state is bitwise the same. With Q > 1 rows (``tkg_mla_block`` ``packed_prior_options``
    ``stage_rows_sbuf``) the stage tile is ``[128, B + 1, W]`` of the Q stage rows (or the G groups of ``p2``) and takes its
    SBUF before the item rows, so its writes do not wait for the item-row stores either; with ``p2`` no loop buffer then
    takes the bytes of the stage store's source, so their first writes need not wait for it.

    ``slot_rows_sbuf`` (``tkg_mla_block`` ``packed_prior_options`` ``slot_rows_sbuf``; needs ``stage_rows_sbuf`` with Q > 1
    rows, no ``row_dma_pool`` or ``row_dma_act``): every later slot row of the loops also comes from the SBUF stage tile
    (row b of ``[128, B + 1, W]``), not from its HBM store, so the loop reads no stage row from HBM (Q x 520 BF16 words
    a partition an item). The stage tile already lives through the loops, and nothing writes it after the staging; the
    same DMAs on the same queue move the same bytes, so the state is bitwise the same. The HBM store stays.

    ``skip_stage_store`` (``tkg_mla_block`` ``packed_prior_options`` ``skip_stage_store``; needs ``slot_rows_sbuf`` with
    ``p2``): the staging does not store the stage rows to HBM (``_stage_query_rows_p2`` ``hbm_store``). With
    ``slot_rows_sbuf`` every slot row of the prologue and the loops reads the SBUF stage tile, so no instruction reads
    that store. The SP queue then moves (B + 1) x 128 x G x 1360 bytes less in each pass. The slot rows read the same
    bytes, so the state is bitwise the same.

    ``cap_first`` (``tkg_mla_block`` ``packed_prior_options`` ``cap_first``; ``bg_interleave`` with Q > 1 rows, no
    ``rope_ahead``): the state adds of sub-slot s - 2 (the GpSimd TI index copy, the VectorE numerator and sum adds),
    traced with its P@V in sub-slot s, trace after the score cap of sub-slot s + 1 instead. VectorE runs in program
    order, so the cap of a sub-slot does not queue behind the numerator add of the P@V just before its QK, and the
    exp, which the weight transposes wait for, starts earlier. The adds keep their order, each on the same operands, so
    the state is bitwise the same; the numerator PSUM of a sub-slot is read by its add before the P@V two sub-slots later
    writes it again (checked at trace time).

    ``clamp_pin`` (``bg_interleave``, Q >= 3 query rows, ``clamp_pages`` on row-major pages, no ``rope_ahead`` or
    ``lookahead`` 2): the page clamp of the next item (or of the next iteration's first item) is two
    VectorE instructions of 2 pages each (``[128, 2, 576]``), not one a page, issued after the score caps of rows 0 and
    1 of the current item (scheduling-only dependencies) instead of where the scheduler puts it, just before the next
    item's key transposes in the last row, where the score cap of that row then queues behind it. The same operation on
    the same bytes, so the state is bitwise the same. False (the default) keeps the program.

    ``deep_pipe`` (``tkg_mla_block`` ``packed_prior_options`` ``deep_pipe``; ``bg_interleave`` with Q > 1 rows and
    ``carry``, no ``rope_ahead``, ``p2`` or ``lookahead`` 2): the weight transposes of sub-slot s - 2 and
    P@V of sub-slot s - 3, not those of s - 1 and s - 2, follow the QK of sub-slot s, so the score cap and the exp of a
    sub-slot have two sub-slots of TensorE work to hide behind, not one. The weights take a ring of 4 tiles (the
    weights of a sub-slot live until its transposes, two sub-slots later) and the weight sums a ring of 8 of their own
    (the sum add of a sub-slot, with ``cap_first`` after the score cap four sub-slots later, comes before the exp of the
    sub-slot that reuses its sum); a carried iteration starts with the transposes of sub-slots -2 and -1 and P@V of -3
    .. -1 (null sub-slots of zero weights on zero pages in the first one), and each drain runs the last two transposes
    and three P@Vs. The same instructions on the same operands, and the state adds keep their order, so the state is
    bitwise the same. False (the default) keeps the program.

    ``numerator_ring`` (opt-in; 0 keeps a new P@V numerator PSUM tensor for each sub-slot, which the compiler places;
    3 with ``bg_interleave`` and the default key form, no ``dr_keys`` / ``mx_keys``, whose pinned banks are all in use):
    P@V of sub-slot s writes buffer s % 3 of a ring of 3 numerator PSUM tensors (a bank each), so it does not wait for the
    state adds of sub-slot s - 2 to read their bank. The same instructions on the same values: the bits are the same.
    ``sum_batch`` (``tkg_mla_block`` passes it for every Q > 1 pass, also with ``p2``, where the G sub-slots of an item take
    the place of its Q rows; it applies where it fits:
    ``bg_interleave`` with 2 to 16 query rows, no ``p2`` or ``rope_ahead``, even item counts, else each sub-slot keeps its
    own sum add): the exp of row j of an item writes its weight sum into column j of a dense
    ``[96, 16]`` tile of the item (a ring of 2; columns Q.. stay zero), and the state adds of the item's last row add all
    Q sums with one VectorE TI add (16 elements: the Q sums, and zeros into distinct null-column words) through the item's
    batched sum offsets, which the last row's stage row holds in its own sum-offset column (``_stage_query_rows``) and
    that row's TI index copy takes along; the other rows add no sum. Each sum element still gets its one add, in item
    order, so the state is bitwise the same.

    ``row_depth`` (None, the default, keeps the program; or 8): the ring of slot rows takes that many tiles (4 by
    default at ``lookahead`` 1), so the row DMA of slot j waits for the readers of slot j - 8, not j - 4, and the
    in-order SP queue that issues the rows and the pages runs further ahead. The same rows in other tiles, so the state
    is bitwise the same.
    ``word_keys`` (opt-in; Q > 1 rows per item on FP8 row-major pages with ``pack_bg_interleave``, the default BF16
    query, no ``dr_keys``, ``mx_keys``, ``deep_pipe`` or ``numerator_ring``): the latent keys of an item take one 32-bit
    TensorE transpose a page in place of 4 FP8 transpose-mode transposes a page. In a sub-slot of the item before it, one
    ScalarE byte permute moves the latent bytes of each page row into words: byte ``4c + t`` of the permuted row is
    feature ``128 t + c``. The 32-bit transpose of page ``b`` (after latent QK ``b`` of the item's last sub-slot) puts
    word ``c`` of every key on partition ``c``, so byte lane ``t`` holds features ``128 t + c``: latent tile ``t``. One
    exact uint16 ScalarE copy moves the words to SBUF, and latent QK ``t`` reads lane ``t`` at element step 4 with the
    query tile ``t``. Each QK matmul takes the same keys and the same query tile as without ``word_keys``, so the
    scores, and the state, are bitwise the same (MEASURED on trn3: 0 differing of 131,072 scores). Per item: TensorE 4
    transposes in place of 16, no VectorE key copy (2 without ``word_keys``), ScalarE the permute and one copy in place
    of 2 copies. The key words take the 8 pinned PSUM banks of ``dr_keys``.
    ``keys_ahead`` (with ``word_keys`` and at least 6 sub-slots an item, as ``p2`` at Q = 8 has): the 32-bit transposes
    and the ScalarE copy of the next item's key words run in sub-slot G - 2 of an item, not G - 1 (and the byte
    permutes of its pages one sub-slot earlier, no earlier than the first permute row without ``keys_ahead``), so the
    first latent QK of the next item does not wait for that copy at the item boundary. The key words go to the PSUM bank
    of the next item's parity and to its own key set (a ring of ``_KEY_SETS``), whose last reader is an item 3 earlier.
    The same matmuls on the same operands, so the state is bitwise the same. False keeps the program.
    ``mx_qk`` (opt-in, accuracy class; ``word_keys`` with ``p2``): the latent part of each sub-slot's scores is one MX
    matmul (``nc_matmul_mx``) on the ``word_keys`` key words (unit E8M0 scales) in place of 4 BF16 QK matmuls. The stage
    rows hold the MX query words of the latent query (``_stage_query_rows_p2`` ``mx_qk``: word ``c`` of a column holds
    features ``c``, ``128 + c``, ``256 + c``, ``384 + c``, the grouping of the key words): "unit" (True) the e4m3 query
    at unit scales, saturated at 448 (the GPU decode's query), "block32" MXFP8 with one E8M0 scale a block of 32 values
    (8 partitions x 4 bytes; ``nisa.quantize_mx``), whose scale bytes the stage row carries to the matmul. The hook of the
    item's last sub-slot issues the 4 key-word transposes and the copy of the next item after its one MX matmul. Column
    n of the scores comes from key-word column n alone, so the scores follow whatever key order the key words have. The
    rope part, the own-key score and the softmax keep the BF16 query; the products are e4m3 x e4m3 with the 11-bit pair
    and quad sums of the MX matmul before the FP32 accumulation, so the bits are not those of the BF16 query (the GPU
    decode's e4m3 query class).
    ``mx_natural`` (opt-in, accuracy class; with ``mx_qk``; False keeps the program): ``q_nope_all`` holds the latent
    query tiles of the natural MX grouping (element ``(c, r, t, h)`` is feature ``4c + t``,
    ``tkg_mla_block._mx_natural_query_tiles``), and the 32-bit TensorE transpose of page ``b`` reads the latent words of the
    FP8 page rows themselves (word ``c`` of a row is features ``4c .. 4c + 3``), so no ScalarE byte permute runs (4 an
    item, each 128 x 512 bytes) and the loop has no permute buffers. The stage rows build the MX query words from the
    tiles as without ``mx_natural``, so word ``c`` of a column holds the same 4 features as the key words. The MX matmul
    contracts the same 512 products of a score ("unit": the same e4m3 query bytes) in other 4-value groups and partition
    order; "block32" scales the 32 features ``32m .. 32m + 31`` of a block (the OCP MX grouping). So the scores are not
    bitwise those of the permuted words (the 11-bit pair / quad sums and the FP32 accumulation round other sums); on
    values whose sums are all exact they are.
    ``p2_entry_act`` (opt-in; with ``p2``; False keeps the program): two DMA changes of the loop entry, the same bytes, so
    the state is bitwise the same. (1) The softmax references of the stage rows move to the partitions of their columns
    by 12 DMAs on the ScalarE HWDGE queue straight into their stage words, issued before the stage tile's zero fill
    (``_stage_query_rows_p2`` ``refs_act``), in place of 2 SBUF DMAs a run on the SP queue, each after the VectorE
    copies of its run, which in turn waited for the DMAs of the run before it (the same stage tile). (2) With
    ``stage_rows_sbuf`` the prologue's slot rows, which read the SBUF stage tile, load on the ScalarE HWDGE queue, so they
    do not queue behind the stage rows' HBM store on the SP queue. Plain copies (no DMA transposes) on both queues.
    (3) The zero fill of the stage tile on VectorE covers only the bytes that no copy writes, about a fifth of the tile.
    ``setup_off_pool`` (opt-in; with ``p2``; False keeps the program): three changes of the pass setup on GpSimd, the same
    values, so the state is bitwise the same. (1) With ``sum_batch`` the batched weight-sum offsets of the stage rows and
    of the prologue's null row (``_sum_batch_offsets``: an IOTA on GpSimd and VectorE work) depend only on the shapes of
    the pass, so the passes and layers share one tile of each through ``shared`` on ``pack_sb`` (once a step with the
    hoisted plans). (2) The prologue's zero fills of the batched sum tiles run on VectorE, not GpSimd. (3) The TI offset
    copies of the stage rows run on VectorE, not GpSimd.

    ``act_spread`` (opt-in; with ``word_keys``; False keeps the program): a new ScalarE order in each sub-slot of the
    loops of at least ``_ACT_SPREAD_ITEMS`` items (the 2-item block of the last key groups keeps the order without ``act_spread``), the same
    instructions on the same operands, so the state is bitwise the same. (1) The weight copy of sub-slot s - 1 issues
    before the exp of sub-slot s (``_slot_scores`` ``softmax_later``): its input is ready before the score cap ends, so
    the weight transposes of sub-slot s wait only for that exp, not for exp, sum read and copy in turn. (2) At most one
    group of the item's other ScalarE work runs in a sub-slot, after its exp (scheduling-only), and the next exp issues
    after it: the byte permutes of the next item's pages one a row in rows 0, 1, 2, 2 (two in row 2) where the loop does
    not clamp its pages (the rows without ``act_spread`` where it does), and the key-word copy (``word_keys_hook``) and
    the rope key copy (``item_rope_ahead``) of the next item after the exp of row G - 2. Without ``act_spread``, the
    scheduler runs three permutes of the item after next (ready once the key-word transposes that read their buffer end)
    and the key-word copy back to back between a weight copy and the next exp, which delays that exp and the weight
    transposes behind it (MEASURED on trn3: the sub-slot after the key-word transposes takes longer; ``act_spread``
    removes that delay, and a 1M main loop of 96 items takes less time, 1-core).

    ``capless`` (with ``p2`` and ``word_keys``; False keeps the program; only with the caller's check): the loop traces
    no score cap (``_slot_scores`` ``capless``). The exp reads the raw scores: while every score stays at or below its
    cap the min is a no-op and the state is bitwise the same; a score above its cap gives a weight of at least about
    ``exp(_WEIGHT_CAP)`` (or Inf / NaN), which the caller's check of the column sums catches and redoes with the cap
    (``tkg_mla_block._capless_redo``).
    The cap anchors of ``cap_first`` and ``clamp_pin`` are then None: the state adds and the pinned clamps issue in
    trace order.

    ``kv_pitch576`` (opt-in; Q > 1 rows with ``mask_column``, ``bg_rope``, ``bg_interleave``, ``lookahead`` 1 and the
    plan's ``clamp_block``, no ``mx_keys`` or ``dr_keys``): the page tiles take the 576 bytes of a cache row, without the
    mask column, so each page buffer is contiguous. With ``clamp_block`` every item of the main and tail loops is a full
    key group (no key at or past its row's threshold) or a null item, so those loops write no mask: their rope transposes
    take the 64 rope columns and the rope QK the 64 rope rows. The null items of the stage rows take a negative reference
    of ``-_NULL_ITEM_REFERENCE``, so every weight of a null item is 0 without the mask (as the masked scores give). The
    loop of the last key groups keeps the mask: GpSimd writes the Q mask columns of each of its items into a small FP8
    tile (``_mask_window``), whose window of 64 + Q columns ending at those columns transposes into the rope PSUM rows
    before the 64 rope columns overwrite rows 0 .. 63, so its rope keys and bias rows are those of the mask column. A
    full key group adds 0 x 2^100 to each score with the mask column and nothing here; the weights are the same, so the
    state is bitwise the same.
    ``kv_gather`` (opt-in; with ``kv_pitch576``, ``carry``, page runs of 0 or 2, an even ``prefetch`` and even main-loop
    slot counts, and fewer than 2^24 page rows): the main loop loads the pages of its items ``prefetch`` .. ``slots`` - 1
    with one GpSimd software DGE gather (a 2-D ``vector_offset``, DMA batching) for each 2 items, in place of their SP page
    DMAs: 1,024 rows of 576 bytes into the 2 adjacent (contiguous) page buffers of the items. The gather reads its index
    tile column-major: element ``e = 128 c + q`` (row ``p = e // 8`` of block ``e % 8`` of the 2 items) sits at row ``q`` and
    column ``c``, so every index of row ``q`` reads a page of block ``q % 8``. The indices build on VectorE once an
    iteration from its item row on every partition (``_kv_gather_index``). The items that load across the back edge (the
    first ``prefetch``) and the tail and clamped loops keep their SP page DMAs (a software DGE load does not stay in
    flight across a loop fence). The same rows in the same tiles, so the state is bitwise the same.
    ``kv_rem4`` (opt-in; with ``kv_pitch576``; accuracy class): each page lands on 32 partitions with 4 consecutive rows
    on each (rows 4 c .. 4 c + 3 of page j on partition 32 j + c, slots 0 .. 3), so partition p holds the item's keys 4
    p .. 4 p + 3 and every partition piece is 2,304 contiguous bytes of one page, whatever the block table. A page DMA
    then moves 32 pieces of 2,304 bytes, and a ``kv_gather`` element is one piece (256 elements a gather of 2 items).
    The key of slot b on partition p is key 4 p + b of the item (``_rem4_offsets_in_place`` gives the mask positions),
    so every 128-key block holds other keys than with row p on partition p: the PE partition sums of the scores and P@V,
    and the order of the weight sums, change (not bitwise; the same keys and weights).
    ``kv_group`` (opt-in, 0, 2, 4 or 8 = G; with ``kv_rem4``, ``carry`` and ``carry_main_only``, no ``kv_gather``, page
    runs of 0 or 2 (2 with G 2), a main-loop slot count that is a multiple of G and at most 2 prefetched slots): the
    main loop gathers the pages of its items G .. ``slots`` - 1 (4 .. with G 2) in groups of G items, one GpSimd
    software DGE gather each (128 G pieces of 2,304 bytes, 4 G pages), into G adjacent buffers of a ring of 2 G page
    buffers (with G 2 the 4 page buffers of Q > 1 rows, so no more SBUF for them). Each group's gather issues in the
    page preparation of its first slot - 2 (of that slot with G 2), after the last readers of its buffers, the items 2 G
    slots before, carried ones included. The items before the first group and the tail and clamped loops keep their SP
    page DMAs (4 rows of 2,304 bytes a partition with ``kv_rem4``). The indices build on VectorE once an iteration
    (``_kv_group_index``: G multiply-adds into the index tile). The next item row is partition 0 of ``threshold_next``
    (no tile of its own, no DMA of its own). The same pieces in the same buffers as ``kv_rem4``'s SP DMAs, so the state
    has the bits of ``kv_rem4``.
    ``mask_pool`` (``_mla_decode_body`` ``pack_mask_pool``; no ``mask_column``): GpSimd writes the mask row of each slot
    (``_mask_row``), not the engine that the compiler selects. The same compare and multiply on the same values, so the
    state is bitwise the same.
    """
    kernel_assert(not mask_pool or not mask_column, "pack mask_pool writes the mask row: it needs no pack_mask_column")
    kernel_assert(step_items is None or not setup_ahead, "pack_step_items does not take pack_setup_ahead")
    kernel_assert(not setup_off_pool or p2, "pack setup_off_pool needs pack_p2")
    kernel_assert(not p2_entry_act or p2, "pack p2_entry_act needs pack_p2")
    kernel_assert(
        not clamp_pin
        or (clamp_pages and not pair_pages and bg_interleave and queries >= 3 and not rope_ahead and lookahead == 1),
        "pack clamp_pin needs the page clamp on row-major pages, pack_bg_interleave with Q >= 3 query rows, and no "
        "pack_rope_ahead or pack_lookahead 2",
    )
    kernel_assert(
        not cap_first or (bg_interleave and queries > 1 and not rope_ahead),
        "pack cap_first needs pack_bg_interleave with Q > 1 query rows, and no pack_rope_ahead",
    )
    kernel_assert(
        not deep_pipe or (bg_interleave and queries > 1 and carry and not rope_ahead and not p2 and lookahead == 1),
        "pack deep_pipe needs pack_bg_interleave with Q > 1 query rows and pack_carry, and no pack_rope_ahead, "
        "pack_p2 or pack_lookahead 2",
    )
    # ``sum_batch`` applies where it fits: bg_interleave with 2 to 16 query rows (with p2: the G groups of an item), no
    # rope_ahead, and loops of an even item count (its sum ring); elsewhere each sub-slot keeps its own sum add.
    sum_batch = sum_batch and bg_interleave and 1 < queries <= _TI_GROUP and not rope_ahead
    kernel_assert(
        not stage_rows_sbuf or (pair is None and not setup_ahead and not pair_pages),
        "pack_stage_rows_sbuf needs row-major pages, and no pack_pair or pack_setup_ahead",
    )
    kernel_assert(
        not slot_rows_sbuf or (stage_rows_sbuf and queries > 1 and not row_dma_pool and not row_dma_act),
        "slot_rows_sbuf needs stage_rows_sbuf with Q > 1 rows, and no row_dma_pool or row_dma_act",
    )
    kernel_assert(
        not skip_stage_store or (slot_rows_sbuf and p2),
        "skip_stage_store needs slot_rows_sbuf on the p2 loop, because a slot row that reads HBM needs the stage "
        "store. Turn on slot_rows_sbuf with p2, or turn off skip_stage_store.",
    )
    kernel_assert(
        not p2 or (queries > 1 and queries * _HEADS % _P_MAX == 0),
        "pack_p2 needs Q > 1 query rows per item with Q x 96 a multiple of 128",
    )
    kernel_assert(
        queries == 1
        or (
            (mx_keys is None or (mx_keys == "single" and not p2))
            and pair is None
            and not pair_pages
            and not rope_ahead
            and key_transpose == "transpose"
            and mask_column
            and bg_rope
            and bg_interleave
            and carry
            and queries * tokens + 1 <= _P_MAX
            and _ROPE + queries <= _P_MAX
            and queries <= _MASK_PAGE_PITCH - _CACHE_WIDTH
        ),
        "a packed loop of Q > 1 query rows per item needs the served slot form (pack_key_transpose 'transpose', "
        "pack_mask_column, pack_bg_rope, pack_bg_interleave, pack_carry; pack_mx_keys only 'single' without pack_p2; no "
        f"pack_pair, pair_pages or pack_rope_ahead) and at most 127 state columns, got queries={queries}, tokens={tokens}",
    )
    kernel_assert(mx_keys in _MX_KEY_FORMS, f"pack_mx_keys must be one of {_MX_KEY_FORMS}, got {mx_keys!r}")
    kernel_assert(
        mx_keys is None
        or (
            bg_interleave
            and mask_column
            and bg_rope
            and pair is None
            and (
                (queries == 1 and mx_q_all is not None and mx_q_rows is None)
                or (queries > 1 and mx_q_all is None and mx_q_rows is not None and kv_pool.dtype == nl.float8_e4m3fn)
            )
        ),
        "pack_mx_keys needs pack_bg_interleave, pack_mask_column and pack_bg_rope, no pack_pair, and the query gather "
        "(one query row per item) or the MX query words of the rows on float8_e4m3fn pages (Q > 1 rows per item)",
    )
    kernel_assert(
        mx_q_rows is None
        or (
            mx_keys == "single"
            and mx_q_rows.dtype == nl.float32
            and tuple(mx_q_rows.shape) == (_P_MAX, queries * tokens, _HEADS)
        ),
        f"mx_q_rows needs pack_mx_keys 'single' and the FP32 shape {(_P_MAX, queries * tokens, _HEADS)}",
    )
    # With ``rope_ahead`` every slot finds its rope keys done by the slot before it (slot 0 of the first iteration by the
    # prologue), so every loop carries the pipeline.
    kernel_assert(
        not rope_ahead or (carry and not carry_main_only and bg_rope),
        "pack_rope_ahead needs pack_carry (not pack_carry_main_only) and pack_bg_rope",
    )
    kernel_assert(not (rope_ahead and mx_keys is not None), "pack_rope_ahead does not take pack_mx_keys")
    kernel_assert(
        not dr_keys
        or (
            mx_keys is None
            and bg_interleave
            and mask_column
            and bg_rope
            and key_transpose == "transpose"
            and not pair_pages
            and (
                (queries == 1 and mx_q_all is not None and dr_q_rows is None)
                or (queries > 1 and mx_q_all is None and dr_q_rows is not None)
            )
            and kv_pool.dtype == nl.float8_e4m3fn
        ),
        'pack_dr_keys needs FP8 row-major pages with pack_key_transpose "transpose", pack_bg_interleave, '
        "pack_mask_column and pack_bg_rope, no pack_mx_keys, and the query gather (one query row per item) or the "
        "double_row stationaries of the rows (Q > 1 rows per item)",
    )
    kernel_assert(
        dr_q_rows is None or (dr_keys and tuple(dr_q_rows.shape) == (_P_MAX, queries * tokens, 2, _DR_BLOCKS, _HEADS)),
        f"dr_q_rows needs dr_keys and the shape {(_P_MAX, queries * tokens, 2, _DR_BLOCKS, _HEADS)}",
    )
    kernel_assert(
        not word_keys
        or (
            queries > 1
            and mx_keys is None
            and not dr_keys
            and not deep_pipe
            and not numerator_ring
            and not psum_per_block
            and bg_interleave
            and mask_column
            and bg_rope
            and key_transpose == "transpose"
            and not pair_pages
            and kv_pool.dtype == nl.float8_e4m3fn
            and _GROUP_BLOCKS == _KV_LORA_TILES
        ),
        "pack word_keys needs Q > 1 query rows per item on FP8 row-major pages with pack_key_transpose 'transpose', "
        "pack_bg_interleave, pack_mask_column and pack_bg_rope, and no pack_dr_keys, pack_mx_keys, pack_deep_pipe, "
        "pack_numerator_ring or pack_psum_per_block",
    )
    mx_qk = "unit" if mx_qk is True else mx_qk
    kernel_assert(
        mx_qk in _MX_QK_FORMS and (not mx_qk or (word_keys and p2)),
        f"pack mx_qk must be one of {_MX_QK_FORMS} (True: unit) and needs pack_word_keys (its key words) with pack_p2, "
        f"got {mx_qk!r}",
    )
    # ``mx_natural``: the key words of the FP8 rows as they are (no byte permute), with the natural MX query tiles.
    natural = bool(mx_natural)
    kernel_assert(not natural or mx_qk, "pack mx_natural needs pack_mx_qk (its MX query words)")
    kernel_assert(
        not act_spread or (word_keys and not rope_ahead), "pack act_spread needs pack_word_keys and no pack_rope_ahead"
    )
    kernel_assert(
        not capless or (p2 and word_keys and mx_keys is None and not dr_keys),
        "pack capless needs pack_p2 and pack_word_keys, no pack_mx_keys or pack_dr_keys",
    )
    kernel_assert(
        not item_rope_ahead or (queries > 1 and not rope_ahead and bg_rope and bg_interleave),
        "item_rope_ahead needs Q > 1 query rows per item, pack_bg_rope and pack_bg_interleave, and no pack_rope_ahead",
    )
    kernel_assert(
        not setup_ahead
        or (
            queries == 1
            and after_setup is not None
            and pack_hbm is not None
            and (pack_plan is not None or (pair is not None and not pair["half"]))
        ),
        "setup_ahead needs one query row per item, after_setup, the HBM rows and items of the unit and the plan of the "
        "unit (none for half 0 of a pair)",
    )
    kernel_assert(
        row_dma_pool in _ROW_DMA_POOL_PERIODS
        and (row_dma_pool == 0 or (queries == 1 and not row_dma_act and not pair_pages)),
        f"pack_row_dma_pool must be one of {_ROW_DMA_POOL_PERIODS}; a period needs one query row per item, row-major "
        f"pages and no pack_row_dma_act, got row_dma_pool={row_dma_pool}, queries={queries}",
    )
    kernel_assert(
        lookahead in _PACK_LOOKAHEADS and (lookahead == 1 or (queries == 1 and bg_interleave and prefetch >= 2)),
        f"pack_lookahead must be one of {_PACK_LOOKAHEADS}; 2 needs one query row per item, pack_bg_interleave and "
        f"pack_prefetch 2 or more, got lookahead={lookahead}, queries={queries}, prefetch={prefetch}",
    )
    bf16 = nl.bfloat16
    fp32 = nl.float32
    int32 = nl.int32
    u16 = nl.uint16
    hwdge = nisa.dge_mode.hwdge
    sync = nisa.engine.sync
    num_pages = kv_pool.shape[0]
    group_count = local_blocks // _GROUP_BLOCKS
    group_keys = _GROUP_BLOCKS * _BLOCK
    pack_sb = sb if pack_sb is None else pack_sb
    kernel_assert(pack_plan is None or tail_slots is None, "pack_tail_slots needs the plan of the layer")
    own_tokens = tokens
    if pair is not None:
        kernel_assert(
            (pack_plan is None or (pair["half"] and pack_plan.get("ti") is None))
            and shared is not None
            and pack_hbm is not None
            and pair["half"] in (0, 1)
            and pair["tokens"] == 2 * tokens,
            "a packed loop pair needs two halves of equal size, the items of the layer or (half 1) a plan of the pair "
            "without TI offsets, and the shared tiles of the layer",
        )
        # The loop of the pair attends the sequences of both halves.
        tokens, block_table = pair["tokens"], pair["block_table"]
    if pair is None:
        plan = pack_plan or _pack_items(
            token_meta[0:tokens, 0:1],
            token_f[0:tokens, 2:3],
            block_table,
            tag,
            sb,
            slots,
            pack_sb,
            shared,
            tail_slots,
            tail_round,
        )
    elif pair["half"]:
        # A plan of the pair built once for the MLA layers of the block (``decode_block`` ``mla_pack_pair_plans``), else
        # the items of this layer.
        plan = pack_plan or mla_pack_plan(
            block_table,
            pair["position"],
            pair["rank_info"],
            tag,
            sb,
            slots,
            tail_slots,
            tail_round,
            plan_sbuf=pack_sb,
            with_ti=False,
            shared=shared,
        )
    else:
        plan = {}
    # Sub-slots of an item: one for each row (96 heads), or the G groups of 128 (row, head) columns (``p2``).
    subs = pack_p2_groups(queries) if p2 else queries
    # ``p2``: the store of the slot rows. The loop buffers take the bytes of its source tile, so their first writes (the
    # first loads and the writes of the prologue) wait for its completion (runtime dependencies).
    p2_stage = ()
    sub_cols = _P_MAX if p2 else _HEADS
    sub_width = _P2_STAGE_WIDTH if p2 else _STAGE_WIDTH
    sub_ti = _P2_STAGE_TI if p2 else _STAGE_TI
    sub_ref_word = _P2_STAGE_REF_WORD if p2 else _STAGE_REF_WORD
    # The state columns of the rows (column j B + b for row j of sequence b; with ``p2`` column g B + b for group g of
    # sequence b); the null items take the next one.
    columns = subs * tokens
    plan_ti = plan.get("ti")
    kernel_assert(queries == 1 or plan_ti is None, "a packed loop of Q > 1 rows builds the TI offsets of its columns")
    if plan_ti is None:
        # A plan without TI offsets (``mla_pack_plan(with_ti=False)``): the groups of a staged layer share one tile.
        def build_ti():
            ti_u = _plan_tile(pack_sb, (sub_cols, columns + 1, _TI_COLS), u16, f"{tag}_mla_plan_ti", 1)
            with sb.scope("pack_ti"):
                _pack_ti(ti_u, columns, sb)
            return ti_u

        plan_ti = _shared(shared, "plan_ti", build_ti)
    # A slot row holds the stage rows of the Q rows (the G groups with ``p2``) of its sequence, one after the other.
    stage_width = subs * sub_width

    def rows_hbm():
        return _shared(
            pack_hbm,
            "rows",
            lambda: nl.ndarray(
                (tokens + 1, _P_MAX, stage_width), dtype=bf16, buffer=nl.hbm, name=f"{tag}_mla_pack_rows"
            ),
        )

    if pair is not None and not pair["half"]:
        stage_hbm = rows_hbm()
        kernel_assert(
            tuple(stage_hbm.shape) == (tokens + 1, _P_MAX, _STAGE_WIDTH),
            "the MLA layers of a kernel share the packed loop HBM tensors, so they need the same shapes",
        )
        _stage_pair_rows(
            stage_hbm,
            plan_ti,
            q_nope_all,
            q_rope_all,
            softmax,
            own_tokens,
            0,
            mask_column,
            tag,
            sb,
            **({"dr_q_all": mx_q_all} if dr_keys else {}),
        )
        return
    slots = plan["slots"]
    tail = plan.get("tail_slots")
    # ``clamp_block`` (``_pack_items``): the tail-only page clamp. The main and tail loops hold no last key group, so
    # every row they read is a written row (finite) and they skip the page clamp; one more loop takes the block of the
    # last key groups and clamps its pages. Without the page clamp (zeroed pages) the block loop clamps nothing either.
    clamp_block = plan.get("clamp_block", 0)
    kernel_assert(
        not clamp_block or (carry_main_only and not deep_pipe),
        "the tail-only page clamp needs carry_main_only and no deep_pipe",
    )
    kernel_assert(
        not kv_pitch576
        or (
            queries > 1
            and mask_column
            and bg_rope
            and bg_interleave
            and lookahead == 1
            and clamp_block
            and not pair_pages
            and mx_keys is None
            and not dr_keys
            and not rope_ahead
        ),
        "pack kv_pitch576 needs Q > 1 rows with pack_mask_column, pack_bg_rope, pack_bg_interleave, pack_lookahead 1 and "
        "the clamp block (clamp_tail_only), and no pack_pair, pack_mx_keys, pack_dr_keys or pack_rope_ahead",
    )
    kernel_assert(
        not kv_gather
        or (
            kv_pitch576
            and carry
            and page_runs in (0, 2)
            and prefetch % 2 == 0
            and plan["slots"] % 2 == 0
            and kv_pool.shape[0] * _BLOCK < 2**24
        ),
        "pack kv_gather needs pack kv_pitch576 and pack_carry, page runs of 0 or 2, an even prefetch and main-loop slot "
        "count, and fewer than 2^24 page rows",
    )
    kernel_assert(
        not kv_rem4 or (kv_pitch576 and kv_pool.shape[0] * (_BLOCK // _REM4_ROWS) < 2**24),
        "pack kv_rem4 needs pack kv_pitch576 and fewer than 2^24 page pieces",
    )
    kernel_assert(
        kv_group in (0, *_KV_GROUP_SIZES)
        and (
            not kv_group
            or (
                kv_rem4
                and not kv_gather
                and carry
                and carry_main_only
                and page_runs in (0, 2)
                and (kv_group != 2 or page_runs == 2)
                and prefetch <= 2
                and plan["slots"] % kv_group == 0
            )
        ),
        f"pack kv_group is 0 or one of {_KV_GROUP_SIZES}, with pack kv_rem4, pack_carry and pack_carry_main_only, "
        "without pack kv_gather, page runs of 0 or 2 (2 with kv_group 2), at most 2 prefetched slots and a main-loop "
        f"slot count that is a multiple of it, got {kv_group}",
    )
    # With the tail-only page clamp the prologue and the main and tail loops do not clamp (``clamp_now``).
    clamp_now = [not clamp_block]
    # ``sum_batch`` needs loops of an even item count (its sum ring), decided before the stage rows take its offsets.
    loop_counts = (slots, *(tail or ()), *((clamp_block,) if clamp_block else ()))
    sum_batch = sum_batch and all(count % _SUM_BATCH_RING == 0 for count in loop_counts)
    max_iterations = plan["item_rows"] // slots
    row_width = slots * _ITEM_FIELDS
    q_width = _KV_LORA_TILES * _HEADS
    # With ``pack_hbm`` (staged groups) the HBM rows and items are one pair for the kernel, which the groups and
    # layers write in turn, so a collective destination of a later layer cannot take their bytes.
    stage_hbm = rows_hbm()
    item_shape = (plan["item_rows"] + plan["slots"], _ITEM_FIELDS)
    # ``step_items``: the item rows and run scale of this loop unit live in the step store (built by its first layer).
    step_form = None if step_items is None else (item_shape, num_pages, bool(clamp_pages), page_runs, tokens)
    reuse_items = step_items is not None and "item_hbm" in step_items
    if step_items is None:
        item_hbm = _shared(
            pack_hbm,
            "items",
            lambda: nl.ndarray(item_shape, dtype=int32, buffer=nl.hbm, name=f"{tag}_mla_pack_items"),
        )
    elif reuse_items:
        kernel_assert(
            step_items["form"] == step_form,
            f"pack_step_items reuses the items of {step_items['form']} (item rows, pages, clamp, page runs, sequences), "
            f"got {step_form}",
        )
        item_hbm = step_items["item_hbm"]
    else:
        item_hbm = nl.ndarray(item_shape, dtype=int32, buffer=nl.hbm, name=f"{step_items['tag']}_mla_pack_items")
    kernel_assert(
        tuple(stage_hbm.shape) == (tokens + 1, _P_MAX, stage_width) and tuple(item_hbm.shape) == item_shape,
        "the MLA layers of a kernel share the packed loop HBM tensors, so they need the same shapes",
    )
    kernel_assert(page_runs in (0, 2, _GROUP_BLOCKS), f"pack_page_runs must be 0, 2 or 4, got {page_runs}")
    kernel_assert(not (page_runs and pair_pages), "pack_page_runs needs row-major pages")
    if step_items is None:
        run_scale = sb.tile((sub_cols, tokens), fp32) if page_runs else None
    else:
        run_scale = step_items.get("run_scale") if page_runs else None
        kernel_assert(
            not page_runs or (run_scale is not None and tuple(run_scale.shape) == (sub_cols, tokens)),
            f"pack_step_items with pack_page_runs needs a step run-scale tile of {(sub_cols, tokens)}",
        )
    # ``stage_rows_sbuf`` with Q > 1 rows: the stage tile, which the prologue's slot rows read, takes its bytes before the
    # item rows (one query row per item allocates its tile with the staging below).
    stage_keep = None
    if stage_rows_sbuf and queries > 1:
        stage_keep = sb.tile((_P_MAX, tokens + 1, stage_width), bf16)
    # The page rows of this layer first, so their gather and store overlap the slot rows. With ``step_items`` only the
    # first layer of the step writes them; the later layers read its rows and run scale.
    if not reuse_items:
        _pack_layer_items(
            plan,
            block_table,
            num_pages,
            tag,
            sb,
            skip_null_page=not clamp_pages,
            item_hbm=item_hbm,
            page_runs=page_runs,
            run_scale=run_scale,
            null_copy_vector=null_copy_vector,
            null_runs=null_runs,
        )
    if step_items is not None and not reuse_items:
        step_items["item_hbm"] = item_hbm
        step_items["form"] = step_form
    if page_runs and queries == 1 and not setup_ahead:
        # sum x 1.0 keeps every bit; sum x NaN fails the sequences whose groups are not page runs.
        sums = state_flat.reshape((_HEADS, tokens + 1, _STATE_WIDTH))[0:_HEADS, 0:tokens, _KV_LORA:_STATE_WIDTH]
        nisa.tensor_tensor(dst=sums, data1=sums, data2=run_scale.reshape((_HEADS, tokens, 1)), op=nl.multiply)
    elif page_runs and p2:
        # The same scale for the state column of each group g of sequence b (column g B + b).
        state_rows = state_flat.reshape((sub_cols, columns + 1, _STATE_WIDTH))
        for group in range(subs):
            sums = state_rows[0:sub_cols, group * tokens : (group + 1) * tokens, _KV_LORA:_STATE_WIDTH]
            nisa.tensor_tensor(
                dst=sums,
                data1=sums,
                data2=run_scale.reshape((sub_cols, tokens, 1)),
                op=nl.multiply,
                engine=nisa.engine.vector,
            )
    elif page_runs and queries > 1:
        # The same scale for the state column of each row j of sequence b (column j B + b).
        state_rows = state_flat.reshape((_HEADS, columns + 1, _STATE_WIDTH))
        for row in range(queries):
            sums = state_rows[0:_HEADS, row * tokens : (row + 1) * tokens, _KV_LORA:_STATE_WIDTH]
            nisa.tensor_tensor(
                dst=sums,
                data1=sums,
                data2=run_scale.reshape((_HEADS, tokens, 1)),
                op=nl.multiply,
                engine=nisa.engine.vector,
            )
    if pair is not None:
        _stage_pair_rows(
            stage_hbm,
            plan_ti,
            q_nope_all,
            q_rope_all,
            softmax,
            own_tokens,
            1,
            mask_column,
            tag,
            sb,
            **({"dr_q_all": mx_q_all} if dr_keys else {}),
        )
    elif p2:
        with sb.scope("pack"):
            p2_store = _stage_query_rows_p2(
                stage_hbm,
                plan_ti,
                q_nope_all,
                q_rope_all,
                softmax,
                tokens,
                queries,
                tag,
                sb,
                **({"dr_rows": dr_q_rows} if dr_keys else {}),
                **({"stage_sb": stage_keep} if stage_keep is not None else {}),
                **({"sum_batch": True} if sum_batch else {}),
                **({"mx_qk": mx_qk} if mx_qk else {}),
                **({"refs_act": True} if p2_entry_act else {}),
                **({"hbm_store": False} if skip_stage_store else {}),
                **(
                    {
                        "ti_engine": nisa.engine.vector,
                        "sums_f": _shared(
                            shared,
                            ("p2_sum_offsets", pack_p2_groups(queries), tokens),
                            lambda: _shared_sum_offsets(pack_p2_groups(queries), tokens, sb, pack_sb, parts=_P_MAX),
                        )
                        if sum_batch
                        else None,
                    }
                    if setup_off_pool
                    else {}
                ),
                **({"null_reference": True} if kv_pitch576 else {}),
            )
            # ``stage_rows_sbuf``: no loop buffer takes the bytes of the kept stage tile.
            p2_stage = () if stage_keep is not None else (p2_store,)
    elif queries > 1:
        with sb.scope("pack"):
            _stage_query_rows(
                stage_hbm,
                plan_ti,
                q_nope_all,
                q_rope_all,
                softmax,
                tokens,
                queries,
                tag,
                sb,
                **({"dr_rows": dr_q_rows} if dr_keys else {}),
                **({"mx_rows": mx_q_rows} if mx_q_rows is not None else {}),
                **({"stage_sb": stage_keep} if stage_keep is not None else {}),
                **({"sum_batch": True} if sum_batch else {}),
                **({"null_reference": True} if kv_pitch576 else {}),
            )
    else:
        # ``stage_rows_sbuf``: the first slot rows of the prologue read the stage tile, so it lives outside the frame.
        if stage_rows_sbuf:
            stage_keep = sb.tile((_P_MAX, tokens + 1, _STAGE_WIDTH), bf16)
        sb.open_scope("pack")
        # Step P1: the slot row of each sequence, [q 4 x 96 | rope 96 | negative reference, cap (2 FP32) | TI
        # offsets (33 uint16) | pad] on each partition, with row B (null items) a copy of sequence 0. The TI offsets
        # are b * 513 + i in the snake layout (offset i at [i % 16, i // 16], the same for each group of 16).
        q_rows = q_nope_all.reshape((_P_MAX, tokens, q_width))
        stage_sb = sb.tile((_P_MAX, tokens + 1, _STAGE_WIDTH), bf16) if stage_keep is None else stage_keep
        nisa.memset(dst=stage_sb, value=0.0)
        stage_f = stage_sb.view(fp32)
        stage_u = stage_sb.view(u16)
        mx_rows = None
        if mx_keys is not None:
            # The MX stationaries of the latent queries in the q words of the slot rows (``_mx_queries``).
            mx_rows = sb.tile((_P_MAX, tokens, _mx_query_words(mx_keys)), fp32)
            for query_set in range(-(-tokens // QUERY_SET_TOKENS)):
                _mx_queries(mx_q_all, query_set, tokens, mx_rows, sb, split=mx_keys == "split")
        dr_rows = _dr_query_rows(mx_q_all, tokens, sb) if dr_keys else None
        for first, count in ((0, tokens), (tokens, 1)):
            if dr_rows is not None:
                _dr_stage_queries(stage_u, first, count, dr_rows)
            elif mx_rows is None:
                nisa.tensor_copy(
                    dst=stage_sb[0:_P_MAX, first : first + count, 0:q_width], src=q_rows[0:_P_MAX, 0:count, 0:q_width]
                )
            else:
                words = _mx_query_words(mx_keys)
                nisa.tensor_copy(
                    dst=stage_f[0:_P_MAX, first : first + count, 0:words],
                    src=mx_rows[0:_P_MAX, 0:count, 0:words],
                    engine=nisa.engine.vector,
                )
            nisa.tensor_copy(
                dst=stage_sb[0 : _BIAS_ROW + 1, first : first + count, q_width : q_width + _HEADS],
                src=q_rope_all[0 : _BIAS_ROW + 1, 0:count, 0:_HEADS],
            )
            for column, source in enumerate((softmax.negative_reference, softmax.score_cap)):
                word = _STAGE_REF_WORD + column
                nisa.tensor_copy(
                    dst=stage_f[0:_HEADS, first : first + count, word : word + 1], src=source[0:_HEADS, 0:count]
                )
        nisa.tensor_copy(dst=stage_u[0:_HEADS, 0 : tokens + 1, _STAGE_TI : _STAGE_TI + _TI_COLS], src=plan_ti)
        if mask_column:
            nisa.memset(
                dst=stage_sb[_BIAS_ROW : _BIAS_ROW + 1, 0 : tokens + 1, q_width : q_width + _HEADS],
                value=_MASK_COLUMN_SCALE,
            )
        nisa.dma_copy(
            dst=stage_hbm.permute((1, 0, 2)),
            src=stage_sb,
            dge_mode=hwdge,
            engine=sync,
            name=f"{tag}_mla_pack_rows_store",
        )
        sb.close_scope()
    if setup_ahead:
        # The caller runs this point of every unit of the layer first, then the rest of each unit in turn. The units
        # share the state tile, so the state of this unit starts here (``after_setup``), after the loops before it.
        yield
        after_setup()
        if page_runs:
            # The sum scale of the page runs (above), once the state starts: the same multiply.
            sums = state_flat.reshape((_HEADS, tokens + 1, _STATE_WIDTH))[0:_HEADS, 0:tokens, _KV_LORA:_STATE_WIDTH]
            nisa.tensor_tensor(
                dst=sums,
                data1=sums,
                data2=run_scale.reshape((_HEADS, tokens, 1)),
                op=nl.multiply,
                engine=nisa.engine.vector,
            )
    iteration_i = plan["iterations"]

    # Step P3: the loop. Slot j of an iteration uses its own page buffer, key set j % 4 and slot row j % 4. With
    # ``mx_keys`` the latent keys of a key set are the 32-bit MX words of its slot (``_mx_key_words``).
    key_sets = [
        (
            sb.tile((_P_MAX, group_keys), fp32)
            if mx_keys is not None or word_keys
            else sb.tile((_P_MAX, _DR_BLOCKS if dr_keys else _KV_LORA_TILES, group_keys), bf16),
            sb.tile((_P_MAX, group_keys), bf16),
            sb.tile((sub_cols, 1), fp32),
            sb.tile((_P_MAX, stage_width), bf16),
        )
        for _ in range(_KEY_SETS)
    ]
    # ``word_keys``: the byte-permuted latent pages of the next item, a ring of 2 by item parity (``mx_natural``: none,
    # the transposes read the page rows).
    word_perm = (
        [sb.tile((_P_MAX, _GROUP_BLOCKS, _KV_LORA), nl.uint8) for _ in range(2)] if word_keys and not natural else None
    )
    fp8_rows = kv_pool.dtype == nl.float8_e4m3fn and not pair_pages
    kernel_assert(fp8_rows or not mask_column, "pack_mask_column needs FP8 row-major pages")
    kernel_assert(mask_column or not bg_rope, "pack_bg_rope needs pack_mask_column")
    # With Q > 1 rows an item lasts Q sub-slots: the pages of item i load at the start of item i - 1 and P@V of its last
    # row runs in sub-slot 1 of item i + 1, before item i + 4 loads at the start of item i + 3, so 4 buffers serve it.
    page_buffers = _PACK_BUFFERS if queries == 1 else _PACK_BUFFERS // 2
    if kv_group:
        # ``kv_group``: a ring of 2 groups, so a group's gather can run while the group before is attended.
        page_buffers = 2 * kv_group
    page_all = None
    if kv_pitch576:
        # ``kv_pitch576``: the 576 bytes of a row, no mask column; with ``kv_gather`` the page buffers in one tile, so the
        # gather of 2 items fills 2 adjacent (contiguous) buffers.
        if kv_gather or kv_group:
            page_all = sb.tile((_P_MAX, page_buffers, _GROUP_BLOCKS, _CACHE_WIDTH), kv_pool.dtype)
            page_rows = [page_all[0:_P_MAX, k, 0:_GROUP_BLOCKS, 0:_CACHE_WIDTH] for k in range(page_buffers)]
        else:
            page_rows = [sb.tile((_P_MAX, _GROUP_BLOCKS, _CACHE_WIDTH), kv_pool.dtype) for _ in range(page_buffers)]
    elif mask_column:
        page_rows = [sb.tile((_P_MAX, _GROUP_BLOCKS, _MASK_PAGE_PITCH), kv_pool.dtype) for _ in range(page_buffers)]
        # On trn3 a DMA write and an engine write in one 32-byte SBUF block are ordered, so the page DMAs (bytes 0-575
        # of each row) and the mask column (byte 576) stay independent only on 32-byte aligned tiles, as the explicit
        # allocator places them.
        for rows in page_rows:
            address = _tile_address(rows)
            kernel_assert(
                address is None or address[1] % 32 == 0,
                f"pack_mask_column needs 32-byte aligned page tiles, got address {address}",
            )
    else:
        page_rows = [_page_tile(sb, _GROUP_BLOCKS, kv_pool.dtype, pair_pages) for _ in range(page_buffers)]
    ti_indices = [
        _shared(
            shared,
            f"pack_ti_{i}",
            lambda i=i: _ti_tile(pack_sb, (sub_cols, _TI_COLS), u16, f"{tag}_mla_pack_ti_{i}", 1 + i),
        )
        for i in range(_KEY_SETS)
    ]
    item_row = sb.tile((1, row_width), int32)
    thresholds = sb.tile((_P_MAX, slots), fp32)
    # The mask column compares the slot offset of each key (block b, row p: 4096 b + p) with the threshold of its slot on
    # every partition, so the thresholds load to all partitions; the first ``prefetch`` slots of an iteration write
    # their column in the iteration before, from their own FP32 copy.
    mask_offsets = sb.tile((_P_MAX, _GROUP_BLOCKS, queries), fp32) if mask_column else None
    thresholds_pf = sb.tile((_P_MAX, prefetch), fp32) if mask_column else None
    # ``kv_pitch576``: the mask columns of the items of the loop of the last key groups (``_mask_window``), one block of Q
    # columns a page buffer after 64 leading columns; the window of buffer k is columns kQ .. kQ + 64 + Q.
    mask_window = None
    if kv_pitch576:
        mask_window = sb.tile((_P_MAX, _GROUP_BLOCKS, _ROPE + page_buffers * queries), kv_pool.dtype)
        nisa.memset(dst=mask_window, value=0.0, engine=nisa.engine.gpsimd)
    item_view = item_hbm.reshape((max_iterations + 1, row_width))
    stage_view = stage_hbm.reshape((tokens + 1, _P_MAX * stage_width))
    # The item row of the next iteration and the slot row and pages of its slot 0 load during the iteration before
    # (the prologue loads those of iteration 0), so an iteration starts on loaded data. The last iteration loads the
    # pad rows of the item table and leaves them unused.
    # With ``mask_column`` the whole item row loads to every partition (128 descriptors of 1 KiB, not 32 x 128 of 4 bytes).
    # ``kv_group``: the next item row is then partition 0 of that tile (the same words), not a tile of its own (1 KiB of
    # every partition).
    items_shared = bool(kv_group) and mask_column
    if items_shared:
        threshold_next = sb.tile((_P_MAX, row_width), int32)
        item_next = threshold_next[0:1, 0:row_width]
    else:
        item_next = sb.tile((1, row_width), int32)
        threshold_next = sb.tile((_P_MAX, row_width if mask_column else slots), int32)
    # With ``prefetch`` > 1 the slot rows and pages of the first ``prefetch`` slots of the next iteration load at the
    # end of the iteration. Slot j uses slot row j % row_sets: more than 4 prefetched slots need their own rows.
    kernel_assert(
        1 <= prefetch <= min(slots if tail is None else tail[-1], page_buffers),
        f"pack_prefetch must be 1 to the {slots} slots (and the tail slots, and {page_buffers}), got {prefetch}",
    )
    # SP otherwise issues every DMA of a slot (the row DMA too); ScalarE has slack in
    # the loop.
    row_engine = nisa.engine.scalar if row_dma_act else sync

    def row_on_pool(slot):
        """``row_dma_pool`` k: the slot row DMA of slot j with j % k == k - 1 issues on GpSimd (software DGE), so the SP
        queue issues the 3 DMAs of the other slots only. SP paces the form-B slot (its SP instructions a
        slot take longer than its TensorE work); GpSimd has the slack."""
        return row_dma_pool > 0 and slot % row_dma_pool == row_dma_pool - 1

    kernel_assert(
        key_transpose in ("matmul", "transpose"), f"pack_key_transpose must be matmul or transpose, got {key_transpose}"
    )
    # Other page forms already transpose in transpose mode. The transpose-mode transposes read an FP8 identity of the
    # loop, not one of the kernel's identity cache (``install_identities``).
    transpose_identity = None
    if key_transpose == "transpose" and kv_pool.dtype == nl.float8_e4m3fn and not pair_pages:
        transpose_identity = sb.tile((_P_MAX, _P_MAX), kv_pool.dtype)
        nl.shared_identity_matrix(_P_MAX, dtype=kv_pool.dtype, dst=transpose_identity)
    kernel_assert(latent_act in _LATENT_ACT_TILES, f"pack_latent_act must be 2 or 3, got {latent_act}")
    kernel_assert(
        transpose_identity is not None or not bg_interleave,
        'pack_bg_interleave needs pack_key_transpose "transpose" on FP8 row-major pages',
    )
    # With ``bg_interleave`` the weights of slot j live until the weight transposes in slot j + 1 and their transposed
    # copy until P@V in slot j + 2 (``interleaved_slots``).
    kernel_assert(
        not deep_pipe or all(count * subs % _DEEP_SUMS == 0 for count in (slots, *(tail or ()))),
        f"pack deep_pipe needs loops of a multiple of {_DEEP_SUMS} sub-slots (its weight-sum ring), got the slot counts "
        f"{(slots, *(tail or ()))} of {subs} sub-slots",
    )
    # ``deep_pipe``: the weights of sub-slot s live until its transposes in sub-slot s + 2 (a ring of 4), and its weight
    # sum until its sum add after the score cap of sub-slot s + 4 (a ring of 8 of its own).
    weight_ring = _DEEP_WEIGHTS if deep_pipe else 2
    deep_sums = [sb.tile((sub_cols, 1), fp32) for _ in range(_DEEP_SUMS)] if deep_pipe else None

    def group_sum_of(slot):
        """The weight sum tile of (sub-)slot ``slot``: that of its key set, or with ``deep_pipe`` of its own ring."""
        return key_sets[slot % _KEY_SETS][2] if deep_sums is None else deep_sums[slot % _DEEP_SUMS]

    # ``sum_batch``: the exp of row j of an item writes its weight sum into column j of the item's dense [96, 16] tile
    # (columns Q.. stay zero), and the last row's state adds add its 16 columns with one TI add (``add_to_state``).
    batch_sums = [sb.tile((sub_cols, _TI_GROUP), fp32) for _ in range(_SUM_BATCH_RING)] if sum_batch else None

    def sum_dst_of(slot):
        """The weight sum destination of sub-slot ``slot``'s exp: ``group_sum_of``, or with ``sum_batch`` its column of the
        item's dense sum tile."""
        if batch_sums is None:
            return group_sum_of(slot)
        row = slot % subs
        return batch_sums[(slot // subs) % _SUM_BATCH_RING][0:sub_cols, row : row + 1]

    weights_bufs = [sb.tile((sub_cols, group_keys), bf16) for _ in range(weight_ring)] if bg_interleave else None
    weights_t_bufs = [sb.tile((_P_MAX, _GROUP_BLOCKS, sub_cols), bf16) for _ in range(2)] if bg_interleave else None
    # With ``bg_interleave`` the scores of slot j + 1 go to the other PSUM buffer, so QK of slot j + 1 does not wait for
    # the exp of slot j.
    # With ``mx_keys`` every PSUM tensor of the loop is allocated here, a whole bank each (the weight transposes padded
    # to one, the two rope buffers in one; 8 tensors with the scores): the 32-bit key transposes of slot j + 1 alternate
    # between two banks, P@V alternates between two numerators, and no transpose tile shares a bank with the scores, the
    # weights or P@V.
    mx_psum = None
    mx_scale = None
    # With ``dr_keys`` the same 8 whole banks (the key-word buffers BF16 2-byte words of the 2 blocks); ``banks`` is the
    # plan of either key form, None without them (the default program).
    key_words = (_P_MAX, group_keys) if mx_keys is not None or word_keys else (_P_MAX, _DR_BLOCKS * group_keys)

    def loop_psum(parts=("scores", "keys", "numerator", "weights", "rope")):
        """The PSUM tensors of the loop: the score buffers (``bg_interleave``) and ``banks`` (``dr_keys`` or ``mx_keys``).
        With ``psum_per_block`` each top-level block takes its own (``parts``: the ones the block uses, the others None)."""
        scores = (
            [nl.ndarray((sub_cols, group_keys), dtype=fp32, buffer=nl.psum) for _ in range(2)]
            if bg_interleave and "scores" in parts
            else None
        )
        if mx_keys is None and not dr_keys and not word_keys:
            return scores, None
        key_dtype = fp32 if mx_keys is not None or word_keys else bf16
        return scores, {
            "keys": [nl.ndarray(key_words, dtype=key_dtype, buffer=nl.psum) for _ in range(2)]
            if "keys" in parts
            else None,
            "numerator": [nl.ndarray((sub_cols, _KV_LORA), dtype=fp32, buffer=nl.psum) for _ in range(2)]
            if "numerator" in parts
            else None,
            "weights": nl.ndarray(
                (_P_MAX, _GROUP_BLOCKS, _PSUM_BANK_BYTES // (2 * _GROUP_BLOCKS)), dtype=bf16, buffer=nl.psum
            )
            if "weights" in parts
            else None,
            # The two rope-word buffers (FP8 words at element step 2, with the mask column, or the Q mask columns) in one
            # tensor: a PSUM tensor takes a bank of its own, and the loop has 8.
            "rope": nl.ndarray(
                (_ROPE + queries, 2 * _FP8_TRANSPOSE_STEP * group_keys), dtype=kv_pool.dtype, buffer=nl.psum
            )
            if "rope" in parts
            else None,
        }

    scores_bufs, banks = loop_psum()
    kernel_assert(
        numerator_ring in (0, 3) and (not numerator_ring or (banks is None and bg_interleave and not psum_per_block)),
        f"pack numerator_ring is 0 or 3, with pack_bg_interleave and the default key form (no dr_keys, mx_keys or "
        f"psum_per_block), got {numerator_ring}",
    )
    numerator_bufs = (
        [nl.ndarray((sub_cols, _KV_LORA), dtype=fp32, buffer=nl.psum) for _ in range(numerator_ring)]
        if numerator_ring
        else None
    )
    # The BF16 identity of the 2-byte key transposes (``dr_keys``).
    key16_identity = None
    if dr_keys:
        key16_identity = sb.tile((_P_MAX, _P_MAX), bf16)
        nl.shared_identity_matrix(_P_MAX, dtype=bf16, dst=key16_identity)
    if mx_keys is not None or mx_qk:
        mx_psum = banks if mx_keys is not None else None

        def unit_scales():
            """The unit E8M0 scales of the MX matmuls, at an explicit address: the loop only reads them."""
            scale = _plan_tile(pack_sb, (_P_MAX, group_keys), nl.uint8, f"{tag}_mla_pack_mx_scale", _MX_SCALE_PLAN_SLOT)
            nisa.memset(dst=scale, value=_MX_UNIT_SCALE, engine=nisa.engine.gpsimd)
            return scale

        mx_scale = _shared(shared, "pack_mx_scale", unit_scales)
    rope_identity = transpose_identity if bg_rope else None
    if bg_rope and rope_identity is None:
        rope_identity = sb.tile((_P_MAX, _P_MAX), kv_pool.dtype)
        nl.shared_identity_matrix(_P_MAX, dtype=kv_pool.dtype, dst=rope_identity)
    # ``lookahead`` 2: the row of slot j + 2 loads in slot j, while the rows of slots j - 2 .. j + 1 are still read.
    row_sets = max(_KEY_SETS if lookahead == 1 else _PACK_BUFFERS, prefetch, row_depth or 0)
    kernel_assert(
        row_depth in (None, _PACK_BUFFERS) and (row_depth is None or row_sets == row_depth),
        f"pack row_depth must be None or {_PACK_BUFFERS} (and at least the ring of lookahead / prefetch), got {row_depth}",
    )
    slot_rows = [key_set[3] for key_set in key_sets]
    slot_rows += [sb.tile((_P_MAX, stage_width), bf16) for _ in range(row_sets - _KEY_SETS)]

    # With ``carry`` slot j - 1 and j - 2 of slot 0 and 1 are the last slots of the iteration before, so every loop's
    # slot count is a multiple of each buffer ring; with 4 slot rows a third prefetched slot would load the slot row of
    # carried slot -2. With ``lookahead`` 2 the slot rows take a ring of 8, as the page buffers do: carried slots -2 and
    # -1 hold rows and pages 6 and 7, and the rows and pages of prefetched slot k (k < 6) were last read by slot
    # count - 8 + k, whose state add ran 2 slots later, before the end of the iteration. So up to 4 slots load before
    # the iteration fence.
    max_carry_prefetch = _PACK_CARRY_PREFETCH_LOOKAHEAD if lookahead > 1 else 2
    kernel_assert(not carry or bg_interleave, "pack_carry needs pack_bg_interleave")
    kernel_assert(not sum_after_copy or bg_interleave, "pack_sum_after_copy needs pack_bg_interleave")
    kernel_assert(carry or not carry_main_only, "pack_carry_main_only needs pack_carry")
    kernel_assert(
        not carry
        or (
            prefetch <= max_carry_prefetch
            and all(c % page_buffers == 0 for c in (slots, *(() if kv_group and carry_main_only else tail or ())))
        ),
        f"pack_carry needs pack_prefetch 1 or 2 (up to {_PACK_CARRY_PREFETCH_LOOKAHEAD} with pack_lookahead 2) and "
        f"slot counts that are multiples of {page_buffers}, got prefetch={prefetch}, lookahead={lookahead}",
    )
    # With Q rows an item has Q mask columns, which the rope transposes carry into bias rows 64 .. 64 + Q - 1.
    rope_width = _ROPE + queries if mask_column else _ROPE
    # The rows of the rope QK matmul: the rope rows and the bias rows (the mask row, or the Q mask columns).
    qk_rope_rows = _BIAS_ROW + queries

    def masked_now():
        """Whether the loop traced now masks its keys: always, but with ``kv_pitch576`` only in the loop of the last key
        groups (``clamp_now``)."""
        return not kv_pitch576 or clamp_now[0]

    def qk_rope_rows_now():
        """The rope QK rows of the loop traced now: the 64 rope rows without the mask (``kv_pitch576`` outside the loop of
        the last key groups)."""
        return qk_rope_rows if masked_now() else _ROPE

    def mask_window_of(slot):
        """``kv_pitch576`` in the loop of the last key groups: the mask window of ``slot`` (``_mask_window``), else None."""
        if not kv_pitch576 or not clamp_now[0]:
            return None
        return _mask_window(mask_window, slot % page_buffers, queries)

    def load_items(dst, dst_thresholds, step, name, view=item_view, count=slots):
        """Load item row ``step`` + 1 (``step`` a loop register), or row 0 (None), of ``count`` items of ``view``, and
        its mask thresholds. Return the item row DMA and, with ``mask_column``, the threshold DMA."""
        width = count * _ITEM_FIELDS
        offset = {"scalar_offset": step, "indirect_dim": 0} if step is not None else {}
        if items_shared:
            # ``kv_group``: the item row is partition 0 of the threshold DMA's tile.
            return [
                nisa.dma_copy(
                    dst=dst_thresholds[0:_P_MAX, 0:width],
                    src=view.ap(pattern=[[0, _P_MAX], [1, width]], offset=width if step is not None else 0, **offset),
                    dge_mode=hwdge,
                    engine=sync,
                    name=f"{tag}_mla_pack_threshold_{name}",
                )
            ]
        item_dma = nisa.dma_copy(
            dst=dst,
            src=view.ap(pattern=[[width, 1], [1, width]], offset=width if step is not None else 0, **offset),
            dge_mode=hwdge,
            engine=sync,
            name=f"{tag}_mla_pack_item_{name}",
        )
        if mask_column:
            threshold_dma = nisa.dma_copy(
                dst=dst_thresholds[0:_P_MAX, 0:width],
                src=view.ap(pattern=[[0, _P_MAX], [1, width]], offset=width if step is not None else 0, **offset),
                dge_mode=hwdge,
                engine=sync,
                name=f"{tag}_mla_pack_threshold_{name}",
            )
            return [item_dma, threshold_dma]
        nisa.dma_copy(
            dst=dst_thresholds[_BIAS_ROW : _BIAS_ROW + 1, 0:count],
            src=view.ap(
                pattern=[[width, 1], [_ITEM_FIELDS, count]],
                offset=(width if step is not None else 0) + 5,
                **offset,
            ),
            dge_mode=hwdge,
            engine=sync,
            name=f"{tag}_mla_pack_threshold_{name}",
        )

    # The page-run rows hold the sequence after the first page of each run (``_pack_layer_items``).
    run_dmas = _GROUP_BLOCKS // page_runs if page_runs else 0
    row_field = run_dmas or 4

    def entry_on_act(name, stage_tile):
        """``p2_entry_act``: the prologue's slot rows that read the SBUF stage tile load on the ScalarE queue."""
        return p2_entry_act and stage_tile is not None and name == "first"

    def load_row(slot, row_tile, name, stage_tile=None):
        """Load the slot row of ``slot`` from the item row in ``row_tile``; return the DMA. With ``row_dma_pool`` k the
        row of slot j with j % k == k - 1 loads on the GpSimd software DGE queue (``row_on_pool``). With ``stage_tile``
        (``stage_rows_sbuf``) the row comes from the SBUF stage tile, not from its HBM store; with ``p2_entry_act`` such a
        prologue row loads on the ScalarE queue (``entry_on_act``)."""
        fields = slot * _ITEM_FIELDS
        on_pool = row_on_pool(slot)
        index = row_tile[0:1, fields + row_field : fields + row_field + 1]
        if stage_tile is None:
            src = stage_view.ap(
                pattern=[[stage_width, _P_MAX], [1, stage_width]], offset=0, scalar_offset=index, indirect_dim=0
            )
        else:
            # Row b of the stage tile [128, B + 1, 520]: the same bytes as row b of its HBM store.
            src = stage_tile.ap(
                pattern=[[(tokens + 1) * stage_width, _P_MAX], [1, stage_width]],
                offset=0,
                scalar_offset=index,
                indirect_dim=1,
            )
        engine = nisa.engine.scalar if entry_on_act(name, stage_tile) else row_engine
        return nisa.dma_copy(
            dst=slot_rows[slot % row_sets],
            src=src,
            dge_mode=nisa.dge_mode.swdge if on_pool else hwdge,
            engine=nisa.engine.dma if on_pool else engine,
            name=f"{tag}_mla_pack_row_{name}_{slot}",
        )

    # ``lookahead`` 2: the load DMAs of the SP queue keep the slot order within an iteration (the pages of a slot, then its
    # row, then the pages of the next slot; scheduling-only dependencies): with 8 slot rows the rows wait for no state add
    # of the iteration, and the compiler would issue them ahead of the pages of earlier slots, which TensorE needs first.
    sp_chain = [None]

    def chained(dma):
        """``lookahead`` 2: issue the SP DMA ``dma`` after the previous one of the chain; return ``dma``."""
        if lookahead > 1:
            sp_chain[0] = _pinned(dma, sp_chain[0])
        return dma

    def load_slot(slot, row_tile, name, threshold=None, stage_tile=None, pages=True):
        """Load the slot row and the 4 pages of ``slot`` from the item row in ``row_tile``. With ``mask_column``, first
        write the mask column of the page tile from ``threshold`` (``[128, 1]`` FP32), before the page DMAs. Return the
        page DMAs (the run DMAs with ``page_runs``; none with ``pair_pages``). With ``lookahead`` 2 the row DMA follows the
        page DMAs (``chained``; a row on the ScalarE queue, ``row_dma_act``, or on GpSimd, ``row_dma_pool``, stays
        out of the chain). ``stage_tile``: the source of the slot row (``load_row``). ``pages`` False (``kv_gather``):
        the slot row only (a gather loads the pages)."""
        rows = page_rows[slot % page_buffers]
        if mask_column and not kv_pitch576:
            _mask_column(rows, mask_offsets, threshold)
        if not pages:
            load_row(slot, row_tile, name, stage_tile)
            return []
        if kv_rem4:
            # ``kv_rem4``: page j (field j, or its run's first page + j % runs) to partitions 32 j .. 32 j + 31, 4 rows
            # (2,304 contiguous bytes) on each; a page run is one DMA of its pages. The row DMA keeps its place.
            if not page_runs and lookahead == 1:
                load_row(slot, row_tile, name, stage_tile)
            page_dmas = []
            runs_each = page_runs or 1
            for run in range(_GROUP_BLOCKS // runs_each):
                entry = slot * _ITEM_FIELDS + run
                first = run * runs_each
                page_dma = nisa.dma_copy(
                    dst=rows[first * _REM4_PARTS : (first + runs_each) * _REM4_PARTS, 0:_GROUP_BLOCKS, 0:_CACHE_WIDTH],
                    src=kv_pool.ap(
                        pattern=[[kv_pool.shape[1], runs_each], [_REM4_PIECE, _REM4_PARTS], [1, _REM4_PIECE]],
                        offset=0,
                        scalar_offset=row_tile[0:1, entry : entry + 1],
                        indirect_dim=0,
                    ),
                    dge_mode=hwdge,
                    engine=sync,
                    name=f"{tag}_mla_pack_page_{name}_{slot}_r{run}",
                )
                page_dmas.append(chained(page_dma))
            if page_runs or lookahead > 1:
                row_dma = load_row(slot, row_tile, name, stage_tile)
                if not row_dma_act and not row_on_pool(slot):
                    chained(row_dma)
            return page_dmas
        if page_runs:
            # Partition p of page j of a run gets row p of the page, the bytes of the page DMA. The run DMAs come
            # before the row DMA, so their index loads (fields 0 .. runs) ascend and can merge into one.
            page_dmas = []
            for run in range(run_dmas):
                entry = slot * _ITEM_FIELDS + run
                page_dma = nisa.dma_copy(
                    dst=rows[0:_P_MAX, run * page_runs : (run + 1) * page_runs, 0:_CACHE_WIDTH],
                    src=kv_pool.ap(
                        pattern=[[_CACHE_WIDTH, _BLOCK], [kv_pool.shape[1], page_runs], [1, _CACHE_WIDTH]],
                        offset=0,
                        scalar_offset=row_tile[0:1, entry : entry + 1],
                        indirect_dim=0,
                    ),
                    dge_mode=hwdge,
                    engine=sync,
                    name=f"{tag}_mla_pack_page_{name}_{slot}_run{run}",
                )
                page_dmas.append(chained(page_dma))
            row_dma = load_row(slot, row_tile, name, stage_tile)
            if not row_dma_act and not row_on_pool(slot) and not entry_on_act(name, stage_tile):
                chained(row_dma)
            return page_dmas
        if lookahead == 1:
            load_row(slot, row_tile, name, stage_tile)
        page_dmas = []
        for local_block in range(_GROUP_BLOCKS):
            entry = slot * _ITEM_FIELDS + local_block
            page_name = f"{tag}_mla_pack_page_{name}_{slot}_{local_block}"
            if pair_pages:
                load_pair_page(kv_pool, rows, local_block, row_tile[0:1, entry : entry + 1], page_name)
                continue
            page_dma = nisa.dma_copy(
                dst=rows[0:_P_MAX, local_block, 0:_CACHE_WIDTH],
                src=kv_pool.ap(
                    pattern=[[_CACHE_WIDTH, _BLOCK], [1, _CACHE_WIDTH]],
                    offset=0,
                    scalar_offset=row_tile[0:1, entry : entry + 1],
                    indirect_dim=0,
                ),
                dge_mode=hwdge,
                engine=sync,
                name=page_name,
            )
            page_dmas.append(chained(page_dma))
        if lookahead > 1:
            row_dma = load_row(slot, row_tile, name, stage_tile)
            if not row_dma_act and not row_on_pool(slot) and not entry_on_act(name, stage_tile):
                chained(row_dma)
        return page_dmas

    # The slot offsets of the mask columns: 4096 b + p for key (b, p), minus j for mask column j of Q > 1 rows
    # (``kv_rem4``: those of ``_rem4_offsets_in_place``, written once into the tile before the loops).
    mask_pattern = [[_GLOBAL_BLOCK_STRIDE, _GROUP_BLOCKS], [1, 1] if queries == 1 else [-1, queries]]
    rem4_offsets = kv_rem4 and mask_column
    if rem4_offsets:
        # No tile of its own: the start of each partition takes 2 words of the prefetch thresholds, which the loop
        # writes later.
        start = thresholds_pf if prefetch >= 2 else sb.tile((_P_MAX, 2), fp32)
        _rem4_offsets_in_place(mask_offsets, start)
    # ``kv_group``: its pass-invariant constants, built once a staged layer in plan tiles. A hoisted plan heap
    # (``pack_sb`` of ``mla_pass_plans``, sized for the plans of the step) does not take them: they then go on the
    # loop's tiles, built once a call.
    vog5 = None
    if kv_rem4 and kv_group:

        def vog5_constants(region):
            return _vog5_constants(region, sb, kv_group, page_runs, tag)

        if pack_sb is sb:
            vog5 = _shared(
                shared,
                f"pack_vog5_{kv_group}_{page_runs}",
                lambda: vog5_constants(pack_sb),
            )
        else:
            vog5 = vog5_constants(sb)

    def write_mask_offsets():
        # ``kv_rem4``: the tile already holds its offsets (only the mask writes read it).
        if not rem4_offsets:
            nisa.iota(dst=mask_offsets, pattern=mask_pattern, offset=0, channel_multiplier=1)

    def prefetch_thresholds():
        """Copy the thresholds of the first ``prefetch`` slots of the next iteration to FP32 for their mask columns."""
        if mask_column:
            write_mask_offsets()
            nisa.tensor_copy(
                dst=thresholds_pf,
                src=threshold_next[0:_P_MAX, 5 : prefetch * _ITEM_FIELDS : _ITEM_FIELDS],
                engine=nisa.engine.gpsimd,
            )

    def slot_threshold(tile, slot):
        return tile[0:_P_MAX, slot : slot + 1] if mask_column else None

    # ``kv_gather``: the gather indices of the items of an iteration (``_kv_gather_index``), built from its item row on
    # every partition, and the row view of the pages.
    kv_gather_state = _kv_gather_constants(sb, slots, page_runs, kv_rem4) if kv_gather else None
    kv_rows = kv_pool.reshape((kv_pool.shape[0] * _BLOCK, _CACHE_WIDTH)) if kv_gather else None
    # ``kv_rem4``: the gather elements are the 2,304-byte pieces of the pages (32 a page).
    kv_pieces = (
        kv_pool.reshape((kv_pool.shape[0] * _REM4_PARTS, _REM4_PIECE)) if (kv_gather or kv_group) and kv_rem4 else None
    )
    # ``kv_group``: the first gathered item and the slots of lead of a group's gather. A gather writes the buffers of the
    # items 2 G before its group, the last of which has its last P@V in the slot after it (the carried ones in slots 0
    # and 1). The gather of the group at item s issues in the page preparation of slot s - lead, which runs in slot
    # s - lead - 1, so it follows those P@V with a lead of at most G - 2: 2 with G 4 or 8 and 0 with G 2, whose groups
    # then start at item 4 (items 2 and 3, on the buffers of the carried slots, keep their SP DMAs).
    kv_first = 2 * kv_group if kv_group == 2 else kv_group
    kv_lead = 0 if kv_group == 2 else 2
    kv_group_state = _kv_group_scratch(sb, slots, kv_group, vog5, kv_first) if kv_group else None

    def kv_group_gather(first, name):
        """``kv_group``: gather the 4 G pages of items ``first`` .. ``first`` + G - 1 (G = ``kv_group``) into their G
        adjacent page buffers with one GpSimd software DGE DMA (128 G pieces of 2,304 bytes, G to each partition, one of
        each item); return it."""
        buffer = first % page_buffers
        column = first - kv_first
        return nisa.dma_copy(
            dst=page_all[0:_P_MAX, buffer : buffer + kv_group, 0:_GROUP_BLOCKS, 0:_CACHE_WIDTH],
            src=kv_pieces.ap(
                pattern=[[_REM4_PIECE, kv_group * _P_MAX], [1, _REM4_PIECE]],
                offset=0,
                vector_offset=kv_group_state["index"][0:_P_MAX, column : column + kv_group],
                indirect_dim=0,
            ),
            dge_mode=nisa.dge_mode.swdge,
            engine=nisa.engine.dma,
            name=f"{tag}_mla_pack_kv_group_{name}_{first}",
        )

    def kv_gather_pair(slot, name):
        """``kv_gather``: gather the 8 pages of items ``slot`` and ``slot`` + 1 into their 2 adjacent page buffers with one
        GpSimd software DGE DMA (DMA batching: 1,024 rows of 576 bytes, 8 to each partition); return it."""
        first = slot % page_buffers
        pair = slot // 2
        if kv_rem4:
            # ``kv_rem4``: 256 pieces of 2,304 bytes, 2 to each partition (one of each item).
            return nisa.dma_copy(
                dst=page_all[0:_P_MAX, first : first + 2, 0:_GROUP_BLOCKS, 0:_CACHE_WIDTH],
                src=kv_pieces.ap(
                    pattern=[[_REM4_PIECE, 2 * _P_MAX], [1, _REM4_PIECE]],
                    offset=0,
                    vector_offset=kv_gather_state["index"][0:_P_MAX, pair * 2 : (pair + 1) * 2],
                    indirect_dim=0,
                ),
                dge_mode=nisa.dge_mode.swdge,
                engine=nisa.engine.dma,
                name=f"{tag}_mla_pack_kv_gather_{name}_{slot}",
            )
        return nisa.dma_copy(
            dst=page_all[0:_P_MAX, first : first + 2, 0:_GROUP_BLOCKS, 0:_CACHE_WIDTH],
            src=kv_rows.ap(
                pattern=[[_CACHE_WIDTH, 2 * _GROUP_BLOCKS * _P_MAX], [1, _CACHE_WIDTH]],
                offset=0,
                vector_offset=kv_gather_state["index"][0:_P_MAX, pair * _KV_PAIR_PAGES : (pair + 1) * _KV_PAIR_PAGES],
                indirect_dim=0,
            ),
            dge_mode=nisa.dge_mode.swdge,
            engine=nisa.engine.dma,
            name=f"{tag}_mla_pack_kv_gather_{name}_{slot}",
        )

    # ``slot_rows_sbuf``: every later slot row also reads the stage tile, as the prologue's rows do.
    loop_stage = stage_keep if slot_rows_sbuf else None

    first_dmas = load_items(item_next, threshold_next, None, "first") or []
    prefetch_thresholds()
    for slot in range(prefetch):
        first_dmas += load_slot(slot, item_next, "first", slot_threshold(thresholds_pf, slot), stage_keep) or []
    if after_writes or p2_stage:
        # The current rows: the pages of the first items can hold rows that this call wrote (the owner writes), which
        # later rows read as prior keys. The first item row and these page loads wait for every write (a semaphore);
        # the SP queue issues every later page load of the loops after them. With ``p2`` they also wait for the store
        # of the slot rows, whose source bytes they take.
        for dma in first_dmas:
            _instruction(dma).depends_on(*after_writes, *p2_stage, kind=nisa.dependency.runtime)

    def clamp_parts(slot):
        """The page clamp of ``slot`` in its 2 parts (pages 0-1, then 2-3): lists of VectorE clamp views, in issue order;
        one view a page, or one view of 2 pages with ``clamp_pin``."""
        rows = page_rows[slot % page_buffers]
        half = _GROUP_BLOCKS // 2
        if clamp_pin:
            return [[rows[0:_P_MAX, block : block + half, 0:_CACHE_WIDTH]] for block in (0, half)]
        views = [_page_view(rows, local_block, pair_pages) for local_block in range(_GROUP_BLOCKS)]
        return [views[:half], views[half:]]

    def clamp_part(slot, part, after=None):
        """Issue part ``part`` of the page clamp of ``slot`` (after ``after``, scheduling-only, if given)."""
        for view in clamp_parts(slot)[part]:
            ins = _sanitize_cache_rows(view)
            if after is not None:
                _instruction(ins).depends_on(after, kind=nisa.dependency.schedule)

    # ``clamp_pin``: the slots whose page clamp waits for the score caps of the current item.
    pinned_clamps = []

    def clamp_slot_pages(slot, defer=False):
        """Clamp the pages of ``slot`` (with zeroed pages, ``zeroed_pages``, every row that a slot reads is finite). With
        ``defer`` and ``clamp_pin`` the clamp issues later, after the score caps of the current item
        (``pinned_clamps``). With ``kv_pitch576`` the loop of the last key groups first writes the mask columns of
        ``slot`` into its mask window (``_mask_window``), from this iteration's threshold of the slot."""
        if kv_pitch576 and clamp_now[0]:
            _write_mask_window(mask_window, slot % page_buffers, mask_offsets, slot_threshold(thresholds, slot))
        if not clamp_pages or not clamp_now[0]:
            return
        if defer and clamp_pin:
            pinned_clamps.append(slot)
            return
        for part in range(len(clamp_parts(slot))):
            clamp_part(slot, part)

    def issue_pinned_clamps(row, cap):
        """``clamp_pin``: after the score cap ``cap`` of row ``row`` of an item, issue part ``row`` of each pinned clamp;
        a clamp leaves ``pinned_clamps`` with its last part (``row`` None: every part left, unpinned)."""
        for slot in list(pinned_clamps):
            for part in range(2) if row is None else ([row] if row < 2 else []):
                clamp_part(slot, part, None if row is None else cap)
            if row is None or row >= 1:
                pinned_clamps.remove(slot)

    def add_to_state(slot, numerator_psum, defer=None, after=None):
        """state[:, b] += [numerator | sum] of ``slot`` through TI: offsets b * 513 + 0..511, then b * 513 + 512. With
        ``defer`` (a list) the sum add is appended to it as a function of the instruction it must follow. With Q > 1 rows
        (``queries``) ``slot`` counts the sub-slots: sub-slot s is row s % Q of item s // Q, which adds to the column of
        stage row s % Q of the slot row of the item. ``after`` (``cap_first``): both adds issue after this instruction
        (scheduling-only dependencies)."""
        slot_row = slot_rows[(slot // subs) % row_sets]
        ti_col = (slot % subs) * sub_width + sub_ti
        group_sum = group_sum_of(slot)
        # A TI index must be a dense tile, so the offsets of the slot row go through a GpSimd copy. ``sum_batch``: the copy
        # of the last row of an item takes the item's batched weight-sum offsets in its sum-offset column.
        ti_index = ti_indices[slot % _KEY_SETS]
        last_row = sum_batch and slot % subs == subs - 1
        nisa.tensor_copy(
            dst=ti_index,
            src=slot_row.view(u16)[0:sub_cols, ti_col : ti_col + _TI_COLS],
            engine=nisa.engine.gpsimd,
        )
        numerator_view = state_flat.indirect(ti_index, num_elem=_KV_LORA)
        sum_view = state_flat.indirect(ti_index[0:sub_cols, _TI_COLS - 1 : _TI_COLS], num_elem=1)
        if sum_batch:
            # The Q weight sums of the item in one TI add of 16 elements: element j < Q of each partition goes to row j's
            # sum column, the zero elements past Q to distinct null-column words. The index slice must start 4-byte aligned
            # (a TI view at an odd uint16 column fails the ISA check), so it reuses the sum-offset column (byte 64).
            sum_view = state_flat.indirect(ti_index[0:sub_cols, _TI_COLS - 1 : _TI_COLS], num_elem=_TI_GROUP)
            group_sum = batch_sums[(slot // subs) % _SUM_BATCH_RING][0:sub_cols, 0:_TI_GROUP]
        # Both adds stay on VectorE: a GpSimd tensor_tensor through TI hangs trn3 hardware.
        numerator_add = nisa.tensor_tensor(
            dst=numerator_view, data1=numerator_view, data2=numerator_psum, op=nl.add, engine=nisa.engine.vector
        )
        if after is not None:
            _instruction(numerator_add).depends_on(after, kind=nisa.dependency.schedule)

        def sum_add(after=None):
            ins = nisa.tensor_tensor(
                dst=sum_view, data1=sum_view, data2=group_sum, op=nl.add, engine=nisa.engine.vector
            )
            if after is not None:
                handle = ins if hasattr(ins, "depends_on") else nisa.NkiInstruction(ins)
                handle.depends_on(after, kind=nisa.dependency.schedule)

        if sum_batch and not last_row:
            return
        if defer is None:
            sum_add(after)
        else:
            defer.append(sum_add)

    def weights_parts(slot, engine=None):
        """The weight transposes of ``slot`` (``bg_interleave``) and their exact BF16 copy, as functions that each trace
        one instruction and return it: a list of the 4 transposes, and the copy. ``engine`` names the engine of the
        transposes (``rope_ahead``: TensorE); None leaves it unset, as the default program has it."""
        if banks is None:
            weights_psum = nl.ndarray((_P_MAX, _GROUP_BLOCKS, sub_cols), dtype=bf16, buffer=nl.psum)
        else:
            weights_psum = banks["weights"]
        # ``p2`` traces its transposes on TensorE (engine= on every op it adds); the default program leaves it unset.
        transpose_engine = {} if engine is None and not p2 else {"engine": engine or nisa.engine.tensor}

        def transpose(local_block):
            return nisa.nc_transpose(
                dst=weights_psum[0:_P_MAX, local_block, 0:sub_cols],
                data=weights_bufs[slot % weight_ring][0:sub_cols, local_block * _BLOCK : (local_block + 1) * _BLOCK],
                **transpose_engine,
            )

        def copy():
            # The exact BF16 copy on ScalarE, as in _attend_group.
            return nisa.tensor_copy(
                dst=weights_t_bufs[slot % 2],
                src=weights_psum if banks is None else weights_psum[0:_P_MAX, 0:_GROUP_BLOCKS, 0:sub_cols],
                engine=nisa.engine.scalar,
            )

        return [functools.partial(transpose, local_block) for local_block in range(_GROUP_BLOCKS)], copy

    def transpose_weights(slot, after):
        """The weight transposes of ``slot`` (``bg_interleave``) and their exact BF16 copy; return the last transpose."""
        transposes, copy = weights_parts(slot)
        for local_block, transpose in enumerate(transposes):
            ins = transpose()
            if after is not None and local_block == 0:
                ins.depends_on(after, kind=nisa.dependency.schedule)
        copy()
        return ins

    def numerator_parts(slot, defer=None):
        """P@V of ``slot`` (``bg_interleave``) and its state adds (``defer``: ``add_to_state``), as a list of 4 functions
        that each trace one P@V matmul and return it, and a function that traces the state adds. With Q > 1 rows ``slot``
        counts the sub-slots, which read the pages of their item."""
        rows = page_rows[(slot // subs) % page_buffers]
        if numerator_bufs is not None:
            numerator_psum = numerator_bufs[slot % numerator_ring]
        elif banks is None:
            numerator_psum = nl.ndarray((sub_cols, _KV_LORA), dtype=fp32, buffer=nl.psum)
        else:
            numerator_psum = banks["numerator"][slot % 2]

        def pv(local_block):
            return nisa.nc_matmul(
                dst=numerator_psum,
                stationary=weights_t_bufs[slot % 2][0:_P_MAX, local_block, 0:sub_cols],
                moving=rows[0:_P_MAX, local_block, 0:_KV_LORA],
                accumulate=local_block > 0,
            )

        return [functools.partial(pv, local_block) for local_block in range(_GROUP_BLOCKS)], functools.partial(
            add_to_state, slot, numerator_psum, defer
        )

    # ``cap_first``: the numerator PSUM buffers whose state adds are not traced yet (buffer -> sub-slot).
    pending_numerators = {}

    def numerator(slot, after=None, defer=None, rope_item=None, adds_to=None, rope_out=None):
        """P@V of ``slot`` (``bg_interleave``) and its state adds (``defer``: ``add_to_state``); return the last P@V
        matmul. With ``rope_item`` (``item_rope_ahead``) the rope key transposes of item ``rope_item`` run in the P@V
        gaps (each transpose, then a P@V matmul; a chain of scheduling-only dependencies after ``after``) and their
        ScalarE copy follows (``rope_out``, a list: ``act_spread`` takes the copy). With ``adds_to`` (a list,
        ``cap_first``) the state adds are appended to it, for the caller to trace later."""
        if numerator_bufs is not None:
            kernel_assert(
                ("ring", slot % numerator_ring) not in pending_numerators,
                f"cap_first: the P@V of sub-slot {slot} writes the numerator ring buffer whose state adds are not "
                "traced yet",
            )
        if banks is not None:
            kernel_assert(
                slot % 2 not in pending_numerators,
                f"cap_first: the P@V of sub-slot {slot} writes the numerator PSUM whose state adds of sub-slot "
                f"{pending_numerators.get(slot % 2)} are not traced yet",
            )
        # ``cap_first``: the sum add follows the numerator add, so it reads the weight sum before the exp of sub-slot
        # s + 4 rewrites it (group sums ring of 4) whatever the key copies of the sub-slots between.
        matmuls, adds = numerator_parts(slot, None if adds_to is not None else defer)
        if adds_to is None:
            state_adds = adds
        else:

            def state_adds():
                if banks is not None:
                    pending_numerators[slot % 2] = slot
                if numerator_bufs is not None:
                    pending_numerators[("ring", slot % numerator_ring)] = slot

                def run(after=None):
                    pending_numerators.pop(slot % 2, None)
                    if numerator_bufs is not None:
                        pending_numerators.pop(("ring", slot % numerator_ring), None)
                    adds(after=after)

                adds_to.append(run)

        if rope_item is not None:
            rope_transposes, rope_copy = rope_parts(rope_item)
            last = after
            for matmul, transpose in zip(matmuls, rope_transposes):
                last = _pinned(transpose(), last)
                last = pv = _pinned(matmul(), last)
            copy_ins = rope_copy()
            if rope_out is not None:
                rope_out.append(copy_ins)
            state_adds()
            return pv
        for local_block, matmul in enumerate(matmuls):
            pv = matmul()
            if after is not None and local_block == 0:
                pv.depends_on(after, kind=nisa.dependency.schedule)
        state_adds()
        return pv

    def rope_parts(slot):
        """``rope_ahead``: the rope key transposes and copy of ``slot`` (``_rope_mode_parts``). With ``kv_pitch576`` the
        loop traced now sets the rows: the 64 rope rows, and in the loop of the last key groups the Q bias rows from the
        mask window of ``slot``."""
        window = mask_window_of(slot)
        return _rope_mode_parts(
            page_rows[slot % page_buffers],
            key_sets[slot % _KEY_SETS][1],
            rope_identity,
            _GROUP_BLOCKS,
            rope_width if masked_now() else _ROPE,
            None if banks is None else (banks["rope"], slot % 2),
            mask_window=window,
        )

    def dr_slot_keys(rows, latent_keys, slot):
        """``dr_keys``: the 2-byte key words of both blocks of the pages ``rows`` of ``slot`` and their 2 copies into
        ``latent_keys`` (the key-word bank of the slot's parity)."""
        words = banks["keys"][slot % 2]
        last = None
        for block in range(_DR_BLOCKS):
            last = _dr_key_words(rows, words, key16_identity, block, after=last)
        _dr_key_copies(latent_keys, words, after=last)

    def dr_keys_hook(keys, block, qk, deferred_sums, copy_after=None):
        """``dr_keys``: after double_row matmul ``block`` of a slot (``qk``), the 2-byte key transposes of block ``block``
        of the next slot ``keys`` (page tile, key words, slot), and after the last block their ScalarE copy (after
        ``copy_after`` if given), then (``deferred_sums``, ``sum_after_copy``) the deferred sum adds; return the last
        transpose."""
        if keys is None:
            return None
        rows, latent_keys, next_slot = keys
        words = banks["keys"][next_slot % 2]
        last = _dr_key_words(rows, words, key16_identity, block, after=qk)
        if block == _DR_BLOCKS - 1:
            _dr_key_copies(latent_keys, words, after=last, act_after=copy_after)
            # No VectorE key copy for the sum adds to follow: they issue here, in slot order.
            while deferred_sums:
                deferred_sums.pop(0)()
        return last

    def word_permute(rows, slot, block):
        """``word_keys``: the ScalarE byte permute of the latent bytes of page ``block`` of the page tile ``rows`` of
        ``slot`` into the permute buffer of its parity: byte ``4c + t`` of the permuted row is feature ``128 t + c``."""
        pitch = rows.shape[2]
        # ``kv_gather``: the page buffers are views of one tile, whose partition step spans all of them.
        base, step, first = rows, _GROUP_BLOCKS * pitch, 0
        if page_all is not None:
            base, step, first = (
                page_all,
                page_buffers * _GROUP_BLOCKS * pitch,
                (slot % page_buffers) * _GROUP_BLOCKS * pitch,
            )
        return nisa.tensor_copy(
            dst=word_perm[slot % 2].ap(
                pattern=[[_GROUP_BLOCKS * _KV_LORA, _P_MAX], [_KV_LORA_TILES, _P_MAX], [1, _KV_LORA_TILES]],
                offset=block * _KV_LORA,
                dtype=nl.uint8,
            ),
            src=base.ap(
                pattern=[[step, _P_MAX], [1, _P_MAX], [_P_MAX, _KV_LORA_TILES]],
                offset=first + block * pitch,
                dtype=nl.uint8,
            ),
            engine=nisa.engine.scalar,
        )

    def word_transpose(slot, block, after=None, rows=None):
        """``word_keys``: the 32-bit TensorE transpose of the permuted page ``block`` of ``slot`` into the key-word bank of
        its parity (after ``after`` if given; scheduling-only). ``mx_natural``: of the latent words of page ``block`` of
        the page tile ``rows`` itself (word ``c`` of a row: features ``4c .. 4c + 3``)."""
        if natural:
            kernel_assert(rows.shape[2] % _MX_PACK == 0, "pack mx_natural needs page rows of whole 32-bit words")
            data = rows.view(fp32)[0:_P_MAX, block, 0 : _KV_LORA // _MX_PACK]
        else:
            data = word_perm[slot % 2].view(fp32)[0:_P_MAX, block, 0 : _KV_LORA // _KV_LORA_TILES]
        return _pinned(
            nisa.nc_transpose(
                dst=banks["keys"][slot % 2][0:_P_MAX, block * _BLOCK : (block + 1) * _BLOCK],
                data=data,
                engine=nisa.engine.tensor,
            ),
            after,
        )

    def word_copy(latent_keys, slot, after, act_after=None):
        """``word_keys``: the exact ScalarE copy of the key words of ``slot`` (through a uint16 view: an FP32 copy could
        flush a subnormal word) into ``latent_keys``, after the last transpose ``after`` and ``act_after`` if given."""
        copy_ins = _pinned(
            nisa.tensor_copy(
                dst=latent_keys.view(u16)[0:_P_MAX, 0 : 2 * group_keys],
                src=banks["keys"][slot % 2].view(u16)[0:_P_MAX, 0 : 2 * group_keys],
                engine=nisa.engine.scalar,
            ),
            after,
        )
        if act_after is not None:
            _pinned(copy_ins, act_after)
        return copy_ins

    def word_slot_keys(rows, latent_keys, slot):
        """``word_keys``: all key words of ``slot`` (permutes, transposes, copy); return its SBUF writes."""
        writes = [] if natural else [word_permute(rows, slot, block) for block in range(_GROUP_BLOCKS)]
        last = None
        for block in range(_GROUP_BLOCKS):
            last = word_transpose(slot, block, after=last, rows=rows)
        writes.append(word_copy(latent_keys, slot, last))
        return writes

    def word_keys_hook(keys, latent_tile, qk, deferred_sums, copy_after=None, copy_out=None):
        """``word_keys``: after latent QK ``latent_tile`` of the last sub-slot of an item (``qk``), the 32-bit transpose of
        page ``latent_tile`` of the next item ``keys`` (page tile, key words, slot), and after the last one their ScalarE
        copy (after ``copy_after`` if given; with ``copy_out``, a list, a function that traces it is appended there for
        the caller, ``act_spread``), then the deferred sum adds (no VectorE key copy for them to follow); return the
        transpose."""
        if keys is None:
            return None
        rows, latent_keys, next_slot = keys
        last = word_transpose(next_slot, latent_tile, after=qk, rows=rows)
        if latent_tile == _KV_LORA_TILES - 1:
            copy = functools.partial(word_copy, latent_keys, next_slot, last, act_after=copy_after)
            if copy_out is None:
                copy()
            else:
                copy_out.append(copy)
            while deferred_sums:
                deferred_sums.pop(0)()
        return last

    def word_mx_hook(keys, qk, deferred_sums, copy_after=None, copy_out=None):
        """``mx_qk``: after the one MX matmul ``qk`` of the last sub-slot of an item, the 32-bit transposes of the 4 pages
        of the next item ``keys`` in turn and their ScalarE copy (``word_keys_hook`` of each page; the copy after
        ``copy_after`` if given, or to ``copy_out``), then the deferred sum adds; return the last transpose."""
        last = qk
        for latent_tile in range(_KV_LORA_TILES):
            last = word_keys_hook(
                keys,
                latent_tile,
                last,
                deferred_sums,
                copy_after if latent_tile == _KV_LORA_TILES - 1 else None,
                copy_out,
            )
        return last

    # ``word_keys``: the sub-slot (row) of an item that permutes page ``block`` of the next item: the last 4 rows, after
    # the pinned page clamp of the next item (rows 0 and 1) where it runs.
    # ``keys_ahead``: the sub-slot (row) of an item that transposes and copies the next item's key words (G - 2, else
    # the last one), with the permutes of its pages in the rows up to it.
    key_row = subs - 2 if word_keys and keys_ahead and subs >= 6 else subs - 1
    word_rows = [
        min(key_row, max(2 if clamp_pin else 0, key_row + 1 - _GROUP_BLOCKS + block)) for block in range(_GROUP_BLOCKS)
    ]
    # ``act_spread``: the rows of the byte permutes of the next item's pages where the loop does not clamp its pages, one
    # a row and two in the last (each traced after its row's exp, so each must lie before the key-word row);
    # ``mx_natural`` has no byte permutes to place.
    spread_rows = [0, 1, 2, 2] if act_spread and not natural and key_row == subs - 2 and subs >= 6 else None

    if carry:
        # Slot 0 of the first iteration (of whichever loop runs first) finds its latent keys transposed, as every
        # later slot 0 does from the slot before it.
        clamp_slot_pages(0)
        if mx_psum is not None:
            (_mx_key_words if queries == 1 else _mx_item_key_words)(page_rows[0], key_sets[0][0], mx_psum["keys"][0])
        if dr_keys:
            dr_slot_keys(page_rows[0], key_sets[0][0], 0)
        prologue_copies = []
        if word_keys:
            word_writes = word_slot_keys(page_rows[0], key_sets[0][0], 0)
            if p2_stage:
                prologue_copies.extend(word_writes)
        for latent_tile in range(0 if banks is not None else _KV_LORA_TILES):
            _transpose_mode_tile(
                page_rows[0],
                key_sets[0][0],
                latent_tile,
                _GROUP_BLOCKS,
                transpose_identity,
                latent_act,
                **({"copy_out": prologue_copies} if p2_stage else {}),
            )
        if rope_ahead or item_rope_ahead:
            # ... and its rope keys, as every later slot does from the slot before it. With ``p2`` the copy is a first
            # write into the loop buffers, so it waits for the store of the slot rows too.
            transposes, copy = rope_parts(0)
            for transpose in transposes:
                transpose()
            prologue_copies.append(copy())
        # Carried slots -1 and -2 of the first iteration are a null slot: zero weights on zero pages, added with a zero
        # weight sum to the state column of the null items (B, row B of the TI offsets).
        if deep_pipe:
            # ``deep_pipe``: sub-slots -3 .. -1 are null; the transposes of -2 and -1 read the weight ring, P@V of -3 the
            # transposed weights of the other parity.
            prologue_writes = [
                nisa.memset(dst=weights_bufs[-2 % weight_ring], value=0.0, engine=nisa.engine.vector),
                nisa.memset(dst=weights_bufs[-1 % weight_ring], value=0.0, engine=nisa.engine.vector),
                nisa.memset(dst=weights_t_bufs[-3 % 2], value=0.0, engine=nisa.engine.vector),
            ]
        else:
            prologue_writes = [
                nisa.memset(dst=weights_bufs[1], value=0.0),
                nisa.memset(dst=weights_t_bufs[0], value=0.0),
            ]
        if queries == 1:
            for slot in (-2, -1):
                nisa.memset(dst=page_rows[slot % page_buffers], value=0.0)
                nisa.memset(dst=key_sets[slot % _KEY_SETS][2], value=0.0)
                nisa.tensor_copy(
                    dst=slot_rows[slot % row_sets].view(u16)[0:_HEADS, _STAGE_TI : _STAGE_TI + _TI_COLS],
                    src=plan_ti[0:_HEADS, tokens, 0:_TI_COLS],
                )
        else:
            # Q > 1 rows: carried sub-slots -2 and -1 are rows Q - 2 and Q - 1 of a null item -1 (one page tile).
            prologue_writes.append(nisa.memset(dst=page_rows[-1 % page_buffers], value=0.0, engine=nisa.engine.gpsimd))
            for slot in (-3, -2, -1) if deep_pipe else (-2, -1):
                prologue_writes.append(nisa.memset(dst=group_sum_of(slot), value=0.0, engine=nisa.engine.gpsimd))
                ti_col = (slot % subs) * sub_width + sub_ti
                prologue_writes.append(
                    nisa.tensor_copy(
                        dst=slot_rows[(slot // subs) % row_sets].view(u16)[0:sub_cols, ti_col : ti_col + _TI_COLS],
                        src=plan_ti[0:sub_cols, columns, 0:_TI_COLS],
                        engine=nisa.engine.gpsimd,
                    )
                )
            if sum_batch:
                # ``sum_batch``: the null item -1 adds zero sums to the null column with its last row (sub-slot -1), and
                # the columns past Q of both sum tiles stay zero.
                fill_engine = nisa.engine.vector if setup_off_pool else nisa.engine.gpsimd
                for batch_sum in batch_sums:
                    prologue_writes.append(nisa.memset(dst=batch_sum, value=0.0, engine=fill_engine))
                sum_col = (subs - 1) * sub_width + sub_ti + _TI_COLS - 1
                if setup_off_pool:
                    # ``setup_off_pool``: the null-row offsets that the passes share.
                    null_offsets = _shared(
                        shared,
                        ("p2_null_offsets", subs, tokens, sub_cols),
                        lambda: _shared_sum_offsets(subs, tokens, sb, pack_sb, null_only=True, parts=sub_cols),
                    )
                else:
                    null_offsets = _sum_batch_offsets(subs, tokens, sb, null_only=True, parts=sub_cols)
                prologue_writes.append(
                    nisa.tensor_copy(
                        dst=slot_rows[-1 % row_sets].view(u16)[0:sub_cols, sum_col : sum_col + 1],
                        src=null_offsets[0:sub_cols, 0:1],
                        engine=nisa.engine.vector,
                    )
                )
        if p2_stage:
            # ``p2``: the first writes of every engine into the loop buffers wait for the store of the slot rows.
            for write in prologue_copies + prologue_writes:
                _instruction(write).depends_on(*p2_stage, kind=nisa.dependency.runtime)

    def next_keys_hook(keys, latent_tile, qk, deferred_sums, copy_after=None):
        """``sum_after_copy``: the latent key transposes of tile ``latent_tile`` of the next slot after QK ``qk``, then
        (tile 0, whose copy runs on VectorE) the deferred sum adds after that copy; return the last transpose. The copy
        issues after ``copy_after`` if given (scheduling-only)."""
        if keys is None:
            return None
        copies = []
        last = _transpose_mode_tile(
            keys[0],
            keys[1],
            latent_tile,
            _GROUP_BLOCKS,
            transpose_identity,
            latent_act,
            after=qk,
            copy_out=copies,
            copy_after=copy_after,
        )
        if latent_tile == 0:
            while deferred_sums:
                deferred_sums.pop(0)(copies[0])
        return last

    def rope_ahead_tail(slot, qk_rope, softmax_ops, defer):
        """``rope_ahead``: the work of ``slot`` after its rope matmul ``qk_rope``. TensorE: P@V of slot j - 2 with the
        rope key transposes of slot j + 1 and the weight transposes of slot j - 1 in its gaps (2 rope transposes, P@V, 2
        rope transposes, P@V, 2 weight transposes, P@V, 2 weight transposes, P@V; a chain of scheduling-only
        dependencies). Then the ScalarE rope key copy of slot j + 1, the score cap and the exp of slot j (the exp after
        that copy, scheduling-only), the ScalarE weight copy of slot j - 1 and the state adds of slot j - 2 (``defer``).
        The rope transposes go first, so the copy fits before the exp; the weight copy, which P@V of slot j - 1 reads,
        follows the exp. The rope transposes read the pages of slot j + 1 (slot 0 of the next iteration in the last
        slot), which slot j loads. Return the last P@V matmul and the weight copy."""
        weight_transposes, weights_copy = weights_parts(slot - 1, engine=nisa.engine.tensor)
        matmuls, state_adds = numerator_parts(slot - 2, defer)
        rope_transposes, rope_copy = rope_parts(slot + 1)
        order = (
            rope_transposes[0:2]
            + matmuls[0:1]
            + rope_transposes[2:4]
            + matmuls[1:2]
            + weight_transposes[0:2]
            + matmuls[2:3]
            + weight_transposes[2:4]
            + matmuls[3:4]
        )
        last = qk_rope
        for trace in order:
            last = _pinned(trace(), last)
        if dr_keys:
            # The weight copy of slot j - 1 (after its transposes, early in the P@V gaps) goes before the exp of slot j,
            # so P@V of slot j - 1 in slot j + 1 does not wait for the exp chain of slot j.
            rope_copy_ins = rope_copy()
            copy = _pinned(weights_copy(), rope_copy_ins)
            softmax_ops(after=copy)
        else:
            softmax_ops(after=rope_copy())
            copy = weights_copy()
        state_adds()
        return last, copy

    def mx_keys_hook(keys, qk, deferred_sums, copies):
        """``mx_keys``: the 32-bit key transposes and copies of the next slot ``keys`` (page tile, key words, slot)
        after the MX matmul ``qk`` (the copies go to ``copies``, ScalarE first), then (``sum_after_copy``) the deferred
        sum adds after the VectorE copy; return the last transpose."""
        if keys is None:
            return None
        rows, latent_keys, next_slot = keys
        last = _mx_key_words(rows, latent_keys, mx_psum["keys"][next_slot % 2], after=qk, copy_out=copies)
        while deferred_sums:
            deferred_sums.pop(0)(copies[-1])
        return last

    def mx_item_hook(keys, qk, deferred_sums, copy_after=None):
        """``mx_keys`` with Q > 1 rows: after the MX matmul ``qk`` of the last sub-slot of an item, the 32-bit key
        transposes of the next item ``keys`` (page tile, key words, slot) into the key-word bank of its parity and their
        ScalarE copy (after ``copy_after`` if given; ``_mx_item_key_words``), then (``sum_after_copy``) the deferred sum
        adds; return the last transpose."""
        if keys is None:
            return None
        rows, latent_keys, next_slot = keys
        last = _mx_item_key_words(rows, latent_keys, mx_psum["keys"][next_slot % 2], after=qk, act_after=copy_after)
        while deferred_sums:
            deferred_sums.pop(0)()
        return last

    # ``deep_pipe``: the weight transposes trail QK by 2 sub-slots (else 1), P@V by 3 (else 2).
    t_lag = 2 if deep_pipe else 1

    def drain_tail(end):
        """``deep_pipe``: the transposes of sub-slots end - 2 and end - 1 and P@V of end - 3 .. end - 1 (interleaved, so
        each P@V reads its transposed weights before the next transposes rewrite their parity)."""
        transpose_weights(end - 2, None)
        numerator(end - 3)
        transpose_weights(end - 1, None)
        numerator(end - 2)
        numerator(end - 1)

    def interleaved_slots(count, prepare_pages, slot_softmax_of, load_next_first, carried, prepare_ahead=None):
        """The slots of one iteration with ``bg_interleave``, in a program order that keeps the compiler's 2-deep
        pipeline: in slot j the latent key transposes of slot j + 1 run between the QK matmuls of slot j (group k after
        QK k, QK k + 1 and the rope matmul after group k; scheduling-only dependencies), then the weight transposes of
        slot j - 1 (after the rope matmul of slot j) and P@V of slot j - 2 with its state adds. The state adds keep the
        slot order, so the state is bitwise the same.

        P@V of slot j - 2 is pinned after the weight transposes of slot j - 1, and QK of slot j + 1 after that P@V
        (scheduling-only). The scores of a slot live in one PSUM tile, so QK of slot j + 1 waits for the exp of slot
        j, and TensorE needs this P@V work to cover the score cap and exp of slot j. The transpose-mode rope keys of
        slot j + 1 issue after that P@V too: hoisted further, they wait on the ScalarE rope copy of an earlier slot.

        With ``carry`` the pipeline runs across the fence: the last slot loads (``load_next_first``) and transposes
        the pages of the next iteration's slot 0, and slots 0 and 1 run the weight transposes and P@V of slots -1
        and -2, the last slots of the iteration before (the buffer rings repeat every iteration), with no drain.

        With ``lookahead`` 2 slot j calls ``prepare_ahead`` (the loads of slot j + 2, or of the next iteration's first
        two slots in the last two slots) in place of ``prepare_pages(j + 1)`` and ``load_next_first``.

        With ``sum_after_copy`` the weight sum add of slot j - 2 issues after the VectorE copy of latent key tile 0 of
        slot j + 2 in the hook of slot j + 1, so that copy does not queue behind it; the last one of the iteration
        issues at its end. The sum adds keep the slot order."""
        last_pv = None
        deferred_sums = [] if sum_after_copy else None
        # ``cap_first``: the state adds of the last numerator, traced after the next sub-slot's score cap, and that cap.
        pending_adds = [] if cap_first else None
        caps = [] if cap_first or clamp_pin else None
        # ``rope_ahead``: the weight copy of slot j - 1, after which the first ScalarE latent key copy of slot j + 1's hooks
        # issues (scheduling-only), so the scheduler does not queue the weight copy behind them.
        last_weights_copy = None
        first_act_tile = min(_LATENT_ACT_TILES[latent_act])
        # ``act_spread``, in a loop of at least ``_ACT_SPREAD_ITEMS`` items (a shorter one, the block of the last key groups,
        # keeps the order without act_spread: its pipeline start, with the pins, costs more than it gains): the ScalarE work of the last
        # sub-slot after its exp (byte permutes, key-word copy, rope key copy), after which the next exp issues; the byte
        # permute rows of this loop (the rows without act_spread where it clamps its pages).
        spread = act_spread and count >= _ACT_SPREAD_ITEMS
        act_extras = []
        permute_rows = (
            spread_rows if spread and spread_rows is not None and not (clamp_pages and clamp_now[0]) else None
        )

        for slot in range(count):
            rows = page_rows[slot % page_buffers]
            latent_keys, rope_keys, group_sum, _ = key_sets[slot % _KEY_SETS]
            slot_row = slot_rows[slot % row_sets]
            # Slot j > 0 has its pages and latent keys from slot j - 1 (with ``carry`` slot 0 too); with ``rope_ahead``
            # its rope keys too.
            if slot == 0 and not carried:
                prepare_pages(0)
                if mx_psum is not None:
                    (_mx_key_words if queries == 1 else _mx_item_key_words)(rows, latent_keys, mx_psum["keys"][0])
                if dr_keys:
                    dr_slot_keys(rows, latent_keys, 0)
                if word_keys:
                    word_slot_keys(rows, latent_keys, 0)
            # ``item_rope_ahead``: an item after the first one of a loop that does not carry has its rope keys from the
            # item before it (slot 0 of a carried iteration from the prologue or the iteration before).
            if not rope_ahead and not (item_rope_ahead and (slot > 0 or carried)):
                _transpose_group(
                    rows,
                    latent_keys,
                    rope_keys,
                    slot_offset,
                    thresholds[_BIAS_ROW : _BIAS_ROW + 1, slot : slot + 1],
                    transpose_identity=transpose_identity,
                    latent_act=latent_act,
                    mask_column=mask_column and masked_now(),
                    rope_identity=rope_identity,
                    skip_latent=carried or slot > 0 or banks is not None,
                    rope_after=last_pv,
                    mask_rows=queries,
                    **({} if banks is None else {"rope_psum": (banks["rope"], slot % 2)}),
                    **({"mask_window": mask_window_of(slot)} if kv_pitch576 else {}),
                    **({"mask_engine": nisa.engine.gpsimd} if mask_pool else {}),
                )
            next_keys = None
            if lookahead > 1:
                prepare_ahead(slot)
            if slot + 1 < count:
                if lookahead == 1:
                    prepare_pages(slot + 1, defer=True)
                next_keys = (page_rows[(slot + 1) % page_buffers], key_sets[(slot + 1) % _KEY_SETS][0])
            elif carried:
                if lookahead == 1:
                    load_next_first(defer=True)
                next_keys = (page_rows[0], key_sets[0][0])
            # ``dr_keys``: the next slot's key words go to the key-word bank of its parity (slot 0 of the next iteration:
            # bank 0, as the slot counts are even).
            dr_next = None if next_keys is None else (*next_keys, slot + 1 if slot + 1 < count else 0)
            # ``item_rope_ahead``: the item whose rope keys this item transposes (the next one, or slot 0 of the next
            # iteration in the last slot of a carried iteration).
            ahead_item = None
            if item_rope_ahead and next_keys is not None:
                ahead_item = slot + 1 if slot + 1 < count else 0
            if mx_psum is not None and queries == 1:
                copies = []
                slot_f = slot_row.view(fp32)
                parts = 2 if mx_keys == "split" else 1
                qk_rope = _slot_scores(
                    latent_keys,
                    rope_keys if rope_identity is None else _fp8_word_rope(rope_keys),
                    None,
                    slot_row[0 : _BIAS_ROW + 1, q_width : q_width + _HEADS],
                    slot_softmax_of(slot_row),
                    group_sum,
                    weights_bufs[slot % 2],
                    lambda latent_tile, qk, keys=next_keys and (*next_keys, slot + 1), copies=copies: mx_keys_hook(
                        keys, qk, deferred_sums if sum_after_copy else [], copies
                    ),
                    first_after=last_pv,
                    scores=scores_bufs[slot % 2],
                    mx=([slot_f[0:_P_MAX, p * _HEADS : (p + 1) * _HEADS] for p in range(parts)], mx_scale, copies),
                )
                last_pv = None
                if slot >= 1 or carried:
                    last_weights = transpose_weights(slot - 1, qk_rope)
                if slot >= 2 or carried:
                    last_pv = numerator(slot - 2, last_weights, deferred_sums)
                continue
            # Sub-slot s = Q slot + row runs row ``row`` of the item (Q = 1: the slot); the latent keys of the next item
            # transpose in the QK gaps of the last sub-slot.
            for row in range(subs):
                sub = slot * subs + row
                base = row * sub_width
                hook_keys = next_keys if row == subs - 1 else None
                dr_hook_keys = dr_next if row == (key_row if word_keys else subs - 1) else None
                if word_keys and not natural and dr_next is not None and permute_rows is None:
                    for block in range(_GROUP_BLOCKS):
                        if word_rows[block] == row:
                            word_permute(dr_next[0], dr_next[2], block)
                # ``act_spread``: the key-word copy of the hook, traced after this sub-slot's exp.
                word_copies = [] if spread else None
                scores_out = _slot_scores(
                    _LaneKeys(latent_keys)
                    if word_keys and not mx_qk
                    else latent_keys
                    if dr_keys or mx_psum is not None or mx_qk
                    else _fp8_word_keys(latent_keys),
                    rope_keys if rope_identity is None else _fp8_word_rope(rope_keys),
                    None
                    if dr_keys or mx_psum is not None or mx_qk
                    else [
                        slot_row[0:_P_MAX, base + t * sub_cols : base + (t + 1) * sub_cols]
                        for t in range(_KV_LORA_TILES)
                    ],
                    slot_row[
                        0 : qk_rope_rows_now(),
                        base + _KV_LORA_TILES * sub_cols : base + (_KV_LORA_TILES + 1) * sub_cols,
                    ],
                    slot_softmax_of(slot_row, row),
                    sum_dst_of(sub),
                    weights_bufs[sub % weight_ring],
                    (
                        (
                            lambda latent_tile, qk, keys=dr_hook_keys, wc=last_weights_copy: mx_item_hook(
                                keys, qk, deferred_sums if sum_after_copy else [], wc
                            )
                        )
                        if mx_psum is not None
                        else (
                            lambda latent_tile, qk, keys=dr_hook_keys, wc=last_weights_copy: dr_keys_hook(
                                keys,
                                latent_tile,
                                qk,
                                deferred_sums if sum_after_copy else [],
                                wc if latent_tile == _DR_BLOCKS - 1 else None,
                            )
                        )
                        if dr_keys
                        else (
                            lambda latent_tile, qk, keys=dr_hook_keys, wc=last_weights_copy, wcs=word_copies: (
                                word_mx_hook(keys, qk, deferred_sums if sum_after_copy else [], wc, wcs)
                            )
                        )
                        if mx_qk
                        else (
                            lambda latent_tile, qk, keys=dr_hook_keys, wc=last_weights_copy, wcs=word_copies: (
                                word_keys_hook(
                                    keys,
                                    latent_tile,
                                    qk,
                                    deferred_sums if sum_after_copy else [],
                                    wc if latent_tile == _KV_LORA_TILES - 1 else None,
                                    wcs,
                                )
                            )
                        )
                        if word_keys
                        else (
                            lambda latent_tile, qk, keys=hook_keys, wc=last_weights_copy: next_keys_hook(
                                keys, latent_tile, qk, deferred_sums, wc if latent_tile == first_act_tile else None
                            )
                        )
                        if sum_after_copy
                        else lambda latent_tile, qk, keys=hook_keys, wc=last_weights_copy: (
                            None
                            if keys is None
                            else _transpose_mode_tile(
                                keys[0],
                                keys[1],
                                latent_tile,
                                _GROUP_BLOCKS,
                                transpose_identity,
                                latent_act,
                                after=qk,
                                copy_after=wc if latent_tile == first_act_tile else None,
                            )
                        )
                    ),
                    first_after=last_pv,
                    scores=scores_bufs[sub % 2],
                    softmax_later=rope_ahead or spread,
                    rope_rows=qk_rope_rows_now(),
                    **({"capless": True} if capless else {}),
                    **(
                        {"dr": [_dr_stationary(slot_row, byte, base, sub_cols) for byte in range(2)]} if dr_keys else {}
                    ),
                    # ``mx_keys``: the MX query words of row ``row`` (the first 96 FP32 words of its stage row); the hook
                    # issues the next item's key copy, so the softmax takes no pins on it.
                    **(
                        {"mx": ([slot_row.view(fp32)[0:_P_MAX, base // 2 : base // 2 + _HEADS]], mx_scale, [])}
                        if mx_psum is not None
                        else {}
                    ),
                    # ``mx_qk``: the MX query words of the sub-slot's 128 columns (the first 128 FP32 words of its stage
                    # row), their scales ("block32": the 128 scale bytes of its stage row) and the unit key scales.
                    **(
                        {
                            "mx_qk": (
                                slot_row.view(fp32)[0:_P_MAX, base // 2 : base // 2 + sub_cols],
                                mx_scale,
                                slot_row.view(nl.uint8)[
                                    0:_P_MAX,
                                    2 * (base + _P2_STAGE_MX_SCALE) : 2 * (base + _P2_STAGE_MX_SCALE) + sub_cols,
                                ]
                                if mx_qk == "block32"
                                else None,
                            )
                        }
                        if mx_qk
                        else {}
                    ),
                    **({"cap_out": caps} if caps is not None else {}),
                )
                if rope_ahead:
                    last_pv, last_weights_copy = rope_ahead_tail(slot, *scores_out, deferred_sums)
                    continue
                exp = None
                if spread:
                    # The weight transposes and copy of sub-slot s - 1 (after the rope matmul), then the score cap and the
                    # exp of sub-slot s, the exp after that copy and after the last sub-slot's other ScalarE work; then
                    # this sub-slot's own (the key-word copy, the byte permutes of its rows), each after the exp.
                    qk_rope, softmax_ops = scores_out
                    weights_copy = None
                    if sub >= t_lag or carried:
                        transposes, weights_copy_of = weights_parts(sub - t_lag)
                        for local_block, transpose in enumerate(transposes):
                            last_weights = transpose()
                            if local_block == 0:
                                last_weights.depends_on(qk_rope, kind=nisa.dependency.schedule)
                        weights_copy = weights_copy_of()
                    exp = softmax_ops(after=weights_copy)
                    for extra in act_extras:
                        _pinned(exp, extra)
                    act_extras = [_pinned(trace(), exp) for trace in word_copies]
                    if permute_rows is not None and dr_next is not None:
                        for block in range(_GROUP_BLOCKS):
                            if permute_rows[block] == row:
                                act_extras.append(_pinned(word_permute(dr_next[0], dr_next[2], block), exp))
                else:
                    qk_rope = scores_out
                # ``cap_first``: the state adds of sub-slot s - 3 follow the score cap of sub-slot s in VectorE order.
                cap = caps.pop() if caps else None
                while pending_adds:
                    pending_adds.pop(0)(cap)
                # ``clamp_pin``: the page clamp of the next item follows the score caps of rows 0 and 1 of this one.
                if pinned_clamps:
                    issue_pinned_clamps(row, cap)
                if queries > 1 and (dr_hook_keys if word_keys else hook_keys) is None:
                    # No key copy in this sub-slot: the deferred sum adds issue here, so each sum add of sub-slot s
                    # issues before the exp of sub-slot s + 4 rewrites its weight sum (group sums ring of 4).
                    while deferred_sums:
                        deferred_sums.pop(0)()
                last_pv = None
                # ``deep_pipe``: the transposes of sub-slot s - 2 and P@V of s - 3 (else s - 1 and s - 2).
                if not spread and (sub >= t_lag or carried):
                    last_weights = transpose_weights(sub - t_lag, qk_rope)
                # ``item_rope_ahead``: the second-to-last sub-slot of an item transposes and copies the rope keys of the
                # next item (sub-slot Q - 2, or G - 2 with ``p2``).
                rope_item = ahead_item if row == subs - 2 else None
                rope_copies = []
                if sub >= t_lag + 1 or carried:
                    last_pv = numerator(
                        sub - t_lag - 1,
                        last_weights,
                        deferred_sums,
                        rope_item=rope_item,
                        adds_to=pending_adds,
                        rope_out=rope_copies,
                    )
                elif rope_item is not None:
                    transposes, copy = rope_parts(rope_item)
                    for transpose in transposes:
                        transpose()
                    rope_copies.append(copy())
                if exp is not None:
                    # ``act_spread``: the rope key copy of the next item after this sub-slot's exp too.
                    act_extras += [_pinned(copy_ins, exp) for copy_ins in rope_copies]
        # The loop body ends here: a pinned page clamp still pending (none in a body of Q >= 3 rows) and the state adds
        # still pending trace before it.
        if pinned_clamps:
            issue_pinned_clamps(None, None)
        while pending_adds:
            pending_adds.pop(0)()
        while deferred_sums:
            deferred_sums.pop(0)()
        if not carried:
            count_subs = count * subs
            if deep_pipe:
                drain_tail(count_subs)
            else:
                transpose_weights(count_subs - 1, None)
                for slot in range(max(count_subs - 2, 0), count_subs):
                    numerator(slot)

    def new_block_psum(parts=("scores", "keys", "numerator", "weights", "rope")):
        """``psum_per_block``: give the top-level block that starts here its own loop PSUM tensors (``loop_psum``)."""
        nonlocal scores_bufs, banks, mx_psum
        if psum_per_block:
            scores_bufs, banks = loop_psum(parts)
            mx_psum = banks if mx_keys is not None else None

    def run_iteration(step, view, count, names, carried=carry, last=False, gather=False):
        """One iteration of ``count`` slots; it loads the next item row of ``view`` (row ``step`` + 1). ``last``: the
        only iteration of its loop, which loads no next item row and prefetches no slot of a next iteration.
        ``gather`` (``kv_gather``, the main loop): the pages of items ``prefetch`` .. load with the gathers of 2 items."""
        # ``psum_per_block``: the loop body is a top-level block of its own.
        new_block_psum()
        # ``lookahead`` 2: the SP chain of the loads starts again in each loop body.
        sp_chain[0] = None
        # The slot offsets are written in the loop body, as in _key_loops.
        if mask_column:
            write_mask_offsets()
        else:
            nisa.iota(
                dst=slot_offset[_BIAS_ROW : _BIAS_ROW + 1, 0:group_keys],
                pattern=[[_GLOBAL_BLOCK_STRIDE, _GROUP_BLOCKS], [1, _BLOCK]],
                offset=0,
            )
        # A tail iteration uses the first ``count`` items of the row tiles.
        width = count * _ITEM_FIELDS
        row, row_next = (item_row, item_next) if count == slots else (item_row[0:1, 0:width], item_next[0:1, 0:width])
        nisa.tensor_copy(dst=row, src=row_next)
        if mask_column:
            nisa.tensor_copy(dst=thresholds[0:_P_MAX, 0:count], src=threshold_next[0:_P_MAX, 5:width:_ITEM_FIELDS])
        else:
            nisa.tensor_copy(
                dst=thresholds[_BIAS_ROW : _BIAS_ROW + 1, 0:count],
                src=threshold_next[_BIAS_ROW : _BIAS_ROW + 1, 0:count],
            )
        if gather and kv_group:
            # ``kv_group``: the group indices of this iteration's gathered items, from its item row.
            _kv_group_index(kv_group_state, threshold_next, count, page_runs)
        elif gather:
            # The gather indices of this iteration's items, from its item row before the next one loads there.
            _kv_gather_index(kv_gather_state, threshold_next, count, page_runs)
        if not last:
            load_items(row_next, threshold_next, step, names[0], view, count)

        def prepare_pages(slot, defer=False):
            """Load the slot row and pages of ``slot`` (unless prefetched) and clamp its pages (``defer``:
            ``clamp_slot_pages``). ``gather``: the first item of each pair gathers the pages of both."""
            if gather and kv_group:
                # ``kv_group``: the group whose first slot is this one + 2 gathers here (its buffers' last readers,
                # the items 8 slots before, the carried ones included, are done by then); slots 4 .. take their
                # pages from the groups, the others from their SP DMAs.
                ahead = slot + kv_lead
                if ahead % kv_group == 0 and kv_first <= ahead < count:
                    kv_group_gather(ahead, names[1])
                if slot >= prefetch:
                    load_slot(
                        slot,
                        item_row,
                        names[1],
                        slot_threshold(thresholds, slot),
                        loop_stage,
                        pages=slot < kv_first,
                    )
            elif slot >= prefetch:
                if gather and (slot - prefetch) % 2 == 0:
                    kv_gather_pair(slot, names[1])
                load_slot(slot, item_row, names[1], slot_threshold(thresholds, slot), loop_stage, pages=not gather)
            clamp_slot_pages(slot, defer)

        def slot_softmax_of(slot_row, row=0):
            """The softmax reference of stage row ``row`` of a slot row (row 0 with one query row per item)."""
            slot_f = slot_row.view(fp32)
            word = row * (sub_width // 2) + sub_ref_word
            return _SlotSoftmax(
                slot_f[0:sub_cols, word : word + 1],
                slot_f[0:sub_cols, word + 1 : word + 2],
            )

        def load_next_first(defer=False):
            """With ``carry``: load and clamp the slot row and pages of the next iteration's slot 0 (``defer``:
            ``clamp_slot_pages``)."""
            prefetch_thresholds()
            load_slot(0, item_next, names[0], slot_threshold(thresholds_pf, 0), loop_stage)
            clamp_slot_pages(0, defer)

        def prepare_ahead(slot):
            """``lookahead`` 2, in slot ``slot``: clamp the pages of slot ``slot`` + 1 if they were prefetched, then load and
            clamp the slot row and pages of slot ``slot`` + 2 unless prefetched, or with ``carried`` (the last two slots)
            those of slot ``slot`` + 2 - ``count`` of the next iteration, one of its first ``prefetch`` = 2 slots, whose
            clamp (slot 1) runs in slot 0 of that iteration, as with ``lookahead`` 1."""
            if slot + 1 < min(prefetch, count):
                clamp_slot_pages(slot + 1)
            ahead = slot + 2
            if ahead < count:
                if ahead >= prefetch:
                    prepare_pages(ahead)
            elif carried:
                first = ahead - count
                if first == 0:
                    prefetch_thresholds()
                load_slot(first, item_next, names[0], slot_threshold(thresholds_pf, first), loop_stage)
                if first == 0:
                    clamp_slot_pages(0)

        if bg_interleave:
            interleaved_slots(count, prepare_pages, slot_softmax_of, load_next_first, carried, prepare_ahead)
        else:
            for slot in range(count):
                rows = page_rows[slot % page_buffers]
                latent_keys, rope_keys, group_sum, _ = key_sets[slot % _KEY_SETS]
                slot_row = slot_rows[slot % row_sets]
                prepare_pages(slot)
                _transpose_group(
                    rows,
                    latent_keys,
                    rope_keys,
                    slot_offset,
                    thresholds[_BIAS_ROW : _BIAS_ROW + 1, slot : slot + 1],
                    pair_blocks=_GROUP_BLOCKS if pair_pages else 0,
                    transpose_identity=transpose_identity,
                    latent_act=latent_act,
                    mask_column=mask_column,
                    rope_identity=rope_identity,
                    **({"mask_engine": nisa.engine.gpsimd} if mask_pool else {}),
                )
                numerator_psum = _attend_group(
                    0,
                    None,
                    None,
                    latent_keys if transpose_identity is None else _fp8_word_keys(latent_keys),
                    rope_keys if rope_identity is None else _fp8_word_rope(rope_keys),
                    rows,
                    slot_softmax_of(slot_row),
                    group_sum,
                    sb,
                    q_tiles=[slot_row[0:_P_MAX, t * _HEADS : (t + 1) * _HEADS] for t in range(_KV_LORA_TILES)],
                    q_rope=slot_row[0 : _BIAS_ROW + 1, q_width : q_width + _HEADS],
                    pair_blocks=_GROUP_BLOCKS if pair_pages else 0,
                )
                add_to_state(slot, numerator_psum)
        if last:
            return
        if not carried:
            prefetch_thresholds()
        # With ``carry`` slot 0 of the next iteration loaded in the last slot; a second prefetched slot loads here,
        # after P@V of slot count - 3 read its slot row. With ``lookahead`` 2 the last two slots loaded both.
        for slot in range(lookahead if carried else 0, prefetch):
            load_slot(slot, item_next, names[0], slot_threshold(thresholds_pf, slot), loop_stage)

    iterations = nisa.scalar_load(iteration_i[0:1, 0:1])
    # The tail bounds load before the main loop, so only branches lie between the loops: no fence there.
    tail_columns = range(1, 1 + 2 * len(tail)) if tail else ()
    tail_bounds = [nisa.scalar_load(iteration_i[0:1, column : column + 1]) for column in tail_columns]
    # The bounds of the clamped block's loop load here too (``clamp_block``: the last two columns), unless its body runs
    # once in place of the loop (``CLAMP_BLOCK_ONCE``).
    block_columns = (iteration_i.shape[1] - 2, iteration_i.shape[1] - 1) if clamp_block and not CLAMP_BLOCK_ONCE else ()
    block_bounds = [nisa.scalar_load(iteration_i[0:1, column : column + 1]) for column in block_columns]
    nl.fori_loop(
        0,
        iterations,
        lambda step: run_iteration(step, item_view, slots, ("next", "load"), gather=kv_gather or bool(kv_group)),
    )
    if carry_main_only:
        # The drain of the main loop (of the null slot when it ran no iteration), before the tail loops' skips.
        new_block_psum(("numerator", "weights"))
        if deep_pipe:
            drain_tail(0)
        else:
            transpose_weights(-1, None)
            for slot in (-2, -1):
                numerator(slot)

    def tail_body(view, count, names):
        """The body of a tail loop of ``count`` slots on ``view``."""
        return lambda step: run_iteration(step, view, count, names, carried=carry and not carry_main_only)

    for level, tail_count in enumerate(tail or ()):
        # The last iteration of the loop before (or the prologue) loaded the item row after its last one, whose first
        # ``tail_count`` items are the first row of this loop, and the slot rows and pages of its first ``prefetch``
        # slots. The second tail loop names its DMAs apart from the first.
        tail_view = item_hbm.reshape((item_shape[0] // tail_count, tail_count * _ITEM_FIELDS))
        names = ("tail_next", "tail_load") if level == 0 else (f"tail{level + 1}_next", f"tail{level + 1}_load")
        nl.fori_loop(tail_bounds[2 * level], tail_bounds[2 * level + 1], tail_body(tail_view, tail_count, names))
    if clamp_block:
        # The block of the last key groups, with the page clamp: its first ``prefetch`` slots loaded at the end of the
        # loop before (unclamped), and this body clamps every slot it runs, so all its pages are clamped.
        clamp_now[0] = True
        block_view = item_hbm.reshape((item_shape[0] // clamp_block, clamp_block * _ITEM_FIELDS))
        if CLAMP_BLOCK_ONCE:
            # One iteration a call: the body once, on the item row that the loop before loaded, with no next row.
            run_iteration(None, block_view, clamp_block, ("clamp_next", "clamp_load"), carried=False, last=True)
        else:
            nl.fori_loop(
                block_bounds[0], block_bounds[1], tail_body(block_view, clamp_block, ("clamp_next", "clamp_load"))
            )
    if carry and not carry_main_only:
        # The drain of the last two slots of the last iteration (of the null slot when no iteration ran).
        new_block_psum(("numerator", "weights"))
        if deep_pipe:
            drain_tail(0)
        else:
            transpose_weights(-1, None)
            for slot in (-2, -1):
                numerator(slot)


def _packed_key_loop(*args, **kwargs):
    """Run ``_packed_key_loop_steps`` to its end in one call."""
    for _ in _packed_key_loop_steps(*args, **kwargs):
        pass


def _sanitize_cache_rows(rows, bound=None, dst=None):
    """Clamp each cache row of ``rows`` (SBUF) to a finite bound, in place (or into ``dst``); return the instruction.

    A masked row of a page can hold bytes that the MLA did not write: a stale row of a reused block, or
    KDA state in a shared null page. In an FP8 page about 1% of such bytes are NaN codes (0x7F, 0xFF); in
    a BF16 page they read as +-Inf or a value up to about 3.4e38. A non-finite or huge key makes the
    masked product ``weight * key`` a NaN or an overflow, though the weight is near 0, and a huge score
    turns a masked key into the score cap. The ISA maximum ignores a NaN operand, so the clamp gives a
    finite value and keeps every normalized cache value, which is far below the bound of either dtype. ``bound``
    (default: by the dtype of ``rows``) is the FP8 bound for FP8 pages in BF16 tiles (``loop_cast_engines`` page forms).
    """
    if bound is None:
        bound = _BF16_KEY_MAX if rows.dtype == nl.bfloat16 else fp8_page_max()
    return nisa.tensor_scalar(
        dst=rows if dst is None else dst,
        data=rows,
        op0=nl.maximum,
        operand0=-bound,
        op1=nl.minimum,
        operand1=bound,
        engine=nisa.engine.vector,
    )


def _cast_page(dst, page, clamp):
    """``loop_cast_engines`` "page": write the FP8 page ``page`` (``[128, 576]`` SBUF) into the BF16 page ``dst`` with one
    VectorE op: the clamp of ``_sanitize_cache_rows`` (FP8 bound) if ``clamp``, else an exact copy."""
    if clamp:
        return _sanitize_cache_rows(page, bound=fp8_page_max(), dst=dst)
    return nisa.tensor_copy(dst=dst, src=page, engine=nisa.engine.vector)


def _slot_scores(
    latent_keys,
    rope_keys,
    q_tiles,
    q_rope,
    softmax,
    group_sum,
    weights,
    qk_hook,
    first_after=None,
    scores=None,
    mx=None,
    softmax_later=False,
    rope_rows=_BIAS_ROW + 1,
    dr=None,
    cap_out=None,
    mx_qk=None,
    capless=False,
):
    """The capped scores and weights of one packed slot, as the first half of ``_attend_group``: QK and the rope
    matmul into PSUM, the score cap, and the exp into ``weights`` (``[96, 512]`` BF16) with its sum in ``group_sum``.
    ``qk_hook(k, qk)`` emits instructions to issue after QK matmul k and returns the last one (or None); the next QK
    matmul (the rope matmul after k = 3) issues after it, and the first QK matmul after ``first_after`` (if given;
    scheduling-only dependencies). Return the rope matmul. With ``softmax_later`` (``pack_rope_ahead``) the cap and the
    exp are not traced here: return the rope matmul and a function ``softmax_ops(after=None)`` that traces them (the
    cap on VectorE, the exp on ScalarE after ``after``, a scheduling-only dependency) and returns the exp.

    With ``mx`` (``pack_mx_keys``: the FP32 views of the 1 or 2 MX query stationaries ``[128, 96]``, the unit E8M0
    scales and a list that the hook fills with its ScalarE and VectorE key copies), ``latent_keys`` ``[128, 512]`` holds
    the 32-bit MX words of the keys and the latent part of the scores is one MX matmul for each stationary; the hook
    runs once, after the last of them (k = 0). The score cap issues after the VectorE key copy of the hook and the exp
    after its ScalarE key copy (scheduling-only), so the next slot's keys do not queue behind this slot's softmax.

    ``rope_rows`` is the contraction of the rope matmul (BF16 queries): the 64 rope rows and the bias rows, one (the
    mask row) or Q (the mask columns of the multi-query packed loop). With ``mx`` and Q > 1 rows (the stationaries of one
    row, an empty copy list) ``cap_out`` takes the score cap, as without ``mx``.

    With ``dr`` (``pack_dr_keys``: the 2 e4m3 double_row stationaries ``[128, 2, 96]``, ``_dr_stationary``),
    ``latent_keys`` ``[128, 2, 512]`` holds the 2-byte key words (``_dr_key_words``) and the latent part of the scores is
    2 double_row matmuls, pass ``i`` on byte ``i`` of the words; the hook runs after each (k = 0, 1).

    With ``mx_qk`` (``pack_mx_qk``: the FP32 view of the MX query words ``[128, parts]``, the unit E8M0 scales and the
    query's own scales ``[128, parts]`` uint8 or None for unit scales), ``latent_keys`` ``[128, 512]`` holds the 32-bit
    ``word_keys`` key words and the latent part of the scores is one MX matmul; the hook runs once, after it (k = 0). The
    rope matmul and the softmax are those of the BF16 query.

    With ``capless`` (``_packed_key_loop`` ``capless``) no score cap is traced (``cap_out`` gets nothing): the exp reads the
    raw scores."""
    num_keys = _GROUP_BLOCKS * _BLOCK
    scores_psum = scores if scores is not None else nl.ndarray((_HEADS, num_keys), dtype=nl.float32, buffer=nl.psum)
    pinned = first_after
    copies = mx[2] if mx is not None else []

    # The partitions of the scores: the 96 heads, or 128 (row, head) columns (``pack_p2``).
    parts = scores_psum.shape[0]

    def softmax_ops(after=None):
        cap = None
        if not capless:
            cap = nisa.tensor_scalar(
                dst=scores_psum,
                data=scores_psum,
                op0=nl.minimum,
                operand0=softmax.score_cap[0:parts, 0:1],
                engine=nisa.engine.vector,
            )
        if cap_out is not None and cap is not None:
            # ``cap_first`` (the Q > 1 MX form): the caller orders VectorE work after this cap.
            cap_out.append(cap)
        exp = nisa.activation(
            dst=weights,
            op=nl.exp,
            data=scores_psum,
            scale=_SOFTMAX_SCALE,
            bias=softmax.negative_reference[0:parts, 0:1],
            reduce_op=nl.add,
            reduce_res=group_sum,
            reduce_cmd=nisa.reduce_cmd.reset_reduce,
            engine=nisa.engine.scalar,
        )
        if copies:
            if cap is not None:
                _pinned(cap, copies[-1])
            _pinned(exp, copies[0])
        if after is not None:
            _pinned(exp, after)
        return exp

    if mx is not None:
        stationaries, mx_scale, copies = mx
        for part, stationary in enumerate(stationaries):
            qk = _pinned(
                nisa.nc_matmul_mx(
                    dst=scores_psum,
                    stationary=stationary.view(nl.float8_e4m3fn_x4),
                    moving=latent_keys[0:_P_MAX, 0:num_keys].view(nl.float8_e4m3fn_x4),
                    stationary_scale=mx_scale[0:_P_MAX, 0:_HEADS],
                    moving_scale=mx_scale[0:_P_MAX, 0:num_keys],
                    accumulate=part > 0,
                ),
                pinned,
            )
            pinned = qk
        pinned = qk_hook(0, qk)
        qk_rope = _pinned(
            nisa.nc_matmul(
                dst=scores_psum, stationary=q_rope, moving=rope_keys[0:rope_rows, 0:num_keys], accumulate=True
            ),
            pinned,
        )
        softmax_ops()
        return qk_rope
    if mx_qk is not None:
        stationary, mx_scale, q_scale = mx_qk
        qk = _pinned(
            nisa.nc_matmul_mx(
                dst=scores_psum,
                stationary=stationary.view(nl.float8_e4m3fn_x4),
                moving=latent_keys[0:_P_MAX, 0:num_keys].view(nl.float8_e4m3fn_x4),
                stationary_scale=mx_scale[0:_P_MAX, 0:parts] if q_scale is None else q_scale,
                moving_scale=mx_scale[0:_P_MAX, 0:num_keys],
                accumulate=False,
            ),
            pinned,
        )
        pinned = qk_hook(0, qk)
    for latent_tile in range(0 if mx_qk is not None else _KV_LORA_TILES if dr is None else 2):
        if dr is None:
            qk = nisa.nc_matmul(
                dst=scores_psum,
                stationary=q_tiles[latent_tile],
                moving=latent_keys[0:_P_MAX, latent_tile, 0:num_keys],
                accumulate=latent_tile > 0,
            )
        else:
            qk = _instruction(
                nisa.nc_matmul(
                    dst=scores_psum,
                    stationary=dr[latent_tile],
                    moving=_dr_word_keys(latent_keys, latent_tile),
                    accumulate=latent_tile > 0,
                    perf_mode=nisa.matmul_perf_mode.double_row,
                )
            )
        if pinned is not None:
            qk.depends_on(pinned, kind=nisa.dependency.schedule)
        pinned = qk_hook(latent_tile, qk)
    qk_rope = nisa.nc_matmul(
        dst=scores_psum, stationary=q_rope, moving=rope_keys[0:rope_rows, 0:num_keys], accumulate=True
    )
    if pinned is not None:
        qk_rope.depends_on(pinned, kind=nisa.dependency.schedule)
    if softmax_later:
        return qk_rope, softmax_ops
    cap = (
        None
        if capless
        else nisa.tensor_scalar(
            dst=scores_psum, data=scores_psum, op0=nl.minimum, operand0=softmax.score_cap[0:parts, 0:1]
        )
    )
    if cap_out is not None and cap is not None:
        # ``cap_first``: the caller orders VectorE work after this cap.
        cap_out.append(cap)
    nisa.activation(
        dst=weights,
        op=nl.exp,
        data=scores_psum,
        scale=_SOFTMAX_SCALE,
        bias=softmax.negative_reference[0:parts, 0:1],
        reduce_op=nl.add,
        reduce_res=group_sum,
        reduce_cmd=nisa.reduce_cmd.reset_reduce,
    )
    return qk_rope


def _transpose_rows(dst, data, cast_engine=None):
    """Transpose cache rows ``data`` (``[128 keys, F]`` SBUF) into ``dst`` (``[F, 128]`` BF16 PSUM).

    A TensorE transpose keeps the dtype of its input. On Trn3, FP8 rows therefore go through a matmul with a BF16
    identity, which upcasts them exactly into BF16 PSUM. Trn2 has no BF16 matmul output, so the rows first upcast
    exactly to a BF16 tile, which a BF16 transpose then moves. ``cast_engine`` (if given) runs that upcast.
    """
    if data.dtype == nl.bfloat16:
        nisa.nc_transpose(dst=dst, data=data)
    elif is_trn3():
        identity = nl.shared_identity_matrix(_P_MAX, dtype=nl.bfloat16)
        nisa.nc_matmul(dst=dst, stationary=data, moving=identity[0:_P_MAX, 0:_P_MAX])
    else:
        rows = nl.ndarray(tuple(data.shape), dtype=nl.bfloat16, buffer=nl.sbuf)
        if cast_engine is None:
            nisa.tensor_copy(dst=rows, src=data)
        else:
            nisa.tensor_copy(dst=rows, src=data, engine=cast_engine)
        nisa.nc_transpose(dst=dst, data=rows)


def _transpose_group(
    rows,
    latent_keys,
    rope_keys,
    slot_offset,
    threshold,
    pair_blocks=0,
    transpose_identity=None,
    latent_act=2,
    mask_column=False,
    rope_identity=None,
    skip_latent=False,
    rope_after=None,
    rope_psum=None,
    mask_rows=1,
    cast_engines=None,
    mask_window=None,
    mask_engine=None,
):
    """Transpose the cache rows of one key group into feature-major tiles, and write the mask row below
    the rope keys: a key with a slot offset of at least ``threshold`` (``[1, 1]`` on the bias partition)
    gets the mask bias. With ``pair_blocks``, ``rows`` holds that many pages as pair rows. With
    ``transpose_identity`` (the FP8 identity of FP8 row-major rows), the latent keys are FP8 words
    (``_transpose_mode_tile``). ``latent_act`` latent tiles copy out of PSUM on ScalarE (``_LATENT_ACT_TILES``), the
    others on VectorE. With ``mask_column`` the rows carry the mask in column 576 (``_mask_column``), which the rope
    transposes move into the mask row, and with ``rope_identity`` (the FP8 identity) those run in transpose mode and
    write FP8 words (``_fp8_word_rope``). ``skip_latent`` leaves out the latent keys (already transposed). With
    ``rope_identity``, the first rope transpose issues after ``rope_after`` (if given; a scheduling-only dependency).
    With ``rope_identity`` and ``rope_psum`` (a PSUM tensor of rope-word buffers and the buffer index, ``pack_mx_keys``),
    the rope transposes write that buffer, else a new PSUM tile. With ``mask_column`` and ``mask_rows`` Q (the Q mask
    columns of the multi-query packed loop) the rope transposes move the Q columns into bias rows 64 .. 64 + Q - 1.
    ``cast_engines`` (``loop_cast_engines``): the engine of the FP8 -> BF16 copy of each page tile, by ``(latent tile,
    block)`` or ``("rope", block)``. ``mask_window`` (``kv_pitch576``, with ``mask_column`` and ``rope_identity``): the mask
    columns come from that window (``_mask_window``), not from ``rows`` (576-byte rows). ``mask_engine`` (``pack_mask_pool``:
    GpSimd) is the engine of the mask row (``_mask_row``); None lets the compiler select it."""
    group_blocks = pair_blocks or rows.shape[1]
    num_keys = group_blocks * _BLOCK
    if pair_blocks:
        pair_key_tiles(rows, pair_blocks, latent_keys, rope_keys)
        _mask_row(rope_keys, slot_offset, threshold, num_keys, engine=mask_engine)
        return
    for latent_tile in range(0 if skip_latent else _KV_LORA_TILES):
        if transpose_identity is not None:
            _transpose_mode_tile(rows, latent_keys, latent_tile, group_blocks, transpose_identity, latent_act)
            continue
        latent_psum = nl.ndarray((_P_MAX, num_keys), dtype=nl.bfloat16, buffer=nl.psum)
        for local_block in range(group_blocks):
            _transpose_rows(
                dst=latent_psum[0:_P_MAX, local_block * _BLOCK : (local_block + 1) * _BLOCK],
                data=rows[0:_P_MAX, local_block, latent_tile * _P_MAX : (latent_tile + 1) * _P_MAX],
                **({"cast_engine": cast_engines[(latent_tile, local_block)]} if cast_engines else {}),
            )
        nisa.tensor_copy(
            dst=latent_keys[0:_P_MAX, latent_tile, 0:num_keys],
            src=latent_psum,
            engine=nisa.engine.scalar if latent_tile in _LATENT_ACT_TILES[latent_act] else nisa.engine.vector,
        )
    rope_width = _ROPE + mask_rows if mask_column else _ROPE
    if rope_identity is not None:
        transposes, copy = _rope_mode_parts(
            rows, rope_keys, rope_identity, group_blocks, rope_width, rope_psum, mask_window=mask_window
        )
        for local_block, transpose in enumerate(transposes):
            rope_mm = transpose()
            if rope_after is not None and local_block == 0:
                rope_mm.depends_on(rope_after, kind=nisa.dependency.schedule)
        copy()
        return
    rope_psum = nl.ndarray((rope_width, num_keys), dtype=nl.bfloat16, buffer=nl.psum)
    for local_block in range(group_blocks):
        _transpose_rows(
            dst=rope_psum[0:rope_width, local_block * _BLOCK : (local_block + 1) * _BLOCK],
            data=rows[0:_P_MAX, local_block, _KV_LORA : _KV_LORA + rope_width],
            **({"cast_engine": cast_engines[("rope", local_block)]} if cast_engines else {}),
        )
    nisa.tensor_copy(dst=rope_keys[0:rope_width, 0:num_keys], src=rope_psum, engine=nisa.engine.scalar)
    if not mask_column:
        _mask_row(rope_keys, slot_offset, threshold, num_keys, engine=mask_engine)


def _rope_mode_parts(rows, rope_keys, identity, group_blocks, rope_width, rope_psum=None, mask_window=None):
    """The transpose-mode rope key transposes of the ``group_blocks`` pages of FP8 row-major ``rows`` (``identity`` is
    the FP8 identity) and their exact 2-byte ScalarE copy into ``rope_keys``, as functions that each trace one
    instruction and return it: a list of the transposes, and the copy (``_transpose_group``). The transposes write a new
    PSUM tile, or with ``rope_psum`` (a PSUM tensor of rope-word buffers and the buffer index, ``pack_mx_keys``) that
    buffer."""
    step = _FP8_TRANSPOSE_STEP
    num_keys = group_blocks * _BLOCK
    if rope_psum is None:
        words, base = nl.ndarray((rope_width, step * num_keys), dtype=rows.dtype, buffer=nl.psum), 0
    else:
        words, base = rope_psum[0], rope_psum[1] * step * num_keys

    def transpose(local_block):
        columns = slice(base + step * local_block * _BLOCK, base + step * (local_block + 1) * _BLOCK, step)
        if mask_window is None:
            return nisa.nc_matmul(
                dst=words[0:rope_width, columns],
                stationary=rows[0:_P_MAX, local_block, _KV_LORA : _KV_LORA + rope_width],
                moving=identity,
                is_transpose=True,
                accumulate=False,
            )
        # ``mask_window``: the window (its Q mask columns last) writes rows 0 .. 64 + Q, then the 64 rope columns of the
        # rows overwrite rows 0 .. 63 (TensorE order), so the rows hold the rope keys and the bias rows of the mask.
        nisa.nc_matmul(
            dst=words[0:rope_width, columns],
            stationary=mask_window(local_block),
            moving=identity,
            is_transpose=True,
            accumulate=False,
        )
        return nisa.nc_matmul(
            dst=words[0:_ROPE, columns],
            stationary=rows[0:_P_MAX, local_block, _KV_LORA:_CACHE_WIDTH],
            moving=identity,
            is_transpose=True,
            accumulate=False,
        )

    def copy():
        return nisa.tensor_copy(
            dst=rope_keys.view(nl.uint16)[0:rope_width, 0:num_keys],
            src=words.view(nl.uint16)
            if rope_psum is None
            else words.view(nl.uint16)[0:rope_width, base // 2 : base // 2 + num_keys],
            engine=nisa.engine.scalar,
        )

    return [functools.partial(transpose, local_block) for local_block in range(group_blocks)], copy


def _transpose_mode_tile(
    rows, latent_keys, latent_tile, group_blocks, identity, latent_act=2, after=None, copy_out=None, copy_after=None
):
    """Write latent tile ``latent_tile`` of the keys of FP8 row-major ``rows`` into ``latent_keys`` with TensorE
    transposes in transpose mode (``identity`` is the FP8 identity), as 2-byte words that hold each FP8 key in the
    low byte (``_fp8_word_keys``).

    A transpose-mode transpose keeps the FP8 dtype and writes every other PSUM byte (element step 2). One 2-byte
    copy moves the words to SBUF at the rate of a BF16 copy, and QK reads the keys through a stride-2 FP8 view.
    The keys have the values of the identity matmul (an FP8 -0 stays -0, which adds no score), and the BF16 x FP8
    products of QK are exact, so the scores are the same. Unlike a matmul, a transpose-mode instruction can run in
    the TensorE background on trn3, beside the matmuls. Each transpose reads its own [128, 128] slice of ``rows``,
    so no two consecutive transposes read the same tile. With ``after``, the first transpose issues after that
    instruction (a scheduling-only dependency). With ``copy_out`` (a list), the copy is appended to it. With
    ``copy_after``, the copy issues after that instruction (scheduling-only). Return the last transpose.
    """
    step = _FP8_TRANSPOSE_STEP
    num_keys = group_blocks * _BLOCK
    words = nl.ndarray((_P_MAX, step * num_keys), dtype=rows.dtype, buffer=nl.psum)
    last = None
    for local_block in range(group_blocks):
        last = nisa.nc_matmul(
            dst=words[0:_P_MAX, step * local_block * _BLOCK : step * (local_block + 1) * _BLOCK : step],
            stationary=rows[0:_P_MAX, local_block, latent_tile * _P_MAX : (latent_tile + 1) * _P_MAX],
            moving=identity,
            is_transpose=True,
            accumulate=False,
        )
        if after is not None and local_block == 0:
            last.depends_on(after, kind=nisa.dependency.schedule)
    copy_ins = nisa.tensor_copy(
        dst=latent_keys.view(nl.uint16)[0:_P_MAX, latent_tile, 0:num_keys],
        src=words.view(nl.uint16),
        engine=nisa.engine.scalar if latent_tile in _LATENT_ACT_TILES[latent_act] else nisa.engine.vector,
    )
    if copy_after is not None:
        _pinned(copy_ins, copy_after)
    if copy_out is not None:
        copy_out.append(copy_ins)
    return last


class _LaneKeys:
    """The latent keys of ``word_keys`` as the QK moving operands: ``[0:128, t, 0:n]`` is byte lane ``t`` of the key words
    ``[128, keys]`` FP32 at element step 4 (latent tile ``t``: features ``128 t + c`` on partition ``c``)."""

    def __init__(self, words):
        self.words = words

    def __getitem__(self, index):
        parts, lane, keys = index
        kernel_assert(
            parts == slice(0, _P_MAX) and keys.start == 0 and keys.step is None, "word_keys QK reads whole key tiles"
        )
        return self.words.ap(
            pattern=[[4 * self.words.shape[1], _P_MAX], [4, keys.stop]], offset=lane, dtype=fp8_page_dtype()
        )


def _fp8_word_keys(latent_keys):
    """Return the FP8 view of the latent keys that ``_transpose_mode_tile`` writes: the low byte of each word."""
    step = _FP8_TRANSPOSE_STEP
    keys = latent_keys.shape[2]
    return latent_keys.view(fp8_page_dtype())[0:_P_MAX, 0:_KV_LORA_TILES, 0 : step * keys : step]


def _tile_address(tile):
    """Return the fixed (partition, byte) SBUF address of ``tile``, or None when the compiler places it."""
    for attr in ("address", "_address"):
        if hasattr(tile, attr):
            return getattr(tile, attr)
    return None


def _fp8_word_rope(rope_keys):
    """Return the FP8 view of the rope keys and mask row that the transpose-mode rope transposes write: the low byte of
    each word."""
    step = _FP8_TRANSPOSE_STEP
    keys = rope_keys.shape[1]
    return rope_keys.view(fp8_page_dtype())[0:_P_MAX, 0 : step * keys : step]


def _instruction(ins):
    """Return the instruction handle of ``ins``."""
    return ins if hasattr(ins, "depends_on") else nisa.NkiInstruction(ins)


def _pinned(ins, after):
    """Return the instruction handle of ``ins``, issued after ``after`` (if given; a scheduling-only dependency)."""
    handle = ins if hasattr(ins, "depends_on") else nisa.NkiInstruction(ins)
    if after is not None:
        handle.depends_on(after, kind=nisa.dependency.schedule)
    return handle


def _mx_key_words(rows, latent_keys, words_psum, after=None, copy_out=None):
    """Write the MX key words of the FP8 row-major page tile ``rows`` (``pack_mx_keys``) into ``latent_keys``
    ``[128, 512]`` FP32: word ``(c, k)`` holds the latent features ``4c .. 4c + 3`` of key ``k`` (``float8_e4m3fn_x4``).

    One 32-bit TensorE transpose of each page into ``words_psum`` (``[128, 512]`` FP32, a whole PSUM bank that takes
    no matmul writes, NCC_IGCA074) moves the bits; the first issues after ``after`` (if given; scheduling-only). An FP32
    copy on ScalarE (keys 0-255) and one on VectorE (keys 256-511) move the words to SBUF; with ``copy_out`` (a list)
    they are appended to it in that order. Return the last transpose.

    Warning: an FP32 copy may flush a subnormal word to zero (a word whose feature 4c + 3 is +-0 and whose feature 4c + 2
    is non-negative), which zeroes its 4 keys; ``_mx_item_key_words`` copies through a uint16 view instead, which keeps
    the bytes.
    """
    group_blocks = rows.shape[1]
    row_words = rows.view(nl.float32)
    words = _KV_LORA // _MX_PACK
    last = None
    for local_block in range(group_blocks):
        last = _pinned(
            nisa.nc_transpose(
                dst=words_psum[0:_P_MAX, local_block * _BLOCK : (local_block + 1) * _BLOCK],
                data=row_words[0:_P_MAX, local_block, 0:words],
                engine=nisa.engine.tensor,
            ),
            after if local_block == 0 else None,
        )
    half = group_blocks * _BLOCK // 2
    for part, engine in enumerate((nisa.engine.scalar, nisa.engine.vector)):
        keys = slice(part * half, (part + 1) * half)
        copy_ins = _pinned(
            nisa.tensor_copy(dst=latent_keys[0:_P_MAX, keys], src=words_psum[0:_P_MAX, keys], engine=engine), None
        )
        if copy_out is not None:
            copy_out.append(copy_ins)
    return last


def _mx_item_key_words(rows, latent_keys, words_psum, after=None, act_after=None):
    """Write the MX key words of the FP8 row-major page tile ``rows`` into ``latent_keys`` ``[128, 512]`` FP32 for the Q > 1
    packed loop (``pack_mx_keys`` "single"), as ``_mx_key_words`` lays them out: one 32-bit TensorE transpose of each page
    into ``words_psum`` (a whole PSUM bank that takes no matmul writes), the first after ``after`` (if given;
    scheduling-only), then one exact ScalarE copy through a uint16 view after the last transpose and ``act_after`` (if
    given; scheduling-only), as ``_dr_key_copies`` copies the double_row key words. One engine reads the bank, and
    VectorE keeps the score cap and the state adds. Return the last transpose."""
    group_blocks = rows.shape[1]
    num_keys = group_blocks * _BLOCK
    row_words = rows.view(nl.float32)
    words = _KV_LORA // _MX_PACK
    last = None
    for local_block in range(group_blocks):
        last = _pinned(
            nisa.nc_transpose(
                dst=words_psum[0:_P_MAX, local_block * _BLOCK : (local_block + 1) * _BLOCK],
                data=row_words[0:_P_MAX, local_block, 0:words],
                engine=nisa.engine.tensor,
            ),
            after if local_block == 0 else None,
        )
    copy_ins = _pinned(
        nisa.tensor_copy(
            dst=latent_keys.view(nl.uint16)[0:_P_MAX, 0 : 2 * num_keys],
            src=words_psum.view(nl.uint16)[0:_P_MAX, 0 : 2 * num_keys],
            engine=nisa.engine.scalar,
        ),
        last,
    )
    if act_after is not None:
        _pinned(copy_ins, act_after)
    return last


def _dr_key_words(rows, words_psum, identity16, block, after=None):
    """The 2-byte transpose-mode transposes of block ``block`` (features 256 block .. 256 block + 255) of the pages of the
    FP8 row-major page tile ``rows`` into ``words_psum`` ``[128, 2 x 512]`` BF16 (``pack_dr_keys``; a whole PSUM bank
    that takes no matmul writes): word ``(c, block, k)`` holds features ``256 block + 2c`` (low byte) and ``+ 1`` (high
    byte) of key ``k``. Transpose mode moves the bits (``identity16`` is the BF16 identity). The first transpose issues
    after ``after`` (if given; scheduling-only). Return the last transpose."""
    group_blocks = rows.shape[1]
    row_words = rows.view(nl.bfloat16)
    col = block * _DR_BLOCK_WORDS
    last = None
    for local_block in range(group_blocks):
        keys = slice((block * group_blocks + local_block) * _BLOCK, (block * group_blocks + local_block + 1) * _BLOCK)
        last = _pinned(
            nisa.nc_matmul(
                dst=words_psum[0:_P_MAX, keys],
                stationary=row_words[0:_P_MAX, local_block, col : col + _DR_BLOCK_WORDS],
                moving=identity16,
                is_transpose=True,
                accumulate=False,
            ),
            after if local_block == 0 else None,
        )
    return last


def _dr_key_copies(latent_keys, words_psum, after=None, act_after=None, copy_out=None):
    """The exact 2-byte copy of the key words ``words_psum`` into ``latent_keys`` ``[128, 2, 512]`` (``pack_dr_keys``): one
    ScalarE copy of both blocks after ``after`` (the last transpose of the bank: a copy that ran beside a transpose of the
    same PSUM bank would make the transpose wait) and ``act_after`` (if given; scheduling-only). One engine reads the bank:
    two engines reading one PSUM bank would serialize, and VectorE keeps the score cap and the state adds. With
    ``copy_out`` (a list) the copy is appended to it."""
    num_keys = latent_keys.shape[2]
    copy_ins = _pinned(
        nisa.tensor_copy(
            dst=latent_keys.view(nl.uint16).reshape((_P_MAX, _DR_BLOCKS * num_keys)),
            src=words_psum.view(nl.uint16)[0:_P_MAX, 0 : _DR_BLOCKS * num_keys],
            engine=nisa.engine.scalar,
        ),
        after,
    )
    if act_after is not None:
        _pinned(copy_ins, act_after)
    if copy_out is not None:
        copy_out.append(copy_ins)


def _dr_word_keys(latent_keys, byte: int):
    """Return the FP8 moving operand ``[128, 2, 512]`` of double_row pass ``byte``: byte ``byte`` of the key words of both
    blocks (``pack_dr_keys``)."""
    num_keys = latent_keys.shape[2]
    return latent_keys.view(fp8_page_dtype())[0:_P_MAX, 0:_DR_BLOCKS, byte : 2 * num_keys : 2]


def _mask_column(rows, mask_offsets, threshold):
    """Write the mask column of a page tile on GpSimd: ``rows[p, b, 576]`` (FP8) is minus the FP8 bound (``fp8_page_max``) where the slot offset of key
    (b, p), ``mask_offsets`` ``[128, 4, 1]`` (4096 b + p), is at least ``threshold`` (``[128, 1]``), else 0.

    The page DMAs write bytes 0-575 of each 608-byte row and this write byte 576, so no 32-byte granule holds both. The
    query bias row of 2^100 makes a masked score about -5.7e32 (exact in FP32), below every capped reference, so its
    weight is 0, as with the -1e30 of ``_mask_row``.

    With ``mask_offsets`` ``[128, 4, Q]`` (4096 b + p - j: the multi-query packed loop) the write fills the Q mask
    columns 576 .. 576 + Q - 1, column j masking the keys at or above ``threshold + j`` (row j of the item).
    """
    nisa.tensor_scalar(
        dst=rows[0:_P_MAX, 0:_GROUP_BLOCKS, _CACHE_WIDTH : _CACHE_WIDTH + mask_offsets.shape[2]],
        data=mask_offsets,
        op0=nl.greater_equal,
        operand0=threshold,
        op1=nl.multiply,
        operand1=-fp8_page_max(),
        engine=nisa.engine.gpsimd,
    )


def _vog5_constants(pack_sb, sb, items: int, page_runs: int, tag: str) -> dict:
    """``kv_group``: the pass-invariant constants of the group gathers of G = ``items`` items in one plan tile ``sc``
    ``[128, 2 G]`` (``_VOG5_PLAN_SLOTS``): ``sel`` in its first G columns and ``const`` in the next G
    (``_kv_group_constants``; ``sb`` unused)."""
    sc = _plan_tile(pack_sb, (_P_MAX, 2 * items), nl.float32, f"{tag}_mla_pack_group_consts", _VOG5_PLAN_SLOTS[1])
    out = {"sc": sc, "sel": sc[0:_P_MAX, 0:items], "const": sc[0:_P_MAX, items : 2 * items]}
    _kv_group_constants(sb, items, page_runs, out)
    return out


def _rem4_offsets_in_place(offsets, start):
    """``kv_rem4``: write into ``offsets`` (``[128, 4, Q]`` FP32) the slot offsets of the mask columns: slot b of
    partition p holds row 4 (p % 32) + b of page p // 32 of the item, so its offset is 4096 (p // 32) + 4 (p % 32) + b,
    minus j for mask column j (the 4096 b + p - j of the row-per-partition map). No tile of its own: its own words hold
    the intermediates, and ``start`` (``[128, >= 2]`` FP32, written again later) the start of each partition. Small
    integers, exact in FP32."""
    int32 = nl.int32
    vector = nisa.engine.vector
    queries = offsets.shape[2]
    words = offsets.view(int32).reshape((_P_MAX, _GROUP_BLOCKS * queries))
    start_i = start.view(int32)
    nisa.iota(dst=words[0:_P_MAX, 0:1], pattern=[[0, 1]], offset=0, channel_multiplier=1)
    nisa.tensor_scalar(
        dst=words[0:_P_MAX, 1:2],
        data=words[0:_P_MAX, 0:1],
        op0=nl.right_shift,
        operand0=5,
        op1=nl.left_shift,
        operand1=12,
        engine=vector,
    )
    nisa.tensor_scalar(
        dst=words[0:_P_MAX, 2:3],
        data=words[0:_P_MAX, 0:1],
        op0=nl.bitwise_and,
        operand0=_REM4_PARTS - 1,
        op1=nl.left_shift,
        operand1=2,
        engine=vector,
    )
    nisa.tensor_tensor(
        dst=start_i[0:_P_MAX, 0:1], data1=words[0:_P_MAX, 1:2], data2=words[0:_P_MAX, 2:3], op=nl.add, engine=vector
    )
    nisa.tensor_copy(dst=start[0:_P_MAX, 1:2], src=start_i[0:_P_MAX, 0:1], engine=vector)
    nisa.iota(dst=offsets, pattern=[[1, _GROUP_BLOCKS], [1, 1] if queries == 1 else [-1, queries]], offset=0)
    nisa.tensor_scalar(dst=offsets, data=offsets, op0=nl.add, operand0=start[0:_P_MAX, 1:2], engine=vector)


def _kv_gather_constants(sb, slots: int, page_runs: int, rem4: bool = False) -> dict:
    """``kv_gather``: the tiles of the gather indices of a loop of ``slots`` items: ``index`` ``[128, slots / 2 x 8]``
    int32 (``_kv_gather_index`` writes it each iteration), its FP32 scratch (the pages, the page of each row), and two
    constants: ``select`` ``[128, 8]`` (1.0 where the column is q % 8 on partition q, else 0) and ``base`` ``[128, 8]``
    (16 c + q // 8 in column c). The 2-D ``vector_offset`` of a gather reads its index tile column-major: element e = 128 c + q of a pair's 8 columns is at row q,
    column c, and lands at partition e // 8, page e % 8 of the pair (item (e % 8) // 4, block e % 4). So row q reads page
    q % 8 of the pair, and its row of that page is p = e // 8 = 16 c + q // 8. Small integers, exact in FP32."""
    int32 = nl.int32
    fp32 = nl.float32
    vector = nisa.engine.vector
    pairs = slots // 2
    if rem4:
        return _kv_gather_constants_rem4(sb, slots, page_runs)
    state = {
        "index": sb.tile((_P_MAX, pairs * _KV_PAIR_PAGES), int32),
        "pages": sb.tile((_P_MAX, slots * _GROUP_BLOCKS), fp32),
        "page": sb.tile((_P_MAX, pairs), fp32),
        "select": sb.tile((_P_MAX, _KV_PAIR_PAGES), fp32),
        "base": sb.tile((_P_MAX, _KV_PAIR_PAGES), fp32),
        "run": None,
        "rem4": False,
    }
    part = sb.tile((_P_MAX, 1), int32)
    nisa.iota(dst=part, pattern=[[0, 1]], offset=0, channel_multiplier=1)
    low = sb.tile((_P_MAX, 1), int32)
    nisa.tensor_scalar(dst=low, data=part, op0=nl.bitwise_and, operand0=_KV_PAIR_PAGES - 1, engine=vector)
    low_f = sb.tile((_P_MAX, 1), fp32)
    nisa.tensor_copy(dst=low_f, src=low, engine=vector)
    columns = sb.tile((_P_MAX, _KV_PAIR_PAGES), int32)
    nisa.iota(dst=columns, pattern=[[1, _KV_PAIR_PAGES]], offset=0, channel_multiplier=0)
    nisa.tensor_scalar(dst=state["select"], data=columns, op0=nl.equal, operand0=low_f[0:_P_MAX, 0:1], engine=vector)
    high = sb.tile((_P_MAX, 1), int32)
    nisa.tensor_scalar(dst=high, data=part, op0=nl.right_shift, operand0=3, engine=vector)
    high_f = sb.tile((_P_MAX, 1), fp32)
    nisa.tensor_copy(dst=high_f, src=high, engine=vector)
    rows16 = sb.tile((_P_MAX, _KV_PAIR_PAGES), int32)
    nisa.iota(dst=rows16, pattern=[[_P_MAX // _KV_PAIR_PAGES, _KV_PAIR_PAGES]], offset=0, channel_multiplier=0)
    nisa.tensor_scalar(dst=state["base"], data=rows16, op0=nl.add, operand0=high_f[0:_P_MAX, 0:1], engine=vector)
    if page_runs:
        # The page of block j of a run of ``page_runs`` pages: the run's first page + j (FP32 0 .. page_runs - 1).
        run = sb.tile((_P_MAX, page_runs), int32)
        nisa.iota(dst=run, pattern=[[1, page_runs]], offset=0, channel_multiplier=0)
        state["run"] = sb.tile((_P_MAX, page_runs), fp32)
        nisa.tensor_copy(dst=state["run"], src=run, engine=vector)
    return state


def _kv_gather_constants_rem4(sb, slots: int, page_runs: int) -> dict:
    """``kv_gather`` with ``kv_rem4``: the tiles of the gather indices of a loop of ``slots`` items: ``index``
    ``[128, slots / 2 x 2]`` int32 (``_kv_gather_index``), its FP32 scratch, and two constants: ``select`` ``[128, 2, 8]``
    (1.0 at column 4 (q % 2) + 2 c + q // 64 of index column c on partition q) and ``base`` ``[128, 1]`` ((q // 2) % 32).
    A gather of 2 items has 256 elements of 2,304 bytes; its 2-D ``vector_offset`` reads its 2 index columns column-major:
    element e = 128 c + q sits at row q, column c, and lands at partition P = e // 2 (P = 64 c + q // 2), item e % 2 =
    q % 2 of the pair, so it is piece P % 32 of page P // 32 = 2 c + q // 64 of that item: index 32 x page + (q // 2) % 32.
    Small integers, exact in FP32."""
    int32 = nl.int32
    fp32 = nl.float32
    vector = nisa.engine.vector
    pairs = slots // 2
    state = {
        "index": sb.tile((_P_MAX, pairs * 2), int32),
        "pages": sb.tile((_P_MAX, slots * _GROUP_BLOCKS), fp32),
        "picked": sb.tile((_P_MAX, pairs, 2, _KV_PAIR_PAGES), fp32),
        "page": sb.tile((_P_MAX, pairs, 2), fp32),
        "select": sb.tile((_P_MAX, 2, _KV_PAIR_PAGES), fp32),
        "base": sb.tile((_P_MAX, 1), fp32),
        "run": None,
        "rem4": True,
    }
    part = sb.tile((_P_MAX, 1), int32)
    nisa.iota(dst=part, pattern=[[0, 1]], offset=0, channel_multiplier=1)
    terms = []
    for op0, operand0, op1, operand1 in (
        (nl.bitwise_and, 1, nl.left_shift, 2),
        (nl.right_shift, 6, None, None),
        (nl.right_shift, 1, nl.bitwise_and, _REM4_PARTS - 1),
    ):
        term = sb.tile((_P_MAX, 1), int32)
        nisa.tensor_scalar(dst=term, data=part, op0=op0, operand0=operand0, engine=vector)
        if op1 is not None:
            nisa.tensor_scalar(dst=term, data=term, op0=op1, operand0=operand1, engine=vector)
        term_f = sb.tile((_P_MAX, 1), fp32)
        nisa.tensor_copy(dst=term_f, src=term, engine=vector)
        terms.append(term_f)
    # The selected column of index column c: 4 (q % 2) + q // 64 + 2 c; a column k matches where k - 2 c equals it.
    target = sb.tile((_P_MAX, 1), fp32)
    nisa.tensor_tensor(dst=target, data1=terms[0], data2=terms[1], op=nl.add, engine=vector)
    columns = sb.tile((_P_MAX, 2, _KV_PAIR_PAGES), fp32)
    nisa.iota(dst=columns, pattern=[[-2, 2], [1, _KV_PAIR_PAGES]], offset=0)
    nisa.tensor_scalar(dst=state["select"], data=columns, op0=nl.equal, operand0=target[0:_P_MAX, 0:1], engine=vector)
    nisa.tensor_copy(dst=state["base"], src=terms[2], engine=vector)
    if page_runs:
        run = sb.tile((_P_MAX, page_runs), int32)
        nisa.iota(dst=run, pattern=[[1, page_runs]], offset=0, channel_multiplier=0)
        state["run"] = sb.tile((_P_MAX, page_runs), fp32)
        nisa.tensor_copy(dst=state["run"], src=run, engine=vector)
    return state


def _kv_group_constants(sb, items: int, page_runs: int, consts: dict):
    """``kv_group``: fill the pass-invariant constants of the group gathers of G = ``items`` items in ``consts``
    (``_vog5_constants``). A group gather has 128 G elements of 2,304 bytes; its 2-D ``vector_offset`` reads its G index
    columns column-major: element e = 128 c + q sits at row q, column c, and lands at partition P = e // G, item
    e % G = q % G of the group, so it is piece P % 32 of block b = P // 32 of that item: index 32 x page + P % 32
    (``_kv_group_index``). With page runs of R the page of block b is the run's first page (field b // R) + b % R.
    ``sel`` ``[128, G]``: 1.0 at column q % G of partition q, else 0. ``const`` ``[128, G]``: 32 (b % R) + P % 32 of
    index column c (P % 32 without page runs). Small integers, exact in FP32. No temporaries: the int32 views of the
    two tiles hold the intermediates (``sb`` unused)."""
    int32 = nl.int32
    vector = nisa.engine.vector
    shift = items.bit_length() - 1
    sc_i = consts["sc"].view(int32)
    sel_i = sc_i[0:_P_MAX, 0:items]
    const_i = sc_i[0:_P_MAX, items : 2 * items]
    # P = (128 c + q) >> log2(G) on partition q, column c.
    nisa.iota(dst=const_i, pattern=[[_P_MAX, items]], offset=0, channel_multiplier=1)
    nisa.tensor_scalar(dst=const_i, data=const_i, op0=nl.right_shift, operand0=shift, engine=vector)
    if page_runs:
        # 32 (b % R) = ((P >> 5) & (R - 1)) << 5, in sel's words for now.
        nisa.tensor_scalar(
            dst=sel_i,
            data=const_i,
            op0=nl.right_shift,
            operand0=5,
            op1=nl.bitwise_and,
            operand1=page_runs - 1,
            engine=vector,
        )
        nisa.tensor_scalar(dst=sel_i, data=sel_i, op0=nl.left_shift, operand0=5, engine=vector)
    nisa.tensor_scalar(dst=const_i, data=const_i, op0=nl.bitwise_and, operand0=_REM4_PARTS - 1, engine=vector)
    if page_runs:
        nisa.tensor_tensor(dst=const_i, data1=const_i, data2=sel_i, op=nl.add, engine=vector)
    nisa.tensor_copy(dst=consts["const"], src=const_i, engine=vector)
    # sel: (q - i) & (G - 1) == 0, column i of partition q.
    nisa.iota(dst=sel_i, pattern=[[-1, items]], offset=0, channel_multiplier=1)
    nisa.tensor_scalar(dst=sel_i, data=sel_i, op0=nl.bitwise_and, operand0=items - 1, engine=vector)
    nisa.tensor_scalar(dst=consts["sel"], data=sel_i, op0=nl.equal, operand0=0, engine=vector)


def _kv_group_scratch(sb, slots: int, items: int, consts: dict, first: int) -> dict:
    """``kv_group``: the per-call tile of the group gather indices of a loop of ``slots`` items (groups of G = ``items``,
    items ``first`` ..): ``index`` ``[128, slots - first]`` int32 (G columns a group), with the constants of ``consts``
    (``_vog5_constants``)."""
    groups = (slots - first) // items
    return {
        "items": items,
        "first": first,
        "index": sb.tile((_P_MAX, groups * items), nl.int32),
        "sc": consts["sc"],
    }


def _kv_group_index(state, item_fields, count: int, page_runs: int):
    """``kv_group``: write the group indices of items F .. ``count`` - 1 of an iteration (``_kv_group_constants``, G =
    ``state["items"]``, F = ``state["first"]``) from its item row on every partition (``item_fields`` ``[128, slots x
    fields]`` int32). Index column c of group k on partition q reads item F + G k + q % G at block b of
    ``_kv_group_constants`` (b = c with G 4, c // 2 with G 8), whose page is field b // R of the item with page runs of
    R (else field b), plus b % R: the page is the sum over the G items i of ``sel[q, i]`` x that field of item F + G k +
    i (G multiply-adds into the index tile), and the index 32 x page + ``const``. VectorE, exact in FP32 (fewer than
    2^24 page pieces)."""
    vector = nisa.engine.vector
    items = state["items"]
    first = state["first"]
    groups = (count - first) // items
    # The distinct fields of a group's G index columns: c = f x (G / fields) + s reads field f; one pass a repeat s
    # (the instructions take at most 2 free dimensions).
    fields = _GROUP_BLOCKS // page_runs if page_runs else _GROUP_BLOCKS
    repeat = items // fields
    index = state["index"]
    width = item_fields.shape[1]
    columns = index.shape[1]
    for rep in range(repeat):
        acc = index.ap(pattern=[[columns, _P_MAX], [items, groups], [repeat, fields]], offset=rep)
        for i in range(items):
            source = item_fields.ap(
                pattern=[[width, _P_MAX], [items * _ITEM_FIELDS, groups], [1, fields]],
                offset=(first + i) * _ITEM_FIELDS,
            )
            if i == 0:
                nisa.tensor_scalar(
                    dst=acc, data=source, op0=nl.multiply, operand0=state["sc"][0:_P_MAX, 0:1], engine=vector
                )
            else:
                nisa.scalar_tensor_tensor(
                    dst=acc,
                    data=source,
                    op0=nl.multiply,
                    operand0=state["sc"][0:_P_MAX, i : i + 1],
                    op1=nl.add,
                    operand1=acc,
                )
    flat = index.reshape((_P_MAX, groups, items))[0:_P_MAX, 0:groups, 0:items]
    nisa.scalar_tensor_tensor(
        dst=flat,
        data=flat,
        op0=nl.multiply,
        operand0=float(_REM4_PARTS),
        op1=nl.add,
        operand1=state["sc"].ap(pattern=[[2 * items, _P_MAX], [0, groups], [1, items]], offset=items),
    )


def _kv_gather_index(state, item_fields, count: int, page_runs: int):
    """``kv_gather``: write the gather indices of the ``count`` items of an iteration (``_kv_gather_constants``) from its
    item row on every partition (``item_fields`` ``[128, slots x fields]`` int32): the page of block j of item i (its
    run's first page + j % runs with page runs, else field j), then for pair k the page of row q (``select``: page q % 8
    of the pair) x 128 + 16 c + q // 8 in column c. VectorE, exact in FP32 (fewer than 2^24 page rows)."""
    vector = nisa.engine.vector
    pairs = count // 2
    fields = item_fields.reshape((_P_MAX, item_fields.shape[1] // _ITEM_FIELDS, _ITEM_FIELDS))
    pages = state["pages"].reshape((_P_MAX, state["pages"].shape[1] // _GROUP_BLOCKS, _GROUP_BLOCKS))
    if page_runs:
        runs = _GROUP_BLOCKS // page_runs
        pages_r = state["pages"].reshape((_P_MAX, pages.shape[1], runs, page_runs))
        for page in range(page_runs):
            nisa.tensor_scalar(
                dst=pages_r[0:_P_MAX, 0:count, 0:runs, page],
                data=fields[0:_P_MAX, 0:count, 0:runs],
                op0=nl.add,
                operand0=state["run"][0:_P_MAX, page : page + 1],
                engine=vector,
            )
    else:
        nisa.tensor_copy(
            dst=pages[0:_P_MAX, 0:count, 0:_GROUP_BLOCKS],
            src=fields[0:_P_MAX, 0:count, 0:_GROUP_BLOCKS],
            engine=vector,
        )
    if state["rem4"]:
        _kv_gather_index_rem4(state, pairs)
        return
    flat = state["pages"].reshape((_P_MAX, pages.shape[1] // 2, _KV_PAIR_PAGES))
    # In place: the page of row q stays in column q % 8 of each pair, the others become 0.
    nisa.tensor_tensor(
        dst=flat[0:_P_MAX, 0:pairs, 0:_KV_PAIR_PAGES],
        data1=flat[0:_P_MAX, 0:pairs, 0:_KV_PAIR_PAGES],
        data2=state["select"].ap(pattern=[[_KV_PAIR_PAGES, _P_MAX], [0, pairs], [1, _KV_PAIR_PAGES]], offset=0),
        op=nl.multiply,
        engine=vector,
    )
    nisa.tensor_reduce(
        dst=state["page"][0:_P_MAX, 0:pairs],
        op=nl.add,
        data=flat[0:_P_MAX, 0:pairs, 0:_KV_PAIR_PAGES],
        axis=2,
    )
    nisa.scalar_tensor_tensor(
        dst=state["index"].reshape((_P_MAX, state["index"].shape[1] // _KV_PAIR_PAGES, _KV_PAIR_PAGES))[
            0:_P_MAX, 0:pairs, 0:_KV_PAIR_PAGES
        ],
        data=state["page"].ap(pattern=[[state["page"].shape[1], _P_MAX], [1, pairs], [0, _KV_PAIR_PAGES]], offset=0),
        op0=nl.multiply,
        operand0=float(_BLOCK),
        op1=nl.add,
        operand1=state["base"].ap(pattern=[[_KV_PAIR_PAGES, _P_MAX], [0, pairs], [1, _KV_PAIR_PAGES]], offset=0),
    )


def _kv_gather_index_rem4(state, pairs: int):
    """``kv_rem4``: the 2 index columns of each of ``pairs`` pairs from the pages of their items (``state["pages"]``):
    column c on partition q picks page 2 c + q // 64 of item q % 2 (``select``), then 32 x that page + (q // 2) % 32."""
    vector = nisa.engine.vector
    pages = state["pages"]
    flat = pages.reshape((_P_MAX, pages.shape[1] // _KV_PAIR_PAGES, _KV_PAIR_PAGES))
    picked = state["picked"]
    width = pages.shape[1]
    nisa.tensor_tensor(
        dst=picked[0:_P_MAX, 0:pairs, 0:2, 0:_KV_PAIR_PAGES],
        data1=flat.ap(pattern=[[width, _P_MAX], [_KV_PAIR_PAGES, pairs], [0, 2], [1, _KV_PAIR_PAGES]], offset=0),
        data2=state["select"].ap(
            pattern=[[2 * _KV_PAIR_PAGES, _P_MAX], [0, pairs], [_KV_PAIR_PAGES, 2], [1, _KV_PAIR_PAGES]], offset=0
        ),
        op=nl.multiply,
        engine=vector,
    )
    nisa.tensor_reduce(
        dst=state["page"][0:_P_MAX, 0:pairs, 0:2], op=nl.add, data=picked[0:_P_MAX, 0:pairs, 0:2], axis=3
    )
    nisa.scalar_tensor_tensor(
        dst=state["index"].reshape((_P_MAX, state["index"].shape[1] // 2, 2))[0:_P_MAX, 0:pairs, 0:2],
        data=state["page"][0:_P_MAX, 0:pairs, 0:2],
        op0=nl.multiply,
        operand0=float(_REM4_PARTS),
        op1=nl.add,
        operand1=state["base"].ap(pattern=[[1, _P_MAX], [0, pairs], [0, 2]], offset=0),
    )


def _mask_window(window, buffer: int, queries: int):
    """``kv_pitch576``: return the stationary of the mask window of page buffer ``buffer`` for a block, as a function of
    the block: columns ``buffer x Q`` .. ``buffer x Q + 64 + Q`` of ``window`` ``[128, 4, 64 + buffers x Q]`` (FP8), which
    end at the buffer's Q mask columns (``_write_mask_window``). Its first 64 columns go to rope PSUM rows 0 .. 63, which
    the rope transposes then overwrite; its last Q columns go to the bias rows 64 .. 64 + Q - 1, as the mask columns of a
    608-byte page row do."""
    first = buffer * queries
    return lambda block: window[0:_P_MAX, block, first : first + _ROPE + queries]


def _write_mask_window(window, buffer: int, mask_offsets, threshold):
    """``kv_pitch576``: write the Q mask columns of page buffer ``buffer`` into ``window`` on GpSimd: columns
    ``64 + buffer x Q + j`` of block b on partition p are minus the FP8 bound (``fp8_page_max``) where the slot offset of
    key (b, p) minus j (``mask_offsets``) is at least ``threshold`` (``[128, 1]``), else 0, the values of ``_mask_column``."""
    queries = mask_offsets.shape[2]
    first = _ROPE + buffer * queries
    nisa.tensor_scalar(
        dst=window[0:_P_MAX, 0:_GROUP_BLOCKS, first : first + queries],
        data=mask_offsets,
        op0=nl.greater_equal,
        operand0=threshold,
        op1=nl.multiply,
        operand1=-fp8_page_max(),
        engine=nisa.engine.gpsimd,
    )


def _mask_row(rope_keys, slot_offset, threshold, num_keys, engine=None):
    """Write the mask row below the rope keys: the mask bias where the slot offset is at least ``threshold``. ``engine``
    (for example GpSimd, ``pack_mask_pool``) runs the op; None lets the compiler select the engine."""
    return nisa.tensor_scalar(
        dst=rope_keys[_BIAS_ROW : _BIAS_ROW + 1, 0:num_keys],
        data=slot_offset[_BIAS_ROW : _BIAS_ROW + 1, 0:num_keys],
        op0=nl.greater_equal,
        operand0=threshold,
        op1=nl.multiply,
        operand1=_MASK_BIAS,
        **({} if engine is None else {"engine": engine}),
    )


def _gather_index(dst, table, group_pages, page_select, row_offsets, num_pages: int, pool: bool = False):
    """Write the gather indices of ``loop_page_gather`` of the table row ``table`` (``[128, 4 G]`` int32, the row on every
    partition) into ``dst`` (``[128, G, 4]`` int32): for partition q and column c of group g, (clamp(entry 4 g + q % 4)
    << 7) + 32 c + q // 4, the clamp into the ``num_pages`` pages. ``group_pages`` (``[128, G]`` int32) is the scratch
    of the pages; ``page_select`` and ``row_offsets`` are the constants of ``_gather_constants``. ``pool``: every op on
    GpSimd, on a table already clamped into the pool (``_clamped_table``; GpSimd has no minimum / maximum on Trn2), with
    ``* 128`` for the shift and ``+`` for the row (the bitwise ops run on VectorE): each value is an integer below 2**24
    and the row is below 128, so the indices are the same. Return the last op."""
    vector = nisa.engine.vector
    gpsimd = nisa.engine.gpsimd
    groups = group_pages.shape[1]
    nisa.nc_n_gather(dst=group_pages, data=table, indices=page_select)
    if pool:
        # An int32 multiply by an int on GpSimd is its native integer form (TensorScalarAddr).
        nisa.tensor_scalar(
            dst=group_pages, data=group_pages, op0=nl.multiply, operand0=1 << _GATHER_ROW_BITS, engine=gpsimd
        )
        return nisa.tensor_tensor(
            dst=dst,
            data1=group_pages.ap(pattern=[[groups, _P_MAX], [1, groups], [0, _GROUP_BLOCKS]], offset=0),
            data2=row_offsets.ap(pattern=[[_GROUP_BLOCKS, _P_MAX], [0, groups], [1, _GROUP_BLOCKS]], offset=0),
            op=nl.add,
            engine=gpsimd,
        )
    # A padded table entry is clamped into the pool, as the page loads clamp it.
    nisa.tensor_scalar(
        dst=group_pages,
        data=group_pages,
        op0=nl.minimum,
        operand0=float(num_pages - 1),
        op1=nl.maximum,
        operand1=0.0,
        engine=vector,
    )
    nisa.tensor_scalar(dst=group_pages, data=group_pages, op0=nl.left_shift, operand0=_GATHER_ROW_BITS, engine=vector)
    # A bitwise op runs on VectorE only (GpSimd has no bitwise ALU ops).
    return nisa.tensor_tensor(
        dst=dst,
        data1=group_pages.ap(pattern=[[groups, _P_MAX], [1, groups], [0, _GROUP_BLOCKS]], offset=0),
        data2=row_offsets.ap(pattern=[[_GROUP_BLOCKS, _P_MAX], [0, groups], [1, _GROUP_BLOCKS]], offset=0),
        op=nl.bitwise_or,
        engine=vector,
    )


def _clamped_table(block_table, tokens: int, local_blocks: int, num_pages: int, tag: str, sb):
    """Return the table entries of the ``tokens`` sequences (``block_table`` ``[B, >= local_blocks]`` int32) clamped into
    the ``num_pages`` pages of the pool, as the page loads clamp them, in an HBM tensor ``[B, local_blocks]`` int32: one SP
    load, one VectorE clamp and one SP store (``loop_page_gather`` "carry_pool"). The SBUF tile returns to
    ``sb``."""
    hwdge = nisa.dge_mode.hwdge
    sync = nisa.engine.sync
    clamped = nl.ndarray((tokens, local_blocks), dtype=nl.int32, buffer=nl.hbm, name=f"{tag}_mla_gather_table_clamped")
    with sb.scope("gather_table_clamp"):
        entries = sb.tile((tokens, local_blocks), nl.int32)
        nisa.dma_copy(
            dst=entries,
            src=block_table[0:tokens, 0:local_blocks],
            dge_mode=hwdge,
            engine=sync,
            name=f"{tag}_mla_gather_table_clamp_load",
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
        nisa.dma_copy(dst=clamped, src=entries, dge_mode=hwdge, engine=sync, name=f"{tag}_mla_gather_table_clamp_store")
    return clamped


def _gather_constants(page_select, row_offsets, sb):
    """Write the constants of the gather indices of ``loop_page_gather``: ``page_select`` ``[128, G]`` uint32, the entry
    4 g + q % 4 of the table row that partition q reads for group g, and ``row_offsets`` ``[128, 4]`` int32, the page row
    32 c + q // 4 of column c of partition q of a group's index tile. The lane scratch returns to ``sb``."""
    vector = nisa.engine.vector
    groups = page_select.shape[1]
    with sb.scope("gather_constants"):
        lanes = sb.tile((_P_MAX, 1), nl.int32)
        nisa.iota(dst=lanes, pattern=[[0, 1]], offset=0, channel_multiplier=1)
        lane_page = sb.tile((_P_MAX, 1), nl.int32)
        nisa.tensor_scalar(dst=lane_page, data=lanes, op0=nl.bitwise_and, operand0=_GROUP_BLOCKS - 1, engine=vector)
        lane_row = sb.tile((_P_MAX, 1), nl.int32)
        nisa.tensor_scalar(dst=lane_row, data=lanes, op0=nl.right_shift, operand0=2, engine=vector)
        group_first = sb.tile((_P_MAX, groups), nl.int32)
        nisa.iota(dst=group_first, pattern=[[_GROUP_BLOCKS, groups]], offset=0)
        nisa.tensor_tensor(
            dst=page_select,
            data1=group_first,
            data2=lane_page.ap(pattern=[[1, _P_MAX], [0, groups]], offset=0),
            op=nl.add,
            engine=vector,
        )
        column_rows = sb.tile((_P_MAX, _GROUP_BLOCKS), nl.int32)
        nisa.iota(dst=column_rows, pattern=[[_P_MAX // _GROUP_BLOCKS, _GROUP_BLOCKS]], offset=0)
        nisa.tensor_tensor(
            dst=row_offsets,
            data1=column_rows,
            data2=lane_row.ap(pattern=[[1, _P_MAX], [0, _GROUP_BLOCKS]], offset=0),
            op=nl.add,
            engine=vector,
        )


def _page_tile(sb, blocks, dtype, pair_pages):
    """Return the SBUF tile of ``blocks`` pages: row-major ``[128, blocks, 576]`` or pair rows (``pair_rows_tile``)."""
    if pair_pages:
        return pair_rows_tile(sb, blocks, dtype)
    return sb.tile((_P_MAX, blocks, _CACHE_WIDTH), dtype)


def _page_view(rows, block, pair_pages):
    """Return the SBUF view of page ``block`` of a page tile."""
    if not pair_pages:
        return rows[0:_P_MAX, block, 0:_CACHE_WIDTH]
    half = (block % 2) * (_BLOCK // 2)
    return rows[half : half + _BLOCK // 2, block // 2, 0 : 2 * _CACHE_WIDTH]


def _mx_emulate_rows(dst, src, sb):
    """Write the MXFP8 values of ``src`` (``[R, 512]`` FP32, the latent of R rows) into ``dst`` (``[R, 512]`` BF16).

    The 32 values of a scale are latent ``128 l + 8 s + r`` for l < 4, r < 8 (the stationary x4 layout of an MXFP8
    query, lane l = latent tile l). The scale is the quantize_mx rule, 2 ** (floor(log2 max |x|) - 7), as FP32 bits from
    the exponent field; each value rounds to e4m3, nearest even, by an FP8 cast. Return the last instruction.
    """
    fp32, i32 = nl.float32, nl.int32
    rows = src.shape[0]
    lanes, octets, width = MX_LANES, _P_MAX // 8, 8
    shape = (rows, lanes, octets, width)
    x = sb.tile(shape, fp32)
    nisa.tensor_copy(dst=x.reshape((rows, _KV_LORA)), src=src)
    neg = sb.tile(shape, fp32)
    nisa.tensor_scalar(dst=neg, data=x, op0=nl.multiply, operand0=-1.0)
    mag = sb.tile(shape, fp32)
    nisa.tensor_tensor(dst=mag, data1=x, data2=neg, op=nl.maximum)
    lane_max = sb.tile((rows, octets, lanes), fp32)
    nisa.tensor_reduce(dst=lane_max, op=nl.maximum, data=mag.permute((0, 2, 1, 3)), axis=3)
    amax = sb.tile((rows, octets), fp32)
    nisa.tensor_reduce(dst=amax, op=nl.maximum, data=lane_max, axis=2)
    exponent = sb.tile((rows, octets), i32)
    nisa.tensor_scalar(
        dst=exponent, data=amax.view(i32), op0=nl.right_shift, operand0=23, op1=nl.bitwise_and, operand1=255
    )
    # 1 / scale = 2 ** (134 - e) has the biased exponent 261 - e; scale has e - 7. An all-zero group gets finite
    # bit patterns, and its values stay 0.
    inverse = sb.tile((rows, octets), i32)
    nisa.tensor_scalar(dst=inverse, data=exponent, op0=nl.multiply, operand0=-1, op1=nl.add, operand1=261)
    nisa.tensor_scalar(dst=inverse, data=inverse, op0=nl.left_shift, operand0=23)
    scale = sb.tile((rows, octets), i32)
    # One tensor_scalar takes arithmetic or bitwise ops, not both.
    nisa.tensor_scalar(dst=scale, data=exponent, op0=nl.subtract, operand0=7)
    nisa.tensor_scalar(dst=scale, data=scale, op0=nl.left_shift, operand0=23)

    def per_group(t):
        return t.view(fp32).expand_dim(2).broadcast(2, lanes).expand_dim(3).broadcast(3, width)

    scaled = sb.tile((rows, octets, lanes, width), fp32)
    nisa.tensor_tensor(dst=scaled, data1=x.permute((0, 2, 1, 3)), data2=per_group(inverse), op=nl.multiply)
    codes = sb.tile((rows, octets, lanes, width), fp8_page_dtype())
    nisa.tensor_copy(dst=codes, src=scaled)
    values = sb.tile((rows, octets, lanes, width), fp32)
    nisa.tensor_copy(dst=values, src=codes)
    out = sb.tile(shape, fp32)
    nisa.tensor_tensor(dst=out.permute((0, 2, 1, 3)), data1=values, data2=per_group(scale), op=nl.multiply)
    return nisa.tensor_copy(dst=dst, src=out.reshape((rows, _KV_LORA)))


_VALUE_CALLS = itertools.count()


class MxKvB(NamedTuple):
    """The MXFP8 ``mla_kv_b_proj`` in SBUF (``_load_mx_kv_b``), of the decode and of its verify twin."""

    absorb: nl.NkiTensor
    """``[128, 512]`` x4: the key absorb weight, head ``h`` at partitions ``32 h ..`` (one quadrant row tile)."""
    absorb_scale: nl.NkiTensor
    """``[128, 512]`` uint8: its scales, 4 rows at the start of each quadrant."""
    value: nl.NkiTensor
    """``[128, 384]`` x4: the value weight, one hT_quad group (lane ``k`` is latent tile ``k``), columns (head, dim)."""
    value_scale: nl.NkiTensor
    """``[128, 384]`` uint8: its scales (the quadrant rows of group 0)."""


class MxOChunk(NamedTuple):
    """One chunk of the MXFP8 o_proj in SBUF, of the decode and of its verify twin."""

    data: nl.NkiTensor
    """``[128, 4 * chunk_tiles * 128]`` uint8: the padded-lane weight bytes of the chunk's hidden columns, or None
    (``o_proj_stream``: the chunk loads before its matmuls)."""
    scale: nl.NkiTensor
    """``[128, 7168]`` uint8: the scales of every hidden column (quadrant rows), shared by the chunks of one load."""


def _load_mx_kv_b(mla_kv_b_proj, mla_kv_b_proj_scale, sb, tag, keep=False):
    """Load the MXFP8 ``mla_kv_b_proj`` ``[128, 4 * (512 + 384)]`` and its compact scales ``[16, 512 + 384]``.

    ``keep`` puts the tiles on the heap (the verify body: its tail reads the value part after the projection scope).
    """
    width = mla_kv_b_proj.shape[1]
    cols = mla_kv_b_proj_scale.shape[1]
    keep_kw = {"keep": True} if keep else {}
    data = sb.tile((_P_MAX, width), mla_kv_b_proj.dtype, **keep_kw)
    gate_weight_dma(
        nisa.dma_copy(
            dst=data,
            src=mla_kv_b_proj,
            dge_mode=nisa.dge_mode.hwdge,
            engine=nisa.engine.sync,
            name=f"{tag}_mla_w_kv_b_mx",
        )
    )
    scales = sb.tile((_P_MAX, cols), nl.uint8, **keep_kw)

    def load(dst, src, quadrant):
        return gate_weight_dma(
            nisa.dma_copy(
                dst=dst,
                src=src,
                dge_mode=nisa.dge_mode.hwdge,
                engine=nisa.engine.sync,
                name=f"{tag}_mla_w_kv_b_scale_{quadrant}",
            )
        )

    load_mx_compact_scales(mla_kv_b_proj_scale, scales, load)
    x4 = data.view(nl.float8_e4m3fn_x4)
    return MxKvB(
        x4[0:_P_MAX, 0:_KV_LORA],
        scales[0:_P_MAX, 0:_KV_LORA],
        x4[0:_P_MAX, _KV_LORA:cols],
        scales[0:_P_MAX, _KV_LORA:cols],
    )


def _absorb_mx_operand(q_nope_src, rows, padded, tag, sb, copy=None):
    """Return the MXFP8 quadrant operand of the q_nope rows ``[rows, 3 * 128]`` in PSUM or SBUF
    (``quantize_quadrant_rows_mx``: one BF16 read-out in lane-major order, then one TensorE transpose a lane)."""
    return quantize_quadrant_rows_mx(
        q_nope_src, rows, _LOCAL_HEADS, padded, tag, lambda shape, dtype, name: sb.tile(shape, dtype, name), copy
    )


def _value_mx(latent_t, kv_b_w, value_psum, rows, tag, sb):
    """Write ``latent @ value`` with the MXFP8 ``kv_b_w`` (``MxKvB``) to ``value_psum`` ``[128, H, rows]``.

    The latent ``[128, 4, H, rows]`` of each (head, row) is quantized (one hT_quad group, the 4 latent tiles in the
    lanes) and each head is one nc_matmul_mx, the value weight stationary (as the BF16 form).
    """
    heads = latent_t.shape[2]
    padded = rows + rows % 2
    lat_data, lat_scale = quantize_heads_mx(
        latent_t, rows, padded, tag, lambda shape, dtype, name: sb.tile(shape, dtype, name)
    )
    for local_head in range(heads):
        cols = slice(local_head * _V_HEAD, (local_head + 1) * _V_HEAD)
        nisa.nc_matmul_mx(
            dst=value_psum[0:_P_MAX, local_head, 0:rows],
            stationary=kv_b_w.value[0:_P_MAX, cols],
            moving=lat_data[0:_P_MAX, local_head, 0:rows],
            stationary_scale=kv_b_w.value_scale[0:_P_MAX, cols],
            moving_scale=lat_scale[0:_P_MAX, local_head, 0:rows],
            accumulate=False,
        )


def _o_proj_mx_quantize(gated, rows, tag, sb):
    """Return the MXFP8 moving operand of the MXFP8 o_proj: the gated value ``[128, heads, rows]`` quantized per row in
    the padded-lane order (the heads in lanes 0 .. 2, lane 3 zero)."""
    return quantize_hT_mx(
        gated, tag, rows + rows % 2, lambda shape, dtype, name: sb.tile(shape, dtype, name), tile_major=True
    )


def _o_proj_mx_chunk(out_banks, tiles_per_bank, chunk_start, chunk_tiles, chunk, o_data, o_scale, rows):
    """Run the MXFP8 o_proj of one chunk (``MxOChunk``): each hidden tile is one nc_matmul_mx over the 512 lane rows,
    the weight stationary, into the hT PSUM banks ``out_banks`` (the layout of the BF16 form)."""
    w_x4 = chunk.data.view(nl.float8_e4m3fn_x4)
    for chunk_tile in range(chunk_tiles):
        hidden_tile = chunk_start + chunk_tile
        bank_start, _bank_tiles, out_psum = out_banks[hidden_tile // tiles_per_bank]
        col = hidden_tile * _P_MAX
        nisa.nc_matmul_mx(
            dst=out_psum[0:_P_MAX, hidden_tile - bank_start, 0:rows],
            stationary=w_x4[0:_P_MAX, chunk_tile * _P_MAX : (chunk_tile + 1) * _P_MAX],
            moving=o_data[0:_P_MAX, 0, 0:rows],
            stationary_scale=chunk.scale[0:_P_MAX, col : col + _P_MAX],
            moving_scale=o_scale[0:_P_MAX, 0, 0:rows],
            accumulate=False,
        )


def _load_mx_o_scales(mla_o_proj_scale, tag, state_reduce_scatter, sb, early=False):
    """Load the compact scales ``[16, 7168]`` of the MXFP8 o_proj into one ``[128, 7168]`` tile (4 DMAs, gated on
    the state reduce-scatter as the chunks; with ``early`` each also waits for it on hardware)."""
    o_scale = sb.tile((_P_MAX, _HIDDEN), nl.uint8)

    def load(dst, src, quadrant):
        dma = gate_weight_dma(
            nisa.dma_copy(
                dst=dst,
                src=src,
                dge_mode=nisa.dge_mode.hwdge,
                engine=nisa.engine.sync,
                name=f"{tag}_mla_w_o_scale_{quadrant}",
            ),
            *(() if state_reduce_scatter is None else (state_reduce_scatter,)),
        )
        if early:
            dma.depends_on(state_reduce_scatter, kind=nisa.dependency.runtime)
        return dma

    load_mx_compact_scales(mla_o_proj_scale, o_scale, load)
    return o_scale


def _load_o_proj_chunk_mx(mla_o_proj, chunk_start, chunk_tiles, tag, state_reduce_scatter, sb, o_scale, early=False):
    """Load one MXFP8 o_proj chunk ``[128, 4 * chunk_tiles * 128]`` (``MxOChunk``), gated as ``_load_o_proj_chunk``."""
    width = chunk_tiles * _P_MAX * MX_LANES
    o_w = sb.tile((_P_MAX, width), nl.uint8)
    first = chunk_start * _P_MAX * MX_LANES
    dma = gate_weight_dma(
        nisa.dma_copy(
            dst=o_w,
            src=mla_o_proj[0, 0:_P_MAX, first : first + width],
            dge_mode=nisa.dge_mode.hwdge,
            engine=nisa.engine.sync,
            name=f"{tag}_mla_w_o_{chunk_start}",
        ),
        *(() if state_reduce_scatter is None else (state_reduce_scatter,)),
    )
    if early:
        dma.depends_on(state_reduce_scatter, kind=nisa.dependency.runtime)
    return MxOChunk(o_w, o_scale)


def _load_mx_q_b(mla_q_b_proj, mla_q_b_proj_scale, sb, tag):
    """Load the MXFP8 q_b weight ``[G, 128, 4 * 576]`` as ``[128, G, 576]`` x4 and the copies of its scales."""
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

    scales = load_mx_scales(mla_q_b_proj_scale, lambda shape, copy: sb.tile(shape, nl.uint8), load)
    return q_b_w.view(nl.float8_e4m3fn_x4), scales


def _load_mx_in_proj(mla_in_proj, mla_in_proj_scale, sb, tag):
    """Load the MXFP8 qkv columns of ``mla_in_proj`` ``[128, 14, 4 * 66]`` and the copies of its scales."""
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

    scales = load_mx_scales(mla_in_proj_scale, lambda shape, copy: sb.tile(shape, nl.uint8), load)
    return ("mx", qkv_w, scales)


def _load_mx_gate(mla_in_proj, qkv_entry, sb, tag, engine, after, wgate_early, gate_loads):
    """Load the MXFP8 gate columns of ``mla_in_proj`` ``[128, 14, 4 * 384]`` in 7 DMAs; return the gate entry."""
    groups = mla_in_proj.shape[0]
    gate_w = sb.tile((_P_MAX, groups, MX_LANES * _GATE_LOCAL), mla_in_proj.dtype)
    per_dma = 2
    for first in range(0, groups, per_dma):
        count = min(per_dma, groups - first)
        dma = gate_weight_dma(
            nisa.dma_copy(
                dst=gate_w[0:_P_MAX, first : first + count, 0 : MX_LANES * _GATE_LOCAL],
                src=mla_in_proj[first : first + count, 0:_P_MAX, MX_LANES * _QKV_LOCAL : MX_LANES * _IN_WIDTH].permute(
                    (1, 0, 2)
                ),
                dge_mode=nisa.dge_mode.hwdge,
                engine=engine,
                name=f"{tag}_mla_w_gate_mx_{first}",
            ),
            after,
        )
        if wgate_early != "off":
            dma.depends_on(after, kind=nisa.dependency.runtime)
        if wgate_early == "scalar":
            gate_loads.append(dma)
    return ("mx", gate_w, qkv_entry[2])


def _qkv_projection_mx(qkv_psum, x_mx, qkv_entry, padded):
    """``[Bp, 66] = x @ W_qkv`` from the MXFP8 hidden (stationary, ``padded`` columns) and the MXFP8 qkv columns."""
    _, qkv_w, scales = qkv_entry
    x_data, x_scale = x_mx
    w_x4 = qkv_w.view(nl.float8_e4m3fn_x4)
    for group in range(qkv_w.shape[1]):
        nisa.nc_matmul_mx(
            dst=qkv_psum[0:padded, 0:_QKV_LOCAL],
            stationary=x_data[0:_P_MAX, group, 0:padded],
            moving=w_x4[0:_P_MAX, group, 0:_QKV_LOCAL],
            stationary_scale=x_scale[0:_P_MAX, group, 0:padded],
            moving_scale=mx_scale_view(scales, group, 0, _QKV_LOCAL),
            accumulate=group > 0,
        )


def _gate_projection(x_hT, gate_chunks, tokens, x_mx=None):
    """Return the FP32 PSUM gate projection ``[128, 3, B]``: element ``(v, h, b)`` is ``g[b, 128 h + v]``.

    Each head accumulates over all hidden tiles before the next head starts, so no two accumulation groups
    interleave in the PSUM bank. With ``x_mx`` (the MXFP8 hidden), ``gate_chunks`` holds the one MXFP8 gate entry.
    """
    gate_psum = nl.ndarray((_P_MAX, _LOCAL_HEADS, tokens), dtype=nl.float32, buffer=nl.psum)
    if x_mx is not None:
        _, gate_w, scales = gate_chunks[0]
        x_data, x_scale = x_mx
        w_x4 = gate_w.view(nl.float8_e4m3fn_x4)
        for local_head in range(_LOCAL_HEADS):
            cols = slice(local_head * _V_HEAD, (local_head + 1) * _V_HEAD)
            for group in range(gate_w.shape[1]):
                nisa.nc_matmul_mx(
                    dst=gate_psum[0:_P_MAX, local_head, 0:tokens],
                    stationary=w_x4[0:_P_MAX, group, cols],
                    moving=x_data[0:_P_MAX, group, 0:tokens],
                    stationary_scale=mx_scale_view(scales, group, _QKV_LOCAL + cols.start, _V_HEAD),
                    moving_scale=x_scale[0:_P_MAX, group, 0:tokens],
                    accumulate=group > 0,
                )
        return gate_psum
    for local_head in range(_LOCAL_HEADS):
        for chunk_start, chunk_tiles, gate_w in gate_chunks:
            for chunk_tile in range(chunk_tiles):
                hidden_tile = chunk_start + chunk_tile
                nisa.nc_matmul(
                    dst=gate_psum[0:_P_MAX, local_head, 0:tokens],
                    stationary=gate_w[0:_P_MAX, chunk_tile, local_head * _V_HEAD : (local_head + 1) * _V_HEAD],
                    moving=x_hT[0:_P_MAX, 0:tokens, hidden_tile],
                    accumulate=hidden_tile > 0,
                )
    return gate_psum


def _value_projection(local_state, kv_b_w, value_all, tokens, sb):
    """Normalize the merged state ``[3, B, 513]`` and write ``latent @ value`` to ``value_all`` (``[128, 3, B]``)."""
    bf16 = nl.bfloat16
    fp32 = nl.float32
    parts = local_state if isinstance(local_state, StateParts) else None
    inverse_sum = sb.tile((_LOCAL_HEADS, tokens, 1), fp32)
    sums = local_state[0:_LOCAL_HEADS, 0:tokens, _KV_LORA : _KV_LORA + 1] if parts is None else parts.sums
    nisa.reciprocal(dst=inverse_sum, data=sums)
    latent = sb.tile((_LOCAL_HEADS, tokens, _KV_LORA), bf16)
    nisa.tensor_tensor(
        dst=latent,
        data1=local_state[0:_LOCAL_HEADS, 0:tokens, 0:_KV_LORA] if parts is None else parts.o,
        data2=inverse_sum.broadcast(2, _KV_LORA),
        op=nl.multiply,
    )
    # Each [3, 128] transpose writes 4 columns, so every BF16 block starts on a 4-byte boundary.
    latent_psum = nl.ndarray((_P_MAX, _KV_LORA_TILES, tokens, 4), dtype=bf16, buffer=nl.psum)
    for token in range(tokens):
        for latent_tile in range(_KV_LORA_TILES):
            nisa.nc_transpose(
                dst=latent_psum[0:_P_MAX, latent_tile, token, 0:_LOCAL_HEADS],
                data=latent[0:_LOCAL_HEADS, token, latent_tile * _P_MAX : (latent_tile + 1) * _P_MAX],
            )
    latent_t = sb.tile((_P_MAX, _KV_LORA_TILES, _LOCAL_HEADS, tokens), bf16)
    nisa.tensor_copy(
        dst=latent_t.permute((0, 1, 3, 2)),
        src=latent_psum[0:_P_MAX, 0:_KV_LORA_TILES, 0:tokens, 0:_LOCAL_HEADS],
    )
    value_psum = nl.ndarray((_P_MAX, _LOCAL_HEADS, tokens), dtype=fp32, buffer=nl.psum)
    if isinstance(kv_b_w, MxKvB):
        _value_mx(latent_t, kv_b_w, value_psum, tokens, f"mla_dv{next(_VALUE_CALLS)}", sb)
        nisa.tensor_copy(dst=value_all, src=value_psum)
        return
    for local_head in range(_LOCAL_HEADS):
        for latent_tile in range(_KV_LORA_TILES):
            value_col = _ABSORB_WIDTH + (latent_tile * _LOCAL_HEADS + local_head) * _V_HEAD
            nisa.nc_matmul(
                dst=value_psum[0:_P_MAX, local_head, 0:tokens],
                stationary=kv_b_w[0:_P_MAX, value_col : value_col + _V_HEAD],
                moving=latent_t[0:_P_MAX, latent_tile, local_head, 0:tokens],
                accumulate=latent_tile > 0,
            )
    nisa.tensor_copy(dst=value_all, src=value_psum)


def _add_current_rows(state, current_hbm, tokens: int, rows: bool, tag: str, sb):
    """Add the current key of each of ``tokens`` sequences once, with its whole weight ``2**-93``, to the merged FP32
    state of each local head (``_mla_decode_body`` ``pack_current_after_rs``).

    ``current_hbm`` is ``[tokens, 512]`` BF16 HBM: the current latent rows. With ``rows`` the state is ``[3 S, 513]``,
    row ``h S + b`` local head h of sequence b (``stage_post``); else it is ``[3, S, 513]``. The weight is a power of
    two, so ``2**-93`` times a row is exact, and the numerator and the sum take one FP32 add each.
    """
    bf16 = nl.bfloat16
    weight = 2.0**_CURRENT_WEIGHT_LOG2
    if rows:
        head_rows = _LOCAL_HEADS * tokens
        current = sb.tile((head_rows, _KV_LORA), bf16)
        for local_head in range(_LOCAL_HEADS):
            nisa.dma_copy(
                dst=current[local_head * tokens : (local_head + 1) * tokens, 0:_KV_LORA],
                src=current_hbm[0:tokens, 0:_KV_LORA],
                dge_mode=nisa.dge_mode.hwdge,
                engine=nisa.engine.sync,
                name=f"{tag}_mla_current_rows_load_{local_head}",
            )
        numerator = state[0:head_rows, 0:_KV_LORA]
        total = state[0:head_rows, _KV_LORA : _KV_LORA + 1]
    else:
        current = sb.tile((_LOCAL_HEADS, tokens, _KV_LORA), bf16)
        nisa.dma_copy(
            dst=current.reshape((_LOCAL_HEADS, tokens * _KV_LORA)),
            src=current_hbm.reshape((1, tokens * _KV_LORA)).broadcast(0, _LOCAL_HEADS),
            dge_mode=nisa.dge_mode.hwdge,
            engine=nisa.engine.sync,
            name=f"{tag}_mla_current_rows_load",
        )
        numerator = state[0:_LOCAL_HEADS, 0:tokens, 0:_KV_LORA]
        total = state[0:_LOCAL_HEADS, 0:tokens, _KV_LORA : _KV_LORA + 1]
    # scalar_tensor_tensor takes no engine argument.
    nisa.scalar_tensor_tensor(
        dst=numerator,
        data=current,
        op0=nl.multiply,
        operand0=weight,
        op1=nl.add,
        operand1=numerator,
    )
    nisa.tensor_scalar(dst=total, data=total, op0=nl.add, operand0=weight, engine=nisa.engine.vector)


def _value_projection_rows(state_rows, kv_b_w, value_all, tokens, sb):
    """``_value_projection`` of ``tokens`` sequences whose merged state holds one row per (head, sequence).

    ``state_rows`` is ``[3 S, 513]``: row ``h S + b`` is local head h of sequence b. The normalization runs on 3 S
    partitions and each latent tile takes one ``[3 S, 128]`` transpose, in place of a pass on 3 partitions and S
    ``[3, 128]`` transposes for each tile. Each element takes the same reciprocal and multiply, the transposes move
    the same BF16 values, and each matmul column holds the same latent, so ``value_all`` (``[128, 3, S]``) is bitwise
    that of ``_value_projection``.
    """
    bf16 = nl.bfloat16
    fp32 = nl.float32
    head_rows = _LOCAL_HEADS * tokens
    inverse_sum = sb.tile((head_rows, 1), fp32)
    nisa.reciprocal(dst=inverse_sum, data=state_rows[0:head_rows, _KV_LORA : _KV_LORA + 1])
    latent = sb.tile((head_rows, _KV_LORA), bf16)
    nisa.tensor_tensor(
        dst=latent,
        data1=state_rows[0:head_rows, 0:_KV_LORA],
        data2=inverse_sum.broadcast(1, _KV_LORA),
        op=nl.multiply,
    )
    latent_psum = nl.ndarray((_P_MAX, _KV_LORA_TILES, head_rows), dtype=bf16, buffer=nl.psum)
    for latent_tile in range(_KV_LORA_TILES):
        nisa.nc_transpose(
            dst=latent_psum[0:_P_MAX, latent_tile, 0:head_rows],
            data=latent[0:head_rows, latent_tile * _P_MAX : (latent_tile + 1) * _P_MAX],
        )
    latent_t = sb.tile((_P_MAX, _KV_LORA_TILES, _LOCAL_HEADS, tokens), bf16)
    nisa.tensor_copy(dst=latent_t, src=latent_psum.reshape_dim(2, (_LOCAL_HEADS, tokens)))
    value_psum = nl.ndarray((_P_MAX, _LOCAL_HEADS, tokens), dtype=fp32, buffer=nl.psum)
    if isinstance(kv_b_w, MxKvB):
        _value_mx(latent_t, kv_b_w, value_psum, tokens, f"mla_dv{next(_VALUE_CALLS)}", sb)
        nisa.tensor_copy(dst=value_all, src=value_psum)
        return
    for local_head in range(_LOCAL_HEADS):
        for latent_tile in range(_KV_LORA_TILES):
            value_col = _ABSORB_WIDTH + (latent_tile * _LOCAL_HEADS + local_head) * _V_HEAD
            nisa.nc_matmul(
                dst=value_psum[0:_P_MAX, local_head, 0:tokens],
                stationary=kv_b_w[0:_P_MAX, value_col : value_col + _V_HEAD],
                moving=latent_t[0:_P_MAX, latent_tile, local_head, 0:tokens],
                accumulate=latent_tile > 0,
            )
    nisa.tensor_copy(dst=value_all, src=value_psum)


def _o_proj_chunks(
    mla_o_proj, tag, state_reduce_scatter, sb, o_proj_stream, o_proj_ring, early=False, mla_o_proj_scale=None
):
    """Return the o_proj weight chunks of the layer as ``(chunk_start, chunk_tiles, tile, ring_tile)``: every chunk's
    loaded tile without ``o_proj_stream``; with it, None tiles (each chunk loads before its matmuls), or with
    ``o_proj_ring`` the first ``o_proj_ring`` chunks loaded into the ring's tiles (``ring_tile``). With ``early``
    (``o_proj_preload``, which has no ``o_proj_stream``), ``state_reduce_scatter`` is the first read of the q_a | kv_a
    gather result, and each chunk load also waits for it on hardware. ``mla_o_proj_scale`` (the MXFP8 o_proj): the
    tiles are ``MxOChunk`` (data None with ``o_proj_stream``) that share one scale tile, loaded here; no ring."""
    if mla_o_proj_scale is not None:
        kernel_assert(not o_proj_ring, "the MXFP8 o_proj takes no o_proj_ring")
        o_scale = _load_mx_o_scales(mla_o_proj_scale, tag, state_reduce_scatter, sb, early)
        mx_chunks = []
        for chunk_start in range(0, _HIDDEN_TILES, _O_PROJ_CHUNK_TILES):
            chunk_tiles = min(_O_PROJ_CHUNK_TILES, _HIDDEN_TILES - chunk_start)
            if o_proj_stream:
                chunk = MxOChunk(None, o_scale)
            else:
                chunk = _load_o_proj_chunk_mx(
                    mla_o_proj, chunk_start, chunk_tiles, tag, state_reduce_scatter, sb, o_scale, early
                )
            mx_chunks.append((chunk_start, chunk_tiles, chunk, None))
        return mx_chunks
    o_w_chunks = []
    for chunk_start in range(0, _HIDDEN_TILES, _O_PROJ_CHUNK_TILES):
        chunk_tiles = min(_O_PROJ_CHUNK_TILES, _HIDDEN_TILES - chunk_start)
        if o_proj_stream and len(o_w_chunks) < o_proj_ring:
            # The ring's first chunks load here, as the non-streamed form loads its chunks.
            ring_tile = sb.tile((_P_MAX, _LOCAL_HEADS, _O_PROJ_CHUNK_TILES * _P_MAX), nl.bfloat16)
            o_w = _load_o_proj_chunk(mla_o_proj, chunk_start, chunk_tiles, tag, state_reduce_scatter, sb, dst=ring_tile)
            o_w_chunks.append((chunk_start, chunk_tiles, o_w, ring_tile))
            continue
        if o_proj_stream:
            o_w_chunks.append((chunk_start, chunk_tiles, None, None))
            continue
        o_w = sb.tile((_P_MAX, _LOCAL_HEADS, chunk_tiles * _P_MAX), nl.bfloat16)
        cols = mla_o_proj[0:_GATE_LOCAL, chunk_start * _P_MAX : (chunk_start + chunk_tiles) * _P_MAX]
        dma = gate_weight_dma(
            nisa.dma_copy(
                dst=o_w,
                src=cols.reshape_dim(0, (_LOCAL_HEADS, _P_MAX)).permute((1, 0, 2)),
                dge_mode=nisa.dge_mode.hwdge,
                engine=nisa.engine.sync,
                name=f"{tag}_mla_w_o_{chunk_start}",
            ),
            state_reduce_scatter,
        )
        if early:
            dma.depends_on(state_reduce_scatter, kind=nisa.dependency.runtime)
        o_w_chunks.append((chunk_start, chunk_tiles, o_w, None))
    return o_w_chunks


def _o_proj_partial(o_w_chunks, gated, out_tokens, mla_o_proj, tag, state_reduce_scatter, sb, o_proj_stream):
    """Return the o_proj partial ``[128, out_tokens, 56]`` BF16 of ``gated`` (``[128, local_heads, out_tokens]``) in hT
    layout, the weight tiles of ``o_w_chunks`` (``_o_proj_chunks``) stationary. One PSUM tile holds the hidden tiles
    that fit one bank. With ring tiles, chunk c + ring loads into chunk c's tile after chunk c's matmuls."""
    fp32 = nl.float32
    bf16 = nl.bfloat16
    tiles_per_bank = min(_HIDDEN_TILES, _PSUM_BANK_FP32 // out_tokens)
    out_banks = []
    for bank_start in range(0, _HIDDEN_TILES, tiles_per_bank):
        bank_tiles = min(tiles_per_bank, _HIDDEN_TILES - bank_start)
        out_banks.append(
            (bank_start, bank_tiles, nl.ndarray((_P_MAX, bank_tiles, out_tokens), dtype=fp32, buffer=nl.psum))
        )
    if o_w_chunks and isinstance(o_w_chunks[0][2], MxOChunk):
        # MXFP8 o_proj: the gated value quantized per row (padded-lane order), one MXFP8 matmul a hidden tile.
        o_data, o_scale = _o_proj_mx_quantize(gated, out_tokens, f"{tag}_mla_o", sb)
        for chunk_start, chunk_tiles, chunk, _ring_tile in o_w_chunks:
            with sb.scope(f"{tag}_o_w_{chunk_start}") if o_proj_stream else nullcontext():
                if chunk.data is None:
                    chunk = _load_o_proj_chunk_mx(
                        mla_o_proj, chunk_start, chunk_tiles, tag, state_reduce_scatter, sb, chunk.scale
                    )
                _o_proj_mx_chunk(
                    out_banks, tiles_per_bank, chunk_start, chunk_tiles, chunk, o_data, o_scale, out_tokens
                )
        o_w_chunks = ()
    ring_tiles = [ring_tile for _, _, _, ring_tile in o_w_chunks if ring_tile is not None]
    ring_loads = {}
    for chunk_idx, (chunk_start, chunk_tiles, o_w, _ring_tile) in enumerate(o_w_chunks):
        if ring_tiles and o_w is None:
            o_w = ring_loads.pop(chunk_idx)
        with sb.scope(f"{tag}_o_w_{chunk_start}") if o_proj_stream and not ring_tiles else nullcontext():
            if o_w is None:
                o_w = _load_o_proj_chunk(mla_o_proj, chunk_start, chunk_tiles, tag, state_reduce_scatter, sb)
            for chunk_tile in range(chunk_tiles):
                hidden_tile = chunk_start + chunk_tile
                bank_start, _bank_tiles, out_psum = out_banks[hidden_tile // tiles_per_bank]
                for local_head in range(_LOCAL_HEADS):
                    nisa.nc_matmul(
                        dst=out_psum[0:_P_MAX, hidden_tile - bank_start, 0:out_tokens],
                        stationary=o_w[0:_P_MAX, local_head, chunk_tile * _P_MAX : (chunk_tile + 1) * _P_MAX],
                        moving=gated[0:_P_MAX, local_head, 0:out_tokens],
                        accumulate=local_head > 0,
                    )
        next_idx = chunk_idx + len(ring_tiles)
        if ring_tiles and next_idx < len(o_w_chunks):
            # Chunk c + ring loads into chunk c's tile: the load waits only for chunk c's matmuls.
            next_start, next_tiles, _, _ = o_w_chunks[next_idx]
            ring_loads[next_idx] = _load_o_proj_chunk(
                mla_o_proj,
                next_start,
                next_tiles,
                tag,
                state_reduce_scatter,
                sb,
                dst=ring_tiles[chunk_idx % len(ring_tiles)],
            )
    partial = sb.tile((_P_MAX, out_tokens, _HIDDEN_TILES), bf16)
    for bank_start, bank_tiles, out_psum in out_banks:
        nisa.tensor_copy(
            dst=partial[0:_P_MAX, 0:out_tokens, bank_start : bank_start + bank_tiles],
            src=out_psum.permute((0, 2, 1)),
        )
    return partial


def _load_o_proj_chunk(mla_o_proj, chunk_start, chunk_tiles, tag, state_reduce_scatter, sb, dst=None):
    """Load one o_proj weight chunk ``[128, local_heads, chunk_tiles * 128]``, gated on the state reduce-scatter.

    ``dst`` (``o_proj_ring``): a chunk tile of the ring, at least ``chunk_tiles * 128`` columns wide; None allocates
    the chunk's tile."""
    if dst is None:
        o_w = sb.tile((_P_MAX, _LOCAL_HEADS, chunk_tiles * _P_MAX), nl.bfloat16)
    else:
        o_w = dst if dst.shape[2] == chunk_tiles * _P_MAX else dst[0:_P_MAX, 0:_LOCAL_HEADS, 0 : chunk_tiles * _P_MAX]
    cols = mla_o_proj[0:_GATE_LOCAL, chunk_start * _P_MAX : (chunk_start + chunk_tiles) * _P_MAX]
    gate_weight_dma(
        nisa.dma_copy(
            dst=o_w,
            src=cols.reshape_dim(0, (_LOCAL_HEADS, _P_MAX)).permute((1, 0, 2)),
            dge_mode=nisa.dge_mode.hwdge,
            engine=nisa.engine.sync,
            name=f"{tag}_mla_w_o_{chunk_start}",
        ),
        *(() if state_reduce_scatter is None else (state_reduce_scatter,)),
    )
    return o_w


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
    tokens,
    mla_in_proj_scale=None,
    mla_q_b_proj_scale=None,
    mla_kv_b_proj_scale=None,
    mla_o_proj_scale=None,
):
    """Check the configuration and the static shapes of the body inputs."""
    kernel_assert(
        nl.program_ndim() == 0 or (nl.program_ndim() == 1 and nl.num_programs(0) == 1),
        "mla_decode_attention supports LNC1 only",
    )
    kernel_assert(nisa.get_nc_version() >= nisa.nc_version.gen3, "mla_decode_attention requires Trn2 or newer")
    kernel_assert(
        tuple(x_hT.shape) == (_P_MAX, tokens, _HIDDEN_TILES),
        f"x_hT must be token-major [128, B, 56], got {tuple(x_hT.shape)}",
    )
    kernel_assert(
        1 <= tokens <= MAX_TOKENS, f"mla_decode_attention supports 1 to {MAX_TOKENS} sequences, got B={tokens}"
    )
    kernel_assert(x_hT.dtype == nl.bfloat16, "x_hT must be BF16")
    if mla_in_proj_scale is None:
        kernel_assert(
            tuple(mla_in_proj.shape) == (_P_MAX, _IN_PROJ_ROW),
            f"mla_in_proj must be [128, {_IN_PROJ_ROW}] in SBUF order, got {tuple(mla_in_proj.shape)}",
        )
    else:
        data_shape, scale_shape = mx_hT_shapes(_HIDDEN, _IN_WIDTH)
        kernel_assert(
            is_trn3() and tuple(mla_in_proj.shape) == data_shape and tuple(mla_in_proj_scale.shape) == scale_shape,
            f"the MXFP8 mla_in_proj / scales must be {data_shape} / {scale_shape} on Trn3, got "
            f"{tuple(mla_in_proj.shape)} / {tuple(mla_in_proj_scale.shape)}",
        )
    if mla_q_b_proj_scale is None:
        q_b_row = _Q_LORA_TILES * _LOCAL_HEADS * _Q_HEAD
        kernel_assert(
            tuple(mla_q_b_proj.shape) == (_P_MAX, q_b_row),
            f"mla_q_b_proj must be [128, {q_b_row}] in SBUF order, got {tuple(mla_q_b_proj.shape)}",
        )
    else:
        q_b_data_shape, q_b_scale_shape = mx_hT_shapes(_Q_LORA, _LOCAL_HEADS * _Q_HEAD)
        kernel_assert(
            is_trn3()
            and tuple(mla_q_b_proj.shape) == q_b_data_shape
            and tuple(mla_q_b_proj_scale.shape) == q_b_scale_shape,
            f"the MXFP8 mla_q_b_proj / scales must be {q_b_data_shape} / {q_b_scale_shape} on Trn3, got "
            f"{tuple(mla_q_b_proj.shape)} / {tuple(mla_q_b_proj_scale.shape)}",
        )
    if mla_kv_b_proj_scale is None:
        kernel_assert(
            tuple(mla_kv_b_proj.shape) == (_P_MAX, 2 * _ABSORB_WIDTH),
            f"mla_kv_b_proj must be [128, {2 * _ABSORB_WIDTH}], got {tuple(mla_kv_b_proj.shape)}",
        )
    else:
        kv_b_cols = _KV_LORA + _GATE_LOCAL
        kernel_assert(
            is_trn3()
            and _LOCAL_HEADS * MX_QUADRANT_ROWS <= _P_MAX
            and tuple(mla_kv_b_proj.shape) == (_P_MAX, MX_LANES * kv_b_cols)
            and tuple(mla_kv_b_proj_scale.shape) == (MX_COMPACT_ROWS, kv_b_cols),
            f"the MXFP8 mla_kv_b_proj / scales must be [128, {MX_LANES * kv_b_cols}] / [{MX_COMPACT_ROWS}, "
            f"{kv_b_cols}] on Trn3, got {tuple(mla_kv_b_proj.shape)} / {tuple(mla_kv_b_proj_scale.shape)}",
        )
    if mla_o_proj_scale is None:
        kernel_assert(
            tuple(mla_o_proj.shape) == (_GATE_LOCAL, _HIDDEN),
            f"mla_o_proj must be [{_GATE_LOCAL}, {_HIDDEN}], got {tuple(mla_o_proj.shape)}",
        )
    else:
        o_data_shape, _ = mx_hT_shapes(MX_LANES * _V_HEAD, _HIDDEN)
        kernel_assert(
            is_trn3()
            and tuple(mla_o_proj.shape) == o_data_shape
            and tuple(mla_o_proj_scale.shape) == (MX_COMPACT_ROWS, _HIDDEN),
            f"the MXFP8 mla_o_proj / scales must be {o_data_shape} / [{MX_COMPACT_ROWS}, {_HIDDEN}] on Trn3, got "
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
    local_blocks = block_table.shape[1] if len(block_table.shape) == 2 else 0
    kernel_assert(
        1 <= local_blocks <= _MAX_STATIC_BLOCKS or local_blocks in (8, 16, 32, 64, 128, 256),
        f"block_table must have 1 to 4 local blocks, or a power of two of 8 to 256 (1M), got {local_blocks}",
    )
    kernel_assert(tuple(slot.shape) == (tokens, 1), f"slot must be [B, 1], got {tuple(slot.shape)}")
    kernel_assert(tuple(position.shape) == (tokens, 1), f"position must be [B, 1], got {tuple(position.shape)}")
