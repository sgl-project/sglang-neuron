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

"""Kimi-K3 multi-server decode latent MoE body: DP x TP32 projections, EP x TP static MXFP4 experts (Trn3, LNC1).

The world has W = 128 nnodes ranks (256 for 2 Trn3 servers). Each DP replica is TP32 inside one server: rank g has TP
rank g % 32 and DP rank g // 32, dp = W / 32. The router, the latent projections and the shared expert are TP32 inside
the replica, with the per-TP-rank shards of ``tkg_moe_dp``, on the B rows of the replica.

Routed experts: EP x TP. The 896 experts form G = W / tp groups of E_L = 896 / G experts. Rank g holds the experts of
group g % G for the intermediate part g // G (I_L = 3072 / tp columns of each expert), in the serving MXFP4 layout of
``moe_tkg_decode`` (its static loop). The world all-reduce of the routed latent adds the parts, so the intermediate
split costs no collective. Two servers: G = 128, E_L = 7, tp = 2, I_L = 1536.

A layer runs each collective once for all rows: the router-row gathers of the B rows of every replica (T = dp B
tokens), the world routed all-reduce of all T tokens and the TP32 output all-reduce. The expert kernel holds the tokens
of a call on its partitions, so it runs on chunks of at most 128 of the T gathered tokens, and each chunk stores its
part of the routed all-reduce source. So the collective count of a layer does not grow with B. By default
(``expert_call_tokens`` "auto", ``auto_expert_call_tokens``) the 2-server expert split runs 129 to 512 tokens in
token-tiled calls of 256 tokens, which load each expert weight once for each call.

Every collective runs on persistent private-HBM tensors (``moe_2srv_collective_hbm``): the runtime runs SBUF
collectives only inside one server, and the DP gather and the routed all-reduce span the servers.

With ``routed_reduce`` ``rs_ag`` (``ROUTED_REDUCES``, opt-in) the routed all-reduce becomes a world reduce-scatter of
replica-major blocks and a TP32 all-gather inside the replica (``routed_norm_rs_ag``). With ``rs_fold`` (opt-in) the
same reduce-scatter feeds the out-projection of each rank's shard with no rstd, and one FP32 TP32 all-reduce carries the
partial outputs and the norm sums of squares (``routed_fold_rs``, ``fold_out_reduce``).

With ``out_proj_form`` ``column`` (``OUT_PROJ_FORMS``, opt-in) each TP rank computes 224 hidden columns of its replica's
rows from all 9,728 out-projection inputs, and a TP32 all-gather of these columns replaces the TP32 output all-reduce
(``column_out_projection``). With both options the TP32 all-gather of ``rs_ag`` gives the column form the full routed
rows of the replica (``routed_full_norm_rs_ag``).
"""

import contextlib
import functools

import nki.collectives as ncc
import nki.experimental as nexp
import nki.isa as nisa
import nki.language as nl
from nki.collectives import ReplicaGroup

from ..utils.kernel_assert import kernel_assert
from ..moe.routed import (
    EXPERT_QUEUES,
    STATIC_ALT_TILE_FORMS,
    STATIC_ALT_TILE_PSUM_CONSUME,
    expert_loop_layer,
    moe_tkg_decode,
    moe_tkg_decode_per_expert,
    moe_tkg_decode_token_tiles,
    per_expert_scratch,
    static_expert_stream,
)
from ..moe.routed.moe_routed import (
    LAST_EXPERT_ORDERS,
    MAX_MATMULT_MX_UNPACKED_CONTRACT_DIM,
    PER_EXPERT_COMBINE_COLS,
    PER_EXPERT_SLOT_GATHERS,
    _all_experts_in_slots,
    _column_group_block_tokens,
    _q_width,
    _quantize_natural_latent,
    _snake_constants,
    _snake_index,
    _to_column_groups_at,
    active_experts_in_slots,
    scale_rows_across_quadrants,
    scale_rows_queue,
)
from ..common import check_pair_group, dma_priority, mark_ffn_output, pair_shared_buffer
from ..tkg_weight_dma_gate import _static_view_of, _view_key, release_gate_consumer, release_weight_gate
from ..tkg_sbuf_alloc import AUTO_SBUF
from ..tkg_moe import (
    _CRITICAL_DMA_PRIORITY,
    _H_TILES,
    _IN_PROJ_DMA_CHUNKS,
    _PMAX,
    _PSUM_BANK_FP32,
    _SELECTION_FLOOR,
    EXPERT_INTERMEDIATE,
    HIDDEN_SIZE,
    LATENT_NORM_EPS,
    LATENT_SIZE,
    NUM_EXPERTS,
    ROUTED_STORE_FORMS,
    ROUTER_SUM_EPS,
    SITU_BETA,
    SITU_LINEAR_BETA,
    _load_hT_rows,
    _project_to_hT,
    _psum_bank,
    _situ,
    pad_row_flags,
    rank_selectors,
    zero_pad_row_scores,
)
from ..tkg_moe import (
    replica_index as _dp_replica_index,
)
from ..tkg_moe_dp import (
    _GATHER_COLS,
    _GATHER_LATENT_WORDS,
    _GATHER_SCORE_COLS,
    _GATHER_SELECTION_COLS,
    _HBM_DMA,
    _OUT_PROJ_DMA_CHUNKS,
    _SHARD_PARTITIONS,
    DP_BIAS_COLS,
    DP_EXPERTS,
    DP_IN_PROJ_COLS,
    DP_LATENT,
    DP_LATENT_UP_ROWS,
    DP_NORM_COLS,
    DP_OUT_PROJ_ROWS,
    DP_ROUTER_COLS,
    DP_SHARED,
    DP_SHARED_DOWN_ROWS,
    DP_VECTOR_COLS,
    TP_SIZE,
    _all_reduce_hbm,
    _load_out_proj,
    _out_k_tiles,
    _psum_rows,
    _transpose_rows,
)
from ..tkg_mx_quant import MX_LANES, load_mx_scales, mx_hT_shapes, mx_scale_view, quantize_hT_mx
from .tkg_ffn_world_cc import world_layout
from .moe_routed_layout import check_layout_tag, rank_expert_shapes, routed_split
from .tkg_moe_routed_dynamic import (
    CSR_GATHER_FORMS,
    ROUTED_DYNAMIC_DEFAULT,
    default_pack,
    routed_dynamic_layer,
    routed_partial_tile,
)
from .tkg_moe_routed_tiles import alloc_tile_weights, load_tile_weights, routed_expert_tile

MOE_FORMS = ("moe_2srv",)
"""The MoE forms of the multi-server Trn3 entry (``decode_model_2srv``): this body."""
GLUE_HOOKS = ("routed_load", "out_trigger", "routed_trigger")
"""Keys of the ``glue_hooks`` of ``latent_moe_decode_2srv``: the points of the layer's glue where a caller traces work of
the next layer, or orders it after the layer's collectives."""
IN_PROJ_GLUE_HOOK = "in_proj_loads"
"""The ``glue_hooks`` key that the layer calls right after its router-row gathers (the dispatch all-gathers) with its head
weight loads (the FP16 router, the MXFP8 in-projection), the rows store and those gathers: a caller traces work there that
follows all of them in the order of the queues (``decode_block_body`` ``moe_in_proj_hook``)."""
TAIL_GLUE_HOOK = "tail_gathers"
"""The ``glue_hooks`` key that a ``routed_compact`` layer calls with its tail's token-order gathers (``decode_block``
``TAIL_POOL_PIN``); the other tails do not call it."""
MOE_SOURCES_KEY = "moe_2srv"
"""The key of this body's persistent HBM tensors in the kernel sources."""
MOE_2SRV_WEIGHT_NAMES = (
    "moe_in_proj_dp",
    "moe_out_proj_dp",
    "moe_vectors_dp",
    "expert_gate_up_weights",
    "expert_gate_up_weights_scale",
    "expert_down_weights",
    "expert_down_weights_scale",
)
EXPERT_LOOP_KEYS = frozenset(
    {
        "weight_buffers",
        "scale_buffers",
        "half_weight_buffers",
        "routed_store_form",
        "early_loads",
        "transpose_lookahead",
        "cross_expert_order",
        "cross_expert_transpose_blocks",
        "token_tiles_transpose_blocks",
        "static_alt_tiles",
        "transpose_block",
        "prefetch",
    }
)
"""The expert-loop options of ``decode_block.decode_block_body`` that the body takes (the static loop of
``moe_tkg_decode``); it fails on any other MoE option."""
ROUTER_KEYS = frozenset({"router_dtype", "router_weight", "x16_hT"})
"""The FP16 router options of ``decode_block.decode_block_body`` that the body takes (``ROUTER_DTYPES_2SRV``)."""
ROUTER_DTYPES_2SRV = ("bf16", "fp16")
"""Router types of the 2-server MoE body: ``fp16`` multiplies the FP16 norm copy of the replica's rows by the FP16
router rows of the 28 experts of the TP rank (the experts of ``DP_ROUTER_COLS``), with FP32 PSUM."""
MX_KEYS = frozenset({"in_proj_scale", "out_proj_scale"})
"""The MXFP8 projection options of ``decode_block.decode_block_body`` that the body takes (``in_proj_scale``,
``out_proj_scale``)."""
LEAD_KEYS = frozenset({"keep_warm_lead"})
"""The lead option of ``decode_block.decode_block_body`` that the body takes when its ``pe_keep_warm`` is a form for
each attention kind (``keep_warm_by_lead``): ``keep_warm_lead`` "kda" or "mla", the kind of the layer's attention."""
PE_KEEP_WARM_LEADS = ("kda", "mla")
"""The attention kinds of a per-kind ``pe_keep_warm`` (``{"kda": form, "mla": form}``)."""
MX_IN_PROJ_COLS = DP_IN_PROJ_COLS - DP_EXPERTS
"""The columns of the MXFP8 ``moe_in_proj_dp``: the latent down, the shared gate and the shared up columns. The FP16
router gives the logits, so the router columns are left out."""
MX_IN_PROJ_SHAPES = mx_hT_shapes(HIDDEN_SIZE, MX_IN_PROJ_COLS)
"""The ``(data, scales)`` shapes of the MXFP8 ``moe_in_proj_dp`` and its ``in_proj_scale`` (``tkg_mx_quant``)."""
MX_OUT_PROJ_SHAPES = mx_hT_shapes(MX_LANES * _PMAX, HIDDEN_SIZE)
"""The ``(data, scales)`` shapes of the MXFP8 padded-lane ``moe_out_proj_dp`` and its ``out_proj_scale``: the
``DP_OUT_PROJ_ROWS`` rows in the hT_quad lanes of one MX group of 512 rows, the rows past them zero."""
_MX_IN_PROJ_DMA_GROUPS = 2
"""The MX groups of 512 hidden rows that one DMA of the MXFP8 in-projection loads."""
MAX_REPLICA_ROWS = _PMAX
"""Most rows of a replica: they sit on the partitions of the in-projection outputs."""
WIDE_ROWS = 32
"""Rows of a replica above which the whole-expert form (I_L = 3072) streams half projections in its expert loop."""
WIDE_HALF_WEIGHT_BUFFERS = 3
"""Half-projection weight buffers of the whole-expert form above ``WIDE_ROWS`` rows: the SBUF of 1.5 whole buffers in
place of the 2 of the decode block, with one projection of lookahead."""
EXPERT_CALL_TOKENS = _PMAX
"""Most tokens of one expert call: they sit on the partitions of ``moe_tkg_decode``."""
_QUANT_ROWS = 4
_HEAD_ROW_CHUNKS = 4
"""Router-row loads of an expert call with ``head_row_chunks``: 8 TP blocks (896 latent columns) each."""
"""The token multiple of the MXFP8 latent quantize (``moe_tkg_decode`` ``quantized_latent``: T a multiple of 4)."""
EXPERT_CALL_FORMS = (EXPERT_CALL_TOKENS, 2 * _PMAX, 4 * _PMAX)
"""The tokens of one expert call (``expert_call_tokens``): 128 calls ``moe_tkg_decode`` for each 128 tokens; 256 or
512 calls ``moe_tkg_decode_token_tiles`` for each 256 or 512 tokens (each weight byte moves once for each call). The
values do not change."""
_TOKEN_TILES_WEIGHT_BUFFERS = 3
"""Most whole weight buffers of a token-tiled expert call (``moe_tkg_decode_token_tiles``)."""
EXPERT_CALL_AUTO = "auto"
"""The default of ``expert_call_tokens``: ``auto_expert_call_tokens`` picks the call size from the gathered tokens."""
AUTO_CALL_TOKENS = 2 * _PMAX
"""The token-tiled call size of "auto"."""
AUTO_MAX_TOKENS = 4 * _PMAX
"""The most gathered tokens of a layer that "auto" takes. Above them, give ``expert_call_tokens``."""
AUTO_TILED_ROUTED_TP = 2
"""The expert split that "auto" tiles (I_L = 1536). Whole experts (routed_tp 1) keep 128-token calls: their tiled
calls do not fit the SBUF of the 93-layer program."""
AUTO_MAX_TWO_BUFFER_CALLS = 2
"""The most 128-token calls with two whole weight buffers that "auto" gives a layer. Each call loads the same expert
weights again, and from the third such call the load merging of the compiler pass ``dma_optimization_sb`` can give a
call the weight buffer of an earlier call."""
_TWO_WEIGHT_BUFFERS = 2
# The whole weight buffers of a token-tiled expert call in the cross-expert order (moe_tkg_decode_token_tiles).
_CROSS_TOKEN_TILE_WEIGHT_BUFFERS = 3
TOKEN_TILES_CROSS_ORDER = True
"""True: a token-tiled expert call (more than 128 tokens, ``moe_tkg_decode_token_tiles``) takes the cross-expert order
(``cross_expert_order``) with 3 whole weight buffers, so a verify form whose expert calls are token-tiled can pass a
given order without the routed compaction (``verify_model_2srv.served_defaults``)."""
MOE_GATHERS = ("dp_tp", "world")
"""How a layer gathers the router rows of all ranks: ``dp_tp`` (default) all-gathers over the DP group (it spans the
servers) and then over the TP32 group (inside a server); ``world`` all-gathers once over the world, one collective
less but every rank's rows cross the servers."""
SCORE_ROW_COLS = _GATHER_SELECTION_COLS[0]
"""Words of a gathered router row with ``drop_selection``: the 56 latent words and the 28 scores, no selection values.
Each destination adds the router bias to the gathered scores (``_add_router_bias``)."""
ROUTER_BIAS_COLS = (DP_VECTOR_COLS, DP_VECTOR_COLS + NUM_EXPERTS)
"""Columns of ``moe_vectors_dp`` with ``drop_selection``: the router correction bias of all 896 experts in expert order
(expert 28 t + e at column 140 + 28 t + e), after the 140 columns of the DP layout. Every rank holds the same values."""


ROWS_LATENTS = ("bf16", "fp8", "fp8a")
"""How a router row carries the rank's 112 latent values of a token (``rows_latent``): ``bf16`` (default) in 56 FP32
words; ``fp8`` as 112 float8_e4m3fn values (28 words) and one FP32 scale (the row's absolute maximum / 448), so the
row has 26 words fewer and the DP and TP32 row gathers move 23 % fewer bytes (31 % with ``drop_selection``). The
destination expands each value to ``bf16(q * scale)``. ``fp8`` rounds the latent: an accuracy-class option.
``fp8a`` (``drop_selection`` and the ``dp_tp`` gather only) is ``fp8`` in a 64-word (256 B) row: 2 zero words, the 58
words of the ``fp8`` row, then 4 zero words. The rows load then moves 256 B elements that start 256-B aligned in HBM and
32-B aligned in the gathered tile (word 24), in place of 232 B elements at word 26. Every value keeps its word in the
gathered tile, so ``fp8a`` gives the bits of ``fp8``. The 2-server verify takes it with the glue key ``rows_align``."""
ROWS_LOAD_SPLITS = (2, 4)
"""The parts of the router-rows load of an expert call (``rows_load_split`` of ``latent_moe_decode_2srv``)."""
_HBM_DMA_ACT = {**_HBM_DMA, "engine": nisa.engine.scalar}
"""``_HBM_DMA`` on the Activation hardware DGE queue: the second queue of a split rows load."""
_FP8_LATENT_WORDS = (0, DP_LATENT // 4)
_FP8_SCALE_WORD = _FP8_LATENT_WORDS[1]
_FP8_SHIFT = _GATHER_SCORE_COLS[0] - (_FP8_SCALE_WORD + 2)
"""Word ``w`` of an ``fp8`` row lands at word ``w + 26`` of the gathered tile, so the scores and the selection values
keep the words of a ``bf16`` row (one pad word follows the scale)."""
_FP8A_PAD = 2
"""The zero words before the ``fp8`` words of an ``fp8a`` row."""
_FP8A_COLS = 64
"""The words of an ``fp8a`` row (``drop_selection``): ``_FP8A_PAD`` + 58 + 4 zero words."""
_PAIR_SLOT = 176
"""The FP32 words of one (partition, TP block) slot of a ``rows_pairs`` rows tile (``_load_pair_rows``): the 58-word
``fp8`` rows of two adjacent tokens (one 464 B DMA element), the 28 selection values of each, and 4 unused words, so a
slot is 704 B, a whole number of 32 B SBUF blocks."""
_FP8_MAX = 448.0
_FP8_TINY = 1e-30
"""The scale floor of an all-zero row (its values quantize to 0)."""


# The latent_gather_mx mechanism in the TP32 x DP8 layout (G1a). TP rank t < 28 computes the 128 latent columns
# [128 t, 128 t + 128) = 4 whole MXFP8 blocks (one quadrant of the MX quantize layout: partitions 32 (t % 4) .. + 31
# of latent tile t // 4), quantizes them with quantize_mx and sends the codes and the 4 E8M0 block scales in the router
# rows; ranks 28..31 compute zeros. The receiver loads the codes and the scales straight into the MX expert input
# (moe_tkg_decode and moe_tkg_decode_token_tiles quantized_latent) and runs no latent expand, no swizzle transpose and
# no quantize.
MX_BLOCK = 32
"""Latent columns of one MXFP8 block (quantize_mx makes one E8M0 scale for each 32-element group)."""
MX_LATENT_COLS = 4 * MX_BLOCK
"""Latent columns of one G1a rank (128): 4 whole MX blocks = partitions 32 (t % 4) .. + 31 of latent tile t // 4."""
MX_LATENT_RANKS = LATENT_SIZE // MX_LATENT_COLS
"""TP ranks that own a G1a latent block (28); ranks 28..31 compute no latent column (zero in-projection columns)."""
_MX_QUADRANT_PARTITIONS = 32
"""Partitions of one MX quadrant: a latent tile of 512 columns is 128 partitions = 4 quadrants of 32 partitions."""
_MX_Q = 4
"""Elements of one partition of a latent tile in the MX quantize layout (512 columns = 128 partitions x 4)."""
_MX_SCALES_PER_QUADRANT = 4
"""E8M0 block scales that quantize_mx writes in a quadrant (partitions 32 Q + 0 .. 3 of the tile)."""
_MX_CODE_WORDS = MX_LATENT_COLS // _MX_Q
"""x4 code words of one rank's 128 latent columns of a token (32)."""
_MX_HEAD_WORDS = _MX_CODE_WORDS + 1
"""Head words of a G1a rank block of a token: the 32 code words and 1 word of the 4 E8M0 scale bytes (33)."""
_MX_SCORE_SLOT = 32
"""Words of the score run of a token in a G1a rank block with ``mx_scores_align``: the 28 scores, then 4 zero words.
The score load then moves 128-B elements, which the DGE packs into 4 KB packets; it does not pack 112-B elements."""
_MX_STREAM_COLS = 64
"""Words of a G1a router row with ``mx_rows_stream`` (one row a token and rank, token-major): the 32 code words, the
word of the 4 E8M0 scale bytes, the 28 scores, then 3 zero words. The rows load of a tile moves each half of a row as
one 128-B element, an element size that the DGE packs into 4 KB packets (MEASURED for the 128-B score elements)."""
G1A_IN_PROJ_COLS = DP_IN_PROJ_COLS - DP_LATENT + MX_LATENT_COLS
"""Columns of the BF16 ``moe_in_proj_dp`` with ``latent_gather_mx`` (540): router (28), the 128-column latent block, the
192 shared gate and the 192 shared up columns."""
MX_G1A_IN_PROJ_COLS = MX_IN_PROJ_COLS - DP_LATENT + MX_LATENT_COLS
"""Columns of the MXFP8 ``moe_in_proj_dp`` with ``latent_gather_mx`` (512: the 128-column latent block, gate and up, no
router columns)."""
MX_G1A_IN_PROJ_SHAPES = mx_hT_shapes(HIDDEN_SIZE, MX_G1A_IN_PROJ_COLS)
"""The ``(data, scales)`` shapes of the MXFP8 ``moe_in_proj_dp`` and its ``in_proj_scale`` with ``latent_gather_mx``."""
LATENT_GATHER_MX_TOKEN_TILES = True
"""``latent_gather_mx`` takes the token-tiled expert calls of 256 and 512 tokens (``moe_tkg_decode_token_tiles``
``quantized_latent``), with whole replicas in each 128-token tile. A caller reads it to know that a layer of more than
128 gathered tokens takes the MXFP8 latent rows."""
ROUTED_COMPACT_LATENT_GATHER_MX = True
"""``routed_compact`` takes ``latent_gather_mx``: the routed-token compaction loads the MXFP8 codes and scales of its
tokens straight into its gather source, with whole replicas in each 128-token tile (``_compact_routed_partial``). A
caller reads it to know that the compaction and the MXFP8 latent rows run together; a body without it refuses them."""


def router_row_cols(
    drop_selection: bool,
    rows_latent: str = "bf16",
    latent_gather_mx: bool = False,
    mx_scores_align: bool = False,
    mx_rows_stream: bool = False,
) -> int:
    """Return the FP32 words of one gathered router row: 112, or 84 with ``drop_selection``; 26 fewer with
    ``rows_latent`` ``fp8``; 89 (61 with ``drop_selection``) with ``latent_gather_mx`` (the 32 code words and the scale
    word, then the 28 scores and, without ``drop_selection``, the 28 selection values); 65 with ``latent_gather_mx``,
    ``drop_selection`` and ``mx_scores_align`` (the score run padded to ``_MX_SCORE_SLOT`` words); 64 with
    ``latent_gather_mx``, ``drop_selection`` and ``mx_rows_stream`` (``_MX_STREAM_COLS``, token-major rows)."""
    kernel_assert(
        not mx_scores_align or (latent_gather_mx and drop_selection),
        f"mx_scores_align pads the score run of the drop_selection latent_gather_mx rows: it needs both, got "
        f"latent_gather_mx={latent_gather_mx}, drop_selection={drop_selection}",
    )
    kernel_assert(
        not mx_rows_stream or (latent_gather_mx and drop_selection and not mx_scores_align),
        "mx_rows_stream lays out the drop_selection latent_gather_mx rows token-major: it needs both, and not "
        "mx_scores_align",
    )
    if latent_gather_mx and mx_rows_stream:
        return _MX_STREAM_COLS
    if latent_gather_mx:
        if mx_scores_align:
            return _MX_HEAD_WORDS + _MX_SCORE_SLOT
        return _MX_HEAD_WORDS + (DP_EXPERTS if drop_selection else 2 * DP_EXPERTS)
    if rows_latent == "fp8a":
        kernel_assert(drop_selection, "rows_latent fp8a is the 64-word fp8 row of drop_selection")
        return _FP8A_COLS
    cols = SCORE_ROW_COLS if drop_selection else _GATHER_COLS
    return cols - _FP8_SHIFT if rows_latent == "fp8" else cols


def _row_shift(rows_latent: str) -> int:
    """Return the word offset of the scores in a router row of ``rows_latent``, against a ``bf16`` row (the word of
    the gathered tile where the row starts)."""
    if rows_latent == "fp8a":
        return _FP8_SHIFT - _FP8A_PAD
    return _FP8_SHIFT if rows_latent == "fp8" else 0


def check_rows_pairs(
    rows_pairs: bool,
    gather: str,
    drop_selection: bool,
    rows_latent: str,
    rows_load_split: int,
    latent_gather_mx: bool,
    tokens: int,
    call_tokens: int,
    routed_compact: bool | str = False,
):
    """Stop a ``rows_pairs`` layer that is not the served row form: the ``dp_tp`` gather of ``fp8`` rows with
    ``drop_selection`` (58 words; not ``fp8a``), one rows load, no ``latent_gather_mx``, and only token-tiled expert
    calls of 256 n tokens (``expert_calls`` of ``tokens`` by ``call_tokens``), so that every pair of token tiles
    covers 256 consecutive tokens. Not with ``routed_compact``: the compaction loads and stores the rows of each
    128-token tile in token order (``_compact_routed_partial``)."""
    if not rows_pairs:
        return
    kernel_assert(
        gather == "dp_tp" and drop_selection and rows_latent == "fp8" and rows_load_split == 1 and not latent_gather_mx,
        f"glue rows_pairs loads the 58-word fp8 drop_selection rows of the dp_tp gather: got gather {gather!r}, "
        f"drop_selection {drop_selection}, rows_latent {rows_latent!r}, rows_load_split {rows_load_split}, "
        f"latent_gather_mx {latent_gather_mx}",
    )
    kernel_assert(
        not routed_compact,
        f"glue rows_pairs takes the token-tiled calls without routed_compact, got routed_compact {routed_compact!r}",
    )
    counts = [count for _, count in expert_calls(tokens, call_tokens)]
    kernel_assert(
        all(count % (2 * _PMAX) == 0 for count in counts),
        f"glue rows_pairs needs expert calls of 256 n tokens (pairs of 128-token tiles), got the calls {counts} of "
        f"{tokens} tokens",
    )


def _pair_row_words(slot: int) -> tuple:
    """Return the words of token ``slot`` (0 or 1) in a ``rows_pairs`` rows tile (``_load_pair_rows``): (the first
    word of its ``fp8`` row, its score columns, its selection columns)."""
    width = SCORE_ROW_COLS - _FP8_SHIFT
    base = slot * width
    scores = (base + _GATHER_SCORE_COLS[0] - _FP8_SHIFT, base + _GATHER_SCORE_COLS[1] - _FP8_SHIFT)
    selection = (2 * width + slot * DP_EXPERTS, 2 * width + (slot + 1) * DP_EXPERTS)
    return base, scores, selection


def quantize_latent_words(latent: nl.NkiTensor, gather_src: nl.NkiTensor, name: str, base: int = 0):
    """Write the ``fp8`` latent words of the router rows: ``gather_src[:, 0:28]`` takes ``fp8(x / s)`` as 112
    float8_e4m3fn values and word 28 the FP32 scale ``s = max(amax, tiny) / 448`` of each row of ``latent``
    ([B, 112] FP32); word 29 is 0. ``base`` moves these words by that many words (``fp8a``: ``_FP8A_PAD``)."""
    rows = latent.shape[0]
    magnitude = nl.ndarray((rows, DP_LATENT), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=magnitude, data=latent, op0=nl.abs, operand0=0.0, name=f"{name}_fp8_abs")
    amax = nl.ndarray((rows, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=amax, data=magnitude, op=nl.maximum, axis=(1,), name=f"{name}_fp8_amax")
    scale = gather_src[:, base + _FP8_SCALE_WORD : base + _FP8_SCALE_WORD + 1]
    nisa.tensor_scalar(
        dst=scale,
        data=amax,
        op0=nl.maximum,
        operand0=_FP8_TINY,
        op1=nl.multiply,
        operand1=1.0 / _FP8_MAX,
        name=f"{name}_fp8_scale",
    )
    inv = nl.ndarray((rows, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.reciprocal(dst=inv, data=scale)
    nisa.tensor_scalar(
        dst=gather_src[:, base + _FP8_LATENT_WORDS[0] : base + _FP8_LATENT_WORDS[1]].view(nl.float8_e4m3fn),
        data=latent,
        op0=nl.multiply,
        operand0=inv,
        name=f"{name}_fp8_quant",
    )
    nisa.memset(dst=gather_src[:, base + _FP8_SCALE_WORD + 1 : base + _FP8_SCALE_WORD + 2], value=0.0)


def latent_from_blocks(gathered: nl.NkiTensor, latent_sb: nl.NkiTensor, rows_latent: str, t0: int, t1: int):
    """``latent_from_rows`` for the TP blocks ``[t0, t1)`` only (one part of a split rows load): the same per-element
    operation, the BF16 words or ``bf16(q * scale)`` of each block's FP8 words and scale."""
    if rows_latent != "fp8":
        nisa.tensor_copy(
            dst=latent_sb[:, t0:t1, :],
            src=gathered[:, t0:t1, _GATHER_LATENT_WORDS[0] : _GATHER_LATENT_WORDS[1]].view(nl.bfloat16),
            engine=nisa.engine.vector,
        )
        return
    count = gathered.shape[0]
    words = (_FP8_SHIFT + _FP8_LATENT_WORDS[0], _FP8_SHIFT + _FP8_LATENT_WORDS[1])
    scale = gathered.ap(
        pattern=[[TP_SIZE * _GATHER_COLS, count], [_GATHER_COLS, t1 - t0], [0, DP_LATENT]],
        offset=t0 * _GATHER_COLS + _FP8_SHIFT + _FP8_SCALE_WORD,
    )
    nisa.tensor_tensor(
        dst=latent_sb[:, t0:t1, :],
        data1=gathered[:, t0:t1, words[0] : words[1]].view(nl.float8_e4m3fn),
        data2=scale,
        op=nl.multiply,
        engine=nisa.engine.vector,
    )


def latent_from_rows(gathered: nl.NkiTensor, latent_sb: nl.NkiTensor, rows_latent: str, base: int = _FP8_SHIFT):
    """Write the BF16 latent ``latent_sb`` [count, 32, 112] of the gathered rows ``gathered`` [count, 32, 112] FP32:
    the BF16 words (``bf16``), or ``bf16(q * scale)`` of the FP8 words and the scale of each block (``fp8``, and
    ``fp8a``, whose rows put them at the same words of the gathered tile). ``base``: the word of the ``fp8`` row in
    each slot (a ``rows_pairs`` tile of ``_PAIR_SLOT`` words: ``_pair_row_words``)."""
    if rows_latent not in ("fp8", "fp8a"):
        nisa.tensor_copy(
            dst=latent_sb,
            src=gathered[:, :, _GATHER_LATENT_WORDS[0] : _GATHER_LATENT_WORDS[1]].view(nl.bfloat16),
            engine=nisa.engine.vector,
        )
        return
    count, slot = gathered.shape[0], gathered.shape[2]
    words = (base + _FP8_LATENT_WORDS[0], base + _FP8_LATENT_WORDS[1])
    kernel_assert(0 <= DEQUANT_ACT_BLOCKS <= TP_SIZE, f"DEQUANT_ACT_BLOCKS is 0 to {TP_SIZE}, got {DEQUANT_ACT_BLOCKS}")
    vector_blocks = TP_SIZE - DEQUANT_ACT_BLOCKS
    if vector_blocks:
        scale = gathered.ap(
            pattern=[[TP_SIZE * slot, count], [slot, vector_blocks], [0, DP_LATENT]],
            offset=base + _FP8_SCALE_WORD,
        )
        nisa.tensor_tensor(
            dst=latent_sb[:, 0:vector_blocks, :],
            data1=gathered[:, 0:vector_blocks, words[0] : words[1]].view(nl.float8_e4m3fn),
            data2=scale,
            op=nl.multiply,
            engine=nisa.engine.vector,
        )
    for block in range(vector_blocks, TP_SIZE):
        # ``DEQUANT_ACT_BLOCKS``: block t of every row on ScalarE, with the row's scale of the block as the per-partition
        # scale.
        nisa.activation(
            dst=latent_sb[:, block : block + 1, :],
            op=nl.copy,
            data=gathered[:, block : block + 1, words[0] : words[1]].view(nl.float8_e4m3fn),
            scale=gathered.ap(pattern=[[TP_SIZE * slot, count], [1, 1]], offset=block * slot + base + _FP8_SCALE_WORD),
            engine=nisa.engine.scalar,
        )


def router_rows_src(
    latent_src, logits, bias_sb, rows: int, drop_selection: bool, rows_latent: str, pad_rows, name: str
):
    """Return this rank's router rows [rows, ``router_row_cols``] FP32 SBUF, the source of the row gathers: the latent
    words of ``rows_latent``, the sigmoid scores of ``logits`` [rows, 28] and, without ``drop_selection``, the selection
    values (score + ``bias_sb``). ``pad_rows`` (or None) zeroes the scores of the pad rows (``zero_pad_row_scores``)."""
    gather_src = nl.ndarray((rows, router_row_cols(drop_selection, rows_latent)), dtype=nl.float32, buffer=nl.sbuf)
    shift = _row_shift(rows_latent)
    if rows_latent == "fp8a":
        # The zero words of the 64-word row, so the gathered bytes do not depend on stale SBUF.
        nisa.memset(dst=gather_src[:, 0:_FP8A_PAD], value=0.0)
        nisa.memset(dst=gather_src[:, SCORE_ROW_COLS - shift : _FP8A_COLS], value=0.0)
    if rows_latent in ("fp8", "fp8a"):
        quantize_latent_words(latent_src, gather_src, name, _FP8A_PAD if rows_latent == "fp8a" else 0)
    else:
        nisa.tensor_copy(
            dst=gather_src[:, _GATHER_LATENT_WORDS[0] : _GATHER_LATENT_WORDS[1]].view(nl.bfloat16), src=latent_src
        )
    scores_src = gather_src[:, _GATHER_SCORE_COLS[0] - shift : _GATHER_SCORE_COLS[1] - shift]
    nisa.activation(dst=scores_src, op=nl.sigmoid, data=logits)
    if not drop_selection:
        nisa.tensor_tensor(
            dst=gather_src[:, _GATHER_SELECTION_COLS[0] - shift : _GATHER_SELECTION_COLS[1] - shift],
            data1=scores_src,
            data2=bias_sb,
            op=nl.add,
        )
    if pad_rows is not None:
        zero_pad_row_scores(scores_src, pad_row_flags(pad_rows, name))
    return gather_src


def moe_vector_cols(drop_selection: bool) -> int:
    """Return the FP32 columns of ``moe_vectors_dp``: 140, or 1,036 with ``drop_selection`` (``ROUTER_BIAS_COLS``)."""
    return ROUTER_BIAS_COLS[1] if drop_selection else DP_VECTOR_COLS


ROUTED_PARTITIONS = _PMAX
"""Partitions of the routed latent layout ``[128, T * 28]``: element (p, 28 i + c) is latent column 28 p + c of token i."""
ROUTED_COLS = LATENT_SIZE // ROUTED_PARTITIONS
ROUTED_LAYOUTS = ("natural", "qsplit")
"""The column order of ``routed_src`` / ``routed_dst`` in HBM (``routed_layout``, ``routed_reduce`` ``ar`` only).
``natural`` (default): token r B + b at columns 28 (r B + b) + c, so the two 64-token owner blocks of a row hold the
replicas 0..dp/2-1 and dp/2..dp-1, and the add order of the world all-reduce of an element depends on the DP rank of
its token. ``qsplit``: half h of each row holds the rows h B / 2 .. (h + 1) B / 2 - 1 of every replica, at column
h dp B / 2 * 28 + r B / 2 * 28 + 28 j + c for row b = h B / 2 + j (the rows of a replica are query-row major, so a half
is a fixed set of query rows). An element then always sits in the same owner block, whatever the DP rank and slot of
its token, so its add order does not depend on the placement. The same bytes and collectives; the stores and loads of
the routed tensors permute the columns, so the SBUF views keep the natural token order. The order of the all-reduce
changes for the elements whose block changes, so the values change."""
DEQUANT_ACT_BLOCKS = 16
"""``latent_from_rows`` with the FP8 rows: the last ``DEQUANT_ACT_BLOCKS`` of the 32 TP blocks of each gathered tile
expand on ScalarE (one copy activation of the block's FP8 words times its per-row scale each) and the others on VectorE
(one multiply), so the expand leaves the VectorE chain before the expert calls. 16 (default) expands half the blocks on
ScalarE, with the bits of the all-VectorE expand on the device. 0 keeps every block on VectorE. Other values need the
device identity: the engines must round ``q * scale`` the same. One difference is in the sign of zero: the trn3 ScalarE
multiply writes +0 for -0, so an FP8 -0 code of a ScalarE block expands to +0 where VectorE gives -0 (the same value).
The split rows load (``latent_from_blocks``) expands every block on VectorE."""
ROUTED_REDUCES = ("ar", "rs_ag", "rs_fold")
"""How a layer reduces the routed partials of the world (``routed_reduce``): ``ar`` (default) all-reduces the whole
``[128, T * 28]`` latent over the world; ``rs_ag`` reduce-scatters it over the world in replica-major blocks (block
32 r + t is rank (r, t)'s B rows x its 112 latent columns) and all-gathers the blocks of the replica over the TP32 group
(``moe_2srv_collective_hbm``); ``rs_fold`` takes the same reduce-scatter and folds the latent norm into the output
all-reduce (``routed_fold_rs``, ``fold_out_reduce``), so a layer keeps two collectives. The reductions have other
owners, so ``rs_ag`` rounds differently from ``ar``; ``rs_fold`` also moves the norm scale after the out-projection
and sums the output in FP32."""
NORM_TAILS = ("rows", "token_major")
"""How a layer with ``routed_reduce`` ``ar`` takes the normed latent of its replica's rows (``norm_tail``): ``rows``
(default) loads the whole all-reduce result, selects the replica's rows with predicated copies and takes the rank's
112 columns with one-hot matmuls and transposes (``routed_normed_ar``); ``token_major`` loads only the replica's rows
for the norm sums and the rank's 112 columns token-major, at runtime offsets of the DP and TP ranks
(``routed_normed_token_major``). The values do not change."""
ROWS_QUEUES = ("sp", "act")
"""The DMA queue of the router-row store and of the router-bias load (``rows_queue``): ``sp`` (default), the SP hardware
DGE queue of the other HBM DMAs of the body; ``act``, the Activation hardware DGE queue. In a decode step the SP queue
also carries the state stores of a KDA attention, and the store of the rows waits behind them. The values do not
change."""
OUT_PROJ_AFTER_ROWS_FORMS = (False, "rows_gather", "shard_gather", "shard_gather_done")
"""The values of the ``out_proj_after_rows`` glue key: False (default) keeps the program; ``rows_gather`` puts the early
MXFP8 out-projection loads after the DP router-row gather in the schedule, ``shard_gather`` after the TP32 one, and
``shard_gather_done`` also makes each load wait for the TP32 gather to complete (a runtime dependency). In a decode step
these loads (about 2 MB on the GpSimd software DGE queue) otherwise issue just before the router-row store and share the
DMA engines with it, so the small packets of the store wait behind them and the gather that waits for the store starts
later. After a gather trigger only, they share the engines with the TP32 gather's flight instead, which then takes
longer; after its completion they load during the compaction, where the expert-weight stream waits for its ring. The
values do not change."""
MX_LOAD_ORDERS = ("quadrant", "tile")
"""The issue order of the ``latent_gather_mx`` code and scale loads of an expert call (``mx_load_order``): ``quadrant``
(default), one code DMA and one scale DMA for each partition quadrant, each over all 7 latent tiles; ``tile``, one code
DMA and one scale DMA for each (tile, quadrant), tile 0 of the 4 quadrants first. A reader waits for each whole DMA
that writes its bytes, so with ``tile`` the first MX matmul (tile 0) waits only for the tile-0 DMAs. The bytes do not
change. Only the static expert calls of at most 128 tokens take a value other than the default: a token-tiled call
refuses it."""
MX_SCALE_QUEUES = ("act", "sp")
"""The DMA queue of the ``latent_gather_mx`` scale loads (``mx_scale_queue``): ``act`` (default), the Activation
hardware DGE queue; ``sp``, the SP hardware DGE queue of the code loads. The bytes do not change."""
MX_RELOAD_ORDERS = ("default", "codes_ahead", "codes_ahead_scores_act")
"""The trace order of the code, scale and score loads of a keep-warm ``latent_gather_mx`` expert call
(``mx_reload_order``; it needs ``pe_keep_warm``): ``default`` keeps the program, which loads the scores and adds the
router bias before it issues the code and scale DMAs. With ``codes_ahead`` the call issues the code and scale DMAs
before the router-bias add, so those DMAs do not follow the bias add in the trace; the score DMAs still issue first
and the first code DMA still follows the last score load. The bias add runs on the same scores and biases after the
code issue, so its output does not change. ``codes_ahead_scores_act`` also puts the score DMAs on the Activation
hardware DGE queue, off the SP queue of the code DMAs: then no queue orders the last score DMA before the tile-0 code
and scale DMAs (with ``mx_scale_queue`` ``sp`` all of those are on SP), so a keep-warm matmul that waits for a late
Activation score DMA can hold the first expert MX matmul behind it in the TensorE stream (perf only;
``_pe_keep_warm_matmuls``). The loaded bytes and the added bias do not change."""
PE_KEEP_WARM_MATMULS = 5
"""The score-tied keep-warm MX matmuls of an expert call with ``pe_keep_warm`` True (``_pe_keep_warm_matmuls``). Each
has the expert matmul shape (512 moving columns), so the TensorE runs that work before the first expert MX matmul of
the call and leaves the PE di/dt throttle (a lower rate after a long TensorE idle) before the expert matmuls."""
PE_KEEP_WARM_MAX = 8
"""Most score-tied keep-warm MX matmuls of an expert call (``pe_keep_warm`` an int or ``matmuls``)."""
_KEEP_WARM_COLS = 512
"""The moving columns of a keep-warm MX matmul: one FP32 PSUM bank (2 KB a partition), the expert matmul shape."""
GLUE_KEYS = {
    "rows_align": False,
    "rows_pairs": False,
    "norm_tail": "rows",
    "rows_queue": "sp",
    # Not the MoE body's: the x gather form of the dense layer 0 (tkg_dense_mlp_2srv.DENSE_X_GATHERS), which the
    # multi-server verify entry reads from its moe_glue, so the serving config passes it with the other glue keys.
    "dense_x_gather": "dp",
    "router_first": False,
    "out_proj_early": False,
    "out_proj_after_rows": False,
    "selectors_once": False,
    "mx_load_order": "quadrant",
    "mx_scale_queue": "act",
    "mx_reload_order": "default",
    "mx_scores_align": False,
    "mx_rows_stream": False,
    "pe_keep_warm": False,
    "routed_compact": False,
    "compact_key_constants": False,
    "store_queues": "sp",
    "accum_memset_after_rows": False,
    "weights_before_compact": False,
    "weights_after_rows": False,
    "ungather_skip": False,
    "prologue_vector_cuts": False,
    "up_product_split": False,
    "scale_rows_quadrants": False,
    "scale_rows_queue": "static",
    "router_norm_one_mul": False,
    "routed_per_expert": False,
    "routed_layout": "natural",
    "compact_slot_gather": "gpsimd",
    "in_proj_lanes_view": False,
    "tail_before_guard": False,
    "tail_store_halves": False,
    "compact_alt_tiles": False,
    "first_add_zero": False,
    "last_expert_order": "default",
    "compact_tables_scalar": False,
    "affinity_sbuf_copy": False,
    "out_proj_priority": None,
    "tail_zero_early": False,
    "slot_static_first": False,
    "out_proj_before_in_proj": False,
    "scale_fill_after_prefetch": False,
    # Not the MoE body's: the engine plan of the KDA layers' AttnRes bank parts (decode_block attn_res_ahead_vector),
    # which the multi-server verify entry reads from its moe_glue, so the serving config passes it with the glue keys.
    "ahead_sums_vector": "off",
    # Not the MoE body's either: the entry-load tiles of those bank parts (decode_block attn_res_ahead_load_tiles).
    "ahead_load_tiles": 2,
    # Nor this: the residual add after those bank parts' weighted sums (decode_block attn_res_ahead_before_resid).
    "ahead_before_resid": False,
    # Nor this: a KDA layer's own MLP bank part after its token chain (decode_block attn_res_ahead_mlp_kda_tail).
    "ahead_mlp_kda_tail": False,
    # Nor this: the first entries of a KDA layer's plain MLP bank sums after its output projection (decode_block
    # attn_res_kda_mlp_tail_entries).
    "kda_mlp_tail_entries": 0,
    # Nor this: the first entries of an MLA layer's MLP bank sums in the flight of its output all-reduce (decode_block
    # attn_res_mla_mlp_tail_entries).
    "mla_mlp_tail_entries": 0,
    # Nor this: the Gram stream sums of the ahead mixes as one matmul a hidden tile (decode_block attn_res_gram_merge).
    "ahead_gram_merge": False,
    # Nor this: the per-call KDA slot-state ring of the verify (verify_model kda_slot_ring_per_call).
    "kda_slot_ring_per_call": False,
    # Nor this: the KDA committed-state stores after the input projection's prefetch fills (verify_model
    # kda_commit_after_w_in).
    "kda_commit_after_w_in": False,
    # Nor this: the KDA slot-state store page ids in registers ahead of the stores (verify_model kda_slot_store_regs).
    "kda_slot_store_regs": False,
    # Nor this: the KDA slot conv-state stores after the output all-reduce (verify_model kda_slot_conv_after_ar).
    "kda_slot_conv_after_ar": False,
    # Nor this: the queue of the KDA verify calls' weight loads (verify_model kda_weight_dma).
    "kda_weight_dma": "pool",
    # Nor this: one weight multiply for all rows of each RMSNorm after an AttnRes mix (decode_block norm_one_mul).
    "norm_one_mul": False,
    # Nor this: the stream squares of the cached-sums AttnRes mixes on ScalarE (decode_block
    # attn_res_cached_act_squares).
    "cached_act_squares": False,
    # Nor this: the first AttnRes bank entries of the verify in the explicit SBUF region (verify_model_2srv; the other
    # entries stay in HBM, common.SplitBank). 0 (the default) keeps the program.
    "attn_res_sbuf_entries": 0,
    # Nor this: the token-stationary MXFP8 KDA input projection of the verify (verify_model_2srv; kda_verify_attention
    # in_proj_x_stationary, bitwise).
    "kda_in_proj_x_stationary": False,
    # Nor this: the KDA state and conv-state loads of the verify after the last prefetch fill of the layer's kda_w_in
    # ring (verify_model_2srv; kda_verify_attention state_loads_after_ring, bitwise).
    "kda_state_loads_after_ring": False,
    # Nor this: the KDA w_out load of the verify after the last two MXFP8 kda_w_in fills (verify_model_2srv;
    # kda_verify_attention w_out_after_w_in_fills, bitwise).
    "kda_w_out_after_w_in_fills": False,
    # Nor this: the KDA prework of the verify in the feed-forward layer before each KDA layer, with the state-slot
    # levers of its form (verify_model_2srv KDA_PREWORK_AHEAD_FORM_SET).
    "kda_prework_ahead": False,
}
"""The glue options of ``latent_moe_decode_2srv`` and their defaults (the default program). ``scale_rows_quadrants``
is the body's own: True traces the static MX scale loads of its expert calls and prefetch streams across the quadrants
(``moe_routed.scale_rows_across_quadrants``; the values do not change). ``scale_rows_queue`` is the body's own
too: the DMA queue of those scale loads (``moe_routed.SCALE_ROWS_QUEUES``; ``static``, the default, is the queue
of the expert weights; ``sync_hw`` / ``act_hw`` give them a hardware-DGE queue of their own, so they do not wait behind
the weight loads; the values do not change). ``router_norm_one_mul`` acts before the
body: ``moe_2srv_body`` takes it out of the glue and gives it to ``decode_block.decode_block_body`` as the body
attribute of the same name (the FP16 router-input norm takes its weight multiply as one instruction for all rows,
``common.rmsnorm_hT`` ``w16_row``: an elementwise multiply, so the values do not change). ``mx_scores_align`` pads the
score run of the G1a router rows to ``_MX_SCORE_SLOT`` words (zeros): the caller must build the sources of
``moe_2srv_collective_hbm`` with the same value (the multi-server entries read it from their ``moe_glue``).
``pe_keep_warm`` runs keep-warm MX matmuls in each static ``latent_gather_mx`` expert call, the score-tied ones before
its first expert MX matmul (``_pe_keep_warm_matmuls``; ``{"kda": form, "mla": form}`` gives each attention kind its own
form); their result reaches the affinities as exact zero bits, so the values do not change.
``routed_per_expert`` runs each token-tiled call of 256 tokens with ``moe_tkg_decode_per_expert``
(``latent_moe_decode_2srv``). ``compact_key_constants`` acts at the multi-server entries, not in the body: with
``routed_compact``, the entry also builds the token keys of the slot tables once for the kernel
(``moe_2srv_compact_constants`` ``key_tiles``), in place of the per-layer iotas, memsets and copies. The values do not
change. ``last_expert_order`` orders the last experts of each static cross-order expert call
(``moe_routed.LAST_EXPERT_ORDERS``), and the body's ``prefetch`` streams take the use order of "up_first"; the
values do not change."""
ROUTED_COMPACT_FORMS = (False, True, "late", "hot_last", "timing_empty_later")
"""The values of the ``routed_compact`` glue key: False keeps the program; True compacts the token-tiled calls; ``late``
compacts them and runs every later slot tile whatever the routed count (the measurement and test form of the one-trip
loops: an empty slot tile adds rows that no token reads, so the outputs do not change); ``hot_last`` compacts them with
the hot-expert-last slot order, and a later slot tile runs only its active experts (``_compact_routed_partial``);
``timing_empty_later`` is a TIMING CONTROL, never served: True with no later slot tile at all (no one-trip loop, no
body), so a call with more than 128 routed tokens loses the tokens past slot 128 (correct only without overflow)."""
ROUTED_COMPACT_ON = True
"""Kernel-side kill switch (default True) of the ``routed_compact`` glue key: False drops every key of
``COMPACT_GLUE_KEYS`` (``routed_compact`` and the compaction keys that need it, ``mx_rows_stream`` included: the stream
rows feed the compaction only) from every glue (``moe_2srv_body``), so each token-tiled call runs uncompacted, the
program of a glue without them, whatever the caller passes (the verify's ``served_defaults`` also drops the cross-expert
order that comes with the compaction, and the entries size the router rows without the stream rows). True keeps the
program."""
TIMING_CONTROLS_ALLOWED = False
"""Test-only switch of the ``routed_compact`` timing control ``timing_empty_later``: False (the default) refuses it at
trace time. A timing harness sets it True for its control arm; no served caller sets it."""
COMPACT_GLUE_KEYS = (
    "routed_compact",
    "weights_before_compact",
    "ungather_skip",
    "prologue_vector_cuts",
    "up_product_split",
    "compact_slot_gather",
    "tail_before_guard",
    "tail_store_halves",
    "compact_alt_tiles",
    "mx_rows_stream",
    "compact_tables_scalar",
    "affinity_sbuf_copy",
    "tail_zero_early",
    "slot_static_first",
)
"""The glue keys of the routed-token compaction (``ROUTED_COMPACT_ON`` drops them)."""
COMPACT_SLOT_GATHERS = ("gpsimd", "scalar", "vector", "split")
"""The engines of the slot-tile gathers of ``routed_compact`` (glue key ``compact_slot_gather``,
``_compact_slot_tile``): ``gpsimd`` (default) runs one ``nc_n_gather`` per D512 tile for the codes and one for the
scales (14 GpSimd gathers a slot tile); ``scalar`` / ``vector`` run one tensor-indirection ``tensor_copy`` for the codes
of all D512 tiles and one for the scales, on ScalarE / VectorE; ``split`` runs the codes copy on ScalarE and the scales
copy on VectorE. The copies move the same bytes as the gathers, so the values do not change."""
COMPACT_TI_MAX_TOKENS = 2 * _PMAX
"""The most tokens of a ``routed_compact`` call with a tensor-indirection slot gather: the gather index moves each
slot's token (or the zero column, index ``count``) through a BF16 0/1 matmul, exact for integers up to 256
(``_snake_index``)."""
COMPACT_RING_WEIGHT_BUFFERS = 4
"""Whole weight buffers of the static expert loop of the first slot tile of ``routed_compact`` (``_compact_routed_partial``)
with the cross-expert order and no prefetched stream, when the loop options give fewer: a deeper expert-weight ring, so more
of the layer's projections load before the loop, in the DMA-light MoE head. The static loop takes 3 to 5 with the
cross-expert order. The loads and their order do not change, so the values do not change; the reported
``weight_buffers`` (3) does not change."""
COMPACT_LATE_WEIGHT_BUFFERS = 2
"""Weight buffers of the one-trip loop of a later slot tile (``routed_compact``): the slot form of
``moe_tkg_decode`` (``_all_experts_in_slots``), which runs only for a call with more than 128 routed tokens."""
HOT_RESIDENT_REUSE = True
"""Kernel-side switch (default True) of the ``hot_last`` later slot tile: True runs its active experts in the static
loop's buffers of the last local expert (E_L - 1; ``moe_tkg_decode`` ``resident_out``), so when that expert is the only
active one of the later tile, its weights and scales need no second load (the same bytes, so the values do not
change). Its static-loop buffers then live until the later tile, which allocates no weight or scale buffers of its
own. False (not the default) gives the program without the reuse (the later tile reloads every active expert into
its own buffers)."""
_HOT_RESIDENT_TRACE = {"calls": 0}
"""Trace-time count of the later slot tiles that took the static loop's resident buffers (``HOT_RESIDENT_REUSE``), for
the tests. No instruction."""
STORE_QUEUES = ("sp", "act", "split")
"""The DMA queues of the routed store and of the out store of the body (``store_queues``): ``sp`` (default) one DMA on
the SP queue; ``act`` one DMA on the ACT hardware queue; ``split`` two DMAs, partitions 0..63 on SP and 64..127 on ACT.
The same bytes, so the values do not change."""
_STORE_HALF = _PMAX // 2
ROWS_DMAS = {
    "sp": _HBM_DMA,
    "act": {"priority": _CRITICAL_DMA_PRIORITY, "dge_mode": nisa.dge_mode.hwdge, "engine": nisa.engine.scalar},
}
_MX_TILE = 512
_GATHER_MAX_ELEMS = 512
"""Most elements of a partition that one GpSimd gather (``nisa.nc_n_gather``) reads: its pool buffer load."""
_VECTOR_TRANSPOSE_MAX = 32
"""Most partitions and free elements of one VectorE transpose (``nisa.nc_transpose``): a data move, exact for FP32."""
_MX_SCALE_ROWS = _PMAX // 8
OUT_PROJ_FORMS = ("reduced", "column")
"""How a layer computes and combines its output (``out_proj_form``):

- ``reduced`` (default): TP rank t multiplies its 112 normed latent columns and its 192 shared-expert activation columns
  by its row shard of the out projection (``moe_out_proj_dp`` [304, 7168], ``tkg_moe_dp`` layout), and the TP32
  all-reduce adds the 32 BF16 partials of the replica's rows.
- ``column``: TP rank t computes the hidden columns 128 h + 4 t + q (h = 0..55, q = 0..3; the hT partitions 4 t .. 4 t + 3)
  of the replica's rows from all 9,728 inputs (``COLUMN_K_TILES``), in one FP32 accumulation rounded to BF16 once, and a
  TP32 all-gather of these columns gives every rank the full rows in the layout of the all-reduce result. The norm runs
  on all 3,584 latent columns of the replica's routed rows, with the arithmetic of ``reduced`` for each element. The
  shared-expert activation reaches every rank of the replica in a second TP32 all-gather (``shared_src`` [4, B * 48] ->
  ``shared_dst`` [128, B * 48]: row 4 t + q holds activation columns 192 t + 48 q .. + 48 of rank t). Its store is traced
  before the expert calls, and the first rows load of the expert calls waits for it, so the bytes of its source can
  go to an expert-loop tile only after the store; the gather is traced after the expert calls and before the routed
  reduction. ``moe_out_proj_dp`` holds the column layout in the same [304, 7168]
  BF16 tensor (``COLUMN_K_TILES``), and ``moe_vectors_dp`` adds
  the 3,584 latent norm weights (``column_norm_cols``). The sum order differs from ``reduced``, so the output does not
  have its bits. The routed rows come from the routed all-reduce result (``routed_reduce`` ``ar``, after the replica
  select), or from the TP32 all-gather of the ``rs_ag`` reduce-scatter blocks (``routed_ag_dst``: the replica's rows in
  the same layout, no replica select). ``stats_gather`` gathers no rows, so the form refuses it.

A peer writes ``shared_dst`` or ``out_dst`` of layer k + 1 only in its gathers of layer k + 1, after its output gather of
layer k, which needs this rank's output columns of layer k, and those need every read of the destinations of layer k.
"""
COLUMN_PRIVATE_SITES = ("moe_out", "moe_routed")
"""The pair-shared sites (``common.PAIR_SHARED_CC_DST_SITES``) that ``out_proj_form`` ``column`` refuses. ``moe_out`` would
put ``out_dst``, the destination of the column form's TP32 output all-gather, in shared HBM; ``moe_routed`` would share
``routed_dst``, whose full rows the column form reads on every rank. Neither is proven, so the column form keeps them
private. ``shared_dst`` and the ``rs_ag`` tensors are always private; ``kda_ar`` holds no MoE tensor."""


def check_column_private(out_proj_form: str, pair_shared: tuple):
    """Refuse ``out_proj_form`` ``column`` with a pair-shared site of ``COLUMN_PRIVATE_SITES``."""
    kernel_assert(
        out_proj_form == "reduced" or not set(pair_shared) & set(COLUMN_PRIVATE_SITES),
        f"pair_shared_cc_dsts {tuple(pair_shared)} with moe_out_proj_form {out_proj_form!r}: the column form keeps its "
        f"output all-gather destination and the routed rows it reads private ({COLUMN_PRIVATE_SITES}): drop those sites "
        "or use moe_out_proj_form 'reduced'",
    )


COLUMN_LATENT_TILES = ROUTED_COLS
"""Latent K-tiles of the ``column`` out projection: partition p of K-tile c is latent input 28 p + c, the element of
the routed layout [128, B, 28] of the routed all-reduce result."""
SHARED_GATHER_COLS = DP_SHARED // _SHARD_PARTITIONS
"""Shared-expert activation columns of one partition of the ``column`` shared all-gather (48)."""
COLUMN_K_TILES = COLUMN_LATENT_TILES + SHARED_GATHER_COLS
"""K-tiles of 128 inputs of the ``column`` out projection (76). ``moe_out_proj_dp`` [304, 7168] of TP rank t viewed as
[128, 76, 224] holds element (p, c, 56 q + h): the weight from latent input 28 p + c (c < 28) or shared input
48 p + c - 28 (c >= 28) to hidden column 128 h + 4 t + q."""
COLUMN_COLS = _SHARD_PARTITIONS * _H_TILES
"""Hidden columns of one TP rank in the ``column`` out projection (224)."""
_COLUMN_WEIGHT_CHUNKS = ((28, 52), (52, 76), (0, 14), (14, 28))
"""K-tile ranges of the ``column`` weight loads: the shared K-tiles first, which the first matmuls read."""


def column_norm_cols(drop_selection: bool = False) -> tuple[int, int]:
    """Return the columns of the 3,584 latent norm weights in ``moe_vectors_dp`` with ``out_proj_form`` ``column``:
    after the columns of ``moe_vector_cols``; column ``first + i`` holds the weight of latent column i.

    Example:
        >>> column_norm_cols()
        (140, 3724)
    """
    first = moe_vector_cols(drop_selection)
    return first, first + LATENT_SIZE


def moe_2srv_vector_cols(drop_selection: bool = False, out_proj_form: str = "reduced") -> int:
    """Return the FP32 columns of ``moe_vectors_dp`` for ``drop_selection`` and ``out_proj_form`` (``OUT_PROJ_FORMS``)."""
    return column_norm_cols(drop_selection)[1] if out_proj_form == "column" else moe_vector_cols(drop_selection)


def _check_out_proj_form(out_proj_form: str, stats_gather: bool = False):
    """Check ``out_proj_form`` (``OUT_PROJ_FORMS``) and its ``stats_gather`` rule."""
    kernel_assert(
        out_proj_form in OUT_PROJ_FORMS, f"out_proj_form must be one of {OUT_PROJ_FORMS}, got {out_proj_form!r}"
    )
    kernel_assert(
        out_proj_form == "reduced" or not stats_gather,
        f"out_proj_form {out_proj_form!r} norms the full routed rows of the replica, and stats_gather gathers only the "
        "sums of squares: use routed_reduce 'ar', or 'rs_ag' without stats_gather",
    )


def moe_2srv_split(world_size: int, routed_tp: int = None, mx_tile: int = None) -> dict:
    """Return the routed-expert split of a world: groups G, experts per rank E_L, parts tp, intermediate I_L.

    The default ``routed_tp`` is the server count (G = 128, E_L = 7). ``mx_tile`` None (default) keeps the 512-column
    MX tile rule (I_L a multiple of 512) and gives the base dict (no tile plan, rank order or layout tag) with
    ``mx_tile`` 512. An ``mx_tile`` gives the split
    of ``moe_routed_layout.routed_split``: any exact EP x TP split whose narrowest MX tile is ``mx_tile``, with
    its tile plan, rank order and layout tag.

    Example:
        >>> moe_2srv_split(256)["local_intermediate"], moe_2srv_split(256, 4, mx_tile=256)["layout_tag"]
        (1536, 'ep64xtp4-t256')
    """
    layout = world_layout(world_size)
    tp = layout["nnodes"] if routed_tp is None else routed_tp
    if mx_tile is not None:
        return routed_split(world_size, tp, mx_tile)
    kernel_assert(tp >= 1 and world_size % tp == 0, f"routed_tp {tp} does not divide the world {world_size}")
    groups = world_size // tp
    kernel_assert(
        groups & (groups - 1) == 0 and NUM_EXPERTS % groups == 0,
        f"{groups} expert groups (world {world_size} / routed_tp {tp}) must be a power of 2 that divides {NUM_EXPERTS}",
    )
    kernel_assert(
        EXPERT_INTERMEDIATE % (tp * _MX_TILE) == 0,
        f"routed_tp {tp} must split the intermediate {EXPERT_INTERMEDIATE} into multiples of {_MX_TILE}",
    )
    return {
        "world": world_size,
        "dp": layout["dp"],
        "routed_tp": tp,
        "groups": groups,
        "local_experts": NUM_EXPERTS // groups,
        "local_intermediate": EXPERT_INTERMEDIATE // tp,
        "mx_tile": _MX_TILE,
    }


def moe_2srv_expert_shapes(world_size: int, routed_tp: int = None, mx_tile: int = None) -> dict:
    """Return the shapes of one rank's per-layer expert tensors (the serving layout of ``moe_tkg_decode``).

    ``mx_tile`` None (default) keeps these shapes; an ``mx_tile`` gives the shapes of the layout of its split
    (``moe_routed_layout.rank_expert_shapes``), the same as these for whole 512 tiles.

    Example:
        >>> moe_2srv_expert_shapes(256)["expert_down_weights"]
        (7, 128, 3, 3584)
        >>> moe_2srv_expert_shapes(256, 4, mx_tile=256)["expert_down_weights"]
        (14, 192, 3584)
    """
    if mx_tile is not None:
        return rank_expert_shapes(moe_2srv_split(world_size, routed_tp, mx_tile), LATENT_SIZE)
    split = moe_2srv_split(world_size, routed_tp)
    e_l, i_l = split["local_experts"], split["local_intermediate"]
    n_d, n_i = LATENT_SIZE // _MX_TILE, i_l // _MX_TILE
    return {
        "expert_gate_up_weights": (e_l, _PMAX, 2, n_d, i_l),
        "expert_gate_up_weights_scale": (e_l, _MX_SCALE_ROWS, 2, n_d, i_l),
        "expert_down_weights": (e_l, _PMAX, n_i, LATENT_SIZE),
        "expert_down_weights_scale": (e_l, _MX_SCALE_ROWS, n_i, LATENT_SIZE),
    }


def token_chunks(tokens: int) -> tuple[tuple[int, int], ...]:
    """Return the expert-call chunks ``(start, count)`` of ``tokens`` gathered tokens, at most 128 each.

    Example:
        >>> token_chunks(512)
        ((0, 128), (128, 128), (256, 128), (384, 128))
    """
    return tuple((start, min(EXPERT_CALL_TOKENS, tokens - start)) for start in range(0, tokens, EXPERT_CALL_TOKENS))


def expert_calls(tokens: int, call_tokens: int) -> tuple[tuple[int, int], ...]:
    """Return the expert calls ``(start, count)`` of ``tokens`` gathered tokens, at most ``call_tokens`` each.

    Example:
        >>> expert_calls(512, 256)
        ((0, 256), (256, 256))
    """
    return tuple((start, min(call_tokens, tokens - start)) for start in range(0, tokens, call_tokens))


def auto_expert_call_tokens(
    tokens: int, routed_tp: int, routed_store_form: str, weight_buffers: int, half_weight_buffers: int
) -> int:
    """Return the expert call size of ``expert_call_tokens="auto"`` for a layer of ``tokens`` gathered tokens.

    Token-tiled calls of ``AUTO_CALL_TOKENS`` need more than 128 tokens, a multiple of 128, routed_tp
    ``AUTO_TILED_ROUTED_TP`` and ``routed_store_form`` ``pe``. Every other layer keeps one ``moe_tkg_decode`` call for
    each 128 tokens, the program of an explicit 128, but with two whole weight buffers (``weight_buffers`` 2 and no
    ``half_weight_buffers``) at most ``AUTO_MAX_TWO_BUFFER_CALLS`` of them. Above ``AUTO_MAX_TOKENS`` tokens, or above
    that call count, the call fails: give an explicit ``expert_call_tokens``.

    Example:
        >>> auto_expert_call_tokens(512, 2, "pe", 2, 0)
        256
    """
    kernel_assert(
        tokens <= AUTO_MAX_TOKENS,
        f"expert_call_tokens 'auto' takes at most {AUTO_MAX_TOKENS} gathered tokens (dp x rows) a layer, got {tokens}: "
        f"give an explicit expert_call_tokens, one of {EXPERT_CALL_FORMS}",
    )
    tiled = tokens > EXPERT_CALL_TOKENS and tokens % _PMAX == 0
    if tiled and routed_tp == AUTO_TILED_ROUTED_TP and routed_store_form == "pe":
        return AUTO_CALL_TOKENS
    calls = len(token_chunks(tokens))
    two_buffers = weight_buffers == _TWO_WEIGHT_BUFFERS and not half_weight_buffers
    kernel_assert(
        not two_buffers or calls <= AUTO_MAX_TWO_BUFFER_CALLS,
        f"expert_call_tokens 'auto' gives at most {AUTO_MAX_TWO_BUFFER_CALLS} calls of {EXPERT_CALL_TOKENS} tokens with "
        f"two whole weight buffers a layer, and {tokens} gathered tokens with routed_tp {routed_tp} and "
        f"routed_store_form {routed_store_form!r} need {calls}: give an explicit expert_call_tokens, one of "
        f"{EXPERT_CALL_FORMS}",
    )
    return EXPERT_CALL_TOKENS


SPLIT_CHECK_NODES = 2
"""Most hosts of a world whose ``moe_2srv_collective_hbm`` checks the default expert split (``moe_2srv_split``). A
larger world checks it only when the caller gives ``routed_tp`` or ``mx_tile``: its tensors need ``dp`` only."""


def moe_2srv_world_split(world_size: int, routed_tp: int = None, mx_tile: int = None) -> dict:
    """Return ``moe_2srv_split`` of the world, with ``mx_tile`` (the sub-512 MX tiles of the expert split) only when it
    is given. None (default) calls the split without ``mx_tile``.

    Example:
        >>> moe_2srv_world_split(256)["groups"]
        128
    """
    return moe_2srv_split(world_size, routed_tp, **({"mx_tile": mx_tile} if mx_tile is not None else {}))


def routed_layers_options(world_size: int, routed_layers, routed_tp: int = None, mx_tile: int = None) -> tuple:
    """Return the ``(routed_tp, mx_tile)`` of the expert split of a per-layer plan of the multi-server entries.

    ``routed_layers`` is None (default: the split of the given ``routed_tp`` and ``mx_tile``) or the
    ``moe_routed_layout.ROUTED_LAYERS_ARGUMENT`` form ``(default_tag, ((layer, tag), ...))``: the split of the
    default tag, which the base expert stacks hold (the layers of other tags take theirs, ``moe_routed_plan``).
    The tag must be a combo of the world (``check_layout_tag``). A served tag (whole 512 tiles) gives ``mx_tile`` None,
    the split of 512-column tiles; a tag of sub-512 tiles gives its tile. A given ``routed_tp`` or ``mx_tile`` must
    agree.

    Example:
        >>> routed_layers_options(512, ("ep128xtp4-t256", ())), routed_layers_options(256, None, 2)
        ((4, 256), (2, None))
    """
    if routed_layers is None:
        return routed_tp, mx_tile
    kernel_assert(
        isinstance(routed_layers, (tuple, list)) and len(routed_layers) == 2,
        f"routed_layers is (default_tag, ((layer, tag), ...)), got {routed_layers!r}",
    )
    default_tag, layer_tags = routed_layers
    for _, tag in layer_tags:
        check_layout_tag(tag, world_size)
    split = check_layout_tag(default_tag, world_size)
    plan = (split["routed_tp"], None if split["layout_form"] == "served" else split["mx_tile"])
    kernel_assert(
        routed_tp in (None, plan[0]) and mx_tile in (None, plan[1]),
        f"routed_layers {default_tag!r} takes routed_tp {plan[0]} and mx_tile {plan[1]}, got routed_tp {routed_tp} and "
        f"mx_tile {mx_tile}",
    )
    return plan


def split_mx_tile(split: dict) -> dict:
    """Return the ``mx_tile`` keyword of the expert-shape calls of ``split``: ``{"mx_tile": tile}`` for a split of a
    tile other than ``_MX_TILE``, else ``{}`` (the 512 tiles: the call takes no new keyword)."""
    tile = split.get("mx_tile")
    return {"mx_tile": tile} if tile is not None and tile != _MX_TILE else {}


def moe_2srv_collective_shapes(
    replica_rows: int,
    world_size: int,
    gather: str = "dp_tp",
    drop_selection: bool = False,
    routed_reduce: str = "ar",
    out_proj_form: str = "reduced",
    rows_latent: str = "bf16",
    latent_gather_mx: bool = False,
    mx_scores_align: bool = False,
    mx_rows_stream: bool = False,
) -> dict:
    """Return ``{key: (shape, dtype)}`` of the first set of ``moe_2srv_collective_hbm`` in its allocation order, without
    the ``rs_ag`` / ``rs_fold`` tensors (``_routed_rs_shapes``). The shapes take the world only through
    ``dp = world / 32``.

    Example:
        >>> moe_2srv_collective_shapes(16, 512)["routed_src"][0]
        (128, 7168)
    """
    tokens = world_layout(world_size)["dp"] * replica_rows
    f32, bf16 = nl.float32, nl.bfloat16
    # mx_scores_align (the glue key of the same name): 65-word G1a rows, the score run padded to 32 words.
    # mx_rows_stream (the glue key of the same name): 64-word token-major G1a rows.
    cols = router_row_cols(drop_selection, rows_latent, latent_gather_mx, mx_scores_align, mx_rows_stream)
    row_dtype = nl.uint32 if latent_gather_mx else f32
    if gather == "dp_tp":
        gathered = {"rows_dst": (tokens, cols), "shard_dst": (TP_SIZE * tokens, cols)}
    else:
        gathered = {"world_dst": (world_size * replica_rows, cols)}
    shapes = {
        "rows_src": ((replica_rows, cols), row_dtype),
        **{key: (shape, row_dtype) for key, shape in gathered.items()},
    }
    if routed_reduce == "ar":
        shapes.update(
            routed_src=((ROUTED_PARTITIONS, tokens * ROUTED_COLS), bf16),
            routed_dst=((ROUTED_PARTITIONS, tokens * ROUTED_COLS), bf16),
        )
    shapes.update(
        out_src=((_SHARD_PARTITIONS if out_proj_form == "column" else _PMAX, replica_rows * _H_TILES), bf16),
        out_dst=((_PMAX, replica_rows * _H_TILES), bf16),
    )
    if out_proj_form == "column":
        shapes.update(
            shared_src=((_SHARD_PARTITIONS, replica_rows * SHARED_GATHER_COLS), bf16),
            shared_dst=((_PMAX, replica_rows * SHARED_GATHER_COLS), bf16),
        )
    return shapes


def moe_2srv_collective_hbm(
    replica_rows: int,
    world_size: int,
    routed_tp: int = None,
    gather: str = "dp_tp",
    name: str = "",
    drop_selection: bool = False,
    routed_reduce: str = "ar",
    stats_gather: bool = False,
    pair_shared: tuple = (),
    out_proj_form: str = "reduced",
    rows_latent: str = "bf16",
    rows_dst_pingpong: bool = True,
    latent_gather_mx: bool = False,
    mx_scores_align: bool = False,
    mx_rows_stream: bool = False,
    mx_tile: int = None,
) -> tuple[dict, dict]:
    """Return the persistent HBM tensors of the body, and their second set for the odd layers.

    For B rows of each replica and T = dp B tokens: ``rows_src`` ``[B, 112]`` FP32, the router rows of this rank;
    with ``gather`` ``dp_tp``, ``rows_dst`` ``[T, 112]`` (the DP gather) and ``shard_dst`` ``[32 T, 112]`` (the TP32
    gather), with ``world``, ``world_dst`` ``[W B, 112]``; ``routed_src`` and ``routed_dst`` ``[128, T * 28]`` BF16
    (the world routed all-reduce); ``out_src`` and ``out_dst`` ``[128, B * 56]`` BF16 (the TP32 output all-reduce). The
    second dict is the same with a second ``rows_src`` and ``routed_src``: the ``pingpong`` set of
    ``decode_block.layer_sources``. With ``drop_selection`` the router rows have 84 words (``router_row_cols``). With
    ``routed_reduce`` ``rs_ag``, the routed tensors are those of ``moe_2srv_routed_hbm`` in place of ``routed_src`` and
    ``routed_dst``. With ``out_proj_form`` ``column`` (``OUT_PROJ_FORMS``), ``out_src`` is [4, B * 56] BF16 (this rank's
    hidden columns), and ``shared_src`` [4, B * 48] / ``shared_dst`` [128, B * 48] BF16 hold the shared all-gather. With
    ``rows_latent`` ``fp8`` the router rows have 26 words fewer (``router_row_cols``). With ``latent_gather_mx`` the
    router rows are the G1a rows (89 words, 61 with ``drop_selection``), and with ``mx_scores_align`` (which needs both)
    65 words: the score run of each token padded to ``_MX_SCORE_SLOT`` words. The body must take the same value
    (``_hbm_views`` checks the shapes).
    With ``rows_dst_pingpong`` (default) and ``gather`` ``dp_tp`` the second set also has its own ``rows_dst``. Two
    consecutive DP gathers then write different tensors, so each one can skip its start handshake: the compiler skips it
    only when nothing touches the destination after the previous gather of the group, and the runtime only when the
    destination does not overlap that gather's destination. False keeps one ``rows_dst`` for all layers.

    Allocate them once for the kernel. A world collective of each layer separates each use of a tensor from the next
    one: a peer writes a destination of layer k + 1 only after the routed all-reduce of layer k, whose source on this
    rank needs every read of the destinations of layer k.

    ``pair_shared`` (``common.PAIR_SHARED_CC_DST_SITES``) puts ``out_dst`` (``moe_out``) and ``routed_dst``
    (``moe_routed``, the ``ar`` routed reduction only) in LNC1 shared HBM; the gathers stay private. () keeps every
    tensor private. ``out_proj_form`` ``column`` refuses both sites (``COLUMN_PRIVATE_SITES``).

    The shapes are those of ``moe_2srv_collective_shapes``: they take the world only through ``dp``. A world of at most
    ``SPLIT_CHECK_NODES`` hosts, or a call with ``routed_tp`` or ``mx_tile``, also checks the expert split
    (``moe_2srv_world_split``); a larger world with neither builds the tensors without it.
    """
    kernel_assert(gather in MOE_GATHERS, f"gather must be one of {MOE_GATHERS}, got {gather!r}")
    _check_routed_reduce(routed_reduce, stats_gather)
    kernel_assert(
        "moe_routed" not in pair_shared or routed_reduce == "ar",
        f"pair_shared moe_routed is the world routed all-reduce: routed_reduce {routed_reduce!r} reduce-scatters, and a "
        "reduce-scatter destination cannot be shared",
    )
    _check_out_proj_form(out_proj_form, stats_gather)
    check_column_private(out_proj_form, pair_shared)
    layout = world_layout(world_size)
    if layout["nnodes"] <= SPLIT_CHECK_NODES or routed_tp is not None or mx_tile is not None:
        moe_2srv_world_split(world_size, routed_tp, mx_tile)
    dp = layout["dp"]
    shapes = moe_2srv_collective_shapes(
        replica_rows,
        world_size,
        gather,
        drop_selection,
        routed_reduce,
        out_proj_form,
        rows_latent,
        latent_gather_mx,
        mx_scores_align,
        mx_rows_stream,
    )
    shared = {"out_dst": "moe_out" in pair_shared, "routed_dst": "moe_routed" in pair_shared}
    first = {
        key: nl.ndarray(
            shape, dtype=dtype, buffer=pair_shared_buffer(shared.get(key, False)), name=f"moe2s{name}_{key}"
        )
        for key, (shape, dtype) in shapes.items()
    }
    second_keys = ("rows_src", "routed_src") if routed_reduce == "ar" else ("rows_src",)
    if rows_dst_pingpong and gather == "dp_tp":
        second_keys += ("rows_dst",)
    second = {
        **first,
        **{
            key: nl.ndarray(shapes[key][0], dtype=shapes[key][1], buffer=nl.private_hbm, name=f"pp_moe2s{name}_{key}")
            for key in second_keys
        },
    }
    if routed_reduce != "ar":
        routed_first, routed_second = moe_2srv_routed_hbm(replica_rows, dp, stats_gather, name, routed_reduce)
        first.update(routed_first)
        second.update(routed_second)
    return first, second


def moe_2srv_routed_hbm(
    replica_rows: int, dp: int, stats_gather: bool = False, name: str = "", routed_reduce: str = "rs_ag"
) -> tuple[dict, dict]:
    """Return the persistent HBM tensors of the ``rs_ag`` routed reduction (``ROUTED_REDUCES``), and their second set
    for the odd layers. Every tensor has a second one, so two consecutive layers use different tensors.

    For B rows of each replica: ``routed_rs_src`` ``[dp * 128, B * 28]`` BF16, the routed partial of every token in the
    replica-major layout (row 128 r + p, column 28 b + c: latent column 28 p + c of row b of replica r), so world
    reduce-scatter block 32 r + t (rows 128 r + 4 t .. + 4) is rank (r, t)'s B rows x its latent columns
    [112 t, 112 t + 112); ``routed_rs_dst`` ``[4, B * 28]`` BF16, that block. Then the TP32 all-gather of the replica,
    without ``stats_gather``: ``routed_ag_dst`` ``[128, B * 28]`` BF16, the replica's full routed rows (element
    (p, 28 b + c)); with ``stats_gather``: ``stats_src`` ``[B, 1]`` FP32, the sum of squares of each row over this rank's
    112 columns, and ``stats_dst`` ``[32 B, 1]`` FP32 (row B t + b: the sum of TP rank t). ``routed_reduce`` ``rs_fold``
    takes the reduce-scatter pair and, in place of the all-gather tensors, the FP32 output all-reduce pair
    ``fold_src`` / ``fold_dst`` ``[128, 2 B * 56 + B]`` (``fold_out_reduce``).

    A peer writes a destination of layer k again only in layer k + 2, after the world reduce-scatter of layer k + 1,
    whose source on this rank needs every read of the destinations of layer k.
    """
    shapes = _routed_rs_shapes(replica_rows, dp, stats_gather, routed_reduce)
    sets = tuple(
        {
            key: nl.ndarray(
                shape,
                dtype=nl.float32 if key.startswith(("stats_", "fold_")) else nl.bfloat16,
                buffer=nl.private_hbm,
                name=f"{prefix}moe2s{name}_{key}",
            )
            for key, shape in shapes.items()
        }
        for prefix in ("", "pp_")
    )
    return sets[0], sets[1]


def _check_routed_reduce(routed_reduce: str, stats_gather: bool):
    """Check ``routed_reduce`` (``ROUTED_REDUCES``) and its ``stats_gather`` sub-key."""
    kernel_assert(
        routed_reduce in ROUTED_REDUCES, f"routed_reduce must be one of {ROUTED_REDUCES}, got {routed_reduce!r}"
    )
    kernel_assert(
        not stats_gather or routed_reduce == "rs_ag",
        f"stats_gather gathers the norm statistics of routed_reduce 'rs_ag', got routed_reduce {routed_reduce!r}",
    )


def check_routed_layout(routed_layout: str, routed_reduce: str, rows: int, out_proj_form: str = "reduced"):
    """Check ``routed_layout`` (``ROUTED_LAYOUTS``): ``qsplit`` needs ``routed_reduce`` ``ar``, the ``reduced``
    out-projection and an even B."""
    kernel_assert(
        routed_layout in ROUTED_LAYOUTS, f"routed_layout must be one of {ROUTED_LAYOUTS}, got {routed_layout!r}"
    )
    kernel_assert(
        routed_layout == "natural" or (routed_reduce == "ar" and out_proj_form == "reduced" and rows % 2 == 0),
        f"routed_layout 'qsplit' takes routed_reduce 'ar', the 'reduced' out-projection and an even B, got "
        f"{routed_reduce!r}, {out_proj_form!r} and B {rows}",
    )


def _routed_halves(tensor: nl.NkiTensor, dp: int, rows: int) -> nl.NkiTensor:
    """Return the ``qsplit`` view of a routed tensor ``[128, dp B * 28]``: ``[128 * 2 * dp, B / 2 * 28]``, row
    (2 p + h) dp + r holds the 28-column groups of rows h B / 2 .. (h + 1) B / 2 - 1 of replica r for partition p."""
    return tensor.reshape((ROUTED_PARTITIONS * 2 * dp, rows // 2 * ROUTED_COLS))


def check_glue_keys(norm_tail: str, rows_queue: str, routed_reduce: str, rows: int):
    """Check the glue options of the body: ``norm_tail`` (``NORM_TAILS``; ``token_major`` needs ``routed_reduce`` ``ar``
    and at most ``_VECTOR_TRANSPOSE_MAX`` rows) and ``rows_queue`` (``ROWS_QUEUES``)."""
    kernel_assert(norm_tail in NORM_TAILS, f"norm_tail must be one of {NORM_TAILS}, got {norm_tail!r}")
    kernel_assert(
        norm_tail == "rows" or (routed_reduce == "ar" and rows <= _VECTOR_TRANSPOSE_MAX),
        f"norm_tail 'token_major' takes routed_reduce 'ar' and at most {_VECTOR_TRANSPOSE_MAX} rows of a replica (one "
        f"VectorE transpose of the inverse RMS), got {routed_reduce!r} and {rows} rows",
    )
    kernel_assert(rows_queue in ROWS_QUEUES, f"rows_queue must be one of {ROWS_QUEUES}, got {rows_queue!r}")


def _instruction(handle):
    """Return the instruction of a traced ISA call, for a dependency edge."""
    return handle if hasattr(handle, "depends_on") else nisa.NkiInstruction(handle)


def latent_moe_decode_2srv(
    x_hT: nl.NkiTensor,
    moe_in_proj_dp: nl.NkiTensor,
    moe_out_proj_dp: nl.NkiTensor,
    moe_vectors_dp: nl.NkiTensor,
    expert_gate_up_weights: nl.NkiTensor,
    expert_gate_up_weights_scale: nl.NkiTensor,
    expert_down_weights: nl.NkiTensor,
    expert_down_weights_scale: nl.NkiTensor,
    rank_info: nl.NkiTensor,
    dp_group: ReplicaGroup,
    world_group: ReplicaGroup,
    tp_group: ReplicaGroup,
    tag: str,
    sources: dict,
    split: dict,
    weight_buffers: int = 3,
    scale_buffers: int = 1,
    half_weight_buffers: int = 0,
    routed_store_form: str = "relayout",
    early_loads: bool = True,
    gather: str = "dp_tp",
    wide_half_weight_buffers: int = WIDE_HALF_WEIGHT_BUFFERS,
    expert_call_tokens: int | str = EXPERT_CALL_AUTO,
    out_proj_form: str = "reduced",
    drop_selection: bool = False,
    routed_reduce: str = "ar",
    stats_gather: bool = False,
    transpose_lookahead: int = None,
    cross_expert_order: bool = False,
    glue_hooks: dict = None,
    quantize_first: bool = False,
    pad_rows: nl.NkiTensor = None,
    cross_expert_transpose_blocks: bool = False,
    token_tiles_transpose_blocks: bool = False,
    weights_after_gather: bool = False,
    in_proj_sb: nl.NkiTensor = None,
    head_row_chunks: bool = False,
    static_alt_tiles: bool | str = False,
    transpose_block: bool = False,
    prefetch=None,
    expert_queues: int = 1,
    rows_latent: str = "bf16",
    rows_load_split: int = 1,
    routed_own_rows: bool = False,
    norm_tail: str = "rows",
    rows_queue: str = "sp",
    router_first: bool = False,
    out_proj_early: bool = False,
    out_proj_after_rows: bool | str = False,
    selectors_once: bool = False,
    routed_compact: bool | str = False,
    store_queues: str = "sp",
    weights_before_compact: bool = False,
    weights_after_rows: bool = False,
    ungather_skip: bool = False,
    accum_memset_after_rows: bool = False,
    prologue_vector_cuts: bool = False,
    up_product_split: bool = False,
    rows_align: bool = False,
    rows_pairs: bool = False,
    routed_per_expert: bool | int | dict = False,
    routed_layout: str = "natural",
    compact_slot_gather: str = "gpsimd",
    in_proj_lanes_view: bool = False,
    tail_before_guard: bool = False,
    tail_store_halves: bool = False,
    compact_alt_tiles: bool | str = False,
    first_add_zero: bool = False,
    last_expert_order: str = "default",
    compact_tables_scalar: bool = False,
    affinity_sbuf_copy: bool = False,
    out_proj_priority: int = None,
    tail_zero_early: bool = False,
    slot_static_first: bool = False,
    out_proj_before_in_proj: bool = False,
    scale_fill_after_prefetch: bool = False,
    router_dtype: str = "bf16",
    router_weight: nl.NkiTensor = None,
    x16_hT: nl.NkiTensor = None,
    in_proj_scale: nl.NkiTensor = None,
    out_proj_scale: nl.NkiTensor = None,
    latent_gather_mx: bool = False,
    mx_load_order: str = "quadrant",
    mx_scale_queue: str = "act",
    mx_reload_order: str = "default",
    mx_scores_align: bool = False,
    mx_rows_stream: bool = False,
    pe_keep_warm: bool | int | dict | None = False,
    keep_warm_lead: str | None = None,
    routed_dynamic: bool = ROUTED_DYNAMIC_DEFAULT,
    routed_dynamic_pack: int = None,
    routed_dynamic_csr: str = None,
) -> nl.NkiTensor:
    """Run one Kimi-K3 latent MoE decode layer of a multi-server world and return this replica's rows.

    Not ``@nki.jit``: a decode kernel inlines one call for each MoE layer.

    Dimensions:
        H: Hidden size, 7168.
        B: Rows of each DP replica, 1 to 128.
        T: Gathered tokens, dp B. Token r B + b is row b of replica r. The expert calls take consecutive tokens:
            chunks of at most 128 (``token_chunks``), or token-tiled calls (``expert_call_tokens``).
        Rank g = 32 r + t is TP rank t of replica r. It holds the experts of group g % G, part g // G.

    Args:
        x_hT (nl.NkiTensor): [128, B, 56] BF16 SBUF, this replica's normalized hidden, token-major hT.
        moe_in_proj_dp, moe_out_proj_dp, moe_vectors_dp (nl.NkiTensor): the DP projections of TP rank t, as
            ``tkg_moe_dp.latent_moe_decode_dp``: [128, 56, 524] BF16, [304, H] BF16, [1, 140] FP32.
        expert_gate_up_weights .. expert_down_weights_scale (nl.NkiTensor): this rank's E_L experts, intermediate part
            g // G (``moe_2srv_expert_shapes``), packed MXFP4 (uint16 words or the float4_e2m1fn_x4 view) and uint8
            E8M0 scales, in the serving layout of ``moe_tkg_decode``.
        rank_info (nl.NkiTensor): [1, 4] INT32 HBM, (global rank, DP rank, TP rank, 0).
        dp_group (ReplicaGroup): [[t + 32 r for r in 0..dp - 1] for t in 0..31] (spans the servers).
        world_group (ReplicaGroup): [[0 .. W - 1]].
        tp_group (ReplicaGroup): [[32 r .. 32 r + 31] for r in 0..dp - 1] (inside one server).
        tag (str): Unique per-layer string. It names the collectives and the DMAs.
        sources (dict): the first or second dict of ``moe_2srv_collective_hbm`` for B rows and ``gather``.
        split (dict): ``moe_2srv_split`` of the world.
        weight_buffers, scale_buffers, half_weight_buffers, early_loads: the static expert loop of ``moe_tkg_decode``.
        routed_store_form (str): ``relayout`` (default): the expert output goes to HBM, and a DMA moves it into the
            routed layout. ``pe``: the expert loop writes the routed all-reduce source in its [128, T, 28] layout
            (``column_groups_out`` of ``moe_tkg_decode``). The values do not change.
        gather (str): the router-row gathers, ``MOE_GATHERS``. The values do not change.
        wide_half_weight_buffers (int): above ``WIDE_ROWS`` rows, the half-projection weight buffers of the
            whole-expert form (I_L = 3072) when ``half_weight_buffers`` is 0: its two whole buffers do not fit beside
            the 64-row attention of the 93-layer program. The values do not change.
        expert_call_tokens (int | str): the tokens of one expert call. "auto" (default): ``auto_expert_call_tokens``
            of dp x rows (at most ``AUTO_MAX_TOKENS``). Or one of ``EXPERT_CALL_FORMS``: 128 runs one
            ``moe_tkg_decode`` call for each 128 tokens; 256 or 512 runs one ``moe_tkg_decode_token_tiles`` call for
            each 256 or 512 tokens (a last call of 128 tokens calls ``moe_tkg_decode``) and needs ``routed_store_form``
            ``pe`` and dp x rows a multiple of 128. The values do not change.
        out_proj_form (str): ``reduced`` (default) or ``column`` (``OUT_PROJ_FORMS``): the column out projection and
            the TP32 output all-gather in place of the TP32 output all-reduce. ``column`` needs the column layout of
            ``moe_out_proj_dp``, ``moe_vectors_dp`` with the latent norm weights (``column_norm_cols``) and ``sources``
            from ``moe_2srv_collective_hbm`` with the same form. With ``routed_reduce`` ``rs_ag`` (no ``stats_gather``)
            the norm reads the replica's rows of the routed TP32 all-gather. The sum order differs: an accuracy-class
            option.
        drop_selection (bool): the router rows leave out the 28 selection values (score + bias): 84 words in place of
            112, so the DP and TP32 row gathers move 25 % fewer bytes. Each rank adds the bias of all 896 experts
            (``ROUTER_BIAS_COLS`` of ``moe_vectors_dp`` [1, 1036]) to the gathered scores on the Vector engine: the
            same FP32 add on the same values as the source add. ``sources`` must come from ``moe_2srv_collective_hbm``
            with the same ``drop_selection``.
        routed_reduce (str): the reduction of the routed partials (``ROUTED_REDUCES``): ``ar`` (default) the world
            all-reduce of the whole routed latent; ``rs_ag`` the world reduce-scatter of the replica-major blocks, then
            the TP32 all-gather for the norm statistics; the rank's own 112 columns come from its block. ``rs_ag``
            rounds differently (other reduction owners): an accuracy-class option. ``rs_fold`` the same reduce-scatter,
            then the out-projection of the shard times the norm weight and one FP32 TP32 all-reduce of the partial
            outputs and the sums of squares; the rstd scales the latent part after it (two collectives a layer, as
            ``ar``). An accuracy-class option: the normed latent is not rounded to BF16 before the out-projection and
            the output sums in FP32.
        rows_latent (str): how the router rows carry the latent (``ROWS_LATENTS``): ``bf16`` (default) keeps the
            program; ``fp8`` sends 112 float8_e4m3fn values and one FP32 scale (amax / 448) for each rank's shard of a
            token, 26 words fewer a row, and expands them to BF16 before the expert loop. An accuracy-class option.
            ``sources`` must come from ``moe_2srv_collective_hbm`` with the same ``rows_latent``.
        stats_gather (bool): with ``rs_ag``, the TP32 all-gather carries the FP32 sum of squares of each row over each
            rank's 112 columns (summed in rank order on every rank) instead of the full rows. False (default).
        transpose_lookahead (int): None (default) keeps the program; 1 or 2 is ``transpose_lookahead`` of
            ``moe_tkg_decode``, the activation chunks transposed before each expert's first down matmul. Only with
            calls of at most 128 tokens (``expert_call_tokens`` 128). The values do not change.
        cross_expert_order (bool): False (default) keeps the program; True is ``cross_expert_order`` of
            ``moe_tkg_decode``: the gate matmuls of the next expert run between the up and the down matmuls of each
            expert. Needs 3 to 5 whole weight buffers and calls of at most 128 tokens; with ``routed_compact`` the
            first slot tile's static call takes it (the later slot tiles keep their own loops). Without
            ``routed_compact``, a token-tiled call (more than 128 tokens, ``moe_tkg_decode_token_tiles``) takes it
            too, with 3 whole weight buffers and without ``cross_expert_transpose_blocks`` or
            ``token_tiles_transpose_blocks``: each step of the order runs every token tile. ``transpose_block``
            applies to both call forms. The values do not change.
        glue_hooks (dict): None (default), or functions (``GLUE_HOOKS``) that the layer calls to let the caller trace
            work of the next layer in its glue. ``routed_load`` is called with the gate collective of the routed
            reduction and the DMA that loads its result, right after that load: the routed all-reduce and its
            ``routed_load`` (``routed_reduce`` ``ar``), or the TP32 all-gather of ``rs_ag`` and the load of its rows
            (the rows load), or with ``stats_gather`` the statistics all-gather and its load.
            ``routed_trigger`` is called with the first collective of the routed reduction right after its trigger:
            the world all-reduce of ``ar``, the world reduce-scatter of ``rs_ag``. Its trigger follows the store of the
            routed partials, so an instruction the hook orders after it follows the expert calls' staging copies.
            ``out_trigger`` is called with the output collective right after its trigger, before the load of its
            result: the TP32 output all-reduce, or the TP32 output all-gather of ``out_proj_form`` ``column``. Every
            tail calls each given hook once. ``tail_gathers`` (``routed_compact`` calls only) is called with the
            token-order gathers of the compaction's tail after the later slot tile's guard, right after they are
            traced (``decode_block`` ``TAIL_POOL_PIN``). The values do not change.
        quantize_first (bool): False (default) keeps the program. True: each expert call quantizes its latent to the
            MXFP8 expert input (``moe_routed._quantize_natural_latent``, on the caller's latent tile) before the
            routing affinities of its tokens (and before the bias add of ``drop_selection``), and gives
            ``moe_tkg_decode`` the codes and scales (``quantized_latent``). The Vector engine runs its instructions in
            program order, so the quantize does not wait behind the affinity instructions, which the expert loop
            needs only after its first down matmuls; the transposes read the caller's tile, without the copy DMA of
            ``moe_tkg_decode``. The same instructions run on the same engines in another order, so the values do not
            change. Needs calls of at most 128 tokens (``expert_call_tokens`` 128 or "auto" giving 128) and a multiple
            of 4 tokens in each call.
        pad_rows (nl.NkiTensor): None (default) keeps the program. The HBM int32 ``[R, 1]`` padding-row input of this
            replica's rows (1 for a padding row; ``tkg_moe.pad_row_flags``): their router scores become 0 before the
            router-row gather (in the BF16 rows, or in the ``latent_gather_mx`` score rows with ``routed_compact``), so
            they route to no expert (affinity 0 on every rank; their routed partial is 0). The other rows keep their
            bits.
        cross_expert_transpose_blocks (bool): False (default) keeps the program; True is
            ``cross_expert_transpose_blocks`` of ``moe_tkg_decode``: the activation transposes of the cross-expert
            order run in tile blocks between the gate chunks of the next expert, with no MX matmul inside a block.
            Needs ``cross_expert_order``. The values do not change.
        token_tiles_transpose_blocks (bool): False (default) keeps the program; True is ``transpose_blocks`` of
            ``moe_tkg_decode_token_tiles`` in the token-tiled expert calls: one fixed TensorE order with every PE
            transpose in a block between MX matmul groups. Needs a call of more than 128 tokens. The values do not
            change.
        weights_after_gather (bool): False (default) keeps the program. True: each expert weight and scale load of
            the expert calls that is the first load into its destination view (the stream start of the static loop)
            gets a schedule dependency on the last router-row gather (the TP32 all-gather of ``dp_tp``, the world
            all-gather of ``world``; ``_stream_start_after``).
            neuronx-cc's DMA prefetch pass skips a DMA that depends on a collective, so it does not move these loads
            to right after the DP gather with an ordered edge. Without the edges it moves them there, so they wait for
            the end of that gather and stream during the TP32 gather, whose last phase they slow down. They start after
            the gather, once the matmuls that read the bytes of their buffers end (the shared gate/up matmuls, whose
            in-projection tile the compiler reuses for the weight ring). These are schedule edges only (no hardware
            wait), so the values do not change.
        in_proj_sb (nl.NkiTensor): None (default) loads ``moe_in_proj_dp`` here. Else the [128, 56, 524] BF16 SBUF
            tile of ``load_moe_in_proj_2srv`` that the caller loaded earlier from the same ``moe_in_proj_dp``
            (``decode_block_body`` ``moe_in_proj_at_kda_reduce``). The values do not change.
        head_row_chunks (bool): False (default) keeps the program. True: each expert call loads its gathered router
            rows in 4 DMAs of 8 TP blocks (896 latent columns) each, copies the latent of each chunk on its own, and
            swizzles and quantizes each D512 tile of the latent as soon as the chunks that hold its 512 columns are
            copied (``_chunked_latent_quantize``), so the latent prologue of the first chunks runs while the later
            chunks stream, and the first MXFP8 matmuls start before the last chunk lands. The same instructions on the
            same bytes, split by TP block and by D512 tile (an MX scale group never spans two D512 tiles), so the values
            do not change. Needs ``quantize_first``, ``gather`` ``dp_tp``, calls of at most 128 tokens and
            ``rows_latent`` ``bf16``.
        static_alt_tiles (bool | str): False (default) keeps the program. Else ``static_alt_tiles`` of each static
            ``moe_tkg_decode`` call and of each token-tiled ``moe_tkg_decode_token_tiles`` call
            (``moe_routed.STATIC_ALT_TILE_FORMS``): the gate sigmoid / up tanh tiles ("act"), and the down PSUM
            copies ("act_copy"), and the activation evictions (True) come from two buffers each, so that under SBUF
            pressure the compiler cannot stack them on one address. Not with ``token_tiles_transpose_blocks``. The
            values do not change.
        transpose_block (bool): False (default) keeps the program; True is ``transpose_block`` of ``moe_tkg_decode``
            (with ``cross_expert_order``): the activation transposes of each expert run in one block with no MX matmul
            inside. The values do not change.
        prefetch: None (default), or the stream of ``expert_prefetch_stream`` for these expert weights, whose first loads
            the caller issued earlier (for example during the attention of the layer). The one expert call of the
            layer (at most 128 tokens, whole buffers) takes it, with its number of weight buffers. With
            ``cross_expert_order`` the stream loads in the cross-expert order (``use_order``, 3 whole buffers). A
            token-tiled call (256 or 512 tokens) takes a stream of 1 to 3 whole buffers in the order of the call;
            with ``routed_compact`` the static call of the first slot tile takes it, in the order of that call (a later
            slot tile loads its own experts). The values do not change.
        expert_queues (int): the DMA queues of the static expert loop's weight loads (``moe_tkg_decode``
            ``expert_queues``, ``EXPERT_QUEUES``): 1 (default) one queue; 2 puts the odd experts' weights on a second
            queue. A ``prefetch`` stream must have the same value. The values do not change.
        rows_load_split (int): 1 (default) keeps the program: one DMA loads the gathered router rows of an expert call
            (or of each 128-token tile of a longer call) on the SP queue, then VectorE copies their latent. 2 or 4
            (``ROWS_LOAD_SPLITS``, ``dp_tp`` gather): the rows load in that many parts of the 32 TP blocks, on the SP
            and the ACT hardware queues in turn, and the latent of each part is written as soon as it is in (the BF16
            copy, or with ``rows_latent`` ``fp8`` the expand of the part's FP8 words and scales), so the latent of a
            part runs beside the load of the next. The same bytes and the same per-element operation, so the values do
            not change. Not with ``head_row_chunks``, the other split of the rows load.
        routed_own_rows (bool): False (default) keeps the program: each rank loads the whole routed all-reduce result
            (all dp replicas) and copies its replica's rows out of it. True (``routed_reduce`` ``ar``): one DMA at the
            offset of this rank's replica (``rank_info[1]``, a scalar-offset access) loads only those rows, 1 / dp of
            the bytes, with no copies. The same values.
        norm_tail (str): with ``routed_reduce`` ``ar``, how the layer takes the normed latent of its replica's rows
            (``NORM_TAILS``): ``rows`` (default) or ``token_major`` (at most ``_VECTOR_TRANSPOSE_MAX`` rows). The values
            do not change.
        rows_queue (str): the DMA queue of the router-row store and of the router-bias load (``ROWS_QUEUES``):
            ``sp`` (default) or ``act``. With ``drop_selection`` the store then waits only for the ScalarE cast and
            sigmoid of the rows. The values do not change.
        router_first (bool): the shared gate / up matmuls follow the last router matmul on TensorE (a schedule edge),
            so the router rows are ready before the shared work. False (default) leaves the order to the scheduler.
            The values do not change.
        out_proj_early (bool): load the out-projection weights before the expert calls, not after the routed
            reduction, so their tile does not take the bytes of the expert-loop tiles. The values do not change. The
            ``reduced`` out projection only: ``out_proj_form`` ``column`` loads its own column-layout weights.
        out_proj_after_rows (bool | str): ``OUT_PROJ_AFTER_ROWS_FORMS``. False (default) keeps the program.
            ``rows_gather`` / ``shard_gather`` (with ``out_proj_early``, the MXFP8 out projection and the ``dp_tp``
            gather): each early out-projection load (the MXFP8 chunks and the scale) follows the DP / TP32 router-row
            gather in the schedule, on the GpSimd software DGE queue, so the loads are not in flight beside the
            router-row store. ``shard_gather_done``: as ``shard_gather``, and each load also waits for the TP32 gather
            to complete, so it is not in that gather's flight either. With ``latent_gather_mx`` too: its MXFP8 router
            rows then record their two gathers. The values do not change.
        selectors_once (bool): take the rank selectors of ``moe_2srv_glue_selectors`` that the caller built once for
            the kernel (``rank_info`` keeps them), not new ones in each layer. The values do not change.
        routed_compact (bool | str): ``ROUTED_COMPACT_FORMS``. False (default) keeps the program. True: each
            token-tiled expert call (more than 128 tokens) runs this rank's experts only on its routed tokens
            (``_compact_routed_partial``). The routed tokens (any nonzero local affinity) take slots in token order;
            the MXFP8 codes of their latent (quantized per 128-token tile, as the token-tiled call does) are gathered
            into one 128-slot tile on GpSimd, and the static loop of ``moe_tkg_decode`` runs on it, so a call with at
            most 128 routed tokens moves the weights once and runs the matmuls of one tile instead of one per 128
            tokens. A later slot tile runs in a one-trip loop (the slot form of ``moe_tkg_decode``) only when the call
            has more routed tokens. An indirect DMA puts each token's output row back in token order, and a token with
            no local expert takes a zero row, the row of the default call. Each token's row has the bits of the default
            call when its latent and outputs are finite; a non-finite value stays in its own token's row. Needs the
            ``dp_tp`` gather and a token-tiled call; ``rows_load_split`` splits the rows load of each 128-token tile as
            in the default call. ``late`` runs every later slot tile whatever the count (the loop cost, for a
            measurement or a test); the outputs do not change. A padding row (``pad_rows``; the 2-server verify entry
            gives them from the runner's ``live_rows`` with this key) has no routed expert, so it takes no slot (also
            with ``latent_gather_mx``, whose score rows zero the scores of the padding rows as the BF16 rows do). With
            ``latent_gather_mx`` (both keys off by default) the call loads the MXFP8 codes of all its tokens straight
            into the gather source (``_load_mx_codes``) and the scores of each 128-token tile (``_load_mx_scores``): no
            latent copy, swizzle or quantize, and the codes of a token are the bytes of the per-tile quantize of the
            BF16-rows path. Each 128-token tile must hold whole replicas.
        routed_layout (str): ``ROUTED_LAYOUTS``. ``natural`` (default) keeps the program. ``qsplit`` (``routed_reduce``
            ``ar``, the ``reduced`` out-projection, even B) puts each row of the routed tensors in the owner block of
            its query-row half, so the world all-reduce adds each element in the same order whatever the DP rank of its
            token. The same bytes; the values change.
        store_queues (str): ``STORE_QUEUES``. ``sp`` (default) keeps the program. ``act`` stores the routed partial of
            each expert call and the out-projection partial (the source of the TP32 out all-reduce) with the ACT
            hardware queue, so they do not wait behind earlier SP loads (such as the next KDA layer's slot-page loads in
            an MLA-MoE tail); ``split`` stores each in two DMAs, partitions 0..63 on SP and 64..127 on ACT. The same
            bytes, so the values do not change. With ``routed_reduce`` ``ar`` or ``rs_ag`` and the ``reduced``
            out-projection only.
        weights_before_compact (bool): False (default) keeps the program. True (with ``routed_compact`` and
            ``out_proj_early``): the first GpSimd and SP instructions of the compaction follow the layer's early
            out-projection weight loads (the MXFP8 chunks and the scale) and the norm-weight load in the schedule, so
            those loads issue before the compaction instead of just before the later slot tile's guard, whose fence
            then drains nothing of them. Schedule edges only: the values do not change.
        weights_after_rows (bool): False (default) keeps the program. True (``dp_tp`` gather, no ``latent_gather_mx``):
            each expert weight and scale load of the expert calls that is the first load into its destination view
            (the stream start of the static loop, ``_stream_start_after``) follows the TP32 router-row all-gather in
            the schedule, as with ``weights_after_gather``, and also waits (a semaphore wait) for the router-row loads
            of the layer traced before it, the first readers of that all-gather's result. The engine of a weight load
            otherwise issues it inside the flight of the DP8 and TP32 row all-gathers, where it shares the DMA rings
            with them; with the wait the stream starts once the rows are in. The loads and their bytes do not change,
            so the values do not change.
        ungather_skip (bool): False (default) keeps the program. True (with ``routed_compact``): the token-order gather
            of the slot-output rows reads only the routed tokens' rows. An unrouted token takes index -1, which the
            indirect DMA skips (out of bounds), and its row of the zeroed (VectorE memset) destination tile stays +0,
            the bits of the zero row it gathered before. The gather then moves the routed rows only (about 30 of 256
            in a decode step) in place of all 256 rows (1.8 MB). The values do not change.
        accum_memset_after_rows (bool): False (default) keeps the program. True (``dp_tp`` gather, only static expert
            calls of at most 128 tokens, no ``routed_compact``; the body refuses other calls, where the key would do
            nothing): the zero fill of each static call's FP32 output accumulator runs on VectorE after the DP
            router-row gather in the schedule (``moe_tkg_decode`` ``accum_memset``), in the gather's wait, not in the
            chain of the router and the norm before it. The same zero bits. It overrides ``first_add_zero``: the calls
            get their zero fill back (``moe_tkg_decode`` takes ``first_add_zero`` only without ``accum_memset``), in
            the gather's wait.
        prologue_vector_cuts (bool): False (default) keeps the program. True (with ``routed_compact``): two VectorE
            cuts in the compaction's prologue, where VectorE paces the first expert matmul. (1) Each token tile's
            local affinities take one fused scalar_tensor_tensor (multiply by the group one-hot, sum over the
            groups) for each local expert, in place of the [T, E_L, G] multiply and the separate reduce: the same
            products and sums. (2) The FP32 output accumulator of the first slot tile's expert loop is zeroed on
            GpSimd after the slot gathers. The values do not change.
        up_product_split (bool): False (default) keeps the program. True (with ``routed_compact``): the first slot
            tile's static loop runs each up chunk's SiTU product as one VectorE instruction for each 128 columns
            (``moe_tkg_decode`` ``up_product_split``), so the activation transposes of the chunk start before the whole
            product is done. The same elementwise operations: the values do not change.
        rows_align (bool): the glue key of the 64-word ``fp8a`` router rows; it must match ``rows_latent`` ``fp8a`` (the
            sources must come from ``moe_2srv_collective_hbm`` with the same ``rows_latent``). The values do not change.
        rows_pairs (bool): False (default) keeps the program. True (``check_rows_pairs``) loads the gathered rows of
            each 256 tokens of a token-tiled expert call in one DMA, two tokens a partition (``_load_pair_rows``):
            4,096 elements of 464 B in place of two DMAs of 4,096 elements of 232 B. Token tile 2 i + j then holds the
            tokens 256 i + 2 p + j, and the call writes its routed partial with the same map
            (``moe_tkg_decode_token_tiles`` ``token_pairs``). Not with ``rows_align`` or ``routed_compact``. The values
            do not change.
        routed_per_expert (bool | int | dict): False (default) keeps the program. True runs each token-tiled expert
            call of 256 tokens with ``moe_tkg_decode_per_expert``: each local expert on its own routed tokens (one block
            of up to 128 slots in the static loop, and a late block in one dynamic loop for an expert with more than
            128), in place of both token tiles. The values do not change. The layers share one HBM scratch for each HBM
            set of the body (``per_expert_scratch``). A column count in place of True is the ``combine_cols`` of the
            calls (True: ``PER_EXPERT_COMBINE_COLS``); a dict sets ``combine_cols``, ``slot_gather``
            (``PER_EXPERT_SLOT_GATHERS``) and / or ``weight_buffers``. The calls take their own whole weight buffers:
            ``PER_EXPERT_WEIGHT_BUFFERS`` (2), or 3 with ``cross_expert_order``, in place of ``weight_buffers``.
            Needs token-tiled calls of 256 tokens, whole weight buffers and one expert queue; not with
            ``routed_compact``, ``latent_gather_mx``, ``static_alt_tiles``, ``token_tiles_transpose_blocks``,
            ``cross_expert_transpose_blocks``, ``rows_pairs`` or ``prefetch``. With ``cross_expert_order`` (and
            ``transpose_block``) the per-expert calls run the cross-expert order.
        compact_slot_gather (str): ``COMPACT_SLOT_GATHERS``. ``gpsimd`` (default) keeps the program. ``scalar``,
            ``vector`` or ``split`` (with ``routed_compact``, calls of at most ``COMPACT_TI_MAX_TOKENS`` tokens) gather
            the codes and the scales of each slot tile with two tensor-indirection copies (ScalarE / VectorE) in place
            of the 14 GpSimd ``nc_n_gather`` of the D512 tiles; the slot tables then build the copies' index in place of
            the per-partition gather columns (``_compact_tables`` ``snake``). The same bytes, so the values do not
            change.
        in_proj_lanes_view (bool): False (default) keeps the program. True (MXFP8 in-projection): the quantize of
            ``x_hT`` reads its lanes in place (``mx_quant.quantize_hT_mx`` ``lanes_view``, as the KDA input projection's
            ``kda_in_proj_lanes_view``), with no VectorE lane copies, when the rows are even (no zero row). The same
            values in the same order, so the result does not change.
        tail_before_guard (bool): False (default) keeps the program. True (``routed_compact`` ``hot_last``, 256-token
            calls): the tail of each compacted call (the token-order gather, the column-group transposes and per-tile
            routed stores) runs before the later slot tile's guard, and the guard's branch runs it again after the later
            tile (``_compact_routed_partial`` ``tail_first``). Faster when no rank takes the later tile, slower on a
            rank that takes it. The same rows on each path, so the values do not change.
        tail_store_halves (bool): False (default) keeps the program. True (``routed_compact`` ``hot_last``, 256-token
            calls): the tail stores each token tile's routed partial right after its transposes (also with the tail
            after the guard), as two DMAs, one per column-group copy (``_compact_routed_partial`` ``store_halves``), so
            the first starts as soon as its copy ends. The same bytes to the same places, so the values do not change.
        compact_alt_tiles (bool | str): False (default) keeps the program. With ``routed_compact``, one of
            ``moe_routed.STATIC_ALT_TILE_PSUM_CONSUME``: the ``static_alt_tiles`` of the first slot tile's static
            expert call (``_compact_routed_partial``): its gate sigmoid / up tanh tiles ("ring_act"), and its
            activation evictions ("ring_act_eviction"), take two alternating SBUF buffers each that live across the
            loop, in place of a compiler-placed tile for each chunk (which SBUF pressure can stack on a few addresses,
            so each ScalarE write waits for VectorE's read of the chunk before). The call keeps ``psum_consume`` and the
            resident last-expert reuse. Only the tiles change, so the values do not change.
        first_add_zero (bool): False (default) keeps the program. True: each ``moe_tkg_decode`` static expert call of
            the layer starts its FP32 accumulator at expert 0 (``moe_tkg_decode`` ``first_add_zero``: no zero fill;
            expert 0 adds a scalar +0.0, the same FP32 add, so the same bits). The token-tiled and per-expert calls
            (``_tiles_routed_partial``) keep their fill. The verify takes it by form (``MOE_FIRST_ADD_FORM_SET``).
        last_expert_order (str): ``moe_routed.LAST_EXPERT_ORDERS``. "default" keeps the program. With
            ``cross_expert_order`` and ``transpose_block``, and only static expert calls of at most 128 tokens (no
            ``routed_compact`` or ``routed_per_expert``): the ``last_expert_order`` of each call. "chunked" issues the
            last expert's activation transposes chunk by chunk; "up_first" also runs up(E_L - 1) and those transposes
            before down(E_L - 2), and the ring (with a ``prefetch`` stream from this body's ``prefetch``) loads
            up(E_L - 1) before down(E_L - 2). The values do not change.
        compact_tables_scalar (bool): False (default) keeps the program. True (with ``routed_compact``): the slot
            tables of each compacted call run the copies that no next VectorE instruction reads on ScalarE and write
            each token offset in one INT32 add (``_compact_tables`` ``scalar_copies``), so VectorE, which paces the
            first expert matmul, runs fewer instructions. Exact integers and 0/1 flags: the values do not change.
        affinity_sbuf_copy (bool): False (default) keeps the program. True (with ``routed_compact``): slot tile 0's
            static expert call copies its gathered SBUF slot affinities into the loop's FP32 affinity tile with one
            VectorE copy (``moe_tkg_decode`` ``affinity_sbuf_copy``), in place of the SBUF -> SBUF DMA that the Sync
            engine issues in order after the slot-affinity gather, in front of the expert loop's weight refills. The
            same values: the result does not change.
        out_proj_priority (int): None (default) keeps the program (the MXFP8 out-projection loads at the critical trn3
            DMA QoS level 0). 0 to 3: the trn3 DMA QoS level of those loads (3 is the lowest); the BF16 out projection
            ignores it. The early loads stream beside the router-row gathers, and the gathers' packets share their DMA
            engines, while the loads have about the whole expert loop of slack. The bytes do not change, so the values
            do not change.
        tail_zero_early (bool): False (default) keeps the program. True (``routed_compact`` ``hot_last`` with
            ``ungather_skip``, the tail after the guard): the zero fills of the tail's token-row gather destinations
            run before the later slot tile's guard, so the first gather issues right after the guard's branch instead
            of waiting for its fill, which VectorE ran after the branch. The same zero bytes: the values do not change.
        slot_static_first (bool): False (default) keeps the program. True (``routed_compact`` ``hot_last``): the
            active-expert loop of a later slot tile runs its first active expert as a static slot before the loop
            (``moe_routed.active_experts_in_slots`` ``static_first``), and the loop the others, so the common tile
            of one active expert runs no loop trip or trip barrier. The same experts in the same order; a zero
            affinity adds exactly zero for finite values, as in the loop.
        out_proj_before_in_proj (bool): False (default) keeps the program. True (with ``out_proj_early``): the early
            out-projection loads trace before the shared in-projection matmuls, not after them. The SBUF allocator then
            sees the out-projection tiles live while the in-projection tiles are read, so it gives them other bytes,
            and the loads wait for no matmul on the in-projection bytes. The same loads of the same bytes.
        scale_fill_after_prefetch (bool): False (default) keeps the program. True (with ``latent_gather_mx``): the
            GpSimd zero fill of the gathered code scales (``_load_mx_codes``) follows the head's weight prefetch DMAs
            (the router, in-projection and early out-projection loads) in the schedule, so it does not hold them in
            the GpSimd queue while it waits for its own inputs. Schedule edges only: the values do not change.
        router_dtype (str): ``ROUTER_DTYPES_2SRV``. ``bf16`` (default) keeps the program. ``fp16`` takes the router
            logits from ``x16_hT`` ([128, B, 56] FP16, the norm copy from the FP32 values) times ``router_weight``
            ([128, 56, 28] FP16 HBM: element (p, j, e) is ``router_weight[28 t + e, 128 j + p]`` for TP rank t), with
            FP32 PSUM, in place of the router columns of ``moe_in_proj_dp``. The collectives do not change.
        in_proj_scale (nl.NkiTensor): None (default, the BF16 ``moe_in_proj_dp``), or the scales of an MXFP8
            in-projection (``MX_IN_PROJ_SHAPES``): ``moe_in_proj_dp`` then holds the MXFP8 latent down and shared gate /
            up columns (``MX_IN_PROJ_COLS``, no router columns) in the hT_quad layout of ``tkg_mx_quant``, and both
            projections run MXFP8 x MXFP8 on the rows quantized per token on VectorE. It needs the FP16 router. The
            values change.
        out_proj_scale (nl.NkiTensor): None (default, the BF16 ``moe_out_proj_dp``), or the scales of an MXFP8
            out-projection (``MX_OUT_PROJ_SHAPES``): ``moe_out_proj_dp`` then holds the padded-lane form, and each
            hidden tile is one MXFP8 x MXFP8 matmul over the 512 rows, with the activation quantized on VectorE. Not
            with ``routed_reduce`` ``rs_fold``. The values change.
        latent_gather_mx (bool): False (default) keeps the program. True ports the ``latent_gather_mx`` mechanism: TP
            rank t < 28 computes the 128 latent columns [128 t, 128 t + 128) (``MX_LATENT_COLS``; ``moe_in_proj_dp`` has
            the 540 columns of ``G1A_IN_PROJ_COLS``, or the MXFP8 ``MX_G1A_IN_PROJ_SHAPES`` with ``in_proj_scale``),
            quantizes them with quantize_mx and sends the 32 code words and the scale word in the router rows (89 words,
            61 with ``drop_selection``; ``router_row_cols``), with its 28 scores; ranks 28..31 send zeros. The receiver
            loads the codes and the E8M0 scales straight into the MX expert input and passes them to ``moe_tkg_decode``
            as ``quantized_latent`` (no latent expand, swizzle transpose or quantize). Its codes and scale are the bytes
            that ``moe_tkg_decode`` makes of the whole BF16 latent, so against the BF16 rows the expert input is byte
            equal. A token-tiled call (256 or 512 tokens) takes the codes and scales of all its tokens
            (``moe_tkg_decode_token_tiles`` ``quantized_latent``) and the scores of each 128-token tile; each tile holds
            whole replicas (128 % B == 0). Needs the ``dp_tp`` gather; not with
            ``rows_latent`` ``fp8`` or ``rows_load_split`` > 1, and not with ``head_row_chunks``, ``quantize_first``,
            ``weights_after_gather`` or an ``in_proj_sb``. With ``pad_rows`` it needs ``routed_compact`` or a
            ``ROUTED_TILE_FORM`` split: the scores of the padding rows become 0 in its score rows as in the BF16 rows
            (``_mx_score_rows``). ``sources`` must come from ``moe_2srv_collective_hbm`` with the same
            ``latent_gather_mx``. A ``ROUTED_TILE_FORM`` split loads the codes, scales and scores of each token tile
            the same way (``_routed_tiles_partial``).
        mx_load_order (str): with ``latent_gather_mx``, the issue order of the code and scale loads of an expert call
            (``MX_LOAD_ORDERS``): ``quadrant`` (default) keeps the program; ``tile`` loads each (tile, quadrant) with
            its own DMA, tile 0 first, so the first MX matmul waits only for tile 0. The values do not change.
        mx_scale_queue (str): with ``latent_gather_mx``, the DMA queue of the scale loads (``MX_SCALE_QUEUES``):
            ``act`` (default) keeps the program; ``sp`` puts them on the SP queue of the code loads. The values do not
            change.
        mx_reload_order (str): with ``latent_gather_mx`` and ``pe_keep_warm``, the trace order of the code, scale and
            score loads of each keep-warm expert call (``MX_RELOAD_ORDERS``): ``default`` keeps the program, which adds
            the router bias before it issues the code and scale DMAs; ``codes_ahead`` issues those DMAs before the
            router-bias add, so they do not follow it in the trace (the score DMAs still issue first and the first
            code DMA still follows the last score load); ``codes_ahead_scores_act`` also puts the score DMAs on the
            Activation queue. The loaded bytes and the added bias do not change.
        mx_scores_align (bool): with ``latent_gather_mx`` and ``drop_selection``, pad the score run of each token in
            the router rows from 28 to ``_MX_SCORE_SLOT`` words (zero words, never read as scores), so the score load
            of each expert call moves 128-B elements. The rows grow from 61 to 65 words. ``sources`` must come from
            ``moe_2srv_collective_hbm`` with the same ``mx_scores_align``. False (default) keeps the program; the
            values do not change.
        mx_rows_stream (bool): with ``latent_gather_mx``, ``drop_selection`` and ``routed_compact`` (every expert call
            of more than 128 tokens), the router rows are token-major, ``_MX_STREAM_COLS`` words a token and rank
            (``_mx_store_rows``). Each 128-token tile of the compaction loads its rows in one DMA of 128-B elements (the
            two halves of each row, ``_load_mx_stream_rows``), and bit-accurate Tensor Engine transposes move the codes
            and the scales into the MX layout (``_mx_stream_to_mx``); the scores are read in place. ``sources`` must
            come from ``moe_2srv_collective_hbm`` with the same ``mx_rows_stream``. False (default) keeps the program;
            the values do not change.
        pe_keep_warm (bool | int | dict | None): with ``latent_gather_mx``, run keep-warm MX matmuls in each static
            expert call (at most 128 tokens) (``_pe_keep_warm_matmuls``): True runs ``PE_KEEP_WARM_MATMULS`` tied to
            the score DMAs, an int 1 to ``PE_KEEP_WARM_MAX`` that many, and a dict ``{"matmuls": k, "tile0": True}`` k
            on the score DMAs and one more on the last tile-0 code / scale DMA (``mx_load_order`` ``tile``). The
            score DMAs of the call then issue before its code and scale DMAs, and the score-tied matmuls precede its
            first expert MX matmul. The tile-0 matmul has no order edge to the expert matmuls: in the served C = 8
            program it runs inside the expert stream, after the first expert MX matmuls. Their
            result reaches the affinities as exact zero bits. False, None or 0 (default False) keeps the program; the
            values do not change. A token-tiled call refuses it. ``{"kda": form, "mla": form}`` gives each attention
            kind its own form (each one of the values above; a missing kind is off): the layer's ``keep_warm_lead``
            picks it.
        keep_warm_lead (str | None): with a per-kind ``pe_keep_warm``, the kind of the layer's attention (``"kda"`` or
            ``"mla"``; ``decode_block.decode_block_body`` gives it); None (default) otherwise (a lead with another form
            fails).
        routed_dynamic (bool): with a ``ROUTED_TILE_FORM`` split, run only the active local experts, each on a
            compact tile of its routed tokens (``tkg_moe_routed_dynamic``) in place of the static loop over
            every expert and token tile. False (default) keeps the program.
        routed_dynamic_pack (int | None): with ``routed_dynamic``, the experts of one packed item; None (default)
            takes ``default_pack`` of the split's I-slice (1 for 768: one expert an item; 16 for 96 at E_L 112).
        routed_dynamic_csr (str | None): with ``routed_dynamic``, the CSR gathers of a packed layer
            (``tkg_moe_routed_dynamic.CSR_GATHER_FORMS``): None (default) keeps "staged" (1 DGE table slot a
            layer); "rows" adds no slot, for at most 128 gathered tokens (``csr_gather_of_layers`` picks it).

    Returns:
        out_hT (nl.NkiTensor): [128, B, 56] BF16 SBUF, this replica's MoE output in the token-major hT layout.

    Notes:
        - Collectives of a layer, each once: all-gather (DP, router rows; spans the servers) and all-gather (TP32), or
          one world all-gather (``gather`` ``world``); all-reduce (world, routed latent; spans the servers), or with
          ``routed_reduce`` ``rs_ag`` reduce-scatter (world) and all-gather (TP32); all-reduce (TP32, output). Each
          runs on the HBM tensors of its site.
        - Up to the routed partials, the values are those of ``tkg_moe_dp`` for each token. The routed latent adds
          the tp intermediate parts of each expert as separate BF16 partials, so it has another rounding than the
          whole-expert sum of one rank.

    Pseudocode:
        r, latent_t, shared_g, shared_u = x @ moe_in_proj_dp                        # [B, 524]
        rows = all_gather_dp([bf16(latent_t), sigmoid(r), sigmoid(r) + bias])      # [T, 112]
        gathered = all_gather_tp(rows)                                              # [T, 32 * 112]
        for each call of at most expert_call_tokens tokens (128-token tiles):
            affinity = top-16 affinities of the E_L experts of group g % G
            routed_src[call] = expert loop(latent[call], affinity)                  # [128, call * 28]
        routed = all_reduce_world(routed_src)                                       # [128, T * 28]
        normed = rms_norm(routed[rows of replica r], w)[:, 112 t : 112 t + 112]
        # rs_ag: own = reduce_scatter_world(routed_rs_src)  # [B, 112] = routed[rows of r][:, 112 t : 112 t + 112]
        #        normed = own * rstd(all_gather_tp(own)) * w   (stats_gather: all_gather_tp(sum(own^2)))
        # rs_fold: own = reduce_scatter_world(routed_rs_src); y_l, y_s, ss = all_reduce_tp_fp32(
        #        [bf16(own * w) @ W_l | situ(shared) @ W_s | sum(own^2)]); out = bf16(rsqrt(ss / 3584 + eps) * y_l + y_s)
        out = all_reduce_tp(bf16([normed | situ(shared_g, shared_u)] @ moe_out_proj_dp))
        # column: shared = all_gather_tp(situ(shared_g, shared_u))                  # [B, 6144]
        #         out = all_gather_tp(bf16([rms_norm(routed[rows of r], w) | shared] @ W_t))   # W_t [9728, 224]
        #         (rs_ag: routed[rows of r] = all_gather_tp(reduce_scatter_world(routed_rs_src)))
    """
    _validate_inputs(
        x_hT,
        moe_in_proj_dp,
        moe_out_proj_dp,
        moe_vectors_dp,
        routed_store_form,
        drop_selection,
        out_proj_form,
        in_proj_scale,
        out_proj_scale,
        latent_gather_mx,
    )
    _check_router(x_hT, router_dtype, router_weight, x16_hT)
    mx = in_proj_scale is not None
    kernel_assert(not mx or router_dtype == "fp16", "the MXFP8 in-projection (in_proj_scale) needs the FP16 router")
    _check_routed_reduce(routed_reduce, stats_gather)
    _check_out_proj_form(out_proj_form, stats_gather)
    glue_hooks = glue_hooks or {}
    kernel_assert(
        set(glue_hooks) <= {*GLUE_HOOKS, TAIL_GLUE_HOOK, IN_PROJ_GLUE_HOOK},
        f"glue_hooks takes the keys {(*GLUE_HOOKS, TAIL_GLUE_HOOK, IN_PROJ_GLUE_HOOK)}, got {sorted(glue_hooks)}",
    )
    kernel_assert(
        not (set(glue_hooks) - {TAIL_GLUE_HOOK, IN_PROJ_GLUE_HOOK})
        or not (routed_reduce == "rs_fold" or (routed_reduce == "ar" and norm_tail == "token_major")),
        "glue_hooks: the rs_fold and the token_major norm tails call no hook",
    )
    kernel_assert(rows_latent in ROWS_LATENTS, f"rows_latent must be one of {ROWS_LATENTS}, got {rows_latent!r}")
    kernel_assert(
        rows_align == (rows_latent == "fp8a"),
        f"glue rows_align is the rows_latent fp8a router rows: got rows_align {rows_align}, rows_latent {rows_latent!r}",
    )
    kernel_assert(
        rows_latent != "fp8a" or (gather == "dp_tp" and drop_selection),
        f"rows_latent fp8a needs the dp_tp gather and drop_selection, got {gather!r}, drop_selection {drop_selection}",
    )
    kernel_assert(
        rows_load_split == 1 or rows_latent != "fp8a",
        f"rows_latent fp8a does not split the router-rows load, got rows_load_split {rows_load_split}",
    )
    kernel_assert(
        rows_load_split == 1 or (rows_load_split in ROWS_LOAD_SPLITS and gather == "dp_tp"),
        f"rows_load_split is 1 or one of {ROWS_LOAD_SPLITS} with the dp_tp gather, got {rows_load_split}",
    )
    kernel_assert(
        not routed_own_rows or routed_reduce == "ar",
        f"routed_own_rows loads the replica rows of the routed all-reduce (routed_reduce 'ar'), got {routed_reduce!r}",
    )
    check_glue_keys(norm_tail, rows_queue, routed_reduce, x_hT.shape[1])
    check_routed_layout(routed_layout, routed_reduce, x_hT.shape[1], out_proj_form)
    kernel_assert(
        routed_layout == "natural" or not (tail_before_guard or tail_store_halves),
        "routed_layout 'qsplit' is not taken with tail_before_guard or tail_store_halves: the compaction tail's per-tile, "
        "half and redo stores have no q-split twin",
    )
    kernel_assert(
        not (routed_own_rows and norm_tail == "token_major"),
        "routed_own_rows and norm_tail 'token_major' each take the replica rows of the routed all-reduce: set one",
    )
    kernel_assert(
        not (out_proj_early and routed_reduce == "rs_fold"),
        "out_proj_early loads the out-projection of the routed tail: rs_fold folds it into its own reduce",
    )
    kernel_assert(
        not accum_memset_after_rows or (gather == "dp_tp" and not routed_compact),
        "accum_memset_after_rows orders the static calls' accumulator fill after the dp_tp router-row gather: it needs "
        f"the dp_tp gather and no routed_compact, got gather {gather!r}, routed_compact {routed_compact!r}",
    )
    kernel_assert(
        out_proj_after_rows in OUT_PROJ_AFTER_ROWS_FORMS,
        f"out_proj_after_rows must be one of {OUT_PROJ_AFTER_ROWS_FORMS}, got {out_proj_after_rows!r}",
    )
    kernel_assert(
        out_proj_priority in (None, 0, 1, 2, 3),
        f"out_proj_priority is a trn3 DMA QoS level (0 to 3) of the MXFP8 out-projection loads, got {out_proj_priority!r}",
    )
    kernel_assert(
        not out_proj_after_rows
        or (out_proj_early and out_proj_form == "reduced" and out_proj_scale is not None and gather == "dp_tp"),
        "out_proj_after_rows orders the early MXFP8 out-projection loads after a dp_tp router-row gather: it needs "
        f"out_proj_early, the reduced MXFP8 out projection and the dp_tp gather, got "
        f"out_proj_early {out_proj_early}, out_proj_form {out_proj_form!r}, out_proj_scale "
        f"{'set' if out_proj_scale is not None else None}, gather {gather!r}",
    )
    kernel_assert(
        out_proj_scale is None or routed_reduce != "rs_fold",
        "the MXFP8 out-projection (out_proj_scale) is not with rs_fold, which folds the BF16 one into its reduce",
    )
    if latent_gather_mx:
        kernel_assert(gather == "dp_tp", f"latent_gather_mx needs the dp_tp gather, got {gather!r}")
        kernel_assert(rows_latent == "bf16", "latent_gather_mx sends MXFP8 codes in the router rows, not fp8 latent")
        kernel_assert(
            rows_load_split == 1,
            f"latent_gather_mx does not split the router-rows load, got rows_load_split {rows_load_split}",
        )
        kernel_assert(
            not (head_row_chunks or quantize_first),
            "latent_gather_mx sends the MXFP8 latent codes in the router rows and runs no latent quantize: not with "
            "head_row_chunks or quantize_first, which quantize the BF16 latent of the rows",
        )
        kernel_assert(
            not (weights_after_gather or weights_after_rows),
            "latent_gather_mx gathers its own router rows: not with weights_after_gather or weights_after_rows, which "
            "order after the BF16 row gathers",
        )
        kernel_assert(
            pad_rows is None or routed_compact or split.get("layout_form") == ROUTED_TILE_FORM,
            "latent_gather_mx zeroes the scores of the padding rows (pad_rows) in its own score rows only with "
            "routed_compact (the served verify form) or the routed-tile loop of a sub-512-tile split",
        )
        kernel_assert(
            in_proj_sb is None,
            "latent_gather_mx loads its own in-projection of the 540-column block layout: not with an in_proj_sb",
        )
    kernel_assert(
        mx_load_order in MX_LOAD_ORDERS, f"mx_load_order must be one of {MX_LOAD_ORDERS}, got {mx_load_order!r}"
    )
    kernel_assert(
        mx_scale_queue in MX_SCALE_QUEUES, f"mx_scale_queue must be one of {MX_SCALE_QUEUES}, got {mx_scale_queue!r}"
    )
    kernel_assert(
        latent_gather_mx or (mx_load_order == "quadrant" and mx_scale_queue == "act"),
        "mx_load_order and mx_scale_queue order the latent_gather_mx loads: they need latent_gather_mx",
    )
    kernel_assert(
        not mx_scores_align or (latent_gather_mx and drop_selection),
        f"mx_scores_align pads the score run of the drop_selection latent_gather_mx rows: it needs both, got "
        f"latent_gather_mx={latent_gather_mx}, drop_selection={drop_selection}",
    )
    if _keep_warm_by_lead(pe_keep_warm):
        kernel_assert(
            keep_warm_lead in PE_KEEP_WARM_LEADS,
            f"a per-kind pe_keep_warm {pe_keep_warm} needs keep_warm_lead (one of {PE_KEEP_WARM_LEADS}), got "
            f"{keep_warm_lead!r}",
        )
        pe_keep_warm = pe_keep_warm.get(keep_warm_lead, False)
    else:
        kernel_assert(
            keep_warm_lead is None,
            f"keep_warm_lead goes with a per-kind pe_keep_warm only, got {keep_warm_lead!r} with {pe_keep_warm!r}",
        )
    keep_warm = _keep_warm_form(pe_keep_warm)
    # The layer kind of a per-kind form, for the refusals below (empty for the other forms).
    kind_note = f" (the form for the {keep_warm_lead} layers)" if keep_warm_lead else ""
    kernel_assert(
        not keep_warm[1] or mx_load_order == "tile",
        f"the pe_keep_warm tile0 matmul{kind_note} follows the last tile-0 DMA of the tile load order: it needs "
        f"mx_load_order tile, got {mx_load_order!r}",
    )
    kernel_assert(
        not keep_warm[0] or latent_gather_mx,
        f"pe_keep_warm{kind_note} ties its matmuls to the latent_gather_mx score loads of an expert call: it needs "
        f"latent_gather_mx, got {latent_gather_mx}",
    )
    kernel_assert(
        mx_reload_order in MX_RELOAD_ORDERS,
        f"mx_reload_order must be one of {MX_RELOAD_ORDERS}, got {mx_reload_order!r}",
    )
    kernel_assert(
        mx_reload_order == "default" or (latent_gather_mx and keep_warm[0]),
        f"mx_reload_order reorders the keep-warm code, scale and score loads of a latent_gather_mx expert call"
        f"{kind_note}: it needs latent_gather_mx and pe_keep_warm, got latent_gather_mx={latent_gather_mx}, "
        f"pe_keep_warm {keep_warm}",
    )
    kernel_assert(
        not mx_rows_stream
        or (
            latent_gather_mx
            and drop_selection
            and not mx_scores_align
            and routed_compact
            and mx_load_order == "quadrant"
            and mx_scale_queue == "act"
        ),
        "mx_rows_stream loads the token-major latent_gather_mx rows of the compaction: it needs latent_gather_mx, "
        "drop_selection, routed_compact and expert calls of more than 128 tokens, and not mx_scores_align, "
        "mx_load_order or mx_scale_queue",
    )

    # weights_before_compact: the early weight loads that the compaction's first instructions follow.
    early_weight_loads = [] if weights_before_compact else None

    def load_out_proj(issued=None, after=None, after_done=False):
        if out_proj_scale is None:
            return _load_out_proj(moe_out_proj_dp, name)
        done = {"after_done": True} if after_done else {}
        level = {"priority": out_proj_priority} if out_proj_priority is not None else {}
        return _load_mx_out_proj(moe_out_proj_dp, out_proj_scale, name, issued=issued, after=after, **done, **level)

    expert_shapes = moe_2srv_expert_shapes(split["world"], split["routed_tp"], **split_mx_tile(split))
    expert_weights = (
        expert_gate_up_weights,
        expert_gate_up_weights_scale,
        expert_down_weights,
        expert_down_weights_scale,
    )
    for key, value in zip(expert_shapes, expert_weights, strict=True):
        kernel_assert(tuple(value.shape) == expert_shapes[key], f"{key}: {value.shape} is not {expert_shapes[key]}")
    rows = x_hT.shape[1]
    dp = split["dp"]
    tokens = dp * rows
    chunks = token_chunks(tokens)
    # A split of sub-512 MX tiles runs the routed-expert tile (ROUTED_TILE_FORM); every other split keeps its program.
    routed_tiles = split.get("layout_form") == ROUTED_TILE_FORM
    kernel_assert(not routed_dynamic or routed_tiles, "routed_dynamic takes a split of sub-512 MX tiles")
    kernel_assert(routed_dynamic or routed_dynamic_pack is None, "routed_dynamic_pack goes with routed_dynamic")
    kernel_assert(
        routed_dynamic_csr is None or (routed_dynamic and routed_dynamic_csr in CSR_GATHER_FORMS),
        f"routed_dynamic_csr goes with routed_dynamic and is one of {CSR_GATHER_FORMS}, got {routed_dynamic_csr!r}",
    )
    if split["local_intermediate"] == EXPERT_INTERMEDIATE and rows > WIDE_ROWS and not half_weight_buffers:
        half_weight_buffers = wide_half_weight_buffers
    if expert_call_tokens == EXPERT_CALL_AUTO:
        expert_call_tokens = (
            EXPERT_CALL_TOKENS
            if routed_tiles
            else auto_expert_call_tokens(
                tokens, split["routed_tp"], routed_store_form, weight_buffers, half_weight_buffers
            )
        )
    kernel_assert(
        expert_call_tokens in EXPERT_CALL_FORMS,
        f"expert_call_tokens must be {EXPERT_CALL_AUTO!r} or one of {EXPERT_CALL_FORMS}, got {expert_call_tokens}",
    )
    # The stream rows feed the compaction of the token-tiled calls only: checked before any instruction.
    kernel_assert(
        not mx_rows_stream or all(count > _PMAX for _, count in expert_calls(tokens, expert_call_tokens)),
        f"mx_rows_stream rows feed the compaction of expert calls of more than {_PMAX} tokens only, got the calls "
        f"{expert_calls(tokens, expert_call_tokens)}",
    )
    check_rows_pairs(
        rows_pairs,
        gather,
        drop_selection,
        rows_latent,
        rows_load_split,
        latent_gather_mx,
        tokens,
        expert_call_tokens,
        routed_compact,
    )
    kernel_assert(
        static_alt_tiles in STATIC_ALT_TILE_FORMS,
        f"static_alt_tiles must be one of {STATIC_ALT_TILE_FORMS}, got {static_alt_tiles!r}",
    )
    kernel_assert(
        not latent_gather_mx or expert_call_tokens == EXPERT_CALL_TOKENS or _PMAX % rows == 0,
        f"latent_gather_mx loads whole replicas in each {_PMAX}-token tile of a token-tiled expert call: B = {rows} "
        f"rows must divide {_PMAX}, got expert_call_tokens {expert_call_tokens}",
    )
    kernel_assert(
        expert_call_tokens == EXPERT_CALL_TOKENS or routed_store_form == "pe",
        f"expert_call_tokens {expert_call_tokens} needs routed_store_form 'pe', got {routed_store_form!r}",
    )
    kernel_assert(
        expert_call_tokens == EXPERT_CALL_TOKENS or tokens % _PMAX == 0,
        f"expert_call_tokens {expert_call_tokens} needs dp x rows a multiple of {_PMAX}, got {tokens}",
    )
    kernel_assert(
        transpose_lookahead is None or expert_call_tokens == EXPERT_CALL_TOKENS,
        f"transpose_lookahead needs calls of at most {EXPERT_CALL_TOKENS} tokens (moe_tkg_decode), got "
        f"expert_call_tokens {expert_call_tokens}",
    )
    # A token-tiled call (more than 128 tokens) takes the cross-expert order of moe_tkg_decode_token_tiles: 3 whole
    # weight buffers and no tile blocks of either order.
    kernel_assert(
        not cross_expert_order
        or expert_call_tokens == EXPERT_CALL_TOKENS
        or routed_compact
        or (
            weight_buffers == _CROSS_TOKEN_TILE_WEIGHT_BUFFERS
            and not half_weight_buffers
            and not cross_expert_transpose_blocks
            and not token_tiles_transpose_blocks
        ),
        f"cross_expert_order needs calls of at most {EXPERT_CALL_TOKENS} tokens (moe_tkg_decode), routed_compact "
        f"(its first slot tile is a static call of {EXPERT_CALL_TOKENS}), or on token-tiled calls "
        f"{_CROSS_TOKEN_TILE_WEIGHT_BUFFERS} whole weight buffers and neither cross_expert_transpose_blocks nor "
        f"token_tiles_transpose_blocks, got expert_call_tokens {expert_call_tokens}, weight_buffers {weight_buffers}, "
        f"half_weight_buffers {half_weight_buffers}, cross_expert_transpose_blocks {cross_expert_transpose_blocks}, "
        f"token_tiles_transpose_blocks {token_tiles_transpose_blocks}",
    )
    kernel_assert(
        not quantize_first or expert_call_tokens == EXPERT_CALL_TOKENS,
        f"quantize_first needs calls of at most {EXPERT_CALL_TOKENS} tokens (the static moe_tkg_decode calls), got "
        f"expert_call_tokens {expert_call_tokens}",
    )
    kernel_assert(
        not cross_expert_transpose_blocks or cross_expert_order,
        "cross_expert_transpose_blocks orders the activation transposes of the cross-expert order: it needs "
        "cross_expert_order",
    )
    kernel_assert(
        not head_row_chunks or (quantize_first and gather == "dp_tp" and expert_call_tokens == EXPERT_CALL_TOKENS),
        f"head_row_chunks splits the rows load and the latent quantize of the static expert calls: it needs "
        f"quantize_first, gather 'dp_tp' and calls of at most {EXPERT_CALL_TOKENS} tokens, got quantize_first "
        f"{quantize_first}, gather {gather!r}, expert_call_tokens {expert_call_tokens}",
    )
    kernel_assert(
        not head_row_chunks or rows_latent == "bf16",
        f"head_row_chunks copies the BF16 latent words of the router rows: not with rows_latent {rows_latent!r}",
    )
    kernel_assert(
        rows_load_split == 1 or not head_row_chunks,
        "rows_load_split and head_row_chunks both split the router-rows load: pass one of them",
    )
    kernel_assert(
        not token_tiles_transpose_blocks or any(count > _PMAX for _, count in expert_calls(tokens, expert_call_tokens)),
        f"token_tiles_transpose_blocks orders the token-tiled expert calls: it needs a call of more than {_PMAX} "
        f"tokens, got {tokens} tokens in calls of {expert_call_tokens}",
    )
    if routed_per_expert:
        per_expert_options = _per_expert_call_options(routed_per_expert, cross_expert_order)
        kernel_assert(
            per_expert_options["combine_cols"] % _MX_TILE == 0
            and per_expert_options["combine_cols"] > 0
            and per_expert_options["slot_gather"] in PER_EXPERT_SLOT_GATHERS
            and per_expert_options["weight_buffers"] in (2, 3)
            and (per_expert_options["weight_buffers"] == 3 or not cross_expert_order),
            f"routed_per_expert is False, True, the combine columns (a multiple of {_MX_TILE}) or a dict of "
            f"combine_cols / slot_gather (one of {PER_EXPERT_SLOT_GATHERS}) / weight_buffers (2 or 3; 3 with "
            f"cross_expert_order), got {routed_per_expert!r}",
        )
        tiled_calls = [count for _, count in expert_calls(tokens, expert_call_tokens) if count > _PMAX]
        kernel_assert(
            tiled_calls and all(count == 2 * _PMAX for count in tiled_calls),
            f"routed_per_expert runs token-tiled expert calls of {2 * _PMAX} tokens: got {tokens} tokens in calls of "
            f"{expert_call_tokens}",
        )
        kernel_assert(
            not half_weight_buffers and expert_queues == 1,
            f"routed_per_expert takes whole weight buffers and one expert queue, got half_weight_buffers "
            f"{half_weight_buffers}, expert_queues {expert_queues}",
        )
        kernel_assert(
            not (
                routed_compact
                or latent_gather_mx
                or static_alt_tiles
                or token_tiles_transpose_blocks
                or cross_expert_transpose_blocks
                or rows_pairs
                or prefetch is not None
            ),
            "routed_per_expert replaces the token-tiled calls: not with routed_compact, latent_gather_mx, "
            "static_alt_tiles, token_tiles_transpose_blocks, cross_expert_transpose_blocks, rows_pairs or prefetch",
        )
    name = f"k3moe2s_{tag}"
    hbm = _hbm_views(
        sources,
        rows,
        split,
        gather,
        drop_selection,
        routed_reduce,
        stats_gather,
        out_proj_form,
        rows_latent,
        latent_gather_mx,
        mx_scores_align,
        mx_rows_stream,
    )

    # The in-projection latent columns: the 128-column block of latent_gather_mx, else the 112-column DP shard.
    latent_width = MX_LATENT_COLS if latent_gather_mx else DP_LATENT
    in_proj_cols = G1A_IN_PROJ_COLS if latent_gather_mx else DP_IN_PROJ_COLS
    latent_end = DP_ROUTER_COLS[1] + latent_width
    shared_gate_start = latent_end
    shared_up_end = shared_gate_start + 2 * DP_SHARED

    fp16 = router_dtype == "fp16"
    # IN_PROJ_GLUE_HOOK: the head's weight loads (the FP16 router, the MXFP8 in-projection) that the hook follows.
    in_proj_loads = []
    if fp16:
        router_w_sb = nl.ndarray((_PMAX, _H_TILES, DP_EXPERTS), dtype=nl.float16, buffer=nl.sbuf)
        in_proj_loads.append(
            nisa.dma_copy(
                dst=router_w_sb, src=router_weight, priority=_CRITICAL_DMA_PRIORITY, name=f"{name}_router_w16"
            )
        )
    if mx:
        in_proj_mx = _load_mx_in_proj(moe_in_proj_dp, in_proj_scale, name, marks=in_proj_loads)
    elif in_proj_sb is None:
        in_proj_sb = nl.ndarray((_PMAX, _H_TILES, in_proj_cols), dtype=nl.bfloat16, buffer=nl.sbuf)
        _load_hT_rows(in_proj_sb, moe_in_proj_dp, _IN_PROJ_DMA_CHUNKS, f"{name}_in_proj")
    else:
        kernel_assert(
            tuple(in_proj_sb.shape) == (_PMAX, _H_TILES, DP_IN_PROJ_COLS) and in_proj_sb.dtype == nl.bfloat16,
            f"in_proj_sb must be BF16 [{_PMAX}, {_H_TILES}, {DP_IN_PROJ_COLS}], got {in_proj_sb.dtype} {in_proj_sb.shape}",
        )
    if not drop_selection:
        bias_sb = nl.ndarray((rows, DP_EXPERTS), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=bias_sb,
            src=moe_vectors_dp.ap(pattern=[[0, rows], [1, DP_EXPERTS]], offset=DP_BIAS_COLS[0]),
            priority=_CRITICAL_DMA_PRIORITY,
            name=f"{name}_bias",
        )
    if routed_reduce == "ar":
        row_select, _ = rank_selectors(rank_info, f"{name}_rank", dp_size=dp)
    replica_index = _replica_index(rank_info, name) if routed_own_rows else None
    cached = getattr(rank_info, _selectors_key(chunks[0][1], split), None)
    if cached is not None:
        shard_select, group_select = cached
    elif selectors_once:
        shard_select, group_select = moe_2srv_glue_selectors(rank_info, chunks[0][1], split, name)
    else:
        shard_select, group_select = _selectors(rank_info, chunks[0][1], split, name)
    shard_offset = _shard_offset(rank_info, dp, name, routed_layout) if norm_tail == "token_major" else None

    # Router and latent down first (the critical path), then the shared gate and up, for all rows.
    x_rows = x_hT.permute((0, 2, 1))
    if mx:
        # The stationary rows of an MX matmul are even: an odd B gets one zero row.
        padded = rows + rows % 2
        x_mx = quantize_hT_mx(x_hT, f"{name}_x", padded, **({"lanes_view": True} if in_proj_lanes_view else {}))
        latent_psum = _psum_rows(padded, latent_width)
        router_last = _mx_in_proj(latent_psum, x_mx, in_proj_mx, 0, latent_width, padded)
        latent_src = latent_psum[0:rows, 0:latent_width]
    else:
        router_psum = _psum_rows(rows, latent_end)
        for h_tile_idx in range(_H_TILES):
            router_last = nisa.nc_matmul(
                dst=router_psum,
                stationary=x_rows[:, h_tile_idx, :],
                moving=in_proj_sb[:, h_tile_idx, 0:latent_end],
            )
        logits = router_psum[:, DP_ROUTER_COLS[0] : DP_ROUTER_COLS[1]]
        latent_src = router_psum[:, DP_ROUTER_COLS[1] : latent_end]
    if fp16:
        x16_rows = x16_hT.permute((0, 2, 1))
        logits = _psum_rows(rows, DP_EXPERTS)
        for h_tile_idx in range(_H_TILES):
            router_last = nisa.nc_matmul(
                dst=logits, stationary=x16_rows[:, h_tile_idx, :], moving=router_w_sb[:, h_tile_idx, :]
            )
    row_gathers = []
    # IN_PROJ_GLUE_HOOK only: the latent_gather_mx gathers. With out_proj_after_rows or accum_memset_after_rows they go
    # to row_gathers (their edges read it); with both keys off row_gathers keeps its program for the other uses.
    hook_gathers = []
    rows_dma = ROWS_DMAS[rows_queue]
    if latent_gather_mx:
        # Quantize the rank's 128-column latent block and send its MXFP8 codes and scales with the token scores.
        score_sel = _mx_score_rows(
            logits, None if drop_selection else bias_sb, pad_rows, name, _MX_SCORE_SLOT if mx_scores_align else None
        )
        codes, scales = _mx_quantize_latent(latent_src, rows, name)
        rows_store = _mx_gather_router_rows(
            codes,
            scales,
            score_sel,
            hbm,
            dp_group,
            tp_group,
            name,
            dma=rows_dma,
            stream=mx_rows_stream,
            # The in-projection glue hook gets the gathers from either list.
            gathers_out=row_gathers if out_proj_after_rows or accum_memset_after_rows else hook_gathers,
        )
    else:
        gather_src = router_rows_src(
            latent_src, logits, None if drop_selection else bias_sb, rows, drop_selection, rows_latent, pad_rows, name
        )
        rows_store = _gather_router_rows(
            gather_src, hbm, gather, dp_group, tp_group, world_group, name, gathers_out=row_gathers, dma=rows_dma
        )
    if glue_hooks.get(IN_PROJ_GLUE_HOOK) is not None:
        # IN_PROJ_GLUE_HOOK: after the head's weight loads and the triggers of the router-row gathers (the dispatch).
        glue_hooks[IN_PROJ_GLUE_HOOK]((*in_proj_loads, rows_store, *row_gathers, *hook_gathers))
    router_bias = None
    if drop_selection:
        # Behind the rows store on its queue: the bias streams in the flight of the row gathers, not beside the
        # in-projection weight loads that the router waits for.
        router_bias = _load_router_bias(moe_vectors_dp, min(tokens, _PMAX), name, after=rows_store, dma=rows_dma)

    # The column out projection loads its own column-layout weights: out_proj_early takes the reduced one only.
    # out_proj_after_rows: the early loads follow the DP (rows_gather) or TP32 (shard_gather[_done]) router-row gather.
    out_proj_after = row_gathers[0 if out_proj_after_rows == "rows_gather" else -1] if out_proj_after_rows else None
    after_done = out_proj_after_rows == "shard_gather_done"
    early_out_proj = out_proj_early and out_proj_form == "reduced"
    # scale_fill_after_prefetch: the early out-projection DMAs join the head's prefetch that the code-scale fill follows.
    out_proj_loads = [] if scale_fill_after_prefetch else early_weight_loads

    def load_early_out_proj():
        loaded = load_out_proj(out_proj_loads, after=out_proj_after, after_done=after_done)
        if out_proj_loads is not early_weight_loads and early_weight_loads is not None:
            early_weight_loads.extend(out_proj_loads)
        return loaded

    # out_proj_before_in_proj: the early out-projection tiles are live while the in-projection matmuls read their tiles.
    out_proj_sb = load_early_out_proj() if early_out_proj and out_proj_before_in_proj else None
    if mx:
        shared_padded = _psum_rows(padded, 2 * DP_SHARED)
        after = router_last if router_first else None
        _mx_in_proj(shared_padded, x_mx, in_proj_mx, latent_width, 2 * DP_SHARED, padded, after=after)
        shared_psum = shared_padded[0:rows, 0 : 2 * DP_SHARED]
    else:
        shared_psum = _psum_rows(rows, 2 * DP_SHARED)
        for h_tile_idx in range(_H_TILES):
            shared = nisa.nc_matmul(
                dst=shared_psum,
                stationary=x_rows[:, h_tile_idx, :],
                moving=in_proj_sb[:, h_tile_idx, shared_gate_start:shared_up_end],
            )
            if router_first and h_tile_idx == 0:
                # The shared matmuls accumulate in order, so the first one after the router keeps all of them after
                # it.
                _instruction(shared).depends_on(_instruction(router_last), kind=nisa.dependency.schedule)
    # Out-projection input, token-major [B, 304]: the normed latent (after the chunks), the shared activation.
    act_tok = nl.ndarray((rows, DP_OUT_PROJ_ROWS), dtype=nl.bfloat16, buffer=nl.sbuf)
    shared_bf16 = nl.ndarray((rows, 2 * DP_SHARED), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=shared_bf16, src=shared_psum)
    _situ(
        shared_bf16[:, 0:DP_SHARED],
        shared_bf16[:, DP_SHARED : 2 * DP_SHARED],
        out=act_tok[:, DP_SHARED_DOWN_ROWS[0] : DP_SHARED_DOWN_ROWS[1]],
    )
    if early_out_proj and not out_proj_before_in_proj:
        out_proj_sb = load_early_out_proj()
    scale_fill_after = (*in_proj_loads, *out_proj_loads) if scale_fill_after_prefetch else ()

    shared_store = None
    if out_proj_form == "column":
        kernel_assert(
            norm_tail != "token_major", "the column out projection takes the default norm tail, not token_major"
        )
        shared_store = store_shared_activation(hbm, act_tok, rows, name)
        norm_weight = load_column_norm_weight(moe_vectors_dp, drop_selection, name)
    else:
        # The token-major tail takes the [B, 112] weights of the rs_ag tail.
        norm_weight = load_norm_weight(
            moe_vectors_dp,
            rows,
            "rs_ag" if norm_tail == "token_major" else routed_reduce,
            name,
            issued=early_weight_loads,
        )
    loop_options = {
        "weight_buffers": weight_buffers,
        "scale_buffers": scale_buffers,
        "early_loads": early_loads,
        **({"half_weight_buffers": half_weight_buffers} if half_weight_buffers else {}),
        **({"transpose_lookahead": transpose_lookahead} if transpose_lookahead is not None else {}),
        **({"cross_expert_order": True} if cross_expert_order else {}),
        **({"cross_expert_transpose_blocks": True} if cross_expert_transpose_blocks else {}),
        **({"transpose_block": True} if transpose_block else {}),
        **({"expert_queues": expert_queues} if expert_queues != 1 else {}),
        **({"first_add_zero": True} if first_add_zero else {}),
    }
    calls = expert_calls(tokens, expert_call_tokens)
    kernel_assert(
        not accum_memset_after_rows or all(count <= _PMAX for _, count in calls),
        "accum_memset_after_rows orders the accumulator fill of the static moe_tkg_decode calls: it needs only calls "
        f"of at most {_PMAX} tokens (a token-tiled call keeps its own fill, so the key would do nothing there), "
        f"got the calls {calls}",
    )
    kernel_assert(
        routed_compact in ROUTED_COMPACT_FORMS,
        f"routed_compact must be one of {ROUTED_COMPACT_FORMS}, got {routed_compact!r}",
    )
    kernel_assert(
        not ungather_skip or routed_compact,
        f"ungather_skip skips the unrouted rows of the compaction's token-order gather: it needs routed_compact, got "
        f"routed_compact {routed_compact!r}",
    )
    kernel_assert(
        not prologue_vector_cuts or routed_compact,
        f"prologue_vector_cuts acts on the compaction's prologue: it needs routed_compact, got {routed_compact!r}",
    )
    kernel_assert(
        not compact_tables_scalar or routed_compact,
        f"compact_tables_scalar acts on the compaction's slot tables: it needs routed_compact, got {routed_compact!r}",
    )
    kernel_assert(
        not affinity_sbuf_copy or routed_compact,
        f"affinity_sbuf_copy acts on the compaction's SBUF slot affinities: it needs routed_compact, got {routed_compact!r}",
    )
    kernel_assert(
        not (tail_zero_early or slot_static_first) or routed_compact == "hot_last",
        f"tail_zero_early and slot_static_first act on the later slot tile of the hot_last compaction: they need "
        f"routed_compact hot_last, got {routed_compact!r}",
    )
    kernel_assert(
        not tail_zero_early or (ungather_skip and not tail_before_guard),
        "tail_zero_early moves the zero fills of the ungather_skip tail after the guard: it needs ungather_skip and no "
        "tail_before_guard",
    )
    kernel_assert(
        not up_product_split or routed_compact,
        f"up_product_split acts on the compaction's static loop: it needs routed_compact, got {routed_compact!r}",
    )
    kernel_assert(
        not weights_before_compact or (routed_compact and out_proj_early),
        f"weights_before_compact orders the early weight loads before the compaction: it needs routed_compact and "
        f"out_proj_early, got routed_compact {routed_compact!r}, out_proj_early {out_proj_early}",
    )
    kernel_assert(
        not weights_after_rows or gather == "dp_tp",
        f"weights_after_rows waits for the router-row loads of the dp_tp TP32 all-gather, got gather {gather!r}",
    )
    kernel_assert(
        routed_compact != "timing_empty_later" or TIMING_CONTROLS_ALLOWED,
        "routed_compact timing_empty_later is a timing control that drops the tokens past slot 128 of an overflow "
        "call: test only (set moe_decode_2srv.TIMING_CONTROLS_ALLOWED in the harness)",
    )
    kernel_assert(
        not routed_compact
        or (
            gather == "dp_tp"
            and not token_tiles_transpose_blocks
            and any(count > _PMAX for _, count in calls)
            and (not latent_gather_mx or rows <= _PMAX and _PMAX % rows == 0)
        ),
        f"routed_compact compacts the token-tiled expert calls: it needs the dp_tp gather, no "
        f"token_tiles_transpose_blocks, a call of more than {_PMAX} tokens, and with latent_gather_mx whole replicas "
        f"in each 128-token tile, got gather {gather!r}, latent_gather_mx {latent_gather_mx}, rows {rows}, "
        f"token_tiles_transpose_blocks {token_tiles_transpose_blocks}, calls {calls}",
    )
    kernel_assert(
        compact_slot_gather in COMPACT_SLOT_GATHERS
        and (
            compact_slot_gather == "gpsimd"
            or (routed_compact and all(count <= COMPACT_TI_MAX_TOKENS for _, count in calls))
        ),
        f"compact_slot_gather must be one of {COMPACT_SLOT_GATHERS}, and other than 'gpsimd' only with routed_compact "
        f"and calls of at most {COMPACT_TI_MAX_TOKENS} tokens, got {compact_slot_gather!r}, routed_compact "
        f"{routed_compact!r}, calls {calls}",
    )
    kernel_assert(
        not tail_store_halves or routed_compact == "hot_last",
        f"tail_store_halves splits the routed stores of the hot_last tail: it needs routed_compact hot_last, got "
        f"{routed_compact!r}",
    )
    kernel_assert(
        not compact_alt_tiles or (routed_compact and compact_alt_tiles in STATIC_ALT_TILE_PSUM_CONSUME),
        f"compact_alt_tiles takes the rings of the compaction's first slot tile: it needs routed_compact and one of "
        f"{STATIC_ALT_TILE_PSUM_CONSUME}, got {compact_alt_tiles!r} with routed_compact {routed_compact!r}",
    )
    kernel_assert(
        not tail_before_guard or routed_compact == "hot_last",
        f"tail_before_guard moves the tail of the hot_last compaction: it needs routed_compact hot_last, got "
        f"{routed_compact!r}",
    )
    kernel_assert(
        store_queues in STORE_QUEUES
        and (store_queues == "sp" or (routed_reduce in ("ar", "rs_ag") and out_proj_form == "reduced")),
        f"store_queues must be one of {STORE_QUEUES}, and other than 'sp' only with routed_reduce 'ar' or 'rs_ag' and "
        f"the reduced out-projection, got {store_queues!r}, {routed_reduce!r}, {out_proj_form!r}",
    )
    if prefetch is not None:
        # A token-tiled call takes a stream of at most 3 buffers (moe_tkg_decode_token_tiles checks its order). With
        # routed_compact the static call of the first slot tile takes it (moe_tkg_decode checks it); a later slot tile
        # loads its own experts.
        tiled_stream = len(prefetch.buffers) <= _TOKEN_TILES_WEIGHT_BUFFERS
        kernel_assert(
            len(calls) == 1 and not half_weight_buffers and (calls[0][1] <= _PMAX or tiled_stream),
            f"A prefetched expert stream feeds one call with whole buffers: of at most {_PMAX} tokens, or token-tiled "
            f"(with or without routed_compact) with 1 to {_TOKEN_TILES_WEIGHT_BUFFERS} buffers, got the calls {calls}, "
            f"{len(prefetch.buffers)} buffers, routed_compact {routed_compact!r} and half_weight_buffers "
            f"{half_weight_buffers}",
        )
        loop_options = {**loop_options, "weight_buffers": len(prefetch.buffers), "stream": prefetch}
    kernel_assert(
        last_expert_order in LAST_EXPERT_ORDERS,
        f"last_expert_order must be one of {LAST_EXPERT_ORDERS}, got {last_expert_order!r}",
    )
    kernel_assert(
        last_expert_order == "default"
        or (
            cross_expert_order
            and transpose_block
            and all(count <= _PMAX for _, count in calls)
            and not routed_compact
            and not routed_per_expert
        ),
        f"last_expert_order {last_expert_order!r} is measured only on static cross-order expert calls with "
        f"transpose_block of at most {_PMAX} tokens (no routed_compact or routed_per_expert), got the calls {calls}, "
        f"cross_expert_order {cross_expert_order}, transpose_block {transpose_block}, routed_compact "
        f"{routed_compact!r}, routed_per_expert {routed_per_expert!r}",
    )
    # static_alt_tiles: the static moe_tkg_decode calls (at most 128 tokens) and the token-tiled calls take it.
    chunk_options = {
        **loop_options,
        **({"static_alt_tiles": static_alt_tiles} if static_alt_tiles else {}),
        **({"last_expert_order": last_expert_order} if last_expert_order != "default" else {}),
    }
    # The expert calls of the layer share one expert-loop scope: it counts the two-buffer static calls.
    stream_gate = (
        _stream_start_after(row_gathers[-1], expert_weights, hbm, rows_wait=weights_after_rows)
        if weights_after_gather or weights_after_rows
        else contextlib.nullcontext()
    )
    if accum_memset_after_rows:
        # The fill of each static call's accumulator follows the DP router-row gather (in its wait).
        chunk_options = {**chunk_options, "accum_memset": (nisa.engine.vector, row_gathers[0])}
    if routed_tiles:
        _check_routed_tile_options(
            expert_call_tokens,
            routed_compact,
            routed_per_expert,
            quantize_first,
            head_row_chunks,
            prefetch,
            rows_pairs,
            static_alt_tiles,
            keep_warm[0],
            cross_expert_order,
            transpose_lookahead,
            weights_after_gather or weights_after_rows,
        )
        _routed_tiles_partial(
            hbm,
            gather,
            tokens,
            rows,
            expert_weights,
            group_select,
            split,
            f"{name}_rt",
            router_bias=router_bias,
            routed_reduce=routed_reduce,
            rows_latent=rows_latent,
            rows_load_split=rows_load_split,
            store_queues=store_queues,
            routed_layout=routed_layout,
            rows_after=shared_store,
            **({"routed_dynamic": True} if routed_dynamic else {}),
            **({"routed_dynamic_pack": routed_dynamic_pack} if routed_dynamic_pack is not None else {}),
            **({"latent_gather_mx": True} if latent_gather_mx else {}),
            **({"mx_load_order": mx_load_order} if mx_load_order != "quadrant" else {}),
            **({"mx_scale_queue": mx_scale_queue} if mx_scale_queue != "act" else {}),
            **({"routed_dynamic_csr": routed_dynamic_csr} if routed_dynamic_csr is not None else {}),
        )
    with stream_gate, expert_loop_layer():
        for call_idx, (start, count) in enumerate(() if routed_tiles else calls):
            # A call of at most 128 tokens is one moe_tkg_decode call, as in the default program.
            if count <= _PMAX:
                _chunk_routed_partial(
                    hbm,
                    gather,
                    start,
                    count,
                    rows,
                    expert_weights,
                    group_select[0:count, :],
                    split,
                    routed_store_form,
                    chunk_options,
                    f"{name}_c{call_idx}",
                    router_bias=router_bias,
                    routed_reduce=routed_reduce,
                    **({"rows_after": shared_store} if shared_store is not None and call_idx == 0 else {}),
                    **({"quantize_first": True} if quantize_first else {}),
                    **({"head_row_chunks": True} if head_row_chunks else {}),
                    rows_latent=rows_latent,
                    rows_load_split=rows_load_split,
                    **({"latent_gather_mx": True} if latent_gather_mx else {}),
                    **({"mx_load_order": mx_load_order} if mx_load_order != "quadrant" else {}),
                    **({"mx_scale_queue": mx_scale_queue} if mx_scale_queue != "act" else {}),
                    **({"mx_reload_order": mx_reload_order} if mx_reload_order != "default" else {}),
                    **({"store_queues": store_queues} if store_queues != "sp" else {}),
                    **({"routed_layout": routed_layout} if routed_layout != "natural" else {}),
                    **({"pe_keep_warm": keep_warm} if keep_warm[0] else {}),
                    **({"scale_fill_after": scale_fill_after} if scale_fill_after else {}),
                )
            else:
                kernel_assert(
                    not (static_alt_tiles and routed_compact),
                    f"static_alt_tiles takes the token-tiled calls without routed_compact (the slot tiles of the "
                    f"compaction run the slot form), got routed_compact {routed_compact!r}",
                )
                kernel_assert(
                    not keep_warm[0],
                    f"pe_keep_warm{kind_note} is measured only for expert calls of at most {_PMAX} tokens, and this "
                    f"layer has a token-tiled call of {count} tokens",
                )
                # The token-tiled calls load the latent_gather_mx codes and scales without the load options.
                kernel_assert(
                    mx_load_order == "quadrant" and mx_scale_queue == "act" and mx_reload_order == "default",
                    f"mx_load_order {mx_load_order!r}, mx_scale_queue {mx_scale_queue!r} and mx_reload_order "
                    f"{mx_reload_order!r} are measured only for expert calls of at most {_PMAX} tokens, and this layer "
                    f"has a token-tiled call of {count} tokens. Use the defaults (quadrant, act, default) on this "
                    "form.",
                )
                tiles_partial = _compact_routed_partial if routed_compact else _tiles_routed_partial
                tiles_partial(
                    hbm,
                    gather,
                    start,
                    count,
                    rows,
                    expert_weights,
                    group_select,
                    split,
                    loop_options,
                    f"{name}_k{call_idx}",
                    router_bias=router_bias,
                    routed_reduce=routed_reduce,
                    **({"rows_after": shared_store} if shared_store is not None and call_idx == 0 else {}),
                    **({} if routed_compact else {"transpose_blocks": token_tiles_transpose_blocks}),
                    rows_latent=rows_latent,
                    rows_load_split=rows_load_split,
                    **({"latent_gather_mx": True} if latent_gather_mx else {}),
                    **({} if routed_compact else {"static_alt_tiles": static_alt_tiles}),
                    **({"late_always": True} if routed_compact == "late" else {}),
                    **({"hot_last": True} if routed_compact == "hot_last" else {}),
                    **({"empty_later": True} if routed_compact == "timing_empty_later" else {}),
                    **({"weights_first": tuple(early_weight_loads)} if weights_before_compact else {}),
                    **({"ungather_skip": True} if ungather_skip else {}),
                    **({"vector_cuts": True} if prologue_vector_cuts else {}),
                    **({"tables_scalar": True} if compact_tables_scalar else {}),
                    **({"affinity_sbuf_copy": True} if affinity_sbuf_copy else {}),
                    **({"zero_early": True} if tail_zero_early else {}),
                    **({"static_first": True} if slot_static_first else {}),
                    **({"up_product_split": True} if up_product_split else {}),
                    **({"mx_rows_stream": True} if mx_rows_stream else {}),
                    **({"store_queues": store_queues} if store_queues != "sp" else {}),
                    **({"routed_layout": routed_layout} if routed_layout != "natural" else {}),
                    **({"per_expert": routed_per_expert} if routed_per_expert else {}),
                    **(
                        {"slot_gather": compact_slot_gather}
                        if routed_compact and compact_slot_gather != "gpsimd"
                        else {}
                    ),
                    **({"tail_first": True} if tail_before_guard else {}),
                    **({"store_halves": True} if tail_store_halves else {}),
                    **({"alt_tiles": compact_alt_tiles} if compact_alt_tiles else {}),
                    # Only the compaction tail takes the hook (TAIL_GLUE_HOOK); the uncompacted tiles have none.
                    **(
                        {"tail_hook": glue_hooks[TAIL_GLUE_HOOK]}
                        if routed_compact and glue_hooks.get(TAIL_GLUE_HOOK)
                        else {}
                    ),
                    # The kernel's slot-table constants, when its entry built them (moe_2srv_compact_constants).
                    **(
                        {"constants": getattr(rank_info, COMPACT_CONSTANTS_KEY)}
                        if routed_compact and getattr(rank_info, COMPACT_CONSTANTS_KEY, None) is not None
                        else {}
                    ),
                    **({"rows_pairs": True} if rows_pairs else {}),
                )
    if out_proj_form == "column":
        out = column_out_projection(
            hbm,
            row_select if routed_reduce == "ar" else None,
            norm_weight,
            moe_out_proj_dp,
            rows,
            dp,
            world_group,
            tp_group,
            name,
            **({"routed_reduce": routed_reduce} if routed_reduce != "ar" else {}),
            **({"glue_hooks": glue_hooks} if glue_hooks else {}),
        )
        return out.reshape((_PMAX, rows, _H_TILES))
    if routed_reduce == "ar" and norm_tail == "token_major":
        routed_normed_token_major(
            hbm,
            _dp_replica_index(rank_info),
            shard_offset,
            norm_weight,
            act_tok[:, DP_LATENT_UP_ROWS[0] : DP_LATENT_UP_ROWS[1]],
            dp,
            world_group,
            name,
            routed_layout=routed_layout,
        )
        if out_proj_sb is None:
            out_proj_sb = load_out_proj()
    elif routed_reduce == "ar":
        normed = routed_normed_ar(
            hbm,
            row_select,
            shard_select,
            norm_weight,
            rows,
            dp,
            world_group,
            name,
            glue_hooks.get("routed_load"),
            after_trigger=glue_hooks.get("routed_trigger"),
            replica_index=replica_index,
            routed_layout=routed_layout,
        )
        if out_proj_sb is None:
            out_proj_sb = load_out_proj()
        normed_to_rows(normed, act_tok[:, DP_LATENT_UP_ROWS[0] : DP_LATENT_UP_ROWS[1]])
    elif routed_reduce == "rs_fold":
        sum_row = routed_fold_rs(
            hbm, norm_weight, act_tok[:, DP_LATENT_UP_ROWS[0] : DP_LATENT_UP_ROWS[1]], world_group, name
        )
        return fold_out_reduce(act_tok, moe_out_proj_dp, sum_row, hbm, tp_group, name)
    else:
        routed_norm_rs_ag(
            hbm,
            norm_weight,
            act_tok[:, DP_LATENT_UP_ROWS[0] : DP_LATENT_UP_ROWS[1]],
            world_group,
            tp_group,
            stats_gather,
            name,
            glue_hooks.get("routed_load"),
            after_trigger=glue_hooks.get("routed_trigger"),
        )
        if out_proj_sb is None:
            out_proj_sb = load_out_proj()
    partial_hT = out_projection(act_tok, out_proj_sb, name)
    # The load of the output all-reduce result is the consumer gate of the next body (with the block's consumer gates).
    out = _out_all_reduce(
        partial_hT,
        (hbm["out_src"], hbm["out_dst"]),
        tp_group,
        f"{name}_out",
        glue_hooks.get("out_trigger"),
        store_queues,
        record_consumer=True,
    )
    return out.reshape((_PMAX, rows, _H_TILES))


latent_moe_decode_2srv.token_major = True


def _load_mx_in_proj(moe_in_proj_dp: nl.NkiTensor, in_proj_scale: nl.NkiTensor, name: str, marks: list = None) -> tuple:
    """Load the MXFP8 in-projection ``[14, 128, 4 x 496]`` and the copies of its scales; return ``(weights, scales)``.

    Each DMA takes ``_MX_IN_PROJ_DMA_GROUPS`` groups, so the first matmuls start before the whole weight arrives.
    ``marks`` (a list, or None) receives the DMAs in their program order.
    """
    groups = moe_in_proj_dp.shape[0]
    # The packed MX width of the tensor (MX_LANES x the in-projection columns; 512 with latent_gather_mx, else 496).
    width = moe_in_proj_dp.shape[2]
    w_sb = nl.ndarray((_PMAX, groups, width), dtype=moe_in_proj_dp.dtype, buffer=nl.sbuf)
    for first in range(0, groups, _MX_IN_PROJ_DMA_GROUPS):
        count = min(_MX_IN_PROJ_DMA_GROUPS, groups - first)
        load = nisa.dma_copy(
            dst=w_sb[0:_PMAX, first : first + count, 0:width],
            src=moe_in_proj_dp[first : first + count, 0:_PMAX, 0:width].permute((1, 0, 2)),
            priority=dma_priority(_CRITICAL_DMA_PRIORITY),
            name=f"{name}_in_proj_mx_{first}",
        )
        if marks is not None:
            marks.append(load)

    def load(dst, src, copy):
        instr = nisa.dma_copy(
            dst=dst, src=src, priority=dma_priority(_CRITICAL_DMA_PRIORITY), name=f"{name}_in_proj_scale_{copy}"
        )
        if marks is not None:
            marks.append(instr)

    scales = load_mx_scales(in_proj_scale, lambda shape, copy: nl.ndarray(shape, dtype=nl.uint8, buffer=nl.sbuf), load)
    return w_sb.view(nl.float8_e4m3fn_x4), scales


def _mx_in_proj(dst, x_mx, in_proj_mx, col: int, cols: int, padded: int, after=None):
    """Accumulate ``dst[0:padded, 0:cols] = x @ W[:, col:col + cols]`` over the MX groups of the MXFP8 hidden
    (stationary) and the MXFP8 in-projection (moving). ``after``: an instruction that the first matmul follows in the
    schedule. Returns the last matmul."""
    x_data, x_scale = x_mx
    w_x4, scales = in_proj_mx
    last = None
    for group in range(w_x4.shape[1]):
        last = nisa.nc_matmul_mx(
            dst=dst[0:padded, 0:cols],
            stationary=x_data[0:_PMAX, group, 0:padded],
            moving=w_x4[0:_PMAX, group, col : col + cols],
            stationary_scale=x_scale[0:_PMAX, group, 0:padded],
            moving_scale=mx_scale_view(scales, group, col, cols),
            accumulate=group > 0,
        )
        if after is not None and group == 0:
            _instruction(last).depends_on(_instruction(after), kind=nisa.dependency.schedule)
    return last


def _load_mx_out_proj(
    moe_out_proj_dp: nl.NkiTensor,
    out_proj_scale: nl.NkiTensor,
    name: str,
    issued: list = None,
    after=None,
    after_done: bool = False,
    priority: int = None,
) -> tuple:
    """Load the MXFP8 padded-lane out-projection ``[1, 128, 4 H]`` in chunks along H, and its scales; return
    ``(weights, scales)``. One MX group reads only the first scale copy of ``load_mx_scales``, so it loads that one.
    ``issued`` (a list), when given, gets the DMAs. ``after`` (``out_proj_after_rows``), when given, is an instruction
    that each DMA follows in the schedule; the DMAs then take the GpSimd software DGE queue by name.
    ``after_done``: each DMA also waits for ``after`` to complete (a runtime dependency). ``priority``
    (``out_proj_priority``): the trn3 DMA QoS level of the DMAs; None (default) keeps the critical level."""
    kinds = (nisa.dependency.schedule, nisa.dependency.runtime) if after_done else (nisa.dependency.schedule,)
    level = _CRITICAL_DMA_PRIORITY if priority is None else priority
    queue = {} if after is None else {"dge_mode": nisa.dge_mode.swdge}
    width = moe_out_proj_dp.shape[2]
    w_sb = nl.ndarray((_PMAX, width), dtype=moe_out_proj_dp.dtype, buffer=nl.sbuf)
    chunk = width // _OUT_PROJ_DMA_CHUNKS
    for chunk_idx in range(_OUT_PROJ_DMA_CHUNKS):
        cols = slice(chunk_idx * chunk, (chunk_idx + 1) * chunk)
        load = nisa.dma_copy(
            dst=w_sb[0:_PMAX, cols],
            src=moe_out_proj_dp[0, 0:_PMAX, cols],
            priority=level,
            name=f"{name}_out_proj_mx_c{chunk_idx}",
            **queue,
        )
        if after is not None:
            for kind in kinds:
                _instruction(load).depends_on(_instruction(after), kind=kind)
        if issued is not None:
            issued.append(load)
    scale_sb = nl.ndarray(tuple(out_proj_scale.shape), dtype=nl.uint8, buffer=nl.sbuf)
    load = nisa.dma_copy(dst=scale_sb, src=out_proj_scale, priority=level, name=f"{name}_out_proj_scale", **queue)
    if after is not None:
        for kind in kinds:
            _instruction(load).depends_on(_instruction(after), kind=kind)
    if issued is not None:
        issued.append(load)
    return w_sb.view(nl.float8_e4m3fn_x4), [scale_sb]


def out_projection(act_tok: nl.NkiTensor, out_proj_sb, name: str) -> nl.NkiTensor:
    """Return the BF16 ``act_tok @ moe_out_proj_dp`` of the ``[B, 304]`` activation in the layout of ``_project_to_hT``.

    ``out_proj_sb`` is the BF16 ``[128, 3, H]`` contraction tiles of ``_load_out_proj``, or the MXFP8
    ``(weights, scales)`` of ``_load_mx_out_proj``.
    """
    k_tiles = _out_k_tiles(DP_OUT_PROJ_ROWS)
    if isinstance(out_proj_sb, tuple):
        return _mx_out_proj(act_tok, out_proj_sb, k_tiles, name)
    act_T = _transpose_rows(act_tok, k_tiles)
    return _project_to_hT(
        lambda h: [
            (out_proj_sb[0 : stop - start, k_idx, h * _PMAX : (h + 1) * _PMAX], act_T[0 : stop - start, k_idx, :])
            for k_idx, (start, stop) in enumerate(k_tiles)
        ],
        act_tok.shape[0],
    )


def _mx_out_proj(act_tok: nl.NkiTensor, out_proj_mx: tuple, k_tiles: tuple, name: str) -> nl.NkiTensor:
    """Return ``bf16(act_tok @ W)`` of the MXFP8 padded-lane out-projection, in the layout of ``_project_to_hT``.

    Each contraction tile of the ``[B, 304]`` activation is one PE transpose into a hT_quad lane of one MX group, with
    zero rows past the 304; VectorE quantizes the group. Each hidden tile is one MXFP8 matmul over the 512 rows.
    """
    rows = act_tok.shape[0]
    act_T = nl.ndarray((_PMAX, len(k_tiles), rows), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.memset(dst=act_T[0:_PMAX, len(k_tiles) - 1, 0:rows], value=0.0)
    for k_idx, (start, stop) in enumerate(k_tiles):
        psum = _psum_bank(stop - start, (rows,), nl.bfloat16)
        nisa.nc_transpose(dst=psum, data=act_tok[:, start:stop])
        nisa.tensor_copy(dst=act_T[0 : stop - start, k_idx, :], src=psum)
    # An even token stride keeps the moving columns on an 8-byte boundary (MatmulMX refuses odd x4 starts).
    act_mx, act_scale = quantize_hT_mx(act_T, f"{name}_out_proj", rows + rows % 2, tile_major=True)
    w_x4, scales = out_proj_mx
    tiles_per_bank = min(_H_TILES, _PSUM_BANK_FP32 // rows)
    partial_hT = nl.ndarray((_PMAX, rows, _H_TILES), dtype=nl.bfloat16, buffer=nl.sbuf)
    for first in range(0, _H_TILES, tiles_per_bank):
        n_tiles = min(tiles_per_bank, _H_TILES - first)
        group_psum = _psum_bank(_PMAX, (n_tiles, rows), nl.float32)
        for offset in range(n_tiles):
            cols = (first + offset) * _PMAX
            nisa.nc_matmul_mx(
                dst=group_psum[:, offset, :],
                stationary=w_x4[0:_PMAX, cols : cols + _PMAX],
                moving=act_mx[0:_PMAX, 0, 0:rows],
                stationary_scale=mx_scale_view(scales, 0, cols, _PMAX),
                moving_scale=act_scale[0:_PMAX, 0, 0:rows],
                accumulate=False,
            )
        nisa.tensor_copy(dst=partial_hT[:, :, first : first + n_tiles], src=group_psum.permute((0, 2, 1)))
    return partial_hT.reshape((_PMAX, rows * _H_TILES))


def _check_router(x_hT: nl.NkiTensor, router_dtype: str, router_weight: nl.NkiTensor, x16_hT: nl.NkiTensor):
    """Check the router of the 2-server MoE body (``ROUTER_DTYPES_2SRV``): the FP16 router takes both inputs."""
    kernel_assert(
        router_dtype in ROUTER_DTYPES_2SRV, f"router_dtype must be one of {ROUTER_DTYPES_2SRV}, got {router_dtype!r}"
    )
    if router_dtype == "bf16":
        kernel_assert(router_weight is None and x16_hT is None, "the BF16 router takes no FP16 router inputs")
        return
    kernel_assert(
        x16_hT is not None and tuple(x16_hT.shape) == tuple(x_hT.shape) and x16_hT.dtype == nl.float16,
        f"router_dtype 'fp16' needs x16_hT FP16 {tuple(x_hT.shape)}",
    )
    shape = (_PMAX, _H_TILES, DP_EXPERTS)
    kernel_assert(
        router_weight is not None and tuple(router_weight.shape) == shape and router_weight.dtype == nl.float16,
        f"router_dtype 'fp16' needs router_weight FP16 {shape}",
    )


def moe_2srv_body(
    tp_group: ReplicaGroup,
    world_size: int,
    routed_tp: int = None,
    gather: str = "dp_tp",
    wide_half_weight_buffers: int = WIDE_HALF_WEIGHT_BUFFERS,
    expert_call_tokens: int | str = EXPERT_CALL_AUTO,
    drop_selection: bool = False,
    routed_reduce: str = "ar",
    stats_gather: bool = False,
    out_proj_form: str = "reduced",
    quantize_first: bool = False,
    weights_after_gather: bool = False,
    head_row_chunks: bool = False,
    expert_queues: int = 1,
    rows_latent: str = "bf16",
    rows_load_split: int = 1,
    routed_own_rows: bool = False,
    latent_gather_mx: bool = False,
    glue: dict = None,
    mx_tile: int = None,
    routed_dynamic: bool = ROUTED_DYNAMIC_DEFAULT,
    routed_dynamic_csr: str = None,
):
    """Return the body of the MoE layers of a multi-server world, with the call form of ``decode_block_body``.

    The block calls ``body(x_hT, *weights, rank_info, dp_group, world_group, tag, sources=..., dp_size=dp,
    **expert_loop_options)`` with the ``MOE_2SRV_WEIGHT_NAMES`` weights of the layer. ``sources`` holds this body's
    tensors under ``MOE_SOURCES_KEY`` (``moe_2srv_collective_hbm``). The body takes the ``EXPERT_LOOP_KEYS`` options
    (and ``keep_warm_lead`` when ``keep_warm_by_lead``: a per-kind ``pe_keep_warm`` in ``glue``) and fails on
    any other. ``tp_group`` is the TP32 group of the replicas (``tkg_ffn_world_cc.replica_groups_world``).
    ``gather`` picks the router-row gathers (``MOE_GATHERS``), ``wide_half_weight_buffers`` the half buffers of the
    whole-expert form above ``WIDE_ROWS`` rows, ``expert_call_tokens`` the tokens of one expert call ("auto" or
    ``EXPERT_CALL_FORMS``), ``drop_selection`` the 84-word router rows, and ``routed_reduce`` / ``stats_gather`` the
    reduction of the routed partials (``ROUTED_REDUCES``; the sources must come from ``moe_2srv_collective_hbm`` with
    the same values) and ``out_proj_form`` the out projection and output combine (``OUT_PROJ_FORMS``; the same rule for
    the sources), ``quantize_first`` the latent quantize of each expert call before its affinities, and
    ``weights_after_gather`` the schedule edges of the stream start of the expert loop after the router-row gathers,
    and ``head_row_chunks`` the chunked rows load and latent quantize of each expert call (``latent_moe_decode_2srv``). The call option ``pad_rows`` (``decode_block_body`` ``moe_pad_rows``;
    ``takes_pad_rows``) routes the padding rows to no expert (``latent_moe_decode_2srv``). ``expert_queues`` sets the
    DMA queues of the static expert loop's weight loads, in every call and in the ``prefetch`` streams.
    ``rows_load_split``: the parts of the router-rows load of each expert call (``latent_moe_decode_2srv``; 1 keeps the
    program). ``routed_own_rows``: load only the replica's rows of the routed all-reduce (``latent_moe_decode_2srv``;
    False keeps the program). ``latent_gather_mx``: send the MXFP8 latent codes in the router rows and run no latent
    quantize in the expert calls, static or token-tiled (``latent_moe_decode_2srv``; False keeps the program; the
    sources must come from
    ``moe_2srv_collective_hbm`` with the same value). ``glue`` holds the glue options of ``GLUE_KEYS`` with values other
    than their defaults (None or empty: the default program). ``mx_tile``: the MX tile width of the expert split
    (``moe_2srv_world_split``); None (default) keeps the split of 512-column MX tiles. ``routed_dynamic``: the dynamic
    routed form of a split of sub-512 MX tiles (``latent_moe_decode_2srv``); False (default) keeps the program.
    ``routed_dynamic_csr``: the CSR gathers of its packed layers (``latent_moe_decode_2srv``); None (default) keeps
    "staged". The entries take it from ``moe_routed_plan.call_csr_gather``.
    """
    split = moe_2srv_world_split(world_size, routed_tp, mx_tile)
    kernel_assert(expert_queues in EXPERT_QUEUES, f"expert_queues must be one of {EXPERT_QUEUES}, got {expert_queues}")
    kernel_assert(gather in MOE_GATHERS, f"gather must be one of {MOE_GATHERS}, got {gather!r}")
    _check_routed_reduce(routed_reduce, stats_gather)
    _check_out_proj_form(out_proj_form, stats_gather)
    kernel_assert(
        not routed_own_rows or routed_reduce == "ar",
        f"routed_own_rows loads the replica rows of the routed all-reduce (routed_reduce 'ar'), got {routed_reduce!r}",
    )
    glue = dict(glue or {})
    kernel_assert(not set(glue) - set(GLUE_KEYS), f"glue takes the keys {sorted(GLUE_KEYS)}, got {sorted(glue)}")
    # dense_x_gather acts at the verify entry (the dense body); the MoE body does not take it.
    glue.pop("dense_x_gather", None)
    # ahead_sums_vector acts at the verify entry (the AttnRes bank parts); the MoE body does not take it.
    glue.pop("ahead_sums_vector", None)
    glue.pop("ahead_load_tiles", None)
    glue.pop("ahead_before_resid", None)
    glue.pop("ahead_mlp_kda_tail", None)
    glue.pop("kda_mlp_tail_entries", None)
    glue.pop("mla_mlp_tail_entries", None)
    glue.pop("ahead_gram_merge", None)
    glue.pop("kda_slot_ring_per_call", None)
    glue.pop("kda_commit_after_w_in", None)
    glue.pop("kda_slot_store_regs", None)
    glue.pop("kda_slot_conv_after_ar", None)
    glue.pop("kda_weight_dma", None)
    glue.pop("norm_one_mul", None)
    glue.pop("cached_act_squares", None)
    glue.pop("kda_in_proj_x_stationary", None)
    glue.pop("kda_state_loads_after_ring", None)
    glue.pop("kda_w_out_after_w_in_fills", None)
    glue.pop("kda_prework_ahead", None)
    glue.pop("attn_res_sbuf_entries", None)
    if not ROUTED_COMPACT_ON:
        # The kernel-side kill switch of the compaction, whatever the caller's glue.
        glue = {key: value for key, value in glue.items() if key not in COMPACT_GLUE_KEYS}
    # A split of sub-512 MX tiles runs the routed-tile loop: the static loop's levers of the caller's glue do not apply.
    tile_form = split.get("layout_form") == ROUTED_TILE_FORM
    if tile_form:
        glue = {key: value for key, value in glue.items() if key not in TILE_FORM_STATIC_GLUE}
    # The norm before the body reads router_norm_one_mul (the body attribute below); the body does not take it.
    router_norm_one_mul = bool(glue.pop("router_norm_one_mul", False))
    # A per-kind pe_keep_warm: the block gives each call the kind of the layer's attention (keep_warm_by_lead).
    by_lead = _keep_warm_by_lead(glue.get("pe_keep_warm"))
    scale_quadrants = bool(glue.get("scale_rows_quadrants", False))
    scale_queue = glue.get("scale_rows_queue", GLUE_KEYS["scale_rows_queue"])
    # last_expert_order "up_first": a ring in the cross-expert order loads up(E_L - 1) before down(E_L - 2).
    last_up_first = glue.get("last_expert_order", GLUE_KEYS["last_expert_order"]) == "up_first"
    # compact_key_constants acts at the entry (moe_2srv_compact_constants); the body does not take it.
    call_glue = {
        key: value
        for key, value in glue.items()
        if key not in ("scale_rows_quadrants", "scale_rows_queue", "compact_key_constants")
    }

    def body(
        x_hT, *args, sources=None, dp_size=split["dp"], glue_hooks=None, pad_rows=None, in_proj_sb=None, **options
    ):
        kernel_assert(dp_size == split["dp"], f"dp_size {dp_size} is not world / 32 = {split['dp']}")
        if tile_form:
            options = {key: value for key, value in options.items() if key not in TILE_FORM_STATIC_OPTIONS}
        taken = EXPERT_LOOP_KEYS | ROUTER_KEYS | MX_KEYS | (LEAD_KEYS if by_lead else set())
        unknown = sorted(set(options) - taken)
        kernel_assert(not unknown, f"The 2-server MoE body takes the options {sorted(taken)} only, got {unknown}")
        kernel_assert(
            not by_lead or options.get("keep_warm_lead") in PE_KEEP_WARM_LEADS,
            f"a per-kind pe_keep_warm needs the layer's keep_warm_lead (one of {PE_KEEP_WARM_LEADS}), got "
            f"{options.get('keep_warm_lead')!r}",
        )
        kernel_assert(
            len(args) == len(MOE_2SRV_WEIGHT_NAMES) + 4,
            f"expected {len(MOE_2SRV_WEIGHT_NAMES) + 4} arguments after x_hT, got {len(args)}",
        )
        kernel_assert(
            sources is not None and MOE_SOURCES_KEY in sources, f"The MoE body needs sources[{MOE_SOURCES_KEY!r}]"
        )
        *weights, rank_info, dp_group, world_group, tag = args
        with scale_rows_across_quadrants(scale_quadrants), scale_rows_queue(scale_queue):
            return _call_body(
                x_hT, weights, rank_info, dp_group, world_group, tag, sources, glue_hooks, pad_rows, in_proj_sb, options
            )

    def _call_body(
        x_hT, weights, rank_info, dp_group, world_group, tag, sources, glue_hooks, pad_rows, in_proj_sb, options
    ):
        return latent_moe_decode_2srv(
            x_hT,
            *weights,
            rank_info,
            dp_group,
            world_group,
            tp_group,
            tag,
            sources[MOE_SOURCES_KEY],
            split,
            gather=gather,
            wide_half_weight_buffers=wide_half_weight_buffers,
            expert_call_tokens=expert_call_tokens,
            routed_reduce=routed_reduce,
            stats_gather=stats_gather,
            **({"drop_selection": True} if drop_selection else {}),
            **({"out_proj_form": out_proj_form} if out_proj_form != "reduced" else {}),
            **({"glue_hooks": glue_hooks} if glue_hooks else {}),
            **({"quantize_first": True} if quantize_first else {}),
            **({"weights_after_gather": True} if weights_after_gather else {}),
            **({"head_row_chunks": True} if head_row_chunks else {}),
            **({"pad_rows": pad_rows} if pad_rows is not None else {}),
            **({"in_proj_sb": in_proj_sb} if in_proj_sb is not None else {}),
            **({"expert_queues": expert_queues} if expert_queues != 1 else {}),
            **({"rows_latent": rows_latent} if rows_latent != "bf16" else {}),
            **({"rows_load_split": rows_load_split} if rows_load_split != 1 else {}),
            **({"routed_own_rows": True} if routed_own_rows else {}),
            **({"latent_gather_mx": True} if latent_gather_mx else {}),
            **({"routed_dynamic": True} if routed_dynamic else {}),
            **({"routed_dynamic_csr": routed_dynamic_csr} if routed_dynamic_csr is not None else {}),
            **call_glue,
            **options,
        )

    def prefetch(
        *weights,
        weight_buffers: int,
        scale_buffers: int = 1,
        use_order: bool = False,
        priority: int = None,
        uses: int = None,
        scales: bool = True,
    ):
        """Start the expert ring of a layer whose ``MOE_2SRV_WEIGHT_NAMES`` weights are ``weights``
        (``expert_prefetch_stream``); the layer's call takes the result as ``prefetch``. ``use_order``: the order of a
        call with ``cross_expert_order``. ``priority``: the DMA QoS level of the ring loads (None: no level). ``uses``:
        with ``use_order``, load only the first ``uses`` buffers now (None: all); ``scales`` False leaves the scale groups
        to the call. A split of sub-512 MX tiles returns None: the stream holds the served layout, and the routed-tile
        loop loads its own weights (the decode block then gives the call no stream)."""
        if tile_form:
            return None
        kernel_assert(len(weights) == len(MOE_2SRV_WEIGHT_NAMES), f"expected the {len(MOE_2SRV_WEIGHT_NAMES)} weights")
        with scale_rows_across_quadrants(scale_quadrants), scale_rows_queue(scale_queue):
            return expert_prefetch_stream(
                tuple(weights[3:7]),
                weight_buffers,
                scale_buffers,
                expert_queues,
                use_order,
                priority=priority,
                **({"uses": uses} if uses is not None else {}),
                **({} if scales else {"scales": False}),
                **({"last_up_first": True} if use_order and last_up_first else {}),
            )

    body.token_major = True
    body.world_size = world_size
    body.prefetch = prefetch
    body.moe_sharding = "dp_2srv"
    body.takes_pad_rows = True
    body.router_dtypes = ROUTER_DTYPES_2SRV
    body.in_proj_mx = True
    body.out_proj_mx = True
    body.moe_gather = gather
    body.expert_call_tokens = expert_call_tokens
    body.drop_selection = drop_selection
    body.moe_routed_reduce = routed_reduce
    body.moe_routed_stats_gather = stats_gather
    body.moe_out_proj_form = out_proj_form
    body.moe_quantize_first = quantize_first
    body.moe_weights_after_gather = weights_after_gather
    body.load_in_proj = load_moe_in_proj_2srv
    body.build_selectors = functools.partial(moe_2srv_selectors, split=split)
    body.moe_head_row_chunks = head_row_chunks
    body.expert_queues = expert_queues
    body.router_norm_one_mul = router_norm_one_mul
    body.rows_latent = rows_latent
    body.rows_load_split = rows_load_split
    body.routed_own_rows = routed_own_rows
    body.latent_gather_mx = latent_gather_mx
    body.routed_dynamic = routed_dynamic
    body.routed_dynamic_csr = routed_dynamic_csr
    body.moe_glue = dict(glue)
    body.keep_warm_by_lead = by_lead
    return body


def load_moe_in_proj_2srv(moe_in_proj_dp: nl.NkiTensor, tag: str, after=None) -> nl.NkiTensor:
    """Load one layer's ``moe_in_proj_dp`` [128, 56, 524] BF16 HBM into a new SBUF tile and return the tile.

    The DMAs are those of ``latent_moe_decode_2srv``: the same call (its 4 hidden-tile chunks, named
    ``k3moe2s_<tag>_in_proj_c<i>``, on the default DGE queue). ``after`` is None, or the instruction that each DMA
    follows in the schedule (the gate collective of the body, for example the attention all-reduce of the layer:
    ``decode_block_body`` ``moe_in_proj_at_kda_reduce``). Pass the tile to the body as ``in_proj_sb``. The values do not
    change.
    """
    kernel_assert(
        tuple(moe_in_proj_dp.shape) == (_PMAX, _H_TILES, DP_IN_PROJ_COLS) and moe_in_proj_dp.dtype == nl.bfloat16,
        f"moe_in_proj_dp must be BF16 [{_PMAX}, {_H_TILES}, {DP_IN_PROJ_COLS}], got "
        f"{moe_in_proj_dp.dtype} {moe_in_proj_dp.shape}",
    )
    in_proj_sb = nl.ndarray((_PMAX, _H_TILES, DP_IN_PROJ_COLS), dtype=nl.bfloat16, buffer=nl.sbuf)
    _load_hT_rows(in_proj_sb, moe_in_proj_dp, _IN_PROJ_DMA_CHUNKS, f"k3moe2s_{tag}_in_proj", after=after)
    return in_proj_sb


def expert_prefetch_stream(
    expert_weights: tuple,
    weight_buffers: int,
    scale_buffers: int = 1,
    expert_queues: int = 1,
    use_order: bool = False,
    priority: int = None,
    uses: int = None,
    scales: bool = True,
    last_up_first: bool = False,
):
    """Allocate the static expert loop's buffers of one MoE layer and issue their first loads now
    (``moe_routed.static_expert_stream``), for the ``prefetch`` of the layer's ``latent_moe_decode_2srv`` call.

    ``expert_weights`` are the layer's (gate_up, gate_up_scale, down, down_scale) of this rank, in the order of
    ``MOE_2SRV_WEIGHT_NAMES``. ``weight_buffers`` (1 to 6 whole buffers) sets the depth of the ring and
    ``scale_buffers`` and ``expert_queues`` must be those of the call (the body defaults 1). ``use_order`` loads the
    buffers in the cross-expert order, for a call with ``cross_expert_order`` (3 whole buffers). ``priority`` gives the
    ring loads a trn3 DMA QoS level (``static_expert_stream``); None (default) keeps the program. ``uses`` (with
    ``use_order``) loads only the first ``uses`` buffers now and leaves the others to the call's start
    (``static_expert_stream`` ``prefetch_uses``); None (default) loads all. ``scales`` False leaves the scale groups to
    the call's start too (``prefetch_scales``). ``last_up_first`` (with ``use_order``): the use order of a call with
    ``last_expert_order`` "up_first" (``static_expert_stream``); False (default) keeps the program.
    """
    gate_up, gate_up_scale, down, down_scale = expert_weights
    if gate_up.dtype == nl.uint16:
        gate_up = gate_up.view(nl.float4_e2m1fn_x4)
    if down.dtype == nl.uint16:
        down = down.view(nl.float4_e2m1fn_x4)
    return static_expert_stream(
        gate_up,
        down,
        gate_up_scale,
        down_scale,
        weight_buffers,
        scale_buffers,
        expert_queues,
        use_order=use_order,
        **({"priority": priority} if priority is not None else {}),
        **({"prefetch_uses": uses} if uses is not None else {}),
        **({} if scales else {"prefetch_scales": False}),
        **({"last_up_first": True} if last_up_first else {}),
    )


def load_norm_weight(
    moe_vectors_dp: nl.NkiTensor, rows: int, routed_reduce: str, name: str, issued: list = None
) -> nl.NkiTensor:
    """Load this rank's latent norm weights from ``moe_vectors_dp`` in the layout of ``routed_reduce``: ``ar`` [28, 4]
    FP32 (element (c, q) is the weight of column 112 t + 28 q + c, ``_latent_norm_rows``); ``rs_ag`` [B, 112] FP32
    (element (b, 28 q + c), the same weight on every row, ``routed_norm_rs_ag``). ``issued`` (a list), when given,
    gets the DMA."""
    if routed_reduce == "ar":
        norm_weight = nl.ndarray((ROUTED_COLS, _SHARD_PARTITIONS), dtype=nl.float32, buffer=nl.sbuf)
        load = nisa.dma_copy(
            dst=norm_weight,
            src=moe_vectors_dp.ap(pattern=[[1, ROUTED_COLS], [ROUTED_COLS, _SHARD_PARTITIONS]], offset=DP_NORM_COLS[0]),
            priority=_CRITICAL_DMA_PRIORITY,
            name=f"{name}_norm_weight",
        )
        if issued is not None:
            issued.append(load)
        return norm_weight
    norm_weight = nl.ndarray((rows, DP_LATENT), dtype=nl.float32, buffer=nl.sbuf)
    load = nisa.dma_copy(
        dst=norm_weight,
        src=moe_vectors_dp.ap(pattern=[[0, rows], [1, DP_LATENT]], offset=DP_NORM_COLS[0]),
        name=f"{name}_norm_weight",
        **_HBM_DMA,
    )
    if issued is not None:
        issued.append(load)
    return norm_weight


def routed_normed_ar(
    hbm: dict,
    row_select: nl.NkiTensor,
    shard_select: nl.NkiTensor,
    norm_weight: nl.NkiTensor,
    rows: int,
    dp: int,
    world_group: ReplicaGroup,
    name: str,
    after_load=None,
    after_trigger=None,
    replica_index: nl.NkiTensor = None,
    routed_layout: str = "natural",
) -> nl.NkiTensor:
    """All-reduce the routed partials of ``hbm`` over ``world_group`` (``routed_reduce`` ``ar``), load the result and
    return this rank's normed latent shard of its replica's rows: [28, B, 4] BF16, element (c, b, q) is column
    112 t + 28 q + c of row b (``_latent_norm_rows``). The all-reduce is the gate of the next body. ``after_load`` is
    None, or a function that is called with the all-reduce and the load of its result, right after that load;
    ``after_trigger`` is None, or a function that is called with the all-reduce right after it. ``replica_index``
    ([1, 1] int32, ``routed_own_rows``): load only the replica's rows, at that scalar offset."""
    tokens = hbm["routed_src"].shape[1] // ROUTED_COLS
    check_pair_group(hbm["routed_dst"], world_group, f"{name}_routed_reduce")
    reduce = release_weight_gate(
        ncc.all_reduce(
            srcs=[hbm["routed_src"]],
            dsts=[hbm["routed_dst"]],
            replica_group=world_group,
            op=nl.add,
            name=f"{name}_routed_reduce",
        )
    )
    if after_trigger is not None:
        after_trigger(reduce)
    if replica_index is not None:
        width = rows * ROUTED_COLS
        own = nl.ndarray((ROUTED_PARTITIONS, width), dtype=nl.bfloat16, buffer=nl.sbuf)
        if routed_layout == "qsplit":
            half_width = width // 2
            load = nisa.dma_copy(
                dst=own.reshape((ROUTED_PARTITIONS, 2, half_width)),
                src=_routed_halves(hbm["routed_dst"], dp, rows).ap(
                    pattern=[[2 * dp * half_width, ROUTED_PARTITIONS], [dp * half_width, 2], [1, half_width]],
                    offset=0,
                    scalar_offset=replica_index,
                    indirect_dim=0,
                ),
                name=f"{name}_routed_own_load",
                **_HBM_DMA,
            )
        else:
            # Row 128 x dp view: element (p, r, w) is row p dp + r; the scalar offset r moves the start by r * width.
            load = nisa.dma_copy(
                dst=own,
                src=hbm["routed_dst"]
                .reshape((ROUTED_PARTITIONS * dp, width))
                .ap(
                    pattern=[[dp * width, ROUTED_PARTITIONS], [1, width]],
                    offset=0,
                    scalar_offset=replica_index,
                    indirect_dim=0,
                ),
                name=f"{name}_routed_own_load",
                **_HBM_DMA,
            )
        if after_load is not None:
            after_load(reduce, load)
        normed = nl.ndarray((ROUTED_COLS, rows, _SHARD_PARTITIONS), dtype=nl.bfloat16, buffer=nl.sbuf)
        _latent_norm_rows(own, None, shard_select, norm_weight, normed, 1, f"{name}_norm")
        return normed
    routed = nl.ndarray((ROUTED_PARTITIONS, tokens * ROUTED_COLS), dtype=nl.bfloat16, buffer=nl.sbuf)
    if routed_layout == "qsplit":
        load = _load_routed_natural(hbm["routed_dst"], routed, rows, dp, f"{name}_routed_load")
    else:
        load = nisa.dma_copy(dst=routed, src=hbm["routed_dst"], name=f"{name}_routed_load", **_HBM_DMA)
    if after_load is not None:
        after_load(reduce, load)
    normed = nl.ndarray((ROUTED_COLS, rows, _SHARD_PARTITIONS), dtype=nl.bfloat16, buffer=nl.sbuf)
    _latent_norm_rows(routed, row_select, shard_select, norm_weight, normed, dp, f"{name}_norm")
    return normed


def _load_routed_natural(routed_dst: nl.NkiTensor, dst: nl.NkiTensor, rows: int, dp: int, name: str):
    """Load the whole ``qsplit`` routed result into the natural [128, dp B * 28] order (``ROUTED_LAYOUTS``): one DMA,
    runs of B / 2 * 28 elements. Return the DMA."""
    half_width = rows // 2 * ROUTED_COLS
    return nisa.dma_copy(
        dst=dst.reshape((ROUTED_PARTITIONS, dp, 2, half_width)),
        src=routed_dst.ap(
            pattern=[[2 * dp * half_width, ROUTED_PARTITIONS], [half_width, dp], [dp * half_width, 2], [1, half_width]],
            offset=0,
        ),
        name=name,
        **_HBM_DMA,
    )


def routed_normed_token_major(
    hbm: dict,
    replica_offset: nl.NkiTensor,
    shard_offset: nl.NkiTensor,
    norm_weight: nl.NkiTensor,
    dst: nl.NkiTensor,
    dp: int,
    world_group: ReplicaGroup,
    name: str,
    routed_layout: str = "natural",
):
    """All-reduce the routed partials of ``hbm`` over ``world_group`` (``routed_reduce`` ``ar``) and write this rank's
    normed latent shard of its replica's rows into ``dst`` token-major (``norm_tail`` ``token_major``).

    The all-reduce is the one of ``routed_normed_ar`` (the gate of the next body). Then two DMAs read the result at
    runtime offsets: the replica's B tokens, [128, B * 28] (``replica_offset``, the INT32 [1, 1] DP rank r, steps one
    block of B * 28 columns), for the norm sums; and the rank's 112 columns of those tokens token-major, [B, 4, 28]
    (``shard_offset``, the INT32 [1, 1] block 4 t dp + r of ``_shard_offset``): element (b, q, c) is partition 4 t + q,
    column 28 (r B + b) + c, latent column 112 t + 28 q + c of row b. The inverse RMS is the one of ``_full_row_rstd``
    (the squares, the sums and the ones matmul of ``_latent_norm_rows``, then a VectorE transpose) and
    ``dst[b, 28 q + c] = bf16(bf16(x * rstd_b) * w)``, the per-element arithmetic of ``_latent_norm_rows``. A DMA
    moves the bytes, so no select copy, one-hot matmul or transpose of the normed values runs. B is at most 32.

    Args:
        norm_weight: [B, 112] FP32 SBUF, the ``rs_ag`` layout of ``load_norm_weight``.
        dst: [B, 112] BF16 SBUF view, the latent rows of the out-projection input.
    """
    rows = dst.shape[0]
    width = rows * ROUTED_COLS
    release_weight_gate(
        ncc.all_reduce(
            srcs=[hbm["routed_src"]],
            dsts=[hbm["routed_dst"]],
            replica_group=world_group,
            op=nl.add,
            name=f"{name}_routed_reduce",
        )
    )
    kernel_assert(
        tuple(hbm["routed_dst"].shape) == (ROUTED_PARTITIONS, dp * width),
        f"routed_dst must be [{ROUTED_PARTITIONS}, {dp * width}], got {hbm['routed_dst'].shape}",
    )
    if routed_layout == "qsplit":
        replica_rows, own = _token_major_loads_qsplit(hbm["routed_dst"], replica_offset, shard_offset, rows, dp, name)
    else:
        blocks = hbm["routed_dst"].reshape((ROUTED_PARTITIONS * dp, width))
        replica_rows = nl.ndarray((ROUTED_PARTITIONS, width), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=replica_rows,
            src=blocks.ap(
                pattern=[[dp * width, ROUTED_PARTITIONS], [1, width]],
                offset=0,
                scalar_offset=replica_offset,
                indirect_dim=0,
            ),
            name=f"{name}_routed_replica_load",
            **_HBM_DMA,
        )
        own = nl.ndarray((rows, _SHARD_PARTITIONS, ROUTED_COLS), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=own,
            src=blocks.ap(
                pattern=[[ROUTED_COLS, rows], [dp * width, _SHARD_PARTITIONS], [1, ROUTED_COLS]],
                offset=0,
                scalar_offset=shard_offset,
                indirect_dim=0,
            ),
            name=f"{name}_routed_own_load",
            **_HBM_DMA,
        )
    rstd = _full_row_rstd(replica_rows, rows)
    own_rows = own.reshape((rows, DP_LATENT))
    scaled = nl.ndarray((rows, DP_LATENT), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=scaled, data=own_rows, op0=nl.multiply, operand0=rstd[:, 0:1], engine=nisa.engine.vector)
    nisa.tensor_tensor(dst=dst, data1=scaled, data2=norm_weight, op=nl.multiply, engine=nisa.engine.vector)


def _token_major_loads_qsplit(
    routed_dst: nl.NkiTensor,
    replica_offset: nl.NkiTensor,
    shard_offset: nl.NkiTensor,
    rows: int,
    dp: int,
    name: str,
) -> tuple:
    """The two loads of ``routed_normed_token_major`` from a ``qsplit`` routed result (``ROUTED_LAYOUTS``), into the
    natural token order: the replica's rows [128, B * 28] (scalar offset r, one DMA over the two halves) and this
    rank's shard [B, 4, 28] (scalar offset 8 t dp + r of ``_shard_offset``, one DMA for each half)."""
    half = rows // 2
    half_width = half * ROUTED_COLS
    halves = _routed_halves(routed_dst, dp, rows)
    replica_rows = nl.ndarray((ROUTED_PARTITIONS, rows * ROUTED_COLS), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=replica_rows.reshape((ROUTED_PARTITIONS, 2, half_width)),
        src=halves.ap(
            pattern=[[2 * dp * half_width, ROUTED_PARTITIONS], [dp * half_width, 2], [1, half_width]],
            offset=0,
            scalar_offset=replica_offset,
            indirect_dim=0,
        ),
        name=f"{name}_routed_replica_load",
        **_HBM_DMA,
    )
    own = nl.ndarray((rows, _SHARD_PARTITIONS, ROUTED_COLS), dtype=nl.bfloat16, buffer=nl.sbuf)
    for h in range(2):
        nisa.dma_copy(
            dst=own[h * half : (h + 1) * half, :, :],
            src=halves.ap(
                pattern=[[ROUTED_COLS, half], [2 * dp * half_width, _SHARD_PARTITIONS], [1, ROUTED_COLS]],
                offset=h * dp * half_width,
                scalar_offset=shard_offset,
                indirect_dim=0,
            ),
            name=f"{name}_routed_own_load_h{h}",
            **_HBM_DMA,
        )
    return replica_rows, own


def _shard_offset(rank_info: nl.NkiTensor, dp: int, name: str, routed_layout: str = "natural") -> nl.NkiTensor:
    """Return the INT32 [1, 1] SBUF block index 4 t dp + r of this rank (TP rank t, DP rank r): the runtime offset of
    the token-major shard load of ``routed_normed_token_major``, in blocks of B * 28 columns of the routed result.
    ``qsplit``: 8 t dp + r, in blocks of B / 2 * 28 columns (``_routed_halves``)."""
    ranks = nl.ndarray((1, 4), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=ranks,
        src=rank_info.ap(pattern=[[0, 1], [1, 4]], offset=0),
        priority=_CRITICAL_DMA_PRIORITY,
        name=f"{name}_shard_ranks",
    )
    ranks_f32 = nl.ndarray((1, 4), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=ranks_f32, src=ranks)
    # Small integers: the FP32 products and sums are exact.
    first = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    blocks_per_partition = 2 * dp if routed_layout == "qsplit" else dp
    nisa.tensor_scalar(
        dst=first, data=ranks_f32[:, 2:3], op0=nl.multiply, operand0=float(_SHARD_PARTITIONS * blocks_per_partition)
    )
    block = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=block, data1=first, data2=ranks_f32[:, 1:2], op=nl.add)
    offset = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=offset, src=block)
    return offset


def normed_to_rows(normed: nl.NkiTensor, dst: nl.NkiTensor):
    """Write the [28, B, 4] normed shard of ``routed_normed_ar`` token-major into the [B, 112] BF16 view ``dst``: one PE
    transpose of each partition block q."""
    rows = normed.shape[1]
    normed_psum = _psum_rows(rows, DP_LATENT, nl.bfloat16)
    for block_idx in range(_SHARD_PARTITIONS):
        nisa.nc_transpose(
            dst=normed_psum[:, block_idx * ROUTED_COLS : (block_idx + 1) * ROUTED_COLS],
            data=normed[:, :, block_idx],
        )
    nisa.tensor_copy(dst=dst, src=normed_psum)


def load_column_norm_weight(moe_vectors_dp: nl.NkiTensor, drop_selection: bool, name: str) -> nl.NkiTensor:
    """Load the 3,584 latent norm weights of ``moe_vectors_dp`` (``column_norm_cols``) in the routed layout of the
    ``column`` out projection: [128, 28] FP32, element (p, c) is the weight of latent column 28 p + c."""
    norm_weight = nl.ndarray((ROUTED_PARTITIONS, ROUTED_COLS), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=norm_weight,
        src=moe_vectors_dp.ap(
            pattern=[[ROUTED_COLS, ROUTED_PARTITIONS], [1, ROUTED_COLS]], offset=column_norm_cols(drop_selection)[0]
        ),
        priority=_CRITICAL_DMA_PRIORITY,
        name=f"{name}_norm_weight",
    )
    return norm_weight


def store_shared_activation(hbm: dict, act_tok: nl.NkiTensor, rows: int, name: str):
    """Store this rank's shared-expert activation (columns 112 .. 303 of the [B, 304] BF16 out-projection input
    ``act_tok``) into ``shared_src`` [4, B * 48] of the ``column`` shared all-gather (row q holds activation columns
    48 q .. 48 q + 47 of each row b), and return the store instruction.

    The compiler can give the bytes of ``act_tok`` to a later tile as soon as the store is issued, so the caller makes
    the first rows load of the expert calls wait for the store (``_load_chunk_rows`` ``after``): every expert-loop
    instruction that waits for the gathered rows then also follows the end of the store."""
    return nisa.dma_copy(
        dst=hbm["shared_src"].ap(
            pattern=[
                [SHARED_GATHER_COLS, rows],
                [rows * SHARED_GATHER_COLS, _SHARD_PARTITIONS],
                [1, SHARED_GATHER_COLS],
            ],
            offset=0,
        ),
        src=act_tok.ap(
            pattern=[[DP_OUT_PROJ_ROWS, rows], [SHARED_GATHER_COLS, _SHARD_PARTITIONS], [1, SHARED_GATHER_COLS]],
            offset=DP_SHARED_DOWN_ROWS[0],
        ),
        name=f"{name}_shared_store",
        **_HBM_DMA,
    )


def column_out_projection(
    hbm: dict,
    row_select: nl.NkiTensor,
    norm_weight: nl.NkiTensor,
    moe_out_proj_dp: nl.NkiTensor,
    rows: int,
    dp: int,
    world_group: ReplicaGroup,
    tp_group: ReplicaGroup,
    name: str,
    routed_reduce: str = "ar",
    glue_hooks: dict = None,
) -> nl.NkiTensor:
    """Run the ``column`` out projection of one layer after its expert calls (``OUT_PROJ_FORMS``) and return this
    replica's MoE output [128, B * 56] BF16 SBUF in the layout of the ``reduced`` all-reduce result.

    The shared-expert activation of this rank (stored by ``store_shared_activation`` before the expert calls) goes to
    the replica in the shared all-gather, and its 48 K-tiles start the FP32 accumulation, so they run in the flight of
    the routed reduction. The routed all-reduce result (``routed_reduce`` ``ar``), or the routed TP32 all-gather of
    ``rs_ag`` (``routed_full_norm_rs_ag``), gives the
    full rows of the replica: the norm of all 3,584 columns (``_latent_norm_full``) feeds the 28 latent K-tiles. The
    [B, 224] BF16 result goes to ``out_src`` as 4 rows of B * 56 (row q holds hidden columns 128 h + 4 t + q), and the
    TP32 output all-gather stacks the rows of the 32 ranks into the [128, B * 56] layout of the all-reduce result.

    Args:
        hbm: the HBM tensors of ``moe_2srv_collective_hbm`` with ``out_proj_form`` ``column`` and ``routed_reduce``.
        row_select: FP32 [128, dp] replica one-hot (``tkg_moe.rank_selectors``); None with ``rs_ag``.
        norm_weight: [128, 28] FP32 SBUF latent norm weights (``load_column_norm_weight``).
        moe_out_proj_dp: [304, 7168] BF16 HBM in the column layout (``COLUMN_K_TILES``).
        rows, dp: the rows of each replica and the replicas.
        world_group, tp_group: the world and the TP32 groups.
        name: Instruction name prefix of the layer.
        routed_reduce: ``ar`` (default) or ``rs_ag`` (``ROUTED_REDUCES``, without ``stats_gather``).
        glue_hooks: None (default), or the ``glue_hooks`` of ``latent_moe_decode_2srv``: ``routed_load`` after the
            load of the routed all-reduce result or of the ``rs_ag`` rows, ``out_trigger`` after the trigger of the
            output all-gather, ``routed_trigger`` after the trigger of the routed all-reduce or reduce-scatter. The
            values do not change.
    """
    kernel_assert(
        routed_reduce in ROUTED_REDUCES, f"routed_reduce must be one of {ROUTED_REDUCES}, got {routed_reduce!r}"
    )
    glue_hooks = glue_hooks or {}
    ncc.all_gather(
        srcs=[hbm["shared_src"]],
        dsts=[hbm["shared_dst"]],
        replica_group=tp_group,
        collective_dim=0,
        name=f"{name}_shared_gather",
    )
    weights = nl.ndarray((_PMAX, COLUMN_K_TILES, COLUMN_COLS), dtype=nl.bfloat16, buffer=nl.sbuf)
    for first, last in _COLUMN_WEIGHT_CHUNKS:
        nisa.dma_copy(
            dst=weights[:, first:last, :],
            src=moe_out_proj_dp.ap(
                pattern=[[COLUMN_K_TILES * COLUMN_COLS, _PMAX], [COLUMN_COLS, last - first], [1, COLUMN_COLS]],
                offset=first * COLUMN_COLS,
            ),
            priority=_CRITICAL_DMA_PRIORITY,
            name=f"{name}_out_proj_k{first}",
        )
    shared = nl.ndarray((_PMAX, rows, SHARED_GATHER_COLS), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=shared.reshape((_PMAX, rows * SHARED_GATHER_COLS)),
        src=hbm["shared_dst"],
        name=f"{name}_shared_load",
        **_HBM_DMA,
    )
    out_psum = _psum_rows(rows, COLUMN_COLS)
    for slot in range(SHARED_GATHER_COLS):
        nisa.nc_matmul(
            dst=out_psum,
            stationary=shared[:, :, slot],
            moving=weights[:, COLUMN_LATENT_TILES + slot, :],
            accumulate=slot > 0,
        )
    after_load = glue_hooks.get("routed_load")
    after_trigger = glue_hooks.get("routed_trigger")
    if routed_reduce == "ar":
        normed = routed_full_norm_ar(
            hbm, row_select, norm_weight, rows, dp, world_group, name, after_load, after_trigger=after_trigger
        )
    else:
        normed = routed_full_norm_rs_ag(
            hbm, norm_weight, rows, world_group, tp_group, name, after_load, after_trigger=after_trigger
        )
    for k_tile in range(COLUMN_LATENT_TILES):
        nisa.nc_matmul(dst=out_psum, stationary=normed[:, :, k_tile], moving=weights[:, k_tile, :], accumulate=True)
    out_rows = nl.ndarray((rows, COLUMN_COLS), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=out_rows, src=out_psum, engine=nisa.engine.vector)
    nisa.dma_copy(
        dst=hbm["out_src"].ap(
            pattern=[[_H_TILES, rows], [rows * _H_TILES, _SHARD_PARTITIONS], [1, _H_TILES]], offset=0
        ),
        src=out_rows.reshape((rows, _SHARD_PARTITIONS, _H_TILES)),
        name=f"{name}_out_store",
        **_HBM_DMA,
    )
    out_gather = release_weight_gate(
        ncc.all_gather(
            srcs=[hbm["out_src"]],
            dsts=[hbm["out_dst"]],
            replica_group=tp_group,
            collective_dim=0,
            name=f"{name}_out_gather",
        )
    )
    if "out_trigger" in glue_hooks:
        glue_hooks["out_trigger"](out_gather)
    out = nl.ndarray((_PMAX, rows * _H_TILES), dtype=nl.bfloat16, buffer=nl.sbuf)
    # Record the load of the gate result, as _out_all_reduce of the reduced form does (record_consumer). The next
    # body's loads on the same queue then take a schedule edge on this load (wait_for_completion), not a runtime edge
    # on the all-gather. neuronx-cc 2.28 dep_reduction counts a runtime edge on a collective twice and stops ("DepGraph:
    # Num descendents already zero").
    release_gate_consumer(
        nisa.dma_copy(dst=out, src=hbm["out_dst"], name=f"{name}_out_load", **_HBM_DMA), engine=_HBM_DMA["engine"]
    )
    return out


def routed_full_norm_ar(
    hbm: dict,
    row_select: nl.NkiTensor,
    norm_weight: nl.NkiTensor,
    rows: int,
    dp: int,
    world_group: ReplicaGroup,
    name: str,
    after_load=None,
    after_trigger=None,
) -> nl.NkiTensor:
    """All-reduce the routed partials of ``hbm`` over ``world_group``, load the result and return the normed latent of
    this replica's rows in the routed layout: [128, B, 28] BF16, element (p, b, c) is column 28 p + c of row b
    (``_latent_norm_full``). The collectives and loads are those of ``routed_normed_ar``, and so are ``after_load``
    and ``after_trigger``."""
    tokens = hbm["routed_src"].shape[1] // ROUTED_COLS
    reduce = release_weight_gate(
        ncc.all_reduce(
            srcs=[hbm["routed_src"]],
            dsts=[hbm["routed_dst"]],
            replica_group=world_group,
            op=nl.add,
            name=f"{name}_routed_reduce",
        )
    )
    if after_trigger is not None:
        after_trigger(reduce)
    routed = nl.ndarray((ROUTED_PARTITIONS, tokens * ROUTED_COLS), dtype=nl.bfloat16, buffer=nl.sbuf)
    load = nisa.dma_copy(dst=routed, src=hbm["routed_dst"], name=f"{name}_routed_load", **_HBM_DMA)
    if after_load is not None:
        after_load(reduce, load)
    return _latent_norm_full(routed, row_select, norm_weight, rows, dp)


def routed_full_norm_rs_ag(
    hbm: dict,
    norm_weight: nl.NkiTensor,
    rows: int,
    world_group: ReplicaGroup,
    tp_group: ReplicaGroup,
    name: str,
    after_load=None,
    after_trigger=None,
) -> nl.NkiTensor:
    """Reduce the routed partials of ``hbm`` with ``routed_reduce`` ``rs_ag`` (no ``stats_gather``), load the
    replica's full routed rows and return their normed latent in the routed layout: [128, B, 28] BF16, element
    (p, b, c) is column 28 p + c of row b (``_norm_full_rows``).

    The world reduce-scatter gives this rank its block ``routed_rs_dst`` [4, B * 28] (rows 4 t .. 4 t + 3 of the
    replica's [128, B * 28] rows), and the TP32 all-gather of the blocks gives ``routed_ag_dst`` [128, B * 28]: the
    rows of this replica in the layout that ``_latent_norm_full`` selects from the all-reduce result, so no replica
    select runs. The collectives and the rows load are those of ``routed_norm_rs_ag``; the all-gather is the gate of
    the next body. ``after_load`` is that of ``_routed_gather_rows``; ``after_trigger`` is None, or a function that is
    called with the reduce-scatter right after it.
    """
    width = rows * ROUTED_COLS
    scatter = _routed_reduce_scatter(hbm, world_group, name)
    if after_trigger is not None:
        after_trigger(scatter)
    full_rows = _routed_gather_rows(hbm, tp_group, width, name, after_load)
    return _norm_full_rows(full_rows.reshape((ROUTED_PARTITIONS, rows, ROUTED_COLS)), norm_weight, rows)


def _latent_norm_full(
    routed: nl.NkiTensor, row_select: nl.NkiTensor, norm_weight: nl.NkiTensor, rows: int, dp: int
) -> nl.NkiTensor:
    """Apply the latent RMSNorm to every column of this replica's rows: [128, B, 28] BF16 (element (p, b, c) is column
    28 p + c of row b).

    Each element has the arithmetic of ``_latent_norm_rows``: the same replica select, square sums, ones matmul (here
    on all 128 output partitions), rsqrt and the two BF16 roundings ``bf16(bf16(r32 * rsqrt(sum(r^2) / 3584 + 1e-5)) *
    w32)``.

    Args:
        routed: [128, T * 28] BF16 SBUF, the full routed latent of every replica.
        row_select: FP32 [128, dp] replica one-hot (``tkg_moe.rank_selectors``).
        norm_weight: [128, 28] FP32: element (p, c) is the latent norm weight of column 28 p + c.
    """
    width = rows * ROUTED_COLS
    routed_by_replica = routed.reshape((ROUTED_PARTITIONS, dp, width))
    # The rows of this replica: a copy of replica 0, then a predicated copy of each other replica (no 0 x NaN).
    replica_rows = nl.ndarray((ROUTED_PARTITIONS, width), dtype=nl.bfloat16, buffer=nl.sbuf)
    is_replica = nl.ndarray((ROUTED_PARTITIONS, dp), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=is_replica, data=row_select, op0=nl.not_equal, operand0=0.0, engine=nisa.engine.vector)
    nisa.tensor_copy(dst=replica_rows, src=routed_by_replica[:, 0, :], engine=nisa.engine.vector)
    for replica_idx in range(1, dp):
        nisa.tensor_copy_predicated(
            dst=replica_rows,
            src=routed_by_replica[:, replica_idx, :],
            predicate=is_replica.ap(pattern=[[dp, ROUTED_PARTITIONS], [0, width]], offset=replica_idx),
        )
    return _norm_full_rows(replica_rows.reshape((ROUTED_PARTITIONS, rows, ROUTED_COLS)), norm_weight, rows)


def _norm_full_rows(rows_by_token: nl.NkiTensor, norm_weight: nl.NkiTensor, rows: int) -> nl.NkiTensor:
    """Apply the latent RMSNorm of ``_latent_norm_full`` to this replica's rows ``rows_by_token`` [128, B, 28] BF16
    (element (p, b, c) is column 28 p + c of row b) and return the normed [128, B, 28] BF16 rows."""
    routed_sq = nl.ndarray((ROUTED_PARTITIONS, rows, ROUTED_COLS), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=routed_sq, data1=rows_by_token, data2=rows_by_token, op=nl.multiply, engine=nisa.engine.vector
    )
    partial_sums = nl.ndarray((ROUTED_PARTITIONS, rows), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=partial_sums, op=nl.add, data=routed_sq, axis=(2,))
    ones = nl.ndarray((ROUTED_PARTITIONS, ROUTED_PARTITIONS), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones, value=1.0, engine=nisa.engine.vector)
    total_psum = _psum_bank(ROUTED_PARTITIONS, (rows,), nl.float32)
    nisa.nc_matmul(dst=total_psum, stationary=ones, moving=partial_sums, is_stationary_onezero=True)
    inv_rms = nl.ndarray((ROUTED_PARTITIONS, rows), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(
        dst=inv_rms,
        op=nl.rsqrt,
        data=total_psum,
        scale=1.0 / LATENT_SIZE,
        bias=LATENT_NORM_EPS,
        engine=nisa.engine.scalar,
    )
    scaled = nl.ndarray((ROUTED_PARTITIONS, rows, ROUTED_COLS), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=scaled,
        data1=rows_by_token,
        data2=inv_rms.ap(pattern=[[rows, ROUTED_PARTITIONS], [1, rows], [0, ROUTED_COLS]], offset=0),
        op=nl.multiply,
        engine=nisa.engine.vector,
    )
    normed = nl.ndarray((ROUTED_PARTITIONS, rows, ROUTED_COLS), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=normed,
        data1=scaled,
        data2=norm_weight.ap(pattern=[[ROUTED_COLS, ROUTED_PARTITIONS], [0, rows], [1, ROUTED_COLS]], offset=0),
        op=nl.multiply,
        engine=nisa.engine.vector,
    )
    return normed


def _gather_router_rows(
    gather_src: nl.NkiTensor,
    hbm: dict,
    gather: str,
    dp_group,
    tp_group,
    world_group,
    name: str,
    gathers_out=None,
    dma: dict = None,
):
    """Store this rank's [B, 112] router rows and gather the rows of every rank in HBM (``MOE_GATHERS``). Return the
    store instruction. ``gathers_out``: a list that receives the gather collectives in trace order (None: nothing).
    ``dma``: the queue of the store (``ROWS_DMAS``; None: the SP queue).

    ``hbm["first_rows_dst"]`` (opt-in) is a list of one tensor of the shape of ``rows_dst``: the first DP gather that
    sees it takes the tensor out of the list and writes it in place of ``rows_dst``, so the first DP gather of a call
    has a destination that no later gather of the call writes.
    """
    store = nisa.dma_copy(dst=hbm["rows_src"], src=gather_src, name=f"{name}_rows_store", **(dma or _HBM_DMA))
    gathers = []
    if gather == "world":
        gathers.append(
            ncc.all_gather(
                srcs=[hbm["rows_src"]],
                dsts=[hbm["world_dst"]],
                replica_group=world_group,
                collective_dim=0,
                name=f"{name}_rows_world_gather",
            )
        )
    else:
        first = hbm.get("first_rows_dst")
        rows_dst = first.pop() if first else hbm["rows_dst"]
        gathers.append(
            ncc.all_gather(
                srcs=[hbm["rows_src"]],
                dsts=[rows_dst],
                replica_group=dp_group,
                collective_dim=0,
                name=f"{name}_rows_gather",
            )
        )
        gathers.append(
            ncc.all_gather(
                srcs=[rows_dst],
                dsts=[hbm["shard_dst"]],
                replica_group=tp_group,
                collective_dim=0,
                name=f"{name}_shard_gather",
            )
        )
    if gathers_out is not None:
        gathers_out.extend(gathers)
    return store


@contextlib.contextmanager
def _stream_start_after(gather, expert_weights: tuple, hbm: dict, rows_wait: bool = False, waits: list = None):
    """Order the stream start of the expert calls after the router-row ``gather`` (``weights_after_gather``).

    Inside the scope, each ``nisa.dma_copy`` that reads a static view of one of ``expert_weights`` (packed weights or
    scales) and writes a destination view that no earlier load of the scope wrote (``tkg_weight_dma_gate.gate_all_dmas`` keys)
    depends on ``gather``, a schedule edge: the DMA prefetch pass of neuronx-cc skips a DMA that depends on a
    collective. A later load into one of these views waits for the readers of the earlier load, which wait for the
    gathered rows anyway.

    The edges run from ``gather`` (traced before the scope) to loads traced inside it, so a load that the expert loop
    creates in a nested block can take one. No edge is a semaphore wait.

    With ``rows_wait`` (``weights_after_rows``), each of these loads also waits (a semaphore wait) for the loads of the
    TP32 gather's result (``hbm["shard_dst"]``, the router rows) that the scope traced before it, so the stream starts
    after the rows are in, not in the flight of the row gathers. A load traced before any rows load, or after the scope
    entered a loop or a branch (``nl.fori_loop``, ``nexp.if_else``: the later slot tiles, a dynamic expert loop), keeps
    only the schedule edge, so no semaphore wait reaches into a loop or branch body. ``waits`` (a list, for a test)
    gets, for each stream-start load, the number of rows loads it waits for.
    """
    original = nisa.dma_copy
    gate_up, gate_up_scale, down, down_scale = expert_weights
    by_kind = {"weight": (gate_up, down), "scale": (gate_up_scale, down_scale)}
    storages = {
        kind: {id(t._storage): t._storage for t in tensors if getattr(t, "_storage", None) is not None}
        for kind, tensors in by_kind.items()
    }
    written = {}
    gather_instr = gather if hasattr(gather, "depends_on") else nisa.NkiInstruction(gather)
    shard = hbm.get("shard_dst") if rows_wait else None
    rows_storage = {id(shard._storage): shard._storage} if getattr(shard, "_storage", None) is not None else {}
    rows_loads = []
    dynamic = []  # rows_wait: non-empty once the scope traced a loop or a branch

    def _entering(control):
        @functools.wraps(control)
        def _control(*args, **kwargs):
            dynamic.append(control)
            return control(*args, **kwargs)

        return _control

    def _ordered_dma_copy(*args, **kwargs):
        instr = original(*args, **kwargs)
        src = kwargs.get("src", args[1] if len(args) > 1 else None)
        dst = kwargs.get("dst", args[0] if args else None)
        if instr is None or src is None or dst is None:
            return instr
        if rows_storage and _static_view_of(src, rows_storage):
            rows_loads.append(instr if hasattr(instr, "depends_on") else nisa.NkiInstruction(instr))
            return instr
        if not (_static_view_of(src, storages["weight"]) or _static_view_of(src, storages["scale"])):
            return instr
        key = _view_key(dst)
        if key is None or key in written:
            return instr
        written[key] = dst._storage
        instr = instr if hasattr(instr, "depends_on") else nisa.NkiInstruction(instr)
        instr.depends_on(gather_instr, kind=nisa.dependency.schedule)
        waited = rows_loads if not dynamic else []
        if waited:
            instr.depends_on(*waited, kind=nisa.dependency.runtime)
        if waits is not None:
            waits.append(len(waited))
        return instr

    controls = (nl.fori_loop, nexp.if_else)
    nisa.dma_copy = _ordered_dma_copy
    if rows_wait:
        nl.fori_loop, nexp.if_else = (_entering(control) for control in controls)
    try:
        yield
    finally:
        nisa.dma_copy = original
        nl.fori_loop, nexp.if_else = controls


def _mx_quantize_latent(latent_src: nl.NkiTensor, rows: int, name: str) -> tuple[nl.NkiTensor, nl.NkiTensor]:
    """Quantize this rank's ``[rows, 128]`` FP32 latent block to MXFP8 as ``moe_tkg_decode`` quantizes the whole latent:
    round the PSUM to BF16, move column ``4 i + q`` of each token to partition ``i``, element ``q`` with 4 TensorE
    transposes of the stride-4 column views, and quantize 32 partitions x 4 elements per token (4 whole MX blocks).

    Returns:
        codes ``[32, rows]`` ``float8_e4m3fn_x4`` (partition i, token b: the 4 codes of latent columns 4 i .. 4 i + 3 of
        the block), scales ``[32, rows]`` UINT8 (partitions 0..3: the 4 block scales of each token).
    """
    block = nl.ndarray((rows, MX_LATENT_COLS), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=block, src=latent_src, engine=nisa.engine.vector)
    swizzle = nl.ndarray((_MX_QUADRANT_PARTITIONS, _MX_Q, rows), dtype=nl.bfloat16, buffer=nl.psum)
    for q in range(_MX_Q):
        nisa.nc_transpose(
            dst=swizzle[:, q, :],
            data=block.ap(pattern=[[MX_LATENT_COLS, rows], [_MX_Q, _MX_QUADRANT_PARTITIONS]], offset=q),
            engine=nisa.engine.tensor,
        )
    quant_in = nl.ndarray((_MX_QUADRANT_PARTITIONS, rows, _MX_Q), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(
        dst=quant_in,
        src=swizzle.ap(pattern=[[_MX_Q * rows, _MX_QUADRANT_PARTITIONS], [1, rows], [rows, _MX_Q]], offset=0),
        engine=nisa.engine.vector,
    )
    codes = nl.ndarray((_MX_QUADRANT_PARTITIONS, rows), dtype=nl.float8_e4m3fn_x4, buffer=nl.sbuf)
    scales = nl.ndarray((_MX_QUADRANT_PARTITIONS, rows), dtype=nl.uint8, buffer=nl.sbuf)
    nisa.quantize_mx(src=quant_in, dst=codes, dst_scale=scales)
    return codes, scales


def _mx_score_rows(
    logits, bias_sb: nl.NkiTensor, pad_rows: nl.NkiTensor, name: str, score_slot: int = None
) -> nl.NkiTensor:
    """Return the FP32 score columns of this rank's ``latent_gather_mx`` router rows: ``[R, 28]`` sigmoid scores of the
    router ``logits`` ``[R, 28]``, then (with ``bias_sb``, the bias ``[R, 28]``; None with ``drop_selection``) the 28
    selection values ``scores + bias``. With ``pad_rows`` the scores of the padding rows become 0 after the selection
    values (``zero_pad_row_scores``): the instructions and the order of the BF16 router rows, so the score words of
    both row forms are equal and a padding row routes to no expert. ``score_slot`` (``mx_scores_align``, without
    ``bias_sb``): ``[R, score_slot]``, the 28 scores then zero words."""
    rows = logits.shape[0]
    score_cols = DP_EXPERTS if bias_sb is None else 2 * DP_EXPERTS
    kernel_assert(
        score_slot is None or bias_sb is None,
        "the padded score run (mx_scores_align) holds the scores of the drop_selection rows only",
    )
    if score_slot is not None:
        score_cols = score_slot
    score_sel = nl.ndarray((rows, score_cols), dtype=nl.float32, buffer=nl.sbuf)
    if score_slot is not None:
        # The pad words of the score run: zero, so no stale SBUF byte moves through the gathers.
        nisa.memset(dst=score_sel[:, DP_EXPERTS:score_slot], value=0.0)
    scores = score_sel[:, 0:DP_EXPERTS]
    nisa.activation(dst=scores, op=nl.sigmoid, data=logits)
    if bias_sb is not None:
        nisa.tensor_tensor(dst=score_sel[:, DP_EXPERTS : 2 * DP_EXPERTS], data1=scores, data2=bias_sb, op=nl.add)
    if pad_rows is not None:
        zero_pad_row_scores(scores, pad_row_flags(pad_rows, name))
    return score_sel


def _mx_store_rows(
    codes, scales, score_sel, rows_src: nl.NkiTensor, name: str, dma: dict = None, base: int = 0, stream: bool = False
):
    """Store this rank's ``latent_gather_mx`` router rows into ``rows_src`` at the word offset ``base`` (0 for the
    rank's own ``[B, cols]`` tensor): the codes partition-major (word ``w B + b`` of latent columns
    ``128 t + 4 w .. + 3`` of token b), then the 4 scale bytes (byte ``k B + b``), then the token-major scores and
    selection values (word ``33 B + b S + c``). Return the last store (the scores store).

    With ``stream`` (``mx_rows_stream``, ``drop_selection`` scores) the rows are token-major, ``_MX_STREAM_COLS`` words
    a token: its 32 code words, its 4 scale bytes, its 28 scores and 3 zero words, in one store. Vector Engine
    transposes (bit-accurate, any dtype) move the codes and the scale bytes to token rows. Return the store."""
    dma = dma or _HBM_DMA
    b = codes.shape[1]
    s = score_sel.shape[1]
    if stream:
        kernel_assert(
            s == DP_EXPERTS and b <= _VECTOR_TRANSPOSE_MAX,
            f"mx_rows_stream rows hold the 28 drop_selection scores of at most {_VECTOR_TRANSPOSE_MAX} rows, got {s} "
            f"score words and {b} rows",
        )
        row = nl.ndarray((b, _MX_STREAM_COLS), dtype=nl.uint32, buffer=nl.sbuf)
        nisa.nc_transpose(dst=row[:, 0:_MX_CODE_WORDS], data=codes.view(nl.uint32), engine=nisa.engine.vector)
        nisa.nc_transpose(
            dst=row.view(nl.uint8)[:, 4 * _MX_CODE_WORDS : 4 * _MX_HEAD_WORDS],
            data=scales[0:_MX_SCALES_PER_QUADRANT, :],
            engine=nisa.engine.vector,
        )
        nisa.tensor_copy(
            dst=row[:, _MX_HEAD_WORDS : _MX_HEAD_WORDS + DP_EXPERTS],
            src=score_sel.view(nl.uint32),
            engine=nisa.engine.vector,
        )
        nisa.memset(dst=row[:, _MX_HEAD_WORDS + DP_EXPERTS : _MX_STREAM_COLS], value=0, engine=nisa.engine.vector)
        return nisa.dma_copy(
            dst=rows_src.ap(pattern=[[_MX_STREAM_COLS, b], [1, _MX_STREAM_COLS]], offset=base),
            src=row,
            name=f"{name}_mx_rows_store",
            **dma,
        )
    nisa.dma_copy(
        dst=rows_src.ap(pattern=[[b, _MX_CODE_WORDS], [1, b]], offset=base),
        src=codes.view(nl.uint32),
        name=f"{name}_mx_codes_store",
        **dma,
    )
    nisa.dma_copy(
        dst=rows_src.view(nl.uint8).ap(
            pattern=[[b, _MX_SCALES_PER_QUADRANT], [1, b]], offset=(base + _MX_CODE_WORDS * b) * 4
        ),
        src=scales[0:_MX_SCALES_PER_QUADRANT, :],
        name=f"{name}_mx_scale_store",
        **dma,
    )
    return nisa.dma_copy(
        dst=rows_src.ap(pattern=[[s, b], [1, s]], offset=base + _MX_HEAD_WORDS * b),
        src=score_sel.view(nl.uint32),
        name=f"{name}_mx_scores_store",
        **dma,
    )


def _mx_gather_router_rows(
    codes,
    scales,
    score_sel,
    hbm: dict,
    dp_group,
    tp_group,
    name: str,
    dma: dict = None,
    stream: bool = False,
    gathers_out=None,
):
    """Store this rank's ``latent_gather_mx`` router rows (``_mx_store_rows``) and gather them over the DP group and the
    TP32 group. Return the last store (the scores store). The gathers move the contiguous rank block, so the layout of
    the receiver (``_load_chunk_mx``) follows. ``gathers_out``: a list that receives the two gathers (None: nothing)."""
    rows_src = hbm["rows_src"]
    store = _mx_store_rows(codes, scales, score_sel, rows_src, name, dma, stream=stream)
    first = hbm.get("first_rows_dst")
    rows_dst = first.pop() if first else hbm["rows_dst"]
    gathers = [
        ncc.all_gather(
            srcs=[rows_src], dsts=[rows_dst], replica_group=dp_group, collective_dim=0, name=f"{name}_rows_gather"
        ),
        ncc.all_gather(
            srcs=[rows_dst],
            dsts=[hbm["shard_dst"]],
            replica_group=tp_group,
            collective_dim=0,
            name=f"{name}_shard_gather",
        ),
    ]
    if gathers_out is not None:
        gathers_out.extend(gathers)
    return store


def _load_pair_rows(hbm: dict, start: int, name: str, after=None) -> nl.NkiTensor:
    """Load the gathered ``fp8`` ``drop_selection`` router rows of tokens ``[start, start + 256)`` (``rows_pairs``) in
    one DMA: [128, 32, ``_PAIR_SLOT``] FP32 SBUF, the slot (p, t) holds the 58-word rows of tokens ``start + 2 p`` and
    ``start + 2 p + 1`` of TP block t (``_pair_row_words``). The two rows are adjacent in ``shard_dst``, so each DMA
    element is 464 B: 4,096 elements in place of the 8,192 elements of 232 B of two one-token-a-partition loads.
    ``after``: an instruction that the load waits for (``_after``)."""
    width = SCORE_ROW_COLS - _FP8_SHIFT
    tokens = hbm["rows_dst"].shape[0]
    kernel_assert(
        tuple(hbm["shard_dst"].shape) == (TP_SIZE * tokens, width),
        f"rows_pairs loads {width}-word rows: shard_dst is {tuple(hbm['shard_dst'].shape)}",
    )
    kernel_assert(
        start % 2 == 0 and 0 <= start and start + 2 * _PMAX <= tokens,
        f"rows_pairs loads tokens [{start}, {start + 2 * _PMAX}) of {tokens}",
    )
    gathered = nl.ndarray((_PMAX, TP_SIZE, _PAIR_SLOT), dtype=nl.float32, buffer=nl.sbuf)
    load = nisa.dma_copy(
        dst=gathered[:, :, 0 : 2 * width],
        src=hbm["shard_dst"].ap(
            pattern=[[2 * width, _PMAX], [tokens * width, TP_SIZE], [1, 2 * width]], offset=start * width
        ),
        name=f"{name}_rows_load",
        **_HBM_DMA,
    )
    _after(load, after)
    return gathered


def _load_chunk_rows(
    hbm: dict,
    gather: str,
    start: int,
    count: int,
    rows: int,
    name: str,
    router_bias: nl.NkiTensor = None,
    rows_latent: str = "bf16",
    rows_load_split: int = 1,
    on_part=None,
    after=None,
    add_bias: bool = True,
    row_chunks: int = 1,
) -> nl.NkiTensor:
    """Load the gathered router rows of tokens ``[start, start + count)`` as [count, 32, 112] FP32 SBUF.

    Row i, block t is the gather row of rank 32 r + t for its row b, token ``start + i = r B + b``. The ``dp_tp``
    destination holds element (t, token) at row ``t T + token``: one DMA. The ``world`` destination holds it at row
    ``(32 r + t) B + b``: one DMA for each replica of the chunk. With ``router_bias`` (``drop_selection``) the rows
    have 84 words: they fill words 0..83 of each block, and ``_add_router_bias`` writes the selection words (later, by
    the caller, with ``add_bias`` False). With ``rows_latent`` ``fp8`` a row fills words 26.. of each block
    (``_FP8_SHIFT``), so the scores keep their words. ``after``: an instruction that each load waits for (a runtime
    dependency, a semaphore wait). ``row_chunks`` (``dp_tp`` only): the load as that many DMAs of 32 / ``row_chunks``
    TP blocks each, in block order (``head_row_chunks``): the same elements.
    With ``rows_load_split`` > 1 (``dp_tp``) the load is that many DMAs of 32 / split TP blocks each, on the SP and the
    ACT hardware queues in turn, and ``on_part(gathered, t0, t1)`` runs after the DMA of blocks ``t0 .. t1 - 1``.
    """
    gathered = nl.ndarray((count, TP_SIZE, _GATHER_COLS), dtype=nl.float32, buffer=nl.sbuf)
    width = router_row_cols(router_bias is not None, rows_latent)
    shift = _row_shift(rows_latent)
    loaded = gathered if router_bias is None and not shift else gathered[:, :, shift : shift + width]
    kernel_assert(
        row_chunks == 1 or (gather == "dp_tp" and TP_SIZE % row_chunks == 0),
        f"the router rows load in chunks of whole TP blocks of the dp_tp gather, got {row_chunks} chunks, {gather!r}",
    )
    if gather == "dp_tp" and row_chunks > 1:
        tokens = hbm["rows_dst"].shape[0]
        blocks = TP_SIZE // row_chunks
        for chunk in range(row_chunks):
            first = chunk * blocks
            load = nisa.dma_copy(
                dst=loaded[:, first : first + blocks, :],
                src=hbm["shard_dst"].ap(
                    pattern=[[width, count], [tokens * width, blocks], [1, width]],
                    offset=(start + first * tokens) * width,
                ),
                name=f"{name}_rows_load_{chunk}",
                **_HBM_DMA,
            )
            _after(load, after)
    elif gather == "dp_tp" and rows_load_split > 1:
        tokens = hbm["rows_dst"].shape[0]
        blocks = TP_SIZE // rows_load_split
        for part in range(rows_load_split):
            t0 = part * blocks
            load = nisa.dma_copy(
                dst=loaded[:, t0 : t0 + blocks, :],
                src=hbm["shard_dst"].ap(
                    pattern=[[width, count], [tokens * width, blocks], [1, width]],
                    offset=(t0 * tokens + start) * width,
                ),
                name=f"{name}_rows_load_p{part}",
                **(_HBM_DMA if part % 2 == 0 else _HBM_DMA_ACT),
            )
            _after(load, after)
            if on_part is not None:
                on_part(gathered, t0, t0 + blocks)
    elif gather == "dp_tp":
        tokens = hbm["rows_dst"].shape[0]
        load = nisa.dma_copy(
            dst=loaded,
            src=hbm["shard_dst"].ap(
                pattern=[[width, count], [tokens * width, TP_SIZE], [1, width]],
                offset=start * width,
            ),
            name=f"{name}_rows_load",
            **_HBM_DMA,
        )
        _after(load, after)
    else:
        for replica_idx in range(start // rows, (start + count - 1) // rows + 1):
            first = max(start, replica_idx * rows)
            last = min(start + count, (replica_idx + 1) * rows)
            load = nisa.dma_copy(
                dst=loaded[first - start : last - start, :, :],
                src=hbm["world_dst"].ap(
                    pattern=[[width, last - first], [rows * width, TP_SIZE], [1, width]],
                    offset=(TP_SIZE * replica_idx * rows + first - replica_idx * rows) * width,
                ),
                name=f"{name}_rows_load_r{replica_idx}",
                **_HBM_DMA,
            )
            _after(load, after)
    if router_bias is not None and add_bias:
        _add_router_bias(gathered, router_bias)
    return gathered


def _after(instruction, after):
    """Make ``instruction`` wait for ``after`` (a runtime dependency, a semaphore wait); nothing when ``after`` is None."""
    if after is None:
        return
    instruction = instruction if hasattr(instruction, "depends_on") else nisa.NkiInstruction(instruction)
    instruction.depends_on(
        after if hasattr(after, "depends_on") else nisa.NkiInstruction(after), kind=nisa.dependency.runtime
    )


def _after_once(instruction, after, first):
    """Make the first of a series of DMAs on one queue wait for ``after`` (a runtime dependency, ``_after``), and each
    later DMA of the series follow that first DMA in the schedule (an order, no wait). Return the first DMA (None while
    ``after`` is None: then nothing changes).

    The queue runs its DMAs in order, so each DMA of the series still starts after ``after`` has completed. One runtime
    edge on a DMA producer in place of one for each DMA of the series: the dependency-reduction pass (dep_reduction) of
    neuronx-cc 2.28 counts runtime edges on a DMA or collective producer twice and stops (NCC_IXDR001, or "DepGraph:
    Num descendents already zero") when one producer has several of them.
    """
    if after is None:
        return None
    if first is None:
        _after(instruction, after)
        return instruction
    _scheduled_after(instruction, first)
    return first


def _scheduled_after(instruction, after):
    """Make ``instruction`` follow ``after`` in the schedule (a schedule edge: an order, no wait); nothing when
    ``after`` is None."""
    if after is None:
        return
    _instruction(instruction).depends_on(_instruction(after), kind=nisa.dependency.schedule)


def load_chunk_latent(
    hbm: dict,
    gather: str,
    start: int,
    count: int,
    rows: int,
    name: str,
    router_bias: nl.NkiTensor = None,
    rows_latent: str = "bf16",
    rows_load_split: int = 1,
    after=None,
    add_bias: bool = True,
) -> tuple[nl.NkiTensor, nl.NkiTensor]:
    """Load the gathered router rows of tokens ``[start, start + count)`` (``_load_chunk_rows``) and write their BF16
    latent: return (rows [count, 32, 112] FP32, latent [count, 32, 112] BF16) in SBUF. With ``rows_load_split`` > 1
    each part of the rows load has its own latent write (``latent_from_blocks``: the BF16 copy, or the FP8 expand), so
    it overlaps the DMAs of the next parts.
    ``after``: an instruction that the rows load waits for, ``add_bias`` False leaves the router bias add to the caller
    (``_load_chunk_rows``)."""
    if rows_load_split == 1:
        gathered = _load_chunk_rows(
            hbm, gather, start, count, rows, name, router_bias, rows_latent, after=after, add_bias=add_bias
        )
        latent_sb = nl.ndarray((count, TP_SIZE, DP_LATENT), dtype=nl.bfloat16, buffer=nl.sbuf)
        latent_from_rows(gathered, latent_sb, rows_latent)
        return gathered, latent_sb
    latent_sb = nl.ndarray((count, TP_SIZE, DP_LATENT), dtype=nl.bfloat16, buffer=nl.sbuf)

    def copy_part(rows_sb, t0, t1):
        latent_from_blocks(rows_sb, latent_sb, rows_latent, t0, t1)

    gathered = _load_chunk_rows(
        hbm,
        gather,
        start,
        count,
        rows,
        name,
        router_bias,
        rows_latent,
        rows_load_split,
        on_part=copy_part,
        after=after,
        add_bias=add_bias,
    )
    return gathered, latent_sb


def _load_chunk_mx(
    hbm: dict,
    start: int,
    count: int,
    rows: int,
    name: str,
    router_bias: nl.NkiTensor = None,
    after=None,
    load_order: str = "quadrant",
    scale_queue: str = "act",
    score_loads: list = None,
    tile0_loads: list = None,
    reload_order: str = "default",
    scale_fill_after: tuple = (),
) -> tuple[nl.NkiTensor, nl.NkiTensor, nl.NkiTensor]:
    """Load the ``latent_gather_mx`` gather of tokens ``[start, start + count)`` (whole replicas: ``start`` and
    ``count`` multiples of ``rows``). Return the gathered rows and the MXFP8 expert input.

    - gathered ``[count, 32, 112]`` FP32: the scores at ``_GATHER_SCORE_COLS`` and the selection values at
      ``_GATHER_SELECTION_COLS`` (for ``_local_affinities``; the latent words are not read). With ``router_bias``
      (``drop_selection``) ``_add_router_bias`` writes the selection words (``_load_mx_scores``).
    - codes ``[128, 7, count]`` ``float8_e4m3fn_x4`` and scales ``[128, 7, count]`` UINT8, straight in the layout of
      ``moe_routed._quantize_natural_latent`` (``_load_mx_codes``).
    - ``load_order`` ``tile`` (``MX_LOAD_ORDERS``): one code DMA and one scale DMA for each (tile h, quadrant Q), in the
      read order of ``moe_tkg_decode`` (h = 0 first, the 4 quadrants of a tile together). ``scale_queue`` ``sp``
      (``MX_SCALE_QUEUES``): the scale DMAs on the SP queue. The loaded bytes do not change (``_load_mx_codes``).
    - ``score_loads`` (a list; ``pe_keep_warm``), when given: the score DMAs issue first, the first code DMA follows the
      last of them in the schedule (the same SP queue, so the queue runs them first; with ``codes_ahead_scores_act`` the
      scores are on the Activation queue, so only the schedule orders them), and the list gets them in issue order.
      ``tile0_loads`` (a list, with ``load_order`` ``tile``) gets the tile-0 code and scale DMAs in issue order. These
      two lists do not change the loaded bytes; only ``codes_ahead_scores_act`` moves the score DMAs to another queue.
    - ``scale_fill_after``: instructions that the zero fill of the code scales follows in the schedule
      (``_load_mx_codes``); empty (default) adds no edge. The loaded bytes do not change.
    - ``reload_order`` (``MX_RELOAD_ORDERS``, keep-warm only): ``default`` loads the scores and adds the router bias
      before the code and scale DMAs, so the first code DMA follows the bias add in the trace. ``codes_ahead`` issues
      the code and scale DMAs before the bias add, so they do not follow it; the scores still issue first and the
      first code DMA still follows the last score load. ``codes_ahead_scores_act`` also puts the score DMAs on the
      Activation queue. The bias add runs on the same scores and biases, so the gathered words do not change.
    """
    kernel_assert(
        reload_order in MX_RELOAD_ORDERS, f"reload_order must be one of {MX_RELOAD_ORDERS}, got {reload_order!r}"
    )
    codes_ahead = reload_order != "default"
    if score_loads is not None:
        # default: the scores load and the router bias add before the code DMAs. codes_ahead: no bias add here, so the
        # code DMAs below do not follow it in the trace; the bias runs after them, on the same words.
        score_dma = _HBM_DMA_ACT if reload_order == "codes_ahead_scores_act" else None
        gathered = _load_mx_scores(
            hbm,
            start,
            count,
            rows,
            name,
            router_bias,
            score_loads=score_loads,
            **({"add_bias": False} if codes_ahead else {}),
            **({"dma": score_dma} if score_dma is not None else {}),
        )
        kernel_assert(
            score_loads, f"the latent_gather_mx score load of tokens [{start}, {start + count}) issued no DMA"
        )
    quant, scale = _load_mx_codes(
        hbm,
        start,
        count,
        rows,
        name,
        after=after,
        load_order=load_order,
        scale_queue=scale_queue,
        **({"tile0_loads": tile0_loads} if tile0_loads is not None else {}),
        **({"codes_after": score_loads[-1]} if score_loads is not None else {}),
        **({"fill_after": scale_fill_after} if scale_fill_after else {}),
    )
    if score_loads is None:
        gathered = _load_mx_scores(hbm, start, count, rows, name, router_bias)
    elif codes_ahead and router_bias is not None:
        # The deferred router bias add of codes_ahead, on the scores and biases of the load above.
        _add_router_bias(gathered, router_bias)
    return gathered, quant, scale


def _check_mx_replicas(start: int, count: int, rows: int):
    """Stop a ``latent_gather_mx`` load of tokens ``[start, start + count)`` that is not whole replicas of ``rows``."""
    kernel_assert(
        start % rows == 0 and count % rows == 0,
        f"latent_gather_mx loads whole replicas: start {start} and count {count} must be multiples of {rows} rows",
    )


def _load_mx_codes(
    hbm: dict,
    start: int,
    count: int,
    rows: int,
    name: str,
    after=None,
    load_order: str = "quadrant",
    scale_queue: str = "act",
    tile0_loads: list = None,
    codes_after=None,
    dst: tuple = None,
    fill_after: tuple = (),
) -> tuple[nl.NkiTensor, nl.NkiTensor]:
    """Load the MXFP8 expert input of the ``latent_gather_mx`` gather of tokens ``[start, start + count)`` (whole
    replicas). Return codes ``[128, 7, count]`` ``float8_e4m3fn_x4`` and scales ``[128, 7, count]`` UINT8, straight in
    the layout of ``moe_routed._quantize_natural_latent``: block t = rank ``4 h + Q`` of ``shard_dst`` (Q the
    partition quadrant, h the latent tile), 4 code DMAs (one per quadrant) into partitions ``[32 Q, 32 Q + 32)`` of a
    tile, and 4 scale DMAs into partitions ``32 Q + 0 .. 3`` of a zeroed tile. The tokens sit on the free axis, so
    ``count`` may exceed 128 (a token-tiled call). ``after``: an instruction that each code load waits for.
    ``tile0_loads`` (a list, ``load_order`` ``tile``): gets the tile-0 code and scale DMAs in issue order.
    ``codes_after``: an instruction that the first code DMA follows in the schedule (``_load_chunk_mx``). ``dst``:
    None allocates the two tiles; a tuple (codes, scales) of ``[128, 7, count]`` views loads into them
    (``routed_compact`` with ``latent_gather_mx``: the first ``count`` columns of its gather source). ``fill_after``:
    instructions that the GpSimd zero fill of the scales follows in the schedule (no wait); empty (default) adds none.
    """
    _check_mx_replicas(start, count, rows)
    shard = hbm["shard_dst"]
    cols = shard.shape[1]
    tokens = shard.shape[0] // TP_SIZE
    n_rep = count // rows
    r0 = start // rows
    n_tiles = LATENT_SIZE // _MX_TILE
    n_quadrants = _PMAX // _MX_QUADRANT_PARTITIONS
    quad = _MX_QUADRANT_PARTITIONS
    if dst is None:
        quant = nl.ndarray((_PMAX, n_tiles, count), dtype=nl.float8_e4m3fn_x4, buffer=nl.sbuf)
        scale = nl.ndarray((_PMAX, n_tiles, count), dtype=nl.uint8, buffer=nl.sbuf)
    else:
        quant, scale = dst
        kernel_assert(
            tuple(quant.shape) == (_PMAX, n_tiles, count) and tuple(scale.shape) == (_PMAX, n_tiles, count),
            f"the latent_gather_mx code destinations must be [{_PMAX}, {n_tiles}, {count}], got {tuple(quant.shape)}",
        )
    fill = nisa.memset(dst=scale, value=0, engine=nisa.engine.gpsimd)
    for earlier in fill_after:
        _instruction(fill).depends_on(_instruction(earlier), kind=nisa.dependency.schedule)
    shard_bytes = shard.view(nl.uint8)
    scale_dma = _HBM_DMA if scale_queue == "sp" else _HBM_DMA_ACT
    first_code = None  # the code DMA that waits for ``after`` (_after_once)
    kernel_assert(
        tile0_loads is None or load_order == "tile",
        f"tile0_loads lists the tile-0 DMAs of the tile load order, got load_order {load_order!r}",
    )
    if load_order == "tile":
        _load_chunk_mx_tiles(
            shard, quant, scale, tokens, cols, r0, n_rep, rows, name, after, scale_dma, tile0_loads, codes_after
        )
    else:
        for q in range(n_quadrants):
            code_load = nisa.dma_copy(
                dst=quant.view(nl.uint32)[q * quad : q * quad + quad].reshape((quad, n_tiles, n_rep, rows)),
                src=shard.ap(
                    pattern=[[rows, quad], [n_quadrants * tokens * cols, n_tiles], [rows * cols, n_rep], [1, rows]],
                    offset=q * tokens * cols + r0 * rows * cols,
                ),
                name=f"{name}_mx_codes_q{q}",
                **_HBM_DMA,
            )
            first_code = _after_once(code_load, after, first_code)
            if q == 0:
                _scheduled_after(code_load, codes_after)
            nisa.dma_copy(
                dst=scale[q * quad : q * quad + _MX_SCALES_PER_QUADRANT].reshape(
                    (_MX_SCALES_PER_QUADRANT, n_tiles, n_rep, rows)
                ),
                src=shard_bytes.ap(
                    pattern=[
                        [rows, _MX_SCALES_PER_QUADRANT],
                        [n_quadrants * tokens * cols * 4, n_tiles],
                        [rows * cols * 4, n_rep],
                        [1, rows],
                    ],
                    offset=(q * tokens * cols + r0 * rows * cols + _MX_CODE_WORDS * rows) * 4,
                ),
                name=f"{name}_mx_scale_q{q}",
                **scale_dma,
            )
    return quant, scale


def _load_mx_scores(
    hbm: dict,
    start: int,
    count: int,
    rows: int,
    name: str,
    router_bias: nl.NkiTensor = None,
    score_loads: list = None,
    add_bias: bool = True,
    dma: dict = None,
) -> nl.NkiTensor:
    """Load the scores of the ``latent_gather_mx`` gather of tokens ``[start, start + count)`` (whole replicas, at most
    128 tokens: one token a partition) as gathered ``[count, 32, 112]`` FP32, one DMA a replica: the scores at
    ``_GATHER_SCORE_COLS`` and the selection values at ``_GATHER_SELECTION_COLS`` (for ``_local_affinities``; the
    latent words are not read). With ``router_bias`` (``drop_selection``) ``_add_router_bias`` writes the selection
    words. The score run of a token is the row's words after the head: 28 (56 without ``drop_selection``), or
    ``_MX_SCORE_SLOT`` with ``mx_scores_align``, whose 4 zero words land on the first 4 selection words that
    ``_add_router_bias`` writes after the load. ``score_loads`` (a list), when given, gets the score DMAs in issue
    order (``pe_keep_warm``). ``add_bias`` False leaves the router bias add to the caller (``mx_reload_order``
    ``codes_ahead``: the bias runs after the code DMAs issue, on the same words). ``dma``: the DMA queue of the score
    loads (None: the SP queue; ``mx_reload_order`` ``codes_ahead_scores_act``: the Activation queue). The loaded bytes
    and the added bias do not change."""
    _check_mx_replicas(start, count, rows)
    kernel_assert(count <= _PMAX, f"the latent_gather_mx scores load takes at most {_PMAX} tokens, got {count}")
    shard = hbm["shard_dst"]
    cols = shard.shape[1]
    tokens = shard.shape[0] // TP_SIZE
    n_rep = count // rows
    r0 = start // rows
    gathered = nl.ndarray((count, TP_SIZE, _GATHER_COLS), dtype=nl.float32, buffer=nl.sbuf)
    score_words = cols - _MX_HEAD_WORDS
    kernel_assert(
        score_words in ((DP_EXPERTS, _MX_SCORE_SLOT) if router_bias is not None else (2 * DP_EXPERTS,)),
        f"the latent_gather_mx rows hold {score_words} score words: 28 or {_MX_SCORE_SLOT} with drop_selection "
        f"(router_bias), 56 without it",
    )
    score_dma = dma or _HBM_DMA
    gathered_u32 = gathered.view(nl.uint32)
    for rr in range(n_rep):
        load = nisa.dma_copy(
            dst=gathered_u32[
                rr * rows : rr * rows + rows, :, _GATHER_SCORE_COLS[0] : _GATHER_SCORE_COLS[0] + score_words
            ],
            src=shard.ap(
                pattern=[[score_words, rows], [tokens * cols, TP_SIZE], [1, score_words]],
                offset=(r0 + rr) * rows * cols + _MX_HEAD_WORDS * rows,
            ),
            name=f"{name}_mx_scores_r{rr}",
            **score_dma,
        )
        if score_loads is not None:
            score_loads.append(load)
    if router_bias is not None and add_bias:
        _add_router_bias(gathered, router_bias)
    return gathered


_MX_STREAM_HALF = _MX_STREAM_COLS // 2
"""Words of each half of an ``mx_rows_stream`` row (32): the code words, then the tail (the scale word, the 28 scores,
3 zero words). The rows load moves each half as one 128-B element."""


def _load_mx_stream_rows(hbm: dict, start: int, count: int, name: str, after=None) -> nl.NkiTensor:
    """Load the ``mx_rows_stream`` rows of the ``dp_tp`` gather of tokens ``[start, start + count)`` (at most 128) in
    one DMA of 128-B elements (the two halves of each row). Return the tile ``[count, 2, 32 * 32]`` UINT32: part 0 holds
    the code words of rank t at words 32 t .. 32 t + 31 (the codes of the 4 ranks of a latent tile are one run), part 1
    the tail of rank t at words 32 t .. (word 0: the scale word, words 1 .. 28: the scores). ``after``: an instruction
    that the load waits for."""
    shard = hbm["shard_dst"]
    kernel_assert(
        shard.shape[1] == _MX_STREAM_COLS and count <= _PMAX,
        f"the mx_rows_stream rows load takes {_MX_STREAM_COLS}-word rows and at most {_PMAX} tokens, got "
        f"{shard.shape[1]} words and {count} tokens",
    )
    tokens = shard.shape[0] // TP_SIZE
    half = _MX_STREAM_HALF
    stream = nl.ndarray((count, 2, TP_SIZE * half), dtype=nl.uint32, buffer=nl.sbuf)
    load = nisa.dma_copy(
        dst=stream.ap(pattern=[[2 * TP_SIZE * half, count], [half, TP_SIZE], [TP_SIZE * half, 2], [1, half]], offset=0),
        src=shard.ap(
            pattern=[[_MX_STREAM_COLS, count], [tokens * _MX_STREAM_COLS, TP_SIZE], [half, 2], [1, half]],
            offset=start * _MX_STREAM_COLS,
        ),
        name=f"{name}_rows_load",
        **_HBM_DMA,
    )
    _after(load, after)
    return stream


def _mx_stream_scores(stream: nl.NkiTensor, router_bias: nl.NkiTensor, name: str) -> tuple:
    """The scores ``[count, 32, 28]`` FP32 of a stream tile (``_load_mx_stream_rows``, a view of its tails) and the
    selection values score + bias of expert 28 t + e (a tile of its own; the FP32 add of ``_add_router_bias``)."""
    count = stream.shape[0]
    tails = stream.view(nl.float32)[:, 1, :].reshape((count, TP_SIZE, _MX_STREAM_HALF))
    scores = tails[:, :, 1 : 1 + DP_EXPERTS]
    selection = nl.ndarray((count, TP_SIZE, DP_EXPERTS), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=selection,
        data1=scores,
        data2=router_bias.ap(pattern=[[NUM_EXPERTS, count], [DP_EXPERTS, TP_SIZE], [1, DP_EXPERTS]], offset=0),
        op=nl.add,
        engine=nisa.engine.vector,
        name=f"{name}_mxs_selection",
    )
    return scores, selection


def _mx_stream_to_mx(stream: nl.NkiTensor, codes: nl.NkiTensor, scales: nl.NkiTensor, col: int, name: str):
    """Move the MXFP8 codes and scales of a stream tile (``_load_mx_stream_rows``, ``count`` tokens) into the MX expert
    input at columns ``[col, col + count)``: ``codes`` ``[128, 7, N]`` ``float8_e4m3fn_x4`` (partition 32 Q + w of
    latent tile h: code word w of rank 4 h + Q, as ``_load_mx_codes``) and ``scales`` ``[128, 7, N]`` UINT8
    (partition 32 Q + k: block k; the other partitions 0).

    Codes: for each latent tile h, 2 Tensor Engine transposes of the BF16 halves (low, high) of the 128 code words of
    ranks 4 h .. 4 h + 3 (one run of the stream tile, read with stride 2), then one Vector Engine copy that puts the 2
    halves of each word together. Scales: one Vector Engine copy of the scale bytes into the low bytes of a zeroed BF16
    staging tile, at column 32 Q + k of tile h; one transpose a tile; one copy of the low bytes. Tensor Engine
    transposes are bit-accurate from NeuronCore-v3, and a Vector Engine copy of one dtype is bit-accurate, so the bytes
    do not change. The reinterpreted views take slices only (an access pattern applies to a tile of its own dtype)."""
    count = stream.shape[0]
    n_tiles = LATENT_SIZE // _MX_TILE
    quad = _MX_QUADRANT_PARTITIONS
    code_run = 2 * _MX_Q * _MX_STREAM_HALF  # BF16 halves of the code words of one latent tile
    halves = stream.view(nl.bfloat16)  # [count, 2, 2 * 32 * 32]
    code_halves = codes.view(nl.bfloat16)  # [128, 7, 2 N]
    for h in range(n_tiles):
        words = nl.ndarray((_PMAX, 2, count), dtype=nl.bfloat16, buffer=nl.psum)
        for half in range(2):
            nisa.nc_transpose(
                dst=words[:, half, :],
                data=halves[:, 0, h * code_run + half : (h + 1) * code_run : 2],
                engine=nisa.engine.tensor,
                name=f"{name}_mxs_codes_h{h}_{half}",
            )
        nisa.tensor_copy(
            dst=code_halves[:, h, 2 * col : 2 * (col + count)].reshape((_PMAX, count, 2)),
            src=words.ap(pattern=[[2 * count, _PMAX], [1, count], [count, 2]], offset=0),
            engine=nisa.engine.vector,
            name=f"{name}_mxs_codes_copy_h{h}",
        )
    # The scale bytes: byte k of the tail word 0 of rank 4 h + Q, to the low byte of staging column 32 Q + k.
    staging = nl.ndarray((count, n_tiles, _PMAX), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.memset(dst=staging, value=0.0, engine=nisa.engine.vector, name=f"{name}_mxs_scale_zero")
    # Staging byte 64 t + 2 k (t = 4 h + Q) is the low byte of column 32 Q + k of tile h.
    staging_bytes = staging.view(nl.uint8).reshape((count, MX_LATENT_RANKS, 2 * quad))
    tail_bytes = stream.view(nl.uint8)[:, 1, :].reshape((count, TP_SIZE, 4 * _MX_STREAM_HALF))
    nisa.tensor_copy(
        dst=staging_bytes[:, :, 0 : 2 * _MX_SCALES_PER_QUADRANT : 2],
        src=tail_bytes[:, 0:MX_LATENT_RANKS, 0:_MX_SCALES_PER_QUADRANT],
        engine=nisa.engine.vector,
        name=f"{name}_mxs_scale_stage",
    )
    block_scales = nl.ndarray((_PMAX, n_tiles, count), dtype=nl.bfloat16, buffer=nl.psum)
    for h in range(n_tiles):
        nisa.nc_transpose(
            dst=block_scales[:, h, :], data=staging[:, h, :], engine=nisa.engine.tensor, name=f"{name}_mxs_scales_h{h}"
        )
    nisa.tensor_copy(
        dst=scales[:, :, col : col + count],
        src=block_scales.view(nl.uint8)[:, :, 0 : 2 * count : 2],
        engine=nisa.engine.vector,
        name=f"{name}_mxs_scales_copy",
    )


def _load_chunk_mx_tiles(
    shard: nl.NkiTensor,
    quant: nl.NkiTensor,
    scale: nl.NkiTensor,
    tokens: int,
    cols: int,
    r0: int,
    n_rep: int,
    rows: int,
    name: str,
    after,
    scale_dma: dict,
    tile0_loads: list = None,
    codes_after=None,
):
    """The ``load_order`` ``tile`` form of the code and scale loads of ``_load_mx_codes``: for each latent tile h
    (h = 0 first, the read order of ``moe_tkg_decode``) and quadrant Q, one code DMA of rank block 4 h + Q into
    partitions ``[32 Q, 32 Q + 32)`` of tile h, then one scale DMA into partitions ``32 Q + 0 .. 3`` of tile h. The same
    bytes as the quadrant form, which loads all 7 tiles of a quadrant in one DMA. ``tile0_loads`` (a list), when
    given, gets the tile-0 DMAs in issue order; ``codes_after``: an instruction that the first code DMA follows in the
    schedule."""
    n_tiles = LATENT_SIZE // _MX_TILE
    n_quadrants = _PMAX // _MX_QUADRANT_PARTITIONS
    quad = _MX_QUADRANT_PARTITIONS
    shard_bytes = shard.view(nl.uint8)
    first_code = None  # the code DMA that waits for ``after`` (_after_once)
    for h in range(n_tiles):
        for q in range(n_quadrants):
            block = (h * n_quadrants + q) * tokens * cols + r0 * rows * cols
            code_load = nisa.dma_copy(
                dst=quant.view(nl.uint32)[q * quad : q * quad + quad, h, :].reshape((quad, n_rep, rows)),
                src=shard.ap(pattern=[[rows, quad], [rows * cols, n_rep], [1, rows]], offset=block),
                name=f"{name}_mx_codes_q{q}_h{h}",
                **_HBM_DMA,
            )
            first_code = _after_once(code_load, after, first_code)
            if h == 0 and q == 0:
                _scheduled_after(code_load, codes_after)
            scale_load = nisa.dma_copy(
                dst=scale[q * quad : q * quad + _MX_SCALES_PER_QUADRANT, h, :].reshape(
                    (_MX_SCALES_PER_QUADRANT, n_rep, rows)
                ),
                src=shard_bytes.ap(
                    pattern=[[rows, _MX_SCALES_PER_QUADRANT], [rows * cols * 4, n_rep], [1, rows]],
                    offset=(block + _MX_CODE_WORDS * rows) * 4,
                ),
                name=f"{name}_mx_scale_q{q}_h{h}",
                **scale_dma,
            )
            if tile0_loads is not None and h == 0:
                tile0_loads += [code_load, scale_load]


def _load_router_bias(
    moe_vectors_dp: nl.NkiTensor, partitions: int, name: str, after=None, dma: dict = None
) -> nl.NkiTensor:
    """Load the router bias of all 896 experts (``ROUTER_BIAS_COLS``) onto ``partitions`` token partitions:
    [partitions, 896] FP32 SBUF, element (i, 28 t + e) is the bias of expert 28 t + e. ``after``: an instruction of
    the same DMA queue that the load follows (a schedule edge). ``dma``: the queue (``ROWS_DMAS``; None: the SP
    queue)."""
    router_bias = nl.ndarray((partitions, NUM_EXPERTS), dtype=nl.float32, buffer=nl.sbuf)
    load = nisa.dma_copy(
        dst=router_bias,
        src=moe_vectors_dp.ap(pattern=[[0, partitions], [1, NUM_EXPERTS]], offset=ROUTER_BIAS_COLS[0]),
        name=f"{name}_router_bias",
        **(dma or _HBM_DMA),
    )
    if after is not None:
        load = load if hasattr(load, "depends_on") else nisa.NkiInstruction(load)
        load.depends_on(
            after if hasattr(after, "depends_on") else nisa.NkiInstruction(after), kind=nisa.dependency.schedule
        )
    return router_bias


def _add_router_bias(
    gathered: nl.NkiTensor,
    router_bias: nl.NkiTensor,
    score_cols: tuple = _GATHER_SCORE_COLS,
    selection_cols: tuple = _GATHER_SELECTION_COLS,
):
    """Write the selection words of a [count, 32, 112] gathered tile of 84-word rows: score + bias of expert 28 t + e,
    the FP32 add that the source rank runs without ``drop_selection``, on the same scores and biases.
    ``score_cols`` and ``selection_cols``: the columns of another tile layout (a ``rows_pairs`` tile:
    ``_pair_row_words``)."""
    count = gathered.shape[0]
    nisa.tensor_tensor(
        dst=gathered[:, :, selection_cols[0] : selection_cols[1]],
        data1=gathered[:, :, score_cols[0] : score_cols[1]],
        data2=router_bias.ap(pattern=[[NUM_EXPERTS, count], [DP_EXPERTS, TP_SIZE], [1, DP_EXPERTS]], offset=0),
        op=nl.add,
        engine=nisa.engine.vector,
    )


def _keep_warm_by_lead(pe_keep_warm) -> bool:
    """True when the glue value ``pe_keep_warm`` is a form for each attention kind: a non-empty dict whose keys are
    ``PE_KEEP_WARM_LEADS`` (not the ``{"matmuls", "tile0"}`` form of one call)."""
    if not isinstance(pe_keep_warm, dict) or not pe_keep_warm:
        return False
    keys = set(pe_keep_warm)
    if keys <= {"matmuls", "tile0"}:
        return False
    kernel_assert(
        keys <= set(PE_KEEP_WARM_LEADS),
        f"pe_keep_warm as a dict takes the keys matmuls / tile0 (one form) or {PE_KEEP_WARM_LEADS} (a form for each "
        f"kind), got {sorted(keys)}",
    )
    for kind, form in pe_keep_warm.items():
        kernel_assert(not _keep_warm_by_lead(form), f"pe_keep_warm[{kind!r}] must be one form, got {form!r}")
        _keep_warm_form(form)
    return True


def _keep_warm_form(pe_keep_warm) -> tuple[int, bool]:
    """The keep-warm form of the glue value ``pe_keep_warm``: (score-tied matmuls, tile-0 matmul). False, None and 0
    give (0, False), the key off; True gives (``PE_KEEP_WARM_MATMULS``, False); an int k from 1 to ``PE_KEEP_WARM_MAX``
    gives (k, False); a dict ``{"matmuls": k, "tile0": bool}`` gives (k, tile0)."""
    if pe_keep_warm is None or pe_keep_warm is False or (type(pe_keep_warm) is int and pe_keep_warm == 0):
        return 0, False
    if pe_keep_warm is True:
        return PE_KEEP_WARM_MATMULS, False
    if isinstance(pe_keep_warm, dict):
        kernel_assert(
            set(pe_keep_warm) <= {"matmuls", "tile0"} and "matmuls" in pe_keep_warm,
            f"pe_keep_warm as a dict takes the keys matmuls and tile0, got {sorted(pe_keep_warm)}",
        )
        matmuls, tile0 = pe_keep_warm["matmuls"], pe_keep_warm.get("tile0", False)
    else:
        matmuls, tile0 = pe_keep_warm, False
    kernel_assert(
        type(matmuls) is int and 1 <= matmuls <= PE_KEEP_WARM_MAX and isinstance(tile0, bool),
        f"pe_keep_warm must be False, None, 0, True, an int 1 to {PE_KEEP_WARM_MAX} or a dict "
        f"{{matmuls: 1 to {PE_KEEP_WARM_MAX}, tile0: bool}}, got {pe_keep_warm!r}",
    )
    return matmuls, tile0


def _pe_keep_warm_operands(name: str, after) -> tuple:
    """The operands of the keep-warm MX matmuls of an expert call (``pe_keep_warm``): x4 codes ``[128, 512]`` and unit
    E8M0 scales ``[128, 512]``, set by two GpSimd memsets. The memsets follow ``after`` (the first score DMA of the
    call) in the schedule, so they and the tiles stay in the MoE front of the layer. Every code byte is finite (+0, -0
    or 1.875: the bytes of the FP32 value 1.0)."""
    data = nl.ndarray((_PMAX, _KEEP_WARM_COLS), dtype=nl.uint32, buffer=nl.sbuf, name=f"{name}_kw_data")
    data_set = nisa.memset(dst=data.view(nl.float32), value=1.0, engine=nisa.engine.gpsimd, name=f"{name}_kw_data_set")
    scale = nl.ndarray((_PMAX, _KEEP_WARM_COLS), dtype=nl.uint8, buffer=nl.sbuf, name=f"{name}_kw_scale")
    scale_set = nisa.memset(dst=scale, value=127, engine=nisa.engine.gpsimd, name=f"{name}_kw_scale_set")
    _scheduled_after(data_set, after)
    _scheduled_after(scale_set, after)
    return data.view(nl.float8_e4m3fn_x4), scale


def _pe_keep_warm_matmuls(operands: tuple, triggers: list, matmuls: int, name: str, tail=None) -> nl.NkiTensor:
    """Issue ``matmuls`` keep-warm MX matmuls (``pe_keep_warm``) into one PSUM bank of their own, as one accumulation
    group, so that no matmul is a dead store, and return their zero word.

    Matmul i follows the DMA ``round((i + 1) n / matmuls) - 1`` of ``triggers`` (the n score DMAs of the call in issue
    order) in the schedule. It is not a semaphore wait: a runtime edge on a DMA breaks the released neuronx-cc 2.28
    (``tkg_weight_dma_gate.wait_for_completion``). ``_load_chunk_mx`` issues the score DMAs before the code and scale DMAs on the
    same SP queue, so the last matmul follows a DMA that the queue runs before the tile-0 code and scale DMAs (the
    latent operands of the first expert MX matmul), and that matmul waits at most for one keep-warm matmul. With
    ``mx_reload_order`` ``codes_ahead_scores_act`` the score DMAs are on the Activation queue: no queue then orders them
    before the tile-0 DMAs, so the first expert MX matmul can wait behind a keep-warm matmul (perf only). These
    score-tied matmuls precede the first expert MX matmul in the TensorE stream. ``tail``, when given (the last tile-0
    code / scale DMA, a prerequisite of the first expert MX matmul), gets one more matmul. It has no order edge to the
    expert matmuls, which become ready at the same DMA: in the served C = 8 program (the TensorE order of its NEFF) it
    runs inside the expert stream, after the tile-0 MX matmuls of the first 6 gate / up chunks. No expert accumulation
    shares the bank. Right after the last matmul, one word of the bank is copied out and ANDed with 0 (any bits, NaN
    included, give 0). Without ``tail`` the bank is then free before the expert matmuls; with ``tail`` the bank stays
    held until that matmul, inside the expert stream. Return that [128, 1] UINT32 zero word (``_pe_keep_warm_fold``)."""
    data, scale = operands
    bank = nl.ndarray((_PMAX, _KEEP_WARM_COLS), dtype=nl.float32, buffer=nl.psum, name=f"{name}_kw_psum")
    n = len(triggers)
    kernel_assert(n > 0, f"pe_keep_warm needs the score DMAs of the call, got {n}")
    waits = [triggers[min(n - 1, max(0, round((i + 1) * n / matmuls) - 1))] for i in range(matmuls)]
    waits += [tail] if tail is not None else []
    for i, trigger in enumerate(waits):
        matmul = nisa.nc_matmul_mx(
            dst=bank,
            stationary=data[:, 0:_PMAX],
            moving=data[:, 0:_KEEP_WARM_COLS],
            stationary_scale=scale[:, 0:_PMAX],
            moving_scale=scale[:, 0:_KEEP_WARM_COLS],
            accumulate=i > 0,
            name=f"{name}_kw_mm{i}" if i < matmuls else f"{name}_kw_mm_tile0",
        )
        # A schedule edge: a runtime edge on a DMA breaks the released neuronx-cc 2.28 (tkg_weight_dma_gate.wait_for_completion).
        _instruction(matmul).depends_on(_instruction(trigger), kind=nisa.dependency.schedule)
    word = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf, name=f"{name}_kw_word")
    nisa.tensor_copy(dst=word, src=bank[:, 0:1], engine=nisa.engine.vector, name=f"{name}_kw_sink")
    zero = nl.ndarray((_PMAX, 1), dtype=nl.uint32, buffer=nl.sbuf, name=f"{name}_kw_zero")
    nisa.tensor_scalar(
        dst=zero,
        data=word.view(nl.uint32),
        op0=nl.bitwise_and,
        operand0=0,
        engine=nisa.engine.vector,
        name=f"{name}_kw_and",
    )
    return zero


def _pe_keep_warm_fold(zero: nl.NkiTensor, affinities: nl.NkiTensor, name: str):
    """OR the keep-warm zero word (``_pe_keep_warm_matmuls``) into the first affinity word of each token: the same bits
    (x OR 0 = x, NaN payloads included). The expert call now reads the keep-warm matmuls, so the compiler keeps them."""
    tokens = affinities.shape[0]
    first = affinities.view(nl.uint32)[0:tokens, 0:1]
    nisa.tensor_tensor(
        dst=first,
        data1=first,
        data2=zero[0:tokens, :],
        op=nl.bitwise_or,
        engine=nisa.engine.vector,
        name=f"{name}_kw_fold",
    )


def _chunk_routed_partial(
    hbm: dict,
    gather: str,
    start: int,
    count: int,
    rows: int,
    expert_weights: tuple,
    group_select: nl.NkiTensor,
    split: dict,
    routed_store_form: str,
    loop_options: dict,
    name: str,
    router_bias: nl.NkiTensor = None,
    routed_reduce: str = "ar",
    rows_after=None,
    quantize_first: bool = False,
    head_row_chunks: bool = False,
    rows_latent: str = "bf16",
    rows_load_split: int = 1,
    latent_gather_mx: bool = False,
    mx_load_order: str = "quadrant",
    mx_scale_queue: str = "act",
    mx_reload_order: str = "default",
    store_queues: str = "sp",
    routed_layout: str = "natural",
    pe_keep_warm: tuple = None,
    scale_fill_after: tuple = (),
):
    """Run this rank's experts on gathered tokens ``[start, start + count)`` and store their routed partial into the
    columns of those tokens of the routed reduction source (``_store_routed_partial``). ``router_bias``: the 84-word
    rows (``drop_selection``). ``rows_after``: an instruction that the rows load waits for (``_load_chunk_rows``).
    ``quantize_first``: the latent quantize runs before the bias add and the affinities, and ``moe_tkg_decode`` takes
    its codes and scales (``latent_moe_decode_2srv`` ``quantize_first``). ``head_row_chunks`` (with ``quantize_first``):
    the rows load, the latent copy and the latent quantize run by chunks of TP blocks (``_chunked_latent_quantize``).
    ``rows_load_split``: the parts of the rows load, each with its latent copy (``load_chunk_latent``).
    ``latent_gather_mx``: load the MXFP8 codes and scales straight into the expert input (``_load_chunk_mx``) and run
    no latent quantize; ``mx_load_order`` and ``mx_scale_queue`` are the order and the scale queue of those loads, and
    ``mx_reload_order`` is the trace order of the code, scale and score loads of a keep-warm call (``MX_RELOAD_ORDERS``;
    ``_load_chunk_mx``). ``store_queues``: the queues of the routed store (``STORE_QUEUES``). ``pe_keep_warm`` (with
    ``latent_gather_mx``): the keep-warm form (``_keep_warm_form``: score-tied matmuls, tile-0 matmul) before the first
    expert MX matmul (``_pe_keep_warm_matmuls``); None or (0, False) for none. ``scale_fill_after`` (with
    ``latent_gather_mx``): instructions that the zero fill of the gathered code scales follows in the schedule
    (``_load_chunk_mx``)."""
    kernel_assert(not head_row_chunks or quantize_first, "head_row_chunks needs quantize_first")
    kw_matmuls, kw_tile0 = pe_keep_warm if pe_keep_warm is not None else (0, False)
    kernel_assert(
        not kw_matmuls or latent_gather_mx,
        f"pe_keep_warm {pe_keep_warm} needs latent_gather_mx, got latent_gather_mx={latent_gather_mx}",
    )
    kernel_assert(
        not kw_tile0 or (kw_matmuls and mx_load_order == "tile"),
        f"the pe_keep_warm tile0 matmul needs score-tied matmuls and mx_load_order tile, got {pe_keep_warm} and "
        f"{mx_load_order!r}",
    )
    latent_input = {}
    keep_warm_zero = None
    if latent_gather_mx:
        score_loads = [] if kw_matmuls else None
        tile0_loads = [] if kw_tile0 else None
        gathered, mx_quant, mx_scale = _load_chunk_mx(
            hbm,
            start,
            count,
            rows,
            name,
            router_bias,
            after=rows_after,
            **({"load_order": mx_load_order} if mx_load_order != "quadrant" else {}),
            **({"scale_queue": mx_scale_queue} if mx_scale_queue != "act" else {}),
            **({"reload_order": mx_reload_order} if mx_reload_order != "default" else {}),
            **({"score_loads": score_loads} if kw_matmuls else {}),
            **({"tile0_loads": tile0_loads} if kw_tile0 else {}),
            **({"scale_fill_after": scale_fill_after} if scale_fill_after else {}),
        )
        if kw_matmuls:
            operands = _pe_keep_warm_operands(name, after=score_loads[0])
            keep_warm_zero = _pe_keep_warm_matmuls(
                operands, score_loads, kw_matmuls, name, tail=tile0_loads[-1] if kw_tile0 else None
            )
        latent_sb = None
        latent_input = {"quantized_latent": (mx_quant, mx_scale)}
    elif head_row_chunks:
        gathered = _load_chunk_rows(
            hbm,
            gather,
            start,
            count,
            rows,
            name,
            router_bias,
            rows_latent,
            after=rows_after,
            add_bias=not quantize_first,
            row_chunks=_HEAD_ROW_CHUNKS,
        )
        latent_sb = nl.ndarray((count, TP_SIZE, DP_LATENT), dtype=nl.bfloat16, buffer=nl.sbuf)
    else:
        gathered, latent_sb = load_chunk_latent(
            hbm,
            gather,
            start,
            count,
            rows,
            name,
            router_bias,
            rows_latent,
            rows_load_split,
            after=rows_after,
            add_bias=not quantize_first,
        )
    if quantize_first:
        kernel_assert(count % _QUANT_ROWS == 0, f"quantize_first needs a multiple of {_QUANT_ROWS} tokens, got {count}")
        if head_row_chunks:
            codes, scales = _chunked_latent_quantize(gathered, latent_sb, count, _HEAD_ROW_CHUNKS)
        else:
            # The quantize of moe_tkg_decode, traced before the affinities: its transposes read latent_sb in place.
            codes, scales = _quantize_natural_latent(
                latent_sb=latent_sb.reshape((count, LATENT_SIZE)), T=count, D=LATENT_SIZE
            )
        latent_input = {"quantized_latent": (codes, scales)}
        if router_bias is not None:
            _add_router_bias(gathered, router_bias)
    local_affinities = _local_affinities(
        gathered[:, :, _GATHER_SCORE_COLS[0] : _GATHER_SCORE_COLS[1]],
        gathered[:, :, _GATHER_SELECTION_COLS[0] : _GATHER_SELECTION_COLS[1]],
        group_select,
        split,
    )
    if keep_warm_zero is not None:
        _pe_keep_warm_fold(keep_warm_zero, local_affinities, name)
    gate_up, gate_up_scale, down, down_scale = expert_weights
    if gate_up.dtype == nl.uint16:
        gate_up = gate_up.view(nl.float4_e2m1fn_x4)
    if down.dtype == nl.uint16:
        down = down.view(nl.float4_e2m1fn_x4)
    staged = nl.ndarray((ROUTED_PARTITIONS, count, ROUTED_COLS), dtype=nl.bfloat16, buffer=nl.sbuf)
    pe_store = {"column_groups_out": staged} if routed_store_form == "pe" else {}
    expert_partial = moe_tkg_decode(
        None if quantize_first or latent_gather_mx else latent_sb.reshape((count, LATENT_SIZE)),
        local_affinities,
        gate_up,
        down,
        gate_up_scale,
        down_scale,
        SITU_BETA,
        SITU_LINEAR_BETA,
        **loop_options,
        **pe_store,
        **latent_input,
    )
    if routed_store_form == "relayout":
        # The 56-byte runs of the layout go to a local SBUF tile first: as a DMA into HBM they run at a fifth of the rate.
        nisa.dma_copy(
            dst=staged,
            src=expert_partial.ap(
                pattern=[[ROUTED_COLS, ROUTED_PARTITIONS], [LATENT_SIZE, count], [1, ROUTED_COLS]], offset=0
            ),
            name=f"{name}_routed_stage",
            **_HBM_DMA,
        )
    _store_routed_partial(hbm, staged, start, count, rows, routed_reduce, name, store_queues, routed_layout)


def _chunked_latent_quantize(
    gathered: nl.NkiTensor, latent_sb: nl.NkiTensor, count: int, chunks: int
) -> tuple[nl.NkiTensor, nl.NkiTensor]:
    """Copy the latent of the gathered router rows ``gathered`` ([count, 32, 112] FP32) into ``latent_sb`` ([count, 32,
    112] BF16) and quantize it to the MXFP8 expert input, by ``chunks`` chunks of TP blocks (``head_row_chunks``).

    For each chunk in block order: the Vector copy of its latent words, then the swizzle and the quantize of every D512
    tile whose 512 columns the chunks copied so far hold (chunks of 896 columns: tile 0, tiles 1-2, tiles 3-4, tiles
    5-6). A tile's swizzle is ``_quantize_natural_latent``'s: 4 TensorE transposes of the stride-4 column views and one
    ScalarE copy out of PSUM; one quantize then covers the tiles of the chunk. An MX scale group (8 partitions x 4
    values) never spans two D512 tiles, and each instruction reads the bytes of the one-piece form, so the codes and
    scales are bit for bit those of ``_quantize_natural_latent`` on the whole latent. Return them.
    """
    D = LATENT_SIZE
    tile_cols = MAX_MATMULT_MX_UNPACKED_CONTRACT_DIM
    n_D512 = D // tile_cols
    blocks = TP_SIZE // chunks
    flat = latent_sb.reshape((count, D))
    swizzled = nl.ndarray((_PMAX, n_D512, count, _q_width), dtype=nl.bfloat16, buffer=nl.sbuf)
    quant = nl.ndarray((_PMAX, n_D512, count), dtype=nl.float8_e4m3fn_x4, buffer=nl.sbuf)
    scale = nl.ndarray((_PMAX, n_D512, count), dtype=nl.uint8, buffer=nl.sbuf)
    done = 0
    for chunk in range(chunks):
        first = chunk * blocks
        nisa.tensor_copy(
            dst=latent_sb[:, first : first + blocks, :],
            src=gathered[:, first : first + blocks, _GATHER_LATENT_WORDS[0] : _GATHER_LATENT_WORDS[1]].view(
                nl.bfloat16
            ),
            engine=nisa.engine.vector,
        )
        ready = (first + blocks) * DP_LATENT // tile_cols
        for tile_h in range(done, ready):
            swizzle_psum = nl.ndarray((_PMAX, _q_width, count), dtype=nl.bfloat16, buffer=nl.psum)
            for q_idx in range(_q_width):
                nisa.nc_transpose(
                    dst=swizzle_psum[:, q_idx, :],
                    data=flat.ap(pattern=[[D, count], [_q_width, _PMAX]], offset=tile_h * tile_cols + q_idx),
                    engine=nisa.engine.tensor,
                )
            nisa.tensor_copy(
                dst=swizzled[:, tile_h, :, :],
                src=swizzle_psum.ap(pattern=[[_q_width * count, _PMAX], [1, count], [count, _q_width]]),
                engine=nisa.engine.scalar,
            )
        if ready > done:
            nisa.quantize_mx(
                src=swizzled[:, done:ready, :, :], dst=quant[:, done:ready, :], dst_scale=scale[:, done:ready, :]
            )
        done = ready
    kernel_assert(done == n_D512, f"the latent chunks cover {done} of {n_D512} D512 tiles")
    return quant, scale


def _tiles_routed_partial(
    hbm: dict,
    gather: str,
    start: int,
    count: int,
    rows: int,
    expert_weights: tuple,
    group_select: nl.NkiTensor,
    split: dict,
    loop_options: dict,
    name: str,
    router_bias: nl.NkiTensor = None,
    routed_reduce: str = "ar",
    rows_after=None,
    transpose_blocks: bool = False,
    rows_latent: str = "bf16",
    rows_load_split: int = 1,
    latent_gather_mx: bool = False,
    static_alt_tiles: bool | str = False,
    store_queues: str = "sp",
    rows_pairs: bool = False,
    per_expert: bool | int | dict = False,
    routed_layout: str = "natural",
):
    """Run this rank's experts on gathered tokens ``[start, start + count)`` (``count`` = 128 n, n >= 2) in one
    ``moe_tkg_decode_token_tiles`` call, and store their routed partial into the columns of those tokens of the routed
    reduction source (``_store_routed_partial``). Each 128-token tile has the glue of ``_chunk_routed_partial``.
    ``rows_after``: an instruction that the rows load of the first tile waits for (``_load_chunk_rows``).
    ``transpose_blocks`` and ``static_alt_tiles`` are those of the call; ``store_queues`` the queues of the routed
    store. ``latent_gather_mx``: load the MXFP8 codes and scales of all ``count`` tokens straight into the expert input
    (``_load_mx_codes``, after ``rows_after``), then the scores of each tile (``_load_mx_scores``), and pass the codes to
    the call as ``quantized_latent``: no latent copy, swizzle or quantize. ``rows_pairs`` (``check_rows_pairs``): one
    rows load for each two tiles (``_load_pair_rows``), token tile 2 i + j = tokens ``start + 256 i + 2 p + j``, and
    the call's ``token_pairs`` output map. ``per_expert``: one ``moe_tkg_decode_per_expert`` call in place of the
    token-tiled call (``latent_moe_decode_2srv`` ``routed_per_expert``: True, its ``combine_cols`` or a dict of call
    options; ``_per_expert_call_options``, which also sets the call's whole weight buffers), with the HBM scratch of
    ``hbm``."""
    # first_add_zero acts on moe_tkg_decode only: the token-tiled and per-expert calls keep their accumulator fill.
    loop_options = {key: value for key, value in loop_options.items() if key != "first_add_zero"}
    latent_tiles, affinity_tiles = [], []
    quantized = _load_mx_codes(hbm, start, count, rows, name, after=rows_after) if latent_gather_mx else None
    for tile_idx in range(count // _PMAX):
        tile_start = start + tile_idx * _PMAX
        if latent_gather_mx:
            gathered = _load_mx_scores(hbm, tile_start, _PMAX, rows, f"{name}_t{tile_idx}", router_bias)
            affinity_tiles.append(
                _local_affinities(
                    gathered[:, :, _GATHER_SCORE_COLS[0] : _GATHER_SCORE_COLS[1]],
                    gathered[:, :, _GATHER_SELECTION_COLS[0] : _GATHER_SELECTION_COLS[1]],
                    group_select[0:_PMAX, :],
                    split,
                )
            )
            continue
        if rows_pairs:
            if tile_idx % 2 == 0:
                pair_rows = _load_pair_rows(
                    hbm, tile_start, f"{name}_t{tile_idx}{tile_idx + 1}", after=rows_after if tile_idx == 0 else None
                )
            gathered, (base, score_cols, selection_cols) = pair_rows, _pair_row_words(tile_idx % 2)
            if router_bias is not None:
                _add_router_bias(gathered, router_bias, score_cols, selection_cols)
            latent_sb = nl.ndarray((_PMAX, TP_SIZE, DP_LATENT), dtype=nl.bfloat16, buffer=nl.sbuf)
            latent_from_rows(gathered, latent_sb, rows_latent, base)
        else:
            gathered, latent_sb = load_chunk_latent(
                hbm,
                gather,
                tile_start,
                _PMAX,
                rows,
                f"{name}_t{tile_idx}",
                router_bias,
                rows_latent,
                rows_load_split,
                after=rows_after if tile_idx == 0 else None,
            )
            score_cols, selection_cols = _GATHER_SCORE_COLS, _GATHER_SELECTION_COLS
        affinity_tiles.append(
            _local_affinities(
                gathered[:, :, score_cols[0] : score_cols[1]],
                gathered[:, :, selection_cols[0] : selection_cols[1]],
                group_select[0:_PMAX, :],
                split,
            )
        )
        latent_tiles.append(latent_sb.reshape((_PMAX, LATENT_SIZE)))
    gate_up, gate_up_scale, down, down_scale = expert_weights
    if gate_up.dtype == nl.uint16:
        gate_up = gate_up.view(nl.float4_e2m1fn_x4)
    if down.dtype == nl.uint16:
        down = down.view(nl.float4_e2m1fn_x4)
    staged = nl.ndarray((ROUTED_PARTITIONS, count, ROUTED_COLS), dtype=nl.bfloat16, buffer=nl.sbuf)
    if per_expert:
        # One scratch for each HBM set of the body: the calls order its reuse (per_expert_scratch).
        if "per_expert_scratch" not in hbm:
            hbm["per_expert_scratch"] = per_expert_scratch(gate_up.shape[0], LATENT_SIZE, f"{name}_pe")
        kernel_assert(
            not set(loop_options) - set(_PER_EXPERT_OPTIONS),
            f"routed_per_expert takes the loop options {_PER_EXPERT_OPTIONS}, got {sorted(loop_options)}",
        )
        moe_tkg_decode_per_expert(
            latent_tiles,
            affinity_tiles,
            gate_up,
            down,
            gate_up_scale,
            down_scale,
            SITU_BETA,
            SITU_LINEAR_BETA,
            **{**loop_options, **_per_expert_call_options(per_expert, loop_options.get("cross_expert_order", False))},
            column_groups_out=staged,
            scratch=hbm["per_expert_scratch"],
            name=f"{name}_pe",
        )
        _store_routed_partial(hbm, staged, start, count, rows, routed_reduce, name, store_queues, routed_layout)
        return
    moe_tkg_decode_token_tiles(
        None if latent_gather_mx else latent_tiles,
        affinity_tiles,
        gate_up,
        down,
        gate_up_scale,
        down_scale,
        SITU_BETA,
        SITU_LINEAR_BETA,
        **loop_options,
        column_groups_out=staged,
        **({"transpose_blocks": True} if transpose_blocks else {}),
        **({"quantized_latent": quantized} if latent_gather_mx else {}),
        **({"static_alt_tiles": static_alt_tiles} if static_alt_tiles else {}),
        **({"token_pairs": True} if rows_pairs else {}),
    )
    _store_routed_partial(hbm, staged, start, count, rows, routed_reduce, name, store_queues, routed_layout)


ROUTED_TILE_FORM = "slice"
"""The ``layout_form`` of an expert split of sub-512 MX tiles (``moe_routed_layout.routed_split``, ``mx_tile``
256, 128, 64 or 32). The body runs such a split with the routed-expert tile
(``tkg_moe_routed_tiles.routed_expert_tile``) in a static loop over the local experts (``_routed_tiles_partial``).
A split of whole 512 tiles keeps the static loop of ``moe_tkg_decode``."""
ROUTED_TILE_WEIGHT_BUFFERS = 2
"""SBUF weight sets of the routed-tile loop: expert e loads into set e % 2, so the loads of the next expert can run
during the matmuls of this one."""
TILE_FORM_STATIC_GLUE = (
    *COMPACT_GLUE_KEYS,
    "weights_after_rows",
    "routed_per_expert",
    "rows_pairs",
    "pe_keep_warm",
    "mx_reload_order",
    "last_expert_order",
    "accum_memset_after_rows",
    "first_add_zero",
)
"""The glue keys of the static loop of ``moe_tkg_decode`` (the routed-token compaction and its keys, the stream gate of
its weights, the per-expert calls, the paired rows load, the keep-warm matmuls and the order of their reloads, the
order of the last experts of a cross-order call, the accumulator fill of the static calls): ``moe_2srv_body`` drops
them for a ``ROUTED_TILE_FORM`` split, which runs the routed-tile loop. Each keeps the values of the served loop, so a
served config's glue (for its served-split layers) runs a sub-512-tile layer of a routed plan unchanged."""
TILE_FORM_STATIC_OPTIONS = (
    "cross_expert_order",
    "cross_expert_transpose_blocks",
    "token_tiles_transpose_blocks",
    "transpose_block",
    "transpose_lookahead",
    "static_alt_tiles",
    "keep_warm_lead",
)
"""The expert-loop options of the decode block that only the static loop takes (the cross-expert order and its
transpose blocks, the lookahead, the alternating tiles, the keep-warm lead): ``moe_2srv_body`` drops them for a
``ROUTED_TILE_FORM`` split. The other options of the loop (weight / scale buffers, early loads) do not apply to it
either; the body takes them and the loop does not read them. A prefetched stream (``moe_expert_prefetch``) holds the
served layout: the body's ``prefetch`` returns None for such a split, so its layers take no stream."""


def _check_routed_tile_options(
    expert_call_tokens: int,
    routed_compact,
    routed_per_expert,
    quantize_first: bool,
    head_row_chunks: bool,
    prefetch,
    rows_pairs: bool,
    static_alt_tiles,
    keep_warm: int,
    cross_expert_order: bool,
    transpose_lookahead,
    weights_after: bool,
):
    """Refuse the expert-loop options that the routed-tile loop (``ROUTED_TILE_FORM``) does not take: it runs the
    128-token tiles of all gathered tokens with its own loop and weight sets. The static-loop buffer options
    (``weight_buffers``, ``scale_buffers``, ``early_loads``) and ``routed_store_form`` do not apply to it. It takes
    ``latent_gather_mx`` (``_routed_tiles_partial``)."""
    taken = {
        "expert_call_tokens": expert_call_tokens != EXPERT_CALL_TOKENS,
        "routed_compact": bool(routed_compact),
        "routed_per_expert": bool(routed_per_expert),
        "quantize_first": quantize_first,
        "head_row_chunks": head_row_chunks,
        "prefetch": prefetch is not None,
        "rows_pairs": rows_pairs,
        "static_alt_tiles": bool(static_alt_tiles),
        "pe_keep_warm": bool(keep_warm),
        "cross_expert_order": cross_expert_order,
        "transpose_lookahead": transpose_lookahead is not None,
        "weights_after_gather / weights_after_rows": weights_after,
    }
    given = sorted(key for key, value in taken.items() if value)
    kernel_assert(
        not given,
        f"the routed-expert tile loop of a sub-512-tile split takes 128-token tiles and none of the static-loop options "
        f"{given}",
    )


def _routed_tile_relayout(hbm: dict, acc: nl.NkiTensor, count: int, name: str) -> nl.NkiTensor:
    """Round the FP32 [count, 3584] routed accumulator of a token tile to BF16 and return it in the routed layout
    [128, count, 28] BF16 SBUF. A tile of 128 tokens moves with the PE transposes of ``_to_column_groups_at``; a
    smaller tile through an HBM scratch (one for each HBM set of the body), as the ``relayout`` store of the static
    loop."""
    rows_bf16 = nl.ndarray((count, LATENT_SIZE), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=rows_bf16, src=acc, engine=nisa.engine.vector)
    staged = nl.ndarray((ROUTED_PARTITIONS, count, ROUTED_COLS), dtype=nl.bfloat16, buffer=nl.sbuf)
    if count == _PMAX:
        _to_column_groups_at(rows_bf16, staged, 0)
        return staged
    scratch = hbm.get("routed_tile_scratch")
    if scratch is None:
        scratch = nl.ndarray((_PMAX, LATENT_SIZE), dtype=nl.bfloat16, buffer=nl.private_hbm, name=f"moe2s_{name}_rows")
        hbm["routed_tile_scratch"] = scratch
    nisa.dma_copy(dst=scratch[0:count, :], src=rows_bf16, name=f"{name}_rows_store", **_HBM_DMA)
    nisa.dma_copy(
        dst=staged,
        src=scratch.ap(pattern=[[ROUTED_COLS, ROUTED_PARTITIONS], [LATENT_SIZE, count], [1, ROUTED_COLS]], offset=0),
        name=f"{name}_routed_stage",
        **_HBM_DMA,
    )
    return staged


def _routed_tiles_partial(
    hbm: dict,
    gather: str,
    tokens: int,
    rows: int,
    expert_weights: tuple,
    group_select: nl.NkiTensor,
    split: dict,
    name: str,
    router_bias: nl.NkiTensor = None,
    routed_reduce: str = "ar",
    rows_latent: str = "bf16",
    rows_load_split: int = 1,
    store_queues: str = "sp",
    routed_layout: str = "natural",
    rows_after=None,
    routed_dynamic: bool = False,
    routed_dynamic_pack: int = None,
    latent_gather_mx: bool = False,
    mx_load_order: str = "quadrant",
    mx_scale_queue: str = "act",
    routed_dynamic_csr: str = None,
):
    """Run this rank's E_L experts on all T gathered tokens with the routed-expert tile and store the routed partial
    of every token (``ROUTED_TILE_FORM`` splits).

    Each token tile of at most 128 tokens (``token_chunks``) loads its gathered router rows and BF16 latent
    (``load_chunk_latent``), takes the local affinities (``_local_affinities``) and quantizes the latent to MXFP8
    (``_quantize_natural_latent``, as ``moe_tkg_decode``). A static loop over the local experts then loads each
    expert's weights once (``load_tile_weights`` into one of ``ROUTED_TILE_WEIGHT_BUFFERS`` SBUF sets) and runs
    ``routed_expert_tile`` on every token tile: the tile's FP32 accumulator takes affinity x the expert's output, in
    expert order. Each accumulator rounds to BF16 once (``_routed_tile_relayout``) and goes to the routed reduction
    source (``_store_routed_partial``). ``rows_after``: an instruction that the first rows load waits for.

    ``routed_dynamic``: the same prep, then only the active experts, each on a compact tile of its routed tokens, into
    an FP32 partial (``tkg_moe_routed_dynamic.routed_dynamic_layer``); each token tile reads its FP32 rows
    back (+0 for a token on no local expert) and takes the same BF16 round, relayout and store.
    ``routed_dynamic_pack``: the experts of one packed item; None takes ``default_pack`` of the I-slice.
    ``routed_dynamic_csr``: the CSR gathers of a packed layer; None keeps "staged".

    ``latent_gather_mx``: each token tile loads its MXFP8 codes and scales and its scores straight from the G1a rows
    (``_load_chunk_mx``, with ``mx_load_order`` and ``mx_scale_queue``) in place of the BF16 latent load and the
    quantize: the same expert input bytes and the same scores, so the same outputs. Each tile holds whole replicas.
    """
    i_slice = split["local_intermediate"]
    tiles = token_chunks(tokens)
    kernel_assert(
        not latent_gather_mx or all(start % rows == 0 and count % rows == 0 for start, count in tiles),
        f"latent_gather_mx loads whole replicas in each token tile of the routed-tile loop: B = {rows} rows must divide "
        f"{_PMAX}, got the tiles {tiles}",
    )
    prepared = []
    for tile_idx, (start, count) in enumerate(tiles):
        tile_name = f"{name}_t{tile_idx}"
        after = rows_after if tile_idx == 0 else None
        if latent_gather_mx:
            gathered, codes, scales = _load_chunk_mx(
                hbm,
                start,
                count,
                rows,
                tile_name,
                router_bias,
                after=after,
                load_order=mx_load_order,
                scale_queue=mx_scale_queue,
            )
        else:
            gathered, latent_sb = load_chunk_latent(
                hbm, gather, start, count, rows, tile_name, router_bias, rows_latent, rows_load_split, after=after
            )
        affinities = _local_affinities(
            gathered[:, :, _GATHER_SCORE_COLS[0] : _GATHER_SCORE_COLS[1]],
            gathered[:, :, _GATHER_SELECTION_COLS[0] : _GATHER_SELECTION_COLS[1]],
            group_select[0:count, :],
            split,
        )
        if not latent_gather_mx:
            codes, scales = _quantize_natural_latent(
                latent_sb=latent_sb.reshape((count, LATENT_SIZE)), T=count, D=LATENT_SIZE
            )
        acc = None
        if not routed_dynamic:
            acc = nl.ndarray((count, LATENT_SIZE), dtype=nl.float32, buffer=nl.sbuf, name=f"{tile_name}_acc")
        prepared.append((count, affinities, codes, scales, acc))
    if routed_dynamic:
        layer = routed_dynamic_layer(
            hbm,
            [entry[1] for entry in prepared],
            [entry[2] for entry in prepared],
            [entry[3] for entry in prepared],
            expert_weights,
            i_slice,
            f"{name}_dyn",
            pack=default_pack(i_slice, split["local_experts"]) if routed_dynamic_pack is None else routed_dynamic_pack,
            **({"csr_gather": routed_dynamic_csr} if routed_dynamic_csr is not None else {}),
        )
        for tile_idx, (start, count) in enumerate(tiles):
            tile_name = f"{name}_t{tile_idx}"
            rows_f = routed_partial_tile(
                layer["partial"], layer["dmap"], tile_idx, f"{tile_name}_dyn", layer["scatters"], nl.float32
            )
            staged = _routed_tile_relayout(hbm, rows_f, count, tile_name)
            _store_routed_partial(
                hbm, staged, start, count, rows, routed_reduce, tile_name, store_queues, routed_layout
            )
        return
    weight_sets = [alloc_tile_weights(i_slice, f"{name}_w{index}") for index in range(ROUTED_TILE_WEIGHT_BUFFERS)]
    gate_up, gate_up_scale, down, down_scale = expert_weights
    last = None
    for expert in range(split["local_experts"]):
        weights = weight_sets[expert % ROUTED_TILE_WEIGHT_BUFFERS]
        load_tile_weights(weights, gate_up, gate_up_scale, down, down_scale, expert, i_slice, f"{name}_e{expert}")
        for tile_idx, (count, affinities, codes, scales, acc) in enumerate(prepared):
            last = routed_expert_tile(
                acc,
                codes,
                scales,
                weights["gate_up_w"],
                weights["gate_up_s"],
                weights["down_w"],
                weights["down_s"],
                count,
                i_slice,
                f"{name}_e{expert}_t{tile_idx}",
                after=last,
                affinity=affinities[:, expert : expert + 1],
                accumulate=expert > 0,
            )
    for tile_idx, ((start, count), (_, _, _, _, acc)) in enumerate(zip(tiles, prepared, strict=True)):
        tile_name = f"{name}_t{tile_idx}"
        staged = _routed_tile_relayout(hbm, acc, count, tile_name)
        _store_routed_partial(hbm, staged, start, count, rows, routed_reduce, tile_name, store_queues, routed_layout)


_PER_EXPERT_OPTIONS = ("weight_buffers", "scale_buffers", "early_loads", "cross_expert_order", "transpose_block")
"""The expert-loop options of a ``moe_tkg_decode_per_expert`` call (``routed_per_expert``)."""
PER_EXPERT_WEIGHT_BUFFERS = 2
"""The whole weight buffers of a ``routed_per_expert`` call in the default expert order (the cross-expert order takes
3). The per-expert call holds its combine tiles next to the expert loop, so 2 buffers keep its SBUF at the MoE layer
below that of the token-tiled call it replaces."""


def _per_expert_call_options(routed_per_expert, cross_expert_order: bool = False) -> dict:
    """The ``moe_tkg_decode_per_expert`` options of a ``routed_per_expert`` value: True (the defaults), the combine
    columns, or a dict of ``combine_cols`` / ``slot_gather`` / ``weight_buffers``. The whole weight buffers default to
    ``PER_EXPERT_WEIGHT_BUFFERS``, or 3 with ``cross_expert_order``."""
    options = {
        "combine_cols": PER_EXPERT_COMBINE_COLS,
        "slot_gather": PER_EXPERT_SLOT_GATHERS[0],
        "weight_buffers": 3 if cross_expert_order else PER_EXPERT_WEIGHT_BUFFERS,
    }
    if isinstance(routed_per_expert, dict):
        kernel_assert(
            not set(routed_per_expert) - set(options),
            f"routed_per_expert takes the keys {sorted(options)}, got {sorted(routed_per_expert)}",
        )
        options.update(routed_per_expert)
    elif routed_per_expert is not True:
        options["combine_cols"] = routed_per_expert
    return options


COMPACT_CONSTANTS_KEY = "_k3_moe2s_compact_constants"
"""The attribute of ``rank_info`` that holds the ``routed_compact`` constants of the kernel (``moe_2srv_compact_constants``)."""


def moe_2srv_compact_constants(
    rank_info: nl.NkiTensor, name: str = "k3moe2s", sb=None, key_tiles: int = 0, local_experts: int = 0
) -> dict:
    """Return the data-independent constants of the ``routed_compact`` slot tables (``_compact_tables``: the 0/1
    prefix stationary ``before``, the all-ones stationary ``ones`` and the slot ids ``slot_ids``), built once per kernel
    and kept on ``rank_info``: the first call builds them, the later calls return the same tiles. Every MoE layer of the
    kernel then takes them in place of its own two iotas, one compare and one memset. The entry builds them before its
    first block, with a heap allocator ``sb`` on a kernel with an MLA loop path (``tkg_sbuf_alloc.SbufTiles(region,
    heap=True)``, as ``moe_2srv_glue_selectors``), so they keep their bytes across the loops. The values do not
    change. With a heap allocator, the ramp that builds ``before`` is a stack tile of a frame that closes after the
    compare, so the heap keeps 1 KiB a partition (``before``, ``ones``, ``slot_ids``).

    ``key_tiles`` > 0 (the glue key ``compact_key_constants``) also builds the token keys of ``key_tiles`` token tiles:
    ``keys16`` (BF16 ``[128, 2]`` = (p + 1, u) for tile u), ``key_p1_rows`` (BF16 ``[128, 128]``, p + 1 on every
    column: the word-0 stationary of every tile) and ``key_u_rows`` (the word-1 stationary of tile u: None for tile 0,
    whose keys are 0, ``ones`` for tile 1, a BF16 ``[128, 128]`` tile of u after). ``local_experts`` > 0 also builds
    ``hot_ids`` (FP32 ``[128, E_L]``, E_L - 1 - e: the expert order of ``hot_last``). These are the values that a
    call builds for itself without these constants (integers of at most 128: exact in BF16). The heap keeps 256 + 4
    ``key_tiles`` + 4 E_L bytes a partition more (+ 256 for each tile from 2)."""
    cached = getattr(rank_info, COMPACT_CONSTANTS_KEY, None)
    if cached is not None:
        kernel_assert(
            len(cached.get("keys16", ())) >= key_tiles and (not local_experts or "hot_ids" in cached),
            f"moe_2srv_compact_constants: the kernel's constants hold {len(cached.get('keys16', ()))} key tiles, "
            f"asked for {key_tiles} (and hot_ids {bool(local_experts)}): build them all at the first call",
        )
    if cached is None:
        fp32, bf16 = nl.float32, nl.bfloat16
        vector = nisa.engine.vector
        out = AUTO_SBUF if sb is None else sb
        framed = sb is not None and sb.explicit and sb.heap
        with sb.region.frame(f"{name}_cmp_ramp") if framed else contextlib.nullcontext():
            ramp = sb.region.stack((_PMAX, _PMAX), fp32) if framed else out.tile((_PMAX, _PMAX), fp32)
            nisa.iota(dst=ramp, pattern=[[1, _PMAX]], offset=0, channel_multiplier=-1, name=f"{name}_cmp_ramp")
            before = out.tile((_PMAX, _PMAX), bf16)
            nisa.tensor_scalar(dst=before, data=ramp, op0=nl.greater, operand0=0.0, engine=vector)
        ones = out.tile((_PMAX, _PMAX), bf16)
        nisa.memset(dst=ones, value=1.0, engine=vector)
        slot_ids = out.tile((_PMAX, _PMAX), fp32)
        nisa.iota(dst=slot_ids, pattern=[[1, _PMAX]], offset=0, channel_multiplier=0, name=f"{name}_cmp_slot_ids")
        cached = {"before": before, "ones": ones, "slot_ids": slot_ids}
        if key_tiles:
            keys16, key_u_rows = [], []
            key_p1_rows = out.tile((_PMAX, _PMAX), bf16)
            for tile_idx in range(key_tiles):
                # The instructions of the per-call keys (_compact_tables), once for the kernel.
                with sb.region.frame(f"{name}_cmp_key_t{tile_idx}") if framed else contextlib.nullcontext():
                    key = sb.region.stack((_PMAX, 2), fp32) if framed else out.tile((_PMAX, 2), fp32)
                    nisa.iota(
                        dst=key[:, 0:1],
                        pattern=[[0, 1]],
                        offset=1,
                        channel_multiplier=1,
                        name=f"{name}_cmp_key_t{tile_idx}",
                    )
                    nisa.memset(dst=key[:, 1:2], value=float(tile_idx), engine=vector)
                    key16 = out.tile((_PMAX, 2), bf16)
                    nisa.tensor_copy(dst=key16, src=key, engine=vector)
                    if tile_idx == 0:
                        nisa.tensor_scalar(
                            dst=key_p1_rows, data=ones, op0=nl.multiply, operand0=key[:, 0:1], engine=vector
                        )
                    u_rows = None if tile_idx == 0 else ones
                    if tile_idx > 1:
                        u_rows = out.tile((_PMAX, _PMAX), bf16)
                        nisa.tensor_scalar(dst=u_rows, data=ones, op0=nl.multiply, operand0=key[:, 1:2], engine=vector)
                keys16.append(key16)
                key_u_rows.append(u_rows)
            cached.update({"keys16": keys16, "key_p1_rows": key_p1_rows, "key_u_rows": key_u_rows})
        if local_experts:
            hot_ids = out.tile((_PMAX, local_experts), fp32)
            nisa.iota(
                dst=hot_ids,
                pattern=[[-1, local_experts]],
                offset=local_experts - 1,
                channel_multiplier=0,
                name=f"{name}_hot_ids",
            )
            cached["hot_ids"] = hot_ids
        object.__setattr__(rank_info, COMPACT_CONSTANTS_KEY, cached)
    return cached


def _compact_tables(
    affinity_tiles: list,
    count: int,
    name: str,
    late_always: bool = False,
    constants: dict = None,
    hot_last: bool = False,
    skip_unrouted: bool = False,
    snake: dict = None,
    scalar_copies: bool = False,
) -> dict:
    """Return the slot tables of ``routed_compact`` for the FP32 ``[128, E_L]`` affinities of the token tiles of a call
    (token ``128 u + p`` of the call on partition p of tile u).

    A token is routed when any of its local affinities is nonzero (NaN counts as nonzero). The routed tokens take the
    slots 0, 1, ... in token order: a token's slot is the count of routed tokens before it, an exclusive prefix sum on
    TensorE (a 0/1 stationary with 1 where p < j). The token of each slot comes from 0/1 matmuls of the one-hot of the
    slots with the token keys (p + 1, u): on the slots' partitions for the affinity rows, and broadcast to every
    partition for the token-axis gather of the codes. Every value is an integer of at most 1,024: the BF16 0/1 operands
    and the FP32 PSUM sums are exact. Index ``count`` is the zero column of the codes / the zero row of the affinity and
    slot-output rows.

    Returns:
        ``token_offsets``: for each token tile, INT32 ``[128, 1]``, the token's slot, or ``count`` when no local expert
        takes it (-1 with ``skip_unrouted``: an out-of-bounds row that the gather skips). ``slot_rows``: for each slot
        tile k, INT32 ``[128, 1]``, the call-token index of slot 128 k + s, or ``count`` for an empty slot.
        ``slot_columns``: for each slot tile k, UINT32 ``[128, 128]`` on every partition, the ``nc_n_gather``
        indices of the slot tile's columns in each ``[128, count + 1]`` D512 tile of the codes of the call: the slot's
        token, or ``count``. ``overflow``: for each slot tile k >= 1, INT32 ``[1, 1]``, 1 when it holds a token (more
        than 128 k routed tokens), else 0; 1 with ``late_always``.

    ``constants``: the kernel's ``moe_2srv_compact_constants``, or None (the call builds its own, the same values).

    ``hot_last`` (``routed_compact`` ``hot_last``): the hot expert is the local expert with the most routed tokens of
    the call (the lowest id on a tie). The routed tokens whose only local expert is the hot one take the last slots, the
    other routed tokens the first ones, each class in token order. A later slot tile then holds mostly tokens of the
    hot expert alone. The counts and slots are integers of at most 1,024 from 0/1 matmuls and exact FP32 compares.

    ``snake`` (the ``_snake_constants`` of the call's codes, ``compact_slot_gather`` other than ``gpsimd``): the tables
    hold ``slot_index`` in place of ``slot_columns``: for each slot tile k, the UINT16 ``[128, 8 D/512]``
    tensor-indirection index of its slots' code columns (``_snake_index`` of the slot's call token on its partition,
    ``count`` for an empty slot: the token of ``slot_rows``, the same integers).

    ``scalar_copies`` (``compact_tables_scalar``): False (default) keeps the program. True moves the copies whose
    consumers are not the next VectorE instructions to ScalarE (the BF16 0/1 flags of the prefix matmuls, the routed
    count, the INT32 slot rows and overflow), and writes each token offset in one VectorE add with an INT32 destination
    in place of the add and its copy. Every value is an exact integer of at most 1,024 or a 0/1 flag, so the bits do
    not change.
    """
    fp32, bf16, i32 = nl.float32, nl.bfloat16, nl.int32
    vector = nisa.engine.vector
    # scalar_copies: the copies that no next VectorE instruction reads run on ScalarE.
    copies = nisa.engine.scalar if scalar_copies else vector
    n_tiles = len(affinity_tiles)
    local_experts = affinity_tiles[0].shape[1]
    if constants is not None:
        before, ones, slot_ids = constants["before"], constants["ones"], constants["slot_ids"]
    else:
        ramp = nl.ndarray((_PMAX, _PMAX), dtype=fp32, buffer=nl.sbuf)
        nisa.iota(dst=ramp, pattern=[[1, _PMAX]], offset=0, channel_multiplier=-1, name=f"{name}_cmp_ramp")
        before = nl.ndarray((_PMAX, _PMAX), dtype=bf16, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=before, data=ramp, op0=nl.greater, operand0=0.0, engine=vector)
        ones = nl.ndarray((_PMAX, _PMAX), dtype=bf16, buffer=nl.sbuf)
        nisa.memset(dst=ones, value=1.0, engine=vector)
        slot_ids = nl.ndarray((_PMAX, _PMAX), dtype=fp32, buffer=nl.sbuf)
        nisa.iota(dst=slot_ids, pattern=[[1, _PMAX]], offset=0, channel_multiplier=0, name=f"{name}_cmp_slot_ids")
    # The kernel's token keys (moe_2srv_compact_constants key_tiles): the same values as the per-call keys below.
    hoisted_keys = constants is not None and "keys16" in constants
    kernel_assert(
        not hoisted_keys or len(constants["keys16"]) >= n_tiles,
        f"_compact_tables: the kernel's constants hold {len(constants['keys16']) if hoisted_keys else 0} key tiles, "
        f"the call has {n_tiles} token tiles",
    )
    flags, flags16, keys16, key_rows16 = [], [], [], []
    for tile_idx, affinities in enumerate(affinity_tiles):
        nonzero = nl.ndarray((_PMAX, local_experts), dtype=fp32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=nonzero, data=affinities, op0=nl.not_equal, operand0=0.0, engine=vector)
        flag = nl.ndarray((_PMAX, 1), dtype=fp32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=flag, op=nl.maximum, data=nonzero, axis=(1,))
        flag16 = nl.ndarray((_PMAX, 1), dtype=bf16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=flag16, src=flag, engine=copies)
        flags.append(flag)
        flags16.append(flag16)
        if hoisted_keys:
            keys16.append(constants["keys16"][tile_idx])
            key_rows16.append([constants["key_p1_rows"], constants["key_u_rows"][tile_idx]])
            continue
        key = nl.ndarray((_PMAX, 2), dtype=fp32, buffer=nl.sbuf)
        nisa.iota(dst=key[:, 0:1], pattern=[[0, 1]], offset=1, channel_multiplier=1, name=f"{name}_cmp_key_t{tile_idx}")
        nisa.memset(dst=key[:, 1:2], value=float(tile_idx), engine=vector)
        key16 = nl.ndarray((_PMAX, 2), dtype=bf16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=key16, src=key, engine=vector)
        # The keys of token p on every column: the stationaries of the broadcast token of each slot.
        key_rows = []
        for word in range(2):
            rows16 = nl.ndarray((_PMAX, _PMAX), dtype=bf16, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=rows16, data=ones, op0=nl.multiply, operand0=key[:, word : word + 1], engine=vector)
            key_rows.append(rows16)
        keys16.append(key16)
        key_rows16.append(key_rows)
    slots, token_offsets = [], []
    if hot_last:
        hot_ids = (constants or {}).get("hot_ids")
        hot_slots = _hot_last_slots(affinity_tiles, flags, before, ones, name, hot_ids=hot_ids)
    for tile_idx in range(n_tiles):
        if hot_last:
            slot = hot_slots[tile_idx]
        else:
            slot_psum = nl.ndarray((_PMAX, 1), dtype=fp32, buffer=nl.psum)
            nisa.nc_matmul(dst=slot_psum, stationary=before, moving=flags16[tile_idx], accumulate=False)
            for earlier in range(tile_idx):
                nisa.nc_matmul(dst=slot_psum, stationary=ones, moving=flags16[earlier], accumulate=True)
            slot = nl.ndarray((_PMAX, 1), dtype=fp32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=slot, src=slot_psum, engine=vector)
        # (slot - count) * routed + count: the slot, or the zero row (skip_unrouted: (slot + 1) * routed - 1, or -1).
        unrouted = -1.0 if skip_unrouted else float(count)
        offset_f = nl.ndarray((_PMAX, 1), dtype=fp32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=offset_f,
            data=slot,
            op0=nl.add,
            operand0=-unrouted,
            op1=nl.multiply,
            operand1=flags[tile_idx],
            engine=vector,
        )
        offset = nl.ndarray((_PMAX, 1), dtype=i32, buffer=nl.sbuf)
        if scalar_copies:
            # One add with the INT32 destination: the exact integer of the add and its copy.
            nisa.tensor_scalar(dst=offset, data=offset_f, op0=nl.add, operand0=unrouted, engine=vector)
        else:
            nisa.tensor_scalar(dst=offset_f, data=offset_f, op0=nl.add, operand0=unrouted, engine=vector)
            nisa.tensor_copy(dst=offset, src=offset_f, engine=vector)
        slots.append(slot)
        token_offsets.append(offset)
    count_psum = nl.ndarray((_PMAX, 1), dtype=fp32, buffer=nl.psum)
    for tile_idx in range(n_tiles):
        nisa.nc_matmul(dst=count_psum, stationary=ones, moving=flags16[tile_idx], accumulate=tile_idx > 0)
    routed_count = nl.ndarray((_PMAX, 1), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=routed_count, src=count_psum, engine=copies)
    slot_rows, slot_columns, slot_index, overflow = [], [], [], []
    for slot_tile in range(n_tiles):
        onehots = []
        for tile_idx in range(n_tiles):
            relative = nl.ndarray((_PMAX, 1), dtype=fp32, buffer=nl.sbuf)
            nisa.tensor_scalar(
                dst=relative, data=slots[tile_idx], op0=nl.add, operand0=float(-_PMAX * slot_tile), engine=vector
            )
            # onehot[p, s] = (s == slot - 128 k) * routed: token p of the tile takes slot s of the slot tile.
            onehot = nl.ndarray((_PMAX, _PMAX), dtype=bf16, buffer=nl.sbuf)
            nisa.tensor_scalar(
                dst=onehot,
                data=slot_ids,
                op0=nl.equal,
                operand0=relative,
                op1=nl.multiply,
                operand1=flags[tile_idx],
                engine=vector,
            )
            onehots.append(onehot)
        # On the slots' partitions: (p + 1, u) of the slot's token, or (0, 0).
        token_psum = nl.ndarray((_PMAX, 2), dtype=fp32, buffer=nl.psum)
        for tile_idx in range(n_tiles):
            nisa.nc_matmul(
                dst=token_psum, stationary=onehots[tile_idx], moving=keys16[tile_idx], accumulate=tile_idx > 0
            )
        keys = nl.ndarray((_PMAX, 2), dtype=fp32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=keys, src=token_psum, engine=vector)
        if snake is not None:
            # The slot's call token on its partition (FP32, exact): the tensor-indirection index and the row index.
            slot_token = _slot_token_index(keys[:, 0:1], keys[:, 1:2], count, (_PMAX, 1), fp32, f"{name}_s{slot_tile}")
            slot_index.append(_snake_index(slot_token, snake))
            rows_i = nl.ndarray((_PMAX, 1), dtype=i32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=rows_i, src=slot_token, engine=copies)
            slot_rows.append(rows_i)
        else:
            slot_rows.append(
                _slot_token_index(keys[:, 0:1], keys[:, 1:2], count, (_PMAX, 1), i32, f"{name}_s{slot_tile}")
            )
        # On every partition, along the slots: the same keys (broadcast stationaries).
        cols = []
        for word in range(0 if snake is not None else 2):
            word_psum = nl.ndarray((_PMAX, _PMAX), dtype=fp32, buffer=nl.psum)
            # A None stationary (the hoisted word-1 keys of tile 0: all 0) adds only +0 to the 0/1 sums: skipped.
            terms = [(t, key_rows16[t][word]) for t in range(n_tiles) if key_rows16[t][word] is not None]
            kernel_assert(terms, "_compact_tables: a call of one token tile has no word-1 keys to sum")
            for term_idx, (tile_idx, stationary) in enumerate(terms):
                nisa.nc_matmul(
                    dst=word_psum,
                    stationary=stationary,
                    moving=onehots[tile_idx],
                    accumulate=term_idx > 0,
                )
            word_sb = nl.ndarray((_PMAX, _PMAX), dtype=fp32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=word_sb, src=word_psum, engine=vector)
            cols.append(word_sb)
        if snake is None:
            slot_columns.append(
                _slot_token_index(cols[0], cols[1], count, (_PMAX, _PMAX), nl.uint32, f"{name}_c{slot_tile}")
            )
        if slot_tile > 0:
            more_f = nl.ndarray((1, 1), dtype=fp32, buffer=nl.sbuf)
            threshold = -1.0 if late_always else float(_PMAX * slot_tile)
            nisa.tensor_scalar(
                dst=more_f, data=routed_count[0:1, 0:1], op0=nl.greater, operand0=threshold, engine=vector
            )
            more = nl.ndarray((1, 1), dtype=i32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=more, src=more_f, engine=copies)
            overflow.append(more)
    tables = {"token_offsets": token_offsets, "slot_rows": slot_rows, "overflow": overflow}
    if snake is not None:
        tables["slot_index"] = slot_index
    else:
        tables["slot_columns"] = slot_columns
    return tables


def _exclusive_prefix(flags16: list, before: nl.NkiTensor, ones: nl.NkiTensor) -> list:
    """Return, for each token tile, the FP32 ``[128, 1]`` count of the flagged tokens before each token (in token
    order over the tiles) from the BF16 0/1 ``[128, 1]`` flags of the tiles: 0/1 matmuls, exact."""
    out = []
    for tile_idx in range(len(flags16)):
        prefix_psum = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=prefix_psum, stationary=before, moving=flags16[tile_idx], accumulate=False)
        for earlier in range(tile_idx):
            nisa.nc_matmul(dst=prefix_psum, stationary=ones, moving=flags16[earlier], accumulate=True)
        prefix = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=prefix, src=prefix_psum, engine=nisa.engine.vector)
        out.append(prefix)
    return out


def _hot_last_slots(
    affinity_tiles: list, flags: list, before: nl.NkiTensor, ones: nl.NkiTensor, name: str, hot_ids=None
) -> list:
    """Return the slot of each token of the token tiles in the hot-expert-last order (``_compact_tables``
    ``hot_last``): FP32 ``[128, 1]`` per tile, valid on the routed tokens (``flags``).

    The 0/1 masks and class flags are BF16 (exact), so the matmuls read them with no copy; the counts, keys, prefix sums
    and slots are FP32 integers of at most 28 x 1,024 + 27 (exact). ``hot_ids``: the kernel's FP32 ``[128, E_L]``
    expert order (``moe_2srv_compact_constants`` ``hot_ids``), or None (the call builds the same values)."""
    fp32, bf16 = nl.float32, nl.bfloat16
    vector = nisa.engine.vector
    n_tiles = len(affinity_tiles)
    local_experts = affinity_tiles[0].shape[1]
    nonzeros, n_locals = [], []
    count_psum = nl.ndarray((_PMAX, local_experts), dtype=fp32, buffer=nl.psum)
    for tile_idx, affinities in enumerate(affinity_tiles):
        # The 0/1 mask of the token's local experts and their count (the sum of the mask), in one VectorE op.
        nonzero = nl.ndarray((_PMAX, local_experts), dtype=bf16, buffer=nl.sbuf)
        n_local = nl.ndarray((_PMAX, 1), dtype=fp32, buffer=nl.sbuf)
        nisa.scalar_tensor_tensor(
            dst=nonzero,
            data=affinities,
            op0=nl.not_equal,
            operand0=0.0,
            op1=nl.multiply,
            operand1=ones[:, 0:local_experts],
            reduce_op=nl.add,
            reduce_res=n_local,
        )
        # The routed tokens of each local expert, on every partition.
        nisa.nc_matmul(dst=count_psum, stationary=ones, moving=nonzero, accumulate=tile_idx > 0)
        nonzeros.append(nonzero)
        n_locals.append(n_local)
    # A unique key for each expert: E_L * count + (E_L - 1 - e), so the maximum is the hot expert (the lowest id on a
    # tie); the multiply-add reads the count PSUM tile.
    order = hot_ids
    if order is None:
        order = nl.ndarray((_PMAX, local_experts), dtype=fp32, buffer=nl.sbuf)
        nisa.iota(
            dst=order,
            pattern=[[-1, local_experts]],
            offset=local_experts - 1,
            channel_multiplier=0,
            name=f"{name}_hot_ids",
        )
    kernel_assert(
        tuple(order.shape) == (_PMAX, local_experts),
        f"_hot_last_slots: hot_ids is [{_PMAX}, {local_experts}], got {tuple(order.shape)}",
    )
    keys = nl.ndarray((_PMAX, local_experts), dtype=fp32, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(
        dst=keys, data=count_psum, op0=nl.multiply, operand0=float(local_experts), op1=nl.add, operand1=order
    )
    best = nl.ndarray((_PMAX, 1), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=best, op=nl.maximum, data=keys, axis=(1,))
    hot = nl.ndarray((_PMAX, local_experts), dtype=bf16, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=hot, data=keys, op0=nl.equal, operand0=best, engine=vector)
    flags_a, flags_b = [], []
    for tile_idx in range(n_tiles):
        # The token takes the hot expert: the hot mask has exactly one 1 (unique keys), so the sum is 0 or 1.
        on_hot = nl.ndarray((_PMAX, local_experts), dtype=bf16, buffer=nl.sbuf)
        has_hot = nl.ndarray((_PMAX, 1), dtype=fp32, buffer=nl.sbuf)
        nisa.scalar_tensor_tensor(
            dst=on_hot,
            data=nonzeros[tile_idx],
            op0=nl.multiply,
            operand0=1.0,
            op1=nl.multiply,
            operand1=hot,
            reduce_op=nl.add,
            reduce_res=has_hot,
        )
        # hot only: the token's one local expert is the hot one (class B, the last slots); class A = routed, not B.
        flag_b = nl.ndarray((_PMAX, 1), dtype=bf16, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=flag_b,
            data=n_locals[tile_idx],
            op0=nl.equal,
            operand0=1.0,
            op1=nl.multiply,
            operand1=has_hot,
            engine=vector,
        )
        flag_a = nl.ndarray((_PMAX, 1), dtype=bf16, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=flag_a, data1=flags[tile_idx], data2=flag_b, op=nl.subtract, engine=vector)
        flags_a.append(flag_a)
        flags_b.append(flag_b)
    prefix_a = _exclusive_prefix(flags_a, before, ones)
    prefix_b = _exclusive_prefix(flags_b, before, ones)
    count_a_psum = nl.ndarray((_PMAX, 1), dtype=fp32, buffer=nl.psum)
    for tile_idx in range(n_tiles):
        nisa.nc_matmul(dst=count_a_psum, stationary=ones, moving=flags_a[tile_idx], accumulate=tile_idx > 0)
    count_a = nl.ndarray((_PMAX, 1), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=count_a, src=count_a_psum, engine=vector)
    slots = []
    for tile_idx in range(n_tiles):
        # slot = (prefix_b + count_a) * b + prefix_a * a (exactly one of a, b is 1 on a routed token).
        b_slot = nl.ndarray((_PMAX, 1), dtype=fp32, buffer=nl.sbuf)
        nisa.scalar_tensor_tensor(
            dst=b_slot,
            data=prefix_b[tile_idx],
            op0=nl.add,
            operand0=count_a,
            op1=nl.multiply,
            operand1=flags_b[tile_idx],
        )
        slot = nl.ndarray((_PMAX, 1), dtype=fp32, buffer=nl.sbuf)
        nisa.scalar_tensor_tensor(
            dst=slot,
            data=flags_a[tile_idx],
            op0=nl.multiply,
            operand0=prefix_a[tile_idx],
            op1=nl.add,
            operand1=b_slot,
        )
        slots.append(slot)
    return slots


def _slot_token_index(first: nl.NkiTensor, tile: nl.NkiTensor, count: int, shape: tuple, dtype, name: str):
    """The call-token index ``128 u + p`` of a slot from its keys (``first`` = p + 1, ``tile`` = u; both 0 for an empty
    slot), or ``count`` for an empty slot: ``(first + 128 u - 1 - count) * (first > 0) + count``, exact in FP32."""
    fp32 = nl.float32
    vector = nisa.engine.vector
    filled = nl.ndarray(shape, dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=filled, data=first, op0=nl.greater, operand0=0.0, engine=vector)
    index = nl.ndarray(shape, dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=index,
        data=tile,
        op0=nl.multiply,
        operand0=float(_PMAX),
        op1=nl.add,
        operand1=float(-1 - count),
        engine=vector,
    )
    nisa.tensor_tensor(dst=index, data1=index, data2=first, op=nl.add, engine=vector)
    nisa.tensor_tensor(dst=index, data1=index, data2=filled, op=nl.multiply, engine=vector)
    out = nl.ndarray(shape, dtype=dtype, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=out, data=index, op0=nl.add, operand0=float(count), engine=vector)
    return out


def _compact_scratch(hbm: dict, count: int, local_experts: int, name: str) -> tuple:
    """Return the HBM scratch of ``routed_compact`` (one set for each HBM set of the body, allocated at its first use):
    the BF16 ``[count + 1, D]`` slot-output rows and the FP32 ``[count + 1, E_L]`` affinity rows; row ``count`` of
    each is the zero row (written by the call)."""
    scratch = hbm.get("compact_scratch")
    if scratch is None:
        scratch = (
            nl.ndarray((count + 1, LATENT_SIZE), dtype=nl.bfloat16, buffer=nl.private_hbm, name=f"moe2s_{name}_slots"),
            nl.ndarray((count + 1, local_experts), dtype=nl.float32, buffer=nl.private_hbm, name=f"moe2s_{name}_aff"),
        )
        hbm["compact_scratch"] = scratch
    kernel_assert(
        tuple(scratch[0].shape) == (count + 1, LATENT_SIZE) and tuple(scratch[1].shape) == (count + 1, local_experts),
        f"routed_compact keeps one [{count + 1}, {LATENT_SIZE}] slot-output tensor, got {tuple(scratch[0].shape)}",
    )
    return scratch


def _gather_rows(dst: nl.NkiTensor, rows: nl.NkiTensor, index: nl.NkiTensor, name: str, skip: bool = False):
    """Gather the HBM ``rows`` ``index`` (INT32 ``[128, 1]``, one row per partition) into ``dst`` with one indirect
    DMA (DGE, so a loop body can issue it), and return the DMA. ``skip``: an out-of-bounds index (-1) leaves its
    partition of ``dst`` unwritten."""
    width = rows.shape[1]
    return nisa.dma_copy(
        dst=dst,
        src=rows.ap(pattern=[[width, _PMAX], [1, width]], offset=0, vector_offset=index, indirect_dim=0),
        dge_mode=nisa.dge_mode.swdge,
        name=name,
        **({"oob_mode": nisa.oob_mode.skip} if skip else {}),
    )


def _compact_slot_tile(
    codes,
    scales,
    affinity_rows,
    tables: dict,
    slot_tile: int,
    name: str,
    issued: list = None,
    slot_gather: str = "gpsimd",
) -> tuple:
    """Return the MXFP8 codes and scales (``[128, D/512, 128]``) and the FP32 ``[128, E_L]`` affinities of slot tile
    ``slot_tile``: the codes and scales of its tokens' columns (``nc_n_gather`` on GpSimd along the token axis of
    every partition, one for each D512 tile: a GpSimd gather reads at most 512 elements of a partition) and their
    affinity rows (one indirect DMA). An empty slot takes the zero column and the zero row. Uses GpSimd and DGE DMAs
    only, so a loop body can call it. ``issued`` (a list), when given, gets the GpSimd gathers in trace order (with a
    tensor-indirection ``slot_gather``, which runs no GpSimd gather, the affinity rows gather, whose descriptors GpSimd
    builds).

    ``slot_gather`` (``COMPACT_SLOT_GATHERS``) other than ``gpsimd``: one tensor-indirection ``tensor_copy`` of the
    codes of every D512 tile (UINT32 words) and one of the scales, the source the whole ``[128, D/512 (count + 1)]``
    tile and the index ``tables["slot_index"]`` (``_compact_tables`` ``snake``), on ScalarE (``scalar``, and the
    codes of ``split``) or VectorE (``vector``, and the scales of ``split``). The same bytes as the gathers."""
    n_d = codes.shape[1]
    local_experts = affinity_rows.shape[1]
    slot_codes = nl.ndarray((_PMAX, n_d, _PMAX), dtype=nl.float8_e4m3fn_x4, buffer=nl.sbuf)
    slot_scales = nl.ndarray((_PMAX, n_d, _PMAX), dtype=nl.uint8, buffer=nl.sbuf)
    if slot_gather != "gpsimd":
        index = tables["slot_index"][slot_tile]
        columns, slots = n_d * codes.shape[2], n_d * _PMAX
        scalar, vector = nisa.engine.scalar, nisa.engine.vector
        nisa.tensor_copy(
            dst=slot_codes.view(nl.uint32).reshape((_PMAX, slots)),
            src=codes.view(nl.uint32).reshape((_PMAX, columns)).indirect(index, slots),
            engine=vector if slot_gather == "vector" else scalar,
            name=f"{name}_slot_codes",
        )
        nisa.tensor_copy(
            dst=slot_scales.reshape((_PMAX, slots)),
            src=scales.reshape((_PMAX, columns)).indirect(index, slots),
            engine=scalar if slot_gather == "scalar" else vector,
            name=f"{name}_slot_scales",
        )
        affinities = nl.ndarray((_PMAX, local_experts), dtype=nl.float32, buffer=nl.sbuf)
        affinity_gather = _gather_rows(
            affinities, affinity_rows, tables["slot_rows"][slot_tile], f"{name}_slot_affinities"
        )
        if issued is not None:
            issued.append(affinity_gather)
        return slot_codes, slot_scales, affinities
    columns = tables["slot_columns"][slot_tile]
    for tile_h in range(n_d):
        code_gather = nisa.nc_n_gather(
            dst=slot_codes.view(nl.uint32)[:, tile_h, :],
            data=codes.view(nl.uint32)[:, tile_h, :],
            indices=columns,
            name=f"{name}_slot_codes_h{tile_h}",
        )
        scale_gather = nisa.nc_n_gather(
            dst=slot_scales[:, tile_h, :],
            data=scales[:, tile_h, :],
            indices=columns,
            name=f"{name}_slot_scales_h{tile_h}",
        )
        if issued is not None:
            issued.extend((code_gather, scale_gather))
    affinities = nl.ndarray((_PMAX, local_experts), dtype=nl.float32, buffer=nl.sbuf)
    _gather_rows(affinities, affinity_rows, tables["slot_rows"][slot_tile], f"{name}_slot_affinities")
    return slot_codes, slot_scales, affinities


def _group_index(group_select: nl.NkiTensor, split: dict, name: str) -> nl.NkiTensor:
    """Return the expert group g of this rank as UINT32 [1, 1] SBUF from the FP32 one-hot ``group_select`` [T, G]
    (``_selectors``: 1 at column g on every partition): the sum of j * onehot[j] over the G columns of row 0, an
    integer below G, exact in FP32. A compute engine's dynamic-offset copy (``_local_affinities`` ``local_index``)
    takes its offset register as UINT32."""
    groups = split["groups"]
    ids = nl.ndarray((1, groups), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(dst=ids, pattern=[[1, groups]], offset=0, channel_multiplier=0, name=f"{name}_ids")
    picked = nl.ndarray((1, groups), dtype=nl.float32, buffer=nl.sbuf)
    group_f = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(
        dst=picked,
        data=group_select[0:1, 0:groups],
        op0=nl.multiply,
        operand0=1.0,
        op1=nl.multiply,
        operand1=ids,
        reduce_op=nl.add,
        reduce_res=group_f,
    )
    group = nl.ndarray((1, 1), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=group, src=group_f, engine=nisa.engine.vector)
    return group


def _compact_routed_partial(
    hbm: dict,
    gather: str,
    start: int,
    count: int,
    rows: int,
    expert_weights: tuple,
    group_select: nl.NkiTensor,
    split: dict,
    loop_options: dict,
    name: str,
    router_bias: nl.NkiTensor = None,
    routed_reduce: str = "ar",
    rows_after=None,
    rows_latent: str = "bf16",
    late_always: bool = False,
    store_queues: str = "sp",
    rows_load_split: int = 1,
    constants: dict = None,
    hot_last: bool = False,
    empty_later: bool = False,
    weights_first: tuple = (),
    ungather_skip: bool = False,
    routed_layout: str = "natural",
    vector_cuts: bool = False,
    up_product_split: bool = False,
    slot_gather: str = "gpsimd",
    tail_first: bool = False,
    store_halves: bool = False,
    latent_gather_mx: bool = False,
    mx_rows_stream: bool = False,
    tail_hook=None,
    alt_tiles: str = None,
    tables_scalar: bool = False,
    affinity_sbuf_copy: bool = False,
    zero_early: bool = False,
    static_first: bool = False,
):
    """Run this rank's experts on the routed tokens of gathered tokens ``[start, start + count)`` (``count`` = 128 n,
    n >= 2) and store the routed partial of every token of the call (``routed_compact``; ``_store_routed_partial``).

    1. Each 128-token tile loads its router rows and writes its latent (``load_chunk_latent``, in ``rows_load_split``
       parts), its affinities and its MXFP8 codes and scales (``_quantize_natural_latent``): the operations of each
       tile of the token-tiled call. The affinities read the rank's E_L selected scores at a runtime offset of its
       expert group (``_local_affinities`` ``local_index``; the fused weighting with ``vector_cuts``). The codes and scales of the call sit in ``[128, D/512, count + 1]`` tiles, the
       last column zero; the affinities in HBM rows (row ``count`` zero). Then the slot tables (``_compact_tables``).
    2. Slot tile 0 (the first 128 routed tokens, ``_compact_slot_tile``: a GpSimd gather of their code columns and an
       indirect DMA of their affinity rows) runs the static loop of ``moe_tkg_decode`` on the gathered codes
       (``quantized_latent``) and stores its rows into the slot-output rows (``_compact_scratch``).
    3. Slot tile k >= 1 runs in a one-trip ``nl.fori_loop`` only when the call has more than 128 k routed tokens: its
       gathers, then every expert in the slot form of ``moe_tkg_decode`` (``_all_experts_in_slots``: DGE DMAs only,
       the bits of the static loop), into its slot-output rows. With ``late_always`` (``routed_compact`` ``late``)
       every later slot tile runs. With ``hot_last`` a then-only ``if_else`` takes the place of the loop.
    4. For each token tile, an indirect DMA gathers each token's slot-output row (a token with no local expert takes
       the zero row), and the column-group transposes of ``moe_tkg_decode_token_tiles`` (``_to_column_groups_at``)
       move it into the routed layout; the routed store takes ``store_queues``. With ``tail_first`` (glue key
       ``tail_before_guard``; ``hot_last`` with one later slot tile) the tail runs before the later tile's guard, each
       token tile stored right after its transposes, and the guard's branch runs it again after the later tile.

    With ``latent_gather_mx`` step 1 loads the MXFP8 codes and scales of all the call's tokens straight into the
    gather source (``_load_mx_codes``) and the scores of each tile (``_load_mx_scores``), and runs no latent copy,
    swizzle or quantize. ``mx_rows_stream`` (with ``latent_gather_mx``): the rows are token-major (``_MX_STREAM_COLS``
    words); each tile loads its rows in one DMA (``_load_mx_stream_rows``) and moves its codes and scales into the
    gather source with bit-accurate transposes (``_mx_stream_to_mx``). The loaded bytes do not change, so the values
    do not change.

    Each slot runs the per-token operations of a 128-token static call: the codes of its token (the per-token MXFP8
    quantization of the token-tiled call), the same K and expert order, and stationary columns that do not mix. So a
    routed token's row has the bits of the token-tiled call. An unrouted token's row there is affinity * output + 0
    summed over the experts: +0 for a finite output, as the zero row here. Data moves between tokens only by gathers
    (GpSimd and DMA), so a non-finite value stays in its own token.

    ``slot_gather`` (``COMPACT_SLOT_GATHERS``, glue key ``compact_slot_gather``) other than ``gpsimd``: each slot tile
    gathers its codes and scales with two tensor-indirection copies (``_compact_slot_tile``), whose index the slot
    tables build (``_compact_tables`` ``snake``, from the data-independent ``_snake_constants``, traced at the start of
    the call), in place of the 14 GpSimd gathers and their per-partition columns. The copies move the same bytes.

    ``store_halves`` (``hot_last`` with one later slot tile): the tail stores each token tile's routed partial right
    after its transposes (the per-tile stores of ``tail_first``, also when the tail runs after the guard), as two DMAs,
    the tokens of each of the two column-group copies (``_to_column_groups_at``: VectorE, then ScalarE), so the first
    half's store starts as soon as its copy ends. The same bytes to the same places. False (default) keeps the program.

    ``alt_tiles`` (glue key ``compact_alt_tiles``; ``moe_routed.STATIC_ALT_TILE_PSUM_CONSUME``): the
    ``static_alt_tiles`` of slot tile 0's static call, which keeps ``psum_consume`` and ``resident_out``. None (default)
    keeps the program. ``tables_scalar`` (glue key ``compact_tables_scalar``): the slot tables' ScalarE copies and
    INT32 offset adds (``_compact_tables`` ``scalar_copies``); False (default) keeps the program.
    ``affinity_sbuf_copy`` (glue key ``affinity_sbuf_copy``): slot tile 0's static call takes its SBUF slot affinities
    with one VectorE copy, no SBUF -> SBUF DMA (``moe_tkg_decode`` ``affinity_sbuf_copy``); False (default) keeps the
    program. ``zero_early``
    (glue key ``tail_zero_early``; ``ungather_skip``, the tail after the guard): the tail's gather destinations take
    their zero fills before the later slot tiles' guards. ``static_first`` (glue key ``slot_static_first``, ``hot_last``):
    the later tile's active-expert loop runs its first expert as a static slot (``active_experts_in_slots``
    ``static_first``). False (default) keeps the program for each.
    """
    kernel_assert(gather == "dp_tp", f"routed_compact takes the token-tiled calls of the dp_tp gather, got {gather!r}")
    kernel_assert(
        count % _PMAX == 0 and count > _PMAX,
        f"routed_compact takes a token-tiled call of 128 n > 128 tokens, got {count}",
    )
    kernel_assert(
        count + 1 <= _GATHER_MAX_ELEMS,
        f"routed_compact gathers the {count + 1} columns of a D512 tile, at most {_GATHER_MAX_ELEMS}",
    )
    kernel_assert(
        slot_gather in COMPACT_SLOT_GATHERS and (slot_gather == "gpsimd" or count <= COMPACT_TI_MAX_TOKENS),
        f"slot_gather must be one of {COMPACT_SLOT_GATHERS}, and other than 'gpsimd' only for calls of at most "
        f"{COMPACT_TI_MAX_TOKENS} tokens, got {slot_gather!r} and {count} tokens",
    )
    n_tiles = count // _PMAX
    n_d = LATENT_SIZE // MAX_MATMULT_MX_UNPACKED_CONTRACT_DIM
    local_experts = split["local_experts"]
    slot_rows, affinity_rows = _compact_scratch(hbm, count, local_experts, name)
    if slot_gather == "gpsimd":
        codes = nl.ndarray((_PMAX, n_d, count + 1), dtype=nl.float8_e4m3fn_x4, buffer=nl.sbuf)
        scales = nl.ndarray((_PMAX, n_d, count + 1), dtype=nl.uint8, buffer=nl.sbuf)
    else:
        # A tensor-indirection source must be a whole 2D tile: the [D/512, count + 1] columns of a partition in a row.
        codes = nl.ndarray((_PMAX, n_d * (count + 1)), dtype=nl.float8_e4m3fn_x4, buffer=nl.sbuf).reshape(
            (_PMAX, n_d, count + 1)
        )
        scales = nl.ndarray((_PMAX, n_d * (count + 1)), dtype=nl.uint8, buffer=nl.sbuf).reshape((_PMAX, n_d, count + 1))
    first_pool = nisa.memset(dst=codes.view(nl.uint32)[:, :, count : count + 1], value=0, engine=nisa.engine.gpsimd)
    nisa.memset(dst=scales[:, :, count : count + 1], value=0, engine=nisa.engine.gpsimd)
    first_sp = None
    if not ungather_skip:
        # The zero row of the slot outputs: the token-order gather reads it for an unrouted token. With ungather_skip
        # the gather skips those tokens, so nothing reads it.
        # A memset costs per element (2-byte elements no less than 4-byte ones): an FP32 tile, read as its BF16 view,
        # takes the same zero bytes in half the elements.
        zero_row = nl.ndarray((1, LATENT_SIZE // 2), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=zero_row, value=0.0, engine=nisa.engine.gpsimd)
        first_sp = nisa.dma_copy(
            dst=slot_rows[count : count + 1, :],
            src=zero_row.view(nl.bfloat16),
            name=f"{name}_slots_zero_row",
            **_HBM_DMA,
        )
    zero_aff = nl.ndarray((1, local_experts), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=zero_aff, value=0.0, engine=nisa.engine.gpsimd)
    aff_store = nisa.dma_copy(
        dst=affinity_rows[count : count + 1, :], src=zero_aff, name=f"{name}_aff_zero_row", **_HBM_DMA
    )
    # weights_first: the call's first GpSimd and SP instructions follow these weight loads in the schedule.
    for load in weights_first:
        for first in (first_pool, aff_store if first_sp is None else first_sp):
            _instruction(first).depends_on(_instruction(load), kind=nisa.dependency.schedule)
    # The data-independent tiles of the tensor-indirection index, ahead of the call's data (after the weights_first
    # edges, so the first GpSimd instruction of the call keeps its place).
    snake = _snake_constants(n_d, count) if slot_gather != "gpsimd" else None
    # The local affinities read the rank's expert group at a runtime offset (the group from the one-hot of the
    # selectors: data of the kernel, not of the rows), except with vector_cuts, which keeps its fused weighting.
    local_kwargs = {} if vector_cuts else {"local_index": _group_index(group_select, split, f"{name}_grp")}
    kernel_assert(
        not mx_rows_stream or latent_gather_mx,
        "mx_rows_stream loads the token-major latent_gather_mx rows in one DMA a tile: it needs latent_gather_mx",
    )
    if latent_gather_mx and not mx_rows_stream:
        _load_mx_codes(
            hbm, start, count, rows, name, after=rows_after, dst=(codes[:, :, 0:count], scales[:, :, 0:count])
        )
    affinity_tiles = []
    for tile_idx in range(n_tiles):
        if mx_rows_stream:
            stream = _load_mx_stream_rows(
                hbm, start + tile_idx * _PMAX, _PMAX, f"{name}_t{tile_idx}", after=rows_after if tile_idx == 0 else None
            )
            stream_scores, stream_selection = _mx_stream_scores(stream, router_bias, f"{name}_t{tile_idx}")
            latent_sb = None
        elif latent_gather_mx:
            gathered = _load_mx_scores(hbm, start + tile_idx * _PMAX, _PMAX, rows, f"{name}_t{tile_idx}", router_bias)
            latent_sb = None
        else:
            gathered, latent_sb = load_chunk_latent(
                hbm,
                gather,
                start + tile_idx * _PMAX,
                _PMAX,
                rows,
                f"{name}_t{tile_idx}",
                router_bias,
                rows_latent,
                rows_load_split,
                after=rows_after if tile_idx == 0 else None,
            )
        # The stream rows and the gathered rows take the same affinity options: _local_affinities reads the scores and
        # the selection only through their access patterns (max8, nc_match_replace8, one scalar_tensor_tensor).
        if mx_rows_stream:
            affinities = _local_affinities(
                stream_scores,
                stream_selection,
                group_select[0:_PMAX, :],
                split,
                **({"fused": True} if vector_cuts else {}),
                **local_kwargs,
            )
        else:
            affinities = _local_affinities(
                gathered[:, :, _GATHER_SCORE_COLS[0] : _GATHER_SCORE_COLS[1]],
                gathered[:, :, _GATHER_SELECTION_COLS[0] : _GATHER_SELECTION_COLS[1]],
                group_select[0:_PMAX, :],
                split,
                **({"fused": True} if vector_cuts else {}),
                **local_kwargs,
            )
        affinity_tiles.append(affinities)
        nisa.dma_copy(
            dst=affinity_rows[tile_idx * _PMAX : (tile_idx + 1) * _PMAX, :],
            src=affinities,
            name=f"{name}_t{tile_idx}_aff_store",
            **_HBM_DMA,
        )
        if mx_rows_stream:
            _mx_stream_to_mx(stream, codes, scales, tile_idx * _PMAX, f"{name}_t{tile_idx}")
            continue
        if latent_sb is None:
            continue
        tile_codes, tile_scales = _quantize_natural_latent(
            latent_sb=latent_sb.reshape((_PMAX, LATENT_SIZE)), T=_PMAX, D=LATENT_SIZE
        )
        tile_cols = slice(tile_idx * _PMAX, (tile_idx + 1) * _PMAX)
        nisa.tensor_copy(
            dst=codes.view(nl.uint32)[:, :, tile_cols], src=tile_codes.view(nl.uint32), engine=nisa.engine.gpsimd
        )
        nisa.tensor_copy(dst=scales[:, :, tile_cols], src=tile_scales, engine=nisa.engine.gpsimd)
    tables = _compact_tables(
        affinity_tiles,
        count,
        name,
        late_always=late_always,
        constants=constants,
        hot_last=hot_last,
        **({"skip_unrouted": True} if ungather_skip else {}),
        **({"snake": snake} if snake is not None else {}),
        **({"scalar_copies": True} if tables_scalar else {}),
    )
    # hot_last: the affinity rows of a later slot tile in HBM, read by its expert loop with a runtime offset.
    slot_affinity_hbm = None
    if hot_last and n_tiles > 1:
        slot_affinity_hbm = hbm.get("compact_slot_affinity")
        if slot_affinity_hbm is None:
            slot_affinity_hbm = nl.ndarray(
                (_PMAX, local_experts), dtype=nl.float32, buffer=nl.private_hbm, name=f"moe2s_{name}_slot_aff"
            )
            hbm["compact_slot_affinity"] = slot_affinity_hbm
    gate_up, gate_up_scale, down, down_scale = expert_weights
    if gate_up.dtype == nl.uint16:
        gate_up = gate_up.view(nl.float4_e2m1fn_x4)
    if down.dtype == nl.uint16:
        down = down.view(nl.float4_e2m1fn_x4)
    # vector_cuts: the accumulator fill follows the slot gathers on GpSimd (off the prologue's VectorE chain); with a
    # tensor-indirection slot_gather it follows the slot affinity gather, GpSimd's last slot-tile work.
    gathers = [] if vector_cuts else None
    slot_codes, slot_scales, slot_affinities = _compact_slot_tile(
        codes,
        scales,
        affinity_rows,
        tables,
        0,
        f"{name}_s0",
        **({"issued": gathers} if vector_cuts else {}),
        slot_gather=slot_gather,
    )
    slot_options = loop_options
    if (
        loop_options.get("cross_expert_order")
        and "stream" not in loop_options
        and not loop_options.get("half_weight_buffers")
        and loop_options.get("weight_buffers", 0) < COMPACT_RING_WEIGHT_BUFFERS
    ):
        slot_options = {**loop_options, "weight_buffers": COMPACT_RING_WEIGHT_BUFFERS}
    if vector_cuts:
        slot_options = {**slot_options, "accum_memset": (nisa.engine.gpsimd, gathers[-1] if gathers else None)}
    if up_product_split:
        slot_options = {**slot_options, "up_product_split": True}
    # The static loop's multiply-add of each down H tile reads the FP32 down PSUM tile (moe_tkg_decode psum_consume): no
    # PSUM -> SBUF copy, the same FP32 operands, so the same bits.
    slot_options = {**slot_options, "psum_consume": True}
    if affinity_sbuf_copy:
        slot_options = {**slot_options, "affinity_sbuf_copy": True}
    if alt_tiles:
        # compact_alt_tiles: the static loop's gate sigmoid / up tanh (and eviction) rings, psum_consume kept.
        kernel_assert(
            alt_tiles in STATIC_ALT_TILE_PSUM_CONSUME,
            f"alt_tiles must be one of {STATIC_ALT_TILE_PSUM_CONSUME}, got {alt_tiles!r}",
        )
        slot_options = {**slot_options, "static_alt_tiles": alt_tiles}
    # HOT_RESIDENT_REUSE: the static loop's buffers of its last expert, for the one hot_last later slot tile (with two
    # or more later tiles, a later tile could find another expert's bytes there).
    resident = {} if hot_last and n_tiles == 2 and not empty_later and HOT_RESIDENT_REUSE else None
    moe_tkg_decode(
        None,
        slot_affinities,
        gate_up,
        down,
        gate_up_scale,
        down_scale,
        SITU_BETA,
        SITU_LINEAR_BETA,
        **slot_options,
        output=slot_rows[0:_PMAX, :],
        quantized_latent=(slot_codes, slot_scales),
        **({"resident_out": resident} if resident is not None else {}),
    )
    if resident:
        _HOT_RESIDENT_TRACE["calls"] += 1
    # tail_first (hot_last with one later slot tile): the tail runs before the later tile's guard, as the output of the
    # not-taken path (every routed token in slot tile 0), each token tile's routed partial stored right after its
    # transposes; the guard's branch runs it again after the later tile and overwrites the stores (the IF's fence
    # orders them). Off, the tail runs once after the slot tiles, with one store.
    tail_first = tail_first and hot_last and n_tiles == 2 and not empty_later
    # store_halves (hot_last with one later slot tile): per-tile stores, each in two halves, with or without tail_first.
    tile_stores = (tail_first or (store_halves and hot_last and n_tiles == 2 and not empty_later)) and _PMAX % rows == 0
    # store_halves: the token bounds of the two column-group copies of a tile (_to_column_groups_at), whole replicas.
    half = _PMAX // 2 // _column_group_block_tokens(LATENT_SIZE) * _column_group_block_tokens(LATENT_SIZE)
    half_bounds = ((0, half), (half, _PMAX)) if store_halves else ((0, _PMAX),)
    kernel_assert(
        not store_halves or (tile_stores and half % rows == 0),
        f"store_halves stores each token tile of the hot_last tail (two slot tiles) in whole replicas of {rows} "
        f"rows split at token {half}; got hot_last {hot_last}, {n_tiles} tiles, per-tile stores {tile_stores}",
    )

    tail_gathers = []
    early_words = []  # zero_early: the tail's gather destinations, filled before the guard

    def _tail(tag: str = "", mark: bool = True):
        """The token-order gather of the slot-output rows, the column-group transposes and the routed store."""
        staged = None
        if not tile_stores:
            staged = nl.ndarray((ROUTED_PARTITIONS, count, ROUTED_COLS), dtype=nl.bfloat16, buffer=nl.sbuf)
        for tile_idx in range(n_tiles):
            if ungather_skip and early_words:
                # zero_early: the tile filled before the later slot tile's guard.
                token_rows = early_words[tile_idx].view(nl.bfloat16)
            elif ungather_skip:
                # The unrouted tokens' rows: +0, the bits of the zero row; the gather writes the routed rows only. The
                # tile is FP32, read as its BF16 view: the same zero bytes in half the memset's elements (a memset
                # costs per element).
                token_words = nl.ndarray((_PMAX, LATENT_SIZE // 2), dtype=nl.float32, buffer=nl.sbuf)
                nisa.memset(dst=token_words, value=0.0, engine=nisa.engine.vector)
                token_rows = token_words.view(nl.bfloat16)
            else:
                token_rows = nl.ndarray((_PMAX, LATENT_SIZE), dtype=nl.bfloat16, buffer=nl.sbuf)
            gather = _gather_rows(
                token_rows,
                slot_rows,
                tables["token_offsets"][tile_idx],
                f"{name}_token_rows_t{tile_idx}{tag}",
                **({"skip": True} if ungather_skip else {}),
            )
            tail_gathers.append(gather)
            if tile_stores:
                # This tile's routed partial in its own staged tile, stored now (whole replicas).
                tile_staged = nl.ndarray((ROUTED_PARTITIONS, _PMAX, ROUTED_COLS), dtype=nl.bfloat16, buffer=nl.sbuf)
                _to_column_groups_at(token_rows, tile_staged, 0)
                for first, last in half_bounds:
                    _store_routed_partial(
                        hbm,
                        tile_staged if (first, last) == (0, _PMAX) else tile_staged[:, first:last, :],
                        start + tile_idx * _PMAX + first,
                        last - first,
                        rows,
                        routed_reduce,
                        f"{name}{tag}_t{tile_idx}" + (f"_h{first}" if store_halves else ""),
                        store_queues,
                        routed_layout,
                        **({} if mark else {"mark": False}),
                    )
                continue
            _to_column_groups_at(token_rows, staged, tile_idx * _PMAX)
        if tile_stores:
            return
        _store_routed_partial(
            hbm,
            staged,
            start,
            count,
            rows,
            routed_reduce,
            f"{name}{tag}",
            store_queues,
            routed_layout,
            **({} if mark else {"mark": False}),
        )

    if tail_first:
        _tail()
    if zero_early and ungather_skip and not tail_first and n_tiles > 1 and not empty_later:
        # The tail's gather destinations, zero-filled before the guard (VectorE waits at its barrier anyway), so the
        # first gather after the guard waits for no fill.
        for _tile_idx in range(n_tiles):
            words = nl.ndarray((_PMAX, LATENT_SIZE // 2), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(dst=words, value=0.0, engine=nisa.engine.vector)
            early_words.append(words)
    # empty_later (routed_compact timing_empty_later, a timing control): no later slot tile.
    for slot_tile in range(1, 1 if empty_later else n_tiles):
        trips = nisa.register_alloc()
        nisa.register_load(trips, tables["overflow"][slot_tile - 1])

        def _later_slot_tile(_iteration, slot_tile=slot_tile):
            codes_k, scales_k, affinities_k = _compact_slot_tile(
                codes, scales, affinity_rows, tables, slot_tile, f"{name}_s{slot_tile}", slot_gather=slot_gather
            )
            if hot_last:
                # Only the active experts of the slot tile: one trip of the dynamic loop for each.
                nisa.dma_copy(
                    dst=slot_affinity_hbm,
                    src=affinities_k,
                    dge_mode=nisa.dge_mode.hwdge,
                    engine=nisa.engine.sync,
                    name=f"{name}_s{slot_tile}_aff_store",
                )
                active_experts_in_slots(
                    codes_k,
                    scales_k,
                    affinities_k,
                    slot_affinity_hbm,
                    gate_up,
                    down,
                    gate_up_scale,
                    down_scale,
                    SITU_BETA,
                    SITU_LINEAR_BETA,
                    output=slot_rows[slot_tile * _PMAX : (slot_tile + 1) * _PMAX, :],
                    **({"resident": resident} if resident else {}),
                    **({"static_first": True} if static_first else {}),
                )
                if tail_first:
                    # The taken path: the tail again, with the later slot tile's rows; it overwrites the stores.
                    _tail("_late", mark=False)
                return
            _all_experts_in_slots(
                input_quant_sb=codes_k,
                input_scale_sb=scales_k,
                local_affinities=affinities_k,
                expert_gate_up_weights=gate_up,
                expert_down_weights=down,
                expert_gate_up_weights_scale=gate_up_scale,
                expert_down_weights_scale=down_scale,
                beta=SITU_BETA,
                linear_beta=SITU_LINEAR_BETA,
                T=_PMAX,
                T_physical=_PMAX,
                weight_buffers=COMPACT_LATE_WEIGHT_BUFFERS,
                output=slot_rows[slot_tile * _PMAX : (slot_tile + 1) * _PMAX, :],
            )

        if hot_last:
            # A then-only IF, not a one-trip loop: the compiler keeps the induction register of a loop that holds a
            # nested dynamic loop (the expert loop) live to the end of the program, one register per layer, and a
            # 93-layer program runs out of registers (NCC_IGCA094). The slot tile has no collective.
            nexp.if_else(trips, functools.partial(_later_slot_tile, 0))
        else:
            nl.fori_loop(0, trips, _later_slot_tile)
    if not tail_first:
        _tail()
        if tail_hook is not None:
            # The caller orders its next layer's GpSimd prework after these gathers (decode_block TAIL_POOL_PIN).
            tail_hook(tail_gathers)


def _store_routed_partial(
    hbm: dict,
    staged: nl.NkiTensor,
    start: int,
    count: int,
    rows: int,
    routed_reduce: str,
    name: str,
    store_queues: str = "sp",
    routed_layout: str = "natural",
    mark: bool = True,
):
    """Store the [128, count, 28] BF16 routed partial ``staged`` of gathered tokens ``[start, start + count)`` into the
    routed reduction source of ``routed_reduce``. ``mark`` False leaves the stores out of ``common.mark_ffn_output``.

    ``ar``: the columns of those tokens of ``routed_src`` ``[128, T * 28]``. ``rs_ag``: the replica-major
    ``routed_rs_src`` ``[dp * 128, B * 28]``, token r B + b at row 128 r + p, columns 28 b + c: one DMA of runs of
    B * 28 elements when the call holds whole replicas, else one DMA for each replica of the call. Each store is
    the consumer of the routed output for the next layer's row sums (``common.mark_ffn_output``). ``store_queues``
    (``STORE_QUEUES``): ``sp`` (default) each DMA on the SP queue; ``act`` on the ACT hardware queue; ``split`` each
    DMA as two, partitions 0..63 on SP and 64..127 on ACT: the same bytes.
    """
    if store_queues == "split":
        parts = ((0, _STORE_HALF, _HBM_DMA, "_p0"), (_STORE_HALF, ROUTED_PARTITIONS, _HBM_DMA_ACT, "_p1"))
    else:
        parts = ((0, ROUTED_PARTITIONS, _HBM_DMA_ACT if store_queues == "act" else _HBM_DMA, ""),)
    if routed_reduce == "ar" and routed_layout == "qsplit":
        kernel_assert(mark, f"{name}: a routed_layout 'qsplit' store is always marked (common.mark_ffn_output)")
        _store_routed_qsplit(hbm["routed_src"], staged, start, count, rows, parts, name)
        return
    marker = mark_ffn_output if mark else _no_mark
    if routed_reduce == "ar":
        staged_cols = staged.reshape((ROUTED_PARTITIONS, count * ROUTED_COLS))
        for p0, p1, dma, tag in parts:
            marker(
                nisa.dma_copy(
                    dst=hbm["routed_src"][p0:p1, start * ROUTED_COLS : (start + count) * ROUTED_COLS],
                    src=staged_cols[p0:p1, :],
                    name=f"{name}_routed_store{tag}",
                    **dma,
                )
            )
        return
    width = rows * ROUTED_COLS
    if start % rows == 0 and count % rows == 0:
        replicas = count // rows
        staged_rows = staged.reshape((ROUTED_PARTITIONS, replicas, width))
        for p0, p1, dma, tag in parts:
            marker(
                nisa.dma_copy(
                    dst=hbm["routed_rs_src"].ap(
                        pattern=[[width, p1 - p0], [ROUTED_PARTITIONS * width, replicas], [1, width]],
                        offset=(start // rows) * ROUTED_PARTITIONS * width + p0 * width,
                    ),
                    src=staged_rows[p0:p1, :, :],
                    name=f"{name}_routed_store{tag}",
                    **dma,
                )
            )
        return
    staged_cols = staged.reshape((ROUTED_PARTITIONS, count * ROUTED_COLS))
    for replica_idx in range(start // rows, (start + count - 1) // rows + 1):
        first = max(start, replica_idx * rows)
        last = min(start + count, (replica_idx + 1) * rows)
        row0 = replica_idx * ROUTED_PARTITIONS
        for p0, p1, dma, tag in parts:
            marker(
                nisa.dma_copy(
                    dst=hbm["routed_rs_src"][
                        row0 + p0 : row0 + p1,
                        (first - replica_idx * rows) * ROUTED_COLS : (last - replica_idx * rows) * ROUTED_COLS,
                    ],
                    src=staged_cols[p0:p1, (first - start) * ROUTED_COLS : (last - start) * ROUTED_COLS],
                    name=f"{name}_routed_store_r{replica_idx}{tag}",
                    **dma,
                )
            )


def _no_mark(*instrs):
    """``common.mark_ffn_output`` without the record: return the last of ``instrs``."""
    return instrs[-1] if instrs else None


def _store_routed_qsplit(
    routed_src: nl.NkiTensor, staged: nl.NkiTensor, start: int, count: int, rows: int, parts: tuple, name: str
):
    """Store the [128, count, 28] routed partial ``staged`` of tokens ``[start, start + count)`` into ``routed_src`` in
    the ``qsplit`` column order (``ROUTED_LAYOUTS``): one DMA (runs of B / 2 * 28 elements) when the call holds whole
    replicas, else one DMA for each (replica, half) piece of the call."""
    tokens = routed_src.shape[1] // ROUTED_COLS
    dp, half = tokens // rows, rows // 2
    half_width = half * ROUTED_COLS
    row_width = tokens * ROUTED_COLS
    if start % rows == 0 and count % rows == 0:
        replicas = count // rows
        staged_halves = staged.reshape((ROUTED_PARTITIONS, replicas, 2, half_width))
        for p0, p1, dma, tag in parts:
            mark_ffn_output(
                nisa.dma_copy(
                    dst=routed_src.ap(
                        pattern=[[row_width, p1 - p0], [half_width, replicas], [dp * half_width, 2], [1, half_width]],
                        offset=p0 * row_width + (start // rows) * half_width,
                    ),
                    src=staged_halves[p0:p1, :, :, :],
                    name=f"{name}_routed_store{tag}",
                    **dma,
                )
            )
        return
    staged_cols = staged.reshape((ROUTED_PARTITIONS, count * ROUTED_COLS))
    for replica_idx in range(start // rows, (start + count - 1) // rows + 1):
        for h in range(2):
            first = max(start, replica_idx * rows + h * half)
            last = min(start + count, replica_idx * rows + (h + 1) * half)
            if first >= last:
                continue
            col0 = (
                h * dp * half_width + replica_idx * half_width + (first - replica_idx * rows - h * half) * ROUTED_COLS
            )
            for p0, p1, dma, tag in parts:
                mark_ffn_output(
                    nisa.dma_copy(
                        dst=routed_src[p0:p1, col0 : col0 + (last - first) * ROUTED_COLS],
                        src=staged_cols[p0:p1, (first - start) * ROUTED_COLS : (last - start) * ROUTED_COLS],
                        name=f"{name}_routed_store_r{replica_idx}h{h}{tag}",
                        **dma,
                    )
                )


def _store_parts(dst: nl.NkiTensor, src: nl.NkiTensor, store_queues: str, name: str):
    """Store the 128-partition SBUF tile ``src`` into the HBM tensor ``dst`` of the same shape on ``store_queues``
    (``STORE_QUEUES``): one DMA on SP (``sp``) or ACT (``act``), or two (``split``: partitions 0..63 on SP, 64..127 on
    ACT, names ``<name>_p0`` / ``_p1``)."""
    if store_queues == "split":
        nisa.dma_copy(dst=dst[0:_STORE_HALF], src=src[0:_STORE_HALF], name=f"{name}_p0", **_HBM_DMA)
        nisa.dma_copy(dst=dst[_STORE_HALF:_PMAX], src=src[_STORE_HALF:_PMAX], name=f"{name}_p1", **_HBM_DMA_ACT)
        return
    nisa.dma_copy(dst=dst, src=src, name=name, **(_HBM_DMA_ACT if store_queues == "act" else _HBM_DMA))


def _out_all_reduce(
    src: nl.NkiTensor,
    pair: tuple,
    group,
    name: str,
    after_trigger=None,
    store_queues: str = "sp",
    record_consumer: bool = False,
):
    """``tkg_moe_dp._all_reduce_hbm`` (store, all-reduce, load) with the store on ``store_queues``
    (``STORE_QUEUES``): ``sp`` is ``_all_reduce_hbm`` itself; ``act`` one store DMA on the ACT hardware queue; ``split``
    two, partitions 0..63 on SP and 64..127 on ACT. The same bytes. ``record_consumer`` records the load of the result
    for the ``after_result`` gates of the next body (``release_gate_consumer``; only with the block's consumer gates), on
    every store form."""
    if store_queues == "sp":
        return _all_reduce_hbm(src, pair, group, name, after_trigger, record_consumer=record_consumer)
    src_hbm, dst_hbm = pair
    _store_parts(src_hbm, src, store_queues, f"{name}_store")
    check_pair_group(dst_hbm, group, f"{name}_reduce")
    reduce = release_weight_gate(
        ncc.all_reduce(srcs=[src_hbm], dsts=[dst_hbm], replica_group=group, op=nl.add, name=f"{name}_reduce")
    )
    if after_trigger is not None:
        after_trigger(reduce)
    reduced = nl.ndarray(tuple(src.shape), dtype=src.dtype, buffer=nl.sbuf)
    load = nisa.dma_copy(dst=reduced, src=dst_hbm, name=f"{name}_load", **_HBM_DMA)
    if record_consumer:
        release_gate_consumer(load, engine=_HBM_DMA["engine"])
    return reduced


def _hbm_views(
    sources: dict,
    rows: int,
    split: dict,
    gather: str,
    drop_selection: bool = False,
    routed_reduce: str = "ar",
    stats_gather: bool = False,
    out_proj_form: str = "reduced",
    rows_latent: str = "bf16",
    latent_gather_mx: bool = False,
    mx_scores_align: bool = False,
    mx_rows_stream: bool = False,
) -> dict:
    """Check the tensors of ``moe_2srv_collective_hbm`` against ``rows`` rows, ``gather``, ``drop_selection``,
    ``routed_reduce``, ``out_proj_form``, ``mx_scores_align`` and ``mx_rows_stream``, and
    return them."""
    tokens = split["dp"] * rows
    cols = router_row_cols(drop_selection, rows_latent, latent_gather_mx, mx_scores_align, mx_rows_stream)
    if gather == "dp_tp":
        gathered = {"rows_dst": (tokens, cols), "shard_dst": (TP_SIZE * tokens, cols)}
    else:
        gathered = {"world_dst": (split["world"] * rows, cols)}
    shapes = {"rows_src": (rows, cols), **gathered}
    if routed_reduce == "ar":
        shapes.update(
            routed_src=(ROUTED_PARTITIONS, tokens * ROUTED_COLS), routed_dst=(ROUTED_PARTITIONS, tokens * ROUTED_COLS)
        )
    else:
        shapes.update(_routed_rs_shapes(rows, split["dp"], stats_gather, routed_reduce))
    column = out_proj_form == "column"
    shapes.update(out_src=(_SHARD_PARTITIONS if column else _PMAX, rows * _H_TILES), out_dst=(_PMAX, rows * _H_TILES))
    if column:
        shapes.update(
            shared_src=(_SHARD_PARTITIONS, rows * SHARED_GATHER_COLS), shared_dst=(_PMAX, rows * SHARED_GATHER_COLS)
        )
    for key, shape in shapes.items():
        kernel_assert(
            key in sources and tuple(sources[key].shape) == shape,
            f"The 2-server MoE body needs the HBM tensor {key} {shape} of moe_2srv_collective_hbm({rows} rows, "
            f"{gather!r}, drop_selection={drop_selection}, routed_reduce {routed_reduce!r}, stats_gather {stats_gather}"
            f"{', out_proj_form ' + repr(out_proj_form) if column else ''}, rows_latent {rows_latent!r}, "
            f"latent_gather_mx={latent_gather_mx}, mx_scores_align={mx_scores_align}), "
            f"got {tuple(sources[key].shape) if key in sources else None}",
        )
    return sources


def _routed_rs_shapes(rows: int, dp: int, stats_gather: bool, routed_reduce: str = "rs_ag") -> dict:
    """The shapes of the ``rs_ag`` / ``rs_fold`` tensors of ``moe_2srv_routed_hbm`` for ``rows`` rows of each of
    ``dp`` replicas."""
    width = rows * ROUTED_COLS
    shapes = {"routed_rs_src": (dp * ROUTED_PARTITIONS, width), "routed_rs_dst": (_SHARD_PARTITIONS, width)}
    if routed_reduce == "rs_fold":
        fold = (_PMAX, _fold_cols(rows))
        shapes.update(fold_src=fold, fold_dst=fold)
    elif stats_gather:
        shapes.update(stats_src=(rows, 1), stats_dst=(TP_SIZE * rows, 1))
    else:
        shapes["routed_ag_dst"] = (ROUTED_PARTITIONS, width)
    return shapes


def _validate_inputs(
    x_hT,
    moe_in_proj_dp,
    moe_out_proj_dp,
    moe_vectors_dp,
    routed_store_form: str,
    drop_selection: bool = False,
    out_proj_form: str = "reduced",
    in_proj_scale=None,
    out_proj_scale=None,
    latent_gather_mx: bool = False,
):
    """Check the shapes and dtypes of the body inputs, and the LNC1 Trn3 target. With ``drop_selection``,
    ``moe_vectors_dp`` must hold the router bias of all experts (``ROUTER_BIAS_COLS``): a body without them fails. With
    ``out_proj_form`` ``column`` it must also hold the latent norm weights (``column_norm_cols``). With
    ``latent_gather_mx`` ``moe_in_proj_dp`` holds the 540-column block layout (``G1A_IN_PROJ_COLS``), or the MXFP8 form
    of ``MX_G1A_IN_PROJ_SHAPES`` with ``in_proj_scale``."""
    kernel_assert(
        nl.program_ndim() == 0 or (nl.program_ndim() == 1 and nl.num_programs(0) == 1),
        "The Kimi-K3 2-server MoE decode body supports LNC1 only",
    )
    kernel_assert(nisa.get_nc_version() >= nisa.nc_version.gen4, "The Kimi-K3 2-server MoE decode body requires Trn3")
    kernel_assert(
        len(x_hT.shape) == 3 and x_hT.shape[0] == _PMAX and x_hT.shape[2] == _H_TILES and x_hT.dtype == nl.bfloat16,
        f"x_hT must be BF16 [{_PMAX}, B, {_H_TILES}], got {x_hT.dtype} {x_hT.shape}",
    )
    kernel_assert(
        1 <= x_hT.shape[1] <= MAX_REPLICA_ROWS, f"x_hT has B={x_hT.shape[1]} rows. Use 1 <= B <= {MAX_REPLICA_ROWS}."
    )
    in_proj_cols = G1A_IN_PROJ_COLS if latent_gather_mx else DP_IN_PROJ_COLS
    if in_proj_scale is None:
        kernel_assert(
            tuple(moe_in_proj_dp.shape) == (_PMAX, _H_TILES, in_proj_cols) and moe_in_proj_dp.dtype == nl.bfloat16,
            f"moe_in_proj_dp must be BF16 [{_PMAX}, {_H_TILES}, {in_proj_cols}], got "
            f"{moe_in_proj_dp.dtype} {moe_in_proj_dp.shape}",
        )
    else:
        data_shape, scale_shape = MX_G1A_IN_PROJ_SHAPES if latent_gather_mx else MX_IN_PROJ_SHAPES
        kernel_assert(
            tuple(moe_in_proj_dp.shape) == data_shape
            and moe_in_proj_dp.dtype == nl.uint8
            and tuple(in_proj_scale.shape) == scale_shape
            and in_proj_scale.dtype == nl.uint8,
            f"the MXFP8 moe_in_proj_dp / in_proj_scale must be uint8 {data_shape} / {scale_shape}, got "
            f"{moe_in_proj_dp.dtype} {moe_in_proj_dp.shape} / {in_proj_scale.dtype} {in_proj_scale.shape}",
        )
    if out_proj_scale is None:
        kernel_assert(
            tuple(moe_out_proj_dp.shape) == (DP_OUT_PROJ_ROWS, HIDDEN_SIZE) and moe_out_proj_dp.dtype == nl.bfloat16,
            f"moe_out_proj_dp must be BF16 [{DP_OUT_PROJ_ROWS}, {HIDDEN_SIZE}], got "
            f"{moe_out_proj_dp.dtype} {moe_out_proj_dp.shape}",
        )
    else:
        kernel_assert(
            out_proj_form == "reduced",
            f"the MXFP8 out-projection (out_proj_scale) takes out_proj_form 'reduced', got {out_proj_form!r}",
        )
        data_shape, scale_shape = MX_OUT_PROJ_SHAPES
        kernel_assert(
            tuple(moe_out_proj_dp.shape) == data_shape
            and moe_out_proj_dp.dtype == nl.uint8
            and tuple(out_proj_scale.shape) == scale_shape
            and out_proj_scale.dtype == nl.uint8,
            f"the MXFP8 moe_out_proj_dp / out_proj_scale must be uint8 {data_shape} / {scale_shape}, got "
            f"{moe_out_proj_dp.dtype} {moe_out_proj_dp.shape} / {out_proj_scale.dtype} {out_proj_scale.shape}",
        )
    vector_cols = moe_2srv_vector_cols(drop_selection, out_proj_form)
    kernel_assert(
        tuple(moe_vectors_dp.shape) == (1, vector_cols) and moe_vectors_dp.dtype == nl.float32,
        f"moe_vectors_dp must be FP32 [1, {vector_cols}], got {moe_vectors_dp.dtype} {moe_vectors_dp.shape}"
        + (
            f": drop_selection needs the router bias of all {NUM_EXPERTS} experts in columns {ROUTER_BIAS_COLS}"
            if drop_selection
            else ""
        )
        + (
            f": out_proj_form 'column' needs the {LATENT_SIZE} latent norm weights in columns "
            f"{column_norm_cols(drop_selection)}"
            if out_proj_form == "column"
            else ""
        ),
    )
    kernel_assert(
        routed_store_form in ROUTED_STORE_FORMS,
        f"routed_store_form {routed_store_form!r} must be one of {ROUTED_STORE_FORMS}",
    )


def moe_2srv_glue_selectors(rank_info: nl.NkiTensor, tokens: int, split: dict, name: str, sb=None) -> tuple:
    """Return the ``_selectors`` tiles of this rank for ``tokens`` and ``split``, built once per kernel
    (``selectors_once``): the first call builds them, the later calls with the same ``tokens`` and groups return the
    same tiles. A kernel with an MLA loop path builds them before its first block with a heap allocator ``sb``
    (``tkg_sbuf_alloc.SbufTiles(region, heap=True)``), as ``tkg_moe.rank_selectors``, so they take explicit
    addresses, the work tiles of the build too."""
    cache = getattr(rank_info, "_k3_moe2s_selectors", None)
    if cache is None:
        cache = {}
        object.__setattr__(rank_info, "_k3_moe2s_selectors", cache)
    key = (tokens, split["groups"], *((split["group_shift"],) if split.get("group_shift") else ()))
    if key not in cache:
        cache[key] = _selectors(rank_info, tokens, split, name, sb, all_tiles=True)
    return cache[key]


def _selectors(
    rank_info: nl.NkiTensor, tokens: int, split: dict, name: str, sb=None, all_tiles: bool = False
) -> tuple[nl.NkiTensor, nl.NkiTensor]:
    """Build the latent-shard one-hot and the expert-group one-hot of this rank.

    ``sb`` is None (the default: tiles that the compiler places), or the allocator of the two results (an
    ``tkg_sbuf_alloc.SbufTiles``; ``moe_2srv_selectors``). With ``all_tiles`` it also places the work tiles of the
    build (``moe_2srv_glue_selectors``).

    Returns:
        ``shard_select`` BF16 [128, 4]: element (p, q) is 1 when p == 4 t + q (the routed-layout partitions of latent
        columns [112 t, 112 t + 112)). ``group_select`` FP32 [T, G]: 1 at column (g >> s) & (G - 1), the expert group
        of g, with s = ``split.get("group_shift", 0)``: g % G for s 0 (the served order), g // r for the part-minor
        order of a split of r parts (s = log2 r).
    """
    groups = split["groups"]
    out = AUTO_SBUF if sb is None else sb
    sb = out if all_tiles else AUTO_SBUF
    ranks = sb.tile((_PMAX, 4), nl.int32)
    nisa.dma_copy(
        dst=ranks,
        src=rank_info.ap(pattern=[[0, _PMAX], [1, 4]], offset=0),
        priority=_CRITICAL_DMA_PRIORITY,
        name=f"{name}_ranks",
    )
    ranks_f32 = sb.tile((_PMAX, 4), nl.float32)
    nisa.tensor_copy(dst=ranks_f32, src=ranks)
    shard_first = sb.tile((_PMAX, 1), nl.float32)
    nisa.tensor_scalar(dst=shard_first, data=ranks_f32[:, 2:3], op0=nl.multiply, operand0=float(_SHARD_PARTITIONS))
    # p - q on partition p, column q: 1 where it equals 4 t.
    offsets = sb.tile((_PMAX, _SHARD_PARTITIONS), nl.float32)
    nisa.iota(dst=offsets, pattern=[[-1, _SHARD_PARTITIONS]], offset=0, channel_multiplier=1)
    shard_select = out.tile((_PMAX, _SHARD_PARTITIONS), nl.bfloat16)
    nisa.tensor_scalar(dst=shard_select, data=offsets, op0=nl.equal, operand0=shard_first)

    # G is a power of 2, so g % G is g & (G - 1). A part-minor split shifts the rank first (one more instruction).
    group_shift = split.get("group_shift", 0)
    group = sb.tile((_PMAX, 1), nl.int32)
    if group_shift:
        shifted = sb.tile((_PMAX, 1), nl.int32)
        nisa.tensor_scalar(dst=shifted, data=ranks[:, 0:1], op0=nl.right_shift, operand0=group_shift)
        nisa.tensor_scalar(dst=group, data=shifted, op0=nl.bitwise_and, operand0=groups - 1)
    else:
        nisa.tensor_scalar(dst=group, data=ranks[:, 0:1], op0=nl.bitwise_and, operand0=groups - 1)
    group_f32 = sb.tile((_PMAX, 1), nl.float32)
    nisa.tensor_copy(dst=group_f32, src=group)
    group_idx = sb.tile((tokens, groups), nl.float32)
    nisa.iota(dst=group_idx, pattern=[[1, groups]], offset=0, channel_multiplier=0)
    group_select = out.tile((tokens, groups), nl.float32)
    nisa.tensor_scalar(dst=group_select, data=group_idx, op0=nl.equal, operand0=group_f32[0:tokens, 0:1])
    return shard_select, group_select


def _selectors_key(tokens: int, split: dict) -> str:
    shift = split.get("group_shift")
    return f"_k3_2srv_selectors_{tokens}_{split['groups']}" + (f"_s{shift}" if shift else "")


def moe_2srv_selectors(rank_info: nl.NkiTensor, rows: int, split: dict, sb, name: str = "k3moe2s_sel"):
    """Build this rank's latent-shard and expert-group one-hots (``_selectors``) of the MoE layers of a replica of
    ``rows`` rows once, with the allocator ``sb``, and keep them on ``rank_info`` (as ``tkg_moe.rank_selectors``):
    every later ``latent_moe_decode_2srv`` call with the same ``rank_info`` reuses them in place of its own rank-row
    load and selector ops (``decode_block_body`` ``moe_2srv_selectors_once``). A kernel with an MLA loop path passes a
    heap allocator (``tkg_sbuf_alloc.SbufTiles(region, heap=True)``), so the tiles keep their bytes across the loops.
    The values do not change."""
    tokens = token_chunks(split["dp"] * rows)[0][1]
    key = _selectors_key(tokens, split)
    if getattr(rank_info, key, None) is None:
        object.__setattr__(rank_info, key, _selectors(rank_info, tokens, split, name, sb))
    return getattr(rank_info, key)


def _local_affinities(
    scores,
    selection,
    group_select,
    split: dict,
    fused: bool = False,
    local_index: nl.NkiTensor = None,
) -> nl.NkiTensor:
    """Compute this rank's FP32 [T, E_L] top-16 affinities from the gathered [T, 32, 28] scores and selection values.

    As ``tkg_moe_dp._dp_local_affinities``: the threshold, the selected scores and their sum run over the 896 experts
    in expert order; the affinities of the local experts are the selected scores of the G groups of E_L, weighted by
    the one-hot of the group and summed over the groups (exact). ``fused`` (``prologue_vector_cuts``): one
    scalar_tensor_tensor for each local expert does that weighting (times 1.0, then times the one-hot) and the sum over
    the groups, in place of one [T, E_L, G] multiply and a reduce: the same products, each sum of one product and zeros.
    With ``local_index`` (the UINT32 [1, 1] expert group g of the rank, ``_group_index``; not with ``fused``) one VectorE
    copy reads the E_L selected scores of group g at a runtime offset in place of the weighting and the sum over the G
    groups: the other groups add +0 to a finite value, so the values do not change (finite scores).
    """
    kernel_assert(not (fused and local_index is not None), "local_index replaces the fused weighting: pass one of them")
    tokens = scores.shape[0]
    groups, local_experts = split["groups"], split["local_experts"]
    top8_first = nl.ndarray((tokens, 8), dtype=nl.float32, buffer=nl.sbuf)
    nisa.max8(dst=top8_first, src=selection)
    selection_rest = nl.ndarray((tokens, TP_SIZE, DP_EXPERTS), dtype=nl.float32, buffer=nl.sbuf)
    nisa.nc_match_replace8(dst=selection_rest, data=selection, vals=top8_first, imm=_SELECTION_FLOOR)
    top8_second = nl.ndarray((tokens, 8), dtype=nl.float32, buffer=nl.sbuf)
    nisa.max8(dst=top8_second, src=selection_rest)
    threshold = top8_second[:, 7:8]

    selected_scores = nl.ndarray((tokens, TP_SIZE, DP_EXPERTS), dtype=nl.float32, buffer=nl.sbuf)
    score_sum = nl.ndarray((tokens, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(
        dst=selected_scores,
        data=selection,
        op0=nl.greater_equal,
        operand0=threshold,
        op1=nl.multiply,
        operand1=scores,
        reduce_op=nl.add,
        reduce_res=score_sum,
    )
    nisa.tensor_scalar(dst=score_sum, data=score_sum, op0=nl.add, operand0=ROUTER_SUM_EPS)
    inv_score_sum = nl.ndarray((tokens, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.reciprocal(dst=inv_score_sum, data=score_sum)

    by_group = selected_scores.reshape((tokens, groups, local_experts)).permute((0, 2, 1))
    if local_index is not None:
        # Element (t, e) is selected_scores[t, E_L g + e]: the group axis of the (T, G, E_L) view takes the offset.
        local_selected = nl.ndarray((tokens, local_experts), dtype=nl.float32, buffer=nl.sbuf)
        select = nisa.tensor_copy(
            dst=local_selected,
            src=selected_scores.reshape((tokens, groups, local_experts)).ap(
                pattern=[[groups * local_experts, tokens], [1, local_experts]],
                offset=0,
                scalar_offset=local_index,
                indirect_dim=1,
            ),
            engine=nisa.engine.vector,
        )
    elif fused:
        local_selected = nl.ndarray((tokens, local_experts), dtype=nl.float32, buffer=nl.sbuf)
        weighted_e = nl.ndarray((tokens, groups), dtype=nl.float32, buffer=nl.sbuf)
        for expert in range(local_experts):
            select = nisa.scalar_tensor_tensor(
                dst=weighted_e,
                data=by_group[:, expert, :],
                op0=nl.multiply,
                operand0=1.0,
                op1=nl.multiply,
                operand1=group_select[0:tokens, 0:groups],
                reduce_op=nl.add,
                reduce_res=local_selected[:, expert : expert + 1],
            )
    else:
        weighted = nl.ndarray((tokens, local_experts, groups), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=weighted,
            data1=by_group,
            data2=group_select.ap(pattern=[[groups, tokens], [0, local_experts], [1, groups]], offset=0),
            op=nl.multiply,
        )
        # The keys-off program: the same allocation and instruction order as without the fused form.
        local_selected = nl.ndarray((tokens, local_experts), dtype=nl.float32, buffer=nl.sbuf)
        select = nisa.tensor_reduce(dst=local_selected, op=nl.add, data=weighted, axis=(2,))
    affinities = nl.ndarray((tokens, local_experts), dtype=nl.float32, buffer=nl.sbuf)
    scale = nisa.tensor_scalar(dst=affinities, data=local_selected, op0=nl.multiply, operand0=inv_score_sum)
    # As in tkg_moe_dp: a runtime dependency keeps the scale after the select on the Vector engine.
    if not isinstance(select, nisa.NkiInstruction):
        select = nisa.NkiInstruction(select)
    scale.depends_on(select, kind=nisa.dependency.runtime)
    return affinities


def _latent_norm_rows(
    routed: nl.NkiTensor,
    row_select: nl.NkiTensor,
    shard_select: nl.NkiTensor,
    norm_weight: nl.NkiTensor,
    normed: nl.NkiTensor,
    dp: int,
    name: str,
):
    """Apply the latent RMSNorm to this replica's rows of a chunk and write this rank's normed shard into ``normed``.

    ``normed = bf16(bf16(r32 * rsqrt(sum(r^2) / 3584 + 1e-5)) * w32)``, as ``tkg_moe_dp._dp_latent_norm``.

    Args:
        routed: [128, T * 28] BF16 SBUF, the full routed latent of the chunk; [128, B * 28], this replica's rows,
            when ``row_select`` is None.
        row_select: FP32 [128, dp] replica one-hot (``tkg_moe.rank_selectors``), or None.
        shard_select: BF16 [128, 4] one-hot of the partitions 4 t .. 4 t + 3.
        norm_weight: FP32 [28, 4]: element (c, q) is the latent norm weight of column 112 t + 28 q + c.
        normed: [28, R, 4] BF16 SBUF view: element (c, b, q) gets normed column 112 t + 28 q + c of row b of the chunk.
    """
    rows = normed.shape[1]
    width = rows * ROUTED_COLS
    replica_rows = routed if row_select is None else _replica_rows(routed, row_select, dp, width)

    rows_by_token = replica_rows.reshape((ROUTED_PARTITIONS, rows, ROUTED_COLS))
    routed_sq = nl.ndarray((ROUTED_PARTITIONS, rows, ROUTED_COLS), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=routed_sq, data1=rows_by_token, data2=rows_by_token, op=nl.multiply)
    partial_sums = nl.ndarray((ROUTED_PARTITIONS, rows), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=partial_sums, op=nl.add, data=routed_sq, axis=(2,))
    ones = nl.ndarray((ROUTED_PARTITIONS, ROUTED_COLS), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones, value=1.0)
    total_psum = _psum_bank(ROUTED_COLS, (rows,), nl.float32)
    nisa.nc_matmul(dst=total_psum, stationary=ones, moving=partial_sums, is_stationary_onezero=True)
    inv_rms = nl.ndarray((ROUTED_COLS, rows), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=inv_rms, op=nl.rsqrt, data=total_psum, scale=1.0 / LATENT_SIZE, bias=LATENT_NORM_EPS)

    # Shard of this rank: element (c, b, q) = routed[4 t + q, 28 b + c] (exact one-hot).
    shard_psum = _psum_bank(ROUTED_COLS, (rows, _SHARD_PARTITIONS), nl.float32)
    for row_idx in range(rows):
        nisa.nc_matmul(
            dst=shard_psum[:, row_idx, :],
            stationary=replica_rows[:, row_idx * ROUTED_COLS : (row_idx + 1) * ROUTED_COLS],
            moving=shard_select,
        )
    scaled = nl.ndarray((ROUTED_COLS, rows, _SHARD_PARTITIONS), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=scaled,
        data1=shard_psum,
        data2=inv_rms.ap(pattern=[[rows, ROUTED_COLS], [1, rows], [0, _SHARD_PARTITIONS]], offset=0),
        op=nl.multiply,
    )
    # The weight multiply of each block q is the instruction of tkg_moe_dp, so the values are the same.
    for block_idx in range(_SHARD_PARTITIONS):
        nisa.tensor_scalar(
            dst=normed[:, :, block_idx],
            data=scaled[:, :, block_idx],
            op0=nl.multiply,
            operand0=norm_weight[:, block_idx : block_idx + 1],
        )


def _replica_rows(routed: nl.NkiTensor, row_select: nl.NkiTensor, dp: int, width: int) -> nl.NkiTensor:
    """Return the [128, width] rows of this replica out of the [128, dp * width] routed latent of all replicas."""
    routed_by_replica = routed.reshape((ROUTED_PARTITIONS, dp, width))
    # The rows of this replica: a copy of replica 0, then a predicated copy of each other replica (no 0 x NaN).
    replica_rows = nl.ndarray((ROUTED_PARTITIONS, width), dtype=nl.bfloat16, buffer=nl.sbuf)
    is_replica = nl.ndarray((ROUTED_PARTITIONS, dp), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=is_replica, data=row_select, op0=nl.not_equal, operand0=0.0)
    nisa.tensor_copy(dst=replica_rows, src=routed_by_replica[:, 0, :])
    for replica_idx in range(1, dp):
        nisa.tensor_copy_predicated(
            dst=replica_rows,
            src=routed_by_replica[:, replica_idx, :],
            predicate=is_replica.ap(pattern=[[dp, ROUTED_PARTITIONS], [0, width]], offset=replica_idx),
        )
    return replica_rows


def _replica_index(rank_info: nl.NkiTensor, name: str) -> nl.NkiTensor:
    """Load this rank's DP replica (``rank_info[1]``) as a [1, 1] int32 SBUF tile: the scalar offset of the
    ``routed_own_rows`` load."""
    index = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=index,
        src=rank_info.ap(pattern=[[1, 1], [1, 1]], offset=1),
        priority=_CRITICAL_DMA_PRIORITY,
        name=f"{name}_replica_index",
    )
    return index


def routed_norm_rs_ag(
    hbm: dict,
    norm_weight: nl.NkiTensor,
    dst: nl.NkiTensor,
    world_group: ReplicaGroup,
    tp_group: ReplicaGroup,
    stats_gather: bool,
    name: str,
    after_load=None,
    after_trigger=None,
) -> nl.NkiTensor:
    """Reduce the routed partials of ``hbm`` (``routed_reduce`` ``rs_ag``) and write this rank's normed latent shard of
    its replica's rows into ``dst``; return the [B, 1] FP32 inverse RMS of the rows.

    The world reduce-scatter gives rank (r, t) its block ``routed_rs_dst`` [4, B * 28]: element (q, 28 b + c) is latent
    column 112 t + 28 q + c of row b of replica r. Its token-major view is this rank's shard of the norm input. The
    statistics of the full rows come from the TP32 all-gather of the blocks (``_full_row_rstd``), or, with
    ``stats_gather``, from the TP32 all-gather of each rank's FP32 sums of squares (``_gathered_rstd``).
    ``dst[b, 28 q + c] = bf16(bf16(x * rstd_b) * w)``, the per-element arithmetic of ``_latent_norm_rows``. The FP32
    statistics change partitions only through VectorE transposes (data moves), so B is at most 32.

    Args:
        norm_weight: [B, 112] FP32 SBUF: element (b, 28 q + c) is the latent norm weight of column 112 t + 28 q + c.
        dst: [B, 112] BF16 SBUF view, the latent rows of the out-projection input.
        world_group, tp_group: the world and the TP32 groups (``latent_moe_decode_2srv``).
        after_load: None (default), or a function that is called with the TP32 all-gather (the gate of the next body)
            and the load of its result, right after that load (``_routed_gather_rows``, ``_gathered_rstd``).
        after_trigger: None (default), or a function that is called with the world reduce-scatter right after it.
    """
    rows = dst.shape[0]
    kernel_assert(
        rows <= _VECTOR_TRANSPOSE_MAX,
        f"routed_reduce 'rs_ag' takes at most {_VECTOR_TRANSPOSE_MAX} rows of a replica (one VectorE transpose of its "
        f"statistics), got {rows}",
    )
    width = rows * ROUTED_COLS
    scatter = _routed_reduce_scatter(hbm, world_group, name)
    if after_trigger is not None:
        after_trigger(scatter)
    own = nl.ndarray((rows, _SHARD_PARTITIONS, ROUTED_COLS), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=own,
        src=hbm["routed_rs_dst"].ap(
            pattern=[[ROUTED_COLS, rows], [width, _SHARD_PARTITIONS], [1, ROUTED_COLS]], offset=0
        ),
        name=f"{name}_routed_own_load",
        **_HBM_DMA,
    )
    own_rows = own.reshape((rows, DP_LATENT))
    if stats_gather:
        rstd = _gathered_rstd(own_rows, hbm, tp_group, name, after_load)
    else:
        rstd = _full_row_rstd(_routed_gather_rows(hbm, tp_group, width, name, after_load), rows)
    scaled = nl.ndarray((rows, DP_LATENT), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=scaled, data=own_rows, op0=nl.multiply, operand0=rstd[:, 0:1], engine=nisa.engine.vector)
    nisa.tensor_tensor(dst=dst, data1=scaled, data2=norm_weight, op=nl.multiply, engine=nisa.engine.vector)
    return rstd


def _routed_reduce_scatter(hbm: dict, world_group: ReplicaGroup, name: str):
    """Reduce-scatter ``routed_rs_src`` over ``world_group`` into this rank's block ``routed_rs_dst`` (``rs_ag``);
    return the collective."""
    return ncc.reduce_scatter(
        srcs=[hbm["routed_rs_src"]],
        dsts=[hbm["routed_rs_dst"]],
        replica_group=world_group,
        collective_dim=0,
        op=nl.add,
        name=f"{name}_routed_rs",
    )


def _routed_gather_rows(hbm: dict, tp_group: ReplicaGroup, width: int, name: str, after_load=None) -> nl.NkiTensor:
    """All-gather the ``rs_ag`` blocks of the replica over ``tp_group`` into ``routed_ag_dst`` (the gate of the next
    body) and load it: [128, B * 28] BF16 SBUF, element (p, 28 b + c) is latent column 28 p + c of row b.
    ``after_load`` is None, or a function that is called with the all-gather and the load, right after the load."""
    gather = release_weight_gate(
        ncc.all_gather(
            srcs=[hbm["routed_rs_dst"]],
            dsts=[hbm["routed_ag_dst"]],
            replica_group=tp_group,
            collective_dim=0,
            name=f"{name}_routed_gather",
        )
    )
    full_rows = nl.ndarray((ROUTED_PARTITIONS, width), dtype=nl.bfloat16, buffer=nl.sbuf)
    load = nisa.dma_copy(dst=full_rows, src=hbm["routed_ag_dst"], name=f"{name}_routed_rows_load", **_HBM_DMA)
    if after_load is not None:
        after_load(gather, load)
    return full_rows


FOLD_K_TILES = ((0, DP_LATENT), (DP_LATENT, DP_LATENT + _PMAX), (DP_LATENT + _PMAX, DP_OUT_PROJ_ROWS))
"""The contraction tiles of the ``rs_fold`` out-projection: the 112 latent rows alone (``y_latent``), then the 192
shared rows in tiles of 128 and 64 (``y_shared``), so the norm scale can act on the latent part only."""


def _fold_cols(rows: int) -> int:
    """FP32 columns of the ``rs_fold`` output all-reduce: ``[y_latent | y_shared]`` in hT order and one sum of
    squares for each row (partition 0)."""
    return 2 * rows * _H_TILES + rows


def routed_fold_rs(
    hbm: dict, norm_weight: nl.NkiTensor, dst: nl.NkiTensor, world_group: ReplicaGroup, name: str
) -> nl.NkiTensor:
    """Reduce-scatter the routed partials of ``hbm`` (``routed_reduce`` ``rs_fold``), write this rank's latent shard of
    its replica's rows times the norm weight, ``bf16(x * w)`` with no rstd, into ``dst``, and return the ``[1, B]``
    FP32 sums of squares of the rows over the shard.

    The world reduce-scatter is that of ``rs_ag`` (``routed_rs_dst``, the rank's B rows x its 112 latent columns) and,
    as the routed all-reduce of ``ar``, the gate of the next body's weights. The rstd of each row needs the sums of all
    32 shards: ``fold_out_reduce`` carries them in the output all-reduce and applies the rstd after the out-projection.

    Args:
        norm_weight: [B, 112] FP32 SBUF (``load_norm_weight``): element (b, 28 q + c) is the weight of column
            112 t + 28 q + c.
        dst: [B, 112] BF16 SBUF view, the latent rows of the out-projection input.
    """
    rows = dst.shape[0]
    kernel_assert(
        rows <= _VECTOR_TRANSPOSE_MAX,
        f"routed_reduce 'rs_fold' takes at most {_VECTOR_TRANSPOSE_MAX} rows of a replica (one VectorE transpose of its "
        f"sums), got {rows}",
    )
    width = rows * ROUTED_COLS
    release_weight_gate(
        ncc.reduce_scatter(
            srcs=[hbm["routed_rs_src"]],
            dsts=[hbm["routed_rs_dst"]],
            replica_group=world_group,
            collective_dim=0,
            op=nl.add,
            name=f"{name}_routed_rs",
        )
    )
    own = nl.ndarray((rows, _SHARD_PARTITIONS, ROUTED_COLS), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=own,
        src=hbm["routed_rs_dst"].ap(
            pattern=[[ROUTED_COLS, rows], [width, _SHARD_PARTITIONS], [1, ROUTED_COLS]], offset=0
        ),
        name=f"{name}_routed_own_load",
        **_HBM_DMA,
    )
    own_rows = own.reshape((rows, DP_LATENT))
    squares = nl.ndarray((rows, DP_LATENT), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=squares, data1=own_rows, data2=own_rows, op=nl.multiply, engine=nisa.engine.vector)
    own_sum = nl.ndarray((rows, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=own_sum, op=nl.add, data=squares, axis=(1,))
    sum_row = nl.ndarray((1, rows), dtype=nl.float32, buffer=nl.sbuf)
    nisa.nc_transpose(dst=sum_row, data=own_sum, engine=nisa.engine.vector)
    nisa.tensor_tensor(dst=dst, data1=own_rows, data2=norm_weight, op=nl.multiply, engine=nisa.engine.vector)
    return sum_row


def _load_fold_out_proj(moe_out_proj_dp: nl.NkiTensor, name: str) -> nl.NkiTensor:
    """Load the [304, H] out-projection as [128, 3, H] contraction tiles of ``FOLD_K_TILES``, in chunks along H."""
    out_proj_sb = nl.ndarray((_PMAX, len(FOLD_K_TILES), HIDDEN_SIZE), dtype=nl.bfloat16, buffer=nl.sbuf)
    chunk = HIDDEN_SIZE // _OUT_PROJ_DMA_CHUNKS
    for chunk_idx in range(_OUT_PROJ_DMA_CHUNKS):
        cols = slice(chunk_idx * chunk, (chunk_idx + 1) * chunk)
        for k_idx, (start, stop) in enumerate(FOLD_K_TILES):
            nisa.dma_copy(
                dst=out_proj_sb[0 : stop - start, k_idx, cols],
                src=moe_out_proj_dp[start:stop, cols],
                priority=_CRITICAL_DMA_PRIORITY,
                name=f"{name}_out_proj_c{chunk_idx}_k{k_idx}",
            )
    return out_proj_sb


def fold_out_reduce(
    act_tok: nl.NkiTensor,
    moe_out_proj_dp: nl.NkiTensor,
    sum_row: nl.NkiTensor,
    hbm: dict,
    tp_group: ReplicaGroup,
    name: str,
) -> nl.NkiTensor:
    """Run the out-projection of ``routed_reduce`` ``rs_fold`` and its one FP32 TP32 all-reduce; return the MoE output
    ``bf16(rstd * y_latent + y_shared)`` [128, B, 56] BF16 in the token-major hT layout.

    ``act_tok`` [B, 304] BF16 holds ``bf16(x * w)`` of this rank's latent shard (``routed_fold_rs``) and the shared
    activation. The latent and the shared contraction rows project apart (``FOLD_K_TILES``) into FP32 ``y_latent`` and
    ``y_shared``. One all-reduce over ``tp_group`` of ``[y_latent | y_shared | sums]`` (``fold_src`` / ``fold_dst``,
    the sums of squares of each row on partition 0) gives every rank of the replica both partial sums and the full sums
    of squares. Then ``rstd = rsqrt(sums / 3584 + eps)`` is a per-row scale of ``y_latent``: the out-projection is
    linear, so ``W^T (bf16(x * rstd) * w) ~= rstd * W^T (x * w)``. Not bitwise to ``ar``: the normed latent is not
    rounded to BF16 before the out-projection, and the output sums in FP32 (``ar`` sums BF16 partials).
    """
    rows = act_tok.shape[0]
    width = rows * _H_TILES
    out_proj_sb = _load_fold_out_proj(moe_out_proj_dp, name)
    act_T = _transpose_rows(act_tok, FOLD_K_TILES)
    y_latent = nl.ndarray((_PMAX, rows, _H_TILES), dtype=nl.float32, buffer=nl.sbuf)
    _project_to_hT(
        lambda h: [(out_proj_sb[0:DP_LATENT, 0, h * _PMAX : (h + 1) * _PMAX], act_T[0:DP_LATENT, 0, :])],
        rows,
        partial_hT=y_latent,
    )
    y_shared = nl.ndarray((_PMAX, rows, _H_TILES), dtype=nl.float32, buffer=nl.sbuf)
    _project_to_hT(
        lambda h: [
            (out_proj_sb[0 : stop - start, k_idx, h * _PMAX : (h + 1) * _PMAX], act_T[0 : stop - start, k_idx, :])
            for k_idx, (start, stop) in enumerate(FOLD_K_TILES)
            if k_idx > 0
        ],
        rows,
        partial_hT=y_shared,
    )
    sums = nl.ndarray((_PMAX, rows), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=sums, value=0.0, engine=nisa.engine.vector)
    nisa.tensor_copy(dst=sums[0:1, 0:rows], src=sum_row)
    src, dst = hbm["fold_src"], hbm["fold_dst"]
    nisa.dma_copy(dst=src[:, 0:width], src=y_latent.reshape((_PMAX, width)), name=f"{name}_fold_lat_store", **_HBM_DMA)
    nisa.dma_copy(
        dst=src[:, width : 2 * width], src=y_shared.reshape((_PMAX, width)), name=f"{name}_fold_sh_store", **_HBM_DMA
    )
    nisa.dma_copy(dst=src[:, 2 * width : 2 * width + rows], src=sums, name=f"{name}_fold_sums_store", **_HBM_DMA)
    release_weight_gate(
        ncc.all_reduce(srcs=[src], dsts=[dst], replica_group=tp_group, op=nl.add, name=f"{name}_out_fold_reduce")
    )
    lat = nl.ndarray((_PMAX, rows, _H_TILES), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=lat.reshape((_PMAX, width)), src=dst[:, 0:width], name=f"{name}_fold_lat_load", **_HBM_DMA)
    shared = nl.ndarray((_PMAX, rows, _H_TILES), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=shared.reshape((_PMAX, width)), src=dst[:, width : 2 * width], name=f"{name}_fold_sh_load", **_HBM_DMA
    )
    # Every partition takes the full sums of row 0 of the sum block (a stride-0 partition pattern).
    total = nl.ndarray((_PMAX, rows), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=total,
        src=dst.ap(pattern=[[0, _PMAX], [1, rows]], offset=2 * width),
        name=f"{name}_fold_sums_load",
        **_HBM_DMA,
    )
    rstd = nl.ndarray((_PMAX, rows), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(
        dst=rstd, op=nl.rsqrt, data=total, scale=1.0 / LATENT_SIZE, bias=LATENT_NORM_EPS, engine=nisa.engine.scalar
    )
    scaled = nl.ndarray((_PMAX, rows, _H_TILES), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=scaled,
        data1=lat,
        data2=rstd.ap(pattern=[[rows, _PMAX], [1, rows], [0, _H_TILES]], offset=0),
        op=nl.multiply,
        engine=nisa.engine.vector,
    )
    out = nl.ndarray((_PMAX, rows, _H_TILES), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=out, data1=scaled, data2=shared, op=nl.add, engine=nisa.engine.vector)
    return out


def _full_row_rstd(full_rows: nl.NkiTensor, rows: int) -> nl.NkiTensor:
    """Return the [B, 1] FP32 inverse RMS of the full routed rows ``full_rows`` [128, B * 28] BF16 (element
    (p, 28 b + c)): the sums of ``_latent_norm_rows`` (squares, the 28-column sum of each partition, the ones matmul
    over the partitions), then one VectorE transpose (a data move) to the token-major column."""
    rows_by_token = full_rows.reshape((ROUTED_PARTITIONS, rows, ROUTED_COLS))
    routed_sq = nl.ndarray((ROUTED_PARTITIONS, rows, ROUTED_COLS), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=routed_sq, data1=rows_by_token, data2=rows_by_token, op=nl.multiply, engine=nisa.engine.vector
    )
    partial_sums = nl.ndarray((ROUTED_PARTITIONS, rows), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=partial_sums, op=nl.add, data=routed_sq, axis=(2,))
    ones = nl.ndarray((ROUTED_PARTITIONS, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones, value=1.0, engine=nisa.engine.vector)
    total_psum = _psum_rows(1, rows)
    nisa.nc_matmul(dst=total_psum, stationary=ones, moving=partial_sums, is_stationary_onezero=True)
    rstd_row = nl.ndarray((1, rows), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(
        dst=rstd_row,
        op=nl.rsqrt,
        data=total_psum,
        scale=1.0 / LATENT_SIZE,
        bias=LATENT_NORM_EPS,
        engine=nisa.engine.scalar,
    )
    rstd = nl.ndarray((rows, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.nc_transpose(dst=rstd, data=rstd_row, engine=nisa.engine.vector)
    return rstd


def _gathered_rstd(
    own_rows: nl.NkiTensor, hbm: dict, tp_group: ReplicaGroup, name: str, after_load=None
) -> nl.NkiTensor:
    """Return the [B, 1] FP32 inverse RMS of the rows from this rank's [B, 112] BF16 shard ``own_rows``: the FP32 sum of
    squares of each row over the shard, the TP32 all-gather of those sums, a VectorE transpose (a data move) and their
    sum in TP rank order (the same instructions on the same bytes on every rank of the replica, so the same bits).
    ``after_load`` is None, or a function that is called with the all-gather and the load of its result, right after
    the load."""
    rows = own_rows.shape[0]
    squares = nl.ndarray((rows, DP_LATENT), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=squares, data1=own_rows, data2=own_rows, op=nl.multiply, engine=nisa.engine.vector)
    own_sum = nl.ndarray((rows, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=own_sum, op=nl.add, data=squares, axis=(1,))
    nisa.dma_copy(dst=hbm["stats_src"], src=own_sum, name=f"{name}_stats_store", **_HBM_DMA)
    gather = release_weight_gate(
        ncc.all_gather(
            srcs=[hbm["stats_src"]],
            dsts=[hbm["stats_dst"]],
            replica_group=tp_group,
            collective_dim=0,
            name=f"{name}_stats_gather",
        )
    )
    # Element (t, b): the sum of TP rank t for row b.
    by_rank = nl.ndarray((TP_SIZE, rows), dtype=nl.float32, buffer=nl.sbuf)
    load = nisa.dma_copy(
        dst=by_rank,
        src=hbm["stats_dst"].ap(pattern=[[rows, TP_SIZE], [1, rows]], offset=0),
        name=f"{name}_stats_load",
        **_HBM_DMA,
    )
    if after_load is not None:
        after_load(gather, load)
    by_row = nl.ndarray((rows, TP_SIZE), dtype=nl.float32, buffer=nl.sbuf)
    nisa.nc_transpose(dst=by_row, data=by_rank, engine=nisa.engine.vector)
    total = nl.ndarray((rows, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=total, op=nl.add, data=by_row, axis=(1,))
    rstd = nl.ndarray((rows, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(
        dst=rstd, op=nl.rsqrt, data=total, scale=1.0 / LATENT_SIZE, bias=LATENT_NORM_EPS, engine=nisa.engine.scalar
    )
    return rstd
