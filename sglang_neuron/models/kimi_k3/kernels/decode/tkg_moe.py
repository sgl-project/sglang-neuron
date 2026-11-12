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

"""Kimi-K3 decode latent MoE body and dense MLP body (TP128 and EP128 over 128 ranks, LNC1).

Both bodies take this replica's normalized hidden in the token-major hT layout [128, B, 56] and return
this replica's block output in the same layout.
"""

import inspect
from typing import NamedTuple

import nki.collectives as ncc
import nki.isa as nisa
import nki.language as nl
from nki.collectives import ReplicaGroup

from .utils.kernel_assert import kernel_assert
from .moe.sel_load_tp import PICK_GATHER_MAX_TOKENS, build_down_group_mask
from .moe.sel_load_tp import moe_sel_load_tp_finish as sel_load_finish
from .moe.sel_load_tp import moe_sel_load_tp_start as sel_load_start
from .moe.sel_load_tp import moe_tp_dense_body as tp_dense_routed
from .moe.routed import STATIC_ALT_TILE_FORMS, moe_tkg_decode
from .moe.routed.moe_routed import _to_column_groups
from .utils.trace_names import next_call
from .moe.selective_load import selective_load_tp as kimi_tp
from .moe.selective_load_t32.expert_body_packed import expert_body_packed as kimi_t32_body
from .common import (
    check_pair_group,
    dma_priority,
    is_trn3,
    mark_ffn_output,
    pair_shared_buffer,
    pair_shared_program,
)
from .tkg_weight_dma_gate import release_gate_consumer, release_weight_gate
from .tkg_sbuf_alloc import AUTO_SBUF, ExplicitSbuf, SbufTiles
from .tkg_mx_quant import load_mx_scales, mx_hT_shapes, mx_scale_view, quantize_hT_mx

_PMAX = 128
HIDDEN_SIZE = 7168
# Hidden tiles of the token-major hT layout: element (p, b, j) is hidden index j * 128 + p of token b.
_H_TILES = HIDDEN_SIZE // _PMAX
DP_SIZE = 4
WORLD_SIZE = 128
# Tokens of each DP replica in one body call (an assert bound only). Decode uses B <= 32, so T = 4 B <= 128; the
# verify step runs feed-forward in chunks of up to ffn_rows = 32 rows, so one body call takes at most 32 rows.
MAX_REPLICA_TOKENS = 32

NUM_EXPERTS = 896
LOCAL_EXPERTS = NUM_EXPERTS // WORLD_SIZE
TOP_K = 16
LATENT_SIZE = 3584
LATENT_SHARD = LATENT_SIZE // WORLD_SIZE
SHARED_SIZE = 6144
SHARED_SHARD = SHARED_SIZE // WORLD_SIZE
EXPERT_INTERMEDIATE = 3072
DENSE_INTERMEDIATE = 33792
DENSE_SHARD = DENSE_INTERMEDIATE // WORLD_SIZE

SITU_BETA = 4.0
SITU_LINEAR_BETA = 25.0
LATENT_NORM_EPS = 1e-5
ROUTER_SUM_EPS = 1e-20

# The weights that take the hidden as input are stored hT-tiled, [128, 56, N]: element (p, j, n)
# is row j * 128 + p of the [in, out] weight. Each partition then loads one contiguous block; the
# natural [7168, N] rows are 7168 DMA descriptors of 2 N bytes.
# Column ranges of `moe_in_proj` [128, 56, 131]: the fused projections of the hidden.
ROUTER_COLS = (0, LOCAL_EXPERTS)
LATENT_COLS = (ROUTER_COLS[1], ROUTER_COLS[1] + LATENT_SHARD)
SHARED_GATE_COLS = (LATENT_COLS[1], LATENT_COLS[1] + SHARED_SHARD)
SHARED_UP_COLS = (SHARED_GATE_COLS[1], SHARED_GATE_COLS[1] + SHARED_SHARD)
MOE_IN_PROJ_COLS = SHARED_UP_COLS[1]
MXFP8_IN_PROJ_COLS = MOE_IN_PROJ_COLS - LOCAL_EXPERTS
"""The columns of the MXFP8 ``moe_in_proj`` (``in_proj_scale``): the latent down and the shared gate / up columns. The
FP16 router gives the logits, so the router columns are left out; column ``c`` is column ``c + 7`` of the BF16 form."""
MX_IN_PROJ_SHAPES = mx_hT_shapes(HIDDEN_SIZE, MXFP8_IN_PROJ_COLS)
"""The ``(data, scales)`` shapes of the MXFP8 ``moe_in_proj`` and its ``in_proj_scale`` (``tkg_mx_quant``)."""
_MX_IN_PROJ_DMA_GROUPS = 2
"""The MX groups of 512 hidden rows that one DMA of the MXFP8 in-projection loads."""
# Row ranges of `moe_out_proj` [76, 7168]: the latent up rows, then the shared down rows. One
# matmul with K = 76 computes both projections for one hidden tile.
LATENT_UP_ROWS = (0, LATENT_SHARD)
SHARED_DOWN_ROWS = (LATENT_SHARD, LATENT_SHARD + SHARED_SHARD)
MOE_OUT_PROJ_ROWS = SHARED_DOWN_ROWS[1]
# Column ranges of `moe_vectors` [1, 35] (FP32): this rank's router correction bias and this
# rank's latent norm weight.
LOCAL_BIAS_COLS = (0, LOCAL_EXPERTS)
LATENT_NORM_COLS = (LOCAL_BIAS_COLS[1], LOCAL_BIAS_COLS[1] + LATENT_SHARD)
MOE_VECTOR_COLS = LATENT_NORM_COLS[1]

ROUTED_KERNELS = ("static", "sel_load", "tp_dense")
"""Routed-expert kernels of the latent MoE body. The choice is a trace-time argument (``routed_kernel``).

- ``static``: EP128, ``moe_tkg_decode`` with the static loop over the 7 local experts.
- ``sel_load``: TP128, the selective-load kernel on this rank's I-slice of the 16 routed experts of each token.
- ``tp_dense``: TP128, the dense kernel on the same packed slices: every expert slice once per call, for a large T.

A deployment keeps one weight layout (EP128 or TP128) and picks the kernel of each call from its T (a trace-time
constant of the NEFF): for example sel_load below a threshold T and tp_dense above it on the TP128 layout.
"""
TP128_KERNELS = ("sel_load", "tp_dense")
"""Routed kernels on the TP128 packed layout (``SEL_LOAD_WEIGHT_NAMES``); the others use the EP128 layout."""
DYNAMIC_EXPERT_LOOP_MAX_TOKENS = 8
"""The largest MoE call (T, the tokens of the 4 DP replicas) of the opt-in dynamic expert loop
(``dynamic_expert_loop``). A decode step has T = 4 B, so the loop applies at B = 1 and B = 2 sequences for each
replica.

The limit reads only when the opt-in is on, so the NEFFs without it do not depend on it. At T = 8 the busiest of the
128 ranks sees 3.7 active local experts on average with uniform routing: the loop costs less there than
the static loop (trn3, one call). Above T = 8 the static loop is faster."""
MOE_WEIGHT_NAMES = (
    "moe_in_proj",
    "moe_out_proj",
    "moe_vectors",
    "expert_gate_up_weights",
    "expert_gate_up_weights_scale",
    "expert_down_weights",
    "expert_down_weights_scale",
)
"""Weights of the ``static`` routed kernel (EP128), in the positional order of ``latent_moe_decode``."""
SEL_LOAD_WEIGHT_NAMES = (
    "moe_in_proj",
    "moe_out_proj",
    "moe_vectors",
    "expert_gate_up_packed",
    "expert_down_weights",
    "expert_down_weights_scale",
)
"""Weights of the ``sel_load`` routed kernel. ``expert_gate_up_packed`` takes the ``expert_gate_up_weights`` slot of
``latent_moe_decode``, and the ``expert_gate_up_weights_scale`` slot is unused (the scales are in the packed rows)."""
ROUTED_REDUCE_DTYPES = ("bf16", "fp32")
"""Add types of the routed all-reduce (``routed_reduce_dtype``)."""
ROUTER_DTYPES = ("bf16", "fp16", "fp16_local")
"""Router matmul types (``router_dtype``): ``bf16`` takes the router columns of the BF16 in-projection (the base bits);
``fp16`` multiplies an FP16 copy of the normed hidden (from the FP32 norm values) by an FP16 router weight, with FP32
PSUM. FP16 keeps 3 more mantissa bits of the router input, so the top-16 set is closer to the FP32 router.

``fp16`` sends the FP16 copy of the tokens in the DP token gather. ``fp16_local`` gives the same logits from the
FP16 copy of this replica's tokens: each rank computes the logits of the experts of its 4 DP peers
(``DP_PEER_EXPERTS``) for its own tokens, and the DP token gather carries these logits instead of the FP16 copy."""
AFFINITY_SCALINGS = ("pre_quant", "post_down")
BASE_ROUTER_DTYPES = ("bf16", "fp16_local")
"""Router types of the base MoE body (the EP128 static loop): ``fp16_local`` sends the local logits in the packed rows
of the DP x gather (``moe_collective_tiles`` ``x_logits``), so it is no MoE option (``moe_options_in_use``)."""
# The down rows of each expert that the kimi body reads (the padded [E, 8, H] form, not compact_down).
_PADDED_DOWN_ROWS = 8
# The MoE call size of the kimi_t32 body (sel_load option kimi_t32_body).
KIMI_T32_TOKENS = 32
# The kimi_t32 body form for verify rows, without the prefetch: repeated experts load once ("dynamic"), 19 static
# groups and the blocks (4, 13). It adds the head and tail options, the gate/up pair-gather ring with its PE order
# and the store-DMA fold, the flat token rank, and the weights arena (one DGE table entry per call). The form has
# no prefetch: with a late latent the prefetch gives no gain, and the prefetch does not combine with the two-phase
# call. Block positions in token order with fp32 affinities keep the per-call
# rel-L2 vs fp64 below the served 32-token call's.
_KIMI_T32_OPTIONS = {
    "dedupe_experts": "dynamic",
    "dedupe_static_groups": 19,
    "dedupe_block_groups": (4, 13),
    "dedupe_block_mode": "alternatives",
    # head and tail options: the same outputs, a faster head and tail.
    "latent_scratch": "heap",
    "dedupe_late_trips": True,
    "dedupe_fold_first": True,
    "setup_memsets": "dve",
    "dedupe_onehot_parts": True,
    "skip_down_padding": True,
    # the served numerics: token-major block positions, fp32 affinities.
    "dedupe_order": "token",
    "affinity_dtype": "fp32",
    # the gate/up pair-gather ring, its PE order and the fold in the store DMA.
    "gate_up_pair_gathers": True,
    "pair_head_in_ring": True,
    "gate_up_slots": 6,
    "moving_slots": 2,
    "act_slots": 1,
    "down_lag": 2,
    "fold_dma": True,
    # one SBUF arena for the gate/up rows, down weights and down scales: one DGE table entry per call.
    "weights_arena": True,
    # the flat token rank (fewer head ops).
    "dedupe_token_rank": "flat",
}
KIMI_T32_FORMS = {
    "base": {},
    "lean": {"gate_up_slots": 4},
    "lean2": {"gate_up_slots": 4, "down_slots": 2},
    "lean3": {"gate_up_slots": 4, "down_slots": 2, "latent_scratch": "ring"},
    "fine": {"dedupe_block_groups": (2, 4, 6, 8, 13)},
    "s23": {"dedupe_static_groups": 23, "dedupe_block_groups": (2, 4, 9)},
    "s25": {"dedupe_static_groups": 25, "dedupe_block_groups": (2, 7)},
    "one": {"dedupe_block_groups": (13,)},
    "s17": {"dedupe_static_groups": 17, "dedupe_block_groups": (2, 15)},
    "probe": {"dedupe_static_groups": 11, "dedupe_block_groups": (2, 4, 6, 8, 21)},
    "leanone": {"gate_up_slots": 4, "dedupe_block_groups": (13,)},
    "lean2one": {"gate_up_slots": 4, "down_slots": 2, "dedupe_block_groups": (13,)},
    "skip": {"skip_empty_slots": True},
    "s11": {"dedupe_static_groups": 11, "dedupe_block_groups": (4, 21)},
    "s13": {"dedupe_static_groups": 13, "dedupe_block_groups": (2, 19)},
    "p7": {"dedupe_static_groups": 7, "dedupe_block_groups": (4, 25)},
    "p9": {"dedupe_static_groups": 9, "dedupe_block_groups": (2, 23)},
    "probe16": {"dedupe_static_groups": 5, "dedupe_block_groups": (2, 4, 6, 8, 27)},
    "probe2": {"dedupe_static_groups": 13, "dedupe_block_groups": (2, 4, 6, 8, 10, 12, 19)},
    # MEASURED ("probe2", 128K B=8, real weights, 32 diverse prompts, 7 sampled steps): the head of each MoE layer.
    "tdiv": {
        "layer_static_groups": (
            25, 23, 21, 19, 23, 25, 23, 23, 23, 23, 23, 15, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23,
            23, 23, 23, 23, 21, 21, 23, 21, 23, 21, 21, 21, 19, 19, 21, 21, 19, 15, 19, 19, 19, 15, 17, 21,
            19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 21, 19, 21, 21, 23, 21, 21, 21, 21, 21, 21, 21, 21, 21,
            19, 19, 21, 21, 21, 21, 21, 21, 19, 21, 21, 21, 21, 21, 19, 21, 21, 21, 21, 23,
        ),
        "layer_block_steps": (4,),
    },
}
"""The forms of the kimi_t32 body (sel_load option kimi_t32_form): the options that replace _KIMI_T32_OPTIONS.

"base" (the default) keeps them. "lean", "lean2" and "lean3" use less SBUF for each partition, so the body can fit
beside the compiler-placed tiles of a decode NEFF without the MLA loop path (16K): "lean" keeps 4 gate/up row slots
(24,576 bytes less), "lean2" also 2 down slots (14,336 bytes less), and "lean3" also puts the latent scratch on the
gate/up ring (8,960 bytes less; the first gate/up gather then waits for the latent quantization).

"fine", "s23" and "s25" change the dedupe table for the decode of 32 independent sequences, where a call has more
distinct experts than a verify call (uniform top-16 routing: 393 of 512 positions, 24.5 groups). The head runs
dedupe_static_groups groups, then the smallest block of dedupe_block_groups that holds the other distinct experts:
"fine" runs 19, 21, 23, 25, 27 or 32 groups, "s23" 23, 25, 27 or 32, and "s25" 25, 27 or 32. "one" runs 19 or 32
groups with one block (one loop boundary in place of two), and "s17" 17, 19 or 32. "probe" (a measurement form) runs
11, 13, 15, 17, 19 or 32 groups: the count of MX matmuls in a device profile (21 for each group) then gives the bin of
the distinct experts of each layer. "leanone" and "lean2one" are "lean" and "lean2" with the block of "one".
"skip" gives the empty positions after the distinct experts out-of-bounds gather indices (``skip_empty_slots``), so
their gathers move no bytes (in place of the rows of expert 0); their groups still run on the engines. "s11" runs 11,
15 or 32 groups and "s13" 13, 15 or 32 groups. MEASURED ("probe", 128K B=8, real weights, 32 GSM8K chat prompts): the
distinct experts of a 32-token call fill at most 11 groups in 83 % of the layer calls, 13 in 11 %, 15 in 5 % and 17 in
1 % (uniform routing needs 25 groups), so "s11" runs 11 groups in most layers. "p7" (7, 11 or 32 groups), "p9" (9, 11
or 32) and "probe16" (5, 7, 9, 11, 13 or 32: a measurement form) are tables for a 16-token call, which pads to 32 rows
and has at most 256 distinct experts (16 groups). "probe2" (a measurement form for diverse traffic) runs 13, 15, 17,
19, 21, 23, 25 or 32 groups. "tdiv" is a per-layer table (``_kimi_t32_layer_options``): MEASURED ("probe2", 32 diverse
served-like prompts with sampled steps), a call needs 19 to 23 groups in most layers (uniform routing: 24.5), so each
layer gets the head of its largest measured need plus 2 groups, less the step of 4.

The outputs of all forms are the same for the same distinct experts, except for the sum order of the groups. A form
value can also join names with "+" (``kimi_t32_form_options``), for example "lean2+one+skip"."""


_KIMI_T32_POSITION_GROUPS = KIMI_T32_TOKENS * TOP_K // 16
"""The dedupe position groups of one 32-token call (16 positions each): the head and the largest block add up to it."""


def _kimi_t32_layer_options(options: dict) -> dict:
    """Return the options of this kimi_t32 call with the dedupe table of its MoE layer (a per-layer form).

    A form with ``layer_static_groups`` (one odd head size for each MoE layer, in layer order) gives the call of MoE
    layer l the head ``layer_static_groups[l]`` and the alternatives ``layer_block_steps`` (the sizes below the rest
    of the 32 groups, default (4,)) plus the rest. The layer is the call number of this trace (one call for each
    layer). A form without ``layer_static_groups`` keeps its options. For example, a layer value of 13 with the
    default steps gives the head 13 and the alternatives (4, 19).
    """
    heads = options.pop("layer_static_groups", None)
    steps = tuple(options.pop("layer_block_steps", (4,)))
    if heads is None:
        return options
    layer = next_call("kimi_t32_layer")
    head = heads[min(layer, len(heads) - 1)]
    rest = _KIMI_T32_POSITION_GROUPS - head
    kernel_assert(
        head % 2 == 1 and 0 < rest,
        f"the kimi_t32 layer table needs an odd head below {_KIMI_T32_POSITION_GROUPS} groups, got {head}",
    )
    blocks = tuple(size for size in steps if size < rest) + (rest,)
    return {**options, "dedupe_static_groups": head, "dedupe_block_groups": blocks}


def kimi_t32_form_options(form: str) -> dict:
    """Return the options of the kimi_t32 form ``form``: one name of ``KIMI_T32_FORMS``, or names joined by "+".

    A later name replaces the options of an earlier one.

    Example:
        >>> kimi_t32_form_options("lean2+one")["dedupe_block_groups"]
        (13,)
    """
    names = form.split("+")
    kernel_assert(
        all(part in KIMI_T32_FORMS for part in names),
        f"sel_load option kimi_t32_form {form!r} has a name that is not in {sorted(KIMI_T32_FORMS)}",
    )
    options = {}
    for part in names:
        options.update(KIMI_T32_FORMS[part])
    return options


# The kimi_t32 body form of a 16-token MoE call (4 DP replicas x 4 sequences), padded to one 32-token call: the
# 16 padding rows have zero selected scores, so they add no experts. At most 256 distinct experts (16 groups) exist,
# so a head of 13 static groups and the alternatives of 2 or 4 more groups hold every case (the 19-group block never
# runs). Uniform top-16 routing gives about 224 distinct experts (14 groups) at T = 16.
_KIMI_T32_PAD16_OPTIONS = {"dedupe_static_groups": 13, "dedupe_block_groups": (2, 4, 19)}
# The token counts of a MoE call that the kimi_t32 body takes: 16 (padded to 32), 32 and 64 (two 32-token calls).
KIMI_T32_CALL_TOKENS = (16, 32, 64)
# The kimi body form of a 4-token call (sel_load option kimi_t4_body): the long-stream schedule of the 4 groups with
# the fast first gather and the last-group tail options. Its output bits equal the served 4-token call's.
_KIMI_T4_OPTIONS = {
    "chunk_tokens": kimi_tp.SHORT_TOKENS,
    "long_stream": True,
    "sbuf_reserve": 32768,
    "long_stream_short": True,
    "fast_first_gather": True,
    "last_gate_up_split": True,
    "last_relayout_runs": 2,
    "last_swiglu_first": True,
}
ROUTER_MODES = ("fused", "split")
"""Router forms of the latent MoE body (``router_mode``):

- ``fused``: one world all-gather carries the latent shard and the router values (default).
- ``split``: the router runs first and its values have their own world all-gather; the latent gather follows. The
  expert index then exists one gather earlier, so the ``sel_load`` weight loads overlap the latent gather. The
  values are the same bits as with ``fused``.
"""
ROUTING_FORMS = (
    "base",
    "index_first",
    "selection_first",
    "index_first_pe32",
    "index_first_pe32_psum",
    "index_first_rows",
    "index_first_block",
)
"""Forms of the TP128 top-16 routing:

- ``base``: the threshold, the selected scores and their sum reciprocal, then the index.
- ``index_first`` (the default of the TP128 kernels): the same instructions, with the expert index before the score
  sum reciprocal and the affinities, and the latent copy after the index. The same bits as ``base``.
- ``selection_first``: the expert index from the positions of the 16 largest selection values and a sort of their
  16 scores (``_top16_selection_first``). The same bits as ``base`` unless more than 16 experts reach the threshold,
  two routed scores are equal or a selected score is 0.
- ``index_first_pe32``: ``index_first`` with the router values of the HBM world gather (router form ``fused``) loaded
  as rank rows ([128, T * 28], one contiguous DMA ahead of the strided load) and moved to the token partitions by
  TensorE transposes of each FP32 word (bit-accurate on trn3, ``_router_from_rank_rows``). The strided load writes
  short rows to T partitions and lands late after the gather at T = 16; the routing does not wait for it. The
  same bits as ``index_first``.
- ``index_first_pe32_psum``: ``index_first_pe32`` with the selection transposes first on TensorE, and the routing reads
  the selection values straight from their PSUM tile (no copies to SBUF; the scores keep theirs). The same bits as
  ``index_first``.
- ``index_first_rows``: ``index_first`` for ``in_proj_form`` ``dp_local`` at one token per DP replica (the B = 1
  decode bucket). One DMA loads the router words of the HBM world gather token-major ([T, 128, 14]: shard g in natural
  order, its 7 scores then its 7 selection values; ``dp_local_route_rows``) ahead of the latent rows, with no rank rows,
  transposes or copies, and the DP x gather waits for that load. The routing reads the values and positions of the
  strided load, so the same bits as ``index_first``.
- ``index_first_block``: ``index_first_rows`` with the whole world gather of each token in one DMA element: at one token
  per DP replica the 128 rows of token r are one contiguous 14,336-byte block of the gather destination (in the shard
  order 4 t + d), so one DMA of 4 elements loads them all (``dp_local_token_block``), with no rank rows, router-word or
  token-row loads (each of those is 512 elements of 112 or 56 bytes). Two Vector copies move the selection values and the
  scores to natural-order [T, 128, 7] tiles (shard g = 32 d + t), and the latent copy reads the latent words in that
  order too. The same values in the same positions, so the same bits as ``index_first``.
"""
# The token-major forms at one token per DP replica (``in_proj_form`` ``dp_local``).
_B1_ROUTING_FORMS = ("index_first_rows", "index_first_block")
# The forms that load the router values as rank rows (``_router_from_rank_rows``).
_PE32_FORMS = ("index_first_pe32", "index_first_pe32_psum")
# TP128 expert slices of the sel_load kernel: rank g owns intermediate rows [24 g, 24 g + 24) of every expert.
EXPERT_SHARD = EXPERT_INTERMEDIATE // WORLD_SIZE
SEL_LOAD_DOWN_ROWS = 8
"""MX rows of one down slice: 8 rows x 4 lanes hold the 24 rows of the slice (row p, lane q = row 6 q + p)."""
SEL_LOAD_COMPACT_DOWN_ROWS = EXPERT_SHARD // 4
"""HBM rows of one down slice in the compact form: only the 6 rows that the kernel reads (the same output)."""
SEL_LOAD_ROW_BYTES = 4 * EXPERT_SHARD * (LATENT_SIZE // 512 + 1)
"""Bytes of one packed gate/up row: the E2M1 x4 gate and up weights [2][7][24], then one 96-byte scale segment."""
DENSE_WEIGHT_NAMES = (
    "dense_gate_up",
    "dense_down",
)


def moe_weight_names(routed_kernel: str) -> tuple[str, ...]:
    """Return the weight names of the latent MoE body for a routed kernel (``ROUTED_KERNELS``)."""
    kernel_assert(
        routed_kernel in ROUTED_KERNELS, f"routed_kernel must be one of {ROUTED_KERNELS}, got {routed_kernel}"
    )
    return SEL_LOAD_WEIGHT_NAMES if routed_kernel in TP128_KERNELS else MOE_WEIGHT_NAMES


def moe_weight_slots(weights, routed_kernel: str) -> tuple:
    """Return the 7 positional weight arguments of ``latent_moe_decode`` from a name -> tensor mapping.

    The ``sel_load`` weights fill the slots of the static layout: the packed gate/up rows take the gate/up weight
    slot, and the gate/up scale slot gets None.
    """
    if routed_kernel not in TP128_KERNELS:
        return tuple(weights[name] for name in moe_weight_names(routed_kernel))
    return (
        weights["moe_in_proj"],
        weights["moe_out_proj"],
        weights["moe_vectors"],
        weights["expert_gate_up_packed"],
        None,
        weights["expert_down_weights"],
        weights["expert_down_weights_scale"],
    )


# Each rank sends, for each token, one FP32 row of 28 words: its 28 BF16 latent values packed
# in words 0..13, the router scores of its 7 experts, and their selection values (score + bias).
_GATHER_LATENT_WORDS = (0, LATENT_SHARD // 2)
_GATHER_SCORE_COLS = (_GATHER_LATENT_WORDS[1], _GATHER_LATENT_WORDS[1] + LOCAL_EXPERTS)
_GATHER_SELECTION_COLS = (_GATHER_SCORE_COLS[1], _GATHER_SCORE_COLS[1] + LOCAL_EXPERTS)
_GATHER_COLS = _GATHER_SELECTION_COLS[1]
SCORE_GATHER_COLS = _GATHER_SELECTION_COLS[0]
"""Words of a world gather row with ``router_rows_drop_selection``: the 14 latent words and the 7 scores. Each rank
adds the router bias of all 896 experts (``ALL_ROUTER_BIAS_COLS``) to the gathered scores (``_add_router_bias``)."""
ALL_ROUTER_BIAS_COLS = (MOE_VECTOR_COLS, MOE_VECTOR_COLS + NUM_EXPERTS)
"""Columns of ``moe_vectors`` with ``router_rows_drop_selection``: the router correction bias of all 896 experts in
expert order (expert 7 g + e at column 35 + 7 g + e), after the 35 columns of the base layout. Every rank holds the
same values."""
MX_BLOCK = 32
"""Latent columns of one MXFP8 block: ``quantize_mx`` scales 8 partitions x 4 elements, and the gate/up weights of the
experts hold one E8M0 scale for each 32 contiguous latent rows."""
MX_LATENT_BLOCKS = LATENT_SIZE // MX_BLOCK
"""Ranks that own a latent block with ``latent_gather_mx``: rank r < 112 computes latent columns [32 r, 32 r + 32);
ranks 112..127 compute no latent column (zero weight columns) and send zero filler."""
MX_IN_PROJ_COLS = MOE_IN_PROJ_COLS + MX_BLOCK - LATENT_SHARD
"""Columns of ``moe_in_proj`` with ``latent_gather_mx``: router 0..6, the latent block 7..38, shared gate 39..86 and
shared up 87..134 (``moe_decode_torch.latent_moe_mx_rank_weights``)."""
MX_LATENT_COLS = (ROUTER_COLS[1], ROUTER_COLS[1] + MX_BLOCK)
_MX_COL_SHIFT = MOE_IN_PROJ_COLS - MX_IN_PROJ_COLS
_MX_CODE_ROWS = MX_BLOCK // 4
"""Rows of the per-rank gather block with ``latent_gather_mx``: row i holds the 4 MXFP8 codes of latent columns 4 i ..
4 i + 3 of the block for each token, so rank r's row i is row 8 r + i = 128 h + p of the gathered table, partition p of
latent tile h = r // 16 of the MX quantize layout (``moe_routed._quantize_natural_latent``)."""
_MX_SCALE_ROW = _MX_CODE_ROWS - 1
# Elements of one partition in an MX block, and the partitions of each 32-partition quadrant that hold the E8M0 scales of
# its 4 blocks (``quantize_mx``: block k of a tile at partition 32 (k // 4) + k % 4).
_MX_Q = 4
_MX_SCALES_PER_QUADRANT = 4
# The largest transpose of the Vector engine (partitions and free elements).
_VECTOR_TRANSPOSE_MAX = 32
# The router rows of fp16_local: the 7 experts of each of the 4 DP peers of this TP rank (DP rank order).
DP_PEER_EXPERTS = DP_SIZE * LOCAL_EXPERTS
# fp16_local: the DP token gather sends each token's 56 BF16 hidden words, then its FP32 logits of the DP peer
# experts as 2 words on partitions 0..27.
_TOKEN_LOGIT_WORDS = 2
_TOKEN_GATHER_WORDS = _H_TILES + _TOKEN_LOGIT_WORDS
# The split router form gathers [scores 7 | selection 7] first, then the 14 latent words.
_ROUTER_GATHER_COLS = 2 * LOCAL_EXPERTS
_LATENT_GATHER_WORDS = LATENT_SHARD // 2
# The in-projection columns after the router columns: latent, shared gate and shared up.
_LATENT_SHARED_COLS = MOE_IN_PROJ_COLS - ROUTER_COLS[1]
IN_PROJ_FORMS = ("gathered", "dp_local")
"""Forms of the TP128 in-projection of the latent MoE body (``in_proj_form``):

- ``gathered`` (default): the rank's router and latent-down columns (``moe_in_proj``) run on the tokens of the 4 DP
  replicas, after the DP x gather.
- ``dp_local``: the rank runs the router and latent-down columns of its 4 DP peer shards (``peer_in_proj``) on the
  tokens of its own replica, and sends them in the world gather at once, so the world gather does not wait for the
  DP x gather. Each value is the one that the peer computes in ``gathered`` (the same bits). The HBM world gather
  carries them in another row order, and the loads give the router values and the latent the layout and the expert
  order of ``gathered``, so the routing, the latent and the output keep their bits. The DP x gather then feeds only
  the shared expert: it is issued after the routed kernel (between the world gather and the routed all-reduce on the
  collective stream), and its load and the shared in-projection (``moe_in_proj`` holds only the shared columns) run
  after the routed experts, in the flight of the routed all-reduce. TP128 kernels, the
  fused BF16 router, DP4, and the HBM world and x gathers (the sites of the MLA loop path) only.
"""
TP_SIZE = WORLD_SIZE // DP_SIZE
# dp_local weights of rank 32 d + t (``moe_decode_torch.latent_moe_dp_local_rank_weights``):
# - peer_in_proj [128, 56, 140] BF16 (hT-tiled): column 7 d' + e is router row 7 (32 d' + t) + e, column 28 + 28 d' + k
#   is latent-down column 28 (32 d' + t) + k, for the DP peers d' = 0..3 in that order;
# - moe_in_proj [128, 56, 96]: the shared gate and up columns of the rank (columns 35..130 of the gathered form);
# - moe_vectors [1, 63] FP32: the 35 columns of the gathered form, then the router biases of the 28 peer experts.
PEER_ROUTER_COLS = (0, DP_PEER_EXPERTS)
PEER_LATENT_COLS = (PEER_ROUTER_COLS[1], PEER_ROUTER_COLS[1] + DP_SIZE * LATENT_SHARD)
PEER_IN_PROJ_COLS = PEER_LATENT_COLS[1]
SHARED_IN_PROJ_COLS = MOE_IN_PROJ_COLS - SHARED_GATE_COLS[0]
PEER_BIAS_COLS = (MOE_VECTOR_COLS, MOE_VECTOR_COLS + DP_PEER_EXPERTS)
DP_LOCAL_VECTOR_COLS = PEER_BIAS_COLS[1]
GLUE_DMA_FORMS = ("replica_rows", "sp_weights", "split_gather")
"""Opt-in forms of the small DMAs between the stages of the TP128 latent MoE body (``glue_dmas``, a tuple of these
names). Neither changes a value:

- ``replica_rows``: the load of the output all-reduce result reads only the B tokens of this replica, at a runtime
  offset (the DP rank of ``rank_info``), into the output tile, on the SP hardware DGE queue. It replaces the load of all
  4 B tokens and the Vector engine copies that select this replica's tokens (``_all_reduce_select_rows``). DP4 only.
  The EP128 base body takes it too (``BASE_GLUE_DMA_FORMS``).
- ``sp_weights``: the latent norm weight load and the shared in-projection loads of ``in_proj_form`` ``dp_local`` run
  on the SP hardware DGE queue. A load of a kernel input otherwise takes the GpSimd software DGE queue, where each DMA
  costs GpSimd time in front of the expert gathers of the routed kernel. The shared in-projection loads follow the
  last rank-row load of the world gather (a ``routing_form`` of ``_PE32_FORMS``) in the schedule, so they stream while
  the router runs, after the in-projection weights and the world gather. TP128 kernels only.
- ``split_gather``: the x gather store and load of the HBM x gather pair (``_gather_replica_tokens``) and the rank-row
  load of the ``pe32`` latent load (``_load_world_gather``) each run as 2 DMAs, partitions 0..63 on the SP and
  partitions 64..127 on the Activation hardware DGE queue, not as one DMA on the queue the compiler picks. The halves
  move the same bytes. The x DMAs exist only with the HBM x gather pair; the ``strided`` latent load keeps its DMA.
  Not with ``x_path_act`` (which puts the x DMAs on the Activation queue). The base body only.
"""
BASE_GLUE_DMA_FORMS = ("replica_rows", "split_gather")
"""The ``GLUE_DMA_FORMS`` of the base body (the EP128 ``static`` kernel with no MoE option). Its output all-reduce
result has the token-major [128, 4 B * 56] layout of the TP128 body, so the replica-rows load reads the same block.
``split_gather`` runs in the base body only."""
# The SP hardware DGE queue of the ``glue_dmas`` loads: the SP engine issues a DMA quickly.
_SP_HWDGE = {"dge_mode": nisa.dge_mode.hwdge, "engine": nisa.engine.sync}
# The Activation hardware DGE queue of the world x gather path with ``x_path_act`` (``_gather_replica_tokens``).
_ACT_HWDGE = {"dge_mode": nisa.dge_mode.hwdge, "engine": nisa.engine.scalar}
OUT_PROJ_FORMS = ("reduced", "column", "reduced_wide", "column_shared_ag")
"""Forms of the TP128 out-projection of the latent MoE body (``out_proj_form``):

- ``reduced`` (default): rank g multiplies its 28 latent-up rows and 48 shared-down rows (``moe_out_proj`` [76, H]) by
  its normed latent shard and its shared activation, and the world all-reduce adds the 128 BF16 partials of [T, H].
- ``column``: rank g computes the hidden columns 128 j + g (j = 0..55) of every token from all 9,728 input rows, in one
  FP32 accumulation, and a world all-gather of these [T, 56] rows replaces the all-reduce. The routed all-reduce runs
  in the wide layout (``WIDE_ROUTED_COLS``) and carries the shared activation of each rank in the slots of its
  partition, with zeros in the slots of the other ranks, so every rank gets the full normed latent and the full shared
  activation. ``moe_out_proj`` holds the column layout and ``moe_vectors`` adds the 3,584 latent norm weights
  (``moe_decode_torch.latent_moe_column_rank_weights``). The sum order differs from ``reduced``, so the output does
  not have its bits.
- ``reduced_wide`` (a diagnostic of ``column``): ``reduced`` with the routed all-reduce in the wide layout and zeros
  in every shared slot. Its output has the bits of ``reduced`` when the all-reduce adds each latent element in the same
  order at the larger message.
- ``column_shared_ag``: the out-projection and the output all-gather of ``column``, with the routed all-reduce of
  ``reduced`` (the [128, T * 28] SBUF tiles). The shared activation goes to every rank in its own world all-gather
  (``SHARED_GATHER_COLS``): each rank packs its [T, 48] activation into one partition, and row g of the [128, T * 48]
  destination comes from rank g. The gather runs during the routed experts: its trigger sits on the Pool queue between
  the weight streams of the first two expert chunks (``moe_sel_load_tp_finish`` ``between_chunks``), so the routed
  all-reduce carries only the latent. The gathered values are those of the wide slots of ``column`` (a slot of
  ``column`` can turn -0 into +0), so the output has the bits of ``column``. It needs the sel_load routed kernel.
  With ``in_proj_form`` ``dp_local`` the shared in-projection and activation run between the gather indices and the
  weight stream of the sel_load kernel (``moe_sel_load_tp_start`` ``before_stream``): they follow the first gather
  index in the schedule, since the Vector engine runs in order and the activation ahead of it would hold the first
  weight gathers behind the shared in-projection matmuls; and the stream follows their PSUM copy in the schedule (no
  hardware wait), so the compiler does not put the consumers of the first gathered weights ahead of the shared
  activation on the Scalar and Vector engines, where the activation would then wait for later gathers.

The forms other than ``reduced`` need a TP128 kernel, DP4, the BF16 routed all-reduce and the collective tiles of
the loop path (``decode_block.collective_sources`` ``moe_out_proj_form``). With ``column`` the world group carries only
all-gathers in the decode (the latent and the output gathers of each layer, with different destinations), so each one
skips its start handshake. A peer writes the output destination of layer k again in its gather of layer k + 1, after
its routed all-reduce of layer k + 1, which needs every rank's routed partial of layer k + 1, and that partial needs
every read of the output of layer k. The latent destination has the same order through the output gather of layer k.
With ``column_shared_ag`` the shared gather skips its start handshake too (the gather before it in the world group is
the latent gather, with another destination). A peer writes the shared destination of layer k again in its shared
gather of layer k + 1, which needs its input of layer k + 1, so its output gather of layer k, which needs every rank's
output columns of layer k, and those need every read of the shared destination of layer k.
"""
COLUMN_FORMS = ("column", "column_shared_ag")
"""The out-projection forms with the column out-projection and the output all-gather (``OUT_PROJ_FORMS``)."""
WIDE_FORMS = ("column", "reduced_wide")
"""The out-projection forms with the routed all-reduce in the wide layout (``WIDE_ROUTED_COLS``)."""
SHARED_GATHER_COLS = SHARED_SHARD
"""Columns of each token in the shared all-gather of ``column_shared_ag``: the source [1, T * 48] holds the shared
activation of this rank (element 48 t + s is activation s of token t), and row g of the destination [128, T * 48] the
one of rank g, so the [128, T, 48] destination view is the shared slots of the wide routed tile of ``column``."""
WIDE_ROUTED_COLS = MOE_OUT_PROJ_ROWS
"""Columns of each token in the wide routed all-reduce [128, T, 76]: partition g holds the routed partial of latent
columns [28 g, 28 g + 28), then the 48 shared slots of rank g (``WIDE_SHARED_COLS``)."""
WIDE_SHARED_COLS = (LATENT_SHARD, WIDE_ROUTED_COLS)
COLUMN_K_TILES = MOE_OUT_PROJ_ROWS
"""K-tiles of 128 input rows of the ``column`` out-projection. ``moe_out_proj`` [76, H] of rank g holds
[128, 76, 56]: element (p, c, j) is the weight from latent input 28 p + c (c < 28) or shared input 48 p + c - 28
(c >= 28) to hidden column 128 j + g. Partition p of K-tile c is partition p, column c of the wide routed tile."""
# Small fills a multi-chunk weight DMA in parallel with the first matmuls.
_IN_PROJ_DMA_CHUNKS = 4
_DENSE_GATE_UP_DMA_CHUNKS = 8
# DMA priority (0 is the highest) of the small loads on the critical path. The expert weight
# stream of the same layer must not delay them.
_CRITICAL_DMA_PRIORITY = 0
# A matmul or a transpose must write into one 2 KiB PSUM bank per partition. A destination that
# spans two banks gets a wrong overwrite and accumulation in the second bank. Every PSUM tile
# here is a whole bank, and the code uses a view of it.
_PSUM_BANK_BYTES = 2048
_PSUM_BANK_FP32 = _PSUM_BANK_BYTES // 4
_DTYPE_BYTES = {nl.float32: 4, nl.bfloat16: 2}
# Router selection value below every real score: a replaced top-8 entry.
_SELECTION_FLOOR = -3.0e38


class MxInProj(NamedTuple):
    """The MXFP8 in-projection in SBUF: the x4 weights ``[128, 14, 124]`` and the copies of their scales."""

    weights: nl.NkiTensor
    scales: list


def _load_mx_in_proj(moe_in_proj: nl.NkiTensor, in_proj_scale: nl.NkiTensor, name: str) -> MxInProj:
    """Load the MXFP8 in-projection ``[14, 128, 4 x 124]`` and its scales; each DMA takes ``_MX_IN_PROJ_DMA_GROUPS``
    groups, so the first matmuls start before the whole weight arrives."""
    groups, width = moe_in_proj.shape[0], moe_in_proj.shape[2]
    w_sb = nl.ndarray((_PMAX, groups, width), dtype=moe_in_proj.dtype, buffer=nl.sbuf)
    for first in range(0, groups, _MX_IN_PROJ_DMA_GROUPS):
        count = min(_MX_IN_PROJ_DMA_GROUPS, groups - first)
        nisa.dma_copy(
            dst=w_sb[0:_PMAX, first : first + count, 0:width],
            src=moe_in_proj[first : first + count, 0:_PMAX, 0:width].permute((1, 0, 2)),
            priority=dma_priority(_CRITICAL_DMA_PRIORITY),
            name=f"{name}_in_proj_mx_{first}",
        )

    def load(dst, src, copy):
        nisa.dma_copy(
            dst=dst, src=src, priority=dma_priority(_CRITICAL_DMA_PRIORITY), name=f"{name}_in_proj_scale_{copy}"
        )

    scales = load_mx_scales(in_proj_scale, lambda shape, copy: nl.ndarray(shape, dtype=nl.uint8, buffer=nl.sbuf), load)
    return MxInProj(w_sb.view(nl.float8_e4m3fn_x4), scales)


def _mx_in_proj_psum(x_tokens_hT: nl.NkiTensor, in_proj: MxInProj, width: int, dst_col: int, name: str):
    """Return the FP32 PSUM ``[T, width]`` whose columns ``dst_col ..`` hold the MXFP8 tokens times the 124 MXFP8
    in-projection columns. The tokens are quantized per token (tile-major ``[128, 56, T]``); an odd T takes one zero
    pad row on the stationary side. ``width`` 131 with ``dst_col`` 7 keeps the column layout of the BF16 form (its
    router columns stay unwritten: the FP16 router gives the logits)."""
    tokens = x_tokens_hT.shape[2]
    padded = tokens + tokens % 2
    x_data, x_scale = quantize_hT_mx(x_tokens_hT, f"{name}_x", padded, tile_major=True)
    psum = _psum_bank(padded, (width,), nl.float32)
    for group in range(in_proj.weights.shape[1]):
        nisa.nc_matmul_mx(
            dst=psum[0:padded, dst_col : dst_col + MXFP8_IN_PROJ_COLS],
            stationary=x_data[0:_PMAX, group, 0:padded],
            moving=in_proj.weights[0:_PMAX, group, 0:MXFP8_IN_PROJ_COLS],
            stationary_scale=x_scale[0:_PMAX, group, 0:padded],
            moving_scale=mx_scale_view(in_proj.scales, group, 0, MXFP8_IN_PROJ_COLS),
            accumulate=group > 0,
        )
    return psum[0:tokens, 0:width]


class LocalRouterLogits(NamedTuple):
    """The FP32 SBUF router logits [T, 7] of this rank's experts, computed before the latent stages (``fp16_local``)."""

    logits: nl.NkiTensor


class RouterScores(NamedTuple):
    """The router values of all tokens over all experts, on token partitions (the output of a router form).

    Expert e = n g + j is element ``[t, g, j]`` of ``scores`` and ``selection`` (FP32 SBUF views [T, G, n]).
    ``local_scores`` and ``local_selection`` [T, 7] hold this rank's EP128 experts 7 g .. 7 g + 6, bit-identical
    to their gathered values.
    """

    scores: nl.NkiTensor
    selection: nl.NkiTensor
    local_scores: nl.NkiTensor
    local_selection: nl.NkiTensor


LATENT_SHARD_FORMS = ("one_hot", "columns")
"""Forms of the transposed latent shard of the latent norm (``latent_shard_form`` of the base body, ``_latent_norm``):

- ``one_hot`` (default): one matmul for each token: the token's [128, 28] routed rows stationary and the world-rank
  one-hot moving give its column of the [28, T] shard (T matmuls).
- ``columns``: one matmul for each of the 28 shard columns: the strided [128, T] view of column c of every token
  stationary and the one-hot moving give column c of the [T, 28] shard, then an exact BF16 copy and one TensorE
  transpose give the [28, T] shard (28 matmuls, T <= 128). Each element is the same one-hot sum of the same products
  over the 128 partitions, and the copy and the transpose keep the bits of the BF16 values, so the output is bitwise
  the same (a NaN stays a NaN).
"""
LATENT_LOADS = ("strided", "pe32")
# How the base MoE body fills the [128, T, 28] routed all-reduce source (``routed_store_form`` of ``latent_moe_decode``).
ROUTED_STORE_FORMS = ("relayout", "pe")


def check_routed_store_buffers(
    routed_store_form: str, persistent_source: bool, weight_buffers: int, half_weight_buffers: int
) -> None:
    """Fail closed on the column-group routed store into a per-call SBUF tile with fewer than 3 weight buffers.

    With ``routed_store_form`` ``pe`` the static expert loop of ``moe_tkg_decode`` writes its output into an SBUF
    tile (``column_groups_out``). When each call allocates that tile and a DMA then stores it to HBM, and the loop
    rotates fewer than 3 whole weight buffers, the outputs of the third and later expert calls of one kernel can be
    wrong on Trn3 hardware, while the simulator gives the right values. The forms that give the right values are
    checked here: the persistent SBUF routed source (``routed_src_sb`` of the collective sources), half-size weight
    buffers (``half_weight_buffers``), 3 whole weight buffers, or the ``relayout`` store (an HBM output of the loop).
    The other forms fail at trace time.

    Args:
        routed_store_form (str): ``ROUTED_STORE_FORMS``.
        persistent_source (bool): the loop writes the persistent SBUF routed source, not a per-call tile.
        weight_buffers (int): whole weight buffers of the expert loop.
        half_weight_buffers (int): half-size weight buffers of the expert loop, 0 for none.

    Example:
        >>> check_routed_store_buffers("pe", False, 2, 4)
        >>> check_routed_store_buffers("relayout", False, 2, 0)
    """
    kernel_assert(
        routed_store_form != "pe" or persistent_source or half_weight_buffers or weight_buffers >= 3,
        f"routed_store_form='pe' (column-group store) into a per-call SBUF tile with {weight_buffers} whole weight "
        "buffers is not supported: on Trn3 the third and later expert calls of a kernel can give wrong values. Use "
        "the persistent SBUF routed source, half_weight_buffers, weight_buffers=3 or routed_store_form='relayout'",
    )


def _load_world_gather(
    gather_dst,
    gather_dst_hbm,
    tokens: int,
    name: str,
    mode: str = "strided",
    cols: int = _GATHER_COLS,
    split: bool = False,
):
    """Load the HBM world gather ``gather_dst_hbm`` ([W * T, C]; row r * T + t is token t of rank r) into ``gather_dst``
    (SBUF [T, W * C]). Both modes move the same bytes and give the same values.

    ``strided``: one DMA of W * T rows of C words. ``pe32``: one contiguous DMA to SBUF [W, T * C], then TensorE
    transposes of each FP32 word to PSUM [T, W] (bit-accurate on trn3), copied to [T, W, C] on the Vector and Scalar
    engines in turn. ``cols`` (``SCORE_GATHER_COLS`` for ``router_rows_drop_selection``): the HBM rows have this many
    words, and they fill words 0 .. cols - 1 of each block of ``gather_dst``. ``split`` (``GLUE_DMA_FORMS``
    ``split_gather``): the ``pe32`` rank-row load runs as 2 DMAs of 64 rank rows, on the SP and the Activation hardware
    DGE queues (``_split_rows_dma``); the ``strided`` load keeps its DMA. Returns the DMA that reads ``gather_dst_hbm``
    (the one DMA, or the rows DMA in ``pe32``; with ``split``, the tuple of the 2 rows DMAs).
    """
    kernel_assert(mode in LATENT_LOADS, f"latent_load {mode!r} must be one of {LATENT_LOADS}")
    gathered = gather_dst.reshape((tokens, WORLD_SIZE, _GATHER_COLS))
    if mode == "strided":
        return nisa.dma_copy(
            dst=gathered if cols == _GATHER_COLS else gathered[:, :, 0:cols],
            src=gather_dst_hbm.ap(pattern=[[cols, tokens], [tokens * cols, WORLD_SIZE], [1, cols]], offset=0),
            priority=_CRITICAL_DMA_PRIORITY,
            name=f"{name}_latent_load",
        )
    rows = nl.ndarray((WORLD_SIZE, tokens * cols), dtype=nl.float32, buffer=nl.sbuf)
    if split:
        load = _split_rows_dma(
            rows, gather_dst_hbm.reshape((WORLD_SIZE, tokens * cols)), f"{name}_latent_rows", _CRITICAL_DMA_PRIORITY
        )
    else:
        load = nisa.dma_copy(
            dst=rows,
            src=gather_dst_hbm.reshape((WORLD_SIZE, tokens * cols)),
            priority=_CRITICAL_DMA_PRIORITY,
            name=f"{name}_latent_rows",
        )
    group = 4
    src3 = rows.reshape((WORLD_SIZE, tokens, cols))
    dst3 = gather_dst.reshape((tokens, WORLD_SIZE, _GATHER_COLS))
    for gi, w0 in enumerate(range(0, cols, group)):
        width = min(group, cols - w0)
        psum = nl.ndarray((tokens, width * WORLD_SIZE), dtype=nl.float32, buffer=nl.psum)
        for j in range(width):
            nisa.nc_transpose(
                dst=psum[0:tokens, j * WORLD_SIZE : (j + 1) * WORLD_SIZE], data=src3[0:WORLD_SIZE, 0:tokens, w0 + j]
            )
        nisa.tensor_copy(
            dst=dst3[0:tokens, 0:WORLD_SIZE, w0 : w0 + width],
            src=psum.ap(pattern=[[width * WORLD_SIZE, tokens], [1, WORLD_SIZE], [WORLD_SIZE, width]], offset=0),
            engine=(nisa.engine.vector, nisa.engine.scalar)[gi % 2],
        )
    return load


def _ring_buffer_args(sources: dict, half_weight_buffers: int, latent_loaded) -> dict:
    """Return the ``moe_tkg_decode`` arguments of the persistent ring buffer ``moe_ring_buffer`` of ``sources``
    (``decode_block.collective_sources`` ``ring_reclaim``), or {} without it.

    The buffer is an explicit-address half weight buffer over collective tiles that the layer does not touch from its
    latent world gather to the end of its expert loop. It joins the ring after the loop's own half buffers. Its first
    load waits for the load of the latent world gather (a runtime dependency): that load reads the gather's output,
    so the gather and every collective before it on the stream have completed, and so has every local read of those
    tiles in the layer. With ``glue_dmas`` ``split_gather`` the load is 2 DMAs, and the first load waits for both.
    """
    if not sources or "moe_ring_buffer" not in sources:
        return {}
    kernel_assert(
        half_weight_buffers, "moe_ring_buffer joins the half buffers of the static loop: it needs half_weight_buffers"
    )
    kernel_assert(latent_loaded is not None, "moe_ring_buffer needs the HBM latent world gather")
    return {"extra_weight_buffers": (sources["moe_ring_buffer"],), "extra_buffers_after": latent_loaded}


def _split_rows_dma(dst: nl.NkiTensor, src: nl.NkiTensor, name: str, priority=None):
    """Copy ``src`` to ``dst`` (views of one shape with 128 rows on dim 0, the partitions of the SBUF side) as 2 DMAs
    (``GLUE_DMA_FORMS`` ``split_gather``): rows 0..63 on the SP hardware DGE queue (``{name}_sp``), then rows 64..127 on
    the Activation hardware DGE queue (``{name}_act``). ``priority``: the DMA priority of both halves (None: the
    default). The halves move the bytes of the one DMA. Returns the 2 DMA instructions (SP half, Activation half)."""
    kernel_assert(
        tuple(dst.shape) == tuple(src.shape) and dst.shape[0] == _PMAX,
        f"split_gather copies views of one shape with {_PMAX} rows, got {tuple(dst.shape)} and {tuple(src.shape)}",
    )
    half = _PMAX // 2
    return tuple(
        nisa.dma_copy(
            dst=dst[lo : lo + half],
            src=src[lo : lo + half],
            name=f"{name}_{suffix}",
            **queue,
            **({} if priority is None else {"priority": priority}),
        )
        for lo, queue, suffix in ((0, _SP_HWDGE, "sp"), (half, _ACT_HWDGE, "act"))
    )


_MX_ROUTER_SCORE_COLS = (0, LOCAL_EXPERTS)
_MX_ROUTER_SELECTION_COLS = (LOCAL_EXPERTS, 2 * LOCAL_EXPERTS)
# The router rows of ``latent_gather_mx`` move from PSUM by the engine that copies score word 14 + j of the 28-word row
# in the pe32 load (``_load_world_gather``: the Scalar engine for the odd 4-word groups), in runs of adjacent scores.
_MX_ROUTER_COPY_RUNS = ((0, 2, nisa.engine.scalar), (2, 6, nisa.engine.vector), (6, 7, nisa.engine.scalar))
_SP_HWDGE_CRITICAL = {"dge_mode": nisa.dge_mode.hwdge, "engine": nisa.engine.sync, "priority": _CRITICAL_DMA_PRIORITY}
_ACT_HWDGE_CRITICAL = {
    "dge_mode": nisa.dge_mode.hwdge,
    "engine": nisa.engine.scalar,
    "priority": _CRITICAL_DMA_PRIORITY,
}


_GROUPED_AG_MAX_BYTES = 262144
"""The largest W128 1-D HBM all-gather (output bytes, inclusive) of the grouped mesh algorithm; larger ones take the
bandwidth-opt mesh algorithm, the class of the 28-word latent gather from T = 32 tokens."""
_MX_PAD_WORDS = 8


def mx_gather_block_shape(tokens: int) -> tuple:
    """Return the UINT32 [8, W] world gather block of one rank with ``latent_gather_mx`` (``_mx_gather_block``): W = 2 T
    words, plus 8 unread pad words (32 bytes) when the gather would be exactly 256 KiB (T = 32), so that it stays on the
    bandwidth-opt algorithm of the 28-word gather of the same T."""
    width = 2 * tokens
    if WORLD_SIZE * _MX_CODE_ROWS * width * 4 == _GROUPED_AG_MAX_BYTES:
        width += _MX_PAD_WORDS
    return (_MX_CODE_ROWS, width)


def _mx_quantize_block(latent_psum: nl.NkiTensor, tokens: int) -> tuple:
    """Quantize this rank's [T, 32] latent block to MXFP8 the way ``moe_routed._quantize_natural_latent`` quantizes
    the whole latent: round the FP32 PSUM to BF16 (the Vector copy of the base body's latent words), move column
    4 i + q of each token to partition i, element q with 4 TensorE transposes of the stride-4 column views, and
    quantize 8 partitions x 4 elements per token as one MX block. ``quantize_mx`` takes a multiple of 32 partitions, so
    partitions 8..31 hold zeros (three all-zero blocks that no rank sends).

    Returns:
        codes [32, T] ``float8_e4m3fn_x4`` (partition i, element t: the 4 codes of columns 4 i .. 4 i + 3 of token t;
        rows 0..7 are the block) and scales [32, T] UINT8 (partition 0: the E8M0 scale of the block for each token).
    """
    block = nl.ndarray((tokens, MX_BLOCK), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=block, src=latent_psum, engine=nisa.engine.vector)
    swizzle_psum = nl.ndarray((_VECTOR_TRANSPOSE_MAX, _MX_Q, tokens), dtype=nl.bfloat16, buffer=nl.psum)
    for q in range(_MX_Q):
        nisa.nc_transpose(
            dst=swizzle_psum[0:_MX_CODE_ROWS, q, :],
            data=block.ap(pattern=[[MX_BLOCK, tokens], [_MX_Q, _MX_CODE_ROWS]], offset=q),
            engine=nisa.engine.tensor,
        )
    quant_in = nl.ndarray((_VECTOR_TRANSPOSE_MAX, tokens, _MX_Q), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.memset(dst=quant_in, value=0.0, engine=nisa.engine.vector)
    nisa.tensor_copy(
        dst=quant_in[0:_MX_CODE_ROWS, :, :],
        src=swizzle_psum.ap(pattern=[[_MX_Q * tokens, _MX_CODE_ROWS], [1, tokens], [tokens, _MX_Q]], offset=0),
        engine=nisa.engine.vector,
    )
    codes = nl.ndarray((_VECTOR_TRANSPOSE_MAX, tokens), dtype=nl.float8_e4m3fn_x4, buffer=nl.sbuf)
    scales = nl.ndarray((_VECTOR_TRANSPOSE_MAX, tokens), dtype=nl.uint8, buffer=nl.sbuf)
    nisa.quantize_mx(src=quant_in, dst=codes, dst_scale=scales)
    return codes, scales


def _mx_gather_block(
    in_psum: nl.NkiTensor, local_bias_sb: nl.NkiTensor, gather_hbm: nl.NkiTensor, tokens: int, name: str
) -> tuple:
    """Build this rank's ``latent_gather_mx`` world gather block in ``gather_hbm`` (UINT32 [8, W], W = 2 T or 2 T + 8,
    ``mx_gather_block_shape``):

    - rows 0..7, words 0 .. T - 1: the MXFP8 codes of the rank's latent block (``_mx_quantize_block``);
    - rows 0..6, words T .. 2 T - 1: router score j (sigmoid of the logit of expert 7 g + j) of each token, the FP32
      bits, moved to rows by one bit-accurate TensorE transpose;
    - row 7, bytes 4 T .. 5 T - 1: the E8M0 scale byte of each token (the other words of the row are not read).

    Returns:
        (local scores [T, 7] FP32, local selection values [T, 7] FP32 = scores + bias on the Vector engine, the last
        block store).
    """
    local_scores = nl.ndarray((tokens, LOCAL_EXPERTS), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(
        dst=local_scores, op=nl.sigmoid, data=in_psum[:, ROUTER_COLS[0] : ROUTER_COLS[1]], engine=nisa.engine.scalar
    )
    local_selection = nl.ndarray((tokens, LOCAL_EXPERTS), dtype=nl.float32, buffer=nl.sbuf)
    # The Vector add of the gathered scores (_add_router_bias): the same instruction for the rank's own experts.
    nisa.tensor_tensor(
        dst=local_selection, data1=local_scores, data2=local_bias_sb, op=nl.add, engine=nisa.engine.vector
    )
    codes, scales = _mx_quantize_block(in_psum[:, MX_LATENT_COLS[0] : MX_LATENT_COLS[1]], tokens)
    scores_psum = nl.ndarray((LOCAL_EXPERTS, tokens), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=scores_psum, data=local_scores, engine=nisa.engine.tensor)
    scores_rows = nl.ndarray((LOCAL_EXPERTS, tokens), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=scores_rows, src=scores_psum, engine=nisa.engine.vector)
    nisa.dma_copy(
        dst=gather_hbm[0:_MX_CODE_ROWS, 0:tokens],
        src=codes[0:_MX_CODE_ROWS, :].view(nl.uint32),
        name=f"{name}_mx_codes_store",
        **_SP_HWDGE_CRITICAL,
    )
    nisa.dma_copy(
        dst=gather_hbm[0:LOCAL_EXPERTS, tokens : 2 * tokens],
        src=scores_rows.view(nl.uint32),
        name=f"{name}_mx_scores_store",
        **_ACT_HWDGE_CRITICAL,
    )
    store = nisa.dma_copy(
        dst=gather_hbm[_MX_SCALE_ROW : _MX_SCALE_ROW + 1, tokens : tokens + tokens // 4],
        src=scales.view(nl.uint32)[0:1, 0 : tokens // 4],
        name=f"{name}_mx_scale_store",
        **_SP_HWDGE_CRITICAL,
    )
    return local_scores, local_selection, store


def _load_mx_gather(gather_dst_hbm: nl.NkiTensor, tokens: int, name: str) -> tuple:
    """Load the ``latent_gather_mx`` world gather (UINT32 [1024, W]; rows 8 r .. 8 r + 7 are the block of rank r).

    - Codes: row 8 r + i = 128 h + p of the gathered table is partition p of latent tile h (r = 16 h + p // 8 < 112),
      so one DMA puts them in the MX expert input [128, 7, T] (``moe_tkg_decode`` ``quantized_latent``).
    - Scales: block k of tile h goes to partition 32 (k // 4) + k % 4, where ``quantize_mx`` writes it; the other
      partitions are zero, as ``quantize_mx`` leaves them. One DMA per partition quadrant.
    - Router rows: [128 ranks, 7, T] by one DMA, then 7 bit-accurate TensorE transposes of the FP32 words to the token
      partitions, copied by the engines of the pe32 load (``_MX_ROUTER_COPY_RUNS``).

    Returns:
        (router [T, 128, 14] FP32: scores of expert 7 g + j in column j, the selection columns 7..13 for
        ``_add_router_bias``; codes [128, 7, T] ``float8_e4m3fn_x4``; scales [128, 7, T] UINT8).
    """
    width = gather_dst_hbm.shape[1]
    block_words = _MX_CODE_ROWS * width
    quant = nl.ndarray((_PMAX, LATENT_SIZE // 512, tokens), dtype=nl.float8_e4m3fn_x4, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=quant.view(nl.uint32),
        src=gather_dst_hbm.ap(pattern=[[width, _PMAX], [_PMAX * width, LATENT_SIZE // 512], [1, tokens]], offset=0),
        name=f"{name}_mx_codes_load",
        **_SP_HWDGE_CRITICAL,
    )
    rows = nl.ndarray((WORLD_SIZE, LOCAL_EXPERTS, tokens), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=rows,
        src=gather_dst_hbm.ap(pattern=[[block_words, WORLD_SIZE], [width, LOCAL_EXPERTS], [1, tokens]], offset=tokens),
        name=f"{name}_mx_router_rows",
        **_ACT_HWDGE_CRITICAL,
    )
    scales = nl.ndarray((_PMAX, LATENT_SIZE // 512, tokens), dtype=nl.uint8, buffer=nl.sbuf)
    nisa.memset(dst=scales, value=0, engine=nisa.engine.gpsimd)
    blocks_per_tile = _PMAX // _MX_CODE_ROWS
    for quadrant in range(_PMAX // _VECTOR_TRANSPOSE_MAX):
        nisa.dma_copy(
            dst=scales.view(nl.uint32)[
                _VECTOR_TRANSPOSE_MAX * quadrant : _VECTOR_TRANSPOSE_MAX * quadrant + _MX_SCALES_PER_QUADRANT, :, :
            ],
            src=gather_dst_hbm.ap(
                pattern=[
                    [block_words, _MX_SCALES_PER_QUADRANT],
                    [blocks_per_tile * block_words, LATENT_SIZE // 512],
                    [1, tokens // 4],
                ],
                offset=_MX_SCALES_PER_QUADRANT * quadrant * block_words + _MX_SCALE_ROW * width + tokens,
            ),
            name=f"{name}_mx_scale_load{quadrant}",
            **(_SP_HWDGE_CRITICAL if quadrant % 2 == 0 else _ACT_HWDGE_CRITICAL),
        )
    router = nl.ndarray((tokens, WORLD_SIZE, 2 * LOCAL_EXPERTS), dtype=nl.float32, buffer=nl.sbuf)
    words = rows.view(nl.float32)
    for first, last, engine in _MX_ROUTER_COPY_RUNS:
        psum = nl.ndarray((tokens, (last - first) * WORLD_SIZE), dtype=nl.float32, buffer=nl.psum)
        for j in range(first, last):
            nisa.nc_transpose(
                dst=psum[0:tokens, (j - first) * WORLD_SIZE : (j - first + 1) * WORLD_SIZE],
                data=words[0:WORLD_SIZE, j, 0:tokens],
                engine=nisa.engine.tensor,
            )
        nisa.tensor_copy(
            dst=router[0:tokens, 0:WORLD_SIZE, first:last],
            src=psum.ap(
                pattern=[[(last - first) * WORLD_SIZE, tokens], [1, WORLD_SIZE], [WORLD_SIZE, last - first]], offset=0
            ),
            engine=engine,
        )
    return router, quant, scales


def _load_router_bias(moe_vectors: nl.NkiTensor, tokens: int, name: str, after=None) -> nl.NkiTensor:
    """Load the router bias of all 896 experts (``ALL_ROUTER_BIAS_COLS``) onto the ``tokens`` token partitions:
    [T, 896] FP32 SBUF, element (i, 7 g + e) is the bias of expert 7 g + e. ``after``: the instruction that the load
    follows (a schedule edge)."""
    router_bias = nl.ndarray((tokens, NUM_EXPERTS), dtype=nl.float32, buffer=nl.sbuf)
    load = nisa.dma_copy(
        dst=router_bias,
        src=moe_vectors.ap(pattern=[[0, tokens], [1, NUM_EXPERTS]], offset=ALL_ROUTER_BIAS_COLS[0]),
        name=f"{name}_router_bias",
        **_SP_HWDGE,
    )
    if after is not None:
        _instruction(load).depends_on(_instruction(after), kind=nisa.dependency.schedule)
    return router_bias


def _add_router_bias(
    gathered: nl.NkiTensor,
    router_bias: nl.NkiTensor,
    score_cols: tuple = _GATHER_SCORE_COLS,
    selection_cols: tuple = _GATHER_SELECTION_COLS,
):
    """Write the selection words of a [T, 128, C] gathered tile: score + bias of expert 7 g + e, the FP32 add that each
    rank runs for its own experts, on the same scores and biases (``router_rows_drop_selection``)."""
    tokens = gathered.shape[0]
    nisa.tensor_tensor(
        dst=gathered[:, :, selection_cols[0] : selection_cols[1]],
        data1=gathered[:, :, score_cols[0] : score_cols[1]],
        data2=router_bias.ap(
            pattern=[[NUM_EXPERTS, tokens], [LOCAL_EXPERTS, WORLD_SIZE], [1, LOCAL_EXPERTS]], offset=0
        ),
        op=nl.add,
        engine=nisa.engine.vector,
    )


def _moe_token_counts(x_hT: nl.NkiTensor, sources: dict | None, dp_size: int) -> tuple[int, int]:
    """Return ``(replica_tokens, tokens)`` of a base MoE body: this replica's output rows and the gathered tokens.

    The standard path gathers this replica's ``x_hT.shape[1]`` rows over the DP group, so it keeps the same count
    and gathers ``dp_size`` times as many. ``attn_rs_moe_input`` instead holds one row per rank and all-gathers them
    by token over the world (``_gather_replica_tokens``), so it keeps ``TP_SIZE`` rows per replica and gathers
    ``WORLD_SIZE`` tokens, the same count and layout as the standard DP4 base body, with a one-row ``x_hT`` input.
    """
    if (sources or {}).get("x_token_world_src_hbm") is not None:
        return TP_SIZE, WORLD_SIZE
    replica_tokens = x_hT.shape[1]
    return replica_tokens, dp_size * replica_tokens


def latent_moe_decode(
    x_hT: nl.NkiTensor,
    moe_in_proj: nl.NkiTensor,
    moe_out_proj: nl.NkiTensor,
    moe_vectors: nl.NkiTensor,
    expert_gate_up_weights: nl.NkiTensor,
    expert_gate_up_weights_scale: nl.NkiTensor,
    expert_down_weights: nl.NkiTensor,
    expert_down_weights_scale: nl.NkiTensor,
    rank_info: nl.NkiTensor,
    dp_group: ReplicaGroup,
    world_group: ReplicaGroup,
    tag: str,
    weight_buffers: int = 3,
    scale_buffers: int = 2,
    sources: dict | None = None,
    routed_kernel: str = "static",
    region: ExplicitSbuf = None,
    routed_reduce_dtype: str = "bf16",
    routing_debug: tuple = None,
    router_mode: str = None,
    router_dtype: str = "bf16",
    router_weight: nl.NkiTensor = None,
    x16_hT: nl.NkiTensor = None,
    expert_offset: int = 0,
    affinity_scaling: str = "pre_quant",
    affinity_hi_lo: bool = True,
    early_loads: bool = True,
    dynamic_experts: bool = False,
    dp_size: int = DP_SIZE,
    routing_form: str = None,
    active_expert_slots: int = 0,
    sel_load_options: tuple = (),
    active_expert_count: nl.NkiTensor = None,
    latent_load: str = "strided",
    half_weight_buffers: int = 0,
    gate_up_half_major: bool = False,
    in_proj_form: str = "gathered",
    peer_in_proj: nl.NkiTensor = None,
    routed_store_form: str = "relayout",
    glue_dmas: tuple = (),
    direct_latent: bool = False,
    latent_shard_form: str = "one_hot",
    out_proj_form: str = "reduced",
    routing_tiles: bool = False,
    preamble_buffers: int = 0,
    hot_experts: dict = None,
    preamble_reuse: bool = False,
    in_proj_scale: nl.NkiTensor = None,
    router_rows_drop_selection: bool = False,
    latent_gather_mx: bool = False,
    shared_scale_buffer: bool = False,
    static_alt_tiles: bool | str = False,
) -> nl.NkiTensor:
    """Run one Kimi-K3 latent MoE decode layer for the tokens of the DP replicas and keep this replica's rows.

    Every rank gathers the tokens of the 4 replicas, routes them with a sharded router, runs its
    7 local MXFP4 experts on the gathered latent, and adds its TP128 shard of the latent up and
    the shared experts. The world all-reduce gives the full output of all tokens. Not
    ``@nki.jit``: the decode giga-kernel inlines one call for each MoE layer. The attribute
    ``token_major`` is True.

    Dimensions:
        H: Hidden size, 7168.
        B: Tokens of each DP replica, 1 to ``MAX_REPLICA_TOKENS`` (32): up to 32 in decode, and up to
            ffn_rows = 32 in a verify feed-forward call.
        T: Gathered tokens, 4 * B <= 128. Token r * B + b is token b of replica r.
        L: Latent size, 3584. Rank g owns latent columns [28 g, 28 g + 28).
        E: Routed experts, 896. Rank g owns experts [7 g, 7 g + 7).

    Args:
        x_hT (nl.NkiTensor): [128, B, 56] BF16 SBUF, this replica's normalized hidden in the
            token-major hT layout. Element (p, b, j) is hidden index j * 128 + p of token b.
        moe_in_proj (nl.NkiTensor): [128, 56, 131] BF16 HBM, the fused [in, out] projections of
            the hidden, hT-tiled (element (p, j, n) is input row j * 128 + p): router rows
            7g..7g+6 (7 columns), latent down shard (28), shared gate shard (48), shared up
            shard (48).
        moe_out_proj (nl.NkiTensor): [76, H] BF16 HBM, the fused [in, out] projections back to the
            hidden: latent up shard (28 rows), then shared down shard (48 rows). One matmul
            with K = 76 computes both for each hidden tile.
        moe_vectors (nl.NkiTensor): [1, 35] FP32 HBM: the router correction bias of this rank's
            7 experts, and this rank's 28 latent norm weights.
        expert_gate_up_weights (nl.NkiTensor): [7, 128, 2, 7, 3072] packed MXFP4 (or its UINT16
            carrier), the serving layout of ``moe_tkg_decode``.
        expert_gate_up_weights_scale (nl.NkiTensor): [7, 16, 2, 7, 3072] UINT8 E8M0 scales.
        expert_down_weights (nl.NkiTensor): [7, 128, 6, 3584] packed MXFP4 (or UINT16 carrier).
        expert_down_weights_scale (nl.NkiTensor): [7, 16, 6, 3584] UINT8 E8M0 scales.
        rank_info (nl.NkiTensor): [1, 4] INT32 HBM, (global rank, DP rank, TP rank, 0).
        dp_group (ReplicaGroup): [[t, t + 32, t + 64, t + 96] for t in 0..31].
        world_group (ReplicaGroup): [[0 .. 127]].
        tag (str): Unique per-layer string. It names the collectives and the DMAs.
        weight_buffers (int): SBUF weight buffers of the expert loop (``moe_tkg_decode``), 1 to 3. The
            values do not change; each buffer less frees 43 KB of SBUF for each partition.
        scale_buffers (int): SBUF gate/up scale buffers of the expert loop, 2 or 1. The values do not
            change; 1 frees 21 KB of SBUF for each partition.
        sources (dict): None, or the persistent SBUF sources of the collectives
            (``decode_block.collective_sources``). The values do not change.
        routed_kernel (str): the MoE options (this argument and the ones below, ``moe_options_in_use``) run
            ``_routed_latent_moe_decode``; the defaults run the base body. ``static``, ``sel_load`` or
            ``tp_dense`` (``ROUTED_KERNELS``). A deployment keeps one layout (EP128 or TP128,
            ``TP128_KERNELS``); the TP128 kernels take the weights of ``SEL_LOAD_WEIGHT_NAMES``.
        region (ExplicitSbuf): None, or the explicit SBUF region of the kernel. The out-projection input that
            lives across the expert call then takes an explicit address, and the TP128 kernels place their tiles
            in the region.
        routed_reduce_dtype (str): ``bf16`` adds the BF16 routed partials in BF16 in the all-reduce. ``fp32``
            adds them in FP32 (twice the bytes) and rounds the sum to BF16 once.
        routing_debug (tuple): test only. None, or HBM ``(index [T, 16] INT32, affinity [T, 16] FP32)`` that
            receive the TP128 routing.
        router_mode (str): router form (``ROUTER_MODES``), ``fused`` (None, the default) or ``split``. The
            values do not change.
        router_dtype (str): router matmul type (``ROUTER_DTYPES``), ``bf16`` (default), ``fp16`` or ``fp16_local``.
            ``fp16_local`` runs in the base body too (``BASE_ROUTER_DTYPES``, DP4): the local logits ride the packed
            rows of the DP x gather, and the base body forms keep their programs.
        router_weight (nl.NkiTensor): with ``fp16``, [128, 56, 7] FP16 HBM, the router rows 7g..7g+6 of this
            rank, hT-tiled (element (p, j, e) is ``router_weight[7 g + e, 128 j + p]``). With ``fp16_local``,
            [128, 56, 28] FP16 HBM, the rows of the DP peers of this TP rank t: element (p, j, 7 d + e) is
            ``router_weight[7 (32 d + t) + e, 128 j + p]``.
        x16_hT (nl.NkiTensor): with ``fp16`` or ``fp16_local``, [128, B, 56] FP16 SBUF, the normed hidden of
            ``x_hT`` rounded to FP16 from the FP32 norm values (``common.rmsnorm_hT(out16=...)``).
        expert_offset (int): ``sel_load`` only: the TP128 expert tensors may be the stacks of all MoE layers
            ([L, 896, ...]), as the caller passes them; the kernel reads expert e of this layer at stack row
            ``expert_offset + e`` (``layer * 896``), so no weight DMA takes a static offset of 4 GiB or more.
        affinity_scaling (str): ``sel_load`` only: ``pre_quant`` (default) scales the activation by the affinity
            before its MXFP8 quantization; ``post_down`` scales each expert's down output (see the kernel).
        affinity_hi_lo (bool): ``sel_load`` only: True (default) splits the scaled activation into a BF16 high
            and a low term before the quantization; False quantizes one term (``post_down`` always uses one term).
        early_loads (bool): ``static`` only: True (default) issues the expert loop's first weight and scale loads
            before its latent work, False in the loop (``moe_tkg_decode``). The values do not change.
        dynamic_experts (bool): base body only: run the dynamic expert loop of ``moe_tkg_decode`` (only the
            active local experts) with 3 weight buffers and 2 scale buffers. The values do not change. The
            decode entries set it through ``dynamic_expert_loop``. False (default) runs the static loop.
        dp_size (int): DP replicas gathered on each rank. 4 (default) gathers the tokens of the 4 DP replicas
            over ``dp_group`` and keeps this replica's rows. 1 is the DP1 layout: no token gather, T = the rows
            of the call on every rank, and the output keeps every row. The world collectives are unchanged.
        routing_form (str): TP128 kernels only: the form of the top-16 routing (``ROUTING_FORMS``). None
            (default): ``index_first`` for the TP128 kernels, ``base`` for the others. The forms other than
            ``base`` give the expert index earlier.
        active_expert_slots (int): base body only: run only the active local experts of ``moe_tkg_decode`` in this
            many straight-line slots, and every expert in a one-trip loop when more are active. The values do not
            change. The decode entries set it through ``routed_expert_path``, which gives the dynamic loop
            precedence; asking for both fails, and so do the slots with a MoE option (the FP16 routers among them).
            0 (default) runs the static loop.
        active_expert_count (nl.NkiTensor): debug, with ``active_expert_slots`` only: an INT32 ``[1, 1]`` HBM
            tensor that gets the number of active local experts of this rank in the call (``active_expert_count`` of
            ``moe_tkg_decode``); more than ``active_expert_slots`` means the overflow loop ran. None (default).
        latent_load (str): how the HBM latent gather loads to SBUF (``LATENT_LOADS``, ``_load_world_gather``). The
            values do not change. ``strided`` (default) keeps the program.
        sel_load_options (tuple): ``sel_load``: (name, value) pairs of extra ``moe_sel_load_tp_start`` options (for
            example ``long_stream``, ``chunk_tokens``, ``sbuf_reserve``). The values do not change. () (default) keeps
            the program. ``("kimi_body", True)``: a call of 8 tokens runs the kimi two-phase body
            (``selective_load_tp_start`` / ``_finish``, its defaults) with the other options; it needs one down scale
            set, pre_quant and the padded down rows, and its output bits equal the served call's. Calls of other sizes
            keep the served call. ``("kimi_hwdge_calls", N)`` (with kimi_body): the first N kimi body calls of the trace
            load gate/up group 0 with HWDGE in place of a Pool gather (``group0_gather=False``), and every kimi call
            takes ``reserve_hwdge_ids``; the output does not change. kimi_body does not take ``overlap_calls``: each
            call reads the output of the layer before, so the calls do not overlap. ``("kimi_t32_body", True)``
            (alone): the one-call body (``_kimi_t32_partial``, the options of ``_KIMI_T32_OPTIONS``) replaces
            the 4 x 8-token packed calls -- every distinct expert of the 32-token call loads and computes once. It runs
            per 32-token chunk (T = 32: one call; T = 64: two), and fails closed at any other T. The block positions in
            token order and the fp32 affinities from the selected scores keep the per-call rel-L2 vs fp64 below the
            served call's, so the output can differ from the served call's in the last bits.
            Default off. ``("kimi_t4_body", True)``:
            a call of 4 tokens runs the kimi body in its 4-token form (``_KIMI_T4_OPTIONS``); it has the kimi_body
            preconditions, takes no kimi_hwdge_calls and no other option than the 8-token call options, and its output
            bits equal the served call's.
        half_weight_buffers (int): static loop only: 0 (default) keeps ``weight_buffers``; N in 2 to 6 streams half
            projections through N half-size buffers in place of them (``half_weight_buffers`` of ``moe_tkg_decode``).
            The values do not change; 4 holds the SBUF of 2 whole buffers.
        gate_up_half_major (bool): static loop only: the gate/up weights are stored half-major
            (``gate_up_half_major`` of ``moe_tkg_decode``), one run per partition for each half. The values do not
            change. False (default) reads the standard layout.
        shared_scale_buffer (bool): static loop only, with ``scale_buffers=1``: one MX scale buffer of 4 slots for
            the gate/up and down scales (``shared_scale_buffer`` of ``moe_tkg_decode``), 21 KB less SBUF per
            partition in the loop. The values do not change. False (default) keeps the two scale buffers.
        in_proj_form (str): TP128 kernels only: the form of the in-projection (``IN_PROJ_FORMS``). ``gathered``
            (default) keeps the program. ``dp_local`` takes ``peer_in_proj``, ``moe_in_proj`` [128, 56, 96] (the
            shared columns) and ``moe_vectors`` [1, 63] (``moe_decode_torch.latent_moe_dp_local_rank_weights``) and
            gives the bits of ``gathered``.
        peer_in_proj (nl.NkiTensor): with ``dp_local``, [128, 56, 140] BF16 HBM, the router and latent-down columns
            of the 4 DP peer shards of this TP rank (``PEER_ROUTER_COLS``, ``PEER_LATENT_COLS``).
        routed_store_form (str): base body, static loop only: how the routed partial fills the routed all-reduce
            source (``ROUTED_STORE_FORMS``). ``relayout`` (default) keeps the program: the expert loop stores its
            [T, 3584] partial to HBM, and one DMA of 128 * T elements of 56 bytes relayouts it into the source.
            ``pe``: the expert loop writes the source itself (``column_groups_out`` of ``moe_tkg_decode``: TensorE
            transposes of its SBUF partial), with no HBM store and no relayout DMA; with an HBM source, one
            contiguous DMA stores it. The source bytes do not change. Without the persistent ``routed_src_sb``
            source, ``pe`` needs ``half_weight_buffers`` or 3 ``weight_buffers`` (``check_routed_store_buffers``).
        glue_dmas (tuple): names of ``GLUE_DMA_FORMS``, the forms of the small DMAs between the stages: any of them
            but ``split_gather`` with a TP128 kernel, and ``BASE_GLUE_DMA_FORMS`` (``replica_rows``, DP4;
            ``split_gather``) in the base body. The values do not change; () (default) keeps the program.
        routing_tiles (bool): ``in_proj_form`` ``dp_local`` with the explicit SBUF region (``region``) and an
            index-first ``routing_form``: the three [T, 16] outputs of the top-16 routing (the top scores, the index and
            the affinities) take stack addresses of the MoE frame of the region, below the buffers of the routed
            kernel. With compiler placement they can land inside a weight slot of ``sel_load``, and the first weight
            gather then waits for the last read of the affinities (a write-after-read) instead of its index. The
            values do not change; False (default) keeps the program.
        hot_experts (dict): the hot-expert tiles of the layer (``decode_block.hot_expert_tiles``) for the kimi_t32
            sel_load body, which takes them as ``prefetch_resident`` with ``dedupe_cap_positions``. The dict passes
            through; the body may fill its ``last_use``. Only when the body takes it (``HOT_EXPERT_BODY``). None
            (default) keeps the program.
        direct_latent (bool): base body, static loop only: the expert loop reads the gathered latent tile in place
            (``direct_latent`` of ``moe_tkg_decode``), without its SBUF-to-SBUF copy DMA. The tile is written once,
            by the Vector copy from the gather destination, and its last readers are the expert loop's latent
            transposes. The values do not change; False (default) keeps the program.
        latent_shard_form (str): base body only: the form of the transposed latent shard of the latent norm
            (``LATENT_SHARD_FORMS``). ``columns`` takes 28 matmuls, a copy and a transpose in place of T matmuls, with
            T <= 128; bitwise the same output. ``one_hot`` (default) keeps the program.
        out_proj_form (str): TP128 kernels only: the form of the out-projection (``OUT_PROJ_FORMS``). ``reduced``
            (default) keeps the program. ``column`` takes the column layout of ``moe_out_proj`` and ``moe_vectors``
            with the latent norm weights after the columns of the in-projection form
            (``moe_decode_torch.latent_moe_column_rank_weights``); its output is not the bits of ``reduced``.
            ``reduced_wide`` takes the weights of ``reduced`` and gives its bits if the routed all-reduce adds in the
            same order at the larger message (a diagnostic). ``column_shared_ag`` (sel_load) takes the weights of
            ``column`` and gives its bits, with the shared activation in its own world all-gather.
        preamble_buffers (int): base body, static loop with whole weight buffers only: K in 1 to 3 extra whole weight
            buffers for the first K projections of the expert loop (``preamble_buffers`` of ``moe_tkg_decode``), so
            they load during the work before the first expert matmul. The values do not change; 0 (default) keeps
            the program.
        preamble_reuse (bool): with ``preamble_buffers``: each extra buffer also takes one more projection of the
            next expert (``preamble_reuse`` of ``moe_tkg_decode``; gate 1 for 1 buffer and 2 ring buffers). The values
            do not change; False (default) keeps the program.
        in_proj_scale (nl.NkiTensor): None (default, the BF16 ``moe_in_proj``), or the scales of an MXFP8
            in-projection (``MX_IN_PROJ_SHAPES``): ``moe_in_proj`` then holds the MXFP8 latent down and shared gate / up
            columns (``MXFP8_IN_PROJ_COLS``, no router columns) in the hT_quad layout of ``tkg_mx_quant``, and they run
            MXFP8 x MXFP8 on the tokens quantized per token on VectorE. It needs an FP16 router (``fp16`` or
            ``fp16_local``) and the gathered in-projection, and it is not with ``latent_gather_mx``. The values change.
        router_rows_drop_selection (bool): base body with the HBM latent gather of the MLA loop path only: the world
            gather rows leave out the 7 selection values (21 words in place of 28, 25 % fewer bytes), and each rank
            adds the router bias of all 896 experts (``ALL_ROUTER_BIAS_COLS`` of ``moe_vectors`` [1, 931]) to the
            gathered scores on the Vector engine, as it adds its own 7 biases. ``sources`` must hold the 21-word latent
            gather tensors (``moe_collective_hbm`` ``gather_cols``). False (default) keeps the program.
        latent_gather_mx (bool): with ``router_rows_drop_selection``, ``latent_load`` ``pe32`` and the static expert
            loop of the base body: the world gather carries the expert input already quantized to MXFP8. Rank r < 112
            computes latent columns [32 r, 32 r + 32) (``moe_in_proj`` [128, 56, 135], ``MX_IN_PROJ_COLS``; ranks
            112..127 have zero latent columns), quantizes that one 32-column block with ``quantize_mx``
            (``_mx_quantize_block``), and sends it with its 7 router scores in one UINT32 [8, 2 T] block (8 more pad
            words at T = 32; ``_mx_gather_block``): 64 bytes per token in place of 112. Each rank loads the codes and
            the E8M0 scales straight into the MX layout of the expert input (``_load_mx_gather``) and runs no latent
            transpose or quantize; the expert loop takes them as ``quantized_latent`` of ``moe_tkg_decode``. A block's
            codes and scale are the bytes that ``moe_routed._quantize_natural_latent`` makes of the whole gathered
            latent, and the scores move by byte copies and bit-accurate TensorE transposes, so the output keeps its
            bits. ``sources`` must hold the UINT32 gather tensors (``moe_collective_hbm`` ``latent_gather_mx``); a
            ``direct_latent`` request has no effect (there is no latent tile). False (default) keeps the program.
        static_alt_tiles (bool | str): base body, static loop only: the expert loop takes its gate sigmoid, up tanh,
            down PSUM copy and activation eviction tiles from two buffers each, so that the compiler cannot stack them
            on one address (``static_alt_tiles`` of ``moe_tkg_decode``): True all of them, "act" only the sigmoid /
            tanh tiles, "act_copy" those and the PSUM copies (``STATIC_ALT_TILE_FORMS``). The values do not change;
            False (default) keeps the program.

    Returns:
        out_hT (nl.NkiTensor): [128, B, 56] BF16 SBUF, this replica's MoE output in the
            token-major hT layout.

    Notes:
        - The router is sharded: each rank computes the scores and the selection values of its
          7 experts, and the world gather that carries the latent also carries them. The 16th
          largest selection value of each token is the selection threshold.
        - Collectives: all-gather (DP, x), all-gather (world, latent and scores), all-reduce
          (world, routed latent), all-reduce (world, output). The routed all-reduce replaces a
          reduce-scatter and a norm-sum all-reduce: each rank sums the squares of the full
          routed latent itself.
        - Weight DMA gates (``tkg_weight_dma_gate``): the routed and the output all-reduces release the gate
          that holds the weight DMAs of the next body.
        - SBUF across the expert call: the inlined expert kernel needs almost all of SBUF, and
          each caller tile that stays live across the call makes the experts slower. Only the
          [76, T] out-projection input and the one-hot vectors stay live. The out-projection
          weight loads after the call, and the shared down runs after the call.
        - The out-projection PSUM [128, 56, T] FP32 needs ceil(56 T / 512) banks (2 at T = 16), so
          it runs in groups of hidden tiles, one bank for each group.

    Pseudocode:
        X = all_gather_dp(x)                                  # [T, H]
        r, latent_g, shared_g, shared_u = X @ moe_in_proj     # this rank's shards
        scores, selection = sigmoid(r), sigmoid(r) + bias
        gathered = all_gather_world([bf16(latent_g), scores, selection])
        threshold = 16th largest selection value per token
        affinity = local score * (local selection >= threshold) / (selected score sum + 1e-20)
        partial = moe_tkg_decode(latent, affinity, local experts)          # [T, L]
        routed = all_reduce_world(partial)                                 # [T, L]
        normed = rms_norm(routed, w_norm)[:, 28 g : 28 g + 28]             # [T, 28]
        y = bf16([normed | situ(shared_g, shared_u)] @ moe_out_proj)       # [T, H]
        out = all_reduce_world(y)[rows of this replica]
    """
    if routing_form is None:
        routing_form = "index_first" if routed_kernel in TP128_KERNELS else "base"
    options = moe_options_in_use(routed_kernel, router_mode, router_dtype, routed_reduce_dtype)
    options = options or region is not None or routing_debug is not None or expert_offset != 0
    options = options or affinity_scaling != "pre_quant" or not affinity_hi_lo or routing_form != "base"
    options = options or in_proj_form != "gathered" or out_proj_form != "reduced"
    kernel_assert(not (dynamic_experts and options), "The dynamic expert loop runs only in the base MoE body")
    kernel_assert(
        not (active_expert_slots and options),
        "The active-expert slots run only in the base MoE body (no MoE option, such as an FP16 router)",
    )
    kernel_assert(
        active_expert_count is None or active_expert_slots,
        "active_expert_count reports the active-expert slots: it needs active_expert_slots > 0",
    )
    kernel_assert(
        not (dynamic_experts and active_expert_slots),
        "A MoE layer runs one expert path: the dynamic loop or the active-expert slots, not both",
    )
    # The dynamic loop and the slots read the standard layout, so half-major weights there would give wrong values.
    kernel_assert(
        not (gate_up_half_major and (dynamic_experts or active_expert_slots)),
        "gate_up_half_major applies to the static loop only",
    )
    kernel_assert(
        not (shared_scale_buffer and (dynamic_experts or active_expert_slots)),
        "shared_scale_buffer applies to the static loop only",
    )
    kernel_assert(not (options and latent_load != "strided"), "latent_load runs only in the base MoE body")
    kernel_assert(
        routed_store_form in ROUTED_STORE_FORMS,
        f"routed_store_form {routed_store_form!r} must be one of {ROUTED_STORE_FORMS}",
    )
    kernel_assert(
        routed_store_form == "relayout" or not (options or dynamic_experts or active_expert_slots),
        "routed_store_form='pe' needs the static expert loop of the base MoE body",
    )
    check_routed_store_buffers(
        routed_store_form, sources is not None and "routed_src_sb" in sources, weight_buffers, half_weight_buffers
    )
    kernel_assert(
        not direct_latent or not (options or dynamic_experts or active_expert_slots),
        "direct_latent needs the static expert loop of the base MoE body",
    )
    kernel_assert(
        static_alt_tiles in STATIC_ALT_TILE_FORMS,
        f"static_alt_tiles must be one of {STATIC_ALT_TILE_FORMS}, got {static_alt_tiles!r}",
    )
    kernel_assert(
        not static_alt_tiles or not (options or dynamic_experts or active_expert_slots),
        "static_alt_tiles needs the static expert loop of the base MoE body",
    )
    kernel_assert(
        not preamble_buffers or not (options or dynamic_experts or active_expert_slots),
        "preamble_buffers needs the static expert loop of the base MoE body",
    )
    kernel_assert(not preamble_reuse or preamble_buffers, "preamble_reuse needs preamble_buffers")
    kernel_assert(
        not sel_load_options or routed_kernel == "sel_load", "sel_load_options needs routed_kernel='sel_load'"
    )
    kernel_assert(
        not glue_dmas
        or routed_kernel in TP128_KERNELS
        or (not options and all(form in BASE_GLUE_DMA_FORMS for form in glue_dmas)),
        f"glue_dmas {glue_dmas}: the base body takes {BASE_GLUE_DMA_FORMS} only; the others need a TP128 routed kernel",
    )
    kernel_assert(not routing_tiles or routed_kernel in TP128_KERNELS, "routing_tiles needs a TP128 routed kernel")
    kernel_assert(
        latent_shard_form in LATENT_SHARD_FORMS,
        f"latent_shard_form must be one of {LATENT_SHARD_FORMS}, got {latent_shard_form!r}",
    )
    kernel_assert(
        latent_shard_form == "one_hot" or (not options and dp_size * x_hT.shape[1] <= _PMAX),
        f"latent_shard_form {latent_shard_form!r} needs the base body and at most {_PMAX} tokens",
    )
    kernel_assert(
        in_proj_scale is None or (router_dtype != "bf16" and in_proj_form == "gathered"),
        "the MXFP8 in-projection (in_proj_scale) needs an FP16 router and the gathered in-projection",
    )
    kernel_assert(
        not (router_rows_drop_selection and options),
        "router_rows_drop_selection needs the base MoE body (the EP128 static kernel and no MoE option)",
    )
    kernel_assert(
        not latent_gather_mx
        or (router_rows_drop_selection and not (options or dynamic_experts or active_expert_slots)),
        "latent_gather_mx needs router_rows_drop_selection and the static expert loop of the base MoE body",
    )
    kernel_assert(
        not latent_gather_mx or latent_load == "pe32",
        f"latent_gather_mx loads the router rows of the gather as pe32 rank rows: it needs latent_load 'pe32', got "
        f"{latent_load!r}",
    )
    kernel_assert(
        not latent_gather_mx or (router_dtype == "bf16" and in_proj_scale is None),
        "latent_gather_mx takes the router logits from the BF16 in-projection of the block layout: not with an FP16 "
        "router or in_proj_scale",
    )
    if options:
        return _routed_latent_moe_decode(
            x_hT,
            moe_in_proj,
            moe_out_proj,
            moe_vectors,
            expert_gate_up_weights,
            expert_gate_up_weights_scale,
            expert_down_weights,
            expert_down_weights_scale,
            rank_info,
            dp_group,
            world_group,
            tag,
            weight_buffers,
            scale_buffers,
            sources,
            routed_kernel,
            region,
            routed_reduce_dtype,
            routing_debug,
            router_mode,
            router_dtype,
            router_weight,
            x16_hT,
            expert_offset,
            affinity_scaling,
            affinity_hi_lo,
            early_loads,
            dp_size,
            routing_form,
            sel_load_options,
            half_weight_buffers,
            gate_up_half_major=gate_up_half_major,
            in_proj_form=in_proj_form,
            peer_in_proj=peer_in_proj,
            **({"glue_dmas": tuple(glue_dmas)} if glue_dmas else {}),
            out_proj_form=out_proj_form,
            **({"routing_tiles": True} if routing_tiles else {}),
            **({"shared_scale_buffer": True} if shared_scale_buffer else {}),
            **({"hot_experts": hot_experts} if hot_experts is not None else {}),
            **({"in_proj_scale": in_proj_scale} if in_proj_scale is not None else {}),
        )
    kernel_assert(hot_experts is None, "hot_experts needs the sel_load routed kernel")
    kernel_assert(in_proj_form == "gathered", "in_proj_form 'dp_local' needs the TP128 body")
    # The base body: glue_dmas holds BASE_GLUE_DMA_FORMS only (checked above).
    replica_rows = "replica_rows" in glue_dmas
    split_gather = "split_gather" in glue_dmas
    kernel_assert(
        not replica_rows or dp_size == DP_SIZE, f"glue_dmas 'replica_rows' needs the DP{DP_SIZE} layout, got {dp_size}"
    )
    _validate_moe_inputs(
        x_hT,
        moe_in_proj,
        moe_out_proj,
        moe_vectors,
        expert_gate_up_weights,
        expert_gate_up_weights_scale,
        expert_down_weights,
        expert_down_weights_scale,
        rank_info,
        in_proj_scale=in_proj_scale,
        **({"drop_selection": True} if router_rows_drop_selection else {}),
        **({"latent_gather_mx": True} if latent_gather_mx else {}),
    )
    fp16_local = router_dtype == "fp16_local"
    if fp16_local:
        _validate_base_fp16_router(x_hT, x16_hT, router_weight, router_dtype, dp_size, sources)
    else:
        kernel_assert(router_dtype == "bf16", f"the base MoE body takes router_dtype {BASE_ROUTER_DTYPES}")
    replica_tokens, tokens = _moe_token_counts(x_hT, sources, dp_size)
    in_cols = MX_IN_PROJ_COLS if latent_gather_mx else MOE_IN_PROJ_COLS
    name = f"k3moe_{tag}"

    # Issue the weight loads first; they depend on no compute.
    if fp16_local:
        # The FP16 router rows of the DP peers go first: the local logits ride the DP x gather.
        router_w_sb = nl.ndarray(tuple(router_weight.shape), dtype=nl.float16, buffer=nl.sbuf)
        nisa.dma_copy(dst=router_w_sb, src=router_weight, priority=_CRITICAL_DMA_PRIORITY, name=f"{name}_router_w16")
    if in_proj_scale is not None:
        in_proj_sb = _load_mx_in_proj(moe_in_proj, in_proj_scale, name)
    else:
        in_proj_sb = nl.ndarray((_PMAX, _H_TILES, in_cols), dtype=nl.bfloat16, buffer=nl.sbuf)
        _load_hT_rows(in_proj_sb, moe_in_proj, _IN_PROJ_DMA_CHUNKS, f"{name}_in_proj")
    local_bias_sb = nl.ndarray((tokens, LOCAL_EXPERTS), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=local_bias_sb,
        src=moe_vectors.ap(pattern=[[0, tokens], [1, LOCAL_EXPERTS]], offset=LOCAL_BIAS_COLS[0]),
        priority=_CRITICAL_DMA_PRIORITY,
        name=f"{name}_local_bias",
    )

    router_logits = None
    if fp16_local and (sources or {}).get("x_token_world_src_hbm") is not None:
        # attn_rs_moe_input: the FP16 rows ride the world token gather; the logits of the peer rows give this rank's.
        row_select, rank_select = rank_selectors(rank_info, f"{name}_rank", dp_size=dp_size)
        x_tokens_hT, x16_tokens_hT = _gather_replica_tokens(
            x_hT,
            dp_group,
            f"{name}_x",
            sources,
            dp_size,
            world_group,
            x16_hT=x16_hT,
            **({"split": True} if split_gather else {}),
        )
        router_logits = _token_router_logits(x16_tokens_hT, router_w_sb, row_select, tokens)
    elif fp16_local:
        row_select, rank_select = rank_selectors(rank_info, f"{name}_rank", dp_size=dp_size)
        x_tokens_hT, peer_logits = _gather_replica_tokens(
            x_hT,
            dp_group,
            f"{name}_x",
            sources,
            dp_size,
            world_group,
            local_logits=_peer_router_logits(x16_hT, router_w_sb),
            **({"split": True} if split_gather else {}),
        )
        router_logits = _own_router_logits(peer_logits, row_select, tokens)
    else:
        x_tokens_hT = _gather_replica_tokens(
            x_hT, dp_group, f"{name}_x", sources, dp_size, world_group, **({"split": True} if split_gather else {})
        )
        row_select, rank_select = rank_selectors(rank_info, f"{name}_rank", dp_size=dp_size)

    # Fused in-projection: [T, 131] = X @ moe_in_proj, the tokens as the stationary operand ([T, 135] with
    # latent_gather_mx).
    in_psum = _fused_in_proj(x_tokens_hT, in_proj_sb, name, in_cols)

    if latent_gather_mx:
        gather_hbm = sources.get("gather_hbm") if sources else None
        mx_dst = sources.get("gather_dst_hbm") if sources else None
        kernel_assert(
            gather_hbm is not None
            and tuple(gather_hbm.shape) == mx_gather_block_shape(tokens)
            and gather_hbm.dtype == nl.uint32
            and tuple(mx_dst.shape) == (WORLD_SIZE * _MX_CODE_ROWS, mx_gather_block_shape(tokens)[1])
            and mx_dst.dtype == nl.uint32,
            f"latent_gather_mx needs the UINT32 HBM latent gather of the MLA loop path, source "
            f"{mx_gather_block_shape(tokens)} (moe_collective_hbm latent_gather_mx), got "
            f"{None if gather_hbm is None else (gather_hbm.dtype, tuple(gather_hbm.shape))}",
        )
        local_scores, local_selection, store = _mx_gather_block(in_psum, local_bias_sb, gather_hbm, tokens, name)
        # An HBM all-gather stacks the ranks on dim 0: rows 8 r .. 8 r + 7 are the block of rank r.
        ncc.all_gather(
            srcs=[gather_hbm], dsts=[mx_dst], replica_group=world_group, collective_dim=0, name=f"{name}_latent_gather"
        )
        # Behind the block stores: the bias streams in the flight of the gather.
        router_bias = _load_router_bias(moe_vectors, tokens, name, after=store)
        mx_router, mx_quant, mx_scale = _load_mx_gather(mx_dst, tokens, name)
        _add_router_bias(mx_router, router_bias, _MX_ROUTER_SCORE_COLS, _MX_ROUTER_SELECTION_COLS)
    else:
        mx_router = mx_quant = mx_scale = None

    # World gather source: the BF16 latent shard packed in FP32 words, and the FP32 router scores
    # and selection values of the local experts.
    gather_src = None if latent_gather_mx else _source(sources, "gather", (tokens, _GATHER_COLS), nl.float32)
    latent_loaded = None
    if not latent_gather_mx:
        nisa.tensor_copy(
            dst=gather_src[:, _GATHER_LATENT_WORDS[0] : _GATHER_LATENT_WORDS[1]].view(nl.bfloat16),
            src=in_psum[:, LATENT_COLS[0] : LATENT_COLS[1]],
        )
        local_scores = gather_src[:, _GATHER_SCORE_COLS[0] : _GATHER_SCORE_COLS[1]]
        logits = in_psum[:, ROUTER_COLS[0] : ROUTER_COLS[1]] if router_logits is None else router_logits
        nisa.activation(dst=local_scores, op=nl.sigmoid, data=logits)
        local_selection = gather_src[:, _GATHER_SELECTION_COLS[0] : _GATHER_SELECTION_COLS[1]]
        if router_rows_drop_selection:
            # The local selection values stay on this rank: the Vector add of the gathered ones (_add_router_bias).
            nisa.tensor_tensor(
                dst=local_selection, data1=local_scores, data2=local_bias_sb, op=nl.add, engine=nisa.engine.vector
            )
        else:
            nisa.tensor_tensor(dst=local_selection, data1=local_scores, data2=local_bias_sb, op=nl.add)
        gather_shape = (tokens, WORLD_SIZE * _GATHER_COLS)
        gather_hbm = sources.get("gather_hbm") if sources else None
        gather_cols = SCORE_GATHER_COLS if router_rows_drop_selection else _GATHER_COLS
        kernel_assert(
            not router_rows_drop_selection
            or (gather_hbm is not None and tuple(gather_hbm.shape) == (tokens, gather_cols)),
            f"router_rows_drop_selection needs the HBM latent gather of the MLA loop path with {gather_cols}-word rows "
            f"(moe_collective_hbm gather_cols), got {None if gather_hbm is None else tuple(gather_hbm.shape)}",
        )
        if sources and "gather_dst" in sources:
            gather_dst = _source(sources, "gather_dst", gather_shape, nl.float32)
        else:
            gather_dst = nl.ndarray(gather_shape, dtype=nl.float32, buffer=nl.sbuf)
        if gather_hbm is None:
            ncc.all_gather(
                srcs=[gather_src],
                dsts=[gather_dst],
                replica_group=world_group,
                collective_dim=1,
                name=f"{name}_latent_gather",
            )
        else:
            # The gather runs in persistent HBM tensors, so its destination takes no persistent SBUF.
            store = nisa.dma_copy(
                dst=gather_hbm,
                src=gather_src if gather_cols == _GATHER_COLS else gather_src[:, 0:gather_cols],
                priority=_CRITICAL_DMA_PRIORITY,
                name=f"{name}_latent_store",
                # The 21-word store names its queue; the 28-word store keeps the program of the base body.
                **(
                    {} if gather_cols == _GATHER_COLS else {"dge_mode": nisa.dge_mode.hwdge, "engine": nisa.engine.sync}
                ),
            )
            # An HBM all-gather stacks the ranks on dim 0: row r * T + t is token t of rank r.
            check_pair_group(sources["gather_dst_hbm"], world_group, f"{name}_latent_gather")
            ncc.all_gather(
                srcs=[gather_hbm],
                dsts=[sources["gather_dst_hbm"]],
                replica_group=world_group,
                collective_dim=0,
                name=f"{name}_latent_gather",
            )
            router_bias = None
            if router_rows_drop_selection:
                # Behind the latent store: the bias streams in the flight of the gather.
                router_bias = _load_router_bias(moe_vectors, tokens, name, after=store)
            latent_loaded = _load_world_gather(
                gather_dst,
                sources["gather_dst_hbm"],
                tokens,
                name,
                latent_load,
                gather_cols,
                **({"split": True} if split_gather else {}),
            )
            if router_bias is not None:
                _add_router_bias(gather_dst.reshape((tokens, WORLD_SIZE, _GATHER_COLS)), router_bias)

    # Out-projection input [76, T]: rows 28..75 get the shared-expert activation now, and rows
    # 0..27 get the normalized routed latent after the experts.
    out_proj_input = (
        _shared_activation_T(in_psum, col_offset=_MX_COL_SHIFT) if latent_gather_mx else _shared_activation_T(in_psum)
    )

    if latent_gather_mx:
        latent_sb = None
        local_affinities = _local_affinities(
            mx_router, local_scores, local_selection, _MX_ROUTER_SCORE_COLS, _MX_ROUTER_SELECTION_COLS
        )
    else:
        # Gathered latent [T, L] in natural column order, for the expert kernel. Only the Vector
        # engine reads gather_dst: the inlined expert kernel reuses its SBUF, and its first Vector
        # instruction (the accumulator memset) can overwrite it while a Scalar or GpSimd copy still
        # reads it. On the Vector engine the copy and that memset run in program order. A strided
        # SBUF-to-SBUF DMA of 56-byte rows is slower. With direct_latent the expert loop's transposes read latent_sb
        # itself.
        gathered = gather_dst.reshape((tokens, WORLD_SIZE, _GATHER_COLS))
        latent_sb = nl.ndarray((tokens, WORLD_SIZE, LATENT_SHARD), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(
            dst=latent_sb,
            src=gathered[:, :, _GATHER_LATENT_WORDS[0] : _GATHER_LATENT_WORDS[1]].view(nl.bfloat16),
            engine=nisa.engine.vector,
        )
        local_affinities = _local_affinities(gathered, local_scores, local_selection)

    if expert_gate_up_weights.dtype == nl.uint16:
        expert_gate_up_weights = expert_gate_up_weights.view(nl.float4_e2m1fn_x4)
    if expert_down_weights.dtype == nl.uint16:
        expert_down_weights = expert_down_weights.view(nl.float4_e2m1fn_x4)
    # The nested kernel call takes positional arguments only.
    if dynamic_experts:
        # The opt-in dynamic expert loop (``dynamic_expert_loop``): only the active local experts run.
        expert_partial = moe_tkg_decode(
            latent_sb.reshape((tokens, LATENT_SIZE)),
            local_affinities,
            expert_gate_up_weights,
            expert_down_weights,
            expert_gate_up_weights_scale,
            expert_down_weights_scale,
            SITU_BETA,
            SITU_LINEAR_BETA,
            weight_buffers,
            scale_buffers,
            early_loads,
            True,
        )
    elif active_expert_slots:
        # The opt-in active-expert slots (``routed_expert_path``): the first K active local experts in straight-line
        # slots, and every expert in a one-trip loop when more are active.
        expert_partial = moe_tkg_decode(
            latent_sb.reshape((tokens, LATENT_SIZE)),
            local_affinities,
            expert_gate_up_weights,
            expert_down_weights,
            expert_gate_up_weights_scale,
            expert_down_weights_scale,
            SITU_BETA,
            SITU_LINEAR_BETA,
            weight_buffers,
            scale_buffers,
            early_loads,
            False,
            0,
            active_expert_slots,
            # The debug count, only when asked for: the call without it does not change.
            *(() if active_expert_count is None else (active_expert_count,)),
        )
    else:
        # routed_store_form "pe": the expert loop writes the routed all-reduce source in its [128, T, 28] layout.
        pe_store = {}
        if routed_store_form == "pe":
            pe_sbuf = sources is not None and "routed_src_sb" in sources
            if pe_sbuf:
                pe_dst = _source(sources, "routed_src_sb", (WORLD_SIZE, tokens, LATENT_SHARD), nl.bfloat16)
            else:
                pe_dst = nl.ndarray((WORLD_SIZE, tokens, LATENT_SHARD), dtype=nl.bfloat16, buffer=nl.sbuf)
            pe_store = {"column_groups_out": pe_dst}
        expert_partial = moe_tkg_decode(
            None if latent_gather_mx else latent_sb.reshape((tokens, LATENT_SIZE)),
            local_affinities,
            expert_gate_up_weights,
            expert_down_weights,
            expert_gate_up_weights_scale,
            expert_down_weights_scale,
            SITU_BETA,
            SITU_LINEAR_BETA,
            weight_buffers,
            scale_buffers,
            early_loads,
            # Only when asked for: the default call does not change.
            **({"half_weight_buffers": half_weight_buffers} if half_weight_buffers else {}),
            **({"gate_up_half_major": True} if gate_up_half_major else {}),
            **pe_store,
            **({"direct_latent": True} if direct_latent and not latent_gather_mx else {}),
            **({"quantized_latent": (mx_quant, mx_scale)} if latent_gather_mx else {}),
            **({"shared_scale_buffer": True} if shared_scale_buffer else {}),
            **_ring_buffer_args(sources, half_weight_buffers, latent_loaded),
            **({"static_alt_tiles": static_alt_tiles} if static_alt_tiles else {}),
            **({"preamble_buffers": preamble_buffers} if preamble_buffers else {}),
            **({"preamble_reuse": True} if preamble_reuse else {}),
        )

    # All-reduce of the routed partial in [128, T * 28] layout: partition g holds latent columns
    # [28 g, 28 g + 28) of all tokens. Every rank gets the full routed latent, so the norm needs
    # no second collective; rank g then keeps partition g (the latent shard of its latent up).
    # The all-reduce runs in HBM: an SBUF all-reduce gets a compiler scratch that can take the bytes of a tile
    # that the rank still reads. The Neuron runtime does not touch it for a group across Trn3 chips; its Trn2 mesh
    # algorithms let a peer write it only after a forced global handshake. With ``sbuf_routed_reduce`` it runs in
    # SBUF, from the persistent source of ``moe_collective_tiles`` into the persistent ``routed_dst``.
    sbuf_routed = sources is not None and "routed_src_sb" in sources
    if sbuf_routed:
        routed_src = _source(sources, "routed_src_sb", (WORLD_SIZE, tokens, LATENT_SHARD), nl.bfloat16)
        rr = tokens
    else:
        routed_src, routed_hbm, rr = _routed_reduce_hbm(sources, tokens)
    routed_store = routed_src if rr == tokens else routed_src[0:WORLD_SIZE, 0:tokens, 0:LATENT_SHARD]
    store = None
    if routed_store_form == "relayout":
        store = nisa.dma_copy(
            dst=routed_store,
            src=expert_partial.ap(
                pattern=[[LATENT_SHARD, WORLD_SIZE], [LATENT_SIZE, tokens], [1, LATENT_SHARD]],
                offset=0,
            ),
            priority=_CRITICAL_DMA_PRIORITY,
            name=f"{name}_routed_store",
        )
    elif not sbuf_routed:
        # The partial is already in the source layout: 128 contiguous rows of T * 56 bytes.
        store = nisa.dma_copy(
            dst=routed_store, src=expert_partial, priority=_CRITICAL_DMA_PRIORITY, name=f"{name}_routed_store"
        )
    routed = _source(sources, "routed_dst", (WORLD_SIZE, tokens * LATENT_SHARD), nl.bfloat16)
    if sbuf_routed:
        routed_hbm = routed
    check_pair_group(routed_hbm, world_group, f"{name}_routed_reduce")
    reduce = release_weight_gate(
        ncc.all_reduce(
            srcs=[routed_src.reshape((WORLD_SIZE, rr * LATENT_SHARD))],
            dsts=[routed_hbm],
            replica_group=world_group,
            op=nl.add,
            name=f"{name}_routed_reduce",
        )
    )
    # The consumer of the routed output for the next layer's row sums (common.mark_ffn_output): the store, or the
    # all-reduce of an SBUF source.
    mark_ffn_output(reduce if store is None else store)
    if not sbuf_routed:
        routed_load = routed_hbm if rr == tokens else routed_hbm[0:WORLD_SIZE, 0 : tokens * LATENT_SHARD]
        nisa.dma_copy(dst=routed, src=routed_load, priority=_CRITICAL_DMA_PRIORITY, name=f"{name}_routed_load")
    out_proj_sb = nl.ndarray((MOE_OUT_PROJ_ROWS, HIDDEN_SIZE), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=out_proj_sb,
        src=moe_out_proj,
        priority=_CRITICAL_DMA_PRIORITY,
        name=f"{name}_out_proj",
    )
    _latent_norm(
        routed,
        moe_vectors,
        rank_select,
        out_proj_input[LATENT_UP_ROWS[0] : LATENT_UP_ROWS[1], :],
        f"{name}_norm",
        **({"shard_form": latent_shard_form} if latent_shard_form != "one_hot" else {}),
    )

    # Latent up and shared down in one K = 76 matmul for each hidden tile.
    partial_hT = _project_to_hT(
        lambda h_tile_idx: [(out_proj_sb[:, h_tile_idx * _PMAX : (h_tile_idx + 1) * _PMAX], out_proj_input)],
        tokens,
        _tile(sources, "out", (_PMAX, tokens, _H_TILES), nl.bfloat16),
    )
    return _all_reduce_select_rows(
        partial_hT,
        row_select,
        replica_tokens,
        world_group,
        f"{name}_out",
        sources,
        dp_size,
        **({"replica_index": replica_index(rank_info)} if replica_rows else {}),
    )


latent_moe_decode.token_major = True
latent_moe_decode.in_proj_mx = True
HOT_EXPERT_BODY = "prefetch_resident" in inspect.signature(kimi_t32_body).parameters
"""True when the kimi_t32 body takes the hot-expert tiles (``prefetch_resident``)."""
latent_moe_decode.hot_experts = HOT_EXPERT_BODY


TP128_LATENT_PE_OPTION = "latent_pe"
"""The sel_load option (in ``sel_load_options``) of the latent load of the TP128 kernels from the HBM world gather.
True (with a rank-row routing form of ``_PE32_FORMS``): the latent words come from the contiguous rank rows of the
gather through TensorE transposes of each FP32 word (``_latent_words_from_rank_rows``), in place of the strided load of
128 * T rows of 112 bytes (a slow DMA at T = 32 on trn3). The transposes keep the bits, so the values do not
change."""


def _latent_words_from_rank_rows(rank_rows: nl.NkiTensor, tokens: int) -> nl.NkiTensor:
    """Return the gathered words [T, 128, 28] FP32 SBUF with the latent words filled from the rank rows of the gather.

    ``rank_rows`` is SBUF [128, T * 28]: rank r on partition r, token t at columns [28 t, 28 t + 28). Each latent word
    w (0 .. 13) moves to [T, 128] in PSUM with a TensorE transpose (bit-accurate on trn3), 4 words for each PSUM tile,
    and the Vector and Scalar engines copy the tiles in turn to words w of the [T, 128, 28] tile, as the ``pe32``
    latent load of the base body does (``_load_world_gather``). The router words of the tile stay unwritten:
    ``_gathered_latent`` reads the latent words only.
    """
    gathered = nl.ndarray((tokens, WORLD_SIZE, _GATHER_COLS), dtype=nl.float32, buffer=nl.sbuf)
    src3 = rank_rows.reshape((WORLD_SIZE, tokens, _GATHER_COLS))
    group = 4
    first, last = _GATHER_LATENT_WORDS
    for gi, w0 in enumerate(range(first, last, group)):
        width = min(group, last - w0)
        psum = nl.ndarray((tokens, width * WORLD_SIZE), dtype=nl.float32, buffer=nl.psum)
        for j in range(width):
            nisa.nc_transpose(
                dst=psum[0:tokens, j * WORLD_SIZE : (j + 1) * WORLD_SIZE], data=src3[0:WORLD_SIZE, 0:tokens, w0 + j]
            )
        nisa.tensor_copy(
            dst=gathered[0:tokens, 0:WORLD_SIZE, w0 : w0 + width],
            src=psum.ap(pattern=[[width * WORLD_SIZE, tokens], [1, WORLD_SIZE], [WORLD_SIZE, width]], offset=0),
            engine=(nisa.engine.vector, nisa.engine.scalar)[gi % 2],
        )
    return gathered


TP128_ROUTED_STORE_OPTION = "routed_store_pe"
"""The sel_load option (in ``sel_load_options``) of the routed store of the TP128 kernels. True: the HBM routed partial
[T, 3584] loads to SBUF in one DMA of whole rows, TensorE transposes of its 32-bit words move it to the [128, T, 28]
column-group layout of the routed all-reduce source (``moe_routed._to_column_groups``), and one DMA of 1,792-byte
rows (or one Vector copy for the SBUF source) fills the source. False (default): one DMA of 128 * T rows of 56 bytes,
which is slow at T = 32 on trn3. The transposes and the copies keep the bits, so the values do not change."""


def _routed_partial_column_groups(expert_partial: nl.NkiTensor, tokens: int, sb, name: str) -> nl.NkiTensor:
    """Return the BF16 SBUF [128, T, 28] column groups of the HBM routed partial [T, 3584] (``routed_store_pe``).

    Group g holds the latent columns [28 g, 28 g + 28) of each token, the layout of the routed all-reduce source. The
    tiles come from ``sb`` (explicit addresses with a region, else compiler placement).
    """
    partial_sb = sb.tile((tokens, LATENT_SIZE), nl.bfloat16, f"{name}_routed_rows")
    nisa.dma_copy(dst=partial_sb, src=expert_partial, priority=_CRITICAL_DMA_PRIORITY, name=f"{name}_routed_rows_load")
    groups = sb.tile((WORLD_SIZE, tokens, LATENT_SHARD), nl.bfloat16, f"{name}_routed_groups")
    _to_column_groups(partial_sb, tokens, groups)
    return groups


def _pop_kimi_body_options(options: dict) -> tuple[bool, int]:
    """Pop the kimi body keys of the sel_load options: kimi_body (bool) and kimi_hwdge_calls (int, needs kimi_body)."""
    kimi_body = options.pop("kimi_body", False)
    hwdge_calls = options.pop("kimi_hwdge_calls", 0)
    kernel_assert(
        isinstance(hwdge_calls, int) and not isinstance(hwdge_calls, bool) and hwdge_calls >= 0,
        f"sel_load option kimi_hwdge_calls must be an integer >= 0, got {hwdge_calls!r}",
    )
    kernel_assert(kimi_body or not hwdge_calls, "sel_load option kimi_hwdge_calls needs kimi_body")
    kernel_assert(
        not options.get("overlap_calls"),
        "sel_load option overlap_calls is not supported: each kimi body call reads the output of the layer before",
    )
    return bool(kimi_body), hwdge_calls


def _kimi_body_call_options(options: dict, call: int, hwdge_calls: int) -> dict:
    """The kimi start options of kimi body call `call` of the trace: the first hwdge_calls calls load gate/up group 0
    by HWDGE (group0_gather False), the others by a Pool gather. A dynamic HWDGE load is slow on a DGE table slot
    above 128, so hwdge_calls must keep these calls below it (check the DGE table of each config's NEFF). With
    hwdge_calls > 0 every call takes reserve_hwdge_ids, so the first Pool call keeps the heap layout of the HWDGE calls."""
    reserve = {"reserve_hwdge_ids": True} if hwdge_calls else {}
    return {**options, "group0_gather": call >= hwdge_calls, **reserve}


def _kimi_t32_partial(
    routing,
    latent: nl.NkiTensor,
    expert_gate_up_weights: nl.NkiTensor,
    expert_down_weights: nl.NkiTensor,
    expert_down_weights_scale: nl.NkiTensor,
    expert_offset: int,
    region,
    name: str,
    hot_experts: dict = None,
    form: str = "base",
) -> nl.NkiTensor:
    """Return the HBM routed partial [T, 3584] of a 32-token MoE call from the kimi_t32 body (_KIMI_T32_OPTIONS).

    ``form`` is a name of ``KIMI_T32_FORMS``: its options replace the options of _KIMI_T32_OPTIONS.

    ``hot_experts`` (``decode_block.hot_expert_tiles``) goes to the body as ``prefetch_resident``, with
    ``dedupe_cap_positions``; the body uses the tiles as its ring slots, so only a one-call layer (T = 32) takes them.

    The body reads its inputs from HBM, so the latent, the top-16 index and affinities and the selected scores
    [T, 896] (the scores of the 16 routed experts, 0 elsewhere) go to HBM scratch first. It computes the affinities
    from the selected scores, so its output can differ from the served call's in the last bits.
    """
    expert_index, affinities, _ = routing
    selected = getattr(routing, "selected_scores", None)
    kernel_assert(selected is not None, "sel_load option kimi_t32_body needs the top-16 routing (_top16_routing)")
    tokens = expert_index.shape[0]
    # A 16-token call pads to one 32-token call. The padding rows repeat the index rows (valid expert ids) and have
    # zero latent, affinities and selected scores, so they add no experts and the body does not store them.
    rows = max(tokens, KIMI_T32_TOKENS)
    pad = rows - tokens
    kernel_assert(pad in (0, tokens), f"the kimi_t32 body pads 16 tokens to 32, got {tokens} tokens")
    inputs = []
    for key, src, cols, dtype in (
        ("latent", latent, LATENT_SIZE, nl.bfloat16),
        ("index", expert_index, TOP_K, nl.int32),
        ("affinities", affinities, TOP_K, nl.float32),
        ("selected", selected.reshape((tokens, NUM_EXPERTS)), NUM_EXPERTS, nl.float32),
    ):
        dst = nl.ndarray((rows, cols), dtype=dtype, buffer=nl.private_hbm, name=f"{name}_t32_{key}")
        nisa.dma_copy(dst=dst[0:tokens], src=src, priority=_CRITICAL_DMA_PRIORITY, name=f"{name}_t32_{key}_store")
        if pad and key == "index":
            nisa.dma_copy(dst=dst[tokens:rows], src=src, priority=_CRITICAL_DMA_PRIORITY, name=f"{name}_t32_index_pad")
        elif pad:
            zeros = nl.ndarray((pad, cols), dtype=dtype, buffer=nl.sbuf, name=f"{name}_t32_{key}_zeros")
            nisa.memset(dst=zeros, value=0, engine=nisa.engine.gpsimd, name=f"{name}_t32_{key}_zero_fill")
            nisa.dma_copy(dst=dst[tokens:rows], src=zeros, priority=_CRITICAL_DMA_PRIORITY, name=f"{name}_t32_{key}_pad")
        inputs.append(dst)
    output = nl.ndarray((rows, LATENT_SIZE), dtype=nl.bfloat16, buffer=nl.private_hbm, name=f"{name}_t32_out")
    # A 16-token call takes the table of _KIMI_T32_PAD16_OPTIONS, and a form can replace it.
    options = {**_KIMI_T32_OPTIONS, **(_KIMI_T32_PAD16_OPTIONS if pad else {}), **kimi_t32_form_options(form)}
    # A per-layer form (layer_static_groups) gives each MoE layer its own head.
    options = _kimi_t32_layer_options(options)
    padded = {"valid_tokens": tokens} if pad else {}
    sbm = getattr(region, "sbm", region)
    # One body call per KIMI_T32_TOKENS chunk: T = 32 is one call; T = 64 is two calls, each writing its 32 output
    # rows. The body's block groups are sized for the 32-token call, so a larger T runs in 32-token chunks.
    chunks = rows // KIMI_T32_TOKENS
    kernel_assert(hot_experts is None or chunks == 1, f"hot_experts needs one kimi_t32 call, got {chunks} calls")
    hot = {} if hot_experts is None else {"prefetch_resident": hot_experts, "dedupe_cap_positions": True}
    for c in range(chunks):
        lo = c * KIMI_T32_TOKENS
        hi = lo + KIMI_T32_TOKENS
        kimi_t32_body(
            inputs[0][lo:hi],
            expert_gate_up_weights,
            expert_down_weights,
            expert_down_weights_scale,
            inputs[1][lo:hi],
            inputs[2][lo:hi],
            inputs[3][lo:hi],
            sbm=sbm,
            **({} if sbm is not None else {"sbuf_reserve": 49152}),
            output=output,
            output_row=lo,
            expert_offset=expert_offset,
            n_experts=NUM_EXPERTS,
            name_suffix=(f"_{name}" if chunks == 1 else f"_{name}_c{c}"),
            **{**options, **padded},
            **hot,
        )
    return output[0:tokens] if pad else output


def _routed_latent_moe_decode(
    x_hT: nl.NkiTensor,
    moe_in_proj: nl.NkiTensor,
    moe_out_proj: nl.NkiTensor,
    moe_vectors: nl.NkiTensor,
    expert_gate_up_weights: nl.NkiTensor,
    expert_gate_up_weights_scale: nl.NkiTensor,
    expert_down_weights: nl.NkiTensor,
    expert_down_weights_scale: nl.NkiTensor,
    rank_info: nl.NkiTensor,
    dp_group: ReplicaGroup,
    world_group: ReplicaGroup,
    tag: str,
    weight_buffers: int,
    scale_buffers: int,
    sources: dict,
    routed_kernel: str,
    region: ExplicitSbuf,
    routed_reduce_dtype: str,
    routing_debug: tuple,
    router_mode: str,
    router_dtype: str,
    router_weight: nl.NkiTensor,
    x16_hT: nl.NkiTensor,
    expert_offset: int,
    affinity_scaling: str,
    affinity_hi_lo: bool,
    early_loads: bool,
    dp_size: int,
    routing_form: str = "base",
    sel_load_options: tuple = (),
    half_weight_buffers: int = 0,
    # Keyword-only from here: a positional mix-up of two defaulted options would run with the wrong values.
    *,
    gate_up_half_major: bool = False,
    in_proj_form: str = "gathered",
    peer_in_proj: nl.NkiTensor = None,
    glue_dmas: tuple = (),
    out_proj_form: str = "reduced",
    routing_tiles: bool = False,
    shared_scale_buffer: bool = False,
    hot_experts: dict = None,
    in_proj_scale: nl.NkiTensor = None,
) -> nl.NkiTensor:
    """Run the latent MoE layer of ``latent_moe_decode`` with the MoE options (the arguments of that function).

    The routed part depends on ``routed_kernel`` (``ROUTED_KERNELS``): EP128 runs the 7 local experts of the rank
    on all tokens (the static loop of ``moe_tkg_decode``), and TP128 (``sel_load``, ``tp_dense``)
    runs the rank's I-slice of the 16 routed experts of each token. Both give the rank's unnormalized partial of
    the routed latent, and the routed all-reduce adds them.

    TP128 weights (``SEL_LOAD_WEIGHT_NAMES``): ``expert_gate_up_weights`` is ``expert_gate_up_packed``
    [896, 128, 768] UINT8, the packed gate/up rows (weights and MX scales) of the rank's slices;
    ``expert_gate_up_weights_scale`` is unused (None); ``expert_down_weights`` is [896, 8, 3584] packed MXFP4 (or
    UINT16 carrier), the 8 MX rows of each slice; ``expert_down_weights_scale`` is [896, 3584] UINT8, or [896, 2, 3584]
    with two scale groups (``sel_load_down_groups``: the checkpoint down values, fp32 down PSUM). With
    ``expert_offset``, the three expert tensors are the stacks of all MoE layers ([L * 896, ...] rows).

    Notes:
        - ``sel_load`` needs the 16 routed experts of each token over all 896 experts: two max8 rounds over the
          selected scores give the expert index [T, 16] (descending score; equal scores in a fixed position order)
          and the scores, with the same affinity numerics as the EP128 path.
        - Stages: 1/ the router (scores, the world gather of the router values, the top-16 index), 2/ the latent
          down projection and its world gather, 3/ the routed experts, 4/ the latent norm, the latent up with the
          shared down, and the output all-reduce. With ``split``, the routed-expert weight loads of stage 3 need
          only the stage-1 index, so they run during the stage-2 gather. With ``fused``, stages 1 and 2 share one
          gather. With ``out_proj_form`` ``column`` (``OUT_PROJ_FORMS``), stage 4 is the column out-projection of
          all latent and shared inputs and the output all-gather.
        - Collective tiles: the sites of the base body take the tiles and HBM pairs of ``sources``
          (``decode_block.collective_sources``); the options add their own (``routed_collective_tiles``,
          ``routed_collective_hbm``). Without ``sources``, each call takes new tiles (a kernel with one MoE call).
        - Every per-token result depends only on the token, so a call on more tokens (a spec verify) gives the
          bits of the calls on fewer tokens.
    """
    kernel_assert(
        routed_kernel in ROUTED_KERNELS, f"routed_kernel must be one of {ROUTED_KERNELS}, got {routed_kernel}"
    )
    _validate_moe_inputs(
        x_hT,
        moe_in_proj,
        moe_out_proj,
        moe_vectors,
        expert_gate_up_weights,
        expert_gate_up_weights_scale,
        expert_down_weights,
        expert_down_weights_scale,
        rank_info,
        routed_kernel,
        in_proj_form,
        out_proj_form,
        in_proj_scale=in_proj_scale,
    )
    kernel_assert(
        routed_kernel == "static" or (weight_buffers == 3 and scale_buffers == 2),
        f"weight_buffers and scale_buffers apply to the static loop only, got {weight_buffers} and {scale_buffers}",
    )
    kernel_assert(
        routed_kernel == "static" or not half_weight_buffers, "half_weight_buffers applies to the static loop only"
    )
    kernel_assert(
        routed_kernel == "static" or not gate_up_half_major, "gate_up_half_major applies to the static loop only"
    )
    kernel_assert(
        routed_kernel == "static" or not shared_scale_buffer, "shared_scale_buffer applies to the static loop only"
    )
    kernel_assert(routed_kernel == "static" or early_loads, "early_loads=False applies to the static loop only")
    kernel_assert(
        routed_reduce_dtype in ROUTED_REDUCE_DTYPES, f"routed_reduce_dtype must be one of {ROUTED_REDUCE_DTYPES}"
    )
    kernel_assert(routing_debug is None or routed_kernel in TP128_KERNELS, "routing_debug needs a TP128 routed kernel")
    kernel_assert(expert_offset == 0 or routed_kernel == "sel_load", "expert_offset applies to sel_load only")
    kernel_assert(
        affinity_scaling in AFFINITY_SCALINGS,
        f"affinity_scaling must be one of {AFFINITY_SCALINGS}, got {affinity_scaling}",
    )
    kernel_assert(
        affinity_scaling == "pre_quant" or routed_kernel == "sel_load", "affinity_scaling applies to sel_load only"
    )
    kernel_assert(affinity_hi_lo or routed_kernel == "sel_load", "affinity_hi_lo applies to sel_load only")
    kernel_assert(routing_form in ROUTING_FORMS, f"routing_form must be one of {ROUTING_FORMS}, got {routing_form}")
    kernel_assert(routing_form == "base" or routed_kernel in TP128_KERNELS, "routing_form applies to the TP128 kernels")
    routed_reduce_fp32 = routed_reduce_dtype == "fp32"
    # The sel_load option routed_store_pe (TP128_ROUTED_STORE_OPTION): the routed partial fills the routed all-reduce
    # source through SBUF and TensorE transposes, in place of a DMA of 56-byte rows.
    sel_load_options = tuple(sel_load_options)
    routed_store_pe = dict(sel_load_options).get(TP128_ROUTED_STORE_OPTION, False)
    latent_pe = dict(sel_load_options).get(TP128_LATENT_PE_OPTION, False)
    sel_load_options = tuple(
        pair for pair in sel_load_options if pair[0] not in (TP128_ROUTED_STORE_OPTION, TP128_LATENT_PE_OPTION)
    )
    if router_mode is None:
        router_mode = "fused"
    kernel_assert(router_mode in ROUTER_MODES, f"router_mode must be one of {ROUTER_MODES}, got {router_mode}")
    kernel_assert(
        routing_form not in _PE32_FORMS or router_mode == "fused",
        f"routing_form {routing_form!r} needs the router form 'fused'",
    )
    kernel_assert(router_dtype in ROUTER_DTYPES, f"router_dtype must be one of {ROUTER_DTYPES}, got {router_dtype}")
    if router_dtype != "bf16":
        _validate_fp16_router(x_hT, x16_hT, router_weight, router_dtype)
    kernel_assert(in_proj_form in IN_PROJ_FORMS, f"in_proj_form must be one of {IN_PROJ_FORMS}, got {in_proj_form}")
    kernel_assert(
        "x_world_src_hbm" not in (sources or {}),
        "the world x gather (moe_x_world_gather) runs in the base MoE body and the dense body only",
    )
    dp_local = in_proj_form == "dp_local"
    if dp_local:
        _validate_dp_local(routed_kernel, router_mode, router_dtype, dp_size, sources, peer_in_proj)
    kernel_assert(
        routing_form not in _B1_ROUTING_FORMS or (dp_local and x_hT.shape[1] == 1),
        f"routing_form {routing_form!r} needs in_proj_form 'dp_local' at one token per DP replica",
    )
    kernel_assert(
        routing_form != "index_first_block" or routed_kernel == "sel_load",
        "routing_form 'index_first_block' needs the sel_load routed kernel",
    )
    kernel_assert(
        not routing_tiles
        or (dp_local and region is not None and routing_form in ("index_first", *_PE32_FORMS, *_B1_ROUTING_FORMS)),
        "routing_tiles needs in_proj_form 'dp_local', the explicit SBUF region and an index-first routing form",
    )
    kernel_assert(
        all(form in GLUE_DMA_FORMS for form in glue_dmas),
        f"glue_dmas {glue_dmas} must be names of {GLUE_DMA_FORMS}",
    )
    kernel_assert("split_gather" not in glue_dmas, "glue_dmas 'split_gather' runs in the base MoE body only")
    replica_rows = "replica_rows" in glue_dmas
    sp_weights = "sp_weights" in glue_dmas
    kernel_assert(
        not replica_rows or dp_size == DP_SIZE, f"glue_dmas 'replica_rows' needs the DP{DP_SIZE} layout, got {dp_size}"
    )
    weight_dma = _SP_HWDGE if sp_weights else None
    _validate_out_proj_form(out_proj_form, routed_kernel, routed_reduce_dtype, dp_size, sources)
    wide_routed = out_proj_form in WIDE_FORMS
    column = out_proj_form == "column"
    shared_gather = out_proj_form == "column_shared_ag"
    replica_tokens = x_hT.shape[1]
    tokens = dp_size * replica_tokens
    name = f"k3moe_{tag}"
    if region is not None:
        region.open_frame(name)

    # Issue the weight loads first; they depend on no compute.
    if dp_local:
        # The peer columns are on the critical path; the shared columns load after the routed experts.
        peer_sb = nl.ndarray((_PMAX, _H_TILES, PEER_IN_PROJ_COLS), dtype=nl.bfloat16, buffer=nl.sbuf)
        _load_hT_rows(peer_sb, peer_in_proj, _IN_PROJ_DMA_CHUNKS, f"{name}_peer_in_proj")
        peer_bias_sb = nl.ndarray((replica_tokens, DP_PEER_EXPERTS), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=peer_bias_sb,
            src=moe_vectors.ap(pattern=[[0, replica_tokens], [1, DP_PEER_EXPERTS]], offset=PEER_BIAS_COLS[0]),
            priority=_CRITICAL_DMA_PRIORITY,
            name=f"{name}_peer_bias",
        )
    else:
        if in_proj_scale is not None:
            in_proj_sb = _load_mx_in_proj(moe_in_proj, in_proj_scale, name)
        else:
            in_proj_sb = nl.ndarray((_PMAX, _H_TILES, MOE_IN_PROJ_COLS), dtype=nl.bfloat16, buffer=nl.sbuf)
            _load_hT_rows(in_proj_sb, moe_in_proj, _IN_PROJ_DMA_CHUNKS, f"{name}_in_proj")
        local_bias_sb = nl.ndarray((tokens, LOCAL_EXPERTS), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=local_bias_sb,
            src=moe_vectors.ap(pattern=[[0, tokens], [1, LOCAL_EXPERTS]], offset=LOCAL_BIAS_COLS[0]),
            priority=_CRITICAL_DMA_PRIORITY,
            name=f"{name}_local_bias",
        )

    router = None
    row_select, rank_select = rank_selectors(rank_info, f"{name}_rank", dp_size=dp_size)
    if router_dtype != "bf16":
        router_w_sb = nl.ndarray(tuple(router_weight.shape), dtype=nl.float16, buffer=nl.sbuf)
        nisa.dma_copy(dst=router_w_sb, src=router_weight, priority=_CRITICAL_DMA_PRIORITY, name=f"{name}_router_w16")
    if dp_local:
        pass
    elif router_dtype == "fp16":
        x_tokens_hT, x16_tokens_hT = _gather_routed_tokens(x_hT, dp_group, f"{name}_x", x16_hT, sources, dp_size)
        router = (x16_tokens_hT, router_w_sb)
    elif router_dtype == "fp16_local":
        x_tokens_hT, logits = _gather_tokens_and_local_logits(
            x_hT, x16_hT, router_w_sb, row_select, dp_group, f"{name}_x", sources
        )
        router = LocalRouterLogits(logits)
    else:
        x_tokens_hT = _gather_routed_tokens(x_hT, dp_group, f"{name}_x", None, sources, dp_size)

    out_proj_sb_tiles = AUTO_SBUF if region is None else SbufTiles(region)
    shared_input = []
    # sp_weights: the last rank-row load of the world gather, which the shared in-projection loads follow.
    rank_rows_done = []
    # index_first_rows: the router-word load of the world gather, which the DP x gather waits for.
    route_done = []
    if dp_local:
        kernel_assert(not latent_pe, f"sel_load option {TP128_LATENT_PE_OPTION} does not apply to in_proj_form dp_local")
        routing, latent_words = _dp_local_router_and_latent(
            x_hT,
            peer_sb,
            peer_bias_sb,
            world_group,
            name,
            routed_kernel,
            sources,
            routing_form,
            **({"rank_rows_done": rank_rows_done} if sp_weights else {}),
            **({"route_done": route_done} if routing_form in _B1_ROUTING_FORMS else {}),
            **({"routing_tiles": SbufTiles(region)} if routing_tiles else {}),
        )
        # The source store of the DP x gather; the gather follows the routed kernel (_dp_local_x_gather).
        x_src_hbm, x_gather_hbm = _dp_local_x_store(x_hT, name, sources)
    else:
        stage = _fused_router_and_latent if router_mode == "fused" else _split_router_and_latent
        kernel_assert(
            not latent_pe or router_mode == "fused", f"sel_load option {TP128_LATENT_PE_OPTION} needs router_mode fused"
        )
        routing, latent_words, out_proj_input = stage(
            x_tokens_hT,
            in_proj_sb,
            local_bias_sb,
            world_group,
            name,
            out_proj_sb_tiles,
            routed_kernel,
            router,
            sources,
            routing_form,
            **({"shared_rows": True} if shared_gather else {}),
            **({"latent_pe": True} if latent_pe else {}),
        )

    def shared_glue() -> dict:
        """The queue and the schedule anchor of the shared in-projection loads (``GLUE_DMA_FORMS`` ``sp_weights``)."""
        if not sp_weights:
            return {}
        return {"weight_dma": weight_dma, "after": rank_rows_done[0] if rank_rows_done else None}

    shared_stream = {}
    shared_tiles = None
    if shared_gather:
        # column_shared_ag: the shared activation [T, 48], packed into the source of the shared gather. With dp_local
        # the DP x gather comes first, and the weight stream of the routed kernel is scheduled after it
        # (``stream_after``), so no weight gather lies between the collectives without depending on them. The shared
        # in-projection and activation then run inside the routed kernel's start (``shared_before_stream``). Their
        # tiles take region addresses here, below the routed kernel's buffers: placed by the compiler, a tile that the
        # schedule frees before the first weight gather could share its destination, and the gather would wait for
        # the shared expert.
        if dp_local:
            x_gather = _dp_local_x_gather(x_src_hbm, x_gather_hbm, dp_group, name, route_done)
            shared_stream["stream_after"] = x_gather
            shared_tiles = dp_local_shared_rows_tiles(replica_tokens, out_proj_sb_tiles)
            shared_src = _source(sources, "shared_ag_src", (1, tokens * SHARED_GATHER_COLS), nl.bfloat16)
        else:
            shared_src = _shared_gather_source(out_proj_input, sources, name)
        shared_dst = _source(sources, "shared_ag_dst", (WORLD_SIZE, shared_src.shape[1]), nl.bfloat16)

        def shared_gather_between_chunks(after):
            """The shared gather on the Pool queue after the weight gathers of the first expert chunk (``after``)
            and before those of the second."""
            gather = _shared_all_gather(shared_src, shared_dst, world_group, name)
            _instruction(gather).depends_on(_instruction(after), kind=nisa.dependency.schedule)
            return gather

    def shared_after_routed_trigger():
        """dp_local: the shared-expert input after the routed all-reduce trigger, so that its loads and matmuls run in
        the flight of the all-reduce and not ahead of the routed partial store. ``column`` sends it in the routed
        all-reduce, so it runs before; ``column_shared_ag`` in its own gather, after the first gather index of the
        routed kernel."""
        if dp_local and not (column or shared_gather):
            shared_input.append(
                dp_local_shared_activation(
                    x_gather_hbm, moe_in_proj, replica_tokens, name, out_proj_sb_tiles, **shared_glue()
                )
            )

    if expert_down_weights.dtype == nl.uint16:
        expert_down_weights = expert_down_weights.view(nl.float4_e2m1fn_x4)
    # A program with pair_shared_cc_dsts shares every shared-HBM scratch tensor between the two cores of an HBM stack:
    # the TP128 kernels then write this rank's routed partial to private HBM.
    private_partial = {"private_out": True} if pair_shared_program(sources) else {}
    if routed_kernel == "sel_load":
        expert_index, affinities, index_done = routing
        if routing_debug is not None:
            _store_routing(expert_index, affinities, routing_debug, f"{name}_routing")
        # Stage 3 starts on the routing: the weight loads need only the index, so they run during the latent gather.
        # With an index-first form, the latent copy follows the gather indices of the first weight gathers.
        latents = []

        def copy_latent(after):
            latents.append(_gathered_latent(latent_words, tokens, None if after is None else _instruction(after)))

        def shared_before_stream(after):
            """column_shared_ag with dp_local: the shared in-projection and activation after the first gather index of
            the routed kernel (``after``) in the schedule, and their pack into the source of the shared gather. Return
            the PSUM copy of the activation, which the weight stream follows in the schedule (no hardware wait)."""
            copies = []
            rows = dp_local_shared_rows(
                x_gather_hbm,
                moe_in_proj,
                replica_tokens,
                name,
                tiles=shared_tiles,
                compute_after=after,
                copies=copies,
                **shared_glue(),
            )
            _shared_gather_source(rows, sources, name)
            return copies[0]

        # Fail closed above the one-pass expert pick (PICK_GATHER_MAX_TOKENS): the sel_load start's nc_n_gather would run
        # several passes there.
        kernel_assert(
            expert_index.shape[0] <= PICK_GATHER_MAX_TOKENS,
            f"sel_load takes at most {PICK_GATHER_MAX_TOKENS} tokens per MoE call, got {expert_index.shape[0]}",
        )
        # Two down scale groups: the checkpoint down values, both group passes in fp32 PSUM (one rounding).
        groups = {}
        if sel_load_down_groups(expert_gate_up_weights, expert_down_weights_scale):
            groups = {"down_group_mask": down_group_mask(rank_info, f"{name}_down_groups"), "down_psum_dtype": "fp32"}
        options = dict(sel_load_options)
        kimi_t4 = options.pop("kimi_t4_body", False)
        kernel_assert(isinstance(kimi_t4, bool), f"sel_load option kimi_t4_body must be a bool, got {kimi_t4!r}")
        # Fail closed off the T = 32 / 64 forms: the kimi_t32 body runs its block groups per 32-token chunk, so a
        # call of another token count has no valid chunking (the target is the 1M-context batch-4 k7 verify, T = 32).
        kimi_t32 = options.pop("kimi_t32_body", False)
        kimi_t32_form = options.pop("kimi_t32_form", "base")
        kernel_assert(
            kimi_t32 or kimi_t32_form == "base", f"sel_load option kimi_t32_form {kimi_t32_form!r} needs kimi_t32_body"
        )
        kimi_t32_form_options(kimi_t32_form)
        if kimi_t32:
            kernel_assert(
                tokens in KIMI_T32_CALL_TOKENS,
                f"sel_load option kimi_t32_body needs one of {KIMI_T32_CALL_TOKENS} tokens per MoE call "
                f"(16 pads to one 32-token call), got {tokens}",
            )
            # The body reads the padded down rows [E, 8, H] or the compact rows [E, 6, H] (compact_down) with the same
            # outputs: the gathers skip the padding rows in the two forms.
            kernel_assert(
                not groups
                and affinity_scaling == "pre_quant"
                and expert_down_weights.shape[-2] in (_PADDED_DOWN_ROWS, SEL_LOAD_COMPACT_DOWN_ROWS),
                "sel_load option kimi_t32_body needs one down scale set, pre_quant and the down rows [E, 8, H] or "
                f"[E, {SEL_LOAD_COMPACT_DOWN_ROWS}, H]",
            )
            # The options of the 8-token calls do not apply to the 32-token body; any other option is refused.
            other = sorted(
                set(options) - {"kimi_body", "kimi_hwdge_calls", "chunk_tokens", "long_stream", "sbuf_reserve"}
            )
            kernel_assert(
                not other and not shared_gather,
                f"sel_load option kimi_t32_body runs without {other or 'column_shared_ag'}",
            )
            latent = latents[0] if latents else _gathered_latent(latent_words, tokens, index_done)
            expert_partial = _kimi_t32_partial(
                routing,
                latent,
                expert_gate_up_weights,
                expert_down_weights,
                expert_down_weights_scale,
                expert_offset,
                region,
                name,
                hot_experts,
                kimi_t32_form,
            )
        else:
            kernel_assert(hot_experts is None, "hot_experts needs the kimi_t32 sel_load body")
            start, finish = sel_load_start, sel_load_finish
            kimi_body, hwdge_calls = _pop_kimi_body_options(options)
            kimi_t4 = kimi_t4 and tokens == kimi_tp.SHORT_TOKENS
            if kimi_t4 or (kimi_body and tokens == kimi_tp.TOKENS):
                key = "kimi_t4_body" if kimi_t4 else "kimi_body"
                kernel_assert(
                    not groups
                    and affinity_scaling == "pre_quant"
                    and expert_down_weights.shape[-2] == _PADDED_DOWN_ROWS,
                    f"sel_load option {key} needs one down scale set, pre_quant and the padded down rows [E, 8, H]",
                )
                # The 4-token form sets its own options: the 8-token call options do not apply, others are refused.
                other = sorted(set(options) - {"chunk_tokens", "long_stream", "sbuf_reserve"}) if kimi_t4 else []
                kernel_assert(
                    not other and not (kimi_t4 and hwdge_calls),
                    f"sel_load option kimi_t4_body runs without {other or 'kimi_hwdge_calls'}",
                )
                # The kimi body builds its own gather indices, so the served start's index options do not apply.
                index_options = ("index_matmul", "index_broadcast_first", "index_onehot_resident")
                kernel_assert(
                    not any(options.get(key) for key in index_options),
                    f"sel_load option kimi_body runs without {', '.join(index_options)}",
                )
                # The kimi start and finish take no shared-expert hooks (stream_after, before_stream, between_chunks).
                kernel_assert(
                    not shared_gather, f"sel_load option {key} runs without the column_shared_ag out-projection"
                )
                # The kimi body keeps its call count and Pool stream chain in module state: restart it in each trace.
                call = next_call("kimi_body_trace")
                if call == 0:
                    kimi_tp.restart_calls()
                options = dict(_KIMI_T4_OPTIONS) if kimi_t4 else _kimi_body_call_options(options, call, hwdge_calls)
                start, finish = kimi_tp.selective_load_tp_start, kimi_tp.selective_load_tp_finish
            handle = start(
                expert_index,
                affinities,
                expert_gate_up_weights,
                expert_down_weights,
                expert_down_weights_scale,
                SITU_BETA,
                SITU_LINEAR_BETA,
                sbuf=region,
                **private_partial,
                **groups,
                **({"expert_offset": expert_offset} if expert_offset else {}),
                **({"affinity_scaling": affinity_scaling} if affinity_scaling != "pre_quant" else {}),
                **({} if affinity_hi_lo else {"affinity_hi_lo": False}),
                **({} if index_done is None else {"after_index": copy_latent}),
                **options,
                **shared_stream,
                # column_shared_ag with dp_local: the shared activation between the gather indices and the weight stream.
                **({} if shared_tiles is None else {"before_stream": shared_before_stream}),
            )
            latent = latents[0] if latents else _gathered_latent(latent_words, tokens, index_done)
            expert_partial = finish(
                handle, latent, **({"between_chunks": shared_gather_between_chunks} if shared_gather else {})
            )
    elif routed_kernel == "tp_dense":
        expert_index, affinities, _ = routing
        if routing_debug is not None:
            _store_routing(expert_index, affinities, routing_debug, f"{name}_routing")
        latent = _gathered_latent(latent_words, tokens) if latent_words.dtype == nl.float32 else latent_words
        expert_partial = tp_dense_routed(
            latent,
            expert_index,
            affinities,
            expert_gate_up_weights,
            expert_down_weights,
            expert_down_weights_scale,
            SITU_BETA,
            SITU_LINEAR_BETA,
            sbuf=region,
            **private_partial,
        )
    else:
        latent = _gathered_latent(latent_words, tokens) if latent_words.dtype == nl.float32 else latent_words
        if expert_gate_up_weights.dtype == nl.uint16:
            expert_gate_up_weights = expert_gate_up_weights.view(nl.float4_e2m1fn_x4)
        # The nested kernel call takes positional arguments only.
        expert_partial = moe_tkg_decode(
            latent,
            routing,
            expert_gate_up_weights,
            expert_down_weights,
            expert_gate_up_weights_scale,
            expert_down_weights_scale,
            SITU_BETA,
            SITU_LINEAR_BETA,
            weight_buffers,
            scale_buffers,
            early_loads,
            # Only when asked for: the default call does not change.
            **({"half_weight_buffers": half_weight_buffers} if half_weight_buffers else {}),
            **({"gate_up_half_major": True} if gate_up_half_major else {}),
            **({"shared_scale_buffer": True} if shared_scale_buffer else {}),
        )
    if dp_local and not shared_gather:
        # In program order after every weight DMA of the routed kernel; on the collective stream between the world
        # gather and the routed all-reduce.
        _dp_local_x_gather(x_src_hbm, x_gather_hbm, dp_group, name, route_done)

    # All-reduce of the routed partial in [128, T * 28] layout, as in the base body: partition g holds latent
    # columns [28 g, 28 g + 28) of all tokens. The all-reduce runs in HBM, or in SBUF with the persistent source
    # routed_src_sb (moe_sbuf_reduces routed_tp128, the MLA loop path, bf16 only).
    partial_pattern = [[LATENT_SHARD, WORLD_SIZE], [LATENT_SIZE, tokens], [1, LATENT_SHARD]]
    kernel_assert(
        not routed_store_pe or not (wide_routed or routed_reduce_fp32),
        f"sel_load option {TP128_ROUTED_STORE_OPTION} needs the bf16 routed all-reduce of the [128, T, 28] source",
    )
    if routed_store_pe:
        routed_groups = _routed_partial_column_groups(expert_partial, tokens, out_proj_sb_tiles, name)
    if wide_routed:
        if column and dp_local:
            out_proj_input = dp_local_shared_activation(
                x_gather_hbm, moe_in_proj, replica_tokens, name, out_proj_sb_tiles
            )
        routed_wide = _wide_routed_reduce(
            expert_partial.ap(pattern=partial_pattern, offset=0),
            out_proj_input if column else None,
            rank_info,
            tokens,
            world_group,
            name,
            sources,
        )
        shared_after_routed_trigger()
        routed = routed_wide[0:WORLD_SIZE, 0:tokens, 0:LATENT_SHARD]
    elif routed_reduce_fp32:
        routed_rows = nl.ndarray((WORLD_SIZE, tokens, LATENT_SHARD), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=routed_rows,
            src=expert_partial.ap(pattern=partial_pattern, offset=0),
            priority=_CRITICAL_DMA_PRIORITY,
            name=f"{name}_routed_load",
        )
        reduce_src = nl.ndarray((WORLD_SIZE, tokens * LATENT_SHARD), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=reduce_src, src=routed_rows.reshape((WORLD_SIZE, tokens * LATENT_SHARD)))
        reduce_dst = _all_reduce_hbm(reduce_src, world_group, f"{name}_routed", sources, "routed32")
        shared_after_routed_trigger()
        routed = nl.ndarray((WORLD_SIZE, tokens * LATENT_SHARD), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=routed, src=reduce_dst)
    elif sources is not None and "routed_src_sb" in sources:
        # The SBUF routed all-reduce (``collective_sources`` ``sbuf_routed_options``, the moe_sbuf_reduces name
        # routed_tp128), as in the base body: the partial goes to the persistent SBUF source, and the all-reduce writes
        # the persistent routed_dst view of the x gather tile. It drops the HBM -> SBUF load of the result.
        routed_src = _source(sources, "routed_src_sb", (WORLD_SIZE, tokens, LATENT_SHARD), nl.bfloat16)
        if routed_store_pe:
            nisa.tensor_copy(dst=routed_src, src=routed_groups, engine=nisa.engine.vector)
        else:
            nisa.dma_copy(
                dst=routed_src,
                src=expert_partial.ap(pattern=partial_pattern, offset=0),
                priority=_CRITICAL_DMA_PRIORITY,
                name=f"{name}_routed_store",
            )
        routed = _source(sources, "routed_dst", (WORLD_SIZE, tokens * LATENT_SHARD), nl.bfloat16)
        release_weight_gate(
            ncc.all_reduce(
                srcs=[routed_src.reshape((WORLD_SIZE, tokens * LATENT_SHARD))],
                dsts=[routed],
                replica_group=world_group,
                op=nl.add,
                name=f"{name}_routed_reduce",
            )
        )
        shared_after_routed_trigger()
    else:
        routed_src, routed_hbm, rr = _routed_reduce_hbm(sources, tokens)
        routed_store = routed_src if rr == tokens else routed_src[0:WORLD_SIZE, 0:tokens, 0:LATENT_SHARD]
        routed_load = routed_hbm if rr == tokens else routed_hbm[0:WORLD_SIZE, 0 : tokens * LATENT_SHARD]
        nisa.dma_copy(
            dst=routed_store,
            src=routed_groups if routed_store_pe else expert_partial.ap(pattern=partial_pattern, offset=0),
            priority=_CRITICAL_DMA_PRIORITY,
            name=f"{name}_routed_store",
        )
        check_pair_group(routed_hbm, world_group, f"{name}_routed_reduce")
        release_weight_gate(
            ncc.all_reduce(
                srcs=[routed_src.reshape((WORLD_SIZE, rr * LATENT_SHARD))],
                dsts=[routed_hbm],
                replica_group=world_group,
                op=nl.add,
                name=f"{name}_routed_reduce",
            )
        )
        shared_after_routed_trigger()
        routed = _tile(sources, "routed_dst", (WORLD_SIZE, tokens * LATENT_SHARD), nl.bfloat16)
        nisa.dma_copy(dst=routed, src=routed_load, priority=_CRITICAL_DMA_PRIORITY, name=f"{name}_routed_load")
    if column or shared_gather:
        norm_offset = _column_norm_offset(in_proj_form)
        if shared_gather:
            out_rows = _column_out_proj(
                routed.reshape((WORLD_SIZE, tokens, LATENT_SHARD)),
                moe_out_proj,
                moe_vectors,
                norm_offset,
                name,
                shared=shared_dst.reshape((WORLD_SIZE, tokens, SHARED_GATHER_COLS)),
            )
        else:
            out_rows = _column_out_proj(routed_wide, moe_out_proj, moe_vectors, norm_offset, name)
        out_hT = _all_gather_select_rows(out_rows, row_select, replica_tokens, world_group, f"{name}_out", sources)
        if region is not None:
            region.close_frame()
        return out_hT
    if dp_local:
        out_proj_input = shared_input[0]
    out_proj_sb = nl.ndarray((MOE_OUT_PROJ_ROWS, HIDDEN_SIZE), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=out_proj_sb,
        src=moe_out_proj,
        priority=_CRITICAL_DMA_PRIORITY,
        name=f"{name}_out_proj",
    )
    _latent_norm(
        routed,
        moe_vectors,
        rank_select,
        out_proj_input[LATENT_UP_ROWS[0] : LATENT_UP_ROWS[1], :],
        f"{name}_norm",
        **({"weight_dma": weight_dma} if sp_weights else {}),
    )

    # Latent up and shared down in one K = 76 matmul for each hidden tile.
    partial_hT = _project_to_hT(
        lambda h_tile_idx: [(out_proj_sb[:, h_tile_idx * _PMAX : (h_tile_idx + 1) * _PMAX], out_proj_input)],
        tokens,
        _tile(sources, "out", (_PMAX, tokens, _H_TILES), nl.bfloat16),
    )
    out_hT = _all_reduce_select_rows(
        partial_hT,
        row_select,
        replica_tokens,
        world_group,
        f"{name}_out",
        sources,
        dp_size,
        **({"replica_index": replica_index(rank_info)} if replica_rows else {}),
    )
    if region is not None:
        region.close_frame()
    return out_hT


def dense_mlp_decode(
    x_hT: nl.NkiTensor,
    dense_gate_up: nl.NkiTensor,
    dense_down: nl.NkiTensor,
    rank_info: nl.NkiTensor,
    dp_group: ReplicaGroup,
    world_group: ReplicaGroup,
    tag: str,
    sources: dict | None = None,
    dp_size: int = DP_SIZE,
) -> nl.NkiTensor:
    """Run the Kimi-K3 dense SiTU MLP (layer 0) for the tokens of the DP replicas and keep this replica's rows.

    Every rank gathers the tokens of the 4 replicas and computes its TP128 shard (264 of the
    33792 intermediate columns) for all tokens. The world all-reduce gives the full output. Not
    ``@nki.jit``: the decode giga-kernel inlines the call. The attribute ``token_major`` is True.

    Dimensions:
        H: Hidden size, 7168.
        B: Tokens of each DP replica, 1 to ``MAX_REPLICA_TOKENS`` (32): up to 32 in decode, and up to
            ffn_rows = 32 in a verify feed-forward call.
        T: Gathered tokens, 4 * B <= 128. Token r * B + b is token b of replica r.
        I: Intermediate shard, 264. Rank g owns intermediate columns [264 g, 264 g + 264).

    Args:
        x_hT (nl.NkiTensor): [128, B, 56] BF16 SBUF, this replica's normalized hidden in the
            token-major hT layout.
        dense_gate_up (nl.NkiTensor): [128, 56, 528] BF16 HBM, [in, out] hT-tiled (element
            (p, j, n) is input row j * 128 + p): the gate shard (264 columns), then the up shard
            (264 columns).
        dense_down (nl.NkiTensor): [264, H] BF16 HBM, [in, out]: the down shard.
        rank_info (nl.NkiTensor): [1, 4] INT32 HBM, (global rank, DP rank, TP rank, 0).
        dp_group (ReplicaGroup): [[t, t + 32, t + 64, t + 96] for t in 0..31].
        world_group (ReplicaGroup): [[0 .. 127]].
        tag (str): Unique per-layer string. It names the collectives and the DMAs.
        sources (dict): None, or the persistent SBUF sources of the collectives
            (``decode_block.collective_sources``). The values do not change.

    Returns:
        out_hT (nl.NkiTensor): [128, B, 56] BF16 SBUF, this replica's MLP output in the
            token-major hT layout.

    Notes:
        - The all-reduce sums the shards of all tokens. Each replica keeps its own rows.
        - Weight DMA gates (``tkg_weight_dma_gate``): the output all-reduce releases the gate that holds the
          weight DMAs of the next body.
        - The down PSUM [128, 56, T] FP32 runs in groups of hidden tiles, one bank for each group.

    Pseudocode:
        X = all_gather_dp(x)                                  # [T, H]
        gate, up = bf16(X @ dense_gate_up)                    # [T, 264] each
        y = bf16(situ(gate, up) @ dense_down)                 # [T, H]
        out = all_reduce_world(y)[rows of this replica]
    """
    _validate_dense_inputs(x_hT, dense_gate_up, dense_down, rank_info)
    replica_tokens, tokens = _moe_token_counts(x_hT, sources, dp_size)
    name = f"k3mlp_{tag}"

    gate_up_sb = nl.ndarray((_PMAX, _H_TILES, 2 * DENSE_SHARD), dtype=nl.bfloat16, buffer=nl.sbuf)
    _load_hT_rows(gate_up_sb, dense_gate_up, _DENSE_GATE_UP_DMA_CHUNKS, f"{name}_gate_up")
    # Down rows in three partition tiles: rows 0-127, 128-255, and 256-263.
    down_tiles = (DENSE_SHARD + _PMAX - 1) // _PMAX
    full_down_tiles = DENSE_SHARD // _PMAX
    down_tail_rows = DENSE_SHARD - full_down_tiles * _PMAX
    down_sb = nl.ndarray((_PMAX, down_tiles, HIDDEN_SIZE), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=down_sb[:, 0:full_down_tiles, :],
        src=dense_down.ap(
            pattern=[[HIDDEN_SIZE, _PMAX], [_PMAX * HIDDEN_SIZE, full_down_tiles], [1, HIDDEN_SIZE]],
            offset=0,
        ),
        name=f"{name}_down",
    )
    nisa.dma_copy(
        dst=down_sb[0:down_tail_rows, full_down_tiles, :],
        src=dense_down[full_down_tiles * _PMAX : DENSE_SHARD, :],
        name=f"{name}_down_tail",
    )

    x_tokens_hT = _gather_replica_tokens(x_hT, dp_group, f"{name}_x", sources, dp_size, world_group)
    row_select, _ = rank_selectors(rank_info, f"{name}_rank", dp_size=dp_size)

    # Gate and up in separate PSUM banks: the 528 FP32 columns of both do not fit in one bank.
    gate_up_bf16 = nl.ndarray((tokens, 2 * DENSE_SHARD), dtype=nl.bfloat16, buffer=nl.sbuf)
    for projection_idx in range(2):
        columns = slice(projection_idx * DENSE_SHARD, (projection_idx + 1) * DENSE_SHARD)
        projection_psum = _psum_bank(tokens, (DENSE_SHARD,), nl.float32)
        for h_tile_idx in range(_H_TILES):
            nisa.nc_matmul(
                dst=projection_psum,
                stationary=x_tokens_hT[:, h_tile_idx, :],
                moving=gate_up_sb[:, h_tile_idx, columns],
            )
        nisa.tensor_copy(dst=gate_up_bf16[:, columns], src=projection_psum)
    activation = _situ(gate_up_bf16[:, 0:DENSE_SHARD], gate_up_bf16[:, DENSE_SHARD : 2 * DENSE_SHARD])

    # Intermediate on partitions: [264, T] in three tiles, the moving operand of the down projection. Each tile
    # takes an even PSUM width, so a one-token tile starts at a 4-byte PSUM offset.
    psum_tokens = tokens + tokens % 2
    activation_T_psum = _psum_bank(_PMAX, (down_tiles, psum_tokens), nl.bfloat16)
    for i_tile_idx in range(down_tiles):
        i_rows = min(_PMAX, DENSE_SHARD - i_tile_idx * _PMAX)
        nisa.nc_transpose(
            dst=activation_T_psum[0:i_rows, i_tile_idx, 0:tokens],
            data=activation[:, i_tile_idx * _PMAX : i_tile_idx * _PMAX + i_rows],
        )
    activation_T = nl.ndarray((_PMAX, down_tiles, tokens), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=activation_T[:, 0:full_down_tiles, :], src=activation_T_psum[:, 0:full_down_tiles, 0:tokens])
    nisa.tensor_copy(
        dst=activation_T[0:down_tail_rows, full_down_tiles, :],
        src=activation_T_psum[0:down_tail_rows, full_down_tiles, 0:tokens],
    )

    # Down projection in hT layout: out tile j accumulates the three intermediate tiles.
    def down_terms(h_tile_idx: int) -> list[tuple[nl.NkiTensor, nl.NkiTensor]]:
        terms = []
        for i_tile_idx in range(down_tiles):
            i_rows = min(_PMAX, DENSE_SHARD - i_tile_idx * _PMAX)
            terms.append(
                (
                    down_sb[0:i_rows, i_tile_idx, h_tile_idx * _PMAX : (h_tile_idx + 1) * _PMAX],
                    activation_T[0:i_rows, i_tile_idx, :],
                )
            )
        return terms

    partial_hT = _project_to_hT(down_terms, tokens, _tile(sources, "out", (_PMAX, tokens, _H_TILES), nl.bfloat16))
    return _all_reduce_select_rows(partial_hT, row_select, replica_tokens, world_group, f"{name}_out", sources, dp_size)


dense_mlp_decode.token_major = True


def _validate_moe_inputs(
    x_hT: nl.NkiTensor,
    moe_in_proj: nl.NkiTensor,
    moe_out_proj: nl.NkiTensor,
    moe_vectors: nl.NkiTensor,
    expert_gate_up_weights: nl.NkiTensor,
    expert_gate_up_weights_scale: nl.NkiTensor,
    expert_down_weights: nl.NkiTensor,
    expert_down_weights_scale: nl.NkiTensor,
    rank_info: nl.NkiTensor,
    routed_kernel: str = "static",
    in_proj_form: str = "gathered",
    out_proj_form: str = "reduced",
    in_proj_scale: nl.NkiTensor = None,
    drop_selection: bool = False,
    latent_gather_mx: bool = False,
):
    """Check the shapes and dtypes of the latent MoE body inputs for the routed kernel and the in- and out-projection
    forms (``in_proj_scale``: the MXFP8 in-projection of ``MX_IN_PROJ_SHAPES``). ``drop_selection``
    (``router_rows_drop_selection``): ``moe_vectors`` also holds the router bias of all experts
    (``ALL_ROUTER_BIAS_COLS``). ``latent_gather_mx``: ``moe_in_proj`` has the 135 columns of ``MX_IN_PROJ_COLS``.
    Without them the body fails."""
    _validate_common_inputs(x_hT, rank_info)
    dp_local = in_proj_form == "dp_local"
    in_cols = SHARED_IN_PROJ_COLS if dp_local else MOE_IN_PROJ_COLS
    if latent_gather_mx:
        in_cols = MX_IN_PROJ_COLS
    vector_cols = _column_norm_offset(in_proj_form)
    if out_proj_form in COLUMN_FORMS:
        vector_cols += LATENT_SIZE
    if drop_selection:
        vector_cols += NUM_EXPERTS
    if in_proj_scale is None:
        kernel_assert(
            tuple(moe_in_proj.shape) == (_PMAX, _H_TILES, in_cols) and moe_in_proj.dtype == nl.bfloat16,
            f"moe_in_proj must be BF16 [{_PMAX}, {_H_TILES}, {in_cols}], got {moe_in_proj.dtype} {moe_in_proj.shape}"
            + (" (latent_gather_mx: the 32-column latent blocks, MX_IN_PROJ_COLS)" if latent_gather_mx else ""),
        )
    else:
        data_shape, scale_shape = MX_IN_PROJ_SHAPES
        kernel_assert(
            not dp_local
            and tuple(moe_in_proj.shape) == data_shape
            and moe_in_proj.dtype == nl.uint8
            and tuple(in_proj_scale.shape) == scale_shape
            and in_proj_scale.dtype == nl.uint8,
            f"the MXFP8 moe_in_proj / in_proj_scale must be uint8 {data_shape} / {scale_shape} (gathered), got "
            f"{moe_in_proj.dtype} {moe_in_proj.shape} / {in_proj_scale.dtype} {in_proj_scale.shape}",
        )
    kernel_assert(
        tuple(moe_out_proj.shape) == (MOE_OUT_PROJ_ROWS, HIDDEN_SIZE) and moe_out_proj.dtype == nl.bfloat16,
        f"moe_out_proj must be BF16 [{MOE_OUT_PROJ_ROWS}, {HIDDEN_SIZE}], got {moe_out_proj.dtype} {moe_out_proj.shape}",
    )
    kernel_assert(
        tuple(moe_vectors.shape) == (1, vector_cols) and moe_vectors.dtype == nl.float32,
        f"moe_vectors must be FP32 [1, {vector_cols}], got {moe_vectors.dtype} {moe_vectors.shape}"
        + (
            f": router_rows_drop_selection needs the router bias of all {NUM_EXPERTS} experts in columns "
            f"{ALL_ROUTER_BIAS_COLS}"
            if drop_selection
            else ""
        ),
    )
    kernel_assert(
        expert_down_weights.dtype == nl.uint16 or expert_down_weights.dtype == nl.float4_e2m1fn_x4,
        f"expert_down_weights must be MXFP4 or UINT16, got {expert_down_weights.dtype}",
    )
    if routed_kernel in TP128_KERNELS:
        _validate_sel_load_experts(
            expert_gate_up_weights, expert_down_weights, expert_down_weights_scale, routed_kernel == "sel_load"
        )
        return
    gate_up_shape = (LOCAL_EXPERTS, _PMAX, 2, LATENT_SIZE // 512, EXPERT_INTERMEDIATE)
    kernel_assert(
        tuple(expert_gate_up_weights.shape) == gate_up_shape,
        f"expert_gate_up_weights must be {gate_up_shape}, got {expert_gate_up_weights.shape}",
    )
    kernel_assert(
        expert_gate_up_weights.dtype == nl.uint16 or expert_gate_up_weights.dtype == nl.float4_e2m1fn_x4,
        f"expert_gate_up_weights must be MXFP4 or UINT16, got {expert_gate_up_weights.dtype}",
    )
    kernel_assert(
        expert_gate_up_weights_scale is not None
        and tuple(expert_gate_up_weights_scale.shape)
        == (LOCAL_EXPERTS, 16, 2, LATENT_SIZE // 512, EXPERT_INTERMEDIATE),
        f"Unexpected expert_gate_up_weights_scale {None if expert_gate_up_weights_scale is None else expert_gate_up_weights_scale.shape}",
    )
    down_shape = (LOCAL_EXPERTS, _PMAX, EXPERT_INTERMEDIATE // 512, LATENT_SIZE)
    kernel_assert(
        tuple(expert_down_weights.shape) == down_shape,
        f"expert_down_weights must be {down_shape}, got {expert_down_weights.shape}",
    )
    kernel_assert(
        tuple(expert_down_weights_scale.shape) == (LOCAL_EXPERTS, 16, EXPERT_INTERMEDIATE // 512, LATENT_SIZE),
        f"Unexpected expert_down_weights_scale shape {expert_down_weights_scale.shape}",
    )


def _validate_dense_inputs(
    x_hT: nl.NkiTensor,
    dense_gate_up: nl.NkiTensor,
    dense_down: nl.NkiTensor,
    rank_info: nl.NkiTensor,
):
    """Check the shapes and dtypes of the dense MLP body inputs."""
    _validate_common_inputs(x_hT, rank_info, trn3_only=False)
    kernel_assert(
        tuple(dense_gate_up.shape) == (_PMAX, _H_TILES, 2 * DENSE_SHARD) and dense_gate_up.dtype == nl.bfloat16,
        f"dense_gate_up must be BF16 [{_PMAX}, {_H_TILES}, {2 * DENSE_SHARD}], "
        f"got {dense_gate_up.dtype} {dense_gate_up.shape}",
    )
    kernel_assert(
        tuple(dense_down.shape) == (DENSE_SHARD, HIDDEN_SIZE) and dense_down.dtype == nl.bfloat16,
        f"dense_down must be BF16 [{DENSE_SHARD}, {HIDDEN_SIZE}], got {dense_down.dtype} {dense_down.shape}",
    )


def _validate_common_inputs(x_hT: nl.NkiTensor, rank_info: nl.NkiTensor, trn3_only: bool = True):
    """Check the hT input and the rank vector, and the LNC1 target (Trn3 unless ``trn3_only`` is False)."""
    kernel_assert(
        nl.program_ndim() == 0 or (nl.program_ndim() == 1 and nl.num_programs(0) == 1),
        "The Kimi-K3 MoE decode bodies support LNC1 only",
    )
    kernel_assert(not trn3_only or is_trn3(), "The Kimi-K3 MoE decode bodies require Trn3")
    kernel_assert(
        len(x_hT.shape) == 3 and x_hT.shape[0] == _PMAX and x_hT.shape[2] == _H_TILES and x_hT.dtype == nl.bfloat16,
        f"x_hT must be BF16 [{_PMAX}, B, {_H_TILES}], got {x_hT.dtype} {x_hT.shape}",
    )
    kernel_assert(
        1 <= x_hT.shape[1] <= MAX_REPLICA_TOKENS,
        f"x_hT has B={x_hT.shape[1]} tokens per replica. Use 1 <= B <= {MAX_REPLICA_TOKENS}.",
    )
    kernel_assert(
        tuple(rank_info.shape) == (1, 4) and rank_info.dtype == nl.int32,
        f"rank_info must be INT32 [1, 4], got {rank_info.dtype} {rank_info.shape}",
    )


def _psum_bank(partitions: int, shape: tuple[int, ...], dtype) -> nl.NkiTensor:
    """Allocate one whole PSUM bank and return its [partitions, *shape] view at offset 0."""
    bank_elems = _PSUM_BANK_BYTES // _DTYPE_BYTES[dtype]
    n_elems = 1
    for size in shape:
        n_elems *= size
    kernel_assert(
        n_elems <= bank_elems,
        f"PSUM tile {shape} has {n_elems} elements per partition; one bank holds {bank_elems}",
    )
    bank = nl.ndarray((partitions, bank_elems), dtype=dtype, buffer=nl.psum)
    view = bank[:, 0:n_elems]
    # A 1-D shape is already [partitions, n_elems]; reshaping it collapses the partition step to 0 at
    # partitions == 1 (DP1 S=1, T=1). Skip the no-op reshape; for partitions > 1 the view is identical.
    if shape == (n_elems,):
        return view
    return view.reshape((partitions, *shape))


def _load_hT_rows(dst: nl.NkiTensor, weight: nl.NkiTensor, n_chunks: int, name: str, after=None, dma: dict = None):
    """Load an hT-tiled [128, 56, N] weight into SBUF with high DMA priority.

    The load is split into ``n_chunks`` DMAs along the hidden tiles, so the first matmuls start
    before the whole weight arrives. Each DMA moves one contiguous block per partition. ``dma`` adds
    ``nisa.dma_copy`` arguments (the queue), and with ``after`` (an instruction) each DMA follows it in
    the schedule.
    """
    tiles_per_chunk = (_H_TILES + n_chunks - 1) // n_chunks
    for chunk_idx in range(n_chunks):
        tiles = slice(chunk_idx * tiles_per_chunk, min(_H_TILES, (chunk_idx + 1) * tiles_per_chunk))
        load = nisa.dma_copy(
            dst=dst[:, tiles, :],
            src=weight[:, tiles, :],
            priority=dma_priority(_CRITICAL_DMA_PRIORITY),
            name=f"{name}_c{chunk_idx}",
            **(dma or {}),
        )
        if after is not None:
            _instruction(load).depends_on(_instruction(after), kind=nisa.dependency.schedule)


# The decode alternates the DP x gather destination up to this many sequences per replica (``x_gather_parity``).
X_GATHER_PARITY_MAX_TOKENS = 8
# BF16 elements before and after each partition row of the alternate x gather destination: 32 bytes, so
# that no other tile shares a 32-byte SBUF block with the bytes that a peer writes.
_X_GUARD_ELEMENTS = 16


def x_gather_parity(replica_tokens: int) -> bool:
    """Return True when the decode alternates the DP x gather destination (``ffn_layer_sources``).

    A 2-D SBUF all-gather skips its global handshake only when neuronx-cc marks its output unique: no
    instruction touches the output between the previous gather of the replica group and this gather.
    The persistent destination ``x_dst`` of ``moe_collective_tiles`` takes the gather of every layer, so
    the previous gather writes the same bytes and the feed-forward body reads them. Every gather then
    keeps the handshake, except the first gather after an MLA loop. With the alternation, most gathers
    skip the handshake (93 layers, B=4 at 16K: 92 of 93, against 0).

    It is on up to ``X_GATHER_PARITY_MAX_TOKENS`` sequences per replica, on the static and the loop path.
    ``x_gather_alternates`` picks the layers that take the alternate destination.

    Example:
        >>> x_gather_parity(4)
        True
    """
    return replica_tokens <= X_GATHER_PARITY_MAX_TOKENS


def x_gather_alternates(layers: range, loop_layers: frozenset = frozenset()) -> frozenset:
    """Return the layers of ``layers`` whose DP x gather takes the alternate destination (``ffn_layer_sources``).

    ``layers`` runs from the first layer of the kernel. ``loop_layers`` holds every layer whose attention body
    runs a loop (the key loops of the MLA loop path, ``tkg_mla_paged.mla_runs_loops``).

    A gather that skips its global handshake lets a peer write the destination on this rank as soon as the
    peer reaches the gather. The peer gets there after the output all-reduce of the previous feed-forward
    body, its last collective with this rank. So between that all-reduce and the gather, no other tile on
    this rank may use the destination bytes. The persistent ``x_dst`` keeps its bytes for the whole kernel,
    so a gather into it is safe whatever the compiler does. The alternate destination is a new
    compiler-placed tile. Its skip is safe only if the unique_tensors check of neuronx-cc covers that whole
    window. The check covers one basic block: from the previous gather of the group, or from the block
    start. A loop ends a basic block, and the check does not look at the blocks before it. With the SB
    color rotation on, the compiler can place the gather after the key loops of an MLA layer on bytes that a
    loop tile uses, and mark it unique. A peer can then write those bytes while this rank is still in its
    loops.

    So a layer takes the alternate destination only when:

    - it is not the first layer of the kernel: its window would reach into the previous execution;
    - its attention runs no loop (not in ``loop_layers``);
    - the previous layer gathered into ``x_dst``. That makes the gathers alternate, so that the compiler can
      mark the gathers into ``x_dst`` unique.

    Every other layer gathers into ``x_dst``. A gather into ``x_dst`` right after an MLA loop is the first
    gather of its basic block, and the compiler marks it unique too. Without loops this gives the odd
    layers from an even first layer. When the alternate destination is persistent too (``x_dst2``), a loop
    does not matter: give no ``loop_layers``.

    Example:
        >>> sorted(x_gather_alternates(range(0, 12), frozenset({3, 7, 11})))
        [1, 4, 6, 8, 10]
    """
    alternates = set()
    previous_alternate = True
    for layer in layers:
        alternate = not previous_alternate and layer not in loop_layers
        if alternate:
            alternates.add(layer)
        previous_alternate = alternate
    return frozenset(alternates)


def ffn_layer_sources(sources: dict, alternate: bool, via_hbm: bool = False) -> dict:
    """Return the collective sources of a feed-forward body, with the alternate x gather destination or not.

    Without ``alternate`` the body gathers into the persistent ``x_dst`` of ``sources``. With it, the body
    gathers into the second persistent destination ``x_dst2`` of the loop path (``moe_collective_tiles``
    ``x_pair``), or else into a new compiler-placed tile with a 32-byte guard before and after each partition
    row. It takes the routed all-reduce result in the first half of that tile, as ``moe_collective_views``
    does for ``x_dst``. ``x_gather_alternates`` picks the layers. Between the gather of an alternate layer
    and the next gather into ``x_dst``, no instruction touches ``x_dst``, and the runtime skips the global
    handshake of that gather.

    The skip is safe: a peer writes ``x_dst`` again only at its next gather into ``x_dst``, after the output
    all-reduce of the body before. The input of that all-reduce needs every read of ``x_dst`` on this rank.
    No other tile takes the bytes of ``x_dst``, nor those of ``x_dst2``, so the same holds for ``x_dst2``.
    neuronx-cc marks the gather into a new tile unique only when no other tile of the basic block uses its
    bytes after the previous gather. Otherwise that gather keeps its handshake.

    ``via_hbm`` stores the input to the persistent HBM source ``x_src_gather_hbm``, gathers into the persistent HBM
    destination ``x_dst_hbm`` (``collective_sources`` ``x_hbm``) and loads it into ``x_dst``: no peer writes the SBUF
    bytes, whatever the compiler marks unique.

    Example:
        >>> f = latent_moe_decode(..., sources=ffn_layer_sources(sources, layer in alternates))
    """
    if via_hbm:
        return {**sources, "x_gather_hbm": sources["x_dst_hbm"], "x_gather_src_hbm": sources["x_src_gather_hbm"]}
    if not alternate:
        return sources
    rows, cols = sources["x_dst"].shape
    if "x_dst2" in sources:
        x_dst = sources["x_dst2"]
    else:
        base = nl.ndarray((rows, cols + 2 * _X_GUARD_ELEMENTS), dtype=nl.bfloat16, buffer=nl.sbuf)
        x_dst = base[0:rows, _X_GUARD_ELEMENTS : _X_GUARD_ELEMENTS + cols]
    return {**sources, "x_dst": x_dst, "routed_dst": _routed_dst_view(x_dst, sources.get("x_logits", False))}


def _routed_dst_view(x_dst: nl.NkiTensor, packed: bool = False) -> nl.NkiTensor:
    """Return the routed all-reduce result [128, T * 28] at the start of an x gather destination (plain or packed)."""
    cols = x_dst.shape[1]
    if packed:
        return x_dst[0:WORLD_SIZE, 0 : cols // _TOKEN_GATHER_WORDS * LATENT_SHARD]
    return x_dst[0:WORLD_SIZE, 0 : cols // 2]


def moe_collective_tiles(
    replica_tokens: int,
    sb=AUTO_SBUF,
    static: bool = True,
    moe_body_tiles: bool = True,
    x_pair: bool = False,
    sbuf_routed_reduce: bool = False,
    dp_size: int = DP_SIZE,
    x_dst: bool = True,
    gather: bool = True,
    x_logits: bool = False,
) -> dict:
    """Return the persistent SBUF sources and destinations of the feed-forward collectives, one tile each.

    A peer can still read the source of an all-gather or an all-reduce after the local output has landed,
    and a peer can write the destination before the local rank reaches the collective. The compiler gives
    the bytes of a tile to another tile after the last local access, so a tile that every layer writes again
    keeps its bytes for the whole kernel. Its next use waits for a later collective, which every peer joins
    only after it is done with the tile.

    The routed all-reduce result is a view of the x gather destination (``moe_collective_views``): a world
    collective separates each use of a byte from the next one. The routed all-reduce itself runs in HBM
    (``moe_collective_hbm``). ``static`` adds the latent gather destination. On the loop path, a block end (a
    drain and a barrier) follows the gather. ``moe_body_tiles=False`` leaves out the tiles that the DP4 x TP32 MoE
    body does not use: the latent gather tiles and the SBUF tiles of the output all-reduce.

    The output all-reduce runs in HBM too, so its SBUF tiles ``out`` (the source of the store) and ``out_dst``
    (the destination of the load) are only local DMA endpoints, which no peer accesses. The compiler orders the
    local accesses of their bytes. On the loop path each body then allocates them (``_tile``), and their bytes
    stay out of the explicit region. The static path keeps them in its arenas with the TP128 body. With the
    DP4 x TP32 body each body allocates them on both paths, so the loop-path layout is the one of the TP128 body.

    ``x_pair`` adds a second persistent x gather destination ``x_dst2`` on the loop path, for the layers that take
    the alternate destination (``x_gather_alternates``, ``ffn_layer_sources``).

    On the loop path, ``sbuf_routed_reduce`` runs the routed all-reduce in SBUF: it adds the persistent source
    ``routed_src_sb``, and the destination is the persistent ``routed_dst`` view of the x gather tile.

    ``x_dst=False`` leaves out the x gather destination: the caller allocates it (``moe_x_dst_tile``). ``gather=False``
    leaves out the latent gather source likewise (``moe_gather_src_tile``).

    ``x_logits`` (router_dtype fp16_local in the base body, DP4) packs the x gather rows: each token takes
    ``_TOKEN_GATHER_WORDS`` words (``_gather_replica_tokens``), and the packed source ``xl_src`` is added.
    """
    tokens = dp_size * replica_tokens
    words = _TOKEN_GATHER_WORDS if x_logits else _H_TILES
    tiles = {}
    if moe_body_tiles and gather:
        tiles["gather"] = moe_gather_src_tile(replica_tokens, sb, dp_size)
    if x_dst:
        tiles["x_dst"] = moe_x_dst_tile(replica_tokens, sb, dp_size, x_logits)
    if x_logits:
        tiles["xl_src"] = sb.tile((_PMAX, replica_tokens, words), nl.bfloat16, "ffn_xl_src")
    if x_pair and not static:
        tiles["x_dst2"] = sb.tile((_PMAX, tokens * words), nl.bfloat16, "ffn_x_dst2")
    if sbuf_routed_reduce and not static:
        tiles["routed_src_sb"] = sb.tile((WORLD_SIZE, tokens, LATENT_SHARD), nl.bfloat16, "ffn_routed_src")
    if static and moe_body_tiles:
        tiles["out"] = sb.tile((_PMAX, tokens, _H_TILES), nl.bfloat16, "ffn_out_src")
        tiles["out_dst"] = sb.tile((_PMAX, tokens * _H_TILES), nl.bfloat16, "ffn_out_dst")
        tiles["gather_dst"] = sb.tile((tokens, WORLD_SIZE * _GATHER_COLS), nl.float32, "moe_gather_dst")
    return tiles


def moe_gather_src_tile(replica_tokens: int, sb=AUTO_SBUF, dp_size: int = DP_SIZE) -> nl.NkiTensor:
    """Return the persistent latent gather source of ``moe_collective_tiles`` (``gather``)."""
    return sb.tile((dp_size * replica_tokens, _GATHER_COLS), nl.float32, "moe_gather_src")


def moe_x_dst_tile(replica_tokens: int, sb=AUTO_SBUF, dp_size: int = DP_SIZE, x_logits: bool = False) -> nl.NkiTensor:
    """Return the persistent x gather destination of ``moe_collective_tiles`` (``x_dst``; ``x_logits``: the packed
    rows)."""
    words = _TOKEN_GATHER_WORDS if x_logits else _H_TILES
    return sb.tile((_PMAX, dp_size * replica_tokens * words), nl.bfloat16, "ffn_x_dst")


def moe_ring_piece_bytes() -> int:
    """Return the bytes per partition of one half weight buffer of the static expert loop (``half_weight_buffers`` of
    ``moe_tkg_decode``) at the Kimi-K3 sizes: half of the I/512 tiles of one projection, D columns of packed MXFP4."""
    return -(-(EXPERT_INTERMEDIATE // 512) // 2) * LATENT_SIZE * 2


def moe_collective_views(tiles: dict, x_logits: bool = False) -> dict:
    """Return the SBUF tile of the routed all-reduce result: the start of the x gather destination (``x_logits``: the
    packed rows of ``moe_collective_tiles``)."""
    return {"routed_dst": _routed_dst_view(tiles["x_dst"], x_logits)}


def moe_collective_hbm(
    replica_tokens: int,
    latent_gather: bool = False,
    name: str = "",
    dp_size: int = DP_SIZE,
    reduce_rows: int = None,
    pair_shared: tuple = (),
    gather_cols: int = _GATHER_COLS,
    latent_gather_mx: bool = False,
) -> dict:
    """Return the persistent HBM sources and destinations of the routed and output all-reduces.

    ``latent_gather`` adds the source and destination of the latent gather, for the loop path, where a persistent
    SBUF destination does not fit the explicit region. ``gather_cols`` is the width of the latent gather rows:
    ``SCORE_GATHER_COLS`` for ``router_rows_drop_selection``. ``latent_gather_mx`` makes them the UINT32 blocks of
    ``latent_moe_decode`` ``latent_gather_mx``: source [8, W], destination [1024, W] (W = 2 T, or 2 T + 8 at T = 32:
    ``mx_gather_block_shape``).

    Each layer uses the same tensors, so a world collective separates each use from the next one (see
    ``moe_collective_tiles``). ``name`` ends the tensor names, for a kernel with more than one set.
    ``reduce_rows`` (default None = ``dp_size*replica_tokens``) sizes the routed and output all-reduce
    tensors at more rows than the call, so the DP1 decode's reductions have the verify's message size
    (the runtime picks the collective algorithm by message size; bitwise decode == verify).
    ``pair_shared`` (``common.PAIR_SHARED_CC_DST_SITES``) puts the output all-reduce destination (``moe_out``), the
    routed all-reduce destination (``moe_routed``) and the latent gather destination (``moe_gather``) in LNC1 shared
    HBM; () keeps every tensor private.
    """
    tokens = dp_size * replica_tokens
    rr = reduce_rows or tokens
    shapes = {
        "routed_hbm": (WORLD_SIZE, rr, LATENT_SHARD),
        "routed_dst_hbm": (WORLD_SIZE, rr * LATENT_SHARD),
        "out_hbm": (_PMAX, rr * _H_TILES),
        "out_dst_hbm": (_PMAX, rr * _H_TILES),
    }
    shared = {
        "out_dst_hbm": "moe_out" in pair_shared,
        "routed_dst_hbm": "moe_routed" in pair_shared,
        "gather_dst_hbm": "moe_gather" in pair_shared,
    }
    tensors = {
        key: nl.ndarray(
            shape, dtype=nl.bfloat16, buffer=pair_shared_buffer(shared.get(key, False)), name=f"moe{name}_{key}"
        )
        for key, shape in shapes.items()
    }
    if latent_gather and latent_gather_mx:
        rows, width = mx_gather_block_shape(tokens)
        for key, shape in (("gather_hbm", (rows, width)), ("gather_dst_hbm", (WORLD_SIZE * rows, width))):
            tensors[key] = nl.ndarray(shape, dtype=nl.uint32, buffer=nl.private_hbm, name=f"moe{name}_{key}")
    elif latent_gather:
        for key, shape in (
            ("gather_hbm", (tokens, gather_cols)),
            ("gather_dst_hbm", (WORLD_SIZE * tokens, gather_cols)),
        ):
            tensors[key] = nl.ndarray(
                shape, dtype=nl.float32, buffer=pair_shared_buffer(shared.get(key, False)), name=f"moe{name}_{key}"
            )
    return tensors


def _hbm_source(sources, key: str, shape: tuple) -> nl.NkiTensor:
    """Return the persistent BF16 HBM tensor ``key`` of ``sources``, or a new one."""
    if sources is None:
        return nl.ndarray(shape, dtype=nl.bfloat16, buffer=nl.private_hbm)
    tensor = sources[key]
    kernel_assert(tuple(tensor.shape) == tuple(shape), f"collective tensor {key}: {tensor.shape} is not {shape}")
    return tensor


def _routed_reduce_hbm(sources, tokens: int):
    """Return the routed all-reduce source, destination and their row count ``rr``.

    The persistent pair may hold more rows than this call (``reduce_rows``): the reduction then runs on the
    whole buffer so the DP1 decode's routed all-reduce has the verify's message size (bitwise decode ==
    verify). The caller stores its ``tokens`` rows in the first rows and reads them back.
    """
    if sources is None:
        src = nl.ndarray((WORLD_SIZE, tokens, LATENT_SHARD), dtype=nl.bfloat16, buffer=nl.private_hbm)
        dst = nl.ndarray((WORLD_SIZE, tokens * LATENT_SHARD), dtype=nl.bfloat16, buffer=nl.private_hbm)
        return src, dst, tokens
    src, dst = sources["routed_hbm"], sources["routed_dst_hbm"]
    rr = src.shape[1]
    kernel_assert(
        rr >= tokens
        and tuple(src.shape) == (WORLD_SIZE, rr, LATENT_SHARD)
        and tuple(dst.shape) == (WORLD_SIZE, rr * LATENT_SHARD),
        f"routed_hbm must be (128, >= {tokens}, {LATENT_SHARD})",
    )
    return src, dst, rr


def _source(sources, key: str, shape: tuple, dtype) -> nl.NkiTensor:
    """Return the persistent collective source ``key`` of ``sources``, or a new SBUF tile when it has none."""
    if sources is None or key not in sources:
        return nl.ndarray(shape, dtype=dtype, buffer=nl.sbuf)
    tile = sources[key]
    kernel_assert(tuple(tile.shape) == tuple(shape), f"collective source {key}: {tile.shape} is not {shape}")
    return tile


def _gather_replica_tokens(
    x_hT: nl.NkiTensor,
    dp_group: ReplicaGroup,
    name: str,
    sources=None,
    dp_size: int = DP_SIZE,
    world_group: ReplicaGroup = None,
    split: bool = False,
    local_logits: nl.NkiTensor = None,
    x16_hT: nl.NkiTensor = None,
) -> nl.NkiTensor:
    """All-gather the token-major hT hidden of the DP replicas.

    With the world x gather source ``x_world_src_hbm`` in ``sources`` (``decode_block.collective_sources``
    ``x_world``), the HBM gather runs over ``world_group``: each rank sends its own 4 partition rows of its
    replica (``_world_x_slice``). The world gather stacks rank 32 d + t at rows 4 (32 d + t) .. + 3, which is
    row 128 d + p of the DP gather for the partitions p = 4 t .. 4 t + 3 of replica d. So the destination gets
    the bytes of the DP gather, and the load is the same.

    With ``x_path_act`` in ``sources`` (``decode_block_body`` ``moe_x_path_act``; the world gather only), the store
    of the HBM x gather source, the slice of the world gather and the load of its destination run on the Activation
    hardware DGE queue, one after the other, not on the queues the compiler picks: the slice and the load then do
    not wait behind the SP queue's earlier DMAs (the KDA recurrent-state stores). The DMAs move the same bytes.

    With ``split`` (``GLUE_DMA_FORMS`` ``split_gather``; not with ``x_path_act``), the store of the HBM x gather source
    and the load of the HBM x gather destination each run as 2 DMAs of 64 partitions, on the SP and the Activation
    hardware DGE queues (``_split_rows_dma``). The other x gather forms have no such DMA and keep their program.

    With ``x_logits`` in ``sources`` (``moe_collective_tiles`` ``x_logits``), the rows are packed: each token sends its
    56 hidden words, then 2 words, which hold the FP32 ``local_logits`` [28, B] (``_peer_router_logits``) on partitions
    0..27. The packed source ``xl_src`` takes a copy of ``x_hT``, and ``x_dst`` holds the packed rows
    (``_packed_peer_logits`` reads the logits). A body without logits (the dense body) sends the hidden only.

    Returns:
        [128, 56, T] BF16 SBUF view. Element (p, j, r * B + b) is hidden index j * 128 + p of
        token b of replica r. ``[:, j, :]`` is the matmul stationary of hidden tile j: one free
        dimension with a step of 56 elements (58 in the packed rows), so no copy is necessary. With
        ``local_logits``, the pair (that view, the [28, T] FP32 peer logits). With ``dp_size`` 1 there is no
        gather: the view of ``x_hT`` is returned directly (T = the rows of this rank).
    """
    replica_tokens = x_hT.shape[1]
    if dp_size == 1:
        return x_hT.permute((0, 2, 1))
    token_world_src = (sources or {}).get("x_token_world_src_hbm")
    if token_world_src is not None:
        # attn_rs_moe_input: each rank holds its one normed token; world all-gather by token (through HBM,
        # collective_dim=0 partition-stack + a permute load, as the DP via_hbm path) gives every rank all WORLD_SIZE
        # tokens in world-rank order, which is the base body's replica-major token layout (world rank 32 d + t = d * R
        # + t at R = TP_SIZE). No extra transpose.
        kernel_assert(
            replica_tokens == 1 and world_group is not None and dp_size == DP_SIZE,
            f"x_token_world needs one token per rank, the world group and the DP{DP_SIZE} layout",
        )
        gathered = _source(sources, "x_dst", (_PMAX, WORLD_SIZE * _H_TILES), nl.bfloat16)
        nisa.dma_copy(dst=token_world_src, src=x_hT.reshape((_PMAX, _H_TILES)), name=f"{name}_token_store")
        world_dst = sources["x_token_world_dst_hbm"]
        srcs, dsts = [token_world_src], [world_dst]
        if x16_hT is not None:
            # The FP16 router rows: a second pair of the same gather (the pairs of one collective take one dtype).
            srcs.append(sources["x16_token_world_src_hbm"])
            dsts.append(sources["x16_token_world_dst_hbm"])
            nisa.dma_copy(
                dst=srcs[1], src=x16_hT.view(nl.bfloat16).reshape((_PMAX, _H_TILES)), name=f"{name}_token16_store"
            )
        ncc.all_gather(
            srcs=srcs,
            dsts=dsts,
            replica_group=world_group,
            collective_dim=0,
            name=f"{name}_token_gather",
        )
        nisa.dma_copy(
            dst=gathered.reshape((_PMAX, WORLD_SIZE, _H_TILES)),
            src=world_dst.reshape((WORLD_SIZE, _PMAX, _H_TILES)).permute((1, 0, 2)),
            name=f"{name}_token_load",
        )
        x_tokens_hT = gathered.reshape((_PMAX, WORLD_SIZE, _H_TILES)).permute((0, 2, 1))
        if x16_hT is None:
            return x_tokens_hT
        gathered16 = nl.ndarray((_PMAX, WORLD_SIZE, _H_TILES), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=gathered16,
            src=dsts[1].reshape((WORLD_SIZE, _PMAX, _H_TILES)).permute((1, 0, 2)),
            name=f"{name}_token16_load",
        )
        return x_tokens_hT, gathered16.view(nl.float16).permute((0, 2, 1))
    kernel_assert(x16_hT is None, "x16_hT rides the world token gather of attn_rs_moe_input only")
    tokens = dp_size * replica_tokens
    packed = bool((sources or {}).get("x_logits")) or local_logits is not None
    words = _TOKEN_GATHER_WORDS if packed else _H_TILES
    gathered = _source(sources, "x_dst", (_PMAX, tokens * words), nl.bfloat16)
    via_hbm = (sources or {}).get("x_gather_hbm")
    world_src = (sources or {}).get("x_world_src_hbm")
    kernel_assert(
        world_src is None or (via_hbm is not None and world_group is not None and dp_size == DP_SIZE),
        f"the world x gather needs the HBM x gather pair, the world group and the DP{DP_SIZE} layout",
    )
    x_path_act = bool((sources or {}).get("x_path_act"))
    kernel_assert(not x_path_act or world_src is not None, "x_path_act needs the world x gather (x_world_src_hbm)")
    kernel_assert(not (split and x_path_act), "glue_dmas 'split_gather' and x_path_act both place the x gather DMAs")
    act = _ACT_HWDGE if x_path_act else {}
    kernel_assert(
        not packed or (world_src is None and dp_size == DP_SIZE),
        f"the packed x gather (router_dtype fp16_local) needs the DP{DP_SIZE} gather, not the world x gather",
    )
    if packed:
        packed_src = _source(sources, "xl_src", (_PMAX, replica_tokens, words), nl.bfloat16)
        nisa.tensor_copy(dst=packed_src[:, :, 0:_H_TILES], src=x_hT)
        if local_logits is not None:
            nisa.tensor_copy(dst=packed_src.view(nl.float32)[0:DP_PEER_EXPERTS, :, _H_TILES // 2], src=local_logits)
        src = packed_src.reshape((_PMAX, replica_tokens * words))
    else:
        src = x_hT.reshape((_PMAX, replica_tokens * _H_TILES))
    if via_hbm is not None:
        if split:
            _split_rows_dma(sources["x_gather_src_hbm"], src, f"{name}_gather_store")
        else:
            nisa.dma_copy(dst=sources["x_gather_src_hbm"], src=src, name=f"{name}_gather_store", **act)
        src = sources["x_gather_src_hbm"]
    if world_src is not None:
        src = _world_x_slice(src, world_src, sources["x_world_rank"], name, act=x_path_act)
    ncc.all_gather(
        srcs=[src],
        dsts=[gathered if via_hbm is None else via_hbm],
        replica_group=dp_group if world_src is None else world_group,
        collective_dim=1 if via_hbm is None else 0,
        name=f"{name}_gather",
    )
    if via_hbm is not None:
        # The HBM gather stacks the replicas on rows: row r * 128 + p goes to columns of replica r of partition p.
        cols = replica_tokens * words
        load_dst = gathered.reshape((_PMAX, dp_size, cols))
        load_src = via_hbm.reshape((dp_size, _PMAX, cols)).permute((1, 0, 2))
        if split:
            _split_rows_dma(load_dst, load_src, f"{name}_gather_load")
        else:
            nisa.dma_copy(dst=load_dst, src=load_src, name=f"{name}_gather_load", **act)
    if not packed:
        return gathered.reshape((_PMAX, tokens, _H_TILES)).permute((0, 2, 1))
    x_tokens_hT = gathered.reshape((_PMAX, tokens, words))[:, :, 0:_H_TILES].permute((0, 2, 1))
    if local_logits is None:
        return x_tokens_hT
    return x_tokens_hT, _packed_peer_logits(gathered, tokens)


def _world_x_slice(
    x_src_hbm: nl.NkiTensor, world_src: nl.NkiTensor, tp_rank: nl.NkiTensor, name: str, act: bool = False
) -> nl.NkiTensor:
    """Copy this rank's partition rows 4 t .. 4 t + 3 (t = its TP rank) of the HBM x gather source to the world
    gather source ``world_src`` and return it.

    ``x_src_hbm`` is the [128, B * 56] BF16 replica hidden. The view of it as 32 blocks of 4 rows steps one block
    for each unit of the runtime offset ``tp_rank`` (the INT32 [1, 1] TP rank, ``tp_rank_tile``), so one DMA on the
    SP hardware DGE queue (with ``act``, the Activation hardware DGE queue) moves the 4 contiguous rows. A DMA moves
    the bytes: the values keep their bits.
    """
    rows = _PMAX // TP_SIZE
    width = rows * x_src_hbm.shape[1]
    kernel_assert(
        tuple(world_src.shape) == (rows, x_src_hbm.shape[1]),
        f"the world x gather source must be [{rows}, {x_src_hbm.shape[1]}], got {world_src.shape}",
    )
    blocks = x_src_hbm.reshape((TP_SIZE, width))
    nisa.dma_copy(
        dst=world_src.reshape((1, width)),
        src=blocks.ap(pattern=[[width, 1], [1, width]], offset=0, scalar_offset=tp_rank, indirect_dim=0),
        priority=_CRITICAL_DMA_PRIORITY,
        name=f"{name}_gather_slice",
        **(_ACT_HWDGE if act else _SP_HWDGE),
    )
    return world_src


def tp_rank_tile(rank_info: nl.NkiTensor, name: str, sb=AUTO_SBUF) -> nl.NkiTensor:
    """Return the INT32 [1, 1] SBUF TP rank of this rank (``rank_info`` column 2), the runtime offset of the world x
    gather slice (``_world_x_slice``). The first call loads it; the later calls of the kernel return the same tile.
    A kernel with an MLA loop path builds it first with a heap allocator ``sb``, as ``rank_selectors``."""
    tile = getattr(rank_info, "_k3_tp_rank", None)
    if tile is None:
        tile = sb.tile((1, 1), nl.int32)
        nisa.dma_copy(
            dst=tile,
            src=rank_info.ap(pattern=[[0, 1], [1, 1]], offset=2),
            priority=_CRITICAL_DMA_PRIORITY,
            name=f"{name}_tp",
        )
        object.__setattr__(rank_info, "_k3_tp_rank", tile)
    return tile


def rank_selectors(
    rank_info: nl.NkiTensor, name: str, sb=AUTO_SBUF, dp_size: int = DP_SIZE
) -> tuple[nl.NkiTensor, nl.NkiTensor]:
    """Return the replica one-hot and the world-rank one-hot of this rank, built once per kernel.

    The decode block calls the FFN bodies once per layer with the same ``rank_info``, so the first
    call builds the tiles and the later calls reuse them. Built in each call, the GpSimd iota of the
    next body waited for its SBUF, which the compiler reused from the output all-reduce buffer of
    the previous body. That iota held back the weight DMAs that the next attention body triggers
    from the in-order GpSimd queue.

    The tiles live for the rest of the kernel. A kernel with an MLA loop path builds them first with a
    heap allocator ``sb`` (``tkg_sbuf_alloc.SbufTiles(region, heap=True)``), so they take explicit
    addresses.

    Returns:
        ``(row_select, rank_select)``: FP32 ``[128, 4]`` (see ``_replica_one_hot``) and BF16
        ``[128, 1]`` (see ``_world_rank_one_hot``).
    """
    selectors = getattr(rank_info, "_k3_rank_selectors", None)
    if selectors is None:
        selectors = (_replica_one_hot(rank_info, name, sb, dp_size), _world_rank_one_hot(rank_info, name, sb))
        object.__setattr__(rank_info, "_k3_rank_selectors", selectors)
    return selectors


def _replica_one_hot(rank_info: nl.NkiTensor, name: str, sb=AUTO_SBUF, dp_size: int = DP_SIZE) -> nl.NkiTensor:
    """Build the FP32 [128, dp_size] one-hot of this rank's DP replica on every partition."""
    dp_rank = sb.tile((_PMAX, 1), nl.int32)
    nisa.dma_copy(
        dst=dp_rank,
        src=rank_info.ap(pattern=[[0, _PMAX], [1, 1]], offset=1),
        priority=dma_priority(_CRITICAL_DMA_PRIORITY),
        name=f"{name}_dp",
    )
    dp_rank_f32 = sb.tile((_PMAX, 1), nl.float32)
    nisa.tensor_copy(dst=dp_rank_f32, src=dp_rank)
    replica_idx = sb.tile((_PMAX, dp_size), nl.float32)
    nisa.iota(dst=replica_idx, pattern=[[1, dp_size]], offset=0, channel_multiplier=0)
    one_hot = sb.tile((_PMAX, dp_size), nl.float32)
    nisa.tensor_scalar(dst=one_hot, data=replica_idx, op0=nl.equal, operand0=dp_rank_f32)
    # The INT32 DP rank also gives the runtime offset of the replica-rows load (``replica_index``).
    object.__setattr__(rank_info, "_k3_dp_rank", dp_rank)
    return one_hot


def replica_index(rank_info: nl.NkiTensor) -> nl.NkiTensor:
    """Return the INT32 [1, 1] SBUF DP rank of this rank, the runtime offset of the replica-rows load
    (``GLUE_DMA_FORMS`` ``replica_rows``): the tile that ``rank_selectors`` loads once per kernel, so the load adds
    no instruction. Call ``rank_selectors`` first."""
    dp_rank = getattr(rank_info, "_k3_dp_rank", None)
    kernel_assert(dp_rank is not None, "replica_index needs the rank selectors (rank_selectors) of rank_info")
    return dp_rank[0:1, 0:1]


def _world_rank_one_hot(rank_info: nl.NkiTensor, name: str, sb=AUTO_SBUF) -> nl.NkiTensor:
    """Build the BF16 [128, 1] one-hot of this rank's global rank: partition g is 1 on rank g."""
    world_rank = sb.tile((WORLD_SIZE, 1), nl.int32)
    nisa.dma_copy(
        dst=world_rank,
        src=rank_info.ap(pattern=[[0, WORLD_SIZE], [1, 1]], offset=0),
        priority=dma_priority(_CRITICAL_DMA_PRIORITY),
        name=f"{name}_world",
    )
    world_rank_f32 = sb.tile((WORLD_SIZE, 1), nl.float32)
    nisa.tensor_copy(dst=world_rank_f32, src=world_rank)
    partition_idx = sb.tile((WORLD_SIZE, 1), nl.float32)
    nisa.iota(dst=partition_idx, pattern=[[0, 1]], offset=0, channel_multiplier=1)
    one_hot = sb.tile((WORLD_SIZE, 1), nl.bfloat16)
    nisa.tensor_scalar(dst=one_hot, data=partition_idx, op0=nl.equal, operand0=world_rank_f32)
    return one_hot


def _situ(
    gate_bf16: nl.NkiTensor, up_bf16: nl.NkiTensor, out: nl.NkiTensor | None = None, temps: tuple | None = None
) -> nl.NkiTensor:
    """SiTU in FP32 on BF16 inputs: ``bf16(4 tanh(g/4) sigmoid(g) * 25 tanh(u/25))``.

    The result goes to ``out`` (a BF16 [rows, cols] tile or view) if it is given, else to a new tile. ``temps``:
    optional three FP32 [rows, cols] tiles for the sigmoid, the gate term and the up tanh, else new tiles.
    """
    rows, cols = gate_bf16.shape
    if temps is None:
        gate_sigmoid = nl.ndarray((rows, cols), dtype=nl.float32, buffer=nl.sbuf)
    else:
        gate_sigmoid, gate_term, up_tanh = temps
    nisa.activation(dst=gate_sigmoid, op=nl.sigmoid, data=gate_bf16)
    if temps is None:
        gate_term = nl.ndarray((rows, cols), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=gate_term, op=nl.tanh, data=gate_bf16, scale=1.0 / SITU_BETA)
    nisa.scalar_tensor_tensor(
        dst=gate_term,
        data=gate_term,
        op0=nl.multiply,
        operand0=SITU_BETA,
        op1=nl.multiply,
        operand1=gate_sigmoid,
    )
    if temps is None:
        up_tanh = nl.ndarray((rows, cols), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=up_tanh, op=nl.tanh, data=up_bf16, scale=1.0 / SITU_LINEAR_BETA)
    activation = out if out is not None else nl.ndarray((rows, cols), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(
        dst=activation,
        data=up_tanh,
        op0=nl.multiply,
        operand0=SITU_LINEAR_BETA,
        op1=nl.multiply,
        operand1=gate_term,
    )
    return activation


def _shared_activation_T(in_psum: nl.NkiTensor, sb=AUTO_SBUF, col_offset: int = 0) -> nl.NkiTensor:
    """Build the [76, T] BF16 out-projection input with this rank's shared-expert activation in rows 28..75.

    Rows 0..27 are zero; the latent norm writes them after the experts. The transpose input has
    76 columns, so that every engine access starts at partition 0.

    Args:
        in_psum: [T, 131] FP32 PSUM in-projection; the shared gate and up columns are used.
        sb: the allocator of the returned tile, which lives across the expert call.
        col_offset: the in-projection column of PSUM column 0 (7 when the router columns have their own PSUM).

    Returns:
        [76, T] BF16 SBUF. Row 28 + c is ``situ(g, u)[:, c]^T``, the moving rows of the shared down.
    """
    tokens = in_psum.shape[0]
    shared_bf16 = nl.ndarray((tokens, 2 * SHARED_SHARD), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=shared_bf16, src=in_psum[:, SHARED_GATE_COLS[0] - col_offset : SHARED_UP_COLS[1] - col_offset])
    activation = nl.ndarray((tokens, MOE_OUT_PROJ_ROWS), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.memset(dst=activation[:, LATENT_UP_ROWS[0] : LATENT_UP_ROWS[1]], value=0.0)
    _situ(
        shared_bf16[:, 0:SHARED_SHARD],
        shared_bf16[:, SHARED_SHARD : 2 * SHARED_SHARD],
        out=activation[:, SHARED_DOWN_ROWS[0] : SHARED_DOWN_ROWS[1]],
    )
    activation_T_psum = _psum_bank(MOE_OUT_PROJ_ROWS, (tokens,), nl.bfloat16)
    nisa.nc_transpose(dst=activation_T_psum, data=activation)
    activation_T = sb.tile((MOE_OUT_PROJ_ROWS, tokens), nl.bfloat16, "moe_out_proj_input")
    nisa.tensor_copy(dst=activation_T, src=activation_T_psum)
    return activation_T


def _project_to_hT(terms, tokens: int, partial_hT: nl.NkiTensor = None) -> nl.NkiTensor:
    """Run a projection to the hidden in hT layout and return the BF16 partial in token-major order.

    Hidden tile j accumulates ``stationary^T @ moving`` over the (stationary, moving) pairs of
    ``terms(j)`` in FP32 PSUM: stationary [K, 128] and moving [K, T]. One PSUM bank holds
    512 // T hidden tiles, so the tiles run in groups, and each group is copied out before the
    next group needs a bank.

    ``partial_hT`` is an optional [128, T, 56] BF16 SBUF tile for the result.

    Returns:
        [128, T * 56] BF16 SBUF. Element (p, t * 56 + j) is hidden column j * 128 + p of token t.
    """
    tiles_per_bank = min(_H_TILES, _PSUM_BANK_FP32 // tokens)
    if partial_hT is None:
        partial_hT = nl.ndarray((_PMAX, tokens, _H_TILES), dtype=nl.bfloat16, buffer=nl.sbuf)
    for first_tile in range(0, _H_TILES, tiles_per_bank):
        n_tiles = min(tiles_per_bank, _H_TILES - first_tile)
        group_psum = _psum_bank(_PMAX, (n_tiles, tokens), nl.float32)
        for tile_offset in range(n_tiles):
            for stationary, moving in terms(first_tile + tile_offset):
                nisa.nc_matmul(dst=group_psum[:, tile_offset, :], stationary=stationary, moving=moving)
        nisa.tensor_copy(
            dst=partial_hT[:, :, first_tile : first_tile + n_tiles],
            src=group_psum.permute((0, 2, 1)),
        )
    return partial_hT.reshape((_PMAX, tokens * _H_TILES))


def _local_affinities(
    gathered: nl.NkiTensor,
    local_scores: nl.NkiTensor,
    local_selection: nl.NkiTensor,
    score_cols: tuple = _GATHER_SCORE_COLS,
    selection_cols: tuple = _GATHER_SELECTION_COLS,
) -> nl.NkiTensor:
    """Compute this rank's FP32 [T, 7] top-16 router affinities from the gathered scores.

    The selection value of expert e is ``score[e] + bias[e]``. The 16th largest selection value
    of a token is its threshold, so an expert is selected when its selection value is at least
    the threshold. The affinity is the score of a selected expert, divided by the sum of the
    selected scores plus 1e-20.

    Args:
        gathered: [T, 128, 28] FP32 SBUF gathered view. Columns 14..20 of block g are the scores
            of experts 7 g .. 7 g + 6, and columns 21..27 are their selection values.
        local_scores: [T, 7] FP32 SBUF scores of this rank's experts.
        local_selection: [T, 7] FP32 SBUF selection values of this rank's experts. They are
            bit-identical to the gathered values of this rank.
        score_cols, selection_cols: the column ranges of the scores and the selection values in each block of
            ``gathered`` (``latent_gather_mx``: a [T, 128, 14] router tile).
    """
    tokens = gathered.shape[0]
    scores = gathered[:, :, score_cols[0] : score_cols[1]]
    selection = gathered[:, :, selection_cols[0] : selection_cols[1]]

    # Two max8 rounds give the 16 largest selection values; the last one is the threshold.
    top8_first = nl.ndarray((tokens, 8), dtype=nl.float32, buffer=nl.sbuf)
    nisa.max8(dst=top8_first, src=selection)
    selection_rest = nl.ndarray((tokens, WORLD_SIZE, LOCAL_EXPERTS), dtype=nl.float32, buffer=nl.sbuf)
    nisa.nc_match_replace8(dst=selection_rest, data=selection, vals=top8_first, imm=_SELECTION_FLOOR)
    top8_second = nl.ndarray((tokens, 8), dtype=nl.float32, buffer=nl.sbuf)
    nisa.max8(dst=top8_second, src=selection_rest)
    threshold = top8_second[:, 7:8]

    selected_scores = nl.ndarray((tokens, WORLD_SIZE, LOCAL_EXPERTS), dtype=nl.float32, buffer=nl.sbuf)
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

    local_selected = nl.ndarray((tokens, LOCAL_EXPERTS), dtype=nl.float32, buffer=nl.sbuf)
    # scalar_tensor_tensor returns a raw instruction, not an NkiInstruction, in this release. Wrap it so
    # the scale below can depend on it.
    select = nisa.NkiInstruction(
        nisa.scalar_tensor_tensor(
            dst=local_selected,
            data=local_selection,
            op0=nl.greater_equal,
            operand0=threshold,
            op1=nl.multiply,
            operand1=local_scores,
        )
    )
    affinities = nl.ndarray((tokens, LOCAL_EXPERTS), dtype=nl.float32, buffer=nl.sbuf)
    scale = nisa.tensor_scalar(dst=affinities, data=local_selected, op0=nl.multiply, operand0=inv_score_sum)
    # The select and the scale both run on the Vector engine, and the scale can issue before the select has
    # written local_selected. The scheduler can place the select before the reciprocal; the scale then waits
    # on the reciprocal, and the in-order rule drops the select -> scale edge. When the reciprocal and the
    # select share no partition, nothing orders the two, so the scale can read a stale local_selected. A
    # runtime dependency adds a semaphore wait that the dependency-reduction pass keeps. The math does not change.
    scale.depends_on(select, kind=nisa.dependency.runtime)
    return affinities


def pad_row_flags(pad_rows: nl.NkiTensor, name: str) -> nl.NkiTensor:
    """Return the UINT32 ``[R, 1]`` SBUF predicate of the padding rows from the kernel input ``pad_rows``.

    ``pad_rows`` is the HBM int32 ``[R, 1]`` padding-row input of the step (``decode_block_body`` ``moe_pad_rows``):
    1 for a padding row, 0 for a real row, in the row order of the MoE body (row ``b`` of a decode step, row
    ``j * B + b`` of a verify step). Any nonzero value is a padding row.
    """
    rows = pad_rows.shape[0]
    kernel_assert(
        len(pad_rows.shape) == 2 and pad_rows.shape[1] == 1 and pad_rows.dtype == nl.int32,
        f"the padding-row input is int32 [R, 1], got {pad_rows.dtype} {tuple(pad_rows.shape)}",
    )
    kernel_assert(rows <= _PMAX, f"the padding-row input takes at most {_PMAX} rows, got {rows}")
    flags_i = nl.ndarray((rows, 1), dtype=nl.int32, buffer=nl.sbuf)
    # A few bytes: the default static queue (no DGE).
    nisa.dma_copy(dst=flags_i, src=pad_rows, name=f"{name}_pad_rows")
    flags_f = nl.ndarray((rows, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=flags_f, src=flags_i, engine=nisa.engine.vector)
    flags = nl.ndarray((rows, 1), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=flags, data=flags_f, op0=nl.not_equal, operand0=0.0, engine=nisa.engine.vector)
    return flags


def zero_pad_row_scores(scores: nl.NkiTensor, pad_rows: nl.NkiTensor):
    """Write 0 into the router scores ``[R, E]`` (SBUF FP32) of the padding rows (``pad_row_flags``), in place.

    A predicated copy: the scores of the other rows keep their bits, and a NaN score of a padding row goes too. A
    padding row then has the affinity 0 for every expert (its selected scores are all 0), so it routes to no expert.
    """
    size, cols = scores.shape
    kernel_assert(
        tuple(pad_rows.shape) == (size, 1), f"the padding-row flags are [{size}, 1], got {tuple(pad_rows.shape)}"
    )
    nisa.tensor_copy_predicated(dst=scores, src=0.0, predicate=pad_rows.ap(pattern=[[1, size], [0, cols]], offset=0))


def _latent_norm(
    routed: nl.NkiTensor,
    moe_vectors: nl.NkiTensor,
    rank_select: nl.NkiTensor,
    normed_T: nl.NkiTensor,
    name: str,
    weight_dma: dict = None,
    shard_form: str = "one_hot",
):
    """Apply the latent RMSNorm and keep this rank's latent shard, with the latent on partitions.

    ``normed = bf16(bf16(r32 * rsqrt(sum(r^2) / 3584 + 1e-5)) * w32)``, where the sum runs over
    all 3584 latent columns of the token.

    Args:
        routed: [128, T * 28] BF16 SBUF, the full routed latent. Element (g, 28 t + c) is latent
            column 28 g + c of token t. Or the [128, T, 28] view of the wide routed all-reduce
            (``WIDE_ROUTED_COLS``), element (g, t, c).
        moe_vectors: [1, 35] FP32 HBM; this rank's latent norm weight is used.
        rank_select: [128, 1] BF16 SBUF one-hot of this rank's global rank.
        normed_T: [28, T] BF16 SBUF destination: ``normed^T`` of this rank's latent columns, the
            moving rows of the latent up projection.
        weight_dma: None, or ``nisa.dma_copy`` arguments (the queue) of the norm weight load.
        shard_form: the form of the transposed shard (``LATENT_SHARD_FORMS``); the output bits do not change.
    """
    by_token = len(routed.shape) == 3
    tokens = routed.shape[1] if by_token else routed.shape[1] // LATENT_SHARD
    routed_by_token = routed if by_token else routed.reshape((WORLD_SIZE, tokens, LATENT_SHARD))
    routed_sq = nl.ndarray((WORLD_SIZE, tokens, LATENT_SHARD), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=routed_sq, data1=routed_by_token, data2=routed_by_token, op=nl.multiply)
    partial_sums = nl.ndarray((WORLD_SIZE, tokens), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=partial_sums, op=nl.add, data=routed_sq, axis=(2,))
    # The ones matmul adds the 128 partition sums and puts the total on the 28 shard partitions.
    ones = nl.ndarray((WORLD_SIZE, LATENT_SHARD), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones, value=1.0)
    total_psum = _psum_bank(LATENT_SHARD, (tokens,), nl.float32)
    nisa.nc_matmul(dst=total_psum, stationary=ones, moving=partial_sums, is_stationary_onezero=True)
    inv_rms = nl.ndarray((LATENT_SHARD, tokens), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=inv_rms, op=nl.rsqrt, data=total_psum, scale=1.0 / LATENT_SIZE, bias=LATENT_NORM_EPS)

    # Shard of this rank, transposed: column t = routed[rank, 28 t : 28 t + 28]^T (exact one-hot).
    if shard_form == "columns":
        # Column c of the shard for every token (``LATENT_SHARD_FORMS``): the same one-hot sums, 28 matmuls.
        kernel_assert(tokens <= _PMAX, f"latent shard_form 'columns' needs at most {_PMAX} tokens, got {tokens}")
        rows_psum = _psum_bank(tokens, (LATENT_SHARD,), nl.float32)
        for col in range(LATENT_SHARD):
            nisa.nc_matmul(
                dst=rows_psum[:, col : col + 1], stationary=routed_by_token[:, 0:tokens, col], moving=rank_select
            )
        # The values are BF16 values, so the copy and the transpose keep their bits.
        rows_sb = nl.ndarray((tokens, LATENT_SHARD), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=rows_sb, src=rows_psum, engine=nisa.engine.vector)
        shard_psum = _psum_bank(LATENT_SHARD, (tokens,), nl.bfloat16)
        nisa.nc_transpose(dst=shard_psum, data=rows_sb, engine=nisa.engine.tensor)
    else:
        kernel_assert(shard_form == "one_hot", f"latent shard_form must be one of {LATENT_SHARD_FORMS}")
        shard_psum = _psum_bank(LATENT_SHARD, (tokens,), nl.float32)
        for token_idx in range(tokens):
            nisa.nc_matmul(
                dst=shard_psum[:, token_idx : token_idx + 1],
                stationary=routed_by_token[:, token_idx, :]
                if by_token
                else routed[:, token_idx * LATENT_SHARD : (token_idx + 1) * LATENT_SHARD],
                moving=rank_select,
            )
    norm_weight = nl.ndarray((LATENT_SHARD, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=norm_weight,
        src=moe_vectors.ap(pattern=[[1, LATENT_SHARD], [1, 1]], offset=LATENT_NORM_COLS[0]),
        priority=_CRITICAL_DMA_PRIORITY,
        name=f"{name}_weight",
        **(weight_dma or {}),
    )
    scaled = nl.ndarray((LATENT_SHARD, tokens), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=scaled, data1=shard_psum, data2=inv_rms, op=nl.multiply)
    nisa.tensor_scalar(dst=normed_T, data=scaled, op0=nl.multiply, operand0=norm_weight)


def _all_reduce_select_rows(
    partial_hT: nl.NkiTensor,
    row_select: nl.NkiTensor,
    replica_tokens: int,
    world_group: ReplicaGroup,
    name: str,
    sources=None,
    dp_size: int = DP_SIZE,
    replica_index: nl.NkiTensor = None,
) -> nl.NkiTensor:
    """All-reduce the token-major [128, T * 56] partial over the world and keep the B tokens of this replica.

    The tokens of replica r are the columns [56 B r, 56 B (r + 1)). A copy of replica 0 and then a
    predicated copy of each other replica select the rows of this replica. A multiply by the one-hot
    would add 0 x row for each other replica, and 0 x NaN is NaN: a NaN in token b of one replica
    would then reach token b of every replica. With ``dp_size`` 1 every rank keeps all T rows, so the
    all-reduce result is the output and no row select runs. With ``replica_index`` (the INT32 [1, 1] DP
    rank, ``GLUE_DMA_FORMS`` ``replica_rows``), the load reads only the columns of this replica
    (``_load_replica_rows``) and no copy runs.

    Returns:
        [128, B, 56] BF16 SBUF.
    """
    width_total = dp_size * replica_tokens * _H_TILES
    shape = (_PMAX, width_total)
    if sources is not None:
        # The persistent pair may hold more rows than this call (``reduce_rows``): the all-reduce runs on
        # the whole buffer, so the DP1 decode's reduction has the verify's message size (the runtime picks
        # the collective algorithm by message size; bitwise decode == verify). Store/read the first rows.
        partial_hbm, reduced_hbm = sources["out_hbm"], sources["out_dst_hbm"]
        kernel_assert(
            tuple(partial_hbm.shape) == tuple(reduced_hbm.shape)
            and partial_hbm.shape[0] == _PMAX
            and partial_hbm.shape[1] >= width_total,
            f"out_hbm must be (128, >= {width_total})",
        )
    else:
        partial_hbm = nl.ndarray(shape, dtype=nl.bfloat16, buffer=nl.private_hbm)
        reduced_hbm = nl.ndarray(shape, dtype=nl.bfloat16, buffer=nl.private_hbm)
    # Without padding use the whole tensor so the IR is unchanged; with padding keep the first rows.
    store_dst = partial_hbm if partial_hbm.shape[1] == width_total else partial_hbm[0:_PMAX, 0:width_total]
    load_src = reduced_hbm if reduced_hbm.shape[1] == width_total else reduced_hbm[0:_PMAX, 0:width_total]
    nisa.dma_copy(
        dst=store_dst,
        src=partial_hT.reshape(shape),
        priority=dma_priority(_CRITICAL_DMA_PRIORITY),
        name=f"{name}_store",
    )
    check_pair_group(reduced_hbm, world_group, f"{name}_reduce")
    release_weight_gate(
        ncc.all_reduce(
            srcs=[partial_hbm],
            dsts=[reduced_hbm],
            replica_group=world_group,
            op=nl.add,
            name=f"{name}_reduce",
        )
    )
    if replica_index is not None:
        kernel_assert(dp_size > 1, "replica_index selects the rows of one of several DP replicas")
        return _load_replica_rows(reduced_hbm, replica_index, replica_tokens, f"{name}_load")
    reduced = _tile(sources, "out_dst", shape, nl.bfloat16)
    release_gate_consumer(
        nisa.dma_copy(dst=reduced, src=load_src, priority=dma_priority(_CRITICAL_DMA_PRIORITY), name=f"{name}_load")
    )
    return _select_replica_rows(reduced, row_select, replica_tokens, dp_size)


def _select_replica_rows(
    reduced: nl.NkiTensor, row_select: nl.NkiTensor, replica_tokens: int, dp_size: int = DP_SIZE
) -> nl.NkiTensor:
    """Return the B tokens of this replica, [128, B, 56] BF16 SBUF, of the token-major [128, T * 56] output ``reduced``
    (``_all_reduce_select_rows``)."""
    width = replica_tokens * _H_TILES
    out_hT = nl.ndarray((_PMAX, replica_tokens, _H_TILES), dtype=nl.bfloat16, buffer=nl.sbuf)
    out_rows = out_hT.reshape((_PMAX, width))
    if dp_size == 1:
        nisa.tensor_copy(dst=out_rows, src=reduced)
        return out_hT
    reduced_by_replica = reduced.reshape((_PMAX, dp_size, width))
    is_replica = nl.ndarray((_PMAX, dp_size), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=is_replica, data=row_select, op0=nl.not_equal, operand0=0.0)
    nisa.tensor_copy(dst=out_rows, src=reduced_by_replica[:, 0, :])
    for replica_idx in range(1, dp_size):
        nisa.tensor_copy_predicated(
            dst=out_rows,
            src=reduced_by_replica[:, replica_idx, :],
            predicate=is_replica.ap(pattern=[[dp_size, _PMAX], [0, width]], offset=replica_idx),
        )
    return out_hT


def _load_replica_rows(
    result_hbm: nl.NkiTensor, replica_index: nl.NkiTensor, replica_tokens: int, name: str
) -> nl.NkiTensor:
    """Load the B tokens of this replica from the token-major HBM result of the output collective (``GLUE_DMA_FORMS``
    ``replica_rows``).

    ``result_hbm`` is [128, C] BF16 with C >= T * 56: partition p holds hidden index 128 j + p of token t at column
    56 t + j, and the tokens of replica r are the columns [56 B r, 56 B (r + 1)). The view of the rows as blocks of 56 B
    columns steps one block for each unit of the runtime offset ``replica_index`` (the INT32 [1, 1] DP rank), so the
    DMA reads the 56 B columns of this replica from each of the 128 rows, on the SP hardware DGE queue. A DMA moves the
    bytes: every value has the bits of the row select of ``_all_reduce_select_rows``, and a NaN keeps its payload,
    which the select copies may replace by the quiet NaN of its sign. The load is the consumer gate of the collective
    (``release_gate_consumer``), as the load of all rows is.

    Returns:
        [128, B, 56] BF16 SBUF.
    """
    width = replica_tokens * _H_TILES
    row_cols = result_hbm.shape[1]
    kernel_assert(
        tuple(result_hbm.shape[:1]) == (_PMAX,) and row_cols % width == 0,
        f"the replica rows need a [{_PMAX}, k * {width}] result, got {result_hbm.shape}",
    )
    blocks = result_hbm.reshape((_PMAX * (row_cols // width), width))
    out_hT = nl.ndarray((_PMAX, replica_tokens, _H_TILES), dtype=nl.bfloat16, buffer=nl.sbuf)
    release_gate_consumer(
        nisa.dma_copy(
            dst=out_hT.reshape((_PMAX, width)),
            src=blocks.ap(
                pattern=[[row_cols, _PMAX], [1, width]], offset=0, scalar_offset=replica_index, indirect_dim=0
            ),
            priority=_CRITICAL_DMA_PRIORITY,
            name=name,
            **_SP_HWDGE,
        ),
        engine=_SP_HWDGE["engine"],
    )
    return out_hT


# The out-projection forms (``OUT_PROJ_FORMS``) other than ``reduced``.


def _validate_out_proj_form(out_proj_form: str, routed_kernel: str, routed_reduce_dtype: str, dp_size: int, sources):
    """Check the conditions of ``out_proj_form`` (``OUT_PROJ_FORMS``)."""
    kernel_assert(
        out_proj_form in OUT_PROJ_FORMS, f"out_proj_form must be one of {OUT_PROJ_FORMS}, got {out_proj_form!r}"
    )
    if out_proj_form == "reduced":
        return
    kernel_assert(routed_kernel in TP128_KERNELS, f"out_proj_form {out_proj_form!r} needs a TP128 routed kernel")
    kernel_assert(dp_size == DP_SIZE, f"out_proj_form {out_proj_form!r} needs the DP{DP_SIZE} layout, got {dp_size}")
    kernel_assert(routed_reduce_dtype == "bf16", f"out_proj_form {out_proj_form!r} needs the BF16 routed all-reduce")
    if out_proj_form in WIDE_FORMS:
        kernel_assert(
            sources is not None and "routed_wide_src_sb" in sources and "routed_wide_dst" in sources,
            f"out_proj_form {out_proj_form!r} needs the wide routed all-reduce tiles of the loop path "
            "(decode_block.collective_sources moe_out_proj_form, with moe_sbuf_reduces routed_tp128)",
        )
    else:
        # column_shared_ag: the gather trigger sits between the first two weight streams of the sel_load kernel.
        kernel_assert(routed_kernel == "sel_load", f"out_proj_form {out_proj_form!r} needs the sel_load routed kernel")
        kernel_assert(
            sources is not None
            and "routed_src_sb" in sources
            and "routed_dst" in sources
            and "shared_ag_src" in sources
            and "shared_ag_dst" in sources,
            f"out_proj_form {out_proj_form!r} needs the SBUF routed all-reduce tiles and the shared gather tiles of the "
            "loop path (decode_block.collective_sources moe_out_proj_form, with moe_sbuf_reduces routed_tp128)",
        )
    kernel_assert(
        out_proj_form not in COLUMN_FORMS or _hbm_site(sources, "outg"),
        f"out_proj_form {out_proj_form!r} needs the HBM pair of the output all-gather (outg_src_hbm, outg_dst_hbm)",
    )


def _column_norm_offset(in_proj_form: str) -> int:
    """Return the first ``moe_vectors`` column of the 3,584 latent norm weights of ``column``: after the columns of the
    in-projection form (``MOE_VECTOR_COLS`` or ``DP_LOCAL_VECTOR_COLS``)."""
    return DP_LOCAL_VECTOR_COLS if in_proj_form == "dp_local" else MOE_VECTOR_COLS


def out_proj_collective_tiles(replica_tokens: int, sb=AUTO_SBUF, dp_size: int = DP_SIZE) -> dict:
    """Return the persistent SBUF source and destination of the wide routed all-reduce of the out-projection forms
    other than ``reduced`` (``OUT_PROJ_FORMS``): ``routed_wide_src_sb`` [128, T, 76] and ``routed_wide_dst``
    [128, T * 76] BF16. The caller clears them at the kernel start, so the shared slots of ``reduced_wide`` stay zero.
    """
    tokens = dp_size * replica_tokens
    return {
        "routed_wide_src_sb": sb.tile((WORLD_SIZE, tokens, WIDE_ROUTED_COLS), nl.bfloat16, "ffn_routed_wide_src"),
        "routed_wide_dst": sb.tile((WORLD_SIZE, tokens * WIDE_ROUTED_COLS), nl.bfloat16, "ffn_routed_wide_dst"),
    }


def out_proj_collective_hbm(replica_tokens: int, name: str = "", dp_size: int = DP_SIZE) -> dict:
    """Return the persistent HBM pair of the output all-gather of ``column``: ``outg_src_hbm`` [1, T * 56] (the hidden
    columns of this rank) and ``outg_dst_hbm`` [128, T * 56] (row g from rank g: the token-major hT layout)."""
    width = dp_size * replica_tokens * _H_TILES
    return {
        key: nl.ndarray(shape, dtype=nl.bfloat16, buffer=nl.private_hbm, name=f"moe{name}_{key}")
        for key, shape in (("outg_src_hbm", (1, width)), ("outg_dst_hbm", (WORLD_SIZE, width)))
    }


def rank_row_one_hot(rank_info: nl.NkiTensor, name: str, sb=AUTO_SBUF) -> nl.NkiTensor:
    """Return the BF16 [128, 128] one-hot of this rank's global rank on the free axis, built once per kernel: element
    (k, p) is 1 for p = g on rank g. Its first 76 rows are the stationary operand that moves the shared activation of
    this rank to its partition of the wide routed all-reduce (``_shared_rank_slots``). Like ``rank_selectors``, a
    kernel with an MLA loop path builds it first with a heap allocator ``sb``."""
    one_hot = getattr(rank_info, "_k3_rank_row", None)
    if one_hot is not None:
        return one_hot
    world_rank = sb.tile((_PMAX, 1), nl.int32)
    nisa.dma_copy(
        dst=world_rank,
        src=rank_info.ap(pattern=[[0, _PMAX], [1, 1]], offset=0),
        priority=_CRITICAL_DMA_PRIORITY,
        name=f"{name}_row_rank",
    )
    world_rank_f32 = sb.tile((_PMAX, 1), nl.float32)
    nisa.tensor_copy(dst=world_rank_f32, src=world_rank)
    rank_idx = sb.tile((_PMAX, WORLD_SIZE), nl.float32)
    nisa.iota(dst=rank_idx, pattern=[[1, WORLD_SIZE]], offset=0, channel_multiplier=0)
    one_hot = sb.tile((_PMAX, WORLD_SIZE), nl.bfloat16)
    nisa.tensor_scalar(dst=one_hot, data=rank_idx, op0=nl.equal, operand0=world_rank_f32)
    object.__setattr__(rank_info, "_k3_rank_row", one_hot)
    return one_hot


def _shared_rank_slots(out_proj_input: nl.NkiTensor, rank_row: nl.NkiTensor, slots: nl.NkiTensor):
    """Write this rank's shared activation to its partition of the shared slots and zeros to the other partitions.

    ``slots[p, t, s] = (p == g) * out_proj_input[28 + s, t]``, exact: each value is one product by 1 or 0 and a sum with
    zeros. The matmul runs over all 76 rows of the out-projection input (rows 0..27 are zeros at this point, and their
    weights are zero too), so every engine access starts at partition 0. A NaN of token t reaches the slots of token t
    on every partition, and token t of the output is NaN in every column anyway (its shared activation reaches every
    hidden column).

    Args:
        out_proj_input: [76, T] BF16 SBUF, the out-projection input of ``_shared_activation_T`` (rows 28..75: the shared
            activation of this rank, transposed).
        rank_row: [128, 128] BF16 SBUF, ``rank_row_one_hot``.
        slots: [128, T, 48] BF16 SBUF view, the shared slots of the wide routed all-reduce source.
    """
    tokens = out_proj_input.shape[1]
    ident = nl.shared_identity_matrix(n=_PMAX, dtype=nl.bfloat16)
    # expand[k, t, s] = out_proj_input[k, t] if k == 28 + s else 0: the moving rows that put slot s of token t in
    # column (t, s) of the matmul output.
    expand = nl.ndarray((MOE_OUT_PROJ_ROWS, tokens, SHARED_SHARD), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=expand,
        data1=out_proj_input.expand_dim(2).broadcast(2, SHARED_SHARD),
        data2=ident[0:MOE_OUT_PROJ_ROWS, SHARED_DOWN_ROWS[0] : SHARED_DOWN_ROWS[1]].expand_dim(1).broadcast(1, tokens),
        op=nl.multiply,
    )
    expand_cols = expand.reshape((MOE_OUT_PROJ_ROWS, tokens * SHARED_SHARD))
    chunk_tokens = _PSUM_BANK_FP32 // SHARED_SHARD
    for first in range(0, tokens, chunk_tokens):
        count = min(chunk_tokens, tokens - first)
        slot_psum = _psum_bank(_PMAX, (count * SHARED_SHARD,), nl.float32)
        nisa.nc_matmul(
            dst=slot_psum,
            stationary=rank_row[0:MOE_OUT_PROJ_ROWS, 0:WORLD_SIZE],
            moving=expand_cols[:, first * SHARED_SHARD : (first + count) * SHARED_SHARD],
        )
        nisa.tensor_copy(
            dst=slots[0:_PMAX, first : first + count, 0:SHARED_SHARD],
            src=slot_psum.reshape((_PMAX, count, SHARED_SHARD)),
        )


def _wide_routed_source(
    src: nl.NkiTensor, expert_rows: nl.NkiTensor, out_proj_input: nl.NkiTensor, rank_info: nl.NkiTensor, name: str
):
    """Fill the wide routed all-reduce source ``src`` [128, T, 76].

    The latent columns take the routed partial as the [128, T, 28] source of ``reduced`` does (``expert_rows``, the
    same 56-byte rows). With ``out_proj_input`` (``column``; the [76, T] tile of ``_shared_activation_T``) this rank
    also writes its shared activation to the slots of its partition and zeros to the other slots, so the sum gathers
    the shared activation of every rank exactly. Without it (``reduced_wide``) the slots keep the zeros of the
    kernel-start clear.
    """
    tokens = src.shape[1]
    nisa.dma_copy(
        dst=src[0:WORLD_SIZE, 0:tokens, 0:LATENT_SHARD],
        src=expert_rows,
        priority=_CRITICAL_DMA_PRIORITY,
        name=f"{name}_routed_store",
    )
    if out_proj_input is not None:
        _shared_rank_slots(
            out_proj_input,
            rank_row_one_hot(rank_info, f"{name}_rank"),
            src[0:WORLD_SIZE, 0:tokens, WIDE_SHARED_COLS[0] : WIDE_SHARED_COLS[1]],
        )


def _wide_routed_reduce(
    expert_rows: nl.NkiTensor,
    out_proj_input: nl.NkiTensor,
    rank_info: nl.NkiTensor,
    tokens: int,
    world_group: ReplicaGroup,
    name: str,
    sources: dict,
) -> nl.NkiTensor:
    """All-reduce the routed partial in the wide layout (``WIDE_ROUTED_COLS``); return the [128, T, 76] SBUF result.

    The source is the persistent ``routed_wide_src_sb`` (``_wide_routed_source``) and the destination the persistent
    ``routed_wide_dst``. The all-reduce is the gate of the next body's weights, as the routed all-reduce of ``reduced``.
    """
    src = _source(sources, "routed_wide_src_sb", (WORLD_SIZE, tokens, WIDE_ROUTED_COLS), nl.bfloat16)
    _wide_routed_source(src, expert_rows, out_proj_input, rank_info, name)
    dst = _source(sources, "routed_wide_dst", (WORLD_SIZE, tokens * WIDE_ROUTED_COLS), nl.bfloat16)
    release_weight_gate(
        ncc.all_reduce(
            srcs=[src.reshape((WORLD_SIZE, tokens * WIDE_ROUTED_COLS))],
            dsts=[dst],
            replica_group=world_group,
            op=nl.add,
            name=f"{name}_routed_reduce",
        )
    )
    return dst.reshape((WORLD_SIZE, tokens, WIDE_ROUTED_COLS))


def shared_gather_tiles(replica_tokens: int, sb=AUTO_SBUF, dp_size: int = DP_SIZE, name: str = "") -> dict:
    """Return the persistent SBUF source and destination of the shared all-gather of ``column_shared_ag``
    (``SHARED_GATHER_COLS``): ``shared_ag_src`` [1, T * 48] and ``shared_ag_dst`` [128, T * 48] BF16. A peer writes
    the destination in its gather and reads the source after the local gather is done, so both keep their bytes for
    the whole kernel, as the other collective tiles do (``moe_collective_tiles``)."""
    width = dp_size * replica_tokens * SHARED_GATHER_COLS
    return {
        "shared_ag_src": sb.tile((1, width), nl.bfloat16, f"ffn_shared_ag_src{name}"),
        "shared_ag_dst": sb.tile((WORLD_SIZE, width), nl.bfloat16, f"ffn_shared_ag_dst{name}"),
    }


def _shared_activation_rows(
    in_psum: nl.NkiTensor,
    sb=AUTO_SBUF,
    col_offset: int = 0,
    tiles: dict = None,
    after=None,
    copies: list = None,
) -> nl.NkiTensor:
    """Return this rank's shared-expert activation [T, 48] BF16 SBUF (``column_shared_ag``): the rows of
    ``_shared_activation_T`` before its transpose, with the same instructions, so the same bits.

    Args:
        in_psum: [T, C] FP32 PSUM in-projection; the shared gate and up columns are used.
        sb: the allocator of the returned tile, which a DMA reads after the call.
        col_offset: the in-projection column of PSUM column 0 (as ``_shared_activation_T``).
        tiles: optional tiles of ``dp_local_shared_rows_tiles`` (the result in ``rows``). None: new tiles, the result
            from ``sb``.
        after: optional instruction that the PSUM copy (the first instruction) follows in the schedule.
        copies: optional list that gets the PSUM copy instruction.
    """
    tokens = in_psum.shape[0]
    if tiles is None:
        shared_bf16 = nl.ndarray((tokens, 2 * SHARED_SHARD), dtype=nl.bfloat16, buffer=nl.sbuf)
    else:
        shared_bf16 = tiles["gate_up"]
    copy = nisa.tensor_copy(
        dst=shared_bf16, src=in_psum[:, SHARED_GATE_COLS[0] - col_offset : SHARED_UP_COLS[1] - col_offset]
    )
    if after is not None:
        _instruction(copy).depends_on(_instruction(after), kind=nisa.dependency.schedule)
    if copies is not None:
        copies.append(copy)
    rows = sb.tile((tokens, SHARED_SHARD), nl.bfloat16, "moe_shared_rows") if tiles is None else tiles["rows"]
    _situ(
        shared_bf16[:, 0:SHARED_SHARD],
        shared_bf16[:, SHARED_SHARD : 2 * SHARED_SHARD],
        out=rows,
        temps=None if tiles is None else tiles["situ"],
    )
    return rows


def _shared_gather_source(rows: nl.NkiTensor, sources: dict, name: str) -> nl.NkiTensor:
    """Pack the shared activation rows [T, 48] (``_shared_activation_rows``) into the persistent single-partition
    source of the shared gather ``shared_ag_src`` [1, T * 48] (element 48 t + s is row t, column s), and return the
    source. The values are copied exactly.

    An SBUF-to-SBUF DMA keeps the partition of each element, so the rows go through a local HBM tensor: one DMA
    stores them, and one loads the [1, T * 48] view of that tensor to the source.
    """
    tokens = rows.shape[0]
    width = tokens * SHARED_GATHER_COLS
    call = next_call("moe_hbm")
    rows_hbm = nl.ndarray(
        (tokens, SHARED_GATHER_COLS), dtype=nl.bfloat16, buffer=nl.private_hbm, name=f"{name}_shared_rows_{call}"
    )
    nisa.dma_copy(dst=rows_hbm, src=rows, priority=_CRITICAL_DMA_PRIORITY, name=f"{name}_shared_store")
    src = _source(sources, "shared_ag_src", (1, width), nl.bfloat16)
    nisa.dma_copy(
        dst=src, src=rows_hbm.reshape((1, width)), priority=_CRITICAL_DMA_PRIORITY, name=f"{name}_shared_pack"
    )
    return src


def _shared_all_gather(src: nl.NkiTensor, dst: nl.NkiTensor, world_group: ReplicaGroup, name: str):
    """All-gather the packed shared activation of each rank over the world (``column_shared_ag``): row g of ``dst``
    [128, T * 48] is the source of rank g, so the [128, T, 48] view is the shared slots of the wide routed tile of
    ``column``. Return the gather instruction."""
    return ncc.all_gather(
        srcs=[src], dsts=[dst], replica_group=world_group, collective_dim=0, name=f"{name}_shared_gather"
    )


def _latent_norm_columns(latent: nl.NkiTensor, norm_weight: nl.NkiTensor, name: str) -> nl.NkiTensor:
    """Apply the latent RMSNorm of ``_latent_norm`` to every latent column (``column``).

    Args:
        latent: [128, T, 28] BF16 SBUF view of the wide routed all-reduce: element (g, t, c) is latent column
            28 g + c of token t.
        norm_weight: [128, 28] FP32 SBUF, element (g, c) the norm weight of latent column 28 g + c.
        name: Instruction name prefix.

    Returns:
        [128, T, 28] BF16 SBUF ``normed``, in the layout of ``latent``. Partition g has the bits of ``normed_T^T``
        of ``_latent_norm`` on rank g: the same square sums, the same ones matmul on every output partition, the same
        rsqrt and the same two roundings.
    """
    tokens = latent.shape[1]
    latent_sq = nl.ndarray((WORLD_SIZE, tokens, LATENT_SHARD), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=latent_sq, data1=latent, data2=latent, op=nl.multiply)
    partial_sums = nl.ndarray((WORLD_SIZE, tokens), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=partial_sums, op=nl.add, data=latent_sq, axis=(2,))
    ones = nl.ndarray((WORLD_SIZE, WORLD_SIZE), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones, value=1.0)
    total_psum = _psum_bank(WORLD_SIZE, (tokens,), nl.float32)
    nisa.nc_matmul(dst=total_psum, stationary=ones, moving=partial_sums, is_stationary_onezero=True)
    inv_rms = nl.ndarray((WORLD_SIZE, tokens), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=inv_rms, op=nl.rsqrt, data=total_psum, scale=1.0 / LATENT_SIZE, bias=LATENT_NORM_EPS)
    scaled = nl.ndarray((WORLD_SIZE, tokens, LATENT_SHARD), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=scaled, data1=latent, data2=inv_rms.expand_dim(2).broadcast(2, LATENT_SHARD), op=nl.multiply)
    normed = nl.ndarray((WORLD_SIZE, tokens, LATENT_SHARD), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=normed, data1=scaled, data2=norm_weight.expand_dim(1).broadcast(1, tokens), op=nl.multiply)
    return normed


def _column_out_proj(
    routed_wide: nl.NkiTensor,
    moe_out_proj: nl.NkiTensor,
    moe_vectors: nl.NkiTensor,
    norm_offset: int,
    name: str,
    shared: nl.NkiTensor = None,
) -> nl.NkiTensor:
    """Return this rank's hidden columns 128 j + g of every token, [T, 56] BF16 SBUF (``column``, ``column_shared_ag``).

    One FP32 PSUM accumulation over the 76 K-tiles (``COLUMN_K_TILES``) of 128 input rows, rounded to BF16 once. The
    stationary operand of K-tile c is column c of the wide routed tile [128, T] (the shared slots, then the normed
    latent), and the moving operand is K-tile c of the weights [128, 56]. The 48 shared K-tiles need only the routed
    all-reduce (with ``shared``, only the shared gather), so they run before the latent norm.

    Args:
        routed_wide: [128, T, 76] BF16 SBUF, the wide routed all-reduce (``_wide_routed_reduce``). With ``shared``:
            the [128, T, 28] routed latent (the view of the ``reduced`` routed all-reduce).
        moe_out_proj: [76, 7168] BF16 HBM in the column layout (``COLUMN_K_TILES``).
        moe_vectors: FP32 HBM [1, C], the latent norm weights at columns [norm_offset, norm_offset + 3584).
        norm_offset: First column of the latent norm weights (``_column_norm_offset``).
        name: Instruction name prefix of the layer.
        shared: optional [128, T, 48] BF16 SBUF, the shared gather of ``column_shared_ag`` (``shared_gather_tiles`` ``shared_ag_dst``):
            the shared slots of the wide routed tile, which ``routed_wide`` then does not hold.
    """
    tokens = routed_wide.shape[1]
    weights_sb = nl.ndarray((_PMAX, COLUMN_K_TILES, _H_TILES), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=weights_sb,
        src=moe_out_proj.reshape((_PMAX, COLUMN_K_TILES, _H_TILES)),
        priority=_CRITICAL_DMA_PRIORITY,
        name=f"{name}_out_proj",
    )
    norm_weight = nl.ndarray((WORLD_SIZE, LATENT_SHARD), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=norm_weight,
        src=moe_vectors.ap(pattern=[[LATENT_SHARD, WORLD_SIZE], [1, LATENT_SHARD]], offset=norm_offset),
        priority=_CRITICAL_DMA_PRIORITY,
        name=f"{name}_norm_weight",
    )
    out_psum = _psum_bank(tokens, (_H_TILES,), nl.float32)
    for k_tile in range(WIDE_SHARED_COLS[0], WIDE_SHARED_COLS[1]):
        nisa.nc_matmul(
            dst=out_psum,
            stationary=routed_wide[:, :, k_tile] if shared is None else shared[:, :, k_tile - WIDE_SHARED_COLS[0]],
            moving=weights_sb[:, k_tile, :],
            accumulate=k_tile > WIDE_SHARED_COLS[0],
        )
    latent = routed_wide[0:WORLD_SIZE, 0:tokens, 0:LATENT_SHARD] if shared is None else routed_wide
    normed = _latent_norm_columns(latent, norm_weight, f"{name}_norm")
    for k_tile in range(LATENT_SHARD):
        nisa.nc_matmul(dst=out_psum, stationary=normed[:, :, k_tile], moving=weights_sb[:, k_tile, :], accumulate=True)
    out_rows = nl.ndarray((tokens, _H_TILES), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=out_rows, src=out_psum)
    return out_rows


def _all_gather_select_rows(
    out_rows: nl.NkiTensor,
    row_select: nl.NkiTensor,
    replica_tokens: int,
    world_group: ReplicaGroup,
    name: str,
    sources: dict,
) -> nl.NkiTensor:
    """All-gather the hidden columns of each rank ([T, 56], ``_column_out_proj``) over the world and keep the B tokens
    of this replica: the output of ``_all_reduce_select_rows`` for ``column``, [128, B, 56] BF16 SBUF.

    The HBM gather stacks the ranks on dim 0, so row g of the [128, T * 56] destination holds the hidden columns
    128 j + g of every token (column 56 t + j): partition g of the token-major hT layout of the all-reduce. The gather
    is the gate of the next body (``release_weight_gate``), and its load the consumer gate (``release_gate_consumer``).
    """
    tokens = out_rows.shape[0]
    width = tokens * _H_TILES
    src_hbm, dst_hbm = _hbm_pair(sources, "outg", (1, width), (WORLD_SIZE, width), nl.bfloat16, f"{name}_gather")
    nisa.dma_copy(
        dst=src_hbm.reshape((tokens, _H_TILES)), src=out_rows, priority=_CRITICAL_DMA_PRIORITY, name=f"{name}_store"
    )
    release_weight_gate(
        ncc.all_gather(
            srcs=[src_hbm], dsts=[dst_hbm], replica_group=world_group, collective_dim=0, name=f"{name}_gather"
        )
    )
    gathered = _tile(sources, "out_dst", (_PMAX, width), nl.bfloat16)
    release_gate_consumer(
        nisa.dma_copy(dst=gathered, src=dst_hbm, priority=_CRITICAL_DMA_PRIORITY, name=f"{name}_load")
    )
    return _select_replica_rows(gathered, row_select, replica_tokens)


# The MoE options (``moe_options_in_use``): routed kernels, router forms and types.


def _validate_fp16_router(x_hT: nl.NkiTensor, x16_hT: nl.NkiTensor, router_weight: nl.NkiTensor, router_dtype: str):
    """Check the FP16 router inputs: the FP16 hidden copy and the rank's FP16 router rows."""
    kernel_assert(
        x16_hT is not None and tuple(x16_hT.shape) == tuple(x_hT.shape) and x16_hT.dtype == nl.float16,
        f"router_dtype={router_dtype!r} needs x16_hT FP16 {tuple(x_hT.shape)}",
    )
    experts = DP_PEER_EXPERTS if router_dtype == "fp16_local" else LOCAL_EXPERTS
    weight_shape = (_PMAX, _H_TILES, experts)
    kernel_assert(
        router_weight is not None and tuple(router_weight.shape) == weight_shape and router_weight.dtype == nl.float16,
        f"router_dtype={router_dtype!r} needs router_weight FP16 {weight_shape}",
    )


def _validate_base_fp16_router(
    x_hT: nl.NkiTensor, x16_hT: nl.NkiTensor, router_weight: nl.NkiTensor, router_dtype: str, dp_size: int, sources
):
    """Check the FP16 router of the base MoE body: ``fp16_local`` at DP4 with the DP x gather (``BASE_ROUTER_DTYPES``)."""
    kernel_assert(
        dp_size == DP_SIZE, f"router_dtype {router_dtype!r} in the base MoE body needs the DP{DP_SIZE} layout"
    )
    kernel_assert(
        (sources or {}).get("x_token_world_src_hbm") is None or "x16_token_world_src_hbm" in sources,
        f"router_dtype {router_dtype!r} with attn_rs_moe_input needs the FP16 token rows of the world token gather",
    )
    _validate_fp16_router(x_hT, x16_hT, router_weight, router_dtype)


def _validate_sel_load_experts(
    expert_gate_up_packed: nl.NkiTensor,
    expert_down_weights: nl.NkiTensor,
    expert_down_weights_scale: nl.NkiTensor,
    stacks: bool = False,
):
    """Check the TP128 expert slices of the sel_load kernel (``SEL_LOAD_WEIGHT_NAMES``).

    With ``stacks``, each tensor may also be the stack of all MoE layers ([L, 896, ...], see ``expert_offset``).
    """

    def matches(tensor, shape):
        dims = tuple(tensor.shape)
        return dims == shape or (stacks and len(dims) == len(shape) + 1 and dims[1:] == shape)

    packed_shape = (NUM_EXPERTS, _PMAX, SEL_LOAD_ROW_BYTES)
    kernel_assert(
        matches(expert_gate_up_packed, packed_shape) and expert_gate_up_packed.dtype == nl.uint8,
        f"expert_gate_up_packed must be UINT8 {packed_shape}, got {expert_gate_up_packed.dtype} {expert_gate_up_packed.shape}",
    )
    down_shape = (NUM_EXPERTS, SEL_LOAD_DOWN_ROWS, LATENT_SIZE)
    compact_shape = (NUM_EXPERTS, SEL_LOAD_COMPACT_DOWN_ROWS, LATENT_SIZE)
    kernel_assert(
        matches(expert_down_weights, down_shape) or matches(expert_down_weights, compact_shape),
        f"sel_load expert_down_weights must be {down_shape} or the compact {compact_shape}, got {expert_down_weights.shape}",
    )
    scale_shape = (NUM_EXPERTS, LATENT_SIZE)
    group_shape = (NUM_EXPERTS, 2, LATENT_SIZE)  # two down scale groups (sel_load_down_groups)
    kernel_assert(
        (matches(expert_down_weights_scale, scale_shape) or matches(expert_down_weights_scale, group_shape))
        and expert_down_weights_scale.dtype == nl.uint8,
        f"sel_load expert_down_weights_scale must be UINT8 {scale_shape} or {group_shape}, got "
        f"{expert_down_weights_scale.dtype} {expert_down_weights_scale.shape}",
    )


def sel_load_down_groups(expert_gate_up_packed: nl.NkiTensor, expert_down_weights_scale: nl.NkiTensor) -> bool:
    """Return True if the sel_load down scales have two scale groups ([..., 896, 2, 3584]).

    The two groups give each down row of a TP128 slice the scale of its own checkpoint block, so the down weights are
    the checkpoint values. The body then runs its down projection with fp32 PSUM (one rounding of each output).
    """
    return len(expert_down_weights_scale.shape) == len(expert_gate_up_packed.shape)


def down_group_mask(rank_info: nl.NkiTensor, name: str, sb=AUTO_SBUF) -> nl.NkiTensor:
    """Return the BF16 [128, 2] down scale group mask of this rank (``build_down_group_mask``), built once per kernel.

    Like ``rank_selectors``, the first call builds the tile and the later calls reuse it. A kernel with an explicit
    region builds it first with a heap allocator ``sb``.
    """
    mask = getattr(rank_info, "_k3_down_group_mask", None)
    if mask is None:
        mask = sb.tile((_PMAX, 2), nl.bfloat16, f"{name}_mask")
        build_down_group_mask(
            rank_info, EXPERT_SHARD, mask, name, lambda shape, dtype, tile: sb.tile(shape, dtype, tile)
        )
        object.__setattr__(rank_info, "_k3_down_group_mask", mask)
    return mask


def _router_logits(x_tokens_hT: nl.NkiTensor, in_proj_sb: nl.NkiTensor, router) -> nl.NkiTensor:
    """Return the [T, 7] FP32 PSUM router logits of the rank's experts.

    ``router`` is None (BF16: the router columns of the in-projection), ``(x16_tokens_hT, router_w16_sb)``
    (FP16: the FP16 hidden copy times the FP16 router rows) or ``LocalRouterLogits`` (``fp16_local``: the logits).
    """
    if isinstance(router, LocalRouterLogits):
        return router.logits
    tokens = x_tokens_hT.shape[2]
    router_psum = _psum_bank(tokens, (LOCAL_EXPERTS,), nl.float32)
    for h_tile_idx in range(_H_TILES):
        if router is None:
            stationary = x_tokens_hT[:, h_tile_idx, :]
            moving = in_proj_sb[:, h_tile_idx, ROUTER_COLS[0] : ROUTER_COLS[1]]
        else:
            stationary = router[0][:, h_tile_idx, :]
            moving = router[1][:, h_tile_idx, :]
        nisa.nc_matmul(dst=router_psum, stationary=stationary, moving=moving)
    return router_psum


def _routing(router: RouterScores, routed_kernel: str, routing_form: str = "base", tiles=AUTO_SBUF):
    """Return the routing input of the routed kernel: (index, affinities, index_done) for the TP128 kernels
    (``_top16_routing``; ``tiles`` places its outputs), else the local affinities."""
    if routed_kernel in TP128_KERNELS:
        return _top16_routing(router, routing_form, tiles)
    return _routed_local_affinities(router)


def _gathered_latent(gathered_words: nl.NkiTensor, tokens: int, after=None) -> nl.NkiTensor:
    """Return the gathered latent [T, L] BF16 in natural column order from the gathered [T, 128, w] FP32 words.

    Words 0..13 of block g hold latent columns [28 g, 28 g + 28). Only the Vector engine reads the gather
    output: the inlined expert kernel reuses its SBUF, and its first Vector instruction (the accumulator
    memset) can overwrite it while a Scalar or GpSimd copy still reads it. On the Vector engine the copy
    and that memset run in program order. A strided SBUF-to-SBUF DMA of 56-byte rows is slower. With ``after`` (an
    instruction), the copy runs after it on the Vector engine.
    """
    latent_sb = nl.ndarray((tokens, WORLD_SIZE, LATENT_SHARD), dtype=nl.bfloat16, buffer=nl.sbuf)
    if isinstance(gathered_words, BlockWords):
        # index_first_block: the latent words of each shard in the natural shard order, read from the block.
        src = _block_view(gathered_words, _GATHER_LATENT_WORDS[0], LATENT_SHARD // 2).view(nl.bfloat16)
        copy = nisa.tensor_copy(
            dst=latent_sb.reshape((tokens, DP_SIZE, TP_SIZE, LATENT_SHARD)), src=src, engine=nisa.engine.vector
        )
    else:
        copy = nisa.tensor_copy(
            dst=latent_sb,
            src=gathered_words[:, :, _GATHER_LATENT_WORDS[0] : _GATHER_LATENT_WORDS[1]].view(nl.bfloat16),
            engine=nisa.engine.vector,
        )
    if after is not None:
        _instruction(copy).depends_on(after, kind=nisa.dependency.runtime)
    return latent_sb.reshape((tokens, LATENT_SIZE))


def _fused_in_proj(x_tokens_hT: nl.NkiTensor, in_proj_sb, name: str, cols: int = MOE_IN_PROJ_COLS) -> nl.NkiTensor:
    """Return the FP32 PSUM ``[T, cols]`` of the fused in-projection: the BF16 ``in_proj_sb``, or the MXFP8 one
    (``MxInProj``: the router columns stay unwritten, the FP16 router gives the logits). ``cols`` is 131, or the 135
    columns of the BF16 block layout (``latent_gather_mx``, ``MX_IN_PROJ_COLS``); the MXFP8 form takes 131 only."""
    if isinstance(in_proj_sb, MxInProj):
        kernel_assert(cols == MOE_IN_PROJ_COLS, f"the MXFP8 in-projection has {MOE_IN_PROJ_COLS} columns, got {cols}")
        return _mx_in_proj_psum(x_tokens_hT, in_proj_sb, MOE_IN_PROJ_COLS, ROUTER_COLS[1], f"{name}_in")
    in_psum = _psum_bank(x_tokens_hT.shape[2], (cols,), nl.float32)
    for h_tile_idx in range(_H_TILES):
        nisa.nc_matmul(
            dst=in_psum,
            stationary=x_tokens_hT[:, h_tile_idx, :],
            moving=in_proj_sb[:, h_tile_idx, :],
        )
    return in_psum


def _fused_router_and_latent(
    x_tokens_hT: nl.NkiTensor,
    in_proj_sb: nl.NkiTensor,
    local_bias_sb: nl.NkiTensor,
    world_group: ReplicaGroup,
    name: str,
    sb,
    routed_kernel: str,
    router=None,
    sources: dict = None,
    routing_form: str = "base",
    shared_rows: bool = False,
    latent_pe: bool = False,
):
    """Stages 1 and 2 with one world gather (router form ``fused``).

    One [T, 131] in-projection gives the router logits of the rank's 7 experts, its latent shard and its
    shared gate/up shards. The world gather carries the latent and the router values together. With an FP16
    router (``router``, see ``_router_logits``), the logits come from their own FP16 matmul.

    Returns:
        ``(routing, latent, out_proj_input)``: the routing input of the routed kernel (``_routing``), the
        gathered latent (the [T, L] BF16 copy, or for sel_load the gathered [T, 128, w] FP32 words that
        ``_gathered_latent`` copies later), and the [76, T] out-projection input (``_shared_activation_T``; with
        ``shared_rows``, the [T, 48] rows of ``_shared_activation_rows``).

    ``latent_pe`` (sel_load option ``TP128_LATENT_PE_OPTION``, with a rank-row routing form of ``_PE32_FORMS`` and the
    HBM world gather): the latent words also come from the rank rows (``_latent_words_from_rank_rows``), and the
    strided load of the gather is not issued.
    """
    tokens = x_tokens_hT.shape[2]
    # Fused in-projection: [T, 131] = X @ moe_in_proj, the tokens as the stationary operand.
    in_psum = _fused_in_proj(x_tokens_hT, in_proj_sb, name)

    # World gather source: the BF16 latent shard packed in FP32 words, and the FP32 router scores
    # and selection values of the local experts.
    gather_src = _tile(sources, "gather", (tokens, _GATHER_COLS), nl.float32)
    nisa.tensor_copy(
        dst=gather_src[:, _GATHER_LATENT_WORDS[0] : _GATHER_LATENT_WORDS[1]].view(nl.bfloat16),
        src=in_psum[:, LATENT_COLS[0] : LATENT_COLS[1]],
    )
    logits = (
        in_psum[:, ROUTER_COLS[0] : ROUTER_COLS[1]] if router is None else _router_logits(x_tokens_hT, None, router)
    )
    local_scores = gather_src[:, _GATHER_SCORE_COLS[0] : _GATHER_SCORE_COLS[1]]
    nisa.activation(dst=local_scores, op=nl.sigmoid, data=logits)
    local_selection = gather_src[:, _GATHER_SELECTION_COLS[0] : _GATHER_SELECTION_COLS[1]]
    nisa.tensor_tensor(dst=local_selection, data1=local_scores, data2=local_bias_sb, op=nl.add)
    rank_rows = None
    if _hbm_site(sources, "gather"):
        pe32 = routing_form in _PE32_FORMS
        kernel_assert(
            not latent_pe or (pe32 and routed_kernel == "sel_load"),
            f"sel_load option {TP128_LATENT_PE_OPTION} needs the sel_load kernel and a routing form of {_PE32_FORMS}",
        )
        gather_dst = _all_gather_hbm(
            gather_src, world_group, WORLD_SIZE, f"{name}_latent", sources, "gather", pe32, strided=not latent_pe
        )
        if pe32:
            gather_dst, rank_rows = gather_dst
    else:
        kernel_assert(not latent_pe, f"sel_load option {TP128_LATENT_PE_OPTION} needs the HBM world gather")
        kernel_assert(routing_form not in _PE32_FORMS, f"routing_form {routing_form!r} needs the HBM world gather")
        gather_dst = _tile(sources, "gather_dst", (tokens, WORLD_SIZE * _GATHER_COLS), nl.float32)
        ncc.all_gather(
            srcs=[gather_src],
            dsts=[gather_dst],
            replica_group=world_group,
            collective_dim=1,
            name=f"{name}_latent_gather",
        )

    # Out-projection input [76, T]: rows 28..75 get the shared-expert activation now, and rows
    # 0..27 get the normalized routed latent after the experts. shared_rows: the [T, 48] activation rows instead.
    out_proj_input = _shared_activation_rows(in_psum, sb) if shared_rows else _shared_activation_T(in_psum, sb)
    if latent_pe:
        # The router values first (the routing waits for them), then the latent words from the same rank rows.
        scores, selection = _router_from_rank_rows(rank_rows, tokens, routing_form == "index_first_pe32_psum")
        router = RouterScores(
            scores=scores, selection=selection, local_scores=local_scores, local_selection=local_selection
        )
        latent = _latent_words_from_rank_rows(rank_rows, tokens)
        return _routing(router, routed_kernel, routing_form), latent, out_proj_input
    gathered = _by_rank(gather_dst, _GATHER_COLS)
    # The EP128 kernels copy the latent before the routing (the program order of the base body). sel_load copies
    # it after the routing and the start of its weight loads.
    latent = gathered if routed_kernel == "sel_load" else _gathered_latent(gathered, tokens)
    if rank_rows is None:
        scores = gathered[:, :, _GATHER_SCORE_COLS[0] : _GATHER_SCORE_COLS[1]]
        selection = gathered[:, :, _GATHER_SELECTION_COLS[0] : _GATHER_SELECTION_COLS[1]]
    else:
        scores, selection = _router_from_rank_rows(rank_rows, tokens, routing_form == "index_first_pe32_psum")
    router = RouterScores(
        scores=scores, selection=selection, local_scores=local_scores, local_selection=local_selection
    )
    return _routing(router, routed_kernel, routing_form), latent, out_proj_input


def _router_from_rank_rows(rank_rows: nl.NkiTensor, tokens: int, selection_psum: bool = False) -> tuple:
    """Return the gathered (scores, selection) [T, 128, 7] FP32 views from the rank rows of the HBM world gather.

    ``rank_rows`` is SBUF [128, T * 28]: rank r on partition r, token t at columns [28 t, 28 t + 28). Each FP32 word of
    the router values moves to [T, 128] in PSUM with a TensorE transpose (bit-accurate on trn3), the selection words
    first: word e of a key at PSUM columns [128 e, 128 e + 128) of one [T, 7 * 128] tile (two banks; each transpose
    stays in one bank). Two copies, one for each half of the ranks, read the tile in the (rank, expert) order and write
    the rank-major [T, 128, 7] tiles (selection: the Scalar and the Vector engine in parallel; scores: the Scalar
    engine). The routing then reads the values in the positions and the unit-stride layout of the strided load.

    With ``selection_psum`` (form ``index_first_pe32_psum``), the scores transposes follow the selection transposes on
    TensorE (schedule edge), and the selection view is the PSUM tile itself in the rank-major order (the view of the
    copies; no copy). The routing reads the same values in the same positions. An instruction of the routing reads at
    most one PSUM operand: the selected-scores op reads the selection from PSUM and the scores from SBUF.
    """
    words = rank_rows.reshape((WORLD_SIZE, tokens, _GATHER_COLS))
    half = WORLD_SIZE // 2
    tiles = {}
    last_selection = None
    for key, (first, end), engines in (
        ("selection", _GATHER_SELECTION_COLS, (nisa.engine.scalar, nisa.engine.vector)),
        ("scores", _GATHER_SCORE_COLS, (nisa.engine.scalar, nisa.engine.scalar)),
    ):
        psum = nl.ndarray((tokens, LOCAL_EXPERTS * WORLD_SIZE), dtype=nl.float32, buffer=nl.psum)
        for j, word in enumerate(range(first, end)):
            transpose = nisa.nc_transpose(
                dst=psum[0:tokens, j * WORLD_SIZE : (j + 1) * WORLD_SIZE], data=words[0:WORLD_SIZE, 0:tokens, word]
            )
            if last_selection is not None and j == 0:
                _instruction(transpose).depends_on(_instruction(last_selection), kind=nisa.dependency.schedule)
        if selection_psum and key == "selection":
            # [t, r, e] = psum[t, 128 e + r]
            tiles[key] = psum.ap(
                pattern=[[LOCAL_EXPERTS * WORLD_SIZE, tokens], [1, WORLD_SIZE], [WORLD_SIZE, LOCAL_EXPERTS]], offset=0
            )
            last_selection = transpose
            continue
        rank_major = nl.ndarray((tokens, WORLD_SIZE, LOCAL_EXPERTS), dtype=nl.float32, buffer=nl.sbuf)
        for part, engine in enumerate(engines):
            nisa.tensor_copy(
                dst=rank_major[0:tokens, part * half : (part + 1) * half, :],
                src=psum.ap(
                    pattern=[[LOCAL_EXPERTS * WORLD_SIZE, tokens], [1, half], [WORLD_SIZE, LOCAL_EXPERTS]],
                    offset=part * half,
                ),
                engine=engine,
            )
        tiles[key] = rank_major
    return tiles["scores"], tiles["selection"]


def _validate_dp_local(
    routed_kernel: str, router_mode: str, router_dtype: str, dp_size: int, sources: dict, peer_in_proj: nl.NkiTensor
):
    """Check the conditions of ``in_proj_form`` ``dp_local`` (``IN_PROJ_FORMS``)."""
    kernel_assert(routed_kernel in TP128_KERNELS, "in_proj_form 'dp_local' needs a TP128 routed kernel")
    kernel_assert(
        router_mode == "fused" and router_dtype == "bf16", "in_proj_form 'dp_local' needs the fused BF16 router"
    )
    kernel_assert(dp_size == DP_SIZE, f"in_proj_form 'dp_local' needs the DP{DP_SIZE} layout, got dp_size {dp_size}")
    kernel_assert(
        _hbm_site(sources, "gather") and _hbm_site(sources, "x"),
        "in_proj_form 'dp_local' needs the HBM world and x gathers (the MLA loop path sites)",
    )
    kernel_assert(
        peer_in_proj is not None
        and tuple(peer_in_proj.shape) == (_PMAX, _H_TILES, PEER_IN_PROJ_COLS)
        and peer_in_proj.dtype == nl.bfloat16,
        f"in_proj_form 'dp_local' needs BF16 peer_in_proj [{_PMAX}, {_H_TILES}, {PEER_IN_PROJ_COLS}], got "
        f"{None if peer_in_proj is None else (peer_in_proj.dtype, peer_in_proj.shape)}",
    )


def _dp_local_gather_source(x_hT: nl.NkiTensor, peer_sb: nl.NkiTensor, peer_bias_sb: nl.NkiTensor) -> nl.NkiTensor:
    """Return the world gather rows of the own tokens for the 4 DP peer shards (``in_proj_form`` ``dp_local``).

    Args:
        x_hT: [128, B, 56] BF16 SBUF, the normed hidden of the own replica.
        peer_sb: [128, 56, 140] BF16 SBUF, the peer in-projection (``PEER_IN_PROJ_COLS``).
        peer_bias_sb: [B, 28] FP32 SBUF, the router biases of the 28 peer experts on each own token.

    Returns:
        [B, 4, 28] FP32 SBUF: row (b, d) is the gather row of ``_fused_router_and_latent`` that rank 32 d + t sends for
        token b of this replica: the BF16 latent shard of shard 32 d + t in words 0..13, the scores of its 7 experts,
        then their selection values. The products, the accumulation order and the instructions per value are those of
        ``_fused_router_and_latent``, so each value has its bits.
    """
    replica_tokens = x_hT.shape[1]
    peer_psum = _psum_bank(replica_tokens, (PEER_IN_PROJ_COLS,), nl.float32)
    for h_tile_idx in range(_H_TILES):
        nisa.nc_matmul(dst=peer_psum, stationary=x_hT[:, :, h_tile_idx], moving=peer_sb[:, h_tile_idx, :])
    gather_src = nl.ndarray((replica_tokens, DP_SIZE, _GATHER_COLS), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(
        dst=gather_src[:, :, _GATHER_LATENT_WORDS[0] : _GATHER_LATENT_WORDS[1]].view(nl.bfloat16),
        src=peer_psum[:, PEER_LATENT_COLS[0] : PEER_LATENT_COLS[1]].reshape_dim(1, (DP_SIZE, LATENT_SHARD)),
    )
    scores = gather_src[:, :, _GATHER_SCORE_COLS[0] : _GATHER_SCORE_COLS[1]]
    nisa.activation(
        dst=scores,
        op=nl.sigmoid,
        data=peer_psum[:, PEER_ROUTER_COLS[0] : PEER_ROUTER_COLS[1]].reshape_dim(1, (DP_SIZE, LOCAL_EXPERTS)),
    )
    nisa.tensor_tensor(
        dst=gather_src[:, :, _GATHER_SELECTION_COLS[0] : _GATHER_SELECTION_COLS[1]],
        data1=scores,
        data2=peer_bias_sb.reshape_dim(1, (DP_SIZE, LOCAL_EXPERTS)),
        op=nl.add,
    )
    return gather_src


def dp_local_rank_rows(gather_dst_hbm: nl.NkiTensor, replica_tokens: int, name: str, priority=None) -> tuple:
    """Load the ``dp_local`` HBM world gather as the rank rows of ``gathered``: SBUF [128, T * 28], shard g on
    partition g, token r * B + b at columns [28 (r B + b), 28 (r B + b) + 28).

    ``gather_dst_hbm`` [128 * T, 28]: row T k + B d + b holds token b of replica k // 32 for shard 32 d + k % 32 (rank
    k sends the rows (d, b) of ``_dp_local_gather_source``). One DMA for each DP peer d: 32 partitions x 4 replicas of
    B * 28 contiguous words.

    Returns:
        ``(rank_rows, last_dma)``.
    """
    tokens = DP_SIZE * replica_tokens
    row_words = tokens * _GATHER_COLS
    block = replica_tokens * _GATHER_COLS
    rank_rows = nl.ndarray((WORLD_SIZE, row_words), dtype=nl.float32, buffer=nl.sbuf)
    last = None
    for peer in range(DP_SIZE):
        last = nisa.dma_copy(
            dst=rank_rows[peer * TP_SIZE : (peer + 1) * TP_SIZE, :].reshape_dim(1, (DP_SIZE, block)),
            src=gather_dst_hbm.ap(
                pattern=[[row_words, TP_SIZE], [TP_SIZE * row_words, DP_SIZE], [1, block]], offset=peer * block
            ),
            **({} if priority is None else {"priority": priority}),
            name=f"{name}_rank_rows_d{peer}",
        )
    return rank_rows, last


def dp_local_route_rows(gather_dst_hbm: nl.NkiTensor, name: str, priority=None) -> tuple:
    """Load the router words of the ``dp_local`` HBM world gather token-major at one token per DP replica (routing form
    ``index_first_rows``): SBUF [4, 128, 14], token r (replica r) on partition r, shard g = 32 d + t at [:, g, :] (the
    natural shard order of ``dp_local_token_rows``), its 7 scores in words 0..6 and its 7 selection values in 7..13.

    ``gather_dst_hbm`` [128 * 4, 28]: row 4 k + d holds the token of replica k // 32 for shard 32 d + k % 32. One DMA of
    4 partitions x (4 peers x 32 TP ranks) elements of 14 words (56 B), on the hardware DGE: with the default mode the
    compiler puts this DMA on a static queue whose data one DMA engine moves.

    Returns:
        ``(route, dma)``.
    """
    tokens = DP_SIZE
    row_words = tokens * _GATHER_COLS
    words = _GATHER_SELECTION_COLS[1] - _GATHER_SCORE_COLS[0]
    route = nl.ndarray((tokens, WORLD_SIZE, words), dtype=nl.float32, buffer=nl.sbuf)
    dma = nisa.dma_copy(
        dst=route.reshape_dim(1, (DP_SIZE, TP_SIZE)),
        src=gather_dst_hbm.ap(
            pattern=[[TP_SIZE * row_words, DP_SIZE], [_GATHER_COLS, DP_SIZE], [row_words, TP_SIZE], [1, words]],
            offset=_GATHER_SCORE_COLS[0],
        ),
        dge_mode=nisa.dge_mode.hwdge,
        **({} if priority is None else {"priority": priority}),
        name=f"{name}_route_rows",
    )
    return route, dma


class BlockWords(NamedTuple):
    """The gathered words of ``index_first_block``: ``block`` [4, 128 * 28] FP32 SBUF in the HBM shard order (token r
    on partition r, shard row 4 t + d at words [28 (4 t + d), 28 (4 t + d) + 28) for shard g = 32 d + t)."""

    block: nl.NkiTensor


def dp_local_token_block(gather_dst_hbm: nl.NkiTensor, name: str, priority=None) -> tuple:
    """Load the ``dp_local`` HBM world gather at one token per DP replica (routing form ``index_first_block``) as one
    contiguous block per token: SBUF [4, 128 * 28] FP32, token r on partition r, the rows [128 r, 128 r + 128) of
    ``gather_dst_hbm`` [128 * 4, 28] in their HBM order (row 128 r + 4 t + d holds shard 32 d + t). One DMA of 4
    elements of 14,336 bytes, on the hardware DGE (as ``dp_local_route_rows``).

    Returns:
        ``(BlockWords, dma)``.
    """
    tokens = DP_SIZE
    row_words = WORLD_SIZE * _GATHER_COLS
    block = nl.ndarray((tokens, row_words), dtype=nl.float32, buffer=nl.sbuf)
    dma = nisa.dma_copy(
        dst=block,
        src=gather_dst_hbm.reshape((tokens, row_words)),
        dge_mode=nisa.dge_mode.hwdge,
        **({} if priority is None else {"priority": priority}),
        name=f"{name}_token_block",
    )
    return BlockWords(block), dma


def _block_view(words: BlockWords, first: int, count: int) -> nl.NkiTensor:
    """Return words [first, first + count) of every shard of ``words`` (``BlockWords``) as a [4, 4, 32, count] FP32 view
    in the natural shard order g = 32 d + t (d, then t): the order of the strided load of ``dp_local_token_rows``."""
    row_words = WORLD_SIZE * _GATHER_COLS
    return words.block.ap(
        pattern=[[row_words, DP_SIZE], [_GATHER_COLS, DP_SIZE], [DP_SIZE * _GATHER_COLS, TP_SIZE], [1, count]],
        offset=first,
    )


def block_router_words(words: BlockWords) -> tuple:
    """Return the (scores, selection) [4, 128, 7] FP32 tiles of ``words`` (``BlockWords``) in the natural shard order:
    one Vector copy each, the selection values first (the routing reads them first). The routing ops take at most 2
    free dimensions, so they cannot read the block views; only the Vector engine reads the gather output (as
    ``_gathered_latent``)."""
    tokens = DP_SIZE
    tiles = {}
    for key, (col, _) in (("selection", _GATHER_SELECTION_COLS), ("scores", _GATHER_SCORE_COLS)):
        tile = nl.ndarray((tokens, WORLD_SIZE, LOCAL_EXPERTS), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(
            dst=tile.reshape((tokens, DP_SIZE, TP_SIZE, LOCAL_EXPERTS)),
            src=_block_view(words, col, LOCAL_EXPERTS),
            engine=nisa.engine.vector,
        )
        tiles[key] = tile
    return tiles["scores"], tiles["selection"]


def dp_local_token_rows(gather_dst_hbm: nl.NkiTensor, replica_tokens: int, name: str, after=None) -> nl.NkiTensor:
    """Load the ``dp_local`` HBM world gather token-major, as the strided load of ``gathered``: SBUF [T, 128, 28], token
    r * B + b on partition r B + b, shard g = 32 d + t at [:, g, :] (the natural shard order).

    One DMA for each replica r: B partitions x (4 peers x 32 TP ranks) rows of 28 words. With ``after`` (an
    instruction), the DMAs follow it in the schedule (the rank rows go first on the queue).
    """
    tokens = DP_SIZE * replica_tokens
    row_words = tokens * _GATHER_COLS
    gathered = nl.ndarray((tokens, WORLD_SIZE, _GATHER_COLS), dtype=nl.float32, buffer=nl.sbuf)
    for replica in range(DP_SIZE):
        rows = gathered[replica * replica_tokens : (replica + 1) * replica_tokens]
        load = nisa.dma_copy(
            dst=rows.reshape_dim(1, (DP_SIZE, TP_SIZE)),
            src=gather_dst_hbm.ap(
                pattern=[
                    [_GATHER_COLS, replica_tokens],
                    [replica_tokens * _GATHER_COLS, DP_SIZE],
                    [row_words, TP_SIZE],
                    [1, _GATHER_COLS],
                ],
                offset=replica * TP_SIZE * row_words,
            ),
            name=f"{name}_token_rows_r{replica}",
        )
        if after is not None:
            _instruction(load).depends_on(_instruction(after), kind=nisa.dependency.schedule)
    return gathered


def _dp_local_router_and_latent(
    x_hT: nl.NkiTensor,
    peer_sb: nl.NkiTensor,
    peer_bias_sb: nl.NkiTensor,
    world_group: ReplicaGroup,
    name: str,
    routed_kernel: str,
    sources: dict,
    routing_form: str,
    rank_rows_done: list = None,
    route_done: list = None,
    routing_tiles=AUTO_SBUF,
):
    """Stages 1 and 2 of ``in_proj_form`` ``dp_local``: the peer in-projection of the own tokens and the HBM world gather.

    ``rank_rows_done`` (a list) gets the last rank-row load (a ``routing_form`` of ``_PE32_FORMS``), or the router-word
    load (``index_first_rows``), if there is one. ``route_done`` (a list) gets the router-word load of
    ``index_first_rows``. ``routing_tiles`` places the three [T, 16] routing outputs (``_top16_routing`` ``tiles``).

    Returns:
        ``(routing, latent)`` as ``_fused_router_and_latent`` (the token-major gathered [T, 128, 28] words).
    """
    replica_tokens = x_hT.shape[1]
    tokens = DP_SIZE * replica_tokens
    gather_src = _dp_local_gather_source(x_hT, peer_sb, peer_bias_sb)
    src_hbm, dst_hbm = _hbm_pair(
        sources, "gather", (tokens, _GATHER_COLS), (WORLD_SIZE * tokens, _GATHER_COLS), nl.float32, f"{name}_latent"
    )
    call = next_call("moe_hbm")
    # Source row d * B + b: own token b for the peer shard d.
    nisa.dma_copy(
        dst=src_hbm.ap(
            pattern=[[_GATHER_COLS, replica_tokens], [replica_tokens * _GATHER_COLS, DP_SIZE], [1, _GATHER_COLS]],
            offset=0,
        ),
        src=gather_src,
        priority=_CRITICAL_DMA_PRIORITY,
        name=f"{name}_latent_store_{call}",
    )
    ncc.all_gather(
        srcs=[src_hbm], dsts=[dst_hbm], replica_group=world_group, collective_dim=0, name=f"{name}_latent_gather"
    )
    first = None
    if routing_form in _PE32_FORMS:
        rank_rows, first = dp_local_rank_rows(dst_hbm, replica_tokens, name, _CRITICAL_DMA_PRIORITY)
        scores, selection = _router_from_rank_rows(rank_rows, tokens, routing_form == "index_first_pe32_psum")
    elif routing_form == "index_first_rows":
        kernel_assert(replica_tokens == 1, "routing_form 'index_first_rows' needs one token per DP replica")
        route, first = dp_local_route_rows(dst_hbm, name, _CRITICAL_DMA_PRIORITY)
        scores = route[:, :, 0:LOCAL_EXPERTS]
        selection = route[:, :, LOCAL_EXPERTS : 2 * LOCAL_EXPERTS]
        if route_done is not None:
            route_done.append(first)
    elif routing_form == "index_first_block":
        kernel_assert(replica_tokens == 1, "routing_form 'index_first_block' needs one token per DP replica")
        gathered, first = dp_local_token_block(dst_hbm, name, _CRITICAL_DMA_PRIORITY)
        scores, selection = block_router_words(gathered)
        if route_done is not None:
            route_done.append(first)
    if first is not None and rank_rows_done is not None:
        rank_rows_done.append(first)
    if routing_form != "index_first_block":
        gathered = dp_local_token_rows(dst_hbm, replica_tokens, name, after=first)
    if first is None:
        scores = gathered[:, :, _GATHER_SCORE_COLS[0] : _GATHER_SCORE_COLS[1]]
        selection = gathered[:, :, _GATHER_SELECTION_COLS[0] : _GATHER_SELECTION_COLS[1]]
    # The TP128 routing reads only the gathered values (the rank-local views serve the EP128 kernels).
    router = RouterScores(scores=scores, selection=selection, local_scores=None, local_selection=None)
    latent = gathered if routed_kernel == "sel_load" else _gathered_latent(gathered, tokens)
    return _routing(router, routed_kernel, routing_form, routing_tiles), latent


def _dp_local_x_store(x_hT: nl.NkiTensor, name: str, sources: dict) -> tuple:
    """Store the own hidden to the HBM source of the DP x gather (``in_proj_form`` ``dp_local``).

    Returns:
        ``(src_hbm, dst_hbm)``: the HBM pair of the x gather, [128, B * 56] and [4 * 128, B * 56].
    """
    replica_tokens = x_hT.shape[1]
    rows, width = _PMAX, replica_tokens * _H_TILES
    src_hbm, dst_hbm = _hbm_pair(sources, "x", (rows, width), (DP_SIZE * rows, width), nl.bfloat16, f"{name}_x")
    call = next_call("moe_hbm")
    nisa.dma_copy(dst=src_hbm, src=x_hT.reshape((rows, width)), name=f"{name}_x_store_{call}")
    return src_hbm, dst_hbm


def _dp_local_x_gather(
    src_hbm: nl.NkiTensor, dst_hbm: nl.NkiTensor, dp_group: ReplicaGroup, name: str, route_done: list = None
):
    """All-gather the stored own hidden over the DP group (``in_proj_form`` ``dp_local``; the load runs after the
    routed all-reduce trigger, ``dp_local_shared_activation``). Return the gather instruction.

    The caller issues it after the routed kernel. The compiler orders each DMA of an external input that lies between
    a collective and the next one, and does not depend on the first, after that collective's last input, together with
    the producers of that DMA. Issued after the world gather, the x gather would order the routed weight loads (and,
    through their buffer reuse, most of the routed kernel) behind the x store, with one semaphore wait each.
    ``column_shared_ag`` issues it before the routed kernel and schedules the weight stream after it
    (``moe_sel_load_tp_start`` ``stream_after``), so the weight loads depend on it and the compiler skips them.
    With ``route_done`` (the router-word load of ``index_first_rows``) the gather waits for that load, so its packets
    do not share the DMA engines with the router words.
    """
    gather = ncc.all_gather(
        srcs=[src_hbm], dsts=[dst_hbm], replica_group=dp_group, collective_dim=0, name=f"{name}_x_gather"
    )
    if route_done:
        _instruction(gather).depends_on(_instruction(route_done[0]), kind=nisa.dependency.runtime)
    return gather


def dp_local_shared_activation(
    x_gather_hbm: nl.NkiTensor,
    moe_in_proj: nl.NkiTensor,
    replica_tokens: int,
    name: str,
    sb=AUTO_SBUF,
    weight_dma: dict = None,
    after=None,
) -> nl.NkiTensor:
    """Load the DP x gather and the shared in-projection, and return the [76, T] out-projection input of
    ``_shared_activation_T`` (``in_proj_form`` ``dp_local``; ``moe_in_proj`` [128, 56, 96] holds the shared columns).

    The shared columns, their products and accumulation order are those of the gathered in-projection, so the
    activation has its bits. ``weight_dma`` (``nisa.dma_copy`` arguments: the queue) and ``after`` (an instruction
    that the loads follow in the schedule) apply to the shared in-projection loads (``GLUE_DMA_FORMS``
    ``sp_weights``)."""
    shared_psum = _dp_local_shared_in_proj(x_gather_hbm, moe_in_proj, replica_tokens, name, weight_dma, after)
    return _shared_activation_T(shared_psum, sb, col_offset=SHARED_GATE_COLS[0])


def dp_local_shared_rows(
    x_gather_hbm: nl.NkiTensor,
    moe_in_proj: nl.NkiTensor,
    replica_tokens: int,
    name: str,
    sb=AUTO_SBUF,
    weight_dma: dict = None,
    after=None,
    tiles: dict = None,
    compute_after=None,
    copies: list = None,
) -> nl.NkiTensor:
    """Return the [T, 48] shared activation rows of ``_shared_activation_rows`` from the DP x gather and the shared
    in-projection, as ``dp_local_shared_activation`` (``out_proj_form`` ``column_shared_ag``): the same bits.
    ``weight_dma`` and ``after`` as there. ``tiles``: optional SBUF tiles of every step
    (``dp_local_shared_rows_tiles``; None: new tiles, the result from ``sb``). ``compute_after``: optional instruction
    that the in-projection matmuls and the activation follow in the schedule (the first gather index of the routed
    kernel). ``copies``: optional list that gets the PSUM copy instruction of the activation."""
    shared_psum = _dp_local_shared_in_proj(
        x_gather_hbm, moe_in_proj, replica_tokens, name, weight_dma, after, compute_after, tiles
    )
    return _shared_activation_rows(
        shared_psum, sb, col_offset=SHARED_GATE_COLS[0], tiles=tiles, after=compute_after, copies=copies
    )


def dp_local_shared_rows_tiles(replica_tokens: int, sb) -> dict:
    """Return the SBUF tiles of every step of ``dp_local_shared_rows`` from the allocator ``sb``: the DP x rows
    [128, 4, B * 56] and the shared in-projection columns [128, 56, 96] (BF16), the gate and up columns [T, 96] (BF16),
    the three FP32 SiTU terms [T, 48] and the activation rows [T, 48] (BF16). With a region, a caller that runs the
    shared activation inside the routed kernel takes them first, so they keep addresses of their own."""
    tokens = DP_SIZE * replica_tokens
    return {
        "x": sb.tile((_PMAX, DP_SIZE, replica_tokens * _H_TILES), nl.bfloat16, "moe_shared_x"),
        "weights": sb.tile((_PMAX, _H_TILES, SHARED_IN_PROJ_COLS), nl.bfloat16, "moe_shared_in_proj"),
        "gate_up": sb.tile((tokens, 2 * SHARED_SHARD), nl.bfloat16, "moe_shared_gate_up"),
        "situ": tuple(sb.tile((tokens, SHARED_SHARD), nl.float32, f"moe_shared_situ{i}") for i in range(3)),
        "rows": sb.tile((tokens, SHARED_SHARD), nl.bfloat16, "moe_shared_rows"),
    }


def _dp_local_shared_in_proj(
    x_gather_hbm: nl.NkiTensor,
    moe_in_proj: nl.NkiTensor,
    replica_tokens: int,
    name: str,
    weight_dma: dict = None,
    after=None,
    compute_after=None,
    tiles: dict = None,
) -> nl.NkiTensor:
    """Load the DP x gather and the shared in-projection columns, and return the [T, 96] FP32 PSUM shared gate and up
    columns (``dp_local_shared_activation``; ``weight_dma`` and ``after`` as there). With ``compute_after`` (an
    instruction), the first matmul, and so the accumulation, follows it in the schedule. ``tiles``: optional SBUF
    tiles (``dp_local_shared_rows_tiles``) for the x rows and the weights."""
    tokens = DP_SIZE * replica_tokens
    width = replica_tokens * _H_TILES
    if tiles is None:
        x_rows = nl.ndarray((_PMAX, DP_SIZE, width), dtype=nl.bfloat16, buffer=nl.sbuf)
    else:
        x_rows = tiles["x"]
    nisa.dma_copy(
        dst=x_rows,
        src=x_gather_hbm.ap(pattern=[[width, _PMAX], [_PMAX * width, DP_SIZE], [1, width]], offset=0),
        name=f"{name}_x_load",
    )
    x_tokens_hT = x_rows.reshape((_PMAX, tokens, _H_TILES)).permute((0, 2, 1))
    if tiles is None:
        shared_sb = nl.ndarray((_PMAX, _H_TILES, SHARED_IN_PROJ_COLS), dtype=nl.bfloat16, buffer=nl.sbuf)
    else:
        shared_sb = tiles["weights"]
    _load_hT_rows(shared_sb, moe_in_proj, _IN_PROJ_DMA_CHUNKS, f"{name}_shared_in_proj", after, weight_dma)
    shared_psum = _psum_bank(tokens, (SHARED_IN_PROJ_COLS,), nl.float32)
    for h_tile_idx in range(_H_TILES):
        matmul = nisa.nc_matmul(
            dst=shared_psum, stationary=x_tokens_hT[:, h_tile_idx, :], moving=shared_sb[:, h_tile_idx, :]
        )
        if h_tile_idx == 0 and compute_after is not None:
            _instruction(matmul).depends_on(_instruction(compute_after), kind=nisa.dependency.schedule)
    return shared_psum


def _split_router_and_latent(
    x_tokens_hT: nl.NkiTensor,
    in_proj_sb: nl.NkiTensor,
    local_bias_sb: nl.NkiTensor,
    world_group: ReplicaGroup,
    name: str,
    sb,
    routed_kernel: str,
    router=None,
    sources: dict = None,
    routing_form: str = "base",
    shared_rows: bool = False,
):
    """Stage 1 (router) and stage 2 (latent) with their own world gathers (router form ``split``).

    The router columns of the in-projection run first, and their [T, 14] values (7 scores, 7 selection
    values) go to the world gather at once. The latent and shared columns follow on the PE, and the latent
    gather starts before the routing waits for the router gather. So the routing (the top-16 index) is ready
    one gather before the latent, and the routed-expert weight loads that need only the index can run during
    the latent gather.

    Returns:
        ``(routing, latent_words, out_proj_input)``: as ``_fused_router_and_latent``, with the gathered latent as
        the [T, 128, 14] FP32 words, which the caller copies (``_gathered_latent``) after the routed kernel starts.
    """
    tokens = x_tokens_hT.shape[2]
    # Stage 1: router logits [T, 7] of the rank's experts (BF16 in-projection columns, or the FP16 router).
    router_psum = _router_logits(x_tokens_hT, in_proj_sb, router)
    router_src = _tile(sources, "router_src", (tokens, _ROUTER_GATHER_COLS), nl.float32)
    local_scores = router_src[:, 0:LOCAL_EXPERTS]
    nisa.activation(dst=local_scores, op=nl.sigmoid, data=router_psum)
    local_selection = router_src[:, LOCAL_EXPERTS:_ROUTER_GATHER_COLS]
    nisa.tensor_tensor(dst=local_selection, data1=local_scores, data2=local_bias_sb, op=nl.add)
    if _hbm_site(sources, "router"):
        router_dst = _all_gather_hbm(router_src, world_group, WORLD_SIZE, f"{name}_router", sources, "router")
    else:
        router_dst = _tile(sources, "router_dst", (tokens, WORLD_SIZE * _ROUTER_GATHER_COLS), nl.float32)
        ncc.all_gather(
            srcs=[router_src],
            dsts=[router_dst],
            replica_group=world_group,
            collective_dim=1,
            name=f"{name}_router_gather",
        )

    # Stage 2: the latent and shared columns [T, 124], and the latent gather.
    if isinstance(in_proj_sb, MxInProj):
        latent_psum = _mx_in_proj_psum(x_tokens_hT, in_proj_sb, _LATENT_SHARED_COLS, 0, f"{name}_latent")
    else:
        latent_psum = _psum_bank(tokens, (_LATENT_SHARED_COLS,), nl.float32)
        for h_tile_idx in range(_H_TILES):
            nisa.nc_matmul(
                dst=latent_psum,
                stationary=x_tokens_hT[:, h_tile_idx, :],
                moving=in_proj_sb[:, h_tile_idx, ROUTER_COLS[1] : MOE_IN_PROJ_COLS],
            )
    latent_src = _tile(sources, "latent_src", (tokens, _LATENT_GATHER_WORDS), nl.float32)
    nisa.tensor_copy(dst=latent_src.view(nl.bfloat16), src=latent_psum[:, 0:LATENT_SHARD])
    if _hbm_site(sources, "latent"):
        latent_dst = _all_gather_hbm(latent_src, world_group, WORLD_SIZE, f"{name}_latent", sources, "latent")
    else:
        latent_dst = _tile(sources, "latent_dst", (tokens, WORLD_SIZE * _LATENT_GATHER_WORDS), nl.float32)
        ncc.all_gather(
            srcs=[latent_src],
            dsts=[latent_dst],
            replica_group=world_group,
            collective_dim=1,
            name=f"{name}_latent_gather",
        )

    # The shared activation frees the PSUM bank of the in-projection before the routed kernel starts (the
    # selective-load kernel owns all PSUM banks from its start). The routing waits for the router gather only, and
    # the latent copy comes after it in program order, so the Vector engine does not wait for the latent gather first.
    shared = _shared_activation_rows if shared_rows else _shared_activation_T
    out_proj_input = shared(latent_psum, sb, col_offset=ROUTER_COLS[1])
    gathered_router = _by_rank(router_dst, _ROUTER_GATHER_COLS)
    router = RouterScores(
        scores=gathered_router[:, :, 0:LOCAL_EXPERTS],
        selection=gathered_router[:, :, LOCAL_EXPERTS:_ROUTER_GATHER_COLS],
        local_scores=local_scores,
        local_selection=local_selection,
    )
    return (
        _routing(router, routed_kernel, routing_form),
        _by_rank(latent_dst, _LATENT_GATHER_WORDS),
        out_proj_input,
    )


def dynamic_expert_loop(enabled: bool, tokens: int, mla_loop_path: bool, moe_options: bool) -> bool:
    """Return True when the latent MoE layers of a decode NEFF run the dynamic expert loop of ``moe_tkg_decode``.

    The dynamic loop runs only the active local experts, that is, the experts with a nonzero affinity for a
    token. It gives the bits of the static loop. It is an opt-in (``moe_dynamic_loop`` of the decode entries),
    and it applies only when all of these conditions are true:

    - ``tokens <= DYNAMIC_EXPERT_LOOP_MAX_TOKENS`` (B <= 2). With more tokens, more local experts are active, and
      the static loop is faster (trn3, busiest rank, at T = 16).
    - The MLA table is unrolled (no MLA loop path). Beside the explicit SBUF region of the loop path, the
      dynamic loop does not compile: neuronx-cc stops with NCC_EGCA111, because spilling is disabled for kernels.
    - The layer runs the base MoE body (the ``static`` routed kernel and no MoE option, ``moe_options_in_use``).

    Otherwise the static loop runs, and the NEFF is the NEFF without the opt-in.

    Args:
        enabled (bool): the opt-in (``moe_dynamic_loop``).
        tokens (int): T of the MoE call, the tokens of the 4 DP replicas.
        mla_loop_path (bool): the NEFF takes the MLA loop path.
        moe_options (bool): the MoE layers run an option body or have an explicit SBUF region.

    Returns:
        bool: True for the dynamic loop, False for the static loop.

    Example:
        >>> dynamic_expert_loop(True, 4, mla_loop_path=False, moe_options=False)
        True
        >>> dynamic_expert_loop(True, 8, mla_loop_path=False, moe_options=False)
        True
        >>> dynamic_expert_loop(True, 16, mla_loop_path=False, moe_options=False)
        False
    """
    return enabled and tokens <= DYNAMIC_EXPERT_LOOP_MAX_TOKENS and not mla_loop_path and not moe_options


def base_moe_body(routed_options: dict, moe_options: bool = False, dp_moe: bool = False) -> bool:
    """Return True when the latent MoE layers of a decode block run the base body of ``latent_moe_decode``.

    The base body is the ``static`` routed kernel with no MoE option, in the TP128 projections. A
    ``routing_form`` of ``base`` is the routing of that body, so it keeps the base body; any other routed
    option, a MoE option (``moe_options_in_use``) or the DP4 x TP32 body (``tkg_moe_dp``) does not.

    Args:
        routed_options (dict): the routed-kernel arguments that ``decode_block_body`` passes to the MoE body
            (``routed_kernel``, ``routing_form``, ``router_mode`` and the others it sets).
        moe_options (bool): a MoE option is in use (``moe_options_in_use``).
        dp_moe (bool): the MoE body is the DP4 x TP32 body.

    Returns:
        bool: True for the base body.

    Example:
        >>> base_moe_body({"routing_form": "base"})
        True
        >>> base_moe_body({"routing_form": "index_first"})
        False
    """
    options = {key: value for key, value in routed_options.items() if (key, value) != ("routing_form", "base")}
    return not (options or moe_options or dp_moe)


def routed_expert_path(dynamic_experts: bool, active_expert_slots: int, base_body: bool) -> str:
    """Return the expert path of the latent MoE layers of a decode NEFF: ``dynamic``, ``slots`` or ``static``.

    This is the one dispatch point of the paths of ``moe_tkg_decode`` that run only the active local experts. The
    precedence is:

    - the dynamic loop, when ``dynamic_experts`` (``dynamic_expert_loop``: the opt-in, at most
      ``DYNAMIC_EXPERT_LOOP_MAX_TOKENS`` tokens, off the MLA loop path, in the base MoE body);
    - else the active-expert slots, when ``active_expert_slots`` > 0;
    - else the static loop.

    A layer never runs both. The slots need the base MoE body (``base_moe_body``: the ``static`` routed kernel in
    the TP128 projections, with no MoE option such as an FP16 router). With another body, or with more slots than
    local experts, the call fails instead of falling back to another path.

    Args:
        dynamic_experts (bool): the dynamic loop applies (``dynamic_expert_loop``).
        active_expert_slots (int): K, the opt-in slots (``moe_active_expert_slots`` of the decode entries), 0 to
            ``LOCAL_EXPERTS``.
        base_body (bool): the MoE layers run the base body (``base_moe_body``).

    Returns:
        str: ``dynamic``, ``slots`` or ``static``.

    Example:
        >>> routed_expert_path(True, 3, base_body=True)
        'dynamic'
        >>> routed_expert_path(False, 3, base_body=True)
        'slots'
    """
    kernel_assert(
        0 <= active_expert_slots <= LOCAL_EXPERTS,
        f"Expected 0 <= moe_active_expert_slots <= {LOCAL_EXPERTS}, got {active_expert_slots}",
    )
    if dynamic_experts:
        return "dynamic"
    if active_expert_slots:
        kernel_assert(
            base_body,
            "The active-expert slots run only in the base MoE body: the static routed kernel in the TP128 "
            "projections, with no MoE option such as an FP16 router",
        )
        return "slots"
    return "static"


def active_expert_counts(expert_path: str, enabled: bool) -> bool:
    """Return True when the MoE layers store their active-expert counts (``moe_active_expert_counts``).

    The count is a debug output of the active-expert slots (``active_expert_count`` of ``moe_tkg_decode``): the
    number of active local experts of the rank in each call, so a count above K marks a call whose overflow loop
    ran. Only the slots compute it, so asking for the counts on another expert path (``routed_expert_path``: the
    static loop with K = 0, or the dynamic loop, which takes precedence) fails instead of leaving them unwritten.

    Args:
        expert_path (str): the path of the layers (``routed_expert_path``).
        enabled (bool): the counts are asked for.

    Returns:
        bool: ``enabled``, after the check.

    Example:
        >>> active_expert_counts("slots", True)
        True
        >>> active_expert_counts("static", False)
        False
    """
    kernel_assert(
        not enabled or expert_path == "slots",
        f"moe_active_expert_counts reports the active-expert slots, but the MoE layers run the {expert_path} path: "
        "set moe_active_expert_slots > 0 where the dynamic loop does not apply",
    )
    return enabled


def moe_options_in_use(
    routed_kernel: str = "static",
    router_mode: str = None,
    router_dtype: str = "bf16",
    routed_reduce_dtype: str = "bf16",
) -> bool:
    """Return True when a MoE option other than the base body is in use (a kernel, a router form or a type).

    A router of ``BASE_ROUTER_DTYPES`` is no option. These options run ``_routed_latent_moe_decode``. They need an explicit SBUF region, and their collective sites
    add persistent tiles (``routed_collective_tiles``) and HBM pairs (``routed_collective_hbm``).
    """
    return (
        routed_kernel != "static"
        or (router_mode or "fused") != "fused"
        or router_dtype not in BASE_ROUTER_DTYPES
        or routed_reduce_dtype != "bf16"
    )


def routed_collective_tiles(
    replica_tokens: int,
    sb=AUTO_SBUF,
    router_mode: str = "fused",
    router_dtype: str = "bf16",
    routed_reduce_dtype: str = "bf16",
    loop_path: bool = False,
    dp_size: int = DP_SIZE,
) -> dict:
    """Return the persistent SBUF tiles that the MoE options add to ``moe_collective_tiles``, one tile for each site.

    ``split`` adds the SBUF router and latent gathers (on the MLA loop path they go through HBM, see
    ``routed_collective_hbm``). ``fp16`` adds the DP gather of the BF16 and FP16 rows, and ``fp16_local`` the DP gather
    of the BF16 rows with the local router logits (on the MLA loop path their gathers run in HBM, and
    ``fp16_local`` keeps only its [28, B] logit source in SBUF). The caller clears the tiles
    (``decode_block.collective_sources``).
    """
    tokens = dp_size * replica_tokens
    tiles = {}
    if router_mode == "split" and not loop_path:
        tiles["router_src"] = sb.tile((tokens, _ROUTER_GATHER_COLS), nl.float32, "moe_router_src")
        tiles["router_dst"] = sb.tile((tokens, WORLD_SIZE * _ROUTER_GATHER_COLS), nl.float32, "moe_router_dst")
        tiles["latent_src"] = sb.tile((tokens, _LATENT_GATHER_WORDS), nl.float32, "moe_latent_src")
        tiles["latent_dst"] = sb.tile((tokens, WORLD_SIZE * _LATENT_GATHER_WORDS), nl.float32, "moe_latent_dst")
    if router_dtype == "fp16":
        tiles["x16_src"] = sb.tile((_PMAX, replica_tokens, 2, _H_TILES), nl.bfloat16, "moe_x16_src")
        if not loop_path:
            tiles["x16_dst"] = sb.tile((_PMAX, tokens * 2 * _H_TILES), nl.bfloat16, "moe_x16_dst")
    if router_dtype == "fp16_local" and loop_path:
        tiles["xl_logits"] = sb.tile((DP_PEER_EXPERTS, replica_tokens), nl.float32, "moe_xl_logits")
    elif router_dtype == "fp16_local":
        tiles["xl_src"] = sb.tile((_PMAX, replica_tokens, _TOKEN_GATHER_WORDS), nl.bfloat16, "moe_xl_src")
        tiles["xl_dst"] = sb.tile((_PMAX, tokens * _TOKEN_GATHER_WORDS), nl.bfloat16, "moe_xl_dst")
    return tiles


def routed_collective_hbm(
    replica_tokens: int,
    router_mode: str = "fused",
    router_dtype: str = "bf16",
    routed_reduce_dtype: str = "bf16",
    loop_path: bool = False,
    dp_size: int = DP_SIZE,
    reduce_rows: int = None,
    pair_shared: tuple = (),
) -> dict:
    """Return the persistent HBM tensors of the feed-forward collectives with the MoE options, one pair for each site.

    The base sites take the keys of ``moe_collective_hbm`` (the routed and output all-reduces). On the MLA loop path,
    the token gather and the world gathers go through HBM too (``_all_gather_hbm``), so no peer writes SBUF there.
    ``fp32`` adds the FP32 routed all-reduce. Each layer uses the same tensors, so a world collective separates each
    use from the next one; the first use of each tensor is its first collective. ``reduce_rows`` sizes the routed
    and output all-reduces as the verify's (bitwise decode == verify; see ``moe_collective_hbm``).
    ``pair_shared`` (``common.PAIR_SHARED_CC_DST_SITES``) puts the output all-reduce destination (``moe_out``), the
    routed all-reduce destinations (``moe_routed``: BF16 and FP32) and the destination of the fused latent world gather of
    the loop path (``moe_gather``) in LNC1 shared HBM. The DP gathers (``x``, ``x16``, ``xl``) and the split-router world
    gathers stay private; () keeps every tensor private.
    """
    tokens = dp_size * replica_tokens
    rr = reduce_rows or tokens
    specs = {
        "routed_hbm": ((WORLD_SIZE, rr, LATENT_SHARD), nl.bfloat16),
        "routed_dst_hbm": ((WORLD_SIZE, rr * LATENT_SHARD), nl.bfloat16),
        "out_hbm": ((_PMAX, rr * _H_TILES), nl.bfloat16),
        "out_dst_hbm": ((_PMAX, rr * _H_TILES), nl.bfloat16),
    }
    sites = {}
    if loop_path:
        if router_dtype == "fp16":
            sites["x16"] = (_PMAX, replica_tokens * 2 * _H_TILES, dp_size, nl.bfloat16)
        else:
            sites["x"] = (_PMAX, replica_tokens * _H_TILES, dp_size, nl.bfloat16)
        if router_dtype == "fp16_local":
            sites["xl"] = (DP_PEER_EXPERTS, replica_tokens * _TOKEN_LOGIT_WORDS, dp_size, nl.bfloat16)
        if router_mode == "fused":
            sites["gather"] = (tokens, _GATHER_COLS, WORLD_SIZE, nl.float32)
        else:
            sites["router"] = (tokens, _ROUTER_GATHER_COLS, WORLD_SIZE, nl.float32)
            sites["latent"] = (tokens, _LATENT_GATHER_WORDS, WORLD_SIZE, nl.float32)
    for site, (rows, width, ranks, dtype) in sites.items():
        specs[f"{site}_src_hbm"] = ((rows, width), dtype)
        specs[f"{site}_dst_hbm"] = ((ranks * rows, width), dtype)
    if routed_reduce_dtype == "fp32":
        # The fp32 routed reduce (not the DP1 default) stays at the call rows; reduce_rows sizes the bf16 path.
        for key in ("routed32_src_hbm", "routed32_dst_hbm"):
            specs[key] = ((WORLD_SIZE, tokens * LATENT_SHARD), nl.float32)
    shared = {
        "out_dst_hbm": "moe_out" in pair_shared,
        "routed_dst_hbm": "moe_routed" in pair_shared,
        "routed32_dst_hbm": "moe_routed" in pair_shared,
        "gather_dst_hbm": "moe_gather" in pair_shared,
    }
    out = {}
    for key, (shape, dtype) in specs.items():
        call = next_call("moe_hbm")
        out[key] = nl.ndarray(
            shape, dtype=dtype, buffer=pair_shared_buffer(shared.get(key, False)), name=f"moe_{key}_{call}"
        )
    return out


def _hbm_site(sources, site: str) -> bool:
    """Return True when ``sources`` has the persistent HBM pair of the collective ``site``."""
    return sources is not None and f"{site}_src_hbm" in sources


def _hbm_pair(sources, site: str, src_shape: tuple, dst_shape: tuple, dtype, name: str):
    """Return the persistent HBM source and destination of a collective ``site``, or new ones (no ``sources``).

    The persistent tensors take part in the same collective in every layer, so their bytes stay with it for the
    whole kernel (``routed_collective_hbm``). The compiler gives the bytes of a dead HBM tensor to a later one, so
    new tensors in each call are not enough.
    """
    if sources is not None and f"{site}_src_hbm" in sources:
        src_hbm, dst_hbm = sources[f"{site}_src_hbm"], sources[f"{site}_dst_hbm"]
        kernel_assert(
            tuple(src_hbm.shape) == tuple(src_shape) and tuple(dst_hbm.shape) == tuple(dst_shape),
            f"HBM collective tensors of {site}: {src_hbm.shape} {dst_hbm.shape}, expected {src_shape} {dst_shape}",
        )
        return src_hbm, dst_hbm
    call = next_call("moe_hbm")
    return (
        nl.ndarray(src_shape, dtype=dtype, buffer=nl.private_hbm, name=f"{name}_src_{call}"),
        nl.ndarray(dst_shape, dtype=dtype, buffer=nl.private_hbm, name=f"{name}_dst_{call}"),
    )


def _by_rank(gathered: nl.NkiTensor, width: int) -> nl.NkiTensor:
    """Return the [T, 128, width] view of a world gather. An HBM gather (``_all_gather_hbm``) is already in this
    shape. A reshape sets the partition step of a one-token view to 0, so one token splits the free axis only."""
    if len(gathered.shape) == 3:
        return gathered
    if gathered.shape[0] == 1:
        return gathered.reshape_dim(1, (WORLD_SIZE, width))
    return gathered.reshape((gathered.shape[0], WORLD_SIZE, width))


def _all_gather_hbm(
    src: nl.NkiTensor,
    group: ReplicaGroup,
    ranks: int,
    name: str,
    sources: dict = None,
    site: str = "",
    rank_rows: bool = False,
    strided: bool = True,
) -> nl.NkiTensor:
    """All-gather the SBUF rows ``src`` [R, F] of ``ranks`` ranks through HBM tensors; return SBUF [R, ranks, F].

    No peer accesses SBUF on the transfer. An HBM gather stacks the ranks on dim 0 ([ranks * R, F]); one strided
    load puts row k * R + r in partition r at columns [k * F, (k + 1) * F), the layout of an SBUF gather on the
    free axis. The HBM tensors are the persistent tensors of ``site`` (``_hbm_pair``). With ``rank_rows``, one
    contiguous DMA also loads the gather as SBUF [ranks, R * F] (rank k on partition k) ahead of the strided load,
    and the return value is the pair (strided tile, rank rows). ``strided=False`` (with ``rank_rows`` only) issues no
    strided load, and the pair is (None, rank rows).
    """
    kernel_assert(strided or rank_rows, "_all_gather_hbm without the strided load needs rank_rows")
    rows, width = src.shape
    src_hbm, dst_hbm = _hbm_pair(sources, site, (rows, width), (ranks * rows, width), src.dtype, name)
    call = next_call("moe_hbm")
    nisa.dma_copy(dst=src_hbm, src=src, name=f"{name}_store_{call}")
    check_pair_group(dst_hbm, group, f"{name}_gather")
    ncc.all_gather(srcs=[src_hbm], dsts=[dst_hbm], replica_group=group, collective_dim=0, name=f"{name}_gather")
    if rank_rows:
        rows_sb = nl.ndarray((ranks, rows * width), dtype=src.dtype, buffer=nl.sbuf)
        rows_load = nisa.dma_copy(
            dst=rows_sb, src=dst_hbm.reshape((ranks, rows * width)), name=f"{name}_rank_rows_{call}"
        )
        if not strided:
            return None, rows_sb
    gathered = nl.ndarray((rows, ranks, width), dtype=src.dtype, buffer=nl.sbuf)
    load = nisa.dma_copy(
        dst=gathered,
        src=dst_hbm.ap(pattern=[[width, rows], [rows * width, ranks], [1, width]], offset=0),
        name=f"{name}_load_{call}",
    )
    if not rank_rows:
        return gathered
    # The strided load to R partitions lands slowly, so the rank rows go first on the queue.
    _instruction(load).depends_on(_instruction(rows_load), kind=nisa.dependency.schedule)
    return gathered, rows_sb


def _all_reduce_hbm(src: nl.NkiTensor, group: ReplicaGroup, name: str, sources: dict = None, site: str = ""):
    """All-reduce (add) the SBUF rows ``src`` [R, F] through HBM tensors; return the sum in a new SBUF tile.

    As ``_all_gather_hbm``: no peer accesses SBUF, and an HBM collective takes no compiler scratch in SBUF. The
    collective is the gate of the next body (``release_weight_gate``).
    """
    src_hbm, dst_hbm = _hbm_pair(sources, site, tuple(src.shape), tuple(src.shape), src.dtype, f"{name}_ar")
    call = next_call("moe_hbm")
    nisa.dma_copy(dst=src_hbm, src=src, name=f"{name}_ar_store_{call}")
    check_pair_group(dst_hbm, group, f"{name}_reduce")
    release_weight_gate(
        ncc.all_reduce(srcs=[src_hbm], dsts=[dst_hbm], replica_group=group, op=nl.add, name=f"{name}_reduce")
    )
    reduced = nl.ndarray(tuple(src.shape), dtype=src.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=reduced, src=dst_hbm, name=f"{name}_ar_load_{call}")
    return reduced


def _selected_scores(router: RouterScores, finish: bool = True) -> tuple:
    """Return the selected scores [T, G, n] (0 for an expert that is not selected), the inverse score sum [T, 1]
    and the selection threshold [T, 1] of each token (see ``_routed_local_affinities``).

    With ``finish=False``, the second value is the score sum itself; ``_inverse_score_sum`` finishes it later.
    """
    scores, selection = router.scores, router.selection
    tokens, groups, group_experts = selection.shape

    # Two max8 rounds give the 16 largest selection values; the last one is the threshold.
    top8_first = nl.ndarray((tokens, 8), dtype=nl.float32, buffer=nl.sbuf)
    nisa.max8(dst=top8_first, src=selection)
    selection_rest = nl.ndarray((tokens, groups, group_experts), dtype=nl.float32, buffer=nl.sbuf)
    nisa.nc_match_replace8(dst=selection_rest, data=selection, vals=top8_first, imm=_SELECTION_FLOOR)
    top8_second = nl.ndarray((tokens, 8), dtype=nl.float32, buffer=nl.sbuf)
    nisa.max8(dst=top8_second, src=selection_rest)
    threshold = top8_second[:, 7:8]

    selected_scores = nl.ndarray((tokens, groups, group_experts), dtype=nl.float32, buffer=nl.sbuf)
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
    if not finish:
        return selected_scores, score_sum, threshold
    inv_score_sum, _ = _inverse_score_sum(score_sum)
    return selected_scores, inv_score_sum, threshold


def _inverse_score_sum(score_sum: nl.NkiTensor) -> tuple:
    """Return ``1 / (score_sum + 1e-20)`` [T, 1] and the first instruction (the in-place add)."""
    tokens = score_sum.shape[0]
    add = nisa.tensor_scalar(dst=score_sum, data=score_sum, op0=nl.add, operand0=ROUTER_SUM_EPS)
    inv_score_sum = nl.ndarray((tokens, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.reciprocal(dst=inv_score_sum, data=score_sum)
    return inv_score_sum, add


def _routed_local_affinities(router: RouterScores) -> nl.NkiTensor:
    """Compute this rank's FP32 [T, 7] top-16 router affinities from the router values of all experts.

    The selection value of expert e is ``score[e] + bias[e]``. The 16th largest selection value
    of a token is its threshold, so an expert is selected when its selection value is at least
    the threshold. The affinity is the score of a selected expert, divided by the sum of the
    selected scores plus 1e-20.

    Args:
        router: the router values (``RouterScores``); ``local_scores`` and ``local_selection`` are
            this rank's experts.
    """
    local_scores, local_selection = router.local_scores, router.local_selection
    tokens = local_scores.shape[0]
    _, inv_score_sum, threshold = _selected_scores(router)

    local_selected = nl.ndarray((tokens, LOCAL_EXPERTS), dtype=nl.float32, buffer=nl.sbuf)
    # scalar_tensor_tensor returns a raw instruction, not an NkiInstruction, in this release. Wrap it so
    # the scale below can depend on it.
    select = nisa.NkiInstruction(
        nisa.scalar_tensor_tensor(
            dst=local_selected,
            data=local_selection,
            op0=nl.greater_equal,
            operand0=threshold,
            op1=nl.multiply,
            operand1=local_scores,
        )
    )
    affinities = nl.ndarray((tokens, LOCAL_EXPERTS), dtype=nl.float32, buffer=nl.sbuf)
    scale = nisa.tensor_scalar(dst=affinities, data=local_selected, op0=nl.multiply, operand0=inv_score_sum)
    # The select and the scale both run on the Vector engine, and the scale can issue before the select has
    # written local_selected. The scheduler can place the select before the reciprocal; the scale then waits
    # on the reciprocal, and the in-order rule drops the select -> scale edge. When the reciprocal and the
    # select share no partition, nothing orders the two, so the scale can read a stale local_selected. A
    # runtime dependency adds a semaphore wait that the dependency-reduction pass keeps. The math does not change.
    scale.depends_on(select, kind=nisa.dependency.runtime)
    return affinities


def _top16_routing(router: RouterScores, routing_form: str = "base", tiles=AUTO_SBUF) -> tuple:
    """Return the 16 routed experts of each token over all 896 experts and their affinities.

    The selected scores and the inverse score sum are those of ``_routed_local_affinities``, so each
    affinity has the bits of the EP128 path. Two max8 rounds over the [T, G, n] selected scores
    (zero for an expert that is not selected) give the 16 scores in descending order and their
    positions, which are the expert ids (expert e = n g + j is at position e).

    With ``routing_form`` ``index_first``, the instructions of the score sum reciprocal follow the index on the
    Vector engine, so the expert index is ready earlier. The instructions and their inputs do not change.

    Args:
        router: the router values of all experts (``RouterScores``).
        routing_form: the routing form (``ROUTING_FORMS``).
        tiles: the allocator (``SbufTiles``) of the three [T, 16] outputs (the top scores, the index and the
            affinities). ``AUTO_SBUF`` (default): compiler placement.

    Returns:
        ``(expert_index, affinities, index_done)``: INT32 [T, 16] and FP32 [T, 16] SBUF, token t on partition t,
        and the last index instruction for the forms other than ``base`` (else None). Column k of a token is its k-th
        largest score; equal scores take a fixed position order.
    """
    if routing_form == "selection_first":
        return _top16_selection_first(router)
    index_first = routing_form in ("index_first", *_PE32_FORMS, *_B1_ROUTING_FORMS)
    selected_scores, score_term, _ = _selected_scores(router, finish=not index_first)
    tokens, groups, group_experts = selected_scores.shape
    top_scores = tiles.tile((tokens, TOP_K), nl.float32, name="route_top_scores")
    top_index = tiles.tile((tokens, TOP_K), nl.uint32, name="route_top_index")
    first = nisa.max8(dst=top_scores[:, 0:8], src=selected_scores)
    scores_rest = nl.ndarray((tokens, groups, group_experts), dtype=nl.float32, buffer=nl.sbuf)
    # Every selected score is positive and every other entry is 0, so -1 takes a found score out of the
    # second round.
    nisa.nc_match_replace8(
        dst=scores_rest, data=selected_scores, vals=top_scores[:, 0:8], imm=-1.0, dst_idx=top_index[:, 0:8]
    )
    second = nisa.max8(dst=top_scores[:, 8:TOP_K], src=scores_rest)
    found = nisa.nc_find_index8(dst=top_index[:, 8:TOP_K], data=scores_rest, vals=top_scores[:, 8:TOP_K])
    index_done = None
    inv_score_sum = score_term
    if index_first:
        index_done = _instruction(found)
        inv_score_sum, add = _inverse_score_sum(score_term)
        _instruction(add).depends_on(index_done, kind=nisa.dependency.runtime)
    affinities = tiles.tile((tokens, TOP_K), nl.float32, name="route_affinities")
    scale = nisa.tensor_scalar(dst=affinities, data=top_scores, op0=nl.multiply, operand0=inv_score_sum)
    # The scale reads the two max8 halves of one tile on the Vector engine. Keep both writes ordered before it
    # with runtime edges, as for the scale of _local_affinities.
    for producer in (first, second):
        scale.depends_on(_instruction(producer), kind=nisa.dependency.runtime)
    routing = _TopRouting((top_index.view(nl.int32), affinities, index_done))
    routing.selected_scores = selected_scores
    return routing


class _TopRouting(tuple):
    """The (index, affinities, index_done) of ``_top16_routing``, with its selected scores [T, G, n] (the kimi_t32
    body reads them)."""

    selected_scores = None


def _top16_selection_first(router: RouterScores) -> tuple:
    """Return the routing of ``_top16_routing`` from the positions of the 16 largest selection values.

    Two max8 rounds over the selection values give the threshold (the instructions of ``_selected_scores``), and
    match_replace8 and find_index8 give the positions of these 16 values. A Pool gather takes their scores, and the
    max8 sequence of ``_top16_routing`` over the 16 scores gives the order; a second gather maps it to the expert ids.
    The score sum is the pass of ``_selected_scores`` after the index, so the affinities keep their bits.

    When exactly 16 experts reach the threshold, these 16 experts are the selected experts, and the max8 sequence
    over their scores gives the order of the base if the 16 scores are different. Equal scores take the order of
    their selection values, not the position order of the base.

    Args:
        router: the router values of all experts (``RouterScores``).

    Returns:
        ``(expert_index, affinities, index_done)`` as ``_top16_routing``.
    """
    scores, selection = router.scores, router.selection
    tokens, groups, group_experts = selection.shape
    top_selection = nl.ndarray((tokens, TOP_K), dtype=nl.float32, buffer=nl.sbuf)
    positions = nl.ndarray((tokens, TOP_K), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.max8(dst=top_selection[:, 0:8], src=selection)
    selection_rest = nl.ndarray((tokens, groups, group_experts), dtype=nl.float32, buffer=nl.sbuf)
    nisa.nc_match_replace8(
        dst=selection_rest,
        data=selection,
        vals=top_selection[:, 0:8],
        imm=_SELECTION_FLOOR,
        dst_idx=positions[:, 0:8],
    )
    nisa.max8(dst=top_selection[:, 8:TOP_K], src=selection_rest)
    nisa.nc_find_index8(dst=positions[:, 8:TOP_K], data=selection_rest, vals=top_selection[:, 8:TOP_K])
    threshold = top_selection[:, TOP_K - 1 : TOP_K]
    # A contiguous copy of the scores for the gather; the Scalar engine makes it during the selection passes.
    scores_flat = nl.ndarray((tokens, groups * group_experts), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=scores_flat.reshape((tokens, groups, group_experts)), src=scores, engine=nisa.engine.scalar)
    picked = nl.ndarray((tokens, TOP_K), dtype=nl.float32, buffer=nl.sbuf)
    nisa.nc_n_gather(dst=picked, data=scores_flat, indices=positions)

    top_scores = nl.ndarray((tokens, TOP_K), dtype=nl.float32, buffer=nl.sbuf)
    slots = nl.ndarray((tokens, TOP_K), dtype=nl.uint32, buffer=nl.sbuf)
    first = nisa.max8(dst=top_scores[:, 0:8], src=picked)
    picked_rest = nl.ndarray((tokens, TOP_K), dtype=nl.float32, buffer=nl.sbuf)
    nisa.nc_match_replace8(dst=picked_rest, data=picked, vals=top_scores[:, 0:8], imm=-1.0, dst_idx=slots[:, 0:8])
    second = nisa.max8(dst=top_scores[:, 8:TOP_K], src=picked_rest)
    found = nisa.nc_find_index8(dst=slots[:, 8:TOP_K], data=picked_rest, vals=top_scores[:, 8:TOP_K])
    expert_index = nl.ndarray((tokens, TOP_K), dtype=nl.uint32, buffer=nl.sbuf)
    index_done = _instruction(nisa.nc_n_gather(dst=expert_index, data=positions, indices=slots))

    selected_scores = nl.ndarray((tokens, groups, group_experts), dtype=nl.float32, buffer=nl.sbuf)
    score_sum = nl.ndarray((tokens, 1), dtype=nl.float32, buffer=nl.sbuf)
    select = _instruction(
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
    )
    # The full pass for the sum follows the 16-score sort on the Vector engine.
    select.depends_on(_instruction(found), kind=nisa.dependency.runtime)
    inv_score_sum, _ = _inverse_score_sum(score_sum)
    affinities = nl.ndarray((tokens, TOP_K), dtype=nl.float32, buffer=nl.sbuf)
    scale = nisa.tensor_scalar(dst=affinities, data=top_scores, op0=nl.multiply, operand0=inv_score_sum)
    for producer in (first, second):
        scale.depends_on(_instruction(producer), kind=nisa.dependency.runtime)
    return expert_index.view(nl.int32), affinities, index_done


def _instruction(handle):
    """Return ``handle`` as an instruction that takes dependency edges (some ISA calls return a raw instruction)."""
    return handle if hasattr(handle, "depends_on") else nisa.NkiInstruction(handle)


def _store_routing(expert_index: nl.NkiTensor, affinities: nl.NkiTensor, outputs: tuple, name: str):
    """Store the [T, 16] routing tiles to the HBM ``outputs`` (index, affinity) of a test."""
    for idx, (src, dst) in enumerate(zip((expert_index, affinities), outputs, strict=True)):
        nisa.dma_copy(dst=dst, src=src, name=f"{name}_{idx}")


def _gather_routed_tokens(
    x_hT: nl.NkiTensor,
    dp_group: ReplicaGroup,
    name: str,
    x16_hT: nl.NkiTensor = None,
    sources: dict = None,
    dp_size: int = DP_SIZE,
):
    """All-gather the token-major hT hidden of the DP replicas (and its FP16 router copy, in the same gather).

    Returns:
        [128, 56, T] BF16 SBUF view. Element (p, j, r * B + b) is hidden index j * 128 + p of
        token b of replica r. ``[:, j, :]`` is the matmul stationary of hidden tile j: one free
        dimension with a step of 56 elements, so no copy is necessary. With ``x16_hT``, the pair
        ``(bf16 view, fp16 view)``: each token sends its BF16 row, then its FP16 row, so the step is
        112 elements. With ``dp_size`` 1 there is no gather: the views of ``x_hT`` (and ``x16_hT``)
        are returned directly.
    """
    replica_tokens = x_hT.shape[1]
    tokens = dp_size * replica_tokens
    if dp_size == 1:
        if x16_hT is not None:
            return x_hT.permute((0, 2, 1)), x16_hT.permute((0, 2, 1))
        return x_hT.permute((0, 2, 1))
    if x16_hT is not None:
        # [128, B, 2, 56] two-byte words: the BF16 row, then the FP16 row of each token (bit copies).
        src = _tile(sources, "x16_src", (_PMAX, replica_tokens, 2, _H_TILES), nl.bfloat16)
        nisa.tensor_copy(dst=src[:, :, 0, :], src=x_hT)
        nisa.tensor_copy(dst=src.view(nl.float16)[:, :, 1, :], src=x16_hT)
        src_rows = src.reshape((_PMAX, replica_tokens * 2 * _H_TILES))
        if _hbm_site(sources, "x16"):
            gathered = _all_gather_hbm(src_rows, dp_group, DP_SIZE, name, sources, "x16")
        else:
            gathered = _tile(sources, "x16_dst", (_PMAX, tokens * 2 * _H_TILES), nl.bfloat16)
            ncc.all_gather(
                srcs=[src_rows], dsts=[gathered], replica_group=dp_group, collective_dim=1, name=f"{name}_gather"
            )
        pairs = gathered.reshape((_PMAX, tokens, 2, _H_TILES))
        pairs16 = gathered.view(nl.float16).reshape((_PMAX, tokens, 2, _H_TILES))
        return pairs[:, :, 0, :].permute((0, 2, 1)), pairs16[:, :, 1, :].permute((0, 2, 1))
    if _hbm_site(sources, "x"):
        rows = _all_gather_hbm(x_hT.reshape((_PMAX, replica_tokens * _H_TILES)), dp_group, dp_size, name, sources, "x")
        return rows.reshape((_PMAX, tokens, _H_TILES)).permute((0, 2, 1))
    gathered = _tile(sources, "x_dst", (_PMAX, tokens * _H_TILES), nl.bfloat16)
    ncc.all_gather(
        srcs=[x_hT.reshape((_PMAX, replica_tokens * _H_TILES))],
        dsts=[gathered],
        replica_group=dp_group,
        collective_dim=1,
        name=f"{name}_gather",
    )
    return gathered.reshape((_PMAX, tokens, _H_TILES)).permute((0, 2, 1))


def _gather_tokens_and_local_logits(
    x_hT: nl.NkiTensor,
    x16_hT: nl.NkiTensor,
    router_w_sb: nl.NkiTensor,
    row_select: nl.NkiTensor,
    dp_group: ReplicaGroup,
    name: str,
    sources: dict = None,
):
    """All-gather the token-major hT hidden of the 4 DP replicas with the local FP16 router logits (``fp16_local``).

    The rank computes the FP32 logits [28, B] of its tokens for the experts of its 4 DP peers from the FP16 copy of
    the norm: the products and the accumulation order of ``_router_logits``, with the experts as the stationary
    operand. Each token sends its 56 BF16 hidden words and, on partitions 0..27, its 28 logits as 2 words each. After
    the gather, the rank keeps the logits of its own 7 experts (the block of its DP rank, selected with the replica
    one-hot ``row_select``) for all T tokens. On the MLA loop path the gather runs in two persistent HBM pairs
    (``_gather_tokens_and_logits_hbm``), so the explicit region holds no copy of the hidden.

    Returns:
        ``(x_tokens_hT, logits)``: the [128, 56, T] BF16 view of ``_gather_routed_tokens``, and the [T, 7] FP32 SBUF
        logits of this rank's experts, bit-identical to the ``fp16`` router logits.
    """
    replica_tokens = x_hT.shape[1]
    tokens = DP_SIZE * replica_tokens
    local_psum = _peer_router_logits(x16_hT, router_w_sb)

    if _hbm_site(sources, "xl"):
        logit_src = _tile(sources, "xl_logits", (DP_PEER_EXPERTS, replica_tokens), nl.float32)
        nisa.tensor_copy(dst=logit_src, src=local_psum)
        x_tokens_hT, peer_logits = _gather_tokens_and_logits_hbm(x_hT, logit_src, dp_group, name, sources)
    else:
        src = _tile(sources, "xl_src", (_PMAX, replica_tokens, _TOKEN_GATHER_WORDS), nl.bfloat16)
        nisa.tensor_copy(dst=src[:, :, 0:_H_TILES], src=x_hT)
        nisa.tensor_copy(dst=src.view(nl.float32)[0:DP_PEER_EXPERTS, :, _H_TILES // 2], src=local_psum)
        src_rows = src.reshape((_PMAX, replica_tokens * _TOKEN_GATHER_WORDS))
        gathered = _tile(sources, "xl_dst", (_PMAX, tokens * _TOKEN_GATHER_WORDS), nl.bfloat16)
        ncc.all_gather(
            srcs=[src_rows], dsts=[gathered], replica_group=dp_group, collective_dim=1, name=f"{name}_gather"
        )
        words = gathered.reshape((_PMAX, tokens, _TOKEN_GATHER_WORDS))
        x_tokens_hT = words[:, :, 0:_H_TILES].permute((0, 2, 1))
        peer_logits = _packed_peer_logits(gathered, tokens)
    return x_tokens_hT, _own_router_logits(peer_logits, row_select, tokens)


def _peer_router_logits(x16_hT: nl.NkiTensor, router_w_sb: nl.NkiTensor) -> nl.NkiTensor:
    """Return the FP32 PSUM logits [28, B] of the experts of the 4 DP peers for this replica's tokens (``fp16_local``).

    ``x16_hT`` is the [128, B, 56] FP16 norm copy and ``router_w_sb`` the [128, 56, 28] FP16 rows of the DP peers:
    the products and the accumulation order of the ``fp16`` router, with the experts as the stationary operand.
    """
    local_psum = _psum_bank(DP_PEER_EXPERTS, (x16_hT.shape[1],), nl.float32)
    for h_tile_idx in range(_H_TILES):
        nisa.nc_matmul(dst=local_psum, stationary=router_w_sb[:, h_tile_idx, :], moving=x16_hT[:, :, h_tile_idx])
    return local_psum


def _packed_peer_logits(gathered: nl.NkiTensor, tokens: int) -> nl.NkiTensor:
    """Return the [28, T] FP32 view of the peer logits in the packed token rows of a DP x gather destination."""
    peer_words = gathered.view(nl.float32).reshape((_PMAX, tokens, _TOKEN_GATHER_WORDS // 2))
    return peer_words[0:DP_PEER_EXPERTS, :, _H_TILES // 2]


def _own_router_logits(peer_logits: nl.NkiTensor, row_select: nl.NkiTensor, tokens: int) -> nl.NkiTensor:
    """Return the [T, 7] FP32 SBUF logits of this rank's experts from the gathered [28, T] peer logits."""
    token_logits = nl.ndarray((tokens, DP_PEER_EXPERTS), dtype=nl.float32, buffer=nl.sbuf)
    if tokens <= _VECTOR_TRANSPOSE_MAX:
        nisa.nc_transpose(dst=token_logits, data=peer_logits, engine=nisa.vector_engine)
    else:
        # The Vector engine transposes at most 32 x 32: one transpose per 32 tokens (data movement, the same bits).
        for start in range(0, tokens, _VECTOR_TRANSPOSE_MAX):
            stop = min(start + _VECTOR_TRANSPOSE_MAX, tokens)
            nisa.nc_transpose(
                dst=token_logits[start:stop, :], data=peer_logits[:, start:stop], engine=nisa.vector_engine
            )
    return _own_block(token_logits, row_select, tokens)


def _token_router_logits(
    x16_tokens_hT: nl.NkiTensor, router_w_sb: nl.NkiTensor, row_select: nl.NkiTensor, tokens: int
) -> nl.NkiTensor:
    """Return the [T, 7] FP32 SBUF logits of this rank's experts from the gathered [128, 56, T] FP16 rows.

    The stack of the DP peers (``_peer_router_logits``) holds this rank's 7 experts as the block of its DP rank, and each
    output column of the matmul is its own dot product, so the logits keep the bits of the ``fp16`` router.
    """
    psum = _psum_bank(tokens, (DP_PEER_EXPERTS,), nl.float32)
    for h_tile_idx in range(_H_TILES):
        nisa.nc_matmul(dst=psum, stationary=x16_tokens_hT[:, h_tile_idx, :], moving=router_w_sb[:, h_tile_idx, :])
    return _own_block(psum, row_select, tokens)


def _own_block(token_logits: nl.NkiTensor, row_select: nl.NkiTensor, tokens: int) -> nl.NkiTensor:
    """Return the [T, 7] block of this rank's DP rank of the [T, 28] peer logits, in a new FP32 SBUF tile."""
    # x * 1 plus the other blocks times 0 (exact for finite logits).
    logits = nl.ndarray((tokens, LOCAL_EXPERTS), dtype=nl.float32, buffer=nl.sbuf)
    for dp_rank in range(DP_SIZE):
        block = token_logits[:, LOCAL_EXPERTS * dp_rank : LOCAL_EXPERTS * (dp_rank + 1)]
        one_hot = row_select[0:tokens, dp_rank : dp_rank + 1]
        if dp_rank == 0:
            nisa.tensor_scalar(dst=logits, data=block, op0=nl.multiply, operand0=one_hot)
        else:
            nisa.scalar_tensor_tensor(
                dst=logits, data=block, op0=nl.multiply, operand0=one_hot, op1=nl.add, operand1=logits
            )
    return logits


def _gather_tokens_and_logits_hbm(
    x_hT: nl.NkiTensor, logit_src: nl.NkiTensor, dp_group: ReplicaGroup, name: str, sources: dict = None
) -> tuple:
    """DP all-gather of the ``fp16_local`` tokens in two persistent HBM pairs (one collective) on the MLA loop path.

    One DMA stores the [128, B * 56] hidden of ``x_hT`` in the pair ``x``, and one the [28, B] FP32 logits of
    ``logit_src`` in the pair ``xl`` (as BF16 words: the pairs of one collective take one dtype); each is one
    contiguous row per partition. The gather stacks the ranks on dim 0, and one strided load of each pair puts them in
    the layout of an SBUF gather on the free axis (``_all_gather_hbm``).

    Returns:
        ``(x_tokens_hT, peer_logits)``: the [128, 56, T] BF16 view of ``_gather_routed_tokens``, and the [28, T] FP32
        logits of the experts of the DP peers.
    """
    replica_tokens = x_hT.shape[1]
    tokens = DP_SIZE * replica_tokens
    width = replica_tokens * _H_TILES
    x_src, x_dst = _hbm_pair(sources, "x", (_PMAX, width), (DP_SIZE * _PMAX, width), nl.bfloat16, name)
    peers = DP_PEER_EXPERTS
    words = replica_tokens * _TOKEN_LOGIT_WORDS
    l_src, l_dst = _hbm_pair(sources, "xl", (peers, words), (DP_SIZE * peers, words), nl.bfloat16, name)
    call = next_call("moe_hbm")
    nisa.dma_copy(dst=x_src, src=x_hT.reshape((_PMAX, width)), name=f"{name}_store_x_{call}")
    nisa.dma_copy(dst=l_src, src=logit_src.view(nl.bfloat16), name=f"{name}_store_l_{call}")
    ncc.all_gather(
        srcs=[x_src, l_src], dsts=[x_dst, l_dst], replica_group=dp_group, collective_dim=0, name=f"{name}_gather"
    )
    x_rows = nl.ndarray((_PMAX, DP_SIZE, width), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=x_rows,
        src=x_dst.ap(pattern=[[width, _PMAX], [_PMAX * width, DP_SIZE], [1, width]], offset=0),
        name=f"{name}_load_x_{call}",
    )
    peer_words = nl.ndarray((peers, DP_SIZE, words), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=peer_words,
        src=l_dst.ap(pattern=[[words, peers], [peers * words, DP_SIZE], [1, words]], offset=0),
        name=f"{name}_load_l_{call}",
    )
    x_tokens_hT = x_rows.reshape((_PMAX, tokens, _H_TILES)).permute((0, 2, 1))
    return x_tokens_hT, peer_words.view(nl.float32).reshape((peers, tokens))


def _tile(sources: dict, key: str, shape: tuple, dtype) -> nl.NkiTensor:
    """Return the persistent collective tile ``key`` of ``sources``, or a new SBUF tile when it has none."""
    if sources is None or key not in sources:
        return nl.ndarray(shape, dtype=dtype, buffer=nl.sbuf)
    tile = sources[key]
    kernel_assert(tuple(tile.shape) == tuple(shape), f"collective tile {key}: {tile.shape} is not {shape}")
    return tile
