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

"""Kimi-K3 multi-server Trn3 verify giga-kernel: the decode layers on Q tokens of each sequence over several servers.

The entry is the multi-server form of ``verify_model.kimi_k3_verify_model`` (speculative decoding), as
``decode_model_2srv.kimi_k3_decode_model_2srv`` is the multi-server form of the decode. It runs the verify bodies of
``verify_model.verify_bodies`` with the feed-forward bodies of the multi-server decode, so that each row is bitwise
equal to the decode step of its token:

- Row ``j * B + b`` is token ``j`` of sequence ``b``, at position ``position_rows[j, b]``.
- KDA: ``kda_verify_attention``: it replays the accepted tokens of the previous step onto the committed state, writes
  that state back to its page, then scores the Q tokens (``kda_replay`` records or states).
- MLA: ``mla_mode`` ``k1`` (default) runs one bitwise multi-query ``mla_block_attention`` call on the R = Q * B rows
  of each layer; ``per_token`` runs one ``mla_decode_attention`` call for each token. Both run on the TP32 / CP32
  replica of one server, as the decode.
- Feed-forward: the MoE body of ``tkg_moe_2srv`` and the dense body of ``tkg_dense_mlp_2srv`` in one call on all R
  rows of the replica. A layer runs the collectives of one decode step on T = dp * R gathered tokens: the MoE expert
  calls take them in calls of 128 tokens, or in token-tiled calls (``moe_expert_call_tokens``).
- Tail: the greedy tokens or the logits of all R rows over the TP32 vocab shard, and the aux states of the drafter.
- Split: the layer-range calls of ``kimi_k3_verify_model`` (``attn_res_bank_in``, ``return_attn_res_bank``,
  ``attn_res_fence_in``). The boundary fence all-reduces over the world group. Every collective of the multi-server
  bodies feeds a later instruction of its call (the DP and TP32 row gathers feed the expert calls, the routed and
  output all-reduces feed the layer output, as do the reduce-scatter and the all-gather of ``moe_routed_reduce``
  ``rs_ag``), so the fence audit of ``verify_model.boundary_fence`` holds for them.

The verify keeps the AttnRes mix of the plain bits. ``attn_res_cached_bank_sums`` (the kept bank rstd; the bits of the
plain mix, ``decode_block.decode_block_body``) applies to the one-call verify only, since the split calls pass no bank
rstd; ``mix_cache`` changes the bits and has no verify form here.
"""

import itertools

import nki
import nki.collectives as ncc
import nki.isa as nisa
import nki.language as nl
from nki.collectives import ReplicaGroup

from ..utils.kernel_assert import kernel_assert
from ..moe.routed import EXPERT_QUEUES
from ..utils.trace_names import restart_calls
from ..common import (
    BANK_DMAS,
    P_MAX,
    SplitBank,
    attn_res_hT,
    check_pair_group,
    load_rows_hT,
    load_rows_hT_batched,
    load_vector_hT,
    new_ones,
    pair_shared_buffer,
    rmsnorm_hT,
    row_sums_after_ffn_output,
    store_hT_rows,
)
from ..decode_block import (
    _BANK_HBM_ROWS,
    ATTN_RES_AHEAD_VECTOR_MODES,
    NUM_LAYERS,
    banks_snapshot,
    block_layer_counts,
    check_pair_shared_forms,
    collective_sources,
    decode_block_body,
    is_mla_layer,
    layer_sources,
    region_ones,
    region_streams,
    trn3_kda_w_in_prefetch,
)
from ..decode_model import MAX_POOLS, TAIL_TP_RANKS, _group_stack, _rows, model_layer_groups
from ..tkg_lm_head_greedy import lm_head_greedy_hT
from ..tkg_weight_dma_gate import set_weight_gate, weight_gates
from ..tkg_sbuf_alloc import AUTO_SBUF, ExplicitSbuf, SbufTiles, check_identities, install_identities, install_null_rows
from ..tkg_kda import (
    KDA_STATE_DTYPES,
    _w_in_buffers,
    kda_conv_zero_rows_tile,
    kda_local_heads,
    kda_page_offsets,
    kda_page_span,
    kda_slot_state_tiles,
)
from ..verify_kda import (
    KDA_PAGE_GROUPS_MAX,
    kda_flush_ring_keeps,
    kda_page_group_tiles,
    kda_slot_conv_rows_tiles,
    kda_verify_constants,
)
from ..tkg_grammar_mask import GrammarMask
from ..tkg_kda import KDA_WEIGHT_DMAS
from ..tkg_mla_block import (
    PASS_Q_PREFETCH_MAX_TILES,
    ROWS_OPTIONS,
    ROWS_PASS_ROWS,
    ROWS_SEQUENCES,
    mla_block_collective_hbm,
    mla_block_collective_sources,
    mla_fp8_pv_step_stores,
    mla_pass_plans,
    q_call_fp8,
)
from ..tkg_mla_paged import LORA_NORM_EPS, QUERY_SET_TOKENS
from ..verify_model import (
    _K1_FIT_BLOCKS,
    _MLA_LOCAL_HEADS,
    AUX_LAYERS,
    AUX_TAPS,
    FENCE_SHAPE,
    MAX_QUERIES,
    _LayerIndex,
    boundary_fence,
    fence_tiles,
    verify_bodies,
)
from . import tkg_moe_2srv as moe_2srv_module
from .decode_model_2srv import (
    ATTENTION_MAX_SEQUENCES,
    ROUTED_RS_AG_FORMS,
    SERVED_ROWS_OPTIONS,
    hold_to_end,
    in_proj_block_options,
    out_proj_block_options,
    router_block_options,
)
from .tkg_dense_mlp_2srv import DENSE_SOURCES_KEY, SPLIT_RIDER_KEY, dense_collective_hbm, dense_ffn_2srv
from .tkg_ffn_world_cc import world_layout
from .moe_routed_plan import (
    ALT_STACK_ARGUMENTS,
    call_csr_gather,
    call_plan,
    check_alternates,
    check_plan_cut,
    moe_layers_of,
    plan_bodies,
    plan_selector_splits,
    plan_weights,
)
from .tkg_moe_2srv import (
    EXPERT_CALL_AUTO,
    EXPERT_CALL_TOKENS,
    MOE_FORMS,
    MOE_GATHERS,
    MOE_SOURCES_KEY,
    OUT_PROJ_FORMS,
    ROUTED_REDUCES,
    ROWS_LOAD_SPLITS,
    WIDE_HALF_WEIGHT_BUFFERS,
    check_column_private,
    moe_2srv_body,
    moe_2srv_collective_hbm,
    moe_2srv_compact_constants,
    moe_2srv_glue_selectors,
    moe_2srv_world_split,
    routed_layers_options,
)

DEFAULT_QUERIES = 4
"""Tokens of each sequence in the first multi-server spec step: the bonus token and 3 drafts (k3)."""
MLA_MODES = ("k1", "per_token")
"""The MLA forms of the verify (``mla_mode``): one multi-query call on all rows, or one call for each token."""
MAX_PREFETCH_WEIGHT_BUFFERS = 6
"""Most whole weight buffers of a prefetched expert ring (``moe_expert_prefetch``): two experts' projections."""
AUX_RING_DEPTH = 4
"""Staging tiles of the aux row stores on the region (``common.store_hT_rows`` ``ring``): the copy of row t waits for
the store of row t - 4, not of row t - 1, so the stores issue back to back. The forms of ``AUX_ROW_RING_FORM_SET`` only;
the other forms keep one tile a row."""
AUX_ROW_RING_FORM_SET = ("k3", "k7")
"""The served forms whose aux row stores take the staging ring by default (``aux_row_ring``): k7 (by its shape, also off
the k1 loop path) and k3 (C = 16, on the k1 loop path). The same bytes."""

MAX_VERIFY_ROWS = ATTENTION_MAX_SEQUENCES
"""Most rows (Q * B) of a replica in one verify call: the attention bound of the multi-server decode. The k1 call of
32 rows fits the 1M table (256 local blocks); a larger call has no measured fit."""
SIZING_VERIFY_ROWS = 2 * MAX_VERIFY_ROWS
"""Most rows of a replica with ``max_verify_rows``: the 2x-token sizing form (T = 512 gathered tokens at world 256, the
MoE token count of a 4-server C = 16 call). Its attention fit is not measured; it is not a served form."""
VERIFY_ROUTED_RS_AG_FORMS = tuple(dict.fromkeys(((256, 16), *ROUTED_RS_AG_FORMS, (256, 8))))
"""The (world_size, R) forms of the verify that take ``moe_routed_reduce`` ``rs_ag``, each once: the k3 form of 16 rows
of each replica (T = 128), the decode forms (``decode_model_2srv.ROUTED_RS_AG_FORMS``) and the 8-row form of k3b2 and
k7b1 (T = 64). At 16 rows the two collectives take more collective table entries than the routed all-reduce, so a call
of all layers can need the layer-range split. The 8-row form takes ``rs_ag`` only when the caller asks for it
(``served_defaults`` keeps the routed all-reduce)."""
VERIFY_ROUTED_RS_AG_FORMS_4SRV = ((512, 16), (512, 32), (512, 8))
"""The 4-server (world 512) forms of ``moe_routed_reduce`` ``rs_ag`` in the verify, beside
``VERIFY_ROUTED_RS_AG_FORMS``: the same rows of a replica (k3 and k7b2 at 16, k7 at 32, k3b2 and k7b1 at 8). Opt-in:
``served_defaults`` gives no 512 call the ``rs_ag`` default (it checks the real world; only the attention levers take
the 2-server form, ``form_world``), and the collective-table entries of a 512 program have no measurement yet."""
ATTENTION_FORM_WORLDS = {512: 256}
"""The world whose ``SERVED_FORMS`` forms a 4-server call takes for its attention levers: a 4-server world (512 ranks)
runs the same TP32 attention replicas (4 a host) as the 2-server world, so its attention keys on the 2-server form.
The MoE keys on the real world (its split, groups and collective shapes). Every other world keys on itself."""


def form_world(world_size: int) -> int:
    """Return the world of the served-form lookups of a call: 256 for a 4-server call (``ATTENTION_FORM_WORLDS``),
    else the call's world.

    Example:
        >>> form_world(256), form_world(512), form_world(128)
        (256, 256, 128)
    """
    return ATTENTION_FORM_WORLDS.get(world_size, world_size)


SERVED_ROUTED_RS_AG_MIN_ROWS = 16
"""The fewest rows of a replica for which ``served_defaults`` gives ``rs_ag``."""

SERVED_FORMS = {
    "k3": (256, 4, 16),
    "k7": (256, 8, 32),
    "k7b2": (256, 8, 16),
    "k7b1": (256, 8, 8),
    "k3b8": (256, 4, 32),
    "k3b2": (256, 4, 8),
}
"""The (world_size, queries, R) forms of the served 2-server verify: k3 (4 sequences of 4 tokens for each replica), k7
(4 sequences of 8 tokens), k7b2 (2 sequences of 8 tokens), k7b1 (1 sequence of 8 tokens), k3b8 (8 sequences of 4
tokens) and k3b2 (2 sequences of 4 tokens). On the k1 MLA loop path they take the validated levers by default
(``served_defaults``). k7b2 takes the bitwise levers of k7 (its 16 rows are one MLA pass, as each of the two passes of
k7 is) and keeps the routed all-reduce: ``rs_ag`` changes the bits (at most 1 BF16 ulp), so it waits for its own
accuracy check. k7b1 takes the levers of k7b2 (its 8 rows are one MLA pass of one sequence) and keeps the routed
all-reduce, as k3b2 does at 8 rows by default (``SERVED_ROUTED_RS_AG_MIN_ROWS``)."""
KDA_IN_PROJ_X_STATIONARY_FORM_SET = ("k7", "k3b2")
"""The served forms whose verify calls take the moe_glue key ``kda_in_proj_x_stationary`` by default
(``served_defaults``): the token-stationary MXFP8 KDA input projection (bitwise), with an MXFP8 ``kda_w_in``
(``kda_w_in_scale``) only. k3b2 takes it with ``kda_state_loads_after_ring`` (the pair measured a shorter 12-layer
C = 8 step); not k3, where the pair measured a longer step. A given False is the kill switch."""
KDA_STATE_LOADS_AFTER_RING_FORM_SET = ("k7", "k3b2")
"""The served forms whose verify calls take the moe_glue key ``kda_state_loads_after_ring`` by default
(``served_defaults``) with ``kda_state_loads_first``: the KDA state and conv-state loads follow the last prefetch fill of
the layer's ``kda_w_in`` ring in the schedule, so on the in-order Sync queue no state load sits between two fills (each
waits for its VectorE page index and for the earlier DMAs of its queue). Bitwise; a given False is the kill switch."""
K3_FORMS = ("k3", "k3b8", "k3b2")
"""The served forms of 4 tokens a sequence, which take the k3 levers."""
FFN_AFTER_ATTN_K7B2 = False
"""Kill switch (default True) of the k7b2 entry of ``FFN_AFTER_ATTN_FORMS``: False takes k7b2 out of the table (so a
caller that renders the key from the table, as the serving caller does, passes False) and gives the k7b2 form
``ffn_weights_after_attn`` False by default (the option off); k7 and the k3 forms keep it."""
FFN_AFTER_ATTN_FORMS = (*K3_FORMS, "k7") + (("k7b2",) if FFN_AFTER_ATTN_K7B2 else ())
"""The served forms whose feed-forward weight loads wait for the residual add of the attention output by default
(``ffn_weights_after_attn``, with the DMA gates): the k3 forms, k7 and, while ``FFN_AFTER_ATTN_K7B2`` is True, k7b2,
whose attention output all-reduces otherwise share the DMA rings with the gated weight stream of the next feed-forward
body. A given ``ffn_weights_after_attn`` False turns it off for one call."""
PACKED_FORMS = (*K3_FORMS, "k7", "k7b2", "k7b1")
"""The served forms whose k1 MLA call takes the packed prior-key loop by default (``served_defaults``): the k3 forms,
k7, k7b2 and k7b1. The loop is bitwise the per-sequence loop of the call."""
P2_FORMS = ("k3", "k3b2", "k7", "k7b2", "k7b1")
"""The served forms whose packed k1 loop takes the 128-column (row, head) sub-slots by default (``served_pack_options``):
k7, k7b2 and k7b1, Q = 8, 6 sub-slots an item in place of 8 at the same pass cost, bitwise the same."""
P2_SBUF_LEVER_FORMS = ("k3", "k7")
"""The ``P2_FORMS`` forms that also take ``stage_rows_sbuf`` and the 8-deep slot-row ring (``served_pack_options``): k7.
k7b2 (one pass of 16 rows) has no SBUF room for them; k7b1 keeps the k7b2 set."""
OWN_STATE_STACK_FORMS = ("k3", "k3b2", "k7", "k7b2")
"""The ``P2_FORMS`` forms whose p2 loop writes the own-key state with one matmul for each group of 128 columns, over the
4 current rows of its period (``served_pack_options``; ``tkg_mla_block`` ``own_state_stack``): 12 TensorE passes in place
of 24 for each pass of the C = 16 verify, and for the one pass of the C = 8 verify (k7b2). Bitwise the same (the state
copy adds +0.0, so a zero stays +0). It loads its own rows, so a form here takes no ``PASS_ROWS_AHEAD_FORM_SET`` rows."""

FP8_PV_LOOKAHEAD_K7B2 = True
"""Kill switch (default True) of the k7b2 entry of ``FP8_PV_LOOKAHEAD_FORMS``: False takes k7b2 out of the table, and
the FP8 P@V pass of the C = 8 verify then keeps the loops, the p2 stage tile and the plan of each layer (the program
without the lookahead); k7 keeps the table's options."""
FP8_PV_K7B1 = True
"""Kill switch (default True) of the k7b1 (C = 4, one sequence of 8 tokens) entries of ``FP8_PV_FORMS``,
``FP8_PV_FORM_SET`` and ``FP8_PV_LOOKAHEAD_FORMS``: False takes k7b1 out of all three, and the C = 4 verify then keeps
the program without the FP8 P@V pass (the packed p2 loop with BF16 P@V) for every call; ``check_fp8_pv`` then refuses a
given ``mla_fp8_pv`` True."""
FP8_PV_LOOKAHEAD_K7B1 = True
"""Kill switch (default True) of the k7b1 entry of ``FP8_PV_LOOKAHEAD_FORMS``: False takes k7b1 out of the table, and
the FP8 P@V pass of the C = 4 verify then keeps the loops, the p2 stage tile and the plan of each layer; the other
forms keep the table's options. It acts only with the FP8 P@V pass of k7b1."""
FP8_PV_LOOKAHEAD_K3B2 = True
"""Kill switch (default True) of the k3b2 entry of ``FP8_PV_LOOKAHEAD_FORMS``: False takes k3b2 out of the table, and
the FP8 P@V pass of the C = 8 k3b2 verify (2 sequences of 4 tokens, one pass of 8 rows) then keeps the loops, the p2
stage tile and the plan of each layer (the program without the lookahead); k3, k7, k7b2 and k7b1 keep the table's
options. It acts only with the FP8 P@V pass of k3b2, which the call gives (k3b2 is in ``FP8_PV_FORMS`` but not
``FP8_PV_FORM_SET``). Q = 4 already takes the lookahead on the k3 form, and ``fp8_pv_chunk_ahead`` needs 2
sequences, which k3b2 has."""
FP8_PV_LOOKAHEAD_FORMS = (
    ("k3", "k7")
    + (("k7b2",) if FP8_PV_LOOKAHEAD_K7B2 else ())
    + (("k7b1",) if FP8_PV_K7B1 and FP8_PV_LOOKAHEAD_K7B1 else ())
    + (("k3b2",) if FP8_PV_LOOKAHEAD_K3B2 else ())
)
"""The ``P2_FORMS`` forms whose FP8 P@V pass takes ``FP8_PV_LOOKAHEAD_OPTIONS`` and ``FP8_PV_LOOP_OPTIONS`` by default
(``served_pack_options``): the trip counts of each sequence's stream loops before its chunk, the item rows and the next
stream body's first pages before each loop boundary, the next chunk's pages ahead, and each 0/1-trip stream loop as a
forward then-only IF, nested in the IFs of its zero-together groups. The same pages and the same bits (R0 bitwise at
16K, 128K and 1M on the 12-layer C = 16 verify). k3 and k7 (C = 16), and, while ``FP8_PV_LOOKAHEAD_K7B2`` is True, k7b2
(C = 8, one pass of 16 rows): there the stage-free setup of ``FP8_PV_LOOP_OPTIONS`` also gives the lookahead's page
buffers their SBUF room (the pass alone leaves ~3 KB a partition at 93 layers, with the set ~13 KB). While
``FP8_PV_LOOKAHEAD_K7B1`` is True, k7b1 (C = 4, one sequence: one pass of 8 rows) takes the table too. With one
sequence a pass the set has no next-sequence chunk pages (``fp8_pv_loop`` ``chunk_ahead`` needs 2 sequences) and one
step store. While ``FP8_PV_LOOKAHEAD_K3B2`` is True, k3b2 (C = 8, 2 sequences of 4 tokens: one pass of 8 rows) takes
the table too, with the k3-form value form (``FP8_PV_VALUE_FORM_BY_FORM``: 3 groups of Q = 4): it has 2 sequences, so
the chunk-ahead pages hold."""

NULL_SKIP_K3B2 = True
"""Kill switch (default True) of the k3b2 entry of ``NULL_SKIP_FORM_SET``: False takes k3b2 out of the set (the
program without the null skip for k3b2); k7b2 keeps it. The chunk of 4 items (``CHUNK_ITEMS``) and its p = (-g) mod 4
null items are per key group, independent of Q, so the ordering holds for the k3b2 2 sequences of Q = 4 as it does for
the k7b2 2 sequences of Q = 8. ``NULL_SKIP_MODE`` 2 needs the lookahead and the "ifs_grouped" stream, which k3b2 takes
with ``FP8_PV_LOOKAHEAD_K3B2`` (it falls back to the two-IF form without them). It is the FP8 P@V chunk order only, not
the batched cache write, so it needs no ``mla_null_rows_once`` (k3b2 keeps the per-row cache writes)."""
NULL_SKIP_FORM_SET = ("k7b2",) + (("k3b2",) if NULL_SKIP_K3B2 else ())
"""The ``FP8_PV_FORMS`` forms whose FP8 P@V call takes ``fp8_pv_null_skip`` (``NULL_SKIP_MODE``) by default
(``served_pack_options``; ``tkg_mla_block`` ``fp8_pv_null_skip``): each chunk's real items before its null items, and a
1-item chunk for a sequence with 3 null items (every sequence at 16K) in place of the 4-item chunk, in forward then-only
IFs on the valid count (the same terms in the same order, the same bits up to the sign of an exact-zero numerator).
k7b2 (C = 8), and, while ``NULL_SKIP_K3B2`` is True, k3b2 (C = 8, Q = 4). Empty is the kill switch; a given
``fp8_pv_null_skip`` wins."""

ATTN_RES_STT_FORM_SET = ("k3", "k7")
"""The ``SERVED_FORMS`` forms whose AttnRes tensor-sums mixes take ``attn_res_scalar_products_stt`` (``decode_block``;
``common._mix_entries`` ``stt``) with their served ``attn_res_scalar_products``: each VectorE row of a bank entry after
the first takes one VectorE multiply-add in place of its multiply and its part of the add. MEASURED on a 93-layer
C = 16 1M profile: about 682 mix entries a step, VectorE-paced (a
multiply of the VectorE rows, then an add of all rows), with ScalarE's row products beside it.
MEASURED on trn3 hardware: the multiply-add rounds the product to FP32 before the add and keeps the
IEEE sign of zeros, so the same bits. The 12-layer verify mixes at layers 0-11 have one bank entry, so those programs do
not change. k7 (C = 16) and k3 (C = 16): MEASURED on a 12-layer k3 C = 16 verify at layers 72-83, a shorter p50 step,
bitwise. k3b2 stays out: no 12-layer form measures it (its 8 rows refuse the layer-range split, and layers 0-11 change
nothing). Empty is the kill switch."""

MOE_RING_QUEUE_ORDER_FORM_SET = ("k3", "k3b2", "k7b2")
"""The ``SERVED_FORMS`` forms whose KDA-led expert ring (``moe_expert_prefetch`` ``kda_after_in_proj``) keeps each
queue's later ring loads after its first one in the schedule (``decode_block.decode_block_body``
``moe_prefetch_kda_queue_order``), so no ring load shares the DMA rings with the KDA input projection's last weight
loads (MEASURED on the 12L C = 8 verify: a shorter KDA attention a layer and a shorter step at 16K). k3b2 (C = 8)
and k3 (C = 16) take the same order with the KDA-led ring of their renders. k7b1 (C = 4) gains nothing (its spans are
the same in either order) and stays out. The values do not change."""

MOE_RING_BEFORE_STORES_FORM_SET = ("k7b2",)
"""The ``SERVED_FORMS`` forms whose KDA-led expert ring (``moe_expert_prefetch`` ``kda_after_in_proj``) goes out
before the KDA body's slot state store chain on the in-order Sync queue (``decode_block.decode_block_body``
``moe_prefetch_kda_before_slot_stores``): the first scored token's slot state stores follow every ring load in the
schedule. Else most ring loads go out between the stores, one a store, and each store shares the DMA rings with ring
bytes (its flight takes longer on a 93-layer C = 8 1M profile), and the stores are a chain. k7b2 (C = 8) only: C = 16
serves no KDA-led ring, and k7b1 (C = 4) and the k3 forms keep their programs. Empty is the kill switch. The values do
not change."""

AUX_STORE_QUEUES = (
    {"engine": nisa.engine.sync, "dge_mode": nisa.dge_mode.hwdge},
    {"engine": nisa.engine.scalar, "dge_mode": nisa.dge_mode.hwdge},
)
"""The DMA queues of the aux row stores of an ``AUX_STORE_FORM_SET`` form (``common.store_hT_rows`` ``queues``): row
t stores on queue t mod 2, the SP and ACT hardware DGE queues, in place of the one default queue of every row. The
same bytes."""
AUX_STORE_RING_DEPTH = 8
"""The aux row ring of an ``AUX_STORE_FORM_SET`` form that takes the ring (``aux_row_ring``): 8 staging tiles in place
of ``AUX_RING_DEPTH``, so the copy of row t waits for the store of row t - 8."""
AUX_STORE_FORM_SET = ("k3", "k7", "k7b2")
"""The ``SERVED_FORMS`` forms whose aux row stores (``store_aux``, the drafter's prefix tap) take ``AUX_STORE_QUEUES``
and ``AUX_STORE_RING_DEPTH``. MEASURED on the 93L C = 16 1M profile: the head after each aux layer runs longer,
paced by the rows' serial stores on one queue (each row's copy waits for the store of row t - 4). The same bytes, so
the same bits. k7 and k3 (C = 16) and k7b2 (C = 8). On the 93L C = 8 1M profile the KDA layer after each of the 3 aux
layers runs longer than its neighbours: its chain goes through the 16 serial row stores on the GpSimd queue,
ahead of the layer's GpSimd weight loads. (On the 12L C = 8 verify, which has no deep aux layer, it measured slower at 16K
and 1M.) k3b2 (k3 C = 8) is not in the set and keeps the default queue and ring; k7b1 (C = 4) stays out. Empty is
the kill switch."""

MOE_FIRST_ADD_FORM_SET = ("k3", "k7")
"""The ``SERVED_FORMS`` forms whose MoE static expert calls start the FP32 accumulator at expert 0 by default
(``served_defaults`` moe glue ``first_add_zero``; ``moe_tkg_decode`` ``first_add_zero``: no zero fill, expert 0 adds a
scalar +0.0, the same FP32 add, so the same bits). k7 and k3 (C = 16) only: k7b2 and k3b2 (C = 8) keep the zero fill,
as the k7b2 12L profile showed the expert loop's down-projection matmuls waiting longer without it, and k7b1 (C = 4) keeps it (its
program does not change). Empty is the kill switch; a given ``first_add_zero`` wins."""

MOE_OUT_PROJ_PRIORITY_BY_FORM = {"k3": 3, "k3b2": 3, "k7": 3}
"""The trn3 DMA QoS level of the MXFP8 MoE out-projection loads by served form (``served_defaults`` moe glue
``out_proj_priority``): the lowest level, so the packets of the DP8 and TP32 router-row all-gathers, which share the DMA
engines with these early loads, go first (on trn3 the DMA QoS level sets the packet size; a bigger packet wins). The loads
have about the expert loop of slack, and the bytes do not change. k7 and k3 (C = 16) and k3b2 (C = 8): k7b2 (C = 8)
and k7b1 (C = 4) keep the critical level (k7b1 keeps its program). Empty is the kill switch; a given ``out_proj_priority`` wins."""

MOE_OUT_PROJ_BEFORE_IN_PROJ_FORM_SET = ("k7b2",)
"""The ``SERVED_FORMS`` forms whose MoE glue takes ``out_proj_before_in_proj`` by default (``served_defaults``), with
``out_proj_early``: the early MXFP8 out-projection loads trace before the shared in-projection matmuls, so the SBUF
allocator gives their tiles bytes apart from the in-projection tiles, and the loads wait for no matmul on those bytes
(MEASURED on the C = 8 1M profile: in most layers they waited for the in-projection matmuls, after the DP8 router-row
all-gather trigger). The same loads of the same bytes, so the same bits. k7b2 (C = 8) only: k7 (C = 16) already places
the two apart. Empty is the kill switch; a given ``out_proj_before_in_proj`` wins."""

MOE_SCALE_FILL_AFTER_PREFETCH_FORM_SET = ("k7b2",)
"""The ``SERVED_FORMS`` forms whose MoE glue takes ``scale_fill_after_prefetch`` by default (``served_defaults``), with
``moe_latent_gather_mx``: the GpSimd zero fill of the gathered code scales follows the head's weight prefetch DMAs in the
schedule (MEASURED on the C = 8 1M profile: a GpSimd zero fill that waits for VectorE or TensorE held those DMAs in the
GpSimd queue, at layer 45 and in the MLA-kind layers). Schedule edges only, so the same bits. k7b2 (C = 8)
only. Empty is the kill switch; a given ``scale_fill_after_prefetch`` wins."""

TAIL_ZERO_EARLY_FORM_SET = ()
"""The ``SERVED_FORMS`` forms whose MoE glue takes ``tail_zero_early`` by default (``served_defaults``): with the
``hot_last`` compaction, ``ungather_skip`` and no ``tail_before_guard``, the tail's token-row gather tiles take their zero
fills before the later slot tile's guard, so the first gather after the guard's branch waits for no fill. The same zero
bytes, so the same bits. Empty is the kill switch; a given ``tail_zero_early`` wins. Empty: on hardware the C = 16
verify ran slower with it."""

NULL_SKIP_MODE = 2
"""The ``fp8_pv_null_skip`` value of a ``NULL_SKIP_FORM_SET`` form: 2 (the merged one-IF chunk form; it needs the
lookahead and the "ifs_grouped" stream, which k7b2 takes, and falls back to the two-IF form True without them)."""

FP8_PV_LOOKAHEAD_OPTIONS = {
    "fp8_pv_lookahead": 4,
    "fp8_pv_trips_ahead": True,
    "fp8_pv_chunk_ahead": True,
    "fp8_pv_stream": "ifs_grouped",
}
"""The packed-loop options of the FP8 P@V boundary lookahead (``tkg_mla_block`` ``_PACKED_PRIOR_OPTIONS``). With them each
0/1-trip stream loop runs as a forward then-only IF on its trip register (no induction register, no back edge), nested
in an IF on any stream body and an IF on any body but the last, so the zero-trip bodies of a short context take one
branch (``fp8_pv_stream`` "ifs_grouped"; the same bits). A given ``fp8_pv_stream`` wins: "ifs" (the flat IFs) or
"loops" (the 0/1-trip loops) is the kill switch."""

FP8_PV_QK_GROUP_K7B2 = True
"""Kill switch (default True) of the k7b2 entry of ``FP8_PV_QK_GROUP_FORMS``: False takes k7b2 out of the table, and
the FP8 P@V pass of the C = 8 verify then issues each page's MX QK right before its rope matmul (the program
without the QK grouping); k7 keeps the table's options."""
FP8_PV_QK_GROUP_K3B2 = True
"""Kill switch (default True) of the k3b2 entry of ``FP8_PV_QK_GROUP_FORMS``: False takes k3b2 out of the table, and
the FP8 P@V pass of the C = 8 k3b2 verify then issues each page's MX QK right before its rope matmul (the program
without the QK grouping); k3, k7 and k7b2 keep the table's options. k3b2 takes the k3-form grouping of 2 pages
(``FP8_PV_QK_GROUP_OPTIONS_BY_FORM``), as its 3 groups of Q = 4 match the k3 form; the group is a schedule of the same
matmuls, so the same bits."""
FP8_PV_QK_GROUP_FORMS = (
    ("k3", "k7") + (("k7b2",) if FP8_PV_QK_GROUP_K7B2 else ()) + (("k3b2",) if FP8_PV_QK_GROUP_K3B2 else ())
)
"""The ``P2_FORMS`` forms whose FP8 P@V pass takes ``FP8_PV_QK_GROUP_OPTIONS`` by default (``served_pack_options``):
the MX QK of the 4 pages of a sub-slot, then their rope matmuls (with 2, the P@V of an earlier sub-slot runs between
the two halves). The same matmuls in each PSUM accumulation, so the same bits."""

FP8_PV_QK_GROUP_OPTIONS = {"fp8_pv_qk_group": 4}
"""The packed-loop option of the FP8 P@V QK grouping (``tkg_mla_block`` ``_PACKED_PRIOR_OPTIONS``) of k7."""
FP8_PV_QK_GROUP_OPTIONS_BY_FORM = {
    "k3": {"fp8_pv_qk_group": 2},
    "k7": FP8_PV_QK_GROUP_OPTIONS,
    "k7b2": FP8_PV_QK_GROUP_OPTIONS,
    "k3b2": {"fp8_pv_qk_group": 2},
}
"""The QK grouping of each ``FP8_PV_QK_GROUP_FORMS`` form: k3 and k3b2 keep 2 pages (3 groups of Q = 4), k7 and k7b2
take 4."""

FP8_PV_PREP_LATE = True
"""Kill switch (default True) of ``FP8_PV_PREP_LATE_FORM_SET``: False empties the set, and the FP8 P@V pass of the
C = 16 verify runs each sub-slot's item preps before its scores (the program without the late preps). k7 (C = 16)
only: the C = 8 verify keeps the program without the late preps. k3 stays out: its pass makes the sub-slot's last
TensorE instruction inside a dynamic loop, and the late-prep edge cannot depend on it from outside that loop (trace
error)."""
FP8_PV_PREP_LATE_FORM_SET = ("k7",) if FP8_PV_PREP_LATE else ()
"""The ``P2_FORMS`` forms whose FP8 P@V pass takes ``FP8_PV_PREP_LATE_OPTIONS`` by default (``served_pack_options``):
each sub-slot's item preps (the key and rope transposes and their PSUM copies) after its scores and P@V. TensorE then
runs that work while the previous prep's ScalarE key-word copy frees the transpose bank, where the next item's key
transposes waited, in the C = 8 and C = 16 MLA layers at 1M (12L profiles, MEASURED). The same
instructions on the same operands, so the same bits."""
FP8_PV_PREP_LATE_OPTIONS = {"fp8_pv_prep_late": True}
"""The packed-loop option of the FP8 P@V late preps (``tkg_mla_block`` ``_PACKED_PRIOR_OPTIONS``)."""

FP8_PV_PAGE_BUFFERS = True
"""Kill switch (default True) of ``FP8_PV_PAGE_BUFFERS_FORM_SET``: False empties the set, and the FP8 P@V pass of the
C = 8 verify keeps 2 page buffers (the program without the third page buffer)."""
FP8_PV_PAGE_BUFFERS_K3 = True
"""Kill switch of the k3 and k3b2 entries of ``FP8_PV_PAGE_BUFFERS_FORM_SET``: False keeps 2 page buffers on the
k3 forms."""
FP8_PV_PAGE_BUFFERS_FORM_SET = (
    (("k7b2",) + (("k3", "k3b2") if FP8_PV_PAGE_BUFFERS_K3 else ())) if FP8_PV_PAGE_BUFFERS else ()
)
"""The ``P2_FORMS`` forms whose FP8 P@V pass takes ``FP8_PV_PAGE_BUFFERS_OPTIONS`` by default (``served_pack_options``):
a third page buffer, so an item's page load waits for the last read of the item three back (not two) and issues one
item earlier. With 2 buffers the load of item k waits for item k - 2's last value-word piece on VectorE, which runs
behind the pipeline's quantizes: on the served MX 1M profiles (12L, MLA layers 2 / 6 / 10, MEASURED) most of the C = 16
page DMAs trigger right after a value-word copy, and TensorE idles in each C = 16 MLA layer (and each C = 8
one) before the first reader of a page that had not landed. k7b2 (C = 8) only: on the 12L C = 16 MX 1M verify (MEASURED)
the page waits fall in each MLA layer, but the compiler then packs the key-word transposes into slower
slots and the layer is slower, while C = 8 gains time in each layer. The same pages reach the same
instructions, so the same bits."""
FP8_PV_PAGE_BUFFERS_OPTIONS = {"fp8_pv_page_buffers": 3}
"""The packed-loop option of the FP8 P@V third page buffer (``tkg_mla_block`` ``_PACKED_PRIOR_OPTIONS``)."""

FP8_PV_LOOP_OPTIONS = {
    "fp8_pv_value_form": 5,
    "fp8_pv_page_runs": 2,
    "fp8_pv_stage_free": True,
    "fp8_pv_step_store": True,
}
"""The packed-loop options of the FP8 P@V loop that an ``FP8_PV_LOOKAHEAD_FORMS`` form takes with the lookahead
(``served_pack_options``): the value-word copy in 4 pieces in the word order of form 0 (``fp8_pv_value_form`` 5) and the
page loads of 2-page runs (``fp8_pv_page_runs`` 2). The same bytes and the same bits (MEASURED on one core at 1M: a
shorter MLA pass). ``fp8_pv_stage_free``: the FP8 P@V setup with no p2 stage tile; ``fp8_pv_step_store``: the item plan
of each pass built once a step (the same bits)."""
MLA_FRONT_TAIL_FORMS = ("k3", "k7")
"""The ``P2_FORMS`` forms whose p2 set takes ``MLA_FRONT_TAIL_OPTIONS`` (``served_pack_options``)."""
MLA_FRONT_TAIL_OPTIONS = {"q_pass_rows": True, "q_prep_act": True}
"""The packed-loop options of the MLA front and tail (``tkg_mla_block`` ``_PACKED_PRIOR_OPTIONS`` ``q_pass_rows`` and
``q_prep_act``; bitwise) that the p2 set of an ``MLA_FRONT_TAIL_FORMS`` form takes; a given key wins. The state sums
on the ScalarE queue (``state_sums_act``) come from ``TAIL_ORDER_OPTIONS``. Empty: every form gives the program
without these options."""
Q_FP8_P2_FORM_SET = ()
"""The ``P2_FORMS`` forms whose p2 loop takes ``q_fp8`` by default (``served_pack_options``): the query of the call
rounded once to e4m3 at unit scale (saturated at 448) before its all-gather, which then moves half the bytes, and every
query consumer takes the e4m3 query, the query form of the GPU decode (accuracy class). Empty (the default): every form
keeps the BF16 query unless the call's ``mla_k1_pack_options`` give ``q_fp8``."""
Q_MX_GATHER_FORM_SET = ("k3", "k7")
"""The ``FP8_PV_FORMS`` forms whose FP8 P@V call takes ``q_mx_gather`` by default (``served_pack_options``): each rank
makes the MX query words and E8M0 scales of its own heads, and their own-key scores, before the query all-gather, which
then moves 1,980 bytes a row in place of 3,456 (bitwise the same, ``tkg_mla_block`` ``q_mx_gather``), 2,316 with the scales
in the word records (``q_mx_scale_words``, served with it: one DMA loads a pass's words and scales). The form has no BF16
gather tiles, so a call that takes the key runs with ``mla_pass_q_prefetch`` 0 and ``mla_q_gather_buffers`` 1 whatever
the caller gives (the verify prints the override)."""
Q_MX_SCALE_WORDS_FORM_SET = ("k3", "k7")
"""The ``Q_MX_GATHER_FORM_SET`` forms whose MX query gather also takes ``q_mx_scale_words`` by default: k7 and k3."""
Q_MX_GATHER_ONE_PASS_FORM_SET = ("k7b2",)
"""The ``FP8_PV_FORMS`` forms of one MLA pass whose FP8 P@V call takes the MX query gather with the scales in the word
records (``q_mx_gather`` with ``q_mx_scale_words``, ``served_pack_options``) while ``Q_MX_GATHER_ONE_PASS`` is True, and
none of the pass keys that ride ``Q_MX_GATHER_FORM_SET`` (those order the loads and stores of a second pass): k7b2
(C = 8, one pass of 16 rows). Each rank makes the MX query words, E8M0 scales and own-key scores of its own heads before
the query all-gather, which then moves 2,316 bytes a row in place of 3,456, and the pass loads them as bytes in place
of the transposes, copies and quantizes of the gathered BF16 query: the same bits (``tkg_mla_block`` ``q_mx_gather``). A
given key wins."""
Q_MX_GATHER_ONE_PASS = True
"""Kill switch (default True) of ``Q_MX_GATHER_ONE_PASS_FORM_SET``: False keeps the BF16 query gather (the program
without ``q_mx_gather``) on every form."""
FRONT_ONE_PASS_FORM_SET = ("k7b2",)
"""The one-pass forms whose FP8 P@V call with the MX query gather takes ``FRONT_ONE_PASS_OPTIONS`` by default while
``FRONT_ONE_PASS`` is True (``served_pack_options``; a given key wins): the front-order keys of the two-pass ``Q_MX_GATHER_FORM_SET`` forms that
act on one pass too. On a 93-layer C = 8 1M profile (MEASURED) the first QK of a k7b2 pass came later after the start
of the query all-gather than the first pass of k7: the current-row load waited on the SP queue
for the query copies, the reference DMAs behind it waited for the query's own-key references, and the step info
behind them gated the loop's register loads and its first IF. ``own_rows_ahead`` loads the current rows before the
query store, ``fp8_pv_pages_first`` puts the setup's chunk tables on the heap and keeps the frame rows' tiles (the chunk
and first page loads do not wait for the readers of freed bytes), and ``fp8_pv_refs_act`` issues the reference
DMAs on the ScalarE queue. With the rows loaded ahead, ``own_state_first`` traces the pass's own-key state before its
query loads, so it does not wait behind the query copies on VectorE. The same instructions on the same bytes, so the
same bits."""
FRONT_ONE_PASS_OPTIONS = {
    "own_rows_ahead": True,
    "fp8_pv_pages_first": True,
    "fp8_pv_refs_act": True,
    "own_state_first": True,
}
"""The packed-loop options of the one-pass MLA front order (``tkg_mla_block`` ``_PACKED_PRIOR_OPTIONS``)."""
FRONT_ONE_PASS = True
"""Kill switch (default True) of ``FRONT_ONE_PASS_FORM_SET``: False keeps the MLA front order without the one-pass order on every
form."""
TAIL_ORDER_FORM_SET = ("k3", "k7")
"""The forms whose FP8 P@V call takes ``TAIL_ORDER_OPTIONS`` by default (``served_pack_options``): the tail gate exps
after the last o_proj weight DMA and the state sums stores on the ScalarE queue (``tkg_mla_block`` ``tail_o_proj_first`` and
``state_sums_act``; the same instructions and bytes, so the same bits). A given key wins."""
TAIL_ORDER_OPTIONS = {"tail_o_proj_first": True, "state_sums_act": True}
"""The packed-loop options of the MLA tail order (``tkg_mla_block`` ``_PACKED_PRIOR_OPTIONS``)."""
PLAN_FIRST_FORM_SET = ("k3", "k7")
"""The forms whose FP8 P@V call takes ``fp8_pv_plan_first`` by default (``served_pack_options``): the input loads of the
item plan of each pass and its HBM round trip go on the ACT hardware DGE queue, not behind the query loads of the SP
queue (``tkg_mla_block`` ``fp8_pv_plan_first``; the same plan, so the same bits). A given key wins."""
PLAN_REDUCE_K3B2 = True
"""Kill switch (default True) of the k3b2 entry of ``PLAN_REDUCE_FORM_SET``: False takes k3b2 out of the set (the
program without the plan reduce for k3b2); k3, k7 and k7b2 keep it. k3b2 (C = 8, S = 2) mirrors k7b2 (S = 2): no
exposed add chain (defensive hardening), the same small-integer terms, so the same item rows and the same bits. Q = 4
already takes the reduce on the k3 form."""
PLAN_REDUCE_FORM_SET = ("k3", "k7", "k7b2") + (("k3b2",) if PLAN_REDUCE_K3B2 else ())
"""The ``FP8_PV_FORMS`` forms whose FP8 P@V call takes ``fp8_pv_plan_reduce`` by default while ``FP8_PV_PLAN_REDUCE`` is
True (``served_pack_options``; ``tkg_mla_block`` ``fp8_pv_plan_reduce``): each sum of the item plan over the S sequences of
a pass is one reduce, not a memset of a tile and a chain of adds into that tile, which the compiler takes as an
order-free reduction (no wait between the adds). k3 (one pass of S = 4 sequences: the longest chains, and the form
where this class gave a wrong stream start on device before the starts took one reduce each), k7 (C = 16), k7b2
(C = 8; S = 2, no exposed chain: defensive hardening) and, while ``PLAN_REDUCE_K3B2`` is True, k3b2 (C = 8; S = 2,
Q = 4; defensive hardening too). The same small-integer terms, so the same item rows and the same bits. A given key
wins."""
MLA_C16_KEYS_FORM_SET = ("k7",)
"""The forms whose verify call keeps a given ``MLA_C16_KEYS`` pack option (``served_pack_options``): k7 (C = 16).
Another form drops the key, so its program does not change (the C = 8 k7b2 chunk with ``fp8_pv_null_skip`` refuses
``fp8_pv_chunks_first``)."""
MLA_C16_KEYS = ("fp8_pv_chunks_first",)
"""The packed-loop options (``tkg_mla_block`` ``_PACKED_PRIOR_OPTIONS``) that only an ``MLA_C16_KEYS_FORM_SET`` form keeps
(``served_pack_options``)."""
FP8_PV_CHUNKS_FIRST = True
"""Kill switch (default True) of ``FP8_PV_CHUNKS_FIRST_FORM_SET``: False keeps the chunk of each sequence inside its IF
tree on every form (the program without ``fp8_pv_chunks_first``). A given ``fp8_pv_chunks_first`` wins."""
FP8_PV_CHUNKS_FIRST_FORM_SET = ("k7",) if FP8_PV_CHUNKS_FIRST else ()
"""The forms whose FP8 P@V call with the lookahead and the grouped IF stream (and no null skip) takes
``fp8_pv_chunks_first`` by default (``served_pack_options``; ``tkg_mla_block`` ``fp8_pv_chunks_first``): the chunk of every
sequence runs ahead of the first sequence's IF tree, so a join leaves no other sequence's chunk behind its fence. k7
(C = 16). The same bits."""
FP8_PV_PLAN_REDUCE = True
"""Kill switch (default True) of ``PLAN_REDUCE_FORM_SET``: False turns the set's default off in every form. A caller
that gives ``fp8_pv_plan_reduce`` True in its pack options still turns it on for that call."""
STATE_SUMS_PRE_IF_K3 = True
"""Kill switch of the k3 entry of ``STATE_SUMS_PRE_IF_FORM_SET``: False keeps the sum blocks of k3 after the redo IF."""
STATE_SUMS_PRE_IF_FORM_SET = ("k7", "k7b2") + (("k3",) if STATE_SUMS_PRE_IF_K3 else ())
"""The forms whose FP8 P@V call takes ``state_sums_pre_if`` by default while ``STATE_SUMS_PRE_IF`` is True
(``served_pack_options``; ``tkg_mla_block`` ``state_sums_pre_if``, with ``state_sum_blocks``): the last pass of the call
stores its sum blocks right after its uncapped FP8 P@V loop, before the capless check and its redo IF (and again at
the end of the redo body), so the sums reduce-scatter's sources land while the check and the IF run. The same words in
the same HBM tensor before the reduce-scatter, so the same bits. k7 (C = 16: two passes, the call's sum blocks) and
k7b2 (C = 8, with the one-pass sum blocks). A given key wins."""
STATE_SUMS_PRE_IF = True
"""The kill switch of ``STATE_SUMS_PRE_IF_FORM_SET``: False keeps the sum blocks after the redo IF on every form."""
SUM_BLOCKS_PASS_K3 = True
"""Kill switch of the k3 entry of ``SUM_BLOCKS_PASS_FORM_SET``: False keeps the per-run sum stores on k3. k3b2 (8
rows) is not added: the kernel takes the one-pass blocks for a pass of 16 rows only."""
SUM_BLOCKS_PASS_FORM_SET = ("k7b2",) + (("k3",) if SUM_BLOCKS_PASS_K3 else ())
"""The forms whose FP8 P@V call of one pass takes ``state_sum_blocks_pass`` by default while ``SUM_BLOCKS_PASS`` is True
(``served_pack_options``; ``tkg_mla_block`` ``state_sum_blocks_pass``, with the BF16-o state and its padded sums pair,
``cc_pad256``): the pass stores its sums in whole 64-byte lines (6 DMAs) in place of 12 8-byte read-modify-write DMAs,
and its tail reads them from the reduced blocks. The same sums in other words of each rank's part of the sums
reduce-scatter, so the same bits. k7b2 (C = 8: one 16-row pass, the per-pass state reduce-scatter, cc_pad256). A
given key wins."""
SUM_BLOCKS_PASS = True
"""The kill switch of ``SUM_BLOCKS_PASS_FORM_SET``: False keeps the per-run sum stores on every form."""
DENSE_TAIL_K3 = True
"""Kill switch of the k3 and k3b2 entries of ``DENSE_TAIL_FORM_SET``: False keeps the head-major tail on the k3
forms."""
DENSE_TAIL_FORM_SET = ("k7b2",) + (("k3", "k3b2") if DENSE_TAIL_K3 else ())
"""The forms whose FP8 P@V call of one pass takes ``state_dense_tail`` by default while ``DENSE_TAIL`` is True
(``served_pack_options``; ``tkg_mla_block`` ``state_dense_tail``, with the BF16-o state): the output tail loads its local
state with one row of a head on each partition, as the call-level tail of k7 does, so the normalize runs on 48
partitions in place of 3 and the latent takes 4 transposes in place of 64. The same multiply on the same values, so the
same bits. k7b2 (C = 8: one 16-row pass, the per-pass state reduce-scatter). A given key wins."""
DENSE_TAIL = True
"""The kill switch of ``DENSE_TAIL_FORM_SET``: False keeps the head-major tail on every form."""

STEP_ITEMS_FORMS = ("k3", "k3b2", "k7", "k7b2")
"""The served forms whose packed k1 loop builds its item plans and item rows once for the MLA layers of the call by
default (``served_defaults`` ``mla_step_items``; ``hoist_plans`` with ``step_items``). Every MLA layer of a verify step
reads the block table of the first one (the runner stops a step whose MLA layers differ), so the plans and the rows are
the same bytes and the state is bitwise the same; the later layers skip the item-row gathers and the run-scale matmul of
their loop entry. k7: a 2-node served C = 16 measurement gave the same outputs and a shorter step at 16K, 128K and 1M. k7b2 (C = 8)
takes the same form: without it each of its MLA layers builds the same plans and rows again."""
P2_ENTRY_ACT_FORMS = ("k3", "k3b2", "k7", "k7b2")
"""The ``P2_FORMS`` forms whose p2 loop takes ``p2_entry_act`` by default (``served_defaults`` ``mla_p2_entry_act``): the
loop-entry references and SBUF slot rows on the ScalarE queue and the narrowed stage fill, bitwise the same. k7: a
2-node served C = 16 measurement gave the same digests on all 256 ranks and a shorter step at 16K, 128K and 1M. k7b2 (C = 8)
takes the same form."""
PASS_Q_PREFETCH_FORM_SET = ()
"""The served forms whose k1 MLA call takes the query prefetch by default (``served_defaults`` ``mla_pass_q_prefetch``):
the first query gather tiles of the second sequence-major pass load on the ScalarE queue before the loop of the first
pass, bitwise the same. The tile count of a form is its ``PASS_Q_PREFETCH_TILES_BY_FORM`` value, else
``PASS_Q_PREFETCH_TILES``. No form yet: k7 (C = 16, two passes) waits for a served measurement."""
PASS_Q_PREFETCH_TILES = 2
"""The prefetch tiles of a ``PASS_Q_PREFETCH_FORM_SET`` form that ``PASS_Q_PREFETCH_TILES_BY_FORM`` does not name
(``tkg_mla_block.PASS_Q_PREFETCH_MAX_TILES``, the most that fit): the 12-layer k7 trace with the step items leaves 2,780
bytes a partition in the second pass's loop with 2 tiles."""
PASS_Q_PREFETCH_TILES_BY_FORM = {"k7": 1}
"""The prefetch tiles of each ``PASS_Q_PREFETCH_FORM_SET`` form that does not take ``PASS_Q_PREFETCH_TILES``: k7 takes 1.
With the per-token KDA tiles (``kda_token_tiles``) at k7, the persistent KDA slot conv sources take SBUF room, and the
12-layer k7 trace fails with "Stack out of memory" in the slot rows of the packed loop at 2 tiles. 1 tile leaves 2,688
bytes a partition free in both packed loops of that trace."""
Q_GATHER_BUFFER_FORM_SET = ()
"""The served forms whose k1 MLA call takes two query gather tiles in turn by default (``served_defaults``
``mla_q_gather_buffers`` 2): the loads of a gather tile do not wait for the transposes of the tile before, bitwise the
same. No form yet: k7 waits for a served measurement."""
TAIL_WEIGHTS_AFTER_RS_FORM_SET = ()
"""The served forms whose k1 MLA call loads its tail weights after the call-level state reduce-scatter by default
(``served_defaults`` ``mla_tail_weights_after_rs``): the weights load on the ScalarE queue after the numerator
reduce-scatter, so they do not delay it on the shared DMA engines, bitwise the same. No form yet: k7 (C = 16, the
call-level BF16-o state reduce-scatter) waits for a served measurement."""
PASS_SETUP_OFF_POOL_FORM_SET = ()
"""The served forms whose packed p2 loop takes ``setup_off_pool`` by default (``served_defaults``
``mla_pass_setup_off_pool``; ``served_pack_options``): the passes share the batched weight-sum offset tiles, and the
pass setup's sum-tile zero fills and stage TI offset copies run on VectorE, not GpSimd, bitwise the same. No form yet:
k7 (C = 16, two passes) waits for a served measurement."""
SLOT_ROWS_SBUF_FORM_SET = ()
"""The served forms whose packed p2 loop takes ``slot_rows_sbuf`` by default (``served_defaults``
``mla_slot_rows_sbuf``; ``served_pack_options``): every slot row of the loops reads the SBUF stage tile, not its HBM
store, bitwise the same. It needs ``stage_rows_sbuf``, so only a ``P2_SBUF_LEVER_FORMS`` form takes it. No form yet:
alone it measured slower on a 1-core k7 verify, so k7 waits for a served measurement with
``SKIP_STAGE_STORE_FORM_SET``."""
SKIP_STAGE_STORE_FORM_SET = ()
"""The served forms whose packed p2 loop takes ``skip_stage_store`` by default (``served_defaults``
``mla_skip_stage_store``; ``served_pack_options``): with ``slot_rows_sbuf`` no slot row reads the HBM store of the stage
rows, so each pass skips that store, bitwise the same. Each form here must also be in ``SLOT_ROWS_SBUF_FORM_SET``. No
form yet: k7 (C = 16, two passes) waits for a served measurement."""
PASS_ROWS_AHEAD_FORM_SET = ()
"""The served forms whose k1 MLA call loads the next pass's own-key current rows and query gather tiles before the
state stores of the pass before by default (``served_defaults`` ``mla_pass_rows_ahead``; ``mla_block_attention``
``pass_rows_ahead``), bitwise the same. No form yet: k7 (C = 16, two passes) waits for a served measurement."""
ACT_SPREAD_FORM_SET = ("k3", "k3b2", "k7", "k7b2", "k7b1")
"""The ``P2_FORMS`` forms whose p2 loop takes ``act_spread`` by default where its set has ``word_keys``
(``served_pack_options``): the ScalarE order of the word-key loop, the same instructions, bitwise the same. k7 and k7b2:
2-node served measurements at 1M gave the same outputs and a shorter step (C = 16 and C = 8); k7b1 takes the k7b2 set.
Empty: the switch is off on every form."""
QUERY_DMA_FORMS = ("k3", "k3b2", "k7", "k7b2")
"""The served forms whose k1 MLA call also gives the query path and the call-level state store / load the critical DMA
priority on the SP hardware DGE queue (``mla_critical_dma_priority`` "query"): the 8-token k7 and k7b2 forms, whose
gathered query loads a layer and call-level state DMAs otherwise wait on the default single-engine queue. The same
bytes."""
PAD_SKIP_FORMS = ("k3", "k3b2", "k7", "k7b2")
"""The served forms whose packed loop takes ``pad_skip`` with the runner's live rows by default: a pad sequence takes no
key group, and every live row keeps its bits. A caller's ``pad_skip`` wins."""
MX_QK_TRN3_MODES = {"off": None, "E": ("k3", "k3b2", "k7", "k7b2", "k7b1"), "RA": ("k3", "k7", "k7b2", "k7b1")}
"""The values of the 2-server verify's ``mx_qk``, each with the ``SERVED_FORMS`` forms that take it (None: every form).
None, the default, keeps the program of each form. The forms are those of ``SERVED_FORMS`` on the k1 loop path; "E" (the
8-token k7, k7b2 and k7b1 forms on the packed p2 loop: two sequence-major passes of 16 rows, or one of 16 or 8) and "RA"
(the CP32 rows form with ``SERVED_ROWS_OPTIONS``: the e4m3 query gather, the MX latent QK, V as the P@V stationary, the
page prefetch and the sequence setup ahead, with FP8 pages) are accuracy class (the e4m3 latent query of the GB300
decode):

| ``mx_qk`` | k3 | k3b8 / k3b2 | k7 | k7b2 / k7b1 |
|---|---|---|---|---|
| None | the default: with FP8 pages the CP32 rows form (``SERVED_ROWS_OPTIONS``, MX latent QK), else the packed loop | the packed loop | the packed p2 loop | the packed p2 loop |
| "off" | the packed loop (BF16 latent QK), also with FP8 pages | as None | as None | as None |
| "E" | refused | refused | the packed p2 loop with ``MX_QK_E_OPTIONS`` | as k7 (the k7b2 p2 set) |
| "RA" | the CP32 rows form with ``SERVED_ROWS_OPTIONS`` (needs FP8 pages) | refused | as k3 | as k3 |

The mode is authoritative over the k1 form keys that a caller renders (``mx_qk_keys``): "RA" takes the rows form (a
non-empty rows-options dict that the caller gives wins), "off" takes no rows form; under "E" and None the caller's keys
hold. A pack option that the caller gives wins; ``check_mx_qk`` refuses a mode that the form or the resolved keys do
not take."""
MX_QK_E_OPTIONS = {"mx_qk": "block32", "hoist_plans": True, "step_items": True}
"""The packed-loop options of ``mx_qk`` "E", on top of the k7 or k7b2 p2 set of ``served_pack_options``: the latent QK
of each sub-slot as one MX matmul on the word_keys key words with the e4m3 latent query in E8M0 scale blocks of 32
(``tkg_mla_paged._packed_key_loop`` ``mx_qk``; accuracy class), and the item plans and item rows of the call built once for
its MLA layers (``hoist_plans`` with ``step_items``; every MLA layer of a verify step reads the block table of the first
one, so the state is bitwise that of a plan built in each layer)."""
MX_QK_NATURAL_FORM_SET = ()
"""The ``MX_QK_TRN3_MODES["E"]`` forms whose ``mx_qk`` "E" set also takes ``mx_natural`` by default
(``served_pack_options``): the key words of the FP8 key rows as they are (word ``c``: features ``4c .. 4c + 3``, no
ScalarE byte permute in the loop, 4 an item) with the MX query words in that grouping (``tkg_mla_block`` ``mx_natural``).
The same products in other 4-value groups and partition order ("block32": scale blocks of 32 consecutive features), so
accuracy class against the E set without ``mx_natural``. Empty: the switch is off on every form; no form yet, it
waits for its measurements."""
FP8_PV_FORMS = ("k3", "k3b2", "k7", "k7b2") + (("k7b1",) if FP8_PV_K7B1 else ())
"""The forms whose "E" set can take the FP8 P@V pass (``served_pack_options`` ``fp8_pv``, given or by default): k7
and k3 (C = 16), k7b2 and k3b2 (C = 8), and k7b1 (C = 4) while ``FP8_PV_K7B1`` is True. Another "E" form keeps the
program without the FP8 P@V pass, also with a given ``mla_fp8_pv`` True, which ``check_fp8_pv`` then refuses."""
FP8_PV_K7B2 = True
"""Kill switch (default True) of the k7b2 entry of ``FP8_PV_FORM_SET``: False takes k7b2 out of the set, and the C = 8
verify then keeps the program without the FP8 P@V pass (the packed p2 loop with BF16 P@V) unless the call gives
``mla_fp8_pv`` True."""
FP8_PV_FORM_SET = (("k7b2",) if FP8_PV_K7B2 else ()) + (("k7b1",) if FP8_PV_K7B1 else ())
"""The ``FP8_PV_FORMS`` forms whose ``mx_qk`` "E" set also takes the FP8 P@V pass by default
(``served_pack_options`` ``fp8_pv``; ``tkg_mla_block`` ``packed_prior_options`` ``fp8_pv`` with ``mx_natural``): the scores
with the keys on partitions, MXFP8 P from ``nisa.quantize_mx`` and one MX P@V a sub-slot (accuracy class, the FP8 P of
the GPU decode). k7b2 (C = 8, one pass of 16 rows) while ``FP8_PV_K7B2`` is True, and k7b1 (C = 4, one pass of 8 rows
of one sequence) while ``FP8_PV_K7B1`` is True: the serving caller gives these verify calls no ``mla_fp8_pv``, so the set decides. k7
(C = 16) is not in the set: its served config gives ``mla_fp8_pv`` True. A given ``mla_fp8_pv`` wins (False gives the
program without the FP8 P@V pass)."""
NULL_ROWS_ONCE_FORMS = ("k3b2", "k7b2")
"""The forms whose batched MLA cache write takes its null-page rows from one partition index column that the kernel
builds once (``mla_null_rows_once``), in place of one GpSimd iota a layer. The rows written do not change. k7b2 and
k3b2 (C = 8) take it; the other forms keep the program."""
BATCHED_CACHE_WRITE_FORMS = ("k3", "k3b8", "k3b2", "k7", "k7b2", "k7b1")
"""The served forms whose k1 MLA call writes its current rows with one vector-indirect DMA by default
(``mla_batched_cache_write``): the forms of 32 rows a replica (k3b8: 8 sequences of 4 tokens; k7: 4 sequences of 8
tokens), k7b2 (2 sequences of 8 tokens, 16 rows, the rows form), k7b1 (1 sequence of 8 tokens), k3 (4 sequences of 4
tokens, 16 rows) and k3b2 (2 sequences of 4 tokens). The values read do not change; only the null page holds other
bytes. In k7b2 the per-row writes ran
one after another on the entry path of the MLA loop (the compiler orders every two dynamic-offset writes of the
pool)."""
KDA_SLOT_DMA_FORMS = ("k3", "k3b8", "k3b2", "k7", "k7b2", "k7b1")
"""The served forms whose KDA verify with the state slots takes the slot DMA levers by default (``served_defaults``):
the conv-state gather and scatter, the slot conv scatter with persistent sources, the contiguous slot-page load and the
bf16 update rows. They were measured as one set on the served b32 verify (k3b8); k7, k7b2 and k7b1 (Q = 8, 4, 2 and 1
sequences) and k3 and k3b2 (Q = 4, 4 and 2 sequences) take the same set, with the persistent sources only with the per-token tiles. At Q = 8 the per-token tiles
need this set (``verify_model.check_kda_token_tiles``)."""
KDA_SMALL_AFTER_OUT_AR_FORM_SET = ()
"""The served forms whose KDA layers issue their GpSimd weight loads (``kda_w_small``, ``kda_params``, ``kda_w_out``)
after the load of the result of the feed-forward output all-reduce before them (the MoE one, or the dense layer's;
``decode_block.decode_block_body`` ``kda_small_after_out_ar``), not in its flight on the GpSimd queue that triggers it.
The same bits. None by default: it waits for served measurements. A given ``kda_small_after_out_ar`` (the entry argument)
wins over the set; ``KDA_SMALL_AFTER_OUT_AR`` is the kernel-side kill switch of the set."""
KDA_SMALL_AFTER_OUT_AR = True
"""Kill switch (default True) of ``KDA_SMALL_AFTER_OUT_AR_FORM_SET``: False turns the set's default off in every form. A
caller that gives ``kda_small_after_out_ar`` True still turns it on for that call."""
ATTN_RES_WEIGHT_ROW_FORMS = ("k3", "k3b2", "k7", "k7b2", "k7b1")
"""The served forms whose AttnRes mixes and bank sums take the weight row of their score weight by default
(``decode_block.decode_block_body`` ``attn_res_weight_rows``): one multiply and one sum for the weighted sums of all
rows of a candidate, the same bits (measured on trn3 and trn2 hardware). The other forms, the decode and the Trn2
programs keep the fused multiply and sum of each row."""
MOE_STATIC_ALT_TILE_DEFAULTS = {}
"""The served default of ``moe_static_alt_tiles`` by form (a form name: True or a ``STATIC_ALT_TILE_FORMS`` value): the
expert loop of a form in the table takes the gate sigmoid / up tanh, down PSUM copy and eviction tiles from two buffers
each, so that SBUF pressure from the other layers cannot stack them on one address (not with the glue's
``routed_compact``, whose slot form takes no alternating tiles). Empty: it is off by default on every form."""
ROUTED_COMPACT_SERVED_IN = ()
"""The served forms whose MoE glue takes ``routed_compact`` by default (``served_defaults``, with the ``dp_tp`` gather;
also with ``moe_latent_gather_mx``, which the compaction takes): a form of 256 gathered tokens a MoE layer (k7), whose
expert calls are token-tiled while about 30 of the tokens route to the 7 experts of a rank. Empty: no form takes it
by default. A glue that the caller gives wins (the serving caller gives its own: its ``routed_compact`` entry is the
switch). The compaction runs the slot form after the first slot tile, so it takes no alternating expert tiles."""
ROUTED_COMPACT_SERVED_VALUE = "hot_last"
"""The ``routed_compact`` value of the served glue of ``ROUTED_COMPACT_SERVED_IN``: ``hot_last`` (the hot-expert-last
slot order; a later slot tile runs only its active experts). The served glue also takes ``weights_before_compact``."""
COMPACT_ORDER_FORM_SET = ("k7",)
"""The served forms whose expert calls take the cross-expert order only with the routed-token compaction
(``routed_compact`` in the MoE glue; ``served_defaults`` ``moe_cross_expert_order``): k7, whose 256-token calls refuse
the order, while the first slot tile of the compaction is a static call of at most 128. In these forms a glue without
``routed_compact`` drops a given order, its transpose block and the 3 weight buffers that come with it, so the glue
alone is the off switch (the serving caller renders the order from ``CROSS_ORDER_FORMS``). Bitwise. While the form takes no served
compaction (not in ``ROUTED_COMPACT_SERVED_IN``) and ``tkg_moe_2srv.TOKEN_TILES_CROSS_ORDER``, a GIVEN order runs
on the token-tiled call instead (with its transpose block and 3 weight buffers); the default stays off."""
CROSS_ORDER_FORMS = ("k3", "k3b2") + tuple(
    form for form in COMPACT_ORDER_FORM_SET if form in ROUTED_COMPACT_SERVED_IN and moe_2srv_module.ROUTED_COMPACT_ON
)
"""The served forms whose expert calls take the cross-expert order by default (``served_defaults``
``moe_cross_expert_order``): k3 and k3b2 (one static call of 128 or 64 tokens), and the forms of
``COMPACT_ORDER_FORM_SET`` whose served glue compacts (``ROUTED_COMPACT_SERVED_IN`` with ``ROUTED_COMPACT_ON``; none
while that set is empty). A caller that reads this table passes the order itself."""
ROUTED_ROW_SUM_SERVED_IN = ("k3b8", "k3b2", "k7b2")
"""The served forms whose row sums follow the routed reduction and the routed store of the MoE layer before
(``moe_sums_after_routed``, ``attn_res_sums_after_moe_output``; ``served_defaults``): k3b8 (32 rows a replica on the
packed k1 loop), whose MLA-attention MoE layers run those sums ahead of the routed copies, and k7b2 and k3b2 (C = 8)
with the routed reductions of ``COMBO_ROUTED_REDUCES_BY_FORM`` (measured together at C = 8). Schedule edges only. The k3 rows form
keeps its program; k7 waits for its own measurement."""
KDA_STATES_AFTER_OUT_AR_FORM_SET = ("k7b2",)
"""The served forms whose next KDA layer's state loads follow the MoE output all-reduce trigger by default
(``kda_states_after_out_ar``; ``served_defaults``): k7b2 (C = 8) with the routed reductions of
``COMBO_ROUTED_REDUCES_BY_FORM``, not with ``kda_prework_ahead`` (whose state loads the KDA call does not trace).
Schedule edges only, so the values do not change. A given ``kda_states_after_out_ar`` wins (False turns it off)."""
COMBO_ROUTED_REDUCES_BY_FORM = {"k3b2": ("rs_ag",), "k7b2": ("rs_ag",)}
"""The routed reductions with which a form of ``ROUTED_ROW_SUM_SERVED_IN`` or ``KDA_STATES_AFTER_OUT_AR_FORM_SET`` takes
those keys by default; a form that is not here takes them with any routed reduction (for example k3b8). k7b2: the
``rs_ag`` tail of the served C = 8 render, which a C = 8 measurement of the set checked (bitwise at 4 position sets on both
servers, a faster 12-layer step). The other k7b2 tails keep their programs: the routed all-reduce, and
``rs_fold`` (no glue hooks). k3b2 takes the row sums of ``ROUTED_ROW_SUM_SERVED_IN`` with the rs_ag tail only (its
default routed all-reduce keeps its program); the k3 forms refuse ``kda_states_after_out_ar``."""
MLA_MLP_EARLY_STATE_RS_FORM_SET = ("k3", "k7")
"""The served forms whose MLA layers sum the bank rows of their MLP mix in the flight of the last state reduce-scatter of
the k1 call without the rows form (``attn_res_mla_mlp_early``; ``served_defaults``), with the kept bank rstd, the early
sums and the BF16-o state reduce-scatter (``mla_state_bf16_o``), which traces the hook. The same instructions, so the
values do not change. The served k3 form (C = 16) takes it here, and the k3 rows form from its own condition. k7b2
(C = 8) keeps its program (a C = 8 measurement
found the C = 16 levers slower there)."""

K7_ONLY_LEVERS_OFF = {
    "kda_early_out_tokens": 0,
    "kda_in_proj_lanes_view": False,
    "mla_sums_rs_first": False,
    "kda_token_hook": False,
    "kda_states_after_gate_result": False,
}
"""The bitwise levers on by default that only the C = 16 forms of ``K7_LEVER_FORM_SET`` (k7 and k3) take: every other form of the 2-server verify
passes them their off values (the program without these levers), with the aux row ring off too (``AUX_RING_DEPTH``): the
KDA early output tail (``verify_kda.EARLY_OUT_TOKENS``), the KDA MXFP8 input projection's in-place lanes, the MLA BF16-o
sums before the numerators (``tkg_mla_block.SUM_RS_FIRST``), the KDA MLP AttnRes ahead part in the token chain
(``decode_block.AHEAD_MLP_TOKEN_STEPS``, which only layers with enough bank entries take) and the KDA state DMAs after
the gate results (``tkg_weight_dma_gate.order_after_gate_result``). A C = 8 (k7b2) measurement found them slower there. A
``SUMS_RS_FIRST_FORM_SET`` form keeps the sums-first order (it does not take ``mla_sums_rs_first`` False)."""
K7_LEVER_FORM_SET = ("k3", "k7")
"""The served forms that take the ``K7_ONLY_LEVERS_OFF`` levers: k7 (by its shape) and k3 (C = 16, on the k1 loop
path). The C = 8 forms keep the program without these levers (a k7b2 measurement found the levers slower). At
Q = 4 (k3) the KDA early output tail (``verify_kda.EARLY_OUT_TOKENS`` 6) has no early tokens, so it keeps its program."""

SUMS_RS_FIRST_K7B2 = True
"""Kill switch of the k7b2 entry of ``SUMS_RS_FIRST_FORM_SET``: True (here, with the one-pass sum blocks of
``SUM_BLOCKS_PASS_FORM_SET``) keeps the sums-first order on k7b2; False passes k7b2 the ``K7_ONLY_LEVERS_OFF`` value
``mla_sums_rs_first`` False (the numerator reduce-scatter first, the program without the sums-first order)."""
SUMS_RS_FIRST_K3B2 = True
"""Kill switch of the k3b2 entry of ``SUMS_RS_FIRST_FORM_SET``: True keeps the sums-first order on k3b2 (C = 8, one
pass of 8 rows); False passes k3b2 the ``K7_ONLY_LEVERS_OFF`` value ``mla_sums_rs_first`` False (the program
without the sums-first order). k3 (C = 16) takes the order with the ``K7_LEVER_FORM_SET`` levers."""
SUMS_RS_FIRST_FORM_SET = (("k7b2",) if SUMS_RS_FIRST_K7B2 else ()) + (("k3b2",) if SUMS_RS_FIRST_K3B2 else ())
"""The forms besides k7 whose MLA BF16-o state stores its sums before its numerators and issues the small FP32 sums
reduce-scatter before the BF16 numerator one (``tkg_mla_block.SUM_RS_FIRST``), so the sums reduce-scatter flies while the
numerators still store, and the reciprocal of the sums does not wait behind the numerator reduce-scatter. The same
two reductions on the same bytes, so the bits are the same. The C = 8 FP8 P@V pass (k7b2) waits in each MLA layer
on that reciprocal (12L profiles, 16K and 1M); the C = 8 measurement that keeps the k7 levers off did not include the pass."""

KDA_MERGED_STATE_LOADS_FORM_SET = ("k7", "k3")
"""The served forms whose KDA verify loads each sequence's state once, from its selected page (``served_defaults``
``kda_merged_state_loads``; ``verify_kda.kda_verify_attention`` ``merged_state_loads``), with the KDA state slots. The
same bits. The C16 verifies (k7, and k3: a shorter 12-layer C = 16 k3 step, bitwise) take it; the other forms keep both
loads (k7b2: a longer 12-layer C = 8 step)."""
KDA_X_QUANT_HOOK_FORM_SET = ("k3b2", "k7b2")
"""The 2-server verify forms whose KDA layers trace the bank sums of their MLP mix right after the quantize of the MXFP8
input projection (``verify_bodies`` ``kda_x_quant_hook``), so they run beside the projection's matmuls; every other form
keeps them at their place without the hook. The same bits. MEASURED on the 12-layer C = 8 (k7b2) verify: a shorter step
at 16K, 128K and 1M. k3b2 (C = 8) mirrors it (the same bits)."""

NORM_SCALAR_SQUARES_BY_FORM = {"k3": 5, "k3b8": 5, "k3b2": 5}
"""The served ``norm_scalar_squares`` by form (``served_defaults``): the last rows of each norm after a mix whose sums of
squares run on ScalarE; 0 (VectorE only) for a form not in the table. The k3 forms take 5. The same bits."""

ATTN_RES_SCALAR_PRODUCTS_BY_FORM = {"k3": 6, "k3b8": 6, "k3b2": 6}
"""The served ``attn_res_scalar_products`` by form (``served_defaults``): the last rows of each bank entry after the first
whose mix products run on ScalarE (only in a mix of 2 or more bank entries); 0 for a form not in the table. The k3 forms
take 6. The same products and adds in the same order; trn3 ScalarE writes +0 for a negative zero product, so an exact-zero
result can differ in its sign."""

KDA_W_IN_BUFFERS_BY_FORM = {"k3": 8, "k3b8": 8, "k3b2": 8}
"""The served ``kda_w_in_buffers`` by form with the KDA state slots (``served_defaults``; kda_replay "states"): the
kda_w_in ring depth of the KDA verify layers; 0 (the default ring) for a form not in the table and with the replay
records. The k3 forms take 8. The same bits."""

KDA_STATE_LOADS_FIRST_FORM_SET = K3_FORMS
"""The served forms whose KDA verify issues its state loads first by default (``served_defaults``
``kda_state_loads_first``), with the KDA state slots. The same bits."""

KDA_SLOT_PAGES_FIRST_FORM_SET = K3_FORMS
"""The served forms whose KDA verify issues its slot-page loads first by default (``served_defaults``
``kda_slot_pages_first``), with the KDA state slots. The same bits."""

KDA_UPDATE_BCAST_DEPTH_BY_FORM = {"k3": 4, "k7": 4}
"""The served ``kda_update_bcast_depth`` by form with the KDA state slots (``served_defaults``): the PSUM tiles of the
state-update broadcast ring of every KDA verify token; 2 (the ``verify_model`` default) for a form not in
the table and with the replay records. The C = 16 forms (k7 and the served k3) take 4 (measured with k7 R0 bitwise). The
same bits."""

KDA_HEAD_PIPELINE_BY_FORM: dict = {}
"""The served ``kda_head_pipeline`` by form (``served_defaults``; ``verify_kda.kda_verify_attention`` ``head_pipeline``,
``HEAD_PIPELINE_FORMS``): ``g_last`` runs the output-gate blocks of the MXFP8 input projection after the decay gate. The
same bits. Empty: every form keeps ``off``. On the C = 8 verify (k7b2), ``g_last`` gives a shorter step (a 12-layer
measurement), but its 93-layer program adds compiler waits to the next MoE layers (a longer
93-layer C = 8 step, MEASURED); the kwarg ``g_last`` still turns it on."""

KDA_OUT_AR_ROWS_FORM_SET = ()
"""ACCURACY CLASS (<= 1 BF16 ulp): the served forms whose KDA output all-reduce runs as two row chunks, the early output
tail's rows inside the token chain (``served_defaults`` ``kda_out_ar_rows``; ``verify_kda.kda_verify_attention``
``out_ar_rows``). Empty: no form takes it until an accuracy check."""

KDA_SLOT_CONV_FROM_INPUTS_FORM_SET = ("k3", "k7")
"""The served forms whose KDA verify slot conv-state stores read their windows from the conv inputs by default
(``served_defaults`` ``kda_slot_conv_from_inputs``), with the KDA state slots, the per-token state tiles
(``kda_token_tiles``) and the sequential chain. The same bits."""

SERVED_GLUE_ADDITIONS_BY_FORM = {
    "k7": {
        "compact_slot_gather": "split",
        "compact_tables_scalar": True,
        "in_proj_lanes_view": True,
        "tail_store_halves": True,
        "weights_after_rows": True,
        "norm_one_mul": True,
    },
}
"""The bitwise MoE glue keys of a served form that ``served_defaults`` adds to the call's glue, given or default, with
the routed-token compaction: each key only where the glue does not set it, so a given value wins, an off value (False,
"gpsimd") included, and is the per-key kill switch. A given glue does not win wholesale for these keys. A glue whose
compaction the kill switch drops (``ROUTED_COMPACT_ON`` False) takes none of them, and a compaction other than
``hot_last`` takes none of ``HOT_LAST_GLUE_ADDITIONS``."""
HOT_LAST_GLUE_ADDITIONS = ("tail_store_halves",)
"""The keys of ``SERVED_GLUE_ADDITIONS_BY_FORM`` that the kernel takes only with ``routed_compact`` ``hot_last``."""

ATTN_RES_AHEAD_VECTOR_BY_FORM = {"k3": "all", "k7": "all"}
"""The served moe_glue key ``ahead_sums_vector`` by form (``served_defaults``; a value of
``decode_block.ATTN_RES_AHEAD_VECTOR_MODES``): with ``attn_res_ahead`` / ``attn_res_ahead_mlp``, the AttnRes bank parts
of the KDA layers run their GpSimd share of the weighted sums and ``B * g`` on VectorE with the same accumulators and
order (``decode_block_body`` ``attn_res_ahead_vector``). Added only to a call with ``attn_res_ahead`` or
``attn_res_ahead_mlp`` and where the glue does not set it, so a given value wins ("off" included). The C16 verify (k7)
and the served k3 form take "all"; the other forms keep "off"."""
ATTN_RES_AHEAD_LOAD_TILES_BY_FORM = {"k3": 4, "k7": 4}
"""The served moe_glue key ``ahead_load_tiles`` by form (``served_defaults``): with ``attn_res_ahead`` /
``attn_res_ahead_mlp``, the SBUF tiles of the entry loads of each AttnRes bank part of the KDA layers
(``decode_block_body`` ``attn_res_ahead_load_tiles``; 2 keeps the program). Added only to a call with ``attn_res_ahead``
or ``attn_res_ahead_mlp`` and where the glue does not set it, so a given value wins. The C16 verify (k7) and the
served k3 form take 4 tiles; the other forms keep 2."""
ATTN_RES_AHEAD_MLP_KDA_TAIL_BY_FORM = {"k3": True, "k7": True}
"""The served moe_glue key ``ahead_mlp_kda_tail`` by form (``served_defaults``): with ``attn_res_ahead_mlp``, a KDA layer
that builds the bank part of its own MLP mix in its body takes its weighted sums and norm after the output projection of
its last rows, not in the token chain (``decode_block_body`` ``attn_res_ahead_mlp_kda_tail``; the same values). Added
only to a call with ``attn_res_ahead_mlp`` and where the glue does not set it, so a given value wins (False is the kill
switch). The C16 verify (k7) and the served k3 form take it; the other forms keep it off."""
KDA_MLP_TAIL_ENTRIES_BY_FORM: dict = {"k7": 4}
"""The served moe_glue key ``kda_mlp_tail_entries`` by form (``served_defaults``): with the kept bank rstd and the early
sums (``attn_res_cached_bank_sums`` and ``attn_res_early_sums``), the first N entries of the plain MLP mix's bank sums
of each KDA layer run after the KDA output projection, in the flight of the KDA all-reduce, not ahead of the KDA body
(``decode_block_body`` ``attn_res_kda_mlp_tail_entries``; the same values).
Added only where the glue does not set it, so a given value wins (0 is the kill switch). The C16 verify (k7) takes 4
entries: the sums of 4 entries at 32 rows fit in the all-reduce flight; the other forms keep the program (at C8
the sums hold only a small part of the step critical path)."""
MLA_MLP_TAIL_ENTRIES_BY_FORM: dict = {"k7": 4}
"""The served moe_glue key ``mla_mlp_tail_entries`` by form (``served_defaults``): with the kept bank rstd, the early sums
and ``attn_res_mla_mlp_early`` (the MLA layer's MLP bank sums at the MLA body's state reduce-scatter), the first N
entries of those sums run after the store of the MLA tail's output partials, in the flight of the MLA output
all-reduce (``decode_block_body`` ``attn_res_mla_mlp_tail_entries``; the same values).
Added only where the glue does not set it, so a given value wins (0 is the kill switch). The C16 verify (k7) takes 4
entries: the sums of 4 entries at 32 rows fit in the all-reduce flight; the other forms keep the program."""
ATTN_RES_AHEAD_BEFORE_RESID_BY_FORM = {"k3": True, "k7": True}
"""The served moe_glue key ``ahead_before_resid`` by form (``served_defaults``): with ``attn_res_ahead``, the residual add
of each layer follows the AttnRes ahead parts' weighted sums (``decode_block_body`` ``attn_res_ahead_before_resid``; the
same instructions in another order, so the values do not change). Added only to a call with ``attn_res_ahead`` and where
the glue does not set it, so a given value wins (False is the kill switch). The C16 verify (k7) and the served k3 form
take it; the other forms keep it off."""
ATTN_RES_AHEAD_GRAM_MERGE_BY_FORM = {"k3": True, "k7": True}
"""The served moe_glue key ``ahead_gram_merge`` by form (``served_defaults``): with ``attn_res_ahead`` /
``attn_res_ahead_mlp``, the ahead mixes take their TensorE Gram stream sums as one matmul a hidden tile on a comb of the
stream, the bank part and the weight split (``decode_block_body`` ``attn_res_gram_merge``; the same sums). Added only to
a call with an ahead form and where the glue does not set it, so a given value wins (False is the kill switch). The C16
verify (k7) and the served k3 form take it; the other forms keep it off."""
KDA_VERIFY_CONSTANTS_FORM_SET = ("k3", "k3b2", "k7")
"""The served forms whose KDA verify calls take their all-ones tiles once a kernel (``kda_verify_constants``: the region's
fp32 ones and one bf16 ones heap tile) in place of the two memsets of each call. The same bits. The C16 (k7) verify and
the k3 forms k3 (C = 16) and k3b2 (C = 8) take it; every other form, the C8 (k7b2) verify included, keeps the memsets of
each call, so its program does not change."""
ATTN_RES_CACHED_ACT_SQUARES_BY_FORM = {"k3": True, "k3b2": True, "k7": True}
"""The served moe_glue key ``cached_act_squares`` by form (``served_defaults``): with ``attn_res_cached_bank_sums`` and
``attn_res_early_sums``, the cached AttnRes mixes take the FP32 squares of the bf16 stream from ScalarE while VectorE runs
the weighted rows of the stream (``decode_block_body`` ``attn_res_cached_act_squares``; the same exact squares and sums).
Added only to a call with those sums and where the glue does not set it, so a given value wins (False is the kill
switch). The C16 verify (k7) and the k3 forms k3 (C = 16) and k3b2 (C = 8) take it; the other forms keep it off."""
KDA_BF16_COLS_ONCE_FORM_SET = ("k3", "k3b2", "k7")
"""The served forms whose KDA verify calls zero the bf16 update rows tiles once a call (``verify_model``
``kda_bf16_cols_once``, ``kda_verify_attention`` ``bf16_cols_once``): with ``kda_bf16_update_rows`` on the per-token
states path (``kda_token_tiles``, ``kda_replay`` "states", the sequential chain), so the token chain issues no GpSimd
memset behind the slot conv scatters. The same bits. The C16 verify (k7) and the k3 forms k3 (C = 16) and k3b2 (C = 8)
take it (by the served shape); the other forms keep the memsets."""
KDA_WEIGHT_DMA_BY_FORM = {"k7": "act", "k3": "act"}
"""The served moe_glue key ``kda_weight_dma`` by form (``served_defaults``): the queue of the KDA verify calls'
output-projection, small-weight and parameter loads (``verify_model`` ``kda_weight_dma``; ``act`` the Activation hardware
DGE queue in place of the GpSimd software DGE). The same bytes. Added only where the glue does not set it, so a given
value wins ("pool" is the kill switch). The C16 verifies (k7, and k3: a shorter 12-layer C = 16 k3 step, bitwise) take
it; the other forms, the C8 (k7b2) verify included, keep "pool"."""
KDA_W_OUT_AFTER_W_IN_FILLS_BY_FORM = {"k3": True}
"""The served moe_glue key ``kda_w_out_after_w_in_fills`` by form (``served_defaults``): with the MXFP8 ``kda_w_in``,
the KDA verify calls load ``kda_w_out`` (and its MX scales) after the last two ``kda_w_in`` fills of their input
projection, so the ``kda_w_in`` stream does not share the DMA queues with it (``verify_model``
``kda_w_out_after_w_in_fills``). The same bytes, so the bits do not change. Added only to a call with the MXFP8
``kda_w_in`` and where the glue does not set it, so a given value wins (False is the kill switch). The C16 k3 verify
takes it (a shorter 12-layer C = 16 k3 step, bitwise); C = 8 a tie, so k3b2 keeps the program, as the other forms
do."""
KDA_COMMIT_AFTER_W_IN_BY_FORM = {"k3": True, "k3b2": True, "k7": True, "k7b2": True}
"""The served moe_glue key ``kda_commit_after_w_in`` by form (``served_defaults``): the KDA verify calls put their
committed-state stores after the input projection's prefetch fills on the Sync queue (``verify_model``
``kda_commit_after_w_in``), so no weight fill waits behind a store that waits for the state commit. The same bits. Added
only to a call with the per-token states ring and where the glue does not set it, so a given value wins (False is the
kill switch). The C16 verify (k7), the C = 8 verify (k7b2; 12L C = 8 MEASURED a shorter step, without the per-call slot
ring) and the k3 forms k3 (C = 16) and k3b2 (C = 8) take it; the other forms keep the program."""
KDA_SLOT_STORE_REGS_BY_FORM = {"k7b2": True, "k3": True}
"""The served moe_glue key ``kda_slot_store_regs`` by form (``served_defaults``): each scored token of the KDA verify
calls loads the slot page ids of its slot state stores into registers ahead of the stores (``verify_model``
``kda_slot_store_regs``). The stores wait for the store before them (the dynamic-page writes of the pool), so a page id
load after that wait adds its time to each link of the chain; ahead of it, the Sync queue runs it while the store before
is in flight. The same pages, so the bits do not change. Added only to a ``states`` call with one Sync slot-store queue
and where the glue does not set it, so a given value wins (False is the kill switch). The C8 verify (k7b2) and the C16 k3
verify (a shorter 12-layer C = 16 k3 step, bitwise) take it; the other forms keep the program (k7: neutral)."""
KDA_SLOT_CONV_AFTER_AR_BY_FORM = {"k7b2": True}
"""The served moe_glue key ``kda_slot_conv_after_ar`` by form (``served_defaults``): the KDA verify calls trace their
slot conv-state stores after the call's output all-reduce (``verify_model`` ``kda_slot_conv_after_ar``). The pool writes
of a call wait for the pool write before them (the dynamic-page writes of the pool), so the first slot state store then
waits for its own state only, not for the conv-state scatters, which run in the all-reduce flight after the last slot
state store. The same rows and pages, so the bits do not change. Added only to a ``states`` call with the per-token
state tiles and the sequential chain, where the glue does not set it, so a given value wins (False is the kill switch).
The C8 verify (k7b2) takes it; the other forms keep the program."""
KDA_SLOT_RING_PER_CALL_BY_FORM = {"k3": True, "k7": True}
"""The served moe_glue key ``kda_slot_ring_per_call`` by form (``served_defaults``): the KDA verify calls take a per-call
slot-state ring of Q tiles (``verify_model`` ``kda_slot_ring_per_call``) in place of the kernel's persistent ring, with
``kda_token_tiles`` on the sequential states chain: no token's state update waits for the slot stores of an earlier
token, and the tiles live only until their stores have read them. The same bits. Added only to a call with that chain
and where the glue does not set it, so a given value wins (False is the kill switch). The C16 verify (k7) and the
served k3 form (C = 16) take it; the other forms keep the persistent ring. The C = 8 forms keep it: on the 12L C = 8
verify the ring with ``kda_commit_after_w_in`` measured a longer step, the commit alone a shorter one."""
KDA_SLOT_RING_KEEP_DEFER = True
"""With the moe_glue key ``kda_slot_ring_per_call``: each KDA verify call's keep-reads of its slot-state ring trace right
after the next 2-server MoE layer's MXFP8 in-projection loads (``decode_block_body`` ``moe_in_proj_hook``; else at the
next KDA call's start, after its first GpSimd copy, ``verify_model`` ``kda_slot_ring_keep_defer``), and the last call's
before the kernel end, in place of right after the call's all-reduce, where they wait for the trailing slot stores on
GpSimd ahead of those loads. The ring tiles stay live until the reads. False: the reads follow each call's all-reduce."""
KDA_CONV_PRODUCTS_ENGINE_BY_FORM = {"k3": "vector", "k7": "vector"}
"""The served ``kda_conv_products_engine`` by form (``served_defaults``): the engine of the one-pass conv tap products of
the KDA verify calls ("gpsimd" or "vector"; the bits of finite values do not change). A GpSimd tensor_tensor stalls the
VectorE ops that share its SBUF port, and with the tap-major products a VectorE tap takes a fraction of a GpSimd one.
The C16 verify (k7) and the served k3 form take "vector"; the other forms keep "gpsimd"."""
KDA_SLOT_STATE_DEPTH_BY_FORM = {"k3b2": 3, "k7b2": 3}
"""The served ``kda_slot_state_depth`` by form (``served_defaults``), with the KDA state slots, the per-token state tiles
(``kda_token_tiles``) and the sequential chain, without the shared ring: the tiles of the slot-state ring; 2 (the ``verify_model`` default) for a
form not in the table and for every other call. The same bits. A third tile lets a token's state update start while the slot
stores of the two tokens before it are in flight (+1.5 KB a partition). The C8 verifies take 3 (k3b2: a shorter
12-layer C = 8 k3 step; k7b2: a 12-layer C = 8 k7 step win; both bitwise); the other forms keep 2."""

KDA_SLOT_STORE_ENGINE_BY_FORM = {}
"""The served ``kda_slot_store_engine`` by form (``served_defaults``), with the KDA state slots, the per-token state
tiles and the sequential chain, without the shared ring (as ``KDA_SLOT_STATE_DEPTH_BY_FORM``): the queue of the
slot-state stores; "sync" (the ``verify_model`` default) for a form not in the table and for every
other call. The same bits. Empty: every form keeps "sync"."""

KDA_TOKEN_TILES_FORM_SET = K3_FORMS
"""The served forms whose KDA verify gives each scored token its own state tiles by default (``served_defaults``
``kda_token_tiles``), with the KDA state slots and without the inline prework ahead; at Q = 8 the slot DMA set and the
persistent slot conv sources come with them (``verify_model.check_kda_token_tiles``). The same bits."""

KDA_PREWORK_AHEAD_FORM_SET = ("k3", "k3b2")
"""The served forms whose verify calls take the moe_glue key ``kda_prework_ahead`` by default (``served_defaults``):
each KDA layer's page metadata and state, conv-state and slot-page loads are traced in the feed-forward layer before it
(``verify_model`` ``kda_prework_ahead``, ``decode_block_body``), and the KDA body keeps the VectorE selects of those
loads for after its input projection's quantize (``verify_model`` ``kda_prework_deferred``). Such a form keeps its
state-slot levers (the per-token tiles, the slot DMA set, the slot-state ring and the glue keys of the states ring).
The prework is then the place of the state loads, so the call drops ``kda_state_loads_first``,
``kda_slot_pages_first`` and ``kda_state_loads_after_ring``. The same loads and selects, so the bits do not change.
Added only to a one-call ``states`` verify and where the glue does not set it, so a given value wins (False is the kill
switch). Only k3 forms may join (the other forms keep the key off). The C16 and C8 k3
verifies take it: a shorter 12-layer C = 16 k3 step and a shorter 12-layer C = 8 k3 step, bitwise; SBUF +~3.4 /
+~1.7 KB a partition live across the MoE (the prework's state and slot tiles)."""

KDA_OUT_AR_SBUF_FORM_SET = ()
"""The served forms whose KDA verify output all-reduce takes one persistent SBUF source by default (``served_defaults``
``kda_out_ar_sbuf``, SB2SB). The same values. Empty: every form keeps the HBM all-reduce."""
SERVED_GLUE = {"rows_queue": "act", "router_first": True, "out_proj_early": True, "selectors_once": True}
"""The MoE glue options (``tkg_moe_2srv.GLUE_KEYS``) of the served forms."""
SCALE_ROWS_QUADRANTS_FORM_SET = ()
"""The served forms whose MoE glue also takes ``scale_rows_quadrants`` by default (``served_defaults``): the static MX
scale loads of the expert calls and of the expert-ring prefetch issue one DMA per scale row over the 4 quadrants (4 DMA
engines) in place of one per quadrant (``moe_routed.scale_rows_across_quadrants``); the same bytes to the same
partitions. A glue that the caller gives wins. Empty: every form keeps the per-quadrant loads."""
SERVED_EARLY_MLP_ENTRIES = 32
"""The bank entries of the early MLP AttnRes sums of the KDA layers in a split call of a served form."""
PREFETCH_DMA_PRIORITY = 3
"""The trn3 DMA QoS level of the expert ring loads of ``moe_expert_prefetch``: the lowest, so the ring streams in the
DMA time that the attention's weight and state loads leave (``decode_block`` ``moe_prefetch_priority``)."""
VERIFY_2SRV_TAKES_KDA_STATE_DTYPE = True
"""The entry takes ``kda_state_dtype`` (the fp32 KDA state pages of the 2-server decode), so a caller can check for it
before it passes any value but ``bf16``."""


def moe_expert_prefetch_parts(moe_expert_prefetch: dict) -> tuple[dict, bool]:
    """Check ``moe_expert_prefetch`` and return its ring depths ``{"kda": W, "mla": W}`` (``{}`` when off) and its
    ``kda_after_in_proj`` placement (False when absent or 0)."""
    prefetch = dict(moe_expert_prefetch or {})
    after_in_proj = prefetch.pop("kda_after_in_proj", 0)
    kernel_assert(
        not prefetch
        or (
            set(prefetch) == {"kda", "mla"}
            and all(isinstance(w, int) and 0 <= w <= MAX_PREFETCH_WEIGHT_BUFFERS for w in prefetch.values())
            and any(prefetch.values())
        ),
        f"moe_expert_prefetch must be None, {{}} or {{'kda': W, 'mla': W}} (and optionally 'kda_after_in_proj': 0 or 1)"
        f" with 0 <= W <= {MAX_PREFETCH_WEIGHT_BUFFERS}, not both 0, got {moe_expert_prefetch}",
    )
    kernel_assert(
        isinstance(after_in_proj, int) and after_in_proj in (0, 1),
        f"moe_expert_prefetch kda_after_in_proj must be 0 or 1, got {after_in_proj!r}",
    )
    kernel_assert(
        not after_in_proj or prefetch.get("kda", 0) > 0,
        f"moe_expert_prefetch kda_after_in_proj needs a KDA ring (kda >= 1), got {moe_expert_prefetch}",
    )
    return prefetch, bool(after_in_proj)


FP8_PV_VALUE_FORM_BY_FORM = {"k3": 0, "k3b2": 0}
"""The ``fp8_pv_value_form`` of a served form whose packed p2 loop takes the FP8 P@V pass, where the form's group
count forbids the ``FP8_PV_LOOP_OPTIONS`` value (``served_pack_options``; a value of the loop's ``VALUE_FORMS``): the
k3 forms have ``pack_p2_groups`` 3 groups, and value_form 4 / 5 (the 4-piece value copy) needs 4+ groups, so they
take the any-groups value_form 0. k7 / k7b2 are absent (6 groups), so their value_form is untouched (byte-identical).
A given ``fp8_pv_value_form`` wins."""


def served_pack_options(
    world_size: int,
    queries: int,
    rows: int,
    packed: bool,
    pack_options: dict,
    state_bf16_o: bool,
    mx_qk: str = None,
    entry_act: bool = False,
    step_items: bool = False,
    setup_off_pool: bool = False,
    slot_rows_sbuf: bool = False,
    skip_stage_store: bool = False,
    fp8_pv: bool = None,
) -> dict:
    """Return the packed-loop options of a verify call: the caller's ``pack_options``, with ``p2`` added in the
    ``P2_FORMS`` forms on the packed loop where it fits, and with it ``clamp_tail_only`` (with the page clamp only the
    loop of the last key groups clamps its pages; bitwise the same), ``stage_rows_sbuf`` (the prologue's slot rows
    read the p2 stage tile in SBUF, not its HBM store, so the loop entry does not wait for that store; the same bytes)
    and ``word_keys`` (the 32-bit word key transposes; bitwise the same) with ``item_rope_ahead`` (the rope keys of the
    next item transposed one sub-slot early; bitwise the same) unless the caller's options take ``dr_keys``,
    ``mx_keys``, ``deep_pipe``, ``numerator_ring`` or ``psum_per_block``, and the 8-deep slot-row ring (``row_depth`` 8)
    in those forms.

    p2 takes the BF16-o state in both state reduce-scatter forms (the one of a call and the one of a pass); it does not
    take mx_keys or deep_pipe. ``stage_rows_sbuf`` and the slot-row ring go only to the ``P2_SBUF_LEVER_FORMS`` forms
    (k7): k7b2 has no SBUF room for them. The p2 set of an ``OWN_STATE_STACK_FORMS`` form also takes ``own_state_stack``
    (one own-key state matmul for each group). With ``mx_qk`` "E" (``MX_QK_TRN3_MODES``) the k7 or k7b2 p2 set with word_keys
    also takes ``MX_QK_E_OPTIONS``, and ``mx_natural`` in the ``MX_QK_NATURAL_FORM_SET`` forms. With ``entry_act``
    (``served_defaults`` ``mla_p2_entry_act``) the p2 set also takes ``p2_entry_act`` (with ``mx_qk`` the loop keeps the
    zero fill of every stage byte but the references), and with ``step_items`` (``served_defaults`` ``mla_step_items``)
    ``hoist_plans`` with ``step_items`` (the item plans and item rows of the call built once for its MLA layers;
    ``MX_QK_E_OPTIONS`` takes them too). With ``setup_off_pool``, ``slot_rows_sbuf`` and ``skip_stage_store``
    (``served_defaults`` ``mla_pass_setup_off_pool``, ``mla_slot_rows_sbuf`` and ``mla_skip_stage_store``) the p2 set also
    takes the loop option of that name. The p2 set of a ``Q_FP8_P2_FORM_SET`` form also takes ``q_fp8`` (the e4m3 query of
    the call). A p2 set with ``word_keys`` of an ``ACT_SPREAD_FORM_SET`` form also takes ``act_spread`` (the ScalarE order
    of the word-key loop: the weight copy before the exp and the item's other ScalarE work spread one group a sub-slot;
    the same instructions, bitwise the same; MEASURED on trn3: a shorter sub-slot after the key-word transposes
    and a shorter 12L 2-node C = 16 step at 1M; an empty set is the switch off). A p2, word_keys, row_depth,
    p2_entry_act, hoist_plans, step_items, q_fp8, mx_natural, act_spread or option of ``MX_QK_E_OPTIONS`` that the
    caller gives wins. With ``fp8_pv`` (None: the form is in ``FP8_PV_FORM_SET``) the "E" set of an ``FP8_PV_FORMS``
    form also takes ``fp8_pv`` and
    ``mx_natural`` (the FP8 P@V pass; accuracy class); a given ``fp8_pv`` or ``mx_natural`` pack option wins. With the FP8
    P@V pass an ``FP8_PV_LOOKAHEAD_FORMS`` form also takes ``FP8_PV_LOOKAHEAD_OPTIONS`` and ``FP8_PV_LOOP_OPTIONS``
    (bitwise), and an ``FP8_PV_QK_GROUP_FORMS`` form ``FP8_PV_QK_GROUP_OPTIONS`` (bitwise); a given key wins. The p2 set
    of an ``MLA_FRONT_TAIL_FORMS`` form also takes ``MLA_FRONT_TAIL_OPTIONS`` (bitwise); a given key wins. Every other
    call keeps ``pack_options``.

    Example:
        >>> served_pack_options(256, 8, 32, True, None, True)["item_rope_ahead"], served_pack_options(256, 4, 32, True, None, False)
        (True, None)
        >>> served_pack_options(256, 8, 32, True, None, True, "E")["mx_qk"], "mx_qk" in served_pack_options(256, 8, 32, True, None, True)
        ('block32', False)
        >>> served_pack_options(256, 8, 32, True, {"q_fp8": True}, True)["q_fp8"], "q_fp8" in served_pack_options(256, 8, 32, True, None, True)
        (True, False)
        >>> served_pack_options(256, 8, 16, True, None, True)["act_spread"], served_pack_options(256, 8, 16, True, {"act_spread": False}, True)["act_spread"]
        (True, False)
        >>> served_pack_options(256, 8, 32, True, None, True, "E", fp8_pv=True)["fp8_pv"], "fp8_pv" in served_pack_options(256, 8, 32, True, None, True, "E")
        (True, False)
        >>> cf = {"fp8_pv_chunks_first": True}
        >>> "fp8_pv_chunks_first" in served_pack_options(256, 8, 32, True, cf, True), "fp8_pv_chunks_first" in served_pack_options(256, 8, 16, True, cf, True)
        (True, False)
    """
    form = next(
        (name for name, value in SERVED_FORMS.items() if value == (form_world(world_size), queries, rows)), None
    )
    given = pack_options or {}
    fits = not (given.get("mx_keys") or given.get("deep_pipe"))
    word_free = not any(
        given.get(key) for key in ("dr_keys", "mx_keys", "deep_pipe", "numerator_ring", "psum_per_block")
    )
    if packed and form in P2_FORMS and fits and "p2" not in given:
        pack_options = {
            "clamp_tail_only": True,
            **({"stage_rows_sbuf": True} if form in P2_SBUF_LEVER_FORMS else {}),
            **({"own_state_stack": True} if form in OWN_STATE_STACK_FORMS else {}),
            **({"word_keys": True, "item_rope_ahead": True} if word_free else {}),
            **(MX_QK_E_OPTIONS if mx_qk == "E" and form in MX_QK_TRN3_MODES["E"] and word_free else {}),
            **({"mx_natural": True} if mx_qk == "E" and form in MX_QK_NATURAL_FORM_SET and word_free else {}),
            # The FP8 P@V pass (accuracy class) reads the natural MX query words.
            **(
                {"fp8_pv": True, "mx_natural": True}
                if mx_qk == "E"
                and form in MX_QK_TRN3_MODES["E"]
                and form in FP8_PV_FORMS
                and word_free
                and (form in FP8_PV_FORM_SET if fp8_pv is None else fp8_pv)
                else {}
            ),
            # The FP8 P@V loop boundary lookahead (bitwise), with the FP8 P@V pass of an FP8_PV_LOOKAHEAD_FORMS form.
            **(
                {**FP8_PV_LOOKAHEAD_OPTIONS, **FP8_PV_LOOP_OPTIONS}
                if mx_qk == "E"
                and form in MX_QK_TRN3_MODES["E"]
                and form in FP8_PV_LOOKAHEAD_FORMS
                and word_free
                and (form in FP8_PV_FORM_SET if fp8_pv is None else fp8_pv)
                else {}
            ),
            # The FP8 P@V QK grouping (bitwise), with the FP8 P@V pass of an FP8_PV_QK_GROUP_FORMS form.
            **(
                FP8_PV_QK_GROUP_OPTIONS_BY_FORM[form]
                if mx_qk == "E"
                and form in MX_QK_TRN3_MODES["E"]
                and form in FP8_PV_QK_GROUP_FORMS
                and word_free
                and (form in FP8_PV_FORM_SET if fp8_pv is None else fp8_pv)
                else {}
            ),
            # The FP8 P@V late preps (bitwise), with the FP8 P@V pass of an FP8_PV_PREP_LATE_FORM_SET form.
            **(
                FP8_PV_PREP_LATE_OPTIONS
                if mx_qk == "E"
                and form in MX_QK_TRN3_MODES["E"]
                and form in FP8_PV_PREP_LATE_FORM_SET
                and word_free
                and (form in FP8_PV_FORM_SET if fp8_pv is None else fp8_pv)
                else {}
            ),
            # The FP8 P@V third page buffer (bitwise), with the FP8 P@V pass of an FP8_PV_PAGE_BUFFERS_FORM_SET form.
            **(
                FP8_PV_PAGE_BUFFERS_OPTIONS
                if mx_qk == "E"
                and form in MX_QK_TRN3_MODES["E"]
                and form in FP8_PV_PAGE_BUFFERS_FORM_SET
                and word_free
                and (form in FP8_PV_FORM_SET if fp8_pv is None else fp8_pv)
                else {}
            ),
            **({"p2_entry_act": True} if entry_act else {}),
            # The item plans and rows of the call, built once for its MLA layers (bitwise: one table for every MLA layer).
            **({"hoist_plans": True, "step_items": True} if step_items else {}),
            **({"setup_off_pool": True} if setup_off_pool else {}),
            # The slot rows from the SBUF stage tile, and with them no HBM store of the stage rows (bitwise).
            **({"slot_rows_sbuf": True} if slot_rows_sbuf else {}),
            **({"skip_stage_store": True} if skip_stage_store and slot_rows_sbuf else {}),
            # The MLA front and tail options (bitwise) of an MLA_FRONT_TAIL_FORMS form.
            **(MLA_FRONT_TAIL_OPTIONS if form in MLA_FRONT_TAIL_FORMS else {}),
            **given,
            "p2": True,
        }
        # The ScalarE order of the word-key loop (bitwise; the same instructions), wherever the set has word_keys.
        if form in ACT_SPREAD_FORM_SET and pack_options.get("word_keys") and "act_spread" not in given:
            pack_options["act_spread"] = True
        # A served k3 form (3 groups) cannot take the FP8_PV_LOOP_OPTIONS value_form 4 / 5: pin its any-groups
        # value (FP8_PV_VALUE_FORM_BY_FORM); k7 / k7b2 are absent, so their value_form stays as the loop set it.
        if pack_options.get("fp8_pv") and form in FP8_PV_VALUE_FORM_BY_FORM and "fp8_pv_value_form" not in given:
            pack_options["fp8_pv_value_form"] = FP8_PV_VALUE_FORM_BY_FORM[form]
        # The FP8 P@V pass builds the item plan of each pass itself: no hoisted plans or step items (built every step
        # and never read with it; their heap tiles stay free). The two switch together (step items need the hoisted
        # plans), and a caller that gives either keeps both.
        if pack_options.get("fp8_pv") and not {"hoist_plans", "step_items"} & set(given):
            for key in ("hoist_plans", "step_items"):
                if pack_options.get(key):
                    pack_options[key] = False
        # The MX query gather of the FP8 P@V pass (bitwise): on in the form set, unless the caller's options give it.
        if pack_options.get("fp8_pv") and form in Q_MX_GATHER_FORM_SET and "q_mx_gather" not in given:
            pack_options["q_mx_gather"] = True
        # The MX query gather of a one-pass FP8 P@V call with the scales in the word records (bitwise): on in the
        # one-pass form set while Q_MX_GATHER_ONE_PASS; a given key wins (the kernel takes no q_fp8 with it).
        if (
            Q_MX_GATHER_ONE_PASS
            and pack_options.get("fp8_pv")
            and form in Q_MX_GATHER_ONE_PASS_FORM_SET
            and not given.get("q_fp8")
        ):
            if "q_mx_gather" not in given:
                pack_options["q_mx_gather"] = True
            if pack_options.get("q_mx_gather") and "q_mx_scale_words" not in given:
                pack_options["q_mx_scale_words"] = True
        # The MX query pass prefetch (bitwise): with the MX query gather of the form set; a given key wins.
        if pack_options.get("q_mx_gather") and form in Q_MX_GATHER_FORM_SET and "q_mx_pass_prefetch" not in given:
            pack_options["q_mx_pass_prefetch"] = True
        # The own-rows-ahead load (bitwise): with the MX query gather and the stacked own-key state of the form set.
        if (
            pack_options.get("q_mx_gather")
            and pack_options.get("own_state_stack")
            and form in Q_MX_GATHER_FORM_SET
            and "own_rows_ahead" not in given
        ):
            pack_options["own_rows_ahead"] = True
        # The FP8 P@V setup's chunk tiles on the heap and its frame rows' tiles kept (bitwise): with the stage-free FP8
        # P@V setup of the form set; a given key wins.
        if pack_options.get("fp8_pv_stage_free") and form in Q_MX_GATHER_FORM_SET and "fp8_pv_pages_first" not in given:
            pack_options["fp8_pv_pages_first"] = True
        # The MLA front order of a one-pass FP8 P@V call (bitwise): with the MX query gather, the stacked own-key state and
        # the stage-free setup of the one-pass front form set while FRONT_ONE_PASS; a given key wins.
        if (
            pack_options.get("fp8_pv")
            and pack_options.get("q_mx_gather")
            and pack_options.get("own_state_stack")
            and pack_options.get("fp8_pv_stage_free")
            and FRONT_ONE_PASS
            and form in FRONT_ONE_PASS_FORM_SET
        ):
            pack_options.update({key: value for key, value in FRONT_ONE_PASS_OPTIONS.items() if key not in given})
        # The heap numerator tile of the pass state store (bitwise): with the FP8 P@V pass and the BF16-o state of the form
        # set, the second pass waits only for the first pass's cast and sum stores; a given key wins.
        if pack_options.get("fp8_pv") and state_bf16_o and form in Q_MX_GATHER_FORM_SET and "state_o_heap" not in given:
            pack_options["state_o_heap"] = True
        # The first pass's sums stored from a heap copy (bitwise): with the heap numerator tile; a given key wins.
        if pack_options.get("state_o_heap") and form in Q_MX_GATHER_FORM_SET and "state_sums_heap" not in given:
            pack_options["state_sums_heap"] = True
        # The sum blocks of the call-level BF16-o state (bitwise): with the FP8 P@V pass and the BF16-o state of the form
        # set, the sums of a call go to HBM once, in whole 64-byte lines; a given key wins.
        if (
            pack_options.get("fp8_pv")
            and state_bf16_o
            and form in Q_MX_GATHER_FORM_SET
            and "state_sum_blocks" not in given
        ):
            pack_options["state_sum_blocks"] = True
        # The sum blocks of a call of one pass (bitwise): with the FP8 P@V pass and the BF16-o state of the form set while
        # SUM_BLOCKS_PASS; a given key wins.
        if (
            SUM_BLOCKS_PASS
            and pack_options.get("fp8_pv")
            and state_bf16_o
            and form in SUM_BLOCKS_PASS_FORM_SET
            and "state_sum_blocks_pass" not in given
        ):
            pack_options["state_sum_blocks_pass"] = True
        # The last pass's sum blocks stored before its capless check (bitwise): with the sum blocks of the form set while
        # STATE_SUMS_PRE_IF; a given key wins.
        if (
            STATE_SUMS_PRE_IF
            and (pack_options.get("state_sum_blocks") or pack_options.get("state_sum_blocks_pass"))
            and form in STATE_SUMS_PRE_IF_FORM_SET
            and "state_sums_pre_if" not in given
        ):
            pack_options["state_sums_pre_if"] = True
        # The dense output tail of a call of one pass (bitwise): with the FP8 P@V pass and the BF16-o state of the form
        # set while DENSE_TAIL; a given key wins.
        if (
            DENSE_TAIL
            and pack_options.get("fp8_pv")
            and state_bf16_o
            and form in DENSE_TAIL_FORM_SET
            and "state_dense_tail" not in given
        ):
            pack_options["state_dense_tail"] = True
        # The second pass's chunk tables and first pages loaded by the first pass (bitwise): with the pages-first FP8 P@V
        # setup and the step stores of the form set; a given key wins.
        if (
            pack_options.get("fp8_pv_pages_first")
            and pack_options.get("fp8_pv_step_store")
            and form in Q_MX_GATHER_FORM_SET
            and "fp8_pv_tables_ahead" not in given
        ):
            pack_options["fp8_pv_tables_ahead"] = True
        # The MX query scales in the word records of the gather pair (bitwise): one DMA loads a pass's words and scales;
        # with the MX query gather of the form set; a given key wins.
        if pack_options.get("q_mx_gather") and form in Q_MX_SCALE_WORDS_FORM_SET and "q_mx_scale_words" not in given:
            pack_options["q_mx_scale_words"] = True
        # The MLA tail order of the FP8 P@V call (bitwise): on in the form set; a given key wins.
        if pack_options.get("fp8_pv") and form in TAIL_ORDER_FORM_SET:
            pack_options.update({key: value for key, value in TAIL_ORDER_OPTIONS.items() if key not in given})
        # The item plan of the FP8 P@V pass off the SP query queue (bitwise): on in the form set; a given key wins.
        if pack_options.get("fp8_pv") and form in PLAN_FIRST_FORM_SET and "fp8_pv_plan_first" not in given:
            pack_options["fp8_pv_plan_first"] = True
        # The FP8 P@V 1-item chunk of a sequence with 3 null items: on in the form set; a given key wins.
        if pack_options.get("fp8_pv") and form in NULL_SKIP_FORM_SET and "fp8_pv_null_skip" not in given:
            pack_options["fp8_pv_null_skip"] = NULL_SKIP_MODE
        # The FP8 P@V item plan with one reduce a sum (bitwise): on in the form set while FP8_PV_PLAN_REDUCE; a given
        # key wins.
        if (
            FP8_PV_PLAN_REDUCE
            and pack_options.get("fp8_pv")
            and form in PLAN_REDUCE_FORM_SET
            and "fp8_pv_plan_reduce" not in given
        ):
            pack_options["fp8_pv_plan_reduce"] = True
        # The FP8 P@V chunks-first order (bitwise): on in the form set, with the lookahead, the grouped IF stream and no
        # null skip; a given key wins.
        if (
            pack_options.get("fp8_pv")
            and pack_options.get("fp8_pv_lookahead")
            and pack_options.get("fp8_pv_stream") == "ifs_grouped"
            and not pack_options.get("fp8_pv_null_skip")
            and form in FP8_PV_CHUNKS_FIRST_FORM_SET
            and "fp8_pv_chunks_first" not in given
        ):
            pack_options["fp8_pv_chunks_first"] = True
    # The 8-deep slot-row ring (bitwise; the row DMAs on the SP queue wait for the readers of 8 slots back, not 4).
    if packed and form in P2_SBUF_LEVER_FORMS and "row_depth" not in given:
        pack_options = {**(pack_options or {}), "row_depth": 8}
    # The e4m3 query of the call (accuracy class): on in the form set, unless the caller's options give q_fp8.
    if (
        packed
        and form in P2_FORMS
        and form in Q_FP8_P2_FORM_SET
        and (pack_options or {}).get("p2")
        and "q_fp8" not in given
    ):
        pack_options = {**pack_options, "q_fp8": True}
    # The ``MLA_C16_KEYS`` only in their form set: another form drops a given key (the program without those keys).
    if pack_options and form not in MLA_C16_KEYS_FORM_SET and any(key in pack_options for key in MLA_C16_KEYS):
        pack_options = {key: value for key, value in pack_options.items() if key not in MLA_C16_KEYS}
    # no_if (a given pack option): the packed loop emits no runtime IF, because the released neuronx-cc 2.28 stops on an
    # IF ([NCC_ICFG019]). The FP8 P@V stream takes its 0/1-trip loops and no null skip (the same bits). The kernel also
    # runs the p2 passes on the capped loop only (tkg_mla_block _PACKED_PRIOR_OPTIONS no_if), or with a given no_if_loops
    # on the capless loop with the redo as a loop of 0 or 1 trips (the same bits). With fp8_pv_batched (a given pack
    # option) the stream takes the "batched" form: one 0/1-trip loop for each body size of the pass, with the bodies of
    # all sequences in it (the same bits up to the sign of an exact zero). With fp8_pv_batched_exact (a given pack
    # option) it takes the "exact" form: shared rolled loops for the rows that every sequence has, then rolled loops of
    # each sequence for its other rows (no masked work; the same bits). The kernel refuses both together.
    if (pack_options or {}).get("no_if"):
        stream = "batched" if pack_options.get("fp8_pv_batched") else "loops"
        stream = "exact" if pack_options.get("fp8_pv_batched_exact") else stream
        pack_options = {**pack_options, "fp8_pv_stream": stream, "fp8_pv_null_skip": False}
    return pack_options


def mx_qk_keys(mx_qk: str, mla_k1_rows: bool, mla_k1_packed: bool, mla_k1_rows_options: dict) -> tuple:
    """Return the k1 form keys ``(mla_k1_rows, mla_k1_packed, mla_k1_rows_options)`` that ``mx_qk`` takes: the mode is
    authoritative over the k1 form keys that a caller renders (the serving caller gives the served defaults explicitly, as
    ``mla_k1_rows`` False and ``mla_k1_packed`` True for k7 and k7b2). "RA" takes the rows form (rows True, packed False)
    with the served rows options (None, so ``served_defaults`` gives ``SERVED_ROWS_OPTIONS``) unless the caller gives a
    non-empty rows-options dict; "off" takes no rows form (the BF16 latent QK) and no rows options, and in place of a
    rendered rows form (rows True with packed False) the default packed loop; "E" and None keep the caller's keys ("E"
    needs the packed loop: ``check_mx_qk`` refuses packed False or the rows form).

    Example:
        >>> mx_qk_keys("RA", False, True, {}), mx_qk_keys("off", True, False, {}), mx_qk_keys(None, False, True, {})
        ((True, False, None), (False, None, None), (False, True, {}))
    """
    if mx_qk == "RA":
        return True, False, mla_k1_rows_options or None
    if mx_qk == "off":
        # A rendered rows form comes with packed False: the BF16 latent QK then takes the packed loop (the default).
        return False, None if mla_k1_rows else mla_k1_packed, None
    return mla_k1_rows, mla_k1_packed, mla_k1_rows_options


def check_mx_qk(
    world_size: int,
    queries: int,
    rows: int,
    mla_mode: str,
    mla_loop_path: bool,
    mx_qk: str,
    mla_fp8_pages: bool,
    mla_k1_rows: bool,
    mla_k1_packed: bool,
    pack_options: dict,
):
    """Refuse an ``mx_qk`` (``MX_QK_TRN3_MODES``) that the form or the resolved k1 keys of the call do not take:
    "E" in a form outside ``MX_QK_TRN3_MODES["E"]`` or without the MX latent QK of the packed p2 loop with word_keys; "RA" in a
    form outside ``MX_QK_TRN3_MODES["RA"]``, without FP8 pages or off the rows form; "off" with the rows form or a packed-loop
    ``mx_qk``. None takes every call.

    Example:
        >>> check_mx_qk(256, 8, 32, "k1", True, "RA", True, True, False, None) is None
        True
    """
    kernel_assert(
        mx_qk is None or mx_qk in MX_QK_TRN3_MODES,
        f"mx_qk must be None or one of {tuple(MX_QK_TRN3_MODES)}, got {mx_qk!r}",
    )
    if mx_qk is None:
        return
    form = next(
        (name for name, value in SERVED_FORMS.items() if value == (form_world(world_size), queries, rows)), None
    )
    if mla_mode != "k1" or not mla_loop_path:
        form = None
    options = pack_options or {}
    if mx_qk == "E":
        kernel_assert(
            form in MX_QK_TRN3_MODES["E"]
            and mla_k1_packed
            and not mla_k1_rows
            and options.get("p2")
            and options.get("word_keys")
            and options.get("mx_qk"),
            f"mx_qk 'E' is the MX latent QK of the packed p2 loop with word_keys in the forms {MX_QK_TRN3_MODES['E']} on the k1 "
            f"loop path, got the form {form} with mla_k1_packed {mla_k1_packed} and pack options {options}",
        )
    elif mx_qk == "RA":
        kernel_assert(
            form in MX_QK_TRN3_MODES["RA"] and mla_fp8_pages and mla_k1_rows,
            f"mx_qk 'RA' is the CP32 rows form with FP8 pages in the forms {MX_QK_TRN3_MODES['RA']} on the k1 loop path, got the "
            f"form {form} with mla_fp8_pages {mla_fp8_pages} and mla_k1_rows {mla_k1_rows}",
        )
    else:
        kernel_assert(
            not mla_k1_rows and not options.get("mx_qk"),
            "mx_qk 'off' is the BF16 latent QK: not with the rows form (its MX latent QK) or a packed-loop mx_qk, got "
            f"mla_k1_rows {mla_k1_rows} and pack options {options}",
        )


VERIFY_TOKEN_CAPS = {
    "routed_compact_cross_expert_order": 256,
    "latent_gather_mx": 256,
    "cross_expert_order": 256,
    "expert_prefetch": 256,
    "routed_per_expert": 256,
}
"""The gathered-token caps of the verify entry that a caller (the serving spec step) reads, by key: with ``routed_compact`` in
the MoE glue, the cross-expert order takes expert calls of up to 256 gathered tokens (the k7 form), since the first
slot tile of the compaction is a static call of at most 128 routed tokens and a later slot tile runs the slot form.
Without the compaction the order takes them too (``cross_expert_order``): a token-tiled call of 256 tokens (2 tiles)
runs it with 3 whole weight buffers and no tile blocks of either order.
``latent_gather_mx`` (the MXFP8 latent rows) takes token-tiled expert calls of up to 256 gathered tokens, each
128-token tile of whole replicas, with ``routed_compact`` off or on (the compaction takes it:
``tkg_moe_2srv.ROUTED_COMPACT_LATENT_GATHER_MX``).
``expert_prefetch`` (``moe_expert_prefetch``) takes token-tiled expert calls of up to 256 gathered tokens with 1 to 3
whole weight buffers (3 with the cross-expert order); with ``routed_compact`` the static call of the first slot tile
takes the ring.
``routed_per_expert`` in the MoE glue runs the token-tiled expert calls of 256 gathered tokens per local expert; the body
takes it only on such calls (the k7 form), with ``routed_compact`` off."""


def check_fp8_pv(fp8_pv: bool, mx_qk: str, pack_options: dict):
    """Refuse a given ``mla_fp8_pv`` True that the call's packed-loop options do not take: the FP8 P@V pass rides the
    ``mx_qk`` "E" set of the packed p2 loop of an ``FP8_PV_FORMS`` form (``served_pack_options``). None and False take
    every call.

    Example:
        >>> check_fp8_pv(None, None, None) is None, check_fp8_pv(False, "E", {}) is None
        (True, True)
    """
    kernel_assert(
        not fp8_pv or (mx_qk == "E" and (pack_options or {}).get("fp8_pv")),
        f"mla_fp8_pv needs mx_qk 'E' on the packed p2 loop of a {' / '.join(FP8_PV_FORMS)} form with word_keys "
        "(served_pack_options), "
        f"got mx_qk={mx_qk!r} and the pack options {pack_options!r}",
    )


def check_fp8_pv_split(fp8_pv: bool, split_args: dict):
    """Refuse a given ``mla_fp8_pv`` True with any ``mla_split*`` argument set: the long-sequence split program does not
    take the FP8 P@V pass (its pack options drop ``fp8_pv``, and the pass builds its items from the positions only, with
    no split group range). ``split_args``: the ``mla_split*`` arguments of the call by name.

    Example:
        >>> check_fp8_pv_split(True, {"mla_split": None}) is None, check_fp8_pv_split(None, {"mla_split": {}}) is None
        (True, True)
    """
    given = sorted(name for name, value in split_args.items() if value is not None)
    kernel_assert(
        not fp8_pv or not given,
        f"mla_fp8_pv does not take the long-sequence split: turn one off (split arguments given: {given})",
    )


def served_defaults(world_size: int, queries: int, rows: int, keys: dict) -> dict:
    """Return the default of each served-form lever of a verify call.

    ``keys`` holds the values of the entry's arguments (None for a lever the caller leaves to its default). In a
    ``SERVED_FORMS`` form on the k1 loop path, a lever is on by default where the kernel has a result for that form
    (a measured gain, a bitwise identity or a simulator check) and the other keys of the call allow it; every other
    call gets the keys-off values. A value that the caller gives wins.

    Example:
        >>> keys = dict(mla_mode="k1", mla_loop_path=True, split_call=False, kda_replay="records")
        >>> served_defaults(256, 4, 16, keys)["moe_routed_reduce"], served_defaults(256, 8, 32, keys)["mla_k1_packed"]
        ('rs_ag', True)
    """
    form = next(
        (name for name, value in SERVED_FORMS.items() if value == (form_world(world_size), queries, rows)), None
    )
    if keys.get("mla_mode") != "k1" or not keys.get("mla_loop_path"):
        form = None
    k3, served = form in K3_FORMS, form is not None

    def given(name, default):
        return default if keys.get(name) is None else keys[name]

    # KDA_PREWORK_AHEAD_FORM_SET: the moe_glue key kda_prework_ahead of the form (given, else by default on the one-call
    # states verify); a form of the set keeps its state-slot levers with the prework, another form drops them.
    prework_given = (keys.get("moe_glue") or {}).get("kda_prework_ahead")
    prework_form = form in KDA_PREWORK_AHEAD_FORM_SET
    prework_served = served and prework_form and keys.get("kda_replay") == "states" and not keys.get("split_call")
    prework_glue = prework_served if prework_given is None else bool(prework_given)
    states = keys.get("kda_replay") == "states" and (
        prework_form or not (keys.get("kda_prework_ahead") or prework_glue)
    )
    token_tiles = given("kda_token_tiles", form in KDA_TOKEN_TILES_FORM_SET and states)
    # The per-token slot-state ring of the KDA verify: the forms of the ring-depth and store-queue tables.
    ring = (
        states
        and bool(token_tiles)
        and keys.get("kda_chain", "sequential") == "sequential"
        and not keys.get("kda_shared_state_ring")
    )
    # The slot DMA levers move only bf16 conv rows and int32 slot ids, at the same page offsets in every state dtype.
    slot_dmas = form in KDA_SLOT_DMA_FORMS and states
    # The k3 form with FP8 pages takes the CP32 rows form of the k1 call (the FP8 double-row scores need FP8 pages),
    # unless the caller asks for the packed loop or mx_qk "off" (the BF16 latent QK). mx_qk "RA" gives the rows form to
    # the MX_QK_TRN3_MODES["RA"] forms with FP8 pages.
    mx_qk = keys.get("mx_qk")
    fp8_pages = bool(keys.get("mla_fp8_pages"))
    fp8_k3 = form == "k3" and fp8_pages
    fp8_rows = (fp8_k3 and mx_qk != "off") or (mx_qk == "RA" and form in MX_QK_TRN3_MODES["RA"] and fp8_pages)
    packed_asked = keys.get("mla_k1_packed") or keys.get("mla_k1_pack_options")
    rows_form = given("mla_k1_rows", fp8_rows and not packed_asked)
    # With the runner's live rows the served rows form also skips the padding sequences (bitwise for the live rows).
    served_rows = {**SERVED_ROWS_OPTIONS, **({"rows_pad_skip": True} if keys.get("live_rows") else {})}
    rows_options = given("mla_k1_rows_options", served_rows if rows_form and fp8_rows else {})
    bank_dma = keys.get("attn_res_bank_dma", "auto")
    cached = given("attn_res_cached_bank_sums", served and not keys.get("split_call") and not keys.get("mix_cache"))
    early_sums = given("attn_res_early_sums", cached)
    critical = given("mla_critical_dma_priority", served)
    # rs_ag takes the forms of VERIFY_ROUTED_RS_AG_FORMS only. k3b2 and k7b1 (R = 8) keep the routed all-reduce
    # by default: the 8-row rs_ag has no accuracy check yet.
    routed_reduce = given(
        "moe_routed_reduce",
        "rs_ag"
        if served
        # k7b2 keeps the routed all-reduce: rs_ag changes the bits (at most 1 BF16 ulp).
        and form != "k7b2"
        and (world_size, rows) in VERIFY_ROUTED_RS_AG_FORMS
        and rows >= SERVED_ROUTED_RS_AG_MIN_ROWS
        and given("cc_source_pingpong", True)
        and not keys.get("moe_routed_own_rows")
        else "ar",
    )
    one_call = given("moe_expert_call_tokens", EXPERT_CALL_AUTO) in (EXPERT_CALL_AUTO, EXPERT_CALL_TOKENS)
    # The expert ring prefetch is off by default in every form: it showed no measured step win.
    prefetch = dict(keys.get("moe_expert_prefetch") or {})
    glue = {**SERVED_GLUE, **({"out_proj_early": False} if routed_reduce == "rs_fold" else {})}
    if form in ROUTED_COMPACT_SERVED_IN and given("moe_gather", "dp_tp") == "dp_tp":
        glue["routed_compact"] = ROUTED_COMPACT_SERVED_VALUE
        glue["weights_before_compact"] = True
    # The static MX scale rows across the 4 quadrants (the same bytes): SCALE_ROWS_QUADRANTS_FORM_SET.
    if form in SCALE_ROWS_QUADRANTS_FORM_SET:
        glue["scale_rows_quadrants"] = True
    moe_glue = given("moe_glue", glue if served else {})
    # tkg_moe_2srv.ROUTED_COMPACT_ON False (the kernel-side kill switch) drops the compaction from any glue.
    compact_keys = moe_2srv_module.COMPACT_GLUE_KEYS
    if not moe_2srv_module.ROUTED_COMPACT_ON and any(key in (moe_glue or {}) for key in compact_keys):
        moe_glue = {key: value for key, value in moe_glue.items() if key not in compact_keys}
    # SERVED_GLUE_ADDITIONS_BY_FORM: the bitwise glue keys of the form, each where the glue does not set it (a given
    # value wins, an off value included), with a compaction that the kill switch keeps.
    additions = SERVED_GLUE_ADDITIONS_BY_FORM.get(form, {}) if served else {}
    if keys.get("moe_latent_gather_mx"):
        # weights_after_rows waits for the BF16 / FP8 router-row loads; the body refuses it with the G1a rows
        # (latent_gather_mx), so a G1a call takes no such addition.
        additions = {key: value for key, value in additions.items() if key != "weights_after_rows"}
    if additions and (moe_glue or {}).get("routed_compact"):
        # tail_store_halves splits the stores of the hot_last tail: another compaction value takes no such key.
        hot_last = moe_glue["routed_compact"] == "hot_last"
        moe_glue = {
            **moe_glue,
            **{
                key: value
                for key, value in additions.items()
                if key not in moe_glue and (hot_last or key not in HOT_LAST_GLUE_ADDITIONS)
            },
        }
    # ATTN_RES_AHEAD_VECTOR_BY_FORM: the glue key ahead_sums_vector of the form, where the glue does not set it, and only
    # with attn_res_ahead / attn_res_ahead_mlp (the bank parts it acts on; the entry refuses the key without them).
    ahead_keys = keys.get("attn_res_ahead") or keys.get("attn_res_ahead_mlp")
    ahead_vector = ATTN_RES_AHEAD_VECTOR_BY_FORM.get(form) if served and ahead_keys else None
    if ahead_vector and "ahead_sums_vector" not in (moe_glue or {}):
        moe_glue = {**(moe_glue or {}), "ahead_sums_vector": ahead_vector}
    # ATTN_RES_AHEAD_LOAD_TILES_BY_FORM: the glue key ahead_load_tiles of the form, with the ahead forms (as above).
    # KDA_SLOT_RING_PER_CALL_BY_FORM: the glue key kda_slot_ring_per_call of the form, on the per-token states ring
    # without the per-token persistent ring (a given value wins).
    ring_per_call = (
        KDA_SLOT_RING_PER_CALL_BY_FORM.get(form)
        if served and ring and not keys.get("kda_slot_state_per_token")
        else None
    )
    if ring_per_call and "kda_slot_ring_per_call" not in (moe_glue or {}):
        moe_glue = {**(moe_glue or {}), "kda_slot_ring_per_call": ring_per_call}
    # KDA_WEIGHT_DMA_BY_FORM: the glue key kda_weight_dma of the form.
    weight_dma = KDA_WEIGHT_DMA_BY_FORM.get(form) if served else None
    if weight_dma and "kda_weight_dma" not in (moe_glue or {}):
        moe_glue = {**(moe_glue or {}), "kda_weight_dma": weight_dma}
    # KDA_W_OUT_AFTER_W_IN_FILLS_BY_FORM: the glue key kda_w_out_after_w_in_fills of the form, with the MXFP8 kda_w_in.
    w_out_fills = KDA_W_OUT_AFTER_W_IN_FILLS_BY_FORM.get(form) if served and keys.get("kda_w_in_mx") else None
    if w_out_fills and "kda_w_out_after_w_in_fills" not in (moe_glue or {}):
        moe_glue = {**(moe_glue or {}), "kda_w_out_after_w_in_fills": w_out_fills}
    # KDA_PREWORK_AHEAD_FORM_SET: the glue key kda_prework_ahead of the form, on the one-call states verify.
    if prework_served and prework_given is None:
        moe_glue = {**(moe_glue or {}), "kda_prework_ahead": True}
    # KDA_COMMIT_AFTER_W_IN_BY_FORM: the glue key kda_commit_after_w_in of the form, on the per-token states ring.
    commit_after = KDA_COMMIT_AFTER_W_IN_BY_FORM.get(form) if served and ring else None
    if commit_after and "kda_commit_after_w_in" not in (moe_glue or {}):
        moe_glue = {**(moe_glue or {}), "kda_commit_after_w_in": commit_after}
    # TAIL_ZERO_EARLY_FORM_SET: the glue key tail_zero_early of the form, where the glue does not set it, and only with
    # the keys it needs (the hot_last compaction, ungather_skip, the tail after the guard).
    glue_now = moe_glue or {}
    if (
        served
        and form in TAIL_ZERO_EARLY_FORM_SET
        and "tail_zero_early" not in glue_now
        and glue_now.get("routed_compact") == "hot_last"
        and glue_now.get("ungather_skip")
        and not glue_now.get("tail_before_guard")
    ):
        moe_glue = {**glue_now, "tail_zero_early": True}
    # KDA_SLOT_STORE_REGS_BY_FORM: the glue key kda_slot_store_regs of the form, with one Sync slot-store queue.
    store_sync = (
        given("kda_slot_store_queues", 1) == 1
        and given("kda_slot_store_engine", KDA_SLOT_STORE_ENGINE_BY_FORM.get(form, "sync") if ring else "sync")
        == "sync"
    )
    store_regs = KDA_SLOT_STORE_REGS_BY_FORM.get(form) if served and states and store_sync else None
    if store_regs and "kda_slot_store_regs" not in (moe_glue or {}):
        moe_glue = {**(moe_glue or {}), "kda_slot_store_regs": store_regs}
    # KDA_SLOT_CONV_AFTER_AR_BY_FORM: the glue key kda_slot_conv_after_ar of the form, on the per-token states chain.
    conv_after_ar = (
        KDA_SLOT_CONV_AFTER_AR_BY_FORM.get(form)
        if served and states and bool(token_tiles) and keys.get("kda_chain", "sequential") == "sequential"
        else None
    )
    if conv_after_ar and "kda_slot_conv_after_ar" not in (moe_glue or {}):
        moe_glue = {**(moe_glue or {}), "kda_slot_conv_after_ar": conv_after_ar}
    # MOE_FIRST_ADD_FORM_SET: the glue key first_add_zero of the form, where the glue does not set it.
    if served and form in MOE_FIRST_ADD_FORM_SET and "first_add_zero" not in (moe_glue or {}):
        moe_glue = {**(moe_glue or {}), "first_add_zero": True}
    # MOE_OUT_PROJ_BEFORE_IN_PROJ_FORM_SET: the glue key out_proj_before_in_proj of the form, with out_proj_early.
    glue_now = moe_glue or {}
    if (
        served
        and form in MOE_OUT_PROJ_BEFORE_IN_PROJ_FORM_SET
        and glue_now.get("out_proj_early")
        and "out_proj_before_in_proj" not in glue_now
    ):
        moe_glue = {**glue_now, "out_proj_before_in_proj": True}
    # MOE_SCALE_FILL_AFTER_PREFETCH_FORM_SET: the glue key scale_fill_after_prefetch of the form, with the
    # latent_gather_mx loads.
    if (
        served
        and form in MOE_SCALE_FILL_AFTER_PREFETCH_FORM_SET
        and keys.get("moe_latent_gather_mx")
        and "scale_fill_after_prefetch" not in (moe_glue or {})
    ):
        moe_glue = {**(moe_glue or {}), "scale_fill_after_prefetch": True}
    # MOE_OUT_PROJ_PRIORITY_BY_FORM: the glue key out_proj_priority of the form, where the glue does not set it.
    out_proj_level = MOE_OUT_PROJ_PRIORITY_BY_FORM.get(form) if served else None
    if out_proj_level is not None and "out_proj_priority" not in (moe_glue or {}):
        moe_glue = {**(moe_glue or {}), "out_proj_priority": out_proj_level}
    ahead_tiles = ATTN_RES_AHEAD_LOAD_TILES_BY_FORM.get(form) if served and ahead_keys else None
    if ahead_tiles and "ahead_load_tiles" not in (moe_glue or {}):
        moe_glue = {**(moe_glue or {}), "ahead_load_tiles": ahead_tiles}
    # ATTN_RES_AHEAD_GRAM_MERGE_BY_FORM: the glue key ahead_gram_merge of the form, with the ahead forms (as above).
    gram_merge = ATTN_RES_AHEAD_GRAM_MERGE_BY_FORM.get(form) if served and ahead_keys else None
    if gram_merge and "ahead_gram_merge" not in (moe_glue or {}):
        moe_glue = {**(moe_glue or {}), "ahead_gram_merge": gram_merge}
    # ATTN_RES_AHEAD_BEFORE_RESID_BY_FORM: the glue key ahead_before_resid of the form, with attn_res_ahead only.
    before_resid = ATTN_RES_AHEAD_BEFORE_RESID_BY_FORM.get(form) if served and keys.get("attn_res_ahead") else None
    if before_resid and "ahead_before_resid" not in (moe_glue or {}):
        moe_glue = {**(moe_glue or {}), "ahead_before_resid": before_resid}
    # ATTN_RES_CACHED_ACT_SQUARES_BY_FORM: the glue key cached_act_squares of the form, with the cached early sums.
    act_squares = ATTN_RES_CACHED_ACT_SQUARES_BY_FORM.get(form) if served and cached and early_sums else None
    if act_squares and "cached_act_squares" not in (moe_glue or {}):
        moe_glue = {**(moe_glue or {}), "cached_act_squares": act_squares}
    # KDA_IN_PROJ_X_STATIONARY_FORM_SET: the moe_glue key kda_in_proj_x_stationary of the form, on the one-call verify
    # with the MXFP8 kda_w_in (the KDA call refuses it with a BF16 one), where the glue does not set it.
    if (
        served
        and form in KDA_IN_PROJ_X_STATIONARY_FORM_SET
        and not keys.get("split_call")
        and keys.get("kda_w_in_mx")
        and "kda_in_proj_x_stationary" not in (moe_glue or {})
    ):
        moe_glue = {**(moe_glue or {}), "kda_in_proj_x_stationary": True}
    # KDA_STATE_LOADS_AFTER_RING_FORM_SET: the moe_glue key kda_state_loads_after_ring of the form, on the one-call
    # verify with kda_state_loads_first, where the glue does not set it.
    if (
        served
        and form in KDA_STATE_LOADS_AFTER_RING_FORM_SET
        and not keys.get("split_call")
        and given("kda_state_loads_first", form in KDA_STATE_LOADS_FIRST_FORM_SET and states)
        and "kda_state_loads_after_ring" not in (moe_glue or {})
    ):
        moe_glue = {**(moe_glue or {}), "kda_state_loads_after_ring": True}
    # ATTN_RES_AHEAD_MLP_KDA_TAIL_BY_FORM: the glue key ahead_mlp_kda_tail of the form, with attn_res_ahead_mlp only.
    mlp_tail = ATTN_RES_AHEAD_MLP_KDA_TAIL_BY_FORM.get(form) if served and keys.get("attn_res_ahead_mlp") else None
    if mlp_tail and "ahead_mlp_kda_tail" not in (moe_glue or {}):
        moe_glue = {**(moe_glue or {}), "ahead_mlp_kda_tail": mlp_tail}
    # KDA_MLP_TAIL_ENTRIES_BY_FORM: the glue key kda_mlp_tail_entries of the form, with the cached early sums only.
    kda_tail = KDA_MLP_TAIL_ENTRIES_BY_FORM.get(form) if served and cached and early_sums else None
    if kda_tail and "kda_mlp_tail_entries" not in (moe_glue or {}):
        moe_glue = {**(moe_glue or {}), "kda_mlp_tail_entries": kda_tail}
    compacts = bool((moe_glue or {}).get("routed_compact"))
    # COMPACT_ORDER_FORM_SET: the order exists only with the compaction, so a glue without it drops a given order, its
    # transpose block and its 3 weight buffers (the call then gets the program of a glue and keys without them).
    # TOKEN_TILES_CROSS_ORDER: while the form takes no served compaction (and the kill switch is on), a GIVEN order
    # runs on the token-tiled call instead; without a given order nothing changes.
    tiled_order = (
        moe_2srv_module.TOKEN_TILES_CROSS_ORDER
        and moe_2srv_module.ROUTED_COMPACT_ON
        and form not in ROUTED_COMPACT_SERVED_IN
        and keys.get("moe_cross_expert_order") is True
    )
    order_off = form in COMPACT_ORDER_FORM_SET and not compacts and not tiled_order
    # The cross-expert order holds one expert call of the gathered tokens: k3 (T = 128) and k3b2 (T = 64), not k3b8
    # (T = 256); a form of COMPACT_ORDER_FORM_SET (T = 256) takes it with a glue that has routed_compact, whose first slot
    # tile is a static call of 128.
    cross = given(
        "moe_cross_expert_order",
        (form in CROSS_ORDER_FORMS or (form in COMPACT_ORDER_FORM_SET and compacts))
        and keys.get("moe_transpose_lookahead") is None
        and given("moe_expert_queues", 1) == 1
        # A prefetched ring of the cross-expert order holds its 3 whole buffers. kda_after_in_proj is a placement flag,
        # not a ring depth.
        and all(depth in (0, 3) for kind, depth in prefetch.items() if kind != "kda_after_in_proj")
        and not keys.get("moe_half_weight_buffers")
        and given("moe_weight_buffers", 3) == 3
        and one_call,
    )
    # The routed compaction runs the slot form of moe_tkg_decode, which takes no alternating tiles.
    alt_tiles = MOE_STATIC_ALT_TILE_DEFAULTS.get(form, False) if not (moe_glue or {}).get("routed_compact") else False
    early_mlp = SERVED_EARLY_MLP_ENTRIES if served and not (cached or early_sums or keys.get("mix_cache")) else 0
    # The served rows form of the k3 form with FP8 pages also sums the bank rows of each MLA layer's MLP mix in the flight
    # of its state reduce-scatter (bitwise; the same instructions).
    # MLA_MLP_EARLY_STATE_RS_FORM_SET: the forms whose BF16-o state reduce-scatter traces the sums in its flight.
    state_rs_early = form in MLA_MLP_EARLY_STATE_RS_FORM_SET and bool(keys.get("mla_state_bf16_o")) and not rows_form
    mla_mlp_early = given("attn_res_mla_mlp_early", bool(((rows_form and fp8_k3) or state_rs_early) and early_sums))
    # MLA_MLP_TAIL_ENTRIES_BY_FORM: the glue key mla_mlp_tail_entries of the form, with the cached early sums and the MLA
    # layers' early MLP sums only.
    mla_tail = MLA_MLP_TAIL_ENTRIES_BY_FORM.get(form) if served and cached and early_sums and mla_mlp_early else None
    if mla_tail and "mla_mlp_tail_entries" not in (moe_glue or {}):
        moe_glue = {**(moe_glue or {}), "mla_mlp_tail_entries": mla_tail}
    # COMBO_ROUTED_REDUCES_BY_FORM: the routed reductions with which the form takes the keys below by default.
    combo_tail = routed_reduce in COMBO_ROUTED_REDUCES_BY_FORM.get(form, (routed_reduce,))
    states_after_out_ar = given(
        "kda_states_after_out_ar",
        form in KDA_STATES_AFTER_OUT_AR_FORM_SET and combo_tail and not keys.get("kda_prework_ahead"),
    )
    row_sums_form = form in ROUTED_ROW_SUM_SERVED_IN and combo_tail
    sums_after_routed = given("moe_sums_after_routed", row_sums_form)
    kernel_assert(
        not states_after_out_ar or form not in K3_FORMS,
        f"kda_states_after_out_ar is not measured on the k3 forms {K3_FORMS}: they refuse it and keep their program "
        f"until their own row (fail closed). The edge is measured on k7b2 (with the state and slot-page loads first); "
        f"the other forms take a given True. Got form {form!r}",
    )
    kernel_assert(
        not (states_after_out_ar and keys.get("kda_prework_ahead")),
        "kda_states_after_out_ar orders the inline KDA state loads after the MoE output all-reduce: not with "
        "kda_prework_ahead, whose KDA prework traces those loads before the call",
    )
    # The MoE tails that call no glue hook (tkg_moe_2srv refuses glue_hooks there): rs_fold, and the
    # token_major norm tail of the routed all-reduce. The two keys below take their edges from those hooks.
    norm_tail = (moe_glue or {}).get("norm_tail", moe_2srv_module.GLUE_KEYS["norm_tail"])
    tail_hooks = routed_reduce != "rs_fold" and not (routed_reduce == "ar" and norm_tail == "token_major")
    kernel_assert(
        tail_hooks or not (states_after_out_ar or sums_after_routed),
        f"kda_states_after_out_ar ({states_after_out_ar}) and moe_sums_after_routed ({sums_after_routed}) take their "
        f"edges from the MoE glue hooks (out_trigger, routed_trigger), and moe_routed_reduce {routed_reduce!r} with "
        f"norm_tail {norm_tail!r} calls no hook (rs_fold and the token_major norm tail call none). Give both False "
        f"there (form {form!r} gives moe_sums_after_routed {row_sums_form} by default on this tail)",
    )
    return {
        "mla_out_ar": given("mla_out_ar", served),
        "ffn_weights_after_attn": given(
            "ffn_weights_after_attn",
            form in FFN_AFTER_ATTN_FORMS and (form != "k7b2" or FFN_AFTER_ATTN_K7B2) and bool(given("dma_gates", True)),
        ),
        "mla_k1_rows": rows_form,
        "mla_k1_rows_options": rows_options,
        "mla_k1_packed": given("mla_k1_packed", form in PACKED_FORMS and not rows_form),
        # The bitwise MLA levers of the one-server k1 verify (the output does not change).
        # QUERY_DMA_FORMS: on (True, as the serving caller passes it for the served forms) takes the query level.
        "mla_critical_dma_priority": "query" if critical is True and form in QUERY_DMA_FORMS else critical,
        "mla_cache_write_off_sp": given("mla_cache_write_off_sp", served),
        "mla_gemv_bands": given("mla_gemv_bands", served),
        # The p2 loop-entry form (bitwise): P2_ENTRY_ACT_FORMS.
        "mla_p2_entry_act": given("mla_p2_entry_act", form in P2_ENTRY_ACT_FORMS),
        # The item plans and rows of the call built once for its MLA layers (bitwise): STEP_ITEMS_FORMS.
        "mla_step_items": given("mla_step_items", form in STEP_ITEMS_FORMS),
        # The query prefetch of the later passes (bitwise): PASS_Q_PREFETCH_FORM_SET.
        "mla_pass_q_prefetch": given(
            "mla_pass_q_prefetch",
            PASS_Q_PREFETCH_TILES_BY_FORM.get(form, PASS_Q_PREFETCH_TILES) if form in PASS_Q_PREFETCH_FORM_SET else 0,
        ),
        # Two query gather tiles in turn (bitwise): Q_GATHER_BUFFER_FORM_SET.
        "mla_q_gather_buffers": given("mla_q_gather_buffers", 2 if form in Q_GATHER_BUFFER_FORM_SET else 1),
        # The tail weights after the call-level state reduce-scatter (bitwise): TAIL_WEIGHTS_AFTER_RS_FORM_SET.
        "mla_tail_weights_after_rs": given("mla_tail_weights_after_rs", form in TAIL_WEIGHTS_AFTER_RS_FORM_SET),
        # The pass setup off GpSimd (bitwise): PASS_SETUP_OFF_POOL_FORM_SET.
        "mla_pass_setup_off_pool": given("mla_pass_setup_off_pool", form in PASS_SETUP_OFF_POOL_FORM_SET),
        # The slot rows from the SBUF stage tile (bitwise): SLOT_ROWS_SBUF_FORM_SET.
        "mla_slot_rows_sbuf": given("mla_slot_rows_sbuf", form in SLOT_ROWS_SBUF_FORM_SET),
        # With them, no HBM store of the stage rows (bitwise): SKIP_STAGE_STORE_FORM_SET.
        "mla_skip_stage_store": given("mla_skip_stage_store", form in SKIP_STAGE_STORE_FORM_SET),
        # The next pass's own-key rows and query tiles ahead of the state stores (bitwise): PASS_ROWS_AHEAD_FORM_SET.
        "mla_pass_rows_ahead": given("mla_pass_rows_ahead", form in PASS_ROWS_AHEAD_FORM_SET),
        # One vector-indirect write of the current MLA rows (the rows that the rank does not own and the pad rows to
        # the null page): BATCHED_CACHE_WRITE_FORMS.
        "mla_batched_cache_write": given("mla_batched_cache_write", form in BATCHED_CACHE_WRITE_FORMS),
        # The null-page row column of the batched cache write built once a kernel (bitwise): NULL_ROWS_ONCE_FORMS.
        "mla_null_rows_once": given("mla_null_rows_once", form in NULL_ROWS_ONCE_FORMS),
        "norm_scalar_squares": given("norm_scalar_squares", NORM_SCALAR_SQUARES_BY_FORM.get(form, 0)),
        "attn_res_scalar_products": given("attn_res_scalar_products", ATTN_RES_SCALAR_PRODUCTS_BY_FORM.get(form, 0)),
        "attn_res_bank_dma": ("act" if served else None) if bank_dma == "auto" else bank_dma,
        "attn_res_cached_bank_sums": cached,
        "attn_res_early_sums": early_sums,
        "attn_res_mla_mlp_early": mla_mlp_early,
        "attn_res_weight_rows": given("attn_res_weight_rows", form in ATTN_RES_WEIGHT_ROW_FORMS),
        # The next KDA layer's state loads after the MoE output all-reduce trigger (bitwise):
        # KDA_STATES_AFTER_OUT_AR_FORM_SET with COMBO_ROUTED_REDUCES_BY_FORM; a given value wins.
        "kda_states_after_out_ar": states_after_out_ar,
        "attn_res_mlp_early_entries": given("attn_res_mlp_early_entries", early_mlp),
        "kda_w_in_buffers": given("kda_w_in_buffers", KDA_W_IN_BUFFERS_BY_FORM.get(form, 0) if states else 0),
        "kda_state_loads_first": given("kda_state_loads_first", form in KDA_STATE_LOADS_FIRST_FORM_SET and states),
        "kda_slot_pages_first": given("kda_slot_pages_first", form in KDA_SLOT_PAGES_FIRST_FORM_SET and states),
        "kda_token_tiles": token_tiles,
        # The slot DMA levers of the KDA verify (KDA_SLOT_DMA_FORMS). Live pages do not change. The conv and slot conv
        # scatters write zeros to the page of an off row (page 0 or the KDA null page). The persistent slot conv sources
        # take the per-token tiles of the sequential chain.
        "kda_conv_load_gather": given("kda_conv_load_gather", slot_dmas),
        "kda_batched_conv_stores": given("kda_batched_conv_stores", slot_dmas),
        # The paired fp32 KDA pages take the slot page ids as they are (the KDA verify refuses the contiguous load).
        "kda_slot_pages_contiguous": given("kda_slot_pages_contiguous", slot_dmas and not keys.get("kda_paired_pages")),
        "kda_batched_slot_conv_stores": given("kda_batched_slot_conv_stores", slot_dmas),
        "kda_bf16_update_rows": given("kda_bf16_update_rows", slot_dmas),
        "kda_slot_conv_persistent": given(
            "kda_slot_conv_persistent",
            slot_dmas and bool(token_tiles) and keys.get("kda_chain", "sequential") == "sequential",
        ),
        "kda_update_bcast_depth": given(
            "kda_update_bcast_depth", KDA_UPDATE_BCAST_DEPTH_BY_FORM.get(form, 2) if states else 2
        ),
        # The output-gate blocks of the MXFP8 input projection after the decay gate (bitwise): KDA_HEAD_PIPELINE_BY_FORM.
        "kda_head_pipeline": given("kda_head_pipeline", KDA_HEAD_PIPELINE_BY_FORM.get(form, "off")),
        # ACCURACY CLASS: the out all-reduce as two row chunks (KDA_OUT_AR_ROWS_FORM_SET, empty).
        "kda_out_ar_rows": given("kda_out_ar_rows", form in KDA_OUT_AR_ROWS_FORM_SET),
        # The slot conv windows from the conv inputs take the per-token state tiles of the sequential chain.
        "kda_slot_conv_from_inputs": given(
            "kda_slot_conv_from_inputs",
            form in KDA_SLOT_CONV_FROM_INPUTS_FORM_SET
            and states
            and bool(token_tiles)
            and keys.get("kda_chain", "sequential") == "sequential",
        ),
        # One slot-store queue: a second queue on ScalarE puts the stores in the chain's ScalarE program order.
        "kda_slot_store_queues": given("kda_slot_store_queues", 1),
        # One slot-state ring tile for each scored token of the sequential chain: off by default.
        "kda_slot_state_per_token": given("kda_slot_state_per_token", False),
        # The slot-state ring depth and the queue of its stores: KDA_SLOT_STATE_DEPTH_BY_FORM /
        # KDA_SLOT_STORE_ENGINE_BY_FORM with the state slots, the per-token tiles and the sequential chain, without the
        # shared ring; a given value wins.
        "kda_slot_state_depth": given("kda_slot_state_depth", KDA_SLOT_STATE_DEPTH_BY_FORM.get(form, 2) if ring else 2),
        "kda_slot_store_engine": given(
            "kda_slot_store_engine", KDA_SLOT_STORE_ENGINE_BY_FORM.get(form, "sync") if ring else "sync"
        ),
        # KDA_CONV_PRODUCTS_ENGINE_BY_FORM: the engine of the KDA conv tap products; a given value wins.
        "kda_conv_products_engine": given(
            "kda_conv_products_engine", KDA_CONV_PRODUCTS_ENGINE_BY_FORM.get(form, "gpsimd")
        ),
        # The step-front load of the stream rows in column chunks (load_rows_hT_batched): off by default.
        "rows_hT_batched": given("rows_hT_batched", False),
        # The ring of AUX_RING_DEPTH staging tiles of the aux row stores: AUX_ROW_RING_FORM_SET (the k7 shape also off
        # the k1 loop path); a given value wins.
        "aux_row_ring": given(
            "aux_row_ring",
            (form_world(world_size), queries, rows) == SERVED_FORMS["k7"] or form in AUX_ROW_RING_FORM_SET,
        ),
        # The KDA MLP bank sums right after the MXFP8 input quantize: KDA_X_QUANT_HOOK_FORM_SET (by the served shape,
        # also off the k1 loop path); a given value wins.
        "kda_x_quant_hook": given(
            "kda_x_quant_hook",
            any((form_world(world_size), queries, rows) == SERVED_FORMS[name] for name in KDA_X_QUANT_HOOK_FORM_SET),
        ),
        # The SBUF -> SBUF KDA output all-reduce: on in KDA_OUT_AR_SBUF_FORM_SET (empty: off by default).
        "kda_out_ar_sbuf": given("kda_out_ar_sbuf", form in KDA_OUT_AR_SBUF_FORM_SET),
        # One state load a sequence from its selected page (bitwise): KDA_MERGED_STATE_LOADS_FORM_SET, with the slots.
        "kda_merged_state_loads": given("kda_merged_state_loads", form in KDA_MERGED_STATE_LOADS_FORM_SET and states),
        # The KDA GpSimd weight loads after the FFN output all-reduce (bitwise): KDA_SMALL_AFTER_OUT_AR_FORM_SET; a given
        # value wins.
        "kda_small_after_out_ar": given(
            "kda_small_after_out_ar", form in KDA_SMALL_AFTER_OUT_AR_FORM_SET and KDA_SMALL_AFTER_OUT_AR
        ),
        "moe_rows_load_split": given(
            "moe_rows_load_split",
            4
            if served
            and given("moe_gather", "dp_tp") == "dp_tp"
            and not keys.get("moe_rows_latent_fp8")
            and not keys.get("moe_latent_gather_mx")
            else 1,
        ),
        "moe_cross_expert_order": False if order_off else cross,
        # One block of transposes an expert: the PE pays a drain per gap of transposes between two MX matmuls.
        "moe_transpose_block": False if order_off else given("moe_transpose_block", cross),
        "moe_static_alt_tiles": given("moe_static_alt_tiles", alt_tiles),
        "moe_expert_prefetch": prefetch,
        "moe_weight_buffers": None
        if order_off and keys.get("moe_weight_buffers") == 3
        else (
            3 if cross and not order_off and keys.get("moe_weight_buffers") is None else keys.get("moe_weight_buffers")
        ),
        "moe_routed_reduce": routed_reduce,
        # The column out projection is opt-in on every form (an accuracy-class option, no served measurement yet).
        "moe_out_proj_form": given("moe_out_proj_form", "reduced"),
        "moe_glue": moe_glue,
        # The next layer's bank sums after the routed copies (schedule edges): ROUTED_ROW_SUM_SERVED_IN with
        # COMBO_ROUTED_REDUCES_BY_FORM.
        "moe_sums_after_routed": sums_after_routed,
        "attn_res_sums_after_moe_output": given("attn_res_sums_after_moe_output", row_sums_form),
    }


def live_pad_rows(live_rows: nl.NkiTensor, queries: int, name: str) -> nl.NkiTensor:
    """Return the HBM INT32 ``[Q B, 1]`` padding-row input (``decode_block_body`` ``moe_pad_rows``) of the runner's
    live-row mask ``live_rows`` (INT32 ``[B, 1]``; 0: an empty slot): row ``j * B + b`` is 1 when slot b is empty,
    else 0. Q DMAs place the mask on the rows of each token j; VectorE compares it with 0."""
    seqs = live_rows.shape[0]
    rows = queries * seqs
    kernel_assert(
        tuple(live_rows.shape) == (seqs, 1) and live_rows.dtype == nl.int32 and rows <= P_MAX,
        f"live_rows is INT32 [B, 1] with Q B <= {P_MAX} rows, got {live_rows.dtype} {tuple(live_rows.shape)}, Q {queries}",
    )
    vector = nisa.engine.vector
    live = nl.ndarray((rows, 1), dtype=nl.int32, buffer=nl.sbuf)
    for query in range(queries):
        # A few bytes each: the default static queue (no DGE).
        nisa.dma_copy(dst=live[query * seqs : (query + 1) * seqs, :], src=live_rows, name=f"{name}_live_q{query}")
    live_f = nl.ndarray((rows, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=live_f, src=live, engine=vector)
    pad_f = nl.ndarray((rows, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=pad_f, data=live_f, op0=nl.equal, operand0=0.0, engine=vector)
    pad = nl.ndarray((rows, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=pad, src=pad_f, engine=vector)
    pad_rows = nl.ndarray((rows, 1), dtype=nl.int32, buffer=nl.private_hbm, name=f"{name}_pad_rows")
    nisa.dma_copy(dst=pad_rows, src=pad, name=f"{name}_pad_rows_store")
    return pad_rows


def verify_2srv_sources(
    rows: int,
    queries: int,
    n_tiles: int,
    dtype,
    region: ExplicitSbuf,
    tail_ranks: int,
    world_size: int,
    routed_tp: int = None,
    moe_gather: str = "dp_tp",
    mla_mode: str = "k1",
    pingpong: bool = True,
    name: str = "",
    routed_reduce: str = "ar",
    stats_gather: bool = False,
    rows_latent: str = "bf16",
    drop_selection: bool = False,
    rows_dst_pingpong: bool = True,
    latent_gather_mx: bool = False,
    out_proj_form: str = "reduced",
    pair_shared: tuple = (),
    mx_scores_align: bool = False,
    dense_x_gather: str = "dp",
    mx_rows_stream: bool = False,
    mx_tile: int = None,
) -> dict:
    """Return the persistent collective tensors of a multi-server verify step of ``rows`` = Q * B rows of each replica.

    The attention and tail sites are those of ``kimi_k3_verify_model`` at DP = W / 32 (``decode_block.collective_sources``
    without the block MLA, MoE and KDA tiles): the k1 sites (static path) or the per-token parity sets, and the TP32
    argmax gather. The feed-forward sites are the HBM tensors of ``tkg_moe_2srv.moe_2srv_collective_hbm`` (under
    ``MOE_SOURCES_KEY``, with its second set in ``pingpong``) and of ``tkg_dense_mlp_2srv.dense_collective_hbm`` (under
    ``DENSE_SOURCES_KEY``) at ``rows`` rows, as ``kimi_k3_decode_model_2srv`` builds them at B rows. On the loop path
    the k1 call takes its own sites (``kimi_k3_verify_model_2srv``). ``name`` suffixes the tensor names.
    ``routed_reduce`` and ``stats_gather`` give the routed tensors of the MoE (``tkg_moe_2srv.ROUTED_REDUCES``),
    ``rows_latent`` the router-row width (``tkg_moe_2srv.ROWS_LATENTS``).
    ``drop_selection``: the 84-word router rows of the MoE gathers (``moe_2srv_collective_hbm``).
    ``latent_gather_mx``, ``mx_scores_align`` and ``mx_rows_stream``: the G1a router rows, their 65-word padded form
    and their 64-word token-major form (``moe_2srv_collective_hbm``); the entry gives the ``mx_scores_align`` and the
    ``mx_rows_stream`` of its ``moe_glue``.
    ``rows_dst_pingpong``: the second DP gather destination of the odd layers (``moe_2srv_collective_hbm``).
    ``out_proj_form``: the out projection of the MoE (``tkg_moe_2srv.OUT_PROJ_FORMS``; ``column`` adds the shared
    and output all-gather tensors of ``moe_2srv_collective_hbm``).
    ``pair_shared``: the checked ``pair_shared_cc_dsts`` of the entry; the MoE and dense builders put the destinations of
    ``moe_out`` and ``moe_routed`` in LNC1 shared HBM, as in ``kimi_k3_decode_model_2srv``, and the sources carry the
    sites (``common.PAIR_SHARED_SOURCES_KEY``). () keeps every tensor private.
    ``dense_x_gather``: the x gather form of the dense layer 0 (``tkg_dense_mlp_2srv.DENSE_X_GATHERS``; the entry gives
    the ``dense_x_gather`` of its ``moe_glue``): ``sharded`` adds the slice tensors of ``dense_collective_hbm``.
    ``mx_tile``: the MX tile width of the expert split (``moe_2srv_collective_hbm``); None (default) keeps the program.
    """
    kernel_assert(mla_mode in MLA_MODES, f"mla_mode must be one of {MLA_MODES}, got {mla_mode!r}")
    dp = world_layout(world_size)["dp"]
    batch = rows // queries
    k1 = mla_mode == "k1"
    set_rows = min(batch, QUERY_SET_TOKENS)
    # Above 16 rows each per-token call takes its own tensors; the static k1 call takes the k1 sites.
    deparity_rows = set_rows if (not k1 and rows > 16) else 0
    sources = collective_sources(
        rows,
        n_tiles,
        dtype,
        region,
        tail_ranks,
        mla=False,
        moe=False,
        kda=False,
        verify_mla_rows=set_rows if (not k1 and region is None) else 0,
        name=name,
        pingpong=pingpong,
        dp_size=dp,
        q_hbm=region is None and rows > 16,
        deparity_rows=deparity_rows,
        deparity_out=region is not None,
        deparity_q=region is not None,
        deparity_calls=queries * -(-batch // deparity_rows) if deparity_rows else 0,
        verify_k1_heads=_MLA_LOCAL_HEADS if (k1 and region is None) else 0,
        **({"pair_shared_cc_dsts": pair_shared} if pair_shared else {}),
    )
    ps = {"pair_shared": pair_shared} if pair_shared else {}
    moe_hbm, moe_second = moe_2srv_collective_hbm(
        rows,
        world_size,
        routed_tp,
        moe_gather,
        name,
        routed_reduce=routed_reduce,
        stats_gather=stats_gather,
        rows_latent=rows_latent,
        rows_dst_pingpong=rows_dst_pingpong,
        **({"drop_selection": True} if drop_selection else {}),
        **({"latent_gather_mx": True} if latent_gather_mx else {}),
        **({"mx_scores_align": True} if mx_scores_align else {}),
        **({"mx_rows_stream": True} if mx_rows_stream else {}),
        **({"out_proj_form": out_proj_form} if out_proj_form != "reduced" else {}),
        **ps,
        **({"mx_tile": mx_tile} if mx_tile is not None else {}),
    )
    kernel_assert(MOE_SOURCES_KEY not in sources and DENSE_SOURCES_KEY not in sources, "feed-forward source keys clash")
    sources[MOE_SOURCES_KEY] = moe_hbm
    sources[DENSE_SOURCES_KEY] = dense_collective_hbm(
        rows, world_size, name, **ps, **({"x_gather": dense_x_gather} if dense_x_gather != "dp" else {})
    )
    if pingpong:
        sources["pingpong"] = {**sources.get("pingpong", {}), MOE_SOURCES_KEY: moe_second}
    return sources


def first_rows_destination(sources: dict, first_layer: int) -> nl.NkiTensor | None:
    """Give the first DP row gather of a call that starts after layer 0 its own destination and return it.

    That gather is the first collective of the DP group in the call. With the shared ``rows_dst`` (or the two of
    ``rows_dst_pingpong``, one for each layer parity) later MoE layers of the call gather into the same bytes; its own
    tensor (``tkg_moe_2srv`` ``first_rows_dst``, in both MoE sets) is written once. A call from layer 0 starts the DP
    group with the dense gather: None, no change.
    """
    moe = sources[MOE_SOURCES_KEY]
    if first_layer == 0 or "rows_dst" not in moe:
        return None
    rows_dst = moe["rows_dst"]
    first = nl.ndarray(rows_dst.shape, dtype=rows_dst.dtype, buffer=nl.private_hbm, name="moe2s_first_rows_dst")
    once = [first]
    moe["first_rows_dst"] = once
    second = sources.get("pingpong", {}).get(MOE_SOURCES_KEY)
    if second is not None:
        second["first_rows_dst"] = once
    return first


def verify_first_cc_destinations(sources: dict, kda_ar_pairs: tuple, first_layer: int, first_rows=None) -> list:
    """Return the HBM destinations of the first collective of each group in a verify call: the KDA all-reduce of the
    first pair (TP32), and the DP gather and the world all-reduce of the dense layer 0, or of the first MoE layer in a
    call that starts after layer 0 (its own DP gather destination ``first_rows``). The other destinations of those
    groups are written later in the call. With ``rs_ag`` the world collective of that MoE layer is its reduce-scatter,
    whose destination alternates by layer parity. The DP gather destination of a call without ``first_rows`` is the
    ``rows_dst`` of the set of the first layer; the other ``rows_dst`` of ``rows_dst_pingpong`` is first written later in
    the call, so it is not a first destination.
    """
    if first_layer == 0:
        dense = sources[DENSE_SOURCES_KEY]
        # The sharded x gather (tkg_dense_mlp_2srv.DENSE_X_GATHERS): the DP gather writes x_mid, and the TP32 gather
        # of x_dst follows the first TP32 collective of the call (the KDA all-reduce).
        firsts = [dense.get("x_mid", dense["x_dst"]), dense["out_dst"]]
    else:
        moe = layer_sources(sources, first_layer)[MOE_SOURCES_KEY]
        gather = [first_rows] if first_rows is not None else [moe[k] for k in ("rows_dst", "world_dst") if k in moe]
        firsts = [*gather, *(moe[k] for k in ("routed_dst", "routed_rs_dst") if k in moe)]
    return [kda_ar_pairs[0][1], *firsts]


def verify_2srv_bodies(
    queries: int,
    kda_state_page,
    kda_record_page,
    kda_replay_len,
    kda_state_position,
    rows: int,
    region: ExplicitSbuf,
    world_size: int,
    tp_group: ReplicaGroup,
    mla_mode: str = "k1",
    mla_sources=None,
    mla_state=None,
    routed_tp: int = None,
    moe_gather: str = "dp_tp",
    wide_half_weight_buffers: int = WIDE_HALF_WEIGHT_BUFFERS,
    expert_call_tokens: int | str = EXPERT_CALL_AUTO,
    expert_queues: int = 1,
    routed_reduce: str = "ar",
    stats_gather: bool = False,
    rows_latent: str = "bf16",
    rows_load_split: int = 1,
    routed_own_rows: bool = False,
    moe_glue: dict = None,
    drop_selection: bool = False,
    latent_gather_mx: bool = False,
    out_proj_form: str = "reduced",
    mx_tile: int = None,
    routed_dynamic: bool = False,
    routed_plan: tuple = None,
    routed_dynamic_csr: str = None,
    **options,
):
    """Return the ``(kda, mla, moe, dense)`` verify bodies of one AttnRes block with the multi-server feed-forward.

    ``verify_model.verify_bodies`` with the MoE body of ``tkg_moe_2srv.moe_2srv_body``, the dense body of
    ``tkg_dense_mlp_2srv.dense_ffn_2srv`` and one feed-forward call on all ``rows`` rows. ``mla_sources`` and
    ``mla_state`` are the MLA sites of ``mla_mode`` (``kimi_k3_verify_model_2srv``); ``routed_reduce`` and
    ``stats_gather`` the routed reduction of the MoE body (the sources must have the same); ``options`` are the other
    keyword arguments of ``verify_bodies`` (the KDA replay, the k1 forms, the MLA norm epsilon). ``moe_glue``,
    ``drop_selection`` and ``out_proj_form`` are the ``glue``, ``drop_selection`` and ``out_proj_form`` of the MoE
    body (the sources must have the same ``out_proj_form``). ``mx_tile``: the MX tile width of the expert split of the
    MoE body (``moe_2srv_body``); None (default) keeps the program. ``routed_dynamic``: the dynamic routed form of the
    MoE body (``moe_2srv_body``); False (default) keeps the program. ``routed_plan``: None (default), or ``(plan,
    default_tag)`` of ``moe_routed_plan.call_plan``: each MoE layer runs the body of its tag (``plan_bodies``),
    the default tag with ``routed_tp`` and ``mx_tile``. ``routed_dynamic_csr``: the CSR gathers of the dynamic packed
    layers (``moe_2srv_body``; ``moe_routed_plan.call_csr_gather``); None (default) keeps "staged".
    """

    def make_moe_body(body_tp: int, body_tile: int):
        return moe_2srv_body(
            tp_group,
            world_size,
            body_tp,
            moe_gather,
            wide_half_weight_buffers,
            expert_call_tokens=expert_call_tokens,
            **({"expert_queues": expert_queues} if expert_queues != 1 else {}),
            routed_reduce=routed_reduce,
            stats_gather=stats_gather,
            rows_latent=rows_latent,
            rows_load_split=rows_load_split,
            routed_own_rows=routed_own_rows,
            **({"drop_selection": True} if drop_selection else {}),
            **({"latent_gather_mx": True} if latent_gather_mx else {}),
            **({"glue": dict(moe_glue)} if moe_glue else {}),
            **({"out_proj_form": out_proj_form} if out_proj_form != "reduced" else {}),
            **({"mx_tile": body_tile} if body_tile is not None else {}),
            # The dynamic form runs the splits of sub-512 MX tiles (body_tile); a served split of a plan keeps its loop.
            **({"routed_dynamic": True} if routed_dynamic and body_tile is not None else {}),
            # Its CSR form goes with it (latent_moe_decode_2srv refuses routed_dynamic_csr without routed_dynamic).
            **(
                {"routed_dynamic_csr": routed_dynamic_csr}
                if routed_dynamic_csr is not None and body_tile is not None
                else {}
            ),
        )

    moe_body = make_moe_body(routed_tp, mx_tile)
    if routed_plan is not None:
        moe_body = plan_bodies(routed_plan[0], routed_plan[1], moe_body, make_moe_body)
    return verify_bodies(
        queries,
        kda_state_page,
        kda_record_page,
        kda_replay_len,
        kda_state_position,
        rows,
        sbuf=AUTO_SBUF if region is None else SbufTiles(region),
        mla_mode=mla_mode,
        mla_sources=mla_sources,
        mla_state=mla_state,
        moe_body=moe_body,
        dense_body=dense_ffn_2srv(world_size, (moe_glue or {}).get("dense_x_gather", "dp")),
        **options,
    )


def split_pair_group(world_size: int) -> ReplicaGroup:
    """Return the pair groups of the long-sequence split (``mla_split``): TP rank t of the DP replicas r and r xor 1 (the
    same server), one group of 2 ranks for each (pair, t)."""
    replicas = world_size // TAIL_TP_RANKS
    return ReplicaGroup(
        [
            [r * TAIL_TP_RANKS + t, (r + 1) * TAIL_TP_RANKS + t]
            for r in range(0, replicas, 2)
            for t in range(TAIL_TP_RANKS)
        ]
    )


def split_handshake(split_info, rank_info, group, sb):
    """The step's split handshake (``mla_split``): one pair all-gather of the [1, 4] INT32 infos (role, X, slot, tag);
    return the effective info (HBM ``[1, 4]``: the effective role, X, slot, tag) and the aux output (``[1, 4]``: a
    mismatch 0 / 1, the effective role, X, the partner's role). The effective role is 1 when this rank says owner and
    the partner says helper with the same X and tag, 2 the other way round, else 0. Small integers (a tag below 2**24),
    exact in FP32. The partner's row of the gather is the one of the other DP parity."""
    fp32 = nl.float32
    int32 = nl.int32
    vector = nisa.engine.vector
    src = nl.ndarray((1, 4), dtype=int32, buffer=nl.hbm, name="mla_split_hs_src")
    dst = nl.ndarray((2, 4), dtype=int32, buffer=nl.hbm, name="mla_split_hs_dst")
    nisa.dma_copy(dst=src, src=split_info, name="mla_split_hs_store")
    ncc.all_gather(srcs=[src], dsts=[dst], replica_group=group, collective_dim=0, name="mla_split_handshake_ag")
    both_i = sb.tile((1, 8), int32)
    nisa.dma_copy(dst=both_i, src=dst.reshape((1, 8)), name="mla_split_hs_load")
    rank_i = sb.tile((1, 4), int32)
    nisa.dma_copy(dst=rank_i, src=rank_info[0:1, 0:4], name="mla_split_hs_rank")
    both = sb.tile((1, 8), fp32)
    nisa.tensor_copy(dst=both, src=both_i, engine=vector)
    parity_i = sb.tile((1, 1), int32)
    nisa.tensor_scalar(dst=parity_i, data=rank_i[0:1, 1:2], op0=nl.bitwise_and, operand0=1)
    parity = sb.tile((1, 1), fp32)
    nisa.tensor_copy(dst=parity, src=parity_i, engine=vector)
    # own = row (parity), partner = row (1 - parity): own = r0 + parity (r1 - r0), partner = r1 - parity (r1 - r0).
    delta = sb.tile((1, 4), fp32)
    nisa.tensor_tensor(dst=delta, data1=both[0:1, 4:8], data2=both[0:1, 0:4], op=nl.subtract, engine=vector)
    nisa.tensor_scalar(dst=delta, data=delta, op0=nl.multiply, operand0=parity, engine=vector)
    own = sb.tile((1, 4), fp32)
    nisa.tensor_tensor(dst=own, data1=both[0:1, 0:4], data2=delta, op=nl.add, engine=vector)
    partner = sb.tile((1, 4), fp32)
    nisa.tensor_tensor(dst=partner, data1=both[0:1, 4:8], data2=delta, op=nl.subtract, engine=vector)
    same = sb.tile((1, 4), fp32)
    nisa.tensor_tensor(dst=same, data1=own, data2=partner, op=nl.equal, engine=vector)
    agree = sb.tile((1, 1), fp32)
    nisa.tensor_tensor(dst=agree, data1=same[0:1, 1:2], data2=same[0:1, 3:4], op=nl.multiply, engine=vector)

    def is_value(src_col, value):
        out = sb.tile((1, 1), fp32)
        nisa.tensor_scalar(dst=out, data=src_col, op0=nl.equal, operand0=float(value), engine=vector)
        return out

    owner = sb.tile((1, 1), fp32)
    nisa.tensor_tensor(
        dst=owner, data1=is_value(own[0:1, 0:1], 1), data2=is_value(partner[0:1, 0:1], 2), op=nl.multiply, engine=vector
    )
    nisa.tensor_tensor(dst=owner, data1=owner, data2=agree, op=nl.multiply, engine=vector)
    helper = sb.tile((1, 1), fp32)
    nisa.tensor_tensor(
        dst=helper,
        data1=is_value(own[0:1, 0:1], 2),
        data2=is_value(partner[0:1, 0:1], 1),
        op=nl.multiply,
        engine=vector,
    )
    nisa.tensor_tensor(dst=helper, data1=helper, data2=agree, op=nl.multiply, engine=vector)
    eff = sb.tile((1, 4), fp32)
    nisa.tensor_copy(dst=eff, src=own, engine=vector)
    # role = owner + 2 helper
    nisa.scalar_tensor_tensor(dst=eff[0:1, 0:1], data=helper, op0=nl.multiply, operand0=2.0, op1=nl.add, operand1=owner)
    # mismatch = (own role > 0 or partner role > 0) and role == 0
    asked = sb.tile((1, 1), fp32)
    nisa.tensor_tensor(dst=asked, data1=own[0:1, 0:1], data2=partner[0:1, 0:1], op=nl.add, engine=vector)
    nisa.tensor_scalar(dst=asked, data=asked, op0=nl.greater, operand0=0.0, engine=vector)
    aux = sb.tile((1, 4), fp32)
    nisa.tensor_scalar(
        dst=aux[0:1, 0:1], data=eff[0:1, 0:1], op0=nl.equal, operand0=0.0, op1=nl.multiply, operand1=asked
    )
    nisa.tensor_copy(dst=aux[0:1, 1:2], src=eff[0:1, 0:1], engine=vector)
    nisa.tensor_copy(dst=aux[0:1, 2:3], src=own[0:1, 1:2], engine=vector)
    nisa.tensor_copy(dst=aux[0:1, 3:4], src=partner[0:1, 0:1], engine=vector)
    eff_i = sb.tile((1, 4), int32)
    nisa.tensor_copy(dst=eff_i, src=eff, engine=vector)
    aux_i = sb.tile((1, 4), int32)
    nisa.tensor_copy(dst=aux_i, src=aux, engine=vector)
    eff_hbm = nl.ndarray((1, 4), dtype=int32, buffer=nl.hbm, name="mla_split_info_eff")
    nisa.dma_copy(dst=eff_hbm, src=eff_i, name="mla_split_eff_store")
    aux_out = nl.ndarray((1, 4), dtype=int32, buffer=nl.shared_hbm, name="mla_split_aux")
    nisa.dma_copy(dst=aux_out, src=aux_i, name="mla_split_aux_store")
    return eff_hbm, aux_out


def split_extended_inputs(block_table, slot, split_table, position_rows, live_rows, split_info, sb):
    """Return the MLA inputs of the split's extended batch B + 2 (``mla_split``): the table ``[n_mla, B + 2, NB]``
    (row B = ``split_table``, row B + 1 = 0), the slots ``[n_mla, Q, B + 2, 1]`` (0 for the two added slots: no cache
    write), the row-0 positions ``[B + 2, 1]`` (X of the info for the foreign slot, 0 for the pad slot) and the live rows
    ``[B + 2, 1]`` (1, 0) or None. One set a call: HBM copies of the inputs."""
    int32 = nl.int32
    n_mla, batch, blocks = block_table.shape
    queries = slot.shape[1]
    ext = batch + 2
    zeros = sb.tile((1, blocks), int32)
    nisa.memset(dst=zeros, value=0, engine=nisa.engine.vector)
    zero_rows = sb.tile((2, 1), int32)
    nisa.memset(dst=zero_rows, value=0, engine=nisa.engine.vector)
    table = nl.ndarray((n_mla, ext, blocks), dtype=int32, buffer=nl.hbm, name="mla_split_table_ext")
    slots = nl.ndarray((n_mla, queries, ext, 1), dtype=int32, buffer=nl.hbm, name="mla_split_slot_ext")
    for g in range(n_mla):
        nisa.dma_copy(dst=table[g, 0:batch, 0:blocks], src=block_table[g, 0:batch, 0:blocks], name=f"mla_split_t{g}")
        nisa.dma_copy(
            dst=table[g, batch : batch + 1, 0:blocks], src=split_table[g, 0:1, 0:blocks], name=f"mla_split_f{g}"
        )
        nisa.dma_copy(dst=table[g, batch + 1 : ext, 0:blocks], src=zeros[0:1, 0:blocks], name=f"mla_split_p{g}")
        for j in range(queries):
            nisa.dma_copy(dst=slots[g, j, 0:batch, 0:1], src=slot[g, j, 0:batch, 0:1], name=f"mla_split_s{g}_{j}")
            nisa.dma_copy(dst=slots[g, j, batch:ext, 0:1], src=zero_rows, name=f"mla_split_z{g}_{j}")
    position = nl.ndarray((ext, 1), dtype=int32, buffer=nl.hbm, name="mla_split_position_ext")
    nisa.dma_copy(dst=position[0:batch, 0:1], src=position_rows[0, 0:batch, 0:1], name="mla_split_pos")
    nisa.dma_copy(dst=position[batch : batch + 1, 0:1], src=split_info[0:1, 1:2], name="mla_split_pos_x")
    nisa.dma_copy(dst=position[batch + 1 : ext, 0:1], src=zeros[0:1, 0:1], name="mla_split_pos_pad")
    live = None
    if live_rows is not None:
        ones = sb.tile((1, 1), int32)
        nisa.memset(dst=ones, value=1, engine=nisa.engine.vector)
        live = nl.ndarray((ext, 1), dtype=int32, buffer=nl.hbm, name="mla_split_live_ext")
        nisa.dma_copy(dst=live[0:batch, 0:1], src=live_rows[0:batch, 0:1], name="mla_split_live")
        nisa.dma_copy(dst=live[batch : batch + 1, 0:1], src=ones, name="mla_split_live_f")
        nisa.dma_copy(dst=live[batch + 1 : ext, 0:1], src=zeros[0:1, 0:1], name="mla_split_live_p")
    return table, slots, position, live


def split_v5_pack_options(
    world_size: int,
    queries: int,
    rows: int,
    state_bf16_o: bool,
    pack_options: dict = None,
    mx_qk: str = None,
    entry_act: bool = False,
    step_items: bool = False,
) -> dict:
    """The packed-loop options of the split v5 DEV program's extended batch (B' = B + F sequences; ``rows`` = Q B, the
    rows of a replica): the served form's own options (``served_pack_options``: k7b2 at C=8, ``rows`` 16; k7 at C=16,
    ``rows`` 32) with the caller's ``pack_options``, ``mx_qk``, ``entry_act`` and ``step_items`` of the call, so that
    the DEV program differs from the served one only by the split. At C=16 the 64-row call fits the SBUF-only stage rows
    and the 8-deep slot-row ring since its foreign passes take no tail and its output holds the request rows only
    (``mla_block_attention`` ``split`` ext). The owner ranges of the step check need the per-pass plans, so the split
    program takes no ``hoist_plans`` or ``step_items`` (the plans of the served form are built before the step check,
    for the B sequences only); a plan built in each layer gives the same state bits. It also takes no
    ``stage_rows_sbuf`` (the slot rows read the stage rows' HBM store; the same bytes): with the served C=16 verify
    levers (the scalar products, the KDA w_in ring of 8) the 64-row call does not fit the MLA SBUF stack with it. It
    takes no FP8 P@V pass either (``fp8_pv`` False, also in a ``FP8_PV_FORM_SET`` form; ``check_fp8_pv_split``).

    Example:
        >>> split_v5_pack_options(256, 8, 16, True)
        {'clamp_tail_only': True, 'word_keys': True, 'item_rope_ahead': True, 'p2': True, 'hoist_plans': False, 'step_items': False, 'stage_rows_sbuf': False}
    """
    form = "k7b2" if rows == SERVED_FORMS["k7b2"][2] else "k7"
    options = served_pack_options(
        world_size,
        queries,
        SERVED_FORMS[form][2],
        True,
        pack_options,
        state_bf16_o,
        mx_qk,
        entry_act=entry_act,
        step_items=step_items,
        # The split program does not take the FP8 P@V pass (check_fp8_pv_split), also in an FP8_PV_FORM_SET form.
        fp8_pv=False,
    )
    return {**(options or {}), "hoist_plans": False, "step_items": False, "stage_rows_sbuf": False}


def split_v5_extended_inputs(block_table, slot, split_table, position_rows, live_rows, sb):
    """Return the MLA inputs of the split program's extended batch B + F (``mla_split`` v5): the table
    ``[n_mla, B + F, NB]`` (row B + f = ``split_table[:, f]``, the helper's copy of foreign slot f), the slots
    ``[n_mla, Q, B + F, 1]`` (0 for the foreign slots: no cache write), the row-0 positions ``[B + F, 1]`` and the live
    rows ``[B + F, 1]`` (or None). The foreign rows of the positions and live rows are 0 here; the step check writes
    them (``foreign_pos`` g1 x 16384, ``foreign_live``) right after layer 0's gather (``_split_v5_rider``)."""
    int32 = nl.int32
    n_mla, batch, blocks = block_table.shape
    F = split_table.shape[1]
    queries = slot.shape[1]
    ext = batch + F
    zero_rows = sb.tile((F, 1), int32)
    nisa.memset(dst=zero_rows, value=0, engine=nisa.engine.vector)
    table = nl.ndarray((n_mla, ext, blocks), dtype=int32, buffer=nl.hbm, name="mla_split_table_ext")
    slots = nl.ndarray((n_mla, queries, ext, 1), dtype=int32, buffer=nl.hbm, name="mla_split_slot_ext")
    for g in range(n_mla):
        nisa.dma_copy(dst=table[g, 0:batch, 0:blocks], src=block_table[g, 0:batch, 0:blocks], name=f"mla_split_t{g}")
        nisa.dma_copy(dst=table[g, batch:ext, 0:blocks], src=split_table[g, 0:F, 0:blocks], name=f"mla_split_f{g}")
        for j in range(queries):
            nisa.dma_copy(dst=slots[g, j, 0:batch, 0:1], src=slot[g, j, 0:batch, 0:1], name=f"mla_split_s{g}_{j}")
            nisa.dma_copy(dst=slots[g, j, batch:ext, 0:1], src=zero_rows, name=f"mla_split_z{g}_{j}")
    position = nl.ndarray((ext, 1), dtype=int32, buffer=nl.hbm, name="mla_split_position_ext")
    nisa.dma_copy(dst=position[0:batch, 0:1], src=position_rows[0, 0:batch, 0:1], name="mla_split_pos")
    nisa.dma_copy(dst=position[batch:ext, 0:1], src=zero_rows, name="mla_split_pos_f")
    live = None
    if live_rows is not None:
        live = nl.ndarray((ext, 1), dtype=int32, buffer=nl.hbm, name="mla_split_live_ext")
        nisa.dma_copy(dst=live[0:batch, 0:1], src=live_rows[0:batch, 0:1], name="mla_split_live")
        nisa.dma_copy(dst=live[batch:ext, 0:1], src=zero_rows, name="mla_split_live_f")
    return table, slots, position, live


SPLIT_MAX_STATS_WORDS = 64
"""Most INT32 words of the v5 split's opaque stats block (``mla_split`` ``stats_words``): 256 B a replica."""


def _check_split_v5(mla_split, stats, info, own, foreign, table, batch, n_mla, blocks):
    """Check the v5 split key and its inputs (``kimi_k3_verify_model_2srv`` ``mla_split``)."""
    F, S, program = mla_split.get("foreign_slots"), mla_split.get("stats_words"), mla_split.get("program")
    priority = mla_split.get("ag_priority")
    kernel_assert(
        {"foreign_slots", "stats_words", "program"}
        <= set(mla_split)
        <= {"foreign_slots", "stats_words", "program", "ag_priority"}
        and program in ("idle", "split", "one")
        and F in (1, 2, 4)
        and isinstance(S, int)
        and 1 <= S <= SPLIT_MAX_STATS_WORDS
        and (priority is None or (program != "idle" and priority in (0, 1, 2, 3))),
        "mla_split (v5) is {'foreign_slots': 1, 2 or 4, 'stats_words': 1 to "
        f"{SPLIT_MAX_STATS_WORDS}, 'program': 'idle', 'split' or 'one'}} (+ 'ag_priority' 0-3, a TEST arm of the split "
        f"programs), got {mla_split}",
    )
    kernel_assert(
        stats is not None and tuple(stats.shape) == (1, S) and stats.dtype == nl.int32,
        f"mla_split (v5) needs mla_split_stats INT32 [1, {S}]",
    )
    kernel_assert(info is None, "mla_split (v5) takes no mla_split_info (the v1 pair info)")
    if program == "idle":
        kernel_assert(
            own is None and foreign is None and table is None,
            "the idle program (mla_split v5) takes mla_split_stats only",
        )
    else:
        for tensor, shape, label in (
            (own, (batch, 2), "mla_split_own"),
            (foreign, (F, 5), "mla_split_foreign"),
            (table, (n_mla, F, blocks), "mla_split_table"),
        ):
            kernel_assert(
                tensor is not None and tuple(tensor.shape) == shape and tensor.dtype == nl.int32,
                f"the split program (mla_split v5) needs {label} INT32 {list(shape)}",
            )


SPLIT_SLOT_WORDS = 8
"""Words of a slot record in the split program's rider: (lo, tag, DP replica, slot, slot parity, 0, 0, 0)."""
SPLIT_PIECE_WORDS = 8
"""Words of a piece record in the split program's rider: (src_replica, src_slot, g0, g1, tag, DP replica, foreign slot,
foreign slot parity)."""
_SPLIT_BIG = float(1 << 20)
"""Above every key group of a split (a 16384-key group; 2**20 groups = 2**34 keys)."""


def _split_v5_records(own, foreign, rank_info, sb, tag: str):
    """Return the split program's rider records ``[1, B x 4]`` and ``[1, F x 8]`` INT32 HBM of this replica: the step's
    ``own`` ``[B, 2]`` (lo, tag) and ``foreign`` ``[F, 5]`` (src_replica, src_slot, g0, g1, tag) with this replica's DP
    rank (``rank_info[0, 1]``) and the slot index appended, so the step check needs no index constants."""
    int32 = nl.int32
    B, F = own.shape[0], foreign.shape[0]
    rec_own = sb.tile((1, B * SPLIT_SLOT_WORDS), int32)
    rec_pc = sb.tile((1, F * SPLIT_PIECE_WORDS), int32)
    own_v = rec_own.reshape((1, B, SPLIT_SLOT_WORDS))
    pc_v = rec_pc.reshape((1, F, SPLIT_PIECE_WORDS))
    nisa.memset(dst=rec_pc, value=0, engine=nisa.engine.vector)
    nisa.memset(dst=rec_own, value=0, engine=nisa.engine.vector)
    for b in range(B):
        nisa.memset(dst=own_v[0:1, b : b + 1, 3:4], value=b, engine=nisa.engine.vector)
        nisa.memset(dst=own_v[0:1, b : b + 1, 4:5], value=b % 2, engine=nisa.engine.vector)
    for f in range(F):
        nisa.memset(dst=pc_v[0:1, f : f + 1, 6:7], value=f, engine=nisa.engine.vector)
        nisa.memset(dst=pc_v[0:1, f : f + 1, 7:8], value=f % 2, engine=nisa.engine.vector)
    nisa.dma_copy(dst=own_v[0:1, 0:B, 0:2], src=own.reshape((1, B, 2)), name=f"{tag}_own_load")
    nisa.dma_copy(dst=pc_v[0:1, 0:F, 0:5], src=foreign.reshape((1, F, 5)), name=f"{tag}_foreign_load")
    nisa.dma_copy(
        dst=own_v[0:1, 0:B, 2:3], src=rank_info.ap(pattern=[[4, 1], [0, B], [1, 1]], offset=1), name=f"{tag}_own_rank"
    )
    nisa.dma_copy(
        dst=pc_v[0:1, 0:F, 5:6], src=rank_info.ap(pattern=[[4, 1], [0, F], [1, 1]], offset=1), name=f"{tag}_pc_rank"
    )
    own_src = nl.ndarray((1, B * SPLIT_SLOT_WORDS), dtype=int32, buffer=nl.hbm, name=f"{tag}_own_src")
    pc_src = nl.ndarray((1, F * SPLIT_PIECE_WORDS), dtype=int32, buffer=nl.hbm, name=f"{tag}_pc_src")
    nisa.dma_copy(dst=own_src, src=rec_own, name=f"{tag}_own_store")
    nisa.dma_copy(dst=pc_src, src=rec_pc, name=f"{tag}_pc_store")
    return own_src, pc_src


def split_v5_step_check(
    own_g, pc_g, rank_info, dp: int, B: int, F: int, Q: int = 8, tag: str = "mla_split_chk", parity: bool = False
):
    """The split program's step check on the DP-gathered records of the step: every
    rank does the same arithmetic on the same bits, so every rank gets the same verdicts.

    ``own_g`` ``[dp, B x 4]`` INT32 HBM: per slot n = r B + b (lo, tag, r, b); ``pc_g`` ``[dp, F x 8]`` INT32 HBM: per
    piece p = h F + f (src_replica, src_slot, g0, g1, tag, h, f, 0); ``rank_info`` ``[1, 4]`` INT32 (DP rank at 1).

    Rules: a piece is well formed when tag != 0, 0 <= g0 < g1 and src_replica != h. It matches slot (r, b) when r =
    src_replica, the tags are equal, lo >= g1 and (src_slot = -1 or src_slot = b); it is matched when it matches exactly
    one slot. A split (lo > 0, tag != 0) is confirmed when its matched pieces are 1 or 2, their lengths sum to lo and
    two pieces are disjoint (with every piece in [0, lo): an exact cover of [0, lo)). lo_eff = lo if confirmed, else
    0; a piece is active when it is matched and its split confirmed.

    Small integers (tags below 2**24, groups below 2**20) are exact in FP32. The slot-major tiles hold slot n on
    partition n and the pieces on the free axis; the partition sums (one 0 / 1 operand each) run on TensorE.

    Returns HBM INT32 tensors: ``lo_mine`` ``[1, B]`` (this replica's lo_eff), ``pieces_mine`` ``[1, F, 5]`` (this
    replica's foreign slots: active, g0, g1, the matched source slot b, and the source's first row src_replica x Q B +
    b in the DP gather of the token-major request rows; zeros when inactive), ``any_mine`` ``[1, 1]`` (1 when this
    replica has an active piece), ``aux`` ``[1, 4]`` (unconfirmed splits, confirmed splits, active pieces, 0), and for
    the extended batch of the split program (B request slots, then F foreign slots): ``range_ext`` ``[B + F, 2]`` (the
    group range of each slot: (lo_eff, 2**20) for a request slot, (g0, g1) for a foreign slot, (0, 0) inactive),
    ``foreign_pos`` ``[1, F]`` (g1 x 16384: the row-0 position of a foreign slot, 0 inactive), ``foreign_live`` ``[1, F]``
    (active) and ``merge_mine`` ``[B, 3]`` (per request slot of this replica: the count of its active pieces and the
    lowest / highest export row h x 3 FP + f // 2 of them, FP = F / 2; 0 when none). ``parity`` (the extended batch):
    a piece also needs f % 2 = the parity of its source slot."""
    fp32, int32 = nl.float32, nl.int32
    v = nisa.engine.vector
    N, P = dp * B, dp * F

    def tile(shape, dtype=fp32):
        return nl.ndarray(shape, dtype=dtype, buffer=nl.sbuf)

    def psum(shape):
        return nl.ndarray(shape, dtype=fp32, buffer=nl.psum)

    slot_i = tile((N, SPLIT_SLOT_WORDS), int32)
    nisa.dma_copy(dst=slot_i, src=own_g.reshape((N, SPLIT_SLOT_WORDS)), name=f"{tag}_slots")
    pcb_i = tile((N, P * SPLIT_PIECE_WORDS), int32)
    nisa.dma_copy(dst=pcb_i, src=pc_g.ap(pattern=[[0, N], [1, P * SPLIT_PIECE_WORDS]], offset=0), name=f"{tag}_pcs")
    me_i = tile((N, 1), int32)
    nisa.dma_copy(dst=me_i, src=rank_info.ap(pattern=[[0, N], [1, 1]], offset=1), name=f"{tag}_me")
    slot = tile((N, SPLIT_SLOT_WORDS))
    nisa.tensor_copy(dst=slot, src=slot_i, engine=v)
    pcb = tile((N, P, SPLIT_PIECE_WORDS))
    nisa.tensor_copy(dst=pcb.reshape((N, P * SPLIT_PIECE_WORDS)), src=pcb_i, engine=v)
    me = tile((N, 1))
    nisa.tensor_copy(dst=me, src=me_i, engine=v)
    s_lo, s_tag, s_r, s_b, s_par = (slot[0:N, k : k + 1] for k in range(5))
    p_sr, p_sb, p_g0, p_g1, p_tag, p_h, p_f, p_par = (pcb[0:N, 0:P, k] for k in range(8))

    def tt(a, b, op):
        out = tile((N, P))
        nisa.tensor_tensor(dst=out, data1=a, data2=b, op=op, engine=v)
        return out

    def ts(a, op, operand, shape=(N, P)):
        out = tile(shape)
        nisa.tensor_scalar(dst=out, data=a, op0=op, operand0=operand, engine=v)
        return out

    # m[n, p]: piece p matches slot n (well formed included).
    m = ts(p_tag, nl.not_equal, 0.0)
    m = tt(m, tt(p_g1, p_g0, nl.greater), nl.multiply)
    m = tt(m, ts(p_g0, nl.greater_equal, 0.0), nl.multiply)
    m = tt(m, tt(p_sr, p_h, nl.not_equal), nl.multiply)
    m = tt(m, ts(p_sr, nl.equal, s_r), nl.multiply)
    m = tt(m, ts(p_tag, nl.equal, s_tag), nl.multiply)
    m = tt(m, ts(p_g1, nl.less_equal, s_lo), nl.multiply)
    m = tt(m, tt(ts(p_sb, nl.equal, s_b), ts(p_sb, nl.equal, -1.0), nl.add), nl.multiply)
    if parity:
        # The extended batch (2 sequences a pass): a piece in foreign slot f serves slots of parity f % 2, so its rows sit
        # on the pass rows of its owner slot (the merge adds whole pass blocks).
        m = tt(m, ts(p_par, nl.equal, s_par), nl.multiply)
    ones_col = tile((N, 1))
    nisa.memset(dst=ones_col, value=1.0, engine=v)
    ones_row = tile((1, N))
    nisa.memset(dst=ones_row, value=1.0, engine=v)
    # Matched = exactly one slot: the column sums over the slots (partitions), broadcast back to every slot.
    col_ps = psum((1, P))
    nisa.nc_matmul(dst=col_ps, stationary=ones_col, moving=m, is_stationary_onezero=True, is_moving_onezero=True)
    uniq = ts(col_ps, nl.equal, 1.0, (1, P))
    uniq_ps = psum((N, P))
    nisa.nc_matmul(dst=uniq_ps, stationary=ones_row, moving=uniq, is_stationary_onezero=True, is_moving_onezero=True)
    m2 = tt(m, uniq_ps, nl.multiply)

    def reduce(a, op):
        out = tile((N, 1))
        nisa.tensor_reduce(dst=out, op=op, data=a, axis=1)
        return out

    cnt = reduce(m2, nl.add)
    sumlen = reduce(tt(tt(p_g1, p_g0, nl.subtract), m2, nl.multiply), nl.add)
    maxg0 = reduce(tt(ts(p_g0, nl.add, 1.0), m2, nl.multiply), nl.maximum)  # max g0 + 1 of the matched pieces
    big_g1 = reduce(tt(ts(p_g1, nl.subtract, _SPLIT_BIG, (N, P)), m2, nl.multiply), nl.minimum)  # min g1 - BIG

    def col_tt(a, b, op):
        out = tile((N, 1))
        nisa.tensor_tensor(dst=out, data1=a, data2=b, op=op, engine=v)
        return out

    def col_ts(a, op, operand):
        return ts(a, op, operand, (N, 1))

    asked = col_tt(col_ts(s_lo, nl.greater, 0.0), col_ts(s_tag, nl.not_equal, 0.0), nl.multiply)
    conf = col_tt(asked, col_ts(cnt, nl.greater_equal, 1.0), nl.multiply)
    conf = col_tt(conf, col_ts(cnt, nl.less_equal, 2.0), nl.multiply)
    conf = col_tt(conf, col_tt(sumlen, s_lo, nl.equal), nl.multiply)
    # Two pieces are disjoint when min g1 <= max g0, i.e. (min g1 - BIG) + BIG <= (max g0 + 1) - 1.
    min_g1 = col_ts(big_g1, nl.add, _SPLIT_BIG + 1.0)
    disjoint = col_tt(min_g1, maxg0, nl.less_equal)
    conf = col_tt(conf, col_tt(disjoint, col_ts(cnt, nl.equal, 1.0), nl.add), nl.multiply)
    # The pieces: active (matched, its split confirmed) and the matched source slot.
    act_ps = psum((1, P))
    nisa.nc_matmul(dst=act_ps, stationary=conf, moving=m2, is_stationary_onezero=True, is_moving_onezero=True)
    bsel_ps = psum((1, P))
    nisa.nc_matmul(dst=bsel_ps, stationary=s_b, moving=m2, is_moving_onezero=True)
    # The source's first row r Q B + b (row j of the source sequence is that + j B).
    krow = tile((N, 1))
    nisa.tensor_scalar(dst=krow, data=s_r, op0=nl.multiply, operand0=float(Q * B), op1=nl.add, operand1=s_b, engine=v)
    ksel_ps = psum((1, P))
    nisa.nc_matmul(dst=ksel_ps, stationary=krow, moving=m2, is_moving_onezero=True)
    ksel = tile((1, P))
    nisa.tensor_copy(dst=ksel, src=ksel_ps, engine=v)
    active, bsel = tile((1, P)), tile((1, P))
    nisa.tensor_copy(dst=active, src=act_ps, engine=v)
    nisa.tensor_copy(dst=bsel, src=bsel_ps, engine=v)
    # This replica's slots: lo_eff[n] for n = me B + b, as one matmul with the one-hot [N, B] selector.
    lo_eff = col_tt(s_lo, conf, nl.multiply)
    jrow = tile((N, B))
    for b in range(B):
        nisa.memset(dst=jrow[0:N, b : b + 1], value=float(b), engine=v)
    sel = ts(jrow, nl.equal, s_b, (N, B))
    nisa.tensor_scalar(dst=sel, data=sel, op0=nl.multiply, operand0=col_tt(s_r, me, nl.equal), engine=v)
    lo_ps = psum((1, B))
    nisa.nc_matmul(dst=lo_ps, stationary=lo_eff, moving=sel, is_moving_onezero=True)
    lo_f = tile((1, B))
    nisa.tensor_copy(dst=lo_f, src=lo_ps, engine=v)
    # This replica's foreign slots (pieces p = me F + f), from the free axis of partition 0.
    me0 = me[0:1, 0:1]
    mine = tile((1, P))
    nisa.tensor_scalar(dst=mine, data=pcb[0:1, 0:P, 5], op0=nl.equal, operand0=me0, engine=v)
    pieces_f = tile((1, F, 5))
    for f in range(F):
        am = tile((1, P))
        nisa.tensor_scalar(dst=am, data=pcb[0:1, 0:P, 6], op0=nl.equal, operand0=float(f), engine=v)
        nisa.tensor_tensor(dst=am, data1=am, data2=mine, op=nl.multiply, engine=v)
        nisa.tensor_tensor(dst=am, data1=am, data2=active, op=nl.multiply, engine=v)
        for k, field in enumerate((None, pcb[0:1, 0:P, 2], pcb[0:1, 0:P, 3], bsel, ksel)):
            prod = am
            if field is not None:
                prod = tile((1, P))
                nisa.tensor_tensor(dst=prod, data1=am, data2=field, op=nl.multiply, engine=v)
            nisa.tensor_reduce(dst=pieces_f[0:1, f, k : k + 1], op=nl.add, data=prod, axis=1)
    # aux: the global counts (partition sums on TensorE, the pieces on the free axis).
    unconf = col_tt(asked, conf, nl.subtract)
    aux_ps = psum((1, 2))
    stat = tile((N, 2))
    nisa.tensor_copy(dst=stat[0:N, 0:1], src=unconf, engine=v)
    nisa.tensor_copy(dst=stat[0:N, 1:2], src=conf, engine=v)
    nisa.nc_matmul(dst=aux_ps, stationary=ones_col, moving=stat, is_stationary_onezero=True)
    aux_f = tile((1, 4))
    nisa.memset(dst=aux_f, value=0.0, engine=v)
    nisa.tensor_copy(dst=aux_f[0:1, 0:2], src=aux_ps, engine=v)
    nisa.tensor_reduce(dst=aux_f[0:1, 2:3], op=nl.add, data=active, axis=1)
    outs = []
    any_f = tile((1, 1))
    nisa.tensor_reduce(dst=any_f, op=nl.add, data=pieces_f[0:1, 0:F, 0], axis=1)
    # The extended batch: the group range of each slot, the foreign slots' positions and live flags.
    range_f = tile((1, B + F, 2))
    nisa.memset(dst=range_f, value=0.0, engine=v)
    nisa.tensor_copy(dst=range_f[0:1, 0:B, 0], src=lo_f, engine=v)
    nisa.memset(dst=range_f[0:1, 0:B, 1], value=float(1 << 20), engine=v)
    nisa.tensor_copy(dst=range_f[0:1, B : B + F, 0], src=pieces_f[0:1, 0:F, 1], engine=v)
    nisa.tensor_copy(dst=range_f[0:1, B : B + F, 1], src=pieces_f[0:1, 0:F, 2], engine=v)
    fpos = tile((1, F))
    nisa.tensor_scalar(dst=fpos, data=pieces_f[0:1, 0:F, 2], op0=nl.multiply, operand0=16384.0, engine=v)
    flive = tile((1, F))
    nisa.tensor_copy(dst=flive, src=pieces_f[0:1, 0:F, 0], engine=v)
    # The merge sources of this replica's slots: C[b, p] = 1 when piece p is active on slot (me, b).
    act_b_ps = psum((N, P))
    nisa.nc_matmul(dst=act_b_ps, stationary=ones_row, moving=active, is_stationary_onezero=True, is_moving_onezero=True)
    m3 = tt(m2, act_b_ps, nl.multiply)
    contrib_ps = psum((B, P))
    nisa.nc_matmul(dst=contrib_ps, stationary=sel, moving=m3, is_stationary_onezero=True, is_moving_onezero=True)
    contrib = tile((B, P))
    nisa.tensor_copy(dst=contrib, src=contrib_ps, engine=v)
    # h x 3 FP + f // 2: the export row of the piece's head 0 (3 local heads, FP = F / 2 foreign passes a replica).
    fpasses = max(F // 2, 1)
    pidx = tile((B, P))
    nisa.tensor_tensor(dst=pidx, data1=pcb[0:B, 0:P, 6], data2=pcb[0:B, 0:P, 7], op=nl.subtract, engine=v)
    nisa.tensor_scalar(dst=pidx, data=pidx, op0=nl.multiply, operand0=0.5, engine=v)
    hrow = tile((B, P))
    nisa.tensor_scalar(dst=hrow, data=pcb[0:B, 0:P, 5], op0=nl.multiply, operand0=float(3 * fpasses), engine=v)
    nisa.tensor_tensor(dst=pidx, data1=pidx, data2=hrow, op=nl.add, engine=v)
    merge_f = tile((B, 3))
    nisa.tensor_reduce(dst=merge_f[0:B, 0:1], op=nl.add, data=contrib, axis=1)
    hi_t = tile((B, P))
    nisa.tensor_tensor(dst=hi_t, data1=contrib, data2=pidx, op=nl.multiply, engine=v)
    nisa.tensor_reduce(dst=merge_f[0:B, 2:3], op=nl.maximum, data=hi_t, axis=1)
    # lowest = BIG - max(contrib x (BIG - p)), then 0 when no piece.
    lo_t = tile((B, P))
    nisa.tensor_scalar(dst=lo_t, data=pidx, op0=nl.subtract, operand0=_SPLIT_BIG, reverse0=True, engine=v)
    nisa.tensor_tensor(dst=lo_t, data1=lo_t, data2=contrib, op=nl.multiply, engine=v)
    lo_max = tile((B, 1))
    nisa.tensor_reduce(dst=lo_max, op=nl.maximum, data=lo_t, axis=1)
    nisa.tensor_scalar(dst=lo_max, data=lo_max, op0=nl.subtract, operand0=_SPLIT_BIG, reverse0=True, engine=v)
    has = tile((B, 1))
    nisa.tensor_scalar(dst=has, data=merge_f[0:B, 0:1], op0=nl.greater, operand0=0.0, engine=v)
    nisa.tensor_tensor(dst=merge_f[0:B, 1:2], data1=lo_max, data2=has, op=nl.multiply, engine=v)
    for name, src, shape in (
        ("lo_mine", lo_f, (1, B)),
        ("pieces_mine", pieces_f, (1, F, 5)),
        ("any_mine", any_f, (1, 1)),
        ("aux", aux_f, (1, 4)),
        ("range_ext", range_f, (1, B + F, 2)),
        ("foreign_pos", fpos, (1, F)),
        ("foreign_live", flive, (1, F)),
        ("merge_mine", merge_f, (B, 3)),
    ):
        as_int = tile(shape, int32)
        nisa.tensor_copy(dst=as_int, src=src, engine=v)
        hbm = nl.ndarray(shape, dtype=int32, buffer=nl.hbm, name=f"{tag}_{name}")
        nisa.dma_copy(dst=hbm, src=as_int, name=f"{tag}_{name}_store")
        outs.append(hbm)
    return tuple(outs)


def _split_v5_rider(
    sources: dict,
    mla_split: dict,
    stats,
    own,
    foreign,
    rank_info,
    dp: int,
    sb,
    queries: int = 8,
    ext_inputs=None,
):
    """Put the v5 split's rider in ``sources`` (``tkg_dense_mlp_2srv.SPLIT_RIDER_KEY``): layer 0's DP x gather takes
    this replica's ``stats`` ``[1, S]`` (and in the split program its slot and piece records,
    ``_split_v5_records``) as more buffers into ``[dp, W]`` destinations; right after it the gathered stats are stored
    to the output and the split program runs its step check (``split_v5_step_check``).

    Return the outputs (the gathered stats ``[dp, S]``, then the aux ``[1, 4]`` INT32: the step check's in the split
    program, zeros in the idle program), the tensors to hold to the end (the gather destinations: a first-of-step
    collective skips its handshake, so no later tensor may take their bytes) and the step check's results (a dict with
    ``lo_mine``, ``pieces_mine``; empty in the idle program). ``ext_inputs`` (the split program's extended batch: ``(position, live, B)``): the step check takes the parity
    rule and writes the foreign slots' rows of ``position`` / ``live`` (``split_v5_extended_inputs``)."""
    int32 = nl.int32
    S = mla_split["stats_words"]
    kernel_assert(SPLIT_RIDER_KEY not in sources, "split rider key clash")
    stats_src = nl.ndarray((1, S), dtype=int32, buffer=nl.hbm, name="mla_split_rider_src")
    stats_dst = nl.ndarray((dp, S), dtype=int32, buffer=nl.hbm, name="mla_split_rider_dst")
    nisa.dma_copy(dst=stats_src, src=stats, name="mla_split_stats_store")
    pairs = [(stats_src, stats_dst)]
    if own is not None:
        own_src, pc_src = _split_v5_records(own, foreign, rank_info, sb, "mla_split_rec")
        own_dst = nl.ndarray((dp, own_src.shape[1]), dtype=int32, buffer=nl.hbm, name="mla_split_own_dst")
        pc_dst = nl.ndarray((dp, pc_src.shape[1]), dtype=int32, buffer=nl.hbm, name="mla_split_pc_dst")
        pairs += [(own_src, own_dst), (pc_src, pc_dst)]
    held = tuple(dst for _, dst in pairs)
    checks = {}

    def step_check():
        if own is not None:
            names = (
                "lo_mine",
                "pieces_mine",
                "any_mine",
                "aux",
                "range_ext",
                "foreign_pos",
                "foreign_live",
                "merge_mine",
            )
            results = split_v5_step_check(
                pairs[1][1],
                pairs[2][1],
                rank_info,
                dp,
                own.shape[0],
                foreign.shape[0],
                queries,
                parity=ext_inputs is not None,
            )
            checks.update(zip(names, results, strict=True))
            if ext_inputs is not None:
                position_ext, live_ext, batch = ext_inputs
                F = foreign.shape[0]
                nisa.dma_copy(
                    dst=position_ext[batch : batch + F, 0:1],
                    src=checks["foreign_pos"].reshape((F, 1)),
                    name="mla_split_pos_foreign",
                )
                if live_ext is not None:
                    nisa.dma_copy(
                        dst=live_ext[batch : batch + F, 0:1],
                        src=checks["foreign_live"].reshape((F, 1)),
                        name="mla_split_live_foreign",
                    )

    gathered = nl.ndarray((dp, S), dtype=int32, buffer=nl.shared_hbm, name="mla_split_gathered")
    aux = nl.ndarray((1, 4), dtype=int32, buffer=nl.shared_hbm, name="mla_split_aux")
    if own is None:
        zeros = sb.tile((1, 4), int32) if sb is not None else nl.ndarray((1, 4), dtype=int32, buffer=nl.sbuf)
        nisa.memset(dst=zeros, value=0, engine=nisa.engine.vector)
        nisa.dma_copy(dst=aux, src=zeros, name="mla_split_aux_store")

    def after():
        nisa.dma_copy(dst=gathered, src=stats_dst, name="mla_split_gathered_store")
        step_check()
        if own is not None:
            nisa.dma_copy(dst=aux, src=checks["aux"], name="mla_split_aux_store")

    sources[SPLIT_RIDER_KEY] = {"pairs": pairs, "after": after}
    return (gathered, aux), held, checks


@nki.jit
def kimi_k3_verify_model_2srv(
    stream: nl.NkiTensor,
    norm_weights: nl.NkiTensor,
    attn_res_weights: nl.NkiTensor,
    kda_w_in: nl.NkiTensor,
    kda_w_out: nl.NkiTensor,
    kda_w_small: nl.NkiTensor,
    kda_params: nl.NkiTensor,
    mla_in_proj: nl.NkiTensor,
    mla_q_a_norm: nl.NkiTensor,
    mla_q_b_proj: nl.NkiTensor,
    mla_kv_a_norm: nl.NkiTensor,
    mla_kv_b_proj: nl.NkiTensor,
    mla_o_proj: nl.NkiTensor,
    moe_in_proj: nl.NkiTensor,
    moe_out_proj: nl.NkiTensor,
    moe_vectors: nl.NkiTensor,
    expert_gate_up_weights: nl.NkiTensor,
    expert_gate_up_weights_scale: nl.NkiTensor,
    expert_down_weights: nl.NkiTensor,
    expert_down_weights_scale: nl.NkiTensor,
    dense_gate_up: nl.NkiTensor,
    dense_down: nl.NkiTensor,
    kda_state_page: nl.NkiTensor,
    kda_record_page: nl.NkiTensor,
    kda_replay_len: nl.NkiTensor,
    kda_state_position: nl.NkiTensor,
    mla_block_table: nl.NkiTensor,
    mla_slot: nl.NkiTensor,
    position_rows: nl.NkiTensor,
    rank_info: nl.NkiTensor,
    tp_group: ReplicaGroup,
    dp_group: ReplicaGroup,
    world_group: ReplicaGroup,
    first_layer: int,
    eps: float,
    queries: int = DEFAULT_QUERIES,
    debug: bool = False,
    dma_gates: bool | str = True,
    pool_of_layer: tuple = None,
    mla_loop_path: bool = False,
    kv_pool_0: nl.NkiTensor = None,
    kv_pool_1: nl.NkiTensor = None,
    kv_pool_2: nl.NkiTensor = None,
    kv_pool_3: nl.NkiTensor = None,
    kv_pool_4: nl.NkiTensor = None,
    kv_pool_5: nl.NkiTensor = None,
    kv_pool_6: nl.NkiTensor = None,
    kv_pool_7: nl.NkiTensor = None,
    kv_pool_8: nl.NkiTensor = None,
    kv_pool_9: nl.NkiTensor = None,
    kv_pool_10: nl.NkiTensor = None,
    kv_pool_11: nl.NkiTensor = None,
    kv_pool_12: nl.NkiTensor = None,
    kv_pool_13: nl.NkiTensor = None,
    kv_pool_14: nl.NkiTensor = None,
    kv_pool_15: nl.NkiTensor = None,
    kv_pool_16: nl.NkiTensor = None,
    kv_pool_17: nl.NkiTensor = None,
    kv_pool_18: nl.NkiTensor = None,
    kv_pool_19: nl.NkiTensor = None,
    kv_pool_20: nl.NkiTensor = None,
    kv_pool_21: nl.NkiTensor = None,
    kv_pool_22: nl.NkiTensor = None,
    kv_pool_23: nl.NkiTensor = None,
    final_attn_res_weight: nl.NkiTensor = None,
    final_norm_weight: nl.NkiTensor = None,
    lm_head_weight: nl.NkiTensor = None,
    greedy: bool = False,
    return_logits: bool = False,
    return_aux: bool = False,
    aux_layers: tuple = AUX_LAYERS,
    aux_tap: str = "mixture",
    mla_fp8_pages: bool = False,
    mla_mode: str = "k1",
    mla_zeroed_pages: bool = False,
    mla_late_tail_weights: bool = False,
    mla_late_o_proj_weights: bool = False,
    mla_late_tail_frames: bool = False,
    mla_pass_q_prefetch: int = None,
    mla_q_gather_buffers: int = None,
    mla_tail_weights_after_rs: bool = None,
    mla_pass_setup_off_pool: bool = None,
    mla_slot_rows_sbuf: bool = None,
    mla_skip_stage_store: bool = None,
    mla_pass_rows_ahead: bool = None,
    mla_out_ar: bool = None,
    mla_k1_rows: bool = None,
    mla_k1_rows_options: dict = None,
    live_rows: nl.NkiTensor = None,
    mla_k1_packed: bool = None,
    mla_k1_pack_options: dict = None,
    mla_state_bf16_o: bool = False,
    mla_critical_dma_priority: bool | str = None,
    mla_cache_write_off_sp: bool = None,
    mla_gemv_bands: bool = None,
    mla_batched_cache_write: bool = None,
    mla_null_rows_once: bool = None,
    mla_p2_entry_act: bool = None,
    mla_step_items: bool = None,
    lora_norm_eps: float = LORA_NORM_EPS,
    cc_source_pingpong: bool = True,
    mix_cache: bool = False,
    attn_res_cached_bank_sums: bool = None,
    attn_res_early_sums: bool = None,
    attn_res_mla_mlp_early: bool = None,
    attn_res_ahead: bool = False,
    attn_res_ahead_mlp: bool = False,
    attn_res_ahead_min_entries: int = 0,
    early_ffn_gate: bool = None,
    ffn_weights_after_attn: bool = None,
    kda_replay: str = "records",
    kda_state_slot_page: nl.NkiTensor = None,
    kda_write_state_slots: nl.NkiTensor = None,
    kda_prework_ahead: bool = False,
    kda_w_in_buffers: int = None,
    kda_w_in_prefetch: str = "off",
    kda_w_in_prefetch_chunks: int = 0,
    kda_state_loads_first: bool = None,
    kda_slot_pages_first: bool = None,
    kda_token_tiles: bool = None,
    kda_slot_store_queues: int = None,
    rows_hT_batched: bool = None,
    attn_res_weight_rows: bool = None,
    kda_states_after_out_ar: bool = None,
    aux_row_ring: bool = None,
    kda_x_quant_hook: bool = None,
    kda_merged_state_loads: bool = None,
    kda_conv_dtype: str = "bf16",
    kda_chain: str = "sequential",
    kda_slot_state_depth: int = None,
    kda_slot_state_per_token: bool = None,
    kda_slot_store_engine: str = None,
    kda_shared_state_ring: bool = False,
    kda_state_dtype: str = "bf16",
    kda_paired_pages: bool = False,
    moe_half_weight_buffers: bool = False,
    moe_weight_buffers: int = None,
    moe_scale_buffers: int = None,
    moe_transpose_lookahead: int = None,
    moe_cross_expert_order: bool = None,
    moe_transpose_block: bool = None,
    moe_static_alt_tiles: bool | str = None,
    moe_routed_store_form: str = "pe",
    world_size: int = 256,
    routed_tp: int = None,
    moe_form: str = "moe_2srv",
    dp_size: int = None,
    moe_gather: str = "dp_tp",
    moe_wide_half_weight_buffers: int = None,
    moe_expert_call_tokens: int | str = EXPERT_CALL_AUTO,
    moe_expert_prefetch: dict = None,
    moe_expert_prefetch_uses: dict = None,
    moe_expert_prefetch_scales: dict = None,
    moe_expert_prefetch_start: dict = None,
    moe_expert_queues: int = 1,
    moe_routed_reduce: str = None,
    moe_routed_stats_ag: bool = False,
    moe_rows_latent_fp8: bool = False,
    moe_rows_load_split: int = None,
    moe_routed_own_rows: bool = False,
    moe_rows_dst_pingpong: bool = True,
    moe_latent_gather_mx: bool = False,
    moe_out_proj_form: str = None,
    pair_shared_cc_dsts: tuple = (),
    norm_scalar_squares: int = None,
    attn_res_scalar_products: int = None,
    attn_res_bank_dma: str = "auto",
    attn_res_mlp_early_entries: int = None,
    moe_router_rows_drop_selection: bool = False,
    moe_glue: dict = None,
    moe_sums_after_routed: bool = None,
    attn_res_sums_after_moe_output: bool = None,
    attn_res_bank_in: nl.NkiTensor = None,
    return_attn_res_bank: bool = False,
    attn_res_fence_in: nl.NkiTensor = None,
    kda_w_in_scale: nl.NkiTensor = None,
    mla_in_proj_scale: nl.NkiTensor = None,
    mla_q_b_proj_scale: nl.NkiTensor = None,
    mla_kv_b_proj_scale: nl.NkiTensor = None,
    mla_o_proj_scale: nl.NkiTensor = None,
    router_dtype: str = "bf16",
    moe_router_weight: nl.NkiTensor = None,
    moe_in_proj_scale: nl.NkiTensor = None,
    kda_w_out_scale: nl.NkiTensor = None,
    moe_out_proj_scale: nl.NkiTensor = None,
    kda_out_ar_sbuf: bool = None,
    kda_small_after_out_ar: bool = None,
    kda_conv_load_gather: bool = None,
    kda_batched_conv_stores: bool = None,
    kda_null_page: int = 0,
    kda_slot_pages_contiguous: bool = None,
    kda_batched_slot_conv_stores: bool = None,
    kda_bf16_update_rows: bool = None,
    kda_slot_conv_persistent: bool = None,
    kda_update_bcast_depth: int = None,
    kda_head_pipeline: str = None,
    kda_w_in_head_major: bool = False,
    kda_out_ar_rows: bool = None,
    kda_slot_conv_from_inputs: bool = None,
    kda_conv_products_engine: str = None,
    kda_recurrence_group: int = None,
    mx_qk: str = None,
    mla_split: dict = None,
    mla_split_info: nl.NkiTensor = None,
    mla_split_table: nl.NkiTensor = None,
    mla_split_stats: nl.NkiTensor = None,
    mla_split_own: nl.NkiTensor = None,
    mla_split_foreign: nl.NkiTensor = None,
    kda_page_groups: tuple = None,
    mla_fp8_pv: bool = None,
    cc_pad256: bool = False,
    moe_mx_tile: int = None,
    routed_layers: tuple = None,
    moe_routed_dynamic: bool = False,
    routed_alt0_expert_gate_up_weights: nl.NkiTensor = None,
    routed_alt0_expert_gate_up_weights_scale: nl.NkiTensor = None,
    routed_alt0_expert_down_weights: nl.NkiTensor = None,
    routed_alt0_expert_down_weights_scale: nl.NkiTensor = None,
    routed_alt1_expert_gate_up_weights: nl.NkiTensor = None,
    routed_alt1_expert_gate_up_weights_scale: nl.NkiTensor = None,
    routed_alt1_expert_down_weights: nl.NkiTensor = None,
    routed_alt1_expert_down_weights_scale: nl.NkiTensor = None,
    max_verify_rows: int = None,
    grammar_ids: nl.NkiTensor = None,
    grammar_rows_in: nl.NkiTensor = None,
    grammar_class_of: nl.NkiTensor = None,
    grammar_next: nl.NkiTensor = None,
    grammar_mask_bias: nl.NkiTensor = None,
    mla_qkv_parity: bool = False,
):
    """Run the Kimi-K3 decoder layers of a range on Q tokens of each sequence of one DP replica of a multi-server world.

    The positional arguments are those of ``verify_model.kimi_k3_verify_model``, with the MoE and dense weights of
    ``decode_model_2srv.kimi_k3_decode_model_2srv``. The keyword arguments are its verify, MLA, KDA and tail options
    with the same names, plus the world and the MoE options of the multi-server decode. The decode-body options of the
    multi-server decode (the packed MLA loop, the staged MLA, the KDA decode forms) do not apply: the call fails with
    an unknown argument. The kernel updates the page pools in place and returns them, so that they alias the inputs.

    Dimensions:
        Q: tokens of each sequence (``queries``), 1 to 9. B: sequences of the replica. R = Q * B rows, at most
            ``MAX_VERIFY_ROWS``. W: ``world_size``, 128 ranks for each server. DP: ``W / 32`` replicas; one
            feed-forward call gathers T = DP * R tokens.

    Args:
        stream: ``[R, H]`` bf16 embeddings; row ``j * B + b`` is token ``j`` of sequence ``b``.
        norm_weights .. dense_down: as in ``kimi_k3_decode_model_2srv`` (TP32 / CP32 attention layouts, the DP
            projections and the experts of group ``g % G``, part ``g // G``, the dense MLP of the world).
        kda_state_page, kda_record_page, kda_replay_len, kda_state_position, mla_block_table, mla_slot, position_rows:
            as in ``kimi_k3_verify_model``.
        rank_info, tp_group, dp_group, world_group, first_layer, eps: as in ``kimi_k3_decode_model_2srv``.
        queries: Q.
        debug: also return the stream after each layer, ``[L, R, H]``.
        dma_gates, pool_of_layer, mla_loop_path, kv_pool_i: as in ``kimi_k3_decode_model_2srv``.
        final_attn_res_weight, final_norm_weight, lm_head_weight, greedy, return_logits: the tail on all R rows, as in
            ``kimi_k3_decode_model_2srv``; ``lm_head_weight`` is the V/32 vocab shard of the TP rank.
        return_aux, aux_layers, aux_tap: the aux states of the drafter, as in ``kimi_k3_verify_model``.
        grammar_ids, grammar_rows_in, grammar_class_of, grammar_next, grammar_mask_bias: the static grammar mask of the
            verify rows (``tkg_grammar_mask``), all or none. None (the default) gives no grammar mask.
            ``grammar_ids`` ``[R, 1]`` INT32 is the input token of each row (the stream order), ``grammar_rows_in``
            ``[B, 4]`` INT32 the start of each sequence, the others the tables of this rank (``grammar_mask_bias``
            ``[P * M, V/32]`` bf16 holds the columns of its vocab shard). The tail adds each row's bias in the
            ``lm_head`` PSUM epilogue, so the logits and the greedy tokens are those of the masked rows.
        mla_fp8_pages, mla_zeroed_pages, lora_norm_eps, kda_conv_dtype: as in ``kimi_k3_decode_model_2srv``; the decode
            of the deployment must take the same values.
        mla_mode: ``k1`` (default) or ``per_token`` (``MLA_MODES``; see the module docstring).
        mla_late_tail_weights: ``k1`` on the loop path: ``mla_block_attention`` ``late_tail_weights``. The values do
            not change.
        mla_late_o_proj_weights: with ``mla_late_tail_weights``: ``mla_block_attention`` ``late_o_proj_weights``
            (each pass loads its o_proj weights after its value projection). The values do not change.
        mla_late_tail_frames: with ``mla_late_o_proj_weights``: ``mla_block_attention`` ``late_tail_frames`` (each pass
            frees its gate weights after its gate projection and loads its state dense, so the MLA tail tiles reuse
            those bytes of the explicit region). The values do not change.
        mla_pass_q_prefetch: ``k1`` with the call query gather: the first gather tiles of each later MLA pass load
            before the loop of the pass before (``mla_block_attention`` ``pass_q_prefetch``). The values do not
            change. 0 to ``PASS_Q_PREFETCH_MAX_TILES`` (2). None (default): in ``PASS_Q_PREFETCH_FORM_SET``
            (``served_defaults``) the tile count of the form (``PASS_Q_PREFETCH_TILES_BY_FORM``, else
            ``PASS_Q_PREFETCH_TILES``), 0 in every other call. A given value wins; a given 0 keeps the program.
        mla_q_gather_buffers: ``k1`` with the call query gather: 2 gives the query gather tiles of each MLA pass two stack
            tiles in turn (``mla_block_attention`` ``q_gather_buffers``), so the loads of a tile do not wait for the
            transposes of the tile before. The values do not change. 1 or 2. None (default): 2 in
            ``Q_GATHER_BUFFER_FORM_SET`` (``served_defaults``), 1 in every other call. A given 1 keeps the program.
        mla_tail_weights_after_rs: ``k1`` above 16 rows with the BF16-o state and the MXFP8 MLA input projection: the
            tail's output gate and o_proj weights wait for the call-level numerator state reduce-scatter and load on
            the ScalarE queue (``mla_block_attention`` ``tail_weights_after_rs``). The values do not change. None
            (default): on in ``TAIL_WEIGHTS_AFTER_RS_FORM_SET`` (``served_defaults``), off in every other call. A given
            False keeps the program. On a call without the call-level BF16-o state reduce-scatter of the fit passes
            or without the MXFP8 mla_in_proj the default has no effect, and a given True fails with a kernel_assert.
        mla_pass_setup_off_pool: ``k1`` on the packed p2 loop: ``setup_off_pool`` in the loop's options
            (``tkg_mla_paged._packed_key_loop`` ``setup_off_pool``: the shared sum offsets, and the sum-tile zero fills and
            stage TI copies of the pass setup on VectorE). The values do not change. None (default): on in
            ``PASS_SETUP_OFF_POOL_FORM_SET`` (``served_defaults``), off in every other call. A given False keeps the
            program. On a call without the p2 loop (for example the mx_qk "RA" rows form) the default has no effect,
            and a given True fails with a kernel_assert.
        mla_slot_rows_sbuf: ``k1`` on the packed p2 loop with ``stage_rows_sbuf`` (the k7 form): ``slot_rows_sbuf`` in
            the loop's options (``tkg_mla_paged._packed_key_loop`` ``slot_rows_sbuf``: every slot row of the loops reads
            the SBUF stage tile, not its HBM store). The values do not change. None (default): on in
            ``SLOT_ROWS_SBUF_FORM_SET`` (``served_defaults``), off in every other call. A given False keeps the program.
        mla_skip_stage_store: with ``mla_slot_rows_sbuf``: ``skip_stage_store`` in the loop's options
            (``tkg_mla_paged._packed_key_loop`` ``skip_stage_store``: no HBM store of the stage rows, which no slot row
            reads then). The values do not change. None (default): on in ``SKIP_STAGE_STORE_FORM_SET``
            (``served_defaults``), off in every other call. A given False keeps the program. On a call without the p2
            loop (for example the mx_qk "RA" rows form) the defaults of both keys have no effect, and a given True
            fails with a kernel_assert.
        mla_pass_rows_ahead: ``k1`` with the packed p2 loop and the call query gather: after the loop of each MLA pass
            and before its state stores, the next pass's own-key current rows (4 rows a DMA) and its query gather
            tiles past the ``mla_pass_q_prefetch`` tiles load into heap tiles, which the next pass frees before its
            loop (``mla_block_attention`` ``pass_rows_ahead``). The values do not change. None (default): True in
            ``PASS_ROWS_AHEAD_FORM_SET`` (``served_defaults``), False in every other call. A given False keeps the
            program.
        mla_out_ar: ``k1`` on the loop path: the MLA output sum of each layer is one TP32 all-reduce in an HBM pair
            (``mla_block_attention`` ``out_ar_call``), in place of the output reduce-scatter and the all-gather: one CC
            less for each MLA layer. The runtime picks each collective's reduction order, so the bits are a device
            check. None: the default is that of ``served_defaults``.
        mla_k1_rows: ``k1`` on the loop path: the key-stationary CP32 rows form of the MLA call on the R rows of each
            replica (``verify_model.kimi_k3_verify_model`` ``mla_k1_rows``). The MLA of this entry is always the CP32
            MLA of each TP32 replica, so the gate is that of the one-server verify without its CP32 and DP4 terms. It
            changes the per-row sums, so its identity twin is the decode with ``mla_rows_twin``
            (``decode_model_2srv.kimi_k3_decode_model_2srv``). B in ``tkg_mla_block.ROWS_SEQUENCES`` sequences, and above
            ``tkg_mla_block.ROWS_PASS_ROWS`` rows only in the fit form (a table of at least 64 local blocks). None: the
            default is that of ``served_defaults`` (on in the k3 form with FP8 pages).
        mla_k1_rows_options: with ``mla_k1_rows``, more keys of the rows form (``tkg_mla_block.ROWS_OPTIONS``; see
            ``verify_model.kimi_k3_verify_model``). The twin decode takes the values of the keys in
            ``tkg_mla_block.ROWS_TWIN_OPTIONS`` (``mla_rows_twin_options``). None: the default is that of ``served_defaults``
            (``SERVED_ROWS_OPTIONS`` with the rows form of the k3 form with FP8 pages, and ``rows_pad_skip`` too when
            the call takes ``live_rows``).
        live_rows: None (default), or the runner's INT32 ``[B, 1]`` HBM live-row mask of the replica (1: slot b holds
            a live request, 0: an empty slot; B as in ``mla_block_table``). The pad skips of the k1 MLA call
            (``rows_pad_skip`` in ``mla_k1_rows_options``, ``pad_skip`` in ``mla_k1_pack_options``) take their padding
            sequences from it, and need it: a live row can sit at position 0. The served k7 form on the packed loop
            takes ``pad_skip`` by default when the live rows are given (a caller's ``pad_skip`` wins). With
            ``routed_compact`` in ``moe_glue``, the rows of an empty slot also route to no expert in the MoE layers
            (``live_pad_rows``), so they take no compaction slot; a live row keeps its bits.
        mla_k1_packed, mla_k1_pack_options: ``k1`` on the loop path: the packed prior-key loop of the CP32 MLA call
            (``verify_model.kimi_k3_verify_model`` ``mla_k1_packed``, with ``mla_k1_pack_options`` as the overrides of
            its packed-loop form). Each row stays bitwise equal to its decode step. Each pass holds whole sequences:
            at most 16 rows, or the sequence-major passes of the fit form. Not with ``mla_k1_rows``: the packed loop
            takes no transposed scores. None: the default is that of ``served_defaults``; in the ``P2_FORMS`` forms the
            packed loop adds ``p2`` to the options where it fits (``served_pack_options``). ``{"q_fp8": True}`` on
            that p2 loop with ``q_hbm`` rounds the query of the call once to e4m3 at unit scale (saturated at 448)
            before its all-gather, which moves half the bytes, and every query consumer takes the e4m3 query, the query
            form of the GPU decode (accuracy class; with ``mx_qk`` "E" its block scales then round nothing more); off
            by default (``Q_FP8_P2_FORM_SET``, empty). ``{"q_fp8": "latent"}`` rounds only the latent query that way
            and keeps the rope query BF16 (``tkg_mla_block`` ``packed_prior_options``; a uint16 gather pair of 1,920 bytes a
            row in place of 1,728 or 3,456), off by default.
        mx_qk: the latent QK form of the k1 MLA call in the ``SERVED_FORMS`` forms (``MX_QK_TRN3_MODES``, the
            per-config verify key): None (default) keeps the program of each form; "off" takes the BF16 latent QK
            (the k3 form takes the packed loop, also with FP8 pages); "E" takes the k7 / k7b2 packed p2 loop with
            ``MX_QK_E_OPTIONS`` (the e4m3 latent query in scale blocks of 32 as one MX matmul a sub-slot, and the item
            plans built once for the MLA layers); "RA" takes the CP32 rows form with ``SERVED_ROWS_OPTIONS`` in
            ``MX_QK_TRN3_MODES["RA"]`` (needs ``mla_fp8_pages``). "E" and "RA" are accuracy class (the GB300 decode's e4m3
            query). The mode is authoritative over ``mla_k1_rows`` / ``mla_k1_packed`` / ``mla_k1_rows_options``
            (``mx_qk_keys``; a non-empty rows-options dict wins under "RA"); a given pack option wins;
            ``check_mx_qk`` refuses a mode that the form or the resolved keys do not take.
        mla_fp8_pv: with ``mx_qk`` "E", the FP8 P@V pass of the k7 / k7b2 packed p2 loop (``served_pack_options``
            ``fp8_pv``; the per-config verify key): the scores with the keys on partitions, MXFP8 P (``nisa.quantize_mx``,
            an E8M0 scale for each 32 keys) and one MX P@V and weight sum a sub-slot, with the natural MX query words
            (``mx_natural``). Accuracy class. None (default): the form's default (``FP8_PV_FORM_SET``: k7b2 and
            k7b1); False gives the program without the FP8 P@V pass. ``check_fp8_pv`` refuses True without the "E" set.
        cc_pad256: the FP32 pair of the MLA state reduce-scatter of each layer has padded columns, so that each
            rank's part is a multiple of 256 bytes (``mla_block_collective_hbm`` ``cc_pad256``: the BF16-o sums
            ``[96 P, 16]`` -> ``[96 P, 64 / P]``). The pad columns are not read, so the values do not change. Needs
            the loop-path ``k1`` form. False (default) keeps the program.
        mla_state_bf16_o: ``k1``: the MLA state reduce-scatter of each layer carries the numerators in BF16 and the
            sums in FP32 in the views of its FP32 HBM pair (``mla_block_attention`` ``state_bf16_o``): two
            reduce-scatters, about half the bytes (the k3 verify's 3.0 MiB state goes under the runtime's 2 MiB
            algorithm switch). It rounds the numerators: an accuracy-class option. The decode twin takes
            ``kimi_k3_decode_model_2srv`` ``mla_state_bf16_o``. False (default) keeps the program.
        mla_critical_dma_priority, mla_cache_write_off_sp, mla_gemv_bands: ``k1``: the bitwise MLA levers of
            ``verify_model.verify_bodies`` (``mla_block_attention`` ``critical_dma_priority``, ``cache_write_off_sp``
            and ``gemv_bands``): DMA priority 0 on the small critical MLA DMAs, the owner cache write off the SP
            queue, and the narrow GEMVs in two PE column bands (at most 64 rows). They act on the shared parts of the
            k1 call, so on the packed and the rows forms. The output does not change. None (default): on in the
            ``SERVED_FORMS`` forms on the k1 loop path (``served_defaults``), off in every other call.
            ``mla_critical_dma_priority`` "query" also puts the query path and the call-level state store / load on
            the SP hardware DGE queue at priority 0. In ``QUERY_DMA_FORMS`` on (None or True, which the serving caller passes for
            the served forms) takes "query"; False turns every part off.
        mla_batched_cache_write: ``k1``: the call writes its current MLA rows with one vector-indirect DMA in place
            of one DMA per row (``verify_model.verify_bodies`` ``mla_batched_cache_write``, ``mla_block_attention``
            ``batched_cache_write``). The compiler orders every two dynamic-offset writes of a pool, so the per-row
            writes of the Q*B rows run one after another. A row that the rank does not own, and a pad row, goes to
            its own row of the null page (page 0) in place of its row 0. The values read do not change; only the
            null page holds other bytes. At most 128 rows a call. None: the default is that of ``served_defaults``
            (on in ``BATCHED_CACHE_WRITE_FORMS``, else off); False keeps the program.
        mla_null_rows_once: with ``mla_batched_cache_write`` on the loop path: the kernel builds the partition index
            column of the null-page rows once, at a heap address of its own (``install_null_rows``), and every MLA
            layer reads it, in place of one GpSimd iota a layer into a reused tile (whose write waits for the last
            reader of those bytes and so holds the GpSimd queue). The same values, so the output does not change.
            None: the default of ``served_defaults`` (on in ``NULL_ROWS_ONCE_FORMS``, else off).
        mla_p2_entry_act: ``k1`` on the packed p2 loop: ``p2_entry_act`` in the loop's options (``tkg_mla_paged._packed_key_loop``
            ``p2_entry_act``: the loop-entry references and the SBUF slot rows on the ScalarE queue, the stage zero
            fill narrowed to the bytes no copy writes; with ``mx_qk`` the fill of every byte but the references). The
            output does not change. None (default): on in ``P2_ENTRY_ACT_FORMS`` (``served_defaults``), off in every
            other call; False turns it off (the kill switch).
        mla_step_items: ``k1`` on the packed p2 loop: ``hoist_plans`` with ``step_items`` in the loop's options (the item
            plans and item rows of the call built by its first MLA layer and read by the later ones, which skip their
            item-row gathers and run-scale matmul). The output does not change. None (default): on in ``STEP_ITEMS_FORMS``
            (``served_defaults``), off in every other call; False turns them off (the kill switch), except
            that ``mx_qk`` "E" takes the two options itself (``MX_QK_E_OPTIONS``).
        moe_expert_prefetch: None (default) or ``{}`` keeps the program (``served_defaults`` turns it on in no form).
            ``{"kda": W, "mla": W}``: each MoE layer's expert ring (W whole weight buffers, 0 to 6, by the type of the
            layer's attention) starts loading at the start of the layer's attention (``decode_block.decode_block_body``
            ``moe_prefetch``), so the expert call starts on a loaded ring. W = 0 keeps the expert call's own first
            loads for that type. Size W to the SBUF that the attention leaves free: the MLA loop path gives every MLA
            tile an explicit address, so a ring that is live across an MLA attention does not fit (use ``mla`` 0, or
            start the MLA ring at the MoE front with ``moe_expert_prefetch_start``), and with the BF16 ``kda_w_in``
            ring of 8 buffers a 93-layer k3 call does not fit ``{"kda": 3}``. With
            ``moe_cross_expert_order`` the ring loads in the cross-expert order and W is 0 or 3. The ring loads take
            the lowest trn3 DMA QoS level (``PREFETCH_DMA_PRIORITY``), so they stream in the attention's idle DMA
            windows and do not delay its weight and state loads. ``"kda_after_in_proj": 1`` (0 or absent keeps the
            program; needs ``kda`` >= 1) starts the ring of each MoE layer after a KDA attention right after that
            attention's input projection, after its last ``kda_w_in`` loads (``decode_block.decode_block_body``
            ``moe_prefetch_kda_after_in_proj``), so the ring does not share the DMA rings with the projection's weight
            stream. The values do not change.
        moe_expert_prefetch_uses: None (default) keeps the program. With ``moe_expert_prefetch`` and
            ``moe_cross_expert_order``, ``{"mla": k}`` (or ``"kda"``, 1 to that type's W): the ring of that layer type
            loads only its first k buffers and its scale groups at the ring start (during the attention, or at the MoE
            front with ``moe_expert_prefetch_start``), and the expert call loads the others at its start
            (``decode_block.decode_block_body`` ``moe_prefetch_uses``). ``{"kda": 3, "mla": 3}``
            with ``{"mla": 2}`` keeps 2 buffers and 2 scale groups (64,512 B a partition) live across each MLA
            attention, where 3 do not fit. The values do not change.
        moe_expert_prefetch_scales: None (default) keeps the program. With ``moe_expert_prefetch_uses``, ``{"mla":
            False}`` leaves that layer type's scale groups to the expert call's start (``decode_block.decode_block_body``
            ``moe_prefetch_scales``). The values do not change.
        moe_expert_prefetch_start: None (default) keeps the program. ``{"mla": "front"}``, with ``moe_expert_prefetch``
            ``mla`` >= 1: the ring of each MoE layer after an MLA attention starts at the MoE front, after the MLA
            tiles, not at the start of the attention (``decode_block.decode_block_body`` ``moe_prefetch_start``). The
            values do not change.
        moe_expert_queues: the DMA queues of the static expert loop's weight loads (``moe_tkg_decode``
            ``expert_queues``): 1 (default) keeps the program; 2 loads the odd experts' weights with hardware DGE from
            the Sync engine, a second queue, in every MoE layer and in the ``moe_expert_prefetch`` rings. The values do
            not change.
        moe_routed_reduce, moe_routed_stats_ag: the reduction of the routed partials of the MoE layers, as in
            ``kimi_k3_decode_model_2srv``, on the T gathered tokens: ``ar`` keeps the program (None: the default is that of ``served_defaults``); ``rs_ag``
            takes only the ``VERIFY_ROUTED_RS_AG_FORMS`` forms and needs ``cc_source_pingpong``. ``rs_ag`` rounds
            differently (other reduction owners): an accuracy-class option. ``rs_fold`` (the same forms and rules) takes
            the same reduce-scatter and folds the latent norm into one FP32 output all-reduce
            (``tkg_moe_2srv.fold_out_reduce``): two collectives a layer, as ``ar``; also an accuracy-class option.
        moe_rows_latent_fp8: the MoE router rows carry each rank's 112 latent values as float8_e4m3fn with one FP32
            scale (amax / 448), 26 words fewer a row (``tkg_moe_2srv`` ``rows_latent`` ``fp8``): the DP and TP32 row
            gathers move 23 % fewer bytes. It rounds the expert input: an accuracy-class option. False (default) keeps
            the program.
        moe_rows_load_split: 1 keeps the program (None: the default is that of ``served_defaults``). 2 or 4 (``tkg_moe_2srv.ROWS_LOAD_SPLITS``): the
            router-rows load of each expert call in that many parts on the SP and ACT hardware queues, each part's
            latent copy (or with ``moe_rows_latent_fp8`` its FP8 expand) right after it (``latent_moe_decode_2srv``
            ``rows_load_split``). The values do not change.
        moe_routed_own_rows: False (default) keeps the program. True (``moe_routed_reduce`` ``ar``): each rank loads
            only its replica's rows of the routed all-reduce, one scalar-offset DMA of 1 / dp of the bytes with no
            copies (``latent_moe_decode_2srv`` ``routed_own_rows``). The values do not change.
        moe_latent_gather_mx: False (default) keeps the program. True ports the ``latent_gather_mx`` mechanism to the
            2-server body (``tkg_moe_2srv`` ``latent_gather_mx``): TP rank t < 28 computes its 128-column latent
            block, quantizes it with quantize_mx and sends the MXFP8 codes and the E8M0 scales in the router rows (89
            words, 61 with ``moe_router_rows_drop_selection``), so the expert calls take the latent in the MX layout and
            run no latent expand or quantize. Needs ``moe_gather`` ``dp_tp``, whole replicas in each 128-token tile of
            an expert call (the token-tiled calls of the k7 and k3b8 forms too) and no
            ``moe_rows_latent_fp8``; it sets ``moe_rows_load_split`` 1 (``served_defaults``). ``moe_in_proj`` has the
            540-column block layout. Against the BF16
            rows the expert input is byte equal; against ``moe_rows_latent_fp8`` it is accuracy class.
        moe_router_rows_drop_selection: the 84-word router rows of the MoE gathers, as in
            ``kimi_k3_decode_model_2srv``; it needs the 1,036-column ``moe_vectors``. The values do not change.
        moe_out_proj_form: the out projection of the MoE layers, as in ``kimi_k3_decode_model_2srv``
            (``tkg_moe_2srv.OUT_PROJ_FORMS``): ``reduced`` keeps the program (None: the default is that of
            ``served_defaults``, ``reduced``); ``column`` computes 224 hidden columns of the replica's rows on each TP
            rank from all 9,728 inputs (one FP32 accumulation, one BF16 rounding) and runs a TP32 all-gather of the
            columns (and one of the shared-expert activation, in the flight of the routed reduction) in place of the
            TP32 output all-reduce. It needs the BF16 ``moe_out_proj`` in the column layout (no
            ``moe_out_proj_scale``) and ``moe_vectors`` with
            the 3,584 latent norm weights after its columns (``tkg_moe_2srv.column_norm_cols``). With
            ``moe_routed_reduce`` ``rs_ag`` the norm reads the replica's rows from the routed TP32 all-gather (not with
            ``moe_routed_stats_ag``); not with ``rs_fold``. With ``moe_latent_gather_mx`` the expert input comes from
            the MXFP8 router rows and the column stage after the expert calls is the same. The values change (the
            output is not summed from BF16 partials): an accuracy-class option.
        pair_shared_cc_dsts: the collective destinations in LNC1 shared HBM, as in ``kimi_k3_decode_model_2srv``:
            ``moe_out`` the TP32 output all-reduce of the MoE layers and the world output all-reduce of the dense
            layer (``tkg_moe_2srv.moe_2srv_collective_hbm`` ``out_dst``, ``tkg_dense_mlp_2srv.dense_collective_hbm``),
            ``moe_routed`` the world routed all-reduce (``routed_dst``; ``moe_routed_reduce`` ``ar`` only) and ``kda_ar``
            the destinations of the two persistent KDA all-reduce pairs, which alternate by KDA layer. ``moe_gather`` is
            refused: the gathers of the multi-server MoE body stay private. ``moe_out_proj_form`` ``column`` refuses
            ``moe_out`` and ``moe_routed`` (``tkg_moe_2srv.COLUMN_PRIVATE_SITES``), and a split call refuses every
            site. The sources stay private and the values do not change. It needs the neuronx-cc option
            --lnc1-with-shared-mem and a runtime with pair-shared HBM. () (default) keeps the program.
        moe_rows_dst_pingpong: with ``cc_source_pingpong`` and ``moe_gather`` ``dp_tp``, the odd MoE layers' DP gather of
            the router rows writes a second persistent destination, as in ``kimi_k3_decode_model_2srv``. True (default):
            the gathers skip their start handshakes. The values do not change; False keeps one destination.
        moe_glue: {} keeps the program (None: the default is that of ``served_defaults``), or a dict of the glue options of the MoE body
            (``tkg_moe_2srv.GLUE_KEYS``: ``norm_tail``, ``rows_queue``, ``router_first``, ``out_proj_early``,
            ``selectors_once``, ``routed_compact``, ``store_queues``). The values do not change. ``mx_scores_align``
            also sizes the collective router-row tensors of the entry (65 words in place of 61).
        moe_sums_after_routed: the row sums of each layer (its AttnRes bank sums and norms) follow the first collective
            of the routed reduction of the previous MoE layer in the schedule (``decode_block.decode_block_body``
            ``moe_sums_after_routed``), so the Vector engine copies the routed partials out of PSUM before it runs the
            next layer's bank sums. Schedule edges only, so the values do not change. None: the default is that of
            ``served_defaults``; False keeps the program.
        attn_res_sums_after_moe_output: the AttnRes bank sums of the layer after each MoE layer, of the first layer of
            each later block and of the final mix follow the store of the routed partial into the source of the routed
            collective in the schedule (``decode_block.decode_block_body`` ``row_sums_after_ffn_output``), so VectorE
            runs the PSUM copies of the column-group partial before them. A schedule edge, not a semaphore wait: the
            values do not change. None: the default is that of ``served_defaults``; False keeps the program.
        cc_source_pingpong: two sets of the attention all-gather sources and of the MoE row and routed sources.
        mix_cache: must be False: the verify keeps the plain AttnRes mix, and its HBM bank push waits only for a
            plain mix.
        attn_res_cached_bank_sums: keep the rstd of each bank entry from its push, so the AttnRes mixes sum no squares
            of the bank (``decode_block.decode_block_body``): the bits of the plain mix. The one-call verify only (the
            split calls pass no bank rstd). None: the default is that of ``served_defaults``.
        attn_res_early_sums: with ``attn_res_cached_bank_sums``, each mix sums the stream squares and takes their rsqrt
            first (ScalarE loads the exponential table beside the VectorE sums), and the bank sums of the attention mix
            (and of the MLP mix of a KDA layer) stream the DRAM bank early, in the flight of the previous collective
            (``decode_block.decode_block_body`` ``attn_res_early_sums``). The same instructions, so the values do not
            change. None: the default is that of ``served_defaults``.
        attn_res_mla_mlp_early: with ``attn_res_early_sums``, the k1 MLA call on the loop path in the rows form
            (``mla_k1_rows``) or with the BF16-o state reduce-scatter (``mla_state_bf16_o``) traces the bank sums of
            the MLP mix of its layer in the flight of its last state reduce-scatter (``decode_block.decode_block_body``
            ``attn_res_mla_mlp_early``). The same instructions, so the values do not change. None: the default is that
            of ``served_defaults`` (the k3 rows form and ``MLA_MLP_EARLY_STATE_RS_FORM_SET``).
        attn_res_ahead: the attention mix of each KDA layer takes the ahead form (``decode_block.decode_block_body``
            ``attn_res_ahead``): its bank part is built in the tail of the layer before, and the mix and its RMSNorm run
            in one pass over the stream. Accuracy class (other FP32 rounding); needs ``attn_res_cached_bank_sums`` and
            not the ``mixture`` aux tap. False (default) keeps the program.
        attn_res_ahead_mlp: the same form for the MLP mix of each KDA layer (``decode_block.decode_block_body``
            ``attn_res_ahead_mlp``; its bank part after the attention norm, no GpSimd in the mix). Accuracy class;
            needs ``attn_res_cached_bank_sums``. False (default) keeps the program.
        attn_res_ahead_min_entries: with the ahead keys, the fewest bank entries of a mix that takes the ahead form
            (``decode_block.decode_block_body`` ``attn_res_ahead_min_entries``); 0 (default): every mix of the keys.
        early_ffn_gate: start the weights of each attention during the MoE tail (``decode_block.decode_block_body``).
        ffn_weights_after_attn: the weight loads of each feed-forward body start after the residual add that reads the
            attention output, not during the flight of the attention output collective (``decode_block.decode_block_body``;
            with ``dma_gates`` on), as in ``kimi_k3_verify_model``. The values do not change. None: the default is that of ``served_defaults``.
        kda_replay, kda_state_slot_page, kda_write_state_slots, kda_prework_ahead, kda_chain: as in
            ``kimi_k3_verify_model``. The ``states`` slot-state ring holds 2 tiles for the ``sequential``
            ``kda_chain`` and Q tiles for ``batched`` (token j writes tile j; the stores and the next call's writes are
            ordered by its edges). ``chunked`` writes the Q slot states from per-call tiles that a later instruction can
            reuse before the slot stores finish (no ring), so it is not for serving (default-off).
        kda_slot_state_depth: tiles of the ``states`` ``sequential`` slot-state ring (``verify_kda.kda_verify_attention``
            ``slot_state_depth``): 2 (default) alternates them, 1 is the in-place fallback. With ``kda_token_tiles`` it
            takes up to Q tiles (token j waits only for the stores of token j - depth). The bits do not change. None:
            the default is that of ``served_defaults`` (``KDA_SLOT_STATE_DEPTH_BY_FORM``, else 2).
        kda_slot_state_per_token: ``states`` with the ``sequential`` ``kda_chain`` and ``kda_token_tiles``: a ring of Q
            tiles in place of ``kda_slot_state_depth`` (token j writes tile j), so no token's state update waits for the
            slot stores of the token two before; only the previous call's stores of the tile order the write. The
            stores leave the token chain. The bits do not change. None: the default is that of ``served_defaults``
            (off).
        kda_slot_store_engine: ``states``: the queue of the slot-state stores of every KDA verify call
            (``verify_model.verify_bodies`` ``kda_slot_store_engine``): "sync", "gpsimd" (the GpSimd software DGE
            queue, off the in-order Sync queue that also carries the next MoE layer's first loads) or "split" (half on
            each). The bits do not change. None: the default is that of ``served_defaults`` ("sync").
        kda_shared_state_ring: ``states`` with the ``sequential`` ``kda_chain`` and the 2-tile ring: the KDA verify
            calls without ``kda_token_tiles`` (more than one head a rank) cycle the slot-state ring in place of one
            state tile updated in place, so the update of a token does not wait for the slot stores of the token
            before (``verify_model.verify_bodies`` ``kda_shared_state_ring``). The bits do not change; False (default)
            keeps the program.
        kda_state_dtype: the form of the KDA recurrent state in the pages (``tkg_kda.KDA_STATE_DTYPES``), as the
            2-server decode's ``kda_state_dtype``: ``bf16`` (default) keeps the program; ``fp32`` and ``fp32_hilo``
            read, chain and write the fp32 state (``verify_model.verify_bodies`` ``kda_state_dtype``; the slot-state
            ring takes the same dtype), so the verify stays bitwise equal to the decode steps of the same form. The pool
            of every KDA layer must be at least ``kda_page_offsets(Hl, kda_state_dtype)["end"]`` bf16 elements wide
            (101760 at 3 local heads: the 512-row pages ``[P, 147456]``) and of even width; the entry refuses a
            narrower pool. Not with ``kda_chain`` ``chunked``.
        kda_paired_pages: with an fp32 ``kda_state_dtype``, the KDA page ids name the even page of an aligned pair of
            the 256-row pools (``tkg_kda.kda_page_span``): each KDA page is the pair, with the bytes of the 512-row
            form at id / 2, and the MLA pages keep their rows. The pools need 2 x width >= the fp32 page end (101760 at
            3 local heads). The valid page ids are the EVEN ids in [1, P - 1) (``tkg_kda.kda_page_bound``), so
            id + 1 < P: an odd id or an id >= P - 1 is off. Bitwise vs the 512-row form of the KDA pages. False
            (default) keeps the program.
        kda_w_in_prefetch, kda_w_in_prefetch_chunks: as in ``kimi_k3_verify_model``: ``after_ffn_cc`` issues each KDA
            layer's ``kda_w_in`` ring (and the MXFP8 scales) after the first output collective of the feed-forward body
            before it, at DMA QoS level 3 (``decode_block.trn3_kda_w_in_prefetch``). ``off`` (default) keeps the
            program. The values do not change.
        kda_w_in_buffers: SBUF buffers of the ``kda_w_in`` ring of every KDA verify layer, as in
            ``kimi_k3_verify_model`` (and the 2-server decode's ``kda_w_in_buffers``). The values do not change; 0
            keeps the program (None: the default is that of ``served_defaults``).
        kda_w_in_scale: None (default, a BF16 ``kda_w_in``), or the scales of an MXFP8 ``kda_w_in`` stack, as in
            ``kimi_k3_verify_model`` (the same stacking as ``kda_w_in``): the KDA verify input projection of every
            KDA layer runs on the MXFP8 weights, through the same ``kda_w_in_buffers`` ring (an MX buffer holds the
            512 rows of a BF16 buffer in half the bytes).
        mla_in_proj_scale: None (default, a BF16 ``mla_in_proj``), or the scales of an MXFP8 ``mla_in_proj`` stack, as
            in ``kimi_k3_verify_model`` (the same stacking as ``mla_in_proj``): the MLA q_a / kv_a and gate projections of every
            MLA layer run on the MXFP8 weights.
        router_dtype, moe_router_weight: the FP16 router of the MoE layers, as in ``kimi_k3_decode_model_2srv``.
            ``bf16`` (default) keeps the program.
        moe_in_proj_scale: None (default, a BF16 ``moe_in_proj``), or the scales of an MXFP8 MoE in-projection,
            stacked as ``moe_in_proj`` (``[n_moe, 128, 2, 496]``; ``tkg_moe_2srv.MX_IN_PROJ_SHAPES``):
            ``moe_in_proj`` then stacks the MXFP8 in-projection (``[n_moe, 14, 128, 1984]`` uint8, no router columns),
            and the latent down and shared gate / up of every MoE layer run MXFP8 x MXFP8 (``latent_moe_decode_2srv``
            ``in_proj_scale``). It needs the FP16 router.
        kda_w_out_scale: None (default, a BF16 ``kda_w_out``), or the scales of an MXFP8 padded-lane ``kda_w_out`` stack
            (``tkg_kda.kda_w_out_form``: ``kda_w_out`` then stacks ``[n, 1, 128, 4 H]`` uint8 and the scales
            ``[n, 128, 1, H]``): the KDA output projection of every KDA layer runs MXFP8 x MXFP8. The values change.
        moe_out_proj_scale: None (default, a BF16 ``moe_out_proj``), or the scales of an MXFP8 padded-lane MoE
            out-projection, stacked as ``moe_out_proj`` (``[n_moe, 128, 1, H]``;
            ``tkg_moe_2srv.MX_OUT_PROJ_SHAPES``): ``moe_out_proj`` then stacks ``[n_moe, 1, 128, 4 H]`` uint8 (the
            304 rows in the lanes of 512, the rows past them zero), and the out-projection of every MoE layer runs
            MXFP8 x MXFP8 (``latent_moe_decode_2srv`` ``out_proj_scale``). The values change.
        kda_state_loads_first, kda_slot_pages_first: ``kda_replay`` ``states`` only: trace the state loads, or the load
            of the slot pages, of every KDA verify layer before its input projection (``verify_model.verify_bodies``).
            Not with ``kda_prework_ahead``. The values do not change. None: the default is that of ``served_defaults``.
        kda_token_tiles, kda_slot_store_queues: ``kda_replay`` ``states`` only: the ``kda_token_tiles`` and
            ``kda_slot_store_queues`` of ``verify_model.verify_bodies``: each scored token of every KDA layer takes its
            own state tiles, so no token waits for the slot DMAs of the token before, and 2 alternates the per-token
            slot state stores between the Sync and the ScalarE queues. The values do not change. False and 1 keep the
            program. None: the default is that of ``served_defaults``.
        attn_res_weight_rows: the weight row of each AttnRes score weight in the mixes and bank sums
            (``decode_block.decode_block_body`` ``attn_res_weight_rows``). The same bits. None: the default is that of
            ``served_defaults`` (``ATTN_RES_WEIGHT_ROW_FORMS``).
        kda_states_after_out_ar: the state loads of each KDA layer after a MoE layer of the same decode block follow
            that layer's output all-reduce trigger (``decode_block.decode_block_body`` ``kda_states_after_out_ar``), so
            the MoE out store is not behind them on the SP queue; the first KDA layer of each block keeps its program.
            The values do not change. None or False (the ``served_defaults`` default) keeps the program. Not with
            ``kda_prework_ahead`` (the KDA call refuses the pair).
        kda_merged_state_loads: the KDA layers load each sequence's state and conv window once, from its selected page
            (``verify_kda.kda_verify_attention`` ``merged_state_loads``), with ``kda_replay`` "states". The same bits.
            None: the default is that of ``served_defaults`` (``KDA_MERGED_STATE_LOADS_FORM_SET``).
        kda_x_quant_hook: the KDA layers trace the bank sums of their MLP mix right after the quantize of the MXFP8
            input projection (``verify_model.verify_bodies`` ``kda_x_quant_hook``), so they run beside its matmuls. It
            acts only with the early sums (``attn_res_early_sums``). The same bits. None: the default is that of
            ``served_defaults`` (``KDA_X_QUANT_HOOK_FORM_SET``, by the served shape).
        aux_row_ring: the aux row stores take a ring of ``AUX_RING_DEPTH`` staging tiles on the region
            (``common.store_hT_rows`` ``ring``), so the stores issue back to back. The same bits. None: the default is
            that of ``served_defaults`` (the served k7 form).
        rows_hT_batched: load the stream rows into the hT stream tile of the first block (here with
            ``attn_res_fence_in``, else in ``decode_block.decode_block_body``) in column chunks
            (``common.load_rows_hT_batched``: one DMA a chunk on the SP / ACT queues in turn, then the TensorE
            transposes) in place of one DMA and one transpose a row (``common.load_rows_hT``). The values do not
            change. None: the default is that of ``served_defaults`` (False).
        kda_conv_load_gather, kda_batched_conv_stores, kda_null_page: as in ``verify_model.kimi_k3_verify_model``: the
            committed conv state of each KDA verify layer loaded with one gather DMA, and written with one scatter DMA,
            per group of sequences (``verify_kda.kda_verify_attention`` ``conv_load_gather`` / ``batched_conv_stores``),
            and the KDA null page of each KDA verify call, whose rows read and write as off rows (``null_page``). The
            values and the live pages do not change. False / 0 keeps the program. None: the default is that of
            ``served_defaults`` (both on in the ``KDA_SLOT_DMA_FORMS`` forms with the state slots; 0 for the null page).
        kda_slot_pages_contiguous: as in ``verify_model.kimi_k3_verify_model``: ``kda_replay`` ``states``, each KDA
            verify layer loads its slot ids in their HBM order (one contiguous run per partition) and transposes them in
            the VectorE cast, so the slot-page load does not hold the Sync DMA queue (ahead of the MoE routed store)
            with 4-byte packets. The bits do not change; False keeps the program. None: the
            default is that of ``served_defaults`` (on in the ``KDA_SLOT_DMA_FORMS`` forms with the state slots, off
            with ``kda_paired_pages``, whose slot ids the KDA verify takes as they are).
        kda_batched_slot_conv_stores: as in ``verify_model.kimi_k3_verify_model``: ``kda_replay`` ``states``, each KDA
            verify layer writes the slot conv windows of its scored tokens with one scatter DMA per group of rows on the
            Pool queue, not one serial Sync DMA per row. The rows of every slot page do not change; an off slot's page
            0 (or the null page) gets zeros. False keeps the program. None: the default is that of
            ``served_defaults`` (on in the ``KDA_SLOT_DMA_FORMS`` forms with the state slots).
        kda_bf16_update_rows: as in ``verify_model.kimi_k3_verify_model``: with the ``decoupled_bf16`` readout, each KDA
            verify layer's state updates take their bf16 rows from one transpose per 4 sequences and one broadcast
            matmul per head. The bits do not change; False keeps the program. None: the default is that of
            ``served_defaults`` (on in the ``KDA_SLOT_DMA_FORMS`` forms with the state slots).
        kda_slot_conv_persistent: ``kda_replay`` ``states`` with the sequential chain and the explicit SBUF region: the
            slot conv-state stores of every KDA verify layer take persistent SBUF source tiles, one set for the whole
            kernel (``verify_kda.kda_slot_conv_rows_tiles``), with their cross-layer waits, so no tile of a token chain
            takes their bytes and waits for the stores (nki-dma-source-liveness). The bits do not change; False keeps
            the program. None: the default is that of ``served_defaults`` (on in the ``KDA_SLOT_DMA_FORMS`` forms with
            the state slots and the per-token tiles of the sequential chain).
        kda_update_bcast_depth: the PSUM tiles of the state-update broadcast ring of every KDA verify token, 2 to 4, each
            in its own PSUM bank (``verify_kda.kda_verify_attention`` ``update_bcast_depth``). The bits do not change; 2
            keeps the program. None (default): that of ``served_defaults`` (``KDA_UPDATE_BCAST_DEPTH_BY_FORM``: 4 in
            the k7 form with the state slots, else 2).
        kda_head_pipeline: the order of the MXFP8 input projection's output blocks in every KDA verify call
            (``verify_kda.kda_verify_attention`` ``head_pipeline``): ``off`` keeps the program; ``g_last`` runs the
            output-gate blocks after the decay gate. The same bits. None (default): that of ``served_defaults``
            (``KDA_HEAD_PIPELINE_BY_FORM``, empty: ``off``) with the MXFP8 ``kda_w_in``, and ``off`` with a BF16 one;
            a served ``heads`` needs ``kda_w_in_head_major`` too, else it is ``off``. A given ``heads`` needs
            ``kda_w_in_head_major`` and ``kda_w_in_prefetch`` off (trace-time refusals).
        kda_w_in_head_major: the caller's ``kda_w_in`` / ``kda_w_in_scale`` stacks are in the head-major column order
            (``kda_decode_torch.kda_w_in_head_major``), which ``kda_head_pipeline`` ``heads`` reads. False (default): the
            original order. The two go together (a trace-time refusal on a mismatch).
        kda_out_ar_rows: ACCURACY CLASS (<= 1 BF16 ulp). The KDA output all-reduce of every KDA verify call runs as two
            row chunks (``verify_kda.kda_verify_attention`` ``out_ar_rows``). None (default): that of ``served_defaults``
            (``KDA_OUT_AR_ROWS_FORM_SET``, empty).
        kda_slot_conv_from_inputs: ``kda_replay`` ``states`` with the sequential chain: the slot conv-state stores of
            every KDA verify layer read their windows from the conv inputs of the scored tokens, with no GpSimd copies
            of a window tile (``verify_kda.kda_verify_attention`` ``slot_conv_from_inputs``). The bits do not change;
            False keeps the program. None (default): that of ``served_defaults`` (on in the
            ``KDA_SLOT_CONV_FROM_INPUTS_FORM_SET`` forms with the state slots, the per-token state tiles and the
            sequential chain).
        kda_conv_products_engine: the engine of the one-pass conv tap products of every KDA verify call, ``gpsimd`` or
            ``vector`` (``verify_kda.kda_verify_attention`` ``conv_products_engine``). The bits of finite values do not
            change; ``gpsimd`` keeps the program. None (default): the form's ``KDA_CONV_PRODUCTS_ENGINE_BY_FORM`` value
            (``served_defaults``), ``gpsimd`` for a form not in the table.
        kda_recurrence_group: the sequences of each delta-rule group of a token in every KDA verify call, each group
            with its own PSUM tiles (``verify_kda.kda_verify_attention`` ``recurrence_group``). The bits do not change;
            None (default) keeps the program.
        kda_page_groups: None (default), or the KV cache group of each KDA layer of the call, in layer order: a tuple
            of ints that numbers the groups 0 .. G - 1 (G at most ``verify_kda.KDA_PAGE_GROUPS_MAX``). The serving
            side gives one block table to each KV cache group, so the KDA layers of a group have the same state and
            slot pages; the caller checks that. Each group then keeps one page set on the SBUF heap
            (``verify_kda.kda_page_group_tiles``): its first KDA layer computes the page metadata (the DMA pages and
            masks, the slot write pages, the conv gather and scatter rows) and its later layers read it. ``states``
            on the explicit SBUF region (the loop path), with the inline prework. The values do not change; None
            keeps the program.
        moe_half_weight_buffers, moe_routed_store_form, world_size, routed_tp, moe_form, dp_size, moe_gather,
            moe_wide_half_weight_buffers, moe_expert_call_tokens, moe_mx_tile, routed_layers, moe_routed_dynamic,
            routed_alt0_expert_gate_up_weights .. routed_alt1_expert_down_weights_scale: as in
            ``kimi_k3_decode_model_2srv``, on the T gathered tokens of the verify. The values do not change.
        max_verify_rows: None (default) keeps the bound ``MAX_VERIFY_ROWS`` on the rows (Q x B) of a replica; a value up to
            ``SIZING_VERIFY_ROWS`` raises it for a sizing call (the MoE at twice the tokens). Not a served form: its
            attention fit is not measured. None keeps the program.
        moe_weight_buffers, moe_scale_buffers, moe_transpose_lookahead: the whole weight buffers (1 to 3), the gate/up
            scale buffers (1 or 2) and the activation chunks transposed ahead (1 or 2) of the static expert loop, as in
            ``kimi_k3_decode_model_2srv`` (``decode_block.decode_block_body``). The values do not change; None
            (default) keeps the program.
        moe_cross_expert_order: the gate matmuls of the next expert run between the up and the down matmuls of each
            expert of the static expert loop, as in ``kimi_k3_decode_model_2srv`` (needs moe_weight_buffers 3). The
            values do not change. None: the default is that of ``served_defaults`` (it then takes moe_weight_buffers 3).
        moe_transpose_block: with the cross-expert order, the activation transposes of each expert run in one block
            with no MX matmul inside, as in ``kimi_k3_decode_model_2srv`` (``transpose_block`` of ``moe_tkg_decode``).
            The values do not change. None: the default is that of ``served_defaults`` (on with the cross-expert order).
        moe_static_alt_tiles: the static expert loop (its calls of 128 tokens and its token-tiled calls) takes its
            gate sigmoid / up tanh tiles (and with "act_copy" or True its PSUM copies, with True its evictions) from
            two buffers each, as in ``kimi_k3_decode_model_2srv`` (``static_alt_tiles`` of ``moe_tkg_decode``,
            ``STATIC_ALT_TILE_FORMS``). The values do not change. None: the default is that of ``served_defaults``
            (``MOE_STATIC_ALT_TILE_DEFAULTS``: True in the k7 form without the glue's ``routed_compact``, else off).
        norm_scalar_squares, attn_res_scalar_products, attn_res_bank_dma: the AttnRes engine and queue options of
            ``decode_block.decode_block_body`` on the R rows of the replica and its DRAM bank: the sums of squares of
            the last rows of each norm after a mix, and the products of the last rows of each bank entry after the
            first, on ScalarE; the queue of the bank-entry loads. Each row keeps its bits. 0 / 0 / None keep the
            program; None / None / ``auto`` (the defaults): the default is that of ``served_defaults``.
        kda_out_ar_sbuf: on the loop path (a region), the output all-reduce of every KDA layer but the call's first runs
            SBUF -> SBUF from one persistent region-heap source tile, held to the call end, into the result tile
            (``verify_kda.kda_verify_attention`` ``ar_sbuf``), with no HBM source store and no destination load. The
            Neuron runtime takes the latency-optimized all-reduce for it, with the reduce order of the HBM form for the
            (128, rows, H/128) tile, so the values do not change. The call's first TP collective skips its handshake,
            so a peer can write its destination early: that layer keeps the HBM pair, whose destination the call holds
            to its end. None: the default of ``served_defaults`` (off).
        kda_small_after_out_ar: the GpSimd weight loads of every KDA layer (``kda_w_small``, ``kda_params``,
            ``kda_w_out``) wait for the load of the result of the last output collective of the feed-forward body before
            it (``decode_block.decode_block_body`` ``kda_small_after_out_ar``), so they do not stream in that
            collective's flight. The bytes and the instructions do not change, so the values do not change. None: the
            default of ``served_defaults`` (``KDA_SMALL_AFTER_OUT_AR_FORM_SET``, empty: off); True turns it on, False
            keeps it off.
        attn_res_mlp_early_entries: N > 0: the MLP AttnRes mix of each KDA layer sums its first N bank entries during
            the KDA body, on the DRAM bank (``decode_block.decode_block_body`` ``attn_res_kda_mlp_early_entries`` with
            ``attn_res_kda_mlp_early_hbm``, as ``kimi_k3_verify_model`` ``attn_res_mlp_early_entries``). Not with
            ``attn_res_cached_bank_sums`` or ``attn_res_early_sums``. The values do not change. 0 keeps the program
            (None: the default is that of ``served_defaults``).
        attn_res_bank_in, return_attn_res_bank, attn_res_fence_in: the split verify of ``kimi_k3_verify_model``: the
            bank of the layers before ``first_layer`` and the fence of the call before, or the bank and the boundary
            fence (over ``world_group``) for the next call. The range rules are those of ``kimi_k3_verify_model``.

        mla_split, mla_split_info, mla_split_table: the long-sequence split across the DP replica pairs (r, r xor 1)
            of a server (``mla_block_attention`` ``split``). ``mla_split`` None (default; the program does not change)
            or ``{"S": 2}``. ``mla_split_info`` ``[1, 4]`` INT32: (role, X, donor slot, tag) of this replica (role 0
            none, 1 owner, 2 helper; X a multiple of 16384; the donor slot must be B - 1; a tag below 2**24).
            ``mla_split_table`` ``[n_mla, 1, NB]`` INT32: the helper's block table row of the partner's donor sequence
            (its copy of the pages of positions [0, X)). At the start of the call one pair all-gather of the infos
            gives each rank its effective role: owner only if the partner says (2, X, tag) back, helper only if it
            says (1, X, tag), else 0 (the call without the split, bitwise). The MLA layers then run the k1 call on
            the extended batch B + 2 (the foreign and a pad slot) with the k7 MLA form of Q (B + 2) rows. v1: B = 2,
            Q = 8, the k1 packed loop path.
        mla_qkv_parity: on the loop path of the k1 MLA call (a region), the qkv all-gather of the MLA layers takes
            two persistent pairs of region-heap tiles, which alternate by MLA layer (``verify_model.verify_bodies``
            ``mla_calls``). False (default) keeps one pair for every MLA layer. The values do
            not change. Test only: a released runtime can give wrong values when one persistent SBUF tile is the
            source of many collectives in one execution (the SBUF KDA output all-reduce, ``kda_out_ar_sbuf``).
    Returns:
        The stream ``[R, H]`` (the ``lm_head`` input with the final weights), then the tokens ``[R, 1]`` int32
        (``greedy``), the logits ``[R, V/32]`` (``return_logits``), the aux states ``[len(aux_layers), R, H]``
        (``return_aux``), the per-layer streams ``[L, R, H]`` (``debug``), the AttnRes bank and the boundary fence
        (``return_attn_res_bank``), the grammar state value of each row ``[B, Q]`` INT32 (with the grammar tensors),
        and the page pools; with ``mla_split`` v1, last, the split aux ``[1, 4]`` INT32
        (a mismatch this call 0 / 1, the effective role, X, the partner's role); with ``mla_split`` v5, last, the
        gathered stats ``[dp, S]`` INT32 (row r = DP replica r's ``mla_split_stats``), then the split aux ``[1, 4]``
        INT32 (unconfirmed splits, confirmed splits, active pieces, 0; zeros in the idle program).

    The v5 split (``mla_split`` ``{"foreign_slots": F, "stats_words": S, "program": "idle" | "split"}``): the call
    starts at layer 0, and layer 0's DP x gather also gathers this replica's ``mla_split_stats`` ``[1, S]`` INT32 (an
    opaque block; one coalesced all-gather). The idle program does nothing else.
    """
    restart_calls()
    rows, H = stream.shape
    mla_k1_rows, mla_k1_packed, mla_k1_rows_options = mx_qk_keys(mx_qk, mla_k1_rows, mla_k1_packed, mla_k1_rows_options)
    # The served form of the call by its shape, for the body's form tables. The k3 forms take these tables on the k1
    # loop path only (as served_defaults); the other forms keep the shape rule.
    body_form = next(
        (name for name, value in SERVED_FORMS.items() if value == (form_world(world_size), queries, rows)), None
    )
    if body_form in K3_FORMS and not (mla_mode == "k1" and mla_loop_path):
        body_form = None
    # MOE_RING_QUEUE_ORDER_FORM_SET: the queue order of the KDA-led expert ring, by the served form of the call.
    ring_order = body_form in MOE_RING_QUEUE_ORDER_FORM_SET
    # MOE_RING_BEFORE_STORES_FORM_SET: the KDA-led expert ring before the slot state stores, by the served form.
    ring_before_stores = body_form in MOE_RING_BEFORE_STORES_FORM_SET
    # ATTN_RES_STT_FORM_SET: the VectorE multiply-adds of the AttnRes mix entries, by the served form of the call.
    mix_stt = body_form in ATTN_RES_STT_FORM_SET
    # AUX_STORE_FORM_SET: the queues and ring depth of the aux row stores, by the served form of the call.
    aux_fast = body_form in AUX_STORE_FORM_SET
    served = served_defaults(
        world_size,
        queries,
        rows,
        dict(
            mla_mode=mla_mode,
            mla_loop_path=mla_loop_path,
            split_call=attn_res_bank_in is not None or return_attn_res_bank,
            mix_cache=mix_cache,
            dma_gates=dma_gates,
            cc_source_pingpong=cc_source_pingpong,
            mla_k1_rows=mla_k1_rows,
            mla_k1_rows_options=mla_k1_rows_options,
            live_rows=live_rows is not None,
            mla_fp8_pages=mla_fp8_pages,
            mla_out_ar=mla_out_ar,
            mla_k1_packed=mla_k1_packed,
            mla_critical_dma_priority=mla_critical_dma_priority,
            mla_cache_write_off_sp=mla_cache_write_off_sp,
            mla_gemv_bands=mla_gemv_bands,
            mla_batched_cache_write=mla_batched_cache_write,
            mla_p2_entry_act=mla_p2_entry_act,
            mla_step_items=mla_step_items,
            mla_pass_q_prefetch=mla_pass_q_prefetch,
            mla_q_gather_buffers=mla_q_gather_buffers,
            mla_tail_weights_after_rs=mla_tail_weights_after_rs,
            mla_pass_setup_off_pool=mla_pass_setup_off_pool,
            mla_slot_rows_sbuf=mla_slot_rows_sbuf,
            mla_skip_stage_store=mla_skip_stage_store,
            mla_pass_rows_ahead=mla_pass_rows_ahead,
            mla_null_rows_once=mla_null_rows_once,
            mla_k1_pack_options=mla_k1_pack_options,
            mx_qk=mx_qk,
            ffn_weights_after_attn=ffn_weights_after_attn,
            norm_scalar_squares=norm_scalar_squares,
            attn_res_scalar_products=attn_res_scalar_products,
            attn_res_bank_dma=attn_res_bank_dma,
            attn_res_cached_bank_sums=attn_res_cached_bank_sums,
            attn_res_early_sums=attn_res_early_sums,
            attn_res_mla_mlp_early=attn_res_mla_mlp_early,
            attn_res_mlp_early_entries=attn_res_mlp_early_entries,
            mla_state_bf16_o=mla_state_bf16_o,
            kda_replay=kda_replay,
            kda_prework_ahead=kda_prework_ahead,
            kda_w_in_buffers=kda_w_in_buffers,
            kda_w_in_mx=kda_w_in_scale is not None,
            kda_state_loads_first=kda_state_loads_first,
            kda_slot_pages_first=kda_slot_pages_first,
            kda_token_tiles=kda_token_tiles,
            kda_slot_store_queues=kda_slot_store_queues,
            kda_slot_state_per_token=kda_slot_state_per_token,
            kda_slot_store_engine=kda_slot_store_engine,
            attn_res_ahead=attn_res_ahead,
            attn_res_ahead_mlp=attn_res_ahead_mlp,
            kda_conv_products_engine=kda_conv_products_engine,
            kda_slot_state_depth=kda_slot_state_depth,
            kda_shared_state_ring=kda_shared_state_ring,
            rows_hT_batched=rows_hT_batched,
            attn_res_weight_rows=attn_res_weight_rows,
            kda_states_after_out_ar=kda_states_after_out_ar,
            aux_row_ring=aux_row_ring,
            kda_x_quant_hook=kda_x_quant_hook,
            kda_merged_state_loads=kda_merged_state_loads,
            kda_out_ar_sbuf=kda_out_ar_sbuf,
            kda_small_after_out_ar=kda_small_after_out_ar,
            kda_state_dtype=kda_state_dtype,
            kda_paired_pages=kda_paired_pages,
            kda_chain=kda_chain,
            kda_conv_load_gather=kda_conv_load_gather,
            kda_batched_conv_stores=kda_batched_conv_stores,
            kda_slot_pages_contiguous=kda_slot_pages_contiguous,
            kda_batched_slot_conv_stores=kda_batched_slot_conv_stores,
            kda_bf16_update_rows=kda_bf16_update_rows,
            kda_slot_conv_persistent=kda_slot_conv_persistent,
            kda_update_bcast_depth=kda_update_bcast_depth,
            kda_head_pipeline=kda_head_pipeline,
            kda_out_ar_rows=kda_out_ar_rows,
            kda_slot_conv_from_inputs=kda_slot_conv_from_inputs,
            moe_gather=moe_gather,
            moe_rows_latent_fp8=moe_rows_latent_fp8,
            moe_rows_load_split=moe_rows_load_split,
            moe_latent_gather_mx=moe_latent_gather_mx,
            moe_routed_own_rows=moe_routed_own_rows,
            moe_routed_reduce=moe_routed_reduce,
            moe_out_proj_form=moe_out_proj_form,
            moe_transpose_lookahead=moe_transpose_lookahead,
            moe_expert_queues=moe_expert_queues,
            moe_expert_prefetch=moe_expert_prefetch,
            moe_half_weight_buffers=moe_half_weight_buffers,
            moe_weight_buffers=moe_weight_buffers,
            moe_expert_call_tokens=moe_expert_call_tokens,
            moe_cross_expert_order=moe_cross_expert_order,
            moe_transpose_block=moe_transpose_block,
            moe_static_alt_tiles=moe_static_alt_tiles,
            moe_glue=moe_glue,
            moe_sums_after_routed=moe_sums_after_routed,
            attn_res_sums_after_moe_output=attn_res_sums_after_moe_output,
        ),
    )
    mla_out_ar, mla_k1_packed = served["mla_out_ar"], served["mla_k1_packed"]
    # K7_ONLY_LEVERS_OFF: every form but those of K7_LEVER_FORM_SET gets these levers off.
    k7_levers = body_form in K7_LEVER_FORM_SET
    # SUMS_RS_FIRST_FORM_SET forms keep the sums-first order of the k7 levers.
    k7_off = K7_ONLY_LEVERS_OFF
    if any((form_world(world_size), queries, rows) == SERVED_FORMS[name] for name in SUMS_RS_FIRST_FORM_SET):
        k7_off = {key: value for key, value in K7_ONLY_LEVERS_OFF.items() if key != "mla_sums_rs_first"}
    # PAD_SKIP_FORMS take the packed loop's pad skip with the runner's live rows: a pad sequence takes no key group,
    # and every live row keeps its bits. A caller's pad_skip wins.
    if (
        live_rows is not None
        and mla_k1_packed
        and any((form_world(world_size), queries, rows) == SERVED_FORMS[form] for form in PAD_SKIP_FORMS)
        and "pad_skip" not in (mla_k1_pack_options or {})
    ):
        mla_k1_pack_options = {**(mla_k1_pack_options or {}), "pad_skip": True}
    mla_k1_pack_given = mla_k1_pack_options
    mla_k1_pack_options = served_pack_options(
        world_size,
        queries,
        rows,
        mla_k1_packed,
        mla_k1_pack_options,
        mla_state_bf16_o,
        mx_qk,
        entry_act=bool(served["mla_p2_entry_act"]),
        step_items=bool(served["mla_step_items"]),
        setup_off_pool=bool(served["mla_pass_setup_off_pool"]),
        slot_rows_sbuf=bool(served["mla_slot_rows_sbuf"]),
        skip_stage_store=bool(served["mla_skip_stage_store"]),
        fp8_pv=mla_fp8_pv,
    )
    check_fp8_pv(mla_fp8_pv, mx_qk, mla_k1_pack_options)
    check_fp8_pv_split(
        mla_fp8_pv,
        {
            "mla_split": mla_split,
            "mla_split_info": mla_split_info,
            "mla_split_table": mla_split_table,
            "mla_split_stats": mla_split_stats,
            "mla_split_own": mla_split_own,
            "mla_split_foreign": mla_split_foreign,
        },
    )
    # Refuse only a value that the caller gives. On a call without the p2 loop (for example the mx_qk "RA" rows form)
    # the table default has no effect, because served_pack_options adds setup_off_pool to the p2 set only.
    kernel_assert(
        not mla_pass_setup_off_pool or (mla_k1_pack_options or {}).get("p2"),
        "mla_pass_setup_off_pool needs the packed p2 loop (the k7 forms on the k1 loop path). Turn it off for this call.",
    )
    # Refuse only a value that the caller gives. On a call without the p2 loop (for example the mx_qk "RA" rows form)
    # the table defaults have no effect, because served_pack_options adds the two options to the p2 set only, and the
    # skipped store only with the slot rows.
    kernel_assert(
        not mla_slot_rows_sbuf
        or ((mla_k1_pack_options or {}).get("p2") and (mla_k1_pack_options or {}).get("stage_rows_sbuf")),
        "mla_slot_rows_sbuf needs the packed p2 loop with stage_rows_sbuf (the k7 form on the k1 loop path). Turn it "
        "off for this call.",
    )
    kernel_assert(
        not mla_skip_stage_store or (served["mla_slot_rows_sbuf"] and (mla_k1_pack_options or {}).get("p2")),
        "mla_skip_stage_store needs mla_slot_rows_sbuf on the packed p2 loop, because a slot row that reads HBM "
        "needs the stage store. Turn on mla_slot_rows_sbuf, or turn off mla_skip_stage_store.",
    )
    mla_critical_dma_priority = served["mla_critical_dma_priority"]
    mla_cache_write_off_sp = served["mla_cache_write_off_sp"]
    mla_gemv_bands = served["mla_gemv_bands"]
    mla_batched_cache_write = served["mla_batched_cache_write"]
    mla_null_rows_once = served["mla_null_rows_once"]
    mla_pass_q_prefetch = served["mla_pass_q_prefetch"]
    mla_q_gather_buffers = served["mla_q_gather_buffers"]
    if (mla_k1_pack_options or {}).get("q_mx_gather") and (mla_pass_q_prefetch, mla_q_gather_buffers) != (0, 1):
        # q_mx_gather loads the MX query bytes of each pass in one set: there are no BF16 gather tiles to prefetch or to
        # double. A served render can still give the BF16 values, so the call takes 0 and 1.
        print(
            f"Kimi-K3 verify: q_mx_gather takes mla_pass_q_prefetch 0 and mla_q_gather_buffers 1 "
            f"(given {mla_pass_q_prefetch} and {mla_q_gather_buffers})",
            flush=True,
        )
        mla_pass_q_prefetch, mla_q_gather_buffers = 0, 1
    # The tail weights wait for the call-level BF16-o state reduce-scatter of the fit passes and take the MXFP8
    # mla_in_proj (mla_block_attention tail_weights_after_rs). Refuse only a value that the caller gives; the table
    # default takes such a call only and has no effect on any other call.
    tail_state_rs = (
        mla_mode == "k1"
        and mla_loop_path
        and rows > 16
        and mla_state_bf16_o
        and mla_in_proj_scale is not None
        and mla_block_table is not None
        and mla_block_table.shape[2] >= _K1_FIT_BLOCKS
    )
    kernel_assert(
        not mla_tail_weights_after_rs or tail_state_rs,
        "mla_tail_weights_after_rs needs the call-level BF16-o state reduce-scatter of the fit passes and the MXFP8 "
        f"mla_in_proj: mla_mode 'k1' on the loop path, more than 16 rows on a table of at least {_K1_FIT_BLOCKS} local "
        "blocks, mla_state_bf16_o and mla_in_proj_scale. Turn it off for this call.",
    )
    mla_tail_weights_after_rs = bool(served["mla_tail_weights_after_rs"]) and tail_state_rs
    mla_pass_rows_ahead = served["mla_pass_rows_ahead"]
    kernel_assert(
        not (mla_k1_pack_options or {}).get("q_mx_gather") or not mla_pass_rows_ahead,
        "mla_k1_pack_options q_mx_gather loads the MX query bytes of a later pass after the state stores of the pass "
        "before: it takes no mla_pass_rows_ahead. Turn it off for this call.",
    )
    mla_k1_rows, mla_k1_rows_options = served["mla_k1_rows"], served["mla_k1_rows_options"]
    check_mx_qk(
        world_size,
        queries,
        rows,
        mla_mode,
        mla_loop_path,
        mx_qk,
        mla_fp8_pages,
        mla_k1_rows,
        mla_k1_packed,
        mla_k1_pack_options,
    )
    ffn_weights_after_attn = served["ffn_weights_after_attn"]
    norm_scalar_squares, attn_res_scalar_products = served["norm_scalar_squares"], served["attn_res_scalar_products"]
    attn_res_bank_dma, attn_res_mlp_early_entries = served["attn_res_bank_dma"], served["attn_res_mlp_early_entries"]
    attn_res_cached_bank_sums, attn_res_early_sums = served["attn_res_cached_bank_sums"], served["attn_res_early_sums"]
    attn_res_mla_mlp_early = served["attn_res_mla_mlp_early"]
    attn_res_weight_rows = served["attn_res_weight_rows"]
    kda_states_after_out_ar = served["kda_states_after_out_ar"]
    kda_w_in_buffers = served["kda_w_in_buffers"]
    kda_state_loads_first, kda_slot_pages_first = served["kda_state_loads_first"], served["kda_slot_pages_first"]
    kda_token_tiles, kda_slot_store_queues = served["kda_token_tiles"], served["kda_slot_store_queues"]
    kda_slot_state_per_token, kda_slot_store_engine = (
        served["kda_slot_state_per_token"],
        served["kda_slot_store_engine"],
    )
    kda_slot_state_depth = served["kda_slot_state_depth"]
    kda_conv_products_engine = served["kda_conv_products_engine"]
    rows_hT_batched = served["rows_hT_batched"]
    aux_row_ring = served["aux_row_ring"]
    x_quant_hook = served["kda_x_quant_hook"]
    kda_merged_state_loads = served["kda_merged_state_loads"]
    kda_out_ar_sbuf = served["kda_out_ar_sbuf"]
    kda_small_after_out_ar = served["kda_small_after_out_ar"]
    kda_conv_load_gather, kda_batched_conv_stores = served["kda_conv_load_gather"], served["kda_batched_conv_stores"]
    kda_slot_pages_contiguous = served["kda_slot_pages_contiguous"]
    kda_batched_slot_conv_stores = served["kda_batched_slot_conv_stores"]
    kda_bf16_update_rows, kda_slot_conv_persistent = served["kda_bf16_update_rows"], served["kda_slot_conv_persistent"]
    kda_update_bcast_depth, kda_slot_conv_from_inputs = (
        served["kda_update_bcast_depth"],
        served["kda_slot_conv_from_inputs"],
    )
    # The served g_last form takes the MXFP8 input projection: with a BF16 kda_w_in the default is off (a given value
    # still meets the kernel's refusal).
    head_pipeline_given = kda_head_pipeline is not None
    kda_head_pipeline = served["kda_head_pipeline"] if head_pipeline_given or kda_w_in_scale is not None else "off"
    # A served "heads" needs the head-major weights; a given "heads" meets the refusals below.
    if not head_pipeline_given and kda_head_pipeline == "heads" and not kda_w_in_head_major:
        kda_head_pipeline = "off"
    kernel_assert(
        (kda_head_pipeline == "heads") == bool(kda_w_in_head_major),
        f"kda_w_in_head_major ({kda_w_in_head_major}) and kda_head_pipeline heads ({kda_head_pipeline!r}) go together: "
        "the heads form reads the head-major kda_w_in order, and no other form does",
    )
    kernel_assert(
        kda_head_pipeline != "heads" or kda_w_in_prefetch == "off",
        f"kda_head_pipeline heads loads the w_in parts itself: kda_w_in_prefetch must be off, got {kda_w_in_prefetch!r}",
    )
    kda_out_ar_rows = served["kda_out_ar_rows"]
    moe_rows_load_split, moe_routed_reduce = served["moe_rows_load_split"], served["moe_routed_reduce"]
    moe_out_proj_form = served["moe_out_proj_form"]
    moe_cross_expert_order, moe_weight_buffers = served["moe_cross_expert_order"], served["moe_weight_buffers"]
    moe_glue, moe_expert_prefetch = served["moe_glue"], served["moe_expert_prefetch"]
    # The moe_glue key kda_prework_ahead (KDA_PREWORK_AHEAD_FORM_SET): the KDA prework of each layer ahead, with the
    # state-slot levers of a form of the set. The prework then takes the state and slot-page loads, so the load-order
    # keys of the inline loads drop, and the body runs the prework's selects (deferred).
    prework_glue = (moe_glue or {}).get("kda_prework_ahead", False)
    kernel_assert(isinstance(prework_glue, bool), f"moe_glue kda_prework_ahead must be a bool, got {prework_glue!r}")
    prework_composed = (
        body_form in KDA_PREWORK_AHEAD_FORM_SET and kda_replay == "states" and (prework_glue or kda_prework_ahead)
    )
    kda_prework_ahead = bool(kda_prework_ahead or prework_glue)
    kernel_assert(
        not prework_composed or not (attn_res_bank_in is not None or return_attn_res_bank),
        "moe_glue kda_prework_ahead composes with the one-call states verify: not with a split call",
    )
    if prework_composed:
        kda_state_loads_first = kda_slot_pages_first = False
    ahead_sums_vector = (moe_glue or {}).get("ahead_sums_vector", "off")
    ahead_load_tiles = (moe_glue or {}).get("ahead_load_tiles", 2)
    kernel_assert(
        isinstance(ahead_load_tiles, int) and ahead_load_tiles >= 2,
        f"moe_glue ahead_load_tiles must be an int of 2 or more, got {ahead_load_tiles!r}",
    )
    attn_res_sbuf_entries = (moe_glue or {}).get("attn_res_sbuf_entries", 0)
    kernel_assert(
        type(attn_res_sbuf_entries) is int and attn_res_sbuf_entries >= 0,
        f"moe_glue attn_res_sbuf_entries must be an int of 0 or more, got {attn_res_sbuf_entries!r}",
    )
    ahead_gram_merge = (moe_glue or {}).get("ahead_gram_merge", False)
    kernel_assert(
        isinstance(ahead_gram_merge, bool), f"moe_glue ahead_gram_merge must be a bool, got {ahead_gram_merge!r}"
    )
    norm_one_mul = (moe_glue or {}).get("norm_one_mul", False)
    kernel_assert(isinstance(norm_one_mul, bool), f"moe_glue norm_one_mul must be a bool, got {norm_one_mul!r}")
    cached_act_squares = (moe_glue or {}).get("cached_act_squares", False)
    kernel_assert(
        isinstance(cached_act_squares, bool), f"moe_glue cached_act_squares must be a bool, got {cached_act_squares!r}"
    )
    ahead_before_resid = (moe_glue or {}).get("ahead_before_resid", False)
    kernel_assert(
        isinstance(ahead_before_resid, bool) and (not ahead_before_resid or attn_res_ahead),
        f"moe_glue ahead_before_resid must be a bool and needs attn_res_ahead, got {ahead_before_resid!r}",
    )
    kda_mlp_tail_entries = (moe_glue or {}).get("kda_mlp_tail_entries", 0)
    kernel_assert(
        isinstance(kda_mlp_tail_entries, int)
        and kda_mlp_tail_entries >= 0
        and (not kda_mlp_tail_entries or (attn_res_cached_bank_sums and attn_res_early_sums)),
        "moe_glue kda_mlp_tail_entries must be an int >= 0 and needs attn_res_cached_bank_sums with "
        f"attn_res_early_sums, got {kda_mlp_tail_entries!r}",
    )
    mla_mlp_tail_entries = (moe_glue or {}).get("mla_mlp_tail_entries", 0)
    kernel_assert(
        isinstance(mla_mlp_tail_entries, int)
        and mla_mlp_tail_entries >= 0
        and (
            not mla_mlp_tail_entries or (attn_res_cached_bank_sums and attn_res_early_sums and attn_res_mla_mlp_early)
        ),
        "moe_glue mla_mlp_tail_entries must be an int >= 0 and needs attn_res_cached_bank_sums with attn_res_early_sums "
        f"and attn_res_mla_mlp_early, got {mla_mlp_tail_entries!r}",
    )
    ahead_mlp_kda_tail = (moe_glue or {}).get("ahead_mlp_kda_tail", False)
    kernel_assert(
        isinstance(ahead_mlp_kda_tail, bool) and (not ahead_mlp_kda_tail or attn_res_ahead_mlp),
        f"moe_glue ahead_mlp_kda_tail must be a bool and needs attn_res_ahead_mlp, got {ahead_mlp_kda_tail!r}",
    )
    kernel_assert(
        ahead_sums_vector in ATTN_RES_AHEAD_VECTOR_MODES
        and (ahead_sums_vector == "off" or attn_res_ahead or attn_res_ahead_mlp),
        f"moe_glue ahead_sums_vector {ahead_sums_vector!r} must be one of {ATTN_RES_AHEAD_VECTOR_MODES}, and other "
        "than 'off' only with attn_res_ahead or attn_res_ahead_mlp",
    )
    # routed_compact fills its slots with the routed tokens of each expert call: with the runner's live rows, the rows
    # of an empty slot route to no expert, so they take no slot in a low-load step.
    moe_pad_rows = None
    if live_rows is not None and moe_glue and moe_glue.get("routed_compact"):
        moe_pad_rows = live_pad_rows(live_rows, queries, "moe_live")
    moe_transpose_block, moe_static_alt_tiles = served["moe_transpose_block"], served["moe_static_alt_tiles"]
    moe_sums_after_routed = served["moe_sums_after_routed"]
    attn_res_sums_after_moe_output = served["attn_res_sums_after_moe_output"]
    mla_rows = rows
    split_v5 = mla_split is not None and "program" in mla_split
    # "split" = the DEV split program (run on split steps only); "one" = the same program with its CC-free runtime guards
    # (the import select, the foreign passes, the owner merge skip a step with no split), the one-NEFF form (a).
    split_program = split_v5 and mla_split["program"] in ("split", "one")
    if split_v5:
        _check_split_v5(
            mla_split,
            mla_split_stats,
            mla_split_info,
            mla_split_own,
            mla_split_foreign,
            mla_split_table,
            rows // queries,
            block_layer_counts(first_layer, norm_weights.shape[0])[1],
            None if mla_block_table is None else mla_block_table.shape[2],
        )
        kernel_assert(first_layer == 0, "mla_split (v5): the stats ride layer 0's DP x gather, so the call starts at 0")
    if split_program:
        # The DEV split program: the MLA layers run the extended batch B + 2 (the 2 foreign slots of the step, one
        # passes), as v1's B + 2, with the k7 MLA levers (each pass of 2 sequences is one k7 pass): C=8 (B = 2) takes
        # F = 2 (B' = 4, the k7 form's 32 rows); C=16 (B = 4) takes F = 4 (B' = 8: the sequence-major passes need
        # B' | 16), two foreign passes.
        kernel_assert(
            mla_split["foreign_slots"] == {16: 2, 32: 4}.get(rows)
            and mla_mode == "k1"
            and mla_loop_path
            and queries == 8
            and not mla_k1_rows
            and mla_block_table is not None,
            "the split program (mla_split v5) is the extended batch of the k7b2 / k7 forms: F = 2 at B = 2, F = 4 at B = 4, "
            f"the k1 MLA loop path, Q = 8 (not the rows form); got Q = {queries}, R = {rows}, F = {mla_split['foreign_slots']}",
        )
        mla_rows = queries * (rows // queries + mla_split["foreign_slots"])
        # The MLA call levers of the served form of the call (k7b2 at C=8, k7 at C=16), so the DEV program differs from
        # the served one only by the split. With the batched cache write the foreign rows (slot 0, an invalid slot)
        # write the null page's scratch rows, as every row a rank does not own does.
        k7_rows = SERVED_FORMS["k7b2" if rows == 16 else "k7"][2]
        split_served = served_defaults(world_size, queries, k7_rows, dict(mla_mode="k1", mla_loop_path=True))
        mla_k1_packed = True
        mla_out_ar = split_served["mla_out_ar"]
        mla_k1_pack_options = split_v5_pack_options(
            world_size,
            queries,
            rows,
            mla_state_bf16_o,
            mla_k1_pack_given,
            mx_qk,
            entry_act=bool(served["mla_p2_entry_act"]),
            step_items=bool(served["mla_step_items"]),
        )
        mla_critical_dma_priority = split_served["mla_critical_dma_priority"]
        mla_cache_write_off_sp = split_served["mla_cache_write_off_sp"]
        mla_gemv_bands = split_served["mla_gemv_bands"]
        mla_batched_cache_write = split_served["mla_batched_cache_write"]
    if mla_split is not None and not split_v5:
        kernel_assert(
            mla_split == {"S": 2}
            and mla_split_info is not None
            and mla_split_table is not None
            and mla_mode == "k1"
            and mla_loop_path
            and queries == 8
            and rows == 16
            and not mla_k1_rows
            and mla_block_table is not None,
            "mla_split (v1) is {'S': 2} with mla_split_info and mla_split_table, on the k1 MLA loop path at Q = 8 and "
            f"B = 2 (the k7b2 form), not the rows form; got Q = {queries}, R = {rows}",
        )
        # The MLA layers run the extended batch B + 2 = the k7 form's Q x 4 rows, with the k7 MLA levers.
        mla_rows = queries * (rows // queries + 2)
        split_served = served_defaults(world_size, queries, mla_rows, dict(mla_mode="k1", mla_loop_path=True))
        mla_k1_packed = True
        mla_out_ar = split_served["mla_out_ar"]
        mla_k1_pack_options = served_pack_options(world_size, queries, mla_rows, True, None, mla_state_bf16_o)
        mla_critical_dma_priority = split_served["mla_critical_dma_priority"]
        mla_cache_write_off_sp = split_served["mla_cache_write_off_sp"]
        mla_gemv_bands = split_served["mla_gemv_bands"]
        mla_batched_cache_write = False
    dp = world_layout(world_size)["dp"]
    routed_tp, moe_mx_tile = routed_layers_options(world_size, routed_layers, routed_tp, moe_mx_tile)
    n_tiles = H // P_MAX
    num_layers = norm_weights.shape[0]
    if kda_page_groups is not None:
        page_groups = tuple(kda_page_groups)
        call_kda_layers = block_layer_counts(first_layer, num_layers)[0]
        kernel_assert(
            len(page_groups) == call_kda_layers,
            f"kda_page_groups has {len(page_groups)} entries, and the call has {call_kda_layers} KDA layers",
        )
        group_ids = set(page_groups)
        kernel_assert(
            all(isinstance(g, int) and not isinstance(g, bool) for g in page_groups)
            and group_ids == set(range(len(group_ids)))
            and len(group_ids) <= KDA_PAGE_GROUPS_MAX,
            f"kda_page_groups must number the KV cache groups 0 .. G - 1 with G at most {KDA_PAGE_GROUPS_MAX}, got "
            f"{sorted(group_ids, key=str)}",
        )
    kernel_assert(dp_size is None or dp_size == dp, f"dp_size {dp_size} is not world_size / 32 = {dp}")
    kernel_assert(dp != 1, "The multi-server verify runs DP replicas of TP32: dp_size 1 has no form here")
    kernel_assert(
        mla_block_table is None or mla_block_table.dtype == nl.int32,
        f"mla_block_table must be INT32 page ids (the MLA loops copy its rows with DMAs, which do not cast), got "
        f"{getattr(mla_block_table, 'dtype', None)}",
    )
    kernel_assert(moe_form in MOE_FORMS, f"moe_form must be one of {MOE_FORMS}, got {moe_form!r}")
    kernel_assert(moe_gather in MOE_GATHERS, f"moe_gather must be one of {MOE_GATHERS}, got {moe_gather!r}")
    kernel_assert(
        moe_rows_load_split == 1 or (moe_rows_load_split in ROWS_LOAD_SPLITS and moe_gather == "dp_tp"),
        f"moe_rows_load_split is 1 or one of {ROWS_LOAD_SPLITS} with moe_gather 'dp_tp', got {moe_rows_load_split}",
    )
    # The moe_glue key rows_align (tkg_moe_2srv.GLUE_KEYS): the 64-word fp8 router rows (rows_latent fp8a), for the
    # sources and the body.
    rows_align = bool((moe_glue or {}).get("rows_align"))
    kernel_assert(
        not rows_align or (moe_rows_latent_fp8 and moe_router_rows_drop_selection and moe_gather == "dp_tp"),
        "moe_glue rows_align pads the fp8 drop_selection rows of the dp_tp gather: it needs moe_rows_latent_fp8, "
        "moe_router_rows_drop_selection and moe_gather 'dp_tp'",
    )
    rows_latent = ("fp8a" if rows_align else "fp8") if moe_rows_latent_fp8 else "bf16"
    kernel_assert(
        not moe_routed_own_rows or moe_routed_reduce == "ar",
        f"moe_routed_own_rows loads the replica rows of the routed all-reduce (moe_routed_reduce 'ar'), got"
        f" {moe_routed_reduce!r}",
    )
    kernel_assert(mla_mode in MLA_MODES, f"mla_mode must be one of {MLA_MODES}, got {mla_mode!r}")
    kernel_assert(1 <= queries <= MAX_QUERIES, f"A verify step takes 1 to {MAX_QUERIES} tokens of each sequence")
    kernel_assert(rows % queries == 0, "The stream holds Q rows for each sequence")
    kernel_assert(
        max_verify_rows is None or MAX_VERIFY_ROWS <= max_verify_rows <= SIZING_VERIFY_ROWS,
        f"max_verify_rows must be None or {MAX_VERIFY_ROWS} .. {SIZING_VERIFY_ROWS}, got {max_verify_rows}",
    )
    row_bound = MAX_VERIFY_ROWS if max_verify_rows is None else max_verify_rows
    kernel_assert(
        rows <= row_bound,
        f"R={rows} rows (Q x B) of a replica: the multi-server verify takes at most {row_bound} (MAX_VERIFY_ROWS = "
        f"{MAX_VERIFY_ROWS}; max_verify_rows {max_verify_rows})",
    )
    kernel_assert(
        not mix_cache,
        "mix_cache has no multi-server verify form: the HBM AttnRes bank push of the verify waits only for a plain mix",
    )
    kernel_assert(
        not (attn_res_cached_bank_sums and (attn_res_bank_in is not None or return_attn_res_bank)),
        "attn_res_cached_bank_sums is for the one-call verify: the split calls pass no bank rstd",
    )
    kernel_assert(
        not attn_res_early_sums or attn_res_cached_bank_sums,
        "attn_res_early_sums needs attn_res_cached_bank_sums (the kept bank rstd)",
    )
    kernel_assert(
        not (attn_res_ahead and return_aux and aux_tap == "mixture"),
        "attn_res_ahead has no mix tile for the mixture aux tap: take aux_tap 'prefix'",
    )
    kernel_assert(
        not attn_res_mla_mlp_early or (attn_res_early_sums and (mla_k1_rows or mla_state_bf16_o)),
        "attn_res_mla_mlp_early needs attn_res_early_sums and the k1 MLA call in the rows form (mla_k1_rows) or with the "
        "BF16-o state reduce-scatter (mla_state_bf16_o)",
    )
    kernel_assert(
        attn_res_bank_dma is None or attn_res_bank_dma in BANK_DMAS,
        f"attn_res_bank_dma must be one of {BANK_DMAS} or None, got {attn_res_bank_dma!r}",
    )
    kernel_assert(first_layer + num_layers <= NUM_LAYERS, f"The model has {NUM_LAYERS} layers")
    kernel_assert(banks_snapshot(first_layer), "Start the layer range at a bank snapshot layer")
    n_in = sum(1 for i in range(first_layer) if banks_snapshot(i))
    moe_expert_prefetch, prefetch_after_in_proj = moe_expert_prefetch_parts(moe_expert_prefetch)
    kernel_assert(
        not moe_expert_prefetch_start
        or (moe_expert_prefetch_start == {"mla": "front"} and moe_expert_prefetch.get("mla", 0) > 0),
        f"moe_expert_prefetch_start must be None, {{}} or {{'mla': 'front'}} with moe_expert_prefetch mla >= 1, got "
        f"{moe_expert_prefetch_start} with moe_expert_prefetch {moe_expert_prefetch}",
    )
    # The token-major G1a rows win only with the KDA-led expert ring resident at the call start (MEASURED: the
    # stream form alone is slower in ring-late layers), so the two ship together.
    kernel_assert(
        not (moe_glue or {}).get("mx_rows_stream") or moe_expert_prefetch.get("kda", 0) >= 1,
        "mx_rows_stream needs the KDA expert ring: set moe_expert_prefetch with kda >= 1 (the served C = 16 form: "
        f"{{'kda': 3, 'mla': 0}}), got moe_expert_prefetch {moe_expert_prefetch}",
    )
    kernel_assert(
        moe_expert_queues in EXPERT_QUEUES, f"moe_expert_queues must be one of {EXPERT_QUEUES}, got {moe_expert_queues}"
    )
    kernel_assert(
        not mla_late_o_proj_weights or mla_late_tail_weights,
        "mla_late_o_proj_weights moves the o_proj loads of the late tail: it needs mla_late_tail_weights",
    )
    kernel_assert(not mla_state_bf16_o or mla_mode == "k1", "mla_state_bf16_o is a k1 MLA call option")
    kernel_assert(
        not mla_out_ar or (mla_mode == "k1" and mla_loop_path),
        "mla_out_ar replaces the output collectives of the k1 MLA call on the loop path: it needs mla_mode 'k1' and "
        "mla_loop_path",
    )
    kernel_assert(
        not mla_late_tail_frames or mla_late_o_proj_weights,
        "mla_late_tail_frames re-scopes the late tail with its late o_proj loads: it needs mla_late_o_proj_weights",
    )
    kernel_assert(
        not mla_k1_rows
        or (
            mla_mode == "k1"
            and mla_loop_path
            and rows // queries in ROWS_SEQUENCES
            and (rows <= ROWS_PASS_ROWS or mla_block_table.shape[2] >= _K1_FIT_BLOCKS)
        ),
        "mla_k1_rows is the CP32 loop-path k1 verify of B in ROWS_SEQUENCES sequences for each replica (the form of "
        f"its decode twin, mla_rows_twin): set mla_mode k1, the MLA loop path, B in {ROWS_SEQUENCES}, and above "
        f"{ROWS_PASS_ROWS} rows the fit form (at least {_K1_FIT_BLOCKS} local blocks)",
    )
    kernel_assert(
        not mla_k1_packed or (mla_mode == "k1" and mla_loop_path and not mla_k1_rows),
        "mla_k1_packed is the packed prior-key loop of the CP32 k1 verify: set mla_mode k1, the MLA loop path, and no "
        "mla_k1_rows",
    )
    kernel_assert(mla_k1_packed or not mla_k1_pack_options, "mla_k1_pack_options needs mla_k1_packed")
    kernel_assert(
        (mla_k1_rows or not mla_k1_rows_options) and set(mla_k1_rows_options or ()) <= set(ROWS_OPTIONS),
        f"mla_k1_rows_options needs mla_k1_rows and takes the keys {ROWS_OPTIONS}, got {mla_k1_rows_options}",
    )
    kernel_assert(
        not ((mla_k1_rows_options or {}).get("rows_pair_tail") and (mla_k1_rows_options or {}).get("rows_mx_keys")),
        "mla_k1_rows_options rows_pair_tail does not take rows_mx_keys (an unmeasured pair)",
    )
    pad_skip = (mla_k1_rows and (mla_k1_rows_options or {}).get("rows_pad_skip")) or (
        mla_k1_packed and (mla_k1_pack_options or {}).get("pad_skip")
    )
    kernel_assert(
        not pad_skip or live_rows is not None,
        "the pad skips of the k1 MLA call need live_rows (the runner's live-row mask): a live row can sit at position 0",
    )
    kernel_assert(
        kda_w_in_buffers >= 0, f"kda_w_in_buffers must be 0 (the default) or a buffer count, got {kda_w_in_buffers}"
    )
    kernel_assert(
        not (kda_prework_ahead and (kda_state_loads_first or kda_slot_pages_first)),
        "kda_state_loads_first and kda_slot_pages_first move loads of the inline KDA prework: not with "
        "kda_prework_ahead",
    )
    kernel_assert(
        not (kda_state_loads_first or kda_slot_pages_first) or kda_replay == "states",
        "kda_state_loads_first and kda_slot_pages_first need kda_replay 'states'",
    )
    kernel_assert(kda_slot_store_queues in (1, 2), f"kda_slot_store_queues must be 1 or 2, got {kda_slot_store_queues}")
    kernel_assert(
        mla_pass_q_prefetch in range(PASS_Q_PREFETCH_MAX_TILES + 1),
        f"mla_pass_q_prefetch {mla_pass_q_prefetch!r} is not a tile count from 0 to {PASS_Q_PREFETCH_MAX_TILES}. 3 tiles "
        f"leave the second MLA pass's packed loop too little SBUF in the k7 verify. Use 0 to {PASS_Q_PREFETCH_MAX_TILES}.",
    )
    kernel_assert(
        mla_q_gather_buffers in (1, 2),
        f"mla_q_gather_buffers {mla_q_gather_buffers!r} is not 1 or 2. Use 1 or 2 query gather tiles.",
    )
    kernel_assert(
        not (kda_token_tiles or kda_slot_store_queues != 1) or kda_replay == "states",
        "kda_token_tiles and kda_slot_store_queues need kda_replay 'states'",
    )
    kernel_assert(
        not kda_shared_state_ring
        or (kda_replay == "states" and kda_chain == "sequential" and kda_slot_state_depth == 2),
        "kda_shared_state_ring needs kda_replay 'states', kda_chain 'sequential' and kda_slot_state_depth 2",
    )
    kernel_assert(
        not kda_slot_state_per_token
        or (kda_replay == "states" and kda_chain == "sequential" and kda_token_tiles and not kda_shared_state_ring),
        "kda_slot_state_per_token needs kda_replay 'states', kda_chain 'sequential', kda_token_tiles and no "
        "kda_shared_state_ring",
    )
    kernel_assert(
        1 <= kda_slot_state_depth <= 2
        or (
            kda_slot_state_depth <= queries
            and kda_replay == "states"
            and kda_chain == "sequential"
            and kda_token_tiles
            and not kda_shared_state_ring
        ),
        f"kda_slot_state_depth {kda_slot_state_depth} above 2 needs at most {queries} (the queries), kda_replay "
        "'states', kda_chain 'sequential', kda_token_tiles and no kda_shared_state_ring",
    )
    kernel_assert(
        kda_slot_store_engine in ("sync", "gpsimd", "split"),
        f"kda_slot_store_engine must be 'sync', 'gpsimd' or 'split', got {kda_slot_store_engine!r}",
    )
    kernel_assert(
        kda_slot_store_engine == "sync" or (kda_replay == "states" and kda_slot_store_queues == 1),
        "kda_slot_store_engine 'gpsimd' / 'split' need kda_replay 'states' and kda_slot_store_queues 1",
    )
    kernel_assert(
        kda_state_dtype in KDA_STATE_DTYPES,
        f"kda_state_dtype {kda_state_dtype!r} must be one of {KDA_STATE_DTYPES}",
    )
    kernel_assert(
        kda_state_dtype == "bf16" or kda_chain != "chunked",
        f"kda_state_dtype {kda_state_dtype!r} needs kda_chain 'sequential' or 'batched': the chunked chain keeps a "
        "bf16 state",
    )
    kernel_assert(
        moe_routed_reduce in ROUTED_REDUCES,
        f"moe_routed_reduce must be one of {ROUTED_REDUCES}, got {moe_routed_reduce!r}",
    )
    kernel_assert(
        not moe_routed_stats_ag or moe_routed_reduce == "rs_ag",
        f"moe_routed_stats_ag needs moe_routed_reduce 'rs_ag', got {moe_routed_reduce!r}",
    )
    kernel_assert(
        moe_out_proj_form in OUT_PROJ_FORMS,
        f"moe_out_proj_form must be one of {OUT_PROJ_FORMS}, got {moe_out_proj_form!r}",
    )
    if moe_out_proj_form == "column":
        kernel_assert(
            moe_routed_reduce in ("ar", "rs_ag") and not moe_routed_stats_ag,
            f"moe_out_proj_form 'column' norms the full routed rows: it takes moe_routed_reduce 'ar' or 'rs_ag' "
            f"without moe_routed_stats_ag, got {moe_routed_reduce!r}, moe_routed_stats_ag {moe_routed_stats_ag}",
        )
        kernel_assert(
            moe_out_proj_scale is None,
            "moe_out_proj_form 'column' runs the BF16 out-projection in the column layout: not with moe_out_proj_scale "
            "(the MXFP8 padded-lane form)",
        )
    kernel_assert(
        moe_routed_reduce == "ar" or (world_size, rows) in VERIFY_ROUTED_RS_AG_FORMS + VERIFY_ROUTED_RS_AG_FORMS_4SRV,
        f"moe_routed_reduce {moe_routed_reduce!r} takes the (world_size, R) forms "
        f"{VERIFY_ROUTED_RS_AG_FORMS + VERIFY_ROUTED_RS_AG_FORMS_4SRV} only, "
        f"got ({world_size}, {rows})",
    )
    kernel_assert(
        moe_routed_reduce == "ar" or cc_source_pingpong,
        f"moe_routed_reduce {moe_routed_reduce!r} alternates its collective tensors by layer parity: it needs "
        "cc_source_pingpong",
    )
    pair_shared = check_pair_shared_forms(pair_shared_cc_dsts)
    kernel_assert(
        "moe_gather" not in pair_shared,
        "pair_shared_cc_dsts moe_gather: the gathers of the multi-server MoE body stay private",
    )
    kernel_assert(
        "moe_routed" not in pair_shared or moe_routed_reduce == "ar",
        f"pair_shared_cc_dsts moe_routed needs moe_routed_reduce 'ar' (the world all-reduce), got {moe_routed_reduce!r}",
    )
    check_column_private(moe_out_proj_form, pair_shared)
    kernel_assert(
        not pair_shared or (attn_res_bank_in is None and not return_attn_res_bank),
        f"pair_shared_cc_dsts {pair_shared} is not proven with the split verify calls: give () with attn_res_bank_in / "
        "return_attn_res_bank",
    )
    split = attn_res_bank_in is not None or return_attn_res_bank
    kernel_assert(
        (attn_res_bank_in is not None) == (first_layer > 0) and (attn_res_fence_in is not None) == (first_layer > 0),
        "Give attn_res_bank_in and attn_res_fence_in exactly when the layer range starts after layer 0",
    )
    kernel_assert(
        attn_res_bank_in is None
        or (tuple(attn_res_bank_in.shape) == (P_MAX, n_in, rows, n_tiles) and attn_res_bank_in.dtype == stream.dtype),
        f"attn_res_bank_in must be [{P_MAX}, {n_in}, {rows}, {n_tiles}] in the stream dtype: the bank entries of the "
        f"{n_in} snapshot layers before layer {first_layer}",
    )
    kernel_assert(
        attn_res_fence_in is None
        or (tuple(attn_res_fence_in.shape) == FENCE_SHAPE and attn_res_fence_in.dtype == stream.dtype),
        f"attn_res_fence_in must be {list(FENCE_SHAPE)} in the stream dtype",
    )
    kernel_assert(
        not split or rows > _BANK_HBM_ROWS,
        f"The split verify needs more than {_BANK_HBM_ROWS} rows: its bank must be in HBM",
    )
    kernel_assert(
        not return_attn_res_bank
        or (
            banks_snapshot(first_layer + num_layers)
            and lm_head_weight is None
            and final_attn_res_weight is None
            and final_norm_weight is None
        ),
        "return_attn_res_bank is for a call that feeds a next call: the layer after the range must be a bank "
        "snapshot layer, and the call takes no final weights and no lm_head_weight",
    )
    kernel_assert(
        lm_head_weight is not None or not (greedy or return_logits),
        "greedy and return_logits need lm_head_weight",
    )
    grammar_inputs = (grammar_ids, grammar_rows_in, grammar_class_of, grammar_next, grammar_mask_bias)
    grammar_on = grammar_ids is not None
    kernel_assert(
        all((t is not None) == grammar_on for t in grammar_inputs),
        "the grammar mask takes all of grammar_ids, grammar_rows_in, grammar_class_of, grammar_next and "
        "grammar_mask_bias, or none of them",
    )
    kernel_assert(
        not grammar_on or (lm_head_weight is not None and (greedy or return_logits)),
        "the grammar mask needs the lm_head tail (lm_head_weight, with greedy or return_logits)",
    )
    kernel_assert(aux_tap in AUX_TAPS, f"aux_tap must be one of {AUX_TAPS}")
    all_pools = (
        kv_pool_0,
        kv_pool_1,
        kv_pool_2,
        kv_pool_3,
        kv_pool_4,
        kv_pool_5,
        kv_pool_6,
        kv_pool_7,
        kv_pool_8,
        kv_pool_9,
        kv_pool_10,
        kv_pool_11,
        kv_pool_12,
        kv_pool_13,
        kv_pool_14,
        kv_pool_15,
        kv_pool_16,
        kv_pool_17,
        kv_pool_18,
        kv_pool_19,
        kv_pool_20,
        kv_pool_21,
        kv_pool_22,
        kv_pool_23,
    )
    if pool_of_layer is None:
        pool_of_layer = tuple(range(num_layers))
    pool_of_layer = tuple(int(i) for i in pool_of_layer)
    kernel_assert(len(pool_of_layer) == num_layers, "Give the pool index of each layer of the range")
    n_pools = max(pool_of_layer) + 1
    kernel_assert(n_pools <= MAX_POOLS, f"The kernel takes at most {MAX_POOLS} page pools")
    kernel_assert(sorted(set(pool_of_layer)) == list(range(n_pools)), "pool_of_layer must use each pool 0 .. k-1")
    pools = all_pools[:n_pools]
    kernel_assert(all(p is not None for p in pools), "Give each page pool that pool_of_layer uses")
    layer_pools = tuple(pools[i] for i in pool_of_layer)
    kernel_assert(
        not kda_paired_pages or kda_state_dtype != "bf16", "kda_paired_pages is a form of the fp32 KDA state pages"
    )
    if kda_state_dtype != "bf16":
        # An fp32 state takes twice the bf16 state bytes of the page: refuse a pool that does not hold them (a paired
        # page is two pool rows: tkg_kda.kda_page_span).
        state_end = kda_page_offsets(kda_local_heads(kda_w_out[0], kda_w_small[0]), kda_state_dtype)["end"]
        span = kda_page_span(kda_paired_pages)
        for li, pool in enumerate(layer_pools):
            kernel_assert(
                is_mla_layer(first_layer + li) or (span * pool.shape[1] >= state_end and pool.shape[1] % 2 == 0),
                f"kda_state_dtype {kda_state_dtype!r}: the pool of KDA layer {first_layer + li} is {pool.shape[1]} "
                f"bf16 elements wide; the fp32 state needs an even width of at least {state_end} (the 512-row pages "
                "[P, 147456]), or with kda_paired_pages 2 x the width",
            )

    aux_layers = tuple(int(i) for i in aux_layers)
    aux_out = None
    if return_aux:
        aux_out = nl.ndarray((len(aux_layers), rows, H), dtype=stream.dtype, buffer=nl.shared_hbm, name="aux_out")
    layer_streams = None
    if debug:
        layer_streams = nl.ndarray(
            (num_layers, rows, H), dtype=stream.dtype, buffer=nl.shared_hbm, name="layer_streams"
        )

    # Persistent staging ring of the prefix aux stores on the region (nki-dma-source-liveness), as
    # kimi_k3_verify_model; off the region the ring is None. The mixture tap keeps its per-row tiles (an FP32 mix).
    aux_rows = {"ring": None, "waits": {}}

    def store_aux(layer, s_hT):
        if aux_out is not None and aux_tap == "prefix" and layer in aux_layers:
            store_hT_rows(
                aux_out[aux_layers.index(layer)],
                s_hT,
                ring=aux_rows["ring"],
                waits=aux_rows["waits"],
                **({"queues": AUX_STORE_QUEUES} if aux_fast else {}),
            )

    def store_mix(layer, mixed):
        if aux_out is None or aux_tap != "mixture" or layer - 1 not in aux_layers:
            return
        # The mix can live in an MLA frame, so the store reads a fresh copy that the frame cannot reuse.
        staged = nl.ndarray(mixed.shape, dtype=mixed.dtype, buffer=nl.sbuf)
        nisa.tensor_copy(dst=staged, src=mixed)
        store_hT_rows(aux_out[aux_layers.index(layer - 1)], staged)

    kda_weights = (kda_w_in, kda_w_out, kda_w_small, kda_params)
    if kda_w_in_scale is not None:
        kda_weights += (kda_w_in_scale,)
    if kda_w_out_scale is not None:
        kda_weights += (kda_w_out_scale,)
    mla_weights = (mla_in_proj, mla_q_a_norm, mla_q_b_proj, mla_kv_a_norm, mla_kv_b_proj, mla_o_proj)
    if mla_in_proj_scale is not None:
        mla_weights += (mla_in_proj_scale,)
    if mla_q_b_proj_scale is not None:
        mla_weights += (mla_q_b_proj_scale,)
    if mla_kv_b_proj_scale is not None:
        mla_weights += (mla_kv_b_proj_scale,)
    if mla_o_proj_scale is not None:
        mla_weights += (mla_o_proj_scale,)
    moe_weights = (
        moe_in_proj,
        moe_out_proj,
        moe_vectors,
        expert_gate_up_weights,
        expert_gate_up_weights_scale,
        expert_down_weights,
        expert_down_weights_scale,
    )
    dense_weights = (dense_gate_up, dense_down)
    totals = block_layer_counts(first_layer, num_layers)
    # A routed_layers plan whose layers of the call take other tags (moe_routed_plan): each MoE layer runs the
    # split and the stacks of its tag; None (no plan, or one tag) keeps the single split.
    check_plan_cut(routed_layers, attn_res_bank_in is not None or return_attn_res_bank)
    routed_plan = call_plan(world_size, routed_layers, first_layer, num_layers)
    # The CSR gathers of the call's dynamic packed MoE layers: "staged" (None) while they fit the DGE table.
    moe_dynamic_csr = None
    if moe_routed_dynamic:
        moe_dynamic_csr = call_csr_gather(world_size, routed_layers, first_layer, num_layers, routed_tp, moe_mx_tile)
        moe_dynamic_csr = None if moe_dynamic_csr == "staged" else moe_dynamic_csr
    alternates = dict(
        zip(
            ALT_STACK_ARGUMENTS,
            (
                routed_alt0_expert_gate_up_weights,
                routed_alt0_expert_gate_up_weights_scale,
                routed_alt0_expert_down_weights,
                routed_alt0_expert_down_weights_scale,
                routed_alt1_expert_gate_up_weights,
                routed_alt1_expert_gate_up_weights_scale,
                routed_alt1_expert_down_weights,
                routed_alt1_expert_down_weights_scale,
            ),
            strict=True,
        )
    )
    check_alternates(alternates, routed_plan)
    kernel_assert(
        not moe_routed_dynamic or moe_mx_tile is not None or routed_plan is not None,
        "moe_routed_dynamic runs the splits of sub-512 MX tiles: it needs moe_mx_tile, or a routed_layers plan with "
        "such a tag",
    )
    if routed_plan is not None:
        moe_weights = plan_weights(
            routed_plan, routed_layers[0], moe_weights, alternates, moe_layers_of(first_layer, num_layers)
        )

    region = None
    if mla_loop_path:
        region = ExplicitSbuf()
        install_identities(region)
        if mla_null_rows_once and mla_batched_cache_write:
            install_null_rows(region)
        if aux_out is not None and aux_row_ring:
            depth = AUX_STORE_RING_DEPTH if aux_fast else AUX_RING_DEPTH
            ring = (region.heap((n_tiles, P_MAX), stream.dtype, f"aux_rows_{r}") for r in range(depth))
            aux_rows["ring"] = tuple(ring)
    n_entries = n_in + sum(1 for i in range(first_layer, first_layer + num_layers) if banks_snapshot(i))
    bank_shape = (P_MAX, n_entries, rows, n_tiles)
    k1 = mla_mode == "k1"
    # As kimi_k3_verify_model: the static bank stays in SBUF up to 16 rows (8 with k1); above, and above 8 rows of a
    # loop path, it goes to HBM.
    k1_hbm_bank = k1 and rows > _BANK_HBM_ROWS
    # moe_glue attn_res_sbuf_entries: above 8 rows, the first entries in the region heap and the others in HBM
    # (common.SplitBank), as kimi_k3_verify_model attn_res_sbuf_entries. Up to 8 rows the bank is in SBUF already.
    sbuf_entries = min(attn_res_sbuf_entries, n_entries) if rows > _BANK_HBM_ROWS else 0
    kernel_assert(
        not sbuf_entries or (k1 and region is not None),
        f"moe_glue attn_res_sbuf_entries above {_BANK_HBM_ROWS} rows needs mla_mode k1 and the explicit SBUF region "
        "(mla_loop_path) for the SBUF entries",
    )
    if region is None and rows <= 2 * _BANK_HBM_ROWS and not k1_hbm_bank:
        bank = (nl.ndarray(bank_shape, dtype=stream.dtype, buffer=nl.sbuf), 0)
    elif sbuf_entries:
        sbuf_part = region.heap((P_MAX, sbuf_entries, rows, n_tiles), stream.dtype, "bank")
        hbm_part = None
        if n_entries > sbuf_entries:
            hbm_part = nl.ndarray(
                (P_MAX, n_entries - sbuf_entries, rows, n_tiles),
                dtype=stream.dtype,
                buffer=nl.private_hbm,
                name="verify_attn_res_bank",
            )
        # The split bank reports a DRAM buffer: the mixes and the pushes take the forms of an HBM bank, and the split
        # calls copy each entry in and out at its home.
        bank = (SplitBank(sbuf_part, hbm_part), 0)
    elif region is None or rows > _BANK_HBM_ROWS:
        bank = (nl.ndarray(bank_shape, dtype=stream.dtype, buffer=nl.private_hbm, name="verify_attn_res_bank"), 0)
    else:
        bank = (region.heap(bank_shape, stream.dtype, "bank"), 0)
    # Every HBM bank push waits for the mix that reads the last entry.
    verify_bank = bank[0].buffer != nl.sbuf
    if attn_res_bank_in is not None:
        # The entries of the call before go first; the pushes of this range follow them.
        for c in range(n_in):
            nisa.dma_copy(dst=bank[0][0:P_MAX, c], src=attn_res_bank_in[0:P_MAX, c], name=f"attn_res_bank_in_{c}")
        bank = (bank[0], n_in)
    s = stream
    if attn_res_fence_in is not None:
        # The first block takes this SBUF stream: on the loop path it must be a stream tile of the region.
        if region is None:
            s = nl.ndarray((P_MAX, rows, n_tiles), dtype=stream.dtype, buffer=nl.sbuf)
        else:
            s = region_streams(region, rows, n_tiles, stream.dtype)[0]
        (load_rows_hT_batched if rows_hT_batched else load_rows_hT)(stream, s)
        fence_word = nl.ndarray(FENCE_SHAPE, dtype=stream.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=fence_word, src=attn_res_fence_in, name="attn_res_fence_in")
        # OR of the zero fence word: the stream of the first layer depends on the fence, and no bit changes.
        bits = s.reshape((P_MAX, rows * n_tiles)).view(nl.uint16)
        nisa.tensor_scalar(
            dst=bits,
            data=bits,
            op0=nl.bitwise_or,
            operand0=fence_word.view(nl.uint16)[0:P_MAX, 0:1],
            engine=nisa.engine.vector,
        )
    gate = ()
    starts = [0, 0, 0, 0]
    greedy_tail = lm_head_weight is not None and greedy
    # As kimi_k3_decode_model_2srv: off the loop path the argmax tiles keep the destination arena non-empty.
    tail_ranks = TAIL_TP_RANKS if greedy_tail or not mla_loop_path else 0
    sources = verify_2srv_sources(
        rows,
        queries,
        n_tiles,
        stream.dtype,
        region,
        tail_ranks,
        world_size,
        routed_tp,
        moe_gather,
        mla_mode,
        cc_source_pingpong,
        routed_reduce=moe_routed_reduce,
        stats_gather=moe_routed_stats_ag,
        rows_latent=rows_latent,
        rows_dst_pingpong=moe_rows_dst_pingpong,
        **({"drop_selection": True} if moe_router_rows_drop_selection else {}),
        **({"latent_gather_mx": True} if moe_latent_gather_mx else {}),
        **({"out_proj_form": moe_out_proj_form} if moe_out_proj_form != "reduced" else {}),
        **({"pair_shared": pair_shared} if pair_shared else {}),
        # The moe_glue key mx_scores_align (tkg_moe_2srv.GLUE_KEYS): the 65-word G1a router rows.
        **({"mx_scores_align": True} if (moe_glue or {}).get("mx_scores_align") else {}),
        # The moe_glue key dense_x_gather (tkg_moe_2srv.GLUE_KEYS): the x gather form of the dense layer 0.
        **(
            {"dense_x_gather": moe_glue["dense_x_gather"]}
            if (moe_glue or {}).get("dense_x_gather", "dp") != "dp"
            else {}
        ),
        # The moe_glue key mx_rows_stream: the 64-word token-major G1a router rows.
        **({"mx_rows_stream": True} if (moe_glue or {}).get("mx_rows_stream") else {}),
        **({"mx_tile": moe_mx_tile} if moe_mx_tile is not None else {}),
    )
    if moe_glue and moe_glue.get("selectors_once"):
        # The selectors live for the whole kernel: on the loop path they take heap addresses before the first block.
        # One name for each tag's selectors: the served name for the default split (plan_selector_splits).
        default_tag = routed_layers[0] if routed_layers is not None else None
        for sel_name, split_tp, split_tile in plan_selector_splits(
            world_size, routed_tp, moe_mx_tile, routed_plan, default_tag
        ):
            moe_split = moe_2srv_world_split(world_size, split_tp, split_tile)
            moe_2srv_glue_selectors(
                rank_info,
                min(P_MAX, moe_split["dp"] * rows),
                moe_split,
                sel_name,
                SbufTiles(region, heap=True) if region is not None else None,
            )
    if moe_glue and moe_glue.get("routed_compact"):
        # routed_compact's slot-table constants: built once for the kernel, before its first block. The moe_glue key
        # compact_key_constants also builds the token keys of each 128-token tile and the hot_last expert order.
        key_split = moe_2srv_world_split(world_size, routed_tp, moe_mx_tile)
        key_options = (
            {
                "key_tiles": -(-key_split["dp"] * rows // P_MAX),
                "local_experts": key_split["local_experts"],
            }
            if moe_glue.get("compact_key_constants")
            else {}
        )
        moe_2srv_compact_constants(
            rank_info, sb=SbufTiles(region, heap=True) if region is not None else None, **key_options
        )
    first_rows = first_rows_destination(sources, first_layer)
    k1_loop = k1 and region is not None
    # Above 16 rows at a 256K or 1M table the loop-path k1 call takes the fit forms, as in kimi_k3_verify_model.
    k1_fit = k1_loop and mla_rows > 16 and mla_block_table.shape[2] >= _K1_FIT_BLOCKS
    kernel_assert(
        mla_split is None or split_v5 or k1_fit,
        "mla_split needs the k1 fit form: the loop path at a table of 64 local blocks or more",
    )
    kernel_assert(
        not mla_k1_packed
        or mla_rows <= 16
        or (k1_fit and 16 % queries == 0 and (mla_rows // queries) % (16 // queries) == 0),
        f"mla_k1_packed needs passes of whole sequences: at most 16 rows, or the sequence-major passes above 16 rows "
        f"(a table of {_K1_FIT_BLOCKS} local blocks or more, Q dividing 16); got {rows} rows of Q = {queries}",
    )
    kernel_assert(not cc_pad256 or k1_loop, "cc_pad256 pads the HBM state pair of the loop-path k1 MLA call")
    kernel_assert(
        not mla_qkv_parity or k1_loop, "mla_qkv_parity is for the loop path of the k1 MLA call (a region and k1)"
    )
    mla_sets = None
    if k1 and region is None:
        mla_sets = sources["verify_k1"]
    elif k1_loop:
        # As kimi_k3_verify_model: region heap tiles plus HBM pairs, with both gathers in HBM.
        k1_kw = {"q_hbm": True, **({"out_ar_call": True} if mla_out_ar else {"out_ag_call": True})}
        mla_sets = {
            **mla_block_collective_sources(
                mla_rows,
                _MLA_LOCAL_HEADS,
                SbufTiles(region, heap=True),
                with_rs=True,
                rs_call=k1_fit,
                qkv_extra_rows=0,
                # The split program's extended batch returns its request rows only.
                **({"out_rows": rows} if split_program else {}),
                **k1_kw,
            ),
            **mla_block_collective_hbm(
                mla_rows,
                _MLA_LOCAL_HEADS,
                name="_vk1_2srv",
                state_call=k1_fit,
                **({"out_rows": rows} if split_program else {}),
                **k1_kw,
                **({"state_sums": True} if mla_state_bf16_o else {}),
                **({"q_fp8": q8} if (q8 := q_call_fp8(mla_k1_pack_options, mla_k1_rows, mla_k1_rows_options)) else {}),
                **({"cc_pad256": True} if cc_pad256 else {}),
            ),
        }
        # ``hoist_plans``: the item plans of the passes, built once for the MLA layers of the call, as in
        # kimi_k3_verify_model. Opt-in here (the caller's mla_k1_pack_options names hoist_plans True); without it each
        # pass builds its own plan.
        if mla_k1_packed and (mla_k1_pack_options or {}).get("hoist_plans", False) and (mla_split is None or split_v5):
            mla_sets["pass_plans"] = mla_pass_plans(
                position_rows[0],
                rank_info,
                mla_block_table[0],
                queries,
                mla_k1_pack_options,
                SbufTiles(region, heap=True),
                "mla_k1_plan",
                live_rows=live_rows,
            )
        elif mla_k1_packed and (mla_k1_pack_options or {}).get("fp8_pv_step_store") and mla_split is None:
            # ``fp8_pv_step_store``: only the FP8 P@V step store of each pass (no hoisted plans of the packed loop): the
            # first MLA layer of the step builds the item plan of each pass and the later ones read it (bitwise: every
            # MLA layer of the step reads the same table).
            mla_sets["pass_plans"] = mla_fp8_pv_step_stores(
                mla_block_table[0], queries, mla_k1_pack_options, SbufTiles(region, heap=True), "mla_k1_plan"
            )
        parity_tiles = []
        if mla_qkv_parity:
            # The second qkv all-gather pair (mla_qkv_parity): MLA layer n takes pair n mod 2. Thus the next MLA layer
            # writes the other source, not the source that a peer of this layer can still read.
            heap = SbufTiles(region, heap=True)
            qkv, qkv_dst = mla_sets["qkv"], mla_sets["qkv_dst"]
            second = {
                "qkv": heap.tile(tuple(qkv.shape), qkv.dtype, "mla_qkv_src_p1"),
                "qkv_dst": heap.tile(tuple(qkv_dst.shape), qkv_dst.dtype, "mla_qkv_dst_p1"),
            }
            mla_sets["qkv_parity"] = ({"qkv": qkv, "qkv_dst": qkv_dst}, second)
            parity_tiles = [second["qkv"], second["qkv_dst"]]
        # Clear the SBUF sites at the start, so the compiler places no tile on their bytes before the first layer.
        rs_tiles = [mla_sets["rs_call"]] if k1_fit else mla_sets["rs"]
        ag_tiles = [] if mla_out_ar else [mla_sets["ag_call"]]
        for tile in [mla_sets["qkv"], mla_sets["qkv_dst"], *parity_tiles, *rs_tiles, *ag_tiles]:
            nisa.memset(dst=tile, value=0.0)
    elif region is None:
        mla_sets = sources["verify_mla"]
    table_mla, slot_mla, split_body, split_aux = mla_block_table, mla_slot, None, None
    split_outputs, split_held = (), ()
    if split_v5:
        ext_inputs = None
        if split_program:
            # The DEV split program: the extended batch B + F; each MLA layer imports the foreign rows, runs the foreign
            # pass, exports its state and merges the helpers' states.
            table_mla, slot_mla, position_ext, live_ext = split_v5_extended_inputs(
                mla_block_table,
                mla_slot,
                mla_split_table,
                position_rows,
                live_rows,
                SbufTiles(region, heap=True) if region is not None else AUTO_SBUF,
            )
            ext_inputs = (position_ext, live_ext, rows // queries)
        split_outputs, split_held, split_checks = _split_v5_rider(
            sources,
            mla_split,
            mla_split_stats,
            mla_split_own,
            mla_split_foreign,
            rank_info,
            dp,
            SbufTiles(region, heap=True) if region is not None else None,
            queries=queries,
            ext_inputs=ext_inputs,
        )
        if split_program:
            split_body = {
                "form": "v5",
                "dp_group": dp_group,
                "dp": dp,
                "foreign_slots": mla_split["foreign_slots"],
                # The step check's results, filled when layer 0's x gather is traced.
                "checks": split_checks,
                "ext": rows // queries,
                "position": position_ext,
                "live": live_ext,
                **({"guards": True} if mla_split["program"] == "one" else {}),
                **({"ag_priority": mla_split["ag_priority"]} if mla_split.get("ag_priority") is not None else {}),
            }
    if mla_split is not None and not split_v5:
        split_group = split_pair_group(world_size)
        split_sb = SbufTiles(region, heap=True)
        split_info_eff, split_aux = split_handshake(mla_split_info, rank_info, split_group, split_sb)
        table_mla, slot_mla, position_ext, live_ext = split_extended_inputs(
            mla_block_table, mla_slot, mla_split_table, position_rows, live_rows, mla_split_info, split_sb
        )
        split_body = {"group": split_group, "info": split_info_eff, "position": position_ext, "live": live_ext}
    mla_state = sources.get("verify_mla_state")
    # Two persistent HBM pairs alternate by KDA layer, as in the static kimi_k3_verify_model, on the loop path too:
    # per-call pairs would add two DRAM tensors to the DGE table for each KDA layer.
    # ``kda_ar`` (pair_shared_cc_dsts): the destinations of both pairs in LNC1 shared HBM; the sources stay private.
    kda_shared = "kda_ar" in pair_shared
    kda_ar_pairs = tuple(
        tuple(
            nl.ndarray(
                (P_MAX, rows, n_tiles),
                dtype=stream.dtype,
                buffer=pair_shared_buffer(kda_shared and kind == "dst"),
                name=f"verify_kda_ar_{kind}{i}",
            )
            for kind in ("src", "dst")
        )
        for i in range(2)
    )
    if kda_shared:
        # The KDA all-reduce runs over the TP32 group, which must hold both cores of every HBM stack.
        for _, dst in kda_ar_pairs:
            check_pair_group(dst, tp_group, "verify_kda_ar")
    # kda_out_ar_sbuf: one persistent SBUF source of the KDA output all-reduce for the call (a region-heap tile, so no
    # other tile takes its bytes while the collective and its peers read it).
    kda_ar_sb = None
    # The KDA verify calls' all-ones tiles, set once for the kernel (kda_verify_constants): the region's fp32 ones and a
    # bf16 ones heap tile, in place of the two memsets of each call; the forms of KDA_VERIFY_CONSTANTS_FORM_SET only.
    consts_form = body_form in KDA_VERIFY_CONSTANTS_FORM_SET
    kda_consts = (
        kda_verify_constants(region_ones(region), SbufTiles(region, heap=True))
        if region is not None and consts_form
        else None
    )
    if kda_out_ar_sbuf:
        kernel_assert(region is not None, "kda_out_ar_sbuf needs the loop path (a region) for its persistent source")
        # The source is held to the call end (hold_to_end): the last all-reduce reads it until its completion.
        kda_ar_sb = SbufTiles(region, heap=True).tile((P_MAX, rows, n_tiles), stream.dtype, "verify_kda_ar_sb_src")
    # The moe_glue key kda_slot_ring_per_call (KDA_SLOT_RING_PER_CALL_BY_FORM): a per-call ring of Q tiles in place of the
    # persistent ring (sequential states chain with the per-token tiles, no shared ring, no per-token persistent ring).
    slot_ring_per_call = (moe_glue or {}).get("kda_slot_ring_per_call", False)
    kernel_assert(
        isinstance(slot_ring_per_call, bool)
        and (
            not slot_ring_per_call
            or (
                kda_replay == "states"
                and kda_chain == "sequential"
                and bool(kda_token_tiles)
                and not kda_shared_state_ring
                and not kda_slot_state_per_token
            )
        ),
        "moe_glue kda_slot_ring_per_call must be a bool and needs kda_replay 'states', kda_chain 'sequential', "
        f"kda_token_tiles, no kda_shared_state_ring and no kda_slot_state_per_token, got {slot_ring_per_call!r}",
    )
    kda_commit_after_w_in = (moe_glue or {}).get("kda_commit_after_w_in", False)
    kernel_assert(
        isinstance(kda_commit_after_w_in, bool),
        f"moe_glue kda_commit_after_w_in must be a bool, got {kda_commit_after_w_in!r}",
    )
    # The moe_glue key kda_slot_store_regs (KDA_SLOT_STORE_REGS_BY_FORM): the slot page ids in registers ahead of the
    # slot state stores (states replay, one Sync slot-store queue).
    kda_slot_store_regs = (moe_glue or {}).get("kda_slot_store_regs", False)
    kernel_assert(
        isinstance(kda_slot_store_regs, bool)
        and (
            not kda_slot_store_regs
            or (kda_replay == "states" and kda_slot_store_queues == 1 and kda_slot_store_engine == "sync")
        ),
        "moe_glue kda_slot_store_regs must be a bool and needs kda_replay 'states', kda_slot_store_queues 1 and "
        f"kda_slot_store_engine 'sync', got {kda_slot_store_regs!r}",
    )
    # The moe_glue key kda_slot_conv_after_ar (KDA_SLOT_CONV_AFTER_AR_BY_FORM): the slot conv-state stores after the
    # output all-reduce (states replay).
    kda_slot_conv_after_ar = (moe_glue or {}).get("kda_slot_conv_after_ar", False)
    kernel_assert(
        isinstance(kda_slot_conv_after_ar, bool) and (not kda_slot_conv_after_ar or kda_replay == "states"),
        f"moe_glue kda_slot_conv_after_ar must be a bool and needs kda_replay 'states', got {kda_slot_conv_after_ar!r}",
    )
    # The moe_glue key kda_weight_dma: the queue of the KDA verify calls' output-projection, small-weight and parameter
    # loads (tkg_kda.KDA_WEIGHT_DMAS).
    kda_weight_dma = (moe_glue or {}).get("kda_weight_dma", "pool")
    kernel_assert(
        kda_weight_dma in KDA_WEIGHT_DMAS,
        f"moe_glue kda_weight_dma must be one of {KDA_WEIGHT_DMAS}, got {kda_weight_dma!r}",
    )
    # The sink of the per-call ring's keep-reads: one tile for the kernel, held to the end (hold_to_end below), so the
    # compiler keeps the reads.
    ring_keep_sink = None
    if slot_ring_per_call:
        sink_dt = nl.bfloat16 if kda_state_dtype == "bf16" else nl.float32
        sink_shape = (1, 2) if sink_dt == nl.bfloat16 else (1, 1)
        if region is not None:
            ring_keep_sink = SbufTiles(region, heap=True).tile(sink_shape, sink_dt, "verify_kda_ring_keep_sink")
        else:
            ring_keep_sink = nl.ndarray(sink_shape, dtype=sink_dt, buffer=nl.sbuf, name="verify_kda_ring_keep_sink")
        nisa.memset(dst=ring_keep_sink, value=0.0, engine=nisa.engine.gpsimd)
    # KDA_SLOT_RING_KEEP_DEFER: each KDA call's keep-reads at the next KDA call's start (the record), the last at the end.
    ring_keep_defer = [] if slot_ring_per_call and KDA_SLOT_RING_KEEP_DEFER else None
    # Persistent slot-state ring of the sequential states chain, shared across the KDA calls and the blocks
    # (nki-dma-source-liveness): a ring tile keeps its bytes past the store trigger, and the waits record carries the
    # cross-call edges, as kda_conv_rows_tiles / kda_conv_waits do for the conv-state stores. The tiles take the state
    # dtype of the pages (kda_state_dtype); off the region the body keeps the within-call ring only.
    kda_slot_state_ring = None
    kda_slot_state_waits = None
    if (
        kda_replay == "states"
        and kda_chain in ("sequential", "batched")
        and region is not None
        and not slot_ring_per_call
    ):
        # The batched chain writes the Q slot states at once: one ring tile per token, as kda_slot_state_per_token.
        depth = queries if kda_chain == "batched" or kda_slot_state_per_token else kda_slot_state_depth
        slot_tiles = kda_slot_state_tiles(
            rows // queries,
            kda_local_heads(kda_w_out[0], kda_w_small[0]),
            kda_state_dtype,
            SbufTiles(region, heap=True),
            depth=depth,
        )
        kda_slot_state_ring = tuple(slot_tiles[f"kda_slot_state_{r}"] for r in range(depth))
        kda_slot_state_waits = {r: [] for r in range(depth)}
    # Persistent sources of the slot conv-state stores (one set for the kernel, as the slot-state ring), with the
    # record of the last stores that read them.
    kda_slot_conv_rows = None
    kda_slot_conv_waits = None
    if kda_slot_conv_persistent:
        kernel_assert(
            kda_replay == "states" and kda_chain == "sequential" and region is not None,
            "kda_slot_conv_persistent needs kda_replay 'states', kda_chain 'sequential' and the explicit SBUF region",
        )
        kda_slot_conv_rows = kda_slot_conv_rows_tiles(rows, SbufTiles(region, heap=True))
        kda_slot_conv_waits = []
    # One zero tile of the conv-state scatters for the kernel: the off writes of every KDA layer's scatters take their
    # zeros from it, so no layer zeroes a tile of its own (768 bytes a partition on the heap).
    kda_conv_zero_rows = None
    if (kda_batched_conv_stores or kda_batched_slot_conv_stores) and region is not None:
        kda_conv_zero_rows = kda_conv_zero_rows_tile(SbufTiles(region, heap=True))
    # One page set for each KV cache group of the KDA layers (kda_page_groups): the first layer of a group fills it.
    kda_page_sets = None
    if kda_page_groups is not None:
        kernel_assert(
            kda_replay == "states" and region is not None and (not kda_prework_ahead or prework_composed),
            "kda_page_groups keeps the page metadata of the states verify on the explicit SBUF region (the loop "
            "path), with the inline prework or the composed prework of a form of KDA_PREWORK_AHEAD_FORM_SET",
        )
        page_heads = kda_local_heads(kda_w_out[0], kda_w_small[0])
        kda_page_sets = [
            kda_page_group_tiles(
                rows // queries,
                queries,
                page_heads,
                SbufTiles(region, heap=True),
                f"kda_pg{group}",
                conv_load_gather=bool(kda_conv_load_gather),
                batched_conv_stores=bool(kda_batched_conv_stores),
                batched_slot_conv_stores=bool(kda_batched_slot_conv_stores),
                per_token=page_heads == 1 or bool(kda_token_tiles) or kda_chain == "batched",
            )
            for group in range(len(set(kda_page_groups)))
        ]
    eps_kw = {} if lora_norm_eps == LORA_NORM_EPS else {"lora_norm_eps": lora_norm_eps}
    w_in_ring = kda_w_in_buffers or _w_in_buffers(
        kda_local_heads(kda_w_out[0], kda_w_small[0]), kda_state_page.shape[1]
    )
    in_proj_x_stationary = (moe_glue or {}).get("kda_in_proj_x_stationary", False)
    kernel_assert(
        isinstance(in_proj_x_stationary, bool),
        f"moe_glue kda_in_proj_x_stationary must be a bool, got {in_proj_x_stationary!r}",
    )
    state_loads_after_ring = (moe_glue or {}).get("kda_state_loads_after_ring", False)
    kernel_assert(
        isinstance(state_loads_after_ring, bool),
        f"moe_glue kda_state_loads_after_ring must be a bool, got {state_loads_after_ring!r}",
    )
    # KDA_PREWORK_AHEAD_FORM_SET: the prework traces the loads, and the body runs their selects after the quantize.
    state_loads_after_ring = state_loads_after_ring and not prework_composed
    kernel_assert(
        not state_loads_after_ring or kda_state_loads_first,
        "moe_glue kda_state_loads_after_ring orders the loads of kda_state_loads_first: give kda_state_loads_first",
    )
    # The moe_glue key kda_w_out_after_w_in_fills: the KDA w_out load after the last MXFP8 kda_w_in fills (bitwise).
    w_out_after_fills = (moe_glue or {}).get("kda_w_out_after_w_in_fills", False)
    kernel_assert(
        isinstance(w_out_after_fills, bool),
        f"moe_glue kda_w_out_after_w_in_fills must be a bool, got {w_out_after_fills!r}",
    )
    kernel_assert(
        not w_out_after_fills or kda_w_in_scale is not None,
        "moe_glue kda_w_out_after_w_in_fills orders the KDA w_out load after the MXFP8 kda_w_in fills: give "
        "kda_w_in_scale",
    )
    w_in_prefetch = trn3_kda_w_in_prefetch(kda_w_in_prefetch, w_in_ring, kda_w_in_prefetch_chunks)
    ffn_calls = itertools.count()
    # The k1 MLA calls of the kernel, in order (mla_qkv_parity takes the qkv pair of a call by its parity).
    mla_calls = itertools.count()
    for start, count in model_layer_groups(first_layer, num_layers):
        li = start - first_layer
        counts = block_layer_counts(start, count)
        kda_k, mla_k, moe_k, dense_k = starts
        n_kda, n_mla, n_moe, n_dense = counts
        bodies = verify_2srv_bodies(
            queries,
            _rows(kda_state_page, kda_k, n_kda),
            _rows(kda_record_page, kda_k, n_kda),
            kda_replay_len,
            kda_state_position,
            rows,
            region,
            world_size,
            tp_group,
            mla_mode=mla_mode,
            mla_sources=mla_sets,
            mla_state=mla_state,
            routed_tp=routed_tp,
            moe_gather=moe_gather,
            wide_half_weight_buffers=(
                WIDE_HALF_WEIGHT_BUFFERS if moe_wide_half_weight_buffers is None else moe_wide_half_weight_buffers
            ),
            expert_call_tokens=moe_expert_call_tokens,
            **({"expert_queues": moe_expert_queues} if moe_expert_queues != 1 else {}),
            routed_reduce=moe_routed_reduce,
            stats_gather=moe_routed_stats_ag,
            rows_latent=rows_latent,
            **({"rows_load_split": moe_rows_load_split} if moe_rows_load_split != 1 else {}),
            **({"routed_own_rows": True} if moe_routed_own_rows else {}),
            **({"latent_gather_mx": True} if moe_latent_gather_mx else {}),
            **({"moe_glue": dict(moe_glue)} if moe_glue else {}),
            **({"drop_selection": True} if moe_router_rows_drop_selection else {}),
            **({"out_proj_form": moe_out_proj_form} if moe_out_proj_form != "reduced" else {}),
            **({"mx_tile": moe_mx_tile} if moe_mx_tile is not None else {}),
            **({"routed_dynamic": True} if moe_routed_dynamic else {}),
            **({"routed_plan": (routed_plan, routed_layers[0])} if routed_plan is not None else {}),
            **({"routed_dynamic_csr": moe_dynamic_csr} if moe_dynamic_csr is not None else {}),
            kda_ar=[kda_ar_pairs[(kda_k + i) % 2] for i in range(n_kda)],
            # The call's first KDA all-reduce (the first TP collective, verify_first_cc_destinations) stays in HBM.
            **(
                {"kda_ar_sbuf": [None if kda_k + i == 0 else kda_ar_sb for i in range(n_kda)]}
                if kda_ar_sb is not None
                else {}
            ),
            ffn_calls=ffn_calls,
            mla_calls=mla_calls,
            kda_replay=kda_replay,
            kda_state_slot_page=None if kda_state_slot_page is None else _rows(kda_state_slot_page, kda_k, n_kda),
            kda_write_state_slots=None if kda_write_state_slots is None else _rows(kda_write_state_slots, kda_k, n_kda),
            kda_prework_ahead=kda_prework_ahead,
            kda_w_in_buffers=kda_w_in_buffers,
            kda_state_loads_first=kda_state_loads_first,
            kda_slot_pages_first=kda_slot_pages_first,
            **({"kda_token_tiles": True} if kda_token_tiles else {}),
            **({"kda_slot_store_queues": kda_slot_store_queues} if kda_slot_store_queues != 1 else {}),
            **({"kda_slot_store_engine": kda_slot_store_engine} if kda_slot_store_engine != "sync" else {}),
            kda_conv_dtype=kda_conv_dtype,
            kda_chain=kda_chain,
            kda_slot_state_ring=kda_slot_state_ring,
            **(
                {"kda_slot_ring_per_call": True, "kda_slot_ring_keep_sink": ring_keep_sink}
                if slot_ring_per_call
                else {}
            ),
            **({"kda_commit_after_w_in": True} if kda_commit_after_w_in else {}),
            **({"kda_slot_store_regs": True} if kda_slot_store_regs else {}),
            **({"kda_slot_conv_after_ar": True} if kda_slot_conv_after_ar else {}),
            **({"kda_weight_dma": kda_weight_dma} if kda_weight_dma != "pool" else {}),
            **({"kda_consts": kda_consts} if kda_consts is not None else {}),
            **({"kda_slot_ring_keep_defer": ring_keep_defer} if ring_keep_defer is not None else {}),
            kda_slot_state_waits=kda_slot_state_waits,
            kda_slot_state_depth=kda_slot_state_depth,
            **({"kda_shared_state_ring": True} if kda_shared_state_ring else {}),
            **({"kda_state_dtype": kda_state_dtype} if kda_state_dtype != "bf16" else {}),
            **({"kda_paired_pages": True} if kda_paired_pages else {}),
            **({"kda_conv_load_gather": True} if kda_conv_load_gather else {}),
            **({"kda_batched_conv_stores": True} if kda_batched_conv_stores else {}),
            **({"kda_null_page": kda_null_page} if kda_null_page else {}),
            **({"kda_update_bcast_depth": kda_update_bcast_depth} if kda_update_bcast_depth != 2 else {}),
            **({"kda_head_pipeline": kda_head_pipeline} if kda_head_pipeline != "off" else {}),
            **({"kda_out_ar_rows": True} if kda_out_ar_rows else {}),
            # The moe_glue key kda_in_proj_x_stationary: the token-stationary MXFP8 KDA input projection (bitwise).
            **({"kda_in_proj_x_stationary": True} if in_proj_x_stationary else {}),
            # The moe_glue key kda_state_loads_after_ring: the KDA state loads after the w_in ring's prefetch (bitwise).
            **({"kda_state_loads_after_ring": True} if state_loads_after_ring else {}),
            **({"kda_slot_pages_contiguous": True} if kda_slot_pages_contiguous else {}),
            **({"kda_batched_slot_conv_stores": True} if kda_batched_slot_conv_stores else {}),
            **({"kda_bf16_update_rows": True} if kda_bf16_update_rows else {}),
            # KDA_BF16_COLS_ONCE_FORM_SET: the update rows tiles zeroed once a call, on the per-token states path.
            **(
                {"kda_bf16_cols_once": True}
                if body_form in KDA_BF16_COLS_ONCE_FORM_SET
                and kda_bf16_update_rows
                and kda_token_tiles
                and kda_replay == "states"
                and kda_chain == "sequential"
                else {}
            ),
            **({"kda_slot_conv_rows": kda_slot_conv_rows} if kda_slot_conv_rows is not None else {}),
            **({"kda_slot_conv_waits": kda_slot_conv_waits} if kda_slot_conv_waits is not None else {}),
            **({"kda_slot_conv_from_inputs": True} if kda_slot_conv_from_inputs else {}),
            **({"kda_conv_products_engine": kda_conv_products_engine} if kda_conv_products_engine != "gpsimd" else {}),
            **({"kda_recurrence_group": kda_recurrence_group} if kda_recurrence_group is not None else {}),
            **({"kda_conv_zero_rows": kda_conv_zero_rows} if kda_conv_zero_rows is not None else {}),
            **(
                {"kda_page_groups": tuple(kda_page_groups[kda_k : kda_k + n_kda]), "kda_page_sets": kda_page_sets}
                if kda_page_sets is not None
                else {}
            ),
            mla_q_hbm=k1_loop,
            mla_out_ag=k1_loop and not mla_out_ar,
            **({"mla_split": split_body} if split_body is not None else {}),
            **({"mla_out_ar": True} if mla_out_ar else {}),
            **({"mla_state_bf16_o": True} if mla_state_bf16_o else {}),
            **({"mla_critical_dma_priority": mla_critical_dma_priority} if mla_critical_dma_priority else {}),
            **({"mla_cache_write_off_sp": True} if mla_cache_write_off_sp else {}),
            **({"mla_gemv_bands": True} if mla_gemv_bands else {}),
            **({"mla_batched_cache_write": True} if mla_batched_cache_write else {}),
            mla_k1_fit=k1_fit,
            mla_zeroed_pages=mla_zeroed_pages,
            mla_late_tail_weights=mla_late_tail_weights,
            **({"mla_pass_q_prefetch": mla_pass_q_prefetch} if mla_pass_q_prefetch else {}),
            **({"mla_q_gather_buffers": mla_q_gather_buffers} if mla_q_gather_buffers != 1 else {}),
            **({"mla_tail_weights_after_rs": True} if mla_tail_weights_after_rs else {}),
            **({"mla_pass_rows_ahead": True} if mla_pass_rows_ahead else {}),
            **({"mla_late_o_proj_weights": True} if mla_late_o_proj_weights else {}),
            **({"mla_late_tail_frames": True} if mla_late_tail_frames else {}),
            **({"mla_k1_rows": True} if mla_k1_rows else {}),
            **({"mla_k1_rows_options": dict(mla_k1_rows_options)} if mla_k1_rows and mla_k1_rows_options else {}),
            **({"live_rows": live_rows} if live_rows is not None else {}),
            **({"mla_k1_packed": True} if mla_k1_packed else {}),
            **({"mla_k1_pack_options": dict(mla_k1_pack_options)} if mla_k1_pack_options else {}),
            **eps_kw,
            **({} if k7_levers else k7_off),
            **({"kda_x_quant_hook": True} if x_quant_hook else {}),
            **({"kda_merged_state_loads": True} if kda_merged_state_loads else {}),
            **({"kda_w_out_after_w_in_fills": True} if w_out_after_fills else {}),
            **({"kda_prework_deferred": True} if prework_composed else {}),
        )
        if w_in_prefetch is not None:
            w_in_prefetch.set_next_block(
                start + count, first_layer + num_layers, kda_w_in, kda_w_in_scale, kda_k + n_kda, totals[0]
            )
        s, bank, gate = decode_block_body(
            s,
            bank,
            _rows(norm_weights, li, count),
            _rows(attn_res_weights, li, count),
            tuple(_group_stack(t, kda_k, n_kda, totals[0]) for t in kda_weights),
            tuple(_group_stack(t, mla_k, n_mla, totals[1]) for t in mla_weights),
            tuple(_group_stack(t, moe_k, n_moe, totals[2]) for t in moe_weights),
            tuple(_group_stack(t, dense_k, n_dense, totals[3]) for t in dense_weights),
            layer_pools[li : li + count],
            _LayerIndex(),
            _rows(table_mla, mla_k, n_mla),
            _rows(slot_mla, mla_k, n_mla),
            position_rows,
            rank_info,
            tp_group,
            dp_group,
            world_group,
            start,
            eps,
            *bodies,
            None if layer_streams is None else _rows(layer_streams, li, count),
            None,
            dma_gates,
            gate,
            mla_fp8_pages=mla_fp8_pages,
            region=region,
            mla_loop_path=mla_loop_path,
            after_layer=store_aux,
            before_attention=store_mix,
            sources=sources,
            last_block=start + count == first_layer + num_layers,
            kernel_first_layer=first_layer,
            moe_half_weight_buffers=moe_half_weight_buffers,
            **({"moe_weight_buffers": moe_weight_buffers} if moe_weight_buffers is not None else {}),
            **({"moe_scale_buffers": moe_scale_buffers} if moe_scale_buffers is not None else {}),
            **({"moe_transpose_lookahead": moe_transpose_lookahead} if moe_transpose_lookahead is not None else {}),
            **({"moe_cross_expert_order": True} if moe_cross_expert_order else {}),
            **({"moe_transpose_block": True} if moe_transpose_block else {}),
            **({"moe_sums_after_routed": True} if moe_sums_after_routed else {}),
            **({"row_sums_after_ffn_output": True} if attn_res_sums_after_moe_output else {}),
            **({"moe_static_alt_tiles": moe_static_alt_tiles} if moe_static_alt_tiles else {}),
            moe_routed_store_form=moe_routed_store_form,
            dp_size=dp,
            early_ffn_gate=bool(early_ffn_gate),
            **({"ffn_weights_after_attn": True} if ffn_weights_after_attn else {}),
            verify_bank=verify_bank,
            **(
                {"moe_prefetch": dict(moe_expert_prefetch), "moe_prefetch_priority": PREFETCH_DMA_PRIORITY}
                if moe_expert_prefetch
                else {}
            ),
            **({"moe_prefetch_kda_after_in_proj": True} if prefetch_after_in_proj else {}),
            **({"moe_prefetch_kda_queue_order": True} if prefetch_after_in_proj and ring_order else {}),
            **({"moe_prefetch_kda_before_slot_stores": True} if prefetch_after_in_proj and ring_before_stores else {}),
            **({"moe_prefetch_uses": dict(moe_expert_prefetch_uses)} if moe_expert_prefetch_uses else {}),
            **({"moe_prefetch_scales": dict(moe_expert_prefetch_scales)} if moe_expert_prefetch_scales else {}),
            **({"moe_prefetch_start": dict(moe_expert_prefetch_start)} if moe_expert_prefetch_start else {}),
            **({"attn_res_cached_bank_sums": True} if attn_res_cached_bank_sums else {}),
            **({"attn_res_early_sums": True} if attn_res_early_sums else {}),
            **({"attn_res_mla_mlp_early": True} if attn_res_mla_mlp_early else {}),
            **({"attn_res_weight_rows": True} if attn_res_weight_rows else {}),
            **({"kda_states_after_out_ar": True} if kda_states_after_out_ar else {}),
            **({"rows_hT_batched": True} if rows_hT_batched else {}),
            **({"attn_res_ahead": True} if attn_res_ahead else {}),
            **({"attn_res_ahead_mlp": True} if attn_res_ahead_mlp else {}),
            **({"attn_res_ahead_min_entries": attn_res_ahead_min_entries} if attn_res_ahead_min_entries else {}),
            # The moe_glue key ahead_sums_vector (tkg_moe_2srv.GLUE_KEYS): the engine plan of the AttnRes bank parts.
            **({"attn_res_ahead_vector": ahead_sums_vector} if ahead_sums_vector != "off" else {}),
            # The moe_glue key ahead_load_tiles: the entry-load tiles of the AttnRes bank parts.
            **({"attn_res_ahead_load_tiles": ahead_load_tiles} if ahead_load_tiles != 2 else {}),
            # The moe_glue key ahead_before_resid: the residual add after the bank parts' weighted sums.
            **({"attn_res_ahead_before_resid": True} if ahead_before_resid else {}),
            # The moe_glue key ahead_mlp_kda_tail: a KDA layer's own MLP bank part after its token chain.
            **({"attn_res_ahead_mlp_kda_tail": True} if ahead_mlp_kda_tail else {}),
            # The moe_glue key kda_mlp_tail_entries: the first entries of a KDA layer's MLP bank sums after its tail.
            **({"attn_res_kda_mlp_tail_entries": kda_mlp_tail_entries} if kda_mlp_tail_entries else {}),
            # The moe_glue key mla_mlp_tail_entries: the first entries of an MLA layer's MLP bank sums in its out-AR flight.
            **({"attn_res_mla_mlp_tail_entries": mla_mlp_tail_entries} if mla_mlp_tail_entries else {}),
            # The moe_glue key ahead_gram_merge: the Gram stream sums of the ahead mixes as one matmul a hidden tile.
            **({"attn_res_gram_merge": True} if ahead_gram_merge else {}),
            # The moe_glue key norm_one_mul: one weight multiply for all rows of each norm after a mix.
            **({"norm_one_mul": True} if norm_one_mul else {}),
            # The moe_glue key cached_act_squares: the ScalarE stream squares of the cached AttnRes mixes.
            **({"attn_res_cached_act_squares": True} if cached_act_squares else {}),
            # KDA_SLOT_RING_KEEP_DEFER: the KDA keep-reads of the per-call slot-state ring, right after the next MoE layer's
            # MXFP8 in-projection loads (its glue hook), so they hold no MoE load behind them on GpSimd.
            **(
                {
                    "moe_in_proj_hook": lambda loads, record=ring_keep_defer, sink=ring_keep_sink: kda_flush_ring_keeps(
                        record, sink, after=tuple(loads) if loads else None
                    )
                }
                if ring_keep_defer is not None
                else {}
            ),
            **({"norm_scalar_squares": norm_scalar_squares} if norm_scalar_squares else {}),
            **({"attn_res_scalar_products": attn_res_scalar_products} if attn_res_scalar_products else {}),
            **({"attn_res_scalar_products_stt": True} if attn_res_scalar_products and mix_stt else {}),
            **({"attn_res_bank_dma": attn_res_bank_dma} if attn_res_bank_dma else {}),
            **(
                {"attn_res_kda_mlp_early_entries": attn_res_mlp_early_entries, "attn_res_kda_mlp_early_hbm": True}
                if attn_res_mlp_early_entries
                else {}
            ),
            **router_block_options(router_dtype, moe_router_weight, moe_k, n_moe, totals[2]),
            **in_proj_block_options(moe_in_proj_scale, moe_k, n_moe, totals[2]),
            **out_proj_block_options(moe_out_proj_scale, moe_k, n_moe, totals[2]),
            **({"kda_w_in_prefetch": w_in_prefetch} if w_in_prefetch is not None else {}),
            **({"moe_pad_rows": moe_pad_rows} if moe_pad_rows is not None else {}),
            **({"kda_small_after_out_ar": True} if kda_small_after_out_ar else {}),
        )
        starts = [k + n for k, n in zip(starts, counts, strict=True)]

    if final_attn_res_weight is not None or final_norm_weight is not None:
        ones = new_ones() if region is None else region.ones
        if final_attn_res_weight is not None:
            bank_hT, n_bank = bank[:2]
            bank_view = bank_hT if n_bank == n_entries else bank_hT[0:P_MAX, 0:n_bank]
            with row_sums_after_ffn_output(attn_res_sums_after_moe_output):
                s = attn_res_hT(bank_view if n_bank else None, s, load_vector_hT(final_attn_res_weight), eps, ones)
        if final_norm_weight is not None:
            # The final hidden takes the FFN input tile, so the collective tiles stay reserved through lm_head.
            final_x = layer_sources(sources, first_layer + num_layers)["ffn_x"]
            s = rmsnorm_hT(s, load_vector_hT(final_norm_weight), eps, ones, out=final_x)

    out = nl.ndarray((rows, H), dtype=stream.dtype, buffer=nl.shared_hbm, name="stream_out")
    bank_out = fence_sb = None
    if return_attn_res_bank:
        bank_out = nl.ndarray(bank_shape, dtype=stream.dtype, buffer=nl.shared_hbm, name="attn_res_bank_out")
        # The outputs in return order; the call has no tokens and no logits.
        fence_outputs = (out, *(t for t in (aux_out, layer_streams) if t is not None), bank_out, *pools)
        fence_sb = fence_tiles(fence_outputs, stream.dtype)
    store_hT_rows(out, s)
    if return_attn_res_bank:
        # No tile comes after the bank stores, so none takes the bytes of an SBUF bank source.
        for c in range(n_entries):
            nisa.dma_copy(dst=bank_out[0:P_MAX, c], src=bank[0][0:P_MAX, c], name=f"attn_res_bank_out_{c}")
    tail = ()
    if lm_head_weight is not None and (greedy or return_logits):
        logits = None
        if return_logits:
            logits = nl.ndarray(
                (rows, lm_head_weight.shape[1]), dtype=stream.dtype, buffer=nl.shared_hbm, name="logits"
            )
        with weight_gates(dma_gates):
            # The lm_head weights wait for the output collectives of the last feed-forward body.
            set_weight_gate(gate)
            tiles = (sources["argmax_src"], sources["argmax_dst"], sources["argmax_token"]) if greedy_tail else None
            grammar_args = {}
            if grammar_on:
                grammar_rows = nl.ndarray(
                    (rows // queries, queries), dtype=nl.int32, buffer=nl.shared_hbm, name="grammar_rows"
                )
                grammar = GrammarMask(
                    grammar_ids, grammar_rows_in, grammar_class_of, grammar_next, grammar_mask_bias, grammar_rows
                )
                grammar_args = {"grammar": grammar, "grammar_queries": queries}
            token = lm_head_greedy_hT(
                s,
                lm_head_weight,
                tp_group,
                "tail",
                sample=greedy,
                logits_out=logits,
                tp_size=TAIL_TP_RANKS,
                gather_tiles=tiles,
                chunk_logits=rows > 16,
                **grammar_args,
            )
        if greedy:
            tokens = nl.ndarray((rows, 1), dtype=nl.int32, buffer=nl.shared_hbm, name="tokens")
            nisa.dma_copy(dst=tokens, src=token, name="tail_tokens_store")
            tail += (tokens,)
        if return_logits:
            tail += (logits,)
    if aux_out is not None:
        tail += (aux_out,)
    if layer_streams is not None:
        tail += (layer_streams,)
    if bank_out is not None:
        tail += (bank_out,)
    if grammar_on:
        tail += (grammar_rows,)
    if ring_keep_defer:
        # The keep-reads of the last KDA call's ring (KDA_SLOT_RING_KEEP_DEFER), before the sink's hold.
        kda_flush_ring_keeps(ring_keep_defer, ring_keep_sink)
    # Keep: a first-of-step collective skips its handshake, so no later tensor may take its destination's bytes.
    hold_to_end(
        [
            *verify_first_cc_destinations(sources, kda_ar_pairs, first_layer, first_rows),
            *([kda_ar_sb] if kda_ar_sb is not None else []),
            *([ring_keep_sink] if ring_keep_sink is not None else []),
            *split_held,
        ],
        list(tail),
        out,
    )
    outputs = (out, *tail, *pools)
    if fence_sb is not None:
        # The last instructions of the call: the boundary fence reads every output it returns.
        outputs = (out, *tail, boundary_fence(outputs, world_group, fence_sb), *pools)
    if region is not None:
        check_identities(region)
    if split_aux is not None:
        outputs = (*outputs, split_aux)
    return (*outputs, *split_outputs)
