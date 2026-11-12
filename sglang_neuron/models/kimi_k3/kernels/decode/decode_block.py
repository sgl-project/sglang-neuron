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

"""Kimi-K3 decode block giga-kernel: a block of decoder layers in one NEFF.

The body keeps the residual stream and the AttnRes bank of the own DP replica (B <= 32 sequences) in
SBUF in the token-major hT layout ``[128, B, H/128]``.
For each layer it runs the attention mix, the snapshot push, the input norm, the attention body (KDA
or MLA), the residual, the MLP mix, the post-attention norm, and the feed-forward body (dense MLP or
latent MoE). The attention and feed-forward bodies run their own collectives.

The layer schedule is static: the global index of the first layer and the layer count fix the type
of each layer at trace time.
"""

import functools
import math
from contextlib import contextmanager, nullcontext

import nki
import nki.isa as nisa
import nki.language as nl
from nki.collectives import ReplicaGroup

from .utils.kernel_assert import kernel_assert
from .moe.routed import STATIC_ALT_TILE_FORMS, expert_loop_layer
from .utils.trace_names import restart_calls
from .common import (
    BANK_DMAS,
    COMPACT_MIX_TOKENS,
    P_MAX,
    PAIR_SHARED_SOURCES_KEY,
    SplitBank,
    WeightRow,
    attn_res_ahead_hT,
    attn_res_ahead_parts,
    attn_res_ahead_parts_staged,
    attn_res_bank_part,
    attn_res_hT,
    attn_res_plain_bank_part,
    attn_res_tensor_hT,
    attn_res_tensor_weights,
    check_pair_shared_cc_dsts,
    dma_priority,
    ffn_output_order,
    is_trn3,
    load_rows_hT,
    load_rows_hT_batched,
    new_ahead_part,
    new_ones,
    one_row_moe_glue,
    rmsnorm_hT,
    row_rstd_hT,
    row_sums_order,
    store_hT_rows,
    take_ffn_output,
    tp_stream_gather,
)
from .tkg_weight_dma_gate import (
    PrefetchGate,
    PrefetchLoad,
    clear_pool_after,
    gate_all_dmas,
    pool_after,
    released,
    set_pool_after,
    set_weight_gate,
    take_released,
    take_released_pair,
    weight_gates,
)
from .tkg_sbuf_alloc import AUTO_SBUF, ExplicitSbuf, SbufTiles, check_identities, explicit_ones, install_identities
from .tkg_kda import (
    KDA_WEIGHT_NAMES,
    _schedule_after,
    after_w_in_loads,
    kda_collective_hbm,
    kda_conv_index_tiles,
    kda_conv_rows_tiles,
    kda_decode_attention,
    kda_local_heads,
    kda_read_scatter_keep,
    kda_w_in_head_bytes,
    kda_w_in_head_loads,
    prefetch_w_in_ring,
)
from .tkg_mla_block import CP_TARGET_HEADS, mla_block_collective_hbm, mla_block_collective_sources, mla_cp_decode_attention
from .tkg_mla_paged import (
    LORA_NORM_EPS,
    MLA_MX_SCALES,
    MLA_SYNC_SITES,
    MLA_WEIGHT_NAMES,
    QUERY_SET_TOKENS,
    mla_collective_hbm,
    mla_collective_sources,
    mla_decode_attention,
    mla_pack_plan,
    mla_pack_step_unit,
    mla_runs_loops,
    mla_stage_collective,
    mla_stage_hbm,
    mla_stage_views,
    mla_state_sum_hbm,
    pack_tail_chain,
    prefetch_mla_head,
)
from .tkg_mla_paged import MAX_TOKENS as MLA_GROUP_TOKENS
from .tkg_moe import (
    BASE_GLUE_DMA_FORMS,
    COLUMN_FORMS,
    DP_SIZE,
    IN_PROJ_FORMS,
    LATENT_SHARD_FORMS,
    OUT_PROJ_FORMS,
    ROUTED_KERNELS,
    ROUTED_REDUCE_DTYPES,
    ROUTER_DTYPES,
    TP128_KERNELS,
    TP_SIZE,
    WIDE_FORMS,
    WORLD_SIZE,
    active_expert_counts,
    base_moe_body,
    dense_mlp_decode,
    down_group_mask,
    dynamic_expert_loop,
    ffn_layer_sources,
    moe_collective_hbm,
    moe_collective_tiles,
    moe_collective_views,
    moe_gather_src_tile,
    moe_options_in_use,
    moe_ring_piece_bytes,
    moe_x_dst_tile,
    out_proj_collective_hbm,
    out_proj_collective_tiles,
    rank_row_one_hot,
    rank_selectors,
    routed_collective_hbm,
    routed_collective_tiles,
    routed_expert_path,
    sel_load_down_groups,
    shared_gather_tiles,
    tp_rank_tile,
    x_gather_alternates,
    x_gather_parity,
)
from .decode_moe_a2av import a2av_moe_hbm
from .tkg_moe_dp import dp_moe_collective_hbm, select_moe_body

NUM_LAYERS = 93
ATTN_RES_BLOCK = 12
FULL_ATTENTION_PERIOD = 4

# Above this many rows the AttnRes bank lives in DRAM, not SBUF: the long verify step gains an entry per
# snapshot layer, so the 93-layer bank is far larger than a decode bank. The mix streams one entry at a time.
# Sequences from which the MLA layers take ``mla_pack_tail_slots``: the staged groups of 8 of 16 sequences and up,
# so the programs of 8 sequences and fewer do not change (unless ``mla_pack_tail_low_batch``).
_PACK_TAIL_MIN_TOKENS = 2 * MLA_GROUP_TOKENS
# attn_res_kda_mlp_early_in_front keeps the conv-state scatter sources in persistent tiles up to this many sequences. At
# 32 sequences the three persistent row tiles (512 bytes per partition each) do not fit beside the expert loop of the
# half-weight buffers (NCC_EGCA111); the persistent index tables alone do.
_KDA_FRONT_ROWS_MAX_TOKENS = 2 * MLA_GROUP_TOKENS
# Above that, the block reads the views of each conv-state scatter (the per-layer source rows and the index tables) once
# more after the feed-forward norm of the layer: the read keeps their bytes until then, so the MLP mix temporaries and
# the MoE in-projection loads that follow the scatter take other bytes and do not wait for the scatter. The reads add
# into small sink tiles that one DMA per block stores to a private HBM tensor, so the compiler keeps them.
# Decode steps keep T <= this, so the bank path and values are unchanged for decode.
_BANK_HBM_ROWS = 2 * QUERY_SET_TOKENS
MLA_STAGE_SPLITS = {"off": (), "ag": ("q",), "ag_rs": ("q", "state")}
"""The forms of ``mla_stage_cc_split``: the staged MLA sites (``tkg_mla_paged.MLA_STAGE_SPLIT_SITES``) that run their
collective once for each unit of groups."""


def mla_stage_unit_tokens(mla_pack_pairs: bool) -> int:
    """Return the sequences of a unit of ``mla_stage_cc_split``: the packed loop of a pair of groups, else one group."""
    return (2 if mla_pack_pairs else 1) * MLA_GROUP_TOKENS


# The collective sources that alternate between two sets by layer parity (``cc_source_pingpong``): SBUF, then HBM.
_PINGPONG_TILES = ("ffn_x", "x16_src", "xl_src")
_PINGPONG_HBM = ("x_src_hbm", "x16_src_hbm", "xl_src_hbm", "dp_rows_src_hbm", "routed_hbm", "kda_ar_hbm")
COLUMN_CHUNK_GATHER_KEYS = ("gather_src_hbm", "gather_dst_hbm", "outg_src_hbm", "outg_dst_hbm")
"""The HBM pairs of the world gathers of ``moe_out_proj_form`` ``column`` that each feed-forward chunk of a layer takes
for itself (``collective_sources`` ``out_sets``, ``gather_hbm_sets``): the latent gather and the output gather."""
SHARED_GATHER_KEYS = ("shared_ag_src", "shared_ag_dst")
"""The SBUF tiles of the shared gather of ``moe_out_proj_form`` ``column_shared_ag`` (``tkg_moe.shared_gather_tiles``)
that each feed-forward chunk of a layer takes for itself (``collective_sources`` ``out_sets``, ``shared_ag_sets``), as
it takes the gather pairs of ``column``."""
MOE_DST_PINGPONG_SITES = ("gather", "x")
"""The feed-forward all-gather sites whose persistent HBM destination can alternate by layer parity
(``collective_sources`` ``dst_pingpong``): ``gather``, the latent world gather (``gather_dst_hbm``), and ``x``, the DP x
gather (``x_dst_hbm``)."""


def dst_pingpong_keys(sites, hbm: dict) -> list:
    """Return the keys of the persistent HBM destinations of ``sites`` (``MOE_DST_PINGPONG_SITES``) in ``hbm``.

    A site with no HBM destination (its gather lands in SBUF, as off the MLA loop path) adds no key.
    """
    sites = tuple(sites)
    kernel_assert(
        all(site in MOE_DST_PINGPONG_SITES for site in sites),
        f"moe_dst_pingpong {sites} must be names of {MOE_DST_PINGPONG_SITES}",
    )
    return [f"{site}_dst_hbm" for site in MOE_DST_PINGPONG_SITES if site in sites and f"{site}_dst_hbm" in hbm]


def check_x_world_gather(mla_stage_cc_split: str, x_dst_pingpong: bool) -> None:
    """Fail closed on the forms of ``moe_x_world_gather`` whose collectives do not fit the collective tables.

    The runtime keeps the control and slot tables of the collectives of a program on chip up to a fixed size, and
    in HBM beyond it. The world x gather adds collective events to each MoE and dense layer. At 32 sequences per
    replica it fits the tables only when each world gather skips its start handshake, which needs a second HBM
    destination (``x_dst_pingpong``: ``"x"`` in ``moe_dst_pingpong``), and only without the per-unit MLA collectives
    of ``mla_stage_cc_split``. Both other forms fail here, at trace time.

    The kernel cannot check one more precondition: the world gather takes each partition of a replica from another TP
    rank of that replica, so its bytes equal the DP gather's only when the TP ranks of a replica hold the same hidden
    (the same replicated weights, and all-reduces that give every rank the same bytes).
    """
    kernel_assert(
        mla_stage_cc_split == "off",
        f"moe_x_world_gather with mla_stage_cc_split {mla_stage_cc_split!r}: the per-unit MLA collectives and the "
        "world x gather together do not fit the collective tables; set mla_stage_cc_split to 'off'",
    )
    kernel_assert(
        x_dst_pingpong,
        "moe_x_world_gather needs the second HBM x gather destination ('x' in moe_dst_pingpong): with one "
        "destination each world x gather keeps its start handshake, and the collectives do not fit the collective "
        "tables",
    )


def check_attn_rs_moe_input(
    rows: int,
    dp_size: int,
    moe_body,
    moe_in_proj_form: str,
    mla_cp: int,
    mla_stage_cc_split: str,
    moe_x_world_gather: bool,
) -> None:
    """Fail closed off the one form of ``attn_rs_moe_input``: the 1-server DP4 x TP32 verify at R = TP_SIZE.

    The reduce-scatter gives each rank exactly one row only when the rows of a replica equal the TP group size, so the
    scheme does not apply at R16 (16 rows over 32 ranks). It runs on the base TP128 MoE body (the world token gather
    fills that body's x gather destination), with the gathered in-projection, no CP MLA, no per-unit MLA collectives,
    and the second x gather destination (``dst_pingpong`` ``"x"``, so the world gather skips its start handshake). It is
    its own world gather, so it does not combine with ``moe_x_world_gather``.
    """
    kernel_assert(dp_size == DP_SIZE, f"attn_rs_moe_input needs the DP{DP_SIZE} layout, got dp_size {dp_size}")
    kernel_assert(
        rows == TP_SIZE,
        f"attn_rs_moe_input needs R = TP_SIZE ({TP_SIZE}) rows per replica (one row per rank after the reduce-scatter); "
        f"got {rows}. It does not apply at R16.",
    )
    kernel_assert(
        getattr(moe_body, "moe_sharding", "tp128") == "tp128" and not getattr(moe_body, "a2av_expert_form", None),
        "attn_rs_moe_input needs the base TP128 MoE body (not the DP4 x TP32 or all-to-all-v body)",
    )
    kernel_assert(moe_in_proj_form == "gathered", "attn_rs_moe_input needs moe_in_proj_form 'gathered'")
    kernel_assert(mla_cp == 0, "attn_rs_moe_input does not apply to the CP128 MLA body")
    kernel_assert(
        mla_stage_cc_split == "off",
        "attn_rs_moe_input and the per-unit MLA collectives do not fit the collective tables; "
        "set mla_stage_cc_split to 'off'",
    )
    kernel_assert(
        not moe_x_world_gather,
        "attn_rs_moe_input brings its own world gather; do not set moe_x_world_gather",
    )


PAIR_SHARED_PRIVATE_EXPERT_PATHS = ("sel_load", "tp_dense")
"""The MoE expert paths (``pair_shared_expert_path``) that keep their per-call routed partial in private HBM when
``pair_shared_cc_dsts`` is not empty: the TP128 routed kernels, which take ``private_out`` from the MoE body in a program
with the sites (``common.PAIR_SHARED_SOURCES_KEY``). Without the sites their partials stay in shared HBM (the default of
the kernels). A caller can check this constant before it gives the sites on these paths."""
PAIR_SHARED_EXPERT_PATHS = (None, "pe", *PAIR_SHARED_PRIVATE_EXPERT_PATHS)
"""The MoE expert paths (``pair_shared_expert_path``) that keep no per-core scratch in shared HBM with the sites:
the multi-server MoE body (None), the base static loop that writes the routed source itself (``pe``), and
``PAIR_SHARED_PRIVATE_EXPERT_PATHS``."""


def check_pair_shared_forms(
    pair_shared_cc_dsts,
    dp_moe: bool = False,
    a2av_moe: str = None,
    mla_stage_split: tuple = (),
    moe_expert_path: str = None,
    token_rs: bool = False,
    x_gather_hbm: bool = False,
    trn2_entry: bool = False,
) -> tuple:
    """Return the sites of ``pair_shared_cc_dsts`` (``common.check_pair_shared_cc_dsts``, ``trn2_entry`` for a Trn2
    entry); fail closed on the forms that the LNC1 shared-HBM destinations are not proven with.

    The neuronx-cc option --lnc1-with-shared-mem shares every ``nl.shared_hbm`` scratch tensor of the program between
    the two cores of an HBM stack. The all-to-all-v MoE body keeps its per-core scratch in ``nl.shared_hbm``
    (``decode_moe_a2av.a2av_moe_hbm``), so with the option the two cores would write one buffer: refused. So is every
    expert path that writes this rank's routed partial to a new ``nl.shared_hbm`` tensor in each call: the static expert
    loop of the MoE options, and the static loop of the base body unless it writes the routed source itself
    (``routed_store_form`` ``pe``), not with the dynamic loop or the active-expert slots (``moe_tkg_decode``). The TP128
    routed kernels (``sel_load``, ``tp_dense``) pass: with the sites in the collective sources
    (``common.PAIR_SHARED_SOURCES_KEY``) the MoE body gives them their ``private_out``, so their routed partials are
    private HBM. ``moe_expert_path`` (``pair_shared_expert_path``) names the path of the caller; None passes (a caller
    whose expert loop keeps no such tensor: the multi-server MoE body). The DP4 x TP32 MoE body and the per-unit MLA
    collectives of
    ``mla_stage_cc_split`` are refused too: their collectives are not measured with shared destinations. So are the
    verify's ``attn_rs_moe_input`` (``token_rs``: the attention outputs reduce-scatter over the TP group, so each core of
    a stack keeps rows of its own) and ``moe_x_gather_hbm`` (``x_gather_hbm``: the x gathers move to HBM, which changes
    the collective order between the latent gathers).
    """
    sites = check_pair_shared_cc_dsts(pair_shared_cc_dsts, trn2_entry=trn2_entry)
    if not sites:
        return sites
    kernel_assert(
        a2av_moe is None,
        f"pair_shared_cc_dsts {sites} with the all-to-all-v MoE exchange: its per-core scratch is in shared HBM, which "
        "--lnc1-with-shared-mem would share between the two cores of a stack",
    )
    kernel_assert(
        moe_expert_path in PAIR_SHARED_EXPERT_PATHS,
        f"pair_shared_cc_dsts {sites} with the MoE expert path {moe_expert_path!r}: it writes each call's routed partial "
        "to a per-core tensor in shared HBM, which --lnc1-with-shared-mem would share between the two cores of a stack; "
        "only the base body's static expert loop with routed_store_form 'pe' and the TP128 routed kernels (sel_load, "
        "tp_dense, with private partials) keep none",
    )
    kernel_assert(not dp_moe, f"pair_shared_cc_dsts {sites} is not proven with the DP4 x TP32 MoE body")
    kernel_assert(
        not mla_stage_split,
        f"pair_shared_cc_dsts {sites} is not proven with the per-unit MLA collectives of mla_stage_cc_split; set it to "
        "'off'",
    )
    kernel_assert(
        not token_rs,
        f"pair_shared_cc_dsts {sites} is not proven with attn_rs_moe_input: its token reduce-scatters give each core of a "
        "stack rows of its own",
    )
    kernel_assert(
        not x_gather_hbm,
        f"pair_shared_cc_dsts {sites} is not proven with moe_x_gather_hbm: its HBM x gathers change the collective order",
    )
    return sites


def pair_shared_expert_path(
    routed_kernel: str,
    moe_options: bool,
    routed_store_form: str,
    dynamic_loop: bool = False,
    active_expert_slots: int = 0,
) -> str:
    """Return the MoE expert path name of ``check_pair_shared_forms`` ``moe_expert_path``: the routed kernel (``sel_load``,
    ``tp_dense``), ``options`` (the static loop of the MoE options), ``dynamic`` / ``slots`` (``moe_tkg_decode``'s dynamic
    loop and active-expert slots), or the base static loop's ``routed_store_form`` (``pe`` or ``relayout``)."""
    if routed_kernel != "static":
        return routed_kernel
    if moe_options:
        return "options"
    if dynamic_loop:
        return "dynamic"
    if active_expert_slots:
        return "slots"
    return routed_store_form


def is_mla_layer(layer: int) -> bool:
    """Return True for an MLA layer (0-indexed global layer)."""
    return (layer + 1) % FULL_ATTENTION_PERIOD == 0 or layer == NUM_LAYERS - 1


def moe_keep_warm_lead(layer: int) -> str:
    """Return the ``keep_warm_lead`` of the MoE body call of ``layer`` (a per-kind ``pe_keep_warm`` of the 2-server
    body): the kind of the layer's attention, ``"mla"`` (``is_mla_layer``) or ``"kda"``."""
    return "mla" if is_mla_layer(layer) else "kda"


def is_dense_layer(layer: int) -> bool:
    """Return True for the dense MLP layer."""
    return layer == 0


def banks_snapshot(layer: int) -> bool:
    """Return True when the layer pushes its input stream into the AttnRes bank."""
    return layer % ATTN_RES_BLOCK == 0


def block_layer_counts(first_layer: int, num_layers: int) -> tuple[int, int, int, int]:
    """Return the KDA, MLA, MoE, and dense layer counts of a block.

    Example:
        >>> block_layer_counts(0, 12)
        (9, 3, 11, 1)
    """
    layers = range(first_layer, first_layer + num_layers)
    n_mla = sum(1 for i in layers if is_mla_layer(i))
    n_dense = sum(1 for i in layers if is_dense_layer(i))
    return num_layers - n_mla, n_mla, num_layers - n_dense, n_dense


# Slots of the TP128 expert tensors in ``tkg_moe.moe_weight_slots`` (packed gate/up, down, down scale).
SEL_LOAD_EXPERT_SLOTS = (3, 5, 6)


def _sel_load_layer_weights(moe_weights, k: int, count: int, base: int):
    """Return the sel_load weight slots of the k-th MoE layer of the block and the ``expert_offset`` kwarg of the body.

    A stack of more than one layer passes whole (the MoE layer ``base + k`` of the stack), with
    ``expert_offset = (base + k) * 896``. A stack of one shared layer passes as its layer. The other slots are the
    views of the layer (``_layer_view``).
    """
    weights = []
    offset = 0
    for i, t in enumerate(moe_weights):
        if t is not None and i in SEL_LOAD_EXPERT_SLOTS:
            weights.append(t if t.shape[0] > 1 else t[0])
            if t.shape[0] > 1:
                offset = (base + k) * t.shape[1]
        else:
            weights.append(_layer_view(t, k, count))
    return tuple(weights), ({"expert_offset": offset} if offset else {})


HOT_EXPERT_GROUP = 16
"""Experts of one hot-expert group (``moe_hot_prefetch``): a group of down rows fills the 128 partitions."""
HOT_SCALE_TILE_GROUPS = 4
"""Hot-expert groups of one down-scale tile (partition 32q + 4l + j holds slot 4q + j of group l)."""
HOT_QUEUE = nisa.engine.scalar
"""The trigger engine of the hot-expert loads: the ACT hardware DGE queue, the least busy one in the attention."""
HOT_TILE_KEYS = ("gate_up", "down", "down_scale")
"""The hot-expert tiles of ``hot_expert_tiles``; each one is a ``PrefetchGate`` slot ``hot_<key>``."""


def check_kda_states_after_out_ar(kda_states_after_out_ar: bool, moe_body, kda_body) -> None:
    """Fail the trace unless the bodies take ``decode_block_body`` ``kda_states_after_out_ar`` (False passes): the
    2-server MoE body (its out_trigger glue hook gives the edge) and a KDA body that takes ``state_loads_after`` (the
    verify body, ``verify_model.verify_bodies``; the decode KDA body does not)."""
    kernel_assert(
        not kda_states_after_out_ar
        or (getattr(moe_body, "moe_sharding", None) == "dp_2srv" and getattr(kda_body, "state_loads_after", False)),
        "kda_states_after_out_ar orders the next KDA layer's state loads after the 2-server MoE body's output "
        "all-reduce trigger (its out_trigger glue hook): it needs the 2-server MoE body and a KDA body that takes "
        "state_loads_after (the verify body, verify_model.verify_bodies)",
    )


def check_moe_hot_prefetch(moe_hot_prefetch, routed_kernel: str, region) -> dict:
    """Return the experts per attention type of ``moe_hot_prefetch`` (``{}`` for None), or fail the trace."""
    if moe_hot_prefetch is None:
        return {}
    kernel_assert(
        isinstance(moe_hot_prefetch, dict) and set(moe_hot_prefetch) == {"kda", "mla"},
        f"moe_hot_prefetch must be a dict with the keys kda and mla, got {moe_hot_prefetch!r}",
    )
    kernel_assert(
        all(
            isinstance(n, int)
            and not isinstance(n, bool)
            and 0 <= n <= 4 * HOT_EXPERT_GROUP
            and n % HOT_EXPERT_GROUP == 0
            for n in moe_hot_prefetch.values()
        ),
        f"moe_hot_prefetch takes whole groups of {HOT_EXPERT_GROUP} experts, at most "
        f"{4 * HOT_EXPERT_GROUP}, got {moe_hot_prefetch}",
    )
    kernel_assert(
        routed_kernel == "sel_load" and region is not None,
        f"moe_hot_prefetch needs the sel_load routed kernel and an explicit SBUF region, got {routed_kernel!r}",
    )
    return dict(moe_hot_prefetch)


def hot_expert_tiles(region: ExplicitSbuf, moe_weights, moe_layer: int, experts: int, tag: str) -> tuple:
    """Return ``(hot_experts, loads)``: the hot-expert tiles of a sel_load MoE layer and, by tile key, their
    ``PrefetchLoad`` items.

    The hottest experts of a layer are its experts 0 .. N - 1 (the loader orders them). The tiles take heap addresses
    of the open frame of ``region``, in the layouts of the selective-load body:

    - ``gate_up`` uint16 [128, N * 384]: words 384 c .. 384 c + 383 of partition p <- packed row p of expert c (one
      DMA per group of 16 experts).
    - ``down`` [128, N / 16 * H] in the down dtype: columns g * H .. of partition m <- row m of the [E * 8, H] down rows
      of the 16 experts of group g (one DMA per group).
    - ``down_scale`` uint8 [128, ceil(N / 64) * H]: columns t * H .. of partition 32q + 4l + j <- the down scale row of
      expert 64 t + 16 l + 4 q + j; the other partitions stay unwritten. One DMA of 4 rows for each (t, q, l): the
      outer HBM dimension of a DMA must be a multiple of its SBUF partitions.

    ``hot_experts`` holds them and ``experts`` = N. ``moe_layer`` is the layer of the stacked weights.
    """
    gate_up, down, down_scale = (moe_weights[i] for i in SEL_LOAD_EXPERT_SLOTS)
    layer = moe_layer if gate_up.shape[0] > 1 else 0
    n_experts = gate_up.shape[1]
    words, hidden = gate_up.shape[-1] // 2, down.shape[-1]
    groups = experts // HOT_EXPERT_GROUP
    tiles = -(-groups // HOT_SCALE_TILE_GROUPS)
    first = layer * n_experts
    hot = {
        "gate_up": region.heap((P_MAX, experts * words), nl.uint16, f"{tag}_hot_gate_up"),
        "down": region.heap((P_MAX, groups * hidden), down.dtype, f"{tag}_hot_down"),
        "down_scale": region.heap((P_MAX, tiles * hidden), down_scale.dtype, f"{tag}_hot_down_scale"),
        "experts": experts,
    }
    rows = gate_up.reshape((gate_up.shape[0] * n_experts, P_MAX, 2 * words)).view(nl.uint16)
    down_rows = down.reshape((down.shape[0] * n_experts * down.shape[2], hidden))
    loads = {key: [] for key in HOT_TILE_KEYS}
    for g in range(groups):
        lead = first + HOT_EXPERT_GROUP * g
        loads["gate_up"].append(
            PrefetchLoad(
                hot["gate_up"][0:P_MAX, HOT_EXPERT_GROUP * words * g : HOT_EXPERT_GROUP * words * (g + 1)],
                rows.ap(
                    pattern=[[words, P_MAX], [P_MAX * words, HOT_EXPERT_GROUP], [1, words]], offset=lead * P_MAX * words
                ),
                HOT_QUEUE,
                name=f"{tag}_hot_gate_up_load_g{g}",
            )
        )
        loads["down"].append(
            PrefetchLoad(
                hot["down"][0:P_MAX, hidden * g : hidden * (g + 1)],
                down_rows[lead * down.shape[2] : lead * down.shape[2] + P_MAX, 0:hidden],
                HOT_QUEUE,
                name=f"{tag}_hot_down_load_g{g}",
            )
        )
    scale_rows = down_scale.reshape((down_scale.shape[0] * n_experts, hidden))
    for t in range(tiles):
        for q in range(4):
            for place in range(min(HOT_SCALE_TILE_GROUPS, groups - t * HOT_SCALE_TILE_GROUPS)):
                part = 32 * q + 4 * place
                row = first + HOT_SCALE_TILE_GROUPS * HOT_EXPERT_GROUP * t + HOT_EXPERT_GROUP * place + 4 * q
                loads["down_scale"].append(
                    PrefetchLoad(
                        hot["down_scale"][part : part + 4, t * hidden : (t + 1) * hidden],
                        scale_rows[row : row + 4, 0:hidden],
                        HOT_QUEUE,
                        name=f"{tag}_hot_down_scale_load_t{t}_q{q}_l{place}",
                    )
                )
    return hot, loads


def keep_hot_tiles(hot: dict, sink: nl.NkiTensor, after: tuple) -> dict:
    """Read one word of each hot-expert tile into ``sink`` after the instructions ``after``; return the read by key.

    For a MoE body that does not take the tiles: the reads keep the tiles and their loads in the program.
    """
    reads = {}
    for i, key in enumerate(HOT_TILE_KEYS):
        read = nisa.tensor_tensor(
            dst=sink[0:1, i : i + 1],
            data1=sink[0:1, i : i + 1],
            data2=hot[key].view(nl.int32)[0:1, 0:1],
            op=nl.add,
            engine=nisa.engine.vector,
        )
        read = read if hasattr(read, "depends_on") else nisa.NkiInstruction(read)
        if after:
            read.depends_on(*after, kind=nisa.dependency.schedule)
        reads[key] = (read,)
    return reads


def _layer_view(stack, k: int, count: int):
    """Return layer ``k`` of a stacked tensor; a leading axis of 1 is shared by all layers. None stays None."""
    if stack is None:
        return None
    return stack[k] if stack.shape[0] == count else stack[0]


KDA_GAP_LOADS = ("off", "w_out", "in_proj", "both")
"""The ``kda_gap_loads`` forms of ``decode_block_body``: which weight loads a KDA layer moves into the gap between the end of
its input projection and its output all-reduce (``w_out``: its own output-projection weight; ``in_proj``: the
in-projection of the next MoE layer; ``both``). ``off`` keeps the program."""


AHEAD_MLP_TOKEN_STEPS = True
"""Kill switch of the token steps of the MLP ahead part (False, not the default, gives the program without the token
steps). True: a KDA layer that builds
the bank part of its own MLP mix in its body (``attn_res_ahead_mlp``: a bank above ``ATTN_RES_AHEAD_SHARED_ENTRIES``, or a
layer that pushes a bank entry) traces the score pass after its attention norm, as without the token steps, and the
weighted sums and the bank norm in the gaps of the KDA token chain (the body's ``token_hook``): the entries in their
order over tokens 0 .. Q - 2, then the rest. The instructions and their order on each engine are those of
``common.attn_res_ahead_parts``, so the part holds the same bits; the build does not queue ahead of the chain's own
VectorE / ScalarE work. Needs a KDA body that takes ``token_hook``."""
ATTN_RES_AHEAD_VECTOR_MODES = ("off", "attn", "all", "all_squares")
"""The values of ``decode_block_body`` ``attn_res_ahead_vector``: the AttnRes bank parts whose GpSimd share runs on
VectorE ("off": none, "attn": the calls that build an attention part, "all": every bank part, "all_squares": every bank
part, with the squares of its norm on VectorE too)."""
ATTN_RES_AHEAD_SHARED_ENTRIES = 4
"""The largest bank whose two ahead parts (the next KDA layer's attention and MLP mixes, ``attn_res_ahead_mlp``) are
built together in the tail of a layer, from one set of entry loads. Above it the two builds do not fit beside the tail's
collectives (1-core device microbench: one part at 8 entries takes most of the collective window), so the MLP
part is built after the attention norm, in the KDA body."""
KDA_W_IN_PREFETCH_PLACES = ("cc_window", "after_ffn_cc", "after_ffn_done")
"""Where a feed-forward layer issues the next KDA layer's ``kda_w_in`` ring (``KdaWInPrefetch`` ``place``): inside the
routed all-reduce of the MoE body (its ``cc_window``, the Trn2 MoE body); after the first output collective of the
feed-forward body (any body, Trn3); after the first output collective, with each fill waiting for the completion of
the last one (``after_ffn_done``, Trn3), so the fills do not share the DMA engines with the output collectives."""

_AFTER_FFN_PLACES = ("after_ffn_cc", "after_ffn_done")


_KDA_W_IN_PREFETCH_PRIORITY = 3
"""The DMA QoS level of the ``after_ffn_cc`` ring fills: the lowest, so they stream in the DMA time of the tail."""


KDA_W_IN_PREFETCH_TRN3_MODES = ("off", "after_ffn_cc", "after_ffn_done")
"""``kda_w_in_prefetch`` of the Trn3 verify entries: off; after the first output collective of each feed-forward body
(``KDA_W_IN_PREFETCH_PLACES``), at DMA QoS level 3; the same after the completion of its last output collective."""


def trn3_kda_w_in_prefetch(mode: str, buffers: int | None, chunks: int = 0, tag: str = ""):
    """Check the ``kda_w_in_prefetch`` keys of a Trn3 entry; return the ``KdaWInPrefetch`` carry, or None (off).

    ``buffers`` is the ring of the KDA calls (their ``w_in_buffers``, None for the whole-weight ring), ``chunks`` the
    chunks of the ring that the feed-forward layer prefetches, from its end (0: all, the only choice for None).
    ``tag`` prefixes the names of the ring loads (a kernel that traces a layer twice gives each trace its own).
    """
    kernel_assert(
        mode in KDA_W_IN_PREFETCH_TRN3_MODES,
        f"kda_w_in_prefetch must be one of {KDA_W_IN_PREFETCH_TRN3_MODES}, got {mode!r}",
    )
    kernel_assert(mode == "off" or is_trn3(), "kda_w_in_prefetch after_ffn_cc needs Trn3 (DMA QoS levels)")
    kernel_assert(
        isinstance(chunks, int) and (chunks == 0 or (mode != "off" and buffers is not None and 0 < chunks <= buffers)),
        f"kda_w_in_prefetch_chunks must be 0 .. {buffers} with a kda_w_in_prefetch mode, got {chunks} ({mode})",
    )
    if mode == "off":
        return None
    first = buffers - chunks if chunks else 0
    return KdaWInPrefetch(buffers, first, place=mode, priority=_KDA_W_IN_PREFETCH_PRIORITY, tag=tag)


class KdaWInPrefetch:
    """The carry of ``decode_block_body`` ``kda_w_in_prefetch``: one object for a kernel (opt-in).

    ``buffers``: the ``kda_w_in`` ring of the KDA layers (None: the whole-weight ring of ``w_in_buffers`` None, with
    ``after_ffn_cc``). ``first``: a feed-forward layer issues the first fills of chunks ``first .. buffers - 1`` of the
    next KDA layer's ring (0: every chunk); the KDA body issues those of the chunks before, with its weight gate, as
    without a prefetch. ``place``: ``KDA_W_IN_PREFETCH_PLACES``. ``priority``: the DMA QoS level of the fills (None: the
    projection's). ``next_w_in`` / ``next_w_scale``: set by the entry before each block call, the HBM ``kda_w_in``
    [H, N] (or MXFP8 data) and its scales (None for BF16) of the first layer of the next block when it is a KDA layer,
    else None. ``pending``: ``(layer, ring)`` of the ring issued for ``layer``. ``tag``: the prefix of the names of
    the ``after_ffn_cc`` ring loads. ``at_trigger``: the window asks the MoE body to issue the fills when the routed
    reduce-scatter starts, not after it (the ``at_trigger`` attribute of the ``cc_window`` function).
    """

    def __init__(
        self,
        buffers: int | None,
        first: int = 0,
        at_trigger: bool = False,
        place: str = "cc_window",
        priority: int = None,
        tag: str = "",
    ):
        kernel_assert(
            place in KDA_W_IN_PREFETCH_PLACES, f"place must be one of {KDA_W_IN_PREFETCH_PLACES}, got {place!r}"
        )
        kernel_assert(
            (buffers is None and first == 0 and place in _AFTER_FFN_PLACES)
            or (isinstance(buffers, int) and buffers >= 1 and isinstance(first, int) and 0 <= first < buffers),
            f"kda_w_in_prefetch needs >= 1 ring buffers and a first prefetched chunk below them, got {buffers}, {first}",
        )
        self.buffers = buffers
        self.first = first
        self.at_trigger = bool(at_trigger)
        self.place = place
        self.priority = priority
        self.tag = tag
        self.next_w_in = None
        self.next_w_scale = None
        self.pending = None

    def after_ffn(self, layer: int, block_weights, in_block: bool, after: tuple, wait: tuple = ()) -> None:
        """``after_ffn_cc`` / ``after_ffn_done``: issue the ring of the KDA layer after global layer ``layer``, when
        there is one, after the instructions ``after`` (the feed-forward body's first output collective), each fill
        waiting for the instructions ``wait`` (a semaphore wait). ``block_weights``: the next KDA layer's weights
        (``KDA_WEIGHT_NAMES``, then the scale stacks) when it is in the block."""
        nxt = layer + 1
        if self.place not in _AFTER_FFN_PLACES or nxt >= NUM_LAYERS or is_mla_layer(nxt):
            return
        if in_block:
            w_in = block_weights[0]
            w_scale = block_weights[len(KDA_WEIGHT_NAMES)] if w_in.dtype != nl.bfloat16 else None
        else:
            w_in, w_scale = self.next_w_in, self.next_w_scale
        if w_in is None:
            return
        kernel_assert(self.pending is None, f"a KDA w_in ring is already pending for layer {self.pending}")
        level = {} if self.priority is None else {"priority": self.priority}
        ring = prefetch_w_in_ring(
            w_in, f"{self.tag}L{nxt}", self.buffers, after, wait, self.first, w_scale=w_scale, **level
        )
        self.pending = (nxt, ring)

    def set_next_block(self, next_layer: int, end: int, kda_w_in, kda_w_in_scale, kda_k: int, n_kda: int) -> None:
        """Set ``next_w_in`` / ``next_w_scale`` before a block call: the weights of global layer ``next_layer`` (the
        first of the next block, below ``end``) when it is a KDA layer (``kda_k`` of the ``n_kda`` stacked), else None."""
        next_kda = next_layer < end and not is_mla_layer(next_layer)
        self.next_w_in = _layer_view(kda_w_in, kda_k, n_kda) if next_kda else None
        self.next_w_scale = _layer_view(kda_w_in_scale, kda_k, n_kda) if next_kda else None

    def window(self, layer: int, moe_body, block_w_in, in_block: bool) -> dict:
        """Return the MoE body option of global layer ``layer`` (``cc_window``), or {} when the next layer is not a
        KDA layer of the kernel (or the place is not ``cc_window``). ``block_w_in``: the next KDA layer's weight when it
        is in the block."""
        if self.place != "cc_window":
            return {}
        nxt = layer + 1
        w_in = block_w_in if in_block else self.next_w_in
        if nxt >= NUM_LAYERS or is_mla_layer(nxt) or w_in is None:
            return {}
        kernel_assert(
            getattr(moe_body, "cc_window", False),
            "kda_w_in_prefetch needs a MoE body with a window in its routed all-reduce (the Trn2 mxfp4 body)",
        )

        def issue(after, wait=(), w=w_in, tag=f"L{nxt}"):
            kernel_assert(self.pending is None, f"a KDA w_in ring is already pending for layer {self.pending}")
            ring = prefetch_w_in_ring(w, tag, self.buffers, after, wait, self.first)
            self.pending = (nxt, ring)
            return ring.fills

        issue.at_trigger = self.at_trigger
        return {"cc_window": issue}

    def check_issued(self, layer: int) -> None:
        """Fail unless the MoE body issued the ring of ``layer``."""
        kernel_assert(
            self.pending is not None and self.pending[0] == layer, f"the MoE body issued no KDA w_in ring for {layer}"
        )

    def take(self, layer: int):
        """Return the pending ring of global layer ``layer`` and forget it."""
        kernel_assert(
            self.pending[0] == layer, f"the pending KDA w_in ring is for layer {self.pending[0]}, not {layer}"
        )
        ring = self.pending[1]
        self.pending = None
        return ring


class MlaHeadPrefetch:
    """The carry of ``decode_block_body`` ``mla_head_prefetch``: one object for a kernel (opt-in).

    A MoE layer whose next layer is an MLA layer of the block issues that layer's first-call qkv and q_b weight loads
    in its routed all-reduce (``window``); the MLA body of that layer takes them (``take``). ``pending``:
    ``(layer, MlaHeadTiles)`` of the loads issued for ``layer``. ``at_trigger``: as ``KdaWInPrefetch``.
    """

    def __init__(self, at_trigger: bool = False):
        self.at_trigger = bool(at_trigger)
        self.pending = None

    def window(self, layer: int, moe_body, next_head) -> dict:
        """Return the MoE body option of global layer ``layer`` (``cc_window``), or {} when the next layer is not an MLA
        layer of the block. ``next_head``: the next MLA layer's HBM ``(mla_in_proj, mla_q_b_proj)`` when it is in the
        block, else None."""
        nxt = layer + 1
        if next_head is None or nxt >= NUM_LAYERS or not is_mla_layer(nxt):
            return {}
        kernel_assert(
            getattr(moe_body, "cc_window", False),
            "mla_head_prefetch needs a MoE body with a window in its routed all-reduce (the Trn2 mxfp4 body)",
        )
        in_proj, q_b = next_head

        def issue(after, wait=(), tag=f"L{nxt}"):
            kernel_assert(self.pending is None, f"MLA head loads are already pending for layer {self.pending}")
            head = prefetch_mla_head(in_proj, q_b, tag, after, wait)
            self.pending = (nxt, head)
            return head.dmas

        issue.at_trigger = self.at_trigger
        return {"cc_window": issue}

    def check_issued(self, layer: int) -> None:
        """Fail unless the MoE body issued the head loads of ``layer``."""
        kernel_assert(
            self.pending is not None and self.pending[0] == layer, f"the MoE body issued no MLA head loads for {layer}"
        )

    def take(self, layer: int):
        """Return the pending head tiles of global layer ``layer`` and forget them."""
        kernel_assert(
            self.pending[0] == layer, f"the pending MLA head loads are for layer {self.pending[0]}, not {layer}"
        )
        head = self.pending[1]
        self.pending = None
        return head


def _bank_sums_hook(bank_hT, w_hT, part, bank_dma, tag, w_row=None, entries=None, in_place=False, with_after=False):
    """Return a ``state_rs_hook`` of ``mla_block_attention`` that writes the bank sums of a mix
    (``common.attn_res_bank_part``) of ``bank_hT`` with the score weight ``w_hT`` (held in ``w_row``, a ``WeightRow``)
    into ``part``; the scratch tiles take the allocator of the MLA body. ``entries`` / ``in_place``: only the entries
    ``c0 .. c1 - 1``, the SBUF ones of a split bank read in place (``attn_res_bank_part``). ``with_after``: an
    ``out_ar_hook`` (``hook(sb, after)``), whose sums follow ``after`` in the schedule."""
    split = {"entries": entries, "in_place": in_place} if entries is not None else {}

    def hook(sb):
        attn_res_bank_part(bank_hT, w_hT, sb, w_row, name=f"{tag}_bank_sums", bank_dma=bank_dma, out=part, **split)

    def hook_after(sb, after):
        attn_res_bank_part(
            bank_hT, w_hT, sb, w_row, name=f"{tag}_bank_sums", after=after, bank_dma=bank_dma, out=part, **split
        )

    return hook_after if with_after else hook


def _plain_sums_hook(bank_hT, w_hT, part, tag, w_row, box):
    """Return a ``state_rs_hook`` of ``mla_block_attention`` that writes the plain part of a plain mix of ``bank_hT``
    (``common.attn_res_plain_bank_part`` of all its entries, the entry loads of a DRAM bank on the Activation queue) with
    the score weight ``w_hT`` (held in ``w_row``, a ``WeightRow``) into ``part``, and records its call in ``box``; the
    scratch and load tiles take the allocator of the MLA body."""

    def hook(sb):
        attn_res_plain_bank_part(
            bank_hT, w_hT, bank_hT.shape[1], sb, name=f"{tag}_sums", bank_dma="act", w_row=w_row, out=part
        )
        box.append(part)

    return hook


def _load_layer_vectors_hT(vectors, region=None):
    """Load hT-ordered per-layer vectors (HBM ``[L, K, H]``) as one SBUF tile ``[128, L*K, H/128]``.

    With ``region``, the tile takes an explicit address in the open frame of the region.
    """
    L, K, H = vectors.shape
    n_tiles = H // P_MAX
    shape = (P_MAX, L * K, n_tiles)
    if region is None:
        dst = nl.ndarray(shape, dtype=vectors.dtype, buffer=nl.sbuf)
    else:
        dst = region.stack(shape, vectors.dtype, "layer_vectors")
    load = nisa.dma_copy(dst=dst, src=vectors.ap(pattern=[[n_tiles, P_MAX], [H, L * K], [1, n_tiles]], offset=0))
    pool_after(load, "layer_vectors")
    return dst


@contextmanager
def _dmas_after_w_in(marks: dict, queue_order: bool = False, record: list = None):
    """Put every DMA that the scope traces after the last two ``kda_w_in`` loads of ``marks`` in the schedule, and make
    the first DMA of each queue (issuing engine and DGE mode) wait for their completion (``after_w_in_loads``): the
    later DMAs of the queue start after it (``moe_prefetch_kda_after_in_proj``). One wait a queue, not one a DMA, so
    the compiler keeps the ring's multi-descriptor DMAs. ``queue_order`` (``moe_prefetch_kda_queue_order``): the later
    DMAs of a queue also follow its first in the schedule: else the compiler can issue one of them first, with no wait,
    and its bytes then share the DMA rings with the projection's last weight loads. ``record``: None, or a list that
    gets every DMA of the scope (``moe_prefetch_kda_before_slot_stores``)."""
    original = nisa.dma_copy
    loads = list(marks["fills"][-2:])
    queues = {}

    def _after(*args, **kwargs):
        instr = original(*args, **kwargs)
        if instr is None:
            return instr
        if record is not None:
            record.append(instr)
        queue = (str(kwargs.get("engine")), str(kwargs.get("dge_mode")))
        if queue in queues:
            _schedule_after(instr, [*loads, queues[queue]] if queue_order else loads)
        else:
            queues[queue] = instr
            after_w_in_loads(instr, marks)
        return instr

    nisa.dma_copy = _after
    try:
        yield
    finally:
        nisa.dma_copy = original


TAIL_POOL_PIN = True
"""True: each 2-server MoE layer hands its tail's token-order gathers to ``tkg_weight_dma_gate.set_pool_after`` (the glue hook
``tail_gathers``), and the next layer's GpSimd prework that ``tkg_weight_dma_gate.pool_after`` marks (the layer-vector load, the
MLA pass-setup iotas) follows them in the schedule, so it does not queue ahead of them on GpSimd. Schedule edges only:
the values do not change. True (default); False gives the program without these edges."""


def _decode_block_body(
    stream,
    bank,
    norm_weights,
    attn_res_weights,
    kda_weights,
    mla_weights,
    moe_weights,
    dense_weights,
    kv_pools,
    kda_page_ids,
    mla_block_table,
    mla_slot,
    position,
    rank_info,
    tp_group,
    dp_group,
    world_group,
    first_layer: int,
    eps: float,
    kda_body,
    mla_body,
    moe_body,
    dense_body,
    layer_streams=None,
    layer_debug=None,
    dma_gates: bool | str = True,
    gate=(),
    mla_fp8_pages: bool = False,
    region=None,
    mla_loop_path: bool = False,
    poison=None,
    sources=None,
    last_block: bool = False,
    routed_kernel: str = "static",
    routed_reduce_dtype: str = "bf16",
    router_mode: str = None,
    router_dtype: str = "bf16",
    moe_router_weights=None,
    moe_in_proj_scales=None,
    moe_out_proj_scales=None,
    moe_expert_base: int = 0,
    affinity_scaling: str = "pre_quant",
    affinity_hi_lo: bool = True,
    before_attention=None,
    after_layer=None,
    mla_shared_sources: bool = True,
    collective_name: str = "",
    mla_zeroed_pages: bool = False,
    mla_loop_cast_engines: str = "off",
    mla_loop_pad_stationary: bool = False,
    mla_loop_first_static: bool = False,
    mla_scores_ahead: bool = False,
    mla_folded_reference: bool = False,
    mla_cp_rows_options: dict | None = None,
    mla_pair_pages: bool = False,
    x_parity: bool = False,
    kernel_first_layer: int | None = None,
    sbuf_reduces: tuple = (),
    lora_norm_eps: float = LORA_NORM_EPS,
    moe_early_loads: str = "all",
    moe_latent_load: str = "strided",
    moe_half_weight_buffers: bool = False,
    moe_weight_buffers: int | None = None,
    moe_scale_buffers: int | None = None,
    moe_transpose_lookahead: int | None = None,
    moe_cross_expert_order: bool = False,
    moe_cross_expert_transpose_blocks: bool = False,
    moe_token_tiles_transpose_blocks: bool = False,
    moe_transpose_block: bool = False,
    moe_gate_up_half_major: bool = False,
    moe_shared_scale_buffer: bool = False,
    moe_ring_reclaim: bool = False,
    moe_half_weight_buffer_count: int | None = None,
    moe_routed_store_form: str = "relayout",
    moe_dynamic_loop: bool = False,
    routing_form: str = None,
    moe_sel_load_options: tuple = (),
    cc_source_pingpong: bool = True,
    moe_active_expert_slots: int = 0,
    moe_active_expert_counts: nl.NkiTensor = None,
    mla_packed_loop: bool = False,
    mla_pack_once: bool = True,
    mla_pack_slots: int | None = None,
    mla_pack_tail_slots: int | tuple | None = None,
    mla_pack_tail_low_batch: bool = False,
    mla_pack_page_runs: int = 0,
    mla_pack_null_copy_vector: bool = False,
    mla_pack_null_runs: bool = False,
    mla_batched_cache_write: bool = False,
    mla_cp: int = 0,
    dp_size: int = DP_SIZE,
    mla_options: dict = None,
    reduce_rows: int = None,
    mix_cache: bool = False,
    early_ffn_gate: bool = False,
    ffn_rows: int = 0,
    wait_on=None,
    verify_bank: bool = False,
    moe_in_proj_form: str = "gathered",
    moe_peer_in_projs=None,
    ffn_tail_order: bool = False,
    kda_small_after_out_ar: bool = False,
    kda_states_after_out_ar: bool = False,
    moe_sums_after_routed: bool = False,
    row_sums_after_ffn_output: bool = False,
    attn_res_cached_bank_sums: bool = False,
    ffn_weights_after_attn: bool = False,
    moe_glue_dmas: tuple = (),
    moe_direct_latent: bool = False,
    moe_router_rows_drop_selection: bool = False,
    moe_latent_gather_mx: bool = False,
    attn_res_early_sums: bool = False,
    attn_res_mla_mlp_early: bool = False,
    attn_res_mla_mlp_plain_early: bool = False,
    attn_res_weight_rows: bool = False,
    attn_res_ahead: bool = False,
    attn_res_ahead_mlp: bool = False,
    attn_res_ahead_min_entries: int = 0,
    attn_res_ahead_vector: str = "off",
    attn_res_ahead_load_tiles: int = 2,
    attn_res_ahead_before_resid: bool = False,
    attn_res_ahead_mlp_kda_tail: bool = False,
    attn_res_kda_mlp_tail_entries: int = 0,
    attn_res_mla_mlp_tail_entries: int = 0,
    attn_res_gram_merge: bool = False,
    attn_res_cached_act_squares: bool = False,
    moe_out_proj_form: str = "reduced",
    mla_pack_group_plans: bool = False,
    moe_routing_tiles: bool = False,
    mla_pack_pairs: bool = False,
    mla_stage_cc_split: str = "off",
    mla_stage_post: bool = False,
    mla_pack_pair_plans: bool = False,
    mla_pack_setup_ahead: bool = False,
    mla_pack_state_store_queues: int = 1,
    mla_current_after_rs: bool = False,
    mla_stage_pre: bool = False,
    moe_latent_shard_form: str = "one_hot",
    attn_res_kda_mlp_early_entries: int = 0,
    attn_res_kda_mlp_early_in_front: bool = False,
    attn_res_kda_mlp_early_hbm: bool = False,
    moe_x_world_gather: bool = False,
    attn_rs_moe_input: bool = False,
    rows_hT_batched: bool = False,
    attn_res_scalar_squares: int = 0,
    norm_scalar_squares: int = 0,
    attn_res_scalar_squares_order: bool = False,
    attn_res_scalar_products: int = 0,
    attn_res_scalar_products_stt: bool = False,
    attn_res_bank_dma: str = None,
    norm_one_mul: bool = False,
    moe_x_gather_hbm: bool = False,
    attn_res_tensor_sums: bool = False,
    attn_res_one_round_norm: bool = False,
    attn_res_tensor_sums_idle: bool = False,
    moe_x_path_act: bool = False,
    kda_conv_keep: bool = False,
    kda_w_in_prefetch=None,
    mla_head_prefetch=None,
    kda_w_in_heads: dict = None,
    kda_ahead_carry: dict = None,
    kda_front_ahead: bool = False,
    kda_prework_window: bool = False,
    moe_static_alt_tiles: bool | str = False,
    moe_pad_rows: nl.NkiTensor = None,
    moe_in_proj_at_kda_reduce: bool = False,
    moe_in_proj_hook=None,
    moe_2srv_selectors_once: bool = False,
    kda_gap_loads: str = "off",
    moe_prefetch: dict = None,
    moe_preamble_buffers: int = 0,
    moe_hot_prefetch: dict = None,
    moe_preamble_reuse: bool = False,
    moe_prefetch_priority: int = None,
    moe_prefetch_uses: dict = None,
    moe_prefetch_scales: dict = None,
    moe_prefetch_kda_after_in_proj: bool = False,
    moe_prefetch_kda_queue_order: bool = False,
    moe_prefetch_kda_before_slot_stores: bool = False,
    moe_prefetch_start: dict = None,
):
    """Run the decoder layers of one block for the own DP replica (see the module docstring).

    Args:
        stream: HBM ``[B, H]`` stream at the block input, one row for each sequence of the replica, or
            the SBUF ``[128, B, H/128]`` stream of the previous block.
        bank: None for an empty AttnRes bank, or ``(bank_hT, count)``: an SBUF ``[128, C, B, H/128]``
            bank whose first ``count`` entries are valid. C must hold ``count`` and the pushes of the
            block. The block writes its pushes into the same tile. The verify also gives a DRAM bank or a
            ``common.SplitBank`` (its first entries in SBUF), whose pushes go to the home of each entry.
        norm_weights: HBM ``[L, 2, H]`` input and post-attention norm weights in hT order.
        attn_res_weights: HBM fp32 ``[L, 2, H]`` attention and MLP AttnRes score weights in hT order.
        kda_weights, mla_weights, moe_weights, dense_weights: tuples of stacked HBM weights, one tuple
            entry for each name of the body's weight list.
        kv_pools: tuple of the HBM page pools, one for each layer of the block.
        kda_page_ids: HBM int32 ``[n_kda, B, 1]`` state page of each KDA layer and sequence.
        mla_block_table: HBM int32 ``[n_mla, B, NB]`` local block table of each MLA layer and sequence.
        mla_slot: HBM int32 ``[n_mla, B, 1]`` cache slot of the current row of each MLA layer and sequence.
        position: HBM int32 ``[B, 1]`` position of the current token of each sequence.
        rank_info: HBM int32 ``[1, 4]`` (global rank, DP rank, TP rank, 0).
        tp_group, dp_group, world_group: replica groups.
        first_layer: global index of the first layer.
        eps: RMSNorm epsilon.
        kda_body, mla_body, moe_body, dense_body: the layer bodies. Each one has the attribute
            ``token_major = True``: it takes and returns ``[128, B, H/128]`` tiles. A MoE body with the attribute
            ``moe_sharding = "dp4xtp32"`` (``tkg_moe_dp.dp_moe_body``) gets the DP collective tensors.
        layer_streams: optional HBM ``[L, B, H]`` that receives the stream after each layer.
        layer_debug: optional HBM ``[L, 4, B, H]`` that receives, for each layer, the attention input,
            the attention output, the feed-forward input, and the feed-forward output.
        dma_gates: gate the weight DMAs of each body on the output collectives of the previous body
            (see ``tkg_weight_dma_gate``). ``"per_queue"``: as True, but a MoE layer with ``sel_load`` routed experts
            gates only the first weight DMA of each queue (fewer edges and a faster trace; in a 93-layer
            decode step at 16K context, a slower step at B = 2 and B = 4).
        gate: the output collectives of the body before the block. The weight DMAs of the first
            attention body wait for them (with ``dma_gates``).
        mla_fp8_pages: the MLA layers keep FP8 pages in their BF16 pools (``fp8_pages`` of the MLA body).
        mla_pair_pages: the FP8 MLA pages hold the token-pair layout (``pair_pages`` of ``mla_decode_attention``);
            CP32 MLA only.
        mla_zeroed_pages: the MLA loop path reads its cache pages with no clamp (``zeroed_pages`` of the MLA
            body). Set it only when the caller zeroes each MLA page at allocation.
        mla_loop_cast_engines: ``loop_cast_engines`` of the MLA body (``tkg_mla_paged.mla_decode_attention``): the engines of
            the FP8 -> BF16 page-tile casts of the per-sequence key loops. Bitwise; "off" (default) keeps the call.
        mla_loop_pad_stationary: ``loop_pad_stationary`` of the MLA body: 128-column stationaries in the per-sequence key
            loops. Bitwise; False (default) keeps the call.
        mla_loop_first_static: with ``mla_cp``, the CP128 MLA body runs its first loop iteration as static code
            (``loop_first_static`` of ``tkg_mla_block.mla_cp_decode_attention``). False (default) keeps the program.
        mla_scores_ahead: with ``mla_cp``, the CP128 MLA loop computes the weights of each pair of key groups before
            the sums of the pair before it (``scores_ahead`` of ``tkg_mla_block.mla_cp_decode_attention``). False
            (default) keeps the program.
        mla_folded_reference: with ``mla_cp``, the CP128 MLA loop adds the softmax reference in its rope matmul
            (``folded_reference`` of ``tkg_mla_block.mla_cp_decode_attention``). False (default).
        mla_cp_rows_options: with ``mla_cp``, keys of the rows form for the CP128 MLA decode
            (``tkg_mla_block.CP_ROWS_OPTIONS``, ``rows_dr`` and its forms; see ``tkg_mla_block.mla_cp_decode_attention``). The
            verify of the deployment must take the same keys. None (default) keeps the program.
        lora_norm_eps: the epsilon of the q_a and kv_a RMSNorms of the MLA layers (``lora_norm_eps`` of
            the MLA body), on every MLA path. The default is 1e-5 (``tkg_mla_paged.LORA_NORM_EPS``).
        moe_active_expert_slots: K, the opt-in active-expert slots of the MoE layers (``active_expert_slots`` of
            ``moe_tkg_decode``): only the first K active local experts run, in straight-line slots, and every
            expert in a one-trip loop when more are active. The values do not change. One dispatch point
            (``tkg_moe.routed_expert_path``) picks the path of a layer: the dynamic loop when
            ``moe_dynamic_loop`` applies, else the slots when K > 0, else the static loop. A layer never runs
            both. K > 0 needs the base MoE body (``tkg_moe.base_moe_body``: no MoE option such as an FP16 router,
            not the DP4 x TP32 body) and at most 4 sequences; else, or with K above 7, the call fails. 0 (default)
            changes nothing.
        moe_active_expert_counts: debug, with the slots only (``tkg_moe.active_expert_counts``): INT32 HBM
            ``[n_moe, 1]``, row k gets the number of active local experts of this rank in the k-th MoE layer of the
            block (``active_expert_count`` of ``moe_tkg_decode``). None (default) adds no instruction.
        x_parity: the feed-forward bodies alternate the DP x gather destination
            (``tkg_moe.ffn_layer_sources``). Set it only with the decode bodies of ``tkg_moe`` and the
            persistent ``x_dst`` of ``collective_sources``. ``tkg_moe.x_gather_alternates`` picks the layers
            of the alternate destination: never the first layer of the kernel, nor an MLA layer that runs
            the key loops. The DP4 x TP32 MoE body (``tkg_moe_dp``) ignores it.
        kernel_first_layer: global index of the first layer of the kernel, when an earlier block of the same
            kernel ran before this one. None: ``first_layer``.
        sbuf_reduces: the MoE all-reduces that run in SBUF on the loop path (``"routed"``), when the block builds
            its ``sources`` (``collective_sources``).
        moe_early_loads: the MoE layers whose static expert loop issues its first weight and scale loads before
            its latent work (``MOE_EARLY_LOADS``, ``early_loads`` of ``tkg_moe.latent_moe_decode``): ``all``
            (default), ``kda`` (the layers of a KDA attention) or ``none``. The others issue them in the loop.
        moe_latent_load: the load of the HBM latent gather in the base MoE body (``LATENT_LOADS``, ``latent_load``
            of ``tkg_moe.latent_moe_decode``). The values do not change; ``strided`` (default) keeps the program.
        moe_half_weight_buffers: above 8 sequences (the MLA groups of 8), the static expert loop streams half
            projections through 4 half-size buffers in place of its 2 whole ones (``half_weight_buffers`` of
            ``moe_tkg_decode``; the same SBUF). The values do not change; False (default) keeps the program.
        moe_weight_buffers: the whole weight buffers of the 2-server static expert loop (``weight_buffers`` of
            ``moe_tkg_decode``, 1 to 5), in place of the block's rule (2 above 8 sequences). A 2-server buffer holds
            one I_L = 1536 projection (21,504 B for each partition), half of a one-server buffer, so 3 cost the SBUF
            of 1.5 one-server buffers. With 4 or 5 and ``moe_cross_expert_order`` each weight load waits for the
            matmuls of the projection 4 or 5 uses before it. The token-tiled expert calls (more than 128 tokens) take
            1 to 3. Not with ``moe_half_weight_buffers``. The values do not change; None (default) keeps the program.
        moe_scale_buffers: the gate/up scale buffers of the 2-server static expert loop (``scale_buffers`` of
            ``moe_tkg_decode``, 1 or 2), in place of the block's 1. The values do not change; None (default) keeps
            the program.
        moe_transpose_lookahead: the activation chunks that the 2-server static expert loop transposes before each
            expert's first down matmul (``transpose_lookahead`` of ``moe_tkg_decode``, 1 or 2). The values do not
            change; None (default) keeps the program.
        moe_cross_expert_order: the 2-server static expert loop runs the gate matmuls of the next expert between
            the up and the down matmuls of each expert (``cross_expert_order`` of ``moe_tkg_decode``). Needs
            ``moe_weight_buffers`` 3 to 5 (any ``moe_scale_buffers``) and not ``moe_transpose_lookahead``. The values
            do not change; False (default) keeps the program.
        moe_cross_expert_transpose_blocks: the activation transposes of ``moe_cross_expert_order`` run in tile
            blocks between the gate chunks of the next expert, with no MX matmul inside a block
            (``cross_expert_transpose_blocks`` of ``moe_tkg_decode``). Needs ``moe_cross_expert_order``. The values do
            not change; False (default) keeps the program.
        moe_token_tiles_transpose_blocks: the token-tiled expert calls of the 2-server static expert loop (more than
            128 tokens) run one fixed TensorE order with every PE transpose in a block between MX matmul groups
            (``transpose_blocks`` of ``moe_tkg_decode_token_tiles``; the 2-server body refuses it without such a
            call). The values do not change; False (default) keeps the program.
        moe_transpose_block: with ``moe_cross_expert_order``, the activation transposes of each expert run in one
            block with no MX matmul inside (``transpose_block`` of ``moe_tkg_decode``). The values do not change;
            False (default) keeps the program.
        moe_gate_up_half_major: the EP128 static gate/up weights are stored half-major (``gate_up_half_major`` of
            ``moe_tkg_decode``). Every static expert loop reads them so. The values do not change; False (default)
            reads the standard layout.
        moe_shared_scale_buffer: the EP128 static expert loop keeps the gate/up and down MX scales in one buffer of
            4 slots (``shared_scale_buffer`` of ``moe_tkg_decode``): 21 KB less SBUF per partition across the loop.
            Needs 1 gate/up scale buffer (the one-server rule, or ``moe_scale_buffers`` 1). The values do not
            change; False (default) keeps the program.
        moe_ring_reclaim: with ``moe_half_weight_buffers`` above 8 sequences, one more half weight buffer of the
            static expert loop at an explicit address, over feed-forward collective tiles of the region that no
            layer touches from its latent world gather to the end of its expert loop (``collective_sources``
            ``ring_reclaim``; the sources must come from such a call). The values do not change; False (default)
            keeps the program.
        moe_half_weight_buffer_count: with ``moe_half_weight_buffers``, the number of compiler-placed half weight
            buffers of the static expert loop, 2 to 6 (with ``moe_ring_reclaim`` at most 5: the explicit buffer
            comes on top). None (default) keeps 4, the SBUF of the 2 whole buffers.
        moe_routed_store_form: how the static expert loop of the base MoE body fills the routed all-reduce source
            (``routed_store_form`` of ``tkg_moe.latent_moe_decode``, ``tkg_moe.ROUTED_STORE_FORMS``).
            ``relayout`` (default) keeps the program; ``pe`` writes the source from the expert loop through TensorE
            transposes, with no relayout DMA. The values do not change. Another expert path or body fails with ``pe``.
        cc_source_pingpong: with ``sources`` None, the ``pingpong`` of ``collective_sources``. The layers take the
            source set of their parity (``layer_sources``) either way.
        region: None, or the ``ExplicitSbuf`` region of the entry. With a region, the long-lived tiles
            of the block and every SBUF tile of an MLA layer take explicit addresses (see
            ``tkg_sbuf_alloc``). An SBUF ``stream`` must then be a stream tile of the region.
        mla_loop_path: let the MLA body take the loop path for a table of more than 16 local blocks.
            Give a region with it: without one, the compiler can overwrite the tiles that live
            through the loops.
        mla_packed_loop: with ``mla_loop_path``, one MLA loop over the key groups of all sequences
            (``mla_decode_attention(packed_loop=True)``); bitwise the same output.
        mla_pack_once: build the items of the packed loop once for the block (``mla_pack_plan``), not in
            each MLA layer.
        mla_pack_slots: items of one packed iteration, a power of two up to 64 (None: 16 at 256K+ tables with
            B >= 4, else 8).
        mla_pack_tail_slots: None, or the items of one iteration of the tail loop of the packed loop, or a pair
            for two tail loops (``mla_decode_attention(pack_tail_slots=...)``), from ``_PACK_TAIL_MIN_TOKENS``
            sequences; bitwise the same output.
        mla_pack_tail_low_batch: also take ``mla_pack_tail_slots`` at 8 sequences and fewer (one MLA group), in the
            plan of the block (``mla_pack_plan``) or else in the items of each layer; bitwise the same output. False
            (default) keeps the programs of 8 sequences and fewer. There the items round up to whole tail
            iterations (``tail_round``) and a single tail count takes a loop of half the slots before it
            (``pack_tail_chain``: 32 slots with 8 run tails of 16 and 8), so each tail loop runs at most once,
            whatever the mix of contexts in the batch.
        mla_pack_page_runs: 0 (default), or 2 or 4: the packed loop loads the 4 pages of a key group with one DMA
            per run of that many pages (``mla_decode_attention(pack_page_runs=...)``), for a block table whose groups
            are runs of pages; bitwise the same output, and NaN for a sequence with a group that is not made of runs.
            Needs ``mla_packed_loop``; 4 needs pools of 512-row FP8 pages (the MLA body refuses it on other pools).
        mla_pack_null_copy_vector: with ``mla_packed_loop`` and ``mla_zeroed_pages``, the packed loop copies the first
            page of each key group for its null table entries on VectorE (``pack_null_copy_vector`` of
            ``mla_decode_attention``). The values do not change. False (default) keeps the program.
        mla_pack_null_runs: with ``mla_pack_page_runs``, a padded sequence (null pages only) counts as runs, so its
            rows are finite, not NaN (``pack_null_runs`` of ``mla_decode_attention``). The rows of real sequences do
            not change. False (default) keeps the program.
        mla_batched_cache_write: each MLA body (each staged group of 8) writes its current cache rows with one
            vector-indirect DMA in place of one DMA per sequence (``batched_cache_write`` of ``mla_decode_attention``).
            The rows of sequences that a rank does not own go to their own rows of the null page. The outputs and the
            owned rows do not change. Not with ``mla_pair_pages`` or ``mla_cp``. False (default) keeps the program.
        sources: None, or the persistent collective tiles of the kernel (``collective_sources``). None
            makes them for the block.
        last_block: the block ends the layers of the kernel. On the static path the last residual then takes
            the FFN input tile, so the collective tiles stay reserved until the caller has read the stream.
        poison: test only. None, or ``(layer, stage, rows)``: add the HBM ``[B, H]`` ``rows`` to one
            tile of block layer ``layer``. ``stage`` is ``"attn_res_in"`` (the stream at the attention
            mix), ``"attn_out"`` (the attention output) or ``"ffn_in"`` (the feed-forward input). A test
            puts a NaN or an Inf in one row and zeros in the others, to find the ops that carry a
            non-finite value of one token into another token. None adds no instruction.
        routed_kernel: the routed-expert kernel of the latent MoE layers (``tkg_moe.ROUTED_KERNELS``).
            ``moe_weights`` then holds the weights of that kernel (``tkg_moe.moe_weight_slots``). A kernel
            other than ``static`` needs a region: the TP128 kernels place their tiles at explicit addresses.
        routed_reduce_dtype: the add type of the routed all-reduce of the MoE layers (``bf16`` or ``fp32``).
        router_mode: the router form of the MoE layers (``tkg_moe.ROUTER_MODES``); None is the default form of
            the routed kernel.
        router_dtype: the router matmul type of the MoE layers (``tkg_moe.ROUTER_DTYPES``). With ``fp16`` or
            ``fp16_local`` the norm before each MoE layer also writes an FP16 copy from its FP32 values, and
            ``moe_router_weights`` (HBM FP16 ``[n_moe, 128, 56, 7]``, ``[n_moe, 128, 56, 28]`` with ``fp16_local``, or
            ``[1, ...]``) gives the router rows of each layer (``tkg_moe.latent_moe_decode`` ``router_weight``).
        moe_in_proj_scales: None (default, the BF16 MoE in-projection), or the scales of an MXFP8 MoE in-projection
            stacked as the in-projection (``[n_moe, ...]`` or ``[1, ...]``): each MoE layer's body takes them as
            ``in_proj_scale`` (a body with ``in_proj_mx``; ``tkg_moe_2srv.latent_moe_decode_2srv``). It needs the
            FP16 router.
        moe_out_proj_scales: None (default, the BF16 MoE out-projection), or the scales of an MXFP8 MoE
            out-projection stacked as the out-projection (``[n_moe, ...]`` or ``[1, ...]``): each MoE layer's body takes
            them as ``out_proj_scale`` (a body with ``out_proj_mx``; ``tkg_moe_2srv.latent_moe_decode_2srv``).
        affinity_scaling, affinity_hi_lo: ``sel_load``: the affinity scaling and its high/low split
            (``tkg_moe.latent_moe_decode``).
        routing_form: TP128 kernels: the form of the top-16 routing (``tkg_moe.ROUTING_FORMS``); None
            (default) takes the default of ``tkg_moe.latent_moe_decode``.
        moe_sel_load_options: ``sel_load``: (name, value) pairs of extra ``moe_sel_load_tp_start`` options (for
            example ``long_stream``, ``chunk_tokens``, ``sbuf_reserve``). The values do not change; () (default) keeps
            the program.
        moe_expert_base: ``sel_load``: the MoE layer of the first MoE layer of the block in the TP128 expert stacks
            (the stacks of all MoE layers pass whole, see ``tkg_moe.latent_moe_decode`` ``expert_offset``).
        moe_dynamic_loop: the opt-in dynamic expert loop of the MoE layers (``tkg_moe.dynamic_expert_loop``). It
            applies only at T = 4 B <= ``DYNAMIC_EXPERT_LOOP_MAX_TOKENS`` tokens (B <= 2), off the MLA loop path, in
            the base MoE body (``tkg_moe.base_moe_body``: ``routing_form="base"`` keeps it). Elsewhere the static
            loop runs; with the DP4 x TP32 body the call fails. The values do not change.
        mla_cp: 0, or 128 for the CP128 MLA body (``tkg_mla_block.mla_cp_decode_attention``): ``sources`` then holds its
            sites (``collective_sources(mla_cp=128)``) and ``tp_group`` is the CP group of all ranks.
        reduce_rows: None, or the rows of the verify step of the deployment (DP1 decode): the world reductions take
            that size, so the decode is bitwise equal to the verify rows (the runtime picks the collective algorithm,
            and so the summation order, by the message size).
        mix_cache: keep the rstd of each bank entry from its push (``bank`` is then ``(bank_hT, count, bank_rstd)``)
            and give the mixes and the norms the ``common.WeightRow`` of their weights, so they take fewer
            instructions and a token's result does not depend on the rows of the step (decode and verify agree).
        early_ffn_gate: the weights of the next body wait for the collective before the last one of a body (the routed
            all-reduce of a MoE layer, the state reduce-scatter of an MLA layer), not the last one, so they stream
            during the tail of the body (``tkg_weight_dma_gate.take_released``). A body with one output collective keeps it.
        wait_on: None, or a tile this block's stream waits for (a bitwise identity multiply, so the stream is
            unchanged). It orders this block after another one that shares the same explicit region.
        verify_bank: the verify above 16 rows: the static AttnRes bank goes to HBM, and an HBM bank push waits for
            the mix that reads the last entry (the plain mix, and the mix of a kept bank rstd). A given bank in HBM takes
            the same wait without the flag. False keeps the program of the decode (its bank is in SBUF).
        moe_in_proj_form: the in-projection form of the MoE layers (``tkg_moe.IN_PROJ_FORMS``). ``gathered``
            (default) keeps the program. ``dp_local`` needs ``moe_peer_in_projs`` (HBM BF16 ``[n_moe, 128, 56, 140]``)
            and the ``dp_local`` ``moe_in_proj`` / ``moe_vectors`` stacks (``latent_moe_dp_local_rank_weights``).
        ffn_tail_order: two orders after the output collective of each feed-forward body; the values do not change.
            The AttnRes mixes of the next layer sum their bank rows after the trigger of that collective
            (``common.row_sums_order``), so the scheduler does not put those sums into the VectorE chain of the body's
            latent norm. The small weight loads of a next KDA body wait for the load of the collective's result
            (``tkg_weight_dma_gate`` consumer gates), so they do not share the DMA rings with its flight. False keeps the program.
            Slower with the EP128 static MoE body (``routed_kernel`` ``static``): the block rejects it there.
        kda_small_after_out_ar: the gated weight loads of every KDA layer on the GpSimd queue (``kda_w_small``,
            ``kda_params`` and ``kda_w_out``) wait for the load of the result of the last output collective of the
            feed-forward body before it (``tkg_weight_dma_gate`` consumer gates: a semaphore wait), so the in-order GpSimd queue,
            which also triggers that collective, issues them after it and not in its flight. Needs a KDA body that takes
            ``small_after_result`` (the verify body, ``verify_model.verify_bodies``). Only a feed-forward body that
            records the load of its result (``release_gate_consumer``: the multi-server MoE body for its ``reduced``
            output all-reduce, the multi-server dense body for its replica rows) gives its next KDA layer the wait;
            after a body that records none the KDA layer keeps its program. The bytes and the instructions do not
            change, so the values do not change. False (default) keeps the program.
        kda_states_after_out_ar: the state loads of a KDA layer that follows a 2-server MoE layer in this call (not the
            first layer of the block) follow the trigger of that layer's output all-reduce (the KDA body's
            ``state_loads_after``: a schedule edge and a runtime wait), so the MoE out store, which the trigger waits
            for, is not behind them on the in-order SP queue. Needs the 2-server MoE body and a KDA body that takes
            ``state_loads_after`` (the verify body, ``verify_model.verify_bodies``; ``check_kda_states_after_out_ar``).
            The bytes and the instructions do not change, so the values do not change. False (default) keeps the
            program.
        moe_sums_after_routed: the row sums of each layer (the bank sums of its AttnRes mixes and its norms,
            ``common.row_sums_order``) follow the first collective of the routed reduction of the previous MoE layer
            (the multi-server body's ``routed_trigger`` glue hook), a schedule edge. The routed partials of that layer
            are copied out of PSUM on the Vector engine (the column-group copies of ``moe_tkg_decode``) before their
            store, and the next layer's bank sums, which need only the bank, otherwise fill the Vector engine's
            in-order stream ahead of those copies and delay the routed store and its reduction. The sums then run in
            the flight of the routed reduction. Needs the multi-server MoE body (``moe_sharding`` ``dp_2srv``). The
            values do not change; False (default) keeps the program.
        row_sums_after_ffn_output: the AttnRes mixes of the next layer sum their bank rows after the instruction that
            consumes the routed output of each MoE body (``common.ffn_output_order``: the DMA that stores the routed
            partial into the source of its routed collective, or that collective for an SBUF source). The bank rows
            do not need the MoE output, so the scheduler can otherwise put them on VectorE ahead of the PSUM copies
            of the column-group routed partial (``routed_store_form`` ``pe``), which delays the routed collective.
            A schedule edge, not a semaphore wait: the values do not change. Not with ``ffn_tail_order`` (which
            orders the same sums after the last collective of the body). With ``moe_sums_after_routed`` both orders
            apply: its edges go to every row sum of the layer, these to the first VectorE row sum. False (default)
            keeps the program.
        attn_res_cached_bank_sums: keep the rstd of each bank entry from its push (``bank`` is then
            ``(bank_hT, count, bank_rstd)``), so the AttnRes mixes sum no squares of the bank. Unlike ``mix_cache``,
            the mixes keep the weighted sums, the exponentials and the norms of the plain mix, and the push sums its
            squares with the instructions of the plain mix, so the result has the bits of the plain mix. With
            ``mix_cache``, or with one row (T = 1, where the plain mix sums a one-entry bank with a fused multiply-sum
            that a kept rstd cannot match), it does nothing. False (default) changes nothing.
        ffn_weights_after_attn: with the DMA gates on, the static weight loads of each feed-forward body also wait
            for the residual add that reads the attention output (a semaphore wait, ``tkg_weight_dma_gate.gate_all_dmas``
            ``after``). The gate on the attention output collective is a schedule edge, so these loads otherwise
            start at that collective's trigger and share the DMA rings with its flight (the KDA all-reduce, the MLA
            output all-gather). The values do not change. False (default) keeps the program.
        moe_glue_dmas: TP128 MoE bodies: names of ``tkg_moe.GLUE_DMA_FORMS``, the forms of the small DMAs between
            the stages of each MoE layer (``replica_rows``: the output load reads only this replica's tokens, without
            the row select copies; ``sp_weights``: the latent norm weight and the shared in-projection loads on the SP
            hardware DGE queue; ``split_gather``: the x gather store and load and the ``pe32`` rank-row load as 2
            DMAs each, on the SP and the Activation hardware DGE queues, base body only). With the EP128 ``static``
            kernel, ``replica_rows`` and ``split_gather`` only, in the base MoE body
            (``tkg_moe.BASE_GLUE_DMA_FORMS``). The values do not change; () (default) keeps the program.
        moe_direct_latent: the static expert loop of the base MoE body reads the gathered latent tile in place,
            without its SBUF-to-SBUF copy DMA (``direct_latent`` of ``tkg_moe.latent_moe_decode``). The values do
            not change; False (default) keeps the program. Another expert path or body fails with it.
        moe_preamble_buffers: K in 1 to 3 extra whole weight buffers of the static expert loop of the base MoE body for
            the first K projections of each call (``preamble_buffers`` of ``tkg_moe.latent_moe_decode``), so they load
            during the work before the first expert matmul. The values do not change; 0 (default) keeps the program.
            Another expert path or body, and half-size weight buffers, fail with it.
        moe_preamble_reuse: with ``moe_preamble_buffers``, each extra buffer also takes one more projection of the next
            expert (``preamble_reuse`` of ``tkg_moe.latent_moe_decode``; gate 1 for 1 buffer and 2 ring buffers). The
            values do not change; False (default) keeps the program.
        moe_router_rows_drop_selection: the base MoE body on the MLA loop path: the latent world gather rows leave out
            the selection values, and each rank adds the router bias of all experts to the gathered scores
            (``router_rows_drop_selection`` of ``tkg_moe.latent_moe_decode``). It needs the 21-word latent gather
            tensors (``collective_sources`` ``moe_gather_cols``) and ``moe_vectors`` with the bias of all experts. The
            values do not change; False (default) keeps the program. Another body fails with it.
        moe_latent_gather_mx: with ``moe_router_rows_drop_selection``, ``moe_latent_load`` ``pe32`` and the static
            expert loop of the base MoE body on the MLA loop path: the latent world gather carries the MXFP8 expert
            input (``latent_gather_mx`` of ``tkg_moe.latent_moe_decode``): 64 bytes per token and rank in place of
            112, and no latent transpose or quantize on the receiving ranks. It needs the UINT32 gather tensors
            (``collective_sources`` ``moe_gather_mx``) and ``moe_in_proj`` in the 32-column block layout
            (``tkg_moe.MX_IN_PROJ_COLS``). The values do not change; False (default) keeps the program.
        moe_static_alt_tiles: the static expert loop of the base MoE body takes its gate sigmoid, up tanh, down PSUM
            copy and activation eviction tiles from two buffers each (``static_alt_tiles`` of
            ``tkg_moe.latent_moe_decode``), so that under SBUF pressure the compiler cannot stack them on one
            address: True all of them, "act" only the sigmoid / tanh tiles, "act_copy" those and the PSUM copies
            (``moe_routed.STATIC_ALT_TILE_FORMS``). The values do not change; False (default) keeps the program.
            Another expert path or body fails with it.
        moe_x_world_gather: the base MoE body and the dense body gather the DP replicas' hidden over the world
            group: each rank sends its 4 partition rows of its replica through HBM (``collective_sources``
            ``x_world``, ``tkg_moe._gather_replica_tokens``), in every layer, into the persistent HBM destination
            of the DP gather, which gets the same bytes. The world gather moves each byte through the switch port of
            a chip once, the DP gather 3 times. The values do not change; False (default) keeps the program. It needs
            the sources of ``collective_sources`` ``x_world`` (the HBM x gather of the MLA loop path, DP4, no MoE
            option) with the second x destination of ``dst_pingpong`` ``"x"``, and ``mla_stage_cc_split`` "off"
            (``check_x_world_gather``); another body fails with it.
        rows_hT_batched: load an HBM ``stream`` into the hT stream tile in column chunks
            (``common.load_rows_hT_batched``) in place of one DMA and one transpose a row (``common.load_rows_hT``).
            The values do not change. False (default) keeps the program.
        attn_rs_moe_input: the MoE-input scheme of the 1-server DP verify at R = TP_SIZE rows per replica. The
            attention output is reduce-scattered over ``tp_group`` (rank t keeps row t, row-major) instead of reduced on
            every rank; the residual, the MLP mix and the norm run on that one row; a world token all-gather of the
            normed rows fills the base body's x gather destination; and a separate ``tp_group`` all-gather of the updated
            stream (s' = s + attention) flies beside the expert loop for the post-MoE residual and the next attention.
            The KDA all-reduce becomes the reduce-scatter; the MLA layers keep their output reduce-scatter and drop the
            output all-gather. It fails closed off the DP4 x TP32 layout at R = TP_SIZE (``check_attn_rs_moe_input``).
            False (default) keeps the program.
        moe_x_gather_hbm: the base MoE body and the dense body gather the DP replicas' hidden through the persistent
            HBM pair of ``collective_sources`` ``x_hbm`` in every layer (``tkg_moe.ffn_layer_sources`` ``via_hbm``),
            not only in the layers after an MLA loop: the gather lands in HBM and a load copies it to ``x_dst``. The
            values do not change; False (default) keeps the program. It needs that HBM pair, the DP4 layout and the
            base MoE body; another body fails with it.
        moe_x_path_act: with ``moe_x_world_gather``, each layer's HBM x gather store, world gather slice and gather
            destination load run on the Activation hardware DGE queue (``tkg_moe._gather_replica_tokens``
            ``x_path_act``). Without it the compiler puts the store on the Activation queue and the slice (explicitly)
            and the load on the SP queue, where in a KDA layer they wait behind the layer's recurrent-state stores, and
            the world gather waits for the slice. The DMAs move the same bytes, so the values do not change; False
            (default) keeps the program.
        kda_conv_keep: keep every KDA conv-state store source until its stores have read it. Each per-layer source
            gets a VectorE read of its whole view after the feed-forward norm, with a runtime dependency on its stores
            (``tkg_kda.kda_read_scatter_keep``), and the first write of a KDA layer into a persistent conv-state tile
            waits at runtime for the conv-state stores of the two KDA layers before (``kda_decode_attention``
            ``conv_copy_wait``; across the blocks through ``sources["kda_conv_waits"]``). The compiler ends the live
            range of a DMA source at the DMA trigger, so without it a later writer of those bytes may not wait for the
            store. Placement and sync only: the values do not change. False (default) keeps the program.
        moe_pad_rows: None (default) keeps the program. The HBM int32 ``[R, 1]`` padding-row input of the step (1 for a
            padding row, 0 for a real row; R = the rows of the stream, row ``j * B + b`` for a verify step): each MoE
            layer gives it to its MoE body (call option ``pad_rows``; the body needs ``takes_pad_rows``), which loads
            the flags (``tkg_moe.pad_row_flags``; 1 DMA and 2 VectorE ops) and writes 0 into their router scores
            before its gather: their affinities are 0, so they load and run no expert in the dynamic expert forms, and
            their routed output is 0. The other rows keep their bits.
        moe_routing_tiles: TP128 MoE bodies with ``moe_in_proj_form`` dp_local on the MLA loop path: the [T, 16]
            outputs of the top-16 routing take explicit region addresses below the routed kernel's buffers
            (``routing_tiles`` of ``tkg_moe.latent_moe_decode``), so the first weight gather waits for its index,
            not for a write-after-read on a routing tile placed in its weight slot. The values do not change; False
            (default) keeps the program.
        attn_res_early_sums: with the kept bank rstd (``attn_res_cached_bank_sums`` or ``mix_cache``), move AttnRes
            work off the VectorE chain of the stream; the values do not change (the same instructions):
            - every mix sums the stream squares and starts their rsqrt before its weighted sums
              (``common.attn_res_hT`` ``squares_first``), so ScalarE takes the rsqrt and its exponential table load
              beside the weighted sums;
            - the attention mix of a layer sums its bank rows in compiler-placed tiles (``common.attn_res_bank_part``)
              before the explicit MLA frame, so the scheduler can run them in the output collective of the previous
              feed-forward body, as for a KDA layer;
            - the MLP mix of a KDA layer sums its bank rows right after the attention norm, where VectorE waits for the
              in-projection of the KDA body.
            With ``verify_bank`` (the DRAM bank of the verify) the bank sums stream the entries through SBUF
            (on ``attn_res_bank_dma``), and the mix streams them again. It does nothing with one row (T = 1), without a
            kept rstd, or with an HBM bank that is not the verify bank. False (default) keeps the program.
        attn_res_mla_mlp_early: with ``attn_res_early_sums``, the MLP mix of an MLA layer takes its bank sums early
            too: the MLA body traces them right after the issue of its last state reduce-scatter
            (``mla_block_attention`` ``state_rs_hook``), where VectorE waits for that collective. On the loop path the
            sums take a tile of the layer frame that outlives the MLA frame. The MLA body must take the hook (the k1
            verify in the rows form or with the BF16-o state reduce-scatter, ``mla_body.state_rs_hook``). The same
            instructions, so the values do not change.
            Not on a layer that pushes a bank entry. False (default) keeps the program.
        attn_res_mla_mlp_plain_early: for the plain MLP mix (no kept bank rstd) of an MLA layer above
            ``COMPACT_MIX_TOKENS`` rows, the MLA body traces the bank sums of that mix (``common.attn_res_plain_bank_part``
            of all its bank entries, the entry loads of a DRAM bank on the Activation queue) right after the issue of its
            state reduce-scatter (``state_rs_hook``), where VectorE waits for that collective, and the mix takes them as
            its ``plain_part``. On the loop path the part takes a tile of the layer frame that outlives the MLA frame. The
            MLA body must take the hook (the k1 call, ``mla_body.state_rs_hook``) and must call it. The mix takes the
            same instructions, so the values do not change. Not on a layer that pushes a bank entry. False (default)
            keeps the program.
        attn_res_weight_rows: the AttnRes mixes and bank sums give their score weight as the ``WeightRow`` of the
            layer's weight tile (as ``mix_cache`` does), so that the weighted sums of all rows of a candidate take one
            multiply and one sum in place of a fused multiply and sum for each row (``common._row_sums``): the attention
            and MLP ``attn_res_hT`` mixes, the early and MLA state-RS bank sums and ``attn_res_plain_bank_part``. The
            same FP32 products added in the same order, so the values do not change. False (default) keeps the program.
        attn_res_ahead: the attention mix of each KDA layer takes the ahead form (``common.attn_res_ahead_hT``): the bank
            part of the mix (``common.attn_res_ahead_parts``: the weighted bank entries, their score max, sum and norm)
            is built in the tail of the layer before, beside its output collective, and the mix and its RMSNorm then
            run in one pass over the stream on TensorE, ScalarE and VectorE (the TensorE Gram form of the stream sums;
            no GpSimd on the critical path: GpSimd shares the bank part). The same math with other FP32
            rounding (accuracy class). Needs more than ``COMPACT_MIX_TOKENS`` rows and the kept bank rstd
            (``attn_res_cached_bank_sums``), not ``attn_res_tensor_sums``. The mix tile of ``before_attention`` does not
            exist: the hook is not called on those layers. False (default) keeps the program.
        attn_res_ahead_mlp: the MLP mix of each KDA layer takes the ahead form: its bank part is built in the tail of
            the layer before, from the same entry loads as the attention part (a layer that pushes a bank entry, or a
            bank above ``ATTN_RES_AHEAD_SHARED_ENTRIES``, builds it after its attention norm), and the mix and its norm (with the FP16 router copy) run in
            one pass over the stream on TensorE, ScalarE and VectorE (no GpSimd: the MoE glue issues its gated DMAs
            there). The same conditions as ``attn_res_ahead``. False (default) keeps the program.
        attn_res_ahead_min_entries: with ``attn_res_ahead`` / ``attn_res_ahead_mlp``, a mix takes the ahead form only
            when its bank holds at least this many entries; a mix with fewer keeps the served form (the ahead mix has a
            fixed cost, the served mix grows with its entry loads). 0 (default): every mix of the key.
        attn_res_ahead_vector: with ``attn_res_ahead`` / ``attn_res_ahead_mlp``, the bank parts whose GpSimd share of
            the weighted sums and ``B * g`` runs on VectorE with the same accumulators and order
            (``common.attn_res_ahead_parts`` ``vector_share``): "attn" the calls that build an attention part, "all"
            also the MLP parts (``build_ahead`` / ``AHEAD_MLP_TOKEN_STEPS``), "all_squares" also the squares of each
            part's norm (``squares_vector``). The same FP32 products and adds on another
            engine; a GpSimd entry product takes much longer than on VectorE and stalls VectorE through the shared SBUF
            port. "off" (default) keeps the program.
        attn_res_ahead_load_tiles: with ``attn_res_ahead`` / ``attn_res_ahead_mlp`` and a DRAM bank, the SBUF tiles of
            each bank part's entry loads (``common.attn_res_ahead_parts`` ``load_tiles``): more than 2 let the loads run
            ahead of the TensorE / VectorE reads and leave more entries held for the weighted sums. The same bytes in the
            same order. 2 (default) keeps the program.
        attn_res_ahead_before_resid: with ``attn_res_ahead``, the residual add of a feed-forward layer that builds the
            next KDA layer's ahead parts in its tail follows their weighted sums (each side's ``B * g`` multiply) in the
            schedule, so the VectorE work of the parts runs in the flight of the layer's output collective, ahead of the
            add that waits for it. A schedule edge only: the same instructions and values. False (default) keeps the
            program.
        attn_res_ahead_mlp_kda_tail: with ``attn_res_ahead_mlp``, a KDA layer that builds the bank part of its own MLP
            mix in its body (``AHEAD_MLP_TOKEN_STEPS``) traces the weighted sums and the bank norm after the output
            projection of its last rows (the KDA body's ``tail_hook``), not in the gaps of the token chain: they run
            after the chain, in the gap up to the KDA all-reduce and in its flight, where VectorE is idle. The first
            weighted sum follows the projection's last VectorE copy in the schedule; the part holds all its entries
            (``attn_res_ahead_load_tiles`` raised to the bank size, so no entry is loaded twice), and its norm squares
            run on VectorE (no ScalarE op ahead of the all-reduce's ScalarE DMAs). The same instructions of the part in
            the same order on each engine, and the same exact FP32 squares, so the values do not change. False
            (default) keeps the program.
        attn_res_kda_mlp_tail_entries: N > 0: with ``attn_res_early_sums`` and a DRAM bank, the first min(N, entries)
            entries of the bank sums of a KDA layer's plain MLP mix (traced right after the attention norm, or at the
            KDA body's ``x_quant_hook`` with the MXFP8 input projection) go to the KDA body's ``tail_hook`` instead:
            after the output projection of its last rows, the first sum after the projection's last VectorE copy, so
            they run in the gap up to the KDA all-reduce and in its flight, where VectorE is idle, not in the VectorE
            chain ahead of the body. There the SBUF entries of a split bank are read in place (no entry load beside the
            all-reduce). The later entries keep their place. Both write the same part tile, the same instructions on
            the same entry values, so the values do not change. 0 (default) keeps the program.
        attn_res_mla_mlp_tail_entries: N > 0: with ``attn_res_mla_mlp_early`` and a DRAM bank, the first min(N, entries)
            entries of the bank sums of an MLA layer's MLP mix (traced at the MLA body's ``state_rs_hook``) go to the
            MLA body's ``out_ar_hook`` instead: after the store of the tail's output partials, so they run in the
            flight of the MLA output all-reduce, where VectorE is idle, not ahead of the tail's VectorE work. There the
            SBUF entries of a split bank are read in place (no entry load beside the all-reduce). The later entries
            keep their place. Both write the same part tile, the same instructions on the same entry values, so the
            values do not change. The MLA body must take the hook (``mla_body.out_ar_hook``: the k1 verify with the
            call-level output all-reduce); otherwise, or on an SBUF bank, the key does nothing. 0 (default) keeps the
            program.
        attn_res_gram_merge: the AttnRes mixes that take the ahead form with the TensorE Gram stream sums
            (``common.attn_res_ahead_hT`` ``stats_form`` "gram") take them as one matmul a hidden tile on a comb of the
            stream, the bank part and the weight split (``gram_merge``): the same sums in a third of the matmuls. False
            (default) keeps the program.
        attn_res_cached_act_squares: the cached mixes (``attn_res_cached_bank_sums`` with ``attn_res_early_sums``) of a
            bf16 stream with a weight row (``attn_res_weight_rows``) take the FP32 squares of the stream from ScalarE
            (``common.attn_res_hT`` ``act_stream_squares``); VectorE sums them and runs the weighted rows of the stream
            while ScalarE squares. The same exact squares and sums, so the bits do not change. False (default) keeps
            the program.
        moe_out_proj_form: the out-projection form of the MoE layers (``tkg_moe.OUT_PROJ_FORMS``). ``reduced``
            (default) keeps the program. ``column`` needs the ``column`` ``moe_out_proj`` / ``moe_vectors`` stacks
            (``latent_moe_column_rank_weights``) and changes the output bits; ``reduced_wide`` is its diagnostic with
            the ``reduced`` weights; ``column_shared_ag`` (sel_load) takes the ``column`` stacks and gives the bits of
            ``column``, with the shared activation in its own world gather. Each needs ``sources`` built with the same
            form (``collective_sources``).
        mla_pack_group_plans: with ``mla_packed_loop`` and ``mla_pack_once`` on the loop path above 8 sequences
            (the staged MLA groups of 8, ``mla_staged``), build one plan for each group once for the MLA layers of the
            block (``mla_pack_plan`` without the TI offsets: the item tiles on the heap of the region, 96 bytes per
            partition for each group at a 64-block table), and give each group of each layer its plan. A plan holds
            the positions, the rank and the table shape, not the page ids, so every MLA layer of the block has the same
            items; without it each group builds its items in each layer. Each layer builds the TI offsets, which
            depend on B only, once for its groups. Bitwise the same output. It does nothing elsewhere. False
            (default) keeps the program.
        mla_pack_pairs: with ``mla_packed_loop`` on the loop path at 32 sequences and more (an even count of staged MLA
            groups of 8), groups 2k and 2k + 1 of each MLA layer run one packed loop over their 16 sequences
            (``pack_pair`` of the MLA body), not one loop each: half the loops, so half the loop fences, plans and page
            row gathers, and fewer null slots (8 items of a short context fill half of a 16-slot iteration in a group,
            all of it in a pair). The pair state takes 17 x 513 FP32 columns (34.9 KB a partition) in place of 9.
            Each sequence adds its own key groups in the same order: bitwise the same output. Not with
            ``mla_pack_group_plans``; fails closed below 32 sequences or with an odd group count. False (default)
            keeps the program.
        mla_stage_cc_split: the collectives of the staged MLA groups that run once for each unit of groups, as soon as
            the last group of the unit reaches them, instead of once for all groups (``MLA_STAGE_SPLITS``). A unit is a
            pair of groups with ``mla_pack_pairs``, else one group. "ag" splits the query all-gather: the gather of
            the first unit flies while the next unit computes its queries. "ag_rs" also splits the state
            reduce-scatter: the reduce-scatter of a unit flies while the next unit runs its packed loop. The gather
            moves the same bytes; each element of the reduce-scatter is the sum over the same ranks, but the runtime
            can add them in another order at the smaller message. Each unit has its own HBM tensors of the split
            sites in place of those of the stage (the same bytes). The block's ``sources`` must come from
            ``collective_sources`` with the same split. Needs the staged groups on the packed loop path. "off"
            (default) keeps the program.
        mla_stage_post: above 8 sequences, the staged MLA groups run the work after the state reduce-scatter once for
            the stage (``stage_post`` of the MLA body): one state load and one value projection on 3 x B partitions,
            and one o_proj on all B sequences, so each o_proj weight tile loads once per layer, not once per group.
            Bitwise the same output. Needs the staged MLA groups; False (default) keeps the program.
        mla_pack_pair_plans: with ``mla_pack_pairs`` and ``mla_pack_once``, build one plan for each pair of staged MLA
            groups once for the MLA layers of the block (``mla_pack_plan`` of the 16 sequences without the TI offsets,
            its item tiles on the heap of the region, 128 bytes a partition for each pair at a 64-block table), and give
            the second group of each pair its plan in each layer; without it that group builds the items of the pair in
            each layer. A plan holds the positions, the rank and the table shape, not the page ids, so every MLA layer
            of the block has the same items: bitwise the same output. False (default) keeps the program.
        mla_pack_setup_ahead: the staged MLA groups of the packed loop with a plan for each loop (``mla_pack_pairs`` with
            ``mla_pack_pair_plans``, or ``mla_pack_group_plans``) run the setup of every loop of a layer (the queries,
            the softmax references, the current-key operands, the page rows and the slot rows) before its first loop,
            not each between the loops before and after it (``pack_setup_ahead`` of the MLA body). Each loop gets its
            own HBM slot rows and items, so a later setup does not write the rows that an earlier loop reads; the loops
            share the state tile, so the state of each loop starts right before it. The same instructions on the same
            data: bitwise the same output. False (default) keeps the program.
        mla_pack_state_store_queues: the staged MLA groups of the packed loop store the FP32 state of each loop before
            its reduce-scatter with this many DMAs on as many DMA queues (``pack_state_store_queues`` of the MLA body: 2
            or 3), not one. The same bytes: bitwise the same output. 1 (default) keeps the program.
        mla_current_after_rs: the MLA layers of the packed loop with an FP32 HBM state add the current key once after
            the state reduce-scatter, to the merged state of each local head, in place of 1/32 of it in the state of
            each rank before it (``pack_current_after_rs`` of the MLA body). The FP32 order of the state sums changes,
            so the output is not bitwise the same (GPU-parity class). False (default) keeps the program.
        mla_stage_pre: the staged MLA groups run the work before the query gather once for the stage (``stage_pre`` of
            the MLA body): the first group runs the qkv projection, one qkv gather store and load, the latent queries,
            the current rows and keys, q_b and the key absorb, and one query gather store for all sequences of the stage;
            the others take their rows. Bitwise the same output. Needs the staged groups with mla_stage_cc_split "off".
            False (default) keeps the program.
        moe_latent_shard_form: the base MoE body at most 128 tokens: the form of the transposed latent shard of its
            latent norm (``tkg_moe.LATENT_SHARD_FORMS``, ``latent_shard_form`` of ``tkg_moe.latent_moe_decode``):
            ``columns`` takes 28 one-hot matmuls, a copy and a transpose in place of one matmul for each token. Bitwise
            the same output. "one_hot" (default) keeps the program; another body fails with ``columns``.
        attn_res_kda_mlp_early_entries: for the plain mix (no kept bank rstd) of an SBUF bank above
            ``COMPACT_MIX_TOKENS`` sequences, the MLP mix of a KDA layer sums the squares and weighted rows of its first
            this-many bank entries right after the attention norm (``common.attn_res_plain_bank_part``), where VectorE
            waits for the in-projection of the KDA body. Without this option, the scheduler puts them after the KDA
            output projection, where they delay the attention residual add after the KDA output arrives. The same
            instructions and weights, so the values do not change. Not with ``mix_cache``,
            ``attn_res_cached_bank_sums``, ``attn_res_early_sums`` or ``verify_bank``, and where it acts it needs
            ``attn_res_kda_mlp_early_in_front`` (alone it is slower). 0 (default) keeps the program.
        attn_res_kda_mlp_early_in_front: with ``attn_res_kda_mlp_early_entries``, the KDA body traces the early bank
            sums (``kda_decode_attention`` ``front_hook``): after its conv-state index ops and gathers, with schedule
            edges on them, so the gathers and the conv transposes that the in-projection waits for do not queue
            behind the sums on VectorE. The conv-state scatter rows and the scatter sources then take persistent
            tiles (``kda_conv_index_tiles``; ``kda_conv_rows_tiles`` up to 16 sequences), so the MLP mix cannot take their
            bytes and wait for the scatter. Above ``_KDA_FRONT_ROWS_MAX_TOKENS`` (16) sequences the sources keep their
            transient tiles, and the block reads each scatter's whole source rows and pool-row index once more after
            the feed-forward norm instead, each read after its scatter's completion (``kda_decode_attention``
            ``scatter_keep``, which needs its ``batched_conv_stores``), so no later tile takes those bytes before the
            scatter has read them. The same instructions and weights, so the values do not change. Needs the KDA
            all-reduce sources (``kda_ar_hbm``). False (default) keeps the program.
        attn_res_kda_mlp_early_hbm: with ``attn_res_kda_mlp_early_entries``, the early bank sums also act on a DRAM bank
            (the verify above 8 rows, ``verify_bank``): ``common.attn_res_plain_bank_part`` loads each entry into an
            SBUF tile of its own, during the KDA body, and the MLP mix sums only the other entries and the stream. The
            same instructions on the same bytes, so the values do not change. False (default) keeps the program.
        attn_res_bank_dma: for a DRAM bank, the queue of the bank-entry loads of every mix and of the early bank
            sums (``common.BANK_DMAS``: ``act`` or ``sp``, at the critical priority; the mixes then alternate two entry
            tiles). The same bytes, so the values do not change. None (default) keeps the program.
        norm_one_mul: without ``mix_cache``, the weight multiply of each RMSNorm after a mix is one instruction on the
            repeated weight row (``common.rmsnorm_hT`` ``w_mul_row``) in place of one for each row: an elementwise
            multiply, so the values do not change. False (default) keeps the program.
        attn_res_scalar_squares: for the plain mix (no kept bank rstd) of an SBUF bank above ``COMPACT_MIX_TOKENS``
            sequences, the sums of squares of the last this-many sequences of the stream of every mix, and of the bank
            entries of the attention mix, run on ScalarE, beside the VectorE sums of the others
            (``common._split_square_sums``): ScalarE is idle there, and VectorE runs the mix chain alone. The ScalarE
            rows of the attention mix's bank entries follow the last collective of the previous feed-forward body in
            the schedule; the MLP mix keeps its bank squares on VectorE, where the attention body's ScalarE work runs.
            Each row keeps its bits. Not with ``mix_cache``,
            ``attn_res_cached_bank_sums``, ``attn_res_early_sums`` or ``verify_bank``. 0 (default) keeps the program;
            20 of 32 is the balanced split.
        norm_scalar_squares: above ``COMPACT_MIX_TOKENS`` sequences, the sums of squares of the last this-many sequences
            of each RMSNorm after a mix run on ScalarE in the same way (``common.rmsnorm_hT`` ``act_square_rows``).
            0 (default) keeps the program; 9 of 32 is the balanced split.
        attn_res_scalar_squares_order: with ``attn_res_scalar_squares``, two schedule edges keep the order of the
            plain-mix program around the ScalarE squares of the attention mix. The TensorE transposes of the KDA body's
            conv states follow the mix's ones-sum matmuls (the body's ``conv_after``): with 120 or more ScalarE rows
            in the mix the scheduler puts them first, and the in-order TensorE holds the ones-sum behind transposes
            that wait for the conv-state loads. The bank push of a block's last layer follows the mix's last ScalarE
            square: else the scheduler puts the push, which waits for the stream, among the squares, and the squares
            after it wait too. The same instructions, so the values do not change. The KDA body must take
            ``conv_after``. False (default) keeps the program.
        attn_res_scalar_products_stt: with ``attn_res_scalar_products``, the mixes (``attn_res_hT`` and
            ``attn_res_tensor_hT``) give each VectorE row of an entry after the first one VectorE multiply-add in place of its multiply and its
            part of the add (``common._mix_entries`` ``stt``; MEASURED on trn3 HW: two roundings, IEEE zero signs; the
            same bits). False (default) keeps the program.
        attn_res_scalar_products: for the plain mix (no kept bank rstd) of an SBUF bank above ``COMPACT_MIX_TOKENS``
            sequences with at least 2 entries, the products of the last this-many sequences of each bank entry after the
            first run on ScalarE, one ``tensor_scalar`` multiply of a sequence's row by its softmax weight each, in
            every mix (``common.attn_res_hT`` ``act_product_rows``). The entry mix is a serial VectorE chain of a
            multiply and an add for each entry, and ScalarE is idle there. VectorE keeps the other products and every
            add, in candidate order, so the values do not change. It also acts on the verify bank in DRAM
            (``verify_bank``) and on the mixes of the kept bank rstd (``attn_res_cached_bank_sums``), also with
            ``attn_res_early_sums``. Not with ``mix_cache``. 0 (default) keeps the program.
        attn_res_tensor_sums: above ``COMPACT_MIX_TOKENS`` sequences, on trn3, the mixes of a decode bank take the GPU
            form of ``common.attn_res_tensor_hT``: the weighted sums of every candidate run on TensorE (exact products
            of the bf16 rows with a two-part bf16 split of the score weight, FP32 sums), the bank entries keep their
            rstd from the push (``bank`` is then ``(bank_hT, count, bank_rstd)``), and the mix stays an FP32
            accumulator that its norm rounds once (``common.rmsnorm_hT`` ``one_round``). The bank-row matmuls of the
            attention mix follow the last collective of the previous feed-forward body, and those of the MLP mix the
            attention output collective, so TensorE runs them in the flights of those collectives. Not bitwise: the sum
            order changes, and the mix and the norm round once, not three times (the GPU form, GPU parity).
            ``attn_res_scalar_squares``, ``norm_scalar_squares`` and ``attn_res_scalar_products`` keep their ScalarE
            rows (the stream squares, the norm squares, the entry products); ``attn_res_kda_mlp_early_entries`` has no
            work left and is inert, and so is ``norm_one_mul`` (the norm writes one fused multiply for each row).
            With ``verify_bank`` (the DRAM bank of the verify) the mixes load each bank entry into two SBUF tiles in
            turn (``attn_res_bank_dma`` sets their queue), the bank push keeps its wait on the attention mix, and
            ``before_attention`` gets the FP32 mix. Not with ``mix_cache``, ``attn_res_cached_bank_sums``,
            ``attn_res_early_sums`` or ``attn_rs_moe_input`` (its one-row glue keeps its mix). False (default) keeps
            the program; the verify entry takes it as ``verify_model`` ``attn_res_tensor_sums``.
        attn_res_one_round_norm: the GPU-parity norm of every mix, without TensorE sums (form F0): each mix keeps its
            weighted sums, squares, softmax and entry products, but returns its FP32 accumulator (``common.attn_res_hT``
            ``out_dtype``), and the norm after it rounds once (``common.rmsnorm_hT`` ``one_round``: ``bf16(x * rstd *
            w)``, the FP32 product kept), not three times. The norm keeps its sums of squares (``norm_scalar_squares``
            keeps its ScalarE rows) and writes one fused multiply for each token, in place of the rstd multiply of all
            tokens and the weight multiply of each token. The KDA MLP mix keeps its early bank sums
            (``attn_res_kda_mlp_early_entries``) and their conv-state scatter protection. Not bitwise: the GPU form of
            the norm (GPU parity). ``norm_one_mul`` is inert with it, and ``before_attention`` gets the FP32 mix. Not
            with ``mix_cache`` (its norms take weight rows) or ``attn_rs_moe_input`` (its one-row glue keeps its mix).
            False (default) keeps the program.
        attn_res_tensor_sums_idle: the GPU form of ``attn_res_tensor_sums`` only at the mixes where TensorE is idle in
            decode (form F1): every mix of an MLA layer and the attention mix of a KDA layer take
            ``common.attn_res_tensor_hT``; the MLP mix of a KDA layer (and of the dense layer 0) keeps its default sums
            on its early path (``attn_res_kda_mlp_early_entries`` and ``attn_res_kda_mlp_early_in_front`` keep acting,
            with the persistent conv-index tiles and ``scatter_keep``), because there TensorE runs the KDA state update
            in the flight of the attention all-reduce. Every mix returns its FP32 accumulator and its norm rounds once
            (``attn_res_one_round_norm``), so the whole block has the GPU-parity norm. The bank entries keep their rstd
            from the push (``bank`` is ``(bank_hT, count, bank_rstd)``), as with ``attn_res_tensor_sums``. Not with
            ``attn_res_tensor_sums`` (it is its subset) and, as that key, not at 4 sequences and fewer, off trn3, nor
            with ``mix_cache``, ``attn_res_cached_bank_sums``, ``attn_res_early_sums``, ``verify_bank``,
            ``attn_rs_moe_input`` or ``before_attention``. False (default) keeps the program.
        kda_w_in_prefetch: None (default), or the ``KdaWInPrefetch`` of the kernel: each MoE layer whose next layer is
            a KDA layer issues the first fills of chunks ``first ..`` of that layer's ``kda_w_in`` ring inside its routed
            all-reduce (the ``cc_window`` of the Trn2 MoE body: at the one-op all-reduce's start,
            or after the host reduce-scatter of ``routed_split``), and the KDA body takes the ring (``w_in_ring``). These
            fills are not weight-gated; the other KDA loads keep their gates. The last layer of the block prefetches for
            the first layer of the next block from ``next_w_in``; the next block call takes the ring from ``pending``.
            Give the KDA body a ring of the same size (its ``w_in_buffers``). The values do not change. Fails with a MoE
            body without the window. With the ``after_ffn_cc`` place (Trn3, ``trn3_kda_w_in_prefetch``) every
            feed-forward layer whose next layer is a KDA layer issues that ring (and the MXFP8 scales) after its first
            output collective instead, at the carry's DMA level, with any MoE body.
        mla_head_prefetch: None (default), or the ``MlaHeadPrefetch`` of the kernel: each MoE layer whose next layer is
            an MLA layer of the block issues the BF16 qkv and q_b weight loads of that layer's first MLA call inside its
            routed all-reduce (``cc_window``, as ``kda_w_in_prefetch``), and the MLA body takes them
            (``head_prefetch``, ``tkg_mla_paged.prefetch_mla_head``). These loads are not weight-gated; the
            other MLA loads keep their gates. MLA layers are never the first layer of a block, so no load crosses a
            block. Not with the grouped MLA forms (more than 8 sequences) or an MXFP8 ``mla_in_proj``. The values do
            not change. Fails with a MoE body without the window.
        kda_gap_loads: ``KDA_GAP_LOADS``; ``off`` (default) keeps the program. A KDA layer has a gap of DMA-quiet compute
            (the recurrence) between the end of its input projection and its output all-reduce. ``w_out``: the KDA
            body's ``kda_w_out`` load waits for its last ``kda_w_in`` loads (``kda_decode_attention``
            ``w_out_after_w_in``), so it streams in that gap, not in the flight of the output collective of the MoE
            layer before. ``in_proj``: the in-projection weight loads of the MoE layer after a KDA layer follow the end
            of the KDA input projection (a schedule edge and a semaphore wait, ``tkg_weight_dma_gate.gate_all_dmas`` ``early``) in
            place of the KDA output all-reduce, so they stream in that gap, not in the all-reduce's flight; the MoE
            layer's other loads keep their gates, and MLA layers do not change. ``both``: the two. Needs the DMA gates
            (not ``per_queue``); ``in_proj`` needs the Trn2 MoE body (in-projection first in its weights) and no
            ``ffn_weights_after_attn``, and takes the BF16 ``kda_w_in``. The values do not change.
        kda_w_in_heads: None (default), or the ``kda_w_in`` heads of KDA layers, ``{layer: (address, chunks, kda_w_in)}``
            with the global layer, the explicit SBUF byte of the first chunk, the chunk count and the layer's HBM
            ``kda_w_in``. The 2-server MoE layer before such a layer loads the first ``chunks`` chunks of its ``kda_w_in``
            into SBUF at the address in its output glue, right after the load of the result of its routed reduction's
            gate collective (the routed all-reduce, or the TP32 all-gather of ``rs_ag``; the MoE body's ``routed_load``
            glue hook): a semaphore wait on that load, on the ACT queue (``tkg_kda.kda_w_in_head_loads``). The KDA
            body reads them as its ``w_in_head``. With a region, the head bytes must be clear of the region's stack and
            heap where the head is loaded and where it is read (checked). Needs the 2-server MoE body and the BF16
            ``kda_w_in``. The values do not change.
        kda_ahead_carry: with ``kda_w_in_heads``, the dict that holds the heads that one block call loads for the first
            layer of the next block call (the entry passes the same dict to each block). None: a dict of the block.
        kda_front_ahead: the 2-server MoE layer before each KDA layer of the block traces that layer's front in the
            flight of its output collective (the TP32 output all-reduce, or the output all-gather of the column out
            projection; the MoE body's ``out_trigger`` glue hook; ``tkg_kda.kda_decode_front_ahead``: page counts,
            conv scatter rows, conv gathers, transposes and copy, and the recurrent-state loads on the ACT queue); its
            residual add follows the front's VectorE ops in the schedule. Needs a KDA body with the ``front_ahead``
            attribute (``tkg_kda.kda_front_ahead_body``) and the 2-server MoE body; not with ``kda_conv_keep``. The
            values do not change; False (default) keeps the program.
        moe_in_proj_hook: None (default), or a function that each 2-server MoE layer calls with the DMAs of its MXFP8
            in-projection load, right after them (``tkg_moe_2srv`` glue hook ``in_proj_loads``): the caller traces
            work there, ordered after those loads on their queues. None keeps the program.
        moe_in_proj_at_kda_reduce: each MoE layer after a KDA layer loads its in-projection (``moe_in_proj_dp``,
            the router, latent-down and shared gate / up columns) right after the KDA body issues its attention
            all-reduce, before the body's recurrent-state stores in the program (``kda_decode_attention``
            ``reduce_hook``; the body's ``load_in_proj``, ``tkg_moe_2srv.load_moe_in_proj_2srv``), and the MoE body
            reads that tile (``in_proj_sb``). The loads keep the gate of the MoE weights (a schedule edge after that
            all-reduce), and their tile is live with the staged recurrent states, so the compiler gives it other bytes
            and the loads do not wait for the state stores. Needs the 2-server MoE body and the DMA gates without
            ``ffn_weights_after_attn``. The values do not change; False (default) keeps the program.
        moe_2srv_selectors_once: the 2-server MoE body's latent-shard and expert-group one-hots (its rank-row load and
            selector ops) are built once, at the start of the first block call, at heap addresses on the MLA loop path
            (``tkg_moe_2srv.moe_2srv_selectors``; the body's ``build_selectors``), and every MoE layer reuses them,
            as the rank selectors. The rank-row load then does not queue behind the attention DMAs of each layer.
            Needs the 2-server MoE body. The values do not change; False (default) keeps the program.
        kda_prework_window: with a KDA body that has a ``prework`` (the Trn2 entries' ``kda_prework_ahead``), each MoE
            layer before a KDA layer of the block traces that layer's prework inside its routed all-reduce (the
            ``cc_window`` of the Trn2 MoE body, after the window's ``kda_w_in_prefetch`` / ``mla_head_prefetch`` loads
            of that layer, ``_window_with_prework``), not after the MoE body, so its loads issue before the load of the
            routed result. Other feed-forward layers keep the prework after their body. The same instructions on the
            same bytes: the values do not change. False (default) keeps the program. Fails with a MoE body without the
            window.
        moe_prefetch: None (default) keeps the program. A dict ``{"kda": W, "mla": W}`` starts the expert ring of each
            MoE layer at the start of the layer's attention (or at the MoE front, ``moe_prefetch_start``), with W
            whole weight buffers by the attention type: the MoE body's ``prefetch`` (the static expert loop's first
            loads, ``moe_routed.static_expert_stream``) runs right after the attention weights' gate on the previous
            feed-forward body, so the loads stream during the attention and the layer's expert call starts on a loaded
            ring. W = 0 keeps the expert call's own first loads for the layers of that type. Needs a MoE body with a
            ``prefetch`` (the 2-server body) and the static routed kernel. With ``moe_cross_expert_order`` the ring
            loads in the cross-expert order, and W is 0 or ``moe_weight_buffers`` (3). The values do not change.
        moe_prefetch_priority: None (default) keeps the program. With ``moe_prefetch``, the trn3 DMA QoS level (0 to
            3, 0 is the highest) of the ring loads (``static_expert_stream`` ``priority``). The attention loads keep
            their level, so at 3 the ring takes only the DMA time that the attention leaves: it streams in the
            attention's idle DMA windows and does not delay its weight and state loads. No level on trn2 (no DMA
            QoS). The values do not change.
        moe_prefetch_uses: None (default) keeps the program. With ``moe_prefetch`` and ``moe_cross_expert_order``, a
            dict ``{"kda": k}`` / ``{"mla": k}`` (0 to the type's W): the ring of that layer type loads only its first
            k buffers (and its scale groups) at the ring start (during the attention, or at the MoE front with
            ``moe_prefetch_start``), and the expert call loads the others at its start (``static_expert_stream``
            ``prefetch_uses``), so only k weight buffers hold bytes before the call.
            The values do not change.
        moe_prefetch_scales: None (default) keeps the program. With ``moe_prefetch_uses``, ``{"mla": False}`` (or
            ``"kda"``) leaves that layer type's scale groups to the expert call's start (``prefetch_scales``). The values
            do not change.
        moe_prefetch_kda_after_in_proj: False (default) keeps the program. True, with a KDA ring (``moe_prefetch``
            ``kda`` >= 1): the expert ring of a MoE layer after a KDA attention starts right after that attention's
            input projection, not before the attention. The KDA body calls back after the projection (its
            ``in_proj_hook``): every ring load follows the projection's last two ``kda_w_in`` loads in the schedule,
            and the first ring load of each queue waits for their completion (``_dmas_after_w_in``), so the ring and
            the ``kda_w_in`` stream do not share the DMA rings. The layers after an MLA attention keep their
            placement. The values do not change.
        moe_prefetch_kda_queue_order: False (default) keeps the program. True, with
            ``moe_prefetch_kda_after_in_proj``: the later ring loads of each queue also follow the queue's first ring
            load in the schedule, so the first one (the one that waits for the projection's last ``kda_w_in`` loads)
            starts the queue, and no ring load shares the DMA rings with those loads. The values do not change.
        moe_prefetch_kda_before_slot_stores: False (default) keeps the program. True, with
            ``moe_prefetch_kda_after_in_proj``: the KDA body's slot state stores of the first scored token follow every
            load of the layer's expert ring in the schedule (``kda_verify_attention`` ``in_proj_hook``, marks key
            ``before_slot_stores``), so the in-order queue issues the whole ring before the store chain. Else the ring
            loads that the scheduler puts after the first store wait behind each store's wait for the store before it
            and go out one a store, in the chain's flight: each store then shares the DMA rings with ring bytes, and its
            flight takes longer (MEASURED on the 93-layer C = 8 1M profile). The ring is not
            needed before the expert call. The values do not change.
        moe_prefetch_start: None (default) keeps the program. ``{"mla": "front"}``, with an MLA ring (``moe_prefetch``
            ``mla`` >= 1): the expert ring of a MoE layer after an MLA attention starts at the MoE front, not before
            the attention. The prefetch is traced right before the MoE body, after the MLA frame, in the feed-forward
            weight gate, so the ring loads follow the attention output collective (and the residual add with
            ``ffn_weights_after_attn``) and are issued before the router and the router-row gathers of the body. The
            ring is not live across the MLA attention, whose tiles have explicit addresses. The layers after a KDA
            attention keep their placement. The values do not change.
        moe_hot_prefetch: None (default) keeps the program. A dict ``{"kda": N, "mla": N}`` loads the hot experts 0 ..
            N - 1 of each sel_load MoE layer (N by the type of the layer's attention, whole groups of 16, at most 64;
            0 for none) into explicit SBUF tiles during the layer's attention (``hot_expert_tiles``): the gate/up,
            down and down-scale loads on the ACT hardware DGE queue wait for the last reads of the tiles of the layer
            before (``tkg_weight_dma_gate.PrefetchGate``), not for the attention output collective. A MoE body with the attribute
            ``hot_experts`` gets the tiles as the dict argument ``hot_experts`` (it can use them as its ring slots, and
            can put the op of its last access of each tile in ``hot_experts["last_use"][key]``; else its output
            collective is the last use), and nothing reads the tiles after the MoE. For another body the block reads
            one word of each tile after the MoE output collective, so the loads stay in the program. Needs the
            sel_load routed kernel and a region. The values do not change.

    Returns:
        ``(stream, bank, released)``: the SBUF ``[128, B, H/128]`` stream after the last layer, the
        bank ``(bank_hT, count)`` after the block (None when it stays empty), and the output
        collectives of the last body (the ``gate`` of the next block).
    """
    sbuf_stream = len(stream.shape) == 3
    T = stream.shape[1] if sbuf_stream else stream.shape[0]
    n_tiles = stream.shape[2] if sbuf_stream else stream.shape[1] // P_MAX
    num_layers = norm_weights.shape[0]
    n_kda, n_mla, n_moe, n_dense = block_layer_counts(first_layer, num_layers)
    kernel_assert(len(kv_pools) == num_layers, "Give one page pool for each layer")
    # A verify entry passes a layer-index stand-in for kda_page_ids (no dtype): only tensors are checked.
    for table_name, table in (("kda_page_ids", kda_page_ids), ("mla_block_table", mla_block_table)):
        kernel_assert(
            not hasattr(table, "dtype") or table.dtype == nl.int32,
            f"{table_name} must be INT32 page ids (DMAs gather by them and do not take a 64-bit dtype), "
            f"got {getattr(table, 'dtype', None)}",
        )
    kernel_assert(
        all(getattr(b, "token_major", False) for b in (kda_body, mla_body, moe_body, dense_body)),
        "Every body must take the token-major tile [128, B, H/128]",
    )

    kernel_assert(region is not None or not mla_loop_path, "The MLA loop path needs an explicit SBUF region")
    kernel_assert(routed_kernel in ROUTED_KERNELS, f"routed_kernel must be one of {ROUTED_KERNELS}")
    kernel_assert(
        not moe_prefetch
        or (
            routed_kernel == "static"
            and callable(getattr(moe_body, "prefetch", None))
            and set(moe_prefetch) == {"kda", "mla"}
        ),
        f"moe_prefetch needs the static routed kernel, a MoE body with a prefetch and the keys kda and mla, got "
        f"{moe_prefetch} with routed_kernel {routed_kernel!r}",
    )
    kernel_assert(
        moe_prefetch_priority is None or (moe_prefetch and moe_prefetch_priority in (0, 1, 2, 3)),
        f"moe_prefetch_priority must be None, or a DMA QoS level 0 to 3 with moe_prefetch, got {moe_prefetch_priority}",
    )
    kernel_assert(
        not moe_prefetch_uses
        or (
            moe_prefetch
            and moe_cross_expert_order
            and all(kind in moe_prefetch and 0 <= k <= moe_prefetch[kind] for kind, k in moe_prefetch_uses.items())
        ),
        f"moe_prefetch_uses needs moe_prefetch and moe_cross_expert_order, and 0 to W uses of a prefetched layer type, "
        f"got {moe_prefetch_uses} with {moe_prefetch}",
    )
    kernel_assert(
        not moe_prefetch_kda_after_in_proj
        or (moe_prefetch and moe_prefetch.get("kda", 0) > 0 and getattr(kda_body, "in_proj_hook", False)),
        f"moe_prefetch_kda_after_in_proj needs a KDA expert ring (moe_prefetch kda >= 1) and a KDA body with an "
        f"in_proj_hook, got moe_prefetch {moe_prefetch}",
    )
    kernel_assert(
        not moe_prefetch_kda_before_slot_stores or moe_prefetch_kda_after_in_proj,
        "moe_prefetch_kda_before_slot_stores needs moe_prefetch_kda_after_in_proj (the ring that the KDA body starts)",
    )
    kernel_assert(
        not moe_prefetch_scales or (moe_prefetch_uses and set(moe_prefetch_scales) <= set(moe_prefetch_uses)),
        f"moe_prefetch_scales takes the layer types of moe_prefetch_uses, got {moe_prefetch_scales}",
    )
    kernel_assert(
        not moe_prefetch_start
        or (moe_prefetch_start == {"mla": "front"} and moe_prefetch and moe_prefetch.get("mla", 0) > 0),
        f"moe_prefetch_start must be None, {{}} or {{'mla': 'front'}} with an MLA expert ring (moe_prefetch mla >= 1), "
        f"got {moe_prefetch_start} with moe_prefetch {moe_prefetch}",
    )
    hot_counts = check_moe_hot_prefetch(moe_hot_prefetch, routed_kernel, region)
    kernel_assert(
        routed_reduce_dtype in ROUTED_REDUCE_DTYPES, f"routed_reduce_dtype must be one of {ROUTED_REDUCE_DTYPES}"
    )
    # The MoE options of a non-default routed kernel. The static kernel keeps the call of the base body.
    routed = {}
    if routed_kernel != "static":
        routed.update(routed_kernel=routed_kernel, region=region)
    if routed_reduce_dtype != "bf16":
        routed.update(routed_reduce_dtype=routed_reduce_dtype)
    if affinity_scaling != "pre_quant":
        routed.update(affinity_scaling=affinity_scaling)
    if not affinity_hi_lo:
        routed.update(affinity_hi_lo=False)
    if routing_form is not None:
        routed.update(routing_form=routing_form)
    if router_mode is not None:
        routed.update(router_mode=router_mode)
    if moe_sel_load_options:
        kernel_assert(routed_kernel == "sel_load", "moe_sel_load_options needs routed_kernel='sel_load'")
        routed.update(sel_load_options=tuple(moe_sel_load_options))
    kernel_assert(
        "split_gather" not in tuple(moe_glue_dmas) or (routed_kernel not in TP128_KERNELS and not moe_x_path_act),
        "moe_glue_dmas 'split_gather' runs in the base MoE body of the EP128 static kernel, without moe_x_path_act "
        "(both place the x gather DMAs)",
    )
    if moe_glue_dmas and routed_kernel in TP128_KERNELS:
        routed.update(glue_dmas=tuple(moe_glue_dmas))
    # The EP128 static kernel takes the replica-rows load and the split gather DMAs only, in the base MoE body
    # (tkg_moe.BASE_GLUE_DMA_FORMS).
    # It goes with the base body options, so the block keeps that body.
    base_glue = tuple(moe_glue_dmas) if moe_glue_dmas and routed_kernel not in TP128_KERNELS else ()
    kernel_assert(
        all(form in BASE_GLUE_DMA_FORMS for form in base_glue),
        f"moe_glue_dmas {tuple(moe_glue_dmas)}: the EP128 static kernel takes {BASE_GLUE_DMA_FORMS} only; the others "
        "need a TP128 routed kernel",
    )
    if moe_routing_tiles:
        kernel_assert(routed_kernel in TP128_KERNELS, "moe_routing_tiles needs a TP128 routed kernel")
        routed.update(routing_tiles=True)
    kernel_assert(router_dtype in ROUTER_DTYPES, f"router_dtype must be one of {ROUTER_DTYPES}")
    kernel_assert(moe_in_proj_form in IN_PROJ_FORMS, f"moe_in_proj_form must be one of {IN_PROJ_FORMS}")
    kernel_assert(
        moe_in_proj_form == "gathered" or moe_peer_in_projs is not None,
        "moe_in_proj_form 'dp_local' needs moe_peer_in_projs",
    )
    kernel_assert(moe_out_proj_form in OUT_PROJ_FORMS, f"moe_out_proj_form must be one of {OUT_PROJ_FORMS}")
    kernel_assert(
        moe_early_loads in MOE_EARLY_LOADS, f"moe_early_loads {moe_early_loads!r} must be one of {MOE_EARLY_LOADS}"
    )
    kernel_assert(
        router_dtype == "bf16" or moe_router_weights is not None, "router_dtype='fp16' needs moe_router_weights"
    )
    kernel_assert(
        moe_in_proj_scales is None or (router_dtype != "bf16" and getattr(moe_body, "in_proj_mx", False)),
        "moe_in_proj_scales (the MXFP8 MoE in-projection) needs the FP16 router and a MoE body with in_proj_mx",
    )
    kernel_assert(
        moe_out_proj_scales is None or getattr(moe_body, "out_proj_mx", False),
        "moe_out_proj_scales (the MXFP8 MoE out-projection) needs a MoE body with out_proj_mx",
    )
    kernel_assert(
        attn_res_kda_mlp_early_entries >= 0
        and not (
            attn_res_kda_mlp_early_entries
            and (
                mix_cache
                or attn_res_cached_bank_sums
                or attn_res_early_sums
                or (verify_bank and not attn_res_kda_mlp_early_hbm)
            )
        ),
        "attn_res_kda_mlp_early_entries is a count >= 0 for the plain mix of the decode bank: not with mix_cache, "
        "attn_res_cached_bank_sums, attn_res_early_sums or verify_bank (verify_bank takes attn_res_kda_mlp_early_hbm)",
    )
    kernel_assert(
        not attn_res_kda_mlp_early_hbm or (attn_res_kda_mlp_early_entries > 0 and not attn_res_kda_mlp_early_in_front),
        "attn_res_kda_mlp_early_hbm needs attn_res_kda_mlp_early_entries and takes no attn_res_kda_mlp_early_in_front",
    )
    kernel_assert(
        not attn_res_kda_mlp_early_in_front or attn_res_kda_mlp_early_entries > 0,
        "attn_res_kda_mlp_early_in_front needs attn_res_kda_mlp_early_entries",
    )
    if attn_res_kda_mlp_early_entries and not attn_res_kda_mlp_early_in_front and T > COMPACT_MIX_TOKENS:
        # The early sums act with an SBUF bank (kda_mlp_early below): the given bank, or the bank that the block makes
        # (in SBUF without a region, and in the region up to _BANK_HBM_ROWS rows; the entries refuse verify_bank above).
        given_bank = None if bank is None else bank[0]
        sbuf_bank = given_bank.buffer == nl.sbuf if given_bank is not None else region is None or T <= _BANK_HBM_ROWS
        kernel_assert(
            not sbuf_bank,
            "attn_res_kda_mlp_early_entries needs attn_res_kda_mlp_early_in_front above COMPACT_MIX_TOKENS sequences: "
            "alone the conv-state index ops queue behind the sums and the MLP mix waits for the conv-state scatter",
        )
    kernel_assert(
        attn_res_scalar_squares == 0
        or (
            T > COMPACT_MIX_TOKENS
            and 0 < attn_res_scalar_squares < T
            and not (mix_cache or attn_res_cached_bank_sums or attn_res_early_sums or verify_bank)
        ),
        f"attn_res_scalar_squares is 0 < rows < {T} above {COMPACT_MIX_TOKENS} sequences, for the plain mix of the "
        "decode bank: not with mix_cache, attn_res_cached_bank_sums, attn_res_early_sums or verify_bank",
    )
    kernel_assert(
        norm_scalar_squares == 0 or (T > COMPACT_MIX_TOKENS and 0 < norm_scalar_squares < T),
        f"norm_scalar_squares is 0 < rows < {T} above {COMPACT_MIX_TOKENS} sequences",
    )
    kernel_assert(
        not attn_res_scalar_squares_order or attn_res_scalar_squares,
        "attn_res_scalar_squares_order orders the ScalarE squares of attn_res_scalar_squares, which is 0",
    )
    kernel_assert(
        attn_res_scalar_products == 0
        or (T > COMPACT_MIX_TOKENS and 0 < attn_res_scalar_products <= T and not mix_cache),
        f"attn_res_scalar_products is 0 < rows <= {T} above {COMPACT_MIX_TOKENS} sequences: not with mix_cache",
    )
    kernel_assert(
        attn_res_bank_dma is None or attn_res_bank_dma in BANK_DMAS,
        f"attn_res_bank_dma must be one of {BANK_DMAS} or None, got {attn_res_bank_dma!r}",
    )
    kernel_assert(
        not (attn_res_tensor_sums and attn_res_tensor_sums_idle),
        "attn_res_tensor_sums_idle is the subset of attn_res_tensor_sums at the TensorE-idle mixes: give one of them",
    )
    kernel_assert(
        not attn_res_tensor_sums
        or (
            T > COMPACT_MIX_TOKENS
            and not (mix_cache or attn_res_cached_bank_sums or attn_res_early_sums or attn_rs_moe_input)
            and is_trn3()
        ),
        f"attn_res_tensor_sums is for the AttnRes bank above {COMPACT_MIX_TOKENS} rows on trn3: not with mix_cache, "
        "attn_res_cached_bank_sums, attn_res_early_sums or attn_rs_moe_input",
    )
    kernel_assert(
        not attn_res_tensor_sums_idle
        or (
            T > COMPACT_MIX_TOKENS
            and before_attention is None
            and not (mix_cache or attn_res_cached_bank_sums or attn_res_early_sums or verify_bank or attn_rs_moe_input)
            and is_trn3()
        ),
        f"attn_res_tensor_sums_idle is for the decode bank above {COMPACT_MIX_TOKENS} sequences on trn3: not with "
        "mix_cache, attn_res_cached_bank_sums, attn_res_early_sums, verify_bank, attn_rs_moe_input or before_attention",
    )
    kernel_assert(
        not attn_res_one_round_norm or not (mix_cache or attn_rs_moe_input),
        "attn_res_one_round_norm is the norm of an FP32 mix with one rounding: not with mix_cache (weight-row norms) "
        "or attn_rs_moe_input (its one-row glue keeps its mix)",
    )
    # The DP4 x TP32 and all-to-all-v MoE bodies take none of the TP128 routing and in-projection options.
    kernel_assert(
        getattr(moe_body, "moe_sharding", "tp128") != "dp4xtp32"
        or (moe_in_proj_form == "gathered" and routing_form is None and not moe_sel_load_options),
        "moe_in_proj_form, routing_form and moe_sel_load_options apply to the TP128 MoE bodies; the DP4 x TP32 and "
        "all-to-all-v MoE bodies take none of them",
    )
    # The feed-forward tail order makes the decode with the EP128 static MoE body (routed_kernel "static" of the TP128
    # projections) slower, so the block rejects that combination.
    kernel_assert(
        not ffn_tail_order or routed_kernel != "static" or getattr(moe_body, "moe_sharding", "tp128") == "dp4xtp32",
        "ffn_tail_order is slower with the EP128 static MoE body (routed_kernel 'static'); use it with sel_load only",
    )
    kernel_assert(
        not (ffn_tail_order and row_sums_after_ffn_output),
        "row_sums_after_ffn_output and ffn_tail_order both order the next layer's bank sums: give one of them",
    )
    check_kda_states_after_out_ar(kda_states_after_out_ar, moe_body, kda_body)
    kernel_assert(
        not kda_small_after_out_ar or getattr(kda_body, "small_after_result", False),
        "kda_small_after_out_ar needs a KDA body that takes small_after_result (the verify body, "
        "verify_model.verify_bodies)",
    )
    kernel_assert(
        getattr(moe_body, "moe_sharding", "tp128") != "dp4xtp32" or not moe_glue_dmas,
        "moe_glue_dmas applies to the TP128 MoE bodies",
    )
    kernel_assert(
        not moe_x_path_act or moe_x_world_gather,
        "moe_x_path_act needs moe_x_world_gather: it moves the DMAs of the world x gather",
    )
    kernel_assert(
        not moe_cross_expert_transpose_blocks or moe_cross_expert_order,
        "moe_cross_expert_transpose_blocks orders the activation transposes of moe_cross_expert_order: it needs "
        "moe_cross_expert_order",
    )
    if moe_x_world_gather:
        # The second x gather destination comes only with given sources (``kimi_k3_decode_model`` moe_dst_pingpong).
        pingpong = (sources or {}).get("pingpong") or {}
        check_x_world_gather(mla_stage_cc_split, "x_dst_hbm" in pingpong)
    if attn_rs_moe_input:
        check_attn_rs_moe_input(T, dp_size, moe_body, moe_in_proj_form, mla_cp, mla_stage_cc_split, moe_x_world_gather)
    kernel_assert(
        not moe_in_proj_at_kda_reduce or getattr(moe_body, "load_in_proj", None) is not None,
        "moe_in_proj_at_kda_reduce needs the 2-server MoE body (moe_decode_2srv.moe_2srv_body)",
    )
    kernel_assert(
        not moe_in_proj_at_kda_reduce or (dma_gates is True and not ffn_weights_after_attn),
        "moe_in_proj_at_kda_reduce keeps the schedule gate of the MoE weights: it needs dma_gates True without "
        "ffn_weights_after_attn",
    )
    # kda_w_in_heads / kda_front_ahead: work of the next KDA layer traced in the glue of a 2-server MoE layer.
    kda_w_in_heads = dict(kda_w_in_heads or {})
    kda_ahead_carry = {} if kda_ahead_carry is None else kda_ahead_carry
    # kda_states_after_out_ar: layer -> the MoE output all-reduce of the layer before it.
    kda_states_after = {}
    front_ahead_of = getattr(kda_body, "front_ahead", None) if kda_front_ahead else None
    kernel_assert(
        not (kda_w_in_heads or kda_front_ahead) or getattr(moe_body, "moe_sharding", "tp128") == "dp_2srv",
        "kda_w_in_heads and kda_front_ahead need the 2-server MoE body (its glue hooks)",
    )
    kernel_assert(
        not kda_front_ahead or front_ahead_of is not None,
        "kda_front_ahead needs a KDA body with the front_ahead attribute (kda_decode.kda_front_ahead_body)",
    )
    kernel_assert(
        not (kda_front_ahead and kda_conv_keep),
        "kda_front_ahead writes the conv scatter rows ahead, without the conv-store waits of kda_conv_keep",
    )
    kernel_assert(
        not kda_front_ahead or getattr(kda_body, "prework", None) is None,
        "kda_front_ahead and a KDA body with prework both trace the next KDA layer's page work: use one",
    )
    kernel_assert(
        not kda_prework_window or getattr(kda_body, "prework", None) is not None,
        "kda_prework_window traces the KDA body's prework in the MoE window: it needs a KDA body with prework",
    )
    kernel_assert(
        not kda_w_in_heads or len(kda_weights) == len(KDA_WEIGHT_NAMES),
        "kda_w_in_heads take the BF16 kda_w_in, not an MXFP8 one",
    )
    kernel_assert(
        not (kda_w_in_heads or kda_front_ahead) or (kda_w_in_prefetch is None and mla_head_prefetch is None),
        "kda_w_in_heads and kda_front_ahead (the 2-server glue hooks) do not take kda_w_in_prefetch or "
        "mla_head_prefetch (the Trn2 MoE window)",
    )
    for head_layer in kda_w_in_heads:
        kernel_assert(
            not is_mla_layer(head_layer) and not is_dense_layer(head_layer - 1),
            f"a kda_w_in head is for a KDA layer after a MoE layer, got layer {head_layer}",
        )
    kernel_assert(
        kda_gap_loads in KDA_GAP_LOADS, f"kda_gap_loads must be one of {KDA_GAP_LOADS}, got {kda_gap_loads!r}"
    )
    gap_w_out = kda_gap_loads in ("w_out", "both")
    gap_in_proj = kda_gap_loads in ("in_proj", "both")
    kernel_assert(
        kda_gap_loads == "off" or (dma_gates and dma_gates != "per_queue"),
        f"kda_gap_loads {kda_gap_loads} moves gated weight loads: it needs the DMA gates (not per_queue)",
    )
    kernel_assert(
        not gap_in_proj or (getattr(moe_body, "cc_window", False) and not ffn_weights_after_attn),
        f"kda_gap_loads {kda_gap_loads} gates the in-projection of the Trn2 MoE body (moe_form mxfp4), without "
        "ffn_weights_after_attn (which holds that load after the attention output)",
    )
    kernel_assert(
        kda_gap_loads == "off" or len(kda_weights) == len(KDA_WEIGHT_NAMES),
        f"kda_gap_loads {kda_gap_loads} takes the BF16 kda_w_in, not an MXFP8 one",
    )
    if region is not None:
        region.open_frame(f"block_{first_layer}")
        ones = region_ones(region)
        streams = region_streams(region, T, n_tiles, stream.dtype)
    else:
        ones = new_ones()
    # The per-layer vector weights of the whole block load once, two DMAs of about 8 KB per partition. Above 4
    # sequences on the loop path, each layer loads its own in a layer frame: the 8 KB that the block keeps
    # for the whole block do not fit beside the explicit region and a MoE body of 32 tokens (NCC_EGCA111). The
    # DP4 x TP32 MoE body needs the 8 KB for its third expert-loop buffer on the static path too.
    dp_moe = getattr(moe_body, "moe_sharding", "tp128") == "dp4xtp32"
    layer_vectors = T > QUERY_SET_TOKENS and (region is not None or dp_moe)
    kernel_assert(
        moe_pad_rows is None or getattr(moe_body, "takes_pad_rows", False),
        f"moe_pad_rows needs a MoE body that takes the padding-row flags (takes_pad_rows); "
        f"{getattr(moe_body, 'moe_sharding', 'tp128')} does not",
    )
    if not layer_vectors:
        norm_sb = _load_layer_vectors_hT(norm_weights, region)
        res_sb = _load_layer_vectors_hT(attn_res_weights, region)
    if sbuf_stream:
        s_hT = stream
        if region is not None:
            kernel_assert(any(s_hT is t for t in streams), "Give a stream tile of the region")
    elif region is not None:
        s_hT = streams[0]
        (load_rows_hT_batched if rows_hT_batched else load_rows_hT)(stream, s_hT)
    else:
        s_hT = nl.ndarray((P_MAX, T, n_tiles), dtype=stream.dtype, buffer=nl.sbuf)
        (load_rows_hT_batched if rows_hT_batched else load_rows_hT)(stream, s_hT)
    if wait_on is not None:
        # (bits(wait_on) & 0) + 1 is 1 for every value, so the multiply orders the block after wait_on bitwise.
        sb_wait = SbufTiles(region) if region is not None else AUTO_SBUF
        wait_bits = wait_on[0:P_MAX, 0, 0:2].view(nl.int32)
        gate_i = sb_wait.tile((P_MAX, 1), nl.int32)
        nisa.tensor_scalar(dst=gate_i, data=wait_bits, op0=nl.bitwise_and, operand0=0, engine=nisa.engine.vector)
        gate_f = sb_wait.tile((P_MAX, 1), nl.float32)
        nisa.tensor_scalar(dst=gate_f, data=gate_i, op0=nl.add, operand0=1.0, engine=nisa.engine.vector)
        nisa.tensor_scalar(dst=s_hT, data=s_hT, op0=nl.multiply, operand0=gate_f, engine=nisa.engine.vector)

    def next_stream():
        """Return the tile of the next residual: the other stream tile of the region, or a new tile."""
        if region is None:
            return nl.ndarray((P_MAX, T, n_tiles), dtype=stream.dtype, buffer=nl.sbuf)
        return streams[1] if s_hT is streams[0] else streams[0]

    # The bank lives in SBUF as [128, C, T, n]. Each snapshot layer of the block adds one entry.
    n_push = sum(1 for i in range(first_layer, first_layer + num_layers) if banks_snapshot(i))
    bank_hT, n_bank, bank_rstd = (None, 0, None) if bank is None else (tuple(bank) + (None,))[:3]
    if bank_hT is None and n_push > 0:
        bank_shape = (P_MAX, n_push, T, n_tiles)
        if region is None and (T <= 16 or not verify_bank):
            bank_hT = nl.ndarray(bank_shape, dtype=stream.dtype, buffer=nl.sbuf)
        elif region is None or T > _BANK_HBM_ROWS:
            # Above 16 rows the bank goes to HBM: in SBUF it evicts the collective destination arena.
            bank_hT = nl.ndarray(
                bank_shape, dtype=stream.dtype, buffer=nl.private_hbm, name=f"bank_{first_layer}{collective_name}"
            )
        else:
            bank_hT = region.heap(bank_shape, stream.dtype, "bank")
    bank_sums = attn_res_cached_bank_sums and not mix_cache and T > 1
    # attn_res_tensor_sums: the mixes of an SBUF bank take the TensorE form, with the bank rstd kept from the pushes.
    # attn_res_tensor_sums_idle: only the mixes where TensorE is idle (all but the MLP mix of a KDA layer).
    tensor_key = attn_res_tensor_sums or attn_res_tensor_sums_idle
    # attn_res_tensor_sums also takes the DRAM bank of the verify (verify_bank).
    tensor_sums = tensor_key and (
        bank_hT is None or bank_hT.buffer == nl.sbuf or (verify_bank and attn_res_tensor_sums)
    )
    kernel_assert(
        not tensor_key or tensor_sums,
        "attn_res_tensor_sums and attn_res_tensor_sums_idle need an SBUF bank (the decode bank); attn_res_tensor_sums "
        "also takes the DRAM bank of the verify (verify_bank)",
    )
    tensor_kda_mlp = tensor_sums and attn_res_tensor_sums
    # The GPU-parity norm: the plain mixes return their FP32 accumulator, and each norm after a mix rounds once.
    one_round = tensor_sums or attn_res_one_round_norm
    mix_out = {"out_dtype": nl.float32} if one_round else {}
    kernel_assert(
        not tensor_sums or n_bank == 0 or bank_rstd is not None,
        "attn_res_tensor_sums needs the bank rstd of the given bank: (bank_hT, count, bank_rstd)",
    )
    # attn_res_early_sums: the mixes with a kept bank rstd sum their stream squares first, and some sum their bank rows
    # early in compiler-placed tiles (early_bank_part).
    early_sums = (
        attn_res_early_sums
        and (mix_cache or bank_sums)
        and T > 1
        and (bank_hT is None or bank_hT.buffer == nl.sbuf or verify_bank)
    )

    # attn_res_ahead: the attention mix of a KDA layer takes the bank part that the tail of the layer before built.
    ahead = attn_res_ahead and T > COMPACT_MIX_TOKENS
    ahead_mlp = attn_res_ahead_mlp and T > COMPACT_MIX_TOKENS
    kernel_assert(attn_res_ahead_min_entries >= 0, "attn_res_ahead_min_entries must be 0 or more")
    kernel_assert(
        attn_res_ahead_load_tiles >= 2, f"attn_res_ahead_load_tiles must be 2 or more, got {attn_res_ahead_load_tiles}"
    )
    kernel_assert(
        attn_res_ahead_vector in ATTN_RES_AHEAD_VECTOR_MODES,
        f"attn_res_ahead_vector {attn_res_ahead_vector!r} must be one of {ATTN_RES_AHEAD_VECTOR_MODES}",
    )
    kernel_assert(
        not (attn_res_ahead or attn_res_ahead_mlp) or ((ahead or ahead_mlp) and bank_sums and not tensor_sums),
        f"attn_res_ahead and attn_res_ahead_mlp need more than {COMPACT_MIX_TOKENS} rows and the kept bank rstd "
        "(attn_res_cached_bank_sums), without attn_res_tensor_sums",
    )
    ahead_next = None
    ahead_mlp_part = None
    # The bank sums of a KDA layer's plain MLP mix: with an MXFP8 input projection, the KDA body traces them right after
    # the quantize of its input (x_quant_hook), so the scheduler cannot put them between that input's last write and the
    # quantize, where the projection's first matmul waits for them.
    quant_bank_sums = (
        getattr(kda_body, "x_quant_hook", False)
        and len(kda_weights) > len(KDA_WEIGHT_NAMES)
        and kda_weights[0].dtype != nl.bfloat16
    )
    # attn_res_kda_mlp_tail_entries: the last entries of those sums at the KDA body's tail_hook, on a DRAM bank only.
    kernel_assert(
        isinstance(attn_res_kda_mlp_tail_entries, int) and attn_res_kda_mlp_tail_entries >= 0,
        f"attn_res_kda_mlp_tail_entries must be an int >= 0, got {attn_res_kda_mlp_tail_entries!r}",
    )
    kda_mlp_tail = (
        attn_res_kda_mlp_tail_entries
        if early_sums and getattr(kda_body, "tail_hook", False) and bank_hT is not None and bank_hT.buffer != nl.sbuf
        else 0
    )

    # attn_res_mla_mlp_early: the MLP mix of an MLA layer takes its bank sums in the MLA body's state reduce-scatter.
    mla_mlp_early = attn_res_mla_mlp_early and early_sums
    kernel_assert(
        not mla_mlp_early or getattr(mla_body, "state_rs_hook", False),
        "attn_res_mla_mlp_early needs an MLA body that takes state_rs_hook (the k1 verify in the rows form or with the "
        "BF16-o state reduce-scatter)",
    )
    # attn_res_mla_mlp_tail_entries: the first entries of those sums at the MLA body's out_ar_hook, on a DRAM bank only.
    kernel_assert(
        isinstance(attn_res_mla_mlp_tail_entries, int) and attn_res_mla_mlp_tail_entries >= 0,
        f"attn_res_mla_mlp_tail_entries must be an int >= 0, got {attn_res_mla_mlp_tail_entries!r}",
    )
    mla_mlp_tail = (
        attn_res_mla_mlp_tail_entries
        if mla_mlp_early
        and getattr(mla_body, "out_ar_hook", False)
        and bank_hT is not None
        and bank_hT.buffer != nl.sbuf
        else 0
    )
    # attn_res_mla_mlp_plain_early: the plain MLP mix of an MLA layer takes its bank sums in the MLA body's state
    # reduce-scatter (its plain_part).
    mla_plain_early = attn_res_mla_mlp_plain_early
    kernel_assert(
        not mla_plain_early
        or (
            T > COMPACT_MIX_TOKENS
            and not (mix_cache or bank_sums or early_sums or tensor_sums)
            and getattr(mla_body, "state_rs_hook", False)
        ),
        "attn_res_mla_mlp_plain_early is for the plain MLP mix above COMPACT_MIX_TOKENS rows (no mix_cache, "
        "attn_res_cached_bank_sums, attn_res_early_sums or attn_res_tensor_sums) with an MLA body that takes "
        "state_rs_hook (the k1 call)",
    )
    # attn_res_kda_mlp_early_entries: the plain MLP mix of a KDA layer sums its first bank entries after the attention
    # norm (kda_mlp_early entries; 0: off).
    kda_mlp_early = (
        attn_res_kda_mlp_early_entries
        if T > COMPACT_MIX_TOKENS
        and (bank_hT is None or bank_hT.buffer == nl.sbuf or attn_res_kda_mlp_early_hbm)
        and not tensor_kda_mlp
        else 0
    )

    def early_bank_part(w_hT, w_index, tag, after=(), entries=None, out=None, in_place=False):
        """Return the bank sums of a mix with the score weight ``w_hT`` (row ``w_index`` of ``res_sb``), or None.
        The sums follow ``after`` in the schedule. ``entries`` / ``out`` / ``in_place``: the entry range of a split
        part, the part tile of all entries and the in-place SBUF entries (``attn_res_kda_mlp_tail_entries``)."""
        if not early_sums or n_bank == 0:
            return None
        # attn_res_weight_rows: the weight row of the mix, one multiply and one sum for each entry pass, the bits of
        # a fused multiply and sum for each row.
        w_row = WeightRow(res_sb, w_index) if mix_cache or attn_res_weight_rows else None
        # A DRAM bank (verify_bank) streams its entries on the queue of the mix loads.
        bank_dma = {"bank_dma": attn_res_bank_dma} if attn_res_bank_dma and bank_hT.buffer != nl.sbuf else {}
        split = {"out": out, "entries": entries, "in_place": in_place} if entries is not None else {}
        return attn_res_bank_part(
            bank_hT[0:P_MAX, 0:n_bank],
            w_hT,
            AUTO_SBUF,
            w_row,
            name=f"{tag}_bank_sums",
            after=after,
            **bank_dma,
            **split,
        )

    def build_ahead(li_next, kinds=("attn",), after=(), marks=None):
        """Return ``{kind: (part, w_split, g_row)}``, the ahead inputs of the mixes ``kinds`` ("attn": the attention
        mix, "mlp": the MLP mix) of layer ``li_next`` of the block (``attn_res_ahead``; part None for an empty bank).
        The parts of one call share their entry loads (``common.attn_res_ahead_parts``); the first TensorE sum follows
        ``after`` in the schedule. ``marks`` (a list) gets each part's last weighted-sum VectorE instruction."""
        if layer_vectors:
            w_vec = _load_layer_vectors_hT(attn_res_weights[li_next : li_next + 1])
            g_vec = _load_layer_vectors_hT(norm_weights[li_next : li_next + 1])
            base = 0
        else:
            w_vec, g_vec, base = res_sb, norm_sb, 2 * li_next
        rows = [base + int(kind == "mlp") for kind in kinds]
        splits = [attn_res_tensor_weights(w_vec[0:P_MAX, row, 0:n_tiles]) for row in rows]
        g_rows = [WeightRow(g_vec, row) for row in rows]
        parts = [None] * len(kinds)
        if n_bank:
            vector_share = attn_res_ahead_vector in ("all", "all_squares") or (
                attn_res_ahead_vector == "attn" and "attn" in kinds
            )
            squares = attn_res_ahead_vector == "all_squares"
            parts = attn_res_ahead_parts(
                bank_hT[0:P_MAX, 0:n_bank],
                bank_rstd[0:P_MAX, 0:n_bank, 0:T],
                [(w_split, g_row, new_ahead_part(T, n_tiles)) for w_split, g_row in zip(splits, g_rows, strict=True)],
                ones,
                bank_dma=attn_res_bank_dma,
                after=after,
                **({"vector_share": True} if vector_share else {}),
                **({"squares_vector": True} if squares else {}),
                **({"load_tiles": attn_res_ahead_load_tiles} if attn_res_ahead_load_tiles != 2 else {}),
                **({"marks": marks} if marks is not None else {}),
            )
        return {kind: inputs for kind, inputs in zip(kinds, zip(parts, splits, g_rows, strict=True), strict=True)}

    def build_ahead_steps(li_next, kind, after=(), tail=False):
        """``build_ahead`` of one mix in steps (``common.attn_res_ahead_parts_staged``): return ``(inputs, advance)``,
        the ahead inputs of the mix ``kind`` of layer ``li_next`` with the score pass traced, and the function that traces
        the next weighted sums (``advance(n)``, ``advance(None)`` the rest). ``tail`` (``attn_res_ahead_mlp_kda_tail``):
        the part holds all its entries and takes its norm squares on VectorE."""
        if layer_vectors:
            w_vec = _load_layer_vectors_hT(attn_res_weights[li_next : li_next + 1])
            g_vec = _load_layer_vectors_hT(norm_weights[li_next : li_next + 1])
            base = 0
        else:
            w_vec, g_vec, base = res_sb, norm_sb, 2 * li_next
        row = base + int(kind == "mlp")
        split = attn_res_tensor_weights(w_vec[0:P_MAX, row, 0:n_tiles])
        g_row = WeightRow(g_vec, row)
        if not n_bank:
            return (None, split, g_row), lambda n=None, after=(): True
        vector_share = attn_res_ahead_vector in ("all", "all_squares") or (
            attn_res_ahead_vector == "attn" and kind == "attn"
        )
        # tail: every entry keeps its own load tile (no reload after the chain), and no ScalarE square.
        load_tiles = max(attn_res_ahead_load_tiles, n_bank) if tail else attn_res_ahead_load_tiles
        parts, advance = attn_res_ahead_parts_staged(
            bank_hT[0:P_MAX, 0:n_bank],
            bank_rstd[0:P_MAX, 0:n_bank, 0:T],
            [(split, g_row, new_ahead_part(T, n_tiles))],
            ones,
            bank_dma=attn_res_bank_dma,
            after=after,
            **({"vector_share": True} if vector_share else {}),
            **({"squares_vector": True} if tail or attn_res_ahead_vector == "all_squares" else {}),
            **({"load_tiles": load_tiles} if load_tiles != 2 else {}),
        )
        return (parts[0], split, g_row), advance

    def mix_row(index):
        """Return the ``WeightRow`` of row ``index`` of ``res_sb`` with ``attn_res_weight_rows``, else None."""
        return WeightRow(res_sb, index) if attn_res_weight_rows else None

    def early_options(mix, part):
        """Return the ``attn_res_hT`` options ``mix`` with the early-sum options of ``attn_res_early_sums``."""
        if not early_sums or not mix:
            return mix
        return {**mix, "squares_first": True, "bank_part": part}

    if (mix_cache or bank_sums or tensor_sums) and bank_rstd is None and n_push > 0:
        bank_rstd = new_bank_rstd(bank_hT.shape[1], T, region)
    if n_push > 0:
        kernel_assert(bank_hT.shape[1] >= n_bank + n_push, "The AttnRes bank has no space for the block pushes")
    # The MoE options (``moe_options_in_use``) give a region also off the loop path; their sites add persistent tiles.
    # A DP body takes the routers of its router_dtypes itself: they are no option of the TP128 body.
    tp128_body = getattr(moe_body, "moe_sharding", "tp128") == "tp128"
    moe_options = moe_options_in_use(
        routed_kernel, router_mode, router_dtype if tp128_body else "bf16", routed_reduce_dtype
    )
    kernel_assert(region is not None or not moe_options, "The MoE options need an explicit SBUF region")
    kernel_assert(not (dp_moe and moe_options), "The DP4 x TP32 MoE body takes the default MoE options only")
    kernel_assert(
        tp128_body or router_dtype in getattr(moe_body, "router_dtypes", ("bf16",)),
        f"router_dtype {router_dtype!r}: the {getattr(moe_body, 'moe_sharding', 'tp128')} MoE body takes "
        f"{getattr(moe_body, 'router_dtypes', ('bf16',))}",
    )
    # fp16_local in the base MoE body: the local router logits ride the packed rows of the DP x gather.
    x_logits = router_dtype == "fp16_local" and not moe_options
    kernel_assert(
        not dp_moe or moe_early_loads == "all",
        "moe_early_loads is for the TP128 projections; the DP4 x TP32 body keeps the default load order",
    )
    kernel_assert(
        not (dp_moe and moe_dynamic_loop),
        "moe_dynamic_loop is for the TP128 projections; the DP4 x TP32 body runs the static expert loop",
    )
    # The collective tiles are arena views off the MLA loop path (``collective_sources``).
    arena_sources = region is None or (moe_options and not mla_loop_path)
    # The opt-in dynamic expert loop: B <= 2, off the MLA loop path, in the base MoE body (no option, no region).
    base_body = base_moe_body(routed, moe_options, dp_moe)
    kernel_assert(not base_glue or base_body, "moe_glue_dmas with the EP128 static kernel needs the base MoE body")
    dynamic_experts = dynamic_expert_loop(
        moe_dynamic_loop, dp_size * T, mla_loop_path, not base_body or region is not None
    )
    # The one dispatch point of the expert paths (``tkg_moe.routed_expert_path``): the dynamic loop, else the
    # active-expert slots (base body, also on the loop path), else the static loop.
    expert_path = routed_expert_path(dynamic_experts, moe_active_expert_slots, base_body)
    kernel_assert(
        not moe_gate_up_half_major or (routed_kernel == "static" and expert_path == "static" and not dp_moe),
        "moe_gate_up_half_major needs the EP128 static expert loop of the TP128 body",
    )
    kernel_assert(
        not moe_shared_scale_buffer
        or (routed_kernel == "static" and expert_path == "static" and not dp_moe and moe_scale_buffers in (None, 1)),
        "moe_shared_scale_buffer needs the EP128 static expert loop of the TP128 body with 1 scale buffer",
    )
    kernel_assert(
        not (moe_shared_scale_buffer and moe_cross_expert_order),
        "moe_shared_scale_buffer keeps the expert order: not with moe_cross_expert_order",
    )
    kernel_assert(
        not (moe_ring_reclaim or moe_half_weight_buffer_count is not None)
        or (routed_kernel == "static" and expert_path == "static" and base_body and moe_half_weight_buffers),
        "moe_ring_reclaim and moe_half_weight_buffer_count need the static expert loop of the base MoE body with "
        "moe_half_weight_buffers",
    )
    kernel_assert(
        moe_half_weight_buffer_count is None or 2 <= moe_half_weight_buffer_count <= (5 if moe_ring_reclaim else 6),
        f"moe_half_weight_buffer_count must be None or 2 to {5 if moe_ring_reclaim else 6}, got "
        f"{moe_half_weight_buffer_count}",
    )
    # The 2-server expert-loop options: its I_L = 1536 buffers are half the size of the one-server ones.
    two_server_loop = (
        getattr(moe_body, "moe_sharding", "tp128") == "dp_2srv" and routed_kernel == "static" and not dynamic_experts
    )
    for key, value, allowed in (
        ("moe_weight_buffers", moe_weight_buffers, (1, 2, 3, 4, 5)),
        ("moe_scale_buffers", moe_scale_buffers, (1, 2)),
        ("moe_transpose_lookahead", moe_transpose_lookahead, (1, 2)),
    ):
        kernel_assert(value is None or value in allowed, f"{key} must be None or one of {allowed}, got {value}")
        kernel_assert(value is None or two_server_loop, f"{key} needs the static expert loop of the 2-server MoE body")
    kernel_assert(
        moe_weight_buffers is None or not moe_half_weight_buffers,
        "moe_weight_buffers sets whole buffers: it does not take moe_half_weight_buffers",
    )
    kernel_assert(
        not moe_cross_expert_order
        or (two_server_loop and moe_weight_buffers in (3, 4, 5) and moe_transpose_lookahead is None),
        "moe_cross_expert_order needs the static expert loop of the 2-server MoE body with moe_weight_buffers 3 to 5, "
        "and no moe_transpose_lookahead",
    )
    kernel_assert(not moe_transpose_block or moe_cross_expert_order, "moe_transpose_block needs moe_cross_expert_order")
    kernel_assert(
        not (moe_cross_expert_order and moe_prefetch)
        or all(w in (0, moe_weight_buffers) for w in moe_prefetch.values()),
        f"With moe_cross_expert_order a prefetched expert ring loads in the cross-expert order with its "
        f"{moe_weight_buffers} whole buffers: moe_prefetch takes 0 or {moe_weight_buffers} per layer type, got "
        f"{moe_prefetch}",
    )
    kernel_assert(
        moe_routed_store_form == "relayout" or (base_body and expert_path == "static"),
        "moe_routed_store_form='pe' needs the static expert loop of the base MoE body",
    )
    kernel_assert(
        not moe_direct_latent or (base_body and expert_path == "static"),
        "moe_direct_latent needs the static expert loop of the base MoE body",
    )
    kernel_assert(
        moe_latent_shard_form in LATENT_SHARD_FORMS and (moe_latent_shard_form == "one_hot" or base_body),
        f"moe_latent_shard_form must be one of {LATENT_SHARD_FORMS}; {moe_latent_shard_form!r} needs the base MoE body",
    )
    kernel_assert(
        moe_preamble_buffers in range(4) and (not moe_preamble_buffers or (base_body and expert_path == "static")),
        f"moe_preamble_buffers takes 0 to 3 extra buffers of the static expert loop of the base MoE body, got "
        f"{moe_preamble_buffers!r}",
    )
    kernel_assert(not moe_preamble_reuse or moe_preamble_buffers, "moe_preamble_reuse needs moe_preamble_buffers")
    kernel_assert(
        not moe_router_rows_drop_selection or (base_body and expert_path == "static"),
        "moe_router_rows_drop_selection needs the static expert loop of the base MoE body (the EP128 static kernel and "
        "no MoE option)",
    )
    kernel_assert(
        not moe_latent_gather_mx
        or (moe_router_rows_drop_selection and base_body and expert_path == "static" and moe_latent_load == "pe32"),
        "moe_latent_gather_mx needs moe_router_rows_drop_selection, moe_latent_load 'pe32' and the static expert loop "
        "of the base MoE body",
    )
    kernel_assert(
        moe_static_alt_tiles in STATIC_ALT_TILE_FORMS,
        f"moe_static_alt_tiles must be one of {STATIC_ALT_TILE_FORMS}, got {moe_static_alt_tiles!r}",
    )
    kernel_assert(
        not moe_static_alt_tiles
        or (
            expert_path == "static"
            and (
                (base_body and getattr(moe_body, "moe_sharding", "tp128") == "tp128")
                or getattr(moe_body, "moe_sharding", None) == "dp_2srv"
            )
        ),
        "moe_static_alt_tiles needs the static expert loop of the base MoE body (TP128 projections) or of the "
        "2-server MoE body",
    )
    if active_expert_counts(expert_path, moe_active_expert_counts is not None):
        kernel_assert(
            tuple(moe_active_expert_counts.shape) == (n_moe, 1) and moe_active_expert_counts.dtype == nl.int32,
            f"Expected INT32 [{n_moe}, 1] moe_active_expert_counts, got {moe_active_expert_counts.shape}",
        )
    kernel_assert(
        mla_stage_cc_split in MLA_STAGE_SPLITS,
        f"mla_stage_cc_split must be one of {tuple(MLA_STAGE_SPLITS)}, got {mla_stage_cc_split!r}",
    )
    split_sites = MLA_STAGE_SPLITS[mla_stage_cc_split]
    split_kw = (
        {"mla_stage_split": split_sites, "mla_stage_unit_tokens": mla_stage_unit_tokens(mla_pack_pairs)}
        if split_sites
        else {}
    )
    if sources is None:
        sites = moe_option_sites(routed_kernel, router_mode, router_dtype, routed_reduce_dtype, mla_loop_path)
        # ffn_rows sizes the MoE gather by the feed-forward chunk; 0 (decode) sizes it by T.
        sources = collective_sources(
            T,
            n_tiles,
            stream.dtype,
            region,
            moe_options=sites,
            mla=mla_shared_sources and not mla_cp,
            name=collective_name,
            x_pair=x_parity,
            x_hbm=mla_loop_path and sites is None and dp_size != 1 and not x_parity and not dp_moe,
            x_world=moe_x_world_gather,
            token_world=attn_rs_moe_input,
            x_logits=x_logits,
            mla_stage=mla_staged(T, mla_loop_path),
            sbuf_routed_reduce="routed" in sbuf_reduces,
            sbuf_routed_options="routed_tp128" in sbuf_reduces,
            dp_moe=dp_moe,
            a2av_moe=getattr(moe_body, "a2av_expert_form", None),
            pingpong=cc_source_pingpong,
            mla_cp=mla_cp,
            reduce_rows=reduce_rows,
            dp_size=dp_size,
            moe_tokens=min(T, ffn_rows) if ffn_rows else 0,
            out_sets=-(-T // min(T, ffn_rows)) if ffn_rows else 1,
            moe_out_proj_form=moe_out_proj_form,
            **split_kw,
            **({"ring_reclaim": True} if moe_ring_reclaim else {}),
        )
    # The dict that the model passes to every block: the step-wide stores (mla_options pack_step_items) go in it, not
    # in the copy of this block call below (moe_x_world_gather).
    model_sources = sources
    # More than 8 sequences run the MLA layers as groups of 8 (``collective_sources``). The verify brings its own
    # MLA tiles and has no MLA groups, so its Q * B rows do not select the group form.
    kernel_assert(
        moe_ring_reclaim == ("moe_ring_buffer" in sources),
        "moe_ring_reclaim needs the sources of collective_sources(ring_reclaim=True), and those need the key",
    )
    mla_groups = sources.get("mla_groups")
    mla_grouped = mla_groups is not None and len(mla_groups) > 1
    stage_set = sources.get("mla_stage") or {}
    kernel_assert(
        stage_set.get("split_sites", ()) == split_sites
        and (not split_sites or (mla_grouped and mla_packed_loop and mla_loop_path and not mla_cp))
        and (not split_sites or stage_set["unit_tokens"] == mla_stage_unit_tokens(mla_pack_pairs)),
        f"mla_stage_cc_split {mla_stage_cc_split!r} needs the staged MLA groups on the packed loop path, with sources "
        f"of the same split and unit (collective_sources mla_stage_split); the sources split "
        f"{stage_set.get('split_sites', ())}",
    )

    # The feed-forward bodies gather the DP replicas (dp_size 4) or run on the own rows (dp_size 1).
    ff_dp = {} if dp_size == DP_SIZE else {"dp_size": dp_size}
    kernel_assert(
        not moe_x_world_gather
        or (
            "x_world_src_hbm" in sources
            and "x_src_gather_hbm" in sources
            and dp_size == DP_SIZE
            and base_body
            and not dp_moe
            and moe_in_proj_form == "gathered"
        ),
        "moe_x_world_gather needs the sources of collective_sources x_world (the HBM x gather of the MLA loop path), "
        f"the DP{DP_SIZE} layout and the base MoE body",
    )
    kernel_assert(
        not moe_x_gather_hbm
        or (
            "x_src_gather_hbm" in sources
            and dp_size == DP_SIZE
            and base_body
            and not dp_moe
            and moe_in_proj_form == "gathered"
            and not attn_rs_moe_input
        ),
        "moe_x_gather_hbm needs the HBM x gather pair (collective_sources x_hbm), "
        f"the DP{DP_SIZE} layout and the base MoE body, and not attn_rs_moe_input (it brings its own token gather)",
    )
    if moe_x_world_gather:
        # The TP rank is the runtime offset of each layer's world x gather slice: built once, at heap addresses on the
        # loop path (as the rank selectors).
        tp_sb = SbufTiles(region, heap=True) if region is not None else AUTO_SBUF
        sources = {**sources, "x_world_rank": tp_rank_tile(rank_info, "k3_rank", tp_sb)}
        if moe_x_path_act:
            sources["x_path_act"] = True

    if region is not None and (n_moe or n_dense):
        # The feed-forward bodies build the rank selectors once and keep them for the whole kernel,
        # across the MLA loops. Build them here at heap addresses.
        rank_selectors(rank_info, "k3_rank", SbufTiles(region, heap=True), dp_size=dp_size)
        if moe_out_proj_form == "column" and n_moe:
            rank_row_one_hot(rank_info, "k3_rank", SbufTiles(region, heap=True))
        if routed_kernel == "sel_load" and n_moe and sel_load_down_groups(moe_weights[3], moe_weights[6]):
            down_group_mask(rank_info, "k3_down_groups", SbufTiles(region, heap=True))
    if moe_2srv_selectors_once and n_moe:
        kernel_assert(
            getattr(moe_body, "build_selectors", None) is not None,
            "moe_2srv_selectors_once needs the 2-server MoE body (moe_decode_2srv.moe_2srv_body)",
        )
        moe_body.build_selectors(rank_info, T, sb=SbufTiles(region, heap=True) if region is not None else AUTO_SBUF)

    def poisoned(tile, li, stage, sb=AUTO_SBUF):
        """Return ``tile`` plus the poison rows at ``(li, stage)`` (tests), else ``tile``."""
        if poison is None or (poison[0], poison[1]) != (li, stage):
            return tile
        rows_hT = sb.tile(tile.shape, tile.dtype)
        load_rows_hT(poison[2], rows_hT)
        out = sb.tile(tile.shape, tile.dtype)
        nisa.tensor_tensor(dst=out, data1=tile, data2=rows_hT, op=nl.add)
        return out

    # The layers whose DP x gather takes the alternate destination. A layer whose MLA body runs the key
    # loops gathers into the persistent x_dst: the loops end a basic block, and the unique_tensors check
    # of neuronx-cc does not see the loop tiles that can hold the bytes of a new tile. With the persistent
    # x_dst2 as the alternate destination, the loops do not matter.
    x_alternates = frozenset()
    # With the HBM x gather pair (``collective_sources`` ``x_hbm``), a layer whose MLA body runs the key loops gathers
    # through HBM. The routed options name their own HBM x gather "x_dst_hbm" ("x" site), so the source key decides.
    x_hbm_layers = frozenset()
    if "x_src_gather_hbm" in sources and mla_loop_path and mla_runs_loops(mla_block_table.shape[2]):
        x_hbm_layers = frozenset(i for i in range(first_layer, first_layer + num_layers) if is_mla_layer(i))
    if moe_x_world_gather or moe_x_gather_hbm:
        # The world x gather, and the DP x gather of moe_x_gather_hbm, run through HBM in every layer.
        x_hbm_layers = frozenset(range(first_layer, first_layer + num_layers))
    if x_parity and not dp_moe:
        mla_loops = mla_loop_path and mla_runs_loops(mla_block_table.shape[2]) and "x_dst2" not in sources
        kernel_start = first_layer if kernel_first_layer is None else kernel_first_layer
        kernel_layers = range(kernel_start, first_layer + num_layers)
        loop_layers = frozenset(i for i in kernel_layers if mla_loops and is_mla_layer(i))
        x_alternates = x_gather_alternates(kernel_layers, loop_layers)

    # attn_res_kda_mlp_early_in_front: persistent conv-state scatter tiles for the KDA layers (the region heap on the
    # loop path; the static path already keeps the scatter sources in its arena). The index tables alternate by KDA layer
    # parity: the compiler hoists the index ops of a layer, so they do not wait for the scatter of the layer before.
    kda_front = kda_mlp_early and attn_res_kda_mlp_early_in_front
    kda_front_tiles = ({}, {})
    front_rows = {}
    if kda_front:
        kernel_assert("kda_ar_hbm" in sources, "attn_res_kda_mlp_early_in_front needs the KDA all-reduce sources")
        front_plan = SbufTiles(region, heap=True) if region is not None else AUTO_SBUF
        if T <= _KDA_FRONT_ROWS_MAX_TOKENS and not any(key.startswith("kda_conv_rows_") for key in sources):
            front_rows = kda_conv_rows_tiles(T, front_plan)
        kda_front_tiles = tuple({**kda_conv_index_tiles(T, front_plan), **front_rows} for _ in range(2))
    # The conv-state stores of a KDA layer read SBUF tiles after the layer. The compiler ends the live range of a DMA
    # source at the DMA trigger, and a later writer of the bytes may not wait for the DMA. A per-layer source is read
    # once more after the feed-forward norm, after its stores (kda_read_scatter_keep), so a later tile that takes its
    # bytes waits for them. A persistent source (kda_conv_rows_tiles / kda_conv_index_tiles) is rewritten only by a
    # later KDA layer, whose first write into it waits for the conv-state stores of the two KDA layers before: the list
    # sources["kda_conv_waits"] (from the kernel, so across the blocks) holds them, else a list of this block.
    kda_keeps = kda_conv_keep and getattr(kda_body, "conv_keep", False)
    kda_tiles = kda_conv_keep and getattr(kda_body, "conv_tiles", False)
    static_rows = "kda_ar_hbm" in sources and any(key.startswith("kda_conv_rows_") for key in sources)
    keep_sinks = {}
    persistent_rows = kda_tiles and (static_rows or front_rows)
    if kda_conv_keep and n_kda and (kda_front and T > _KDA_FRONT_ROWS_MAX_TOKENS or kda_keeps and not persistent_rows):
        keep_plan = SbufTiles(region, heap=True) if region is not None else AUTO_SBUF
        for dtype, kind in ((nl.bfloat16, "rows"), (nl.int32, "index")):
            sink = keep_plan.tile((P_MAX, 1), dtype, f"kda_keep_{kind}")
            nisa.memset(dst=sink, value=0)
            hbm = nl.ndarray((P_MAX, 1), dtype=dtype, buffer=nl.private_hbm, name=f"kda_keep_{kind}_L{first_layer}")
            keep_sinks[str(dtype)] = (sink, hbm)
    elif not kda_conv_keep and kda_front and T > _KDA_FRONT_ROWS_MAX_TOKENS:
        # Without kda_conv_keep: the scatter-view sinks of the front path above 16 sequences only.
        for dtype, kind in ((nl.bfloat16, "rows"), (nl.int32, "index")):
            sink = front_plan.tile((P_MAX, 1), dtype, f"kda_keep_{kind}")
            nisa.memset(dst=sink, value=0)
            hbm = nl.ndarray((P_MAX, 1), dtype=dtype, buffer=nl.private_hbm, name=f"kda_keep_{kind}_L{first_layer}")
            keep_sinks[str(dtype)] = (sink, hbm)
    conv_waits = sources.get("kda_conv_waits")
    conv_waits = [] if conv_waits is None else conv_waits
    # The prefetch loads of the block (moe_hot_prefetch): one gate, a slot for each prefetch tile.
    prefetch_gate = PrefetchGate()
    # A MoE body with the attribute hot_experts takes the hot-expert tiles (moe_hot_prefetch).
    hot_body = bool(getattr(moe_body, "hot_experts", False))
    if hot_counts and not hot_body:
        # One word of each hot-expert tile adds into this heap tile after each MoE layer; one DMA stores it at the end.
        hot_sink = region.heap((1, 3), nl.int32, "hot_keep")
        nisa.memset(dst=hot_sink, value=0)
        hot_hbm = nl.ndarray((1, 3), dtype=nl.int32, buffer=nl.private_hbm, name=f"hot_keep_L{first_layer}")
        keep_sinks["hot_experts"] = (hot_sink, hot_hbm)

    kda_k = 0
    mla_k = 0
    moe_k = 0
    in_proj_tile = None  # moe_in_proj_at_kda_reduce: the in-projection tile of this layer's MoE body
    kda_prework = getattr(kda_body, "prework", None)
    prework = None
    # kda_front_ahead: the front of the next KDA layer, traced in the MoE layer before.
    front_ahead = None
    ffn_released = tuple(gate)
    ffn_late = ffn_released
    # The tail loop runs in the MLA groups of 8, which build their own items, and with ``mla_pack_tail_low_batch`` in
    # the one group of 8 sequences and fewer.
    tail_tokens = T >= _PACK_TAIL_MIN_TOKENS or (mla_pack_tail_low_batch and T <= MLA_GROUP_TOKENS)
    pack_tail = mla_packed_loop and bool(mla_pack_tail_slots) and tail_tokens
    # At 8 sequences and fewer one program serves every mix of contexts, so the item count of a rank can leave any
    # remainder: the low-batch tail rounds the items up to whole tail iterations and splits a remainder over tail
    # loops of halving width, one iteration each (8 to 15 items run one iteration of 16, not two of 8).
    pack_tails, tail_round = (mla_pack_tail_slots if pack_tail else None), False
    if pack_tail and T <= MLA_GROUP_TOKENS:
        pack_tails = pack_tail_chain(mla_block_table.shape[2], mla_pack_slots, mla_pack_tail_slots, T)
        tail_round = True
    kernel_assert(
        not mla_pack_page_runs or (mla_packed_loop and not mla_cp),
        "mla_pack_page_runs needs mla_packed_loop and the CP32 MLA body",
    )
    kernel_assert(
        not mla_batched_cache_write or not (mla_cp or mla_pair_pages),
        "mla_batched_cache_write needs the CP32 MLA body with row-major pages",
    )
    # The packed MLA loop builds its items once for the MLA layers of the block, on the heap of the region.
    pack_plan = None
    pack_once = mla_packed_loop and mla_pack_once and mla_loop_path and region is not None and n_mla
    # mla_options pack_step_items: one plan for each packed-loop unit and its item rows for the whole step, in a store of
    # ``sources`` that every block of the model shares (built by the first block with an MLA layer).
    mla_options = dict(mla_options or {})
    step_items = mla_options.pop("pack_step_items", False)
    kernel_assert(step_items in (False, True), f"mla_options pack_step_items must be a bool, got {step_items!r}")
    step_store = None
    if step_items and n_mla:
        kernel_assert(
            pack_once
            and not mla_pack_setup_ahead
            and not mla_cp
            and sources is not None
            and (not mla_grouped or ("mla_stage" in sources and (mla_pack_pair_plans or mla_pack_group_plans))),
            "mla_options pack_step_items needs the packed MLA loop built once for the block (mla_pack_once, a region) with "
            "one plan for each loop (8 sequences and fewer, or the staged groups with mla_pack_pair_plans or "
            "mla_pack_group_plans), no mla_pack_setup_ahead and no mla_cp",
        )
        step_store = model_sources.setdefault("mla_step_pack", {})
    step_plans = None if step_store is None else step_store.get("plans")
    if pack_once and not mla_grouped and step_plans is not None:
        pack_plan = step_plans
    elif pack_once and not mla_grouped:
        # One plan for each block call: the tag names the first layer, as the model calls the body for each group.
        with region.frame(f"L{first_layer}_pack"):
            pack_plan = mla_pack_plan(
                mla_block_table[0],
                position,
                rank_info,
                f"L{first_layer}_pack",
                SbufTiles(region),
                mla_pack_slots,
                pack_tails,
                tail_round,
            )
    # mla_pack_group_plans: one plan for each staged MLA group of 8, built once for the MLA layers of the block. Only
    # the item tiles take the heap (96 bytes a partition for each group); the scratch tiles take compiler placement,
    # and each layer builds the TI offsets, which depend on B only, once for its groups.
    group_plans = None
    if mla_pack_pairs:
        kernel_assert(
            T >= 4 * MLA_GROUP_TOKENS
            and T % (2 * MLA_GROUP_TOKENS) == 0
            and mla_grouped
            and len(mla_groups) % 2 == 0
            and "mla_stage" in sources
            and mla_packed_loop
            and mla_loop_path
            and region is not None
            and not mla_cp
            and not mla_pack_group_plans,
            f"mla_pack_pairs needs the packed MLA loop of 32 or more staged sequences in an even count of groups of "
            f"{MLA_GROUP_TOKENS}, with a region, no mla_cp and no mla_pack_group_plans; got {T} sequences",
        )
    kernel_assert(
        not mla_stage_post or (mla_grouped and "mla_stage" in sources and "state" not in split_sites),
        f"mla_stage_post needs the staged MLA groups (more than {MLA_GROUP_TOKENS} sequences with sources mla_stage) "
        f"and one state reduce-scatter for the stage (mla_stage_cc_split {mla_stage_cc_split!r})",
    )
    if pack_once and mla_grouped and mla_pack_group_plans and "mla_stage" in sources and step_plans is not None:
        group_plans = step_plans
    elif pack_once and mla_grouped and mla_pack_group_plans and "mla_stage" in sources:
        group_plans = [
            mla_pack_plan(
                mla_block_table[0][g * MLA_GROUP_TOKENS : (g + 1) * MLA_GROUP_TOKENS],
                position[g * MLA_GROUP_TOKENS : (g + 1) * MLA_GROUP_TOKENS],
                rank_info,
                f"L{first_layer}_g{g}_pack",
                None,
                mla_pack_slots,
                pack_tails,
                tail_round,
                plan_sbuf=SbufTiles(region),
                with_ti=False,
            )
            for g in range(len(mla_groups))
        ]
    # mla_pack_pair_plans: one plan for each pair of staged MLA groups (mla_pack_pairs), built once for the MLA layers of
    # the block with the arguments of the plan that the second group of the pair builds in each layer without it.
    pair_plans = None
    if mla_pack_pair_plans:
        kernel_assert(
            mla_pack_pairs and pack_once,
            "mla_pack_pair_plans needs mla_pack_pairs and mla_pack_once (the packed MLA loop pairs on the loop path)",
        )
        pair_tokens = 2 * MLA_GROUP_TOKENS
        pair_plans = step_plans or [
            mla_pack_plan(
                mla_block_table[0][p0 : p0 + pair_tokens],
                position[p0 : p0 + pair_tokens],
                rank_info,
                f"L{first_layer}_p{p0 // pair_tokens}_pack",
                None,
                mla_pack_slots,
                pack_tails,
                tail_round,
                plan_sbuf=SbufTiles(region),
                with_ti=False,
            )
            for p0 in range(0, T, pair_tokens)
        ]
    layer_plan = pack_plan is None and group_plans is None and pair_plans is None
    if step_store is not None and step_plans is None:
        # The first block with an MLA layer: its plans and the unit stores serve every MLA layer of the step. The
        # run-scale tiles take the heap here, outside the layer frames, so they live through the step.
        step_store["plans"] = pack_plan if not mla_grouped else (pair_plans if pair_plans is not None else group_plans)
        if not mla_grouped:
            units = [("u", T)]
        elif pair_plans is not None:
            units = [(f"p{p}", 2 * MLA_GROUP_TOKENS) for p in range(len(pair_plans))]
        else:
            units = [(f"g{g}", min(MLA_GROUP_TOKENS, T - g * MLA_GROUP_TOKENS)) for g in range(len(mla_groups))]
        step_store["units"] = {name: mla_pack_step_unit(region, name, n, mla_pack_page_runs) for name, n in units}
    kernel_assert(
        mla_pack_state_store_queues == 1 or (mla_grouped and "mla_stage" in sources and mla_packed_loop),
        "mla_pack_state_store_queues needs the staged MLA groups of the packed loop",
    )
    kernel_assert(not (mla_current_after_rs and mla_cp), "mla_current_after_rs needs the CP32 MLA body")
    kernel_assert(
        not mla_stage_pre or (mla_grouped and "mla_stage" in sources and not split_sites and not mla_cp),
        'mla_stage_pre needs the staged MLA groups (sources mla_stage) with mla_stage_cc_split "off"',
    )
    kernel_assert(
        not mla_pack_setup_ahead
        or (pack_once and mla_grouped and "mla_stage" in sources and (pair_plans if mla_pack_pairs else group_plans)),
        "mla_pack_setup_ahead needs the staged MLA groups of the packed loop with a plan for each loop (mla_pack_pairs "
        "with mla_pack_pair_plans, or mla_pack_group_plans)",
    )
    kernel_assert(
        not moe_sums_after_routed or getattr(moe_body, "moe_sharding", None) == "dp_2srv",
        "moe_sums_after_routed orders the row sums after the routed_trigger hook of the multi-server MoE body "
        f"(moe_sharding 'dp_2srv'), got {getattr(moe_body, 'moe_sharding', None)!r}",
    )
    # moe_sums_after_routed: the routed reduction collectives of the previous MoE layer (routed_trigger hook).
    routed_after = []
    with (
        weight_gates(dma_gates, consumer_gates=ffn_tail_order or kda_small_after_out_ar),
        row_sums_order() as order_row_sums,
        ffn_output_order(first_block=kernel_first_layer in (None, first_layer))
        if row_sums_after_ffn_output
        else nullcontext(),
    ):
        for li in range(num_layers):
            # The row sums of the layer (the bank sums of its mixes) follow the last collective of the previous
            # feed-forward body, not only its data; with row_sums_after_ffn_output, the consumer of its routed output.
            if ffn_tail_order:
                order_row_sums((*ffn_late, *routed_after))
            else:
                # One edge for each layer: its first VectorE row sum follows the consumer of the routed output. Every
                # row sum follows the routed reduction of the previous MoE layer (moe_sums_after_routed).
                order_row_sums(routed_after, once_instrs=take_ffn_output() if row_sums_after_ffn_output else ())
            layer = first_layer + li
            layer_src = layer_sources(sources, layer)
            tag = f"L{layer}"
            k = 2 * li
            # kda_gap_loads in_proj: the marks of this layer's KDA input projection (None: the gate without
            # kda_gap_loads).
            gap_marks = None
            if layer_vectors:
                if region is not None:
                    region.open_frame(f"{tag}_vectors")
                norm_sb = _load_layer_vectors_hT(norm_weights[li : li + 1], region)
                res_sb = _load_layer_vectors_hT(attn_res_weights[li : li + 1], region)
                k = 0
            att_kind = "mla" if is_mla_layer(layer) else "kda"
            if region is not None:
                region.open_span(f"{att_kind}_attention")
            hot = None
            ahead_mlp_mix = False
            hot_n = 0 if is_dense_layer(layer) else hot_counts.get(att_kind, 0)
            # The prefetch tiles of the layer take heap addresses of a frame that ends after the feed-forward body.
            prefetch_frame = bool(hot_n)
            if prefetch_frame:
                region.open_frame(f"{tag}_prefetch", release_heap=True)
            if hot_n:
                hot, hot_loads = hot_expert_tiles(region, moe_weights, moe_expert_base + moe_k, hot_n, tag)
            # attn_res_mla_mlp_early: the bank sums of the MLA layer's MLP mix. On the loop path they take a tile of the
            # layer frame (the frame of the prefetch tiles, else of the layer vectors), so the tile outlives the MLA
            # frame; a static MLA body has no loop, so the compiler places the tile.
            mla_mlp_part = None
            loop_frame = region is not None and mla_loop_path
            if (
                mla_mlp_early
                and is_mla_layer(layer)
                and not mla_grouped
                and n_bank
                and not banks_snapshot(layer)
                and (prefetch_frame or layer_vectors or not loop_frame)
            ):
                part_shape = (P_MAX, (n_bank + 1) * T)
                if loop_frame:
                    mla_mlp_part = region.stack(part_shape, nl.float32, f"{tag}_mlp_bank_sums")
                else:
                    mla_mlp_part = nl.ndarray(part_shape, dtype=nl.float32, buffer=nl.sbuf)
            # attn_res_mla_mlp_plain_early: the plain part of the MLA layer's MLP mix, in the same frames.
            mla_plain_part = None
            mla_plain_box = []
            if (
                mla_plain_early
                and is_mla_layer(layer)
                and not mla_grouped
                and n_bank
                and not banks_snapshot(layer)
                and (prefetch_frame or layer_vectors or not loop_frame)
            ):
                plain_shape = (P_MAX, 2, (n_bank + 1) * T)
                if loop_frame:
                    mla_plain_part = region.stack(plain_shape, nl.float32, f"{tag}_mlp_plain_sums")
                else:
                    mla_plain_part = nl.ndarray(plain_shape, dtype=nl.float32, buffer=nl.sbuf)
            if prefetch_frame:
                prefetch_marks = (len(region.sbm.heap), region.sbm.stack_curr_addr)
            in_w = norm_sb[0:P_MAX, k, 0:n_tiles]
            post_w = norm_sb[0:P_MAX, k + 1, 0:n_tiles]
            attn_w = res_sb[0:P_MAX, k, 0:n_tiles]
            mlp_w = res_sb[0:P_MAX, k + 1, 0:n_tiles]

            # An MLA layer on the loop path gives every SBUF tile an explicit address in a frame, from the
            # attention mix to the residual: these are the tiles that live through the loops of the MLA body.
            # Without the loop path the MLA body has no loop, so the compiler places its tiles also when the
            # block has a region for its routed kernel (at B = 8 they do not fit the explicit stack).
            # Above 8 sequences the frame does not fit the region (NCC_INKI016), so the compiler places the MLA tiles.
            mla_frame = region is not None and mla_loop_path and is_mla_layer(layer) and not mla_grouped
            # The bank rows of the attention mix, in compiler-placed tiles before the MLA frame (attn_res_early_sums).
            ahead_mix = ahead and not is_mla_layer(layer) and n_bank >= attn_res_ahead_min_entries
            attn_part = None if ahead_mix else early_bank_part(attn_w, k, f"{tag}{collective_name}_attn")
            sb = SbufTiles(region) if mla_frame else AUTO_SBUF
            with region.frame(f"{tag}_mla", release_heap=True) if mla_frame else nullcontext():
                s_hT = poisoned(s_hT, li, "attn_res_in", sb)
                bank_view = None if n_bank == 0 else bank_hT[0:P_MAX, 0:n_bank]
                # Above 4 sequences the mix tiles return to the frame before the MLA body, which needs their
                # bytes with a full bank; B <= 4 keeps the SBUF layout of 4 sequences.
                mix_frame = mla_frame and T > QUERY_SET_TOKENS
                h = sb.tile((P_MAX, T, n_tiles), stream.dtype) if mix_frame else None
                with sb.scope("mix") if mix_frame else nullcontext():
                    # read_out takes the final mix compute, so the HBM bank push can wait for it.
                    mix = _mix_options(bank_rstd, n_bank, WeightRow(res_sb, k)) if mix_cache else {}
                    if bank_sums:
                        mix = _bank_sum_options(bank_rstd, n_bank)
                    # Any HBM bank (verify_bank, or a bank in HBM that the caller gave) takes the push wait.
                    mix_edge = [] if verify_bank or (bank_view is not None and bank_view.buffer != nl.sbuf) else None
                    mix = early_options(mix, attn_part)
                    # attn_res_weight_rows: the weighted sums of the mix take the weight row (one multiply and one sum
                    # for each candidate).
                    if attn_res_weight_rows:
                        mix = _with_weight_row(mix, WeightRow(res_sb, k), n_bank)
                    mix = _act_squares(mix, attn_res_cached_act_squares, s_hT)
                    # attn_res_scalar_squares: the ScalarE squares of the bank entries follow the last collective of the
                    # previous feed-forward body, so they cannot move into its expert loop, where ScalarE is busy.
                    act_mix = (
                        {
                            "act_square_rows": attn_res_scalar_squares,
                            "act_bank_square_rows": attn_res_scalar_squares,
                            "act_square_after": ffn_late,
                        }
                        if attn_res_scalar_squares and n_bank
                        else {}
                    )
                    # attn_res_scalar_squares_order: the mix's ones-sum matmuls and its last ScalarE square, for the
                    # conv-state transposes of a KDA body and for the bank push.
                    mix_sums, mix_act = [], []
                    if attn_res_scalar_squares_order and act_mix:
                        act_mix = {**act_mix, "sums_done": mix_sums, "act_done": mix_act}
                    # attn_res_scalar_products: ScalarE takes part of the entry-mix products (2 or more entries).
                    if attn_res_scalar_products and n_bank > 1:
                        act_mix = {**act_mix, "act_product_rows": attn_res_scalar_products}
                        if attn_res_scalar_products_stt:
                            act_mix["act_product_stt"] = True
                    bank_dma = {"bank_dma": attn_res_bank_dma} if attn_res_bank_dma else {}
                    # The stream rstd of this layer's bank push when its mix writes it (rstd_out), else None.
                    push_rstd = None
                    if ahead_mix:
                        # The bank part comes from the tail of the layer before (the first layer of a call builds it
                        # here); the mix and its norm run below, in one pass over the stream.
                        if ahead_next is None:
                            ahead_next = build_ahead(li)["attn"]
                        mixed = None
                    elif tensor_sums and n_bank:
                        # The bank-row matmuls follow the last collective of the previous feed-forward body.
                        push_rstd = bank_rstd[0:P_MAX, n_bank, 0:T] if banks_snapshot(layer) else None
                        mixed = attn_res_tensor_hT(
                            bank_view,
                            s_hT,
                            attn_res_tensor_weights(attn_w, sb),
                            eps,
                            ones,
                            sb,
                            bank_rstd=bank_rstd[0:P_MAX, 0:n_bank, 0:T],
                            bank_after=tuple(ffn_late),
                            rstd_out=push_rstd,
                            **_tensor_mix_options(act_mix),
                            **_tensor_bank_options(bank_view, mix_edge, attn_res_bank_dma),
                        )
                    else:
                        # A pushing layer's mix of the kept bank rstd writes the stream rstd of the push (the values of
                        # row_rstd_hT), so the push sums no squares of its own.
                        if banks_snapshot(layer) and "bank_rstd" in mix and T > 1:
                            push_rstd = bank_rstd[0:P_MAX, n_bank, 0:T]
                        mixed = attn_res_hT(
                            bank_view,
                            s_hT,
                            attn_w,
                            eps,
                            ones,
                            sb,
                            read_out=mix_edge,
                            **mix,
                            **act_mix,
                            **bank_dma,
                            **mix_out,
                            **({} if push_rstd is None else {"rstd_out": push_rstd}),
                        )
                    if before_attention is not None and mixed is not None:
                        before_attention(layer, mixed)
                    if banks_snapshot(layer):
                        push_bank_entry(bank_hT, n_bank, s_hT, mix_act, mix_edge)
                        if (mix_cache or bank_sums or (tensor_sums and n_bank == 0)) and push_rstd is None:
                            row_rstd_hT(s_hT, eps, ones, bank_rstd[0:P_MAX, n_bank, 0:T], sb)
                        n_bank += 1
                    norm = {"w_row": WeightRow(norm_sb, k)} if mix_cache else {}
                    if norm_one_mul and not mix_cache and not one_round:
                        norm = {"w_mul_row": WeightRow(norm_sb, k)}
                    norm_done = [] if early_sums or (kda_mlp_early and not is_mla_layer(layer)) else None
                    act_norm = {"act_square_rows": norm_scalar_squares} if norm_scalar_squares else {}
                    if one_round:
                        # The FP32 mix (or the bf16 stream of an empty bank) and its norm round once.
                        act_norm = {**act_norm, "one_round": True, "out_dtype": stream.dtype}
                    if ahead_mix:
                        part, w_split, g_row = ahead_next
                        ahead_next = None
                        # The measured form (a 1-core device microbench): the TensorE Gram stream sums, no GpSimd.
                        h = attn_res_ahead_hT(
                            s_hT,
                            part,
                            w_split,
                            g_row,
                            eps,
                            ones,
                            sb,
                            out=h,
                            done=norm_done,
                            pool=False,
                            stats_form="gram",
                            **({"gram_merge": True} if attn_res_gram_merge else {}),
                        )
                    else:
                        h = rmsnorm_hT(mixed, in_w, eps, ones, sb, out=h, done=norm_done, **norm, **act_norm)
                    # A KDA body starts with its in-projection, so VectorE is idle there: the bank rows of the MLP mix
                    # go here, after the norm (else the scheduler puts them into the VectorE chain of the mix). An MLA
                    # body loops, and a compiler-placed tile must not live across its loops.
                    mlp_part = mla_mlp_part
                    mlp_part_args = None
                    mlp_tail_args = None
                    # n_bank counts the push of this layer here: the bank of the MLP mix.
                    ahead_mlp_mix = ahead_mlp and not is_mla_layer(layer) and n_bank >= attn_res_ahead_min_entries
                    ahead_advance = None
                    ahead_tail = False
                    if ahead_mlp_mix:
                        # attn_res_ahead_mlp: the bank part of this layer's MLP mix, unless the tail of the layer
                        # before built it with the attention part (a layer that pushes a bank entry builds it here,
                        # after the push).
                        if ahead_mlp_part is None and AHEAD_MLP_TOKEN_STEPS and getattr(kda_body, "token_hook", False):
                            # AHEAD_MLP_TOKEN_STEPS: the score pass here, the weighted sums in the KDA token chain.
                            # attn_res_ahead_mlp_kda_tail: the weighted sums after the chain (the body's tail_hook).
                            ahead_tail = attn_res_ahead_mlp_kda_tail and getattr(kda_body, "tail_hook", False)
                            ahead_mlp_part, ahead_advance = build_ahead_steps(
                                li, "mlp", after=tuple(norm_done or ()), tail=ahead_tail
                            )
                        elif ahead_mlp_part is None:
                            ahead_mlp_part = build_ahead(li, ("mlp",), after=tuple(norm_done or ()))["mlp"]
                    elif not is_mla_layer(layer) and quant_bank_sums and early_sums and n_bank:
                        mlp_part_args = (mlp_w, k + 1, f"{tag}{collective_name}_mlp", tuple(norm_done or ()))
                    elif not is_mla_layer(layer) and kda_mlp_tail and early_sums and n_bank:
                        # attn_res_kda_mlp_tail_entries: the later entries here, after the norm; the first ones at the
                        # KDA body's tail_hook (below), into the same part tile.
                        count = min(kda_mlp_tail, n_bank)
                        part = AUTO_SBUF.tile((P_MAX, (n_bank + 1) * T), nl.float32)
                        mlp_tail_args = (mlp_w, k + 1, f"{tag}{collective_name}_mlp", count, part)
                        if count < n_bank:
                            early_bank_part(
                                mlp_w,
                                k + 1,
                                f"{tag}{collective_name}_mlp",
                                after=norm_done or (),
                                entries=(count, n_bank),
                                out=part,
                            )
                    elif not is_mla_layer(layer):
                        mlp_part = early_bank_part(mlp_w, k + 1, f"{tag}{collective_name}_mlp", after=norm_done or ())
                    plain_mlp_part = None
                    kda_front_hook = None
                    scatter_keep = None
                    if kda_mlp_early and n_bank and not is_mla_layer(layer) and kda_front:
                        # The KDA body calls the hook after its conv-state index ops and gathers (in_front).
                        early_box = []

                        def kda_front_hook(
                            front,
                            box=early_box,
                            bank=bank_hT[0:P_MAX, 0:n_bank],
                            w=mlp_w,
                            w_row=mix_row(k + 1),
                            after=norm_done,
                            name=f"{tag}{collective_name}",
                        ):
                            box.append(
                                attn_res_plain_bank_part(
                                    bank,
                                    w,
                                    kda_mlp_early,
                                    name=f"{name}_mlp_early",
                                    after=[*after, *front],
                                    bank_dma=attn_res_bank_dma,
                                    w_row=w_row,
                                )
                            )

                    elif kda_mlp_early and n_bank and not is_mla_layer(layer):
                        plain_mlp_part = attn_res_plain_bank_part(
                            bank_hT[0:P_MAX, 0:n_bank],
                            mlp_w,
                            kda_mlp_early,
                            name=f"{tag}{collective_name}_mlp_early",
                            after=norm_done,
                            bank_dma=attn_res_bank_dma,
                            w_row=mix_row(k + 1),
                        )
                if layer_debug is not None:
                    store_hT_rows(layer_debug[li, 0], h, sb)

                # The attention weights wait for the feed-forward collectives of the previous layer.
                set_weight_gate(ffn_released, late=ffn_late)
                if hot is not None:
                    # The hot-expert loads wait for the last use of their bytes in the layer before, not for a gate.
                    for key in HOT_TILE_KEYS:
                        prefetch_gate.issue(hot_loads[key], slot=f"hot_{key}")
                prefetch = None
                prefetch_box = None
                front_prefetch = None
                kind = "mla" if is_mla_layer(layer) else "kda"
                depth = moe_prefetch[kind] if moe_prefetch else 0
                if depth and not is_dense_layer(layer):
                    # The expert ring of this layer's MoE loads during the attention, behind the same gate.
                    uses = (moe_prefetch_uses or {}).get(kind)
                    scales = (moe_prefetch_scales or {}).get(kind, True)

                    def start_prefetch(
                        w_in_marks=None, depth=depth, uses=uses, scales=scales, moe_k=moe_k, gate=True, layer=layer
                    ):
                        record = None
                        if w_in_marks is not None and moe_prefetch_kda_before_slot_stores:
                            record = w_in_marks.setdefault("before_slot_stores", [])
                        after = (
                            _dmas_after_w_in(w_in_marks, moe_prefetch_kda_queue_order, record)
                            if w_in_marks is not None
                            else nullcontext()
                        )
                        with gate_all_dmas() if gate else nullcontext(), after:
                            level = None if moe_prefetch_priority is None else dma_priority(moe_prefetch_priority)
                            return moe_body.prefetch(
                                *(_layer_view(t, moe_k, n_moe) for t in moe_weights),
                                weight_buffers=depth,
                                **({"use_order": True} if moe_cross_expert_order else {}),
                                **({"priority": level} if level is not None else {}),
                                **({"uses": uses} if uses is not None else {}),
                                **({} if scales else {"scales": False}),
                                # A routed plan's body starts the ring of the layer's tag (None for a sub-512-tile split).
                                **({"layer": layer} if getattr(moe_body.prefetch, "takes_layer", False) else {}),
                            )

                    if moe_prefetch_kda_after_in_proj and kind == "kda":
                        # The KDA body starts the ring after its input projection (in_proj_hook below).
                        prefetch_box = []
                    elif moe_prefetch_start and kind == "mla":
                        # The ring starts at the MoE front, in the feed-forward weight gate (before the MoE body).
                        front_prefetch = start_prefetch
                    else:
                        prefetch = start_prefetch()
                if is_mla_layer(layer):
                    weights = tuple(_layer_view(t, mla_k, n_mla) for t in mla_weights)
                    mla_mx = {}
                    if len(weights) > len(MLA_WEIGHT_NAMES):
                        # The scale stacks follow the six weights: mla_in_proj's, mla_q_b_proj's, mla_kv_b_proj's,
                        # then mla_o_proj's (MXFP8 classes only, in MLA_WEIGHT_NAMES order). A class is MX when its
                        # weight is not BF16.
                        weights, scales = weights[: len(MLA_WEIGHT_NAMES)], list(weights[len(MLA_WEIGHT_NAMES) :])
                        for name, index in MLA_MX_SCALES:
                            if weights[index].dtype != nl.bfloat16:
                                mla_mx[name] = scales.pop(0)
                        kernel_assert(not scales, "each MLA scale stack needs the MXFP8 weight of its class")
                    mla_kw = {
                        **({"fp8_pages": True} if mla_fp8_pages else {}),
                        **({"zeroed_pages": True} if mla_zeroed_pages else {}),
                        **({"loop_cast_engines": mla_loop_cast_engines} if mla_loop_cast_engines != "off" else {}),
                        **({"loop_pad_stationary": True} if mla_loop_pad_stationary else {}),
                        **({"loop_first_static": True} if mla_cp and mla_loop_first_static else {}),
                        **({"scores_ahead": True} if mla_cp and mla_scores_ahead else {}),
                        **({"folded_reference": True} if mla_cp and mla_folded_reference else {}),
                        **(dict(mla_cp_rows_options or {}) if mla_cp else {}),
                        **({"pair_pages": True} if mla_pair_pages else {}),
                        **({"lora_norm_eps": lora_norm_eps} if lora_norm_eps != LORA_NORM_EPS else {}),
                        **({"packed_loop": True, "pack_slots": mla_pack_slots} if mla_packed_loop else {}),
                        **({"pack_page_runs": mla_pack_page_runs} if mla_pack_page_runs else {}),
                        **({"pack_null_copy_vector": True} if mla_packed_loop and mla_pack_null_copy_vector else {}),
                        **({"pack_null_runs": True} if mla_pack_page_runs and mla_pack_null_runs else {}),
                        **({"batched_cache_write": True} if mla_batched_cache_write else {}),
                        # A plan carries its tail loop.
                        **({"pack_tail_slots": pack_tails} if pack_tail and layer_plan else {}),
                        **({"pack_tail_round": True} if tail_round and layer_plan else {}),
                        **({"reduce_rows": reduce_rows} if mla_cp and reduce_rows else {}),
                        **({"pack_current_after_rs": True} if mla_current_after_rs else {}),
                        **(mla_options or {}),
                        **mla_mx,
                    }
                    head_pending = mla_head_prefetch is not None and mla_head_prefetch.pending is not None
                    kernel_assert(
                        not head_pending or (not mla_grouped and not mla_mx),
                        "mla_head_prefetch takes the BF16 head weights of an ungrouped MLA layer",
                    )
                    if mla_mlp_part is not None:
                        # attn_res_mla_mlp_tail_entries: the first tail_n entries (the SBUF ones of a split bank, read in
                        # place) at the body's out_ar_hook; the others at its state_rs_hook. One part tile.
                        tail_n = min(mla_mlp_tail, n_bank)
                        # The body may name the queue of the hook's bank-entry loads (state_rs_hook_dma).
                        mla_dma = getattr(mla_body, "state_rs_hook_dma", None) or attn_res_bank_dma
                        if tail_n < n_bank:
                            mla_kw["state_rs_hook"] = _bank_sums_hook(
                                bank_hT[0:P_MAX, 0:n_bank],
                                mlp_w,
                                mla_mlp_part,
                                mla_dma,
                                f"{tag}{collective_name}_mlp",
                                mix_row(k + 1),
                                **({"entries": (tail_n, n_bank)} if tail_n else {}),
                            )
                        if tail_n:
                            mla_kw["out_ar_hook"] = _bank_sums_hook(
                                bank_hT[0:P_MAX, 0:n_bank],
                                mlp_w,
                                mla_mlp_part,
                                mla_dma,
                                f"{tag}{collective_name}_mlp_tail",
                                mix_row(k + 1),
                                entries=(0, tail_n),
                                in_place=True,
                                with_after=True,
                            )
                    if mla_plain_part is not None:
                        mla_kw["state_rs_hook"] = _plain_sums_hook(
                            bank_hT[0:P_MAX, 0:n_bank],
                            mlp_w,
                            mla_plain_part,
                            f"{tag}{collective_name}_mlp_plain",
                            mix_row(k + 1),
                            mla_plain_box,
                        )
                    if not mla_grouped:
                        y = mla_body(
                            h,
                            *weights,
                            kv_pools[li],
                            mla_block_table[mla_k],
                            mla_slot[mla_k],
                            position,
                            rank_info,
                            tp_group,
                            tag,
                            loop_path=mla_loop_path,
                            sbuf=sb if mla_frame else None,
                            # mla_rs_hold: the reduce-scatter tensors of the parity set of this MLA layer.
                            sources={**sources, **mla_rs_hold_sources(sources, layer)},
                            **({"head_prefetch": mla_head_prefetch.take(layer)} if head_pending else {}),
                            **mla_kw,
                            **({"pack_plan": pack_plan} if mla_packed_loop else {}),
                            **({"pack_step_items": step_store["units"]["u"]} if step_store is not None else {}),
                        )
                        if mla_plain_part is not None:
                            kernel_assert(
                                len(mla_plain_box) == 1,
                                f"the MLA body of layer {layer} did not call state_rs_hook once "
                                "(attn_res_mla_mlp_plain_early needs the one-pass k1 call)",
                            )
                            plain_mlp_part = (mla_plain_part, n_bank)
                    elif "mla_stage" in sources:
                        # Above 8 sequences the groups of 8 run in lockstep with one collective per site.
                        kernel_assert(
                            hasattr(mla_body, "stages")
                            and not mla_kw.get("bf16_state")
                            and not mla_kw.get("state_bf16_o"),
                            "the staged MLA groups need a staged body with an FP32 HBM state (not state_bf16_o)",
                        )
                        shared, bodies = {}, []
                        # The packed loop tiles that need explicit addresses take a frame of the region, which the
                        # groups share; the compiler places the other tiles.
                        pack_frame = mla_packed_loop and region is not None
                        pack_kw = {"pack_sbuf": SbufTiles(region)} if pack_frame else {}
                        for g, group in enumerate(mla_groups):
                            g0 = g * MLA_GROUP_TOKENS
                            pack_hbm = sources["mla_pack_hbm"]
                            if mla_pack_setup_ahead:
                                # The HBM slot rows and items of this loop (a pair or a group), for every layer.
                                pack_hbm = pack_hbm.setdefault(f"unit{g // 2 if mla_pack_pairs else g}", {})
                            pair_kw = {}
                            if mla_pack_pairs:
                                # Groups 2k and 2k + 1 run one packed loop over their 16 sequences.
                                p0 = g0 - (g % 2) * MLA_GROUP_TOKENS
                                p1 = p0 + 2 * MLA_GROUP_TOKENS
                                pair_kw["pack_pair"] = {
                                    "half": g % 2,
                                    "tokens": 2 * MLA_GROUP_TOKENS,
                                    "block_table": mla_block_table[mla_k][p0:p1],
                                    "position": position[p0:p1],
                                    "rank_info": rank_info,
                                    "state_hbm": mla_stage_views(sources["mla_stage"], p0, 2 * MLA_GROUP_TOKENS)[
                                        "state_hbm"
                                    ],
                                }
                            bodies.append(
                                mla_body.stages(
                                    h[0:P_MAX, g0 : g0 + MLA_GROUP_TOKENS, :],
                                    *weights,
                                    kv_pools[li],
                                    mla_block_table[mla_k][g0 : g0 + MLA_GROUP_TOKENS],
                                    mla_slot[mla_k][g0 : g0 + MLA_GROUP_TOKENS],
                                    position[g0 : g0 + MLA_GROUP_TOKENS],
                                    rank_info,
                                    tp_group,
                                    f"{tag}_g{g}",
                                    loop_path=mla_loop_path,
                                    sbuf=None,
                                    sources={
                                        **group,
                                        "stage_shared": shared,
                                        "mla_pack_hbm": pack_hbm,
                                        **({"post_stage": sources["mla_stage"]} if mla_stage_post else {}),
                                        **(
                                            {"pre_stage": {"stage": sources["mla_stage"], "x_hT": h}}
                                            if mla_stage_pre
                                            else {}
                                        ),
                                    },
                                    **mla_kw,
                                    **({"stage_post": True} if mla_stage_post else {}),
                                    **({"stage_pre": True} if mla_stage_pre else {}),
                                    **pack_kw,
                                    **({"pack_plan": group_plans[g]} if group_plans else {}),
                                    **({"pack_plan": pair_plans[g // 2]} if pair_plans and g % 2 else {}),
                                    **(
                                        {
                                            "pack_step_items": step_store["units"][
                                                f"g{g}" if group_plans else f"p{g // 2}"
                                            ]
                                        }
                                        if step_store is not None and (group_plans or (pair_plans and g % 2))
                                        else {}
                                    ),
                                    **({"pack_setup_ahead": True} if mla_pack_setup_ahead else {}),
                                    **(
                                        {"pack_state_store_queues": mla_pack_state_store_queues}
                                        if mla_pack_state_store_queues != 1
                                        else {}
                                    ),
                                    **pair_kw,
                                )
                            )
                        with region.frame(f"{tag}_mla_pack", release_heap=True) if pack_frame else nullcontext():
                            _run_staged(bodies, sources["mla_stage"], tag)
                        y = sb.tile((P_MAX, T, n_tiles), stream.dtype)
                        nisa.dma_copy(
                            dst=y,
                            src=sources["mla_stage"]["out_ag_dst_hbm"].reshape((P_MAX, T, n_tiles)),
                            name=f"{tag}_mla_out_load",
                        )
                    else:
                        # More than 8 sequences run as groups of 8, each with its own collective tensors.
                        # Above 16 an SBUF accumulator lives across the MLA loops, so the outputs stay in HBM to the end.
                        hbm_out = len(mla_groups) > 2
                        y = None if hbm_out else sb.tile((P_MAX, T, n_tiles), stream.dtype)
                        outs = []
                        for g, group in enumerate(mla_groups):
                            g0 = g * MLA_GROUP_TOKENS
                            gc = min(MLA_GROUP_TOKENS, T - g0)
                            yg = mla_body(
                                h[0:P_MAX, g0 : g0 + gc, :],
                                *weights,
                                kv_pools[li],
                                mla_block_table[mla_k][g0 : g0 + gc],
                                mla_slot[mla_k][g0 : g0 + gc],
                                position[g0 : g0 + gc],
                                rank_info,
                                tp_group,
                                f"{tag}_g{g}",
                                loop_path=mla_loop_path,
                                sbuf=None,
                                sources={
                                    **group,
                                    **mla_rs_hold_sources(sources, layer, g),
                                    **({"out_in_hbm": True} if hbm_out else {}),
                                },
                                **mla_kw,
                            )
                            if hbm_out:
                                outs.append((g0, gc, yg))
                            else:
                                nisa.tensor_copy(dst=y[0:P_MAX, g0 : g0 + gc, :], src=yg)
                        if hbm_out:
                            y = sb.tile((P_MAX, T, n_tiles), stream.dtype)
                            for g0, gc, out in outs:
                                nisa.dma_copy(
                                    dst=y[0:P_MAX, g0 : g0 + gc, :],
                                    src=out.reshape((P_MAX, gc, n_tiles)),
                                    name=f"{tag}_mla_out_load_{g0}",
                                )
                    mla_k += 1
                else:
                    weights = tuple(_layer_view(t, kda_k, n_kda) for t in kda_weights)
                    kda_sources = {"sources": layer_src} if "kda_ar_hbm" in sources else {}
                    if len(weights) > len(KDA_WEIGHT_NAMES):
                        # The scale stacks follow the four weights: kda_w_in's, then kda_w_out's (MXFP8 classes only).
                        weights, scales = weights[: len(KDA_WEIGHT_NAMES)], list(weights[len(KDA_WEIGHT_NAMES) :])
                        for name, weight in (("kda_w_in_scale", weights[0]), ("kda_w_out_scale", weights[1])):
                            if weight.dtype != nl.bfloat16:
                                kda_sources[name] = scales.pop(0)
                        kernel_assert(not scales, "each KDA scale stack needs an MXFP8 kda_w_in or kda_w_out")
                    if prework is not None:
                        kda_sources["prework"] = prework
                    if front_ahead is not None:
                        kda_sources["front_ahead"] = front_ahead
                        front_ahead = None
                    head_tiles = kda_ahead_carry.pop(layer, None)
                    if head_tiles is not None:
                        _check_head_room(region, kda_w_in_heads[layer], f"read in layer {layer}")
                        kda_sources["w_in_head"] = head_tiles
                    if kda_front_hook is not None:
                        kda_sources.update(
                            front_hook=kda_front_hook, sources={**layer_src, **kda_front_tiles[kda_k % 2]}
                        )
                        if T > _KDA_FRONT_ROWS_MAX_TOKENS:
                            scatter_keep = []
                            kda_sources["scatter_keep"] = scatter_keep
                    chained = False
                    if kda_keeps:
                        kda_src = (kda_sources.get("sources") or {}) if kda_tiles else {}
                        if scatter_keep is None and not any(key.startswith("kda_conv_rows_") for key in kda_src):
                            scatter_keep = []
                            kda_sources["scatter_keep"] = scatter_keep
                        if any(key.startswith(("kda_conv_rows_", "kda_conv_index_")) for key in kda_src):
                            chained = True
                            kda_sources["conv_copy_wait"] = tuple(dma for stores in conv_waits for dma in stores)
                    if mix_sums:
                        kda_sources["conv_after"] = tuple(mix_sums)
                    if kda_w_in_prefetch is not None and kda_w_in_prefetch.pending is not None:
                        kda_sources["w_in_ring"] = kda_w_in_prefetch.take(layer)
                    in_proj_box = []
                    if moe_in_proj_at_kda_reduce and not is_dense_layer(layer):
                        # This layer's MoE in-projection: right after the attention all-reduce, before the state
                        # stores in the program, so its tile does not take the bytes of the staged states.

                        def in_proj_at_reduce(
                            reduce, box=in_proj_box, w=_layer_view(moe_weights[0], moe_k, n_moe), name=tag
                        ):
                            box.append(moe_body.load_in_proj(w, name, after=reduce))

                        kda_sources["reduce_hook"] = in_proj_at_reduce
                    if gap_w_out:
                        kda_sources["w_out_after_w_in"] = True
                    if kda_small_after_out_ar:
                        # The GpSimd weight loads wait for the result load of the previous body's last collective.
                        kda_sources["small_after_result"] = True
                    if layer in kda_states_after:
                        # The state loads follow the previous MoE layer's output all-reduce trigger.
                        kda_sources["state_loads_after"] = kda_states_after.pop(layer)
                    if gap_in_proj and not is_dense_layer(layer):
                        gap_marks = {}
                        kda_sources["w_in_marks"] = gap_marks
                    if prefetch_box is not None:

                        def in_proj_hook(marks, box=prefetch_box, start=start_prefetch):
                            box.append(start(marks))

                        kda_sources["in_proj_hook"] = in_proj_hook
                    if ahead_advance is not None and ahead_tail:
                        # attn_res_ahead_mlp_kda_tail: the weighted sums and the bank norm after the output projection
                        # of the last rows, the first sum after its last VectorE copy.

                        def tail_hook(marks, advance=ahead_advance):
                            last = marks.get("vector")
                            advance(None, after=() if last is None else (last,))

                        kda_sources["tail_hook"] = tail_hook
                    elif ahead_advance is not None:
                        # AHEAD_MLP_TOKEN_STEPS: the entries of the weighted sums spread over tokens 0 .. Q - 2 of the
                        # chain; the last scored token's hook (or the call below) traces the rest and the bank norm.

                        def token_hook(j, q, advance=ahead_advance, entries=n_bank):
                            if j < q - 1:
                                per = -(-entries // max(1, q - 1))
                                advance(per)
                            if j >= q - 2:
                                advance(None)

                        kda_sources["token_hook"] = token_hook
                    quant_box = []
                    tail_box = []
                    # attn_res_kda_mlp_tail_entries: the first tail_n entries (the SBUF ones of a split bank, read in
                    # place) at the body's tail_hook, after the output projection; the others at the quantize (or after
                    # the norm, traced above).
                    tail_n = min(kda_mlp_tail, n_bank) if mlp_part_args is not None else 0
                    tail_part = AUTO_SBUF.tile((P_MAX, (n_bank + 1) * T), nl.float32) if tail_n else None
                    tail_args = mlp_part_args
                    if mlp_tail_args is not None:
                        w_t, i_t, n_t, tail_n, tail_part = mlp_tail_args
                        tail_args = (w_t, i_t, n_t, ())
                    kernel_assert(
                        not tail_n or "tail_hook" not in kda_sources,
                        f"attn_res_kda_mlp_tail_entries: the KDA body of layer {layer} already takes a tail_hook",
                    )
                    if mlp_part_args is not None and tail_n < n_bank:

                        def x_quant_hook(quantize, box=quant_box, args=mlp_part_args, first=tail_n, part=tail_part):
                            w_hT, index, name, after = args
                            box.append(
                                early_bank_part(
                                    w_hT,
                                    index,
                                    name,
                                    after=(*after, quantize),
                                    **({"entries": (first, n_bank), "out": part} if first else {}),
                                )
                            )

                        kda_sources["x_quant_hook"] = x_quant_hook
                    if tail_n:

                        def tail_hook(marks, box=tail_box, args=tail_args, count=tail_n, part=tail_part):
                            w_hT, index, name, _ = args
                            last = marks.get("vector")
                            box.append(
                                early_bank_part(
                                    w_hT,
                                    index,
                                    f"{name}_tail",
                                    after=() if last is None else (last,),
                                    entries=(0, count),
                                    out=part,
                                    in_place=True,
                                )
                            )

                        kda_sources["tail_hook"] = tail_hook
                    y = kda_body(h, *weights, kv_pools[li], kda_page_ids[kda_k], position, tp_group, tag, **kda_sources)
                    if ahead_advance is not None:
                        # A body with fewer than two scored tokens did not trace every step: trace the rest here.
                        ahead_advance(None)
                    if mlp_part_args is not None and tail_n < n_bank:
                        kernel_assert(len(quant_box) == 1, f"the KDA body of layer {layer} did not call x_quant_hook")
                        mlp_part = quant_box[0]
                    if tail_n:
                        kernel_assert(len(tail_box) == 1, f"the KDA body of layer {layer} did not call tail_hook")
                        mlp_part = tail_part
                    if prefetch_box is not None:
                        kernel_assert(
                            len(prefetch_box) == 1, f"the KDA body of layer {layer} did not call in_proj_hook"
                        )
                        prefetch = prefetch_box[0]
                    if "reduce_hook" in kda_sources:
                        kernel_assert(len(in_proj_box) == 1, f"the KDA body of layer {layer} did not call reduce_hook")
                        in_proj_tile = in_proj_box[0]
                    if chained:
                        conv_waits[:] = [*conv_waits[-1:], tuple(getattr(y, "conv_stores", ()))]
                    if kda_front_hook is not None:
                        kernel_assert(len(early_box) == 1, "the KDA body calls front_hook once")
                        plain_mlp_part = early_box[0]
                    prework = None
                    kda_k += 1
                attn_released = take_released(early=early_ffn_gate)
                y = poisoned(y, li, "attn_out", sb)

                if layer_debug is not None:
                    store_hT_rows(layer_debug[li, 1], y, sb)
                attn_read = None
                if not attn_rs_moe_input:
                    resid = next_stream()
                    if banks_snapshot(layer):
                        attn_read = nisa.tensor_copy(dst=resid, src=y)
                    else:
                        attn_read = nisa.tensor_tensor(dst=resid, data1=s_hT, data2=y, op=nl.add)
                    s_hT = resid

            if region is not None:
                region.close_span()
                region.open_span("dense_ffn" if is_dense_layer(layer) else f"{att_kind}_ffn")
            bank_view = bank_hT[0:P_MAX, 0:n_bank] if n_bank else None
            h16 = None
            if attn_rs_moe_input:
                # attn_rs_moe_input: the attention body returned this rank's one reduced row. Run the residual + MLP mix
                # + norm on that one row (one_row_moe_glue, padded to the compact path), all-gather the updated stream
                # over the TP group for the post-MoE residual and the next attention, and feed the one normed row to
                # the MoE body, which gathers it over the world group.
                # The glue reads the bank with a strided DMA over its whole storage: pass the bank tensor and its
                # filled entries, not the slice.
                glue = one_row_moe_glue(
                    s_hT,
                    y,
                    bank_hT if n_bank else None,
                    tp_rank_tile(rank_info, "k3_rank", sb),
                    mlp_w,
                    post_w,
                    eps,
                    ones,
                    sb,
                    residual=not banks_snapshot(layer),
                    name=f"{tag}_rs_glue",
                    n_bank=n_bank,
                    **({"fp16": True} if router_dtype != "bf16" and not is_dense_layer(layer) else {}),
                )
                resid_t, moe_in = glue[:2]
                if len(glue) > 2:
                    h16 = glue[2]
                s_hT = tp_stream_gather(
                    resid_t,
                    tp_group,
                    sources["s_prime_src_hbm"],
                    sources["s_prime_dst_hbm"],
                    T,
                    sb,
                    name=f"{tag}_sprime",
                )
                h = moe_in
                # The one-row glue runs the norm: the conv-state keep reads (kda_read_scatter_keep) follow only
                # their stores.
                ffn_norm_done = ()
            elif ahead_mlp_mix:
                # attn_res_ahead_mlp: the MLP mix and its norm in one pass, from the bank part of the attention side.
                if router_dtype != "bf16" and not is_dense_layer(layer):
                    h16 = nl.ndarray((P_MAX, T, n_tiles), dtype=nl.float16, buffer=nl.sbuf)
                ffn_norm_done = [] if scatter_keep else None
                part, w_split, g_row = ahead_mlp_part
                ahead_mlp_part = None
                h = poisoned(
                    attn_res_ahead_hT(
                        s_hT,
                        part,
                        w_split,
                        g_row,
                        eps,
                        ones,
                        out=layer_src["ffn_x"],
                        done=ffn_norm_done,
                        pool=False,
                        out16=h16,
                        stats_form="gram",
                        **({"gram_merge": True} if attn_res_gram_merge else {}),
                    ),
                    li,
                    "ffn_in",
                )
            else:
                mix = _mix_options(bank_rstd, n_bank, WeightRow(res_sb, k + 1)) if mix_cache else {}
                if bank_sums:
                    mix = _bank_sum_options(bank_rstd, n_bank)
                plain = {"plain_part": plain_mlp_part} if plain_mlp_part is not None else {}
                act_mix = {"act_square_rows": attn_res_scalar_squares} if attn_res_scalar_squares and n_bank else {}
                if attn_res_scalar_products and n_bank > 1:
                    act_mix = {**act_mix, "act_product_rows": attn_res_scalar_products}
                    if attn_res_scalar_products_stt:
                        act_mix["act_product_stt"] = True
                bank_dma = {"bank_dma": attn_res_bank_dma} if attn_res_bank_dma else {}
                if tensor_sums and n_bank and (tensor_kda_mlp or is_mla_layer(layer)):
                    # The bank-row matmuls follow the attention output collective, so TensorE runs them in its flight.
                    # attn_res_tensor_sums_idle: not in a KDA layer, where TensorE runs the KDA state update then.
                    mixed = attn_res_tensor_hT(
                        bank_view,
                        s_hT,
                        attn_res_tensor_weights(mlp_w),
                        eps,
                        ones,
                        bank_rstd=bank_rstd[0:P_MAX, 0:n_bank, 0:T],
                        bank_after=tuple(attn_released),
                        **_tensor_mix_options(act_mix),
                        **_tensor_bank_options(bank_view, None, attn_res_bank_dma),
                    )
                else:
                    mixed = attn_res_hT(
                        bank_view,
                        s_hT,
                        mlp_w,
                        eps,
                        ones,
                        **_act_squares(
                            (
                                _with_weight_row(early_options(mix, mlp_part), WeightRow(res_sb, k + 1), n_bank)
                                if attn_res_weight_rows
                                else early_options(mix, mlp_part)
                            ),
                            attn_res_cached_act_squares,
                            s_hT,
                        ),
                        **plain,
                        **act_mix,
                        **bank_dma,
                        **mix_out,
                    )
                # The FP16 router of a MoE layer reads an FP16 copy of the norm, rounded from its FP32 values.
                if router_dtype != "bf16" and not is_dense_layer(layer):
                    h16 = nl.ndarray((P_MAX, T, n_tiles), dtype=nl.float16, buffer=nl.sbuf)
                norm = {"w_row": WeightRow(norm_sb, k + 1)} if mix_cache else {}
                if norm_one_mul and not mix_cache and not one_round:
                    norm = {"w_mul_row": WeightRow(norm_sb, k + 1)}
                # The 2-server MoE body's glue key router_norm_one_mul: the FP16 copy takes its weight multiply as one
                # instruction on the repeated weight row (the same bits).
                if h16 is not None and getattr(moe_body, "router_norm_one_mul", False):
                    norm = {**norm, "w16_row": WeightRow(norm_sb, k + 1)}
                ffn_norm_done = [] if scatter_keep else None
                if scatter_keep:
                    norm["done"] = ffn_norm_done
                act_norm = {"act_square_rows": norm_scalar_squares} if norm_scalar_squares else {}
                if one_round:
                    act_norm = {**act_norm, "one_round": True, "out_dtype": stream.dtype}
                h = poisoned(
                    rmsnorm_hT(mixed, post_w, eps, ones, out=layer_src["ffn_x"], out16=h16, **norm, **act_norm),
                    li,
                    "ffn_in",
                )
            if scatter_keep and kda_conv_keep:
                # The last reads of the scatter views, after the norm in the schedule and after each scatter's
                # completion, into the sinks.
                kda_read_scatter_keep(scatter_keep, {key: sink for key, (sink, _) in keep_sinks.items()}, ffn_norm_done)
            elif scatter_keep:
                # Without kda_conv_keep: one read of each scatter's source column and pool-row index, after the norm
                # in the schedule only, into the sink of its dtype.
                for src_rows, index, _ in scatter_keep:
                    views = (src_rows[0 : index.shape[0], 0:1], index) if index is not None else (src_rows[:, 0:1],)
                    for view in views:
                        rows = view.shape[0]
                        sink = keep_sinks[str(view.dtype)][0][0:rows, 0:1]
                        read = nisa.tensor_tensor(
                            dst=sink, data1=sink, data2=view, op=nl.add, engine=nisa.engine.vector
                        )
                        read = read if hasattr(read, "depends_on") else nisa.NkiInstruction(read)
                        read.depends_on(*ffn_norm_done, kind=nisa.dependency.schedule)
            if layer_debug is not None:
                store_hT_rows(layer_debug[li, 2], h)
            # The weight DMAs of the feed-forward body (also the expert stream of moe_tkg_decode) wait
            # for the attention output collective. gate_all_dmas gates the DMAs from HBM, and only the
            # first DMA into each buffer view (per_queue: the first DMA of each queue). The activation DMAs
            # wait for the collective anyway.
            set_weight_gate(attn_released)
            per_queue = dma_gates == "per_queue" and routed_kernel == "sel_load" and not is_dense_layer(layer)
            # With ffn_weights_after_attn the weight loads also wait for the residual add, the first reader of the
            # attention output, so they do not stream during the flight of the attention output collective.
            ffn_after, ffn_weights = (), ()
            if ffn_weights_after_attn:
                ffn_after = (attn_read if hasattr(attn_read, "depends_on") else nisa.NkiInstruction(attn_read),)
                ffn_weights = (
                    *(dense_weights or ()),
                    *(moe_weights or ()),
                    moe_router_weights,
                    moe_in_proj_scales,
                    moe_out_proj_scales,
                    moe_peer_in_projs,
                )
            # kda_gap_loads in_proj: the MoE in-projection follows the end of the KDA input projection.
            gap_early, gap_weights = (), ()
            if gap_marks is not None:
                kernel_assert(gap_marks.get("end") is not None, f"{tag}: the KDA body gave no input-projection marks")
                end = gap_marks["end"]
                gap_early = (end if hasattr(end, "depends_on") else nisa.NkiInstruction(end),)
                gap_weights = (moe_weights[0],)
            with gate_all_dmas(
                per_queue=per_queue, after=ffn_after, weights=ffn_weights, early=gap_early, early_weights=gap_weights
            ):
                if is_dense_layer(layer):
                    weights = tuple(_layer_view(t, 0, n_dense) for t in dense_weights)
                    ffn_sources = ffn_layer_sources(layer_src, layer in x_alternates, layer in x_hbm_layers)
                    f = dense_body(h, *weights, rank_info, dp_group, world_group, tag, sources=ffn_sources, **ff_dp)
                else:
                    weights = tuple(_layer_view(t, moe_k, n_moe) for t in moe_weights)
                    stacked = {}
                    if routed_kernel == "sel_load":
                        weights, stacked = _sel_load_layer_weights(moe_weights, moe_k, n_moe, moe_expert_base)
                    # Fewer expert-loop buffers make room for the collective tiles: else the compiler spills them.
                    # The dynamic expert loop takes all its buffers (see ``tkg_moe.dynamic_expert_loop``).
                    # The DP4 x TP32 body frees the TP128 body tiles (collective_sources), so it keeps three buffers.
                    lean = {}
                    if dynamic_experts:
                        lean = {"dynamic_experts": True}
                    elif routed_kernel == "static":
                        # With the MLA groups of 8 (above 8 sequences) 2 weight buffers fit, so the next expert's weights
                        # load during the current expert; the DP body keeps the default.
                        buffers = 2 if mla_grouped or (arena_sources and not dp_moe) else None
                        lean = {"scale_buffers": 1, **({"weight_buffers": buffers} if buffers else {})}
                        if mla_grouped and moe_half_weight_buffers:
                            lean["half_weight_buffers"] = moe_half_weight_buffer_count or 2 * buffers
                        # The 2-server options (each None keeps the rule above).
                        if moe_weight_buffers is not None:
                            lean["weight_buffers"] = moe_weight_buffers
                        if moe_scale_buffers is not None:
                            lean["scale_buffers"] = moe_scale_buffers
                        if moe_transpose_lookahead is not None:
                            lean["transpose_lookahead"] = moe_transpose_lookahead
                        if moe_cross_expert_order:
                            lean["cross_expert_order"] = True
                        if moe_cross_expert_transpose_blocks:
                            lean["cross_expert_transpose_blocks"] = True
                        if moe_token_tiles_transpose_blocks:
                            lean["token_tiles_transpose_blocks"] = True
                        if moe_transpose_block:
                            lean["transpose_block"] = True
                        if moe_gate_up_half_major:
                            lean["gate_up_half_major"] = True
                        if moe_shared_scale_buffer:
                            lean["shared_scale_buffer"] = True
                        if moe_routed_store_form != "relayout":
                            lean["routed_store_form"] = moe_routed_store_form
                        if moe_direct_latent:
                            lean["direct_latent"] = True
                        if moe_preamble_buffers:
                            lean["preamble_buffers"] = moe_preamble_buffers
                        if moe_preamble_reuse:
                            lean["preamble_reuse"] = True
                        if moe_router_rows_drop_selection:
                            lean["router_rows_drop_selection"] = True
                        if moe_latent_gather_mx:
                            lean["latent_gather_mx"] = True
                        if moe_static_alt_tiles:
                            lean["static_alt_tiles"] = moe_static_alt_tiles
                        # In an MLA layer, the early first loads of the expert loop overlap the MLA output
                        # reduce-scatter and slow it down.
                        if moe_early_loads == "none" or (moe_early_loads == "kda" and is_mla_layer(layer)):
                            lean["early_loads"] = False
                        if moe_latent_load != "strided":
                            lean["latent_load"] = moe_latent_load
                        if expert_path == "slots":
                            lean["active_expert_slots"] = moe_active_expert_slots
                            if moe_active_expert_counts is not None:
                                lean["active_expert_count"] = moe_active_expert_counts[moe_k : moe_k + 1, 0:1]
                    if base_glue:
                        lean["glue_dmas"] = base_glue
                    if moe_latent_shard_form != "one_hot":
                        lean["latent_shard_form"] = moe_latent_shard_form
                    if h16 is not None:
                        lean.update(
                            router_dtype=router_dtype,
                            router_weight=_layer_view(moe_router_weights, moe_k, n_moe),
                            x16_hT=h16,
                        )
                    if moe_in_proj_scales is not None:
                        lean["in_proj_scale"] = _layer_view(moe_in_proj_scales, moe_k, n_moe)
                    if moe_out_proj_scales is not None:
                        lean["out_proj_scale"] = _layer_view(moe_out_proj_scales, moe_k, n_moe)
                    if moe_in_proj_form != "gathered":
                        lean.update(
                            in_proj_form=moe_in_proj_form, peer_in_proj=_layer_view(moe_peer_in_projs, moe_k, n_moe)
                        )
                    if moe_out_proj_form != "reduced":
                        lean.update(out_proj_form=moe_out_proj_form)
                    ffn_sources = ffn_layer_sources(layer_src, layer in x_alternates, layer in x_hbm_layers)
                    window, window_owner = {}, None
                    if kda_w_in_prefetch is not None:
                        in_block = li + 1 < num_layers
                        next_kda = in_block and not is_mla_layer(layer + 1)
                        next_w_in = _layer_view(kda_weights[0], kda_k, n_kda) if next_kda else None
                        window = kda_w_in_prefetch.window(layer, moe_body, next_w_in, in_block)
                        window_owner = kda_w_in_prefetch if window else None
                    if mla_head_prefetch is not None and not window:
                        # mla_k already counts this layer: it indexes the next MLA layer of the block.
                        next_mla = li + 1 < num_layers and is_mla_layer(layer + 1)
                        next_head = (
                            (_layer_view(mla_weights[0], mla_k, n_mla), _layer_view(mla_weights[2], mla_k, n_mla))
                            if next_mla
                            else None
                        )
                        window = mla_head_prefetch.window(layer, moe_body, next_head)
                        window_owner = mla_head_prefetch if window else None
                    prework_box = None
                    if kda_prework_window and li + 1 < num_layers and not is_mla_layer(layer + 1):
                        # The next KDA layer's prework: inside this layer's routed all-reduce, after the window's loads.
                        kernel_assert(
                            getattr(moe_body, "cc_window", False),
                            "kda_prework_window needs a MoE body with a window in its routed all-reduce (the Trn2 mxfp4 "
                            "body)",
                        )
                        prework_box = []
                        next_heads = _layer_view(kda_weights[1], kda_k, n_kda).shape[0] // P_MAX
                        window = {
                            "cc_window": _window_with_prework(
                                window.get("cc_window"),
                                prework_box,
                                lambda pool=kv_pools[li + 1],
                                ids=kda_page_ids[kda_k],
                                heads=next_heads,
                                name=f"L{layer + 1}",
                                k=kda_k: kda_prework(
                                    pool, ids, heads, name, **_prework_weights(kda_prework, kda_weights, k, n_kda)
                                ),
                            )
                        }
                    glue_hooks = {}
                    if TAIL_POOL_PIN and getattr(moe_body, "moe_sharding", None) == "dp_2srv":
                        # Only the 2-server body takes glue hooks (its compaction tail calls this one).
                        glue_hooks["tail_gathers"] = set_pool_after
                    if moe_in_proj_hook is not None and getattr(moe_body, "moe_sharding", None) == "dp_2srv":
                        # moe_in_proj_hook: after the layer's MXFP8 in-projection loads.
                        glue_hooks["in_proj_loads"] = moe_in_proj_hook
                    if layer + 1 in kda_w_in_heads:
                        # The head of the next KDA layer: in this layer's output glue, after the routed result load.

                        def head_after_routed_load(reduce, load, nxt=layer + 1):
                            _check_head_room(region, kda_w_in_heads[nxt], f"loaded in layer {nxt - 1}")
                            address, chunks, w_in = kda_w_in_heads[nxt]
                            kda_ahead_carry[nxt] = kda_w_in_head_loads(
                                w_in, address, chunks, f"L{nxt}", after_load=load, after_collective=reduce
                            )

                        glue_hooks["routed_load"] = head_after_routed_load
                    front_box = []
                    if front_ahead_of is not None and li + 1 < num_layers and not is_mla_layer(layer + 1):
                        # The front of the next KDA layer: in the flight of this layer's output collective.
                        next_heads = _layer_view(kda_weights[1], kda_k, n_kda).shape[0] // P_MAX

                        def front_after_out_trigger(
                            reduce,
                            box=front_box,
                            pool=kv_pools[li + 1],
                            page_ids=kda_page_ids[kda_k],
                            heads=next_heads,
                            name=f"L{layer + 1}",
                            tiles=kda_front_tiles[kda_k % 2],
                        ):
                            box.append(
                                front_ahead_of(
                                    pool, page_ids, position, heads, name, sources=tiles or None, after=(reduce,)
                                )
                            )

                        glue_hooks["out_trigger"] = front_after_out_trigger
                    if kda_states_after_out_ar and li + 1 < num_layers and not is_mla_layer(layer + 1):
                        # The next KDA layer's state loads follow this layer's output all-reduce trigger.

                        def states_after_out_trigger(reduce, prev=glue_hooks.get("out_trigger"), nxt=layer + 1):
                            if prev is not None:
                                prev(reduce)
                            kda_states_after[nxt] = (reduce,)

                        glue_hooks["out_trigger"] = states_after_out_trigger
                    routed_box = []
                    if moe_sums_after_routed:
                        # The routed reduction of this layer: the next layer's row sums follow its trigger.
                        glue_hooks["routed_trigger"] = routed_box.append
                    # The body loads the padding-row flags from the input: no tile lives across the block loops.
                    pad = {} if moe_pad_rows is None else {"pad_rows": moe_pad_rows}
                    # One expert-loop scope for the layer: it counts the two-buffer static calls (trace_hazards).
                    in_proj = {}
                    if in_proj_tile is not None:
                        in_proj, in_proj_tile = {"in_proj_sb": in_proj_tile}, None
                    if front_prefetch is not None:
                        # moe_prefetch_start: the scope above gates the ring loads as the body's own weight loads.
                        prefetch = front_prefetch(gate=False)
                    with expert_loop_layer():
                        f = moe_body(
                            h,
                            *weights,
                            rank_info,
                            dp_group,
                            world_group,
                            tag,
                            sources=ffn_sources,
                            **lean,
                            **routed,
                            **stacked,
                            **ff_dp,
                            **window,
                            **({"glue_hooks": glue_hooks} if glue_hooks else {}),
                            **pad,
                            **in_proj,
                            **({"prefetch": prefetch} if prefetch is not None else {}),
                            **({"hot_experts": hot} if hot is not None and hot_body else {}),
                            # The per-kind pe_keep_warm of the 2-server body: this layer's attention kind.
                            **(
                                {"keep_warm_lead": moe_keep_warm_lead(layer)}
                                if getattr(moe_body, "keep_warm_by_lead", False)
                                else {}
                            ),
                        )
                    if window_owner is not None:
                        window_owner.check_issued(layer + 1)
                    if prework_box is not None:
                        kernel_assert(
                            len(prework_box) == 1,
                            f"the MoE body of layer {layer} did not trace the KDA prework of layer {layer + 1} once",
                        )
                        prework = prework_box[0]
                    kernel_assert(
                        layer + 1 not in kda_w_in_heads or layer + 1 in kda_ahead_carry,
                        f"the MoE body of layer {layer} did not load the kda_w_in head of layer {layer + 1}",
                    )
                    kernel_assert(
                        front_ahead_of is None or "out_trigger" not in glue_hooks or len(front_box) == 1,
                        f"the MoE body of layer {layer} did not trace the KDA front of layer {layer + 1}",
                    )
                    front_ahead = front_box[0] if front_box else None
                    kernel_assert(
                        not (kda_states_after_out_ar and li + 1 < num_layers and not is_mla_layer(layer + 1))
                        or layer + 1 in kda_states_after,
                        f"kda_states_after_out_ar: the MoE body of layer {layer} did not call its out_trigger hook, so "
                        f"the state loads of KDA layer {layer + 1} would lose their edge",
                    )
                    kernel_assert(
                        not moe_sums_after_routed or len(routed_box) == 1,
                        f"the MoE body of layer {layer} did not call its routed_trigger hook once ({len(routed_box)})",
                    )
                    routed_after[:] = routed_box
                    moe_k += 1
            after_ffn = kda_w_in_prefetch is not None and kda_w_in_prefetch.place in _AFTER_FFN_PLACES
            ffn_first = released()[:1] if after_ffn else ()
            ffn_done = released()[-1:] if after_ffn and kda_w_in_prefetch.place == "after_ffn_done" else ()
            ffn_released, ffn_late = take_released_pair(early=early_ffn_gate)
            if hot is not None:
                # The last use of each hot-expert tile: the op that the body reports after the call (it uses the
                # tiles as ring slots), else its output collective; for another body, one word read after the MoE.
                if hot_body:
                    reported = hot.get("last_use") or {}
                    late = tuple(ffn_late or ffn_released)
                    uses = {key: (reported[key],) if key in reported else late for key in HOT_TILE_KEYS}
                else:
                    uses = keep_hot_tiles(hot, keep_sinks["hot_experts"][0], ffn_late or ffn_released)
                for key, last in uses.items():
                    kernel_assert(last, f"{tag}: moe_hot_prefetch needs the weight gates or the last use from the body")
                    prefetch_gate.release(f"hot_{key}", *last)
            if prefetch_frame:
                kernel_assert(
                    (len(region.sbm.heap), region.sbm.stack_curr_addr) == prefetch_marks,
                    f"{tag}: a body left SBUF tiles in the frame of the prefetch tiles",
                )
                region.close_frame()
            if after_ffn:
                # after_ffn_cc: the next KDA layer's ring fills follow this body's first output collective, so they
                # stream beside the later ones, not after the last one.
                in_block = li + 1 < num_layers
                nxt = in_block and not is_mla_layer(layer + 1)
                weights = tuple(_layer_view(t, kda_k, n_kda) for t in kda_weights) if nxt else None
                kda_w_in_prefetch.after_ffn(layer, weights, in_block, ffn_first, ffn_done)
            # The next KDA layer's page metadata and state loads need neither the residual nor the weights, so
            # they are traced here and run beside the output collective of this body.
            if kda_prework is not None and prework is None and li + 1 < num_layers and not is_mla_layer(layer + 1):
                heads = kda_local_heads(*(_layer_view(t, kda_k, n_kda) for t in kda_weights[1:3]))
                prework = kda_prework(
                    kv_pools[li + 1],
                    kda_page_ids[kda_k],
                    heads,
                    f"L{layer + 1}",
                    **_prework_weights(kda_prework, kda_weights, kda_k, n_kda),
                )
            # attn_res_ahead: the bank parts of the next KDA layer's mixes, beside this body's output collective. The
            # MLP mix of a layer that pushes a bank entry, or of a bank above ATTN_RES_AHEAD_SHARED_ENTRIES, takes its part
            # after the attention norm.
            ahead_marks = None
            if li + 1 < num_layers and not is_mla_layer(layer + 1):
                ahead_entries = n_bank >= attn_res_ahead_min_entries
                kinds = ("attn",) if ahead and ahead_entries else ()
                shared = n_bank <= ATTN_RES_AHEAD_SHARED_ENTRIES and not banks_snapshot(layer + 1)
                if ahead_mlp and ahead_entries and shared:
                    kinds = (*kinds, "mlp")
                if kinds:
                    # attn_res_ahead_before_resid: the residual add below follows the parts' weighted sums.
                    ahead_marks = [] if attn_res_ahead_before_resid else None
                    built = build_ahead(li + 1, kinds, after=tuple(ffn_late or ()), marks=ahead_marks)
                    ahead_next = built.get("attn", ahead_next)
                    ahead_mlp_part = built.get("mlp", ahead_mlp_part)
            if layer_debug is not None:
                store_hT_rows(layer_debug[li, 3], f)

            final = last_block and arena_sources and li == num_layers - 1
            resid = layer_sources(sources, layer + 1)["ffn_x"] if final else next_stream()
            resid_add = nisa.tensor_tensor(dst=resid, data1=s_hT, data2=f, op=nl.add)
            if ahead_marks:
                # attn_res_ahead_before_resid: the parts' VectorE work runs in the flight of the output collective.
                resid_add = resid_add if hasattr(resid_add, "depends_on") else nisa.NkiInstruction(resid_add)
                resid_add.depends_on(*ahead_marks, kind=nisa.dependency.schedule)
            if front_ahead is not None:
                # The front of the next KDA layer (traced in this body's output collective): the residual add follows
                # its VectorE ops in the schedule, so they run in the flight.
                resid_add = resid_add if hasattr(resid_add, "depends_on") else nisa.NkiInstruction(resid_add)
                resid_add.depends_on(*front_ahead["vector_done"], kind=nisa.dependency.schedule)
            s_hT = resid
            if layer_streams is not None:
                store_hT_rows(layer_streams[li], s_hT)
            if after_layer is not None:
                after_layer(layer, s_hT)
            if region is not None:
                region.close_span()
            if layer_vectors and region is not None:
                region.close_frame()
    for sink, hbm in keep_sinks.values():
        nisa.dma_copy(dst=hbm, src=sink)
    if region is not None:
        region.close_frame()
    if bank_hT is None:
        return s_hT, None, ffn_released
    kept_rstd = mix_cache or bank_sums or tensor_sums
    return s_hT, ((bank_hT, n_bank, bank_rstd) if kept_rstd else (bank_hT, n_bank)), ffn_released


@functools.wraps(_decode_block_body)
def decode_block_body(*args, **kwargs):
    # The MoE tail gather pin (TAIL_POOL_PIN) ends with the body's trace, also when the trace fails, so a pin never
    # reaches a later trace.
    try:
        return _decode_block_body(*args, **kwargs)
    finally:
        clear_pool_after()


def _tensor_mix_options(act_mix: dict) -> dict:
    """Return the ``common.attn_res_tensor_hT`` options of the ScalarE options ``act_mix`` of a plain mix.

    The stream squares keep ``act_square_rows`` and the entry products ``act_product_rows`` (and ``act_product_stt``); the
    bank squares are kept
    from the pushes (no ``act_bank_square_rows``), and the ``sums_done`` / ``act_done`` lists pass through.
    """
    keys = ("act_square_rows", "act_product_rows", "act_product_stt", "sums_done", "act_done")
    return {key: act_mix[key] for key in keys if key in act_mix}


def push_bank_entry(bank_hT, index: int, s_hT: nl.NkiTensor, mix_act=(), mix_edge=None):
    """Write the stream ``s_hT`` (SBUF ``[128, T, n]``) to entry ``index`` of the AttnRes bank, in the home of the entry.

    An SBUF bank takes a copy that follows the last ScalarE square of the mix (``mix_act``: the ``act_done`` of the mix
    of ``attn_res_scalar_squares_order``). A DRAM bank takes a DMA that waits for the mix that reads its last entry
    (``mix_edge``: the ``read_out`` of that mix; None or empty adds no wait). An SBUF entry of a ``common.SplitBank``
    takes the DMA of ``SplitBank.push``, which waits for no mix: no mix has read the entry yet. Returns the push.
    """
    hbm_bank = bank_hT.buffer != nl.sbuf
    sbuf_entry = isinstance(bank_hT, SplitBank) and bank_hT.in_sbuf(index)
    push = nisa.dma_copy if hbm_bank else nisa.tensor_copy
    if sbuf_entry:
        push_h = bank_hT.push(index, s_hT)
    else:
        push_h = push(dst=bank_hT[0:P_MAX, index], src=s_hT)
    if mix_act and not hbm_bank:
        push_instr = push_h if hasattr(push_h, "depends_on") else nisa.NkiInstruction(push_h)
        push_instr.depends_on(mix_act[-1], kind=nisa.dependency.schedule)
    if hbm_bank and mix_edge and not sbuf_entry:
        # The HBM bank push waits for the mix that reads the last bank entry.
        push_instr = push_h if hasattr(push_h, "depends_on") else nisa.NkiInstruction(push_h)
        mix_done = mix_edge[-1]
        mix_done = mix_done if hasattr(mix_done, "depends_on") else nisa.NkiInstruction(mix_done)
        push_instr.depends_on(mix_done, kind=nisa.dependency.runtime)
    return push_h


def _prework_weights(kda_prework, kda_weights, kda_k: int, n_kda: int) -> dict:
    """Return the ``weights`` argument of a KDA body's ``prework`` that takes them (its ``takes_weights``, the verify
    ``kda_replay_ahead``): the ``(w_small, params)`` of KDA layer ``kda_k`` of the block. Empty for any other prework, so
    its call does not change."""
    if not getattr(kda_prework, "takes_weights", False):
        return {}
    return {"weights": tuple(_layer_view(t, kda_k, n_kda) for t in kda_weights[2:4])}


def _window_with_prework(window, box: list, prework):
    """Return the ``cc_window`` function of a MoE layer (``kda_prework_window``): it runs ``window`` (the S3P loads of
    the layer's window, or None), then traces ``prework()`` (the next KDA layer's prework) and appends it to ``box``. It
    returns the DMAs of ``window``, and keeps the attributes of ``window``."""

    def run(after, wait=()):
        dmas = [] if window is None else list(window(after, wait))
        box.append(prework())
        return dmas

    if window is not None:
        run.__dict__.update(getattr(window, "__dict__", {}))
    return run


def _check_head_room(region, head, where: str) -> None:
    """Fail the trace if the explicit bytes of a ``kda_w_in`` head ``(address, chunks, kda_w_in)`` meet the stack or the
    heap of the explicit region (None: no region, nothing to check)."""
    if region is None:
        return
    address, chunks, w_in = head
    end = address + chunks * kda_w_in_head_bytes(w_in)
    stack_top, heap_low = region.sbm.stack_curr_addr, region.sbm.heap_curr_addr
    kernel_assert(
        stack_top <= address and end <= heap_low,
        f"kda_w_in head [{address}, {end}) {where} meets the explicit region (stack up to {stack_top}, heap from "
        f"{heap_low})",
    )


def _tensor_bank_options(bank_view, read_out, bank_dma: str = None) -> dict:
    """Return the ``common.attn_res_tensor_hT`` options of a DRAM bank (the verify bank): ``read_out`` (the list that
    the HBM bank push waits on, None for none) and the entry-load queue ``bank_dma``; {} for an SBUF bank."""
    if bank_view.buffer == nl.sbuf:
        return {}
    options = {} if read_out is None else {"read_out": read_out}
    return {**options, **({"bank_dma": bank_dma} if bank_dma else {})}


def new_bank_rstd(entries: int, tokens: int, region: ExplicitSbuf = None) -> nl.NkiTensor:
    """Return the fp32 ``[128, C, T]`` tile of the bank-entry rstd values (``mix_cache``), for the whole kernel."""
    shape = (P_MAX, entries, tokens)
    if region is None:
        return nl.ndarray(shape, dtype=nl.float32, buffer=nl.sbuf)
    return region.heap(shape, nl.float32, "bank_rstd")


def _act_squares(mix: dict, on: bool, stream_hT) -> dict:
    """Return the ``attn_res_hT`` options ``mix`` with ``act_stream_squares`` (``attn_res_cached_act_squares``) when
    ``on`` and the mix is a cached squares-first mix (kept bank rstd, early bank part) with a weight row over a bf16
    stream of more than one row; else ``mix``."""
    _, T, _ = stream_hT.shape
    takes = (
        on
        and mix.get("squares_first")
        and mix.get("bank_part") is not None
        and "w_row" in mix
        and T > 1
        and stream_hT.dtype == nl.bfloat16
    )
    return {**mix, "act_stream_squares": True} if takes else mix


def _with_weight_row(mix: dict, w_row: WeightRow, count: int) -> dict:
    """Return the ``attn_res_hT`` options ``mix`` with the weight row ``w_row`` of the mix's score weight, so that its
    weighted sums take one multiply and one sum for each candidate (``common._row_sums``), unless the options already
    hold one or the bank is empty."""
    if count == 0 or "w_row" in mix:
        return mix
    return {**mix, "w_row": w_row}


def _bank_sum_options(bank_rstd, count: int) -> dict:
    """Return the ``attn_res_hT`` options of ``attn_res_cached_bank_sums``: the kept rstd of the bank entries only."""
    if count == 0:
        return {}
    return {"bank_rstd": bank_rstd[0:P_MAX, 0:count]}


def _mix_options(bank_rstd, count: int, w_row: WeightRow) -> dict:
    """Return the ``attn_res_hT`` options of ``mix_cache``: the kept rstd of the bank entries, the weight row and the
    VectorE softmax exponentials."""
    if count == 0:
        return {}
    return {"bank_rstd": bank_rstd[0:P_MAX, 0:count], "w_row": w_row, "vector_exp": True}


# Per-token MLA parity sets of persistent gather tiles. Two sets let a token's all-gather source keep its bytes
# until a peer has read it (the next token takes the other set); one set keeps both collective arenas smaller at
# 93 layers but relies on the per-sub-call reduce-scatter to separate consecutive reuses (checked by world_scan).
_VERIFY_MLA_PARITY = 1


MOE_SBUF_REDUCES = ("routed", "routed_tp128")
"""The MoE all-reduces that can run in SBUF on the loop path (``kimi_k3_decode_block`` ``moe_sbuf_reduces``):
``routed`` for the base MoE body, ``routed_tp128`` for the bodies of the MoE options (``moe_options_in_use``: the
TP128 routed kernels sel_load and tp_dense, the router forms and types)."""


MOE_EARLY_LOADS = ("all", "kda", "none")
"""The MoE layers whose expert loop issues its first loads before its latent work (``kimi_k3_decode_block``
``moe_early_loads``): every MoE layer, the layers of a KDA attention, or none."""


def check_moe_sbuf_reduces(sbuf_reduces) -> tuple:
    """Return ``sbuf_reduces`` as a tuple after checking each name against ``MOE_SBUF_REDUCES``."""
    sbuf_reduces = tuple(sbuf_reduces)
    kernel_assert(
        all(name in MOE_SBUF_REDUCES for name in sbuf_reduces),
        f"moe_sbuf_reduces {sbuf_reduces} must be names of {MOE_SBUF_REDUCES}",
    )
    return sbuf_reduces


def _flat_sites(prefix: str, sites: dict, tiles: dict) -> tuple:
    """Add each tile of ``sites`` (a tile or a list of tiles per key) to ``tiles`` as ``{prefix}_{key}_{i}``.

    Returns the ``(key, count)`` pairs for ``_regroup_sites``; ``count`` is 0 for a single tile.
    """
    keys = ()
    for key, site in sites.items():
        parts = site if isinstance(site, list) else [site]
        keys += ((key, len(parts) if isinstance(site, list) else 0),)
        for i, part in enumerate(parts):
            tiles[f"{prefix}_{key}_{i}"] = part
    return keys


def _regroup_sites(prefix: str, keys: tuple, tiles: dict) -> dict:
    """Return the sites that ``_flat_sites`` added to ``tiles``, with each list key as a list again."""
    return {
        key: [tiles[f"{prefix}_{key}_{i}"] for i in range(count)] if count else tiles[f"{prefix}_{key}_0"]
        for key, count in keys
    }


def collective_sources(
    tokens: int,
    n_tiles: int,
    dtype,
    region: ExplicitSbuf = None,
    tail_ranks: int = 0,
    moe_options: dict = None,
    mla: bool = True,
    verify_mla_rows: int = 0,
    name: str = "",
    moe: bool = True,
    kda: bool = True,
    x_pair: bool = False,
    sbuf_routed_reduce: bool = False,
    dp_moe: bool = False,
    a2av_moe: str = None,
    pingpong: bool = True,
    x_hbm: bool = False,
    mla_cp: int = 0,
    dp_size: int = DP_SIZE,
    reduce_rows: int = None,
    mla_stage: bool = False,
    sbuf_routed_options: bool = False,
    moe_tokens: int = 0,
    q_hbm: bool = False,
    deparity_rows: int = 0,
    deparity_out: bool = False,
    deparity_q: bool = False,
    deparity_calls: int = 0,
    ffn_x: bool = True,
    out_sets: int = 1,
    verify_k1_heads: int = 0,
    verify_k1_q_hbm: bool = False,
    verify_k1_out_ag: bool = False,
    dst_pingpong: tuple = (),
    moe_out_proj_form: str = "reduced",
    mla_stage_split: tuple = (),
    mla_stage_unit_tokens: int = 0,
    x_world: bool = False,
    token_world: bool = False,
    pair_shared_cc_dsts: tuple = (),
    pair_shared_trn2_entry: bool = False,
    moe_gather_cols: int = None,
    moe_gather_mx: bool = False,
    ring_reclaim: bool = False,
    mla_state_hbm: bool = False,
    x_logits: bool = False,
    mla_rs_hold: bool = False,
) -> dict:
    """Return the persistent SBUF tiles of the all-gathers and all-reduces of the kernel, one tile for each site.

    A peer can read the source of an all-gather or an all-reduce after the local output has landed, and it can
    write a destination before the local rank reaches the collective. The compiler gives the bytes of a dead
    tile to another tile at once. Each layer uses the same tile for a site, so its bytes stay with that site, and
    a world collective separates each use from the next one. With a region, the tiles are heap tiles of the
    region. Without one, they are views of two compiler-placed arenas, one for the destinations and one for the
    other tiles, so that the source and the destination of a collective are never in one SBUF tensor. The arenas
    live to their last use, the tail token store when ``tail_ranks`` is set, so no tile of the last layers loses
    its bytes early. A spill would reload a tile to new bytes, so the caller must leave room (see the expert-loop
    buffers of ``decode_block_body``). The routed and output all-reduces use the persistent HBM tensors of
    ``moe_collective_hbm``.

    ``moe_options`` (``moe_option_sites``) adds the sites of the MoE options (``moe_options_in_use``). Their region
    exists also off the MLA loop path, so the static-path tiles (the arenas) then follow the loop path, not the region.

    ``moe_gather_cols`` (None: 28 words) sizes the HBM latent gather rows of the base MoE body
    (``tkg_moe.moe_collective_hbm`` ``gather_cols``): ``tkg_moe.SCORE_GATHER_COLS`` for
    ``moe_router_rows_drop_selection``. ``moe_gather_mx`` makes them the UINT32 MXFP8 blocks of
    ``moe_latent_gather_mx`` (``moe_collective_hbm`` ``latent_gather_mx``).

    ``x_pair`` adds the second persistent x gather destination of the loop path (``tkg_moe.moe_collective_tiles``).
    ``x_hbm`` adds the persistent HBM x gather destination ``x_dst_hbm`` for the layers after an MLA loop.
    ``ring_reclaim`` (region, base MoE body, HBM x and latent gathers) puts the latent gather source above, and the x
    gather destination ``x_dst`` below, the other feed-forward tiles of the region (``ffn_x``, its ping-pong twin and
    the routed all-reduce source). None of these is touched by a layer from the load of its latent world gather to the
    end of its expert loop, apart from the first half of ``x_dst`` (the routed all-reduce destination, which peers
    write); the routing reads the latent gather source after that load. The second half of ``x_dst`` and the tiles
    above it up to the latent gather source are then one run of bytes, and ``moe_ring_buffer`` is an explicit-address
    half weight buffer of the expert loop on that run (``tkg_moe._ring_buffer_args``).
    ``x_world`` (with ``x_hbm``, DP4) adds the [4, B * 56] HBM source ``x_world_src_hbm`` of the world x gather
    (``decode_block_body`` ``moe_x_world_gather``). Each layer writes it after its world collectives of the layer
    before, which every peer joins only after it has read the source, so it needs no second set.
    ``mla_stage`` gives the MLA groups one set of HBM collective tensors for all groups (``mla_staged``).
    ``mla_stage_split`` (``MLA_STAGE_SPLITS`` of ``decode_block_body`` ``mla_stage_cc_split``) gives the named sites of
    that set one HBM source and destination for each unit of ``mla_stage_unit_tokens`` sequences instead
    (``tkg_mla_paged.mla_stage_hbm``).
    ``mla_state_hbm`` gives each MLA group of the loop path (no arenas, one group, not staged) the persistent FP32 HBM
    source and destination of its state reduce-scatter (``state_hbm`` / ``state_dst_hbm``), which the BF16-o state
    (``mla_decode_attention`` ``state_bf16_o``) needs, and the FP32 pair of its BF16-o sums (``mla_state_sum_hbm``), so
    the two reduce-scatters of the state write tensors of their own; the arenas and the 2-group sets have the state
    pair already. False (default) keeps the program.
    ``mla_rs_hold`` gives the MLA group of the loop path (one group, not staged) the HBM tensors of the static path:
    the state and output reduce-scatter pairs and the output all-gather destination, so no MLA reduce-scatter runs
    from SBUF tiles. It also adds two parity sets of the reduce-scatter sources and destinations of each MLA group
    (``MLA_RS_HOLD_KEYS``), under the key ``"mla_rs_hold"`` (``mla_rs_hold_sources``). The MLA layers take the sets in
    turn by their MLA layer count, so the reduce-scatter source of an MLA layer is written again only two MLA layers
    later. See ``mla_rs_hold_sources`` for the reason. Set 0 is the HBM set of the group. Staged groups
    (``mla_stage``) keep their one HBM set. False (default) keeps the program.
    ``x_logits`` (router_dtype fp16_local in the base MoE body) packs the DP x gather rows with the local router logits
    (``tkg_moe.moe_collective_tiles`` ``x_logits``; the result has the key ``x_logits``). With ``token_world`` it
    adds the FP16 row pair of the world token gather instead (``x16_token_world_src_hbm`` / ``_dst_hbm``).
    ``sbuf_routed_reduce`` runs the routed all-reduce of the MoE bodies in SBUF on the loop path, from a persistent
    source into a persistent destination (``tkg_moe.moe_collective_tiles``). It is for the TP128 projections:
    with ``dp_moe`` the kernel fails, because the DP4 x TP32 body keeps its HBM routed all-reduce.
    ``sbuf_routed_options`` does the same for the bodies of the MoE options (``moe_options`` not None; the
    ``routed_tp128`` name of ``MOE_SBUF_REDUCES``), on the loop path, with the same persistent tiles. The bf16
    routed all-reduce only: the FP32 one (``routed_reduce_dtype`` fp32) keeps its HBM pair.
    ``dp_moe`` gives the sites of the DP4 x TP32 MoE body (``tkg_moe_dp.dp_moe_collective_hbm``) in place of the
    TP128 body tiles (``tkg_moe.moe_collective_tiles`` ``moe_body_tiles``), with the MoE tiles (``moe``), at the
    rows of one MoE call (``moe_tokens``, else ``tokens``). ``a2av_moe`` (an expert form) adds the HBM tensors of the
    all-to-all-v body (``decode_moe_a2av.a2av_moe_hbm``) for all ``tokens`` rows: that body runs one call on them.

    ``pingpong`` adds a second set of the sources that a peer can read after the local completion of their
    collective, under the key ``"pingpong"``: the DP4 gather sources of ``_PINGPONG_TILES`` (with ``moe``; else the
    block FFN input is not a gather source) and the HBM sources of ``_PINGPONG_HBM``. The odd layers take the second
    set (``layer_sources``). Layer k + 2 writes the set of layer k only after a collective of layer k + 1 on the same
    group, which completes locally only after each peer has completed the collective of layer k.

    ``dst_pingpong`` (default ()) adds to the ``"pingpong"`` set a second persistent HBM destination for each gather
    site it names (``dst_pingpong_keys``), with or without ``pingpong``. Two consecutive layers then gather into
    different tensors: the runtime skips the start handshake of an all-gather only when its destination does not
    overlap the one of the previous all-gather of the group. A peer writes the tensor of layer k again in its gather
    of layer k + 2, after its gather of layer k + 1 on the same group, and the source of that gather on each rank
    needs every read of the tensor of layer k. The source of a gather is never one of its destinations.

    ``mla_cp`` (128) adds the sites of the CP128 MLA body (``tkg_mla_block.mla_cp_decode_attention``) for ``tokens`` rows,
    under the key ``mla_cp``. ``reduce_rows`` sizes the reduction tensors for that many rows (see ``decode_block_body``).
    ``ffn_x=False`` leaves out the FFN input tile, for feed-forward sources whose body gathers from another tile.
    ``out_sets`` > 1 adds the output all-reduce sources of the feed-forward chunks of one layer under ``out_hbm_sets``
    (the first is ``out_hbm``): in a dense layer no world collective separates the all-reduce of one chunk from the
    output store of the next. With ``moe_out_proj_form`` ``column`` it also adds ``gather_hbm_sets``: for each chunk,
    the HBM pairs of the latent and the output world gathers (the first set is the base pairs). Those gathers skip
    their start handshake, and nothing orders a peer's gather of the next chunk of a layer after this rank's reads of
    the chunk before, so each chunk takes its own pairs. ``column_shared_ag`` adds them too, and ``shared_ag_sets``:
    the SBUF tiles of the shared gather for each chunk (the first set is the base tiles), for the same reason.

    ``verify_k1_heads`` (the TP32 heads of a rank) adds the sites of the multi-query verify MLA call
    (``tkg_mla_block.mla_block_attention`` on the ``tokens`` rows) under the key ``verify_k1``, with the state
    reduce-scatter in persistent HBM. ``verify_k1_q_hbm`` puts the query gather of that call in one persistent HBM
    pair (``q_call_src_hbm`` / ``q_call_dst_hbm``) in place of the SBUF ``q`` tiles of each 4-token tile.
    ``verify_k1_out_ag`` gives that call one ``ag_call`` tile for one output all-gather a call.

    ``moe_out_proj_form`` (``tkg_moe.OUT_PROJ_FORMS``) other than ``reduced`` needs the SBUF routed all-reduce of
    the MoE option bodies on the loop path (``sbuf_routed_options``). ``column`` and ``reduced_wide`` run it in the
    wide layout: the persistent ``routed_wide_src_sb`` / ``routed_wide_dst`` tiles
    (``tkg_moe.out_proj_collective_tiles``) take the place of ``routed_src_sb``. ``column_shared_ag`` keeps
    ``routed_src_sb`` and adds the persistent tiles of the shared gather (``tkg_moe.shared_gather_tiles``). The
    column forms (``tkg_moe.COLUMN_FORMS``) add the HBM pair of the output all-gather (``out_proj_collective_hbm``).

    ``pair_shared_cc_dsts`` (``common.PAIR_SHARED_CC_DST_SITES``, ``check_pair_shared_forms``) puts the persistent HBM
    destinations of those sites in LNC1 shared HBM: ``moe_out`` the output all-reduce destination ``out_dst_hbm``,
    ``moe_routed`` the routed all-reduce destinations ``routed_dst_hbm`` / ``routed32_dst_hbm``, ``moe_gather`` the latent
    world gather destination ``gather_dst_hbm`` (the first set only: the second destination of ``dst_pingpong`` and the
    per-chunk pairs of ``out_sets`` stay private), and ``kda_ar`` the KDA all-reduce destination ``kda_ar_dst_hbm``. A
    site whose collective does not run in these tensors (an SBUF latent gather or routed all-reduce, a column output
    form) shares nothing. Every source and every other destination stays private. ``pair_shared_trn2_entry``: the caller
    is a Trn2 entry, which takes the sites on trn2 too (``common.check_pair_shared_cc_dsts`` ``trn2_entry``).
    """
    pair_shared = check_pair_shared_forms(
        pair_shared_cc_dsts,
        dp_moe=dp_moe,
        a2av_moe=a2av_moe,
        mla_stage_split=mla_stage_split,
        trn2_entry=pair_shared_trn2_entry,
    )
    kernel_assert(
        not (dp_moe and sbuf_routed_reduce),
        "moe_sbuf_reduces routed is for the TP128 projections; the DP4 x TP32 body keeps its HBM routed all-reduce",
    )
    static = region is None if moe_options is None else not moe_options["loop_path"]
    # The static-path tiles are arena views also with the region of the options: a heap tile keeps its bytes only
    # to its last access, while the arenas live to the end of the kernel (see ``decode_block_body`` ``last_block``).
    plan = _ArenaPlan() if static else SbufTiles(region, heap=True)
    kernel_assert(
        not ring_reclaim
        or (region is not None and not static and moe and ffn_x and pingpong and not x_pair and not dp_moe),
        "ring_reclaim needs the region heap tiles of the base MoE body (loop path, ffn_x with its ping-pong twin)",
    )
    # With ring_reclaim the latent gather source goes above the run: the layer reads it after its latent gather.
    ring_gather = moe_gather_src_tile(moe_tokens or tokens, plan, dp_size) if ring_reclaim else None
    ring_top = region.sbm.heap_curr_addr if ring_reclaim else None
    tiles = {"ffn_x": plan.tile((P_MAX, tokens, n_tiles), dtype, "ffn_x_src")} if ffn_x else {}
    if ring_gather is not None:
        tiles["gather"] = ring_gather
    # On the loop path the latent gather runs in persistent HBM tensors: its destination does not fit the region.
    hbm_gather = region is not None and not dp_moe
    # The verify sizes its MoE gather tiles by the feed-forward chunk (ffn_rows), not the block tokens, so a
    # separate ffn_sources carries them (moe=True) and the block drops them (moe=False). The gather is over
    # DP_SIZE*tokens rows; at the block tokens (Q*B, up to 64) that would be 256 > 128 SBUF partitions.
    # The verify sizes the MoE gather tiles by the feed-forward chunk; moe_tokens=0 sizes them by tokens.
    moe_tok = tokens if not moe_tokens else moe_tokens
    kernel_assert(moe_out_proj_form in OUT_PROJ_FORMS, f"moe_out_proj_form must be one of {OUT_PROJ_FORMS}")
    wide_routed = moe and moe_out_proj_form in WIDE_FORMS
    shared_gather = moe and moe_out_proj_form == "column_shared_ag"
    kernel_assert(
        not x_logits or (moe_options is None and not dp_moe and a2av_moe is None and dp_size == DP_SIZE),
        f"x_logits (router_dtype fp16_local) needs the base MoE body of the TP128 projections at DP{DP_SIZE}",
    )
    kernel_assert(
        not (ring_reclaim and x_logits),
        "ring_reclaim reclaims the second half of the plain x gather destination: not with x_logits (router_dtype "
        "fp16_local, the packed x gather rows)",
    )
    packed = x_logits and not token_world
    if moe:
        base_moe = moe_options is None
        options_sbuf = sbuf_routed_options and not base_moe and moe_options.get("routed_reduce_dtype", "bf16") == "bf16"
        sbuf_routed_reduce = (sbuf_routed_reduce and base_moe or options_sbuf) and not static and not dp_moe
        kernel_assert(
            moe_out_proj_form == "reduced" or (options_sbuf and sbuf_routed_reduce),
            f"moe_out_proj_form {moe_out_proj_form!r} needs the SBUF routed all-reduce of the MoE options on the loop "
            "path (sbuf_routed_options)",
        )
        tiles.update(
            moe_collective_tiles(
                moe_tok,
                plan,
                static=static,
                moe_body_tiles=not dp_moe,
                x_pair=x_pair and base_moe,
                sbuf_routed_reduce=sbuf_routed_reduce and not wide_routed,
                dp_size=dp_size,
                **({"x_dst": False, "gather": False} if ring_reclaim else {}),
                **({"x_logits": True} if packed else {}),
            )
        )
        if moe_options is not None:
            tiles.update(routed_collective_tiles(tokens, plan, dp_size=dp_size, **moe_options))
        if wide_routed:
            tiles.update(out_proj_collective_tiles(moe_tok, plan, dp_size=dp_size))
        if shared_gather:
            tiles.update(shared_gather_tiles(moe_tok, plan, dp_size=dp_size, name=name))
            # A verify with feed-forward chunks: each chunk of a layer takes its own shared gather tiles.
            for i in range(1, out_sets):
                chunk = shared_gather_tiles(moe_tok, plan, dp_size=dp_size, name=f"{name}_{i}")
                tiles.update({f"{key}_c{i}": tile for key, tile in chunk.items()})
    # The verify MLA body runs a different row count than the block tokens (per-token, or one multi-query MLA call), so
    # it allocates its own gather tiles. Dropping the unused block MLA tiles gives the tight loop-path frame
    # room; decode keeps them (mla=True) because its MLA body reads them.
    # No world collective separates two MLA groups of a layer, so each group keeps its own collective tensors.
    # Staged groups (``mla_staged``) run each collective once for all groups, on one set of HBM tensors.
    groups = _mla_groups(tokens) if mla else []
    # The SBUF collective tiles of 2 groups do not fit, so above 8 sequences the MLA collectives run in HBM.
    group_set = len(groups) > 1
    staged = mla and mla_stage
    group_keys = []
    for prefix, group_tokens in groups:
        group = {} if group_set else mla_collective_sources(group_tokens, plan, static=static, name_prefix=prefix)
        tiles.update({f"{prefix}{key}": tile for key, tile in group.items()})
        group_keys.append(list(group))
    cp_keys = k1_keys = ()
    if mla_cp:
        cp_sites = mla_block_collective_sources(tokens, 1, plan, cp=mla_cp, heads=CP_TARGET_HEADS)
        cp_keys = _flat_sites("mla_cp", cp_sites, tiles)
    if verify_k1_heads:
        k1_sites = mla_block_collective_sources(
            tokens, verify_k1_heads, plan, q_hbm=verify_k1_q_hbm, out_ag_call=verify_k1_out_ag
        )
        k1_keys = _flat_sites("vk1", k1_sites, tiles)
    # Above 8 sequences a per-layer conv-state source can share bytes with a MoE tile that waits only for the trigger.
    if kda and static and group_set:
        tiles.update(kda_conv_rows_tiles(tokens, plan))
    if tail_ranks:
        tiles["argmax_src"] = plan.tile((tokens, 2), nl.float32, "argmax_src")
        tiles["argmax_dst"] = plan.tile((tokens, 2 * tail_ranks), nl.float32, "argmax_dst")
        # The greedy token of the tail is the last write of the kernel. In the source arena it keeps the arena to
        # the token store, so no tail tile takes the bytes of a gather source that a peer can still read.
        tiles["argmax_token"] = plan.tile((tokens, 1), nl.int32, "argmax_token")
    # The block FFN input is a gather source also without the MoE tiles: the chunked verify gathers from its rows.
    pp_tiles = [key for key in _PINGPONG_TILES if key in tiles] if pingpong else []
    for key in pp_tiles:
        tiles[f"{key}_pp"] = _twin_tile(plan, tiles[key], f"{key}_pp_src")
    ring_buffer = None
    if ring_reclaim:
        kernel_assert(
            "routed_src_sb" in tiles and "ffn_x_pp" in tiles,
            "ring_reclaim needs the persistent SBUF routed source and the ffn_x twin",
        )
        tiles["x_dst"] = moe_x_dst_tile(moe_tok, plan, dp_size)
        run = [tiles[key] for key in ("ffn_x", "routed_src_sb", "ffn_x_pp", "x_dst")]
        ring_buffer = _ring_reclaim_buffer(region, ring_top, run)
    # The verify runs its MLA body per token (``verify_mla_rows`` rows per call), so it cannot reuse the
    # block MLA tiles. Allocate two parity sets of per-token MLA gather tiles here, so they live in the same
    # arena (reserved to the tail) and the last layer's gather source is not freed and reused before a peer
    # has read it. verify_mla_rows=0 (decode and the multi-query MLA verify) leaves the arena byte-identical.
    vmla_keys = ()
    vmla_hbm = []
    for i in range(_VERIFY_MLA_PARITY if verify_mla_rows else 0):
        pset = mla_collective_sources(verify_mla_rows, plan, static=static, q_hbm=q_hbm)
        vmla_keys = tuple(pset)
        for key, slot in pset.items():
            tiles[f"vmla{i}_{key}"] = slot
        # Each parity set gets its own persistent HBM state reduce-scatter tensors, so the per-token
        # Each parity set has its own state reduce-scatter HBM; above 16 rows the out all-gather runs in SBUF.
        vmla_hbm.append(
            mla_collective_hbm(verify_mla_rows, name=f"{name}_vmla{i}", out_rs=True, q_hbm=q_hbm, out_ag_sbuf=q_hbm)
        )
    # Each per-query MLA call of the verify takes its own persistent HBM collective tensors.
    verify_mla_state = [
        mla_collective_hbm(
            deparity_rows,
            name=f"{name}_vmlast{c}",
            out_rs=deparity_out,
            q_hbm=deparity_q,
            qkv_hbm=deparity_q,
        )
        for c in range(deparity_calls)
    ]
    if static:
        # An arena with no tile gets no allocation: the device tracer refuses a zero-width tensor.
        arenas = tuple(
            nl.ndarray((P_MAX, width), dtype=nl.bfloat16, buffer=nl.sbuf, name=f"collective_arena{name}_{i}")
            if width
            else None
            for i, width in enumerate(plan.widths)
        )
        tiles = {key: plan.view(arenas, slot) for key, slot in tiles.items()}
        bases = tuple(arena for arena in arenas if arena is not None)
    else:
        bases = tuple(tiles.values())
    # Clear the tiles at the kernel start, so that no earlier tile shares the bytes that a peer can write.
    for tile in bases:
        nisa.memset(dst=tile, value=0.0)
    # The options take their own HBM tensors for all their MoE sites (``routed_collective_hbm``), the base keys
    # included. The KDA all-reduce pair is the base one (``kda_collective_hbm``).
    hbm = {}
    # ``reduce_rows`` sizes the world reductions as the verify's: the runtime picks the algorithm by message size.
    rr = {"reduce_rows": reduce_rows} if reduce_rows else {}
    ps = {"pair_shared": pair_shared} if pair_shared else {}
    if moe and moe_options is None:
        hbm = moe_collective_hbm(
            moe_tok,
            latent_gather=hbm_gather,
            name=name,
            dp_size=dp_size,
            **rr,
            **ps,
            **({"gather_cols": moe_gather_cols} if moe_gather_cols else {}),
            **({"latent_gather_mx": True} if moe_gather_mx else {}),
        )
    elif moe:
        hbm = routed_collective_hbm(tokens, dp_size=dp_size, **rr, **moe_options, **ps)
    if moe and moe_out_proj_form in COLUMN_FORMS:
        hbm.update(out_proj_collective_hbm(moe_tok, name=name, dp_size=dp_size))
    # The DP body takes its tensors at the rows of one MoE call (the verify feed-forward chunk; the tokens in the decode).
    if dp_moe and moe:
        hbm.update(dp_moe_collective_hbm(moe_tok, name=name))
    if a2av_moe is not None:
        hbm.update(a2av_moe_hbm(tokens, name=name, expert_form=a2av_moe))
    if x_hbm and moe:
        rows, cols = tiles["x_dst"].shape
        for key, shape in (
            ("x_dst_hbm", (dp_size * rows, cols // dp_size)),
            ("x_src_gather_hbm", (rows, cols // dp_size)),
        ):
            hbm[key] = nl.ndarray(shape, dtype=nl.bfloat16, buffer=nl.private_hbm, name=f"moe{name}_{key}")
    kernel_assert(
        not x_world or ("x_src_gather_hbm" in hbm and dp_size == DP_SIZE),
        f"x_world needs the HBM x gather pair (x_hbm, the MLA loop path) and the DP{DP_SIZE} layout",
    )
    if x_world:
        rows, cols = hbm["x_src_gather_hbm"].shape
        hbm["x_world_src_hbm"] = nl.ndarray(
            (rows // TP_SIZE, cols), dtype=nl.bfloat16, buffer=nl.private_hbm, name=f"moe{name}_x_world_src_hbm"
        )
    if token_world and moe:
        # attn_rs_moe_input: the world token all-gather of the one normed row per rank (1 token -> WORLD_SIZE tokens)
        # and the stream all-gather of the one residual row over the TP group (1 -> R rows), both through HBM with the
        # collective_dim=0 partition-stack + permute-load path (``tkg_moe._gather_replica_tokens``, the s' all-gather).
        token_pairs = (
            ("x_token_world_src_hbm", (P_MAX, n_tiles)),
            ("x_token_world_dst_hbm", (WORLD_SIZE * P_MAX, n_tiles)),
            ("s_prime_src_hbm", (P_MAX, n_tiles)),
            ("s_prime_dst_hbm", (TP_SIZE * P_MAX, n_tiles)),
        )
        if x_logits:
            token_pairs += (
                ("x16_token_world_src_hbm", (P_MAX, n_tiles)),
                ("x16_token_world_dst_hbm", (WORLD_SIZE * P_MAX, n_tiles)),
            )
        for key, shape in token_pairs:
            hbm[key] = nl.ndarray(shape, dtype=nl.bfloat16, buffer=nl.private_hbm, name=f"moe{name}_{key}")
    # The verify KDA body has its own all-reduce pair at its row count (kda=False).
    if kda:
        kda_ps = {"pair_shared": True} if "kda_ar" in pair_shared else {}
        hbm.update(kda_collective_hbm(tokens, n_tiles, name=name, **rr, **kda_ps))
    if out_sets > 1:
        out = hbm["out_hbm"]
        hbm["out_hbm_sets"] = [out] + [
            nl.ndarray(out.shape, dtype=out.dtype, buffer=nl.private_hbm, name=f"moe{name}_out_hbm_{i}")
            for i in range(1, out_sets)
        ]
    if out_sets > 1 and moe and moe_out_proj_form in COLUMN_FORMS:
        keys = [key for key in COLUMN_CHUNK_GATHER_KEYS if key in hbm]
        hbm["gather_hbm_sets"] = [{key: hbm[key] for key in keys}] + [
            {
                key: nl.ndarray(
                    hbm[key].shape, dtype=hbm[key].dtype, buffer=nl.private_hbm, name=f"moe{name}_{key}_{i}"
                )
                for key in keys
            }
            for i in range(1, out_sets)
        ]
    kernel_assert(staged or not mla_stage_split, "mla_stage_split needs the staged MLA groups (mla_stage)")
    stage = (
        mla_stage_hbm(tokens, name=f"{name}_stage", split_sites=mla_stage_split, unit_tokens=mla_stage_unit_tokens)
        if staged
        else None
    )
    # mla_rs_hold: the loop path also takes the HBM set of the static path, so its reduce-scatters do not run from
    # SBUF tiles. The BF16-o state keeps the FP32 pair of its sums there.
    loop_hold = mla_rs_hold and not static and not group_set
    if (static or group_set or mla_rs_hold) and not staged:
        for (prefix, group_tokens), keys in zip(groups, group_keys, strict=True):
            group_name = f"{name}_{prefix[:-1]}" if prefix else name
            group = mla_collective_hbm(group_tokens, name=group_name, out_rs=True, gathers=group_set)
            if loop_hold and mla_state_hbm:
                group.update(mla_state_sum_hbm(group_tokens, name=group_name))
            hbm.update({f"{prefix}{key}": tensor for key, tensor in group.items()})
            keys.extend(group)
    elif mla_state_hbm and not staged:
        for (prefix, group_tokens), keys in zip(groups, group_keys, strict=True):
            group_name = f"{name}_{prefix[:-1]}" if prefix else name
            group = {
                **mla_collective_hbm(group_tokens, name=group_name, keys=("state_hbm", "state_dst_hbm")),
                **mla_state_sum_hbm(group_tokens, name=group_name),
            }
            hbm.update({f"{prefix}{key}": tensor for key, tensor in group.items()})
            keys.extend(group)
    hold_sets = _mla_rs_hold_sets(groups, hbm, name) if mla_rs_hold and groups and not staged else None
    views = moe_collective_views(tiles, packed) if moe else {}
    pp_hbm = [key for key in _PINGPONG_HBM if key in hbm] if pingpong else []
    pp_hbm += dst_pingpong_keys(dst_pingpong, hbm)
    second = {key: tiles.pop(f"{key}_pp") for key in pp_tiles}
    second.update(
        {
            key: nl.ndarray(hbm[key].shape, dtype=hbm[key].dtype, buffer=nl.private_hbm, name=f"pp{name}_{key}")
            for key in pp_hbm
        }
    )
    result = {**tiles, **views, **hbm}
    if ring_buffer is not None:
        result["moe_ring_buffer"] = ring_buffer
    if packed and moe:
        result["x_logits"] = True
    if second:
        result["pingpong"] = second
    if hold_sets is not None:
        result["mla_rs_hold"] = hold_sets
    if shared_gather and out_sets > 1:
        result["shared_ag_sets"] = [{key: result[key] for key in SHARED_GATHER_KEYS}] + [
            {key: result.pop(f"{key}_c{i}") for key in SHARED_GATHER_KEYS} for i in range(1, out_sets)
        ]
    if staged:
        result["mla_stage"] = stage
        result["mla_groups"] = [
            mla_stage_views(stage, g * MLA_GROUP_TOKENS, count) for g, (_, count) in enumerate(groups)
        ]
        result["mla_pack_hbm"] = {}
    elif mla:
        result["mla_groups"] = [
            {key: result[f"{prefix}{key}"] for key in keys}
            for (prefix, _), keys in zip(groups, group_keys, strict=True)
        ]
    if mla_cp:
        cp_sources = _regroup_sites("mla_cp", cp_keys, tiles)
        cp_sources.update(
            mla_block_collective_hbm(
                tokens, 1, cp=mla_cp, heads=CP_TARGET_HEADS, name=f"{name}_cp", reduce_rows=reduce_rows
            )
        )
        result["mla_cp"] = cp_sources
    if verify_k1_heads:
        result["verify_k1"] = {
            **_regroup_sites("vk1", k1_keys, tiles),
            **mla_block_collective_hbm(
                tokens, verify_k1_heads, name=f"{name}_vk1", q_hbm=verify_k1_q_hbm, out_ag_call=verify_k1_out_ag
            ),
        }
    if verify_mla_rows:
        # Regroup the flat parity slots plus each set's state-RS HBM tensors into the per-token MLA
        # source dicts the verify body passes.
        result["verify_mla"] = [
            {**{k: tiles[f"vmla{i}_{k}"] for k in vmla_keys}, **vmla_hbm[i]} for i in range(_VERIFY_MLA_PARITY)
        ]
    if verify_mla_state:
        result["verify_mla_state"] = verify_mla_state
    if pair_shared:
        # The program takes --lnc1-with-shared-mem: the MoE bodies read this key to keep their scratch private.
        result[PAIR_SHARED_SOURCES_KEY] = pair_shared
    return result


def layer_sources(sources: dict, layer: int) -> dict:
    """Return the collective sources of global layer ``layer``: an odd layer takes the second set (``pingpong``)."""
    second = sources.get("pingpong")
    return {**sources, **second} if second and layer % 2 else sources


MLA_RS_HOLD_KEYS = ("state_hbm", "state_dst_hbm", "state_sum_hbm", "state_sum_dst_hbm", "out_rs_src_hbm", "out_rs_dst_hbm")
"""The HBM tensors of the state and output reduce-scatters of an MLA group that ``collective_sources`` ``mla_rs_hold``
gives two parity sets of. The output all-gather destination keeps its one tensor."""


def mla_layer_ordinal(layer: int) -> int:
    """Return the count of MLA layers before global layer ``layer`` (0 for layer 3, 23 for layer 92)."""
    return sum(1 for i in range(layer) if is_mla_layer(i))


def mla_rs_hold_sources(sources: dict, layer: int, group: int = 0) -> dict:
    """Return the HBM reduce-scatter tensors of MLA group ``group`` of global layer ``layer`` (``mla_rs_hold``).

    Return {} for sources without ``"mla_rs_hold"``. Each MLA layer takes the set of its MLA layer count modulo 2.

    The hold is a kernel workaround for a defect of the switch-mesh reduce-scatter in Neuron runtime 2.35. A rank can
    complete its reduce-scatter while a slower peer on its chip still reads the source of the rank. The compiler orders
    the next write of the source bytes only after the local completion. Thus the write can change the bytes before the
    peer reads them, and the peer sends a wrong sum to another rank. With async scheduling (2 queued executions), about
    1% of the decode streams at the 1M bucket get NaN logits. On the MLA loop path the reduce-scatters run from SBUF
    tiles, and the compiler can give their bytes to a new tile at once. With the hold, MLA layer k uses set k mod 2, and
    only MLA layer k + 2 writes that set again. That write follows the reduce-scatters of layer k + 1 on the same group,
    and those start only after each peer completes the reduce-scatters of layer k. A step has 24 MLA layers, so the hold
    continues into the next step. The output all-gather of the MLA layer keeps its one HBM destination.
    """
    hold = sources.get("mla_rs_hold")
    if not hold:
        return {}
    return hold[mla_layer_ordinal(layer) % 2][group]


def _mla_rs_hold_sets(groups: list, hbm: dict, name: str) -> list:
    """Return the two parity sets of ``collective_sources`` ``mla_rs_hold``: a list of one dict for each MLA group.

    Set 0 takes the HBM tensors that the group already has (``MLA_RS_HOLD_KEYS`` in ``hbm``) and adds the others. Set 1
    has tensors of its own. The BF16-o sums pair (``mla_state_sum_hbm``) is in the sets only when the group has it.
    """
    sets = [[], []]
    for prefix, group_tokens in groups:
        group_name = f"{name}_{prefix[:-1]}" if prefix else name
        base = {key: hbm[f"{prefix}{key}"] for key in MLA_RS_HOLD_KEYS if f"{prefix}{key}" in hbm}
        sums = "state_sum_hbm" in base
        for parity, parity_sets in enumerate(sets):
            have = base if parity == 0 else {}
            own_name = f"{group_name}_rs_hold{parity}"
            keys = tuple(
                key for key in ("state_hbm", "state_dst_hbm", "out_rs_src_hbm", "out_rs_dst_hbm") if key not in have
            )
            own = mla_collective_hbm(group_tokens, name=own_name, out_rs=True, out_ag_sbuf=True, keys=keys) if keys else {}
            if sums and "state_sum_hbm" not in have:
                own.update(mla_state_sum_hbm(group_tokens, name=own_name))
            parity_sets.append({**have, **own})
    return sets


def _ring_reclaim_buffer(region: ExplicitSbuf, top: int, run: list) -> nl.NkiTensor:
    """Return the explicit-address half weight buffer of ``collective_sources`` ``ring_reclaim``.

    ``top`` is the heap address before ``ffn_x``; ``run`` holds the heap tiles allocated since then, in order:
    ``ffn_x``, the routed all-reduce source, the ``ffn_x`` twin and ``x_dst`` (the last).
    The check below fails the trace if any other heap tile lies between them. The buffer takes the bytes from the
    second half of ``x_dst`` up, within ``top``.
    """
    bottom = region.sbm.heap_curr_addr
    sizes = [
        -(
            -(tile.shape[1] * (2 if tile.dtype == nl.bfloat16 else 4) * (tile.shape[2] if len(tile.shape) > 2 else 1))
            // 32
        )
        * 32
        for tile in run
    ]
    kernel_assert(
        top - bottom == sum(sizes),
        f"ring_reclaim: the heap from {bottom} to {top} holds other tiles besides the feed-forward run {sizes}",
    )
    x_dst_bytes = run[-1].shape[1] * 2
    start = bottom + x_dst_bytes // 2
    piece = moe_ring_piece_bytes()
    kernel_assert(
        start + piece <= top,
        f"ring_reclaim: the run [{start}, {top}) above the routed all-reduce destination holds {top - start} B, "
        f"less than one half weight buffer ({piece} B)",
    )
    region.count += 1
    return nl.ndarray(
        (P_MAX, piece // 2),
        dtype=nl.float4_e2m1fn_x4,
        buffer=nl.sbuf,
        name=f"xsb_moe_ring_buffer_{region.count}",
        address=(0, start),
    )


def _twin_tile(plan, tile, name: str):
    """Return a new tile of ``plan`` with the shape and dtype of ``tile`` (an ``_ArenaPlan`` slot or a heap tile)."""
    shape, dtype = plan.slots[tile][2:] if isinstance(plan, _ArenaPlan) else (tuple(tile.shape), tile.dtype)
    return plan.tile(shape, dtype, name)


def mla_staged(tokens: int, loop_path: bool) -> bool:
    """Return True if the MLA layers of ``tokens`` sequences run their groups of 8 in lockstep.

    Above 8 sequences, in full groups, every group yields its collectives at the same sites, and the block issues each
    one once for all groups (``mla_stage_collective``): 5 collectives for each MLA layer instead of 5 per group. Off the
    loop path the key pages of every group can live across the collectives, so there it stages at most 16 sequences.
    """
    if tokens <= MLA_GROUP_TOKENS or tokens % MLA_GROUP_TOKENS:
        return False
    return loop_path or tokens <= 2 * MLA_GROUP_TOKENS


def _run_staged(bodies, stage, tag: str, issue=mla_stage_collective):
    """Run the generator ``bodies`` of the groups of one MLA layer in lockstep. Return their results.

    Each body runs to its next collective; when all of them wait at the same site, the collective is issued once on
    the tensors of ``stage`` and its handle goes to every body. A site without a collective (``MLA_SYNC_SITES``: the
    ``pack_setup_ahead`` site of the MLA body) only syncs the bodies there; each gets None. The groups run their work between two collectives in
    order, so the SBUF holds the tiles of one group at a time, besides what a group keeps across a collective.
    A split site of ``stage`` (``mla_stage_hbm`` ``split_sites``) is issued for each unit of groups as soon as the
    last group of the unit reaches it, on the tensors of the unit, before the next group runs. Every body then gets
    the handle of the last unit, so the weight loads that follow the collective of a site keep their place.
    ``issue`` (tests: a collective on one rank) issues a collective with the arguments of ``mla_stage_collective``.
    """
    split = stage.get("split_sites", ())
    unit_groups = stage.get("unit_tokens", MLA_GROUP_TOKENS) // MLA_GROUP_TOKENS
    kernel_assert(not split or len(bodies) % unit_groups == 0, "the staged MLA groups must fill whole units")
    results = [None] * len(bodies)
    pending = [None] * len(bodies)
    unit_handle = [None]

    def advance(i, handle):
        """Run body ``i`` to its next collective (``handle`` None: from its start); issue a split unit's collective."""
        try:
            pending[i] = next(bodies[i]) if handle is None else bodies[i].send(handle)
        except StopIteration as done:
            pending[i], results[i] = None, done.value
        cc = pending[i]
        if cc is not None and cc.site in split and (i + 1) % unit_groups == 0:
            members = pending[i + 1 - unit_groups : i + 1]
            kernel_assert(all(m is not None and m.site == cc.site for m in members), f"MLA groups apart at {cc.site}")
            unit_handle[0] = issue(stage, cc, tag, unit=i // unit_groups)

    for i in range(len(bodies)):
        advance(i, None)
    while pending[0] is not None:
        site = pending[0].site
        kernel_assert(all(cc is not None and cc.site == site for cc in pending), f"MLA groups apart at {site}")
        if site in MLA_SYNC_SITES:
            # A sync site without a collective (the MLA body's pack_setup_ahead): the groups only meet there.
            handle = None
        else:
            handle = unit_handle[0] if site in split else issue(stage, pending[0], tag)
        unit_handle[0] = None
        for i in range(len(bodies)):
            advance(i, handle)
    kernel_assert(all(cc is None for cc in pending), "MLA groups end apart")
    return results


def _mla_groups(tokens: int) -> list[tuple[str, int]]:
    """Return the name prefix and the sequences of each MLA group of 8 (one unprefixed group up to 8)."""
    if tokens <= MLA_GROUP_TOKENS:
        return [("", tokens)]
    return [
        (f"g{g}_", min(MLA_GROUP_TOKENS, tokens - g * MLA_GROUP_TOKENS)) for g in range(-(-tokens // MLA_GROUP_TOKENS))
    ]


def moe_option_sites(
    routed_kernel: str, router_mode: str, router_dtype: str, routed_reduce_dtype: str, mla_loop_path: bool
) -> dict | None:
    """Return the ``moe_options`` of ``collective_sources`` for the MoE options, or None for the base body."""
    if not moe_options_in_use(routed_kernel, router_mode, router_dtype, routed_reduce_dtype):
        return None
    return {
        "router_mode": router_mode or "fused",
        "router_dtype": router_dtype,
        "routed_reduce_dtype": routed_reduce_dtype,
        "loop_path": mla_loop_path,
    }


class _ArenaPlan:
    """Column ranges of two BF16 SBUF arenas: one for the collective destinations, one for the other tiles."""

    def __init__(self):
        self.slots = []
        self.widths = [0, 0]

    def tile(self, shape, dtype, name=None):
        """Record a tile of ``shape`` and ``dtype`` at the next 32-byte aligned columns; return its slot."""
        arena = 1 if name and name.endswith("_dst") else 0
        self.slots.append((arena, self.widths[arena], tuple(shape), dtype))
        self.widths[arena] += -(-math.prod(shape[1:]) * _DTYPE_BYTES[dtype] // 32) * 16
        return len(self.slots) - 1

    def view(self, arenas, slot: int) -> nl.NkiTensor:
        """Return the arena view of a slot."""
        arena, offset, shape, dtype = self.slots[slot]
        base = arenas[arena] if dtype == nl.bfloat16 else arenas[arena].view(dtype)
        start = offset * 2 // _DTYPE_BYTES[dtype]
        view = base[0 : shape[0], start : start + math.prod(shape[1:])]
        return view.reshape_dim(1, shape[1:]) if len(shape) > 2 else view


_DTYPE_BYTES = {nl.bfloat16: 2, nl.float32: 4, nl.int32: 4}


def region_ones(region: ExplicitSbuf):
    """Return the all-ones tile of the region, and create it on the first call."""
    if region.ones is None:
        region.ones = explicit_ones(region)
    return region.ones


def region_streams(region: ExplicitSbuf, tokens: int, n_tiles: int, dtype):
    """Return the two stream tiles of the region (the residual alternates between them)."""
    if region.streams is None:
        region.streams = tuple(region.heap((P_MAX, tokens, n_tiles), dtype, f"stream_{i}") for i in range(2))
    return region.streams


@nki.jit
def kimi_k3_decode_block(
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
    kda_page_ids: nl.NkiTensor,
    mla_block_table: nl.NkiTensor,
    mla_slot: nl.NkiTensor,
    position: nl.NkiTensor,
    rank_info: nl.NkiTensor,
    tp_group: ReplicaGroup,
    dp_group: ReplicaGroup,
    world_group: ReplicaGroup,
    first_layer: int,
    eps: float,
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
    mla_fp8_pages: bool = False,
    poison_rows: nl.NkiTensor = None,
    poison_at: tuple = None,
    routed_kernel: str = "static",
    routed_reduce_dtype: str = "bf16",
    router_mode: str = None,
    router_dtype: str = "bf16",
    moe_router_weight: nl.NkiTensor = None,
    affinity_scaling: str = "pre_quant",
    affinity_hi_lo: bool = True,
    mla_packed_loop: bool = False,
    mla_pack_once: bool = True,
    mla_pack_slots: int | None = None,
    mla_pack_tail_slots: int | tuple | None = None,
    mla_pack_tail_low_batch: bool = False,
    mla_pack_page_runs: int = 0,
    mla_pack_null_copy_vector: bool = False,
    mla_pack_null_runs: bool = False,
    mla_zeroed_pages: bool = False,
    mla_loop_first_static: bool = False,
    mla_scores_ahead: bool = False,
    mla_folded_reference: bool = False,
    mla_cp_rows_options: dict | None = None,
    mla_pair_pages: bool = False,
    moe_sbuf_reduces: tuple = (),
    lora_norm_eps: float = LORA_NORM_EPS,
    moe_early_loads: str = "all",
    moe_dynamic_loop: bool = False,
    routing_form: str = None,
    shared_sharding: str = "tp128",
    latent_sharding: str = "tp128",
    moe_routed_exchange: str = "all_reduce",
    cc_source_pingpong: bool = True,
    moe_active_expert_slots: int = 0,
    mla_cp: int = 0,
    dp_size: int = DP_SIZE,
    mla_options: dict = None,
    reduce_rows: int = None,
    mix_cache: bool = False,
    moe_in_proj_form: str = "gathered",
    moe_peer_in_proj: nl.NkiTensor = None,
    moe_out_proj_form: str = "reduced",
):
    """Run a decode block of Kimi-K3 layers for one DP replica on one rank.

    The block runs the layers ``first_layer .. first_layer + L - 1``, where ``L`` is the leading
    dimension of ``norm_weights``. The AttnRes bank starts empty, so the block starts at a multiple
    of 12. The entry does not take the bank of earlier blocks: for a range of the full model, use
    ``decode_model.kimi_k3_decode_model``. It updates the page pools in place and returns them, so that
    they alias the inputs.

    Dimensions:
        H: hidden size, 7168. L: layers in the block, 1 to 12.

    Args:
        stream: ``[B, H]`` bf16 stream at the block input, one row for each sequence of the replica (the
            token embeddings for layer 0), B <= 32. Above 8, the MLA layers run as groups of 8.
        norm_weights: ``[L, 2, H]`` bf16 input and post-attention norm weights, in hT order.
        attn_res_weights: ``[L, 2, H]`` fp32 AttnRes score weights (``norm * proj``) of the
            attention and MLP mixes, in hT order.
        kda_*: the KDA weights (``tkg_kda.KDA_WEIGHT_NAMES``), stacked ``[n_kda, ...]``.
        mla_*: the MLA weights (``tkg_mla_paged.MLA_WEIGHT_NAMES``), stacked ``[n_mla, ...]``.
        moe_*, expert_*: the latent MoE weights of ``routed_kernel`` (``tkg_moe.moe_weight_names``),
            stacked ``[n_moe, ...]``. With ``sel_load``, ``expert_gate_up_weights`` holds the packed gate/up rows
            ``[n_moe, 896, 128, 768]`` uint8, ``expert_down_weights`` ``[n_moe, 896, 8, 3584]`` and
            ``expert_down_weights_scale`` ``[n_moe, 896, 3584]`` (or ``[n_moe, 896, 2, 3584]``, two scale groups:
            ``tkg_moe.sel_load_down_groups``); ``expert_gate_up_weights_scale`` is not read.
        dense_*: the dense MLP weights (``tkg_moe.DENSE_WEIGHT_NAMES``), stacked ``[1, ...]``.
        kda_page_ids: ``[n_kda, B, 1]`` int32 state page of each KDA layer and sequence.
        mla_block_table: ``[n_mla, B, NB]`` int32 local block table of each MLA layer and sequence.
        mla_slot: ``[n_mla, B, 1]`` int32 slot of the current row of each MLA layer and sequence.
        position: ``[B, 1]`` int32 position of the current token of each sequence.
        rank_info: ``[1, 4]`` int32 (global rank, DP rank, TP rank, 0).
        tp_group, dp_group, world_group: the TP32, DP4 and world replica groups.
        first_layer: global index of the first layer.
        eps: RMSNorm epsilon.
        debug: also return the stream after each layer and the input and output of each body (tests).
        dma_gates: gate the weight DMAs of each body on the collectives of the previous body
            (``"per_queue"``: see ``decode_block_body``).
        pool_of_layer: static index of the page pool of each layer of the block, for pools that several
            layers share. The cache groups the layers by cache type, and one pool holds the pages of one layer
            of each group, so layers 4k .. 4k+3 share pool k. The layers that share a pool use disjoint
            pages, and page 0 is the null page of the pool. None gives each layer its own pool.
        mla_loop_path: let the MLA layers take the loop path, for a block table of more than 16 local
            blocks (up to 256, 1M context). The cost of the loop path follows the context of each
            sequence. The entry then gives explicit SBUF addresses to the tiles that live through the
            loops (see ``tkg_sbuf_alloc``).
        kv_pool_i: ``[P, 73728]`` bf16 page pool i. Pools ``0 .. k-1`` are used, where k is the number
            of pools (L when ``pool_of_layer`` is None).
        mla_fp8_pages: the MLA layers keep float8_e4m3fn pages in their bf16 pools. Each MLA layer views
            its pool, with no copy, as ``[2P, 73728]`` fp8 pages of 128 rows, which its block table and
            slot index. The KDA layers read the same pools as bf16.
        mla_zeroed_pages: the MLA loop path reads its cache pages with no clamp (see ``decode_block_body``).
        mla_loop_first_static: the CP128 MLA loop path runs its first iteration as static code (see
            ``decode_block_body``).
        mla_scores_ahead: the CP128 MLA loop computes each weight pair ahead of its sums (see ``decode_block_body``).
        mla_folded_reference: the CP128 MLA loop adds the reference in its rope matmul (see ``decode_block_body``).
        mla_cp_rows_options: the rows-form keys of the CP128 MLA decode (see ``decode_block_body``).
        mla_pair_pages: the FP8 MLA pages hold the token-pair layout (see ``decode_block_body``).
        mla_pack_page_runs: one DMA per run of 2 or 4 pages of a packed-loop key group (see ``decode_block_body``).
        mla_pack_null_copy_vector: the null-entry page fix of the packed loop on VectorE (see ``decode_block_body``).
        mla_pack_null_runs: a padded sequence counts as page runs (see ``decode_block_body``).
        moe_sbuf_reduces: the MoE all-reduces that run in SBUF on the loop path: ``("routed",)`` for the routed
            all-reduce (``collective_sources``). Default: all in HBM.
        lora_norm_eps: the epsilon of the MLA q_a and kv_a RMSNorms (see ``decode_block_body``).
        moe_early_loads: the MoE layers whose expert loop issues its first loads before its latent work:
            ``all`` (default), ``kda`` or ``none`` (see ``decode_block_body``).
        moe_active_expert_slots: K, the active-expert slots of the MoE layers; the dynamic loop takes precedence
            where it applies (see ``decode_block_body``).
        poison_rows, poison_at: test only. ``[B, H]`` bf16 rows, and ``(layer, stage)`` of the block
            layer and the tile that gets them added (the ``poison`` of ``decode_block_body``).
        routed_kernel: routed-expert kernel of the MoE layers, ``static`` (default), ``sel_load`` or ``tp_dense``
            (``tkg_moe.ROUTED_KERNELS``). One choice for the decode and the spec verify of a deployment. A
            kernel other than ``static`` (or any MoE option of ``tkg_moe.moe_options_in_use``) gives the entry an
            explicit SBUF region, as ``mla_loop_path`` does.
        routed_reduce_dtype: add type of the routed all-reduce of the MoE layers, ``bf16`` (default) or ``fp32``.
        router_mode: router form of the MoE layers, ``fused`` (None, the default) or ``split``
            (``tkg_moe.ROUTER_MODES``).
        affinity_scaling: ``sel_load``: ``pre_quant`` (default) or ``post_down`` (``tkg_moe.AFFINITY_SCALINGS``).
        affinity_hi_lo: ``sel_load``: True (default) or False, one quantized term (``tkg_moe.latent_moe_decode``).
        routing_form: TP128 kernels: the form of the top-16 routing (``tkg_moe.ROUTING_FORMS``); None (default)
            takes the default of ``tkg_moe.latent_moe_decode`` (``index_first``).
        router_dtype: router matmul type of the MoE layers, ``bf16`` (default), ``fp16`` or ``fp16_local``
            (``tkg_moe.ROUTER_DTYPES``).
        moe_router_weight: with ``fp16``, ``[n_moe, 128, 56, 7]`` fp16 router rows of this rank for each MoE layer
            (or ``[1, ...]``), hT-tiled: element ``(k, p, j, e)`` is ``router_weight[7 g + e, 128 j + p]`` of layer k.
            With ``fp16_local``, ``[n_moe, 128, 56, 28]``: the rows of the experts of the 4 DP peers of TP rank t,
            element ``(k, p, j, 7 d + e)`` is ``router_weight[7 (32 d + t) + e, 128 j + p]``
            (``moe_decode_torch.router_weight_fp16_local_rank``). The router columns of ``moe_in_proj`` keep their
            place and are not read.
        moe_dynamic_loop: the opt-in dynamic expert loop of the MoE layers (``tkg_moe.dynamic_expert_loop``): only
            the active local experts run, at B = 1 off the MLA loop path.
            The values do not change.
        shared_sharding, latent_sharding: the sharding of the shared expert and of the latent projections of the MoE
            layers: both ``"tp128"`` (``latent_moe_decode``), or both ``"dp4xtp32"`` (``tkg_moe_dp``; the ``moe_*``
            weights then have its layout, ``tkg_moe_dp.DP_MOE_WEIGHT_NAMES``, and the routed kernel is ``static``).
        moe_routed_exchange: ``all_reduce`` (default), or with the ``dp4xtp32`` shardings ``a2av`` / ``a2av_ragged`` /
            ``a2av_blocks``: the all-to-all-v MoE body (``decode_moe_a2av``; B below 32 is padded to 32 rows), which sums the routed
            partials in FP32 at the token owner.
        cc_source_pingpong: alternate the collective sources that a peer can read after the local completion between
            two sets by layer parity (default). Without a runtime that waits for the remote reads, the next write of a
            source must wait for the next collective of the same group. False keeps one set: for a runtime that waits
            for the remote reads.
        mla_cp: 0, or 128: the MLA layers run ``tkg_mla_block.mla_cp_decode_attention`` (one replica on all ranks,
            ``tp_group`` = the world group, the CP128 weights of ``mla_block_torch.mla_block_cp_rank_weights``).
        mix_cache: keep the rstd of each AttnRes bank entry from its push (``decode_block_body``). Always on with
            ``dp_size`` 1.
        moe_in_proj_form, moe_peer_in_proj: the in-projection form of the MoE layers (``tkg_moe.IN_PROJ_FORMS``)
            and, with ``dp_local``, its peer in-projection stack ``[n_moe, 128, 56, 140]`` (``decode_block_body``).
        moe_out_proj_form: the out-projection form of the MoE layers (``tkg_moe.OUT_PROJ_FORMS``,
            ``decode_block_body``). ``reduced`` (default) keeps the program.

    Returns:
        The stream ``[B, H]`` after the block, then (with ``debug``) the stream after each layer
        ``[L, B, H]`` and the per-layer body inputs and outputs ``[L, 4, B, H]``, then the k page pools.
    """
    restart_calls()
    B, H = stream.shape
    mix_cache = mix_cache or dp_size == 1
    num_layers = norm_weights.shape[0]
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
    )
    if pool_of_layer is None:
        pool_of_layer = tuple(range(num_layers))
    pool_of_layer = tuple(int(i) for i in pool_of_layer)
    kernel_assert(len(pool_of_layer) == num_layers, "Give the pool index of each layer of the block")
    n_pools = max(pool_of_layer) + 1
    kernel_assert(sorted(set(pool_of_layer)) == list(range(n_pools)), "pool_of_layer must use each pool 0 .. k-1")
    pools = all_pools[:n_pools]
    kernel_assert(all(p is not None for p in pools), "Give each page pool that pool_of_layer uses")
    layer_pools = tuple(pools[i] for i in pool_of_layer)
    kernel_assert(banks_snapshot(first_layer), "The block starts at a bank snapshot layer")
    layer_streams = None
    layer_debug = None
    if debug:
        layer_streams = nl.ndarray((num_layers, B, H), dtype=stream.dtype, buffer=nl.shared_hbm, name="layer_streams")
        layer_debug = nl.ndarray((num_layers, 4, B, H), dtype=stream.dtype, buffer=nl.shared_hbm, name="layer_debug")
    region = None
    if mla_loop_path or moe_options_in_use(routed_kernel, router_mode, router_dtype, routed_reduce_dtype):
        region = ExplicitSbuf()
        install_identities(region)
    s_hT, _, _ = decode_block_body(
        stream,
        None,
        norm_weights,
        attn_res_weights,
        (kda_w_in, kda_w_out, kda_w_small, kda_params),
        (mla_in_proj, mla_q_a_norm, mla_q_b_proj, mla_kv_a_norm, mla_kv_b_proj, mla_o_proj),
        (
            moe_in_proj,
            moe_out_proj,
            moe_vectors,
            expert_gate_up_weights,
            expert_gate_up_weights_scale,
            expert_down_weights,
            expert_down_weights_scale,
        ),
        (dense_gate_up, dense_down),
        layer_pools,
        kda_page_ids,
        mla_block_table,
        mla_slot,
        position,
        rank_info,
        tp_group,
        dp_group,
        world_group,
        first_layer,
        eps,
        kda_decode_attention,
        mla_cp_decode_attention if mla_cp else mla_decode_attention,
        select_moe_body(shared_sharding, latent_sharding, tp_group, moe_routed_exchange),
        dense_mlp_decode,
        layer_streams,
        layer_debug,
        dma_gates,
        mla_fp8_pages=mla_fp8_pages,
        region=region,
        mla_loop_path=mla_loop_path,
        mla_packed_loop=mla_packed_loop,
        mla_pack_once=mla_pack_once,
        mla_pack_slots=mla_pack_slots,
        mla_pack_tail_slots=mla_pack_tail_slots,
        mla_pack_tail_low_batch=mla_pack_tail_low_batch,
        mla_pack_page_runs=mla_pack_page_runs,
        mla_pack_null_copy_vector=mla_pack_null_copy_vector,
        mla_pack_null_runs=mla_pack_null_runs,
        poison=None if poison_at is None else (int(poison_at[0]), str(poison_at[1]), poison_rows),
        last_block=True,
        routed_kernel=routed_kernel,
        routed_reduce_dtype=routed_reduce_dtype,
        router_mode=router_mode,
        router_dtype=router_dtype,
        moe_router_weights=moe_router_weight,
        affinity_scaling=affinity_scaling,
        affinity_hi_lo=affinity_hi_lo,
        mla_zeroed_pages=mla_zeroed_pages,
        mla_loop_first_static=mla_loop_first_static,
        mla_scores_ahead=mla_scores_ahead,
        mla_folded_reference=mla_folded_reference,
        mla_cp_rows_options=mla_cp_rows_options,
        mla_pair_pages=mla_pair_pages,
        lora_norm_eps=lora_norm_eps,
        moe_active_expert_slots=moe_active_expert_slots,
        x_parity=not moe_options_in_use(routed_kernel, router_mode, router_dtype, routed_reduce_dtype)
        and x_gather_parity(B),
        sbuf_reduces=check_moe_sbuf_reduces(moe_sbuf_reduces),
        moe_early_loads=moe_early_loads,
        moe_dynamic_loop=moe_dynamic_loop,
        routing_form=routing_form,
        cc_source_pingpong=cc_source_pingpong,
        mla_cp=mla_cp,
        reduce_rows=reduce_rows,
        dp_size=dp_size,
        moe_in_proj_form=moe_in_proj_form,
        moe_peer_in_projs=moe_peer_in_proj,
        moe_out_proj_form=moe_out_proj_form,
    )
    out = nl.ndarray((B, H), dtype=stream.dtype, buffer=nl.shared_hbm, name="stream_out")
    store_hT_rows(out, s_hT)
    if region is not None:
        check_identities(region)
    if debug:
        return (out, layer_streams, layer_debug, *pools)
    return (out, *pools)
