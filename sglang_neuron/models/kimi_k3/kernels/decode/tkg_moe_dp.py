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

"""Kimi-K3 decode latent MoE body with DP4 x TP32 projections (128 ranks, LNC1).

The router, the latent projections and the shared expert are TP32 inside each DP replica: each rank computes its
1/32 shard for the R rows of its own replica only, so the router needs no all-gather of the hidden. The routed
experts stay EP128 (7 experts for each rank) on the T = 4 R tokens of the 4 replicas.

Up to the routed latent, the values are bit-identical to ``latent_moe_decode``: the same router scores, selection,
affinities, expert inputs, and routed all-reduce. The output projection sums other partials (K = 304 on 32 ranks,
against K = 76 on 128 ranks), so the output has a different BF16 rounding.

The four collectives run on persistent private-HBM tensors (``dp_moe_collective_hbm``), so no peer accesses SBUF
and no all-reduce takes a compiler scratch in SBUF.
"""

import nki.collectives as ncc
import nki.isa as nisa
import nki.language as nl
from nki.collectives import ReplicaGroup

from .utils.kernel_assert import kernel_assert
from .moe.routed import moe_tkg_decode
from .common import check_pair_group
from .tkg_weight_dma_gate import release_gate_consumer, release_weight_gate
from .tkg_moe import (
    _CRITICAL_DMA_PRIORITY,
    _H_TILES,
    _IN_PROJ_DMA_CHUNKS,
    _PMAX,
    _PSUM_BANK_BYTES,
    _SELECTION_FLOOR,
    DP_SIZE,
    EXPERT_INTERMEDIATE,
    HIDDEN_SIZE,
    LATENT_NORM_EPS,
    LATENT_SHARD,
    LATENT_SIZE,
    LOCAL_EXPERTS,
    NUM_EXPERTS,
    ROUTER_SUM_EPS,
    SHARED_SIZE,
    SITU_BETA,
    SITU_LINEAR_BETA,
    WORLD_SIZE,
    _load_hT_rows,
    _project_to_hT,
    _psum_bank,
    _situ,
    latent_moe_decode,
    moe_collective_hbm,
    pad_row_flags,
    rank_selectors,
    zero_pad_row_scores,
)

MOE_SHARDINGS = ("tp128", "dp4xtp32")
TP_SIZE = WORLD_SIZE // DP_SIZE
# TP rank t computes the router columns of experts [28 t, 28 t + 28), latent columns [112 t, 112 t + 112) and
# shared intermediate columns [192 t, 192 t + 192).
DP_EXPERTS = NUM_EXPERTS // TP_SIZE
DP_LATENT = LATENT_SIZE // TP_SIZE
DP_SHARED = SHARED_SIZE // TP_SIZE
# Column ranges of `moe_in_proj_dp` [128, 56, 524].
DP_ROUTER_COLS = (0, DP_EXPERTS)
DP_LATENT_COLS = (DP_ROUTER_COLS[1], DP_ROUTER_COLS[1] + DP_LATENT)
DP_SHARED_GATE_COLS = (DP_LATENT_COLS[1], DP_LATENT_COLS[1] + DP_SHARED)
DP_SHARED_UP_COLS = (DP_SHARED_GATE_COLS[1], DP_SHARED_GATE_COLS[1] + DP_SHARED)
DP_IN_PROJ_COLS = DP_SHARED_UP_COLS[1]
DP_IN_PROJ_LATENT_COLS = DP_LATENT_COLS[1]
# Row ranges of `moe_out_proj_dp` [304, 7168].
DP_LATENT_UP_ROWS = (0, DP_LATENT)
DP_SHARED_DOWN_ROWS = (DP_LATENT, DP_LATENT + DP_SHARED)
DP_OUT_PROJ_ROWS = DP_SHARED_DOWN_ROWS[1]
# Column ranges of `moe_vectors_dp` [1, 140] (FP32): the router bias of the 28 experts, the latent norm weight.
DP_BIAS_COLS = (0, DP_EXPERTS)
DP_NORM_COLS = (DP_BIAS_COLS[1], DP_BIAS_COLS[1] + DP_LATENT)
DP_VECTOR_COLS = DP_NORM_COLS[1]

DP_MOE_WEIGHT_NAMES = (
    "moe_in_proj_dp",
    "moe_out_proj_dp",
    "moe_vectors_dp",
    "expert_gate_up_weights",
    "expert_gate_up_weights_scale",
    "expert_down_weights",
    "expert_down_weights_scale",
)

# World gather row of each rank and row: 56 words of BF16 latent pairs, 28 scores, 28 selection values.
_GATHER_LATENT_WORDS = (0, DP_LATENT // 2)
_GATHER_SCORE_COLS = (_GATHER_LATENT_WORDS[1], _GATHER_LATENT_WORDS[1] + DP_EXPERTS)
_GATHER_SELECTION_COLS = (_GATHER_SCORE_COLS[1], _GATHER_SCORE_COLS[1] + DP_EXPERTS)
_GATHER_COLS = _GATHER_SELECTION_COLS[1]
# Latent shard of a TP rank in the [128, T * 28] routed layout: partitions 4 t .. 4 t + 3.
_SHARD_PARTITIONS = DP_LATENT // LATENT_SHARD
# The out-projection loads after the experts, in chunks along the hidden: the first matmuls start early.
_OUT_PROJ_DMA_CHUNKS = 4
# Most replica rows: T = 4 R tokens on the partitions of the expert kernel.
MAX_DP_ROWS = _PMAX // DP_SIZE
# The HWDGE queue of the HBM collective stores and loads. In the 93-layer model, a split over the SP and ACT queues
# is slower for each step (16K, 8 sequences): the ACT triggers wait behind the ScalarE work.
_HBM_DMA = {"priority": _CRITICAL_DMA_PRIORITY, "dge_mode": nisa.dge_mode.hwdge, "engine": nisa.engine.sync}


def latent_moe_decode_dp(
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
    weight_buffers: int = 3,
    scale_buffers: int = 1,
    pad_rows: nl.NkiTensor = None,
) -> nl.NkiTensor:
    """Run one Kimi-K3 latent MoE decode layer with DP4 x TP32 projections and return this replica's rows.

    Not ``@nki.jit``: a decode kernel inlines one call for each MoE layer. The attribute ``token_major`` is True.

    Dimensions:
        H: Hidden size, 7168.
        R: Rows of each DP replica, 1 to 32.
        T: Gathered tokens, 4 R. Token r * R + b is row b of replica r.
        Rank g = 32 r + t is TP rank t of replica r. It owns experts [7 g, 7 g + 7).

    Args:
        x_hT (nl.NkiTensor): [128, R, 56] BF16 SBUF, this replica's normalized hidden, token-major hT.
        moe_in_proj_dp (nl.NkiTensor): [128, 56, 524] BF16 HBM, hT-tiled [in, out]: router columns of experts
            [28 t, 28 t + 28), latent down shard (112), shared gate shard (192), shared up shard (192).
        moe_out_proj_dp (nl.NkiTensor): [304, H] BF16 HBM, [in, out]: latent up shard (112 rows), then the shared
            down shard (192 rows).
        moe_vectors_dp (nl.NkiTensor): [1, 140] FP32 HBM: the router bias of experts [28 t, 28 t + 28), then the
            latent norm weight of latent columns [112 t, 112 t + 112).
        expert_gate_up_weights .. expert_down_weights_scale (nl.NkiTensor): this rank's 7 MXFP4 experts in the
            serving layout of ``moe_tkg_decode`` (as ``latent_moe_decode``).
        rank_info (nl.NkiTensor): [1, 4] INT32 HBM, (global rank, DP rank, TP rank, 0).
        dp_group (ReplicaGroup): [[t, t + 32, t + 64, t + 96] for t in 0..31].
        world_group (ReplicaGroup): [[0 .. 127]].
        tp_group (ReplicaGroup): [[32 r .. 32 r + 31] for r in 0..3].
        tag (str): Unique per-layer string. It names the collectives and the DMAs.
        sources (dict): the persistent HBM tensors of the collectives (``dp_moe_collective_hbm``), allocated once
            for the kernel. The values do not change.
        weight_buffers (int): SBUF weight buffers of the expert loop, 1 to 3.
        scale_buffers (int): SBUF gate/up scale buffers of the expert loop, 2 or 1.
        pad_rows (nl.NkiTensor): None (default) keeps the program. The HBM int32 ``[R, 1]`` padding-row input of this
            replica's rows (1 for a padding row; ``tkg_moe.pad_row_flags``): their router scores become 0 before the
            gather, so they route to no expert. The other rows keep their bits.

    Returns:
        out_hT (nl.NkiTensor): [128, R, 56] BF16 SBUF, this replica's MoE output in the token-major hT layout.

    Notes:
        - Collectives: all-gather (DP, router rows), all-gather (TP32, router shards), all-reduce (world, routed
          latent), all-reduce (TP32, output). Each one runs on the HBM tensors of its site: a DMA from SBUF, the
          collective in HBM, a DMA back to SBUF.
        - The local affinities of rank g come from the gathered scores with a one-hot of g: x * 1 and the sum with
          zeros are exact, so the affinities are the ones of ``latent_moe_decode``.

    Pseudocode:
        r, latent_t, shared_g, shared_u = x_r @ moe_in_proj_dp                  # [R, 524]
        rows = all_gather_dp([bf16(latent_t), sigmoid(r), sigmoid(r) + bias])  # [T, 112]
        gathered = all_gather_tp(rows)                                          # [T, 32 * 112]
        affinity = top-16 affinities of experts [7 g, 7 g + 7)
        routed = all_reduce_world(moe_tkg_decode(latent, affinity))             # [128, T * 28]
        normed = rms_norm(routed[rows of replica r], w)[:, 112 t : 112 t + 112]
        y = bf16([normed | situ(shared_g, shared_u)] @ moe_out_proj_dp)         # [R, H]
        out = all_reduce_tp(y)
    """
    _validate_dp_inputs(x_hT, moe_in_proj_dp, moe_out_proj_dp, moe_vectors_dp)
    rows = x_hT.shape[1]
    tokens = DP_SIZE * rows
    name = f"k3moedp_{tag}"
    hbm = _dp_hbm_views(sources, rows)

    in_proj_sb = nl.ndarray((_PMAX, _H_TILES, DP_IN_PROJ_COLS), dtype=nl.bfloat16, buffer=nl.sbuf)
    _load_hT_rows(in_proj_sb, moe_in_proj_dp, _IN_PROJ_DMA_CHUNKS, f"{name}_in_proj")
    bias_sb = nl.ndarray((rows, DP_EXPERTS), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=bias_sb,
        src=moe_vectors_dp.ap(pattern=[[0, rows], [1, DP_EXPERTS]], offset=DP_BIAS_COLS[0]),
        priority=_CRITICAL_DMA_PRIORITY,
        name=f"{name}_bias",
    )
    row_select, _ = rank_selectors(rank_info, f"{name}_rank")
    shard_select, expert_select = _dp_selectors(rank_info, tokens, name)

    # Router and latent down first (the critical path), then the shared gate and up.
    x_rows = x_hT.permute((0, 2, 1))
    router_psum = _psum_rows(rows, DP_IN_PROJ_LATENT_COLS)
    for h_tile_idx in range(_H_TILES):
        nisa.nc_matmul(
            dst=router_psum,
            stationary=x_rows[:, h_tile_idx, :],
            moving=in_proj_sb[:, h_tile_idx, 0:DP_IN_PROJ_LATENT_COLS],
        )
    gather_src = nl.ndarray((rows, _GATHER_COLS), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(
        dst=gather_src[:, _GATHER_LATENT_WORDS[0] : _GATHER_LATENT_WORDS[1]].view(nl.bfloat16),
        src=router_psum[:, DP_LATENT_COLS[0] : DP_LATENT_COLS[1]],
    )
    scores_src = gather_src[:, _GATHER_SCORE_COLS[0] : _GATHER_SCORE_COLS[1]]
    nisa.activation(dst=scores_src, op=nl.sigmoid, data=router_psum[:, DP_ROUTER_COLS[0] : DP_ROUTER_COLS[1]])
    nisa.tensor_tensor(
        dst=gather_src[:, _GATHER_SELECTION_COLS[0] : _GATHER_SELECTION_COLS[1]],
        data1=scores_src,
        data2=bias_sb,
        op=nl.add,
    )
    if pad_rows is not None:
        zero_pad_row_scores(scores_src, pad_row_flags(pad_rows, name))
    gathered = _gather_rows(gather_src, dp_group, tp_group, name, hbm)

    shared_psum = _psum_rows(rows, 2 * DP_SHARED)
    for h_tile_idx in range(_H_TILES):
        nisa.nc_matmul(
            dst=shared_psum,
            stationary=x_rows[:, h_tile_idx, :],
            moving=in_proj_sb[:, h_tile_idx, DP_SHARED_GATE_COLS[0] : DP_SHARED_UP_COLS[1]],
        )
    # Out-projection input, token-major [R, 304]: the normed latent (after the experts), the shared activation.
    act_tok = nl.ndarray((rows, DP_OUT_PROJ_ROWS), dtype=nl.bfloat16, buffer=nl.sbuf)
    shared_bf16 = nl.ndarray((rows, 2 * DP_SHARED), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=shared_bf16, src=shared_psum)
    _situ(
        shared_bf16[:, 0:DP_SHARED],
        shared_bf16[:, DP_SHARED : 2 * DP_SHARED],
        out=act_tok[:, DP_SHARED_DOWN_ROWS[0] : DP_SHARED_DOWN_ROWS[1]],
    )

    latent_sb = nl.ndarray((tokens, TP_SIZE, DP_LATENT), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(
        dst=latent_sb,
        src=gathered[:, :, _GATHER_LATENT_WORDS[0] : _GATHER_LATENT_WORDS[1]].view(nl.bfloat16),
        engine=nisa.engine.vector,
    )
    local_affinities = _dp_local_affinities(
        gathered[:, :, _GATHER_SCORE_COLS[0] : _GATHER_SCORE_COLS[1]],
        gathered[:, :, _GATHER_SELECTION_COLS[0] : _GATHER_SELECTION_COLS[1]],
        expert_select,
    )

    if expert_gate_up_weights.dtype == nl.uint16:
        expert_gate_up_weights = expert_gate_up_weights.view(nl.float4_e2m1fn_x4)
    if expert_down_weights.dtype == nl.uint16:
        expert_down_weights = expert_down_weights.view(nl.float4_e2m1fn_x4)
    expert_partial = moe_tkg_decode(
        latent_sb.reshape((tokens, LATENT_SIZE)),
        local_affinities,
        expert_gate_up_weights,
        expert_down_weights,
        expert_gate_up_weights_scale,
        expert_down_weights_scale,
        SITU_BETA,
        SITU_LINEAR_BETA,
        weight_buffers=weight_buffers,
        scale_buffers=scale_buffers,
    )

    # Routed all-reduce in the [128, T * 28] layout of ``latent_moe_decode``. The 56-byte runs of the layout go to
    # a local SBUF tile first: as a DMA into HBM they run at a fifth of the rate.
    staged = nl.ndarray((WORLD_SIZE, tokens, LATENT_SHARD), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=staged,
        src=expert_partial.ap(pattern=[[LATENT_SHARD, WORLD_SIZE], [LATENT_SIZE, tokens], [1, LATENT_SHARD]], offset=0),
        name=f"{name}_routed_stage",
        **_HBM_DMA,
    )
    routed = _all_reduce_hbm(
        staged.reshape((WORLD_SIZE, tokens * LATENT_SHARD)), hbm["routed"], world_group, f"{name}_routed"
    )
    out_proj_sb = _load_out_proj(moe_out_proj_dp, name)
    normed_tok = _dp_latent_norm(routed, row_select, shard_select, moe_vectors_dp, rows, f"{name}_norm")

    nisa.tensor_copy(dst=act_tok[:, DP_LATENT_UP_ROWS[0] : DP_LATENT_UP_ROWS[1]], src=normed_tok)
    k_tiles = _out_k_tiles(DP_OUT_PROJ_ROWS)
    act_T = _transpose_rows(act_tok, k_tiles)
    partial_hT = _project_to_hT(
        lambda h: [
            (out_proj_sb[0 : stop - start, k_idx, h * _PMAX : (h + 1) * _PMAX], act_T[0 : stop - start, k_idx, :])
            for k_idx, (start, stop) in enumerate(k_tiles)
        ],
        rows,
    )
    out = _all_reduce_hbm(partial_hT, hbm["out"], tp_group, f"{name}_out")
    return out.reshape((_PMAX, rows, _H_TILES))


latent_moe_decode_dp.token_major = True


def select_moe_body(
    shared_sharding: str, latent_sharding: str, tp_group: ReplicaGroup, routed_exchange: str = "all_reduce"
):
    """Return the MoE body of the projection shardings: ``latent_moe_decode`` (``tp128``) or the DP4 x TP32 body.

    Both shardings are ``tp128`` or both are ``dp4xtp32``. The DP body (``dp_moe_body``) has the call form of
    ``latent_moe_decode`` in ``decode_block_body``. ``routed_exchange`` other than ``all_reduce`` selects the
    all-to-all-v body of the DP4 x TP32 projections (``decode_moe_a2av.ROUTED_EXCHANGES``).
    """
    kernel_assert(
        shared_sharding in MOE_SHARDINGS and latent_sharding == shared_sharding,
        f"shared_sharding and latent_sharding must be both tp128 or both dp4xtp32, got {shared_sharding}, "
        f"{latent_sharding}",
    )
    if routed_exchange != "all_reduce":
        from .decode_moe_a2av import a2av_moe_body

        kernel_assert(shared_sharding == "dp4xtp32", "The all-to-all-v MoE body needs the dp4xtp32 projections")
        return a2av_moe_body(tp_group, routed_exchange)
    return latent_moe_decode if shared_sharding == "tp128" else dp_moe_body(tp_group)


def dp_moe_body(tp_group: ReplicaGroup):
    """Return a body with the call form of ``latent_moe_decode`` in ``decode_block_body`` that runs the DP form.

    The decode block calls ``moe_body(x_hT, *weights, rank_info, dp_group, world_group, tag, sources=...,
    scale_buffers=..., weight_buffers=...)``. The returned body adds the TP32 group. Its attribute ``moe_sharding``
    tells ``decode_block.collective_sources`` to add the DP tensors (``dp_moe_collective_hbm``). The DP form keeps
    one gate/up scale buffer in the expert loop.
    """

    def body(
        x_hT,
        moe_in_proj_dp,
        moe_out_proj_dp,
        moe_vectors_dp,
        expert_gate_up_weights,
        expert_gate_up_weights_scale,
        expert_down_weights,
        expert_down_weights_scale,
        rank_info,
        dp_group,
        world_group,
        tag,
        sources: dict,
        weight_buffers: int = 3,
        scale_buffers: int = 1,
        pad_rows=None,
    ):
        return latent_moe_decode_dp(
            x_hT,
            moe_in_proj_dp,
            moe_out_proj_dp,
            moe_vectors_dp,
            expert_gate_up_weights,
            expert_gate_up_weights_scale,
            expert_down_weights,
            expert_down_weights_scale,
            rank_info,
            dp_group,
            world_group,
            tp_group,
            tag,
            sources,
            weight_buffers=weight_buffers,
            scale_buffers=min(scale_buffers, 1),
            **({"pad_rows": pad_rows} if pad_rows is not None else {}),
        )

    body.token_major = True
    body.moe_sharding = "dp4xtp32"
    body.takes_pad_rows = True
    return body


def dp_moe_collective_hbm(replica_rows: int, name: str = "") -> dict:
    """Return the persistent HBM tensors of the DP body collectives that ``moe_collective_hbm`` does not have.

    These are the tensors of the router row gathers: the DP4 source and destination, and the TP32 destination. The
    routed all-reduce uses the routed pair of ``moe_collective_hbm`` (the same shape). The TP32 output all-reduce uses
    the first quarter of its output pair. The output all-reduce of the dense layer uses that pair first, so the pair
    stays live for the whole kernel. Each layer uses the same tensors, and the routed all-reduce of each layer (a
    world collective) separates each use of a byte from the next one. Before the first MoE layer, only persistent HBM
    tensors are live (this output pair and the KDA pair of ``tkg_kda.kda_collective_hbm``), so no other tensor
    gives its bytes to these tensors before a peer can access them.
    """
    tokens = DP_SIZE * replica_rows
    shapes = {
        "dp_rows_src_hbm": (replica_rows, _GATHER_COLS),
        "dp_rows_dst_hbm": (tokens, _GATHER_COLS),
        "dp_shard_dst_hbm": (TP_SIZE * tokens, _GATHER_COLS),
    }
    return {
        key: nl.ndarray(shape, dtype=nl.float32, buffer=nl.private_hbm, name=f"moe{name}_{key}")
        for key, shape in shapes.items()
    }


def dp_moe_test_sources(replica_rows: int, name: str = "") -> dict:
    """Return the HBM tensors of a kernel that calls the DP body without the decode block (tests).

    ``name`` tells the tensors apart when one kernel builds more than one set.
    """
    return {**moe_collective_hbm(replica_rows, name=name), **dp_moe_collective_hbm(replica_rows, name=name)}


def _dp_hbm_views(sources: dict, rows: int) -> dict:
    """Return the HBM tensors of each collective site of the DP body for ``rows`` rows of each replica."""
    tokens = DP_SIZE * rows
    shapes = {
        "dp_rows_src_hbm": (rows, _GATHER_COLS),
        "dp_rows_dst_hbm": (tokens, _GATHER_COLS),
        "dp_shard_dst_hbm": (TP_SIZE * tokens, _GATHER_COLS),
        "routed_hbm": (WORLD_SIZE, tokens, LATENT_SHARD),
        "routed_dst_hbm": (WORLD_SIZE, tokens * LATENT_SHARD),
        "out_hbm": (_PMAX, tokens * _H_TILES),
        "out_dst_hbm": (_PMAX, tokens * _H_TILES),
    }
    for key, shape in shapes.items():
        kernel_assert(key in sources, f"The DP MoE body needs the persistent HBM tensor {key}")
        kernel_assert(tuple(sources[key].shape) == shape, f"{key}: {sources[key].shape} is not {shape}")
    width = rows * _H_TILES
    return {
        "rows": (sources["dp_rows_src_hbm"], sources["dp_rows_dst_hbm"], sources["dp_shard_dst_hbm"]),
        "routed": (sources["routed_hbm"].reshape((WORLD_SIZE, tokens * LATENT_SHARD)), sources["routed_dst_hbm"]),
        # The first quarter of the output pair of the 4 replicas holds the [128, R * 56] rows of this replica.
        "out": tuple(sources[key].reshape((DP_SIZE, _PMAX, width))[0] for key in ("out_hbm", "out_dst_hbm")),
    }


def _gather_rows(gather_src: nl.NkiTensor, dp_group, tp_group, name: str, hbm: dict) -> nl.NkiTensor:
    """All-gather the [R, 112] rows of every rank as [T, 32, 112]: the rows of the 4 replicas, then the 32 shards.

    An HBM all-gather stacks the ranks on dim 0: [R, 112] to [T, 112] (DP4), then to [32 T, 112] (TP32).

    Returns:
        [T, 32, 112] FP32 SBUF tile. Row r R + b, block t is the gather row of rank 32 r + t for its row b.
    """
    rows = gather_src.shape[0]
    tokens = DP_SIZE * rows
    rows_src, rows_dst, shard_dst = hbm["rows"]
    nisa.dma_copy(dst=rows_src, src=gather_src, priority=_CRITICAL_DMA_PRIORITY, name=f"{name}_rows_store")
    ncc.all_gather(
        srcs=[rows_src], dsts=[rows_dst], replica_group=dp_group, collective_dim=0, name=f"{name}_rows_gather"
    )
    ncc.all_gather(
        srcs=[rows_dst], dsts=[shard_dst], replica_group=tp_group, collective_dim=0, name=f"{name}_shard_gather"
    )
    gathered = nl.ndarray((tokens, TP_SIZE, _GATHER_COLS), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=gathered,
        src=shard_dst.ap(
            pattern=[[_GATHER_COLS, tokens], [tokens * _GATHER_COLS, TP_SIZE], [1, _GATHER_COLS]], offset=0
        ),
        name=f"{name}_shard_load",
        **_HBM_DMA,
    )
    return gathered


def _all_reduce_hbm(
    src: nl.NkiTensor, pair: tuple, group, name: str, after_trigger=None, record_consumer: bool = False
) -> nl.NkiTensor:
    """All-reduce (add) the SBUF tile ``src`` through the HBM ``pair`` (source, destination); return a new SBUF tile.

    The collective is the gate of the next body (``release_weight_gate``). ``after_trigger`` is None, or a function
    that is called with the all-reduce right after its trigger, before the load of its result: the caller traces work
    there that runs in the flight of the collective. ``record_consumer`` also records the load of the result for the
    ``after_result`` gates of the next body (``release_gate_consumer``; only with the block's consumer gates).
    """
    src_hbm, dst_hbm = pair
    nisa.dma_copy(dst=src_hbm, src=src, name=f"{name}_store", **_HBM_DMA)
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


def _validate_dp_inputs(x_hT, moe_in_proj_dp, moe_out_proj_dp, moe_vectors_dp):
    """Check the shapes and dtypes of the DP body inputs, and the LNC1 Trn3 target."""
    kernel_assert(
        nl.program_ndim() == 0 or (nl.program_ndim() == 1 and nl.num_programs(0) == 1),
        "The Kimi-K3 DP MoE decode body supports LNC1 only",
    )
    kernel_assert(nisa.get_nc_version() >= nisa.nc_version.gen4, "The Kimi-K3 DP MoE decode body requires Trn3")
    kernel_assert(
        len(x_hT.shape) == 3 and x_hT.shape[0] == _PMAX and x_hT.shape[2] == _H_TILES and x_hT.dtype == nl.bfloat16,
        f"x_hT must be BF16 [{_PMAX}, R, {_H_TILES}], got {x_hT.dtype} {x_hT.shape}",
    )
    kernel_assert(1 <= x_hT.shape[1] <= MAX_DP_ROWS, f"x_hT has R={x_hT.shape[1]} rows. Use 1 <= R <= {MAX_DP_ROWS}.")
    kernel_assert(
        tuple(moe_in_proj_dp.shape) == (_PMAX, _H_TILES, DP_IN_PROJ_COLS) and moe_in_proj_dp.dtype == nl.bfloat16,
        f"moe_in_proj_dp must be BF16 [{_PMAX}, {_H_TILES}, {DP_IN_PROJ_COLS}], got "
        f"{moe_in_proj_dp.dtype} {moe_in_proj_dp.shape}",
    )
    kernel_assert(
        tuple(moe_out_proj_dp.shape) == (DP_OUT_PROJ_ROWS, HIDDEN_SIZE) and moe_out_proj_dp.dtype == nl.bfloat16,
        f"moe_out_proj_dp must be BF16 [{DP_OUT_PROJ_ROWS}, {HIDDEN_SIZE}], got "
        f"{moe_out_proj_dp.dtype} {moe_out_proj_dp.shape}",
    )
    kernel_assert(
        tuple(moe_vectors_dp.shape) == (1, DP_VECTOR_COLS) and moe_vectors_dp.dtype == nl.float32,
        f"moe_vectors_dp must be FP32 [1, {DP_VECTOR_COLS}], got {moe_vectors_dp.dtype} {moe_vectors_dp.shape}",
    )
    kernel_assert(EXPERT_INTERMEDIATE % 512 == 0, "Unexpected expert intermediate size")


def _psum_rows(partitions: int, cols: int, dtype=nl.float32) -> nl.NkiTensor:
    """Return a [partitions, cols] view at offset 0 of one whole PSUM bank, with no reshape (valid for 1 partition)."""
    bank = nl.ndarray((partitions, _PSUM_BANK_BYTES // (4 if dtype == nl.float32 else 2)), dtype=dtype, buffer=nl.psum)
    return bank[:, 0:cols]


def _out_k_tiles(rows: int) -> tuple[tuple[int, int], ...]:
    """Split the contraction rows of an out-projection into partition tiles of at most 128 rows."""
    return tuple((start, min(start + _PMAX, rows)) for start in range(0, rows, _PMAX))


def _load_out_proj(moe_out_proj_dp: nl.NkiTensor, name: str) -> nl.NkiTensor:
    """Load the [K, H] out-projection as [128, n_k, H] contraction tiles, in chunks along H so the matmuls start early."""
    k_tiles = _out_k_tiles(moe_out_proj_dp.shape[0])
    out_proj_sb = nl.ndarray((_PMAX, len(k_tiles), HIDDEN_SIZE), dtype=nl.bfloat16, buffer=nl.sbuf)
    chunk = HIDDEN_SIZE // _OUT_PROJ_DMA_CHUNKS
    for chunk_idx in range(_OUT_PROJ_DMA_CHUNKS):
        cols = slice(chunk_idx * chunk, (chunk_idx + 1) * chunk)
        for k_idx, (start, stop) in enumerate(k_tiles):
            nisa.dma_copy(
                dst=out_proj_sb[0 : stop - start, k_idx, cols],
                src=moe_out_proj_dp[start:stop, cols],
                priority=_CRITICAL_DMA_PRIORITY,
                name=f"{name}_out_proj_c{chunk_idx}_k{k_idx}",
            )
    return out_proj_sb


def _dp_selectors(rank_info: nl.NkiTensor, tokens: int, name: str) -> tuple[nl.NkiTensor, nl.NkiTensor]:
    """Build the latent-shard one-hot and the local-expert one-hot of this rank.

    Returns:
        ``shard_select`` BF16 [128, 4]: element (p, q) is 1 when p == 4 t + q (the routed-layout partitions of
        latent columns [112 t, 112 t + 112)). ``expert_select`` FP32 [T, 128]: element (i, g') is 1 when g' == g.
    """
    ranks = nl.ndarray((_PMAX, 4), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=ranks,
        src=rank_info.ap(pattern=[[0, _PMAX], [1, 4]], offset=0),
        priority=_CRITICAL_DMA_PRIORITY,
        name=f"{name}_dp_ranks",
    )
    ranks_f32 = nl.ndarray((_PMAX, 4), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=ranks_f32, src=ranks)
    shard_first = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=shard_first, data=ranks_f32[:, 2:3], op0=nl.multiply, operand0=float(_SHARD_PARTITIONS))
    # p - q on partition p, column q: 1 where it equals 4 t.
    offsets = nl.ndarray((_PMAX, _SHARD_PARTITIONS), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(dst=offsets, pattern=[[-1, _SHARD_PARTITIONS]], offset=0, channel_multiplier=1)
    shard_select = nl.ndarray((_PMAX, _SHARD_PARTITIONS), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=shard_select, data=offsets, op0=nl.equal, operand0=shard_first)
    expert_groups = nl.ndarray((tokens, WORLD_SIZE), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(dst=expert_groups, pattern=[[1, WORLD_SIZE]], offset=0, channel_multiplier=0)
    expert_select = nl.ndarray((tokens, WORLD_SIZE), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=expert_select, data=expert_groups, op0=nl.equal, operand0=ranks_f32[0:tokens, 0:1])
    return shard_select, expert_select


def _dp_local_affinities(scores: nl.NkiTensor, selection: nl.NkiTensor, expert_select: nl.NkiTensor) -> nl.NkiTensor:
    """Compute this rank's FP32 [T, 7] top-16 affinities from the gathered [T, 32, 28] scores and selection values.

    The threshold, the selected scores and their sum run over the 896 experts in expert order, as in
    ``latent_moe_decode``. The affinities of experts [7 g, 7 g + 7) are the selected scores of the 128 groups
    of 7, weighted by the one-hot of g and summed over the groups.
    """
    tokens = scores.shape[0]
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

    by_group = selected_scores.reshape((tokens, WORLD_SIZE, LOCAL_EXPERTS)).permute((0, 2, 1))
    weighted = nl.ndarray((tokens, LOCAL_EXPERTS, WORLD_SIZE), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=weighted,
        data1=by_group,
        data2=expert_select.ap(pattern=[[WORLD_SIZE, tokens], [0, LOCAL_EXPERTS], [1, WORLD_SIZE]], offset=0),
        op=nl.multiply,
    )
    local_selected = nl.ndarray((tokens, LOCAL_EXPERTS), dtype=nl.float32, buffer=nl.sbuf)
    select = nisa.tensor_reduce(dst=local_selected, op=nl.add, data=weighted, axis=(2,))
    affinities = nl.ndarray((tokens, LOCAL_EXPERTS), dtype=nl.float32, buffer=nl.sbuf)
    scale = nisa.tensor_scalar(dst=affinities, data=local_selected, op0=nl.multiply, operand0=inv_score_sum)
    # As in ``latent_moe_decode``: a runtime dependency keeps the scale after the select on the Vector engine.
    if not isinstance(select, nisa.NkiInstruction):
        select = nisa.NkiInstruction(select)
    scale.depends_on(select, kind=nisa.dependency.runtime)
    return affinities


def _dp_latent_norm(
    routed: nl.NkiTensor,
    row_select: nl.NkiTensor,
    shard_select: nl.NkiTensor,
    moe_vectors_dp: nl.NkiTensor,
    rows: int,
    name: str,
) -> nl.NkiTensor:
    """Apply the latent RMSNorm to this replica's rows and return this rank's normed shard, token-major.

    ``normed = bf16(bf16(r32 * rsqrt(sum(r^2) / 3584 + 1e-5)) * w32)``, as in ``latent_moe_decode``.

    Args:
        routed: [128, T * 28] BF16 SBUF, the full routed latent: element (p, 28 i + c) is latent column 28 p + c of
            token i.
        row_select: FP32 [128, 4] replica one-hot (``rank_selectors``).
        shard_select: BF16 [128, 4] one-hot of the partitions 4 t .. 4 t + 3.

    Returns:
        [R, 112] BF16 SBUF: element (b, 28 q + c) is normed latent column 112 t + 28 q + c of row b of this replica.
    """
    width = rows * LATENT_SHARD
    routed_by_replica = routed.reshape((WORLD_SIZE, DP_SIZE, width))
    # The rows of this replica: a copy of replica 0, then a predicated copy of each other replica (no 0 x NaN).
    replica_rows = nl.ndarray((WORLD_SIZE, width), dtype=nl.bfloat16, buffer=nl.sbuf)
    is_replica = nl.ndarray((WORLD_SIZE, DP_SIZE), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=is_replica, data=row_select, op0=nl.not_equal, operand0=0.0)
    nisa.tensor_copy(dst=replica_rows, src=routed_by_replica[:, 0, :])
    for replica_idx in range(1, DP_SIZE):
        nisa.tensor_copy_predicated(
            dst=replica_rows,
            src=routed_by_replica[:, replica_idx, :],
            predicate=is_replica.ap(pattern=[[DP_SIZE, WORLD_SIZE], [0, width]], offset=replica_idx),
        )

    rows_by_token = replica_rows.reshape((WORLD_SIZE, rows, LATENT_SHARD))
    routed_sq = nl.ndarray((WORLD_SIZE, rows, LATENT_SHARD), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=routed_sq, data1=rows_by_token, data2=rows_by_token, op=nl.multiply)
    partial_sums = nl.ndarray((WORLD_SIZE, rows), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=partial_sums, op=nl.add, data=routed_sq, axis=(2,))
    ones = nl.ndarray((WORLD_SIZE, LATENT_SHARD), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones, value=1.0)
    total_psum = _psum_bank(LATENT_SHARD, (rows,), nl.float32)
    nisa.nc_matmul(dst=total_psum, stationary=ones, moving=partial_sums, is_stationary_onezero=True)
    inv_rms = nl.ndarray((LATENT_SHARD, rows), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=inv_rms, op=nl.rsqrt, data=total_psum, scale=1.0 / LATENT_SIZE, bias=LATENT_NORM_EPS)

    # Shard of this rank: element (c, b, q) = routed[4 t + q, 28 b + c] (exact one-hot).
    shard_psum = _psum_bank(LATENT_SHARD, (rows, _SHARD_PARTITIONS), nl.float32)
    for row_idx in range(rows):
        nisa.nc_matmul(
            dst=shard_psum[:, row_idx, :],
            stationary=replica_rows[:, row_idx * LATENT_SHARD : (row_idx + 1) * LATENT_SHARD],
            moving=shard_select,
        )
    norm_weight = nl.ndarray((LATENT_SHARD, _SHARD_PARTITIONS), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=norm_weight,
        src=moe_vectors_dp.ap(pattern=[[1, LATENT_SHARD], [LATENT_SHARD, _SHARD_PARTITIONS]], offset=DP_NORM_COLS[0]),
        priority=_CRITICAL_DMA_PRIORITY,
        name=f"{name}_weight",
    )
    scaled = nl.ndarray((LATENT_SHARD, rows, _SHARD_PARTITIONS), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=scaled,
        data1=shard_psum,
        data2=inv_rms.ap(pattern=[[rows, LATENT_SHARD], [1, rows], [0, _SHARD_PARTITIONS]], offset=0),
        op=nl.multiply,
    )
    # The weight multiply of each block q is the instruction of ``latent_moe_decode``, so the values are the same.
    normed = nl.ndarray((LATENT_SHARD, rows, _SHARD_PARTITIONS), dtype=nl.bfloat16, buffer=nl.sbuf)
    for block_idx in range(_SHARD_PARTITIONS):
        nisa.tensor_scalar(
            dst=normed[:, :, block_idx],
            data=scaled[:, :, block_idx],
            op0=nl.multiply,
            operand0=norm_weight[:, block_idx : block_idx + 1],
        )

    # Token-major [R, 112]: one PE transpose of each partition block q.
    normed_psum = _psum_rows(rows, DP_LATENT, nl.bfloat16)
    for block_idx in range(_SHARD_PARTITIONS):
        nisa.nc_transpose(
            dst=normed_psum[:, block_idx * LATENT_SHARD : (block_idx + 1) * LATENT_SHARD],
            data=normed[:, :, block_idx],
        )
    normed_tok = nl.ndarray((rows, DP_LATENT), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=normed_tok, src=normed_psum)
    return normed_tok


def _transpose_rows(act_tok: nl.NkiTensor, k_tiles) -> nl.NkiTensor:
    """Transpose a token-major [R, K] BF16 tile into [128, n_k, R] contraction tiles (one PE transpose each)."""
    rows = act_tok.shape[0]
    act_T = nl.ndarray((_PMAX, len(k_tiles), rows), dtype=nl.bfloat16, buffer=nl.sbuf)
    for k_idx, (start, stop) in enumerate(k_tiles):
        psum = _psum_bank(stop - start, (rows,), nl.bfloat16)
        nisa.nc_transpose(dst=psum, data=act_tok[:, start:stop])
        nisa.tensor_copy(dst=act_T[0 : stop - start, k_idx, :], src=psum)
    return act_T
