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

"""Kimi-K3 decode latent MoE body with all-to-all-v routed experts (128 ranks, LNC1).

The dense parts are those of ``latent_moe_decode_dp``: the router, the latent down projection and the shared experts
are TP32 inside each DP replica, and the output projection ends with a TP32 all-reduce. The routed experts stay
EP128, but each rank runs its 7 experts only on the tokens routed to them:

1. A TP32 all-to-all gives each rank the full latent row and the 896 router values of its owner rows: rank 32 r + t
   owns rows [t n, t n + n) of replica r, with n = R / 32.
2. The owner routes its rows (top-16) and sends one dispatch row per (token, destination rank): the BF16 latent, the
   7 FP32 affinities of that rank's experts and a token id. One ``all_to_all_v`` moves the rows; source s writes its
   rows at slot s of the receive buffer (n rows per slot).
3. Each rank packs the rows that it received and runs ``moe_tkg_decode`` on them, then sends each partial row back
   to its owner with a second ``all_to_all_v`` (the receive counts of the first one are its send counts).
4. The owner adds the partials of each row in FP32 (one BF16 rounding), applies the latent RMSNorm scale, and a TP32
   all-to-all returns the 112 latent columns of each TP rank for all R rows. The output projection follows.

Up to the BF16 partial of each (token, rank), the values are the ones of ``latent_moe_decode_dp``. The sum over the
ranks is an FP32 sum at the owner instead of the BF16 world all-reduce. Every exchange of rows is a DMA, so a NaN in
one token reaches no other token.
"""

import nki.collectives as ncc
import nki.isa as nisa
import nki.language as nl
from nki.collectives import ReplicaGroup

from .utils.kernel_assert import kernel_assert
from .moe.routed import moe_tkg_decode, moe_tkg_decode_blocks, moe_tkg_decode_ragged
from .tkg_weight_dma_gate import release_weight_gate
from .tkg_moe import (
    _CRITICAL_DMA_PRIORITY,
    _H_TILES,
    _IN_PROJ_DMA_CHUNKS,
    _PMAX,
    _SELECTION_FLOOR,
    LATENT_NORM_EPS,
    LATENT_SIZE,
    LOCAL_EXPERTS,
    NUM_EXPERTS,
    ROUTER_SUM_EPS,
    SITU_BETA,
    SITU_LINEAR_BETA,
    TOP_K,
    WORLD_SIZE,
    _load_hT_rows,
    _project_to_hT,
    _psum_bank,
    _situ,
)
from .tkg_moe_dp import (
    _GATHER_COLS,
    _GATHER_LATENT_WORDS,
    _GATHER_SCORE_COLS,
    _GATHER_SELECTION_COLS,
    DP_BIAS_COLS,
    DP_EXPERTS,
    DP_IN_PROJ_LATENT_COLS,
    DP_LATENT,
    DP_LATENT_COLS,
    DP_LATENT_UP_ROWS,
    DP_NORM_COLS,
    DP_OUT_PROJ_ROWS,
    DP_ROUTER_COLS,
    DP_SHARED,
    DP_SHARED_DOWN_ROWS,
    DP_SHARED_GATE_COLS,
    DP_SHARED_UP_COLS,
    TP_SIZE,
    _load_out_proj,
    _out_k_tiles,
    _validate_dp_inputs,
)

# Dispatch row, in UINT16 elements: the BF16 latent, the 7 FP32 affinities of the destination rank, an INT32 id, and
# padding to whole 64-byte blocks, so that no two rows (or a latent and a tail) share a block that a DMA rewrites.
A2AV_AFFINITY_COL = LATENT_SIZE
A2AV_ID_COL = A2AV_AFFINITY_COL + 2 * LOCAL_EXPERTS
_BLOCK_U16 = 32
A2AV_ROW = -(-(A2AV_ID_COL + 2) // _BLOCK_U16) * _BLOCK_U16
_TAIL = A2AV_ROW - LATENT_SIZE
_TAIL_USED = A2AV_ID_COL + 2 - LATENT_SIZE
# Combine row: the BF16 partial of one (token, rank).
COMBINE_ROW = LATENT_SIZE
# Rows of each replica in one call: 32 to 256 (n = 1 to 8 owner rows), so that the 16 destinations of the owner rows
# fit the 128 partitions of the permute.
A2AV_MIN_ROWS = TP_SIZE
A2AV_MAX_ROWS = 8 * TP_SIZE
_EXPERT_CALL_ROWS = _PMAX
# Rows of each block of the blocks expert form.
_BLOCK_ROWS = 32
_LATENT_TILES = LATENT_SIZE // _PMAX
# The keys of ``a2av_moe_hbm`` carry this prefix, so they join the collective sources of a decode kernel.
_KEY_PREFIX = "a2av_"
# The smallest per-peer block of an all-to-all-v on 128 ranks: one transfer tile.
_A2AV_PEER_TILE_BYTES = 72817
_TRANSPOSES_PER_BANK = 8
A2AV_DEFAULT_CAPACITY = {rows: WORLD_SIZE * rows // TP_SIZE for rows in range(32, 257, 32)}
"""Received rows each rank packs for the experts (``recv_capacity``), by rows per replica: 128 n, the most that a rank
can receive (every owner row of every source), so no row is dropped whatever the routing. A smaller capacity drops the
received rows past it (their partials are wrong). The busiest rank under uniform top-16 routing gets about 25 / 44 / 80
/ 148 rows at T = 128 / 256 / 512 / 1,024 tokens per server; with one prompt on every row it gets T."""
A2AV_EXPERT_FORMS = ("dense", "ragged", "blocks")


def a2av_moe_hbm(replica_rows: int, recv_capacity: int = None, name: str = "", expert_form: str = "dense") -> dict:
    """Return the persistent HBM tensors of the a2a-v MoE body for ``replica_rows`` rows of each replica.

    Each layer uses the same tensors. A layer's combine is a later world collective than its dispatch, and the next
    layer's TP32 all-to-all follows its output all-reduce, so each use of a byte follows the last access of the
    previous use.
    """
    replica_rows = _padded_rows(replica_rows)
    _check_rows(replica_rows)
    recv_capacity = recv_capacity or A2AV_DEFAULT_CAPACITY[replica_rows]
    owner_rows = replica_rows // TP_SIZE
    slots = WORLD_SIZE * owner_rows
    width = replica_rows * _H_TILES
    specs = {
        "tp_in_src": ((replica_rows, _GATHER_COLS), nl.uint32),
        "tp_in_dst": ((replica_rows, _GATHER_COLS), nl.uint32),
        "owner_latent": ((owner_rows, LATENT_SIZE), nl.uint16),
        "owner_tail": ((owner_rows * WORLD_SIZE, _TAIL), nl.uint16),
        "dispatch_meta": ((3, WORLD_SIZE), nl.uint32),
        "dispatch_send": ((TOP_K * owner_rows, A2AV_ROW), nl.uint16),
        "dispatch_recv": ((slots, A2AV_ROW), nl.uint16),
        "combine_meta": ((3, WORLD_SIZE), nl.uint32),
        "combine_send": ((recv_capacity, COMBINE_ROW), nl.bfloat16),
        "combine_recv": ((slots, COMBINE_ROW), nl.bfloat16),
        "recv_rows": ((2, WORLD_SIZE), nl.float32),
        "tp_out_src": ((replica_rows, DP_LATENT), nl.uint16),
        "tp_out_dst": ((replica_rows, DP_LATENT), nl.uint16),
    }
    if expert_form in ("ragged", "blocks"):
        specs["packed_latent"] = ((recv_capacity, LATENT_SIZE), nl.uint16)
        specs["packed_affinity"] = ((recv_capacity, LOCAL_EXPERTS), nl.float32)
        specs["accumulator"] = ((recv_capacity, LATENT_SIZE), nl.float32)
    if expert_form == "blocks":
        specs["row_table"] = ((LOCAL_EXPERTS * recv_capacity, 1), nl.int32)
    tensors = {
        key: nl.ndarray(shape, dtype=dtype, buffer=nl.shared_hbm, name=f"a2av{name}_{key}")
        for key, (shape, dtype) in specs.items()
    }
    for key in ("out_src", "out_dst"):
        tensors[key] = nl.ndarray((_PMAX, width), dtype=nl.bfloat16, buffer=nl.private_hbm, name=f"a2av{name}_{key}")
    return {f"{_KEY_PREFIX}{key}": tensor for key, tensor in tensors.items()}


def latent_moe_decode_a2av(
    x_hT: nl.NkiTensor,
    moe_in_proj_dp: nl.NkiTensor,
    moe_out_proj_dp: nl.NkiTensor,
    moe_vectors_dp: nl.NkiTensor,
    expert_gate_up_weights: nl.NkiTensor,
    expert_gate_up_weights_scale: nl.NkiTensor,
    expert_down_weights: nl.NkiTensor,
    expert_down_weights_scale: nl.NkiTensor,
    rank_info: nl.NkiTensor,
    world_group: ReplicaGroup,
    tp_group: ReplicaGroup,
    tag: str,
    sources: dict,
    weight_buffers: int = 3,
    scale_buffers: int = 1,
    row_valid: nl.NkiTensor = None,
    expert_form: str = "dense",
    expert_rows: int = None,
    debug: dict = None,
) -> nl.NkiTensor:
    """Run one Kimi-K3 latent MoE decode layer with all-to-all-v routed experts and return this replica's rows.

    Not ``@nki.jit``: a decode kernel inlines one call for each MoE layer. The attribute ``token_major`` is True.

    Dimensions:
        H: Hidden size, 7168.
        R: Rows of each DP replica, 32, 64, 128 or 256 (T = 4 R = 128 to 1,024 tokens of the server).
        n: Owner rows of each rank, R / 32.
        C: Received rows that each rank packs for its experts (``sources["combine_send"]`` rows).

    Args:
        x_hT (nl.NkiTensor): [128, R, 56] BF16 SBUF, this replica's normalized hidden, token-major hT.
        moe_in_proj_dp .. moe_vectors_dp (nl.NkiTensor): the DP4 x TP32 projections of ``latent_moe_decode_dp``.
        expert_gate_up_weights .. expert_down_weights_scale (nl.NkiTensor): this rank's 7 MXFP4 experts in the
            serving layout of ``moe_tkg_decode``.
        rank_info (nl.NkiTensor): [1, 4] INT32 HBM, (global rank, DP rank, TP rank, 0).
        world_group (ReplicaGroup): [[0 .. 127]].
        tp_group (ReplicaGroup): [[32 r .. 32 r + 31] for r in 0..3].
        tag (str): Unique per-layer string. It names the collectives and the DMAs.
        sources (dict): the persistent HBM tensors (``a2av_moe_hbm``), allocated once for the kernel.
        weight_buffers (int): SBUF weight buffers of the expert loop, 1 to 3.
        scale_buffers (int): SBUF gate/up scale buffers of the expert loop, 2 or 1.
        row_valid (nl.NkiTensor): None (every row is valid), or [R, 1] FP32 HBM, 1.0 for a valid row and 0.0 for an
            empty or padding row. R below 32 runs as 32 rows: the body pads the rows with invalid zero rows. An invalid row gets zero router scores, so it has no destination: it sends no
            dispatch row, takes no expert row on any rank, and its routed latent is 0.
        expert_form (str): ``dense`` runs ``moe_tkg_decode`` (every local expert) on tiles of 128 packed rows; one
            weight stream for each tile. ``ragged`` runs ``moe_tkg_decode_ragged``: each local expert on its own rows,
            one weight stream for all C rows (``sources`` from ``a2av_moe_hbm(..., expert_form="ragged")``).
            ``blocks`` runs ``moe_tkg_decode_blocks``: each local expert on its own rows in blocks of ``expert_rows``
            (32 by default), the blocks past the first in a dynamic loop, so no row is dropped (``sources`` from
            ``a2av_moe_hbm(..., expert_form="blocks")``).
        expert_rows (int): ``ragged``: the rows of each expert tile (the capacity C <= 128 by default, so no
            expert drops a row). ``blocks``: the rows of each block.
        debug (dict): test only. None, or HBM tensors that receive intermediates: ``affinity`` [n, 896] FP32 (owner
            rows), ``send_counts`` [1, 128] FP32, ``recv_rows`` [1, 128] FP32, ``partials`` [C, 3584] BF16 (the
            packed expert outputs), ``routed`` [n, 3584] BF16 (the routed latent of the owner rows), ``expert_counts``
            [7, 1] INT32 (``ragged``, ``blocks``: the rows of each local expert).
            ``partials`` runs the static loop over all C rows.

    Returns:
        out_hT (nl.NkiTensor): [128, R, 56] BF16 SBUF, this replica's MoE output in the token-major hT layout.

    Notes:
        - Collectives: all-to-all (TP32, router rows), all-to-all-v (world, dispatch), all-to-all-v (world, combine),
          all-to-all (TP32, normed latent), all-reduce (TP32, output).
        - A rank that receives more than C rows drops the rows above C (the test checks ``recv_rows``).
        - A tie at the top-16 threshold can select more than 16 experts. The kernel then sends each row to its first
          16 destination ranks only.

    Pseudocode:
        r, latent_t, shared_g, shared_u = x_r @ moe_in_proj_dp            # [R, 524]
        own = all_to_all_tp([bf16(latent_t), sigmoid(r), sigmoid(r) + bias])[owner rows]   # [n, 32, 112]
        affinity = top-16 affinities of the 896 experts of each owner row
        recv = all_to_all_v_world(one row per (owner row, destination rank))
        partial = moe_tkg_decode(packed recv rows)                          # [C, 3584]
        routed = fp32_sum(all_to_all_v_world(partial))                      # [n, 3584]
        scaled = bf16(routed * rsqrt(mean(routed^2) + 1e-5))
        normed = bf16(all_to_all_tp(scaled)[:, 112 t : 112 t + 112] * w)     # [R, 112]
        y = bf16([normed | situ(shared_g, shared_u)] @ moe_out_proj_dp)     # [R, H]
        out = all_reduce_tp(y)
    """
    _validate_dp_inputs_a2av(x_hT, moe_in_proj_dp, moe_out_proj_dp, moe_vectors_dp)
    kernel_assert(
        expert_form in A2AV_EXPERT_FORMS, f"expert_form must be one of {A2AV_EXPERT_FORMS}, got {expert_form}"
    )
    rows_in = x_hT.shape[1]
    rows = _padded_rows(rows_in)
    owner_rows = rows // TP_SIZE
    sources = {key[len(_KEY_PREFIX) :]: value for key, value in sources.items() if key.startswith(_KEY_PREFIX)}
    kernel_assert(
        expert_form == "dense" or "accumulator" in sources,
        f"The {expert_form} expert form needs the sources of a2av_moe_hbm(..., expert_form='{expert_form}')",
    )
    capacity = sources["combine_send"].shape[0]
    _check_sources(sources, rows, capacity)
    kernel_assert(
        expert_form != "ragged" or capacity <= _PMAX,
        f"The ragged form takes at most 128 received rows per rank (one expert tile), got capacity {capacity}",
    )
    name = f"k3moea2av_{tag}"
    consts = _constants(rank_info, name)
    if rows_in < rows:
        x_hT, row_valid = _pad_rows(x_hT, row_valid, rows, consts, name)

    in_proj_sb = nl.ndarray((_PMAX, _H_TILES, moe_in_proj_dp.shape[2]), dtype=nl.bfloat16, buffer=nl.sbuf)
    _load_hT_rows(in_proj_sb, moe_in_proj_dp, _IN_PROJ_DMA_CHUNKS, f"{name}_in_proj")
    act_tok = _in_projection(x_hT, in_proj_sb, moe_vectors_dp, sources["tp_in_src"], row_valid, name)
    ncc.all_to_all(
        srcs=[sources["tp_in_src"]],
        dsts=[sources["tp_in_dst"]],
        replica_group=tp_group,
        collective_dim=0,
        name=f"{name}_tp_in",
    )

    own = nl.ndarray((owner_rows, TP_SIZE, _GATHER_COLS), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=own.view(nl.uint32),
        src=sources["tp_in_dst"].ap(
            pattern=[[_GATHER_COLS, owner_rows], [owner_rows * _GATHER_COLS, TP_SIZE], [1, _GATHER_COLS]], offset=0
        ),
        priority=_CRITICAL_DMA_PRIORITY,
        name=f"{name}_own_load",
    )
    affinity, hit = _owner_routing(own, name)
    _store_owner_tables(own, affinity, rank_info, sources, owner_rows, rows, name)
    plan = _dispatch_plan(hit, consts, owner_rows, name)
    _write_meta(sources["dispatch_meta"], plan["send_counts"], plan["send_displs"], None, A2AV_ROW, name, "dispatch")
    _permute(plan, sources, owner_rows, name)
    ncc.all_to_all_v(
        srcs=[sources["dispatch_send"]],
        dsts=[sources["dispatch_recv"]],
        replica_group=world_group,
        metadata_tensor=sources["dispatch_meta"],
        recv_counts_known=False,
        has_rdispls=False,
        name=f"{name}_dispatch",
    )

    recv_rows, recv_displs = _received_rows(sources, consts, owner_rows, name)
    _write_meta(sources["combine_meta"], recv_rows, recv_displs, plan["send_counts"], COMBINE_ROW, name, "combine")
    slots = _packed_slots(sources, consts, owner_rows, capacity, name)
    if expert_gate_up_weights.dtype == nl.uint16:
        expert_gate_up_weights = expert_gate_up_weights.view(nl.float4_e2m1fn_x4)
    if expert_down_weights.dtype == nl.uint16:
        expert_down_weights = expert_down_weights.view(nl.float4_e2m1fn_x4)
    experts = (expert_gate_up_weights, expert_down_weights, expert_gate_up_weights_scale, expert_down_weights_scale)
    if expert_form == "ragged":
        _pack_received_hbm(sources, slots, capacity, name)
        moe_tkg_decode_ragged(
            sources["packed_latent"],
            sources["packed_affinity"],
            *experts,
            sources["accumulator"],
            sources["combine_send"],
            expert_rows or capacity,
            SITU_BETA,
            SITU_LINEAR_BETA,
            weight_buffers,
            scale_buffers,
            debug.get("expert_counts") if debug else None,
        )
    if expert_form == "blocks":
        _pack_received_hbm(sources, slots, capacity, name)
        moe_tkg_decode_blocks(
            sources["packed_latent"],
            sources["packed_affinity"],
            *experts,
            sources["accumulator"],
            sources["combine_send"],
            sources["row_table"],
            block_rows=expert_rows or _BLOCK_ROWS,
            beta=SITU_BETA,
            linear_beta=SITU_LINEAR_BETA,
            scale_buffers=scale_buffers,
            expert_counts=debug.get("expert_counts") if debug else None,
            name=f"{name}_blocks",
            weight_buffers=max(weight_buffers, 2),
        )
    # Dense form: pack, run and store one tile of 128 received rows at a time.
    for tile_idx, slot_idx in enumerate(slots if expert_form == "dense" else []):
        first = tile_idx * _EXPERT_CALL_ROWS
        size = min(_EXPERT_CALL_ROWS, capacity - first)
        latent_sb, affinity_sb = _pack_tile(sources, slot_idx, first, size, name)
        moe_tkg_decode(
            latent_sb,
            affinity_sb,
            expert_gate_up_weights,
            expert_down_weights,
            expert_gate_up_weights_scale,
            expert_down_weights_scale,
            SITU_BETA,
            SITU_LINEAR_BETA,
            weight_buffers,
            scale_buffers,
            True,
            False,
            0,
            0,
            None,
            output=sources["combine_send"][first : first + size, :],
        )
    ncc.all_to_all_v(
        srcs=[sources["combine_send"]],
        dsts=[sources["combine_recv"]],
        replica_group=world_group,
        metadata_tensor=sources["combine_meta"],
        recv_counts_known=True,
        has_rdispls=False,
        name=f"{name}_combine",
    )

    scaled_rows = _owner_reduce_and_scale(sources, plan, consts, owner_rows, name, debug)
    nisa.dma_copy(
        dst=sources["tp_out_src"].ap(
            pattern=[[DP_LATENT, owner_rows], [owner_rows * DP_LATENT, TP_SIZE], [1, DP_LATENT]], offset=0
        ),
        src=scaled_rows.ap(
            pattern=[[LATENT_SIZE, owner_rows], [DP_LATENT, TP_SIZE], [1, DP_LATENT]], offset=0, dtype=nl.uint16
        ),
        priority=_CRITICAL_DMA_PRIORITY,
        name=f"{name}_tp_out_store",
    )
    ncc.all_to_all(
        srcs=[sources["tp_out_src"]],
        dsts=[sources["tp_out_dst"]],
        replica_group=tp_group,
        collective_dim=0,
        name=f"{name}_tp_out",
    )
    out_proj_sb = _load_out_proj(moe_out_proj_dp, name)
    _normed_rows(sources["tp_out_dst"], moe_vectors_dp, act_tok, name)
    out = _out_projection(act_tok, out_proj_sb, rows, sources, tp_group, name)
    if debug:
        _store_debug(debug, affinity, plan, sources, capacity, name)
    out = out.reshape((_PMAX, rows, _H_TILES))
    if rows_in < rows:
        out_in = nl.ndarray((_PMAX, rows_in, _H_TILES), dtype=out.dtype, buffer=nl.sbuf)
        nisa.tensor_copy(dst=out_in, src=out[:, 0:rows_in, :])
        out = out_in
    return out


latent_moe_decode_a2av.token_major = True


ROUTED_EXCHANGES = ("all_reduce", "a2av", "a2av_ragged", "a2av_blocks")
"""Routed exchanges of the decode entries (``moe_routed_exchange``): the world all-reduce of the routed partials
(default), or all-to-all-v with the dense (``a2av``), the ragged (``a2av_ragged``) or the blocks (``a2av_blocks``)
expert form."""


def a2av_moe_body(tp_group: ReplicaGroup, routed_exchange: str = "a2av"):
    """Return a body with the call form of ``latent_moe_decode`` in ``decode_block_body`` that runs this body.

    The decode block calls ``moe_body(x_hT, *weights, rank_info, dp_group, world_group, tag, sources=...,
    scale_buffers=..., weight_buffers=...)``. The attributes ``moe_sharding`` (``dp4xtp32``: the DP weights and
    tiles) and ``a2av_expert_form`` (the persistent tensors of ``a2av_moe_hbm``) tell ``collective_sources`` what to
    add.
    """
    kernel_assert(
        routed_exchange in ROUTED_EXCHANGES[1:],
        f"routed_exchange must be one of {ROUTED_EXCHANGES[1:]}, got {routed_exchange}",
    )
    expert_form = routed_exchange[len("a2av_") :] if routed_exchange.startswith("a2av_") else "dense"

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
    ):
        return latent_moe_decode_a2av(
            x_hT,
            moe_in_proj_dp,
            moe_out_proj_dp,
            moe_vectors_dp,
            expert_gate_up_weights,
            expert_gate_up_weights_scale,
            expert_down_weights,
            expert_down_weights_scale,
            rank_info,
            world_group,
            tp_group,
            tag,
            sources,
            weight_buffers=weight_buffers,
            scale_buffers=min(scale_buffers, 1),
            expert_form=expert_form,
        )

    body.token_major = True
    body.moe_sharding = "dp4xtp32"
    body.a2av_expert_form = expert_form
    return body


def _padded_rows(rows: int) -> int:
    """Return the rows the body runs: R below 32 runs as 32 rows (one owner row per TP rank), the rest invalid."""
    kernel_assert(rows >= 1, f"The a2a-v MoE body needs at least one row, got R={rows}")
    return max(rows, A2AV_MIN_ROWS)


def _pad_rows(x_hT: nl.NkiTensor, row_valid, rows: int, consts: dict, name: str) -> tuple:
    """Pad [128, R, 56] to ``rows`` rows of zeros; return it and the SBUF [rows, 1] FP32 validity (the pad is 0)."""
    rows_in = x_hT.shape[1]
    padded = nl.ndarray((_PMAX, rows, x_hT.shape[2]), dtype=x_hT.dtype, buffer=nl.sbuf)
    nisa.memset(dst=padded, value=0.0)
    nisa.tensor_copy(dst=padded[:, 0:rows_in, :], src=x_hT)
    valid = nl.ndarray((rows, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=valid, data=consts["part"][0:rows, :], op0=nl.less, operand0=float(rows_in))
    if row_valid is not None:
        given = nl.ndarray((rows, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=given, value=0.0)
        nisa.dma_copy(dst=given[0:rows_in, :], src=row_valid, name=f"{name}_row_valid")
        nisa.tensor_tensor(dst=valid, data1=valid, data2=given, op=nl.multiply)
    return padded, valid


def _check_rows(rows: int):
    kernel_assert(
        rows % TP_SIZE == 0 and A2AV_MIN_ROWS <= rows <= A2AV_MAX_ROWS,
        f"The a2a-v MoE body takes R = 32, 64, ..., 256 rows of each replica, got R={rows}",
    )
    # A rank sends at most n rows to each peer (one per owner row), and the runtime drops the tail of a block
    # larger than its per-peer tile.
    peer_bytes = (rows // TP_SIZE) * max(A2AV_ROW, COMBINE_ROW) * 2
    kernel_assert(
        peer_bytes <= _A2AV_PEER_TILE_BYTES,
        f"The largest all-to-all-v block ({peer_bytes} B) is larger than the per-peer tile ({_A2AV_PEER_TILE_BYTES} B)",
    )


def _check_sources(sources: dict, rows: int, capacity: int):
    """Check the persistent HBM tensors against the call."""
    owner_rows = rows // TP_SIZE
    expected = {
        "tp_in_src": (rows, _GATHER_COLS),
        "tp_in_dst": (rows, _GATHER_COLS),
        "owner_latent": (owner_rows, LATENT_SIZE),
        "owner_tail": (owner_rows * WORLD_SIZE, _TAIL),
        "dispatch_send": (TOP_K * owner_rows, A2AV_ROW),
        "dispatch_recv": (WORLD_SIZE * owner_rows, A2AV_ROW),
        "combine_recv": (WORLD_SIZE * owner_rows, COMBINE_ROW),
        "tp_out_src": (rows, DP_LATENT),
        "tp_out_dst": (rows, DP_LATENT),
        "out_src": (_PMAX, rows * _H_TILES),
        "out_dst": (_PMAX, rows * _H_TILES),
    }
    if "accumulator" in sources:
        expected.update(
            packed_latent=(capacity, LATENT_SIZE),
            packed_affinity=(capacity, LOCAL_EXPERTS),
            accumulator=(capacity, LATENT_SIZE),
        )
    for key, shape in expected.items():
        kernel_assert(key in sources, f"The a2a-v MoE body needs the persistent HBM tensor {_KEY_PREFIX}{key}")
        kernel_assert(tuple(sources[key].shape) == shape, f"{key}: {sources[key].shape} is not {shape}")
    kernel_assert(
        capacity % 4 == 0 and 4 <= capacity <= WORLD_SIZE * owner_rows,
        f"combine_send rows (the packed capacity) must be a multiple of 4 up to {WORLD_SIZE * owner_rows}, got {capacity}",
    )


def _validate_dp_inputs_a2av(x_hT, moe_in_proj_dp, moe_out_proj_dp, moe_vectors_dp):
    """Check the inputs as ``latent_moe_decode_dp`` does, with the row range of this body."""
    rows = x_hT.shape[1]
    kernel_assert(
        rows < A2AV_MIN_ROWS or rows % TP_SIZE == 0,
        f"The a2a-v MoE body takes R < 32 (padded to 32) or R = 32, 64, ..., 256 rows, got R={rows}",
    )
    _check_rows(_padded_rows(rows))
    rows_view = x_hT[:, 0:1, :]
    _validate_dp_inputs(rows_view, moe_in_proj_dp, moe_out_proj_dp, moe_vectors_dp)


def _row_chunks(rows: int) -> list:
    """Split R rows into chunks of at most 128 token partitions."""
    return [(start, min(_PMAX, rows - start)) for start in range(0, rows, _PMAX)]


def _constants(rank_info: nl.NkiTensor, name: str) -> dict:
    """Return the constant tiles of the body, built once for each kernel.

    ``part`` [128, 1] holds the partition index, ``free`` [128, 128] the free index, ``lower`` [128, 128] is 1 where
    the partition index is below the free index (exclusive prefix sums over partitions), and ``ones`` [128, 128]
    is 1.
    """
    consts = getattr(rank_info, "_k3_a2av_constants", None)
    if consts is not None:
        return consts
    part = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(dst=part, pattern=[[0, 1]], offset=0, channel_multiplier=1)
    free = nl.ndarray((_PMAX, _PMAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(dst=free, pattern=[[1, _PMAX]], offset=0, channel_multiplier=0)
    lower = nl.ndarray((_PMAX, _PMAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=lower, data=free, op0=nl.greater, operand0=part)
    ones = nl.ndarray((_PMAX, _PMAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones, value=1.0)
    consts = {"part": part, "free": free, "lower": lower, "ones": ones}
    object.__setattr__(rank_info, "_k3_a2av_constants", consts)
    return consts


def _in_projection(
    x_hT: nl.NkiTensor,
    in_proj_sb: nl.NkiTensor,
    moe_vectors_dp: nl.NkiTensor,
    tp_in_src: nl.NkiTensor,
    row_valid: nl.NkiTensor,
    name: str,
) -> list:
    """Run the TP32 in-projection of the R rows; store the router rows to ``tp_in_src`` and return the out-projection
    input tiles.

    Each 128-row chunk stores its [rows, 112] router rows (56 words of BF16 latent pairs, 28 scores, 28 selection
    values), as ``latent_moe_decode_dp`` gathers them. The shared gate and up columns give the SiTU activation in
    columns 112..303 of the returned token-major [rows, 304] BF16 tiles; columns 0..111 get the normed latent later.
    """
    rows = x_hT.shape[1]
    x_rows = x_hT.permute((0, 2, 1))
    act_tok = []
    for start, size in _row_chunks(rows):
        bias_sb = nl.ndarray((size, DP_EXPERTS), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=bias_sb,
            src=moe_vectors_dp.ap(pattern=[[0, size], [1, DP_EXPERTS]], offset=DP_BIAS_COLS[0]),
            priority=_CRITICAL_DMA_PRIORITY,
            name=f"{name}_bias_{start}",
        )
        router_psum = _psum_bank(size, (DP_IN_PROJ_LATENT_COLS,), nl.float32)
        for h_tile_idx in range(_H_TILES):
            nisa.nc_matmul(
                dst=router_psum,
                stationary=x_rows[:, h_tile_idx, start : start + size],
                moving=in_proj_sb[:, h_tile_idx, 0:DP_IN_PROJ_LATENT_COLS],
            )
        gather_src = nl.ndarray((size, _GATHER_COLS), dtype=nl.float32, buffer=nl.sbuf)
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
        if row_valid is not None:
            _zero_invalid_scores(scores_src, row_valid[start : start + size, :], f"{name}_valid_{start}")
        nisa.dma_copy(
            dst=tp_in_src[start : start + size, :],
            src=gather_src.view(nl.uint32),
            priority=_CRITICAL_DMA_PRIORITY,
            name=f"{name}_tp_in_store_{start}",
        )

        shared_psum = _psum_bank(size, (2 * DP_SHARED,), nl.float32)
        for h_tile_idx in range(_H_TILES):
            nisa.nc_matmul(
                dst=shared_psum,
                stationary=x_rows[:, h_tile_idx, start : start + size],
                moving=in_proj_sb[:, h_tile_idx, DP_SHARED_GATE_COLS[0] : DP_SHARED_UP_COLS[1]],
            )
        shared_bf16 = nl.ndarray((size, 2 * DP_SHARED), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=shared_bf16, src=shared_psum)
        tile = nl.ndarray((size, DP_OUT_PROJ_ROWS), dtype=nl.bfloat16, buffer=nl.sbuf)
        _situ(
            shared_bf16[:, 0:DP_SHARED],
            shared_bf16[:, DP_SHARED : 2 * DP_SHARED],
            out=tile[:, DP_SHARED_DOWN_ROWS[0] : DP_SHARED_DOWN_ROWS[1]],
        )
        act_tok.append(tile)
    return act_tok


def _zero_invalid_scores(scores: nl.NkiTensor, row_valid: nl.NkiTensor, name: str):
    """Write 0 into the router scores of the invalid rows (a predicated copy: a NaN score of such a row goes too)."""
    size, cols = scores.shape
    valid = row_valid
    if row_valid.buffer != nl.sbuf:
        valid = nl.ndarray((size, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=valid, src=row_valid, priority=_CRITICAL_DMA_PRIORITY, name=name)
    invalid = nl.ndarray((size, 1), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=invalid, data=valid, op0=nl.equal, operand0=0.0)
    zeros = nl.ndarray((size, cols), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=zeros, value=0.0)
    nisa.tensor_copy_predicated(dst=scores, src=zeros, predicate=invalid.ap(pattern=[[1, size], [0, cols]], offset=0))


def _owner_routing(own: nl.NkiTensor, name: str) -> tuple:
    """Route the owner rows: return the FP32 affinities [n, 896] and the destination hits [n, 128] (1.0 or 0.0).

    The threshold, the selected scores, their sum and its reciprocal are the instructions of
    ``tkg_moe_dp._dp_local_affinities`` on the same values, so each affinity has its bits. A destination rank is
    hit when one of its 7 experts has a nonzero affinity.
    """
    owner_rows = own.shape[0]
    scores = own[:, :, _GATHER_SCORE_COLS[0] : _GATHER_SCORE_COLS[1]]
    selection = own[:, :, _GATHER_SELECTION_COLS[0] : _GATHER_SELECTION_COLS[1]]
    top8_first = nl.ndarray((owner_rows, 8), dtype=nl.float32, buffer=nl.sbuf)
    nisa.max8(dst=top8_first, src=selection)
    selection_rest = nl.ndarray((owner_rows, TP_SIZE, DP_EXPERTS), dtype=nl.float32, buffer=nl.sbuf)
    nisa.nc_match_replace8(dst=selection_rest, data=selection, vals=top8_first, imm=_SELECTION_FLOOR)
    top8_second = nl.ndarray((owner_rows, 8), dtype=nl.float32, buffer=nl.sbuf)
    nisa.max8(dst=top8_second, src=selection_rest)
    threshold = top8_second[:, 7:8]
    # No reshape of these [n, ...] tiles: at n = 1 a reshape gives a partition step of 0.
    selected = nl.ndarray((owner_rows, TP_SIZE, DP_EXPERTS), dtype=nl.float32, buffer=nl.sbuf)
    score_sum = nl.ndarray((owner_rows, 1), dtype=nl.float32, buffer=nl.sbuf)
    select = nisa.NkiInstruction(
        nisa.scalar_tensor_tensor(
            dst=selected,
            data=selection,
            op0=nl.greater_equal,
            operand0=threshold,
            op1=nl.multiply,
            operand1=scores,
            reduce_op=nl.add,
            reduce_res=score_sum,
        )
    )
    nisa.tensor_scalar(dst=score_sum, data=score_sum, op0=nl.add, operand0=ROUTER_SUM_EPS)
    inv_score_sum = nl.ndarray((owner_rows, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.reciprocal(dst=inv_score_sum, data=score_sum)
    affinity = nl.ndarray((owner_rows, NUM_EXPERTS), dtype=nl.float32, buffer=nl.sbuf)
    scale = nisa.tensor_scalar(
        dst=affinity.ap(pattern=[[NUM_EXPERTS, owner_rows], [DP_EXPERTS, TP_SIZE], [1, DP_EXPERTS]], offset=0),
        data=selected,
        op0=nl.multiply,
        operand0=inv_score_sum,
    )
    scale.depends_on(select, kind=nisa.dependency.runtime)
    rank_sum = nl.ndarray((owner_rows, WORLD_SIZE), dtype=nl.float32, buffer=nl.sbuf)
    by_rank = affinity.ap(
        pattern=[[NUM_EXPERTS, owner_rows], [LOCAL_EXPERTS, WORLD_SIZE], [1, LOCAL_EXPERTS]], offset=0
    )
    nisa.tensor_reduce(dst=rank_sum, op=nl.add, data=by_rank, axis=(2,))
    hit = nl.ndarray((owner_rows, WORLD_SIZE), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=hit, data=rank_sum, op0=nl.greater, operand0=0.0)
    return affinity, hit


def _store_owner_tables(own, affinity, rank_info, sources, owner_rows: int, rows: int, name: str):
    """Store the owner latent rows [n, 3584] and the dispatch row tails [n * 128, 32] (row j * 128 + d) to HBM.

    The tail of (j, d) holds the 7 FP32 affinities of rank d's experts and the id r R + t n + j + 1 of owner row j
    (replica r, TP rank t): the 1-indexed row of the T tokens of the server.
    """
    nisa.dma_copy(
        dst=sources["owner_latent"].reshape((owner_rows, TP_SIZE, DP_LATENT)),
        src=own.ap(
            pattern=[[2 * TP_SIZE * _GATHER_COLS, owner_rows], [2 * _GATHER_COLS, TP_SIZE], [1, DP_LATENT]],
            offset=0,
            dtype=nl.uint16,
        ),
        priority=_CRITICAL_DMA_PRIORITY,
        name=f"{name}_owner_latent",
    )
    tail = nl.ndarray((owner_rows, WORLD_SIZE, _TAIL), dtype=nl.uint16, buffer=nl.sbuf)
    nisa.memset(dst=tail[:, :, _TAIL_USED:_TAIL], value=0)
    nisa.tensor_copy(
        dst=tail[:, :, 0 : 2 * LOCAL_EXPERTS],
        src=affinity.ap(
            pattern=[[2 * NUM_EXPERTS, owner_rows], [2 * LOCAL_EXPERTS, WORLD_SIZE], [1, 2 * LOCAL_EXPERTS]],
            offset=0,
            dtype=nl.uint16,
        ),
    )
    ranks = nl.ndarray((owner_rows, 4), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=ranks, src=rank_info.ap(pattern=[[0, owner_rows], [1, 4]], offset=0), name=f"{name}_id_ranks")
    ranks_f32 = nl.ndarray((owner_rows, 4), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=ranks_f32, src=ranks)
    # id = R * dp + n * tp + j + 1 for owner row j on partition j.
    base = nl.ndarray((owner_rows, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(dst=base, pattern=[[0, 1]], offset=1, channel_multiplier=1)
    nisa.scalar_tensor_tensor(
        dst=base, data=ranks_f32[:, 1:2], op0=nl.multiply, operand0=float(rows), op1=nl.add, operand1=base
    )
    nisa.scalar_tensor_tensor(
        dst=base, data=ranks_f32[:, 2:3], op0=nl.multiply, operand0=float(owner_rows), op1=nl.add, operand1=base
    )
    ids = nl.ndarray((owner_rows, WORLD_SIZE), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=ids, src=base.ap(pattern=[[1, owner_rows], [0, WORLD_SIZE]], offset=0))
    nisa.tensor_copy(
        dst=tail[:, :, 2 * LOCAL_EXPERTS : _TAIL_USED],
        src=ids.ap(pattern=[[2 * WORLD_SIZE, owner_rows], [2, WORLD_SIZE], [1, 2]], offset=0, dtype=nl.uint16),
    )
    nisa.dma_copy(
        dst=sources["owner_tail"].reshape((owner_rows, WORLD_SIZE, _TAIL)),
        src=tail,
        priority=_CRITICAL_DMA_PRIORITY,
        name=f"{name}_owner_tail",
    )


def _transpose_f32(src: nl.NkiTensor, name: str) -> nl.NkiTensor:
    """Return the FP32 transpose [F, P] of an FP32 SBUF tile [P, F] (P, F <= 128)."""
    partitions, cols = src.shape
    psum = _psum_bank(cols, (partitions,), nl.float32)
    nisa.nc_transpose(dst=psum, data=src)
    out = nl.ndarray((cols, partitions), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=out, src=psum)
    return out


def _dispatch_plan(hit: nl.NkiTensor, consts: dict, owner_rows: int, name: str) -> dict:
    """Plan the dispatch rows from the hits: destination rank d on partition d.

    Send row ``pos[d, j] = displ[d] + #{j' < j : hit[j', d]}`` carries owner row j to rank d. ``order[d, j]`` is the
    rank of d among the destinations of row j (the slot of the partial in the owner reduce).

    Returns:
        dict of FP32 SBUF tiles: ``send_counts`` and ``send_displs`` [128, 1] (rows), ``hit_T`` [128, n], ``prior``
        [128, n] (#{j' < j : hit[j', d]}), ``pos`` [128, n] (-1 where no hit), ``order`` [128, n], and the gather
        indices ``latent_idx`` / ``tail_idx`` [128, 1] INT32 of each send row (-1 past the last row).
    """
    hit_T = _transpose_f32(hit, f"{name}_hit")
    order_psum = _psum_bank(WORLD_SIZE, (owner_rows,), nl.float32)
    nisa.nc_matmul(dst=order_psum, stationary=consts["lower"], moving=hit_T, is_stationary_onezero=True)
    order = nl.ndarray((WORLD_SIZE, owner_rows), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=order, src=order_psum)
    # A tie at the threshold can select more than 16 experts: keep the first 16 destinations of each row.
    kept = nl.ndarray((WORLD_SIZE, owner_rows), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=kept, data=order, op0=nl.less, operand0=float(TOP_K))
    nisa.tensor_tensor(dst=hit_T, data1=hit_T, data2=kept, op=nl.multiply)
    send_counts = nl.ndarray((WORLD_SIZE, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=send_counts, op=nl.add, data=hit_T, axis=(1,))
    displ_psum = _psum_bank(WORLD_SIZE, (1,), nl.float32)
    nisa.nc_matmul(dst=displ_psum, stationary=consts["lower"], moving=send_counts, is_stationary_onezero=True)
    send_displs = nl.ndarray((WORLD_SIZE, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=send_displs, src=displ_psum)

    # One matmul over the owner rows: a column-by-column recurrence read stale columns on hardware.
    hit_kept = _transpose_f32(hit_T, f"{name}_hit_kept")
    prior_psum = _psum_bank(WORLD_SIZE, (owner_rows,), nl.float32)
    nisa.nc_matmul(
        dst=prior_psum,
        stationary=hit_kept,
        moving=consts["lower"][0:owner_rows, 0:owner_rows],
        is_stationary_onezero=True,
    )
    prior = nl.ndarray((WORLD_SIZE, owner_rows), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=prior, src=prior_psum)
    # pos = hit ? displ + prior : -1, as (displ + prior + 1) * hit - 1.
    pos = nl.ndarray((WORLD_SIZE, owner_rows), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=pos, data=prior, op0=nl.add, operand0=send_displs, op1=nl.add, operand1=1.0)
    _masked_minus_one(pos, hit_T)

    # Inverse map of the send rows: for send row p, its destination d(p) and owner row j(p).
    rows_of = nl.ndarray((WORLD_SIZE, _PMAX), dtype=nl.float32, buffer=nl.sbuf)
    owner_of = nl.ndarray((WORLD_SIZE, _PMAX), dtype=nl.float32, buffer=nl.sbuf)
    _inverse_tables(pos, consts["free"], rows_of, owner_of)
    lookup = nl.ndarray((WORLD_SIZE, 2), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=lookup[:, 0:1], src=consts["part"])
    nisa.memset(dst=lookup[:, 1:2], value=1.0)
    dest_psum = _psum_bank(_PMAX, (2,), nl.float32)
    nisa.nc_matmul(dst=dest_psum, stationary=rows_of, moving=lookup)
    owner_psum = _psum_bank(_PMAX, (1,), nl.float32)
    nisa.nc_matmul(dst=owner_psum, stationary=owner_of, moving=lookup[:, 1:2])
    dest = nl.ndarray((_PMAX, 2), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=dest, src=dest_psum)
    owner = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=owner, src=owner_psum)
    # latent row = j(p); tail row = j(p) * 128 + d(p); -1 where the send row is unused.
    latent_f = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=latent_f, data=owner, op0=nl.add, operand0=1.0)
    _masked_minus_one(latent_f, dest[:, 1:2])
    tail_f = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(
        dst=tail_f, data=owner, op0=nl.multiply, operand0=float(WORLD_SIZE), op1=nl.add, operand1=dest[:, 0:1]
    )
    nisa.tensor_scalar(dst=tail_f, data=tail_f, op0=nl.add, operand0=1.0)
    _masked_minus_one(tail_f, dest[:, 1:2])
    latent_idx = nl.ndarray((_PMAX, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=latent_idx, src=latent_f)
    tail_idx = nl.ndarray((_PMAX, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=tail_idx, src=tail_f)
    return {
        "send_counts": send_counts,
        "send_displs": send_displs,
        "hit_T": hit_T,
        "prior": prior,
        "pos": pos,
        "order": order,
        "latent_idx": latent_idx,
        "tail_idx": tail_idx,
    }


def _masked_minus_one(values: nl.NkiTensor, mask: nl.NkiTensor):
    """In place: ``values = values * mask - 1`` (mask is 0 or 1, per element or per partition [P, 1])."""
    if mask.shape == values.shape:
        nisa.tensor_tensor(dst=values, data1=values, data2=mask, op=nl.multiply)
    else:
        nisa.tensor_scalar(dst=values, data=values, op0=nl.multiply, operand0=mask)
    nisa.tensor_scalar(dst=values, data=values, op0=nl.subtract, operand0=1.0)


def _inverse_tables(index: nl.NkiTensor, free: nl.NkiTensor, rows_of: nl.NkiTensor, owner_of: nl.NkiTensor = None):
    """Build the one-hot inverse of an integer table ``index`` [128, n] (values < 128, -1 unused).

    ``rows_of[d, p] = sum_j [index[d, j] == p]`` and ``owner_of[d, p] = sum_j j * [index[d, j] == p]``. Each p is
    taken by at most one (d, j), so a matmul over d with these tiles reads the d and the j of p exactly.
    """
    owner_rows = index.shape[1]
    nisa.memset(dst=rows_of, value=0.0)
    if owner_of is not None:
        nisa.memset(dst=owner_of, value=0.0)
    for j in range(owner_rows):
        equal = nl.ndarray(rows_of.shape, dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=equal, data=free, op0=nl.equal, operand0=index[:, j : j + 1])
        nisa.tensor_tensor(dst=rows_of, data1=rows_of, data2=equal, op=nl.add)
        if j and owner_of is not None:
            nisa.scalar_tensor_tensor(
                dst=owner_of, data=equal, op0=nl.multiply, operand0=float(j), op1=nl.add, operand1=owner_of
            )


def _write_meta(meta_hbm, counts, displs, recv_counts, row_elems: int, name: str, site: str):
    """Write the [3, 128] UINT32 all-to-all-v metadata (elements) from FP32 [128, 1] row counts and displacements.

    Row 2 is ``recv_counts`` when it is given (the combine knows them), else 0 (the dispatch fills it).
    """
    columns = nl.ndarray((WORLD_SIZE, 3), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=columns[:, 0:1], data=counts, op0=nl.multiply, operand0=float(row_elems))
    nisa.tensor_scalar(dst=columns[:, 1:2], data=displs, op0=nl.multiply, operand0=float(row_elems))
    if recv_counts is None:
        nisa.memset(dst=columns[:, 2:3], value=0.0)
    else:
        nisa.tensor_scalar(dst=columns[:, 2:3], data=recv_counts, op0=nl.multiply, operand0=float(row_elems))
    rows_f = _transpose_f32(columns, f"{name}_{site}_meta")
    rows_u32 = nl.ndarray((3, WORLD_SIZE), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=rows_u32, src=rows_f)
    nisa.dma_copy(dst=meta_hbm, src=rows_u32, priority=_CRITICAL_DMA_PRIORITY, name=f"{name}_{site}_meta")


def _permute(plan: dict, sources: dict, owner_rows: int, name: str):
    """Fill the dispatch send rows (grouped by destination) from the owner tables.

    Each part (latent, tail) is gathered into its own SBUF tile and stored with a plain DMA: an HBM-to-HBM gather with
    a vector offset read the wrong rows.
    """
    send_rows = TOP_K * owner_rows
    for table, idx_key, first_col, cols in (
        ("owner_latent", "latent_idx", 0, LATENT_SIZE),
        ("owner_tail", "tail_idx", LATENT_SIZE, _TAIL),
    ):
        part = nl.ndarray((send_rows, cols), dtype=nl.uint16, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=part,
            src=sources[table].ap(
                pattern=[[cols, send_rows], [1, cols]],
                offset=0,
                vector_offset=plan[idx_key][0:send_rows, :],
                indirect_dim=0,
            ),
            oob_mode=nisa.oob_mode.skip,
            dge_mode=nisa.dge_mode.swdge,
            name=f"{name}_permute_{table}",
        )
        nisa.dma_copy(
            dst=sources["dispatch_send"][:, first_col : first_col + cols],
            src=part,
            priority=_CRITICAL_DMA_PRIORITY,
            name=f"{name}_send_{table}",
        )


def _received_rows(sources: dict, consts: dict, owner_rows: int, name: str) -> tuple:
    """Return the rows received from each source and their exclusive prefix, FP32 [128, 1] (source on partitions).

    The dispatch writes the received element counts into row 2 of its metadata. A count is a multiple of the row
    size, so the rows are counted by comparisons, with no division.
    """
    counts_u32 = nl.ndarray((1, WORLD_SIZE), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.dma_copy(dst=counts_u32, src=sources["dispatch_meta"][2:3, :], name=f"{name}_recv_counts")
    counts_f = nl.ndarray((1, WORLD_SIZE), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=counts_f, src=counts_u32)
    counts_T = _transpose_f32(counts_f, f"{name}_recv_counts")
    rows = nl.ndarray((WORLD_SIZE, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=rows, value=0.0)
    for slot in range(owner_rows):
        nisa.scalar_tensor_tensor(
            dst=rows,
            data=counts_T,
            op0=nl.greater_equal,
            operand0=float((slot + 1) * A2AV_ROW),
            op1=nl.add,
            operand1=rows,
        )
    displ_psum = _psum_bank(WORLD_SIZE, (1,), nl.float32)
    nisa.nc_matmul(dst=displ_psum, stationary=consts["lower"], moving=rows, is_stationary_onezero=True)
    displs = nl.ndarray((WORLD_SIZE, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=displs, src=displ_psum)
    # Keep [rows | inclusive prefix] in HBM, for the packing broadcast.
    table = nl.ndarray((WORLD_SIZE, 2), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=table[:, 0:1], src=rows)
    nisa.tensor_tensor(dst=table[:, 1:2], data1=displs, data2=rows, op=nl.add)
    nisa.dma_copy(
        dst=sources["recv_rows"], src=_transpose_f32(table, f"{name}_recv_rows"), name=f"{name}_recv_rows_store"
    )
    return rows, displs


def _packed_slots(sources: dict, consts: dict, owner_rows: int, capacity: int, name: str) -> list:
    """Return the dispatch slot row of each packed row, INT32 [128, 1] per tile of 128 packed rows (-1 past the total).

    Packed row i (source order) is slot row ``i + sum_s [incl_s <= i] (n - rows_s)``.
    """
    table = nl.ndarray((_PMAX, 2, WORLD_SIZE), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=table,
        src=sources["recv_rows"].ap(pattern=[[0, _PMAX], [WORLD_SIZE, 2], [1, WORLD_SIZE]], offset=0),
        name=f"{name}_recv_rows_load",
    )
    rows_b, incl_b = table[:, 0, :], table[:, 1, :]
    gap = nl.ndarray((_PMAX, WORLD_SIZE), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=gap, data=rows_b, op0=nl.multiply, operand0=-1.0, op1=nl.add, operand1=float(owner_rows))
    slots = []
    for first in range(0, capacity, _EXPERT_CALL_ROWS):
        packed = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=packed, data=consts["part"], op0=nl.add, operand0=float(first))
        before = nl.ndarray((_PMAX, WORLD_SIZE), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=before, data=incl_b, op0=nl.less_equal, operand0=packed)
        weighted = nl.ndarray((_PMAX, WORLD_SIZE), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=weighted, data1=before, data2=gap, op=nl.multiply)
        skipped = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=skipped, op=nl.add, data=weighted, axis=(1,))
        slot = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=slot, data1=packed, data2=skipped, op=nl.add)
        valid = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=valid, data1=packed, data2=incl_b[:, WORLD_SIZE - 1 : WORLD_SIZE], op=nl.less)
        nisa.tensor_scalar(dst=slot, data=slot, op0=nl.add, operand0=1.0)
        _masked_minus_one(slot, valid)
        slot_idx = nl.ndarray((_PMAX, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=slot_idx, src=slot)
        slots.append(slot_idx)
    return slots


def _as_u16(tile: nl.NkiTensor):
    """Return the UINT16 access pattern of the bytes of a 2-D FP32 SBUF tile [P, F] (2 F elements per partition)."""
    partitions, cols = tile.shape
    return tile.ap(pattern=[[2 * cols, partitions], [1, 2 * cols]], offset=0, dtype=nl.uint16)


def _gather_slots(dst, rows: int, recv: nl.NkiTensor, slot_idx: nl.NkiTensor, first_col: int, cols: int, name: str):
    """Gather columns [first_col, first_col + cols) of the dispatch slot rows ``slot_idx`` into ``rows`` rows of
    ``dst`` (UINT16)."""
    nisa.dma_copy(
        dst=dst,
        src=recv.ap(
            pattern=[[A2AV_ROW, rows], [1, cols]], offset=first_col, vector_offset=slot_idx[0:rows, :], indirect_dim=0
        ),
        oob_mode=nisa.oob_mode.skip,
        dge_mode=nisa.dge_mode.swdge,
        name=name,
    )


def _pack_tile(sources: dict, slot_idx: nl.NkiTensor, first, size: int, name: str) -> tuple:
    """Gather ``size`` received rows (packed rows ``first`` on) into SBUF: BF16 latent and FP32 affinities.

    Rows past the received total stay zero (latent and affinities), so their partials are 0 and they are not sent
    back.
    """
    recv = sources["dispatch_recv"]
    latent_sb = nl.ndarray((size, LATENT_SIZE), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.memset(dst=latent_sb, value=0.0)
    affinity_sb = nl.ndarray((size, LOCAL_EXPERTS), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=affinity_sb, value=0.0)
    _gather_slots(latent_sb.view(nl.uint16), size, recv, slot_idx, 0, LATENT_SIZE, f"{name}_pack_latent_{first}")
    _gather_slots(
        _as_u16(affinity_sb), size, recv, slot_idx, A2AV_AFFINITY_COL, 2 * LOCAL_EXPERTS, f"{name}_pack_aff_{first}"
    )
    return latent_sb, affinity_sb


def _pack_received_hbm(sources: dict, slots: list, capacity: int, name: str):
    """Gather the received rows into the packed HBM rows of the ragged experts (``packed_latent``,
    ``packed_affinity``). The affinities of the rows past the received total are zero, so no expert takes them.
    """
    recv = sources["dispatch_recv"]
    for tile_idx, slot_idx in enumerate(slots):
        first = tile_idx * _EXPERT_CALL_ROWS
        size = min(_EXPERT_CALL_ROWS, capacity - first)
        rows = slice(first, first + size)
        # Through SBUF: an HBM-to-HBM gather with a vector offset reads the wrong rows.
        latent_sb = nl.ndarray((size, LATENT_SIZE), dtype=nl.uint16, buffer=nl.sbuf)
        nisa.memset(dst=latent_sb, value=0)
        _gather_slots(latent_sb, size, recv, slot_idx, 0, LATENT_SIZE, f"{name}_pack_latent_hbm_{first}")
        nisa.dma_copy(dst=sources["packed_latent"][rows, :], src=latent_sb, name=f"{name}_latent_store_{first}")
        affinity_sb = nl.ndarray((size, LOCAL_EXPERTS), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=affinity_sb, value=0.0)
        _gather_slots(
            _as_u16(affinity_sb),
            size,
            recv,
            slot_idx,
            A2AV_AFFINITY_COL,
            2 * LOCAL_EXPERTS,
            f"{name}_pack_aff_hbm_{first}",
        )
        nisa.dma_copy(dst=sources["packed_affinity"][rows, :], src=affinity_sb, name=f"{name}_aff_store_{first}")


def _owner_reduce_and_scale(sources: dict, plan: dict, consts: dict, owner_rows: int, name: str, debug) -> nl.NkiTensor:
    """Add the partials of each owner row in FP32 and return the RMS-scaled routed rows [n, 3584] BF16.

    Gather partition q = 16 j + k takes the partial of the k-th destination of row j: its combine slot is
    ``d * n + prior[d, j]``. PE transposes move each 128-column tile to [128 columns, q], and VectorE adds the 16
    slots of each row, so the sum of a row reads no other row. ``scaled = bf16(routed * rsqrt(sum(routed^2) / 3584
    + 1e-5))``, with routed = bf16(FP32 sum).
    """
    hit_T, order, prior = plan["hit_T"], plan["order"], plan["prior"]
    # gather partition of (d, j): hit ? 16 j + order : -1.
    gather_pos = nl.ndarray((WORLD_SIZE, owner_rows), dtype=nl.float32, buffer=nl.sbuf)
    for j in range(owner_rows):
        nisa.tensor_scalar(
            dst=gather_pos[:, j : j + 1], data=order[:, j : j + 1], op0=nl.add, operand0=float(TOP_K * j + 1)
        )
    _masked_minus_one(gather_pos, hit_T)
    rows_of = nl.ndarray((WORLD_SIZE, _PMAX), dtype=nl.float32, buffer=nl.sbuf)
    _inverse_tables(gather_pos, consts["free"], rows_of)
    # prior_of[d, q] = prior[d, j] for the (d, j) of q.
    prior_of = nl.ndarray((WORLD_SIZE, _PMAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=prior_of, value=0.0)
    for j in range(owner_rows):
        equal = nl.ndarray((WORLD_SIZE, _PMAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=equal, data=consts["free"], op0=nl.equal, operand0=gather_pos[:, j : j + 1])
        nisa.scalar_tensor_tensor(
            dst=prior_of, data=equal, op0=nl.multiply, operand0=prior[:, j : j + 1], op1=nl.add, operand1=prior_of
        )
    lookup = nl.ndarray((WORLD_SIZE, 2), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=lookup[:, 0:1], src=consts["part"])
    nisa.memset(dst=lookup[:, 1:2], value=1.0)
    dest_psum = _psum_bank(_PMAX, (2,), nl.float32)
    nisa.nc_matmul(dst=dest_psum, stationary=rows_of, moving=lookup)
    prior_psum = _psum_bank(_PMAX, (1,), nl.float32)
    nisa.nc_matmul(dst=prior_psum, stationary=prior_of, moving=lookup[:, 1:2])
    dest = nl.ndarray((_PMAX, 2), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=dest, src=dest_psum)
    slot = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=slot, src=prior_psum)
    nisa.scalar_tensor_tensor(
        dst=slot, data=dest[:, 0:1], op0=nl.multiply, operand0=float(owner_rows), op1=nl.add, operand1=slot
    )
    nisa.tensor_scalar(dst=slot, data=slot, op0=nl.add, operand0=1.0)
    _masked_minus_one(slot, dest[:, 1:2])
    slot_idx = nl.ndarray((_PMAX, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=slot_idx, src=slot)

    gathered = TOP_K * owner_rows
    partials = nl.ndarray((gathered, COMBINE_ROW), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.memset(dst=partials, value=0.0)
    nisa.dma_copy(
        dst=partials,
        src=sources["combine_recv"].ap(
            pattern=[[COMBINE_ROW, gathered], [1, COMBINE_ROW]],
            offset=0,
            vector_offset=slot_idx[0:gathered, :],
            indirect_dim=0,
        ),
        oob_mode=nisa.oob_mode.skip,
        dge_mode=nisa.dge_mode.swdge,
        name=f"{name}_owner_gather",
    )
    routed_T = nl.ndarray((_PMAX, _LATENT_TILES, owner_rows), dtype=nl.bfloat16, buffer=nl.sbuf)
    for first in range(0, _LATENT_TILES, _TRANSPOSES_PER_BANK):
        tiles = min(_TRANSPOSES_PER_BANK, _LATENT_TILES - first)
        psum = _psum_bank(_PMAX, (tiles, gathered), nl.bfloat16)
        for tile_offset in range(tiles):
            col = (first + tile_offset) * _PMAX
            nisa.nc_transpose(dst=psum[:, tile_offset, :], data=partials[:, col : col + _PMAX])
        sums = nl.ndarray((_PMAX, tiles, owner_rows), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=sums, op=nl.add, data=psum.reshape((_PMAX, tiles, owner_rows, TOP_K)), axis=(3,))
        nisa.tensor_copy(dst=routed_T[:, first : first + tiles, :], src=sums)

    squares = nl.ndarray((_PMAX, owner_rows, _LATENT_TILES), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=squares, data1=routed_T.permute((0, 2, 1)), data2=routed_T.permute((0, 2, 1)), op=nl.multiply
    )
    partial_sums = nl.ndarray((_PMAX, owner_rows), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=partial_sums, op=nl.add, data=squares, axis=(2,))
    total_psum = _psum_bank(_PMAX, (owner_rows,), nl.float32)
    nisa.nc_matmul(dst=total_psum, stationary=consts["ones"], moving=partial_sums, is_stationary_onezero=True)
    inv_rms = nl.ndarray((_PMAX, owner_rows), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=inv_rms, op=nl.rsqrt, data=total_psum, scale=1.0 / LATENT_SIZE, bias=LATENT_NORM_EPS)
    scaled_T = nl.ndarray((_PMAX, _LATENT_TILES, owner_rows), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=scaled_T,
        data1=routed_T,
        data2=inv_rms.ap(pattern=[[owner_rows, _PMAX], [0, _LATENT_TILES], [1, owner_rows]], offset=0),
        op=nl.multiply,
    )

    scaled_rows = _rows_from_T(scaled_T, owner_rows)
    if debug and "routed" in debug:
        nisa.dma_copy(dst=debug["routed"], src=_rows_from_T(routed_T, owner_rows), name=f"{name}_debug_routed")
    return scaled_rows


def _rows_from_T(values_T: nl.NkiTensor, owner_rows: int) -> nl.NkiTensor:
    """Transpose [128, 28, n] BF16 (column 128 c + p of row j at (p, c, j)) back to rows [n, 3584] BF16."""
    rows_sb = nl.ndarray((owner_rows, LATENT_SIZE), dtype=nl.bfloat16, buffer=nl.sbuf)
    for first in range(0, _LATENT_TILES, _TRANSPOSES_PER_BANK):
        tiles = min(_TRANSPOSES_PER_BANK, _LATENT_TILES - first)
        # One whole 2-byte PSUM bank, sliced (no reshape: n may be 1).
        bank = nl.ndarray((owner_rows, _TRANSPOSES_PER_BANK * _PMAX), dtype=nl.bfloat16, buffer=nl.psum)
        for tile_offset in range(tiles):
            nisa.nc_transpose(
                dst=bank[:, tile_offset * _PMAX : (tile_offset + 1) * _PMAX], data=values_T[:, first + tile_offset, :]
            )
        nisa.tensor_copy(dst=rows_sb[:, first * _PMAX : (first + tiles) * _PMAX], src=bank[:, 0 : tiles * _PMAX])
    return rows_sb


def _normed_rows(tp_out_dst: nl.NkiTensor, moe_vectors_dp: nl.NkiTensor, act_tok: list, name: str):
    """Write ``bf16(scaled * w)`` of this rank's 112 latent columns into columns 0..111 of the out-projection input.

    ``tp_out_dst`` [R, 112] holds the scaled latent columns of each row of the replica, in row order.
    """
    start = 0
    for tile in act_tok:
        size = tile.shape[0]
        scaled = nl.ndarray((size, DP_LATENT), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=scaled.view(nl.uint16),
            src=tp_out_dst[start : start + size, :],
            priority=_CRITICAL_DMA_PRIORITY,
            name=f"{name}_tp_out_load_{start}",
        )
        weight = nl.ndarray((size, DP_LATENT), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=weight,
            src=moe_vectors_dp.ap(pattern=[[0, size], [1, DP_LATENT]], offset=DP_NORM_COLS[0]),
            priority=_CRITICAL_DMA_PRIORITY,
            name=f"{name}_norm_weight_{start}",
        )
        nisa.tensor_tensor(
            dst=tile[:, DP_LATENT_UP_ROWS[0] : DP_LATENT_UP_ROWS[1]], data1=scaled, data2=weight, op=nl.multiply
        )
        start += size


def _out_projection(act_tok: list, out_proj_sb, rows: int, sources: dict, tp_group, name: str) -> nl.NkiTensor:
    """Run the K = 304 out-projection of the R rows and the TP32 output all-reduce; return [128, R * 56] BF16."""
    k_tiles = _out_k_tiles(DP_OUT_PROJ_ROWS)
    act_T = nl.ndarray((_PMAX, len(k_tiles), rows), dtype=nl.bfloat16, buffer=nl.sbuf)
    start = 0
    for tile in act_tok:
        size = tile.shape[0]
        for k_idx, (k_start, k_stop) in enumerate(k_tiles):
            psum = _psum_bank(k_stop - k_start, (size,), nl.bfloat16)
            nisa.nc_transpose(dst=psum, data=tile[:, k_start:k_stop])
            nisa.tensor_copy(dst=act_T[0 : k_stop - k_start, k_idx, start : start + size], src=psum)
        start += size
    partial_hT = _project_to_hT(
        lambda h: [
            (out_proj_sb[0 : stop - k_start, k_idx, h * _PMAX : (h + 1) * _PMAX], act_T[0 : stop - k_start, k_idx, :])
            for k_idx, (k_start, stop) in enumerate(k_tiles)
        ],
        rows,
    )
    nisa.dma_copy(dst=sources["out_src"], src=partial_hT, name=f"{name}_out_store")
    release_weight_gate(
        ncc.all_reduce(
            srcs=[sources["out_src"]], dsts=[sources["out_dst"]], replica_group=tp_group, op=nl.add, name=f"{name}_out"
        )
    )
    out = nl.ndarray((_PMAX, rows * _H_TILES), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=out, src=sources["out_dst"], name=f"{name}_out_load")
    return out


def _store_debug(debug: dict, affinity, plan: dict, sources: dict, capacity: int, name: str):
    """Copy the intermediates that ``debug`` asks for into its HBM tensors."""
    if "affinity" in debug:
        nisa.dma_copy(dst=debug["affinity"], src=affinity, name=f"{name}_debug_affinity")
    if "send_counts" in debug:
        nisa.dma_copy(
            dst=debug["send_counts"],
            src=_transpose_f32(plan["send_counts"], f"{name}_debug"),
            name=f"{name}_debug_send",
        )
    if "recv_rows" in debug:
        nisa.dma_copy(dst=debug["recv_rows"], src=sources["recv_rows"][0:1, :], name=f"{name}_debug_recv")
    if "partials" in debug:
        nisa.dma_copy(
            dst=debug["partials"],
            src=sources["combine_send"][0:capacity, :],
            name=f"{name}_debug_partials",
        )
