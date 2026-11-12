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

"""Kimi-K3 multi-host decode feed-forward glue: the dense MLP over a world of several hosts, and its collectives.

A multi-host decode runs on ``nnodes`` hosts of 128 LNC1 ranks each (Trn2 or Trn3), so the world has ``128 * nnodes``
ranks. The attention keeps the replicas of the DP4 form inside each host: DP replicas of TP32 / CP32,
``dp = world / 32``, 4 replicas on each host. The feed-forward layers shard over the whole world. So each feed-forward
body first gathers the ``dp * B`` tokens of all replicas over the DP group, and its output all-reduce runs over the
world.

Both groups span the hosts: the DP group of TP rank ``t`` holds rank ``t`` of every replica, and the world holds every
rank. The runtime runs SBUF collectives only inside one host. So every feed-forward collective of this module runs
in HBM: the DP x gather (``x_src_gather_hbm`` to ``x_dst_hbm``) and the output all-reduce (``out_hbm`` to
``out_dst_hbm``), both persistent tensors of the kernel (``ffn_collective_hbm``). Their SBUF ends are local DMA
endpoints, which no peer accesses. The code has no target branch: the helpers of ``tkg_moe`` that it calls pick
the DMA priority by the target.

``dense_stand_in_moe`` runs the dense MLP in a MoE layer, for an entry whose MoE body does not exist yet.
"""

import functools

import nki.isa as nisa
import nki.language as nl
from nki.collectives import ReplicaGroup

from ..utils.kernel_assert import kernel_assert
from ..common import dma_priority, pair_shared_buffer
from ..tkg_moe import (
    _CRITICAL_DMA_PRIORITY,
    _DENSE_GATE_UP_DMA_CHUNKS,
    _H_TILES,
    _PMAX,
    DENSE_INTERMEDIATE,
    HIDDEN_SIZE,
    _all_reduce_select_rows,
    _gather_replica_tokens,
    _load_hT_rows,
    _project_to_hT,
    _psum_bank,
    _situ,
    _tile,
    rank_selectors,
)

HOST_RANKS = 128
"""LNC1 ranks of one host (16 Trn2 or Trn3 devices of 8 physical cores)."""
ATTN_TP = 32
"""Ranks of one attention replica (TP32 / CP32), inside one host."""
MAX_GATHERED_TOKENS = _PMAX
"""Largest ``dp * B``: the gathered tokens are the stationary columns and the PSUM partitions of the feed-forward
matmuls."""
TRN3_EXPERT_LOOP_KEYS = frozenset({"scale_buffers", "weight_buffers"})
"""The expert-loop options of the Trn3 static MoE kernel that ``decode_block_body`` passes to every MoE body. A MoE
body of this module drops them."""


def world_layout(world_size: int) -> dict:
    """Return the multi-host decode layout of a world of ``world_size`` ranks.

    Returns:
        dict with ``world`` (ranks), ``nnodes`` (hosts), ``dp`` (attention replicas), ``tp`` (32), and
        ``dense_shard`` (intermediate columns of the dense MLP on each rank).

    Example:
        >>> world_layout(256)["dp"], world_layout(512)["dp"]
        (8, 16)
    """
    kernel_assert(
        world_size > 0 and world_size % HOST_RANKS == 0 and DENSE_INTERMEDIATE % world_size == 0,
        f"A multi-host world takes whole hosts of {HOST_RANKS} ranks that divide the {DENSE_INTERMEDIATE} dense "
        f"columns (256, 384 or 512 ranks for 2, 3 or 4 hosts), got {world_size}",
    )
    return {
        "world": world_size,
        "nnodes": world_size // HOST_RANKS,
        "dp": world_size // ATTN_TP,
        "tp": ATTN_TP,
        "dense_shard": DENSE_INTERMEDIATE // world_size,
    }


def replica_groups_world(world_size: int) -> tuple[ReplicaGroup, ReplicaGroup, ReplicaGroup]:
    """Return the TP32, DP and world replica groups of a multi-host world.

    Rank ``g`` has TP rank ``g % 32`` and DP rank ``g // 32``, so each TP32 group lies inside one host.
    """
    dp = world_layout(world_size)["dp"]
    tp_group = ReplicaGroup([[ATTN_TP * d + t for t in range(ATTN_TP)] for d in range(dp)])
    dp_group = ReplicaGroup([[ATTN_TP * d + t for d in range(dp)] for t in range(ATTN_TP)])
    world_group = ReplicaGroup([list(range(world_size))])
    return tp_group, dp_group, world_group


def ffn_collective_hbm(replica_tokens: int, dp_size: int, name: str = "", pair_shared: tuple = ()) -> tuple[dict, dict]:
    """Return the persistent HBM tensors of the feed-forward collectives, and the second x gather source.

    - ``x_src_gather_hbm`` ``[128, B * 56]`` and ``x_dst_hbm`` ``[dp * 128, B * 56]``: the DP x gather. The gather
      stacks the replicas on rows; ``x_src_gather_hbm`` alternates by layer parity (the second dict, for the
      ``pingpong`` set of ``decode_block.collective_sources``).
    - ``out_hbm`` and ``out_dst_hbm`` ``[128, dp * B * 56]``: the output all-reduce over the world.

    A peer reads the x gather source of a layer at the latest in that gather, and it writes the output all-reduce
    destination of a layer at the earliest in that all-reduce. A world collective separates each use from the next one.

    ``pair_shared`` (the checked ``pair_shared_cc_dsts`` of the entry, ``common.PAIR_SHARED_CC_DST_SITES``): with
    ``moe_out`` the output all-reduce destination ``out_dst_hbm`` goes to LNC1 shared HBM (the world holds both cores
    of every HBM stack). The x gather runs over the strided DP group, so its tensors always stay private, as every
    source does.
    """
    cols = replica_tokens * _H_TILES
    shapes = {
        "x_src_gather_hbm": (_PMAX, cols),
        "x_dst_hbm": (dp_size * _PMAX, cols),
        "out_hbm": (_PMAX, dp_size * cols),
        "out_dst_hbm": (_PMAX, dp_size * cols),
    }
    tensors = {
        key: nl.ndarray(
            shape,
            dtype=nl.bfloat16,
            buffer=pair_shared_buffer(key == "out_dst_hbm" and "moe_out" in pair_shared),
            name=f"ffn2{name}_{key}",
        )
        for key, shape in shapes.items()
    }
    second = {
        "x_src_gather_hbm": nl.ndarray(
            shapes["x_src_gather_hbm"], dtype=nl.bfloat16, buffer=nl.private_hbm, name=f"ffn2{name}_x_src_gather_hbm_pp"
        )
    }
    return tensors, second


def _x_gather_sources(sources, replica_tokens: int, dp_size: int, name: str) -> dict:
    """Return the HBM pair of the DP x gather (the keys of ``tkg_moe._gather_replica_tokens``), new without
    ``sources``."""
    if sources is not None and "x_src_gather_hbm" in sources:
        src, dst = sources["x_src_gather_hbm"], sources["x_dst_hbm"]
    else:
        cols = replica_tokens * _H_TILES
        src = nl.ndarray((_PMAX, cols), dtype=nl.bfloat16, buffer=nl.private_hbm, name=f"{name}_x_src_hbm")
        dst = nl.ndarray((dp_size * _PMAX, cols), dtype=nl.bfloat16, buffer=nl.private_hbm, name=f"{name}_x_dst_hbm")
    kernel_assert(
        tuple(src.shape) == (_PMAX, replica_tokens * _H_TILES)
        and tuple(dst.shape) == (dp_size * _PMAX, replica_tokens * _H_TILES),
        f"x gather pair must be [128, {replica_tokens * _H_TILES}] and [{dp_size * _PMAX}, "
        f"{replica_tokens * _H_TILES}], got {src.shape} and {dst.shape}",
    )
    return {"x_gather_src_hbm": src, "x_gather_hbm": dst}


def _validate_dense_world(x_hT, dense_gate_up, dense_down, rank_info, dp_size: int, layout: dict):
    """Check the inputs of the multi-host dense MLP body."""
    kernel_assert(
        nl.program_ndim() == 0 or (nl.program_ndim() == 1 and nl.num_programs(0) == 1),
        "The Kimi-K3 multi-host feed-forward bodies support LNC1 only",
    )
    kernel_assert(dp_size == layout["dp"], f"dp_size {dp_size} is not world / 32 = {layout['dp']}")
    kernel_assert(
        len(x_hT.shape) == 3 and x_hT.shape[0] == _PMAX and x_hT.shape[2] == _H_TILES and x_hT.dtype == nl.bfloat16,
        f"x_hT must be BF16 [{_PMAX}, B, {_H_TILES}], got {x_hT.dtype} {x_hT.shape}",
    )
    kernel_assert(
        dp_size * x_hT.shape[1] <= MAX_GATHERED_TOKENS,
        f"{dp_size} replicas x B={x_hT.shape[1]} gather more than {MAX_GATHERED_TOKENS} tokens: use B <= "
        f"{MAX_GATHERED_TOKENS // dp_size}",
    )
    shard = layout["dense_shard"]
    kernel_assert(
        tuple(dense_gate_up.shape) == (_PMAX, _H_TILES, 2 * shard) and dense_gate_up.dtype == nl.bfloat16,
        f"dense_gate_up must be BF16 [{_PMAX}, {_H_TILES}, {2 * shard}] for {layout['world']} ranks, "
        f"got {dense_gate_up.dtype} {dense_gate_up.shape}",
    )
    kernel_assert(
        tuple(dense_down.shape) == (shard, HIDDEN_SIZE) and dense_down.dtype == nl.bfloat16,
        f"dense_down must be BF16 [{shard}, {HIDDEN_SIZE}] for {layout['world']} ranks, "
        f"got {dense_down.dtype} {dense_down.shape}",
    )
    kernel_assert(
        tuple(rank_info.shape) == (1, 4) and rank_info.dtype == nl.int32,
        f"rank_info must be INT32 [1, 4], got {rank_info.dtype} {rank_info.shape}",
    )


def _dense_ffn(
    x_hT: nl.NkiTensor,
    dense_gate_up: nl.NkiTensor,
    dense_down: nl.NkiTensor,
    rank_info: nl.NkiTensor,
    dp_group: ReplicaGroup,
    world_group: ReplicaGroup,
    tag: str,
    sources: dict | None,
    dp_size: int,
    layout: dict,
) -> nl.NkiTensor:
    """Run the dense SiTU MLP for the tokens of all DP replicas and keep this replica's rows (``dense_ffn_world``)."""
    _validate_dense_world(x_hT, dense_gate_up, dense_down, rank_info, dp_size, layout)
    shard = layout["dense_shard"]
    replica_tokens = x_hT.shape[1]
    tokens = dp_size * replica_tokens
    name = f"k3mlp_{tag}"

    gate_up_sb = nl.ndarray((_PMAX, _H_TILES, 2 * shard), dtype=nl.bfloat16, buffer=nl.sbuf)
    _load_hT_rows(gate_up_sb, dense_gate_up, _DENSE_GATE_UP_DMA_CHUNKS, f"{name}_gate_up")
    down_tiles = -(-shard // _PMAX)
    full_down_tiles = shard // _PMAX
    down_tail_rows = shard - full_down_tiles * _PMAX
    down_sb = nl.ndarray((_PMAX, down_tiles, HIDDEN_SIZE), dtype=nl.bfloat16, buffer=nl.sbuf)
    if full_down_tiles:
        down_rows = dense_down.ap(
            pattern=[[HIDDEN_SIZE, _PMAX], [_PMAX * HIDDEN_SIZE, full_down_tiles], [1, HIDDEN_SIZE]], offset=0
        )
        _copy(down_sb[:, 0:full_down_tiles, :], down_rows, f"{name}_down")
    if down_tail_rows:
        _copy(
            down_sb[0:down_tail_rows, full_down_tiles, :],
            dense_down[full_down_tiles * _PMAX : shard, :],
            f"{name}_down_tail",
        )

    x_sources = _x_gather_sources(sources, replica_tokens, dp_size, name)
    x_tokens_hT = _gather_replica_tokens(x_hT, dp_group, f"{name}_x", x_sources, dp_size)
    row_select, _ = rank_selectors(rank_info, f"{name}_rank", dp_size=dp_size)

    gate_up_bf16 = nl.ndarray((tokens, 2 * shard), dtype=nl.bfloat16, buffer=nl.sbuf)
    for projection_idx in range(2):
        columns = slice(projection_idx * shard, (projection_idx + 1) * shard)
        projection_psum = _psum_bank(tokens, (shard,), nl.float32)
        for h_tile_idx in range(_H_TILES):
            nisa.nc_matmul(
                dst=projection_psum,
                stationary=x_tokens_hT[:, h_tile_idx, :],
                moving=gate_up_sb[:, h_tile_idx, columns],
            )
        nisa.tensor_copy(dst=gate_up_bf16[:, columns], src=projection_psum)
    activation = _situ(gate_up_bf16[:, 0:shard], gate_up_bf16[:, shard : 2 * shard])

    # Each transposed tile takes an even PSUM width, so a one-token tile starts at a 4-byte PSUM offset.
    psum_tokens = tokens + tokens % 2
    activation_T_psum = _psum_bank(_PMAX, (down_tiles, psum_tokens), nl.bfloat16)
    for i_tile_idx in range(down_tiles):
        i_rows = min(_PMAX, shard - i_tile_idx * _PMAX)
        nisa.nc_transpose(
            dst=activation_T_psum[0:i_rows, i_tile_idx, 0:tokens],
            data=activation[:, i_tile_idx * _PMAX : i_tile_idx * _PMAX + i_rows],
        )
    activation_T = nl.ndarray((_PMAX, down_tiles, tokens), dtype=nl.bfloat16, buffer=nl.sbuf)
    if full_down_tiles:
        nisa.tensor_copy(
            dst=activation_T[:, 0:full_down_tiles, :], src=activation_T_psum[:, 0:full_down_tiles, 0:tokens]
        )
    if down_tail_rows:
        nisa.tensor_copy(
            dst=activation_T[0:down_tail_rows, full_down_tiles, :],
            src=activation_T_psum[0:down_tail_rows, full_down_tiles, 0:tokens],
        )

    def down_terms(h_tile_idx: int) -> list[tuple[nl.NkiTensor, nl.NkiTensor]]:
        terms = []
        for i_tile_idx in range(down_tiles):
            i_rows = min(_PMAX, shard - i_tile_idx * _PMAX)
            terms.append(
                (
                    down_sb[0:i_rows, i_tile_idx, h_tile_idx * _PMAX : (h_tile_idx + 1) * _PMAX],
                    activation_T[0:i_rows, i_tile_idx, :],
                )
            )
        return terms

    partial_hT = _project_to_hT(down_terms, tokens, _tile(sources, "out", (_PMAX, tokens, _H_TILES), nl.bfloat16))
    return _all_reduce_select_rows(partial_hT, row_select, replica_tokens, world_group, f"{name}_out", sources, dp_size)


def _copy(dst, src, name: str):
    """DMA ``src`` to ``dst`` with the critical priority on trn3 (none on trn2)."""
    nisa.dma_copy(dst=dst, src=src, priority=dma_priority(_CRITICAL_DMA_PRIORITY), name=name)


@functools.cache
def dense_ffn_world(world_size: int):
    """Return the dense SiTU MLP body (layer 0) of a multi-host world of ``world_size`` ranks.

    The body has the call form of ``tkg_moe.dense_mlp_decode``:
    ``body(x_hT, dense_gate_up, dense_down, rank_info, dp_group, world_group, tag, sources=None, dp_size=dp)``.
    Every rank gathers the ``dp * B`` tokens of the replicas over the DP group in HBM, computes its shard of
    ``33792 / world`` intermediate columns for all of them, all-reduces the output over the world in HBM, and keeps
    the B rows of its replica.

    Dimensions:
        B: tokens of each replica, ``dp * B <= 128``. I: the intermediate shard, ``33792 / world`` (132 at 256 ranks).

    Weights (``ffn_world_torch.dense_ffn_world_rank_weights``): ``dense_gate_up`` ``[128, 56, 2 I]`` BF16 hT-tiled (the gate shard,
    then the up shard), ``dense_down`` ``[I, 7168]`` BF16 ``[in, out]``. Rank ``g`` owns the columns
    ``[I g, I g + I)``.

    Returns:
        A body with ``token_major = True``; it returns the ``[128, B, 56]`` BF16 SBUF output of the replica.
    """
    layout = world_layout(world_size)

    def body(
        x_hT: nl.NkiTensor,
        dense_gate_up: nl.NkiTensor,
        dense_down: nl.NkiTensor,
        rank_info: nl.NkiTensor,
        dp_group: ReplicaGroup,
        world_group: ReplicaGroup,
        tag: str,
        sources: dict | None = None,
        dp_size: int = layout["dp"],
    ) -> nl.NkiTensor:
        return _dense_ffn(
            x_hT, dense_gate_up, dense_down, rank_info, dp_group, world_group, tag, sources, dp_size, layout
        )

    body.token_major = True
    body.world_size = world_size
    return body


@functools.cache
def dense_stand_in_moe(world_size: int):
    """Return a MoE body that runs the dense MLP of ``dense_ffn_world`` on the MoE weight slots 0 and 1.

    The body has the call form of a MoE body of ``decode_block.decode_block_body``:
    ``body(h, gate_up, down, *other_slots, rank_info, dp_group, world_group, tag, sources=..., dp_size=dp)``, with the
    dense layout of the world in slots 0 and 1 (``dense_gate_up``, ``dense_down``). It drops the Trn3 expert-loop
    options (``TRN3_EXPERT_LOOP_KEYS``) and fails on any other option. A stand-in for compile and bring-up runs of an
    entry without its MoE body, not the model.
    """
    dense = dense_ffn_world(world_size)
    layout = world_layout(world_size)

    def body(x_hT, gate_up, down, *slots, sources=None, dp_size=layout["dp"], **options):
        unknown = sorted(set(options) - TRN3_EXPERT_LOOP_KEYS)
        kernel_assert(not unknown, f"The dense stand-in takes no MoE option of the Trn3 bodies, got {unknown}")
        *_, rank_info, dp_group, world_group, tag = slots
        return dense(
            x_hT, gate_up, down, rank_info, dp_group, world_group, f"{tag}_moe", sources=sources, dp_size=dp_size
        )

    body.token_major = True
    body.world_size = world_size
    return body
