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

"""Kimi-K3 multi-server decode dense MLP (layer 0) with one DP x gather and one world output all-reduce for all rows.

The weights, the layout and the math are those of ``tkg_ffn_world_cc.dense_ffn_world``: rank g computes the intermediate
columns ``[I g, I g + I)`` of the dense SiTU MLP, ``I = 33792 / W``, for the T = dp B tokens of all DP replicas. Here
each collective runs once for all rows: the DP x gather of the B rows of every replica and the world all-reduce of the
output of all T tokens, both in persistent HBM (``dense_collective_hbm``). The matmuls take chunks of at most 128 of the
gathered tokens (the stationary columns and the PSUM partitions), and each chunk stores its part of the all-reduce
source. The rank loads the rows of its replica with the runtime DP-rank offset (``tkg_moe._load_replica_rows``).
"""

import functools

import nki.collectives as ncc
import nki.isa as nisa
import nki.language as nl
from nki.collectives import ReplicaGroup

from ..utils.kernel_assert import kernel_assert
from ..common import check_pair_group, pair_shared_buffer
from ..tkg_weight_dma_gate import release_weight_gate
from ..tkg_moe import (
    _DENSE_GATE_UP_DMA_CHUNKS,
    _H_TILES,
    _PMAX,
    HIDDEN_SIZE,
    _load_hT_rows,
    _load_replica_rows,
    _project_to_hT,
    _psum_bank,
    _situ,
    rank_selectors,
    replica_index,
)
from ..tkg_moe_dp import _HBM_DMA
from .tkg_ffn_world_cc import ATTN_TP, _copy, replica_groups_world, world_layout
from .tkg_moe_2srv import token_chunks

DENSE_X_GATHERS = ("dp", "sharded")
"""The forms of the x gather of the dense body (``dense_x_gather``): ``dp`` (default) gathers this replica's whole
normalized hidden over the DP group (each of the 32 TP ranks of a replica sends the same bytes); ``sharded`` gathers
1 / 32 of it over the DP group (TP rank t sends partitions ``t + 32 q``, q = 0 .. 3, of the [128, B 56] hT), then the
32 TP ranks of a replica all-gather those slices inside the host, so the cross-server traffic drops 32x. The gathered
values are the same bytes, in another row order of ``x_dst`` (row ``(t dp + r) 4 + q``); the chunk loads take one DMA
for each q (partitions ``32 q .. 32 q + 31``), and the matmuls read the same chunks."""
_SLICE_ROWS = _PMAX // ATTN_TP
"""Partitions of the [128, B 56] hT that one TP rank sends in the ``sharded`` x gather."""
DENSE_SOURCES_KEY = "dense_2srv"
"""The key of the persistent HBM tensors of the dense body in the kernel sources (``dense_collective_hbm``)."""
SPLIT_RIDER_KEY = "mla_split_rider"
"""The key of the long-sequence split's rider in the kernel sources (``verify_model_2srv`` ``mla_split``): a dict with
``pairs`` (a list of ``(src [1, W], dst [dp, W])`` INT32 HBM pairs) and ``after`` (a callable or None). The DP x gather
of layer 0 takes each ``src`` as one more buffer (one coalesced all-gather) into its ``dst``, row r = DP replica r, and
calls ``after()`` right after the gather."""


def dense_collective_hbm(
    replica_rows: int, world_size: int, name: str = "", pair_shared: tuple = (), x_gather: str = "dp"
) -> dict:
    """Return the persistent HBM tensors of the dense body for B rows of each replica.

    ``x_src`` ``[128, B * 56]`` and ``x_dst`` ``[dp * 128, B * 56]``: the DP x gather (the replicas stack on rows).
    With ``x_gather`` ``sharded`` (``DENSE_X_GATHERS``) also ``x_slice`` ``[4, B * 56]`` (this TP rank's partitions
    ``t + 32 q``) and
    ``x_mid`` ``[dp * 4, B * 56]`` (the DP gather of the slices); ``x_dst`` then holds the TP32 gather of ``x_mid``.
    ``out_src`` and ``out_dst`` ``[128, dp * B * 56]``: the world output all-reduce, token ``r B + b`` at columns
    ``56 (r B + b) .. + 56``. Allocate them once for the kernel. ``pair_shared`` with ``moe_out``
    (``common.PAIR_SHARED_CC_DST_SITES``) puts ``out_dst`` in LNC1 shared HBM; () keeps every tensor private.
    """
    dp = world_layout(world_size)["dp"]
    cols = replica_rows * _H_TILES
    kernel_assert(x_gather in DENSE_X_GATHERS, f"dense_x_gather must be one of {DENSE_X_GATHERS}, got {x_gather!r}")
    shapes = {
        "x_src": (_PMAX, cols),
        "x_dst": (dp * _PMAX, cols),
        "out_src": (_PMAX, dp * cols),
        "out_dst": (_PMAX, dp * cols),
        **({"x_slice": (_SLICE_ROWS, cols), "x_mid": (dp * _SLICE_ROWS, cols)} if x_gather == "sharded" else {}),
    }
    return {
        key: nl.ndarray(
            shape,
            dtype=nl.bfloat16,
            buffer=pair_shared_buffer(key == "out_dst" and "moe_out" in pair_shared),
            name=f"dense2s{name}_{key}",
        )
        for key, shape in shapes.items()
    }


@functools.cache
def dense_ffn_2srv(world_size: int, x_gather: str = "dp"):
    """Return the dense body of layer 0 for a multi-server world, with the call form of ``decode_block_body``.

    The block calls ``body(x_hT, dense_gate_up, dense_down, rank_info, dp_group, world_group, tag, sources=...,
    dp_size=dp)`` with the weights of ``ffn_world_torch.dense_ffn_world_rank_weights``; ``sources`` holds this body's
    tensors under ``DENSE_SOURCES_KEY`` (``dense_collective_hbm`` with the same ``x_gather``, ``DENSE_X_GATHERS``).
    """
    layout = world_layout(world_size)
    kernel_assert(x_gather in DENSE_X_GATHERS, f"dense_x_gather must be one of {DENSE_X_GATHERS}, got {x_gather!r}")

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
        kernel_assert(dp_size == layout["dp"], f"dp_size {dp_size} is not world / 32 = {layout['dp']}")
        kernel_assert(
            sources is not None and DENSE_SOURCES_KEY in sources, f"The dense body needs sources[{DENSE_SOURCES_KEY!r}]"
        )
        return dense_mlp_2srv(
            x_hT,
            dense_gate_up,
            dense_down,
            rank_info,
            dp_group,
            world_group,
            tag,
            sources[DENSE_SOURCES_KEY],
            layout,
            rider=sources.get(SPLIT_RIDER_KEY),
            **({"x_gather": x_gather, "tp_group": replica_groups_world(world_size)[0]} if x_gather != "dp" else {}),
        )

    body.token_major = True
    body.world_size = world_size
    body.x_gather = x_gather
    return body


def dense_mlp_2srv(
    x_hT: nl.NkiTensor,
    dense_gate_up: nl.NkiTensor,
    dense_down: nl.NkiTensor,
    rank_info: nl.NkiTensor,
    dp_group: ReplicaGroup,
    world_group: ReplicaGroup,
    tag: str,
    hbm: dict,
    layout: dict,
    rider: dict = None,
    x_gather: str = "dp",
    tp_group: ReplicaGroup = None,
) -> nl.NkiTensor:
    """Run the dense SiTU MLP for the tokens of all DP replicas and return this replica's rows.

    Dimensions:
        B: rows of each replica, 1 to 128. T: gathered tokens, dp B. I: the intermediate shard, ``33792 / W``.

    Args:
        x_hT (nl.NkiTensor): [128, B, 56] BF16 SBUF, this replica's normalized hidden, token-major hT.
        dense_gate_up (nl.NkiTensor): [128, 56, 2 I] BF16 HBM, hT-tiled: the gate shard, then the up shard.
        dense_down (nl.NkiTensor): [I, 7168] BF16 HBM, ``[in, out]``.
        rank_info (nl.NkiTensor): [1, 4] INT32 HBM, (global rank, DP rank, TP rank, 0).
        dp_group, world_group (ReplicaGroup): the DP groups and the world (``tkg_ffn_world_cc.replica_groups_world``).
        tag (str): Unique per-layer string. It names the collectives and the DMAs.
        hbm (dict): the tensors of ``dense_collective_hbm`` for B rows.
        layout (dict): ``tkg_ffn_world_cc.world_layout`` of the world.
        rider (dict): None, or the long-sequence split's rider (``SPLIT_RIDER_KEY``): the DP x gather also gathers
            each ``src`` of ``rider["pairs"]`` into its ``dst`` (one coalesced all-gather), then calls
            ``rider["after"]()``.
        x_gather (str): ``DENSE_X_GATHERS``. ``dp`` (default) keeps the program. ``sharded``: the DP gather moves
            this TP rank's 4 partitions (``x_slice``, a copy of ``x_src`` rows t + 32 q at the runtime TP-rank offset),
            and a TP32 all-gather over ``tp_group`` fills ``x_dst`` with the slices of all ranks (rows in
            [TP rank][DP rank][q] order). The same bytes reach the matmuls.
        tp_group (ReplicaGroup): the TP32 groups (``tkg_ffn_world_cc.replica_groups_world``), with ``sharded`` only.

    Returns:
        [128, B, 56] BF16 SBUF, the dense MLP output of this replica's rows.
    """
    rows = x_hT.shape[1]
    dp, shard = layout["dp"], layout["dense_shard"]
    tokens = dp * rows
    cols = rows * _H_TILES
    sharded = x_gather == "sharded"
    kernel_assert(
        x_gather in DENSE_X_GATHERS and (tp_group is not None) == sharded,
        f"dense_x_gather must be one of {DENSE_X_GATHERS} (sharded with the TP32 group), got {x_gather!r}",
    )
    _validate_inputs(x_hT, dense_gate_up, dense_down, rank_info, hbm, layout, x_gather)
    if rider is not None:
        _check_rider(rider, dp)
    name = f"k3mlp2s_{tag}"

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
    # The rank selectors give the runtime DP-rank offset of the replica-rows load.
    rank_selectors(rank_info, f"{name}_rank", dp_size=dp)

    gather_x(x_hT, hbm, rank_info, dp_group, name, rider=rider, tp_group=tp_group if sharded else None)
    for chunk_idx, (start, count) in enumerate(token_chunks(tokens)):
        chunk_name = f"{name}_c{chunk_idx}"
        x_chunk = load_x_chunk(hbm, start, count, rows, dp, chunk_name, sharded=sharded)
        partial_hT = _dense_chunk(x_chunk.permute((0, 2, 1)), gate_up_sb, down_sb, shard, count)
        nisa.dma_copy(
            dst=hbm["out_src"][:, start * _H_TILES : (start + count) * _H_TILES],
            src=partial_hT,
            name=f"{chunk_name}_out_store",
            **_HBM_DMA,
        )
    check_pair_group(hbm["out_dst"], world_group, f"{name}_out_reduce")
    release_weight_gate(
        ncc.all_reduce(
            srcs=[hbm["out_src"]],
            dsts=[hbm["out_dst"]],
            replica_group=world_group,
            op=nl.add,
            name=f"{name}_out_reduce",
        )
    )
    return _load_replica_rows(hbm["out_dst"], replica_index(rank_info), rows, f"{name}_out_load")


def gather_x(
    x_hT: nl.NkiTensor,
    hbm: dict,
    rank_info: nl.NkiTensor,
    dp_group: ReplicaGroup,
    name: str,
    rider: dict = None,
    tp_group: ReplicaGroup = None,
):
    """Store this replica's [128, B, 56] BF16 hT ``x_hT`` to ``hbm["x_src"]`` and gather all replicas into
    ``hbm["x_dst"]``: one DP gather of ``x_src`` (the ``dp`` form), or with ``tp_group`` (the ``sharded`` form,
    ``DENSE_X_GATHERS``) the DP gather of this TP rank's 4 partitions (``x_slice`` -> ``x_mid``) and the TP32 gather of
    ``x_mid``. The rider (``SPLIT_RIDER_KEY``) rides in the DP gather."""
    cols = x_hT.shape[1] * _H_TILES
    nisa.dma_copy(dst=hbm["x_src"], src=x_hT.reshape((_PMAX, cols)), name=f"{name}_x_store", **_HBM_DMA)
    sharded = tp_group is not None
    if sharded:
        # This TP rank's 4 partitions t + 32 q of x_src: rows of stride 32, from row t (the runtime TP-rank offset).
        nisa.dma_copy(
            dst=hbm["x_slice"],
            src=hbm["x_src"].ap(
                pattern=[[ATTN_TP * cols, _SLICE_ROWS], [1, cols]],
                offset=0,
                scalar_offset=_tp_index(rank_info, name),
                indirect_dim=0,
            ),
            name=f"{name}_x_slice",
            **_HBM_DMA,
        )
    # A coalesced collective moves one dtype: the INT32 rider buffers ride as their BF16 views (the same bytes).
    pairs = rider["pairs"] if rider is not None else []
    riders = ([src.view(nl.bfloat16) for src, _ in pairs], [dst.view(nl.bfloat16) for _, dst in pairs])
    ncc.all_gather(
        srcs=[hbm["x_slice" if sharded else "x_src"], *riders[0]],
        dsts=[hbm["x_mid" if sharded else "x_dst"], *riders[1]],
        replica_group=dp_group,
        collective_dim=0,
        name=f"{name}_x_gather",
    )
    if rider is not None and rider.get("after") is not None:
        rider["after"]()
    if sharded:
        ncc.all_gather(
            srcs=[hbm["x_mid"]],
            dsts=[hbm["x_dst"]],
            replica_group=tp_group,
            collective_dim=0,
            name=f"{name}_x_tp_gather",
        )


def load_x_chunk(hbm: dict, start: int, count: int, rows: int, dp: int, name: str, sharded: bool = False):
    """Load the gathered tokens [start, start + count) (the replica-major order: the rows of each replica of the chunk)
    from ``hbm["x_dst"]`` and return them as [128, count, 56] BF16 SBUF; ``sharded``: ``x_dst`` holds the TP32 gather
    of the slices (``gather_x``; row ``(t dp + r) 4 + q`` holds partition ``t + 32 q`` of replica r), read with one DMA
    for each q (partitions ``32 q .. 32 q + 31``) over the replicas of the chunk."""
    cols = rows * _H_TILES
    x_chunk = nl.ndarray((_PMAX, count, _H_TILES), dtype=nl.bfloat16, buffer=nl.sbuf)
    first_replica, last_replica = start // rows, (start + count - 1) // rows
    if sharded and start % rows == 0 and count % rows == 0:
        # Whole replicas: partition 32 q + t, token r B + b <- x_dst row (t dp + r) 4 + q, column 56 b + h.
        for q in range(_SLICE_ROWS):
            nisa.dma_copy(
                dst=x_chunk[q * ATTN_TP : (q + 1) * ATTN_TP, :, :],
                src=hbm["x_dst"].ap(
                    pattern=[[dp * _SLICE_ROWS * cols, ATTN_TP], [_SLICE_ROWS * cols, count // rows], [1, cols]],
                    offset=(first_replica * _SLICE_ROWS + q) * cols,
                ),
                name=f"{name}_x_load_q{q}",
                **_HBM_DMA,
            )
        return x_chunk
    for replica_idx in range(first_replica, last_replica + 1):
        first = max(start, replica_idx * rows)
        last = min(start + count, (replica_idx + 1) * rows)
        col0, col1 = (first - replica_idx * rows) * _H_TILES, (last - replica_idx * rows) * _H_TILES
        if not sharded:
            nisa.dma_copy(
                dst=x_chunk[:, first - start : last - start, :],
                src=hbm["x_dst"][replica_idx * _PMAX : (replica_idx + 1) * _PMAX, col0:col1],
                name=f"{name}_x_load_r{replica_idx}",
                **_HBM_DMA,
            )
            continue
        for q in range(_SLICE_ROWS):
            nisa.dma_copy(
                dst=x_chunk[q * ATTN_TP : (q + 1) * ATTN_TP, first - start : last - start, :],
                src=hbm["x_dst"].ap(
                    pattern=[[dp * _SLICE_ROWS * cols, ATTN_TP], [1, col1 - col0]],
                    offset=(replica_idx * _SLICE_ROWS + q) * cols + col0,
                ),
                name=f"{name}_x_load_r{replica_idx}_q{q}",
                **_HBM_DMA,
            )
    return x_chunk


def _dense_chunk(
    x_tokens_hT: nl.NkiTensor, gate_up_sb: nl.NkiTensor, down_sb: nl.NkiTensor, shard: int, tokens: int
) -> nl.NkiTensor:
    """Run the dense MLP shard on ``tokens`` gathered tokens ([128, 56, T] view, T <= 128); return the [128, T * 56]
    BF16 partial, as ``tkg_ffn_world_cc.dense_ffn_world`` does for its tokens."""
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

    down_tiles = down_sb.shape[1]
    full_down_tiles = shard // _PMAX
    down_tail_rows = shard - full_down_tiles * _PMAX
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

    return _project_to_hT(down_terms, tokens)


def _tp_index(rank_info: nl.NkiTensor, name: str) -> nl.NkiTensor:
    """Return the INT32 [1, 1] SBUF TP rank of this rank (rank_info column 2): the runtime block offset of the
    ``sharded`` x slice."""
    tp = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=tp, src=rank_info.ap(pattern=[[0, 1], [1, 1]], offset=2), name=f"{name}_tp_rank", **_HBM_DMA)
    return tp


def _check_rider(rider: dict, dp: int):
    """Check the split rider of the DP x gather: ``pairs`` of ``src`` [1, W] and ``dst`` [dp, W] INT32."""
    kernel_assert(set(rider) <= {"pairs", "after"} and rider.get("pairs"), "rider: pairs (non-empty), after")
    for src, dst in rider["pairs"]:
        kernel_assert(
            len(src.shape) == 2 and src.shape[0] == 1 and tuple(dst.shape) == (dp, src.shape[1]),
            f"rider: src [1, W] and dst [{dp}, W], got {tuple(src.shape)} and {tuple(dst.shape)}",
        )
        kernel_assert(src.dtype == nl.int32 and dst.dtype == nl.int32, "rider: INT32 src and dst")


def _validate_inputs(x_hT, dense_gate_up, dense_down, rank_info, hbm: dict, layout: dict, x_gather: str = "dp"):
    """Check the inputs and the HBM tensors of the dense body."""
    kernel_assert(
        nl.program_ndim() == 0 or (nl.program_ndim() == 1 and nl.num_programs(0) == 1),
        "The Kimi-K3 multi-server dense body supports LNC1 only",
    )
    kernel_assert(
        len(x_hT.shape) == 3 and x_hT.shape[0] == _PMAX and x_hT.shape[2] == _H_TILES and x_hT.dtype == nl.bfloat16,
        f"x_hT must be BF16 [{_PMAX}, B, {_H_TILES}], got {x_hT.dtype} {x_hT.shape}",
    )
    rows = x_hT.shape[1]
    kernel_assert(1 <= rows <= _PMAX, f"x_hT has B={rows} rows. Use 1 <= B <= {_PMAX}.")
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
    cols = rows * _H_TILES
    shapes = {
        "x_src": (_PMAX, cols),
        "x_dst": (layout["dp"] * _PMAX, cols),
        "out_src": (_PMAX, layout["dp"] * cols),
        "out_dst": (_PMAX, layout["dp"] * cols),
        **(
            {"x_slice": (_SLICE_ROWS, cols), "x_mid": (layout["dp"] * _SLICE_ROWS, cols)}
            if x_gather == "sharded"
            else {}
        ),
    }
    for key, shape in shapes.items():
        kernel_assert(
            key in hbm and tuple(hbm[key].shape) == shape,
            f"The dense body needs the HBM tensor {key} {shape} of dense_collective_hbm({rows} rows)",
        )
