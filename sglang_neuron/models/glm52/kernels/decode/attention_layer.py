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

"""One GLM-5.2 DSA attention layer (projections, indexer, MLA attention, output) in a single trace, collectives
included, so the collectives interleave with the matmuls instead of bracketing separate launches::

    AG[hidden] -> down_proj -> AG[latents] -> q/kv_up -> RS[seq]
        -> indexer: q_proj -> AG[q] -> kv_score -> topk_local -> AG[cand] -> merge -> AG[sel]
        -> MLA attention -> AG[lse] -> cp_combine -> RS[ctx] -> out_proj -> RS[out]

The sub-kernels live under ``attention/``; this module is the wiring.

Sharding. A DP group's ranks factor as ``cp_degree x head_shard x token_shard``: cp shards the keys (every rank sees
every token and a share of the context), head_shard the heads and token_shard the tokens (each rank owns its own
sequences). The served points use ``head_shard = 1``: MLA's cache is head-independent, so head sharding only
replicates the cache and puts the head count on the poor axis of both attention matmuls. The output path is sharded
over the CP group by heads anyway, which is where head sharding's weight savings come from. With fewer
``out_absorb_w`` heads than ``Nh / cp_degree`` the output path shards over a larger block group
(:func:`_out_proj_block_sharded`); with fewer ``wq_a`` / ``wkv_a`` columns the down-projections shard over a latent
group (:func:`_exchange_latent_pair`). Entry and exit are token-sharded across all ranks, the layout the MoE layer
consumes and produces.

One program for every rank. The kernel is traced once and the NEFF runs on every rank, so a per-rank Python scalar
would silently take rank 0's value everywhere. The per-rank scalars the core kernels take are therefore removed:

* the sequence slice is a ReduceScatter with ``op=maximum`` over identical operands, an exact scatter (an
  ``all_to_all`` on the strided sequence group aborts in the runtime);
* ``key_base`` is 0: the residency threshold rides ``idx_mask_add`` and the local positions get ``key_offset``, a
  ``[1, 1]`` tensor, which the local select folds into its live labels;
* ``cp_rank`` is 0: every position this rank sees is shifted by ``(cp_degree - cp_rank) * block_size``, which turns
  the block-ownership test into a test against 0; the caller supplies the shifted positions and a block table
  shifted by one column (column 0 holds -1);
* the CP combine is inlined (:func:`_cp_softmax_combine`), using this rank's own ``lse`` tensor instead of a static
  ``cp_rank`` slice.

``spec`` is padded to 8: the exit's two ReduceScatters need a token count of 128, so each sequence gets an eighth
speculative slot at the next position, whose output is discarded. Eight is also the indexer's limit (16 partitions
per query row).

The layer norms ``x_shard`` itself before the entry AllGather rather than in the down-projection, because the
indexer reads the same normed rows. ``input_layernorm`` uses ``rms_norm_eps`` (1e-5); the q/kv latent norms and the
indexer's LayerNorm use 1e-6, hence three epsilon arguments.
"""

import os

import nki
import nki.collectives as ncc
import nki.isa as nisa
import nki.language as nl
from nki.collectives import ReplicaGroup

from .attention.indexer.indexer_block_rows import indexer_block_rows
from .attention.indexer.indexer_constants import (
    IDX_CAND_DIRECT,
    IDX_CAND_HWDGE,
    IDX_SELECT_VEC,
    indexer_cache_is_blocked,
    indexer_cache_keys,
    local_topk_chunked,
    snake_score_layout,
)
from .attention.indexer.indexer_kv_score import indexer_key_prologue, indexer_kv_score
from .attention.indexer.indexer_q_proj import indexer_q_proj
from .attention.indexer.indexer_residency_mask import indexer_residency_mask
from .attention.mla.mla_decode_attention import ATTN_Q_HOOK, mla_decode_attention, take_rank_major_heads
from .attention.mla.mla_decode_constants import P_MAX, MlaDecodeShape
from .attention.mla.mla_decode_down_proj import mla_decode_down_proj, mla_decode_hidden_k_tiles
from .attention.mla.mla_decode_out_proj import mla_decode_out_proj, mla_decode_out_proj_sbuf
from .attention.mla.mla_decode_q_kv_up import mla_decode_q_block_exchange, mla_decode_q_block_finish, mla_decode_q_kv_up
from .attention.mla.mla_proj_primitives import linear, load_gamma_broadcast, rmsnorm_inplace
from .attention.mla.prior_index import build_prior_index, export_prior_index
from .utils.cc_dst_ring import cc_dst
from .utils.cc_pin import pin_attn
from .utils.kernel_assert import kernel_assert
from .utils.row_layout import spread_rows_shape
from .utils.scratch_hbm import layer_scratch_hbm, scratch_hbm
from .utils.small_loads import small_load
from .utils.trace_names import unique_name

_ERROR_PREFIX = "[GLM-5.2 DSA Attention Decode Layer DP/CP/KVDP Kernel]"

# Buffer class for every HBM tensor that is a collective operand. At LNC=1 the RUNTIME rejects a
# collective whose sources are shared_hbm, so both ends are private_hbm: reduce_scatter still enforces
# the strict rule (every src AND dst one buffer type), which rules out mixing. The sub-kernels whose
# outputs are gathered directly -- mla_decode_attention's lse, mla_decode_out_proj, indexer_q_proj,
# indexer_topk_local_packed -- return private_hbm for the same reason.
_COLLECTIVE_HBM = nl.private_hbm

_SNAKE_PARTITIONS = 16
"""Partitions ``nisa.topk`` devotes to one query row. Mirrors ``indexer_constants.SNAKE_PARTITIONS``."""

_A2A_CELL_ALIGN = 8
"""bf16 elements (16 bytes) each head cell of the fused combine's all_to_all payload is padded to."""


ROPE_TABLES_ONCE = os.environ.get("GLM52K_ROPE_TABLES_ONCE", "0") == "1"
"""Gather the block's RoPE ``cos`` / ``sin`` tables once per kernel and reuse them in every layer. The tables depend
on the step's positions only, and every layer of the giga kernel passes the same tensors, so the per-layer table
AllGather of :func:`_gather_block_latent` (``2 -> 128 KB`` fp32 over the block group) repeats identical bytes. Off:
one table gather per layer (the default)."""

Q_NORM_INDEXER_ONLY = os.environ.get("GLM52K_Q_NORM_INDEXER_ONLY", "0") == "1"
"""Under the block-sharded query path, normalize and store this rank's own query latent only in a layer that runs
the indexer (its only reader); a shared layer skips both."""

Q_BLOCK_AG_FIRST = os.environ.get("GLM52K_Q_BLOCK_AG_FIRST", "0") == "1"
"""Issue the block query-latent AllGather (and its tables) BEFORE the latent-pair exchange. Both read the
down-projection's output; collectives complete in issue order, and the query path (up-projection, the 576 KB q
all_to_all) is the longer chain, so its gather does not wait behind the pair's. Off: pair first (the default)."""

Q_BLOCK_SPLIT_EXCHANGE = os.environ.get("GLM52K_Q_BLOCK_SPLIT_EXCHANGE", "0") == "1"
"""Block-sharded query path: the ``cp`` ranks of a token slice (one token group, on one chip) attend the same tokens,
so the block ``all_to_all`` delivers every block to them ``cp`` times. Instead hand each of them only its ``T / cp``
tokens of every head (an ``all_to_all`` of a ``cp``-th of the bytes over the block group) and complete the slice with
an AllGather over the token group. The same bytes reach the same rows: bit-identical. Off: one ``all_to_all``."""

ATTN_COMBINE_RECV_QUEUE = os.environ.get("GLM52K_ATTN_COMBINE_RECV_QUEUE", "")
"""The served fused CP combine's receive (one HBM -> SBUF DMA per slice of the ``all_to_all`` result,
:func:`_out_proj_block_fused_combine`) on a queue of its own instead of the default (Sync) one. The decode's weight
stream keeps the Sync queue busy for the whole layer, and the queue is in order, so on it the receive waits behind the
weight descriptors already queued (a post-``all_to_all`` gap in each layer at 128k and at 1M, with the Sync
queue busy through it). ``act``: every slice on the Scalar engine's hardware DGE queue; ``pool``: on the GpSimd software
DGE. The same bytes land in the same SBUF cells: bit-identical. Empty (default): the default queue.

The two-stage combine (:func:`_out_proj_block_two_stage_combine`, the 1M point) stages its CP ``all_to_all`` result the
same way: its context halves on the Sync and ACT queues (``LAT_COMBINE_LSE_FIRST`` / ``LAT_COMBINE_TS``), or one DMA on
the default queue. The knob moves every one of those context loads (``act``: the Scalar engine's hardware DGE; ``pool``:
the GpSimd software DGE); the lse slots stay on the ACT queue. At 1M the combine's first multiply (the
``ctx_r * w_r`` scaling) waits in each layer with no compute or collective running (64k x 32, MTP K7)."""
if ATTN_COMBINE_RECV_QUEUE not in ("", "act", "pool"):
    raise ValueError(f"GLM52K_ATTN_COMBINE_RECV_QUEUE={ATTN_COMBINE_RECV_QUEUE!r}: set act or pool, or leave it unset")


def _combine_recv_queue(engine=None) -> dict:
    """``nisa.dma_copy`` keywords of one CP combine receive load (HBM -> SBUF after the combine's ``all_to_all``): the
    knob's queue (``ATTN_COMBINE_RECV_QUEUE``), else ``engine``'s hardware DGE queue, else none (the default queue)."""
    if ATTN_COMBINE_RECV_QUEUE == "act":
        return dict(dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.scalar)
    if ATTN_COMBINE_RECV_QUEUE == "pool":
        return dict(dge_mode=nisa.dge_mode.swdge)
    return {} if engine is None else dict(dge_mode=nisa.dge_mode.hwdge, engine=engine)

CP_COMBINE_TWO_STAGE = os.environ.get("GLM52K_CP_COMBINE_TWO_STAGE", "0") == "1"
"""Fused CP combine on the block-sharded exit: combine the ``cp`` key partials of a
token slice on its own chip, then send only the combined context over the block group
(:func:`_out_proj_block_two_stage_combine`). An ``all_to_all`` over the CP group hands each CP rank all partials of
its ``T / cp`` tokens, it combines them with the receiver's arithmetic of :func:`_out_proj_block_fused_combine`, and
an ``all_to_all`` of the combined bf16 context (a ``cp``-th of the tokens, no lse cells) over the block group lands
the block's tokens token-major. Same values, same summation order: bit-identical. Off: one ``all_to_all``."""

LAT_QBLOCK_SBUF = os.environ.get("GLM52K_LAT_QBLOCK_SBUF", "0")
"""Block-sharded query path, shared layers (no indexer): hand ``mla_decode_q_kv_up`` the block's query latent in the
SBUF tile :func:`_gather_block_latent` assembles it in, instead of storing it to HBM and loading it back (two DMA hops
on the q chain). Same bytes, bit-identical. Off (0) by default. (Also splitting the assembling gather read,
64-byte runs, into 4 DMAs over the Sync and ACT queues measured no gain.)"""

LAT_DOWNPROJ_AG_SRC = os.environ.get("GLM52K_LAT_DOWNPROJ_AG_SRC", "0") == "1"
"""Block-sharded query path: the down-projection stores its query shard straight into the block AllGather's source
too (``mla_decode_down_proj(q_out_extra=)``), instead of the gather copying it out of the latent-pair source with an
HBM -> HBM DMA after the store. Same bytes, bit-identical. Off by default."""

_ATTN_RS = []
"""The block-sharded exit ReduceScatter's destination, published (``take_attn_rs``) so the giga stack hands it to the
next MoE layer, whose norm forms ``residual + attn_out`` from the two HBM tensors instead of reloading the stored sum."""


def take_attn_rs():
    """The last block-sharded exit ReduceScatter destination, or None."""
    return _ATTN_RS.pop() if _ATTN_RS else None


LAT_COMBINE_LSE_FIRST = os.environ.get("GLM52K_LAT_COMBINE_LSE_FIRST", "0") == "1"
"""Two-stage CP combine: read the received cells' lse slots back first (a small DMA of their own), then the context
values, so the CP softmax weights (max, exp, the ACT table switch, sum, reciprocal) run while the 512 KB of
context is still landing. Same bytes, bit-identical. Off by default."""

LAT_COMBINE_TS = os.environ.get("GLM52K_LAT_COMBINE_TS", "0") == "1"
""":func:`_combine_cp_cells`: scale each CP partial with a per-partition ``tensor_scalar`` (one input stream) instead of
one broadcast ``tensor_tensor`` over all partials (two), and write the last fp32 add straight to the bf16 result
instead of adding into fp32 and casting after (the add's fp32 result is rounded to bf16 on write either way). The
two-stage combine's receive also splits over the Sync and ACT queues. Same arithmetic per element, bit-identical.
Off by default."""

# id(cos), id(sin), group, shapes -> (cos, sin, cos_block, sin_block); the entry holds the tables, so an id
# cannot be reused while it is cached. Cleared at the start of every giga-kernel trace.
_ROPE_TABLE_CACHE: dict = {}


def reset_rope_table_cache() -> None:
    """Forget the gathered RoPE tables (``ROPE_TABLES_ONCE``); call at the start of a kernel trace."""
    _ROPE_TABLE_CACHE.clear()


def _pin_all(*tensors):
    """Pin every collective operand while the attention pins are on (long context; see ``utils/cc_pin.py``)."""
    pin_attn(*tensors)


def _all_gather_columns(
    src_hbm: nl.NkiTensor,
    replica_group: ReplicaGroup,
    group_size: int,
    tokens: int,
    width: int,
    name: str,
) -> nl.NkiTensor:
    """AllGather a ``[tokens, width]`` COLUMN shard and concatenate on the FEATURE axis.

    An HBM collective only concatenates on dim 0, so this gathers on the token axis -- landing rank
    ``r``'s block at rows ``[tokens*r, tokens*(r+1))`` -- and then re-lays it with a single strided
    DMA, because rank ``r``'s columns belong at ``[width*r, width*(r+1))``. The DMA reads ``(t, r, c)``
    with strides ``(width, tokens*width, 1)``, which is the transpose of the gather's two leading axes
    and nothing more.

    Args:
        src_hbm (nl.NkiTensor): [tokens, width] this rank's column slice in HBM.
        replica_group (ReplicaGroup): The group whose ranks hold the other slices, IN COLUMN ORDER.
        group_size (int): Ranks in that group.
        tokens (int): Rows, at most 128 (they ride the repack DMA's partition axis).
        width (int): This rank's column count.
        name (str): Debug name stem for the two allocations.

    Returns:
        nl.NkiTensor: [tokens, group_size * width] in HBM, the complete tensor.
    """
    gathered = cc_dst((tokens * group_size, width), src_hbm.dtype, _COLLECTIVE_HBM, f"{name}_rows")
    _pin_all(src_hbm, gathered)  # while pinning is on; see utils/cc_pin.py
    ncc.all_gather(srcs=[src_hbm], dsts=[gathered], replica_group=replica_group, collective_dim=0)

    # One HBM -> HBM DMA does the relayout; a bounce through SBUF (HBM -> SB -> HBM) would put two
    # DMA latencies on the critical path where one suffices.
    wide = nl.ndarray(
        (tokens, group_size * width), dtype=src_hbm.dtype, buffer=_COLLECTIVE_HBM, name=unique_name(f"{name}_wide")
    )
    nisa.dma_copy(
        dst=wide.reshape((tokens, group_size, width)),
        src=gathered.ap(pattern=[[width, tokens], [tokens * width, group_size], [1, width]], offset=0),
    )
    return wide


def _gather_latent_pair(
    packed: nl.NkiTensor,
    replica_group: ReplicaGroup,
    group_size: int,
    tokens: int,
    q_width: int,
    kv_width: int,
) -> tuple[nl.NkiTensor, nl.NkiTensor]:
    """AllGather the (query latent | cache row) column pair and split it back into two complete tensors.

    Rank ``r``'s pair lands at rows ``[tokens*r, tokens*(r+1))`` of the gather; its query columns belong
    at ``[q_width*r, ...)`` of the latent and its cache-row columns at ``[kv_width*r, ...)`` of the row, so
    each output is one strided HBM -> HBM DMA reading ``(t, r, c)`` with strides ``(pair, tokens*pair, 1)``.
    Replaces ``_all_gather_columns`` and a separate split on this path: the same result in 2 DMAs that ran
    in parallel instead of 6 in three dependent rounds (gather -> wide -> split).

    Args:
        packed (nl.NkiTensor): [tokens, q_width + kv_width] this rank's pair in HBM (a collective source).
        replica_group (ReplicaGroup): The group whose ranks hold the other column slices, IN COLUMN ORDER.
        group_size (int): Ranks in that group.
        tokens (int): Row count.
        q_width (int): One rank's query-latent columns.
        kv_width (int): One rank's cache-row columns.

    Returns:
        tuple[nl.NkiTensor, nl.NkiTensor]: (q_latent [tokens, group_size*q_width], kv_row
            [tokens, group_size*kv_width]), both complete and in HBM.
    """
    pair = q_width + kv_width
    gathered = cc_dst((tokens * group_size, pair), packed.dtype, _COLLECTIVE_HBM, "latent_pair_rows")
    _pin_all(packed, gathered)  # while pinning is on; see utils/cc_pin.py
    ncc.all_gather(srcs=[packed], dsts=[gathered], replica_group=replica_group, collective_dim=0)
    return _split_gathered_pair(gathered, group_size, tokens, q_width, kv_width)


def _split_gathered_pair(
    gathered: nl.NkiTensor, group_size: int, tokens: int, q_width: int, kv_width: int
) -> tuple[nl.NkiTensor, nl.NkiTensor]:
    """Split ``group_size`` row blocks of ``(query latent | cache row)`` column pairs into the two complete tensors.

    Block ``r`` (rows ``[tokens*r, tokens*(r+1))``) holds group position ``r``'s columns, so each output is one
    strided HBM -> HBM DMA reading ``(t, r, c)`` with strides ``(pair, tokens*pair, 1)``.

    Args:
        gathered (nl.NkiTensor): [group_size * tokens, q_width + kv_width] in HBM.
        group_size (int): Column-shard degree.
        tokens (int): Rows of one block.
        q_width (int): One rank's query-latent columns.
        kv_width (int): One rank's cache-row columns.

    Returns:
        tuple[nl.NkiTensor, nl.NkiTensor]: (q_latent [tokens, group_size*q_width], kv_row
            [tokens, group_size*kv_width]), both complete and in HBM.
    """
    pair = q_width + kv_width
    q_latent = nl.ndarray(
        (tokens, group_size * q_width), dtype=gathered.dtype, buffer=_COLLECTIVE_HBM, name=unique_name("q_latent_full")
    )
    kv_row = nl.ndarray(
        (tokens, group_size * kv_width), dtype=gathered.dtype, buffer=_COLLECTIVE_HBM, name=unique_name("kv_row_full")
    )
    nisa.dma_copy(
        dst=q_latent.reshape((tokens, group_size, q_width)),
        src=gathered.ap(pattern=[[pair, tokens], [tokens * pair, group_size], [1, q_width]], offset=0),
    )
    nisa.dma_copy(
        dst=kv_row.reshape((tokens, group_size, kv_width)),
        src=gathered.ap(pattern=[[pair, tokens], [tokens * pair, group_size], [1, kv_width]], offset=q_width),
    )
    return q_latent, kv_row


def _exchange_latent_pair(
    packed: nl.NkiTensor,
    replica_group: ReplicaGroup,
    group_size: int,
    token_group: int,
    tokens: int,
    q_width: int,
    kv_width: int,
) -> tuple[nl.NkiTensor, nl.NkiTensor]:
    """The latent-sharded down-projection's exchange: each rank receives its own token slice's complete
    (query latent | cache row), split back into the two tensors.

    ``packed`` holds this rank's ``q_width + kv_width`` columns for every token of the latent group, slice-major
    (``group_size / token_group`` slices of ``tokens`` rows). Group position ``q`` needs slice ``q // token_group``, so
    the all-to-all source repeats each slice for its ``token_group`` ranks (a zero stride), and receive block ``r`` is
    this rank's slice in rank ``r``'s columns, the layout :func:`_gather_latent_pair` lands. The destination is the
    group position, so no per-rank index is needed.

    Args:
        packed: [slices * tokens, q_width + kv_width] this rank's pair for the group's tokens, in HBM.
        replica_group: the latent group, ``group_size`` consecutive ranks in column order.
        group_size, token_group: ranks in the group and in one token slice.
        tokens: tokens of one slice, at most 128.
        q_width, kv_width: one rank's query-latent and cache-row columns.

    Returns:
        ``(q_latent [tokens, group_size * q_width], kv_row [tokens, group_size * kv_width])`` for this rank's slice,
        in HBM.
    """
    pair = q_width + kv_width
    slices = group_size // token_group
    send = nl.ndarray(
        (group_size * tokens, pair), dtype=packed.dtype, buffer=_COLLECTIVE_HBM, name=unique_name("latent_send")
    )
    for s in range(slices):
        nisa.dma_copy(
            dst=send.ap(
                pattern=[[pair, tokens], [tokens * pair, token_group], [1, pair]],
                offset=s * token_group * tokens * pair,
            ),
            src=packed.ap(pattern=[[pair, tokens], [0, token_group], [1, pair]], offset=s * tokens * pair),
        )
    recv = cc_dst((group_size * tokens, pair), packed.dtype, _COLLECTIVE_HBM, "latent_recv")
    _pin_all(send, recv)  # while pinning is on; see utils/cc_pin.py
    ncc.all_to_all(srcs=[send], dsts=[recv], replica_group=replica_group, collective_dim=0)
    return _split_gathered_pair(recv, group_size, tokens, q_width, kv_width)


def _gather_block_latent(
    q_latent_shard: nl.NkiTensor,
    cos: nl.NkiTensor,
    sin: nl.NkiTensor,
    block_replica_group: ReplicaGroup,
    block_degree: int,
    token_group: int,
    tokens: int,
    width: int,
    name: str,
    latent_shards: int = None,
    src_written: nl.NkiTensor = None,
    sbuf_out: bool = False,
) -> tuple[nl.NkiTensor, nl.NkiTensor, nl.NkiTensor]:
    """AllGather the query latent's column shards and the RoPE tables over the block group and assemble the block's
    tokens: one dim-0 gather, then one re-laying DMA per token slice (as :func:`_all_gather_columns` does for one
    slice). The fp32 tables take a second gather on the same group, since a coalesced collective has one dtype.

    Args:
        q_latent_shard: this rank's column slice of the query latent (in its first ``width`` columns), HBM.
        cos, sin: [tokens, rope / 2] fp32 RoPE tables.
        block_replica_group, block_degree, token_group, tokens, width: the group and extents.
        name: DMA name stem.
        latent_shards: the latent's column-shard count under a latent-sharded down-projection (None: the token group).
        src_written: an optional gather source the caller already wrote (``LAT_DOWNPROJ_AG_SRC``).
        sbuf_out: return the latent as the SBUF tile it is assembled in (``LAT_QBLOCK_SBUF``).

    Returns:
        The block's query latent ``[slices * tokens, shards * width]`` and its ``cos`` / ``sin`` tables.
    """
    slices = block_degree // token_group
    shards = latent_shards or token_group
    shard_tokens = tokens * (shards // token_group)  # tokens of one rank's latent columns
    per_group = shards // token_group  # token slices of one shard group
    half = cos.shape[1]
    dtype = q_latent_shard.dtype
    # private_hbm for the collective sources: the down-projection's outputs and the tables are kernel IO.
    if src_written is not None:
        kernel_assert(
            tuple(src_written.shape) == (shard_tokens, width),
            f"{_ERROR_PREFIX} src_written must be [{shard_tokens}, {width}], got {src_written.shape}",
        )
        src = src_written
    else:
        src = nl.ndarray((shard_tokens, width), dtype=dtype, buffer=_COLLECTIVE_HBM, name=unique_name(f"{name}_src"))
        nisa.dma_copy(dst=src, src=q_latent_shard[0:shard_tokens, 0:width])
    key = (id(cos), id(sin), block_degree, token_group, tokens, half)
    cached = _ROPE_TABLE_CACHE.get(key) if ROPE_TABLES_ONCE else None
    if cached is not None and (cached[0] is not cos or cached[1] is not sin):
        cached = None
    gathered = nl.ndarray(
        (shard_tokens * block_degree, width), dtype=dtype, buffer=_COLLECTIVE_HBM, name=unique_name(f"{name}_rows")
    )
    if cached is None:
        tab_src = nl.ndarray(
            (tokens, 2 * half), dtype=nl.float32, buffer=_COLLECTIVE_HBM, name=unique_name(f"{name}_tab_src")
        )
        nisa.dma_copy(dst=tab_src[0:tokens, 0:half], src=cos[0:tokens, 0:half])
        nisa.dma_copy(dst=tab_src[0:tokens, half : 2 * half], src=sin[0:tokens, 0:half])
        tab_rows = nl.ndarray(
            (tokens * block_degree, 2 * half),
            dtype=nl.float32,
            buffer=_COLLECTIVE_HBM,
            name=unique_name(f"{name}_tab_rows"),
        )
        _pin_all(src, gathered, tab_src, tab_rows)  # while pinning is on; see utils/cc_pin.py
    else:
        _pin_all(src, gathered)
    ncc.all_gather(srcs=[src], dsts=[gathered], replica_group=block_replica_group, collective_dim=0)
    if cached is None:
        ncc.all_gather(srcs=[tab_src], dsts=[tab_rows], replica_group=block_replica_group, collective_dim=0)

    wide_sb = nl.ndarray((slices * tokens, shards, width), dtype=dtype, buffer=nl.sbuf)
    tab_sb = nl.ndarray((slices * tokens, 2 * half), dtype=nl.float32, buffer=nl.sbuf) if cached is None else None
    for g in range(block_degree // shards):
        # Shard group g: ranks [g * shards, (g + 1) * shards), each holding its columns of the group's
        # per_group slices; one DMA lays the group's shard_tokens tokens side by side.
        r0 = g * shards * shard_tokens  # first row of shard group g (its shard 0)
        nisa.dma_copy(
            dst=wide_sb[g * shard_tokens : (g + 1) * shard_tokens, 0:shards, 0:width],
            src=gathered.ap(
                pattern=[[width, shard_tokens], [shard_tokens * width, shards], [1, width]], offset=r0 * width
            ),
        )
        for s in range(g * per_group, (g + 1) * per_group):
            if cached is not None:
                break
            r0 = s * token_group * tokens  # first table row of slice s (its shard 0)
            nisa.dma_copy(
                dst=tab_sb[s * tokens : (s + 1) * tokens, 0 : 2 * half], src=tab_rows[r0 : r0 + tokens, 0 : 2 * half]
            )
    if sbuf_out:
        block = wide_sb.reshape((slices * tokens, shards * width))
    else:
        block = nl.ndarray(
            (slices * tokens, shards * width), dtype=dtype, buffer=_COLLECTIVE_HBM, name=unique_name(f"{name}_wide")
        )
        nisa.dma_copy(dst=block, src=wide_sb.reshape((slices * tokens, shards * width)))
    if cached is not None:
        return block, cached[2], cached[3]
    cos_block = nl.ndarray(
        (slices * tokens, half), dtype=nl.float32, buffer=_COLLECTIVE_HBM, name=unique_name(f"{name}_cos")
    )
    sin_block = nl.ndarray(
        (slices * tokens, half), dtype=nl.float32, buffer=_COLLECTIVE_HBM, name=unique_name(f"{name}_sin")
    )
    nisa.dma_copy(dst=cos_block, src=tab_sb[0 : slices * tokens, 0:half])
    nisa.dma_copy(dst=sin_block, src=tab_sb[0 : slices * tokens, half : 2 * half])
    if ROPE_TABLES_ONCE:
        _ROPE_TABLE_CACHE[key] = (cos, sin, cos_block, sin_block)
    return block, cos_block, sin_block


def build_norm_reduce_ones(T_shard: int, H: int) -> nl.NkiTensor | None:
    """The block-diagonal ones matrix the partition-spread input norm reduces with, or None when the rows do not spread.

    ``[P_MAX, P_MAX]`` fp32, ones where both indices fall in the same token's partition block. Layer-invariant: the
    giga stack builds it once and hands it to every layer (``input_norm_reduce_ones``) so no layer spends
    memsets on it.

    Args:
        T_shard (int): Rows of the residual shard.
        H (int): Row width.

    Returns:
        nl.NkiTensor | None: The matrix in SBUF, or None.
    """
    spread = spread_rows_shape(T_shard, H)
    if spread is None:
        return None
    P, _ = spread
    ppt = P // T_shard
    ones_blk = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.sbuf)
    if ppt == P:
        nisa.memset(ones_blk, 1.0)
    else:
        nisa.memset(ones_blk, 0.0)
        for t in range(T_shard):
            nisa.memset(ones_blk[t * ppt : (t + 1) * ppt, t * ppt : (t + 1) * ppt], 1.0)
    return ones_blk


def _rmsnorm_shard(
    x_shard: nl.NkiTensor,
    input_norm_w: nl.NkiTensor,
    eps: float,
    x_sb: nl.NkiTensor = None,
    reduce_ones: nl.NkiTensor = None,
) -> nl.NkiTensor:
    """Apply the layer's ``input_layernorm`` to this rank's token shard, before the entry AllGather.

    Runs here rather than inside ``mla_decode_down_proj`` because the DSA indexer consumes the SAME
    normed tensor -- see the module docstring.

    Args:
        x_shard (nl.NkiTensor): [T_shard, H] this rank's slots of the raw residual stream.
        input_norm_w (nl.NkiTensor): [1, H] fp32 RMSNorm gamma.
        eps (float): ``config.rms_norm_eps``, 1e-5 -- NOT the 1e-6 the latent norms use.
        x_sb (nl.NkiTensor): Optional SBUF copy of ``x_shard`` in the ``spread_rows_shape`` layout (the giga
            stack's residual add leaves one behind), which skips the load.
        reduce_ones (nl.NkiTensor): Optional ``build_norm_reduce_ones`` result to reuse; None builds one here.

    Returns:
        nl.NkiTensor: [T_shard, H] normed, in HBM, ready to be a collective source.
    """
    T_shard, H = x_shard.shape
    normed = nl.ndarray(
        (T_shard, H), dtype=x_shard.dtype, buffer=_COLLECTIVE_HBM, name=unique_name("hidden_normed_shard")
    )

    spread = spread_rows_shape(T_shard, H)
    if spread is None:
        x_tok = nl.ndarray((P_MAX, H), dtype=x_shard.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=x_tok[0:T_shard, 0:H], src=x_shard[0:T_shard, 0:H])
        rmsnorm_inplace(x_tok[0:P_MAX, 0:H], load_gamma_broadcast(input_norm_w, T_shard), T_shard, eps)
        nisa.dma_copy(dst=normed[0:T_shard, 0:H], src=x_tok[0:T_shard, 0:H])
        return normed

    # Partition-spread: partition p holds elements [p*F, (p+1)*F) of the flattened rows, so token t is
    # partitions [t*ppt, (t+1)*ppt). Same arithmetic as rmsnorm_inplace -- fp32 sum of squares,
    # rsqrt(sum/H + eps), x * rstd * gamma -- with the sum formed as per-partition partials and one
    # block-diagonal ones matmul, so it can differ from the token-major sum in fp32 rounding order only.
    P, F = spread
    ppt = P // T_shard
    if x_sb is None:
        x_sb = nl.ndarray((P, F), dtype=x_shard.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=x_sb, src=x_shard.reshape((P, F)))
    kernel_assert(
        tuple(x_sb.shape) == (P, F) and x_sb.buffer == nl.sbuf,
        f"{_ERROR_PREFIX} x_sb must be a [{P}, {F}] SBUF tile, got {x_sb.shape} in {x_sb.buffer}",
    )

    squares_discard = nl.ndarray((P, F), dtype=nl.float32, buffer=nl.sbuf)
    partial = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation_reduce(dst=squares_discard, op=nl.square, data=x_sb, reduce_op=nl.add, reduce_res=partial)

    ones_blk = reduce_ones if reduce_ones is not None else build_norm_reduce_ones(T_shard, H)
    total_ps = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=total_ps, stationary=ones_blk, moving=partial)
    inv_rms = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=inv_rms, op=nl.rsqrt, data=total_ps, scale=1.0 / H, bias=eps)

    # gamma in the same layout: every token's block of partitions sees the whole [1, H] row.
    gamma_sb = nl.ndarray((P, F), dtype=nl.float32, buffer=nl.sbuf)
    for t in range(T_shard):
        small_load(dst=gamma_sb[t * ppt : (t + 1) * ppt, 0:F], src=input_norm_w[0:1, 0:H].reshape((ppt, F)))

    normed_sb = nl.ndarray((P, F), dtype=x_shard.dtype, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(
        dst=normed_sb, data=x_sb, op0=nl.multiply, operand0=inv_rms[0:P, 0:1], op1=nl.multiply, operand1=gamma_sb
    )
    nisa.dma_copy(dst=normed.reshape((P, F)), src=normed_sb)
    return normed


def build_shared_prior_index(
    gather_indices_true: nl.NkiTensor,
    cp_pos_offset: nl.NkiTensor,
    kv_cache: nl.NkiTensor,
    block_table: nl.NkiTensor,
    positions: nl.NkiTensor,
    T_local: int,
    num_seqs_local: int,
    spec: int,
    index_topk: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    cp_degree: int,
    io_dtype,
) -> tuple[nl.NkiTensor, nl.NkiTensor, nl.NkiTensor]:
    """Resolve one selection to this rank's prior-index tables once, for every layer that shares it.

    The tables (local cache rows, the snake-permuted gather index and the key-position mask operands) depend on the
    selection, ``block_table`` and the CP shift, not on a layer's weights or cache, so the shared layers of an
    index-share group would each rebuild the same bytes. This applies the layer's CP shift and the
    core's ``build_prior_index`` and stages the result for ``glm52_dsa_attention_decode_layer_kernel(prior_snake_rows,
    prior_key_pos, prior_natural_rows)``.

    Args:
        gather_indices_true: [T_local, index_topk] int32 global positions, the full layer's ``gather_indices_out``.
        cp_pos_offset: [1, 1] int32 per-rank CP position shift.
        kv_cache: any layer's cache in the group; only its shape is read.
        block_table: [num_seqs_local, table_width] int32 this rank's table.
        positions: [T_local] int32 shifted query positions; only its shape is read.
        T_local, num_seqs_local, spec, index_topk, kv_lora_rank, qk_rope_head_dim, cp_degree: shapes.
        io_dtype: the attention io dtype.

    Returns:
        ``(snake_hbm, key_pos_hbm, natural_hbm)`` in ``shared_hbm``, each ``[P_MAX, T_local, index_topk // P_MAX]``.
    """
    gather_3d = gather_indices_true.reshape((num_seqs_local, spec, index_topk))
    # The same extents ``get_mla_decode_shape`` derives inside the core, from the same operands. Only
    # the fields ``build_prior_index`` reads matter (block geometry, CP, token counts); the head count
    # and dtype are recorded for completeness and never used on this path.
    block_size = kv_cache.shape[2]
    kernel_assert(positions.shape[0] == T_local, f"positions must be [{T_local}], got {positions.shape}")
    block_shift = 0
    while (1 << block_shift) < block_size:
        block_shift += 1
    mla = MlaDecodeShape(
        num_seqs=num_seqs_local,
        spec=spec,
        num_heads=1,
        kv_lora_dim=kv_lora_rank,
        qk_rope_dim=qk_rope_head_dim,
        num_prior=index_topk,
        num_blocks=kv_cache.shape[0],
        block_size=block_size,
        max_blocks=block_table.shape[1],
        block_shift=block_shift,
        cp_degree=cp_degree,
        cp_rank=0,
        dtype=io_dtype,
    )
    # cp_pos_offset is folded into the table build (live slots only), see build_prior_index.
    snake_rows, key_pos_f, natural_rows = build_prior_index(gather_3d, block_table, mla, pos_offset=cp_pos_offset)
    return export_prior_index(snake_rows, key_pos_f, natural_rows, name="shared_prior_index")


def _cp_softmax_combine(
    ctx_local: nl.NkiTensor,
    lse_local: nl.NkiTensor,
    lse_all: nl.NkiTensor,
    tokens: int,
    num_heads: int,
    latent_dim: int,
    cp_degree: int,
) -> nl.NkiTensor:
    """Scale this rank's partial context by its softmax weight over the CP ranks.

    ``mla_cp_combine``'s arithmetic, except that the numerator comes from ``lse_local`` rather than
    ``lse_all[cp_rank]``, since a static slice by rank cannot be rank-invariant:

        w_r = exp(lse_r - m) / sum_s exp(lse_s - m),   m = max_s lse_s

    It never exponentiates a positive argument, so it cannot overflow; an empty rank's sentinel ``lse`` gives weight
    0, and all-empty gives a finite uniform weight. The CP ranks' key sets must be disjoint (not asserted): a key on
    two ranks would be counted twice by the ReduceScatter that follows.

    Args:
        ctx_local: [T, Nh, C] this rank's partial context, normalized by its own denominator.
        lse_local: [T, Nh] fp32 this rank's log-sum-exp.
        lse_all: [cp_degree * T, Nh] fp32 the all-gather of every rank's ``lse``.
        tokens, num_heads, latent_dim, cp_degree: T, Nh, C and the CP degree.

    Returns:
        [Nh, T * C] the weighted context in HBM, head-major because the ReduceScatter(sum) that combines it also
        scatters the head axis, and an HBM collective splits dim 0 only.
    """
    T, Nh, C = tokens, num_heads, latent_dim

    """
    THE PARTITION AXIS IS THE LARGER OF Nh AND T, and that choice matters at the production
    sharding.

    A Vector or Scalar instruction costs ``elements_per_partition`` cycles and is INDEPENDENT of how many
    partitions it uses, so the cheapest layout is the one that spreads the work over the most lanes. The
    widest tensor here is the context scale over ``Nh * T * C`` elements; putting the SHORTER axis on
    partitions leaves most lanes idle and multiplies the cost by the ratio. At ``head_shard = 1`` that is
    ``T = 16`` against ``Nh = 64``, so head-major runs 8192 elements deep on 64 lanes instead of 32768
    deep on 16 -- a quarter of the depth.

    It is also an SBUF constraint, not merely a cost one: the free width is ``other_axis * C``, and at
    ``head_shard = 8`` (``T = 128``, ``Nh = 8``) head-major would need 128 KB per partition for each of
    the two context tiles against a 240 KB budget -- it fails to compile with NCC_IBIR229. Choosing by
    size satisfies both at once.

    The arithmetic is identical either way -- every op is elementwise or a reduction over the CP axis,
    which stays innermost and contiguous -- so this is bit-exact whichever branch is taken.
    """
    head_major = Nh >= T
    rows, cols = (Nh, T) if head_major else (T, Nh)

    # lse_all is [cp, T, Nh] flattened; lse_local is [T, Nh]; ctx_local is [T, Nh, C]. Each pattern below
    # is just those strides read in the chosen partition order.
    lse_all_pat = [[1, Nh], [Nh, T], [T * Nh, cp_degree]] if head_major else [[Nh, T], [1, Nh], [T * Nh, cp_degree]]
    lse_own_pat = [[1, Nh], [Nh, T]] if head_major else [[Nh, T], [1, Nh]]
    ctx_pat = [[C, Nh], [Nh * C, T], [1, C]] if head_major else [[Nh * C, T], [C, Nh], [1, C]]

    lse_sb = nl.ndarray((P_MAX, cols, cp_degree), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=lse_sb[0:rows, 0:cols, 0:cp_degree], src=lse_all.ap(pattern=lse_all_pat, offset=0))
    own_sb = nl.ndarray((P_MAX, cols), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=own_sb[0:rows, 0:cols], src=lse_local.ap(pattern=lse_own_pat, offset=0))

    max_sb = nl.ndarray((P_MAX, cols, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(max_sb[0:rows, 0:cols, 0:1], op=nl.max, data=lse_sb[0:rows, 0:cols, 0:cp_degree], axis=2)

    shifted_sb = nl.ndarray((P_MAX, cols, cp_degree), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=shifted_sb[0:rows, 0:cols, 0:cp_degree],
        data1=lse_sb[0:rows, 0:cols, 0:cp_degree],
        data2=max_sb[0:rows, 0:cols, 0:1].broadcast(2, cp_degree),
        op=nl.subtract,
    )
    exp_sb = nl.ndarray((P_MAX, cols, cp_degree), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(exp_sb[0:rows, 0:cols, 0:cp_degree], op=nl.exp, data=shifted_sb[0:rows, 0:cols, 0:cp_degree])

    sum_sb = nl.ndarray((P_MAX, cols, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(sum_sb[0:rows, 0:cols, 0:1], op=nl.add, data=exp_sb[0:rows, 0:cols, 0:cp_degree], axis=2)
    recip_sb = nl.ndarray((P_MAX, cols, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(recip_sb[0:rows, 0:cols, 0:1], op=nl.reciprocal, data=sum_sb[0:rows, 0:cols, 0:1])

    # The numerator: exp(my own lse - m). Every argument is <= 0, so this can only underflow.
    own_shift_sb = nl.ndarray((P_MAX, cols), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=own_shift_sb[0:rows, 0:cols], data1=own_sb[0:rows, 0:cols], data2=max_sb[0:rows, 0:cols, 0], op=nl.subtract
    )
    weight_sb = nl.ndarray((P_MAX, cols, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(weight_sb[0:rows, 0:cols, 0], op=nl.exp, data=own_shift_sb[0:rows, 0:cols])
    nisa.tensor_tensor(
        dst=weight_sb[0:rows, 0:cols, 0:1],
        data1=weight_sb[0:rows, 0:cols, 0:1],
        data2=recip_sb[0:rows, 0:cols, 0:1],
        op=nl.multiply,
    )

    # The innermost run of both context patterns is C elements -- 1 KB descriptors in bfloat16, and the
    # runs walk HBM forwards, which is the good case rather than the strided one.
    ctx_sb = nl.ndarray((P_MAX, cols, C), dtype=ctx_local.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=ctx_sb[0:rows, 0:cols, 0:C], src=ctx_local.ap(pattern=ctx_pat, offset=0))
    scaled_sb = nl.ndarray((P_MAX, cols, C), dtype=ctx_local.dtype, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=scaled_sb[0:rows, 0:cols, 0:C],
        data1=ctx_sb[0:rows, 0:cols, 0:C],
        data2=weight_sb[0:rows, 0:cols, 0:1].broadcast(2, C),
        op=nl.multiply,
    )

    # The result must be HEAD-MAJOR whichever way it was computed: the ReduceScatter that follows scatters
    # the HEAD axis and an HBM collective splits dim 0 only. Head-major already is one contiguous copy;
    # token-major needs one strided DMA, writing element (t, h, c) at h*(T*C) + t*C + c.
    scaled = nl.ndarray(
        (Nh, T * C), dtype=ctx_local.dtype, buffer=_COLLECTIVE_HBM, name=unique_name("scaled_ctx_head_major")
    )
    if head_major:
        nisa.dma_copy(dst=scaled[0:Nh, 0 : T * C], src=scaled_sb[0:Nh, 0:T, 0:C])
    else:
        nisa.dma_copy(dst=scaled.ap(pattern=[[C, T], [T * C, Nh], [1, C]], offset=0), src=scaled_sb[0:T, 0:Nh, 0:C])
    return scaled


def _numel(shape) -> int:
    """Element count of a shape tuple."""
    n = 1
    for d in shape:
        n *= int(d)
    return n


def _deliver_attn_out(attn_out: nl.NkiTensor, summed_hbm: nl.NkiTensor) -> None:
    """Move the exit ReduceScatter's result into the layer's output.

    A collective's destination cannot be a kernel IO tensor, so a shared_hbm output is reached through an
    SBUF bounce (two DMAs). When the caller handed in an SBUF output -- the giga stack does, so that its
    residual add reads the rows straight from SBUF -- the load IS the delivery and the store is skipped.

    Args:
        attn_out (nl.NkiTensor): [T_shard, H] the output, in SBUF or shared HBM.
        summed_hbm (nl.NkiTensor): [T_shard, H] the collective's destination in private HBM.
    """
    if attn_out.buffer == nl.sbuf:
        # Any SBUF shape with the same element count: the giga stack hands in the partition-spread tile
        # its residual add works on (see utils/row_layout.py).
        nisa.dma_copy(dst=attn_out, src=summed_hbm.reshape(tuple(attn_out.shape)))
        return
    out_sb = nl.ndarray(tuple(attn_out.shape), dtype=attn_out.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=out_sb, src=summed_hbm)
    nisa.dma_copy(dst=attn_out, src=out_sb)


def _scatter_rows(
    src_hbm: nl.NkiTensor,
    replica_group: ReplicaGroup,
    degree: int,
    rows_out: int,
    name: str,
) -> nl.NkiTensor:
    """Hand each rank of ``replica_group`` its own ``rows_out`` rows of a tensor every member holds identically.

    A ReduceScatter with ``op=maximum`` over identical operands is an exact scatter (position ``s`` gets rows
    ``[rows_out * s, rows_out * (s + 1))``) with no per-rank offset, which a kernel compiled once from rank 0's inputs
    cannot have. ``op=add`` would not be exact in bf16.

    Args:
        src_hbm: [rows_out * degree, W] in HBM, identical on every member.
        replica_group: the group, in row-block order.
        degree: group size; 1 hands ``src_hbm`` through.
        rows_out: rows this rank receives.
        name: debug name.

    Returns:
        [rows_out, W] in HBM.
    """
    width = src_hbm.shape[1]
    if degree == 1:
        # Every rank of a one-rank group already holds exactly its rows: hand the source through. The
        # copy (HBM -> HBM, one DMA latency on the indexer's critical path) would buy
        # nothing but a fresh buffer.
        kernel_assert(src_hbm.shape[0] == rows_out, f"{_ERROR_PREFIX} {name}: degree 1 needs rows_out == rows")
        return src_hbm
    dst = cc_dst((rows_out, width), src_hbm.dtype, _COLLECTIVE_HBM, name, reduce_scatter=True)
    _pin_all(src_hbm, dst)  # while pinning is on; see utils/cc_pin.py
    ncc.reduce_scatter(srcs=[src_hbm], dsts=[dst], op=nl.maximum, replica_group=replica_group, collective_dim=0)
    return dst


def _gather_rows(
    src_hbm: nl.NkiTensor,
    replica_group: ReplicaGroup,
    degree: int,
    name: str,
) -> nl.NkiTensor:
    """AllGather on dim 0, with the degenerate one-rank case short-circuited to the source itself.

    Args:
        src_hbm (nl.NkiTensor): [rows, W] this rank's block in HBM.
        replica_group (ReplicaGroup): The group, in increasing block order.
        degree (int): Group size. 1 returns ``src_hbm`` unchanged.
        name (str): Debug name.

    Returns:
        nl.NkiTensor: [rows * degree, W] in HBM.
    """
    rows, width = src_hbm.shape
    if degree == 1:
        # A one-rank gather is the identity: hand the source through rather than copy it (see _scatter_rows).
        return src_hbm
    dst = cc_dst((rows * degree, width), src_hbm.dtype, _COLLECTIVE_HBM, name)
    _pin_all(src_hbm, dst)  # while pinning is on; see utils/cc_pin.py
    ncc.all_gather(srcs=[src_hbm], dsts=[dst], replica_group=replica_group, collective_dim=0)
    return dst


@nki.jit
def glm52_dsa_attention_decode_layer_kernel(
    # ── input: this rank's 128-way token shard of the residual stream ──
    x_shard: nl.NkiTensor,  # [T_shard, H]
    input_norm_w: nl.NkiTensor,  # [1, H] input_layernorm weight
    # ── MLA projections ──
    wq_a: nl.NkiTensor,  # [H, q_lora_shard]   column shard over the TOKEN group, or over the LATENT group (latent_tp)
    wkv_a: nl.NkiTensor,  # [H, kv_row_shard]   column shard, columns ordered [k_pe | kv_latent]
    q_norm_w: nl.NkiTensor,  # [1, q_lora_rank]
    kv_norm_w: nl.NkiTensor,  # [1, C]
    wq_b: nl.NkiTensor,  # [q_lora_rank, Nh * (nope + rope) / q_degree]  COLUMNS over the token group, or the block group
    q_absorb_w: nl.NkiTensor,  # [Nh / q_degree, nope, C]         heads of this rank's wq_b columns
    out_absorb_w: nl.NkiTensor,  # [C, Nh / cp_degree, v_head]         HEAD shard x CP, latent-major
    o_proj_w: nl.NkiTensor,  # [(Nh / cp_degree) * v_head, H]      HEAD shard x CP
    cos: nl.NkiTensor,  # [T_local, rope // 2] fp32, TRUE positions
    sin: nl.NkiTensor,  # [T_local, rope // 2] fp32
    # ── DSA indexer ──
    idx_wq_b: nl.NkiTensor,  # [q_lora_rank, idx_q_shard]  column shard over the key group
    idx_wk: nl.NkiTensor,  # [H, idx_head_dim]           REPLICATED
    idx_k_norm_w: nl.NkiTensor,  # [1, idx_head_dim] fp32
    idx_k_norm_b: nl.NkiTensor,  # [1, idx_head_dim] fp32
    idx_weights_proj: nl.NkiTensor,  # [H, idx_n_heads]
    idx_cos: nl.NkiTensor,  # [spec, idx_rope // 2] fp32, this rank's sequence
    idx_sin: nl.NkiTensor,  # [spec, idx_rope // 2] fp32
    # ── caches and per-step runtime tensors ──
    kv_cache: nl.NkiTensor,  # [num_blocks, 1, block_size, 640] uint8, MUTATED
    kv_new_slot_mapping: nl.NkiTensor,  # [num_seqs_local, spec] int32
    block_table: nl.NkiTensor,  # [num_seqs_local, local_blocks + 1] int32, column 0 unused
    positions: nl.NkiTensor,  # [T_local] int32, CP-SHIFTED
    idx_k_cache: nl.NkiTensor,  # [n_blocks, 1, idx_block_size, idx_head_dim], MUTATED; unpaged is one block
    idx_new_slot_mapping: nl.NkiTensor,  # [spec] int32
    idx_positions: nl.NkiTensor,  # [spec] int32, KEY-SHIFTED
    # ── the two per-rank offsets, tensors because the kernel is compiled once for all ranks ──
    key_offset: nl.NkiTensor,  # [1, 1] int32, key_rank * idx_keys
    cp_pos_offset: nl.NkiTensor,  # [1, 1] int32, (cp_degree - cp_rank) * block_size
    # ── replica groups ──
    token_replica_group: ReplicaGroup,
    cp_replica_group: ReplicaGroup,
    head_replica_group: ReplicaGroup,
    seq_replica_group: ReplicaGroup,
    key_replica_group: ReplicaGroup,
    # ── degrees, all rank-invariant ──
    cp_degree: int,
    head_shard: int,
    token_shard: int,
    key_size: int,
    # ── model scalars ──
    num_heads: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    kv_lora_rank: int,
    idx_n_heads: int,
    idx_rope_dim: int,
    index_topk: int,
    spec: int,
    softmax_scale: float,
    kv_scale: float,
    input_norm_eps: float,
    latent_norm_eps: float,
    idx_norm_eps: float,
    block_replica_group: ReplicaGroup = None,  # the o_proj BLOCK group: cp_degree x slices consecutive ranks; block-sharded o_proj only
    qblock_replica_group: ReplicaGroup = None,  # the query path's block group: token_group x q_slices consecutive ranks; block-sharded query path only
    latent_replica_group: ReplicaGroup = None,  # the down-projection's LATENT group: token_group x latent slices consecutive ranks; latent-sharded wq_a / wkv_a only
    prev_gather_indices: nl.NkiTensor = None,  # [T_local, index_topk] int32 TRUE positions, or None
    prior_snake_rows: nl.NkiTensor = None,  # [P_MAX, T_local, index_topk / P_MAX] uint32, or None
    prior_key_pos: nl.NkiTensor = None,  # [P_MAX, T_local, index_topk / P_MAX] fp32, or None
    prior_natural_rows: nl.NkiTensor = None,  # [P_MAX, T_local, index_topk / P_MAX] uint32, or None
    idx_mask_add: nl.NkiTensor = None,  # [128, idx_keys / 128 * spec] fp32 key-major additive residency mask, or None to build it here
    idx_block_table: nl.NkiTensor = None,  # [1, idx_keys / idx_block_size + 1] int32 logical->physical, or None for a position-ordered cache
    idx_block_rows: nl.NkiTensor = None,  # [128, idx_keys / 128] uint32 from indexer_block_rows, or None to build it from idx_block_table
    # ── the POOLED indexer (idx_pool_degree > 0): this rank scores its key shard of EVERY sequence of its pool ──
    pool_replica_group: ReplicaGroup = None,  # the E ranks at this rank's key position, one per CP group, in slot order
    idx_pool_degree: int = 0,  # E, the CP groups pooled; 0 = the dealt-per-group indexer above
    idx_pool_tokens: int = 0,  # query tokens scored per sequence on the pooled path (1 without MTP, spec with it)
    # ── the CHUNK-DEALT pool (CDP, idx_pool.py deal="chunk") ──
    idx_pool_items: nl.NkiTensor = None,  # [E, 8] int32, this rank's live foreign regions (layout/decode/idx_pool.py)
    idx_pool_n_items: nl.NkiTensor = None,  # [1, 1] int32, the foreign loop's trip count
    idx_pool_region_base: nl.NkiTensor = None,  # [E] int32, the step cut of each region's owner (-1 = empty)
    idx_pool_new_key_rows: nl.NkiTensor = None,  # [E * idx_pool_tokens, 1] int32, this rank's row of each pool new key, -1 = not here
    idx_pool_unrot: nl.NkiTensor = None,  # [E, 2] int32, recv offsets (values, positions) of phase-1 slot q
    output: nl.NkiTensor = None,  # [T_shard, H] destination for attn_out, in SBUF or shared HBM; None allocates a shared_hbm output
    x_shard_sb: nl.NkiTensor = None,  # optional SBUF copy of x_shard in the spread_rows_shape layout, skips the input norm's load
    input_norm_reduce_ones: nl.NkiTensor = None,  # optional build_norm_reduce_ones(T_shard, H) result, built once by a stack
    prior_tables_out: list = None,  # optional: a FULL layer appends the (snake, key_pos, natural) tables it resolved, for the shared layers
    idx_shard_degree: int = 1,  # degree the indexer's keys are DEALT over; 1 = its own contiguous slice
    idx_block_size: int = 0,  # the indexer cache's block size; also the deal unit under a shared table
):
    """One rank's GLM-5.2 DSA attention layer, collectives included (see the module docstring).

    Shapes: ``T_shard`` token slots this rank owns on entry and exit, ``T_local = T_shard * cp_degree * head_shard``
    slots its MLA scores, ``H`` hidden, ``C`` the KV latent rank, ``Nh`` the heads it attends over, ``nh_out`` the heads
    it projects through ``o_proj``.

    Args:
        x_shard: [T_shard, H] this rank's slots of the residual stream; ``x_shard_sb`` is an optional SBUF copy of it
            (the stack's residual add leaves one), and ``input_norm_reduce_ones`` the stack's shared norm constant.
        input_norm_w, q_norm_w, kv_norm_w: fp32 ``input_layernorm``, ``q_a_layernorm`` and ``kv_a_layernorm`` weights.
        wq_a, wkv_a: this rank's columns of the query and KV down-projections. The column-shard degree is read off the
            weights: the token group's size, or a multiple of it over ``latent_replica_group``. ``wkv_a``'s columns are
            ordered ``[k_pe | kv_latent]``.
        wq_b, q_absorb_w: this rank's heads of the query up-projection and the nope half of ``wkv_b`` for them, sharded
            over the token group or, wider, over ``qblock_replica_group``. The shard is on the output, so it is exact.
        out_absorb_w, o_proj_w: the value half of ``wkv_b`` and the rows of ``o_proj`` for this rank's ``nh_out`` output
            heads, sharded over the CP group (``nh_out = Nh / cp_degree``) or over ``block_replica_group``. The
            projection is a partial sum over heads that the exit ReduceScatter completes.
        cos, sin: [T_local, rope // 2] fp32 RoPE tables at the true positions.
        idx_wq_b, idx_wk, idx_k_norm_w, idx_k_norm_b, idx_weights_proj: the indexer's query columns, key projection,
            key LayerNorm and per-head mixing weights.
        idx_cos, idx_sin: [spec, idx_rope // 2] fp32 RoPE tables of this rank's sequence.
        kv_cache: [num_blocks, 1, block_size, 640] uint8 paged latent cache of this rank's own sequences.
        kv_new_slot_mapping: [num_seqs_local, spec] int32 destination rows, negative to skip.
        block_table: [num_seqs_local, local_blocks + 1] int32, shifted one column by the CP position shift (column 0
            negative).
        positions: [T_local] int32 positions plus ``(cp_degree - cp_rank) * block_size``; ``cp_pos_offset`` is that
            shift as a [1, 1] tensor.
        idx_k_cache: [n_blocks, 1, idx_block_size, idx_head_dim] this rank's indexer keys, updated in place.
        idx_new_slot_mapping: [spec] int32 indexer-cache rows, negative to skip.
        idx_positions: [spec] int32, local positions under a contiguous key deal, true positions under an interleaved
            one; ``key_offset`` [1, 1] is the rank term of the local-to-global map.
        idx_shard_degree, idx_block_size: the degree and unit the indexer keys are dealt in (``cp_degree`` and MLA's
            block size let one block table serve both caches).
        idx_mask_add: the step's residency mask (``indexer_residency_mask``), built once per step by the caller.
        idx_block_table, idx_block_rows: the indexer cache's block table and its row table, or None when the cache is
            position-ordered.
        token_replica_group, cp_replica_group, head_replica_group, seq_replica_group, key_replica_group: the groups
            of the entry gather, the CP gathers and exit scatters, the head scatter (unused at ``head_shard = 1``), the
            sequence scatter and selection gather (exact inverses), and the indexer gathers.
        block_replica_group, qblock_replica_group, latent_replica_group: the contiguous groups of the block-sharded
            output path, query path and down-projections, or None. Contiguous because the runtime's ``all_to_all``
            refuses strided groups.
        cp_degree, head_shard, token_shard, key_size: the sharding degrees.
        num_heads, qk_nope_head_dim, qk_rope_head_dim, kv_lora_rank, idx_n_heads, idx_rope_dim, index_topk: model
            dimensions on this rank.
        spec: token slots per sequence, padded to 8.
        softmax_scale, kv_scale: the attention scale and the latent dequantization scale (a power of two).
        input_norm_eps, latent_norm_eps, idx_norm_eps: 1e-5, 1e-6 and 1e-6.
        prev_gather_indices: [T_local, index_topk] int32 in true positions, or None. Given, this is a shared layer: it
            reuses the preceding full layer's selection, skips the indexer, and its indexer inputs may be None.
        prior_snake_rows, prior_natural_rows: the full layer's prior-index tables a shared layer reuses, or None.
        prior_tables_out: a list a full layer appends its resolved prior-index tables to, for the shared layers.
        pool_replica_group, idx_pool_degree, idx_pool_tokens, idx_pool_items, idx_pool_n_items, idx_pool_region_base,
            idx_pool_new_key_rows, idx_pool_unrot: the chunk-dealt indexer pool of the 1M point
            (``layout/decode/idx_pool.py``), or None. Each member scores its own chunk 0 and its live foreign regions, and
            only the phase-1 candidates cross the pool (:func:`_pooled_chunk_indexer`).
        output: optional [T_shard, H] destination for ``attn_out``; an SBUF tile skips its HBM store.

    Returns:
        ``(attn_out, gather_indices_out, kv_cache, idx_k_cache)``: the layer's attention output for this rank's slots;
        the selection for this rank's own sequences in true positions (a shared layer returns
        ``prev_gather_indices``); and the two caches, so their updates are part of the result (``idx_k_cache`` is
        None on a shared layer).
    """
    T_shard, H = x_shard.shape
    token_group = cp_degree * head_shard  # ranks sharing this rank's token slice
    T_local = T_shard * token_group
    num_seqs_local = T_local // spec
    latent_dim = kv_lora_rank
    q_lora_shard, kv_row_shard = wq_a.shape[1], wkv_a.shape[1]
    q_lora_rank = q_norm_w.shape[1]
    # The down-projections' column-shard degree, read off wq_a: the token group, or the LATENT group of
    # latent_slices token slices (latent_replica_group).
    latent_degree = q_lora_rank // q_lora_shard
    latent_slices = latent_degree // token_group
    # A "shared" layer (``config.indexer_types[layer_idx] == "shared"``) has no indexer submodule at
    # all in the reference, so it carries no indexer weights and no indexer key cache -- every one of
    # those arguments is None and the selection arrives as ``prev_gather_indices`` instead. The branch
    # is on a Python value that is identical on all 128 ranks, so it constant-folds safely.
    run_indexer = prev_gather_indices is None
    idx_q_shard = idx_wq_b.shape[1] if run_indexer else 0
    k_cols = index_topk // _SNAKE_PARTITIONS
    # Heads this rank projects THROUGH o_proj, as opposed to the num_heads it attends over. The output
    # path is head-sharded a further cp_degree ways on top of head_shard, so it is read off the weight
    # rather than derived -- the weight is the authority on what the caller actually handed us.
    nh_out = out_absorb_w.shape[1]

    _validate_attention_layer(
        T_shard=T_shard,
        T_local=T_local,
        num_seqs_local=num_seqs_local,
        spec=spec,
        cp_degree=cp_degree,
        head_shard=head_shard,
        token_shard=token_shard,
        key_size=key_size,
        index_topk=index_topk,
        idx_q_shard=idx_q_shard,
        idx_n_heads=idx_n_heads,
        idx_head_dim=idx_wk.shape[1] if run_indexer else 0,
        positions_len=positions.shape[0],
        num_heads=num_heads,
        nh_out=nh_out,
        o_proj_rows=o_proj_w.shape[0],
        v_head_dim=out_absorb_w.shape[2],
        token_replica_group=token_replica_group,
        head_replica_group=head_replica_group,
        key_replica_group=key_replica_group,
        run_indexer=run_indexer,
    )

    # ═══════ C1: input norm on this rank's rows, then rebuild THIS RANK'S TOKEN SLICE ═══════
    # Over the token group, not the whole DP group: a rank only needs the tokens it will score, and
    # under token sharding that is T_local rather than every slot in the group. A latent-sharded
    # down-projection gathers its LATENT group's rows instead, and a full layer gathers its token slice as
    # well, for the indexer: the slice sits at a per-rank offset of the latent group's rows, which a kernel
    # compiled once for every rank cannot slice (see the module docstring).
    kernel_assert(
        q_lora_shard * latent_degree == q_lora_rank
        and kv_row_shard * latent_degree == qk_rope_head_dim + kv_lora_rank
        and latent_degree % token_group == 0,
        f"{_ERROR_PREFIX} wq_a [H, {q_lora_shard}] and wkv_a [H, {kv_row_shard}] must hold the same 1/latent_degree "
        f"column shard of q_lora_rank={q_lora_rank} and of the {qk_rope_head_dim + kv_lora_rank}-wide cache row, "
        f"latent_degree the token group ({token_group}) or a multiple of it; got latent_degree={latent_degree}",
    )
    normed_shard = _rmsnorm_shard(
        x_shard, input_norm_w, input_norm_eps, x_sb=x_shard_sb, reduce_ones=input_norm_reduce_ones
    )
    if latent_slices == 1:
        hidden = _gather_rows(normed_shard, token_replica_group, token_group, "hidden_all")
        hidden_latent = hidden
    else:
        kernel_assert(
            latent_replica_group is not None
            and head_shard == 1
            and token_shard % latent_slices == 0
            and latent_slices * T_local <= P_MAX,
            f"{_ERROR_PREFIX} wq_a holds 1/{latent_degree} of the query-latent columns, a latent-sharded "
            f"down-projection over {latent_slices} token slices: needs the latent_replica_group ({latent_degree} "
            f"consecutive ranks), head_shard == 1 (got {head_shard}), latent_slices | token_shard={token_shard} "
            f"and latent_slices * T_local <= {P_MAX}",
        )
        # Aligned too (g[0] % latent_degree == 0): then each group's token_group sub-blocks are whole token groups
        # in token-slice order, which _exchange_latent_pair's slice math and the q-block containment rely on.
        kernel_assert(
            all(
                len(g) == latent_degree
                and g[0] % latent_degree == 0
                and list(g) == list(range(g[0], g[0] + latent_degree))
                for g in latent_replica_group._value
            ),
            f"{_ERROR_PREFIX} latent_replica_group must be consecutive, aligned blocks of {latent_degree} ranks "
            f"(each starting at a multiple of {latent_degree}), got {latent_replica_group._value}",
        )
        hidden_latent = _gather_rows(normed_shard, latent_replica_group, latent_degree, "hidden_latent")
        hidden = _gather_rows(normed_shard, token_replica_group, token_group, "hidden_all") if run_indexer else None

    # ═══════ Down-projections, then C2: one collective moves both latents ═══════
    # The query path's shard degree, read off wq_b: the token group (its members hold the same tokens,
    # so the absorbed query is AllGathered inside the kernel), or the block group -- q_slices token
    # slices x the token group -- whose members hold the BLOCK's tokens. For the latter, one more
    # AllGather over the block group assembles the whole block's query latent (this rank's own slice is
    # among them), and the kernel's all_to_all hands every rank its own tokens back.
    q_degree = num_heads * (qk_nope_head_dim + qk_rope_head_dim) // wq_b.shape[1]
    idx_front, idx_prologue, idx_q_gathered = False, None, None
    q_slices = q_degree // token_group
    q_latent_block = cos_block = sin_block = None
    if q_slices > 1:
        kernel_assert(
            qblock_replica_group is not None
            and q_degree % token_group == 0
            and head_shard == 1
            and token_shard % q_slices == 0,
            f"{_ERROR_PREFIX} wq_b holds 1/{q_degree} of the query columns, a block-sharded query path over "
            f"{q_slices} token slices: needs the qblock_replica_group ({q_degree} consecutive ranks), head_shard "
            f"== 1 (got {head_shard}) and q_slices | token_shard={token_shard}",
        )
        kernel_assert(
            q_degree % latent_degree == 0,
            f"{_ERROR_PREFIX} the query block group ({q_degree} ranks) must hold whole latent groups "
            f"({latent_degree} ranks)",
        )
    # The down-projection writes both shards straight into the collective source (1 DMA hop each) -- for the
    # latent group's latent_slices * T_local tokens under a latent shard -- and the collective's two outputs
    # are split off with one strided DMA each.
    latent_pair_src = nl.ndarray(
        (latent_slices * T_local, q_lora_shard + kv_row_shard),
        dtype=hidden_latent.dtype,
        buffer=_COLLECTIVE_HBM,
        name=unique_name("latent_pair"),
    )
    # The indexer front, on a full layer with one sequence a rank: its key side (projection, norm, RoPE, cache write,
    # head-sum weights; indexer_key_prologue) runs on the down-projection's hidden K-tiles right after the latent
    # AllGather is issued, and its query projection on the q-norm's K-tiles inside mla_decode_q_kv_up, so the query
    # gather is issued before wq_b and the head gather. The scoring, the local top-k and the candidate gather are
    # traced right before wq_b and the merge after it. Same values in the same order as the plain path.
    idx_front = run_indexer and num_seqs_local == 1 and spec == T_local
    # The indexer's keys project this rank's token slice: the down-projection's own hidden K-tiles, unless
    # the down-projection runs on the latent group's rows.
    hidden_tiles = [] if idx_front and latent_slices == 1 else None
    # GLM52K_LAT_DOWNPROJ_AG_SRC: the block gather's source, written by the down-projection itself.
    q_ag_src = None
    if LAT_DOWNPROJ_AG_SRC and q_slices > 1:
        q_ag_src = nl.ndarray(
            (latent_slices * T_local, q_lora_shard),
            dtype=hidden_latent.dtype,
            buffer=_COLLECTIVE_HBM,
            name=unique_name("q_block_src"),
        )
    mla_decode_down_proj(
        hidden_states=hidden_latent,
        wq_a=wq_a,
        wkv_a=wkv_a,
        input_norm_w=None,
        norm_eps=input_norm_eps,
        output=latent_pair_src,
        hidden_tiles_out=hidden_tiles,
        q_out_extra=q_ag_src,
    )

    def gather_block():
        # The block's query latent, from the down-projection's own columns (the collective source's query half).
        return _gather_block_latent(
            latent_pair_src,
            cos,
            sin,
            qblock_replica_group,
            q_degree,
            token_group,
            T_local,
            q_lora_shard,
            "q_block",
            latent_shards=latent_degree if latent_slices > 1 else None,
            src_written=q_ag_src,
            # GLM52K_LAT_QBLOCK_SBUF: shared layers only (an indexer layer would hold the tile across its scoring)
            sbuf_out=LAT_QBLOCK_SBUF == "1" and not run_indexer,
        )

    if q_slices > 1 and Q_BLOCK_AG_FIRST:
        q_latent_block, cos_block, sin_block = gather_block()
    if latent_slices == 1:
        q_latent, kv_row = _gather_latent_pair(
            latent_pair_src, token_replica_group, token_group, T_local, q_lora_shard, kv_row_shard
        )
    else:
        q_latent, kv_row = _exchange_latent_pair(
            latent_pair_src, latent_replica_group, latent_degree, token_group, T_local, q_lora_shard, kv_row_shard
        )
    if q_slices > 1 and not Q_BLOCK_AG_FIRST:
        q_latent_block, cos_block, sin_block = gather_block()
    if idx_front:
        if hidden_tiles is None:
            hidden_tiles = [mla_decode_hidden_k_tiles(hidden)]
        # the indexer front: its key side, in the latent collective's PE gap.
        idx_prologue = indexer_key_prologue(
            hidden_tiles[0][0],
            hidden_tiles[0][1],
            idx_wk,
            idx_k_norm_w,
            idx_k_norm_b,
            idx_weights_proj,
            idx_cos,
            idx_sin,
            idx_k_cache,
            idx_n_heads,
            idx_rope_dim,
            idx_norm_eps,
        )

        def idx_q_early(q_tiles, q_fn):
            # indexer_q_proj on the q-norm's K-tiles, each chain stored straight to the gather's source.
            nonlocal idx_q_gathered
            q_src = nl.ndarray(
                (T_local, idx_q_shard), dtype=q_tiles.dtype, buffer=_COLLECTIVE_HBM, name=unique_name("idx_q_src")
            )
            linear(q_tiles, idx_wq_b, T_local, stationary_fn=q_fn, out_hbm=q_src)
            idx_q_gathered = _all_gather_columns(q_src, key_replica_group, key_size, spec, idx_q_shard, "idx_q")


    def run_indexer_now(q_latent_normed_arg, after_q_gather, split_merge=False):
        nonlocal idx_mask_add, idx_block_rows
        # The residency mask is a per-STEP quantity (it depends only on idx_positions[0]); a caller that
        # runs several indexer layers builds it once and passes it in, a standalone layer builds its own.
        # The legacy flat cache selects the pre-block path: contiguous shard, no deal.
        kernel_assert(
            idx_shard_degree == 1 or indexer_cache_is_blocked(idx_k_cache),
            f"idx_shard_degree={idx_shard_degree} needs the block-form indexer cache; the legacy flat "
            f"[idx_keys, idx_head_dim] cache keeps the contiguous shard only",
        )
        if idx_mask_add is None:
            idx_mask_add = indexer_residency_mask(
                idx_positions,
                indexer_cache_keys(idx_k_cache),
                spec,
                key_base=0,
                shard_degree=idx_shard_degree,
                block_size=idx_block_size,
                # The SAME tensor the select folds into its labels, so one runtime scalar serves both.
                rank_offset=key_offset if idx_shard_degree > 1 else None,
                # The chunk-dealt pool: one cut per region, its owner's (a standalone layer; the GK builds it once).
                **({"region_bases": idx_pool_region_base} if idx_pool_items is not None else {}),
            )

        # Same per-STEP sharing as the residency mask: the row table depends on the block table only,
        # so a caller running several indexer layers builds it once. Both None leaves the cache
        # position-ordered, which is the pre-paging behaviour and every existing caller's path.
        if idx_block_rows is None and idx_block_table is not None:
            idx_keys = indexer_cache_keys(idx_k_cache)
            idx_block_rows = indexer_block_rows(idx_block_table, idx_keys, idx_keys // (idx_block_table.shape[1] - 1))

        return _run_indexer(
            q_latent_normed=q_latent_normed_arg,
            after_q_gather=after_q_gather,
            hidden=hidden,
            idx_wq_b=idx_wq_b,
            idx_wk=idx_wk,
            idx_k_norm_w=idx_k_norm_w,
            idx_k_norm_b=idx_k_norm_b,
            idx_weights_proj=idx_weights_proj,
            idx_cos=idx_cos,
            idx_sin=idx_sin,
            idx_k_cache=idx_k_cache,
            idx_new_slot_mapping=idx_new_slot_mapping,
            idx_mask_add=idx_mask_add,
            idx_block_rows=idx_block_rows,
            pool_replica_group=pool_replica_group,
            idx_pool_degree=idx_pool_degree,
            idx_pool_tokens=idx_pool_tokens,
            idx_pool_items=idx_pool_items,
            idx_pool_n_items=idx_pool_n_items,
            idx_pool_new_key_rows=idx_pool_new_key_rows,
            idx_pool_unrot=idx_pool_unrot,
            idx_shard_degree=idx_shard_degree,
            idx_block_size=idx_block_size,
            key_offset=key_offset,
            idx_positions=idx_positions,
            seq_replica_group=seq_replica_group,
            key_replica_group=key_replica_group,
            num_seqs_local=num_seqs_local,
            key_size=key_size,
            idx_n_heads=idx_n_heads,
            idx_rope_dim=idx_rope_dim,
            index_topk=index_topk,
            spec=spec,
            idx_norm_eps=idx_norm_eps,
            idx_q_shard=idx_q_shard,
            k_cols=k_cols,
            q_idx=idx_q_gathered,
            prologue=idx_prologue,
            split_merge=split_merge,
            own_selection=(idx_shard_degree > 1 and idx_shard_degree == cp_degree == key_size and head_shard == 1
                           and idx_block_size == kv_cache.shape[2]),
        )

    idx_finish = None

    def idx_scores_early():
        # the indexer front: scoring, local top-k and the candidate gather (AG0#4) ahead of wq_b; the merge
        # runs after mla_decode_q_kv_up, whose head gather (AG0#2) is then issued after AG0#4.
        nonlocal idx_finish
        idx_finish = run_indexer_now(None, lambda: None, split_merge=True)

    # ═══════ Up-projection, RoPE, W_UK absorb ═══════
    # Also the ONLY producer of the normalized query latent the indexer needs.
    # GLM52K_Q_BLOCK_SPLIT_EXCHANGE: the token group (the slice's cp ranks, head_shard == 1 here) splits the slice.
    q_split = Q_BLOCK_SPLIT_EXCHANGE and q_latent_block is not None and T_local % token_group == 0
    q_exchange_kw = dict(token_replica_group=token_replica_group) if q_split else {}
    q_absorbed, q_pe, kv_row_new_flat, q_latent_normed = mla_decode_q_kv_up(
        q_latent=q_latent,
        kv_row=kv_row,
        q_norm_w=q_norm_w,
        kv_norm_w=kv_norm_w,
        wq_b=wq_b,
        q_absorb_w=q_absorb_w,
        cos=cos,
        sin=sin,
        num_heads=num_heads,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        kv_lora_rank=kv_lora_rank,
        norm_eps=latent_norm_eps,
        # wq_b is COLUMN-sharded over the token group -- the same group that column-shards the two
        # down-projections, and the only one whose members hold the same tokens -- or over the block
        # group, whose members were just handed the block's tokens. The token-group shard cuts wq_b from
        # 67.1 MB to 16.8 MB per rank at CP4 for one 576 KB AllGather; the block shard divides that by
        # q_slices again (4.2 MB at 4 slices) for one 576 KB all_to_all.
        replica_group=token_replica_group if q_latent_block is None else qblock_replica_group,
        q_latent_block=q_latent_block,
        cos_block=cos_block,
        sin_block=sin_block,
        after_q_norm=idx_q_early if idx_front else None,
        before_up_proj=idx_scores_early if idx_front else None,
        # GLM52K_Q_NORM_INDEXER_ONLY: only the indexer reads the normalized own latent.
        need_q_latent_normed=run_indexer or not Q_NORM_INDEXER_ONLY,
        **(dict(q_block_split=True) if q_split else {}),
    )

    # ═══════ The indexer, on the seq x key view of the token group ═══════
    # On an ``indexer_types == "shared"`` layer the reference has no indexer at all and reuses the
    # nearest preceding "full" layer's selection verbatim (modeling_glm_moe_dsa.py: prev_topk_indices).
    # Skipping the four sub-kernels also removes four collectives from this rank's layer.
    if run_indexer:
        # The block-sharded query path's all_to_all is issued INSIDE the indexer, after its query gather:
        # collectives complete in issue order, so issued before the indexer's it would stall them for its
        # latency; issued after the query gather it overlaps the indexer's scoring.
        def issue_q_exchange():
            nonlocal q_absorbed, q_pe
            if q_latent_block is not None:
                q_absorbed, q_pe = mla_decode_q_block_exchange(q_absorbed, q_pe, qblock_replica_group, **q_exchange_kw)

        if idx_finish is not None:
            # the indexer front issued its collectives inside mla_decode_q_kv_up, so the block
            # exchange goes out ahead of the merge, which issues none at num_seqs_local == 1.
            issue_q_exchange()
            gather_indices_true, gather_indices_km = idx_finish()
        else:
            gather_indices_true, gather_indices_km = run_indexer_now(q_latent_normed, issue_q_exchange)
    else:
        gather_indices_true = prev_gather_indices
        gather_indices_km = None

    # The block-sharded query path's receive, placed AFTER the indexer on purpose: the engines run in
    # program order, so this is what hides the all_to_all's latency behind the indexer's work. (On a
    # shared layer the exchange is issued here and there is nothing to hide it behind.)
    # GLM52K_ATTN_Q_HOOK: on a shared layer the receive is handed to the attention core instead, which emits it after
    # pair 0's gathers and key transposes, so they run under the exchange's flight.
    q_hook = None
    if q_latent_block is not None:
        if not run_indexer:
            q_absorbed, q_pe = mla_decode_q_block_exchange(q_absorbed, q_pe, qblock_replica_group, **q_exchange_kw)
        q_finish_kw = dict(token_split=token_group) if q_split else {}
        if ATTN_Q_HOOK and not run_indexer:
            q_sent = (q_absorbed, q_pe)

            def q_hook():
                qa, qp = mla_decode_q_block_finish(q_sent[0], q_sent[1], **q_finish_kw)
                rm = take_rank_major_heads(qa)
                return (rm[0], rm[1], rm[2]) if rm is not None else (qa, qp, 1)
        else:
            q_absorbed, q_pe = mla_decode_q_block_finish(q_absorbed, q_pe, **q_finish_kw)


    # ═══════ MLA attention over this rank's tokens and its share of their selected keys ═══════
    # The CP position shift (``cp_pos_offset``) is folded into build_prior_index: it lands on the
    # key-major tile there as two Vector ops instead of a token-major pass plus an HBM round trip here.
    # On the fused block-sharded exit the core writes its context and lse STRAIGHT into the all_to_all
    # payload (head-major cells, see _out_proj_block_fused_combine), so the 512 KB context never takes
    # the detour through its own HBM tensor and back.
    oproj_tp = num_heads // nh_out
    # Block-sharded exit: the lse rides the context all_to_all and the receiver combines
    # (_out_proj_block_fused_combine), one collective and a sender-side pass fewer than gathering the lse.
    fused_combine = oproj_tp != cp_degree
    # GLM52K_CP_COMBINE_TWO_STAGE: the payload in cp_degree token pieces, combined on the CP group's chip first.
    two_stage = (
        fused_combine
        and CP_COMBINE_TWO_STAGE
        and T_local % cp_degree == 0
        and num_heads * (T_local // cp_degree) <= P_MAX
        and (oproj_tp // cp_degree) * T_local <= P_MAX
    )
    payload = None
    if fused_combine:
        payload = nl.ndarray(
            (num_heads * T_local, _a2a_cell(latent_dim)),
            dtype=x_shard.dtype,
            buffer=_COLLECTIVE_HBM,
            name=unique_name("ctx_lse_cells"),
        )
    # The head gather left the queries rank-major in its own destination.
    heads_blocked = 1
    rank_major = take_rank_major_heads(q_absorbed) if q_hook is None else None
    if rank_major is not None:
        q_absorbed, q_pe, heads_blocked = rank_major
    ctx_local, lse_local, kv_cache = mla_decode_attention(
        q_absorbed=q_absorbed if q_hook is None else None,
        q_pe=q_pe if q_hook is None else None,
        kv_cache=kv_cache,
        kv_row_new=kv_row_new_flat.reshape((num_seqs_local, spec, qk_rope_head_dim + kv_lora_rank)),
        new_slot_mapping=kv_new_slot_mapping,
        gather_indices=gather_indices_true.reshape((num_seqs_local, spec, index_topk)),
        block_table=block_table,
        positions=positions,
        softmax_scale=softmax_scale,
        kv_scale=kv_scale,
        spec=spec,
        kv_lora_rank=kv_lora_rank,
        cp_degree=cp_degree,
        cp_rank=0,
        # A SHARED layer may be handed the index tables its group's full layer already resolved (see
        # :func:`build_shared_prior_index`); they are a function of the selection, the block table and
        # the CP shift only, none of which change within the group.
        prior_snake_rows=prior_snake_rows,
        prior_key_pos=prior_key_pos,
        prior_natural_rows=prior_natural_rows,
        pos_offset=cp_pos_offset,
        positions_km=gather_indices_km,
        ctx_lse_out=payload,
        prior_tables_out=prior_tables_out if run_indexer else None,
        heads_blocked=heads_blocked,
        **(dict(ctx_lse_token_split=cp_degree) if two_stage else {}),
        **(
            dict(q_hook=q_hook, q_shape=(P_MAX, kv_lora_rank // P_MAX, T_local, num_heads // head_shard))
            if q_hook
            else {}
        ),
    )

    # ═══════ Every CP rank's log-sum-exp, so the combine can weight this rank's share ═══════
    # With the fused block-sharded exit the lse travels inside the context all_to_all instead and the
    # receiver forms the weights (see _out_proj_block_fused_combine); nothing to gather here.
    scaled_ctx = None
    if not fused_combine:
        lse_all = nl.ndarray(
            (cp_degree * T_local, num_heads), dtype=nl.float32, buffer=_COLLECTIVE_HBM, name=unique_name("lse_all")
        )
        _pin_all(lse_local, lse_all)  # while pinning is on; see utils/cc_pin.py
        ncc.all_gather(srcs=[lse_local], dsts=[lse_all], replica_group=cp_replica_group, collective_dim=0)
        scaled_ctx = _cp_softmax_combine(ctx_local, lse_local, lse_all, T_local, num_heads, latent_dim, cp_degree)

    # The exit depends on the heads out_absorb_w carries. o_proj's output is a partial sum over heads either way
    # (mla_decode_out_proj asserts that), and the head count is read off the operand:
    # * nh_out = Nh / cp_degree, heads over the CP group: the CP ReduceScatter scatters heads while it sums the key
    #   partials, o_proj runs on T_local tokens, and a second CP ReduceScatter sums the head partials into T_shard rows.
    # * nh_out = Nh / oproj_tp, a multiple of cp_degree: heads over the block group (_out_proj_block_sharded).
    attn_out = output
    if attn_out is None:
        attn_out = nl.ndarray(
            (T_shard, H), dtype=x_shard.dtype, buffer=nl.shared_hbm, name=unique_name("attn_layer_out")
        )
    kernel_assert(
        (
            tuple(attn_out.shape) == (T_shard, H)
            or (attn_out.buffer == nl.sbuf and _numel(attn_out.shape) == T_shard * H)
        )
        and attn_out.dtype == x_shard.dtype,
        f"{_ERROR_PREFIX} output must hold [{T_shard}, {H}] {x_shard.dtype}, got {attn_out.shape} {attn_out.dtype}",
    )
    if oproj_tp != cp_degree:
        kernel_assert(
            block_replica_group is not None,
            f"{_ERROR_PREFIX} out_absorb_w carries {nh_out} of {num_heads} heads, an o_proj sharded "
            f"{oproj_tp} ways: that needs the block_replica_group ({oproj_tp} consecutive ranks)",
        )
        slices = oproj_tp // cp_degree
        kernel_assert(
            oproj_tp % cp_degree == 0 and token_shard % slices == 0 and head_shard == 1 and slices * T_local <= P_MAX,
            f"{_ERROR_PREFIX} a block-sharded o_proj ({oproj_tp} ways) needs oproj_tp to be a multiple of "
            f"cp_degree={cp_degree}, its {slices} token slices to divide token_shard={token_shard}, head_shard "
            f"== 1 (got {head_shard}) and the block's {slices} x T_local={T_local} tokens to fit one partition "
            f"tile",
        )
        if two_stage:
            _out_proj_block_two_stage_combine(
                attn_out,
                payload,
                out_absorb_w,
                o_proj_w,
                block_replica_group,
                cp_replica_group,
                T_local,
                T_shard,
                oproj_tp,
                cp_degree,
                nh_out,
                latent_dim,
                H,
            )
        elif fused_combine:
            _out_proj_block_fused_combine(
                attn_out,
                payload,
                out_absorb_w,
                o_proj_w,
                block_replica_group,
                T_local,
                T_shard,
                oproj_tp,
                cp_degree,
                nh_out,
                latent_dim,
                H,
            )
        else:
            _out_proj_block_sharded(
                attn_out,
                scaled_ctx,
                out_absorb_w,
                o_proj_w,
                block_replica_group,
                T_local,
                T_shard,
                oproj_tp,
                cp_degree,
                nh_out,
                latent_dim,
                H,
            )
    else:
        # ═══════ The CP sum, which is also the re-shard onto the OUTPUT head slice ═══════
        # This collective scatters HEADS, not tokens, so that o_proj runs on
        # 1/cp_degree of the heads for ALL T_local tokens rather than every head for T_local/cp_degree
        # of them -- the same product of work, but with o_proj_w sharded cp_degree ways instead of
        # replicated, and with the matmul's stationary free dim cp_degree times wider.
        #
        # The sum it performs is unchanged: it still reduces the CP ranks' key-partials. Only which
        # slice of the result each rank keeps has moved from the token axis to the head axis.
        ctx_heads = nl.ndarray(
            (nh_out, T_local * latent_dim),
            dtype=scaled_ctx.dtype,
            buffer=_COLLECTIVE_HBM,
            name=unique_name("ctx_head_shard"),
        )
        _pin_all(scaled_ctx, ctx_heads)  # while pinning is on; see utils/cc_pin.py
        ncc.reduce_scatter(
            srcs=[scaled_ctx], dsts=[ctx_heads], op=nl.add, replica_group=cp_replica_group, collective_dim=0
        )

        # Back to the token-major [T, Nh_out, C] that mla_decode_out_proj indexes. Two plain DMAs through
        # SBUF -- the head count here is at most 16 and the runs are C wide, so this is block movement
        # rather than a transpose.
        ctx_sb = nl.ndarray((P_MAX, nh_out, latent_dim), dtype=scaled_ctx.dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=ctx_sb[0:T_local, 0:nh_out, 0:latent_dim],
            src=ctx_heads.ap(
                pattern=[[latent_dim, T_local], [T_local * latent_dim, nh_out], [1, latent_dim]], offset=0
            ),
        )
        ctx_tok_major = nl.ndarray(
            (T_local, nh_out, latent_dim),
            dtype=scaled_ctx.dtype,
            buffer=_COLLECTIVE_HBM,
            name=unique_name("ctx_token_major"),
        )
        nisa.dma_copy(
            dst=ctx_tok_major[0:T_local, 0:nh_out, 0:latent_dim], src=ctx_sb[0:T_local, 0:nh_out, 0:latent_dim]
        )

        # ═══════ W_UV absorb + o_proj, then the CP ReduceScatter that completes the head sum ═══════
        attn_partial = mla_decode_out_proj(ctx=ctx_tok_major, out_absorb_w=out_absorb_w, o_proj_w=o_proj_w)
        tokens_cp = T_local // cp_degree
        attn_cp = cc_dst((tokens_cp, H), attn_partial.dtype, _COLLECTIVE_HBM, "attn_cp_summed", reduce_scatter=True)
        _pin_all(attn_partial, attn_cp)  # while pinning is on; see utils/cc_pin.py
        ncc.reduce_scatter(
            srcs=[attn_partial], dsts=[attn_cp], op=nl.add, replica_group=cp_replica_group, collective_dim=0
        )

        # At head_shard == 1 the CP group covered every head, so attn_cp is COMPLETE and tokens_cp is
        # already T_shard. Otherwise a second reduction over the head group finishes the sum and
        # scatters tokens head_shard ways more, which composes onto global slot T_shard*i exactly as
        # before -- the token scatter moved from one collective to another, and the digit order is
        # untouched.
        if head_shard == 1:
            _deliver_attn_out(attn_out, attn_cp)
        else:
            attn_shard = nl.ndarray(
                (T_shard, H),
                dtype=attn_partial.dtype,
                buffer=_COLLECTIVE_HBM,
                name=unique_name("attn_shard"),
            )
            _pin_all(attn_cp, attn_shard)  # while pinning is on; see utils/cc_pin.py
            ncc.reduce_scatter(
                srcs=[attn_cp], dsts=[attn_shard], op=nl.add, replica_group=head_replica_group, collective_dim=0
            )
            _deliver_attn_out(attn_out, attn_shard)

    return _layer_outputs(
        attn_out,
        run_indexer,
        prev_gather_indices,
        gather_indices_true,
        kv_cache,
        idx_k_cache,
        num_seqs_local,
        spec,
        index_topk,
        T_local,
    )


def _layer_outputs(
    attn_out: nl.NkiTensor,
    run_indexer: bool,
    prev_gather_indices: nl.NkiTensor,
    gather_indices_true: nl.NkiTensor,
    kv_cache: nl.NkiTensor,
    idx_k_cache: nl.NkiTensor,
    num_seqs_local: int,
    spec: int,
    index_topk: int,
    T_local: int,
):
    """The layer kernel's return tuple."""
    # A "shared" layer produced no selection of its own, so it hands the one it was given straight
    # back -- which is exactly how the reference threads ``topk_indices`` through consecutive shared
    # layers (the identical tensor object propagates). No copy, and no indexer key cache to return.
    if not run_indexer:
        return attn_out, prev_gather_indices, kv_cache, None

    gather_indices_out = nl.ndarray(
        (num_seqs_local, spec, index_topk), dtype=nl.int32, buffer=scratch_hbm(), name=unique_name("gather_indices_out")
    )
    nisa.dma_copy(dst=gather_indices_out.reshape((T_local, index_topk)), src=gather_indices_true)

    return attn_out, gather_indices_out, kv_cache, idx_k_cache


def _out_proj_block_sharded(
    attn_out: nl.NkiTensor,
    scaled_ctx: nl.NkiTensor,
    out_absorb_w: nl.NkiTensor,
    o_proj_w: nl.NkiTensor,
    block_replica_group: ReplicaGroup,
    T_local: int,
    T_shard: int,
    oproj_tp: int,
    cp_degree: int,
    nh_out: int,
    latent_dim: int,
    H: int,
) -> None:
    """The output projection sharded ``oproj_tp`` ways over heads across the block group: the CP group and its
    ``oproj_tp / cp_degree`` neighbouring token slices, ``oproj_tp`` consecutive ranks. Rank ``(kk, cp)`` sits at
    position ``d = cp_degree * (kk mod slices) + cp`` and projects heads ``[d * nh_out, (d + 1) * nh_out)`` for all the
    block's ``slices * T_local`` tokens, which divides each rank's o_proj rows by ``slices`` for the same number of
    collectives.

    1. An all-to-all over the block group on the head axis of the key-partial context: position ``d`` receives its
       heads from every slice's CP partials.
    2. The CP combine on chip: each slice's ``cp_degree`` partials summed in fp32 (the weights sum to 1), laid out
       token-major ``[slices * T_local, nh_out, C]``.
    3. ``mla_decode_out_proj`` to ``[slices * T_local, H]``, partial over the head groups.
    4. ReduceScatter(add) over the block group on dim 0: position ``p`` receives exactly the MoE rows it owns, so the
       destination is the group position and the kernel stays rank-invariant.

    Args:
        attn_out: [T_shard, H] kernel output, written here.
        scaled_ctx: [Nh, T_local * C] this rank's weighted key-partial context, head-major (``_cp_softmax_combine``).
        out_absorb_w: [C, nh_out, v_head] latent-major W_UV for this rank's heads.
        o_proj_w: [nh_out * v_head, H] this rank's head rows.
        block_replica_group: the ``oproj_tp`` consecutive ranks, in position order.
        T_local, T_shard, oproj_tp, cp_degree, nh_out, latent_dim, H: extents.
    """

    slices = oproj_tp // cp_degree
    T_all = slices * T_local
    C = latent_dim
    TC = T_local * C
    num_heads = oproj_tp * nh_out
    dtype = scaled_ctx.dtype
    rows = nh_out * T_local  # (head, token) pairs of this rank's heads, on partitions below

    # 1. The all_to_all. Dim 0 of [Nh, T_local * C] splits into oproj_tp blocks of nh_out heads: block d
    #    to position d, block s from source s. HBM operands, private_hbm like every collective here.
    ctx_a2a = cc_dst((num_heads, TC), dtype, _COLLECTIVE_HBM, "ctx_a2a")
    _pin_all(scaled_ctx, ctx_a2a)  # while pinning is on; see utils/cc_pin.py
    ncc.all_to_all(srcs=[scaled_ctx], dsts=[ctx_a2a], replica_group=block_replica_group, collective_dim=0)

    # 2. The CP sum. Received row (s, h) = (slice, cp, h) holds tokens x C. Loaded with (h, t) on
    #    partitions -- partition h * T_local + t reads at offset (h * T_local + t) * C, one stride -- and
    #    (slice, cp, C) on the free axis, the cp partials of a slice are free-axis neighbours: three
    #    fp32 adds over [rows, slices, C], one cast back. Summing in fp32 and rounding once is what the
    #    ReduceScatter it replaces does for the CP-sharded exit.
    recv_sb = nl.ndarray((P_MAX, slices, cp_degree, C), dtype=dtype, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=recv_sb[0:rows, 0:slices, 0:cp_degree, 0:C],
        src=ctx_a2a.ap(
            pattern=[[C, rows], [cp_degree * nh_out * TC, slices], [nh_out * TC, cp_degree], [1, C]], offset=0
        ),
    )
    acc_f32 = nl.ndarray((P_MAX, slices, C), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=acc_f32[0:rows, 0:slices, 0:C],
        data1=recv_sb[0:rows, 0:slices, 0, 0:C],
        data2=recv_sb[0:rows, 0:slices, 1, 0:C],
        op=nl.add,
    )
    for cp in range(2, cp_degree):
        nisa.tensor_tensor(
            dst=acc_f32[0:rows, 0:slices, 0:C],
            data1=acc_f32[0:rows, 0:slices, 0:C],
            data2=recv_sb[0:rows, 0:slices, cp, 0:C],
            op=nl.add,
        )
    ctx_sum_sb = nl.ndarray((P_MAX, slices, C), dtype=dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=ctx_sum_sb[0:rows, 0:slices, 0:C], src=acc_f32[0:rows, 0:slices, 0:C])

    #    Token-major [T_all, nh_out, C] for mla_decode_out_proj, token (slice, t) at row slice * T_local
    #    + t. Partition h * T_local + t writes at ((slice * T_local + t) * nh_out + h) * C: two partition
    #    strides, so one DMA per head, each a plain [T_local, slices, C] block copy of C-wide runs.
    ctx_tok_major = nl.ndarray(
        (T_all, nh_out, C), dtype=dtype, buffer=_COLLECTIVE_HBM, name=unique_name("ctx_block_token_major")
    )
    for h in range(nh_out):
        nisa.dma_copy(
            dst=ctx_tok_major.ap(pattern=[[nh_out * C, T_local], [T_local * nh_out * C, slices], [1, C]], offset=h * C),
            src=ctx_sum_sb[h * T_local : (h + 1) * T_local, 0:slices, 0:C],
        )

    # 3. + 4. The projection over the whole block, then the head sum as a ReduceScatter over the block
    #    group; T_all / oproj_tp = T_shard rows land on each rank, already in MoE order.
    _out_proj_block_tail(attn_out, ctx_tok_major, out_absorb_w, o_proj_w, block_replica_group, T_shard, H)


def _out_proj_block_tail(
    attn_out: nl.NkiTensor,
    ctx_tok_major: nl.NkiTensor,
    out_absorb_w: nl.NkiTensor,
    o_proj_w: nl.NkiTensor,
    block_replica_group: ReplicaGroup,
    T_shard: int,
    H: int,
    attn_partial: nl.NkiTensor = None,
    weight_gate=None,
) -> None:
    """``o_proj`` over the block's tokens, the head-group sum as a ReduceScatter, the output copy.

    Shared by :func:`_out_proj_block_sharded` and :func:`_out_proj_block_fused_combine`, which differ
    only in how they produce the combined token-major context.

    Args:
        attn_out (nl.NkiTensor): [T_shard, H] kernel output, written here.
        ctx_tok_major (nl.NkiTensor): [slices * T_local, nh_out, C] the CP-combined context of the
            block's tokens for this rank's heads, in HBM.
        out_absorb_w (nl.NkiTensor): [C, nh_out, v_head] latent-major W_UV for this rank's heads.
        o_proj_w (nl.NkiTensor): [nh_out * v_head, H] this rank's head rows, full hidden.
        block_replica_group (ReplicaGroup): The ``oproj_tp`` consecutive ranks, in position order.
        T_shard, H (int): Extents.
        attn_partial (nl.NkiTensor): Optional [slices * T_local, H] o_proj output already computed by the caller
            (the on-chip context path); ``ctx_tok_major`` is then unused. (default: None)
        weight_gate: An instruction the o_proj weight loads wait on, or None (see ``mla_decode_out_proj``).
    """
    if attn_partial is None:
        attn_partial = mla_decode_out_proj(
            ctx=ctx_tok_major, out_absorb_w=out_absorb_w, o_proj_w=o_proj_w, weight_gate=weight_gate
        )
    attn_rs = cc_dst((T_shard, H), attn_partial.dtype, _COLLECTIVE_HBM, "attn_block_summed", reduce_scatter=True)
    _pin_all(attn_partial, attn_rs)  # while pinning is on; see utils/cc_pin.py
    ncc.reduce_scatter(
            srcs=[attn_partial], dsts=[attn_rs], op=nl.add, replica_group=block_replica_group, collective_dim=0
        )
    _ATTN_RS.clear()
    _ATTN_RS.append(attn_rs)
    _deliver_attn_out(attn_out, attn_rs)


def _a2a_cell(latent_dim: int) -> int:
    """Width of one head's cell in the fused-combine payload: C context values, 2 bf16 slots of fp32 lse, padding to 16 B."""
    return -(-(latent_dim + 2) // _A2A_CELL_ALIGN) * _A2A_CELL_ALIGN


def _out_proj_block_fused_combine(
    attn_out: nl.NkiTensor,
    payload: nl.NkiTensor,
    out_absorb_w: nl.NkiTensor,
    o_proj_w: nl.NkiTensor,
    block_replica_group: ReplicaGroup,
    T_local: int,
    T_shard: int,
    oproj_tp: int,
    cp_degree: int,
    nh_out: int,
    latent_dim: int,
    H: int,
) -> None:
    """:func:`_out_proj_block_sharded` with the CP softmax combine done by the receiver of the ``all_to_all``, which drops
    the ``lse`` AllGather and the sender-side scaling. Each partial's ``lse`` travels next to its unscaled context, so the
    receiver forms the CP weights itself with the same arithmetic as :func:`_cp_softmax_combine`, and no per-rank index
    is needed.

    The payload is head-major ``[Nh * T_local, chunk]`` bf16: a cell's ``C`` context values, its fp32 ``lse`` as two raw
    bf16 slots, then padding. The attention core writes it directly from its context tile. The receiver stages the
    partials of each (token, head) on one partition, combines them, and writes the context token-major for
    ``mla_decode_out_proj``.
    """
    slices = oproj_tp // cp_degree
    T = T_local
    T_all = slices * T
    C = latent_dim
    dtype = payload.dtype
    R = nh_out * T  # rows per all_to_all block = (head, token) pairs of one source
    P_recv = slices * R  # (slice, head, token) triples this rank projects
    kernel_assert(
        P_recv <= P_MAX and C % 2 == 0,
        f"{_ERROR_PREFIX} fused CP combine: slices * nh_out * T_local = {P_recv} must fit {P_MAX} partitions and "
        f"C = {C} must be even",
    )
    kernel_assert(
        dtype == nl.bfloat16 and tuple(payload.shape) == (oproj_tp * R, _a2a_cell(C)),
        f"{_ERROR_PREFIX} fused CP combine payload must be [{oproj_tp * R}, {_a2a_cell(C)}] bfloat16, got "
        f"{payload.shape} {dtype}",
    )
    chunk = _a2a_cell(C)

    # 1. The all_to_all: block d (rows [d * R, (d + 1) * R)) to position d; block s arrives from source
    #    s = cp_degree * slice + cp with token slice ``slice``'s partial ``cp`` of THIS rank's heads.
    recv_hbm = cc_dst((oproj_tp * R, chunk), dtype, _COLLECTIVE_HBM, "ctx_lse_a2a")
    _pin_all(payload, recv_hbm)  # while pinning is on; see utils/cc_pin.py
    ncc.all_to_all(srcs=[payload], dsts=[recv_hbm], replica_group=block_replica_group, collective_dim=0)

    # 2. Receiver: (slice, h, t) on partition slice * R + h * T + t, the cp partials on the free axis.
    #    Within a slice the partition stride is one row, the cp stride one block: one DMA per slice.
    recv_sb = nl.ndarray((P_MAX, cp_degree, chunk), dtype=dtype, buffer=nl.sbuf)
    recv_kw = _combine_recv_queue()
    for sl in range(slices):
        nisa.dma_copy(
            dst=recv_sb[sl * R : (sl + 1) * R, 0:cp_degree, 0:chunk],
            src=recv_hbm.ap(
                pattern=[[chunk, R], [R * chunk, cp_degree], [1, chunk]], offset=sl * cp_degree * R * chunk
            ),
            **recv_kw,
        )
    ctx_sum_sb = _combine_cp_cells(recv_sb, P_recv, cp_degree, C, dtype)
    if C % P_MAX == 0:  # the combined context goes to the o_proj latent-major on chip, not through HBM
        ctx_t_sb = _ctx_latent_major_on_chip(ctx_sum_sb, slices, nh_out, T, C)
        attn_partial = mla_decode_out_proj_sbuf(
            ctx_t_sb, T_all, out_absorb_w, o_proj_w
        )
        _out_proj_block_tail(
            attn_out, None, out_absorb_w, o_proj_w, block_replica_group, T_shard, H, attn_partial=attn_partial
        )
        return
    # Token-major for mla_decode_out_proj: partition slice * R + h * T + t is row (slice * T + t) of head h,
    # so each (slice, head) run of T partitions is one DMA to T rows nh_out * C apart.
    ctx_tok_major = nl.ndarray(
        (T_all, nh_out, C), dtype=dtype, buffer=_COLLECTIVE_HBM, name=unique_name("ctx_block_token_major")
    )
    for sl in range(slices):
        for h in range(nh_out):
            p0 = sl * R + h * T
            nisa.dma_copy(
                dst=ctx_tok_major[sl * T : (sl + 1) * T, h, 0:C],
                src=ctx_sum_sb[p0 : p0 + T, 0:C],
            )

    _out_proj_block_tail(attn_out, ctx_tok_major, out_absorb_w, o_proj_w, block_replica_group, T_shard, H)


def _combine_cp_cells(recv_sb: nl.NkiTensor, P_recv: int, cp_degree: int, C: int, dtype) -> nl.NkiTensor:
    """The fused combine's receiver arithmetic on staged cells: ``[P_recv, cp_degree, chunk]`` -> ``[P_recv, C]``.

    Each partition is one ``(token, head)``, its ``cp_degree`` key partials on the free axis in CP order, each cell
    ``C`` context values and the fp32 lse in bf16 slots ``[C, C + 2)``. The weights are :func:`_cp_softmax_combine`'s
    (``m = max_r lse_r``, ``exp(lse_r - m)``, the sum, its reciprocal, the fp32 weight), then the bf16
    ``ctx_r * w_r``, the fp32 CP sum in CP order and one cast. Every op is per partition, so the result of a cell
    does not depend on which partition, or which rank, combines it.

    Returns:
        nl.NkiTensor: [P_MAX, C] SBUF of ``dtype``, the combined context in the first ``P_recv`` partitions.
    """
    # The CP softmax weights, exactly as _cp_softmax_combine forms them. Every op reads the lse bits in
    # place, through an fp32 view of the received cells: [P_recv, cp_degree], cp on the free axis.
    lse_v = recv_sb[0:P_recv, 0:cp_degree, C : C + 2].view(nl.float32)[0:P_recv, 0:cp_degree, 0]
    max_sb = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(max_sb[0:P_recv, 0:1], op=nl.max, data=lse_v, axis=1)
    shifted_sb = nl.ndarray((P_MAX, cp_degree), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=shifted_sb[0:P_recv, 0:cp_degree],
        data1=lse_v,
        data2=max_sb[0:P_recv, 0:1].broadcast(1, cp_degree),
        op=nl.subtract,
    )
    exp_sb = nl.ndarray((P_MAX, cp_degree), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(exp_sb[0:P_recv, 0:cp_degree], op=nl.exp, data=shifted_sb[0:P_recv, 0:cp_degree])
    sum_sb = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(sum_sb[0:P_recv, 0:1], op=nl.add, data=exp_sb[0:P_recv, 0:cp_degree], axis=1)
    recip_sb = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(recip_sb[0:P_recv, 0:1], op=nl.reciprocal, data=sum_sb[0:P_recv, 0:1])
    weight_sb = nl.ndarray((P_MAX, cp_degree, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=weight_sb[0:P_recv, 0:cp_degree, 0],
        data1=exp_sb[0:P_recv, 0:cp_degree],
        data2=recip_sb[0:P_recv, 0:1].broadcast(1, cp_degree),
        op=nl.multiply,
    )
    # ctx_r * w_r in bf16 (the sender's scaling, moved here), then the fp32 CP sum and one cast.
    scaled_sb = nl.ndarray((P_MAX, cp_degree, C), dtype=dtype, buffer=nl.sbuf)
    if LAT_COMBINE_TS and cp_degree >= 2:
        for cp in range(cp_degree):
            nisa.tensor_scalar(
                dst=scaled_sb[0:P_recv, cp, 0:C],
                data=recv_sb[0:P_recv, cp, 0:C],
                op0=nl.multiply,
                operand0=weight_sb[0:P_recv, cp, 0:1],
            )
        ctx_sum_sb = nl.ndarray((P_MAX, C), dtype=dtype, buffer=nl.sbuf)
        if cp_degree == 2:
            nisa.tensor_tensor(
                dst=ctx_sum_sb[0:P_recv, 0:C],
                data1=scaled_sb[0:P_recv, 0, 0:C],
                data2=scaled_sb[0:P_recv, 1, 0:C],
                op=nl.add,
            )
            return ctx_sum_sb
        acc_f32 = nl.ndarray((P_MAX, C), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=acc_f32[0:P_recv, 0:C], data1=scaled_sb[0:P_recv, 0, 0:C], data2=scaled_sb[0:P_recv, 1, 0:C], op=nl.add
        )
        for cp in range(2, cp_degree):
            last = cp == cp_degree - 1
            nisa.tensor_tensor(
                dst=ctx_sum_sb[0:P_recv, 0:C] if last else acc_f32[0:P_recv, 0:C],
                data1=acc_f32[0:P_recv, 0:C],
                data2=scaled_sb[0:P_recv, cp, 0:C],
                op=nl.add,
            )
        return ctx_sum_sb
    nisa.tensor_tensor(
        dst=scaled_sb[0:P_recv, 0:cp_degree, 0:C],
        data1=recv_sb[0:P_recv, 0:cp_degree, 0:C],
        data2=weight_sb[0:P_recv, 0:cp_degree, 0:1].broadcast(2, C),
        op=nl.multiply,
    )
    acc_f32 = nl.ndarray((P_MAX, C), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=acc_f32[0:P_recv, 0:C],
        data1=scaled_sb[0:P_recv, 0, 0:C],
        data2=scaled_sb[0:P_recv, 1, 0:C],
        op=nl.add,
    )
    for cp in range(2, cp_degree):
        nisa.tensor_tensor(
            dst=acc_f32[0:P_recv, 0:C],
            data1=acc_f32[0:P_recv, 0:C],
            data2=scaled_sb[0:P_recv, cp, 0:C],
            op=nl.add,
        )
    ctx_sum_sb = nl.ndarray((P_MAX, C), dtype=dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=ctx_sum_sb[0:P_recv, 0:C], src=acc_f32[0:P_recv, 0:C])
    return ctx_sum_sb


def _out_proj_block_two_stage_combine(
    attn_out: nl.NkiTensor,
    payload: nl.NkiTensor,
    out_absorb_w: nl.NkiTensor,
    o_proj_w: nl.NkiTensor,
    block_replica_group: ReplicaGroup,
    cp_replica_group: ReplicaGroup,
    T_local: int,
    T_shard: int,
    oproj_tp: int,
    cp_degree: int,
    nh_out: int,
    latent_dim: int,
    H: int,
) -> None:
    """:func:`_out_proj_block_fused_combine` with the CP combine done inside the CP group, on one chip
    (``GLM52K_CP_COMBINE_TWO_STAGE``), so only combined context crosses chips: an ``all_to_all`` of token pieces over
    the CP group, the combine (:func:`_combine_cp_cells`, the same per-cell arithmetic), then an ``all_to_all`` of the
    combined bf16 context over the block group, which lands already token-major for the o-projection
    (:func:`_out_proj_block_tail`). It moves ``1 / cp_degree`` of the fused form's cross-chip cells.
    """
    slices = oproj_tp // cp_degree
    T = T_local
    T_all = slices * T
    C = latent_dim
    dtype = payload.dtype
    num_heads = oproj_tp * nh_out
    Tq = T // cp_degree
    P1 = num_heads * Tq  # (head, token) cells of one token piece: the combine's partitions
    chunk = _a2a_cell(C)
    kernel_assert(
        T % cp_degree == 0 and P1 <= P_MAX and T_all <= P_MAX and C % 2 == 0,
        f"{_ERROR_PREFIX} two-stage CP combine: T_local={T_local} must split into {cp_degree} pieces whose "
        f"{num_heads} x {Tq} cells fit {P_MAX} partitions, and C = {C} must be even",
    )
    kernel_assert(
        dtype == nl.bfloat16 and tuple(payload.shape) == (cp_degree * P1, chunk),
        f"{_ERROR_PREFIX} two-stage CP combine payload must be [{cp_degree * P1}, {chunk}] bfloat16, got "
        f"{payload.shape} {dtype}",
    )

    # 1. Over the CP group: piece g to CP position g; block r of the result is partial r of this rank's piece.
    recv_hbm = cc_dst((cp_degree * P1, chunk), dtype, _COLLECTIVE_HBM, "ctx_lse_cp_a2a")
    _pin_all(payload, recv_hbm)  # while pinning is on; see utils/cc_pin.py
    ncc.all_to_all(srcs=[payload], dsts=[recv_hbm], replica_group=cp_replica_group, collective_dim=0)

    # 2. Cell (n, j) on partition n * Tq + j, partial r at free index r: one DMA.
    recv_sb = nl.ndarray((P_MAX, cp_degree, chunk), dtype=dtype, buffer=nl.sbuf)

    if LAT_COMBINE_LSE_FIRST and P1 % 2 == 0:
        # The lse slots first (ACT queue), then the context halves on Sync and ACT.
        nisa.dma_copy(
            dst=recv_sb[0:P1, 0:cp_degree, C : C + 2],
            src=recv_hbm.ap(pattern=[[chunk, P1], [P1 * chunk, cp_degree], [1, 2]], offset=C),
            dge_mode=nisa.dge_mode.hwdge,
            engine=nisa.engine.scalar,
        )
        half = P1 // 2
        for i, eng in enumerate((nisa.engine.sync, nisa.engine.scalar)):
            nisa.dma_copy(
                dst=recv_sb[i * half : (i + 1) * half, 0:cp_degree, 0:C],
                src=recv_hbm.ap(pattern=[[chunk, half], [P1 * chunk, cp_degree], [1, C]], offset=i * half * chunk),
                **_combine_recv_queue(eng),
            )
    elif LAT_COMBINE_TS and P1 % 2 == 0:
        # Two halves of the cells over the Sync and ACT queues.
        half = P1 // 2
        for i, eng in enumerate((nisa.engine.sync, nisa.engine.scalar)):
            nisa.dma_copy(
                dst=recv_sb[i * half : (i + 1) * half, 0:cp_degree, 0:chunk],
                src=recv_hbm.ap(pattern=[[chunk, half], [P1 * chunk, cp_degree], [1, chunk]], offset=i * half * chunk),
                **_combine_recv_queue(eng),
            )
    else:
        nisa.dma_copy(
            dst=recv_sb[0:P1, 0:cp_degree, 0:chunk],
            src=recv_hbm.ap(pattern=[[chunk, P1], [P1 * chunk, cp_degree], [1, chunk]], offset=0),
            **_combine_recv_queue(),
        )
    ctx_sum_sb = _combine_cp_cells(recv_sb, P1, cp_degree, C, dtype)

    # 3. Block d = heads [d * nh_out, (d + 1) * nh_out) of this piece, token-major [Tq, nh_out, C]. Partition
    #    (d * nh_out + h) * Tq + j goes to row (d * Tq + j) * nh_out + h: at one head, row for row.
    rows2 = oproj_tp * Tq * nh_out
    send_hbm = nl.ndarray((rows2, C), dtype=dtype, buffer=_COLLECTIVE_HBM, name=unique_name("ctx_comb_src"))
    if nh_out == 1:
        nisa.dma_copy(dst=send_hbm[0:rows2, 0:C], src=ctx_sum_sb[0:P1, 0:C])
    else:
        for d in range(oproj_tp):
            for h in range(nh_out):
                p0 = (d * nh_out + h) * Tq
                nisa.dma_copy(
                    dst=send_hbm.ap(pattern=[[nh_out * C, Tq], [1, C]], offset=(d * Tq * nh_out + h) * C),
                    src=ctx_sum_sb[p0 : p0 + Tq, 0:C],
                )
    ctx_tok_major = cc_dst((rows2, C), dtype, _COLLECTIVE_HBM, "ctx_comb_a2a")
    _pin_all(send_hbm, ctx_tok_major)  # while pinning is on; see utils/cc_pin.py
    ncc.all_to_all(srcs=[send_hbm], dsts=[ctx_tok_major], replica_group=block_replica_group, collective_dim=0)

    # 4. Row s * Tq + j is block token s * Tq + j: token-major [T_all, nh_out, C].
    if C % P_MAX == 0 and nh_out == 1:
        # One head: partition = block token = (slice, head 0, t), the layout _ctx_latent_major_on_chip takes.
        ctx_sb = nl.ndarray((P_MAX, C), dtype=dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=ctx_sb[0:T_all, 0:C], src=ctx_tok_major[0:T_all, 0:C])
        ctx_t_sb = _ctx_latent_major_on_chip(ctx_sb, slices, nh_out, T, C)
        attn_partial = mla_decode_out_proj_sbuf(
            ctx_t_sb, T_all, out_absorb_w, o_proj_w
        )
        _out_proj_block_tail(
            attn_out, None, out_absorb_w, o_proj_w, block_replica_group, T_shard, H, attn_partial=attn_partial
        )
        return
    _out_proj_block_tail(
        attn_out, ctx_tok_major.reshape((T_all, nh_out, C)), out_absorb_w, o_proj_w, block_replica_group, T_shard, H
    )


def _ctx_latent_major_on_chip(ctx_sum_sb: nl.NkiTensor, slices: int, nh_out: int, T: int, C: int) -> nl.NkiTensor:
    """The fused combine's context, ``(slice, head, token)`` on partitions, -> ``mla_decode_out_proj_sbuf``'s layout.

    One PE transpose per 128-latent chunk (bf16 in and out, pass-through) puts the latent on partitions with the
    triples on the free axis; one Vector copy per chunk permutes them to head-major over the block's tokens,
    column ``(chunk * nh_out + h) * slices * T + slice * T + t``.

    Args:
        ctx_sum_sb (nl.NkiTensor): [slices * nh_out * T, C] bf16 SBUF, partition ``(slice * nh_out + h) * T + t``.
        slices, nh_out, T, C (int): Extents; C a multiple of 128.

    Returns:
        nl.NkiTensor: [P_MAX, C / 128 * nh_out * slices * T] bf16 SBUF.
    """
    R = nh_out * T
    P_recv = slices * R
    T_all = slices * T
    nC = C // P_MAX
    width = nh_out * T_all
    ctx_t_sb = nl.ndarray((P_MAX, nC * width), dtype=ctx_sum_sb.dtype, buffer=nl.sbuf)
    for ci in range(nC):
        ct_psum = nl.ndarray((P_MAX, P_recv), dtype=ctx_sum_sb.dtype, buffer=nl.psum)
        nisa.nc_transpose(ct_psum[0:P_MAX, 0:P_recv], ctx_sum_sb[0:P_recv, ci * P_MAX : (ci + 1) * P_MAX])
        # (slice, h, t) -> (h, slice, t)
        nisa.tensor_copy(
            dst=ctx_t_sb.ap(pattern=[[nC * width, P_MAX], [T_all, nh_out], [T, slices], [1, T]], offset=ci * width),
            src=ct_psum.ap(pattern=[[P_recv, P_MAX], [T, nh_out], [R, slices], [1, T]], offset=0),
            engine=nisa.vector_engine,
        )
    return ctx_t_sb


def _cdp_kernel():
    """The chunk-dealt pool's score kernel, ``indexer_kv_score.indexer_kv_score_cdp``."""
    from .attention.indexer.indexer_kv_score import indexer_kv_score_cdp

    return indexer_kv_score_cdp


def _view_cols(cols: int, src_dtype, dst_dtype) -> int:
    """Columns of a row of ``cols`` ``src_dtype`` elements viewed as ``dst_dtype`` (4-byte words <-> bf16 pairs)."""
    wide = (nl.uint32, nl.int32, nl.float32)
    if src_dtype in wide and dst_dtype == nl.bfloat16:
        return 2 * cols
    if src_dtype == nl.bfloat16 and dst_dtype in wide:
        kernel_assert(cols % 2 == 0, f"{_ERROR_PREFIX} {cols} bf16 columns do not pack into 4-byte words")
        return cols // 2
    return cols


def _cdp_cand_dst(dst: nl.NkiTensor, dtype, off, rows: int, topk: int):
    """Where one chunk select's ``[rows, topk]`` candidates go (``select_to``; see ``_pooled_chunk_indexer``).

    ``dst`` is a whole 2-D tensor (the send buffer, its bf16 / int32 views, or a phase-1 buffer), viewed as ``dtype``
    when it is not (the send buffer's 4-byte words as bf16 pairs for the values, as int32 for the positions). ``off``
    is the [1, 1] int32 SBUF element offset of row 0 in that flat view, rows one view row apart; None is offset 0
    (static: columns ``[0, topk)`` of the first ``rows`` rows)."""
    if off is None:
        return (dst if dst.dtype == dtype else dst.view(dtype))[0:rows, 0:topk]
    n_rows, n_cols = dst.shape
    v_cols = _view_cols(n_cols, dst.dtype, dtype)
    flat = dst.reshape((1, n_rows * n_cols))
    if dst.dtype != dtype:
        flat = flat.view(dtype)
    return flat.ap(pattern=[[v_cols, rows], [1, topk]], offset=0, scalar_offset=off, indirect_dim=1)


def _cdp_snake_loads(scores: nl.NkiTensor, shapes, two_queues: bool):
    """``indexer_topk_common.load_snake`` (the same per-row DMAs on the same queues), also returning the DMAs, which
    the slot copies are ordered behind: ``(snake tile, [dma, ...])``."""
    n_cand, n_cols = shapes.num_candidates, shapes.snake_columns
    snake = nl.ndarray((shapes.snake_partitions, n_cols), dtype=nl.bfloat16, buffer=nl.sbuf)
    dmas = []
    for row in range(shapes.num_tokens):
        base = row * _SNAKE_PARTITIONS
        queue = (
            {"dge_mode": nisa.dge_mode.hwdge, "engine": nisa.engine.scalar if row % 2 else nisa.engine.sync}
            if two_queues
            else {}
        )
        dmas.append(nisa.dma_copy(
            dst=snake[base : base + _SNAKE_PARTITIONS, 0:n_cols],
            src=scores[row, 0:n_cand].reshape((_SNAKE_PARTITIONS, n_cols)),
            **queue,
        ))
    return snake, dmas


def _cdp_slot_copies(recv: nl.NkiTensor, phase1: tuple, unrot_sb: nl.NkiTensor, R: int, T: int, topk: int, name: str,
                     after=None):
    """Phase-1 slot ``q`` (``q = 1..R-1``, the owner's chunk q) <- the AllToAll part of its holder ``(e + q) % E``.

    The host's ``idx_pool_unrot`` row q holds that part's (value, position) element offsets in the flat bf16 / int32
    views of ``recv``; 2 (R - 1) HBM->HBM copies on the Scalar hardware DGE queue. Slot 0 is the own select's."""
    cand_v, cand_p = phase1
    n_rows, n_words = recv.shape
    flat = recv.reshape((1, n_rows * n_words))
    views = ((cand_v, flat.view(nl.bfloat16), _view_cols(n_words, recv.dtype, nl.bfloat16)),
             (cand_p, flat.view(nl.int32), _view_cols(n_words, recv.dtype, nl.int32)))
    for q in range(1, R):
        for k, (dst, src, cols) in enumerate(views):
            dma = nisa.dma_copy(
                dst=dst[0:T, q * topk : (q + 1) * topk],
                src=src.ap(
                    pattern=[[cols, T], [1, topk]], offset=0, scalar_offset=unrot_sb[0:1, 2 * q + k : 2 * q + k + 1],
                    indirect_dim=1,
                ),
                dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.scalar,
                name=f"{name}_slot{q}_{'vp'[k]}",
            )
            if after:
                dma.depends_on(*after)  # behind the own select's snake loads


def _pooled_chunk_indexer(
    q_idx: nl.NkiTensor,
    hidden_seq: nl.NkiTensor,
    idx_wk: nl.NkiTensor,
    idx_k_norm_w: nl.NkiTensor,
    idx_k_norm_b: nl.NkiTensor,
    idx_weights_proj: nl.NkiTensor,
    idx_cos: nl.NkiTensor,
    idx_sin: nl.NkiTensor,
    idx_k_cache: nl.NkiTensor,
    idx_mask_add: nl.NkiTensor,
    prologue: tuple,
    key_offset: nl.NkiTensor,
    pool_replica_group: ReplicaGroup,
    pool_degree: int,
    pool_tokens: int,
    idx_pool_items: nl.NkiTensor,
    idx_pool_n_items: nl.NkiTensor,
    idx_pool_new_key_rows: nl.NkiTensor,
    idx_pool_unrot: nl.NkiTensor,
    idx_shard_degree: int,
    idx_block_size: int,
    idx_n_heads: int,
    idx_rope_dim: int,
    index_topk: int,
    spec: int,
    idx_norm_eps: float,
) -> tuple[nl.NkiTensor, nl.NkiTensor, tuple]:
    """The chunk-dealt pooled indexer (``layout/decode/idx_pool.py``): this rank's phase-1 candidates, each 32,768-key chunk
    scored and selected on the pool member that holds it.

    Chunk ``q`` of pool slot ``s`` lives in region ``q`` of member ``(s + q) % E``. The score kernel
    (``indexer_kv_score_cdp``) runs the chunk loop over the own chunk and the live foreign ones; this function supplies
    its hooks: ``gather_pool`` (the pool AllGather of the published query, weights and new keys), ``select_to`` (the
    chunk select into a part of the send buffer, or slot 0), ``exchange`` (the candidate AllToAll over the pool; the slot
    copies of the received parts are traced with the own select) and ``own_slot``. Phase 2 stays in
    :func:`_run_indexer`.

    Returns:
        ``(index_scores, idx_k_cache, phase1)``: an unwritten score tensor whose shape the phase-2 select reads, the
        updated cache, and the filled phase-1 buffers.
    """
    from .attention.indexer.indexer_constants import IDX_SELECT_2Q, get_indexer_topk_shape
    from .attention.indexer.indexer_topk_local import indexer_topk_local_chunk
    from .attention.indexer.indexer_topk_local_carry_index import _pick_local_chunks, chunk_phase1_alloc

    E, T = pool_degree, pool_tokens
    kernel_assert(
        E > 1 and T == spec,
        f"{_ERROR_PREFIX} the chunk-dealt pool needs a pool of >= 2 and idx_pool_tokens == spec, got E={E} T={T}",
    )
    kernel_assert(IDX_CAND_DIRECT, f"{_ERROR_PREFIX} the chunk-dealt pool needs GLM52K_IDX_CAND_DIRECT=1")
    kernel_assert(
        prologue is not None and q_idx is not None,
        f"{_ERROR_PREFIX} the chunk-dealt pool publishes the front's stationaries: needs the indexer front",
    )
    n_keys = indexer_cache_keys(idx_k_cache)
    # R regions (top-k chunks) a rank, at most one chunk of a sequence per member (8 = E at 1m).
    R, chunk_keys = _pick_local_chunks(n_keys, index_topk)
    kernel_assert(
        R <= E and tuple(idx_pool_unrot.shape) == (R, 2),
        f"{_ERROR_PREFIX} the chunk-dealt pool holds at most one top-k chunk of a sequence per member: {n_keys} keys "
        f"are {R} chunks of {chunk_keys}, the pool has {E} members; idx_pool_unrot {tuple(idx_pool_unrot.shape)}",
    )
    words = index_topk // 2 + index_topk
    name = unique_name("idx_cdp")
    # Every slot is written (slot 0 by the own select, slots 1.. by the copies, dead ones carry the sender's
    # sentinel), so no sentinel fill.
    phase1 = chunk_phase1_alloc(spec, R, index_topk, sentinel=False)
    # The un-rotation offsets, loaded up front (one tiny DMA off every queue's critical path).
    unrot_sb = nl.ndarray((1, 2 * R), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=unrot_sb[0:1, 0 : 2 * R], src=idx_pool_unrot.reshape((1, 2 * R)), name=f"{name}_unrot")
    pending = []

    def flush(after=None):
        while pending:
            pending.pop(0)(after)

    def gather_pool(publish):
        pool_qw = _gather_rows(publish, pool_replica_group, E, "idx_cdp_qw")
        return pool_qw

    def select_to(chunk_scores, dst_v, dst_p, off_v, off_p, label_p):
        # The slot copies go with the own select, the only select after the exchange: behind its snake loads (the
        # same loads indexer_topk_local_chunk would issue) and ahead of its top-k.
        snake, snake_dmas = None, None
        if pending:
            tshape = get_indexer_topk_shape(chunk_scores, index_topk)
            snake, snake_dmas = _cdp_snake_loads(chunk_scores, tshape, IDX_SELECT_2Q)
        flush(snake_dmas)
        cand_name = unique_name("idx_cdp_cand")
        cands = indexer_topk_local_chunk(
            chunk_scores,
            index_topk,
            0,
            key_offset,
            idx_shard_degree,
            idx_block_size,
            extra_offset=label_p,
            vector_inverse=IDX_SELECT_VEC,
            src_snake=snake,
        )
        for k, (src, dst, off) in enumerate(zip(cands, (dst_v, dst_p), (off_v, off_p))):
            nisa.dma_copy(
                dst=_cdp_cand_dst(dst, src.dtype, off, T, index_topk),
                src=src.ap(pattern=[[index_topk, T], [1, index_topk]], offset=0),
                name=f"{cand_name}_st{k}",
                **({"dge_mode": nisa.dge_mode.hwdge, "engine": nisa.engine.sync} if IDX_CAND_HWDGE else {}),
            )

    def exchange(send, after=None):
        kernel_assert(
            tuple(send.shape) == (E * T, words) and send.dtype in (nl.uint32, nl.int32),
            f"{_ERROR_PREFIX} the CDP send buffer must be [{E * T}, {words}] 4-byte words, got {tuple(send.shape)} "
            f"{send.dtype}",
        )
        recv = cc_dst((E * T, words), send.dtype, _COLLECTIVE_HBM, "idx_cdp_recv")
        _pin_all(send, recv)  # while pinning is on; see utils/cc_pin.py
        a2a = ncc.all_to_all(srcs=[send], dsts=[recv], replica_group=pool_replica_group, collective_dim=0)
        if after:
            a2a.depends_on(*after)  # the own blocks' preloaded key loads land first
        pending.append(lambda after=None: _cdp_slot_copies(recv, phase1, unrot_sb, R, T, index_topk, name, after=after))
        return recv

    own_slot = phase1
    idx_k_cache = _cdp_kernel()(
        q=q_idx,
        hidden_states=hidden_seq,
        wk=idx_wk,
        k_norm_weight=idx_k_norm_w,
        k_norm_bias=idx_k_norm_b,
        weights_proj=idx_weights_proj,
        cos=idx_cos,
        sin=idx_sin,
        k_cache=idx_k_cache,
        mask_add=idx_mask_add,
        num_heads=idx_n_heads,
        rope_dim=idx_rope_dim,
        norm_eps=idx_norm_eps,
        prologue=prologue,
        items=idx_pool_items,
        n_items=idx_pool_n_items,
        new_key_rows=idx_pool_new_key_rows,
        shard_degree=idx_shard_degree,
        block_size=idx_block_size,
        store_keys=chunk_keys,
        gather_pool=gather_pool,
        select_to=select_to,
        exchange=exchange,
        own_slot=own_slot,
        index_topk=index_topk,
    )
    flush()  # a kernel whose own select did not go through select_to after the exchange
    index_scores = nl.ndarray((spec, n_keys), dtype=nl.bfloat16, buffer=layer_scratch_hbm())
    return index_scores, idx_k_cache, phase1


def _run_indexer(
    q_latent_normed: nl.NkiTensor,
    after_q_gather,
    hidden: nl.NkiTensor,
    idx_wq_b: nl.NkiTensor,
    idx_wk: nl.NkiTensor,
    idx_k_norm_w: nl.NkiTensor,
    idx_k_norm_b: nl.NkiTensor,
    idx_weights_proj: nl.NkiTensor,
    idx_cos: nl.NkiTensor,
    idx_sin: nl.NkiTensor,
    idx_k_cache: nl.NkiTensor,
    idx_new_slot_mapping: nl.NkiTensor,
    idx_mask_add: nl.NkiTensor,
    idx_block_rows: nl.NkiTensor,
    idx_shard_degree: int,
    idx_block_size: int,
    key_offset: nl.NkiTensor,
    seq_replica_group: ReplicaGroup,
    key_replica_group: ReplicaGroup,
    num_seqs_local: int,
    key_size: int,
    idx_n_heads: int,
    idx_rope_dim: int,
    index_topk: int,
    spec: int,
    idx_norm_eps: float,
    idx_q_shard: int,
    k_cols: int,
    pool_replica_group: ReplicaGroup = None,
    idx_pool_degree: int = 0,
    idx_pool_tokens: int = 0,
    q_idx: nl.NkiTensor | None = None,
    prologue: tuple | None = None,
    split_merge: bool = False,
    own_selection: bool = False,
    idx_positions: nl.NkiTensor | None = None,
    idx_pool_items: nl.NkiTensor = None,
    idx_pool_n_items: nl.NkiTensor = None,
    idx_pool_new_key_rows: nl.NkiTensor = None,
    idx_pool_unrot: nl.NkiTensor = None,
) -> tuple[nl.NkiTensor, nl.NkiTensor]:
    """Run the DSA indexer on the ``seq x key`` view and return the selection for this rank's tokens. The sequence scatter
    in and the selection gather out use the same group and are exact inverses, so the gather lands exactly this rank's
    ``T_local`` query rows.

    Args:
        after_q_gather: a trace-time hook run right after the query gather is issued.
        q_latent_normed, hidden: [T_local, q_lora_rank] normed query latent and [T_local, H] normed rows.
        idx_wq_b, idx_wk, idx_k_norm_w, idx_k_norm_b, idx_weights_proj, idx_cos, idx_sin: the indexer weights and RoPE
            tables.
        idx_k_cache, idx_new_slot_mapping, idx_mask_add: this rank's key cache (updated), the new keys' rows and the
            residency mask.
        key_offset, idx_shard_degree, idx_block_size: the key deal.
        seq_replica_group, key_replica_group, num_seqs_local, key_size: the indexer's groups and degrees.
        idx_n_heads, idx_rope_dim, index_topk, spec, idx_norm_eps, idx_q_shard, k_cols: dimensions.
        q_idx: the already gathered query (the indexer front), or None to project it here.
        prologue: :func:`indexer_key_prologue`'s result.
        own_selection: a CP rank's own share of the global top-k is its selection (no shares gather, no merge).
        split_merge: stop after the candidate gather and return a callable that traces the merge.
        idx_positions: the step positions for the dynamic score loop, or None.
        idx_pool_items, idx_pool_n_items, idx_pool_new_key_rows, idx_pool_unrot: the chunk-dealt pool's tables, with
            ``idx_pool_degree``.

    Returns:
        ``(gathered, gathered_km)``: [T_local, index_topk] int32 selected global positions (-1 for no candidate), and the
        same selection key-major for ``build_prior_index`` (None on the plain path).
    """
    # This rank's ONE sequence, scattered out of the token slice it shares.
    if q_idx is None:
        q_resid = _scatter_rows(q_latent_normed, seq_replica_group, num_seqs_local, spec, "idx_q_resid")
    hidden_seq = _scatter_rows(hidden, seq_replica_group, num_seqs_local, spec, "idx_hidden")

    # ═══════ The query up-projection is column-sharded over the key group ═══════
    # RoPE rotates whole heads and the head axis is reduced by the weighted sum, so the score kernel
    # needs every head -- which is why this gather cannot be deferred past it.
    if q_idx is None:
        q_idx = _all_gather_columns(
            indexer_q_proj(q_resid=q_resid, wq_b=idx_wq_b), key_replica_group, key_size, spec, idx_q_shard, "idx_q"
        )
    after_q_gather()

    # The scores stay in SBUF as the local top-k's snake where the layout allows (only that stage reads them).
    snake_scores = snake_score_layout(indexer_cache_keys(idx_k_cache), spec)
    idx_keys_local = indexer_cache_keys(idx_k_cache)
    phase1 = None
    # The chunk-dealt pool (CDP): its own chunk loop and selects (_pooled_chunk_indexer).
    cdp = idx_pool_items is not None
    kernel_assert(
        not cdp or (idx_pool_degree and local_topk_chunked(idx_keys_local)),
        f"{_ERROR_PREFIX} the chunk-dealt pool needs idx_pool_degree and a chunked local top-k "
        f"({idx_keys_local} keys a rank)",
    )
    if cdp:
        kernel_assert(not snake_scores, f"{_ERROR_PREFIX} the chunk-dealt pool hands phase 1 over, not the snake")
        index_scores, idx_k_cache, phase1 = _pooled_chunk_indexer(
            q_idx, hidden_seq, idx_wk, idx_k_norm_w, idx_k_norm_b, idx_weights_proj, idx_cos, idx_sin,
            idx_k_cache, idx_mask_add, prologue, key_offset, pool_replica_group, idx_pool_degree, idx_pool_tokens,
            idx_pool_items, idx_pool_n_items, idx_pool_new_key_rows, idx_pool_unrot, idx_shard_degree, idx_block_size,
            idx_n_heads, idx_rope_dim, index_topk, spec, idx_norm_eps,
        )
    else:
        kernel_assert(not idx_pool_degree, f"{_ERROR_PREFIX} a pooled indexer is chunk-dealt: needs idx_pool_items")
        index_scores, idx_k_cache = indexer_kv_score(
            q=q_idx,
            hidden_states=hidden_seq,
            wk=idx_wk,
            k_norm_weight=idx_k_norm_w,
            k_norm_bias=idx_k_norm_b,
            weights_proj=idx_weights_proj,
            cos=idx_cos,
            sin=idx_sin,
            k_cache=idx_k_cache,
            new_slot_mapping=idx_new_slot_mapping,
            mask_add=idx_mask_add,
            num_heads=idx_n_heads,
            rope_dim=idx_rope_dim,
            # ZERO, not this rank's key range start: the residency threshold rides idx_mask_add and the
            # position labels are offset below. See the module docstring.
            key_base=0,
            norm_eps=idx_norm_eps,
            # None keeps the position-ordered read; a table makes the same cache tensor paged.
            block_rows=idx_block_rows,
            snake_out=snake_scores,
            prologue=prologue,
        )

    # ═══════ Stage 1, then the candidate gathers ═══════
    # Each shard offers the FULL budget, not a fraction: that is what makes the merge exact. The local select writes the
    # packed hand-off tile itself (indexer_topk_local_packed), and one all_gather of it feeds the packed merge.
    from .attention.indexer.indexer_topk_merge_carry_index import indexer_topk_merge_carry_index_packed

    from .attention.indexer.indexer_topk_local import indexer_topk_local_packed

    wide_merge = False
    if local_topk_chunked(indexer_cache_keys(idx_k_cache)):
        # A shard wider than one select may take (IDX_TOPK_MAX_CALL_N): the carry-index local ranks it in chunks,
        # dealing each chunk's winners and folding key_offset as indexer_topk_local_packed does, and publishes the
        # same packed tile. The shards then trade their top-k scores first (the wide rounds), and the merge compacts
        # each shard's share of the global top-k. The key count comes from the cache: under the snake layout
        # index_scores is 16x narrower.
        from .attention.indexer.indexer_topk_local_carry_index import indexer_topk_local_carry_index_packed

        kernel_assert(not snake_scores, f"{_ERROR_PREFIX} a chunked local select reads the HBM scores")
        wide_merge = True
        packed_local = indexer_topk_local_carry_index_packed(
            index_scores=index_scores,
            topk=index_topk,
            key_offset=key_offset,
            shard_degree=idx_shard_degree,
            block_size=idx_block_size,
            phase1=phase1,
            wide_gather=lambda tile, name: _gather_rows(tile, key_replica_group, key_size, name),
        )
    elif snake_scores:
        packed_local = indexer_topk_local_packed(
            index_scores=None,
            topk=index_topk,
            key_offset=key_offset,
            src_snake=index_scores,
            shard_degree=idx_shard_degree,
            block_size=idx_block_size,
        )
    else:
        packed_local = indexer_topk_local_packed(
            index_scores=index_scores,
            topk=index_topk,
            key_offset=key_offset,
            shard_degree=idx_shard_degree,
            block_size=idx_block_size,
        )
    # Under the wide rounds each rank's tile is already its share of the global top-k; where its key shard is the
    # keys it holds, that share is its selection (see own_selection), so the shares are not gathered or merged.
    own_share = wide_merge and own_selection
    packed_all = None if own_share else _gather_rows(packed_local, key_replica_group, key_size, "cand_packed_all")

    def merge():
        if own_share:
            from .attention.indexer.indexer_topk_merge_carry_index import indexer_topk_own_selection_packed

            selected, selected_key_major = indexer_topk_own_selection_packed(packed_local, index_topk)
        else:
            selected, selected_key_major = indexer_topk_merge_carry_index_packed(
                gathered=packed_all, topk=index_topk, degree=key_size,
                block_size=idx_block_size if idx_shard_degree > 1 else 0, wide=wide_merge,
            )
        gathered = _gather_rows(
            selected.reshape((spec, _SNAKE_PARTITIONS * k_cols)),
            seq_replica_group,
            num_seqs_local,
            "gather_indices",
        )
        gathered_km = _gather_rows(
            selected_key_major.reshape((P_MAX, spec * (index_topk // P_MAX))),
            seq_replica_group,
            num_seqs_local,
            "gather_indices_km",
        )
        return gathered, gathered_km

    return merge if split_merge else merge()


def _validate_attention_layer(
    T_shard: int,
    T_local: int,
    num_seqs_local: int,
    spec: int,
    cp_degree: int,
    head_shard: int,
    token_shard: int,
    key_size: int,
    index_topk: int,
    idx_q_shard: int,
    idx_n_heads: int,
    idx_head_dim: int,
    positions_len: int,
    num_heads: int,
    nh_out: int,
    o_proj_rows: int,
    v_head_dim: int,
    token_replica_group: ReplicaGroup,
    head_replica_group: ReplicaGroup,
    key_replica_group: ReplicaGroup,
    run_indexer: bool,
) -> None:
    """Refuse shardings and shapes this layer does not handle. The shard factorization and the exit's divisibility are
    checked here because a wrong one still runs, with tokens silently assigned to the wrong ranks; a sub-kernel only sees
    one rank's shapes. ``run_indexer`` is False on a shared layer, which has no indexer weights to check.
    """
    token_group = cp_degree * head_shard

    kernel_assert(
        nl.program_ndim() == 0 or nl.program_ndim() == 1,
        f"{_ERROR_PREFIX} only no specialization or specialization along one axis is supported, got "
        f"nl.program_ndim()={nl.program_ndim()}",
    )
    n_programs = nl.num_programs(axes=0) if nl.program_ndim() != 0 else 1
    kernel_assert(n_programs == 1, f"{_ERROR_PREFIX} runs on a single NeuronCore (LNC=1), got {n_programs} programs")

    kernel_assert(
        T_local <= nl.tile_size.pmax,
        f"{_ERROR_PREFIX} this rank's token slots must fit one partition tile: T_shard={T_shard} x "
        f"cp_degree={cp_degree} x head_shard={head_shard} = {T_local} > {nl.tile_size.pmax}",
    )
    kernel_assert(
        positions_len == T_local,
        f"{_ERROR_PREFIX} positions must cover this rank's slots: got {positions_len}, expected {T_local}",
    )
    kernel_assert(
        num_seqs_local * spec == T_local,
        f"{_ERROR_PREFIX} num_seqs_local={num_seqs_local} x spec={spec} must equal T_local={T_local}. "
        f"spec is the PADDED speculation length, so pick it to make this exact -- 8 for GLM-5.2's 7. A "
        f"token shard must also land on a whole number of sequences, or a rank would hold half of one "
        f"and the indexer could not run on it.",
    )

    # THE EXIT, which is what pins the linearization. Rank i must finish owning global slots
    # [T_shard*i, T_shard*(i+1)); the CP scatter hands it T_local/cp_degree rows and the head scatter
    # T_shard of those, so the two compose only when T_local/cp_degree == T_shard*head_shard.
    kernel_assert(
        T_local % cp_degree == 0 and (T_local // cp_degree) == T_shard * head_shard,
        f"{_ERROR_PREFIX} the exit does not line up: T_local={T_local} / cp_degree={cp_degree} must "
        f"equal T_shard={T_shard} x head_shard={head_shard}",
    )
    # THE OUTPUT HEAD SHARD. o_proj and W_UV are sharded over the CP group, or over a block group of a
    # multiple of cp_degree consecutive ranks -- which is legitimate only because that group's
    # collective (ReduceScatter or all_to_all) is what hands each rank its head slice, so a rank's slice
    # is chosen by its GROUP POSITION and never by a scalar it computes (see the module docstring).
    kernel_assert(
        nh_out > 0 and num_heads % nh_out == 0 and (num_heads // nh_out) % cp_degree == 0,
        f"{_ERROR_PREFIX} the output projection is head-sharded over the CP group or a block of token "
        f"slices, so out_absorb_w must carry num_heads={num_heads} / (k x cp_degree={cp_degree}) heads, "
        f"got {nh_out}",
    )
    kernel_assert(
        o_proj_rows == nh_out * v_head_dim,
        f"{_ERROR_PREFIX} o_proj_w must hold this rank's OUTPUT head rows: got {o_proj_rows}, expected "
        f"nh_out={nh_out} x v_head_dim={v_head_dim} = {nh_out * v_head_dim}",
    )
    kernel_assert(
        num_seqs_local * key_size == token_group,
        f"{_ERROR_PREFIX} a token group's {token_group} ranks must factor as num_seqs_local="
        f"{num_seqs_local} x key_size={key_size}, or the indexer cannot cover its slice's sequences",
    )

    # nisa.topk spends 16 partitions per query row, so the indexer's row count is hard-capped at 8.
    kernel_assert(
        spec * _SNAKE_PARTITIONS <= nl.tile_size.pmax,
        f"{_ERROR_PREFIX} spec={spec} query rows need {spec * _SNAKE_PARTITIONS} top-k partitions, "
        f"over the {nl.tile_size.pmax} available",
    )
    kernel_assert(
        index_topk % nl.tile_size.pmax == 0,
        f"{_ERROR_PREFIX} index_topk={index_topk} must be a multiple of {nl.tile_size.pmax}",
    )
    if run_indexer:
        kernel_assert(
            idx_q_shard * key_size == idx_n_heads * idx_head_dim,
            f"{_ERROR_PREFIX} idx_q_shard={idx_q_shard} x key_size={key_size} must equal the complete "
            f"indexer query width idx_n_heads={idx_n_heads} x idx_head_dim={idx_head_dim}",
        )
        kernel_assert(
            all(len(g) == key_size for g in key_replica_group._value),
            f"{_ERROR_PREFIX} key_replica_group must have {key_size} ranks per group, got {key_replica_group._value}",
        )

    # Group SHAPES, checked rather than trusted: token and head must be consecutive blocks because
    # their gather/scatter order is what maps a rank onto the global token slots it owns, and key must
    # be in increasing key-shard order because _all_gather_columns reads group position as column
    # position.
    kernel_assert(
        all(
            len(g) == token_group and list(g) == list(range(g[0], g[0] + token_group))
            for g in token_replica_group._value
        ),
        f"{_ERROR_PREFIX} token_replica_group must be consecutive blocks of {token_group} ranks, got "
        f"{token_replica_group._value}",
    )
    kernel_assert(
        all(
            len(g) == head_shard and list(g) == list(range(g[0], g[0] + head_shard)) for g in head_replica_group._value
        ),
        f"{_ERROR_PREFIX} head_replica_group must be consecutive blocks of {head_shard} ranks, got "
        f"{head_replica_group._value}",
    )
    kernel_assert(
        token_shard * token_group == token_shard * cp_degree * head_shard,
        f"{_ERROR_PREFIX} token_shard={token_shard} is inconsistent with cp_degree={cp_degree} and "
        f"head_shard={head_shard}",
    )
