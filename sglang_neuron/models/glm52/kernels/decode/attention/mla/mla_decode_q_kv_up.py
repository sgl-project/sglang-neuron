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

"""
GLM-5.2 MLA decode: everything between the down-projection AllGather and the attention core.

The boundary is set by the norms: ``q_norm`` and ``kv_norm`` reduce over the axis the down-projection
shards, and RoPE rotates ``k_pe`` in pairs a narrow shard would split, so all three need the complete
axis. Everything after is per head: a rank projects, rotates and absorbs only the heads whose ``wq_b``
columns it holds, and the kernel's one collective gathers the finished ``q_absorbed`` / ``q_pe``.
Gathering after the absorb keeps it from being replicated, since every rank of the group attends over
all heads.
"""

import os

import nki
import nki.collectives as ncc
import nki.isa as nisa
import nki.language as nl
from nki.collectives import ReplicaGroup

from ...utils.cc_dst_ring import cc_dst
from ...utils.cc_pin import pin_attn
from ...utils.kernel_assert import kernel_assert
from ...utils.kernel_helpers import div_ceil
from ...utils.scratch_hbm import layer_scratch_hbm
from ...utils.small_loads import small_load
from ...utils.tiled_range import TiledRange
from .mla_decode_constants import F_MAX, P_MAX
from .mla_proj_constants import QKvUpShape, get_q_kv_up_shape
from .mla_proj_primitives import (
    PSUM_BANKS,
    linear,
    load_gamma_broadcast,
    load_rope_tables,
    pe_column_band,
    rmsnorm_inplace,
    rope_inplace,
    transpose_into_k_tiles,
)

# The head-sharded query path's schedule, all bit-identical:
# - The spread q-norm's transpose is evicted contiguously, and ``linear`` reads k-tile ``j`` as the stride-``P_MAX / T``
#   columns ``j, j + P_MAX / T, ...`` instead of permuting it on a 2-byte strided eviction.
# - The kv norm, rope and ``kv_row_new`` store are traced between the q norm and ``wq_b``, with the kv load ordered
#   after the q-norm eviction. Their SBUF tiles then do not reuse the absorb gather sources' addresses (a WAR edge
#   that would hold Sync until the q_pe source store lands), and the core's first gathers, which wait for these
#   rows, start early.
# - Where ``_wqb_banded_fits``, the ``wq_b`` result stays in ``linear``'s banded layout (head ``2b + hh`` of chunk ``k``
#   at partitions ``32b .. 32b + T``): bank-wide evictions with no band DMAs, RoPE on a few tall head blocks, and one
#   full-height transpose per 4 heads. The transposes are pass-through, so the stale band rows past T only reach
#   columns nothing reads. The evictions run on ACT, idle there while Vector runs the core's prologue.

LAT_QSTAGE_DGE = os.environ.get("GLM52K_LAT_QSTAGE_DGE", "")
"""Queue of the two stage DMAs in :func:`_stage_absorbed_blocks_split` with one head per rank (the block
``all_to_all``'s sources, which its trigger waits for). ``sync``: the Sync HWDGE queue; ``swdge``: a software-DGE
DMA on the GpSimd queue (slower); empty (default): the compiler's choice, the ACT queue, where they sit behind the
core's mask-bias row stores. Same bytes."""

_ABSORB_HEAD_CHUNK = 8
"""Most heads per absorb chunk of :func:`_absorb_head_chunk`: one weight DMA per nope tile and one PSUM eviction
per (chunk, tensor) instead of one per head. :func:`_absorb_chunk` shrinks it until the chunk's accumulator fits
one PSUM bank."""
_PSUM_TOKEN_ALIGN = 8
"""Token stride of the chunk-wide PSUM tiles, so per-head blocks start 16 B (bf16) or 32 B (fp32) apart; the
verifier rejects the 2 B spacing a one-token block would get."""
_PSUM_BANK_BYTES = F_MAX * 4
"""One PSUM bank per partition: 512 fp32."""
_BAND = P_MAX // 4
"""``linear``'s PE column band at T <= 32, the partition stride of the banded ``wq_b`` layout."""


def _pin_all(*tensors):
    """Pin every collective operand while the attention pins are on (long context; see ``utils/cc_pin.py``)."""
    pin_attn(*tensors)


@nki.jit
def mla_decode_q_kv_up(
    q_latent: nl.NkiTensor,
    kv_row: nl.NkiTensor,
    q_norm_w: nl.NkiTensor,
    kv_norm_w: nl.NkiTensor,
    wq_b: nl.NkiTensor,
    q_absorb_w: nl.NkiTensor,
    cos: nl.NkiTensor,
    sin: nl.NkiTensor,
    num_heads: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    kv_lora_rank: int,
    norm_eps: float = 1e-6,
    replica_group: ReplicaGroup | None = None,
    q_latent_block: nl.NkiTensor | None = None,
    cos_block: nl.NkiTensor | None = None,
    sin_block: nl.NkiTensor | None = None,
    after_q_norm=None,
    before_up_proj=None,
    need_q_latent_normed: bool = True,
    q_block_split: bool = False,
) -> tuple[nl.NkiTensor, nl.NkiTensor, nl.NkiTensor, nl.NkiTensor]:
    """
    Normalize the gathered latents, project the query up, apply RoPE, and absorb W_UK.

    Usage:
        Tuned for: GLM-5.2 MLA decode, wq_b column-sharded on heads, latents after the AllGather, LNC=1
        Applicable to: any absorbed-MLA front-fold whose down-projections were column-sharded

    ``q_latent_normed`` is published for the DSA indexer, which takes the same normed latent. The
    down-projection's output cannot stand in for it: that one is un-normed, because the norm has to wait
    for the AllGather.

    The query outputs are feature-major, the layout ``mla_decode_attention``'s score matmuls index:
    feature ``c`` of token ``t``, head ``h`` at ``[c % P_MAX, c // P_MAX, t, h]``.

    ``kv_row_new`` stays logical bf16 and is not written to the cache: the attention core quantizes it
    to fp8, packs the 640-byte row and appends it, so the layout lives in one module. The row is
    ``[k_pe | kv_latent]``, the reverse of the reference implementation's order; the ``wkv_a`` columns fix it, and
    ``k_pe`` first saves the core a transpose.

    Args:
        q_latent (nl.NkiTensor): [T, q_lora_rank] @ HBM. The complete query latent, gathered from every
            rank's column slice. Completeness is asserted, since a forgotten AllGather would quietly
            scale the norm by the shard degree.
        kv_row (nl.NkiTensor): [T, R + C] @ HBM. The complete cache row ``cat([k_pe, kv_latent])``, not
            yet normed or rotated.
        q_norm_w (nl.NkiTensor): [1, q_lora_rank] fp32 @ HBM. RMSNorm gamma for the query latent.
        kv_norm_w (nl.NkiTensor): [1, C] fp32 @ HBM. RMSNorm gamma for the KV latent field only; the
            model does not normalize ``k_pe``.
        wq_b (nl.NkiTensor): [q_lora_rank, cols] @ HBM. This rank's columns of the query up-projection,
            a whole number of heads dividing ``num_heads * (qk_nope + qk_rope)``: the full width runs
            unsharded, ``1 / degree`` of it with a ``replica_group`` of that degree shards it. The shard
            is on the output, so the result is bit-identical to the unsharded kernel.
        q_absorb_w (nl.NkiTensor): [num_heads / degree, qk_nope, C] @ HBM. The nope half of ``wkv_b`` for
            the heads whose ``wq_b`` columns this rank holds (group position ``r`` holds heads
            ``[r * num_heads / degree, (r + 1) * num_heads / degree)``).
        cos (nl.NkiTensor): [T, qk_rope // 2] fp32 @ HBM. Interleaved RoPE cosine table.
        sin (nl.NkiTensor): [T, qk_rope // 2] fp32 @ HBM. Interleaved RoPE sine table.
        num_heads (int): Query heads this rank attends over, the outputs' head width; it projects and
            absorbs ``num_heads / degree`` of them.
        qk_nope_head_dim (int): Per-head QK width RoPE does not rotate.
        qk_rope_head_dim (int): Per-head QK width RoPE rotates. Must be even.
        kv_lora_rank (int): Latent width C.
        norm_eps (float): Epsilon for both RMSNorms.
        replica_group (ReplicaGroup | None): The group holding ``wq_b``'s column shards, in column order.
            Required exactly when ``wq_b`` is narrower than this rank's heads. Its members hold the same
            tokens; not used with ``q_latent_block``.
        q_latent_block (nl.NkiTensor | None): [slices * T, q_lora_rank] @ HBM, the complete query latent
            of this rank's block: ``slices`` consecutive token slices, this rank's own among them. Selects
            the block-sharded projection: each of the ``degree = slices * cp`` ranks (position
            ``cp * slice + c``) projects and absorbs its ``num_heads / degree`` heads for all the block's
            tokens, which reads ``slices`` times less weight than the AllGather form. The first two outputs
            are then the staged exchange sources; the caller runs :func:`mla_decode_q_block_exchange` and
            :func:`mla_decode_q_block_finish`, and may place them after unrelated work to hide the
            collective's latency.
        cos_block (nl.NkiTensor | None): [slices * T, qk_rope // 2] fp32 @ HBM, the block's cosine table in
            ``q_latent_block``'s token order. Required with it.
        sin_block (nl.NkiTensor | None): [slices * T, qk_rope // 2] fp32 @ HBM, likewise.
        after_q_norm (callable | None): ``after_q_norm(k_tiles, stationary_fn)``, traced right after the q
            norm with the normed latent's ``linear`` K-tiles, so the indexer's query projection can go
            ahead of ``wq_b``. With ``q_latent_block`` these are this rank's own tokens, normed first.
        before_up_proj (callable | None): ``before_up_proj()``, traced right before ``wq_b``, after the q
            norm and the kv path (the indexer's scoring). With ``q_latent_block``: after the own tokens'
            norm, before the block's norm and up-projection; the kv path then runs at the end.
        need_q_latent_normed (bool): With ``q_latent_block`` and no hooks, whether to norm this rank's own
            latent for ``q_latent_normed``; None is returned for it otherwise. (default: True)
        q_block_split (bool): With ``q_latent_block``, stage the sources for the split exchange
            (:func:`_stage_absorbed_blocks_split`). (default: False)

    Returns:
        tuple[nl.NkiTensor, nl.NkiTensor, nl.NkiTensor, nl.NkiTensor]:
            - q_absorbed [P_MAX, C / P_MAX, T, num_heads] @ HBM, feature-major. With ``q_latent_block``,
              the staged ``[degree, num_heads / degree, T, C]`` exchange source.
            - q_pe [P_MAX, T, num_heads] @ HBM, feature-major. With ``q_latent_block``, the staged
              ``[degree, num_heads / degree, T, qk_rope]`` source, rotated.
            - kv_row_new [T, R + C] @ HBM, normed and rotated, logical bf16.
            - q_latent_normed [T, q_lora_rank] @ HBM, the normed query latent, or None (see
              ``need_q_latent_normed``).
    """

    shapes = get_q_kv_up_shape(
        q_latent, kv_row, wq_b, q_absorb_w, num_heads, qk_nope_head_dim, qk_rope_head_dim, kv_lora_rank
    )
    T, Nh = shapes.num_tokens, shapes.num_heads
    C, R = shapes.kv_lora_dim, shapes.qk_rope_dim
    nC = shapes.num_latent_tiles

    cos_sbuf, sin_sbuf = load_rope_tables(cos, sin, T)
    kv_row_out = None

    # This rank's slice of the query: the heads whose wq_b columns it holds.
    degree = shapes.qk_packed_dim // wq_b.shape[1]
    heads_local = Nh // degree

    if q_latent_block is not None:
        # Block-sharded projection: this rank's heads for every token of the block, then an all_to_all
        # hands each rank its own tokens. The indexer still needs this rank's own normed latent, which
        # is not a projection input here, so it is normed separately.
        T_all = q_latent_block.shape[0]
        slices = T_all // T
        kernel_assert(
            cos_block is not None
            and sin_block is not None
            and degree > 1
            and T_all == slices * T
            and degree % slices == 0,
            f"q_latent_block ({T_all} tokens) must hold a whole number of {T}-token slices and wq_b's column "
            f"degree {degree} must be a multiple of that slice count; the block's cos_block / sin_block tables "
            f"are required",
        )
        if after_q_norm is None and before_up_proj is None:
            # The indexer is the only reader of the own normed latent; a caller without one skips it.
            q_latent_normed_sbuf = _q_norm(q_latent, q_norm_w, norm_eps, shapes) if need_q_latent_normed else None
        else:
            # The indexer's query projection reads this rank's normed latent as ``linear`` K-tiles, so it
            # is normed and K-tiled as on the token-group path, and the hooks run in that path's order,
            # ahead of the block's up-projection.
            own_fn = None
            if _q_spread_fits(T, shapes.q_lora_rank):
                q_latent_normed_sbuf, own_tiles, own_fn, _ = _q_norm_and_transpose_spread(
                    q_latent, q_norm_w, norm_eps, shapes
                )
            else:
                q_latent_normed_sbuf, own_tiles = _q_norm_and_transpose(q_latent, q_norm_w, norm_eps, shapes)
            if after_q_norm is not None:
                after_q_norm(own_tiles, own_fn)
            if before_up_proj is not None:
                before_up_proj()
        shapes_block = QKvUpShape(
            num_tokens=T_all,
            num_heads=Nh,
            q_lora_rank=shapes.q_lora_rank,
            qk_nope_dim=shapes.qk_nope_dim,
            qk_rope_dim=R,
            kv_lora_dim=C,
            dtype=shapes.dtype,
        )
        q_block_sbuf, _ = _q_norm_and_up_projection(q_latent_block, q_norm_w, wq_b, norm_eps, shapes_block)
        cos_block_sbuf, sin_block_sbuf = load_rope_tables(cos_block, sin_block, T_all)
        rope_inplace(
            q_block_sbuf, cos_block_sbuf, sin_block_sbuf, shapes.qk_nope_dim, shapes.qk_head_dim, heads_local, T_all
        )
        q_abs_tm_sbuf, q_pe_tm_sbuf = _absorb_token_major(q_block_sbuf, q_absorb_w, shapes_block, heads_local)
        stage = _stage_absorbed_blocks_split if q_block_split else _stage_absorbed_blocks
        q_absorbed_out, q_pe_out = stage(q_abs_tm_sbuf, q_pe_tm_sbuf, degree, slices, heads_local, shapes)
    else:
        q_absorbed_out = nl.ndarray((P_MAX, nC, T, Nh), dtype=shapes.dtype, buffer=layer_scratch_hbm())
        q_pe_out = nl.ndarray((P_MAX, T, Nh), dtype=shapes.dtype, buffer=layer_scratch_hbm())

        def kv_path_early(q_norm_evict):
            nonlocal kv_row_out
            kv_row_sbuf = _kv_norm_and_rope(kv_row, kv_norm_w, cos_sbuf, sin_sbuf, norm_eps, shapes, q_norm_evict)
            kv_row_out = _store_kv_row(kv_row_sbuf, shapes)

        banded = _wqb_banded_fits(shapes, wq_b.shape[1])
        if banded:
            band_rows = (P_MAX // _BAND - 1) * _BAND + T
        q_sbuf, q_latent_normed_sbuf = _q_norm_and_up_projection(
            q_latent,
            q_norm_w,
            wq_b,
            norm_eps,
            shapes,
            before_wq_b=kv_path_early,
            banded=banded,
            after_q_norm=after_q_norm,
            before_up_proj=before_up_proj,
        )
        if banded:
            cos_band, sin_band = _load_rope_tables_banded(cos, sin, T)
            rope_inplace(
                q_sbuf,
                cos_band,
                sin_band,
                shapes.qk_nope_dim,
                shapes.qk_head_dim,
                q_sbuf.shape[1] // shapes.qk_head_dim,
                band_rows,
            )
        else:
            rope_inplace(q_sbuf, cos_sbuf, sin_sbuf, shapes.qk_nope_dim, shapes.qk_head_dim, heads_local, T)

        q_absorbed_local_sbuf = nl.ndarray((P_MAX, nC, T, heads_local), dtype=shapes.dtype, buffer=nl.sbuf)
        q_pe_local_sbuf = nl.ndarray((P_MAX, T, heads_local), dtype=shapes.dtype, buffer=nl.sbuf)
        # q_pe is P_MAX tall by the core's contract, but only the first qk_rope_dim partitions carry
        # values. Zero the whole tile so the published tensor is deterministic; a memset of just the
        # tail would break the quadrant-alignment rule.
        nisa.memset(q_pe_local_sbuf[0:P_MAX, 0:T, 0:heads_local], 0)
        heads_per_chunk, token_stride = _absorb_chunk(shapes)
        if banded:
            _absorb_banded(
                q_sbuf,
                q_absorb_w,
                shapes,
                heads_local,
                heads_per_chunk,
                token_stride,
                band_rows,
                q_absorbed_local_sbuf,
                q_pe_local_sbuf,
            )
        else:
            for head_chunk in TiledRange(heads_local, heads_per_chunk):
                _absorb_head_chunk(
                    q_sbuf,
                    q_absorb_w,
                    shapes,
                    head_chunk.start_offset,
                    head_chunk.size,
                    token_stride,
                    q_absorbed_local_sbuf,
                    q_pe_local_sbuf,
                )

        if degree == 1:
            nisa.dma_copy(dst=q_absorbed_out, src=q_absorbed_local_sbuf[0:P_MAX, 0:nC, 0:T, 0:Nh])
            nisa.dma_copy(dst=q_pe_out, src=q_pe_local_sbuf[0:P_MAX, 0:T, 0:Nh])
        else:
            kernel_assert(
                replica_group is not None,
                f"wq_b is column-sharded ({wq_b.shape[1]} of {shapes.qk_packed_dim} columns) but no replica_group "
                f"was given; the group's absorbed queries have to be gathered for the attention core",
            )
            _gather_absorbed_heads(
                q_absorbed_local_sbuf,
                q_pe_local_sbuf,
                replica_group,
                degree,
                heads_local,
                shapes,
                q_absorbed_out,
                q_pe_out,
            )

    if kv_row_out is None:
        kv_row_out = _store_kv_row(_kv_norm_and_rope(kv_row, kv_norm_w, cos_sbuf, sin_sbuf, norm_eps, shapes), shapes)

    # private_hbm: the layer ReduceScatters the normed latent to the indexer's sequence groups, and at
    # LNC=1 the runtime rejects a collective whose sources are shared_hbm.
    if q_latent_normed_sbuf is None:
        return q_absorbed_out, q_pe_out, kv_row_out, None
    q_latent_normed_out = nl.ndarray((T, shapes.q_lora_rank), dtype=shapes.dtype, buffer=nl.private_hbm)
    if q_latent_normed_sbuf.shape[1] == shapes.q_lora_rank:
        nisa.dma_copy(
            dst=q_latent_normed_out[0:T, 0 : shapes.q_lora_rank],
            src=q_latent_normed_sbuf[0:T, 0 : shapes.q_lora_rank],
        )
    else:
        # The partition-spread tile of _q_norm_and_transpose_spread: the same bytes, row-major.
        nisa.dma_copy(dst=q_latent_normed_out.reshape(tuple(q_latent_normed_sbuf.shape)), src=q_latent_normed_sbuf)

    return q_absorbed_out, q_pe_out, kv_row_out, q_latent_normed_out


def _q_norm_and_up_projection(
    q_latent: nl.NkiTensor,
    q_norm_w: nl.NkiTensor,
    wq_b: nl.NkiTensor,
    eps: float,
    shapes: QKvUpShape,
    before_wq_b=None,
    banded: bool = False,
    after_q_norm=None,
    before_up_proj=None,
) -> tuple[nl.NkiTensor, nl.NkiTensor]:
    """
    Normalizes the gathered query latent and projects it up to this rank's wq_b columns.

    Computes ``q = RMSNorm(q_latent, q_norm_w) @ wq_b``. Returns the packed query ``[P_MAX, wq_b.shape[1]]``
    (each head's ``[q_nope | q_pe]``) and the normed latent, which the DSA indexer also reads.

    ``linear`` caps N at ``F_MAX * PSUM_BANKS`` = 4096 columns, 16 heads of 256, so a wider shard is
    projected in whole-head groups and nothing downstream sees the split.

    ``before_wq_b(q_norm_evict)`` is traced between the norm and the projection with the q norm's last
    instruction (None off the spread path); it runs the early kv path. ``banded`` returns ``linear``'s
    banded layout.
    """

    T, q_lora_rank, head_dim = shapes.num_tokens, shapes.q_lora_rank, shapes.qk_head_dim

    stationary_fn, q_norm_evict = None, None
    if _q_spread_fits(T, q_lora_rank):  # the q-norm on the partition-spread view, one transpose
        q_latent_sbuf, q_latent_tiles_sbuf, stationary_fn, q_norm_evict = _q_norm_and_transpose_spread(
            q_latent, q_norm_w, eps, shapes
        )
    else:
        q_latent_sbuf, q_latent_tiles_sbuf = _q_norm_and_transpose(q_latent, q_norm_w, eps, shapes)
    if after_q_norm is not None:
        after_q_norm(q_latent_tiles_sbuf, stationary_fn)
    if before_wq_b is not None:
        before_wq_b(q_norm_evict)
    if before_up_proj is not None:
        before_up_proj()

    # Only this rank's columns of wq_b, so the weight read and the matmuls shrink by the shard degree.
    shard = wq_b.shape[1]

    chunk = (F_MAX * PSUM_BANKS // head_dim) * head_dim
    if banded:
        q_shard_sbuf = linear(
            q_latent_tiles_sbuf, wq_b, T, stationary_fn=stationary_fn, banded=True, evict_engine=nisa.engine.scalar
        )
    elif shard <= chunk:
        q_shard_sbuf = linear(q_latent_tiles_sbuf, wq_b, T, stationary_fn=stationary_fn)
    else:
        q_shard_sbuf = nl.ndarray((P_MAX, shard), dtype=shapes.dtype, buffer=nl.sbuf)
        for n_tile in TiledRange(shard, chunk):
            n_start, n_end, n_size = n_tile.start_offset, n_tile.end_offset, n_tile.size
            part = linear(q_latent_tiles_sbuf, wq_b[0:q_lora_rank, n_start:n_end], T, stationary_fn=stationary_fn)
            nisa.tensor_copy(dst=q_shard_sbuf[0:T, n_start:n_end], src=part[0:T, 0:n_size])

    return q_shard_sbuf, q_latent_sbuf


def _q_norm(q_latent: nl.NkiTensor, q_norm_w: nl.NkiTensor, eps: float, shapes: QKvUpShape) -> nl.NkiTensor:
    """Loads and RMS-normalizes a ``[num_tokens, q_lora_rank]`` query latent into ``[P_MAX, q_lora_rank]``."""

    T, q_lora_rank = shapes.num_tokens, shapes.q_lora_rank

    if q_latent.buffer == nl.sbuf:
        # The block latent is already on chip, token-major: norm it in place.
        kernel_assert(
            tuple(q_latent.shape) == (T, q_lora_rank) and T == P_MAX,
            f"an SBUF query latent must be [{P_MAX}, {q_lora_rank}], got {q_latent.shape}",
        )
        q_latent_sbuf = q_latent
    else:
        q_latent_sbuf = nl.ndarray((P_MAX, q_lora_rank), dtype=shapes.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=q_latent_sbuf[0:T, 0:q_lora_rank], src=q_latent[0:T, 0:q_lora_rank])

    q_gamma_sbuf = load_gamma_broadcast(q_norm_w, T)
    rmsnorm_inplace(q_latent_sbuf[0:P_MAX, 0:q_lora_rank], q_gamma_sbuf, T, eps)
    return q_latent_sbuf


def _q_spread_fits(T: int, q_lora_rank: int) -> bool:
    """Whether the latent's ``T * q_lora_rank`` elements fill ``[P_MAX, P_MAX]`` with whole tokens per partition block."""
    return T * q_lora_rank == P_MAX * P_MAX and P_MAX % T == 0


def _token_block_ones(T: int) -> nl.NkiTensor:
    """``[P_MAX, P_MAX]`` fp32, ones where both partitions belong to the same token's ``P_MAX / T`` block.

    At T = 8 the blocks are narrower than a partition quadrant and cannot be memset, so the ``[T, P_MAX]``
    token indicator is built from an iota (``f - ppt * t`` in ``[0, ppt)``) and squared on the PE.
    """
    ppt = P_MAX // T
    pos = nl.ndarray((T, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(pos, pattern=[[1, P_MAX]], offset=0, channel_multiplier=-ppt)
    lo = nl.ndarray((T, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=lo, data=pos, op0=nl.greater_equal, operand0=0.0)
    hi = nl.ndarray((T, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=hi, data=pos, op0=nl.less, operand0=float(ppt))
    indicator = nl.ndarray((T, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=indicator, data1=lo, data2=hi, op=nl.multiply)
    ones_ps = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=ones_ps, stationary=indicator, moving=indicator)
    ones_blk = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=ones_blk, src=ones_ps)
    return ones_blk


def _q_norm_and_transpose_spread(
    q_latent: nl.NkiTensor, q_norm_w: nl.NkiTensor, eps: float, shapes: QKvUpShape
) -> tuple[nl.NkiTensor, nl.NkiTensor]:
    """``_q_norm_and_transpose`` on the partition-spread view.

    Partition ``p = t * ppt + j`` holds latent columns ``[j * 128, (j + 1) * 128)`` of token ``t``. The norm
    runs on all 128 partitions (per-partition partials, one block-diagonal ones matmul, ``x * rstd * gamma``),
    and one PE transpose yields the k-tiles in ``[f, (t, j)]`` order, which ``linear`` reads through a
    strided stationary map.

    Returns:
        tuple: the normed latent ``[P_MAX, P_MAX]`` in the spread layout, the K-tiles
            ``[P_MAX, (q_lora_rank / P_MAX) * T]``, the stationary map for k-tile ``j``, and the eviction
            instruction.
    """
    T, K = shapes.num_tokens, shapes.q_lora_rank
    ppt = P_MAX // T
    x_sb = nl.ndarray((P_MAX, P_MAX), dtype=shapes.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=x_sb, src=q_latent.reshape((P_MAX, P_MAX)))

    gamma_sb = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    for t in range(T):
        small_load(dst=gamma_sb[t * ppt : (t + 1) * ppt, 0:P_MAX], src=q_norm_w[0:1, 0:K].reshape((ppt, P_MAX)))

    squares_discard = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    partial = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation_reduce(dst=squares_discard, op=nl.square, data=x_sb, reduce_op=nl.add, reduce_res=partial)
    total_ps = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(
        dst=total_ps,
        stationary=_token_block_ones(T),
        moving=partial,
    )
    inv_rms = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=inv_rms, op=nl.rsqrt, data=total_ps, scale=1.0 / K, bias=eps)

    normed_sb = nl.ndarray((P_MAX, P_MAX), dtype=shapes.dtype, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(
        dst=normed_sb, data=x_sb, op0=nl.multiply, operand0=inv_rms[0:P_MAX, 0:1], op1=nl.multiply, operand1=gamma_sb
    )

    tp_psum = nl.ndarray((P_MAX, P_MAX), dtype=shapes.dtype, buffer=nl.psum)
    nisa.nc_transpose(tp_psum, normed_sb)
    tiles_sbuf = nl.ndarray((P_MAX, ppt * T), dtype=shapes.dtype, buffer=nl.sbuf)
    # tiles[f, t * ppt + j] as transposed; k-tile j is the stride-ppt column set starting at j.
    evict = nisa.tensor_copy(dst=tiles_sbuf, src=tp_psum)
    return normed_sb, tiles_sbuf, lambda j: tiles_sbuf.ap(pattern=[[ppt * T, P_MAX], [ppt, T]], offset=j), evict


def _q_norm_and_transpose(
    q_latent: nl.NkiTensor, q_norm_w: nl.NkiTensor, eps: float, shapes: QKvUpShape
) -> tuple[nl.NkiTensor, nl.NkiTensor]:
    """Loads and RMS-normalizes the gathered query latent; returns it and its K-tiled transpose."""

    q_latent_sbuf = _q_norm(q_latent, q_norm_w, eps, shapes)
    return q_latent_sbuf, transpose_into_k_tiles(q_latent_sbuf[0:P_MAX, 0 : shapes.q_lora_rank], shapes.num_tokens)


def _gather_absorbed_heads(
    q_absorbed_local_sbuf: nl.NkiTensor,
    q_pe_local_sbuf: nl.NkiTensor,
    replica_group: ReplicaGroup,
    degree: int,
    heads_local: int,
    shapes: QKvUpShape,
    q_absorbed_out: nl.NkiTensor,
    q_pe_out: nl.NkiTensor,
) -> None:
    """
    AllGather the group's absorbed queries for the attention core.

    The projection is sharded on its output, not its contraction: both cut the weight read and the
    matmuls equally, but an output shard keeps every dot product on one rank, so the result is
    bit-identical to the unsharded kernel with no partial sums to add.

    The gather moves the absorbed query, not the raw one. Every rank attends over all heads, so gathering
    the raw query would make each rank absorb every head; gathering after the absorb leaves each rank only
    its own ``heads_local``, for about 2.5x the collective bytes.

    An HBM collective concatenates on dim 0 only, so the gather runs on the partition axis and the core
    reads the gathered blocks in place.
    """

    T, nC, R = shapes.num_tokens, shapes.num_latent_tiles, shapes.qk_rope_dim
    hl = heads_local

    # Both ends private_hbm: at LNC=1 the runtime rejects a collective whose sources are shared_hbm, and
    # a collective needs one buffer class across all of its operands.
    q_absorbed_src = nl.ndarray((P_MAX, nC, T, hl), dtype=shapes.dtype, buffer=nl.private_hbm)
    q_pe_src = nl.ndarray((R, T, hl), dtype=shapes.dtype, buffer=nl.private_hbm)
    nisa.dma_copy(dst=q_absorbed_src, src=q_absorbed_local_sbuf[0:P_MAX, 0:nC, 0:T, 0:hl])
    nisa.dma_copy(dst=q_pe_src, src=q_pe_local_sbuf[0:R, 0:T, 0:hl])

    q_absorbed_all = cc_dst((degree * P_MAX, nC, T, hl), shapes.dtype, nl.private_hbm, "q_absorbed_all")
    q_pe_all = cc_dst((degree * R, T, hl), shapes.dtype, nl.private_hbm, "q_pe_all")
    _pin_all(q_absorbed_src, q_pe_src, q_absorbed_all, q_pe_all)  # while pinning is on; see utils/cc_pin.py
    ncc.all_gather(
        srcs=[q_absorbed_src, q_pe_src], dsts=[q_absorbed_all, q_pe_all], replica_group=replica_group, collective_dim=0
    )

    from .mla_decode_attention import publish_rank_major_heads

    # The core reads the gathered blocks in place (its heads_blocked contract), so the two outputs are not written.
    publish_rank_major_heads(q_absorbed_out, q_absorbed_all, q_pe_all, degree)


def _absorb_token_major(
    q_sbuf: nl.NkiTensor,
    q_absorb_w: nl.NkiTensor,
    shapes: QKvUpShape,
    heads_local: int,
) -> tuple[nl.NkiTensor, nl.NkiTensor]:
    """
    Folds W_UK into this rank's heads for every token of the block, token-major: returns
    ``q_absorbed [T_all, heads_local, C]`` and ``q_pe [T_all, heads_local, R]`` (rotated by the caller).

    The matmul is :func:`_absorb_head_chunk`'s transposed: the nope query is the stationary and the
    weight streams, so the product lands ``[token, latent]``. The exchange's blocks are token slices,
    and the receiver re-tiles to feature-major (:func:`mla_decode_q_block_finish`).
    """

    qk_nope_dim, qk_head_dim, R = shapes.qk_nope_dim, shapes.qk_head_dim, shapes.qk_rope_dim
    C, T_all, nN = shapes.kv_lora_dim, shapes.num_tokens, shapes.num_nope_tiles
    hl = heads_local
    Tp = div_ceil(T_all, _PSUM_TOKEN_ALIGN) * _PSUM_TOKEN_ALIGN

    # The whole weight, one DMA per nope tile (see _absorb_head_chunk).
    w_sbuf = nl.ndarray((P_MAX, nN, hl, C), dtype=shapes.dtype, buffer=nl.sbuf)
    for nope_tile in TiledRange(qk_nope_dim, P_MAX):
        nisa.dma_copy(
            dst=w_sbuf[0 : nope_tile.size, nope_tile.index, 0:hl, 0:C],
            src=q_absorb_w.ap(
                pattern=[[C, nope_tile.size], [qk_nope_dim * C, hl], [1, C]], offset=nope_tile.start_offset * C
            ),
            dge_mode=nisa.dge_mode.none,
        )

    # q_nope transposed per (head, nope tile): the contraction on partitions, tokens on the free axis.
    q_nope_psum = nl.ndarray((P_MAX, nN, hl, Tp), dtype=shapes.dtype, buffer=nl.psum)
    for hi in range(hl):
        q_nope_start = hi * qk_head_dim
        for nope_tile in TiledRange(qk_nope_dim, P_MAX):
            nisa.nc_transpose(
                q_nope_psum[0 : nope_tile.size, nope_tile.index, hi, 0:T_all],
                q_sbuf[0:T_all, q_nope_start + nope_tile.start_offset : q_nope_start + nope_tile.end_offset],
            )
    q_nope_t_sbuf = nl.ndarray((P_MAX, nN, hl, T_all), dtype=shapes.dtype, buffer=nl.sbuf)
    for nope_tile in TiledRange(qk_nope_dim, P_MAX):
        nisa.tensor_copy(
            dst=q_nope_t_sbuf[0 : nope_tile.size, nope_tile.index, 0:hl, 0:T_all],
            src=q_nope_psum[0 : nope_tile.size, nope_tile.index, 0:hl, 0:T_all],
        )

    # absorbed[t, c] = sum_d q_nope[t, d] * W_UK[d, c], one PSUM bank per head, accumulated over the nope
    # tiles; heads in chunks of half the banks, each chunk evicted once.
    q_abs_tm_sbuf = nl.ndarray((P_MAX, hl, C), dtype=shapes.dtype, buffer=nl.sbuf)
    heads_per_chunk = max(1, min(hl, (_PSUM_BANK_BYTES * PSUM_BANKS // 2) // (C * 4)))
    for head_chunk in TiledRange(hl, heads_per_chunk):
        h0, hc = head_chunk.start_offset, head_chunk.size
        acc_psum = nl.ndarray((P_MAX, hc, C), dtype=nl.float32, buffer=nl.psum)
        for hi in range(hc):
            is_first_nope = True
            for nope_tile in TiledRange(qk_nope_dim, P_MAX):
                nisa.nc_matmul(
                    dst=acc_psum[0:T_all, hi, 0:C],
                    stationary=q_nope_t_sbuf[0 : nope_tile.size, nope_tile.index, h0 + hi, 0:T_all],
                    moving=w_sbuf[0 : nope_tile.size, nope_tile.index, h0 + hi, 0:C],
                    accumulate=not is_first_nope,
                )
                is_first_nope = False
        nisa.tensor_copy(dst=q_abs_tm_sbuf[0:T_all, h0 : h0 + hc, 0:C], src=acc_psum[0:T_all, 0:hc, 0:C])

    # q_pe: head hi's rope field sits at hi * qk_head_dim + qk_nope_dim of the packed query.
    q_pe_tm_sbuf = nl.ndarray((P_MAX, hl, R), dtype=shapes.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(
        dst=q_pe_tm_sbuf[0:T_all, 0:hl, 0:R],
        src=q_sbuf.ap(pattern=[[q_sbuf.shape[1], T_all], [qk_head_dim, hl], [1, R]], offset=qk_nope_dim),
    )
    return q_abs_tm_sbuf, q_pe_tm_sbuf


def _stage_absorbed_blocks(
    q_abs_tm_sbuf: nl.NkiTensor,
    q_pe_tm_sbuf: nl.NkiTensor,
    degree: int,
    slices: int,
    heads_local: int,
    shapes: QKvUpShape,
) -> tuple[nl.NkiTensor, nl.NkiTensor]:
    """
    Stage this rank's absorbed heads as the ``all_to_all`` source that hands every rank of the block its
    own ``T`` tokens for all heads (:func:`mla_decode_q_block_exchange`).

    Position ``d = cp * slice + c`` (``cp = degree / slices`` ranks per slice) holds heads
    ``[d * heads_local, (d + 1) * heads_local)`` for every token of the block, and destination ``d`` wants
    slice ``d // cp``'s tokens, so source block ``d`` is that slice's rows. The ``cp`` ranks of a slice
    receive identical blocks, since they attend the same tokens over different keys. Received block ``s``
    carries source ``s``'s heads, so the result read as ``[(s, hh), t, ...]`` is every head in order with
    no per-rank index, which keeps the kernel rank-invariant.

    Blocks are laid ``[head, token, feature]`` so :func:`mla_decode_q_block_finish` can pull one token's
    ``[num_heads, feature]`` rows with a single stride.

    Returns:
        tuple[nl.NkiTensor, nl.NkiTensor]: the ``[degree, heads_local, T, C]`` and
        ``[degree, heads_local, T, R]`` sources, private HBM.
    """

    T, C, R = shapes.num_tokens, shapes.kv_lora_dim, shapes.qk_rope_dim
    hl = heads_local
    cp = degree // slices
    dtype = shapes.dtype

    # private_hbm: at LNC=1 the runtime rejects a collective whose sources are shared_hbm.
    abs_src = nl.ndarray((degree, hl, T, C), dtype=dtype, buffer=nl.private_hbm)
    pe_src = nl.ndarray((degree, hl, T, R), dtype=dtype, buffer=nl.private_hbm)
    for slice_idx in range(slices):
        t0 = slice_idx * T
        for c in range(cp):
            d = slice_idx * cp + c
            nisa.dma_copy(
                dst=abs_src[d].ap(pattern=[[C, T], [T * C, hl], [1, C]], offset=0),
                src=q_abs_tm_sbuf[t0 : t0 + T, 0:hl, 0:C],
            )
            nisa.dma_copy(
                dst=pe_src[d].ap(pattern=[[R, T], [T * R, hl], [1, R]], offset=0),
                src=q_pe_tm_sbuf[t0 : t0 + T, 0:hl, 0:R],
            )
    return abs_src, pe_src


def _stage_absorbed_blocks_split(
    q_abs_tm_sbuf: nl.NkiTensor,
    q_pe_tm_sbuf: nl.NkiTensor,
    degree: int,
    slices: int,
    heads_local: int,
    shapes: QKvUpShape,
) -> tuple[nl.NkiTensor, nl.NkiTensor]:
    """
    :func:`_stage_absorbed_blocks` for the split exchange.

    The ``cp = degree / slices`` ranks of a slice attend the same ``T`` tokens and sit on one chip, so the
    plain exchange sends them each block ``cp`` times. Here destination ``d = cp * slice + c`` gets only the
    ``Tq = T / cp`` tokens ``[c * Tq, (c + 1) * Tq)`` of its slice (block tokens ``[d * Tq, (d + 1) * Tq)``):
    the ``all_to_all`` moves a ``cp``-th of the bytes, and an AllGather over the slice's ranks completes it
    on chip. Same bytes, different route.

    Blocks are laid ``[head, token, feature]``. With one head per rank the source is the token-major tile
    itself, staged with one DMA per tensor.

    Returns:
        tuple[nl.NkiTensor, nl.NkiTensor]: the ``[degree, heads_local, Tq, C]`` and
        ``[degree, heads_local, Tq, R]`` sources, private HBM.
    """

    T, C, R = shapes.num_tokens, shapes.kv_lora_dim, shapes.qk_rope_dim
    hl = heads_local
    cp = degree // slices
    kernel_assert(
        T % cp == 0 and degree * (T // cp) == slices * T,
        f"the split query exchange needs the {cp} ranks of a slice to split its T={T} tokens evenly",
    )
    Tq = T // cp
    T_all = degree * Tq
    dtype = shapes.dtype

    # private_hbm: at LNC=1 the runtime rejects a collective whose sources are shared_hbm.
    abs_src = nl.ndarray((degree, hl, Tq, C), dtype=dtype, buffer=nl.private_hbm)
    pe_src = nl.ndarray((degree, hl, Tq, R), dtype=dtype, buffer=nl.private_hbm)
    if hl == 1:
        # Element (d, 0, j) is token d * Tq + j: the token-major tile, row for row.
        stage_kw = {
            "sync": dict(dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync),
            "swdge": dict(dge_mode=nisa.dge_mode.swdge),
        }.get(LAT_QSTAGE_DGE, {})
        nisa.dma_copy(dst=abs_src.reshape((T_all, 1, C)), src=q_abs_tm_sbuf[0:T_all, 0:1, 0:C], **stage_kw)
        nisa.dma_copy(dst=pe_src.reshape((T_all, 1, R)), src=q_pe_tm_sbuf[0:T_all, 0:1, 0:R], **stage_kw)
        return abs_src, pe_src
    for d in range(degree):
        t0 = d * Tq
        nisa.dma_copy(
            dst=abs_src[d].ap(pattern=[[C, Tq], [Tq * C, hl], [1, C]], offset=0),
            src=q_abs_tm_sbuf[t0 : t0 + Tq, 0:hl, 0:C],
        )
        nisa.dma_copy(
            dst=pe_src[d].ap(pattern=[[R, Tq], [Tq * R, hl], [1, R]], offset=0),
            src=q_pe_tm_sbuf[t0 : t0 + Tq, 0:hl, 0:R],
        )
    return abs_src, pe_src


@nki.jit
def mla_decode_q_block_exchange(
    q_abs_src: nl.NkiTensor,
    q_pe_src: nl.NkiTensor,
    replica_group: ReplicaGroup,
    token_replica_group: ReplicaGroup | None = None,
    deferred_gather: list = None,
    own_tokens_only: bool = False,
) -> tuple[nl.NkiTensor, nl.NkiTensor]:
    """
    The block-sharded query path's ``all_to_all``: source block ``d`` goes to position ``d``, and result
    block ``s`` comes from position ``s`` (see :func:`_stage_absorbed_blocks`).

    With ``token_replica_group`` (the split exchange): the ``all_to_all`` hands each rank its ``Tq`` tokens
    of every head, then an AllGather over the token group concatenates the ``cp`` pieces into
    ``[cp * degree, heads_local, Tq, X]``, block ``(c, s)`` holding tokens ``[c * Tq, (c + 1) * Tq)`` of
    source ``s``'s heads (read by :func:`mla_decode_q_block_finish` with ``token_split=cp``).

    Usage:
        Tuned for: GLM-5.2 MLA decode, block-sharded query path, LNC=1
        Applicable to: the sources :func:`mla_decode_q_kv_up` stages in its ``q_latent_block`` mode

    This is a separate step so the caller picks when the collective issues. Collectives complete in
    issue order: right after the projection it would delay every later collective, the indexer's
    included, while after the indexer's query gather it overlaps the indexer's scoring.

    Args:
        q_abs_src (nl.NkiTensor): [degree, heads_local, T, C] @ private HBM.
        q_pe_src (nl.NkiTensor): [degree, heads_local, T, R] @ private HBM.
        replica_group (ReplicaGroup): The ``degree`` consecutive ranks of the block, in position order.
        token_replica_group (ReplicaGroup | None): The split exchange's token group (the ``cp`` ranks of this
            rank's slice, in position order), or None for the plain exchange. (default: None)
        deferred_gather (list): Split exchange only. When given, only the ``all_to_all`` is issued here and a
            function that issues the AllGather and returns this function's result is appended to it, so the caller
            can trace work between the two (the AllGather's trigger waits for the ``all_to_all``). (default: None)
        own_tokens_only (bool): Split exchange only. Skip the AllGather and return the ``all_to_all``'s
            ``[degree, heads_local, Tq, X]`` receive: token group position ``c``'s ``Tq`` tokens
            ``[c * Tq, (c + 1) * Tq)`` of every head, all a token-sharded rank attends (read by
            :func:`mla_decode_q_block_finish` with ``token_split=1``). With ``deferred_gather`` the appended
            function returns that receive. (default: False)

    Returns:
        tuple[nl.NkiTensor, nl.NkiTensor]: the received blocks, private HBM: the sources' shapes, or with
            ``token_replica_group`` ``cp`` times their leading extent.
    """

    if token_replica_group is None:
        abs_dst = nl.ndarray(q_abs_src.shape, dtype=q_abs_src.dtype, buffer=nl.private_hbm)
        pe_dst = nl.ndarray(q_pe_src.shape, dtype=q_pe_src.dtype, buffer=nl.private_hbm)
        _pin_all(q_abs_src, q_pe_src, abs_dst, pe_dst)  # while pinning is on; see utils/cc_pin.py
        ncc.all_to_all(
            srcs=[q_abs_src, q_pe_src], dsts=[abs_dst, pe_dst], replica_group=replica_group, collective_dim=0
        )
        return abs_dst, pe_dst

    cp = len(token_replica_group._value[0])
    abs_mid = cc_dst(tuple(q_abs_src.shape), q_abs_src.dtype, nl.private_hbm, "q_abs_split_a2a")
    pe_mid = cc_dst(tuple(q_pe_src.shape), q_pe_src.dtype, nl.private_hbm, "q_pe_split_a2a")
    _pin_all(q_abs_src, q_pe_src, abs_mid, pe_mid)  # while pinning is on; see utils/cc_pin.py
    _LAST_Q_EXCHANGE[0] = ncc.all_to_all(
        srcs=[q_abs_src, q_pe_src], dsts=[abs_mid, pe_mid], replica_group=replica_group, collective_dim=0
    )
    if own_tokens_only:
        if deferred_gather is not None:
            deferred_gather.append(lambda: (abs_mid, pe_mid))
            return None, None
        return abs_mid, pe_mid

    def gather():
        abs_dst = cc_dst(
            (cp * q_abs_src.shape[0],) + tuple(q_abs_src.shape[1:]), q_abs_src.dtype, nl.private_hbm, "q_abs_split_ag"
        )
        pe_dst = cc_dst(
            (cp * q_pe_src.shape[0],) + tuple(q_pe_src.shape[1:]), q_pe_src.dtype, nl.private_hbm, "q_pe_split_ag"
        )
        _pin_all(abs_dst, pe_dst)
        ncc.all_gather(
            srcs=[abs_mid, pe_mid], dsts=[abs_dst, pe_dst], replica_group=token_replica_group, collective_dim=0
        )
        return abs_dst, pe_dst

    if deferred_gather is not None:
        deferred_gather.append(gather)
        return None, None
    return gather()


@nki.jit
def mla_decode_q_block_finish(
    q_abs_recv: nl.NkiTensor,
    q_pe_recv: nl.NkiTensor,
    token_split: int = 1,
    sbuf_out: bool = False,
) -> tuple[nl.NkiTensor, nl.NkiTensor]:
    """
    Re-tile the blocks :func:`mla_decode_q_kv_up` received in its ``q_latent_block`` mode into the
    attention core's feature-major queries.

    Usage:
        Tuned for: GLM-5.2 MLA decode, block-sharded query path, LNC=1
        Applicable to: any ``[degree, heads_local, T, feature]`` head-block exchange whose consumer wants
        ``[feature, T, heads]``

    Kept apart from the projection so the caller can place it after work that does not depend on the
    exchange (the DSA indexer): the engines run in program order, so a receive right after the
    ``all_to_all`` would stall them for the collective's latency.

    Element ``(s, hh, t, x)`` of a received block is head ``s * heads_local + hh`` of token ``t`` at
    offset ``((s * heads_local + hh) * T + t) * X + x``, one stride of ``T * X`` over the merged head
    axis. So one DMA per token pulls its ``[num_heads, X]`` rows, two tokens fill a 128-row tile, and one
    PE transpose per feature tile turns ``[(token, head), feature]`` into the core's
    ``[feature, token, head]``. Needs an even ``T`` and ``2 * num_heads <= 128``.

    With ``token_split = g`` the input is ``[g * degree, heads_local, T / g, X]`` (block ``(c, s)`` holds
    tokens ``[c * T / g, (c + 1) * T / g)`` of source ``s``'s heads): token ``t``'s heads sit at a stride of
    ``(T / g) * X`` from ``((t // (T / g)) * num_heads * (T / g) + t % (T / g)) * X``, still one DMA per token.

    Args:
        q_abs_recv (nl.NkiTensor): [degree, heads_local, T, C] @ HBM, the absorbed query blocks
            ([token_split * degree, heads_local, T / token_split, C] with a split).
        q_pe_recv (nl.NkiTensor): [degree, heads_local, T, R] @ HBM, the rotated rope blocks (likewise).
        token_split (int): The token pieces of the split exchange, 1 for the plain one. (default: 1)
        sbuf_out (bool): Return the two re-tiled SBUF tiles themselves (same shapes) instead of HBM copies.
            (default: False)

    Returns:
        tuple[nl.NkiTensor, nl.NkiTensor]:
            - q_absorbed [P_MAX, C / P_MAX, T, num_heads] @ HBM, feature-major.
            - q_pe [P_MAX, T, num_heads] @ HBM, feature-major, zero above R.
    """

    split_degree, hl, Tq, C = q_abs_recv.shape
    degree = split_degree // token_split
    T = Tq * token_split
    R = q_pe_recv.shape[3]
    Nh = degree * hl
    nC = div_ceil(C, P_MAX)
    rows = 2 * Nh  # (token parity, head) rows of one tile
    dtype = q_abs_recv.dtype
    kernel_assert(
        T % 2 == 0
        and rows <= P_MAX
        and C % P_MAX == 0
        and R <= P_MAX
        and split_degree == degree * token_split
        and tuple(q_pe_recv.shape[:3]) == (split_degree, hl, Tq),
        f"the block re-tile needs an even T={T}, 2 x num_heads={rows} <= {P_MAX}, C={C} a multiple of {P_MAX}, "
        f"R={R} <= {P_MAX} and {token_split} whole token pieces in both inputs",
    )

    q_abs_sb = nl.ndarray((P_MAX, nC, T, Nh), dtype=dtype, buffer=nl.sbuf)
    q_pe_sb = nl.ndarray((P_MAX, T, Nh), dtype=dtype, buffer=nl.sbuf)
    # Partitions [R, P_MAX) of q_pe are the contract's padding: deterministic zeros (full-tile memset).
    nisa.memset(q_pe_sb[0:P_MAX, 0:T, 0:Nh], 0)
    loads = []
    for t0 in range(0, T, 2):
        abs_rows_sb = nl.ndarray((P_MAX, C), dtype=dtype, buffer=nl.sbuf)
        pe_rows_sb = nl.ndarray((P_MAX, R), dtype=dtype, buffer=nl.sbuf)
        for parity in range(2):
            t = t0 + parity
            r0 = parity * Nh
            # Row of token t, head 0, in units of one feature run; heads follow Tq runs apart.
            row = (t // Tq) * Nh * Tq + t % Tq
            loads.append(
                nisa.dma_copy(
                    dst=abs_rows_sb[r0 : r0 + Nh, 0:C],
                    src=q_abs_recv.ap(pattern=[[Tq * C, Nh], [1, C]], offset=row * C),
                )
            )
            loads.append(
                nisa.dma_copy(
                    dst=pe_rows_sb[r0 : r0 + Nh, 0:R],
                    src=q_pe_recv.ap(pattern=[[Tq * R, Nh], [1, R]], offset=row * R),
                )
            )

        abs_psum = nl.ndarray((P_MAX, nC, rows), dtype=dtype, buffer=nl.psum)
        for kv_tile in TiledRange(C, P_MAX):
            nisa.nc_transpose(
                abs_psum[0 : kv_tile.size, kv_tile.index, 0:rows],
                abs_rows_sb[0:rows, kv_tile.start_offset : kv_tile.end_offset],
            )
            nisa.tensor_copy(
                dst=q_abs_sb[0 : kv_tile.size, kv_tile.index, t0 : t0 + 2, 0:Nh],
                src=abs_psum.ap(pattern=[[nC * rows, kv_tile.size], [Nh, 2], [1, Nh]], offset=kv_tile.index * rows),
            )
        pe_psum = nl.ndarray((P_MAX, rows), dtype=dtype, buffer=nl.psum)
        nisa.nc_transpose(pe_psum[0:R, 0:rows], pe_rows_sb[0:rows, 0:R])
        nisa.tensor_copy(
            dst=q_pe_sb[0:R, t0 : t0 + 2, 0:Nh], src=pe_psum.ap(pattern=[[rows, R], [Nh, 2], [1, Nh]], offset=0)
        )

    if sbuf_out:
        return q_abs_sb, q_pe_sb
    q_absorbed_out = nl.ndarray((P_MAX, nC, T, Nh), dtype=dtype, buffer=nl.shared_hbm)
    q_pe_out = nl.ndarray((P_MAX, T, Nh), dtype=dtype, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=q_absorbed_out, src=q_abs_sb[0:P_MAX, 0:nC, 0:T, 0:Nh])
    nisa.dma_copy(dst=q_pe_out, src=q_pe_sb[0:P_MAX, 0:T, 0:Nh])
    return q_absorbed_out, q_pe_out


_LAST_Q_EXCHANGE = [None]
"""The latest split query exchange's all_to_all trigger (:func:`take_q_exchange_trigger`)."""


def take_q_exchange_trigger():
    """The latest split query exchange's all_to_all instruction, then forget it (None when there was none)."""
    t, _LAST_Q_EXCHANGE[0] = _LAST_Q_EXCHANGE[0], None
    return t


def _kv_norm_and_rope(
    kv_row: nl.NkiTensor,
    kv_norm_w: nl.NkiTensor,
    cos_sbuf: nl.NkiTensor,
    sin_sbuf: nl.NkiTensor,
    eps: float,
    shapes: QKvUpShape,
    load_after=None,
) -> nl.NkiTensor:
    """
    Normalizes the latent field of the gathered cache row and rotates its ``k_pe`` field.

    As in the model, ``kv_a_layernorm`` covers only the latent and RoPE only ``k_pe``; the norm gets a
    slice of exactly the latent field. ``load_after``, when given, is an instruction the row load is
    ordered after (the early kv path).
    """

    T, R, kv_row_dim = shapes.num_tokens, shapes.qk_rope_dim, shapes.kv_row_dim

    kv_row_sbuf = nl.ndarray((P_MAX, kv_row_dim), dtype=shapes.dtype, buffer=nl.sbuf)
    kv_load = nisa.dma_copy(dst=kv_row_sbuf[0:T, 0:kv_row_dim], src=kv_row[0:T, 0:kv_row_dim])
    if load_after is not None:
        kv_load.depends_on(load_after)

    kv_gamma_sbuf = load_gamma_broadcast(kv_norm_w, T)
    rmsnorm_inplace(kv_row_sbuf[0:P_MAX, R:kv_row_dim], kv_gamma_sbuf, T, eps)

    # One rope block covering k_pe, which occupies the start of the row.
    rope_inplace(kv_row_sbuf, cos_sbuf, sin_sbuf, 0, R, 1, T)

    return kv_row_sbuf


def _store_kv_row(kv_row_sbuf: nl.NkiTensor, shapes: QKvUpShape) -> nl.NkiTensor:
    """Publishes the normed, rotated cache row as the ``kv_row_new`` output ``[T, R + C]``."""
    T, kv_row_dim = shapes.num_tokens, shapes.kv_row_dim
    kv_row_out = nl.ndarray((T, kv_row_dim), dtype=shapes.dtype, buffer=layer_scratch_hbm())
    nisa.dma_copy(dst=kv_row_out[0:T, 0:kv_row_dim], src=kv_row_sbuf[0:T, 0:kv_row_dim])
    return kv_row_out


def _absorb_chunk(shapes: QKvUpShape) -> tuple[int, int]:
    """
    Heads per absorb chunk and the token stride of its PSUM tiles.

    The chunk's fp32 accumulator ``[heads, latent tiles, token stride]`` must fit one PSUM bank, so the
    chunk shrinks as the token count grows: 8 heads at 8 tokens, one head at 128.
    """

    token_stride = div_ceil(shapes.num_tokens, _PSUM_TOKEN_ALIGN) * _PSUM_TOKEN_ALIGN
    acc_bytes_per_head = shapes.num_latent_tiles * token_stride * 4
    heads = max(1, min(_ABSORB_HEAD_CHUNK, _PSUM_BANK_BYTES // acc_bytes_per_head))
    kernel_assert(
        heads * acc_bytes_per_head <= _PSUM_BANK_BYTES,
        f"one head's absorb accumulator ({acc_bytes_per_head} B) exceeds a PSUM bank ({_PSUM_BANK_BYTES} B)",
    )
    return heads, token_stride


def _absorb_head_chunk(
    q_sbuf: nl.NkiTensor,
    q_absorb_w: nl.NkiTensor,
    shapes: QKvUpShape,
    h0: int,
    hc: int,
    Tp: int,
    q_absorbed_sbuf: nl.NkiTensor,
    q_pe_sbuf: nl.NkiTensor,
) -> None:
    """
    Splits heads ``[h0, h0 + hc)`` of the packed query into nope and rope halves and folds W_UK into the
    nope half, writing both feature-major for the core's score matmuls:
    ``q_absorbed[:, :, :, h] = einsum("td,dc->ct", q_nope[:, h], q_absorb_w[h])``.

    At T = 8 every per-head instruction is fixed overhead, so two things are batched per chunk:

    * The absorb weight loads with one static-descriptor DMA per nope tile (nope rows on partitions,
      ``(head, latent)`` on the free axis) instead of per-head DMAs, whose descriptor generation on
      GpSimd is slow. Each (head, nope) source row is a contiguous 1 KB run.
    * The transposes and absorb matmuls of all ``hc`` heads land in chunk-wide PSUM tiles, evicted once
      per (chunk, tensor). A small PSUM-to-SBUF copy has a fixed cost, and the per-head copies paced the
      absorb. The absorb eviction permutes ``(head, tile, token)`` into the core's
      ``(tile, token, head)`` through an access pattern on the PSUM source. ``Tp`` is the tiles' token
      stride (see ``_PSUM_TOKEN_ALIGN``).
    """

    qk_nope_dim, qk_rope_dim, qk_head_dim = shapes.qk_nope_dim, shapes.qk_rope_dim, shapes.qk_head_dim
    C, T, nC, nN = shapes.kv_lora_dim, shapes.num_tokens, shapes.num_latent_tiles, shapes.num_nope_tiles

    # The whole chunk's weight, one DMA per nope tile: element (d, h, c) of the tile is read from
    # q_absorb_w[h0 + h, tile_start + d, c].
    w_chunk_sbuf = nl.ndarray((P_MAX, nN, hc, C), dtype=shapes.dtype, buffer=nl.sbuf)
    for nope_tile in TiledRange(qk_nope_dim, P_MAX):
        nisa.dma_copy(
            dst=w_chunk_sbuf[0 : nope_tile.size, nope_tile.index, 0:hc, 0:C],
            src=q_absorb_w.ap(
                pattern=[[C, nope_tile.size], [qk_nope_dim * C, hc], [1, C]],
                offset=(h0 * qk_nope_dim + nope_tile.start_offset) * C,
            ),
            dge_mode=nisa.dge_mode.none,
        )

    # Transpose every head's q_nope (per nope tile, so the absorb contracts it on partitions) and q_pe
    # into chunk-wide PSUM tiles; one eviction per nope tile, one for q_pe.
    q_nope_psum = nl.ndarray((P_MAX, nN, hc, Tp), dtype=shapes.dtype, buffer=nl.psum)
    q_pe_psum = nl.ndarray((P_MAX, hc, Tp), dtype=shapes.dtype, buffer=nl.psum)
    for hi in range(hc):
        q_nope_start = (h0 + hi) * qk_head_dim
        q_pe_start = q_nope_start + qk_nope_dim
        for nope_tile in TiledRange(qk_nope_dim, P_MAX):
            nisa.nc_transpose(
                q_nope_psum[0 : nope_tile.size, nope_tile.index, hi, 0:T],
                q_sbuf[0:T, q_nope_start + nope_tile.start_offset : q_nope_start + nope_tile.end_offset],
            )
        nisa.nc_transpose(
            q_pe_psum[0:qk_rope_dim, hi, 0:T],
            q_sbuf[0:T, q_pe_start : q_pe_start + qk_rope_dim],
        )

    q_nope_chunk_sbuf = nl.ndarray((P_MAX, nN, hc, T), dtype=shapes.dtype, buffer=nl.sbuf)
    for nope_tile in TiledRange(qk_nope_dim, P_MAX):
        nisa.tensor_copy(
            dst=q_nope_chunk_sbuf[0 : nope_tile.size, nope_tile.index, 0:hc, 0:T],
            src=q_nope_psum[0 : nope_tile.size, nope_tile.index, 0:hc, 0:T],
        )
    # q_pe is (head, token) in PSUM and (token, head) in the output.
    nisa.tensor_copy(
        dst=q_pe_sbuf[0:qk_rope_dim, 0:T, h0 : h0 + hc],
        src=q_pe_psum.ap(pattern=[[hc * Tp, qk_rope_dim], [1, T], [Tp, hc]], offset=0),
    )

    # The nope width (192) spans two partition tiles, which accumulate into one PSUM region. All heads
    # of the chunk share one [(head, latent tile, token)] PSUM tile, evicted once below.
    acc_psum = nl.ndarray((P_MAX, hc, nC, Tp), dtype=nl.float32, buffer=nl.psum)
    for hi in range(hc):
        for kv_tile in TiledRange(C, P_MAX):
            is_first_nope = True
            for nope_tile in TiledRange(qk_nope_dim, P_MAX):
                nisa.nc_matmul(
                    dst=acc_psum[0 : kv_tile.size, hi, kv_tile.index, 0:T],
                    stationary=w_chunk_sbuf[
                        0 : nope_tile.size, nope_tile.index, hi, kv_tile.start_offset : kv_tile.end_offset
                    ],
                    moving=q_nope_chunk_sbuf[0 : nope_tile.size, nope_tile.index, hi, 0:T],
                    accumulate=not is_first_nope,
                )
                is_first_nope = False

    # One copy for the chunk, from (head, tile, token) in PSUM to (tile, token, head) in the output.
    nisa.tensor_copy(
        dst=q_absorbed_sbuf[0:P_MAX, 0:nC, 0:T, h0 : h0 + hc],
        src=acc_psum.ap(pattern=[[hc * nC * Tp, P_MAX], [Tp, nC], [1, T], [nC * Tp, hc]], offset=0),
    )


def _wqb_banded_fits(shapes: QKvUpShape, shard: int) -> bool:
    """Whether the banded ``wq_b`` layout applies: the spread q-norm, a 32-column band, one ``linear`` whose banks hold
    whole heads, and an absorb chunk of exactly one bank."""
    T, head_dim = shapes.num_tokens, shapes.qk_head_dim
    n_bands = P_MAX // _BAND
    return (
        _q_spread_fits(T, shapes.q_lora_rank)
        and pe_column_band(T) == _BAND
        and F_MAX % head_dim == 0
        and shard <= F_MAX * PSUM_BANKS
        and shard % (F_MAX * n_bands) == 0
        and _absorb_chunk(shapes)[0] == n_bands * F_MAX // head_dim
    )


def _load_rope_tables_banded(cos: nl.NkiTensor, sin: nl.NkiTensor, T: int) -> tuple[nl.NkiTensor, nl.NkiTensor]:
    """``load_rope_tables`` for the banded ``wq_b`` layout: token ``t``'s row at partitions ``32b + t`` of every band
    (the rows between are never read back). On the ACT queue, which is idle at the end of ``wq_b``."""
    half = cos.shape[1]
    out = []
    for table in (cos, sin):
        band_sbuf = nl.ndarray((P_MAX, half), dtype=nl.float32, buffer=nl.sbuf)
        for b in range(P_MAX // _BAND):
            nisa.dma_copy(
                dst=band_sbuf[b * _BAND : b * _BAND + T, 0:half],
                src=table[0:T, 0:half],
                dge_mode=nisa.dge_mode.hwdge,
                engine=nisa.engine.scalar,
            )
        out.append(band_sbuf)
    return out[0], out[1]


def _absorb_banded(
    q_band: nl.NkiTensor,
    q_absorb_w: nl.NkiTensor,
    shapes: QKvUpShape,
    heads_local: int,
    hc: int,
    Tp: int,
    rows: int,
    q_absorbed_sbuf: nl.NkiTensor,
    q_pe_sbuf: nl.NkiTensor,
) -> None:
    """
    :func:`_absorb_head_chunk` for every chunk, reading the banded projection: chunk ``k`` is PSUM bank ``k``,
    its head ``2b + hh`` sits at partitions ``[32b, 32b + T)``, columns ``k * F_MAX + hh * qk_head_dim``. One
    full-height transpose per (hh, nope tile) catches the 4 bands' heads at once (column ``32b + t``) and the
    evictions pick columns ``32b + t`` into the same ``(head, token)`` tiles. Same absorb matmuls in the same order.
    A chunk's q_pe transposes trail its absorb matmuls, since they also wait on the RoPE.
    """

    qk_nope_dim, R, head_dim = shapes.qk_nope_dim, shapes.qk_rope_dim, shapes.qk_head_dim
    C, T, nC, nN = shapes.kv_lora_dim, shapes.num_tokens, shapes.num_latent_tiles, shapes.num_nope_tiles
    n_bands, hpc = P_MAX // _BAND, F_MAX // head_dim
    hl = q_pe_sbuf.shape[2]
    dtype = shapes.dtype

    for k in range(heads_local // hc):
        h0 = k * hc
        w_chunk_sbuf = nl.ndarray((P_MAX, nN, hc, C), dtype=dtype, buffer=nl.sbuf)
        for nope_tile in TiledRange(qk_nope_dim, P_MAX):
            nisa.dma_copy(
                dst=w_chunk_sbuf[0 : nope_tile.size, nope_tile.index, 0:hc, 0:C],
                src=q_absorb_w.ap(
                    pattern=[[C, nope_tile.size], [qk_nope_dim * C, hc], [1, C]],
                    offset=(h0 * qk_nope_dim + nope_tile.start_offset) * C,
                ),
                dge_mode=nisa.dge_mode.none,
            )

        # [feature, nope tile, hh, 32b + t].
        q_nope_psum = nl.ndarray((P_MAX, nN, hpc, P_MAX), dtype=dtype, buffer=nl.psum)
        for hh in range(hpc):
            c0 = k * F_MAX + hh * head_dim
            for nope_tile in TiledRange(qk_nope_dim, P_MAX):
                nisa.nc_transpose(
                    q_nope_psum[0 : nope_tile.size, nope_tile.index, hh, 0:rows],
                    q_band[0:rows, c0 + nope_tile.start_offset : c0 + nope_tile.end_offset],
                )
        q_nope_chunk_sbuf = nl.ndarray((P_MAX, nN, hc, T), dtype=dtype, buffer=nl.sbuf)
        copy_engine = nisa.engine.scalar
        copy_kw = {} if copy_engine is None else dict(engine=copy_engine)
        for nope_tile in TiledRange(qk_nope_dim, P_MAX):
            # Entry (b, hh, t) is column 32b + t of hh's transpose, i.e. head 2b + hh.
            nisa.tensor_copy(
                dst=q_nope_chunk_sbuf.ap(
                    pattern=[[nN * hc * T, nope_tile.size], [hpc * T, n_bands], [T, hpc], [1, T]],
                    offset=nope_tile.index * hc * T,
                ),
                src=q_nope_psum.ap(
                    pattern=[[nN * hpc * P_MAX, nope_tile.size], [_BAND, n_bands], [P_MAX, hpc], [1, T]],
                    offset=nope_tile.index * hpc * P_MAX,
                ),
                **copy_kw,
            )

        acc_psum = nl.ndarray((P_MAX, hc, nC, Tp), dtype=nl.float32, buffer=nl.psum)
        for hi in range(hc):
            for kv_tile in TiledRange(C, P_MAX):
                is_first_nope = True
                for nope_tile in TiledRange(qk_nope_dim, P_MAX):
                    nisa.nc_matmul(
                        dst=acc_psum[0 : kv_tile.size, hi, kv_tile.index, 0:T],
                        stationary=w_chunk_sbuf[
                            0 : nope_tile.size, nope_tile.index, hi, kv_tile.start_offset : kv_tile.end_offset
                        ],
                        moving=q_nope_chunk_sbuf[0 : nope_tile.size, nope_tile.index, hi, 0:T],
                        accumulate=not is_first_nope,
                    )
                    is_first_nope = False

        q_pe_psum = nl.ndarray((P_MAX, hpc, P_MAX), dtype=dtype, buffer=nl.psum)
        for hh in range(hpc):
            c0 = k * F_MAX + hh * head_dim + qk_nope_dim
            nisa.nc_transpose(q_pe_psum[0:R, hh, 0:rows], q_band[0:rows, c0 : c0 + R])

        evict_engine = nisa.engine.scalar
        nisa.tensor_copy(
            dst=q_absorbed_sbuf[0:P_MAX, 0:nC, 0:T, h0 : h0 + hc],
            src=acc_psum.ap(pattern=[[hc * nC * Tp, P_MAX], [Tp, nC], [1, T], [nC * Tp, hc]], offset=0),
            **({} if evict_engine is None else dict(engine=evict_engine)),
        )
        # Column (t, b, hh) goes to q_pe[:, t, h0 + 2b + hh].
        nisa.tensor_copy(
            dst=q_pe_sbuf.ap(pattern=[[T * hl, R], [hl, T], [hpc, n_bands], [1, hpc]], offset=h0),
            src=q_pe_psum.ap(pattern=[[hpc * P_MAX, R], [1, T], [_BAND, n_bands], [P_MAX, hpc]], offset=0),
            **copy_kw,
        )
