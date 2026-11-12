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

"""Paged FA2 cache attention: the history read through a block table, the commit written in place.

Exact online-softmax (FlashAttention-2) attention of a query shard over the cached history and the
current chunk, whose K/V are all-gathered over the CP group inside the kernel, with the history in
a paged KV pool (the pass inputs are described in ``sglang_neuron/diffusion/attention/paged_attention.py``):

* **CP gather.** Each rank's token-major shard of the chunk's K/V is staged head-major and
  all-gathered per head, issued inside the history passes so the collective overlaps compute.
  Q arrives token-major and is PE-transposed to d-major in SBUF; the output is PE-transposed back
  and written d-major ``[N, D, Sq]`` (the o-projection layout). The online-softmax state of every
  head stays in SBUF across the history and the gathered chunk (two walk geometries over the same
  buffers: the history of every head, then the chunk of every head).
* **Pool.** One layer's views of the pool. V is ``[num_blocks, N, block, D]``; K is d-major
  ``[num_blocks, N, D, block]`` (the history keys stream as ``[D, count]`` columns, no
  transpose). The kernel folds the head axis into the block axis
  (``g = block * N + head``, as LingBot-World's ``kv_pool.per_head_block_table`` builds it), so every (block, head) is one
  plane.
* **History read (every pass).** ``block_table`` ``[N, W]`` int32 lists, per head, the folded
  blocks of the visible history in the first ``num_history_blocks`` entries (null-padded). The
  history is walked as one contiguous range (``Sh = num_history_blocks * block``), and each Sk
  block's K and V are DMA'd from the pool into SBUF one block-table entry at a time
  (``scalar_offset`` = the entry); a tile that crosses a block boundary takes its rows from both
  blocks, so the walk and the arithmetic are those of a dense history buffer.
* **Mask (every pass).** ``bounds`` ``[1, 3]`` int32 ``(sink_end, window_start, kv_len)``: history
  position ``p`` of the listed blocks is visible iff ``p < sink_end`` or
  ``window_start <= p < kv_len``. The key bias row (0 visible, ``bf16(-30000)`` masked) is
  generated on chip from an iota and added to the scores by a rank-1 bias matmul. With
  ``all_visible`` (every listed position visible, the steady state) neither is built: the history
  is attended unmasked, which is bit-identical (the bias is all zeros there).
* **Commit write (commit pass only).** ``kv_cache_update_idx`` ``[N, S]`` int32: the per-head slot
  ``g * block + offset`` of every chunk token (shard order; every slot inside the pool: the block
  writes have no range guard), as built by LingBot-World's ``kv_pool.fold_slots_per_head``. The gathered chunk arrives in the
  collective's order, so each gathered shard is placed at its source CP rank's offset (from the
  gathered rank ids). The chunk must cover whole blocks: V is written as whole ``[block, D]``
  planes and K is PE-transposed per 128-token tile and written as whole ``[D, block]`` planes. The
  kernel returns the pool tensors, so the compiler aliases them to the inputs (in place). With
  ``write_only`` (a commit whose attention output is unused) only the CP gather and the commit
  write run: no query, history read or attention.

Trainium 2 and Trainium 3 (BF16; the FA2 primitive selects its engine plan from the NeuronCore
version, see ``hardware``), each at LNC 1 and LNC 2. At LNC 2 the two cores of the logical rank split the work (``lnc_split``): whole
heads per core, an odd head's query rows split at a 128-row tile boundary; both cores issue every
CP all-gather (a logical-rank collective), in the same order, from shared-HBM sources they staged
together; the commit write is split over disjoint items, so every pool slot is written once.
At LNC 1 the instruction stream is that of the one-core kernel.
"""

import nki
import nki.collectives as ncc
import nki.isa as nisa
import nki.language as nl
from nki.collectives import ReplicaGroup

from nkilib.core.utils.kernel_assert import kernel_assert
from nkilib.experimental import neurotile as nt
from .config import D_TILE, SQ_TILE, SV_TILE, FlashAttentionConfig
from .flash_attention_fwd import FlashAttentionFwd
from .lnc_split import attention_items, gather_slots, split_range
from .remainder_stream import remainder_block_stream

#: Score bias of a masked history key: bf16(-30000).
MASK_BIAS = -29952.0
#: Columns of the per-rank CP id row gathered next to the chunk on the commit pass.
_RANK_ID_COLS = 128


def _use_geometry(fwd: FlashAttentionFwd, geometry) -> None:
    """Point ``fwd`` at another walk geometry over the same buffers.

    Every buffer :class:`FlashAttentionFwd` allocates is sized from the config knobs (``sk_tile``,
    ``D``, the pipeline offsets and ``sq_tiles_per_blk``), which two geometries of one config share;
    only the Sk walk (``sk_blocks``, ``grid``, ``num_steps``) differs.
    """
    fwd.geometry = geometry
    fwd.grid = geometry.grid


def _load_q_dmajor_block(q, head, bn, d, tile0, blk_rows, q_block, scratch, psum):
    """Rows [tile0*128, +blk_rows) of q[0, :, head, :] -> q_block d-major tiles, via PE transposes."""
    num_tiles = q_block.shape[1]
    for t in range(num_tiles):
        rows = min(SQ_TILE, blk_rows - t * SQ_TILE)
        row0 = (tile0 + t) * SQ_TILE
        nisa.dma_copy(
            dst=scratch[0:rows, 0:d],
            src=q.ap(pattern=[[bn * d, rows], [1, d]], offset=row0 * bn * d + head * d),
        )
        slot = t % psum.shape[1]
        nisa.nc_transpose(psum[0:d, slot, 0:rows], data=scratch[0:rows, 0:d], engine=nisa.engine.tensor)
        nisa.tensor_copy(dst=q_block[0, t].data[:, 0:rows], src=psum[0:d, slot, 0:rows], engine=nisa.engine.vector)


def _store_out_dmajor_block(o_sbuf, out_head, d, tile0, blk_rows, o_t, psum):
    """Token-major output tiles of one Sq block -> d-major ``out_head[:, block]``, via PE transposes."""
    num_tiles = o_sbuf.shape[0]
    for t in range(num_tiles):
        rows = min(SQ_TILE, blk_rows - t * SQ_TILE)
        slot = t % psum.shape[1]
        nisa.nc_transpose(psum[0:d, slot, 0:rows], data=o_sbuf[t, 0].data[0:rows, 0:d], engine=nisa.engine.tensor)
        nisa.tensor_copy(
            dst=o_t[0:d, t * SQ_TILE : t * SQ_TILE + rows], src=psum[0:d, slot, 0:rows], engine=nisa.engine.vector
        )
    col0 = tile0 * SQ_TILE
    nisa.dma_copy(dst=out_head[:, col0 : col0 + blk_rows], src=o_t[0:d, 0:blk_rows])


class _View(nl.NKIObject):
    """A tile handle with the ``.data`` the FlashAttention stages read."""

    def __init__(self, data):
        self.data = data


class _KBlock(nl.NKIObject):
    """One loaded Sk block of K: a ``(1, tiles)`` grid of ``[D, width]`` views of one SBUF buffer."""

    def __init__(self, buf, widths, sk_tile):
        self.shape = (1, len(widths))
        self._tiles = []
        for t in range(len(widths)):
            self._tiles.append(_View(buf[:, t * sk_tile : t * sk_tile + widths[t]]))

    def __getitem__(self, key):
        return self._tiles[key[1]]


class _BiasBlock(nl.NKIObject):
    """One Sk block of the on-chip key-bias row: a ``(1, tiles)`` grid of ``[1, sk_tile]`` views."""

    def __init__(self, row, k0, widths, sk_tile):
        self.shape = (1, len(widths))
        self._tiles = []
        for t in range(len(widths)):
            self._tiles.append(_View(row[0:1, k0 + t * sk_tile : k0 + t * sk_tile + widths[t]]))

    def __getitem__(self, key):
        return self._tiles[key[1]]


class _BiasStream(nl.NKIObject):
    """``load(b)`` over the bias row (already in SBUF: no DMA)."""

    def __init__(self, row, starts, widths, sk_tile):
        self._row = row
        self._starts = starts
        self._widths = widths
        self._sk_tile = sk_tile

    def load(self, index, dge_mode=None, engine=None):
        return _BiasBlock(self._row, self._starts[index], self._widths[index], self._sk_tile)


def _segments(k0, n, block):
    """Static pieces of the history range ``[k0, k0 + n)``, one per block-table entry it touches.

    Returns ``(entry, offset_in_block, count, offset_in_range)`` tuples.
    """
    out = []
    p = k0
    while p < k0 + n:
        entry = p // block
        a = p - entry * block
        count = min(block - a, k0 + n - p)
        out.append((entry, a, count, p - k0))
        p = p + count
    return out


class _PagedKStream(nl.NKIObject):
    """The history keys of one head, Sk block by Sk block, read from the pool by block table."""

    def __init__(
        self,
        k_planes,
        bt_sb,
        bt_col0,
        starts,
        extents,
        widths,
        sk_tile,
        sk_blk,
        block,
        d,
        count,
        dtype,
    ):
        self._k = k_planes
        self._bt = bt_sb
        self._col0 = bt_col0
        self._starts = starts
        self._extents = extents
        self._widths = widths
        self._sk_tile = sk_tile
        self._block = block
        self._d = d
        self._bufs = []
        for i in range(count):
            self._bufs.append(nl.ndarray((d, sk_blk), dtype=dtype, buffer=nl.sbuf))

    def load(self, index, dge_mode=None, engine=None):
        buf = self._bufs[index % len(self._bufs)]
        pieces = _segments(self._starts[index], self._extents[index], self._block)
        for i in range(len(pieces)):
            entry, a, count, dst0 = pieces[i]
            col = self._col0 + entry
            reg = nisa.scalar_load(self._bt[0:1, col : col + 1])
            nisa.dma_copy(
                dst=buf[:, dst0 : dst0 + count],
                src=self._k.ap(
                    pattern=[[self._block, self._d], [1, count]],
                    offset=a,
                    scalar_offset=reg,
                    indirect_dim=0,
                ),
            )
        return _KBlock(buf, self._widths[index], self._sk_tile)


class _PagedVStream(nl.NKIObject):
    """The history values of one head, Sk block by Sk block: ``[SV_TILE, tiles, D]`` SBUF buffers."""

    def __init__(self, v_planes, bt_sb, bt_col0, starts, extents, sv_tiles, block, d, count, dtype):
        self._v = v_planes
        self._bt = bt_sb
        self._col0 = bt_col0
        self._starts = starts
        self._extents = extents
        self._block = block
        self._d = d
        self._bufs = []
        for i in range(count):
            self._bufs.append(nl.ndarray((SV_TILE, sv_tiles, d), dtype=dtype, buffer=nl.sbuf))

    def load(self, index, dge_mode=None, engine=None):
        buf = self._bufs[index % len(self._bufs)]
        d = self._d
        pieces = _segments(self._starts[index], self._extents[index], self._block)
        for i in range(len(pieces)):
            entry, a, count, r0 = pieces[i]
            col = self._col0 + entry
            reg = nisa.scalar_load(self._bt[0:1, col : col + 1])
            # Rows [r0, r0 + count) of the block land on (partition r % 128, tile r // 128): a leading
            # partial tile, a run of whole tiles, a trailing partial tile -- one DMA each.
            r = r0
            end = r0 + count
            while r < end:
                tile = r // SV_TILE
                p0 = r - tile * SV_TILE
                if p0 == 0 and end - r >= SV_TILE:
                    tiles = (end - r) // SV_TILE
                    nisa.dma_copy(
                        dst=buf[:, tile : tile + tiles, :],
                        src=self._v.ap(
                            pattern=[[d, SV_TILE], [SV_TILE * d, tiles], [1, d]],
                            offset=(a + r - r0) * d,
                            scalar_offset=reg,
                            indirect_dim=0,
                        ),
                    )
                    r = r + tiles * SV_TILE
                else:
                    rows = min(SV_TILE - p0, end - r)
                    nisa.dma_copy(
                        dst=buf[p0 : p0 + rows, tile, :],
                        src=self._v.ap(
                            pattern=[[d, rows], [1, d]],
                            offset=(a + r - r0) * d,
                            scalar_offset=reg,
                            indirect_dim=0,
                        ),
                    )
                    r = r + rows
        return buf


class _PagedFwd(FlashAttentionFwd):
    """FlashAttentionFwd whose V stream may be a paged one (already tiled ``[SV_TILE, tiles, D]``)."""

    def __init__(self, scale, geometry, split_tp_evict=False, split_period=2):
        FlashAttentionFwd.__init__(
            self, scale, geometry, split_tp_evict=split_tp_evict, split_period=split_period
        )
        self.paged_v = False

    def _pv(self, step, v_stream):
        if not self.paged_v:
            FlashAttentionFwd._pv(self, step, v_stream)
            return
        info = self._step_info(step)
        index = info.index
        sk_tile_idx = index[2]
        if index[1] == 0 and sk_tile_idx == 0:  # once per sk_block
            self._v_buf = v_stream.load(
                index[0], dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync
            )
        if not info.exists:
            return
        block_out = self.get_buffer((index[0], index[1], None), "block_out")
        probs_t = self.get_buffer(index, "probs_t")
        sv_per_sk = self.geometry.sv_tiles_per_sk_tile
        for sv_tile_idx in range(info.sv_tiles_valid):
            rows = SV_TILE
            if sv_tile_idx == info.sv_tiles_valid - 1:
                rows = info.sv_last_extent
            nisa.nc_matmul(
                block_out,
                stationary=probs_t[0:rows, sv_tile_idx, :],
                moving=self._v_buf[0:rows, sk_tile_idx * sv_per_sk + sv_tile_idx, :],
                accumulate=not (info.is_first_tile and sv_tile_idx == 0),
            )


def _history_bias_row(bounds, sh, dtype):
    """``[1, >= sh]`` key-bias row from ``(sink_end, window_start, kv_len)``, generated on chip."""
    f = (sh + SQ_TILE - 1) // SQ_TILE
    pos = nl.ndarray((SQ_TILE, f), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(dst=pos, pattern=[[1, f]], offset=0, channel_multiplier=f)  # p = partition * f + col
    bnd_i = nl.ndarray((SQ_TILE, 3), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=bnd_i, src=bounds.ap(pattern=[[0, SQ_TILE], [1, 3]], offset=0))
    bnd = nl.ndarray((SQ_TILE, 3), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=bnd, src=bnd_i)
    sink = nl.ndarray((SQ_TILE, f), dtype=nl.float32, buffer=nl.sbuf)
    lo = nl.ndarray((SQ_TILE, f), dtype=nl.float32, buffer=nl.sbuf)
    hi = nl.ndarray((SQ_TILE, f), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=sink, data=pos, op0=nl.less, operand0=bnd[:, 0:1])
    nisa.tensor_scalar(dst=lo, data=pos, op0=nl.greater_equal, operand0=bnd[:, 1:2])
    nisa.tensor_scalar(dst=hi, data=pos, op0=nl.less, operand0=bnd[:, 2:3])
    nisa.tensor_tensor(dst=lo, data1=lo, data2=hi, op=nl.multiply)
    nisa.tensor_tensor(dst=sink, data1=sink, data2=lo, op=nl.maximum)  # visible: 1, masked: 0
    bias = nl.ndarray((SQ_TILE, f), dtype=dtype, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=bias, data=sink, op0=nl.subtract, operand0=1.0, op1=nl.multiply, operand1=-MASK_BIAS
    )
    # [128, f] (position p = partition * f + col) -> one partition, position-ordered, through a
    # small HBM scratch (row-major there).
    scratch = nl.ndarray((SQ_TILE, f), dtype=dtype, buffer=nl.private_hbm, name="fa2p_bias")
    nisa.dma_copy(dst=scratch, src=bias)
    row = nl.ndarray((1, SQ_TILE * f), dtype=dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=row, src=scratch.ap(pattern=[[SQ_TILE * f, 1], [1, SQ_TILE * f]], offset=0))
    return row


def _commit_blocks(
    k_new,
    v_new,
    k_pool,
    v_pool,
    kv_cache_update_idx,
    src_offsets,
    psum_slots,
    cp,
    s_loc,
    block,
    num_shards=1,
    shard_id=0,
):
    """The commit write block by block: whole-plane DMAs, no per-token scatter.

    ``k_new`` / ``v_new`` ``[N, S, D]``: the gathered chunk in gather order; ``src_offsets``
    ``[1, cp]`` int32 the update-index offset of every gathered shard (``None``: already in
    order). The slot ``g * block`` of block ``f``'s first token is the scalar offset of a
    ``[*, D]`` row view, so V's rows ``[slot, slot + block)`` and d-major K's plane (element
    ``slot * D``) are addressed without a division.

    At LNC 2 the reorder copies (``(shard, head)`` items) and the block writes (``(head, block)``
    items) are each split contiguously over the two cores, with a barrier between them (the
    reordered chunk is then in shared HBM): every pool block is written by exactly one core.
    """
    bn, s_cur, d = k_new.shape
    frames = s_cur // block
    nblocks = k_pool.shape[0]
    rows_total = nblocks * bn * block
    v_rows = v_pool.reshape((rows_total, d))
    k_rows = k_pool.reshape((rows_total, d))  # d-major: row ``slot`` starts plane ``slot // block``

    # 1. The gathered chunk in update-index order (one HBM copy per CP shard).
    k_ord = k_new
    v_ord = v_new
    if src_offsets is not None:
        # LNC 2: both cores read every reordered head (shared HBM, after a barrier).
        ord_buffer = nl.shared_hbm if num_shards > 1 else nl.private_hbm
        k_ord = nl.ndarray((bn, s_cur, d), dtype=k_new.dtype, buffer=ord_buffer, name="fa2p_k_ord")
        v_ord = nl.ndarray((bn, s_cur, d), dtype=v_new.dtype, buffer=ord_buffer, name="fa2p_v_ord")
        k_ord_rows = k_ord.reshape((bn * s_cur, d))
        v_ord_rows = v_ord.reshape((bn * s_cur, d))
        copy_lo, copy_hi = split_range(cp * bn, num_shards, shard_id)  # items i * bn + head
        for i in range(cp):
            if i * bn + bn <= copy_lo or i * bn >= copy_hi:
                continue
            reg = nisa.scalar_load(src_offsets[0:1, i : i + 1])
            for head in range(bn):
                if i * bn + head < copy_lo or i * bn + head >= copy_hi:
                    continue
                nisa.dma_copy(
                    dst=k_ord_rows.ap(
                        pattern=[[d, s_loc], [1, d]],
                        offset=head * s_cur * d,
                        scalar_offset=reg,
                        indirect_dim=0,
                    ),
                    src=k_new[head, i * s_loc : (i + 1) * s_loc, :],
                )
                nisa.dma_copy(
                    dst=v_ord_rows.ap(
                        pattern=[[d, s_loc], [1, d]],
                        offset=head * s_cur * d,
                        scalar_offset=reg,
                        indirect_dim=0,
                    ),
                    src=v_new[head, i * s_loc : (i + 1) * s_loc, :],
                )
        if num_shards > 1:
            nisa.core_barrier(k_ord, (0, 1))
            nisa.core_barrier(v_ord, (0, 1))

    # 2. Every block's first slot, all heads on one partition.
    first = nl.ndarray((1, bn * frames), dtype=nl.int32, buffer=nl.sbuf)
    for head in range(bn):
        nisa.dma_copy(
            dst=first[0:1, head * frames : (head + 1) * frames],
            src=kv_cache_update_idx.ap(pattern=[[0, 1], [block, frames]], offset=head * s_cur),
        )

    # 3. Whole-block writes.
    full = block // SV_TILE
    rem = block - full * SV_TILE
    tiles = full + (1 if rem else 0)
    stages = []
    k_ts = []
    for i in range(2):
        stages.append(nl.ndarray((SV_TILE, tiles, d), dtype=k_new.dtype, buffer=nl.sbuf))
        k_ts.append(nl.ndarray((d, block), dtype=k_new.dtype, buffer=nl.sbuf))
    n = 0
    write_lo, write_hi = split_range(bn * frames, num_shards, shard_id)  # items head * frames + f
    for head in range(bn):
        for f in range(frames):
            if head * frames + f < write_lo or head * frames + f >= write_hi:
                continue
            reg = nisa.scalar_load(first[0:1, head * frames + f : head * frames + f + 1])
            r0 = f * block
            nisa.dma_copy(
                dst=v_rows.ap(
                    pattern=[[d, block], [1, d]], offset=0, scalar_offset=reg, indirect_dim=0
                ),
                src=v_ord[head, r0 : r0 + block, :],
            )
            j = (head * frames + f) % 2
            stage = stages[j]
            k_t = k_ts[j]
            if full:
                nisa.dma_copy(
                    dst=stage[:, 0:full, :],
                    src=k_ord.ap(
                        pattern=[[d, SV_TILE], [SV_TILE * d, full], [1, d]],
                        offset=(head * s_cur + r0) * d,
                    ),
                )
            if rem:
                nisa.dma_copy(
                    dst=stage[0:rem, full, :],
                    src=k_ord[head, r0 + full * SV_TILE : r0 + block, :],
                )
            for t in range(tiles):
                rows = SV_TILE if t < full else rem
                ps = psum_slots[n % len(psum_slots)]
                slot = (n // len(psum_slots)) % ps.shape[1]
                n = n + 1
                nisa.nc_transpose(
                    ps[0:d, slot, 0:rows], data=stage[0:rows, t, :], engine=nisa.engine.tensor
                )
                nisa.tensor_copy(
                    dst=k_t[:, t * SV_TILE : t * SV_TILE + rows],
                    src=ps[0:d, slot, 0:rows],
                    engine=nisa.engine.scalar,
                )
            nisa.dma_copy(
                dst=k_rows.ap(
                    pattern=[[block, d], [1, block]], offset=0, scalar_offset=reg, indirect_dim=0
                ),
                src=k_t,
            )


@nki.jit
def fa2_paged_cache_attention_cp_gather(
    q,
    k_shard,
    v_shard,
    k_pool,
    v_pool,
    block_table,
    bounds,
    rank_id,
    kv_cache_update_idx=None,
    replica_groups=(),
    cp_size=1,
    scale=1.0,
    num_history_blocks=1,
    gather=True,
    sq_tiles_per_blk=5,
    sk_tiles_per_blk=6,
    gather_dst_private=False,
    all_visible=False,
    write_only=False,
):
    """Exact attention of ``q`` over ``paged history ++ all_gather_CP(k_shard, v_shard)``.

    Runs on Trainium 2 and Trainium 3 at LNC 1 and LNC 2; at LNC 2 the two cores split the
    heads (an odd head by query rows), stage the shard together, both issue every all-gather and
    split the commit (``lnc_split``). torch-neuronx's NKI cache hashes only this entry point's
    source, so edit this line when a nested helper (``flash_attention_fwd``, ``hardware``,
    ``lnc_split``) changes.

    Args:
        q: [1, Sq, N, D] bf16 token-major queries of this rank (not pre-scaled).
        k_shard, v_shard: [1, S_local, N, D] bf16 token-major, this rank's shard of the chunk's K/V
            (``gather=False``: the whole chunk, no collective).
        k_pool: [num_blocks, N, D, block] bf16 (d-major), this layer's K pool.
        v_pool: [num_blocks, N, block, D] bf16, this layer's V pool.
        block_table: [N, W] int32 per-head folded block ids (``block * N + head``), ``W >=
            num_history_blocks``. Only entries ``[0, num_history_blocks)`` are read: the listed
            history, null-padded. Entries past them are ignored: the read plan's table also lists
            the current chunk's blocks (``[5, 18]`` = 15 history + 3 current per rank at 480p
            TP8), which the commit does not read (it writes through ``kv_cache_update_idx``) and
            the tests use to read the written chunk back.
        bounds: [1, 3] int32 ``(sink_end, window_start, kv_len)`` in listed-history positions.
        rank_id: fp32 [1, 128], this rank's index in its CP group (every column the same). Passed
            on every pass but read only on a commit pass with ``gather``: it is all-gathered next to
            the chunk so each gathered block's update indices are read at its source rank's offset.
        kv_cache_update_idx: [N, S] int32 per-head pool slots of the chunk tokens (shard order),
            or ``None`` for a probe (no write). The chunk must cover whole blocks: tokens
            ``[f * block, (f + 1) * block)`` in this order go to consecutive slots of one block
            (the host checks it), so only every block's first slot is read. The gathered chunk is
            put in that order (one copy per CP shard, at its source rank's offset), V is written
            as whole ``[block, D]`` planes and K is PE-transposed per 128-token tile and written as
            whole ``[D, block]`` planes.
        replica_groups / cp_size: every CP group of the world as global ranks / its size.
        scale: softmax scale.
        num_history_blocks: listed history blocks (static), ``Sh = num_history_blocks * block``.
        gather: all-gather the chunk (default).
        sq_tiles_per_blk / sk_tiles_per_blk: FlashAttention-2 tiling knobs.
        gather_dst_private: allocate the all-gather destinations in private HBM (nki releases that
            require a collective's sources and destinations to share a buffer type).
        all_visible: ``bounds`` masks no listed history position (static; the caller verifies
            it): the history is attended without the mask.
        write_only: a commit pass whose attention output is unused (static): only the CP gather
            and the commit write run.

    Returns:
        out [N, D, Sq] bf16 (d-major); on a commit pass ``(out, k_pool, v_pool)`` (written in place);
        with ``write_only`` ``(k_pool, v_pool)``.
    """
    _, sq, bn, d = q.shape
    _, s_loc, kn, kd = k_shard.shape
    nblocks, pn, pd, block = k_pool.shape
    cp = cp_size if gather else 1
    s_cur = s_loc * cp
    commit = kv_cache_update_idx != None
    sh = num_history_blocks * block
    width_bt = block_table.shape[1]
    kernel_assert(d == D_TILE, f"head_dim must be {D_TILE}, got {d}")
    kernel_assert(
        kn == bn and kd == d and v_shard.shape == k_shard.shape, "k/v shard shape mismatch"
    )
    kernel_assert(
        pn == bn and pd == d, "k_pool must be [num_blocks, N, D, block]"
    )
    kernel_assert(
        v_pool.shape == (nblocks, bn, block, d), "v_pool must be [num_blocks, N, block, D]"
    )
    kernel_assert(
        block_table.shape[0] == bn and num_history_blocks <= width_bt, "block_table must be [N, W]"
    )
    kernel_assert(bounds.shape == (1, 3), "bounds must be [1, 3]")
    kernel_assert(not write_only or commit, "write_only needs a commit (kv_cache_update_idx)")
    kernel_assert(nisa.get_nc_version() >= nisa.nc_version.gen3, "needs Trainium 2 or later")
    program_ndims = nl.program_ndim()
    num_shards = nl.num_programs(0) if program_ndims == 1 else 1
    shard_id = nl.program_id(0) if program_ndims == 1 else 0
    kernel_assert(
        num_shards == 1 or num_shards == 2,
        f"fa2_paged_cache_attention_cp_gather supports LNC 1 and 2, got {num_shards} programs",
    )
    lnc2 = num_shards > 1
    kernel_assert(
        not (lnc2 and gather_dst_private),
        "at LNC 2 the gather destinations are shared HBM (both cores read them)",
    )
    if commit:
        kernel_assert(
            s_cur % block == 0,
            f"the commit needs a chunk of whole blocks ({s_cur} tokens, block {block})",
        )
        kernel_assert(kv_cache_update_idx.shape == (bn, s_cur), "kv_cache_update_idx must be [N, S]")

    out = None
    if not write_only:
        out = nl.ndarray((bn, d, sq), dtype=q.dtype, buffer=nl.shared_hbm)
    k_planes = k_pool.reshape((nblocks * bn, d, block))
    v_planes = v_pool.reshape((nblocks * bn, block, d))

    # ---- block table (all heads, one partition) and the history key bias ----
    bt_sb = nl.ndarray((1, bn * width_bt), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=bt_sb, src=block_table.ap(pattern=[[0, 1], [1, bn * width_bt]], offset=0))
    bias_row = None
    if not (all_visible or write_only):
        bias_row = _history_bias_row(bounds, sh, q.dtype)

    # ---- 1. Stage this rank's shard head-major and set up the CP gather ----
    # LNC 2: the two cores stage half the heads each into shared HBM, the collectives' sources
    # (a collective is the logical rank's: both cores issue it over the same buffers).
    width = bn * d
    src_buffer = nl.shared_hbm if lnc2 else nl.private_hbm
    k_src = nl.ndarray((bn, s_loc, d), dtype=k_shard.dtype, buffer=src_buffer, name="fa2p_k_src")
    v_src = nl.ndarray((bn, s_loc, d), dtype=v_shard.dtype, buffer=src_buffer, name="fa2p_v_src")
    stage_lo, stage_hi = split_range(bn, num_shards, shard_id)
    for head in range(stage_lo, stage_hi):
        nisa.dma_copy(
            dst=k_src[head], src=k_shard.ap(pattern=[[width, s_loc], [1, d]], offset=head * d)
        )
        nisa.dma_copy(
            dst=v_src[head], src=v_shard.ap(pattern=[[width, s_loc], [1, d]], offset=head * d)
        )
    dst_buffer = nl.private_hbm if gather_dst_private else nl.shared_hbm
    if gather:
        k_new = nl.ndarray(
            (bn, s_cur, d), dtype=k_shard.dtype, buffer=dst_buffer, name="fa2p_k_new"
        )
        v_new = nl.ndarray(
            (bn, s_cur, d), dtype=v_shard.dtype, buffer=dst_buffer, name="fa2p_v_new"
        )
    else:
        k_new = k_src
        v_new = v_src
    replica_group = ReplicaGroup([list(g) for g in replica_groups])
    gather_ids = commit and gather
    id_src = None
    src_all = None
    if gather_ids:
        id_src = nl.ndarray(
            (1, _RANK_ID_COLS), dtype=nl.float32, buffer=src_buffer, name="fa2p_id_src"
        )
        if shard_id == 0:
            nisa.dma_copy(dst=id_src, src=rank_id)
        src_all = nl.ndarray(
            (cp_size, _RANK_ID_COLS), dtype=nl.float32, buffer=dst_buffer, name="fa2p_id_all"
        )
    if lnc2:
        # Every gather (and, with gather=False, the chunk phase) reads both cores' staging.
        nisa.core_barrier(k_src, (0, 1))
        nisa.core_barrier(v_src, (0, 1))
        if gather_ids:
            nisa.core_barrier(id_src, (0, 1))

    # This core's attention items and, the same on every core, the gathers each history pipeline
    # slot issues (LNC 1: one item and one gather per head, in head order).
    items = attention_items(bn, sq, num_shards, shard_id, SQ_TILE)
    slots = gather_slots(bn, sq, num_shards, SQ_TILE)

    def _make_gather(slot):
        def _run() -> None:
            heads = slots[slot]
            for i in range(len(heads)):
                head = heads[i]
                ncc.all_gather(
                    dsts=[k_new[head]],
                    srcs=[k_src[head]],
                    replica_group=replica_group,
                    collective_dim=0,
                )
                ncc.all_gather(
                    dsts=[v_new[head]],
                    srcs=[v_src[head]],
                    replica_group=replica_group,
                    collective_dim=0,
                )
            if slot == 0 and gather_ids:
                ncc.all_gather(
                    dsts=[src_all], srcs=[id_src], replica_group=replica_group, collective_dim=0
                )

        return _run

    # ---- 2. Geometries: the listed history (one contiguous walk) and the gathered chunk ----
    config = FlashAttentionConfig(
        sq_tiles_per_blk=sq_tiles_per_blk, sk_tiles_per_blk=sk_tiles_per_blk
    )
    geo_hist = config.geometry(D=d, Sq=sq, Sk=sh)
    geo_new = config.geometry(D=d, Sq=sq, Sk=s_cur)
    fwd = _PagedFwd(scale=scale, geometry=geo_hist)
    tp_psum = fwd.buffers["probs_psum"][0]
    k_psum = fwd.buffers["probs_psum"]  # idle after the pipelines (the commit's K transposes)
    scratch = nl.ndarray((SQ_TILE, d), dtype=q.dtype, buffer=nl.sbuf)
    o_t = nl.ndarray((d, sq), dtype=q.dtype, buffer=nl.sbuf)

    # The history's Sk blocks: start, extent, per-tile widths (the contiguous walk's own blocks).
    sk_tile = geo_hist.sk_tile
    starts = []
    extents = []
    widths = []
    k0 = 0
    for b in range(len(geo_hist.sk_blocks)):
        count, last = geo_hist.sk_blocks[b]
        extent = (count - 1) * sk_tile + last
        starts.append(k0)
        extents.append(extent)
        w = []
        for t in range(count):
            w.append(sk_tile if t < count - 1 else last)
        widths.append(w)
        k0 = k0 + extent
    sv_tiles_blk = geo_hist.sk_blk // SV_TILE

    # Per item: its first Sq tile, its tile count and its Sq blocks (LNC 1: every head the whole
    # query, the one-core walk).
    item_tile0 = []
    item_tiles = []
    item_blocks = []
    num_sq_blks = 0
    for j in range(len(items)):
        head, row0, rows = items[j]
        tiles = (rows + SQ_TILE - 1) // SQ_TILE
        item_tile0.append(row0 // SQ_TILE)
        item_tiles.append(tiles)
        item_blocks.append((tiles + sq_tiles_per_blk - 1) // sq_tiles_per_blk)
        num_sq_blks = max(num_sq_blks, item_blocks[j])
    if write_only:
        num_sq_blks = 0  # no attention: the gathers and the commit only
    if gather and num_sq_blks == 0:
        # No history pipeline issues them (no attention work on this core, or write_only).
        for slot in range(len(slots)):
            _make_gather(slot)()
    for sq_blk_idx in range(num_sq_blks):
        active = []
        for j in range(len(items)):
            if item_blocks[j] > sq_blk_idx:
                active.append(j)
        q_blocks = []
        o_sbufs = []
        states = []
        tile0s = []
        rows_in_blk = []
        for a in range(len(active)):
            j = active[a]
            head = items[j][0]
            tile0 = item_tile0[j] + sq_blk_idx * sq_tiles_per_blk
            blk_tiles = min(sq_tiles_per_blk, item_tile0[j] + item_tiles[j] - tile0)
            row_end = items[j][1] + items[j][2]
            blk_rows = min(sq_tiles_per_blk * SQ_TILE, row_end - tile0 * SQ_TILE)
            q_block = nt.alloc_tiles(
                tile_size=(D_TILE, SQ_TILE), grid=(1, blk_tiles), buffer_type=nl.sbuf, dtype=q.dtype
            )
            _load_q_dmajor_block(q, head, bn, d, tile0, blk_rows, q_block, scratch, tp_psum)
            o_sbufs.append(
                nt.alloc_tiles(
                    tile_size=(SQ_TILE, d), grid=(blk_tiles, 1), buffer_type=nl.sbuf, dtype=q.dtype
                )
            )
            q_blocks.append(q_block)
            states.append(fwd.new_sbuf_state())
            tile0s.append(tile0)
            rows_in_blk.append(blk_rows)

        # Chunk-outer: every item's paged history, then every item's gathered chunk.
        for chunk in range(2):
            geometry = geo_hist if chunk == 0 else geo_new
            _use_geometry(fwd, geometry)
            for a in range(len(active)):
                head = items[active[a]][0]
                bias_stream = None
                if chunk == 0:
                    k_stream = _PagedKStream(
                        k_planes,
                        bt_sb,
                        head * width_bt,
                        starts,
                        extents,
                        widths,
                        sk_tile,
                        geometry.sk_blk,
                        block,
                        d,
                        geometry.kv_buffer_count,
                        k_pool.dtype,
                    )
                    v_stream = _PagedVStream(
                        v_planes,
                        bt_sb,
                        head * width_bt,
                        starts,
                        extents,
                        sv_tiles_blk,
                        block,
                        d,
                        geometry.kv_buffer_count,
                        v_pool.dtype,
                    )
                    if bias_row is not None:
                        bias_stream = _BiasStream(bias_row, starts, widths, sk_tile)
                    fwd.paged_v = True
                else:
                    k_stream = remainder_block_stream(
                        k_new[head],
                        block_size=(geometry.sk_tiles_per_blk, geometry.d_tiles),
                        tile_size=(geometry.sk_tile, D_TILE),
                        dim=0,
                        buffer_count=geometry.kv_buffer_count,
                        align=SV_TILE,
                        transpose=True,
                    )
                    v_stream = remainder_block_stream(
                        v_new[head],
                        block_size=(geometry.sk_blk // SV_TILE, 1),
                        tile_size=(SV_TILE, geometry.D),
                        dim=0,
                        buffer_count=geometry.kv_buffer_count,
                        align=SV_TILE,
                    )
                    fwd.paged_v = False
                callback = None
                if chunk == 0 and gather and sq_blk_idx == 0:
                    # Every item is active in the first Sq block: item a issues slot a.
                    callback = _make_gather(a)
                fwd.use_sbuf_state(states[a])
                fwd.run_pipeline(
                    q_block=q_blocks[a],
                    k_stream=k_stream,
                    v_stream=v_stream,
                    callback=callback,
                    bias_stream=bias_stream,
                )
            if chunk == 0 and gather and sq_blk_idx == 0:
                # A core with fewer items than slots (LNC 2, an unsplittable odd head) issues the
                # remaining gathers before its chunk phase, in slot order like the other core.
                for slot in range(len(active), len(slots)):
                    _make_gather(slot)()

        for a in range(len(active)):
            head = items[active[a]][0]
            fwd.use_sbuf_state(states[a])
            fwd._sq_tiles_valid = q_blocks[a].shape[1]
            fwd.normalize(o_sbufs[a])
            _store_out_dmajor_block(
                o_sbufs[a], out[head], d, tile0s[a], rows_in_blk[a], o_t, tp_psum
            )

    if not commit:
        return out

    # ---- 3. Commit: write the gathered chunk into the pool block by block (in place) ----
    # Both cores read the gathered chunk (shared HBM) and write disjoint pool blocks.
    # Gathered block i holds the shard of CP rank src[i]; its tokens' indices start at src[i] * s_loc.
    off_i = None
    if gather_ids:
        off_i = nl.ndarray((1, cp), dtype=nl.int32, buffer=nl.sbuf)
        ids = nl.ndarray((1, cp), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=ids, src=src_all.ap(pattern=[[0, 1], [_RANK_ID_COLS, cp]], offset=0))
        nisa.tensor_scalar(dst=ids, data=ids, op0=nl.multiply, operand0=float(s_loc))
        nisa.tensor_copy(dst=off_i, src=ids)
    _commit_blocks(
        k_new,
        v_new,
        k_pool,
        v_pool,
        kv_cache_update_idx,
        off_i,
        k_psum,
        cp=cp,
        s_loc=s_loc,
        block=block,
        num_shards=num_shards,
        shard_id=shard_id,
    )
    if write_only:
        return k_pool, v_pool
    return out, k_pool, v_pool
