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

"""Full-vocab token sampler: the GPU top-k and top-p rule over a vocab-sharded logit tensor.

Each rank holds a vocab shard ``logits [R, V_s]`` (global id = ``rank * V_s + local``, ranks in rank
order). One call samples one token per row. A greedy row (temperature below 1e-5) takes the lowest
global id among the row maxima. A sampled row keeps ``l >= k-th largest logit`` (top-k), then the
nucleus ``sum(e[l >= t]) >= top_p * Z`` (top-p), then draws by inverse CDF of the kept masses in
ascending global id order at ``u * total``.

The cuts come from a 16-ary search over the BF16 order key, 4 rounds each. The key of a value is a
monotone integer, so a value threshold compares as an integer key threshold; the kernel converts the
logit bits to keys once and searches in key space. One collective per search round makes every rank
compute the same global cuts; the final draw broadcasts the owning rank's token.
"""

import nki
import nki.collectives as ncc
import nki.isa as nisa
import nki.language as nl
from nki.collectives import ReplicaGroup

from ..utils.allocator import sizeinbytes
from ..utils.kernel_assert import kernel_assert

_SAMPLING_EPS = 1e-5
"""A temperature below this is a greedy row (the GPU sampler rule)."""
_U_MIN = 1.1754944e-38
"""Smallest normal FP32; a draw ``u`` below it is clamped up to it."""
_BELOW_ONE_2 = 1.0 - 2.0**-23
"""``x * (1 - 2^-23)`` is strictly below ``x`` for a positive FP32 ``x``."""
_SEARCH_TRIALS = 15
"""Trial keys per search round: a 16-ary digit resolves 4 bits of the order key."""
_ROUNDS = 4
"""Rounds of the BF16 order-key search (4 rounds * 4 bits = 16 bits)."""
_HALF = 32768.0
"""Order-key origin: ``+0`` and ``-0`` map here."""
_WHOLE = 65536.0
_FINITE_MAX = 3.4e38
"""Above the largest finite BF16 value (3.39e38): ``x < _FINITE_MAX`` is false only for +inf and NaN."""
_ID_BIG = float(1 << 30)
"""An id above every real global id, for the tie-break minimum of a non-matching rank."""


def _pad256_pair(rows, width, dtype):
    """Return the HBM pair of a ``cc_pad256`` all-reduce of ``[rows, width]`` values and their ``[rows, width]`` views.

    Each tensor holds the next multiple of 256 bytes. The views are the first ``rows * width`` values; the pad values
    are not written and not read, so the reduce of the real values does not change.
    """
    per = 256 // sizeinbytes(dtype)
    n = -(-(rows * width) // per) * per
    hsrc = nl.ndarray((n,), dtype=dtype, buffer=nl.private_hbm)
    hdst = nl.ndarray((n,), dtype=dtype, buffer=nl.private_hbm)
    real = rows * width
    return (
        hsrc.reshape((n // per, per)),
        hdst.reshape((n // per, per)),
        hsrc[0:real].reshape((rows, width)),
        hdst[0:real].reshape((rows, width)),
    )


def _allreduce(src, dst, op, tp_group, tp_size, name, pad256=False):
    """All-reduce ``src`` into ``dst`` over the group, or copy when there is no group.

    The reduce runs through a fresh pair of HBM scratch tensors: the Neuron runtime's collective path
    faults on an SBUF all-reduce of these small tiles, and HBM is the collective-safe route. ``pad256``
    (``cc_pad256``) takes the pair of ``_pad256_pair``.
    """
    if tp_size == 1:
        nisa.tensor_copy(dst=dst, src=src)
        return
    rows, width = src.shape
    if pad256:
        hsrc, hdst, src_view, dst_view = _pad256_pair(rows, width, src.dtype)
        nisa.dma_copy(dst=src_view, src=src)
        ncc.all_reduce(srcs=[hsrc], dsts=[hdst], replica_group=tp_group, op=op, name=name)
        nisa.dma_copy(dst=dst, src=dst_view)
        return
    hsrc = nl.ndarray((rows, width), dtype=src.dtype, buffer=nl.private_hbm)
    hdst = nl.ndarray((rows, width), dtype=src.dtype, buffer=nl.private_hbm)
    nisa.dma_copy(dst=hsrc, src=src)
    ncc.all_reduce(srcs=[hsrc], dsts=[hdst], replica_group=tp_group, op=op, name=name)
    nisa.dma_copy(dst=dst, src=hdst)


def _allgather(src, dst, tp_group, tp_size, name):
    """All-gather ``src [R, 1]`` into ``dst [R, tp_size]`` in rank order, or copy when there is no group."""
    if tp_size == 1:
        nisa.tensor_copy(dst=dst, src=src)
    else:
        ncc.all_gather(srcs=[src], dsts=[dst], replica_group=tp_group, collective_dim=1, name=name)


_GATHER_PASS = 512
"""The most data elements per partition that one ``nc_n_gather`` reads in a single pass; a larger gather runs
multi-pass, and multi-pass gathers race."""


def _gather(dst, data, index_f):
    """Gather ``data [R, W]`` at the FP32 indices ``index_f [R, K]`` (each in ``[0, W)``) into ``dst [R, K]``.

    Each ``nc_n_gather`` call reads at most ``_GATHER_PASS`` elements of a partition, so it runs in one pass. The
    chunks run in order; chunk ``c`` gathers at the index minus its start (clamped into the chunk) and overwrites
    ``dst`` where the index is at or past its start, so each output keeps the value of the chunk that holds its
    index. The result is the one-call gather bit for bit.

    The chunk index is clamped again in the integer domain, after its uint32 copy, to the chunk size - 1: a NaN index
    can pass the FP32 clamps (a compare-select min / max keeps it), and its uint32 copy is implementation-defined. An
    index in range keeps its value, so the gather is the same.
    """
    r, w = data.shape
    k = index_f.shape[1]
    vector = nisa.engine.vector
    for start in range(0, w, _GATHER_PASS):
        stop = min(start + _GATHER_PASS, w)
        local = nl.ndarray((r, k), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=local,
            data=index_f,
            op0=nl.subtract,
            operand0=float(start),
            op1=nl.maximum,
            operand1=0.0,
            engine=vector,
        )
        nisa.tensor_scalar(dst=local, data=local, op0=nl.minimum, operand0=float(stop - start - 1), engine=vector)
        local_u = nl.ndarray((r, k), dtype=nl.uint32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=local_u, src=local, engine=vector)
        nisa.tensor_scalar(dst=local_u, data=local_u, op0=nl.minimum, operand0=float(stop - start - 1), engine=vector)
        if start == 0:
            nisa.nc_n_gather(dst=dst, data=data[0:r, start:stop], indices=local_u)
            continue
        part = nl.ndarray((r, k), dtype=dst.dtype, buffer=nl.sbuf)
        nisa.nc_n_gather(dst=part, data=data[0:r, start:stop], indices=local_u)
        inside = nl.ndarray((r, k), dtype=nl.uint8, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=inside, data=index_f, op0=nl.greater_equal, operand0=float(start), engine=vector)
        nisa.tensor_copy_predicated(dst=dst, src=part, predicate=inside)


def _hold_source(dst, value, src):
    """Write ``value [R, 1]`` to ``dst``; with a gather source ``src [R, 1]``, read it in the same op.

    A peer can still read this rank's SBUF gather source after this rank holds the gathered output. Here
    ``value`` is the output of the next collective on the same group, so this read keeps the source bytes
    live until that collective has ended, and every peer is past the gather. ``src`` is a kept mass (0 or
    more), so ``src != -1`` is 1 and ``dst`` equals ``value`` bit for bit.
    """
    if src is None:
        nisa.tensor_copy(dst=dst, src=value)
    else:
        nisa.scalar_tensor_tensor(dst=dst, data=src, op0=nl.not_equal, operand0=-1.0, op1=nl.multiply, operand1=value)


def _chunks(r, v_s):
    """Return (chunks per row C, chunk free length L) that put ``C`` vocab chunks of each row on partitions.

    ``C`` is the largest power of two with ``R * C <= 128`` that divides ``V_s``, so a search trial is one
    reduce over ``L = V_s / C`` free elements across ``R * C`` partitions instead of over all ``V_s``.
    """
    c = 1
    while c * 2 <= 128 // r and v_s % (c * 2) == 0:
        c *= 2
    return c, v_s // c


def _group_reduce(dst_rm, src_pc, r, c, op, rows_readback=False):
    """Reduce the ``C`` chunk-partitions of each row of ``src_pc [R*C, M]`` into ``dst_rm [R, M]``.

    Move the partials through HBM and read them back as ``[R, M, C]`` (row on the partition, chunk on the
    last free axis), then one free-dim reduce over the chunk axis. The DMA moves bits and the adds run on
    the Vector engine, so the result is exact; it needs no 32-aligned partition reduce.

    That readback gathers 4-byte elements at a stride of ``M`` elements, so it moves as hundreds of small DMA
    packets. With ``rows_readback`` it reads each row's ``C * M`` partials contiguously into an SBUF tile
    ``[R, C * M]`` (chunk-major), and the reduce reads that tile through the view ``[R, M, C]`` (chunk stride
    ``M``). The reduce gets the same values in the same order, so the result keeps its bits.
    """
    m = src_pc.shape[1]
    hbm = nl.ndarray((r * c, m), dtype=src_pc.dtype, buffer=nl.private_hbm)
    if not rows_readback:
        nisa.dma_copy(dst=hbm, src=src_pc)
        rmc = nl.ndarray((r, m, c), dtype=src_pc.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=rmc, src=hbm.ap(pattern=[[c * m, r], [1, m], [m, c]]))
        nisa.tensor_reduce(dst=dst_rm, op=op, data=rmc, axis=(2,))
        return
    nisa.dma_copy(dst=hbm, src=src_pc, dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.scalar)
    rcm = nl.ndarray((r, c * m), dtype=src_pc.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=rcm, src=hbm.reshape((r, c * m)), dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)
    nisa.tensor_reduce(dst=dst_rm, op=op, data=rcm.ap(pattern=[[c * m, r], [1, m], [m, c]], offset=0), axis=(2,))


def _group_bcast(dst_pc, src_r1, r, c):
    """Replicate each row of ``src_r1 [R, M]`` to its ``C`` chunk-partitions of ``dst_pc [R*C, M]``.

    Replicate the scalar across C free columns on chip, store that ``[R, C]`` block to HBM, then read it
    back contiguously as ``[R*C, 1]`` so each row's value lands on its C chunk-partitions in one DMA.
    """
    bc = nl.ndarray((r, c), dtype=src_r1.dtype, buffer=nl.sbuf)
    nisa.memset(dst=bc, value=0.0)
    nisa.tensor_scalar(dst=bc, data=bc, op0=nl.add, operand0=src_r1)
    hbm = nl.ndarray((r, c), dtype=src_r1.dtype, buffer=nl.private_hbm)
    nisa.dma_copy(dst=hbm, src=bc)
    nisa.dma_copy(dst=dst_pc, src=hbm.reshape((r * c, 1)))


def _walk_total(data):
    """Return ``[R, 1]`` FP32 SBUF: the last entry of the add scan of ``data [R, N]``.

    ``_draw`` walks this scan, so a total taken from it is the walk's total bit for bit. A reduce adds in another
    order and can differ by a few ulps; the rank pick would then give the difference to the last kept id of a
    shard (or take it from the shard's tail).
    """
    r, n = data.shape
    scan = nl.ndarray((r, n), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar_cumulative(
        dst=scan, src=data, op0=nl.add, op1=nl.add, imm0=0.0, reduce_cmd=nisa.reduce_cmd.reset_reduce
    )
    total = nl.ndarray((r, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=total, src=scan[0:r, n - 1 : n], engine=nisa.engine.vector)
    return total


def _draw(masses, kept, u, rank_f, ids, tp_size):
    """Return ``(owner, local_token)`` of the inverse-CDF draw at ``u * total`` over the kept masses.

    ``masses [R, tp_size]`` is the kept mass of each rank in rank order (the same on every rank),
    ``kept [R, V_s]`` this rank's kept masses, ``ids [R, V_s]`` the local ids and ``rank_f [R, 1]`` this
    rank's index. ``owner`` is the rank whose shard holds the draw point; ``local_token`` is the global id
    of the drawn token if this rank is the owner (only the owner's value is used).
    """
    r, v_s = kept.shape
    fp32 = nl.float32

    def sb(shape, dtype):
        return nl.ndarray(shape, dtype=dtype, buffer=nl.sbuf)

    inclusive = sb((r, tp_size), fp32)
    nisa.tensor_scalar_cumulative(
        dst=inclusive, src=masses, op0=nl.add, op1=nl.add, imm0=0.0, reduce_cmd=nisa.reduce_cmd.reset_reduce
    )
    total = sb((r, 1), fp32)
    nisa.tensor_copy(dst=total, src=inclusive[0:r, tp_size - 1 : tp_size])
    point = sb((r, 1), fp32)
    nisa.tensor_tensor(dst=point, data1=u, data2=total, op=nl.multiply)

    ranks_iota = sb((r, tp_size), fp32)
    nisa.iota(dst=ranks_iota, pattern=[[1, tp_size]])
    masspos = sb((r, tp_size), fp32)
    nisa.tensor_scalar(dst=masspos, data=masses, op0=nl.greater, operand0=0.0)
    nisa.tensor_tensor(dst=masspos, data1=masspos, data2=ranks_iota, op=nl.multiply)
    last_rank = sb((r, 1), fp32)
    nisa.tensor_reduce(dst=last_rank, data=masspos, op=nl.maximum, axis=(1,))
    le_mask = sb((r, tp_size), fp32)
    nisa.tensor_scalar(dst=le_mask, data=inclusive, op0=nl.less_equal, operand0=point)
    le_cnt = sb((r, 1), fp32)
    nisa.tensor_reduce(dst=le_cnt, data=le_mask, op=nl.add, axis=(1,))
    owner = sb((r, 1), fp32)
    nisa.tensor_tensor(dst=owner, data1=le_cnt, data2=last_rank, op=nl.minimum)

    # This rank's exclusive scan value (the inclusive total of the ranks before it), selected exactly from the same
    # scan as the rank pick, and this rank's mass, the walk's total.
    mine = sb((r, tp_size), fp32)
    nisa.tensor_scalar(dst=mine, data=ranks_iota, op0=nl.equal, operand0=rank_f)
    shifted = sb((r, tp_size), fp32)
    nisa.memset(dst=shifted, value=0.0, engine=nisa.engine.vector)
    if tp_size > 1:
        nisa.tensor_copy(dst=shifted[0:r, 1:tp_size], src=inclusive[0:r, 0 : tp_size - 1], engine=nisa.engine.vector)
    nisa.tensor_tensor(dst=shifted, data1=shifted, data2=mine, op=nl.multiply, engine=nisa.engine.vector)
    before = sb((r, 1), fp32)
    nisa.tensor_reduce(dst=before, data=shifted, op=nl.add, axis=(1,))
    own = sb((r, tp_size), fp32)
    nisa.tensor_tensor(dst=own, data1=masses, data2=mine, op=nl.multiply, engine=nisa.engine.vector)
    own_mass = sb((r, 1), fp32)
    nisa.tensor_reduce(dst=own_mass, data=own, op=nl.add, axis=(1,))
    # The point inside this rank, clamped to [0, own_mass): ``before`` is the exact exclusive scan value, so the only gap
    # is the rounding of the inclusive scan, at most half an ulp of the total (one step of the FP32 uniform). The upper
    # clamp keeps the point strictly below the walk's total, so the walk ends on a kept id by its own CDF.
    pm_before = sb((r, 1), fp32)
    nisa.tensor_tensor(dst=pm_before, data1=point, data2=before, op=nl.subtract)
    nisa.tensor_scalar(dst=pm_before, data=pm_before, op0=nl.maximum, operand0=0.0)
    top = sb((r, 1), fp32)
    nisa.tensor_scalar(dst=top, data=own_mass, op0=nl.multiply, operand0=_BELOW_ONE_2, engine=nisa.engine.vector)
    nisa.tensor_tensor(dst=pm_before, data1=pm_before, data2=top, op=nl.minimum, engine=nisa.engine.vector)

    cdf = sb((r, v_s), fp32)
    nisa.tensor_scalar_cumulative(
        dst=cdf, src=kept, op0=nl.add, op1=nl.add, imm0=0.0, reduce_cmd=nisa.reduce_cmd.reset_reduce
    )
    cdf_le = sb((r, v_s), fp32)
    nisa.tensor_scalar(dst=cdf_le, data=cdf, op0=nl.less_equal, operand0=pm_before)
    idx_cnt = sb((r, 1), fp32)
    nisa.tensor_reduce(dst=idx_cnt, data=cdf_le, op=nl.add, axis=(1,))
    keptpos = sb((r, v_s), fp32)
    nisa.tensor_scalar(dst=keptpos, data=kept, op0=nl.greater, operand0=0.0)
    nisa.tensor_tensor(dst=keptpos, data1=keptpos, data2=ids, op=nl.multiply)
    last_id = sb((r, 1), fp32)
    nisa.tensor_reduce(dst=last_id, data=keptpos, op=nl.maximum, axis=(1,))
    index = sb((r, 1), fp32)
    nisa.tensor_tensor(dst=index, data1=idx_cnt, data2=last_id, op=nl.minimum)
    local_token = sb((r, 1), fp32)
    nisa.tensor_scalar(dst=local_token, data=rank_f, op0=nl.multiply, operand0=float(v_s), op1=nl.add, operand1=index)

    return owner, local_token


def _verify_candidates(candidate_ids, excl_u, kept, masses, offset, rank_f, ids, tp_group, tp_size, pad256=False):
    """Return ``(probs_at [R, K] FP32, excl_token [R, 1] INT32)`` HBM of the candidates S of each row.

    ``probs_at``: each rank gathers the kept mass of every id of its shard (``nc_n_gather`` at the clamped local
    index, times the in-shard mask), and one sum all-reduce joins the ranks; the kept total divides it.
    ``excl_token``: a rank zeroes the kept masses of its own candidates (the ``J`` ids of its shard block, found
    by a max over the shard axis of the candidate layout), re-gathers the per-rank masses, and draws as the main
    sample with ``excl_u``. When the candidates hold all the kept mass (no excluded mass), ``excl_token`` is -1.
    Collectives: one all-reduce ``[R, K]``, one all-gather ``[R, 1]`` and one max all-reduce ``[R, 1]``.
    """
    r, v_s = kept.shape
    k = candidate_ids.shape[1]
    kernel_assert(k % tp_size == 0, f"candidate_ids needs K a multiple of tp_size, got K={k}, tp_size={tp_size}")
    j_count = k // tp_size
    kernel_assert(tuple(excl_u.shape) == (r, 1), f"excl_u must be [R, 1], got {tuple(excl_u.shape)}")
    fp32, u8, i32, u32 = nl.float32, nl.uint8, nl.int32, nl.uint32

    def sb(shape, dtype):
        return nl.ndarray(shape, dtype=dtype, buffer=nl.sbuf)

    cand_i = sb((r, k), i32)
    nisa.dma_copy(dst=cand_i, src=candidate_ids[0:r, 0:k])
    local = sb((r, k), fp32)
    nisa.tensor_copy(dst=local, src=cand_i, engine=nisa.engine.scalar)
    nisa.tensor_scalar(dst=local, data=local, op0=nl.subtract, operand0=offset, engine=nisa.engine.vector)
    # 1 for an id of this shard, else 0 (an id of another shard or -1).
    inshard = sb((r, k), fp32)
    nisa.tensor_scalar(dst=inshard, data=local, op0=nl.greater_equal, operand0=0.0, engine=nisa.engine.vector)
    nisa.scalar_tensor_tensor(
        dst=inshard, data=local, op0=nl.less, operand0=float(v_s), op1=nl.multiply, operand1=inshard
    )

    # probs_at: the kept mass at the clamped local index, masked to this shard's ids, summed over the ranks.
    index_f = sb((r, k), fp32)
    nisa.tensor_scalar(
        dst=index_f,
        data=local,
        op0=nl.maximum,
        operand0=0.0,
        op1=nl.minimum,
        operand1=float(v_s - 1),
        engine=nisa.engine.vector,
    )
    gathered = sb((r, k), fp32)
    _gather(gathered, kept, index_f)
    contrib = sb((r, k), fp32)
    nisa.tensor_tensor(dst=contrib, data1=gathered, data2=inshard, op=nl.multiply, engine=nisa.engine.vector)
    mass_at = sb((r, k), fp32)
    _allreduce(contrib, mass_at, nl.add, tp_group, tp_size, "cand_probs", pad256=pad256)
    # The total of the draw's rank walk, so probs_at is the drawn law (up to the walk's own rounding).
    total = _walk_total(masses)
    nisa.tensor_scalar(dst=total, data=total, op0=nl.maximum, operand0=_U_MIN, engine=nisa.engine.vector)
    inv_total = sb((r, 1), fp32)
    nisa.reciprocal(dst=inv_total, data=total)
    probs_sb = sb((r, k), fp32)
    # Clamped to 1: a candidate with all the kept mass can round one ulp above it.
    nisa.tensor_scalar(
        dst=probs_sb,
        data=mass_at,
        op0=nl.multiply,
        operand0=inv_total,
        op1=nl.minimum,
        operand1=1.0,
        engine=nisa.engine.vector,
    )

    # This rank's own candidates: local index + 1 where in this shard, max over the shard axis, minus 1 (-1 = none).
    plus = sb((r, k), fp32)
    nisa.scalar_tensor_tensor(dst=plus, data=local, op0=nl.add, operand0=1.0, op1=nl.multiply, operand1=inshard)
    own = sb((r, j_count), fp32)
    nisa.tensor_reduce(dst=own, data=plus.reshape((r, j_count, tp_size)), op=nl.maximum, axis=(2,))
    nisa.tensor_scalar(dst=own, data=own, op0=nl.subtract, operand0=1.0, engine=nisa.engine.vector)
    excl_mask = sb((r, v_s), fp32)
    nisa.memset(dst=excl_mask, value=0.0, engine=nisa.engine.vector)
    for j in range(j_count):
        nisa.scalar_tensor_tensor(
            dst=excl_mask, data=ids, op0=nl.equal, operand0=own[0:r, j : j + 1], op1=nl.maximum, operand1=excl_mask
        )
    keep = sb((r, v_s), fp32)
    nisa.tensor_scalar(
        dst=keep, data=excl_mask, op0=nl.subtract, operand0=1.0, reverse0=True, engine=nisa.engine.vector
    )
    kept_e = sb((r, v_s), fp32)
    nisa.tensor_tensor(dst=kept_e, data1=kept, data2=keep, op=nl.multiply, engine=nisa.engine.vector)
    shard_kept_e = _walk_total(kept_e)
    masses_e = sb((r, tp_size), fp32)
    nisa.memset(dst=masses_e, value=0.0, engine=nisa.engine.vector)
    _allgather(shard_kept_e, masses_e, tp_group, tp_size, "excl_masses")
    u_e = sb((r, 1), fp32)
    nisa.dma_copy(dst=u_e, src=excl_u[0:r, 0:1])
    nisa.tensor_scalar(dst=u_e, data=u_e, op0=nl.maximum, operand0=_U_MIN, engine=nisa.engine.vector)
    owner_e, local_token_e = _draw(masses_e, kept_e, u_e, rank_f, ids, tp_size)
    is_owner_e = sb((r, 1), u8)
    nisa.tensor_scalar(dst=is_owner_e, data=owner_e, op0=nl.equal, operand0=rank_f, engine=nisa.engine.vector)
    result_e = sb((r, 1), fp32)
    nisa.memset(dst=result_e, value=-1.0, engine=nisa.engine.vector)
    nisa.tensor_copy_predicated(dst=result_e, src=local_token_e, predicate=is_owner_e)
    drawn_e = sb((r, 1), fp32)
    _allreduce(result_e, drawn_e, nl.maximum, tp_group, tp_size, "excl_draw", pad256=pad256)
    if tp_size > 1:
        _hold_source(drawn_e, drawn_e, shard_kept_e)
    nisa.tensor_scalar(
        dst=drawn_e,
        data=drawn_e,
        op0=nl.maximum,
        operand0=0.0,
        op1=nl.minimum,
        operand1=float(v_s * tp_size - 1),
        engine=nisa.engine.vector,
    )
    # No excluded mass: -1, so the caller never takes a token without mass (the gathered masses agree on all ranks).
    no_mass = sb((r, 1), u8)
    nisa.tensor_scalar(
        dst=no_mass, data=_walk_total(masses_e), op0=nl.less_equal, operand0=0.0, engine=nisa.engine.vector
    )
    minus_one = sb((r, 1), fp32)
    nisa.memset(dst=minus_one, value=-1.0, engine=nisa.engine.vector)
    nisa.tensor_copy_predicated(dst=drawn_e, src=minus_one, predicate=no_mass)
    excl_sb = sb((r, 1), i32)
    nisa.tensor_copy(dst=excl_sb, src=drawn_e, engine=nisa.engine.scalar)

    probs_at = nl.ndarray((r, k), dtype=fp32, buffer=nl.shared_hbm)
    excl_token = nl.ndarray((r, 1), dtype=i32, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=probs_at, src=probs_sb)
    nisa.dma_copy(dst=excl_token, src=excl_sb)
    return probs_at, excl_token


_REPLACE8_MAX_FREE = 16384
"""The most free elements per partition of one ``nc_match_replace8`` (the ``max8`` class limit)."""


def _verify_fused_prepare(candidate_ids, kept, offset, ids, tp_group, tp_size, pad256=False):
    """``verify_fused``: the candidate work of ``_verify_candidates`` that needs only ``kept`` and the candidates, so it
    runs before the main draw. Return ``(mass_at [R, K], kept_e [R, V_s], shard_kept_e [R, 1])``.

    ``mass_at`` is the kept mass of each candidate id (the clamped gather, the in-shard mask and the sum all-reduce of
    ``_verify_candidates``). ``kept_e`` is ``kept`` without this rank's own candidates: one ``nc_match_replace8`` of the
    id iota a group of 8 own local ids (-1 marks a match), then one select-multiply, in place of one compare pass of
    ``[R, V_s]`` an own candidate. The products are the same 1.0 / 0.0 multiplies, so ``kept_e`` and its walk total
    ``shard_kept_e`` keep their bits."""
    r, v_s = kept.shape
    k = candidate_ids.shape[1]
    kernel_assert(k % tp_size == 0, f"candidate_ids needs K a multiple of tp_size, got K={k}, tp_size={tp_size}")
    kernel_assert(v_s <= _REPLACE8_MAX_FREE, f"verify_fused needs V_s <= {_REPLACE8_MAX_FREE}, got {v_s}")
    j_count = k // tp_size
    fp32, i32 = nl.float32, nl.int32
    vector = nisa.engine.vector

    def sb(shape, dtype):
        return nl.ndarray(shape, dtype=dtype, buffer=nl.sbuf)

    cand_i = sb((r, k), i32)
    nisa.dma_copy(dst=cand_i, src=candidate_ids[0:r, 0:k])
    local = sb((r, k), fp32)
    nisa.tensor_copy(dst=local, src=cand_i, engine=nisa.engine.scalar)
    nisa.tensor_scalar(dst=local, data=local, op0=nl.subtract, operand0=offset, engine=vector)
    inshard = sb((r, k), fp32)
    nisa.tensor_scalar(dst=inshard, data=local, op0=nl.greater_equal, operand0=0.0, engine=vector)
    nisa.scalar_tensor_tensor(
        dst=inshard, data=local, op0=nl.less, operand0=float(v_s), op1=nl.multiply, operand1=inshard
    )
    index_f = sb((r, k), fp32)
    nisa.tensor_scalar(
        dst=index_f,
        data=local,
        op0=nl.maximum,
        operand0=0.0,
        op1=nl.minimum,
        operand1=float(v_s - 1),
        engine=vector,
    )
    gathered = sb((r, k), fp32)
    _gather(gathered, kept, index_f)
    contrib = sb((r, k), fp32)
    nisa.tensor_tensor(dst=contrib, data1=gathered, data2=inshard, op=nl.multiply, engine=vector)
    mass_at = sb((r, k), fp32)
    _allreduce(contrib, mass_at, nl.add, tp_group, tp_size, "cand_probs", pad256=pad256)

    # This rank's own candidates: local index + 1 where in this shard, max over the shard axis, minus 1 (-1 = none).
    plus = sb((r, k), fp32)
    nisa.scalar_tensor_tensor(dst=plus, data=local, op0=nl.add, operand0=1.0, op1=nl.multiply, operand1=inshard)
    own = sb((r, j_count), fp32)
    nisa.tensor_reduce(dst=own, data=plus.reshape((r, j_count, tp_size)), op=nl.maximum, axis=(2,))
    nisa.tensor_scalar(dst=own, data=own, op0=nl.subtract, operand0=1.0, engine=vector)
    # Mark the own ids in a copy of the id iota (-1 at a match; an own -1 or the -2 pad matches no id), 8 a pass.
    marked = sb((r, v_s), fp32)
    src = ids
    for c0 in range(0, j_count, 8):
        width = min(8, j_count - c0)
        vals = sb((r, 8), fp32)
        nisa.memset(dst=vals, value=-2.0, engine=vector)
        nisa.tensor_copy(dst=vals[0:r, 0:width], src=own[0:r, c0 : c0 + width], engine=vector)
        nisa.nc_match_replace8(dst=marked, data=src, vals=vals, imm=-1.0)
        src = marked
    kept_e = sb((r, v_s), fp32)
    nisa.scalar_tensor_tensor(
        dst=kept_e, data=marked, op0=nl.greater_equal, operand0=0.0, op1=nl.multiply, operand1=kept
    )
    shard_kept_e = _walk_total(kept_e)
    return mass_at, kept_e, shard_kept_e


def _verify_fused_finish(mass_at, masses, masses_e, drawn_e, hold_src, tp_size, v_s):
    """``verify_fused``: the outputs of ``_verify_candidates`` from the fused collectives: ``probs_at`` from ``mass_at``
    and the main ``masses``, ``excl_token`` from the excluded draw ``drawn_e`` (the second column of the draw
    all-reduce) and ``masses_e``. ``hold_src`` is the excluded-mass column of the masses all-gather source (``None``
    without a gather). The same instructions on the same values as ``_verify_candidates``, so the outputs keep their
    bits. Return ``(probs_at, excl_token)`` HBM."""
    r, k = mass_at.shape
    fp32, u8, i32 = nl.float32, nl.uint8, nl.int32
    vector = nisa.engine.vector

    def sb(shape, dtype):
        return nl.ndarray(shape, dtype=dtype, buffer=nl.sbuf)

    total = _walk_total(masses)
    nisa.tensor_scalar(dst=total, data=total, op0=nl.maximum, operand0=_U_MIN, engine=vector)
    inv_total = sb((r, 1), fp32)
    nisa.reciprocal(dst=inv_total, data=total)
    probs_sb = sb((r, k), fp32)
    nisa.tensor_scalar(
        dst=probs_sb,
        data=mass_at,
        op0=nl.multiply,
        operand0=inv_total,
        op1=nl.minimum,
        operand1=1.0,
        engine=vector,
    )
    excl = sb((r, 1), fp32)
    if hold_src is not None:
        _hold_source(excl, drawn_e, hold_src)
    else:
        nisa.tensor_copy(dst=excl, src=drawn_e, engine=vector)
    nisa.tensor_scalar(
        dst=excl,
        data=excl,
        op0=nl.maximum,
        operand0=0.0,
        op1=nl.minimum,
        operand1=float(v_s * tp_size - 1),
        engine=vector,
    )
    no_mass = sb((r, 1), u8)
    nisa.tensor_scalar(dst=no_mass, data=_walk_total(masses_e), op0=nl.less_equal, operand0=0.0, engine=vector)
    minus_one = sb((r, 1), fp32)
    nisa.memset(dst=minus_one, value=-1.0, engine=vector)
    nisa.tensor_copy_predicated(dst=excl, src=minus_one, predicate=no_mass)
    excl_sb = sb((r, 1), i32)
    nisa.tensor_copy(dst=excl_sb, src=excl, engine=nisa.engine.scalar)
    probs_at = nl.ndarray((r, k), dtype=fp32, buffer=nl.shared_hbm)
    excl_token = nl.ndarray((r, 1), dtype=i32, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=probs_at, src=probs_sb)
    nisa.dma_copy(dst=excl_token, src=excl_sb)
    return probs_at, excl_token


def _topn_rounds(data, n):
    """Return the ``n`` largest values of each row of ``data`` (descending) and their flattened positions.

    ``data`` has ``R`` partitions and 8..16384 free elements. Ties resolve to the lowest position, because
    ``nc_find_index8`` pairs equal values with ascending positions. Found entries are masked with -3e38
    (below any real logit) between rounds, so the next ``max8`` skips them.
    """
    r = data.shape[0]
    rounds = (n + 7) // 8
    work = nl.ndarray(data.shape, dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=work, src=data)
    vals = nl.ndarray((r, n), dtype=nl.float32, buffer=nl.sbuf)
    pos = nl.ndarray((r, n), dtype=nl.uint32, buffer=nl.sbuf)
    val8 = nl.ndarray((r, 8), dtype=nl.float32, buffer=nl.sbuf)
    idx8 = nl.ndarray((r, 8), dtype=nl.uint32, buffer=nl.sbuf)
    for rd in range(rounds):
        lo = rd * 8
        w = min(8, n - lo)
        nisa.max8(dst=val8, src=work)
        nisa.nc_find_index8(dst=idx8, data=work, vals=val8)
        nisa.tensor_copy(dst=vals[0:r, lo : lo + w], src=val8[0:r, 0:w])
        nisa.tensor_copy(dst=pos[0:r, lo : lo + w], src=idx8[0:r, 0:w])
        if rd < rounds - 1:
            nisa.nc_match_replace8(dst=work, data=work, vals=val8, imm=-3e38)
    return vals, pos


def _merge_rounds(cvals, cids, n):
    """Global top-``n`` over the rank-major compact candidates: the ``max8`` rounds over ``cvals``, then
    ``nc_n_gather`` the matching global ids from ``cids``. Rank order then slot order keeps the lowest id.

    A position that ``nc_find_index8`` does not find (0xFFFFFFFF, e.g. for a NaN value) is clamped to the last
    candidate in the integer domain before the one-pass gather, so the gather index stays in range; every found
    position is below the bound, so the ids are the same."""
    r = cvals.shape[0]
    gvals, gpos = _topn_rounds(cvals, n)
    gids = nl.ndarray((r, n), dtype=nl.float32, buffer=nl.sbuf)
    if cids.shape[1] <= _GATHER_PASS:
        nisa.tensor_scalar(
            dst=gpos, data=gpos, op0=nl.minimum, operand0=float(cids.shape[1] - 1), engine=nisa.engine.vector
        )
        nisa.nc_n_gather(dst=gids, data=cids, indices=gpos)
    else:
        gpos_f = nl.ndarray((r, n), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=gpos_f, src=gpos)
        _gather(gids, cids, gpos_f)
    return gvals, gids


def _merge_topn(vals_s, gids_s, n, tp_group, tp_size):
    """Merge each rank's top-``n`` ``(value, global id)`` pairs into the row's global top-``n``.

    All-gather the ``[R, 2n]`` pairs through a fresh private-HBM pair, like ``_allreduce`` does, along the
    partition dim (collective_dim 0 is the only HBM-supported axis): rank ``k`` lands in rows ``[k*R,
    (k+1)*R)`` of ``[tp_size*R, 2n]``. Read that back with an access pattern into compact rank-major
    ``[R, tp_size*n]`` value and id tiles, then re-run the ``max8`` rounds. ``tp_size == 1`` is already the
    answer. This is the standalone (no-fold) path: it owns a separate "topn" all-gather.
    """
    r = vals_s.shape[0]
    fp32 = nl.float32
    if tp_size == 1:
        return vals_s, gids_s
    width = tp_size * n
    cw = max(width, 8)  # max8 / nc_find_index8 need >= 8 elements per partition
    send = nl.ndarray((r, 2 * n), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=send[0:r, 0:n], src=vals_s)
    nisa.tensor_copy(dst=send[0:r, n : 2 * n], src=gids_s)
    hsrc = nl.ndarray((r, 2 * n), dtype=fp32, buffer=nl.private_hbm)
    nisa.dma_copy(dst=hsrc, src=send)
    hdst = nl.ndarray((tp_size * r, 2 * n), dtype=fp32, buffer=nl.private_hbm)
    ncc.all_gather(srcs=[hsrc], dsts=[hdst], replica_group=tp_group, collective_dim=0, name="topn")
    cvals = nl.ndarray((r, cw), dtype=fp32, buffer=nl.sbuf)
    cids = nl.ndarray((r, cw), dtype=fp32, buffer=nl.sbuf)
    if cw != width:
        # Pad the short candidate set so max8 has >= 8 columns; -3e38 never wins over a real logit.
        nisa.memset(dst=cvals, value=-3e38)
        nisa.memset(dst=cids, value=0.0)
    # rank-major: cvals[i, k*n + j] = hdst[k*R + i, j], picking the value (offset 0) or id (offset n) half.
    nisa.dma_copy(dst=cvals[0:r, 0:width], src=hdst.ap(pattern=[[2 * n, r], [r * 2 * n, tp_size], [1, n]], offset=0))
    nisa.dma_copy(dst=cids[0:r, 0:width], src=hdst.ap(pattern=[[2 * n, r], [r * 2 * n, tp_size], [1, n]], offset=n))
    return _merge_rounds(cvals, cids, n)


def _any_asks(ask, r):
    """Return a register that holds 1 when any row of ``ask [R, 1]`` is set (> 0.5), else 0.

    Built like ``_any_active_top_k``: the per-row flags go through a small HBM round trip into one
    partition for the max. ``params`` is bit-identical on every rank, so every rank takes the same trip.
    """
    fp32 = nl.float32
    pos = nl.ndarray((r, 1), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=pos, data=ask, op0=nl.greater, operand0=0.5)
    hbm = nl.ndarray((r, 1), dtype=fp32, buffer=nl.private_hbm)
    nisa.dma_copy(dst=hbm, src=pos)
    row = nl.ndarray((1, r), dtype=fp32, buffer=nl.sbuf)
    nisa.dma_copy(dst=row, src=hbm.reshape((1, r)))
    any_f = nl.ndarray((1, 1), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=any_f, data=row, op=nl.maximum, axis=(1,))
    any_i = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=any_i, src=any_f)
    trip = nisa.register_alloc()
    nisa.register_load(dst=trip, src=any_i)
    return trip


def _fold_gather(shard_kept, vals_s, gids_s, masses, n, tp_group, tp_size):
    """Ride each rank's top-``n`` (value, id) pairs on the masses all-gather and split the result.

    The send is ``[R, 1 + 2n]`` = ``[shard_kept | vals_s | gids_s]``, gathered along the free dim
    (collective_dim 1 on SBUF) into ``[R, tp_size*(1+2n)]`` rank-major, into the stable ``gathered`` tile the
    caller passed as ``masses``'s sibling. ``masses [R, tp_size]`` takes column 0 of each rank block; the
    compact rank-major ``[R, tp_size*n]`` value and id tiles take the other two. ``tp_size == 1`` needs no
    gather: masses is the row's own kept mass and the candidates are its own top-``n``. The third return is the
    gather's SBUF source column (for ``_hold_source``), or None without a gather.
    """
    r = shard_kept.shape[0]
    fp32 = nl.float32
    if tp_size == 1:
        nisa.tensor_copy(dst=masses, src=shard_kept)
        return vals_s, gids_s, None
    w1 = 1 + 2 * n
    send = nl.ndarray((r, w1), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=send[0:r, 0:1], src=shard_kept)
    nisa.tensor_copy(dst=send[0:r, 1 : 1 + n], src=vals_s)
    nisa.tensor_copy(dst=send[0:r, 1 + n : w1], src=gids_s)
    width = tp_size * n
    cw = max(width, 8)
    gathered = nl.ndarray((r, tp_size * w1), dtype=fp32, buffer=nl.sbuf)
    _allgather(send, gathered, tp_group, tp_size, "masses")
    hbm = nl.ndarray((r, tp_size * w1), dtype=fp32, buffer=nl.private_hbm)
    nisa.dma_copy(dst=hbm, src=gathered)
    # masses[i, k] = hbm[i, k*w1 + 0]; cvals[i, k*n+j] = hbm[i, k*w1 + 1 + j]; cids at offset 1 + n.
    nisa.dma_copy(dst=masses, src=hbm.ap(pattern=[[tp_size * w1, r], [w1, tp_size]]))
    cvals = nl.ndarray((r, cw), dtype=fp32, buffer=nl.sbuf)
    cids = nl.ndarray((r, cw), dtype=fp32, buffer=nl.sbuf)
    if cw != width:
        nisa.memset(dst=cvals, value=-3e38)
        nisa.memset(dst=cids, value=0.0)
    nisa.dma_copy(dst=cvals[0:r, 0:width], src=hbm.ap(pattern=[[tp_size * w1, r], [w1, tp_size], [1, n]], offset=1))
    nisa.dma_copy(dst=cids[0:r, 0:width], src=hbm.ap(pattern=[[tp_size * w1, r], [w1, tp_size], [1, n]], offset=1 + n))
    return cvals, cids, send[0:r, 0:1]


def _any_active_top_k(top_k, v, r):
    """Return a register that holds 1 when a row of ``top_k [R, 1]`` is active (``0 < top_k < V``), else 0.

    The rows sit on partitions, so their max goes through a small HBM round trip into one partition.
    ``params`` is bit-identical on every rank, so every rank takes the same trip count.
    """
    fp32 = nl.float32
    act = nl.ndarray((r, 1), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=act, data=top_k, op0=nl.greater, operand0=0.0)
    below = nl.ndarray((r, 1), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=below, data=top_k, op0=nl.less, operand0=float(v))
    nisa.tensor_tensor(dst=act, data1=act, data2=below, op=nl.multiply)
    hbm = nl.ndarray((r, 1), dtype=fp32, buffer=nl.private_hbm)
    nisa.dma_copy(dst=hbm, src=act)
    row = nl.ndarray((1, r), dtype=fp32, buffer=nl.sbuf)
    nisa.dma_copy(dst=row, src=hbm.reshape((1, r)))
    any_f = nl.ndarray((1, 1), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=any_f, data=row, op=nl.maximum, axis=(1,))
    any_i = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=any_i, src=any_f)
    trip = nisa.register_alloc()
    nisa.register_load(dst=trip, src=any_i)
    return trip


def _batched_allreduce(local_list, op, tp_group, tp_size, name, rr, pad256=False):
    """One all-reduce over all row chunks: stage each chunk's ``[chunk_r, width]`` local into a single
    ``[R, width]`` private-HBM tile, reduce once, read back per chunk. The collective count stays at one
    regardless of the chunk count (the row-chunking CC budget). ``rr`` is the list of ``(lo, chunk_r)``.
    ``pad256`` (``cc_pad256``) takes the pair of ``_pad256_pair``.
    """
    width = local_list[0].shape[1]
    dst_list = [nl.ndarray((hr, width), dtype=nl.float32, buffer=nl.sbuf) for (_lo, hr) in rr]
    if tp_size == 1:
        for ci in range(len(local_list)):
            nisa.tensor_copy(dst=dst_list[ci], src=local_list[ci])
        return dst_list
    rows = sum(hr for (_lo, hr) in rr)
    if pad256:
        hsrc, hdst, src_view, dst_view = _pad256_pair(rows, width, nl.float32)
        for ci, (lo, hr) in enumerate(rr):
            nisa.dma_copy(dst=src_view[lo : lo + hr, :], src=local_list[ci])
        ncc.all_reduce(srcs=[hsrc], dsts=[hdst], replica_group=tp_group, op=op, name=name)
        for ci, (lo, hr) in enumerate(rr):
            nisa.dma_copy(dst=dst_list[ci], src=dst_view[lo : lo + hr, :])
        return dst_list
    hsrc = nl.ndarray((rows, width), dtype=nl.float32, buffer=nl.private_hbm)
    hdst = nl.ndarray((rows, width), dtype=nl.float32, buffer=nl.private_hbm)
    for ci, (lo, hr) in enumerate(rr):
        nisa.dma_copy(dst=hsrc[lo : lo + hr, :], src=local_list[ci])
    ncc.all_reduce(srcs=[hsrc], dsts=[hdst], replica_group=tp_group, op=op, name=name)
    for ci, (lo, hr) in enumerate(rr):
        nisa.dma_copy(dst=dst_list[ci], src=hdst[lo : lo + hr, :])
    return dst_list


def _batched_masses(shard_kept_list, tp_group, tp_size, name, rr):
    """One all-gather of the per-rank kept mass over all row chunks, through HBM along the partition dim
    (collective_dim 0 is the only HBM-supported axis). Returns per-chunk ``masses [chunk_r, tp_size]``,
    rank ``k`` in column ``k``. ``tp_size == 1`` is the row's own kept mass.
    """
    rows = sum(hr for (_lo, hr) in rr)
    masses_list = [nl.ndarray((hr, max(tp_size, 1)), dtype=nl.float32, buffer=nl.sbuf) for (_lo, hr) in rr]
    if tp_size == 1:
        for ci in range(len(shard_kept_list)):
            nisa.tensor_copy(dst=masses_list[ci], src=shard_kept_list[ci])
        return masses_list
    hsrc = nl.ndarray((rows, 1), dtype=nl.float32, buffer=nl.private_hbm)
    for ci, (lo, hr) in enumerate(rr):
        nisa.dma_copy(dst=hsrc[lo : lo + hr, :], src=shard_kept_list[ci])
    hdst = nl.ndarray((tp_size * rows, 1), dtype=nl.float32, buffer=nl.private_hbm)
    ncc.all_gather(srcs=[hsrc], dsts=[hdst], replica_group=tp_group, collective_dim=0, name=name)
    # masses[ci][r, k] = hdst[k*rows + lo + r, 0]
    for ci, (lo, hr) in enumerate(rr):
        nisa.dma_copy(dst=masses_list[ci], src=hdst.ap(pattern=[[1, hr], [rows, tp_size]], offset=lo))
    return masses_list


def _sample_rows(
    logits,
    params,
    rank_info,
    tp_group,
    tp_size,
    return_masses,
    num_logprobs,
    skip_inactive_top_k,
    fold_topn,
    skip_unasked_logprobs,
    rows_readback=False,
    pad256=False,
):
    """Row-chunked full-vocab sampler for ``R > 128``: process the rows in 128-row chunks, one partition
    per row within a chunk, with every collective batched once over all rows (constant collective count).

    This path is the sampling core (tokens, flags, optional kept masses). It rejects the logprobs and the skip
    keys, which the chunked path does not give. ``rows_readback`` is the readback form of
    the chunk reduces (``_group_reduce``).
    """
    fp32, u8, i32, u16, u32, bf16 = nl.float32, nl.uint8, nl.int32, nl.uint16, nl.uint32, nl.bfloat16
    _MAX_CHUNKS = 8
    r, v_s = logits.shape
    kernel_assert(1 <= r <= 128 * _MAX_CHUNKS, f"row_chunk takes 1 to {128 * _MAX_CHUNKS} rows, got R={r}")
    kernel_assert(logits.dtype == bf16, "logits must be BF16")
    kernel_assert(params.dtype == fp32, "params must be FP32")
    kernel_assert(tuple(rank_info.shape) == (1, 4), f"rank_info must be [1, 4], got {tuple(rank_info.shape)}")
    kernel_assert(tp_size >= 1 and (tp_size == 1 or tp_group is not None), "tp_size > 1 needs tp_group")
    kernel_assert(
        num_logprobs < 0 and not fold_topn and not skip_unasked_logprobs and not skip_inactive_top_k,
        "row_chunk logprobs / skip keys land in a follow-up; core only for now",
    )
    kernel_assert(tuple(params.shape) == (r, 4), f"params must be [R, 4] = [{r}, 4], got {tuple(params.shape)}")
    v = v_s * tp_size
    n_chunks = (r + 127) // 128
    rr = [(ci * 128, min(128, r - ci * 128)) for ci in range(n_chunks)]

    def sb(shape, dtype):
        return nl.ndarray(shape, dtype=dtype, buffer=nl.sbuf)

    def stage(tile):
        # Park a big per-chunk tile in HBM so only one chunk's tiles stay resident (SBUF fits at v_s=5120).
        h = nl.ndarray(tile.shape, dtype=tile.dtype, buffer=nl.private_hbm)
        nisa.dma_copy(dst=h, src=tile)
        return h

    def rl(h):
        t = nl.ndarray(h.shape, dtype=h.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=t, src=h)
        return t

    # The big per-chunk tiles live in HBM (one chunk resident at a time); the small state stays in SBUF.
    key_l, keyc_l, safe_l, safec_l, isfin_l, isfinc_l = [], [], [], [], [], []
    u_l, rankf_l, topk_l, topp_l, temp_l = [], [], [], [], []
    statssrc_l, pc_l, c_l, el_l = [], [], [], []
    # ---- per-chunk prep: keys, NaN/+inf-safe logits, local max + stats source ----
    for lo, cr in rr:
        logits_sb = sb((cr, v_s), bf16)
        nisa.dma_copy(dst=logits_sb, src=logits[lo : lo + cr, 0:v_s])
        logits_f = sb((cr, v_s), fp32)
        nisa.tensor_copy(dst=logits_f, src=logits_sb)
        params_sb = sb((cr, 4), fp32)
        nisa.dma_copy(dst=params_sb, src=params[lo : lo + cr, 0:4])
        topk_l.append(params_sb[0:cr, 0:1])
        topp_l.append(params_sb[0:cr, 1:2])
        temp_l.append(params_sb[0:cr, 2:3])
        u = sb((cr, 1), fp32)
        nisa.tensor_scalar(dst=u, data=params_sb[0:cr, 3:4], op0=nl.maximum, operand0=_U_MIN)
        u_l.append(u)
        rank_f = sb((cr, 1), fp32)
        nisa.dma_copy(dst=rank_f, src=rank_info[0:1, 0:1].broadcast(0, cr))
        rankf_l.append(rank_f)
        offset = sb((cr, 1), fp32)
        nisa.tensor_scalar(dst=offset, data=rank_f, op0=nl.multiply, operand0=float(v_s))
        c, el = _chunks(cr, v_s)
        pc = cr * c
        c_l.append(c)
        el_l.append(el)
        pc_l.append(pc)
        # order key of each logit
        bits_f = sb((cr, v_s), fp32)
        nisa.tensor_copy(dst=bits_f, src=logits_sb.view(u16))
        key = sb((cr, v_s), fp32)
        nisa.tensor_scalar(dst=key, data=bits_f, op0=nl.subtract, operand0=_WHOLE, reverse0=True)
        keypos = sb((cr, v_s), fp32)
        nisa.tensor_scalar(dst=keypos, data=bits_f, op0=nl.add, operand0=_HALF)
        poscond = sb((cr, v_s), u8)
        nisa.tensor_scalar(dst=poscond, data=bits_f, op0=nl.less, operand0=_HALF)
        nisa.tensor_copy_predicated(dst=key, src=keypos, predicate=poscond)
        key_l.append(stage(key))
        # chunked copy of the shard rows (the row slice is contiguous, so it reshapes to [cr*c, el])
        logits_ch = sb((pc, el), bf16)
        nisa.dma_copy(dst=logits_ch, src=logits[lo : lo + cr, 0:v_s].reshape((pc, el)))
        bits_ch = sb((pc, el), fp32)
        nisa.tensor_copy(dst=bits_ch, src=logits_ch.view(u16))
        key_ch = sb((pc, el), fp32)
        nisa.tensor_scalar(dst=key_ch, data=bits_ch, op0=nl.subtract, operand0=_WHOLE, reverse0=True)
        keypos_ch = sb((pc, el), fp32)
        nisa.tensor_scalar(dst=keypos_ch, data=bits_ch, op0=nl.add, operand0=_HALF)
        poscond_ch = sb((pc, el), u8)
        nisa.tensor_scalar(dst=poscond_ch, data=bits_ch, op0=nl.less, operand0=_HALF)
        nisa.tensor_copy_predicated(dst=key_ch, src=keypos_ch, predicate=poscond_ch)
        keyc_l.append(stage(key_ch))
        logits_ch_f = sb((pc, el), fp32)
        nisa.tensor_copy(dst=logits_ch_f, src=logits_ch)
        notnan_ch = sb((pc, el), fp32)
        nisa.tensor_tensor(dst=notnan_ch, data1=logits_ch_f, data2=logits_ch_f, op=nl.equal)
        nisa.scalar_tensor_tensor(
            dst=notnan_ch, data=logits_ch_f, op0=nl.less, operand0=_FINITE_MAX, op1=nl.multiply, operand1=notnan_ch
        )
        nan_elem_ch = sb((pc, el), u8)
        nisa.tensor_scalar(dst=nan_elem_ch, data=notnan_ch, op0=nl.less, operand0=0.5)
        neg_big_ch = sb((pc, el), fp32)
        nisa.memset(dst=neg_big_ch, value=-1e30)
        logits_safe_ch = sb((pc, el), fp32)
        nisa.tensor_copy(dst=logits_safe_ch, src=logits_ch_f)
        nisa.tensor_copy_predicated(dst=logits_safe_ch, src=neg_big_ch, predicate=nan_elem_ch)
        safec_l.append(stage(logits_safe_ch))
        is_finite_ch = sb((pc, el), fp32)
        nisa.scalar_tensor_tensor(
            dst=is_finite_ch, data=logits_ch_f, op0=nl.greater, operand0=-1e38, op1=nl.multiply, operand1=notnan_ch
        )
        isfinc_l.append(stage(is_finite_ch))
        # NaN / +inf per row and a safe copy
        notnan = sb((cr, v_s), fp32)
        nisa.tensor_tensor(dst=notnan, data1=logits_f, data2=logits_f, op=nl.equal)
        nisa.scalar_tensor_tensor(
            dst=notnan, data=logits_f, op0=nl.less, operand0=_FINITE_MAX, op1=nl.multiply, operand1=notnan
        )
        nan_inv = sb((cr, 1), fp32)
        nisa.tensor_reduce(dst=nan_inv, data=notnan, op=nl.minimum, axis=(1,))
        nan_row = sb((cr, 1), fp32)
        nisa.tensor_scalar(dst=nan_row, data=nan_inv, op0=nl.subtract, operand0=1.0, reverse0=True)
        nan_elem = sb((cr, v_s), u8)
        nisa.tensor_scalar(dst=nan_elem, data=notnan, op0=nl.less, operand0=0.5)
        neg_big = sb((cr, v_s), fp32)
        nisa.memset(dst=neg_big, value=-1e30)
        logits_safe = sb((cr, v_s), fp32)
        nisa.tensor_copy(dst=logits_safe, src=logits_f)
        nisa.tensor_copy_predicated(dst=logits_safe, src=neg_big, predicate=nan_elem)
        safe_l.append(stage(logits_safe))
        is_finite = sb((cr, v_s), fp32)
        nisa.scalar_tensor_tensor(
            dst=is_finite, data=logits_f, op0=nl.greater, operand0=-1e38, op1=nl.multiply, operand1=notnan
        )
        isfin_l.append(stage(is_finite))
        # local max and stable argmax
        local_max = sb((cr, 1), fp32)
        nisa.tensor_reduce(dst=local_max, data=logits_safe, op=nl.maximum, axis=(1,))
        ids = sb((cr, v_s), fp32)
        nisa.iota(dst=ids, pattern=[[1, v_s]])
        eqmax = sb((cr, v_s), u8)
        nisa.tensor_scalar(dst=eqmax, data=logits_safe, op0=nl.equal, operand0=local_max)
        masked_id = sb((cr, v_s), fp32)
        nisa.memset(dst=masked_id, value=float(v_s))
        nisa.tensor_copy_predicated(dst=masked_id, src=ids, predicate=eqmax)
        local_idx = sb((cr, 1), fp32)
        nisa.tensor_reduce(dst=local_idx, data=masked_id, op=nl.minimum, axis=(1,))
        local_gid = sb((cr, 1), fp32)
        nisa.tensor_scalar(dst=local_gid, data=local_idx, op0=nl.add, operand0=offset)
        stats_src = sb((cr, 4), fp32)
        nisa.tensor_scalar(dst=stats_src[0:cr, 0:1], data=local_max, op0=nl.maximum, operand0=-1e30)
        nisa.tensor_scalar(dst=stats_src[0:cr, 1:2], data=local_max, op0=nl.greater, operand0=-1e38)
        nisa.tensor_copy(dst=stats_src[0:cr, 2:3], src=nan_row)
        nisa.tensor_copy(dst=stats_src[0:cr, 3:4], src=local_gid)
        statssrc_l.append(stats_src)

    # ---- STATS + GREEDY: two batched collectives over all rows ----
    stats_src3 = [s[0 : rr[ci][1], 0:3] for ci, s in enumerate(statssrc_l)]
    stats_dst_l = _batched_allreduce(stats_src3, nl.maximum, tp_group, tp_size, "stats", rr, pad256=pad256)
    gmax_l, ghasfin_l, gnan_l, cand_l = [], [], [], []
    for ci, (lo, cr) in enumerate(rr):
        global_max = sb((cr, 1), fp32)
        nisa.tensor_copy(dst=global_max, src=stats_dst_l[ci][0:cr, 0:1])
        gmax_l.append(global_max)
        ghasfin_l.append(stats_dst_l[ci][0:cr, 1:2])
        global_nan = sb((cr, 1), fp32)
        nisa.tensor_copy(dst=global_nan, src=stats_dst_l[ci][0:cr, 2:3])
        gnan_l.append(global_nan)
        is_top = sb((cr, 1), u8)
        nisa.tensor_scalar(dst=is_top, data=statssrc_l[ci][0:cr, 0:1], op0=nl.equal, operand0=global_max)
        cand_id = sb((cr, 1), fp32)
        nisa.memset(dst=cand_id, value=_ID_BIG)
        nisa.tensor_copy_predicated(dst=cand_id, src=statssrc_l[ci][0:cr, 3:4], predicate=is_top)
        cand_l.append(cand_id)
    greedy_l = _batched_allreduce(cand_l, nl.minimum, tp_group, tp_size, "greedy", rr, pad256=pad256)

    # ---- top-k count search: 4 rounds, each a single batched all-reduce ----
    lowerk_l = [sb((rr[ci][1], 1), fp32) for ci in range(n_chunks)]
    lowerkch_l = [sb((pc_l[ci], 1), fp32) for ci in range(n_chunks)]
    trialsch_l = [sb((pc_l[ci], _SEARCH_TRIALS), fp32) for ci in range(n_chunks)]
    for ci in range(n_chunks):
        nisa.memset(dst=lowerk_l[ci], value=0.0)
        nisa.memset(dst=lowerkch_l[ci], value=0.0)
        nisa.iota(dst=trialsch_l[ci], pattern=[[1, _SEARCH_TRIALS]], offset=1)
    for i in range(_ROUNDS):
        step = float(16 ** (_ROUNDS - 1 - i))
        local_cnt_l = []
        for ci, (lo, cr) in enumerate(rr):
            pc, c, el = pc_l[ci], c_l[ci], el_l[ci]
            trial_keys_ch = sb((pc, _SEARCH_TRIALS), fp32)
            nisa.tensor_scalar(
                dst=trial_keys_ch,
                data=trialsch_l[ci],
                op0=nl.multiply,
                operand0=step,
                op1=nl.add,
                operand1=lowerkch_l[ci],
            )
            cnt_partial = sb((pc, _SEARCH_TRIALS), fp32)
            ge_ch = sb((pc, el), fp32)
            keych = rl(keyc_l[ci])
            for t in range(_SEARCH_TRIALS):
                nisa.tensor_scalar(dst=ge_ch, data=keych, op0=nl.greater_equal, operand0=trial_keys_ch[0:pc, t : t + 1])
                nisa.tensor_reduce(dst=cnt_partial[0:pc, t : t + 1], data=ge_ch, op=nl.add, axis=(1,))
            local_cnt = sb((cr, _SEARCH_TRIALS), fp32)
            _group_reduce(local_cnt, cnt_partial, cr, c, nl.add, rows_readback)
            local_cnt_l.append(local_cnt)
        glob_cnt_l = _batched_allreduce(local_cnt_l, nl.add, tp_group, tp_size, f"topk_{i}", rr, pad256=pad256)
        for ci, (lo, cr) in enumerate(rr):
            digit_mask = sb((cr, _SEARCH_TRIALS), fp32)
            nisa.tensor_scalar(dst=digit_mask, data=glob_cnt_l[ci], op0=nl.greater_equal, operand0=topk_l[ci])
            digit = sb((cr, 1), fp32)
            nisa.tensor_reduce(dst=digit, data=digit_mask, op=nl.add, axis=(1,))
            nisa.tensor_scalar(
                dst=lowerk_l[ci], data=digit, op0=nl.multiply, operand0=step, op1=nl.add, operand1=lowerk_l[ci]
            )
            digit_ch = sb((pc_l[ci], 1), fp32)
            _group_bcast(digit_ch, digit, cr, c_l[ci])
            nisa.tensor_scalar(
                dst=lowerkch_l[ci], data=digit_ch, op0=nl.multiply, operand0=step, op1=nl.add, operand1=lowerkch_l[ci]
            )

    # ---- kfloor, e, e_ch per chunk ----
    greedy_tok_l, kfloor_l, e_l, kech_l, invt_l, shardtot_l = greedy_l, [], [], [], [], []
    for ci, (lo, cr) in enumerate(rr):
        pc, c, el = pc_l[ci], c_l[ci], el_l[ci]
        safe, isfin, keyr = rl(safe_l[ci]), rl(isfin_l[ci]), rl(key_l[ci])
        safec, isfinc, keych = rl(safec_l[ci]), rl(isfinc_l[ci]), rl(keyc_l[ci])
        active = sb((cr, 1), u8)
        nisa.tensor_scalar(dst=active, data=topk_l[ci], op0=nl.greater, operand0=0.0)
        below_v = sb((cr, 1), u8)
        nisa.tensor_scalar(dst=below_v, data=topk_l[ci], op0=nl.less, operand0=float(v))
        nisa.tensor_tensor(dst=active, data1=active, data2=below_v, op=nl.multiply)
        kfloor = sb((cr, 1), fp32)
        nisa.memset(dst=kfloor, value=0.0)
        nisa.tensor_copy_predicated(dst=kfloor, src=lowerk_l[ci], predicate=active)
        kfloor_l.append(kfloor)
        is_greedy = sb((cr, 1), u8)
        nisa.tensor_scalar(dst=is_greedy, data=temp_l[ci], op0=nl.less, operand0=_SAMPLING_EPS)
        temp_safe = sb((cr, 1), fp32)
        nisa.tensor_copy(dst=temp_safe, src=temp_l[ci])
        one_const = sb((cr, 1), fp32)
        nisa.memset(dst=one_const, value=1.0)
        nisa.tensor_copy_predicated(dst=temp_safe, src=one_const, predicate=is_greedy)
        inv_t = sb((cr, 1), fp32)
        nisa.reciprocal(dst=inv_t, data=temp_safe)
        invt_l.append((inv_t, is_greedy))
        max_safe = sb((cr, 1), fp32)
        nisa.tensor_scalar(dst=max_safe, data=gmax_l[ci], op0=nl.maximum, operand0=-1e30)
        neg_max_t = sb((cr, 1), fp32)
        nisa.tensor_scalar(
            dst=neg_max_t, data=max_safe, op0=nl.multiply, operand0=inv_t, op1=nl.multiply, operand1=-1.0
        )
        arg = sb((cr, v_s), fp32)
        nisa.tensor_scalar(dst=arg, data=safe, op0=nl.multiply, operand0=inv_t, op1=nl.add, operand1=neg_max_t)
        nisa.tensor_scalar(dst=arg, data=arg, op0=nl.maximum, operand0=-80.0)
        e = sb((cr, v_s), fp32)
        nisa.activation(dst=e, op=nl.exp, data=arg)
        nisa.tensor_tensor(dst=e, data1=e, data2=isfin, op=nl.multiply)
        nisa.scalar_tensor_tensor(dst=e, data=keyr, op0=nl.greater_equal, operand0=kfloor, op1=nl.multiply, operand1=e)
        e_l.append(stage(e))
        inv_t_ch = sb((pc, 1), fp32)
        _group_bcast(inv_t_ch, inv_t, cr, c)
        neg_max_t_ch = sb((pc, 1), fp32)
        _group_bcast(neg_max_t_ch, neg_max_t, cr, c)
        kfloor_ch = sb((pc, 1), fp32)
        _group_bcast(kfloor_ch, kfloor, cr, c)
        arg_ch = sb((pc, el), fp32)
        nisa.tensor_scalar(
            dst=arg_ch, data=safec, op0=nl.multiply, operand0=inv_t_ch, op1=nl.add, operand1=neg_max_t_ch
        )
        nisa.tensor_scalar(dst=arg_ch, data=arg_ch, op0=nl.maximum, operand0=-80.0)
        e_ch = sb((pc, el), fp32)
        nisa.activation(dst=e_ch, op=nl.exp, data=arg_ch)
        nisa.tensor_tensor(dst=e_ch, data1=e_ch, data2=isfinc, op=nl.multiply)
        nisa.scalar_tensor_tensor(
            dst=e_ch, data=keych, op0=nl.greater_equal, operand0=kfloor_ch, op1=nl.multiply, operand1=e_ch
        )
        kech_l.append(stage(e_ch))
        shard_total = sb((cr, 1), fp32)
        nisa.tensor_reduce(dst=shard_total, data=e, op=nl.add, axis=(1,))
        shardtot_l.append(shard_total)

    # ---- top-p mass search: 4 rounds, each a single batched all-reduce (round 0 carries the shard total) ----
    lowerp_l = [sb((rr[ci][1], 1), fp32) for ci in range(n_chunks)]
    lowerpch_l = [sb((pc_l[ci], 1), fp32) for ci in range(n_chunks)]
    target_l = [sb((rr[ci][1], 1), fp32) for ci in range(n_chunks)]
    width0 = _SEARCH_TRIALS + 1
    for ci in range(n_chunks):
        nisa.memset(dst=lowerp_l[ci], value=0.0)
        nisa.memset(dst=lowerpch_l[ci], value=0.0)
    for i in range(_ROUNDS):
        step = float(16 ** (_ROUNDS - 1 - i))
        local_mass_l = []
        for ci, (lo, cr) in enumerate(rr):
            pc, c, el = pc_l[ci], c_l[ci], el_l[ci]
            w = width0 if i == 0 else _SEARCH_TRIALS
            keych = rl(keyc_l[ci])
            ech = rl(kech_l[ci])
            trial_keys_ch = sb((pc, _SEARCH_TRIALS), fp32)
            nisa.tensor_scalar(
                dst=trial_keys_ch,
                data=trialsch_l[ci],
                op0=nl.multiply,
                operand0=step,
                op1=nl.add,
                operand1=lowerpch_l[ci],
            )
            mass_partial = sb((pc, _SEARCH_TRIALS), fp32)
            mass_ch = sb((pc, el), fp32)
            for t in range(_SEARCH_TRIALS):
                nisa.scalar_tensor_tensor(
                    dst=mass_ch,
                    data=keych,
                    op0=nl.greater_equal,
                    operand0=trial_keys_ch[0:pc, t : t + 1],
                    op1=nl.multiply,
                    operand1=ech,
                )
                nisa.tensor_reduce(dst=mass_partial[0:pc, t : t + 1], data=mass_ch, op=nl.add, axis=(1,))
            local_mass = sb((cr, w), fp32)
            _group_reduce(local_mass[0:cr, 0:_SEARCH_TRIALS], mass_partial, cr, c, nl.add, rows_readback)
            if i == 0:
                nisa.tensor_copy(dst=local_mass[0:cr, _SEARCH_TRIALS:width0], src=shardtot_l[ci])
            local_mass_l.append(local_mass)
        glob_mass_l = _batched_allreduce(local_mass_l, nl.add, tp_group, tp_size, f"topp_{i}", rr, pad256=pad256)
        for ci, (lo, cr) in enumerate(rr):
            if i == 0:
                nisa.tensor_tensor(
                    dst=target_l[ci],
                    data1=topp_l[ci],
                    data2=glob_mass_l[ci][0:cr, _SEARCH_TRIALS:width0],
                    op=nl.multiply,
                )
            digit_mask = sb((cr, _SEARCH_TRIALS), fp32)
            nisa.tensor_scalar(
                dst=digit_mask,
                data=glob_mass_l[ci][0:cr, 0:_SEARCH_TRIALS],
                op0=nl.greater_equal,
                operand0=target_l[ci],
            )
            digit = sb((cr, 1), fp32)
            nisa.tensor_reduce(dst=digit, data=digit_mask, op=nl.add, axis=(1,))
            nisa.tensor_scalar(
                dst=lowerp_l[ci], data=digit, op0=nl.multiply, operand0=step, op1=nl.add, operand1=lowerp_l[ci]
            )
            digit_ch = sb((pc_l[ci], 1), fp32)
            _group_bcast(digit_ch, digit, cr, c_l[ci])
            nisa.tensor_scalar(
                dst=lowerpch_l[ci], data=digit_ch, op0=nl.multiply, operand0=step, op1=nl.add, operand1=lowerpch_l[ci]
            )

    # ---- cut, kept, shard kept, then the batched masses all-gather ----
    kept_l, shardkept_l = [], []
    for ci, (lo, cr) in enumerate(rr):
        keep_all = sb((cr, 1), u8)
        nisa.tensor_scalar(dst=keep_all, data=topp_l[ci], op0=nl.greater_equal, operand0=1.0)
        cut = sb((cr, 1), fp32)
        nisa.tensor_copy(dst=cut, src=lowerp_l[ci])
        zero_cut = sb((cr, 1), fp32)
        nisa.memset(dst=zero_cut, value=0.0)
        nisa.tensor_copy_predicated(dst=cut, src=zero_cut, predicate=keep_all)
        keyr = rl(key_l[ci])
        er = rl(e_l[ci])
        kept = sb((cr, v_s), fp32)
        nisa.scalar_tensor_tensor(dst=kept, data=keyr, op0=nl.greater_equal, operand0=cut, op1=nl.multiply, operand1=er)
        kept_l.append(stage(kept))
        # The shard mass is the total of the draw's walk over these kept masses (see _walk_total).
        shardkept_l.append(_walk_total(kept))
    masses_l = _batched_masses(shardkept_l, tp_group, tp_size, "masses", rr)

    # ---- draw, then one batched all-reduce of the owner's token ----
    result_l = []
    for ci, (lo, cr) in enumerate(rr):
        keptr = rl(kept_l[ci])
        ids = sb((cr, v_s), fp32)
        nisa.iota(dst=ids, pattern=[[1, v_s]])
        owner, local_token = _draw(masses_l[ci], keptr, u_l[ci], rankf_l[ci], ids, tp_size)
        is_owner = sb((cr, 1), u8)
        nisa.tensor_scalar(dst=is_owner, data=owner, op0=nl.equal, operand0=rankf_l[ci])
        result = sb((cr, 1), fp32)
        nisa.memset(dst=result, value=-1.0)
        nisa.tensor_copy_predicated(dst=result, src=local_token, predicate=is_owner)
        result_l.append(result)
    sampled_l = _batched_allreduce(result_l, nl.maximum, tp_group, tp_size, "draw", rr, pad256=pad256)

    # ---- combine greedy / sampled, NaN and -inf overrides, write tokens + flags ----
    tokens = nl.ndarray((r, 1), dtype=i32, buffer=nl.shared_hbm)
    flags = nl.ndarray((r, 1), dtype=i32, buffer=nl.shared_hbm)
    for ci, (lo, cr) in enumerate(rr):
        inv_t, is_greedy = invt_l[ci]
        token_f = sb((cr, 1), fp32)
        nisa.tensor_copy(dst=token_f, src=sampled_l[ci])
        nisa.tensor_copy_predicated(dst=token_f, src=greedy_tok_l[ci], predicate=is_greedy)
        nan_tok = sb((cr, 1), fp32)
        nisa.memset(dst=nan_tok, value=0.0)
        last_const = sb((cr, 1), fp32)
        nisa.memset(dst=last_const, value=float(v - 1))
        nisa.tensor_copy_predicated(dst=nan_tok, src=last_const, predicate=is_greedy)
        nan_pred = sb((cr, 1), u8)
        nisa.tensor_scalar(dst=nan_pred, data=gnan_l[ci], op0=nl.greater, operand0=0.0)
        nisa.tensor_copy_predicated(dst=token_f, src=nan_tok, predicate=nan_pred)
        nisa.tensor_scalar(
            dst=token_f, data=token_f, op0=nl.minimum, operand0=float(v - 1), op1=nl.maximum, operand1=0.0
        )
        tokens_sb = sb((cr, 1), i32)
        nisa.tensor_copy(dst=tokens_sb, src=token_f)
        nisa.dma_copy(dst=tokens[lo : lo + cr, :], src=tokens_sb)
        is_ninf = sb((cr, 1), fp32)
        nisa.tensor_scalar(dst=is_ninf, data=ghasfin_l[ci], op0=nl.less, operand0=0.5)
        not_nan = sb((cr, 1), fp32)
        nisa.tensor_scalar(dst=not_nan, data=gnan_l[ci], op0=nl.multiply, operand0=-1.0, op1=nl.add, operand1=1.0)
        notok = sb((cr, 1), fp32)
        nisa.tensor_tensor(dst=notok, data1=not_nan, data2=is_ninf, op=nl.multiply)
        flag_f = sb((cr, 1), fp32)
        nisa.tensor_scalar(dst=flag_f, data=notok, op0=nl.multiply, operand0=2.0, op1=nl.add, operand1=gnan_l[ci])
        flags_sb = sb((cr, 1), i32)
        nisa.tensor_copy(dst=flags_sb, src=flag_f)
        nisa.dma_copy(dst=flags[lo : lo + cr, :], src=flags_sb)
    if not return_masses:
        return tokens, flags
    kept_out = nl.ndarray((r, v_s), dtype=fp32, buffer=nl.shared_hbm)
    masses_out = nl.ndarray((r, tp_size), dtype=fp32, buffer=nl.shared_hbm)
    for ci, (lo, cr) in enumerate(rr):
        nisa.dma_copy(dst=kept_out[lo : lo + cr, :], src=rl(kept_l[ci]))
        nisa.dma_copy(dst=masses_out[lo : lo + cr, :], src=masses_l[ci][0:cr, 0:tp_size])
    return tokens, flags, kept_out, masses_out


ROWS_READBACK_OFF_FORMS = ((32, 32, True),)
"""The ``(tp_size, rows, verify)`` calls whose ``rows_readback`` default stays off on trn3 too: the 2-server b32 verify
sampler (TP32, 32 rows, the candidates of the verify). Its 93-layer step measured EQUAL with the contiguous readback, with
every point estimate slower (at 16K / 128K / 1M, inside 2 SE)."""


def verify_fused_default(trn3: bool) -> bool:
    """Return the ``verify_fused`` default of a call: on for trn3 (NeuronCore-v4 and newer), off on trn2."""
    return trn3


def rows_readback_default(tp_size: int, rows: int, verify: bool, trn3: bool) -> bool:
    """Return the ``rows_readback`` default of a call: on for trn3 (NeuronCore-v4 and newer) except the
    ``ROWS_READBACK_OFF_FORMS`` calls, off on trn2."""
    return trn3 and (tp_size, rows, verify) not in ROWS_READBACK_OFF_FORMS


@nki.jit
def full_vocab_sample(
    logits: nl.NkiTensor,
    params: nl.NkiTensor,
    rank_info: nl.NkiTensor,
    tp_group: ReplicaGroup = None,
    tp_size: int = 1,
    return_masses: bool = False,
    num_logprobs: int = -1,
    skip_inactive_top_k: bool = False,
    fold_topn: bool = False,
    skip_unasked_logprobs: bool = False,
    row_chunk: bool = False,
    candidate_ids: nl.NkiTensor = None,
    excl_u: nl.NkiTensor = None,
    rows_readback: bool = None,
    verify_fused: bool = None,
    cc_pad256: bool = False,
):
    """Sample one token per row from the full-vocab softmax with the GPU top-k and top-p rule.

    Args:
        logits: ``[R, V_s]`` BF16 HBM, this rank's vocab shard (global id = ``rank * V_s + local``).
        params: ``[R, 4]`` FP32 HBM columns ``[top_k, top_p, temperature, u]``.
        rank_info: ``[1, 4]`` INT32 HBM, this rank's index in column 0.
        tp_group: the vocab-shard replica group (``None`` for the single-rank path).
        tp_size: the group size (compile-time); global vocab is ``V_s * tp_size``.
        return_masses: also return the kept masses for the speculative exact accept (default off, so the
            sampling-only form does not pay for the two extra stores).
        num_logprobs: ``-1`` (default) is off: the program, outputs and trace are exactly the sampling
            form. ``0..20`` also returns the raw-logprobs outputs (``N = num_logprobs``), appended after
            tokens, flags and the optional ``return_masses`` pair. The log-softmax is of the raw logits
            (the BF16 input upcast to FP32): no temperature, no top-k / top-p. Needs ``V_s <= 16384``.
        skip_inactive_top_k: run the top-k trial counts only on a step with an active top-k row
            (``0 < top_k < V``), each round's counts in a 0/1-trip runtime loop; the all-reduces stay outside
            the loops. The outputs are the same as without it: a row without an active top-k takes the 0 floor
            either way. Default off keeps the program.
        fold_topn: ride each rank's top-N ``(value, global id)`` pairs on the always-on masses all-gather
            instead of a separate collective, so logprobs add no all-gather. Needs ``num_logprobs > 0``.
            Default off. The outputs keep their bits.
        skip_unasked_logprobs: run the top-N compute (per-shard rounds, merge, gather and the -inf flush) only
            on a step where a row asks for logprobs, in 0/1-trip runtime loops with no collective inside. Needs
            ``fold_topn``. ``params`` is then ``[R, 5]`` with column 4 = ``1.0`` on a row that asks. On a step
            with no ask the top-N outputs are sentinels (ids 0, logits -inf); logz and sampled_logit are still
            the true values. Default off.
        row_chunk: handle ``R > 128`` rows in 128-row chunks (one partition per row within a chunk). Each
            collective runs once over all rows (constant collective count, not per chunk). Default off keeps
            ``R <= 128`` and the default program byte-identical.
        candidate_ids: ``[R, K]`` INT32 HBM, the draft candidates S of each row for the verify of probabilistic
            drafts, or ``None`` (default; the program does not change). ``K = J * tp_size``, and column
            ``j * tp_size + s`` holds candidate ``j`` of vocab shard ``s`` (a global id of that shard), or -1 for
            no candidate. With it the kernel returns ``probs_at`` (the kept-rule probability of each id) and
            ``excl_token`` (an inverse-CDF draw with ``excl_u`` over the kept masses without S, renormalized).
            Needs ``excl_u`` and neither ``num_logprobs`` nor ``row_chunk``. With ``return_masses`` (a check
            form; the served verify does not pass it) the kept masses come before ``probs_at``.
        excl_u: ``[R, 1]`` FP32 HBM, the host draw of the ``excl_token`` draw (independent of ``u``).
        rows_readback: the chunk reduces of the 4 top-k and 4 top-p search rounds read their HBM partials back one
            contiguous row of ``C * M`` partials per row, and the reduce reads them through a strided SBUF view
            (``_group_reduce``). The other form gathers ``R * C * M`` 4-byte elements, which the DMA moves as
            hundreds of small packets on one engine. The reduce takes the same values in the same order, so the
            outputs keep their bits. None (default): ``rows_readback_default``, i.e. on for trn3 (NeuronCore-v4 and
            newer) except the ``ROWS_READBACK_OFF_FORMS`` calls (the 2-server b32 verify), and off on trn2, whose
            programs keep the gathering readback until trn2 rows measure it. False keeps the gathering readback.
        verify_fused: with ``candidate_ids``, the verify outputs from the main collectives (True; False is the kill
            switch, the program without the fused verify; None, the default: ``verify_fused_default``, i.e. on for trn3
            (NeuronCore-v4 and newer) and off on trn2, whose programs keep the separate verify until trn2 rows measure
            it): the candidate masses, the kept masses without the candidates and their shard total run
            before the main draw (they need only ``kept`` and the candidates); the excluded masses ride the main masses
            all-gather (one ``[R, 2 tp]`` gather) and the excluded draw the main draw's max all-reduce (one ``[R, 2]``
            reduce), so the verify takes 2 fewer collectives; the exclusion of the own candidates is one
            ``nc_match_replace8`` pass of 8 ids in place of one ``[R, V_s]`` pass an id. The same values, so every output
            keeps its bits. Needs ``V_s <= 16384``.
        cc_pad256: every all-reduce moves a multiple of 256 bytes: its HBM pair holds the next multiple of 256 bytes
            and only the real values are written and read (``_pad256_pair``), so every output keeps its bits.
            Default off keeps the program.

    Returns:
        tokens: ``[R, 1]`` INT32 sampled token ids.
        flags: ``[R, 1]`` INT32 row flags (1 = a NaN or +inf logit, 2 = all -inf with no NaN or +inf).
        kept: ``[R, V_s]`` FP32 unnormalized kept mass of this shard (0 for a cut token); only with
            ``return_masses``.
        masses: ``[R, tp_size]`` FP32 kept mass of each rank in rank order; only with ``return_masses``.
        topn_ids: ``[R, N]`` INT32 global ids of the N largest raw logits (value desc, ties by lowest id);
            only with ``num_logprobs > 0``.
        topn_logits: ``[R, N]`` FP32 those raw logits, bit-identical to the BF16->FP32 inputs; only with
            ``num_logprobs > 0``.
        logz: ``[R, 1]`` FP32 the logsumexp over the full vocab (NaN on a flagged row); only with logprobs.
        sampled_logit: ``[R, 1]`` FP32 the raw logit of the returned token (the global max on a greedy row,
            NaN on a flagged row); only with logprobs.
        probs_at: ``[R, K]`` FP32 probability of each ``candidate_ids`` id (0 for -1 or a cut id); only with
            ``candidate_ids``.
        excl_token: ``[R, 1]`` INT32 the draw over the kept masses without the candidates (-1 when they
            hold all the kept mass); only with ``candidate_ids``.

    Contract (the kernel asserts the shapes and dtypes; the caller keeps the rest):
        - ``R <= 128``: each row (and each of its vocab chunks) takes one partition.
        - ``logits`` is a contiguous ``[R, V_s]`` HBM tensor; the kernel reads it as ``[R * C, V_s / C]``.
        - Any vocab padding holds -inf. A greedy NaN or +inf row returns the last id, which is a padding id
          when the vocab is padded.
        - ``params`` is bit-identical on every rank; otherwise the ranks can disagree on the owner of the
          draw, and the token falls back to 0 with no flag.
    """
    kernel_assert(len(logits.shape) == 2, f"logits must be [R, V_s], got {tuple(logits.shape)}")
    r, v_s = logits.shape
    verify = candidate_ids is not None
    kernel_assert(verify == (excl_u is not None), "candidate_ids and excl_u go together")
    kernel_assert(
        not verify or not (row_chunk or num_logprobs >= 0),
        "candidate_ids takes no row_chunk or num_logprobs",
    )
    if verify:
        kernel_assert(
            len(candidate_ids.shape) == 2 and candidate_ids.shape[0] == r and candidate_ids.dtype == nl.int32,
            f"candidate_ids must be [R, K] = [{r}, K] INT32, got {tuple(candidate_ids.shape)} {candidate_ids.dtype}",
        )
        kernel_assert(excl_u.dtype == nl.float32, f"excl_u must be FP32, got {excl_u.dtype}")
    if rows_readback is None:
        rows_readback = rows_readback_default(tp_size, r, verify, nisa.get_nc_version() >= nisa.nc_version.gen4)
    if verify_fused is None:
        verify_fused = verify_fused_default(nisa.get_nc_version() >= nisa.nc_version.gen4)
    if row_chunk:
        # The row-chunked path handles R > 128 with one collective over all rows; R <= 128 stays on the
        # default path below, byte-identical.
        return _sample_rows(
            logits,
            params,
            rank_info,
            tp_group,
            tp_size,
            return_masses,
            num_logprobs,
            skip_inactive_top_k,
            fold_topn,
            skip_unasked_logprobs,
            rows_readback,
            pad256=cc_pad256,
        )
    kernel_assert(1 <= r <= 128, f"full_vocab_sample takes 1 to 128 rows; pass row_chunk=True for R > 128, got R={r}")
    kernel_assert(logits.dtype == nl.bfloat16, "logits must be BF16")
    kernel_assert(params.dtype == nl.float32, "params must be FP32")
    kernel_assert(tuple(rank_info.shape) == (1, 4), f"rank_info must be [1, 4], got {tuple(rank_info.shape)}")
    kernel_assert(tp_size >= 1 and (tp_size == 1 or tp_group is not None), "tp_size > 1 needs tp_group")
    lp_on = num_logprobs >= 0
    n_top = num_logprobs if lp_on else 0
    if lp_on:
        kernel_assert(num_logprobs <= 20, f"num_logprobs must be 0..20, got {num_logprobs}")
        kernel_assert(v_s <= 16384, f"logprobs needs V_s <= 16384 (max8 limit), got V_s={v_s}")
    kernel_assert(not fold_topn or n_top > 0, "fold_topn needs num_logprobs > 0")
    kernel_assert(not skip_unasked_logprobs or fold_topn, "skip_unasked_logprobs needs fold_topn")
    p_cols = 5 if skip_unasked_logprobs else 4
    kernel_assert(
        tuple(params.shape) == (r, p_cols), f"params must be [R, {p_cols}] = [{r}, {p_cols}], got {tuple(params.shape)}"
    )
    v = v_s * tp_size
    c, el = _chunks(r, v_s)
    pc = r * c
    fp32, u8, i32, u16, u32, bf16 = nl.float32, nl.uint8, nl.int32, nl.uint16, nl.uint32, nl.bfloat16

    def sb(shape, dtype):
        return nl.ndarray(shape, dtype=dtype, buffer=nl.sbuf)

    # The masses all-gather destination lives for the whole kernel, so it never reuses the bytes of a freed tile.
    masses = sb((r, tp_size), fp32)
    nisa.memset(dst=masses, value=0.0)

    # ---- load inputs ----
    logits_sb = sb((r, v_s), bf16)
    nisa.dma_copy(dst=logits_sb, src=logits[0:r, 0:v_s])
    logits_f = sb((r, v_s), fp32)
    nisa.tensor_copy(dst=logits_f, src=logits_sb)
    params_sb = sb((r, p_cols), fp32)
    nisa.dma_copy(dst=params_sb, src=params[0:r, 0:p_cols])
    top_k = params_sb[0:r, 0:1]
    top_p = params_sb[0:r, 1:2]
    temp = params_sb[0:r, 2:3]
    u = sb((r, 1), fp32)
    nisa.tensor_scalar(dst=u, data=params_sb[0:r, 3:4], op0=nl.maximum, operand0=_U_MIN)
    rank_f = sb((r, 1), fp32)
    nisa.dma_copy(dst=rank_f, src=rank_info[0:1, 0:1].broadcast(0, r))
    offset = sb((r, 1), fp32)
    nisa.tensor_scalar(dst=offset, data=rank_f, op0=nl.multiply, operand0=float(v_s))

    # ---- order key of each logit: key = bits < half ? half + bits : whole - bits ----
    bits_f = sb((r, v_s), fp32)
    nisa.tensor_copy(dst=bits_f, src=logits_sb.view(u16))
    key = sb((r, v_s), fp32)
    nisa.tensor_scalar(dst=key, data=bits_f, op0=nl.subtract, operand0=_WHOLE, reverse0=True)
    keypos = sb((r, v_s), fp32)
    nisa.tensor_scalar(dst=keypos, data=bits_f, op0=nl.add, operand0=_HALF)
    poscond = sb((r, v_s), u8)
    nisa.tensor_scalar(dst=poscond, data=bits_f, op0=nl.less, operand0=_HALF)
    nisa.tensor_copy_predicated(dst=key, src=keypos, predicate=poscond)

    # ---- chunked copy of the shard: C vocab chunks of each row on partitions, for the fast search ----
    logits_ch = sb((pc, el), bf16)
    nisa.dma_copy(dst=logits_ch, src=logits.reshape((pc, el)))
    bits_ch = sb((pc, el), fp32)
    nisa.tensor_copy(dst=bits_ch, src=logits_ch.view(u16))
    key_ch = sb((pc, el), fp32)
    nisa.tensor_scalar(dst=key_ch, data=bits_ch, op0=nl.subtract, operand0=_WHOLE, reverse0=True)
    keypos_ch = sb((pc, el), fp32)
    nisa.tensor_scalar(dst=keypos_ch, data=bits_ch, op0=nl.add, operand0=_HALF)
    poscond_ch = sb((pc, el), u8)
    nisa.tensor_scalar(dst=poscond_ch, data=bits_ch, op0=nl.less, operand0=_HALF)
    nisa.tensor_copy_predicated(dst=key_ch, src=keypos_ch, predicate=poscond_ch)
    # NaN- and +inf-free chunked logits and finite mask, for the chunked exp of the top-p search.
    logits_ch_f = sb((pc, el), fp32)
    nisa.tensor_copy(dst=logits_ch_f, src=logits_ch)
    notnan_ch = sb((pc, el), fp32)
    nisa.tensor_tensor(dst=notnan_ch, data1=logits_ch_f, data2=logits_ch_f, op=nl.equal)
    nisa.scalar_tensor_tensor(
        dst=notnan_ch, data=logits_ch_f, op0=nl.less, operand0=_FINITE_MAX, op1=nl.multiply, operand1=notnan_ch
    )
    nan_elem_ch = sb((pc, el), u8)
    nisa.tensor_scalar(dst=nan_elem_ch, data=notnan_ch, op0=nl.less, operand0=0.5)
    neg_big_ch = sb((pc, el), fp32)
    nisa.memset(dst=neg_big_ch, value=-1e30)
    logits_safe_ch = sb((pc, el), fp32)
    nisa.tensor_copy(dst=logits_safe_ch, src=logits_ch_f)
    nisa.tensor_copy_predicated(dst=logits_safe_ch, src=neg_big_ch, predicate=nan_elem_ch)
    is_finite_ch = sb((pc, el), fp32)
    nisa.scalar_tensor_tensor(
        dst=is_finite_ch, data=logits_ch_f, op0=nl.greater, operand0=-1e38, op1=nl.multiply, operand1=notnan_ch
    )

    # ---- NaN or +inf per row, and a copy of the logits without them ----
    # Hardware traps (SIGFPE) on an invalid FP op such as exp(NaN) or (-inf)+(+inf); the simulator does
    # not. A +inf logit makes the max +inf, so the exp argument would be (+inf)+(-inf). NaN and +inf rows are
    # flag/token overridden below, so replace a NaN or +inf logit with a finite sentinel for the max and exp.
    # The -inf clamp of the max keeps a row of all -inf from forming (-inf)+(+inf) in the bias.
    notnan = sb((r, v_s), fp32)
    nisa.tensor_tensor(dst=notnan, data1=logits_f, data2=logits_f, op=nl.equal)
    nisa.scalar_tensor_tensor(
        dst=notnan, data=logits_f, op0=nl.less, operand0=_FINITE_MAX, op1=nl.multiply, operand1=notnan
    )
    nan_inv = sb((r, 1), fp32)
    nisa.tensor_reduce(dst=nan_inv, data=notnan, op=nl.minimum, axis=(1,))
    nan_row = sb((r, 1), fp32)
    nisa.tensor_scalar(dst=nan_row, data=nan_inv, op0=nl.subtract, operand0=1.0, reverse0=True)
    nan_elem = sb((r, v_s), u8)
    nisa.tensor_scalar(dst=nan_elem, data=notnan, op0=nl.less, operand0=0.5)
    neg_big = sb((r, v_s), fp32)
    nisa.memset(dst=neg_big, value=-1e30)
    logits_safe = sb((r, v_s), fp32)
    nisa.tensor_copy(dst=logits_safe, src=logits_f)
    nisa.tensor_copy_predicated(dst=logits_safe, src=neg_big, predicate=nan_elem)
    # Finite mask (0 for NaN, -inf and +inf): a NaN or -inf > -1e38 is false, and notnan is 0 for +inf. It
    # zeros those exp masses exactly.
    is_finite = sb((r, v_s), fp32)
    nisa.scalar_tensor_tensor(
        dst=is_finite, data=logits_f, op0=nl.greater, operand0=-1e38, op1=nl.multiply, operand1=notnan
    )

    # ---- local max and stable argmax (lowest local id among the maxima) ----
    local_max = sb((r, 1), fp32)
    nisa.tensor_reduce(dst=local_max, data=logits_safe, op=nl.maximum, axis=(1,))
    ids = sb((r, v_s), fp32)
    nisa.iota(dst=ids, pattern=[[1, v_s]])
    eqmax = sb((r, v_s), u8)
    nisa.tensor_scalar(dst=eqmax, data=logits_safe, op0=nl.equal, operand0=local_max)
    masked_id = sb((r, v_s), fp32)
    nisa.memset(dst=masked_id, value=float(v_s))
    nisa.tensor_copy_predicated(dst=masked_id, src=ids, predicate=eqmax)
    local_idx = sb((r, 1), fp32)
    nisa.tensor_reduce(dst=local_idx, data=masked_id, op=nl.minimum, axis=(1,))
    local_gid = sb((r, 1), fp32)
    nisa.tensor_scalar(dst=local_gid, data=local_idx, op0=nl.add, operand0=offset)

    # ---- global stats (one coalesced max all-reduce) ----
    # The collective max does not preserve -inf on hardware, so reduce a finite-clamped max plus a
    # per-rank "has a finite logit" flag; a row with no finite logit on any rank is the no-token row.
    stats_src = sb((r, 3), fp32)
    nisa.tensor_scalar(dst=stats_src[0:r, 0:1], data=local_max, op0=nl.maximum, operand0=-1e30)
    nisa.tensor_scalar(dst=stats_src[0:r, 1:2], data=local_max, op0=nl.greater, operand0=-1e38)
    nisa.tensor_copy(dst=stats_src[0:r, 2:3], src=nan_row)
    stats_dst = sb((r, 3), fp32)
    _allreduce(stats_src, stats_dst, nl.maximum, tp_group, tp_size, "stats", pad256=cc_pad256)
    global_max = sb((r, 1), fp32)
    nisa.tensor_copy(dst=global_max, src=stats_dst[0:r, 0:1])
    global_has_finite = stats_dst[0:r, 1:2]
    global_nan = sb((r, 1), fp32)
    nisa.tensor_copy(dst=global_nan, src=stats_dst[0:r, 2:3])
    is_top = sb((r, 1), u8)
    nisa.tensor_scalar(dst=is_top, data=stats_src[0:r, 0:1], op0=nl.equal, operand0=global_max)
    cand_id = sb((r, 1), fp32)
    nisa.memset(dst=cand_id, value=_ID_BIG)
    nisa.tensor_copy_predicated(dst=cand_id, src=local_gid, predicate=is_top)
    greedy_token = sb((r, 1), fp32)
    _allreduce(cand_id, greedy_token, nl.minimum, tp_group, tp_size, "greedy", pad256=cc_pad256)

    # ---- per-shard top-N raw logits (stage 1 of the logprobs merge) ----
    ask_trip = None
    if lp_on and n_top > 0:
        if skip_unasked_logprobs:
            # Build the top-N only on a step where a row asks; sentinels otherwise. One register says any
            # row asks; the per-shard rounds run in a 0/1-trip loop with no collective inside.
            ask_trip = _any_asks(params_sb[0:r, 4:5], r)
            tn_vals_s = sb((r, n_top), fp32)
            tn_gids_s = sb((r, n_top), fp32)
            nisa.memset(dst=tn_vals_s, value=float("-inf"))
            nisa.memset(dst=tn_gids_s, value=0.0)

            def stage1_body(_iteration):
                vals, pos = _topn_rounds(logits_safe, n_top)
                pos_f = sb((r, n_top), fp32)
                nisa.tensor_copy(dst=pos_f, src=pos)
                nisa.tensor_copy(dst=tn_vals_s, src=vals)
                nisa.tensor_scalar(dst=tn_gids_s, data=pos_f, op0=nl.add, operand0=offset)

            nl.fori_loop(0, ask_trip, stage1_body)
        else:
            tn_vals_s, tn_pos_s = _topn_rounds(logits_safe, n_top)
            tn_pos_f = sb((r, n_top), fp32)
            nisa.tensor_copy(dst=tn_pos_f, src=tn_pos_s)
            tn_gids_s = sb((r, n_top), fp32)
            nisa.tensor_scalar(dst=tn_gids_s, data=tn_pos_f, op0=nl.add, operand0=offset)

    # ---- top-k count search over the order key (chunked trials, per-row chunk reduction) ----
    # ``lower_k`` rides in both layouts: the row copy drives the digit and the floor, the chunked copy the
    # per-partition trial keys. A shared digit (0..15) updates both, so a key never crosses the group.
    lower_k = sb((r, 1), fp32)
    nisa.memset(dst=lower_k, value=0.0)
    lower_k_ch = sb((pc, 1), fp32)
    nisa.memset(dst=lower_k_ch, value=0.0)
    trials_ch = sb((pc, _SEARCH_TRIALS), fp32)
    nisa.iota(dst=trials_ch, pattern=[[1, _SEARCH_TRIALS]], offset=1)
    local_cnt = [sb((r, _SEARCH_TRIALS), fp32) for _ in range(_ROUNDS)]
    glob_cnt = [sb((r, _SEARCH_TRIALS), fp32) for _ in range(_ROUNDS)]
    cnt_partial = sb((pc, _SEARCH_TRIALS), fp32)
    ge_ch = sb((pc, el), fp32)

    def trial_counts(i):
        step = float(16 ** (_ROUNDS - 1 - i))
        trial_keys_ch = sb((pc, _SEARCH_TRIALS), fp32)
        nisa.tensor_scalar(
            dst=trial_keys_ch, data=trials_ch, op0=nl.multiply, operand0=step, op1=nl.add, operand1=lower_k_ch
        )
        for t in range(_SEARCH_TRIALS):
            nisa.tensor_scalar(
                dst=ge_ch, data=key_ch, op0=nl.greater_equal, operand0=trial_keys_ch[0:pc, t : t + 1]
            )
            nisa.tensor_reduce(dst=cnt_partial[0:pc, t : t + 1], data=ge_ch, op=nl.add, axis=(1,))
        _group_reduce(local_cnt[i], cnt_partial, r, c, nl.add, rows_readback)

    def round_counts(_iteration):
        # The loop body of round i; fori_loop calls it inside the round loop below, so i is the current round.
        trial_counts(i)

    # The runtime does not take a collective inside a runtime loop, so only the trial counts sit in the 0/1-trip
    # loop. A skipped step all-reduces zeros, and its digits are not used: every row then takes the 0 floor below.
    trip = _any_active_top_k(top_k, v, r) if skip_inactive_top_k else None
    for i in range(_ROUNDS):
        step = float(16 ** (_ROUNDS - 1 - i))
        if skip_inactive_top_k:
            nisa.memset(dst=local_cnt[i], value=0.0)
            nl.fori_loop(0, trip, round_counts)
        else:
            trial_counts(i)
        _allreduce(local_cnt[i], glob_cnt[i], nl.add, tp_group, tp_size, f"topk_{i}", pad256=cc_pad256)
        digit_mask = sb((r, _SEARCH_TRIALS), fp32)
        nisa.tensor_scalar(dst=digit_mask, data=glob_cnt[i], op0=nl.greater_equal, operand0=top_k)
        digit = sb((r, 1), fp32)
        nisa.tensor_reduce(dst=digit, data=digit_mask, op=nl.add, axis=(1,))
        nisa.tensor_scalar(dst=lower_k, data=digit, op0=nl.multiply, operand0=step, op1=nl.add, operand1=lower_k)
        digit_ch = sb((pc, 1), fp32)
        _group_bcast(digit_ch, digit, r, c)
        nisa.tensor_scalar(
            dst=lower_k_ch, data=digit_ch, op0=nl.multiply, operand0=step, op1=nl.add, operand1=lower_k_ch
        )

    # k_floor key: active top-k keeps the search result, else 0 (keep every non-NaN logit).
    active = sb((r, 1), u8)
    nisa.tensor_scalar(dst=active, data=top_k, op0=nl.greater, operand0=0.0)
    below_v = sb((r, 1), u8)
    nisa.tensor_scalar(dst=below_v, data=top_k, op0=nl.less, operand0=float(v))
    nisa.tensor_tensor(dst=active, data1=active, data2=below_v, op=nl.multiply)
    kfloor = sb((r, 1), fp32)
    nisa.memset(dst=kfloor, value=0.0)
    nisa.tensor_copy_predicated(dst=kfloor, src=lower_k, predicate=active)

    # ---- e = exp((l - max) / T), masked by the top-k floor ----
    # A greedy row (T < eps) uses temperature 1 for the softmax, as the reference does; its token is the
    # argmax, so this only sets the kept masses (not the token). Non-greedy rows use 1 / T.
    is_greedy = sb((r, 1), u8)
    nisa.tensor_scalar(dst=is_greedy, data=temp, op0=nl.less, operand0=_SAMPLING_EPS)
    temp_safe = sb((r, 1), fp32)
    nisa.tensor_copy(dst=temp_safe, src=temp)
    one_const = sb((r, 1), fp32)
    nisa.memset(dst=one_const, value=1.0)
    nisa.tensor_copy_predicated(dst=temp_safe, src=one_const, predicate=is_greedy)
    inv_t = sb((r, 1), fp32)
    nisa.reciprocal(dst=inv_t, data=temp_safe)
    max_safe = sb((r, 1), fp32)
    nisa.tensor_scalar(dst=max_safe, data=global_max, op0=nl.maximum, operand0=-1e30)
    neg_max_t = sb((r, 1), fp32)
    nisa.tensor_scalar(dst=neg_max_t, data=max_safe, op0=nl.multiply, operand0=inv_t, op1=nl.multiply, operand1=-1.0)
    # Build the exp argument explicitly and clamp its floor so exp never sees -inf (would SIGFPE). The
    # finite mask then zeros the NaN and -inf positions exactly, matching the reference's exp(-inf) = 0.
    arg = sb((r, v_s), fp32)
    nisa.tensor_scalar(dst=arg, data=logits_safe, op0=nl.multiply, operand0=inv_t, op1=nl.add, operand1=neg_max_t)
    nisa.tensor_scalar(dst=arg, data=arg, op0=nl.maximum, operand0=-80.0)
    e = sb((r, v_s), fp32)
    nisa.activation(dst=e, op=nl.exp, data=arg)
    nisa.tensor_tensor(dst=e, data1=e, data2=is_finite, op=nl.multiply)
    nisa.scalar_tensor_tensor(dst=e, data=key, op0=nl.greater_equal, operand0=kfloor, op1=nl.multiply, operand1=e)

    # ---- chunked e (same per-element values) for the fast top-p mass search ----
    inv_t_ch = sb((pc, 1), fp32)
    _group_bcast(inv_t_ch, inv_t, r, c)
    neg_max_t_ch = sb((pc, 1), fp32)
    _group_bcast(neg_max_t_ch, neg_max_t, r, c)
    kfloor_ch = sb((pc, 1), fp32)
    _group_bcast(kfloor_ch, kfloor, r, c)
    arg_ch = sb((pc, el), fp32)
    nisa.tensor_scalar(
        dst=arg_ch, data=logits_safe_ch, op0=nl.multiply, operand0=inv_t_ch, op1=nl.add, operand1=neg_max_t_ch
    )
    nisa.tensor_scalar(dst=arg_ch, data=arg_ch, op0=nl.maximum, operand0=-80.0)
    e_ch = sb((pc, el), fp32)
    nisa.activation(dst=e_ch, op=nl.exp, data=arg_ch)
    nisa.tensor_tensor(dst=e_ch, data1=e_ch, data2=is_finite_ch, op=nl.multiply)
    nisa.scalar_tensor_tensor(
        dst=e_ch, data=key_ch, op0=nl.greater_equal, operand0=kfloor_ch, op1=nl.multiply, operand1=e_ch
    )

    # ---- Z = global sum of e (shard total, reduced globally with the first top-p round) ----
    shard_total = sb((r, 1), fp32)
    nisa.tensor_reduce(dst=shard_total, data=e, op=nl.add, axis=(1,))
    target = sb((r, 1), fp32)

    # ---- raw exp sum for logz: sum over finite logits of exp(max(l - global_max, -80)) per shard ----
    # It rides as an extra column of the top-p round-0 all-reduce, so logz costs no new collective. The
    # -80 clamp keeps exp off -inf; the finite mask zeros the NaN, +inf and -inf positions exactly.
    if lp_on:
        raw_arg = sb((r, v_s), fp32)
        nisa.tensor_scalar(dst=raw_arg, data=logits_safe, op0=nl.subtract, operand0=global_max)
        nisa.tensor_scalar(dst=raw_arg, data=raw_arg, op0=nl.maximum, operand0=-80.0)
        raw_e = sb((r, v_s), fp32)
        nisa.activation(dst=raw_e, op=nl.exp, data=raw_arg)
        nisa.tensor_tensor(dst=raw_e, data1=raw_e, data2=is_finite, op=nl.multiply)
        raw_z_shard = sb((r, 1), fp32)
        nisa.tensor_reduce(dst=raw_z_shard, data=raw_e, op=nl.add, axis=(1,))

    # ---- top-p mass search over the order key (chunked trials, per-row chunk reduction) ----
    # Round 0 carries the shard total as an extra column, so Z folds into that round's all-reduce.
    lower_p = sb((r, 1), fp32)
    nisa.memset(dst=lower_p, value=0.0)
    lower_p_ch = sb((pc, 1), fp32)
    nisa.memset(dst=lower_p_ch, value=0.0)
    width0 = _SEARCH_TRIALS + 1
    raw_col = width0  # raw-exp-sum column, present only when logprobs are on
    mass_w0 = width0 + (1 if lp_on else 0)
    local_mass = [sb((r, mass_w0 if i == 0 else _SEARCH_TRIALS), fp32) for i in range(_ROUNDS)]
    glob_mass = [sb((r, mass_w0 if i == 0 else _SEARCH_TRIALS), fp32) for i in range(_ROUNDS)]
    mass_partial = sb((pc, _SEARCH_TRIALS), fp32)
    mass_ch = sb((pc, el), fp32)
    for i in range(_ROUNDS):
        step = float(16 ** (_ROUNDS - 1 - i))
        trial_keys_ch = sb((pc, _SEARCH_TRIALS), fp32)
        nisa.tensor_scalar(
            dst=trial_keys_ch, data=trials_ch, op0=nl.multiply, operand0=step, op1=nl.add, operand1=lower_p_ch
        )
        for t in range(_SEARCH_TRIALS):
            nisa.scalar_tensor_tensor(
                dst=mass_ch,
                data=key_ch,
                op0=nl.greater_equal,
                operand0=trial_keys_ch[0:pc, t : t + 1],
                op1=nl.multiply,
                operand1=e_ch,
            )
            nisa.tensor_reduce(dst=mass_partial[0:pc, t : t + 1], data=mass_ch, op=nl.add, axis=(1,))
        _group_reduce(local_mass[i][0:r, 0:_SEARCH_TRIALS], mass_partial, r, c, nl.add, rows_readback)
        if i == 0:
            nisa.tensor_copy(dst=local_mass[0][0:r, _SEARCH_TRIALS : _SEARCH_TRIALS + 1], src=shard_total)
            if lp_on:
                nisa.tensor_copy(dst=local_mass[0][0:r, raw_col : raw_col + 1], src=raw_z_shard)
        _allreduce(local_mass[i], glob_mass[i], nl.add, tp_group, tp_size, f"topp_{i}", pad256=cc_pad256)
        if i == 0:
            nisa.tensor_tensor(
                dst=target,
                data1=top_p,
                data2=glob_mass[0][0:r, _SEARCH_TRIALS : _SEARCH_TRIALS + 1],
                op=nl.multiply,
            )
        digit_mask = sb((r, _SEARCH_TRIALS), fp32)
        nisa.tensor_scalar(
            dst=digit_mask, data=glob_mass[i][0:r, 0:_SEARCH_TRIALS], op0=nl.greater_equal, operand0=target
        )
        digit = sb((r, 1), fp32)
        nisa.tensor_reduce(dst=digit, data=digit_mask, op=nl.add, axis=(1,))
        nisa.tensor_scalar(dst=lower_p, data=digit, op0=nl.multiply, operand0=step, op1=nl.add, operand1=lower_p)
        digit_ch = sb((pc, 1), fp32)
        _group_bcast(digit_ch, digit, r, c)
        nisa.tensor_scalar(
            dst=lower_p_ch, data=digit_ch, op0=nl.multiply, operand0=step, op1=nl.add, operand1=lower_p_ch
        )

    # logz = global_max + ln(Z_raw); the Z_raw floor keeps ln off 0 on a degenerate row (overridden to NaN).
    if lp_on:
        z_safe = sb((r, 1), fp32)
        nisa.tensor_scalar(dst=z_safe, data=glob_mass[0][0:r, raw_col : raw_col + 1], op0=nl.maximum, operand0=_U_MIN)
        logz_f = sb((r, 1), fp32)
        nisa.activation(dst=logz_f, op=nl.log, data=z_safe)
        nisa.tensor_tensor(dst=logz_f, data1=logz_f, data2=global_max, op=nl.add)

    # cut key: top_p >= 1 keeps everything the top-k kept (key 0 threshold).
    keep_all = sb((r, 1), u8)
    nisa.tensor_scalar(dst=keep_all, data=top_p, op0=nl.greater_equal, operand0=1.0)
    cut = sb((r, 1), fp32)
    nisa.tensor_copy(dst=cut, src=lower_p)
    zero_cut = sb((r, 1), fp32)
    nisa.memset(dst=zero_cut, value=0.0)
    nisa.tensor_copy_predicated(dst=cut, src=zero_cut, predicate=keep_all)

    # ---- kept masses and per-rank mass ----
    kept = sb((r, v_s), fp32)
    nisa.scalar_tensor_tensor(dst=kept, data=key, op0=nl.greater_equal, operand0=cut, op1=nl.multiply, operand1=e)
    # The shard mass is the total of the draw's walk over these kept masses (see _walk_total).
    shard_kept = _walk_total(kept)
    fused = verify and verify_fused
    if fused:
        # ``verify_fused``: the candidate masses and the kept masses without the candidates first; then one gather of
        # [shard_kept | shard_kept_e] for each rank (rank k at columns 2 k, 2 k + 1).
        mass_at, kept_e, shard_kept_e = _verify_fused_prepare(
            candidate_ids, kept, offset, ids, tp_group, tp_size, pad256=cc_pad256
        )
        send = sb((r, 2), fp32)
        nisa.tensor_copy(dst=send[0:r, 0:1], src=shard_kept, engine=nisa.engine.vector)
        nisa.tensor_copy(dst=send[0:r, 1:2], src=shard_kept_e, engine=nisa.engine.vector)
        both = sb((r, 2 * tp_size), fp32)
        _allgather(send, both, tp_group, tp_size, "masses")
        pairs = both.reshape((r, tp_size, 2))
        nisa.tensor_copy(dst=masses, src=pairs[0:r, 0:tp_size, 0], engine=nisa.engine.vector)
        masses_e = sb((r, tp_size), fp32)
        nisa.tensor_copy(dst=masses_e, src=pairs[0:r, 0:tp_size, 1], engine=nisa.engine.vector)
        gather_src = send[0:r, 0:1] if tp_size > 1 else None
    elif fold_topn:
        # The top-N (value, id) pairs ride the masses all-gather, so logprobs add no collective.
        topn_cvals, topn_cids, gather_src = _fold_gather(
            shard_kept, tn_vals_s, tn_gids_s, masses, n_top, tp_group, tp_size
        )
    else:
        _allgather(shard_kept, masses, tp_group, tp_size, "masses")
        gather_src = shard_kept if tp_size > 1 else None

    # ---- inverse-CDF draw over the kept masses in global id order ----
    owner, local_token = _draw(masses, kept, u, rank_f, ids, tp_size)

    is_owner = sb((r, 1), u8)
    nisa.tensor_scalar(dst=is_owner, data=owner, op0=nl.equal, operand0=rank_f)
    if lp_on:
        # The draw all-reduce also carries the owner's drawn raw logit, so sampled_logit costs no new
        # collective. The owner gathers its drawn token's logit at the local draw index.
        draw_idx = sb((r, 1), fp32)
        nisa.tensor_tensor(dst=draw_idx, data1=local_token, data2=offset, op=nl.subtract)
        drawn_logit_bf = sb((r, 1), bf16)
        _gather(drawn_logit_bf, logits_sb, draw_idx)
        drawn_logit = sb((r, 1), fp32)
        nisa.tensor_copy(dst=drawn_logit, src=drawn_logit_bf)
        result = sb((r, 2), fp32)
        nisa.memset(dst=result, value=-1.0)
        nisa.memset(dst=result[0:r, 1:2], value=-3e38)
        nisa.tensor_copy_predicated(dst=result[0:r, 0:1], src=local_token, predicate=is_owner)
        nisa.tensor_copy_predicated(dst=result[0:r, 1:2], src=drawn_logit, predicate=is_owner)
        sampled_pair = sb((r, 2), fp32)
        _allreduce(result, sampled_pair, nl.maximum, tp_group, tp_size, "draw", pad256=cc_pad256)
        sampled_token = sb((r, 1), fp32)
        nisa.tensor_copy(dst=sampled_token, src=sampled_pair[0:r, 0:1])
        sampled_logit_cand = sampled_pair[0:r, 1:2]
    elif fused:
        # ``verify_fused``: the excluded draw on the excluded masses, and one max all-reduce of both draws.
        u_e = sb((r, 1), fp32)
        nisa.dma_copy(dst=u_e, src=excl_u[0:r, 0:1])
        nisa.tensor_scalar(dst=u_e, data=u_e, op0=nl.maximum, operand0=_U_MIN, engine=nisa.engine.vector)
        owner_e, local_token_e = _draw(masses_e, kept_e, u_e, rank_f, ids, tp_size)
        is_owner_e = sb((r, 1), u8)
        nisa.tensor_scalar(dst=is_owner_e, data=owner_e, op0=nl.equal, operand0=rank_f, engine=nisa.engine.vector)
        result = sb((r, 2), fp32)
        nisa.memset(dst=result, value=-1.0)
        nisa.tensor_copy_predicated(dst=result[0:r, 0:1], src=local_token, predicate=is_owner)
        nisa.tensor_copy_predicated(dst=result[0:r, 1:2], src=local_token_e, predicate=is_owner_e)
        drawn_pair = sb((r, 2), fp32)
        _allreduce(result, drawn_pair, nl.maximum, tp_group, tp_size, "draw", pad256=cc_pad256)
        sampled_token = sb((r, 1), fp32)
        nisa.tensor_copy(dst=sampled_token, src=drawn_pair[0:r, 0:1], engine=nisa.engine.vector)
        drawn_e = drawn_pair[0:r, 1:2]
    else:
        result = sb((r, 1), fp32)
        nisa.memset(dst=result, value=-1.0)
        nisa.tensor_copy_predicated(dst=result, src=local_token, predicate=is_owner)
        sampled_token = sb((r, 1), fp32)
        _allreduce(result, sampled_token, nl.maximum, tp_group, tp_size, "draw", pad256=cc_pad256)

    # ---- combine greedy / sampled, apply NaN and -inf overrides, clamp ----
    token_f = sb((r, 1), fp32)
    _hold_source(token_f, sampled_token, gather_src)
    nisa.tensor_copy_predicated(dst=token_f, src=greedy_token, predicate=is_greedy)
    # A NaN or +inf row: greedy -> last id, sampled -> 0.
    nan_tok = sb((r, 1), fp32)
    nisa.memset(dst=nan_tok, value=0.0)
    last_const = sb((r, 1), fp32)
    nisa.memset(dst=last_const, value=float(v - 1))
    nisa.tensor_copy_predicated(dst=nan_tok, src=last_const, predicate=is_greedy)
    nan_pred = sb((r, 1), u8)
    nisa.tensor_scalar(dst=nan_pred, data=global_nan, op0=nl.greater, operand0=0.0)
    nisa.tensor_copy_predicated(dst=token_f, src=nan_tok, predicate=nan_pred)
    nisa.tensor_scalar(dst=token_f, data=token_f, op0=nl.minimum, operand0=float(v - 1), op1=nl.maximum, operand1=0.0)
    tokens_sb = sb((r, 1), i32)
    nisa.tensor_copy(dst=tokens_sb, src=token_f)

    # ---- flags: 1 = NaN or +inf logit, 2 = all -inf without NaN or +inf ----
    is_ninf = sb((r, 1), fp32)
    nisa.tensor_scalar(dst=is_ninf, data=global_has_finite, op0=nl.less, operand0=0.5)
    not_nan = sb((r, 1), fp32)
    nisa.tensor_scalar(dst=not_nan, data=global_nan, op0=nl.multiply, operand0=-1.0, op1=nl.add, operand1=1.0)
    notok = sb((r, 1), fp32)
    nisa.tensor_tensor(dst=notok, data1=not_nan, data2=is_ninf, op=nl.multiply)
    flag_f = sb((r, 1), fp32)
    nisa.tensor_scalar(dst=flag_f, data=notok, op0=nl.multiply, operand0=2.0, op1=nl.add, operand1=global_nan)
    flags_sb = sb((r, 1), i32)
    nisa.tensor_copy(dst=flags_sb, src=flag_f)

    tokens = nl.ndarray((r, 1), dtype=i32, buffer=nl.shared_hbm)
    flags = nl.ndarray((r, 1), dtype=i32, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=tokens, src=tokens_sb)
    nisa.dma_copy(dst=flags, src=flags_sb)
    if verify:
        if fused:
            probs_at, excl_token = _verify_fused_finish(
                mass_at, masses, masses_e, drawn_e, send[0:r, 1:2] if tp_size > 1 else None, tp_size, v_s
            )
        else:
            probs_at, excl_token = _verify_candidates(
                candidate_ids, excl_u, kept, masses, offset, rank_f, ids, tp_group, tp_size, pad256=cc_pad256
            )
        if not return_masses:
            return tokens, flags, probs_at, excl_token
        # The check form: the kept masses that the draws and probs_at read, before the verify outputs.
        kept_out = nl.ndarray((r, v_s), dtype=fp32, buffer=nl.shared_hbm)
        masses_out = nl.ndarray((r, tp_size), dtype=fp32, buffer=nl.shared_hbm)
        nisa.dma_copy(dst=kept_out, src=kept)
        nisa.dma_copy(dst=masses_out, src=masses)
        return tokens, flags, kept_out, masses_out, probs_at, excl_token
    if not return_masses and not lp_on:
        return tokens, flags
    outs = [tokens, flags]
    if return_masses:
        kept_out = nl.ndarray((r, v_s), dtype=fp32, buffer=nl.shared_hbm)
        masses_out = nl.ndarray((r, tp_size), dtype=fp32, buffer=nl.shared_hbm)
        nisa.dma_copy(dst=kept_out, src=kept)
        nisa.dma_copy(dst=masses_out, src=masses)
        outs += [kept_out, masses_out]
    if not lp_on:
        return tuple(outs)

    # ---- logprobs outputs (raw log-softmax: top-N ids/logits, logz, sampled logit) ----
    nan_c = sb((r, 1), fp32)
    nisa.memset(dst=nan_c, value=float("nan"))
    flagged = sb((r, 1), u8)
    nisa.tensor_scalar(dst=flagged, data=flag_f, op0=nl.greater, operand0=0.0)
    nisa.tensor_copy_predicated(dst=logz_f, src=nan_c, predicate=flagged)
    sampled_logit = sb((r, 1), fp32)
    nisa.tensor_copy(dst=sampled_logit, src=sampled_logit_cand)
    nisa.tensor_copy_predicated(dst=sampled_logit, src=global_max, predicate=is_greedy)
    nisa.tensor_copy_predicated(dst=sampled_logit, src=nan_c, predicate=flagged)
    if n_top > 0:
        # Flush rule: slots past the finite logits (a search sentinel, or -inf padding) carry no real logit;
        # upstream returns -inf there. Flush anything below the real logit range to -inf; ids stay as they are.
        if fold_topn:
            topn_vals = sb((r, n_top), fp32)
            topn_gids_f = sb((r, n_top), fp32)

            def merge_body(_iteration):
                if tp_size == 1:
                    gvals, gids = tn_vals_s, tn_gids_s
                else:
                    gvals, gids = _merge_rounds(topn_cvals, topn_cids, n_top)
                nisa.tensor_copy(dst=topn_vals, src=gvals)
                nisa.tensor_copy(dst=topn_gids_f, src=gids)
                ninf_c = sb((r, n_top), fp32)
                nisa.memset(dst=ninf_c, value=float("-inf"))
                past = sb((r, n_top), u8)
                nisa.tensor_scalar(dst=past, data=topn_vals, op0=nl.less, operand0=-1e20)
                nisa.tensor_copy_predicated(dst=topn_vals, src=ninf_c, predicate=past)

            if skip_unasked_logprobs:
                # Sentinels for a no-ask step; full write before the 0/1-trip loop (NCC_IGCA108).
                nisa.memset(dst=topn_vals, value=float("-inf"))
                nisa.memset(dst=topn_gids_f, value=0.0)
                nl.fori_loop(0, ask_trip, merge_body)
            else:
                merge_body(0)
        else:
            topn_vals, topn_gids_f = _merge_topn(tn_vals_s, tn_gids_s, n_top, tp_group, tp_size)
            ninf_c = sb((r, n_top), fp32)
            nisa.memset(dst=ninf_c, value=float("-inf"))
            past = sb((r, n_top), u8)
            nisa.tensor_scalar(dst=past, data=topn_vals, op0=nl.less, operand0=-1e20)
            nisa.tensor_copy_predicated(dst=topn_vals, src=ninf_c, predicate=past)
        # Every output id is in the vocab (a sentinel slot holds an in-range id too).
        nisa.tensor_scalar(
            dst=topn_gids_f,
            data=topn_gids_f,
            op0=nl.minimum,
            operand0=float(v - 1),
            op1=nl.maximum,
            operand1=0.0,
            engine=nisa.engine.vector,
        )
        topn_ids_sb = sb((r, n_top), i32)
        nisa.tensor_copy(dst=topn_ids_sb, src=topn_gids_f)
        topn_ids = nl.ndarray((r, n_top), dtype=i32, buffer=nl.shared_hbm)
        topn_logits = nl.ndarray((r, n_top), dtype=fp32, buffer=nl.shared_hbm)
        nisa.dma_copy(dst=topn_ids, src=topn_ids_sb)
        nisa.dma_copy(dst=topn_logits, src=topn_vals)
        outs += [topn_ids, topn_logits]
    logz = nl.ndarray((r, 1), dtype=fp32, buffer=nl.shared_hbm)
    sampled_logit_out = nl.ndarray((r, 1), dtype=fp32, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=logz, src=logz_f)
    nisa.dma_copy(dst=sampled_logit_out, src=sampled_logit)
    outs += [logz, sampled_logit_out]
    return tuple(outs)
