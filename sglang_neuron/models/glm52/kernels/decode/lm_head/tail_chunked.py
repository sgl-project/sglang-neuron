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

"""GLM-5.2 fork of the GPT-OSS decode sampling tail: RMSNorm -> lm_head -> all_to_all -> chunked top-k -> sampler, one
collective.

Forked from the kernel library's GPT-OSS decode tail so the GLM stack can carry what it needs without
touching the GPT-OSS kernel:

* explicit PSUM accumulate flags on every ``nc_matmul`` (here and in the forked ``chunked_topk``): the compiler
  refuses a NEFF that mixes explicit and inferred flags (``NCC_ILMM003``);
* one PSUM bank per GEMM accumulator: an accumulator spanning banks left the second 128-row tile's logits wrong at
  ``V_s > 512`` with more than one row tile;
* ``norm_rounding="bf16"``: HF's ``RMSNorm`` rounding, ``bf16(bf16(x * rms) * gamma)``;
* ``fold_deinterleave``: lets the caller force the explicit last-hop de-interleave;
* ``top_p_rule="keep_crossing"``: top-p keeps the candidate whose cumulative probability crosses ``top_p``, as HF
  does.

The design is the parent's. The vocab is column-parallel across ``VTP`` ranks; one all-to-all brings each rank the full
vocab for its own rows, and ``chunked_topk`` ranks it locally, so no index correction or packed value/index exchange
is needed: over a rank-ordered group, a column index in the assembled row is the global vocab id. On trn3 an
all-to-all group must sit within one device or hold one core per device, so the exchange runs as hops over such
groups (``hop_meshes``), and row ownership becomes a closed-form mixed-radix map (``owned_row_block``).

Caller contracts: ``lm_head_weight_t`` is ``[H, V_s]`` bf16, already transposed and contiguous; ``replica_groups``
lists all ``VTP`` ranks in rank order (a permuted group returns wrong vocab ids); ``sampling_params`` holds this
rank's rows only. LNC=1: the stages after the collective are not program-sharded, which ``n_prgs == 1`` asserts.
"""

from typing import Optional, Tuple

import nki
import nki.collectives as ncc
import nki.isa as nisa
import nki.language as nl

from .tail_helpers import get_verified_program_sharding_info
from ..utils.kernel_assert import kernel_assert
from ..utils.kernel_helpers import div_ceil
from ..utils.scratch_hbm import scratch_hbm
from .chunked_topk import chunked_topk, create_chunked_topk_config, interleaved_fold_supported


_PMAX = nl.tile_size.pmax  # 128


def compose_hops(vtp: int, hops: Tuple[int, ...]):
    """Replica-group meshes for a sequence of all-to-all hops whose sizes multiply to ``vtp``.

    Hop ``h`` groups ranks that agree on every LATER hop's coordinate, so with ``hops=(8, 16)``
    and ``r = q*8 + s``: hop 0 groups the 8 CONTIGUOUS ranks sharing ``q``, and hop 1 groups
    the 16 ranks sharing ``s``, strided by 8. Listing each group in ascending rank order is
    what keeps a column index equal to a global vocab id.

    Returns a tuple of meshes, one per hop, each a tuple of rank-tuples.
    """
    prod = 1
    for h in hops:
        prod *= h
    if prod != vtp:
        raise ValueError(f"hop sizes {hops} multiply to {prod}, not vtp={vtp}")
    stride = 1
    meshes = []
    for h in hops:
        groups = []
        for base in range(vtp):
            # One group per distinct (everything except this hop's coordinate).
            if (base // stride) % h != 0:
                continue
            groups.append(tuple(base + i * stride for i in range(h)))
        meshes.append(tuple(sorted(groups)))
        stride *= h
    return tuple(meshes)


def owned_row_block(rank: int, vtp: int, gbs: int, hops=None):
    """Batch rows that ``rank`` owns after the all-to-all. Returns ``(row_start, n_rows)``.

    THE caller-facing half of the kernel's row-ownership contract, living next to
    ``compose_hops`` because the two must agree: a production caller de-permutes this
    kernel's per-rank outputs with exactly this map. With a single hop it is the IDENTITY
    -- rank ``r`` owns ``[r*gbs/vtp, (r+1)*gbs/vtp)`` -- which is the point of routing the
    collective over a rank-ordered group. A multi-hop decomposition keeps, per hop, only
    the sub-block selected by this rank's coordinate within that hop's group, so the
    offsets compose as a mixed-radix sum. Kept as one named function so the tests can
    assert the map tiles the batch exactly once and a change to hop structure has exactly
    one place to break.

    ``hops=None`` means a single hop over all ``vtp`` ranks, which the general formula
    reproduces exactly (``start = rank * gbs/vtp``); it is not a separate code path.
    """
    if gbs % vtp:
        raise ValueError(f"gbs ({gbs}) must be divisible by VTP ({vtp})")
    rows = gbs // vtp
    hop_sizes = (vtp,) if hops is None else tuple(hops)
    prod = 1
    for h in hop_sizes:
        prod *= h
    if prod != vtp:
        raise ValueError(f"hop sizes {hop_sizes} multiply to {prod}, not vtp={vtp}")
    span, stride, start = gbs, 1, 0
    for h in hop_sizes:
        span //= h
        start += ((rank // stride) % h) * span
        stride *= h
    return start, rows


# The mxfp4 hidden shuffle is defined over 4 interleaved groups, so the padding boundary is
# expressed in a [4, H/4] view. See THE mxfp4 VARIANCE CONVENTION above.
_MXFP4_SHUFFLE_GROUPS = 4

# Sampler constants. The below-threshold sentinel is FINITE by design: -inf would make a
# fully-masked row's manual softmax produce NaN.
_TOPK_MASK_SENTINEL = -3000.0
# temperature < eps => treat the row as greedy.
_SAMPLING_EPS = 1e-5
# The inverse-transform draw when ``sampling_params`` carries no 4th column: COMPILED IN, so
# the 3-column NEFF stays deterministic by construction. A 4-column ``sampling_params``
# supplies the draw per row at RUNTIME instead (see _sampler_stage), which is what lets a
# test sweep u as data without rebuilding the NEFF.
_DETERMINISTIC_DRAW = 0.5
# How top-p treats the candidate whose cumulative probability crosses ``top_p`` (see _sampler_stage).
# ``drop_crossing`` drops it (the default); ``keep_crossing`` keeps it, which is the
# HF rule. A trace-time choice: the two compile to different compare instructions.
TOP_P_RULES = ("drop_crossing", "keep_crossing")


def _const_tile(shape, value: float):
    """Materialize a constant SBUF tile.

    ``nisa.tensor_copy_predicated`` validates its ``src`` as a partitioned tensor and
    rejects a Python scalar despite the type hint.
    """
    tile = nl.ndarray(shape, dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=tile, value=value)
    return tile


def _make_scan_triangle():
    """``[128, 128]`` fp32 upper-triangular ones, ``T[p, n] = 1 iff p <= n``.

    The stationary-side constant of the Tensor Engine prefix sum in ``_inclusive_scan_pe``:
    contracting data against it turns "sum over partitions" into "sum over partitions up to
    n", which IS the inclusive prefix. Built from two iotas and one compare, with no data
    dependencies, so the scheduler hoists it off every critical path.
    """
    part = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(part, pattern=[[1, 1]], offset=0, channel_multiplier=1)
    cols = nl.ndarray((_PMAX, _PMAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(cols, pattern=[[1, _PMAX]], offset=0, channel_multiplier=0)
    tri = nl.ndarray((_PMAX, _PMAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=tri, data=cols, op0=nl.greater_equal, operand0=part)
    return tri


def _inclusive_scan_pe(dst, src, n_rows: int, width: int, tri) -> None:
    """Inclusive prefix sum along the free dim on the TENSOR Engine: ``[rows, w] -> [rows, w]``.

    Per 128-wide chunk: transpose the chunk so its slice of the row sits on partitions, then
    contract against the upper-triangular ones -- ``out[r, n] = sum_{p <= n} chunk_t[p, r]`` --
    and read the PSUM back with the previous chunks' running total fused into the copy as a
    per-partition scalar add (the last column of the previous chunk's output, which already
    carries every earlier chunk).

    This replaces a Hillis-Steele scan whose ceil(log2(w)) shifted adds were the sampler's
    single largest block of Vector work, and whose every step was serial ON VECTOR. Here
    Vector pays two 128-wide copies per chunk and the contraction runs on the PE, which is
    otherwise idle through the whole sampler phase. Each output element is ONE fp32 PE
    contraction of its prefix rather than a tree of partial adds -- a different (not worse)
    fp32 summation order than either the tree or a sequential cumsum, the same class of
    last-ulp wiggle the gates' tie-guards already absorb.
    """
    for c in range(div_ceil(width, _PMAX)):
        c0 = c * _PMAX
        c_len = min(_PMAX, width - c0)
        chunk_ps = nl.ndarray((_PMAX, _PMAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(
            dst=chunk_ps[0:c_len, 0:n_rows], data=src[0:n_rows, nl.ds(c0, c_len)], engine=nisa.tensor_engine
        )
        chunk_t = nl.ndarray((_PMAX, _PMAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=chunk_t[0:c_len, 0:n_rows], src=chunk_ps[0:c_len, 0:n_rows])
        pre_ps = nl.ndarray((_PMAX, _PMAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(
            dst=pre_ps[0:n_rows, 0:c_len],
            stationary=chunk_t[0:c_len, 0:n_rows],
            moving=tri[0:c_len, 0:c_len],
            accumulate=False,
        )
        if c == 0:
            nisa.tensor_copy(dst=dst[0:n_rows, nl.ds(c0, c_len)], src=pre_ps[0:n_rows, 0:c_len])
        else:
            # The carry rides the PSUM read: previous chunk's last output column is the total
            # of everything before this chunk, and it lives on the same partition as its row.
            nisa.tensor_scalar(
                dst=dst[0:n_rows, nl.ds(c0, c_len)],
                data=pre_ps[0:n_rows, 0:c_len],
                op0=nl.add,
                operand0=dst[0:n_rows, nl.ds(c0 - 1, 1)],
            )


def _skew_fence(hop_meshes, hidden):
    """All-reduce a ZERO on every hop mesh, and return it for the norm to consume.

    PROFILING INSTRUMENT, not computation. Without it the FIRST collective absorbs all the
    rank-arrival skew and every LATER collective absorbs whatever skew re-accumulates while the
    ranks run compute -- so a per-hop duration reads as transfer cost when it is mostly dispatch
    jitter. Fencing each mesh separately moves that charge onto the fence.

    It cannot change the answer: the reduced value is 0.0 and the caller adds it into the
    hidden states, and ``x + 0.0`` is bit-identical for every finite fp32 x. The result MUST
    stay consumed by the norm -- that data dependency is the only thing stopping the scheduler
    from sinking the fence past the real work and making it useless.
    """
    acc = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.shared_hbm)
    zero_sb = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=zero_sb, value=0.0)
    nisa.dma_copy(dst=acc, src=zero_sb)
    for hop in range(len(hop_meshes)):
        src = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.shared_hbm)
        nisa.dma_copy(dst=src, src=zero_sb)
        dst = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.shared_hbm)
        ncc.all_reduce(srcs=[src], dsts=[dst], replica_group=ncc.ReplicaGroup(hop_meshes[hop]), op=nl.add)
        got = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=got, src=dst)
        prev = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=prev, src=acc)
        nisa.tensor_tensor(dst=prev, data1=prev, data2=got, op=nl.add)
        nisa.dma_copy(dst=acc, src=prev)
    return acc


def _load_gamma_broadcast(gamma_hbm, hidden, hidden_actual, zero_interleaved_pad):
    """gamma ``[1, H]`` -> ``[_PMAX, H]`` SBUF broadcast, with the padding lanes zeroed.

    gamma is ``[1, H]`` and every row needs it, but a PARTITION-dim broadcast of an on-chip
    tensor is illegal -- do it at the DMA boundary with an HBM-side ``.broadcast`` view. Loaded
    ONCE per kernel, not once per 128-row tile: gamma never changes between tiles, and the
    broadcast moves ``_PMAX * H`` fp32 (1.5 MB at the production shape) per load.

    ZEROING GAMMA'S PADDING REPLACES RE-ZEROING THE NORMED OUTPUT. The input's padding lanes
    are exactly zero by contract -- the variance formula divides by ``H_u`` on exactly that
    assumption (see THE mxfp4 VARIANCE CONVENTION) -- so the normed padding is
    ``(0 * rms) * gamma_pad``: zero for any finite ``gamma_pad``, and NaN for a non-finite one
    only when the variance (and therefore every normed lane) is already poisoned anyway. Doing
    it on gamma keeps the memsets off the per-tile critical path, where a per-tile form re-runs
    them per tile AFTER the full-width gamma multiply, and lets the whole normed row feed the
    GEMM the moment it is scaled. The mxfp4 padding is INTERLEAVED, so kill columns
    ``>= H_u/4`` of each of 4 groups.
    """
    gamma_sb = nl.ndarray((_PMAX, hidden), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=gamma_sb, src=gamma_hbm[0:1, :].broadcast(0, _PMAX))
    if hidden_actual < hidden:
        if zero_interleaved_pad:
            gw = hidden // _MXFP4_SHUFFLE_GROUPS
            keep = hidden_actual // _MXFP4_SHUFFLE_GROUPS
            for g in range(_MXFP4_SHUFFLE_GROUPS):
                if gw - keep > 0:
                    nisa.memset(dst=gamma_sb[:, nl.ds(g * gw + keep, gw - keep)], value=0)
        else:
            nisa.memset(dst=gamma_sb[:, nl.ds(hidden_actual, hidden - hidden_actual)], value=0)
    return gamma_sb


# DMA granule for the row tile's hidden dim in _norm_lm_head_stage. One monolithic [rows, H]
# load serialises the whole norm behind the full transfer (engine idle time at the
# production shape); block loads let the sum of squares run behind each block while the next
# is still in flight, so "rms ready" trails the LAST block's arrival by one activation
# instead of trailing the whole row by three full-width ops.
_NORM_DMA_BLOCK = 6 * _PMAX


def _load_weight_resident(wt_hbm, hidden, vocab_shard):
    """Load the whole ``[H, V_s]`` bf16 lm_head shard into SBUF ONCE, as per-chunk tiles.

    Streamed, the GEMM re-reads the shard from HBM once per pair of 128-row tiles
    ((((((2x/4x the 9.65 MB shard at gbs 512/1024). In isolation that costs no wall clock -- the
    head is PE-bound and the stream keeps pace -- but in INTEGRATION the repeated stream is
    contended HBM bandwidth that cannot be prefetched behind preceding work, whereas one
    resident copy (24 x 393 KB = 9.4 MB of SBUF, ~40%, live only through the head) loads
    once, entirely prefetchable before the first matmul needs chunk 0. Static descriptors
    for the same reason as the input blocks above.
    """
    wt_res = []
    for c in range(div_ceil(hidden, _PMAX)):
        h0 = c * _PMAX
        h_len = min(_PMAX, hidden - h0)
        wt_c = nl.ndarray((_PMAX, vocab_shard), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=wt_c[0:h_len, :], src=wt_hbm[nl.ds(h0, h_len), :], dge_mode=nisa.dge_mode.none)
        wt_res.append(wt_c)
    return wt_res


def _norm_lm_head_stage(
    x_hbm,
    gamma_sb,
    wt_res,
    eps,
    n_rows,
    row_start,
    hidden,
    hidden_actual,
    vocab_shard,
    fence,
    deferred_rms,
    norm_rounding="fp32",
):
    """RMSNorm and the lm_head GEMM for one 128-row tile, pipelined by column block and chunk.

    The row loads in ``_NORM_DMA_BLOCK`` blocks while the Scalar engine accumulates ``sum(x^2)`` block by block, the
    ``rsqrt`` is one activation, and gamma is applied per 128-column chunk right before that chunk's transpose and
    matmul, so no phase waits for the whole row. The weight arrives transposed (``[H, V_s]`` bf16), making each chunk's
    read one contiguous run.

    With ``deferred_rms`` (the default) the GEMM computes ``(x * gamma) @ W`` and the caller applies the per-row
    ``rms`` on its PSUM reads, which is exact (rms is a per-row scalar) and keeps the matmuls from waiting on the row's
    sum of squares; ``deferred_rms=False`` applies it in the chain. ``norm_rounding="bf16"`` gives HF's
    ``bf16(bf16(x * rms) * gamma)`` (GLM's ``GlmMoeDsaRMSNorm``) and implies the in-chain rms.

    Returns ``(acc_sb, rms)``: the tile's fp32 logits in SBUF, unnormalized under ``deferred_rms``, and the per-row rms.
    """
    # ---- load + sum of squares, block by block ----
    # The block DMAs go out before the fence broadcast: a queue runs its descriptors in order, so a fence-dependent load
    # ahead of them would hold the whole input behind the skew-absorbing all-reduce (idle time in a fenced profile).
    # The fence rides two +0.0 operands, the squares' bias (gating rms) and each GEMM chunk's add (gating the matmuls,
    # which deferred rms does not order behind rms). A 16-bit input is widened to the fp32 xt by an explicit per-block
    # copy, which is exact; feeding 16-bit data straight into the square and gamma chains computes at the input's
    # precision and gave different outputs on hardware.
    widen = x_hbm.dtype != nl.float32
    x_raw = nl.ndarray((_PMAX, hidden), dtype=x_hbm.dtype, buffer=nl.sbuf) if widen else None
    xt = nl.ndarray((_PMAX, hidden), dtype=nl.float32, buffer=nl.sbuf)
    ssq = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    sq_scratch = nl.ndarray((_PMAX, _NORM_DMA_BLOCK), dtype=nl.float32, buffer=nl.sbuf)
    n_blk = div_ceil(hidden, _NORM_DMA_BLOCK)
    for b in range(n_blk):
        b0 = b * _NORM_DMA_BLOCK
        b_len = min(_NORM_DMA_BLOCK, hidden - b0)
        # Static descriptors (dge_mode.none): generated at compile time, so the load neither
        # waits on a DGE engine nor -- in fenced profiling NEFFs -- queues behind the skew
        # fence's trigger, which lives on the GpSimd DGE queue.
        nisa.dma_copy(
            dst=(x_raw if widen else xt)[0:n_rows, nl.ds(b0, b_len)],
            src=x_hbm[row_start : row_start + n_rows, nl.ds(b0, b_len)],
            dge_mode=nisa.dge_mode.none,
        )
    if widen:
        for b in range(n_blk):
            b0 = b * _NORM_DMA_BLOCK
            b_len = min(_NORM_DMA_BLOCK, hidden - b0)
            nisa.tensor_copy(dst=xt[0:n_rows, nl.ds(b0, b_len)], src=x_raw[0:n_rows, nl.ds(b0, b_len)])
    f_sb = None
    if fence is not None:
        # Broadcast on the partition dim at the DMA boundary, as with gamma.
        f_sb = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=f_sb, src=fence[0:1, 0:1].broadcast(0, _PMAX))
    for b in range(n_blk):
        b0 = b * _NORM_DMA_BLOCK
        b_len = min(_NORM_DMA_BLOCK, hidden - b0)
        nisa.activation(
            dst=sq_scratch[0:n_rows, 0:b_len],
            op=nl.square,
            data=xt[0:n_rows, nl.ds(b0, b_len)],
            bias=f_sb[0:n_rows, :] if f_sb is not None else None,
            reduce_op=nl.add,
            reduce_res=ssq[0:n_rows, :] if b == n_blk - 1 else None,
            reduce_cmd=nisa.reduce_cmd.reset_reduce if b == 0 else nisa.reduce_cmd.reduce,
        )
    # Divide by H_u, NOT H: the mxfp4 convention (padding contributes exactly zero to ssq).
    # DEFERRED: nothing below consumes rms -- it runs concurrently with the GEMM on the
    # Scalar Engine and the CALLER multiplies it onto the PSUM-read copies.
    rms = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(
        dst=rms[0:n_rows, :], op=nl.rsqrt, data=ssq[0:n_rows, :], scale=1.0 / float(hidden_actual), bias=float(eps)
    )

    # ---- per-128-column chunk: scale -> transpose -> bf16 ----
    # Deferred: (x + fence0) * gamma -- chunk c depends only on its OWN input block (plus the
    # fence), so the first transpose can start one block-DMA after the fence instead of after
    # the whole row's ssq. deferred_rms=False: (x * rms) * gamma, the rms edge
    # carrying the fence into the matmuls.
    xt_chunks = []
    for c in range(div_ceil(hidden, _PMAX)):
        h0 = c * _PMAX
        h_len = min(_PMAX, hidden - h0)
        if norm_rounding == "bf16":
            # HF RMSNorm realization, two bf16 roundings: ``normed = bf16(x * rms)`` (the
            # ``.to(input_dtype)`` cast), then ``bf16(normed * gamma)`` (a bf16 x bf16 multiply,
            # computed in fp32 and rounded). Both products are rounded by the engine's bf16 store,
            # so this is the reference's arithmetic to the ulp wherever ``rms`` agrees.
            normed_bf = nl.ndarray((_PMAX, _PMAX), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_scalar(
                dst=normed_bf[0:n_rows, 0:h_len],
                data=xt[0:n_rows, nl.ds(h0, h_len)],
                op0=nl.multiply,
                operand0=rms[0:n_rows, :],
            )
            scaled_bf = nl.ndarray((_PMAX, _PMAX), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_tensor(
                dst=scaled_bf[0:n_rows, 0:h_len],
                data1=normed_bf[0:n_rows, 0:h_len],
                data2=gamma_sb[0:n_rows, nl.ds(h0, h_len)],
                op=nl.multiply,
            )
            xt_ps = nl.ndarray((_PMAX, _PMAX), dtype=nl.bfloat16, buffer=nl.psum)
            nisa.nc_transpose(
                dst=xt_ps[0:h_len, 0:n_rows], data=scaled_bf[0:n_rows, 0:h_len], engine=nisa.tensor_engine
            )
        else:
            scaled_c = nl.ndarray((_PMAX, _PMAX), dtype=nl.float32, buffer=nl.sbuf)
            if deferred_rms:
                nisa.scalar_tensor_tensor(
                    dst=scaled_c[0:n_rows, 0:h_len],
                    data=xt[0:n_rows, nl.ds(h0, h_len)],
                    op0=nl.add,
                    operand0=f_sb[0:n_rows, :] if f_sb is not None else 0.0,
                    op1=nl.multiply,
                    operand1=gamma_sb[0:n_rows, nl.ds(h0, h_len)],
                )
            else:
                nisa.scalar_tensor_tensor(
                    dst=scaled_c[0:n_rows, 0:h_len],
                    data=xt[0:n_rows, nl.ds(h0, h_len)],
                    op0=nl.multiply,
                    operand0=rms[0:n_rows, :],
                    op1=nl.multiply,
                    operand1=gamma_sb[0:n_rows, nl.ds(h0, h_len)],
                )
            xt_ps = nl.ndarray((_PMAX, _PMAX), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_transpose(dst=xt_ps[0:h_len, 0:n_rows], data=scaled_c[0:n_rows, 0:h_len], engine=nisa.tensor_engine)
        xt_sb = nl.ndarray((_PMAX, _PMAX), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=xt_sb[0:h_len, 0:n_rows], src=xt_ps[0:h_len, 0:n_rows])
        xt_chunks.append((xt_sb, h_len))

    # ---- GEMM, ONE PSUM BANK PER ACCUMULATOR ----
    # The shard is wider than a bank (1,210 or 1,571 fp32 columns against 512), and an accumulator
    # spanning banks was observed to leave the SECOND 128-row tile's logits wrong (16 ranks x 16
    # rows and 128 ranks x 2 rows; one tile is fine). So the K loop runs per
    # bank-sized column block into its own [rows, <=512] accumulator, evicted to the fp32 SBUF
    # tile as soon as its last chunk lands. The weight chunk is RESIDENT (loaded once for all
    # row tiles by _load_weight_resident), not re-streamed per tile.
    acc_sb = nl.ndarray((_PMAX, vocab_shard), dtype=nl.float32, buffer=nl.sbuf)
    col_block = nl.tile_size.psum_bank_fmax
    for j0 in range(0, vocab_shard, col_block):
        cw = min(col_block, vocab_shard - j0)
        acc_ps = nl.ndarray((_PMAX, cw), dtype=nl.float32, buffer=nl.psum)
        for c, (xt_sb, h_len) in enumerate(xt_chunks):
            nisa.nc_matmul(
                dst=acc_ps[0:n_rows, :],
                stationary=xt_sb[0:h_len, 0:n_rows],
                moving=wt_res[c][0:h_len, nl.ds(j0, cw)],
                accumulate=c > 0,
            )
        nisa.tensor_copy(dst=acc_sb[0:n_rows, nl.ds(j0, cw)], src=acc_ps[0:n_rows, :])
    return acc_sb, rms


def _select_by_position(data, ramp, position, n_rows: int, k: int, inf_safe: bool = False):
    """Return ``data[row, position[row]]`` as a ``[P, 1]`` fp32 tile, without nc_n_gather.

    Selection is a masked sum: build ``ramp == position`` and reduce ``mask * data``. That is
    four vector ops on a ``[rows, k]`` tile, against one gather instruction -- but the gather
    is avoided deliberately. ``nc_n_gather`` returned ZERO here for a narrow ``data`` tile
    (k=8) even though the position and the data were both verified correct on device, and the
    same primitive is documented as corrupting results in its multi-group form above
    ``gather_group_size`` (see rotational_topk_utils). A masked sum has no such
    width-dependent behaviour, so this is correct for every k the kernel accepts.

    ``position`` is fp32 holding a small exact integer, and ``ramp`` holds exact integers, so
    the equality is exact rather than approximate. ``(ramp == position) * data`` is ONE fused
    ``scalar_tensor_tensor`` on the Vector Engine -- the compare's 1.0/0.0 feeds the multiply
    inside the instruction -- so the whole selection is two instructions, and every one of the
    sampler's instructions counts: at ``gbs = VTP`` the sampler owns a single batch row, every
    op runs on ONE partition, mostly dispatch time, and this function sits twice on
    that serial chain.

    ``inf_safe``: the fused form NaN-poisons on ``+/-inf`` data -- ``0.0 * inf = NaN`` in
    every NON-selected slot, and the add-reduce then returns NaN instead of the selected
    value. The values this kernel samples may legally be ``+/-inf`` (GEMM overflow saturates,
    declared in-contract by ``skip_nan_fold``), so the top-k THRESHOLD selection must use the
    inf-safe form: a predicated copy of the selected slot into a zeroed tile (never a
    multiply against inf), then the same add-reduce. Costs two extra vector ops, so the
    id-gather call keeps the fused form (vocab ids are finite by contract).
    """
    out = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    prod = nl.ndarray((_PMAX, k), dtype=nl.float32, buffer=nl.sbuf)
    if inf_safe:
        sel = nl.ndarray((_PMAX, k), dtype=nl.uint8, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=sel[0:n_rows, :], data=ramp[0:n_rows, :], op0=nl.equal, operand0=position[0:n_rows, :])
        nisa.memset(dst=prod[0:n_rows, :], value=0.0)
        nisa.tensor_copy_predicated(dst=prod[0:n_rows, :], src=data[0:n_rows, :], predicate=sel[0:n_rows, :])
    else:
        nisa.scalar_tensor_tensor(
            dst=prod[0:n_rows, :],
            data=ramp[0:n_rows, :],
            op0=nl.equal,
            operand0=position[0:n_rows, :],
            op1=nl.multiply,
            operand1=data[0:n_rows, :],
        )
    nisa.tensor_reduce(dst=out[0:n_rows, :], data=prod[0:n_rows, :], op=nl.add, axis=1)
    return out


def _sampler_stage(vals_hbm, idx_hbm, params_hbm, n_rows, k, token_out, top_p_rule="drop_crossing"):
    """Sampler on [rows, k]: top-k mask, temperature, softmax, top-p, position, gather.

    Each operand is loaded from HBM INSIDE this function, immediately before its use, and the
    index tile in particular is loaded last. That is not stylistic: a version that takes
    the index tile as an SBUF argument from the caller fails, because the gather is the very
    last thing this function does, so the sampler's own allocations reuse that tile
    before it gets there. The gather then reads zeros and every rank emits token id 0 --
    while ``pos`` is correct and the top-k outputs validate clean, so only an
    end-to-end token check catches it.

    ``top_p_rule`` (``TOP_P_RULES``) decides the token whose cumulative probability crosses
    ``top_p``. ``"drop_crossing"`` (default) uses a STRICT ``>`` on the inclusive cumsum, so the
    crossing token is DROPPED. ``"keep_crossing"`` drops a candidate only when the mass BEFORE it
    (the exclusive cumsum) is already ``>= top_p``, so the crossing token is KEPT: this is HF's
    rule. Either way slot 0 is always kept. The mask sentinel is a finite ``-3000.0``
    (which rounds to -3008.0 in bf16), not -inf.
    """
    P = _PMAX
    vals = nl.ndarray((P, k), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=vals[0:n_rows, :], src=vals_hbm[0:n_rows, :])
    # Column ramp 0..k-1, identical on every partition (channel_multiplier=0). Used to pick a
    # slot by EQUALITY instead of by nc_n_gather -- see _select_by_position.
    ramp = nl.ndarray((P, k), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(ramp, pattern=[[1, k]], offset=0, channel_multiplier=0)
    # 3 columns [top_k, top_p, temperature], or 4 with a per-row runtime draw appended -- the
    # width is a trace-time fact of the caller's tensor, so each form specialises cleanly.
    p_cols = params_hbm.shape[1]
    params = nl.ndarray((P, p_cols), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=params[0:n_rows, :], src=params_hbm[0:n_rows, :])
    top_k_col, top_p_col, temp_col = params[0:n_rows, 0:1], params[0:n_rows, 1:2], params[0:n_rows, 2:3]

    # --- top-k threshold: mask everything below the (top_k-1)-th value ---
    kth = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=kth[0:n_rows, :], data=top_k_col, op0=nl.minimum, operand0=float(k))
    nisa.tensor_scalar(dst=kth[0:n_rows, :], data=kth[0:n_rows, :], op0=nl.subtract, operand0=1.0)
    nisa.tensor_scalar(dst=kth[0:n_rows, :], data=kth[0:n_rows, :], op0=nl.maximum, operand0=0.0)
    thresh = _select_by_position(vals, ramp, kth, n_rows, k, inf_safe=True)

    below = nl.ndarray((P, k), dtype=nl.uint8, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=below[0:n_rows, :], data=vals[0:n_rows, :], op0=nl.less, operand0=thresh[0:n_rows, :])
    masked = nl.ndarray((P, k), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=masked[0:n_rows, :], src=vals[0:n_rows, :])
    sent = _const_tile((P, k), _TOPK_MASK_SENTINEL)
    nisa.tensor_copy_predicated(dst=masked[0:n_rows, :], src=sent[0:n_rows, :], predicate=below[0:n_rows, :])

    # --- temperature (greedy rows divide by 1.0 instead) ---
    greedy = nl.ndarray((P, 1), dtype=nl.uint8, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=greedy[0:n_rows, :], data=temp_col, op0=nl.less, operand0=_SAMPLING_EPS)
    temp_safe = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=temp_safe[0:n_rows, :], src=temp_col)
    ones = _const_tile((P, 1), 1.0)
    nisa.tensor_copy_predicated(dst=temp_safe[0:n_rows, :], src=ones[0:n_rows, :], predicate=greedy[0:n_rows, :])
    inv_t = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=inv_t[0:n_rows, :], data=temp_safe[0:n_rows, :], op=nl.reciprocal)

    # --- softmax (max-shifted), never normalized ---
    # The temperature rides the exp's scale and the max shift its bias, so ``exp((masked - rmax) / temp)`` is a
    # max-reduce plus one activation that also emits the denominator. Nothing divides by it: the consumers compare
    # against scaled thresholds instead (``cum / den > p`` iff ``cum > p * den``, likewise for the draw), which saves
    # two reciprocals, two full-width multiplies and an activation-table reload (all on the serial chain). The
    # last-ulp difference from dividing is arbitrated by the tie-guarded gates, like the PE scan's summation order.
    rmax_neg = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=rmax_neg[0:n_rows, :], data=masked[0:n_rows, :], op=nl.max, axis=1, negate=True)
    exp_bias = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=exp_bias[0:n_rows, :], data1=rmax_neg[0:n_rows, :], data2=inv_t[0:n_rows, :], op=nl.multiply)
    expv = nl.ndarray((P, k), dtype=nl.float32, buffer=nl.sbuf)
    den = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(
        dst=expv[0:n_rows, :],
        data=masked[0:n_rows, :],
        op=nl.exp,
        scale=inv_t[0:n_rows, :],
        bias=exp_bias[0:n_rows, :],
        reduce_op=nl.add,
        reduce_res=den[0:n_rows, :],
        reduce_cmd=nisa.reduce_cmd.reset_reduce,
    )

    # --- top-p: drop_crossing is strict > on the INCLUSIVE cumsum (the crossing token is dropped);
    # keep_crossing is >= on the EXCLUSIVE cumsum (the crossing token is kept) ---
    # The compare is against top_p * den, so the scan runs on the raw exponentials.
    tri = _make_scan_triangle()
    cum = nl.ndarray((P, k), dtype=nl.float32, buffer=nl.sbuf)
    _inclusive_scan_pe(cum, expv, n_rows, k, tri)
    pden = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=pden[0:n_rows, :], data1=top_p_col, data2=den[0:n_rows, :], op=nl.multiply)
    # The compare writes 1.0/0.0 straight into fp32 so the mask multiplies without a cast.
    over_f = nl.ndarray((P, k), dtype=nl.float32, buffer=nl.sbuf)
    if top_p_rule == "keep_crossing":
        # The exclusive cumsum of slot i is the inclusive cumsum of slot i - 1, so the compare reads
        # cum shifted by one column: the same single instruction, and no extra rounding.
        if k > 1:
            nisa.tensor_scalar(
                dst=over_f[0:n_rows, 1:k],
                data=cum[0:n_rows, 0 : k - 1],
                op0=nl.greater_equal,
                operand0=pden[0:n_rows, :],
            )
    else:
        nisa.tensor_scalar(dst=over_f[0:n_rows, :], data=cum[0:n_rows, :], op0=nl.greater, operand0=pden[0:n_rows, :])
    nisa.memset(dst=over_f[0:n_rows, 0:1], value=0)  # never drop the argmax
    # filt = (1 - over) * expv in ONE fused op.
    filt = nl.ndarray((P, k), dtype=nl.float32, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(
        dst=filt[0:n_rows, :],
        data=over_f[0:n_rows, :],
        op0=nl.subtract,
        operand0=1.0,
        reverse0=True,
        op1=nl.multiply,
        operand1=expv[0:n_rows, :],
    )
    # --- position: count cdf entries below the draw; greedy rows take slot 0 ---
    # The cdf is never materialised: the reference's ``cumsum(filt)/cumsum(filt)[k-1] < draw``
    # is evaluated as ``fcum < draw * last`` (same last-ulp caveat as the top-p compare above).
    fcum = nl.ndarray((P, k), dtype=nl.float32, buffer=nl.sbuf)
    _inclusive_scan_pe(fcum, filt, n_rows, k, tri)
    # With a 4-column params tensor the draw u is a RUNTIME per-row operand, so sweeping u is
    # a data change rather than a recompile; with 3 columns the compiled-in constant keeps the
    # legacy contract (and its bit-exact determinism) unchanged.
    draw_thresh = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=draw_thresh[0:n_rows, :],
        data=fcum[0:n_rows, k - 1 : k],
        op0=nl.multiply,
        operand0=params[0:n_rows, 3:4] if p_cols == 4 else _DETERMINISTIC_DRAW,
    )
    lt_f = nl.ndarray((P, k), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=lt_f[0:n_rows, :], data=fcum[0:n_rows, :], op0=nl.less, operand0=draw_thresh[0:n_rows, :])
    pos = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=pos[0:n_rows, :], data=lt_f[0:n_rows, :], op=nl.add, axis=1)

    # Greedy rows take slot 0, by contract (the reference sampler does the same): greedy returns the highest kept
    # candidate, and slot 0 is always kept because the input is descending and the top-k cut is a strict compare against
    # a value <= vals[0]. This is not the argmax of the masked array, which differs only when every real value is below
    # the -3000 sentinel (legal bf16, never real logits) and would then return a token top_k excluded. It replaces ~30
    # Vector instructions at k=256 with one memset; an unsorted value tile would break it, which the descending-order
    # test guards.
    zero_pos = _const_tile((P, 1), 0.0)
    nisa.tensor_copy_predicated(dst=pos[0:n_rows, :], src=zero_pos[0:n_rows, :], predicate=greedy[0:n_rows, :])

    # --- gather the token id at the chosen position ---
    idx_u32 = nl.ndarray((P, k), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.dma_copy(dst=idx_u32[0:n_rows, :], src=idx_hbm[0:n_rows, :])
    idx_f = nl.ndarray((P, k), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=idx_f[0:n_rows, :], src=idx_u32[0:n_rows, :])
    tok_f = _select_by_position(idx_f, ramp, pos, n_rows, k)
    tok_u32 = nl.ndarray((P, 1), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=tok_u32[0:n_rows, :], src=tok_f[0:n_rows, :])
    nisa.dma_copy(dst=token_out[0:n_rows, :], src=tok_u32[0:n_rows, :])


@nki.jit
def tail_chunked(
    hidden_states: nl.NkiTensor,
    gamma: nl.NkiTensor,
    lm_head_weight_t: nl.NkiTensor,
    eps: float = 1e-5,
    hidden_actual: Optional[int] = None,
    zero_interleaved_pad: bool = True,
    k: int = 256,
    sampling_params: Optional[nl.NkiTensor] = None,
    replica_groups: Optional[tuple] = None,
    hop_meshes: Optional[tuple] = None,
    dummy_collectives: bool = False,
    emit_logits: bool = True,
    debug_taps: bool = False,
    deferred_rms: bool = True,
    skip_nan_fold: bool = True,
    norm_rounding: str = "fp32",
    fold_deinterleave: bool = True,
    topk_chunks: int = 0,
    top_p_rule: str = "drop_crossing",
) -> Tuple[nl.NkiTensor, ...]:
    """The decode sampling tail with a single collective (see the module docstring).

    Args:
        hidden_states: [gbs, H] hidden states in HBM, fp32 or 16-bit (a 16-bit input is widened exactly on chip).
        gamma: [1, H] fp32 final RMSNorm weight.
        lm_head_weight_t: [H, V_s] bf16 lm_head shard of this rank, transposed and contiguous.
        eps: RMSNorm epsilon.
        hidden_actual, zero_interleaved_pad: the unpadded hidden and the GPT-OSS mxfp4 padding convention
            (GLM: neither).
        k: candidates kept.
        sampling_params: [gbs / VTP, 3 or 4] fp32 ``[top_k, top_p, temperature, u]`` for this rank's rows (``u`` the
            per-row draw, optional), or None to skip the sampler.
        replica_groups, hop_meshes: all ``VTP`` ranks in rank order as one group, or one mesh per all-to-all hop
            (``collective_probe.compose_hops``); give one of the two.
        dummy_collectives: a zero all-reduce per hop mesh ahead of the hops, for profiling (absorbs arrival skew).
        emit_logits: also return the fp32 pre-collective logits (tests use them; serving does not need them).
        debug_taps: append ``chunked_topk``'s diagnostic outputs.
        deferred_rms: apply the RMSNorm scale on the GEMM's PSUM reads (default); exact, and the matmuls stop waiting on
            the row's sum of squares.
        skip_nan_fold: skip ``chunked_topk``'s NaN fold (default): the input is this kernel's own GEMM output, which is
            finite or saturates to +/-inf. Bitwise the same on NaN-free input.
        norm_rounding: ``"fp32"`` (default) or ``"bf16"``, HF's ``RMSNorm`` rounding (implies the in-chain rms).
        fold_deinterleave: fold the last hop's de-interleave into the top-k loads where the geometry allows (default);
            False forces the explicit one.
        topk_chunks: ``chunked_topk``'s stage-1 chunk count, 0 to pick it.
        top_p_rule: ``"drop_crossing"`` (default) or ``"keep_crossing"``, HF's rule (``TOP_P_RULES``).

    Returns:
        ``(logits, topk_values, topk_indices)`` and, when sampling, ``tokens``: this rank's [gbs, V_s] fp32 logits
        shard, its [gbs / VTP, k] top-k values and global vocab ids, and [gbs / VTP, 1] tokens. The rows this rank
        owns are ``owned_row_block(rank, VTP, gbs, hops)``: the identity for one hop, a mixed-radix shuffle for
        several.
    """
    # hidden_states may also be a LIST of row-slab tensors [rows_i, H] (concatenated-by-rows
    # == one [gbs, H] tensor) -- the pipelined-seam form, reachable only jit-in-jit (a
    # top-level kernel invocation rejects list args). Separate slab tensors are the point:
    # each lm_head row tile then data-depends only on ITS slab's producing collective, so a
    # caller that splits its final all_reduce into K row-slab collectives gets the head
    # overlapped under the later slabs (measured on the fused GPT-OSS decode: the tail
    # window from seam-collective start to tail end shrinks at gbs 512/1024 with
    # K = 2; gate-validated, not bitwise -- splitting a collective changes the runtime's
    # internal reduction realization).
    hidden_slabs = list(hidden_states) if isinstance(hidden_states, (list, tuple)) else [hidden_states]
    for hs in hidden_slabs:
        kernel_assert(len(hs.shape) == 2, "hidden_states (slab) must be [rows, H]")
        kernel_assert(
            hs.dtype in (nl.float32, nl.float16, nl.bfloat16),
            f"hidden_states must be fp32/fp16/bf16 (16-bit widens exactly), got {hs.dtype}",
        )
        kernel_assert(hs.shape[1] == hidden_slabs[0].shape[1], "hidden_states slabs must share H")
        if len(hidden_slabs) > 1:
            kernel_assert(
                hs.shape[0] % _PMAX == 0,
                f"each hidden_states slab must hold whole {_PMAX}-row tiles, got {hs.shape[0]} rows",
            )
    kernel_assert(len(lm_head_weight_t.shape) == 2, "lm_head_weight_t must be [H, V_s]")
    kernel_assert(
        (replica_groups is None) != (hop_meshes is None),
        "supply exactly one of replica_groups (single hop, all VTP ranks in rank order) or hop_meshes",
    )
    hops = (replica_groups,) if hop_meshes is None else tuple(hop_meshes)
    kernel_assert(len(hops) >= 1, "at least one hop is required")

    gbs = sum(hs.shape[0] for hs in hidden_slabs)
    hidden = hidden_slabs[0].shape[1]
    w_hidden, v_s = lm_head_weight_t.shape
    kernel_assert(w_hidden == hidden, f"weight hidden {w_hidden} must match hidden_states hidden {hidden}")

    vtp = 1
    for mesh in hops:
        vtp *= len(mesh[0])
    kernel_assert(gbs % vtp == 0, f"gbs ({gbs}) must be divisible by VTP ({vtp})")
    rows_per_rank = gbs // vtp
    vocab = vtp * v_s

    h_actual = hidden if hidden_actual is None else hidden_actual
    kernel_assert(0 < h_actual <= hidden, f"hidden_actual ({h_actual}) must be in (0, {hidden}]")
    if h_actual < hidden and zero_interleaved_pad:
        kernel_assert(
            hidden % _MXFP4_SHUFFLE_GROUPS == 0 and h_actual % _MXFP4_SHUFFLE_GROUPS == 0,
            f"mxfp4 padding needs H ({hidden}) and H_u ({h_actual}) divisible by {_MXFP4_SHUFFLE_GROUPS}",
        )
    if sampling_params is not None:
        kernel_assert(len(sampling_params.shape) == 2, "sampling_params must be [gbs/VTP, 3]")
        kernel_assert(
            sampling_params.shape[0] == rows_per_rank,
            f"sampling_params must hold THIS RANK'S {rows_per_rank} rows, got {sampling_params.shape[0]}; "
            f"passing the full [gbs, 3] batch makes every rank use rows 0..{rows_per_rank - 1}",
        )
        kernel_assert(
            sampling_params.shape[1] in (3, 4),
            f"sampling_params needs columns [top_k, top_p, temperature] plus an optional 4th "
            f"per-row draw in [0, 1), got {sampling_params.shape[1]} columns",
        )

    # Every stage after the collective is unsharded and carries no core_barrier, so two
    # programs would race on the same shared_hbm buffers. Assert rather than assume: passing
    # num_programs=1 to an inner kernel does not pin the live grid.
    _, n_prgs, _ = get_verified_program_sharding_info("tail_chunked", (0, 1), 2)
    kernel_assert(n_prgs == 1, f"this kernel is LNC=1 only, got n_prgs={n_prgs}")

    kernel_assert(norm_rounding in ("fp32", "bf16"), f"norm_rounding must be 'fp32' or 'bf16', got {norm_rounding!r}")
    kernel_assert(top_p_rule in TOP_P_RULES, f"top_p_rule must be one of {TOP_P_RULES}, got {top_p_rule!r}")
    # The HF realization applies rms in-chain (it is what gets rounded to bf16), so the deferred
    # multiply on the PSUM reads must not run a second time.
    if norm_rounding == "bf16":
        deferred_rms = False

    fence = _skew_fence(hops, hidden) if dummy_collectives else None

    # DEBUG-ONLY output, gated: see emit_logits. When off, neither the buffer nor its store
    # exists, so it costs nothing at all rather than costing a store nobody reads.
    logits = nl.ndarray((gbs, v_s), dtype=nl.float32, buffer=nl.shared_hbm) if emit_logits else None
    # bf16 mirror for the collective: chunked_topk is bf16-only and the collective then moves
    # half the bytes. Same buffer class as the collective's destinations below (every collective
    # src/dst must share one), and per-rank data, hence scratch_hbm().
    logits_bf16 = nl.ndarray((gbs, v_s), dtype=nl.bfloat16, buffer=scratch_hbm())

    # ---- Stages 1-2: norm + lm_head, fused through SBUF, per 128-row tile ----
    gamma_sb = _load_gamma_broadcast(gamma, hidden, h_actual, zero_interleaved_pad)
    wt_res = _load_weight_resident(lm_head_weight_t, hidden, v_s)
    # Slab-start offsets, so each 128-row tile reads ITS slab at a local offset (a tile
    # never straddles slabs: slab rows are whole multiples of _PMAX, asserted above).
    slab_row0 = []
    _acc_rows = 0
    for hs in hidden_slabs:
        slab_row0.append(_acc_rows)
        _acc_rows += hs.shape[0]
    for t in range(div_ceil(gbs, _PMAX)):
        row0 = t * _PMAX
        rows = min(_PMAX, gbs - row0)
        s = max(i for i in range(len(hidden_slabs)) if slab_row0[i] <= row0)
        acc_ps, rms = _norm_lm_head_stage(
            hidden_slabs[s],
            gamma_sb,
            wt_res,
            eps,
            rows,
            row0 - slab_row0[s],
            hidden,
            h_actual,
            v_s,
            fence,
            deferred_rms,
            norm_rounding,
        )
        # Both readers pull straight from PSUM. Under deferred_rms the per-row rms rides each read as the multiply that
        # replaces a plain copy, so normalizing costs no extra op. The two reads stay independent. The mirror store is
        # split by row halves over two queues, since it is the last transfer between the GEMM and the first hop's
        # trigger.
        bf = nl.ndarray((_PMAX, v_s), dtype=nl.bfloat16, buffer=nl.sbuf)
        if deferred_rms:
            nisa.tensor_scalar(dst=bf[0:rows, :], data=acc_ps[0:rows, :], op0=nl.multiply, operand0=rms[0:rows, :])
        else:
            nisa.tensor_copy(dst=bf[0:rows, :], src=acc_ps[0:rows, :])
        st_half = rows // 2
        if st_half > 0:
            nisa.dma_copy(dst=logits_bf16[row0 : row0 + st_half, :], src=bf[0:st_half, :])
            nisa.dma_copy(dst=logits_bf16[row0 + st_half : row0 + rows, :], src=bf[st_half:rows, :])
        else:
            nisa.dma_copy(dst=logits_bf16[row0 : row0 + rows, :], src=bf[0:rows, :])
        if emit_logits:
            logits_sb = nl.ndarray((_PMAX, v_s), dtype=nl.float32, buffer=nl.sbuf)
            if deferred_rms:
                nisa.tensor_scalar(
                    dst=logits_sb[0:rows, :], data=acc_ps[0:rows, :], op0=nl.multiply, operand0=rms[0:rows, :]
                )
            else:
                nisa.tensor_copy(dst=logits_sb[0:rows, :], src=acc_ps[0:rows, :])
            nisa.dma_copy(dst=logits[row0 : row0 + rows, :], src=logits_sb[0:rows, :])

    # ---- Stages 3-4, plain: raw logits through every hop, de-interleaving between ----
    # Each hop is an all_to_all followed by a rows->columns de-interleave, and hops
    # compose: a hop over g peers turns [rows, w] into [rows/g, g*w]. HBM forces
    # collective_dim=0, so the all_to_all splits AND concatenates on rows -- rank r ships
    # row block p to peer p and gets every peer's block back. Because every hop's group is
    # listed in ASCENDING RANK ORDER and the first hop's is 8 contiguous ranks, peer p's
    # block is peer p's vocab shard at every hop, so after the last hop a column index
    # already IS a global vocab id and no index correction of any kind appears. What
    # multi-hop costs instead is row ownership: see owned_row_block.
    cur, cur_rows, cur_w = logits_bf16, gbs, v_s
    # The LAST hop's de-interleave can be FOLDED INTO chunked_topk's own load patterns when
    # every stage-1 snake partition covers whole peer blocks (see interleaved_peer_width in
    # ChunkedTopkConfig): stage 5 then reads the receive buffer directly and the [rows, vocab]
    # materialisation -- a full HBM round trip of the received logits, the head of the
    # post-collective serial chain -- is deleted. Bitwise identical: only DMA descriptors
    # change. Holds for the single-hop production shape (peer width == V_s divides the snake
    # row width); the two-hop form's last-hop peer blocks are wider than a snake partition,
    # so it keeps the explicit de-interleave.
    recv_peer_width = 0
    for hop in range(len(hops)):
        groups = hops[hop]
        g = len(groups[0])
        kernel_assert(cur_rows % g == 0, f"hop {hop}: rows ({cur_rows}) must divide by group size ({g})")
        out_rows = cur_rows // g

        recv = nl.ndarray((cur_rows, cur_w), dtype=nl.bfloat16, buffer=scratch_hbm())
        ncc.all_to_all(dsts=[recv], srcs=[cur], replica_group=ncc.ReplicaGroup(groups), collective_dim=0)

        # Eligibility is the CALLEE's predicate, not a re-derivation: interleaved_fold_supported
        # carries every chunked_topk precondition (chunks > 1, the desnake-path alignments,
        # the peer-width divisibilities). A geometry-only condition here would turn legal
        # small-vocab configs into trace-time refusals on the chunks == 1 delegate path.
        fold_deint = (
            fold_deinterleave
            and topk_chunks == 0
            and hop == len(hops) - 1
            and out_rows > 1
            and interleaved_fold_supported(rows_per_rank, vocab, k, cur_w)
        )
        # De-interleave so peer p's block becomes columns [p*cur_w, (p+1)*cur_w). At
        # out_rows==1 the received rows are ALREADY in peer order, so laying them end to
        # end is exactly the target layout and the de-interleave is a pure reshape.
        if fold_deint:
            cur = recv
            recv_peer_width = cur_w
        elif out_rows == 1:
            cur = recv.reshape((1, g * cur_w))
        else:
            # shared_hbm, not private_hbm: this buffer is the SOURCE of the next hop's
            # collective, and every collective src/dst must be a shared buffer.
            # One strided DMA per output row gathers its g peer blocks (recv rows p * out_rows + r), with pre-built
            # descriptors (dge_mode.none) so the peer runs spread over the static DMA rings: one DMA a row in place of
            # two a peer, which issued serially on one queue.
            wide = nl.ndarray((out_rows, g * cur_w), dtype=nl.bfloat16, buffer=scratch_hbm())
            for r in range(out_rows):
                nisa.dma_copy(
                    dst=wide.ap(pattern=[[cur_w, g], [1, cur_w]], offset=r * g * cur_w),
                    src=recv.ap(pattern=[[out_rows * cur_w, g], [1, cur_w]], offset=r * cur_w),
                    dge_mode=nisa.dge_mode.none,
                )
            cur = wide
        cur_rows, cur_w = out_rows, g * cur_w

    kernel_assert(
        cur_rows == rows_per_rank and cur_w == vocab,
        f"hops left [{cur_rows}, {cur_w}], expected [{rows_per_rank}, {vocab}]",
    )
    full_vocab = cur

    # ---- Stage 5: the whole top-k, locally. Indices come back as GLOBAL vocab ids ----
    topk_out = chunked_topk(
        full_vocab,
        create_chunked_topk_config(
            inp_shape=(rows_per_rank, vocab),
            inp_dtype=nl.bfloat16,
            k=k,
            chunks=topk_chunks,
            sorted=True,
            debug_taps=debug_taps,
            # The logits are this kernel's own GEMM output, provably NaN-free -- see the
            # skip_nan_fold docstring for the argument and the flag to restore the fold.
            skip_nan_fold=skip_nan_fold,
            # Non-zero when the last hop's de-interleave was folded into stage 1's loads:
            # full_vocab is then the RECEIVE buffer and stage 1 reads it in place.
            interleaved_peer_width=recv_peer_width,
        ),
    )
    tk_vals = topk_out[0]
    tk_idx = topk_out[1]
    # The taps, if any, ride through unchanged and are appended to this kernel's outputs
    # LAST, so no existing output position moves and a consumer that ignores them works.
    tk_taps = topk_out[2:] if debug_taps else ()

    # ---- Stage 6: publish the top-k results as this kernel's own outputs ----
    # chunked_topk's buffers are private_hbm (invisible to the caller), so a republish is
    # required -- but it is a straight HBM->HBM DMA per tensor, not a bounce through SBUF.
    topk_values = nl.ndarray((rows_per_rank, k), dtype=nl.bfloat16, buffer=scratch_hbm())
    topk_indices = nl.ndarray((rows_per_rank, k), dtype=nl.uint32, buffer=scratch_hbm())
    nisa.dma_copy(dst=topk_values, src=tk_vals)
    nisa.dma_copy(dst=topk_indices, src=tk_idx)

    if sampling_params is None:
        base = (logits, topk_values, topk_indices) if emit_logits else (topk_values, topk_indices)
        return base + tk_taps

    # ---- Stage 7: sampler, in its OWN loop over this rank's rows ----
    # Separate from the loop above so each operand is read from HBM inside the sampler, close
    # to its use. See _sampler_stage for why that ordering is load-bearing. The sampler reads
    # chunked_topk's own tk_* buffers, NOT the published shared copies: same bytes, but this
    # way the two republish DMAs above run concurrently with the sampler instead of gating it.
    tokens = nl.ndarray((rows_per_rank, 1), dtype=nl.uint32, buffer=scratch_hbm())
    for t in range(div_ceil(rows_per_rank, _PMAX)):
        r0 = t * _PMAX
        rows = min(_PMAX, rows_per_rank - r0)
        _sampler_stage(
            tk_vals[r0 : r0 + rows, :],
            tk_idx[r0 : r0 + rows, :],
            sampling_params[r0 : r0 + rows, :],
            rows,
            k,
            tokens[r0 : r0 + rows, :],
            top_p_rule,
        )
    base = (logits, topk_values, topk_indices, tokens) if emit_logits else (topk_values, topk_indices, tokens)
    return base + tk_taps
