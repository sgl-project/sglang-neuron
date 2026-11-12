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
"""AttnRes bank scores --- the score table of a Kimi-K3 prefill stage, as a NKI kernel.

``attn_res_cte`` takes an optional ``bank_scores`` table: the score of every bank entry
for that call's ``score_w``. Within a prefill stage the bank only grows once, at the
stage's first layer, where the stage's incoming residual stream becomes its newest
entry. So all of a stage's calls score the same ``C + 1`` snapshots, each with its own
``score_w``, and one pass over the snapshots gives the table of every call::

    v_k       = bank[t, k] for k < C, v_C = stream[t]
    ss_k      = sum_h v_k[h] ** 2
    out[l, t, k] = (sum_h v_k[h] * score_ws[l, h]) * rsqrt(ss_k / H + eps)

``out[l]`` is the ``bank_scores`` of the stage's call ``l``. The call at bank width ``C``
reads its first ``C`` columns; the calls after the append read all ``C + 1``.

The sums over ``H`` run on TensorE, which needs ``H`` on the partitions: each snapshot
of a 128-token tile is DMA-transposed from HBM into ``[h, j, t] = v_t[j * 128 + h]``
blocks, the one read of the snapshots. One accumulation over the ``H / 128`` blocks then
gives, per tile and snapshot, ``[t, :] = sum_h v_t[h] * [v[:, h] | w_hi[:, h] | w_lo[:, h]]``:
the diagonal of the first 128 columns is ``ss``, and the last ``2 L`` columns, the score
vectors split into a bf16 high and low part, sum to the ``L`` dot products with about 16
mantissa bits of ``score_w`` (exactly, for a ``score_w`` that is the product of two bf16
weights). This is the arithmetic of ``attn_res_cte``'s own TensorE scoring (the same
products, accumulation order, ``rsqrt`` and final multiply), with ``L`` score vectors
instead of one. The split score vectors are transposed once on TensorE and sit next to
the snapshot columns of every transpose buffer, so each block is one matmul.

Each token of a tile depends only on its own snapshots, and the instruction sequence is
fixed, so the result is deterministic and the same for any token slice: the scores of
rows ``[a, b)`` computed alone equal those rows of the full table, bit for bit. A
tensor-parallel group can therefore split the tokens and all-gather the table.

The table is stored with one DMA per group of up to ``_GROUP_TILES`` tiles. A group's
tiles interleave their tokens (tile ``g`` holds the rows ``p * G + g``), so each
partition holds ``G`` consecutive rows and the store moves ``G * (C + 1) * 4``-byte
runs instead of ``(C + 1) * 4``-byte ones.

LNC: a 1-D SPMD grid shards the tokens across the programs, as in ``attn_res_cte``.
"""

import nki
import nki.isa as nisa
import nki.language as nl

from ..utils.allocator import sizeinbytes
from ..utils.kernel_assert import kernel_assert
from ..utils.kernel_helpers import div_ceil, get_verified_program_sharding_info

# Partition (token) tile size, and the fp32 elements of one PSUM bank.
_PMAX = 128
_PSUM_BANK = 512
# Bank widths and score vectors the kernel supports.
_MAX_C = 15
_MAX_L = 128
# Bytes of a DMA transpose's output block step: it writes 32-byte aligned blocks.
_DMA_T_ALIGN = 32
# Transpose buffers, at most: enough to keep the DMA queues a few snapshots ahead of
# TensorE.
_MAX_BUFS = 6
# PSUM accumulators of the snapshot scores, one bank each.
_PSUM_RING = 4
# Tiles per group of the output store.
_GROUP_TILES = 8
# The first and the last snapshots load in two DMAs each, so TensorE starts on the
# first half of a snapshot while its second half loads: it starts sooner and ends
# sooner after the last load. Splitting every snapshot slows the transposes (a lower
# transpose rate at T=4096).
_SPLIT_FIRST = 1
_SPLIT_LAST = 2
# SBUF bytes per partition that the kernel leaves free for the compiler.
_SBUF_RESERVE = 8 * 1024
# DMA options of a transfer put on the Sync hardware DGE, off the transposes' queues.
_SIDE = {"dge_mode": nisa.dge_mode.hwdge, "engine": nisa.engine.sync}


def _largest_divisor(n, limit):
    """The largest divisor of ``n`` that is at most ``limit``."""
    return max(d for d in range(1, min(n, limit) + 1) if n % d == 0)


def _load_transposed(dst, src, row, off, n_hb, parts=1):
    """DMA-transpose 128 rows of ``src`` (row stride ``row``, first row at ``off``) into
    ``dst[h, j, t] = row_t[j * 128 + h]``; only the first 128 columns of each block.
    One DMA per ``parts``-th of the blocks, so the first matmuls wait for less."""
    blk = dst.shape[2]
    for p in range(parts):
        j0, j1 = p * n_hb // parts, (p + 1) * n_hb // parts
        nisa.dma_transpose(
            dst=dst.ap([[n_hb * blk, _PMAX], [1, 1], [blk, j1 - j0], [1, _PMAX]], offset=j0 * blk),
            src=src.ap([[row, _PMAX], [1, 1], [_PMAX, j1 - j0], [1, _PMAX]], offset=off + j0 * _PMAX),
        )


def _transposed_score_vectors(score_ws, L, H, dst):
    """``dst[h, j, 128:128 + 2 L] = [w_hi[:, j * 128 + h] | w_lo[:, j * 128 + h]]``: the
    score vectors split as ``hi = dtype(w)``, ``lo = dtype(w - hi)`` (as ``attn_res_cte``
    splits its ``score_w``), with ``H`` on the partitions, next to the snapshot columns
    of the transpose buffer ``dst``.

    The ``[L, H]`` vectors load in one DMA as ``[L * Q, H / Q]`` rows (``q``-th part of
    vector ``l`` on partition ``l * Q + q``), are split on VectorE, and are transposed in
    ``128``-column blocks on TensorE: block ``b`` of partition ``l * Q + q`` is block
    ``j = q * H / (128 Q) + b`` of vector ``l``."""
    n_hb = H // _PMAX
    dtype, blk = dst.dtype, dst.shape[2]
    Q = _largest_divisor(n_hb, _PMAX // L)
    LQ, hq, nbq = L * Q, H // Q, n_hb // Q
    w_nat = nl.ndarray((LQ, hq), dtype=nl.float32, buffer=nl.sbuf)
    # Ahead of the transposes on their queue: the split and the transposes take longer
    # than one snapshot's load, so the vectors are the first thing TensorE waits for.
    nisa.dma_copy(dst=w_nat, src=score_ws.ap(pattern=[[hq, LQ], [1, hq]], offset=0), dge_mode=nisa.dge_mode.none)
    hilo = nl.ndarray((LQ, 2, hq), dtype=dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=hilo[0:LQ, 0, 0:hq], src=w_nat[0:LQ, 0:hq], engine=nisa.engine.vector)
    nisa.tensor_tensor(dst=hilo[0:LQ, 1, 0:hq], data1=w_nat[0:LQ, 0:hq], data2=hilo[0:LQ, 0, 0:hq], op=nl.subtract)
    for b in range(nbq):
        pt = nl.ndarray((_PMAX, 2, LQ), dtype=dtype, buffer=nl.psum)
        for part in range(2):
            nisa.nc_transpose(
                dst=pt[0:_PMAX, part, 0:LQ],
                data=hilo[0:LQ, part, b * _PMAX : (b + 1) * _PMAX],
                engine=nisa.engine.tensor,
            )
        # pt[h, part, l * Q + q] -> dst[h, q * nbq + b, 128 + part * L + l], in (q, part, l)
        # order.
        nisa.tensor_copy(
            dst=dst.ap([[n_hb * blk, _PMAX], [nbq * blk, Q], [L, 2], [1, L]], offset=b * blk + _PMAX),
            src=pt.ap([[2 * LQ, _PMAX], [1, Q], [LQ, 2], [Q, L]]),
            engine=nisa.engine.vector if b % 2 == 0 else nisa.engine.scalar,
        )


@nki.jit
def attn_res_bank_scores(
    stream: nl.NkiTensor,
    score_ws: nl.NkiTensor,
    eps: float,
    bank: nl.NkiTensor = None,
):
    """Score every snapshot of a prefill stage's AttnRes bank for all the stage's calls.

    Args:
        stream: ``[T, H]`` bf16 or fp32 --- the stage's incoming residual stream, which
            the stage's first layer appends to the bank as its last entry.
        score_ws: ``[L, H]`` fp32 --- the fused score vectors
            (``norm.weight * proj.weight``) of the stage's ``L`` AttnRes calls, in call
            order (``1 <= L <= 128``).
        eps: RMSNorm epsilon.
        bank: optional ``[T, C, H]`` in ``stream``'s dtype --- the bank before the
            append (``1 <= C <= 15``), row-major. ``None`` is the empty bank of the
            first stage.

    Returns:
        ``out: [L, T, C + 1]`` fp32, where ``out[l, t, k]`` is the score of snapshot
        ``k`` (bank entry ``k`` for ``k < C``, the stream for ``k = C``) of token ``t``
        for ``score_ws[l]``: ``dot(v, score_ws[l]) * rsqrt(sum(v ** 2) / H + eps)``.
        ``out[l]`` is the ``bank_scores`` argument of ``attn_res_cte`` for call ``l``.

    Notes:
        - The sums run in fp32 on TensorE over the exact bf16 products, with each
          score vector split into a bf16 high and low part (about 16 mantissa bits).
          With fp32 inputs the products and the vectors are fp32.
        - ``T`` must be a multiple of 128 times the SPMD grid size, and ``H`` a
          multiple of 128. Both are checked with ``kernel_assert``.
        - Deterministic, and the same for any 128-row-aligned token slice, bit for bit.
        - A non-finite snapshot element of one token changes only that token's scores.
    """
    T, H = stream.shape
    C = 0 if bank is None else bank.shape[1]
    K = C + 1
    L = score_ws.shape[0]
    dtype = stream.dtype

    kernel_assert(H % _PMAX == 0, f"attn_res_bank_scores requires H % {_PMAX} == 0, got H={H}")
    kernel_assert(C <= _MAX_C, f"attn_res_bank_scores supports C <= {_MAX_C} bank entries, got C={C}")
    kernel_assert(1 <= L <= _MAX_L, f"attn_res_bank_scores supports 1 <= L <= {_MAX_L} score vectors, got L={L}")
    kernel_assert(
        score_ws.dtype == nl.float32 and tuple(score_ws.shape) == (L, H),
        f"attn_res_bank_scores requires score_ws of shape [L, H] in float32, got {score_ws.shape} {score_ws.dtype}",
    )
    kernel_assert(
        dtype == nl.bfloat16 or dtype == nl.float32,
        f"attn_res_bank_scores requires a bf16 or fp32 stream, got {dtype}",
    )
    if bank is not None:
        kernel_assert(
            bank.dtype == dtype and bank.shape[0] == T and bank.shape[2] == H,
            f"attn_res_bank_scores requires bank [T, C, H] in the stream's dtype, got {bank.shape} {bank.dtype}",
        )

    _, num_shards, shard_id = get_verified_program_sharding_info("attn_res_bank_scores", (0, 1))
    kernel_assert(
        T % (_PMAX * num_shards) == 0,
        f"attn_res_bank_scores requires T % (128 * num_shards) == 0, got T={T}, num_shards={num_shards}",
    )
    seq_per_shard = T // num_shards
    n_tiles = seq_per_shard // _PMAX
    shard_off = shard_id * seq_per_shard

    out = nl.ndarray((L, T, K), dtype=nl.float32, buffer=nl.shared_hbm)

    n_hb = H // _PMAX
    dsize = sizeinbytes(dtype)
    W2 = 2 * L
    align = _DMA_T_ALIGN // dsize
    blk = div_ceil(_PMAX + W2, align) * align
    G = _largest_divisor(n_tiles, _GROUP_TILES)
    n_ent = n_tiles * K

    def entry_src(i, k):
        """HBM tensor, row stride and offset of snapshot ``k`` of tile ``i``: the rows
        ``base + p * G + g`` of its group."""
        grp, g = divmod(i, G)
        first = shard_off + grp * _PMAX * G + g
        if k < C:
            return bank, G * C * H, first * C * H + k * H
        return stream, G * H, first * H

    budget = nl.tile_size.total_available_sbuf_size - _SBUF_RESERVE
    fixed = (
        3 * (H // _largest_divisor(n_hb, _PMAX // L)) * 4  # the score vectors and their split
        + _PMAX * (dsize + 2)  # ones and ident, the discarded diagonal products
        + (_PMAX if dsize == 4 else 0)  # the identity predicate (uint8, fp32 calls only)
        + 2 * L * G * K * 4  # the output groups
        + 3 * (2 * K + K * L) * 4  # ss, r and the dot products of the live tiles
    )
    n_bufs = max(1, min(_MAX_BUFS, n_ent, (budget - fixed) // (n_hb * blk * dsize)))
    bufs = [nl.ndarray((_PMAX, n_hb, blk), dtype=dtype, buffer=nl.sbuf) for _ in range(n_bufs)]

    def load(e):
        i, k = divmod(e, K)
        src, row, off = entry_src(i, k)
        split = e < _SPLIT_FIRST or e >= n_ent - _SPLIT_LAST
        _load_transposed(bufs[e % n_bufs], src, row, off, n_hb, parts=2 if split else 1)

    # The score vectors go into the first buffer, then from there into the others.
    _transposed_score_vectors(score_ws, L, H, bufs[0])
    for e in range(n_bufs):
        load(e)
    for n, buf in enumerate(bufs[1:]):
        nisa.tensor_copy(
            dst=buf[0:_PMAX, 0:n_hb, _PMAX : _PMAX + W2],
            src=bufs[0][0:_PMAX, 0:n_hb, _PMAX : _PMAX + W2],
            engine=nisa.engine.vector if n % 2 == 0 else nisa.engine.scalar,
        )

    # Identity: selects the Gram diagonal.
    ones = nl.ndarray((_PMAX, _PMAX), dtype=dtype, buffer=nl.sbuf)
    nisa.memset(dst=ones, value=1.0)
    ident = nl.ndarray((_PMAX, _PMAX), dtype=dtype, buffer=nl.sbuf)
    nisa.affine_select(
        dst=ident,
        pattern=[[-1, _PMAX]],
        channel_multiplier=1,
        on_true_tile=ones,
        on_false_value=0.0,
        cmp_op=nl.equal,
    )
    # Predicate of the diagonal select: the bf16 identity's own bits as uint16 (1.0 is
    # nonzero, +0 is zero), or a uint8 copy of an fp32 identity.
    if dsize == 2:
        ident_sel = ident.view(nl.uint16)
    else:
        ident_sel = nl.ndarray((_PMAX, _PMAX), dtype=nl.uint8, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=ident_sel, data=ident, op0=nl.not_equal, operand0=0.0, engine=nisa.engine.vector)
    # Write-only destination of the selected diagonal: only its fused max is read. fp32
    # (the select keeps its input dtype), as many bytes as the fp8 [128, 128] it was,
    # written through a zero-stride access pattern.
    diag_junk = nl.ndarray((_PMAX, _PMAX // 4), dtype=nl.float32, buffer=nl.sbuf)

    inv_h = 1.0 / float(H)
    psum_ring = [nl.ndarray((_PMAX, _PSUM_BANK), dtype=nl.float32, buffer=nl.psum) for _ in range(_PSUM_RING)]
    out_ring = [nl.ndarray((_PMAX, L, G * K), dtype=nl.float32, buffer=nl.sbuf) for _ in range(2)]

    for i in range(n_tiles):
        grp, g = divmod(i, G)
        ss = nl.ndarray((_PMAX, K), dtype=nl.float32, buffer=nl.sbuf)
        dot = nl.ndarray((_PMAX, K, L), dtype=nl.float32, buffer=nl.sbuf)
        for k in range(K):
            e = i * K + k
            vt = bufs[e % n_bufs]
            acc = psum_ring[e % _PSUM_RING]
            for j in range(n_hb):
                nisa.nc_matmul(
                    dst=acc[0:_PMAX, 0 : _PMAX + W2],
                    stationary=vt[0:_PMAX, j, 0:_PMAX],
                    moving=vt[0:_PMAX, j, 0 : _PMAX + W2],
                    accumulate=j > 0,
                )
            # The buffer is read: refill it with the snapshot n_bufs entries ahead.
            if e + n_bufs < n_ent:
                load(e + n_bufs)
            # ss = the Gram diagonal, by a select and a max over the row. acc[t, u] pairs
            # token t with token u, so a multiply by the identity would turn token u's
            # non-finite entry into token t's NaN (0 * NaN). The diagonal sum v ** 2 is
            # >= 0, so the max of it and the selected zeros is the diagonal itself.
            nisa.select_reduce(
                dst=diag_junk.ap(pattern=[[_PMAX // 4, _PMAX], [0, _PMAX]], offset=0),
                predicate=ident_sel[0:_PMAX, 0:_PMAX],
                on_true=acc[0:_PMAX, 0:_PMAX],
                on_false=0.0,
                reduce_res=ss[0:_PMAX, nl.ds(k, 1)],
                reduce_cmd=nisa.reduce_cmd.reset_reduce,
                reduce_op=nl.maximum,
            )
            # dot = hi + lo, as attn_res_cte sums its two score columns.
            nisa.tensor_reduce(
                dst=dot[0:_PMAX, k, 0:L],
                data=acc.ap([[_PSUM_BANK, _PMAX], [1, L], [L, 2]], offset=_PMAX),
                op=nl.add,
                axis=(2,),
            )
        r = nl.ndarray((_PMAX, K), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=r[0:_PMAX, 0:K], data=ss[0:_PMAX, 0:K], op=nl.rsqrt, scale=inv_h, bias=eps)
        ob = out_ring[grp % 2]
        # ob[p, l, g * K + k] = dot[p, k, l] * r[p, k]
        nisa.tensor_tensor(
            dst=ob.ap([[L * G * K, _PMAX], [G * K, L], [1, K]], offset=g * K),
            data1=dot.ap([[K * L, _PMAX], [1, L], [L, K]]),
            data2=r.ap([[K, _PMAX], [0, L], [1, K]]),
            op=nl.multiply,
        )
        if g == G - 1:
            # Partition p holds rows base + p * G .. base + p * G + G - 1 of every l. The
            # Sync hardware DGE starts the last store sooner than the default
            # software DGE.
            nisa.dma_copy(
                dst=out.ap(pattern=[[G * K, _PMAX], [T * K, L], [1, G * K]], offset=(shard_off + grp * _PMAX * G) * K),
                src=ob[0:_PMAX, 0:L, 0 : G * K],
                **_SIDE,
            )

    return out
