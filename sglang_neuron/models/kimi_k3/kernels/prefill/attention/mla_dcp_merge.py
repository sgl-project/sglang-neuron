# SPDX-License-Identifier: Apache-2.0
"""MLA DCP merge-scale: weight one rank's flash partial for the cross-rank merge of a query tile.

With decode context parallelism (DCP), each of ``W`` ranks attends a query tile to its slice
of the keys and returns an unnormalized partial ``acc`` with its row maximum ``m`` and row
sum ``l``. After an all-gather of the row statistics of all ranks, each rank weights its
partial, and a reduce-scatter then sums the weighted partials::

    g_max    = max_w all_m[w]                       [H, S]
    g_sum    = sum_w all_l[w] * exp(all_m[w] - g_max)
    corr     = exp(m - g_max) / g_sum
    weighted = acc * corr[:, None, :]               [H, VD, S]

The output is bit-identical to neuronx-cc's lowering of this torch expression (the merge of
the Kimi-K3 prefill), because every step that rounds is the same instruction on the
same engine, in the same order:

- ``g_sum``: the products ``l * exp`` (VectorE multiply), summed over chunks of ``512 // H``
  ranks (five at ``H = 96``): within a chunk sequentially in rank order, then the chunk sums
  sequentially in rank order onto a zeroed accumulator (VectorE adds);
- each exponential: a VectorE subtraction, then a ScalarE ``exp`` (scale 1, no bias);
- the division: a VectorE ``reciprocal`` of ``g_sum``, then a VectorE multiply.

``g_max`` is exact in any order. The VectorE maximum ignores NaN and keeps its first operand
on a tie of signed zeros, so the order can change only a NaN or the sign of a zero in
``g_max``; neither reaches the output (a NaN ``m`` makes its exponent argument NaN, and
``exp(m - (+0)) == exp(m - (-0))``). The kernel therefore takes the maximum as a tree of
wide VectorE maxima, then applies the compiler's final ``max(g_max, -inf)``.

Layout. Each statistics tile puts ``(head, 128-row block)`` pairs on the 128 partitions
(32 heads at ``S = 512``), so the rank reductions use every partition and each tile yields
the weights of its heads. A TensorE transpose (bit exact) puts the rows of a tile's weights
on the partitions; a transpose of one weight column repeated along the free axis is that
head's weight row broadcast to all partitions. ``acc`` keeps its value dimension on the
partitions, so its DMAs move whole 2 KiB rows. The tiles pipeline: while VectorE weights the
heads of one tile, the DMA queues load the next tile's statistics and ``acc`` rows.
"""

import nki
import nki.isa as nisa
import nki.language as nl


_P_MAX = 128
# fp32 elements of one PSUM bank: the widest weight broadcast, and the free size of the
# compiler's tiles, which sets its rank chunk to _F_MAX // H.
_F_MAX = 512
# Heads per acc load and store (VD * S * 4 bytes each, 2 KiB rows).
_ACC_GROUP = 4
# acc buffers: the loads run up to this many groups ahead of the multiplies.
_ACC_BUFS = 10
# PSUM banks of weight broadcasts in flight.
_BCAST_BUFS = 4
# Statistics tiles in flight: the next tile loads while the current one reduces.
_STAT_BUFS = 2
# All DMAs: hardware descriptor generation on the Sync engine, which has no other work.
_DMA = {"dge_mode": nisa.dge_mode.hwdge, "engine": nisa.engine.sync}
_VEC = nisa.engine.vector
_F32 = nl.float32


def kernel_assert(condition: bool, error_text: str):
    """Raise a kernel input-validation error (a self-contained copy of the NKI library helper)."""
    assert condition, f"Kernel validation exception: {error_text}"  # noqa: S101


def _pe_transpose(dst, data, ident):
    """Transpose ``data`` [K, n] into the PSUM tile ``dst`` [n, K] on TensorE (bit exact).

    The instruction of ``nc_transpose`` (transpose mode), with a kernel-owned identity:
    NKI's shared identity has a fixed SBUF address that the allocator does not reserve.
    """
    k = data.shape[0]
    return nisa.nc_matmul(
        dst=dst,
        stationary=data,
        moving=ident[0:k, 0:k],
        is_transpose=True,
        accumulate=False,
    )


@nki.jit
def kimi_k3_mla_dcp_merge(all_stats: nl.NkiTensor, m: nl.NkiTensor, acc: nl.NkiTensor):
    """Weight one rank's MLA flash partial for the DCP merge of a query tile.

    For one rank of a ``W``-rank DCP group and one query tile, after the all-gather of
    the row statistics. Serving shape (Kimi-K3 prefill, DCP32): ``W = 32``, ``H = 96``,
    ``S = 512``, ``VD = 128``; one call per 512-row query tile of each MLA layer.

    Dimensions:
        W: ranks of the DCP group
        H: query heads (all heads)
        S: query rows of the tile (128, 256 or 512)
        VD: value dimension of the partial (at most 128)

    Args:
        all_stats (nl.NkiTensor): [W, 2 * H, S] fp32 (HBM). The all-gathered statistics
            of all ranks: rows ``0 .. H - 1`` of rank ``w`` are its row maxima ``m``, rows
            ``H .. 2H - 1`` its row sums ``l``.
        m (nl.NkiTensor): [H, S, 1] fp32 (HBM). This rank's row maxima (the rank's rows of
            ``all_stats``).
        acc (nl.NkiTensor): [H, VD, S] fp32 (HBM). This rank's unnormalized partial,
            value-major (the flash kernel's output with ``w_v``).

    Returns:
        out (nl.NkiTensor): [H, VD, S] fp32 (HBM). ``acc * corr[:, None, :]``.

    Notes:
        - Bit-identical to neuronx-cc's lowering of the torch chain, including NaN and
          infinity propagation (see the module docstring).
        - HBM-bound: about 63 MB of traffic per call at the serving shape.
        - The sum's rank chunks are those of the compiler's lowering at H = 96 (five ranks);
          the bit identity is verified for that shape.

    Pseudocode:
        for each tile of heads:
            g_max = max(max_w all_m[w], -inf)
            p[w] = all_l[w] * exp(all_m[w] - g_max)
            g_sum = 0
            for each rank chunk c of 512 // H ranks, in order:
                g_sum = (((p[c0] + p[c1]) + ...) + p[c_last]) + g_sum
            corr = exp(m - g_max) * reciprocal(g_sum)
            for each head h of the tile:
                out[h] = acc[h] * broadcast_rows(corr[h])
    """
    W, H2, S = all_stats.shape
    H = H2 // 2
    VD = acc.shape[1]
    kernel_assert(
        H2 == 2 * H and H >= 1,
        f"all_stats must be [W, 2 * H, S], got {all_stats.shape}",
    )
    kernel_assert(S in (128, 256, 512), f"S must be 128, 256 or 512, got S={S}")
    kernel_assert(_F_MAX // H >= 1, f"H must be at most {_F_MAX}, got H={H}")
    kernel_assert(
        tuple(m.shape) == (H, S, 1),
        f"m must be [H, S, 1] = [{H}, {S}, 1], got {m.shape}",
    )
    kernel_assert(
        acc.shape[0] == H and acc.shape[2] == S and VD <= _P_MAX,
        f"acc must be [H, VD, S] = [{H}, VD <= 128, {S}], got {acc.shape}",
    )
    kernel_assert(
        all_stats.dtype == _F32 and m.dtype == _F32 and acc.dtype == _F32,
        f"all inputs must be float32, got {all_stats.dtype}, {m.dtype}, {acc.dtype}",
    )
    out = nl.ndarray((H, VD, S), dtype=_F32, buffer=nl.shared_hbm)

    # Statistics rows: (head, block) pairs, row r = h * n_blocks + b = query rows
    # b * 128 .. b * 128 + 127 of head h. A tile is 128 rows = heads_per_tile whole heads.
    n_blocks = S // _P_MAX
    heads_per_tile = _P_MAX // n_blocks
    n_rows = H * n_blocks
    tiles = [(r0, min(_P_MAX, n_rows - r0)) for r0 in range(0, n_rows, _P_MAX)]
    rank_chunk = _F_MAX // H
    chunks = [(w0, min(rank_chunk, W - w0)) for w0 in range(0, W, rank_chunk)]
    n_full = W // rank_chunk
    groups = [(h0, min(_ACC_GROUP, H - h0)) for h0 in range(0, H, _ACC_GROUP)]

    # ---- Buffers ----
    m_bufs = [
        nl.ndarray((_P_MAX, W, _P_MAX), dtype=_F32, buffer=nl.sbuf)
        for _ in range(_STAT_BUFS)
    ]
    l_bufs = [
        nl.ndarray((_P_MAX, W, _P_MAX), dtype=_F32, buffer=nl.sbuf)
        for _ in range(_STAT_BUFS)
    ]
    own_bufs = [
        nl.ndarray((_P_MAX, _P_MAX), dtype=_F32, buffer=nl.sbuf)
        for _ in range(_STAT_BUFS)
    ]
    acc_bufs = [
        nl.ndarray((VD, _ACC_GROUP, S), dtype=_F32, buffer=nl.sbuf)
        for _ in range(_ACC_BUFS)
    ]

    def load_stats(ti):
        r0, rows = tiles[ti]
        k = ti % _STAT_BUFS
        # Row r of the statistics of rank w is all_stats[w, r // n_blocks, (r % n_blocks) * 128 :].
        nisa.dma_copy(
            dst=m_bufs[k][0:rows, 0:W, 0:_P_MAX],
            src=all_stats.ap(
                pattern=[[_P_MAX, rows], [H2 * S, W], [1, _P_MAX]], offset=r0 * _P_MAX
            ),
            **_DMA,
        )
        nisa.dma_copy(
            dst=l_bufs[k][0:rows, 0:W, 0:_P_MAX],
            src=all_stats.ap(
                pattern=[[_P_MAX, rows], [H2 * S, W], [1, _P_MAX]],
                offset=H * S + r0 * _P_MAX,
            ),
            **_DMA,
        )
        nisa.dma_copy(
            dst=own_bufs[k][0:rows, 0:_P_MAX],
            src=m.ap(pattern=[[_P_MAX, rows], [1, _P_MAX]], offset=r0 * _P_MAX),
            **_DMA,
        )

    def load_group(gi):
        h0, g = groups[gi]
        nisa.dma_copy(
            dst=acc_bufs[gi % _ACC_BUFS][0:VD, 0:g, 0:S],
            src=acc.ap(pattern=[[S, VD], [VD * S, g], [1, S]], offset=h0 * VD * S),
            **_DMA,
        )

    # Statistics of the first tiles, then the first acc groups.
    for ti in range(min(_STAT_BUFS, len(tiles))):
        load_stats(ti)
    next_group = min(_ACC_BUFS, len(groups))
    for gi in range(next_group):
        load_group(gi)

    # Identity of the TensorE transposes.
    ones = nl.ndarray((_P_MAX, _P_MAX), dtype=_F32, buffer=nl.sbuf)
    nisa.memset(dst=ones, value=1.0, engine=_VEC)
    ident = nl.ndarray((_P_MAX, _P_MAX), dtype=_F32, buffer=nl.sbuf)
    nisa.affine_select(
        dst=ident,
        pattern=[[-1, _P_MAX]],
        channel_multiplier=1,
        on_true_tile=ones,
        on_false_value=0.0,
        cmp_op=nl.equal,
    )

    scratch = nl.ndarray((_P_MAX, (W + 1) // 2, _P_MAX), dtype=_F32, buffer=nl.sbuf)
    bcasts = [
        nl.ndarray((_P_MAX, S), dtype=_F32, buffer=nl.psum) for _ in range(_BCAST_BUFS)
    ]

    for ti, (r0, rows) in enumerate(tiles):
        k = ti % _STAT_BUFS
        m_t, l_t, own = m_bufs[k], l_bufs[k], own_bufs[k]
        P = rows

        # ---- g_max: a tree of maxima over the ranks (exact in any order), then max(., -inf). ----
        n = W
        src = m_t
        while n > 1:
            half = n // 2
            nisa.tensor_tensor(
                dst=scratch[0:P, 0:half, :],
                data1=src[0:P, 0:half, :],
                data2=src[0:P, n - half : n, :],
                op=nl.maximum,
                engine=_VEC,
            )
            if n % 2 == 1:
                # The odd middle rank joins the first slot.
                nisa.tensor_tensor(
                    dst=scratch[0:P, 0, :],
                    data1=scratch[0:P, 0, :],
                    data2=src[0:P, half, :],
                    op=nl.maximum,
                    engine=_VEC,
                )
            src, n = scratch, half
        g_max = nl.ndarray((_P_MAX, _P_MAX), dtype=_F32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=g_max[0:P, :],
            data=src[0:P, 0, :],
            op0=nl.maximum,
            operand0=float("-inf"),
            engine=_VEC,
        )

        # ---- p = l * exp(m - g_max), in place in m_t. ----
        x = m_t[0:P, 0:W, :]
        nisa.tensor_tensor(
            dst=x,
            data1=x,
            data2=g_max[0:P, :].expand_dim(1).broadcast(1, W),
            op=nl.subtract,
            engine=_VEC,
        )
        nisa.activation(dst=x, op=nl.exp, data=x)
        nisa.tensor_tensor(
            dst=x, data1=l_t[0:P, 0:W, :], data2=x, op=nl.multiply, engine=_VEC
        )

        # ---- Chunk sums, sequential in rank order within each chunk, into the chunk's first rank. ----
        if n_full > 0:
            full = m_t[0:P, 0 : n_full * rank_chunk, :].reshape_dim(
                1, (n_full, rank_chunk)
            )
            for j in range(1, rank_chunk):
                nisa.tensor_tensor(
                    dst=full[0:P, :, 0, :],
                    data1=full[0:P, :, 0, :],
                    data2=full[0:P, :, j, :],
                    op=nl.add,
                    engine=_VEC,
                )
        w0_last, n_last = chunks[-1]
        if n_last < rank_chunk:
            for j in range(1, n_last):
                nisa.tensor_tensor(
                    dst=m_t[0:P, w0_last, :],
                    data1=m_t[0:P, w0_last, :],
                    data2=m_t[0:P, w0_last + j, :],
                    op=nl.add,
                    engine=_VEC,
                )
        # ---- g_sum: the chunk sums in rank order onto zero. ----
        g_sum = nl.ndarray((_P_MAX, _P_MAX), dtype=_F32, buffer=nl.sbuf)
        nisa.memset(dst=g_sum[0:P, :], value=0.0, engine=_VEC)
        for w0, _ in chunks:
            nisa.tensor_tensor(
                dst=g_sum[0:P, :],
                data1=m_t[0:P, w0, :],
                data2=g_sum[0:P, :],
                op=nl.add,
                engine=_VEC,
            )

        # ---- corr = exp(m - g_max) * (1 / g_sum), in place in own. ----
        nisa.tensor_tensor(
            dst=own[0:P, :],
            data1=own[0:P, :],
            data2=g_max[0:P, :],
            op=nl.subtract,
            engine=_VEC,
        )
        nisa.activation(dst=own[0:P, :], op=nl.exp, data=own[0:P, :])
        rcp = nl.ndarray((_P_MAX, _P_MAX), dtype=_F32, buffer=nl.sbuf)
        nisa.reciprocal(dst=rcp[0:P, :], data=g_sum[0:P, :])
        nisa.tensor_tensor(
            dst=own[0:P, :],
            data1=own[0:P, :],
            data2=rcp[0:P, :],
            op=nl.multiply,
            engine=_VEC,
        )

        # ---- corr_t[q, r] = corr of row r, query row q of its block: the rows on the partitions. ----
        ps_t = nl.ndarray((_P_MAX, _P_MAX), dtype=_F32, buffer=nl.psum)
        _pe_transpose(ps_t[0:_P_MAX, 0:P], own[0:P, 0:_P_MAX], ident)
        corr_t = nl.ndarray((_P_MAX, _P_MAX), dtype=_F32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=corr_t[:, 0:P], src=ps_t[:, 0:P], engine=_VEC)
        # The statistics buffers are free: load the tile after next.
        if ti + _STAT_BUFS < len(tiles):
            load_stats(ti + _STAT_BUFS)

        # ---- out[h] = acc[h] * corr[h] broadcast to the VD partitions, for the tile's heads. ----
        h_first = r0 // n_blocks
        h_end = h_first + P // n_blocks
        for gi, (h0, g) in enumerate(groups):
            if not (h_first <= h0 < h_end):
                continue
            buf = acc_bufs[gi % _ACC_BUFS]
            for hl in range(g):
                h = h0 + hl
                bc = bcasts[h % _BCAST_BUFS]
                for b in range(n_blocks):
                    col = (h - h_first) * n_blocks + b
                    _pe_transpose(
                        bc[0:_P_MAX, b * _P_MAX : (b + 1) * _P_MAX],
                        corr_t[0:_P_MAX, col : col + 1].broadcast(1, _P_MAX),
                        ident,
                    )
                nisa.tensor_tensor(
                    dst=buf[0:VD, hl, 0:S],
                    data1=buf[0:VD, hl, 0:S],
                    data2=bc[0:VD, 0:S],
                    op=nl.multiply,
                    engine=_VEC,
                )
            nisa.dma_copy(
                dst=out.ap(pattern=[[S, VD], [VD * S, g], [1, S]], offset=h0 * VD * S),
                src=buf[0:VD, 0:g, 0:S],
                **_DMA,
            )
            if next_group < len(groups):
                load_group(next_group)
                next_group += 1
    return out
