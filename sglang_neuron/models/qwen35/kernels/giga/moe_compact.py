"""Routed-row compaction for the high-concurrency fused MoE block (NKI, LNC=1).

Why: experts_body (tokens stationary, weights moving) costs one full pass of the 4 local experts'
weights through the PE per 128-row token block (PE-bound for each expert), although only
~7.6 % of the EP-gathered rows are routed to ANY of a rank's 4 local experts (1 - (1 - 10/512)^4):
~20 rows at T_ep = 256, ~39 at 512, ~78 at 1024. So the union of the routed rows is compacted into
ONE <= ``cap``-row block (cap = 128) and experts_body runs once on it, keeping its cross-expert
weight-chunk prefetch and its per-row numerics (each compacted row carries its token's full local
affinity row, so expert math, MX activation quantisation and the fp32 sum over experts are exactly
the blocked path's for that token).

    m[t]      = max_e aff[t, e] > 0                     routed to some local expert (DVE)
    idx[t]    = #routed rows before t                   exclusive prefix count on the PE (strictly upper
                                                        triangular ones within a block, all-ones for
                                                        earlier blocks); n = total routed
    O_b[t, j] = m[t] * (idx[t] == k*cap + j)            one-hot slot matrix per block, chunk k (DVE iota
                                                        compare, [rows_b, cap])
    [tok | ac] = sum_b O_b^T @ [t | aff_b]              slot -> token id and slot -> affinity row in ONE
                                                        fp32 PE matmul per block (exact for t < 2^24)
    xc        = xn_hbm[tok]                             ONE indirect row-gather DMA (cap x H bf16)
    y         = experts_body(xc, ac)                    fp32 [cap, H] -> bf16, unused slots have ac = 0
    e_src_b   = so_b + O_b @ y                          combine on the PE: stationary O_b^T [cap, rows_b],
                                                        8 x 512 moving columns per block, + shared partial

No per-element DMA anywhere (the earlier verify_moe compaction built its lists with per-element
SWDGE scatters, with long near-idle time in each layer).

Overflow stays exact: chunk 0 (slots 0..cap-1) runs statically; chunks 1..ceil(n/cap)-1 run in an
``nl.fori_loop`` with a runtime trip count (0 in normal routing). Each overflow iteration rebuilds its
one-hot with a loop-carried slot offset, gathers, runs experts_body (reloading the expert weights:
one extra blocked-style pass, only when it happens) and adds its contribution into the per-block
shared-expert partials so_b in place, before the static combine reads them.
"""

import os

import nki
import nki.isa as nisa
import nki.language as nl

from nkilib.core.utils.kernel_assert import kernel_assert
from .moe_experts_mx import experts_body
from .moe_lowc import _count_reg

P = 128
OC = 512  # combine output columns per PSUM tile (fp32: one bank)
# Rows per compacted chunk (even, <= 128); ~7.6 % of the rows route to a rank, so 128 covers T_ep <= ~1024.
CAP = int(os.environ.get("QWEN35_MOE_COMPACT_CAP", "128"))
# DMA queue for the indirect row gathers (vector offsets): swdge (validated) or hwdge_act / hwdge_sync.
_GATHER_DGE = os.environ.get("QWEN35_MOE_COMPACT_DGE", "swdge")


def _gather_rows(dst, src_hbm, idx, n):
    """dst[j, :] = src_hbm[idx[j], :] for j < n: one indirect DMA (idx int32 [n, 1] SBUF)."""
    W = src_hbm.shape[1]
    src = src_hbm.ap(pattern=[[W, n], [1, W]], offset=0, vector_offset=idx, indirect_dim=0)
    if _GATHER_DGE.startswith("hwdge"):
        eng = nisa.engine.scalar if _GATHER_DGE.endswith("act") else nisa.engine.sync
        nisa.dma_copy(dst=dst, src=src, dge_mode=nisa.dge_mode.hwdge, engine=eng)
    else:
        nisa.dma_copy(dst=dst, src=src, dge_mode=getattr(nisa.dge_mode, _GATHER_DGE))


def _chunk_tables(blocks, idx_l, m_l, v_l, iota_j, cap, E, koff=None):
    """Slot tables of one chunk: (tok int32 [cap, 1], ac fp32 [cap, E], [O_b^T bf16 [cap, rows_b]]).

    koff (fp32 [128, 1] SBUF, all partitions = k*cap) selects chunk k at run time; None = chunk 0.
    """
    ps_l = nl.ndarray((P, E + 2), dtype=nl.float32, buffer=nl.psum)[:cap]
    ot_l = []
    for b, (r0, n) in enumerate(blocks):
        key = idx_l[b]
        if koff is not None:
            key = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)[:n]
            nisa.tensor_tensor(dst=key, data1=idx_l[b], data2=koff[:n], op=nl.subtract)
        o = nl.ndarray((P, cap), dtype=nl.float32, buffer=nl.sbuf)[:n]
        nisa.tensor_scalar(dst=o, data=iota_j[:n], op0=nl.equal, operand0=key, op1=nl.multiply, operand1=m_l[b])
        nisa.nc_matmul(ps_l, o, v_l[b])  # [cap, 2 + E] += O_b^T @ [t mod 128 | t div 128 | aff_b]
        tp = nl.ndarray((P, n), dtype=nl.float32, buffer=nl.psum)[:cap]
        nisa.nc_transpose(dst=tp, data=o)
        ot = nl.ndarray((P, n), dtype=nl.bfloat16, buffer=nl.sbuf)[:cap]
        nisa.tensor_copy(dst=ot, src=tp)  # exact 0 / 1
        ot_l.append(ot)
    # token id = lo + 128 * hi: both parts are small integers (< 128 / < T/128), exact even if the PE
    # rounds fp32 operands below 24 bits (a single fp32 id column is not exact past 2^8 in bf16)
    lohi = nl.ndarray((P, 2), dtype=nl.float32, buffer=nl.sbuf)[:cap]
    nisa.tensor_copy(dst=lohi, src=ps_l[:, 0:2])  # DVE ops may not read two PSUM operands
    tid = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)[:cap]
    nisa.scalar_tensor_tensor(
        dst=tid, data=lohi[:, 1:2], op0=nl.multiply, operand0=float(P), op1=nl.add, operand1=lohi[:, 0:1]
    )
    tok = nl.ndarray((P, 1), dtype=nl.int32, buffer=nl.sbuf)[:cap]
    nisa.tensor_copy(dst=tok, src=tid)
    ac = nl.ndarray((P, E), dtype=nl.float32, buffer=nl.sbuf)[:cap]
    nisa.tensor_copy(dst=ac, src=ps_l[:, 2 : E + 2])
    return tok, ac, ot_l


def _chunk_experts(xn_hbm, tok, ac, cap, gate_up_w, gate_up_scale, down_w, down_scale, pre, in_loop=False):
    """Gather the chunk's rows and run all local experts on them -> bf16 y [cap, H] (affinity-scaled)."""
    H = xn_hbm.shape[1]
    xc = nl.ndarray((P, H), dtype=nl.bfloat16, buffer=nl.sbuf)[:cap]
    _gather_rows(xc, xn_hbm, tok, cap)
    acc = experts_body(xc, ac, gate_up_w, gate_up_scale, down_w, down_scale, pre=pre, zero_scales=in_loop)
    y = nl.ndarray((P, H), dtype=nl.bfloat16, buffer=nl.sbuf)[:cap]
    nisa.tensor_copy(dst=y, src=acc)
    return y


def moe_compact_body(
    xn_hbm, aff_l, so_l, gate_up_w, gate_up_scale, down_w, down_scale, e_src, cap=None, pre=None, on_block=None
):
    """Routed experts of all T gathered rows with one compacted pass, plus the shared partials -> e_src.

    xn_hbm: bf16 [T, H] HBM normed tokens (the gather source). aff_l: per-block fp32 [rows_b, E] SBUF
    local affinities (blocks of <= 128 rows covering T in order), so_l: per-block [rows_b, H] SBUF
    shared-expert partials (bf16; updated in place by overflow chunks). e_src: HBM [T, H] output
    (its dtype, e.g. bf16 or fp8 for the EP ReduceScatter) = so + sum_e aff * expert_e(xn), or a list of
    per-block HBM destinations. pre: {expert: weight tiles} already in flight (issue_expert_weights),
    used by chunk 0. on_block(b): called after block b's rows are written (pipelined ReduceScatter).
    """
    cap = cap or CAP
    T, H = xn_hbm.shape
    E = gate_up_w.shape[0]
    blocks = []
    r0 = 0
    for a in aff_l:
        blocks.append((r0, a.shape[0]))
        r0 += a.shape[0]
    kernel_assert(r0 == T and all(n <= P for _, n in blocks), "aff_l blocks must cover T in <= 128-row blocks")
    kernel_assert(all(b0 % P == 0 for b0, _ in blocks), "blocks must start at multiples of 128 (token id = lo + 128 hi)")
    kernel_assert(cap % 2 == 0 and 2 <= cap <= P, f"cap must be even and <= 128, got {cap}")
    kernel_assert(T < (1 << 24), "token ids must be exact in fp32")
    kernel_assert(H % OC == 0, f"H must be a multiple of {OC}")

    # ---- constants ----
    iu = nl.ndarray((P, P), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(iu, pattern=[[1, P]], offset=0, channel_multiplier=-1)  # column - partition
    upper = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.sbuf)  # upper[k, t] = k < t
    nisa.tensor_scalar(dst=upper, data=iu, op0=nl.greater, operand0=0)
    ones = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(ones, 1.0)
    ij = nl.ndarray((P, cap), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(ij, pattern=[[1, cap]], offset=0, channel_multiplier=0)
    iota_j = nl.ndarray((P, cap), dtype=nl.float32, buffer=nl.sbuf)  # iota_j[t, j] = j
    nisa.tensor_copy(dst=iota_j, src=ij)

    # ---- routed mask, [token id | affinities] rows, exclusive prefix count, total ----
    m_l, v_l, idx_l = [], [], []
    for b, (r0, n) in enumerate(blocks):
        amax = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)[:n]
        nisa.tensor_reduce(dst=amax, op=nl.max, data=aff_l[b], axis=1)
        m = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)[:n]
        nisa.tensor_scalar(dst=m, data=amax, op0=nl.greater, operand0=0.0)
        m_l.append(m)
        ti = nl.ndarray((P, 1), dtype=nl.int32, buffer=nl.sbuf)[:n]
        nisa.iota(ti, pattern=[[0, 1]], offset=0, channel_multiplier=1)  # t mod 128 (blocks are 128-aligned)
        v = nl.ndarray((P, E + 2), dtype=nl.float32, buffer=nl.sbuf)[:n]
        nisa.tensor_copy(dst=v[:, 0:1], src=ti)
        nisa.memset(v[:, 1:2], float(r0 // P))  # t div 128
        nisa.tensor_copy(dst=v[:, 2 : E + 2], src=aff_l[b])
        v_l.append(v)
        ps = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.psum)[:n]
        for q in range(b):
            nisa.nc_matmul(ps, ones[: blocks[q][1], :n], m_l[q])
        nisa.nc_matmul(ps, upper[:n, :n], m)
        idx = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)[:n]
        nisa.tensor_copy(dst=idx, src=ps)
        idx_l.append(idx)
    ps_n = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.psum)
    for b, (r0, n) in enumerate(blocks):
        nisa.nc_matmul(ps_n, m_l[b], ones[:n, 0:1])
    n_sb = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=n_sb, src=ps_n)

    # ---- chunk 0 (static): slots 0..cap-1, experts with the prefetched weights ----
    tok, ac, ot_l = _chunk_tables(blocks, idx_l, m_l, v_l, iota_j, cap, E)
    y = _chunk_experts(xn_hbm, tok, ac, cap, gate_up_w, gate_up_scale, down_w, down_scale, pre)

    # ---- overflow chunks 1..ceil(n/cap)-1 (runtime count, normally 0): added into so_b in place ----
    reg = _count_reg(n_sb, T, cap, skip=1)
    koff = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(koff, float(cap))

    def overflow(_):
        tok_k, ac_k, ot_k = _chunk_tables(blocks, idx_l, m_l, v_l, iota_j, cap, E, koff)
        y_k = _chunk_experts(xn_hbm, tok_k, ac_k, cap, gate_up_w, gate_up_scale, down_w, down_scale, None, True)
        for b, (r0, n) in enumerate(blocks):
            for c0 in range(0, H, OC):
                ps = nl.ndarray((P, OC), dtype=nl.float32, buffer=nl.psum)[:n]
                nisa.nc_matmul(ps, ot_k[b], y_k[:, c0 : c0 + OC])
                nisa.tensor_tensor(dst=so_l[b][:, c0 : c0 + OC], data1=so_l[b][:, c0 : c0 + OC], data2=ps, op=nl.add)
        nisa.tensor_scalar(dst=koff, data=koff, op0=nl.add, operand0=float(cap))

    nl.fori_loop(0, reg, overflow)

    # ---- combine: e_src_b = so_b + O_b @ y (scatter back on the PE) ----
    for b, (r0, n) in enumerate(blocks):
        dst = e_src[b] if isinstance(e_src, (list, tuple)) else e_src[r0 : r0 + n]
        e_sb = nl.ndarray((P, H), dtype=dst.dtype, buffer=nl.sbuf)[:n]
        for c0 in range(0, H, OC):
            ps = nl.ndarray((P, OC), dtype=nl.float32, buffer=nl.psum)[:n]
            nisa.nc_matmul(ps, ot_l[b], y[:, c0 : c0 + OC])
            nisa.tensor_tensor(dst=e_sb[:, c0 : c0 + OC], data1=ps, data2=so_l[b][:, c0 : c0 + OC], op=nl.add)
        nisa.dma_copy(dst=dst, src=e_sb)
        if on_block is not None:  # e.g. start the ReduceScatter of a finished row group
            on_block(b)


@nki.jit
def moe_compact_mx(
    hidden,  # [T, H] bf16 normed tokens
    gate_up_w,  # [E, 128, 2, H/512, I] x4
    gate_up_scale,  # [E, 16, 2, H/512, I] uint8
    down_w,  # [E, 128, I/512, H] x4
    down_scale,  # [E, 16, I/512, H] uint8
    affinities,  # [T, E] fp32 (zeros for experts a token is not routed to)
    base,  # [T, H] bf16 added to the output (the shared-expert partial in the layer)
    cap: int,
):
    """Standalone (tests): out = base + sum_e aff[:, e] * expert_e(hidden), routed rows compacted."""
    T, H = hidden.shape
    E = gate_up_w.shape[0]
    aff_l, so_l = [], []
    for r0 in range(0, T, P):
        n = min(P, T - r0)
        a = nl.ndarray((P, E), dtype=nl.float32, buffer=nl.sbuf)[:n]
        nisa.dma_copy(dst=a, src=affinities[r0 : r0 + n])
        s = nl.ndarray((P, H), dtype=nl.bfloat16, buffer=nl.sbuf)[:n]
        nisa.dma_copy(dst=s, src=base[r0 : r0 + n])
        aff_l.append(a)
        so_l.append(s)
    out = nl.ndarray((T, H), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    moe_compact_body(hidden, aff_l, so_l, gate_up_w, gate_up_scale, down_w, down_scale, out, cap)
    return out
