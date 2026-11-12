"""Qwen3.5 MoE block for PREFILL (compute-bound, many tokens per rank), EP over a rank group (NKI, LNC=1).

Per rank, for its T_s residual rows (sequence-sharded over the EP group, as the mixer ReduceScatter leaves them):

    home rank (T_s rows, <= 128-row blocks):
      xn      = RMSNorm_zc(x) * (1 + w_ln)                                 bf16, the expert input
      probs   = softmax(bf16(xn @ W_r)); exact top-10 with decode's tie-break (moe_front: selection on
                probs * (1 - e*2^-21), lower id wins), renormalised  ->  meta = [w_0..w_9 | id_0..id_9] fp32
    dispatch:  AllGather[EP] of meta (80 B per row) and xn (bf16 rows)  ->  mg [T, 20], xg [T, H],  T = T_s * EP
               (QWEN35_MOEP_CC=2hop: chip-aware two-hop gather, same rows permuted)
    home rank, under the AllGather:
      so      = sigmoid(bf16(xn . w_sg)) * down_s(silu(xn W_g) * (xn W_u))  the WHOLE shared expert as a GEMM
                (no gather needed: it is dense over the rank's own rows)
    per local expert j (exact, runtime counts):
      routed rows of j: m_j[t] = any_k(id[t, k] == gid_j); slot = exclusive prefix count of m_j over all T rows
                (block-local prefix and block totals on the PE, block bases by one DVE scan per expert)
      list_j[slot] = t: bf16 one-hot matmuls (stationary = row index shifted to the (block, expert) row, moving
                = one-hot slot windows), 128 // E_L blocks per PSUM tile; the window rows written at their
                runtime base with ONE vector-offset DMA per block (windows are written in block order; later
                blocks overwrite the previous window's stale tail)
      for each 128-slot chunk of list_j (nl.fori_loop, runtime trip count ceil(n_j / 128)):
                gather the chunk's rows of xg and mg (indirect DMA), aff = sum_k w (id == gid_j),
                experts_body on ONE expert (tokens stationary, weights loaded once per expert, MXFP8 activation
                quantisation exactly as in decode), scatter-ADD the fp32 result into e_full[t] (DMA RMW,
                bf16 destination); invalid tail slots go to dump rows T..T+127
    combine:   ReduceScatter[EP] of e_full[0:T]  ->  e [T_s, H];   out = x + bf16(e + so)

Why per-expert lists and tokens stationary: a 128-row token block costs one pass of an expert's weights through
the PE (~24.6k moving columns) whatever its row count, so the cost per
expert is ceil(n_j / 128) passes. Weights-stationary (tokens moving) costs the same columns per row but 192
LDWEIGHTS_MX (a fixed cost each, not hidden behind <= 512 moving columns) per 512-token group: it never beats tokens-
stationary at these shapes.

Dispatch: nki.collectives.all_to_all_v (exact routed-row dispatch) requires LNC=2 on trn3 and the fixed-size
all_to_all only runs on 8-core intra-device or one-core-per-device groups (two hops for 128 ranks) with a
per-peer capacity, so an exact a2a dispatch is not available at LNC=1: the block gathers instead.
EP is a parameter (ep_group / ep_size): EP128 = one rank list of 128, EP32
replicas = four rank lists of 32 (each rank holds 16 experts, tokens are gathered within the replica only).
"""

import os

import nki
import nki.collectives as ncc
import nki.isa as nisa
import nki.language as nl
from nki.collectives import ReplicaGroup

from nkilib.core.utils.kernel_assert import kernel_assert
from .moe_experts_mx import _expert_chunks, experts_body
from .moe_front import rmsnorm_zc_tok
from .moe_lowc import _count_reg

P = 128
TOPK = 10
NMETA = 2 * TOPK  # meta row: [w_0..w_9 | id_0..id_9] fp32
OC = 512  # output columns per PSUM tile
# Front-end row blocks processed together (shared-expert weight chunks are reused across them).
_FRONT_GROUP = int(os.environ.get("QWEN35_MOEP_FRONT_GROUP", "2"))
# Blocks per meta super-chunk in the routing scan (bounds the SBUF footprint at large T).
_SCAN_CHUNK = int(os.environ.get("QWEN35_MOEP_SCAN_CHUNK", "128"))
# DMA queue for the indirect row gathers: swdge (validated in decode) or hwdge_act / hwdge_sync.
_GATHER_DGE = os.environ.get("QWEN35_MOEP_GATHER_DGE", "swdge")
# EP dispatch / combine collectives: "flat" = one AllGather / ReduceScatter over each EP rank list; "2hop" =
# chip-aware: AllGather over the replica's devices (one core per device, same core index) then within the device
# (8 cores), ReduceScatter in reverse. Same result up to a row permutation of the gathered tokens (the lists
# index rows wherever they land); cross-device bytes per rank drop from (ep-1)*T_s to (ep/8-1)*T_s rows.
_CC = os.environ.get("QWEN35_MOEP_CC", "flat")
CORES_PER_DEVICE = 8


def _dma_indirect(dst, src):
    if _GATHER_DGE.startswith("hwdge"):
        eng = nisa.engine.scalar if _GATHER_DGE.endswith("act") else nisa.engine.sync
        nisa.dma_copy(dst=dst, src=src, dge_mode=nisa.dge_mode.hwdge, engine=eng)
    else:
        nisa.dma_copy(dst=dst, src=src, dge_mode=getattr(nisa.dge_mode, _GATHER_DGE))


def _gather(dst, src_hbm, idx, n):
    """dst[j, :] = src_hbm[idx[j], :] for j < n (idx int32 [n, 1] SBUF)."""
    W = src_hbm.shape[1]
    _dma_indirect(dst, src_hbm.ap(pattern=[[W, n], [1, W]], offset=0, vector_offset=idx, indirect_dim=0))


def _zero_hbm(dst2d, dtype):
    """Zero an HBM [R, W] tensor (R a multiple of 128 or <= 128) with DMAs from one zero tile."""
    R, W = dst2d.shape
    for c0 in range(0, W, 4096):
        cw = min(4096, W - c0)
        z = nl.ndarray((P, cw), dtype=dtype, buffer=nl.sbuf)
        nisa.memset(z, 0.0)
        for r0 in range(0, R, P):
            n = min(P, R - r0)
            nisa.dma_copy(dst=dst2d[r0 : r0 + n, c0 : c0 + cw], src=z[:n])


# ----------------------------------------------------------------------------- front end (home rank)


def _router_meta(xT, T, wr, nudge, meta_sb):
    """Router logits (tokens stationary), softmax, exact top-10 -> meta_sb [T, 20] fp32 (weights | ids)."""
    H1, E = wr.shape[1], wr.shape[2]
    ps_l = nl.ndarray((P, E), dtype=nl.float32, buffer=nl.psum)[:T]
    for h1 in range(H1):
        nisa.nc_matmul(ps_l, xT[:, h1, :], wr[:, h1, :])
    logit = nl.ndarray((P, E), dtype=nl.bfloat16, buffer=nl.sbuf)[:T]
    nisa.tensor_copy(dst=logit, src=ps_l)  # HF router runs F.linear in the model dtype
    mx = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)[:T]
    nisa.tensor_reduce(dst=mx, op=nl.max, data=logit, axis=1)
    nisa.tensor_scalar(dst=mx, data=mx, op0=nl.multiply, operand0=-1.0)
    ex = nl.ndarray((P, E), dtype=nl.float32, buffer=nl.sbuf)[:T]
    ssum = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)[:T]
    nisa.activation_reduce(dst=ex, op=nl.exp, data=logit, reduce_op=nl.add, reduce_res=ssum, bias=mx)
    rsum = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)[:T]
    nisa.reciprocal(dst=rsum, data=ssum)
    probs = nl.ndarray((P, E), dtype=nl.float32, buffer=nl.sbuf)[:T]
    nisa.tensor_scalar(dst=probs, data=ex, op0=nl.multiply, operand0=rsum)
    # selection exactly as moe_front (decode): probs nudged by expert id, threshold at the 10th largest
    pt = nl.ndarray((P, E), dtype=nl.float32, buffer=nl.sbuf)[:T]
    nisa.tensor_tensor(dst=pt, data1=probs, data2=nudge[:T], op=nl.multiply)
    top8 = nl.ndarray((P, 8), dtype=nl.float32, buffer=nl.sbuf)[:T]
    nisa.max8(dst=top8, src=pt)
    rest = nl.ndarray((P, E), dtype=nl.float32, buffer=nl.sbuf)[:T]
    nisa.nc_match_replace8(dst=rest, data=pt, vals=top8, imm=-1.0)
    next8 = nl.ndarray((P, 8), dtype=nl.float32, buffer=nl.sbuf)[:T]
    nisa.max8(dst=next8, src=rest)
    keep = nl.ndarray((P, E), dtype=nl.float32, buffer=nl.sbuf)[:T]
    nisa.tensor_scalar(dst=keep, data=pt, op0=nl.greater_equal, operand0=next8[:, TOPK - 9 : TOPK - 8])
    sel_p = nl.ndarray((P, E), dtype=nl.float32, buffer=nl.sbuf)[:T]
    nisa.tensor_tensor(dst=sel_p, data1=probs, data2=keep, op=nl.multiply)
    ksum = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)[:T]
    nisa.tensor_reduce(dst=ksum, op=nl.add, data=sel_p, axis=1)
    nisa.reciprocal(dst=ksum, data=ksum)
    aff = nl.ndarray((P, E), dtype=nl.float32, buffer=nl.sbuf)[:T]
    nisa.tensor_scalar(dst=aff, data=sel_p, op0=nl.multiply, operand0=ksum)
    # (weight, id) of the 10 selected experts: the 10 positive entries of aff. match_replace8's dst_idx pairs
    # equal values with distinct positions, so exact ties (bf16 logits tie often) still give 10 distinct ids.
    a8 = nl.ndarray((P, 8), dtype=nl.float32, buffer=nl.sbuf)[:T]
    nisa.max8(dst=a8, src=aff)
    i8 = nl.ndarray((P, 8), dtype=nl.uint32, buffer=nl.sbuf)[:T]
    r1 = nl.ndarray((P, E), dtype=nl.float32, buffer=nl.sbuf)[:T]
    nisa.nc_match_replace8(dst=r1, data=aff, vals=a8, imm=-1.0, dst_idx=i8)
    b8 = nl.ndarray((P, 8), dtype=nl.float32, buffer=nl.sbuf)[:T]
    nisa.max8(dst=b8, src=r1)
    j8 = nl.ndarray((P, 8), dtype=nl.uint32, buffer=nl.sbuf)[:T]
    r2 = nl.ndarray((P, E), dtype=nl.float32, buffer=nl.sbuf)[:T]
    nisa.nc_match_replace8(dst=r2, data=r1, vals=b8, imm=-1.0, dst_idx=j8)
    nisa.tensor_copy(dst=meta_sb[:, 0:8], src=a8)
    nisa.tensor_copy(dst=meta_sb[:, 8:TOPK], src=b8[:, 0 : TOPK - 8])
    nisa.tensor_copy(dst=meta_sb[:, TOPK : TOPK + 8], src=i8)
    nisa.tensor_copy(dst=meta_sb[:, TOPK + 8 : NMETA], src=j8[:, 0 : TOPK - 8])


def _xT_of(xn, n, H1):
    """Token-major bf16 [n, H] SBUF -> feature-major [128, H/128, n] (PE transposes)."""
    xT = nl.ndarray((P, H1, n), dtype=nl.bfloat16, buffer=nl.sbuf)
    for h1 in range(H1):
        tp = nl.ndarray((P, n), dtype=nl.bfloat16, buffer=nl.psum)
        nisa.nc_transpose(dst=tp, data=xn[:, h1 * P : (h1 + 1) * P])
        nisa.tensor_copy(dst=xT[:, h1, :], src=tp)
    return xT


def prefill_route(x_hbm, ln_w, w_router_p, xn_dst, meta_dst):
    """Home-rank RMSNorm + router over all rows of x_hbm [T_s, H]: xn -> xn_dst (bf16 HBM [T_s, H]),
    meta -> meta_dst (fp32 HBM [T_s, 20]). These are the dispatch payload, so they go first."""
    T, H = x_hbm.shape
    E = w_router_p.shape[2]
    H1 = H // P
    kernel_assert(H % OC == 0 and E >= 16 and E <= OC, "H % 512 == 0 and 16 <= E <= 512")
    wr = nl.ndarray((P, H1, E), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=wr, src=w_router_p)
    w_row = nl.ndarray((P, H), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=w_row, src=ln_w.ap(pattern=[[0, P], [1, H]], offset=0))
    nisa.tensor_scalar(dst=w_row, data=w_row, op0=nl.add, operand0=1.0)
    tb_i = nl.ndarray((P, E), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(tb_i, pattern=[[1, E]], offset=0, channel_multiplier=0)
    nudge = nl.ndarray((P, E), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=nudge, data=tb_i, op0=nl.multiply, operand0=-(2.0**-21), op1=nl.add, operand1=1.0)
    for r0 in range(0, T, P):
        n = min(P, T - r0)
        x_sb = nl.ndarray((P, H), dtype=nl.bfloat16, buffer=nl.sbuf)[:n]
        nisa.dma_copy(dst=x_sb, src=x_hbm[r0 : r0 + n])
        xn = rmsnorm_zc_tok(x_sb, w_row[:n], n, H, 1e-6, nl.bfloat16)
        nisa.dma_copy(dst=xn_dst[r0 : r0 + n], src=xn)
        xT = _xT_of(xn, n, H1)
        meta = nl.ndarray((P, NMETA), dtype=nl.float32, buffer=nl.sbuf)[:n]
        _router_meta(xT, n, wr, nudge, meta)
        nisa.dma_copy(dst=meta_dst[r0 : r0 + n], src=meta)


def prefill_shared(xn_hbm, w_shared_p, w_sd, so_dst):
    """The whole shared expert on the home rank's rows as a GEMM (tokens stationary, bf16 weights streamed in
    512-column chunks once per group of _FRONT_GROUP row blocks): so = sigmoid(bf16(xn . w_sg)) *
    down(bf16(silu(bf16(xn W_g)) * bf16(xn W_u))) -> so_dst bf16 HBM [T_s, H]. Reads xn back from HBM so it can
    be issued after the dispatch collective and run under it."""
    T, H = xn_hbm.shape
    NS = w_shared_p.shape[2]
    Is = (NS - 1) // 2
    H1 = H // P
    I1 = Is // P
    kernel_assert(Is % P == 0 and NS == 2 * Is + 1, "w_shared_p must be [128, H/128, 2*Is + 1], Is % 128 == 0")
    G = _FRONT_GROUP * P
    for g0 in range(0, T, G):
        blocks = [(r0, min(P, T - r0)) for r0 in range(g0, min(T, g0 + G), P)]
        xts, sbs = [], []
        for r0, n in blocks:
            xn = nl.ndarray((P, H), dtype=nl.bfloat16, buffer=nl.sbuf)[:n]
            nisa.dma_copy(dst=xn, src=xn_hbm[r0 : r0 + n])
            xts.append(_xT_of(xn, n, H1))
            sbs.append(nl.ndarray((P, NS), dtype=nl.bfloat16, buffer=nl.sbuf)[:n])
        # gate | up | gate-logit columns
        for c0 in range(0, NS, OC):
            cw = min(OC, NS - c0)
            ws = nl.ndarray((P, H1, OC), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=ws[:, :, 0:cw], src=w_shared_p[:, :, c0 : c0 + cw])
            for b, (r0, n) in enumerate(blocks):
                ps = nl.ndarray((P, OC), dtype=nl.float32, buffer=nl.psum)[:n]
                for h1 in range(H1):
                    nisa.nc_matmul(ps[:, 0:cw], xts[b][:, h1, :], ws[:, h1, 0:cw])
                nisa.tensor_copy(dst=sbs[b][:, c0 : c0 + cw], src=ps[:, 0:cw])
        hts, sgs = [], []
        for b, (r0, n) in enumerate(blocks):
            s_bf = sbs[b]
            g_act = nl.ndarray((P, Is), dtype=nl.float32, buffer=nl.sbuf)[:n]
            nisa.activation(dst=g_act, op=nl.silu, data=s_bf[:, 0:Is])
            h_s = nl.ndarray((P, Is), dtype=nl.bfloat16, buffer=nl.sbuf)[:n]
            nisa.tensor_tensor(dst=h_s, data1=g_act, data2=s_bf[:, Is : 2 * Is], op=nl.multiply)
            sg = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)[:n]
            nisa.activation(dst=sg, op=nl.sigmoid, data=s_bf[:, 2 * Is : 2 * Is + 1])
            hts.append(_xT_of(h_s, n, I1))
            sgs.append(sg)
        sos = [nl.ndarray((P, H), dtype=nl.bfloat16, buffer=nl.sbuf)[:n] for (r0, n) in blocks]
        for c0 in range(0, H, OC):
            wd = nl.ndarray((P, I1, OC), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=wd, src=w_sd.ap(pattern=[[H, P], [P * H, I1], [1, OC]], offset=c0))
            for b, (r0, n) in enumerate(blocks):
                ps = nl.ndarray((P, OC), dtype=nl.float32, buffer=nl.psum)[:n]
                for i1 in range(I1):
                    nisa.nc_matmul(ps, hts[b][:, i1, :], wd[:, i1, :])
                nisa.tensor_scalar(dst=sos[b][:, c0 : c0 + OC], data=ps, op0=nl.multiply, operand0=sgs[b])
        for b, (r0, n) in enumerate(blocks):
            nisa.dma_copy(dst=so_dst[r0 : r0 + n], src=sos[b])


# ----------------------------------------------------------------------------- per-expert routing lists


def route_lists(mg, gid_bc, L2d, CAPL):
    """Per-local-expert token lists from the gathered routing meta.

    mg: HBM fp32 [T, 20] gathered meta (T % 128 == 0). gid_bc: fp32 [128, E_L] SBUF, global expert id of local
    expert j on every partition. L2d: HBM fp32 [E_L * CAPL, 1] (zeroed), list j occupies elements
    j*CAPL .. j*CAPL + CAPL - 1; CAPL >= T + 128. Returns cnt fp32 [128, E_L] (n_j on every partition).

    Batched over blocks (the per-block chain of ~25 small PE -> DVE -> DMA ops was the cost, not the volume):
    everything but the window DMAs runs for GB = 128 // E_L blocks at a time with (block, expert) on the
    partitions. Same windows, same DMAs in the same order as the per-block version (QWEN35_MOEP_LISTS=old),
    so L2d is bit-identical (incl. the stale window tails).
    """
    if os.environ.get("QWEN35_MOEP_LISTS", "new") == "old":
        return _route_lists_old(mg, gid_bc, L2d, CAPL)
    T = mg.shape[0]
    E_L = gid_bc.shape[1]
    nb = T // P
    kernel_assert(T % P == 0, f"gathered rows must be a multiple of 128, got {T}")
    kernel_assert(CAPL >= T + P, "list capacity must cover T + 128")
    kernel_assert(E_L <= 32, "at most 32 local experts")
    NE = nb * E_L  # (block, expert) pairs; block-major layouts everywhere below
    GB = P // E_L  # blocks per group: a group's (block, expert) rows fill <= 128 partitions
    NG = (nb + GB - 1) // GB
    EGm = 4 if E_L % 4 == 0 else (2 if E_L % 2 == 0 else 1)  # experts per window matmul (EGm * 128 columns)
    OB = max(1, min(GB, 4096 // (E_L * P)))  # blocks per one-hot op (<= 8 KB bf16 per partition)

    # ---- constants ----
    iu = nl.ndarray((P, P), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(iu, pattern=[[1, P]], offset=0, channel_multiplier=-1)  # column - partition
    upper = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.sbuf)  # upper[k, t] = k < t
    nisa.tensor_scalar(dst=upper, data=iu, op0=nl.greater, operand0=0)
    ident = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=ident, data=iu, op0=nl.equal, operand0=0)
    ones = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.memset(ones, 1.0)
    isl = nl.ndarray((P, P), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(isl, pattern=[[1, P]], offset=0, channel_multiplier=0)
    iota_s = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.sbuf)  # iota_s[t, s] = s
    nisa.tensor_copy(dst=iota_s, src=isl)
    ip = nl.ndarray((P, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(ip, pattern=[[0, 1]], offset=0, channel_multiplier=1)
    # shifted stationary: wsh[:, P-1-k : P-1-k+R] holds the row index t (exact in bf16) in columns k..k+EGm-1
    wsh = nl.ndarray((P, 2 * P + EGm), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.memset(wsh, 0.0)
    for i in range(EGm):
        nisa.tensor_copy(dst=wsh[:, P - 1 + i : P + i], src=ip)
    # row p = bi*E_L + j of group g: boff[p, g] = 128 (g*GB + bi)
    iq = nl.ndarray((P, GB), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(iq, pattern=[[-E_L, GB]], offset=0, channel_multiplier=1)  # p - E_L*q
    geq = nl.ndarray((P, GB), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=geq, data=iq, op0=nl.greater_equal, operand0=0)
    bi1 = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)  # sum_q (p >= E_L*q) = bi + 1 -> 128 bi
    nisa.tensor_reduce(dst=bi1, op=nl.add, data=geq, axis=1)
    nisa.tensor_scalar(dst=bi1, data=bi1, op0=nl.subtract, operand0=1.0, op1=nl.multiply, operand1=float(P))
    ig = nl.ndarray((P, NG), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(ig, pattern=[[GB * P, NG]], offset=0, channel_multiplier=0)
    boff = nl.ndarray((P, NG), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=boff, src=ig)
    nisa.tensor_scalar(dst=boff, data=boff, op0=nl.add, operand0=bi1)
    irb = nl.ndarray((P, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(irb, pattern=[[0, 1]], offset=0, channel_multiplier=CAPL)
    rowbase = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)  # rowbase[j] = j * CAPL
    nisa.tensor_copy(dst=rowbase, src=irb)
    if EGm > 1:  # wmask[p, i] = (p % EGm == i): row p keeps window i of its EGm-wide matmul output
        wmask = nl.ndarray((P, EGm), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(
            dst=wmask, op=nl.add, data=ident.ap(pattern=[[P, P], [1, EGm], [EGm, P // EGm]], offset=0), axis=2
        )

    # ---- routed masks M[t, b, j] = any_k id[b*128 + t, k] == gid_j ----
    Mf = nl.ndarray((P, nb, E_L), dtype=nl.float32, buffer=nl.sbuf)
    for s0 in range(0, nb, _SCAN_CHUNK):
        sn = min(_SCAN_CHUNK, nb - s0)
        ids = nl.ndarray((P, sn, TOPK), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=ids, src=mg.ap(pattern=[[NMETA, P], [P * NMETA, sn], [1, TOPK]], offset=s0 * P * NMETA + TOPK)
        )
        for j in range(E_L):
            eq = nl.ndarray((P, sn, TOPK), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=eq, data=ids, op0=nl.equal, operand0=gid_bc[:, j : j + 1])
            nisa.tensor_reduce(dst=Mf[:, s0 : s0 + sn, j], op=nl.max, data=eq, axis=2)
    Mf2 = Mf.ap(pattern=[[NE, P], [1, NE]], offset=0)
    Mb = nl.ndarray((P, NE), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=Mb, src=Mf2)
    # unrouted tokens get a negative slot (never matches a one-hot column): mneg = (M - 1) * 1024
    mneg = nl.ndarray((P, NE), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=mneg, data=Mf2, op0=nl.subtract, operand0=1.0, op1=nl.multiply, operand1=1024.0)

    # ---- block-local exclusive prefix (upper^T M) and block totals (ones^T M), all (block, expert) at once ----
    idxm = nl.ndarray((P, NE), dtype=nl.float32, buffer=nl.sbuf)  # slot of (t, b, j), < 0 when not routed
    tot = nl.ndarray((P, nb, E_L), dtype=nl.float32, buffer=nl.sbuf)
    for c0 in range(0, NE, OC):
        cw = min(OC, NE - c0)
        ps = nl.ndarray((P, OC), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(ps[:, 0:cw], upper, Mb[:, c0 : c0 + cw])
        nisa.tensor_tensor(dst=idxm[:, c0 : c0 + cw], data1=ps[:, 0:cw], data2=mneg[:, c0 : c0 + cw], op=nl.add)
        ps2 = nl.ndarray((P, OC), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(ps2[:, 0:cw], ones, Mb[:, c0 : c0 + cw])
        nisa.tensor_copy(dst=tot.ap(pattern=[[NE, P], [1, cw]], offset=c0), src=ps2[:, 0:cw])
    # ---- block bases: inclusive scan of the totals along blocks, minus the totals ----
    ones_nb = nl.ndarray((P, nb), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(ones_nb, 1.0)
    incl = nl.ndarray((P, nb, E_L), dtype=nl.float32, buffer=nl.sbuf)
    for j in range(E_L):
        nisa.tensor_tensor_scan(
            dst=incl[:, :, j], data0=ones_nb, data1=tot[:, :, j], initial=0.0, op0=nl.multiply, op1=nl.add
        )
    cnt = nl.ndarray((P, E_L), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=cnt, src=incl[:, nb - 1, :])
    base = nl.ndarray((P, nb, E_L), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=base, data1=incl, data2=tot, op=nl.subtract)
    # destination element of window (b, j) for every block at once, expert j on partition j (the vector-DGE index
    # tensor must start at partition 0): doff[j, b] = j*CAPL + base[j, b] (diagonal of the broadcast base tile)
    dsel = nl.ndarray((P, nb, E_L), dtype=nl.float32, buffer=nl.sbuf)[:E_L]
    nisa.tensor_tensor(
        dst=dsel, data1=base[0:E_L], data2=ident.ap(pattern=[[P, E_L], [0, nb], [1, E_L]], offset=0), op=nl.multiply
    )
    doff = nl.ndarray((P, nb), dtype=nl.float32, buffer=nl.sbuf)[:E_L]
    nisa.tensor_reduce(dst=doff, op=nl.add, data=dsel, axis=2)
    nisa.tensor_scalar(dst=doff, data=doff, op0=nl.add, operand0=rowbase[0:E_L])
    doff_i = nl.ndarray((P, nb), dtype=nl.int32, buffer=nl.sbuf)[:E_L]
    nisa.tensor_copy(dst=doff_i, src=doff)

    # ---- per group of GB blocks: one-hot slot windows (OB blocks per op) -> token ids in PSUM with (block,
    # expert) on the partitions (shifted stationary, EGm windows per matmul) -> one vector-offset DMA per block ----
    for g in range(NG):
        b0 = g * GB
        gn = min(GB, nb - b0)
        R = gn * E_L
        ps = nl.ndarray((P, EGm * P), dtype=nl.float32, buffer=nl.psum)[:R]
        for o0 in range(0, gn, OB):
            on = min(OB, gn - o0)
            # oh[t, (bl, j), s] = (s == idxm[t, b0 + o0 + bl, j])
            oh = nl.ndarray((P, OB * E_L, P), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_tensor(
                dst=oh[:, 0 : on * E_L, :],
                data1=iota_s.ap(pattern=[[P, P], [0, on * E_L], [1, P]], offset=0),
                data2=idxm.ap(pattern=[[NE, P], [1, on * E_L], [0, P]], offset=(b0 + o0) * E_L),
                op=nl.equal,
            )
            oh2 = oh.reshape((P, OB * E_L * P))
            for bl in range(on):
                for j0 in range(0, E_L, EGm):
                    k = (o0 + bl) * E_L + j0  # first output row of this matmul
                    c = (bl * E_L + j0) * P
                    nisa.nc_matmul(ps[:, 0 : EGm * P], wsh[:, P - 1 - k : P - 1 - k + R], oh2[:, c : c + EGm * P])
        # row p = (bi, j) keeps window p % EGm of its matmul output; + 128 b
        win = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.sbuf)[:R]
        if EGm > 1:
            msk = nl.ndarray((P, EGm, P), dtype=nl.float32, buffer=nl.sbuf)[:R]
            nisa.tensor_tensor(
                dst=msk, data1=ps.ap(pattern=[[EGm * P, R], [P, EGm], [1, P]], offset=0),
                data2=wmask.ap(pattern=[[EGm, R], [1, EGm], [0, P]], offset=0), op=nl.multiply,
            )
            nisa.tensor_reduce(
                dst=win, op=nl.add, data=msk.ap(pattern=[[EGm * P, R], [1, P], [P, EGm]], offset=0), axis=2
            )
            nisa.tensor_scalar(dst=win, data=win, op0=nl.add, operand0=boff[:R, g : g + 1])
        else:
            nisa.tensor_scalar(dst=win, data=ps[:, 0:P], op0=nl.add, operand0=boff[:R, g : g + 1])
        # windows in block order (block b's stale tail is overwritten by block b+1's window, as before)
        for bl in range(gn):
            b = b0 + bl
            dst = L2d.ap(pattern=[[1, E_L], [1, P]], offset=0, vector_offset=doff_i[:, b : b + 1], indirect_dim=0)
            _dma_indirect(dst, win[bl * E_L : (bl + 1) * E_L])
    return cnt


def _route_lists_old(mg, gid_bc, L2d, CAPL):
    """Per-block route_lists (QWEN35_MOEP_LISTS=old, the reference for the batched version): same contract."""
    T = mg.shape[0]
    E_L = gid_bc.shape[1]
    nb = T // P
    kernel_assert(T % P == 0, f"gathered rows must be a multiple of 128, got {T}")
    kernel_assert(CAPL >= T + P, "list capacity must cover T + 128")
    kernel_assert(E_L <= 32, "at most 32 local experts")

    # ---- constants ----
    iu = nl.ndarray((P, P), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(iu, pattern=[[1, P]], offset=0, channel_multiplier=-1)  # column - partition
    upper = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.sbuf)  # upper[k, t] = k < t
    nisa.tensor_scalar(dst=upper, data=iu, op0=nl.greater, operand0=0)
    ones = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.memset(ones, 1.0)
    isl = nl.ndarray((P, P), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(isl, pattern=[[1, P]], offset=0, channel_multiplier=0)
    iota_s = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.sbuf)  # iota_s[t, s] = s
    nisa.tensor_copy(dst=iota_s, src=isl)
    ip = nl.ndarray((P, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(ip, pattern=[[0, 1]], offset=0, channel_multiplier=1)
    lo_f = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=lo_f, src=ip)
    EG = min(E_L, OC // P)  # experts per compaction matmul (<= 512 moving columns)
    lo_st = nl.ndarray((P, E_L), dtype=nl.bfloat16, buffer=nl.sbuf)  # stationary: row index (exact in bf16)
    for j in range(E_L):
        nisa.tensor_copy(dst=lo_st[:, j : j + 1], src=lo_f)
    # window masks per expert group: wmask_g0[j, g*128 + s] = (j == g0 + g), on partitions 0..E_L-1
    wmasks = []
    for g0 in range(0, E_L, EG):
        wi = nl.ndarray((P, EG * P), dtype=nl.int32, buffer=nl.sbuf)
        nisa.iota(wi, pattern=[[1, EG], [0, P]], offset=g0, channel_multiplier=-1)  # g0 + g - j
        wm = nl.ndarray((P, EG * P), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=wm, data=wi, op0=nl.equal, operand0=0)
        wmasks.append(wm)
    # eye[j, j'] = (j == j') and rowbase[j] = j * CAPL
    ie = nl.ndarray((P, E_L), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(ie, pattern=[[1, E_L]], offset=0, channel_multiplier=-1)
    eye = nl.ndarray((P, E_L), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=eye, data=ie, op0=nl.equal, operand0=0)
    irb = nl.ndarray((P, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(irb, pattern=[[0, 1]], offset=0, channel_multiplier=CAPL)
    rowbase = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=rowbase, src=irb)

    # ---- routed masks M[t, j, b] = any_k id[b*128 + t, k] == gid_j ----
    Mf = nl.ndarray((P, E_L, nb), dtype=nl.float32, buffer=nl.sbuf)
    for s0 in range(0, nb, _SCAN_CHUNK):
        sn = min(_SCAN_CHUNK, nb - s0)
        ids = nl.ndarray((P, sn, TOPK), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=ids, src=mg.ap(pattern=[[NMETA, P], [P * NMETA, sn], [1, TOPK]], offset=s0 * P * NMETA + TOPK)
        )
        for j in range(E_L):
            eq = nl.ndarray((P, sn, TOPK), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=eq, data=ids, op0=nl.equal, operand0=gid_bc[:, j : j + 1])
            nisa.tensor_reduce(dst=Mf[:, j, s0 : s0 + sn], op=nl.max, data=eq, axis=2)
    Mb = nl.ndarray((P, E_L, nb), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=Mb, src=Mf)

    # ---- block-local exclusive prefix (upper^T M) and block totals on every partition (ones^T M) ----
    idx = nl.ndarray((P, E_L, nb), dtype=nl.float32, buffer=nl.sbuf)
    tot = nl.ndarray((P, E_L, nb), dtype=nl.float32, buffer=nl.sbuf)
    for j in range(E_L):
        for c0 in range(0, nb, OC):
            cw = min(OC, nb - c0)
            ps = nl.ndarray((P, OC), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(ps[:, 0:cw], upper, Mb[:, j, c0 : c0 + cw])
            nisa.tensor_copy(dst=idx[:, j, c0 : c0 + cw], src=ps[:, 0:cw])
            ps2 = nl.ndarray((P, OC), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(ps2[:, 0:cw], ones, Mb[:, j, c0 : c0 + cw])
            nisa.tensor_copy(dst=tot[:, j, c0 : c0 + cw], src=ps2[:, 0:cw])
    # ---- block bases: inclusive scan of the totals along blocks, minus the totals ----
    ones_nb = nl.ndarray((P, nb), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(ones_nb, 1.0)
    incl = nl.ndarray((P, E_L, nb), dtype=nl.float32, buffer=nl.sbuf)
    for j in range(E_L):
        nisa.tensor_tensor_scan(
            dst=incl[:, j, :], data0=ones_nb, data1=tot[:, j, :], initial=0.0, op0=nl.multiply, op1=nl.add
        )
    base = nl.ndarray((P, E_L, nb), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=base, data1=incl, data2=tot, op=nl.subtract)
    cnt = nl.ndarray((P, E_L), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=cnt, src=incl[:, :, nb - 1])

    # ---- per block: one-hot slot windows -> token ids -> one vector-offset DMA into the lists ----
    for b in range(nb):
        oh = nl.ndarray((P, E_L * P), dtype=nl.bfloat16, buffer=nl.sbuf)
        for j in range(E_L):
            nisa.tensor_scalar(
                dst=oh[:, j * P : (j + 1) * P], data=iota_s, op0=nl.equal, operand0=idx[:, j, b : b + 1],
                op1=nl.multiply, operand1=Mf[:, j, b : b + 1],
            )
        # win[j, s] = row index of slot s of window j (+ 128 b below), rows 0..E_L-1
        win = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.sbuf)[:E_L]
        first = True
        for gi, g0 in enumerate(range(0, E_L, EG)):
            ps = nl.ndarray((P, OC), dtype=nl.float32, buffer=nl.psum)[:E_L]
            nisa.nc_matmul(ps[:, 0 : EG * P], lo_st, oh[:, g0 * P : (g0 + EG) * P])
            msk = nl.ndarray((P, EG * P), dtype=nl.float32, buffer=nl.sbuf)[:E_L]
            nisa.tensor_tensor(dst=msk, data1=ps[:, 0 : EG * P], data2=wmasks[gi][:E_L], op=nl.multiply)
            for g in range(EG):  # row j keeps only its own window (j == g0 + g)
                if first:
                    nisa.tensor_copy(dst=win, src=msk[:, g * P : (g + 1) * P])
                    first = False
                else:
                    nisa.tensor_tensor(dst=win, data1=win, data2=msk[:, g * P : (g + 1) * P], op=nl.add)
        nisa.tensor_scalar(dst=win, data=win, op0=nl.add, operand0=float(b * P))
        # destination element of window j: j * CAPL + base[j, b] (diagonal of the broadcast base tile)
        dsel = nl.ndarray((P, E_L), dtype=nl.float32, buffer=nl.sbuf)[:E_L]
        nisa.tensor_tensor(dst=dsel, data1=base[0:E_L, :, b], data2=eye[0:E_L], op=nl.multiply)
        doff = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)[:E_L]
        nisa.tensor_reduce(dst=doff, op=nl.add, data=dsel, axis=1)
        nisa.tensor_tensor(dst=doff, data1=doff, data2=rowbase[0:E_L], op=nl.add)
        doff_i = nl.ndarray((P, 1), dtype=nl.int32, buffer=nl.sbuf)[:E_L]
        nisa.tensor_copy(dst=doff_i, src=doff)
        _dma_indirect(L2d.ap(pattern=[[1, E_L], [1, P]], offset=0, vector_offset=doff_i, indirect_dim=0), win)
    return cnt


# ----------------------------------------------------------------------------- experts over the lists


def experts_lists(xg, mg, L2d, CAPL, cnt, gid_bc, gate_up_w, gate_up_scale, down_w, down_scale, e_full):
    """e_full[t] += sum_j aff_j(t) * expert_j(xg[t]) for every routed (t, j), via the per-expert lists.

    e_full: HBM bf16 [T + 128, H] (zeroed; rows T.. are dump rows for invalid tail slots)."""
    T, H = xg.shape
    E_L = gate_up_w.shape[0]
    ip = nl.ndarray((P, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(ip, pattern=[[0, 1]], offset=0, channel_multiplier=1)
    ip_f = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=ip_f, src=ip)
    dump_f = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)  # T + p
    nisa.tensor_scalar(dst=dump_f, data=ip_f, op0=nl.add, operand0=float(T))

    for j in range(E_L):
        chunks = _expert_chunks(gate_up_w, gate_up_scale, down_w, down_scale, j)
        reg = _count_reg(cnt[0:1, j : j + 1], T, P)
        koff_f = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(koff_f, 0.0)
        koff_i = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.memset(koff_i, 0)

        def body(_, j=j, chunks=chunks, koff_f=koff_f, koff_i=koff_i):
            tl = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=tl, src=L2d.ap(pattern=[[1, P], [1, 1]], offset=j * CAPL, scalar_offset=koff_i, indirect_dim=0)
            )
            # valid[s] = koff + s < n_j
            pos = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=pos, data1=ip_f, data2=koff_f, op=nl.add)
            valid = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=valid, data1=pos, data2=cnt[:, j : j + 1], op=nl.less)
            tg = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=tg, data1=tl, data2=valid, op=nl.multiply)
            tg_i = nl.ndarray((P, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=tg_i, src=tg)  # gather row (0 for invalid slots)
            ts = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=ts, data1=tl, data2=dump_f, op=nl.subtract)
            nisa.tensor_tensor(dst=ts, data1=ts, data2=valid, op=nl.multiply)
            nisa.tensor_tensor(dst=ts, data1=ts, data2=dump_f, op=nl.add)
            ts_i = nl.ndarray((P, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=ts_i, src=ts)  # scatter row (dump row T + s for invalid slots)
            # affinity of expert j for each slot's token from its gathered meta row
            mrow = nl.ndarray((P, NMETA), dtype=nl.float32, buffer=nl.sbuf)
            _gather(mrow, mg, tg_i, P)
            eq = nl.ndarray((P, TOPK), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=eq, data=mrow[:, TOPK:NMETA], op0=nl.equal, operand0=gid_bc[:, j : j + 1])
            nisa.tensor_tensor(dst=eq, data1=eq, data2=mrow[:, 0:TOPK], op=nl.multiply)
            ac = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=ac, op=nl.add, data=eq, axis=1)
            nisa.tensor_tensor(dst=ac, data1=ac, data2=valid, op=nl.multiply)
            xc = nl.ndarray((P, H), dtype=nl.bfloat16, buffer=nl.sbuf)
            _gather(xc, xg, tg_i, P)
            acc = experts_body(
                xc, ac, gate_up_w[j : j + 1], gate_up_scale[j : j + 1], down_w[j : j + 1], down_scale[j : j + 1],
                pre={0: chunks}, act_evac=True,
            )
            dst = e_full.ap(pattern=[[H, P], [1, H]], offset=0, vector_offset=ts_i, indirect_dim=0)
            nisa.dma_compute(dst=dst, srcs=[dst, acc], reduce_op=nl.add)
            nisa.tensor_scalar(dst=koff_f, data=koff_f, op0=nl.add, operand0=float(P))
            nisa.tensor_scalar(dst=koff_i, data=koff_i, op0=nl.add, operand0=P)

        nl.fori_loop(0, reg, body)


def _gid_bc_from(expert_ids, E_L):
    """fp32 [128, E_L]: global ids of the local experts on every partition (iota 0..E_L-1 when None)."""
    g = nl.ndarray((P, E_L), dtype=nl.float32, buffer=nl.sbuf)
    if expert_ids is None:
        gi = nl.ndarray((P, E_L), dtype=nl.int32, buffer=nl.sbuf)
        nisa.iota(gi, pattern=[[1, E_L]], offset=0, channel_multiplier=0)
        nisa.tensor_copy(dst=g, src=gi)
    else:
        nisa.dma_copy(dst=g, src=expert_ids.ap(pattern=[[0, P], [1, E_L]], offset=0))
    return g


def routed_buffers(T, H, E_L, tag=""):
    """Zeroed scratch of the routed part (issue early: the zeroing DMAs then run under the front end):
    lists L2d fp32 [E_L * (T + 128), 1] and e_full bf16 [T + 128, H]."""
    hbm = nl.private_hbm
    CAPL = T + P
    L2d = nl.ndarray((E_L * CAPL, 1), dtype=nl.float32, buffer=hbm, name="moep_lists" + tag)
    _zero_hbm(L2d.reshape((P, E_L * CAPL // P)), nl.float32)
    e_full = nl.ndarray((T + P, H), dtype=nl.bfloat16, buffer=hbm, name="moep_efull" + tag)
    _zero_hbm(e_full, nl.bfloat16)
    return L2d, e_full


def moe_prefill_routed(xg, mg, expert_ids, gate_up_w, gate_up_scale, down_w, down_scale, bufs):
    """Routed experts of the T gathered rows on this rank, accumulated into bufs' e_full (rows 0..T-1)."""
    T, H = xg.shape
    E_L = gate_up_w.shape[0]
    L2d, e_full = bufs
    gid_bc = _gid_bc_from(expert_ids, E_L)
    cnt = route_lists(mg, gid_bc, L2d, T + P)
    experts_lists(xg, mg, L2d, T + P, cnt, gid_bc, gate_up_w, gate_up_scale, down_w, down_scale, e_full)
    return e_full


def _finish(out, x_rows, e_rows, so_rows, T, H):
    """out = x + bf16(e + so) per 128-row block (all HBM [T, H])."""
    for r0 in range(0, T, P):
        n = min(P, T - r0)
        a = nl.ndarray((P, H), dtype=nl.bfloat16, buffer=nl.sbuf)[:n]
        nisa.dma_copy(dst=a, src=e_rows[r0 : r0 + n])
        s = nl.ndarray((P, H), dtype=nl.bfloat16, buffer=nl.sbuf)[:n]
        nisa.dma_copy(dst=s, src=so_rows[r0 : r0 + n])
        m = nl.ndarray((P, H), dtype=nl.bfloat16, buffer=nl.sbuf)[:n]
        nisa.tensor_tensor(dst=m, data1=a, data2=s, op=nl.add)
        x = nl.ndarray((P, H), dtype=nl.bfloat16, buffer=nl.sbuf)[:n]
        nisa.dma_copy(dst=x, src=x_rows[r0 : r0 + n])
        o = nl.ndarray((P, H), dtype=out.dtype, buffer=nl.sbuf)[:n]
        nisa.tensor_tensor(dst=o, data1=x, data2=m, op=nl.add)
        nisa.dma_copy(dst=out[r0 : r0 + n], src=o)


# ----------------------------------------------------------------------------- kernels


@nki.jit
def moe_prefill_local(
    x,  # [T, H] bf16 residual rows (T % 128 == 0)
    ln_w,  # [H] fp32 post_attention_layernorm (zero-centred)
    w_router_p,  # [128, H/128, E] bf16
    w_shared_p,  # [128, H/128, 2*Is + 1] bf16 (shared gate | up | shared_expert_gate)
    w_sd,  # [Is, H] bf16
    gate_up_w,  # [E, 128, 2, H/512, I] x4 (all experts local)
    gate_up_scale,  # [E, 16, 2, H/512, I] uint8
    down_w,  # [E, 128, I/512, H] x4
    down_scale,  # [E, 16, I/512, H] uint8
):
    """Single-core prefill MoE block (EP = 1, every expert local): returns (out, xn, meta).

    out = x + bf16(routed + shared); xn [T, H] bf16 and meta [T, 20] fp32 (top-10 weights | ids) are returned so
    the test can check routing against torch and the expert math against a reference built from the same routing.
    """
    T, H = x.shape
    hbm = nl.private_hbm
    xn = nl.ndarray((T, H), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    meta = nl.ndarray((T, NMETA), dtype=nl.float32, buffer=nl.shared_hbm)
    so = nl.ndarray((T, H), dtype=nl.bfloat16, buffer=hbm, name="moep_so")
    bufs = routed_buffers(T, H, gate_up_w.shape[0])
    prefill_route(x, ln_w, w_router_p, xn, meta)
    prefill_shared(xn, w_shared_p, w_sd, so)
    e_full = moe_prefill_routed(xn, meta, None, gate_up_w, gate_up_scale, down_w, down_scale, bufs)
    out = nl.ndarray((T, H), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    _finish(out, x, e_full, so, T, H)
    return out, xn, meta


def moe_prefill_ep_body(
    x_shard, ln_w, w_router_p, w_shared_p, w_sd, expert_ids, gate_up_w, gate_up_scale, down_w, down_scale,
    ep_group, ep_size, world=0, out_buffer=nl.shared_hbm, tag="",
):
    """Prefill MoE block over an EP group (positional args only: the tracer rejects keywords to nested
    functions). x_shard [T_s, H] bf16 residual rows of this rank. Returns out [T_s, H] (x + MoE)."""
    T_s, H = x_shard.shape
    T = T_s * ep_size
    hbm = nl.private_hbm
    kernel_assert(T % P == 0, f"T_s * ep_size must be a multiple of 128, got {T}")
    xs = nl.ndarray((T_s, H), dtype=nl.bfloat16, buffer=hbm, name="moep_xs" + tag)
    ms = nl.ndarray((T_s, NMETA), dtype=nl.float32, buffer=hbm, name="moep_ms" + tag)
    so = nl.ndarray((T_s, H), dtype=nl.bfloat16, buffer=hbm, name="moep_so" + tag)
    bufs = routed_buffers(T, H, gate_up_w.shape[0], tag)
    prefill_route(x_shard, ln_w, w_router_p, xs, ms)
    xg = nl.ndarray((T, H), dtype=nl.bfloat16, buffer=hbm, name="moep_xg" + tag)
    mg = nl.ndarray((T, NMETA), dtype=nl.float32, buffer=hbm, name="moep_mg" + tag)
    if _CC == "2hop":
        kernel_assert(world > 0 and world % ep_size == 0, "QWEN35_MOEP_CC=2hop needs world (total ranks)")
        kernel_assert(ep_size % CORES_PER_DEVICE == 0 and ep_size > CORES_PER_DEVICE, "2hop needs ep = 8 * devices")
        D = ep_size // CORES_PER_DEVICE
        g_dev = ReplicaGroup(  # same core index on each device of the replica
            [[r0 + d * CORES_PER_DEVICE + c for d in range(D)] for r0 in range(0, world, ep_size)
             for c in range(CORES_PER_DEVICE)]
        )
        g_core = ReplicaGroup([list(range(b, b + CORES_PER_DEVICE)) for b in range(0, world, CORES_PER_DEVICE)])
        xa = nl.ndarray((D * T_s, H), dtype=nl.bfloat16, buffer=hbm, name="moep_xa" + tag)
        ma = nl.ndarray((D * T_s, NMETA), dtype=nl.float32, buffer=hbm, name="moep_ma" + tag)
        ncc.all_gather(dsts=[ma], srcs=[ms], replica_group=g_dev, collective_dim=0)
        ncc.all_gather(dsts=[mg], srcs=[ma], replica_group=g_core, collective_dim=0)
        ncc.all_gather(dsts=[xa], srcs=[xs], replica_group=g_dev, collective_dim=0)
        ncc.all_gather(dsts=[xg], srcs=[xa], replica_group=g_core, collective_dim=0)
    else:
        ncc.all_gather(dsts=[mg], srcs=[ms], replica_group=ep_group, collective_dim=0)
        ncc.all_gather(dsts=[xg], srcs=[xs], replica_group=ep_group, collective_dim=0)
    prefill_shared(xs, w_shared_p, w_sd, so)  # dense shared expert of the home rows, under the AllGather
    e_full = moe_prefill_routed(xg, mg, expert_ids, gate_up_w, gate_up_scale, down_w, down_scale, bufs)
    e_dst = nl.ndarray((T_s, H), dtype=nl.bfloat16, buffer=hbm, name="moep_edst" + tag)
    if _CC == "2hop":
        e_a = nl.ndarray((D * T_s, H), dtype=nl.bfloat16, buffer=hbm, name="moep_ea" + tag)
        ncc.reduce_scatter(dsts=[e_a], srcs=[e_full[0:T]], op=nl.add, replica_group=g_core, collective_dim=0)
        ncc.reduce_scatter(dsts=[e_dst], srcs=[e_a], op=nl.add, replica_group=g_dev, collective_dim=0)
    else:
        ncc.reduce_scatter(dsts=[e_dst], srcs=[e_full[0:T]], op=nl.add, replica_group=ep_group, collective_dim=0)
    out = nl.ndarray((T_s, H), dtype=nl.bfloat16, buffer=out_buffer, name="moep_out" + tag)
    _finish(out, x_shard, e_dst, so, T_s, H)
    return out


@nki.jit
def moe_prefill_ep(
    x_shard,  # [T_s, H] bf16 residual rows of this rank (token-sharded over the EP group)
    ln_w,  # [H] fp32
    w_router_p,  # [128, H/128, E] bf16 (replicated)
    w_shared_p,  # [128, H/128, 2*Is + 1] bf16 (replicated: the shared expert runs on the home rank)
    w_sd,  # [Is, H] bf16
    expert_ids,  # [1, E_L] fp32 global ids of this rank's local experts
    gate_up_w, gate_up_scale, down_w, down_scale,  # local MX experts [E_L, ...]
    ep_group: ReplicaGroup,
    ep_size: int,
    world: int = 0,  # total ranks (needed by QWEN35_MOEP_CC=2hop)
):
    """Prefill MoE block with EP over ``ep_group`` (rank lists of ``ep_size`` consecutive ranks)."""
    return moe_prefill_ep_body(
        x_shard, ln_w, w_router_p, w_shared_p, w_sd, expert_ids, gate_up_w, gate_up_scale, down_w, down_scale,
        ep_group, ep_size, world,
    )


@nki.jit
def moe_prefill_partial(
    xg,  # [T, H] bf16 gathered normed rows (the EP AllGather output)
    mg,  # [T, 20] fp32 gathered meta (top-10 weights | global ids)
    expert_ids,  # [1, E_L] fp32 global ids of this rank's local experts
    gate_up_w, gate_up_scale, down_w, down_scale,  # local MX experts [E_L, ...]
):
    """One rank's routed partial (tests): e[t] = sum over local j of aff_j(t) * expert_j(xg[t]) (the ReduceScatter
    input); rows routed to no local expert are exactly zero."""
    T, H = xg.shape
    bufs = routed_buffers(T, H, gate_up_w.shape[0])
    xg_i = nl.ndarray((T, H), dtype=nl.bfloat16, buffer=nl.private_hbm, name="moep_xg_in")
    mg_i = nl.ndarray((T, NMETA), dtype=nl.float32, buffer=nl.private_hbm, name="moep_mg_in")
    for r0 in range(0, T, P):
        t = nl.ndarray((P, H), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=t, src=xg[r0 : r0 + P])
        nisa.dma_copy(dst=xg_i[r0 : r0 + P], src=t)
        m = nl.ndarray((P, NMETA), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=m, src=mg[r0 : r0 + P])
        nisa.dma_copy(dst=mg_i[r0 : r0 + P], src=m)
    e_full = moe_prefill_routed(xg_i, mg_i, expert_ids, gate_up_w, gate_up_scale, down_w, down_scale, bufs)
    out = nl.ndarray((T, H), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    for r0 in range(0, T, P):
        t = nl.ndarray((P, H), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=t, src=e_full[r0 : r0 + P])
        nisa.dma_copy(dst=out[r0 : r0 + P], src=t)
    return out
