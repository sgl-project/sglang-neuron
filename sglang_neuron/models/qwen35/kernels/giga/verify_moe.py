"""Routed-row compaction for the MoE of the speculative-verify pass (NKI, LNC=1).

Why: the verify pass EP-gathers C * (k + 1) tokens (256-512 at C = 128). The token-stationary MX
expert kernel (``moe_experts_mx.experts_body``) costs one full pass of the expert weights through the
PE per 128-token tile, so a dense all-expert MoE over 512 rows costs four weight passes against the decode MoE's one, wiping out
MTP's gain. But each local expert only sees ~10 * T / 512 of the T tokens (~2.5 per expert per k+1
at C = 128), so compacting each expert's routed rows into one <= ``cap``-row tile keeps the expert
matmuls (and the weight stream) at exactly one pass per expert, as in plain decode.

Per local expert e:
  mask[t]  = aff[t, e] > 0;  idx[t] = #routed rows before t  (exclusive prefix count on the PE:
             strictly-upper-triangular ones stationary within a tile + all-ones for earlier tiles)
  list_e   : int32 [cap + 1] in HBM, initialised to the zero row T, then token t scattered into slot
             idx[t] (unrouted / overflow tokens go to the dump slot cap)
  xc_e     = rows list_e[0:cap] of [xn ; 0]  (indirect gather), ac_e likewise from [aff ; 0]
  y_e      = ac_e * expert_e(xc_e)           (experts_body with one expert: weights stream once)
Combine: out[t] = shared[t] + sum_e ybuf[g_e[t]] with g_e[t] = e*cap + idx[t] (routed) or the zero
row E*cap (indirect gathers + vector adds; DMA/DVE work that overlaps the PE-bound expert phase).

Overflow (more than ``cap`` routed rows for one expert) DROPS the extra rows' contributions; the
kernel keeps memory safe but the caller must size cap (<= 126) for its routing statistics.
"""

import nki
import nki.isa as nisa
import nki.language as nl

from nkilib.core.utils.kernel_assert import kernel_assert
from .moe_experts_mx import experts_body
from .moe_front import moe_front_body

P = 128


def experts_compact_body(xn_hbm, aff_hbm, gate_up_w, gate_up_scale, down_w, down_scale, T, cap):
    """xn_hbm [T+1, H] bf16 and aff_hbm [T+1, E] fp32 in HBM (row T all zeros).

    Returns (ybuf [E*cap + 1, H] bf16 HBM with row E*cap zero, [(r0, r1, gidx [rows, E] int32 SBUF)]).
    """
    H = xn_hbm.shape[1]
    E = gate_up_w.shape[0]
    kernel_assert(cap % 2 == 0 and cap + 1 <= P, f"cap must be even and <= 126, got {cap}")
    kernel_assert(T * 2 < (1 << 24), "token count too large")
    tiles = [(r0, min(T, r0 + P)) for r0 in range(0, T, P)]
    hbm = nl.private_hbm

    # constants
    iu = nl.ndarray((P, P), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(iu, pattern=[[1, P]], offset=0, channel_multiplier=-1)  # m - k
    upper = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.sbuf)  # upper[k, m] = (k < m)
    nisa.tensor_scalar(dst=upper, data=iu, op0=nl.greater, operand0=0)
    ones = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(ones, 1.0)
    ecap_i = nl.ndarray((P, E), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(ecap_i, pattern=[[cap, E]], offset=0, channel_multiplier=0)
    ecap = nl.ndarray((P, E), dtype=nl.float32, buffer=nl.sbuf)  # e * cap
    nisa.tensor_copy(dst=ecap, src=ecap_i)

    # masks and exclusive prefix counts, tile by tile
    masks = []
    for r0, r1 in tiles:
        a = nl.ndarray((P, E), dtype=nl.float32, buffer=nl.sbuf)[: r1 - r0]
        nisa.dma_copy(dst=a, src=aff_hbm[r0:r1])
        m = nl.ndarray((P, E), dtype=nl.float32, buffer=nl.sbuf)[: r1 - r0]
        nisa.tensor_scalar(dst=m, data=a, op0=nl.greater, operand0=0.0)
        masks.append(m)
    lst = nl.ndarray((E * (cap + 1), 1), dtype=nl.int32, buffer=hbm)
    init = nl.ndarray((P, E), dtype=nl.int32, buffer=nl.sbuf)[: cap + 1]
    nisa.iota(init, pattern=[[0, E]], offset=T, channel_multiplier=0)
    nisa.dma_copy(dst=lst.ap(pattern=[[1, cap + 1], [cap + 1, E]], offset=0), src=init)
    gidx_tiles = []
    for rt, (r0, r1) in enumerate(tiles):
        n = r1 - r0
        ps = nl.ndarray((P, E), dtype=nl.float32, buffer=nl.psum)[:n]
        for q in range(rt):
            nq = tiles[q][1] - tiles[q][0]
            nisa.nc_matmul(ps, ones[:nq, :n], masks[q])
        nisa.nc_matmul(ps, upper[:n, :n], masks[rt])
        idx = nl.ndarray((P, E), dtype=nl.float32, buffer=nl.sbuf)[:n]
        nisa.tensor_copy(dst=idx, src=ps)
        valid = nl.ndarray((P, E), dtype=nl.float32, buffer=nl.sbuf)[:n]
        nisa.tensor_scalar(dst=valid, data=idx, op0=nl.less, operand0=float(cap))
        nisa.tensor_tensor(dst=valid, data1=valid, data2=masks[rt], op=nl.multiply)
        # slot = valid ? idx : cap ;  gidx = valid ? e*cap + idx : E*cap
        slot_f = nl.ndarray((P, E), dtype=nl.float32, buffer=nl.sbuf)[:n]
        nisa.tensor_scalar(dst=slot_f, data=idx, op0=nl.subtract, operand0=float(cap))
        nisa.tensor_tensor(dst=slot_f, data1=slot_f, data2=valid, op=nl.multiply)
        nisa.tensor_scalar(dst=slot_f, data=slot_f, op0=nl.add, operand0=float(cap))
        slot = nl.ndarray((P, E), dtype=nl.int32, buffer=nl.sbuf)[:n]
        nisa.tensor_copy(dst=slot, src=slot_f)
        g_f = nl.ndarray((P, E), dtype=nl.float32, buffer=nl.sbuf)[:n]
        nisa.tensor_tensor(dst=g_f, data1=idx, data2=ecap[:n], op=nl.add)
        nisa.tensor_scalar(dst=g_f, data=g_f, op0=nl.subtract, operand0=float(E * cap))
        nisa.tensor_tensor(dst=g_f, data1=g_f, data2=valid, op=nl.multiply)
        nisa.tensor_scalar(dst=g_f, data=g_f, op0=nl.add, operand0=float(E * cap))
        gidx = nl.ndarray((P, E), dtype=nl.int32, buffer=nl.sbuf)[:n]
        nisa.tensor_copy(dst=gidx, src=g_f)
        gidx_tiles.append((r0, r1, gidx))
        tok = nl.ndarray((P, 1), dtype=nl.int32, buffer=nl.sbuf)[:n]
        nisa.iota(tok, pattern=[[0, 1]], offset=r0, channel_multiplier=1)
        for e in range(E):
            nisa.dma_copy(
                dst=lst.ap(pattern=[[1, n], [1, 1]], offset=e * (cap + 1), vector_offset=slot[:, e : e + 1], indirect_dim=0),
                src=tok,
                dge_mode=nisa.dge_mode.swdge,
            )

    ybuf = nl.ndarray((E * cap + 1, H), dtype=nl.bfloat16, buffer=hbm)
    zrow = nl.ndarray((1, H), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.memset(zrow, 0.0)
    nisa.dma_copy(dst=ybuf[E * cap : E * cap + 1], src=zrow)
    for e in range(E):
        li = nl.ndarray((P, 1), dtype=nl.int32, buffer=nl.sbuf)[:cap]
        nisa.dma_copy(dst=li, src=lst[e * (cap + 1) : e * (cap + 1) + cap])
        xc = nl.ndarray((P, H), dtype=nl.bfloat16, buffer=nl.sbuf)[:cap]
        nisa.dma_copy(
            dst=xc, src=xn_hbm.ap(pattern=[[H, cap], [1, H]], offset=0, vector_offset=li, indirect_dim=0),
            dge_mode=nisa.dge_mode.swdge,
        )
        ac = nl.ndarray((P, E), dtype=nl.float32, buffer=nl.sbuf)[:cap]
        nisa.dma_copy(
            dst=ac, src=aff_hbm.ap(pattern=[[E, cap], [1, E]], offset=0, vector_offset=li, indirect_dim=0),
            dge_mode=nisa.dge_mode.swdge,
        )
        ac_e = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)[:cap]
        nisa.tensor_copy(dst=ac_e, src=ac[:, e : e + 1])
        acc = experts_body(
            xc, ac_e, gate_up_w[e : e + 1], gate_up_scale[e : e + 1], down_w[e : e + 1], down_scale[e : e + 1]
        )
        y = nl.ndarray((P, H), dtype=nl.bfloat16, buffer=nl.sbuf)[:cap]
        nisa.tensor_copy(dst=y, src=acc)
        nisa.dma_copy(dst=ybuf[e * cap : (e + 1) * cap], src=y)
    return ybuf, gidx_tiles


def combine_rows(ybuf, gidx_tiles, base_tiles, out_hbm, E):
    """out_hbm[r0:r1] = base + sum_e ybuf[gidx[:, e]] per tile (base: SBUF [rows, H] tiles or None)."""
    H = ybuf.shape[1]
    for (r0, r1, gidx), base in zip(gidx_tiles, base_tiles):
        n = r1 - r0
        acc = nl.ndarray((P, H), dtype=nl.float32, buffer=nl.sbuf)[:n]
        for e in range(E):
            g = nl.ndarray((P, H), dtype=nl.bfloat16, buffer=nl.sbuf)[:n]
            nisa.dma_copy(
                dst=g, src=ybuf.ap(pattern=[[H, n], [1, H]], offset=0, vector_offset=gidx[:, e : e + 1], indirect_dim=0),
                dge_mode=nisa.dge_mode.swdge,
            )
            if e == 0:
                if base is None:
                    nisa.tensor_copy(dst=acc, src=g)
                else:
                    nisa.tensor_tensor(dst=acc, data1=base, data2=g, op=nl.add)
            else:
                nisa.tensor_tensor(dst=acc, data1=acc, data2=g, op=nl.add)
        o = nl.ndarray((P, H), dtype=out_hbm.dtype, buffer=nl.sbuf)[:n]
        nisa.tensor_copy(dst=o, src=acc)
        nisa.dma_copy(dst=out_hbm[r0:r1], src=o)


@nki.jit
def moe_experts_compact(hidden, gate_up_w, gate_up_scale, down_w, down_scale, affinities, cap: int):
    """Standalone: out[t] = sum_e aff[t, e] * expert_e(hidden[t]) for any T (routed rows compacted)."""
    T, H = hidden.shape
    E = gate_up_w.shape[0]
    xn_hbm = nl.ndarray((T + 1, H), dtype=nl.bfloat16, buffer=nl.private_hbm)
    aff_hbm = nl.ndarray((T + 1, E), dtype=nl.float32, buffer=nl.private_hbm)
    nisa.dma_copy(dst=xn_hbm[0:T], src=hidden)
    nisa.dma_copy(dst=aff_hbm[0:T], src=affinities)
    z = nl.ndarray((1, H), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.memset(z, 0.0)
    nisa.dma_copy(dst=xn_hbm[T : T + 1], src=z)
    za = nl.ndarray((1, E), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(za, 0.0)
    nisa.dma_copy(dst=aff_hbm[T : T + 1], src=za)
    ybuf, gidx = experts_compact_body(xn_hbm, aff_hbm, gate_up_w, gate_up_scale, down_w, down_scale, T, cap)
    out = nl.ndarray((T, H), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    combine_rows(ybuf, gidx, [None] * len(gidx), out, E)
    return out


def moe_rows_compact_body(m_dst, post_ln_w, w_router_p, sel, w_shared_p, w_sd, gate_up_w, gate_up_scale, down_w,
                          down_scale, e_src, cap):
    """Fused MoE (router + shared + compacted routed experts) for T > 128 gathered rows -> e_src [T, H]."""
    T, H = m_dst.shape
    E = gate_up_w.shape[0]
    xn_hbm = nl.ndarray((T + 1, H), dtype=nl.bfloat16, buffer=nl.private_hbm)
    aff_hbm = nl.ndarray((T + 1, E), dtype=nl.float32, buffer=nl.private_hbm)
    z = nl.ndarray((1, H), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.memset(z, 0.0)
    nisa.dma_copy(dst=xn_hbm[T : T + 1], src=z)
    za = nl.ndarray((1, E), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(za, 0.0)
    nisa.dma_copy(dst=aff_hbm[T : T + 1], src=za)
    shared = []
    for r0 in range(0, T, P):
        r1 = min(T, r0 + P)
        xn_sb, aff_sb, so_sb = moe_front_body(m_dst[r0:r1], post_ln_w, w_router_p, sel, w_shared_p, w_sd)
        nisa.dma_copy(dst=xn_hbm[r0:r1], src=xn_sb)
        nisa.dma_copy(dst=aff_hbm[r0:r1], src=aff_sb)
        shared.append(so_sb)
    ybuf, gidx = experts_compact_body(xn_hbm, aff_hbm, gate_up_w, gate_up_scale, down_w, down_scale, T, cap)
    combine_rows(ybuf, gidx, shared, e_src, E)
