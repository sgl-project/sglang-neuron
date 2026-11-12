"""Routed-only MoE for low concurrency (C = 1..16 tokens), one rank of a TP32 x EP4 expert layout.

At low C only ~10*C of the 512 experts are selected per layer, so instead of running every local
expert with zero affinities (the C=128 path), this kernel:

  1. routes in-kernel (softmax, exactly top-10 with lowest-id tie-break, renormalise) from the
     router logits of the C tokens;
  2. compacts the experts of its EP group that at least one token selected into slots 0..n-1
     (prefix sum over the active mask on the PE) -> per-tile slot -> expert id / affinity tables;
  3. runs ONE loop with a runtime trip count over tiles of TS (8 or 16) slots. Per tile: every
     slot's gate/up and down slices are fetched by runtime expert id (``select`` on the expert dim);
     gate/up uses the MX-quantised tokens as the stationary shared by all slots; h = silu(g) * aff * u
     stays in SBUF; the TS slots' 32-row down slices stack into ONE MX tile (slot k on partitions
     8k..8k+7 with its own scale block), so the 4096 down output columns stream once per tile instead
     of once per expert. Tile t+1's gate/up slices load during tile t's down and its down slices
     during tile t+1's gate/up (prologue / loop / epilogue).

Unused slots of the last tile point at local expert 0 with affinity 0 (finite weights, zero h): a
little DMA / PE, never a different result. Weight layouts: the arguments of ``moe_lowc`` below.
The fp32 partial [C, H] is summed over all ranks by the layer's MoE all-reduce.
"""

import os

import nki
import nki.isa as nisa
import nki.language as nl

from nkilib.core.utils.kernel_assert import kernel_assert
from .moe_front import TOPK

P = 128
Q = 4  # x4 packing
S = 32  # I-slice per rank (= one MX scale block of the down projection)
PS = S // Q  # partitions per slot in the stacked down tile
SPT = P // PS  # max slots per stacked down tile (16)
GCH = 8  # slots per gate/up PSUM tile (8 x 64 fp32 columns = one bank)
OCH = 512  # down output columns per PSUM tile (fp32, one bank)


# --- trace-time switches (read once at import) ---
# Queue of the per-slot weight DMAs: auto (compiler's choice, all HWDGE) | sp (all on the Sync-engine
# HWDGE ring) | spread (gate/up data alternates SP / ACT HWDGE per slot, its 4 scale-row DMAs go to
# the other ring, the down slice + down scale to the GpSimd SWDGE ring) | spread2 (as spread, but
# down on SP / ACT too, no SWDGE)
_WQ = os.environ.get("QWEN35_LOWC_WQ", "spread")
# Unused slots of the last tile get an out-of-range expert id and their weight DMAs are skipped
# (oob_mode=skip) instead of fetching local expert 0: at C=1 a group has ~2.5-4.5 of 8 slots in use.
_SKIP = os.environ.get("QWEN35_LOWC_SKIP", "1") == "1"
# One register load per slot, shared by its 7 weight DMAs (else every DMA reloads the id).
_REG = os.environ.get("QWEN35_LOWC_REG", "1") == "1"
# Tile 0's expert ids come straight from the slot table in SBUF (no HBM write + read hop).
_ID0 = os.environ.get("QWEN35_LOWC_ID0", "1") == "1"
# Down slices of all a tile's slots as ONE per-partition-row gather DMA (SWDGE, row index on partition
# p = id[p // 8] * 8 + p % 8) and the down scales as one 4-row gather per SBUF quadrant, instead of two
# select DMAs per slot. The row indices come from the tile's id row broadcast over partitions (PE).
# Off by default. This path is supported in the simulator only, not in a 128-rank compile.
_DGATHER = os.environ.get("QWEN35_LOWC_DGATHER", "0") == "1"

def _wdma(dst, src, q):
    """Weight DMA on queue q (auto | sp | act | pool); unused-slot ids are skipped when _SKIP."""
    oob = nisa.oob_mode.skip if _SKIP else nisa.oob_mode.error
    if q == "pool":
        nisa.dma_copy(dst=dst, src=src, dge_mode=nisa.dge_mode.swdge, oob_mode=oob)
    elif q == "auto":
        nisa.dma_copy(dst=dst, src=src, dge_mode=nisa.dge_mode.hwdge, oob_mode=oob)
    else:
        eng = nisa.engine.sync if q == "sp" else nisa.engine.scalar
        nisa.dma_copy(dst=dst, src=src, dge_mode=nisa.dge_mode.hwdge, engine=eng, oob_mode=oob)


def _queues(k):
    """(gate/up data, gate/up scales, down + down scale) queues of slot k under _WQ."""
    if _WQ in ("auto", "sp", "act", "pool"):
        return _WQ, _WQ, _WQ
    a, b = ("sp", "act") if k % 2 == 0 else ("act", "sp")
    return a, b, ("pool" if _WQ == "spread" else b)


def tile_slots(C):
    """Slots per loop iteration: 8 for C <= 2 (~10C selections, at most ~5-8 per group), else 16."""
    return 8 if C <= 2 else SPT


def route_topk(logits_sb, C, E):
    """[C, E] fp32 logits (SBUF, base partition 0) -> renormalised top-10 affinities [C, E] fp32."""
    logit_bf = nl.ndarray((P, E), dtype=nl.bfloat16, buffer=nl.sbuf)[:C]
    nisa.tensor_copy(dst=logit_bf, src=logits_sb)  # HF router logits are bf16
    mx = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)[:C]
    nisa.tensor_reduce(dst=mx, op=nl.max, data=logit_bf, axis=1)
    nisa.tensor_scalar(dst=mx, data=mx, op0=nl.multiply, operand0=-1.0)
    ex = nl.ndarray((P, E), dtype=nl.float32, buffer=nl.sbuf)[:C]
    ssum = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)[:C]
    nisa.activation_reduce(dst=ex, op=nl.exp, data=logit_bf, reduce_op=nl.add, reduce_res=ssum, bias=mx)
    rsum = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)[:C]
    nisa.reciprocal(dst=rsum, data=ssum)
    probs = nl.ndarray((P, E), dtype=nl.float32, buffer=nl.sbuf)[:C]
    nisa.tensor_scalar(dst=probs, data=ex, op0=nl.multiply, operand0=rsum)
    # Exactly TOPK experts, ties to the lower expert id: bf16 logits tie often near the 10th place
    # (~0.02 spacing at |logit| ~ 2), and a ">= 10th" threshold would then keep 11+ experts. Select on
    # probs * (1 - e * 2^-21) (exact factors, a relative nudge <= 2.4e-4, far below the ratio between
    # distinct bf16 logits unless |logit| < ~0.03); the affinities still use the unperturbed probs.
    tb_i = nl.ndarray((P, E), dtype=nl.int32, buffer=nl.sbuf)[:C]
    nisa.iota(tb_i, pattern=[[1, E]], offset=0, channel_multiplier=0)
    tb = nl.ndarray((P, E), dtype=nl.float32, buffer=nl.sbuf)[:C]
    nisa.tensor_scalar(dst=tb, data=tb_i, op0=nl.multiply, operand0=-(2.0**-21), op1=nl.add, operand1=1.0)
    pt = nl.ndarray((P, E), dtype=nl.float32, buffer=nl.sbuf)[:C]
    nisa.tensor_tensor(dst=pt, data1=probs, data2=tb, op=nl.multiply)
    top8 = nl.ndarray((P, 8), dtype=nl.float32, buffer=nl.sbuf)[:C]
    nisa.max8(dst=top8, src=pt)
    rest = nl.ndarray((P, E), dtype=nl.float32, buffer=nl.sbuf)[:C]
    nisa.nc_match_replace8(dst=rest, data=pt, vals=top8, imm=-1.0)
    next8 = nl.ndarray((P, 8), dtype=nl.float32, buffer=nl.sbuf)[:C]
    nisa.max8(dst=next8, src=rest)
    thr = next8[:, TOPK - 9 : TOPK - 8]  # 10th largest (nudged) probability
    keep = nl.ndarray((P, E), dtype=nl.float32, buffer=nl.sbuf)[:C]
    nisa.tensor_scalar(dst=keep, data=pt, op0=nl.greater_equal, operand0=thr)
    sel_p = nl.ndarray((P, E), dtype=nl.float32, buffer=nl.sbuf)[:C]
    ksum = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)[:C]
    nisa.tensor_tensor(dst=sel_p, data1=probs, data2=keep, op=nl.multiply)
    nisa.tensor_reduce(dst=ksum, op=nl.add, data=sel_p, axis=1)
    nisa.reciprocal(dst=ksum, data=ksum)
    aff = nl.ndarray((P, E), dtype=nl.float32, buffer=nl.sbuf)[:C]
    nisa.tensor_scalar(dst=aff, data=sel_p, op0=nl.multiply, operand0=ksum)
    return aff


def _count_reg(n_sb, n_max, step, skip=0):
    """Register holding max(ceil(n / step) - skip, 0) for an exact-integer fp32 n in SBUF [1, 1]."""
    m = n_max // step
    thr_i = nl.ndarray((1, m), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(thr_i, pattern=[[step, m]], offset=step * skip, channel_multiplier=0)
    thr = nl.ndarray((1, m), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=thr, src=thr_i)
    lt = nl.ndarray((1, m), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=lt, data=thr, op0=nl.less, operand0=n_sb)  # step*i < n
    cnt = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=cnt, op=nl.add, data=lt, axis=1)
    cnt_i = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=cnt_i, src=cnt)
    reg = nisa.register_alloc()
    nisa.register_load(reg, cnt_i)
    return reg


def load_sel(sel):
    """[E, E_L] fp32 one-hot (HBM) -> SBUF [128, E/128, E_L] (no routing dependency: issue early)."""
    E, E_L = sel.shape
    n_et = E // P
    sel_sb = nl.ndarray((P, n_et, E_L), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=sel_sb, src=sel.ap(pattern=[[E_L, P], [P * E_L, n_et], [1, E_L]], offset=0))
    return sel_sb


def slot_tables(aff, sel_sb, C, E, E_L, TS, tag):
    """Compact this group's selected experts into slots.

    aff [C, E] fp32 (SBUF), sel_sb = load_sel(sel). Returns (ids_hbm [E_L/TS, 1, TS] int32 slot -> local
    expert, affs_hbm [E_L/TS, C, TS] fp32 slot affinities, register max(ceil(n/TS) - 1, 0), ids_i [1, E_L]
    int32 SBUF copy of the id table). With _SKIP, unused slots get id E_L (out of range).
    """
    ET = P
    n_et = E // ET
    hbm = nl.private_hbm
    # local affinities, expert-major: affTl[l, t] = sum_e sel[e, l] * aff[t, e]
    affT = nl.ndarray((P, n_et, C), dtype=nl.float32, buffer=nl.sbuf)
    for et in range(n_et):
        tp = nl.ndarray((P, C), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=tp, data=aff[:, et * ET : (et + 1) * ET])
        nisa.tensor_copy(dst=affT[:, et, :], src=tp)
    ps_l = nl.ndarray((P, C), dtype=nl.float32, buffer=nl.psum)[:E_L]
    for et in range(n_et):
        nisa.nc_matmul(ps_l, sel_sb[:, et, :], affT[:, et, :])
    affTl = nl.ndarray((P, C), dtype=nl.float32, buffer=nl.sbuf)[:E_L]
    nisa.tensor_copy(dst=affTl, src=ps_l)

    # active mask and exclusive prefix sum (slot position) over local experts
    amax = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)[:E_L]
    nisa.tensor_reduce(dst=amax, op=nl.max, data=affTl, axis=1)
    act = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)[:E_L]
    nisa.tensor_scalar(dst=act, data=amax, op0=nl.greater, operand0=0.0)
    tri_i = nl.ndarray((P, E_L), dtype=nl.int32, buffer=nl.sbuf)[:E_L]
    nisa.iota(tri_i, pattern=[[1, E_L]], offset=0, channel_multiplier=-1)  # f - p
    tri = nl.ndarray((P, E_L), dtype=nl.float32, buffer=nl.sbuf)[:E_L]
    nisa.tensor_scalar(dst=tri, data=tri_i, op0=nl.greater, operand0=0)  # tri[l', l] = l' < l
    ps_pos = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.psum)[:E_L]
    nisa.nc_matmul(ps_pos, tri, act)
    pos = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)[:E_L]
    nisa.tensor_copy(dst=pos, src=ps_pos)
    ones = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)[:E_L]
    nisa.memset(ones, 1.0)
    ps_n = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(ps_n, act, ones)
    n_sb = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=n_sb, src=ps_n)

    # one-hot slot matrix M[l, k] = act[l] * (pos[l] == k)
    NS = E_L
    kk_i = nl.ndarray((P, NS), dtype=nl.int32, buffer=nl.sbuf)[:E_L]
    nisa.iota(kk_i, pattern=[[1, NS]], offset=0, channel_multiplier=0)
    kk = nl.ndarray((P, NS), dtype=nl.float32, buffer=nl.sbuf)[:E_L]
    nisa.tensor_copy(dst=kk, src=kk_i)
    msl = nl.ndarray((P, NS), dtype=nl.float32, buffer=nl.sbuf)[:E_L]
    nisa.tensor_scalar(dst=msl, data=kk, op0=nl.equal, operand0=pos, op1=nl.multiply, operand1=act)

    # slot -> local expert id (row on partition 0), slot affinities token-major [C, NS]
    lid_i = nl.ndarray((P, 1), dtype=nl.int32, buffer=nl.sbuf)[:E_L]
    nisa.iota(lid_i, pattern=[[0, 1]], offset=0, channel_multiplier=1)
    lid = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)[:E_L]
    nisa.tensor_copy(dst=lid, src=lid_i)
    ps_id = nl.ndarray((1, NS), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(ps_id, lid, msl)
    ids_i = nl.ndarray((1, NS), dtype=nl.int32, buffer=nl.sbuf)
    if _SKIP:  # unused slots -> id E_L (out of range: their weight DMAs are skipped)
        ps_u = nl.ndarray((1, NS), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(ps_u, ones, msl)
        off = nl.ndarray((1, NS), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=off, data=ps_u, op0=nl.multiply, operand0=-float(E_L), op1=nl.add, operand1=float(E_L))
        nisa.tensor_tensor(dst=ids_i, data1=ps_id, data2=off, op=nl.add)
    else:
        nisa.tensor_copy(dst=ids_i, src=ps_id)
    ids_hbm = nl.ndarray((NS // TS, 1, TS), dtype=nl.int32, buffer=hbm, name="lc_ids" + tag)
    nisa.dma_copy(dst=ids_hbm.reshape((1, NS)), src=ids_i)

    ps_a = nl.ndarray((P, NS), dtype=nl.float32, buffer=nl.psum)[:C]
    nisa.nc_matmul(ps_a, affTl, msl)
    affs = nl.ndarray((P, NS), dtype=nl.float32, buffer=nl.sbuf)[:C]
    nisa.tensor_copy(dst=affs, src=ps_a)
    affs_hbm = nl.ndarray((NS // TS, C, TS), dtype=nl.float32, buffer=hbm, name="lc_affs" + tag)
    nisa.dma_copy(dst=affs_hbm.ap(pattern=[[TS, C], [C * TS, NS // TS], [1, TS]], offset=0), src=affs)

    return ids_hbm, affs_hbm, _count_reg(n_sb, NS, TS, skip=1), ids_i


def quantize_x_fm(xT, C, Tq):
    """Feature-major bf16 x^T [128, H/128, C] (SBUF) -> MX (xq, xs) [128, H/512, Tq] ("h" packing)."""
    H1 = xT.shape[1]
    n_h = H1 // Q
    swz = nl.ndarray((P, n_h, Tq, Q), dtype=nl.bfloat16, buffer=nl.sbuf)
    if Tq > C:
        nisa.memset(swz, 0.0)
    for j in range(Q):  # x4 slot j of (p, tile) = hidden column j*H/4 + tile*128 + p
        nisa.tensor_copy(dst=swz[:, :, 0:C, j], src=xT[:, j * n_h : (j + 1) * n_h, :])
    xq = nl.ndarray((P, n_h, Tq), dtype=nl.float8_e4m3fn_x4, buffer=nl.sbuf)
    xs = nl.ndarray((P, n_h, Tq), dtype=nl.uint8, buffer=nl.sbuf)
    nisa.quantize_mx(dst=xq, src=swz.reshape((P, n_h, Tq * Q)), dst_scale=xs)
    return xq, xs


def moe_lowc_prep(sel, w_gu, w_d, C):
    """Routing-independent part of moe_lowc_body: the one-hot ``sel`` load and the weight tiles with
    their memsets (GpSimd). Call it before anything the routing waits on (e.g. before the layer's first
    all-reduce) so none of it sits between the logits and the first expert weight DMA.
    """
    E_L, _, n_h, _ = w_gu.shape
    H = w_d.shape[2]
    TS = tile_slots(C)
    TP = TS * PS
    g = nisa.engine.gpsimd
    # Weight tiles live outside the loop: tile t+1's gate/up slices stream in while tile t's down
    # runs and its down slices while tile t+1's gate/up runs (one buffer each; the compiler orders
    # the write-after-read). MX scale rows other than 32q..32q+3 are never written (kept 0); with
    # _SKIP the slots a tile does not use keep the previous tile's (finite) weights, or 0 in tile 0.
    wgu = nl.ndarray((P, n_h, TS * 2 * S), dtype=w_gu.dtype, buffer=nl.sbuf)
    wgs = nl.ndarray((P, n_h, TS * 2 * S), dtype=nl.uint8, buffer=nl.sbuf)
    wd = nl.ndarray((P, H), dtype=w_d.dtype, buffer=nl.sbuf)
    wds = nl.ndarray((P, H), dtype=nl.uint8, buffer=nl.sbuf)
    nisa.memset(wgs, 0, engine=g)
    nisa.memset(wds[0:TP], 0, engine=g)
    if _SKIP:
        nisa.memset(wgu, 0, engine=g)
        nisa.memset(wd[0:TP], 0, engine=g)
    pre = {"sel_sb": load_sel(sel), "wgu": wgu, "wgs": wgs, "wd": wd, "wds": wds}
    if _DGATHER:
        pre.update(_gather_masks(TS))
    return pre


def _gather_masks(TS):
    """Constants for the down gathers of a TS-slot tile: md[p, k] = (p // 8 == k), ms[p, k] = (p == scale
    row of slot k), rmod[p] = p % 8, ones_row [1, 128]."""
    v_i = nl.ndarray((P, TS), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(v_i, pattern=[[-PS, TS]], offset=0, channel_multiplier=1)  # p - 8k
    v = nl.ndarray((P, TS), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=v, src=v_i)
    lo = nl.ndarray((P, TS), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=lo, data=v, op0=nl.greater_equal, operand0=0.0)
    md = nl.ndarray((P, TS), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=md, data=v, op0=nl.less, operand0=float(PS))
    nisa.tensor_tensor(dst=md, data1=md, data2=lo, op=nl.multiply)
    pid_i = nl.ndarray((P, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(pid_i, pattern=[[0, 1]], offset=0, channel_multiplier=1)
    pid = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=pid, src=pid_i)
    ms = nl.ndarray((P, TS), dtype=nl.float32, buffer=nl.sbuf)
    for k in range(TS):
        nisa.tensor_scalar(dst=ms[:, k : k + 1], data=pid, op0=nl.equal, operand0=float(32 * (k // 4) + k % 4))
    rmod = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=rmod, op=nl.add, data=lo, axis=1)  # number of k with p - 8k >= 0 = p // 8 + 1
    nisa.tensor_scalar(dst=rmod, data=rmod, op0=nl.multiply, operand0=-float(PS), op1=nl.add, operand1=float(PS))
    nisa.tensor_tensor(dst=rmod, data1=rmod, data2=pid, op=nl.add)  # p - 8 * (p // 8)  (valid for p < 8 TS)
    ones_row = nl.ndarray((1, P), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(ones_row, 1.0)
    return {"md": md, "ms": ms, "rmod": rmod, "ones_row": ones_row}


def moe_lowc_body(xT, logits_sb, sel, w_gu, s_gu, w_d, s_d, tag, acc=None, pre=None):
    """Routed-only MoE partial for this rank. xT bf16 [128, H/128, C], logits_sb fp32 [C, E] (SBUF).

    Returns the fp32 partial [C, H] (SBUF), to be summed over ranks by the MoE all-reduce. ``acc``
    (optional fp32 [C, H] SBUF, e.g. the shared-expert output) is accumulated into in place. ``pre``:
    the result of an earlier ``moe_lowc_prep(sel, w_gu, w_d, C)`` (else it is done here).
    """
    H1, C = xT.shape[1], xT.shape[2]
    H = H1 * P
    E = logits_sb.shape[1]
    E_L = w_gu.shape[0]
    n_h = w_gu.shape[2]
    kernel_assert(C <= P and E % P == 0 and TOPK <= E, f"need C <= 128 and E % 128 == 0 (C={C}, E={E})")
    kernel_assert(w_gu.shape[1] == P and w_gu.shape[3] == 2 * S and n_h * Q * P == H, "w_gu must be [E_L, 128, H/512, 64]")
    kernel_assert(w_d.shape[1] == PS and w_d.shape[2] == H and s_d.shape[1] == 1, "w_d must be [E_L, 8, H]")
    kernel_assert(E_L % SPT == 0 and E_L <= P and sel.shape[1] == E_L, f"E_L must be a multiple of {SPT}, <= 128")
    TS = tile_slots(C)
    TP = TS * PS  # partitions of the stacked down tile in use (64 or 128: an MX stationary size)
    Tq = C + C % 2  # MX stationaries take an even token count (pad column is zero)
    if pre is None:
        pre = moe_lowc_prep(sel, w_gu, w_d, C)
    wgu, wgs, wd, wds = pre["wgu"], pre["wgs"], pre["wd"], pre["wds"]
    swz = nl.ndarray((P, 1, Tq, Q), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.memset(swz, 0.0, engine=nisa.engine.gpsimd)
    cnt = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)  # current tile
    nisa.memset(cnt, 0)
    nxt = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)  # tile being prefetched
    nisa.memset(nxt, 1)

    aff = route_topk(logits_sb, C, E)
    ids_hbm, affs_hbm, reg_t, ids_i = slot_tables(aff, pre["sel_sb"], C, E, E_L, TS, tag)
    xq, xs = quantize_x_fm(xT, C, Tq)

    if acc is None:
        acc = nl.ndarray((P, H), dtype=nl.float32, buffer=nl.sbuf)[:C]
        nisa.memset(acc, 0.0)

    def load_ids(index):
        ids = nl.ndarray((1, TS), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=ids, src=ids_hbm.select(dim=0, index=index), dge_mode=nisa.dge_mode.hwdge)
        return ids

    def with_idx(ids):
        """Per-slot runtime index (one register per slot shared by its DMAs, or the SBUF scalar); with
        _DGATHER also the down-gather row indices. Returns (indices, gather rows or None)."""
        idx = []
        for k in range(TS):
            eid = ids[0:1, k : k + 1]
            if _REG:
                reg = nisa.register_alloc()
                nisa.register_load(reg, eid)
                eid = reg
            idx.append(eid)
        rows = None
        if _DGATHER:
            # B[p, k] = id[k] on every partition (ids <= E_L are exact through the PE)
            idf = nl.ndarray((1, TS), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=idf, src=ids)
            ps_b = nl.ndarray((P, TS), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(ps_b, pre["ones_row"], idf)
            sel_b = nl.ndarray((P, TS), dtype=nl.float32, buffer=nl.sbuf)
            dsum = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=sel_b, data1=ps_b, data2=pre["md"], op=nl.multiply)
            nisa.tensor_reduce(dst=dsum, op=nl.add, data=sel_b, axis=1)
            drow = nl.ndarray((P, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=drow, data=dsum, op0=nl.multiply, operand0=float(PS), op1=nl.add, operand1=pre["rmod"])
            sel_s = nl.ndarray((P, TS), dtype=nl.float32, buffer=nl.sbuf)
            ssum = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=sel_s, data1=ps_b, data2=pre["ms"], op=nl.multiply)
            nisa.tensor_reduce(dst=ssum, op=nl.add, data=sel_s, axis=1)
            srow = nl.ndarray((P, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=srow, src=ssum)
            rows = (drow, srow)
        return idx, rows

    def issue_gu(idx, rows=None):
        for k in range(TS):
            qd, qs, _ = _queues(k)
            cols = slice(k * 2 * S, (k + 1) * 2 * S)
            _wdma(wgu[:, :, cols], w_gu.select(dim=0, index=idx[k]), qd)
            sg = s_gu.select(dim=0, index=idx[k])
            # rows 4q + i -> partition 32q + i: four DMAs (an SBUF access pattern has ONE partition
            # dim with unit step, so a two-level partition pattern is rejected by NKI)
            for q in range(4):
                _wdma(wgs[32 * q : 32 * q + 4, :, cols], sg.slice(dim=0, start=4 * q, end=4 * q + 4), qs)

    def issue_dn(idx, rows=None):
        if rows is not None:
            drow, srow = rows
            oob = nisa.oob_mode.skip if _SKIP else nisa.oob_mode.error
            src = w_d.reshape((E_L * PS, H)).ap(pattern=[[H, TP], [1, H]], offset=0, vector_offset=drow[0:TP], indirect_dim=0)
            nisa.dma_copy(dst=wd[0:TP, :], src=src, dge_mode=nisa.dge_mode.swdge, oob_mode=oob)
            for q in range(TS // 4):
                src = s_d.reshape((E_L, H)).ap(pattern=[[H, 4], [1, H]], offset=0, vector_offset=srow[32 * q : 32 * q + 4], indirect_dim=0)
                nisa.dma_copy(dst=wds[32 * q : 32 * q + 4, :], src=src, dge_mode=nisa.dge_mode.swdge, oob_mode=oob)
            return
        for k in range(TS):
            _, _, qn = _queues(k)
            _wdma(wd[PS * k : PS * (k + 1), :], w_d.select(dim=0, index=idx[k]), qn)
            r = 32 * (k // 4) + k % 4  # scale block k -> SBUF scale row
            _wdma(wds[r : r + 1, :], s_d.select(dim=0, index=idx[k]), qn)

    def compute_tile(nxt_idx):
        """acc += this tile's experts; prefetches tile ``nxt`` when its slot indices are given."""
        a_t = nl.ndarray((P, TS), dtype=nl.float32, buffer=nl.sbuf)[:C]
        nisa.dma_copy(dst=a_t, src=affs_hbm.select(dim=0, index=cnt), dge_mode=nisa.dge_mode.hwdge)
        # gate/up: one stationary per H-tile, all slots streamed as moving columns
        n_g = (TS + GCH - 1) // GCH
        ps_g = [nl.ndarray((P, GCH * 2 * S), dtype=nl.float32, buffer=nl.psum)[:Tq] for _ in range(n_g)]
        for th in range(n_h):
            for gi in range(n_g):
                c0, c1 = gi * GCH * 2 * S, min(TS, (gi + 1) * GCH) * 2 * S
                nisa.nc_matmul_mx(
                    dst=ps_g[gi][:, 0 : c1 - c0], stationary=xq[:, th, :], moving=wgu[:, th, c0:c1],
                    stationary_scale=xs[:, th, :], moving_scale=wgs[:, th, c0:c1],
                )
        if nxt_idx is not None:
            issue_gu(*nxt_idx)
        h = nl.ndarray((P, TS * S), dtype=nl.bfloat16, buffer=nl.sbuf)[:C]
        for k in range(TS):
            gi, kk = k // GCH, k % GCH
            g = ps_g[gi][0:C, kk * 2 * S : kk * 2 * S + S]
            u = ps_g[gi][0:C, kk * 2 * S + S : (kk + 1) * 2 * S]
            sg_act = nl.ndarray((P, S), dtype=nl.float32, buffer=nl.sbuf)[:C]
            nisa.activation(dst=sg_act, op=nl.silu, data=g)
            nisa.scalar_tensor_tensor(
                dst=h[:, k * S : (k + 1) * S], data=sg_act, op0=nl.multiply, operand0=a_t[:, k : k + 1],
                op1=nl.multiply, operand1=u,
            )
        # stack: x4 slot j of partition p = h column 4p + j (slot p // 8, its row 4*(p % 8) + j)
        for j in range(Q):
            tp = nl.ndarray((P, C), dtype=nl.bfloat16, buffer=nl.psum)[:TP]
            nisa.nc_transpose(dst=tp, data=h.ap(pattern=[[TS * S, C], [Q, TP]], offset=j))
            nisa.tensor_copy(dst=swz[0:TP, 0, 0:C, j], src=tp)
        hq = nl.ndarray((P, 1, Tq), dtype=nl.float8_e4m3fn_x4, buffer=nl.sbuf)
        hs = nl.ndarray((P, 1, Tq), dtype=nl.uint8, buffer=nl.sbuf)
        nisa.quantize_mx(dst=hq, src=swz.reshape((P, 1, Tq * Q)), dst_scale=hs)
        # down: the stacked tile streams the H output columns once (only its TP loaded partitions)
        for c0 in range(0, H, OCH):
            ps_o = nl.ndarray((P, OCH), dtype=nl.float32, buffer=nl.psum)[:Tq]
            nisa.nc_matmul_mx(
                dst=ps_o, stationary=hq[0:TP, 0, :], moving=wd[0:TP, c0 : c0 + OCH],
                stationary_scale=hs[0:TP, 0, :], moving_scale=wds[0:TP, c0 : c0 + OCH],
            )
            nisa.tensor_tensor(dst=acc[:, c0 : c0 + OCH], data1=acc[:, c0 : c0 + OCH], data2=ps_o[0:C, :], op=nl.add)
        if nxt_idx is not None:
            issue_dn(*nxt_idx)

    # prologue: tile 0's weights; loop: tiles 0 .. T-2, each prefetching the next; epilogue: the last
    # tile (also run when no expert of this group is active: zero affinities, no effect)
    idx0 = with_idx(ids_i[0:1, 0:TS] if _ID0 else load_ids(cnt))
    issue_gu(*idx0)
    issue_dn(*idx0)

    def body(_):
        compute_tile(with_idx(load_ids(nxt)))
        nisa.tensor_scalar(dst=cnt, data=cnt, op0=nl.add, operand0=1)
        nisa.tensor_scalar(dst=nxt, data=nxt, op0=nl.add, operand0=1)

    nl.fori_loop(0, reg_t, body)
    compute_tile(None)
    return acc


@nki.jit
def moe_lowc(
    xT,  # [128, H/128, C] bf16 normed tokens, feature-major
    logits,  # [C, E] fp32 router logits
    sel,  # [E, E_L] fp32 one-hot: global expert -> this group's local index
    w_gu,  # [E_L, 128, H/512, 64] x4 (gate | up of this rank's 32-wide I-slice)
    s_gu,  # [E_L, 16, H/512, 64] uint8
    w_d,  # [E_L, 8, H] x4 (down rows of the slice, 4 consecutive rows per x4 element)
    s_d,  # [E_L, 1, H] uint8
):
    H1, C = xT.shape[1], xT.shape[2]
    E = logits.shape[1]
    x_sb = nl.ndarray((P, H1, C), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=x_sb, src=xT)
    lg = nl.ndarray((P, E), dtype=nl.float32, buffer=nl.sbuf)[:C]
    nisa.dma_copy(dst=lg, src=logits)
    acc = moe_lowc_body(x_sb, lg, sel, w_gu, s_gu, w_d, s_d, "_m")
    out = nl.ndarray((C, H1 * P), dtype=nl.float32, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=out, src=acc)
    return out
