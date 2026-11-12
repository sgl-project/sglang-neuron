"""Low-concurrency (C = 1..16) Qwen3.5 decoder layers: replicated residual, two all-reduces per layer.

Every rank holds the whole residual stream x [C, H] (fp32, token-major); no token-SP (C=1 has nothing
to shard). Per layer, on every rank:

    xn  = RMSNorm(x)                                         (replicated, ~free at C <= 16)
    mix = mixer shard(xn) @ [W_out | M_r]                    GDN TP64 x2 / attention TP32 x4 replicas,
                                                             replica weights pre-scaled 1/R (exact)
    xp  = x[:, rows_r] @ W_x,r                               this rank's H/ranks rows of the router fold
    AR1 = all_reduce([mix | mix_router + xp])                ONE collective: residual + router logits
    x1  = x + AR1[:, :H];  logits = rms(x1) * AR1[:, H:H+E]; shared gate = sigmoid(rms(x1) * AR1[:, -1])
    moe = moe_lowc(norm(x1)) + shared slice                  routed-only experts (TP32 x EP4 slices)
    AR2 = all_reduce(moe);  x2 = x1 + AR2                    ONE collective

Router fold: logits = norm(x1) @ W_r = rms(x1) * ((x + sum_r p_r) * (1 + w_post)) @ W_r, and every
mixer partial is p_r = core_r @ W_out,r, so p_r's router contribution is core_r @ M_r with
M_r = W_out,r diag(1 + w_post) W_r ([rows_r, E + 1], precomputed). The x term is split over the
ranks by hidden rows. The router (4 MB bf16, PE-bound per rank if replicated) thus becomes
~513 extra out-projection columns plus a 32-row slice per rank, and the logits arrive with AR1.

Rank-specific behaviour comes only from per-rank inputs (weights, ``rank_id``), so all ranks share
one NEFF. ``n_ranks == 1`` skips the collectives (single-rank simulation of the full layer math).
"""

import os

import nki
import nki.collectives as ncc
import nki.isa as nisa
import nki.language as nl

from nkilib.core.utils.kernel_assert import kernel_assert
from .attn_decode import attn_decode_tkg
from .attn_verify import attn_verify_tkg
from .gdn_decode import gdn_decode_tkg
from .gdn_verify import gdn_verify_tkg
from .moe_lowc import moe_lowc_body, moe_lowc_prep

P = 128
EPS = 1e-6
XCH = 512  # fp32 PSUM columns per bank
# all-reduce form: flat (one 128-rank mesh) | 2hop (8 cores of a device, then one core per device) |
# rsag (ReduceScatter + AllGather over all ranks) | 2hop_rs (RS in device, AR across devices, AG in device)
_AR = os.environ.get("QWEN35_LOWC_AR", "rsag")
# debug: qwen35_lowc_gdn_layer returns this rank's pre-all-reduce "pay" (AR1 source) or "moe" (AR2 source)
_DBG = os.environ.get("QWEN35_LOWC_DBG", "")
# Issue the MoE half's routing-independent loads (shared-expert slice, post-norm weight, the expert
# one-hot, the expert weight tiles' memsets) before the first all-reduce instead of after it.
_EARLY = os.environ.get("QWEN35_LOWC_EARLY", "1") == "1"


def _all_reduce(src, n_ranks, tag):
    """Sum ``src`` (HBM [C, N]) over all ranks; returns the HBM result."""
    if n_ranks == 1:
        return src
    shape, dt = src.shape, src.dtype
    if _AR in ("rsag", "2hop_rs") and (shape[0] * shape[1]) % n_ranks == 0:
        # all-reduce as ReduceScatter + AllGather on the flat payload viewed [n, C*N/n] (the high-C
        # path measures these for 1 MB over 128 ranks); 2hop_rs reduces inside each
        # 8-core device first, all-reduces only that 1/8 slice across devices, then gathers it back.
        n = 8 if _AR == "2hop_rs" and n_ranks > 8 and n_ranks % 8 == 0 else n_ranks
        M = shape[0] * shape[1] // n
        grp = ncc.ReplicaGroup([list(range(i, i + n)) for i in range(0, n_ranks, n)])
        part = nl.ndarray((1, M), dtype=dt, buffer=nl.private_hbm, name="lc_arp" + tag)
        ncc.reduce_scatter(dsts=[part], srcs=[src.reshape((n, M))], op=nl.add, replica_group=grp, collective_dim=0)
        if n < n_ranks:
            red = nl.ndarray((1, M), dtype=dt, buffer=nl.private_hbm, name="lc_arx" + tag)
            inter = ncc.ReplicaGroup([list(range(j, n_ranks, n)) for j in range(n)])
            ncc.all_reduce(dsts=[red], srcs=[part], op=nl.add, replica_group=inter)
            part = red
        dst = nl.ndarray((n, M), dtype=dt, buffer=nl.private_hbm, name="lc_ard" + tag)
        ncc.all_gather(dsts=[dst], srcs=[part], replica_group=grp, collective_dim=0)
        return dst.reshape(shape)
    if _AR == "2hop" and n_ranks > 8 and n_ranks % 8 == 0:
        mid = nl.ndarray(shape, dtype=dt, buffer=nl.private_hbm, name="lc_arm" + tag)
        intra = ncc.ReplicaGroup([list(range(i, i + 8)) for i in range(0, n_ranks, 8)])
        ncc.all_reduce(dsts=[mid], srcs=[src], op=nl.add, replica_group=intra)
        dst = nl.ndarray(shape, dtype=dt, buffer=nl.private_hbm, name="lc_ard" + tag)
        inter = ncc.ReplicaGroup([list(range(j, n_ranks, 8)) for j in range(8)])
        ncc.all_reduce(dsts=[dst], srcs=[mid], op=nl.add, replica_group=inter)
        return dst
    dst = nl.ndarray(shape, dtype=dt, buffer=nl.private_hbm, name="lc_ard" + tag)
    ncc.all_reduce(dsts=[dst], srcs=[src], op=nl.add, replica_group=ncc.ReplicaGroup([list(range(n_ranks))]))
    return dst


def _ln_row(w, C, H):
    """(1 + w) of a zero-centred RMSNorm weight [H], broadcast to [C, H] fp32 (SBUF)."""
    w_row = nl.ndarray((P, H), dtype=nl.float32, buffer=nl.sbuf)[:C]
    nisa.dma_copy(dst=w_row, src=w.ap(pattern=[[0, C], [1, H]], offset=0))
    nisa.tensor_scalar(dst=w_row, data=w_row, op0=nl.add, operand0=1.0)
    return w_row


def _rms(x_sb, C, H):
    """rsqrt(mean(x^2) + eps) per token of a token-major fp32 [C, H] SBUF tile -> [C, 1]."""
    sq = nl.ndarray((P, H), dtype=nl.float32, buffer=nl.sbuf)[:C]
    ss = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)[:C]
    nisa.activation_reduce(dst=sq, op=nl.square, data=x_sb, reduce_op=nl.add, reduce_res=ss)
    rs = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)[:C]
    nisa.activation(dst=rs, op=nl.rsqrt, data=ss, scale=1.0 / H, bias=EPS)
    return rs


def _norm_to_hbm(x_sb, rs, w_row, C, H, name):
    """bf16(x * rs * (1 + w)) -> SBUF tile and HBM copy (token-major)."""
    xn = nl.ndarray((P, H), dtype=nl.bfloat16, buffer=nl.sbuf)[:C]
    nisa.scalar_tensor_tensor(dst=xn, data=x_sb, op0=nl.multiply, operand0=rs, op1=nl.multiply, operand1=w_row)
    xn_hbm = nl.ndarray((C, H), dtype=nl.bfloat16, buffer=nl.private_hbm, name=name)
    nisa.dma_copy(dst=xn_hbm, src=xn)
    return xn_hbm


def fm_from_tm(x_hbm, C, H):
    """Token-major [C, H] bf16 (HBM) -> feature-major x^T [128, H/128, C] (SBUF).

    The HBM copy is re-read as [(c, t), 128] rows (512 B runs), so one PE transpose handles
    128/(H/128) tokens: 1 transpose for C <= 4 at H=4096 instead of H/128 = 32.
    """
    H1 = H // P
    kernel_assert(H1 <= P and H % P == 0, "H must be a multiple of 128 and <= 16384")
    flat = x_hbm.reshape((C * H1, P))
    xT = nl.ndarray((P, H1, C), dtype=nl.bfloat16, buffer=nl.sbuf)
    per = max(1, P // H1)
    for c0 in range(0, C, per):
        nc = min(per, C - c0)
        r = nc * H1
        blk = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.sbuf)[:r]
        nisa.dma_copy(dst=blk, src=flat[c0 * H1 : c0 * H1 + r, :])
        tp = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.psum)
        nisa.nc_transpose(dst=tp[:, 0:r], data=blk)
        for c in range(nc):
            nisa.tensor_copy(dst=xT[:, :, c0 + c], src=tp[:, c * H1 : (c + 1) * H1])
    return xT


def _x_fold(x_hbm, rank_sb, wx, C, H, src=None):
    """This rank's rows of the router fold: x[:, rows_r] (bf16) @ wx -> list of PSUM chunks [C, <=512].

    ``x_hbm`` is fp32 (low-C residual) or bf16 (high-C gathered residual); ``rank_sb`` [1, 1] int32
    selects the row slice at run time (also used by layer_decode's router fold with the TP position).
    ``src``: an explicit [C, XR] source access pattern instead (home-group layers: home rows + TP slice).
    """
    XR, NX = wx.shape
    kernel_assert(H % XR == 0 and (XR <= P or XR % P == 0), f"bad x-fold rows {XR}")
    xr = nl.ndarray((P, XR), dtype=x_hbm.dtype, buffer=nl.sbuf)[:C]
    nisa.dma_copy(dst=xr, src=src if src is not None else x_hbm.reshape((C, H // XR, XR)).select(dim=1, index=rank_sb))
    xr_bf = nl.ndarray((P, XR), dtype=nl.bfloat16, buffer=nl.sbuf)[:C]
    nisa.tensor_copy(dst=xr_bf, src=xr)
    RT = min(P, XR)
    n_rt = XR // RT
    xrT = nl.ndarray((P, n_rt, C), dtype=nl.bfloat16, buffer=nl.sbuf)[:RT]
    for t in range(n_rt):
        tp = nl.ndarray((P, C), dtype=nl.bfloat16, buffer=nl.psum)[:RT]
        nisa.nc_transpose(dst=tp, data=xr_bf[:, t * RT : (t + 1) * RT])
        nisa.tensor_copy(dst=xrT[:, t, :], src=tp)
    wx_sb = nl.ndarray((P, n_rt, NX), dtype=nl.bfloat16, buffer=nl.sbuf)[:RT]
    nisa.dma_copy(dst=wx_sb, src=wx.ap(pattern=[[NX, RT], [RT * NX, n_rt], [1, NX]], offset=0))
    chunks = []
    for c0 in range(0, NX, XCH):
        cn = min(XCH, NX - c0)
        ps = nl.ndarray((P, XCH), dtype=nl.float32, buffer=nl.psum)[:C]
        for t in range(n_rt):
            nisa.nc_matmul(ps[:, 0:cn], xrT[:, t, :], wx_sb[:, t, c0 : c0 + cn])
        chunks.append((c0, cn, ps))
    return chunks


def _shared_load(w_sh_p, w_shd, H):
    """This rank's shared-expert slice weights -> SBUF (gate | up [128, H/128, 2 Is], down [Is, H])."""
    Is = w_shd.shape[0]
    wsh = nl.ndarray((P, H // P, 2 * Is), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=wsh, src=w_sh_p)
    wsd = nl.ndarray((P, H), dtype=nl.bfloat16, buffer=nl.sbuf)[:Is]
    nisa.dma_copy(dst=wsd, src=w_shd)
    return wsh, wsd


def _shared_slice(x1n_fm, w_sh_p, w_shd, sg, C, H, w_sb=None):
    """sg * down_s(silu(x1n Wg_s) * (x1n Wu_s)) for this rank's I-slice of the shared expert -> fp32 [C, H].

    Issued before the routed experts: it needs no routing, so its PE work fills the routing /
    slot-table DMA latency, and its output becomes the routed experts' accumulator. ``w_sb``: the
    weights already loaded by ``_shared_load``.
    """
    H1 = H // P
    Is = w_shd.shape[0]
    wsh, wsd = w_sb if w_sb is not None else _shared_load(w_sh_p, w_shd, H)
    ps_s = nl.ndarray((P, 2 * Is), dtype=nl.float32, buffer=nl.psum)[:C]
    for h1 in range(H1):
        nisa.nc_matmul(ps_s, x1n_fm[:, h1, :], wsh[:, h1, :])
    s_bf = nl.ndarray((P, 2 * Is), dtype=nl.bfloat16, buffer=nl.sbuf)[:C]
    nisa.tensor_copy(dst=s_bf, src=ps_s)
    g_act = nl.ndarray((P, Is), dtype=nl.float32, buffer=nl.sbuf)[:C]
    nisa.activation(dst=g_act, op=nl.silu, data=s_bf[:, 0:Is])
    h_s = nl.ndarray((P, Is), dtype=nl.bfloat16, buffer=nl.sbuf)[:C]
    nisa.tensor_tensor(dst=h_s, data1=g_act, data2=s_bf[:, Is : 2 * Is], op=nl.multiply)
    tph = nl.ndarray((P, C), dtype=nl.bfloat16, buffer=nl.psum)[:Is]
    nisa.nc_transpose(dst=tph, data=h_s)
    hT = nl.ndarray((P, C), dtype=nl.bfloat16, buffer=nl.sbuf)[:Is]
    nisa.tensor_copy(dst=hT, src=tph)
    acc = nl.ndarray((P, H), dtype=nl.float32, buffer=nl.sbuf)[:C]
    for c0 in range(0, H, XCH):
        ps_d = nl.ndarray((P, XCH), dtype=nl.float32, buffer=nl.psum)[:C]
        nisa.nc_matmul(ps_d, hT, wsd[:, c0 : c0 + XCH])
        nisa.tensor_scalar(dst=acc[:, c0 : c0 + XCH], data=ps_d, op0=nl.multiply, operand0=sg)
    return acc


def _moe_half(x_sb, r1, post_ln_w, post_row, sh_w, pre, sel, w_gu, s_gu, w_d, s_d, w_sh_p, w_shd, n_ranks, tag, dbg, ar1_src):
    """x1 = x + AR1[:, :H] -> routed-only MoE + shared slice -> AR2 -> x2 (SBUF fp32 [C, H]).

    ``r1`` is the reduced AR1 payload in SBUF ([C, >= H + E + 1] fp32). Shared by the replicated-mixer
    layer and the home-group layer (the MoE half is per row either way).
    """
    C, H = x_sb.shape[0], x_sb.shape[1]
    E = sel.shape[0]
    hbm = nl.private_hbm
    x1 = nl.ndarray((P, H), dtype=nl.float32, buffer=nl.sbuf)[:C]
    nisa.tensor_tensor(dst=x1, data1=x_sb, data2=r1[:, 0:H], op=nl.add)

    rs1 = _rms(x1, C, H)
    x1n_hbm = _norm_to_hbm(x1, rs1, post_row if post_row is not None else _ln_row(post_ln_w, C, H), C, H, "lc_x1n" + tag)
    x1n_fm = fm_from_tm(x1n_hbm, C, H)
    logits = nl.ndarray((P, E), dtype=nl.float32, buffer=nl.sbuf)[:C]
    nisa.tensor_scalar(dst=logits, data=r1[:, H : H + E], op0=nl.multiply, operand0=rs1)
    sgl = nl.ndarray((P, 1), dtype=nl.bfloat16, buffer=nl.sbuf)[:C]
    nisa.tensor_scalar(dst=sgl, data=r1[:, H + E : H + E + 1], op0=nl.multiply, operand0=rs1)
    sg = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)[:C]
    nisa.activation(dst=sg, op=nl.sigmoid, data=sgl)

    acc = _shared_slice(x1n_fm, w_sh_p, w_shd, sg, C, H, sh_w)
    if dbg is not None:  # shared-expert part alone (the routed experts accumulate into acc in place)
        sh_hbm = nl.ndarray((C, H), dtype=nl.float32, buffer=hbm, name="lc_sh" + tag)
        nisa.dma_copy(dst=sh_hbm, src=acc)
        dbg["shared"] = sh_hbm
    acc = moe_lowc_body(x1n_fm, logits, sel, w_gu, s_gu, w_d, s_d, tag, acc, pre)
    ar2_src = nl.ndarray((C, H), dtype=nl.float32, buffer=hbm, name="lc_a2" + tag)
    nisa.dma_copy(dst=ar2_src, src=acc)
    if dbg is not None:
        lg_hbm = nl.ndarray((C, E), dtype=nl.float32, buffer=hbm, name="lc_lg" + tag)
        nisa.dma_copy(dst=lg_hbm, src=logits)
        dbg.update(pay=ar1_src, moe=ar2_src, x1n=x1n_hbm, logits=lg_hbm)
    ar2 = _all_reduce(ar2_src, n_ranks, tag + "b")

    r2 = nl.ndarray((P, H), dtype=nl.float32, buffer=nl.sbuf)[:C]
    nisa.dma_copy(dst=r2, src=ar2)
    x2 = nl.ndarray((P, H), dtype=nl.float32, buffer=nl.sbuf)[:C]
    nisa.tensor_tensor(dst=x2, data1=x1, data2=r2, op=nl.add)
    return x2


def lowc_layer_body(
    x_hbm, rank_sb, in_ln_w, post_ln_w, mixer, wx, sel, w_gu, s_gu, w_d, s_d, w_sh_p, w_shd, n_ranks, tag,
    out_buffer, dbg=None,
):
    """One low-C layer. ``mixer(xn_hbm) -> [C, H + E + 1] bf16 partial`` (HBM). Returns x2 (HBM fp32).

    ``dbg`` (a dict, tests only) receives the HBM intermediates: payload, moe partial, x1n.
    """
    C, H = x_hbm.shape
    E = sel.shape[0]
    NX = wx.shape[1]
    kernel_assert(NX == E + 1, "wx must have E + 1 columns (router | shared gate)")
    NP = H + NX
    hbm = nl.private_hbm

    # ---------------- mixer half ----------------
    x_sb = nl.ndarray((P, H), dtype=nl.float32, buffer=nl.sbuf)[:C]
    nisa.dma_copy(dst=x_sb, src=x_hbm)
    xn_hbm = _norm_to_hbm(x_sb, _rms(x_sb, C, H), _ln_row(in_ln_w, C, H), C, H, "lc_xn" + tag)
    mix = mixer(xn_hbm)
    xf = _x_fold(x_hbm, rank_sb, wx, C, H)
    pre = sh_w = post_row = None
    if _EARLY:  # MoE-half loads that need no routing: queued behind the mixer's, done before AR1 returns
        post_row = _ln_row(post_ln_w, C, H)
        sh_w = _shared_load(w_sh_p, w_shd, H)
        pre = moe_lowc_prep(sel, w_gu, w_d, C)

    # the RS + AG all-reduce forms split the flat payload over ranks: pad it to a multiple of 128
    NA = -(-NP // P) * P if _AR in ("rsag", "2hop_rs") else NP
    pay = nl.ndarray((P, NA), dtype=nl.float32, buffer=nl.sbuf)[:C]
    if NA > NP:
        nisa.memset(pay[:, NP:NA], 0.0)
    mix_sb = nl.ndarray((P, NP), dtype=nl.bfloat16, buffer=nl.sbuf)[:C]
    nisa.dma_copy(dst=mix_sb, src=mix)
    nisa.tensor_copy(dst=pay[:, 0:NP], src=mix_sb)
    for c0, cn, ps in xf:
        nisa.tensor_tensor(dst=pay[:, H + c0 : H + c0 + cn], data1=pay[:, H + c0 : H + c0 + cn], data2=ps[:, 0:cn], op=nl.add)
    ar1_src = nl.ndarray((C, NA), dtype=nl.float32, buffer=hbm, name="lc_a1" + tag)
    nisa.dma_copy(dst=ar1_src, src=pay)
    ar1 = _all_reduce(ar1_src, n_ranks, tag + "a")

    r1 = nl.ndarray((P, NA), dtype=nl.float32, buffer=nl.sbuf)[:C]
    nisa.dma_copy(dst=r1, src=ar1)
    x2 = _moe_half(x_sb, r1, post_ln_w, post_row, sh_w, pre, sel, w_gu, s_gu, w_d, s_d, w_sh_p, w_shd, n_ranks, tag, dbg, ar1_src)
    out = nl.ndarray((C, H), dtype=nl.float32, buffer=out_buffer, name="lc_out" + tag)
    nisa.dma_copy(dst=out, src=x2)
    return out


# ================= home-group mixers =================
# Each replica r of a mixer (GDN TP64 x 2, attention TP32 x 4) owns the residual rows
# r*Ch .. r*Ch + Ch - 1 ("home group"); it runs its mixer only on those rows with UNSCALED weights and
# holds only those sequences' GDN states / KV caches. AR1 then assembles the full payload:
#   rowsplit (default): ReduceScatter over the replica's TP group of its home block [Ch, NA], then
#                       AllGather over all ranks (rank order = block order) -> [R*Ch, NA]
#   pad:                the home block written at rows r*Ch of a zero [R*Ch, NA] payload, then the
#                       ordinary all-reduce (_all_reduce, QWEN35_LOWC_AR form)
# The residual between home layers is padded to Cp >= R*Ch rows (pad rows kept at zero), so the
# runtime row-block DMAs never leave the tensor.
_HOME_AR = os.environ.get("QWEN35_LOWC_HOME_AR", "rowsplit")


def home_rows(C, R, gdn=True):
    """Rows per home group for C sequences over R replicas (host and kernel agree on this).

    ceil(C / R); for GDN rounded up to a multiple of 8 above 8 rows (gdn_decode needs B % min(8, B) == 0).
    Groups past the last real row are idle (pad rows only).
    """
    ch = -(-C // R)
    if gdn and ch > 8:
        ch = -(-ch // 8) * 8
    return ch


def home_pad_rows(C, groups):
    """Padded residual rows for a stack whose mixer kinds use the given (R, is_gdn) groupings."""
    return max([C] + [R * home_rows(C, R, g) for R, g in groups])


def _home_all_reduce(pay_sb, row0_sb, Rg, n_ranks, tag):
    """AR1 of a home-group layer: pay_sb = this rank's home block [Ch, NA] fp32 (SBUF).

    Returns the HBM sum [Rg * Ch, NA] (row block r = replica r's home rows, summed over its TP group).
    n_ranks == 1 (single-rank simulation): the pad placement only (no collective).
    """
    Ch, NA = pay_sb.shape[0], pay_sb.shape[1]
    hbm = nl.private_hbm
    if n_ranks > 1 and _HOME_AR == "rowsplit":
        TP = n_ranks // Rg
        kernel_assert(TP * Rg == n_ranks and (Ch * NA) % TP == 0, "rowsplit needs n_ranks = R * TP, TP | Ch * NA")
        src = nl.ndarray((Ch, NA), dtype=nl.float32, buffer=hbm, name="lh_a1" + tag)
        nisa.dma_copy(dst=src, src=pay_sb)
        M = Ch * NA // TP
        part = nl.ndarray((1, M), dtype=nl.float32, buffer=hbm, name="lh_arp" + tag)
        if TP > 1:
            grp = ncc.ReplicaGroup([list(range(i, i + TP)) for i in range(0, n_ranks, TP)])
            ncc.reduce_scatter(dsts=[part], srcs=[src.reshape((TP, M))], op=nl.add, replica_group=grp, collective_dim=0)
        else:
            part = src.reshape((1, M))
        dst = nl.ndarray((n_ranks, M), dtype=nl.float32, buffer=hbm, name="lh_ard" + tag)
        ncc.all_gather(dsts=[dst], srcs=[part], replica_group=ncc.ReplicaGroup([list(range(n_ranks))]), collective_dim=0)
        return dst.reshape((Rg * Ch, NA)), src
    # pad: zero payload, home block at rows row0 .. row0 + Ch - 1 (runtime offset), full all-reduce
    full = nl.ndarray((Rg * Ch, NA), dtype=nl.float32, buffer=hbm, name="lh_a1" + tag)
    if Rg > 1:
        z = nl.ndarray((P, NA), dtype=nl.float32, buffer=nl.sbuf)[: Rg * Ch]
        nisa.memset(z, 0.0)
        nisa.dma_copy(dst=full, src=z)
    nisa.dma_copy(dst=full.ap(pattern=[[NA, Ch], [1, NA]], offset=0, scalar_offset=row0_sb, indirect_dim=0), src=pay_sb)
    return _all_reduce(full, n_ranks, tag + "a"), full


def lowc_home_layer_body(
    x_hbm, C, Rg, Ch, row0_sb, xf_sb, in_ln_w, post_ln_w, mixer, wx, sel, w_gu, s_gu, w_d, s_d, w_sh_p, w_shd, n_ranks,
    tag, out_buffer, out_rows, dbg=None,
):
    """One low-C layer whose mixer runs only on this replica's home rows.

    x_hbm [Cp, H] fp32: residual, rows >= C zero (Cp >= Rg * Ch; Ch = home_rows(C, Rg, kind)). ``row0_sb`` [1, 1] int32 = r * Ch (first
    home row), ``xf_sb`` [1, 1] int32 = r * Ch * H / XR + tp (this rank's x-fold block in XR-element
    units; XR = wx rows = H / TP). ``mixer(xn_home [Ch, H] bf16 HBM) -> [Ch, H + E + 1]`` with unscaled
    weights. Returns x2 as HBM fp32 [out_rows, H] (rows >= C zero).
    """
    Cp, H = x_hbm.shape
    E = sel.shape[0]
    NX = wx.shape[1]
    XR = wx.shape[0]
    kernel_assert(NX == E + 1, "wx must have E + 1 columns (router | shared gate)")
    kernel_assert(Cp >= Rg * Ch and Cp >= C, f"residual must be padded to >= {Rg * Ch} rows, got {Cp}")
    NP = H + NX
    NA = -(-NP // P) * P  # RS / AG split the flat payload: multiple of 128 columns

    # ---------------- mixer half: home rows only ----------------
    x_sb = nl.ndarray((P, H), dtype=nl.float32, buffer=nl.sbuf)[:C]
    nisa.dma_copy(dst=x_sb, src=x_hbm[0:C, :])
    xh = nl.ndarray((P, H), dtype=nl.float32, buffer=nl.sbuf)[:Ch]
    nisa.dma_copy(dst=xh, src=x_hbm.ap(pattern=[[H, Ch], [1, H]], offset=0, scalar_offset=row0_sb, indirect_dim=0))
    xn_hbm = _norm_to_hbm(xh, _rms(xh, Ch, H), _ln_row(in_ln_w, Ch, H), Ch, H, "lh_xn" + tag)
    mix = mixer(xn_hbm)
    xf_src = x_hbm.reshape((Cp * H // XR, XR)).ap(pattern=[[H, Ch], [1, XR]], offset=0, scalar_offset=xf_sb, indirect_dim=0)
    xf = _x_fold(x_hbm, None, wx, Ch, H, src=xf_src)
    pre = sh_w = post_row = None
    if _EARLY:
        post_row = _ln_row(post_ln_w, C, H)
        sh_w = _shared_load(w_sh_p, w_shd, H)
        pre = moe_lowc_prep(sel, w_gu, w_d, C)

    pay = nl.ndarray((P, NA), dtype=nl.float32, buffer=nl.sbuf)[:Ch]
    if NA > NP:
        nisa.memset(pay[:, NP:NA], 0.0)
    mix_sb = nl.ndarray((P, NP), dtype=nl.bfloat16, buffer=nl.sbuf)[:Ch]
    nisa.dma_copy(dst=mix_sb, src=mix)
    nisa.tensor_copy(dst=pay[:, 0:NP], src=mix_sb)
    for c0, cn, ps in xf:
        nisa.tensor_tensor(dst=pay[:, H + c0 : H + c0 + cn], data1=pay[:, H + c0 : H + c0 + cn], data2=ps[:, 0:cn], op=nl.add)
    ar1, ar1_src = _home_all_reduce(pay, row0_sb, Rg, n_ranks, tag)

    r1 = nl.ndarray((P, NA), dtype=nl.float32, buffer=nl.sbuf)[:C]
    nisa.dma_copy(dst=r1, src=ar1[0:C, :])
    x2 = _moe_half(x_sb, r1, post_ln_w, post_row, sh_w, pre, sel, w_gu, s_gu, w_d, s_d, w_sh_p, w_shd, n_ranks, tag, dbg, ar1_src)
    out = nl.ndarray((out_rows, H), dtype=nl.float32, buffer=out_buffer, name="lh_out" + tag)
    nisa.dma_copy(dst=out[0:C, :], src=x2)
    if out_rows > C:  # pad rows stay exactly zero (finite mixer inputs for idle / partial groups)
        z = nl.ndarray((P, H), dtype=nl.float32, buffer=nl.sbuf)[: out_rows - C]
        nisa.memset(z, 0.0)
        nisa.dma_copy(dst=out[C:out_rows, :], src=z)
    return out


def _pad_residual(x, Cp):
    """x [C, H] fp32 -> HBM [Cp, H] with zero rows C .. Cp - 1 (x itself when Cp == C)."""
    C, H = x.shape
    if Cp == C:
        return x
    xp = nl.ndarray((Cp, H), dtype=nl.float32, buffer=nl.private_hbm, name="lh_xpad")
    t = nl.ndarray((P, H), dtype=nl.float32, buffer=nl.sbuf)[:Cp]
    nisa.memset(t, 0.0)
    nisa.dma_copy(dst=t[0:C, :], src=x)
    nisa.dma_copy(dst=xp, src=t)
    return xp


def _load_scalar(t, k):
    """Element k of an HBM [1, n] int32 tensor -> SBUF [1, 1]."""
    r = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=r, src=t[:, k : k + 1])
    return r


def _load_rank(rank_id):
    r = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=r, src=rank_id)
    return r


@nki.jit
def qwen35_lowc_gdn_layer(
    x,  # [C, H] fp32 residual (replicated on every rank)
    rank_id,  # [1, 1] int32 this rank's index (selects its router-fold rows)
    in_ln_w, post_ln_w,  # [H] fp32 zero-centred norm weights
    w_proj_p, conv_w, conv_state, a_log, dt_bias, norm_w, rec_state,  # GDN shard (gdn_decode_tkg layouts)
    w_out_ext,  # [Vd_r, H + E + 1] bf16: replica-scaled [W_out,r | M_r]
    wx,  # [H / n_ranks, E + 1] bf16 router-fold rows of this rank
    sel, w_gu, s_gu, w_d, s_d,  # routed experts (moe_lowc layouts)
    w_sh_p,  # [128, H/128, 2 * Is] bf16 shared gate | up slice (rows pre-permuted)
    w_shd,  # [Is, H] bf16 shared down slice
    n_ranks: int,
):
    """One GDN + MoE layer at low concurrency (correctness entry). Returns (x_out, new_conv, new_rec)."""
    states = {}

    def mixer(xn):
        mix, nc, nr = gdn_decode_tkg(xn, w_proj_p, conv_w, conv_state, a_log, dt_bias, norm_w, rec_state, w_out_ext)
        states.update(conv=nc, rec=nr)
        return mix

    dbg = {} if _DBG else None
    out = lowc_layer_body(
        x, _load_rank(rank_id), in_ln_w, post_ln_w, mixer, wx, sel, w_gu, s_gu, w_d, s_d, w_sh_p, w_shd, n_ranks,
        "_L0", nl.shared_hbm, dbg,
    )
    if _DBG:  # return this rank's pre-all-reduce intermediate in place of x_out
        src = dbg[_DBG]
        out = nl.ndarray(src.shape, dtype=src.dtype, buffer=nl.shared_hbm, name="lc_dbg")
        t = nl.ndarray((P, src.shape[1]), dtype=src.dtype, buffer=nl.sbuf)[: src.shape[0]]
        nisa.dma_copy(dst=t, src=src)
        nisa.dma_copy(dst=out, src=t)
    return out, states["conv"], states["rec"]


@nki.jit
def qwen35_lowc_stack(
    x, rank_id, in_ln_w, post_ln_w,
    w_proj_p, conv_w, conv_state, a_log, dt_bias, norm_w, rec_state, w_out_ext,  # GDN shard
    w_qkv, q_norm_w, k_norm_w, w_gate_p, cos, sin, k_cache_t, v_cache, pos, w_o_ext,  # attention shard
    wx, sel, w_gu, s_gu, w_d, s_d, w_sh_p, w_shd,
    n_ranks: int,
    num_layers: int,
    kv_slot=None,  # [C, 1] int32 KV-pool slot per row (caches [NSLOT, ...]); None: row b -> slot b
    kv_write="flat",  # attn_decode_tkg new-token K/V write mode with kv_slot
):
    """Qwen3.5 layer pattern at low concurrency in one NEFF (every 4th layer gated attention).

    Timing scaffold like ``qwen35_hybrid_stack``: layers of a kind share weights / states / KV cache,
    but every layer issues all of its own DMAs and both all-reduces.
    """
    rank_sb = _load_rank(rank_id)
    cur = x
    for i in range(num_layers):
        if i % 4 == 3:
            def mixer(xn):
                mix, _, _ = attn_decode_tkg(
                    xn, w_qkv, q_norm_w, k_norm_w, w_gate_p, cos, sin, k_cache_t, v_cache, pos, w_o_ext,
                    slot=kv_slot, kv_write=kv_write,
                )
                return mix
        else:
            def mixer(xn):
                mix, _, _ = gdn_decode_tkg(xn, w_proj_p, conv_w, conv_state, a_log, dt_bias, norm_w, rec_state, w_out_ext)
                return mix
        cur = lowc_layer_body(
            cur, rank_sb, in_ln_w, post_ln_w, mixer, wx, sel, w_gu, s_gu, w_d, s_d, w_sh_p, w_shd, n_ranks,
            f"_L{i}", nl.shared_hbm if i == num_layers - 1 else nl.private_hbm,
        )
    return cur


@nki.jit
def qwen35_lowc_verify_stack(
    x, rank_id, in_ln_w, post_ln_w,
    w_proj_p, conv_w, conv_in, a_log, dt_bias, norm_w, rec_in, src_slot, w_out_ext,  # GDN shard (verify)
    w_qkv, q_norm_w, k_norm_w, w_gate_p, cos, sin, k_cache_t, v_cache, pos, w_o_ext,  # attention shard
    wx, sel, w_gu, s_gu, w_d, s_d, w_sh_p, w_shd,
    n_ranks: int,
    num_layers: int,
):
    """MTP verify step at low concurrency: S = k + 1 rows per sequence (x [C * S, H], sequence-major).

    The low-C layer is row-wise outside the token mixer (router fold, MoE, both all-reduces), so only
    the mixers change: gdn_verify_tkg (S sequential steps, per-step state snapshots, slot-indexed
    restart) and attn_verify_tkg (S causal queries, KV written at pos + s), both with the folded
    out-projection. Timing scaffold like ``qwen35_lowc_stack``.
    """
    rank_sb = _load_rank(rank_id)
    cur = x
    for i in range(num_layers):
        if i % 4 == 3:
            def mixer(xn):
                mix, _, _ = attn_verify_tkg(xn, w_qkv, q_norm_w, k_norm_w, w_gate_p, cos, sin, k_cache_t, v_cache, pos, w_o_ext)
                return mix
        else:
            def mixer(xn):
                mix, _, _ = gdn_verify_tkg(xn, w_proj_p, conv_w, conv_in, a_log, dt_bias, norm_w, rec_in, src_slot, w_out_ext)
                return mix
        cur = lowc_layer_body(
            cur, rank_sb, in_ln_w, post_ln_w, mixer, wx, sel, w_gu, s_gu, w_d, s_d, w_sh_p, w_shd, n_ranks,
            f"_L{i}", nl.shared_hbm if i == num_layers - 1 else nl.private_hbm,
        )
    return cur


@nki.jit
def qwen35_lowc_home_gdn_layer(
    x,  # [C, H] fp32 residual (replicated on every rank)
    home,  # [1, 2] int32: (r * Ch, r * Ch * H / XR + tp) of this rank's GDN replica r / TP position tp
    in_ln_w, post_ln_w,
    w_proj_p, conv_w, conv_state, a_log, dt_bias, norm_w, rec_state,  # GDN shard; states: the Ch home rows only
    w_out_ext,  # [Vd_r, H + E + 1] bf16: UNSCALED [W_out,r | M_r]
    wx,  # [H / TP, E + 1] bf16 router-fold rows of TP position tp
    sel, w_gu, s_gu, w_d, s_d, w_sh_p, w_shd,
    n_ranks: int,
    groups: int,  # GDN replicas R (n_ranks = R * TP; any R with n_ranks == 1: one simulated replica)
):
    """Home-group GDN + MoE layer at low concurrency (correctness entry). Returns (x_out [C, H], new_conv, new_rec)."""
    C, H = x.shape
    Ch = home_rows(C, groups, True)
    kernel_assert(conv_state.shape[0] == Ch, f"GDN states must hold the {Ch} home rows")
    states = {}

    def mixer(xn):
        mix, nc, nr = gdn_decode_tkg(xn, w_proj_p, conv_w, conv_state, a_log, dt_bias, norm_w, rec_state, w_out_ext)
        states.update(conv=nc, rec=nr)
        return mix

    xp = _pad_residual(x, groups * Ch)
    out = lowc_home_layer_body(
        xp, C, groups, Ch, _load_scalar(home, 0), _load_scalar(home, 1), in_ln_w, post_ln_w, mixer, wx, sel, w_gu, s_gu,
        w_d, s_d, w_sh_p, w_shd, n_ranks, "_L0", nl.shared_hbm, C,
    )
    return out, states["conv"], states["rec"]


@nki.jit
def qwen35_lowc_home_stack(
    x, home,  # home [1, 4] int32: (GDN row0, GDN x-fold block, attention row0, attention x-fold block)
    in_ln_w, post_ln_w,
    w_proj_p, conv_w, conv_state, a_log, dt_bias, norm_w, rec_state, w_out_ext,  # GDN shard, Ch_g home rows
    w_qkv, q_norm_w, k_norm_w, w_gate_p, cos, sin, k_cache_t, v_cache, pos, w_o_ext,  # attention shard, Ch_a home rows
    wx_g, wx_a,  # router-fold rows: [H / TP_g, E + 1] (GDN layers), [H / TP_a, E + 1] (attention layers)
    sel, w_gu, s_gu, w_d, s_d, w_sh_p, w_shd,
    n_ranks: int,
    num_layers: int,
    gdn_groups: int,
    attn_groups: int,
):
    """``qwen35_lowc_stack`` with home-group mixers (timing scaffold; same sharing of weights / states)."""
    C, H = x.shape
    Ch_g = home_rows(C, gdn_groups, True)
    Ch_a = home_rows(C, attn_groups, False)
    kernel_assert(conv_state.shape[0] == Ch_g and k_cache_t.shape[0] == Ch_a, "states / KV must hold the home rows")
    Cp = home_pad_rows(C, [(gdn_groups, True), (attn_groups, False)])
    sc = [_load_scalar(home, k) for k in range(4)]
    cur = _pad_residual(x, Cp)
    for i in range(num_layers):
        last = i == num_layers - 1
        if i % 4 == 3:
            def mixer(xn):
                mix, _, _ = attn_decode_tkg(xn, w_qkv, q_norm_w, k_norm_w, w_gate_p, cos, sin, k_cache_t, v_cache, pos, w_o_ext)
                return mix
            R, Ch, r0, xf, wx = attn_groups, Ch_a, sc[2], sc[3], wx_a
        else:
            def mixer(xn):
                mix, _, _ = gdn_decode_tkg(xn, w_proj_p, conv_w, conv_state, a_log, dt_bias, norm_w, rec_state, w_out_ext)
                return mix
            R, Ch, r0, xf, wx = gdn_groups, Ch_g, sc[0], sc[1], wx_g
        cur = lowc_home_layer_body(
            cur, C, R, Ch, r0, xf, in_ln_w, post_ln_w, mixer, wx, sel, w_gu, s_gu, w_d, s_d, w_sh_p, w_shd, n_ranks,
            f"_L{i}", nl.shared_hbm if last else nl.private_hbm, C if last else Cp,
        )
    return cur
