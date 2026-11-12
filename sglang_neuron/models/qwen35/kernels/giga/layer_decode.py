"""Qwen3.5 GDN + MoE decoder layer per rank, with its collectives (NKI, LNC=1).

    x_shard [T_shard, H]                                   residual stream, token-SP over all ranks
    -> AllGather[TP]  -> RMSNorm_zc -> GDN decode (TP head shard) -> ReduceScatter[TP] -> + x
    -> AllGather[EP]  -> moe_front (RMSNorm, router top-10, shared slice)
                      -> moe_experts_mx (local experts) + shared  -> ReduceScatter[EP] -> + resid
    -> out_shard [T_shard, H]

The TP group is this rank's mixer group (e.g. 8 consecutive ranks), the EP group spans all
ranks, exactly the GPT-OSS giga-kernel token bookkeeping.

Router fold (QWEN35_ROUTER_FOLD=1, the low-C trick of lowc_layer.py at high concurrency): with
x1 = x + sum_r p_r (residual + the mixer partials the TP ReduceScatter sums),
    logits = norm(x1) @ W_r = rms(x1) * (x @ Wf + sum_r p_r @ Wf),   Wf = diag(1 + w_post) [W_r | w_sg]
so each mixer rank appends M_r = W_out,r @ Wf as E + 1 extra out-projection columns (``w_out_ext``)
and adds its H/tp rows of the residual term, x[:, rows_tp] @ Wf[rows_tp] (``wx``, rows picked at
run time by the TP position ``tp_rank``), before the ReduceScatter. The widened [T, H + E + 1] rows
[resid1 | unscaled logits] then ride the EP AllGather, and moe_front only applies rms(x1): the 4 MB
replicated router matmul (PE time per 128-row block on every rank) disappears. ``_layer_body`` is plain traced
code so the mega kernel can inline it once per layer (buffer names carry a per-layer tag:
NKI needs unique names within one NEFF). The validated sub-kernels are called as nested
kernels with HBM hand-offs; fusing those hand-offs into SBUF is a later optimisation.
"""

import os

import nki
import nki.collectives as ncc
import nki.isa as nisa
import nki.language as nl
from nki.collectives import ReplicaGroup

from .attn_decode import attn_cp_combine, attn_cp_core, attn_cp_front, attn_cp_tail, attn_decode_tkg
from .gdn_decode import gdn_decode_body, gdn_decode_tkg, gdn_load_weights
from .lowc_layer import _load_rank, _x_fold
from .moe_compact import moe_compact_body
from .moe_experts_mx import _PREFETCH, experts_body, issue_expert_weights, moe_experts_mx, prefetch_expert_weights
from .moe_front import moe_front, moe_front_body, rmsnorm_zc_tok
from nkilib.core.utils.kernel_assert import kernel_assert

P = 128
# Number of local experts whose weights are prefetched at the top of each layer (0 = off).
_MOE_PREFETCH = int(os.environ.get("QWEN35_MOE_PREFETCH", "0"))
# MoE combine ReduceScatter in OCP FP8 E4M3 (halves the largest collective's payload).
_FP8_RS = os.environ.get("QWEN35_FP8_RS", "0") == "1"
# Fused MoE block (front end + experts in one body, SBUF hand-offs) and how many experts it prefetches.
_MOE_FUSED = os.environ.get("QWEN35_MOE_FUSED", "1") == "1"
_MOE_FUSED_PRE = int(os.environ.get("QWEN35_MOE_FUSED_PRE", "0"))
# Where the fused block issues those expert weight DMAs: "front" (after the EP AllGather, before the
# router), "moe" (top of the MoE half, before the AllGather) or "layer" (before the token mixer).
_MOE_PRE_AT = os.environ.get("QWEN35_MOE_PRE_AT", "front")
# Hand HBM tensors straight to the collectives instead of first copying them into a fresh source
# buffer (each copy is one more serial DMA hop on the critical path between layers).
_DIRECT_CC = os.environ.get("QWEN35_DIRECT_CC", "0") == "1"
# SBUF-resident residual stream between blocks and layers: AllGathers land in SBUF, the MoE
# ReduceScatter runs SBUF -> SBUF, residual adds stay in SBUF (GPT-OSS giga kernel defaults).
_SBUF_CC = os.environ.get("QWEN35_SBUF_CC", "0") == "1"
# Experts quantise their input from the router's feature-major tokens (no second set of transposes).
_REUSE_XT = os.environ.get("QWEN35_REUSE_XT", "1") == "1"
# Router fold: router logits computed by the mixer out-projections + an x row slice per TP rank and
# carried through the TP ReduceScatter / EP AllGather (module docstring). Needs the fold inputs.
_FOLD = os.environ.get("QWEN35_ROUTER_FOLD", "1") == "1"
# Routed-row compaction for > 128 EP-gathered rows (moe_compact): the union of the rows routed to this
# rank's experts goes through the experts once instead of one full expert pass per 128-row block.
_COMPACT = os.environ.get("QWEN35_MOE_COMPACT", "1") == "1"
# > 128 rows per GDN TP group: per-block AllGather / GDN / ReduceScatter micro-batches (collectives of
# block b overlap block b+1's GDN) instead of one AllGather and one ReduceScatter around all blocks.
_GDN_PIPE = os.environ.get("QWEN35_GDN_PIPE", "1") == "1"
# MoE half as two micro-batches (two EP AllGathers / ReduceScatters around one compacted expert pass).
_MOE_PIPE = os.environ.get("QWEN35_MOE_PIPE", "1") == "1"
# Micro-batched GDN mixer: load the GDN weights once per layer for all row blocks and keep each block's
# mixer partial in SBUF (inlined gdn_decode_body) instead of one nested kernel call per block.
_GDN_WSHARE = os.environ.get("QWEN35_GDN_WSHARE", "1") == "1"


_INTER = nl.sbuf if _SBUF_CC else nl.private_hbm  # buffer of the residual stream between layers


def _is_sbuf(t):
    return getattr(t, "buffer", None) == nl.sbuf


def _rows_sb(x, T, H):
    """SBUF [T, H] view of x (loaded from HBM if needed)."""
    if _is_sbuf(x):
        return x
    t = nl.ndarray((P, H), dtype=x.dtype, buffer=nl.sbuf)[:T]
    nisa.dma_copy(dst=t, src=x)
    return t


def _moe_pipe_ok(T_shard, T_ep):
    """_moe_half runs as two micro-batches (_moe_half_pipe) for this geometry."""
    return _MOE_PIPE and _MOE_FUSED and _COMPACT and not _FP8_RS and T_shard % 2 == 0 and (T_ep // 2) % P == 0


def moe_row_valid(row_valid, T_shard, ep_size, tag=""):
    """Per-row MoE validity, loaded ONCE per launch: [fp32 [rows_b, 1] SBUF 1 / 0] per MoE front block, in the
    order _moe_half hands the EP-gathered rows to moe_front_body (its ``valid`` argument). None -> None.

    row_valid: int32 [T_ep, 1] HBM, the same on every rank of the EP group, in EP-gathered order (EP position r
    owns rows r * T_shard ..), 1 = real row, 0 = padding. A padding row gets all-zero local affinities on
    every rank: it is routed nowhere, takes no compaction slot and never overflows the cap.
      flat (one EP AllGather): front block b = gathered rows b * 128 ..: row_valid[b * 128 ..] as is;
      micro-batched (_moe_half_pipe): half h gathers rows h * q .. h * q + q - 1 of every EP position
        (q = T_shard / 2), so half h's row j = r * q + i is row_valid[r * T_shard + h * q + i]: reordered
        by one strided HBM -> HBM DMA per half."""
    if row_valid is None:
        return None
    T_ep = T_shard * ep_size
    kernel_assert(tuple(row_valid.shape) == (T_ep, 1), f"row_valid must be [T_ep = {T_ep}, 1], got {row_valid.shape}")

    def tiles(src, nrows):
        out = []
        for r0 in range(0, nrows, P):
            n = min(P, nrows - r0)
            vi = nl.ndarray((P, 1), dtype=nl.int32, buffer=nl.sbuf)[:n]
            nisa.dma_copy(dst=vi, src=src[r0 : r0 + n])
            vf = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)[:n]
            nisa.tensor_scalar(dst=vf, data=vi, op0=nl.greater, operand0=0)  # exact 1.0 / 0.0
            out.append(vf)
        return out

    if _SBUF_CC or not _moe_pipe_ok(T_shard, T_ep):
        return tiles(row_valid, T_ep)
    q = T_shard // 2
    Th = q * ep_size
    out = []
    for h in range(2):
        vh = nl.ndarray((Th, 1), dtype=nl.int32, buffer=nl.private_hbm, name=f"moe_rv{tag}h{h}")
        nisa.dma_copy(
            dst=vh.ap(pattern=[[q, ep_size], [1, q]], offset=0),
            src=row_valid.ap(pattern=[[T_shard, ep_size], [1, q]], offset=h * q),
        )
        out += tiles(vh, Th)
    return out


def _add_rows(dst_hbm, a_hbm, b_hbm, T, H):
    """dst = a + b for token-major [T, H] HBM tensors (bf16 math; DMAs cast to/from other dtypes)."""
    a = nl.ndarray((P, H), dtype=nl.bfloat16, buffer=nl.sbuf)[:T]
    b = nl.ndarray((P, H), dtype=nl.bfloat16, buffer=nl.sbuf)[:T]
    nisa.dma_copy(dst=a, src=a_hbm)
    nisa.dma_copy(dst=b, src=b_hbm)
    s = nl.ndarray((P, H), dtype=dst_hbm.dtype, buffer=nl.sbuf)[:T]
    nisa.tensor_tensor(dst=s, data1=a, data2=b, op=nl.add)
    nisa.dma_copy(dst=dst_hbm, src=s)


def _gathered_norm(x_shard, in_ln_w, group, size, tag, sbuf_out=False, keep_x=False):
    """AllGather the residual shard over ``group`` and apply the input RMSNorm -> bf16 HBM [T, H].

    ``keep_x``: return (normed, gathered x HBM [T, H]) (the router fold reads x's row slice)."""
    T_shard, H = x_shard.shape
    T_tp = T_shard * size
    dt = x_shard.dtype
    hbm = nl.private_hbm
    if _SBUF_CC:
        x_src = _rows_sb(x_shard, T_shard, H)
        x_sb = nl.ndarray((P, H), dtype=dt, buffer=nl.sbuf)[:T_tp]
        ncc.all_gather(dsts=[x_sb], srcs=[x_src], replica_group=group, collective_dim=0)
        return _norm_to_hbm(x_sb, in_ln_w, T_tp, H, tag, keep_sbuf=sbuf_out)
    a_dst = nl.ndarray((T_tp, H), dtype=dt, buffer=hbm, name="a_dst" + tag)
    if _DIRECT_CC and tag != "_L0":  # layer 0's input is a kernel argument: keep the copy there
        a_src = x_shard
    else:
        a_src = nl.ndarray((T_shard, H), dtype=dt, buffer=hbm, name="a_src" + tag)
        nisa.dma_copy(dst=a_src, src=x_shard)
    ncc.all_gather(dsts=[a_dst], srcs=[a_src], replica_group=group, collective_dim=0)
    x_sb = nl.ndarray((P, H), dtype=dt, buffer=nl.sbuf)[:T_tp]
    nisa.dma_copy(dst=x_sb, src=a_dst)
    xn_hbm = _norm_to_hbm(x_sb, in_ln_w, T_tp, H, tag)
    return (xn_hbm, a_dst) if keep_x else xn_hbm


def _norm_to_hbm(x_sb, in_ln_w, T_tp, H, tag, keep_sbuf=False):
    """Input RMSNorm of gathered SBUF rows -> bf16 HBM [T_tp, H] (the mixer kernels read HBM)."""
    hbm = nl.private_hbm
    w_row = nl.ndarray((P, H), dtype=nl.float32, buffer=nl.sbuf)[:T_tp]
    nisa.dma_copy(dst=w_row, src=in_ln_w.ap(pattern=[[0, T_tp], [1, H]], offset=0))
    nisa.tensor_scalar(dst=w_row, data=w_row, op0=nl.add, operand0=1.0)
    xn = rmsnorm_zc_tok(x_sb, w_row, T_tp, H, 1e-6, nl.bfloat16)
    if keep_sbuf:
        return xn
    xn_hbm = nl.ndarray((T_tp, H), dtype=nl.bfloat16, buffer=hbm, name="xn" + tag)
    nisa.dma_copy(dst=xn_hbm, src=xn)
    return xn_hbm


def _scatter_residual(x_shard, mix_part, group, size, tag):
    """ReduceScatter the mixer partials over ``group`` and add the residual -> HBM [T_shard, H]."""
    T_shard, H = x_shard.shape
    dt = x_shard.dtype
    hbm = nl.private_hbm
    if _SBUF_CC:
        mix_sb = _rows_sb(mix_part, T_shard * size, H)  # collectives need src and dst in one buffer type
        rs_sb = nl.ndarray((P, H), dtype=dt, buffer=nl.sbuf)[:T_shard]
        ncc.reduce_scatter(dsts=[rs_sb], srcs=[mix_sb], op=nl.add, replica_group=group, collective_dim=0)
        resid1 = nl.ndarray((P, H), dtype=dt, buffer=nl.sbuf)[:T_shard]
        nisa.tensor_tensor(dst=resid1, data1=_rows_sb(x_shard, T_shard, H), data2=rs_sb, op=nl.add)
        return resid1
    # collectives need src and dst in the same buffer type (the mixer kernels return shared HBM)
    rs_dst = nl.ndarray((T_shard, H), dtype=dt, buffer=nl.shared_hbm if _DIRECT_CC else hbm, name="rs_dst" + tag)
    if _DIRECT_CC:
        rs_src = mix_part
    else:
        rs_src = nl.ndarray((T_shard * size, H), dtype=dt, buffer=hbm, name="rs_src" + tag)
        nisa.dma_copy(dst=rs_src, src=mix_part)
    ncc.reduce_scatter(dsts=[rs_dst], srcs=[rs_src], op=nl.add, replica_group=group, collective_dim=0)
    resid1 = nl.ndarray((T_shard, H), dtype=dt, buffer=hbm, name="resid1" + tag)
    _add_rows(resid1, x_shard, rs_dst, T_shard, H)
    return resid1


def _fold_x_rows(x_tp, fold):
    """Residual term of the router fold, x[:, rows_tp] @ Wf[rows_tp] -> fp32 SBUF [T_tp, E + 1].

    Issued before the mixer (the PE is idle while the mixer's first weight slab loads); evacuated
    to SBUF at once so its PSUM banks are free for the mixer.
    """
    _, wx, rank_sb = fold
    T_tp, H = x_tp.shape
    NX = wx.shape[1]
    xf = nl.ndarray((P, NX), dtype=nl.float32, buffer=nl.sbuf)[:T_tp]
    for c0, cn, ps in _x_fold(x_tp, rank_sb, wx, T_tp, H):
        nisa.tensor_copy(dst=xf[:, c0 : c0 + cn], src=ps[:, 0:cn])
    return xf


def _fold_partial(mix_part, xf, H, dt, tag, buffer=nl.private_hbm):
    """Widened mixer partial [T_tp, H + E + 1] with this rank's x-row term added to the router columns
    (one bf16 rounding of the sum); the residual columns go HBM -> HBM in parallel."""
    T_tp, W = mix_part.shape
    NX = W - H
    rs_src = nl.ndarray((T_tp, W), dtype=dt, buffer=buffer, name="rs_src" + tag)
    nisa.dma_copy(dst=rs_src[:, 0:H], src=mix_part[:, 0:H])
    mr = nl.ndarray((P, NX), dtype=nl.bfloat16, buffer=nl.sbuf)[:T_tp]
    nisa.dma_copy(dst=mr, src=mix_part[:, H:W])
    ms = nl.ndarray((P, NX), dtype=dt, buffer=nl.sbuf)[:T_tp]
    nisa.tensor_tensor(dst=ms, data1=mr, data2=xf, op=nl.add)
    nisa.dma_copy(dst=rs_src[:, H:W], src=ms)
    return rs_src


def _fold_partial_sb(o_sb, xf, H, dt, tag):
    """_fold_partial for a mixer partial already in SBUF ([rows, H + E + 1] bf16): no HBM round trip."""
    T_tp, W = o_sb.shape
    rs_src = nl.ndarray((T_tp, W), dtype=dt, buffer=nl.private_hbm, name="rs_src" + tag)
    nisa.dma_copy(dst=rs_src[:, 0:H], src=o_sb[:, 0:H])
    ms = nl.ndarray((P, W - H), dtype=dt, buffer=nl.sbuf)[:T_tp]
    nisa.tensor_tensor(dst=ms, data1=o_sb[:, H:W], data2=xf, op=nl.add)
    nisa.dma_copy(dst=rs_src[:, H:W], src=ms)
    return rs_src


def _scatter_fold(x_shard, mix_part, xf, group, size, tag):
    """Router-fold ReduceScatter: widened partials [T_tp, H + E + 1] -> [resid1 | unscaled logits] HBM.

    Returns private HBM [T_shard, H + E + 1] (collectives need src and dst in one buffer type).
    """
    T_shard, H = x_shard.shape
    W = mix_part.shape[1]
    dt = x_shard.dtype
    hbm = nl.private_hbm
    rs_src = _fold_partial(mix_part, xf, H, dt, tag)
    rs_dst = nl.ndarray((T_shard, W), dtype=dt, buffer=hbm, name="rs_dst" + tag)
    ncc.reduce_scatter(dsts=[rs_dst], srcs=[rs_src], op=nl.add, replica_group=group, collective_dim=0)
    # resid1 = x + rs[:, :H] (bf16 math, as _add_rows); the logit columns pass through
    a = nl.ndarray((P, H), dtype=nl.bfloat16, buffer=nl.sbuf)[:T_shard]
    nisa.dma_copy(dst=a, src=x_shard)
    b = nl.ndarray((P, W), dtype=nl.bfloat16, buffer=nl.sbuf)[:T_shard]
    nisa.dma_copy(dst=b, src=rs_dst)
    s = nl.ndarray((P, W), dtype=dt, buffer=nl.sbuf)[:T_shard]
    nisa.tensor_tensor(dst=s[:, 0:H], data1=a, data2=b[:, 0:H], op=nl.add)
    nisa.tensor_copy(dst=s[:, H:W], src=b[:, H:W])
    resid1 = nl.ndarray((T_shard, W), dtype=dt, buffer=hbm, name="resid1" + tag)
    nisa.dma_copy(dst=resid1, src=s)
    return resid1


def _fold_inputs(w_ext, wx, rank, w_plain, what):
    """(w_ext, wx, rank SBUF [1, 1]) when the router fold is on and its inputs are given, else None."""
    if not _FOLD or w_ext is None:
        return None
    kernel_assert(
        wx is not None and rank is not None,
        f"the router fold needs all folded {what} inputs (out-projection, x-fold rows, TP position)",
    )
    kernel_assert(not _SBUF_CC and _MOE_FUSED, "QWEN35_ROUTER_FOLD needs the fused MoE block and HBM collectives")
    kernel_assert(w_ext.shape[0] == w_plain.shape[0], f"folded {what} out-projection rows differ")
    kernel_assert(w_ext.shape[1] == w_plain.shape[1] + wx.shape[1], f"folded {what} out-projection width")
    return (w_ext, wx, _load_rank(rank))


def _attn_mixer_cp(
    x_shard, in_ln_w, w_qkv, q_norm_w, k_norm_w, w_gate_p, cos, sin, k_cache_t, v_cache, pos, w_o, atp_group,
    atp_size, tag, cp_rank, cp_sink, block_table, pool_off=0, fold=None,
):
    """DCP attention mixer (kv_cp = atp_size > 1; patch 0009): each rank's attn_cp_front/core
    over its share of every page, then the in-layer merge — AllGather q, AllGather (M, L), the combine
    scale, a ReduceScatter over the head rows — and the gate + o_proj tail. Same residual/fold contract as
    _attn_mixer. ``cp_rank`` [1,1] int32 = this rank's atp position; ``cp_sink`` [1,1] int32 = the pool row
    the non-owners' new-token writes drop into."""
    G = atp_size
    hbm = nl.private_hbm
    if fold is not None:
        xn_hbm, x_tp = _gathered_norm(x_shard, in_ln_w, atp_group, atp_size, tag, keep_x=True)
        xf = _fold_x_rows(x_tp, fold)
    else:
        xn_hbm = _gathered_norm(x_shard, in_ln_w, atp_group, atp_size, tag)
    q_loc, k_new, v_new = attn_cp_front(xn_hbm, w_qkv, q_norm_w, k_norm_w, cos, sin)
    B, ndl = q_loc.shape
    q_src = nl.ndarray((B, ndl), dtype=nl.bfloat16, buffer=hbm, name="cp_q_src" + tag)
    nisa.dma_copy(dst=q_src, src=q_loc)
    q_g = nl.ndarray((G * B, ndl), dtype=nl.bfloat16, buffer=hbm, name="cp_q_ag" + tag)  # rows g*B + b
    ncc.all_gather(dsts=[q_g], srcs=[q_src], replica_group=atp_group, collective_dim=0)
    q_all = nl.ndarray((B, G * ndl), dtype=nl.bfloat16, buffer=hbm, name="cp_q_all" + tag)  # token-major
    nisa.dma_copy(
        dst=q_all.ap(pattern=[[G * ndl, B], [ndl, G], [1, ndl]], offset=0),
        src=q_g.ap(pattern=[[ndl, B], [B * ndl, G], [1, ndl]], offset=0),
    )
    o_hat, ms, _, _ = attn_cp_core(
        q_all, k_new, v_new, k_cache_t, v_cache, pos, cp_rank, cp_sink, block_table, pool_off=pool_off,
        cp_size=G,
    )
    R, d = o_hat.shape
    ms_src = nl.ndarray((2, R), dtype=nl.float32, buffer=hbm, name="cp_ms_src" + tag)
    nisa.dma_copy(dst=ms_src, src=ms)
    ms_all = nl.ndarray((G * 2, R), dtype=nl.float32, buffer=hbm, name="cp_ms_ag" + tag)  # rank g: rows 2g, 2g+1
    ncc.all_gather(dsts=[ms_all], srcs=[ms_src], replica_group=atp_group, collective_dim=0)
    scaled = attn_cp_combine(o_hat, ms_all, cp_rank, cp_size=G)
    o_loc = nl.ndarray((R // G, d), dtype=nl.bfloat16, buffer=hbm, name="cp_o_loc" + tag)  # this rank's own heads
    ncc.reduce_scatter(dsts=[o_loc], srcs=[scaled], op=nl.add, replica_group=atp_group, collective_dim=0)
    mix_part = attn_cp_tail(o_loc, xn_hbm, w_gate_p, fold[0] if fold is not None else w_o)
    if fold is not None:
        return _scatter_fold(x_shard, mix_part, xf, atp_group, atp_size, tag)
    return _scatter_residual(x_shard, mix_part, atp_group, atp_size, tag)


def _attn_mixer(
    x_shard, in_ln_w, w_qkv, q_norm_w, k_norm_w, w_gate_p, cos, sin, k_cache_t, v_cache, pos, w_o, atp_group,
    atp_size, tag, fold=None, kv_slot=None, kv_write="flat", block_table=None, pool_off=0, kv_cp=1,
    cp_rank=None, cp_sink=None,
):
    """Gated full attention over the attention TP group (heads sharded, 1 KV head per rank).
    ``block_table`` [B, MP] int32 or None: the rows' pages of the page-pool caches (attn_decode_tkg's paged KV, "dp"
    pages; patch 0004), in place of ``kv_slot``.

    ``fold`` = (w_o_ext, wx, rank_sb) or None: router fold (returns [resid1 | unscaled logits]).
    ``kv_slot`` [B, 1] int32 or None: KV-pool slot of each attention row (attn_decode_tkg ``slot``;
    None = row b in slot b), ``kv_write`` its new-token write mode.
    ``kv_cp`` > 1: decode context parallelism over the atp group (_attn_mixer_cp; the caches are the 4-D
    physical-block pool, ``cp_rank`` this rank's atp position, ``cp_sink`` the sink pool row)."""
    if kv_cp > 1:
        kernel_assert(block_table is not None and cp_rank is not None and cp_sink is not None,
                      "kv_cp > 1 needs block_table, cp_rank and cp_sink")
        kernel_assert(kv_cp == atp_size, f"kv_cp must equal the atp group size, got {kv_cp} != {atp_size}")
        return _attn_mixer_cp(
            x_shard, in_ln_w, w_qkv, q_norm_w, k_norm_w, w_gate_p, cos, sin, k_cache_t, v_cache, pos, w_o,
            atp_group, atp_size, tag, cp_rank, cp_sink, block_table, pool_off=pool_off, fold=fold,
        )
    if fold is not None:
        xn_hbm, x_tp = _gathered_norm(x_shard, in_ln_w, atp_group, atp_size, tag, keep_x=True)
        xf = _fold_x_rows(x_tp, fold)
        mix_part, _, _ = attn_decode_tkg(
            xn_hbm, w_qkv, q_norm_w, k_norm_w, w_gate_p, cos, sin, k_cache_t, v_cache, pos, fold[0],
            slot=kv_slot, kv_write=kv_write, block_table=block_table, pool_off=pool_off,
        )
        return _scatter_fold(x_shard, mix_part, xf, atp_group, atp_size, tag)
    xn_hbm = _gathered_norm(x_shard, in_ln_w, atp_group, atp_size, tag)
    mix_part, k_cache_t, v_cache = attn_decode_tkg(
        xn_hbm, w_qkv, q_norm_w, k_norm_w, w_gate_p, cos, sin, k_cache_t, v_cache, pos, w_o,
        slot=kv_slot, kv_write=kv_write, block_table=block_table, pool_off=pool_off,
    )
    return _scatter_residual(x_shard, mix_part, atp_group, atp_size, tag)


def _gdn_mixer(
    x_shard, in_ln_w, w_proj_p, conv_w, conv_state, a_log, dt_bias, norm_w, rec_state, w_out, tp_group,
    tp_size, tag, fold=None, inplace=False, row0=0,
):
    """Gated DeltaNet over the GDN TP group; returns (resid1, new_conv, new_rec).

    ``fold`` = (w_out_ext, wx, rank_sb) or None: router fold (resid1 is [resid1 | unscaled logits]).
    ``inplace``: the new states overwrite conv_state / rec_state (multi-step decode carries them).
    ``row0``: this call's states are rows row0 .. row0 + T_tp - 1 of conv_state / rec_state (a layer-major
    state pool: GDN layer g's block starts at g * slots, multistep.py)."""
    T_tp = x_shard.shape[0] * tp_size
    if T_tp > P:
        kernel_assert(fold is not None, "more than 128 rows per GDN TP group needs the router fold path")
        return _gdn_mixer_rows(
            x_shard, in_ln_w, w_proj_p, conv_w, conv_state, a_log, dt_bias, norm_w, rec_state, tp_group, tp_size,
            tag, fold, inplace, row0,
        )
    if row0 or conv_state.shape[0] != T_tp:
        conv_state = conv_state[row0 : row0 + T_tp]
    if fold is not None:
        xn_hbm, x_tp = _gathered_norm(x_shard, in_ln_w, tp_group, tp_size, tag, keep_x=True)
        xf = _fold_x_rows(x_tp, fold)
        mix_part, new_conv, new_rec = gdn_decode_tkg(
            xn_hbm, w_proj_p, conv_w, conv_state, a_log, dt_bias, norm_w, rec_state, fold[0], row0, inplace
        )
        return _scatter_fold(x_shard, mix_part, xf, tp_group, tp_size, tag), new_conv, new_rec
    if _SBUF_CC:  # inlined body: normed tokens in, mixer partials out, both SBUF
        xn_sb = _gathered_norm(x_shard, in_ln_w, tp_group, tp_size, tag, sbuf_out=True)
        mix_part, new_conv, new_rec = gdn_decode_body(
            xn_sb, w_proj_p, conv_w, conv_state, a_log, dt_bias, norm_w, rec_state, w_out, rec_row0=row0,
            inplace=inplace,
        )
        return _scatter_residual(x_shard, mix_part, tp_group, tp_size, tag), new_conv, new_rec
    xn_hbm = _gathered_norm(x_shard, in_ln_w, tp_group, tp_size, tag)
    mix_part, new_conv, new_rec = gdn_decode_tkg(
        xn_hbm, w_proj_p, conv_w, conv_state, a_log, dt_bias, norm_w, rec_state, w_out, row0, inplace
    )
    return _scatter_residual(x_shard, mix_part, tp_group, tp_size, tag), new_conv, new_rec


def _gdn_mixer_rows(
    x_shard, in_ln_w, w_proj_p, conv_w, conv_state, a_log, dt_bias, norm_w, rec_state, tp_group, tp_size, tag,
    fold, inplace=False, row0=0,
):
    """_gdn_mixer (router fold) for > 128 rows per TP group: one AllGather and one ReduceScatter, the
    norm / x-fold / GDN kernel per 128-row block of sequences (their states are row slices)."""
    T_shard, H = x_shard.shape
    T_tp = T_shard * tp_size
    dt = x_shard.dtype
    hbm = nl.private_hbm
    W = fold[0].shape[1]
    nb = -(-T_tp // P)
    if _GDN_PIPE and T_shard % nb == 0 and (T_shard // nb) * tp_size == P:
        return _gdn_mixer_pipe(
            x_shard, in_ln_w, w_proj_p, conv_w, conv_state, a_log, dt_bias, norm_w, rec_state, tp_group, tp_size,
            tag, fold, nb, inplace, row0,
        )
    a_src = nl.ndarray((T_shard, H), dtype=dt, buffer=hbm, name="a_src" + tag)
    nisa.dma_copy(dst=a_src, src=x_shard)
    a_dst = nl.ndarray((T_tp, H), dtype=dt, buffer=hbm, name="a_dst" + tag)
    ncc.all_gather(dsts=[a_dst], srcs=[a_src], replica_group=tp_group, collective_dim=0)
    rs_src = nl.ndarray((T_tp, W), dtype=dt, buffer=hbm, name="rs_src" + tag)
    convs, recs = [], []
    for b, r0 in enumerate(range(0, T_tp, P)):
        r1 = min(T_tp, r0 + P)
        x_sb = nl.ndarray((P, H), dtype=dt, buffer=nl.sbuf)[: r1 - r0]
        nisa.dma_copy(dst=x_sb, src=a_dst[r0:r1])
        xn_hbm = _norm_to_hbm(x_sb, in_ln_w, r1 - r0, H, f"{tag}b{b}")
        xf = _fold_x_rows(a_dst[r0:r1], fold)
        mix_part, nc, nr = gdn_decode_tkg(
            xn_hbm, w_proj_p, conv_w, conv_state[row0 + r0 : row0 + r1], a_log, dt_bias, norm_w, rec_state, fold[0],
            row0 + r0, inplace,
        )
        part = _fold_partial(mix_part, xf, H, dt, f"{tag}b{b}")
        nisa.dma_copy(dst=rs_src[r0:r1], src=part)
        convs.append(nc)
        recs.append(nr)
    rs_dst = nl.ndarray((T_shard, W), dtype=dt, buffer=hbm, name="rs_dst" + tag)
    ncc.reduce_scatter(dsts=[rs_dst], srcs=[rs_src], op=nl.add, replica_group=tp_group, collective_dim=0)
    a = nl.ndarray((P, H), dtype=nl.bfloat16, buffer=nl.sbuf)[:T_shard]
    nisa.dma_copy(dst=a, src=x_shard)
    rb = nl.ndarray((P, W), dtype=nl.bfloat16, buffer=nl.sbuf)[:T_shard]
    nisa.dma_copy(dst=rb, src=rs_dst)
    s = nl.ndarray((P, W), dtype=dt, buffer=nl.sbuf)[:T_shard]
    nisa.tensor_tensor(dst=s[:, 0:H], data1=a, data2=rb[:, 0:H], op=nl.add)
    nisa.tensor_copy(dst=s[:, H:W], src=rb[:, H:W])
    resid1 = nl.ndarray((T_shard, W), dtype=dt, buffer=hbm, name="resid1" + tag)
    nisa.dma_copy(dst=resid1, src=s)
    return resid1, convs, recs


def _gdn_mixer_pipe(
    x_shard, in_ln_w, w_proj_p, conv_w, conv_state, a_log, dt_bias, norm_w, rec_state, tp_group, tp_size, tag,
    fold, nb, inplace=False, row0=0,
):
    """_gdn_mixer_rows as nb independent micro-batches, each holding T_shard/nb of every rank's rows:
    AllGather -> norm / x-fold / GDN -> ReduceScatter per block, so block b's collectives can overlap
    block b+1's GDN. Recurrent / conv states are stored block-major (block b = rows b*128 .. b*128+127
    of the state tensors, in that block's gathered order)."""
    T_shard, H = x_shard.shape
    q = T_shard // nb
    dt = x_shard.dtype
    hbm = nl.private_hbm
    W = fold[0].shape[1]
    resid1 = nl.ndarray((T_shard, W), dtype=dt, buffer=hbm, name="resid1" + tag)
    gathered = []
    for b in range(nb):  # issue every block's AllGather up front
        a_src = nl.ndarray((q, H), dtype=dt, buffer=hbm, name=f"a_src{tag}b{b}")
        nisa.dma_copy(dst=a_src, src=x_shard[b * q : (b + 1) * q])
        a_dst = nl.ndarray((P, H), dtype=dt, buffer=hbm, name=f"a_dst{tag}b{b}")
        ncc.all_gather(dsts=[a_dst], srcs=[a_src], replica_group=tp_group, collective_dim=0)
        gathered.append(a_dst)
    convs, recs = [], []
    if _GDN_WSHARE:  # GDN weights loaded once for all blocks of this layer
        w_all, wo_all = gdn_load_weights(w_proj_p, fold[0], a_log.shape[0])
    for b in range(nb):
        a_dst = gathered[b]
        r0 = b * P
        x_sb = nl.ndarray((P, H), dtype=dt, buffer=nl.sbuf)
        nisa.dma_copy(dst=x_sb, src=a_dst)
        xn_hbm = _norm_to_hbm(x_sb, in_ln_w, P, H, f"{tag}b{b}")
        xf = _fold_x_rows(a_dst, fold)
        if _GDN_WSHARE:  # inlined body: shared weights, mixer partial stays in SBUF
            o_sb, nc, nr = gdn_decode_body(
                None, w_proj_p, conv_w, conv_state[row0 + r0 : row0 + r0 + P], a_log, dt_bias, norm_w, rec_state,
                fold[0], x_hbm=xn_hbm, rec_row0=row0 + r0, w_proj_sb=w_all, w_out_sb=wo_all, inplace=inplace,
            )
            rs_src = _fold_partial_sb(o_sb, xf, H, dt, f"{tag}b{b}")
        else:
            mix_part, nc, nr = gdn_decode_tkg(
                xn_hbm, w_proj_p, conv_w, conv_state[row0 + r0 : row0 + r0 + P], a_log, dt_bias, norm_w, rec_state,
                fold[0], row0 + r0, inplace,
            )
            rs_src = _fold_partial(mix_part, xf, H, dt, f"{tag}b{b}")
        convs.append(nc)
        recs.append(nr)
        rs_dst = nl.ndarray((q, W), dtype=dt, buffer=hbm, name=f"rs_dst{tag}b{b}")
        ncc.reduce_scatter(dsts=[rs_dst], srcs=[rs_src], op=nl.add, replica_group=tp_group, collective_dim=0)
        a = nl.ndarray((P, H), dtype=nl.bfloat16, buffer=nl.sbuf)[:q]
        nisa.dma_copy(dst=a, src=x_shard[b * q : (b + 1) * q])
        rb = nl.ndarray((P, W), dtype=nl.bfloat16, buffer=nl.sbuf)[:q]
        nisa.dma_copy(dst=rb, src=rs_dst)
        s = nl.ndarray((P, W), dtype=dt, buffer=nl.sbuf)[:q]
        nisa.tensor_tensor(dst=s[:, 0:H], data1=a, data2=rb[:, 0:H], op=nl.add)
        nisa.tensor_copy(dst=s[:, H:W], src=rb[:, H:W])
        nisa.dma_copy(dst=resid1[b * q : (b + 1) * q], src=s)
    return resid1, convs, recs


def _layer_body(
    x_shard, in_ln_w, w_proj_p, conv_w, conv_state, a_log, dt_bias, norm_w, rec_state, w_out, post_ln_w,
    w_router_p, sel, w_shared_p, w_sd, gate_up_w, gate_up_scale, down_w, down_scale, tp_group, ep_group,
    tp_size, ep_size, tag, out_buffer, fold=None,
):
    """One GDN+MoE layer (called positionally: the tracer rejects keyword args to nested functions)."""
    if _MOE_PREFETCH:
        prefetch_expert_weights(gate_up_w, gate_up_scale, down_w, down_scale, _MOE_PREFETCH)
    pre = _early_experts(gate_up_w, gate_up_scale, down_w, down_scale, "layer")
    resid1, new_conv, new_rec = _gdn_mixer(
        x_shard, in_ln_w, w_proj_p, conv_w, conv_state, a_log, dt_bias, norm_w, rec_state, w_out, tp_group,
        tp_size, tag, fold,
    )
    out = _moe_half(
        resid1, post_ln_w, w_router_p, sel, w_shared_p, w_sd, gate_up_w, gate_up_scale, down_w, down_scale,
        ep_group, ep_size, tag, out_buffer, pre,
    )
    return out, new_conv, new_rec


def _early_experts(gate_up_w, gate_up_scale, down_w, down_scale, where):
    """Issue the first experts' weight DMAs here if this is the configured prefetch point."""
    if _MOE_FUSED and _MOE_FUSED_PRE and _MOE_PRE_AT == where:
        return issue_expert_weights(gate_up_w, gate_up_scale, down_w, down_scale, _MOE_FUSED_PRE)
    return None


def _moe_half_pipe(
    resid1, post_ln_w, w_router_p, sel, w_shared_p, w_sd, gate_up_w, gate_up_scale, down_w, down_scale,
    ep_group, ep_size, tag, out_buffer, pre, valid=None,
):
    """_moe_half (fused, compacted) as two micro-batches, each holding half of every rank's rows: two EP
    AllGathers issued up front (half 0's fronts start while half 1 gathers), one compacted expert pass
    over all rows, and each half's EP ReduceScatter issued as soon as its rows are combined."""
    T_shard, W = resid1.shape
    H = post_ln_w.shape[0]
    fold = W != H
    q = T_shard // 2
    Th = q * ep_size
    dt = resid1.dtype
    hbm = nl.private_hbm
    halves = []
    for h in range(2):
        m_src = nl.ndarray((q, W), dtype=dt, buffer=hbm, name=f"m_src{tag}h{h}")
        nisa.dma_copy(dst=m_src, src=resid1[h * q : (h + 1) * q])
        m_dst = nl.ndarray((Th, W), dtype=dt, buffer=hbm, name=f"m_dst{tag}h{h}")
        ncc.all_gather(dsts=[m_dst], srcs=[m_src], replica_group=ep_group, collective_dim=0)
        halves.append(m_dst)
    if pre is None:
        pre = _early_experts(gate_up_w, gate_up_scale, down_w, down_scale, "front")
    fronts, e_srcs, dsts = [], [], []
    xn_hbm = nl.ndarray((2 * Th, H), dtype=nl.bfloat16, buffer=hbm, name="mc_xn" + tag)
    for h in range(2):
        e_src = nl.ndarray((Th, H), dtype=dt, buffer=hbm, name=f"e_src{tag}h{h}")
        e_srcs.append(e_src)
        for r0 in range(0, Th, P):
            vb = None if valid is None else valid[len(fronts)]
            f = moe_front_body(halves[h][r0 : r0 + P], post_ln_w, w_router_p, sel, w_shared_p, w_sd, False, fold, vb)
            nisa.dma_copy(dst=xn_hbm[h * Th + r0 : h * Th + r0 + P], src=f[0])
            fronts.append(f)
            dsts.append(e_src[r0 : r0 + P])
    nbh = Th // P
    e_dsts = []

    def reduce_half(b):
        if (b + 1) % nbh == 0:
            e_dst = nl.ndarray((q, H), dtype=dt, buffer=hbm, name=f"e_dst{tag}h{b // nbh}")
            ncc.reduce_scatter(dsts=[e_dst], srcs=[e_srcs[b // nbh]], op=nl.add, replica_group=ep_group, collective_dim=0)
            e_dsts.append(e_dst)

    moe_compact_body(
        xn_hbm, [f[1] for f in fronts], [f[2] for f in fronts], gate_up_w, gate_up_scale, down_w, down_scale, dsts,
        pre=pre, on_block=reduce_half,
    )
    out = nl.ndarray((T_shard, H), dtype=dt, buffer=out_buffer, name="out" + tag)
    for h in range(2):
        _add_rows(out[h * q : (h + 1) * q], resid1[h * q : (h + 1) * q, 0:H], e_dsts[h], q, H)
    return out


def _moe_half(
    resid1, post_ln_w, w_router_p, sel, w_shared_p, w_sd, gate_up_w, gate_up_scale, down_w, down_scale,
    ep_group, ep_size, tag, out_buffer, pre=None, valid=None,
):
    """MoE over the EP group: AllGather -> router/shared/experts -> ReduceScatter -> + resid1.

    Router fold: resid1 is [T_shard, H + E + 1] (residual | unscaled logits); the AllGather carries
    the widened rows and moe_front applies the per-token RMS scale instead of the router matmul.
    valid: moe_row_valid(...) tiles (padding rows routed nowhere) or None (every row routed); ignored by
    the unfused path (QWEN35_MOE_FUSED=0)."""
    T_shard, W = resid1.shape
    H = post_ln_w.shape[0]
    fold = W != H
    T_ep = T_shard * ep_size
    dt = resid1.dtype
    hbm = nl.private_hbm
    if pre is None:
        pre = _early_experts(gate_up_w, gate_up_scale, down_w, down_scale, "moe")

    if _SBUF_CC:
        kernel_assert(_MOE_FUSED and not fold, "QWEN35_SBUF_CC needs the fused MoE block (and no router fold)")
        m_sb = nl.ndarray((P, H), dtype=dt, buffer=nl.sbuf)[:T_ep]
        ncc.all_gather(dsts=[m_sb], srcs=[resid1], replica_group=ep_group, collective_dim=0)
        if pre is None:
            pre = _early_experts(gate_up_w, gate_up_scale, down_w, down_scale, "front")
        fr = moe_front_body(m_sb, post_ln_w, w_router_p, sel, w_shared_p, w_sd, _REUSE_XT, False,
                            None if valid is None else valid[0])
        acc = experts_body(
            fr[0], fr[1], gate_up_w, gate_up_scale, down_w, down_scale, acc_init=fr[2], pre=pre,
            xT=fr[3] if _REUSE_XT else None,
        )
        e_sb = nl.ndarray((P, H), dtype=dt, buffer=nl.sbuf)[:T_ep]
        nisa.tensor_copy(dst=e_sb, src=acc)
        e_dst = nl.ndarray((P, H), dtype=dt, buffer=nl.sbuf)[:T_shard]
        ncc.reduce_scatter(dsts=[e_dst], srcs=[e_sb], op=nl.add, replica_group=ep_group, collective_dim=0)
        out_sb = nl.ndarray((P, H), dtype=dt, buffer=nl.sbuf)[:T_shard]
        nisa.tensor_tensor(dst=out_sb, data1=resid1, data2=e_dst, op=nl.add)
        if out_buffer == nl.sbuf:
            return out_sb
        out = nl.ndarray((T_shard, H), dtype=dt, buffer=out_buffer, name="out" + tag)
        nisa.dma_copy(dst=out, src=out_sb)
        return out
    if _moe_pipe_ok(T_shard, T_ep):
        return _moe_half_pipe(
            resid1, post_ln_w, w_router_p, sel, w_shared_p, w_sd, gate_up_w, gate_up_scale, down_w, down_scale,
            ep_group, ep_size, tag, out_buffer, pre, valid,
        )
    m_dst = nl.ndarray((T_ep, W), dtype=dt, buffer=hbm, name="m_dst" + tag)
    if _DIRECT_CC:
        m_src = resid1
    else:
        m_src = nl.ndarray((T_shard, W), dtype=dt, buffer=hbm, name="m_src" + tag)
        nisa.dma_copy(dst=m_src, src=resid1)
    ncc.all_gather(dsts=[m_dst], srcs=[m_src], replica_group=ep_group, collective_dim=0)

    rs_dt = nl.float8_e4m3fn if _FP8_RS else dt
    e_src = nl.ndarray((T_ep, H), dtype=rs_dt, buffer=hbm, name="e_src" + tag)
    if _MOE_FUSED:
        # One MoE block: the first experts' weights start streaming, the router / shared expert run
        # under that DMA, and the experts accumulate on top of the shared-expert output in SBUF.
        if pre is None:
            pre = _early_experts(gate_up_w, gate_up_scale, down_w, down_scale, "front")
        # > 128 gathered tokens: router / shared expert per 128-row block, experts reuse each
        # expert's weights for every block
        blocks = [(r0, min(P, T_ep - r0)) for r0 in range(0, T_ep, P)]
        if _COMPACT and T_ep > P:
            fronts = [
                moe_front_body(m_dst[r0 : r0 + n], post_ln_w, w_router_p, sel, w_shared_p, w_sd, False, fold,
                               None if valid is None else valid[b])
                for b, (r0, n) in enumerate(blocks)
            ]
            xn_hbm = nl.ndarray((T_ep, H), dtype=nl.bfloat16, buffer=hbm, name="mc_xn" + tag)
            for (r0, n), f in zip(blocks, fronts):
                nisa.dma_copy(dst=xn_hbm[r0 : r0 + n], src=f[0])
            moe_compact_body(
                xn_hbm, [f[1] for f in fronts], [f[2] for f in fronts], gate_up_w, gate_up_scale, down_w, down_scale,
                e_src, pre=pre,
            )
        else:
            fronts = [
                moe_front_body(m_dst[r0 : r0 + n], post_ln_w, w_router_p, sel, w_shared_p, w_sd, _REUSE_XT, fold,
                               None if valid is None else valid[b])
                for b, (r0, n) in enumerate(blocks)
            ]
            accs = experts_body(
                [f[0] for f in fronts], [f[1] for f in fronts], gate_up_w, gate_up_scale, down_w, down_scale,
                acc_init=[f[2] for f in fronts], pre=pre, xT=[f[3] for f in fronts] if _REUSE_XT else None,
            )
            for (r0, n), acc in zip(blocks, accs):
                e_sb = nl.ndarray((P, H), dtype=rs_dt, buffer=nl.sbuf)[:n]
                nisa.tensor_copy(dst=e_sb, src=acc)
                nisa.dma_copy(dst=e_src[r0 : r0 + n], src=e_sb)
    else:
        kernel_assert(not fold, "the router fold needs the fused MoE block")
        xn_m, aff_l, shared = moe_front(m_dst, post_ln_w, w_router_p, sel, w_shared_p, w_sd)
        routed = moe_experts_mx(xn_m, gate_up_w, gate_up_scale, down_w, down_scale, aff_l)
        _add_rows(e_src, routed, shared, T_ep, H)
    e_dst = nl.ndarray((T_shard, H), dtype=rs_dt, buffer=hbm, name="e_dst" + tag)
    ncc.reduce_scatter(dsts=[e_dst], srcs=[e_src], op=nl.add, replica_group=ep_group, collective_dim=0)

    out = nl.ndarray((T_shard, H), dtype=dt, buffer=out_buffer, name="out" + tag)
    _add_rows(out, resid1[:, 0:H] if fold else resid1, e_dst, T_shard, H)
    return out


@nki.jit
def qwen35_gdn_moe_layer(
    x_shard,  # [T_shard, H] bf16
    in_ln_w,  # [H] fp32 input_layernorm (zero-centred)
    w_proj_p, conv_w, conv_state, a_log, dt_bias, norm_w, rec_state, w_out,  # GDN TP shard
    post_ln_w,  # [H] fp32 post_attention_layernorm
    w_router_p, sel, w_shared_p, w_sd,  # MoE front (router replicated, shared-expert slice)
    gate_up_w, gate_up_scale, down_w, down_scale,  # local MX experts
    tp_group: ReplicaGroup,
    ep_group: ReplicaGroup,
    tp_size: int,
    ep_size: int,
    # router fold (QWEN35_ROUTER_FOLD=1 only; ignored otherwise):
    w_out_ext=None,  # [Vd, H + E + 1] bf16 [W_out | W_out diag(1 + w_post) [W_r | w_sg]]
    wx=None,  # [H / tp_size, E + 1] bf16 diag(1 + w_post) [W_r | w_sg] rows of this rank's TP position
    tp_rank=None,  # [1, 1] int32 this rank's position in its TP group
):
    _PREFETCH.clear()  # trace-time state must not leak from an earlier (possibly failed) trace
    fold = _fold_inputs(w_out_ext, wx, tp_rank, w_out, "GDN")
    out, new_conv, new_rec = _layer_body(
        x_shard, in_ln_w, w_proj_p, conv_w, conv_state, a_log, dt_bias, norm_w, rec_state, w_out, post_ln_w,
        w_router_p, sel, w_shared_p, w_sd, gate_up_w, gate_up_scale, down_w, down_scale, tp_group, ep_group,
        tp_size, ep_size, "_L0", nl.shared_hbm, fold,
    )
    if isinstance(new_conv, list):  # > 128 rows per TP group: per-block states; return the first block's
        new_conv, new_rec = new_conv[0], new_rec[0]
    return out, new_conv, new_rec


@nki.jit
def qwen35_gdn_moe_stack(
    x_shard,
    in_ln_w, w_proj_p, conv_w, conv_state, a_log, dt_bias, norm_w, rec_state, w_out,
    post_ln_w, w_router_p, sel, w_shared_p, w_sd, gate_up_w, gate_up_scale, down_w, down_scale,
    tp_group: ReplicaGroup,
    ep_group: ReplicaGroup,
    tp_size: int,
    ep_size: int,
    num_layers: int,
    w_out_ext=None, wx=None, tp_rank=None,  # router fold (as qwen35_gdn_moe_layer)
):
    """``num_layers`` GDN+MoE layers in one NEFF, all reusing the same (shared) weights.

    Profiling scaffold for the per-layer steady-state cost: every layer still issues all of
    its weight DMAs (nested kernels load their inputs explicitly), so sharing the weight
    tensors does not hide DMA. Recurrent/conv states are also shared (benchmark only).
    """
    _PREFETCH.clear()
    fold = _fold_inputs(w_out_ext, wx, tp_rank, w_out, "GDN")
    x = x_shard
    new_conv = None
    new_rec = None
    for i in range(num_layers):
        x, new_conv, new_rec = _layer_body(
            x, in_ln_w, w_proj_p, conv_w, conv_state, a_log, dt_bias, norm_w, rec_state, w_out, post_ln_w,
            w_router_p, sel, w_shared_p, w_sd, gate_up_w, gate_up_scale, down_w, down_scale, tp_group, ep_group,
            tp_size, ep_size, f"_L{i}", nl.shared_hbm if i == num_layers - 1 else _INTER, fold,
        )
    return x, new_conv, new_rec


@nki.jit
def qwen35_hybrid_stack(
    x_shard,
    in_ln_w, w_proj_p, conv_w, conv_state, a_log, dt_bias, norm_w, rec_state, w_out,  # GDN shard
    w_qkv, q_norm_w, k_norm_w, w_gate_p, cos, sin, k_cache_t, v_cache, pos, w_o,  # attention shard
    post_ln_w, w_router_p, sel, w_shared_p, w_sd, gate_up_w, gate_up_scale, down_w, down_scale,
    tp_group: ReplicaGroup,
    atp_group: ReplicaGroup,
    ep_group: ReplicaGroup,
    tp_size: int,
    atp_size: int,
    ep_size: int,
    num_layers: int,
    # router fold (QWEN35_ROUTER_FOLD=1 only): GDN inputs as qwen35_gdn_moe_layer, attention equivalents
    w_out_ext=None, wx=None, tp_rank=None,
    w_o_ext=None,  # [nq * d, H + E + 1] bf16 folded attention out-projection
    wx_a=None,  # [H / atp_size, E + 1] bf16 x-fold rows of this rank's attention TP position
    atp_rank=None,  # [1, 1] int32 this rank's position in its attention TP group
    kv_slot=None,  # [B_attn, 1] int32 KV-pool slot per attention row (caches [NSLOT, ...]); None: row b -> slot b
    kv_write="flat",  # attn_decode_tkg new-token K/V write mode with kv_slot
):
    """Qwen3.5 layer pattern in one NEFF: every 4th layer is gated attention, the rest GDN, all MoE.

    Timing scaffold like ``qwen35_gdn_moe_stack``: layers of a kind share weights / states / the KV
    cache, but every layer issues all of its own DMAs and collectives.
    """
    _PREFETCH.clear()
    fold_g = _fold_inputs(w_out_ext, wx, tp_rank, w_out, "GDN")
    fold_a = _fold_inputs(w_o_ext, wx_a, atp_rank, w_o, "attention")
    x = x_shard
    for i in range(num_layers):
        tag = f"_L{i}"
        pre = _early_experts(gate_up_w, gate_up_scale, down_w, down_scale, "layer")
        if i % 4 == 3:
            resid1 = _attn_mixer(
                x, in_ln_w, w_qkv, q_norm_w, k_norm_w, w_gate_p, cos, sin, k_cache_t, v_cache, pos, w_o, atp_group,
                atp_size, tag, fold_a, kv_slot, kv_write,
            )
        else:
            resid1, _, _ = _gdn_mixer(
                x, in_ln_w, w_proj_p, conv_w, conv_state, a_log, dt_bias, norm_w, rec_state, w_out, tp_group,
                tp_size, tag, fold_g,
            )
        x = _moe_half(
            resid1, post_ln_w, w_router_p, sel, w_shared_p, w_sd, gate_up_w, gate_up_scale, down_w, down_scale,
            ep_group, ep_size, tag, nl.shared_hbm if i == num_layers - 1 else _INTER, pre,
        )
    return x


@nki.jit
def qwen35_gdn_fold_partial(
    x_tp,  # [T_tp, H] bf16 gathered residual of this rank's TP group
    in_ln_w, w_proj_p, conv_w, conv_state, a_log, dt_bias, norm_w, rec_state,  # GDN TP shard
    w_out_ext,  # [Vd, H + E + 1] bf16 folded out-projection
    wx,  # [H / tp_size, E + 1] bf16 x-fold rows of this rank's TP position
    tp_rank,  # [1, 1] int32
):
    """Single-core check of the router fold's mixer side (tests): input RMSNorm -> GDN with the widened
    out-projection -> + x-fold rows = the ReduceScatter source [T_tp, H + E + 1] of one TP rank."""
    T_tp, H = x_tp.shape
    x_sb = nl.ndarray((P, H), dtype=x_tp.dtype, buffer=nl.sbuf)[:T_tp]
    nisa.dma_copy(dst=x_sb, src=x_tp)
    xn_hbm = _norm_to_hbm(x_sb, in_ln_w, T_tp, H, "_T")
    xf = _fold_x_rows(x_tp, (w_out_ext, wx, _load_rank(tp_rank)))
    mix_part, _, _ = gdn_decode_tkg(xn_hbm, w_proj_p, conv_w, conv_state, a_log, dt_bias, norm_w, rec_state, w_out_ext)
    return _fold_partial(mix_part, xf, H, x_tp.dtype, "_T", nl.shared_hbm)
