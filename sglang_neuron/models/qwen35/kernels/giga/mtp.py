"""Qwen3.5 MTP (multi-token prediction) draft path + greedy acceptance for speculative decoding (NKI, LNC=1).

The checkpoint's MTP module (``mtp.*`` weights; sglang ``Qwen3_5ForCausalLMMTP``, vLLM ``qwen3_5_mtp``):

    x0   = fc( [ RMSNorm_zc(embed(tok); pre_fc_norm_embedding) | RMSNorm_zc(h; pre_fc_norm_hidden) ] )   fc: 2H -> H
    x1   = decoder layer (gated FULL ATTENTION + the same 512-expert MoE; its own KV cache)    (mtp.layers.0)
    h'   = RMSNorm_zc(x1; mtp.norm)        -> next draft step's h
    tok' = argmax(lm_head(h'))             (lm_head / embed_tokens shared with the target)

where ``h`` is the target's FINAL-NORMED hidden state at position p and ``tok`` the token at
p + 1; the pair sits at position p (rope + MTP KV slot p). Drafting k tokens = 1 "extend" pass
over the round's S = k + 1 verified rows (fills the MTP KV for every accepted position and yields
draft 1 at row n_b - 1) + (k - 1) single-row chain steps feeding h' back.

Building blocks here (each with a single-rank torch-referenced test) and the multi-rank draft
kernel composed from them and from ``verify_stack``'s attention / MoE pieces:
  embed_rows        token ids -> embedding rows (indirect gather from a replicated table)
  rope_rows         positions -> cos / sin rows (indirect gather from a precomputed table)
  mtp_fc_body       gathered [e | h] rows -> both pre-fc norms -> this rank's contraction slice of fc
  lm_argmax_body    normed rows x this rank's vocab slice -> (max logit, global argmax) per row
  argmax_combine    every rank's (max, idx) candidates -> global greedy token per row
  spec_accept_greedy  draft vs target argmax -> accepted count, bonus token, next state slot
Rank-dependent offsets (fc slice, vocab slice, own rows) arrive as small int32 DATA inputs, so every
rank runs the same NEFF.
"""

import nki
import nki.collectives as ncc
import nki.isa as nisa
import nki.language as nl
from nki.collectives import ReplicaGroup

from nkilib.core.utils.kernel_assert import kernel_assert
from .moe_experts_mx import _PREFETCH
from .moe_front import rmsnorm_zc_tok
from .verify_stack import _attn_verify_mixer, _moe_half_rows

P = 128
MCH = 512


def _bcast_scalar(src, n):
    """[1, 1] int32 HBM -> [n, 1] int32 SBUF (same value on n partitions)."""
    t = nl.ndarray((P, 1), dtype=nl.int32, buffer=nl.sbuf)[:n]
    nisa.dma_copy(dst=t, src=src.ap(pattern=[[0, n], [1, 1]], offset=0))
    return t


def _row_offsets(n, stride, base_i32):
    """[n, 1] int32: p * stride + base[p] (base an [n, 1] int32 SBUF tile)."""
    off = nl.ndarray((P, 1), dtype=nl.int32, buffer=nl.sbuf)[:n]
    nisa.iota(off, pattern=[[0, 1]], offset=0, channel_multiplier=stride)
    nisa.tensor_tensor(dst=off, data1=off, data2=base_i32, op=nl.add)
    return off


# ----------------------------------------------------------------------------- gathers


def embed_rows_body(table, ids_hbm, T, out_hbm):
    """out_hbm[t] = table[ids[t]] for T <= 128 rows (one indirect DMA)."""
    H = table.shape[1]
    ids = nl.ndarray((P, 1), dtype=nl.int32, buffer=nl.sbuf)[:T]
    nisa.dma_copy(dst=ids, src=ids_hbm)
    e = nl.ndarray((P, H), dtype=table.dtype, buffer=nl.sbuf)[:T]
    nisa.dma_copy(
        dst=e, src=table.ap(pattern=[[H, T], [1, H]], offset=0, vector_offset=ids, indirect_dim=0),
        dge_mode=nisa.dge_mode.swdge,
    )
    nisa.dma_copy(dst=out_hbm, src=e)


@nki.jit
def embed_rows(table, ids):  # [V, H] bf16, [T, 1] int32
    T = ids.shape[0]
    out = nl.ndarray((T, table.shape[1]), dtype=table.dtype, buffer=nl.shared_hbm)
    embed_rows_body(table, ids, T, out)
    return out


def rope_rows_body(cos_tab, sin_tab, pos_sb, T, cos_out, sin_out):
    """cos_out[t] = cos_tab[pos[t]] (and sin) for T <= 128 rows; pos_sb an [T, 1] int32 SBUF tile."""
    half = cos_tab.shape[1]
    for tab, dst in ((cos_tab, cos_out), (sin_tab, sin_out)):
        r = nl.ndarray((P, half), dtype=nl.float32, buffer=nl.sbuf)[:T]
        nisa.dma_copy(
            dst=r, src=tab.ap(pattern=[[half, T], [1, half]], offset=0, vector_offset=pos_sb, indirect_dim=0),
            dge_mode=nisa.dge_mode.swdge,
        )
        nisa.dma_copy(dst=dst, src=r)


@nki.jit
def rope_rows(cos_tab, sin_tab, pos):  # [Lmax, 32] fp32 x2, [T, 1] int32
    T = pos.shape[0]
    half = cos_tab.shape[1]
    p = nl.ndarray((P, 1), dtype=nl.int32, buffer=nl.sbuf)[:T]
    nisa.dma_copy(dst=p, src=pos)
    c = nl.ndarray((T, half), dtype=nl.float32, buffer=nl.shared_hbm)
    s = nl.ndarray((T, half), dtype=nl.float32, buffer=nl.shared_hbm)
    rope_rows_body(cos_tab, sin_tab, p, T, c, s)
    return c, s


# ----------------------------------------------------------------------------- MTP fc


def mtp_fc_body(eh, ln_e, ln_h, fc_p, fc_row, out_hbm, tag=""):
    """This rank's partial of fc([norm_e(e) | norm_h(h)]) for the gathered rows.

    eh     [T, 2H] bf16 HBM: raw embedding | target hidden, T <= 128 gathered rows
    ln_e, ln_h [H] fp32 zero-centred pre-fc norm weights
    fc_p   [128, KS/128, H] bf16: rows [r*KS, (r+1)*KS) of fc^T (W^T[i, o] = fc.weight[o, i]),
           pre-permuted (fc_p[p, k1, o] = W^T[r*KS + k1*128 + p, o])
    fc_row [1, 1] int32: r, this rank's slice index (2H / KS slices per group)
    out_hbm [T, H] bf16: partial sum (ReduceScatter over the slice group gives fc's output)
    """
    T, H2 = eh.shape
    H = H2 // 2
    KS = fc_p.shape[1] * P
    NSL = H2 // KS
    K1 = KS // P
    kernel_assert(T <= P and H2 % KS == 0, "T <= 128 and KS must divide 2H")
    xcat = nl.ndarray((T, H2), dtype=nl.bfloat16, buffer=nl.private_hbm)
    for half, w in ((0, ln_e), (1, ln_h)):
        x_sb = nl.ndarray((P, H), dtype=nl.bfloat16, buffer=nl.sbuf)[:T]
        nisa.dma_copy(dst=x_sb, src=eh[:, half * H : (half + 1) * H])
        w_row = nl.ndarray((P, H), dtype=nl.float32, buffer=nl.sbuf)[:T]
        nisa.dma_copy(dst=w_row, src=w.ap(pattern=[[0, T], [1, H]], offset=0))
        nisa.tensor_scalar(dst=w_row, data=w_row, op0=nl.add, operand0=1.0)
        xn = rmsnorm_zc_tok(x_sb, w_row, T, H, 1e-6, nl.bfloat16)
        nisa.dma_copy(dst=xcat[:, half * H : (half + 1) * H], src=xn)
    # this rank's KS columns of every row: xcat viewed as [T*NSL, KS], row t*NSL + r
    off = _row_offsets(T, NSL, _bcast_scalar(fc_row, T))
    x_sl = nl.ndarray((P, KS), dtype=nl.bfloat16, buffer=nl.sbuf)[:T]
    nisa.dma_copy(
        dst=x_sl,
        src=xcat.reshape((T * NSL, KS)).ap(pattern=[[KS, T], [1, KS]], offset=0, vector_offset=off, indirect_dim=0),
        dge_mode=nisa.dge_mode.swdge,
    )
    xT = nl.ndarray((P, K1, T), dtype=nl.bfloat16, buffer=nl.sbuf)
    for k1 in range(K1):
        tp = nl.ndarray((P, T), dtype=nl.bfloat16, buffer=nl.psum)
        nisa.nc_transpose(dst=tp, data=x_sl[:, k1 * P : (k1 + 1) * P])
        nisa.tensor_copy(dst=xT[:, k1, :], src=tp)
    o_sb = nl.ndarray((P, H), dtype=nl.bfloat16, buffer=nl.sbuf)[:T]
    for c0 in range(0, H, MCH):
        w_sb = nl.ndarray((P, K1, MCH), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=w_sb, src=fc_p[:, :, c0 : c0 + MCH])
        ps = nl.ndarray((P, MCH), dtype=nl.float32, buffer=nl.psum)[:T]
        for k1 in range(K1):
            nisa.nc_matmul(ps, xT[:, k1, :], w_sb[:, k1, :])
        nisa.tensor_copy(dst=o_sb[:, c0 : c0 + MCH], src=ps)
    nisa.dma_copy(dst=out_hbm, src=o_sb)


@nki.jit
def mtp_fc(eh, ln_e, ln_h, fc_p, fc_row):
    T, H2 = eh.shape
    out = nl.ndarray((T, H2 // 2), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    mtp_fc_body(eh, ln_e, ln_h, fc_p, fc_row, out)
    return out


# ----------------------------------------------------------------------------- lm_head + greedy argmax


def lm_argmax_body(hn, w_lm_p, vbase, cand_hbm, vb_sb=None):
    """cand[t] = (max logit, global argmax) of hn[t] over this rank's vocab slice.

    hn [T, H] bf16 HBM (final-normed, any T: 128-row tiles share each streamed weight chunk),
    w_lm_p [128, H/128, Vs] bf16 (lm_head^T slice, rows pre-permuted), vbase [1, 1] int32 (global
    id of the slice's first token). Logits are rounded to bf16 like the HF model. vb_sb: optional [>= 128 or T, 1] int32
    SBUF broadcast of vbase (multi-step decode loads it once per launch instead of once per tile and step).
    """
    T, H = hn.shape
    Vs = w_lm_p.shape[2]
    H1 = H // P
    tiles = [(r0, min(T, r0 + P)) for r0 in range(0, T, P)]
    xT = nl.ndarray((P, H1, T), dtype=nl.bfloat16, buffer=nl.sbuf)
    for r0, r1 in tiles:
        x_sb = nl.ndarray((P, H), dtype=nl.bfloat16, buffer=nl.sbuf)[: r1 - r0]
        nisa.dma_copy(dst=x_sb, src=hn[r0:r1])
        for h1 in range(H1):
            tp = nl.ndarray((P, r1 - r0), dtype=nl.bfloat16, buffer=nl.psum)
            nisa.nc_transpose(dst=tp, data=x_sb[:, h1 * P : (h1 + 1) * P])
            nisa.tensor_copy(dst=xT[:, h1, r0:r1], src=tp)
    logit = [nl.ndarray((P, Vs), dtype=nl.float32, buffer=nl.sbuf)[: r1 - r0] for r0, r1 in tiles]
    for c0 in range(0, Vs, MCH):
        cn = min(MCH, Vs - c0)
        w_sb = nl.ndarray((P, H1, MCH), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=w_sb[:, :, 0:cn], src=w_lm_p[:, :, c0 : c0 + cn])
        for rt, (r0, r1) in enumerate(tiles):
            ps = nl.ndarray((P, MCH), dtype=nl.float32, buffer=nl.psum)[: r1 - r0]
            for h1 in range(H1):
                nisa.nc_matmul(ps[:, 0:cn], xT[:, h1, r0:r1], w_sb[:, h1, 0:cn])
            lb = nl.ndarray((P, MCH), dtype=nl.bfloat16, buffer=nl.sbuf)[: r1 - r0]
            nisa.tensor_copy(dst=lb[:, 0:cn], src=ps[:, 0:cn])
            nisa.tensor_copy(dst=logit[rt][:, c0 : c0 + cn], src=lb[:, 0:cn])
    for rt, (r0, r1) in enumerate(tiles):
        n = r1 - r0
        top8 = nl.ndarray((P, 8), dtype=nl.float32, buffer=nl.sbuf)[:n]
        nisa.max8(dst=top8, src=logit[rt])
        idx8 = nl.ndarray((P, 8), dtype=nl.uint32, buffer=nl.sbuf)[:n]
        nisa.nc_find_index8(dst=idx8, data=logit[rt], vals=top8)
        vb = vb_sb[:n] if vb_sb is not None else _bcast_scalar(vbase, n)
        c = nl.ndarray((P, 2), dtype=nl.float32, buffer=nl.sbuf)[:n]
        nisa.tensor_copy(dst=c[:, 0:1], src=top8[:, 0:1])
        gi = nl.ndarray((P, 1), dtype=nl.int32, buffer=nl.sbuf)[:n]
        nisa.tensor_copy(dst=gi, src=idx8[:, 0:1])
        nisa.tensor_tensor(dst=gi, data1=gi, data2=vb, op=nl.add)
        nisa.tensor_copy(dst=c[:, 1:2], src=gi)
        nisa.dma_copy(dst=cand_hbm[r0:r1], src=c)


@nki.jit
def lm_argmax(hn, w_lm_p, vbase):
    cand = nl.ndarray((hn.shape[0], 2), dtype=nl.float32, buffer=nl.shared_hbm)
    lm_argmax_body(hn, w_lm_p, vbase, cand)
    return cand


def argmax_combine_body(cand_all, T, R, t0=0, tn=None, row_off=None):
    """cand_all [R*T, 2] (rank-major (max, idx) candidates) -> [tn, 1] int32 SBUF greedy token ids of
    rows t0 .. t0 + tn (default: all T <= 128 rows).

    row_off: [tn, 1] int32 SBUF runtime row indices (e.g. ep_rank * n + i: only this rank's own rows) -
    the candidates of exactly those rows are gathered (one indirect DMA) instead of rows t0 .. t0 + tn.
    Ties pick the larger token id (bf16 logits tie occasionally; HF argmax would pick the smaller).
    """
    tn = T if tn is None else tn
    kernel_assert(tn <= P and t0 + tn <= T, "argmax_combine_body: at most 128 rows per call")
    T, n_all = tn, T  # the reductions below run on tn rows; the DMA strides use the full row count
    c = nl.ndarray((P, R, 2), dtype=nl.float32, buffer=nl.sbuf)[:T]
    if row_off is None:
        nisa.dma_copy(dst=c, src=cand_all.ap(pattern=[[2, T], [2 * n_all, R], [1, 2]], offset=2 * t0))
    else:
        nisa.dma_copy(
            dst=c,
            src=cand_all.ap(pattern=[[2, T], [2 * n_all, R], [1, 2]], offset=0, vector_offset=row_off, indirect_dim=0),
            dge_mode=nisa.dge_mode.swdge,
        )
    v = nl.ndarray((P, R), dtype=nl.float32, buffer=nl.sbuf)[:T]
    nisa.tensor_copy(dst=v, src=c[:, :, 0])
    ix = nl.ndarray((P, R), dtype=nl.float32, buffer=nl.sbuf)[:T]
    nisa.tensor_scalar(dst=ix, data=c[:, :, 1], op0=nl.add, operand0=1.0)
    mx = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)[:T]
    nisa.tensor_reduce(dst=mx, op=nl.max, data=v, axis=1)
    m = nl.ndarray((P, R), dtype=nl.float32, buffer=nl.sbuf)[:T]
    nisa.tensor_scalar(dst=m, data=v, op0=nl.greater_equal, operand0=mx)
    nisa.tensor_tensor(dst=m, data1=m, data2=ix, op=nl.multiply)
    best = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)[:T]
    nisa.tensor_reduce(dst=best, op=nl.max, data=m, axis=1)
    nisa.tensor_scalar(dst=best, data=best, op0=nl.subtract, operand0=1.0)
    ids = nl.ndarray((P, 1), dtype=nl.int32, buffer=nl.sbuf)[:T]
    nisa.tensor_copy(dst=ids, src=best)
    return ids


@nki.jit
def argmax_combine(cand_all, ntok: int):
    R = cand_all.shape[0] // ntok
    ids = argmax_combine_body(cand_all, ntok, R)
    out = nl.ndarray((ntok, 1), dtype=nl.int32, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=out, src=ids)
    return out


# ----------------------------------------------------------------------------- acceptance


def accept_greedy_body(draft, target):
    """Greedy acceptance for B <= 128 sequences.

    draft [B, k] int32: draft tokens d_1..d_k (verify rows 1..k); target [B, k+1] int32: argmax of the
    target logits at verify rows 0..k. d_i is accepted iff d_1..d_i all match target rows 0..i-1.
    Returns SBUF int32 [B, 1] tiles (n_acc, next_tok, slot): n = 1 + #accepted tokens to commit
    (row 0, the previous round's bonus token, always commits), next_tok = target[n - 1] (the bonus
    token that starts the next round), slot = n - 1 (state snapshot to resume from).
    """
    B, k = draft.shape
    d_i = nl.ndarray((P, k), dtype=nl.int32, buffer=nl.sbuf)[:B]
    y_i = nl.ndarray((P, k + 1), dtype=nl.int32, buffer=nl.sbuf)[:B]
    nisa.dma_copy(dst=d_i, src=draft)
    nisa.dma_copy(dst=y_i, src=target)
    d = nl.ndarray((P, k), dtype=nl.float32, buffer=nl.sbuf)[:B]
    y = nl.ndarray((P, k + 1), dtype=nl.float32, buffer=nl.sbuf)[:B]
    nisa.tensor_copy(dst=d, src=d_i)
    nisa.tensor_copy(dst=y, src=y_i)
    eq = nl.ndarray((P, k), dtype=nl.float32, buffer=nl.sbuf)[:B]
    nisa.tensor_tensor(dst=eq, data1=d, data2=y[:, 0:k], op=nl.equal)
    acc = nl.ndarray((P, k), dtype=nl.float32, buffer=nl.sbuf)[:B]  # prefix products
    nisa.tensor_copy(dst=acc[:, 0:1], src=eq[:, 0:1])
    for i in range(1, k):
        nisa.tensor_tensor(dst=acc[:, i : i + 1], data1=acc[:, i - 1 : i], data2=eq[:, i : i + 1], op=nl.multiply)
    n_f = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)[:B]
    nisa.tensor_reduce(dst=n_f, op=nl.add, data=acc, axis=1)
    nisa.tensor_scalar(dst=n_f, data=n_f, op0=nl.add, operand0=1.0)
    # next = y_0 + sum_i acc_i * (y_i - y_{i-1})  (telescopes to y_{n-1}; ids < 2^24 are exact in fp32)
    dy = nl.ndarray((P, k), dtype=nl.float32, buffer=nl.sbuf)[:B]
    nisa.tensor_tensor(dst=dy, data1=y[:, 1 : k + 1], data2=y[:, 0:k], op=nl.subtract)
    nisa.tensor_tensor(dst=dy, data1=dy, data2=acc, op=nl.multiply)
    nxt = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)[:B]
    nisa.tensor_reduce(dst=nxt, op=nl.add, data=dy, axis=1)
    nisa.tensor_tensor(dst=nxt, data1=nxt, data2=y[:, 0:1], op=nl.add)
    outs = []
    for src, sub in ((n_f, 0.0), (nxt, 0.0), (n_f, 1.0)):
        t = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)[:B]
        nisa.tensor_scalar(dst=t, data=src, op0=nl.subtract, operand0=sub)
        ti = nl.ndarray((P, 1), dtype=nl.int32, buffer=nl.sbuf)[:B]
        nisa.tensor_copy(dst=ti, src=t)
        outs.append(ti)
    return outs


@nki.jit
def spec_accept_greedy(draft, target):
    B = draft.shape[0]
    res = []
    for t in accept_greedy_body(draft, target):
        o = nl.ndarray((B, 1), dtype=nl.int32, buffer=nl.shared_hbm)
        nisa.dma_copy(dst=o, src=t)
        res.append(o)
    return tuple(res)


# ----------------------------------------------------------------------------- multi-rank draft


def _select_rows(src_hbm, rows_hbm, n, tag):
    """dst[i] = src[rows[i]] (HBM -> private HBM), rows an [n, 1] int32 HBM tensor."""
    H = src_hbm.shape[1]
    r = nl.ndarray((P, 1), dtype=nl.int32, buffer=nl.sbuf)[:n]
    nisa.dma_copy(dst=r, src=rows_hbm)
    t = nl.ndarray((P, H), dtype=src_hbm.dtype, buffer=nl.sbuf)[:n]
    nisa.dma_copy(
        dst=t, src=src_hbm.ap(pattern=[[H, n], [1, H]], offset=0, vector_offset=r, indirect_dim=0),
        dge_mode=nisa.dge_mode.swdge,
    )
    out = nl.ndarray((n, H), dtype=src_hbm.dtype, buffer=nl.private_hbm, name="sel" + tag)
    nisa.dma_copy(dst=out, src=t)
    return out


def _mtp_forward(
    e_shard, h_shard, ln_e, ln_h, fc_p, fc_row, in_ln_w, w_qkv, q_norm_w, k_norm_w, w_gate_p, cos, sin, k_cache_t,
    v_cache, pos, w_o, post_ln_w, w_router_p, sel, w_shared_p, w_sd, gate_up_w, gate_up_scale, down_w, down_scale,
    mtp_norm_w, atp_group, ep_group, atp_size, ep_size, tag,
):
    """MTP block over this rank's rows ([S * seqs, H] e / h shards) -> final-normed h' shard (HBM)."""
    Ts, H = e_shard.shape
    hbm = nl.private_hbm
    eh_src = nl.ndarray((Ts, 2 * H), dtype=nl.bfloat16, buffer=hbm, name="eh_src" + tag)
    nisa.dma_copy(dst=eh_src[:, 0:H], src=e_shard)
    nisa.dma_copy(dst=eh_src[:, H : 2 * H], src=h_shard)
    eh_dst = nl.ndarray((Ts * atp_size, 2 * H), dtype=nl.bfloat16, buffer=hbm, name="eh_dst" + tag)
    ncc.all_gather(dsts=[eh_dst], srcs=[eh_src], replica_group=atp_group, collective_dim=0)
    part = nl.ndarray((Ts * atp_size, H), dtype=nl.bfloat16, buffer=hbm, name="fc_part" + tag)
    mtp_fc_body(eh_dst, ln_e, ln_h, fc_p, fc_row, part, tag)
    x0 = nl.ndarray((Ts, H), dtype=nl.bfloat16, buffer=hbm, name="x0" + tag)
    ncc.reduce_scatter(dsts=[x0], srcs=[part], op=nl.add, replica_group=atp_group, collective_dim=0)
    resid1 = _attn_verify_mixer(
        x0, in_ln_w, w_qkv, q_norm_w, k_norm_w, w_gate_p, cos, sin, k_cache_t, v_cache, pos, w_o, atp_group,
        atp_size, tag,
    )
    x1 = _moe_half_rows(
        resid1, post_ln_w, w_router_p, sel, w_shared_p, w_sd, gate_up_w, gate_up_scale, down_w, down_scale,
        ep_group, ep_size, tag, hbm,
    )
    x_sb = nl.ndarray((P, H), dtype=nl.bfloat16, buffer=nl.sbuf)[:Ts]
    nisa.dma_copy(dst=x_sb, src=x1)
    w_row = nl.ndarray((P, H), dtype=nl.float32, buffer=nl.sbuf)[:Ts]
    nisa.dma_copy(dst=w_row, src=mtp_norm_w.ap(pattern=[[0, Ts], [1, H]], offset=0))
    nisa.tensor_scalar(dst=w_row, data=w_row, op0=nl.add, operand0=1.0)
    hn = rmsnorm_zc_tok(x_sb, w_row, Ts, H, 1e-6, nl.bfloat16)
    h_out = nl.ndarray((Ts, H), dtype=nl.bfloat16, buffer=hbm, name="h_out" + tag)
    nisa.dma_copy(dst=h_out, src=hn)
    return h_out


def _greedy_tokens(h_rows, w_lm_p, vbase, ep_rank, ep_group, ep_size, tag):
    """Greedy next token for this rank's rows: EP AllGather -> vocab-slice logits -> candidates
    AllGather -> combine -> own rows. Returns an [n, 1] int32 HBM tensor."""
    n, H = h_rows.shape
    T = n * ep_size
    hbm = nl.private_hbm
    g = nl.ndarray((T, H), dtype=nl.bfloat16, buffer=hbm, name="lm_in" + tag)
    ncc.all_gather(dsts=[g], srcs=[h_rows], replica_group=ep_group, collective_dim=0)
    cand = nl.ndarray((T, 2), dtype=nl.float32, buffer=hbm, name="cand" + tag)
    lm_argmax_body(g, w_lm_p, vbase, cand)
    cand_all = nl.ndarray((T * ep_size, 2), dtype=nl.float32, buffer=hbm, name="cand_all" + tag)
    ncc.all_gather(dsts=[cand_all], srcs=[cand], replica_group=ep_group, collective_dim=0)
    kernel_assert(T <= P, "greedy combine handles <= 128 gathered rows")
    ids = argmax_combine_body(cand_all, T, ep_size)
    ids_hbm = nl.ndarray((T, 1), dtype=nl.int32, buffer=hbm, name="ids_all" + tag)
    nisa.dma_copy(dst=ids_hbm, src=ids)
    # own rows: ep_rank * n + i
    off = _row_offsets(n, 1, _bcast_scalar(ep_rank, n))
    own = nl.ndarray((P, 1), dtype=nl.int32, buffer=nl.sbuf)[:n]
    nisa.dma_copy(
        dst=own, src=ids_hbm.ap(pattern=[[1, n], [1, 1]], offset=0, vector_offset=off, indirect_dim=0),
        dge_mode=nisa.dge_mode.swdge,
    )
    out = nl.ndarray((n, 1), dtype=nl.int32, buffer=hbm, name="tok" + tag)
    nisa.dma_copy(dst=out, src=own)
    return out


@nki.jit
def qwen35_mtp_draft(
    ext_ids,  # [S * seqs, 1] int32 MTP extend input tokens of this rank's rows (verify inputs shifted by one, bonus token at row n-1)
    h_shard,  # [S * seqs, H] bf16 target final-normed hidden states of this rank's verify rows
    sel_rows,  # [seqs, 1] int32 own row (b*S + n_b - 1) to continue drafting from
    embed_table,  # [V, H] bf16 (replicated)
    ln_e, ln_h, fc_p, fc_row,  # MTP pre-fc norms and this rank's fc slice
    in_ln_w, w_qkv, q_norm_w, k_norm_w, w_gate_p,  # MTP attention layer (attention-TP shard)
    cos, sin,  # [B_attn * S, 32] rope rows of the extend pass
    cos_tab, sin_tab,  # [Lmax, 32] rope tables (chain steps look positions up on device)
    k_cache_t, v_cache,  # the MTP layer's own KV cache (attention-group slots)
    pos,  # [B_attn, 1] int32 first verify position of each attention-group sequence
    n_grp,  # [B_attn, 1] int32 accepted counts of the attention-group sequences this round
    w_o,
    post_ln_w, w_router_p, sel, w_shared_p, w_sd, gate_up_w, gate_up_scale, down_w, down_scale,  # MTP MoE
    mtp_norm_w,
    w_lm_p, vbase, ep_rank,  # shared lm_head vocab slice, its first id, this rank's EP position
    atp_group: ReplicaGroup,
    ep_group: ReplicaGroup,
    atp_size: int,
    ep_size: int,
    k: int,
):
    """k greedy MTP draft tokens per sequence: extend pass over the S verified rows, then k-1 chain steps.

    Returns drafts [seqs, k] int32 for this rank's sequences. Writes the MTP KV cache at the extend
    positions pos .. pos+S-1 and the chain positions pos+n-1+j (overwritten by the next extend).
    """
    _PREFETCH.clear()
    Ts, H = h_shard.shape
    seqs = sel_rows.shape[0]
    B_attn = pos.shape[0]
    half = cos_tab.shape[1]
    hbm = nl.private_hbm
    drafts = nl.ndarray((seqs, k), dtype=nl.int32, buffer=nl.shared_hbm)

    e = nl.ndarray((Ts, H), dtype=nl.bfloat16, buffer=hbm, name="e_ext")
    embed_rows_body(embed_table, ext_ids, Ts, e)
    h_all = _mtp_forward(
        e, h_shard, ln_e, ln_h, fc_p, fc_row, in_ln_w, w_qkv, q_norm_w, k_norm_w, w_gate_p, cos, sin, k_cache_t,
        v_cache, pos, w_o, post_ln_w, w_router_p, sel, w_shared_p, w_sd, gate_up_w, gate_up_scale, down_w,
        down_scale, mtp_norm_w, atp_group, ep_group, atp_size, ep_size, "_mx",
    )
    h = _select_rows(h_all, sel_rows, seqs, "_mx")
    tok = _greedy_tokens(h, w_lm_p, vbase, ep_rank, ep_group, ep_size, "_mx")
    nisa.dma_copy(dst=drafts[:, 0:1], src=tok)

    # chain positions: pos + n - 1 + j for the attention group's sequences
    pos_sb = nl.ndarray((P, 1), dtype=nl.int32, buffer=nl.sbuf)[:B_attn]
    nisa.dma_copy(dst=pos_sb, src=pos)
    n_sb = nl.ndarray((P, 1), dtype=nl.int32, buffer=nl.sbuf)[:B_attn]
    nisa.dma_copy(dst=n_sb, src=n_grp)
    base = nl.ndarray((P, 1), dtype=nl.int32, buffer=nl.sbuf)[:B_attn]
    nisa.tensor_tensor(dst=base, data1=pos_sb, data2=n_sb, op=nl.add)
    for j in range(1, k):
        tag = f"_mc{j}"
        pj = nl.ndarray((P, 1), dtype=nl.int32, buffer=nl.sbuf)[:B_attn]
        nisa.tensor_scalar(dst=pj, data=base, op0=nl.add, operand0=j - 1)
        pos_j = nl.ndarray((B_attn, 1), dtype=nl.int32, buffer=hbm, name="pos" + tag)
        nisa.dma_copy(dst=pos_j, src=pj)
        cos_j = nl.ndarray((B_attn, half), dtype=nl.float32, buffer=hbm, name="cos" + tag)
        sin_j = nl.ndarray((B_attn, half), dtype=nl.float32, buffer=hbm, name="sin" + tag)
        rope_rows_body(cos_tab, sin_tab, pj, B_attn, cos_j, sin_j)
        e_j = nl.ndarray((seqs, H), dtype=nl.bfloat16, buffer=hbm, name="e" + tag)
        embed_rows_body(embed_table, tok, seqs, e_j)
        h = _mtp_forward(
            e_j, h, ln_e, ln_h, fc_p, fc_row, in_ln_w, w_qkv, q_norm_w, k_norm_w, w_gate_p, cos_j, sin_j, k_cache_t,
            v_cache, pos_j, w_o, post_ln_w, w_router_p, sel, w_shared_p, w_sd, gate_up_w, gate_up_scale, down_w,
            down_scale, mtp_norm_w, atp_group, ep_group, atp_size, ep_size, tag,
        )
        tok = _greedy_tokens(h, w_lm_p, vbase, ep_rank, ep_group, ep_size, tag)
        nisa.dma_copy(dst=drafts[:, j : j + 1], src=tok)
    return drafts
