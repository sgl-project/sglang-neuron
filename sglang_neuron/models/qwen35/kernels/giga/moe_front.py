"""Qwen3.5 MoE front end for one EP rank: post-attention RMSNorm, router, local affinities and
this rank's slice of the gated shared expert (NKI, LNC=1).

For the T gathered tokens (token-major [T, H], T <= 128):
    xn        = RMSNorm_zc(x) * (1 + w_ln)                       -> bf16 [T, H] (expert input)
    probs     = softmax(bf16(xn @ W_router))                     (fp32, over all E experts)
    top-k     = probs >= 10th largest (max8 / match_replace8 / max8), renormalised to sum 1
    aff_local = aff @ sel                                        -> [T, E_L] (sel: per-rank one-hot)
    shared    = sigmoid(bf16(xn . w_sg)) * down_s(silu(xn Wg_s) * (xn Wu_s))  -> [T, H] partial

The shared expert's intermediate dim is sliced across EP ranks (``Is`` columns here), so
summing ``shared`` over ranks (the MoE ReduceScatter) reconstructs it exactly.

Layouts: w_router_p [128, H/128, E], w_shared_p [128, H/128, 2*Is + 1] (gate | up | sg column),
rows pre-permuted like the GDN projection (w_p[p, h1, c] = W[h1*128 + p, c]); w_sd [Is, H].

Padding rows (``valid``, a per-row 0 / 1 column): aff_local of a row with valid = 0 is all zero, so the
row is routed to NO expert: routed-row compaction (moe_compact) gives it no slot and the blocked expert
path adds 0 for it. Its shared-expert partial is still computed (dense, per block: no extra cost). Real
rows are unchanged bit for bit (x 1.0). Without it, padding rows that share one state row and position
have near-identical hidden states, pick the same top-10 experts, and push the few ranks holding those
experts past the compaction cap (an overflow expert pass per MoE layer, every rank waits at the EP
ReduceScatter).

Router fold (``fold=True``, layer_decode QWEN35_ROUTER_FOLD): x arrives as [T, H + E + 1] whose
trailing columns are the unscaled router logits u = x @ diag(1 + w_ln) [W_router | w_sg], computed
upstream as extra mixer out-projection columns (see layer_decode). Then
    logits = bf16(rms(x) * u[:, :E]),   shared gate = sigmoid(bf16(rms(x) * u[:, E]))
and the router weights / matmul and the shared-expert gate column are skipped.
"""

import nki
import nki.isa as nisa
import nki.language as nl

from nkilib.core.utils.kernel_assert import kernel_assert

P = 128
TOPK = 10


def rmsnorm_zc_tok(x_sb, w_bc, T, H, eps, out_dtype, return_rs=False):
    """Token-major zero-centred RMSNorm on SBUF [T, H]; w_bc is (1 + w) broadcast to [T, H].

    ``return_rs``: also return the per-token scale rsqrt(mean(x^2) + eps) [T, 1] fp32."""
    sq = nl.ndarray((P, H), dtype=nl.float32, buffer=nl.sbuf)[:T]
    ss = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)[:T]
    nisa.activation_reduce(dst=sq, op=nl.square, data=x_sb, reduce_op=nl.add, reduce_res=ss)
    rs = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)[:T]
    nisa.activation(dst=rs, op=nl.rsqrt, data=ss, scale=1.0 / H, bias=eps)
    out = nl.ndarray((P, H), dtype=out_dtype, buffer=nl.sbuf)[:T]
    nisa.scalar_tensor_tensor(dst=out, data=x_sb, op0=nl.multiply, operand0=rs, op1=nl.multiply, operand1=w_bc)
    if return_rs:
        return out, rs
    return out


def moe_front_body(
    x,  # [T, H] bf16 residual stream (gathered), HBM or SBUF; fold: [T, H + E + 1] (+ unscaled logits)
    ln_w,  # [H] fp32 post_attention_layernorm weight (zero-centred: scale = 1 + w)
    w_router_p,  # [128, H/128, E] bf16 (unused with fold)
    sel,  # [E, E_L] fp32 one-hot: sel[e, j] = 1 iff global expert e is this rank's local expert j
    w_shared_p,  # [128, H/128, 2*Is + 1] bf16 (shared gate | shared up | shared_expert_gate)
    w_sd,  # [Is, H] bf16 shared down-projection rows of this rank's slice
    return_xT=False,  # also return the feature-major normed tokens [128, H/128, T] (the experts reuse them)
    fold=False,  # router fold: x carries the unscaled router + shared-gate logits in its last E + 1 columns
    valid=None,  # optional fp32 [T, 1] SBUF 1 / 0 per row: 0 zeroes the row's local affinities (padding rows)
):
    T, W = x.shape
    H = ln_w.shape[0]
    E = sel.shape[0]
    E_L = sel.shape[1]
    Is = (w_shared_p.shape[2] - 1) // 2
    H1 = H // P
    eps = 1e-6
    ET = min(P, E)  # expert-dim tile for the local-column selection
    kernel_assert(T <= P and E % ET == 0 and 16 <= E <= 512, "T <= 128, 16 <= E <= 512, E a multiple of min(128, E)")
    kernel_assert(W == (H + E + 1 if fold else H), f"x must be [T, H{' + E + 1' if fold else ''}], got {W} columns")
    if not fold:
        kernel_assert(w_router_p.shape[2] == E, "w_router_p and sel disagree on E")

    # ---------------- RMSNorm ----------------
    if getattr(x, "buffer", None) == nl.sbuf:
        x_w = x
    else:
        x_w = nl.ndarray((P, W), dtype=nl.bfloat16, buffer=nl.sbuf)[:T]
        nisa.dma_copy(dst=x_w, src=x)
    x_sb = x_w[:, 0:H] if fold else x_w
    w_row = nl.ndarray((P, H), dtype=nl.float32, buffer=nl.sbuf)[:T]
    nisa.dma_copy(dst=w_row, src=ln_w.ap(pattern=[[0, T], [1, H]], offset=0))  # broadcast over tokens
    nisa.tensor_scalar(dst=w_row, data=w_row, op0=nl.add, operand0=1.0)
    xn, rs = rmsnorm_zc_tok(x_sb, w_row, T, H, eps, nl.bfloat16, return_rs=True)

    xT = nl.ndarray((P, H1, T), dtype=nl.bfloat16, buffer=nl.sbuf)
    for h1 in range(H1):
        tp = nl.ndarray((P, T), dtype=nl.bfloat16, buffer=nl.psum)
        nisa.nc_transpose(dst=tp, data=xn[:, h1 * P : (h1 + 1) * P])
        nisa.tensor_copy(dst=xT[:, h1, :], src=tp)

    # ---------------- router: logits, softmax, top-10, renormalise ----------------
    logit_bf = nl.ndarray((P, E), dtype=nl.bfloat16, buffer=nl.sbuf)[:T]
    if fold:
        # logits = rms(x) * (x (1 + w_ln)) @ W_router: only the per-token scale is left to apply
        nisa.tensor_scalar(dst=logit_bf, data=x_w[:, H : H + E], op0=nl.multiply, operand0=rs)
    else:
        wr = nl.ndarray((P, H1, E), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=wr, src=w_router_p)
        ps_l = nl.ndarray((P, E), dtype=nl.float32, buffer=nl.psum)[:T]
        for h1 in range(H1):
            nisa.nc_matmul(ps_l, xT[:, h1, :], wr[:, h1, :])
        nisa.tensor_copy(dst=logit_bf, src=ps_l)  # HF router runs F.linear in the model dtype
    mx = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)[:T]
    nisa.tensor_reduce(dst=mx, op=nl.max, data=logit_bf, axis=1)
    nisa.tensor_scalar(dst=mx, data=mx, op0=nl.multiply, operand0=-1.0)
    ex = nl.ndarray((P, E), dtype=nl.float32, buffer=nl.sbuf)[:T]
    ssum = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)[:T]
    nisa.activation_reduce(dst=ex, op=nl.exp, data=logit_bf, reduce_op=nl.add, reduce_res=ssum, bias=mx)
    rsum = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)[:T]
    nisa.reciprocal(dst=rsum, data=ssum)
    probs = nl.ndarray((P, E), dtype=nl.float32, buffer=nl.sbuf)[:T]
    nisa.tensor_scalar(dst=probs, data=ex, op0=nl.multiply, operand0=rsum)

    # Select exactly TOPK: bf16 logits tie, and a ">= 10th value" threshold would then keep 11+.
    # Selection runs on probs nudged by expert id (x (1 - e*2^-21): lower id wins a tie, like
    # moe_lowc and torch.topk in practice); the affinities below use the exact probabilities.
    tb_i = nl.ndarray((P, E), dtype=nl.int32, buffer=nl.sbuf)[:T]
    nisa.iota(tb_i, pattern=[[1, E]], offset=0, channel_multiplier=0)
    pt = nl.ndarray((P, E), dtype=nl.float32, buffer=nl.sbuf)[:T]
    nisa.tensor_scalar(dst=pt, data=tb_i, op0=nl.multiply, operand0=-(2.0**-21), op1=nl.add, operand1=1.0)
    nisa.tensor_tensor(dst=pt, data1=probs, data2=pt, op=nl.multiply)
    top8 = nl.ndarray((P, 8), dtype=nl.float32, buffer=nl.sbuf)[:T]
    nisa.max8(dst=top8, src=pt)
    rest = nl.ndarray((P, E), dtype=nl.float32, buffer=nl.sbuf)[:T]
    nisa.nc_match_replace8(dst=rest, data=pt, vals=top8, imm=-1.0)
    next8 = nl.ndarray((P, 8), dtype=nl.float32, buffer=nl.sbuf)[:T]
    nisa.max8(dst=next8, src=rest)
    thr = next8[:, TOPK - 9 : TOPK - 8]  # 10th largest (nudged) probability
    keep = nl.ndarray((P, E), dtype=nl.float32, buffer=nl.sbuf)[:T]
    nisa.tensor_scalar(dst=keep, data=pt, op0=nl.greater_equal, operand0=thr)
    sel_p = nl.ndarray((P, E), dtype=nl.float32, buffer=nl.sbuf)[:T]
    ksum = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)[:T]
    nisa.tensor_tensor(dst=sel_p, data1=probs, data2=keep, op=nl.multiply)
    nisa.tensor_reduce(dst=ksum, op=nl.add, data=sel_p, axis=1)
    nisa.reciprocal(dst=ksum, data=ksum)
    aff = nl.ndarray((P, E), dtype=nl.float32, buffer=nl.sbuf)[:T]
    nisa.tensor_scalar(dst=aff, data=sel_p, op0=nl.multiply, operand0=ksum)

    # local columns: aff [T, E] @ sel [E, E_L] on the PE (exact: sel is one-hot)
    sel_sb = nl.ndarray((P, E // ET, E_L), dtype=nl.float32, buffer=nl.sbuf)[:ET]
    nisa.dma_copy(dst=sel_sb, src=sel.ap(pattern=[[E_L, ET], [ET * E_L, E // ET], [1, E_L]], offset=0))
    ps_a = nl.ndarray((P, E_L), dtype=nl.float32, buffer=nl.psum)[:T]
    for et in range(E // ET):
        tpa = nl.ndarray((P, T), dtype=nl.float32, buffer=nl.psum)[:ET]
        nisa.nc_transpose(dst=tpa, data=aff[:, et * ET : (et + 1) * ET])
        affT = nl.ndarray((P, T), dtype=nl.float32, buffer=nl.sbuf)[:ET]
        nisa.tensor_copy(dst=affT, src=tpa)
        nisa.nc_matmul(ps_a, affT, sel_sb[:, et, :])
    aff_l = nl.ndarray((P, E_L), dtype=nl.float32, buffer=nl.sbuf)[:T]
    if valid is None:
        nisa.tensor_copy(dst=aff_l, src=ps_a)
    else:  # invalid (padding) rows route nowhere: all-zero affinities, so moe_compact gives them no slot
        nisa.tensor_scalar(dst=aff_l, data=ps_a, op0=nl.multiply, operand0=valid)

    # ---------------- shared expert slice ----------------
    NS = 2 * Is if fold else 2 * Is + 1  # fold: the shared-gate logit arrived with x
    ws = nl.ndarray((P, H1, NS), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=ws, src=w_shared_p[:, :, 0:NS] if fold else w_shared_p)
    ps_s = nl.ndarray((P, NS), dtype=nl.float32, buffer=nl.psum)[:T]
    for h1 in range(H1):
        nisa.nc_matmul(ps_s, xT[:, h1, :], ws[:, h1, :])
    s_bf = nl.ndarray((P, NS), dtype=nl.bfloat16, buffer=nl.sbuf)[:T]
    nisa.tensor_copy(dst=s_bf, src=ps_s)
    g_act = nl.ndarray((P, Is), dtype=nl.float32, buffer=nl.sbuf)[:T]
    nisa.activation(dst=g_act, op=nl.silu, data=s_bf[:, 0:Is])
    h_s = nl.ndarray((P, Is), dtype=nl.bfloat16, buffer=nl.sbuf)[:T]
    nisa.tensor_tensor(dst=h_s, data1=g_act, data2=s_bf[:, Is : 2 * Is], op=nl.multiply)
    sg = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)[:T]
    if fold:
        sgl = nl.ndarray((P, 1), dtype=nl.bfloat16, buffer=nl.sbuf)[:T]
        nisa.tensor_scalar(dst=sgl, data=x_w[:, H + E : H + E + 1], op0=nl.multiply, operand0=rs)
        nisa.activation(dst=sg, op=nl.sigmoid, data=sgl)
    else:
        nisa.activation(dst=sg, op=nl.sigmoid, data=s_bf[:, 2 * Is : 2 * Is + 1])
    tph = nl.ndarray((P, T), dtype=nl.bfloat16, buffer=nl.psum)[:Is]
    nisa.nc_transpose(dst=tph, data=h_s)
    hT = nl.ndarray((P, T), dtype=nl.bfloat16, buffer=nl.sbuf)[:Is]
    nisa.tensor_copy(dst=hT, src=tph)
    wsd = nl.ndarray((P, H), dtype=nl.bfloat16, buffer=nl.sbuf)[:Is]
    nisa.dma_copy(dst=wsd, src=w_sd)
    so = nl.ndarray((P, H), dtype=nl.bfloat16, buffer=nl.sbuf)[:T]
    for c0 in range(0, H, 512):
        ps_d = nl.ndarray((P, 512), dtype=nl.float32, buffer=nl.psum)[:T]
        nisa.nc_matmul(ps_d, hT, wsd[:, c0 : c0 + 512])
        nisa.tensor_scalar(dst=so[:, c0 : c0 + 512], data=ps_d, op0=nl.multiply, operand0=sg)
    if return_xT:
        return xn, aff_l, so, xT
    return xn, aff_l, so


@nki.jit
def moe_front(x, ln_w, w_router_p, sel, w_shared_p, w_sd):
    """Standalone kernel around moe_front_body (HBM outputs, used by tests)."""
    T, H = x.shape
    xn, aff_l, so = moe_front_body(x, ln_w, w_router_p, sel, w_shared_p, w_sd)
    xn_out = nl.ndarray((T, H), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    aff_out = nl.ndarray((T, sel.shape[1]), dtype=nl.float32, buffer=nl.shared_hbm)
    shared_out = nl.ndarray((T, H), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=xn_out, src=xn)
    nisa.dma_copy(dst=aff_out, src=aff_l)
    nisa.dma_copy(dst=shared_out, src=so)
    return xn_out, aff_out, shared_out


@nki.jit
def moe_front_fold(xl, ln_w, sel, w_shared_p, w_sd):
    """Router-fold variant (tests): xl = [x | unscaled router logits | unscaled shared-gate logit]."""
    T = xl.shape[0]
    H = ln_w.shape[0]
    xn, aff_l, so = moe_front_body(xl, ln_w, None, sel, w_shared_p, w_sd, False, True)
    xn_out = nl.ndarray((T, H), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    aff_out = nl.ndarray((T, sel.shape[1]), dtype=nl.float32, buffer=nl.shared_hbm)
    shared_out = nl.ndarray((T, H), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=xn_out, src=xn)
    nisa.dma_copy(dst=aff_out, src=aff_l)
    nisa.dma_copy(dst=shared_out, src=so)
    return xn_out, aff_out, shared_out


@nki.jit
def moe_front_valid(x, ln_w, w_router_p, sel, w_shared_p, w_sd, valid):
    """moe_front with a per-row valid column (tests): valid int32 [T, 1], 0 = padding row (aff_local = 0)."""
    T, H = x.shape
    vi = nl.ndarray((P, 1), dtype=nl.int32, buffer=nl.sbuf)[:T]
    nisa.dma_copy(dst=vi, src=valid)
    vf = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)[:T]
    nisa.tensor_scalar(dst=vf, data=vi, op0=nl.greater, operand0=0)
    xn, aff_l, so = moe_front_body(x, ln_w, w_router_p, sel, w_shared_p, w_sd, False, False, vf)
    xn_out = nl.ndarray((T, H), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    aff_out = nl.ndarray((T, sel.shape[1]), dtype=nl.float32, buffer=nl.shared_hbm)
    shared_out = nl.ndarray((T, H), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=xn_out, src=xn)
    nisa.dma_copy(dst=aff_out, src=aff_l)
    nisa.dma_copy(dst=shared_out, src=so)
    return xn_out, aff_out, shared_out
